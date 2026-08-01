"""Jiterator-compiled LIF kernels (spec 02a §6).

Why this module exists at all
-----------------------------
The Phase-1 cost model is `time ~= 14.5 us x kernels-per-timestep x L`
(01_reconnaissance.md §3.4): on this box, in eager mode, wall-clock is
proportional to the *number* of CUDA launches and almost independent of how much
work each one does. A textbook LIF timestep issues ~13 elementwise kernels. The
whole point of this module is to make it issue one.

`torch.compile`/Inductor is dead here (no Triton), and there is no nvcc and no
MSVC, so an ahead-of-time C++ extension is impossible. What *is* available is
NVRTC, which ships inside the torch wheel (`torch/lib/nvrtc64_130_0.dll`), and
`torch.cuda.jiterator` compiles elementwise CUDA C++ against it at runtime
(§3.11 C1, re-verified in `scripts/audit/05_verify_literature_claims.py`). The
LIF update is purely elementwise, so it is precisely the one custom kernel this
project needs and precisely the one it can have.

The costs of that choice, all real:
  * jiterator is a private API (`torch.cuda.jiterator._create_*`), hence risk R11
    and the hard torch pin in requirements.txt. It is isolated behind this module
    so a break is a one-file repair;
  * its outputs carry no autograd history, so the backward is hand-written --
    risk R10, the most dangerous bug class in the project, because a wrong
    backward still trains and still produces plausible bpc. `neuron.py` wraps
    these kernels in an `autograd.Function` and `tests/test_neuron_equivalence.py`
    is the gate;
  * it is elementwise-only: no reductions, no gather/scatter, no matmul (§2.3).

One kernel per reset mode
-------------------------
`reset` is a compile-time constant, substituted into the source text, not a
runtime branch. Four modes therefore mean up to four compiled forward kernels and
four backward kernels, cached for the process lifetime; measured build cost here
is 0.06-0.14 s each, in line with the 0.17 s of §3.11 C1. ("detached" costs zero:
its forward source is byte-identical to "hard"'s and jiterator caches on the
source text.) A runtime branch would cost nothing in launches but would make the
generated PTX carry all four reset rules, and -- worse -- would let a caller
change the reset rule mid-scan, which is meaningless and would be silent.

Numerical contract with the eager reference
-------------------------------------------
`v_pre = v_prev * beta + cur` is written with `__fmul_rn` / `__fadd_rn` rather
than the natural `v_prev * beta + cur`. NVRTC contracts the natural form into an
FMA (measured: it does, on this stack), which rounds once instead of twice. The
eager reference cannot contract, because its multiply and add are two separate
CUDA kernels. The resulting one-ulp difference is invisible in the 1e-5 membrane
tolerance -- but it can flip a spike whose `v_pre` lands within an ulp of the
threshold, and the R10 gate requires spikes to be *bit-identical*, not close.
Suppressing the contraction makes `v_pre` bit-identical for every beta; with the
contraction left in, the drift was measured at up to 2.9e-06 at beta = 0.95.
The intrinsics are float-only, which is safe because §6 mandates fp32 for every
kernel input and output regardless of `cfg.dtype`.
"""

from __future__ import annotations

import functools
from typing import Callable

import torch

from snn.surrogate import ATAN_CUDA_GRAD

__all__ = [
    "RESET_MODES",
    "RESET_CODES",
    "jiterator_available",
    "lif_forward_kernel",
    "lif_backward_kernel",
    "clear_kernel_cache",
]

# Order is normative: `FusedLIFScan` takes an int `reset_code` so that its
# non-tensor arguments stay hashable and CUDA-graph-capture safe (spec §7).
RESET_MODES: tuple[str, ...] = ("hard", "soft", "detached", "none")
RESET_CODES: dict[str, int] = {name: i for i, name in enumerate(RESET_MODES)}


def _check_reset(reset: str) -> str:
    if reset not in RESET_CODES:
        raise ValueError(
            f"unknown reset mode {reset!r}; expected one of {RESET_MODES}"
        )
    return reset


# --------------------------------------------------------------------------
# CUDA source
# --------------------------------------------------------------------------

# v_pre computed with explicitly-rounded multiply and add; see module docstring.
_VPRE = (
    "    const float vp = __fadd_rn(__fmul_rn(static_cast<float>(v_prev),\n"
    "                                         static_cast<float>(beta)),\n"
    "                               static_cast<float>(cur));\n"
    "    const T v = static_cast<T>(vp);\n"
)

# forward: v_next as a function of (v_pre, spike). "detached" is identical to
# "hard" in the forward -- the two differ only in what the backward treats as
# constant, which is the entire point of having both.
_FORWARD_RESET: dict[str, str] = {
    "hard": "v * (T(1) - s)",
    "soft": "v - thr * s",
    "detached": "v * (T(1) - s)",
    "none": "v",
}

_FORWARD_SRC = """
template <typename T>
void lif_forward(T v_prev, T cur, T beta, T thr,
                 T& v_next, T& spike, T& v_pre) {{
{vpre}    const T s = (v >= thr) ? T(1) : T(0);
    v_pre  = v;
    spike  = s;
    v_next = {reset_expr};
}}
"""

# backward: d v_next / d v_pre, derived from the forward above with
# s' = atan_grad(v_pre - thr, alpha) and cross-checked against spec §6.2.
#
#   soft      v_next = v_pre - thr*s          ->  1 - thr*sg
#   hard      v_next = v_pre*(1 - s)          ->  (1 - s) - v_pre*sg
#   detached  v_next = v_pre*(1 - s.detach()) ->  (1 - s)
#   none      v_next = v_pre                  ->  1
_BACKWARD_DVNEXT: dict[str, str] = {
    "hard": "(T(1) - s) - v_pre * sg",
    "soft": "T(1) - thr * sg",
    "detached": "T(1) - s",
    "none": "T(1)",
}

_BACKWARD_SRC = """
template <typename T>
void lif_backward(T grad_spike, T grad_v_next, T v_pre,
                  T beta, T thr, T alpha,
                  T& grad_cur, T& grad_v_prev) {{
    const T x  = v_pre - thr;
    const T s  = (x >= T(0)) ? T(1) : T(0);
    const T sg = {atan_grad};
    const T dv = {dvnext_dvpre};
    const T g  = grad_v_next * dv + grad_spike * sg;
    grad_cur    = g;
    grad_v_prev = beta * g;
}}
"""


def forward_source(reset: str) -> str:
    """CUDA C++ text for the forward kernel. Exposed for inspection/tests."""
    _check_reset(reset)
    return _FORWARD_SRC.format(vpre=_VPRE, reset_expr=_FORWARD_RESET[reset])


def backward_source(reset: str) -> str:
    """CUDA C++ text for the backward kernel. Exposed for inspection/tests."""
    _check_reset(reset)
    return _BACKWARD_SRC.format(
        atan_grad=ATAN_CUDA_GRAD, dvnext_dvpre=_BACKWARD_DVNEXT[reset]
    )


# --------------------------------------------------------------------------
# availability probe
# --------------------------------------------------------------------------

_TRIVIAL_SRC = """
template <typename T>
void probe(T a, T b, T& o0, T& o1) { o0 = a + b; o1 = a - b; }
"""


@functools.lru_cache(maxsize=1)
def jiterator_available() -> bool:
    """True iff a jiterator kernel can actually be compiled and run here.

    Deliberately does the whole round trip -- import, compile, launch, read back
    -- rather than checking for the attribute's existence. The failure mode this
    guards against is a torch upgrade that keeps the private symbol but breaks
    NVRTC discovery, which would otherwise surface as a crash in the middle of a
    training run rather than as a clean fall back to the eager path.

    Cached: the probe costs one NVRTC compile (~0.1-0.25 s).
    """
    try:
        if not torch.cuda.is_available():
            return False
        from torch.cuda.jiterator import _create_multi_output_jit_fn

        fn = _create_multi_output_jit_fn(_TRIVIAL_SRC, num_outputs=2)
        a = torch.ones(4, device="cuda")
        b = torch.full((4,), 2.0, device="cuda")
        o0, o1 = fn(a, b)
        torch.cuda.synchronize()
        return bool(o0.eq(3.0).all()) and bool(o1.eq(-1.0).all())
    except Exception:
        return False


# --------------------------------------------------------------------------
# compiled-kernel cache
# --------------------------------------------------------------------------

_FORWARD_CACHE: dict[str, Callable] = {}
_BACKWARD_CACHE: dict[str, Callable] = {}


def lif_forward_kernel(reset: str) -> Callable:
    """Compiled forward kernel for one reset mode. Cached; lazy on first use.

    Returned callable: `fn(v_prev, cur, beta=..., thr=...)` over fp32 tensors of
    any broadcastable shape, giving `(v_next, spike, v_pre)`.

    `v_pre` is a third output rather than being recomputed in the backward
    because under hard reset it is *not* recoverable from `v_next`: `v_next` is
    identically zero wherever the neuron fired. Emitting it costs one store;
    recomputing it is impossible.
    """
    _check_reset(reset)
    fn = _FORWARD_CACHE.get(reset)
    if fn is None:
        from torch.cuda.jiterator import _create_multi_output_jit_fn

        fn = _create_multi_output_jit_fn(
            forward_source(reset), num_outputs=3, beta=0.5, thr=1.0
        )
        _FORWARD_CACHE[reset] = fn
    return fn


def lif_backward_kernel(reset: str) -> Callable:
    """Compiled backward kernel for one reset mode. Cached; lazy on first use.

    Returned callable:
    `fn(grad_spike, grad_v_next, v_pre, beta=..., thr=..., alpha=...)` giving
    `(grad_cur, grad_v_prev)`.

    `beta` is a fixed hyperparameter in Phase 2, so no `grad_beta` is emitted.
    Learnable per-channel decay is a Phase-3 candidate admitted by the §4.6
    ruling; it needs a third output and jiterator's limit is eight.
    """
    _check_reset(reset)
    fn = _BACKWARD_CACHE.get(reset)
    if fn is None:
        from torch.cuda.jiterator import _create_multi_output_jit_fn

        fn = _create_multi_output_jit_fn(
            backward_source(reset), num_outputs=2, beta=0.5, thr=1.0, alpha=2.0
        )
        _BACKWARD_CACHE[reset] = fn
    return fn


def clear_kernel_cache() -> None:
    """Drop the compiled kernels. Only useful in tests that edit the source."""
    _FORWARD_CACHE.clear()
    _BACKWARD_CACHE.clear()
