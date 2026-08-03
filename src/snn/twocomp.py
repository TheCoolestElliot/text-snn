"""Two-compartment LIF with a reset-shielded slow pole (EXP_004, candidate #1).

Pre-registered in `experiments/logs/EXP_004_two_compartment.md`. Read that first:
the hypotheses, the initialisation, and the decision rule were all fixed before
this file existed, and the reasons for each design choice live there rather than
here.

The neuron, per channel `c`, per timestep `t`:

    vf_t  =  beta_f   * vf_{t-1} + cur_t        fast pole; beta_f fixed at 0.5
    vs_t  =  beta_s,c * vs_{t-1} + cur_t        slow pole; beta_s,c learned
    v_t   =  vf_t + w_c * vs_t                  the mix that reaches the threshold
    s_t   =  1[ v_t >= thr ]                    I3, hard threshold
    vf_t <-  vf_t * (1 - s_t)                   hard reset, the baseline's rule
    vs_t <-  vs_t                               RESET-SHIELDED

Admitted by the §4.6 ruling, row 2 (multiple learned timescales per neuron, O(1)
parameters per neuron, no lateral mixing). Two parameters per channel per layer:
the mix `w_c` and the slow decay `beta_s,c`.

Why the mix is additive rather than convex
------------------------------------------
`EXP_002` priced the convex form `v = w*vf + (1-w)*vs`, which nests the Phase-2
baseline at `w = 1` -- a *boundary* of a constrained parameter. The additive form
nests it at `w = 0`, in the interior: the readout is then exactly `vf`, `vf` is a
plain LIF at beta = 0.5 with hard reset, and `vs` influences nothing. That is the
committed baseline exactly, not approximately, and
`tests/test_twocomp_equivalence.py::test_w_zero_is_the_committed_lif_scan` asserts
it bit-for-bit against `snn.neuron.lif_scan` in both the forward and the backward.
The kernel's shape, arity and cost are unchanged from `EXP_002`'s N2, so its
pricing carries over untouched.

Why this lives in its own module rather than in kernels.py / neuron.py
----------------------------------------------------------------------
Two reasons, both mechanical.

  * `scripts/audit/08_mutation_campaign.py` requires every mutation's anchor text
    to appear EXACTLY ONCE in its target file, and aborts otherwise. A second
    neuron in `kernels.py` would duplicate lines like
    `const T s = (v >= thr) ? T(1) : T(0);` and turn the campaign's own safety
    check into a failure. Separate files keep both neurons' anchors unique, and
    keep the two-compartment mutations cleanly separable from the baseline's 25.
  * The Phase-2 kernel and its gate are mutation-tested and committed. Nothing
    here edits them, so a Phase-4 candidate cannot regress the arm it is measured
    against (EXP_004 failure mode J9).

The cost of the split, stated because it is real: `snn/kernels.py`'s docstring
claims the private jiterator API is isolated behind one module so a torch upgrade
that breaks it is a one-file repair (risk R11). With this file importing
`_create_multi_output_jit_fn` too, that repair is now two files. Accepted.

Numerical contract
------------------
Every arithmetic step that the eager reference performs as a separate CUDA kernel
is written here with `__fmul_rn` / `__fadd_rn` so NVRTC cannot contract it into an
FMA. That includes BOTH membrane recursions *and the mix* `vf + w*vs`, which is
the new one: the eager path computes it as a multiply and an add and rounds twice,
and an FMA there rounds once. The R10 gate requires spikes to be bit-identical
rather than close, and a one-ulp drift in `v` flips any spike whose membrane lands
within an ulp of the threshold. The intrinsics are float-only, which is safe
because fp32 is enforced on every entry point below.

R10 applies here in full: the backward is hand-written, jiterator outputs carry no
autograd history, and a wrong backward still trains and still produces a plausible
bpc. `tests/test_twocomp_equivalence.py` is the gate.
"""

from __future__ import annotations

import functools
from typing import Callable

import torch
from torch import Tensor

from snn.surrogate import ATAN_CUDA_GRAD, atan_spike

__all__ = [
    "twocomp_scan_eager",
    "FusedTwoCompartmentScan",
    "twocomp_scan",
    "twocomp_forward_kernel",
    "twocomp_backward_kernel",
    "twocomp_forward_source",
    "twocomp_backward_source",
    "clear_twocomp_kernel_cache",
]


# --------------------------------------------------------------------------
# CUDA source
# --------------------------------------------------------------------------

# ARGUMENT ORDER IS LOAD-BEARING. jiterator binds every TENSOR argument first, in
# declaration order, and only then the scalars as keywords. `EXP_002` §8.4 records
# what happens when that is broken: a draft of the two-timescale kernel interleaved
# a scalar before a per-channel tensor, which compiled, ran, produced sensible
# spikes, and priced within 2% of the correct kernel while silently binding the
# per-channel tensor into a scalar slot. So: five tensors, then two scalars.
#
# `vmix` rather than `v` for the mixed membrane, and `xg`/`sh` in the backward, are
# not stylistic. They keep this file's source text distinct from `kernels.py`'s, so
# the mutation campaign's per-file "anchor appears exactly once" guard stays a
# guard rather than becoming a false alarm.

_TC_MEMBRANE = (
    "    const float vfp = __fadd_rn(__fmul_rn(static_cast<float>(vf_prev),\n"
    "                                          static_cast<float>(beta_f)),\n"
    "                                static_cast<float>(cur));\n"
    "    const float vsp = __fadd_rn(__fmul_rn(static_cast<float>(vs_prev),\n"
    "                                          static_cast<float>(beta_s_c)),\n"
    "                                static_cast<float>(cur));\n"
    "    const float vmp = __fadd_rn(vfp, __fmul_rn(static_cast<float>(w_c), vsp));\n"
    "    const T vf   = static_cast<T>(vfp);\n"
    "    const T vs   = static_cast<T>(vsp);\n"
    "    const T vmix = static_cast<T>(vmp);\n"
)

_TC_FORWARD_SRC = """
template <typename T>
void tc_forward(T vf_prev, T vs_prev, T cur, T w_c, T beta_s_c,
                T beta_f, T thr,
                T& vf_next, T& vs_next, T& spike, T& v_pre) {{
{membrane}    const T sp = (vmix >= thr) ? T(1) : T(0);
    v_pre   = vmix;
    spike   = sp;
    vf_next = vf * (T(1) - sp);
    vs_next = vs;
}}
"""

# Derived from the forward above, term by term. With vf, vs, v = vf + w*vs,
# s = H(v - thr), sg = d s / d v, and vf_out = vf*(1 - s):
#
#   dL/dv       = sg * (grad_spike - grad_vf_next * vf)     ... vf_out depends on v
#                                                                only through s
#   dL/dvf      = grad_vf_next * (1 - s) + dL/dv
#   dL/dvs      = grad_vs_next + w * dL/dv                  ... the slow pole is
#                                                                shielded, so its
#                                                                only extra path
#                                                                is through the mix
#   dL/dcur     = dL/dvf + dL/dvs                           ... cur enters both
#   dL/dvf_prev = beta_f   * dL/dvf
#   dL/dvs_prev = beta_s_c * dL/dvs
#
# `dL/dvf` is computed in the REGROUPED form `grad_vf_next*dv + grad_spike*sg` with
# `dv = (1 - s) - vf*sg`, which is the same expression algebraically and is written
# this way on purpose: it is character-for-character the grouping `snn/kernels.py`
# uses for its `g`, so at `w = 0` this kernel's backward is **bit-identical** to the
# Phase-2 LIF's rather than merely equal to it within fp32 rounding. Written the
# other way round it agreed only to ~2 ulp, and
# `test_w_zero_is_the_committed_lif_scan` -- the leg that inherits the committed
# 25/25 mutation campaign's evidence -- is worth much more asserted at `== 0.0`
# than at a tolerance. `dL/dv` is still needed separately for `dL/dw` and for the
# slow adjoint, so the two expressions are different quantities, not a redundant
# computation of one.
#
# `vf` is not saved: it is recovered as `v_pre - w_c*vs`, which costs one multiply
# and one subtract inside a kernel that is already memory-bound, against a whole
# [B, L, d] tensor per layer if it were stored. At `w_c = 0` the recovery is exact
# (`v_pre - 0.0*vs` is `v_pre` bitwise for either sign of `vs`), which is what lets
# the nesting be bit-exact rather than nearly so.
#
# The two per-channel gradients are REDUCTIONS over batch and time, and jiterator
# cannot reduce (§2.3). `EXP_002` §8.3 measured the two ways to pay for that:
# accumulating in the loop costs +1 kernel per timestep per parameter, while
# emitting the per-element adjoint and reducing ONCE at the end costs +0.146 GiB
# and keeps the one-kernel floor. This kernel emits `dL/dv` and `dL/dvs` and the
# caller does both reductions outside the loop.
_TC_BACKWARD_SRC = """
template <typename T>
void tc_backward(T grad_spike, T grad_vf_next, T grad_vs_next, T v_pre, T vs,
                 T w_c, T beta_s_c, T beta_f, T thr, T alpha,
                 T& grad_cur, T& grad_vf_prev, T& grad_vs_prev,
                 T& gv_elem, T& gvs_elem) {{
    const T x   = v_pre - thr;
    const T sh  = (x >= T(0)) ? T(1) : T(0);
    const T sgd = {atan_grad};
    const T vf  = v_pre - w_c * vs;
    const T dv  = (T(1) - sh) - vf * sgd;
    const T gf  = grad_vf_next * dv + grad_spike * sgd;
    const T gv  = sgd * (grad_spike - grad_vf_next * vf);
    const T gs  = grad_vs_next + w_c * gv;
    grad_cur     = gf + gs;
    grad_vf_prev = beta_f * gf;
    grad_vs_prev = beta_s_c * gs;
    gv_elem      = gv;
    gvs_elem     = gs;
}}
"""


def twocomp_forward_source() -> str:
    """CUDA C++ text for the forward kernel. Exposed for inspection/tests."""
    return _TC_FORWARD_SRC.format(membrane=_TC_MEMBRANE)


def twocomp_backward_source() -> str:
    """CUDA C++ text for the backward kernel. Exposed for inspection/tests."""
    return _TC_BACKWARD_SRC.format(atan_grad=ATAN_CUDA_GRAD)


# --------------------------------------------------------------------------
# compiled-kernel cache
# --------------------------------------------------------------------------

@functools.lru_cache(maxsize=1)
def twocomp_forward_kernel() -> Callable:
    """`fn(vf_prev, vs_prev, cur, w_c, beta_s_c, beta_f=, thr=)`.

    Returns `(vf_next, vs_next, spike, v_pre)`. There is no reset-mode argument:
    this neuron's reset rule is fixed by EXP_004 §2 (hard on the fast pole,
    shielded on the slow one). The reset ablation is a separate candidate (#7) and
    would be a separate kernel, not a runtime branch here.
    """
    from torch.cuda.jiterator import _create_multi_output_jit_fn

    return _create_multi_output_jit_fn(
        twocomp_forward_source(), num_outputs=4, beta_f=0.5, thr=1.0
    )


@functools.lru_cache(maxsize=1)
def twocomp_backward_kernel() -> Callable:
    """`fn(grad_spike, grad_vf_next, grad_vs_next, v_pre, vs, w_c, beta_s_c,
    beta_f=, thr=, alpha=)`.

    Returns `(grad_cur, grad_vf_prev, grad_vs_prev, gv_elem, gvs_elem)`. Seven
    tensor inputs and five outputs, against jiterator's 8/8 limit -- the widest
    kernel in the project, and still not binding.
    """
    from torch.cuda.jiterator import _create_multi_output_jit_fn

    return _create_multi_output_jit_fn(
        twocomp_backward_source(), num_outputs=5, beta_f=0.5, thr=1.0, alpha=2.0
    )


def clear_twocomp_kernel_cache() -> None:
    """Drop the compiled kernels. Only useful in tests that edit the source."""
    twocomp_forward_kernel.cache_clear()
    twocomp_backward_kernel.cache_clear()


# --------------------------------------------------------------------------
# shape / dtype contract
# --------------------------------------------------------------------------

def _check(cur: Tensor, v0: Tensor, w: Tensor, beta_s: Tensor) -> int:
    """Validate and return `d`. `v0` is [B, 2d]: the fast half then the slow half.

    One tensor rather than two so that the per-layer state stays a single object
    and `evaluate.py`'s carried protocol, `train.py`, and `EXP_001`'s horizon probe
    keep working unchanged for every arm. The concatenation costs one kernel per
    layer per forward pass, against K*L in the time loop.
    """
    if cur.dim() != 3:
        raise ValueError(f"cur must be [B, L, d]; got shape {tuple(cur.shape)}")
    if v0.dim() != 2:
        raise ValueError(f"v0 must be [B, 2d]; got shape {tuple(v0.shape)}")
    d = cur.shape[2]
    if v0.shape[0] != cur.shape[0] or v0.shape[1] != 2 * d:
        raise ValueError(
            f"v0 shape {tuple(v0.shape)} does not match cur {tuple(cur.shape)}; "
            f"expected ({cur.shape[0]}, {2 * d})"
        )
    if cur.shape[1] < 1:
        raise ValueError("cur must have at least one timestep")
    for name, p in (("w", w), ("beta_s", beta_s)):
        if p.dim() != 2 or p.shape[0] != 1 or p.shape[1] != d:
            raise ValueError(
                f"{name} must be [1, d] = (1, {d}); got {tuple(p.shape)}"
            )
    return d


def _check_fp32(*tensors: Tensor) -> None:
    """Membrane state and the threshold comparison are fp32 unconditionally.

    Same rule and same reason as `snn.neuron._check_fp32` (spec §2, bottleneck
    B8): bf16's spacing at v = 1.0 is 0.78% of the firing threshold. `cfg.dtype`
    selects the GEMM dtype only. Enforced on both branches so the `--fused` and
    `--no-fused` arms stay comparable.
    """
    bad = [str(t.dtype) for t in tensors if t.dtype is not torch.float32]
    if bad:
        raise TypeError(
            "the two-compartment scan requires fp32 current, membrane and "
            f"per-channel parameters; got {bad}. Cast with .float() before the "
            "scan -- cfg.dtype selects the GEMM dtype only (spec §2)"
        )


# --------------------------------------------------------------------------
# eager reference -- the ground truth for R10
# --------------------------------------------------------------------------

def twocomp_scan_eager(cur: Tensor, v0: Tensor, w: Tensor, beta_s: Tensor,
                       beta_f: float, thr: float,
                       alpha: float) -> tuple[Tensor, Tensor]:
    """Reference two-compartment scan built from ordinary autograd ops.

    `cur` [B, L, d], `v0` [B, 2d], `w` and `beta_s` [1, d]; returns
    (spikes [B, L, d], v_final [B, 2d]). Runs on CPU or CUDA and is the
    `--no-fused` path.

    Written for legibility, not for speed: this is what the hand-written backward
    is checked against, so it has to be obviously the equations in the module
    docstring. Unlike `twocomp_scan` it does not insist on fp32 -- the gate runs it
    in float64 to get a reference better than the thing it is checking, and
    `torch.autograd.gradcheck` needs float64 to mean anything.

    The order of operations matters and is not free to change. `vf * beta_f +
    cur[:, t]` is a multiply and an add, rounding twice; so is `vf + w * vs`. The
    fused kernel suppresses FMA contraction at exactly those two places so that
    the two paths agree bit-for-bit rather than to a tolerance.
    """
    d = _check(cur, v0, w, beta_s)
    length = cur.shape[1]
    vf = v0[:, :d]
    vs = v0[:, d:]
    spikes: list[Tensor] = []
    for t in range(length):
        vf = vf * beta_f + cur[:, t]
        vs = vs * beta_s + cur[:, t]
        s = atan_spike(vf + w * vs - thr, alpha)
        spikes.append(s)
        vf = vf * (1.0 - s)
        # vs is NOT reset -- that is the whole point of the candidate.
    return torch.stack(spikes, dim=1), torch.cat([vf, vs], dim=1)


# --------------------------------------------------------------------------
# fused path
# --------------------------------------------------------------------------

class FusedTwoCompartmentScan(torch.autograd.Function):
    """One jiterator kernel per timestep forward, one per timestep backward.

    Non-tensor arguments are three floats, so nothing unhashable or
    device-dependent crosses the autograd boundary and the scan stays
    CUDA-graph-capture safe -- the same contract `FusedLIFScan` keeps.
    """

    @staticmethod
    def forward(ctx, cur: Tensor, v0: Tensor, w: Tensor, beta_s: Tensor,
                beta_f: float, thr: float, alpha: float) -> tuple[Tensor, Tensor]:
        # Repeated rather than trusted to `twocomp_scan`, because `.apply` is a
        # public entry point that skips it, and because the kernel's membrane
        # recursion is written with float-only intrinsics: handed float64 it would
        # narrow to fp32 and cast back, returning a double that is secretly a
        # float. Host-side check only -- no sync, so it is capture-safe.
        _check_fp32(cur, v0, w, beta_s)
        d = _check(cur, v0, w, beta_s)
        fwd = twocomp_forward_kernel()
        length = cur.shape[1]

        spikes: list[Tensor] = []
        v_pres: list[Tensor] = []
        vs_all: list[Tensor] = []
        vf = v0[:, :d]
        vs = v0[:, d:]
        vs0 = vs
        # jiterator outputs carry no autograd history anyway; no_grad is stated
        # explicitly so this loop cannot build a graph even if a future caller
        # invokes forward() outside autograd's own grad-disabled region.
        with torch.no_grad():
            for t in range(length):
                vf, vs, s, vp = fwd(vf, vs, cur[:, t], w, beta_s,
                                    beta_f=beta_f, thr=thr)
                spikes.append(s)
                v_pres.append(vp)
                vs_all.append(vs)

        out_spikes = torch.stack(spikes, dim=1)
        v_pre = torch.stack(v_pres, dim=1)
        vs_seq = torch.stack(vs_all, dim=1)
        spikes.clear()
        v_pres.clear()
        vs_all.clear()

        # Saved: the mixed membrane and the slow-compartment trajectory. The fast
        # compartment is recovered inside the backward kernel as
        # `v_pre - w*vs`, and the spike and reset factor from `v_pre >= thr` --
        # three [B, L, d] tensors' worth of state carried in two.
        ctx.save_for_backward(v_pre, vs_seq, vs0, w, beta_s)
        ctx.beta_f = beta_f
        ctx.thr = thr
        ctx.alpha = alpha
        return out_spikes, torch.cat([vf, vs], dim=1)

    @staticmethod
    def backward(ctx, grad_spikes: Tensor, grad_v_final: Tensor):
        # Built from jiterator calls, so it is not itself differentiable: under
        # `create_graph=True` the scan's second-order contribution would simply be
        # absent, silently, whenever some other term in the loss also depends on
        # `cur`. `@once_differentiable` does not work here (see the note in
        # `snn/neuron.py`), so the check is explicit. The engine enables grad
        # inside a backward if and only if the caller asked for `create_graph`, so
        # this fires exactly then -- never on the normal path and never inside a
        # capture, where grad is disabled.
        if torch.is_grad_enabled():
            raise RuntimeError(
                "FusedTwoCompartmentScan does not support double backward: its "
                "backward is hand-written jiterator code with no autograd "
                "history, so the second-order term would be silently missing. "
                "Use twocomp_scan(..., fused=False) if you need create_graph=True."
            )
        v_pre, vs_seq, vs0, w, beta_s = ctx.saved_tensors
        bwd = twocomp_backward_kernel()
        beta_f, thr, alpha = ctx.beta_f, ctx.thr, ctx.alpha
        length = v_pre.shape[1]
        d = v_pre.shape[2]

        # set_materialize_grads is left at its default True, so both cotangents
        # are real zero tensors when an output is unused. That keeps the loop free
        # of Python-side conditionals, which a captured graph would bake in anyway.
        grad_vf = grad_v_final[:, :d]
        grad_vs = grad_v_final[:, d:]
        grad_curs: list[Tensor] = [None] * length   # type: ignore[list-item]
        gv_elems: list[Tensor] = [None] * length    # type: ignore[list-item]
        gvs_elems: list[Tensor] = [None] * length   # type: ignore[list-item]
        for t in range(length - 1, -1, -1):
            (grad_curs[t], grad_vf, grad_vs,
             gv_elems[t], gvs_elems[t]) = bwd(
                grad_spikes[:, t], grad_vf, grad_vs, v_pre[:, t], vs_seq[:, t],
                w, beta_s, beta_f=beta_f, thr=thr, alpha=alpha,
            )
        grad_cur = torch.stack(grad_curs, dim=1)
        gv = torch.stack(gv_elems, dim=1)
        gvs = torch.stack(gvs_elems, dim=1)
        grad_curs.clear()
        gv_elems.clear()
        gvs_elems.clear()

        # The two reductions jiterator cannot do, paid once each rather than once
        # per timestep (EXP_002 §8.3). `vs_prev` is the slow trajectory shifted by
        # one step with the initial state in front -- dL/dbeta_s at step t is
        # dL/dvs_t * vs_{t-1}, and vs_{-1} is v0's slow half.
        vs_prev = torch.cat([vs0.unsqueeze(1), vs_seq[:, :-1]], dim=1)
        grad_w = (gv * vs_seq).sum(dim=(0, 1)).reshape(1, d)
        grad_beta_s = (gvs * vs_prev).sum(dim=(0, 1)).reshape(1, d)

        grad_v0 = torch.cat([grad_vf, grad_vs], dim=1)
        # grads for (cur, v0, w, beta_s, beta_f, thr, alpha)
        return grad_cur, grad_v0, grad_w, grad_beta_s, None, None, None


def twocomp_scan(cur: Tensor, v0: Tensor, w: Tensor, beta_s: Tensor,
                 beta_f: float, thr: float, alpha: float,
                 fused: bool) -> tuple[Tensor, Tensor]:
    """Dispatching entry point. Returns (spikes [B, L, d], v_final [B, 2d]).

    Falls back to `twocomp_scan_eager` whenever the fused path is unavailable or
    disabled. The two must stay numerically interchangeable; that is the R10 gate,
    not a nice-to-have. fp32 is enforced on both branches, because the `--fused`
    and `--no-fused` arms have to be comparable and would not be if one of them
    silently accepted a bf16 membrane.
    """
    from snn.kernels import jiterator_available

    _check(cur, v0, w, beta_s)
    _check_fp32(cur, v0, w, beta_s)
    if fused and cur.is_cuda and jiterator_available():
        return FusedTwoCompartmentScan.apply(cur, v0, w, beta_s, beta_f, thr, alpha)
    return twocomp_scan_eager(cur, v0, w, beta_s, beta_f, thr, alpha)
