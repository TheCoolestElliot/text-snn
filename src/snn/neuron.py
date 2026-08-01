"""LIF scan: eager reference and jiterator-fused autograd.Function (spec 02a §7).

The neuron is the canonical single-timescale leaky integrate-and-fire cell:

    v_pre_t = beta * v_{t-1} + cur_t
    s_t     = 1[v_pre_t >= thr]                       (I3, hard threshold)
    v_t     = reset(v_pre_t, s_t)

with the surrogate of `snn.surrogate` standing in for ds/dv_pre during the
backward pass (I4). There is no lateral recurrent matrix anywhere in this file
(I5): the only thing carried across time is the neuron's own membrane.

Two implementations of the same function live here, and that duplication is
deliberate. `lif_scan_eager` is built from ordinary autograd ops and is the
*ground truth*; `FusedLIFScan` issues one hand-written CUDA kernel per timestep
and one per backward step.

Measured at the spec §13 baseline shape (B=128, L=256, d=512, beta=0.5, hard
reset), the *kernel counts* are stable to three figures across runs: 13.01 ->
1.02 per timestep forward, 27.03 -> 2.06 for forward+backward. The wall-clock
speedups they buy are not stable to three figures and must not be quoted as if
they were -- across repeated runs on this box they span roughly 4.8-7.7x
forward-only (50-64 ms -> 6.6-9.6 ms) and 11.8-17.4x forward+backward
(246-263 ms -> 15.9-20.3 ms); clock and power state move them by ~30%. Captured
into a CUDA graph the same forward+backward step takes 3.5-3.7 ms, i.e. ~4-4.5x
over the ungraphed fused path and ~67-72x over eager. That last figure is the
§8.1.2 "fusion and graphing compose" claim measured on the real object rather
than on a microbenchmark. Report ranges, not points; the numbers a run actually
produced come from `tests/test_kernel_count.py -s`.

Risk R10 says a hand-written backward that is silently wrong is the worst bug
class in the project, because the model still trains and the bpc still looks
plausible. The mitigation is structural: the fast path never gets to be the only
path, and `tests/test_neuron_equivalence.py` checks spikes bit-identically and
gradients to fp32 tolerance across all four reset modes on every change. That
test file has itself been mutation-tested -- sixteen plausible mis-derivations
injected into the kernel source, sixteen caught.

Layout is `[B, L, d]` throughout (spec §2). `cur[:, t]` is a strided view, which
is fine: the fused kernel is elementwise and TensorIterator reads it directly, so
no contiguity copy is issued inside the time loop.
"""

from __future__ import annotations

import torch
from torch import Tensor

from snn.kernels import (
    RESET_CODES,
    RESET_MODES,
    jiterator_available,
    lif_backward_kernel,
    lif_forward_kernel,
)
from snn.surrogate import atan_spike

__all__ = [
    "lif_scan_eager",
    "FusedLIFScan",
    "lif_scan",
    "RESET_MODES",
    "RESET_CODES",
]


def _check_shapes(cur: Tensor, v0: Tensor, reset: str) -> None:
    if reset not in RESET_CODES:
        raise ValueError(f"unknown reset mode {reset!r}; expected one of {RESET_MODES}")
    if cur.dim() != 3:
        raise ValueError(f"cur must be [B, L, d]; got shape {tuple(cur.shape)}")
    if v0.dim() != 2:
        raise ValueError(f"v0 must be [B, d]; got shape {tuple(v0.shape)}")
    if v0.shape[0] != cur.shape[0] or v0.shape[1] != cur.shape[2]:
        raise ValueError(
            f"v0 shape {tuple(v0.shape)} does not match cur {tuple(cur.shape)}"
        )
    if cur.shape[1] < 1:
        raise ValueError("cur must have at least one timestep")
    if cur.dtype is not v0.dtype:
        raise TypeError(f"cur ({cur.dtype}) and v0 ({v0.dtype}) must share a dtype")


def _check_fp32(cur: Tensor, v0: Tensor) -> None:
    """Membrane state and the threshold comparison are fp32 unconditionally.

    Spec §2 and bottleneck B8: bf16's representable spacing at v = 1.0 is 0.78%
    of the firing threshold, so a low-precision membrane quantises onto a grid
    straddling the threshold. `cfg.dtype` selects the GEMM dtype only. The cast
    belongs to the caller so that it happens once per layer rather than once per
    timestep -- and so that a caller who forgot it fails loudly here instead of
    silently training a quantised neuron.
    """
    if cur.dtype is not torch.float32 or v0.dtype is not torch.float32:
        raise TypeError(
            "lif_scan requires fp32 current and membrane (got "
            f"cur={cur.dtype}, v0={v0.dtype}); cast with .float() before the scan "
            "-- cfg.dtype selects the GEMM dtype only (spec §2)"
        )


# --------------------------------------------------------------------------
# eager reference
# --------------------------------------------------------------------------

def lif_scan_eager(cur: Tensor, v0: Tensor, beta: float, thr: float,
                   alpha: float, reset: str) -> tuple[Tensor, Tensor]:
    """Reference LIF scan built from ordinary autograd ops.

    `cur` [B, L, d] fp32, `v0` [B, d] fp32; returns (spikes [B, L, d] fp32,
    v_final [B, d] fp32). Runs on CPU or CUDA and is the `--no-fused` path.

    This is the ground truth for R10, so it is written for legibility rather than
    for speed: ~13 kernels per timestep, exactly the count the Phase-1 cost model
    predicts (§3.4) and the count `tests/test_kernel_count.py` asserts.

    Unlike `lif_scan` this does not insist on fp32. The model must never call it
    in another dtype -- `lif_scan` enforces that -- but the tests do run it in
    float64 to get a reference that is better than the thing it is checking, and
    `torch.autograd.gradcheck` needs float64 to mean anything at all.
    """
    _check_shapes(cur, v0, reset)
    L = cur.shape[1]
    v = v0
    spikes: list[Tensor] = []
    for t in range(L):
        v_pre = v * beta + cur[:, t]
        s = atan_spike(v_pre - thr, alpha)
        if reset == "hard":
            v = v_pre * (1.0 - s)
        elif reset == "soft":
            v = v_pre - thr * s
        elif reset == "detached":
            # The reset factor is a constant here: same forward as "hard", but
            # the -v_pre*sg term never appears in the backward.
            v = v_pre * (1.0 - s.detach())
        else:  # "none"
            v = v_pre
        spikes.append(s)
    return torch.stack(spikes, dim=1), v


# --------------------------------------------------------------------------
# fused path
# --------------------------------------------------------------------------

class FusedLIFScan(torch.autograd.Function):
    """One jiterator kernel per timestep forward, one per timestep backward.

    Non-tensor arguments are a float triple and an int `reset_code`
    (0=hard, 1=soft, 2=detached, 3=none) so that nothing unhashable or
    device-dependent crosses the autograd boundary and the whole scan stays
    CUDA-graph-capture safe.
    """

    @staticmethod
    def forward(ctx, cur: Tensor, v0: Tensor, beta: float, thr: float,
                alpha: float, reset_code: int) -> tuple[Tensor, Tensor]:
        # Repeated here rather than trusted to `lif_scan`, because `.apply` is a
        # public entry point that skips it. The membrane recursion inside the
        # kernel is written with `__fmul_rn`/`__fadd_rn`, which are float-only:
        # handed a float64 tensor the kernel narrows v_pre to fp32 and casts it
        # back, returning a float64 tensor whose membrane is only fp32-accurate
        # (measured 4.7e-07 against the eager float64 reference). A double that
        # is secretly a float is precisely the kind of silent wrongness R10 is
        # about, and it would corrupt any future use of the fused path as a
        # high-precision reference. Host-side check only -- no sync, so it is
        # safe inside a captured region.
        if cur.dtype is not torch.float32 or v0.dtype is not torch.float32:
            raise TypeError(
                "FusedLIFScan requires fp32 inputs (got "
                f"cur={cur.dtype}, v0={v0.dtype}); the kernel's membrane "
                "recursion is float-only (spec §2, §6)"
            )
        fwd = lif_forward_kernel(RESET_MODES[reset_code])
        L = cur.shape[1]

        spikes: list[Tensor] = []
        v_pres: list[Tensor] = []
        v = v0
        # jiterator outputs carry no autograd history anyway; no_grad is stated
        # explicitly because this loop must never build a graph even if a future
        # caller invokes forward() outside autograd's own grad-disabled region.
        with torch.no_grad():
            for t in range(L):
                v, s, vp = fwd(v, cur[:, t], beta=beta, thr=thr)
                spikes.append(s)
                v_pres.append(vp)

        # One stack of L tensors is a handful of kernels in total (torch.cat
        # batches its inputs), not one per timestep -- which is the whole reason
        # the per-step body may not contain anything else.
        out_spikes = torch.stack(spikes, dim=1)
        v_pre = torch.stack(v_pres, dim=1)
        spikes.clear()
        v_pres.clear()

        # Only v_pre is saved. `spike` and the reset factor are both recovered
        # from it inside the backward kernel at the cost of one comparison,
        # which halves the saved-activation memory of the scan.
        ctx.save_for_backward(v_pre)
        ctx.beta = beta
        ctx.thr = thr
        ctx.alpha = alpha
        ctx.reset_code = reset_code
        return out_spikes, v

    @staticmethod
    def backward(ctx, grad_spikes: Tensor, grad_v_final: Tensor):
        # This backward is built from jiterator calls, whose outputs carry no
        # autograd history, so it is not itself differentiable: under
        # `create_graph=True` the scan's second-order contribution is simply
        # absent from the result. That divergence from `lif_scan_eager` (which
        # *is* twice differentiable) is silent -- measured d2 sum 280.7 fused
        # against 497.7 eager, no warning -- whenever some other term in the
        # loss also depends on `cur`, because then the grad-of-grad still
        # requires grad and autograd has no reason to complain.
        #
        # `@once_differentiable` is the usual marker for this and it does NOT
        # work here, in two separate ways, both verified on this stack. Its
        # DelayedError node has inputs with no history, so it never lies on a
        # path from the grad-of-grad back to `cur` and the engine never executes
        # it -- the mixed case still returned 280.7 with the decorator applied.
        # And it runs the wrapped body inside `torch.no_grad()`, which would
        # make the check below unreachable. Hence the explicit test, undecorated.
        #
        # The engine enables grad inside a backward if and only if the caller
        # asked for `create_graph`, so this fires exactly then and never on the
        # normal path -- including inside a CUDA-graph capture, where grad is
        # disabled and no device work is added.
        if torch.is_grad_enabled():
            raise RuntimeError(
                "FusedLIFScan does not support double backward: its backward is "
                "hand-written jiterator code with no autograd history, so the "
                "second-order term would be silently missing. Use "
                "lif_scan(..., fused=False) if you need create_graph=True."
            )
        (v_pre,) = ctx.saved_tensors
        bwd = lif_backward_kernel(RESET_MODES[ctx.reset_code])
        beta, thr, alpha = ctx.beta, ctx.thr, ctx.alpha
        L = v_pre.shape[1]

        # ctx.set_materialize_grads is left at its default True, so both
        # cotangents are real zero tensors when an output is unused. That keeps
        # the loop free of Python-side conditionals, which a captured graph
        # would bake in anyway.
        grad_v = grad_v_final
        grad_curs: list[Tensor] = [None] * L  # type: ignore[list-item]
        for t in range(L - 1, -1, -1):
            grad_curs[t], grad_v = bwd(
                grad_spikes[:, t], grad_v, v_pre[:, t],
                beta=beta, thr=thr, alpha=alpha,
            )
        grad_cur = torch.stack(grad_curs, dim=1)
        grad_curs.clear()
        # grads for (cur, v0, beta, thr, alpha, reset_code)
        return grad_cur, grad_v, None, None, None, None


def lif_scan(cur: Tensor, v0: Tensor, beta: float, thr: float, alpha: float,
             reset: str, fused: bool) -> tuple[Tensor, Tensor]:
    """Dispatching entry point. Returns (spikes [B, L, d], v_final [B, d]).

    Falls back to `lif_scan_eager` whenever the fused path is unavailable or
    disabled, which is the `--no-fused` path required by the Phase-2 entry
    checklist. The two must stay numerically interchangeable; that is the R10
    gate, not a nice-to-have.

    This is the model-facing entry point, so fp32 is enforced on both branches:
    the `--fused` and `--no-fused` arms have to be comparable, which they would
    not be if one of them silently accepted a bf16 membrane.
    """
    _check_shapes(cur, v0, reset)
    _check_fp32(cur, v0)
    if fused and cur.is_cuda and jiterator_available():
        return FusedLIFScan.apply(cur, v0, beta, thr, alpha, RESET_CODES[reset])
    return lif_scan_eager(cur, v0, beta, thr, alpha, reset)
