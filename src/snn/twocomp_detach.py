"""Two-compartment LIF with a **detached** fast-pole reset (EXP_017).

Pre-registered in `experiments/logs/EXP_017_bounded_reset_jacobian.md`. Read that
first: the derivation, the bars and the decision rule were all fixed before this
file existed, and the reasons for each design choice live there rather than here.

The neuron, per channel `c`, per timestep `t` -- **identical to `snn.twocomp` in
every line**:

    vf_t  =  beta_f   * vf_{t-1} + cur_t        fast pole; beta_f fixed at 0.5
    vs_t  =  beta_s,c * vs_{t-1} + cur_t        slow pole; beta_s,c learned
    v_t   =  vf_t + w_c * vs_t                  the mix that reaches the threshold
    s_t   =  1[ v_t >= thr ]                    I3, hard threshold
    vf_t <-  vf_t * (1 - s_t)                   hard reset, the baseline's rule
    vs_t <-  vs_t                               RESET-SHIELDED

**The forward is not merely equivalent to the adopted arm's -- it is the same
compiled kernel.** `twocomp_forward_kernel` is imported from `snn.twocomp` and
called, not re-declared, so there is no source text to drift and no second NVRTC
compile. The arm therefore adds **no parameter, no state variable, no function and
no inference cost**, and a checkpoint trained here loads into `arch="twocomp"` and
evaluates **bit-identically** (`tests/test_twocomp_detach_equivalence.py::
test_the_inference_model_is_the_adopted_arm_bitwise`). Cf. `EXP_008`/`EXP_011`,
whose fold-in identities were exact only up to a GEMM-reassociation residual;
this one has no residual because there is nothing to fold.

The single variable is the **backward**: the reset factor is treated as a
constant, exactly as `kernels.py`'s `"detached"` mode does for the plain LIF.

Why this arm exists
-------------------
`EXP_016` proved that the adopted arm's backward-through-time is **bounded by
nothing**, and that this is why it cannot be trained at 5.0M parameters
(`EXP_015`: NaN at step 5138, deterministically, where the plain LIF and the GRU
at the identical width, seed, recipe and tree train cleanly).

The mechanism is one line. Both neurons thread an adjoint over all `L = 256`
steps with `grad_vf_prev = beta_f * (grad_vf_next * dv + grad_spike * sgd)`, so
`beta_f * dv` is the per-step multiplier of the homogeneous reverse recurrence:

    plain LIF   dv = (1 - s)  - v_pre * sg(v_pre - thr)     SAME variable twice
    twocomp     dv = (1 - sh) - vf    * sgd(v_pre - thr)    DIFFERENT variables

The LIF's is self-limiting -- `sg` decays exactly as fast as `v_pre` grows -- and
at the frozen constants `max|beta * dv| = 0.5123596 < 1` for every finite
membrane at **every width**. The two-compartment neuron breaks that because the
spike test is on the *mixed* membrane while the reset applies to the *fast* one
alone: `vf * sgd = v_pre * sgd - w_c * vs * sgd`, and `vs` is never reset. The
crossover is at `|w_c * vs| ~ 1`; `EXP_016` measured `max|w*vs| = 279.7` at
`d = 1481` and `max|beta_f*dv| = 8.96`, 17.5x the LIF's hard ceiling, with 85
consecutive expanding timesteps on the batch the run actually died on.

**No recipe term appears in `dv`.** No learning rate, no schedule, no weight
decay, and -- decisively -- no gradient clip, which reads a norm that is already
NaN by the time `backward()` returns. So the remedy cannot be a hyperparameter,
and `EXP_016` §10 item 2 referred it as a new arm. This is that arm.

What detaching the reset does to the Jacobian
---------------------------------------------
Write the per-step homogeneous backward as a 2x2 matrix acting on
`(grad_vf_next, grad_vs_next)`. For the adopted arm:

    J_hard = [[ beta_f * dv          ,  0      ],
              [ -beta_s * w * sgd*vf ,  beta_s ]]        dv = (1 - sh) - vf*sgd

Both the diagonal entry and the off-diagonal one carry `vf`, which is unbounded
relative to the surrogate's argument. Detaching the reset sends
`d vf_out / d v` to zero, which removes `vf` from **both**:

    J_detach = [[ beta_f * (1 - s) ,  0      ]
                [ 0                ,  beta_s ]]

* **Diagonal.** The off-diagonal coupling is not merely bounded, it is exactly
  zero, so there is no non-normal transient either -- the product over any window
  is exactly the product of the diagonals.
* `(1 - s)` is in `{0, 1}`, so `|beta_f * (1 - s)| <= beta_f = 0.5`. That is a
  **strictly tighter** bound than the plain LIF's own 0.5123596, and it holds for
  every `w`, every `vs`, every batch, every channel, every step and **every
  width** -- nothing in it depends on `d_model`, on the mix, or on the slow pole.
* `beta_s = sigmoid(beta_s_raw)` is in `(0, 1)` by construction and weight decay
  pulls `beta_s_raw` toward 0, i.e. toward *more* damping (`EXP_015` ruled out a
  runaway slow pole on exactly this argument, without a GPU).

So the reverse recurrence is a strict contraction on both compartments. The slow
pole's contraction rate is `beta_s`, **unchanged from the adopted arm** -- the
memory mechanism the arm exists for is not touched. What is removed is only the
two terms that carry the reset back through the spike.

The cost, stated plainly
------------------------
**This is a biased gradient estimator, and the bias is not small in kind.** The
pathway "firing now lowers my own future membrane" is dropped: the model still
learns *through* the spike (via `grad_spike * sgd`, which is untouched) but no
longer receives the reset's own contribution to `d vf_out / d v`. Whether that
costs bpc is an empirical question `EXP_017` H3 measures at 735K against a
freshly trained anchor, and it is the reason this arm is not "free".

It is also not novel: detaching the reset is standard practice in the wider
surrogate-gradient literature (it is `detach_reset=True` in spikingjelly, and it
is `reset="detached"` in this project's own `kernels.py`, where it has been a
first-class mode with its own compiled kernel and its own mutation coverage --
M04, M05 -- since Phase 2). What is new here is the *derivation above*: that on a
**two-compartment** neuron it is not a stylistic choice about a gradient but the
one change that restores the contraction the second compartment destroyed.

Why this lives in its own module rather than in twocomp.py
----------------------------------------------------------
The same two mechanical reasons `twocomp.py`'s own docstring gives for not living
in `kernels.py`, and one more that matters more than both.

  * `scripts/audit/08_mutation_campaign.py` requires every mutation's anchor text
    to appear EXACTLY ONCE in its target file. Adding a second `dv` expression to
    `twocomp.py` would break T09's and T16's anchors and turn the campaign's own
    safety check into a failure.
  * A separate file keeps this arm's mutations cleanly separable from the adopted
    arm's 22.
  * **`twocomp.py` is not edited by this experiment at all** (`EXP_017` G1), so a
    Phase-4 candidate cannot regress the arm it is measured against -- `EXP_004`
    failure mode J9. The adopted arm's committed kernel, its committed gate and
    its committed mutation evidence are untouched.

The cost of the split is the same one `twocomp.py` records: a torch upgrade that
breaks the private jiterator API is now a three-file repair rather than two
(risk R11). Accepted, and reduced by importing the forward kernel rather than
re-declaring it.

Numerical contract
------------------
The forward's contract is `snn.twocomp`'s, inherited by construction -- same
source text, same kernel, same suppressed FMA contractions.

The backward is written in the SAME regrouped form as `twocomp.py`'s and
`kernels.py`'s, `grad_vf_next * dv + grad_spike * sgd`, so that at `w = 0` this
kernel's backward is **bit-identical** to the Phase-2 LIF's `reset="detached"`
mode rather than merely equal to it within fp32 rounding. That nesting is this
file's strongest gate: `reset="detached"` on the plain LIF has survived the
committed mutation campaign, so tying the new kernel to it bitwise inherits that
evidence.

`vf` is **not recovered** in this backward and `vs` is **not passed to it** --
neither appears in any of the five outputs, which is the whole point of the arm
stated as an argument list. That drops the kernel from seven tensor inputs to six.

R10 applies here in full: the backward is hand-written, jiterator outputs carry no
autograd history, and a wrong backward still trains and still produces a plausible
bpc. `tests/test_twocomp_detach_equivalence.py` is the gate.
"""

from __future__ import annotations

import functools
from typing import Callable

import torch
from torch import Tensor

from snn.surrogate import ATAN_CUDA_GRAD, atan_spike

# Imported rather than re-declared, deliberately, in all three cases:
#   * `twocomp_forward_kernel` -- so the forward is the SAME compiled kernel and
#     there is no second copy of the source text to drift;
#   * `_check` / `_check_fp32` -- so the shape and dtype contracts cannot
#     desynchronise from the adopted arm's. Private names crossing a module
#     boundary inside one package is the honest way to say "this is the same
#     contract", rather than a second copy that looks like a second decision.
from snn.twocomp import _check, _check_fp32, twocomp_forward_kernel

__all__ = [
    "twocomp_detach_scan_eager",
    "FusedTwoCompDetachScan",
    "twocomp_detach_scan",
    "twocomp_detach_backward_kernel",
    "twocomp_detach_backward_source",
    "clear_twocomp_detach_kernel_cache",
    "MAX_CHAIN_FACTOR",
]

#: The derived bound on `|beta_f * dv|`, the per-step multiplier of the
#: homogeneous reverse recurrence, at this project's frozen `beta_f = 0.5`.
#: `dv = 1 - s` is in {0, 1}, so the bound is `beta_f` exactly -- with no
#: dependence on `w`, `vs`, `d_model` or the batch. Quoted here so the probe, the
#: test and the results script read one number instead of three copies of it.
#:
#: For comparison, `snn.kernels`' hard-reset LIF is bounded by 0.5123596 (a
#: transcendental extremum of `|v_pre * sg(v_pre - thr)|`, `EXP_016` §2.2) and
#: `snn.twocomp` is bounded by nothing at all.
MAX_CHAIN_FACTOR = 0.5


# --------------------------------------------------------------------------
# CUDA source -- backward only; the forward is snn.twocomp's, unmodified
# --------------------------------------------------------------------------

# ARGUMENT ORDER IS LOAD-BEARING, for `EXP_002` §8.4's reason: jiterator binds
# every TENSOR argument first, in declaration order, and only then the scalars as
# keywords. A declaration that interleaves them compiles, runs, produces
# plausible spikes, and silently binds a per-channel tensor into a scalar slot.
# So: six tensors, then three scalars.
#
# Derived from `snn.twocomp`'s forward, term by term, with the reset factor held
# constant. With vf, vs, v = vf + w*vs, s = H(v - thr), sg = ds/dv, and
# vf_out = vf*(1 - s.detach()):
#
#   dL/dv       = sg * grad_spike                  ... vf_out no longer depends
#                                                      on v AT ALL. This is the
#                                                      single changed line, and
#                                                      the `- grad_vf_next * vf`
#                                                      term that used to sit here
#                                                      is the unbounded one.
#   dL/dvf      = grad_vf_next * (1 - s) + dL/dv
#   dL/dvs      = grad_vs_next + w * dL/dv         ... the slow pole is shielded,
#                                                      so its only extra path is
#                                                      still through the mix
#   dL/dcur     = dL/dvf + dL/dvs                  ... cur enters both
#   dL/dvf_prev = beta_f   * dL/dvf
#   dL/dvs_prev = beta_s_c * dL/dvs
#
# `dL/dvf` is computed in the REGROUPED form `grad_vf_next*dv + grad_spike*sgd`
# with `dv = 1 - s`, which is the same expression algebraically and is written
# this way on purpose: it is character-for-character the grouping
# `snn/kernels.py` uses for its `g`, so at `w = 0` this kernel's backward is
# **bit-identical** to the Phase-2 LIF's `"detached"` mode rather than merely
# equal to it within fp32 rounding. Written the other way round the adopted arm's
# equivalent leg agreed only to ~2 ulp (`twocomp.py:146-156` records the
# measurement), and `test_w_zero_is_the_committed_detached_lif_scan` is worth much
# more asserted at `== 0.0` than at a tolerance.
#
# NOTE WHAT IS ABSENT, because the absences ARE the arm:
#   * `vf` is never recovered -- `v_pre - w_c*vs` does not appear. It cannot: no
#     output depends on it.
#   * `vs` is not even an argument. The adopted arm needs it only to recover
#     `vf`; with `vf` gone the kernel drops from seven tensor inputs to six.
#     `vs_seq` is still SAVED, because the two per-channel reductions outside the
#     loop need it -- but it no longer crosses the kernel boundary.
#
# The two per-channel gradients are REDUCTIONS over batch and time, and jiterator
# cannot reduce (§2.3). This kernel emits `dL/dv` and `dL/dvs` per element and the
# caller does both reductions once at the end, which is `EXP_002` §8.3's measured
# choice and is unchanged from the adopted arm.
_TCD_BACKWARD_SRC = """
template <typename T>
void tcd_backward(T grad_spike, T grad_vf_next, T grad_vs_next, T v_pre,
                  T w_c, T beta_s_c,
                  T beta_f, T thr, T alpha,
                  T& grad_cur, T& grad_vf_prev, T& grad_vs_prev,
                  T& gv_elem, T& gvs_elem) {{
    const T x   = v_pre - thr;
    const T sh  = (x >= T(0)) ? T(1) : T(0);
    const T sgd = {atan_grad};
    const T dv  = T(1) - sh;
    const T gf  = grad_vf_next * dv + grad_spike * sgd;
    const T gv  = sgd * grad_spike;
    const T gs  = grad_vs_next + w_c * gv;
    grad_cur     = gf + gs;
    grad_vf_prev = beta_f * gf;
    grad_vs_prev = beta_s_c * gs;
    gv_elem      = gv;
    gvs_elem     = gs;
}}
"""


def twocomp_detach_backward_source() -> str:
    """CUDA C++ text for the backward kernel. Exposed for inspection/tests."""
    return _TCD_BACKWARD_SRC.format(atan_grad=ATAN_CUDA_GRAD)


# --------------------------------------------------------------------------
# compiled-kernel cache
# --------------------------------------------------------------------------

@functools.lru_cache(maxsize=1)
def twocomp_detach_backward_kernel() -> Callable:
    """`fn(grad_spike, grad_vf_next, grad_vs_next, v_pre, w_c, beta_s_c,
    beta_f=, thr=, alpha=)`.

    Returns `(grad_cur, grad_vf_prev, grad_vs_prev, gv_elem, gvs_elem)`. Six
    tensor inputs and five outputs, against jiterator's 8/8 limit -- one narrower
    than the adopted arm's backward, because `vs` is no longer needed.

    There is no forward kernel in this module. `snn.twocomp.twocomp_forward_kernel`
    is used unchanged, which is what makes the forward identical by construction
    rather than by test.
    """
    from torch.cuda.jiterator import _create_multi_output_jit_fn

    return _create_multi_output_jit_fn(
        twocomp_detach_backward_source(), num_outputs=5,
        beta_f=0.5, thr=1.0, alpha=2.0,
    )


def clear_twocomp_detach_kernel_cache() -> None:
    """Drop the compiled kernel. Only useful in tests that edit the source."""
    twocomp_detach_backward_kernel.cache_clear()


# --------------------------------------------------------------------------
# eager reference -- the ground truth for R10
# --------------------------------------------------------------------------

def twocomp_detach_scan_eager(cur: Tensor, v0: Tensor, w: Tensor, beta_s: Tensor,
                              beta_f: float, thr: float,
                              alpha: float) -> tuple[Tensor, Tensor]:
    """Reference detached-reset two-compartment scan, built from autograd ops.

    `cur` [B, L, d], `v0` [B, 2d], `w` and `beta_s` [1, d]; returns
    (spikes [B, L, d], v_final [B, 2d]). Runs on CPU or CUDA and is the
    `--no-fused` path.

    Written for legibility, not speed: this is what the hand-written backward is
    checked against, so it has to be obviously the equations in the module
    docstring. Unlike `twocomp_detach_scan` it does not insist on fp32 -- the gate
    runs it in float64 to get a reference better than the thing it is checking,
    and `torch.autograd.gradcheck` needs float64 to mean anything.

    **This body differs from `snn.twocomp.twocomp_scan_eager` in exactly one
    token**: `s.detach()` in the reset. Kept as a separate function rather than a
    flag on that one so the adopted arm's reference -- the ground truth every
    committed two-compartment number rests on -- is not edited by this experiment.

    The order of operations matters and is not free to change, for the reason the
    adopted arm's reference records: `vf * beta_f + cur[:, t]` is a multiply and
    an add, rounding twice, and so is `vf + w * vs`. The fused path suppresses FMA
    contraction at exactly those two places.
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
        # THE SINGLE VARIABLE. Same forward value as the adopted arm's
        # `vf * (1.0 - s)` -- `detach()` changes no number -- and a different
        # backward: d vf_out / d v is zero instead of -vf, which is the term
        # EXP_016 showed is bounded by nothing.
        vf = vf * (1.0 - s.detach())
        # vs is NOT reset -- inherited from the adopted arm, and untouched here.
    return torch.stack(spikes, dim=1), torch.cat([vf, vs], dim=1)


# --------------------------------------------------------------------------
# fused path
# --------------------------------------------------------------------------

class FusedTwoCompDetachScan(torch.autograd.Function):
    """One jiterator kernel per timestep forward, one per timestep backward.

    Non-tensor arguments are three floats, so nothing unhashable or
    device-dependent crosses the autograd boundary and the scan stays
    CUDA-graph-capture safe -- the same contract `FusedLIFScan` and
    `FusedTwoCompartmentScan` keep.

    The forward body is `FusedTwoCompartmentScan`'s, calling the same kernel on
    the same tensors and saving the same things; only `backward` dispatches
    elsewhere. It is written out rather than inherited because an
    `autograd.Function`'s `forward` and `backward` are a matched pair and
    subclassing one half of it is how they silently stop matching.
    """

    @staticmethod
    def forward(ctx, cur: Tensor, v0: Tensor, w: Tensor, beta_s: Tensor,
                beta_f: float, thr: float, alpha: float) -> tuple[Tensor, Tensor]:
        # Repeated rather than trusted to `twocomp_detach_scan`, because `.apply`
        # is a public entry point that skips it, and because the kernel's membrane
        # recursion is written with float-only intrinsics: handed float64 it would
        # narrow to fp32 and cast back, returning a double that is secretly a
        # float. Host-side check only -- no sync, so it is capture-safe.
        _check_fp32(cur, v0, w, beta_s)
        d = _check(cur, v0, w, beta_s)
        fwd = twocomp_forward_kernel()   # snn.twocomp's, unchanged
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

        # Saved: the mixed membrane and the slow-compartment trajectory -- the
        # same two tensors the adopted arm saves, at the same cost. `vs_seq` no
        # longer reaches the backward KERNEL (nothing in it needs `vf`), but the
        # two per-channel reductions after the loop still do.
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
                "FusedTwoCompDetachScan does not support double backward: its "
                "backward is hand-written jiterator code with no autograd "
                "history, so the second-order term would be silently missing. "
                "Use twocomp_detach_scan(..., fused=False) for create_graph=True."
            )
        v_pre, vs_seq, vs0, w, beta_s = ctx.saved_tensors
        bwd = twocomp_detach_backward_kernel()
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
                grad_spikes[:, t], grad_vf, grad_vs, v_pre[:, t],
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
        #
        # Both gradients stay LIVE under the detached reset, which is not
        # self-evident and is asserted in the gate rather than argued here:
        # `gv = sgd * grad_spike` is nonzero wherever the surrogate is, so `w`
        # still learns through the spike, and `gs` carries `grad_vs_next` besides.
        vs_prev = torch.cat([vs0.unsqueeze(1), vs_seq[:, :-1]], dim=1)
        grad_w = (gv * vs_seq).sum(dim=(0, 1)).reshape(1, d)
        grad_beta_s = (gvs * vs_prev).sum(dim=(0, 1)).reshape(1, d)

        grad_v0 = torch.cat([grad_vf, grad_vs], dim=1)
        # grads for (cur, v0, w, beta_s, beta_f, thr, alpha)
        return grad_cur, grad_v0, grad_w, grad_beta_s, None, None, None


def twocomp_detach_scan(cur: Tensor, v0: Tensor, w: Tensor, beta_s: Tensor,
                        beta_f: float, thr: float, alpha: float,
                        fused: bool) -> tuple[Tensor, Tensor]:
    """Dispatching entry point. Returns (spikes [B, L, d], v_final [B, 2d]).

    Falls back to `twocomp_detach_scan_eager` whenever the fused path is
    unavailable or disabled. The two must stay numerically interchangeable; that
    is the R10 gate, not a nice-to-have. fp32 is enforced on both branches,
    because the `--fused` and `--no-fused` arms have to be comparable and would
    not be if one of them silently accepted a bf16 membrane.
    """
    from snn.kernels import jiterator_available

    _check(cur, v0, w, beta_s)
    _check_fp32(cur, v0, w, beta_s)
    if fused and cur.is_cuda and jiterator_available():
        return FusedTwoCompDetachScan.apply(cur, v0, w, beta_s, beta_f, thr, alpha)
    return twocomp_detach_scan_eager(cur, v0, w, beta_s, beta_f, thr, alpha)
