"""R10 gate for the detached-reset two-compartment scan (EXP_017).

Risk R10 is the project's worst bug class: a hand-written jiterator backward that
is *silently* wrong. No crash, no NaN, no broken loss curve -- the model trains,
the bpc looks plausible, and every conclusion drawn from it is worthless. This
file is `snn.twocomp_detach`'s version of `test_twocomp_equivalence.py`, and
`EXP_017` entry condition J1 is that it passes before a single training step runs.

It carries the five legs the adopted arm's gate carries, and **three more that
only exist because of what this arm claims**:

  6. **the forward is the ADOPTED ARM'S, bitwise.** Not "identical in function
     class" as `EXP_008`'s and `EXP_011`'s threshold arms were -- those folded
     with a GEMM-reassociation residual of 1.9e-07 to 1.9e-03 -- but the same
     compiled kernel on the same tensors, asserted at `== 0.0`. This is the claim
     that makes the arm free at inference, so it is checked rather than argued.
  7. **the bound**, measured at the kernel boundary on the intermediate itself,
     against the closed form `snn.twocomp_detach.MAX_CHAIN_FACTOR`. And measured
     the same way on the *adopted* arm in the same regime, where it must exceed 1
     -- because a bound nobody could have violated is not evidence.
  8. **nothing died.** A detached reset removes two terms from the backward, and
     the failure mode of removing terms is a parameter that silently stops
     learning. `EXP_004`'s §7.2 screen caught exactly that once already.

WHY THE PARAMETERS ARE SWEPT, AND SWEPT PER CHANNEL. Inherited verbatim from the
adopted arm's gate, and the reasons are unchanged: `beta_f == beta_s` makes the
two compartments indistinguishable, and a per-channel parameter that is CONSTANT
across channels cannot detect `EXP_002` §8.4's failure -- jiterator binding a
`[1, d]` tensor into a scalar slot, which compiles, runs and produces plausible
spikes. This kernel has SIX tensor arguments where the adopted arm's has seven,
so its binding order is a different order and inherits none of that evidence.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest
import torch

_SRC = str(Path(__file__).resolve().parents[1] / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from snn.kernels import jiterator_available  # noqa: E402
from snn.neuron import lif_scan  # noqa: E402
from snn.twocomp import twocomp_backward_kernel, twocomp_scan  # noqa: E402
from snn.twocomp_detach import (  # noqa: E402
    MAX_CHAIN_FACTOR,
    FusedTwoCompDetachScan,
    twocomp_detach_backward_kernel,
    twocomp_detach_scan,
    twocomp_detach_scan_eager,
)

requires_fused = pytest.mark.skipif(
    not (torch.cuda.is_available() and jiterator_available()),
    reason="needs a CUDA device with a working jiterator",
)

ALPHA = 2.0
THR = 1.0

SHAPES = [(4, 8, 16), (3, 64, 32), (2, 128, 48), (5, 256, 16)]

# (beta_f, beta_s_centre, w_centre, thr). Identical to the adopted arm's sweep, on
# purpose: the two arms are compared against each other in several legs below and
# a differing sweep would make those comparisons two experiments instead of one.
REGIMES = [
    (0.5, 0.95, 0.10, 1.0),
    (0.9, 0.70, 0.50, 1.0),
    (0.5, 0.99, -0.30, 0.5),
    (0.7, 0.80, 0.90, 1.5),
    (0.3, 0.95, 0.25, 2.0),
]


def _inputs(B, L, d, seed, thr=1.0, beta_s=0.95, w=0.1,
            device="cuda", dtype=torch.float32, scale=1.1):
    """Currents scaled with the threshold so the firing rate stays sane.

    Byte-for-byte the adopted arm's helper, deliberately: several legs below
    compare the two arms on the same inputs, and a second copy that drifted would
    turn an equality assertion into a coincidence.
    """
    g = torch.Generator(device="cpu").manual_seed(seed)
    cur = torch.randn(B, L, d, generator=g, dtype=torch.float64) * (scale * thr)
    v0 = torch.randn(B, 2 * d, generator=g, dtype=torch.float64) * (0.3 * thr)
    bs = (beta_s + 0.04 * torch.randn(1, d, generator=g, dtype=torch.float64))
    bs = bs.clamp(0.05, 0.995)
    ww = w + 0.25 * torch.randn(1, d, generator=g, dtype=torch.float64)
    to = lambda t: t.to(device=device, dtype=dtype)  # noqa: E731
    return to(cur), to(v0), to(ww), to(bs)


def _grads(cur, v0, w, beta_s, beta_f, thr, gs, gv, fused):
    """Gradients of every differentiable input, from one backward pass."""
    c = cur.detach().clone().requires_grad_(True)
    v = v0.detach().clone().requires_grad_(True)
    ww = w.detach().clone().requires_grad_(True)
    bb = beta_s.detach().clone().requires_grad_(True)
    if fused:
        s, vf = twocomp_detach_scan(c, v, ww, bb, beta_f, thr, ALPHA, fused=True)
    else:
        s, vf = twocomp_detach_scan_eager(c, v, ww, bb, beta_f, thr, ALPHA)
    ((s * gs).sum() + (vf * gv).sum()).backward()
    return c.grad, v.grad, ww.grad, bb.grad


def _cotangents(B, L, d, seed, device="cuda", dtype=torch.float32):
    g = torch.Generator(device="cpu").manual_seed(seed)
    gs = torch.randn(B, L, d, generator=g, dtype=torch.float64)
    gv = torch.randn(B, 2 * d, generator=g, dtype=torch.float64)
    return gs.to(device=device, dtype=dtype), gv.to(device=device, dtype=dtype)


# --------------------------------------------------------------------------
# leg 2: forward and backward transcribed from EXP_017 §2, independent of src/
# --------------------------------------------------------------------------

def _spec_forward(cur, v0, w, beta_s, beta_f, thr):
    """EXP_017 §2, transcribed by hand. Returns (spikes, v_final, v_pre, vs_seq).

    Identical to the adopted arm's spec forward, because the forward IS identical
    -- `detach()` changes no number. Transcribed here anyway rather than imported,
    so that leg 6's "the two forwards agree" is two independent statements of the
    forward agreeing, not one statement compared with itself.
    """
    L, d = cur.shape[1], cur.shape[2]
    vf, vs = v0[:, :d], v0[:, d:]
    spikes, v_pres, vs_seq = [], [], []
    for t in range(L):
        vf = vf * beta_f + cur[:, t]
        vs = vs * beta_s + cur[:, t]
        v = vf + w * vs
        s = (v >= thr).to(v.dtype)
        spikes.append(s)
        v_pres.append(v)
        vs_seq.append(vs)
        vf = vf * (1 - s)
        # vs is NOT reset
    return (torch.stack(spikes, 1), torch.cat([vf, vs], 1),
            torch.stack(v_pres, 1), torch.stack(vs_seq, 1))


def _spec_backward(v_pre, vs_seq, vs0, grad_spikes, grad_v_final,
                   w, beta_s, beta_f, thr, alpha):
    """The adjoint recursion, derived here from §2's forward rather than read off
    `twocomp_detach.py`.

    With v = vf + w*vs, s = H(v - thr), vf_out = vf*(1 - s.detach()), vs shielded:

        dL/dv       = sg * grad_spike            <-- the ONE line that differs
                                                     from the adopted arm, which
                                                     has `sg*(grad_spike
                                                     - grad_vf_next*vf)`
        dL/dvf      = grad_vf_next * (1 - s) + dL/dv
        dL/dvs      = grad_vs_next + w * dL/dv
        dL/dcur     = dL/dvf + dL/dvs
        dL/dvf_prev = beta_f * dL/dvf ,   dL/dvs_prev = beta_s * dL/dvs
        dL/dw       = sum_t (dL/dv_t)  * vs_t
        dL/dbeta_s  = sum_t (dL/dvs_t) * vs_{t-1}

    `math.pi` is used directly rather than `snn.surrogate`, so a wrong constant
    there shows up here instead of cancelling out.
    """
    L, d = v_pre.shape[1], v_pre.shape[2]
    x = v_pre - thr
    s = (x >= 0).to(v_pre.dtype)
    u = (math.pi / 2.0) * alpha * x
    sg = (alpha / 2.0) / (1.0 + u * u)
    vs_prev = torch.cat([vs0.unsqueeze(1), vs_seq[:, :-1]], dim=1)

    grad_vf, grad_vs = grad_v_final[:, :d], grad_v_final[:, d:]
    grad_cur, gv_all, gvs_all = [None] * L, [None] * L, [None] * L
    for t in range(L - 1, -1, -1):
        gv = sg[:, t] * grad_spikes[:, t]
        gf = grad_vf * (1 - s[:, t]) + gv
        gsl = grad_vs + w * gv
        grad_cur[t], gv_all[t], gvs_all[t] = gf + gsl, gv, gsl
        grad_vf, grad_vs = beta_f * gf, beta_s * gsl
    gv_all = torch.stack(gv_all, 1)
    gvs_all = torch.stack(gvs_all, 1)
    return (torch.stack(grad_cur, 1),
            torch.cat([grad_vf, grad_vs], 1),
            (gv_all * vs_seq).sum(dim=(0, 1)).reshape(1, d),
            (gvs_all * vs_prev).sum(dim=(0, 1)).reshape(1, d))


# --------------------------------------------------------------------------
# leg 6 -- the claim that makes this arm free: the forward is the adopted arm's
# --------------------------------------------------------------------------

@pytest.mark.cuda
@requires_fused
@pytest.mark.parametrize("shape", SHAPES)
def test_the_forward_is_the_adopted_arm_bitwise(shape):
    """Spikes and BOTH membranes bit-identical to `snn.twocomp`, at `== 0.0`.

    This is the leg the arm's whole cost argument rests on. `detach()` changes no
    number, so anything but exact equality here means the two modules' forwards
    have drifted -- which is possible in principle even though this module imports
    `twocomp_forward_kernel` rather than re-declaring it, because the SAVED
    tensors and the stacking are written out in both files.

    Asserted over the full shape and regime sweep rather than once, because a
    drift that only shows at one sequence length is exactly the kind that survives
    a single spot check.
    """
    B, L, d = shape
    for beta_f, beta_s, w, thr in REGIMES:
        cur, v0, ww, bb = _inputs(B, L, d, seed=B * 977 + L * 31 + d,
                                  thr=thr, beta_s=beta_s, w=w)
        s_a, v_a = twocomp_scan(cur, v0, ww, bb, beta_f, thr, ALPHA, fused=True)
        s_d, v_d = twocomp_detach_scan(cur, v0, ww, bb, beta_f, thr, ALPHA,
                                       fused=True)
        assert bool((s_a == s_d).all()), (
            f"beta_f={beta_f} beta_s~{beta_s} w~{w} thr={thr}: "
            f"{int((s_a != s_d).sum())} spikes differ from the ADOPTED arm"
        )
        assert float((v_a - v_d).abs().max()) == 0.0, (beta_f, beta_s, w, thr)
        rate = float(s_d.mean())
        assert 0.02 < rate < 0.95, (beta_f, beta_s, w, thr, rate)


@pytest.mark.cuda
@requires_fused
def test_it_is_literally_the_same_forward_kernel():
    """Not equal source text -- the same cached callable, asserted by identity.

    `snn.twocomp_detach` imports `twocomp_forward_kernel` rather than declaring a
    forward source of its own. That is what makes leg 6 true by construction
    instead of by coincidence, and it is what stops a future edit to one arm's
    forward from silently applying to only one of them.

    Also asserted: this module declares no forward source at all. If someone later
    adds one "for symmetry", the identity above becomes a claim about two strings
    that happen to match today, and this test says so.
    """
    import snn.twocomp as tc
    import snn.twocomp_detach as tcd

    assert tcd.twocomp_forward_kernel is tc.twocomp_forward_kernel
    assert tcd.twocomp_forward_kernel() is tc.twocomp_forward_kernel()
    assert not hasattr(tcd, "twocomp_detach_forward_source"), (
        "this module must not declare a forward source; the arm's whole "
        "inference-cost claim is that the forward is snn.twocomp's, unmodified"
    )
    # ... and the backward text is genuinely different, or the arm is a no-op.
    from snn.twocomp import twocomp_backward_source
    assert tcd.twocomp_detach_backward_source() != twocomp_backward_source()


# --------------------------------------------------------------------------
# leg 7 -- the bound, at the kernel boundary, with a control that violates it
# --------------------------------------------------------------------------

def _chain_factor(kernel, v_pre, w, beta_s, beta_f, thr, alpha, vs=None):
    """`beta_f * dv`, read straight out of a backward kernel.

    Seeding the kernel with `grad_vf_next = 1`, `grad_vs_next = 0`,
    `grad_spike = 0` makes `gf = dv` exactly, so the returned `grad_vf_prev` IS
    the per-step multiplier of the homogeneous reverse recurrence -- the quantity
    `EXP_016` §2.1 defines and `EXP_017` §2 bounds.

    Read at the kernel boundary on the intermediate itself, which is
    `CONTRIBUTING.md` §5's rule: a numerical contract is only guarded where the
    quantity it constrains is actually observed. Measuring this through a whole
    scan would fold `dv` together with the injections and the sign cancellations
    and bound nothing.
    """
    ones = torch.ones_like(v_pre)
    zeros = torch.zeros_like(v_pre)
    if vs is None:      # detached kernel: six tensors, no `vs`
        out = kernel(zeros, ones, zeros, v_pre, w, beta_s,
                     beta_f=beta_f, thr=thr, alpha=alpha)
    else:               # adopted kernel: seven tensors
        out = kernel(zeros, ones, zeros, v_pre, vs, w, beta_s,
                     beta_f=beta_f, thr=thr, alpha=alpha)
    return out[1]       # grad_vf_prev


@pytest.mark.cuda
@requires_fused
def test_the_chain_factor_is_bounded_by_beta_f():
    """`max |beta_f * dv| <= MAX_CHAIN_FACTOR`, over a membrane sweep that
    deliberately includes the states `EXP_016` measured on the dying run.

    The derivation says `dv = 1 - s` is in {0, 1} for EVERY finite membrane, every
    `w`, every `vs` and every width -- so the sweep below is not looking for the
    bound, it is looking for a reason the kernel might not implement it. `v_pre`
    spans 1e-8 to 1e6 in both signs and `w*vs` reaches 400, past the 356.6 that
    `EXP_016` §9.4 recorded at layer 0 on the batch that killed the run.

    The realised set is asserted to be exactly {0, beta_f}, which is stronger than
    the inequality and is what the closed form actually says.
    """
    beta_f, thr, alpha = 0.5, 1.0, 2.0
    mags = torch.tensor([0.0, 1e-8, 1e-3, 0.5, 0.999999, 1.0, 1.000001,
                         2.0, 10.0, 400.0, 1e6], dtype=torch.float32)
    v_pre = torch.cat([mags, -mags, thr + mags, thr - mags]).cuda().reshape(1, -1)
    d = v_pre.shape[1]
    kernel = twocomp_detach_backward_kernel()

    for w_c in (0.0, 0.1, -0.3, 5.0, -400.0):
        for beta_s_c in (0.05, 0.5, 0.95, 0.995):
            w = torch.full((1, d), w_c, device="cuda")
            bs = torch.full((1, d), beta_s_c, device="cuda")
            g = _chain_factor(kernel, v_pre, w, bs, beta_f, thr, alpha)
            assert float(g.abs().max()) <= MAX_CHAIN_FACTOR, (
                f"w={w_c} beta_s={beta_s_c}: max|beta_f*dv| = "
                f"{float(g.abs().max()):.9f} > {MAX_CHAIN_FACTOR}"
            )
            realised = set(round(x, 7) for x in g.flatten().tolist())
            assert realised <= {0.0, beta_f}, (
                f"w={w_c} beta_s={beta_s_c}: dv is not in {{0, 1}}; realised "
                f"beta_f*dv values {sorted(realised)[:8]}"
            )
            assert 0.0 in realised and beta_f in realised, (
                "the sweep did not exercise both branches of the reset -- the "
                "bound would then hold vacuously"
            )


@pytest.mark.cuda
@requires_fused
def test_the_backward_threshold_comparison_is_inclusive():
    """`v_pre == thr` must take the FIRED branch of `dv`, at the kernel boundary.

    **ADDED BECAUSE A MUTATION ESCAPED.** The 2026-08-11 campaign ran D09 --
    "`>` instead of `>=` in the BACKWARD kernel only" -- and this gate passed,
    58/59. The analogous mutation on the adopted arm (T15) has always been
    caught, so the harness was working and this file had a specific hole.

    The hole is instructive and is the reason this test is separate from
    `test_the_chain_factor_is_bounded_by_beta_f` rather than folded into it.
    **The bound cannot see this mutation**: under `>=` the site at `x == 0` gives
    `sh = 1` and `g = 0`; under `>` it gives `sh = 0` and `g = beta_f`. *Both
    values are inside `{0, beta_f}`*, so every assertion about the bound holds
    exactly as before while the derivative is wrong. A bound is a statement about
    a set, and an off-by-one in a comparison moves a point *within* that set.

    That is `CONTRIBUTING.md` §5's rule arriving from a new direction: a numerical
    contract is only guarded where the quantity it constrains is actually
    observed, and "the quantity" here is not `|g|` but *which branch* `g` came
    from. Random inputs cannot supply it either -- exact equality has measure
    zero -- so the site is constructed.

    Asserted at the kernel boundary rather than through a scan, because that is
    where the comparison lives and a scan would fold it together with the forward
    kernel's own separate copy of the same comparison.
    """
    beta_f, thr, alpha = 0.5, 1.0, 2.0
    d = 16
    kernel = twocomp_detach_backward_kernel()
    for w_c in (0.0, 0.25, -3.0):
        w = torch.full((1, d), w_c, device="cuda")
        bs = torch.full((1, d), 0.9, device="cuda")

        at = torch.full((1, d), thr, device="cuda")          # x == 0 exactly
        g_at = _chain_factor(kernel, at, w, bs, beta_f, thr, alpha)
        assert float(g_at.abs().max()) == 0.0, (
            f"w={w_c}: at v_pre == thr the backward did not take the fired "
            f"branch -- dv = {float(g_at.abs().max()) / beta_f}, expected 0. "
            "The comparison is `>` where it must be `>=`."
        )

        # ... and the neighbouring representable value below thr must NOT fire,
        # or the assertion above would pass for a kernel that fires everywhere.
        below = torch.nextafter(at, torch.full_like(at, -1.0))
        g_below = _chain_factor(kernel, below, w, bs, beta_f, thr, alpha)
        assert float(g_below.min()) == beta_f, (
            f"w={w_c}: one ulp below thr the backward took the fired branch; "
            "the comparison is `>=` where it must be `>` on that side"
        )


@pytest.mark.cuda
@requires_fused
def test_threshold_comparison_is_inclusive_end_to_end():
    """The same contract through the whole scan, forward and backward.

    The adopted arm's gate carries this leg and it is mirrored here rather than
    inherited: the forward and backward kernels each hold their **own** copy of
    the comparison (`vmix >= thr` in the forward, `x >= T(0)` in the backward),
    and only the forward one is visible in the spike pattern.

    With `v0 = 0` and `w = 0` the first step gives `v_pre = cur = thr` exactly,
    which is the one input that distinguishes `>` from `>=` and which no random
    draw will ever produce.
    """
    d = 3
    cur = torch.zeros(2, 4, d, device="cuda")
    cur[:, 0] = THR
    v0 = torch.zeros(2, 2 * d, device="cuda")
    ww = torch.zeros(1, d, device="cuda")
    bb = torch.full((1, d), 0.95, device="cuda")

    s_f, _ = twocomp_detach_scan(cur, v0, ww, bb, 0.5, THR, ALPHA, fused=True)
    s_e, _ = twocomp_detach_scan_eager(cur, v0, ww, bb, 0.5, THR, ALPHA)
    assert bool((s_f[:, 0] == 1.0).all()), "v == thr did not fire"
    assert bool((s_f == s_e).all())

    gs = torch.ones(2, 4, d, device="cuda")
    gv = torch.ones(2, 2 * d, device="cuda")
    gf = _grads(cur, v0, ww, bb, 0.5, THR, gs, gv, fused=True)
    ge = _grads(cur, v0, ww, bb, 0.5, THR, gs, gv, fused=False)
    for name, a, b in zip(("grad_cur", "grad_v0", "grad_w", "grad_beta_s"), gf, ge):
        assert torch.allclose(a, b, rtol=1e-4, atol=1e-6), (
            f"{name} disagrees at v == thr, max|d|={float((a - b).abs().max()):.3e}"
        )


@pytest.mark.cuda
@requires_fused
def test_the_adopted_arm_violates_the_same_bound_on_the_same_inputs():
    """The control, and it is what makes the test above mean anything.

    A bound that no reachable input could have violated is not evidence. Handed
    the identical membranes, `snn.twocomp`'s kernel is measured here exceeding 1
    -- so the assertion above is discriminating between two kernels rather than
    restating an arithmetic tautology.

    The crossover `EXP_017` §2 derives is `|w_c * vs| ~ 1`; the row below sits far
    past it, at the magnitude `EXP_016` §9.1 recorded at `d = 1481`.
    """
    beta_f, thr, alpha = 0.5, 1.0, 2.0
    d = 64
    # v_pre at the surrogate's peak (sgd = 1), where dv = -vf is largest.
    v_pre = torch.full((1, d), thr, device="cuda")
    vs = torch.full((1, d), 20.0, device="cuda")
    w = torch.full((1, d), -10.0, device="cuda")     # w*vs = -200
    bs = torch.full((1, d), 0.95, device="cuda")

    g_adopted = _chain_factor(twocomp_backward_kernel(), v_pre, w, bs,
                              beta_f, thr, alpha, vs=vs)
    g_detach = _chain_factor(twocomp_detach_backward_kernel(), v_pre, w, bs,
                             beta_f, thr, alpha)

    assert float(g_adopted.abs().max()) > 1.0, (
        "the adopted arm did not expand on an input chosen to make it expand; "
        "the control leg is not exercising what it claims"
    )
    assert float(g_detach.abs().max()) <= MAX_CHAIN_FACTOR
    # ~100x on this input, and unbounded in |w*vs| by construction.
    assert float(g_adopted.abs().max()) > 50 * float(g_detach.abs().max())


# --------------------------------------------------------------------------
# leg 8 -- nothing died
# --------------------------------------------------------------------------

@pytest.mark.cuda
@requires_fused
def test_both_per_channel_gradients_stay_live():
    """Removing terms from a backward can silently freeze a parameter.

    `EXP_004`'s §7.2 screen caught exactly this once -- an initialisation that
    nested the baseline exactly and was an exact saddle, which would have trained
    a frozen parameter in every seed with nothing in any log to say so. This arm
    deletes `- grad_vf_next * vf` from `dL/dv`, and `dL/dw` is built FROM `dL/dv`,
    so the question "is `w` still reachable?" is a real one and is answered here
    rather than assumed.

    It is reachable because `dL/dv = sgd * grad_spike` survives: the model still
    learns through the spike. Asserted at the committed initialisation
    (`w_init = 0.1`, `beta_slow = 0.95`) as well as over the sweep, because the
    init is the only point every run actually passes through.
    """
    B, L, d = 4, 64, 32
    for beta_f, beta_s, w, thr in REGIMES + [(0.5, 0.95, 0.1, 1.0)]:
        cur, v0, ww, bb = _inputs(B, L, d, seed=int(1000 * thr) + 7,
                                  thr=thr, beta_s=beta_s, w=w)
        gs, gv = _cotangents(B, L, d, seed=21)
        gv.zero_()          # gradient reaches the parameters ONLY through spikes
        _gc, _gv0, gw, gbs = _grads(cur, v0, ww, bb, beta_f, thr, gs, gv,
                                    fused=True)
        assert float(gw.abs().max()) > 1e-8, (
            f"grad_w is dead at beta_f={beta_f} w~{w} thr={thr}: the detached "
            "reset froze the mix"
        )
        assert float(gbs.abs().max()) > 1e-8, (
            f"grad_beta_s is dead at beta_f={beta_f} beta_s~{beta_s} thr={thr}"
        )
        # every channel, not just the max: one live channel would satisfy a
        # max-based check while the rest of the layer sat frozen.
        assert int((gw.abs() > 0).sum()) == d, (beta_f, beta_s, w, thr)
        assert int((gbs.abs() > 0).sum()) == d, (beta_f, beta_s, w, thr)


@pytest.mark.cuda
@requires_fused
def test_the_spike_pathway_is_untouched():
    """`dL/dv` must equal the adopted arm's `sgd * grad_spike` term exactly.

    The arm's claim is that it removes the RESET pathway and nothing else. That is
    checkable: with `grad_vf_next = 0` the adopted kernel's `gv` reduces to
    `sgd * grad_spike`, which is what this kernel computes unconditionally. If the
    two disagree, the arm has changed the surrogate as well as the reset and its
    whole framing is wrong.
    """
    beta_f, thr, alpha = 0.5, 1.0, 2.0
    d = 128
    g = torch.Generator(device="cpu").manual_seed(90210)
    v_pre = (torch.randn(1, d, generator=g) * 2.0).cuda()
    vs = (torch.randn(1, d, generator=g) * 30.0).cuda()
    w = (torch.randn(1, d, generator=g) * 3.0).cuda()
    bs = torch.full((1, d), 0.9, device="cuda")
    gspike = torch.randn(1, d, generator=g).cuda()
    zeros = torch.zeros_like(v_pre)

    a = twocomp_backward_kernel()(gspike, zeros, zeros, v_pre, vs, w, bs,
                                  beta_f=beta_f, thr=thr, alpha=alpha)
    dtch = twocomp_detach_backward_kernel()(gspike, zeros, zeros, v_pre, w, bs,
                                            beta_f=beta_f, thr=thr, alpha=alpha)
    for i, name in enumerate(("grad_cur", "grad_vf_prev", "grad_vs_prev",
                              "gv_elem", "gvs_elem")):
        assert float((a[i] - dtch[i]).abs().max()) == 0.0, (
            f"{name} differs with grad_vf_next = 0, so the two arms disagree "
            f"about something other than the reset: "
            f"max|d| = {float((a[i] - dtch[i]).abs().max()):.3e}"
        )


# --------------------------------------------------------------------------
# leg 1 -- fused vs eager
# --------------------------------------------------------------------------

@pytest.mark.cuda
@requires_fused
@pytest.mark.parametrize("shape", SHAPES)
def test_backward(shape):
    """All four gradients within rtol=1e-4, atol=1e-6, over the regime sweep.

    Both scan outputs get a random cotangent so the `grad_spike * sgd` and
    `grad_vf_next * dv` terms are both live and cannot cancel each other's errors.
    `v0` requires grad even though training feeds detached zeros, because carried
    state would need that path and an untested path is a trap.
    """
    B, L, d = shape
    for beta_f, beta_s, w, thr in REGIMES:
        cur, v0, ww, bb = _inputs(B, L, d, seed=B * 811 + L * 17 + d,
                                  thr=thr, beta_s=beta_s, w=w)
        gs, gv = _cotangents(B, L, d, seed=B + L + d)
        gf = _grads(cur, v0, ww, bb, beta_f, thr, gs, gv, fused=True)
        ge = _grads(cur, v0, ww, bb, beta_f, thr, gs, gv, fused=False)
        for name, a, b in zip(("grad_cur", "grad_v0", "grad_w", "grad_beta_s"),
                              gf, ge):
            assert torch.allclose(a, b, rtol=1e-4, atol=1e-6), (
                f"{name} beta_f={beta_f} beta_s~{beta_s} w~{w} thr={thr} "
                f"max|d|={float((a - b).abs().max()):.3e}"
            )
            # a zero gradient would satisfy allclose against a zero reference
            assert float(b.abs().max()) > 1e-8, (name, beta_f, beta_s, w, thr)


@pytest.mark.cuda
@requires_fused
def test_backward_matches_spec_transcription():
    """Fused backward vs the recursion transcribed in this file from §2.

    This is the leg that survives `twocomp_detach.py`'s kernel and its eager
    reference agreeing on the same wrong derivation, because the recursion above
    was written from the equations rather than from either of them.
    """
    B, L, d = 4, 96, 32
    for beta_f, beta_s, w, thr in REGIMES:
        cur, v0, ww, bb = _inputs(B, L, d, seed=17, thr=thr, beta_s=beta_s, w=w)
        gs, gv = _cotangents(B, L, d, seed=5)
        gf = _grads(cur, v0, ww, bb, beta_f, thr, gs, gv, fused=True)
        _s, _v, v_pre, vs_seq = _spec_forward(cur, v0, ww, bb, beta_f, thr)
        gr = _spec_backward(v_pre, vs_seq, v0[:, d:], gs, gv,
                            ww, bb, beta_f, thr, ALPHA)
        for name, a, b in zip(("grad_cur", "grad_v0", "grad_w", "grad_beta_s"),
                              gf, gr):
            assert torch.allclose(a, b, rtol=1e-4, atol=1e-6), (
                f"{name} beta_f={beta_f} thr={thr} "
                f"max|d|={float((a - b).abs().max()):.3e}"
            )


@pytest.mark.cuda
@requires_fused
def test_the_two_arms_backwards_actually_differ():
    """Guards against a gate that passes because the arm changed nothing.

    Every leg above compares this arm against a reference for this arm. If the
    detach had silently not taken effect -- a plausible failure, since `detach()`
    is one token and the fused path routes through a different kernel entirely --
    all of them would still pass. So the difference is asserted directly, on both
    dispatch paths, in a regime where the reset term is large.
    """
    B, L, d = 4, 96, 32
    cur, v0, ww, bb = _inputs(B, L, d, seed=404, thr=1.0, beta_s=0.99, w=-0.3)
    gs, gv = _cotangents(B, L, d, seed=808)
    for fused in (True, False):
        c = cur.detach().clone().requires_grad_(True)
        s, vf = twocomp_scan(c, v0, ww, bb, 0.5, THR, ALPHA, fused=fused)
        ((s * gs).sum() + (vf * gv).sum()).backward()
        c2 = cur.detach().clone().requires_grad_(True)
        s2, vf2 = twocomp_detach_scan(c2, v0, ww, bb, 0.5, THR, ALPHA, fused=fused)
        ((s2 * gs).sum() + (vf2 * gv).sum()).backward()
        assert bool((s == s2).all()), f"fused={fused}: the FORWARDS differ"
        rel = float((c.grad - c2.grad).abs().max()) / float(c.grad.abs().max())
        assert rel > 1e-3, (
            f"fused={fused}: grad_cur is within {rel:.2e} of the adopted arm's -- "
            "the detached reset did not take effect"
        )


@pytest.mark.cuda
@requires_fused
def test_backward_against_float64_reference():
    """fp32 fused vs the eager reference in float64.

    Legs 1 and 2 both run in fp32, so they cannot distinguish "correct" from
    "identically rounded". The spike pattern is asserted unchanged by the
    precision switch first: if a membrane landed within fp32 rounding of the
    threshold the trajectories would legitimately diverge, and that would be a
    property of the input rather than a bug.
    """
    B, L, d, beta_f = 4, 96, 32, 0.9
    cur, v0, ww, bb = _inputs(B, L, d, seed=29, beta_s=0.95, w=0.3)
    gs, gv = _cotangents(B, L, d, seed=13)

    s32, _ = twocomp_detach_scan(cur, v0, ww, bb, beta_f, THR, ALPHA, fused=True)
    s64, _ = twocomp_detach_scan_eager(cur.double(), v0.double(), ww.double(),
                                       bb.double(), beta_f, THR, ALPHA)
    assert bool((s32.double() == s64).all()), "precision changed the spike pattern"

    gf = _grads(cur, v0, ww, bb, beta_f, THR, gs, gv, fused=True)
    gd = _grads(cur.double(), v0.double(), ww.double(), bb.double(),
                beta_f, THR, gs.double(), gv.double(), fused=False)
    for name, a, b in zip(("grad_cur", "grad_v0", "grad_w", "grad_beta_s"), gf, gd):
        assert torch.allclose(a.double(), b, rtol=1e-4, atol=1e-5), (
            f"{name} max|d|={float((a.double() - b).abs().max()):.3e}"
        )


def test_gradcheck_slow_compartment_is_exactly_linear():
    """A genuine finite-difference gradcheck of the slow pole and `dL/dbeta_s`.

    `gradcheck` compares against a numerical Jacobian, and the surrogate gradient
    is deliberately *not* the true derivative of a hard threshold -- so
    gradchecking the spike output would fail by construction and prove nothing.
    But the slow compartment is reset-shielded, so `vs_L = beta_s^L*vs_0 + sum_t
    beta_s^(L-1-t)*cur_t` exactly: the surrogate never enters that path, and a
    finite difference is meaningful.

    That path is untouched by this arm -- the detach removes terms from `dL/dv`,
    not from the slow recursion -- so this leg is expected to pass identically to
    the adopted arm's, and it is run because "expected to" is not "checked".
    """
    torch.manual_seed(7)
    B, L, d = 2, 12, 3
    cur = torch.randn(B, L, d, dtype=torch.float64, requires_grad=True)
    v0 = torch.randn(B, 2 * d, dtype=torch.float64, requires_grad=True)
    beta_s = (0.5 + 0.4 * torch.rand(1, d, dtype=torch.float64)).requires_grad_(True)
    w = torch.full((1, d), 0.3, dtype=torch.float64)

    def f(c, v, bs):
        return twocomp_detach_scan_eager(c, v, w, bs, 0.9, THR, ALPHA)[1][:, d:]

    assert torch.autograd.gradcheck(f, (cur, v0, beta_s), eps=1e-6, atol=1e-9)


# --------------------------------------------------------------------------
# leg 5 -- nesting: at w = 0 this IS the mutation-tested LIF's "detached" mode
# --------------------------------------------------------------------------

@pytest.mark.cuda
@requires_fused
def test_w_zero_is_the_committed_detached_lif_scan():
    """At `w = 0` this scan must BE `lif_scan(..., reset="detached")`.

    The adopted arm's counterpart of this leg nests the `"hard"` mode; this one
    nests `"detached"`, which is the same relationship one level over. Both LIF
    modes have survived the committed 25/25 mutation campaign -- M04 and M05 are
    precisely the two mutations that swap their reset derivatives -- so tying the
    new kernel to `"detached"` bit-for-bit inherits that evidence rather than
    resting only on references written alongside the thing they check.

    Both directions are asserted at `== 0.0` rather than at a tolerance:
      * forward -- spikes bit-identical and the fast half of the membrane exact;
      * backward -- `grad_cur` and the fast half of `grad_v0` identical to the
        LIF's, and `grad_beta_s` **exactly zero**, the saddle EXP_004's T0
        predicts, which this arm does not move.

    Bit-exactness in the backward is not automatic: `twocomp_detach.py` writes
    `gf` in `kernels.py`'s grouping deliberately so that this can be asserted at
    zero. An equality that holds to a tolerance could hide a small systematic
    error; one that holds bitwise cannot.
    """
    B, L, d = 4, 128, 32
    for beta_f, thr in ((0.5, 1.0), (0.9, 1.0), (0.5, 2.0), (0.95, 0.5)):
        cur, v02, _ww, bb = _inputs(B, L, d, seed=613, thr=thr, beta_s=0.95)
        w0 = torch.zeros(1, d, device="cuda")
        v0f = v02[:, :d].contiguous()

        s_tc, v_tc = twocomp_detach_scan(cur, v02, w0, bb, beta_f, thr, ALPHA,
                                         fused=True)
        s_lif, v_lif = lif_scan(cur, v0f, beta_f, thr, ALPHA, "detached",
                                fused=True)
        assert bool((s_tc == s_lif).all()), (beta_f, thr)
        assert float((v_tc[:, :d] - v_lif).abs().max()) == 0.0, (beta_f, thr)

        gs, gv2 = _cotangents(B, L, d, seed=97)
        gv2[:, d:] = 0.0   # no cotangent on the slow half: it has no counterpart
        gc_tc, gv0_tc, gw_tc, gbs_tc = _grads(
            cur, v02, w0, bb, beta_f, thr, gs, gv2, fused=True)

        c = cur.detach().clone().requires_grad_(True)
        v = v0f.detach().clone().requires_grad_(True)
        s, vfin = lif_scan(c, v, beta_f, thr, ALPHA, "detached", fused=True)
        ((s * gs).sum() + (vfin * gv2[:, :d]).sum()).backward()

        assert float((gc_tc - c.grad).abs().max()) == 0.0, (
            f"beta_f={beta_f} thr={thr} grad_cur is not bit-identical to the "
            f"committed detached LIF's: "
            f"max|d|={float((gc_tc - c.grad).abs().max()):.3e}")
        assert float((gv0_tc[:, :d] - v.grad).abs().max()) == 0.0, (beta_f, thr)
        assert float(gbs_tc.abs().max()) == 0.0, (
            f"beta_s gradient is not exactly zero at w=0: "
            f"{float(gbs_tc.abs().max())}")
        assert float(gw_tc.abs().max()) > 0.0


@pytest.mark.cuda
@requires_fused
def test_w_zero_does_NOT_match_the_hard_reset_lif():
    """...and the nesting is against `"detached"` specifically.

    Without this, `test_w_zero_is_the_committed_detached_lif_scan` would still
    pass if the two LIF modes happened to agree on these inputs, and the leg would
    be asserting "it is a LIF" rather than "it is the detached one".
    """
    B, L, d, beta_f, thr = 4, 128, 32, 0.5, 1.0
    cur, v02, _ww, bb = _inputs(B, L, d, seed=613, thr=thr, beta_s=0.95)
    w0 = torch.zeros(1, d, device="cuda")
    gs, gv2 = _cotangents(B, L, d, seed=97)
    gv2[:, d:] = 0.0
    gc_tc, _, _, _ = _grads(cur, v02, w0, bb, beta_f, thr, gs, gv2, fused=True)

    c = cur.detach().clone().requires_grad_(True)
    v = v02[:, :d].contiguous().detach().clone().requires_grad_(True)
    s, vfin = lif_scan(c, v, beta_f, thr, ALPHA, "hard", fused=True)
    ((s * gs).sum() + (vfin * gv2[:, :d]).sum()).backward()
    assert float((gc_tc - c.grad).abs().max()) > 0.0, (
        "at w=0 this scan is bit-identical to the HARD-reset LIF too, so the "
        "nesting leg above does not distinguish the two reset derivatives"
    )


# --------------------------------------------------------------------------
# the inference-model identity
# --------------------------------------------------------------------------

@pytest.mark.cuda
@requires_fused
def test_the_inference_model_is_the_adopted_arm_bitwise():
    """A `twocomp_detach` checkpoint loaded as `twocomp` gives identical logits.

    This is the arm's cost argument at the level of the whole model rather than
    the scan: same parameter names, same shapes, same count, same forward kernel,
    so `as_twocomp_state_dict` is a plain copy and `load_state_dict` is strict.

    `EXP_008` W1 and `EXP_011` C4 both had to report a fold-in RESIDUAL here
    (1.9e-03 and 1.05e-06 bpc) because their folds rescaled a `Linear` and the
    GEMM re-accumulated. This arm folds nothing, so the assertion is `== 0.0` and
    decision #6's tolerance question does not arise for it at all.
    """
    from snn.config import Config
    from snn.model import TwoCompartmentCharLM, build_model, count_params

    cfg = Config(vocab_size=205, d_model=64, n_layers=2, arch="twocomp_detach",
                 device="cuda", seed=3)
    detach = build_model(cfg)
    adopted = TwoCompartmentCharLM(
        vocab_size=205, d_model=64, n_layers=2, beta=cfg.beta,
        threshold=cfg.threshold, reset="hard",
        surrogate_alpha=cfg.surrogate_alpha, dtype=cfg.dtype, fused=cfg.fused,
    ).to("cuda")

    sd = detach.as_twocomp_state_dict()
    missing, unexpected = adopted.load_state_dict(sd, strict=True), None
    assert not missing.missing_keys and not missing.unexpected_keys, missing
    assert count_params(detach) == count_params(adopted)

    idx = torch.randint(0, 205, (4, 128), device="cuda")
    detach.eval()
    adopted.eval()
    with torch.no_grad():
        la, _, _ = detach(idx)
        lb, _, _ = adopted(idx)
    assert float((la - lb).abs().max()) == 0.0, (
        f"the inference models differ by {float((la - lb).abs().max()):.3e}; "
        "the arm's zero-inference-cost claim rests on this being exactly 0"
    )
    assert unexpected is None


@pytest.mark.cuda
@requires_fused
def test_parameter_count_is_identical_to_the_adopted_arm():
    """Parameter-IDENTICAL, not parameter-matched.

    `EXP_015`'s comparison at `d = 1481` was parameter-matched to within 0.18%.
    This arm is at 0.00% against `twocomp` by construction, which is what lets
    `EXP_017`'s width leg be read directly against `EXP_015`'s table.
    """
    from snn.config import Config
    from snn.model import build_model, count_params, twocomp_param_count

    for d in (64, 512, 1481):
        a = build_model(Config(vocab_size=205, d_model=d, n_layers=2,
                               arch="twocomp", device="cpu"))
        b = build_model(Config(vocab_size=205, d_model=d, n_layers=2,
                               arch="twocomp_detach", device="cpu"))
        assert count_params(a) == count_params(b) == twocomp_param_count(205, d, 2)
        assert sorted(a.state_dict()) == sorted(b.state_dict())


def test_build_model_rejects_a_reset_that_never_ran():
    """`config.json` must describe the neuron that actually ran.

    The forward reset of every two-compartment arm is hard. Accepting
    `reset="detached"` on `arch="twocomp_detach"` would look natural and would
    write a config recording a forward rule the kernel does not implement.
    """
    from snn.config import Config
    from snn.model import build_model

    with pytest.raises(NotImplementedError, match="twocomp_detach"):
        build_model(Config(vocab_size=205, d_model=32, n_layers=1,
                           arch="twocomp_detach", reset="detached", device="cpu"))


# --------------------------------------------------------------------------
# capture, validation, and the double-backward guard
# --------------------------------------------------------------------------

@pytest.mark.cuda
@requires_fused
def test_survives_cuda_graph_capture():
    """The whole scan -- forward and backward -- inside a CUDAGraph.

    The trainer captures the full step, so a candidate neuron that cannot be
    captured is a candidate that runs at ~1/11 throughput. Proven on the object
    the trainer will actually capture, not on a bare kernel.
    """
    B, L, d, beta_f = 4, 32, 16, 0.9
    cur, v0, ww, bb = _inputs(B, L, d, seed=53, beta_s=0.95, w=0.3)
    cur = cur.detach().requires_grad_(True)
    v0 = v0.detach().requires_grad_(True)
    ww = ww.detach().requires_grad_(True)
    bb = bb.detach().requires_grad_(True)
    gs, gv = _cotangents(B, L, d, seed=31)

    def step():
        for p in (cur, v0, ww, bb):
            if p.grad is not None:
                p.grad.zero_()
        s, vf = twocomp_detach_scan(cur, v0, ww, bb, beta_f, THR, ALPHA, fused=True)
        ((s * gs).sum() + (vf * gv).sum()).backward()

    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(3):
            step()
    torch.cuda.current_stream().wait_stream(side)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        step()

    # Replay against data the graph has never seen -- what would catch a captured
    # kernel reading a stale address instead of the static buffer (risk R5).
    fresh, _, _, _ = _inputs(B, L, d, seed=97, beta_s=0.95, w=0.3)
    cur.data.copy_(fresh)
    graph.replay()
    torch.cuda.synchronize()
    got = (cur.grad.clone(), v0.grad.clone(), ww.grad.clone(), bb.grad.clone())

    ref = _grads(cur.detach(), v0.detach(), ww.detach(), bb.detach(),
                 beta_f, THR, gs, gv, fused=False)
    for name, a, b in zip(("grad_cur", "grad_v0", "grad_w", "grad_beta_s"), got, ref):
        assert torch.allclose(a, b, rtol=1e-4, atol=1e-6), (
            f"{name} after replay max|d|={float((a - b).abs().max()):.3e}")


def test_no_fused_path_runs_on_cpu():
    """`--no-fused` must work with no CUDA at all; it is the reference path."""
    cur = torch.randn(2, 16, 8)
    v0 = torch.zeros(2, 16)
    w = torch.full((1, 8), 0.2)
    bs = torch.full((1, 8), 0.9)
    s, v = twocomp_detach_scan(cur, v0, w, bs, 0.9, THR, ALPHA, fused=False)
    assert s.shape == (2, 16, 8) and v.shape == (2, 16)
    assert bool(((s == 0.0) | (s == 1.0)).all())


def test_input_validation():
    """The shape and dtype contracts, which are `snn.twocomp`'s by import."""
    cur = torch.randn(2, 8, 4)
    v0 = torch.zeros(2, 8)
    w = torch.full((1, 4), 0.1)
    bs = torch.full((1, 4), 0.9)
    with pytest.raises(ValueError):
        twocomp_detach_scan(cur[0], v0, w, bs, 0.9, THR, ALPHA, fused=False)
    with pytest.raises(ValueError):
        # [B, d] state, not [B, 2d] -- the shape a copy-paste from the LIF gives
        twocomp_detach_scan(cur, torch.zeros(2, 4), w, bs, 0.9, THR, ALPHA,
                            fused=False)
    with pytest.raises(ValueError):
        twocomp_detach_scan(cur, v0, torch.full((4,), 0.1), bs, 0.9, THR, ALPHA,
                            fused=False)
    with pytest.raises(TypeError):
        # spec §2 / B8: the membrane is fp32 whatever cfg.dtype says
        twocomp_detach_scan(cur.bfloat16(), v0.bfloat16(), w.bfloat16(),
                            bs.bfloat16(), 0.9, THR, ALPHA, fused=False)


@pytest.mark.cuda
@requires_fused
def test_fused_apply_rejects_non_fp32():
    """`.apply` must enforce fp32 itself, not lean on the dispatcher having done it.

    The forward kernel's membrane recursions use float-only intrinsics. Handed
    float64 it would narrow to fp32 and cast back, returning a double that is only
    fp32-accurate, with no error anywhere. Silence is the problem.
    """
    cur, v0, w, bs = _inputs(2, 8, 4, seed=67, dtype=torch.float64)
    with pytest.raises(TypeError):
        FusedTwoCompDetachScan.apply(cur, v0, w, bs, 0.9, THR, ALPHA)


@pytest.mark.cuda
@requires_fused
def test_double_backward_raises_rather_than_lying():
    """`create_graph=True` through the fused scan must fail loudly."""
    def first_order(fused):
        torch.manual_seed(0)
        c = torch.randn(2, 8, 4, device="cuda", requires_grad=True)
        v0 = torch.zeros(2, 8, device="cuda", requires_grad=True)
        w = torch.full((1, 4), 0.3, device="cuda")
        bs = torch.full((1, 4), 0.9, device="cuda")
        s, vf = twocomp_detach_scan(c, v0, w, bs, 0.9, THR, ALPHA, fused=fused)
        loss = (s * s).sum() + (vf * vf).sum() + (c * c * c).sum()
        return c, torch.autograd.grad(loss, c, create_graph=True)[0]

    c_e, g_e = first_order(fused=False)
    (gg_e,) = torch.autograd.grad(g_e.sum(), c_e)
    assert float(gg_e.abs().sum()) > 0.0

    with pytest.raises(RuntimeError, match="double backward"):
        first_order(fused=True)


@pytest.mark.cuda
@requires_fused
def test_one_kernel_per_timestep():
    """`EXP_002`'s finding, as a regression check at the baseline shape.

    The systems column must not discriminate between this arm and the adopted one:
    the forward is the same kernel, and the backward is one kernel per timestep
    with one fewer input. A figure above 1.05 would mean the split into a second
    module cost a launch somewhere.
    """
    from snn.metrics import count_cuda_kernels

    B, L, d = 128, 256, 512
    cur = torch.randn(B, L, d, device="cuda") * 0.5
    v0 = torch.zeros(B, 2 * d, device="cuda")
    w = torch.full((1, d), 0.1, device="cuda")
    bs = torch.full((1, d), 0.95, device="cuda")

    def scan():
        with torch.no_grad():
            twocomp_detach_scan(cur, v0, w, bs, 0.5, THR, ALPHA, fused=True)

    scan()  # force the NVRTC compile out of the measured region
    counts = count_cuda_kernels(scan)
    per_step = counts["total"] / L
    assert per_step < 1.05, f"{per_step:.4f} kernels per timestep"
