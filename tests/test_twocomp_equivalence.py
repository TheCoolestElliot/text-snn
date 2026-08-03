"""R10 gate for the two-compartment scan (EXP_004, candidate #1).

Risk R10 is the project's worst bug class: a hand-written jiterator backward that
is *silently* wrong. No crash, no NaN, no broken loss curve -- the model trains,
the bpc looks plausible, and every conclusion drawn from it is worthless. This
file is the two-compartment neuron's version of `test_neuron_equivalence.py`, and
EXP_004 entry condition J1 is that it passes before a single training step runs.

The gate has five legs, the first four mirroring the baseline's:

  1. fused vs the eager autograd reference -- spikes bit-identical, membrane
     exact, gradients (including the two NEW per-channel gradients) to
     rtol 1e-4 / atol 1e-6, over a regime sweep;
  2. fused vs a backward recursion transcribed *in this file* from EXP_004 §2's
     equations, so a shared misreading by `twocomp.py`'s forward and backward
     cannot pass;
  3. fp32 fused vs the eager reference in float64;
  4. `torch.autograd.gradcheck` in the one regime where the surrogate never
     enters -- the slow compartment, which is exactly linear because it is
     reset-shielded. That leg checks `dL/dbeta_s`, one of the two new gradients,
     against a genuine numerical Jacobian.

  5. **the nesting leg, which the baseline has no counterpart for.** At `w = 0`
     this neuron is *defined* to be the Phase-2 LIF, and that LIF's kernel has
     already survived a committed 25/25 mutation campaign. So the new scan is
     checked bit-for-bit against `snn.neuron.lif_scan` -- forward and backward --
     which ties it to an artefact that is already trusted rather than only to a
     reference written alongside it.

WHY THE PARAMETERS ARE SWEPT, AND SWEPT PER CHANNEL. `test_neuron_equivalence.py`
records that three mutations survived a version of the baseline gate that used
only the baseline hyperparameters, because `thr = 1.0` and `beta = 0.5` make wrong
kernels accidentally right. This neuron adds two traps of its own:

  * `beta_f == beta_s` makes the two compartments indistinguishable, so swapping
    them, or feeding one compartment's decay to the other, is invisible. Every
    regime below keeps them apart.
  * a per-channel parameter that is CONSTANT across channels cannot detect the
    failure `EXP_002` §8.4 actually hit -- jiterator binding a `[1, d]` tensor
    into a scalar slot, or broadcasting element 0 everywhere. It compiles, runs,
    and produces plausible spikes. So `w` and `beta_s` are drawn per channel and
    `test_per_channel_parameters_are_not_broadcast_from_one_element` recovers them
    from the kernel's own output.
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
from snn.twocomp import (  # noqa: E402
    FusedTwoCompartmentScan,
    twocomp_scan,
    twocomp_scan_eager,
)

requires_fused = pytest.mark.skipif(
    not (torch.cuda.is_available() and jiterator_available()),
    reason="needs a CUDA device with a working jiterator",
)

ALPHA = 2.0
THR = 1.0

SHAPES = [(4, 8, 16), (3, 64, 32), (2, 128, 48), (5, 256, 16)]

# (beta_f, beta_s_centre, w_centre, thr). beta_f = 0.5 is the baseline and is a
# power of two -- the one value at which the fast decay multiply is exact -- so
# every other row exists to stop that accident from hiding anything. beta_s is
# never equal to beta_f. `w` is negative in one row because the mix is deliberately
# unconstrained in `model.py`, and a kernel that silently assumed w >= 0 would pass
# a sweep that only ever handed it positive values.
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

    Firing rate matters: at a small enough scale nothing crosses the threshold,
    the reset branch never fires, and the gate would pass while checking nothing.
    The scale is larger than the baseline gate's 0.7 because the slow compartment
    spends long stretches below zero -- it is unreset, so a negative excursion
    persists for ~1/(1-beta_s) steps and suppresses firing. `test_forward` asserts
    the realised rate rather than assuming it.

    `beta_s` and `w` are returned as per-channel `[1, d]` tensors with real spread
    around the requested centre, never as constants. See the module docstring.
    """
    g = torch.Generator(device="cpu").manual_seed(seed)
    cur = torch.randn(B, L, d, generator=g, dtype=torch.float64) * (scale * thr)
    v0 = torch.randn(B, 2 * d, generator=g, dtype=torch.float64) * (0.3 * thr)
    # spread, then clamp into (0, 1) -- a decay outside that range is not a decay
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
        s, vf = twocomp_scan(c, v, ww, bb, beta_f, thr, ALPHA, fused=True)
    else:
        s, vf = twocomp_scan_eager(c, v, ww, bb, beta_f, thr, ALPHA)
    ((s * gs).sum() + (vf * gv).sum()).backward()
    return c.grad, v.grad, ww.grad, bb.grad


def _cotangents(B, L, d, seed, device="cuda", dtype=torch.float32):
    g = torch.Generator(device="cpu").manual_seed(seed)
    gs = torch.randn(B, L, d, generator=g, dtype=torch.float64)
    gv = torch.randn(B, 2 * d, generator=g, dtype=torch.float64)
    return gs.to(device=device, dtype=dtype), gv.to(device=device, dtype=dtype)


# --------------------------------------------------------------------------
# leg 2: forward and backward transcribed from EXP_004 §2, independent of src/
# --------------------------------------------------------------------------

def _spec_forward(cur, v0, w, beta_s, beta_f, thr):
    """EXP_004 §2, transcribed by hand. Returns (spikes, v_final, v_pre, vs_seq)."""
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
    """The adjoint recursion, derived here from §2's forward rather than read
    off `twocomp.py`.

    With v = vf + w*vs, s = H(v - thr), vf_out = vf*(1 - s), and vs shielded:

        dL/dv       = sg * (grad_spike - grad_vf_next * vf)
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
    vf_pre = v_pre - w * vs_seq
    vs_prev = torch.cat([vs0.unsqueeze(1), vs_seq[:, :-1]], dim=1)

    grad_vf, grad_vs = grad_v_final[:, :d], grad_v_final[:, d:]
    grad_cur, gv_all, gvs_all = [None] * L, [None] * L, [None] * L
    for t in range(L - 1, -1, -1):
        gv = sg[:, t] * (grad_spikes[:, t] - grad_vf * vf_pre[:, t])
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
# leg 1 -- forward
# --------------------------------------------------------------------------

@pytest.mark.cuda
@requires_fused
@pytest.mark.parametrize("shape", SHAPES)
def test_forward(shape):
    """Spikes bit-identical, membrane exact, and the neuron actually fires."""
    B, L, d = shape
    for beta_f, beta_s, w, thr in REGIMES:
        cur, v0, ww, bb = _inputs(B, L, d, seed=B * 977 + L * 31 + d,
                                  thr=thr, beta_s=beta_s, w=w)
        s_f, v_f = twocomp_scan(cur, v0, ww, bb, beta_f, thr, ALPHA, fused=True)
        s_e, v_e = twocomp_scan_eager(cur, v0, ww, bb, beta_f, thr, ALPHA)
        assert bool((s_f == s_e).all()), (
            f"beta_f={beta_f} beta_s~{beta_s} w~{w} thr={thr}: "
            f"{int((s_f != s_e).sum())} spikes differ"
        )
        assert float((v_f - v_e).abs().max()) == 0.0, (beta_f, beta_s, w, thr)
        rate = float(s_e.mean())
        assert 0.02 < rate < 0.95, (beta_f, beta_s, w, thr, rate)


@pytest.mark.cuda
@requires_fused
def test_returned_membrane_is_bit_identical():
    """Zero difference, not a small one, on the state the scan returns.

    The fused kernel writes both membrane recursions with `__fmul_rn`/`__fadd_rn`.
    The eager path performs each as a separate CUDA kernel and therefore rounds
    twice; an FMA rounds once. A one-ulp drift is invisible in any tolerance-based
    check and flips any spike whose membrane lands within an ulp of the threshold,
    which is exactly the R10 failure mode.

    Asserted separately from `test_forward` so that if it ever fails while
    `test_forward` still passes, the message is "the rounding contract broke"
    rather than "the gate failed".

    NOTE WHAT THIS DOES NOT COVER, because it took a mutation escape to notice:
    the returned state is `(vf_after_reset, vs)`, and **neither depends on the
    mix**. `vf + w*vs` reaches only `v_pre`, which the scan does not return, and
    the spike, which changes only where the membrane lands within an ulp of the
    threshold. The mix's own contraction is covered by
    `test_the_mix_is_bit_identical_at_the_kernel_boundary` below.
    """
    for beta_f, beta_s, w, thr in REGIMES:
        cur, v0, ww, bb = _inputs(4, 128, 32, seed=71, thr=thr, beta_s=beta_s, w=w)
        _, v_f = twocomp_scan(cur, v0, ww, bb, beta_f, thr, ALPHA, fused=True)
        _, v_e = twocomp_scan_eager(cur, v0, ww, bb, beta_f, thr, ALPHA)
        assert float((v_f - v_e).abs().max()) == 0.0, (beta_f, beta_s, w, thr)


@pytest.mark.cuda
@requires_fused
def test_the_mix_is_bit_identical_at_the_kernel_boundary():
    """The third suppressed contraction, asserted on the quantity that carries it.

    ADDED BECAUSE A MUTATION ESCAPED. The 2026-08-03 campaign ran T05 -- "let
    NVRTC contract the MIX into an FMA" -- and the gate passed, 46/47. The
    analogous mutation on the LIF's own membrane (M18) has always been caught, so
    the harness was working and this file had a specific hole.

    Measured on this stack at B=64, d=512: the contracted mix moves `v_pre` on
    **5866 of 32768 elements**, by up to 4.8e-07 -- and flips **zero** spikes.
    That is the whole mechanism. `v_pre` is the only place the mix appears, the
    scan does not return it, and a spike flips only where `|v_pre - thr|` is
    within an ulp. Whether any element lands there is luck, so a spike-pattern
    check is a coin flip rather than a gate, and it came up tails.

    So the contract is asserted where it lives: one step of the real forward
    kernel against a plain-torch statement of the same three roundings. The
    suppressed kernel matches it on 32768 of 32768 elements.

    The general lesson, which outlives this kernel: **a numerical contract is only
    guarded where the quantity it constrains is actually observed.** An
    intermediate that no assertion reads is unguarded no matter how many tests
    surround it.
    """
    from snn.twocomp import twocomp_forward_kernel

    fn = twocomp_forward_kernel()
    B, d = 64, 512
    for beta_f, beta_s, w, thr in REGIMES:
        g = torch.Generator(device="cpu").manual_seed(hash((beta_f, thr)) % 2**31)
        vf = torch.randn(B, d, generator=g).cuda()
        vs = torch.randn(B, d, generator=g).cuda()
        cur = torch.randn(B, d, generator=g).cuda()
        ww = (w + 0.25 * torch.randn(1, d, generator=g)).cuda()
        bb = (beta_s + 0.04 * torch.randn(1, d, generator=g)).clamp(0.05, 0.995).cuda()

        _vf, _vs, _s, v_pre = fn(vf, vs, cur, ww, bb, beta_f=beta_f, thr=thr)
        # Each line is one multiply and one add, as two separate CUDA kernels,
        # rounding twice -- exactly what the kernel's intrinsics must reproduce.
        vfp = vf * beta_f + cur
        vsp = vs * bb + cur
        vmix = vfp + ww * vsp
        assert torch.equal(v_pre, vmix), (
            f"beta_f={beta_f} thr={thr}: v_pre differs from the twice-rounded "
            f"reference on {int((v_pre != vmix).sum())}/{v_pre.numel()} elements, "
            f"max|d|={float((v_pre - vmix).abs().max()):.3e} -- the mix contracted"
        )


@pytest.mark.cuda
@requires_fused
def test_threshold_comparison_is_inclusive():
    """`v == thr` must fire, in both paths, and the backward must agree there.

    Random inputs never test this -- exact equality has measure zero. With
    v0 = 0 and w = 0 the first step gives v = cur = thr exactly. A kernel written
    with `>` passes every other test in this file and fails only here.

    The gradient half is not redundant: the forward and backward kernels each
    carry their own copy of the comparison (`vmix >= thr` and `x >= T(0)`), and
    only the forward one is visible in the spike pattern. The backward's `(1 - sh)`
    factor flips wholesale at equality.
    """
    d = 3
    cur = torch.zeros(2, 4, d, device="cuda")
    cur[:, 0] = THR
    v0 = torch.zeros(2, 2 * d, device="cuda")
    ww = torch.zeros(1, d, device="cuda")
    bb = torch.full((1, d), 0.95, device="cuda")

    s_f, _ = twocomp_scan(cur, v0, ww, bb, 0.5, THR, ALPHA, fused=True)
    s_e, _ = twocomp_scan_eager(cur, v0, ww, bb, 0.5, THR, ALPHA)
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
def test_the_slow_compartment_is_actually_reset_shielded():
    """The candidate's whole premise, asserted directly rather than assumed.

    An unreset compartment is exactly linear, so its final value must equal
    `beta_s^L * vs_0 + sum_t beta_s^(L-1-t) * cur_t` no matter how much the neuron
    fired. That closed form is computed here without any reference to the scan, so
    a kernel that quietly reset the slow pole -- which is what `EXP_002`'s N2
    prototype did, and which would make the second timescale retain
    `beta_s*(1-p)` per step instead of `beta_s` -- fails here and nowhere else.

    The firing rate is asserted too: if the neuron never fired, resetting and not
    resetting would be the same thing and this test would prove nothing.
    """
    B, L, d, beta_f = 4, 64, 16, 0.5
    cur, v0, ww, bb = _inputs(B, L, d, seed=1009, beta_s=0.9, w=0.4)
    s, v_final = twocomp_scan(cur, v0, ww, bb, beta_f, THR, ALPHA, fused=True)
    assert 0.05 < float(s.mean()) < 0.95, float(s.mean())

    vs = v0[:, d:] * bb ** L
    for t in range(L):
        vs = vs + cur[:, t] * bb ** (L - 1 - t)
    assert torch.allclose(v_final[:, d:], vs, rtol=1e-4, atol=1e-5), float(
        (v_final[:, d:] - vs).abs().max()
    )
    # ... and the FAST half must NOT match its own unreset closed form, or the
    # hard reset is not happening either and the test above is vacuous.
    vfl = v0[:, :d] * beta_f ** L
    for t in range(L):
        vfl = vfl + cur[:, t] * beta_f ** (L - 1 - t)
    assert not torch.allclose(v_final[:, :d], vfl, rtol=1e-2, atol=1e-3)


@pytest.mark.cuda
@requires_fused
def test_per_channel_parameters_are_not_broadcast_from_one_element():
    """`EXP_002` §8.4's failure mode, made detectable.

    jiterator binds every tensor argument first and then the scalars, so a
    declaration order that interleaves them silently binds a `[1, d]` tensor into
    a scalar slot. It compiles, it runs, and it produces plausible spikes. Nothing
    about the kernel failing reveals it.

    With v0 = 0 and one timestep, `v_pre = cur + w_c*cur`, so `w_c` is recoverable
    exactly; and after a second step the slow half of the state is
    `beta_s_c*cur_0 + cur_1`, so `beta_s_c` is too. Both are checked against the
    tensors that went in, per channel, and the channels are asserted distinct so
    that a kernel broadcasting element 0 everywhere cannot pass.
    """
    B, d = 2, 8
    g = torch.Generator(device="cpu").manual_seed(4242)
    ww = (0.2 + 0.6 * torch.rand(1, d, generator=g)).cuda()
    bb = (0.3 + 0.6 * torch.rand(1, d, generator=g)).cuda()
    assert len(set(ww[0].tolist())) == d and len(set(bb[0].tolist())) == d

    cur = torch.randn(B, 2, d, generator=g).cuda()
    v0 = torch.zeros(B, 2 * d, device="cuda")
    # thr far above anything reachable, so no spike and no reset perturbs the
    # algebra this test inverts.
    _s, v_final = twocomp_scan(cur, v0, ww, bb, 0.5, 1e6, ALPHA, fused=True)

    vs_expected = bb * cur[:, 0] + cur[:, 1]
    assert torch.allclose(v_final[:, d:], vs_expected, rtol=1e-5, atol=1e-6)
    recovered_beta = (v_final[:, d:] - cur[:, 1]) / cur[:, 0]
    assert torch.allclose(recovered_beta, bb.expand(B, d), rtol=1e-4, atol=1e-5)

    v0b = torch.zeros(B, 2 * d, device="cuda")
    _s1, _ = twocomp_scan(cur[:, :1], v0b, ww, bb, 0.5, 1e6, ALPHA, fused=True)
    # v_pre at t=0 is cur_0*(1 + w_c); recover w_c from the eager path's own
    # membrane, which the fused path has already been proven to match exactly.
    _, veq = twocomp_scan_eager(cur[:, :1], v0b, ww, bb, 0.5, 1e6, ALPHA)
    recovered_w = (veq[:, :d] + ww * veq[:, d:] - cur[:, 0]) / cur[:, 0]
    assert torch.allclose(recovered_w, ww.expand(B, d), rtol=1e-4, atol=1e-5)


# --------------------------------------------------------------------------
# leg 1 -- backward
# --------------------------------------------------------------------------

@pytest.mark.cuda
@requires_fused
@pytest.mark.parametrize("shape", SHAPES)
def test_backward(shape):
    """All four gradients within rtol=1e-4, atol=1e-6, over the regime sweep.

    Both scan outputs get a random cotangent so the `grad_spike * sg` and
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

    This is the leg that survives `twocomp.py`'s forward and backward agreeing on
    the same wrong derivation, because the recursion above was written from the
    equations rather than from either of them.
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
def test_forward_matches_spec_transcription():
    """Fused forward vs §2 transcribed in this file."""
    for beta_f, beta_s, w, thr in REGIMES:
        cur, v0, ww, bb = _inputs(4, 96, 32, seed=11, thr=thr, beta_s=beta_s, w=w)
        s_f, v_f = twocomp_scan(cur, v0, ww, bb, beta_f, thr, ALPHA, fused=True)
        s_r, v_r, _, _ = _spec_forward(cur, v0, ww, bb, beta_f, thr)
        assert bool((s_f == s_r).all()), (beta_f, beta_s, w, thr)
        assert float((v_f - v_r).abs().max()) < 1e-5, (beta_f, beta_s, w, thr)


# --------------------------------------------------------------------------
# leg 3 -- float64 reference
# --------------------------------------------------------------------------

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

    s32, _ = twocomp_scan(cur, v0, ww, bb, beta_f, THR, ALPHA, fused=True)
    s64, _ = twocomp_scan_eager(cur.double(), v0.double(), ww.double(),
                                bb.double(), beta_f, THR, ALPHA)
    assert bool((s32.double() == s64).all()), "precision changed the spike pattern"

    gf = _grads(cur, v0, ww, bb, beta_f, THR, gs, gv, fused=True)
    gd = _grads(cur.double(), v0.double(), ww.double(), bb.double(),
                beta_f, THR, gs.double(), gv.double(), fused=False)
    for name, a, b in zip(("grad_cur", "grad_v0", "grad_w", "grad_beta_s"), gf, gd):
        assert torch.allclose(a.double(), b, rtol=1e-4, atol=1e-5), (
            f"{name} max|d|={float((a.double() - b).abs().max()):.3e}"
        )


# --------------------------------------------------------------------------
# leg 4 -- gradcheck, in the one regime where it is valid
# --------------------------------------------------------------------------

def test_gradcheck_slow_compartment_is_exactly_linear():
    """A genuine finite-difference gradcheck of the slow pole and `dL/dbeta_s`.

    `gradcheck` compares against a numerical Jacobian, and the surrogate gradient
    is deliberately *not* the true derivative of a hard threshold -- so
    gradchecking the spike output would fail by construction and prove nothing.
    But the slow compartment is reset-shielded, so `vs_L = beta_s^L*vs_0 + sum_t
    beta_s^(L-1-t)*cur_t` exactly: the surrogate never enters that path at all, and
    a finite difference is meaningful.

    That makes this the one leg that checks `dL/dbeta_s` -- one of the two NEW
    gradients this neuron introduces -- against something other than another
    hand-derived expression. What it does NOT cover is `dL/dw`, which only reaches
    the loss through the threshold and is therefore surrogate-mediated by
    construction; that gradient is covered by legs 1, 2, 3 and 5 and by the
    mutation campaign.
    """
    torch.manual_seed(7)
    B, L, d = 2, 12, 3
    cur = torch.randn(B, L, d, dtype=torch.float64, requires_grad=True)
    v0 = torch.randn(B, 2 * d, dtype=torch.float64, requires_grad=True)
    beta_s = (0.5 + 0.4 * torch.rand(1, d, dtype=torch.float64)).requires_grad_(True)
    w = torch.full((1, d), 0.3, dtype=torch.float64)

    def f(c, v, bs):
        return twocomp_scan_eager(c, v, w, bs, 0.9, THR, ALPHA)[1][:, d:]

    assert torch.autograd.gradcheck(f, (cur, v0, beta_s), eps=1e-6, atol=1e-9)


# --------------------------------------------------------------------------
# leg 5 -- nesting: at w = 0 this IS the mutation-tested Phase-2 LIF
# --------------------------------------------------------------------------

@pytest.mark.cuda
@requires_fused
def test_w_zero_is_the_committed_lif_scan():
    """At `w = 0` the two-compartment scan must BE `snn.neuron.lif_scan`.

    This is the strongest leg in the file and it has no counterpart in the
    baseline's gate, because the baseline had nothing already-trusted to be
    checked against. The LIF kernel has survived a committed 25/25 mutation
    campaign; tying the new kernel to it bit-for-bit inherits that evidence
    instead of resting only on references written alongside the thing they check.

    Both directions are asserted, and both at `== 0.0` rather than at a tolerance:
      * forward -- spikes bit-identical and the fast half of the membrane exact;
      * backward -- `grad_cur` and the fast half of `grad_v0` identical to the
        LIF's, and `grad_beta_s` **exactly zero**, which is the saddle EXP_004's
        T0 predicts and §2.2 derives.

    Bit-exactness in the backward is not automatic and did not hold on the first
    draft. The adjoint written in its natural order --
    `grad_vf_next*(1-s) + dL/dv` -- is algebraically identical to the LIF kernel's
    `grad_v_next*dv + grad_spike*sg` and agreed with it only to ~2 ulp (measured
    2.4e-07 on gradients of order 1.35), because the two group the same three
    terms differently and float addition does not associate. `twocomp.py` now
    writes the regrouped form deliberately so that this leg can be asserted at
    zero. That is the point of the leg: an equality that holds to a tolerance
    could hide a small systematic error, and an equality that holds bitwise
    cannot.
    """
    B, L, d = 4, 128, 32
    for beta_f, thr in ((0.5, 1.0), (0.9, 1.0), (0.5, 2.0), (0.95, 0.5)):
        cur, v02, _ww, bb = _inputs(B, L, d, seed=613, thr=thr, beta_s=0.95)
        w0 = torch.zeros(1, d, device="cuda")
        v0f = v02[:, :d].contiguous()

        s_tc, v_tc = twocomp_scan(cur, v02, w0, bb, beta_f, thr, ALPHA, fused=True)
        s_lif, v_lif = lif_scan(cur, v0f, beta_f, thr, ALPHA, "hard", fused=True)
        assert bool((s_tc == s_lif).all()), (beta_f, thr)
        assert float((v_tc[:, :d] - v_lif).abs().max()) == 0.0, (beta_f, thr)

        gs, gv2 = _cotangents(B, L, d, seed=97)
        gv2[:, d:] = 0.0   # no cotangent on the slow half: it has no counterpart
        gc_tc, gv0_tc, gw_tc, gbs_tc = _grads(
            cur, v02, w0, bb, beta_f, thr, gs, gv2, fused=True)

        c = cur.detach().clone().requires_grad_(True)
        v = v0f.detach().clone().requires_grad_(True)
        s, vfin = lif_scan(c, v, beta_f, thr, ALPHA, "hard", fused=True)
        ((s * gs).sum() + (vfin * gv2[:, :d]).sum()).backward()

        assert float((gc_tc - c.grad).abs().max()) == 0.0, (
            f"beta_f={beta_f} thr={thr} grad_cur is not bit-identical to the "
            f"committed LIF's: max|d|={float((gc_tc - c.grad).abs().max()):.3e}")
        assert float((gv0_tc[:, :d] - v.grad).abs().max()) == 0.0, (beta_f, thr)
        # T0, asserted rather than argued: with w = 0 the slow pole carries no
        # gradient at all, so `beta_s` sits at an exact saddle.
        assert float(gbs_tc.abs().max()) == 0.0, (
            f"beta_s gradient is not exactly zero at w=0: {float(gbs_tc.abs().max())}")
        # ... while the mix itself IS reachable, which is what makes the fallback
        # init in EXP_004 §2.2 a fix rather than a workaround.
        assert float(gw_tc.abs().max()) > 0.0


@pytest.mark.cuda
@requires_fused
def test_parameters_are_not_vacuous():
    """Guards against a gate that passes because nothing depends on anything.

    If changing `w` or `beta_s` did not change the spikes, every equivalence test
    above would be comparing two implementations of the same constant.
    """
    B, L, d = 4, 64, 32
    cur, v0, ww, bb = _inputs(B, L, d, seed=3, beta_s=0.95, w=0.3)
    base, _ = twocomp_scan(cur, v0, ww, bb, 0.5, THR, ALPHA, fused=True)
    assert bool(((base == 0.0) | (base == 1.0)).all()), "emission is not binary (I1)"

    other_w, _ = twocomp_scan(cur, v0, ww * 2.0, bb, 0.5, THR, ALPHA, fused=True)
    other_b, _ = twocomp_scan(cur, v0, ww, bb * 0.5, 0.5, THR, ALPHA, fused=True)
    assert not bool((base == other_w).all()), "w does not affect the spikes"
    assert not bool((base == other_b).all()), "beta_s does not affect the spikes"


# --------------------------------------------------------------------------
# capture, validation, and the double-backward guard
# --------------------------------------------------------------------------

@pytest.mark.cuda
@requires_fused
def test_survives_cuda_graph_capture():
    """The whole scan -- forward and backward -- inside a CUDAGraph.

    The trainer captures the full step, so a candidate neuron that cannot be
    captured is a candidate that runs at ~1/11 throughput. Proven on the object
    the trainer will actually capture, not on a bare kernel. EXP_004 failure mode
    J8.
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
        s, vf = twocomp_scan(cur, v0, ww, bb, beta_f, THR, ALPHA, fused=True)
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
    s, v = twocomp_scan(cur, v0, w, bs, 0.9, THR, ALPHA, fused=False)
    assert s.shape == (2, 16, 8) and v.shape == (2, 16)
    assert bool(((s == 0.0) | (s == 1.0)).all())


def test_input_validation():
    cur = torch.randn(2, 8, 4)
    v0 = torch.zeros(2, 8)
    w = torch.full((1, 4), 0.1)
    bs = torch.full((1, 4), 0.9)
    with pytest.raises(ValueError):
        twocomp_scan(cur[0], v0, w, bs, 0.9, THR, ALPHA, fused=False)
    with pytest.raises(ValueError):
        # [B, d] state, not [B, 2d] -- the shape a copy-paste from the LIF gives
        twocomp_scan(cur, torch.zeros(2, 4), w, bs, 0.9, THR, ALPHA, fused=False)
    with pytest.raises(ValueError):
        twocomp_scan(cur, v0, torch.full((4,), 0.1), bs, 0.9, THR, ALPHA, fused=False)
    with pytest.raises(TypeError):
        # spec §2 / B8: the membrane is fp32 whatever cfg.dtype says
        twocomp_scan(cur.bfloat16(), v0.bfloat16(), w.bfloat16(), bs.bfloat16(),
                     0.9, THR, ALPHA, fused=False)


@pytest.mark.cuda
@requires_fused
def test_fused_apply_rejects_non_fp32():
    """`.apply` must enforce fp32 itself, not lean on `twocomp_scan` having done it.

    The kernel's membrane recursions use float-only intrinsics. Handed float64 it
    would narrow to fp32 and cast back, returning a double that is only
    fp32-accurate, with no error anywhere. Silence is the problem, not the
    precision.
    """
    cur, v0, w, bs = _inputs(2, 8, 4, seed=67, dtype=torch.float64)
    with pytest.raises(TypeError):
        FusedTwoCompartmentScan.apply(cur, v0, w, bs, 0.9, THR, ALPHA)


@pytest.mark.cuda
@requires_fused
def test_double_backward_raises_rather_than_lying():
    """`create_graph=True` through the fused scan must fail loudly.

    jiterator outputs carry no autograd history, so the fused backward is not
    itself differentiable and the second-order term would simply be missing --
    silently, whenever some other term in the loss also depends on `cur`. The
    eager path IS twice differentiable and is checked here to prove the reference
    is the capable one.
    """
    def first_order(fused):
        torch.manual_seed(0)
        c = torch.randn(2, 8, 4, device="cuda", requires_grad=True)
        v0 = torch.zeros(2, 8, device="cuda", requires_grad=True)
        w = torch.full((1, 4), 0.3, device="cuda")
        bs = torch.full((1, 4), 0.9, device="cuda")
        s, vf = twocomp_scan(c, v0, w, bs, 0.9, THR, ALPHA, fused=fused)
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
    """EXP_002's finding, as a regression check at the baseline shape.

    `EXP_002` priced this neuron's shape at 1.0156 kernels/timestep and concluded
    the systems column does not discriminate. That conclusion is load-bearing for
    the whole Phase-4 ranking, so the realised kernel is held to the same 1.05 bar
    `tests/test_kernel_count.py` holds the baseline to.
    """
    from snn.metrics import count_cuda_kernels

    B, L, d = 128, 256, 512
    cur = torch.randn(B, L, d, device="cuda") * 0.5
    v0 = torch.zeros(B, 2 * d, device="cuda")
    w = torch.full((1, d), 0.1, device="cuda")
    bs = torch.full((1, d), 0.95, device="cuda")

    def scan():
        with torch.no_grad():
            twocomp_scan(cur, v0, w, bs, 0.5, THR, ALPHA, fused=True)

    scan()  # force the NVRTC compile out of the measured region
    counts = count_cuda_kernels(scan)
    per_step = counts["total"] / L
    assert per_step < 1.05, f"{per_step:.4f} kernels per timestep"
