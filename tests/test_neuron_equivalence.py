"""R10 gate: the fused LIF scan must be indistinguishable from the eager one.

Risk R10 (01_reconnaissance.md §7) is the most dangerous class of bug in this
project: a hand-written jiterator backward that is *silently* wrong. There is no
crash, no NaN, no obviously broken loss curve -- the model trains, the bpc looks
plausible, and every conclusion drawn from it is worthless. The only defence is
an assertion that the fast path and the slow path compute the same function, run
on every change to either.

The gate has four independent legs, in increasing order of paranoia:

  1. fused vs the eager autograd reference -- spikes bit-identical, membrane to
     1e-5, gradients to rtol 1e-4 / atol 1e-6, for all four reset modes;
  2. fused vs an implementation transcribed *in this file* directly from the spec
     §6.1/§6.2 tables, so a shared misreading of the spec by `kernels.py` and
     `neuron.py` cannot pass;
  3. fused fp32 vs the eager reference run in float64, so the check is against
     something more accurate than the thing being checked;
  4. `torch.autograd.gradcheck` in the one regime where the true derivative and
     the surrogate coincide (reset="none", membrane output only, where the scan
     is exactly linear), plus a cross-check against snnTorch -- a third-party
     implementation nobody here wrote.

Legs 2-4 exist because leg 1 alone only proves the two implementations agree, not
that either is right.

WHY THE PARAMETERS ARE SWEPT RATHER THAN LEFT AT THEIR DEFAULTS. This file was
itself mutation-tested: eleven plausible mis-derivations were injected into the
kernel source and the gate was rerun. Three of them slipped through a version of
these tests that used only the baseline hyperparameters, and all three for the
same reason -- the baseline makes a wrong kernel accidentally right.
`thr = 1.0` makes `1 - thr*sg` and `1 - sg` the same expression, so dropping the
threshold factor from the soft-reset derivative was invisible. `beta = 0.5` is a
power of two, so `v_prev*beta` is exact and an FMA contraction in the forward
changes nothing. And `>` versus `>=` at the threshold never differs on random
floats, because equality has measure zero. Hence the threshold sweep, the beta
sweep, and `test_threshold_comparison_is_inclusive` below: each one closes a hole
that a mutation walked through.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest
import torch

# The repo keeps its package under src/ and expects a root conftest.py to insert
# it (spec §1). Repeated here so this file is runnable on its own; it is a no-op
# if the conftest already did it.
_SRC = str(Path(__file__).resolve().parents[1] / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from snn.kernels import RESET_CODES, RESET_MODES, jiterator_available  # noqa: E402
from snn.neuron import FusedLIFScan, lif_scan, lif_scan_eager  # noqa: E402
from snn.surrogate import atan_grad, atan_value  # noqa: E402

requires_fused = pytest.mark.skipif(
    not (torch.cuda.is_available() and jiterator_available()),
    reason="needs a CUDA device with a working jiterator",
)

ALPHA = 2.0
THR = 1.0  # spec §13 baseline; the sweeps below deliberately leave it

# L >= 64 in three of these so accumulated BPTT error is actually exercised;
# 256 is the baseline sequence length from spec §13.
SHAPES = [(4, 8, 16), (3, 64, 32), (2, 128, 48), (5, 256, 16)]

# (beta, thr). beta=0.5 is the baseline and is a power of two, so it is the one
# value at which the decay multiply is exact -- every other entry is here to stop
# that accident from hiding anything. thr != 1 separates `thr*sg` from `sg`.
REGIMES = [(0.5, 1.0), (0.9, 1.0), (0.9, 0.5), (0.5, 2.0), (0.95, 1.5)]


def _inputs(B, L, d, seed, thr=1.0, device="cuda", dtype=torch.float32, scale=0.7):
    """Currents scaled with the threshold so the firing rate stays sane.

    Firing rate matters for this test: at scale 0.05 nothing ever crosses the
    threshold, the reset rules become indistinguishable, and the test would pass
    while checking nothing. ~20-40% firing keeps every branch live, which is why
    `test_forward` asserts the rate as well as the values.
    """
    g = torch.Generator(device="cpu").manual_seed(seed)
    cur = torch.randn(B, L, d, generator=g, dtype=torch.float64) * (scale * thr)
    v0 = torch.randn(B, d, generator=g, dtype=torch.float64) * (0.3 * thr)
    return cur.to(device=device, dtype=dtype), v0.to(device=device, dtype=dtype)


def _fused_scan(cur, v0, beta, thr, alpha, reset):
    """Same signature as `lif_scan_eager`, so the two are interchangeable here."""
    return lif_scan(cur, v0, beta, thr, alpha, reset, fused=True)


def _grads(scan, cur, v0, beta, thr, reset, gs, gv):
    c = cur.detach().clone().requires_grad_(True)
    v = v0.detach().clone().requires_grad_(True)
    s, vf = scan(c, v, beta, thr, ALPHA, reset)
    ((s * gs).sum() + (vf * gv).sum()).backward()
    return c.grad, v.grad


# --------------------------------------------------------------------------
# leg 2: an implementation transcribed from the spec, independent of src/snn
# --------------------------------------------------------------------------

def _spec_forward(cur, v0, beta, thr, reset):
    """Spec §6.1, transcribed by hand. Returns (spikes, v_final, v_pre)."""
    L = cur.shape[1]
    v = v0
    spikes, v_pres = [], []
    for t in range(L):
        v_pre = v * beta + cur[:, t]
        s = (v_pre >= thr).to(v_pre.dtype)
        if reset == "soft":
            v = v_pre - thr * s
        elif reset in ("hard", "detached"):
            v = v_pre * (1 - s)
        elif reset == "none":
            v = v_pre
        else:
            raise AssertionError(reset)
        spikes.append(s)
        v_pres.append(v_pre)
    return (torch.stack(spikes, 1), v, torch.stack(v_pres, 1))


def _spec_backward(v_pre, grad_spikes, grad_v_final, beta, thr, alpha, reset):
    """Spec §6.2, transcribed by hand. Returns (grad_cur, grad_v0).

    Uses math.pi directly rather than `snn.surrogate`, so a wrong constant there
    would show up here rather than cancelling out.
    """
    x = v_pre - thr
    s = (x >= 0).to(v_pre.dtype)
    u = (math.pi / 2.0) * alpha * x
    sg = (alpha / 2.0) / (1.0 + u * u)
    if reset == "soft":
        dv = 1.0 - thr * sg
    elif reset == "hard":
        dv = (1.0 - s) - v_pre * sg
    elif reset == "detached":
        dv = 1.0 - s
    elif reset == "none":
        dv = torch.ones_like(v_pre)
    else:
        raise AssertionError(reset)

    L = v_pre.shape[1]
    grad_v = grad_v_final
    grad_cur = [None] * L
    for t in range(L - 1, -1, -1):
        g = grad_v * dv[:, t] + grad_spikes[:, t] * sg[:, t]
        grad_cur[t] = g
        grad_v = beta * g
    return torch.stack(grad_cur, 1), grad_v


# --------------------------------------------------------------------------
# premise: the eager reference's autograd derivative IS the analytic surrogate
# --------------------------------------------------------------------------

def test_surrogate_matches_analytic():
    """Eager autograd derivative equals `atan_grad` to 1e-6 (spec §12).

    Everything below compares the fused backward against the eager one. If the
    eager one were not exactly the analytic surrogate, all of it would be
    measuring the wrong quantity.
    """
    x = torch.linspace(-8.0, 8.0, 8001, requires_grad=True)
    (g,) = torch.autograd.grad(atan_value(x, ALPHA).sum(), x)
    assert torch.allclose(g, atan_grad(x.detach(), ALPHA), rtol=1e-5, atol=1e-6)


# --------------------------------------------------------------------------
# leg 1 -- forward
# --------------------------------------------------------------------------

@pytest.mark.cuda
@requires_fused
@pytest.mark.parametrize("reset", RESET_MODES)
@pytest.mark.parametrize("shape", SHAPES)
def test_forward(reset, shape):
    """Spikes bit-identical, membrane within 1e-5 (spec §12).

    "Bit-identical" is the real requirement and it is not pedantry: one differing
    spike changes what the next layer's GEMM sees, and under hard reset it
    changes the membrane trajectory from that timestep onward.
    """
    B, L, d = shape
    for beta, thr in REGIMES:
        cur, v0 = _inputs(B, L, d, seed=(B * 977 + L * 31 + d), thr=thr)
        s_f, v_f = lif_scan(cur, v0, beta, thr, ALPHA, reset, fused=True)
        s_e, v_e = lif_scan_eager(cur, v0, beta, thr, ALPHA, reset)

        assert bool((s_f == s_e).all()), (
            f"{reset} beta={beta} thr={thr}: "
            f"{int((s_f != s_e).sum())} spikes differ"
        )
        assert float((v_f - v_e).abs().max()) < 1e-5, (reset, beta, thr)
        # the test is only meaningful if the neuron actually fires
        assert 0.02 < float(s_e.mean()) < 0.95, (reset, beta, thr, float(s_e.mean()))


@pytest.mark.cuda
@requires_fused
@pytest.mark.parametrize("reset", RESET_MODES)
def test_membrane_is_bit_identical(reset):
    """Stronger than spec §12's 1e-5, and deliberately so.

    Every operation the fused kernel performs has an exact counterpart in the
    eager path -- provided NVRTC does not contract `v_prev*beta + cur` into an
    FMA, which rounds once where the eager path's two separate kernels round
    twice. `snn/kernels.py` suppresses that contraction with `__fmul_rn` /
    `__fadd_rn`, so the correct outcome is *zero* difference, not a small one.

    Asserted separately from `test_forward` on purpose: if this ever fails while
    `test_forward` still passes, the model is still within its stated tolerance
    and the message is "the rounding contract broke", not "the gate failed".
    With the contraction left in, this test fails at beta=0.9 and beta=0.95
    (measured drift up to 2.9e-06) while `test_forward` does not.
    """
    for beta, thr in REGIMES:
        cur, v0 = _inputs(4, 128, 32, seed=71, thr=thr)
        _, v_f = lif_scan(cur, v0, beta, thr, ALPHA, reset, fused=True)
        _, v_e = lif_scan_eager(cur, v0, beta, thr, ALPHA, reset)
        assert float((v_f - v_e).abs().max()) == 0.0, (reset, beta, thr)


@pytest.mark.cuda
@requires_fused
@pytest.mark.parametrize("reset", RESET_MODES)
def test_threshold_comparison_is_inclusive(reset):
    """`v_pre == thr` must fire, in both paths (spec §2: the comparison is `>=`).

    Random inputs never test this -- exact equality has measure zero -- so the
    input is constructed to land exactly on the threshold: with v0 = 0 the first
    step gives v_pre = 0*beta + thr. A kernel written with `>` passes every other
    test in this file and fails only here.

    THE GRADIENT IS CHECKED AS WELL, and that half is not redundant. The forward
    kernel and the backward kernel each carry their own copy of the comparison
    (`v >= thr` and `x >= T(0)` respectively), and only the forward one is
    visible in the spike pattern. Mutating *just* the backward comparison to `>`
    was measured to survive every other assertion in this file while moving
    `grad_cur` by 0.206 under hard and detached reset -- the two modes whose
    `dv_next/dv_pre` contains a bare `(1 - s)` factor that flips wholesale at
    equality. That is a 20% gradient error hiding behind a measure-zero event,
    which is the R10 failure mode exactly.
    """
    thr = 1.0
    cur = torch.zeros(2, 4, 3, device="cuda")
    cur[:, 0] = thr  # v_pre = thr exactly at t=0
    v0 = torch.zeros(2, 3, device="cuda")
    s_f, _ = lif_scan(cur, v0, 0.5, thr, ALPHA, reset, fused=True)
    s_e, _ = lif_scan_eager(cur, v0, 0.5, thr, ALPHA, reset)
    assert bool((s_f[:, 0] == 1.0).all()), f"{reset}: v_pre == thr did not fire"
    assert bool((s_f == s_e).all())

    gs = torch.ones(2, 4, 3, device="cuda")
    gv = torch.ones(2, 3, device="cuda")
    gc_f, gv_f = _grads(_fused_scan, cur, v0, 0.5, thr, reset, gs, gv)
    gc_e, gv_e = _grads(lif_scan_eager, cur, v0, 0.5, thr, reset, gs, gv)
    assert torch.allclose(gc_f, gc_e, rtol=1e-4, atol=1e-6), (
        f"{reset}: backward disagrees at v_pre == thr, "
        f"max|d|={float((gc_f - gc_e).abs().max()):.3e}"
    )
    assert torch.allclose(gv_f, gv_e, rtol=1e-4, atol=1e-6), reset


@pytest.mark.cuda
@requires_fused
@pytest.mark.parametrize("reset", RESET_MODES)
def test_forward_matches_spec_transcription(reset):
    """Fused forward vs the §6.1 table transcribed in this file."""
    for beta, thr in REGIMES:
        cur, v0 = _inputs(4, 96, 32, seed=11, thr=thr)
        s_f, v_f = lif_scan(cur, v0, beta, thr, ALPHA, reset, fused=True)
        s_r, v_r, _ = _spec_forward(cur, v0, beta, thr, reset)
        assert bool((s_f == s_r).all()), (reset, beta, thr)
        assert float((v_f - v_r).abs().max()) < 1e-5, (reset, beta, thr)


@pytest.mark.cuda
@requires_fused
def test_spikes_are_binary_and_reset_modes_differ():
    """Guards against a test suite that passes because nothing happens.

    If every reset mode produced the same trajectory the equivalence tests would
    be vacuous, so assert that they do not. "detached" is deliberately excluded
    from the difference check: it is *defined* to have the same forward as
    "hard" and to differ only in the backward.
    """
    cur, v0 = _inputs(4, 64, 32, seed=3)
    outs = {r: lif_scan(cur, v0, 0.9, THR, ALPHA, r, fused=True) for r in RESET_MODES}
    for r, (s, _) in outs.items():
        assert bool(((s == 0.0) | (s == 1.0)).all()), r
    assert bool((outs["hard"][0] == outs["detached"][0]).all())
    for a, b in (("hard", "soft"), ("hard", "none"), ("soft", "none")):
        assert not bool((outs[a][0] == outs[b][0]).all()), (a, b)


# --------------------------------------------------------------------------
# leg 1 -- backward
# --------------------------------------------------------------------------

@pytest.mark.cuda
@requires_fused
@pytest.mark.parametrize("reset", RESET_MODES)
@pytest.mark.parametrize("shape", SHAPES)
def test_backward(reset, shape):
    """grad_cur and grad_v0 within rtol=1e-4, atol=1e-6 (spec §12).

    `v0` requires grad here even though Phase 2 feeds detached zeros, because
    carried-state TBPTT would need that path and an untested path is a trap.
    Both outputs of the scan get a random cotangent so that the `grad_spike * sg`
    and `grad_v_next * dv` terms are both exercised and cannot cancel each
    other's errors.
    """
    B, L, d = shape
    torch.manual_seed(1234)
    for beta, thr in REGIMES:
        cur, v0 = _inputs(B, L, d, seed=(B * 811 + L * 17 + d), thr=thr)
        gs = torch.randn(B, L, d, device="cuda")
        gv = torch.randn(B, d, device="cuda")

        gc_f, gv_f = _grads(_fused_scan, cur, v0, beta, thr, reset, gs, gv)
        gc_e, gv_e = _grads(lif_scan_eager, cur, v0, beta, thr, reset, gs, gv)

        assert torch.allclose(gc_f, gc_e, rtol=1e-4, atol=1e-6), (
            f"{reset} beta={beta} thr={thr} grad_cur "
            f"max|d|={float((gc_f - gc_e).abs().max()):.3e}"
        )
        assert torch.allclose(gv_f, gv_e, rtol=1e-4, atol=1e-6), (
            f"{reset} beta={beta} thr={thr} grad_v0 "
            f"max|d|={float((gv_f - gv_e).abs().max()):.3e}"
        )
        # a zero gradient would satisfy allclose against a zero reference
        assert float(gc_e.abs().max()) > 1e-3
        assert float(gv_e.abs().max()) > 1e-8


@pytest.mark.cuda
@requires_fused
@pytest.mark.parametrize("reset", RESET_MODES)
def test_backward_matches_spec_transcription(reset):
    """Fused backward vs the §6.2 table transcribed in this file.

    This is the leg that would survive `neuron.py` and `kernels.py` agreeing on
    a wrong derivation, because the recursion here was written from the spec
    table rather than from either of them.
    """
    B, L, d = 4, 96, 32
    for beta, thr in REGIMES:
        cur, v0 = _inputs(B, L, d, seed=17, thr=thr)
        torch.manual_seed(5)
        gs = torch.randn(B, L, d, device="cuda")
        gv = torch.randn(B, d, device="cuda")

        gc_f, gv_f = _grads(_fused_scan, cur, v0, beta, thr, reset, gs, gv)
        _, _, v_pre = _spec_forward(cur, v0, beta, thr, reset)
        gc_r, gv_r = _spec_backward(v_pre, gs, gv, beta, thr, ALPHA, reset)

        assert torch.allclose(gc_f, gc_r, rtol=1e-4, atol=1e-6), (
            reset, beta, thr, float((gc_f - gc_r).abs().max())
        )
        assert torch.allclose(gv_f, gv_r, rtol=1e-4, atol=1e-6), (
            reset, beta, thr, float((gv_f - gv_r).abs().max())
        )


@pytest.mark.cuda
@requires_fused
def test_detached_reset_differs_from_hard_in_the_backward():
    """"detached" must drop the -v_pre*sg term, and only that term.

    Getting this wrong is easy and invisible: the forward is byte-for-byte the
    same as "hard", so every forward test passes either way. Checked as its own
    case rather than trusted to the parametrised sweep.
    """
    B, L, d, beta = 4, 64, 32, 0.9
    cur, v0 = _inputs(B, L, d, seed=23)
    torch.manual_seed(9)
    gs = torch.randn(B, L, d, device="cuda")
    gv = torch.randn(B, d, device="cuda")

    s_hard, v_hard = lif_scan(cur, v0, beta, THR, ALPHA, "hard", fused=True)
    s_det, v_det = lif_scan(cur, v0, beta, THR, ALPHA, "detached", fused=True)
    assert bool((s_hard == s_det).all()) and float((v_hard - v_det).abs().max()) == 0.0

    gc_hard, _ = _grads(_fused_scan, cur, v0, beta, THR, "hard", gs, gv)
    gc_det, _ = _grads(_fused_scan, cur, v0, beta, THR, "detached", gs, gv)
    assert not torch.allclose(gc_hard, gc_det, rtol=1e-3, atol=1e-5)


# --------------------------------------------------------------------------
# leg 3 -- float64 reference
# --------------------------------------------------------------------------

@pytest.mark.cuda
@requires_fused
@pytest.mark.parametrize("reset", RESET_MODES)
def test_backward_against_float64_reference(reset):
    """fp32 fused vs the eager reference in float64.

    Legs 1 and 2 both run in fp32, so they cannot distinguish "correct" from
    "identically rounded". Running the reference in float64 removes that excuse.
    The spike pattern is asserted to be unchanged by the precision switch first:
    if a v_pre landed within fp32 rounding of the threshold the two trajectories
    would legitimately diverge, and that would be a property of the input, not a
    bug in the kernel.
    """
    B, L, d, beta = 4, 96, 32, 0.9
    cur32, v032 = _inputs(B, L, d, seed=29)
    cur64, v064 = cur32.double(), v032.double()
    torch.manual_seed(13)
    gs = torch.randn(B, L, d, device="cuda")
    gv = torch.randn(B, d, device="cuda")

    gc_f, gv_f = _grads(_fused_scan, cur32, v032, beta, THR, reset, gs, gv)
    gc_d, gv_d = _grads(
        lif_scan_eager, cur64, v064, beta, THR, reset, gs.double(), gv.double()
    )

    s32, _ = lif_scan(cur32, v032, beta, THR, ALPHA, reset, fused=True)
    s64, _ = lif_scan_eager(cur64, v064, beta, THR, ALPHA, reset)
    assert bool((s32.double() == s64).all()), "precision changed the spike pattern"

    assert torch.allclose(gc_f.double(), gc_d, rtol=1e-4, atol=1e-5), float(
        (gc_f.double() - gc_d).abs().max()
    )
    assert torch.allclose(gv_f.double(), gv_d, rtol=1e-4, atol=1e-5), float(
        (gv_f.double() - gv_d).abs().max()
    )


# --------------------------------------------------------------------------
# leg 4 -- gradcheck and a third-party implementation
# --------------------------------------------------------------------------

def test_gradcheck_linear_regime():
    """A genuine finite-difference gradcheck, in the one place it is valid.

    `torch.autograd.gradcheck` compares against a numerical Jacobian, and the
    surrogate gradient is deliberately *not* the true derivative of a hard
    threshold -- so gradchecking the spike output would fail by construction and
    prove nothing. With reset="none" and only the membrane output taken, the map
    cur -> v_final is exactly linear (v_L = beta^L v_0 + sum beta^(L-1-t) cur_t),
    the surrogate never enters, and gradcheck is meaningful. It covers the leak
    recursion and the BPTT accumulation, which is the half of the scan a finite
    difference can see.
    """
    torch.manual_seed(7)
    cur = torch.randn(2, 12, 3, dtype=torch.float64, requires_grad=True)
    v0 = torch.randn(2, 3, dtype=torch.float64, requires_grad=True)

    def f(c, v):
        return lif_scan_eager(c, v, 0.9, THR, ALPHA, "none")[1]

    assert torch.autograd.gradcheck(f, (cur, v0), eps=1e-6, atol=1e-9)


@pytest.mark.cuda
@requires_fused
def test_against_snntorch():
    """Cross-check against snnTorch, an implementation nobody here wrote.

    Only `reset="none"` is comparable, and the reason is worth recording.
    snnTorch's `Leaky` applies the reset with a one-step delay by default
    (`reset_delay=True`): the spike at step t subtracts the threshold at step
    t+1, *after* the decay, so its membrane recursion is
    `mem_t = beta*mem_{t-1} + i_t - s_{t-1}*thr` where ours is
    `v_t = beta*(v_{t-1} - s_{t-1}*thr) + i_t`. Those are different neurons, not
    different spellings of one. Its `reset_delay=False` path is worse: it decides
    the spike on a membrane that has had the *previous* step's reset subtracted
    and then corrects the membrane afterwards, so a neuron sitting above
    threshold after a soft reset fails to fire (verified by hand-tracing on this
    stack, snntorch 1.0.0).

    With `reset_mechanism="none"` no reset happens under either convention, so
    the comparison is exact and worth having: it independently confirms the leak
    recursion, the arctan surrogate's value and derivative, and the BPTT
    accumulation over 64 steps.
    """
    snntorch = pytest.importorskip("snntorch")
    from snntorch import surrogate as sn_surrogate

    B, L, d, beta = 4, 64, 32, 0.9
    cur, _ = _inputs(B, L, d, seed=41, scale=0.5)
    v0z = torch.zeros(B, d, device="cuda")
    torch.manual_seed(19)
    gs = torch.randn(B, L, d, device="cuda")

    cell = snntorch.Leaky(
        beta=beta, threshold=THR,
        spike_grad=sn_surrogate.atan(alpha=ALPHA),
        reset_mechanism="none",
    ).cuda()
    c_sn = cur.detach().clone().requires_grad_(True)
    mem = torch.zeros(B, d, device="cuda")
    spk = []
    for t in range(L):
        s, mem = cell(c_sn[:, t], mem)
        spk.append(s)
    s_sn = torch.stack(spk, dim=1)
    (s_sn * gs).sum().backward()

    c_f = cur.detach().clone().requires_grad_(True)
    s_f, v_f = lif_scan(c_f, v0z, beta, THR, ALPHA, "none", fused=True)
    (s_f * gs).sum().backward()

    assert bool((s_sn == s_f).all())
    assert float((mem.detach() - v_f.detach()).abs().max()) < 1e-5
    assert torch.allclose(c_sn.grad, c_f.grad, rtol=1e-4, atol=1e-6)


# --------------------------------------------------------------------------
# the --no-fused path, input validation, and CUDA-graph capture
# --------------------------------------------------------------------------

def test_no_fused_path_runs_on_cpu():
    """`--no-fused` must work with no CUDA at all; it is the reference path."""
    cur = torch.randn(2, 16, 8)
    v0 = torch.zeros(2, 8)
    s, v = lif_scan(cur, v0, 0.9, THR, ALPHA, "hard", fused=False)
    assert s.shape == (2, 16, 8) and v.shape == (2, 8)
    assert bool(((s == 0.0) | (s == 1.0)).all())


@pytest.mark.cuda
@requires_fused
def test_fused_false_dispatches_to_eager():
    cur, v0 = _inputs(3, 32, 16, seed=5)
    s_a, v_a = lif_scan(cur, v0, 0.9, THR, ALPHA, "hard", fused=False)
    s_b, v_b = lif_scan_eager(cur, v0, 0.9, THR, ALPHA, "hard")
    assert bool((s_a == s_b).all()) and float((v_a - v_b).abs().max()) == 0.0


def test_input_validation():
    cur = torch.randn(2, 8, 4)
    v0 = torch.zeros(2, 4)
    with pytest.raises(ValueError):
        lif_scan(cur, v0, 0.9, THR, ALPHA, "reset-to-taste", fused=False)
    with pytest.raises(ValueError):
        lif_scan(cur[0], v0, 0.9, THR, ALPHA, "hard", fused=False)
    with pytest.raises(ValueError):
        lif_scan(cur, torch.zeros(2, 5), 0.9, THR, ALPHA, "hard", fused=False)
    with pytest.raises(TypeError):
        # spec §2 / B8: the membrane is fp32 whatever cfg.dtype says
        lif_scan(cur.bfloat16(), v0.bfloat16(), 0.9, THR, ALPHA, "hard", fused=False)


@pytest.mark.cuda
@requires_fused
def test_fused_apply_rejects_non_fp32():
    """`.apply` must enforce fp32 itself, not lean on `lif_scan` having done it.

    The kernel's membrane recursion uses `__fmul_rn`/`__fadd_rn`, which are
    float-only. Handed float64 it narrows v_pre to fp32 and casts back, so the
    call returns a float64 tensor that is only fp32-accurate (measured 4.7e-07
    against the eager float64 reference) with no error anywhere. Silence is the
    problem, not the precision.
    """
    cur, v0 = _inputs(2, 8, 4, seed=67, dtype=torch.float64)
    with pytest.raises(TypeError):
        FusedLIFScan.apply(cur, v0, 0.9, THR, ALPHA, RESET_CODES["hard"])


@pytest.mark.cuda
@requires_fused
def test_double_backward_raises_rather_than_lying():
    """`create_graph=True` through the fused scan must fail loudly.

    jiterator outputs carry no autograd history, so the fused backward is not
    itself differentiable. Without `@once_differentiable` the grad-of-grad is
    simply missing the scan's contribution -- and when some *other* term in the
    loss also depends on `cur`, the result still requires grad, so nothing
    raises and the number is merely wrong (measured 280.7 against the eager
    path's 497.7). The eager path is genuinely twice-differentiable and is
    checked here to prove the reference is the capable one.
    """
    def first_order(fused):
        torch.manual_seed(0)
        c = torch.randn(2, 8, 4, device="cuda", requires_grad=True)
        v0 = torch.zeros(2, 4, device="cuda", requires_grad=True)
        s, vf = lif_scan(c, v0, 0.9, THR, ALPHA, "hard", fused=fused)
        # the (c**3) term makes the grad-of-grad require grad regardless of the
        # scan, which is what removes the accidental "does not require grad"
        # error the fused path would otherwise be rescued by
        loss = (s * s).sum() + (vf * vf).sum() + (c * c * c).sum()
        return c, torch.autograd.grad(loss, c, create_graph=True)[0]

    c_e, g_e = first_order(fused=False)
    (gg_e,) = torch.autograd.grad(g_e.sum(), c_e)
    assert float(gg_e.abs().sum()) > 0.0

    with pytest.raises(RuntimeError, match="double backward"):
        first_order(fused=True)

    # and the ordinary single backward is untouched by the guard
    c, v0 = _inputs(2, 8, 4, seed=83)
    c = c.detach().requires_grad_(True)
    s, vf = lif_scan(c, v0, 0.9, THR, ALPHA, "hard", fused=True)
    (s.sum() + vf.sum()).backward()
    assert c.grad is not None and float(c.grad.abs().sum()) > 0.0


def test_reset_codes_are_the_documented_integers():
    """Spec §7 fixes 0=hard, 1=soft, 2=detached, 3=none. Other modules index it."""
    assert RESET_CODES == {"hard": 0, "soft": 1, "detached": 2, "none": 3}
    assert RESET_MODES == ("hard", "soft", "detached", "none")


@pytest.mark.cuda
@requires_fused
@pytest.mark.parametrize("reset", RESET_MODES)
def test_survives_cuda_graph_capture(reset):
    """The whole fused scan -- forward *and* backward -- inside a CUDAGraph.

    §3.11 C1 proved a bare jiterator call is capturable. That is not the same as
    proving an L-step scan wrapped in an autograd.Function is: capture forbids
    host syncs, forbids allocation outside the graph's private pool, and bakes in
    every Python-side branch it meets. Design principle §8.1.2 says fusion and
    graph capture have to compose or the ~51x combined figure is fiction, so it
    is proven here on the object the trainer will actually capture.
    """
    B, L, d, beta = 4, 32, 16, 0.9
    cur, v0 = _inputs(B, L, d, seed=53)
    cur = cur.detach().requires_grad_(True)
    v0 = v0.detach().requires_grad_(True)
    torch.manual_seed(31)
    gs = torch.randn(B, L, d, device="cuda")
    gv = torch.randn(B, d, device="cuda")

    def step():
        if cur.grad is not None:
            cur.grad.zero_()
            v0.grad.zero_()
        s, vf = lif_scan(cur, v0, beta, THR, ALPHA, reset, fused=True)
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

    # Replay against data the graph has never seen. This is what would catch a
    # captured kernel reading a stale address instead of the static buffer, which
    # is risk R5 in miniature.
    fresh, _ = _inputs(B, L, d, seed=97)
    cur.data.copy_(fresh)
    graph.replay()
    torch.cuda.synchronize()
    gc_graph = cur.grad.clone()
    gv_graph = v0.grad.clone()

    gc_e, gv_e = _grads(
        lif_scan_eager, cur.detach(), v0.detach(), beta, THR, reset, gs, gv
    )
    assert torch.allclose(gc_graph, gc_e, rtol=1e-4, atol=1e-6), float(
        (gc_graph - gc_e).abs().max()
    )
    assert torch.allclose(gv_graph, gv_e, rtol=1e-4, atol=1e-6)


@pytest.mark.cuda
@requires_fused
def test_autograd_function_apply_signature():
    """FusedLIFScan.apply takes an int reset_code, not a string (spec §7)."""
    cur, v0 = _inputs(2, 16, 8, seed=61)
    s, v = FusedLIFScan.apply(cur, v0, 0.9, THR, ALPHA, RESET_CODES["soft"])
    s_ref, v_ref = lif_scan_eager(cur, v0, 0.9, THR, ALPHA, "soft")
    assert bool((s == s_ref).all()) and float((v - v_ref).abs().max()) < 1e-5
