"""EXP_013's gates G1-G6, as tests.

Pre-registered in `experiments/logs/EXP_013_noise_injection.md` §6. Every gate
here is an *entry condition* for the experiment rather than a regression test:
G1 is what licenses using the five committed Phase-2 baseline seeds as the
noise-off control instead of training three more, and G2 is what makes the arm a
training-only intervention rather than a different model.

Each of G1, G3 and G4 carries a **mutation leg** that deliberately breaks the
property and asserts the check notices. `03_phase3_candidates.md` §7.3: a screen
nothing has ever tripped is indistinguishable from one that cannot trip.
"""

from __future__ import annotations

import math

import pytest
import torch

from snn.config import Config
from snn.data import _mix64
from snn.model import NoisyCharLM, SpikingCharLM, build_model
from snn.noise import (NOISE_STREAM_SALT, add_background_noise,
                       fill_background_noise, noise_scale)

V, D, K, B, L = 37, 16, 2, 4, 8


def _pair(noise_amp: float, seed: int = 0):
    """A NoisyCharLM and a SpikingCharLM with bit-identical weights."""
    torch.manual_seed(seed)
    noisy = NoisyCharLM(V, D, K, noise_amp=noise_amp, noise_p=0.1, fused=False)
    torch.manual_seed(seed)
    plain = SpikingCharLM(V, D, K, fused=False)
    plain.load_state_dict({k: v.clone() for k, v in noisy.state_dict().items()})
    return noisy, plain


def _idx(seed: int = 3):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, V, (B, L), generator=g)


# --------------------------------------------------------------------------
# G1 -- the nesting, bitwise, forward AND backward
# --------------------------------------------------------------------------


def test_g1_amp_zero_is_the_baseline_bitwise_forward():
    """At amp=0 the arm IS SpikingCharLM. Not a tolerance -- torch.equal.

    This is what earns the right to use the committed baseline seeds as the
    noise-off control. EXP_011 K4 is the precedent for asserting a nesting
    bitwise rather than to 1e-12.
    """
    noisy, plain = _pair(0.0)
    noisy.train(); plain.train()
    idx = _idx()
    a, _, _ = noisy(idx, None)
    b, _, _ = plain(idx, None)
    assert torch.equal(a, b)


def test_g1_amp_zero_is_the_baseline_bitwise_backward():
    noisy, plain = _pair(0.0)
    noisy.train(); plain.train()
    idx = _idx()
    for m in (noisy, plain):
        logits, _, _ = m(idx, None)
        logits.square().mean().backward()
    for (na, pa), (nb, pb) in zip(sorted(noisy.named_parameters()),
                                  sorted(plain.named_parameters())):
        assert na == nb
        assert (pa.grad is None) == (pb.grad is None)
        if pa.grad is not None:
            assert torch.equal(pa.grad, pb.grad), f"gradient differs at {na}"


def test_g1_amp_zero_allocates_no_buffer():
    """The nesting costs no memory either, so the control is free in every sense."""
    noisy, _ = _pair(0.0)
    noisy.allocate_noise(B, L)
    assert noisy._noise == []
    assert noisy.noise_active is False


def test_g1_mutation_leg_a_real_amplitude_does_change_the_forward():
    """The mutation leg for G1: if amp>0 did NOT change the output, G1 would pass
    vacuously and the whole experiment would be measuring the baseline twice."""
    noisy, plain = _pair(0.5)
    noisy.train(); plain.train()
    noisy.allocate_noise(B, L)
    noisy.refill_noise(seed=0, step=0)
    idx = _idx()
    a, _, _ = noisy(idx, None)
    b, _, _ = plain(idx, None)
    assert not torch.equal(a, b)


# --------------------------------------------------------------------------
# G2 -- training only
# --------------------------------------------------------------------------


@pytest.mark.parametrize("amp", [0.1, 0.4, 1.6])
def test_g2_eval_mode_is_the_baseline_bitwise(amp):
    """Under eval() the arm must be the baseline at EVERY amplitude.

    `snn.evaluate.evaluate` sets eval() and restores the previous mode, so this
    is what makes every committed evaluation path correct for this arm without
    knowing it exists.
    """
    noisy, plain = _pair(amp)
    noisy.allocate_noise(B, L)
    noisy.refill_noise(seed=0, step=0)
    noisy.eval(); plain.eval()
    idx = _idx()
    a, _, _ = noisy(idx, None)
    b, _, _ = plain(idx, None)
    assert torch.equal(a, b)
    assert noisy.noise_active is False


def test_g2_training_mode_without_buffers_raises_rather_than_trains_the_baseline():
    """A silently-unnoised noise arm looks exactly like a null result.

    `03_phase3_candidates.md` §8 is the precedent this guards against: a run that
    trained 7 750 steps against a mutated kernel with nothing in its own logs to
    say so.
    """
    noisy, _ = _pair(0.4)
    noisy.train()
    with pytest.raises(RuntimeError, match="no noise buffer"):
        noisy(_idx(), None)


# --------------------------------------------------------------------------
# G3 -- the draw is centred and correctly scaled
# --------------------------------------------------------------------------


@pytest.mark.parametrize("amp,p", [(0.1, 0.1), (0.4, 0.1), (1.6, 0.1),
                                   (0.4, 0.5), (0.4, 0.9)])
def test_g3_draw_is_zero_mean_and_amp_is_its_sd(amp, p):
    """EXP_013 §1.2: an uncentred draw is a threshold shift, which EXP_008 already
    measured at 0.0590 bpc -- ten times the whole gap this arm is aimed at."""
    n = 2_000_000
    buf = torch.empty(n)
    g = torch.Generator().manual_seed(11)
    fill_background_noise(buf, amp, p, g)

    se = amp / math.sqrt(n)
    assert abs(float(buf.mean())) < 4 * se, "draw is not centred"
    assert abs(float(buf.std()) - amp) < 0.01 * amp, "amp is not the sd"


def test_g3_mutation_leg_uncentred_draw_is_caught():
    """The same assertions applied to the uncentred form must FAIL."""
    n = 2_000_000
    amp, p = 0.4, 0.1
    g = torch.Generator().manual_seed(11)
    bad = torch.empty(n).bernoulli_(p, generator=g).mul_(noise_scale(amp, p))
    se = amp / math.sqrt(n)
    assert abs(float(bad.mean())) > 4 * se, (
        "the uncentred draw passed the centring check, so the check is not a check"
    )


def test_g3_scale_rejects_degenerate_p():
    for p in (0.0, 1.0, -0.1, 1.5):
        with pytest.raises(ValueError):
            noise_scale(0.4, p)
    with pytest.raises(ValueError):
        noise_scale(-0.1, 0.5)


# --------------------------------------------------------------------------
# G4 -- resumability: the noise is a pure function of (seed, step)
# --------------------------------------------------------------------------


def test_g4_refill_is_a_pure_function_of_seed_and_step():
    """R6: a run resumed at step k must inject what an uninterrupted run injected.

    `snn.data` §3 makes the same guarantee for the batch, by the same construction
    -- a generator seeded per call rather than advanced from a stateful stream.
    """
    noisy, _ = _pair(0.4)
    noisy.allocate_noise(B, L)

    noisy.refill_noise(seed=7, step=1234)
    first = [b.clone() for b in noisy._noise]
    # Advance the global stream to prove the draw does not depend on it.
    torch.randn(1000)
    noisy.refill_noise(seed=7, step=1234)
    for a, b in zip(first, noisy._noise):
        assert torch.equal(a, b), "refill at the same (seed, step) is not reproducible"

    noisy.refill_noise(seed=7, step=1235)
    assert not any(torch.equal(a, b) for a, b in zip(first, noisy._noise))

    noisy.refill_noise(seed=8, step=1234)
    assert not any(torch.equal(a, b) for a, b in zip(first, noisy._noise))


def test_g4_layers_get_different_draws():
    """One shared draw across layers would be a different, correlated experiment."""
    noisy, _ = _pair(0.4)
    noisy.allocate_noise(B, L)
    noisy.refill_noise(seed=0, step=0)
    assert not torch.equal(noisy._noise[0], noisy._noise[1])


def test_g4_mutation_leg_step_independent_seed_is_caught():
    """If the seed ignored `step`, the same noise would be injected every step and
    the arm would be a fixed perturbation rather than a stochastic one."""
    a = _mix64(7 ^ NOISE_STREAM_SALT, 1234)
    b = _mix64(7 ^ NOISE_STREAM_SALT, 1235)
    assert a != b, "the (seed, step) mix is not step-dependent"


# --------------------------------------------------------------------------
# G5 -- the noise stream is independent of the data stream
# --------------------------------------------------------------------------


def test_g5_noise_stream_is_salted_away_from_the_sampler():
    """Without the salt both would be `_mix64(seed, step)` and the injected noise
    would be a deterministic function of which windows the batch drew -- a
    confound no metric in this project would reveal."""
    for seed in (0, 1, 2, 42):
        for step in (0, 1, 19_999):
            assert _mix64(seed, step) != _mix64(seed ^ NOISE_STREAM_SALT, step)


# --------------------------------------------------------------------------
# shape contract -- a broadcastable-but-wrong buffer still trains
# --------------------------------------------------------------------------


def test_add_rejects_broadcastable_shapes():
    """A [1, L, d] buffer broadcasts against [B, L, d] and shares one draw across
    the batch. It would train, and it would not be this experiment."""
    cur = torch.zeros(B, L, D)
    with pytest.raises(ValueError, match="match cur exactly"):
        add_background_noise(cur, torch.zeros(1, L, D))
    with pytest.raises(ValueError, match="match cur exactly"):
        add_background_noise(cur, torch.zeros(B, L, 1))
    with pytest.raises(ValueError):
        add_background_noise(torch.zeros(B, L), torch.zeros(B, L))


# --------------------------------------------------------------------------
# wiring: Config -> build_model, and the parameter count is the baseline's
# --------------------------------------------------------------------------


def test_build_model_wires_the_arm():
    cfg = Config(arch="noise", vocab_size=V, d_model=D, n_layers=K,
                 noise_amp=0.4, noise_p=0.1, device="cpu", fused=False)
    m = build_model(cfg)
    assert isinstance(m, NoisyCharLM)
    assert m.noise_amp == 0.4 and m.noise_p == 0.1


def test_arm_adds_no_parameters():
    """No parameter, no state variable, no kernel: the arm changes only how the
    model is trained, which is what `04_phase4_interim.md` §7 item 5 asks for."""
    noisy, plain = _pair(0.4)
    assert (sum(p.numel() for p in noisy.parameters())
            == sum(p.numel() for p in plain.parameters()))
    assert sorted(dict(noisy.named_parameters())) == sorted(dict(plain.named_parameters()))


def test_config_rejects_degenerate_noise_p():
    for p in (0.0, 1.0):
        with pytest.raises(ValueError, match="noise_p"):
            Config(noise_p=p)
    with pytest.raises(ValueError, match="noise_amp"):
        Config(noise_amp=-1.0)
