"""Gate for the two pre-scan arms (EXP_007 token-shift, EXP_008 threshold).

**This is deliberately NOT an R10 gate, and the reason is the point of both
arms.** R10 is about a hand-written jiterator backward being silently wrong.
Neither arm has one: both transform `cur` with ordinary autograd ops outside the
time loop and hand the result to `snn.neuron.lif_scan` -- the committed Phase-2
kernel, still guarded by its own gate and its 25/25 mutation campaign. What has
to be checked here is different and smaller:

  1. **the nesting is exact.** Each arm is *defined* to be the Phase-2 baseline at
     its initialisation, so at `mu = 1` and `theta = 0` the logits AND the
     gradients w.r.t. every shared parameter must be **bitwise** identical to
     `SpikingCharLM`'s. This is EXP_007's G1 and EXP_008's H1, and it is the leg
     that ties both arms to an artefact that is already trusted.
  2. **the shift is the shift.** An off-by-one, a leak across the batch, or a
     roll that wraps the last timestep into the first all produce a plausible
     model. Asserted against a hand-written reference rather than against the
     implementation's own idea of itself.
  3. **Identity 2 holds numerically** (EXP_008 §1.1): folding `exp(-theta)` into
     the layer's `Linear` reproduces the threshold arm's own logits. This is the
     tested form of the claim that the arm adds no functions -- the claim W1 then
     re-measures end-to-end in bpc, through a different code path.
  4. **the parameters are per channel and reach the model.** A `[1, d]` parameter
     that is broadcast from element 0, or that never enters the graph, produces a
     working model and a meaningless experiment (`EXP_002` §8.4's failure, and
     `EXP_004`'s P2).
  5. **the transform stays outside the time loop.** The whole cost argument for
     both arms is O(1) kernels per layer rather than O(L); a transform that
     silently moved inside the loop would keep every number in this project
     correct and every cost claim wrong.
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

from snn.config import Config  # noqa: E402
from snn.model import (  # noqa: E402
    LearnedThresholdCharLM,
    SpikingCharLM,
    TokenShiftCharLM,
    build_model,
    count_params,
    prescan_param_count,
    spiking_param_count,
)
from snn.prescan import shift_by_one, threshold_gain, token_shift  # noqa: E402

V, D, K, B, L = 37, 16, 2, 3, 11


def _cfg(**kw) -> Config:
    base = dict(vocab_size=V, d_model=D, n_layers=K, device="cpu",
                fused=False, deterministic=False)
    base.update(kw)
    return Config(**base)


def _idx(seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, V, (B, L), generator=g)


def _build(arch: str, seed: int = 0, **kw):
    """Both arms share `_CharLMStack`'s module order, so the same seed gives the
    same embedding/Linear/head init -- which is what makes the nesting checks
    bitwise rather than approximate. Same construction as the analogue control's."""
    torch.manual_seed(seed)
    return build_model(_cfg(arch=arch, **kw))


# ---------------------------------------------------------------------------
# 1. the nesting, bitwise -- EXP_007 G1 and EXP_008 H1
# ---------------------------------------------------------------------------


def _forward_and_grads(model, idx):
    logits, _, _ = model(idx, None)
    loss = logits.float().pow(2).mean()
    loss.backward()
    grads = {n: p.grad.detach().clone()
             for n, p in model.named_parameters() if p.grad is not None}
    return logits.detach().clone(), grads


@pytest.mark.parametrize("arch,shared_only", [("tokenshift", "mu"),
                                              ("threshold", "thr_log")])
def test_arm_at_its_init_is_the_committed_baseline_bitwise(arch, shared_only):
    """The nesting both pre-registrations depend on, forward *and* backward.

    Gradients matter as much as logits here: a transform that is the identity in
    the forward but perturbs the backward would leave the bpc verdict measuring
    two changes instead of one, and would do it invisibly.
    """
    idx = _idx()
    base = _build("snn")
    arm = _build(arch)

    base_logits, base_grads = _forward_and_grads(base, idx)
    arm_logits, arm_grads = _forward_and_grads(arm, idx)

    assert torch.equal(base_logits, arm_logits), "logits differ at the nesting init"
    for name, g in base_grads.items():
        assert name in arm_grads, f"{name} got no gradient in the {arch} arm"
        assert torch.equal(g, arm_grads[name]), f"gradient differs for {name}"
    # ... and the new parameter is in the graph rather than merely present.
    new = [n for n in arm_grads if n.startswith(shared_only)]
    assert len(new) == K, f"expected {K} {shared_only} gradients, got {new}"


@pytest.mark.parametrize("arch", ["tokenshift", "threshold"])
def test_the_new_parameter_is_reachable_at_the_nesting_init(arch):
    """§7.2's rule, at the scale this test file can afford.

    `03_phase3_candidates.md` §7.2 exists because three of the four exact-nesting
    initialisations this project has proposed are gradient saddles. Both arms here
    are derived non-saddle (EXP_007 §1.2, EXP_008 §2.1) and screened by
    `scripts/audit/09_gradient_reachability.py`; this is the cheap standing check
    that the property survives an edit to the model wiring.
    """
    idx = _idx(1)
    arm = _build(arch, seed=3)
    logits, _, _ = arm(idx, None)
    logits.float().pow(2).mean().backward()
    for name, p in arm.named_parameters():
        if name.startswith(("mu", "thr_log")):
            assert p.grad is not None and p.grad.abs().max() > 0, (
                f"{name} is an exact saddle at its initialisation")


# ---------------------------------------------------------------------------
# 2. the shift is the shift -- EXP_007 M2
# ---------------------------------------------------------------------------


def test_shift_by_one_matches_a_hand_written_reference():
    g = torch.Generator().manual_seed(4)
    cur = torch.randn(B, L, D, generator=g)
    got = shift_by_one(cur)
    want = torch.zeros_like(cur)
    for t in range(1, L):
        want[:, t] = cur[:, t - 1]
    assert torch.equal(got, want)
    assert torch.equal(got[:, 0], torch.zeros(B, D)), "t=0 must see a zero, not a wrap"


def test_shift_does_not_leak_across_the_batch():
    """A reshape that treats [B, L] as one stream would pass the previous test on
    row 0 and quietly hand row b the last timestep of row b-1."""
    cur = torch.zeros(B, L, D)
    cur[0] = 1.0                       # only sequence 0 is non-zero
    got = shift_by_one(cur)
    assert torch.equal(got[1:], torch.zeros(B - 1, L, D))


def test_token_shift_is_the_identity_at_mu_one_and_a_mix_below_it():
    g = torch.Generator().manual_seed(5)
    cur = torch.randn(B, L, D, generator=g)
    mu_one = torch.ones(1, D)
    assert torch.equal(token_shift(cur, mu_one), cur)

    mu = torch.full((1, D), 0.25)
    got = token_shift(cur, mu)
    want = 0.25 * cur + 0.75 * shift_by_one(cur)
    assert torch.allclose(got, want, rtol=0, atol=0)


# ---------------------------------------------------------------------------
# 3. Identity 2 -- EXP_008 §1.1, the fold
# ---------------------------------------------------------------------------


def test_threshold_gain_is_the_identity_at_theta_zero():
    g = torch.Generator().manual_seed(6)
    cur = torch.randn(B, L, D, generator=g)
    assert torch.equal(threshold_gain(cur, torch.zeros(1, D)), cur)


def test_folding_the_gain_into_the_linear_reproduces_the_arm():
    """Identity 2: the threshold arm's function class IS the baseline's.

    Tolerance rather than equality, and stated in EXP_008 §5 item 1: `g*(W·h + b)`
    and `(g*W)·h + g*b` round differently because the GEMM accumulates in a
    different order. What is being asserted is that the two models are the same
    *function*, not that fp32 evaluates it in the same order.
    """
    arm = _build("threshold", seed=7)
    g = torch.Generator().manual_seed(70)
    with torch.no_grad():                      # a trained-looking, non-trivial gain
        for k in range(K):
            arm.thr_log[k].copy_(torch.randn(1, D, generator=g) * 0.25 - 0.3)

    folded_sd = arm.fold_into_spiking_state_dict()
    base = _build("snn", seed=99)              # different init: only the load matters
    base.load_state_dict(folded_sd, strict=True)

    idx = _idx(8)
    with torch.no_grad():
        a, _, _ = arm(idx, None)
        b, _, _ = base(idx, None)
    assert a.shape == b.shape
    assert torch.allclose(a, b, rtol=1e-4, atol=1e-4), (
        f"folded model differs by {(a - b).abs().max().item():.3e}")


def test_the_fold_produces_exactly_a_baseline_state_dict():
    """No leftover `thr_log`, no missing key, and the baseline's own parameter
    count -- so the folded artefact is loadable as `arch="snn"` and is not merely
    similar to one."""
    arm = _build("threshold", seed=10)
    folded = arm.fold_into_spiking_state_dict()
    base = _build("snn", seed=11)
    assert set(folded) == set(base.state_dict())
    assert sum(v.numel() for v in folded.values()) == spiking_param_count(V, D, K)


# ---------------------------------------------------------------------------
# 4. per-channel, and actually used
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("arch,attr", [("tokenshift", "mu"), ("threshold", "thr_log")])
def test_parameter_is_per_channel_and_not_broadcast_from_one_element(arch, attr):
    """`EXP_002` §8.4's failure, in this file's terms: a `[1, d]` parameter bound
    as a scalar, or broadcast from element 0, compiles and produces plausible
    spikes. Recovered from the model's own output rather than asserted on shapes:
    perturbing channel 0 alone must change channel 0's current alone."""
    arm = _build(arch, seed=12)
    for k in range(K):
        p = getattr(arm, attr)[k]
        assert tuple(p.shape) == (1, D)

    cur = torch.randn(B, L, D, generator=torch.Generator().manual_seed(13))
    p = getattr(arm, attr)[0].detach().clone()
    fn = token_shift if arch == "tokenshift" else threshold_gain
    before = fn(cur, p)
    p2 = p.clone()
    p2[0, 0] += 0.5
    after = fn(cur, p2)
    changed = (before != after).flatten(0, 1).any(dim=0)
    assert bool(changed[0]), "channel 0's own parameter did not affect channel 0"
    assert not bool(changed[1:].any()), "a per-channel parameter reached other channels"


@pytest.mark.parametrize("arch", ["tokenshift", "threshold"])
def test_parameter_count_is_the_baseline_plus_one_per_channel_per_layer(arch):
    """EXP_007 G2 / EXP_008 H2, at test scale: +K*d and nothing else."""
    arm = _build(arch, seed=14)
    assert count_params(arm) == prescan_param_count(V, D, K)
    assert count_params(arm) == spiking_param_count(V, D, K) + K * D


def test_parameter_count_at_the_committed_configuration():
    """The number the experiment logs quote: 735_437 + 1_024 = 736_461, +0.14%."""
    assert prescan_param_count(205, 512, 2) == 736_461
    assert spiking_param_count(205, 512, 2) == 735_437


# ---------------------------------------------------------------------------
# 5. still outside the time loop
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("arch", ["tokenshift", "threshold"])
def test_the_transform_runs_once_per_layer_not_once_per_timestep(arch):
    """The entire cost argument for both arms.

    Counted by intercepting the transform itself: a refactor that moved it inside
    the scan would keep every bpc in this project correct and every cost claim in
    EXP_007 §1 and EXP_008 §1.1 wrong. `L` is varied so that "once per layer" is
    distinguished from "a small number that happens to match".
    """
    import snn.model as model_mod

    name = "token_shift" if arch == "tokenshift" else "threshold_gain"
    real = getattr(model_mod, name)
    calls = []

    def counting(cur, p):
        calls.append(cur.shape[1])
        return real(cur, p)

    setattr(model_mod, name, counting)
    try:
        arm = _build(arch, seed=15)
        for length in (4, 16):
            calls.clear()
            with torch.no_grad():
                arm(torch.randint(0, V, (B, length)), None)
            assert len(calls) == K, (
                f"{name} ran {len(calls)} times for {K} layers at L={length}")
            assert calls == [length] * K, "the transform saw a per-timestep slice"
    finally:
        setattr(model_mod, name, real)


# ---------------------------------------------------------------------------
# 6. the arms are wired into the ordinary construction path
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("arch,cls", [("tokenshift", TokenShiftCharLM),
                                      ("threshold", LearnedThresholdCharLM)])
def test_build_model_dispatches_and_records_its_init(arch, cls):
    arm = build_model(_cfg(arch=arch, mu_init=0.75, thr_log_init=-0.2))
    assert isinstance(arm, cls)
    if arch == "tokenshift":
        assert arm.mu_init == 0.75
        assert torch.allclose(arm.mu[0], torch.full((1, D), 0.75))
    else:
        assert arm.thr_log_init == -0.2
        assert torch.allclose(arm.threshold_multiplier(0),
                              torch.full((1, D), math.exp(-0.2)))
        assert torch.allclose(arm.input_gain(0), torch.full((1, D), math.exp(0.2)))


def test_unknown_arch_still_names_every_arm():
    """`Config` validates `arch`, so this can only be reached by mutating the
    dataclass afterwards -- which is exactly what a half-finished refactor does."""
    cfg = _cfg(arch="snn")
    cfg.arch = "bogus"
    with pytest.raises(ValueError, match="tokenshift"):
        build_model(cfg)
