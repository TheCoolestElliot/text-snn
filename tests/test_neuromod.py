"""Gates for the two neuromodulation forms (EXP_021).

Like `tests/test_dopamine.py`, and for the same reason, **this is deliberately
not an R10 gate**. Neither form here owns a scan or a hand-written backward:
`local_rpe` is a shift, a sigmoid, two reductions over the channel axis and a
broadcast multiply, all outside the time loop; `three_factor` never touches the
forward pass at all. `snn.neuron.lif_scan` and its mutation campaign are
untouched, and G1 is what asserts they are.

What has to be checked is four things, and three of them are properties the
module docstrings *derive* rather than measure -- which is exactly why they are
asserted here rather than reported by a script:

  1. **`phi` is zero-mean by construction.** `E_{s~Bern(p)}[-log P(s)] = H_b(p)`
     exactly, so the baseline is the predictor's own entropy and there is no EMA,
     no critic and nothing to tune. D1 checks the derivation at its own fixed
     point: with the predictor sitting on the channels' true rates, `phi.mean()`
     must be zero to fp32 noise. A sign error, a missing term or a `log` where a
     `logsigmoid` belongs all move it off zero.
  2. **the reduction is a MEAN and not a sum.** Summed, `phi` is O(d) -- +101
     nats at `d = 512` -- and `tanh(101/tau)` is 1.0 to fp32. That failure is
     silent and total: the predictor's gradient vanishes and `cur*(1+kappa*DA)`
     degenerates into a constant per-channel gain, which by `EXP_008` Identity 1
     is a learned per-channel threshold -- an arm this project has already
     adopted (#5). The saturated arm would have trained, scored plausibly, and
     been measuring `EXP_008` under a different name. D2 is the gate against it.
  3. **alignment.** `snn.dopamine.rpe` builds `phi_u` from `logits[:, u-1]`
     because that signal is consumed inside the forward pass at `u`;
     `three_factor_loss` builds it from `logits[:, t]` because that one is
     consumed by the loss at `t`, whose target is `y_t` by definition. The two
     are the same algebra deliberately aligned two different ways, and either
     one wearing the other's indexing would still train and would still look
     plausible. T3 asserts it against a hand-rolled reference AND behaviourally,
     and carries the shifted variant as a mutation leg.
  4. **causality**, by perturbation rather than by argument -- C1, as
     `tests/test_dopamine.py`'s G2 is.

ON G2, WHICH DOCUMENTS A SADDLE RATHER THAN A BUG. At `nm_gain_init = 0.0` the
modulated current is `cur * (1 + 0*DA)`, so `dL/da`, `dL/db` and `dL/dlog_tau`
are all **exactly zero** -- the chain to each of them passes through `nm_gain`.
That is derivable from the form and is not a defect: `nm_gain` itself is
reachable (measured at 5.8e-05 here), it moves off zero on the first optimiser
step, and every other modulator parameter becomes reachable immediately
afterwards. It is recorded as a gate because §7.2's screen flags anything
exactly zero, and a reader meeting three exact zeros in that screen's output
needs to find this test rather than file a bug. G2's second leg asserts the
escape: at `nm_gain_init = 0.1` all four are non-zero.
"""

from __future__ import annotations

import inspect
import math

import pytest
import torch
import torch.nn.functional as F

import snn.dopamine
import snn.model
from snn.config import Config
from snn.model import (LocalDopamineCharLM, SpikingCharLM, build_model,
                       count_params, local_dopamine_param_count,
                       spiking_param_count)
from snn.neuromod import (NM_SOURCES, bernoulli_rpe, three_factor_loss,
                          three_factor_weights)
from snn.prescan import shift_by_one

V, D, K, B, L = 37, 16, 2, 4, 8
TAU = 1.0

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(),
                                   reason="needs a GPU")


def _idx(seed: int = 3, b: int = B, length: int = L) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, V, (b, length), generator=g)


def _spikes(seed: int = 5, b: int = B, length: int = 64, d: int = D,
            rate: float = 0.3) -> torch.Tensor:
    """A binary tensor with the alphabet the modulator actually reads: {0, 1}."""
    g = torch.Generator().manual_seed(seed)
    return (torch.rand(b, length, d, generator=g) < rate).float()


def _prior_from(spikes: torch.Tensor) -> torch.Tensor:
    """`b_c = logit(r_c)` for the EMPIRICAL per-channel rate `r_c`.

    The predictor's own fixed point when `a = 0`: `p_c = sigmoid(b_c) = r_c`,
    which is what makes D1's zero mean exact rather than approximate.
    """
    rate = spikes.mean(dim=(0, 1)).clamp(1e-4, 1.0 - 1e-4)
    return torch.log(rate / (1.0 - rate)).reshape(1, -1)


def _pair(seed: int = 0, **kw):
    """A LocalDopamineCharLM and a SpikingCharLM with bit-identical shared
    weights. Matched by name, never positionally -- `nm_a.0` sorts between
    `layers.1.weight` and neither model's parameter order is the other's."""
    torch.manual_seed(seed)
    arm = LocalDopamineCharLM(V, D, K, fused=False, **kw)
    torch.manual_seed(seed)
    plain = SpikingCharLM(V, D, K, fused=False)
    plain.load_state_dict({k: v.clone() for k, v in arm.state_dict().items()
                           if not k.startswith("nm_")})
    return arm, plain


def _reference_three_factor(logits, targets, *, kappa, tau, roll=False):
    """The documented formula, recomputed from scratch inside the test.

    Deliberately NOT a call into `snn.neuromod`: a reference that shares the
    implementation's code agrees with it by construction and measures nothing
    (`EXP_004` §9.2). Returns `(loss, w)` so that T3 can assert both the scalar
    the trainer sees and the per-position weights it never gets to look at.
    """
    flat = logits.reshape(-1, logits.shape[-1]).float()
    tgt = targets.reshape(-1)
    per = F.cross_entropy(flat, tgt, reduction="none")
    logp = F.log_softmax(flat, dim=-1)
    entropy = -(logp.exp() * logp).sum(-1)
    w = (1.0 + kappa * torch.tanh((entropy - per) / tau)).detach()
    if roll:
        w = torch.roll(w.reshape(targets.shape), shifts=1, dims=0).reshape(-1)
    return (per * w).sum() / w.sum(), w


# ---------------------------------------------------------------------------
# D1 -- phi is zero-mean by construction
# ---------------------------------------------------------------------------

def test_d1_phi_is_zero_mean_at_the_predictors_own_fixed_point():
    """With `a = 0` and `b_c = logit(r_c)`, `mean(phi)` is zero to fp32 noise.

    This is the derivation check and not a statistical one. `H_t` is the
    predictor's exact expectation of `S_t` under its own belief, so averaged
    over positions the two agree term by term once `p_c` equals the realised
    rate -- no baseline, no EMA, no batch statistic. Measured here at -4.3e-08
    against an sd of 0.090, i.e. six orders of magnitude below the signal.
    """
    spikes = _spikes()
    a = torch.zeros(1, D)
    b = _prior_from(spikes)
    phi = bernoulli_rpe(spikes, a, b)
    assert phi.shape == spikes.shape[:2]
    assert abs(float(phi.mean())) < 1e-3
    # ...and the signal it is zero-mean *in* is not itself zero, so the check
    # above is not passing on a tensor of zeros.
    assert float(phi.std()) > 1e-3


def test_d1_mutation_leg_a_prior_off_the_true_rate_moves_the_mean():
    """The gate above must be able to fire.

    A predictor that is wrong about the population's rate is *surprised on
    average*, so `phi` acquires a negative mean -- which is exactly what a
    frozen `tau` and a frozen `b` would produce once the firing rate rises 7x
    during training, and exactly why `b` and `tau` are learned here.
    """
    spikes = _spikes()
    a = torch.zeros(1, D)
    honest = _prior_from(spikes)
    phi_ok = bernoulli_rpe(spikes, a, honest)
    phi_bad = bernoulli_rpe(spikes, a, honest - 2.0)
    assert abs(float(phi_ok.mean())) < 1e-3
    assert float(phi_bad.mean()) < -1e-2


def test_d1_phi_is_a_surprise_gap_and_not_a_cross_entropy():
    """`phi = H - S`, so a perfectly predicted channel contributes zero.

    Asserted against a hand-rolled Bernoulli entropy rather than against the
    implementation's own reduction: with `p` saturated at the observed value the
    surprise IS the entropy and the difference is exactly zero, which no
    sign-flipped or absolute-valued variant reproduces.
    """
    spikes = torch.tensor([[[1.0, 0.0], [1.0, 0.0]]])
    # a huge |logit| makes p = s exactly, so H = S = 0 per channel
    a = torch.zeros(1, 2)
    b = torch.tensor([[30.0, -30.0]])
    phi = bernoulli_rpe(spikes, a, b)
    assert torch.allclose(phi, torch.zeros(1, 2), atol=1e-6)


# ---------------------------------------------------------------------------
# D2 -- the reduction is a MEAN, and that is not cosmetic
# ---------------------------------------------------------------------------

def test_d2_the_reduction_is_a_mean_and_not_a_sum():
    """Double `d` at identical per-channel statistics; the scale must not move.

    The width is doubled by DUPLICATING every channel -- same rate, same
    predictor, same realised spike train -- so a mean is invariant exactly and a
    sum is exactly 2x. That makes the discrimination a factor of two rather than
    a distributional argument: an iid doubling would have separated 1.0 from
    only sqrt(2), and `EXP_014`/`EXP_015` run arms at three widths, so this is
    the property that lets one calibrated `tau` cover all of them.
    """
    spikes = _spikes()
    a = torch.zeros(1, D)
    b = _prior_from(spikes)
    phi = bernoulli_rpe(spikes, a, b)

    wide = torch.cat([spikes, spikes], dim=-1)
    phi_wide = bernoulli_rpe(wide, torch.cat([a, a], -1), torch.cat([b, b], -1))

    ratio = float(phi_wide.std() / phi.std())
    assert abs(ratio - 1.0) < 1e-3, f"phi's scale moved with d: ratio {ratio}"
    # The same tensors under a SUM reduction, which is the mutation this gate
    # exists for: the ratio is 2, and `tanh` would be saturated at any tau of
    # order one.
    summed_ratio = float((phi_wide * 2 * D).std() / (phi * D).std())
    assert abs(summed_ratio - 2.0) < 1e-2


def test_d2_a_summed_phi_would_saturate_the_squash():
    """Why D2 matters, stated in the units the arm actually runs at.

    At `d = 512` and a realistic firing rate the summed RPE is tens of nats, and
    `tanh` of that is 1.0 to fp32 -- a constant gain, zero gradient to the
    predictor, and an arm that has silently become `EXP_008`.
    """
    spikes = _spikes(d=512, length=16, rate=0.34)
    a = torch.zeros(1, 512)
    b = _prior_from(spikes) - 1.0        # a predictor that is merely wrong
    phi = bernoulli_rpe(spikes, a, b)
    assert float(phi.abs().max()) < 5.0                    # meaned: usable
    summed = phi * 512
    assert float(torch.tanh(summed / TAU).abs().min()) == 1.0   # summed: dead


# ---------------------------------------------------------------------------
# C1 -- causality, by perturbation
# ---------------------------------------------------------------------------

def test_c1_phi_at_t_is_bitwise_invariant_to_the_future():
    """`p_t` is built from `s_{t-1}` and scored against `s_t`. Nothing later.

    Both depend on `x_{<=t}` and the target at `t` is `x_{t+1}`, so `phi[:, :t+1]`
    must not move by a single bit when everything after `t` is flipped. `y` is
    not an argument to anything in `snn.neuromod`'s forward-pass form, which the
    signature check below states outright; this is the functional half.
    """
    spikes = _spikes(length=L)
    a = torch.full((1, D), 1.5)
    b = torch.full((1, D), -0.7)
    base = bernoulli_rpe(spikes, a, b)
    for t in range(L - 1):
        future = spikes.clone()
        future[:, t + 1:] = 1.0 - future[:, t + 1:]      # flip every later spike
        got = bernoulli_rpe(future, a, b)
        assert torch.equal(got[:, :t + 1], base[:, :t + 1]), (
            f"phi leaked the future at t={t}: max|d|="
            f"{float((got[:, :t + 1] - base[:, :t + 1]).abs().max()):.3e}")


def test_c1_mutation_leg_a_predictor_reading_forward_is_caught():
    """The gate above must be able to fire.

    One character: `shift_by_one` -> a forward shift. The predictor then scores
    `s_t` against a belief formed from `s_{t+1}`, which is a leak in the only
    direction that matters. It is caught at t = 0.
    """
    def anti_causal_rpe(spikes, a, b):
        s = spikes.float()
        ahead = F.pad(s[:, 1:], (0, 0, 0, 1))            # MUTANT: shift forward
        logit = a * ahead + b
        logp, log1mp = F.logsigmoid(logit), F.logsigmoid(-logit)
        p = torch.sigmoid(logit)
        observed = -(s * logp + (1.0 - s) * log1mp).mean(-1)
        expected = -(p * logp + (1.0 - p) * log1mp).mean(-1)
        return expected - observed

    spikes = _spikes(length=L)
    a = torch.full((1, D), 1.5)
    b = torch.full((1, D), -0.7)
    base = anti_causal_rpe(spikes, a, b)
    caught = False
    for t in range(L - 1):
        future = spikes.clone()
        future[:, t + 1:] = 1.0 - future[:, t + 1:]
        if not torch.equal(anti_causal_rpe(future, a, b)[:, :t + 1],
                           base[:, :t + 1]):
            caught = True
            break
    assert caught, "the causality gate cannot detect a forward-reading predictor"


def test_c1_neither_form_can_reach_the_targets_by_signature():
    """`bernoulli_rpe` never sees `y`; `three_factor_loss` sees it because it IS
    the loss, and detaches the weight it builds from it."""
    assert list(inspect.signature(bernoulli_rpe).parameters) == \
        ["spikes", "a", "b"]
    assert list(inspect.signature(LocalDopamineCharLM.forward).parameters) == \
        ["self", "idx", "state"]


def test_c1_position_zero_is_scored_rather_than_zeroed():
    """The documented difference from `snn.dopamine.rpe`, asserted.

    That function zeroes position 0 because there is no preceding *prediction*
    there. Here there is one: `s_{-1}` is the zero vector by `shift_by_one`'s
    padding, so `p_0 = sigmoid(b)` is the channel's own prior rate -- a real
    prediction, scored as one. Zeroing it would throw away the position
    `EXP_004` §10.6 says the cheap gains have been.
    """
    spikes = _spikes(length=L)
    b = torch.full((1, D), -2.97)
    phi = bernoulli_rpe(spikes, torch.zeros(1, D), b)
    assert float(phi[:, 0].abs().min()) > 0.0
    # ...and it really is `sigmoid(b)` that scored it: the shift pads with zeros
    assert torch.equal(shift_by_one(spikes)[:, 0], torch.zeros(B, D))


@pytest.mark.parametrize("spikes,a,b", [
    (torch.zeros(L, D), torch.zeros(1, D), torch.zeros(1, D)),
    (torch.zeros(B, L, D), torch.zeros(D), torch.zeros(1, D)),
    (torch.zeros(B, L, D), torch.zeros(1, D), torch.zeros(2, D)),
    (torch.zeros(B, L, D), torch.zeros(1, D), torch.zeros(1, D + 1)),
])
def test_d3_shapes_that_broadcast_but_mean_something_else_are_rejected(spikes, a, b):
    """A `[d]` predictor broadcasts against `[B, L, d]` perfectly well and is a
    different experiment; a `[2, d]` one silently shares a predictor across two
    batch rows, which is the batch statistic this file forbids."""
    with pytest.raises(ValueError):
        bernoulli_rpe(spikes, a, b)


# ---------------------------------------------------------------------------
# T1/T2 -- the three-factor objective's nesting and its guards
# ---------------------------------------------------------------------------

def test_t1_kappa_zero_is_the_unweighted_objective_exactly():
    """`torch.equal`, not `allclose`: the nesting is a CODE PATH.

    `kappa = 0.0` returns `F.cross_entropy`'s own mean reduction rather than
    multiplying by a tensor of ones, so the anchor is the identical sequence of
    the identical kernels -- the same construction `nm_source='off'` and
    `da_source='off'` use, and for the same reason.
    """
    torch.manual_seed(0)
    logits = torch.randn(B, L, V)
    y = _idx(seed=9)
    got = three_factor_loss(logits, y, kappa=0.0, tau=TAU)
    want = F.cross_entropy(logits.reshape(-1, V).float(), y.reshape(-1))
    assert torch.equal(got, want)
    # and it is reached without ever looking at tau, which may be anything
    assert torch.equal(three_factor_loss(logits, y, kappa=0.0, tau=-1.0), want)


def test_t2_weights_reject_a_kappa_that_could_delete_a_position():
    """At |kappa| >= 1 a weight reaches 0 -- deleting a position from the
    gradient -- or goes negative, which rewards the model for getting it
    wrong. Caught at construction, not three hours in."""
    da = torch.tanh(torch.randn(B, L) * 4.0)     # the squash's own range
    for kappa in (1.0, -1.0, 1.5, -2.0):
        with pytest.raises(ValueError, match="kappa must be in"):
            three_factor_weights(da, kappa)
    # Inside the range the guarantee is strict positivity, at either sign and
    # at a saturated DA -- which is where the bound is actually load-bearing.
    saturated = torch.tensor([[-1.0, 1.0]])
    for kappa in (0.99, -0.99, 0.5, -0.5):
        assert float(three_factor_weights(saturated, kappa).min()) > 0.0
        assert float(three_factor_weights(da, kappa).min()) > 0.0


def test_t2_the_weight_is_detached_even_when_its_input_is_not():
    """`w` is a function of the model's own logits. Undetached, the model could
    lower its loss by manipulating the weights rather than by predicting better
    -- and it would. A modulator the modulated system can edit is not one."""
    da = torch.randn(B, L, requires_grad=True)
    w = three_factor_weights(da, 0.5)
    assert w.requires_grad is False
    assert w.grad_fn is None


def test_t2_the_detachment_changes_the_gradient_and_is_therefore_load_bearing():
    """Not a style choice: the attached form has a different gradient.

    Asserted rather than argued, because a `.detach()` that was silently dropped
    would still train, would still produce a plausible bpc, and would be
    measuring the model's ability to edit its own objective.
    """
    torch.manual_seed(0)
    logits = torch.randn(B, L, V, requires_grad=True)
    y = _idx(seed=9)
    three_factor_loss(logits, y, kappa=-0.5, tau=TAU).backward()
    detached_grad = logits.grad.clone()

    attached = logits.detach().clone().requires_grad_(True)
    flat = attached.reshape(-1, V)
    per = F.cross_entropy(flat, y.reshape(-1), reduction="none")
    logp = F.log_softmax(flat, dim=-1)
    entropy = -(logp.exp() * logp).sum(-1)
    w = 1.0 + (-0.5) * torch.tanh((entropy - per) / TAU)      # NOT detached
    ((per * w).sum() / w.sum()).backward()
    assert not torch.allclose(detached_grad, attached.grad, atol=1e-6)


def test_t2_tau_must_be_positive_once_the_weighting_is_live():
    with pytest.raises(ValueError, match="tau must be > 0"):
        three_factor_loss(torch.randn(B, L, V), _idx(), kappa=0.5, tau=0.0)


# ---------------------------------------------------------------------------
# T3 -- alignment. The one thing here that must not be wrong.
# ---------------------------------------------------------------------------

def test_t3_the_loss_is_the_documented_formula_recomputed_independently():
    """Against a reference written from the docstring, not from the source.

    Bitwise, because the two compute the same expression in the same order --
    which is worth knowing: it means the implementation carries no undocumented
    epsilon, clamp or mask. (It carries no NaN masking either, on purpose: a
    diverged model must reach the run's divergence scan rather than hide behind
    a healthy-looking arm.)
    """
    torch.manual_seed(0)
    logits = torch.randn(B, L, V)
    y = _idx(seed=9)
    for kappa in (0.5, -0.5, 0.25):
        want, _ = _reference_three_factor(logits, y, kappa=kappa, tau=TAU)
        assert torch.equal(three_factor_loss(logits, y, kappa=kappa, tau=TAU),
                           want), f"kappa={kappa}"


def test_t3_mutation_leg_the_dopamine_arms_indexing_gives_a_different_loss():
    """`snn.dopamine.rpe`'s shift, applied here, must not be a no-op.

    That is the whole reason this is a separate function rather than a call into
    `snn.dopamine`: weighting position `t` by the error made at `t-1` would
    still train, would still look plausible, and would be measuring a different
    arm. If the two agreed, T3 above could not tell them apart.
    """
    torch.manual_seed(0)
    logits = torch.randn(B, L, V)
    y = _idx(seed=9)
    aligned = three_factor_loss(logits, y, kappa=-0.5, tau=TAU)

    flat = logits.reshape(-1, V).float()
    per = F.cross_entropy(flat, y.reshape(-1), reduction="none")
    logp = F.log_softmax(flat, dim=-1)
    entropy = -(logp.exp() * logp).sum(-1)
    da = torch.tanh((entropy - per) / TAU).reshape(B, L)
    shifted = F.pad(da[:, :-1], (1, 0)).reshape(-1)          # MUTANT: rpe's shift
    w = (1.0 + (-0.5) * shifted).detach()
    misaligned = (per * w).sum() / w.sum()
    assert not torch.allclose(aligned, misaligned, atol=1e-5)


def _one_bad_position(bad_b: int = 1, bad_t: int = 5):
    """Confident and right everywhere except one position, which is confident
    and WRONG. The RPE at that position is large and negative; everywhere else
    it is ~0, because a confident correct prediction has both a small surprise
    and a small entropy."""
    y = _idx(seed=4)
    logits = torch.full((B, L, V), -8.0)
    logits.scatter_(-1, y.unsqueeze(-1), 8.0)
    logits[bad_b, bad_t] = -8.0
    logits[bad_b, bad_t, (int(y[bad_b, bad_t]) + 1) % V] = 8.0
    return logits, y, (bad_b, bad_t)


def test_t3_a_confidently_wrong_position_carries_the_largest_weight():
    """The behavioural half of the alignment check.

    `kappa < 0` up-weights positions that went WORSE than the model expected, so
    the weight must peak at the position that went wrong -- at THAT position and
    not at the one after it, which is what a shifted RPE would produce. The
    weights are recomputed independently (`_reference_three_factor`) and the
    loss they imply is checked against `three_factor_loss`, so this pins the
    alignment inside the function and not merely at its output.
    """
    logits, y, (bad_b, bad_t) = _one_bad_position()
    loss, w = _reference_three_factor(logits, y, kappa=-0.5, tau=TAU)
    assert torch.equal(three_factor_loss(logits, y, kappa=-0.5, tau=TAU), loss)

    w = w.reshape(B, L)
    assert divmod(int(w.argmax()), L) == (bad_b, bad_t)
    assert float(w[bad_b, bad_t]) > 1.4          # tanh is saturated: 1 + |kappa|
    others = torch.ones(B, L, dtype=torch.bool)
    others[bad_b, bad_t] = False
    assert float(w[others].max()) < 1.01         # everything else is ~1


def test_t3_kappa_reverses_the_story_and_both_directions_are_real():
    """`kappa > 0` up-weights positions that went BETTER than expected. Both are
    real dopaminergic stories, neither is derivable from the other, and both are
    pre-registered -- so the sign must actually do this."""
    logits, y, (bad_b, bad_t) = _one_bad_position()
    _, w_pos = _reference_three_factor(logits, y, kappa=0.5, tau=TAU)
    assert divmod(int(w_pos.argmin()), L) == (bad_b, bad_t)
    assert float(w_pos.reshape(B, L)[bad_b, bad_t]) < 0.6


def test_t3_the_normalisation_holds_the_gradient_scale_fixed():
    """`sum_t w_t` in the denominator, not `L`.

    Dividing by a constant would let the arm change the effective learning rate
    as a side effect, and what is being measured would be a learning-rate change
    wearing a reweighting's clothes. The two differ here by 25 %, so this is not
    a distinction without a difference.
    """
    torch.manual_seed(0)
    logits = torch.randn(B, L, V)
    y = _idx(seed=9)
    per = F.cross_entropy(logits.reshape(-1, V), y.reshape(-1), reduction="none")
    _, w = _reference_three_factor(logits, y, kappa=0.5, tau=TAU)
    got = three_factor_loss(logits, y, kappa=0.5, tau=TAU)
    by_length = (per * w).sum() / w.numel()
    assert torch.equal(got, (per * w).sum() / w.sum())
    assert not torch.allclose(got, by_length, atol=1e-6)


def test_t3_the_mean_weight_is_one_under_the_models_own_belief():
    """"`E[phi] = 0` so the mean weight is ~1 already, but only ~1."

    The claim holds under the distribution the RPE is zero-mean in, which is the
    model's OWN -- `E_{x~p}[-log p(x)] = H(p)` exactly. So the targets are drawn
    from the logits rather than uniformly: against uniform targets `E[S] = H +
    KL(true||model) > H`, the mean weight comes out at 0.67 at `kappa = 0.5`,
    and a test that asserted ~1 there would be asserting that the random model
    is already right about the data.
    """
    torch.manual_seed(0)
    logits = torch.randn(32, 128, V)
    y = torch.distributions.Categorical(logits=logits).sample()
    _, w = _reference_three_factor(logits, y, kappa=0.5, tau=TAU)
    assert abs(float(w.mean()) - 1.0) < 0.05


def test_t3_the_roll_control_moves_the_weights_across_the_batch_only():
    """The control preserves the marginal and the temporal autocorrelation and
    destroys only the alignment -- `snn.dopamine.roll_across_batch`'s
    construction, including that it cannot leak the future because it moves
    nothing along time."""
    torch.manual_seed(0)
    logits = torch.randn(B, L, V)
    y = _idx(seed=9)
    rolled, w_rolled = _reference_three_factor(logits, y, kappa=0.5, tau=TAU,
                                               roll=True)
    plain, w_plain = _reference_three_factor(logits, y, kappa=0.5, tau=TAU)
    assert torch.equal(three_factor_loss(logits, y, kappa=0.5, tau=TAU,
                                         roll_control=True), rolled)
    assert not torch.equal(rolled, plain)
    assert torch.equal(torch.sort(w_rolled).values, torch.sort(w_plain).values)
    wr, wp = w_rolled.reshape(B, L), w_plain.reshape(B, L)
    for b in range(B):
        assert torch.equal(wr[b], wp[(b - 1) % B])


# ---------------------------------------------------------------------------
# G1 -- the local arm nests the Phase-2 baseline
# ---------------------------------------------------------------------------

def test_g1_nm_off_is_the_baseline_bitwise_forward():
    """`nm_source='off'` skips the modulation by CODE PATH.

    So the nesting does not rest on the float identity `cur*(1.0 + 0.0)`, which
    is not `cur` for a negative zero. `EXP_011` K4 is the precedent for
    asserting a nesting bitwise rather than to 1e-12.
    """
    arm, plain = _pair(nm_source="off")
    arm.train(); plain.train()
    idx = _idx()
    a, _, _ = arm(idx, None)
    b, _, _ = plain(idx, None)
    assert torch.equal(a, b)
    assert arm.nm_active() is False


def test_g1_nm_off_is_the_baseline_bitwise_backward():
    """Every SHARED parameter's gradient, matched BY NAME.

    By name and not by a positional zip: `nm_a.0` sorts between `layers.1.weight`
    and `head.weight`, so zipping two sorted lists would compare the arm's
    modulator parameters against the baseline's projections and pass on 3 of 7.
    `tests/test_dopamine.py`'s header records that exact bug being caught in
    review rather than by a test, which is why it is stated here too.
    """
    arm, plain = _pair(nm_source="off")
    arm.train(); plain.train()
    idx = _idx()
    for m in (arm, plain):
        logits, _, _ = m(idx, None)
        logits.square().mean().backward()
    a = dict(arm.named_parameters())
    b = dict(plain.named_parameters())
    shared = sorted(set(a) & set(b))
    assert len(shared) == 7
    assert [n for n in shared if not torch.equal(a[n].grad, b[n].grad)] == []
    extra = sorted(set(a) - set(b))
    assert extra == ["nm_a.0", "nm_b.0", "nm_gain.0", "nm_log_tau.0"]


def test_g1_nm_off_leaves_the_modulator_out_of_the_graph_entirely():
    """The parameters exist and reach nothing -- `None`, not zero.

    That is what "skipped by code path" has to mean: `off` costs no kernels as
    well as no difference, so the arm's own anchor is free in every sense. A
    zero gradient here instead of `None` would mean the modulation ran and
    cancelled, which is a different (and slower) claim.
    """
    arm, _ = _pair(nm_source="off")
    logits, _, _ = arm(_idx(), None)
    logits.square().mean().backward()
    modulator = {n: p.grad for n, p in arm.named_parameters()
                 if n.startswith("nm_")}
    assert len(modulator) == 4
    assert all(g is None for g in modulator.values()), modulator


def test_g1_a_live_modulator_does_change_the_forward():
    """The mutation leg for G1: at a real gain the arm must not be the baseline,
    or the whole experiment would be measuring the baseline twice.

    Run at 32x64: a spiking forward is quantised by its own threshold, so a
    perturbation of `cur` that flips no spike anywhere is bitwise invisible.
    `tests/test_dopamine.py`'s G1 leg records the same shape requirement for the
    same reason.
    """
    arm, plain = _pair(nm_source="local", nm_gain_init=0.5)
    arm.eval(); plain.eval()
    idx = _idx(b=32, length=64)
    a, _, _ = arm(idx, None)
    b, _, _ = plain(idx, None)
    assert not torch.equal(a, b)


# ---------------------------------------------------------------------------
# G2 -- gradient reachability, and the derived saddle at gain zero
# ---------------------------------------------------------------------------

def test_g2_at_gain_zero_the_predictor_is_at_an_exact_saddle_by_derivation():
    """EXPECTED, NOT A BUG. See the header.

    `cur * (1 + kappa_c * DA)` routes every path to `a`, `b` and `log_tau`
    through `kappa`, so at `kappa = 0` all three gradients are identically zero
    -- not small, exactly zero. `kappa` itself is reachable (`dL/dkappa_c` picks
    up `cur * DA`, which is not zero), so the saddle is escaped on the first
    optimiser step and every modulator parameter is live from step 2.

    This is recorded as a gate because §7.2's screen flags anything exactly
    zero, and `EXP_004` §2.2 is where this project has already met a zero-init
    saddle that was NOT escapable. This one is; the test below proves the exit.
    """
    arm, _ = _pair(nm_source="local", nm_gain_init=0.0)
    arm.train()
    logits, _, _ = arm(_idx(), None)
    logits.square().mean().backward()
    grads = {n: p.grad for n, p in arm.named_parameters() if n.startswith("nm_")}
    assert len(grads) == 4
    for name in ("nm_a.0", "nm_b.0", "nm_log_tau.0"):
        assert grads[name] is not None, f"{name} is not in the graph at all"
        assert float(grads[name].abs().max()) == 0.0, (
            f"{name} is not at the derived saddle: "
            f"{float(grads[name].abs().max()):.3e}")
    assert float(grads["nm_gain.0"].abs().max()) > 0.0, \
        "the sensitivity is unreachable, so the saddle would be permanent"


def test_g2_a_real_gain_makes_every_modulator_parameter_reachable():
    """The exit from the saddle above, at the first non-zero gain.

    §7.2's rule is "flag anything exactly zero or ~100x below". Exactly zero is
    the fatal half -- AdamW's update is identically 0 forever and nothing in the
    logs says so -- and this asserts it is gone.
    """
    arm, _ = _pair(nm_source="local", nm_gain_init=0.1)
    arm.train()
    logits, _, _ = arm(_idx(), None)
    logits.square().mean().backward()
    for name in ("nm_a.0", "nm_b.0", "nm_gain.0", "nm_log_tau.0"):
        grad = dict(arm.named_parameters())[name].grad
        assert grad is not None, f"{name} is not in the graph"
        assert float(grad.abs().max()) > 0.0, f"{name} is an exact saddle"


def _captured_currents(bump_channel_zero: bool):
    """Layer `k`'s input current as the scan actually receives it.

    Observed by wrapping `snn.model.lif_scan` -- `tests/test_dopamine.py`'s G6
    uses the same instrument -- because the quantity this test is about never
    leaves the forward pass. Reading the LOGITS instead would measure nothing at
    init: layer `K-1` is silent (`02_baseline_report.md` §5.2), so the whole
    model emits `head.bias` at every position and any modulation of the last
    layer's current is invisible downstream.
    """
    arm, _ = _pair(nm_source="local", nm_gain_init=0.1)
    if bump_channel_zero:
        with torch.no_grad():
            arm.nm_gain[0][0, 0] += 2.0        # one channel only
    arm.eval()
    seen: list[torch.Tensor] = []
    real = snn.model.lif_scan
    snn.model.lif_scan = lambda cur, *a, **kw: (
        seen.append(cur.detach().clone()), real(cur, *a, **kw))[1]
    try:
        arm(_idx(), None)
    finally:
        snn.model.lif_scan = real
    return seen


def test_g2_the_modulator_is_per_channel_and_not_broadcast_from_one_element():
    """Recovered functionally rather than by shape: a `[1, d]` and a `[d]`
    broadcast alike, and `EXP_002` §8.4's failure was exactly a parameter that
    looked per-channel and was not."""
    base, bumped = _captured_currents(False), _captured_currents(True)
    assert len(base) == K
    assert torch.equal(base[0], bumped[0]), "layer 0 must be unmodulated"
    diff = (bumped[1] - base[1]).abs()
    assert float(diff[..., 0].max()) > 0.0     # the bumped channel moved
    assert float(diff[..., 1:].max()) == 0.0   # and only that one


def test_g2_the_driving_signal_is_detached_from_the_previous_layer():
    """The modulator is a broadcast signal ABOUT layer k-1's activity, not a
    second gradient path into it -- the contract `DopamineCharLM._da` holds for
    the head-driven signal, kept here so that the two arms differ in where the
    signal comes from and in nothing else.

    Observed at the call rather than read out of the source text: what is
    asserted is that the tensor `bernoulli_rpe` receives carries no graph, which
    is the property, and it survives any reformatting of the line that produces
    it.
    """
    arm, _ = _pair(nm_source="local", nm_gain_init=0.1)
    arm.train()
    seen: list[bool] = []
    real = snn.model.bernoulli_rpe
    snn.model.bernoulli_rpe = lambda spikes, *a, **kw: (
        seen.append(spikes.requires_grad), real(spikes, *a, **kw))[1]
    try:
        arm(_idx(), None)
    finally:
        snn.model.bernoulli_rpe = real
    assert seen == [False] * (K - 1), \
        "the modulator opened a second gradient path into the previous layer"


# ---------------------------------------------------------------------------
# G3 -- the control, dispatch and configuration
# ---------------------------------------------------------------------------

def test_g3_the_rolled_control_is_the_dopamine_arms_own_function():
    """Identity of the imported object, not equality of behaviour.

    `snn.neuromod`'s docstring says the control is `snn.dopamine.roll_across_batch`
    "reused unchanged so that this arm's control and EXP_018's are the same
    function and not two functions with the same name". Asserted with `is`,
    because two identical copies would satisfy every behavioural check and would
    be exactly the thing that sentence forbids.
    """
    assert snn.model.roll_across_batch is snn.dopamine.roll_across_batch


def test_g3_rolled_needs_a_batch_of_two():
    """At B = 1 the roll is the identity, so the control would silently BE the
    arm. Caught by Config at construction, not at the first batch."""
    with pytest.raises(ValueError, match="batch_size >= 2"):
        Config(arch="localdopamine", nm_source="rolled", batch_size=1)
    # ...and it is specific to the arm that uses it
    assert Config(arch="snn", nm_source="rolled", batch_size=1).nm_source == \
        "rolled"


def test_g3_the_rolled_control_really_differs_from_the_local_signal():
    """A control that accidentally reproduced the arm would report a null."""
    local, _ = _pair(nm_source="local", nm_gain_init=0.3)
    rolled, _ = _pair(nm_source="rolled", nm_gain_init=0.3)
    idx = _idx(b=32, length=64)
    local.eval(); rolled.eval()
    a, _, _ = local(idx, None)
    b, _, _ = rolled(idx, None)
    assert not torch.equal(a, b)


def test_g3_build_model_dispatches_and_records_its_init():
    cfg = Config(arch="localdopamine", vocab_size=V, d_model=D, n_layers=K,
                 device="cpu", fused=False, deterministic=False,
                 nm_source="local", nm_tau=0.5, nm_gain_init=0.25,
                 nm_a_init=0.1, nm_b_init=-2.97)
    model = build_model(cfg)
    assert isinstance(model, LocalDopamineCharLM)
    assert (model.nm_source, model.nm_tau) == ("local", 0.5)
    assert float(model.nm_gain[0].detach().min()) == 0.25
    assert abs(float(model.nm_a[0].detach().max()) - 0.1) < 1e-7
    assert abs(float(torch.exp(model.nm_log_tau[0].detach())) - 0.5) < 1e-6
    # sigmoid(-2.97) = 0.0488 -- layer 0's MEASURED init firing rate, which is
    # the rate the predictor must start at for phi to start near zero.
    assert abs(1.0 / (1.0 + math.exp(2.97)) - 0.0488) < 1e-3


def test_nm_b_init_is_the_value_the_runs_actually_used():
    """`Config.nm_b_init` is -0.7, and `EXP_021` SS2.6 certifies the arm on -2.97.

    Both statements are true, and until this test existed nothing observed the
    first one. `LocalDopamineCharLM.__init__` defaults to -2.97 -- the value the
    pre-registration measures `mean sech^2 = 0.777` at -- but `build_model`
    forwards `cfg.nm_b_init`, so the class default is dead code and every
    committed `nm_*` run trained at -0.7.

    Three existing mechanisms each checked a neighbouring quantity and missed it:
    `test_g3_build_model_dispatches_and_records_its_init` passes -2.97
    EXPLICITLY and then asserts the arithmetic *of* -2.97;
    `test_g3_the_defaults_nest_phase_two` reads `Config()` for `nm_source`,
    `nm_gain_init` and `tf_kappa` but not for this field; and the driver's K1
    diff records fields that CHANGED against a reference, which this one did not.

    This test asserts the shipped value, asserts that `build_model` is what makes
    it shipped, and puts the resulting init saturation in a gate so that the
    number is observed rather than merely derivable. It does NOT argue for
    changing the value: -0.7 is what every committed number used.
    """
    # 1. The shipped value, pinned.
    assert Config().nm_b_init == -0.7

    # 2. The class default disagrees, and Config is what wins.
    class_default = inspect.signature(
        LocalDopamineCharLM.__init__).parameters["nm_b_init"].default
    assert class_default == -2.97
    assert class_default != Config().nm_b_init, (
        "if these ever agree, delete this branch of the test rather than "
        "loosening it -- the discrepancy is the thing being pinned")
    cfg = Config(arch="localdopamine", vocab_size=V, d_model=D, n_layers=K,
                 device="cpu", fused=False, deterministic=False,
                 nm_source="local", nm_tau=0.057321)
    model = build_model(cfg)
    # `b` is [1, d] filled from the init; Config's value reaches the parameter.
    assert abs(float(model.nm_b[0].detach().max()) - (-0.7)) < 1e-6

    # 3. What that does to the squash at INITIALISATION, asserted as a number.
    #    With `nm_a_init = 0.0` the init logit is `b` exactly, so
    #        phi = mean_c (s_c - sigmoid(b)) * b = (rbar - sigmoid(b)) * b.
    #    `rbar` = 0.0488 is layer 0's measured init rate (audit_07), the same
    #    figure SS2.5 uses. This is the module docstring's failure mode 2 --
    #    "the predictor's gradient vanishes and cur*(1+kappa*DA) degenerates
    #    into a constant per-channel gain" -- reached through the PRIOR instead
    #    of through a summed reduction. D2 gates the summed route; nothing gated
    #    this one.
    tau, rbar = 0.057321, 0.0488
    def sech2(b):
        phi = (rbar - 1.0 / (1.0 + math.exp(-b))) * b
        return 1.0 - math.tanh(phi / tau) ** 2

    assert sech2(-2.97) > 0.99, "SS2.6's certified init is unsaturated"
    assert sech2(-0.7) < 0.01, "the shipped init is saturated at step 0"
    # The attenuation against the point SS2.6 certifies, to one significant
    # figure. Two orders of magnitude, and it is reported rather than rounded
    # away: EXP_023 SS9 and report rev 14 both cite this ratio.
    assert 100.0 < sech2(-2.97) / sech2(-0.7) < 1000.0


def test_g3_the_defaults_nest_phase_two():
    """A run that forgets to set `nm_source` trains the baseline, not something
    undocumented. The same rule as `noise_amp = 0.0` and `da_gain_init = 0.0`."""
    assert Config().nm_source == "off"
    assert Config().nm_gain_init == 0.0
    assert Config().tf_kappa == 0.0
    # The ladder EXP_023 runs, in order of increasing signal content. `off`
    # must stay first because it is the rung a forgotten flag lands on.
    assert NM_SOURCES == ("off", "const", "pos", "rolled", "local")
    assert NM_SOURCES[0] == "off"


def test_g3_config_choices_are_the_modules_own_tuple():
    """Two tuples naming the same set is how they stop agreeing. `NM_SOURCE_CHOICES`
    is what `Config` validates against and `NM_SOURCES` is what the model
    validates against, so a rung added to one and not the other is a config that
    constructs and a model that raises three hours in."""
    from snn.config import NM_SOURCE_CHOICES
    assert NM_SOURCE_CHOICES == NM_SOURCES
    for src in NM_SOURCES:
        # Every rung must be constructible through the config path it will
        # actually be launched by, not only through the class.
        cfg = Config(arch="localdopamine", d_model=D, n_layers=K, vocab_size=V,
                     nm_source=src, seq_len=8, nm_pos_len=8, device="cpu")
        assert cfg.nm_source == src


@pytest.mark.parametrize("source,per_layer", [
    ("off", 3 * D + 1), ("local", 3 * D + 1), ("rolled", 3 * D + 1),
    ("const", D), ("pos", D + 8),
])
def test_g4_every_rung_realises_its_own_closed_form_exactly(source, per_layer):
    """G2 asserts realised == closed form EXACTLY, so a rung that allocates a
    predictor it never reads would abort the ladder -- or worse, pass a
    width-matching gate while describing a model that is not the one that ran.

    `off` deliberately KEEPS the predictor: its nesting claim is about the code
    path, and carrying the parameters is what lets G2 assert they reach nothing.
    The scored rungs do not, because AdamW decays what it is handed."""
    model = LocalDopamineCharLM(V, D, K, fused=False, nm_source=source,
                                nm_pos_len=8)
    realised = sum(p.numel() for p in model.parameters())
    closed = snn.model.local_dopamine_param_count(V, D, K, source=source,
                                                  pos_len=8)
    assert realised == closed
    assert closed - snn.model.spiking_param_count(V, D, K) == per_layer * (K - 1)


def test_g4_the_const_rung_is_a_learned_per_channel_gain_and_nothing_else():
    """`DA = 1` makes `cur * (1 + kappa_c)` a static per-channel gain, which by
    EXP_008 Identity 1 is a learned per-channel THRESHOLD -- adopted arm #5,
    measured at 0.0590 bpc. The rung exists precisely so that EXP_021's 0.0337
    headline can be read against it, so the constancy is asserted rather than
    assumed."""
    model = LocalDopamineCharLM(V, D, K, fused=False, nm_source="const",
                                nm_gain_init=0.1)
    idx = _idx()
    model.keep_spikes = True
    _, _, aux = model(idx, None)
    for da in aux["da_mean"]:
        assert float(da) == 1.0
    assert len(model.nm_a) == 0 and len(model.nm_b) == 0
    assert len(model.nm_log_tau) == 0 and len(model.nm_pos) == 0


def test_g4_the_pos_rung_is_data_independent_and_covers_the_window():
    """`pos` is the signal `rolled` LEAKS: batch-rolling preserves the time
    index, so a misaligned RPE still carries window position. If the same DA
    can be produced from position alone, the alignment was never the mechanism.

    Two properties make it that control: DA must not depend on the input at
    all, and it must raise rather than wrap when the window outruns the table."""
    model = LocalDopamineCharLM(V, D, K, fused=False, nm_source="pos",
                                nm_gain_init=0.1, nm_pos_len=L)
    with torch.no_grad():
        for p in model.nm_pos:
            p.normal_(0.0, 0.5)
    a = model._driving_da(torch.rand(B, L, D), 1)
    b = model._driving_da(torch.rand(B, L, D), 1)
    assert torch.equal(a, b)                      # data-independent, exactly
    assert torch.equal(a[0], a[-1])               # and identical across batch
    narrow = LocalDopamineCharLM(V, D, K, fused=False, nm_source="pos",
                                 nm_gain_init=0.1, nm_pos_len=L - 1)
    with pytest.raises(ValueError, match="positions but the window"):
        narrow(_idx(), None)


def test_g4_the_pos_rung_unlocks_kappa_in_two_steps_rather_than_never():
    """At `w = 0`, `DA` is identically zero, so `dL/dkappa` is EXACTLY zero --
    which looks like EXP_004 §2.2's saddle and is not one. `w`'s own gradient
    is `kappa * cur * dL/dcur` and is nonzero at `kappa = 0.1`, so `w` moves at
    step 1 and `kappa` becomes reachable at step 2. Asserted rather than
    reasoned about, because the difference between a two-step unlock and a dead
    parameter is a whole arm."""
    torch.manual_seed(0)
    model = LocalDopamineCharLM(V, D, K, fused=False, nm_source="pos",
                                nm_gain_init=0.1, nm_pos_len=L)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    idx, tgt = _idx(), _idx()
    grads = []
    for _ in range(3):
        opt.zero_grad()
        logits, _, _ = model(idx, None)
        F.cross_entropy(logits.reshape(-1, V), tgt.reshape(-1)).backward()
        grads.append((float(model.nm_gain[0].grad.abs().sum()),
                      float(model.nm_pos[0].grad.abs().sum())))
        opt.step()
    assert grads[0][0] == 0.0        # kappa unreachable at step 0, exactly
    assert grads[0][1] > 0.0         # w is not
    assert grads[1][0] > 0.0         # and kappa is reachable by step 1


@pytest.mark.parametrize("kw,match", [
    ({"nm_source": "shuffle"}, "nm_source"),
    ({"nm_tau": 0.0}, "nm_tau must be > 0"),
])
def test_g3_the_module_rejects_impossible_settings(kw, match):
    with pytest.raises(ValueError, match=match):
        LocalDopamineCharLM(V, D, K, fused=False, **kw)


def test_g3_config_rejects_a_kappa_that_could_delete_a_position():
    with pytest.raises(ValueError, match=r"tf_kappa must be in"):
        Config(tf_kappa=1.0)
    with pytest.raises(ValueError, match="unweighted objective"):
        Config(tf_rolled=True, tf_kappa=0.0)
    with pytest.raises(ValueError, match="batch_size >= 2"):
        Config(tf_rolled=True, tf_kappa=0.5, batch_size=1)


def test_g3_state_round_trips_under_protocol_b():
    arm, _ = _pair(nm_source="local", nm_gain_init=0.1)
    arm.eval()
    idx = _idx()
    _, state, aux = arm(idx, None)
    assert len(state) == K and all(s.shape == (B, D) for s in state)
    assert len(aux["da_mean"]) == K - 1
    logits, state2, _ = arm(idx, [s.detach() for s in state])
    assert logits.shape == (B, L, V) and len(state2) == K


# ---------------------------------------------------------------------------
# G4 -- accounting. Layer 0 is unmodulated by construction.
# ---------------------------------------------------------------------------

def test_g4_layer_zero_carries_no_modulator():
    """Structural, not a special case: layer 0 has no previous layer to read.

    Driving it from the input code would make the arm's first layer a different
    mechanism from its others, which is the thing decision #3's admission is
    narrow about.
    """
    for layers in (1, 2, 4):
        torch.manual_seed(0)
        arm = LocalDopamineCharLM(V, D, layers, fused=False, nm_source="local")
        assert len(arm.nm_a) == layers - 1
        assert len(arm.nm_b) == len(arm.nm_gain) == len(arm.nm_log_tau) == \
            layers - 1


def test_g4_the_realised_parameters_are_the_baseline_plus_the_modulator():
    """Realised, and itemised. `count_params` walks the module tree, so this is
    what the run actually optimises rather than what a formula says it does.

    Per modulated layer: `a`, `b` and `kappa` at `[1, d]` each, plus ONE scalar
    `log_tau` -- `tau` is initialised from the calibration and then LEARNED,
    which is `EXP_021`'s deliberate departure from `EXP_018`.
    """
    torch.manual_seed(0)
    arm = LocalDopamineCharLM(V, D, K, fused=False, nm_source="local")
    extra = count_params(arm) - spiking_param_count(V, D, K)
    assert extra == (3 * D + 1) * (K - 1)


def test_g4_the_closed_form_matches_the_realised_count():
    """The gate every parameter-matched arm depends on.

    This test shipped as `xfail(strict=True)` against a real defect: the closed
    form omitted `nm_log_tau`, one learned scalar per modulated layer, so it sat
    `n_layers - 1` below the realised count -- 736,973 against 736,974 at the
    committed shape. `EXP_021`'s G2 asserts the two are EQUAL, not close, and
    would have aborted the ladder on its first run. The marker came off when the
    closed form was corrected, which is what strict xfail is for.

    `snn.model` uses these closed forms to width-match the arms against each
    other, and `snn.model.InterfaceCharLM`'s own docstring states the standard:
    a formula that disagrees with the model it describes makes every
    parameter-matching gate "describe a model that is not the one that ran".
    Here the disagreement is `n_layers - 1` parameters -- 1 of 736,973 at the
    committed shape, so it changes no matched width -- but the assertion is
    written as the contract rather than as the current behaviour.
    """
    for shape in ((37, 16, 2), (97, 32, 3), (205, 512, 2)):
        vocab, d, layers = shape
        torch.manual_seed(0)
        arm = LocalDopamineCharLM(vocab, d, layers, fused=False)
        assert count_params(arm) == local_dopamine_param_count(vocab, d, layers), \
            shape


def test_g4_the_modulator_is_o_of_d_against_an_o_of_d_squared_stack():
    """Which is why width-matching this arm IS parameter-matching it -- 0.2 % at
    the committed shape, on either accounting."""
    committed = spiking_param_count(205, 512, 2)
    torch.manual_seed(0)
    realised = count_params(LocalDopamineCharLM(205, 512, 2, fused=False))
    assert (realised - committed) / committed < 0.003
    assert (local_dopamine_param_count(205, 512, 2) - committed) / committed \
        < 0.003


# ---------------------------------------------------------------------------
# CUDA -- the fused kernel is the one the arms actually run
# ---------------------------------------------------------------------------

@pytest.mark.cuda
@requires_cuda
def test_cuda_the_nesting_is_bitwise_on_the_fused_kernel_too():
    """G1 on CPU exercises the eager reference. The runs use the jiterator."""
    torch.manual_seed(0)
    arm = LocalDopamineCharLM(V, D, K, fused=True, nm_source="off").cuda()
    torch.manual_seed(0)
    plain = SpikingCharLM(V, D, K, fused=True).cuda()
    plain.load_state_dict({k: v.clone() for k, v in arm.state_dict().items()
                           if not k.startswith("nm_")})
    idx = _idx().cuda()
    a, _, _ = arm(idx, None)
    b, _, _ = plain(idx, None)
    assert torch.equal(a, b)


@pytest.mark.cuda
@requires_cuda
def test_cuda_fused_and_eager_agree_on_the_live_arm():
    """R10 is inherited rather than new -- the arm's own path is asserted
    anyway, because the modulation multiplies the current the kernel reads."""
    torch.manual_seed(0)
    fused = LocalDopamineCharLM(V, D, K, fused=True, nm_source="local",
                                nm_gain_init=0.2).cuda()
    torch.manual_seed(0)
    eager = LocalDopamineCharLM(V, D, K, fused=False, nm_source="local",
                                nm_gain_init=0.2).cuda()
    eager.load_state_dict(fused.state_dict())
    idx = _idx().cuda()
    a, _, _ = fused(idx, None)
    b, _, _ = eager(idx, None)
    assert torch.equal(a, b)
