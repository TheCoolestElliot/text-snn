"""Dopamine, moved to where dopamine acts.

Pre-registered in `experiments/logs/EXP_021_neuromodulation.md`. Read that
first. This file is a response to a specific measured result, so it starts from
that result rather than from an idea.

WHAT EXP_018 ESTABLISHED, AND WHAT IT LEFT
--------------------------------------------
`EXP_018` built the reward prediction error `phi_t = H(p_{t-1}) + log p_{t-1}(x_t)`
-- zero-mean under the model's own belief **by construction**, no baseline to
tune -- squashed it to `DA_t = tanh(phi/tau)`, and multiplied it into every
layer's input current with a learned per-channel sensitivity `k_c`. Three
findings, all of which this file is built on:

  1. **The signal is real and the optimiser wants it.** It assigns the aligned
     RPE `rms(k) = 0.195` and the batch-misaligned control `0.0075` -- a factor
     of **26**. The prediction error IS information the model can find.
  2. **Paying attention to it costs 0.0062 bpc on 5 of 5 seeds** (t = -7.48,
     p = 1.7e-03), and it buys no extra training fit while losing at test, so
     the train->test gap *widens* by 0.0057. A capability loss, not overfitting.
  3. **It is head-driven, so it is permanently a diagnostic** (decision #3,
     ruled 2026-08-14): the head is downstream of every layer, so the signal
     cannot be supplied inline and needs a second forward pass -- measured at
     **1.71x** wall-clock, at inference as well as at training.

So the arm did not fail for lack of signal. It failed because of **where the
signal was injected**. This file changes that, in two independent ways, and
neither of them is a re-tuning of `EXP_018`.

--------------------------------------------------------------------------
FORM 1: `local_rpe` -- the same construction, driven by the previous layer
--------------------------------------------------------------------------
Decision #3 **admits** a modulator driven by layer `k-1`'s output, because that
activity is already computed before layer `k`'s time loop begins, so the signal
costs O(1) kernels and preserves the depth-sequential evaluation. `EXP_018`
§10's own referral names a feedforward data-dependent arm as the obvious
Phase-5 candidate. This is that arm, built with `EXP_018`'s derivation rather
than a new one.

Layer `k-1` emits a binary vector `s_t`. Give each channel a **diagonal**
one-step predictor of its own next spike --

    p_{t,c} = sigmoid(a_c * s_{t-1,c} + b_c)          2 params per channel

-- and the identical three quantities follow, per position, over the channel
axis instead of over the vocabulary axis:

    S_t   = -mean_c [ s log p + (1-s) log(1-p) ]   the surprise  (nats/channel)
    H_t   =  mean_c H_bernoulli(p_{t,c})            its OWN expectation
    phi_t =  H_t - S_t = mean_c (s - p)*logit       zero-mean by construction
    DA_t  =  tanh(phi_t / tau)                      bounded, in (-1, 1)

The third line's right-hand side is an exact identity, not an approximation,
and it is what the code computes: the two terms share a `log(1-p)` that
cancels. Computing `H` and `S` separately measured **2.01x the anchor's
wall-clock**, which would have cost this arm its entire cost argument.

The reduction is a MEAN, so `phi` is O(1) in the width and `tau` is a property
of the task rather than of `d`. Summed it is O(d) and saturates the squash at
any `tau` of order one, which is a silent failure -- see `bernoulli_rpe`.

**Why the entropy is again the right baseline, and again decides the
implementation.** `E_{s~Bern(p)}[-log P(s)] = H_b(p)` exactly, so no EMA, no
critic, no batch statistic and nothing to tune -- and, as in `EXP_018`, the
alternative (a running mean over `t`) is a **sequential scan**, `L` kernels on
the one code path this architecture exists to keep at one kernel per timestep.
The entropy is a parallel reduction. Same derivation, same consequence.

**Why the predictor is diagonal, and why that is not a compromise.** A `[d, d]`
predictor applied to `s_{t-1}` is `W_rec @ s_{t-1}`, which is **exactly the form
I5 forbids** -- lateral mixing, memory in a weight matrix rather than in state.
`a_c` and `b_c` are per-channel and O(1) per neuron, which the 2026-08-01 I5
ruling explicitly admits. The constraint is doing real work here: it picked the
predictor.

**Causality.** `p_t` is built from `s_{t-1}` and scored against `s_t`; both
depend on `x_{<=t}`, and the target at `t` is `x_{t+1}`. `y` is not an argument
to anything in this file. `tests/test_neuromod.py`'s C1 asserts it by
perturbation, as `tests/test_dopamine.py`'s G2 does.

**Three things this fixes that were structural in `EXP_018`, not incidental:**

  * **No second forward pass.** The driving signal is an activation the forward
    pass has already produced. The 1.71x goes away.
  * **It is active at zero context.** `EXP_018` §9's decomposition found its
    modulation density grows with context and is structurally absent at `c = 0`,
    where `EXP_004` §10.6 says the cheap gains have been. `phi_t` here is
    defined from `t = 1`, and `p_t` at `t = 0` is `sigmoid(b_c)` -- a real
    prediction from a real prior, not a zero.
  * **It is adoptable.** Under decision #3 this shape is a candidate, not a
    labelled diagnostic.

Layer 0 has no previous layer and is therefore **unmodulated**, by construction
rather than by a special case. Driving layer 0 from the input code would make the
arm's first layer a different mechanism from its others.

`tau` is initialised from calibration and then **learned**, one scalar per
modulated layer. That departs from `EXP_018`, which froze it, and the reason is
a property of this signal: `EXP_018`'s surprise is over a 205-way vocabulary
whose scale is near-stationary, while this one is over a population whose firing
rate rises **7x** during training (0.049 at init to 0.34 at convergence, measured
in `audit_07` and `EXP_000` F3 respectively). A frozen `tau` that is right at
step 0 is wrong at step 20,000, and a saturated `tanh` is the one failure this
arm cannot survive.

--------------------------------------------------------------------------
FORM 2: `three_factor` -- dopamine on the learning rule, not on the forward
--------------------------------------------------------------------------
`EXP_018` put a neuromodulator on the forward current. In the brain, the
best-established action of dopamine is not moment-to-moment gain: it is a
**third factor gating plasticity** -- `dw ~ pre x post x DA`. Nothing in this
project has ever tested that, and it is the reading under which `EXP_018`'s
result is not a refutation of the signal but a statement about the slot.

    w_t  = 1 + kappa * DA_t         (detached)
    loss = sum_t w_t * CE_t / sum_t w_t

Under BPTT the per-position gradient IS the eligibility trace, so weighting the
per-position loss is the three-factor rule written in the only form this
trainer can express. `kappa < 0` up-weights positions that went **worse** than
the model expected; `kappa > 0` up-weights positions that went **better**. Both
are real dopaminergic stories and neither is derivable from the other, so both
are pre-registered as arms, exactly as `EXP_018` pre-registered `mult` and `add`.

**Four properties that follow, and that make this the sharper test:**

  * **The forward pass is not touched at all.** At eval the model is the Phase-2
    baseline **bitwise**, by code path -- so unlike `EXP_018` this arm
    *structurally cannot* cost capability at inference. Whatever it does, it
    does to the optimiser.
  * **There is no inference cost.** Not 1.71x, not 1.0x-and-a-caveat: the
    weighting exists only inside `Trainer._step_body`.
  * **The function class is unchanged**, so this is not an I5 or decision-#3
    question at all. It is a training objective, and this project already ships
    one: `snnchat.train` weights `<|bot|>` turns by `bot_loss_weight` with the
    identical `(per_char * w).sum() / w.sum()` normalisation. The precedent is
    in the tree.
  * **`w` is DETACHED, and that is load-bearing.** Undetached, the model could
    lower its loss by manipulating the weights rather than by predicting better
    -- and it would, because `w` is a function of its own logits. A modulator
    that the modulated system can edit is not a modulator.

`sum_t w_t` in the denominator rather than `L`: with `E[phi] = 0` the mean
weight is ~1 already, but only ~1, and dividing by a constant would let the arm
change the effective learning rate as a side effect. Normalising by the realised
weight sum holds the gradient scale fixed so that what is measured is the
*reweighting* and not a learning-rate change wearing its clothes.

WHAT IS NOT DONE, AND WHY
--------------------------
**`kappa` is not swept and `EXP_018`'s `tau` is not reused.** `EXP_018` fixed
`tau = 1.1291` from four checkpoints agreeing to 3.1 %, and that constant is
reused unchanged wherever the head-driven signal appears. `local_rpe`'s `tau` is
a different quantity in different units -- nats per channel over `d` channels,
not nats over `V` symbols -- so it is **calibrated separately** by
`scripts/exp/021_calibrate_nm.py`, measured and not chosen, exactly as
`EXP_018`'s was. Inheriting it would be the mistake `CONTRIBUTING.md` §4 names:
a constant is re-measured on every structurally new arm, never inherited.

**No batch statistics.** Not a mean, not an sd, not a running buffer. Same rule
and same reason as `snn.dopamine`: it would make one example's output depend on
which examples shared its batch, and the fresh/carried protocols would then
measure different functions.

**Non-finite values are not masked.** A diverged model produces NaN logits,
hence NaN `phi`, hence NaN weights, and the NaN reaches the loss where the run's
own divergence scan sees it. Clamping here would hide a divergence behind a
healthy-looking arm.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from snn.prescan import shift_by_one

#: Where the local modulator's driving signal comes from.
#:
#: THE FOUR ACTIVE VALUES ARE A LADDER OF DECREASING SIGNAL CONTENT IN ONE
#: FIXED ALGEBRAIC FORM. `EXP_021` established that `cur * (1 + kappa_c * DA_t)`
#: is worth ~0.034 bpc with `DA` the aligned local RPE, and ~0.028 with `DA`
#: the batch-misaligned one -- so the form pays and the alignment adds at most
#: 0.006, which that design could not resolve. `EXP_022`/`EXP_023` therefore
#: hold the form EXACTLY fixed and vary only what drives it:
#:
#: ``off``     the arm is the Phase-2 baseline, by code path.  DA is not built.
#: ``const``   DA = 1, a constant. `cur * (1 + kappa_c)` is a learned
#:             per-channel gain on the current, which by `EXP_008` Identity 1
#:             is exactly a learned per-channel THRESHOLD -- adopted arm #5,
#:             measured at 0.0590 bpc on a different tree. This rung exists
#:             because that number is LARGER than `EXP_021`'s headline, so
#:             without it the project cannot say whether the local modulator
#:             beat its anchor by being time-varying or merely by being a gain.
#: ``pos``     DA = tanh(w_t), a learned scalar per WINDOW POSITION and nothing
#:             else. Time-varying, data-INDEPENDENT. It is the signal
#:             ``rolled`` leaks: batch-rolling preserves the time index, so a
#:             misaligned RPE still carries "how far into the window am I",
#:             which is the strongest single predictor of surprise there is.
#:             DIAGNOSTIC ONLY and structurally non-deployable -- window
#:             position is a training artefact and means nothing at decode.
#: ``rolled``  THE CONTROL -- `snn.dopamine.roll_across_batch`, reused
#:             unchanged so that this arm's control and EXP_018's are the
#:             same function and not two functions with the same name.
#:             Data-dependent, time-aligned, content-misaligned.
#: ``local``   the real signal: layer k-1's own Bernoulli prediction error.
#:
#: Read as a ladder: ``const`` < ``pos`` < ``rolled`` < ``local`` in what the
#: driving signal knows. Whatever the first rung already buys is not evidence
#: for any rung above it.
NM_SOURCES = ("off", "const", "pos", "rolled", "local")

__all__ = [
    "NM_SOURCES",
    "bernoulli_rpe",
    "three_factor_weights",
    "three_factor_loss",
]


def bernoulli_rpe(spikes: Tensor, a: Tensor, b: Tensor) -> Tensor:
    """The local reward prediction error `phi`, in nats PER CHANNEL. `[B, L]`.

    `spikes` `[B, L, d]` in {0, 1}; `a`, `b` `[1, d]`.

        p_t   = sigmoid(a * s_{t-1} + b)
        S_t   = BCE(p_t, s_t)  averaged over channels
        H_t   = mean_c H_bernoulli(p_{t,c})
        phi_t = H_t - S_t

    **The reduction is a MEAN and not a sum, and that is not cosmetic.** Summed,
    `phi` is O(d): measured at `d = 512` with the predictor at its prior it is
    **+101 nats**, and `tanh(101/tau)` is `1.0` to fp32 for any `tau` of order
    one. A saturated squash does two things, both fatal and neither loud. The
    gradient through `tanh` vanishes, so the predictor `(a, b)` stops learning;
    and a constant `DA = 1` makes `cur*(1 + kappa*DA)` a **constant per-channel
    gain**, which by `EXP_008` Identity 1 is exactly a learned per-channel
    threshold -- an arm this project has already adopted (#5). The saturated
    arm would therefore have trained, scored plausibly, and been measuring
    `EXP_008` under a different name.

    Meaned, `phi` is nats per channel and is O(1) in the width, so `tau` is a
    property of the task rather than of `d` -- the same claim `EXP_018` makes
    for `da_scale`, which agreed to 3.1 % across three architectures, and which
    matters here because `EXP_014`/`EXP_015` run arms at three widths.

    THE FORM THIS IS COMPUTED IN, AND WHY IT IS NOT THE FORM ABOVE.
    Written literally, `H - S` needs `log p`, `log(1-p)`, `p` and two weighted
    reductions -- about eight elementwise passes over a `[B, L, d]` tensor per
    layer, plus their backward. At 735K parameters this model is
    memory-bandwidth-bound, not FLOP-bound, and the pre-flight measured that
    literal form at **2.01x the anchor's wall-clock** -- worse than the 1.71x of
    the head-driven arm it exists to beat, which would have refuted the arm's
    entire cost argument.

    The two terms share their `log(1-p)`, and it cancels exactly. Using
    `log p = log(1-p) + logit`:

        S = -[log(1-p) + s*logit]
        H = -[log(1-p) + p*logit]
        phi = H - S = (s - p) * logit                      EXACTLY

    so

        phi_t = mean_c (s_{t,c} - p_{t,c}) * logit_{t,c}

    which is four passes, contains no logarithm at all, and is the general
    exponential-family identity `H - S = (s - E[s]) . eta` written for a
    Bernoulli. It also reads as what it is: the RPE is the prediction error
    projected onto the predictor's own confidence. A channel the predictor is
    unsure about (`logit ~ 0`) contributes nothing however wrong it turns out to
    be, and a confident channel contributes in proportion to how wrong it was.

    Because there is no logarithm there is nothing to underflow, so the
    `logsigmoid` guard the literal form needed is not needed here. Computed in
    fp32 regardless of `cfg.dtype`, which selects the GEMM dtype only.

    Position 0 is NOT zeroed. `snn.dopamine.rpe` zeroes its position 0 because
    there is no preceding *prediction* there; here there is one -- `s_{-1}` is
    the zero vector by `shift_by_one`'s padding, so `p_0 = sigmoid(b)` is the
    channel's own prior rate, which is a real prediction and is scored as one.
    """
    if spikes.dim() != 3:
        raise ValueError(f"spikes must be [B, L, d]; got {tuple(spikes.shape)}")
    d = spikes.shape[2]
    for name, t in (("a", a), ("b", b)):
        if t.dim() != 2 or t.shape[0] != 1 or t.shape[1] != d:
            raise ValueError(f"{name} must be [1, {d}]; got {tuple(t.shape)}")

    s = spikes.float()
    logit = a * shift_by_one(s) + b                      # [B, L, d]
    # phi = H - S = (s - p) * logit, exactly. See the docstring: the literal
    # H and S share a log(1-p) that cancels, and computing them separately
    # costs 2x the model's whole training step for a quantity that is four
    # elementwise ops.
    return ((s - torch.sigmoid(logit)) * logit).mean(-1)         # [B, L]


def three_factor_weights(da: Tensor, kappa: float) -> Tensor:
    """`w = 1 + kappa * DA`, detached. `[B, L]`, strictly positive for |kappa| < 1.

    Detached here rather than at the call site so that no caller can forget:
    `da` is a function of the model's own output, and an attached weight is a
    term the model can optimise instead of the objective.
    """
    if not -1.0 < kappa < 1.0:
        # At |kappa| >= 1 the weight can reach 0 (or go negative), which would
        # delete a position from the gradient entirely -- or reward the model
        # for getting it wrong. Caught at construction, not three hours in.
        raise ValueError(
            f"kappa must be in (-1, 1) so weights stay positive; got {kappa!r}"
        )
    return (1.0 + kappa * da).detach()


def three_factor_loss(logits: Tensor, targets: Tensor, *, kappa: float,
                      tau: float, roll_control: bool = False) -> Tensor:
    """The three-factor objective. Scalar.

        S_t   = -log p_t(y_t)          the per-position cross-entropy
        H_t   =  H(p_t)                the model's OWN expected surprise
        phi_t =  H_t - S_t             the RPE, zero-mean by construction
        w_t   =  1 + kappa*tanh(phi_t/tau)                  DETACHED
        loss  =  sum w_t S_t / sum w_t

    ALIGNMENT, which is the one thing here that must not be wrong.
    `snn.dopamine.rpe` deliberately builds `phi_u` from `logits[:, u-1]` and
    `idx[:, u]`, because that signal is consumed **inside the forward pass at
    position u** and must therefore not contain the target. This one is consumed
    **by the loss at position t**, whose target is `y_t` by definition, so the
    RPE that describes that prediction is the one built from `logits[:, t]` and
    `y_t` -- no shift. Using `snn.dopamine.rpe`'s indexing here would weight
    position `t` by the error made at `t-1`, which would still train, would still
    look plausible, and would be measuring a different arm.

    That difference is why this is a separate function and not a call into
    `snn.dopamine`: the two are the same algebra deliberately aligned two
    different ways, and a shared helper would make the difference invisible.

    There is no leak. `w` is detached, so nothing about `y` reaches a parameter
    except through `S_t`, which is the objective. The forward pass never sees
    `w`; at eval it does not exist.

    `roll_control` rolls the weights one row along the BATCH axis, which
    preserves the marginal distribution and the temporal autocorrelation exactly
    and destroys only the alignment -- `snn.dopamine.roll_across_batch`'s
    construction, for its reasons, including that it cannot leak the future
    because it moves nothing along time.

    `kappa = 0.0` takes the identical code path as the unweighted trainer --
    `F.cross_entropy` with its default mean reduction -- rather than multiplying
    by a tensor of ones, so the nesting is a code path and not a float identity.
    """
    V = logits.shape[-1]
    flat = logits.reshape(-1, V).float()
    tgt = targets.reshape(-1)
    if kappa == 0.0:
        return F.cross_entropy(flat, tgt)
    if tau <= 0.0:
        raise ValueError(f"tau must be > 0, got {tau!r}")

    per = F.cross_entropy(flat, tgt, reduction="none")            # S_t, [B*L]
    logp = F.log_softmax(flat, dim=-1)
    # E_{x~p}[-log p(x)] = H(p) exactly: the model's own statement of the
    # surprise it expects, as one parallel reduction over V.
    entropy = -(logp.exp() * logp).sum(-1)                        # H_t, [B*L]
    da = torch.tanh((entropy - per) / tau)
    w = three_factor_weights(da, kappa)

    if roll_control:
        w = torch.roll(w.reshape(targets.shape), shifts=1, dims=0).reshape(-1)
    return (per * w).sum() / w.sum()
