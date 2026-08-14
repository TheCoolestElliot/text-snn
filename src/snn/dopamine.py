"""A dopamine signal: the model's own reward prediction error, broadcast back in.

Pre-registered in `experiments/logs/EXP_018_dopamine.md`. Read that first: the
signal, the squash scale, the control and the decision rule were all fixed before
this file existed, and `tau` descends from
`docs/reports/data/exp_018_da_calibration.json` rather than from taste.

This is the fourth transform of the input current in this project, after
`snn.prescan`'s token shift (EXP_007) and threshold gain (EXP_008) and
`snn.noise`'s background spikes (EXP_013), and it sits in the same place and for
the same reason: **once, for the whole sequence, before the scan starts**. On a
launch-bound box the metric is kernel count (`01_reconnaissance.md` §3.4), and a
transform outside the time loop costs O(1) kernels per layer against the scan's
`K*L`. The committed `snn.neuron.lif_scan`, its hand-written backward, its
25-mutation campaign and its R10 gate are untouched: this arm has no new
recurrence, so it needs no new gate.

THE SIGNAL
----------
Dopamine encodes a **reward prediction error** -- how much better the outcome was
than predicted. For a character LM the outcome at position `t` is the character
`x_t`, and the prediction of it was made at `t-1`:

    S_t   = -log p_{t-1}(x_t)      the surprise: how bad the outcome was   (nats)
    H_t   =  H(p_{t-1})            the model's OWN expected surprise       (nats)
    phi_t =  H_t - S_t             the reward prediction error             (nats)
    DA_t  =  tanh(phi_t / tau)     the bounded neuromodulator,  DA_0 = 0

`phi_t > 0` means the observed character was **less** surprising than the model
expected -- "better than predicted", which is the classic dopamine burst.

WHY THE BASELINE IS THE ENTROPY, AND WHY THAT DECIDED THE IMPLEMENTATION
-------------------------------------------------------------------------
A prediction *error* needs the prediction subtracted. The obvious baseline is a
running mean of recent surprise, and it is the wrong one here twice over.

`E_{x~p}[-log p(x)] = H(p)` **exactly**. So the model's own predictive
distribution already states what it expects the surprise to be, and `phi` is
zero-mean under that distribution **by construction** -- no subtracted baseline,
no exponential moving average, no learned critic, and nothing to tune.

The second reason is mechanical and is why no other baseline was considered. An
EMA over `t` is a **sequential scan**: `L` kernels per layer, on the one code
path this architecture exists to keep at one kernel per timestep, and a
data-dependent recurrence inside a region `snn.train` captures into a CUDA graph.
The entropy is a single parallel reduction over the vocabulary axis. The
derivation therefore changed what the experiment is, which is this project's
standing rule for a derivation done before implementing (`EXP_008` §1.1 is the
precedent, `EXP_013` §1.2 the most recent case).

CAUSALITY, WHICH IS THE ONE THING THAT MUST NOT BE WRONG
---------------------------------------------------------
`snn.data` builds each window as `x = block[:, :-1]`, `y = block[:, 1:]`, so
`y[:, u] == x[:, u+1]`. `phi_t` is assembled from `logits[:, t-1]` and
`x[:, t]`, so:

    phi_t  depends on  x_{<=t}          the target at t is  x_{t+1}

**`phi_t` therefore uses only characters the model has already read, and never
the target.** `y` is not an argument to anything in this file, and
`DopamineCharLM.forward` takes `idx` alone.

An off-by-one in that shift is the single failure this arm can have that would
still train, still produce a plausible bpc, and be a leak. It is guarded twice:
`018_calibrate_da.py`'s C1 asserts the gathered surprise equals
`F.cross_entropy` on the committed targets **elementwise and at 0.0**, and
`tests/test_dopamine.py`'s G2 asserts `DA[:, :t+1]` is bitwise invariant to any
perturbation of `x[:, t+1:]`.

WHY IT IS SQUASHED, MEASURED RATHER THAN ASSUMED
-------------------------------------------------
`phi` is bounded above by `H_t` and unbounded below. Measured on four committed
checkpoints (`exp_018_da_calibration.json`): sd **1.129**, max **+2.15**, min
**-12.97**, and **73.5 %** of values positive. An unsquashed `-12.97` through a
multiplicative gain of even 0.1 inverts the sign of the current. `tanh` is
therefore load-bearing, not cosmetic; it bounds the signal to (-1, 1) while
preserving the asymmetry, which is itself the biologically correct shape (phasic
bursts are frequent and small, dips rare and floor-limited).

`tau = 1.1291` is the median sd across those checkpoints, which span three
architectures and agree to **3.1 %** -- so the scale is a property of the task
rather than of the neuron, and is not re-tuned per arm.

WHAT IS NOT DONE, AND WHY
--------------------------
**`phi` is not normalised by batch statistics.** Subtracting a batch mean or
dividing by a batch sd would make one example's output depend on the other
examples in its batch. No other arm in this project does that, and it would make
the "fresh" and "carried" protocols measure different functions -- the carried
protocol cuts the split into `batch_size` contiguous streams, so batch
composition is part of the protocol, not an accident of it. The fixed calibrated
`tau` is used instead, and the cost of that choice is stated: at initialisation
`phi`'s sd is **46x** smaller than at convergence, so the modulation ramps in as
the model becomes predictive rather than being scale-free from step 0.

**Non-finite values are not masked.** A diverged model produces NaN logits, hence
NaN `phi`, hence NaN `DA`, and the NaN propagates into the loss where the run's
own divergence scan will see it. Clamping it here would hide a divergence behind
a healthy-looking arm, which is the failure mode `03_phase3_candidates.md` §8
records.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

__all__ = [
    "DA_MODES",
    "DA_SOURCES",
    "rpe",
    "dopamine",
    "roll_across_batch",
    "apply_dopamine",
]

#: How the broadcast scalar reaches the current.
#:
#: ``mult`` is the *time-varying* form of `EXP_008`'s learned per-channel
#: threshold, and the difference is the whole scientific content of this arm.
#: `EXP_008` Identity 1 makes a **constant** per-channel gain exactly a threshold
#: rescale, and Identity 2 folds it into `layers.k.weight`; its 0.0590 bpc is
#: therefore provably an *optimisation* effect on an unchanged function class.
#: `DA_t` varies with `t`, so it folds into no weight at any value of `k`, and
#: this arm's function class is **strictly larger** than the baseline's.
#:
#: ``add`` is a per-channel excitability shift rather than a gain. Both forms are
#: real dopaminergic actions and neither is derivable from the other.
DA_MODES = ("mult", "add")

#: Where the broadcast scalar comes from.
#:
#: ``off``      the arm is the Phase-2 baseline, by code path.
#: ``rpe``      the real signal, aligned to the position it describes.
#: ``rolled``   THE CONTROL. See `roll_across_batch`.
DA_SOURCES = ("off", "rpe", "rolled")


def _check_cur(cur: Tensor, k: Tensor) -> None:
    """Same shape contract `snn.prescan._check` enforces, and for the same reason.

    A `[1, d]` parameter and a `[d]` one broadcast identically against
    `[B, L, d]`, so a wrong shape here produces a plausible model rather than an
    error. Host-side only, so it costs nothing inside a CUDA-graph capture.
    """
    if cur.dim() != 3:
        raise ValueError(f"cur must be [B, L, d]; got shape {tuple(cur.shape)}")
    d = cur.shape[2]
    if k.dim() != 2 or k.shape[0] != 1 or k.shape[1] != d:
        raise ValueError(f"k must be [1, d] = (1, {d}); got {tuple(k.shape)}")


def rpe(logits: Tensor, idx: Tensor) -> Tensor:
    """The raw reward prediction error `phi`, in nats. `[B, L]`, `phi[:, 0] = 0`.

    `logits` `[B, L, V]` and `idx` `[B, L]`. Entry `t` is built from
    `logits[:, t-1]` and `idx[:, t]` and may only ever be consumed at position
    `t`; see the module docstring on causality, and `tests/test_dopamine.py`'s G2,
    which asserts it.

    `phi[:, 0] = 0` because position 0 has no preceding prediction. It is written
    with `F.pad` rather than an in-place write into a zeros tensor for the reason
    `snn.prescan.shift_by_one` is: the in-place form is a mutation of a buffer
    whose address a CUDA-graph capture would bake in, and this runs inside the
    captured training step.

    Computed in fp32 regardless of `cfg.dtype`, which selects the GEMM dtype only
    (spec §2). A bf16 log-softmax over a 205-way vocabulary would quantise the
    surprise onto a grid coarser than the differences this signal is made of.
    """
    if logits.dim() != 3:
        raise ValueError(f"logits must be [B, L, V]; got {tuple(logits.shape)}")
    if idx.dim() != 2:
        raise ValueError(f"idx must be [B, L]; got {tuple(idx.shape)}")
    if idx.shape != logits.shape[:2]:
        raise ValueError(
            f"idx {tuple(idx.shape)} does not match logits {tuple(logits.shape)}"
        )
    if logits.shape[1] < 2:
        raise ValueError(
            "the reward prediction error needs at least two positions: entry t is "
            f"built from the prediction at t-1, and L = {logits.shape[1]}"
        )

    logp = F.log_softmax(logits.float(), dim=-1)
    prev = logp[:, :-1]                                        # prediction at t-1
    # E_{x~p}[-log p(x)] = H(p) exactly, so this is the model's own expectation of
    # the surprise it is about to receive -- the subtrahend that makes phi an
    # ERROR rather than a magnitude, computed as one parallel reduction over V
    # rather than as a scan over t.
    expected = -(prev.exp() * prev).sum(-1)                    # H(p_{t-1})
    observed = -prev.gather(-1, idx[:, 1:].unsqueeze(-1)).squeeze(-1)   # S_t
    return F.pad(expected - observed, (1, 0))


def dopamine(phi: Tensor, tau: float) -> Tensor:
    """`tanh(phi / tau)` -- the bounded neuromodulator. `[B, L]` in (-1, 1).

    A function rather than an inline expression so that the model, the tests, the
    calibration script and the results script read the same quantity, for the
    same reason `snn.noise.noise_scale` and
    `LearnedThresholdCharLM.threshold_multiplier` are functions.

    `tanh(0) = 0.0` exactly in fp32, so a zero RPE is exactly no modulation --
    which is what makes `phi[:, 0] = 0` mean "position 0 is unmodulated" rather
    than "position 0 is modulated by approximately nothing".
    """
    if tau <= 0.0:
        raise ValueError(f"tau must be > 0, got {tau!r}")
    return torch.tanh(phi / tau)


def roll_across_batch(da: Tensor) -> Tensor:
    """THE CONTROL: each row receives its neighbour's dopamine. `[B, L]`.

    This is what decides whether the arm has a mechanism or a perturbation, and
    it is a roll along the **batch** axis rather than a shuffle along the time
    axis. That choice is the point, so it is written down here:

      * The **marginal distribution is preserved exactly**, not in distribution --
        the returned tensor holds the same multiset of values.
      * The **temporal autocorrelation is preserved exactly** -- every row keeps
        its own time series intact, merely attached to the wrong sequence. A time
        shuffle would destroy it, and would then be controlling for two things at
        once. That matters here specifically because the calibration measured the
        lag-1 autocorrelation at only **0.046**: on a nearly white signal a time
        shuffle is barely a control at all, while this is a total one.
      * **Alignment is destroyed completely.** Row `b` is driven by row `b-1`'s
        prediction error about a different stretch of the corpus.
      * **It cannot leak the future.** A roll along *time* with wraparound would
        bring `da[L-k:]` to the front and hand the model its own future; this
        moves nothing along the time axis at all. Row `b` at position `t` sees a
        quantity that depends on row `b-1`'s `x_{<=t}` and on nothing else.
      * **No RNG.** So there is no per-step draw to keep out of the captured
        region, no static buffer whose address the graph would bake in, and no
        seed stream to keep independent of the sampler's -- three failure modes
        `snn.noise` had to solve and this arm simply does not have.

    A batch of one makes the roll the identity, which would silently turn the
    control into the arm. That raises.
    """
    if da.dim() != 2:
        raise ValueError(f"da must be [B, L]; got {tuple(da.shape)}")
    if da.shape[0] < 2:
        raise ValueError(
            f"the rolled control needs batch_size >= 2; got {da.shape[0]}. At "
            "B = 1 the roll is the identity and the control would silently BE "
            "the arm it is controlling for."
        )
    return torch.roll(da, shifts=1, dims=0)


def apply_dopamine(cur: Tensor, k: Tensor, da: Tensor, mode: str) -> Tensor:
    """Modulate the input current by the broadcast scalar. `[B, L, d]`.

        mult:  cur'_{t,c} = cur_{t,c} * (1 + k_c * DA_t)
        add:   cur'_{t,c} = cur_{t,c} +       k_c * DA_t

    `cur` `[B, L, d]`, `k` `[1, d]`, `da` `[B, L]`.

    At `k = 0` the multiplicative form is `cur * 1.0`, which is `cur` bitwise for
    every finite current -- the nesting the arm's G1 gate depends on. The
    additive form is `cur + 0.0`, which is **not** `cur` for `cur = -0.0`, so the
    additive arm's nesting is a code-path branch in `DopamineCharLM._scan` rather
    than a float identity. `snn.noise`'s docstring makes the same distinction for
    the same reason.

    One broadcast multiply and one add per layer per forward pass, outside the
    time loop.
    """
    if mode not in DA_MODES:
        raise ValueError(f"mode must be one of {DA_MODES}, got {mode!r}")
    _check_cur(cur, k)
    if da.dim() != 2:
        raise ValueError(f"da must be [B, L]; got {tuple(da.shape)}")
    if da.shape != cur.shape[:2]:
        raise ValueError(
            f"da must be [B, L] matching cur; got {tuple(da.shape)} against "
            f"{tuple(cur.shape)}. A broadcastable-but-different shape would share "
            "one row's dopamine across the batch and would still train."
        )
    drive = k * da.unsqueeze(-1)                       # [B, L, d]
    return cur * (1.0 + drive) if mode == "mult" else cur + drive
