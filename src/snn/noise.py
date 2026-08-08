"""Injected background spike noise on the input current -- training only.

Pre-registered in `experiments/logs/EXP_013_noise_injection.md`. Read that first:
the parameterisation, the amplitude ladder and the decision rule were all fixed
before this file existed, and the amplitude ladder is calibrated against
`docs/reports/data/exp_013_noise_calibration.json` rather than chosen by taste.

This module is the third transform of the input current in this project, after
`snn.prescan`'s token shift (EXP_007) and threshold gain (EXP_008), and it sits
in the same place and for the same reason: **once, for the whole sequence, before
the scan starts**. On a launch-bound box the metric is kernel count
(`01_reconnaissance.md` 3.4), and a transform outside the time loop costs O(1)
kernels per layer against the scan's `K*L`. The committed `snn.neuron.lif_scan`,
its hand-written backward, its 25-mutation campaign and its R10 gate are
untouched: this arm has no new recurrence, so it needs no new gate.

THE TRANSFORM
-------------
    cur'_{t,c} = cur_{t,c} + amp * (B_{t,c} - p) / sqrt(p*(1-p)),
    B_{t,c} ~ Bernoulli(p), i.i.d. over batch, time and channel.

`B` is a literal background spike train: each (step, channel) either receives a
background event or does not. The affine rescaling makes the process zero-mean
and unit-variance, so **`amp` is the standard deviation of the injected current**
in the same units as the firing threshold (which is 1.0), and `p` sets only how
sparse the events are. That separation is what makes `amp` sweepable and `p`
holdable, which is the single-variable design the experiment needs.

WHY IT IS CENTRED, WHICH IS THE WHOLE REASON THE FORMULA HAS A `- p` IN IT
--------------------------------------------------------------------------
Uncentred background spikes -- `cur + a*B` -- add a constant `a*p` to every
channel's current at every step. A constant added to the current is exactly a
uniform *reduction of the effective firing threshold*: with `u = v - a*p/(1-beta)`
in the steady state, firing at `v >= thr` is firing at `u >= thr - a*p/(1-beta)`.
That is EXP_008's Identity 1 in its uniform form, and EXP_008 measured a
per-channel version of precisely that mechanism to be worth **0.0590 bpc** on its
own -- 44 % of the adopted arm's entire gain.

So an uncentred noise arm would confound "regularisation" with "threshold shift",
and the confound is not small compared with the effect being looked for: it is
larger than the whole generalisation gap this arm is aimed at. Subtracting `p`
costs one kernel and removes it. The arm is then a pure second-moment
intervention -- it changes the variance of the current and nothing about its
mean -- which is the only version of it that can answer the question it is being
run for.

`EXP_013` 1.2 states this derivation as the reason the experiment has the shape
it has, per the project's standing rule that a derivation done before
implementing is allowed to change what the experiment is.

WHY THE NOISE IS DRAWN OUTSIDE THE CAPTURED REGION
---------------------------------------------------
`snn.train` captures a full fwd+bwd+AdamW step into a CUDA graph, and the
training sampler is a pure function of `(seed, step)` precisely so that a run
resumed at step k sees what an uninterrupted run saw at step k (`snn.data` 3,
risk R6). Drawing the noise *inside* the graph would make the arm's trajectory
depend on a captured philox offset that `set_rng_state` does not restore, and R6
would stop being testable for this arm alone.

So the noise is generated into **static buffers** owned by the model and refilled
by the trainer outside the captured region, from a `torch.Generator` seeded
`mix64(seed ^ NOISE_STREAM_SALT, step)` -- the same construction the data sampler
uses, on a deliberately different stream so that the noise is independent of
which windows the batch drew. `NoisyCharLM.refill_noise` is the only writer.

AT amp = 0 THE ARM IS THE PHASE-2 BASELINE, BITWISE
----------------------------------------------------
`add_background_noise` is not called at all when `amp == 0` (a Python branch on a
constant fixed at construction, resolved long before capture), so the nesting is
not "x + 0.0 == x" -- which is false for x = -0.0 -- but the identical code path.
`tests/test_noise.py` asserts forward AND backward bitwise against
`SpikingCharLM`, which is EXP_011 K4's standard for a nesting claim.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

__all__ = ["add_background_noise", "noise_scale", "fill_background_noise",
           "NOISE_STREAM_SALT"]

#: Mixed into the seed so the noise stream is independent of the data sampler's.
#: An arm whose injected noise is correlated with which windows the batch drew is
#: not the arm this experiment registered.
NOISE_STREAM_SALT = 0x5EED0153


def noise_scale(amp: float, p: float) -> float:
    """`amp / sqrt(p*(1-p))` -- the multiplier that makes `amp` the noise sd.

    A function rather than an inline expression so that the model, the trainer,
    the tests and the results script read the same quantity, for the same reason
    `LearnedThresholdCharLM.threshold_multiplier` is a method.
    """
    if not 0.0 < p < 1.0:
        raise ValueError(f"noise_p must be in (0, 1), got {p!r}")
    if amp < 0.0:
        raise ValueError(f"noise_amp must be >= 0, got {amp!r}")
    return float(amp) / math.sqrt(p * (1.0 - p))


def fill_background_noise(buf: Tensor, amp: float, p: float,
                          generator: torch.Generator) -> Tensor:
    """Write one draw of the centred background spike process into `buf`, in place.

    In place, and through `generator` rather than the global stream, because the
    buffer's address is baked into a CUDA graph and the stream must be a pure
    function of `(seed, step)`. Returns `buf` for convenience.

    The three ops are one Bernoulli draw and two elementwise updates, all outside
    the time loop and outside the captured region: O(1) kernels per layer per
    step, against the scan's `L` per layer.
    """
    scale = noise_scale(amp, p)
    buf.bernoulli_(p, generator=generator)
    buf.sub_(p).mul_(scale)
    return buf


def add_background_noise(cur: Tensor, noise: Tensor) -> Tensor:
    """`cur + noise`, with the shape contract `snn.prescan._check` enforces.

    The check is not ceremonial: a `[B, L, d]` buffer and a `[1, L, d]` one
    broadcast identically here, so a wrong shape would produce a plausible model
    that shares one noise draw across the batch -- a different experiment that
    looks exactly like this one.
    """
    if cur.dim() != 3:
        raise ValueError(f"cur must be [B, L, d]; got shape {tuple(cur.shape)}")
    if noise.shape != cur.shape:
        raise ValueError(
            f"noise must match cur exactly; got {tuple(noise.shape)} against "
            f"{tuple(cur.shape)}. A broadcastable-but-different shape would "
            "share draws across an axis and would still train."
        )
    return cur + noise
