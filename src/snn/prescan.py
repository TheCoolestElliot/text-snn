"""Pre-scan transforms of the input current: EXP_007's mix and EXP_008's gain.

Pre-registered in `experiments/logs/EXP_007_token_shift.md` and
`experiments/logs/EXP_008_learned_threshold.md`. Read those first: the
parameterisations, the initialisations and the reasons for both were fixed before
this file existed.

Both transforms act on the layer's input current `cur` **once, for the whole
sequence, before the scan starts** -- the same place `model.py`'s single GEMM per
layer lives, and for the same reason. On a launch-bound box the metric is kernel
count (`01_reconnaissance.md` §3.4), and a transform outside the time loop costs
O(1) kernels per layer against the scan's `K*L`. Nothing here touches the time
loop, the neuron, or any compiled kernel.

WHAT THAT BUYS, AND IT IS THE POINT OF BOTH ARMS
------------------------------------------------
The scan these feed is `snn.neuron.lif_scan` -- the committed Phase-2 kernel, its
hand-written backward, its 25-mutation campaign and its R10 gate, all unchanged
and all still guarding the same code. Neither arm needs a new kernel, a new
backward, or a new gradient gate, because neither arm has a new recurrence.
The gradients through these transforms are ordinary autograd.

That is not a convenience. `03_phase3_candidates.md` §7.1 prices "the real cost
of every new-neuron row" as its mutation-tested gradient gate; these two rows do
not pay it, and that is most of why they are cheap.

TOKEN SHIFT (EXP_007)
---------------------
    cur'_t,c = mu_c * cur_t,c + (1 - mu_c) * cur_{t-1,c},     cur_{-1} = 0

One parameter per neuron, unconstrained, initialised at 1.0 where the transform
is the identity. A length-2 FIR of the neuron's own input, which the Phase-2 LIF
-- an exponential IIR at fixed beta with a hard reset -- cannot represent at any
weight. So this arm's function class is strictly larger than the baseline's.

`cur_{-1} = 0` at the start of every window and the shift register does not cross
window boundaries; EXP_007 §1.3 states what that costs and why it is not
corrected.

LEARNED PER-CHANNEL THRESHOLD (EXP_008)
---------------------------------------
    cur'_t,c = cur_t,c * exp(-theta_c)          <=>      thr_c = thr * exp(theta_c)

The equivalence is exact and is EXP_008 §1.1's Identity 1: with `u = v / g`,

    v_t = beta*v_{t-1} + cur_t ,  s = 1[v_t >= thr*g] ,  v_t <- v_t*(1 - s)
    u_t = beta*u_{t-1} + cur_t/g ,  s = 1[u_t >= thr] ,  u_t <- u_t*(1 - s)

are the same spike train, because the hard reset is *multiplicative* and
therefore commutes with the scaling. Implementing the second form is what keeps
the committed threshold and the committed kernel untouched.

`exp` rather than a raw multiplier keeps `g > 0`, which Identity 1 requires: a
gain that crossed zero would invert the comparison and would not be the arm
EXP_004 §10.6 described. `exp(0) = 1.0` exactly in fp32, so `theta = 0` nests the
baseline bitwise and in the *interior* of an unconstrained parameter -- the same
choice, for the same reason, as `snn.twocomp`'s additive mix.

Identity 2, which is the arm's whole scientific content: `cur = W*h + b`, so
scaling output channel `c` by `g_c` is the model with `(W_c, b_c)` replaced by
`(g_c*W_c, g_c*b_c)`. The gain folds into the layer's own `nn.Linear`.
`LearnedThresholdCharLM.fold_into_spiking_state_dict` performs that fold, and
EXP_008's W1 requires the folded model to reproduce the arm's own bpc.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

__all__ = ["token_shift", "threshold_gain", "shift_by_one"]


def _check(cur: Tensor, p: Tensor, name: str) -> None:
    """Same shape contract `snn.twocomp._check` enforces, and for the same reason.

    A `[1, d]` parameter and a `[d]` one broadcast identically against `[B, L, d]`,
    so a wrong shape here produces a plausible model rather than an error. The
    check is cheap and host-side only, so it costs nothing inside a capture.
    """
    if cur.dim() != 3:
        raise ValueError(f"cur must be [B, L, d]; got shape {tuple(cur.shape)}")
    d = cur.shape[2]
    if p.dim() != 2 or p.shape[0] != 1 or p.shape[1] != d:
        raise ValueError(f"{name} must be [1, d] = (1, {d}); got {tuple(p.shape)}")


def shift_by_one(cur: Tensor) -> Tensor:
    """`cur` delayed one timestep along L, with a zero in front. [B, L, d].

    Written with `F.pad` on a slice rather than with an in-place write into a
    zeros tensor: the in-place form is a mutation of a leaf-adjacent buffer that
    autograd handles but CUDA-graph capture would bake an address into, and this
    runs inside the captured training step.
    """
    if cur.dim() != 3:
        raise ValueError(f"cur must be [B, L, d]; got shape {tuple(cur.shape)}")
    if cur.shape[1] < 1:
        raise ValueError("cur must have at least one timestep")
    return F.pad(cur[:, :-1], (0, 0, 1, 0))


def token_shift(cur: Tensor, mu: Tensor) -> Tensor:
    """`mu*cur_t + (1-mu)*cur_{t-1}`, per channel. `cur` [B, L, d], `mu` [1, d].

    At `mu = 1` this is `1.0*cur + 0.0*shifted`, which is `cur` bitwise for every
    finite current -- the nesting EXP_007 §1.2 depends on.

    `dL/dmu_c = sum_t (dL/dcur'_{t,c}) * (cur_{t,c} - cur_{t-1,c})`, which is zero
    only for a current that is constant in time. That is why `mu = 1` is not a
    saddle, derived in EXP_007 §1.2 before the §7.2 screen measured it.
    """
    _check(cur, mu, "mu")
    return mu * cur + (1.0 - mu) * shift_by_one(cur)


def threshold_gain(cur: Tensor, theta: Tensor) -> Tensor:
    """`cur * exp(-theta)`, per channel -- equivalently `thr_c = thr*exp(theta_c)`.

    `cur` [B, L, d], `theta` [1, d]. See the module docstring for Identity 1,
    which is what makes the two forms the same neuron rather than two neurons that
    happen to agree.
    """
    _check(cur, theta, "theta")
    return cur * torch.exp(-theta)
