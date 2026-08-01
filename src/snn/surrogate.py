"""Surrogate gradient for the hard-threshold spike (spec 02a §5).

The forward spike is a Heaviside step, whose derivative is zero almost
everywhere and undefined at the threshold. Surrogate-gradient BPTT (invariant I4)
replaces that derivative -- and only that derivative -- with a bounded bump
function. Phase 2 fixes the shape to the arctangent surrogate at alpha = 2,
because the literature review (01_reconnaissance.md §4.3 item 3) found the
reported differences between surrogate *shapes* to be confounded by a ~19x width
mismatch at library defaults, while the *width* is genuinely fragile. Shape is
therefore frozen and width is the thing an ablation would sweep.

With x = v_pre - threshold and alpha = surrogate_alpha:

    value(x)      = (1/pi) * arctan(pi/2 * alpha * x) + 1/2
    derivative(x) = (alpha/2) / (1 + (pi/2 * alpha * x)^2)

`derivative` is the exact analytic d value / d x, so the straight-through
construction in `atan_spike` has an autograd derivative that is *identically*
`atan_grad`. That identity is what makes the eager path a usable ground truth for
the R10 gate: the hand-written jiterator backward is checked against it, so if
the eager derivative were only approximately the analytic form the gate would be
measuring the wrong thing.

`ATAN_CUDA_GRAD` is the same expression transcribed to CUDA C++ so that the
fused kernel and the eager reference cannot drift apart through independent
edits. It is a single expression, not a function definition, and it assumes two
identifiers are in scope at the point of substitution: `x` and `alpha`, both of
the jiterator template type `T`.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

__all__ = [
    "atan_value",
    "atan_grad",
    "atan_spike",
    "ATAN_CUDA_GRAD",
    "ATAN_CUDA_HALF_PI",
]

# pi/2 kept as a module constant so the eager path and the CUDA text below are
# demonstrably the same number to the last digit either format can hold.
_HALF_PI = math.pi / 2.0
_INV_PI = 1.0 / math.pi

# The literal handed to NVRTC. float(math.pi/2) is exactly this value.
ATAN_CUDA_HALF_PI = "1.57079632679489661923"


def atan_value(x: Tensor, alpha: float) -> Tensor:
    """Smooth surrogate for the Heaviside step, mapping R -> (0, 1).

    Never used as a spike in the SNN arm -- the spike is always hard (I3). This
    exists only to carry gradient in the straight-through construction, and as
    the continuous activation of the `analogue` control arm, which is exactly the
    point of that control: same parameters, same everything, binarity removed.
    """
    return torch.atan(x * (_HALF_PI * alpha)) * _INV_PI + 0.5


def atan_grad(x: Tensor, alpha: float) -> Tensor:
    """Exact d atan_value / d x. `atan_grad(0, alpha) == alpha / 2`."""
    u = x * (_HALF_PI * alpha)
    return (0.5 * alpha) / (1.0 + u * u)


def atan_spike(x: Tensor, alpha: float) -> Tensor:
    """Hard spike forward, surrogate backward (spec §5).

    NOTE ON THE PARENTHESES -- they are load-bearing, and the spec text omits
    them. Spec §5 writes `s = s_hard.detach() + sv - sv.detach()`, which Python
    evaluates left to right as `(s_hard + sv) - sv`. That is *not* exactly
    `s_hard` in floating point: adding sv to 1.0 rounds away up to half an ulp of
    1.0, and subtracting sv back does not recover it. Measured on this stack,
    7.2% of elements came out at 1 +/- 5.96e-08 rather than 1.0.

    That matters three times over. It violates invariant I1 (the inter-layer
    signal is no longer binary, so the next layer's GEMM sees 1.00000006). It
    breaks the R10 gate, which requires fused and eager spikes to be
    bit-identical rather than close. And it is invisible: the model trains fine.

    Grouped as `s_hard + (sv - sv.detach())` the surrogate terms cancel to
    exactly 0.0 first -- same tensor, same values, exact subtraction -- and the
    result is bit-exactly `s_hard` for every input. The algebra is unchanged and
    so is the derivative, which is still `atan_grad`.
    """
    s_hard = (x >= 0).to(x.dtype)
    sv = atan_value(x, alpha)
    return s_hard.detach() + (sv - sv.detach())


# CUDA C++ transcription of `atan_grad`, for substitution into a jiterator
# kernel body. Requires `T x` and `T alpha` in scope. Written with the
# (pi/2 * alpha) product first so it associates the same way the eager path does.
ATAN_CUDA_GRAD = (
    "(T(0.5) * alpha / (T(1) + (T({half_pi}) * alpha * x) * (T({half_pi}) * alpha * x)))"
).format(half_pi=ATAN_CUDA_HALF_PI)
