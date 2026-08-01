"""Surrogate-gradient checks (spec 02a §12).

The R10 gate in `test_neuron_equivalence.py` compares a hand-written CUDA
backward against the eager path's autograd derivative. That comparison is only
worth anything if the eager derivative is *itself* exactly the analytic
`atan_grad`; otherwise the gate measures the agreement of two things that are
both wrong in the same way. These tests establish that premise, and they also
check the one CUDA text (`ATAN_CUDA_GRAD`) that the fused kernel and the eager
path are supposed to share, so the two cannot drift apart through independent
edits.
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

from snn.surrogate import (  # noqa: E402
    ATAN_CUDA_GRAD,
    atan_grad,
    atan_spike,
    atan_value,
)

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="needs a CUDA device"
)

ALPHAS = (0.5, 1.0, 2.0, 4.0)


def test_derivative_at_zero_is_alpha_over_two():
    """derivative(0) = alpha/2, exactly -- the surrogate's peak gain.

    This is the number that sets the *scale* of the surrogate, which §4.3 item 3
    of the reconnaissance identifies as the fragile axis (shape is robust, width
    is not). At the default alpha = 2 it is exactly 1.0, so the surrogate is
    gain-neutral at the threshold.
    """
    for alpha in ALPHAS:
        z = torch.zeros(1, dtype=torch.float64)
        assert float(atan_grad(z, alpha)) == alpha / 2.0


def test_value_at_zero_is_half():
    z = torch.zeros(1, dtype=torch.float64)
    assert float(atan_value(z, 2.0)) == pytest.approx(0.5, abs=1e-15)


def test_value_is_monotone_and_bounded():
    x = torch.linspace(-50.0, 50.0, 4001, dtype=torch.float64)
    v = atan_value(x, 2.0)
    assert bool((v > 0.0).all()) and bool((v < 1.0).all())
    assert bool((v.diff() > 0.0).all())


def test_half_width_matches_closed_form():
    """derivative falls to half its peak at |x| = 2/(pi*alpha).

    An independent property of the analytic form: if `atan_grad` were mis-scaled
    inside the squared term, the peak height test above would still pass but this
    one would not.
    """
    for alpha in ALPHAS:
        x = torch.tensor([2.0 / (math.pi * alpha)], dtype=torch.float64)
        assert float(atan_grad(x, alpha)) == pytest.approx(alpha / 4.0, rel=1e-12)


@pytest.mark.parametrize("alpha", ALPHAS)
def test_autograd_derivative_matches_analytic_float64(alpha):
    """d atan_value / dx computed by autograd equals atan_grad.

    Run in float64 so that agreement is evidence about the *algebra* rather than
    about fp32 rounding being coarse enough to hide a small error.
    """
    x = torch.linspace(-8.0, 8.0, 2001, dtype=torch.float64, requires_grad=True)
    (g,) = torch.autograd.grad(atan_value(x, alpha).sum(), x)
    assert torch.allclose(g, atan_grad(x.detach(), alpha), rtol=1e-12, atol=1e-14)


@pytest.mark.parametrize("alpha", ALPHAS)
def test_autograd_derivative_matches_analytic_float32(alpha):
    """Same, in the precision the model actually trains in (spec §12: 1e-6)."""
    x = torch.linspace(-8.0, 8.0, 2001, dtype=torch.float32, requires_grad=True)
    (g,) = torch.autograd.grad(atan_value(x, alpha).sum(), x)
    assert torch.allclose(g, atan_grad(x.detach(), alpha), rtol=1e-5, atol=1e-6)


def test_straight_through_spike_is_exactly_binary():
    """The straight-through spike must be bit-exactly 0.0 or 1.0.

    Not "close to": invariant I1 says binary spikes carry all information between
    layers, and the next layer multiplies them by a weight matrix. The naive
    left-associative form `s_hard + sv - sv.detach()` fails this on ~7% of
    elements by ~6e-08 -- see the note in `snn.surrogate.atan_spike`. That error
    is far too small to notice in a loss curve and would silently make the
    "spiking" arm not quite spiking.
    """
    torch.manual_seed(0)
    x = torch.randn(1 << 20, dtype=torch.float32) * 3.0
    s = atan_spike(x, 2.0)
    assert bool(((s == 0.0) | (s == 1.0)).all())
    assert bool((s == (x >= 0).to(s.dtype)).all())


def test_straight_through_gradient_is_atan_grad():
    x = torch.linspace(-6.0, 6.0, 4001, dtype=torch.float64, requires_grad=True)
    (g,) = torch.autograd.grad(atan_spike(x, 2.0).sum(), x)
    assert torch.allclose(g, atan_grad(x.detach(), 2.0), rtol=1e-12, atol=1e-14)


def test_threshold_comparison_is_greater_or_equal():
    """Spec §2 fixes the comparison as `>=`, matching the Phase-1 kernel."""
    x = torch.zeros(1)
    assert float(atan_spike(x, 2.0)) == 1.0


@pytest.mark.cuda
@requires_cuda
@pytest.mark.parametrize("alpha", ALPHAS)
def test_cuda_expression_matches_atan_grad(alpha):
    """`ATAN_CUDA_GRAD` compiled by NVRTC must equal `atan_grad`.

    This is the seam between `surrogate.py` and `kernels.py`: the fused backward
    substitutes this text into its kernel body, so if it disagreed with the eager
    surrogate every gradient in the fused path would be quietly wrong -- exactly
    the R10 failure mode. Checked here on its own so that a failure points at the
    surrogate rather than at the scan.
    """
    from torch.cuda.jiterator import _create_multi_output_jit_fn

    src = (
        "template <typename T>\n"
        "void probe(T x, T alpha, T& out0, T& out1) {\n"
        f"    out0 = {ATAN_CUDA_GRAD};\n"
        "    out1 = T(0);\n"
        "}\n"
    )
    fn = _create_multi_output_jit_fn(src, num_outputs=2, alpha=2.0)
    x = torch.linspace(-8.0, 8.0, 20001, device="cuda", dtype=torch.float32)
    got, _ = fn(x, alpha=alpha)
    ref = atan_grad(x, alpha)
    assert torch.allclose(got, ref, rtol=1e-6, atol=1e-8), float(
        (got - ref).abs().max()
    )
