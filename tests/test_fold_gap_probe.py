"""EXP_012 S1: the leg-3 probe must reproduce the committed kernels bit-for-bit.

`u`, the decision variable, is consumed inside the scan and never returned, so
EXP_012's leg 3 recomputes the membrane recursion in order to observe it. A probe
that reimplements a neuron will agree with itself and measure nothing -- EXP_004
section 9.2's lesson that a contract is only guarded where the quantity it
constrains is actually observed.

The script asserts S1 inline on the real checkpoints and aborts on a mismatch.
These tests guard the same contract in the suite, on random data, across firing
rates -- and the last two are mutation legs, because a check that has never been
seen to fail is not known to be a check.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch

from snn.neuron import lif_scan
from snn.twocomp import twocomp_scan

_REPO = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location(
    "chase012", _REPO / "scripts" / "exp" / "012_chase_fold_gap.py")
chase = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(chase)

BETA, THR, ALPHA = 0.5, 1.0, 2.0
# three scales, chosen to straddle the firing rate: near-silent, ~0.13, ~0.33.
SCALES = [0.2, 1.0, 5.0]


@pytest.fixture(scope="module")
def device() -> str:
    """CUDA when there is one, so the probe is checked against the *fused*
    kernel -- the path the arms actually ran. On a CPU-only machine the eager
    path is still a real check of the same recursion, and `fused=cur.is_cuda`
    at each call site is what keeps one test body covering both."""
    return "cuda" if torch.cuda.is_available() else "cpu"


def _inputs(scale: float, device: str, seed: int = 0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    cur = (torch.randn(4, 48, 64, generator=g) * scale).to(device)
    w = torch.rand(1, 64, generator=g).to(device)
    beta_s = torch.sigmoid(torch.randn(1, 64, generator=g)).to(device)
    return cur, w, beta_s


@pytest.mark.parametrize("scale", SCALES)
def test_probe_reproduces_lif_scan_bitwise(scale, device):
    cur, _, _ = _inputs(scale, device)
    v0 = cur.new_zeros(cur.shape[0], cur.shape[2])
    ref, _ = lif_scan(cur, v0, BETA, THR, ALPHA, "hard", fused=cur.is_cuda)
    got, u = chase.decision_variable(cur, kind="lif", thr=THR, beta=BETA)
    assert torch.equal(ref, got)
    assert torch.equal(ref, (u >= 0).to(cur.dtype))


@pytest.mark.parametrize("scale", SCALES)
def test_probe_reproduces_twocomp_scan_bitwise(scale, device):
    cur, w, beta_s = _inputs(scale, device)
    v0 = cur.new_zeros(cur.shape[0], 2 * cur.shape[2])
    ref, _ = twocomp_scan(cur, v0, w, beta_s, BETA, THR, ALPHA,
                          fused=cur.is_cuda)
    got, u = chase.decision_variable(cur, kind="twocomp", thr=THR, beta=BETA,
                                     w=w, beta_s=beta_s)
    assert torch.equal(ref, got)
    assert torch.equal(ref, (u >= 0).to(cur.dtype))


def test_the_check_can_fail_if_the_slow_pole_is_reset(device):
    """Mutation leg. Resetting `vs` is the one thing the candidate neuron does
    NOT do (`twocomp_scan_eager`: "vs is NOT reset -- that is the whole point").
    A probe that reset it would still look like a two-compartment neuron."""
    cur, w, beta_s = _inputs(1.0, device)
    v0 = cur.new_zeros(cur.shape[0], 2 * cur.shape[2])
    ref, _ = twocomp_scan(cur, v0, w, beta_s, BETA, THR, ALPHA,
                          fused=cur.is_cuda)

    B, L, d = cur.shape
    vf, vs, spikes = cur.new_zeros(B, d), cur.new_zeros(B, d), []
    for t in range(L):
        vf = vf * BETA + cur[:, t]
        vs = vs * beta_s + cur[:, t]
        s = ((vf + w * vs - THR) >= 0).to(cur.dtype)
        spikes.append(s)
        vf = vf * (1.0 - s)
        vs = vs * (1.0 - s)          # <-- the mutation
    assert not torch.equal(ref, torch.stack(spikes, dim=1))


def test_the_check_can_fail_on_a_soft_reset(device):
    """Mutation leg for the LIF side: the committed arm uses `reset="hard"`."""
    cur, _, _ = _inputs(1.0, device)
    v0 = cur.new_zeros(cur.shape[0], cur.shape[2])
    ref, _ = lif_scan(cur, v0, BETA, THR, ALPHA, "hard", fused=cur.is_cuda)
    soft, _ = lif_scan(cur, v0, BETA, THR, ALPHA, "soft", fused=cur.is_cuda)
    assert not torch.equal(ref, soft)
    got, _ = chase.decision_variable(cur, kind="lif", thr=THR, beta=BETA)
    assert torch.equal(ref, got)      # and the probe matches the committed one
