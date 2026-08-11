"""`016_reset_jacobian.py` measures an intermediate of the BACKWARD kernel that
the forward never materialises, using a scan re-implemented inside the probe.

That is a licence to be silently wrong in two directions at once, so these tests
pin both: that the local scans ARE the committed eager scans (T2, on CPU, before
any GPU is spent), and that the closed form the probe validates itself against
(T1) is the derivation in `EXP_016` §2.2 rather than a number typed in.

None of this touches the GPU or trains anything.
"""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import torch

from snn.neuron import lif_scan_eager
from snn.twocomp import twocomp_scan_eager

_REPO = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location(
    "rj016", _REPO / "scripts" / "exp" / "016_reset_jacobian.py")
rj = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(rj)

ALPHA, THR, BETA_F = 2.0, 1.0, 0.5


# -- T1: the bound is derived, not typed in --------------------------------

def test_lif_bound_matches_the_analytic_extremum():
    """§2.2 solves max|v_pre*sg| at x* = -1 + sqrt(1 + 1/pi^2). The probe's
    numerical scan must land on the same value, or T1 is validating the probe
    against a different quantity than the one §2.2 derives."""
    x_star = -1.0 + math.sqrt(1.0 + 1.0 / math.pi ** 2)
    sg = 1.0 / (1.0 + (math.pi * x_star) ** 2)      # alpha = 2 => 0.5*alpha = 1
    analytic = BETA_F * abs((x_star + THR) * sg)

    assert rj.lif_bound(ALPHA, THR, BETA_F) == pytest_approx(analytic, 1e-6)
    assert rj.lif_bound(ALPHA, THR, BETA_F) == pytest_approx(
        rj.LIF_BOUND_AT_FROZEN_CONSTANTS, 1e-6)


def test_the_bound_is_not_a_project_constant():
    """§6 item 5: the bound is exact only at the frozen constants and must be
    re-derived if any of alpha, thr or beta moves. If this ever passes with all
    three equal, `lif_bound` has stopped depending on its arguments and T1 would
    silently validate every arm against the baseline's number -- the exact
    failure `CONTRIBUTING.md` §4 records for inherited fold-in tolerances."""
    base = rj.lif_bound(ALPHA, THR, BETA_F)
    assert rj.lif_bound(4.0, THR, BETA_F) != base
    assert rj.lif_bound(ALPHA, 2.0, BETA_F) != base
    assert rj.lif_bound(ALPHA, THR, 0.9) != base


def test_lif_factor_never_exceeds_its_bound_on_random_membranes():
    """T1 as a unit test, over a range far wider than any measured membrane."""
    v_pre = torch.linspace(-5000.0, 5000.0, 2_000_001, dtype=torch.float64)
    x = v_pre - THR
    s = (x >= 0).to(v_pre.dtype)
    from snn.surrogate import atan_grad
    g = BETA_F * ((1.0 - s) - v_pre * atan_grad(x, ALPHA))
    assert float(g.abs().max()) <= rj.lif_bound(ALPHA, THR, BETA_F) + rj.T1_TOL


def test_twocomp_factor_crosses_one_when_the_drive_does():
    """§2.3's crossover. `dv = (1 - sh) - vf*sgd` with `vf = v_pre - w*vs`, so a
    channel near threshold with |w*vs| > 1 expands. This is the whole claim, and
    it is checked here against explicit numbers rather than inferred."""
    from snn.surrogate import atan_grad

    def worst(drive: float) -> float:
        v_pre = torch.linspace(-50.0, 50.0, 2_000_001, dtype=torch.float64)
        x = v_pre - THR
        sh = (x >= 0).to(v_pre.dtype)
        vf = v_pre - drive
        return float((BETA_F * ((1.0 - sh) - vf * atan_grad(x, ALPHA))).abs().max())

    # w*vs = 0 is the LIF exactly -- the nesting the kernel comment relies on.
    assert worst(0.0) == pytest_approx(rj.lif_bound(ALPHA, THR, BETA_F), 1e-5)
    assert worst(-1.0) > 1.0
    assert worst(-10.0) > 5.0
    assert worst(-100.0) > 50.0


# -- T2: the local scans are the committed scans ---------------------------

def test_local_lif_scan_is_the_committed_eager_scan():
    torch.manual_seed(0)
    b, length, d = 4, 32, 16
    cur = torch.randn(b, length, d) * 3.0
    v0 = torch.zeros(b, d)
    acc = rj.Acc(b, d, "cpu")
    mine, v_end = rj.scan_lif(cur, v0, BETA_F, THR, ALPHA, acc)
    ref, v_ref = lif_scan_eager(cur, v0, BETA_F, THR, ALPHA, "hard")

    assert torch.equal(mine, ref), "spikes must be bitwise equal, not close"
    assert torch.equal(v_end, v_ref)
    assert acc.n_total == b * length * d


def test_local_twocomp_scan_is_the_committed_eager_scan():
    torch.manual_seed(0)
    b, length, d = 4, 32, 16
    cur = torch.randn(b, length, d) * 3.0
    v0 = torch.zeros(b, 2 * d)
    w = torch.full((1, d), 0.1)
    beta_s = torch.full((1, d), 0.95)
    acc = rj.Acc(b, d, "cpu")
    mine, v_end = rj.scan_twocomp(cur, v0, w, beta_s, BETA_F, THR, ALPHA, acc)
    ref, v_ref = twocomp_scan_eager(cur, v0, w, beta_s, BETA_F, THR, ALPHA)

    assert torch.equal(mine, ref), "spikes must be bitwise equal, not close"
    assert torch.equal(v_end, v_ref)


def test_twocomp_scan_at_w_zero_is_the_lif_scan():
    """The kernel comment at `twocomp.py:146-162` says the grouping was chosen so
    that `w = 0` nests the committed LIF bit-exactly. If the probe's two local
    scans disagree there, one of them is not the scan it claims to be."""
    torch.manual_seed(0)
    b, length, d = 4, 32, 16
    cur = torch.randn(b, length, d) * 3.0
    acc_tc = rj.Acc(b, d, "cpu")
    acc_lif = rj.Acc(b, d, "cpu")
    tc, _ = rj.scan_twocomp(cur, torch.zeros(b, 2 * d), torch.zeros(1, d),
                           torch.full((1, d), 0.95), BETA_F, THR, ALPHA, acc_tc)
    lif, _ = rj.scan_lif(cur, torch.zeros(b, d), BETA_F, THR, ALPHA, acc_lif)

    assert torch.equal(tc, lif)
    assert float(acc_tc.max_abs) == float(acc_lif.max_abs)


# -- the accumulator itself ------------------------------------------------

def test_chain_gain_is_the_log_of_the_product():
    """`max_chain_log_gain` must be sum_t log|g_t| and nothing else. Fed a
    constant factor it has a closed form, so an off-by-one in the time loop or a
    stray mean would show here rather than in a result nobody can check."""
    b, d, length = 2, 3, 10
    acc = rj.Acc(b, d, "cpu")
    for _ in range(length):
        acc.add(torch.full((b, d), 2.0), None)
    rep = acc.report()

    assert rep["max_chain_log_gain"] == pytest_approx(length * math.log(2.0), 1e-9)
    assert rep["max_chain_log10_gain"] == pytest_approx(length * math.log10(2.0), 1e-9)
    assert rep["n_sites"] == b * d * length
    assert rep["frac_over"]["1.0"] == 1.0
    assert rep["n_chains_expanding"] == b * d


def test_an_exact_zero_kills_its_chain_rather_than_poisoning_every_chain():
    """log(0) is -inf; an unguarded accumulation would make one dead channel turn
    the whole layer's mean into nan and hide every other chain."""
    acc = rj.Acc(2, 2, "cpu")
    g = torch.tensor([[2.0, 0.0], [2.0, 2.0]])
    for _ in range(4):
        acc.add(g, None)
    rep = acc.report()

    assert math.isfinite(rep["mean_chain_log_gain"])
    assert rep["n_exact_zero_g"] == 4
    assert rep["max_chain_log_gain"] == pytest_approx(4 * math.log(2.0), 1e-9)


def test_thresholds_include_the_only_one_with_a_meaning():
    """1.0 separates contraction from expansion and is what P1/P2/P4 are placed
    on. `report()` indexes it by value, so its presence is load-bearing."""
    assert 1.0 in rj.THRESHOLDS
    assert rj.LIF_BOUND_AT_FROZEN_CONSTANTS in rj.THRESHOLDS


def pytest_approx(value, rel):
    import pytest
    return pytest.approx(value, rel=rel)
