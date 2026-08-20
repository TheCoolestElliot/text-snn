"""The one statistic `EXP_022`'s resolver adds on its own authority.

`EXP_022` §4's H4 is pre-registered as `|c0| > |total|`, and the resolver's own
notes establish that the bar is defective: the decomposition is additive, so
`|c0| > |total|` requires the rung to LOSE outside zero context, and the 83 %
figure H4 is motivated by (`binin`: −0.0448 total, −0.0371 at c = 0) does not
satisfy it. The pre-registered statistic is still emitted as it fires —
`CONTRIBUTING.md` §3 — and `gain_is_at_zero_context` is emitted beside it as a
labelled post-hoc correction.

**A replacement statistic invented by a resolver is the one number in the
artifact with no pre-registration behind it, so it is the one that most needs a
gate.** It shipped with its sign inverted: `010.decompose` returns
*positive = the arm is better*, and the flag was written `total < 0`, making it
True only when a rung was worse. Nothing could observe that, because the
expression was inline inside `main()`. These tests exist so it stays observable.

Not an R10 gate: no scan, no kernel, no hand-written backward.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]


def _resolver():
    """Imported by path, the same way `EXP_011`'s resolver imports `010`."""
    spec = importlib.util.spec_from_file_location(
        "_r022", _REPO / "scripts" / "exp" / "022_input_output_results.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_r022"] = mod
    spec.loader.exec_module(mod)
    return mod


def _decompose():
    spec = importlib.util.spec_from_file_location(
        "_r010", _REPO / "scripts" / "exp" / "010_phase4_arm_results.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_r010"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_decompose_still_returns_positive_for_a_better_arm():
    """The contract the corrected statistic depends on, asserted rather than read.

    If `010.decompose` ever flips its sign, `gain_is_at_zero_context` flips
    meaning silently. `010` is committed and closed, so this test is a tripwire
    on an imported contract, not a request to change it.
    """
    r010 = _decompose()
    cmax = r010.CMAX
    # An arm that is better everywhere: lower bpc at every context.
    ref = {"0": 4.0, "7": 2.5, cmax: 2.30}
    arm = {"0": 3.9, "7": 2.4, cmax: 2.20}
    d = r010.decompose(ref, arm)
    assert d["total_gain_bpc"] > 0, "positive = the arm is better"
    assert d["zero_context_gain_bpc"] > 0
    # Additivity, which is what makes the pre-registered H4 unsatisfiable by a
    # clean gain and is the whole reason a correction was needed.
    assert abs(d["total_gain_bpc"]
               - (d["zero_context_gain_bpc"] + d["within_reach_gain_bpc"]
                  + d["beyond_horizon_gain_bpc"])) < 1e-12


def test_gain_flag_fires_on_a_gain_and_not_on_a_loss():
    m = _resolver()
    # A real gain, most of it at c = 0: the canonical success H4 was written for.
    assert m.gain_is_at_zero_context(zero=+0.030, total=+0.040) is True
    assert m.loss_is_at_zero_context(zero=+0.030, total=+0.040) is False
    # The same shape as a LOSS must not read as a gain. This is the case the
    # inverted version reported as "the gain is at zero context".
    assert m.gain_is_at_zero_context(zero=-0.030, total=-0.040) is False
    assert m.loss_is_at_zero_context(zero=-0.030, total=-0.040) is True


def test_the_flag_is_not_true_of_binins_own_cost():
    """§2.1's motivating row, fed back through the statistic that replaced its bar.

    `binin` is −0.0448 total with −0.0371 at c = 0 under *positive = better*,
    i.e. a pure cost with 83 % of it at zero context. The inverted flag fired
    `gain_is_at_zero_context = True` on exactly this row — announcing a gain on
    the experiment's canonical cost. It must read False, and the loss mirror
    must read True.
    """
    m = _resolver()
    zero, total = -0.0371, -0.0448
    assert m.gain_is_at_zero_context(zero, total) is False
    assert m.loss_is_at_zero_context(zero, total) is True
    # And the 83 % that motivates H4 is recovered from the same two numbers,
    # which is what makes this row the right one to test against.
    assert 0.82 < zero / total < 0.84


def test_a_gain_spread_thinly_over_zero_context_does_not_qualify():
    """The >= 50 % share is the part of the correction that was NOT wrong, so it
    is pinned separately from the sign."""
    m = _resolver()
    assert m.gain_is_at_zero_context(zero=+0.010, total=+0.040) is False
    assert m.gain_is_at_zero_context(zero=+0.020, total=+0.040) is True
    # A rung whose c = 0 component moved the WRONG WAY inside a net gain is not
    # a gain at zero context, whatever its magnitude. The pre-registered
    # `abs(z) > abs(t)` admits it; this does not.
    assert m.gain_is_at_zero_context(zero=-0.060, total=+0.040) is False
