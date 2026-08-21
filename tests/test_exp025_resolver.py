"""The three statistics `EXP_025`'s resolver adds on its own authority.

`EXP_025` §3's H3 states a Fisher p-value of **0.0048** for the 6/6-against-1/4
table.  It is **0.0333** — `C(7,6)*C(3,0)/C(10,6)` = 7/210 — a factor of 6.9.
That figure reached a *committed* pre-registration because it was arithmetic
done in prose, where nothing could assert on it.  `CONTRIBUTING.md` §3 forbids
repairing it there, so it is computed forward instead, and these tests are what
make the forward computation trustworthy.

**Every test here was verified to fail against a wrong implementation before it
was trusted** (`CONTRIBUTING.md` §5, and `scripts/audit/08_mutation_campaign.py`
is the standing instrument for the same principle in `src/`):

  * `fisher_exact_one_sided` — checked against three hand-computed tables, and
    against the specific value the pre-registration got wrong.
  * `wilson_ci` — checked at `k = n`, where the normal interval degenerates to a
    point and would report certainty from six runs.
  * `tost` — checked to REFUSE equivalence when the interval straddles the
    margin, which is the failure mode that makes an equivalence claim
    worthless: "we did not find a difference" is not "there is no difference".

Not an R10 gate: no scan, no kernel, no hand-written backward.
"""
from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]


def _resolver():
    """Imported by path, the same way `EXP_022`'s test imports its own."""
    spec = importlib.util.spec_from_file_location(
        "_r025", _REPO / "scripts" / "exp" / "025_detached_scaling_results.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_r025"] = mod
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# fisher_exact_one_sided
# ---------------------------------------------------------------------------

def test_fisher_reproduces_the_value_the_prereg_got_wrong():
    """H3's own table. 7/210, not the 0.0048 §3 states."""
    r = _resolver()
    got = r.fisher_exact_one_sided(6, 0, 1, 3)
    assert math.isclose(got, 7 / 210, rel_tol=0, abs_tol=1e-15)
    assert not math.isclose(got, 0.0048, abs_tol=1e-4), (
        "the pre-registered 0.0048 must not be reproducible by this function; "
        "if it becomes so, one of the two is wrong and it is no longer this one"
    )


def test_fisher_against_hand_computed_tables():
    r = _resolver()
    # 3 of 3 against 0 of 3: C(3,3)C(3,0)/C(6,3) = 1/20.
    assert math.isclose(r.fisher_exact_one_sided(3, 0, 0, 3), 1 / 20, abs_tol=1e-15)
    # The most null table there is: P(X >= 1) with margins all 2 of 4.
    assert math.isclose(r.fisher_exact_one_sided(1, 1, 1, 1), 5 / 6, abs_tol=1e-15)
    # A table with no evidence at all in the direction asked about.
    assert math.isclose(r.fisher_exact_one_sided(0, 3, 3, 0), 1.0, abs_tol=1e-15)


def test_fisher_is_one_sided_and_monotone_in_the_count_of_interest():
    """A larger `a` at fixed margins cannot be LESS surprising."""
    r = _resolver()
    ps = [r.fisher_exact_one_sided(a, 4 - a, 4 - a, a) for a in range(5)]
    assert ps == sorted(ps, reverse=True), ps


def test_fisher_selfcheck_runs_and_passes():
    r = _resolver()
    out = r._selfcheck_fisher()
    assert out["passed"] and out["worst_abs_diff"] == 0.0


# ---------------------------------------------------------------------------
# wilson_ci
# ---------------------------------------------------------------------------

def test_wilson_at_k_equals_n_does_not_report_certainty():
    """6/6 must not come back as [1.0, 1.0]. §3 quotes [0.61, 1.00]."""
    r = _resolver()
    lo, hi = r.wilson_ci(6, 6)
    assert hi == 1.0
    assert 0.60 < lo < 0.62, lo
    assert lo < 1.0, "an interval that excludes nothing is not an interval"


def test_wilson_is_wider_at_smaller_denominators():
    """The denominator states the metric's resolution (`CONTRIBUTING.md` §3)."""
    r = _resolver()
    narrow = r.wilson_ci(6, 6)
    wide = r.wilson_ci(3, 3)
    assert (wide[1] - wide[0]) > (narrow[1] - narrow[0])


def test_wilson_is_symmetric_under_relabelling():
    r = _resolver()
    lo_a, hi_a = r.wilson_ci(2, 7)
    lo_b, hi_b = r.wilson_ci(5, 7)
    assert math.isclose(lo_a, 1.0 - hi_b, abs_tol=1e-12)
    assert math.isclose(hi_a, 1.0 - lo_b, abs_tol=1e-12)


# ---------------------------------------------------------------------------
# tost
# ---------------------------------------------------------------------------

def _t_ppf():
    spec = importlib.util.spec_from_file_location(
        "_r018", _REPO / "scripts" / "exp" / "018_dopamine_results.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_r018"] = mod
    spec.loader.exec_module(mod)
    return mod.t_ppf


def test_tost_refuses_equivalence_when_the_interval_straddles_the_margin():
    """The whole point of an equivalence test: a wide interval around zero is
    UNRESOLVED, never FLAT."""
    r = _resolver()
    out = r.tost(delta=0.0, se=0.02, margin=0.010648, df=4.0, t_ppf=_t_ppf())
    assert out["verdict"] == "UNRESOLVED", out


def test_tost_returns_flat_only_when_the_interval_fits_inside_the_margin():
    r = _resolver()
    out = r.tost(delta=0.0, se=0.0005, margin=0.010648, df=4.0, t_ppf=_t_ppf())
    assert out["verdict"] == "FLAT", out
    assert out["interval_inside_margin"]


def test_tost_names_the_direction_the_advantage_moved():
    """A DoD more negative means the detached arm's ADVANTAGE grew, so the
    verdict must not be read off the raw sign of delta."""
    r = _resolver()
    grew = r.tost(delta=-0.05, se=0.001, margin=0.010648, df=4.0, t_ppf=_t_ppf())
    eroded = r.tost(delta=+0.05, se=0.001, margin=0.010648, df=4.0, t_ppf=_t_ppf())
    assert grew["verdict"] == "GROWS", grew
    assert eroded["verdict"] == "ERODES", eroded


def test_tost_is_unresolved_without_a_usable_standard_error():
    r = _resolver()
    assert r.tost(0.0, 0.0, 0.01, 4.0, _t_ppf())["verdict"] == "UNRESOLVED"


# ---------------------------------------------------------------------------
# The bar must not fire a verdict on absent data
# ---------------------------------------------------------------------------

def test_h3_is_not_run_when_nothing_has_been_attempted():
    """A run that has not happened is not a divergence. Scoring it as one made
    H3 read FAILED against an empty tree, which is a verdict on absent data."""
    r = _resolver()
    empty = {(a, d): {"scoreable": [], "diverged": [], "not_run": ["x"],
                      "missing": ["x"], "n": 0, "attempted": 0,
                      "reduced_n": True}
             for a in r.ARCHES for d in r.RUNGS}
    assert r.resolve_h3(empty)["verdict"] == "NOT RUN"


def test_h3_distinguishes_a_divergence_from_a_run_that_never_started():
    r = _resolver()
    cells = {(a, d): {"scoreable": [], "diverged": [], "not_run": [],
                      "missing": [], "n": 0, "attempted": 0, "reduced_n": True}
             for a in r.ARCHES for d in r.RUNGS}
    for d in r.RUNGS:
        cells[("twocomp_detach", d)] = {
            "scoreable": [{"seed": s} for s in (0, 1)], "diverged": ["dead"],
            "not_run": [], "missing": ["dead"], "n": 2, "attempted": 3,
            "reduced_n": True}
    out = r.resolve_h3(cells)
    assert out["verdict"] == "FAILED", out
    assert out["rate"] == "4/6"
