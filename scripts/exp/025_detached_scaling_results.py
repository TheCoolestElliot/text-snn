"""EXP_025 resolver: applies §3's bars mechanically to §2's runs.

    python scripts/exp/025_detached_scaling_results.py \
        --out docs/reports/data/exp_025_scaling_results.json

COMMITTED BEFORE ANY RUN'S BPC IS READ.  `EXP_020` and `EXP_021` did not meet
that condition and report §20.7 records it; `EXP_022` and `EXP_023` did.

WHAT IT REFUSES TO DO
---------------------
  * It does not adopt, rank, or recommend.
  * It does not repair a bar that misfired; a corrected rule is referred.
  * It does not substitute a seed. A missing or diverged run reduces n and marks
    every verdict touching it **provisional** (§3.0).
  * It does not compare any number with a committed figure as if it were a bar
    (decision #10). The cross-tree comparisons it prints are labelled.

TWO DEFECTS IN THE PRE-REGISTRATION, FOUND WHILE WRITING THIS FILE AND BEFORE
ANY BPC WAS READ
-----------------------------------------------------------------------------
`CONTRIBUTING.md` §2 requires every bar be checked against its own limitations
section before it is committed.  For `EXP_025` that check happened here, one
commit late, and it found two things.  **Neither is repaired in the
pre-registration** -- that log is committed at `d9f8d45` and
`CONTRIBUTING.md` §3's "a bar is reported as it fired" governs.  Both are
resolved *forward*, in this file, mechanically.

**1. H3's stated Fisher p-value is wrong, and this file computes it instead of
quoting it.**  §3's H3 says Fisher's exact on 6/6 against 1/4 gives
**p = 0.0048**.  It does not.  With row totals (6, 4) and column totals (7, 3),
the one-sided probability is `C(7,6)*C(3,0) / C(10,6) = 7/210` = **0.0333** --
**6.9x** the stated figure.  The direction of the claim survives and its force
does not: 0.0333 is a far weaker statement than 0.0048, and it is a *secondary,
cross-tree* comparison whose verdict H3 was already written not to depend on.
`fisher_exact_one_sided` is module level **so it can be tested**, and
`_selfcheck_fisher` checks it against hand-computed tables before any verdict
is emitted.  This is the fourth instance of open decision #9's class -- a bar
whose guard clause was written and whose arithmetic was not -- after `EXP_012`'s
Y5, `EXP_013`'s N1 and `EXP_022`'s H4.

**2. H2's equivalence margin cannot be the measured sigma, and reading it that
way would make the test degenerate.**  §3 states the margin as `2*sigma_DoD`
while §3.0 says "bars are scored against sigma measured in this experiment".
Taken together those make TOST vacuous: if the margin is `2*se_DoD` and the 90 %
interval's half-width is `t_90 * se_DoD` (~1.9 * se at df ~ 4), equivalence
would require `|DoD| < 0.1 * se_DoD` -- a bar essentially nothing can clear, so
"UNRESOLVED" would be pre-determined by the arithmetic rather than measured.
**The margin is therefore read as the pre-declared constant 0.010648 bpc**,
which is what §3.0's own sizing table publishes for it, computed from the
*conservative transferred* sigma = 0.00461 that the same table says was chosen
"so every bar quoted here is the conservative one".  The measured sigma is used
for the test statistic's standard error, which is what §3.0's sentence is
actually about.  **Both readings are emitted** -- `margin_prereg_constant` and
`margin_from_measured_sigma` -- so a reader can see the choice rather than
inherit it, and H2 is UNRESOLVED if the two disagree.

STATISTICS ARE IMPORTED, NOT REIMPLEMENTED
------------------------------------------
`EXP_011`'s resolver set the precedent: one implementation of each statistic,
not one file.  The Student-t and the Welch interval come from
`018_dopamine_results.py`, committed and closed; the context decomposition comes
from `010_phase4_arm_results.py`, likewise.  Both are imported by path rather
than edited.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import statistics
import sys
from math import comb
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO / "src") not in sys.path:
    sys.path.insert(0, str(_REPO / "src"))

RUNS = _REPO / "experiments" / "runs"
PREREG = _REPO / "experiments" / "logs" / "EXP_025_detached_scaling.md"
MANIFEST = _REPO / "docs" / "reports" / "data" / "exp_025_run_manifest.json"
HORIZON = _REPO / "docs" / "reports" / "data" / "exp_025_memory_horizon.json"


def _import_by_path(stem: str, alias: str):
    spec = importlib.util.spec_from_file_location(
        alias, _REPO / "scripts" / "exp" / f"{stem}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


RUNGS = (512, 1481)
ARCHES = ("snn", "twocomp_detach")
SEEDS = (0, 1, 2)

#: §3.0.  The conservative transferred sigma, and the three bars sized from it.
#: These are the PRE-DECLARED constants; measured sigma is reported beside them.
SIGMA_TRANSFERRED = 0.00461
BAR_H1_2SIGMA = 0.007528          # difference between arms at one rung
MARGIN_H2_DOD = 0.010648          # difference-of-differences across rungs
H4_FACTOR = 1.5                   # §3, H4
H5_SHARE = 0.5                    # §3, H5
ALPHA = 0.05


def run_name(arch: str, d: int, seed: int) -> str:
    short = {"snn": "snn", "twocomp_detach": "detach"}[arch]
    return f"e25_{short}_d{d}_s{seed}"


# ---------------------------------------------------------------------------
# Statistics that are this file's own, at module level SO THEY CAN BE TESTED
# ---------------------------------------------------------------------------

def fisher_exact_one_sided(a: int, b: int, c: int, d: int) -> float:
    """One-sided Fisher p for the 2x2 table [[a, b], [c, d]].

    `a` is the count of interest in row 1.  Returns P(X >= a) under the
    hypergeometric null with both margins fixed.  Written as a function rather
    than an inline expression **because the inline version is what let
    `EXP_025` §3's 0.0048 go unchecked** -- nothing could assert on it.
    """
    n = a + b + c + d
    r1, c1 = a + b, a + c
    lo, hi = max(0, c1 - (n - r1)), min(r1, c1)
    denom = comb(n, r1)
    return sum(comb(c1, k) * comb(n - c1, r1 - k)
               for k in range(a, hi + 1)) / denom if lo <= a <= hi else float("nan")


def non_beyond_share(dec: dict) -> float | None:
    """H5's statistic: what fraction of the gain is NOT beyond the horizon.

    `010.decompose` is additive by construction -- `total = zero + within +
    beyond` -- so the three shares sum to 1 and this is exactly
    `1 - beyond_horizon_share`.  **It must be computed from the SIGNED shares.**
    The first version of this took `(|zero| + |within|) / |total|`, which on an
    `io_mall`-shaped decomposition (zero and within of opposite sign) returns
    **3.393** -- a "share" of 339 %, and a bar that would clear its 50 %
    threshold on any arm whose components merely disagreed with each other.

    Module level SO IT CAN BE TESTED, for the same reason
    `fisher_exact_one_sided` is: the wrong version was an inline expression
    inside `resolve_h5`, where nothing could assert on it.
    """
    total = dec["total_gain_bpc"]
    if not total:
        return None
    return (dec["zero_context_gain_bpc"] + dec["within_reach_gain_bpc"]) / total


def wilson_ci(k: int, n: int, z: float = 1.959963985) -> list[float]:
    """Wilson score interval.  §3's H3 reports a rate with its interval and its
    denominator, per `CONTRIBUTING.md` §3; Wilson is used because it is closed
    form and does not collapse at k = n, where the normal interval reports a
    point."""
    if n == 0:
        return [float("nan"), float("nan")]
    p = k / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = (z / denom) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return [max(0.0, centre - half), min(1.0, centre + half)]


def _selfcheck_fisher() -> dict:
    """Checked against hand-computed tables before any verdict is emitted.

    `EXP_017` §9.8's lesson: a statistic no test can fail is not evidence.
    """
    rows = [
        # (a, b, c, d, expected one-sided p, why)
        (6, 0, 1, 3, 7 / 210, "EXP_025 H3's own table: C(7,6)C(3,0)/C(10,6)"),
        (3, 0, 0, 3, 1 / 20, "the 3v3 table; C(3,3)C(3,0)/C(6,3) = 1/20"),
        (1, 1, 1, 1, 5 / 6, "the null-est table there is"),
    ]
    worst, out = 0.0, []
    for a, b, c, d, want, why in rows:
        got = fisher_exact_one_sided(a, b, c, d)
        worst = max(worst, abs(got - want))
        out.append({"table": [a, b, c, d], "expected": want, "got": got, "why": why})
    if worst >= 1e-12:
        raise SystemExit(
            f"the hand-rolled Fisher test is wrong: worst |diff| = {worst:.2e}. "
            "No bar may be resolved with it.")
    return {"passed": True, "worst_abs_diff": worst, "rows": out}


def tost(delta: float, se: float, margin: float, df: float, t_ppf) -> dict:
    """Two one-sided tests for equivalence within +/- `margin`.

    §3's H2 is an equivalence claim and is scored as one: "we failed to find a
    difference" is not a result, so FLAT requires the (1 - 2*ALPHA) interval to
    lie INSIDE the margin, not merely to contain zero.
    """
    if not (se > 0) or not (margin > 0):
        return {"verdict": "UNRESOLVED", "why": "no usable standard error"}
    tcrit = t_ppf(1.0 - ALPHA, df)
    half = tcrit * se
    lo, hi = delta - half, delta + half
    inside = (lo > -margin) and (hi < margin)
    resolves_away = (lo > margin) or (hi < -margin)
    if inside:
        verdict = "FLAT"
    elif resolves_away:
        verdict = "GROWS" if delta < 0 else "ERODES"
    else:
        verdict = "UNRESOLVED"
    return {
        "verdict": verdict, "delta": delta, "se": se, "df": df,
        "margin": margin, "t_crit_one_sided": tcrit,
        "ci_1_minus_2alpha": [lo, hi],
        "interval_inside_margin": inside,
        "note": ("GROWS means the detached arm's ADVANTAGE grew, i.e. the "
                 "difference-of-differences went more negative; the sign is "
                 "stated because 'grows' and 'delta increases' point opposite "
                 "ways here"),
    }


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_bpc(run: str) -> dict | None:
    """Both protocols, always -- `src/snn/evaluate.py`'s standing rule."""
    p = RUNS / run / "final_test.json"
    if not p.exists():
        return None
    res = json.loads(p.read_text(encoding="utf-8")).get("results", {})
    return {"fresh": res.get("fresh", {}).get("bpc"),
            "carried": res.get("carried", {}).get("bpc")}


def load_cells() -> dict:
    """Every (arch, rung, seed) cell, with divergence recorded rather than
    dropped.  A missing run is `None`, and §3.0's reduced-n rule reads it."""
    manifest = (json.loads(MANIFEST.read_text(encoding="utf-8"))
                if MANIFEST.exists() else {})
    by_run = {c["run"]: c for c in manifest.get("cells", [])}
    cells: dict = {}
    for arch in ARCHES:
        for d in RUNGS:
            got, diverged, not_run = [], [], []
            for s in SEEDS:
                run = run_name(arch, d, s)
                bpc = load_bpc(run)
                if bpc is None:
                    # No `final_test.json` at all: the run has not happened.
                    # NOT the same as a divergence, and conflating the two
                    # would let a bar fire a verdict on absent data.
                    not_run.append(run)
                    continue
                carried = bpc.get("carried")
                if carried is None or not math.isfinite(carried):
                    diverged.append(run)
                    continue
                got.append({"seed": s, "run": run, **bpc,
                            "clip": _clip_row(by_run.get(run, {}))})
            cells[(arch, d)] = {
                "scoreable": got, "diverged": diverged, "not_run": not_run,
                "missing": diverged + not_run,
                "n": len(got), "attempted": len(got) + len(diverged),
                "reduced_n": len(got) < len(SEEDS),
            }
    return cells


def _clip_row(cell: dict) -> dict:
    """D1 -- a diagnostic, not a bar."""
    return {k: cell.get(k) for k in
            ("wall_clock_s", "peak_vram_gib", "params")}


def _vals(cells: dict, arch: str, d: int, protocol: str) -> list[float]:
    return [c[protocol] for c in cells[(arch, d)]["scoreable"]
            if c.get(protocol) is not None]


# ---------------------------------------------------------------------------
# Bars
# ---------------------------------------------------------------------------

def resolve_h1(cells: dict, protocol: str, welch) -> dict:
    """H1 -- the advantage exists at 5.0M, with seeds.  A SANITY bar, and §3
    says so in advance."""
    a = _vals(cells, "twocomp_detach", 1481, protocol)
    b = _vals(cells, "snn", 1481, protocol)
    if len(a) < 2 or len(b) < 2:
        return {"verdict": "NOT RUN", "why": f"n = {len(a)} detach, {len(b)} snn",
                "provisional": True}
    w = welch(a, b)
    resolves = bool(w["excludes_zero"]) and abs(w["delta"]) >= BAR_H1_2SIGMA
    verdict = ("HELD" if (w["delta"] < 0 and resolves)
               else "UNRESOLVED")
    return {
        "statement": ("mean carried test bpc of twocomp_detach at d=1481 is "
                      "below snn's, resolving at >= 2 sigma"),
        "verdict": verdict, "welch": w,
        "bar_2sigma_transferred": BAR_H1_2SIGMA,
        "provisional": cells[("twocomp_detach", 1481)]["reduced_n"]
                       or cells[("snn", 1481)]["reduced_n"],
        "guards": ["H1 may not be reported as held unless D1 is published "
                   "beside it (EXP_025 §3)",
                   "H1 may not be reported as held unless H5's decomposition "
                   "is published beside it"],
        "power_stated_in_advance": ("EXP_017's committed n=1 reading is "
                                    "-0.13783, which is 18x this bar; a FAILURE "
                                    "here is a systems finding, not a "
                                    "scientific one (EXP_025 §4 rule 4)"),
    }


def resolve_h2(cells: dict, protocol: str, t_ppf) -> dict:
    """H2 -- the advantage is FLAT in width.  An EQUIVALENCE claim, scored by
    TOST against a pre-declared margin (see this file's header, defect 2)."""
    out: dict = {"statement": ("|Delta(d=1481) - Delta(d=512)| < margin; the "
                              "arm's advantage neither grows nor erodes across "
                              "2.76 octaves of parameters"),
                 "margin_prereg_constant": MARGIN_H2_DOD}
    means, ses, ns = {}, {}, {}
    for d in RUNGS:
        for arch in ARCHES:
            v = _vals(cells, arch, d, protocol)
            if len(v) < 2:
                return {**out, "verdict": "NOT RUN",
                        "why": f"{arch} d={d} has n = {len(v)}",
                        "provisional": True}
            means[(arch, d)] = statistics.fmean(v)
            ses[(arch, d)] = statistics.variance(v) / len(v)
            ns[(arch, d)] = len(v)

    delta = {d: means[("twocomp_detach", d)] - means[("snn", d)] for d in RUNGS}
    dod = delta[1481] - delta[512]
    var = sum(ses[k] for k in ses)
    se = math.sqrt(var)
    # Welch-Satterthwaite over the four means.
    num = var ** 2
    den = sum(ses[k] ** 2 / max(ns[k] - 1, 1) for k in ses)
    df = num / den if den > 0 else float(sum(ns.values()) - 4)

    primary = tost(dod, se, MARGIN_H2_DOD, df, t_ppf)
    alternate = tost(dod, se, 2.0 * se, df, t_ppf)
    out.update({
        "delta_per_rung": delta,
        "difference_of_differences": dod,
        "margin_from_measured_sigma": 2.0 * se,
        "primary": primary,
        "alternate_reading_margin_2se": alternate,
        "readings_agree": primary["verdict"] == alternate.get("verdict"),
        "verdict": (primary["verdict"]
                    if primary["verdict"] == alternate.get("verdict")
                    else "UNRESOLVED"),
        "why_unresolved_if_disagreeing": ("EXP_025 §3.0: a verdict that flips "
                                          "depending on which sigma is used is "
                                          "reported as UNRESOLVED, and both are "
                                          "published"),
        "guard": ("there is no middle rung, so NO per-octave slope may be "
                  "quoted from H2; 'flat' means flat across this interval, "
                  "never linear within it"),
        "provisional": any(cells[(a, d)]["reduced_n"] for a in ARCHES for d in RUNGS),
    })
    return out


def resolve_h3(cells: dict) -> dict:
    """H3 -- stability, scored on THIS experiment's runs alone."""
    total = trained = attempted = 0
    per_rung = {}
    for d in RUNGS:
        c = cells[("twocomp_detach", d)]
        per_rung[d] = {"trained": c["n"], "of": len(SEEDS),
                       "diverged": c["diverged"], "not_run": c["not_run"]}
        trained += c["n"]
        attempted += c["attempted"]
        total += len(SEEDS)
    # The interval is on what was ATTEMPTED. A run that has not happened is not
    # a divergence, and scoring it as one would fire a verdict on absent data.
    ci = wilson_ci(trained, attempted) if attempted else [float("nan")] * 2
    # Secondary, CROSS-TREE. EXP_017 §9.4 (1/3 at d=512) + EXP_015 (0/1 at 1481).
    p = fisher_exact_one_sided(trained, total - trained, 1, 3)
    return {
        "statement": ("6 of 6 twocomp_detach runs complete 20,000 steps with "
                      "zero non-finite losses"),
        "rate": f"{trained}/{attempted}",
        "rate_value": (trained / attempted) if attempted else None,
        "ci95_wilson": ci,
        "attempted": attempted, "pre_registered_n": total,
        "per_rung": per_rung,
        "verdict": ("NOT RUN" if attempted < total
                    else "HELD" if trained == total else "FAILED"),
        "secondary_cross_tree": {
            "labelled": "CROSS-TREE, and H3's verdict does not depend on it",
            "control": "twocomp: 1/3 trained at d=512 (EXP_017 §9.4), 0/1 at "
                       "d=1481 (EXP_015)",
            "fisher_one_sided_p": p,
            "correction": ("EXP_025 §3 states this p as 0.0048. It is 7/210 = "
                           "0.0333. The pre-registration is NOT edited "
                           "(CONTRIBUTING.md §3); the value is computed here "
                           "and the discrepancy is reported as a defect."),
            "prereg_stated": 0.0048,
        },
        "note": ("a rate compared against a threshold carries an interval and "
                 "its denominator (CONTRIBUTING.md §3)"),
    }


def resolve_h4(cells: dict, protocol: str) -> dict:
    """H4 -- sigma at width, measured for the first time in this project."""
    out: dict = {"statement": ("the pooled within-arm sd at d=1481 differs from "
                              "the transferred 0.00461 by more than 1.5x"),
                 "sigma_transferred": SIGMA_TRANSFERRED,
                 "factor_threshold": H4_FACTOR, "per_cell": {}}
    for d in RUNGS:
        for arch in ARCHES:
            v = _vals(cells, arch, d, protocol)
            out["per_cell"][f"{arch}_d{d}"] = {
                "n": len(v),
                "mean": statistics.fmean(v) if v else None,
                "sd": statistics.stdev(v) if len(v) > 1 else None,
            }
    top = [out["per_cell"][f"{a}_d1481"]["sd"] for a in ARCHES]
    if any(s is None for s in top):
        return {**out, "verdict": "NOT RUN", "why": "a d=1481 cell has n < 2"}
    pooled = math.sqrt(statistics.fmean([s * s for s in top]))
    ratio = pooled / SIGMA_TRANSFERRED
    out.update({
        "pooled_sd_d1481": pooled,
        "ratio_to_transferred": ratio,
        "verdict": "HELD" if (ratio > H4_FACTOR or ratio < 1 / H4_FACTOR)
                   else "UNRESOLVED",
        "deliverable": ("the NUMBER, not the bar: every EXP_014 bar placed "
                        "against the transferred 0.00461 is re-reported against "
                        "this pooled sd in §9"),
        "exp_014_bars_rescored": {
            "S1_margin_required_bpc_at_transferred": 0.0461,
            "S1_margin_required_bpc_at_measured": 10.0 * pooled,
            "S2_pair_gate_at_transferred": 0.00922,
            "S2_pair_gate_at_measured": 2.0 * pooled,
        },
    })
    return out


def resolve_h5(cells: dict) -> dict:
    """H5 -- where the bits are, decomposed by context.

    Needs the horizon sweep, which is a separate probe.  Absent it, this reports
    NOT RUN rather than inventing a decomposition -- `CONTRIBUTING.md` §3:
    prose that describes an artifact must name one that exists.
    """
    if not HORIZON.exists():
        return {"verdict": "NOT RUN",
                "why": f"{HORIZON.relative_to(_REPO)} does not exist yet; the "
                       "horizon sweep is a separate probe",
                "statement": ("the gain at d=1481 is NOT predominantly a "
                              "beyond-horizon gain: zero-context and "
                              "within-reach together carry >= 50 % of it")}
    r010 = _import_by_path("010_phase4_arm_results", "_r010")
    blob = json.loads(HORIZON.read_text(encoding="utf-8"))
    ref = r010.absolute_curve(blob, "e25_snn_d1481")
    arm = r010.absolute_curve(blob, "e25_detach_d1481")
    dec = r010.decompose(ref, arm)
    beyond = dec["beyond_horizon_gain_bpc"]
    share = non_beyond_share(dec)
    return {
        "statement": ("the gain at d=1481 is NOT predominantly a beyond-horizon "
                      "gain: zero-context and within-reach together carry "
                      ">= 50 % of it"),
        "decomposition": dec,
        "zero_plus_within_share": share,
        "verdict": ("HELD" if share is not None and share >= H5_SHARE
                    else "FAILED" if share is not None else "NOT RUN"),
        "beyond_horizon_is_a_MARKER_not_a_bar": {
            "value": beyond,
            "why": ("report §8's rev-15 note: this component has read outside "
                    "+/-0.0016 on 7 of 24 committed primary arm-gain readings, "
                    "three clearing a 2 sigma bar, and its sign carries a "
                    "compression artifact on losing arms"),
            "horizon_reported_beside_it": r010.median_horizon(blob, "e25_detach_d1481"),
            "inherited_conclusion": "two markers agreeing is two markers agreeing",
        },
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="docs/reports/data/exp_025_scaling_results.json")
    ap.add_argument("--protocol", default="carried", choices=("carried", "fresh"))
    args = ap.parse_args(argv)

    r018 = _import_by_path("018_dopamine_results", "_r018")
    fisher_check = _selfcheck_fisher()
    t_check = r018._selfcheck_t()

    cells = load_cells()
    out: dict = {
        "experiment": "EXP_025_detached_scaling",
        "log": "experiments/logs/EXP_025_detached_scaling.md",
        "protocol_primary": args.protocol,
        "adopts_nothing": True,
        "ranks_nothing": True,
        "rules_no_open_decision": True,
        "engages_decision_11": False,
        "selfchecks": {"student_t": t_check, "fisher": fisher_check},
        "sigma_transferred": SIGMA_TRANSFERRED,
        "bars_pre_declared": {"H1_2sigma": BAR_H1_2SIGMA,
                              "H2_margin_dod": MARGIN_H2_DOD,
                              "H4_factor": H4_FACTOR, "H5_share": H5_SHARE},
        "prereg_defects_found_before_any_bpc_was_read": [
            {"bar": "H3", "defect": "stated Fisher p = 0.0048; correct is 7/210 "
                                    "= 0.0333, a factor of 6.9",
             "repaired_in_prereg": False,
             "handled": "computed mechanically here; reported as it fired"},
            {"bar": "H2", "defect": "equivalence margin stated as 2*sigma_DoD "
                                    "while §3.0 says bars use measured sigma; "
                                    "together those make TOST degenerate",
             "repaired_in_prereg": False,
             "handled": "margin read as §3.0's pre-declared 0.010648; BOTH "
                        "readings emitted and disagreement resolves UNRESOLVED"},
        ],
        "resolver_defects_fixed_in_this_file": [
            {"bar": "H3", "defect": "returned FAILED against an empty tree -- a "
                                    "run that had not happened was counted as a "
                                    "divergence, firing a verdict on absent data",
             "found": "before the driver ran; before any bpc existed"},
            {"bar": "H5", "defect": "the non-beyond-horizon share was computed "
                                    "from ABSOLUTE components, which returns "
                                    "3.393 on an io_mall-shaped decomposition "
                                    "and would clear the 50 % threshold on any "
                                    "arm whose components merely disagreed in "
                                    "sign; now computed from the signed shares, "
                                    "which are additive and sum to 1",
             "found": "after the d=512 rung's bpc was read and BEFORE H5 had "
                      "ever fired -- H5's input artifact "
                      "(exp_025_memory_horizon.json) needs d=1481 checkpoints "
                      "that did not exist. Recorded rather than slipped in, "
                      "because the resolver's entry condition is that it was "
                      "committed before any bpc was read and this edit is after."},
        ],
        "cells": {f"{a}_d{d}": {"n": cells[(a, d)]["n"],
                                "missing": cells[(a, d)]["missing"],
                                "seeds": cells[(a, d)]["scoreable"]}
                  for a in ARCHES for d in RUNGS},
    }
    out["H1"] = resolve_h1(cells, args.protocol, r018.welch)
    out["H2"] = resolve_h2(cells, args.protocol, r018.t_ppf)
    out["H3"] = resolve_h3(cells)
    out["H4"] = resolve_h4(cells, args.protocol)
    out["H5"] = resolve_h5(cells)

    dest = Path(args.out)
    if not dest.is_absolute():
        dest = _REPO / dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")

    print(f"\nEXP_025 -- protocol: {args.protocol}")
    for h in ("H1", "H2", "H3", "H4", "H5"):
        print(f"  {h}  {out[h].get('verdict')}")
    print(f"\nwrote {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
