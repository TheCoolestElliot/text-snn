"""EXP_005: does sigma transfer from the one-state baseline to a two-state neuron?

Pre-registered in `experiments/logs/EXP_005_sigma_transfer.md`. Read that first --
S1 to S5, their thresholds and the 2x2 decision rule were all fixed before any of
the four extra seeds was launched, and this file only resolves them.

Reads, all produced by unmodified instruments:
  * `experiments/runs/{twocomp_s0..s6,snn_beta0.5_s0..s4}/final_test.json`
      -- scripts/evaluate.py, full test split, both protocols
  * `docs/reports/data/exp_005_memory_horizon.json`
      -- scripts/exp/001_memory_horizon.py, whose only change for this experiment
         was four registry entries (failure mode K5)

THE THREE BARS, AND WHY THERE ARE THREE
---------------------------------------
The pre-registration's 1.2 flags that the per-context bar in use is built on a
different scale from the quantity it judges. The bar is `2 x per-seed sd of the
baseline's excess curve`; the quantity is a difference between two arm MEANS. So
this script reports three bars rather than silently picking one:

  (a) as-built      2 * sd_excess(baseline, n=5)
                    EXP_004 10.4's bar, reproduced so its "5 of 128" stays
                    comparable.
  (b) SE on excess  2 * sqrt( s_ex,base^2 / 5  +  s_ex,twocomp^2 / 7 )
                    1.2's formula read on the excess curve.
  (c) SE on the absolute curve                                   <-- the correct one
                    2 * sqrt( s_abs,base^2 / 5  +  s_abs,twocomp^2 / 7 )

(c) is the one the comparison actually calls for, and the distinction is not
pedantic. The excess curve is measured relative to *each seed's own* asymptote, so
its spread excludes the seed-to-seed spread of the asymptote itself -- which is
0.0045 bpc, the very quantity EXP_000 called the noise floor. The delta being
tested contains both. At c >= 16 bar (a) is ~0.0003 while the asymptote noise
alone contributes ~0.0026, so (a) is not conservative there at all; it is roughly
an order of magnitude too small, and it happens not to matter only because the
two-compartment arm is far better than the baseline at those contexts and is
nowhere near being flagged.

S3 is resolved against (c). All three are reported, (a) first, so nothing in
EXP_004's table becomes uncomparable -- and so that a reader can see that the
choice of bar, not the data, moves the count.

WHAT THIS SCRIPT DOES NOT DO
----------------------------
It does not change 6.2's criterion. That is a Phase-3 deliverable and 4's rider
refers the change to Elliot with both numbers on the table. This script computes;
it does not adopt.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import statistics
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
RUNS = _REPO / "experiments" / "runs"

BASELINE_SEEDS = [f"snn_beta0.5_s{i}" for i in range(5)]
TWOCOMP_SEEDS = [f"twocomp_s{i}" for i in range(7)]

# --- everything below this line was fixed in the pre-registration ------------

#: EXP_000's whole-split sd, measured on the one-state baseline over 5 seeds.
BASELINE_SIGMA_CARRIED = 0.00461
BASELINE_SIGMA_FRESH = 0.00450
#: The bar every adoption verdict in the project has been held to.
TWO_SIGMA = 0.00922

#: S1's band, fixed in 3 before any seed ran.
S1_BAND = (0.7, 1.4)
#: S5: EXP_004's n=3 carried mean, and the tolerance it must hold to.
EXP_004_CARRIED_MEAN = 2.1174
EXP_004_FRESH_MEAN = 2.1443
#: EXP_004 10.4's count, which S3 predicts is a floor.
EXP_004_CONTEXTS_WORSE = 5
#: T1's threshold and T5's ceiling, re-resolved at n=7 for S2.
T1_HORIZON_MIN = 14
T5_CEILING_BPC = 0.2279
#: 5.2: what an F(6,4) test can actually detect, two-sided at 95%.
F_DETECTABLE_VARIANCE_RATIO = (0.1605, 9.20)


def _load_module(path: Path, name: str):
    """Import a module whose filename is not a valid identifier.

    `001_memory_horizon.py` owns `horizon_from_excess`, and re-implementing it
    here would be a second implementation of a statistic whose whole value is
    that there is only one of it.
    """
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def mean_sd(xs: list[float]) -> tuple[float, float]:
    n = len(xs)
    m = sum(xs) / n
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1)) if n > 1 else 0.0
    return m, sd


def test_bpc(run: str) -> dict[str, float]:
    path = RUNS / run / "final_test.json"
    if not path.exists():
        raise SystemExit(f"missing {path.relative_to(_REPO)}; run the seeds first")
    res = json.loads(path.read_text(encoding="utf-8"))["results"]
    return {p: res[p]["bpc"] for p in ("fresh", "carried")}


def whole_split(runs: list[str]) -> dict:
    vals = {p: [test_bpc(r)[p] for r in runs] for p in ("fresh", "carried")}
    out = {"runs": runs, "n": len(runs)}
    for p, xs in vals.items():
        m, sd = mean_sd(xs)
        out[p] = {"values": xs, "mean": m, "sd": sd}
    return out


def main(argv: list[str] | None = None) -> int:
    global TWOCOMP_SEEDS
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--horizon",
                    default="docs/reports/data/exp_005_memory_horizon.json")
    ap.add_argument("--out", default="docs/reports/data/exp_005_sigma_transfer.json")
    ap.add_argument("--twocomp-seeds", default=",".join(TWOCOMP_SEEDS),
                    help="override the arm's seed list. Its reason for existing is "
                         "the self-check: pointed at EXP_004's three seeds and "
                         "EXP_004's committed horizon JSON, this script must "
                         "reproduce EXP_004 §10.1's 2.1174 +/- 0.00487 and "
                         "§10.4's 5-of-128 -- numbers it did not produce, computed "
                         "by a different script.")
    args = ap.parse_args(argv)

    TWOCOMP_SEEDS = [s.strip() for s in args.twocomp_seeds.split(",") if s.strip()]

    probe = _load_module(_REPO / "scripts" / "exp" / "001_memory_horizon.py",
                         "exp001_horizon")

    # ================= whole-split sigma =================================
    base = whole_split(BASELINE_SEEDS)
    tc = whole_split(TWOCOMP_SEEDS)

    # Reproduce a number this script did not produce, on every run and not only
    # in the self-check mode. EXP_004 §10.1 and §10.7 committed the three-seed
    # mean and sd, computed by `scripts/exp/004_twocomp_results.py`; the first
    # three entries of this arm's seed list must still give them. It is the same
    # discipline as EXP_001's F1 check, and it is what would notice if
    # `final_test.json` had been regenerated by a different instrument, or if a
    # seed's directory had been silently reused.
    repro = {"checked": False}
    first3 = [r for r in ("twocomp_s0", "twocomp_s1", "twocomp_s2")
              if r in TWOCOMP_SEEDS]
    if len(first3) == 3:
        sub = whole_split(first3)
        repro = {
            "checked": True,
            "source": "EXP_004 §10.1 / §10.7, via scripts/exp/004_twocomp_results.py",
            "fields": {},
        }
        for name, got, want in (
                ("carried_mean", sub["carried"]["mean"], EXP_004_CARRIED_MEAN),
                ("fresh_mean", sub["fresh"]["mean"], EXP_004_FRESH_MEAN),
                ("carried_sd", sub["carried"]["sd"], 0.00487),
                ("fresh_sd", sub["fresh"]["sd"], 0.00445)):
            # The committed values are quoted to 4-5 dp, so agreement is asserted
            # at half an ulp of the quoted precision rather than at fp equality.
            tol = 5e-5 if "sd" in name else 5e-5
            repro["fields"][name] = {"ours": got, "committed": want,
                                     "residual": got - want,
                                     "held": abs(got - want) < tol}
        repro["held"] = all(f["held"] for f in repro["fields"].values())
        if not repro["held"]:
            bad = {k: v for k, v in repro["fields"].items() if not v["held"]}
            raise SystemExit(
                "EXP_004 reproduction FAILED -- the three seeds EXP_004 reported "
                f"no longer give its committed numbers: {bad}. Chase this before "
                "trusting anything below it; do not widen the tolerance.")

    ratio_carried = tc["carried"]["sd"] / BASELINE_SIGMA_CARRIED
    ratio_fresh = tc["fresh"]["sd"] / BASELINE_SIGMA_FRESH
    var_ratio = (tc["carried"]["sd"] ** 2) / (BASELINE_SIGMA_CARRIED ** 2)

    s1 = S1_BAND[0] <= ratio_carried <= S1_BAND[1]
    s5_delta = tc["carried"]["mean"] - EXP_004_CARRIED_MEAN
    s5 = abs(s5_delta) <= TWO_SIGMA

    # The bar this family should be held to from here on.
    new_sigma = tc["carried"]["sd"]
    new_two_sigma = 2 * new_sigma

    # ================= S2: do EXP_004's verdicts survive? =================
    delta_vs_base = tc["carried"]["mean"] - base["carried"]["mean"]
    gain = -delta_vs_base
    hz = json.loads((_REPO / args.horizon).read_text(encoding="utf-8"))
    tc_horizons = hz["by_arm"]["twocomp"]["_paired"]["horizon_2sigma_per_seed"]
    base_horizons = hz["by_arm"]["snn_beta0.5"]["_paired"]["horizon_2sigma_per_seed"]

    # The horizon is defined against a 2-sigma tolerance, so a new sigma moves it
    # too. Recomputed with the probe's own function rather than a second copy.
    tc_horizons_newbar = [
        probe.horizon_from_excess(
            hz["arms"][r]["paired_context"]["excess_bpc_at_context"], new_two_sigma)
        for r in TWOCOMP_SEEDS if r in hz["arms"]
    ]

    s2_checks = {
        "adoption_still_beyond_2sigma": {
            "gain_bpc": gain, "old_bar": TWO_SIGMA, "new_bar": new_two_sigma,
            "sigma_units_old": gain / BASELINE_SIGMA_CARRIED,
            "sigma_units_new": gain / new_sigma,
            "held": gain > new_two_sigma,
        },
        "T1_horizon_median_ge_14": {
            "per_seed": tc_horizons,
            "median": statistics.median([h for h in tc_horizons if h is not None]),
            "per_seed_at_new_bar": tc_horizons_newbar,
            "median_at_new_bar": statistics.median(
                [h for h in tc_horizons_newbar if h is not None]),
            "held": statistics.median(
                [h for h in tc_horizons_newbar if h is not None]) >= T1_HORIZON_MIN,
        },
        "T5_gain_below_ceiling": {
            "gain_bpc": gain, "ceiling": T5_CEILING_BPC,
            "held": gain < T5_CEILING_BPC,
        },
    }
    s2 = all(c["held"] for c in s2_checks.values())

    # ================= per-context bars ===================================
    contexts = sorted(
        int(c) for c in
        hz["arms"][TWOCOMP_SEEDS[0]]["paired_context"]["excess_bpc_at_context"])

    def per_seed_curves(runs: list[str]) -> tuple[dict, dict]:
        """(excess, absolute) per-context curves, one list of seed values per c."""
        ex: dict[int, list[float]] = {c: [] for c in contexts}
        ab: dict[int, list[float]] = {c: [] for c in contexts}
        for r in runs:
            entry = hz["arms"][r]
            # The k = L point: the untruncated score, i.e. this seed's asymptote.
            asym = entry["curve"][str(entry["seq_len"])]["bpc"]
            e = entry["paired_context"]["excess_bpc_at_context"]
            for c in contexts:
                ex[c].append(e[str(c)])
                ab[c].append(asym + e[str(c)])
        return ex, ab

    ex_b, ab_b = per_seed_curves(BASELINE_SEEDS)
    ex_t, ab_t = per_seed_curves(TWOCOMP_SEEDS)
    n_b, n_t = len(BASELINE_SEEDS), len(TWOCOMP_SEEDS)

    rows, counts = [], {"as_built": [], "se_excess": [], "se_absolute": []}
    for c in contexts:
        mb_ex, sb_ex = mean_sd(ex_b[c])
        mt_ex, st_ex = mean_sd(ex_t[c])
        mb_ab, sb_ab = mean_sd(ab_b[c])
        mt_ab, st_ab = mean_sd(ab_t[c])
        delta = mt_ab - mb_ab
        bar_a = 2 * sb_ex
        bar_b = 2 * math.sqrt(sb_ex ** 2 / n_b + st_ex ** 2 / n_t)
        bar_c = 2 * math.sqrt(sb_ab ** 2 / n_b + st_ab ** 2 / n_t)
        row = {
            "c": c, "delta_bpc": delta,
            "sd_excess_baseline": sb_ex, "sd_excess_twocomp": st_ex,
            "sd_absolute_baseline": sb_ab, "sd_absolute_twocomp": st_ab,
            "bar_as_built": bar_a, "bar_se_excess": bar_b, "bar_se_absolute": bar_c,
            "worse_as_built": delta > bar_a,
            "worse_se_excess": delta > bar_b,
            "worse_se_absolute": delta > bar_c,
        }
        rows.append(row)
        for key, flag in (("as_built", row["worse_as_built"]),
                          ("se_excess", row["worse_se_excess"]),
                          ("se_absolute", row["worse_se_absolute"])):
            if flag:
                counts[key].append(c)

    s3 = len(counts["se_absolute"]) > EXP_004_CONTEXTS_WORSE

    # S4: shape. Largest bar at short context, >=10x smaller by c >= 16.
    short = [r["sd_excess_twocomp"] for r in rows if r["c"] <= 4]
    long_ = [r["sd_excess_twocomp"] for r in rows if r["c"] >= 16]
    s4_peak_short = max(short) >= max(
        r["sd_excess_twocomp"] for r in rows if 4 < r["c"] < 16)
    s4_falls = max(short) >= 10 * (sum(long_) / len(long_))
    s4 = bool(s4_peak_short and s4_falls)

    # ================= F1, K1, K3 ========================================
    f1 = {r: hz["arms"][r].get("f1_pass") for r in TWOCOMP_SEEDS if r in hz["arms"]}
    manifest_path = _REPO / "docs/reports/data/exp_005_run_manifest.json"
    manifest = (json.loads(manifest_path.read_text(encoding="utf-8"))
                if manifest_path.exists() else None)

    results = {
        "experiment": "EXP_005_sigma_transfer",
        "log": "experiments/logs/EXP_005_sigma_transfer.md",
        "candidate": "#14 -- re-measure sigma on a two-state neuron (blocking)",
        "whole_split": {
            "baseline": base, "twocomp": tc,
            "baseline_committed_sigma": {"fresh": BASELINE_SIGMA_FRESH,
                                         "carried": BASELINE_SIGMA_CARRIED},
            "sd_ratio_carried": ratio_carried,
            "sd_ratio_fresh": ratio_fresh,
            "variance_ratio_carried": var_ratio,
            "f_test": {
                "df": [n_t - 1, n_b - 1],
                "detectable_variance_ratio_outside": F_DETECTABLE_VARIANCE_RATIO,
                "distinguishable_from_1": not (
                    F_DETECTABLE_VARIANCE_RATIO[0] < var_ratio
                    < F_DETECTABLE_VARIANCE_RATIO[1]),
                "note": "critical values fixed in the pre-registration (5.2); "
                        "reported as a resolution statement, not a p-value",
            },
            "new_bar_for_this_family": {"sigma": new_sigma,
                                        "two_sigma": new_two_sigma},
        },
        "per_context": {
            "n_baseline": n_b, "n_twocomp": n_t,
            "bars_explained": {
                "as_built": "2 * per-seed sd of the baseline's EXCESS curve "
                            "(EXP_004 10.4's bar)",
                "se_excess": "2 * sqrt(s_ex,b^2/n_b + s_ex,t^2/n_t)",
                "se_absolute": "2 * sqrt(s_abs,b^2/n_b + s_abs,t^2/n_t) -- the "
                               "correct scale for a difference of arm means",
            },
            "contexts_worse": {k: v for k, v in counts.items()},
            "n_contexts_worse": {k: len(v) for k, v in counts.items()},
            "exp_004_reported": EXP_004_CONTEXTS_WORSE,
            "rows": rows,
        },
        "predictions": {
            "S1": {"statement": "carried sd within [0.7x, 1.4x] of 0.00461",
                   "measured_ratio": ratio_carried, "band": list(S1_BAND),
                   "held": s1,
                   "power_note": "at n=7 the 95% CI for sigma spans [0.64x, 2.20x] "
                                 "of the estimate, so holding S1 is weak evidence "
                                 "and failing it is strong evidence (3)"},
            "S2": {"statement": "no EXP_004 verdict changes at the new bar",
                   "checks": s2_checks, "held": s2},
            "S3": {"statement": "worse at MORE than 5 of 128 contexts against the "
                                "standard-error bar",
                   "measured": len(counts["se_absolute"]),
                   "exp_004_reported": EXP_004_CONTEXTS_WORSE, "held": s3,
                   "direction": "stated in the direction that hurts the adopted arm"},
            "S4": {"statement": "per-context sd keeps the baseline's shape",
                   "peak_at_short_context": bool(s4_peak_short),
                   "falls_10x_by_c16": bool(s4_falls), "held": s4},
            "S5": {"statement": "n=7 carried mean within 2 sigma of EXP_004's 2.1174",
                   "n3_mean": EXP_004_CARRIED_MEAN,
                   "n7_mean": tc["carried"]["mean"],
                   "delta": s5_delta, "tolerance": TWO_SIGMA, "held": s5,
                   "direction": "stated so that it can fail"},
        },
        "decision_cell": (
            "S5 holds and S2 holds -> #14 CLOSED" if (s5 and s2) else
            "S5 holds, S2 fails -> #14 closed, EXP_004 gets an appended correction"
            if s5 else
            "S5 FAILS -> EXP_004's headline was measured on three lucky seeds; the "
            "arm is re-reported at n=7 and the adoption is re-opened"),
        "reproduction_checks": {
            "exp_004_three_seed_numbers": repro,
            "f1_per_new_checkpoint": f1,
            "f1_all_pass": all(v is not False for v in f1.values()),
            "baseline_horizons_unchanged": base_horizons,
            "run_manifest": manifest["runs"] if manifest else None,
        },
    }

    out = _REPO / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")

    # ---------------- report ------------------------------------------------
    print("=" * 78)
    print("EXP_005 -- does sigma transfer to a two-state neuron?  (candidate #14)")
    print("=" * 78)
    print(f"\n  whole-split test bpc, full split")
    for label, blk in (("baseline", base), ("twocomp ", tc)):
        print(f"    {label}  n={blk['n']}   fresh {blk['fresh']['mean']:.4f} +/- "
              f"{blk['fresh']['sd']:.5f}   carried {blk['carried']['mean']:.4f} "
              f"+/- {blk['carried']['sd']:.5f}")
    print(f"\n    sigma ratio (carried): {ratio_carried:.3f}x   "
          f"variance ratio {var_ratio:.3f}   "
          f"distinguishable from 1 at F(6,4): "
          f"{results['whole_split']['f_test']['distinguishable_from_1']}")
    print(f"    new bar for this family: 2 sigma = {new_two_sigma:.5f} "
          f"(was {TWO_SIGMA})")

    print(f"\n  per-context contexts significantly worse than the baseline:")
    for key, label in (("as_built", "as built (EXP_004's bar)"),
                       ("se_excess", "SE on excess"),
                       ("se_absolute", "SE on absolute  <- correct")):
        cs = counts[key]
        rng = f"c = {min(cs)}..{max(cs)}" if cs else "none"
        print(f"    {label:28s} {len(cs):3d} / 128   {rng}")

    print(f"\n  predictions, resolved as written:")
    for k, v in results["predictions"].items():
        print(f"    {k}  {'HELD' if v['held'] else 'FAILED':6s}  {v['statement']}")

    if repro.get("checked"):
        worst = max(abs(f["residual"]) for f in repro["fields"].values())
        print(f"\n  reproduces EXP_004's committed 3-seed numbers: "
              f"{'yes' if repro['held'] else 'NO'}  "
              f"(worst residual {worst:.2e} bpc)")
    print(f"\n  F1 self-check on the new checkpoints: "
          f"{'all pass' if results['reproduction_checks']['f1_all_pass'] else 'FAILURE'}")
    print(f"\n  DECISION: {results['decision_cell']}")
    print(f"\nWROTE {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
