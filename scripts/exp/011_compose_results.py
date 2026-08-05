"""Resolve EXP_011 against its pre-registration, mechanically.

Every threshold this script applies was fixed in
`experiments/logs/EXP_011_composition.md` **before** the composed arm was
trained, and this file is committed before any of its numbers are read. It reads
committed artifacts and applies the stated bars; it does not choose them. A
script that both picks the bar and reports the score is a script that can
rationalise (`scripts/exp/004_twocomp_results.py`'s docstring, followed here for
the third time).

WHY THIS IS A SEPARATE FILE FROM `010_phase4_arm_results.py`
------------------------------------------------------------
010 is committed and closed: it resolved EXP_007, EXP_008 and EXP_009, and its
numbers are quoted in the Phase-4 interim report. Adding a fourth arm to it means
editing a file whose output three closed experiments depend on, and the failure
mode is obvious -- a change intended for EXP_011 that moves an EXP_008 number.
So the decision rules live here and the **statistics are imported from there**,
which is the arrangement EXP_006's failure mode L5 actually asks for: one
implementation of each statistic, not one file.

What is imported and therefore NOT reimplemented: `reference` (which recomputes a
committed mean and asserts it against the reports), `decompose` (EXP_004 §10.3),
`context_comparison` (§6.2), `absolute_curve`, `median_horizon`,
`learned_parameters`, and every shared constant. If any of those moves, this
script moves with it.

WHAT IS REIMPLEMENTED, AND WHY IT HAS TO BE
--------------------------------------------
`fold_check`. 010's version is hard-wired to fold into `arch="snn"` -- it is
EXP_008's fold, from the pre-scan arm down to the Phase-2 baseline. EXP_011's
fold is a *different* one: from the composed arm down to `arch="twocomp"`, via
`fold_into_twocomp_state_dict`. Same identity, different target, so it is a
different procedure rather than a duplicated statistic. The bpc it produces is
still compared through `snn.evaluate.evaluate`, the same code path that produced
every committed number, which is what makes it a check rather than a tautology.

THE BAR, AND WHY IT IS NOT THE ONE THE TASKING NAMED
-----------------------------------------------------
EXP_011 §0 item 1 records this in full. C1 is **≤ 2.11869**, the adopted arm's
carried figure, because that is what the hypothesis says. The tasking's 2.083 is
reported by this script as a stretch marker with no verdict attached, so the
number is not lost and cannot flip a verdict it was never derived for.
"""

from __future__ import annotations

import argparse
import dataclasses
import importlib.util
import json
import math
import sys
from pathlib import Path

import torch

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO / "src") not in sys.path:
    sys.path.insert(0, str(_REPO / "src"))

from snn.config import Config  # noqa: E402
from snn.data import Corpus  # noqa: E402
from snn.evaluate import evaluate  # noqa: E402
from snn.model import build_model  # noqa: E402


def _load_010():
    """Import `010_phase4_arm_results.py`, whose name is not an identifier.

    importlib rather than a rename: renaming a committed file that three closed
    experiments' artifacts point at would break the trail from a report back to
    the script that produced its numbers.
    """
    path = _REPO / "scripts" / "exp" / "010_phase4_arm_results.py"
    spec = importlib.util.spec_from_file_location("_exp010", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_exp010"] = mod
    spec.loader.exec_module(mod)
    return mod


R = _load_010()
RUNS = _REPO / "experiments" / "runs"

# ===========================================================================
# Everything below this line was fixed in EXP_011's pre-registration.
# ===========================================================================

ARM = "compose"
COMPOSE_RUNS = [f"compose_s{i}" for i in range(3)]
THRESHOLD_RUNS = [f"threshold_s{i}" for i in range(3)]

#: C1. The adopted arm's carried figure (EXP_005, n=7). `R.reference` recomputes
#: it from per-run artifacts and aborts if it does not reproduce, so this literal
#: is guarded rather than trusted.
C1_BAR = 2.11869

#: C2.
C2_SD_MAX = 0.005

#: C3. Full additivity = the adopted arm's figure minus the threshold arm's
#: measured gain. Recomputed below; this literal is the pre-registered value and
#: is asserted against the recomputation.
C3_ADDITIVE_BPC = 2.05973

#: EXP_008's committed carried mean, for the gain that C3 is built on.
THRESHOLD_REPORTED_CARRIED = 2.19416

#: §4.3 -- reported, never a verdict. See the module docstring.
STRETCH_TASKING = 2.083
STRETCH_HALF_GAIN = 2.08921


def _mean_sd_of(runs: list[str], protocol: str) -> tuple[float, float]:
    vals = [R._test_bpc(r)[protocol] for r in runs]
    return R._mean_sd(vals)


def fold_check_composed(run: str) -> dict:
    """Fold `exp(-theta)` into each layer's Linear, load as `arch="twocomp"`, score.

    EXP_011 §1.1's Identity 2, executed on the composed arm. The folded model is
    built through `build_model` and scored by `snn.evaluate.evaluate` -- a
    different code path from the arm's own forward, which is what makes this a
    check rather than a tautology.

    **The residual is reported and not gated.** EXP_011 C4 says why: the fold-in
    tolerance is Elliot's decision #6 and is still open, and `EXP_008` §9.5 chased
    the same residual to a discontinuity rather than to a broken identity. This
    function returns the number and states no verdict.
    """
    ck = torch.load(RUNS / run / "ckpt_final.pt", map_location="cpu",
                    weights_only=False)
    known = {f.name for f in dataclasses.fields(Config)}
    cfg = Config(**{k: v for k, v in ck["config"].items() if k in known})

    arm = build_model(cfg)
    arm.load_state_dict(ck["model"])
    arm.eval()
    folded_sd = arm.fold_into_twocomp_state_dict()

    tc_cfg = dataclasses.replace(cfg, arch="twocomp", run_name=f"{run}_folded")
    tc = build_model(tc_cfg)
    tc.load_state_dict(folded_sd, strict=True)
    tc.eval()

    corpus = Corpus(cfg.corpus, cfg.data_dir)
    folded_bpc = evaluate(tc, corpus, "test", tc_cfg, "fresh")["bpc"]
    committed = R._test_bpc(run)["fresh"]
    residual = abs(folded_bpc - committed)
    return {
        "run": run,
        "arm_fresh_bpc": committed,
        "folded_fresh_bpc": folded_bpc,
        "residual_bpc": residual,
        "gated": False,
        "note": "reported, not gated -- EXP_011 C4; fold-in tolerance is "
                "decision #6 and is open",
        "folded_param_count": sum(v.numel() for v in folded_sd.values()),
        "arm_param_count": sum(p.numel() for p in arm.parameters()),
        "folds_to_arch": "twocomp",
    }


def resolve(ctx: dict) -> dict:
    """EXP_011 §4's table, applied to the measured context."""
    bpc = ctx["bpc"]["carried"]
    m = bpc["mean"]
    sd = bpc["sd"]
    p: dict = {}

    p["C1"] = {
        "statement": f"composed carried mean <= {C1_BAR} (no negative interaction)",
        "measured": m,
        "held": m <= C1_BAR,
        "margin_bpc": C1_BAR - m,
        "note": "the hypothesis's own bar; see EXP_011 §0 item 1 for why it is "
                "not the tasking's 2.083",
    }
    p["C2"] = {
        "statement": f"cross-seed sd <= {C2_SD_MAX}",
        "measured": sd,
        "held": sd is not None and sd <= C2_SD_MAX,
    }
    p["C3"] = {
        "statement": f"composed carried mean > {C3_ADDITIVE_BPC} (gain is sub-additive)",
        "measured": m,
        "held": m > C3_ADDITIVE_BPC,
        "note": "a prediction that can fail: if it does, EXP_004 §10.6's overlap "
                "argument is wrong about the composed case",
    }
    folds = ctx.get("fold") or []
    p["C4"] = {
        "statement": "the composed arm folds into arch=twocomp; residual REPORTED",
        "measured": [f["residual_bpc"] for f in folds] or None,
        "held": None if not folds else True,
        "note": "no bar applied -- EXP_011 C4. `held` is null when the fold was "
                "skipped and true when it ran, because there is nothing to fail.",
    }
    p["C5"] = {
        "statement": "Identity 1 holds for the two-compartment neuron (fp64)",
        "measured": ctx["identity_gate"],
        "held": ctx["identity_gate"] == "passed",
        "note": "entry condition; checked by tests/test_compose_equivalence.py "
                "before training, not here",
    }

    if not p["C5"]["held"]:
        cell, verdict = "C5 fails", (
            "STOP. Identity 1 does not hold for the two-compartment neuron, so "
            "EXP_011 §1.1 is wrong and no bpc verdict is issued.")
    elif p["C1"]["held"] and p["C2"]["held"]:
        cell, verdict = "C1 holds, C2 holds", (
            "THE ARMS COMPOSE. The composed arm becomes a Phase-5 candidate. "
            "Recommendation, not adoption: adoption is Elliot's and §9 is "
            "unticked.")
    elif p["C1"]["held"]:
        cell, verdict = "C1 holds, C2 fails", (
            "Composition works but is unstable across seeds. Report; do not rank "
            "without more seeds.")
    else:
        cell, verdict = "C1 fails", (
            "NEGATIVE INTERACTION DEMONSTRATED. The composed arm is not carried "
            "forward, and EXP_004 §10.6's overlap argument is confirmed in the "
            "strong form.")
    return {"predictions": p,
            "decision": {"cell": cell, "verdict": verdict, "rule": "EXP_011 §4"}}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--horizon",
                    default="docs/reports/data/exp_011_memory_horizon.json")
    ap.add_argument("--out", default="docs/reports/data/exp_011_arm_results.json")
    ap.add_argument("--no-fold", action="store_true",
                    help="skip C4's fold (needs a GPU); the prediction is then null")
    ap.add_argument("--identity-gate", default="passed",
                    choices=("passed", "failed"),
                    help="C5's outcome, from tests/test_compose_equivalence.py")
    args = ap.parse_args(argv)

    hz = json.loads((_REPO / args.horizon).read_text(encoding="utf-8"))
    bars = hz["per_context_2sigma"]["bars"]

    # ---- references, recomputed and asserted against the reports -----------
    refs = {
        "snn_beta0.5": R.reference("snn_beta0.5", R.BASELINE_RUNS),
        "twocomp": R.reference("twocomp", R.TWOCOMP_RUNS),
        "gru": R.reference("gru", R.GRU_RUNS),
    }
    tc_mean = refs["twocomp"]["carried"]["mean"]
    base_mean = refs["snn_beta0.5"]["carried"]["mean"]

    # The threshold arm has no REPORTED entry in 010, so it is recomputed and
    # asserted here, to the same tolerance.
    thr_mean, thr_sd = _mean_sd_of(THRESHOLD_RUNS, "carried")
    if abs(thr_mean - THRESHOLD_REPORTED_CARRIED) >= R.REPORT_TOL:
        raise SystemExit(
            f"the recomputed threshold-arm carried mean {thr_mean:.6f} does not "
            f"reproduce EXP_008's reported {THRESHOLD_REPORTED_CARRIED:.6f}. "
            "Chase it rather than widening the tolerance.")
    threshold_gain_bpc = base_mean - thr_mean          # positive = arm is better

    # C3's additivity point, recomputed rather than pasted.
    additive_point = tc_mean - threshold_gain_bpc
    if abs(additive_point - C3_ADDITIVE_BPC) >= 1e-3:
        raise SystemExit(
            f"the recomputed full-additivity point {additive_point:.6f} does not "
            f"reproduce the pre-registered {C3_ADDITIVE_BPC:.6f}.")

    # ---- F1 on every composed run ------------------------------------------
    f1 = {r: hz["arms"][r].get("f1_pass") for r in COMPOSE_RUNS if r in hz["arms"]}
    if not f1 or not all(f1.values()):
        raise SystemExit(
            f"F1 self-check failed or missing for {ARM}: {f1}. The probe's k=L "
            "point must reproduce the independently committed fresh bpc -- that "
            "is the check that caught a contaminated run in Phase 3.")

    # ---- the arm ------------------------------------------------------------
    per_seed = {r: R._test_bpc(r) for r in COMPOSE_RUNS}
    bpc: dict = {}
    for protocol in ("fresh", "carried"):
        vals = [per_seed[r][protocol] for r in COMPOSE_RUNS]
        m, sd = R._mean_sd(vals)
        ref_m = refs["twocomp"][protocol]["mean"]
        sd_ref = refs["twocomp"][protocol]["sd"]
        n_ref = refs["twocomp"]["n"]
        se = math.sqrt(sd_ref ** 2 / n_ref + sd ** 2 / len(vals))
        bpc[protocol] = {
            "values": vals, "mean": m, "sd": sd, "n": len(vals),
            "reference_arm": "twocomp", "reference_mean": ref_m,
            "delta_vs_reference": m - ref_m,
            "se_of_difference": se,
            "difference_in_se": (m - ref_m) / se if se else None,
            "delta_vs_phase2_baseline": m - refs["snn_beta0.5"][protocol]["mean"],
        }

    compose_curve = R.absolute_curve(hz, ARM)
    twocomp_curve = R.absolute_curve(hz, "twocomp")
    base_curve = R.absolute_curve(hz, "snn_beta0.5")
    gru_curve = R.absolute_curve(hz, "gru")
    per_seed_hz, median_hz = R.median_horizon(hz, ARM)

    # How much of EXP_008's gain survived on top of the adopted arm.
    realised_gain = tc_mean - bpc["carried"]["mean"]
    carry_over = realised_gain / threshold_gain_bpc if threshold_gain_bpc else None

    ctx = {
        "bpc": bpc,
        "identity_gate": args.identity_gate,
        "decomposition": R.decompose(twocomp_curve, compose_curve),
        "decomposition_vs_phase2_baseline": R.decompose(base_curve, compose_curve),
        "context_vs_twocomp": R.context_comparison(compose_curve, twocomp_curve, bars),
        "context_vs_baseline": R.context_comparison(compose_curve, base_curve, bars),
        "horizon": {"per_seed": per_seed_hz, "median": median_hz},
        "learned": [R.learned_parameters(r, "thr_log") for r in COMPOSE_RUNS],
        "fold": ([] if args.no_fold
                 else [fold_check_composed(r) for r in COMPOSE_RUNS]),
        "composition": {
            "twocomp_carried_mean": tc_mean,
            "threshold_arm_carried_mean": thr_mean,
            "threshold_arm_gain_vs_baseline_bpc": threshold_gain_bpc,
            "full_additivity_point_bpc": additive_point,
            "realised_gain_over_twocomp_bpc": realised_gain,
            "carry_over_fraction": carry_over,
            "carry_over_note": (
                "a point estimate with a wide interval -- EXP_011 §3.1 states "
                "this design resolves ~0.004 bpc at 2sigma and cannot separate "
                "90% from 100% carry-over"),
        },
        "stretch_markers_no_verdict": {
            "tasking_2.083": {"bar": STRETCH_TASKING,
                              "below": bpc["carried"]["mean"] <= STRETCH_TASKING},
            "half_gain_midpoint": {"bar": STRETCH_HALF_GAIN,
                                   "below": bpc["carried"]["mean"] <= STRETCH_HALF_GAIN},
            "note": "EXP_011 §4.3: reported, no verdict attached, cannot change C1/C2",
        },
        "gap_to_gru": {
            "gru_carried_mean": refs["gru"]["carried"]["mean"],
            "gap_before_bpc": tc_mean - refs["gru"]["carried"]["mean"],
            "gap_after_bpc": bpc["carried"]["mean"] - refs["gru"]["carried"]["mean"],
        },
    }
    ctx.update(resolve(ctx))

    results = {
        "experiment": "EXP_011_composition",
        "log": "experiments/logs/EXP_011_composition.md",
        "horizon_artifact": args.horizon,
        "bars": {"C1": C1_BAR, "C2_sd_max": C2_SD_MAX,
                 "C3_additive_point": C3_ADDITIVE_BPC,
                 "whole_split_2sigma": R.TWO_SIGMA,
                 "twocomp_family_2sigma": R.TWO_SIGMA_FAMILY,
                 "baseline_horizon": R.BASELINE_HORIZON},
        "references": refs,
        "arm": ctx,
    }
    out = _REPO / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")

    # ---- report -------------------------------------------------------------
    c = ctx
    print("=" * 78)
    print("EXP_011 -- the composed arm, resolved against its pre-registration")
    print("=" * 78)
    b = c["bpc"]["carried"]
    print(f"  carried bpc   {b['mean']:.5f} +/- {b['sd']:.5f}  (n={b['n']}, "
          f"seeds {[round(v, 5) for v in b['values']]})")
    print(f"  vs twocomp    {b['delta_vs_reference']:+.5f} bpc "
          f"({b['difference_in_se']:+.1f} se of the difference)")
    print(f"  vs Phase-2    {b['delta_vs_phase2_baseline']:+.5f} bpc")
    comp = c["composition"]
    print(f"  threshold arm's own gain {comp['threshold_arm_gain_vs_baseline_bpc']:.5f} "
          f"-> realised on twocomp {comp['realised_gain_over_twocomp_bpc']:+.5f} "
          f"({100 * comp['carry_over_fraction']:.1f}% carried over)")
    d = c["decomposition"]
    print(f"  decomposition vs twocomp: zero {d['zero_context_gain_bpc']:+.4f}  "
          f"within-reach {d['within_reach_gain_bpc']:+.4f}  "
          f"beyond {d['beyond_horizon_gain_bpc']:+.4f}")
    cv = c["context_vs_twocomp"]
    print(f"  §6.2 vs twocomp: worse at {cv['n_contexts_significantly_worse']}/128 "
          f"contexts, worst short-c {cv['worst_short_context_delta_bpc']:+.4f} "
          f"at c={cv['worst_short_context_at']}")
    print(f"  median horizon {c['horizon']['median']} (per seed {c['horizon']['per_seed']})")
    for f in c["fold"]:
        print(f"  C4 fold {f['run']}: arm {f['arm_fresh_bpc']:.6f} vs folded "
              f"{f['folded_fresh_bpc']:.6f}  residual {f['residual_bpc']:.2e}  "
              f"(REPORTED, not gated)  "
              f"({f['arm_param_count']:,} -> {f['folded_param_count']:,} params)")
    for ly in c["learned"]:
        for row in ly["layers"]:
            print(f"  {ly['run']} layer {row['layer']}: {ly['reported_quantity']} "
                  f"mean {row['mean']:.4f} +/- {row['sd']:.4f}  "
                  f"p10-p90 {row['p10']:.3f}-{row['p90']:.3f}")
    sm = c["stretch_markers_no_verdict"]
    print(f"  stretch markers (NO VERDICT): below 2.083 = {sm['tasking_2.083']['below']}, "
          f"below {STRETCH_HALF_GAIN} = {sm['half_gain_midpoint']['below']}")
    print("  " + "-" * 74)
    for pid, pr in c["predictions"].items():
        state = ("HELD" if pr["held"] else "FAILED") if pr["held"] is not None \
            else "NOT EVALUABLE"
        print(f"  {pid}  {state:13s} {pr['statement']}")
    print(f"  -> {c['decision']['cell']}: {c['decision']['verdict']}")
    print(f"\nWROTE {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
