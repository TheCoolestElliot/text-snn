"""Resolve EXP_007, EXP_008 and EXP_009 against their pre-registrations, mechanically.

Every threshold this script applies was fixed in
`experiments/logs/EXP_007_token_shift.md`, `..._EXP_008_learned_threshold.md` and
`..._EXP_009_distillation_ceiling.md` **before** the arms were trained, and this
file was committed before any of their numbers were read. It reads committed
artifacts and applies the stated bars; it does not choose them. A script that
both picks the bar and reports the score is a script that can rationalise
(`scripts/exp/004_twocomp_results.py`'s docstring, followed here).

WHY ONE SCRIPT FOR THREE EXPERIMENTS
------------------------------------
They resolve against one set of statistics: whole-split bpc against a committed
reference, the horizon, the §6.2 per-context curve, and `EXP_004` §10.3's
decomposition of a gain into zero-context / within-reach / beyond-horizon. Those
numbers have to be **comparable across the three arms and with Phase 4's earlier
rows**, and `EXP_006`'s failure mode L5 names how they stop being: two
implementations of one statistic. The decision rules stay separate -- each arm is
resolved only against its own file's thresholds, and no verdict is pooled.

The references are RECOMPUTED from committed per-run artifacts and asserted
against the numbers the reports quote, never pasted. That is `EXP_001`'s F1
discipline applied to the baselines themselves: a number this script did not
produce has to come out of it.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import statistics
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

RUNS = _REPO / "experiments" / "runs"

# ===========================================================================
# Everything below this line was fixed in the three pre-registrations.
# ===========================================================================

BASELINE_ARM = "snn_beta0.5"
BASELINE_RUNS = [f"snn_beta0.5_s{i}" for i in range(5)]
GRU_RUNS = [f"gru_s{i}" for i in range(3)]
TWOCOMP_RUNS = [f"twocomp_s{i}" for i in range(7)]

#: the reported means these recomputations must reproduce, and the tolerance
REPORTED = {
    "snn_beta0.5": {"fresh": 2.2697, "carried": 2.2531},     # phase-2 report
    "twocomp": {"fresh": 2.14566, "carried": 2.11869},       # EXP_005 §9, n=7
    "gru": {"fresh": 1.80443, "carried": 1.76741},           # phase-2 report
}
REPORT_TOL = 5e-4

TWO_SIGMA = 0.00922              # EXP_000 whole-split noise floor, x2
TWO_SIGMA_FAMILY = 0.00627       # EXP_005: the two-compartment family's own bar
BASELINE_HORIZON = 7             # EXP_001, five seeds agreeing exactly
CMAX = "127"                     # the absolute curve's asymptote
FOLD_TOLERANCE_BPC = 1e-3        # EXP_008 W1, = EXP_001's F1 tolerance

#: EXP_007 §3
V1_BAR = TWO_SIGMA
V3_HORIZON_MAX = 14
V4_WITHIN_CEILING = 0.0581       # half the baseline-vs-GRU within-reach 0.1161
V6_MU_MOVEMENT = 0.05

#: EXP_008 §3
W2_BAR = TWO_SIGMA
W3_ZERO_SHARE = 0.5
W4_GAIN_MAX = 0.95
W5_ZERO_CEILING = 0.0506         # EXP_005 n=7: the twocomp's own zero-context gain
W6_HORIZON_MAX = 8

#: EXP_009 §3
X1_BAR = TWO_SIGMA_FAMILY
X2_GAP_SHARE = 0.25
X3_HORIZON_MIN = 38
X5_FLOOR_BPC = 1.85

ARMS = {
    "tokenshift": {
        "experiment": "EXP_007_token_shift",
        "log": "experiments/logs/EXP_007_token_shift.md",
        "runs": [f"tokenshift_s{i}" for i in range(3)],
        "reference_arm": BASELINE_ARM,
        "labelled": "DIAGNOSTIC -- I5 status not ruled; see EXP_007 header",
    },
    "threshold": {
        "experiment": "EXP_008_learned_threshold",
        "log": "experiments/logs/EXP_008_learned_threshold.md",
        "runs": [f"threshold_s{i}" for i in range(3)],
        "reference_arm": BASELINE_ARM,
        "labelled": "candidate arm (EXP_004 §10.11 item 2)",
    },
    "twocomp_distill": {
        "experiment": "EXP_009_distillation_ceiling",
        "log": "experiments/logs/EXP_009_distillation_ceiling.md",
        "runs": [f"twocomp_distill_s{i}" for i in range(3)],
        "reference_arm": "twocomp",
        "labelled": "DIAGNOSTIC -- never the headline; EXP_009 §4's rider",
        # ---------------------------------------------------------------
        # POST-HOC, and flagged as such everywhere it has an effect.
        #
        # `twocomp_distill_s1` diverged at step 12 497 of 20 000 and scores NaN.
        # EXP_009 fixed n = 3 and did not anticipate a divergence, so there is no
        # pre-registered rule for this and one is being made after seeing the
        # outcome. The rule chosen is the one that keeps the failure visible:
        #
        #   * the seed is excluded from every mean, because a NaN is not a score;
        #   * it is NOT replaced by a fourth seed. EXP_009 §2 pairs student seed i
        #     with teacher `gru_s{i}` and there is no `gru_s3`, so a replacement
        #     would have to reuse a teacher and break the design that keeps the
        #     seeds independent -- and swapping a diverged seed for a fresh one is
        #     how a failure disappears from a record;
        #   * every X-prediction it touches is reported at **n = 2** with the
        #     pre-registered n = 3 unmet, so the verdicts are PROVISIONAL and the
        #     JSON says so in `provisional_reason`.
        #
        # `scripts/exp/009_chase_divergence.py` reproduces the divergence
        # deterministically and localises it; it is a finding of this experiment,
        # not an accident to be routed around.
        "excluded_runs": {
            "twocomp_distill_s1": "diverged at step 12497/20000; scores NaN. "
                                  "Reproduced and localised in "
                                  "docs/reports/data/exp_009_divergence.json",
        },
    },
}


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _mean_sd(xs: list[float]) -> tuple[float, float]:
    n = len(xs)
    m = sum(xs) / n
    sd = statistics.stdev(xs) if n > 1 else 0.0
    return m, sd


def _test_bpc(run: str) -> dict[str, float]:
    path = RUNS / run / "final_test.json"
    if not path.exists():
        raise SystemExit(f"{path} missing -- run scripts/evaluate.py on {run}")
    blob = json.loads(path.read_text(encoding="utf-8"))
    res = blob.get("results", blob)
    return {p: res[p]["bpc"] for p in ("fresh", "carried")}


def reference(name: str, runs: list[str]) -> dict:
    """Recompute a committed reference and assert it against what the reports say."""
    out: dict = {"runs": runs, "n": len(runs)}
    for protocol in ("fresh", "carried"):
        vals = [_test_bpc(r)[protocol] for r in runs]
        m, sd = _mean_sd(vals)
        out[protocol] = {"values": vals, "mean": m, "sd": sd}
        residual = m - REPORTED[name][protocol]
        out[protocol]["reported"] = REPORTED[name][protocol]
        out[protocol]["residual_vs_reported"] = residual
        if abs(residual) >= REPORT_TOL:
            raise SystemExit(
                f"the recomputed {name} {protocol} mean {m:.6f} does not reproduce "
                f"the reported {REPORTED[name][protocol]:.6f} (residual "
                f"{residual:+.2e}). Something has moved; chase it rather than "
                "widening the tolerance.")
    return out


def decompose(ref_curve: dict, arm_curve: dict, horizon: int = BASELINE_HORIZON) -> dict:
    """`EXP_004` §10.3's decomposition of a gain, on the absolute context curve.

    Positive = the arm is better. The cut is at the BASELINE's horizon of 7 for
    every arm, including arms whose own horizon differs, because the three
    components have to stay comparable with §10.3's and with `EXP_005` §9.6's.
    `EXP_007` §5 item 4 records that choice as a limitation rather than a
    convenience.
    """
    h = str(horizon)
    total = ref_curve[CMAX] - arm_curve[CMAX]
    zero = ref_curve["0"] - arm_curve["0"]
    at_h = ref_curve[h] - arm_curve[h]
    within = at_h - zero
    beyond = total - at_h
    share = (lambda x: x / total if total else float("nan"))
    return {
        "cut_at_context": horizon,
        "total_gain_bpc": total,
        "zero_context_gain_bpc": zero,
        "within_reach_gain_bpc": within,
        "beyond_horizon_gain_bpc": beyond,
        "zero_context_share": share(zero),
        "within_reach_share": share(within),
        "beyond_horizon_share": share(beyond),
    }


def context_comparison(arm_curve: dict, ref_curve: dict, bars: dict) -> dict:
    """§6.2, against an arbitrary reference rather than only against the baseline.

    The probe already publishes this against the Phase-2 baseline; the distilled
    arm also needs it against the un-distilled two-compartment arm, and computing
    it twice in two places is how two numbers that should agree stop agreeing.
    The bars are the baseline's per-context 2σ, which is what every other §6.2
    table in this project uses.
    """
    deltas = {c: arm_curve[c] - ref_curve[c] for c in arm_curve}
    worse = [int(c) for c in deltas if deltas[c] > bars.get(c, TWO_SIGMA)]
    short = [c for c in deltas if int(c) <= 8]
    worst_short = max(short, key=lambda c: deltas[c])
    worst = max(deltas, key=lambda c: deltas[c])
    return {
        "n_contexts_significantly_worse": len(worse),
        "contexts_significantly_worse": sorted(worse),
        "n_short_contexts_significantly_worse": len([c for c in worse if c <= 8]),
        "worst_delta_bpc": deltas[worst],
        "worst_delta_at_context": int(worst),
        "worst_short_context_delta_bpc": deltas[worst_short],
        "worst_short_context_at": int(worst_short),
        "short_context_table": [
            {"c": int(c), "delta_bpc": deltas[c], "bar_2sigma": bars.get(c, TWO_SIGMA),
             "multiple_of_bar": deltas[c] / bars[c] if bars.get(c) else None}
            for c in sorted(short, key=int)
        ],
    }


def absolute_curve(horizon_blob: dict, arm: str) -> dict[str, float]:
    return horizon_blob["absolute_context_curves"][arm]["bpc_at_context"]


def median_horizon(horizon_blob: dict, arm: str) -> tuple[list, float | None]:
    per_seed = horizon_blob["by_arm"][arm]["_paired"]["horizon_2sigma_per_seed"]
    # `EXP_006` §8.1: an unresolved horizon means "> 127", the LONGEST outcome,
    # and dropping it would bias the median downward. Encoded, not dropped.
    vals = [128 if h is None else h for h in per_seed]
    return per_seed, (statistics.median(vals) if vals else None)


def learned_parameters(run: str, key: str) -> dict:
    """The new parameter's realised value per layer, read from the checkpoint.

    Reported as the quantity the equations contain -- `mu` itself for the shift,
    and `exp(theta)` for the threshold -- rather than as the stored raw, which is
    a number no equation in the project mentions (`004_twocomp_results.py`'s rule).
    """
    ck = torch.load(RUNS / run / "ckpt_final.pt", map_location="cpu",
                    weights_only=False)
    sd = ck["model"]
    layers = []
    k = 0
    while f"{key}.{k}" in sd:
        raw = sd[f"{key}.{k}"].flatten().float()
        realised = raw if key == "mu" else torch.exp(raw)
        q = torch.quantile(realised, torch.tensor([0.1, 0.5, 0.9]))
        layers.append({
            "layer": k,
            "mean": float(realised.mean()),
            "sd": float(realised.std()),
            "min": float(realised.min()),
            "max": float(realised.max()),
            "p10": float(q[0]), "p50": float(q[1]), "p90": float(q[2]),
            "abs_movement_from_one": float((realised - 1.0).abs().mean()),
        })
        k += 1
    return {"run": run, "parameter": key, "step": ck.get("step"), "layers": layers}


# ---------------------------------------------------------------------------
# EXP_008 W1 -- the fold, executed
# ---------------------------------------------------------------------------

def fold_check(run: str) -> dict:
    """Fold `exp(-theta)` into each layer's Linear, load as `arch="snn"`, score.

    The folded model is built through `build_model` and scored by
    `snn.evaluate.evaluate` -- the same code path that produced every committed
    baseline number, and a different one from the arm's own forward. That is what
    makes this a check rather than a tautology (EXP_008's P3).
    """
    ck = torch.load(RUNS / run / "ckpt_final.pt", map_location="cpu",
                    weights_only=False)
    known = {f.name for f in dataclasses.fields(Config)}
    cfg = Config(**{k: v for k, v in ck["config"].items() if k in known})

    arm = build_model(cfg)
    arm.load_state_dict(ck["model"])
    arm.eval()
    folded_sd = arm.fold_into_spiking_state_dict()

    base_cfg = dataclasses.replace(cfg, arch="snn", run_name=f"{run}_folded")
    base = build_model(base_cfg)
    base.load_state_dict(folded_sd, strict=True)
    base.eval()

    corpus = Corpus(cfg.corpus, cfg.data_dir)
    folded_bpc = evaluate(base, corpus, "test", base_cfg, "fresh")["bpc"]
    committed = _test_bpc(run)["fresh"]
    residual = abs(folded_bpc - committed)
    n_params = sum(v.numel() for v in folded_sd.values())
    return {
        "run": run,
        "arm_fresh_bpc": committed,
        "folded_fresh_bpc": folded_bpc,
        "residual_bpc": residual,
        "pass": residual < FOLD_TOLERANCE_BPC,
        "folded_param_count": n_params,
        "arm_param_count": sum(p.numel() for p in arm.parameters()),
    }


# ---------------------------------------------------------------------------
# the three decision rules
# ---------------------------------------------------------------------------

def resolve_tokenshift(ctx: dict) -> dict:
    d, hz = ctx["decomposition"], ctx["horizon"]
    p = {}
    p["V1"] = {"statement": f"mean carried improves by more than 2sigma={V1_BAR}",
               "measured_delta": ctx["bpc"]["carried"]["delta_vs_reference"],
               "held": ctx["bpc"]["carried"]["delta_vs_reference"] < -V1_BAR}
    p["V2"] = {"statement": "within-reach gain > beyond-horizon gain",
               "measured": [d["within_reach_gain_bpc"], d["beyond_horizon_gain_bpc"]],
               "held": d["within_reach_gain_bpc"] > d["beyond_horizon_gain_bpc"],
               "note": "flatters the plan this experiment exists to serve"}
    p["V3"] = {"statement": f"median horizon <= {V3_HORIZON_MAX}",
               "measured": hz["median"],
               "held": hz["median"] is not None and hz["median"] <= V3_HORIZON_MAX}
    p["V4"] = {"statement": f"within-reach gain < {V4_WITHIN_CEILING} bpc",
               "measured": d["within_reach_gain_bpc"],
               "held": d["within_reach_gain_bpc"] < V4_WITHIN_CEILING,
               "note": "stated against the plan"}
    p["V5"] = {"statement": "no context c<=8 worse than its own 2sigma bar",
               "measured": ctx["context_vs_reference"]
                              ["n_short_contexts_significantly_worse"],
               "held": ctx["context_vs_reference"]
                          ["n_short_contexts_significantly_worse"] == 0}
    moves = [ly["abs_movement_from_one"]
             for r in ctx["learned"] for ly in r["layers"]]
    p["V6"] = {"statement": f"mean |1-mu| >= {V6_MU_MOVEMENT} in both layers",
               "measured": moves, "held": bool(moves) and all(m >= V6_MU_MOVEMENT
                                                              for m in moves)}
    if not p["V1"]["held"]:
        cell = "V1 fails"
        verdict = ("NULL, retained. A two-tap FIR does not beat the baseline at "
                   "3 seeds; the within-reach component stays UNATTACKED rather "
                   "than attacked-and-hard.")
    elif p["V2"]["held"]:
        cell = "V1 holds, V2 holds"
        verdict = ("A two-tap input mix buys bits inside the reach. The "
                   "within-reach component is attackable by a mechanism this "
                   "project has not been ranking. I5 ruling stays Elliot's.")
    else:
        cell = "V1 holds, V2 fails"
        verdict = ("It buys bits, but not the ones this experiment aimed at. "
                   "The within-reach claim is not made.")
    return {"predictions": p, "decision": {"cell": cell, "verdict": verdict,
                                           "rule": "EXP_007 §4"}}


def resolve_threshold(ctx: dict) -> dict:
    d, hz = ctx["decomposition"], ctx["horizon"]
    folds = ctx["fold"]
    p = {}
    p["W1"] = {"statement": f"folded model reproduces the arm to < {FOLD_TOLERANCE_BPC} bpc",
               "measured": [f["residual_bpc"] for f in folds],
               "held": all(f["pass"] for f in folds),
               "note": "load-bearing: this is Identity 2, measured"}
    p["W2"] = {"statement": f"mean carried improves by more than 2sigma={W2_BAR}",
               "measured_delta": ctx["bpc"]["carried"]["delta_vs_reference"],
               "held": ctx["bpc"]["carried"]["delta_vs_reference"] < -W2_BAR}
    if p["W2"]["held"]:
        share = d["zero_context_share"]
        p["W3"] = {"statement": f"zero-context share of the gain >= {W3_ZERO_SHARE}",
                   "measured": share, "held": share >= W3_ZERO_SHARE}
    else:
        p["W3"] = {"statement": f"zero-context share of the gain >= {W3_ZERO_SHARE}",
                   "measured": None, "held": None,
                   "note": "NOT EVALUABLE -- EXP_008 §3: a share of a gain is "
                           "meaningless without a gain. Components reported in bpc."}
    gains = [ly["mean"] for r in ctx["learned"] for ly in r["layers"]]
    p["W4"] = {"statement": f"mean exp(theta) < {W4_GAIN_MAX} in both layers",
               "measured": gains,
               "held": bool(gains) and all(g < W4_GAIN_MAX for g in gains)}
    p["W5"] = {"statement": f"zero-context gain <= {W5_ZERO_CEILING} bpc",
               "measured": d["zero_context_gain_bpc"],
               "held": d["zero_context_gain_bpc"] <= W5_ZERO_CEILING,
               "note": "stated against the plan"}
    p["W6"] = {"statement": f"median horizon <= {W6_HORIZON_MAX}",
               "measured": hz["median"],
               "held": hz["median"] is not None and hz["median"] <= W6_HORIZON_MAX,
               "note": "stated against the plan"}
    if not p["W1"]["held"]:
        cell = "W1 fails"
        verdict = ("STOP AND CHASE IT. A failed fold-in means EXP_008 §1.1's "
                   "algebra or its implementation is wrong. No bpc verdict is "
                   "issued and the tolerance is not widened.")
    elif p["W2"]["held"]:
        cell = "W1 holds, W2 holds"
        verdict = ("A free win, and a correction to §10.6: the zero-context "
                   "component is reachable by the baseline's own parameterisation "
                   "and is an OPTIMISATION effect, not a capacity one. Recommend "
                   "adoption AND folding away at inference -- zero parameters, "
                   "zero kernels. Recommendation, not adoption: §9 is unticked.")
    else:
        cell = "W1 holds, W2 fails"
        verdict = ("Null, retained, and still informative. The arm cannot LACK "
                   "the capacity (Identity 2), so the two-compartment arm's "
                   "zero-context gain localises in the interaction with the slow "
                   "pole rather than in the threshold alone -- a correction to "
                   "§10.6 in the other direction.")
    return {"predictions": p, "decision": {"cell": cell, "verdict": verdict,
                                           "rule": "EXP_008 §4"}}


def resolve_distill(ctx: dict) -> dict:
    d, hz = ctx["decomposition"], ctx["horizon"]
    p = {}
    p["X1"] = {"statement": f"mean carried improves by more than 2sigma_family={X1_BAR}",
               "measured_delta": ctx["bpc"]["carried"]["delta_vs_reference"],
               "held": ctx["bpc"]["carried"]["delta_vs_reference"] < -X1_BAR}
    closed = ctx["gap_closed"]
    p["X2"] = {"statement": f"closes >= {X2_GAP_SHARE:.0%} of the asymptotic gap",
               "measured": closed["asymptotic_share_closed"],
               "held": closed["asymptotic_share_closed"] >= X2_GAP_SHARE}
    p["X3"] = {"statement": f"median horizon >= {X3_HORIZON_MIN}",
               "measured": hz["median"],
               "held": hz["median"] is not None and hz["median"] >= X3_HORIZON_MIN}
    p["X4"] = {"statement": "within-reach gain > beyond-horizon gain (vs twocomp)",
               "measured": [d["within_reach_gain_bpc"], d["beyond_horizon_gain_bpc"]],
               "held": d["within_reach_gain_bpc"] > d["beyond_horizon_gain_bpc"],
               "note": "stated against the plan"}
    p["X5"] = {"statement": f"mean carried >= {X5_FLOOR_BPC} (does NOT reach the anchor)",
               "measured": ctx["bpc"]["carried"]["mean"],
               "held": ctx["bpc"]["carried"]["mean"] >= X5_FLOOR_BPC,
               "note": "stated against the plan; if this fails it is the headline"}
    if not p["X1"]["held"]:
        cell = "X1 fails"
        verdict = ("THIS RECIPE DID NOT HELP. Recorded as a null OF THE RECIPE, "
                   "NOT OF THE HYPOTHESIS (EXP_009 §1.1). It is explicitly not "
                   "evidence that the remaining gap is representational.")
    elif p["X2"]["held"]:
        cell = "X1 holds, X2 holds"
        verdict = ("At least a quarter of the remaining gap is OPTIMISATIONAL, "
                   "demonstrated constructively. Phase 5 must carry a "
                   "training-signal arm. The number is a LOWER BOUND.")
    else:
        cell = "X1 holds, X2 fails"
        verdict = ("Some of the gap is optimisational; less than a quarter of it "
                   "by this route. The architecture ranking stands.")
    return {"predictions": p, "decision": {"cell": cell, "verdict": verdict,
                                           "rule": "EXP_009 §4"}}


RESOLVERS = {"tokenshift": resolve_tokenshift,
             "threshold": resolve_threshold,
             "twocomp_distill": resolve_distill}
PARAM_KEY = {"tokenshift": "mu", "threshold": "thr_log"}


# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--horizon",
                    default="docs/reports/data/exp_007_009_memory_horizon.json")
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--out", default="docs/reports/data/exp_007_009_arm_results.json")
    ap.add_argument("--no-fold", action="store_true",
                    help="skip EXP_008's W1 (needs a GPU); the prediction is then null")
    args = ap.parse_args(argv)
    wanted = [a.strip() for a in args.arms.split(",") if a.strip()]

    hz = json.loads((_REPO / args.horizon).read_text(encoding="utf-8"))
    bars = hz["per_context_2sigma"]["bars"]

    refs = {
        BASELINE_ARM: reference("snn_beta0.5", BASELINE_RUNS),
        "twocomp": reference("twocomp", TWOCOMP_RUNS),
        "gru": reference("gru", GRU_RUNS),
    }
    gru_curve = absolute_curve(hz, "gru")
    twocomp_curve = absolute_curve(hz, "twocomp")

    results: dict = {
        "experiments": [ARMS[a]["experiment"] for a in wanted],
        "horizon_artifact": args.horizon,
        "bars": {"whole_split_2sigma": TWO_SIGMA,
                 "twocomp_family_2sigma": TWO_SIGMA_FAMILY,
                 "baseline_horizon": BASELINE_HORIZON},
        "references": refs,
        "arms": {},
    }

    for arm in wanted:
        spec = ARMS[arm]
        excluded = spec.get("excluded_runs", {})
        runs = [r for r in spec["runs"] if r not in excluded]
        ref_arm = spec["reference_arm"]
        if excluded:
            print(f"  NOTE: {arm} excludes {sorted(excluded)} -- see the ARMS "
                  f"registry for the post-hoc rule and why it was chosen")

        # ---- F1, on every run of the arm -----------------------------------
        f1 = {r: hz["arms"][r].get("f1_pass") for r in runs if r in hz["arms"]}
        if not f1 or not all(f1.values()):
            raise SystemExit(
                f"F1 self-check failed or missing for {arm}: {f1}. The probe's "
                "k=L point must reproduce the independently committed fresh bpc "
                "-- that is the check that caught a contaminated run in Phase 3.")

        per_seed = {r: _test_bpc(r) for r in runs}
        bpc: dict = {}
        for protocol in ("fresh", "carried"):
            vals = [per_seed[r][protocol] for r in runs]
            m, sd = _mean_sd(vals)
            ref_m = refs[ref_arm][protocol]["mean"]
            n_ref = refs[ref_arm]["n"]
            sd_ref = refs[ref_arm][protocol]["sd"]
            se = math.sqrt(sd_ref ** 2 / n_ref + sd ** 2 / len(vals))
            bpc[protocol] = {
                "values": vals, "mean": m, "sd": sd, "n": len(vals),
                "reference_arm": ref_arm, "reference_mean": ref_m,
                "delta_vs_reference": m - ref_m,
                "se_of_difference": se,
                "difference_in_se": (m - ref_m) / se if se else None,
                "sigma_units_at_inherited_bar": (m - ref_m) / (TWO_SIGMA / 2.0),
            }

        curve = absolute_curve(hz, arm)
        ref_curve = absolute_curve(hz, ref_arm)
        per_seed_h, med_h = median_horizon(hz, arm)

        ctx = {
            "runs": runs,
            "labelled": spec["labelled"],
            "log": spec["log"],
            "excluded_runs": excluded,
            "n_preregistered": len(spec["runs"]),
            "n_used": len(runs),
            "provisional_reason": (
                f"pre-registered n={len(spec['runs'])} not met: "
                f"{sorted(excluded)} excluded ({'; '.join(excluded.values())}). "
                "Every verdict below is PROVISIONAL." if excluded else None),
            "f1_pass": f1,
            "bpc": bpc,
            "horizon": {"per_seed": per_seed_h, "median": med_h,
                        "reference_arm_median": median_horizon(hz, ref_arm)[1]},
            "decomposition": decompose(ref_curve, curve),
            "decomposition_vs_baseline": decompose(
                absolute_curve(hz, BASELINE_ARM), curve),
            "remaining_gap_vs_gru": decompose(curve, gru_curve),
            "context_vs_reference": context_comparison(curve, ref_curve, bars),
            "context_vs_baseline": context_comparison(
                curve, absolute_curve(hz, BASELINE_ARM), bars),
            "gap_closed": {
                "reference_asymptote": ref_curve[CMAX],
                "arm_asymptote": curve[CMAX],
                "gru_asymptote": gru_curve[CMAX],
                "asymptotic_gap_before": ref_curve[CMAX] - gru_curve[CMAX],
                "asymptotic_gap_after": curve[CMAX] - gru_curve[CMAX],
                "asymptotic_share_closed": (
                    (ref_curve[CMAX] - curve[CMAX])
                    / (ref_curve[CMAX] - gru_curve[CMAX])
                    if ref_curve[CMAX] != gru_curve[CMAX] else float("nan")),
                "whole_split_carried_gap_before":
                    refs[ref_arm]["carried"]["mean"] - refs["gru"]["carried"]["mean"],
                "whole_split_carried_gap_after":
                    bpc["carried"]["mean"] - refs["gru"]["carried"]["mean"],
            },
            "learned": ([learned_parameters(r, PARAM_KEY[arm]) for r in runs]
                        if arm in PARAM_KEY else []),
        }
        if arm == "threshold":
            ctx["fold"] = ([] if args.no_fold else [fold_check(r) for r in runs])
            if args.no_fold:
                ctx["fold_skipped"] = True

        ctx.update(RESOLVERS[arm](ctx))
        results["arms"][arm] = ctx

    out = _REPO / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")

    # ---- report ------------------------------------------------------------
    w = 78
    print("=" * w)
    print("PHASE-4 ARM RESULTS -- EXP_007, EXP_008, EXP_009, resolved as written")
    print("=" * w)
    for name, r in refs.items():
        print(f"  reference {name:14s} n={r['n']}  fresh {r['fresh']['mean']:.5f}  "
              f"carried {r['carried']['mean']:.5f}  (reproduces the report)")
    for arm, c in results["arms"].items():
        print("\n" + "-" * w)
        print(f"{arm}   [{c['labelled']}]")
        print("-" * w)
        if c.get("provisional_reason"):
            print(f"  !! {c['provisional_reason']}")
        for p in ("fresh", "carried"):
            b = c["bpc"][p]
            print(f"  test bpc {p:8s} {b['mean']:.5f} +/- {b['sd']:.5f}  "
                  f"vs {b['reference_arm']} {b['reference_mean']:.5f}  "
                  f"delta {b['delta_vs_reference']:+.5f}  "
                  f"({b['difference_in_se']:+.1f} se)")
        d = c["decomposition"]
        print(f"  gain decomposed (cut at c={d['cut_at_context']}): "
              f"zero {d['zero_context_gain_bpc']:+.4f} "
              f"({100*d['zero_context_share']:.1f}%)  "
              f"within {d['within_reach_gain_bpc']:+.4f} "
              f"({100*d['within_reach_share']:.1f}%)  "
              f"beyond {d['beyond_horizon_gain_bpc']:+.4f} "
              f"({100*d['beyond_horizon_share']:.1f}%)")
        print(f"  horizon per seed {c['horizon']['per_seed']}  "
              f"median {c['horizon']['median']}")
        cv = c["context_vs_reference"]
        print(f"  §6.2 vs {c['bpc']['carried']['reference_arm']}: worse at "
              f"{cv['n_contexts_significantly_worse']}/128 contexts, "
              f"{cv['n_short_contexts_significantly_worse']} of them at c<=8; "
              f"worst short-c {cv['worst_short_context_delta_bpc']:+.4f} "
              f"at c={cv['worst_short_context_at']}")
        g = c["gap_closed"]
        print(f"  asymptotic gap to the GRU: {g['asymptotic_gap_before']:.4f} -> "
              f"{g['asymptotic_gap_after']:.4f}  "
              f"({100*g['asymptotic_share_closed']:.1f}% closed)")
        if c.get("fold"):
            for f in c["fold"]:
                print(f"  W1 fold {f['run']}: arm {f['arm_fresh_bpc']:.6f} vs "
                      f"folded {f['folded_fresh_bpc']:.6f}  residual "
                      f"{f['residual_bpc']:.2e}  {'OK' if f['pass'] else 'FAILED'}"
                      f"   ({f['arm_param_count']:,} -> {f['folded_param_count']:,} params)")
        for ly in c.get("learned", []):
            for row in ly["layers"]:
                print(f"  {ly['run']} layer {row['layer']}: {ly['parameter']} "
                      f"mean {row['mean']:.4f} +/- {row['sd']:.4f}  "
                      f"p10-p90 {row['p10']:.3f}-{row['p90']:.3f}")
        print("  " + "-" * (w - 2))
        for pid, p in c["predictions"].items():
            state = ("HELD" if p["held"] else "FAILED") if p["held"] is not None \
                else "NOT EVALUABLE"
            print(f"  {pid}  {state:13s} {p['statement']}")
        print(f"  -> {c['decision']['cell']}: {c['decision']['verdict']}")

    print(f"\nWROTE {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
