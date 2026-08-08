"""EXP_014 resolver -- S1-S4 and D1, against the bars as pre-registered.

    python scripts/exp/014_scaling_results.py --print-bars
    python scripts/exp/014_scaling_results.py --out docs/reports/data/exp_014_scaling_results.json

Written and committed BEFORE any rung's bpc was read, which is the standard
`010_phase4_arm_results.py` set and `013_noise_results.py` followed.

WHAT IT MAY NOT DO
------------------
It may not adopt anything, may not rank anything, and may not widen a bar that
fails.  `EXP_014` is a pilot at n = 1 per rung; §6.2's acceptance rule needs
three seeds, so no result here can clear it however large.  The artifact carries
`adopts_nothing` / `ranks_nothing` structurally rather than in prose.

Every threshold below is TRANSCRIBED from `EXP_014_parameter_scaling.md` §3 and
recomputed nowhere.  `--print-bars` dumps them for eyeball comparison against
that file, which is the cheapest way to catch a transcription error before a
verdict rests on it.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
DATA = _REPO / "docs" / "reports" / "data"
RUNS = _REPO / "experiments" / "runs"

# --- bars, transcribed from the pre-registration -------------------------
#: §3.0.  Measured at d = 512, arch="snn", n = 5.  TRANSFERRED to every other
#: rung, which is why S1's bar is at 10x it rather than at 2x.
SIGMA = 0.00461
TWO_SIGMA = 0.00922
TEN_SIGMA = 0.0461

#: §3, S1.  The committed FIVE-SEED mean, not seed 0's own 2.2554769668134713.
ANCHOR_CARRIED = 2.25311
ANCHOR_FRESH = 2.26969

#: §3, S3.  `EXP_013` §2.2's cross-seed sd of the gap, at n = 5, transferred.
SIGMA_GAP_FRESH = 0.00268
TWO_SIGMA_GAP_FRESH = 0.00536
BASELINE_GAP_FRESH = 0.00594

#: §6.5's reparameterisation noise floor.  A quantity below it is reported as
#: "below what this instrument resolves", never as a signed number.
NOISE_FLOOR_BPC = 2e-3

#: §3, S4.  The cut is at the BASELINE's horizon, for every rung, so the three
#: components stay comparable with `EXP_004` §10.3 and `EXP_005` §9.6.
BASELINE_HORIZON = 7
HORIZON_CEILING = 8

#: §2.2.
LADDER = ((512, 735_437), (1020, 2_501_245), (1481, 4_997_099))

BARS = {
    "sigma_transferred_from_d512_n5": SIGMA,
    "S1_threshold_bpc_carried": ANCHOR_CARRIED - TEN_SIGMA,
    "S1_anchor_carried_5seed_mean": ANCHOR_CARRIED,
    "S1_margin_required_bpc": TEN_SIGMA,
    "S2_pair_resolution_gate_bpc": TWO_SIGMA,
    "S3_gap_baseline_fresh": BASELINE_GAP_FRESH,
    "S3_margin_required_bpc": TWO_SIGMA_GAP_FRESH,
    "S3_requires_delta_test_le_zero": True,
    "S4_cut_at_context": BASELINE_HORIZON,
    "S4_horizon_ceiling": HORIZON_CEILING,
    "S4_share_scored_only_above_bpc": TEN_SIGMA,
    "noise_floor_bpc": NOISE_FLOOR_BPC,
}


def _load(name: str) -> dict | None:
    p = DATA / name
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def _run_name(d: int) -> str:
    return f"scale_d{d}_s0"


def _arm_name(d: int) -> str:
    return f"scale_d{d}"


def _final_test(run: str) -> dict | None:
    p = RUNS / run / "final_test.json"
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))["results"]


def _clip_engagement(run: str) -> dict:
    """D1's clip leg.  `grad_clip = 1.0`; the committed baseline never reaches it."""
    p = RUNS / run / "log.jsonl"
    if not p.exists():
        return {"available": False}
    norms = []
    nonfinite = 0
    for line in p.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("event") != "train":
            continue
        g = r.get("grad_norm")
        if g is None:
            continue
        norms.append(g)
        loss = r.get("loss")
        if loss is not None and (loss != loss or abs(loss) == float("inf")):
            nonfinite += 1
    if not norms:
        return {"available": False}
    over = [g for g in norms if g > 1.0]
    return {
        "available": True,
        "n_logged_steps": len(norms),
        "n_over_clip": len(over),
        "frac_over_clip": round(len(over) / len(norms), 4),
        "max_grad_norm": max(norms),
        "n_nonfinite_loss_records": nonfinite,
    }


def resolve() -> dict:
    manifest = _load("exp_014_run_manifest.json")
    horizon = _load("exp_014_memory_horizon.json")
    gap = _load("exp_014_gap_scaling.json")

    out: dict = {
        "experiment": "EXP_014_parameter_scaling",
        "log": "experiments/logs/EXP_014_parameter_scaling.md",
        "bars": BARS,
        "adopts_nothing": True,
        "ranks_nothing": True,
        "n_per_rung": 1,
        "sigma_is_transferred": True,
        "rungs": [],
    }

    # ---- the scoreboard --------------------------------------------------
    per_d: dict[int, dict] = {}
    for d, params in LADDER:
        run = _run_name(d)
        ft = _final_test(run)
        row: dict = {"d_model": d, "params": params, "run": run}
        if ft is None:
            row["missing"] = "final_test.json"
        else:
            row["test_bpc_carried"] = ft["carried"]["bpc"]
            row["test_bpc_fresh"] = ft["fresh"]["bpc"]
            row["firing_rate_carried"] = ft["carried"]["firing_rate"]
        row["d1_clip"] = _clip_engagement(run)
        per_d[d] = row
        out["rungs"].append(row)

    top = per_d.get(1481, {})
    have_top = "test_bpc_carried" in top

    # ---- S1 --------------------------------------------------------------
    s1: dict = {
        "statement": ("test bpc carried at d=1481 is at least 10 sigma below the "
                      "committed five-seed anchor"),
        "anchor_carried": ANCHOR_CARRIED,
        "threshold": ANCHOR_CARRIED - TEN_SIGMA,
    }
    if not have_top:
        s1["verdict"] = "NOT RUN"
    else:
        delta = top["test_bpc_carried"] - ANCHOR_CARRIED
        s1["measured_bpc_carried"] = top["test_bpc_carried"]
        s1["delta_vs_anchor_bpc"] = delta
        s1["delta_in_transferred_sigma"] = round(delta / SIGMA, 2)
        if delta <= -TEN_SIGMA:
            s1["verdict"] = "HELD"
        elif delta >= TEN_SIGMA:
            # §4 row 4: a finding, not a null.
            s1["verdict"] = "HELD IN THE HARM DIRECTION"
        else:
            # §3.0: at n = 1 a miss is UNRESOLVED, never FAILED.  This design can
            # show an effect exceeds a threshold; it can never show one is absent.
            s1["verdict"] = "UNRESOLVED"
            s1["wording_fixed_in_advance"] = (
                "capacity was not shown to move the number at this recipe and "
                "this n -- NOT 'capacity does not help'")
    out["S1"] = s1

    # ---- S2 --------------------------------------------------------------
    pairs = []
    ds = [d for d, _ in LADDER]
    for lo, hi in zip(ds, ds[1:]):
        a, b = per_d.get(lo, {}), per_d.get(hi, {})
        if "test_bpc_carried" not in a or "test_bpc_carried" not in b:
            pairs.append({"from": lo, "to": hi, "verdict": "NOT RUN"})
            continue
        delta = b["test_bpc_carried"] - a["test_bpc_carried"]
        row = {"from": lo, "to": hi, "delta_bpc": delta,
               "resolves": abs(delta) >= TWO_SIGMA}
        if not row["resolves"]:
            row["verdict"] = "TIED AT THIS INSTRUMENT'S RESOLUTION"
        else:
            row["verdict"] = "NON-INCREASING" if delta <= 0 else "INCREASING"
        pairs.append(row)
    scored = [p for p in pairs if p.get("resolves")]
    out["S2"] = {
        "statement": ("bpc carried non-increasing across the ladder, scored only "
                      "on adjacent pairs resolving at 2 sigma"),
        "octaves": [1.766, 0.998],
        "steps_are_unequal": True,
        "note": ("the rungs are 1.766 then 0.998 octaves apart, so no per-octave "
                 "slope may be quoted from the pair as if the steps matched"),
        "pairs": pairs,
        "verdict": ("NOT RUN" if any(p.get("verdict") == "NOT RUN" for p in pairs)
                    else "UNRESOLVED - no pair resolves" if not scored
                    else "HELD" if all(p["verdict"] == "NON-INCREASING" for p in scored)
                    else "FAILED"),
    }

    # ---- S3 --------------------------------------------------------------
    s3: dict = {
        "statement": ("gap(fresh) rises with parameter count and at d=1481 "
                      "exceeds the baseline by more than 2 sigma_gap, counted "
                      "ONLY where delta_test <= 0 at that rung"),
        "baseline_gap_fresh": BASELINE_GAP_FRESH,
        "margin_required": TWO_SIGMA_GAP_FRESH,
        "guard": ("a bar on a difference must constrain both terms -- this is "
                  "EXP_013 N1's defect, and the delta_test<=0 condition is the "
                  "correction EXP_013 §10 item 1 referred"),
    }
    if gap is None:
        s3["verdict"] = "NOT RUN"
    else:
        rows = {r["run"]: r for r in gap.get("runs", [])}
        legs = []
        for d, _ in LADDER:
            r = rows.get(_run_name(d))
            if not r:
                continue
            gfresh = r["fresh"]["gap"]
            dtest = r["fresh"]["test_bpc"] - ANCHOR_FRESH
            dgap = gfresh - BASELINE_GAP_FRESH
            leg = {
                "d_model": d, "gap_fresh": gfresh,
                "delta_gap_vs_baseline": dgap,
                "delta_test_fresh_vs_anchor": dtest,
                "delta_test_le_zero": dtest <= 0,
                "below_noise_floor": abs(dgap) < NOISE_FLOOR_BPC,
            }
            if leg["below_noise_floor"]:
                leg["verdict"] = "BELOW WHAT THIS INSTRUMENT RESOLVES"
            elif not leg["delta_test_le_zero"]:
                leg["verdict"] = ("GAP MOVED BY DAMAGE - does not count "
                                  "(test leg is worse)")
            elif dgap >= TWO_SIGMA_GAP_FRESH:
                leg["verdict"] = "COUNTS - gap grew with a non-worsening test leg"
            else:
                leg["verdict"] = "UNRESOLVED"
            legs.append(leg)
        s3["legs"] = legs
        top_leg = next((x for x in legs if x["d_model"] == 1481), None)
        s3["verdict"] = ("NOT RUN" if top_leg is None
                         else "HELD" if top_leg["verdict"].startswith("COUNTS")
                         else top_leg["verdict"])
    out["S3"] = s3

    # ---- S4 --------------------------------------------------------------
    s4: dict = {
        "statement": ("above 10 sigma, more than half of delta bpc lands in "
                      "zero-context + within-reach; and the horizon does not "
                      "exceed 8 at any rung"),
        "cut_at_context": BASELINE_HORIZON,
        "horizon_ceiling": HORIZON_CEILING,
        "horizon_resolution_note": ("6, 7 and 8 all count as 'did not move' -- "
                                    "EXP_013 §9.6 saw 6 and 7 across nine runs it "
                                    "concluded did not move the horizon, against a "
                                    "baseline of 5/5 seeds at exactly 7"),
    }
    if horizon is None:
        s4["verdict"] = "NOT RUN"
    else:
        curves = horizon.get("absolute_context_curves", {})
        by_arm = horizon.get("by_arm", {})
        base_curve = (curves.get("snn_beta0.5") or {}).get("bpc_at_context")
        legs = []
        for d, _ in LADDER:
            arm = _arm_name(d)
            hz = by_arm.get(arm, {}).get("_paired", {})
            per_seed = hz.get("horizon_2sigma_per_seed") or []
            leg: dict = {"d_model": d, "horizon_per_seed": per_seed}
            leg["horizon_within_ceiling"] = all(
                h is not None and h <= HORIZON_CEILING for h in per_seed) if per_seed else None
            cur = (curves.get(arm) or {}).get("bpc_at_context")
            if base_curve and cur:
                cmax = str(max(int(k) for k in cur))
                h = str(BASELINE_HORIZON)
                total = base_curve[cmax] - cur[cmax]
                zero = base_curve["0"] - cur["0"]
                at_h = base_curve[h] - cur[h]
                within = at_h - zero
                beyond = total - at_h
                leg["decomposition"] = {
                    "total_gain_bpc": total,
                    "zero_context_gain_bpc": zero,
                    "within_reach_gain_bpc": within,
                    "beyond_horizon_gain_bpc": beyond,
                    "sums_to_total": abs((zero + within + beyond) - total) < 1e-9,
                }
                if abs(total) >= TEN_SIGMA:
                    share = (zero + within) / total if total else float("nan")
                    leg["near_share"] = share
                    leg["share_verdict"] = "HELD" if share > 0.5 else "FAILED"
                else:
                    leg["share_verdict"] = (
                        "NOT SCORED - |delta| below 10 sigma, shares of a delta "
                        "smaller than the noise are meaningless")
            else:
                leg["decomposition"] = None
                leg["share_verdict"] = (
                    "NOT SCORED - absolute_context_curves absent from the horizon "
                    "artifact; the baseline arm must be probed alongside the rungs "
                    "for the per-context curves to be computed")
            legs.append(leg)
        s4["legs"] = legs
        hv = [x["horizon_within_ceiling"] for x in legs if x["horizon_within_ceiling"] is not None]
        s4["horizon_verdict"] = ("NOT RUN" if not hv
                                 else "HELD - horizon did not move" if all(hv)
                                 else "FAILED - a rung exceeds the ceiling")
    out["S4"] = s4

    # ---- D1 and G1 -------------------------------------------------------
    anchor_clip = per_d.get(512, {}).get("d1_clip", {})
    top_clip = per_d.get(1481, {}).get("d1_clip", {})
    out["D1"] = {
        "statement": "diagnostic, not a bar: does the frozen recipe still train this width?",
        "why_not_a_bar": ("a health check that passes says nothing about whether "
                          "the optimiser is the binding constraint; D1 gates "
                          "INTERPRETATION (§4) and is reported whatever it says"),
        "anchor_frac_over_clip": anchor_clip.get("frac_over_clip"),
        "top_rung_frac_over_clip": top_clip.get("frac_over_clip"),
        "clip_binds_materially_harder_at_top": (
            None if not anchor_clip.get("available") or not top_clip.get("available")
            else top_clip["frac_over_clip"] > anchor_clip["frac_over_clip"] + 0.05),
        "any_nonfinite_loss": any(
            (per_d.get(d, {}).get("d1_clip", {}) or {}).get("n_nonfinite_loss_records", 0) > 0
            for d, _ in LADDER),
    }
    if manifest:
        out["G1"] = manifest.get("g1")
        out["K1"] = [r.get("k1") for r in manifest.get("rungs", [])]
        out["K3"] = [r.get("k3") for r in manifest.get("rungs", []) if r.get("k3")]

    # ---- §4's decision cell ---------------------------------------------
    v1 = out["S1"].get("verdict")
    harder = out["D1"]["clip_binds_materially_harder_at_top"]
    if v1 == "HELD" and harder is False:
        cell = ("Capacity moves the number at the frozen recipe. No size is "
                "adopted and no candidate is ranked -- n=1 cannot clear §6.2's "
                ">=3-seed rule. Refer to Elliot.")
    elif v1 == "HELD" and harder:
        cell = ("Capacity moved the number, AND this design cannot attribute how "
                "much of it to capacity -- the clip binds materially harder at "
                "the top rung. Report both; do not quote the delta as a capacity "
                "coefficient; refer a clip/lr sweep at fixed width.")
    elif v1 == "UNRESOLVED":
        cell = out["S1"]["wording_fixed_in_advance"]
    elif v1 == "HELD IN THE HARM DIRECTION":
        cell = ("The frozen recipe does not scale. A FINDING, not a null: it "
                "bounds every capacity-flavoured candidate §7.1 promoted.")
    else:
        cell = "NOT RUN"
    out["decision_cell"] = cell
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=None)
    ap.add_argument("--print-bars", action="store_true")
    args = ap.parse_args()

    if args.print_bars:
        print(json.dumps(BARS, indent=1))
        print("\nladder:", LADDER)
        return 0

    res = resolve()
    print(f"S1 {res['S1'].get('verdict')}   S2 {res['S2'].get('verdict')}   "
          f"S3 {res['S3'].get('verdict')}   S4 horizon "
          f"{res['S4'].get('horizon_verdict')}")
    print(f"\ndecision cell: {res['decision_cell']}")
    for r in res["rungs"]:
        print(f"  d={r['d_model']:>5} params={r['params']:>9,} "
              f"carried={r.get('test_bpc_carried')} "
              f"clip={r['d1_clip'].get('n_over_clip')}/{r['d1_clip'].get('n_logged_steps')}")
    if args.out:
        p = Path(args.out)
        if not p.is_absolute():
            p = _REPO / p
        p.write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8")
        print(f"\nwrote {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
