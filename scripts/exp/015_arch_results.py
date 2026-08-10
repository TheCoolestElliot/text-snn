"""EXP_015 resolver -- score the architecture ladder against its pre-registered bars.

    python scripts/exp/015_arch_results.py --print-bars     # transcription check
    python scripts/exp/015_arch_results.py --out docs/reports/data/exp_015_arch_results.json

WRITTEN AND COMMITTED BEFORE ANY RUNG'S bpc WAS READ.  Every threshold below is
transcribed from `experiments/logs/EXP_015_architecture_at_width.md` and
recomputed nowhere; `--print-bars` dumps them next to the pre-registration so a
transcription error is caught by reading rather than by luck.

WHAT THIS FILE MAY NOT DO, AND THE REASONS ARE IN THE LOG
----------------------------------------------------------
* It may not adopt an arm or choose a size.  n = 1 per leg against Sec 6.2's
  three-seed rule.  `adopts_nothing` and `ranks_nothing` are emitted as fields.
* It may not resolve a miss as "width subsumes the architecture".  At n = 1 the
  design can show an effect exceeds a threshold and can never show one is absent
  (Sec 3.0).  The wording of a miss is fixed here, in advance: UNRESOLVED.
* It may not widen a bar that fails, or quote P1b as a verdict.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
RUNS = _REPO / "experiments" / "runs"
DATA = _REPO / "docs" / "reports" / "data"

# ---------------------------------------------------------------------------
# Bars, transcribed from EXP_015. Nothing here is computed from a result.
# ---------------------------------------------------------------------------

#: Sec 3.0. Baseline cross-seed sd, n=5 at d=512 (EXP_000). Applied to BOTH terms
#: of every difference -- the conservative choice, since EXP_005 puts the
#: two-compartment family at 0.00313.
SIGMA = 0.00461
#: Both terms are n=1, so the statistic is the sd of a difference.
SIGMA_DIFF = 0.00652          # = sqrt(2) * 0.00461, to 3 s.f.
TEN_SIGMA_DIFF = 0.0652       # P1 and P2's bar
TWO_SIGMA_DIFF = 0.0130

CONTROL_RUN = "scale_d1481_s0"
D_MODEL = 1481

#: Sec 1's committed 735K table, for P1b's retained-fraction marker only.
REF_735K = {
    "baseline_carried": 2.25311,
    "twocomp_carried": 2.11869,
    "composed_carried": 2.08326,
    "gru_carried": 1.76741,
    "delta_twocomp": -0.13442,
    "delta_composed": -0.16985,
    "delta_gru": -0.48570,
}

#: Sec P4. The baseline's horizon at 735K and at this width, and twocomp's at
#: 735K. The probe lattice is k in {1,2,4,...,256}; a one-step move is a marker.
HORIZON = {"baseline_735k": 7, "twocomp_735k": 47, "control_1481": 8}

#: Sec P5. EXP_014's measured fresh gaps along the width ladder.
GAP_FRESH_CONTROL_1481 = 0.05509
GAP_FRESH_BASELINE_735K = 0.00594

#: Sec 6.5's reparameterisation noise floor, and EXP_014 Sec 9.12's tree offset.
NOISE_FLOOR_BPC = 2e-3
CLIP_TREE_OFFSET_BPC = 1.97e-3

ARMS = ("twocomp", "twocomp_threshold", "gru")
BAR_ARMS = ("twocomp", "twocomp_threshold")   # P1 and P2; gru has no bar (P3)

BARS = {
    "sigma": SIGMA,
    "sigma_diff": SIGMA_DIFF,
    "sigma_diff_derivation": "sqrt(SIGMA^2 + SIGMA^2); both terms are n=1",
    "P1": {
        "arm": "twocomp",
        "rule": "bpc_carried(arm) <= bpc_carried(control) - 0.0652",
        "bar_bpc": TEN_SIGMA_DIFF,
        "bar_in_sigma_diff": 10,
        "miss_resolves_as": "UNRESOLVED",
        "miss_may_not_be_written_as": "width subsumes the architecture",
    },
    "P2": {
        "arm": "twocomp_threshold",
        "rule": "bpc_carried(arm) <= bpc_carried(control) - 0.0652",
        "bar_bpc": TEN_SIGMA_DIFF,
        "bar_in_sigma_diff": 10,
        "composed_minus_twocomp_at_735k": -0.03543,
        "composed_minus_twocomp_in_sigma_diff": 5.4,
        "note": "arm-vs-arm is BELOW the bar and is a marker with no verdict",
    },
    "P3": {"arm": "gru", "rule": "no bar; report the gap at matched size",
           "why": "the anchor violates I5 and is not a competitor"},
    "P4": {"rule": "horizon per arm, marker only", "lattice_is_coarse": True,
           "no_error_bar_at_n1": True, **HORIZON},
    "P5": {
        "rule": "gap(fresh, twocomp) > gap(fresh, control), counted only where "
                "delta_test <= 0 at that arm",
        "control_gap_fresh": GAP_FRESH_CONTROL_1481,
        "guard": "delta_test <= 0 -- the correction EXP_013's N1 forced",
    },
    "P6": {"rule": "wall-clock, peak VRAM, firing rate; reusable, no bar"},
    "structural": {"adopts_nothing": True, "ranks_nothing": True,
                   "n_per_arm": 1, "sigma_is_transferred": True},
}


def _load(name: str) -> dict | None:
    p = DATA / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def _bpc(run: str) -> dict | None:
    p = RUNS / run / "final_test.json"
    if not p.exists():
        return None
    d = json.loads(p.read_text(encoding="utf-8"))["results"]
    return {"fresh": d["fresh"]["bpc"], "carried": d["carried"]["bpc"],
            "firing_rate_carried": d["carried"].get("firing_rate")}


def _verdict_vs_control(delta: float | None) -> str:
    """The only place a P1/P2 verdict is decided, and its vocabulary is fixed."""
    if delta is None:
        return "NOT RUN"
    if delta <= -TEN_SIGMA_DIFF:
        return "HELD"
    if delta >= TEN_SIGMA_DIFF:
        return "HELD IN THE HARM DIRECTION"
    return "UNRESOLVED"


def resolve() -> dict:
    manifest = _load("exp_015_run_manifest.json") or {}
    horizon = _load("exp_015_memory_horizon.json")
    gap = _load("exp_015_gap.json")

    control = _bpc(CONTROL_RUN)
    if control is None:
        raise SystemExit(f"the control leg {CONTROL_RUN} has no final_test.json")

    runs = {r["arch"]: r for r in manifest.get("rungs", [])}
    out: dict = {
        "experiment": "EXP_015_architecture_at_width",
        "d_model": D_MODEL,
        "adopts_nothing": True,
        "ranks_nothing": True,
        "n_per_arm": 1,
        "sigma_is_transferred": True,
        "bars": BARS,
        "control": {"run": CONTROL_RUN, **control},
        "arms": {},
    }

    for arch in ARMS:
        run = runs.get(arch, {})
        name = run.get("run")
        b = _bpc(name) if name else None
        row: dict = {
            "run": name,
            "completed": bool(run.get("completed")),
            "params": run.get("params"),
            "wall_clock_s": run.get("wall_clock_s"),
            "peak_vram_gib": run.get("peak_vram_gib"),
            "divergence_scan": run.get("divergence_scan"),
        }
        if b:
            row.update(b)
            row["delta_carried_vs_control"] = b["carried"] - control["carried"]
            row["delta_fresh_vs_control"] = b["fresh"] - control["fresh"]
            row["delta_in_sigma_diff"] = round(
                row["delta_carried_vs_control"] / SIGMA_DIFF, 2)
        if arch in BAR_ARMS:
            row["verdict"] = _verdict_vs_control(row.get("delta_carried_vs_control"))
            # P1b: a MARKER. Cross-tree and one n=1 term, so no verdict attaches
            # unless it clears the bar in its own right.
            ref = (REF_735K["delta_twocomp"] if arch == "twocomp"
                   else REF_735K["delta_composed"])
            d = row.get("delta_carried_vs_control")
            row["p1b_marker"] = {
                "delta_at_735k": ref,
                "delta_at_1481": d,
                "retained_fraction": None if d is None else round(d / ref, 3),
                "verdict": "MARKER - NO VERDICT AT n=1",
                "why": ("a difference of differences with one n=1 term and one "
                        f"cross-tree term (the {CLIP_TREE_OFFSET_BPC:.2e} bpc "
                        "offset EXP_014 Sec 9.12 established)"),
            }
        else:
            row["verdict"] = "NO BAR - ANCHOR (P3)"
            d = row.get("delta_carried_vs_control")
            row["gap_to_anchor_at_matched_size"] = None if d is None else -d
            row["gap_to_anchor_at_735k"] = -REF_735K["delta_gru"]
        out["arms"][arch] = row

    # P2's arm-vs-arm marker, explicitly not a verdict.
    tc = out["arms"]["twocomp"].get("carried")
    cp = out["arms"]["twocomp_threshold"].get("carried")
    out["composition_marker"] = {
        "composed_minus_twocomp_at_1481": None if (tc is None or cp is None) else cp - tc,
        "composed_minus_twocomp_at_735k": -0.03543,
        "bar_would_be": TEN_SIGMA_DIFF,
        "verdict": "MARKER - NO VERDICT AT n=1",
        "why": ("5.4 sigma_diff at 735K, below the 10 sigma_diff bar. "
                "'The composition stopped helping at width' may not be written "
                "from this. Seeds are the remedy."),
    }

    # P4 -- horizon, marker only.
    #
    # CORRECTED AFTER THE RUN, and the correction is a crash rather than a bar.
    # As committed, this block assumed `exp_015_memory_horizon.json` carried a
    # LIST of per-arm dicts; `001_memory_horizon.py` actually keys `by_arm` by
    # arm name, with the horizons under a `_paired` sub-key. The resolver raised
    # AttributeError and produced nothing. No threshold, bar or verdict is
    # touched by this fix -- P4 has no bar (Sec P4: "marker with no verdict") --
    # and the pre-registration's text is unchanged. Recorded in Sec 9 rather than
    # repaired silently.
    if horizon:
        by_arm = {}
        for arm, block in (horizon.get("by_arm") or {}).items():
            paired = block.get("_paired", {}) if isinstance(block, dict) else {}
            by_arm[arm] = {
                "n": paired.get("n"),
                "horizon_2sigma_per_seed": paired.get("horizon_2sigma_per_seed"),
                "horizon_0.05bpc_per_seed": paired.get("horizon_0.05bpc_per_seed"),
            }
        out["P4_horizon"] = {
            "verdict": "MARKER - NO VERDICT AT n=1",
            "reference": HORIZON,
            "f1_failures": horizon.get("f1_failures"),
            "by_arm": by_arm,
        }
    else:
        out["P4_horizon"] = {"verdict": "NOT RUN"}

    # P5 -- the gap, with the delta_test <= 0 guard that EXP_013's N1 forced.
    if gap:
        legs = {g.get("run"): g for g in gap.get("results", gap.get("legs", []))}
        rows = {}
        for arch in ARMS:
            name = out["arms"][arch].get("run")
            g = legs.get(name)
            if not g:
                continue
            gf = g.get("gap_fresh", g.get("fresh", {}).get("gap"))
            dtest = out["arms"][arch].get("delta_fresh_vs_control")
            counts = dtest is not None and dtest <= 0
            rows[arch] = {
                "gap_fresh": gf,
                "delta_vs_control_gap": None if gf is None else gf - GAP_FRESH_CONTROL_1481,
                "delta_test_fresh_vs_control": dtest,
                "counts": counts,
                "verdict": ("COUNTS" if counts
                            else "DOES NOT COUNT - test leg is not better"),
            }
        out["P5_gap"] = {"control_gap_fresh": GAP_FRESH_CONTROL_1481, "by_arm": rows}
    else:
        out["P5_gap"] = {"verdict": "NOT RUN"}

    # P6 -- systems, no bar.
    out["P6_systems"] = {
        "denominator": {"run": CONTROL_RUN, "wall_clock_s": 2191.44,
                        "peak_vram_gib": 1.6434,
                        "firing_rate_carried": [0.20868549799840702,
                                                0.18502129967275419]},
        "by_arm": {
            a: {"wall_clock_s": out["arms"][a].get("wall_clock_s"),
                "x_control": (None if not out["arms"][a].get("wall_clock_s")
                              else round(out["arms"][a]["wall_clock_s"] / 2191.44, 4)),
                "peak_vram_gib": out["arms"][a].get("peak_vram_gib"),
                "firing_rate_carried": out["arms"][a].get("firing_rate_carried")}
            for a in ARMS},
    }

    # Gates, read from the manifest rather than re-derived.
    out["gates"] = {
        "k1": {a: runs.get(a, {}).get("k1") for a in ARMS},
        "k3": {a: runs.get(a, {}).get("k3") for a in ARMS},
        "prereg_unchanged": manifest.get("prereg_unchanged"),
        "prereg_sha256_before_first_run": manifest.get("prereg_sha256_before_first_run"),
        "prereg_sha256_after_last_run": manifest.get("prereg_sha256_after_last_run"),
        "g1": "not applicable -- every leg is anchored to the control on this tree",
        "any_nonfinite_loss": any(
            (runs.get(a, {}).get("divergence_scan") or {}).get(
                "n_nonfinite_loss_records") for a in ARMS),
    }

    # The decision cell, mapped in advance from (P1, P2) -- Sec 4.
    p1 = out["arms"]["twocomp"]["verdict"]
    p2 = out["arms"]["twocomp_threshold"]["verdict"]
    if p1 == "HELD" and p2 == "HELD":
        cell = ("The architecture's advantage survives at 5.0M parameters. "
                "Nothing is adopted -- n=1. Refer to Elliot: whether Phase 5 "
                "scales the composed arm, and whether the top leg is reseeded "
                "to n=3 before any number from it is quoted.")
    elif p1 == "HELD":
        cell = ("The two-compartment advantage survives at width; the composed "
                "arm at width is NOT established. Refer whether to reseed leg B.")
    elif p1.startswith("HELD IN THE HARM"):
        cell = ("The arm is WORSE than the plain LIF at matched size. Reported, "
                "not explained here. Refer -- a reversal at width would be the "
                "most consequential result in Phase 4.")
    else:
        cell = ("The difference was not established at n=1. This is NOT 'width "
                "subsumes the architecture'. The measurement that would settle "
                "it is seeds, at ~0.73 GPU-h per twocomp seed.")
    out["decision_cell"] = cell
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(DATA / "exp_015_arch_results.json"))
    ap.add_argument("--print-bars", action="store_true",
                    help="dump the transcribed bars and exit; reads no result")
    args = ap.parse_args(argv)

    if args.print_bars:
        print(json.dumps(BARS, indent=1))
        return 0

    res = resolve()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8")

    c = res["control"]["carried"]
    print(f"control {CONTROL_RUN}: {c:.5f} carried\n")
    print(f"{'arm':<20}{'carried':>10}{'delta':>10}{'sigma_d':>9}  verdict")
    for a in ARMS:
        r = res["arms"][a]
        if r.get("carried") is None:
            print(f"{a:<20}{'--':>10}{'--':>10}{'--':>9}  {r['verdict']}")
            continue
        print(f"{a:<20}{r['carried']:>10.5f}{r['delta_carried_vs_control']:>10.5f}"
              f"{r['delta_in_sigma_diff']:>9.1f}  {r['verdict']}")
    print(f"\n{res['decision_cell']}")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
