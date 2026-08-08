"""EXP_013: resolve every prediction and the decision rule, mechanically.

Pre-registered in `experiments/logs/EXP_013_noise_injection.md` §3 and §4.
**Written and run before any noise-arm number was read**, which is the only
property that makes a resolver worth having: `scripts/exp/010_phase4_arm_results.py`
was committed at `f9c509c` before `EXP_007`-`EXP_009`'s numbers were read, and
this file follows it.

Every bar below is transcribed from the pre-registration, and the transcription
is asserted against nothing -- there is no way to check a constant against a
prose document automatically, so they are quoted here beside the section that
fixed them and `--print-bars` dumps them for eyeball comparison. `EXP_012`'s
"sets_no_tolerance" row is the precedent for making a file's relationship to its
own thresholds inspectable.

WHAT THIS FILE MAY NOT DO
-------------------------
It may not adopt anything, may not rank anything, and may not widen a bar that
fails. Four consecutive precedents -- `EXP_005` §9.4, `EXP_008` §9.4, `EXP_011`
§0, `EXP_012` §11.4 -- establish that a threshold which falls the wrong way is
reported as it fired and the correction is *referred*.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
DATA = _REPO / "docs" / "reports" / "data"
RUNS = _REPO / "experiments" / "runs"

# --------------------------------------------------------------------------
# Bars, transcribed from EXP_013 §3. Nothing below recomputes them.
# --------------------------------------------------------------------------

#: Phase-2 baseline whole-split sd (EXP_000), the standing bar for this neuron.
SIGMA_BPC = 0.00461
BAR_BPC = 2 * SIGMA_BPC                       # 0.00922

#: Measured on the five committed baseline seeds, §2.2. NOT inherited.
SIGMA_GAP = {"fresh": 0.00268, "carried": 0.00306}
BAR_GAP = {p: 2 * s for p, s in SIGMA_GAP.items()}   # 0.00536 / 0.00612

#: §5 item 7's guard: the reparameterisation noise floor `04_phase4_interim.md`
#: §6.5 measured. A Delta whose magnitude is below this is reported as "below
#: what this instrument resolves", never as a number with a sign.
NOISE_FLOOR = 2e-3

#: §2.2's committed baseline gaps, for Delta-gap. Recomputed from the artifact
#: rather than hard-coded, so a re-run of the calibration cannot silently
#: disagree with this file -- but asserted against these to catch the reverse.
BASELINE_GAP_EXPECTED = {"fresh": 0.00594, "carried": -0.00180}

#: §2.4's ladder.
AMPLITUDES = (0.1, 0.4, 1.6)

PROTOCOLS = ("fresh", "carried")


def _load(path: Path) -> dict:
    if not path.exists():
        raise SystemExit(f"missing artifact: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _resolved(delta: float) -> bool:
    """§5 item 7: is this Delta above the instrument's own floor?"""
    return abs(delta) >= NOISE_FLOOR


def collect(gap_baseline: dict, gap_noise: dict, manifest: dict) -> dict:
    """Group the noise runs by amplitude and compute every quantity §3 needs."""
    base = {p: gap_baseline["summary"][p] for p in PROTOCOLS}
    for p in PROTOCOLS:
        want = BASELINE_GAP_EXPECTED[p]
        got = base[p]["gap_mean"]
        if abs(got - want) > 1e-5:
            raise SystemExit(
                f"the baseline gap artifact says {got:.5f} for {p} but §2.2 "
                f"fixed the bar against {want:.5f}. One of them has been "
                "recomputed; the bar may not move to follow it."
            )

    by_amp: dict[float, list[dict]] = {a: [] for a in AMPLITUDES}
    for row in gap_noise["runs"]:
        amp = float(row["noise_amp"])
        if amp in by_amp:
            by_amp[amp].append(row)

    rates = {}
    for row in manifest["runs"]:
        rates.setdefault(float(row["noise_amp"]), []).append(
            row.get("test_firing_rate"))

    out = {"baseline": base, "amplitudes": {}}
    for amp in AMPLITUDES:
        rows = by_amp[amp]
        if not rows:
            continue
        entry: dict = {"n": len(rows), "runs": [r["run"] for r in rows]}
        for p in PROTOCOLS:
            gaps = [r[p]["gap"] for r in rows]
            tests = [r[p]["test_bpc"] for r in rows]
            trains = [r[p]["train_bpc"] for r in rows]
            n = len(rows)
            d_gap = statistics.fmean(gaps) - base[p]["gap_mean"]
            d_bpc = statistics.fmean(tests) - base[p]["test_bpc_mean"]
            # se of a difference of two independent means, house convention
            # (§6.1 quotes arms in "se"). n_base is 5.
            se_gap = ((SIGMA_GAP[p] ** 2) * (1 / n + 1 / 5)) ** 0.5
            se_bpc = ((SIGMA_BPC ** 2) * (1 / n + 1 / 5)) ** 0.5
            entry[p] = {
                "gap_mean": statistics.fmean(gaps),
                "gap_sd": statistics.stdev(gaps) if n > 1 else None,
                "test_bpc_mean": statistics.fmean(tests),
                "test_bpc_sd": statistics.stdev(tests) if n > 1 else None,
                "train_bpc_mean": statistics.fmean(trains),
                "delta_gap": d_gap,
                "delta_gap_se": d_gap / se_gap if se_gap else None,
                "delta_gap_resolved": _resolved(d_gap),
                "delta_test_bpc": d_bpc,
                "delta_test_bpc_se": d_bpc / se_bpc if se_bpc else None,
                "gaps": gaps,
                "test_bpcs": tests,
            }
        entry["test_firing_rate_mean"] = (
            [statistics.fmean(x) for x in zip(*[r for r in rates.get(amp, [])
                                                if r])]
            if rates.get(amp) and all(rates[amp]) else None)
        out["amplitudes"][str(amp)] = entry
    return out


def resolve(c: dict) -> dict:
    """§3's six predictions and §4's decision cell, applied as written."""
    amps = [a for a in AMPLITUDES if str(a) in c["amplitudes"]]
    A = {a: c["amplitudes"][str(a)] for a in amps}
    preds: dict = {}

    # N1 -- primary. At least one amplitude reduces the FRESH gap by >= 2 sigma_gap.
    n1_hits = [a for a in amps if A[a]["fresh"]["delta_gap"] <= -BAR_GAP["fresh"]]
    preds["N1"] = {
        "statement": "at least one amplitude has delta_gap <= -2*sigma_gap (fresh)",
        "bar": -BAR_GAP["fresh"],
        "values": {str(a): A[a]["fresh"]["delta_gap"] for a in amps},
        "held": bool(n1_hits),
        "amplitudes_that_hit": n1_hits,
        "predicted_to_fail_in_advance": True,
        "note": "§2.3 states the ceiling that makes this underpowered: the whole "
                "fresh gap is 0.00594 including a region-difficulty term no "
                "regulariser can touch, against a bar of 0.00536.",
    }

    # N2 -- test bpc degrades monotonically, and 1.6 is worse than the bar.
    seq = [A[a]["carried"]["delta_test_bpc"] for a in amps]
    monotone = all(seq[i] <= seq[i + 1] for i in range(len(seq) - 1))
    worst = A[max(amps)]["carried"]["delta_test_bpc"] if amps else None
    preds["N2"] = {
        "statement": "delta test bpc (carried) is monotone increasing in amplitude "
                     "AND at the top of the ladder exceeds +2*sigma",
        "bar": BAR_BPC,
        "values": {str(a): A[a]["carried"]["delta_test_bpc"] for a in amps},
        "monotone": monotone,
        "top_of_ladder_delta": worst,
        "held": bool(monotone and worst is not None and worst > BAR_BPC),
    }

    # N3 -- the bottom of the ladder is in the null region, both directions.
    lo = min(amps) if amps else None
    d_lo = A[lo]["carried"]["delta_test_bpc"] if lo is not None else None
    preds["N3"] = {
        "statement": "at the bottom of the ladder |delta test bpc| < 2*sigma",
        "bar": BAR_BPC,
        "amplitude": lo,
        "value": d_lo,
        "held": bool(d_lo is not None and abs(d_lo) < BAR_BPC),
        "direction": None if d_lo is None else ("harm" if d_lo > 0 else "help"),
    }

    # N4 -- firing rate rises monotonically, both layers.
    rate_rows = [A[a].get("test_firing_rate_mean") for a in amps]
    n4 = None
    if all(r for r in rate_rows):
        n4 = all(
            all(rate_rows[i][k] <= rate_rows[i + 1][k]
                for i in range(len(rate_rows) - 1))
            for k in range(len(rate_rows[0]))
        )
    preds["N4"] = {
        "statement": "mean firing rate rises monotonically in amplitude, every layer",
        "values": {str(a): A[a].get("test_firing_rate_mean") for a in amps},
        "held": n4,
    }

    # N5 -- the direction opposite to N1. Fails only on a RESOLVED negative.
    n5_fails = [a for a in amps
                if A[a]["fresh"]["delta_gap"] <= -NOISE_FLOOR]
    preds["N5"] = {
        "statement": "delta_gap >= 0 at every amplitude (fresh); fails only on a "
                     "delta below -noise_floor, per §5 item 7",
        "noise_floor": NOISE_FLOOR,
        "values": {str(a): A[a]["fresh"]["delta_gap"] for a in amps},
        "resolved": {str(a): A[a]["fresh"]["delta_gap_resolved"] for a in amps},
        "held": not n5_fails,
        "amplitudes_that_fail": n5_fails,
    }

    # N6 -- the arm's own sigma, within a factor of 2 of the baseline's.
    sds = {str(a): A[a]["carried"]["test_bpc_sd"] for a in amps}
    ok = [s for s in sds.values() if s is not None
          and SIGMA_BPC / 2 <= s <= SIGMA_BPC * 2]
    preds["N6"] = {
        "statement": "the arm's cross-seed sd of test bpc is within a factor of 2 "
                     "of the baseline's 0.00461",
        "band": [SIGMA_BPC / 2, SIGMA_BPC * 2],
        "values": sds,
        "held": len(ok) == len([s for s in sds.values() if s is not None]),
        "power_note": "n=3 estimates a standard deviation to about +/-40 % "
                      "(EXP_004 §10.7); stated in §5 item 5 before the run.",
    }

    # -- §4's decision rule, applied as written -----------------------------
    if preds["N1"]["held"]:
        cell = "N1 holds"
        verdict = ("The arm reduces the generalisation gap. REPORT the effect and "
                   "REFER adoption to Elliot, with §2.3's ceiling quoted beside "
                   "it: a reduction of >=90 % of a gap that includes an "
                   "irreducible region-difficulty term demands an explanation "
                   "before it is believed.")
    elif preds["N3"]["held"]:
        cell = "N1 fails AND N3 holds"
        verdict = ("NULL, and it is reported as 'there was no gap to close' -- "
                   "never as 'noise injection does not regularise'. The arm is "
                   "free at small amplitude and buys nothing measurable. NOT "
                   "recommended for adoption. The transferable result is §2.2's: "
                   "this model does not overfit.")
    elif preds["N3"]["direction"] == "harm":
        cell = "N1 fails AND N3 fails, harm direction"
        verdict = ("NEGATIVE RESULT: the arm costs bits even at the bottom of the "
                   "ladder. Report the amplitude at which harm begins; it bounds "
                   "every future arm that perturbs the current, including the "
                   "quantisation `04_phase4_interim.md` §6.5 names as unmeasured.")
    else:
        cell = "N1 fails AND N3 fails, help direction"
        verdict = ("STOP AND CHASE before reporting anything. An improvement from "
                   "a zero-mean perturbation with no gap to close is most likely "
                   "§1.2's confound leaking back in; check first that the injected "
                   "mean is zero to machine precision on the actual draws.")

    return {
        "predictions": preds,
        "held": sorted(k for k, v in preds.items() if v["held"] is True),
        "failed": sorted(k for k, v in preds.items() if v["held"] is False),
        "unresolved": sorted(k for k, v in preds.items() if v["held"] is None),
        "decision_cell": cell,
        "verdict": verdict,
        "adopts_nothing": True,
        "ranks_nothing": True,
        "bars_are_transcribed_not_computed": True,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gap-baseline", default=str(DATA / "exp_013_gap_baseline.json"))
    ap.add_argument("--gap-noise", default=str(DATA / "exp_013_gap_noise.json"))
    ap.add_argument("--manifest", default=str(DATA / "exp_013_run_manifest.json"))
    ap.add_argument("--out", default=str(DATA / "exp_013_noise_results.json"))
    ap.add_argument("--print-bars", action="store_true")
    args = ap.parse_args()

    if args.print_bars:
        print(json.dumps({
            "SIGMA_BPC": SIGMA_BPC, "BAR_BPC": BAR_BPC,
            "SIGMA_GAP": SIGMA_GAP, "BAR_GAP": BAR_GAP,
            "NOISE_FLOOR": NOISE_FLOOR, "AMPLITUDES": list(AMPLITUDES),
            "BASELINE_GAP_EXPECTED": BASELINE_GAP_EXPECTED,
        }, indent=2))
        return 0

    collected = collect(_load(Path(args.gap_baseline)),
                        _load(Path(args.gap_noise)),
                        _load(Path(args.manifest)))
    resolution = resolve(collected)

    payload = {
        "artifact": "exp_013_noise_results",
        "experiment": "EXP_013_noise_injection",
        "log": "experiments/logs/EXP_013_noise_injection.md",
        "bars": {"SIGMA_BPC": SIGMA_BPC, "BAR_BPC": BAR_BPC,
                 "SIGMA_GAP": SIGMA_GAP, "BAR_GAP": BAR_GAP,
                 "NOISE_FLOOR": NOISE_FLOOR},
        "collected": collected,
        "resolution": resolution,
    }
    Path(args.out).write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("=" * 72)
    for amp in AMPLITUDES:
        e = collected["amplitudes"].get(str(amp))
        if not e:
            continue
        c, f = e["carried"], e["fresh"]
        print(f"amp {amp:<5g} n={e['n']}  carried bpc {c['test_bpc_mean']:.5f} "
              f"({c['delta_test_bpc']:+.5f}, {c['delta_test_bpc_se']:+.1f} se)   "
              f"fresh gap {f['gap_mean']:+.5f} (Dgap {f['delta_gap']:+.5f}, "
              f"resolved={f['delta_gap_resolved']})")
    print("-" * 72)
    for k in ("N1", "N2", "N3", "N4", "N5", "N6"):
        p = resolution["predictions"][k]
        state = {True: "HELD", False: "FAILED", None: "UNRESOLVED"}[p["held"]]
        print(f"  {k}: {state:10s} {p['statement']}")
    print("-" * 72)
    print(f"decision cell: {resolution['decision_cell']}")
    print(resolution["verdict"])
    print(f"\nWROTE {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
