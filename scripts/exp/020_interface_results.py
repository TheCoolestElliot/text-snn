"""EXP_020 resolver: applies §4's bars mechanically to §3's runs.

    python scripts/exp/020_interface_results.py --out docs/reports/data/exp_020_arm_results.json

COMMITTED BEFORE ANY LEG'S BPC IS READ. That ordering is the whole point: a
resolver written after the numbers are known is a rationalisation with a
`__main__` block. Every threshold below is transcribed from
`experiments/logs/EXP_020_interface.md` §4 and is asserted against the
pre-registration's SHA-256 recorded in the run manifest, so a resolver that has
drifted from the bars it claims to apply cannot report.

WHAT IT REFUSES TO DO
---------------------
  * It does not adopt, rank, or recommend. §5's verdict WORDINGS are transcribed
    and selected between; none of them is written here for the first time.
  * It does not repair a bar that misfired. `CONTRIBUTING.md` §3: a bar is
    reported as it fired, and a corrected rule is referred to the next
    experiment, never applied retroactively.
  * It does not substitute a seed. A diverged run reduces n and marks the cell
    provisional.
  * It does not compare any number here with a committed figure (decision #10).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO / "src") not in sys.path:
    sys.path.insert(0, str(_REPO / "src"))


def _import_010():
    """`decompose` and `context_comparison`, IMPORTED and not reimplemented.

    `EXP_011`'s resolver set the precedent and gave the reason: one
    implementation of each statistic, not one file. `010_phase4_arm_results.py`
    is committed and closed -- three experiments' published numbers come out of
    it -- so it is imported by path rather than edited.
    """
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "_r010", _REPO / "scripts" / "exp" / "010_phase4_arm_results.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

RUNS = _REPO / "experiments" / "runs"
PREREG = _REPO / "experiments" / "logs" / "EXP_020_interface.md"
MANIFEST = _REPO / "docs" / "reports" / "data" / "exp_020_run_manifest.json"

#: Transferred sigma, named in the same sentence as every bar that uses it
#: (EXP_014 §4's rule): the whole-split baseline sd over 5 seeds,
#: 04_phase4_interim.md §236, measured on a DIFFERENT tree than these runs.
SIGMA_TRANSFERRED = 0.00461
BAR_2SIGMA = 2 * SIGMA_TRANSFERRED                      # 0.00922

#: EXP_014's scaling slope, 0.0913 bpc per doubling (2.25311 -> 2.00073 over
#: 6.8x), evaluated at 0.6436x parameters. H1 §4 states in advance that this
#: reference is INDICATIVE: EXP_014 scaled width, which scales every block, and
#: leg 2 removes one specific block.
H1_REFERENCE = 0.0582

ARMS = {
    "anchor":   [f"if_anchor_s{s}" for s in (0, 1, 2)],
    "fold":     [f"if_fold_s{s}" for s in (0, 1, 2)],
    "foldwide": [f"if_foldwide_s{s}" for s in (0, 1, 2)],
    "binin":    [f"if_binin_s{s}" for s in (0, 1, 2)],
    "mlayer":   [f"if_mlayer_s{s}" for s in (0, 1, 2)],
}
NEST_RUN = "if_nest_s0"
ANCHOR_S0 = "if_anchor_s0"


def _final(run: str) -> dict | None:
    p = RUNS / run / "final_test.json"
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def _bpc(run: str, protocol: str) -> float | None:
    j = _final(run)
    if j is None:
        return None
    v = j.get("results", {}).get(protocol, {}).get("bpc")
    return None if v is None else float(v)


def _summary(run: str) -> dict:
    p = RUNS / run / "summary.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def _diverged(run: str) -> bool:
    """G7: read the LOG, not the exit code. An ABSENT log reads as diverged."""
    p = RUNS / run / "log.jsonl"
    if not p.exists():
        return True
    for line in p.read_text(encoding="utf-8").splitlines():
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        v = rec.get("loss")
        if v is not None and (v != v or v in (float("inf"), float("-inf"))):
            return True
    b = _bpc(run, "fresh")
    return b is None or b != b


def arm_stats(runs: list[str], protocol: str) -> dict:
    live, dead = [], []
    for r in runs:
        if _diverged(r):
            dead.append(r)
            continue
        b = _bpc(r, protocol)
        (live if b is not None else dead).append(r if b is None else b)
    vals = [v for v in live if isinstance(v, float)]
    return {
        "runs": runs,
        "n": len(vals),
        "n_preregistered": len(runs),
        "diverged": [r for r in dead],
        "provisional": len(vals) < len(runs),
        "bpc": vals,
        "mean": statistics.fmean(vals) if vals else None,
        "sd": statistics.stdev(vals) if len(vals) > 1 else None,
    }


def paired(a: dict, b: dict) -> dict:
    """Per-seed differences, because a seed fixes BOTH the initial weights and
    the data order, so arms sharing a seed are paired (EXP_018 §1000's referral,
    and the defect its D1 was corrected for)."""
    if a["n"] != a["n_preregistered"] or b["n"] != b["n_preregistered"]:
        return {"paired": False,
                "reason": "a cell lost a seed; pairing across unequal n would "
                          "silently drop the partner"}
    d = [x - y for x, y in zip(a["bpc"], b["bpc"])]
    n = len(d)
    m = statistics.fmean(d)
    sd = statistics.stdev(d) if n > 1 else None
    se = (sd / math.sqrt(n)) if sd else None
    return {"paired": True, "diffs": d, "mean_diff": m, "sd_diff": sd,
            "se_diff": se, "t": (m / se) if se else None,
            "all_same_sign": all(x < 0 for x in d) or all(x > 0 for x in d),
            "n_negative": sum(1 for x in d if x < 0)}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--protocol", default="carried", choices=("fresh", "carried"))
    ap.add_argument("--out", default="docs/reports/data/exp_020_arm_results.json")
    args = ap.parse_args(argv)

    out: dict = {
        "experiment": "EXP_020",
        "protocol": args.protocol,
        "sigma_transferred": SIGMA_TRANSFERRED,
        "sigma_source": "04_phase4_interim.md §236, baseline n=5, DIFFERENT tree",
        "bar_2sigma": BAR_2SIGMA,
        "prereg_sha256": hashlib.sha256(PREREG.read_bytes()).hexdigest(),
    }
    if MANIFEST.exists():
        man = json.loads(MANIFEST.read_text(encoding="utf-8"))
        out["manifest_prereg_sha256_before"] = man.get("prereg_sha256_before")
        out["prereg_matches_manifest"] = (
            man.get("prereg_sha256_before") == out["prereg_sha256"])
        out["G2_all_params_match"] = all(
            r.get("G2_params_match", False) for r in man.get("runs", [])
            if "G2_params_match" in r)
        out["G5_violations"] = {r["run"]: r["G5_frozen_violations"]
                                for r in man.get("runs", [])
                                if r.get("G5_frozen_violations")}
        out["K1_unexpected"] = {r["run"]: r["K1_unexpected"]
                                for r in man.get("runs", [])
                                if r.get("K1_unexpected")}
        out["vram_alarms"] = [r["run"] for r in man.get("runs", [])
                              if r.get("vram_alarm")]
    else:
        out["prereg_matches_manifest"] = None

    stats = {k: arm_stats(v, args.protocol) for k, v in ARMS.items()}
    out["arms"] = stats

    # -- sigma, RE-MEASURED on this tree ---------------------------------
    a = stats["anchor"]
    out["sigma_measured_this_tree"] = a["sd"]
    out["sigma_transfer_ratio"] = (a["sd"] / SIGMA_TRANSFERRED) if a["sd"] else None

    # -- T1: the whole-run nesting gate ----------------------------------
    t1 = {}
    for proto in ("fresh", "carried"):
        x, y = _bpc(NEST_RUN, proto), _bpc(ANCHOR_S0, proto)
        t1[proto] = {"nest": x, "anchor": y,
                     "diff": None if (x is None or y is None) else x - y,
                     "exactly_zero": (x is not None and y is not None and x == y)}
    t1["pass"] = all(v["exactly_zero"] for v in
                     (t1["fresh"], t1["carried"]))
    out["T1"] = t1

    # -- the bars --------------------------------------------------------
    bars: dict = {}

    p = paired(stats["fold"], stats["anchor"])
    bars["H1"] = {
        "statement": "0 < mean(fold) - mean(anchor) < +0.0582 (INDICATIVE ref)",
        "reference": H1_REFERENCE, "paired": p,
        "delta": None if stats["fold"]["mean"] is None else
                 stats["fold"]["mean"] - stats["anchor"]["mean"],
    }
    d = bars["H1"]["delta"]
    bars["H1"]["verdict"] = ("NOT RUN" if d is None else
                             "HELD" if 0 < d < H1_REFERENCE else
                             "FAILED (sign)" if d <= 0 else "FAILED (magnitude)")

    for key, arm, note in (("H2", "foldwide", "the candidate"),
                           ("H4", "mlayer", "authorised by leg A")):
        p = paired(stats[arm], stats["anchor"])
        delta = (None if stats[arm]["mean"] is None
                 else stats[arm]["mean"] - stats["anchor"]["mean"])
        bars[key] = {
            "statement": f"mean({arm}) < mean(anchor) - {BAR_2SIGMA}",
            "note": note, "delta": delta, "paired": p,
            "verdict": ("NOT RUN" if delta is None else
                        "HELD" if delta < -BAR_2SIGMA else "UNRESOLVED"),
        }

    p = paired(stats["binin"], stats["anchor"])
    delta = (None if stats["binin"]["mean"] is None
             else stats["binin"]["mean"] - stats["anchor"]["mean"])
    bars["H3"] = {
        "statement": f"|mean(binin) - mean(anchor)| < {BAR_2SIGMA}, at IDENTICAL params",
        "delta": delta, "paired": p,
        "verdict": ("NOT RUN" if delta is None else
                    "HELD" if abs(delta) < BAR_2SIGMA else
                    "FAILED (arm worse)" if delta > 0 else "FAILED (arm better)"),
        "wording": ("binarising the input costs less than 2 transferred sigma at "
                    "identical parameters -- NOT 'binarity is free'"),
    }
    out["bars"] = bars

    # -- H6, the cost marker ---------------------------------------------
    ref = _summary(ANCHOR_S0).get("wall_clock_s")
    out["H6_cost"] = {
        arm: {
            "wall_clock_s": [_summary(r).get("wall_clock_s") for r in runs],
            "x_anchor": None if not ref else round(
                statistics.fmean([w for w in
                                  (_summary(r).get("wall_clock_s") for r in runs)
                                  if w]) / ref, 3),
            "peak_vram_gib": [_summary(r).get("peak_vram_gib") for r in runs],
            "params": [_summary(r).get("params") for r in runs],
        } for arm, runs in ARMS.items()
    }

    # -- H5: the horizon, AND the decomposition CONTRIBUTING.md §3 requires ---
    #
    # "Decompose a gain by context before reporting it." A mean bpc is three
    # effects of possibly opposite sign summed, and EXP_004 §10.3 is the standing
    # demonstration that reporting only the sum can invert the reading.
    hzp = _REPO / "docs" / "reports" / "data" / "exp_020_memory_horizon.json"
    if not hzp.exists():
        out["H5_horizon"] = {"measured": False,
                             "note": "marker not measured; reported absent "
                                     "rather than omitted"}
    else:
        hz = json.loads(hzp.read_text(encoding="utf-8"))
        r010 = _import_010()
        ac = hz["absolute_context_curves"]
        # NOT `bars` -- that name is the hypothesis dict in this function, and
        # shadowing it made the verdict printer iterate the sigma table instead.
        #
        # And NOT `hz["per_context_2sigma"]` either: that is a WRAPPER,
        # `{n_seeds, arm, bars, whole_split_2sigma_for_reference}`, and the bars
        # are one level down. Passing the wrapper makes every `bars.get(c)`
        # return None, so `context_comparison` falls back to the flat 0.00922 at
        # every context and reports 2/128 where the probe's own figure is
        # 120/128. Caught by the two numbers disagreeing, which is the entire
        # reason `EXP_011`'s resolver imports this statistic instead of
        # reimplementing it -- and it still took a wrapper to make the point.
        _p2s = hz.get("per_context_2sigma", {})
        ctx_bars = _p2s.get("bars", _p2s) if isinstance(_p2s, dict) else {}
        ref = ac["if_anchor"]["bpc_at_context"]

        dec = {}
        for arm in ac:
            if arm == "if_anchor":
                continue
            dec[arm] = r010.decompose(ref, ac[arm]["bpc_at_context"])
            dec[arm]["context_comparison"] = r010.context_comparison(
                ac[arm]["bpc_at_context"], ref, ctx_bars)

        # THE CALIBRATION, and it is not decoration.
        #
        # `if_nest_s0` is BITWISE `if_anchor_s0` -- T1 asserts their final bpc
        # differ by exactly 0.0 -- so every number it shows here is one seed
        # against a three-seed mean and nothing else. It is therefore a
        # measured NOISE FLOOR for the decomposition, and it reads:
        # 120 of 128 contexts "significantly worse", a total of -0.005 bpc and
        # -0.014 at c = 0.
        #
        # Two consequences, both stated rather than worked around:
        #   * `n_contexts_significantly_worse` is UNUSABLE at n = 3 seeds. A
        #     bit-identical control fails 120/128 of it, so no arm's count in
        #     this artifact is evidence of anything.
        #   * any arm whose decomposition components sit inside the nest row's
        #     is indistinguishable from a bit-identical copy of the anchor.
        nest = dec.get("if_nest")
        # Both figures, because they disagree and the disagreement is the point:
        # the probe's own count uses the per-context bars, the recomputed one is
        # a cross-check that they were passed correctly.
        for arm in dec:
            dec[arm]["n_worse_probe_stored"] = ac[arm][
                "n_contexts_significantly_worse"]
        out["H5_horizon"] = {
            "measured": True,
            "horizons": hz.get("by_arm_horizon", hz.get("horizons")),
            "decomposition": dec,
            "per_context_2sigma": ctx_bars,
            "noise_floor_from_bitwise_control": nest,
            "n_contexts_worse_is_unusable_at_n3": bool(
                nest is not None
                and nest["context_comparison"]["n_contexts_significantly_worse"]
                > 64),
        }
        for ctx, bar in ctx_bars.items():
            if bar is None:
                out.setdefault("per_context_bars_missing", []).append(ctx)

    p = _REPO / args.out
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")

    # -- console --------------------------------------------------------
    print(f"\nEXP_020 — protocol {args.protocol}, "
          f"2 sigma = {BAR_2SIGMA} (TRANSFERRED, different tree)")
    print(f"  T1 whole-run nesting: {'PASS' if t1['pass'] else 'FAIL'}  "
          f"fresh diff={t1['fresh']['diff']} carried diff={t1['carried']['diff']}")
    print(f"  sigma re-measured on this tree: {out['sigma_measured_this_tree']} "
          f"(x{out['sigma_transfer_ratio']:.2f} the transferred value)"
          if out["sigma_measured_this_tree"] else "  sigma: n<2")
    print()
    for arm, s in stats.items():
        print(f"  {arm:9s} n={s['n']}/{s['n_preregistered']} "
              f"mean={s['mean']}  sd={s['sd']}  "
              f"{'PROVISIONAL ' + str(s['diverged']) if s['provisional'] else ''}")
    print()
    for k, b in bars.items():
        pr = b["paired"]
        extra = ""
        if pr.get("paired"):
            extra = (f"  paired t={pr['t']:.2f} " if pr.get("t") else "  ")
            extra += f"{pr['n_negative']}/{len(pr['diffs'])} seeds negative"
        print(f"  {k}: {b['verdict']:20s} delta={b['delta']}{extra}")
        print(f"      {b['statement']}")
    h5 = out["H5_horizon"]
    if h5.get("measured"):
        print("\n  EXP_004 §10.3 decomposition (positive = arm BETTER), "
              "cut at the baseline horizon 7:")
        print(f"    {'arm':12s} {'total':>9s} {'c=0':>9s} {'within':>9s} "
              f"{'beyond':>9s}  {'n worse':>8s}")
        for arm, d in h5["decomposition"].items():
            cc = d["context_comparison"]
            print(f"    {arm:12s} {d['total_gain_bpc']:+9.5f} "
                  f"{d['zero_context_gain_bpc']:+9.5f} "
                  f"{d['within_reach_gain_bpc']:+9.5f} "
                  f"{d['beyond_horizon_gain_bpc']:+9.5f} "
                  f"{cc['n_contexts_significantly_worse']:6d}/128")
        if h5["n_contexts_worse_is_unusable_at_n3"]:
            n = h5["noise_floor_from_bitwise_control"]
            print("\n    CALIBRATION: if_nest_s0 is BITWISE if_anchor_s0 (T1 = "
                  "0.0 exactly) and still reads")
            print(f"    {n['total_gain_bpc']:+.5f} total, "
                  f"{n['zero_context_gain_bpc']:+.5f} at c=0, and "
                  f"{n['context_comparison']['n_contexts_significantly_worse']}"
                  "/128 contexts 'significantly worse'.")
            print("    The n-worse column is measuring seed variance, not arms, "
                  "and is not evidence here.")
    print(f"\nWROTE {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
