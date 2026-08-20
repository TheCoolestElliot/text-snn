"""EXP_021 resolver: applies §4's bars mechanically to §3's runs.

    python scripts/exp/021_neuromod_results.py --out docs/reports/data/exp_021_arm_results.json

COMMITTED BEFORE ANY LEG'S BPC IS READ.

Every bar here is on a **paired per-seed difference**, and that is not a style
choice: `EXP_018` D1 used an unpaired Welch test on paired data, resolved
UNRESOLVED by 3.6e-06 bpc, and the corrected paired test -- run because it hurt
-- found the arm cost 0.0062 bpc on 5 of 5 seeds. A seed fixes both the initial
weights and the data order, so arms sharing seeds are paired. The per-seed signs
are reported next to every mean.

WHAT IT REFUSES TO DO
---------------------
  * It does not adopt, rank, or recommend.
  * It does not repair a bar that misfired; a corrected rule is referred.
  * It does not substitute a seed.
  * It does not compare any number with a committed figure (decision #10).
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

import torch  # noqa: E402


def _import_010():
    """`decompose` and `context_comparison`, IMPORTED not reimplemented.

    `EXP_011`'s resolver set the precedent: one implementation of each
    statistic, not one file. `010_phase4_arm_results.py` is committed and closed
    -- three experiments' published numbers come out of it -- so it is imported
    by path rather than edited.
    """
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "_r010", _REPO / "scripts" / "exp" / "010_phase4_arm_results.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

RUNS = _REPO / "experiments" / "runs"
PREREG = _REPO / "experiments" / "logs" / "EXP_021_neuromodulation.md"
MANIFEST = _REPO / "docs" / "reports" / "data" / "exp_021_run_manifest.json"

SIGMA_TRANSFERRED = 0.00461
BAR_2SIGMA = 2 * SIGMA_TRANSFERRED

#: H3's threshold. Deliberately far below EXP_018's measured factor of 26 on the
#: analogous quantity: this arm's phi is a different signal at a different scale,
#: and §2.7's FAINT reachability ratio is a named risk that would surface here.
H3_RATIO = 3.0

ARMS = {
    "anchor":    [f"if_anchor_s{s}" for s in (0, 1, 2)],
    "nm_local":  [f"nm_local_s{s}" for s in (0, 1, 2)],
    "nm_rolled": [f"nm_rolled_s{s}" for s in (0, 1, 2)],
    "tf_neg":    [f"tf_neg_s{s}" for s in (0, 1, 2)],
    "tf_pos":    [f"tf_pos_s{s}" for s in (0, 1, 2)],
    "tf_roll":   [f"tf_roll_s{s}" for s in (0, 1, 2)],
}


def _final(run: str) -> dict | None:
    p = RUNS / run / "final_test.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


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


def _final_train_loss(run: str) -> float | None:
    """The last logged training loss, for H7's train->test gap.

    `EXP_018`'s clearest structural finding was that its arm bought no extra
    training fit while losing at test. The same quantity, computed the same way,
    is the only thing that makes "capability loss" distinguishable from
    "overfitting" without a second evaluation pass.
    """
    p = RUNS / run / "log.jsonl"
    if not p.exists():
        return None
    last = None
    for line in p.read_text(encoding="utf-8").splitlines():
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "loss" in rec and rec.get("loss") is not None:
            last = float(rec["loss"])
    return None if last is None else last / math.log(2.0)


def arm_stats(runs: list[str], protocol: str) -> dict:
    vals, dead = [], []
    for r in runs:
        if _diverged(r):
            dead.append(r)
            continue
        b = _bpc(r, protocol)
        if b is None:
            dead.append(r)
        else:
            vals.append(b)
    return {"runs": runs, "n": len(vals), "n_preregistered": len(runs),
            "diverged": dead, "provisional": len(vals) < len(runs),
            "bpc": vals,
            "mean": statistics.fmean(vals) if vals else None,
            "sd": statistics.stdev(vals) if len(vals) > 1 else None,
            "train_bpc": [_final_train_loss(r) for r in runs]}


def paired(a: dict, b: dict) -> dict:
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
            "n_negative": sum(1 for x in d if x < 0)}


def gain_rms(run: str) -> float | None:
    """rms of `nm_gain` from the committed checkpoint -- H3's quantity.

    EXP_018's reusable result was exactly this measurement on the analogous
    parameter: 0.195 aligned against 0.0075 rolled, a factor of 26. The
    checkpoint is read rather than the model rebuilt, so nothing here can
    disagree with what trained.
    """
    p = RUNS / run / "ckpt_final.pt"
    if not p.exists():
        return None
    ck = torch.load(p, map_location="cpu", weights_only=False)
    sd = ck.get("model", ck)
    vals = [v.float().pow(2).mean() for k, v in sd.items() if k.startswith("nm_gain")]
    if not vals:
        return None
    return float(torch.stack(vals).mean().sqrt())


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--protocol", default="carried", choices=("fresh", "carried"))
    ap.add_argument("--out", default="docs/reports/data/exp_021_arm_results.json")
    args = ap.parse_args(argv)

    out: dict = {
        "experiment": "EXP_021", "protocol": args.protocol,
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
        out["tau_check"] = man.get("tau_check")
        out["G2_all_params_match"] = all(
            r.get("G2_params_match", False) for r in man.get("runs", [])
            if "G2_params_match" in r)
        out["G5_violations"] = {r["run"]: r["G5_frozen_violations"]
                                for r in man.get("runs", [])
                                if r.get("G5_frozen_violations")}
        out["K1_unexpected"] = {r["run"]: r["K1_unexpected"]
                                for r in man.get("runs", [])
                                if r.get("K1_unexpected")}

    stats = {k: arm_stats(v, args.protocol) for k, v in ARMS.items()}
    out["arms"] = stats

    bars: dict = {}

    def _one_sided(key, arm, ref, statement, note=""):
        p = paired(stats[arm], stats[ref])
        delta = (None if stats[arm]["mean"] is None or stats[ref]["mean"] is None
                 else stats[arm]["mean"] - stats[ref]["mean"])
        bars[key] = {"statement": statement, "note": note, "delta": delta,
                     "paired": p,
                     "verdict": ("NOT RUN" if delta is None else
                                 "HELD" if delta < -BAR_2SIGMA else "UNRESOLVED")}

    _one_sided("H1", "nm_local", "anchor",
               f"mean(nm_local) < mean(anchor) - {BAR_2SIGMA}",
               "predicts the OPPOSITE sign from EXP_018's nearest precedent")
    _one_sided("H2", "nm_local", "nm_rolled",
               f"mean(nm_local) < mean(nm_rolled) - {BAR_2SIGMA}",
               "the bar that matters most: mechanism vs perturbation")
    _one_sided("H6", "tf_neg", "tf_pos",
               f"mean(tf_neg) < mean(tf_pos) - {BAR_2SIGMA}",
               "the sign of dopaminergic plasticity gating")

    # H4 predicts NO improvement, and is FAILED by any improvement beyond 2 sigma.
    p = paired(stats["tf_neg"], stats["anchor"])
    delta = (None if stats["tf_neg"]["mean"] is None
             else stats["tf_neg"]["mean"] - stats["anchor"]["mean"])
    bars["H4"] = {
        "statement": f"mean(tf_neg) is NOT below mean(anchor) - {BAR_2SIGMA}",
        "note": "a prediction of no improvement, falsified by any improvement "
                "exceeding 2 transferred sigma",
        "delta": delta, "paired": p,
        "verdict": ("NOT RUN" if delta is None else
                    "FAILED (the arm improved)" if delta < -BAR_2SIGMA
                    else "HELD"),
    }

    # H5 is two-sided: alignment must MATTER, in either direction.
    p = paired(stats["tf_neg"], stats["tf_roll"])
    delta = (None if stats["tf_neg"]["mean"] is None or
             stats["tf_roll"]["mean"] is None
             else stats["tf_neg"]["mean"] - stats["tf_roll"]["mean"])
    bars["H5"] = {
        "statement": f"|mean(tf_neg) - mean(tf_roll)| > {BAR_2SIGMA}",
        "note": "if aligned and misaligned weightings are indistinguishable, "
                "the RPE is decorative",
        "delta": delta, "paired": p,
        "verdict": ("NOT RUN" if delta is None else
                    "HELD" if abs(delta) > BAR_2SIGMA else "UNRESOLVED"),
    }

    # H3: is the signal findable at all?
    g_local = [gain_rms(r) for r in ARMS["nm_local"]]
    g_roll = [gain_rms(r) for r in ARMS["nm_rolled"]]
    gl = [v for v in g_local if v is not None]
    gr = [v for v in g_roll if v is not None]
    ratio = (statistics.fmean(gl) / statistics.fmean(gr)) if gl and gr and \
        statistics.fmean(gr) > 0 else None
    bars["H3"] = {
        "statement": f"rms(nm_gain) for nm_local >= {H3_RATIO}x that for nm_rolled",
        "note": "MARKER with a threshold, no verdict. EXP_018 measured 26x on "
                "the analogous parameter.",
        "rms_local": g_local, "rms_rolled": g_roll, "ratio": ratio,
        "marker": ("NOT RUN" if ratio is None else
                   "above threshold" if ratio >= H3_RATIO else
                   "BELOW threshold -- the optimiser could not distinguish the "
                   "aligned signal from its control"),
    }
    out["bars"] = bars

    # -- H7: the train->test gap, EXP_018's structural quantity ----------
    out["H7_train_test_gap"] = {
        arm: {
            "test_bpc_mean": s["mean"],
            "train_bpc_last": s["train_bpc"],
            "gap": (None if s["mean"] is None or
                    any(v is None for v in s["train_bpc"])
                    else s["mean"] - statistics.fmean(s["train_bpc"])),
        } for arm, s in stats.items()
    }

    ref_w = _summary(ARMS["anchor"][0]).get("wall_clock_s")
    out["H8_cost"] = {
        arm: {"wall_clock_s": [_summary(r).get("wall_clock_s") for r in runs],
              "x_anchor": None if not ref_w else round(
                  statistics.fmean([w for w in
                                    (_summary(r).get("wall_clock_s") for r in runs)
                                    if w]) / ref_w, 3),
              "peak_vram_gib": [_summary(r).get("peak_vram_gib") for r in runs],
              "params": [_summary(r).get("params") for r in runs]}
        for arm, runs in ARMS.items()
    }

    # -- the decomposition CONTRIBUTING.md §3 requires before a gain is quoted --
    hzp = _REPO / "docs" / "reports" / "data" / "exp_021_memory_horizon.json"
    if not hzp.exists():
        out["decomposition"] = {"measured": False}
    else:
        hz = json.loads(hzp.read_text(encoding="utf-8"))
        r010 = _import_010()
        ac = hz["absolute_context_curves"]
        _p2s = hz.get("per_context_2sigma", {})
        # NOTE the wrapper: per_context_2sigma is
        # {n_seeds, arm, bars, whole_split_2sigma_for_reference} and the bars are
        # one level down. Passing the wrapper silently falls back to the flat bar
        # at every context. EXP_020 §9.8 item 3 is that defect, caught there.
        ctx_bars = _p2s.get("bars", _p2s) if isinstance(_p2s, dict) else {}
        ref = ac["if_anchor"]["bpc_at_context"]
        dec = {}
        for arm in ac:
            if arm == "if_anchor":
                continue
            dec[arm] = r010.decompose(ref, ac[arm]["bpc_at_context"])
            dec[arm]["context_comparison"] = r010.context_comparison(
                ac[arm]["bpc_at_context"], ref, ctx_bars)
            dec[arm]["n_worse_probe_stored"] = ac[arm][
                "n_contexts_significantly_worse"]
        out["decomposition"] = {"measured": True, "arms": dec,
                                "note": "EXP_020 §9.5 measured this statistic's "
                                        "noise floor with a BITWISE copy of the "
                                        "anchor: -0.005 total, -0.014 at c=0, "
                                        "120/128 contexts 'worse'. Read every "
                                        "row against that."}
        print("\n  EXP_004 §10.3 decomposition (positive = arm BETTER), "
              "cut at the baseline horizon 7:")
        print(f"    {'arm':11s} {'total':>9s} {'c=0':>9s} {'within':>9s} "
              f"{'beyond':>9s}")
        for arm, d in dec.items():
            print(f"    {arm:11s} {d['total_gain_bpc']:+9.5f} "
                  f"{d['zero_context_gain_bpc']:+9.5f} "
                  f"{d['within_reach_gain_bpc']:+9.5f} "
                  f"{d['beyond_horizon_gain_bpc']:+9.5f}")
        print("    (EXP_020's bitwise-identical control reads -0.00499 / "
              "-0.01395 / +0.00944 / -0.00047)")

    p = _REPO / args.out
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")

    print(f"\nEXP_021 — protocol {args.protocol}, "
          f"2 sigma = {BAR_2SIGMA} (TRANSFERRED, different tree)\n")
    for arm, s in stats.items():
        print(f"  {arm:10s} n={s['n']}/{s['n_preregistered']} mean={s['mean']} "
              f"sd={s['sd']} "
              f"{'PROVISIONAL ' + str(s['diverged']) if s['provisional'] else ''}")
    print()
    for k in ("H1", "H2", "H4", "H5", "H6"):
        b = bars[k]
        pr = b["paired"]
        extra = ""
        if pr.get("paired") and pr.get("t") is not None:
            extra = (f"  paired t={pr['t']:.2f}  "
                     f"{pr['n_negative']}/{len(pr['diffs'])} seeds negative")
        print(f"  {k}: {b['verdict']:26s} delta={b['delta']}{extra}")
        print(f"      {b['statement']}")
    print(f"\n  H3: {bars['H3']['marker']}  ratio={bars['H3']['ratio']}")
    print(f"\nWROTE {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
