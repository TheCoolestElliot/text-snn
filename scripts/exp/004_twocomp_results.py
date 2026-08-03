"""EXP_004 results: resolve T1-T5 against the pre-registration, mechanically.

Everything this script decides was fixed in
`experiments/logs/EXP_004_two_compartment.md` before the arm was trained. It reads
committed artifacts and applies the stated thresholds; it does not choose them.
That separation is the point -- a script that both picks the bar and reports the
score is a script that can rationalise.

Inputs, all produced by other tools:
  * `experiments/runs/twocomp_s*/final_test.json`  -- scripts/evaluate.py
  * `docs/reports/data/exp_004_memory_horizon.json` -- scripts/exp/001_memory_horizon.py
  * `experiments/runs/twocomp_s*/ckpt_final.pt`     -- the trainer (T4 reads these)
  * `docs/reports/data/phase2_final_scores.json`    -- the committed baseline

The baseline figures are RECOMPUTED from the five committed Phase-2 seeds rather
than pasted from the report, and the recomputation is asserted against the report's
quoted 2.2697 / 2.2531. That is the same self-check discipline as `EXP_001`'s F1:
a number this script did not produce has to come out of it.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO / "src") not in sys.path:
    sys.path.insert(0, str(_REPO / "src"))

# ---- everything below was fixed in the pre-registration -------------------
SEEDS = (0, 1, 2)
ARM = "twocomp"
BASELINE_ARM = "snn_beta0.5"
BASELINE_RUNS = [f"snn_beta0.5_s{i}" for i in range(5)]

TWO_SIGMA = 0.00922            # EXP_000 whole-split noise floor, x2
BASELINE_HORIZON = 7           # EXP_001, five seeds agreeing exactly
T1_HORIZON_BAR = 14            # EXP_004 T1: >= 2x the baseline, == K=8's reach
T4_MIX_GROWTH = 2.0            # EXP_004 T4: |w| must at least double
T5_CEILING_BPC = 0.2279        # §5's beyond-horizon component -- the whole ceiling

# The report's quoted 5-seed means; recomputed and asserted, never trusted.
REPORTED_FRESH = 2.2697
REPORTED_CARRIED = 2.2531
REPORT_TOL = 5e-4


def _mean_sd(xs: list[float]) -> tuple[float, float]:
    n = len(xs)
    m = sum(xs) / n
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1)) if n > 1 else 0.0
    return m, sd


def _test_bpc(run: str) -> dict[str, float]:
    path = _REPO / "experiments" / "runs" / run / "final_test.json"
    if not path.exists():
        raise SystemExit(f"{path} missing -- run scripts/evaluate.py on {run} first")
    blob = json.loads(path.read_text(encoding="utf-8"))
    res = blob.get("results", blob)
    return {p: res[p]["bpc"] for p in ("fresh", "carried")}


def baseline_reference() -> dict:
    """The five committed Phase-2 seeds, recomputed and checked against the report."""
    blob = json.loads(
        (_REPO / "docs/reports/data/phase2_final_scores.json").read_text("utf-8"))
    out: dict = {"runs": BASELINE_RUNS}
    for protocol in ("fresh", "carried"):
        vals = [blob[r]["test"]["results"][protocol]["bpc"] for r in BASELINE_RUNS]
        m, sd = _mean_sd(vals)
        out[protocol] = {"values": vals, "mean": m, "sd": sd, "n": len(vals)}
    out["reproduces_report"] = {
        "fresh": abs(out["fresh"]["mean"] - REPORTED_FRESH) < REPORT_TOL,
        "carried": abs(out["carried"]["mean"] - REPORTED_CARRIED) < REPORT_TOL,
        "fresh_residual": out["fresh"]["mean"] - REPORTED_FRESH,
        "carried_residual": out["carried"]["mean"] - REPORTED_CARRIED,
    }
    return out


def learned_parameters(run: str) -> dict:
    """T4: did the arm actually use its slow compartment?

    `w` at initialisation is a config field, so "moved off its init" is measured
    against the value the run was launched with rather than against an assumption
    about it. `beta_s` is reported as the realised sigmoid, which is the quantity
    the neuron uses -- reporting the raw parameter would be reporting a number no
    equation in the project contains.
    """
    ck = torch.load(_REPO / "experiments" / "runs" / run / "ckpt_final.pt",
                    map_location="cpu", weights_only=False)
    cfg = ck["config"]
    sd = ck["model"]
    w_init = float(cfg["w_init"])
    beta_init = float(cfg["beta_slow"])
    layers = []
    k = 0
    while f"w.{k}" in sd:
        w = sd[f"w.{k}"].flatten().float()
        bs = torch.sigmoid(sd[f"beta_s_raw.{k}"].flatten().float())
        layers.append({
            "layer": k,
            "w_abs_mean": float(w.abs().mean()),
            "w_mean": float(w.mean()),
            "w_sd": float(w.std()),
            "w_min": float(w.min()), "w_max": float(w.max()),
            "w_growth_vs_init": (float(w.abs().mean()) / abs(w_init)
                                 if w_init != 0 else None),
            "beta_s_mean": float(bs.mean()),
            "beta_s_sd": float(bs.std()),
            "beta_s_min": float(bs.min()), "beta_s_max": float(bs.max()),
            "beta_s_moved_from_init": float((bs - beta_init).abs().mean()),
            # 1/(1-beta) -- the slow pole's time constant in characters, which is
            # the quantity the horizon is supposed to be about.
            "slow_tau_chars_mean": float((1.0 / (1.0 - bs)).mean()),
        })
        k += 1
    return {"w_init": w_init, "beta_slow_init": beta_init,
            "step": ck.get("step"), "layers": layers}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--horizon",
                    default="docs/reports/data/exp_004_memory_horizon.json")
    ap.add_argument("--out", default="docs/reports/data/exp_004_twocomp_results.json")
    args = ap.parse_args(argv)

    runs = [f"twocomp_s{s}" for s in SEEDS]
    base = baseline_reference()
    if not all(base["reproduces_report"][p] for p in ("fresh", "carried")):
        raise SystemExit(
            "the recomputed Phase-2 baseline does not reproduce the report's "
            f"quoted means: {base['reproduces_report']}. Something has moved; "
            "chase it rather than widening the tolerance.")

    # ---- bpc ---------------------------------------------------------------
    per_seed = {r: _test_bpc(r) for r in runs}
    bpc: dict = {}
    for protocol in ("fresh", "carried"):
        vals = [per_seed[r][protocol] for r in runs]
        m, sd = _mean_sd(vals)
        delta = m - base[protocol]["mean"]
        bpc[protocol] = {
            "values": vals, "mean": m, "sd": sd, "n": len(vals),
            "baseline_mean": base[protocol]["mean"],
            "delta_vs_baseline": delta,
            "improves_by_more_than_2sigma": delta < -TWO_SIGMA,
            "sigma_units": delta / (TWO_SIGMA / 2.0),
        }

    # ---- horizon and the §6.2 curve ----------------------------------------
    hz = json.loads((_REPO / args.horizon).read_text(encoding="utf-8"))
    arm = hz["by_arm"][ARM]["_paired"]
    horizons = [h for h in arm["horizon_2sigma_per_seed"]]
    finite = [h for h in horizons if h is not None]
    median = sorted(finite)[len(finite) // 2] if finite else None
    dom = hz["absolute_context_curves"][ARM]
    bars = hz["per_context_2sigma"]["bars"]
    deltas = dom["delta_vs_baseline"]
    short_worse = sorted(
        ((int(c), deltas[c], bars.get(c, TWO_SIGMA))
         for c in deltas
         if int(c) <= 8 and deltas[c] > bars.get(c, TWO_SIGMA)),
        key=lambda t: -t[1])

    f1 = {r: hz["arms"][r].get("f1_pass") for r in runs if r in hz["arms"]}
    if not all(f1.values()):
        raise SystemExit(
            f"F1 self-check failed for {[r for r, ok in f1.items() if not ok]}: the "
            "probe's k=L point does not reproduce the independently committed "
            "fresh bpc. That is the check that caught a contaminated run in Phase 3 "
            "(report §8). Chase the residual; do not widen the tolerance.")

    # ---- T4 -----------------------------------------------------------------
    params = {r: learned_parameters(r) for r in runs}
    w_growth = [
        ly["w_growth_vs_init"]
        for r in runs for ly in params[r]["layers"]
        if ly["w_growth_vs_init"] is not None
    ]
    t4 = (bool(w_growth) and all(g >= T4_MIX_GROWTH for g in w_growth))

    # ---- resolve the predictions as written ---------------------------------
    t1 = median is not None and median >= T1_HORIZON_BAR
    t2 = not bpc["carried"]["improves_by_more_than_2sigma"]
    t3 = len(short_worse) >= 1
    gain = -bpc["carried"]["delta_vs_baseline"]
    t5 = gain < T5_CEILING_BPC

    if t1 and not t2:
        cell, verdict = "T1 holds, T2 falsified", "ADOPT"
    elif t1 and t2:
        cell, verdict = "T1 holds, T2 holds", "DO NOT ADOPT -- horizon is not sufficient"
    elif not t1 and not t2:
        cell, verdict = "T1 fails, T2 falsified", "GAIN IS NOT A MEMORY GAIN -- re-attribute"
    else:
        cell, verdict = "T1 fails, T2 holds", "NULL -- retain as a negative result"

    results = {
        "experiment": "EXP_004_two_compartment",
        "preregistration": "experiments/logs/EXP_004_two_compartment.md",
        "runs": runs,
        "baseline": base,
        "bpc": bpc,
        "horizon": {
            "per_seed": horizons, "median": median,
            "baseline": BASELINE_HORIZON, "bar": T1_HORIZON_BAR,
            "excess_bpc_at_context_mean": arm["excess_bpc_at_context_mean"],
            "f1_pass": f1,
        },
        "acceptance_criterion_6_2": {
            "n_contexts_significantly_worse": dom["n_contexts_significantly_worse"],
            "worst_delta_bpc": dom["worst_delta_bpc"],
            "worst_delta_at_context": dom["worst_delta_at_context"],
            "worst_short_context_delta_bpc": dom["worst_short_context_delta_bpc"],
            "worst_short_context_at": dom["worst_short_context_at"],
            "short_contexts_significantly_worse": [
                {"c": c, "delta_bpc": d, "bar_2sigma": b} for c, d, b in short_worse],
            "nowhere_worse_than_noise": dom["nowhere_worse_than_noise"],
            "bpc_at_context": dom["bpc_at_context"],
        },
        "learned_parameters": params,
        "predictions": {
            "T1": {"statement": f"median horizon over 3 seeds >= {T1_HORIZON_BAR}",
                   "measured": median, "held": t1},
            "T2": {"statement": "carried bpc does NOT improve by more than 2 sigma",
                   "measured_delta": bpc["carried"]["delta_vs_baseline"],
                   "held": t2,
                   "note": "stated in the direction that hurts the candidate"},
            "T3": {"statement": "worse at >= 1 short context (c <= 8) beyond its bar",
                   "measured": len(short_worse), "held": t3},
            "T4": {"statement": f"|w| grows to >= {T4_MIX_GROWTH}x its init in both layers",
                   "measured": w_growth, "held": t4},
            "T5": {"statement": f"any improvement is < {T5_CEILING_BPC} bpc",
                   "measured_gain": gain, "held": t5},
        },
        "decision": {"cell": cell, "verdict": verdict,
                     "rule": "EXP_004 §4, fixed before the arm was trained"},
    }

    out = _REPO / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")

    w = 78
    print("=" * w)
    print("EXP_004 -- two-compartment neuron, 3 seeds, resolved against §3")
    print("=" * w)
    print(f"  baseline reproduced from 5 committed seeds: "
          f"fresh {base['fresh']['mean']:.4f}  carried {base['carried']['mean']:.4f}")
    for p in ("fresh", "carried"):
        b = bpc[p]
        print(f"  test bpc {p:8s} {b['mean']:.4f} +/- {b['sd']:.4f}  "
              f"(baseline {b['baseline_mean']:.4f}, delta {b['delta_vs_baseline']:+.4f}"
              f" = {b['sigma_units']:+.1f} sigma)")
    print(f"  horizon per seed {horizons}  median {median}  "
          f"(baseline {BASELINE_HORIZON}, bar {T1_HORIZON_BAR})")
    a = results["acceptance_criterion_6_2"]
    print(f"  §6.2: worse at {a['n_contexts_significantly_worse']}/128 contexts; "
          f"worst short-c {a['worst_short_context_delta_bpc']:+.4f} at "
          f"c={a['worst_short_context_at']}")
    for r in runs:
        for ly in params[r]["layers"]:
            print(f"  {r} layer {ly['layer']}: |w| {ly['w_abs_mean']:.4f} "
                  f"(init {params[r]['w_init']}), beta_s {ly['beta_s_mean']:.4f} "
                  f"+/- {ly['beta_s_sd']:.4f}, tau {ly['slow_tau_chars_mean']:.1f} chars")
    print("-" * w)
    for pid, p in results["predictions"].items():
        print(f"  {pid}  {'HELD' if p['held'] else 'FAILED'}   {p['statement']}")
    print("-" * w)
    print(f"  {cell}  ->  {verdict}")
    print(f"\nWROTE {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
