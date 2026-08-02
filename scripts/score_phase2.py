"""Final scoring pass: full val AND test, both protocols, every campaign run.

Why this is a separate pass rather than part of training. `Trainer` writes a
`summary.json` whose `val` block is a *subsampled* evaluation (`eval_max_windows`),
because scoring the whole split every `eval_every` steps would cost more than the
training it is monitoring. A monitoring number is not a headline number, and the
two must not be confused -- so the headline is produced here, once, over the
complete split, from the final checkpoint.

The test split is scored ONLY here, and only once per run, after every design
decision in Phase 2 was already fixed. That ordering is the point.

Emits docs/reports/data/phase2_final_scores.json, which is what the report quotes,
and prints the seed-variance statistics that EXP_000's decision rule needs.
"""

import argparse
import json
import math
import statistics
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RUNS = REPO / "experiments" / "runs"


def score(ckpt: Path, split: str) -> dict:
    """Full-split, both-protocol score for one checkpoint."""
    out_path = ckpt.parent / f"final_{split}.json"
    cmd = [sys.executable, str(REPO / "scripts" / "evaluate.py"),
           "--ckpt", str(ckpt), "--split", split,
           "--max-windows", "0", "--out", str(out_path)]
    proc = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True)
    if proc.returncode != 0:
        return {"status": "FAILED",
                "stderr_tail": proc.stderr.strip().splitlines()[-15:]}
    return json.loads(out_path.read_text(encoding="utf-8"))


def group_of(name: str) -> str:
    if name.startswith("snn_beta0.5_"):
        return "A_seed_variance"
    if name.startswith("analogue_"):
        return "B_control_analogue"
    if name.startswith("gru_"):
        return "B_control_gru"
    if name.startswith("snn_beta0.9_"):
        return "C_beta_check_0.9"
    if name.startswith("snn_beta0.95_"):
        return "C_beta_check_0.95"
    return "other"


def bpc_of(block: dict, protocol: str) -> float | None:
    """Pull bpc out of whatever shape scripts/evaluate.py emitted."""
    if not isinstance(block, dict):
        return None
    if protocol in block and isinstance(block[protocol], dict):
        return block[protocol].get("bpc")
    for key in ("results", "protocols", "scores"):
        sub = block.get(key)
        if isinstance(sub, dict) and protocol in sub:
            return sub[protocol].get("bpc")
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs/reports/data/phase2_final_scores.json")
    ap.add_argument("--ckpt-name", default="ckpt_final.pt")
    args = ap.parse_args()

    runs = sorted(p for p in RUNS.iterdir() if p.is_dir())
    results: dict[str, dict] = {}

    for run_dir in runs:
        ckpt = run_dir / args.ckpt_name
        if not ckpt.exists():
            cands = sorted(run_dir.glob("ckpt_*.pt"))
            if not cands:
                print(f"  {run_dir.name:24s} no checkpoint, skipped")
                continue
            ckpt = cands[-1]
        print(f"  {run_dir.name:24s} scoring val + test ...", flush=True)
        entry = {"run": run_dir.name, "group": group_of(run_dir.name),
                 "ckpt": ckpt.name}
        for split in ("val", "test"):
            entry[split] = score(ckpt, split)
        summary_path = run_dir / "summary.json"
        if summary_path.exists():
            s = json.loads(summary_path.read_text(encoding="utf-8"))
            entry["params"] = s.get("params")
            entry["wall_clock_s"] = s.get("wall_clock_s")
            entry["peak_vram_gib"] = s.get("peak_vram_gib")
            entry["steps"] = s.get("steps")
        results[run_dir.name] = entry
        for split in ("val", "test"):
            f, c = bpc_of(entry[split], "fresh"), bpc_of(entry[split], "carried")
            fs = f"{f:.4f}" if isinstance(f, float) else str(f)
            cs = f"{c:.4f}" if isinstance(c, float) else str(c)
            print(f"      {split:5s} fresh {fs}  carried {cs}")

        out = REPO / args.out
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(results, indent=2), encoding="utf-8")

    # ---- the statistic EXP_000 exists to produce --------------------------
    stats = {}
    by_group: dict[str, list] = {}
    for name, e in results.items():
        by_group.setdefault(e["group"], []).append(e)

    print("\n" + "=" * 78)
    print("PER-GROUP TEST bpc (the headline is the 'carried' column)")
    print("=" * 78)
    print(f"{'group':24s} {'n':>2s} {'fresh mean':>11s} {'sd':>8s} "
          f"{'carried mean':>13s} {'sd':>8s}")
    for group, entries in sorted(by_group.items()):
        row = {}
        for protocol in ("fresh", "carried"):
            vals = [bpc_of(e["test"], protocol) for e in entries]
            vals = [v for v in vals if isinstance(v, float) and not math.isnan(v)]
            if not vals:
                continue
            row[protocol] = {
                "n": len(vals),
                "values": [round(v, 6) for v in vals],
                "mean": round(statistics.fmean(vals), 6),
                "sd": round(statistics.stdev(vals), 6) if len(vals) > 1 else 0.0,
                "min": round(min(vals), 6),
                "max": round(max(vals), 6),
            }
        stats[group] = row
        f, c = row.get("fresh"), row.get("carried")
        if f and c:
            print(f"{group:24s} {f['n']:>2d} {f['mean']:>11.4f} {f['sd']:>8.4f} "
                  f"{c['mean']:>13.4f} {c['sd']:>8.4f}")

    sv = stats.get("A_seed_variance", {}).get("carried")
    if sv and sv["n"] > 1:
        sigma = sv["sd"]
        stats["noise_floor"] = {
            "sigma_bpc_carried": sigma,
            "n_seeds": sv["n"],
            "adoption_threshold_2sigma": round(2 * sigma, 6),
            "exceeds_0.05_replan_trigger": sigma > 0.05,
        }
        print("\n" + "=" * 78)
        print(f"NOISE FLOOR (EXP_000): sigma = {sigma:.5f} bpc over {sv['n']} seeds")
        print(f"  adoption threshold (2 sigma) = {2 * sigma:.5f} bpc")
        print(f"  EXP_000 5 replan trigger (sigma > 0.05): "
              f"{'TRIGGERED' if sigma > 0.05 else 'not triggered'}")
        print("=" * 78)

    results["_statistics"] = stats
    (REPO / args.out).write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nWROTE {args.out}")


if __name__ == "__main__":
    main()
