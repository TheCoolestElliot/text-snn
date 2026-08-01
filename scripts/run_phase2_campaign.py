"""Phase-2 campaign: the seed-variance noise floor, the controls, and the beta check.

Runs sequentially -- there is one GPU, and concurrent runs would contaminate every
wall-clock and VRAM figure the report quotes.

Three groups, in priority order:

  A  EXP_000 seed variance. 5 identical configs differing only in `seed`. This is
     the noise floor; until it exists, no comparison in this project means
     anything (01_reconnaissance.md S8.3, risk R8).

  B  Controls at matched parameter count, 3 seeds each.
       analogue  identical architecture, continuous emission -- isolates I1+I3
       gru       external anchor, deliberately violates I5

  C  Baseline establishment: beta in {0.9, 0.95} at 3 seeds. NOT an ablation.
     Its only job is to show whether the reference beta=0.5 is or is not
     accidentally crippling the arm, as declared in advance in
     experiments/logs/EXP_000_seed_variance.md S6.

Results land in experiments/runs/<name>/summary.json; this script collects them
into docs/reports/data/phase2_campaign.json, which is what the report quotes.
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RUNS = REPO / "experiments" / "runs"

COMMON = [
    "--corpus", "enwik8",
    "--d-model", "512",
    "--n-layers", "2",
    "--batch-size", "128",
    "--seq-len", "256",
    "--threshold", "1.0",
    "--reset", "hard",
    "--surrogate-alpha", "2.0",
    "--t-steps", "1",
    "--dtype", "fp32",
    "--max-steps", "20000",
    "--log-every", "250",
    "--eval-every", "2500",
    "--eval-max-windows", "2560",
    "--ckpt-every", "5000",
]


def plan():
    jobs = []
    # A -- seed variance, the headline
    for seed in range(5):
        jobs.append(("A_seed_variance", f"snn_beta0.5_s{seed}",
                     ["--arch", "snn", "--beta", "0.5", "--seed", str(seed)]))
    # B -- controls
    for seed in range(3):
        jobs.append(("B_control_analogue", f"analogue_beta0.5_s{seed}",
                     ["--arch", "analogue", "--beta", "0.5", "--seed", str(seed)]))
    for seed in range(3):
        jobs.append(("B_control_gru", f"gru_s{seed}",
                     ["--arch", "gru", "--beta", "0.5", "--seed", str(seed)]))
    # C -- baseline establishment, explicitly not an ablation
    for beta in ("0.9", "0.95"):
        for seed in range(3):
            jobs.append((f"C_beta_check_{beta}", f"snn_beta{beta}_s{seed}",
                         ["--arch", "snn", "--beta", beta, "--seed", str(seed)]))
    return jobs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--only", default="", help="substring filter on the group name")
    ap.add_argument("--out", default="docs/reports/data/phase2_campaign.json")
    args = ap.parse_args()

    jobs = [j for j in plan() if args.only in j[0]]
    print(f"{len(jobs)} runs planned\n")
    for group, name, extra in jobs:
        print(f"  {group:22s} {name}")
    if args.dry_run:
        return

    results = {}
    t_campaign = time.perf_counter()
    for i, (group, name, extra) in enumerate(jobs, 1):
        cmd = [sys.executable, str(REPO / "scripts" / "train.py"),
               *COMMON, *extra, "--run-name", name]
        print(f"\n[{i}/{len(jobs)}] {group} :: {name}", flush=True)
        t0 = time.perf_counter()
        proc = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True)
        dt = time.perf_counter() - t0
        if proc.returncode != 0:
            # A failed run is data, not an excuse to stop: record it and continue,
            # so one bad arm cannot cost the whole campaign.
            print(f"    FAILED rc={proc.returncode} after {dt:.0f}s")
            print("    " + "\n    ".join(proc.stderr.strip().splitlines()[-12:]))
            results.setdefault(group, {})[name] = {
                "status": "FAILED", "returncode": proc.returncode,
                "stderr_tail": proc.stderr.strip().splitlines()[-25:],
                "wall_clock_s": round(dt, 1),
            }
            continue

        summary_path = RUNS / name / "summary.json"
        summary = (json.loads(summary_path.read_text(encoding="utf-8"))
                   if summary_path.exists() else {"status": "NO_SUMMARY"})
        summary["wall_clock_s"] = round(dt, 1)
        summary["status"] = summary.get("status", "OK")
        results.setdefault(group, {})[name] = summary

        tail = [l for l in proc.stdout.strip().splitlines() if l.strip()][-3:]
        print(f"    done in {dt:.0f}s")
        for l in tail:
            print("    " + l)

        out = REPO / args.out
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print(f"\ncampaign wall-clock {time.perf_counter() - t_campaign:.0f}s")
    print(f"WROTE {args.out}")


if __name__ == "__main__":
    main()
