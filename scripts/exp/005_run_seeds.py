"""EXP_005: train and score four more seeds of the adopted two-compartment arm.

Pre-registered in `experiments/logs/EXP_005_sigma_transfer.md`. Read that first.
This file only executes §2's design; every choice it makes was fixed there.

WHY A DRIVER RATHER THAN FOUR LAUNCHES
--------------------------------------
§2 requires the runs to be **strictly sequential**, because the phase-3 report §8
declined to quote wall-clock on the depth ladder for exactly the reason that its
arms overlapped other GPU work. A driver that blocks on each child in turn is the
only way to make that a property of the code rather than of whoever typed the
commands. `scripts/launch.py` then detaches *this* process, so the whole chain
survives its parent going away (risk R6) while still running one run at a time.

WHAT IT GUARDS, AND WHY EACH GUARD IS HERE
------------------------------------------
K1 -- **the extra seeds must be the same arm.** Every new run's `config.json` is
compared field-by-field against `twocomp_s0`'s and only `seed` and `run_name` may
differ. "Four more seeds" is a claim about sameness, and an unnoticed config drift
would turn this experiment into a comparison of two different neurons reported as
a variance.

K2 -- **no mutation campaign while training.** Phase-3 report §8: a run that
started inside the campaign's window trained 7 750 steps against a mutated kernel
with nothing in its own logs to say so, because Python reads a module once at
import. The campaign leaves `src/snn/.MUTATION_CAMPAIGN_RUNNING` behind while
mutations are applied; this refuses to start, and refuses to start each
subsequent run, while that file exists.

K3 -- wall-clock is recorded per run and compared against `EXP_004`'s 525 s.

The F1 self-check (K2's other half) is not here: it belongs to the horizon probe,
which owns the `k = L` reproduction, and duplicating it would be a second
implementation of a check whose whole value is that there is only one.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
RUNS = _REPO / "experiments" / "runs"
LOCKFILE = _REPO / "src" / "snn" / ".MUTATION_CAMPAIGN_RUNNING"

REFERENCE_RUN = "twocomp_s0"
SEEDS = (3, 4, 5, 6)

#: The only two fields a new seed may differ from the reference in (K1).
MAY_DIFFER = {"seed", "run_name"}

#: EXP_004 §10.8's measured wall-clock for this arm, for the K3 comparison.
EXP_004_WALL_CLOCK_S = 525.3


def _check_no_campaign() -> None:
    if LOCKFILE.exists():
        raise SystemExit(
            f"K2: {LOCKFILE.relative_to(_REPO)} exists -- a mutation campaign is "
            "applying wrong kernels to the source tree. Refusing to start a "
            "training run that would import one (phase-3 report §8)."
        )


def _train_argv(seed: int) -> list[str]:
    """`twocomp_s0`'s launch command, with only the seed and the name changed.

    Taken from `experiments/runs/twocomp_s0/launch.json` rather than retyped, so
    that the arm is the adopted one by construction. K1 verifies the result.
    """
    return [
        "--arch", "twocomp",
        "--run_name", f"twocomp_s{seed}",
        "--seed", str(seed),
        "--beta_slow", "0.95",
        "--w_init", "0.1",
        "--eval_every", "2500",
        "--eval_max_windows", "2560",
        "--ckpt_every", "5000",
        "--log_every", "250",
    ]


def _run(argv: list[str], label: str) -> float:
    print(f"\n=== {label}\n    {' '.join(argv)}", flush=True)
    t0 = time.time()
    proc = subprocess.run([sys.executable, "-u", *argv], cwd=str(_REPO))
    dt = time.time() - t0
    if proc.returncode != 0:
        raise SystemExit(f"{label} failed with exit code {proc.returncode}")
    print(f"    done in {dt:.1f}s", flush=True)
    return dt


def check_same_arm(run: str, reference: dict) -> dict:
    """K1: only `seed` and `run_name` may differ from the reference config."""
    cfg = json.loads((RUNS / run / "config.json").read_text(encoding="utf-8"))
    diffs = {k: [reference.get(k), cfg.get(k)]
             for k in set(reference) | set(cfg)
             if reference.get(k) != cfg.get(k)}
    unexpected = {k: v for k, v in diffs.items() if k not in MAY_DIFFER}
    if unexpected:
        raise SystemExit(
            f"K1: {run} is not the same arm as {REFERENCE_RUN}. Fields that "
            f"differ but may not: {unexpected}"
        )
    return {"run": run, "differs_in": sorted(diffs), "same_arm": True}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", default=",".join(str(s) for s in SEEDS))
    ap.add_argument("--out", default="docs/reports/data/exp_005_run_manifest.json")
    args = ap.parse_args(argv)
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]

    reference = json.loads(
        (RUNS / REFERENCE_RUN / "config.json").read_text(encoding="utf-8"))
    started = time.time()
    manifest: dict = {
        "experiment": "EXP_005_sigma_transfer",
        "log": "experiments/logs/EXP_005_sigma_transfer.md",
        "reference_run": REFERENCE_RUN,
        "reference_wall_clock_s": EXP_004_WALL_CLOCK_S,
        "sequential": True,
        "seeds": seeds,
        "runs": [],
    }

    for seed in seeds:
        run = f"twocomp_s{seed}"
        _check_no_campaign()          # re-checked before EVERY run, not once
        t_train = _run(["scripts/train.py", *_train_argv(seed)], f"train {run}")
        _check_no_campaign()
        t_eval = _run([
            "scripts/evaluate.py",
            "--ckpt", str(RUNS / run / "ckpt_final.pt"),
            "--split", "test",
            "--out", str(RUNS / run / "final_test.json"),
        ], f"evaluate {run}")

        row = check_same_arm(run, reference)
        summary = json.loads((RUNS / run / "summary.json").read_text(encoding="utf-8"))
        test = json.loads((RUNS / run / "final_test.json").read_text(encoding="utf-8"))
        row.update({
            "seed": seed,
            "train_seconds": round(t_train, 1),
            "eval_seconds": round(t_eval, 1),
            "reported_wall_clock_s": summary.get("wall_clock_s"),
            "wall_clock_vs_exp_004": (
                round(summary["wall_clock_s"] / EXP_004_WALL_CLOCK_S, 3)
                if summary.get("wall_clock_s") else None),
            "peak_vram_gib": summary.get("peak_vram_gib"),
            "params": test.get("params"),
            "test_bpc": {p: test["results"][p]["bpc"] for p in ("fresh", "carried")},
        })
        manifest["runs"].append(row)
        print(f"    {run}: fresh {row['test_bpc']['fresh']:.4f}  "
              f"carried {row['test_bpc']['carried']:.4f}", flush=True)

        out = _REPO / args.out
        out.parent.mkdir(parents=True, exist_ok=True)
        manifest["total_seconds"] = round(time.time() - started, 1)
        out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("\n" + "=" * 70)
    print(f"EXP_005: {len(manifest['runs'])} seeds trained and scored "
          f"sequentially in {manifest['total_seconds']:.0f}s")
    for r in manifest["runs"]:
        print(f"  {r['run']:12s} carried {r['test_bpc']['carried']:.4f}   "
              f"{r['reported_wall_clock_s']:.0f}s "
              f"({r['wall_clock_vs_exp_004']:.2f}x EXP_004)")
    print(f"WROTE {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
