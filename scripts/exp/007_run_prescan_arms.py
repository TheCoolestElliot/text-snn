"""EXP_007 and EXP_008: train and score the two pre-scan arms, strictly sequentially.

Pre-registered in `experiments/logs/EXP_007_token_shift.md` and
`experiments/logs/EXP_008_learned_threshold.md`. This file only executes their §2;
every choice it makes was fixed there.

WHY ONE DRIVER FOR TWO EXPERIMENTS
----------------------------------
They share a reference (the five committed `snn_beta0.5` seeds), a guard set, and
a machine. Interleaving them would be the thing `EXP_005`'s driver exists to
prevent -- overlapping GPU work makes every wall-clock number unquotable
(phase-3 report §8). One driver that blocks on each child in turn makes
"sequential" a property of the code rather than of whoever typed the commands.
The two arms are still two experiments with two pre-registrations and two
decision rules; nothing here mixes their verdicts.

WHAT IT GUARDS
--------------
K1 -- **each arm is the baseline plus one thing.** Every new run's `config.json`
is compared field-by-field against `snn_beta0.5_s0`'s, and only `seed`,
`run_name`, `arch` and the arm's own init field may differ. Fields the baseline's
config *predates* (it was written before `beta_slow`, `w_init`, `mu_init` and
`thr_log_init` existed) are excused only when the new run leaves them at the
`Config` default, and every such excusal is recorded in the manifest rather than
being silently swallowed. That distinction matters: "absent from an older config"
is a different fact from "set to something else", and collapsing them is how a
sameness check stops checking.

K2 -- **no mutation campaign while training**, re-checked before every run.
Phase-3 report §8: a run that started inside the campaign's window trained 7 750
steps against a mutated kernel with nothing in its own logs to say so.

K3 -- wall-clock recorded per run and compared against the Phase-2 baseline's
measured 398.9 s, since these arms are that arm plus one elementwise transform
per layer.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO / "src") not in sys.path:
    sys.path.insert(0, str(_REPO / "src"))

from snn.config import Config  # noqa: E402

RUNS = _REPO / "experiments" / "runs"
LOCKFILE = _REPO / "src" / "snn" / ".MUTATION_CAMPAIGN_RUNNING"

REFERENCE_RUN = "snn_beta0.5_s0"
SEEDS = (0, 1, 2)

#: `summary.json`'s wall_clock_s for the reference run, for the K3 comparison.
BASELINE_WALL_CLOCK_S = 398.88

#: arch -> (run-name prefix, the one init flag that may differ, its value)
ARMS = {
    "tokenshift": ("tokenshift", "mu_init", "1.0"),
    "threshold": ("threshold", "thr_log_init", "0.0"),
}

#: Fields that may differ from the reference for any arm (K1).
MAY_DIFFER = {"seed", "run_name", "arch"}

_DEFAULTS = {f.name: f.default for f in dataclasses.fields(Config)}


def is_complete(run: str) -> bool:
    """True iff `run` finished its full budget and was scored on test.

    Run-level idempotence, added 2026-08-03 **after the first launch was killed
    by a harness timeout mid-`tokenshift_s1`** and before any result was read.
    Recorded here rather than absorbed silently, because a driver that skips work
    is a driver that can skip work it should have done.

    The bar is deliberately both halves: `summary.json` is written only after the
    trainer's own loop completes `max_steps`, and `final_test.json` only after
    `scripts/evaluate.py` scores it. A directory holding a `ckpt_last.pt` from an
    interrupted run satisfies neither, and `main` deletes such a directory rather
    than resuming into it -- a resumed run is a different object from an
    uninterrupted one until `tests/test_determinism.py`'s guarantee is checked for
    this arm, and it has not been.
    """
    d = RUNS / run
    if not (d / "final_test.json").exists() or not (d / "summary.json").exists():
        return False
    summary = json.loads((d / "summary.json").read_text(encoding="utf-8"))
    cfg = json.loads((d / "config.json").read_text(encoding="utf-8"))
    return int(summary.get("steps", 0)) == int(cfg.get("max_steps", -1))


def _check_no_campaign() -> None:
    if LOCKFILE.exists():
        raise SystemExit(
            f"K2: {LOCKFILE.relative_to(_REPO)} exists -- a mutation campaign is "
            "applying wrong kernels to the source tree. Refusing to start a "
            "training run that would import one (phase-3 report §8)."
        )


def _train_argv(arch: str, seed: int) -> list[str]:
    """The reference run's launch command with the arch, the seed and the name
    changed, and nothing else. Everything not named here is a `Config` default and
    is verified as such by K1."""
    prefix, init_flag, init_value = ARMS[arch]
    return [
        "--arch", arch,
        "--run_name", f"{prefix}_s{seed}",
        "--seed", str(seed),
        f"--{init_flag}", init_value,
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


def check_same_arm(run: str, arch: str, reference: dict) -> dict:
    """K1. Returns the row; raises if anything differs that may not."""
    allowed = MAY_DIFFER | {ARMS[arch][1]}
    cfg = json.loads((RUNS / run / "config.json").read_text(encoding="utf-8"))
    diffs = {k: [reference.get(k), cfg.get(k)]
             for k in set(reference) | set(cfg)
             if reference.get(k) != cfg.get(k)}

    excused_as_default = []
    unexpected = {}
    for k, (ref_v, new_v) in diffs.items():
        if k in allowed:
            continue
        if ref_v is None and k in _DEFAULTS and new_v == _DEFAULTS[k]:
            # The reference config predates this field; the new run left it at
            # the Config default, so the two models agree on it.
            excused_as_default.append(k)
            continue
        unexpected[k] = [ref_v, new_v]
    if unexpected:
        raise SystemExit(
            f"K1: {run} is not {REFERENCE_RUN} plus one thing. Fields that "
            f"differ but may not: {unexpected}"
        )
    return {
        "run": run,
        "arch": arch,
        "differs_in": sorted(k for k in diffs if k in allowed),
        "absent_from_reference_and_left_at_default": sorted(excused_as_default),
        "same_arm": True,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--seeds", default=",".join(str(s) for s in SEEDS))
    ap.add_argument("--out", default="docs/reports/data/exp_007_008_run_manifest.json")
    args = ap.parse_args(argv)

    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    bad = [a for a in arms if a not in ARMS]
    if bad:
        raise SystemExit(f"unknown arm(s) {bad}; known: {sorted(ARMS)}")

    reference = json.loads(
        (RUNS / REFERENCE_RUN / "config.json").read_text(encoding="utf-8"))
    started = time.time()
    manifest: dict = {
        "experiments": ["EXP_007_token_shift", "EXP_008_learned_threshold"],
        "logs": ["experiments/logs/EXP_007_token_shift.md",
                 "experiments/logs/EXP_008_learned_threshold.md"],
        "reference_run": REFERENCE_RUN,
        "reference_wall_clock_s": BASELINE_WALL_CLOCK_S,
        "sequential": True,
        "arms": arms,
        "seeds": seeds,
        "runs": [],
    }

    for arch in arms:
        prefix = ARMS[arch][0]
        for seed in seeds:
            run = f"{prefix}_s{seed}"
            if is_complete(run):
                print(f"\n=== {run} already complete, skipping", flush=True)
                t_train = t_eval = 0.0
            else:
                if (RUNS / run).exists():
                    # A partial directory would append to log.jsonl and leave a
                    # ckpt_last.pt from a run that never finished, so the artifacts
                    # would describe two runs at once. Removed, not resumed.
                    print(f"\n=== {run} is partial; removing and retraining",
                          flush=True)
                    shutil.rmtree(RUNS / run)
                _check_no_campaign()      # before EVERY run, not once
                t_train = _run(["scripts/train.py", *_train_argv(arch, seed)],
                               f"train {run}")
                _check_no_campaign()
                t_eval = _run([
                    "scripts/evaluate.py",
                    "--ckpt", str(RUNS / run / "ckpt_final.pt"),
                    "--split", "test",
                    "--out", str(RUNS / run / "final_test.json"),
                ], f"evaluate {run}")

            row = check_same_arm(run, arch, reference)
            summary = json.loads(
                (RUNS / run / "summary.json").read_text(encoding="utf-8"))
            test = json.loads(
                (RUNS / run / "final_test.json").read_text(encoding="utf-8"))
            row.update({
                "seed": seed,
                "trained_in_this_invocation": t_train > 0,
                "train_seconds": round(t_train, 1),
                "eval_seconds": round(t_eval, 1),
                "reported_wall_clock_s": summary.get("wall_clock_s"),
                "wall_clock_vs_baseline": (
                    round(summary["wall_clock_s"] / BASELINE_WALL_CLOCK_S, 3)
                    if summary.get("wall_clock_s") else None),
                "peak_vram_gib": summary.get("peak_vram_gib"),
                "params": test.get("params"),
                "test_bpc": {p: test["results"][p]["bpc"]
                             for p in ("fresh", "carried")},
            })
            manifest["runs"].append(row)
            print(f"    {run}: fresh {row['test_bpc']['fresh']:.4f}  "
                  f"carried {row['test_bpc']['carried']:.4f}  "
                  f"params {row['params']:,}", flush=True)

            out = _REPO / args.out
            out.parent.mkdir(parents=True, exist_ok=True)
            manifest["total_seconds"] = round(time.time() - started, 1)
            out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("\n" + "=" * 70)
    print(f"EXP_007/008: {len(manifest['runs'])} runs trained and scored "
          f"sequentially in {manifest['total_seconds']:.0f}s")
    for r in manifest["runs"]:
        print(f"  {r['run']:16s} carried {r['test_bpc']['carried']:.4f}   "
              f"{r['reported_wall_clock_s']:.0f}s "
              f"({r['wall_clock_vs_baseline']:.2f}x baseline)")
    print(f"WROTE {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
