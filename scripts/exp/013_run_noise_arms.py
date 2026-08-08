"""EXP_013: train and score the noise-injection ladder, strictly sequentially.

Pre-registered in `experiments/logs/EXP_013_noise_injection.md`. This file only
executes its §2; every choice it makes was fixed there, including the three
amplitudes and the three seeds.

WHY THE NOISE-OFF ARM IS NOT IN THIS DRIVER
-------------------------------------------
Because it does not need training. `tests/test_noise.py`'s G1 asserts that
`arch="noise"` at `noise_amp = 0` is `SpikingCharLM` **bitwise, forward and
backward** -- `torch.equal` on the logits and on every gradient, not a tolerance.
So the noise-off control is the five committed `snn_beta0.5` seeds, at n = 5
rather than the n = 3 a retrained control would have had, and the three runs that
would have produced a numerically identical answer are not spent.

That is only sound because the gate is bitwise. `EXP_011` K4 is the precedent
(`== 0.0`, not a tolerance), and the reason both are bitwise rather than 1e-12 is
that a nesting claim which holds "to within" something is a claim that the two
models are different.

WHAT IT GUARDS
--------------
K1 -- **each run is the baseline plus one thing**, and the one thing is
`noise_amp`. Every run's `config.json` is compared field-by-field against
`snn_beta0.5_s0`'s. Fields the baseline's config predates -- it was written
before `beta_slow`, `w_init`, `mu_init`, `thr_log_init`, `noise_amp` and
`noise_p` existed -- are excused **only** when the new run leaves them at the
`Config` default, and every excusal is recorded in the manifest instead of being
swallowed. `noise_p` is excused this way rather than allowed outright, which is
the check that would catch a run that swept it: §2.4 holds it fixed at 0.1 and
that is the `Config` default, so any other value is an unexpected difference and
stops the driver.

K2 -- **no mutation campaign while training**, re-checked before every run
(phase-3 report §8).

K3 -- wall-clock per run against the baseline's measured 398.9 s. This arm draws
`K * B * L * d` Bernoullis per step outside the captured region, which is O(1)
kernels per layer but not zero work, and `EXP_007` is this project's standing
reminder that "outside the time loop" is not "free" -- token-shift costs 1.49x.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
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

#: EXP_013 §2.4's ladder, geometric at x4. Fixed before any run.
AMPLITUDES = (0.1, 0.4, 1.6)

#: EXP_013 §2.4 holds this fixed and names it an unswept point.
NOISE_P = 0.1

#: Fields that may differ from the reference (K1).
MAY_DIFFER = {"seed", "run_name", "arch", "noise_amp"}

_DEFAULTS = {f.name: f.default for f in dataclasses.fields(Config)}

#: The pre-registration. Its SHA-256 is stamped into the manifest before the
#: first run and re-checked after the last, which is EXP_013 §8 item 5: this
#: driver makes no git commit, so the hash is what guarantees no threshold in §3
#: or §4 moved after the numbers arrived.
PREREG = _REPO / "experiments" / "logs" / "EXP_013_noise_injection.md"


def prereg_sha256() -> str:
    return hashlib.sha256(PREREG.read_bytes()).hexdigest()


def run_name(amp: float, seed: int) -> str:
    """`noise_a0p4_s1`. The amplitude is in the directory name because a ladder
    whose points are distinguishable only by reading each `config.json` is a
    ladder someone will eventually mis-attribute."""
    return f"noise_a{('%g' % amp).replace('.', 'p')}_s{seed}"


def is_complete(run: str) -> bool:
    """True iff `run` finished its full budget and was scored on test.

    Both halves, for the reason `007_run_prescan_arms.py` records: `summary.json`
    is written only after the trainer completes `max_steps` and `final_test.json`
    only after `scripts/evaluate.py` scores it, so a directory holding a
    `ckpt_last.pt` from an interrupted run satisfies neither. `main` deletes such
    a directory rather than resuming into it.
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


def _train_argv(amp: float, seed: int) -> list[str]:
    """The reference run's launch command with the arch, the amplitude, the seed
    and the name changed, and nothing else. `--noise_p` is passed explicitly at
    its default so the run's `config.json` records the value §2.4 fixed rather
    than inheriting it silently."""
    return [
        "--arch", "noise",
        "--run_name", run_name(amp, seed),
        "--seed", str(seed),
        "--noise_amp", repr(float(amp)),
        "--noise_p", repr(float(NOISE_P)),
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


def check_same_arm(run: str, amp: float, reference: dict) -> dict:
    """K1. Returns the row; raises if anything differs that may not."""
    cfg = json.loads((RUNS / run / "config.json").read_text(encoding="utf-8"))
    diffs = {k: [reference.get(k), cfg.get(k)]
             for k in set(reference) | set(cfg)
             if reference.get(k) != cfg.get(k)}

    excused_as_default = []
    unexpected = {}
    for k, (ref_v, new_v) in diffs.items():
        if k in MAY_DIFFER:
            continue
        if ref_v is None and k in _DEFAULTS and new_v == _DEFAULTS[k]:
            excused_as_default.append(k)
            continue
        unexpected[k] = [ref_v, new_v]
    if unexpected:
        raise SystemExit(
            f"K1: {run} is not {REFERENCE_RUN} plus one thing. Fields that "
            f"differ but may not: {unexpected}"
        )
    if float(cfg.get("noise_amp", -1)) != float(amp):
        raise SystemExit(
            f"K1: {run} records noise_amp={cfg.get('noise_amp')!r}, expected {amp}"
        )
    if float(cfg.get("noise_p", -1)) != float(NOISE_P):
        raise SystemExit(
            f"K1: {run} records noise_p={cfg.get('noise_p')!r}, expected {NOISE_P} "
            "-- §2.4 holds it fixed and a swept p is a different experiment"
        )
    return {
        "run": run,
        "arch": "noise",
        "noise_amp": float(amp),
        "noise_p": float(NOISE_P),
        "differs_in": sorted(k for k in diffs if k in MAY_DIFFER),
        "absent_from_reference_and_left_at_default": sorted(excused_as_default),
        "same_arm": True,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--amps", default=",".join(str(a) for a in AMPLITUDES))
    ap.add_argument("--seeds", default=",".join(str(s) for s in SEEDS))
    ap.add_argument("--out", default="docs/reports/data/exp_013_run_manifest.json")
    args = ap.parse_args(argv)

    amps = [float(a) for a in args.amps.split(",") if a.strip()]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]

    reference = json.loads(
        (RUNS / REFERENCE_RUN / "config.json").read_text(encoding="utf-8"))
    prereg_before = prereg_sha256()
    started = time.time()
    manifest: dict = {
        "experiment": "EXP_013_noise_injection",
        "log": "experiments/logs/EXP_013_noise_injection.md",
        "prereg_sha256_before_first_run": prereg_before,
        "reference_run": REFERENCE_RUN,
        "reference_wall_clock_s": BASELINE_WALL_CLOCK_S,
        "noise_off_control": {
            "runs": [f"snn_beta0.5_s{i}" for i in range(5)],
            "trained_here": False,
            "why": "tests/test_noise.py G1 asserts arch=noise at noise_amp=0 is "
                   "SpikingCharLM bitwise, forward and backward, so the committed "
                   "baseline seeds ARE the noise-off arm, at n=5 rather than n=3.",
        },
        "sequential": True,
        "amplitudes": amps,
        "noise_p": NOISE_P,
        "seeds": seeds,
        "runs": [],
    }

    for amp in amps:
        for seed in seeds:
            run = run_name(amp, seed)
            if is_complete(run):
                print(f"\n=== {run} already complete, skipping", flush=True)
                t_train = t_eval = 0.0
            else:
                if (RUNS / run).exists():
                    print(f"\n=== {run} is partial; removing and retraining",
                          flush=True)
                    shutil.rmtree(RUNS / run)
                _check_no_campaign()      # before EVERY run, not once
                t_train = _run(["scripts/train.py", *_train_argv(amp, seed)],
                               f"train {run}")
                _check_no_campaign()
                t_eval = _run([
                    "scripts/evaluate.py",
                    "--ckpt", str(RUNS / run / "ckpt_final.pt"),
                    "--split", "test",
                    "--out", str(RUNS / run / "final_test.json"),
                ], f"evaluate {run}")

            row = check_same_arm(run, amp, reference)
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
                "test_firing_rate": test["results"]["carried"]["firing_rate"],
            })
            manifest["runs"].append(row)
            print(f"    {run}: fresh {row['test_bpc']['fresh']:.4f}  "
                  f"carried {row['test_bpc']['carried']:.4f}  "
                  f"params {row['params']:,}", flush=True)

            out = _REPO / args.out
            out.parent.mkdir(parents=True, exist_ok=True)
            manifest["total_seconds"] = round(time.time() - started, 1)
            out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    prereg_after = prereg_sha256()
    manifest["prereg_sha256_after_last_run"] = prereg_after
    manifest["prereg_unchanged"] = (prereg_after == prereg_before)
    out = _REPO / args.out
    manifest["total_seconds"] = round(time.time() - started, 1)
    out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if not manifest["prereg_unchanged"]:
        raise SystemExit(
            "§8 item 5: the pre-registration changed while the runs were in "
            f"flight ({prereg_before} -> {prereg_after}). Every threshold in it "
            "is now unverifiable and the experiment is VOID. Do not report these "
            "numbers."
        )

    print("\n" + "=" * 70)
    print(f"EXP_013: {len(manifest['runs'])} runs trained and scored sequentially "
          f"in {manifest.get('total_seconds', 0):.0f}s")
    print(f"pre-registration SHA-256 unchanged: {manifest['prereg_unchanged']} "
          f"({prereg_before[:16]}...)")
    for r in manifest["runs"]:
        print(f"  {r['run']:18s} amp {r['noise_amp']:<5g} carried "
              f"{r['test_bpc']['carried']:.4f}   "
              f"{r['reported_wall_clock_s']:.0f}s "
              f"({r['wall_clock_vs_baseline']:.2f}x baseline)")
    print(f"WROTE {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
