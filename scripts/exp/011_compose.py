"""EXP_011: train and score the composed arm, strictly sequentially.

Pre-registered in `experiments/logs/EXP_011_composition.md`. This file only
executes its §2; every choice it makes was fixed there, and this file is
committed before any of its numbers are read.

WHAT IS DIFFERENT FROM `007_run_prescan_arms.py`, AND IT IS THE WHOLE POINT
---------------------------------------------------------------------------
That driver's K1 anchors to `snn_beta0.5_s0` -- each pre-scan arm is *the
Phase-2 baseline* plus one thing. This one anchors to **`twocomp_s0`**, because
the composed arm is *the adopted arm* plus one thing. Checking it against the
Phase-2 baseline would pass while saying nothing: the two-compartment neuron's
own two parameters would show up as "differs", and the check that matters --
that nothing except the threshold moved -- would be swallowed by them.

WHAT IT GUARDS (EXP_011 §6)
---------------------------
K1 -- **the composed run is `twocomp_s0` plus one thing.** Field-by-field
against the adopted arm's config; only `seed`, `run_name`, `arch` and
`thr_log_init` may differ. `twocomp_s0`'s config predates `thr_log_init`, so
that field is excused only when the new run leaves it at the `Config` default,
and the excusal is recorded in the manifest rather than swallowed. "Absent from
an older config" and "set to something else" are different facts.

K2 -- **no mutation campaign while training**, re-checked before every run.

K3 -- wall-clock per run against **two** denominators, both named: the Phase-2
baseline's measured 398.9 s (the standing convention) and the adopted arm's own
three-seed mean, **recomputed from `summary.json` rather than pasted**. Rev 2 of
the Phase-4 report found two ratio columns quoting a reference they had not been
measured against; recomputing here is that correction applied in advance.

K5 -- **the fp64 identity gate runs before any training starts.** EXP_011 §8
makes it an entry condition: if Identity 1 does not hold for the two-compartment
neuron, §1.1 is wrong and there is nothing worth training. Run as a subprocess
so that a failure stops the driver rather than being reported at the end.

M5 -- the fused/eager dispatch actually taken is recorded per run. A composed
arm that silently fell back to the eager scan would be a different wall-clock
number and a different rounding, and nothing else in the artifacts would say so.

RESTARTABILITY
--------------
Complete runs are skipped; partial ones are **deleted** and retrained, never
resumed, so no artifact describes two runs at once. This is not defensive
programming -- a harness timeout killed the EXP_007/008 chain mid-run once
already, and this driver is expected to outlive the session that starts it.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import shutil
import statistics
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

#: K1 anchors to the ADOPTED arm, not the Phase-2 baseline. See the docstring.
REFERENCE_RUN = "twocomp_s0"
SEEDS = (0, 1, 2)
ARCH = "twocomp_threshold"
PREFIX = "compose"

#: The one init flag that may differ from the reference (K1).
INIT_FLAG = "thr_log_init"
INIT_VALUE = "0.0"

#: `summary.json`'s wall_clock_s for `snn_beta0.5_s0`; the standing denominator
#: for every ratio this project has quoted since EXP_004.
BASELINE_WALL_CLOCK_S = 398.88

#: The adopted arm's runs, for K3's second denominator. Recomputed, not pasted.
TWOCOMP_RUNS = ("twocomp_s0", "twocomp_s1", "twocomp_s2")

#: The identity gate that EXP_011 §8 makes an entry condition.
IDENTITY_GATE = "tests/test_compose_equivalence.py"

MAY_DIFFER = {"seed", "run_name", "arch", INIT_FLAG}

_DEFAULTS = {f.name: f.default for f in dataclasses.fields(Config)}


def is_complete(run: str) -> bool:
    """True iff `run` finished its full budget and was scored on test.

    Both halves, for `007_run_prescan_arms.py`'s reason: `summary.json` is written
    only after the trainer completes `max_steps`, and `final_test.json` only after
    `scripts/evaluate.py` scores it. A directory holding a `ckpt_last.pt` from an
    interrupted run satisfies neither.
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


def _run(argv: list[str], label: str) -> float:
    print(f"\n=== {label}\n    {' '.join(argv)}", flush=True)
    t0 = time.time()
    proc = subprocess.run([sys.executable, "-u", *argv], cwd=str(_REPO))
    dt = time.time() - t0
    if proc.returncode != 0:
        raise SystemExit(f"{label} failed with exit code {proc.returncode}")
    print(f"    done in {dt:.1f}s", flush=True)
    return dt


def _identity_gate() -> None:
    """K5 / EXP_011 §8: Identity 1 in fp64 before a single training step runs."""
    print(f"\n=== K5 entry condition: {IDENTITY_GATE}", flush=True)
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", IDENTITY_GATE], cwd=str(_REPO)
    )
    if proc.returncode != 0:
        raise SystemExit(
            "K5 FAILED: Identity 1 does not hold for the two-compartment neuron. "
            "EXP_011 §1.1's derivation is wrong and the experiment does not run. "
            "Chase the residual; do not widen a tolerance and do not train."
        )


def _dispatch() -> dict:
    """M5: what the scan will actually dispatch to, recorded rather than assumed."""
    import torch

    from snn.kernels import jiterator_available

    return {
        "cuda_available": bool(torch.cuda.is_available()),
        "jiterator_available": bool(jiterator_available()),
        "fused_path_taken": bool(torch.cuda.is_available() and jiterator_available()),
        "device_name": (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        ),
        "torch": torch.__version__,
    }


def _train_argv(seed: int) -> list[str]:
    """The reference run's launch command with the arch, the seed and the name
    changed, and nothing else. Everything not named here is a `Config` default and
    is verified as such by K1."""
    return [
        "--arch", ARCH,
        "--run_name", f"{PREFIX}_s{seed}",
        "--seed", str(seed),
        f"--{INIT_FLAG}", INIT_VALUE,
        "--eval_every", "2500",
        "--eval_max_windows", "2560",
        "--ckpt_every", "5000",
        "--log_every", "250",
    ]


def check_same_arm(run: str, reference: dict) -> dict:
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
            # The reference config predates this field; the new run left it at the
            # Config default, so the two models agree on it.
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
        "arch": ARCH,
        "differs_in": sorted(k for k in diffs if k in MAY_DIFFER),
        "absent_from_reference_and_left_at_default": sorted(excused_as_default),
        "same_arm": True,
    }


def _twocomp_wall_clock() -> dict:
    """K3's second denominator, recomputed from the adopted arm's own summaries."""
    vals = []
    for r in TWOCOMP_RUNS:
        p = RUNS / r / "summary.json"
        if p.exists():
            v = json.loads(p.read_text(encoding="utf-8")).get("wall_clock_s")
            if v:
                vals.append(float(v))
    return {
        "runs": list(TWOCOMP_RUNS),
        "n": len(vals),
        "mean_wall_clock_s": round(statistics.fmean(vals), 2) if vals else None,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", default=",".join(str(s) for s in SEEDS))
    ap.add_argument("--out", default="docs/reports/data/exp_011_run_manifest.json")
    ap.add_argument("--skip-gate", action="store_true",
                    help="skip the K5 identity gate (for a restart that already "
                         "passed it; NOT for a first run)")
    args = ap.parse_args(argv)

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]

    ref_path = RUNS / REFERENCE_RUN / "config.json"
    if not ref_path.exists():
        raise SystemExit(
            f"K1 has no anchor: {ref_path.relative_to(_REPO)} is missing. The "
            "composed arm is defined as the adopted arm plus one thing, so the "
            "adopted arm's config has to be present to check against."
        )
    reference = json.loads(ref_path.read_text(encoding="utf-8"))

    if not args.skip_gate:
        _identity_gate()

    twocomp_wc = _twocomp_wall_clock()
    started = time.time()
    manifest: dict = {
        "experiment": "EXP_011_composition",
        "log": "experiments/logs/EXP_011_composition.md",
        "arch": ARCH,
        "reference_run": REFERENCE_RUN,
        "reference_note": (
            "K1 anchors to the ADOPTED arm, not the Phase-2 baseline: the composed "
            "arm is twocomp plus one thing."
        ),
        "baseline_wall_clock_s": BASELINE_WALL_CLOCK_S,
        "baseline_wall_clock_run": "snn_beta0.5_s0",
        "twocomp_wall_clock": twocomp_wc,
        "sequential": True,
        "seeds": seeds,
        "dispatch": _dispatch(),
        "runs": [],
    }

    for seed in seeds:
        run = f"{PREFIX}_s{seed}"
        if is_complete(run):
            print(f"\n=== {run} already complete, skipping", flush=True)
            t_train = t_eval = 0.0
        else:
            if (RUNS / run).exists():
                # A partial directory would append to log.jsonl and leave a
                # ckpt_last.pt from a run that never finished, so the artifacts
                # would describe two runs at once. Removed, not resumed.
                print(f"\n=== {run} is partial; removing and retraining", flush=True)
                shutil.rmtree(RUNS / run)
            _check_no_campaign()          # before EVERY run, not once
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
        wc = summary.get("wall_clock_s")
        row.update({
            "seed": seed,
            "trained_in_this_invocation": t_train > 0,
            "train_seconds": round(t_train, 1),
            "eval_seconds": round(t_eval, 1),
            "reported_wall_clock_s": wc,
            "wall_clock_vs_phase2_baseline": (
                round(wc / BASELINE_WALL_CLOCK_S, 3) if wc else None),
            "wall_clock_vs_twocomp": (
                round(wc / twocomp_wc["mean_wall_clock_s"], 3)
                if wc and twocomp_wc["mean_wall_clock_s"] else None),
            "peak_vram_gib": summary.get("peak_vram_gib"),
            "params": test.get("params"),
            "test_bpc": {p: test["results"][p]["bpc"] for p in test["results"]},
        })
        manifest["runs"].append(row)
        print(f"    {run}: test bpc carried = "
              f"{test['results']['carried']['bpc']:.5f}", flush=True)

    carried = [r["test_bpc"]["carried"] for r in manifest["runs"]]
    # A diverged seed scores NaN, and a NaN in this list would poison the mean
    # and the sd into NaN -- a manifest that reports nothing about the seeds that
    # DID finish. Both lists are recorded: the raw one so the failure is visible
    # in the artifact, and the finite one so the surviving seeds are readable.
    #
    # This changes no bar and no verdict. `011_compose_results.py` applies every
    # EXP_011 criterion and does its own exclusion under the post-hoc rule in its
    # header; this block is a record, not a decision.
    finite = [v for v in carried if v == v and abs(v) != float("inf")]
    diverged = [r["run"] for r in manifest["runs"]
                if not (r["test_bpc"]["carried"] == r["test_bpc"]["carried"])]
    manifest["elapsed_s"] = round(time.time() - started, 1)
    manifest["carried_bpc"] = {
        "values": carried,
        "n": len(carried),
        "diverged_runs": diverged,
        "n_finite": len(finite),
        "mean_finite_only": statistics.fmean(finite) if finite else None,
        # Sample sd (n-1). EXP_011 C2 bars this at 0.005; the driver reports it
        # and does not judge it -- `011_compose_results.py` applies the bars.
        "sd_finite_only": statistics.stdev(finite) if len(finite) > 1 else None,
    }

    out = _REPO / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"\nwrote {out.relative_to(_REPO)}", flush=True)
    cb = manifest["carried_bpc"]
    if cb["diverged_runs"]:
        print(f"DIVERGED (excluded from the means below, NOT replaced): "
              f"{cb['diverged_runs']}", flush=True)
    if cb["mean_finite_only"] is not None:
        print(f"carried bpc (n={cb['n_finite']} finite of {cb['n']}): "
              f"mean {cb['mean_finite_only']:.5f} sd {cb['sd_finite_only']}",
              flush=True)
    else:
        print("carried bpc: every seed diverged; there is no mean to report",
              flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
