"""EXP_015 driver -- the architecture ladder at one width, one arm at a time.

    python scripts/launch.py --script scripts/exp/015_run_arch_ladder.py \
        --run-name _exp015_driver -- --out docs/reports/data/exp_015_run_manifest.json

Trains `twocomp`, `twocomp_threshold` and `gru` at `d_model = 1481, seed = 0` and
scores each on the test split.  The fourth leg -- the plain `snn` control -- is
`scale_d1481_s0`, already trained by `EXP_014`, and is **not retrained**: it is
the same arch at the same width, seed, recipe and tree, and entry condition E1
(`EXP_015` Sec 8) discharged the tree question by measurement.

WHY THIS IS A DRIVER AND NOT THREE COMMANDS
-------------------------------------------
`CONTRIBUTING.md` Sec 6: long runs die under the harness's interactive timeout, so
this is launched detached and made restartable.  A rung is skipped only when it
is genuinely complete -- `summary.json` at full step count **and**
`final_test.json` present -- and a partial directory is **deleted** rather than
resumed, so no artifact ever describes two runs at once.

WHAT IT REFUSES TO DO
---------------------
* It reads no bpc and resolves no bar.  `015_arch_results.py` does that, and is
  committed before any rung's bpc is read.
* It does not retrain the control.  Doing so would spend 0.61 GPU-hours to
  reproduce a number this tree has already been shown to reproduce.
* It does not stop the ladder when a rung diverges.  `EXP_015` Sec P2 names the
  risk in advance -- `EXP_011`'s `compose_s0` was this arch at this seed -- and
  pre-registers the response: report the divergence, resolve that prediction
  NOT RUN, and let the remaining legs stand.  A driver that aborted here would
  turn one arm's known failure mode into three arms of lost GPU time.
* It does not reseed anything.  `CONTRIBUTING.md` Sec 4: seeds are added, never
  substituted.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib.util
import json
import math
import shutil
import subprocess
import sys
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO / "src") not in sys.path:
    sys.path.insert(0, str(_REPO / "src"))

from snn.config import Config  # noqa: E402

# `param_count` dispatches the committed closed form per arch. Imported from the
# calibration script rather than re-derived here: two copies of a parameter
# formula is exactly how G3 stops being a check.
_CALIB_SPEC = importlib.util.spec_from_file_location(
    "calib014", _REPO / "scripts" / "exp" / "014_calibrate_cost.py")
_calib = importlib.util.module_from_spec(_CALIB_SPEC)
_CALIB_SPEC.loader.exec_module(_calib)
param_count = _calib.param_count

RUNS = _REPO / "experiments" / "runs"
LOCKFILE = _REPO / "src" / "snn" / ".MUTATION_CAMPAIGN_RUNNING"
PREREG = _REPO / "experiments" / "logs" / "EXP_015_architecture_at_width.md"
CALIB = _REPO / "docs" / "reports" / "data" / "exp_015_cost_calibration.json"

#: The three arms to train. `snn` is the control and is already trained.
ARMS: tuple[str, ...] = ("twocomp", "twocomp_threshold", "gru")

#: Sec 2.2. Fixed, and the only reason the comparison is parameter-matched.
D_MODEL = 1481
SEED = 0

#: Sec 2.3. Stricter than EXP_014's: no "absent from the reference" excusal,
#: because `scale_d1481_s0` carries every field of the current `Config`.
MAY_DIFFER = {"arch", "run_name"}
REFERENCE_RUN = "scale_d1481_s0"

#: Sec 6, derived from Sec 2.1's measured 2.9767 GiB maximum on a 7.96 GiB card.
#: EXP_014's 3.0 was a modelling assumption about a *width*; this one is 34%
#: above a measurement and is fixed before any bpc exists.
VRAM_ALARM_GIB = 4.0

_VOCAB = 205
_LAYERS = 2
_DEFAULTS = {f.name: f.default for f in dataclasses.fields(Config)}


def prereg_sha256() -> str:
    return hashlib.sha256(PREREG.read_bytes()).hexdigest()


def run_name(arch: str) -> str:
    return f"arch_{arch}_d{D_MODEL}_s{SEED}"


def is_complete(run: str) -> bool:
    """True iff `run` finished its full budget and was scored on test.

    Both halves: `summary.json` is written only after the trainer completes
    `max_steps`, and `final_test.json` only after `scripts/evaluate.py` scores
    it, so a directory holding a `ckpt_last.pt` from an interrupted run
    satisfies neither.
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
            "applying wrong kernels to the source tree. Refusing to start a run "
            "that would import one (03_phase3_candidates.md Sec 8)."
        )


def _train_argv(arch: str) -> list[str]:
    """The control run's launch command with `arch` and `run_name` changed, and
    nothing else. Every other flag is `scale_d1481_s0`'s, which is what makes K1
    a check rather than a formality."""
    return [
        str(_REPO / "scripts" / "train.py"),
        "--arch", arch,
        "--d_model", str(D_MODEL),
        "--n_layers", str(_LAYERS),
        "--run_name", run_name(arch),
        "--seed", str(SEED),
        "--eval_every", "2500",
        "--eval_max_windows", "2560",
        "--ckpt_every", "5000",
        "--log_every", "250",
    ]


def _eval_argv(arch: str) -> list[str]:
    run = run_name(arch)
    return [
        str(_REPO / "scripts" / "evaluate.py"),
        "--ckpt", str(RUNS / run / "ckpt_final.pt"),
        "--split", "test",
        "--out", str(RUNS / run / "final_test.json"),
    ]


def _run(argv: list[str], label: str, fatal: bool = True) -> tuple[float, int]:
    """Returns (seconds, returncode). `fatal=False` lets a diverging rung be
    recorded and stepped over rather than taking the ladder down with it."""
    print(f"\n=== {label}\n    {' '.join(argv)}", flush=True)
    t0 = time.time()
    proc = subprocess.run([sys.executable, "-u", *argv], cwd=str(_REPO))
    dt = time.time() - t0
    if proc.returncode != 0:
        if fatal:
            raise SystemExit(f"{label} failed with exit code {proc.returncode}")
        print(f"    FAILED after {dt:.1f}s with exit code {proc.returncode}",
              flush=True)
    else:
        print(f"    done in {dt:.1f}s", flush=True)
    return dt, proc.returncode


def check_same_arm(arch: str, reference: dict) -> dict:
    """K1 (Sec 7). Returns the row; raises if anything differs that may not."""
    run = run_name(arch)
    cfg = json.loads((RUNS / run / "config.json").read_text(encoding="utf-8"))
    diffs = {k: [reference.get(k), cfg.get(k)]
             for k in set(reference) | set(cfg)
             if reference.get(k) != cfg.get(k)}

    excused_as_default = []
    unexpected = {}
    for k, (ref_v, new_v) in diffs.items():
        if k in MAY_DIFFER:
            continue
        # Retained from EXP_014's driver so a Config field added between this
        # run and the reference cannot fail K1 spuriously. Sec 2.3 says it is
        # expected to stay empty here, and the manifest records whether it did.
        if ref_v is None and k in _DEFAULTS and new_v == _DEFAULTS[k]:
            excused_as_default.append(k)
            continue
        unexpected[k] = [ref_v, new_v]
    if unexpected:
        raise SystemExit(
            f"K1: {run} is not {REFERENCE_RUN} plus one thing. Fields that "
            f"differ but may not: {unexpected}"
        )
    if int(cfg.get("d_model", -1)) != D_MODEL:
        raise SystemExit(
            f"K1: {run} records d_model={cfg.get('d_model')!r}, expected "
            f"{D_MODEL} -- the shared width is what makes this comparison "
            "parameter-matched, so a differing one is a different experiment"
        )
    if int(cfg.get("seed", -1)) != SEED:
        raise SystemExit(
            f"K1: {run} records seed={cfg.get('seed')!r}, expected {SEED} -- "
            "Sec 2.3 holds every leg at one seed"
        )
    if str(cfg.get("arch")) != arch:
        raise SystemExit(
            f"K1: {run} records arch={cfg.get('arch')!r}, expected {arch!r}"
        )
    return {
        "run": run,
        "arch": arch,
        "d_model": int(cfg.get("d_model")),
        "seed": SEED,
        "differs_in": sorted(k for k in diffs if k in MAY_DIFFER),
        "absent_from_reference_and_left_at_default": sorted(excused_as_default),
        "same_arm": True,
    }


def check_params_and_vram(arch: str) -> dict:
    """G3 and G4 (Sec 6). Both abort."""
    run = run_name(arch)
    summary = json.loads((RUNS / run / "summary.json").read_text(encoding="utf-8"))
    expected = param_count(arch, _VOCAB, D_MODEL, _LAYERS)
    realised = int(summary.get("params", -1))
    if realised != expected:
        raise SystemExit(
            f"G3: {run} realised {realised:,} parameters, pre-registered "
            f"{expected:,}. Parameter matching is the premise of this "
            "experiment; a mismatch removes it silently."
        )
    peak = float(summary.get("peak_vram_gib", -1.0))
    if peak > VRAM_ALARM_GIB:
        raise SystemExit(
            f"G4: {run} peaked at {peak} GiB, past the {VRAM_ALARM_GIB} GiB "
            "alarm. Under WDDM a spill is a ~50x slowdown that does not raise, "
            "so this is investigated rather than absorbed."
        )
    return {"params": realised, "params_expected": expected,
            "peak_vram_gib": peak, "wall_clock_s": summary.get("wall_clock_s")}


def scan_for_divergence(arch: str) -> dict:
    """Sec P2's named risk, measured rather than inferred from an exit code.

    A run can finish cleanly and still have gone non-finite, and it can be killed
    for reasons that have nothing to do with the arm, so the log is read either
    way: how many logged losses are non-finite, and the step of the first.
    """
    p = RUNS / run_name(arch) / "log.jsonl"
    if not p.exists():
        return {"log_present": False}
    train = []
    for line in p.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            if r.get("event") == "train":
                train.append(r)
    bad = [r for r in train
           if r.get("loss") is not None and not math.isfinite(float(r["loss"]))]
    return {
        "log_present": True,
        "n_train_records": len(train),
        "last_logged_step": train[-1]["step"] if train else None,
        "n_nonfinite_loss_records": len(bad),
        "first_nonfinite_step": bad[0]["step"] if bad else None,
        "max_grad_norm": max((r.get("grad_norm", 0.0) for r in train), default=None),
        "n_logged_steps_over_clip": sum(
            1 for r in train if (r.get("grad_norm") or 0.0) > 1.0),
    }


def leg_completed(returncode: int, scan: dict, ckpt_exists: bool) -> bool:
    """Did this leg produce a result that may be scored against a bar?

    **Added after the fact, and the fact is worth recording.** As first written
    this driver asked only `returncode == 0 and ckpt_final.pt exists`, so both
    diverged legs of `EXP_015` were marked `completed: true`, evaluated to NaN,
    and handed to the resolver as if they were results.  No evidence was lost --
    `scan_for_divergence` recorded all 60 and 73 non-finite records, and the
    manifest carries them -- but the field said the opposite of what happened.

    A training process that goes non-finite still exits 0 and still writes a
    checkpoint, because nothing in the trainer treats NaN as an error.  So the
    exit code cannot answer this question and the log has to.

    `log_present` is required rather than assumed, which is the same defect one
    level down: a missing log makes `n_nonfinite_loss_records` absent, and
    "absent" would otherwise read as "zero" and mark an unverifiable leg
    complete.  No log is not the same as a clean log.
    """
    return bool(returncode == 0 and ckpt_exists
                and scan.get("log_present")
                and not scan.get("n_nonfinite_loss_records"))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--out", default="docs/reports/data/exp_015_run_manifest.json")
    args = ap.parse_args(argv)

    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    reference = json.loads(
        (RUNS / REFERENCE_RUN / "config.json").read_text(encoding="utf-8"))

    prereg_before = prereg_sha256()
    calib = json.loads(CALIB.read_text(encoding="utf-8")) if CALIB.exists() else None
    projected = {}
    if calib:
        for r in calib.get("results", []):
            if "projected_20k_s" in r:
                projected[r["arch"]] = r["projected_20k_s"]

    manifest: dict = {
        "experiment": "EXP_015_architecture_at_width",
        "prereg": str(PREREG.relative_to(_REPO)).replace("\\", "/"),
        "prereg_sha256_before_first_run": prereg_before,
        "reference_run": REFERENCE_RUN,
        "control_leg": {
            "run": REFERENCE_RUN, "arch": "snn", "retrained": False,
            "why": ("same arch, width, seed, recipe and tree as this ladder; "
                    "EXP_015 Sec 8 E1 discharged the tree question by measurement, "
                    "so retraining it would spend 0.61 GPU-h to reproduce a "
                    "number already shown to reproduce"),
        },
        "may_differ": sorted(MAY_DIFFER),
        "d_model": D_MODEL,
        "seed": SEED,
        "arms": arms,
        "vram_alarm_gib": VRAM_ALARM_GIB,
        "adopts_nothing": True,
        "ranks_nothing": True,
        "reads_no_bpc": True,
        "rungs": [],
    }
    out = Path(args.out)
    if not out.is_absolute():
        out = _REPO / out
    out.parent.mkdir(parents=True, exist_ok=True)

    def flush() -> None:
        out.write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")

    flush()

    for arch in arms:
        run = run_name(arch)
        row: dict = {"arch": arch, "run": run}
        if is_complete(run):
            print(f"\n=== SKIP {run}: already complete", flush=True)
            row["skipped_as_complete"] = True
        else:
            if (RUNS / run).exists():
                print(f"\n=== DELETE partial {run} (never resumed)", flush=True)
                shutil.rmtree(RUNS / run)
                row["deleted_partial"] = True
            _check_no_campaign()
            dt, rc = _run(_train_argv(arch), f"train {run}", fatal=False)
            row["train_wall_clock_s"] = round(dt, 1)
            row["train_returncode"] = rc
            scan = scan_for_divergence(arch)
            row["divergence_scan"] = scan
            ckpt_exists = (RUNS / run / "ckpt_final.pt").exists()
            if rc != 0 or not ckpt_exists:
                # Sec P2: recorded, not reseeded, and the ladder continues.
                row["completed"] = False
                row["note"] = (
                    "training did not finish; this leg's prediction resolves "
                    "NOT RUN. Not reseeded -- CONTRIBUTING.md Sec 4."
                )
                manifest["rungs"].append(row)
                flush()
                continue
            _check_no_campaign()
            # A diverged run is still evaluated, deliberately: the NaN in
            # `final_test.json` is evidence, and suppressing it would leave the
            # record showing an arm with no number rather than an arm that
            # produced one and it was NaN.
            ev_dt, ev_rc = _run(_eval_argv(arch), f"evaluate {run}", fatal=False)
            row["eval_wall_clock_s"] = round(ev_dt, 1)
            row["eval_returncode"] = ev_rc
            if ev_rc != 0:
                row["completed"] = False
                row["note"] = "evaluation failed; this leg's prediction is NOT RUN"
                manifest["rungs"].append(row)
                flush()
                continue

        scan = row.get("divergence_scan") or scan_for_divergence(arch)
        row["divergence_scan"] = scan
        row["diverged"] = bool(scan.get("n_nonfinite_loss_records"))
        row["completed"] = leg_completed(
            row.get("train_returncode", 0), scan,
            (RUNS / run / "ckpt_final.pt").exists())
        if row["diverged"]:
            row["note"] = (
                f"DIVERGED: {scan['n_nonfinite_loss_records']} of "
                f"{scan['n_train_records']} logged losses non-finite, first at "
                f"step {scan['first_nonfinite_step']}. This leg's prediction "
                "resolves NOT RUN. Not reseeded -- CONTRIBUTING.md Sec 4."
            )
        row.setdefault("divergence_scan", scan_for_divergence(arch))
        row["k1"] = check_same_arm(arch, reference)
        row.update(check_params_and_vram(arch))
        if arch in projected:
            row["k3"] = {
                "projected_20k_s_from_calibration": projected[arch],
                "realised_train_wall_clock_s": row.get("wall_clock_s"),
                "ratio_realised_over_projected": (
                    None if not row.get("wall_clock_s")
                    else round(row["wall_clock_s"] / projected[arch], 4)),
                "note": ("the projection is a 400-step per-step rate scaled by "
                         "scale_d1481_s0's committed 2191.44 s; it excludes "
                         "capture cost and the in-training evaluations, so a "
                         "ratio near EXP_014's measured 0.925-0.949 is expected "
                         "and a figure far outside it is a systems finding"),
            }
        manifest["rungs"].append(row)
        flush()

    prereg_after = prereg_sha256()
    manifest["prereg_sha256_after_last_run"] = prereg_after
    manifest["prereg_unchanged"] = prereg_after == prereg_before
    flush()
    if not manifest["prereg_unchanged"]:
        raise SystemExit(
            "The pre-registration changed while the ladder was running. The "
            "experiment is void; nothing here may be reported."
        )
    done = sum(1 for r in manifest["rungs"] if r.get("completed"))
    print(f"\nwrote {out}")
    print("EXP015_ALL_DONE", done, "/", len(arms), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
