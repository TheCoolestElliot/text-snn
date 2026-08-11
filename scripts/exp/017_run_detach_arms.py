"""EXP_017 Legs B and C -- the bounded arm at width, and what the bound costs.

    python scripts/launch.py --script scripts/exp/017_run_detach_arms.py \
        --run-name _exp017_driver -- --out docs/reports/data/exp_017_run_manifest.json

Leg B is one run: `arch = "twocomp_detach"` at `d_model = 1481`, seed 0, under the
**unchanged** frozen Phase-2 recipe -- every flag identical to
`arch_twocomp_d1481_s0`, the run that went non-finite at step 5138. It answers
`EXP_017` H2 and nothing else.

Leg C is six runs at `d_model = 512`: three seeds of the arm and three seeds of a
**freshly trained `twocomp` anchor**. The anchor is trained rather than read off
the committed 2.11869 because `EXP_014` §9.12 established that decision #7's fp64
clip moved the trajectory, and because **Elliot ruled on 2026-08-10 that this
project does not re-baseline: every new experiment trains its own anchor.**
Spending 0.45 GPU-hours on it is that ruling being obeyed.

WHY THIS IS A DRIVER AND NOT SEVEN COMMANDS
-------------------------------------------
`CONTRIBUTING.md` §6: long runs die under the harness's interactive timeout, so
this is launched detached and made restartable. A run is skipped only when it is
genuinely complete -- `summary.json` at full step count **and** `final_test.json`
present -- and a partial directory is **deleted** rather than resumed, so no
artifact ever describes two runs at once.

WHAT IT REFUSES TO DO
---------------------
* It reads no bpc and resolves no bar. `017_detach_results.py` does that, and is
  committed before any run's bpc is read.
* **It changes no hyperparameter, and G5 proves it rather than promising it.**
  Every frozen term is read back out of each run's own `config.json` and asserted
  against the pre-registered value. H2's entire meaning is "it survived the recipe
  that killed the adopted arm", so a driver that quietly passed `--lr` would make
  the result unfalsifiable.
* It does not stop when a run diverges. `EXP_015`'s driver learned this: aborting
  turns one arm's known failure mode into six arms of lost GPU time. A diverged
  run is recorded, evaluated (the NaN in `final_test.json` is evidence), and
  stepped over.
* It does not reseed anything. `CONTRIBUTING.md` §4: seeds are added, never
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
# formula is exactly how G2 stops being a check.
_CALIB_SPEC = importlib.util.spec_from_file_location(
    "calib014", _REPO / "scripts" / "exp" / "014_calibrate_cost.py")
_calib = importlib.util.module_from_spec(_CALIB_SPEC)
_CALIB_SPEC.loader.exec_module(_calib)
param_count = _calib.param_count

RUNS = _REPO / "experiments" / "runs"
LOCKFILE = _REPO / "src" / "snn" / ".MUTATION_CAMPAIGN_RUNNING"
PREREG = _REPO / "experiments" / "logs" / "EXP_017_bounded_reset_jacobian.md"

_VOCAB = 205
_LAYERS = 2
_DEFAULTS = {f.name: f.default for f in dataclasses.fields(Config)}

#: Leg B: the width that kills the adopted arm, its seed, its reference run.
D_WIDE = 1481
#: Leg C: the width the adopted arm was adopted at.
D_NARROW = 512
SEEDS = (0, 1, 2)

#: `EXP_017` §7 G6. `EXP_015`'s figure at this width; the adopted arm peaked at
#: 2.61 GiB there, and this arm saves the same tensors. Under WDDM a spill is a
#: ~50x slowdown that never raises, so this aborts rather than being absorbed.
VRAM_ALARM_GIB = 4.0

#: `EXP_017` §7 G5. THE POINT OF THE EXPERIMENT IS THAT NONE OF THESE MOVED.
#: Read back out of each run's own `config.json` after it finishes, not merely
#: omitted from the command line -- a default that changed under the experiment
#: would otherwise be invisible.
FROZEN: dict[str, object] = {
    "lr": 3e-3,
    "weight_decay": 0.1,
    "grad_clip": 1.0,
    "warmup_steps": 200,
    "max_steps": 20_000,
    "lr_schedule": "cosine",
    "beta1": 0.9,
    "beta2": 0.95,
    "beta": 0.5,
    "beta_slow": 0.95,
    "w_init": 0.1,
    "threshold": 1.0,
    "reset": "hard",
    "surrogate": "atan",
    "surrogate_alpha": 2.0,
    "seq_len": 256,
    "batch_size": 128,
    "corpus": "enwik8",
    "dtype": "fp32",
    "fused": True,
    "cuda_graph": True,
    "deterministic": True,
    "n_layers": 2,
}

#: (leg, run_name, arch, d_model, seed, reference_run, may_differ).
#: ORDER IS LOAD-BEARING TWICE. Leg B runs first because it is the headline and
#: the one leg whose result could make the rest not worth running. Within Leg C
#: `anchor_twocomp_d512_s0` runs before anything that names it as a reference.
PLAN: list[tuple] = [
    ("B", "detach_d1481_s0", "twocomp_detach", D_WIDE, 0,
     "arch_twocomp_d1481_s0", frozenset({"arch", "run_name"})),
    ("C", "anchor_twocomp_d512_s0", "twocomp", D_NARROW, 0,
     None, frozenset()),
] + [
    ("C", f"anchor_twocomp_d512_s{s}", "twocomp", D_NARROW, s,
     "anchor_twocomp_d512_s0", frozenset({"arch", "run_name", "seed"}))
    for s in SEEDS if s != 0
] + [
    ("C", f"detach_d512_s{s}", "twocomp_detach", D_NARROW, s,
     "anchor_twocomp_d512_s0", frozenset({"arch", "run_name", "seed"}))
    for s in SEEDS
]


def prereg_sha256() -> str:
    return hashlib.sha256(PREREG.read_bytes()).hexdigest()


def is_complete(run: str) -> bool:
    """True iff `run` finished its full budget and was scored on test.

    Both halves: `summary.json` is written only after the trainer completes
    `max_steps`, and `final_test.json` only after `scripts/evaluate.py` scores
    it, so a directory holding a `ckpt_last.pt` from an interrupted run satisfies
    neither.
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
            "that would import one (03_phase3_candidates.md §8)."
        )


def _train_argv(run: str, arch: str, d_model: int, seed: int) -> list[str]:
    """`arch_twocomp_d1481_s0`'s launch command with arch/width/seed/name changed.

    Every other flag is that run's, which is what makes G3 a check rather than a
    formality. NOTE WHAT IS ABSENT: no `--lr`, no `--grad_clip`, no `--weight_decay`
    and no `--beta_slow`. G5 asserts afterwards that none of them moved anyway.
    """
    return [
        str(_REPO / "scripts" / "train.py"),
        "--arch", arch,
        "--d_model", str(d_model),
        "--n_layers", str(_LAYERS),
        "--run_name", run,
        "--seed", str(seed),
        "--eval_every", "2500",
        "--eval_max_windows", "2560",
        "--ckpt_every", "5000",
        "--log_every", "250",
    ]


def _eval_argv(run: str) -> list[str]:
    return [
        str(_REPO / "scripts" / "evaluate.py"),
        "--ckpt", str(RUNS / run / "ckpt_final.pt"),
        "--split", "test",
        "--out", str(RUNS / run / "final_test.json"),
    ]


def _run(argv: list[str], label: str, fatal: bool = True) -> tuple[float, int]:
    """Returns (seconds, returncode). `fatal=False` lets a diverging run be
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


def check_frozen(run: str) -> dict:
    """G5. Every frozen term, read back out of the run's own config.json.

    This is the gate H2's meaning depends on. "The bounded arm trains at width"
    is only interesting if it trained under the recipe that killed the unbounded
    one; under a gentler recipe it would be a different experiment wearing this
    one's name.
    """
    cfg = json.loads((RUNS / run / "config.json").read_text(encoding="utf-8"))
    moved = {k: [v, cfg.get(k)] for k, v in FROZEN.items() if cfg.get(k) != v}
    if moved:
        raise SystemExit(
            f"G5: {run} did not train under the frozen Phase-2 recipe. Fields "
            f"that moved (pre-registered -> realised): {moved}. EXP_017 §0 says "
            "this experiment changes no hyperparameter; a run that did is not "
            "this experiment."
        )
    return {"frozen_recipe_verified": True, "n_fields_checked": len(FROZEN)}


def check_same_arm(run: str, reference: str | None, may_differ: frozenset) -> dict:
    """G3 (K1). Returns the row; raises if anything differs that may not."""
    cfg = json.loads((RUNS / run / "config.json").read_text(encoding="utf-8"))
    if reference is None:
        return {"run": run, "reference": None, "same_arm": True,
                "why": "this run IS the reference for its leg"}
    ref = json.loads((RUNS / reference / "config.json").read_text(encoding="utf-8"))
    diffs = {k: [ref.get(k), cfg.get(k)]
             for k in set(ref) | set(cfg) if ref.get(k) != cfg.get(k)}

    excused_as_default, unexpected = [], {}
    for k, (ref_v, new_v) in diffs.items():
        if k in may_differ:
            continue
        # Retained from EXP_014's and EXP_015's drivers so a Config field added
        # between this run and the reference cannot fail G3 spuriously.
        if ref_v is None and k in _DEFAULTS and new_v == _DEFAULTS[k]:
            excused_as_default.append(k)
            continue
        unexpected[k] = [ref_v, new_v]
    if unexpected:
        raise SystemExit(
            f"G3: {run} is not {reference} plus one thing. Fields that differ "
            f"but may not: {unexpected}"
        )
    return {
        "run": run, "reference": reference,
        "differs_in": sorted(k for k in diffs if k in may_differ),
        "absent_from_reference_and_left_at_default": sorted(excused_as_default),
        "same_arm": True,
    }


def check_params_and_vram(run: str, arch: str, d_model: int) -> dict:
    """G2 and G6. Both abort.

    G2 is stronger here than in any previous experiment and the strength is the
    point: `twocomp_detach` adds no parameter of any shape, so its count must
    equal `twocomp`'s **exactly**, not to within `EXP_015`'s 0.18%. That is what
    lets Leg B's number be read straight against `EXP_015`'s table.
    """
    summary = json.loads((RUNS / run / "summary.json").read_text(encoding="utf-8"))
    expected = param_count(arch, _VOCAB, d_model, _LAYERS)
    twocomp_expected = param_count("twocomp", _VOCAB, d_model, _LAYERS)
    realised = int(summary.get("params", -1))
    if realised != expected:
        raise SystemExit(
            f"G2: {run} realised {realised:,} parameters, pre-registered "
            f"{expected:,}. Parameter identity is the premise of this comparison; "
            "a mismatch removes it silently."
        )
    if realised != twocomp_expected:
        raise SystemExit(
            f"G2: {run} realised {realised:,} parameters against `twocomp`'s "
            f"{twocomp_expected:,} at the same width. EXP_017 §2.3 claims the arm "
            "adds no parameter; this says otherwise."
        )
    peak = float(summary.get("peak_vram_gib", -1.0))
    if peak > VRAM_ALARM_GIB:
        raise SystemExit(
            f"G6: {run} peaked at {peak} GiB, past the {VRAM_ALARM_GIB} GiB "
            "alarm. Under WDDM a spill is a ~50x slowdown that does not raise, "
            "so this is investigated rather than absorbed."
        )
    return {"params": realised, "params_expected": expected,
            "params_equal_twocomp": realised == twocomp_expected,
            "peak_vram_gib": peak, "wall_clock_s": summary.get("wall_clock_s")}


def scan_for_divergence(run: str) -> dict:
    """G7, and H2's actual measurement.

    A run can finish cleanly and still have gone non-finite -- nothing in the
    trainer treats NaN as an error, so it exits 0 and writes a checkpoint. So the
    exit code cannot answer this and the log has to. `log_present` is required
    rather than assumed: an *absent* log would otherwise read as a *clean* one,
    which is the second half of the defect `EXP_015` §9.10 records.

    `passed_step_5138` is this experiment's headline reduced to one boolean. The
    adopted arm at this width died there, deterministically, three times.
    """
    p = RUNS / run / "log.jsonl"
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
    last = train[-1]["step"] if train else None
    return {
        "log_present": True,
        "n_train_records": len(train),
        "last_logged_step": last,
        "n_nonfinite_loss_records": len(bad),
        "first_nonfinite_step": bad[0]["step"] if bad else None,
        "passed_step_5138": bool(last is not None and last > 5138 and not bad),
        "max_grad_norm": max((r.get("grad_norm", 0.0) for r in train), default=None),
        "n_logged_steps_over_clip": sum(
            1 for r in train if (r.get("grad_norm") or 0.0) > 1.0),
    }


def leg_completed(returncode: int, scan: dict, ckpt_exists: bool) -> bool:
    """Did this run produce a result that may be scored against a bar?

    `EXP_015`'s driver got this wrong and recorded two dead legs as
    `completed: true`; the corrected form is inherited here verbatim rather than
    re-derived, because re-deriving it is how the same defect comes back.
    """
    return bool(returncode == 0 and ckpt_exists
                and scan.get("log_present")
                and not scan.get("n_nonfinite_loss_records"))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--legs", default="B,C")
    ap.add_argument("--out", default="docs/reports/data/exp_017_run_manifest.json")
    args = ap.parse_args(argv)

    want = {s.strip() for s in args.legs.split(",") if s.strip()}
    plan = [p for p in PLAN if p[0] in want]

    prereg_before = prereg_sha256()
    manifest: dict = {
        "experiment": "EXP_017_bounded_reset_jacobian",
        "prereg": str(PREREG.relative_to(_REPO)).replace("\\", "/"),
        "prereg_sha256_before_first_run": prereg_before,
        "legs": sorted(want),
        "frozen_recipe": {k: v for k, v in FROZEN.items()},
        "vram_alarm_gib": VRAM_ALARM_GIB,
        "adopts_nothing": True,
        "ranks_nothing": True,
        "reads_no_bpc": True,
        "changes_no_hyperparameter": True,
        "runs": [],
    }
    out = Path(args.out)
    if not out.is_absolute():
        out = _REPO / out
    out.parent.mkdir(parents=True, exist_ok=True)

    def flush() -> None:
        out.write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")

    flush()

    for leg, run, arch, d_model, seed, reference, may_differ in plan:
        row: dict = {"leg": leg, "run": run, "arch": arch,
                     "d_model": d_model, "seed": seed}
        if is_complete(run):
            print(f"\n=== SKIP {run}: already complete", flush=True)
            row["skipped_as_complete"] = True
        else:
            if (RUNS / run).exists():
                print(f"\n=== DELETE partial {run} (never resumed)", flush=True)
                shutil.rmtree(RUNS / run)
                row["deleted_partial"] = True
            _check_no_campaign()
            dt, rc = _run(_train_argv(run, arch, d_model, seed),
                          f"train {run}", fatal=False)
            row["train_wall_clock_s"] = round(dt, 1)
            row["train_returncode"] = rc
            scan = scan_for_divergence(run)
            row["divergence_scan"] = scan
            ckpt_exists = (RUNS / run / "ckpt_final.pt").exists()
            if rc != 0 or not ckpt_exists:
                row["completed"] = False
                row["note"] = ("training did not finish; this run's prediction "
                               "resolves NOT RUN. Not reseeded -- "
                               "CONTRIBUTING.md §4.")
                manifest["runs"].append(row)
                flush()
                continue
            _check_no_campaign()
            # A diverged run is still evaluated, deliberately: the NaN in
            # `final_test.json` is evidence, and suppressing it would leave the
            # record showing an arm with no number rather than an arm that
            # produced one and it was NaN.
            ev_dt, ev_rc = _run(_eval_argv(run), f"evaluate {run}", fatal=False)
            row["eval_wall_clock_s"] = round(ev_dt, 1)
            row["eval_returncode"] = ev_rc
            if ev_rc != 0:
                row["completed"] = False
                row["note"] = "evaluation failed; this run's prediction is NOT RUN"
                manifest["runs"].append(row)
                flush()
                continue

        scan = row.get("divergence_scan") or scan_for_divergence(run)
        row["divergence_scan"] = scan
        row["diverged"] = bool(scan.get("n_nonfinite_loss_records"))
        row["completed"] = leg_completed(
            row.get("train_returncode", 0), scan,
            (RUNS / run / "ckpt_final.pt").exists())
        if row["diverged"]:
            row["note"] = (
                f"DIVERGED: {scan['n_nonfinite_loss_records']} of "
                f"{scan['n_train_records']} logged losses non-finite, first at "
                f"step {scan['first_nonfinite_step']}. Not reseeded -- "
                "CONTRIBUTING.md §4."
            )
        row["g3"] = check_same_arm(run, reference, may_differ)
        row["g5"] = check_frozen(run)
        row.update(check_params_and_vram(run, arch, d_model))
        manifest["runs"].append(row)
        flush()
        print(f"    {run}: completed={row['completed']} "
              f"diverged={row['diverged']} "
              f"passed_5138={scan.get('passed_step_5138')}", flush=True)

    prereg_after = prereg_sha256()
    manifest["prereg_sha256_after_last_run"] = prereg_after
    manifest["prereg_unchanged"] = prereg_after == prereg_before
    flush()
    if not manifest["prereg_unchanged"]:
        raise SystemExit(
            "The pre-registration changed while the ladder was running. The "
            "experiment is void; nothing here may be reported."
        )
    done = sum(1 for r in manifest["runs"] if r.get("completed"))
    print(f"\nwrote {out}")
    print("EXP017_ALL_DONE", done, "/", len(plan), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
