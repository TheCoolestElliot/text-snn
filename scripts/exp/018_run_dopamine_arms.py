"""EXP_018 legs 1-4: the anchor, the dopamine arm, its control, and the additive form.

    python scripts/launch.py --script scripts/exp/018_run_dopamine_arms.py \
        --run-name _exp018_driver -- --out docs/reports/data/exp_018_run_manifest.json

WHY THIS IS A DRIVER AND NOT SIXTEEN COMMANDS
----------------------------------------------
`CONTRIBUTING.md` §6: long runs die under an interactive timeout, and a driver
must be restartable -- skip what is complete, **delete** what is partial rather
than resuming it, so no artifact ever describes two runs at once. Sixteen
hand-typed commands cannot do the second thing, and the failure is silent: a
resumed run writes a `summary.json` whose step count looks right and whose
trajectory is two runs spliced.

It also exists so that "one thing changed" is a *check* rather than a claim.
Every arm's `config.json` is diffed against its leg's reference run, and
anything outside the declared `may_differ` set aborts the ladder.

WHAT IT REFUSES TO DO
---------------------
  * It reads no bpc and resolves no bar. `018_dopamine_results.py` does that,
    and is committed before any run's bpc is read.
  * It changes no hyperparameter, and **proves it**: G5 reads all 24 frozen
    recipe terms back out of each run's own `config.json`.
  * It does not stop when a run diverges. A divergence is evidence; the run is
    recorded, evaluated anyway (the NaN is the finding), and the ladder
    continues.
  * **It does not reseed.** A failed seed is reported at reduced n and marked
    provisional (`CONTRIBUTING.md` §4); a promotion is by seeds *added*, never
    by one swapped in for a failure.
  * It adopts nothing and ranks nothing.

THE PRE-REGISTRATION HASH
-------------------------
`EXP_018` §8 makes the log's SHA-256 an entry condition. It is stamped into the
manifest before the first run and re-checked after the last; a mismatch **voids
the experiment** and this script raises rather than writing a manifest that
would look complete.
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
from snn.model import dopamine_param_count, spiking_param_count  # noqa: E402

RUNS = _REPO / "experiments" / "runs"
LOCKFILE = _REPO / "src" / "snn" / ".MUTATION_CAMPAIGN_RUNNING"
PREREG = _REPO / "experiments" / "logs" / "EXP_018_dopamine.md"
CALIB = _REPO / "docs" / "reports" / "data" / "exp_018_da_calibration.json"

_DEFAULTS = {f.name: f.default for f in dataclasses.fields(Config)}

D_MODEL = 512
LAYERS = 2

#: Peak VRAM measured at 0.98 GiB for the widest variant (EXP_018 §2.5) on an
#: 8 GiB card. An alarm rather than a limit: something has changed structurally
#: if a 735K arm is using four.
VRAM_ALARM_GIB = 4.0

#: tau, transcribed from the calibration artifact. Asserted against the file
#: below rather than trusted, so a driver that drifts from its own calibration
#: cannot start a ladder.
TAU = 1.1291

#: The frozen Phase-2 recipe, all 24 terms. Read back out of every run's own
#: config.json by G5. EXP_018 §0 says this experiment changes no hyperparameter;
#: a run that did is not this experiment.
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
    "d_model": 512,
    "da_scale": TAU,
    "da_gain_init": 0.0,
}

ANCHOR_SEEDS = (0, 1, 2, 3, 4)
ARM_SEEDS = (0, 1, 2, 3, 4)
CTRL_SEEDS = (0, 1, 2)

#: (leg, run_name, arch, da_mode, da_source, seed, reference_run, may_differ).
#:
#: ORDER IS LOAD-BEARING TWICE, the same two ways `EXP_017`'s is. The anchor
#: runs first because every other leg names it as a reference and a reference
#: must exist before the run that is diffed against it. Within the arms, the
#: headline (leg 2) precedes its control (leg 3) and the marker (leg 4), because
#: leg 2 is the one whose result could make the rest not worth the GPU time.
#:
#: Run names encode the arm rather than only the seed, for `EXP_013`'s reason:
#: "a ladder whose points are distinguishable only by reading each config.json
#: is a ladder someone will eventually mis-attribute."
PLAN: list[tuple] = (
    [("1", f"da_anchor_s{s}", "snn", "mult", "off", s,
      None if s == 0 else "da_anchor_s0",
      frozenset() if s == 0 else frozenset({"run_name", "seed"}))
     for s in ANCHOR_SEEDS]
    + [("2", f"da_mult_s{s}", "dopamine", "mult", "rpe", s, "da_anchor_s0",
        frozenset({"arch", "run_name", "seed", "da_source"}))
       for s in ARM_SEEDS]
    + [("3", f"da_rolled_s{s}", "dopamine", "mult", "rolled", s, "da_mult_s0",
        frozenset({"run_name", "seed", "da_source"}))
       for s in CTRL_SEEDS]
    + [("4", f"da_add_s{s}", "dopamine", "add", "rpe", s, "da_mult_s0",
        frozenset({"run_name", "seed", "da_mode"}))
       for s in CTRL_SEEDS]
)


def prereg_sha256() -> str:
    return hashlib.sha256(PREREG.read_bytes()).hexdigest()


def check_tau_matches_calibration() -> dict:
    """The ladder's tau must be the one the calibration artifact recommends.

    A transcribed constant that has silently drifted from the measurement it
    descends from is the defect `017_detach_results.py`'s header warns about,
    one level up: the bar is right and the thing it was computed from moved.
    """
    if not CALIB.exists():
        raise SystemExit(
            f"entry condition 2: {CALIB.relative_to(_REPO)} is missing. tau is a "
            "measured constant and the ladder may not start without it.")
    got = float(json.loads(CALIB.read_text(encoding="utf-8"))["tau_recommended"])
    if abs(got - TAU) > 1e-9:
        raise SystemExit(
            f"T1: this driver uses tau = {TAU} and the calibration artifact "
            f"recommends {got}. One of them has drifted; the ladder does not start.")
    return {"tau": TAU, "matches_calibration": True}


def is_complete(run: str) -> bool:
    """True iff `run` finished its full budget AND was scored on test.

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
            f"{LOCKFILE.relative_to(_REPO)} exists -- a mutation campaign is "
            "applying wrong kernels to the source tree. Refusing to start a run "
            "that would import one (03_phase3_candidates.md §8).")


def _train_argv(run: str, arch: str, mode: str, source: str, seed: int) -> list[str]:
    """NOTE WHAT IS ABSENT: no --lr, no --grad_clip, no --weight_decay, no
    --da_scale and no --da_gain_init. They are Config defaults, and G5 asserts
    afterwards that none of them moved anyway."""
    argv = [
        str(_REPO / "scripts" / "train.py"),
        "--arch", arch,
        "--d_model", str(D_MODEL),
        "--n_layers", str(LAYERS),
        "--run_name", run,
        "--seed", str(seed),
        "--eval_every", "2500",
        "--eval_max_windows", "2560",
        "--ckpt_every", "5000",
        "--log_every", "250",
    ]
    if arch == "dopamine":
        # Passed explicitly rather than left to a default, so the run's own
        # config.json records the arm that trained. A dopamine run that silently
        # took da_source="off" would train the baseline and be indistinguishable
        # from a null -- the failure NoisyCharLM raises to prevent.
        argv += ["--da_mode", mode, "--da_source", source]
    return argv


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
    """G5. Every frozen term, read back out of the run's own config.json."""
    cfg = json.loads((RUNS / run / "config.json").read_text(encoding="utf-8"))
    moved = {k: [v, cfg.get(k)] for k, v in FROZEN.items() if cfg.get(k) != v}
    if moved:
        raise SystemExit(
            f"G5: {run} did not train under the frozen Phase-2 recipe. Fields "
            f"that moved (pre-registered -> realised): {moved}. EXP_018 §0 says "
            "this experiment changes no hyperparameter; a run that did is not "
            "this experiment.")
    return {"frozen_recipe_verified": True, "n_fields_checked": len(FROZEN)}


def check_same_arm(run: str, reference: str | None, may_differ: frozenset) -> dict:
    """One thing changed, checked rather than claimed.

    A field the reference predates and this run leaves at the `Config` default is
    excused and RECORDED, never swallowed -- the reference simply had no opinion
    about it.
    """
    if reference is None:
        return {"reference": None, "note": "this run IS the reference"}
    a = json.loads((RUNS / reference / "config.json").read_text(encoding="utf-8"))
    b = json.loads((RUNS / run / "config.json").read_text(encoding="utf-8"))
    excused, differ = [], {}
    for key in sorted(set(a) | set(b)):
        if key in may_differ or a.get(key) == b.get(key):
            continue
        if key not in a and b.get(key) == _DEFAULTS.get(key):
            excused.append(key)
            continue
        differ[key] = [a.get(key), b.get(key)]
    if differ:
        raise SystemExit(
            f"K1: {run} differs from {reference} in more than the declared "
            f"variable. Undeclared differences: {differ}. may_differ was "
            f"{sorted(may_differ)}.")
    return {"reference": reference, "may_differ": sorted(may_differ),
            "absent_from_reference_and_left_at_default": excused}


def check_params_and_vram(run: str, arch: str) -> dict:
    """The realised parameter count against its committed closed form, and the
    VRAM alarm. A count that disagrees means the run is not the arm it says."""
    summary = json.loads((RUNS / run / "summary.json").read_text(encoding="utf-8"))
    cfg = json.loads((RUNS / run / "config.json").read_text(encoding="utf-8"))
    expect = (dopamine_param_count if arch == "dopamine" else spiking_param_count)(
        int(cfg["vocab_size"]), D_MODEL, LAYERS)
    got = int(summary.get("params", -1))
    if got != expect:
        raise SystemExit(
            f"G2: {run} has {got} parameters, closed form says {expect}. The run "
            "is not the arm its name claims.")
    vram = float(summary.get("peak_vram_gib", 0.0))
    return {"params": got, "params_closed_form": expect,
            "peak_vram_gib": vram, "vram_alarm": vram > VRAM_ALARM_GIB,
            "wall_clock_s": summary.get("wall_clock_s")}


def scan_for_divergence(run: str) -> dict:
    """Read the run's own log.jsonl. `log_present` is required, not assumed --
    an ABSENT log would otherwise read as a CLEAN one."""
    path = RUNS / run / "log.jsonl"
    out = {"log_present": path.exists(), "n_train_records": 0,
           "last_logged_step": None, "n_nonfinite_loss_records": 0,
           "first_nonfinite_step": None, "max_grad_norm": None,
           "n_logged_steps_over_clip": 0}
    if not path.exists():
        return out
    import math
    gmax = None
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if rec.get("event") != "train":
            continue
        out["n_train_records"] += 1
        out["last_logged_step"] = rec.get("step")
        loss = rec.get("loss")
        if loss is None or not math.isfinite(float(loss)):
            out["n_nonfinite_loss_records"] += 1
            if out["first_nonfinite_step"] is None:
                out["first_nonfinite_step"] = rec.get("step")
        g = rec.get("grad_norm")
        if g is not None and math.isfinite(float(g)):
            gmax = float(g) if gmax is None else max(gmax, float(g))
            if float(g) > FROZEN["grad_clip"]:
                out["n_logged_steps_over_clip"] += 1
    out["max_grad_norm"] = gmax
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="EXP_018 training ladder")
    ap.add_argument("--legs", default="1,2,3,4")
    ap.add_argument("--out", default="docs/reports/data/exp_018_run_manifest.json")
    args = ap.parse_args(argv)

    legs = {s.strip() for s in args.legs.split(",") if s.strip()}
    plan = [row for row in PLAN if row[0] in legs]

    out = Path(args.out)
    if not out.is_absolute():
        out = _REPO / out
    out.parent.mkdir(parents=True, exist_ok=True)

    prereg_before = prereg_sha256()
    manifest = {
        "experiment": "EXP_018",
        "prereg": str(PREREG.relative_to(_REPO)),
        "prereg_sha256_before_first_run": prereg_before,
        "legs": sorted(legs),
        "frozen_recipe": FROZEN,
        "calibration": check_tau_matches_calibration(),
        "vram_alarm_gib": VRAM_ALARM_GIB,
        "adopts_nothing": True,
        "ranks_nothing": True,
        "reads_no_bpc": True,
        "changes_no_hyperparameter": True,
        "does_not_reseed": True,
        "runs": [],
    }

    def flush() -> None:
        out.write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")

    flush()   # written BEFORE any run, so a killed ladder still has a manifest

    done = 0
    for leg, run, arch, mode, source, seed, reference, may_differ in plan:
        row = {"leg": leg, "run": run, "arch": arch, "da_mode": mode,
               "da_source": source, "seed": seed}
        if is_complete(run):
            row["skipped_as_complete"] = True
            rc, scan = 0, scan_for_divergence(run)
        else:
            if (RUNS / run).exists():
                # DELETE rather than resume: a resumed run's summary.json would
                # describe two runs at once (CONTRIBUTING.md §6).
                shutil.rmtree(RUNS / run)
                row["deleted_partial"] = True
            _check_no_campaign()
            dt, rc = _run(_train_argv(run, arch, mode, source, seed),
                          f"train {run}", fatal=False)
            row["train_seconds"] = round(dt, 1)
            scan = scan_for_divergence(run)
            if rc != 0 or not (RUNS / run / "ckpt_final.pt").exists():
                row["completed"] = False
                row["divergence_scan"] = scan
                row["note"] = ("training did not produce ckpt_final.pt; this leg "
                               "resolves NOT RUN. Not reseeded -- CONTRIBUTING.md §4.")
                manifest["runs"].append(row)
                flush()
                print(f"    {run}: NOT RUN", flush=True)
                continue
            _check_no_campaign()
            ev_dt, ev_rc = _run(_eval_argv(run), f"evaluate {run}", fatal=False)
            row["eval_seconds"] = round(ev_dt, 1)
            row["eval_returncode"] = ev_rc

        row["divergence_scan"] = scan
        row["diverged"] = bool(scan.get("n_nonfinite_loss_records"))
        row["completed"] = is_complete(run)
        row["k1"] = check_same_arm(run, reference, may_differ)
        row["g5"] = check_frozen(run)
        row.update(check_params_and_vram(run, arch))
        manifest["runs"].append(row)
        flush()
        done += int(bool(row["completed"]))
        print(f"    {run}: completed={row['completed']} diverged={row['diverged']} "
              f"params={row['params']} vram={row['peak_vram_gib']:.3f}", flush=True)

    prereg_after = prereg_sha256()
    manifest["prereg_sha256_after_last_run"] = prereg_after
    manifest["prereg_unchanged"] = prereg_after == prereg_before
    flush()
    if not manifest["prereg_unchanged"]:
        raise SystemExit(
            "The pre-registration changed while the ladder was running. The "
            "experiment is void; nothing here may be reported.")
    print(f"\nEXP018_ALL_DONE {done} / {len(plan)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
