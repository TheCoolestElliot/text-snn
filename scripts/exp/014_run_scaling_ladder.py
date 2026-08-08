"""EXP_014 driver -- the parameter-scaling ladder, one rung at a time.

    python scripts/launch.py --script scripts/exp/014_run_scaling_ladder.py \
        --run-name _exp014_driver -- --out docs/reports/data/exp_014_run_manifest.json

Pre-registered in `experiments/logs/EXP_014_parameter_scaling.md`; read that
first, because every threshold this file enforces was fixed there before this
file ran.

WHAT IT DOES
------------
Trains `arch="snn"` at three widths -- 512, 1020, 1481 -- at seed 0, then scores
each on the test split.  512 is the committed baseline's width and is retrained
here on purpose; that retrain is gate **G1** (§7), not redundancy.

The contracts below are `013_run_noise_arms.py`'s, kept deliberately identical
so that "restartable" means the same thing in both files:

* **Strictly sequential.**  `subprocess.run` blocks on each child, so "one run at
  a time" is a property of the code rather than of who typed the commands.  A
  second run sharing the card would corrupt every wall-clock figure K3 reports.
* **Restartable, and it deletes rather than resumes.**  A run counts as complete
  only when `summary.json` is at the full step count *and* `final_test.json`
  exists.  Anything else is removed and retrained, so that no artifact ever
  describes two runs at once (`CONTRIBUTING.md` §6).
* **K2 before every child**, not once at the top: a mutation campaign that starts
  mid-ladder would otherwise supply wrong kernels to every later run.
* **The manifest is rewritten after every rung**, so a crash at rung 3 still
  leaves rungs 1-2 fully described.

WHAT IS NEW HERE
----------------
`G3` asserts each run's realised parameter count against the pre-registered one:
the parameter count is this experiment's x-axis, and an off-by-one width would
move it silently.  `G4` aborts on a VRAM peak past 3.0 GiB, because
`docs/chat/BUILD_NOTES.md` §2 records that a spill under Windows' WDDM is a ~50x
slowdown that does not raise -- it would corrupt a wall-clock figure while
looking merely slow.  `K3` compares realised wall-clock against
`exp_014_cost_calibration.json`'s 400-step projection; unlike every previous
experiment's systems row this is a measurement rather than a sameness check,
because no width-cost measurement existed in this project before it.
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
from snn.model import spiking_param_count  # noqa: E402

RUNS = _REPO / "experiments" / "runs"
LOCKFILE = _REPO / "src" / "snn" / ".MUTATION_CAMPAIGN_RUNNING"
PREREG = _REPO / "experiments" / "logs" / "EXP_014_parameter_scaling.md"
CALIB = _REPO / "docs" / "reports" / "data" / "exp_014_cost_calibration.json"

#: §2.2.  512 is the committed width (G1); 1020 and 1481 are the
#: exact-error-minimising integers for 2.5M and 5M parameters.
WIDTHS: tuple[int, ...] = (512, 1020, 1481)
SEED = 0

#: §2.3.  `seed` is deliberately NOT here -- every rung is seed 0, which makes
#: K1 a stronger claim than any previous Phase-4 experiment's.
MAY_DIFFER = {"d_model", "run_name"}

REFERENCE_RUN = "snn_beta0.5_s0"

#: §6, G4.  The calibration measured 1.643 GiB at the widest rung, on a 7.96 GiB
#: card, so this is an alarm and not a budget.
VRAM_ALARM_GIB = 3.0

_VOCAB = 205
_LAYERS = 2

_DEFAULTS = {f.name: f.default for f in dataclasses.fields(Config)}


def prereg_sha256() -> str:
    return hashlib.sha256(PREREG.read_bytes()).hexdigest()


def run_name(d: int) -> str:
    return f"scale_d{d}_s{SEED}"


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
            "that would import one (03_phase3_candidates.md §8)."
        )


def _train_argv(d: int) -> list[str]:
    """The reference run's launch command with `d_model` and `run_name` changed,
    and nothing else."""
    return [
        str(_REPO / "scripts" / "train.py"),
        "--arch", "snn",
        "--d_model", str(d),
        "--n_layers", str(_LAYERS),
        "--run_name", run_name(d),
        "--seed", str(SEED),
        "--eval_every", "2500",
        "--eval_max_windows", "2560",
        "--ckpt_every", "5000",
        "--log_every", "250",
    ]


def _eval_argv(d: int) -> list[str]:
    run = run_name(d)
    return [
        str(_REPO / "scripts" / "evaluate.py"),
        "--ckpt", str(RUNS / run / "ckpt_final.pt"),
        "--split", "test",
        "--out", str(RUNS / run / "final_test.json"),
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


def check_same_arm(d: int, reference: dict) -> dict:
    """K1 (§7). Returns the row; raises if anything differs that may not."""
    run = run_name(d)
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
    if int(cfg.get("d_model", -1)) != int(d):
        raise SystemExit(
            f"K1: {run} records d_model={cfg.get('d_model')!r}, expected {d}"
        )
    if int(cfg.get("seed", -1)) != SEED:
        raise SystemExit(
            f"K1: {run} records seed={cfg.get('seed')!r}, expected {SEED} -- "
            "§2.3 holds every rung at one seed, so a differing seed is a "
            "different experiment"
        )
    return {
        "run": run,
        "arch": "snn",
        "d_model": int(d),
        "seed": SEED,
        "differs_in": sorted(k for k in diffs if k in MAY_DIFFER),
        "absent_from_reference_and_left_at_default": sorted(excused_as_default),
        "same_arm": True,
    }


def check_params_and_vram(d: int) -> dict:
    """G3 and G4 (§6). Both abort."""
    run = run_name(d)
    summary = json.loads((RUNS / run / "summary.json").read_text(encoding="utf-8"))
    expected = spiking_param_count(_VOCAB, d, _LAYERS)
    realised = int(summary.get("params", -1))
    if realised != expected:
        raise SystemExit(
            f"G3: {run} realised {realised:,} parameters, pre-registered "
            f"{expected:,}. The parameter count is this experiment's x-axis; a "
            "mismatch moves it silently."
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


def check_g1() -> dict:
    """G1 (§7). Does today's tree still produce the committed baseline?

    Before decision #7, `src/snn/train.py` had exactly one commit, so every
    committed baseline was trained with the stock `clip_grad_norm_`. This
    ladder's bottom rung would otherwise be the only rung trained on a different
    `train.py`. Bitwise, not a tolerance -- `EXP_013` §9.8 demonstrated 20,000
    steps of a *stochastic* arm reproducing bit-exactly, so anything looser
    would be a claim that the two models differ.
    """
    # `evaluate.py` nests both protocols under "results"; the committed figures
    # quoted in the reports (2.26969 / 2.25311) are the FIVE-SEED MEANS, not
    # this run's own. G1 compares seed 0 against seed 0.
    new = json.loads(
        (RUNS / run_name(512) / "final_test.json").read_text(encoding="utf-8"))["results"]
    old = json.loads(
        (RUNS / REFERENCE_RUN / "final_test.json").read_text(encoding="utf-8"))["results"]

    row: dict = {"gate": "G1", "reference": REFERENCE_RUN, "retrain": run_name(512),
                 "compares": "seed 0 against seed 0, not against the 5-seed mean"}
    for protocol in ("fresh", "carried"):
        a = new.get(protocol, {}).get("bpc")
        b = old.get(protocol, {}).get("bpc")
        row[protocol] = {"retrain": a, "committed": b, "bitwise_equal": a == b,
                         "delta": (None if a is None or b is None else a - b)}
    row["passes"] = all(row[p]["bitwise_equal"] for p in ("fresh", "carried"))
    return row


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--widths", default=",".join(str(w) for w in WIDTHS))
    ap.add_argument("--out", default="docs/reports/data/exp_014_run_manifest.json")
    args = ap.parse_args(argv)

    widths = [int(w) for w in args.widths.split(",") if w.strip()]
    reference = json.loads(
        (RUNS / REFERENCE_RUN / "config.json").read_text(encoding="utf-8"))

    prereg_before = prereg_sha256()
    calib = json.loads(CALIB.read_text(encoding="utf-8")) if CALIB.exists() else None
    projected = {}
    if calib:
        for r in calib.get("results", []):
            if "projected_20k_s" in r:
                projected[int(r["d_model"])] = r["projected_20k_s"]

    manifest: dict = {
        "experiment": "EXP_014_parameter_scaling",
        "prereg": str(PREREG.relative_to(_REPO)).replace("\\", "/"),
        "prereg_sha256_before_first_run": prereg_before,
        "reference_run": REFERENCE_RUN,
        "may_differ": sorted(MAY_DIFFER),
        "seed": SEED,
        "widths": widths,
        "vram_alarm_gib": VRAM_ALARM_GIB,
        "adopts_nothing": True,
        "ranks_nothing": True,
        "rungs": [],
    }
    out = Path(args.out)
    if not out.is_absolute():
        out = _REPO / out
    out.parent.mkdir(parents=True, exist_ok=True)

    def flush() -> None:
        out.write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")

    for d in widths:
        run = run_name(d)
        row: dict = {"d_model": d, "run": run}
        if is_complete(run):
            print(f"\n=== SKIP {run}: already complete", flush=True)
            row["skipped_as_complete"] = True
        else:
            if (RUNS / run).exists():
                print(f"\n=== DELETE partial {run} (never resumed)", flush=True)
                shutil.rmtree(RUNS / run)
                row["deleted_partial"] = True
            _check_no_campaign()
            row["train_wall_clock_s"] = round(_run(_train_argv(d), f"train {run}"), 1)
            _check_no_campaign()
            row["eval_wall_clock_s"] = round(_run(_eval_argv(d), f"evaluate {run}"), 1)

        row["k1"] = check_same_arm(d, reference)
        row.update(check_params_and_vram(d))
        if d in projected:
            row["k3"] = {
                "projected_20k_s_from_calibration": projected[d],
                "realised_train_wall_clock_s": row.get("wall_clock_s"),
                "ratio_realised_over_projected": (
                    None if not row.get("wall_clock_s")
                    else round(row["wall_clock_s"] / projected[d], 4)),
                "note": ("the projection is a 400-step per-step rate scaled by "
                         "the committed 398.88 s; it excludes capture cost and "
                         "the eight in-training evaluations, so a ratio above 1 "
                         "is expected and its size is the measurement"),
            }
        manifest["rungs"].append(row)
        flush()

    if 512 in widths:
        manifest["g1"] = check_g1()
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
    print(f"\nwrote {out}")
    print("EXP014_ALL_DONE", len(manifest["rungs"]), "/", len(widths), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
