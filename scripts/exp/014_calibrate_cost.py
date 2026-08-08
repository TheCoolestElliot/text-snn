"""EXP_014 entry condition: price width, and fix the ladder's two new widths.

    python scripts/exp/014_calibrate_cost.py --out docs/reports/data/exp_014_cost_calibration.json

WHY THIS RUNS BEFORE THE LADDER
-------------------------------
`03_phase3_candidates.md` §6.3 #11 prices the parameter-scaling pilot at 0.35
GPU-hours and **no measurement in this tree supports that number**.  It descends
from `01_reconnaissance.md` §3.5's width sweep, which is eager, forward-only and
under `no_grad` -- and `02_baseline_report.md` §6.2, titled "The 14.5 us constant
does not transfer", withdrew that section's cost model and closed by instructing
Phase 3's ROI estimates not to reuse it.  `EXP_003`'s depth ladder cannot price
width either: it holds `K*d^2` nearly constant, so `K`, `K*d` and `K*d^2` are
collinear across its four points and a fit over them is degenerate.

So this file measures what the ladder will cost instead of asserting it, in the
same spirit as `013_calibrate_noise_scale.py`, which measured `sd(cur)` before
fixing an amplitude ladder rather than choosing three round numbers.

WHAT IT MEASURES, AND WHY EACH THING IS HERE
--------------------------------------------
For each candidate width: steps/s, peak VRAM, the firing rate at step 0 and at
the end, the step at which layer 1 stops being silent, and the largest gradient
norm seen.

* **steps/s** converts §7.1's GPU-h column from an assumption into a number.
* **peak VRAM** guards `docs/chat/BUILD_NOTES.md` §2's trap: under Windows' WDDM
  a spill past the card does not fail, it becomes ~50x slower.  A spill would
  therefore corrupt a wall-clock figure while looking like a slow run.
* **firing rates and the escape step** are `audit_07_init_pathology.json`'s
  measurement repeated at the widths this experiment will actually train.  That
  artifact already carries a `d=1024` row; the widths above it are untested.
* **max grad norm** bears on whether the clip fires at width.  It never exceeds
  0.384 across `snn_beta0.5_s0`'s logged steps.

WHY MEASURING FIRST DOES NOT CONTAMINATE THE PRE-REGISTRATION
--------------------------------------------------------------
The widths are chosen from **wall-clock**, and the experiment's decision variable
is **bpc**.  Nothing here reads a bpc against any bar, and no run scored here
survives -- every calibration directory is deleted once its numbers are
extracted.  Choosing a config from a measurement is normally what pre-registration
forbids; it is admissible here precisely because the measured quantity and the
decided quantity are disjoint, and this paragraph exists so that argument is on
the record rather than assumed.

The two nominal targets are 2.5M and 5M parameters.  Both widths are the
exact-error-minimising integers, which is `EXP_003` §4's own convention (widths
derived from a parameter target, not picked round).  Odd widths are included
because `d = 1481` is 4-byte aligned only, and a GEMM shape cliff there is
possible and unmeasured; if one appears, the even fallbacks are what the ladder
uses instead.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO / "src") not in sys.path:
    sys.path.insert(0, str(_REPO / "src"))

from snn.model import spiking_param_count  # noqa: E402

#: Candidate widths.  512 is the committed baseline and anchors the ratios.
#: 1020/1481 are the exact-error-minimising integers for 2.5M/5M parameters;
#: 1024/1480/1488 are the aligned fallbacks tested for a GEMM shape cliff.
WIDTHS: tuple[int, ...] = (512, 1020, 1024, 1480, 1481, 1488)

#: Long enough to clear `warmup_steps = 200` and leave steady-state records,
#: short enough that the whole probe is minutes.  No evaluation runs.
CALIB_STEPS = 400
LOG_EVERY = 25

#: A spill under WDDM is a slowdown, not a failure, so it needs a threshold
#: rather than an exception.  The modelled top rung is ~1.5 GiB on 7.96 total.
VRAM_ALARM_GIB = 3.0

#: Layer 1's firing rate at the first logged step is ~3e-07, not 0.0 -- a
#: handful of spikes out of millions, which is `audit_07_init_pathology.json`'s
#: silent layer as it actually appears in a training log rather than at init.
#: "First step with a rate above zero" is therefore step 0 at every width and
#: discriminates nothing; the escape has to be measured against a rate that
#: means the layer is carrying signal.  1 % is two orders below the ~35 % these
#: runs settle at and two orders above the initial leak.
ESCAPE_RATE = 0.01

_VOCAB = 205
_LAYERS = 2


def _calib_dir() -> Path:
    return _REPO / "experiments" / "runs" / "_exp014_cost"


def _run_one(d: int) -> dict:
    """Train `CALIB_STEPS` steps at width `d` and read its own log back."""
    name = f"calib_d{d}"
    out_dir = _calib_dir()
    run_dir = out_dir / name
    if run_dir.exists():
        shutil.rmtree(run_dir)

    cmd = [
        sys.executable, str(_REPO / "scripts" / "train.py"),
        "--arch", "snn",
        "--d_model", str(d),
        "--n_layers", str(_LAYERS),
        "--seed", "0",
        "--max_steps", str(CALIB_STEPS),
        "--log_every", str(LOG_EVERY),
        # Larger than max_steps, so neither evaluation nor checkpointing runs:
        # both would land inside the window whose wall-clock is being measured.
        "--eval_every", str(CALIB_STEPS * 10),
        "--ckpt_every", str(CALIB_STEPS * 10),
        "--run_name", name,
        "--out_dir", str(out_dir),
    ]
    t0 = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    wall = time.time() - t0

    rec: dict = {
        "d_model": d,
        "params": spiking_param_count(_VOCAB, d, _LAYERS),
        "returncode": proc.returncode,
        "process_wall_clock_s": round(wall, 2),
    }
    if proc.returncode != 0:
        rec["error"] = "\n".join(
            (proc.stderr or proc.stdout or "").strip().splitlines()[-12:]
        )
        return rec

    rows = []
    for line in (run_dir / "log.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    train = [r for r in rows if r.get("event") == "train"]
    if not train:
        rec["error"] = "no train records in log.jsonl"
        return rec

    # Steady state only: the first records carry capture and warm-up, which are
    # a fixed cost per run and not the per-step cost being extrapolated.
    steady = [r for r in train if r["step"] >= CALIB_STEPS // 2] or train[-1:]
    sps = sorted(r["steps_per_s"] for r in steady)
    median_sps = sps[len(sps) // 2]

    first = train[0]
    last = train[-1]
    escape = next(
        (r["step"] for r in train if len(r.get("firing_rate", [])) > 1
         and r["firing_rate"][1] >= ESCAPE_RATE),
        None,
    )
    grad_norms = [r.get("grad_norm", 0.0) for r in train]

    rec.update({
        "median_steps_per_s_steady": round(median_sps, 4),
        "s_per_step_steady": round(1.0 / median_sps, 6),
        "peak_vram_gib": max(r.get("peak_vram_gib", 0.0) for r in train),
        "firing_rate_step0": first.get("firing_rate"),
        "firing_rate_final": last.get("firing_rate"),
        "layer1_escape_step": escape,
        "layer1_escape_rate_threshold": ESCAPE_RATE,
        "max_grad_norm": max(grad_norms),
        # `grad_clip = 1.0` never engaged in the committed baseline (0 of its 81
        # logged steps exceed 1.0, max 0.3844).  Whether it engages at width is
        # a difference in the *optimiser's* behaviour between rungs, so it is
        # recorded per rung rather than assumed constant across the ladder.
        "n_logged_steps_over_clip": sum(g > 1.0 for g in grad_norms),
        "final_train_bpc": last.get("bpc"),
        "n_train_records": len(train),
        # Retained in full: these records cost GPU time, the run directory is
        # deleted, and a statistic nobody thought to extract cannot be
        # recovered afterwards without paying for them again.
        "series": [
            {"step": r["step"], "steps_per_s": r.get("steps_per_s"),
             "grad_norm": r.get("grad_norm"), "firing_rate": r.get("firing_rate"),
             "bpc": r.get("bpc"), "peak_vram_gib": r.get("peak_vram_gib")}
            for r in train
        ],
    })
    return rec


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="destination JSON")
    ap.add_argument("--widths", default=",".join(str(w) for w in WIDTHS))
    ap.add_argument("--keep", action="store_true",
                    help="do not delete the calibration run directories")
    args = ap.parse_args()

    widths = [int(w) for w in args.widths.split(",") if w.strip()]
    results = []
    for d in widths:
        print(f"[calib] d={d} ...", flush=True)
        rec = _run_one(d)
        results.append(rec)
        if "error" in rec:
            print(f"[calib] d={d} FAILED: {rec['error'][:200]}", flush=True)
        else:
            print(f"[calib] d={d}  {rec['median_steps_per_s_steady']} steps/s  "
                  f"{rec['peak_vram_gib']} GiB  params={rec['params']:,}", flush=True)

    ok = [r for r in results if "error" not in r]
    base = next((r for r in ok if r["d_model"] == 512), None)

    for r in ok:
        if base:
            r["s_per_step_vs_d512"] = round(
                r["s_per_step_steady"] / base["s_per_step_steady"], 4)
            # The committed baseline's own 20,000-step wall-clock, scaled by the
            # measured per-step ratio.  Named against its denominator, per
            # CONTRIBUTING.md §3.
            r["projected_20k_s"] = round(398.88 * r["s_per_step_vs_d512"], 1)
        r["vram_alarm"] = r["peak_vram_gib"] > VRAM_ALARM_GIB

    payload = {
        "experiment": "EXP_014_parameter_scaling",
        "purpose": "price width before the ladder; fix the two new widths",
        "calib_steps": CALIB_STEPS,
        "wall_clock_denominator": {
            "run": "snn_beta0.5_s0", "steps": 20000, "wall_clock_s": 398.88,
            "peak_vram_gib": 0.6089,
        },
        "vram_alarm_gib": VRAM_ALARM_GIB,
        "adopts_nothing": True,
        "ranks_nothing": True,
        "reads_no_bpc_against_any_bar": True,
        "results": results,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")
    print(f"\nwrote {out}")

    if not args.keep and _calib_dir().exists():
        shutil.rmtree(_calib_dir())
        print("deleted calibration run directories")


if __name__ == "__main__":
    main()
