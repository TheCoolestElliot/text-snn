"""Price a training configuration before an experiment is pre-registered against it.

    # EXP_014's original use -- price WIDTH on the baseline neuron
    python scripts/exp/014_calibrate_cost.py \
        --out docs/reports/data/exp_014_cost_calibration.json

    # EXP_015's use -- price ARCH at a fixed width, against the d=1481 rung
    python scripts/exp/014_calibrate_cost.py \
        --arch snn,twocomp,twocomp_threshold,gru --widths 1481 \
        --base-arch snn --base-width 1481 --denominator-run scale_d1481_s0 \
        --out docs/reports/data/exp_015_cost_calibration.json

Sweeps the cartesian product of `--arch` x `--widths`, 400 steps each, and
reports what a full run of each would cost.  Kept at its original `014_` name
because it is now shared tooling rather than one experiment's script -- the same
precedent `001_memory_horizon.py` set when it grew a per-experiment registry
instead of being copied.

WHY THIS RUNS BEFORE THE THING IT PRICES
----------------------------------------
`03_phase3_candidates.md` §6.3 #11 priced the parameter-scaling pilot at 0.35
GPU-hours and **no measurement in this tree supported that number**.  It descended
from `01_reconnaissance.md` §3.5's width sweep, which is eager, forward-only and
under `no_grad` -- and `02_baseline_report.md` §6.2, titled "The 14.5 us constant
does not transfer", withdrew that section's cost model and closed by instructing
Phase 3's ROI estimates not to reuse it.  `EXP_003`'s depth ladder cannot price
width either: it holds `K*d^2` nearly constant, so `K`, `K*d` and `K*d^2` are
collinear across its four points and a fit over them is degenerate.

So this file measures what a ladder will cost instead of asserting it, in the
same spirit as `013_calibrate_noise_scale.py`, which measured `sd(cur)` before
fixing an amplitude ladder rather than choosing three round numbers.  **The
measurement was worth making: EXP_014's realised cost was 3.1x its ranked
estimate**, and `04_phase4_interim.md` §7.1's rev-8 note records that every other
row in that column descends from the same withdrawn model.

WHAT IT MEASURES, AND WHY EACH THING IS HERE
--------------------------------------------
For each (arch, width): steps/s, peak VRAM, firing rate at step 0 and at the end,
the step at which layer 1 stops being silent, and the largest gradient norm seen.

* **steps/s** converts §7.1's GPU-h column from an assumption into a number.
* **peak VRAM** guards `docs/chat/BUILD_NOTES.md` §2's trap: under Windows' WDDM
  a spill past the card does not fail, it becomes ~50x slower.  A spill would
  therefore corrupt a wall-clock figure while looking like a slow run.
* **firing rates and the escape step** are `audit_07_init_pathology.json`'s
  measurement repeated at the configuration that will actually train.  Both are
  absent for `arch="gru"`, which has no spikes; the fields are recorded as null
  rather than faked.
* **max grad norm** bears on whether the clip fires.  This is not cosmetic:
  `EXP_014` §9.12 established that `_clip_grad_norm_fp64` changes the trajectory
  wherever it fires, so "does the clip engage in this configuration" is now a
  question about reproducibility and not only about optimisation.

WHY MEASURING FIRST DOES NOT CONTAMINATE THE PRE-REGISTRATION
--------------------------------------------------------------
What is measured here is **wall-clock and VRAM**; what the experiments decide on
is **bpc**.  Nothing here reads a bpc against any bar, and no run scored here
survives -- every calibration directory is deleted once its numbers are
extracted.  Choosing a config from a measurement is normally what pre-registration
forbids; it is admissible here precisely because the measured quantity and the
decided quantity are disjoint, and this paragraph exists so that argument is on
the record rather than assumed.

`final_train_bpc` IS recorded, because a configuration that fails to train at all
must be visible.  It is a 400-step training-split number against no bar, it is
never compared across arms here, and `reads_no_bpc_against_any_bar` is carried in
the payload as a structural claim.
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

from snn.model import (  # noqa: E402
    dopamine_param_count,
    gru_param_count,
    match_gru_width,
    prescan_param_count,
    spiking_param_count,
    twocomp_param_count,
    twocomp_threshold_param_count,
                       interface_param_count,
                       local_dopamine_param_count,)

#: Candidate widths for EXP_014's original width sweep.  512 is the committed
#: baseline and anchors the ratios.  1020/1481 are the exact-error-minimising
#: integers for 2.5M/5M parameters; 1024/1480/1488 are the aligned fallbacks
#: tested for a GEMM shape cliff.
WIDTHS: tuple[int, ...] = (512, 1020, 1024, 1480, 1481, 1488)

#: Long enough to clear `warmup_steps = 200` and leave steady-state records,
#: short enough that the whole probe is minutes.  No evaluation runs.
CALIB_STEPS = 400
LOG_EVERY = 25

#: A spill under WDDM is a slowdown, not a failure, so it needs a threshold
#: rather than an exception.  This is the ALARM, not a limit: it flags a row and
#: does not abort, so a sweep still returns the number that would justify raising
#: it.  EXP_014's widest rung measured 1.6434 GiB on a 7.96 GiB card.
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


def param_count(arch: str, vocab_size: int, d_model: int, n_layers: int) -> int:
    """The committed closed form for `arch`, so a run's own `summary.json` has
    something independent to be checked against.

    `gru` is derived rather than chosen: `GRUCharLM` picks the hidden width whose
    parameter count is closest to the spiking arm's at the same `d_model`, so the
    anchor is parameter-matched **by construction** and not by arithmetic done
    here.  This function reproduces that derivation rather than duplicating it.
    """
    if arch in ("snn", "analogue", "noise"):
        return spiking_param_count(vocab_size, d_model, n_layers)
    if arch in ("threshold", "tokenshift"):
        return prescan_param_count(vocab_size, d_model, n_layers)
    if arch == "dopamine":
        # EXP_018's arm adds one `[1, d]` per-channel sensitivity per layer, so
        # its count is numerically the pre-scan arms'. It gets its OWN closed
        # form rather than joining that branch, because the two agree by
        # coincidence of shape and not by construction: the pre-scan arms'
        # parameter is exactly redundant (it folds into `layers.k.weight`) and
        # this one is not (it multiplies a signal that varies with `t`). A future
        # edit to either must not silently move the other.
        return dopamine_param_count(vocab_size, d_model, n_layers)
    if arch in ("twocomp", "twocomp_detach"):
        # EXP_017's arm adds no parameter of any shape -- it is the adopted arm's
        # forward kernel with a bounded backward -- so it is parameter-IDENTICAL
        # to `twocomp`, not merely parameter-matched. Sharing the closed form is
        # the statement of that, and a future arm that did add a parameter would
        # have to add a branch here rather than inherit a wrong count silently.
        return twocomp_param_count(vocab_size, d_model, n_layers)
    if arch == "twocomp_threshold":
        return twocomp_threshold_param_count(vocab_size, d_model, n_layers)
    if arch == "interface":
        # EXP_020's arm. Its count depends on the SWITCHES, not on `arch` alone
        # -- the fold removes `d*d + d`, a multi-block readout adds
        # `(blocks - 1)*d*V` -- and this signature carries no switches. Rather
        # than return a number that is right only at the defaults, it returns
        # the defaults' count and the caller must pass `--base-arch snn` when
        # pricing a non-default switch. That is stated as a raise below rather
        # than as a comment nobody reads.
        return interface_param_count(vocab_size, d_model, n_layers,
                                     fold=False, read_layers="last",
                                     read_lags=1)
    if arch == "localdopamine":
        # EXP_021's arm adds `3*d + 1` per MODULATED layer -- the diagonal
        # predictor's `a` and `b`, the per-channel sensitivity, and the learned
        # `log_tau`. It does NOT share `dopamine_param_count`: that one is
        # head-driven and adds `d` per layer including layer 0, this one is
        # previous-layer-driven and adds nothing to layer 0 because layer 0 has
        # no previous layer. The two differ by construction, not by shape.
        return local_dopamine_param_count(vocab_size, d_model, n_layers)
    if arch == "gru":
        target = spiking_param_count(vocab_size, d_model, n_layers)
        return gru_param_count(
            vocab_size, match_gru_width(vocab_size, n_layers, target), n_layers
        )
    raise ValueError(f"no committed parameter closed form for arch={arch!r}")


def _calib_dir() -> Path:
    return _REPO / "experiments" / "runs" / "_exp014_cost"


def _train_argv(arch: str, d: int, name: str, out_dir: Path,
                flags: list[str] | None = None) -> list[str]:
    """The child command.  Pinned by `tests/test_calibrate_cost.py` for
    `arch="snn"`, so generalising this file cannot silently change what EXP_014's
    committed calibration was measured with.

    `flags` appends extra `--field value` pairs so that an arm whose cost
    depends on a SWITCH rather than on `arch` and `d_model` -- every EXP_020,
    EXP_022 and EXP_023 arm does -- can be priced by this instrument instead of
    a second one.  Two implementations of one statistic is how they stop
    agreeing, and this project's single checked GPU-hour estimate was low by
    3.1x, so the pre-flight is not optional and must not fork.
    """
    return [
        sys.executable, str(_REPO / "scripts" / "train.py"),
        "--arch", arch,
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
    ] + list(flags or ())


def _run_name_for(arch: str, d: int) -> str:
    """`snn` keeps EXP_014's original `calib_d<width>` spelling; only the arms
    added when this file was generalised carry the arch in the name."""
    return f"calib_d{d}" if arch == "snn" else f"calib_{arch}_d{d}"


def _run_one(arch: str, d: int, *, label: str | None = None,
             flags: list[str] | None = None,
             params: int | None = None) -> dict:
    """Train `CALIB_STEPS` steps in one configuration and read its own log back."""
    name = _run_name_for(arch, d) if label is None else f"calib_{label}"
    out_dir = _calib_dir()
    run_dir = out_dir / name
    if run_dir.exists():
        shutil.rmtree(run_dir)

    t0 = time.time()
    proc = subprocess.run(_train_argv(arch, d, name, out_dir, flags),
                          capture_output=True, text=True)
    wall = time.time() - t0

    rec: dict = {
        "arch": arch,
        "d_model": d,
        # `param_count` dispatches on `arch` alone, so for a switched arm the
        # caller passes the count its own closed form gives.  Passed rather than
        # guessed: the realised-vs-closed check below is the thing that caught
        # `local_dopamine_param_count` missing `nm_log_tau`, and it only works
        # if the closed form being checked is the arm's.
        "params": (param_count(arch, _VOCAB, d, _LAYERS) if params is None
                   else int(params)),
        "returncode": proc.returncode,
        "process_wall_clock_s": round(wall, 2),
    }
    if label is not None:
        rec["label"] = label
        rec["flags"] = list(flags or ())
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

    # The run writes its own parameter count; disagreeing with the closed form
    # means one of the two is wrong and the sweep must not be trusted either way.
    summary_path = run_dir / "summary.json"
    if summary_path.exists():
        realised = json.loads(summary_path.read_text(encoding="utf-8")).get("params")
        rec["params_realised"] = realised
        if realised is not None and realised != rec["params"]:
            rec["error"] = (
                f"parameter count disagrees: closed form {rec['params']}, "
                f"run reported {realised}"
            )
            return rec

    # Steady state only: the first records carry capture and warm-up, which are
    # a fixed cost per run and not the per-step cost being extrapolated.
    steady = [r for r in train if r["step"] >= CALIB_STEPS // 2] or train[-1:]
    sps = sorted(r["steps_per_s"] for r in steady)
    median_sps = sps[len(sps) // 2]

    first = train[0]
    last = train[-1]
    escape = next(
        (r["step"] for r in train if len(r.get("firing_rate") or []) > 1
         and r["firing_rate"][1] >= ESCAPE_RATE),
        None,
    )
    grad_norms = [r.get("grad_norm", 0.0) for r in train]

    rec.update({
        "median_steps_per_s_steady": round(median_sps, 4),
        "s_per_step_steady": round(1.0 / median_sps, 6),
        "peak_vram_gib": max(r.get("peak_vram_gib", 0.0) for r in train),
        # Null rather than zero for `gru`, which has no spikes to rate.
        "firing_rate_step0": first.get("firing_rate"),
        "firing_rate_final": last.get("firing_rate"),
        "layer1_escape_step": escape,
        "layer1_escape_rate_threshold": ESCAPE_RATE,
        "max_grad_norm": max(grad_norms),
        # `grad_clip = 1.0` never engaged in the committed baseline's 81 logged
        # steps (max 0.3844), and EXP_014 §9.12 later found it firing three times
        # in 20,000 -- between the logging cadence.  Whether it engages here is
        # recorded per configuration rather than assumed constant across a ladder.
        "n_logged_steps_over_clip": sum(g > 1.0 for g in grad_norms),
        "steps_over_clip": [r["step"] for r in train if r.get("grad_norm", 0.0) > 1.0],
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


def _denominator(run_name: str) -> dict:
    """Read the reference run's own committed `summary.json` rather than
    hard-coding its wall-clock.  `CONTRIBUTING.md` §3 requires a ratio to name
    its denominator; reading it means the name and the number cannot drift."""
    p = _REPO / "experiments" / "runs" / run_name / "summary.json"
    s = json.loads(p.read_text(encoding="utf-8"))
    if int(s["steps"]) != 20_000:
        raise SystemExit(
            f"{run_name} ran {s['steps']} steps; `projected_20k_s` assumes 20,000. "
            "Pick a 20k reference run or extend this script to carry the step count."
        )
    return {
        "run": run_name,
        "steps": int(s["steps"]),
        "wall_clock_s": float(s["wall_clock_s"]),
        "peak_vram_gib": float(s["peak_vram_gib"]),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="destination JSON")
    ap.add_argument("--arch", default="snn",
                    help="comma-separated arch names; swept against --widths")
    ap.add_argument("--widths", default=",".join(str(w) for w in WIDTHS))
    ap.add_argument("--base-arch", default="snn",
                    help="arch of the row the ratios are taken against")
    ap.add_argument("--base-width", type=int, default=512,
                    help="d_model of the row the ratios are taken against")
    ap.add_argument("--denominator-run", default="snn_beta0.5_s0",
                    help="committed 20k run whose wall-clock the projection scales")
    ap.add_argument("--keep", action="store_true",
                    help="do not delete the calibration run directories")
    ap.add_argument("--extra", default=None,
                    help="JSON list of switched rows: "
                         '[{"label":..,"arch":..,"d_model":..,'
                         '"flags":[..],"params":N}]. For arms whose cost '
                         "depends on a switch rather than on arch and width")
    args = ap.parse_args()

    arches = [a.strip() for a in args.arch.split(",") if a.strip()]
    widths = [int(w) for w in args.widths.split(",") if w.strip()]
    denom = _denominator(args.denominator_run)

    results = []
    for arch in arches:
        for d in widths:
            print(f"[calib] {arch} d={d} ...", flush=True)
            rec = _run_one(arch, d)
            results.append(rec)
            if "error" in rec:
                print(f"[calib] {arch} d={d} FAILED: {rec['error'][:200]}", flush=True)
            else:
                print(f"[calib] {arch} d={d}  "
                      f"{rec['median_steps_per_s_steady']} steps/s  "
                      f"{rec['peak_vram_gib']} GiB  params={rec['params']:,}  "
                      f"clip {rec['n_logged_steps_over_clip']}/{rec['n_train_records']}",
                      flush=True)

    for row in json.loads(Path(args.extra).read_text(encoding="utf-8")
                          if Path(args.extra).exists() else args.extra) \
            if args.extra else []:
        label = str(row["label"])
        print(f"[calib] {label} ...", flush=True)
        rec = _run_one(str(row.get("arch", "snn")), int(row["d_model"]),
                       label=label, flags=[str(f) for f in row.get("flags", ())],
                       params=row.get("params"))
        results.append(rec)
        if "error" in rec:
            print(f"[calib] {label} FAILED: {rec['error'][:300]}", flush=True)
        else:
            print(f"[calib] {label}  {rec['median_steps_per_s_steady']} steps/s  "
                  f"{rec['peak_vram_gib']} GiB  params={rec['params']:,}",
                  flush=True)

    ok = [r for r in results if "error" not in r]
    # `"label" not in r` so a switched `--extra` row at the base arch and width
    # can never become the denominator every ratio is taken against.
    base = next((r for r in ok
                 if r["d_model"] == args.base_width
                 and r["arch"] == args.base_arch and "label" not in r),
                None)

    for r in ok:
        if base:
            r["s_per_step_vs_base"] = round(
                r["s_per_step_steady"] / base["s_per_step_steady"], 4)
            # The reference run's own 20,000-step wall-clock, scaled by the
            # measured per-step ratio.  Named against its denominator, per
            # CONTRIBUTING.md §3.
            r["projected_20k_s"] = round(
                denom["wall_clock_s"] * r["s_per_step_vs_base"], 1)
        r["vram_alarm"] = r["peak_vram_gib"] > VRAM_ALARM_GIB

    payload = {
        "purpose": "price a configuration before anything is pre-registered against it",
        "calib_steps": CALIB_STEPS,
        "arches": arches,
        "widths": widths,
        "ratio_base": {"arch": args.base_arch, "d_model": args.base_width},
        "wall_clock_denominator": denom,
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
