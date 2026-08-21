"""EXP_025 driver -- the detached arm against the plain LIF, at two widths.

    # the LOW rung. Entry condition 6: this runs first, alone, and stops.
    python scripts/launch.py --script scripts/exp/025_run_detached_scaling.py \
        --run-name _exp025_driver -- --rung 512 \
        --out docs/reports/data/exp_025_run_manifest.json

    # the HIGH rung. 4.07 GPU-h. NOT started until the low rung is green.
    python scripts/launch.py --script scripts/exp/025_run_detached_scaling.py \
        --run-name _exp025_driver -- --rung 1481 \
        --out docs/reports/data/exp_025_run_manifest.json

Pre-registered in `experiments/logs/EXP_025_detached_scaling.md` (committed at
`d9f8d45`, before this file existed); read that first, because every threshold
enforced here was fixed there before this file ran.

WHAT IT DOES
------------
Trains `arch="snn"` and `arch="twocomp_detach"` at one width, seeds 0/1/2, then
scores each on the test split.  Twelve runs across the two rungs.

The contracts are `014_run_scaling_ladder.py`'s, kept deliberately identical so
that "restartable" means the same thing in both files:

* **Strictly sequential.**  `subprocess.run` blocks on each child, so "one run at
  a time" is a property of the code rather than of who typed the commands.  A
  second run sharing the card would corrupt every wall-clock figure K3 reports
  (§8 item 7).
* **Restartable, and it deletes rather than resumes.**  A run counts as complete
  only when `summary.json` is at the full step count *and* `final_test.json`
  exists.  Anything else is removed and retrained, so no artifact ever describes
  two runs at once (`CONTRIBUTING.md` §6).
* **A5 before every child**, not once at the top: a mutation campaign that
  started mid-ladder would otherwise supply wrong kernels to every later run.
* **The manifest is rewritten after every cell**, so a crash at cell 9 still
  leaves cells 1-8 fully described.

WHY THE RUNGS ARE SEPARATE INVOCATIONS
--------------------------------------
§2.3 measured the high rung at **4.07 GPU-h -- 65 % of the phase's remaining
budget** -- against the low rung's 0.77.  §8 item 6 therefore makes it a separate
authorisation rather than a later stage of one command, so that a defect visible
in the low rung costs 0.77 GPU-h and not 5.1.  `--rung` takes one width and
refuses a list; that refusal is the entry condition, expressed in code.

WHAT IS NEW HERE, AGAINST EXP_014's DRIVER
------------------------------------------
* **K1a has no exemptions at all.**  `EXP_014` diffed against a *committed*
  reference and had to excuse fields absent from it.  Here the K1a reference is
  **this experiment's own** `e25_snn_d512_s0`: all twelve runs come off one tree,
  so `absent_from_reference` is empty by construction and a non-empty one is a
  defect rather than a bookkeeping note.
* **K1b keeps the committed comparison anyway**, against `snn_beta0.5_s0`, and
  exempts the 24 post-dating fields **by name** (§7).  A field absent from the
  reference and *not* on that list aborts, even at its default -- which is what
  makes the list evidence rather than a wildcard.
* **G5 is the gate that keeps decision #11 unengaged.**  Ten recipe fields are
  read back out of every run's own `config.json` and compared to the frozen
  Phase-2 values.  §0 claims this experiment does not touch the recipe; G5 is
  why that is checkable rather than asserted.
* **G3 dispatches the closed form on arch.**  `twocomp_param_count` is
  `spiking_param_count + 2*K*d`, and `model.py:643` records that it applies to
  the detached arm unchanged because the arm is parameter-identical.
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
from snn.model import spiking_param_count, twocomp_param_count  # noqa: E402

RUNS = _REPO / "experiments" / "runs"
LOCKFILE = _REPO / "src" / "snn" / ".MUTATION_CAMPAIGN_RUNNING"
PREREG = _REPO / "experiments" / "logs" / "EXP_025_detached_scaling.md"
CALIB = _REPO / "docs" / "reports" / "data" / "exp_025_cost_calibration.json"

#: §2.1.  512 is the width every committed figure is measured at; 1481 is
#: `EXP_014` §2.2's exact-error-minimising integer for 5M parameters, reused by
#: `EXP_015` and `EXP_017`.  Neither is chosen here.
RUNGS: tuple[int, ...] = (512, 1481)
ARCHES: tuple[str, ...] = ("snn", "twocomp_detach")
SEEDS: tuple[int, ...] = (0, 1, 2)

#: §2.2.  At fixed `d_model` and `seed` the two arms' Configs differ in exactly
#: one field of 58.  Everything outside this set aborts the ladder.
MAY_DIFFER = {"arch", "d_model", "seed", "run_name"}

#: §7, K1a.  This experiment's own first run -- not a committed one -- so K1a
#: carries no exemptions.
K1A_REFERENCE = "e25_snn_d512_s0"

#: §7, K1b.  The committed reference, kept so the ladder is still tied to the
#: project's baseline rather than only to itself.
K1B_REFERENCE = "snn_beta0.5_s0"

#: §7, K1b.  The 24 fields added to `Config` between `snn_beta0.5_s0` and
#: today's tree.  Listed BY NAME: a field absent from the reference and not on
#: this list aborts even at its default.
K1B_EXEMPT_NEW_AT_DEFAULT = frozenset({
    "beta_slow", "w_init", "mu_init", "thr_log_init", "noise_amp", "noise_p",
    "da_mode", "da_source", "da_scale", "da_gain_init",
    "iface_fold", "iface_binary_input", "iface_read_layers", "iface_read_lags",
    "iface_thr_in", "iface_code_sd",
    "nm_source", "nm_pos_len", "nm_tau", "nm_gain_init", "nm_a_init",
    "nm_b_init",
    "tf_kappa", "tf_rolled",
})

#: §7, G5.  The frozen Phase-2 recipe (`02a_phase2_spec.md`).  Decision #11 asks
#: whether these may be re-derived at a new size; this experiment's claim is
#: that it never asks, and this is where that claim is checked.
FROZEN_RECIPE = {
    "lr": 0.003,
    "lr_schedule": "cosine",
    "max_steps": 20000,
    "warmup_steps": 200,
    "grad_clip": 1.0,
    "weight_decay": 0.1,
    "beta1": 0.9,
    "beta2": 0.95,
    "batch_size": 128,
    "seq_len": 256,
}

#: §2.3, G4.  The widest cell peaks at 2.6138 GiB on a 7.96 GiB card, so this is
#: an alarm and not a budget.
VRAM_ALARM_GIB = 3.0

#: §6, A4.  A realised wall-clock this far from the projection means the
#: calibration does not describe the ladder and every budget figure is wrong.
WALL_CLOCK_HALT_RATIO = 1.5

#: §2.3.  The committed `snn_beta0.5_s0` ran 20,000 steps in 398.88 s against
#: its own calibration's 380.8 s projection.
REALISATION_FACTOR = 1.0475

_VOCAB = 205
_LAYERS = 2

_DEFAULTS = {f.name: f.default for f in dataclasses.fields(Config)}


def prereg_sha256() -> str:
    return hashlib.sha256(PREREG.read_bytes()).hexdigest()


def run_name(arch: str, d: int, seed: int) -> str:
    """`e25_` prefixed so no name collides with `EXP_017`'s `detach_d512_s0`."""
    short = {"snn": "snn", "twocomp_detach": "detach"}[arch]
    return f"e25_{short}_d{d}_s{seed}"


def expected_params(arch: str, d: int) -> int:
    """G3.  Dispatched on arch: `model.py:643` records that
    `twocomp_param_count` applies to the detached arm unchanged."""
    if arch == "snn":
        return spiking_param_count(_VOCAB, d, _LAYERS)
    return twocomp_param_count(_VOCAB, d, _LAYERS)


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
    """A5.  Before every child, not once at the top."""
    if LOCKFILE.exists():
        raise SystemExit(
            f"A5: {LOCKFILE.relative_to(_REPO)} exists -- a mutation campaign is "
            "applying wrong kernels to the source tree. Refusing to start a run "
            "that would import one."
        )


def _train_argv(arch: str, d: int, seed: int) -> list[str]:
    """The reference run's launch command with `arch`, `d_model`, `seed` and
    `run_name` changed, and nothing else."""
    return [
        str(_REPO / "scripts" / "train.py"),
        "--arch", arch,
        "--d_model", str(d),
        "--n_layers", str(_LAYERS),
        "--run_name", run_name(arch, d, seed),
        "--seed", str(seed),
        "--eval_every", "2500",
        "--eval_max_windows", "2560",
        "--ckpt_every", "5000",
        "--log_every", "250",
    ]


def _eval_argv(arch: str, d: int, seed: int) -> list[str]:
    run = run_name(arch, d, seed)
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


def _load_cfg(run: str) -> dict:
    return json.loads((RUNS / run / "config.json").read_text(encoding="utf-8"))


def check_k1(run: str, arch: str, d: int, seed: int,
             ref_cfg: dict, ref_name: str, *,
             exempt_new_at_default: frozenset[str] | None) -> dict:
    """K1a / K1b (§7).  Returns the row; raises if anything differs that may not.

    `exempt_new_at_default is None` is K1a's contract: NO field may be absent
    from the reference, because the reference is one of this experiment's own
    runs and every run comes off one tree.
    """
    cfg = _load_cfg(run)
    diffs = {k: [ref_cfg.get(k), cfg.get(k)]
             for k in set(ref_cfg) | set(cfg)
             if ref_cfg.get(k) != cfg.get(k)}

    excused: list[str] = []
    unexpected: dict[str, list] = {}
    for k, (ref_v, new_v) in diffs.items():
        if k in MAY_DIFFER:
            continue
        absent_from_ref = k not in ref_cfg
        if (exempt_new_at_default is not None
                and absent_from_ref
                and k in exempt_new_at_default
                and k in _DEFAULTS
                and new_v == _DEFAULTS[k]):
            excused.append(k)
            continue
        unexpected[k] = [ref_v, new_v]
    if unexpected:
        raise SystemExit(
            f"K1 ({ref_name}): {run} is not the reference plus the declared "
            f"variables. Fields that differ but may not: {unexpected}"
        )
    if cfg.get("arch") != arch:
        raise SystemExit(f"K1: {run} records arch={cfg.get('arch')!r}, expected {arch!r}")
    if int(cfg.get("d_model", -1)) != int(d):
        raise SystemExit(f"K1: {run} records d_model={cfg.get('d_model')!r}, expected {d}")
    if int(cfg.get("seed", -1)) != int(seed):
        raise SystemExit(f"K1: {run} records seed={cfg.get('seed')!r}, expected {seed}")
    return {
        "reference": ref_name,
        "differs_in": sorted(k for k in diffs if k in MAY_DIFFER),
        "absent_from_reference_and_left_at_default": sorted(excused),
        "same_arm": True,
    }


def check_g5(run: str) -> dict:
    """G5 (§7).  The frozen recipe, read back out of the run's own config.

    This is the gate that keeps decision #11 unengaged: §0 claims no recipe term
    moves at any width, and a claim that only appears in prose is not a claim.
    """
    cfg = _load_cfg(run)
    got = {k: cfg.get(k) for k in FROZEN_RECIPE}
    moved = {k: [FROZEN_RECIPE[k], got[k]] for k in FROZEN_RECIPE
             if got[k] != FROZEN_RECIPE[k]}
    if moved:
        raise SystemExit(
            f"G5: {run} did not train under the frozen Phase-2 recipe. Fields "
            f"that moved: {moved}. EXP_025 §0 states it does not engage decision "
            "#11; a moved recipe term would make that false, so the ladder stops."
        )
    return {"gate": "G5", "frozen_fields_checked": sorted(FROZEN_RECIPE),
            "all_frozen": True}


def check_params_and_vram(run: str, arch: str, d: int) -> dict:
    """G3 and G4 (§6).  Both abort."""
    summary = json.loads((RUNS / run / "summary.json").read_text(encoding="utf-8"))
    expected = expected_params(arch, d)
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


def _projection(calib: dict | None, arch: str, d: int) -> float | None:
    """§2.3's projected 20k wall-clock for one cell, with the realisation
    factor applied -- the same number the pre-registration's budget table is
    built from."""
    if not calib:
        return None
    for r in calib.get("results", []):
        if r.get("arch") == arch and int(r.get("d_model", -1)) == int(d):
            return round(float(r["s_per_step_steady"]) * 20000 * REALISATION_FACTOR, 1)
    return None


def check_k3(run: str, realised: float | None, projected: float | None) -> dict:
    """K3 (§7) and A4 (§6).  The cost model is checked, not assumed."""
    row: dict = {
        "projected_20k_s_from_calibration": projected,
        "realised_train_wall_clock_s": realised,
        "realisation_factor_applied": REALISATION_FACTOR,
        "note": ("the projection is a 400-step per-step rate scaled by 20,000 "
                 "and by the committed realisation factor; it excludes capture "
                 "cost and the eight in-training evaluations, so a ratio "
                 "slightly above 1 is expected and its size is the measurement"),
    }
    if realised and projected:
        ratio = realised / projected
        row["ratio_realised_over_projected"] = round(ratio, 4)
        if ratio > WALL_CLOCK_HALT_RATIO or ratio < 1.0 / WALL_CLOCK_HALT_RATIO:
            raise SystemExit(
                f"A4: {run} realised {realised:.1f}s against a projected "
                f"{projected:.1f}s (ratio {ratio:.2f}, halt at "
                f"{WALL_CLOCK_HALT_RATIO}). The calibration does not describe "
                "this ladder, so every budget figure in EXP_025 §2.3 is wrong. "
                "Halting rather than continuing."
            )
    return row


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rung", type=int, required=True, choices=list(RUNGS),
                    help="ONE width. §8 item 6 makes the high rung a separate "
                         "authorisation, so this deliberately takes no list.")
    ap.add_argument("--out", default="docs/reports/data/exp_025_run_manifest.json")
    args = ap.parse_args(argv)

    d = int(args.rung)
    calib = json.loads(CALIB.read_text(encoding="utf-8")) if CALIB.exists() else None
    prereg_before = prereg_sha256()

    out = Path(args.out)
    if not out.is_absolute():
        out = _REPO / out
    out.parent.mkdir(parents=True, exist_ok=True)

    # A rung appends to the manifest rather than replacing it, so the low rung's
    # evidence survives the high rung's invocation.
    manifest: dict = (json.loads(out.read_text(encoding="utf-8"))
                      if out.exists() else {})
    manifest.setdefault("experiment", "EXP_025_detached_scaling")
    manifest.setdefault(
        "prereg", str(PREREG.relative_to(_REPO)).replace("\\", "/"))
    manifest.setdefault("prereg_sha256_before_first_run", prereg_before)
    manifest.setdefault("k1a_reference", K1A_REFERENCE)
    manifest.setdefault("k1b_reference", K1B_REFERENCE)
    manifest.setdefault("may_differ", sorted(MAY_DIFFER))
    manifest.setdefault("k1b_exempt_new_at_default",
                        sorted(K1B_EXEMPT_NEW_AT_DEFAULT))
    manifest.setdefault("frozen_recipe", FROZEN_RECIPE)
    manifest.setdefault("vram_alarm_gib", VRAM_ALARM_GIB)
    manifest.setdefault("seeds", list(SEEDS))
    manifest.setdefault("arches", list(ARCHES))
    manifest.setdefault("adopts_nothing", True)
    manifest.setdefault("ranks_nothing", True)
    manifest.setdefault("rules_no_open_decision", True)
    manifest.setdefault("engages_decision_11", False)
    manifest.setdefault("cells", [])

    if manifest["prereg_sha256_before_first_run"] != prereg_before:
        raise SystemExit(
            "The pre-registration changed between rungs. The experiment is "
            "void; nothing here may be reported."
        )

    def flush() -> None:
        out.write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")

    done = {(c["arch"], c["d_model"], c["seed"]) for c in manifest["cells"]}
    k1b_ref_cfg = _load_cfg(K1B_REFERENCE)

    for arch in ARCHES:
        for seed in SEEDS:
            if (arch, d, seed) in done:
                print(f"\n=== SKIP {run_name(arch, d, seed)}: already in manifest",
                      flush=True)
                continue
            run = run_name(arch, d, seed)
            row: dict = {"arch": arch, "d_model": d, "seed": seed, "run": run}
            if is_complete(run):
                print(f"\n=== SKIP {run}: already complete", flush=True)
                row["skipped_as_complete"] = True
            else:
                if (RUNS / run).exists():
                    print(f"\n=== DELETE partial {run} (never resumed)", flush=True)
                    shutil.rmtree(RUNS / run)
                    row["deleted_partial"] = True
                _check_no_campaign()
                row["train_wall_clock_s"] = round(
                    _run(_train_argv(arch, d, seed), f"train {run}"), 1)
                _check_no_campaign()
                row["eval_wall_clock_s"] = round(
                    _run(_eval_argv(arch, d, seed), f"evaluate {run}"), 1)

            row.update(check_params_and_vram(run, arch, d))
            row["g5"] = check_g5(run)
            row["k1b"] = check_k1(run, arch, d, seed, k1b_ref_cfg, K1B_REFERENCE,
                                  exempt_new_at_default=K1B_EXEMPT_NEW_AT_DEFAULT)
            if (RUNS / K1A_REFERENCE / "config.json").exists():
                row["k1a"] = check_k1(run, arch, d, seed,
                                      _load_cfg(K1A_REFERENCE), K1A_REFERENCE,
                                      exempt_new_at_default=None)
            elif run != K1A_REFERENCE:
                raise SystemExit(
                    f"K1a: reference run {K1A_REFERENCE} does not exist, so the "
                    "zero-exemption gate cannot be applied. The low rung runs "
                    "first for this reason (§8 item 6)."
                )
            row["k3"] = check_k3(run, row.get("wall_clock_s"),
                                 _projection(calib, arch, d))

            manifest["cells"].append(row)
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
    n_this = sum(1 for c in manifest["cells"] if c["d_model"] == d)
    print(f"\nwrote {out}")
    print("EXP025_RUNG_DONE", d, n_this, "/", len(ARCHES) * len(SEEDS), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
