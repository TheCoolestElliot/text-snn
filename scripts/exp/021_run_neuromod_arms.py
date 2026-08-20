"""EXP_021 legs 1-5: the local modulator, its control, and the three-factor rule.

    python scripts/launch.py --script scripts/exp/021_run_neuromod_arms.py \
        --run-name _exp021_driver -- --out docs/reports/data/exp_021_run_manifest.json

THE ANCHOR IS NOT RE-RUN. `EXP_020` leg 1 trained `if_anchor_s0/1/2` on this
same tree in this same session, and decision #10 says this project does not
re-baseline. Training a second anchor would produce two figures for one quantity
and invite the comparison the decision forbids. The driver **requires** those
three runs to be complete and refuses to start otherwise.

Everything else is `020_run_interface_arms.py`'s contract: restartable, deletes
partials rather than resuming them, checks "one thing changed" field by field
against a named reference, reads divergence from the log rather than the exit
code, reads no bpc and resolves no bar.
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
from snn.model import local_dopamine_param_count, spiking_param_count  # noqa: E402

RUNS = _REPO / "experiments" / "runs"
LOCKFILE = _REPO / "src" / "snn" / ".MUTATION_CAMPAIGN_RUNNING"
PREREG = _REPO / "experiments" / "logs" / "EXP_021_neuromodulation.md"
CALIB = _REPO / "docs" / "reports" / "data" / "exp_021_nm_calibration.json"
SCREEN = _REPO / "docs" / "reports" / "data" / "exp_020_021_reachability.json"

_DEFAULTS = {f.name: f.default for f in dataclasses.fields(Config)}

VOCAB, D_MODEL, LAYERS = 205, 512, 2
VRAM_ALARM_GIB = 4.0

#: tau, transcribed from the calibration artifact and ASSERTED against it below.
#: A driver that has drifted from its own calibration does not start a ladder.
#:
#: Five significant figures, which is the precision the pre-registration quotes
#: and therefore the precision the check compares at. The artifact holds
#: 0.0573208462446928; asserting bit-equality against a transcribed decimal
#: would fail on the transcription rather than on a drift, which is the opposite
#: of what this constant is guarding. EXP_018 transcribes its tau the same way
#: (1.1291 against a full-precision median).
NM_TAU = 0.057321
NM_TAU_SIGFIGS = 5

#: EXP_004's pre-registered fallback value, reused because §2.7 DERIVED that
#: kappa = 0 is a saddle for the predictor and the screen confirmed it exactly.
NM_GAIN_INIT = 0.1

#: |kappa| < 1 is enforced at construction so a weight can never reach zero.
#: 0.5 is the value at which the extreme weight ratio is exactly 3.0. Not swept.
TF_KAPPA = 0.5

ANCHOR = [f"if_anchor_s{s}" for s in (0, 1, 2)]

FROZEN: dict[str, object] = {
    "lr": 3e-3, "weight_decay": 0.1, "grad_clip": 1.0, "warmup_steps": 200,
    "max_steps": 20_000, "lr_schedule": "cosine", "beta1": 0.9, "beta2": 0.95,
    "beta": 0.5, "threshold": 1.0, "reset": "hard", "surrogate": "atan",
    "surrogate_alpha": 2.0, "seq_len": 256, "batch_size": 128,
    "corpus": "enwik8", "dtype": "fp32", "fused": True, "cuda_graph": True,
    "deterministic": True, "n_layers": 2, "d_model": 512,
    "nm_tau": NM_TAU, "nm_gain_init": NM_GAIN_INIT, "da_scale": 1.1291,
}

SEEDS = (0, 1, 2)

#: (leg, run, arch, overrides, seed, reference, may_differ)
#:
#: ORDER IS LOAD-BEARING: within each form the headline precedes its control,
#: because the headline is the run whose result could make the rest not worth
#: the GPU time. `nm_local` precedes `tf_*` because it is the leg decision #3
#: already admits, and therefore the one that can produce an adoptable result.
PLAN: list[tuple] = (
    [("1", f"nm_local_s{s}", "localdopamine", {"nm_source": "local"}, s,
      ANCHOR[0], frozenset({"arch", "run_name", "seed", "nm_source",
                            "nm_tau", "nm_gain_init"}))
     for s in SEEDS]
    + [("2", f"nm_rolled_s{s}", "localdopamine", {"nm_source": "rolled"}, s,
        "nm_local_s0", frozenset({"run_name", "seed", "nm_source"}))
       for s in SEEDS]
    + [("3", f"tf_neg_s{s}", "snn", {"tf_kappa": -TF_KAPPA}, s,
        ANCHOR[0], frozenset({"run_name", "seed", "tf_kappa"}))
       for s in SEEDS]
    + [("4", f"tf_pos_s{s}", "snn", {"tf_kappa": +TF_KAPPA}, s,
        "tf_neg_s0", frozenset({"run_name", "seed", "tf_kappa"}))
       for s in SEEDS]
    + [("5", f"tf_roll_s{s}", "snn", {"tf_kappa": -TF_KAPPA, "tf_rolled": True},
        s, "tf_neg_s0", frozenset({"run_name", "seed", "tf_rolled"}))
       for s in SEEDS]
)


def expected_params(arch: str) -> int:
    if arch == "localdopamine":
        return local_dopamine_param_count(VOCAB, D_MODEL, LAYERS)
    return spiking_param_count(VOCAB, D_MODEL, LAYERS)


def prereg_sha256() -> str:
    return hashlib.sha256(PREREG.read_bytes()).hexdigest()


def check_tau_matches_calibration() -> dict:
    """The ladder's tau must be the one the calibration artifact recommends.

    A transcribed constant that has silently drifted from the measurement it
    descends from is a defect this project has met before, one level up: the bar
    is right and the thing it was computed from moved.
    """
    if not CALIB.exists():
        raise SystemExit(
            f"entry condition: {CALIB.relative_to(_REPO)} is missing. tau is a "
            "measured constant and the ladder may not start without it.")
    j = json.loads(CALIB.read_text(encoding="utf-8"))
    got = float(j["nm_tau_median_sd"])
    rounded = float(f"%.{NM_TAU_SIGFIGS}g" % got)
    if rounded != NM_TAU:
        raise SystemExit(
            f"T2: this driver uses nm_tau = {NM_TAU} and the calibration "
            f"artifact recommends {got} (= {rounded} at {NM_TAU_SIGFIGS} s.f.). "
            "One of them has drifted; the ladder does not start.")
    if not j.get("C1_pass"):
        raise SystemExit("T2: the calibration's C1 did not pass; tau is not "
                         "established and the ladder does not start.")
    return {"nm_tau": NM_TAU, "nm_tau_full_precision": got,
            "matches_calibration": True, "sigfigs": NM_TAU_SIGFIGS,
            "relative_spread": j.get("nm_tau_relative_spread")}


def is_complete(run: str) -> bool:
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
            "applying wrong kernels to the source tree.")


def _train_argv(run: str, arch: str, ov: dict, seed: int, steps: int) -> list[str]:
    argv = [
        str(_REPO / "scripts" / "train.py"),
        "--arch", arch, "--d_model", str(D_MODEL), "--n_layers", str(LAYERS),
        "--run_name", run, "--seed", str(seed), "--max_steps", str(steps),
        "--eval_every", "2500", "--eval_max_windows", "2560",
        "--ckpt_every", "5000", "--log_every", "250",
    ]
    # Passed explicitly rather than left to a default, so the run's own
    # config.json records the arm that trained. A localdopamine run that
    # silently took nm_source="off" would train the baseline and be
    # indistinguishable from a null.
    if arch == "localdopamine":
        argv += ["--nm_source", str(ov["nm_source"]),
                 "--nm_tau", repr(NM_TAU),
                 "--nm_gain_init", repr(NM_GAIN_INIT)]
    if "tf_kappa" in ov:
        argv += ["--tf_kappa", repr(ov["tf_kappa"])]
    if ov.get("tf_rolled"):
        argv += ["--tf_rolled"]
    return argv


def _eval_argv(run: str) -> list[str]:
    return [str(_REPO / "scripts" / "evaluate.py"),
            "--ckpt", str(RUNS / run / "ckpt_final.pt"),
            "--split", "test",
            "--out", str(RUNS / run / "final_test.json")]


def _run(argv: list[str], label: str, fatal: bool = True) -> tuple[float, int]:
    print(f"\n=== {label}\n    {' '.join(argv)}", flush=True)
    t0 = time.time()
    proc = subprocess.run([sys.executable, "-u", *argv], cwd=str(_REPO))
    dt = time.time() - t0
    if proc.returncode != 0 and fatal:
        raise SystemExit(f"{label} exited {proc.returncode}")
    return dt, proc.returncode


def _diverged(run: str) -> bool:
    """G7: read the LOG, not the exit code; an ABSENT log reads as diverged."""
    p = RUNS / run / "log.jsonl"
    if not p.exists():
        return True
    for line in p.read_text(encoding="utf-8").splitlines():
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        v = rec.get("loss")
        if v is not None and (v != v or v in (float("inf"), float("-inf"))):
            return True
    return False


def _config_diff(run: str, ref: str) -> dict:
    a = json.loads((RUNS / run / "config.json").read_text(encoding="utf-8"))
    b = json.loads((RUNS / ref / "config.json").read_text(encoding="utf-8"))
    return {k: (b.get(k), a.get(k)) for k in set(a) | set(b) if a.get(k) != b.get(k)}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out",
                    default="docs/reports/data/exp_021_run_manifest.json")
    ap.add_argument("--steps", type=int, default=20_000)
    ap.add_argument("--only", default="")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    _check_no_campaign()
    for f, why in ((PREREG, "the pre-registration is committed before the first "
                            "run, not after it"),
                   (SCREEN, "G8: the §7.2 reachability screen must be green "
                            "before the first training step")):
        if not f.exists():
            raise SystemExit(f"entry condition: {f.relative_to(_REPO)} missing "
                             f"-- {why}")
    missing = [r for r in ANCHOR if not is_complete(r)]
    if missing and not args.dry_run:
        raise SystemExit(
            f"entry condition: the EXP_020 leg-1 anchor is incomplete ({missing}). "
            "Decision #10: this project does not re-baseline, so this ladder "
            "reads that anchor and does not train its own.")

    tau_check = check_tau_matches_calibration()
    sha_before = prereg_sha256()
    only = {s for s in args.only.split(",") if s}
    manifest = {"experiment": "EXP_021", "prereg_sha256_before": sha_before,
                "steps": args.steps, "tau_check": tau_check,
                "anchor": ANCHOR, "vram_alarm_gib": VRAM_ALARM_GIB, "runs": []}

    for leg, run, arch, ov, seed, ref, may_differ in PLAN:
        if only and run not in only:
            continue
        exp = expected_params(arch)
        rec = {"leg": leg, "run": run, "arch": arch, "seed": seed,
               "overrides": ov, "reference": ref, "expected_params": exp,
               "completed": False}

        if is_complete(run):
            print(f"\n=== SKIP {run}: already complete", flush=True)
            rec["skipped_complete"] = True
        else:
            d_run = RUNS / run
            if d_run.exists():
                print(f"\n=== DELETE partial {run}", flush=True)
                rec["deleted_partial"] = True
                if not args.dry_run:
                    shutil.rmtree(d_run)
            if args.dry_run:
                print(f"[dry-run] {leg} {run} {arch} {ov} seed={seed} "
                      f"params~{exp:,}", flush=True)
                manifest["runs"].append(rec)
                continue
            t, code = _run(_train_argv(run, arch, ov, seed, args.steps),
                           f"leg {leg}: train {run}", fatal=False)
            rec["train_s"], rec["train_rc"] = round(t, 2), code
            t, code = _run(_eval_argv(run), f"leg {leg}: eval {run}", fatal=False)
            rec["eval_s"], rec["eval_rc"] = round(t, 2), code

        if args.dry_run:
            manifest["runs"].append(rec)
            continue

        rec["diverged"] = _diverged(run)
        sm = RUNS / run / "summary.json"
        if sm.exists():
            s = json.loads(sm.read_text(encoding="utf-8"))
            rec["params"] = s.get("params")
            rec["wall_clock_s"] = s.get("wall_clock_s")
            rec["peak_vram_gib"] = s.get("peak_vram_gib")
            rec["vram_alarm"] = bool((s.get("peak_vram_gib") or 0) > VRAM_ALARM_GIB)
            rec["G2_params_match"] = (s.get("params") == exp)
            rec["completed"] = int(s.get("steps", 0)) == args.steps
        cf = RUNS / run / "config.json"
        if cf.exists():
            c = json.loads(cf.read_text(encoding="utf-8"))
            rec["G5_frozen_violations"] = {
                k: c.get(k) for k, v in FROZEN.items()
                # nm_tau / nm_gain_init are meaningless on an arch="snn" run and
                # keep their Config defaults there; asserting them everywhere
                # would fail the tf_* legs for a field they do not use.
                if not (arch == "snn" and k in ("nm_tau", "nm_gain_init"))
                and c.get(k) != v
            }
            diff = _config_diff(run, ref)
            extra = set(diff) - set(may_differ)
            rec["K1_diff"] = {k: list(v) for k, v in diff.items()}
            rec["K1_unexpected"] = sorted(extra)
            if extra:
                raise SystemExit(
                    f"K1: {run} differs from {ref} in {sorted(extra)}, outside "
                    f"the declared may_differ {sorted(may_differ)}.")
        manifest["runs"].append(rec)
        print(f"    -> params={rec.get('params')} (expected {exp}) "
              f"diverged={rec['diverged']} wall={rec.get('wall_clock_s')}s",
              flush=True)

    manifest["prereg_sha256_after"] = prereg_sha256()
    manifest["prereg_unchanged"] = (sha_before == manifest["prereg_sha256_after"])
    if not manifest["prereg_unchanged"]:
        raise SystemExit("the pre-registration changed while the ladder ran; "
                         "that voids the experiment.")

    p = _REPO / args.out
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")
    print(f"\nWROTE {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
