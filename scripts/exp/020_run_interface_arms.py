"""EXP_020 legs 1-6: the anchor, the fold, the width it buys, the binary input
code, and the multi-layer readout.

    python scripts/launch.py --script scripts/exp/020_run_interface_arms.py \
        --run-name _exp020_driver -- --out docs/reports/data/exp_020_run_manifest.json

WHY THIS IS A DRIVER AND NOT THIRTY COMMANDS
---------------------------------------------
`CONTRIBUTING.md` §6: a driver must be restartable -- skip what is complete,
**delete** what is partial rather than resuming it, so no artifact ever describes
two runs at once. It also makes "one thing changed" a *check* rather than a
claim: every arm's `config.json` is diffed field by field against its leg's
named reference run, and anything outside the declared `may_differ` set aborts
the ladder.

WHAT IT REFUSES TO DO
---------------------
  * It reads no bpc and resolves no bar. `020_interface_results.py` does that,
    and is committed before any run's bpc is read.
  * It changes no hyperparameter, and proves it: G5 reads the frozen recipe
    terms back out of each run's own `config.json`.
  * It does not stop when a run diverges. A divergence is evidence; the run is
    recorded, evaluated anyway, and the ladder continues.
  * **It does not reseed.** A failed seed is reported at reduced n and marked
    provisional; promotion is by seeds *added*, never by one swapped in.
  * It adopts nothing and ranks nothing.

WIDTH MATCHING, AND WHERE IT IS EXACT
--------------------------------------
`iface_binin` is **exactly parameter-identical** to the anchor -- 735,437 both
-- because binarising the input code changes the code's alphabet and not its
shape. That leg is therefore a clean single-variable comparison in the strongest
sense the protocol has.

The other two are matched on the integer width grid and are not exact:

    anchor / binin   d = 512   735,437     (identical)
    fold             d = 512   473,293     64.4 % of the anchor -- ON PURPOSE
    foldwide         d = 675   733,930     99.80 %
    mlayer           d = 471   734,494     99.87 %

`fold` is deliberately NOT matched. It is the control that prices the
factorisation at unchanged width, and matching it would confound the two things
`EXP_020` exists to separate: what removing a provably redundant parameter block
costs the optimiser, and what spending it on width buys. Reported as realised
counts by G2; no leg claims identity it does not have.
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
from snn.interface import folded_param_count  # noqa: E402
from snn.model import spiking_param_count  # noqa: E402

RUNS = _REPO / "experiments" / "runs"
LOCKFILE = _REPO / "src" / "snn" / ".MUTATION_CAMPAIGN_RUNNING"
PREREG = _REPO / "experiments" / "logs" / "EXP_020_interface.md"
PROBE = _REPO / "docs" / "reports" / "data" / "exp_020_readout_probe.json"

_DEFAULTS = {f.name: f.default for f in dataclasses.fields(Config)}

VOCAB = 205
LAYERS = 2

#: Peak VRAM measured at 0.61 GiB for the committed 735K baseline. An alarm
#: rather than a limit: under WDDM a spill is a ~50x slowdown that never raises.
VRAM_ALARM_GIB = 4.0

#: The frozen Phase-2 recipe. Read back out of every run's own config.json by
#: G5. EXP_020 §0 says this experiment changes no hyperparameter; a run that did
#: is not this experiment. `d_model` is NOT in this set -- it is the thing three
#: legs deliberately move, and G2 checks the realised parameter counts instead.
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
    "tf_kappa": 0.0,
    "iface_thr_in": 0.0,
}

SEEDS = (0, 1, 2)

#: (leg, run_name, d_model, overrides, seed, reference_run, may_differ)
#:
#: ORDER IS LOAD-BEARING. The anchor runs first because every other leg names it
#: as a reference and a reference must exist before the run diffed against it.
#: Leg 1b is one seed only: it is a GATE, not an arm -- `arch="interface"` at
#: its defaults is bitwise `arch="snn"` (tests/test_interface.py N1), so leg 1b
#: must reproduce `if_anchor_s0` EXACTLY, and one seed either does that or does
#: not. Spending three on it would buy no additional information.
def _plan() -> list[tuple]:
    p: list[tuple] = []
    for s in SEEDS:
        p.append(("1", f"if_anchor_s{s}", 512, {"arch": "snn"}, s,
                  None if s == 0 else "if_anchor_s0",
                  frozenset() if s == 0 else frozenset({"run_name", "seed"})))
    p.append(("1b", "if_nest_s0", 512, {"arch": "interface"}, 0, "if_anchor_s0",
              frozenset({"arch", "run_name"})))
    for s in SEEDS:
        p.append(("2", f"if_fold_s{s}", 512,
                  {"arch": "interface", "iface_fold": True}, s, "if_nest_s0",
                  frozenset({"run_name", "seed", "iface_fold"})))
    for s in SEEDS:
        p.append(("3", f"if_foldwide_s{s}", 675,
                  {"arch": "interface", "iface_fold": True}, s, "if_fold_s0",
                  frozenset({"run_name", "seed", "d_model"})))
    for s in SEEDS:
        p.append(("4", f"if_binin_s{s}", 512,
                  {"arch": "interface", "iface_binary_input": True}, s,
                  "if_nest_s0",
                  frozenset({"run_name", "seed", "iface_binary_input"})))
    for s in SEEDS:
        p.append(("5", f"if_mlayer_s{s}", 471,
                  {"arch": "interface", "iface_read_layers": "all"}, s,
                  "if_nest_s0",
                  frozenset({"run_name", "seed", "d_model",
                             "iface_read_layers"})))
    return p


PLAN = _plan()


def expected_params(d: int, ov: dict) -> int:
    if ov.get("arch") == "snn":
        return spiking_param_count(VOCAB, d, LAYERS)
    blocks = LAYERS if ov.get("iface_read_layers") == "all" else 1
    blocks *= int(ov.get("iface_read_lags", 1))
    return folded_param_count(VOCAB, d, LAYERS, folded=bool(ov.get("iface_fold")),
                              read_blocks=blocks)


def prereg_sha256() -> str:
    return hashlib.sha256(PREREG.read_bytes()).hexdigest()


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
            "that would import one.")


def _train_argv(run: str, d: int, ov: dict, seed: int, steps: int) -> list[str]:
    """NOTE WHAT IS ABSENT: no --lr, no --grad_clip, no --weight_decay. They are
    Config defaults, and G5 asserts afterwards that none of them moved."""
    argv = [
        str(_REPO / "scripts" / "train.py"),
        "--arch", str(ov["arch"]),
        "--d_model", str(d),
        "--n_layers", str(LAYERS),
        "--run_name", run,
        "--seed", str(seed),
        "--max_steps", str(steps),
        "--eval_every", "2500",
        "--eval_max_windows", "2560",
        "--ckpt_every", "5000",
        "--log_every", "250",
    ]
    # Passed explicitly rather than left to a default, so the run's own
    # config.json records the arm that trained. An interface run that silently
    # took every default would train the baseline and be indistinguishable from
    # a null.
    if ov.get("iface_fold"):
        argv += ["--iface_fold"]
    if ov.get("iface_binary_input"):
        argv += ["--iface_binary_input"]
    if ov.get("iface_read_layers"):
        argv += ["--iface_read_layers", str(ov["iface_read_layers"])]
    if ov.get("iface_read_lags"):
        argv += ["--iface_read_lags", str(ov["iface_read_lags"])]
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
    if proc.returncode != 0 and fatal:
        raise SystemExit(f"{label} exited {proc.returncode}")
    return dt, proc.returncode


def _diverged(run: str) -> bool:
    """G7: read the LOG, not the exit code.

    `EXP_015` §9.10: a trainer that goes non-finite exits 0 and writes a
    checkpoint, and its driver marked both dead legs `completed: true`. An
    ABSENT log reads as diverged here rather than as clean -- the same defect
    one level down.
    """
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
                    default="docs/reports/data/exp_020_run_manifest.json")
    ap.add_argument("--steps", type=int, default=20_000)
    ap.add_argument("--only", default="", help="comma-separated run names")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    _check_no_campaign()
    if not PREREG.exists():
        raise SystemExit(
            f"entry condition: {PREREG.relative_to(_REPO)} does not exist. The "
            "pre-registration is committed before the first run, not after it.")
    if not PROBE.exists():
        raise SystemExit(
            f"entry condition: {PROBE.relative_to(_REPO)} is missing. Leg A "
            "authorises leg 5 and de-authorises the tap ladder; the training "
            "ladder does not start without it.")

    sha_before = prereg_sha256()
    only = {s for s in args.only.split(",") if s}
    manifest = {
        "experiment": "EXP_020",
        "prereg_sha256_before": sha_before,
        "steps": args.steps,
        "vram_alarm_gib": VRAM_ALARM_GIB,
        "runs": [],
    }

    for leg, run, d, ov, seed, ref, may_differ in PLAN:
        if only and run not in only:
            continue
        exp = expected_params(d, ov)
        rec = {"leg": leg, "run": run, "d_model": d, "seed": seed,
               "overrides": ov, "reference": ref,
               "expected_params": exp, "completed": False}

        if is_complete(run):
            print(f"\n=== SKIP {run}: already complete", flush=True)
            rec["skipped_complete"] = True
        else:
            d_run = RUNS / run
            if d_run.exists():
                # DELETE rather than resume: a resumed run writes a summary.json
                # whose step count looks right and whose trajectory is two runs
                # spliced (CONTRIBUTING.md §6).
                print(f"\n=== DELETE partial {run}", flush=True)
                rec["deleted_partial"] = True
                if not args.dry_run:
                    shutil.rmtree(d_run)
            if args.dry_run:
                print(f"[dry-run] {leg} {run} d={d} {ov} seed={seed} "
                      f"params~{exp:,}", flush=True)
                manifest["runs"].append(rec)
                continue
            t, code = _run(_train_argv(run, d, ov, seed, args.steps),
                           f"leg {leg}: train {run}", fatal=False)
            rec["train_s"], rec["train_rc"] = round(t, 2), code
            t, code = _run(_eval_argv(run), f"leg {leg}: eval {run}", fatal=False)
            rec["eval_s"], rec["eval_rc"] = round(t, 2), code

        if args.dry_run:
            manifest["runs"].append(rec)
            continue

        # -- gates, after the fact and on the run's own artifacts -------------
        rec["diverged"] = _diverged(run)
        sm = RUNS / run / "summary.json"
        if sm.exists():
            s = json.loads(sm.read_text(encoding="utf-8"))
            rec["params"] = s.get("params")
            rec["wall_clock_s"] = s.get("wall_clock_s")
            rec["peak_vram_gib"] = s.get("peak_vram_gib")
            rec["vram_alarm"] = bool((s.get("peak_vram_gib") or 0) > VRAM_ALARM_GIB)
            # G2: realised parameter count, against the closed form. Not a
            # tolerance -- these must be equal, because a closed form that does
            # not describe the model that ran is a defect in the accounting.
            rec["G2_params_match"] = (s.get("params") == exp)
            rec["completed"] = int(s.get("steps", 0)) == args.steps
        cf = RUNS / run / "config.json"
        if cf.exists():
            c = json.loads(cf.read_text(encoding="utf-8"))
            # G5: no hyperparameter moved, read from the run's own config.
            rec["G5_frozen_violations"] = {
                k: c.get(k) for k, v in FROZEN.items() if c.get(k) != v
            }
            if ref is not None:
                diff = _config_diff(run, ref)
                extra = set(diff) - set(may_differ)
                rec["K1_diff"] = {k: list(v) for k, v in diff.items()}
                rec["K1_unexpected"] = sorted(extra)
                if extra:
                    raise SystemExit(
                        f"K1: {run} differs from {ref} in {sorted(extra)}, which "
                        f"is outside the declared may_differ {sorted(may_differ)}. "
                        "More than one thing changed; the ladder does not continue.")
        manifest["runs"].append(rec)
        print(f"    -> params={rec.get('params')} (expected {exp}) "
              f"diverged={rec['diverged']} wall={rec.get('wall_clock_s')}s",
              flush=True)

    sha_after = prereg_sha256()
    manifest["prereg_sha256_after"] = sha_after
    manifest["prereg_unchanged"] = (sha_before == sha_after)
    if not manifest["prereg_unchanged"]:
        raise SystemExit(
            "the pre-registration changed while the ladder ran. That voids the "
            "experiment and no manifest is written that would look complete.")

    p = _REPO / args.out
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")
    print(f"\nWROTE {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
