"""EXP_022 legs A1-A3 and B1-B2: the input density ladder and the unmatched readout.

    python scripts/launch.py --script scripts/exp/022_run_input_output_arms.py \
        --run-name _exp022_driver -- --out docs/reports/data/exp_022_run_manifest.json

WHY THIS IS A DRIVER AND NOT FIFTEEN COMMANDS
----------------------------------------------
`CONTRIBUTING.md` §6: a driver must be restartable -- skip what is complete,
**delete** what is partial rather than resuming it, so no artifact ever describes
two runs at once. It also makes "one thing changed" a *check* rather than a
claim: every arm's `config.json` is diffed field by field against its leg's
named reference run, and anything outside the declared `may_differ` set aborts
the ladder.

WHAT IT REFUSES TO DO
---------------------
  * It reads no bpc and resolves no bar. `022_input_output_results.py` does
    that, and is committed before any run's bpc is read.
  * It changes no hyperparameter, and proves it: G5 reads the frozen recipe
    terms back out of each run's own `config.json`.
  * It does not stop when a run diverges. A divergence is evidence; the run is
    recorded, evaluated anyway, and the ladder continues.
  * **It does not reseed.** A failed seed is reported at reduced n and marked
    provisional; promotion is by seeds *added*, never by one swapped in.
  * It adopts nothing and ranks nothing.
  * **It does not re-run `if_anchor` or `if_binin`.** Both are EXP_020 cells
    trained in this session on this tree at this recipe, and `if_binin` IS the
    `q = 0.50` rung of leg A's own ladder. Re-training them would spend
    0.7 GPU-h to reproduce numbers the session already holds -- and would then
    leave two anchors that could disagree.

WHAT IS PARAMETER-IDENTICAL HERE, AND WHAT IS NOT
--------------------------------------------------
    if_anchor / if_binin / io_sp34 / io_sp10 / io_sp05    d = 512   735,437
    io_mall                                              d = 512   840,397
    io_wide                                              d = 553   839,659

**The entire density ladder is parameter-identical to the anchor** -- 735,437 at
all four rungs -- because changing a code's density changes its alphabet and not
its shape. That makes leg A a single-variable comparison in the strongest sense
this protocol has.

Leg B is deliberately UNMATCHED against the anchor and matched against
`io_wide`, which is the whole reason it exists: EXP_020 leg 5 paid for the wider
head by cutting `d` 512 -> 471, and EXP_014's slope prices that cut at ~0.023
bpc against a probe estimate of 0.019-0.022. The confound and the signal were
the same size. `io_wide` carries the same 104,960 extra parameters as plain
width, 738 short of `io_mall`'s count (0.088 %), reported as realised by G2.
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
PREREG = _REPO / "experiments" / "logs" / "EXP_022_input_output.md"
PREFLIGHT = _REPO / "docs" / "reports" / "data" / "exp_022_023_cost_preflight.json"
SCREEN = _REPO / "docs" / "reports" / "data" / "exp_022_023_reachability.json"

_DEFAULTS = {f.name: f.default for f in dataclasses.fields(Config)}

VOCAB = 205
LAYERS = 2

#: Peak VRAM measured at 0.693 GiB for the widest arm here (`io_mall`) against
#: the 0.609 GiB baseline. An alarm rather than a limit: under WDDM a spill is a
#: ~50x slowdown that never raises.
VRAM_ALARM_GIB = 4.0

#: G5. Read back out of each run's own config.json after it finishes, so a
#: hyperparameter that moved is caught by the artifact and not by a claim.
FROZEN = {
    "lr": 3e-3,
    "weight_decay": 0.1,
    "grad_clip": 1.0,
    "warmup_steps": 200,
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
    # The plasticity axis is NOT run (pre-registration §2.3 / §6 item 1). Frozen
    # at EXP_020's value so a rung that quietly moved it would abort rather than
    # measure two things.
    "iface_code_sd": 1.0,
}

SEEDS = (0, 1, 2)

#: The density ladder's thresholds, to ten decimal places. `q = 1 - Phi(thr)`,
#: and the gain the module derives from each is `1/sqrt(q(1-q))`. NOT tuned:
#: 0.50 is EXP_020's rung, 0.34 is the stack's own converged firing rate
#: (EXP_000 F3), and 0.10/0.05 span an order of magnitude below it while still
#: carrying 240/147 bits against an alphabet needing 7.7.
THR = {"sp34": 0.4124631294, "sp10": 1.2815515655, "sp05": 1.6448536270}


#: (leg, run_name, d_model, overrides, seed, reference_run, may_differ)
#:
#: ORDER IS LOAD-BEARING. Leg B runs last because it is the only leg whose runs
#: are not parameter-identical to the anchor, so a VRAM or width problem there
#: cannot cost the density ladder its GPU time.
def _plan() -> list[tuple]:
    p: list[tuple] = []
    for name, thr in THR.items():
        for s in SEEDS:
            p.append(("A", f"io_{name}_s{s}", 512,
                      {"arch": "interface", "iface_binary_input": True,
                       "iface_thr_in": thr}, s, "if_binin_s0",
                      frozenset({"run_name", "seed", "iface_thr_in"})))
    for s in SEEDS:
        p.append(("B1", f"io_mall_s{s}", 512,
                  {"arch": "interface", "iface_read_layers": "all"}, s,
                  "if_nest_s0",
                  frozenset({"run_name", "seed", "iface_read_layers"})))
    for s in SEEDS:
        p.append(("B2", f"io_wide_s{s}", 553, {"arch": "snn"}, s,
                  "if_anchor_s0", frozenset({"run_name", "seed", "d_model"})))
    return p


PLAN = _plan()

#: Cells reused from EXP_020 rather than re-run. Named here so the manifest
#: records what the resolver will read, and so a missing one aborts BEFORE
#: 1.8 GPU-h are spent on a ladder that cannot be resolved.
REUSED = tuple(f"if_anchor_s{s}" for s in SEEDS) + tuple(
    f"if_binin_s{s}" for s in SEEDS)


def expected_params(d: int, ov: dict) -> int:
    if ov.get("arch") == "snn":
        return spiking_param_count(VOCAB, d, LAYERS)
    blocks = LAYERS if ov.get("iface_read_layers") == "all" else 1
    blocks *= int(ov.get("iface_read_lags", 1))
    # `iface_thr_in` does not appear: the density ladder changes the code's
    # alphabet and not its shape, which is what makes leg A parameter-identical.
    return folded_param_count(VOCAB, d, LAYERS, folded=bool(ov.get("iface_fold")),
                              read_blocks=blocks)


def prereg_sha256() -> str:
    return hashlib.sha256(PREREG.read_bytes()).hexdigest()


def is_complete(run: str) -> bool:
    """True iff `run` finished its full budget AND was scored on test."""
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
    if ov.get("iface_binary_input"):
        argv += ["--iface_binary_input"]
    if "iface_thr_in" in ov:
        # Ten decimal places: the gain the module derives is a function of this
        # number, so a truncated threshold is a different arm with the same name.
        argv += ["--iface_thr_in", f"{ov['iface_thr_in']:.10f}"]
    if ov.get("iface_read_layers"):
        argv += ["--iface_read_layers", str(ov["iface_read_layers"])]
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
    ABSENT log reads as diverged here rather than as clean.
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


def _config_diff(run: str, ref: str) -> tuple[dict, dict]:
    """(real differences, fields that did not exist when the reference trained).

    K1 ASKS WHETHER ONE THING CHANGED IN THE ARM. A `Config` field added to the
    codebase after the reference run finished is not that: the reference's own
    `config.json` has no such key, and the arm's value for it is the dataclass
    default -- which is, by construction, the behaviour the reference had.

    This gate could not tell those apart, and it fired on exactly that:
    `iface_code_sd` and `nm_pos_len` were added in this session and are ABSENT
    from every EXP_020 and EXP_021 config. `iface_code_sd = 1.0` multiplies the
    code's shadow by 1.0 and leaves `q = 1 - Phi(thr/1.0)`, i.e. the formula the
    reference ran; `nm_pos_len` is read only by `LocalDopamineCharLM`, which
    `arch="interface"` never constructs.

    The exemption is deliberately narrow and deliberately loud. A field is
    exempt ONLY if it is absent from the reference AND exactly equal to the
    `Config` default in the arm -- a new field set to a NON-default value is
    still a second thing changed and still aborts. Every exempted field is
    written into the manifest as `K1_new_fields_at_default`, so it appears in
    the record rather than disappearing into a widened `may_differ`.
    """
    a = json.loads((RUNS / run / "config.json").read_text(encoding="utf-8"))
    b = json.loads((RUNS / ref / "config.json").read_text(encoding="utf-8"))
    diff, new_at_default = {}, {}
    for k in set(a) | set(b):
        if a.get(k) == b.get(k):
            continue
        if k not in b and k in _DEFAULTS and a.get(k) == _DEFAULTS[k]:
            new_at_default[k] = a.get(k)
            continue
        diff[k] = (b.get(k), a.get(k))
    return diff, new_at_default


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out",
                    default="docs/reports/data/exp_022_run_manifest.json")
    ap.add_argument("--steps", type=int, default=20_000)
    ap.add_argument("--only", default="", help="comma-separated run names")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    _check_no_campaign()
    if not PREREG.exists():
        raise SystemExit(
            f"entry condition: {PREREG.relative_to(_REPO)} does not exist. The "
            "pre-registration is committed before the first run, not after it.")
    for art, why in ((PREFLIGHT, "the cost pre-flight (§2.7) prices every arm "
                                 "before it runs; the project's one checked "
                                 "GPU-hour estimate was low by 3.1x"),
                     (SCREEN, "the §7.2 reachability screen (§2.6) runs before "
                              "the first step, not after the ladder")):
        if not art.exists():
            raise SystemExit(
                f"entry condition: {art.relative_to(_REPO)} is missing -- {why}.")
    missing = [r for r in REUSED if not is_complete(r)]
    if missing:
        raise SystemExit(
            f"entry condition: the reused EXP_020 cells {missing} are not "
            "complete. Leg A's q = 0.50 rung and the analogue reference are "
            "those runs; the ladder does not start without the cells its own "
            "bars are taken against.")

    sha_before = prereg_sha256()
    only = {s for s in args.only.split(",") if s}
    manifest = {
        "experiment": "EXP_022",
        "prereg_sha256_before": sha_before,
        "steps": args.steps,
        "vram_alarm_gib": VRAM_ALARM_GIB,
        "reused_from_exp_020": list(REUSED),
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
                diff, new_fields = _config_diff(run, ref)
                extra = set(diff) - set(may_differ)
                rec["K1_diff"] = {k: list(v) for k, v in diff.items()}
                rec["K1_new_fields_at_default"] = new_fields
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
