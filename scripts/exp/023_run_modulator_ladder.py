"""EXP_023 legs 1-5: four rungs of one modulator form, and three cells taken to n = 6.

    python scripts/launch.py --script scripts/exp/023_run_modulator_ladder.py \
        --run-name _exp023_driver -- --out docs/reports/data/exp_023_run_manifest.json

WHY THIS IS A DRIVER AND NOT FIFTEEN COMMANDS
----------------------------------------------
`CONTRIBUTING.md` §6: restartable, deletes partials rather than resuming them,
and makes "one thing changed" a *check* rather than a claim -- every arm's
`config.json` is diffed field by field against its leg's named reference run,
and anything outside the declared `may_differ` set aborts the ladder.

THE LADDER, AND WHY IT IS ONE FORM WITH FOUR DRIVERS
-----------------------------------------------------
    cur^k = cur^k * (1 + kappa_k * DA^k_t)          IDENTICAL at every rung

    const   DA = 1                    a static per-channel gain, which by
                                      EXP_008 Identity 1 is a learned per-channel
                                      THRESHOLD -- adopted arm #5, measured at
                                      0.0590 bpc, i.e. LARGER than EXP_021's
                                      headline. Until this rung runs, the project
                                      cannot say whether its feedforward
                                      modulator won by being a modulator.
    pos     DA = tanh(w_t)            a learned scalar per WINDOW POSITION.
                                      Data-independent. It is the signal `rolled`
                                      LEAKS: batch-rolling preserves the time
                                      index, so a misaligned RPE still carries
                                      "how far into the window am I".
    rolled  another sequence's RPE    EXP_021's control, unchanged.
    local   this sequence's RPE       EXP_021's arm, unchanged.

WHAT IT REFUSES TO DO
---------------------
  * It reads no bpc and resolves no bar. `023_modulator_results.py` does that,
    and is committed before any run's bpc is read.
  * It changes no hyperparameter, and proves it: G5 reads the frozen recipe back
    out of each run's own `config.json`, INCLUDING `nm_tau` to five significant
    figures -- EXP_021's calibrated constant, reused and never re-fitted.
  * It does not stop when a run diverges, does not reseed, adopts nothing and
    ranks nothing.

WHY ADDING SEEDS IS LEGAL HERE
-------------------------------
`CONTRIBUTING.md` §4 forbids adding seeds to a cell after seeing its result
**within one experiment**, which is exactly why EXP_021 §11 item 2 priced this
at ~0.6 GPU-h and did not run it. These are fresh seeds under a NEW
pre-registration whose §4 registers the fresh-seed-only replication (H6) and the
pooled n = 6 estimate SEPARATELY, so a replication and a power increase are
never the same number.

G8's SECOND CLAUSE, WHICH THIS DRIVER ENFORCES
------------------------------------------------
`nm_pos` starts at a one-sided saddle: at `w = 0` the gradient reaching `kappa`
is EXACTLY zero, and `w`'s own gradient is not, so `w` moves at step 1 and
`kappa` becomes reachable at step 2. Measured in a 400-step GPU run before the
pre-registration was committed (`rms(nm_gain) = 0.2559` from a 0.1 init). This
driver reads `rms(nm_gain)` back out of every `nm_pos` checkpoint, because a
rung whose gain never unlocked would train, score plausibly, and be
`nm_source="off"` wearing an arm's name.
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

import torch

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO / "src") not in sys.path:
    sys.path.insert(0, str(_REPO / "src"))

from snn.config import Config  # noqa: E402
from snn.model import local_dopamine_param_count, spiking_param_count  # noqa: E402

RUNS = _REPO / "experiments" / "runs"
LOCKFILE = _REPO / "src" / "snn" / ".MUTATION_CAMPAIGN_RUNNING"
PREREG = _REPO / "experiments" / "logs" / "EXP_023_modulator_ladder.md"
PREFLIGHT = _REPO / "docs" / "reports" / "data" / "exp_022_023_cost_preflight.json"
SCREEN = _REPO / "docs" / "reports" / "data" / "exp_022_023_reachability.json"
CALIB = _REPO / "docs" / "reports" / "data" / "exp_021_nm_calibration.json"

_DEFAULTS = {f.name: f.default for f in dataclasses.fields(Config)}

VOCAB = 205
LAYERS = 2
VRAM_ALARM_GIB = 4.0

#: EXP_021's calibrated squash scale, reused unchanged. Compared to five
#: significant figures rather than exactly: the value in the pre-registration is
#: a transcribed decimal and a 1e-9 comparison against a full-precision float
#: would fail for a reason that is about transcription and not about the arm.
NM_TAU = 0.057321
NM_TAU_SIGFIGS = 5
NM_GAIN_INIT = 0.1
NM_POS_LEN = 256

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
    "d_model": 512,
}

FRESH = (3, 4, 5)          # legs 1, 4, 5: seeds added to EXP_021's own cells
NEW = (0, 1, 2)            # legs 2, 3: rungs that did not exist before


#: (leg, run_name, arch, nm_source, seed, reference_run, may_differ)
#:
#: ORDER IS LOAD-BEARING. Leg 1 runs first because legs 2 and 3 are scored
#: against the POOLED anchor, and a reference must exist before the run diffed
#: against it.
def _plan() -> list[tuple]:
    p: list[tuple] = []
    for s in FRESH:
        p.append(("1", f"if_anchor_s{s}", "snn", None, s, "if_anchor_s0",
                  frozenset({"run_name", "seed"})))
    for s in NEW:
        p.append(("2", f"nm_const_s{s}", "localdopamine", "const", s,
                  "nm_local_s0", frozenset({"run_name", "seed", "nm_source"})))
    for s in NEW:
        p.append(("3", f"nm_pos_s{s}", "localdopamine", "pos", s,
                  "nm_local_s0", frozenset({"run_name", "seed", "nm_source"})))
    for s in FRESH:
        p.append(("4", f"nm_local_s{s}", "localdopamine", "local", s,
                  "nm_local_s0", frozenset({"run_name", "seed"})))
    for s in FRESH:
        p.append(("5", f"nm_rolled_s{s}", "localdopamine", "rolled", s,
                  "nm_rolled_s0", frozenset({"run_name", "seed"})))
    return p


PLAN = _plan()

#: EXP_021 cells this experiment pools with. A missing one aborts BEFORE
#: 2.1 GPU-h are spent on a ladder that cannot be resolved.
REUSED = tuple(f"nm_local_s{s}" for s in NEW) + tuple(
    f"nm_rolled_s{s}" for s in NEW) + tuple(f"if_anchor_s{s}" for s in NEW)


def expected_params(arch: str, source: str | None) -> int:
    if arch == "snn":
        return spiking_param_count(VOCAB, 512, LAYERS)
    return local_dopamine_param_count(VOCAB, 512, LAYERS, source=source,
                                      pos_len=NM_POS_LEN)


def prereg_sha256() -> str:
    return hashlib.sha256(PREREG.read_bytes()).hexdigest()


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
            "applying wrong kernels to the source tree. Refusing to start a run "
            "that would import one.")


def _train_argv(run: str, arch: str, source: str | None, seed: int,
                steps: int) -> list[str]:
    argv = [
        str(_REPO / "scripts" / "train.py"),
        "--arch", arch,
        "--d_model", "512",
        "--n_layers", str(LAYERS),
        "--run_name", run,
        "--seed", str(seed),
        "--max_steps", str(steps),
        "--eval_every", "2500",
        "--eval_max_windows", "2560",
        "--ckpt_every", "5000",
        "--log_every", "250",
    ]
    if source is not None:
        # Passed explicitly rather than left to a default: a localdopamine run
        # that silently took `nm_source="off"` would train the baseline and be
        # indistinguishable from a null.
        argv += ["--nm_source", source,
                 "--nm_gain_init", str(NM_GAIN_INIT),
                 "--nm_tau", f"{NM_TAU:.6f}",
                 "--nm_pos_len", str(NM_POS_LEN)]
    return argv


def _eval_argv(run: str) -> list[str]:
    return [
        str(_REPO / "scripts" / "evaluate.py"),
        "--ckpt", str(RUNS / run / "ckpt_final.pt"),
        "--split", "test",
        "--out", str(RUNS / run / "final_test.json"),
    ]


def _run(argv: list[str], label: str, fatal: bool = True) -> tuple[float, int]:
    print(f"\n=== {label}\n    {' '.join(argv)}", flush=True)
    t0 = time.time()
    proc = subprocess.run([sys.executable, "-u", *argv], cwd=str(_REPO))
    dt = time.time() - t0
    if proc.returncode != 0 and fatal:
        raise SystemExit(f"{label} exited {proc.returncode}")
    return dt, proc.returncode


def _diverged(run: str) -> bool:
    """G7: read the LOG, not the exit code."""
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


def _gain_rms(run: str) -> dict | None:
    """G8 second clause: did `kappa` actually leave its init?

    Read from the run's own final checkpoint rather than from a log line,
    because `EXP_018` §10 item 11 refers exactly the class of error where a
    training-time number is used as a stand-in for a model property.
    """
    ck = RUNS / run / "ckpt_final.pt"
    if not ck.exists():
        return None
    sd = torch.load(ck, map_location="cpu", weights_only=False)
    sd = sd.get("model", sd)
    out: dict = {}
    for k, v in sd.items():
        if k.startswith("nm_gain") or k.startswith("nm_pos"):
            out[k] = round(float(v.float().pow(2).mean().sqrt()), 6)
    return out or None


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
                    default="docs/reports/data/exp_023_run_manifest.json")
    ap.add_argument("--steps", type=int, default=20_000)
    ap.add_argument("--only", default="", help="comma-separated run names")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    _check_no_campaign()
    if not PREREG.exists():
        raise SystemExit(
            f"entry condition: {PREREG.relative_to(_REPO)} does not exist. The "
            "pre-registration is committed before the first run, not after it.")
    for art, why in ((PREFLIGHT, "the cost pre-flight (§2.6) prices every rung"),
                     (SCREEN, "the §7.2 reachability screen (§2.5) runs before "
                              "the first step"),
                     (CALIB, "nm_tau is EXP_021's calibrated constant and this "
                             "ladder reuses it rather than re-fitting it")):
        if not art.exists():
            raise SystemExit(
                f"entry condition: {art.relative_to(_REPO)} is missing -- {why}.")
    missing = [r for r in REUSED if not is_complete(r)]
    if missing:
        raise SystemExit(
            f"entry condition: the EXP_021 cells {missing} are not complete. "
            "This ladder pools with them; it does not start without them.")

    sha_before = prereg_sha256()
    only = {s for s in args.only.split(",") if s}
    manifest = {
        "experiment": "EXP_023",
        "prereg_sha256_before": sha_before,
        "steps": args.steps,
        "vram_alarm_gib": VRAM_ALARM_GIB,
        "nm_tau": NM_TAU,
        "nm_gain_init": NM_GAIN_INIT,
        "pooled_with_exp_021": list(REUSED),
        "runs": [],
    }

    for leg, run, arch, source, seed, ref, may_differ in PLAN:
        if only and run not in only:
            continue
        exp = expected_params(arch, source)
        rec = {"leg": leg, "run": run, "arch": arch, "nm_source": source,
               "seed": seed, "reference": ref, "expected_params": exp,
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
                print(f"[dry-run] {leg} {run} {arch}/{source} seed={seed} "
                      f"params~{exp:,}", flush=True)
                manifest["runs"].append(rec)
                continue
            t, code = _run(_train_argv(run, arch, source, seed, args.steps),
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
            viol = {k: c.get(k) for k, v in FROZEN.items() if c.get(k) != v}
            if source is not None:
                got, want = c.get("nm_tau"), NM_TAU
                if got is None or (round(float(got), NM_TAU_SIGFIGS)
                                   != round(want, NM_TAU_SIGFIGS)):
                    viol["nm_tau"] = got
                if c.get("nm_gain_init") != NM_GAIN_INIT:
                    viol["nm_gain_init"] = c.get("nm_gain_init")
            rec["G5_frozen_violations"] = viol
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
        if source is not None:
            # G8 second clause. Recorded for every modulated rung, not only
            # `pos`, so the ladder's own findability figure (H7) comes from the
            # checkpoints rather than from a separate pass over them.
            rec["gain_rms"] = _gain_rms(run)
            if source == "pos" and rec["gain_rms"]:
                moved = any(abs(v - NM_GAIN_INIT) > 1e-6
                            for k, v in rec["gain_rms"].items()
                            if k.startswith("nm_gain"))
                rec["G8_pos_unlocked"] = moved
                if not moved:
                    print(f"    !! G8: {run} finished with nm_gain still at its "
                          f"{NM_GAIN_INIT} init -- the pos rung never unlocked. "
                          "Recorded; its bpc is NOT an arm result.", flush=True)
        manifest["runs"].append(rec)
        print(f"    -> params={rec.get('params')} (expected {exp}) "
              f"diverged={rec['diverged']} wall={rec.get('wall_clock_s')}s "
              f"gain_rms={rec.get('gain_rms')}", flush=True)

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
