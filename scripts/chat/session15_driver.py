"""Session 15 driver: the recall round, two training seeds, one detached process.

Pre-registered in `docs/chat/PREDICTION_v15.md`; scored by
`scripts/chat/score_v15.py`. NOT PART OF THE RESEARCH PROTOCOL: nothing under
`src/snn/` is touched, no research decision is ruled, and
`experiments/chat/SHIPPED` and `build_corpus.DEFAULT_MIX` are edited under no
outcome.

    python scripts/launch.py --script scripts/chat/session15_driver.py \
        --run-name _v15_driver --out-dir <OUT_ROOT> \
        -- --out-root <OUT_ROOT> --data-dir <PACKED CORPUS WITH recall>

Everything this driver's own flags need goes AFTER the `--`. Its flags are
deliberately not called `--out-dir` or `--run-name`: `launch.py` scans the
forwarded arguments for those two to decide where ITS log goes.

ONE VARIABLE
------------
The recipe is `chat-v3d-aligned`'s, and it is READ from that run's
`config.json` rather than retyped: every `ChatConfig` field becomes an explicit
flag, because `train.py`'s defaults disagree with the shipped recipe on a dozen
fields and a flag nobody passed is a default nobody chose. The one change is the
mix: `recall` enters at 0.06 and `alpaca` gives up exactly that; every other
weight is carried over untouched and the driver checks that it was.

`--init-from` is NOT in `config.json` (the trainer does not record it). The
parent is `chat-v2-anneal/ckpt_best.pt`, as `session13_driver.COMMON` records
it, recovered there from `experiments/chat/session7-driver/stdout.log`. The
`--budget-minutes 75` of that command line is kept for the same reason.

THE K1-STYLE GATE
-----------------
Before anything trains, the config the child WILL build is constructed with
`train.py`'s own parser from the exact command line, and diffed field by field
against the reference `config.json`. Anything outside `MAY_DIFFER` aborts. The
trainer fills `vocab_size` itself, so that field is checked on the smoke run's
real `config.json` instead, and each trained run's real `config.json` is diffed
again before it is scored.

STAGES, IN ORDER
----------------
0. **preflight** -- refuse if another python process holds the GPU (by process
   NAME: on this WDDM box `used_memory` reads N/A); the corpus has `recall` on
   disk and in the manifest; the pre-registration, the generator, both probes,
   the scorer and this file are COMMITTED and unmodified, with HEAD and each
   file's blob recorded; the config gate above. On a restart, a recorded blob
   that has changed while a v15 checkpoint exists is refused outright: the
   scorer must precede the checkpoints it scores.
1. **floors** -- the committed probe, `memory_probe_v2` and the binding probe on
   the SHIPPED checkpoint.
2. **smoke** -- 20 steps of the real command; the banner's mix line must show
   `recall 6%` and no "requested source(s) not in the corpus" warning.
   `MixtureSampler` renormalises over what it finds, so a missing source does
   not fail -- it trains a different mixture and says so in a log nobody reads.
3. **train** -- `chat-v15-recall-s0`, then `-s1`. Sequentially, never
   concurrently. A run whose `summary.json` records the full step count is
   skipped; a partial run directory is DELETED, never resumed, so no artifact
   describes two runs at once.
4. **score** -- per checkpoint: committed probe, `memory_probe_v2` (n=8, all
   strata), its one n=256 pass at distance 0, the binding probe, the battery,
   HELDOUT at the shipped decoder. Each is cached on its artifact.
5. **verdict** -- `score_v15.py`, which uses no GPU.

A failure in stages 0-2 stops the driver: nothing trains behind a refused
pre-flight, a probe that cannot read SHIPPED, or a wrong mixture. After that, a
seed that fails is reported as failed and the driver moves on to the next; it
is not retried and not replaced.

POLL THE STATUS FILE, NOT THE PROCESS
-------------------------------------
`launch.py` returns at once and is invisible to a harness waiting on it.
`<OUT_ROOT>/status.json` is rewritten after every stage and every 15 s while a
child runs: `state` is RUNNING, DONE, FAILED or DIVERGED, `finished` says
whether the driver has exited, and each stage carries its start, finish, exit
code, artifacts and error. A `divergence` event in a run's `log.jsonl` sets
DIVERGED while the trainer is still rolling back, and a crash in this file
writes FAILED with the traceback before it dies.

`--dry-run` prints every command with every path resolved, and runs nothing.
"""
from __future__ import annotations

import argparse
import dataclasses
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

#: Where the checkpoints live. The worktree has no `experiments/chat/*.pt`; the
#: main tree is READ here and never written.
DEFAULT_CHAT_ROOT = Path(r"C:\Elliot's Stuff\SNN\Text SNN\experiments\chat")

REFERENCE_RUN = "chat-v3d-aligned"
#: Not recorded in `config.json`; see the module docstring.
PARENT = "chat-v2-anneal/ckpt_best.pt"
BUDGET_MINUTES = "75"

#: (run name, training seed). n = 2, fixed in advance.
RUNS: tuple[tuple[str, int], ...] = (("chat-v15-recall-s0", 0), ("chat-v15-recall-s1", 1))
SMOKE_RUN = "chat-v15-smoke"
SMOKE_STEPS = 20

NEW_SOURCE, DONOR, NEW_WEIGHT = "recall", "alpaca", 0.06

#: The declared set. Anything else that differs from the reference aborts.
MAY_DIFFER = frozenset({"mix", "run_name", "seed", "out_dir", "data_dir"})
#: `ChatConfig` fields `train.py` has no flag for.
_NO_FLAG = frozenset({"mix", "vocab_size", "vocab_version"})

#: What must be committed and unmodified before anything runs.
PREREGISTERED: tuple[str, ...] = (
    "docs/chat/PREDICTION_v15.md",
    "src/snnchat/recall.py",
    "scripts/chat/memory_probe.py",
    "scripts/chat/memory_probe_v2.py",
    "scripts/chat/binding_probe.py",
    "scripts/chat/score_v15.py",
    "scripts/chat/session15_driver.py",
)

HEARTBEAT_S = 15.0

#: Nothing may train past a failure in one of these: a refused pre-flight, an
#: instrument that cannot read the SHIPPED checkpoint, or a smoke run whose
#: mixture is not the one that was asked for.
_BLOCKING = frozenset({"preflight", "floors", "smoke"})


class Refusal(RuntimeError):
    """A pre-flight condition is not met. The message says which, and why."""


# ---------------------------------------------------------------------------
# the recipe
# ---------------------------------------------------------------------------


def v15_mix(ref_mix: dict[str, float]) -> dict[str, float]:
    """The reference mix with `NEW_WEIGHT` moved from `DONOR` to `NEW_SOURCE`."""
    if NEW_SOURCE in ref_mix or DONOR not in ref_mix:
        raise Refusal(f"reference mix {ref_mix} cannot donate {DONOR} -> {NEW_SOURCE}")
    mix = dict(ref_mix)
    mix[DONOR] = round(ref_mix[DONOR] - NEW_WEIGHT, 10)
    mix[NEW_SOURCE] = NEW_WEIGHT
    if mix[DONOR] <= 0 or abs(sum(mix.values()) - sum(ref_mix.values())) > 1e-9:
        raise Refusal(f"mix does not conserve weight: {mix}")
    return mix


def check_mix(ref_mix: dict, mix: dict) -> None:
    """Every non-donor weight exactly as committed; the donor down by the new weight."""
    for name, w in ref_mix.items():
        want = round(w - NEW_WEIGHT, 10) if name == DONOR else w
        if mix.get(name) != want:
            raise Refusal(f"mix[{name!r}] is {mix.get(name)!r}, the recipe says {want!r}")
    extra = set(mix) - set(ref_mix)
    if extra != {NEW_SOURCE} or mix[NEW_SOURCE] != NEW_WEIGHT:
        raise Refusal(f"the only new source may be {NEW_SOURCE}={NEW_WEIGHT}; got {mix}")


def train_command(ref_cfg: dict, run: str, seed: int, *, data_dir: Path, out_dir: Path,
                  parent: Path, max_steps: int | None = None) -> list[str]:
    """`train.py`'s command line with EVERY `ChatConfig` field stated."""
    from snnchat.model import ChatConfig

    ours = {"run_name": run, "seed": seed, "out_dir": str(out_dir), "data_dir": str(data_dir)}
    if max_steps is not None:
        ours["max_steps"] = max_steps
    cmd = [sys.executable, "-u", str(REPO / "scripts" / "chat" / "train.py")]
    for f in dataclasses.fields(ChatConfig):
        if f.name in _NO_FLAG:
            continue
        if f.name not in ref_cfg and f.name not in ours:
            # A field newer than the reference run: left at its default, which
            # nests the reference exactly. `config_diff` records the exemption.
            continue
        value = ours.get(f.name, ref_cfg.get(f.name))
        flag = "--" + f.name.replace("_", "-")
        if isinstance(value, bool):
            cmd.append(flag if value else "--no-" + f.name.replace("_", "-"))
        else:
            cmd += [flag, str(value)]
    cmd += ["--init-from", str(parent), "--no-resume", "--budget-minutes", BUDGET_MINUTES]
    cmd += ["--mix", *(f"{k}={v}" for k, v in v15_mix(ref_cfg["mix"]).items())]
    return cmd


def simulated_config(cmd: list[str]) -> dict:
    """The `ChatConfig` the child will build, from `train.py`'s own parser."""
    spec = importlib.util.spec_from_file_location(
        "_chat_train_cli", REPO / "scripts" / "chat" / "train.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    from snnchat.model import ChatConfig

    args = mod.build_parser().parse_args(cmd[3:])
    fields = {f.name for f in dataclasses.fields(ChatConfig)}
    cfg = ChatConfig(**{k: v for k, v in vars(args).items() if k in fields})
    cfg.mix = mod._parse_mix(args.mix)
    return dataclasses.asdict(cfg)


def config_diff(ref: dict, new: dict, *, may_differ=MAY_DIFFER, defaults: dict | None = None,
                skip=frozenset()) -> dict:
    """Field-by-field. `violations` is what aborts the ladder.

    A field the reference run predates is exempt ONLY at its default, and the
    exemption is recorded -- the rule `CLAUDE.md` states for the research
    drivers' K1 gate, applied here for the same reason.
    """
    differs, violations, exempt = {}, [], []
    for key in sorted(set(ref) | set(new)):
        if key in skip:
            continue
        if key not in new:
            violations.append(f"{key}: in the reference, missing from the new config")
        elif key not in ref:
            if defaults is not None and new[key] == defaults.get(key):
                exempt.append(key)
            else:
                violations.append(f"{key}: new field away from its default ({new[key]!r})")
        elif ref[key] != new[key]:
            differs[key] = [ref[key], new[key]]
            if key not in may_differ:
                violations.append(f"{key}: {ref[key]!r} -> {new[key]!r}")
    return {"differs": differs, "violations": violations,
            "new_fields_at_default_exempted": exempt}


def check_recipe(ref_cfg: dict, cmd: list[str]) -> dict:
    from snnchat.model import ChatConfig

    new = simulated_config(cmd)
    diff = config_diff(ref_cfg, new, defaults=dataclasses.asdict(ChatConfig()),
                       skip=frozenset({"vocab_size"}))
    if diff["violations"]:
        raise Refusal("config differs from the reference outside the declared set: "
                      + "; ".join(diff["violations"]))
    check_mix(ref_cfg["mix"], new["mix"])
    return diff


def check_written_config(ref_cfg: dict, path: Path, *, extra=frozenset()) -> dict:
    """The same gate on a `config.json` the trainer actually wrote."""
    new = json.loads(path.read_text(encoding="utf-8"))
    diff = config_diff(ref_cfg, new, may_differ=MAY_DIFFER | extra)
    if diff["violations"] or diff["new_fields_at_default_exempted"]:
        raise Refusal(f"{path} differs from the reference outside the declared set: {diff}")
    check_mix(ref_cfg["mix"], new["mix"])
    return diff


# ---------------------------------------------------------------------------
# pre-flight
# ---------------------------------------------------------------------------


def gpu_python_processes(run=subprocess.run) -> list[tuple[int, str]]:
    """Other python processes with a compute context on the GPU.

    By process NAME and pid. `used_memory` is `[N/A]` under WDDM, so memory
    cannot tell an idle context from a training run and is not consulted. This
    process and its parent are excluded.
    """
    try:
        r = run(["nvidia-smi", "--query-compute-apps=pid,name", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError) as exc:
        raise Refusal(f"cannot ask nvidia-smi who holds the GPU: {exc}") from exc
    if r.returncode != 0:
        raise Refusal(f"nvidia-smi exited {r.returncode}: {(r.stderr or '').strip()[:200]}")
    mine = {os.getpid(), os.getppid()}
    out = []
    for line in r.stdout.splitlines():
        pid, _, name = line.partition(",")
        if not pid.strip().isdigit():
            continue
        exe = name.strip().replace("\\", "/").rsplit("/", 1)[-1].lower()
        if exe.startswith("python") and int(pid) not in mine:
            out.append((int(pid), name.strip()))
    return out


def check_gpu(*, force: bool, run=subprocess.run) -> list:
    if force:
        return []
    busy = gpu_python_processes(run)
    if busy:
        raise Refusal("another python process holds the GPU (a timed run may be in "
                      f"flight): {busy}. Wait for it, or pass --force.")
    return busy


def check_corpus(data_dir: Path, mix: dict) -> dict:
    manifest_path = data_dir / "manifest.json"
    if not manifest_path.exists():
        raise Refusal(f"no manifest.json in {data_dir}")
    sources = json.loads(manifest_path.read_text(encoding="utf-8")).get("sources", {})
    for name in mix:
        if name not in sources:
            raise Refusal(f"{name!r} is not in {manifest_path}; listed: {sorted(sources)}")
        for suffix in (".bin", ".val.bin"):
            if not (data_dir / f"{name}{suffix}").exists():
                raise Refusal(f"{data_dir / (name + suffix)} does not exist")
    return {"manifest": str(manifest_path), "sources": sorted(mix),
            NEW_SOURCE: {k: v for k, v in sources[NEW_SOURCE].items() if k != "report"}}


def _git(args: list[str], run) -> subprocess.CompletedProcess:
    return run(["git", *args], cwd=str(REPO), capture_output=True, text=True)


def check_prereg(run=subprocess.run) -> dict:
    """HEAD and each pre-registered file's blob, or a Refusal.

    `git status --porcelain` reports a modified, staged or untracked path; a
    path git has never heard of fails `rev-parse HEAD:<path>`.
    """
    dirty = _git(["status", "--porcelain", "--", *PREREGISTERED], run)
    if dirty.returncode != 0:
        raise Refusal(f"git status failed: {dirty.stderr.strip()[:200]}")
    if dirty.stdout.strip():
        raise Refusal("the pre-registration is not committed as it stands:\n"
                      + dirty.stdout.rstrip())
    head = _git(["rev-parse", "HEAD"], run)
    if head.returncode != 0:
        raise Refusal(f"git rev-parse HEAD failed: {head.stderr.strip()[:200]}")
    blobs = {}
    for path in PREREGISTERED:
        blob = _git(["rev-parse", f"HEAD:{path}"], run)
        if blob.returncode != 0 or not blob.stdout.strip():
            raise Refusal(f"{path} is not committed at HEAD")
        blobs[path] = blob.stdout.strip()
    return {"head": head.stdout.strip(), "blobs": blobs}


def check_ordering(record_path: Path, prereg: dict, runs_root: Path) -> None:
    """On a restart: the scorer that judges a checkpoint must predate it."""
    if not record_path.exists() or not any(runs_root.glob("*/ckpt_*.pt")):
        return
    before = json.loads(record_path.read_text(encoding="utf-8")).get("prereg", {}).get("blobs")
    changed = sorted(p for p in prereg["blobs"] if before and before.get(p) != prereg["blobs"][p])
    if changed:
        raise Refusal("a v15 checkpoint already exists under this --out-root and these "
                      f"pre-registered files changed since it was started: {changed}. "
                      "Use a fresh --out-root; --force does not override this.")


def shipped_checkpoint(chat_root: Path) -> Path:
    first = (chat_root / "SHIPPED").read_text(encoding="utf-8").split("#", 1)[0].strip()
    ckpt = chat_root / first
    if not first or not ckpt.exists():
        raise Refusal(f"SHIPPED names {first!r}, which is not under {chat_root}")
    return ckpt


# ---------------------------------------------------------------------------
# the plan
# ---------------------------------------------------------------------------


def _py(script: str, *args) -> list[str]:
    return [sys.executable, "-u", str(REPO / "scripts" / "chat" / script), *map(str, args)]


def probe_commands(ckpt: Path, out: Path, device: str, *, full: bool) -> list[tuple[list, Path]]:
    """(command, artifact) pairs for one checkpoint. `full` adds the scoring-only ones."""
    cmds = [
        (_py("memory_probe.py", "--ckpt", ckpt, "--seeds", 8, "--device", device,
             "--out", out / "memory_probe.json"), out / "memory_probe.json"),
        (_py("memory_probe_v2.py", "--ckpt", ckpt, "--device", device,
             "--out", out / "memory_probe_v2_n8.json"), out / "memory_probe_v2_n8.json"),
        (_py("binding_probe.py", "--ckpt", ckpt, "--device", device,
             "--out", out / "binding_probe.json"), out / "binding_probe.json"),
    ]
    if full:
        cmds += [
            (_py("memory_probe_v2.py", "--ckpt", ckpt, "--device", device, "--shipped-decoder",
                 "--out", out / "memory_probe_v2_n256.json"), out / "memory_probe_v2_n256.json"),
            (_py("quality.py", "--ckpt", ckpt, "--device", device, "--seeds", 4, "--n", 16,
                 "--no-dependence", "--out", out / "battery.json"), out / "battery.json"),
            (_py("echo_holdout.py", "--ckpt", ckpt, "--device", device, "--set", "heldout",
                 "--seeds", 6, "--n", 256, "--out", out / "heldout_n256.json"),
             out / "heldout_n256.json"),
        ]
    return cmds


def build_plan(args, ref_cfg: dict) -> list[dict]:
    out_root, chat_root = Path(args.out_root), Path(args.chat_root)
    data_dir, parent = Path(args.data_dir), chat_root / PARENT
    plan: list[dict] = [{"stage": "preflight", "commands": []}]
    shipped = shipped_checkpoint(chat_root)
    plan.append({"stage": "floors", "ckpt": shipped,
                 "commands": probe_commands(shipped, out_root / "floors", args.device,
                                            full=False)})
    plan.append({"stage": "smoke", "run_dir": out_root / "smoke" / SMOKE_RUN, "commands": [
        (train_command(ref_cfg, SMOKE_RUN, 0, data_dir=data_dir, out_dir=out_root / "smoke",
                       parent=parent, max_steps=SMOKE_STEPS),
         out_root / "smoke" / SMOKE_RUN / "summary.json")]})
    for run, seed in RUNS:
        plan.append({"stage": f"train:{run}", "run_dir": out_root / "runs" / run, "commands": [
            (train_command(ref_cfg, run, seed, data_dir=data_dir,
                           out_dir=out_root / "runs", parent=parent),
             out_root / "runs" / run / "summary.json")]})
    for run, _seed in RUNS:
        ckpt = out_root / "runs" / run / "ckpt_best.pt"
        plan.append({"stage": f"score:{run}", "ckpt": ckpt, "run_dir": out_root / "runs" / run,
                     "commands": probe_commands(ckpt, out_root / "scores" / run, args.device,
                                                full=True)})
    plan.append({"stage": "verdict", "commands": [
        (_py("score_v15.py", "--out-root", out_root), out_root / "score_v15.json")]})
    return plan


# ---------------------------------------------------------------------------
# running it
# ---------------------------------------------------------------------------


class Status:
    """`status.json`: what the lead polls instead of a process exit."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.data: dict = {"state": "RUNNING", "finished": False, "pid": os.getpid(),
                           "started": _now(), "stage": None, "stages": [], "alerts": []}

    def write(self) -> None:
        self.data["updated"] = _now()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(self.data, indent=1, default=str), encoding="utf-8")
        for _ in range(20):
            try:
                os.replace(tmp, self.path)
                return
            except PermissionError:      # a reader has it open; Windows
                time.sleep(0.25)

    def begin(self, stage: str) -> dict:
        entry = {"stage": stage, "started": _now(), "finished": None, "result": None,
                 "artifacts": [], "error": None}
        self.data["stage"] = stage
        self.data["stages"].append(entry)
        self.write()
        return entry

    def end(self, entry: dict, result: str, error: str | None = None) -> None:
        entry.update(finished=_now(), result=result, error=error)
        if result == "failed":
            # FAILED outranks DIVERGED; the divergence stays in `alerts`.
            self.data["state"] = "FAILED"
        self.write()

    def alert(self, state: str, message: str) -> None:
        if self.data["state"] != "FAILED":
            self.data["state"] = state
        self.data["alerts"].append({"at": _now(), "state": state, "message": message})
        self.write()


def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def log(msg: str) -> None:
    print(f"[{_now()}] {msg}", flush=True)


def complete(run_dir: Path, steps: int) -> bool:
    try:
        summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return summary.get("steps") == steps and (run_dir / "ckpt_best.pt").exists()


def artifact_ok(path: Path) -> bool:
    """A cached probe artifact: parses, and does not call itself a partial."""
    try:
        blob = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return blob.get("complete", True) is not False


def divergences(log_path: Path) -> list[dict]:
    if not log_path.exists():
        return []
    out = []
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            rec = json.loads(line)
        except ValueError:
            continue
        if rec.get("event") == "divergence":
            out.append(rec)
    return out


def run_child(cmd: list[str], stdout_path: Path, status: Status, *, watch: Path | None = None,
              popen=subprocess.Popen) -> int:
    """Run one child with its stdout captured, heartbeating while it runs."""
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    seen = 0
    with open(stdout_path, "a", encoding="utf-8", errors="replace") as fh:
        fh.write(f"\n=== {_now()} === {' '.join(cmd)}\n")
        fh.flush()
        proc = popen(cmd, cwd=str(REPO), stdin=subprocess.DEVNULL, stdout=fh,
                     stderr=subprocess.STDOUT)
        while True:
            try:
                rc = proc.wait(timeout=HEARTBEAT_S)
            except subprocess.TimeoutExpired:
                rc = None
            if watch is not None:
                events = divergences(watch)
                for ev in events[seen:]:
                    status.alert("DIVERGED", f"{watch.parent.name}: loss {ev.get('loss')!r} at "
                                             f"step {ev.get('step')}, {ev.get('action')}")
                seen = len(events)
            if rc is not None:
                return rc
            status.write()


def _safe_rmtree(path: Path, out_root: Path) -> None:
    if not path.resolve().is_relative_to(out_root.resolve()) or path.resolve() == out_root.resolve():
        raise Refusal(f"refusing to delete {path}: not under --out-root")
    shutil.rmtree(path)


def check_smoke(run_dir: Path, ref_cfg: dict) -> dict:
    text = (run_dir / "stdout.log").read_text(encoding="utf-8", errors="replace")
    mix_lines = [ln.strip() for ln in text.splitlines() if ln.strip().startswith("mix:")]
    if not mix_lines or f"{NEW_SOURCE} {NEW_WEIGHT:.0%}" not in mix_lines[-1]:
        raise Refusal(f"smoke run's mix line does not show {NEW_SOURCE} at "
                      f"{NEW_WEIGHT:.0%}: {mix_lines[-1:] or 'no mix line'}")
    if "not in the corpus" in text:
        raise Refusal("smoke run printed a missing-source warning; the mixture that "
                      "trained is not the one that was asked for")
    if not complete(run_dir, SMOKE_STEPS):
        raise Refusal(f"smoke run did not finish {SMOKE_STEPS} steps")
    diff = check_written_config(ref_cfg, run_dir / "config.json",
                                extra=frozenset({"max_steps"}))
    return {"mix_line": mix_lines[-1], "config_diff": diff}


def execute(args, plan: list[dict], ref_cfg: dict, status: Status, *,
            run=subprocess.run, popen=subprocess.Popen) -> int:
    out_root = Path(args.out_root)
    record_path = out_root / "driver_record.json"
    failed_runs: set[str] = set()

    for step in plan:
        stage = step["stage"]
        run_name = stage.split(":", 1)[1] if ":" in stage else None
        entry = status.begin(stage)
        log(f"STAGE_START {stage}")
        try:
            if stage == "preflight":
                check_gpu(force=args.force, run=run)
                prereg = check_prereg(run)
                check_ordering(record_path, prereg, out_root / "runs")
                corpus = check_corpus(Path(args.data_dir), v15_mix(ref_cfg["mix"]))
                first_train = next(s for s in plan if s["stage"].startswith("train:"))
                recipe = check_recipe(ref_cfg, first_train["commands"][0][0])
                record_path.write_text(json.dumps({
                    "prereg": prereg, "corpus": corpus, "recipe_diff": recipe,
                    "reference": str(Path(args.chat_root) / REFERENCE_RUN / "config.json"),
                    "parent": str(Path(args.chat_root) / PARENT),
                    "mix": v15_mix(ref_cfg["mix"]), "recorded": _now(),
                    "plan": [{"stage": s["stage"], "commands": [c for c, _a in s["commands"]]}
                             for s in plan],
                }, indent=1, default=str), encoding="utf-8")
                entry["artifacts"].append(str(record_path))
                log(f"  HEAD {prereg['head']}; config differs only in "
                    f"{sorted(recipe['differs'])}")
                status.end(entry, "ok")
                continue

            if stage.startswith("score:") and run_name in failed_runs:
                status.end(entry, "skipped", f"{run_name} did not train to completion")
                continue
            if stage != "verdict":
                check_gpu(force=args.force, run=run)

            if stage == "smoke" or stage.startswith("train:"):
                steps = SMOKE_STEPS if stage == "smoke" else ref_cfg["max_steps"]
                run_dir = step["run_dir"]
                if stage != "smoke" and complete(run_dir, steps):
                    log(f"  SKIP {run_dir.name} (complete at {steps} steps)")
                    entry["artifacts"].append(str(run_dir / "summary.json"))
                    status.end(entry, "skipped-complete")
                    continue
                if run_dir.exists():
                    log(f"  DELETE partial {run_dir}")
                    _safe_rmtree(run_dir, out_root)
            elif stage.startswith("score:"):
                check_written_config(ref_cfg, step["run_dir"] / "config.json")

            errors = []
            for cmd, artifact in step["commands"]:
                if stage != "smoke" and not stage.startswith("train:") \
                        and stage != "verdict" and artifact_ok(artifact):
                    log(f"  CACHED {artifact}")
                    entry["artifacts"].append(str(artifact))
                    continue
                log(f"  $ {' '.join(cmd)}")
                is_train = stage == "smoke" or stage.startswith("train:")
                stdout = (step["run_dir"] / "stdout.log" if is_train
                          else artifact.with_suffix(".stdout.log"))
                watch = step["run_dir"] / "log.jsonl" if is_train else None
                rc = run_child(cmd, stdout, status, watch=watch, popen=popen)
                if rc != 0 or not artifact.exists():
                    # One probe dying must not cost the stage its other readouts.
                    errors.append(f"{Path(cmd[2]).name}: rc={rc}, artifact "
                                  f"present={artifact.exists()}, see {stdout}")
                    log(f"  FAILED {errors[-1]}")
                    continue
                entry["artifacts"].append(str(artifact))
            if errors:
                raise RuntimeError("; ".join(errors))

            if stage == "smoke":
                entry["smoke"] = check_smoke(step["run_dir"], ref_cfg)
            if stage.startswith("train:") and not complete(step["run_dir"], ref_cfg["max_steps"]):
                raise RuntimeError(f"{run_name} exited 0 without reaching "
                                   f"{ref_cfg['max_steps']} steps (budget stop?)")
            status.end(entry, "ok")
            log(f"STAGE_DONE {stage}")
        except Refusal as exc:
            log(f"REFUSED at {stage}: {exc}")
            status.end(entry, "failed", f"refused: {exc}")
            if stage in _BLOCKING:
                return 2
            failed_runs.add(run_name or stage)
        except Exception as exc:
            log(f"STAGE_FAILED {stage}: {exc}")
            status.end(entry, "failed", f"{type(exc).__name__}: {exc}")
            if stage in _BLOCKING:
                return 1
            failed_runs.add(run_name or stage)
    return 1 if failed_runs else 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-root", required=True,
                    help="everything is written under here; must be OUTSIDE the main tree")
    ap.add_argument("--data-dir", required=True,
                    help="packed corpus: the seven shipped sources plus recall, one manifest")
    ap.add_argument("--chat-root", default=str(DEFAULT_CHAT_ROOT),
                    help="experiments/chat holding SHIPPED, the reference run and the parent")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--force", action="store_true",
                    help="start even though another python process holds the GPU")
    ap.add_argument("--dry-run", action="store_true",
                    help="print every command with its paths resolved; run nothing")
    args = ap.parse_args(argv)
    args.out_root = str(Path(args.out_root).resolve())
    args.data_dir = str(Path(args.data_dir).resolve())
    args.chat_root = str(Path(args.chat_root).resolve())

    main_tree = Path(args.chat_root).parents[1]
    if Path(args.out_root).is_relative_to(main_tree):
        raise SystemExit(f"--out-root {args.out_root} is inside {main_tree}; this driver "
                         "writes nothing into the tree the checkpoints are read from")
    ref_cfg = json.loads((Path(args.chat_root) / REFERENCE_RUN / "config.json")
                         .read_text(encoding="utf-8"))
    plan = build_plan(args, ref_cfg)

    if args.dry_run:
        print(f"out-root  {args.out_root}\ndata-dir  {args.data_dir}\n"
              f"chat-root {args.chat_root}\nmix       {v15_mix(ref_cfg['mix'])}")
        for step in plan:
            print(f"\n[{step['stage']}]")
            if step["stage"] == "preflight":
                print("  checks: gpu free of other python processes; corpus has "
                      f"{NEW_SOURCE}; committed and clean: {', '.join(PREREGISTERED)}; "
                      f"config diff vs {REFERENCE_RUN} within {sorted(MAY_DIFFER)}")
            for cmd, artifact in step["commands"]:
                print(f"  $ {' '.join(cmd)}\n      -> {artifact}")
        return 0

    status = Status(Path(args.out_root) / "status.json")
    log(f"V15_START out-root {args.out_root}")
    try:
        rc = execute(args, plan, ref_cfg, status)
    except BaseException as exc:
        status.data["state"] = "FAILED"
        status.data["crash"] = "".join(traceback.format_exception(exc))[-4000:]
        status.data["finished"] = True
        status.write()
        raise
    if status.data["state"] == "RUNNING":
        status.data["state"] = "DONE" if rc == 0 else "FAILED"
    status.data["finished"] = True
    status.data["stage"] = None
    status.write()
    log(f"V15_DONE state={status.data['state']} rc={rc}")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
