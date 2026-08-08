"""Session-7 driver: train + score the seed-robustness campaign in
`docs/chat/PREDICTION_v7.md`, unattended.

Launched detached via `scripts/launch.py` (see `docs/chat/README.md` and
`snn-v2-research-protocol`'s "long runs die under the harness's 10-minute
Bash timeout" rule) so it survives across the whole session regardless of
what happens to the interactive harness. Restartable by design, same rule as
every other unattended driver in this project: a run whose `summary.json`
already shows the full step count is skipped; a partial run directory is
deleted, never resumed, so no artifact on disk describes two runs at once.

Not part of the research protocol. Writes only under `experiments/chat/`
and `docs/chat/` (the latter not from this file -- write-up is done by hand
after this driver reports done).

    python scripts/chat/session7_driver.py
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PY = sys.executable
MAX_STEPS = 14000
COMMON = [
    "--out-dir", "experiments/chat",
    "--init-from", "experiments/chat/chat-v2-anneal/ckpt_best.pt",
    "--no-resume",
    "--max-steps", str(MAX_STEPS),
    "--batch-size", "160",
    "--seq-len", "256",
    "--lr", "5e-4",
    "--bot-loss-weight", "3.0",
    "--align-frac", "0.75",
    "--align-lookahead", "1024",
    "--budget-minutes", "75",
]
V3D_MIX = ["--mix",
           "stories_topic=0.40", "soda=0.20", "alpaca=0.15", "tinystories=0.09",
           "persona=0.08", "dolly=0.06", "oasst1=0.02"]
INSTA_MIX = ["--mix",
             "stories_short=0.40", "soda=0.07", "alpaca=0.26", "tinystories=0.09",
             "persona=0.08", "dolly=0.08", "oasst1=0.02"]

# (run_name, seed, mix, extra_flags) -- PREDICTION_v7.md SS2, fixed in advance.
RUNS = [
    ("chat-v3d-aligned-s1", 1, V3D_MIX, ["--warmup-steps", "200"]),
    ("chat-v3d-aligned-s2", 2, V3D_MIX, ["--warmup-steps", "200"]),
    ("chat-v3d-aligned-s3", 3, V3D_MIX, ["--warmup-steps", "200"]),
    ("chat-v6-inst-a-s2", 2, INSTA_MIX, []),
    ("chat-v6-inst-a-s3", 3, INSTA_MIX, []),
]


def log(msg: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def is_complete(run_dir: Path) -> bool:
    summary = run_dir / "summary.json"
    if not summary.exists():
        return False
    try:
        d = json.loads(summary.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        log(f"  WARN unreadable summary.json at {summary}: {exc}")
        return False
    return int(d.get("steps", 0)) >= MAX_STEPS


def train_one(run_name: str, seed: int, mix: list[str], extra: list[str]) -> bool:
    run_dir = REPO / "experiments" / "chat" / run_name
    if is_complete(run_dir):
        log(f"SKIP {run_name} (summary.json already shows {MAX_STEPS} steps)")
        return True
    if run_dir.exists():
        log(f"DELETE partial {run_dir} (never resumed, per restart rule)")
        shutil.rmtree(run_dir)

    cmd = [PY, "-u", "scripts/chat/train.py",
           "--run-name", run_name, "--seed", str(seed),
           *COMMON, *extra, *mix]
    log(f"TRAIN_START {run_name} seed={seed}")
    log(f"  cmd: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=REPO)
    if result.returncode != 0:
        log(f"TRAIN_FAILED {run_name} rc={result.returncode}")
        return False
    if not is_complete(run_dir):
        log(f"TRAIN_INCOMPLETE {run_name} (exited 0 but steps < {MAX_STEPS} -- "
            f"budget-minutes likely elapsed first)")
        return False
    log(f"TRAIN_DONE {run_name}")
    return True


def eval_one(run_name: str) -> bool:
    ckpt = REPO / "experiments" / "chat" / run_name / "ckpt_best.pt"
    out = REPO / "experiments" / "chat" / "_quality" / f"{run_name}.json"
    if out.exists():
        log(f"SKIP_EVAL {run_name} (quality json already exists)")
        return True
    cmd = [PY, "-u", "scripts/chat/quality.py",
           "--ckpt", str(ckpt.relative_to(REPO)),
           "--label", run_name,
           "--out", str(out.relative_to(REPO))]
    log(f"EVAL_START {run_name}")
    result = subprocess.run(cmd, cwd=REPO)
    if result.returncode != 0:
        log(f"EVAL_FAILED {run_name} rc={result.returncode}")
        return False
    log(f"EVAL_DONE {run_name}")
    return True


def main() -> int:
    log(f"SESSION7_START {len(RUNS)} runs planned")
    ok_count = 0
    for run_name, seed, mix, extra in RUNS:
        trained = train_one(run_name, seed, mix, extra)
        if trained:
            evaled = eval_one(run_name)
            if evaled:
                ok_count += 1
        else:
            log(f"  skipping eval for {run_name} (training did not complete cleanly)")
    log(f"SESSION7_ALL_DONE {ok_count}/{len(RUNS)} runs trained and scored")
    return 0 if ok_count == len(RUNS) else 1


if __name__ == "__main__":
    raise SystemExit(main())
