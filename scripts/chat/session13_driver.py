"""Session 13 driver: the bounded-estimator round, `arch` the single variable.

Pre-registered in `docs/chat/PREDICTION_v13.md`, whose sha256 was stamped at
`027542a6...` **before this driver was launched**. Scored by
`scripts/chat/score_v13.py`.

NOT PART OF THE RESEARCH PROTOCOL. `snnchat` is outside the Phase-1..5 study.
Nothing under `src/snn/` is touched and no research decision is ruled.

    python scripts/launch.py --script scripts/chat/session13_driver.py \
        --run-name _v13_driver

TWO ARMS, ONE VARIABLE
----------------------
Both arms are `chat-v3d-aligned`'s session-7 command line, recovered verbatim
from `experiments/chat/session7-driver/stdout.log`, with `--arch` and
`--run-name` the only differences. **Same seed on purpose**:
`MixtureSampler.batch(s)` is a pure function of `(seed, step)` and both arms
start from the same weights, so the two trajectories see the identical data in
the identical order from the identical point and separate only through the
gradient.

RESTARTABLE THE WAY `CLAUDE.md` ASKS
-------------------------------------
A run whose `summary.json` records the full step count is skipped; a partial run
directory is **deleted** rather than resumed, so no artifact ever describes two
runs at once.

THE CHILD'S STDOUT IS CAPTURED, AND THAT IS A FIX
---------------------------------------------------
`docs/chat/BUILD_NOTES.md` line 522 records that `session7_driver.py:91` calls
`subprocess.run(cmd, cwd=REPO)` with no `stdout=`, so **no child's stdout was
captured for any run** -- which is why `chat-v3d-aligned-s1`'s 7-second death
with rc 3221225786 could not be diagnosed, and why "no banner, not even the
pre-CUDA one" said nothing about how far it got. That note names the remedy:
*"one keyword argument: capture the child's stdout to the run directory."* This
driver does that.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
CHAT = REPO / "experiments" / "chat"
STEPS = 14000

#: `chat-v3d-aligned`'s session-7 command line, minus `--run-name` and `--arch`.
COMMON = [
    "--out-dir", "experiments/chat",
    "--init-from", "experiments/chat/chat-v2-anneal/ckpt_best.pt",
    "--no-resume", "--max-steps", str(STEPS), "--batch-size", "160",
    "--seq-len", "256", "--lr", "5e-4", "--bot-loss-weight", "3.0",
    "--align-frac", "0.75", "--align-lookahead", "1024",
    "--budget-minutes", "75", "--warmup-steps", "200", "--seed", "0",
    "--mix", "stories_topic=0.40", "soda=0.20", "alpaca=0.15",
    "tinystories=0.09", "persona=0.08", "dolly=0.06", "oasst1=0.02",
]

ARMS = [("chat-v13-ctl", "twocomp_threshold"),
        ("chat-v13-det", "twocomp_threshold_detach")]


def complete(name: str) -> bool:
    s = CHAT / name / "summary.json"
    if not s.exists():
        return False
    try:
        return json.loads(s.read_text(encoding="utf-8")).get("steps") == STEPS
    except Exception:
        return False


def log(msg: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def main() -> int:
    log(f"V13_START {len(ARMS)} arms planned")
    for name, arch in ARMS:
        if complete(name):
            log(f"SKIP {name} (already complete at {STEPS} steps)")
            continue
        d = CHAT / name
        if d.exists():
            log(f"DELETE partial {name}")
            shutil.rmtree(d)
        cmd = [sys.executable, "-u", "scripts/chat/train.py",
               "--run-name", name, "--arch", arch] + COMMON
        log(f"TRAIN_START {name} arch={arch}")
        log(f"  cmd: {' '.join(cmd)}")
        d.mkdir(parents=True, exist_ok=True)
        with open(d / "stdout.log", "w", encoding="utf-8", errors="replace") as fh:
            rc = subprocess.run(cmd, cwd=REPO, stdout=fh,
                                stderr=subprocess.STDOUT).returncode
        if rc != 0 or not complete(name):
            log(f"TRAIN_FAILED {name} rc={rc} complete={complete(name)}")
            continue
        log(f"TRAIN_DONE {name}")
    log("V13_DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
