"""Run the v12 request-slot arm: `chat-v12-rare`, four seeds, sequentially.

    python scripts/launch.py --run-name session12-driver --out-dir experiments/chat \\
        -- scripts/chat/session12_driver.py

The bar this arm is scored against is `docs/chat/PREDICTION_v12.md`, committed
before this file ran. Nothing here chooses a metric or a threshold.

WHY A DRIVER RATHER THAN FOUR COMMAND LINES
-------------------------------------------
Same reason `session9_driver.py` exists: a stage that has finished leaves a
marker, so a session that dies half way resumes instead of retraining what it
already has, and the exact command of every seed is recorded next to its
result rather than in a shell history.

WHY THERE IS NO CONTROL ARM HERE
--------------------------------
`PREDICTION_v12.md` §2: the control is the four `chat-v3d-aligned` checkpoints
already on disk, re-scored with today's decoder. They were trained on the
identical recipe and the identical mixture -- the only thing this round changes
is which 400 M characters `stories_topic` names -- so re-measuring a frozen
artifact is what decision #10 asks for rather than what it forbids. It also
removes training-seed drift from the comparison, because the control's seeds
are the ones that are already there.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MARKERS = ROOT / "experiments" / "chat" / "_session12"

#: The shipped recipe every arm anneals from. EVERY VALUE IS READ OFF
#: `chat-v3d-aligned/config.json`, not inherited from `train.py`'s defaults,
#: which disagree with it on five keys (batch_size 64 vs 160, seq_len 384 vs
#: 256, lr 2e-3 vs 5e-4, warmup 500 vs 200, eval_batches 40 vs 30).
#:
#: `--eval-every/--sample-every/--log-every` are pinned here and were NOT pinned
#: in `session9_driver.py`, so the v9 arms silently ran at 2000/2000/100 against
#: the incumbent's 3500/3500/250. It changes no weight, but it is a wall-clock
#: tax and it makes a config diff noisy; this round has parity.
_BASE = [
    "--init-from", "experiments/chat/chat-v2-anneal/ckpt_best.pt",
    "--max-steps", "14000",
    "--bot-loss-weight", "3.0",
    "--align-frac", "0.75",
    "--align-lookahead", "1024",
    "--batch-size", "160",
    "--seq-len", "256",
    "--lr", "0.0005",
    "--warmup-steps", "200",
    "--eval-batches", "30",
    "--eval-every", "3500",
    "--sample-every", "3500",
    "--log-every", "250",
]

#: The incumbent's mixture with ONE substitution, at the same weight.
_MIX = ["--mix", "stories_topic_rare=0.40", "soda=0.20", "alpaca=0.15",
        "tinystories=0.09", "persona=0.08", "dolly=0.06", "oasst1=0.02"]

ARM = "chat-v12-rare"
SEEDS = 4


def commands() -> list[tuple[str, list[str]]]:
    """One command per seed. Seed 0 keeps the bare name, per `score_v9.py`'s
    `seed_runs` convention (`<arm>`, `<arm>-s1`, ...), which is what lets a
    scorer pool them."""
    out = []
    for seed in range(SEEDS):
        run = ARM if seed == 0 else f"{ARM}-s{seed}"
        out.append((run, [sys.executable, "scripts/chat/train.py",
                          "--run-name", run, "--out-dir", "experiments/chat",
                          *_MIX, *_BASE, "--seed", str(seed)]))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true",
                    help="rerun seeds that already have a marker")
    args = ap.parse_args()

    MARKERS.mkdir(parents=True, exist_ok=True)
    plan = commands()
    if args.dry_run:
        for run, cmd in plan:
            print(f"[{run}] $ {' '.join(cmd)}")
        return 0

    # The corpus must be visible BEFORE the first seed starts. `MixtureSampler`
    # renormalises over sources it can find and prints one warning line, so a
    # missing source does not fail -- it trains a different mixture and says so
    # in a log nobody is watching.
    sys.path.insert(0, str(ROOT / "src"))
    from snnchat.data import ChatCorpus

    available = ChatCorpus(str(ROOT / "data" / "chat")).available
    if "stories_topic_rare" not in available:
        raise SystemExit("stories_topic_rare is not packed or not in the "
                         f"manifest; available: {sorted(available)}")
    print(f"corpus ok: stories_topic_rare present among {len(available)} sources",
          flush=True)

    for run, cmd in plan:
        marker = MARKERS / f"{run}.done"
        if marker.exists() and not args.force:
            print(f"[{run}] SKIP (done {marker.read_text(encoding='utf-8')[:60]})",
                  flush=True)
            continue
        print(f"\n[{run}] $ {' '.join(cmd)}", flush=True)
        t0 = time.perf_counter()
        rc = subprocess.run(cmd, cwd=ROOT).returncode
        dt = time.perf_counter() - t0
        if rc != 0:
            print(f"[{run}] FAILED rc={rc} after {dt / 60:.1f} min", flush=True)
            return 1
        marker.write_text(json.dumps({"run": run, "cmd": cmd,
                                      "seconds": round(dt, 1)}), encoding="utf-8")
        print(f"[{run}] done in {dt / 60:.1f} min", flush=True)

    print("\nall seeds done; score with scripts/chat/score_v12.py", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
