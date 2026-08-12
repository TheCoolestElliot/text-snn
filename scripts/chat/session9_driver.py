"""Run the v9 corpus round: repack, train, score, in the order PREDICTION_v9 fixes.

    python scripts/chat/session9_driver.py --list          # what would run
    python scripts/chat/session9_driver.py --only A12      # one arm
    python scripts/chat/session9_driver.py                 # everything not done

RESTARTABLE BY DESIGN, because it has to be. The whole queue is ~30 GPU-hours on
this box, which is far longer than any harness session and longer than most
uninterrupted uptimes. Every stage writes a marker under
`experiments/chat/_session9/` when it completes, and a rerun of the identical
command skips finished stages and resumes at the first unfinished one. Training
itself is separately restartable (`snnchat.train` resumes from `ckpt_last.pt`
and its sampler is a pure function of `(seed, step)`), so a stage interrupted
mid-run also resumes rather than restarting.

STAGES RUN STRICTLY SEQUENTIALLY AND NOTHING ELSE MAY TOUCH THE GPU. A contended
run looks exactly like a slow one, and `--budget-minutes` is wall clock.

THE ORDER IS NOT ARBITRARY. Repacks come first because several arms share
sources; A12 -- the arm this round exists for -- runs before the cheaper ones so
that a session which dies early has still answered the question worth asking;
A15 is last and is provisionally cancelled by `PREDICTION_v9.md` §3 rule 1.

Not part of the research protocol. No number here is a reported figure.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MARKERS = ROOT / "experiments/chat/_session9"

#: (id, human description, kind, argv). `kind` is only used for reporting and
#: for `--skip-gpu`, which lets the CPU repacks run while a GPU job is busy.
STAGES: list[tuple[str, str, str, list[str]]] = [
    # ---- repacks (CPU) --------------------------------------------------
    ("R9", "pack stories_topic_r05 (raw fraction 0.05)", "cpu",
     [sys.executable, "scripts/chat/build_topic_stories.py",
      "--name", "stories_topic_r05", "--raw-fraction", "0.05"]),
    ("R12", "pack soda_narrative (~218 M chars of adult prose)", "cpu",
     [sys.executable, "scripts/chat/build_data.py", "--only", "soda_narrative"]),
    ("R11", "repack alpaca/dolly/oasst1 at MAX_TURN_CHARS 600", "cpu",
     [sys.executable, "scripts/chat/build_data.py", "--only", "alpaca", "dolly",
      "oasst1", "--max-turn-chars", "600", "--suffix", "_t600"]),

    # ---- arms (GPU), A12 first: see the module docstring -----------------
    ("A12", "soda_narrative at 0.10, taken from soda", "gpu",
     ["--run-name", "chat-v9-narrative", "--seeds", "4",
      "--mix", "stories_topic=0.40", "soda=0.10", "soda_narrative=0.10",
      "alpaca=0.15", "tinystories=0.09", "persona=0.08", "dolly=0.06",
      "oasst1=0.02"]),
    ("A9", "stories_topic_r05 at 0.40", "gpu",
     ["--run-name", "chat-v9-raw05", "--seeds", "4",
      "--mix", "stories_topic_r05=0.40", "soda=0.20", "alpaca=0.15",
      "tinystories=0.09", "persona=0.08", "dolly=0.06", "oasst1=0.02"]),
    ("A11", "instruction sources at MAX_TURN_CHARS 600", "gpu",
     ["--run-name", "chat-v9-turn600", "--seeds", "4",
      "--mix", "stories_topic=0.40", "soda=0.20", "alpaca_t600=0.15",
      "tinystories=0.09", "persona=0.08", "dolly_t600=0.06", "oasst1_t600=0.02"]),
    ("A13a", "bot_loss_weight 3.0 -> 1.0, align_frac held", "gpu",
     ["--run-name", "chat-v9-bot1", "--seeds", "4", "--bot-loss-weight", "1.0"]),
    ("A13b", "align_frac 0.75 -> 0.0, bot_loss_weight held", "gpu",
     ["--run-name", "chat-v9-align0", "--seeds", "4", "--align-frac", "0.0"]),
    ("A14", "seq_len 256 -> 1024 at matched B*L", "gpu",
     ["--run-name", "chat-v9-len1024", "--seeds", "1",
      "--seq-len", "1024", "--batch-size", "40"]),
]

#: `PREDICTION_v9.md` §3 rule 1: A15 does not run. Its bpc conjunct already
#: fails on the existing seed (1.2399 against the incumbent's 1.1743), and the
#: rule says that cancels it unless a reader argues otherwise IN WRITING BEFORE
#: the run. Listed here so the cancellation is visible rather than an omission.
CANCELLED = {"A15": "PREDICTION_v9.md §3 rule 1 -- weighted bpc conjunct already fails"}


#: The shipped recipe every arm anneals from, and the budget it was trained at.
#: `chat-v3d-aligned` is 14,000 steps continuing `chat-v2-anneal`, so an arm that
#: skipped `--init-from` would be comparing a corpus change against a from-scratch
#: run -- which is `chat-v6-scratch`, a different question already answered.
#: EVERY VALUE HERE IS READ OFF `chat-v3d-aligned/config.json`, NOT INHERITED
#: FROM `train.py`'s DEFAULTS. A smoke run at `--max-steps 20` before the first
#: arm launched showed the two disagree on five keys -- `batch_size` 64 vs 160,
#: `seq_len` 384 vs 256, `lr` 2e-3 vs 5e-4, `warmup_steps` 500 vs 200,
#: `eval_batches` 40 vs 30. An arm trained on those defaults would have differed
#: from the incumbent in five ways while claiming to differ in one, and 2.8
#: GPU-hours would have bought an uninterpretable number.
_BASE = ["--init-from", "experiments/chat/chat-v2-anneal/ckpt_best.pt",
         "--max-steps", "14000", "--bot-loss-weight", "3.0",
         "--align-frac", "0.75", "--align-lookahead", "1024",
         "--batch-size", "160", "--seq-len", "256",
         "--lr", "0.0005", "--warmup-steps", "200", "--eval-batches", "30"]


def _train_commands(argv: list[str]) -> list[list[str]]:
    """Expand one arm spec into one `train.py` invocation per seed.

    `scripts/chat/train.py` takes `--seed` (singular). Seeds get their own run
    names -- `<name>` for seed 0 then `<name>-s1`, `-s2`, ... -- which is the
    convention `chat-v3d-aligned-s1..s3` and `chat-v6-inst-a-s1..s3` already use,
    so `scripts/chat/session7_stats.py` can pool them without a new rule.

    An explicit flag in `argv` overrides the same flag in `_BASE`, so an arm that
    varies `--bot-loss-weight` says so once and inherits the rest.
    """
    spec = list(argv)
    n_seeds = 1
    if "--seeds" in spec:
        i = spec.index("--seeds")
        n_seeds = int(spec[i + 1])
        del spec[i:i + 2]
    name = spec[spec.index("--run-name") + 1]

    overridden = {spec[i] for i in range(len(spec)) if spec[i].startswith("--")}
    base = []
    skip = False
    for tok in _BASE:
        if skip:
            skip = False
            continue
        if tok.startswith("--") and tok in overridden:
            skip = True
            continue
        base.append(tok)

    out = []
    for seed in range(n_seeds):
        run = name if seed == 0 else f"{name}-s{seed}"
        argv_seed = [t if t != name else run for t in spec]
        out.append([sys.executable, "scripts/chat/train.py", *argv_seed,
                    *base, "--seed", str(seed)])
    return out


def marker(stage_id: str) -> Path:
    return MARKERS / f"{stage_id}.done"


def run(stage_id: str, desc: str, kind: str, argv: list[str], args) -> bool:
    m = marker(stage_id)
    if m.exists() and not args.force:
        print(f"[{stage_id}] SKIP (done {json.loads(m.read_text())['finished']})")
        return True
    if kind == "gpu" and args.skip_gpu:
        print(f"[{stage_id}] SKIP (--skip-gpu)")
        return True

    # One command for a repack; one command PER SEED for an arm, because
    # `scripts/chat/train.py` takes `--seed` and not `--seeds`.
    cmds = [argv] if kind == "cpu" else _train_commands(argv)

    print(f"\n[{stage_id}] {desc}", flush=True)
    for cmd in cmds:
        print(f"    $ {' '.join(cmd)}", flush=True)
    if args.dry_run:
        return True
    t0 = time.perf_counter()
    for cmd in cmds:
        proc = subprocess.run(cmd, cwd=ROOT)
        if proc.returncode != 0:
            dt = time.perf_counter() - t0
            print(f"[{stage_id}] FAILED rc={proc.returncode} after "
                  f"{dt / 60:.1f} min", flush=True)
            return False
    dt = time.perf_counter() - t0
    MARKERS.mkdir(parents=True, exist_ok=True)
    m.write_text(json.dumps({
        "stage": stage_id, "desc": desc, "cmds": cmds,
        "seconds": round(dt, 1),
        "finished": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }, indent=2), encoding="utf-8")
    print(f"[{stage_id}] done in {dt / 60:.1f} min", flush=True)
    return True


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--only", nargs="*", default=None, help="stage ids to run")
    p.add_argument("--skip-gpu", action="store_true",
                   help="run the CPU repacks only; safe alongside another GPU job")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--list", action="store_true")
    p.add_argument("--force", action="store_true", help="rerun completed stages")
    args = p.parse_args(argv)

    if args.list:
        print(f"{'id':<6} {'kind':<4} {'state':<8} description")
        for sid, desc, kind, _ in STAGES:
            state = "done" if marker(sid).exists() else "pending"
            print(f"{sid:<6} {kind:<4} {state:<8} {desc}")
        for sid, why in CANCELLED.items():
            print(f"{sid:<6} {'gpu':<4} {'CANCELLED':<8} {why}")
        return 0

    todo = [s for s in STAGES if not args.only or s[0] in args.only]
    if not todo:
        print(f"no stage matches {args.only}; ids are {[s[0] for s in STAGES]}")
        return 2

    print(f"session9 driver: {len(todo)} stage(s)")
    for sid, desc, kind, cmd in todo:
        if not run(sid, desc, kind, cmd, args):
            print(f"\nstopping at {sid}. Fix it and rerun -- completed stages are "
                  f"skipped via {MARKERS}", flush=True)
            return 1
    print("\nall requested stages complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
