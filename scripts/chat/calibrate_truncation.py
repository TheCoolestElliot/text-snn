"""What a truncation target actually produces, measured before it is committed.

    python scripts/chat/calibrate_truncation.py --stories 20000

`snnchat.shortform.short_story` stops at the last sentence boundary *before* a
sampled target, so the kept length it produces is systematically **under** the
target it was given -- 161 characters against a midpoint of 187.5 at `SHORT`.
`docs/chat/QUALITY_v4.md` §7 item 2 asks for "~300 characters", and 300 is
`chat-v3d-aligned`'s mean *reply* length, so the number that has to land on 300
is the mean kept length and not the target midpoint.

This script measures that mapping so the range can be fixed by measurement rather
than by arithmetic that the sentence splitter does not obey. It reads
`data/chat_raw/tinystories.txt` and nothing else: **no model, no checkpoint, no
arm and no held-out split**. It is safe to run before a pre-registration because
there is no outcome in it to peek at.

Both named targets are reported side by side, because the comparison
`PREDICTION_v5.md` rests on is between two sources built by this shaper and the
claim that they differ in length and in nothing else needs the length difference
stated in the same units for both.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO / "src") not in sys.path:
    sys.path.insert(0, str(_REPO / "src"))

from snnchat.shortform import (  # noqa: E402
    SHORT, SHORT300, Truncation, short_conversation, short_story, topic_for,
)
from snnchat.sources import read_source  # noqa: E402
from snnchat.tokenizer import ChatTokenizer  # noqa: E402


def _pct(xs: list[int], q: float) -> int:
    return sorted(xs)[min(len(xs) - 1, int(q * len(xs)))]


def measure(stories: list[str], trunc: Truncation, seed: int) -> dict:
    """Kept length, conversation length and topic-survival rate at `trunc`."""
    tok = ChatTokenizer()
    kept_lens: list[int] = []
    conv_lens: list[int] = []
    with_topic = 0
    rng = random.Random(seed)
    for story in stories:
        kept = short_story(story, rng, trunc)
        if len(kept) < 40:
            continue
        kept_lens.append(len(kept))
        if topic_for(story, kept):
            with_topic += 1
    rng = random.Random(seed)
    for story in stories:
        conv = short_conversation(story, rng, trunc)
        if conv is not None:
            conv_lens.append(len(tok.render_conversation(conv)))
    return {
        "target": {"lo": trunc.lo, "hi": trunc.hi, "cap": trunc.cap},
        "target_midpoint": (trunc.lo + trunc.hi) / 2,
        "n_usable": len(kept_lens),
        "kept_mean": round(statistics.fmean(kept_lens), 1),
        "kept_median": statistics.median(kept_lens),
        "kept_p10": _pct(kept_lens, 0.10),
        "kept_p90": _pct(kept_lens, 0.90),
        "kept_max": max(kept_lens),
        "kept_over_midpoint": round(statistics.fmean(kept_lens) / ((trunc.lo + trunc.hi) / 2), 3),
        "topic_survival": round(with_topic / max(1, len(kept_lens)), 4),
        "conv_mean": round(statistics.fmean(conv_lens), 1),
        "conv_p90": _pct(conv_lens, 0.90),
        "conv_fits_256": round(sum(c <= 256 for c in conv_lens) / len(conv_lens), 4),
        "conv_fits_512": round(sum(c <= 512 for c in conv_lens) / len(conv_lens), 4),
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--raw-dir", default="data/chat_raw")
    p.add_argument("--stories", type=int, default=20_000)
    p.add_argument("--seed", type=int, default=11)
    p.add_argument("--candidates", default=None,
                   help="extra ranges to measure, 'lo,hi;lo,hi;...'. The search "
                        "that chose SHORT300 is the default below, so a reader "
                        "can reproduce the ranking rather than take it on trust")
    p.add_argument("--out", default=None)
    args = p.parse_args(argv)

    targets = [("short", SHORT), ("short300", SHORT300)]
    if args.candidates:
        for spec in args.candidates.split(";"):
            lo, hi = (int(x) for x in spec.split(","))
            targets.append((f"[{lo},{hi}]", Truncation(lo, hi, hi)))

    raw = str(Path(args.raw_dir) / "tinystories.txt")
    stories: list[str] = []
    for conv in read_source("tinystories", raw):
        stories.append(conv[0][1])
        if len(stories) >= args.stories:
            break
    print(f"read {len(stories):,} stories from {raw}", flush=True)

    out = {"stories": len(stories), "seed": args.seed,
           "targets": {name: measure(stories, t, args.seed) for name, t in targets}}

    hdr = f"{'target':>10} {'range':>12} {'mid':>6} {'kept mean':>10} {'median':>7} " \
          f"{'p90':>5} {'kept/mid':>9} {'topic':>7} {'conv mean':>10} {'fits 256':>9}"
    print(hdr)
    print("-" * len(hdr))
    for name, m in out["targets"].items():
        t = m["target"]
        print(f"{name:>10} {f'[{t['lo']}, {t['hi']}]':>12} {m['target_midpoint']:>6.1f} "
              f"{m['kept_mean']:>10.1f} {m['kept_median']:>7} {m['kept_p90']:>5} "
              f"{m['kept_over_midpoint']:>9.3f} {m['topic_survival']:>7.3f} "
              f"{m['conv_mean']:>10.1f} {m['conv_fits_256']:>9.3f}")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(out, fh, indent=2, sort_keys=True)
            fh.write("\n")
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
