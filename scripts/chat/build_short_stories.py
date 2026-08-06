"""Pack topic-conditioned stories short enough to fit a training window whole.

    python scripts/chat/build_short_stories.py --limit-chars 320000000

Adds ONE new file pair to `data/chat/` (`stories_short.bin`, `.val.bin`) and
registers it in the manifest. Nothing already packed is read, rewritten or
deleted, so `tinystories.bin` and `stories_topic.bin` are exactly as they were
and every checkpoint trained before this still has its corpus. An arm that wants
the previous behaviour leaves this source out of its `--mix`.

Same shape of decision as `build_topic_stories.py`, and for the same reason: the
balance between long stories and short ones is then a mixture weight a later run
can change, rather than a fork in the corpus.

See `snnchat.shortform` for what "short" means here, what it costs, and the
three judgements involved.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO / "src") not in sys.path:
    sys.path.insert(0, str(_REPO / "src"))

from snnchat.build_corpus import _pack_source, _RawAwareTokenizer  # noqa: E402
from snnchat.shortform import SHORT, SHORT300, Truncation, short_conversation  # noqa: E402
from snnchat.sources import read_source  # noqa: E402

#: The named targets. `--trunc short` reproduces `stories_short` exactly;
#: `--trunc short300` is `docs/chat/QUALITY_v4.md` §7 item 2's arm. A name rather
#: than three loose integers, so a packed source's manifest entry records which
#: pre-registered target it was built at instead of numbers a reader must match
#: back to a document by hand.
_TARGETS = {"short": SHORT, "short300": SHORT300}

#: Fraction of stories emitted whole, as bare narrative rather than as an answer.
#: Lower than `build_topic_stories.py`'s 0.15 and much lower than
#: `build_corpus`'s 0.5, because the raw-narrative signal is already bought twice
#: over by the two long story sources still in the mixture. What is scarce is the
#: conditioned short reply, and that is what this file is for.
#:
#: It is not zero. Emitting the full story sometimes is what keeps this source
#: from teaching that stories are *only* ever four sentences long, and it is the
#: only place in this file where the text past the truncation is used at all.
RAW_FRACTION = 0.08


def _conversations(path: str, rng: random.Random, trunc: Truncation):
    for conv in read_source("tinystories", path):
        story = conv[0][1]
        if rng.random() < RAW_FRACTION:
            yield [("_raw", story)]
            continue
        short = short_conversation(story, rng, trunc)
        if short is not None:
            yield short


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--raw-dir", default="data/chat_raw")
    p.add_argument("--out-dir", default="data/chat")
    p.add_argument("--name", default="stories_short")
    p.add_argument("--trunc", choices=sorted(_TARGETS), default="short",
                   help="named truncation target from snnchat.shortform; "
                        "'short' reproduces stories_short exactly")
    p.add_argument("--limit-chars", type=int, default=320_000_000)
    p.add_argument("--seed", type=int, default=11)
    p.add_argument("--force", action="store_true")
    args = p.parse_args(argv)
    trunc = _TARGETS[args.trunc]

    raw = os.path.join(args.raw_dir, "tinystories.txt")
    if not os.path.exists(raw):
        raise SystemExit(f"{raw} not found; run scripts/chat/build_data.py first")
    out = os.path.join(args.out_dir, f"{args.name}.bin")
    if os.path.exists(out) and not args.force:
        raise SystemExit(f"{out} exists; pass --force to repack")

    tok = _RawAwareTokenizer()
    rng = random.Random(args.seed)
    print(f"packing {args.name} from {raw} "
          f"(limit {args.limit_chars / 1e6:,.0f} M chars, raw fraction {RAW_FRACTION}, "
          f"trunc {args.trunc} = [{trunc.lo}, {trunc.hi}] cap {trunc.cap})",
          flush=True)
    stats = _pack_source(args.name, _conversations(raw, rng, trunc), args.out_dir, tok,
                         limit_chars=args.limit_chars)

    manifest_path = os.path.join(args.out_dir, "manifest.json")
    with open(manifest_path, encoding="utf-8") as fh:
        manifest = json.load(fh)
    manifest["sources"][args.name] = {
        **stats,
        "built_from": "tinystories.txt",
        "raw_fraction": RAW_FRACTION,
        "shaper": "snnchat.shortform.short_conversation",
        "truncation": {"name": args.trunc, "lo": trunc.lo, "hi": trunc.hi,
                       "cap": trunc.cap},
        "seed": args.seed,
    }
    tmp = manifest_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, sort_keys=True)
        fh.write("\n")
    os.replace(tmp, manifest_path)
    print(f"registered {args.name} in {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
