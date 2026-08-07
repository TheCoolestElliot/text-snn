"""Pack a story corpus whose requests name the story's SUBJECT, not its cast.

    python scripts/chat/build_topic_stories.py --limit-chars 400000000

Adds ONE new file pair to `data/chat/` (`stories_topic.bin`, `.val.bin`) and
registers it in the manifest. Nothing already packed is read, rewritten or
deleted -- `tinystories.bin` is left exactly as it was, so every checkpoint
trained before this still has the corpus it was trained on, and an arm that
wants the old behaviour just leaves the new source out of its `--mix`.

WHY A NEW SOURCE RATHER THAN A REPACK
-------------------------------------
Two reasons, one practical and one about evidence. Practically, `tinystories.bin`
is 747 MB and repacking it in place would make every earlier run
unreproducible. About evidence: keeping both files means the mixture weight
between them is a knob rather than a fork, so "how much of this does the model
need" is a number a later run can change, and the comparison in
`docs/chat/QUALITY.md` is between two mixes of the same packing code rather than
between two corpora.

See `snnchat.topics` for how a story's subject is chosen and for the honest
caveat about measuring this with a probe that asks for exactly this form.
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
from snnchat.sources import read_source  # noqa: E402
from snnchat.topics import story_request  # noqa: E402

#: Fraction of stories emitted as bare narrative rather than as an answer to a
#: request. Lower than `build_corpus`'s 0.5 on purpose: `tinystories.bin` is
#: still in the mix and is still 50 % raw, so the raw-narrative signal is
#: already paid for and this file's job is the conditioning it does not carry.
RAW_FRACTION = 0.15


def _conversations(path: str, rng: random.Random):
    for conv in read_source("tinystories", path):
        story = conv[0][1]
        if rng.random() < RAW_FRACTION:
            yield [("_raw", story)]
            continue
        request = story_request(story, rng)
        yield [("user", request), ("bot", story)]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--raw-dir", default="data/chat_raw")
    p.add_argument("--out-dir", default="data/chat")
    p.add_argument("--name", default="stories_topic")
    p.add_argument("--limit-chars", type=int, default=400_000_000)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--force", action="store_true")
    args = p.parse_args(argv)

    raw = os.path.join(args.raw_dir, "tinystories.txt")
    if not os.path.exists(raw):
        raise SystemExit(f"{raw} not found; run scripts/chat/build_data.py first")
    out = os.path.join(args.out_dir, f"{args.name}.bin")
    if os.path.exists(out) and not args.force:
        raise SystemExit(f"{out} exists; pass --force to repack")

    tok = _RawAwareTokenizer()
    rng = random.Random(args.seed)
    print(f"packing {args.name} from {raw} "
          f"(limit {args.limit_chars / 1e6:,.0f} M chars, raw fraction {RAW_FRACTION})",
          flush=True)
    stats = _pack_source(args.name, _conversations(raw, rng), args.out_dir, tok,
                         limit_chars=args.limit_chars)

    manifest_path = os.path.join(args.out_dir, "manifest.json")
    with open(manifest_path, encoding="utf-8") as fh:
        manifest = json.load(fh)
    manifest["sources"][args.name] = {
        **stats,
        "built_from": "tinystories.txt",
        "raw_fraction": RAW_FRACTION,
        "topic_prompts": "snnchat.topics.story_request",
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
