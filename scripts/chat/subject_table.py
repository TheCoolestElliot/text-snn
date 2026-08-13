"""How often is each word AVAILABLE as a story subject, and how often requested?

WHY THIS EXISTS
---------------
`QUALITY_v11.md` §1 measured the thing that predicts whether this model can
draft a reply about a noun: not how often it has read the word (`seen`, partial
rho -0.027) but how often it was *asked* for it (`asked`, partial rho +0.328).
And the request slot is brutally concentrated -- 35 subjects carry a third of
all 298,709 requests, because `snnchat.topics.story_request` always takes
`topics[0]`, the most frequent content word of the story, and TinyStories is
about birds, balls and cats.

Flattening that slot needs a frequency to flatten *against*, and no such table
exists. `src/snnchat/word_freq.json` is not it: that is an all-text word count
over a 5 % prefix, built for the decoder's IDF tier, and it counts a word in a
story body the same as a word in a request.

This builds the right table, in one pass over the raw corpus, using
`topic_of` itself so the counts describe exactly the candidates the packer will
choose among:

* `primary` -- how often a word is `topics[0]`, i.e. what the SHIPPED rule
  requests. This is the distribution being flattened, and it is what
  `QUALITY_v11.md` §1's histogram reports.
* `counts` -- how often a word is eligible at all (`topics[:3]`). This is what
  the new policy weights against, because a word that is available in a
  thousand stories and requested in none is exactly the case the flattening is
  for, and `primary` would score it 0 and make it maximally attractive on the
  strength of never having been picked.

Not part of the research protocol; no number here is a reported figure.

    python scripts/chat/subject_table.py
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from snnchat.sources import read_source  # noqa: E402
from snnchat.topics import topic_of  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--raw-dir", default="data/chat_raw")
    p.add_argument("--limit-stories", type=int, default=0,
                   help="0 = the whole file. The packer stops at a CHARACTER "
                        "budget, so this table deliberately covers at least as "
                        "many stories as any pack will consume.")
    p.add_argument("--min-count", type=int, default=2,
                   help="drop subjects rarer than this; they are noise and they "
                        "would otherwise dominate an inverse-frequency weight")
    p.add_argument("--out", default="src/snnchat/subject_freq.json")
    args = p.parse_args(argv)

    raw = ROOT / args.raw_dir / "tinystories.txt"
    if not raw.exists():
        raise SystemExit(f"{raw}: not downloaded")

    counts: dict[str, int] = {}
    primary: dict[str, int] = {}
    n_stories = n_with = 0
    t0 = time.perf_counter()
    for conv in read_source("tinystories", str(raw)):
        n_stories += 1
        topics = topic_of(conv[0][1])
        if topics:
            n_with += 1
            primary[topics[0]] = primary.get(topics[0], 0) + 1
            for w in topics:
                counts[w] = counts.get(w, 0) + 1
        if args.limit_stories and n_stories >= args.limit_stories:
            break
        if n_stories % 100_000 == 0:
            print(f"  {n_stories:,} stories, {len(counts):,} subjects, "
                  f"{time.perf_counter() - t0:.0f}s", flush=True)

    counts = {w: c for w, c in counts.items() if c >= args.min_count}
    primary = {w: c for w, c in primary.items() if w in counts}

    out = ROOT / args.out
    out.write_text(json.dumps({
        "source": f"{args.raw_dir}/tinystories.txt",
        "stories_scanned": n_stories,
        "stories_with_a_subject": n_with,
        "min_count": args.min_count,
        "selector": "snnchat.topics.topic_of (max_rank=3)",
        "counts": counts,
        "primary": primary,
    }, sort_keys=True), encoding="utf-8")

    top = sorted(primary.items(), key=lambda kv: -kv[1])[:12]
    total_p = sum(primary.values())
    print(f"\n  {n_stories:,} stories, {n_with:,} yielded a subject")
    print(f"  eligible subjects (topics[:3], count >= {args.min_count}): {len(counts):,}")
    print(f"  requested subjects (topics[0]): {len(primary):,} over {total_p:,} stories")
    print("  most-requested: " + ", ".join(f"{w} {c}" for w, c in top))
    for thresh in (1000, 300, 100, 30, 10):
        k = sum(1 for c in primary.values() if c >= thresh)
        share = sum(c for c in primary.values() if c >= thresh) / max(total_p, 1)
        print(f"    {k:>5} requested >= {thresh:>4} times ({share:.1%} of requests)")
    print(f"\nwrote {out}  ({out.stat().st_size / 2**20:.1f} MiB, "
          f"{time.perf_counter() - t0:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
