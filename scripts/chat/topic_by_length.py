"""Is the topic hit rate a property of the model, or of how long its replies are?

    python scripts/chat/topic_by_length.py experiments/chat/_quality/chat-v*.json

`topic` is scored as "does the reply contain the word that was asked for". That
is a fine metric between two models that write replies of the same length and a
treacherous one otherwise, because a longer reply has more chances to contain any
given word. The 2026-08-06 short-conversation round changed reply length on
purpose -- roughly 300 characters to roughly 165 -- so its topic column cannot be
read against the previous round's without asking this question first.

Three views, from the same saved draws, no GPU:

1.  **raw** -- the hit rate as reported. This is what a user experiences and it
    is the headline whatever the other two say.
2.  **per 100 characters** -- raw divided by mean candidate length. Crude: hit
    probability is not linear in length, it saturates, so this flatters short
    replies. It is here as an upper bound on how much of a difference length
    could explain.
3.  **within a length band** -- the honest version, and the one that usually
    refuses to answer. Two arms whose length distributions barely overlap have
    almost no candidates in a common band, and a comparison made on the overlap
    is a comparison between one arm's typical replies and the other arm's tail.
    The counts are printed next to every rate so that a reader can see when this
    has happened rather than being told a number.

Not part of the research protocol.
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

_BANDS = ((0, 100), (100, 200), (200, 300), (300, 10_000))


def _topic_candidates(blob: dict):
    """Yield `(text, hit)` for every candidate drawn against a `topic` probe."""
    probes = {p["prompt"]: p for p in blob["draws"]["probes"]}
    for draw in blob["draws"]["draws"]:
        probe = probes.get(draw["prompt"])
        if not probe or probe["kind"] != "topic":
            continue
        expect = [w.lower() for w in probe.get("expect", [])]
        for text in draw["texts"]:
            yield text, any(w in text.lower() for w in expect)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("results", nargs="+")
    args = p.parse_args(argv)

    paths: list[str] = []
    for pattern in args.results:
        paths.extend(sorted(glob.glob(pattern)) or [pattern])

    print("### raw, and what reply length could explain\n")
    print("| arm | candidates | mean chars | topic (raw) | per 100 chars |")
    print("| --- | ---: | ---: | ---: | ---: |")
    banded = []
    for path in paths:
        with open(path, encoding="utf-8") as fh:
            blob = json.load(fh)
        label = blob["report"]["label"]
        cands = list(_topic_candidates(blob))
        if not cands:
            continue
        n = len(cands)
        hits = sum(h for _, h in cands)
        chars = sum(len(t) for t, _ in cands) / n
        rate = hits / n
        print(f"| `{label}` | {n:,} | {chars:.0f} | **{rate:.4f}** | "
              f"{100 * rate / max(chars, 1):.4f} |")
        counts = {b: [0, 0] for b in _BANDS}
        for text, hit in cands:
            for b in _BANDS:
                if b[0] <= len(text) < b[1]:
                    counts[b][1] += 1
                    counts[b][0] += hit
        banded.append((label, counts))

    print("\n### within a length band -- rate (candidates in the band)\n")
    print("A band holding a handful of an arm's candidates is that arm's tail, not")
    print("its behaviour. Read the counts before the rates.\n")
    print("| arm | " + " | ".join(f"{lo}-{hi if hi < 10_000 else ''}"
                                  for lo, hi in _BANDS) + " |")
    print("| --- | " + " | ".join("---:" for _ in _BANDS) + " |")
    for label, counts in banded:
        cells = []
        for b in _BANDS:
            hit, n = counts[b]
            cells.append(f"{hit / n:.3f} ({n})" if n else "none (0)")
        print(f"| `{label}` | " + " | ".join(cells) + " |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
