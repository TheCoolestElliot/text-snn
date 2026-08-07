"""How many bits the story owes to the request that asked for it.

    python scripts/chat/topic_dependence.py experiments/chat/*/ckpt_best.pt

`scripts/chat/quality.py` measures prompt dependence on the dialogue and
instruction splits, which is where the model was ALREADY being trained and is
not where the 2026-08-06 round intervened. This measures the same quantity on
`stories_topic`'s held-out tail: a real story scored behind its own request, and
behind a different story's request. The difference is how much of the story the
subject line is worth.

It is the metric with the least in it -- no sampler, no seed, no lexicon, no
probe list -- and it is the one that answers "did the model learn to condition on
the subject" without any of the machinery that could flatter the answer.

**One asymmetry has to be stated.** `chat-v2-anneal` and `chat-v3c-control` never
trained on this source, so their absolute bits per character on it are worse for
a reason that has nothing to do with conditioning. The *delta* is a within-model
quantity and is the column to read; the absolute columns are printed so that the
asymmetry is visible rather than hidden.

Not part of the research protocol. No number here is a reported figure.
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO / "src") not in sys.path:
    sys.path.insert(0, str(_REPO / "src"))

import torch  # noqa: E402

from snnchat.data import ChatCorpus  # noqa: E402
from snnchat.generate import load_chat_checkpoint  # noqa: E402
from snnchat.quality import prompt_dependence  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("checkpoints", nargs="+")
    p.add_argument("--source", default="stories_topic")
    p.add_argument("--pairs", type=int, default=256)
    p.add_argument("--device", default=None)
    p.add_argument("--data-dir", default="data/chat")
    p.add_argument("--out", default=None)
    args = p.parse_args(argv)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    paths: list[str] = []
    for pattern in args.checkpoints:
        paths.extend(sorted(glob.glob(pattern)) or [pattern])

    print(f"### {args.source}: how much a story owes to its own request\n")
    print("| arm | step | bpc, own request | bpc, another request | delta |")
    print("| --- | ---: | ---: | ---: | ---: |")
    results = {}
    for path in paths:
        model, cfg, ck = load_chat_checkpoint(path, device=device)
        if ck.get("kind") != "snnchat":
            continue
        corpus = ChatCorpus(args.data_dir, vocab_version=cfg.vocab_version)
        # A story is a whole turn and runs to several hundred characters, well
        # past the range the dialogue splits need, so the window is widened here
        # rather than in the default -- narrowing it would silently sample only
        # the shortest stories, which are the least topic-bearing ones.
        dep = prompt_dependence(model, corpus, sources=(args.source,),
                                per_source=args.pairs, device=device,
                                bot_range=(80, 900), batch_size=16)
        row = dep["per_source"].get(args.source)
        if not row:
            continue
        label = Path(path).parent.name
        results[label] = {**row, "step": ck.get("step")}
        print(f"| `{label}` | {ck.get('step'):,} | {row['bpc_matched']:.4f} | "
              f"{row['bpc_shuffled']:.4f} | **{row['delta']:+.4f}** |")
        del model
        if device == "cuda":
            torch.cuda.empty_cache()

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(results, fh, indent=1)
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
