"""Can the model carry a fact from one turn to the next? Measured, with a control.

WHY THIS DID NOT EXIST
----------------------
"Told your name, it cannot repeat it a turn later" is stated in
`docs/chat/README.md` §2, in `RESULTS.md`, and in five rounds of QUALITY
documents. **No committed artifact contains that measurement.** It has always
been an anecdote, and `snnchat.quality`'s battery has no multi-turn probe at all
-- its `fact` kind asks world-knowledge questions in a single turn, which is a
different capability.

That gap matters beyond tidiness. Every proposal to change the training window
(`seq_len` 256 -> 1024) or to widen the model is argued partly on reach, and
none of them can be evaluated without a readout of the thing they are supposed
to improve. This builds the readout FIRST, against the current checkpoint, so a
later arm has something to be compared with.

THE CONTROL IS THE WHOLE DESIGN
-------------------------------
Recall cannot be read off the answer rate alone. Asked "what is my name?" the
model may say "Elliot" because it was told, or because "Elliot" is simply a
likely name -- and at 4.4 M parameters on a TinyStories-heavy corpus the second
is not a silly hypothesis. Each item is therefore run twice at the same sampler
seed:

* **told**   -- establishing turn, then (optionally) filler turns, then the question
* **untold** -- the identical question with the establishing turn REMOVED

The difference is the estimate; the `untold` rate is the floor. Both runs share
a seed, so the comparison is paired and McNemar applies.

Distance is swept as well as presence: 0, 1 and 2 intervening turns. A model
whose state decays should show recall falling with distance, and a model that
never had it should show a flat line at the control rate. Those two are not
distinguishable from a single distance, which is why the sweep is cheap
insurance rather than decoration.

    python scripts/chat/memory_probe.py --seeds 8

Not part of the research protocol. No number here is a reported figure.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import torch  # noqa: E402

from snnchat.generate import ChatSession, SamplingParams, load_chat_checkpoint  # noqa: E402
from snnchat.rerank import RerankParams  # noqa: E402
from snnchat.tokenizer import ChatTokenizer  # noqa: E402

#: (establishing turn, question, what the answer must contain).
#: The targets are deliberately ordinary words the corpus contains -- a target
#: the model could never spell would measure the tokenizer, not the memory.
ITEMS: list[tuple[str, str, tuple[str, ...]]] = [
    ("my name is elliot", "what is my name?", ("elliot",)),
    ("my name is sarah", "what is my name?", ("sarah",)),
    ("i have a dog called rex", "what is my dog called?", ("rex",)),
    ("i have a cat called mittens", "what is my cat called?", ("mittens",)),
    ("my favourite colour is blue", "what is my favourite colour?", ("blue",)),
    ("my favourite colour is green", "what is my favourite colour?", ("green",)),
    ("i live in a town called ashby", "where do i live?", ("ashby",)),
    ("i am seven years old", "how old am i?", ("seven", "7")),
    ("my favourite food is pizza", "what is my favourite food?", ("pizza",)),
    ("i have a red bicycle", "what colour is my bicycle?", ("red",)),
]

#: Turns that carry no information about the item, used to put distance between
#: the fact and the question. Neutral small talk rather than silence, because a
#: real conversation is what the distance is meant to model.
FILLERS = ["how are you?", "that is interesting", "tell me something else"]


def hit(text: str, targets) -> float:
    low = text.lower()
    return 1.0 if any(re.search(r"\b" + re.escape(t) + r"\b", low) for t in targets) else 0.0


def wilson(k: int, n: int, z: float = 1.95996) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p, d = k / n, 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def mcnemar(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    return min(1.0, 2.0 * sum(math.comb(n, k) for k in range(0, min(b, c) + 1)) / (2 ** n))


@torch.no_grad()
def ask(model, tok, device, params, rp, establish, fillers, question):
    """Run one conversation and return the reply to `question`."""
    s = ChatSession(model, tok, device=device, params=params, rerank=rp)
    if establish is not None:
        s.send(establish)
    for f in fillers:
        s.send(f)
    return s.send(question)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", default="experiments/chat/chat-v3d-aligned/ckpt_best.pt")
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--lam", type=float, default=0.6)
    ap.add_argument("--max-new", type=int, default=120)
    ap.add_argument("--distances", type=int, nargs="*", default=[0, 1, 2])
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default="experiments/chat/_quality/memory_probe.json")
    args = ap.parse_args()

    model, _cfg, _ck = load_chat_checkpoint(ROOT / args.ckpt, device=args.device)
    model.eval()
    tok = ChatTokenizer()
    rp = RerankParams(n=args.n, lam=args.lam)

    rows = []
    for dist in args.distances:
        fillers = FILLERS[:dist]
        for establish, question, targets in ITEMS:
            for seed in range(args.seeds):
                params = SamplingParams(seed=seed, max_new=args.max_new)
                told = ask(model, tok, args.device, params, rp, establish, fillers, question)
                untold = ask(model, tok, args.device, params, rp, None, fillers, question)
                rows.append({
                    "distance": dist, "establish": establish, "question": question,
                    "targets": list(targets), "seed": seed,
                    "told_hit": hit(told, targets), "untold_hit": hit(untold, targets),
                    "told": told[:200], "untold": untold[:200],
                })

    print(f"\ncross-turn recall, {len(ITEMS)} items x {args.seeds} seeds, "
          f"n={args.n} lambda={args.lam}, ckpt {Path(args.ckpt).parent.name}")
    print(f"{'distance':>9} {'told':>16} {'untold (control)':>18} {'diff':>8} "
          f"{'b/c':>8} {'McNemar p':>10}")
    summary = {}
    for dist in args.distances:
        sub = [r for r in rows if r["distance"] == dist]
        n = len(sub)
        kt = int(sum(r["told_hit"] for r in sub))
        ku = int(sum(r["untold_hit"] for r in sub))
        b = sum(1 for r in sub if r["untold_hit"] > r["told_hit"])
        c = sum(1 for r in sub if r["told_hit"] > r["untold_hit"])
        lo_t, hi_t = wilson(kt, n)
        p = mcnemar(b, c)
        summary[dist] = {"n": n, "told": [kt, n], "untold": [ku, n],
                         "told_ci": [lo_t, hi_t], "discordant": [b, c], "mcnemar_p": p}
        print(f"{dist:>9} {kt:>4}/{n:<4} {kt/n:>6.4f} {ku:>6}/{n:<4} {ku/n:>6.4f} "
              f"{(kt-ku)/n:>+8.4f} {c:>3}/{b:<4} {p:>10.4f}")

    total_told = sum(r["told_hit"] for r in rows)
    print(f"\ntold-condition recall over all distances: "
          f"{int(total_told)}/{len(rows)} = {total_told/len(rows):.4f}")
    if total_told == 0:
        print("  ZERO. The claim in docs/chat/README.md now has an artifact behind it.")

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "ckpt": args.ckpt, "seeds": args.seeds, "n": args.n, "lam": args.lam,
        "items": len(ITEMS), "summary": {str(k): v for k, v in summary.items()},
        "rows": rows,
    }, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
