"""Does the echo partition help on prompts the probe battery has never seen?

WHY THIS EXISTS AND NOT JUST A REPLAY OF `_quality/*.json`
---------------------------------------------------------
The committed draws answer a weaker question than they look like they answer.
`snnchat.quality`'s `topic` probes ask whether the reply contains the requested
noun, and `snnchat.rerank`'s echo partition prefers replies containing the
requested noun. Measured on those probes the two are very nearly the same
function, so a large gain there is partly definitional rather than evidence.

Two things fix that, and this script does both:

* **Held-out topics.** Every noun below is absent from `quality.PROBES`. The
  battery's sixteen topics are also, per `_quality/topic_frequency.json`, taught
  between 2 and 1,348 times per 200,000 stories -- so no committed number has
  ever tested a topic the corpus did not drill. Some of these are rare and that
  is the point.
* **A paired comparison on ONE pool.** Both selectors choose from the identical
  candidates, so the model, the sampler, the seeds and the draws are held fixed
  and the only thing that varies is the selection rule. That makes McNemar's
  test the right one and removes sampling noise from the comparison entirely.

Not part of the research protocol; no number here is a reported figure.

    python scripts/chat/echo_holdout.py --seeds 6
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

from snnchat.generate import SamplingParams, load_chat_checkpoint  # noqa: E402
from snnchat.rerank import (  # noqa: E402
    RerankParams,
    echo_count,
    echo_weight,
    prompt_content_words,
    rerank,
    trim_to_sentence,
)
from snnchat.tokenizer import BOS, BOT, ChatTokenizer  # noqa: E402

#: (prompt, the word a reply about it should contain). None of these nouns
#: appear in `snnchat.quality.PROBES`.
HELDOUT: list[tuple[str, tuple[str, ...]]] = [
    ("tell me a story about a penguin", ("penguin",)),
    ("tell me a story about a lighthouse", ("lighthouse",)),
    ("tell me a story about a wizard", ("wizard",)),
    ("tell me a story about a turtle", ("turtle",)),
    ("tell me a story about a violin", ("violin",)),
    ("tell me a story about a castle", ("castle",)),
    ("tell me a story about a squirrel", ("squirrel",)),
    ("tell me a story about a balloon", ("balloon",)),
    ("tell me a story about a whale", ("whale",)),
    ("tell me a story about a mountain", ("mountain",)),
    ("tell me a story about a bakery", ("bakery", "baker")),
    ("tell me a story about a mermaid", ("mermaid",)),
    ("tell me a story about a clock", ("clock",)),
    ("tell me a story about a bear", ("bear",)),
    ("tell me a story about a garden", ("garden",)),
    ("tell me a story about a firefighter", ("firefighter", "fireman")),
    ("write me a little story about a butterfly", ("butterfly",)),
    ("can you tell me a story about a snail", ("snail",)),
    ("i want a story about a rainbow", ("rainbow",)),
    ("make up a story about an elephant", ("elephant",)),
]


def hit(text: str, words) -> float:
    """1.0 if the reply uses any of `words`. Plurals count; substrings do not.

    Deliberately the same shape as `snnchat.quality._word_hit` -- a different
    matcher here would make this table incomparable with every other table in
    `docs/chat/`.
    """
    low = text.lower()
    for w in words:
        if re.search(r"\b" + re.escape(w) + r"(s|es)?\b", low):
            return 1.0
    return 0.0


def mcnemar(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value on the discordant pairs (b, c).

    Exact rather than the chi-square approximation because the discordant count
    here is small, which is precisely where the approximation is worst.
    """
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, k) for k in range(0, min(b, c) + 1)) / (2 ** n)
    return min(1.0, 2.0 * tail)


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, centre - half), min(1.0, centre + half))


@torch.no_grad()
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", default="experiments/chat/chat-v3d-aligned/ckpt_best.pt")
    ap.add_argument("--seeds", type=int, default=6)
    ap.add_argument("--n", type=int, default=8, help="candidates per draw")
    ap.add_argument("--lam", type=float, default=0.6)
    ap.add_argument("--max-new", type=int, default=300)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default="experiments/chat/_quality/echo_holdout.json")
    args = ap.parse_args()

    model, _cfg, _ck = load_chat_checkpoint(ROOT / args.ckpt, device=args.device)
    model.eval()
    tok = ChatTokenizer()
    rp = RerankParams(n=args.n, lam=args.lam)

    rows = []
    for prompt, words in HELDOUT:
        echo_words = prompt_content_words(prompt)
        for seed in range(args.seeds):
            params = SamplingParams(seed=seed, max_new=args.max_new)
            prefix = [BOS, *tok.render_turn("user", prompt), BOT]
            ids = torch.tensor([prefix], device=args.device)
            logits, state, _ = model(ids, state=None)

            # ONE pool, both selectors. `rerank` is called twice on the same
            # candidates rather than sampled twice, so the comparison is paired
            # and the draw is not a source of variance.
            _, cands = rerank(model, logits[:, -1, :].float(), state, params, rp,
                              device=args.device, tok=tok, echo_words=echo_words)
            for c in cands:
                text = tok.decode_visible(c.ids)
                c.echo = echo_count(text, echo_words)
                c.echo_weight = echo_weight(text, echo_words)

            # The SAME three partitions `rerank` applies, tiered on the weighted
            # sum. If this copy and `rerank` ever disagree, every number in
            # QUALITY_v8 describes a selector the REPL does not use.
            pool = [c for c in cands if c.n_chars >= rp.min_chars] or cands
            base = max(pool, key=lambda c: c.score)
            best = max(c.echo_weight for c in pool)
            tier = ([c for c in pool if c.echo_weight >= best - 1e-9]
                    if best > 0.0 else pool)
            picked = max(tier, key=lambda c: c.score)

            def text_of(c):
                out = list(c.ids)
                if rp.trim_to_sentence and not c.closed:
                    out = trim_to_sentence(out, tok, min_keep=rp.min_chars)
                return tok.decode_visible(out).strip()

            tb, tp = text_of(base), text_of(picked)
            rows.append({
                "prompt": prompt, "seed": seed,
                "base_hit": hit(tb, words), "echo_hit": hit(tp, words),
                "changed": picked is not base,
                "oracle_hit": max(hit(text_of(c), words) for c in cands),
                "base_chars": len(tb), "echo_chars": len(tp),
                "base_logp": base.logp_cond / max(len(base.scored), 1),
                "echo_logp": picked.logp_cond / max(len(picked.scored), 1),
                "base_text": tb[:300], "echo_text": tp[:300],
                # THE WHOLE POOL, so any later selection rule can be replayed
                # against this generation instead of drawing its own. Without it
                # every question about lambda, about N, or about a different
                # partition costs another two GPU-minutes and compares itself
                # against a different draw.
                "expect": list(words),
                "pool": [{
                    "text": text_of(c),
                    "n_chars": c.n_chars,
                    "logp_cond": c.logp_cond,
                    "logp_null": c.logp_null,
                    "echo": c.echo,
                    "echo_weight": c.echo_weight,
                } for c in cands],
            })

    n = len(rows)
    nb = sum(r["base_hit"] for r in rows)
    ne = sum(r["echo_hit"] for r in rows)
    no = sum(r["oracle_hit"] for r in rows)
    b = sum(1 for r in rows if r["base_hit"] > r["echo_hit"])
    c = sum(1 for r in rows if r["echo_hit"] > r["base_hit"])
    p = mcnemar(b, c)
    lo_b, hi_b = wilson(int(nb), n)
    lo_e, hi_e = wilson(int(ne), n)

    print(f"\nheld-out topics, {len(HELDOUT)} prompts x {args.seeds} seeds = {n} draws, "
          f"n={args.n} lambda={args.lam}")
    print(f"  shipped selector : {nb/n:.4f}  ({int(nb)}/{n})  95% CI [{lo_b:.4f}, {hi_b:.4f}]")
    print(f"  + echo partition : {ne/n:.4f}  ({int(ne)}/{n})  95% CI [{lo_e:.4f}, {hi_e:.4f}]")
    print(f"  oracle over pool : {no/n:.4f}  ({int(no)}/{n})")
    print(f"  discordant pairs : echo-only {c}, shipped-only {b}   McNemar p = {p:.5f}")
    print(f"  selection changed on {sum(r['changed'] for r in rows)}/{n} draws")
    print(f"  mean chars {sum(r['base_chars'] for r in rows)/n:.1f} -> "
          f"{sum(r['echo_chars'] for r in rows)/n:.1f}")
    print(f"  mean per-char logp {sum(r['base_logp'] for r in rows)/n:+.4f} -> "
          f"{sum(r['echo_logp'] for r in rows)/n:+.4f}")

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "ckpt": args.ckpt, "seeds": args.seeds, "n": args.n, "lam": args.lam,
        "draws": n,
        "base_rate": nb / n, "echo_rate": ne / n, "oracle_rate": no / n,
        "base_ci": [lo_b, hi_b], "echo_ci": [lo_e, hi_e],
        "discordant_echo_only": c, "discordant_base_only": b, "mcnemar_p": p,
        "rows": rows,
    }, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
