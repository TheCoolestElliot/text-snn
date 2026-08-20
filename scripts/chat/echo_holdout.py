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
    final_ids,
    prompt_content_words,
    rerank,
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

#: ONE EXCEPTION TO THE CLAIM ABOVE, FOUND 2026-08-12 AND LEFT IN PLACE.
#: "garden" *is* in `quality.PROBES` -- `src/snnchat/quality.py:270`, as one of
#: the accepted words of the autumn-haiku `fact` probe. So 19 of the 20 nouns
#: are held out and one is not. It is left alone rather than swapped, because
#: every committed number in `QUALITY_v8.md` and `QUALITY_v9.md` was measured on
#: this exact list and changing it would make them incomparable with each other.
#: `FRESH` below is disjoint from both lists and is the clean set.
_HELDOUT_PROBE_OVERLAP = ("garden",)

#: A SECOND held-out set, written before it was ever run.
#:
#: `QUALITY_v9.md` §2 named the outstanding weakness in the package's own
#: evidence: the inverse-document-frequency tier is the SECOND selection rule
#: chosen by comparing candidates on the 120 draws above, so those draws have
#: been used for model selection and are no longer fully held out for it. The
#: document says a fresh noun list is the clean test and that it had not been
#: run.
#:
#: This is that list. Disjoint from `HELDOUT` and from every word in
#: `quality.PROBES`, same twenty-prompt shape, same spread of request frames,
#: and a deliberate mix of nouns the corpus will have seen often (drum, pumpkin)
#: and hardly at all (igloo, seashell).
FRESH: list[tuple[str, tuple[str, ...]]] = [
    ("tell me a story about a dolphin", ("dolphin",)),
    ("tell me a story about a windmill", ("windmill",)),
    ("tell me a story about a hedgehog", ("hedgehog",)),
    ("tell me a story about a trumpet", ("trumpet",)),
    ("tell me a story about a volcano", ("volcano",)),
    ("tell me a story about a scarecrow", ("scarecrow",)),
    ("tell me a story about a kangaroo", ("kangaroo",)),
    ("tell me a story about a teapot", ("teapot",)),
    ("tell me a story about an igloo", ("igloo",)),
    ("tell me a story about a peacock", ("peacock",)),
    ("tell me a story about a drum", ("drum",)),
    ("tell me a story about a campfire", ("campfire",)),
    ("tell me a story about a beaver", ("beaver",)),
    ("tell me a story about a seashell", ("seashell",)),
    ("tell me a story about a giraffe", ("giraffe",)),
    ("write me a little story about a pumpkin", ("pumpkin",)),
    ("can you tell me a story about a submarine", ("submarine",)),
    ("i want a story about a crocodile", ("crocodile",)),
    ("make up a story about a telescope", ("telescope",)),
    ("tell me a story about a raccoon", ("raccoon",)),
]

#: A THIRD list, sixty nouns, written before it was ever scored.
#:
#: WHY A WIDER LIST EXISTS, AND WHY IT HAD TO
#: -------------------------------------------
#: `PREDICTION_v12.md` §1 fixes a gate on a corpus change whose predicted effect
#: is a handful of nouns crossing a dose threshold. Twenty prompts cannot resolve
#: that: the per-prompt sign test's smallest attainable two-sided p is
#: `2^-(k-1)` on `k` discordant prompts, so with two or three prompts moving it
#: cannot reach 0.05 however many sampler seeds are drawn. More seeds shrink
#: draw noise; only more PROMPTS shrink prompt-set noise. Sixty is what the
#: power calculation in `PREDICTION_v12.md` §1 asks for and it is fixed here.
#:
#: THE SELECTION RULE, STATED BECAUSE IT IS THE PART THAT COULD BE RIGGED
#: ----------------------------------------------------------------------
#: 1. Candidates are the corpus's own eligible subjects
#:    (`src/snnchat/subject_freq.json`, built by `topic_of` over the raw file).
#: 2. Kept if the INCUMBENT rule requests them 20..400 times over the full scan
#:    -- roughly 10..200 times in the pack, the band where the incumbent's
#:    measured hit rate is ~0.04 and where an intervention on the request slot
#:    has room to act. **This is a pre-treatment covariate.** It is the
#:    incumbent's own distribution; the flattened corpus's counts were not
#:    consulted at any point in choosing these words, which is what stops the
#:    list being selected for nouns the treatment happens to help.
#: 3. Disjoint from `quality.PROBES`, `HELDOUT` and `FRESH` including plural and
#:    singular variants, so no number already published shares a noun with it.
#: 4. Singular, four characters or more, not in `topics.STOP_WORDS`.
#: 5. Then filtered by hand on ONE criterion -- whether "a story about a X" is
#:    grammatical English. The mechanical rule alone yields "a story about a
#:    wealthy" and "a story about a matches", which test the model's handling of
#:    malformed input rather than its topicality. Twenty were kept from each
#:    third of the band so the list spans it evenly.
#:
#: A finding that fell out of step 5 and is worth recording: `topic_of` admits
#: adjectives, because "a wealthy man" satisfies its determiner test. So the
#: SHIPPED corpus contains requests of the form "tell me a story about a
#: wealthy". That is a real defect in the subject extractor, it predates this
#: round, and it is documented rather than fixed here -- fixing it would change
#: the corpus under a comparison this round is in the middle of running.
#:
#: The trailing comment on each line is the incumbent's request count, kept so a
#: reader can check rule 2 without rebuilding the table.
WIDE: list[tuple[str, tuple[str, ...]]] = [
    # --- incumbent request count 300-399 ---
    ("tell me a story about a palace", ("palace",)),    # 399
    ("tell me a story about a reindeer", ("reindeer",)),# 399
    ("tell me a story about a billboard", ("billboard",)),# 398
    ("tell me a story about a buckle", ("buckle",)),    # 398
    ("tell me a story about a sunrise", ("sunrise",)),  # 398
    ("tell me a story about a zipper", ("zipper",)),    # 395
    ("tell me a story about an oven", ("oven",)),       # 390
    ("tell me a story about an alarm", ("alarm",)),     # 386
    ("tell me a story about a brick", ("brick",)),      # 385
    ("tell me a story about an apron", ("apron",)),     # 383
    ("tell me a story about a mustache", ("mustache",)),# 377
    ("tell me a story about a napkin", ("napkin",)),    # 375
    ("tell me a story about a shampoo", ("shampoo",)),  # 375
    ("tell me a story about a garage", ("garage",)),    # 373
    ("tell me a story about a feather", ("feather",)),  # 364
    ("tell me a story about a trunk", ("trunk",)),      # 356
    ("tell me a story about a leash", ("leash",)),      # 354
    ("tell me a story about a teaspoon", ("teaspoon",)),# 354
    ("tell me a story about a drawer", ("drawer",)),    # 352
    ("tell me a story about a sunset", ("sunset",)),    # 351
    # --- incumbent request count 150-299 ---
    ("tell me a story about a shelter", ("shelter",)),  # 298
    ("tell me a story about an ostrich", ("ostrich",)), # 290
    ("tell me a story about a poppy", ("poppy",)),      # 286
    ("tell me a story about a pocket", ("pocket",)),    # 282
    ("tell me a story about a battery", ("battery",)),  # 281
    ("tell me a story about an anchor", ("anchor",)),   # 278
    ("tell me a story about a lightning", ("lightning",)),# 273
    ("tell me a story about a veterinarian", ("veterinarian",)),# 271
    ("tell me a story about a dessert", ("dessert",)),  # 266
    ("tell me a story about a birdcage", ("birdcage",)),# 256
    ("tell me a story about a knob", ("knob",)),        # 253
    ("tell me a story about a ceiling", ("ceiling",)),  # 246
    ("tell me a story about a menu", ("menu",)),        # 244
    ("tell me a story about a daisy", ("daisy",)),      # 235
    ("tell me a story about a blueberry", ("blueberry",)),# 218
    ("tell me a story about a meadow", ("meadow",)),    # 213
    ("tell me a story about a necklace", ("necklace",)),# 213
    ("tell me a story about a mechanic", ("mechanic",)),# 183
    ("tell me a story about a creature", ("creature",)),# 173
    ("tell me a story about a playground", ("playground",)),# 173
    # --- incumbent request count 20-149 ---
    ("tell me a story about a sandcastle", ("sandcastle",)),# 145
    ("tell me a story about a fort", ("fort",)),        # 137
    ("tell me a story about a ribbon", ("ribbon",)),    # 131
    ("tell me a story about an invitation", ("invitation",)),# 128
    ("tell me a story about a printer", ("printer",)),  # 123
    ("tell me a story about a mailman", ("mailman",)),  # 122
    ("tell me a story about a bandage", ("bandage",)),  # 113
    ("tell me a story about a kitty", ("kitty",)),      # 106
    ("tell me a story about a superhero", ("superhero",)),# 101
    ("tell me a story about a grandmother", ("grandmother",)),# 87
    ("tell me a story about a sandbox", ("sandbox",)),  # 53
    ("tell me a story about a captain", ("captain",)),  # 50
    ("tell me a story about a surfboard", ("surfboard",)),# 48
    ("tell me a story about an airplane", ("airplane",)),# 45
    ("tell me a story about a sailboat", ("sailboat",)),# 45
    ("tell me a story about a spaceship", ("spaceship",)),# 39
    ("tell me a story about a tricycle", ("tricycle",)),# 37
    ("tell me a story about a trampoline", ("trampoline",)),# 36
    ("tell me a story about an eagle", ("eagle",)),     # 35
    ("tell me a story about a unicorn", ("unicorn",)), # 33
]

SETS = {"heldout": HELDOUT, "fresh": FRESH, "wide": WIDE}


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
    ap.add_argument("--steer-every", type=int, default=0,
                    help="resample the pool toward the subject every N characters "
                         "(0 = off, the pool as every earlier round drew it)")
    ap.add_argument("--steer-frac", type=float, default=0.25)
    ap.add_argument("--set", dest="probe_set", default="heldout", choices=sorted(SETS),
                    help="'heldout' is the list every committed number used; "
                         "'fresh' is the disjoint second list")
    ap.add_argument("--no-graph", action="store_true",
                    help="disable the captured-graph stepper; same draws, slower")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default="experiments/chat/_quality/echo_holdout.json")
    args = ap.parse_args()

    model, _cfg, _ck = load_chat_checkpoint(ROOT / args.ckpt, device=args.device)
    model.eval()
    tok = ChatTokenizer()
    rp = RerankParams(n=args.n, lam=args.lam, graph=not args.no_graph,
                      steer_every=args.steer_every, steer_frac=args.steer_frac)

    probes = SETS[args.probe_set]
    rows = []
    for prompt, words in probes:
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
                # The returned text, matching `rerank.final_ids`. Scoring the
                # untrimmed draft here is what made QUALITY_v9's n=128 row
                # (0.4167, replayed off the stored trimmed text) disagree with
                # the shipped selector (0.3917).
                text = tok.decode_visible(final_ids(c, tok, rp))
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
                return tok.decode_visible(final_ids(c, tok, rp)).strip()

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

    steer_note = (f" steer={args.steer_every}/{args.steer_frac:g}"
                  if args.steer_every else "")
    print(f"\n{args.probe_set} topics, {len(probes)} prompts x {args.seeds} seeds "
          f"= {n} draws, n={args.n} lambda={args.lam}{steer_note}")
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
        "ckpt": args.ckpt, "probe_set": args.probe_set,
        "seeds": args.seeds, "n": args.n, "lam": args.lam,
        "steer_every": args.steer_every, "steer_frac": args.steer_frac,
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
