"""Is `lambda` still earning the forward pass it costs?

WHAT LAMBDA WAS FOR, AND WHY THAT IS THE QUESTION
--------------------------------------------------
`docs/chat/QUALITY.md` chose `lambda = 0.6` under a stated rule, and the thing
it bought was **not** topicality. The argmax of the pre-registered headline was
`lambda = 0`; 0.6 was taken because `lambda = 0` answers a prompt that asks for
no narrative WITH a narrative 16.7 % of the time against plain sampling's 8.3 %.
On a mixture that is 40 % stories, best-of-N by likelihood picks the story, and
the anti-LM term is what cancels that. So the question "does lambda still pay"
is a question about `story_dodge`, not about the topic column, and measuring it
on the topic column would answer the wrong one.

WHY IT IS WORTH ASKING NOW
--------------------------
Two things changed underneath it. `QUALITY_v8.md` found lambda **inert** on
held-out topic prompts -- the selected reply is identical at every lambda from
0.00 to 1.50 -- and the echo partition now decides most turns before the score
is consulted at all, so lambda only ever breaks ties inside the surviving tier.
Meanwhile it costs a full teacher-forced pass over the whole pool, which is the
single largest remaining item in a turn now that the draw is graph-captured.

Lambda was chosen in a world with no echo partition, so both worlds are measured
here: `echo=on` is what the REPL does, `echo=off` is the regime the 0.6 decision
was actually made in.

THE DESIGN
----------
One pool per (prompt, seed), selected twice. Lambda changes only `score`, so a
paired comparison on a shared pool holds the model, the sampler, the seeds and
the draws fixed and varies exactly one thing -- the same design
`echo_holdout.py` uses, and the reason McNemar is the right test.

`_is_story` is a pure text function, so no extra generation is needed to score
either arm.

Not part of the research protocol; no number here is a reported figure.

    python scripts/chat/lambda_dodge.py --seeds 6 --n 256
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import torch  # noqa: E402

from snnchat.generate import SamplingParams, load_chat_checkpoint  # noqa: E402
from snnchat.quality import PROBES, _is_story, rate_ci  # noqa: E402
from snnchat.rerank import (RerankParams, _score_null, echo_weight,  # noqa: E402
                            final_ids, prompt_content_words, sample_candidates)
from snnchat.tokenizer import BOS, BOT, ChatTokenizer  # noqa: E402

#: The kinds that do not ask for a narrative, so a narrative is a dodge. Same
#: rule `snnchat.quality` applies, named here so the two cannot drift.
DODGE_KINDS = ("list", "fact")


def mcnemar(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, k) for k in range(0, min(b, c) + 1)) / (2 ** n)
    return min(1.0, 2.0 * tail)


def _pick(cands, rp: RerankParams, lam: float, echo: bool):
    """`rerank`'s partitions, with `lam` and `echo` as free variables."""
    pool = [c for c in cands if c.n_chars >= rp.min_chars] or cands
    if echo:
        best = max((c.echo_weight for c in pool), default=0.0)
        if best > 0.0:
            pool = [c for c in pool if c.echo_weight >= best - 1e-9]
    key = lambda c: (c.logp_cond - lam * c.logp_null) / max(len(c.scored), 1)  # noqa: E731
    return max(pool, key=key)


@torch.no_grad()
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", default="experiments/chat/chat-v3d-aligned/ckpt_best.pt")
    ap.add_argument("--seeds", type=int, default=6)
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--max-new", type=int, default=300)
    ap.add_argument("--lam", type=float, default=0.6)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default="experiments/chat/_quality/lambda_dodge.json")
    args = ap.parse_args()

    model, _cfg, _ck = load_chat_checkpoint(ROOT / args.ckpt, device=args.device)
    model.eval()
    tok = ChatTokenizer()
    rp = RerankParams(n=args.n, lam=args.lam)
    probes = [p for p in PROBES if p.kind in DODGE_KINDS]

    rows = []
    for probe in probes:
        words = prompt_content_words(probe.prompt)
        for seed in range(args.seeds):
            params = SamplingParams(seed=seed, max_new=args.max_new)
            prefix = [BOS, *tok.render_turn("user", probe.prompt), BOT]
            lg, state, _ = model(torch.tensor([prefix], device=args.device), state=None)

            cands = sample_candidates(model, lg[:, -1, :].float(), state, params,
                                      rp.n, temperatures=None)
            # Every candidate gets a real null score: this script is measuring
            # what lambda does, so it may not use the shortcut that exists
            # precisely because lambda often cannot matter.
            _score_null(model, cands, rp, args.device)
            for c in cands:
                c.echo_weight = echo_weight(tok.decode_visible(final_ids(c, tok, rp)),
                                            words)

            entry = {"prompt": probe.prompt, "kind": probe.kind, "seed": seed}
            for echo in (True, False):
                tag = "echo" if echo else "noecho"
                hi = _pick(cands, rp, args.lam, echo)
                lo = _pick(cands, rp, 0.0, echo)
                t_hi = tok.decode_visible(final_ids(hi, tok, rp)).strip()
                t_lo = tok.decode_visible(final_ids(lo, tok, rp)).strip()
                entry[f"{tag}_dodge_lam"] = _is_story(t_hi)
                entry[f"{tag}_dodge_0"] = _is_story(t_lo)
                entry[f"{tag}_changed"] = hi is not lo
                entry[f"{tag}_logp_lam"] = hi.logp_cond / max(len(hi.scored), 1)
                entry[f"{tag}_logp_0"] = lo.logp_cond / max(len(lo.scored), 1)
                entry[f"{tag}_text_lam"] = t_hi[:200]
                entry[f"{tag}_text_0"] = t_lo[:200]
            rows.append(entry)

    n = len(rows)
    print(f"\nstory_dodge on {len(probes)} non-narrative probes x {args.seeds} seeds "
          f"= {n} draws, n={args.n}")
    summary = {}
    for tag, label in (("echo", "echo ON  (what the REPL does)"),
                       ("noecho", "echo OFF (the regime 0.6 was chosen in)")):
        k_hi = sum(r[f"{tag}_dodge_lam"] for r in rows)
        k_lo = sum(r[f"{tag}_dodge_0"] for r in rows)
        # b = lambda better, c = lambda worse
        b = sum(1 for r in rows if r[f"{tag}_dodge_0"] and not r[f"{tag}_dodge_lam"])
        c = sum(1 for r in rows if r[f"{tag}_dodge_lam"] and not r[f"{tag}_dodge_0"])
        p = mcnemar(c, b)
        changed = sum(r[f"{tag}_changed"] for r in rows)
        lp_hi = sum(r[f"{tag}_logp_lam"] for r in rows) / n
        lp_lo = sum(r[f"{tag}_logp_0"] for r in rows) / n
        print(f"\n  {label}")
        ci_hi, ci_lo = rate_ci(k_hi, n), rate_ci(k_lo, n)
        print(f"    lambda={args.lam:g} dodge {k_hi/n:.4f} ({k_hi}/{n})  "
              f"95% CI [{ci_hi['ci_low']:.4f}, {ci_hi['ci_high']:.4f}]")
        print(f"    lambda=0   dodge {k_lo/n:.4f} ({k_lo}/{n})  "
              f"95% CI [{ci_lo['ci_low']:.4f}, {ci_lo['ci_high']:.4f}]")
        print(f"    lambda fixes {b}, lambda breaks {c}   McNemar p = {p:.5f}")
        print(f"    lambda changed the pick on {changed}/{n} draws")
        print(f"    mean per-char logp {lp_lo:+.4f} (0) -> {lp_hi:+.4f} ({args.lam:g})")
        verdict = "unresolved" if p >= 0.05 else ("above" if k_hi < k_lo else "below")
        print(f"    verdict for lambda: {verdict}")
        summary[tag] = {"dodge_lam": k_hi / n, "dodge_0": k_lo / n, "k_lam": k_hi,
                        "k_0": k_lo, "n": n, "fixes": b, "breaks": c, "p": p,
                        "changed": changed, "logp_lam": lp_hi, "logp_0": lp_lo,
                        "verdict": verdict}

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"ckpt": args.ckpt, "n": args.n, "lam": args.lam,
                               "seeds": args.seeds, "summary": summary,
                               "rows": rows}, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
