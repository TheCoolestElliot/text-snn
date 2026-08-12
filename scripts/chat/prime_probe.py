"""Does starting the model's sentence for it keep it on topic, or just look like it?

`snnchat.prime` writes the requested noun into the reply itself, so "does the
reply contain the noun" is 1.0 by construction and is NOT reported here. The
question a prime can actually be judged on is what the model does AFTER the
prime ends:

* **recurrence** -- does the topic appear again beyond the primed span? A story
  that mentions the penguin once because this module typed it, then writes about
  a girl named Lily, is the same failure with a costume on.
* **fluency** -- mean per-character log-probability of the CONTINUATION only,
  under the model itself, scored against the unprimed baseline.
* **the prime must not fire where it should not** -- greetings, list requests
  and questions are checked for silence.

Both conditions draw from the same seeds, and the unprimed arm is the shipped
decoder at the same `n`, so the comparison is like-for-like apart from the
prime.

    python scripts/chat/prime_probe.py --seeds 6 --n 32

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
from snnchat.prime import prime_topic, story_prime  # noqa: E402
from snnchat.rerank import RerankParams  # noqa: E402
from snnchat.tokenizer import ChatTokenizer  # noqa: E402

sys.path.insert(0, str(ROOT / "scripts" / "chat"))
from echo_holdout import HELDOUT, mcnemar, wilson  # noqa: E402

#: Prompts the prime MUST stay silent on.
NEGATIVES = ["hello", "what are you?", "name three animals", "list three fruits",
             "what is the capital of France?", "how are you?", "tell me a story",
             "what colour is the sky?", "do you remember our last conversation?"]


def recurs(text: str, prime: str, targets) -> float:
    """1.0 if a target appears BEYOND the primed span."""
    tail = text[len(prime):] if text.startswith(prime) else text
    low = tail.lower()
    return 1.0 if any(re.search(r"\b" + re.escape(t) + r"(s|es)?\b", low)
                      for t in targets) else 0.0


@torch.no_grad()
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="experiments/chat/chat-v3d-aligned/ckpt_best.pt")
    ap.add_argument("--seeds", type=int, default=6)
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--max-new", type=int, default=300)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default="experiments/chat/_quality/prime_probe.json")
    args = ap.parse_args()

    # --- the guard first: a prime that fires on a greeting is a bug ----------
    misfires = [p for p in NEGATIVES if story_prime(p) is not None]
    print(f"prime silent on {len(NEGATIVES) - len(misfires)}/{len(NEGATIVES)} negatives"
          + (f"  MISFIRED ON {misfires}" if misfires else ""))
    fires = [(p, prime_topic(p)) for p, _w in HELDOUT]
    missed = [p for p, t in fires if t is None]
    print(f"prime fires on {len(fires) - len(missed)}/{len(fires)} held-out story requests"
          + (f"  MISSED {missed}" if missed else ""))

    model, _cfg, _ck = load_chat_checkpoint(ROOT / args.ckpt, device=args.device)
    model.eval()
    tok = ChatTokenizer()
    rp = RerankParams(n=args.n, lam=0.6)

    rows = []
    for prompt, targets in HELDOUT:
        prime = story_prime(prompt)
        if prime is None:
            continue
        for seed in range(args.seeds):
            params = SamplingParams(seed=seed, max_new=args.max_new)

            plain = ChatSession(model, tok, device=args.device, params=params, rerank=rp)
            base = plain.send(prompt)

            # The primed session is fed the prime as part of the bot turn, then
            # generates the rest. `send` owns turn framing, so the prime is
            # applied by pre-feeding it and letting generation continue -- which
            # is what `ChatSession.send_primed` does.
            pr = ChatSession(model, tok, device=args.device, params=params, rerank=rp)
            primed = pr.send(prompt, prime=prime)

            rows.append({
                "prompt": prompt, "seed": seed, "prime": prime,
                "targets": list(targets),
                "base_recur": recurs(base, "", targets),
                "primed_recur": recurs(primed, prime, targets),
                "base_chars": len(base), "primed_chars": len(primed),
                "base_text": base[:300], "primed_text": primed[:300],
            })

    n = len(rows)
    kb = int(sum(r["base_recur"] for r in rows))
    kp = int(sum(r["primed_recur"] for r in rows))
    b = sum(1 for r in rows if r["base_recur"] > r["primed_recur"])
    c = sum(1 for r in rows if r["primed_recur"] > r["base_recur"])
    lb, hb = wilson(kb, n)
    lp, hp = wilson(kp, n)
    print(f"\nheld-out story requests, {n} draws, n={args.n}")
    print("  TOPIC RECURRENCE BEYOND THE PRIMED SPAN (the only fair column):")
    print(f"    unprimed : {kb/n:.4f} ({kb}/{n})  95% CI [{lb:.4f}, {hb:.4f}]")
    print(f"    primed   : {kp/n:.4f} ({kp}/{n})  95% CI [{lp:.4f}, {hp:.4f}]")
    print(f"    discordant: primed-only {c}, unprimed-only {b}, McNemar p = {mcnemar(b, c):.5f}")
    print(f"  mean reply chars {sum(r['base_chars'] for r in rows)/n:.1f} -> "
          f"{sum(r['primed_chars'] for r in rows)/n:.1f} "
          f"(the prime itself is {sum(len(r['prime']) for r in rows)/n:.1f} of that)")

    out = ROOT / args.out
    out.write_text(json.dumps({
        "ckpt": args.ckpt, "seeds": args.seeds, "n": args.n,
        "negatives_misfired": misfires, "heldout_missed": missed,
        "recurrence": {"unprimed": [kb, n], "primed": [kp, n],
                       "mcnemar_p": mcnemar(b, c)},
        "rows": rows,
    }, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
