"""What `lambda` is for, now that it is not the thing selecting for topicality.

`scripts/chat.py` fixed lambda = 0.6 in 2026-08-06 under a rule it states
explicitly: "maximise topicality subject to not making the model less fluent
than it is without reranking". At that time the anti-LM score was the ONLY thing
pushing a reply towards the prompt, so one constant was doing two jobs and the
setting was a compromise between them.

`QUALITY_v8.md`'s echo partition took the first job away. Inside a tier every
candidate already echoes the prompt equally, so lambda no longer decides whether
the reply is on topic -- it decides which of several on-topic drafts is kept.
That is a different question and the old answer was never tested against it.

This script replays the committed draws (no generation) across a lambda grid,
with the echo partition ON and OFF, and reports what actually moves.

    python scripts/chat/lambda_sweep.py

Not part of the research protocol. No number here is a reported figure.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from snnchat.quality import _score_one  # noqa: E402
from snnchat.rerank import echo_count, prompt_content_words  # noqa: E402

MIN_CHARS = 12


def select(texts, logp_cond, logp_null, lam, words, use_echo):
    """Reproduce `rerank`'s three partitions on stored draws."""
    n = len(texts)
    score = [(logp_cond[i] - lam * logp_null[i]) / max(len(texts[i]), 1) for i in range(n)]
    pool = [i for i in range(n) if len(texts[i]) >= MIN_CHARS] or list(range(n))
    if use_echo and words:
        ec = [echo_count(texts[i], words) for i in range(n)]
        best = max(ec[i] for i in pool)
        if best > 0:
            pool = [i for i in pool if ec[i] == best]
    return max(pool, key=lambda i: score[i])


def sweep(path, lams, n_use, use_echo):
    d = json.load(open(path, encoding="utf-8"))
    probes = {p["prompt"]: p for p in d["draws"]["probes"]}
    out = {}
    for lam in lams:
        agg, logp, chars, nd = {}, 0.0, 0, 0
        for dr in d["draws"]["draws"]:
            texts = dr["texts"][:n_use]
            if len(texts) < n_use:
                continue
            i = select(texts, dr["logp_cond"][:n_use], dr["logp_null"][:n_use],
                       lam, prompt_content_words(dr["prompt"]), use_echo)
            p = probes[dr["prompt"]]
            a = agg.setdefault(dr["kind"], [0.0, 0])
            a[0] += _score_one(p, texts[i])
            a[1] += 1
            # Fluency is read under the model's own distribution, unweighted by
            # lambda -- otherwise the metric moves with the knob being tuned.
            logp += dr["logp_cond"][i] / max(len(texts[i]), 1)
            chars += len(texts[i])
            nd += 1
        row = {k: v[0] / v[1] for k, v in agg.items()}
        row["mean_logp"] = logp / nd
        row["mean_chars"] = chars / nd
        row["headline"] = round(0.5 * row.get("topic", 0.0) + 0.3 * row.get("list", 0.0)
                                + 0.2 * row.get("social", 0.0), 4)
        out[lam] = row
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="*", default=[
        "chat-v3d-aligned", "chat-v6-scratch", "chat-v5a-short300", "chat-v6-inst-a"])
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--out", default="experiments/chat/_quality/lambda_sweep_v8.json")
    args = ap.parse_args()

    lams = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.25, 1.5]
    result = {}
    for arm in args.arms:
        path = ROOT / "experiments/chat/_quality" / f"{arm}.json"
        result[arm] = {
            "echo_on": sweep(path, lams, args.n, True),
            "echo_off": sweep(path, lams, args.n, False),
        }

    for mode in ("echo_off", "echo_on"):
        print(f"\n===== echo {mode.split('_')[1].upper()}, n={args.n} "
              f"(pooled over {len(args.arms)} arms) =====")
        print(f"{'lambda':>7} {'topic':>7} {'list':>7} {'social':>7} {'fact':>7} "
              f"{'identity':>9} {'logp':>8} {'chars':>7}")
        for lam in lams:
            rows = [result[a][mode][lam] for a in args.arms]
            def m(k):
                return sum(r.get(k, 0.0) for r in rows) / len(rows)
            star = " <- shipped" if lam == 0.6 else ""
            print(f"{lam:>7.2f} {m('topic'):>7.4f} {m('list'):>7.4f} {m('social'):>7.4f} "
                  f"{m('fact'):>7.4f} {m('identity'):>9.4f} {m('mean_logp'):>+8.4f} "
                  f"{m('mean_chars'):>7.1f}{star}")

    p = ROOT / args.out
    p.write_text(json.dumps({"n": args.n, "lambdas": lams, "arms": args.arms,
                             "result": result}, indent=2), encoding="utf-8")
    print(f"\nwrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
