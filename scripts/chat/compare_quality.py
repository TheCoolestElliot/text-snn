"""Put several `quality.py` result files side by side, as markdown.

    python scripts/chat/compare_quality.py experiments/chat/_quality/*.json

Exists so that the tables in `docs/chat/QUALITY.md` are generated from the
result files rather than typed, which is the difference between a number that
can be traced and a number that was once true.

Prints two tables: the reranking-independent one (prompt dependence, which needs
no sampler) and the battery, each arm shown at its own best reranking setting
AND at `n = 1`, so an improvement from training is never confused with one from
decoding.
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path


def _load(paths: list[str]) -> list[dict]:
    out = []
    for pattern in paths:
        for path in sorted(glob.glob(pattern)) or [pattern]:
            with open(path, encoding="utf-8") as fh:
                out.append(json.load(fh)["report"])
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("results", nargs="+")
    p.add_argument("--n", type=int, default=None,
                   help="score every arm at this candidate count instead of its best")
    p.add_argument("--lam", type=float, default=None)
    args = p.parse_args(argv)

    reports = _load(args.results)
    if not reports:
        print("nothing to compare", file=sys.stderr)
        return 1

    print("### prompt dependence (held-out, no sampler, no lexicon)\n")
    print("| arm | step | bpc, own prompt | bpc, another prompt | delta |")
    print("| --- | ---: | ---: | ---: | ---: |")
    for r in reports:
        d = r.get("prompt_dependence") or {}
        if "delta" not in d:
            continue
        print(f"| `{r['label']}` | {r['step']:,} | {d['bpc_matched']:.4f} | "
              f"{d['bpc_shuffled']:.4f} | **{d['delta']:+.4f}** |")

    def row(r, s):
        k = s["by_kind"]
        spread = r.get("temperature_spread", 0.0) or 0.0
        tag = f"{r['label']}" + (f" +spread {spread:g}" if spread else "")
        return (f"| `{tag}` | {s['n']} | {s['lambda']:g} | "
                f"{k.get('topic', 0):.3f} | {s.get('list_strict', 0):.3f} | "
                f"{k.get('social', 0):.3f} | {k.get('identity', 0):.3f} | "
                f"{k.get('fact', 0):.3f} | {s['fallback_rate']:.3f} | "
                f"{s.get('story_dodge', 0):.3f} | {s.get('mean_chars', 0):.0f} | "
                f"{s.get('mean_logp', 0):.4f} | "
                f"**{s['headline']:.3f}** |")

    # `mean chars` is in this table on purpose. `list (strict)` requires a reply
    # under 120 characters, so an arm that simply produces shorter replies scores
    # better on it without following any instruction. The two columns have to be
    # read together or the strict metric is a length detector wearing a hat.
    print("\n### battery\n")
    print("| arm | n | lam | topic | list (strict) | social | identity | fact | "
          "fallback | story dodge | mean chars | logp/char | headline |")
    print("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | "
          "---: | ---: | ---: |")
    for r in reports:
        sweep = r["sweep"]
        plain = next((s for s in sweep if s["n"] == 1), None)
        if plain:
            print(row(r, plain))
        # Default to the SHIPPED decoder setting, not each arm's own argmax:
        # letting every arm pick its best row compares four different decoders
        # and calls the result a model comparison.
        want_n = args.n if args.n is not None else 8
        want_lam = args.lam if args.lam is not None else 0.6
        pick = next((s for s in sweep
                     if s["n"] == want_n and abs(s["lambda"] - want_lam) < 1e-9), None)
        if pick and pick is not plain:
            print(row(r, pick))

    print("\n### topic propensity over EVERY candidate drawn\n")
    print("Not a measure of the decoder: the hit rate over all ~1,024 drawn")
    print("candidates rather than over the 64 selected ones, so it is the model's")
    print("own tendency to mention what it was asked about, at 16x the sample")
    print("size. The selected-reply column above cannot resolve a difference")
    print("below about 0.07; this one resolves to about 0.02.\n")
    print("| arm | topic | 2 s.e. | n | list | social |")
    print("| --- | ---: | ---: | ---: | ---: | ---: |")
    for r in reports:
        s = max(r["sweep"], key=lambda x: x["headline"])
        pool = s.get("pool_by_kind") or {}
        ns = s.get("pool_n") or {}
        if not pool:
            continue
        p, n = pool.get("topic", 0.0), ns.get("topic", 1)
        se = 2 * (max(p, 1e-9) * (1 - p) / max(n, 1)) ** 0.5
        print(f"| `{r['label']}` | **{p:.4f}** | ±{se:.4f} | {n:,} | "
              f"{pool.get('list', 0):.4f} | {pool.get('social', 0):.4f} |")

    print("\n### per probe, at each arm's best setting\n")
    labels = [r["label"] for r in reports]
    bests = [max(r["sweep"], key=lambda s: s["headline"]) for r in reports]
    probes = list(bests[0]["per_probe"])
    print("| probe | " + " | ".join(f"`{n}`" for n in labels) + " |")
    print("| --- | " + " | ".join("---:" for _ in labels) + " |")
    for probe in probes:
        cells = " | ".join(f"{b['per_probe'].get(probe, float('nan')):.2f}" for b in bests)
        print(f"| {probe} | {cells} |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
