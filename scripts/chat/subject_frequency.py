"""Is the topic failure a coverage problem, and coverage of WHAT?

THE QUESTION
------------
`QUALITY_v10.md` §6 leaves one live diagnosis: selection is at the pool's
ceiling, so what is left is whether the model ever *drafts* about a noun at all.
Two very different things could cause that, and they imply opposite next rounds:

* **Exposure.** The model has barely read the word. Fix: more text.
* **Conditioning.** The model has read the word plenty, but was rarely *asked*
  for it, so the request-to-subject mapping was never drilled on it. Fix: more
  distinct subjects in the request slot, which is a repack and not a new corpus.

`QUALITY_v9.md` §1 already spent 2.68 GPU-hours on a third hypothesis --
register, i.e. "it has only ever read a three-year-old's vocabulary" -- and that
came back `below` its bar. Exposure and conditioning have never been separated.

They are separable directly out of the packed corpus, with no GPU and no model.
A packed conversation is `<|user|> request <|eot|> <|bot|> reply <|eot|>`, so
counting a noun inside user turns gives how often it was ASKED for, and counting
it everywhere gives how often it was SEEN. This script computes both for the
forty nouns of `echo_holdout.HELDOUT` and `FRESH` and correlates each against
the per-noun hit rate those runs measured.

WHAT WOULD DISTINGUISH THEM
---------------------------
If hit rate tracks `asked` and not `seen`, the subject slot is the bottleneck
and widening it is cheap. If it tracks `seen` and not `asked`, the model needs
more text about more things, which is expensive. If it tracks neither, coverage
is not a frequency story at all and the next round has to look somewhere else.

Rank correlation rather than Pearson: the counts span four orders of magnitude
and the hit rate is a proportion over six seeds, so the relationship is not
expected to be linear and nothing here needs it to be.

Not part of the research protocol; no number here is a reported figure.

    python scripts/chat/subject_frequency.py
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "chat"))

from snnchat.tokenizer import BOS, BOT, EOT, USER, ChatTokenizer  # noqa: E402

from echo_holdout import FRESH, HELDOUT  # noqa: E402

#: Sentinel bytes the four markers decode to, so a turn boundary survives into
#: the byte string and a regex word boundary cannot run across one.
_MARK = {BOS: 0x00, USER: 0x01, BOT: 0x02, EOT: 0x03}


def _byte_view(path: Path) -> bytes:
    """A packed `.bin` as ASCII, markers replaced by sentinel bytes.

    The vocabulary is 102 ids and the packer writes `uint8`, so this is a table
    lookup over the whole file rather than a decode loop.
    """
    tok = ChatTokenizer()
    lut = np.zeros(256, dtype=np.uint8)
    for i in range(tok.n_special, tok.vocab_size):
        lut[i] = tok._itos[i]
    for i, b in _MARK.items():
        lut[i] = b
    ids = np.fromfile(path, dtype=np.uint8)
    return lut[ids].tobytes()


def _user_turns(blob: bytes) -> bytes:
    """Just the request text, concatenated, one per `<|user|>` marker.

    Each user turn runs from its marker to the next `<|eot|>`. Joined with a
    sentinel so a word boundary cannot straddle two requests.
    """
    out = []
    for seg in blob.split(b"\x01")[1:]:
        out.append(seg.split(b"\x03", 1)[0])
    return b"\x01".join(out)


def _count(blob: bytes, word: str) -> int:
    """Occurrences of `word` on a word boundary, plural tolerated.

    Case-insensitive: the corpus capitalises a subject at the start of a
    sentence and that is the same word.
    """
    pat = re.compile(rb"\b" + re.escape(word.encode()) + rb"(s|es)?\b", re.IGNORECASE)
    return sum(1 for _ in pat.finditer(blob))


def spearman(xs, ys) -> float:
    """Rank correlation, ties averaged. Stdlib only -- there is no scipy here."""
    def ranks(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r

    rx, ry = ranks(list(xs)), ranks(list(ys))
    n = len(rx)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
    return num / den if den else 0.0


def spearman_p(rho: float, n: int) -> float:
    """Two-sided p via the t approximation. Adequate at n = 40, crude at n = 20."""
    if n < 4 or abs(rho) >= 1.0:
        return 0.0 if abs(rho) >= 1.0 else 1.0
    t = rho * math.sqrt((n - 2) / (1 - rho * rho))
    df = n - 2
    # regularised incomplete beta via continued fraction is overkill; this is
    # the standard survival function of |t| under Student-t, by numeric quadrature
    steps = 20000
    hi = abs(t) + 60.0
    lo = abs(t)
    h = (hi - lo) / steps
    lg = math.lgamma((df + 1) / 2) - math.lgamma(df / 2) - 0.5 * math.log(df * math.pi)
    total = 0.0
    for i in range(steps + 1):
        x = lo + i * h
        w = 1 if i in (0, steps) else (4 if i % 2 else 2)
        total += w * math.exp(lg - (df + 1) / 2 * math.log1p(x * x / df))
    return min(1.0, 2.0 * total * h / 3.0)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sources", nargs="+",
                    default=["stories_topic", "tinystories"],
                    help="packed corpora to scan, in mixture order")
    ap.add_argument("--data-dir", default="data/chat")
    ap.add_argument("--heldout-run", default="experiments/chat/_quality/v10_heldout_n128.json")
    ap.add_argument("--fresh-run", default="experiments/chat/_quality/v10_fresh_n128.json")
    ap.add_argument("--field", default="oracle_hit", choices=("oracle_hit", "echo_hit"),
                    help="oracle_hit is POOL COVERAGE, which is the quantity this "
                         "diagnostic is about; echo_hit adds the selector")
    ap.add_argument("--out", default="experiments/chat/_quality/subject_frequency.json")
    args = ap.parse_args()

    nouns: list[tuple[str, str]] = []          # (prompt, the noun scored)
    for prompt, words in HELDOUT:
        nouns.append((prompt, words[0], "heldout"))
    for prompt, words in FRESH:
        nouns.append((prompt, words[0], "fresh"))

    counts: dict[str, dict[str, int]] = {w: {} for _, w, _ in nouns}
    for src in args.sources:
        path = ROOT / args.data_dir / f"{src}.bin"
        if not path.exists():
            print(f"  (skipping {src}: not packed)")
            continue
        print(f"  scanning {src} ({path.stat().st_size / 2**20:.0f} MiB) ...")
        blob = _byte_view(path)
        asked_blob = _user_turns(blob)
        for _, w, _set in nouns:
            counts[w][f"{src}_seen"] = _count(blob, w)
            counts[w][f"{src}_asked"] = _count(asked_blob, w)
        del blob, asked_blob

    hits: dict[str, tuple[float, int]] = {}
    for path, label in ((args.heldout_run, "heldout"), (args.fresh_run, "fresh")):
        blob = json.loads((ROOT / path).read_text(encoding="utf-8"))
        per: dict[str, list[float]] = {}
        for row in blob["rows"]:
            per.setdefault(row["prompt"], []).append(row[args.field])
        for prompt, vals in per.items():
            hits[prompt] = (sum(vals) / len(vals), len(vals))

    rows = []
    for prompt, w, which in nouns:
        rate, n_seeds = hits.get(prompt, (float("nan"), 0))
        rows.append({
            "noun": w, "set": which, "prompt": prompt,
            "hit_rate": rate, "seeds": n_seeds,
            **counts[w],
        })

    total_asked = [sum(r.get(f"{s}_asked", 0) for s in args.sources) for r in rows]
    total_seen = [sum(r.get(f"{s}_seen", 0) for s in args.sources) for r in rows]
    for r, a, s in zip(rows, total_asked, total_seen):
        r["asked"], r["seen"] = a, s

    rows.sort(key=lambda r: -r["asked"])
    print(f"\n{'noun':<13}{'set':<9}{'asked':>9}{'seen':>10}{'asked/seen':>12}"
          f"{'  ' + args.field:>14}")
    for r in rows:
        ratio = r["asked"] / r["seen"] if r["seen"] else float("nan")
        print(f"{r['noun']:<13}{r['set']:<9}{r['asked']:>9}{r['seen']:>10}"
              f"{ratio:>12.3f}{r['hit_rate']:>14.3f}")

    rates = [r["hit_rate"] for r in rows]
    print(f"\nrank correlation with {args.field}, n = {len(rows)} nouns")
    summary = {}
    for name, xs in (("asked", [r["asked"] for r in rows]),
                     ("seen", [r["seen"] for r in rows]),
                     ("log asked", [math.log1p(r["asked"]) for r in rows]),
                     ("asked/seen", [r["asked"] / r["seen"] if r["seen"] else 0.0
                                     for r in rows])):
        rho = spearman(xs, rates)
        p = spearman_p(rho, len(rows))
        print(f"  {name:<12} rho = {rho:+.3f}   p = {p:.4f}")
        summary[name] = {"rho": rho, "p": p}

    # PARTIALS, WHICH ARE THE WHOLE POINT
    # -----------------------------------
    # `asked` and `seen` are mechanically coupled -- a story about a penguin is
    # what makes "penguin" available as a subject in the first place -- so their
    # marginal correlations with hit rate cannot separate the two hypotheses.
    # The partial rank correlation asks each one's question with the other held
    # fixed: does being ASKED more, at matched exposure, predict a hit?
    r_ha = spearman([r["asked"] for r in rows], rates)
    r_hs = spearman([r["seen"] for r in rows], rates)
    r_as = spearman([r["asked"] for r in rows], [r["seen"] for r in rows])
    print(f"\n  asked vs seen themselves: rho = {r_as:+.3f}  "
          f"(this is why the marginals above cannot decide anything)")

    def partial(r_xy, r_xz, r_yz):
        den = math.sqrt(max(1e-12, (1 - r_xz ** 2) * (1 - r_yz ** 2)))
        return (r_xy - r_xz * r_yz) / den

    p_asked = partial(r_ha, r_as, r_hs)
    p_seen = partial(r_hs, r_as, r_ha)
    # One fewer degree of freedom for the controlled variable.
    print(f"  hit ~ asked | seen   rho = {p_asked:+.3f}   p = "
          f"{spearman_p(p_asked, len(rows) - 1):.4f}")
    print(f"  hit ~ seen  | asked  rho = {p_seen:+.3f}   p = "
          f"{spearman_p(p_seen, len(rows) - 1):.4f}")
    summary["partial_asked_given_seen"] = {
        "rho": p_asked, "p": spearman_p(p_asked, len(rows) - 1)}
    summary["partial_seen_given_asked"] = {
        "rho": p_seen, "p": spearman_p(p_seen, len(rows) - 1)}
    summary["asked_vs_seen"] = {"rho": r_as}

    # The same correlation WITHIN each list, because the two lists differ in
    # difficulty and a between-list effect could masquerade as a within-list one.
    for which in ("heldout", "fresh"):
        sub = [r for r in rows if r["set"] == which]
        rho = spearman([r["asked"] for r in sub], [r["hit_rate"] for r in sub])
        print(f"  asked, {which:<7} rho = {rho:+.3f}   p = "
              f"{spearman_p(rho, len(sub)):.4f}   (n = {len(sub)})")
        summary[f"asked_{which}"] = {"rho": rho, "p": spearman_p(rho, len(sub))}

    # HOW BIG IS THE SUBJECT VOCABULARY THE CORPUS ACTUALLY DRILLS?
    # -------------------------------------------------------------
    # The partial above says the request slot is what matters, which makes the
    # size and shape of that slot's vocabulary the thing to cost. Read straight
    # off the user turns rather than re-derived through `snnchat.topics`, so it
    # reports what was PACKED and not what the packer intended.
    vocab: dict[str, int] = {}
    subj_re = re.compile(rb"about (?:a|an|the) ([a-z]+)")
    for src in args.sources:
        path = ROOT / args.data_dir / f"{src}.bin"
        if not path.exists():
            continue
        asked_blob = _user_turns(_byte_view(path))
        for m in subj_re.finditer(asked_blob):
            w = m.group(1).decode()
            vocab[w] = vocab.get(w, 0) + 1
    if vocab:
        ordered = sorted(vocab.values(), reverse=True)
        total = sum(ordered)
        print(f"\n  subject vocabulary actually packed: {len(vocab)} distinct, "
              f"{total} requests")
        for thresh in (1000, 300, 100, 30, 10, 3):
            k = sum(1 for v in ordered if v >= thresh)
            print(f"    {k:>5} subjects asked for >= {thresh:>4} times "
                  f"({sum(v for v in ordered if v >= thresh) / total:.1%} of requests)")
        top = sorted(vocab.items(), key=lambda kv: -kv[1])[:12]
        print("    most-drilled: " + ", ".join(f"{w} {n}" for w, n in top))
        summary["subject_vocab"] = {"distinct": len(vocab), "requests": total}

    zero = [r["noun"] for r in rows if r["asked"] == 0]
    print(f"\n  nouns NEVER asked for: {len(zero)}/{len(rows)}  {zero}")
    hit_if_asked = [r["hit_rate"] for r in rows if r["asked"] > 0]
    hit_if_not = [r["hit_rate"] for r in rows if r["asked"] == 0]
    if hit_if_not:
        print(f"  mean {args.field}: asked {sum(hit_if_asked)/len(hit_if_asked):.3f}"
              f"  vs never-asked {sum(hit_if_not)/len(hit_if_not):.3f}")

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "sources": args.sources, "field": args.field,
        "correlations": summary, "rows": rows,
    }, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
