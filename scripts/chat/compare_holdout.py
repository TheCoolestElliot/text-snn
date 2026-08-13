"""Compare two `echo_holdout.py` artifacts draw by draw.

WHY A SEPARATE SCRIPT
---------------------
`echo_holdout.py` compares two SELECTORS on one pool, which is exactly paired --
the draw is held fixed and only the rule varies. Comparing two ways of *drawing*
the pool is a different and weaker comparison: the candidates differ, so the
pairing is only at the (prompt, seed) level and the model's own sampling noise
is inside the difference rather than outside it.

That is still the right pairing to use -- the same twenty prompts and the same
six seeds appear in both runs -- but it must be computed rather than eyeballed
off two summary lines, because a four-draw difference in a hundred and twenty is
inside the noise of this metric and looks like a result in a table.

Reports McNemar over draws and, because the draws are 20 prompts x 6 seeds and
therefore clustered, a sign test over prompt-level totals as well. If the two
disagree, believe the cluster-level one.

    python scripts/chat/compare_holdout.py A.json B.json --field oracle_hit
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def mcnemar(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value on the discordant pairs (b, c)."""
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, k) for k in range(0, min(b, c) + 1)) / (2 ** n)
    return min(1.0, 2.0 * tail)


def sign_test(better: int, worse: int) -> float:
    """Two-sided exact sign test. Same arithmetic as McNemar; different unit."""
    return mcnemar(worse, better)


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("baseline")
    ap.add_argument("challenger")
    ap.add_argument("--field", default="echo_hit",
                    choices=("echo_hit", "base_hit", "oracle_hit"),
                    help="echo_hit = what the decoder returns; oracle_hit = what "
                         "the pool contained, i.e. coverage rather than selection")
    args = ap.parse_args()

    a, b = load(ROOT / args.baseline), load(ROOT / args.challenger)
    if a.get("probe_set") != b.get("probe_set"):
        raise SystemExit(f"different probe sets: {a.get('probe_set')} vs {b.get('probe_set')}")

    key = lambda r: (r["prompt"], r["seed"])            # noqa: E731
    ra = {key(r): r for r in a["rows"]}
    rb = {key(r): r for r in b["rows"]}
    shared = sorted(set(ra) & set(rb))
    if not shared:
        raise SystemExit("no (prompt, seed) pairs in common")

    f = args.field
    ka = sum(ra[k][f] for k in shared)
    kb = sum(rb[k][f] for k in shared)
    better = sum(1 for k in shared if rb[k][f] > ra[k][f])
    worse = sum(1 for k in shared if rb[k][f] < ra[k][f])
    n = len(shared)

    by_prompt_a: dict[str, float] = defaultdict(float)
    by_prompt_b: dict[str, float] = defaultdict(float)
    for k in shared:
        by_prompt_a[k[0]] += ra[k][f]
        by_prompt_b[k[0]] += rb[k][f]
    p_better = sum(1 for p in by_prompt_a if by_prompt_b[p] > by_prompt_a[p])
    p_worse = sum(1 for p in by_prompt_a if by_prompt_b[p] < by_prompt_a[p])

    print(f"\n{f}  on {a.get('probe_set')}  ({n} paired draws, "
          f"{len(by_prompt_a)} prompts)")
    print(f"  baseline   {Path(args.baseline).name}")
    print(f"             n={a['n']} steer={a.get('steer_every', 0)}   "
          f"{ka/n:.4f}  ({int(ka)}/{n})")
    print(f"  challenger {Path(args.challenger).name}")
    print(f"             n={b['n']} steer={b.get('steer_every', 0)}   "
          f"{kb/n:.4f}  ({int(kb)}/{n})")
    print(f"  per draw   +{better} / -{worse}     McNemar p = {mcnemar(worse, better):.5f}")
    print(f"  per prompt +{p_better} / -{p_worse}     sign test p = "
          f"{sign_test(p_better, p_worse):.5f}")
    verdict = "unresolved"
    if mcnemar(worse, better) < 0.05 and sign_test(p_better, p_worse) < 0.05:
        verdict = "above" if kb > ka else "below"
    print(f"  verdict    {verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
