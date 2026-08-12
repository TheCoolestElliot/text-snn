"""How much of each arm's score is the model reciting `snnchat.persona`?

`fallback_rate` was built to catch the model dodging a substantive prompt with
its stock identity/greeting line, and it is five hand-picked phrases. It misses
most of what the model actually reaches for: at the pinned reading row of the
shipped arm it reports 0.0000 while 7 of the 20 `list` picks are verbatim
persona text.

That is not a rounding error in a minor column. `list` is 20 draws at weight 0.3
and carries ~89 % of the headline's between-training-seed variance, so the
column that has decided every ship gate since 2026-08-06 is roughly a third
recitation, scored by a penalty term that cannot see it.

This script measures that across every committed arm, from the draws already on
disk. **It writes ONE new artifact and rewrites none of the 21** -- the existing
files are the record five rounds of documents quote, and a scorer that edits
them in place makes those documents uncheckable.

    python scripts/chat/canned_rate.py

Not part of the research protocol. No number here is a reported figure.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from snnchat.quality import (  # noqa: E402
    _is_canned,
    _is_fallback,
    _SUBSTANTIVE,
    rate_ci,
)

QDIR = ROOT / "experiments/chat/_quality"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="experiments/chat/_quality/canned_rate.json")
    args = ap.parse_args()

    files = sorted(p for p in QDIR.glob("chat-*.json")
                   if ".spread." not in p.name and ".narrow-social." not in p.name)

    rows = {}
    print(f"{'arm':<22} {'list picks':>11} {'rate':>7} {'95% CI':>18} "
          f"{'all subst.':>11} {'fallback':>9}")
    for f in files:
        rep = json.load(open(f, encoding="utf-8"))["report"]
        picks = rep.get("baseline_picks") or []
        if not picks:
            continue
        subst = [p for p in picks if p["kind"] in _SUBSTANTIVE]
        lst = [p for p in picks if p["kind"] == "list"]
        ck = sum(_is_canned(p["text"]) for p in lst)
        cs = sum(_is_canned(p["text"]) for p in subst)
        fb = sum(_is_fallback(p["text"]) for p in subst)
        ci = rate_ci(ck, len(lst))
        rows[f.stem] = {
            "list_canned": [ck, len(lst)], "list_rate": ci["rate"],
            "list_ci": [ci["ci_low"], ci["ci_high"]],
            "substantive_canned": [cs, len(subst)],
            "fallback_fired": [fb, len(subst)],
            "reported_fallback_rate": rep.get("fallback_rate"),
        }
        print(f"{f.stem:<22} {ck:>4}/{len(lst):<6} {ci['rate']:>7.4f} "
              f"[{ci['ci_low']:.3f}, {ci['ci_high']:.3f}]".ljust(60)
              + f"{cs:>4}/{len(subst):<6} {fb:>4}/{len(subst)}")

    n_arms = len(rows)
    tot_c = sum(r["list_canned"][0] for r in rows.values())
    tot_n = sum(r["list_canned"][1] for r in rows.values())
    tot_f = sum(r["fallback_fired"][0] for r in rows.values())
    tot_s = sum(r["fallback_fired"][1] for r in rows.values())
    pooled = rate_ci(tot_c, tot_n)
    print(f"\npooled over {n_arms} arms, `list` picks only: "
          f"{tot_c}/{tot_n} = {pooled['rate']:.4f} "
          f"95% CI [{pooled['ci_low']:.4f}, {pooled['ci_high']:.4f}]")
    print(f"`_is_fallback` fires on {tot_f}/{tot_s} substantive picks "
          f"({tot_f / max(tot_s, 1):.4f}) across the same arms")

    out = ROOT / args.out
    out.write_text(json.dumps({
        "note": "retrospective; computed from committed baseline_picks, "
                "rewrites nothing",
        "pooled_list": {"k": tot_c, "n": tot_n, "rate": pooled["rate"],
                        "ci": [pooled["ci_low"], pooled["ci_high"]]},
        "pooled_fallback_fired": [tot_f, tot_s],
        "arms": rows,
    }, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
