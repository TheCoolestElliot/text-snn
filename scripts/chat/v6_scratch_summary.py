"""Summarise `chat-v6-scratch` under the shipped decoder against the four incumbent seeds.

WHY THIS EXISTS
---------------
`chat-v6-scratch` was set aside on 2026-08-07 on the old `n = 8` headline
(`QUALITY_v6.md` section 2) and on a worse held-out bpc. It had never met the
`n = 256` echo decoder. The 2026-09-20 design panel asked for that reading as a
plain diagnostic for choosing a parent, and fixed its reading BEFORE any draw:
the WIDE oracle is "different from the incumbent" only outside 52-106 of 360
(`docs/chat/V6_SCRATCH_NOTE.md` section 1 quotes it).

This script draws nothing. It reads the pools `echo_holdout.py` wrote for
`chat-v6-scratch` on 2026-09-21 (stored gzip-compressed beside it) and the
committed `v12_*` files for the four `chat-v3d-aligned` seeds, and writes every
count the note quotes with its denominator and Wilson interval, per
`CONVENTIONS.md` section 1. The note quotes this file's output and nothing else.

Not part of the research protocol. No number here is a reported figure.

    python scripts/chat/v6_scratch_summary.py
"""
from __future__ import annotations

import argparse
import gzip
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from snnchat.quality import rate_ci, resolves_against  # noqa: E402

QDIR = ROOT / "experiments" / "chat" / "_quality"
SETS = ("heldout", "fresh", "wide")
INCUMBENT_TAGS = ("", "-s1", "-s2", "-s3")

#: The panel's pre-stated band on the WIDE oracle count, of 360. Inside it the
#: reading is UNRESOLVED; it is transcribed, not recomputed.
WIDE_ORACLE_BAND = (52, 106)


def _load(path: Path) -> dict:
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            return json.load(fh)
    return json.loads(path.read_text(encoding="utf-8"))


def _counts(d: dict) -> dict:
    rows = d["rows"]
    n = len(rows)
    return {
        "n": n,
        "selected": sum(1 for r in rows if r["echo_hit"]),
        "oracle": sum(1 for r in rows if r["oracle_hit"]),
        "score_only": sum(1 for r in rows if r["base_hit"]),
    }


def _ci(k: int, n: int) -> dict:
    c = rate_ci(k, n)
    return {"k": k, "n": n, "rate": c["rate"], "ci": [c["ci_low"], c["ci_high"]]}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=str(QDIR / "v6scratch_summary.json"))
    args = ap.parse_args()

    report: dict = {"decoder": {"n": 256, "lam": 0.6, "seeds": 6}, "sets": {}}
    for s in SETS:
        arm = _load(QDIR / f"v6scratch_{s}_n256.json.gz")
        assert (arm["n"], arm["lam"], arm["seeds"]) == (256, 0.6, 6), s
        a = _counts(arm)
        inc = [_counts(_load(QDIR / f"v12_{s}_chat-v3d-aligned{t}.json")) for t in INCUMBENT_TAGS]
        pooled = {k: sum(c[k] for c in inc) for k in ("n", "selected", "oracle")}
        block = {
            "v6_scratch": {k: _ci(a[k], a["n"]) for k in ("selected", "oracle", "score_only")},
            "incumbent_per_seed": {k: [c[k] for c in inc] for k in ("selected", "oracle")},
            "incumbent_pooled": {k: _ci(pooled[k], pooled["n"]) for k in ("selected", "oracle")},
            "shipped_checkpoint": {k: _ci(inc[0][k], inc[0]["n"]) for k in ("selected", "oracle")},
        }
        # The arm's interval against the incumbent's pooled rate: the
        # `resolves_against` rule, the same one every guard in the note uses.
        for k in ("selected", "oracle"):
            block["v6_scratch"][k]["vs_incumbent_pooled"] = resolves_against(
                a[k], a["n"], pooled[k] / pooled["n"])
        report["sets"][s] = block

    wide = report["sets"]["wide"]["v6_scratch"]["oracle"]["k"]
    lo, hi = WIDE_ORACLE_BAND
    report["prestated_wide_oracle"] = {
        "band": [lo, hi], "of": 360, "k": wide,
        "verdict": "different-above" if wide > hi else "different-below" if wide < lo else "unresolved",
    }

    dodge = json.loads((QDIR / "v6scratch_dodge_n256.json").read_text(encoding="utf-8"))["summary"]["echo"]
    inc_dodge = [json.loads((QDIR / f"v12_dodge_chat-v3d-aligned{t}.json").read_text(encoding="utf-8"))
                 ["summary"]["echo"] for t in INCUMBENT_TAGS]
    k_inc, n_inc = sum(d["k_lam"] for d in inc_dodge), sum(d["n"] for d in inc_dodge)
    report["story_dodge"] = {
        "v6_scratch": _ci(dodge["k_lam"], dodge["n"]),
        "incumbent_per_seed": [d["k_lam"] for d in inc_dodge],
        "incumbent_pooled": _ci(k_inc, n_inc),
        "vs_incumbent_pooled": resolves_against(dodge["k_lam"], dodge["n"], k_inc / n_inc),
    }

    mem = json.loads((QDIR / "v6scratch_memory_probe.json").read_text(encoding="utf-8"))["summary"]
    report["memory_probe"] = {dist: {"told": _ci(*v["told"]), "untold": _ci(*v["untold"])}
                              for dist, v in mem.items()}

    Path(args.out).write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    for s in SETS:
        b = report["sets"][s]
        v, p = b["v6_scratch"], b["incumbent_pooled"]
        print(f"{s:8s} selected {v['selected']['k']}/{v['selected']['n']} "
              f"[{v['selected']['ci'][0]:.3f}, {v['selected']['ci'][1]:.3f}] "
              f"({v['selected']['vs_incumbent_pooled']}) vs pooled {p['selected']['k']}/{p['selected']['n']}"
              f" | oracle {v['oracle']['k']}/{v['oracle']['n']} ({v['oracle']['vs_incumbent_pooled']})"
              f" vs pooled {p['oracle']['k']}/{p['oracle']['n']}; per seed {b['incumbent_per_seed']}")
    print("pre-stated WIDE oracle:", report["prestated_wide_oracle"])
    print("story_dodge:", report["story_dodge"]["v6_scratch"], report["story_dodge"]["vs_incumbent_pooled"],
          "vs pooled", report["story_dodge"]["incumbent_pooled"])
    print("memory probe d0:", report["memory_probe"]["0"])
    print("wrote", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
