"""Score a v9 arm against the gate `PREDICTION_v9.md` fixed before it trained.

    python scripts/chat/score_v9.py --arm chat-v9-narrative --seeds 4

THE GATE, QUOTED RATHER THAN RE-DECIDED
---------------------------------------
`PREDICTION_v9.md` §1 fixes the primary as **held-out ORACLE topicality** from
`scripts/chat/echo_holdout.py --seeds 6 --n 8`, reported as k/120 with a 95 %
Wilson interval, against the incumbent's **11/120**. Not the battery's `topic`
column, which v8 showed coincides exactly with the echo partition (0.2969 both)
and is therefore circular; not the headline, which is 89 % `list` variance at 20
draws and 38 % persona recitation.

The oracle is the right column for a CORPUS arm specifically because it is
selector-independent: it asks whether the model can produce a reply about the
noun at all, which is the thing a corpus change is supposed to move.

Per-arm thresholds are read from this file's `THRESHOLDS`, transcribed from
§2 and not recomputed here. Each sits at a half-integer count so it cannot be
landed on exactly -- `CONVENTIONS.md` §4 rule 2, which P1 died of.

WHAT THIS DOES NOT DO
---------------------
It does not choose the metric after seeing the numbers, and it does not promote
a secondary column. Secondaries are printed because the pre-registration says to
report them, and they gate nothing.

Not part of the research protocol. No number here is a reported figure.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from snnchat.quality import rate_ci, resolves_against  # noqa: E402

#: k/120 an arm must reach, transcribed from PREDICTION_v9.md §2.
THRESHOLDS = {
    "chat-v9-narrative": 22.5,   # A12, "the round's real bet"
    "chat-v9-raw05": 17.5,       # A9, expected to FAIL, recorded anyway
    "chat-v9-turn600": 17.5,     # A11, oracle expected unchanged
    "chat-v9-bot1": 17.5,        # A13a
    "chat-v9-align0": 17.5,      # A13b
}

#: The incumbent, from the committed `echo_holdout.json`.
INCUMBENT_ORACLE = (11, 120)
INCUMBENT_BPC = 1.1742902997363112


def seed_runs(arm: str, n_seeds: int) -> list[str]:
    return [arm if i == 0 else f"{arm}-s{i}" for i in range(n_seeds)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--holdout-seeds", type=int, default=6)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--skip-run", action="store_true",
                    help="reuse existing echo_holdout artifacts")
    args = ap.parse_args()

    runs = seed_runs(args.arm, args.seeds)
    rows = []
    for run in runs:
        ckpt = ROOT / "experiments/chat" / run / "ckpt_best.pt"
        out = ROOT / "experiments/chat/_quality" / f"echo_holdout_{run}.json"
        if not ckpt.exists():
            print(f"  {run}: NO CHECKPOINT, skipped")
            continue
        if not (args.skip_run and out.exists()):
            cmd = [sys.executable, "scripts/chat/echo_holdout.py",
                   "--ckpt", str(ckpt.relative_to(ROOT)).replace("\\", "/"),
                   "--seeds", str(args.holdout_seeds), "--n", str(args.n),
                   "--out", str(out.relative_to(ROOT)).replace("\\", "/")]
            print(f"  $ {' '.join(cmd)}", flush=True)
            if subprocess.run(cmd, cwd=ROOT).returncode != 0:
                print(f"  {run}: SCORING FAILED")
                continue
        d = json.loads(out.read_text(encoding="utf-8"))
        summary = ROOT / "experiments/chat" / run / "summary.json"
        bpc = None
        if summary.exists():
            bpc = json.loads(summary.read_text(encoding="utf-8")).get("val", {}).get("weighted")
        rows.append({
            "run": run,
            "oracle_k": int(round(d["oracle_rate"] * d["draws"])),
            "oracle_n": d["draws"],
            "echo_k": int(round(d["echo_rate"] * d["draws"])),
            "base_k": int(round(d["base_rate"] * d["draws"])),
            "weighted_bpc": bpc,
        })

    if not rows:
        print("no seeds scored")
        return 1

    print(f"\n=== {args.arm}: PRIMARY = held-out ORACLE (PREDICTION_v9 §1) ===")
    print(f"{'run':<26} {'oracle':>12} {'95% CI':>20} {'echo':>8} {'bpc':>9}")
    for r in rows:
        ci = rate_ci(r["oracle_k"], r["oracle_n"])
        b = f"{r['weighted_bpc']:.4f}" if r["weighted_bpc"] else "-"
        print(f"{r['run']:<26} {r['oracle_k']:>4}/{r['oracle_n']:<7} "
              f"[{ci['ci_low']:.4f}, {ci['ci_high']:.4f}]".rjust(20)
              + f" {r['echo_k']:>4}/{r['oracle_n']:<3} {b:>9}")

    ik, ino = INCUMBENT_ORACLE
    ici = rate_ci(ik, ino)
    print(f"{'chat-v3d-aligned (incumbent)':<26} {ik:>4}/{ino:<7} "
          f"[{ici['ci_low']:.4f}, {ici['ci_high']:.4f}]".rjust(20)
          + f" {'-':>8} {INCUMBENT_BPC:>9.4f}")

    ks = [r["oracle_k"] for r in rows]
    pooled_k, pooled_n = sum(ks), sum(r["oracle_n"] for r in rows)
    pooled = rate_ci(pooled_k, pooled_n)
    mean_k = statistics.mean(ks)
    sd_k = statistics.stdev(ks) if len(ks) > 1 else float("nan")
    thr = THRESHOLDS.get(args.arm)

    print(f"\n  per-seed oracle counts : {ks}")
    print(f"  mean {mean_k:.2f}/120   SD_seed {sd_k:.2f}   "
          f"pooled {pooled_k}/{pooled_n} = {pooled['rate']:.4f} "
          f"95% CI [{pooled['ci_low']:.4f}, {pooled['ci_high']:.4f}]")
    if thr is not None:
        verdict = resolves_against(pooled_k, pooled_n, thr / 120.0)
        print(f"  threshold {thr}/120 = {thr / 120:.4f}  ->  VERDICT: {verdict.upper()}")
        # A mean below the bar is a miss whatever the interval does; the
        # interval decides whether the comparison was DECIDED, never whether a
        # bar that fired against the arm can be waived (CONVENTIONS.md §4 r5).
        print(f"  4-seed mean {mean_k:.2f} vs bar {thr}: "
              f"{'ABOVE' if mean_k > thr else 'BELOW'}")

    bpcs = [r["weighted_bpc"] for r in rows if r["weighted_bpc"]]
    if bpcs:
        m = statistics.mean(bpcs)
        print(f"\n  weighted held-out bpc: mean {m:.4f} vs incumbent "
              f"{INCUMBENT_BPC:.4f}  (delta {m - INCUMBENT_BPC:+.4f})")
        print("  NOTE: comparable only if the mixture is identical; A12 swaps "
              "0.10 of soda for soda_narrative, so it is NOT.")

    out = ROOT / "experiments/chat/_quality" / f"score_v9_{args.arm}.json"
    out.write_text(json.dumps({
        "arm": args.arm, "rows": rows,
        "pooled": pooled, "mean_k": mean_k, "sd_seed_k": sd_k,
        "threshold_k": thr,
        "verdict": resolves_against(pooled_k, pooled_n, thr / 120.0) if thr else None,
        "incumbent_oracle": list(INCUMBENT_ORACLE),
    }, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
