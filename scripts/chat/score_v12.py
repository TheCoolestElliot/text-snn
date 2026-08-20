"""Score `chat-v12-rare` against the bars `PREDICTION_v12.md` fixed in advance.

THE GATE, QUOTED RATHER THAN RE-DECIDED
---------------------------------------
Every threshold in `THRESHOLDS` is transcribed from `PREDICTION_v12.md` §2 and
is not recomputed here. Each sits at a half-integer count over its stated
denominator so it cannot be landed on exactly -- `CONVENTIONS.md` §4 rule 2,
which P1 died of.

The control is the four `chat-v3d-aligned` checkpoints, re-scored with today's
decoder rather than quoted from a committed figure (`PREDICTION_v12.md` §2).
They are frozen artifacts trained on the identical recipe and mixture, so this
measures the anchor on today's tree, which is what decision #10 asks for.

WHAT THIS DOES NOT DO
---------------------
It does not choose the metric after seeing the numbers, and it does not promote
a secondary column. `FRESH`, the selected (non-oracle) rates and the weighted
bpc are printed because the pre-registration says to report them, and they gate
nothing. Weighted bpc is **not comparable** across the two arms at all -- the
mixture keeps every source and weight, but `stories_topic_rare` is a different
400 M characters with its own held-out tail.

Not part of the research protocol. No number here is a reported figure.

    python scripts/chat/score_v12.py
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from snnchat.quality import rate_ci, resolves_against  # noqa: E402

ARM = "chat-v12-rare"
CONTROL = "chat-v3d-aligned"
SEEDS = 4
QDIR = ROOT / "experiments" / "chat" / "_quality"

#: Transcribed from `PREDICTION_v12.md` §2. `(count, denominator, direction)`;
#: direction "min" means the arm must reach it, "max" means it must stay under.
THRESHOLDS = {
    # P16, primary: pooled WIDE oracle over 4 seeds x 60 prompts x 6 sampler seeds
    "P16_wide_oracle": (150.5, 1440, "min"),
    # P18, guard: pooled HELDOUT oracle, 4 x 20 x 6
    "P18_heldout_oracle": (216.5, 480, "min"),
    # P19, guard: pooled story_dodge, 4 x 12 probes x 6 seeds
    "P19_story_dodge": (72.5, 288, "max"),
}

#: Sampler seeds and pool size, fixed by `PREDICTION_v12.md` §1.
HOLDOUT_SEEDS = 6
POOL_N = 256


def seed_runs(arm: str, n: int) -> list[str]:
    """`<arm>`, `<arm>-s1`, ... -- the naming every driver in this repo uses."""
    return [arm if i == 0 else f"{arm}-s{i}" for i in range(n)]


def mcnemar(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, k) for k in range(0, min(b, c) + 1)) / (2 ** n)
    return min(1.0, 2.0 * tail)


def run_holdout(run: str, probe_set: str, *, force: bool) -> dict:
    """One `echo_holdout.py` invocation, cached on its artifact."""
    out = QDIR / f"v12_{probe_set}_{run}.json"
    if out.exists() and not force:
        return json.loads(out.read_text(encoding="utf-8"))
    cmd = [sys.executable, "scripts/chat/echo_holdout.py",
           "--ckpt", f"experiments/chat/{run}/ckpt_best.pt",
           "--set", probe_set, "--seeds", str(HOLDOUT_SEEDS),
           "--n", str(POOL_N), "--out", str(out.relative_to(ROOT))]
    print(f"    $ {' '.join(cmd[1:])}", flush=True)
    rc = subprocess.run(cmd, cwd=ROOT).returncode
    if rc != 0:
        raise SystemExit(f"{run}/{probe_set}: echo_holdout failed rc={rc}")
    return json.loads(out.read_text(encoding="utf-8"))


def run_dodge(run: str, *, force: bool) -> dict:
    out = QDIR / f"v12_dodge_{run}.json"
    if out.exists() and not force:
        return json.loads(out.read_text(encoding="utf-8"))
    cmd = [sys.executable, "scripts/chat/lambda_dodge.py",
           "--ckpt", f"experiments/chat/{run}/ckpt_best.pt",
           "--seeds", str(HOLDOUT_SEEDS), "--n", str(POOL_N),
           "--out", str(out.relative_to(ROOT))]
    print(f"    $ {' '.join(cmd[1:])}", flush=True)
    rc = subprocess.run(cmd, cwd=ROOT).returncode
    if rc != 0:
        raise SystemExit(f"{run}: lambda_dodge failed rc={rc}")
    return json.loads(out.read_text(encoding="utf-8"))


def pooled(blobs: list[dict], field: str) -> tuple[int, int]:
    k = sum(int(round(sum(r[field] for r in b["rows"]))) for b in blobs)
    n = sum(len(b["rows"]) for b in blobs)
    return k, n


def paired(arm_blobs: list[dict], ctl_blobs: list[dict], field: str):
    """Per-draw McNemar and per-prompt sign test, pairing on (prompt, seed).

    Training seed i of the arm is paired with training seed i of the control:
    both are four independent draws of the same recipe, so pairing them by index
    is arbitrary but unbiased, and it keeps the prompt and the sampler seed --
    the two large variance sources -- held fixed inside every pair.
    """
    b = c = 0
    per_prompt: dict[str, list[float]] = {}
    for a_blob, c_blob in zip(arm_blobs, ctl_blobs):
        ca = {(r["prompt"], r["seed"]): r[field] for r in c_blob["rows"]}
        for r in a_blob["rows"]:
            key = (r["prompt"], r["seed"])
            if key not in ca:
                continue
            av, cv = r[field], ca[key]
            if av > cv:
                c += 1
            elif cv > av:
                b += 1
            per_prompt.setdefault(r["prompt"], [0.0, 0.0])
            per_prompt[r["prompt"]][0] += av
            per_prompt[r["prompt"]][1] += cv
    p_better = sum(1 for v in per_prompt.values() if v[0] > v[1])
    p_worse = sum(1 for v in per_prompt.values() if v[1] > v[0])
    return {
        "draw_better": c, "draw_worse": b, "draw_p": mcnemar(b, c),
        "prompt_better": p_better, "prompt_worse": p_worse,
        "prompt_p": mcnemar(p_worse, p_better), "prompts": len(per_prompt),
        # The floor a sign test can reach on this many discordant prompts. A
        # test that cannot reach 0.05 has not failed to find an effect; it was
        # never able to.
        "prompt_p_floor": mcnemar(0, max(p_better + p_worse, 1)),
    }


def verdict_line(name: str, k: float, n: int, thr: float, direction: str) -> str:
    ci = rate_ci(int(round(k)), n)
    v = resolves_against(int(round(k)), n, thr / n)
    if direction == "max":
        # "below the bar" is the pass here, so the interval reading flips.
        passed = "PASS" if v == "below" else ("FAIL" if v == "above" else "unresolved")
    else:
        passed = "PASS" if v == "above" else ("FAIL" if v == "below" else "unresolved")
    return (f"  {name:<22} {k:.0f}/{n} = {ci['rate']:.4f}  "
            f"95% CI [{ci['ci_low']:.4f}, {ci['ci_high']:.4f}]  "
            f"bar {thr}/{n} = {thr / n:.5f} ({direction})  -> {passed}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--force", action="store_true", help="re-run cached scorings")
    ap.add_argument("--skip-dodge", action="store_true")
    ap.add_argument("--out", default="experiments/chat/_quality/score_v12.json")
    args = ap.parse_args()

    QDIR.mkdir(parents=True, exist_ok=True)
    arm_runs, ctl_runs = seed_runs(ARM, SEEDS), seed_runs(CONTROL, SEEDS)
    for r in arm_runs + ctl_runs:
        if not (ROOT / "experiments" / "chat" / r / "ckpt_best.pt").exists():
            raise SystemExit(f"missing checkpoint for {r}")

    data: dict = {"arm": ARM, "control": CONTROL, "seeds": SEEDS,
                  "holdout_seeds": HOLDOUT_SEEDS, "n": POOL_N, "sets": {}}
    for probe_set in ("wide", "heldout", "fresh"):
        print(f"\n[{probe_set}]", flush=True)
        a = [run_holdout(r, probe_set, force=args.force) for r in arm_runs]
        c = [run_holdout(r, probe_set, force=args.force) for r in ctl_runs]
        entry = {}
        for field in ("oracle_hit", "echo_hit"):
            ak, an = pooled(a, field)
            ck, cn = pooled(c, field)
            entry[field] = {
                "arm": {"k": ak, "n": an, **rate_ci(ak, an)},
                "control": {"k": ck, "n": cn, **rate_ci(ck, cn)},
                "paired": paired(a, c, field),
                "per_seed_arm": [int(round(sum(r[field] for r in b["rows"]))) for b in a],
                "per_seed_control": [int(round(sum(r[field] for r in b["rows"]))) for b in c],
            }
        data["sets"][probe_set] = entry

    if not args.skip_dodge:
        print("\n[story_dodge]", flush=True)
        ad = [run_dodge(r, force=args.force) for r in arm_runs]
        cd = [run_dodge(r, force=args.force) for r in ctl_runs]
        f = "echo_dodge_lam"
        data["story_dodge"] = {
            "arm_k": sum(sum(r[f] for r in b["rows"]) for b in ad),
            "arm_n": sum(len(b["rows"]) for b in ad),
            "control_k": sum(sum(r[f] for r in b["rows"]) for b in cd),
            "control_n": sum(len(b["rows"]) for b in cd),
        }

    # ---- the pre-registered bars ----------------------------------------
    print("\n" + "=" * 78)
    print("PRE-REGISTERED BARS (PREDICTION_v12.md §2, transcribed not recomputed)")
    print("=" * 78)
    results = {}
    w = data["sets"]["wide"]["oracle_hit"]["arm"]
    thr, den, d = THRESHOLDS["P16_wide_oracle"]
    print(verdict_line("P16 wide oracle", w["k"], w["n"], thr, d))
    results["P16"] = resolves_against(w["k"], w["n"], thr / den)

    h = data["sets"]["heldout"]["oracle_hit"]["arm"]
    thr, den, d = THRESHOLDS["P18_heldout_oracle"]
    print(verdict_line("P18 heldout guard", h["k"], h["n"], thr, d))
    results["P18"] = resolves_against(h["k"], h["n"], thr / den)

    if not args.skip_dodge:
        sd = data["story_dodge"]
        thr, den, d = THRESHOLDS["P19_story_dodge"]
        print(verdict_line("P19 story_dodge guard", sd["arm_k"], sd["arm_n"], thr, d))
        results["P19"] = resolves_against(int(sd["arm_k"]), sd["arm_n"], thr / den)

    print("\nP17 paired, BOTH gates must clear p < 0.05 IN THE ARM'S FAVOUR:")
    for probe_set in ("wide", "heldout", "fresh"):
        p = data["sets"][probe_set]["oracle_hit"]["paired"]
        # DIRECTION IS PART OF THE GATE, and the first version of this line
        # omitted it. `p < 0.05` on its own is "these differ", not "the arm
        # won": on `fresh` the draw-level test fires at p = 0.00209 with the
        # CONTROL ahead 35 to 13, and a gate that read only the p-value would
        # have printed PASS for an arm that is significantly worse. Caught
        # because the numbers were read before the verdict was believed.
        arm_ahead = (p["draw_better"] > p["draw_worse"]
                     and p["prompt_better"] > p["prompt_worse"])
        resolved = p["draw_p"] < 0.05 and p["prompt_p"] < 0.05
        if resolved and arm_ahead:
            gate = "PASS"
        elif resolved and not arm_ahead:
            gate = "FAIL (control ahead)"
        else:
            gate = "unresolved"
        print(f"  {probe_set:<8} draws +{p['draw_better']}/-{p['draw_worse']} "
              f"p={p['draw_p']:.5f}   prompts +{p['prompt_better']}/-{p['prompt_worse']} "
              f"p={p['prompt_p']:.5f} (floor {p['prompt_p_floor']:.5f})   -> {gate}")
        if probe_set == "wide":
            results["P17"] = gate

    data["thresholds"] = {k: list(v) for k, v in THRESHOLDS.items()}
    data["verdicts"] = results
    out = ROOT / args.out
    out.write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
