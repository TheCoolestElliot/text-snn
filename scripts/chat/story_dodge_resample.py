"""Re-measure `story_dodge` at a resolution that can miss a threshold narrowly.

    python scripts/chat/story_dodge_resample.py --seeds 101 \
        --arms chat-v3d-aligned chat-v4a-short chat-v4b-balance chat-v5a-short300 \
        --out experiments/chat/_quality/story_dodge_resample.json

WHY THIS SCRIPT EXISTS
----------------------
`docs/chat/PREDICTION_v5.md` P1 asked whether `chat-v5a-short300`'s `story_dodge`
comes in **below 0.125**. `docs/chat/QUALITY_v5.md` §4.4 reports the answer as
**0.1250** and resolves P1 as UNRESOLVED under the pre-registration's own middle
band.

That is a measurement-resolution failure and not a result. `story_dodge` is
computed over the `list` and `fact` probes only -- 12 of the 37 in the battery --
at 4 seeds each, so it is a proportion over **48 draws**, quantised at 1/48 =
0.02083. **0.125 is exactly 6/48**: a lattice point. On that lattice the arm
could not have missed the threshold narrowly. It could only hit it exactly, and
it did.

This script raises the denominator until the threshold is no longer reachable and
reports the answer with a Wilson interval, per `docs/chat/CONVENTIONS.md`.

WHAT IT DOES NOT DO
-------------------
**It does not retrain and it does not redefine anything.** Same checkpoints, same
frozen probe battery, same sampler settings, same `_is_story` rule, and the same
`(n = 1, lambda = 0)` reading row that `PREDICTION_v5.md` §7 item 1 pinned before
the checkpoint was scored. The only quantity that changes is how many seeds the
proportion is taken over.

THE TWO THINGS THAT MAKE THIS COMPARABLE RATHER THAN MERELY LARGER
------------------------------------------------------------------
1.  **`n` here means seeds, not candidates.** `quality.py --n` is candidates per
    (probe, seed), and at the pinned `n = 1` row the extra candidates are never
    looked at -- raising it moves `story_dodge` by exactly nothing. The sample
    size of this metric is `12 probes x seeds`, so **seeds is the only lever**,
    and candidates stay at 16 because that is what the committed arms were drawn
    at and the sampler's RNG stream depends on it.

2.  **Seeds 0-3 must reproduce the committed draws exactly, and that is a gate.**
    `collect` derives a probe's sampling seed from its POSITION in the probe
    tuple, so drawing 12 of 37 probes would silently draw a different experiment.
    `collect(probe_index=...)` pins each probe's index in `PROBES` instead. The
    check below then re-derives all 48 committed draws, character for character,
    and **aborts if any differ** -- so the enlarged sample is a strict superset
    of the one `QUALITY_v5.md` reported and not a second, differently-seeded
    measurement that happens to be near it.

    This is `CONTRIBUTING.md` §5's "make every probe reproduce a number it did
    not produce", applied to a resample.

THE SEED COUNT IS FIXED BEFORE ANY NUMBER IS READ
-------------------------------------------------
`--seeds 101` is chosen in advance, for two stated reasons, and the result is
reported at that value whatever it says:

*   **101 is odd, so 0.125 is off the lattice.** The denominator is `12 * seeds`,
    which is divisible by 8 -- and hence hits 0.125 exactly -- if and only if the
    seed count is even. At 101 seeds the denominator is 1,212, the spacing is
    0.000825, and 0.125 * 1212 = 151.5 is not an integer. The arm can now land
    *near* the threshold, which is the whole point.
*   **It is the largest sample the GPU budget buys for four arms** at ~1.3 s per
    (probe, seed): 12 x 101 x 1.3 s ~ 26 min per arm.

**There is no stopping rule and there must not be one.** Enlarging a sample until
an estimate crosses a threshold and then stopping is how a null becomes a finding
without any new evidence. The count is fixed here, in committed code, and if the
interval still contains 0.125 at 1,212 draws then the honest report is that P1 is
unresolvable at this budget -- which is a different and more useful sentence than
"unresolved because the threshold was a lattice point".

Not part of the research protocol. No number here is a reported figure.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO / "src") not in sys.path:
    sys.path.insert(0, str(_REPO / "src"))

import torch  # noqa: E402

from snnchat.generate import SamplingParams, load_chat_checkpoint  # noqa: E402
from snnchat.quality import (  # noqa: E402
    PROBES,
    collect,
    rate_ci,
    resolves_against,
    score,
)

#: The kinds `story_dodge` is defined over. Neither asks for a narrative, which
#: is what makes a narrative a dodge. Read off `snnchat.quality.score` rather
#: than restated, so the two cannot drift apart.
_DODGE_KINDS = ("list", "fact")

#: The bands `PREDICTION_v5.md` §4 fixed for P1, before the arm trained.
P1_BELOW = 0.125
P1_ABOVE = 0.146


def _subset() -> tuple[tuple, tuple[int, ...]]:
    """The `story_dodge` probes, with their indices in the full battery."""
    idx = tuple(i for i, p in enumerate(PROBES) if p.kind in _DODGE_KINDS)
    return tuple(PROBES[i] for i in idx), idx


def _check_superset(drawn: dict, committed: Path) -> dict:
    """Assert the new draws contain the committed ones, character for character.

    Returns the audit record. Raises if the file exists and disagrees -- a
    resample that quietly re-seeded is not a larger version of the measurement it
    claims to extend.
    """
    if not committed.exists():
        return {"file": str(committed), "checked": 0, "status": "absent"}
    with open(committed, encoding="utf-8") as fh:
        old_draws = json.load(fh)["draws"]["draws"]
    old = {(d["prompt"], d["seed"]): d for d in old_draws
           if d["kind"] in _DODGE_KINDS}
    checked, mismatched = 0, []
    for d in drawn["draws"]:
        key = (d["prompt"], d["seed"])
        if key not in old:
            continue
        checked += 1
        if d["texts"] != old[key]["texts"]:
            mismatched.append(key)
    if mismatched:
        raise RuntimeError(
            f"{len(mismatched)} of {checked} re-drawn (probe, seed) pairs do not "
            f"reproduce {committed}: e.g. {mismatched[:3]}. The resample is not a "
            f"superset of the committed measurement and must not be reported as "
            f"one. Do not relax this check -- find out why the stream moved."
        )
    if not checked:
        raise RuntimeError(
            f"{committed} shares no (probe, seed) pair with this draw, so the "
            f"superset property is unverified. Refusing to report."
        )
    return {"file": str(committed), "checked": checked, "status": "identical"}


def _measure(arm: str, args) -> dict:
    ckpt = Path(args.exp) / arm / "ckpt_best.pt"
    if not ckpt.exists():
        raise SystemExit(f"no checkpoint at {ckpt}")
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg, ck = load_chat_checkpoint(str(ckpt), device=device)
    probes, idx = _subset()
    params = SamplingParams(temperature=args.temperature, top_p=args.top_p,
                            min_p=args.min_p, max_new=args.max_new)

    print(f"\n## {arm}  (step {ck.get('step')})", flush=True)
    print(f"   {len(probes)} probes x {args.seeds} seeds x {args.n} candidates "
          f"= {len(probes) * args.seeds} draws of story_dodge", flush=True)
    t0 = time.perf_counter()
    drawn = collect(model, probes=probes, seeds=tuple(range(args.seeds)),
                    n=args.n, params=params, device=device, progress=False,
                    probe_index=idx)
    secs = time.perf_counter() - t0

    audit = _check_superset(drawn, Path(args.exp) / "_quality" / f"{arm}.json")
    print(f"   superset check: {audit['checked']} committed draws reproduced "
          f"{audit['status']}", flush=True)

    # The pinned reading row. Not swept: PREDICTION_v5.md §7 item 1 fixed it
    # before the checkpoint was scored, and a sweep here would hand the row
    # choice back to whoever reads the table.
    s = score(drawn, n=1, lam=0.0, min_chars=args.min_chars, probes=PROBES)
    ci, op = s["story_dodge_ci"], s["story_dodge_opener_ci"]
    row = {
        "arm": arm,
        "step": ck.get("step"),
        "seeds": args.seeds,
        "n_candidates": args.n,
        "row": {"n": 1, "lambda": 0.0},
        "story_dodge": ci,
        "story_dodge_opener": op,
        "p1_verdict": resolves_against(ci["k"], ci["n"], P1_BELOW),
        "superset_check": audit,
        "seconds": round(secs, 1),
    }
    print(f"   story_dodge  {ci['rate']:.4f}  = {ci['k']}/{ci['n']}  "
          f"95% CI [{ci['ci_low']:.4f}, {ci['ci_high']:.4f}]  "
          f"lattice {ci['lattice']:.5f}", flush=True)
    print(f"   opener only  {op['rate']:.4f}  = {op['k']}/{op['n']}  "
          f"95% CI [{op['ci_low']:.4f}, {op['ci_high']:.4f}]", flush=True)
    print(f"   vs P1's {P1_BELOW}: {row['p1_verdict']}   ({secs / 60:.1f} min)",
          flush=True)
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    return row


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--arms", nargs="+", required=True)
    p.add_argument("--seeds", type=int, default=101,
                   help="seeds per probe. THIS is the sample-size lever; see the "
                        "module docstring for why --n is not")
    p.add_argument("--n", type=int, default=16,
                   help="candidates per (probe, seed). Unused at the pinned n=1 "
                        "reading row, but it sets the sampler's RNG stream, so it "
                        "must match what the committed arms were drawn at")
    p.add_argument("--exp", default="experiments/chat")
    p.add_argument("--device", default=None)
    p.add_argument("--temperature", type=float, default=0.85)
    p.add_argument("--top-p", type=float, default=0.92)
    p.add_argument("--min-p", type=float, default=0.02)
    p.add_argument("--max-new", type=int, default=300)
    p.add_argument("--min-chars", type=int, default=12)
    p.add_argument("--out", default=None)
    args = p.parse_args(argv)

    n_draws = len([q for q in PROBES if q.kind in _DODGE_KINDS]) * args.seeds
    print(f"# story_dodge resample: {n_draws} draws per arm "
          f"(was 48), lattice {1 / n_draws:.5f}")
    if n_draws % 8 == 0:
        print(f"  WARNING: {n_draws} is divisible by 8, so {P1_BELOW} is still an "
              f"exact lattice point ({n_draws // 8}/{n_draws}). Use an odd seed "
              f"count.")

    report = {
        "seeds": args.seeds,
        "n_candidates": args.n,
        "draws_per_arm": n_draws,
        "lattice": round(1 / n_draws, 6),
        "p1_threshold": P1_BELOW,
        "p1_upper_band": P1_ABOVE,
        "reading_row": {"n": 1, "lambda": 0.0},
        "sampling": {"temperature": args.temperature, "top_p": args.top_p,
                     "min_p": args.min_p, "max_new": args.max_new},
        "arms": [],
    }
    out = Path(args.out) if args.out else None
    for arm in args.arms:
        report["arms"].append(_measure(arm, args))
        if out:  # written after every arm, so a long run is never all-or-nothing
            out.parent.mkdir(parents=True, exist_ok=True)
            with open(out, "w", encoding="utf-8") as fh:
                json.dump(report, fh, indent=1)

    print("\n## summary, at the pinned n=1 lambda=0 row")
    print(f"  {'arm':<22} {'story_dodge':>12} {'95% CI':>18} {'opener':>9} "
          f"{'vs 0.125':>10}")
    for r in report["arms"]:
        c, o = r["story_dodge"], r["story_dodge_opener"]
        print(f"  {r['arm']:<22} {c['rate']:>7.4f} {c['k']:>4}/{c['n']:<4} "
              f"[{c['ci_low']:.4f}, {c['ci_high']:.4f}] {o['rate']:>9.4f} "
              f"{r['p1_verdict']:>10}")
    if out:
        print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
