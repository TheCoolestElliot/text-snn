"""Re-measure ANY probe subset at a resolution that can miss a threshold narrowly.

    # the story_dodge use case story_dodge_resample.py already covers, done here
    # generically instead:
    python scripts/chat/probe_resample.py --seeds 101 \\
        --arms chat-v3d-aligned --kind list --kind fact \\
        --out experiments/chat/_quality/story_dodge_generic.json

    # a probe subset picked by prompt substring, not by kind -- what this file
    # was actually built for: the nine topic probes docs/chat/README.md and
    # QUALITY.md have reported at zero every round
    python scripts/chat/probe_resample.py --seeds 301 \\
        --arms chat-v3d-aligned \\
        --probes dragon robot pirate snowman moon spider teacher bicycle train \\
        --out experiments/chat/_quality/longtail_topic_resample.json

WHY THIS FILE EXISTS
---------------------
`scripts/chat/story_dodge_resample.py` fixed one metric's resolution problem:
`story_dodge` is a proportion over 48 draws (12 probes x 4 seeds), quantised at
1/48, and `docs/chat/PREDICTION_v5.md`'s P1 landed exactly on a threshold that
sat exactly on that lattice. The fix -- raise the seed count until the
denominator no longer divides the threshold evenly, report a Wilson interval,
call it `unresolved` rather than force a verdict -- is general. The script that
implemented it was not: `_DODGE_KINDS = ("list", "fact")` and the P1 bands are
hard-coded, so re-using it for a different probe subset meant copying the file.

`docs/chat/QUALITY_v6.md` §7 named the missing piece directly: "A generic
resample tool (parallel to `story_dodge_resample.py`, which is hard-coded to
the `list`+`fact` kinds) does not exist." This is that tool. It generalises
`--arms` unchanged and replaces the hard-coded kind pair with `--kind`
(any `Probe.kind`, repeatable) and/or `--probes` (a subset by prompt
substring, case-insensitive) -- the latter is what a topicality question needs,
since `topic` is one kind covering sixteen very different-frequency prompts and
the question worth asking is about a NAMED subset of them, not the whole kind.

WHAT IT DOES NOT DO
--------------------
Same discipline as its ancestor: **it does not retrain and it does not
redefine anything.** Same checkpoints, same frozen probe battery
(`snnchat.quality.PROBES`), same sampler settings, same `(n=1, lambda=0)`
reading row every prior round has used. The only quantity that changes is how
many seeds the proportion is taken over.

TWO THINGS CARRIED OVER FROM THE ANCESTOR, BOTH LOAD-BEARING
--------------------------------------------------------------
1.  **`n` here means seeds, not candidates.** `quality.py --n` is candidates
    per (probe, seed); at the pinned `n=1` reading row the extra candidates are
    never looked at. Seeds is the only lever, `--n` exists only to keep the
    sampler's RNG stream matched to whatever the committed arm was drawn at.
2.  **`probe_index` pins each selected probe's position in the FULL battery**,
    not its position in the subset, so drawing nine of the sixteen topic probes
    does not silently become a different experiment from drawing all sixteen --
    each probe's sampling seed is `seed * 10_007 + its index in PROBES`
    regardless of what else was selected alongside it
    (`snnchat.quality.collect`'s own docstring). Seeds 0-3 of any resample must
    reproduce the committed draws for the probes it shares with a prior round's
    file, character for character -- checked below, and it raises rather than
    silently reporting a different experiment as a superset.

BERNOULLI ONLY
--------------
`snnchat.quality.rate_ci` (Wilson) assumes a 0/1 outcome per draw. Every probe
in the battery scores that way EXCEPT `kind == "list"` probes with `want > 1`
("name three animals"), which take fractional values in {0, 1/3, 2/3, 1}
(`docs/chat/CONVENTIONS.md` §2). This script refuses a selection that includes
one of those -- report them with the sample standard error instead, by hand,
the way `CONVENTIONS.md` §2 says to. `story_dodge`'s own `list`+`fact`
combination is safe because it scores whether the reply IS a narrative, a
property of the text and not of the lexicon-hit fraction, so it stays 0/1 even
though `list` probes are mixed in -- this script's Bernoulli check is on the
DODGE-style use of `list`, and callers asking for `--kind list` directly (the
literal instruction-following hit rate) will be refused, correctly.

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


def _subset(kinds: list[str] | None, substrings: list[str] | None) -> tuple[tuple, tuple[int, ...]]:
    """The probes this call is about, with their indices in the full battery.

    `kinds` and `substrings` combine as AND when both are given: a probe must
    match one of `kinds` (if any given) and contain one of `substrings` (if any
    given, matched case-insensitively against `Probe.prompt`). At least one of
    the two must be given -- an unrestricted resample of all 37 probes is what
    `quality.py`'s own `--seeds` flag already does more cheaply, since it need
    not carry the superset-reproduction machinery this script pays for.
    """
    if not kinds and not substrings:
        raise SystemExit("pass --kind and/or --probes -- an unrestricted resample "
                          "of the whole battery is scripts/chat/quality.py's job")
    subs = [s.lower() for s in (substrings or [])]
    idx = []
    for i, p in enumerate(PROBES):
        if kinds and p.kind not in kinds:
            continue
        if subs and not any(s in p.prompt.lower() for s in subs):
            continue
        idx.append(i)
    if not idx:
        raise SystemExit(f"no probe matches kind={kinds} probes={substrings}")
    bad = [PROBES[i].prompt for i in idx if PROBES[i].want > 1]
    if bad:
        raise SystemExit(
            f"{len(bad)} selected probe(s) score fractionally (want > 1), not "
            f"0/1, so a Wilson interval does not apply: {bad}. Per "
            f"CONVENTIONS.md §2, report those with the sample standard error "
            f"instead -- this script only does Bernoulli probes."
        )
    return tuple(PROBES[i] for i in idx), tuple(idx)


def _check_superset(drawn: dict, committed: Path, prompts: set[str]) -> dict:
    """Assert the new draws reproduce any committed ones for these prompts.

    Returns the audit record. Raises if the file exists, shares at least one
    (prompt, seed) with this draw, and disagrees on it -- a resample that
    quietly re-seeded is not a larger version of the measurement it claims to
    extend. If the committed file has none of these prompts at all (a probe
    subset no prior round happened to draw), that is reported, not raised on --
    there is nothing to be a superset of yet.
    """
    if not committed.exists():
        return {"file": str(committed), "checked": 0, "status": "absent"}
    with open(committed, encoding="utf-8") as fh:
        old_draws = json.load(fh)["draws"]["draws"]
    old = {(d["prompt"], d["seed"]): d for d in old_draws if d["prompt"] in prompts}
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
    return {"file": str(committed), "checked": checked,
            "status": "identical" if checked else "no shared prompts"}


def _measure(arm: str, probes: tuple, idx: tuple[int, ...], args) -> dict:
    ckpt = Path(args.exp) / arm / "ckpt_best.pt"
    if not ckpt.exists():
        raise SystemExit(f"no checkpoint at {ckpt}")
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg, ck = load_chat_checkpoint(str(ckpt), device=device)
    params = SamplingParams(temperature=args.temperature, top_p=args.top_p,
                            min_p=args.min_p, max_new=args.max_new)

    print(f"\n## {arm}  (step {ck.get('step')})", flush=True)
    print(f"   {len(probes)} probes x {args.seeds} seeds x {args.n} candidates "
          f"= {len(probes) * args.seeds} draws", flush=True)
    t0 = time.perf_counter()
    drawn = collect(model, probes=probes, seeds=tuple(range(args.seeds)),
                    n=args.n, params=params, device=device, progress=False,
                    probe_index=idx)
    secs = time.perf_counter() - t0

    prompts = {p.prompt for p in probes}
    audit = _check_superset(drawn, Path(args.exp) / "_quality" / f"{arm}.json", prompts)
    print(f"   superset check: {audit['checked']} committed draws reproduced "
          f"{audit['status']}", flush=True)

    # The pinned reading row. Not swept, same reason story_dodge_resample.py
    # gives: PREDICTION_v5.md §7 item 1 fixed it before any checkpoint was
    # scored under it, and a sweep here would hand the row choice back to
    # whoever reads the table.
    s = score(drawn, n=1, lam=0.0, min_chars=args.min_chars, probes=PROBES)
    by_prompt = {pick["prompt"]: pick for pick in s["picks"]}
    hits = [pick["hit"] for pick in s["picks"] if pick["prompt"] in prompts]
    k, n = int(round(sum(hits))), len(hits)
    ci = rate_ci(k, n)

    per_probe = {}
    for p in probes:
        p_hits = [pick["hit"] for pick in s["picks"] if pick["prompt"] == p.prompt]
        pk, pn = int(round(sum(p_hits))), len(p_hits)
        per_probe[p.prompt] = rate_ci(pk, pn)

    row = {
        "arm": arm,
        "step": ck.get("step"),
        "seeds": args.seeds,
        "n_candidates": args.n,
        "row": {"n": 1, "lambda": 0.0},
        "probes": [p.prompt for p in probes],
        "pooled": ci,
        "per_probe": per_probe,
        "superset_check": audit,
        "seconds": round(secs, 1),
    }
    if args.threshold is not None:
        row["verdict"] = resolves_against(ci["k"], ci["n"], args.threshold)

    print(f"   pooled  {ci['rate']:.4f}  = {ci['k']}/{ci['n']}  "
          f"95% CI [{ci['ci_low']:.4f}, {ci['ci_high']:.4f}]  "
          f"lattice {ci['lattice']:.5f}", flush=True)
    if args.threshold is not None:
        print(f"   vs {args.threshold}: {row['verdict']}   ({secs / 60:.1f} min)",
              flush=True)
    for prompt, pci in per_probe.items():
        print(f"     {prompt[:50]:<50s} {pci['rate']:.4f} = {pci['k']}/{pci['n']} "
              f"[{pci['ci_low']:.4f}, {pci['ci_high']:.4f}]", flush=True)
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    return row


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--arms", nargs="+", required=True)
    p.add_argument("--kind", action="append", default=None,
                   help="restrict to this Probe.kind; repeatable")
    p.add_argument("--probes", nargs="+", default=None,
                   help="restrict to probes whose prompt contains any of these "
                        "substrings (case-insensitive)")
    p.add_argument("--threshold", type=float, default=None,
                   help="optional pre-registered bar; verdict is below/above/"
                        "unresolved via CONVENTIONS.md's rule. Omit to just "
                        "report the interval with no verdict")
    p.add_argument("--seeds", type=int, default=101,
                   help="seeds per probe. THIS is the sample-size lever; see "
                        "the module docstring for why --n is not")
    p.add_argument("--n", type=int, default=16,
                   help="candidates per (probe, seed). Unused at the pinned "
                        "n=1 reading row, but it sets the sampler's RNG "
                        "stream, so it must match what the committed arms "
                        "were drawn at")
    p.add_argument("--exp", default="experiments/chat")
    p.add_argument("--device", default=None)
    p.add_argument("--temperature", type=float, default=0.85)
    p.add_argument("--top-p", type=float, default=0.92)
    p.add_argument("--min-p", type=float, default=0.02)
    p.add_argument("--max-new", type=int, default=300)
    p.add_argument("--min-chars", type=int, default=12)
    p.add_argument("--out", default=None)
    args = p.parse_args(argv)

    probes, idx = _subset(args.kind, args.probes)
    n_draws = len(probes) * args.seeds
    print(f"# probe resample: {len(probes)} probes x {args.seeds} seeds = "
          f"{n_draws} draws per arm, lattice {1 / n_draws:.6f}")
    for p in probes:
        print(f"    [{PROBES.index(p) if p in PROBES else '?'}] {p.kind:8s} {p.prompt}")
    if args.threshold is not None:
        lattice_hit = round(args.threshold * n_draws)
        if abs(args.threshold * n_draws - lattice_hit) < 1e-9:
            print(f"  WARNING: {args.threshold} * {n_draws} = {lattice_hit} is an "
                  f"integer -- the threshold sits exactly on this lattice. Per "
                  f"CONVENTIONS.md §4 rule 2, pick a seed count that moves it off, "
                  f"or the arm can only hit the bar exactly, not miss it narrowly.")

    report = {
        "seeds": args.seeds,
        "n_candidates": args.n,
        "kind_filter": args.kind,
        "probe_filter": args.probes,
        "threshold": args.threshold,
        "draws_per_arm": n_draws,
        "lattice": round(1 / n_draws, 6),
        "reading_row": {"n": 1, "lambda": 0.0},
        "sampling": {"temperature": args.temperature, "top_p": args.top_p,
                     "min_p": args.min_p, "max_new": args.max_new},
        "arms": [],
    }
    out = Path(args.out) if args.out else None
    for arm in args.arms:
        report["arms"].append(_measure(arm, probes, idx, args))
        if out:  # written after every arm, so a long run is never all-or-nothing
            out.parent.mkdir(parents=True, exist_ok=True)
            with open(out, "w", encoding="utf-8") as fh:
                json.dump(report, fh, indent=1)

    print("\n## summary, at the pinned n=1 lambda=0 row")
    header = f"  {'arm':<22} {'pooled':>10} {'95% CI':>18}"
    if args.threshold is not None:
        header += f" {'vs ' + str(args.threshold):>12}"
    print(header)
    for r in report["arms"]:
        c = r["pooled"]
        line = (f"  {r['arm']:<22} {c['rate']:>6.4f} {c['k']:>3}/{c['n']:<4} "
                f"[{c['ci_low']:.4f}, {c['ci_high']:.4f}]")
        if args.threshold is not None:
            line += f" {r['verdict']:>12}"
        print(line)
    if out:
        print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
