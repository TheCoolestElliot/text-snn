"""Score a chat checkpoint's replies mechanically, and sweep decoding settings.

    python scripts/chat/quality.py --ckpt experiments/chat/chat-v2-anneal/ckpt_best.pt

What it does, in order:

1.  **Prompt dependence** on the held-out splits: the bits per character of a
    real reply behind its own user turn, against the same reply behind somebody
    else's. No sampler, no lexicon, no seed. This is the number to believe.
2.  **Draws** N candidate replies for every probe at several seeds, recording
    each candidate's log-probability under the real prompt and under an empty
    one.
3.  **Sweeps** reranking settings over those saved draws. Every setting is
    scored on the identical candidate pool, so a difference between two rows of
    the table is a difference in selection, not in luck.

`--out results.json` writes everything, including the winning reply for every
(probe, seed), so a table in `docs/chat/QUALITY.md` can be traced to the text it
came from.

Not part of the research protocol. n=1 per arm, no control arms, and the probe
set is a judgement -- see the module docstring of `snnchat.quality`.
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

from snnchat.data import ChatCorpus  # noqa: E402
from snnchat.generate import SamplingParams, load_chat_checkpoint  # noqa: E402
from snnchat.quality import PROBES, collect, prompt_dependence, score  # noqa: E402

#: The reranking grid. `n=1` is the current shipped behaviour and is the row
#: every other row is compared against; `lambda=0` at n>1 isolates "best of N by
#: likelihood" from "best of N by mutual information", which are different
#: interventions and are worth being able to tell apart.
#:
#: lambda runs past 1.0 deliberately. At 1.0 the score is pointwise mutual
#: information and is the natural stopping point in theory; whether the best
#: setting on THIS model is below or above it is a measurement, and a grid that
#: stops at the theoretically tidy value cannot report that it was exceeded.
DEFAULT_GRID = [(1, 0.0)] + [
    (n, lam)
    for lam in (0.0, 0.3, 0.6, 0.8, 1.0, 1.25)
    for n in (4, 8, 16)
]


def _sweep(drawn: dict, args, *, probes=None) -> list[dict]:
    print(f"  {'n':>3} {'lam':>5} | {'topic':>6} {'list':>6} {'strict':>6} "
          f"{'social':>6} {'ident':>6} {'fact':>6} | {'fallbk':>6} {'closed':>6} "
          f"{'chars':>6} | {'HEAD':>6}")
    rows = []
    for n, lam in DEFAULT_GRID:
        if n > drawn["n_candidates"]:
            continue
        s = score(drawn, n=n, lam=lam, min_chars=args.min_chars, probes=probes)
        rows.append(s)
        k = s["by_kind"]
        print(f"  {n:>3} {lam:>5.2f} | {k.get('topic', 0):6.3f} {k.get('list', 0):6.3f} "
              f"{s['list_strict']:6.3f} {k.get('social', 0):6.3f} "
              f"{k.get('identity', 0):6.3f} {k.get('fact', 0):6.3f} | "
              f"{s['fallback_rate']:6.3f} {s['closed_rate']:6.3f} {s['mean_chars']:6.1f} "
              f"| {s['headline']:6.3f}", flush=True)
    return rows


def _store(report: dict, rows: list[dict]) -> None:
    best = max(rows, key=lambda r: r["headline"])
    report["sweep"] = [{k: v for k, v in r.items() if k != "picks"} for r in rows]
    report["best"] = {k: v for k, v in best.items() if k != "picks"}
    report["best_picks"] = best["picks"]
    report["baseline_picks"] = next(
        (r["picks"] for r in rows if r["n"] == 1), rows[0]["picks"]
    )
    print(f"\n  best: n={best['n']} lambda={best['lambda']:g}  "
          f"headline {best['headline']:.3f}")


def _rescore(path: Path, args) -> int:
    with open(path, encoding="utf-8") as fh:
        blob = json.load(fh)
    report, drawn = blob["report"], blob["draws"]
    print(f"# rescoring {report['label']} under the current probe definitions")
    before = report.get("best", {}).get("headline")
    rows = _sweep(drawn, args, probes=PROBES)
    _store(report, rows)
    report["rescored"] = True
    if before is not None:
        print(f"  (headline at the previous best was {before:.3f})")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump({"report": report, "draws": drawn}, fh, indent=1)
    print(f"rewrote {path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", default=None,
                   help="required unless --rescore, which reads saved draws")
    p.add_argument("--device", default=None)
    p.add_argument("--data-dir", default="data/chat")
    p.add_argument("--n", type=int, default=16, help="candidates drawn per probe/seed")
    p.add_argument("--seeds", type=int, default=4)
    p.add_argument("--temperature", type=float, default=0.85)
    p.add_argument("--top-p", type=float, default=0.92)
    p.add_argument("--min-p", type=float, default=0.02)
    p.add_argument("--temperature-spread", type=float, default=0.0,
                   help="propose the candidates over a range of temperatures "
                        "rather than all at --temperature. Changes the DRAWS, so "
                        "it is a separate run and a separate output file, not a "
                        "row of the sweep")
    p.add_argument("--max-new", type=int, default=300)
    p.add_argument("--min-chars", type=int, default=12)
    p.add_argument("--null", default="empty_user", choices=("empty_user", "bot_only"))
    p.add_argument("--pairs", type=int, default=128, help="held-out pairs per source")
    p.add_argument("--no-dependence", action="store_true")
    p.add_argument("--out", default=None)
    p.add_argument("--label", default=None, help="name for this arm in the report")
    p.add_argument("--rescore", default=None, metavar="RESULTS.JSON",
                   help="re-run the sweep over an existing file's saved draws "
                        "under the CURRENT probe definitions, and rewrite it. No "
                        "GPU, no regeneration -- this is how a correction to the "
                        "scorer is applied to every arm equally, including arms "
                        "drawn before the correction existed")
    args = p.parse_args(argv)

    if args.rescore:
        return _rescore(Path(args.rescore), args)
    if not args.ckpt:
        raise SystemExit("--ckpt is required (or --rescore an existing result file)")

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg, ck = load_chat_checkpoint(args.ckpt, device=device)
    if ck.get("kind") != "snnchat":
        raise SystemExit(f"{args.ckpt} is a research checkpoint; it has no chat format.")
    label = args.label or Path(args.ckpt).parent.name

    print(f"# {label}  ({args.ckpt}, step {ck.get('step')})", flush=True)
    report: dict = {
        "label": label,
        "ckpt": str(args.ckpt),
        "step": ck.get("step"),
        "params": sum(q.numel() for q in model.parameters()),
    }

    t0 = time.perf_counter()
    if not args.no_dependence:
        corpus = ChatCorpus(args.data_dir, vocab_version=cfg.vocab_version)
        dep = prompt_dependence(model, corpus, per_source=args.pairs, device=device)
        report["prompt_dependence"] = dep
        print("\n## prompt dependence (held-out, no sampler)")
        print(f"  {'source':<12} {'matched':>9} {'shuffled':>9} {'delta':>9}")
        for name, row in dep["per_source"].items():
            print(f"  {name:<12} {row['bpc_matched']:9.4f} {row['bpc_shuffled']:9.4f} "
                  f"{row['delta']:9.4f}")
        if "delta" in dep:
            print(f"  {'ALL':<12} {dep['bpc_matched']:9.4f} {dep['bpc_shuffled']:9.4f} "
                  f"{dep['delta']:9.4f}")
        if "check" in dep:
            c = dep["check"]
            print(f"  (scorer check: batched {c['batched']:.5f} vs sequential "
                  f"{c['sequential']:.5f} nats, relative {c['relative']:.1e})")

    params = SamplingParams(temperature=args.temperature, top_p=args.top_p,
                            min_p=args.min_p, max_new=args.max_new)
    spread = f", temperature spread {args.temperature_spread:g}" if args.temperature_spread else ""
    print(f"\n## drawing {args.n} candidates x {args.seeds} seeds x {len(PROBES)} probes{spread}")
    drawn = collect(model, seeds=tuple(range(args.seeds)), n=args.n,
                    params=params, null=args.null, device=device,
                    temperature_spread=args.temperature_spread)
    report["temperature_spread"] = args.temperature_spread
    draw_s = time.perf_counter() - t0
    report["collect_seconds"] = round(draw_s, 1)

    print("\n## reranking sweep")
    rows = _sweep(drawn, args)
    _store(report, rows)
    report["seconds"] = round(time.perf_counter() - t0, 1)

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as fh:
            json.dump({"report": report, "draws": drawn}, fh, indent=1)
        print(f"\nwrote {out} ({out.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
