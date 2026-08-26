"""Does this working tree still reproduce a committed run's held-out bpc?

NOT PART OF THE RESEARCH PROTOCOL. `snnchat` is outside the Phase-1..5 study.
Trains nothing, writes nothing outside `--out`.

WHY THIS EXISTS
---------------
Before spending GPU-hours on a comparison whose control is a *committed* number,
something has to establish that the tree the new arm will train on still computes
that number. `docs/reports/04_phase4_interim.md` §13.5 is the standing example of
what happens when it does not: a fix to `_clip_grad_norm_fp64` that changed no
returned norm still moved a finished run by +1.97e-03 bpc, `EXP_014`'s G1 gate
failed, and decision #10 exists because of it.

This re-runs `snnchat.train.Trainer.evaluate`'s protocol -- fresh state, per
source, `eval_batch_size` and `eval_batches` read from the run's own
`config.json` -- against the `val` block in the run's `summary.json`, and reports
the residual per source.

It is a *reproduction* check, not an equivalence proof. It exercises the data
path, the tokenizer, the model construction and the forward. It says nothing
about the backward, and a run whose eval reproduces can still train differently.

    python scripts/chat/reproduce_eval.py --run chat-v3d-aligned
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from snnchat.data import ChatCorpus, MixtureSampler, SequentialEval  # noqa: E402
from snnchat.train import bits_per_char  # noqa: E402
from snnchat.model import ChatConfig, build_chat_model  # noqa: E402

CHAT = REPO / "experiments" / "chat"


def evaluate(run: str, ckpt_name: str, device: str,
             eval_batches: int | None = None) -> dict:
    run_dir = CHAT / run
    raw = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    known = {f.name for f in dataclasses.fields(ChatConfig)}
    cfg = ChatConfig(**{k: v for k, v in raw.items() if k in known})
    cfg.device = device
    cfg.spread_tau = False        # the spread is baked into the checkpoint
    if eval_batches:
        # `eval_batches` decides HOW MUCH of each held-out tail is scored, so it
        # changes the estimand and not just the cost. Overridable because a run
        # scored at 40 and one scored at 30 are not the same number, and a guard
        # band derived from 30-batch replicates has to be checked at 30.
        cfg.eval_batches = int(eval_batches)

    model = build_chat_model(cfg)
    ck = torch.load(run_dir / ckpt_name, map_location=device, weights_only=False)
    model.load_state_dict(ck["model"])
    model.eval()

    corpus = ChatCorpus(cfg.data_dir)
    sampler = MixtureSampler(corpus, cfg.mix, cfg.batch_size, cfg.seq_len, cfg.seed,
                             align_frac=cfg.align_frac,
                             align_lookahead=cfg.align_lookahead)
    eval_bs = cfg.eval_batch_size or cfg.batch_size

    out: dict[str, float] = {}
    with torch.no_grad():
        for name in sampler.names:
            try:
                it = SequentialEval(corpus, name, eval_bs, cfg.seq_len)
            except FileNotFoundError:
                continue
            if it.n_batches() == 0:
                windows = max(len(it.data) - 1, 0) // cfg.seq_len
                if windows == 0:
                    continue
                it = SequentialEval(corpus, name, windows, cfg.seq_len)
            nats, chars = 0.0, 0
            for i, (x, y) in enumerate(it):
                if i >= cfg.eval_batches:
                    break
                x, y = x.to(device), y.to(device)
                logits, _, _ = model(x, None)
                nats += float(F.cross_entropy(
                    logits.reshape(-1, cfg.vocab_size).float(), y.reshape(-1),
                    reduction="sum"))
                chars += y.numel()
            if chars:
                out[name] = bits_per_char(nats / chars)
    if out:
        num = sum(out[n] * w for n, w in zip(sampler.names, sampler.weights) if n in out)
        den = sum(w for n, w in zip(sampler.names, sampler.weights) if n in out)
        out["weighted"] = num / max(den, 1e-9)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run", default="chat-v3d-aligned")
    ap.add_argument("--ckpt", default="ckpt_best.pt")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--eval-batches", type=int, default=0,
                    help="override cfg.eval_batches; 0 = the run's own value")
    ap.add_argument("--baseline", default="",
                    help="run whose summary.json to compare against "
                         "(default: --run itself)")
    ap.add_argument("--out", default="")
    args = ap.parse_args(argv)

    got = evaluate(args.run, args.ckpt, args.device, args.eval_batches or None)
    base = args.baseline or args.run
    committed = json.loads(
        (CHAT / base / "summary.json").read_text(encoding="utf-8"))["val"]

    print(f"{args.run}/{args.ckpt} -- this tree vs the committed summary.json\n")
    print(f"  {'source':16} {'committed':>12} {'this tree':>12} {'residual':>12}")
    worst = 0.0
    for k in sorted(set(committed) | set(got)):
        c, g = committed.get(k), got.get(k)
        if c is None or g is None:
            print(f"  {k:16} {str(c):>12} {str(g):>12} {'MISSING':>12}")
            continue
        r = g - c
        worst = max(worst, abs(r))
        star = "  <-- weighted" if k == "weighted" else ""
        print(f"  {k:16} {c:>12.9f} {g:>12.9f} {r:>+12.2e}{star}")
    print(f"\n  worst absolute residual: {worst:.3e}")
    # 0.000890 is the committed n=4 training-seed sd of this recipe's weighted
    # bpc. A reproduction residual is a DIFFERENT quantity -- same weights, same
    # data -- and should be orders below it, not merely inside it.
    print(f"  committed SD_seed for this recipe (n=4): 8.90e-04 "
          f"-- residual is {8.90e-04 / max(worst, 1e-15):.0f}x smaller"
          if worst else "  exact")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps({
            "script": "scripts/chat/reproduce_eval.py",
            "not_a_research_artifact": True,
            "run": args.run, "checkpoint": args.ckpt,
            "baseline_run": base, "eval_batches_override": args.eval_batches,
            "committed": committed, "this_tree": got,
            "worst_abs_residual": worst,
        }, indent=1), encoding="utf-8")
        print(f"  wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
