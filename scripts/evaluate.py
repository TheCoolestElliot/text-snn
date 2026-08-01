"""Score a checkpoint under BOTH evaluation protocols.

    python scripts/evaluate.py --ckpt experiments/runs/<run>/ckpt_best.pt --split test

The configuration comes from the checkpoint, not from the command line, so a
number produced here cannot silently be a number for a different model.  Only
things that must not change the model -- which split, how many windows, which
device, batch size, and the fused/eager kernel path -- are overridable, and
every override is recorded in the emitted JSON.

Both protocols are always run.  There is no flag to run only one, deliberately:
01_reconnaissance.md 4.3 requires both for any reported figure, and a flag to
skip one is a flag someone will eventually use.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

import torch

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO / "src") not in sys.path:
    sys.path.insert(0, str(_REPO / "src"))

from snn.config import Config, config_hash, seed_everything  # noqa: E402
from snn.data import Corpus  # noqa: E402
from snn.evaluate import evaluate  # noqa: E402
from snn.model import build_model, count_params  # noqa: E402

_PROTOCOLS = ("fresh", "carried")


def _config_from_checkpoint(ck: dict) -> Config:
    raw = ck.get("config")
    if not raw:
        raise SystemExit("checkpoint has no 'config' block; cannot rebuild the model")
    known = {f.name for f in dataclasses.fields(Config)}
    unknown = sorted(set(raw) - known)
    if unknown:
        print(f"warning: ignoring unknown config fields {unknown}", file=sys.stderr)
    return Config(**{k: v for k, v in raw.items() if k in known})


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", required=True, help="path to a .pt written by Trainer")
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--batch-size", type=int, default=0,
                   help="0 = use the checkpoint's batch_size")
    p.add_argument("--max-windows", type=int, default=0, help="0 = whole split")
    p.add_argument("--device", default=None)
    p.add_argument("--data-dir", default=None)
    p.add_argument("--no-fused", action="store_true",
                   help="force the eager reference LIF scan")
    p.add_argument("--out", default=None,
                   help="JSON output path (default: <ckpt-dir>/eval_<split>.json)")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    ckpt_path = Path(args.ckpt)
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    cfg = _config_from_checkpoint(ck)
    overrides: dict = {}
    if args.batch_size:
        overrides["batch_size"] = args.batch_size
    if args.device:
        overrides["device"] = args.device
    if args.data_dir:
        overrides["data_dir"] = args.data_dir
    if args.no_fused:
        overrides["fused"] = False
    for k, v in overrides.items():
        setattr(cfg, k, v)
    cfg.eval_max_windows = args.max_windows

    seed_everything(cfg.seed, cfg.deterministic)
    device = torch.device(cfg.device)
    corpus = Corpus(cfg.corpus, cfg.data_dir)
    if int(cfg.vocab_size) != int(corpus.vocab_size):
        raise SystemExit(
            f"checkpoint vocab_size={cfg.vocab_size} but corpus reports "
            f"{corpus.vocab_size}; refusing to evaluate a mismatched model"
        )

    model = build_model(cfg).to(device)
    model.load_state_dict(ck["model"])
    model.eval()

    results = {p: evaluate(model, corpus, args.split, cfg, p,
                           max_windows=args.max_windows)
               for p in _PROTOCOLS}

    report = {
        "checkpoint": str(ckpt_path),
        "checkpoint_step": ck.get("step"),
        "checkpoint_config_hash": ck.get("config_hash"),
        "eval_config_hash": config_hash(cfg),
        "overrides": overrides,
        "split": args.split,
        "max_windows": args.max_windows,
        "batch_size": cfg.batch_size,
        "params": count_params(model),
        "results": results,
    }

    out_path = Path(args.out) if args.out else ckpt_path.parent / f"eval_{args.split}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)

    print(f"checkpoint  {ckpt_path}  (step {ck.get('step')})")
    print(f"params      {report['params']:,}")
    print(f"split       {args.split}")
    print()
    print("| protocol | bpc | nats/char | chars scored | windows | firing rate |")
    print("|---|---:|---:|---:|---:|---|")
    for p in _PROTOCOLS:
        r = results[p]
        nats_per_char = r["nats"] / r["n_chars"] if r["n_chars"] else float("nan")
        rates = ", ".join(f"{x:.4f}" for x in r["firing_rate"]) or "-"
        print(f"| {p} | {r['bpc']:.4f} | {nats_per_char:.4f} | "
              f"{r['n_chars']:,} | {r['n_windows']:,} | {rates} |")
    print()
    print(f"WROTE {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
