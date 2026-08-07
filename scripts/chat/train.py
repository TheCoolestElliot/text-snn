"""Train the spiking chat model.

    python scripts/chat/train.py --run-name chat-v1 --max-steps 60000

Restartable by design, because the run outlives the process that starts it:
re-running the identical command resumes from `ckpt_last.pt` unless
`--no-resume` is passed. The sampler is a pure function of `(seed, step)`, so a
resumed run draws exactly the batches an uninterrupted one would have -- there is
no data-order state to lose.

    --budget-minutes N     stop cleanly after N minutes, checkpoint, and exit 0

is the flag that makes it usable under a supervisor with a wall-clock budget.
The learning-rate schedule still runs against `--max-steps`, so a run stopped by
the budget is a run stopped part way down its cosine; `--lr-final-frac` keeps
that from being catastrophic (see `ChatTrainer._lr_at`).

Not part of the research protocol. Writes only under `experiments/chat/`.
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO / "src") not in sys.path:
    sys.path.insert(0, str(_REPO / "src"))

from snnchat.build_corpus import DEFAULT_MIX  # noqa: E402
from snnchat.model import ChatConfig, param_count  # noqa: E402
from snnchat.train import ChatTrainer  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    defaults = ChatConfig()
    for f in dataclasses.fields(ChatConfig):
        if f.name in ("mix", "vocab_size", "vocab_version"):
            continue
        flag = "--" + f.name.replace("_", "-")
        default = getattr(defaults, f.name)
        if isinstance(default, bool):
            p.add_argument(flag, dest=f.name, action=argparse.BooleanOptionalAction,
                           default=default)
        else:
            p.add_argument(flag, dest=f.name, type=type(default), default=default)
    p.add_argument("--mix", nargs="*", default=None, metavar="NAME=WEIGHT",
                   help=f"override the corpus mixture; default {DEFAULT_MIX}")
    p.add_argument("--budget-minutes", type=float, default=0.0,
                   help="stop cleanly after this many minutes (0 = run to max-steps)")
    p.add_argument("--no-resume", action="store_true",
                   help="ignore an existing ckpt_last.pt and start from scratch")
    p.add_argument("--init-from", default=None, metavar="CKPT",
                   help="start a NEW run from another checkpoint's weights "
                        "(optimiser and step start fresh). This is how the "
                        "chat-focused anneal is run; see docs/chat/README.md")
    p.add_argument("--dry-run", action="store_true",
                   help="print the size and the mix, build nothing")
    return p


def _parse_mix(spec: list[str] | None) -> dict[str, float]:
    if not spec:
        return dict(DEFAULT_MIX)
    out: dict[str, float] = {}
    for item in spec:
        name, _, value = item.partition("=")
        if not value:
            raise SystemExit(f"--mix wants NAME=WEIGHT, got {item!r}")
        out[name] = float(value)
    return out


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    fields = {f.name for f in dataclasses.fields(ChatConfig)}
    cfg = ChatConfig(**{k: v for k, v in vars(args).items() if k in fields})
    cfg.mix = _parse_mix(args.mix)

    from snnchat.tokenizer import ChatTokenizer
    v = ChatTokenizer().vocab_size
    n = param_count(v, cfg.d_model, cfg.n_layers, cfg.arch)
    chars = cfg.max_steps * cfg.batch_size * cfg.seq_len
    print(
        f"chat run {cfg.run_name!r}\n"
        f"  arch {cfg.arch}, d={cfg.d_model}, K={cfg.n_layers}, V={v} "
        f"-> {n:,} parameters\n"
        f"  B={cfg.batch_size}, L={cfg.seq_len}, {cfg.max_steps:,} steps "
        f"= {chars / 1e9:.2f} G characters\n"
        f"  mix {cfg.mix}",
        flush=True,
    )
    if args.dry_run:
        return 0

    trainer = ChatTrainer(cfg)
    last = trainer.run_dir / "ckpt_last.pt"
    if last.exists() and not args.no_resume:
        # Resume wins over --init-from: a run that was interrupted must carry on
        # from where it stopped, not silently restart from its seed weights.
        trainer.load_checkpoint(last)
    else:
        if last.exists():
            print(f"  ignoring {last} (--no-resume)", flush=True)
        if args.init_from:
            trainer.init_from(args.init_from)

    trainer.train(wall_clock_budget_s=args.budget_minutes * 60.0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
