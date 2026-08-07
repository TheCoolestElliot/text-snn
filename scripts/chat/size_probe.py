"""Pick the training configuration by measuring, at matched wall clock.

    python scripts/chat/size_probe.py --minutes 5

The question this answers is not "which model is best" -- it is "which model is
best *per hour*", which is a different ordering and the one that matters when
the budget is wall clock rather than steps. A bigger network sees proportionally
less data in the same time, and on this box the two effects are close enough in
magnitude that the answer is not predictable from parameter counts.

Every arm runs at CONSTANT learning rate. A cosine schedule would put each arm at
a different point on its own decay when the timer stops, which would make the
comparison partly a comparison of schedules.

The arms run strictly sequentially and nothing else may touch the GPU while they
do -- `03_phase3_candidates.md` §8's standard, and it applies here even though
none of these numbers is a reported figure, because a contended arm looks exactly
like a slow one.

This is a selection tool, not an experiment: n=1 per arm, one seed, no error
bars, and the differences it resolves are the large ones. Its output picks a
config and is not evidence of anything about the architecture.
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

from snnchat.build_corpus import DEFAULT_MIX  # noqa: E402
from snnchat.model import ChatConfig  # noqa: E402
from snnchat.train import ChatTrainer  # noqa: E402

ARMS = [
    ("small_K3",  dict(d_model=768,  n_layers=3, batch_size=256, seq_len=256, dtype="fp32")),
    ("mid_K4",    dict(d_model=1024, n_layers=4, batch_size=160, seq_len=256, dtype="fp32")),
    ("large_K3",  dict(d_model=1536, n_layers=3, batch_size=128, seq_len=256, dtype="fp32")),
    ("mid_K4_bf", dict(d_model=1024, n_layers=4, batch_size=160, seq_len=256, dtype="bf16")),
]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--minutes", type=float, default=5.0)
    p.add_argument("--lr", type=float, default=2.0e-3)
    p.add_argument("--out", default="experiments/chat/_probe/results.json")
    args = p.parse_args(argv)

    results = []
    for name, spec in ARMS:
        print(f"\n=== {name}: {spec} ===", flush=True)
        cfg = ChatConfig(
            run_name=f"_probe/{name}", lr=args.lr, lr_schedule="constant",
            warmup_steps=200, max_steps=10**9,
            eval_every=0, sample_every=0, ckpt_every=10**9, log_every=200,
            **spec,
        )
        cfg.mix = dict(DEFAULT_MIX)
        t0 = time.perf_counter()
        trainer = ChatTrainer(cfg)
        try:
            trainer.train(wall_clock_budget_s=args.minutes * 60.0)
        except Exception as exc:
            print(f"  {name} FAILED: {type(exc).__name__}: {exc}", flush=True)
            results.append({"arm": name, "error": str(exc), **spec})
            continue
        val = trainer.evaluate()
        elapsed = time.perf_counter() - t0
        row = {
            "arm": name,
            "params": trainer.n_params,
            "steps": trainer.global_step,
            "chars": trainer.global_step * cfg.batch_size * cfg.seq_len,
            "elapsed_s": round(elapsed, 1),
            "val": {k: round(v, 4) for k, v in val.items()},
            **spec,
        }
        results.append(row)
        print(f"  {name}: {row['steps']:,} steps, {row['chars'] / 1e6:,.0f} Mchars, "
              f"weighted val bpc {val.get('weighted', float('nan')):.4f}", flush=True)
        del trainer
        import torch
        torch.cuda.empty_cache()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print("\n=== ranking (lower bpc is better, matched wall clock) ===")
    ranked = sorted(
        (r for r in results if "val" in r), key=lambda r: r["val"].get("weighted", 9e9)
    )
    for r in ranked:
        print(f"  {r['arm']:<12} {r['params']:>10,} params  {r['chars'] / 1e6:>7,.0f} Mchars  "
              f"weighted {r['val'].get('weighted', float('nan')):.4f}  "
              f"stories {r['val'].get('tinystories', float('nan')):.4f}  "
              f"soda {r['val'].get('soda', float('nan')):.4f}")
    if ranked:
        print(f"\nwinner: {ranked[0]['arm']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
