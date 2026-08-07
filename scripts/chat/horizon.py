"""How far back does the chat model's membrane actually reach?

    python scripts/chat/horizon.py --ckpt experiments/chat/chat-v1/ckpt_best.pt

The probe is `scripts/exp/001_memory_horizon.py`'s, unchanged in method and
re-implemented here only because the corpus and the vocabulary differ. Quoting
it, because the reason it is shaped this way is the reason to trust it:

    Every arm carries exactly one thing across time [...] All three zero it when
    `forward(idx, state=None)` is called. So "force the state to zero every k
    characters" needs no hook, no model edit, and no new code path in the model
    -- it is just `x.reshape(B * L // k, k)` run with `state=None`.

    [...] the probe cannot measure its own implementation. A hook that zeroed
    state mid-scan would be new code between the model and the number, and a bug
    in it would look exactly like a short memory horizon.

**F1, the self-check.** At `k = L` the reshape is the identity, so the probe must
reproduce the ordinary fresh-state bpc computed by a completely different code
path (`snnchat.train.ChatTrainer.evaluate`). That is asserted rather than
eyeballed. `snn-v2-research-protocol`'s standing rule -- make every probe
reproduce a number it did not produce -- is what caught a training run that had
silently imported a mutated kernel, and it costs one extra evaluation here.

WHAT THE NUMBER MEANS, AND WHAT IT DOES NOT
-------------------------------------------
This measures the horizon the trained model *has*. It does not measure whether
`spread_slow_poles` is why it has it: there is no control arm here, no matched
seed, and n=1. Attributing the result to the spread would need a run with the
committed uniform initialisation and everything else held fixed, which belongs on
the research side as a pre-registered candidate.

Not part of the research protocol. No number it prints is a reported figure.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO / "src") not in sys.path:
    sys.path.insert(0, str(_REPO / "src"))

from snnchat.data import ChatCorpus, SequentialEval  # noqa: E402
from snnchat.generate import load_chat_checkpoint  # noqa: E402

_LOG2E = 1.4426950408889634


def _divisors_up_to(n: int) -> list[int]:
    """Every k that divides n, so the reshape is exact at every point.

    A k that does not divide L would need the batch trimmed, and then the sweep
    would score different characters at different k -- which is precisely the
    confound the reshape exists to avoid.
    """
    return [k for k in range(1, n + 1) if n % k == 0]


@torch.no_grad()
def bpc_at_k(model, batches, k: int, device, vocab_size: int) -> tuple[float, int]:
    """Fresh-state bpc when the state is zeroed every `k` characters."""
    total_nats, total_chars = 0.0, 0
    for x, y in batches:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        b, length = x.shape
        if length % k:
            raise ValueError(f"seq_len {length} is not divisible by k={k}")
        xr = x.reshape(b * length // k, k)
        yr = y.reshape(b * length // k, k)
        logits, _, _ = model(xr, None)
        loss = F.cross_entropy(
            logits.reshape(-1, vocab_size).float(), yr.reshape(-1), reduction="sum"
        )
        total_nats += float(loss)
        total_chars += yr.numel()
    return (total_nats / max(total_chars, 1)) * _LOG2E, total_chars


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--data-dir", default="data/chat")
    p.add_argument("--source", default="soda",
                   help="which val split to measure on; dialogue by default, "
                        "because that is the regime the horizon matters in")
    p.add_argument("--seq-len", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--max-batches", type=int, default=16)
    p.add_argument("--device", default=None)
    p.add_argument("--out", default=None)
    args = p.parse_args(argv)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg, ck = load_chat_checkpoint(args.ckpt, device=device)
    vocab_size = cfg.vocab_size

    corpus = ChatCorpus(args.data_dir, vocab_version=cfg.vocab_version)
    src = SequentialEval(corpus, args.source, args.batch_size, args.seq_len)
    batches = []
    for i, batch in enumerate(src):
        if i >= args.max_batches:
            break
        batches.append(batch)
    if not batches:
        raise SystemExit(f"no val data for source {args.source!r}")

    ks = [k for k in _divisors_up_to(args.seq_len) if k >= 1]
    print(f"checkpoint {args.ckpt}  (step {ck.get('step')})")
    print(f"source {args.source}, {len(batches)} batches of "
          f"[{args.batch_size}, {args.seq_len}]\n")
    print(f"{'k':>6}  {'bpc':>9}  {'gain vs k-1':>12}")

    rows = []
    prev = None
    t0 = time.perf_counter()
    for k in ks:
        bpc, n_chars = bpc_at_k(model, batches, k, device, vocab_size)
        gain = "" if prev is None else f"{prev - bpc:+.5f}"
        print(f"{k:>6}  {bpc:>9.5f}  {gain:>12}")
        rows.append({"k": k, "bpc": bpc, "n_chars": n_chars})
        prev = bpc

    # --- F1: k = L must reproduce the ordinary fresh evaluation ------------
    full = rows[-1]["bpc"]
    total_nats, total_chars = 0.0, 0
    with torch.no_grad():
        for x, y in batches:
            x = x.to(device)
            y = y.to(device)
            logits, _, _ = model(x, None)
            total_nats += float(F.cross_entropy(
                logits.reshape(-1, vocab_size).float(), y.reshape(-1), reduction="sum"))
            total_chars += y.numel()
    reference = (total_nats / total_chars) * _LOG2E
    residual = abs(full - reference)
    print(f"\nF1  k=L reshape is the identity: probe {full:.6f} vs direct "
          f"{reference:.6f}  residual {residual:.2e}")
    if residual > 1e-6:
        raise SystemExit(
            f"F1 FAILED: the probe and a direct evaluation disagree by {residual:.2e} "
            f"bpc at k=L, where the reshape is the identity. Chase this rather than "
            f"widening the tolerance -- it means the two paths are not scoring the "
            f"same characters."
        )

    # --- where does the curve flatten? -------------------------------------
    # "Horizon" here is the smallest k beyond which no doubling of the context
    # buys 0.01 bpc. Stated explicitly because it is a threshold on a smooth
    # curve, not a property of the model: a different threshold gives a
    # different number and neither is more correct.
    best = rows[-1]["bpc"]
    horizon = rows[-1]["k"]
    for row in rows:
        if row["bpc"] - best <= 0.01:
            horizon = row["k"]
            break
    print(f"\nhorizon (smallest k within 0.01 bpc of k=L): {horizon} characters")
    print(f"total {time.perf_counter() - t0:.1f}s")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps({
            "checkpoint": str(args.ckpt), "step": ck.get("step"),
            "source": args.source, "seq_len": args.seq_len,
            "rows": rows, "horizon_0p01": horizon,
            "f1_residual": residual,
        }, indent=2), encoding="utf-8")
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
