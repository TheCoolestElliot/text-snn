"""Measured kernel counts for the Phase-2 report, and the A1 check.

01_reconnaissance.md 3.4 derived `t = 14.5 us x kernels` from microbenchmarks
and 8.1.1 promoted "kernels issued" to the primary performance metric.
Assumption A1 says that constant holds for the *real* model, not just for
chains of torch.add.  This script produces the evidence for both claims in the
form the report needs: a table of counts, and the predicted-versus-measured
wall clock beside it.

    python scripts/bench/kernel_count.py [out.json] [--seq-len 256] [--no-trainer]

Nothing here asserts.  The assertions live in tests/test_kernel_count.py; this
is the instrument, not the gate.

Read the two numbers that matter together: the fused scan should issue exactly
one kernel per timestep per layer (spec 12 / A1), and the whole graphed step
should show the ~11x that 3.8 measured.  If the fused scan issues two, the
kernel is being launched with a cast or a stack in the loop, and the cost model
says that costs 14.5 us x L x K -- which at L=256, K=2 is 7.4 ms per step, or
roughly a third of the entire step budget.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO / "src") not in sys.path:
    sys.path.insert(0, str(_REPO / "src"))

from snn.config import Config, seed_everything  # noqa: E402
from snn.kernels import jiterator_available  # noqa: E402
from snn.metrics import count_cuda_kernels  # noqa: E402
from snn.model import build_model, count_params  # noqa: E402
from snn.neuron import lif_scan  # noqa: E402

# 01_reconnaissance.md 3.4: 14.5 us +- 0.7 us per kernel, flat across a 32x
# range of ops-per-step and three orders of magnitude of tensor size.
US_PER_KERNEL = 14.5e-6


def timeit(fn, iters: int = 20, warmup: int = 5) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters


def _row(name: str, fn, timesteps: int, iters: int = 20) -> dict:
    counts = count_cuda_kernels(fn)
    total = int(counts["total"])
    measured = timeit(fn, iters=iters)
    return {
        "variant": name,
        "kernels": total,
        "kernels_per_timestep": round(total / timesteps, 3) if timesteps else None,
        "measured_ms": round(measured * 1e3, 3),
        "predicted_ms": round(total * US_PER_KERNEL * 1e3, 3),
        "measured_over_predicted": round(measured / (total * US_PER_KERNEL), 3)
        if total else None,
        # memcpy is billed at the same launch cost but is a different bug when it
        # shows up in the inner loop, so it is reported beside the count rather
        # than folded into it (see snn.metrics.count_cuda_kernels).
        "memcpy": counts.get("memcpy"),
        "memset": counts.get("memset"),
        "launch_calls": counts.get("launch_calls"),
        "top_kernels": _top(counts.get("by_short_name") or counts.get("by_name", {}), 6),
    }


def _top(by_name: dict, n: int) -> list:
    return sorted(by_name.items(), key=lambda kv: -kv[1])[:n]


def bench_scan(cfg: Config, results: dict) -> None:
    """The inner loop in isolation: this is where kernel count is decided."""
    B, L, d = cfg.batch_size, cfg.seq_len, cfg.d_model
    cur = torch.randn(B, L, d, device="cuda", dtype=torch.float32)
    v0 = torch.zeros(B, d, device="cuda", dtype=torch.float32)
    kw = dict(beta=cfg.beta, thr=cfg.threshold, alpha=cfg.surrogate_alpha,
              reset=cfg.reset)

    rows = []
    with torch.no_grad():
        rows.append(_row("lif_scan eager (fwd, 1 layer)",
                         lambda: lif_scan(cur, v0, fused=False, **kw), L))
        if jiterator_available():
            rows.append(_row("lif_scan fused (fwd, 1 layer)",
                             lambda: lif_scan(cur, v0, fused=True, **kw), L))

    # Backward doubles the timestep loop, so it is reported separately rather
    # than folded into a single "training" number that hides where the cost is.
    cur_g = cur.clone().requires_grad_(True)

    def _bwd(fused: bool):
        def run():
            spikes, _ = lif_scan(cur_g, v0, fused=fused, **kw)
            spikes.sum().backward()
            cur_g.grad = None
        return run

    rows.append(_row("lif_scan eager (fwd+bwd, 1 layer)", _bwd(False), L, iters=10))
    if jiterator_available():
        rows.append(_row("lif_scan fused (fwd+bwd, 1 layer)", _bwd(True), L, iters=10))
    results["scan"] = rows


def bench_model(cfg: Config, results: dict) -> None:
    model = build_model(cfg).cuda()
    model.train()
    x = torch.randint(0, cfg.vocab_size, (cfg.batch_size, cfg.seq_len),
                      device="cuda", dtype=torch.int64)
    y = torch.randint(0, cfg.vocab_size, (cfg.batch_size, cfg.seq_len),
                      device="cuda", dtype=torch.int64)
    results["params"] = count_params(model)

    def fwd():
        with torch.no_grad():
            model(x, None)

    def fwd_bwd():
        logits, _, _ = model(x, None)
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, cfg.vocab_size).float(), y.reshape(-1))
        loss.backward()
        model.zero_grad(set_to_none=True)

    steps = cfg.seq_len * cfg.n_layers
    results["model"] = [
        _row("model forward", fwd, steps, iters=10),
        _row("model forward+backward", fwd_bwd, steps, iters=10),
    ]


def bench_trainer(cfg: Config, results: dict) -> None:
    """The real training step, eager and graphed -- the A9/3.8 comparison."""
    import tempfile

    from snn.train import Trainer

    # A benchmark must not leave run directories in experiments/runs/ that look
    # like real experiments to whoever reads the tree later.
    scratch = tempfile.mkdtemp(prefix="snn_bench_")
    rows = []
    cfg = _clone(cfg, out_dir=scratch, log_every=0, eval_every=0, ckpt_every=0)
    eager_cfg = _clone(cfg, cuda_graph=False, run_name="bench_eager")
    tr = Trainer(eager_cfg)
    tr._ensure_ready()
    tr._fill_inputs(0, non_blocking=False)
    steps = cfg.seq_len * cfg.n_layers
    rows.append(_row("training step (eager)", tr._step_body, steps, iters=10))
    eager_ms = rows[-1]["measured_ms"]
    del tr

    graph_cfg = _clone(cfg, cuda_graph=True, run_name="bench_graph")
    tg = Trainer(graph_cfg)
    tg._ensure_ready()
    tg._fill_inputs(0, non_blocking=False)
    try:
        row = _row("training step (graph replay)", tg.graph.replay, steps, iters=20)
    except Exception as exc:  # profiling a graph replay is not guaranteed
        row = {"variant": "training step (graph replay)",
               "error": f"{type(exc).__name__}: {str(exc)[:200]}"}
    else:
        # Derived fields are computed after the row is built, and separately
        # from it: a division by a sub-microsecond replay time must not turn a
        # successful measurement into an "error" row (or, worse, append a second
        # row for the same variant).
        row["capture_seconds"] = round(tg.capture_seconds, 3)
        row["speedup_vs_eager"] = (round(eager_ms / row["measured_ms"], 2)
                                   if row["measured_ms"] else None)
    rows.append(row)
    del tg
    results["trainer"] = rows


def _clone(cfg: Config, **over) -> Config:
    import dataclasses
    return dataclasses.replace(cfg, **over)


def _fmt(value, spec: str) -> str:
    """`-` for a missing measurement rather than a crash.

    `measured_over_predicted` is None whenever the profiler attributed zero
    device kernels to the call, which is the expected outcome for a CUDA-graph
    replay on some builds (kineto sees one cudaGraphLaunch and nothing under
    it). Formatting that as a float used to raise here -- *after* the
    measurements had been taken and *before* the JSON was written, so a
    profiler quirk destroyed the whole run's output.
    """
    return "-" if value is None else format(value, spec)


def print_table(results: dict) -> None:
    print()
    print("| variant | kernels | kernels/timestep | measured ms | predicted ms "
          "(14.5 us x k) | measured/predicted |")
    print("|---|---:|---:|---:|---:|---:|")
    for section in ("scan", "model", "trainer"):
        for row in results.get(section, []):
            if "error" in row:
                print(f"| {row['variant']} | - | - | - | - | {row['error']} |")
                continue
            print(f"| {row['variant']} | {row.get('kernels', '-')} | "
                  f"{_fmt(row.get('kernels_per_timestep'), '.3f')} | "
                  f"{_fmt(row.get('measured_ms'), '.3f')} | "
                  f"{_fmt(row.get('predicted_ms'), '.3f')} | "
                  f"{_fmt(row.get('measured_over_predicted'), '.2f')} |")
    print()


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("out", nargs="?",
                   default="docs/reports/data/bench_kernel_count.json")
    p.add_argument("--corpus", default="enwik8")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--seq-len", type=int, default=256)
    p.add_argument("--d-model", type=int, default=512)
    p.add_argument("--n-layers", type=int, default=2)
    p.add_argument("--vocab-size", type=int, default=205)
    p.add_argument("--no-trainer", action="store_true",
                   help="skip the rows that need a downloaded corpus")
    args = p.parse_args(argv)

    if not torch.cuda.is_available():
        raise SystemExit("this benchmark measures CUDA kernel launches; no CUDA here")

    seed_everything(0, deterministic=True)
    cfg = Config(
        corpus=args.corpus, data_dir=args.data_dir,
        batch_size=args.batch_size, seq_len=args.seq_len,
        d_model=args.d_model, n_layers=args.n_layers,
        vocab_size=args.vocab_size, device="cuda",
    )

    results: dict = {
        "config": {
            "batch_size": cfg.batch_size, "seq_len": cfg.seq_len,
            "d_model": cfg.d_model, "n_layers": cfg.n_layers,
            "vocab_size": cfg.vocab_size, "reset": cfg.reset,
            "us_per_kernel_assumed": US_PER_KERNEL,
        },
        "jiterator_available": jiterator_available(),
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(0),
    }

    bench_scan(cfg, results)
    bench_model(cfg, results)
    if not args.no_trainer:
        try:
            bench_trainer(cfg, results)
        except Exception as exc:
            results["trainer_error"] = f"{type(exc).__name__}: {str(exc)[:300]}"
            print(f"trainer rows skipped: {results['trainer_error']}", file=sys.stderr)

    # Persist before printing: the measurements cost minutes of GPU time and the
    # table is cosmetic, so a formatting problem must never be able to discard
    # them.
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)
    print_table(results)
    print(f"WROTE {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
