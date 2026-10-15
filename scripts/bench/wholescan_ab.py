"""A/B the two fused-scan implementations on the REAL training and evaluation paths.

**Outside the Phase 1-5 protocol.** Nothing this prints is a reported figure; it
prices a performance change (`src/snn/wholescan.py`, `docs/perf/WHOLESCAN.md`)
whose only permitted effect on any reported figure is none. Its artifact therefore
lives under `docs/perf/data/`.

What it measures, per committed run (config and checkpoint READ from the run
directory; nothing is written there):

  train   `Trainer(cfg)` built from the run's own `config.json`, CUDA-graphed, and
          stepped `--steps` times: ms/step (median of windows, first excluded),
          peak allocated memory, and a SHA-256 over every parameter and every
          AdamW moment afterwards. The two implementations are INTERLEAVED in one
          process -- A B, B A, A B -- because on this box the clock and power
          state move a whole run by ~6% while windows inside a run are tight, so
          two separate sessions would mostly compare the weather.
  eval    `ckpt_final.pt` scored on the full test split under BOTH protocols, the
          way `scripts/evaluate.py` does it: ms/batch, peak memory, and the bpc.

Two assertions make the timings worth reading, and the script FAILS if either
does not hold: the two implementations reach bit-identical weights and optimiser
state from the same seed, and the evaluation reproduces the run's COMMITTED
`final_test.json` bpc exactly (`==` on the float, not a tolerance) under both.
A speed-up measured on a path that computes something else is not a speed-up.

Never run this while anything else is on the GPU (CLAUDE.md), and never while a
mutation campaign is running.
"""

from __future__ import annotations

import argparse
import dataclasses
import gc
import hashlib
import json
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO / "src") not in sys.path:
    sys.path.insert(0, str(_REPO / "src"))

from snn.config import Config, seed_everything  # noqa: E402
import torch  # noqa: E402
from snn.data import Corpus  # noqa: E402
from snn.evaluate import evaluate  # noqa: E402
from snn.model import build_model  # noqa: E402
from snn.train import Trainer  # noqa: E402
from snn.wholescan import SCAN_IMPLS, scan_impl_record, use_scan_impl  # noqa: E402

DEFAULT_RUNS = ("e25_snn_d512_s0", "e25_detach_d512_s0", "e25_detach_d1481_s0")
IMPLS = ("jiterator", "wholescan")
assert set(IMPLS) == set(SCAN_IMPLS)


def _config(raw: dict, **over) -> Config:
    known = {f.name for f in dataclasses.fields(Config)}
    return Config(**{**{k: v for k, v in raw.items() if k in known}, **over})


def _release() -> None:
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


def _sha(tensors) -> str:
    h = hashlib.sha256()
    for name, t in tensors:
        h.update(name.encode())
        h.update(t.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def train_leg(raw: dict, impl: str, data_dir: str, out_dir: str, tag: str,
              n_steps: int, n_windows: int) -> dict:
    cfg = _config(raw, data_dir=data_dir, out_dir=out_dir, run_name=tag, cuda_graph=True)
    _release()
    torch.cuda.reset_peak_memory_stats()
    with use_scan_impl(impl):
        tr = Trainer(cfg)
        t0 = time.perf_counter()
        tr._ensure_ready()
        torch.cuda.synchronize()
        ready = time.perf_counter() - t0
        per = max(1, n_steps // n_windows)
        windows = []
        for _ in range(n_windows):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(per):
                tr.step()
            torch.cuda.synchronize()
            windows.append((time.perf_counter() - t0) / per * 1e3)
        record = scan_impl_record()
    out = {
        "impl": impl, "steps": per * n_windows, "captured": tr.graph is not None,
        "ensure_ready_s": round(ready, 2),
        "ms_per_step_windows": [round(w, 3) for w in windows],
        "ms_per_step": round(statistics.median(windows[1:]), 3),
        "peak_alloc_gib": round(torch.cuda.max_memory_allocated() / 1024 ** 3, 3),
        "sha256_model": _sha(sorted(tr.model.state_dict().items())),
        "sha256_optim": _sha(
            (f"{i}.{k}", v) for i, st in sorted(tr.opt.state_dict()["state"].items())
            for k, v in sorted(st.items()) if torch.is_tensor(v)),
        "scan_impl_record": record,
    }
    del tr
    _release()
    return out


def eval_leg(run_dir: Path, impl: str, data_dir: str) -> dict:
    ck = torch.load(run_dir / "ckpt_final.pt", map_location="cpu", weights_only=False)
    cfg = _config(ck["config"], data_dir=data_dir)
    cfg.eval_max_windows = 0
    seed_everything(cfg.seed, cfg.deterministic)
    corpus = Corpus(cfg.corpus, cfg.data_dir)
    model = build_model(cfg).to(cfg.device)
    model.load_state_dict(ck["model"])
    model.eval()
    out: dict = {"impl": impl}
    with use_scan_impl(impl):
        evaluate(model, corpus, "test", cfg, "fresh", max_windows=cfg.batch_size * 2)  # warm
        for protocol in ("fresh", "carried"):
            _release()
            torch.cuda.reset_peak_memory_stats()
            t0 = time.perf_counter()
            r = evaluate(model, corpus, "test", cfg, protocol, max_windows=0)
            torch.cuda.synchronize()
            dt = time.perf_counter() - t0
            n_batches = -(-r["n_windows"] // cfg.batch_size)
            out[protocol] = {
                "bpc": r["bpc"], "nats": r["nats"], "n_windows": r["n_windows"],
                "seconds": round(dt, 3), "ms_per_batch": round(dt / n_batches * 1e3, 2),
                "peak_alloc_gib": round(torch.cuda.max_memory_allocated() / 1024 ** 3, 3),
            }
    del model
    _release()
    return out


def _median(legs: list[dict], *path: str) -> float:
    def get(d):
        for p in path:
            d = d[p]
        return d
    return round(statistics.median(get(leg) for leg in legs), 3)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs-root", required=True, help="directory holding the e25_* runs")
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--runs", default=",".join(DEFAULT_RUNS))
    ap.add_argument("--steps", type=int, default=240)
    ap.add_argument("--windows", type=int, default=6)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--out", default="docs/perf/data/wholescan_ab.json")
    args = ap.parse_args(argv)

    head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=_REPO,
                          capture_output=True, text=True).stdout.strip()
    smi = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
        capture_output=True, text=True).stdout.strip()
    report: dict = {
        "bench": "wholescan_ab", "protocol": "outside Phase 1-5; see docs/perf/WHOLESCAN.md",
        "head": head, "torch": torch.__version__, "gpu": smi,
        "steps": args.steps, "windows": args.windows, "rounds": args.rounds, "runs": {},
    }
    scratch = tempfile.mkdtemp(prefix="wholescan_ab_")
    failures: list[str] = []
    try:
        for run in [r for r in args.runs.split(",") if r]:
            run_dir = Path(args.runs_root) / run
            raw = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
            committed = json.loads((run_dir / "final_test.json").read_text(encoding="utf-8"))
            # d=1481 steps are ~5x longer; same wall budget, fewer steps
            steps = args.steps if raw["d_model"] <= 512 else max(args.windows, args.steps // 3)
            legs: dict[str, list] = {"train": [], "eval": []}
            for rnd in range(args.rounds):
                order = IMPLS if rnd % 2 == 0 else IMPLS[::-1]   # A B, B A, A B
                for impl in order:
                    leg = train_leg(raw, impl, args.data_dir, scratch, f"{run}_{impl}_{rnd}",
                                    steps, args.windows)
                    legs["train"].append(leg)
                    print(f"{run:22s} train r{rnd} {impl:9s} {leg['ms_per_step']:8.3f} ms/step  "
                          f"peak {leg['peak_alloc_gib']:.3f} GiB", flush=True)
                for impl in order:
                    leg = eval_leg(run_dir, impl, args.data_dir)
                    legs["eval"].append(leg)
                    print(f"{run:22s} eval  r{rnd} {impl:9s} fresh "
                          f"{leg['fresh']['ms_per_batch']:7.2f} carried "
                          f"{leg['carried']['ms_per_batch']:7.2f} ms/batch", flush=True)

            # ---- the two assertions that make the timings worth reading
            shas = {(leg["sha256_model"], leg["sha256_optim"]) for leg in legs["train"]}
            weights_equal = len(shas) == 1
            bpc_equal = all(
                leg[p]["bpc"] == committed["results"][p]["bpc"]
                and leg[p]["nats"] == committed["results"][p]["nats"]
                for leg in legs["eval"] for p in ("fresh", "carried"))
            if not weights_equal:
                failures.append(f"{run}: weights/optimiser state differ between implementations")
            if not bpc_equal:
                failures.append(f"{run}: evaluation does not reproduce final_test.json")

            summary: dict = {"weights_and_optimiser_bit_identical": weights_equal,
                             "eval_bpc_equals_committed_final_test_json": bpc_equal,
                             "committed_bpc": {p: committed["results"][p]["bpc"]
                                               for p in ("fresh", "carried")}}
            for impl in IMPLS:
                tl = [x for x in legs["train"] if x["impl"] == impl]
                el = [x for x in legs["eval"] if x["impl"] == impl]
                summary[impl] = {
                    "train_ms_per_step": _median(tl, "ms_per_step"),
                    "train_ms_per_step_range": [min(x["ms_per_step"] for x in tl),
                                                max(x["ms_per_step"] for x in tl)],
                    "train_peak_gib": _median(tl, "peak_alloc_gib"),
                    "eval_fresh_ms_per_batch": _median(el, "fresh", "ms_per_batch"),
                    "eval_carried_ms_per_batch": _median(el, "carried", "ms_per_batch"),
                    "eval_peak_gib": _median(el, "fresh", "peak_alloc_gib"),
                }
            old, new = summary["jiterator"], summary["wholescan"]
            summary["speedup"] = {
                "train": round(old["train_ms_per_step"] / new["train_ms_per_step"], 3),
                "eval_fresh": round(
                    old["eval_fresh_ms_per_batch"] / new["eval_fresh_ms_per_batch"], 3),
                "eval_carried": round(
                    old["eval_carried_ms_per_batch"] / new["eval_carried_ms_per_batch"], 3),
                "train_peak_ratio": round(new["train_peak_gib"] / old["train_peak_gib"], 3),
                "eval_peak_ratio": round(new["eval_peak_gib"] / old["eval_peak_gib"], 3),
            }
            report["runs"][run] = {
                "arch": raw["arch"], "d_model": raw["d_model"], "n_layers": raw["n_layers"],
                "batch_size": raw["batch_size"], "seq_len": raw["seq_len"],
                "train_steps_per_leg": steps, "summary": summary, "legs": legs}
            print(json.dumps({run: summary}, indent=1), flush=True)
    finally:
        shutil.rmtree(scratch, ignore_errors=True)

    report["failures"] = failures
    out = _REPO / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"WROTE {args.out}")
    for f in failures:
        print(f"FAILED: {f}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
