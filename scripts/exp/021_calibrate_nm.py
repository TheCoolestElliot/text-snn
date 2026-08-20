"""EXP_021 calibration: what is the scale of the LOCAL prediction error?

    python scripts/exp/021_calibrate_nm.py --out docs/reports/data/exp_021_nm_calibration.json

`snn.neuromod.bernoulli_rpe` squashes `phi` through `tanh(phi/tau)`, and `tau`
sets the whole arm's operating point. `EXP_018` fixed its `tau` by measuring
`phi`'s sd on four committed checkpoints and taking the median, and the four
agreed to 3.1 % -- so the scale was a property of the task rather than of the
neuron. This is the same procedure for a different quantity, and it is run
BEFORE the pre-registration commits to a value, for the same reason: a squash
scale chosen by taste is a hyperparameter the arm's result would depend on.

WHAT IS MEASURED
----------------
For every committed checkpoint, and for every layer `k >= 1`, `phi` is computed
from layer `k-1`'s emission under two predictors, both with slope `a = 0`:

  `init`     `b = nm_b_init`, the state the arm actually starts in;
  `matched`  `b_c = logit(r_c)` with `r_c` layer `k-1`'s MEASURED per-channel
             firing rate on this checkpoint -- the fixed point a per-channel
             predictor converges to, and therefore the state the arm spends
             almost all of training near.

`tau` is taken from `matched`. That choice was FORCED by a measurement, and the
measurement is recorded rather than replaced: calibrating on `init` returned
`phi mean = -0.879 to -1.018` with **100.0 % of values negative** and
`|phi| > 2 sd` at 79-100 % of positions, because `nm_b_init = -2.97` is
`logit(0.0488)` -- layer 0's rate at INITIALISATION -- while these checkpoints
have converged to 0.345-0.392. A `tau` fitted to that is fitted to a constant
offset, and `tanh` of a constant offset is a constant gain, which by `EXP_008`
Identity 1 is exactly a learned per-channel threshold: the arm would have
measured an already-adopted arm (#5) under a new name, and it would have
trained and scored plausibly while doing so.

C1 (abort-on-fail): under `matched`, `phi`'s mean must be within `--c1-tol` of
zero. This is not a hope about the data. With `p_c` equal to the true marginal
`r_c`, `E[S] = mean_c H_b(r_c) = E[H]` **exactly**, so `E[phi] = 0` identically.
C1 therefore checks the derivation and the implementation, not the corpus.

C2 (reported, not a bar): the fraction of `|phi|/tau` above 2.0, i.e. how much
of the signal `tanh` is throwing away at the proposed scale. Reported per
checkpoint so the pre-registration can state it rather than discover it.

WHAT THIS IS NOT
----------------
It trains nothing, adopts nothing, and reads no bpc. Every number is a marker.
It does not choose `nm_gain_init`, which stays at the 0.0 that nests the
baseline, and it does not sweep anything.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

import torch

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO / "src") not in sys.path:
    sys.path.insert(0, str(_REPO / "src"))

from snn.config import Config  # noqa: E402
from snn.data import Corpus, windowed_eval_batches  # noqa: E402
from snn.model import build_model  # noqa: E402
from snn.neuromod import bernoulli_rpe  # noqa: E402


@torch.no_grad()
def measure(run_dir: Path, args, device: str) -> dict:
    cfg = Config.from_dict(json.loads((run_dir / "config.json").read_text()))
    cfg.device = device
    corpus = Corpus(cfg.corpus, cfg.data_dir)
    cfg.vocab_size = corpus.vocab_size
    model = build_model(cfg)
    ck = torch.load(run_dir / args.ckpt, map_location=device, weights_only=False)
    model.load_state_dict(ck["model"] if "model" in ck else ck)
    model.eval()
    model.keep_spikes = True

    per_layer: dict[int, dict] = {}
    chunks: dict[int, list[torch.Tensor]] = {}
    chan_rate: dict[int, torch.Tensor] = {}
    rates: dict[int, list[float]] = {}
    taken = 0
    for x, _ in windowed_eval_batches(corpus.split(args.split), cfg.batch_size,
                                      cfg.seq_len):
        x = x.to(device, non_blocking=True)
        _, _, aux = model(x, None)
        spikes = aux["spikes"]
        for k in range(1, cfg.n_layers):
            s = spikes[k - 1]                       # the DRIVING layer
            chunks.setdefault(k, []).append(s.to(torch.uint8))
            rates.setdefault(k, []).append(float(s.mean()))
        taken += x.shape[0]
        if taken >= args.n_windows:
            break
    model.keep_spikes = False

    # Per-channel rates first, then phi under each predictor -- two passes over
    # a tensor already in memory, rather than one pass with a guessed prior.
    for k, blocks in chunks.items():
        s = torch.cat(blocks, 0).float()
        blocks.clear()
        d = s.shape[2]
        a = torch.zeros(1, d, device=device)
        r = s.mean(dim=(0, 1)).reshape(1, d)                    # [1, d]
        rc = r.clamp(1e-6, 1.0 - 1e-6)
        b_matched = torch.log(rc / (1.0 - rc))
        b_init = torch.full((1, d), float(args.b_init), device=device)
        per_layer[k] = {
            "matched": bernoulli_rpe(s, a, b_matched).flatten(),
            "init": bernoulli_rpe(s, a, b_init).flatten(),
        }
        chan_rate[k] = r.flatten()
        del s
    torch.cuda.empty_cache()

    out = {"run": run_dir.name, "arch": cfg.arch, "d_model": cfg.d_model,
           "n_layers": cfg.n_layers, "layers": {}}
    for k, both in per_layer.items():
        cell = {
            "driving_layer": k - 1,
            "driving_rate": statistics.fmean(rates[k]),
            "chan_rate_min": float(chan_rate[k].min()),
            "chan_rate_max": float(chan_rate[k].max()),
            "chan_rate_frac_silent": float((chan_rate[k] < 1e-6).float().mean()),
        }
        for mode in ("matched", "init"):
            phi = both[mode]
            sd = float(phi.std())
            q = torch.quantile(
                phi, torch.tensor([0.01, 0.5, 0.99], device=phi.device))
            cell[mode] = {
                "phi_mean": float(phi.mean()),
                "phi_sd": sd,
                "phi_min": float(phi.min()),
                "phi_max": float(phi.max()),
                "phi_q01": float(q[0]), "phi_median": float(q[1]),
                "phi_q99": float(q[2]),
                "frac_positive": float((phi > 0).float().mean()),
                "n": int(phi.numel()),
                # C2: how much of the signal tanh discards at tau = sd
                "frac_abs_over_2sd": float((phi.abs() > 2.0 * sd).float().mean()),
            }
        out["layers"][str(k)] = cell
        both.clear()
    torch.cuda.empty_cache()
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", nargs="+", default=[
        "experiments/runs/snn_beta0.5_s0",
        "experiments/runs/snn_beta0.5_s1",
        "experiments/runs/snn_beta0.5_s2",
        "experiments/runs/twocomp_s0",
    ])
    ap.add_argument("--ckpt", default="ckpt_final.pt")
    ap.add_argument("--split", default="val")
    ap.add_argument("--n-windows", type=int, default=256)
    ap.add_argument("--b-init", type=float, default=-2.97,
                    help="predictor prior logit; the arm's nm_b_init")
    ap.add_argument("--c1-tol", type=float, default=0.05,
                    help="abort if |phi mean| exceeds this at a matched prior")
    ap.add_argument("--out",
                    default="docs/reports/data/exp_021_nm_calibration.json")
    args = ap.parse_args(argv)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    rows = []
    for r in args.runs:
        rd = _REPO / r
        if not (rd / "config.json").exists():
            print(f"[021] SKIP {rd.name}: no config.json", flush=True)
            continue
        res = measure(rd, args, device)
        rows.append(res)
        for k, v in res["layers"].items():
            for mode in ("matched", "init"):
                m = v[mode]
                print(f"[021] {res['run']:22s} L{k}<-L{v['driving_layer']} "
                      f"{mode:8s} rate={v['driving_rate']:.4f} "
                      f"mean={m['phi_mean']:+.5f} sd={m['phi_sd']:.5f} "
                      f"[{m['phi_min']:+.4f},{m['phi_max']:+.4f}] "
                      f"pos={m['frac_positive']:.3f} "
                      f">2sd={m['frac_abs_over_2sd']:.4f}", flush=True)

    sds = [v["matched"]["phi_sd"] for r in rows for v in r["layers"].values()]
    means = [abs(v["matched"]["phi_mean"]) for r in rows
             for v in r["layers"].values()]
    tau = statistics.median(sds) if sds else None
    spread = (max(sds) - min(sds)) / tau if sds and tau else None
    c1 = max(means) if means else None

    out = {
        "torch": torch.__version__,
        "device": device,
        "args": vars(args),
        "checkpoints": rows,
        "C1_max_abs_phi_mean": c1,
        "C1_tol": args.c1_tol,
        "C1_pass": bool(c1 is not None and c1 <= args.c1_tol),
        "nm_tau_median_sd": tau,
        "nm_tau_relative_spread": spread,
        "n_cells": len(sds),
    }
    p = _REPO / args.out
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    print(f"\n[021] nm_tau = median sd = {tau:.6f} over {len(sds)} cells, "
          f"relative spread {spread:.1%}" if tau else "[021] no cells measured")
    print(f"[021] C1 max |phi mean| = {c1:.6f} "
          f"({'PASS' if out['C1_pass'] else 'FAIL'} at tol {args.c1_tol})")
    print(f"[021] wrote {p}")
    return 0 if out["C1_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
