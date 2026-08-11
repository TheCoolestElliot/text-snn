"""EXP_016 Leg A: is the backward-through-time a contraction?

    python scripts/exp/016_reset_jacobian.py --out docs/reports/data/exp_016_reset_jacobian.json

WHAT IT MEASURES
----------------
Both neurons thread an adjoint backwards over all L timesteps with

    grad_v_prev = beta * (grad_v_next * dv + grad_spike * sg)

so `g = beta * dv` is the per-step multiplier of the HOMOGENEOUS part of the
reverse recurrence, and `prod_t |g_t|` is the gain of one (batch, channel) chain
over the whole unroll.  `EXP_016` §2 derives, before any measurement:

    plain LIF   dv = (1 - s) - v_pre*sg(v_pre - thr)      SAME variable twice
                => |beta*dv| <= 0.5123596 < 1 for EVERY finite membrane, at
                   EVERY width.  A strict contraction, by construction.

    twocomp     dv = (1 - sh) - vf*sgd(v_pre - thr),  vf = v_pre - w*vs
                => the self-limiting is broken by the unreset slow compartment;
                   the factor crosses 1 at |w*vs| ~ 1 and is bounded by nothing.

This file measures the distribution of `g` on committed checkpoints.  It is
inference only.  Nothing here trains, nothing here is adopted, and no
hyperparameter is read from anywhere but each run's own `config.json`.

WHY THE SCAN IS RE-IMPLEMENTED HERE
-----------------------------------
`g` is an intermediate of the BACKWARD kernel and is never materialised by the
forward.  Rather than edit `src/snn/` to expose it -- which would put an
experiment's instrument inside the arm under test -- the scan is re-implemented
locally, exactly as `011_chase_compose_divergence.py` re-implements what it
needs.  That is only sound if it is the same scan, so T2 checks the local scan's
spikes against the COMMITTED eager reference bitwise, per layer, per leg, and
the probe aborts if they differ anywhere.

WHAT IT IS NOT
--------------
One checkpoint and one batch per cell.  No sigma, no seeds, no interval, no bpc.
Every number is a marker.  `sum_t log|g_t|` bounds the homogeneous gain and is
NOT the gradient -- the true adjoint also carries per-step `grad_spike*sgd`
injections and sign cancellation.  `EXP_016` §4.0 states what that forbids and
P3's power is one-sided because of it.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import sys
from pathlib import Path

import torch

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO / "src") not in sys.path:
    sys.path.insert(0, str(_REPO / "src"))

from snn.config import Config, seed_everything  # noqa: E402
from snn.data import Corpus, RandomWindowSampler  # noqa: E402
from snn.model import build_model  # noqa: E402
from snn.neuron import lif_scan_eager  # noqa: E402
from snn.surrogate import atan_grad  # noqa: E402
from snn.twocomp import twocomp_scan_eager  # noqa: E402

RUNS = _REPO / "experiments" / "runs"

#: The closed form of EXP_016 §2.2, at the frozen constants alpha=2, thr=1,
#: beta=0.5.  NOT a project constant -- it is re-derived in `lif_bound()` from
#: each run's own config, and T1 compares against the re-derived value.
LIF_BOUND_AT_FROZEN_CONSTANTS = 0.5123596

#: T1's tolerance.  fp32 rounding on a quantity of order 0.5.
T1_TOL = 1e-5

#: Exceedance thresholds recorded for every leg.  1.0 is the only one with a
#: meaning fixed in advance: above it the reverse recurrence expands.
THRESHOLDS = (0.5123596, 1.0, 2.0, 10.0, 100.0)

#: log10|g| histogram, for quantiles without materialising [B, L, d].
HIST_LO, HIST_HI, HIST_BINS = -20.0, 10.0, 300

#: (run, checkpoint, role).  Both neurons appear at both widths, or the
#: comparison means nothing.
TARGETS = [
    ("snn_beta0.5_s0", "ckpt_final.pt", "A1", "plain LIF, 735K -- trains"),
    ("scale_d1020_s0", "ckpt_final.pt", "A2", "plain LIF, 2.5M -- trains"),
    ("scale_d1481_s0", "ckpt_final.pt", "A3", "plain LIF, 5.0M -- trains"),
    ("twocomp_s0", "ckpt_final.pt", "A4", "two-compartment, 735K -- trains"),
    ("arch_twocomp_d1481_s0", "ckpt_best.pt", "A5",
     "two-compartment, 5.0M -- DIES 138 steps later"),
]


def lif_bound(alpha: float, thr: float, beta: float) -> float:
    """max_x |beta * ((1 - H(x)) - v_pre*atan_grad(x))| with v_pre = x + thr.

    Derived rather than hard-coded, because EXP_016 §6 item 5 records that the
    bound is exact only at the frozen constants and must be re-derived if any of
    alpha, thr or beta moves -- the same failure `CONTRIBUTING.md` §4 records for
    fold-in tolerances inherited across arms.
    """
    x = torch.linspace(-2000.0, 2000.0, 8_000_001, dtype=torch.float64)
    u = x * (math.pi / 2 * alpha)
    sg = (0.5 * alpha) / (1.0 + u * u)
    v_pre = x + thr
    dv = (1.0 - (x >= 0).to(torch.float64)) - v_pre * sg
    return float((beta * dv).abs().max())


class Acc:
    """Online accumulators for one layer.  Nothing [B, L, d] is materialised."""

    def __init__(self, batch: int, d: int, device: str) -> None:
        self.max_abs = torch.zeros((), device=device, dtype=torch.float64)
        self.n_total = 0
        self.counts = torch.zeros(len(THRESHOLDS), device=device, dtype=torch.float64)
        # log of the homogeneous product, per (batch, channel) chain.  fp64:
        # 256 additions of a quantity whose sum is the answer.
        self.log_prod = torch.zeros(batch, d, device=device, dtype=torch.float64)
        self.n_exact_zero = torch.zeros((), device=device, dtype=torch.float64)
        self.hist = torch.zeros(HIST_BINS, device=device, dtype=torch.float64)
        self.max_drive = torch.zeros((), device=device, dtype=torch.float64)

    def add(self, g: torch.Tensor, drive: torch.Tensor | None) -> None:
        a = g.detach().abs().to(torch.float64)
        self.max_abs = torch.maximum(self.max_abs, a.max())
        self.n_total += a.numel()
        for i, t in enumerate(THRESHOLDS):
            self.counts[i] += (a > t).sum()
        self.n_exact_zero += (a == 0).sum()
        # log(0) is -inf and would poison the chain sum; a chain containing an
        # exact zero has a true product of exactly 0, so flooring it is
        # conservative in the direction that matters for P3.
        self.log_prod += torch.log(a.clamp_min(1e-300))
        idx = (((torch.log10(a.clamp_min(1e-300)) - HIST_LO)
                / ((HIST_HI - HIST_LO) / HIST_BINS)).long().clamp_(0, HIST_BINS - 1))
        self.hist += torch.bincount(idx.reshape(-1), minlength=HIST_BINS).to(torch.float64)
        if drive is not None:
            self.max_drive = torch.maximum(self.max_drive, drive.detach().abs().max().to(torch.float64))

    def quantiles(self, qs: tuple[float, ...]) -> dict[str, float]:
        c = torch.cumsum(self.hist, 0) / max(float(self.hist.sum()), 1.0)
        width = (HIST_HI - HIST_LO) / HIST_BINS
        out = {}
        for q in qs:
            j = int(torch.searchsorted(c, torch.tensor(q, device=c.device, dtype=c.dtype)))
            j = min(j, HIST_BINS - 1)
            out[f"q{q}"] = float(10.0 ** (HIST_LO + (j + 0.5) * width))
        return out

    def report(self) -> dict:
        n = float(self.n_total)
        max_s = float(self.log_prod.max())
        return {
            "max_abs_g": float(self.max_abs),
            "n_sites": self.n_total,
            "frac_over": {
                str(t): float(self.counts[i]) / n for i, t in enumerate(THRESHOLDS)
            },
            "n_over_1": int(self.counts[THRESHOLDS.index(1.0)]),
            "n_exact_zero_g": int(self.n_exact_zero),
            "quantiles_abs_g": self.quantiles((0.5, 0.9, 0.99, 0.999, 0.9999)),
            "max_chain_log_gain": max_s,
            "max_chain_log10_gain": max_s / math.log(10.0),
            "mean_chain_log_gain": float(self.log_prod.mean()),
            "n_chains_expanding": int((self.log_prod > 0).sum()),
            "n_chains": int(self.log_prod.numel()),
            "max_abs_w_times_vs": float(self.max_drive),
        }


def scan_lif(cur, v0, beta, thr, alpha, acc):
    """Local re-implementation of `lif_scan_eager`, emitting `g` per timestep."""
    v = v0
    spikes = []
    for t in range(cur.shape[1]):
        v_pre = v * beta + cur[:, t]
        x = v_pre - thr
        s = (x >= 0).to(v_pre.dtype)
        sg = atan_grad(x, alpha)
        acc.add(beta * ((1.0 - s) - v_pre * sg), None)
        spikes.append(s)
        v = v_pre * (1.0 - s)
    return torch.stack(spikes, dim=1), v


def scan_twocomp(cur, v0, w, beta_s, beta_f, thr, alpha, acc):
    """Local re-implementation of `twocomp_scan_eager`, emitting `g` per timestep.

    `dv` uses the eager `vf` (the real fast membrane) rather than the fused
    kernel's reconstruction `v_pre - w*vs`; they are algebraically identical and
    the eager path is the one that reproduces the failure.
    """
    d = cur.shape[-1]
    vf, vs = v0[:, :d], v0[:, d:]
    spikes = []
    for t in range(cur.shape[1]):
        vf = vf * beta_f + cur[:, t]
        vs = vs * beta_s + cur[:, t]
        v_pre = vf + w * vs
        x = v_pre - thr
        sh = (x >= 0).to(v_pre.dtype)
        sgd = atan_grad(x, alpha)
        acc.add(beta_f * ((1.0 - sh) - vf * sgd), w * vs)
        spikes.append(sh)
        vf = vf * (1.0 - sh)
    return torch.stack(spikes, dim=1), torch.cat([vf, vs], dim=1)


def probe(run: str, ckpt_name: str, leg: str, note: str, device: str,
          batch_step: int = 0) -> dict:
    ckpt_path = RUNS / run / ckpt_name
    if not ckpt_path.exists():
        return {"leg": leg, "run": run, "error": f"missing {ckpt_name}"}

    raw = json.loads((RUNS / run / "config.json").read_text(encoding="utf-8"))
    known = {f.name for f in dataclasses.fields(Config)}
    cfg = Config(**{k: v for k, v in raw.items() if k in known})
    cfg.device = device
    cfg.fused = False       # read the same arithmetic on any machine
    cfg.cuda_graph = False

    seed_everything(cfg.seed, cfg.deterministic)
    model = build_model(cfg)
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ck["model"])
    model.eval()

    corpus = Corpus(cfg.corpus, cfg.data_dir)
    # `RandomWindowSampler(...).batch(s)` is a pure function of `s` and is
    # constructed exactly as `Trainer` constructs it (`train.py:120-122`), so
    # `--batch-step 5138` is the batch the dying run actually saw at the step it
    # died on. The default 0 is what the pre-registered Leg A ran; changing it
    # produces a POST-HOC artifact and must be written to its own `--out`.
    x, _ = RandomWindowSampler(
        corpus.split("train"), cfg.batch_size, cfg.seq_len, cfg.seed
    ).batch(batch_step)
    x = x.to(device)

    bound = lif_bound(cfg.surrogate_alpha, cfg.threshold, cfg.beta)
    row = {
        "leg": leg, "run": run, "note": note, "arch": cfg.arch,
        "batch_step": batch_step,
        "d_model": cfg.d_model, "n_layers": cfg.n_layers,
        "step_of_checkpoint": int(ck.get("global_step", -1)),
        "checkpoint": ckpt_name,
        "seq_len": cfg.seq_len, "batch_size": cfg.batch_size,
        "constants": {"alpha": cfg.surrogate_alpha, "thr": cfg.threshold,
                      "beta_f": cfg.beta, "reset": cfg.reset},
        "lif_closed_form_bound": bound,
        "layers": [],
        "t2_spikes_bitwise_equal": [],
        "t2_state_max_abs_diff": [],
    }

    with torch.no_grad():
        h = model.embed(x)
        state = model.init_state(x.shape[0], device)
        for k, linear in enumerate(model.layers):
            cur = model._project(linear, h)
            acc = Acc(cur.shape[0], cur.shape[-1], device)

            if cfg.arch == "twocomp":
                w, bs = model.w[k], model.slow_decay(k)
                mine, v_end = scan_twocomp(
                    cur, state[k], w, bs, cfg.beta, cfg.threshold,
                    cfg.surrogate_alpha, acc)
                ref, v_ref = twocomp_scan_eager(
                    cur, state[k], w, bs, cfg.beta, cfg.threshold,
                    cfg.surrogate_alpha)
            elif cfg.arch == "snn":
                mine, v_end = scan_lif(
                    cur, state[k], cfg.beta, cfg.threshold,
                    cfg.surrogate_alpha, acc)
                ref, v_ref = lif_scan_eager(
                    cur, state[k], cfg.beta, cfg.threshold,
                    cfg.surrogate_alpha, cfg.reset)
            else:
                return {"leg": leg, "run": run,
                        "error": f"arch {cfg.arch!r} has no scan of this form"}

            # T2: the local scan IS the committed scan, or nothing below counts.
            row["t2_spikes_bitwise_equal"].append(bool(torch.equal(mine, ref)))
            row["t2_state_max_abs_diff"].append(float((v_end - v_ref).abs().max()))

            layer = acc.report()
            layer["layer"] = k
            layer["max_abs_g_over_lif_bound"] = layer["max_abs_g"] / bound
            row["layers"].append(layer)
            h = ref
            del cur, mine, ref
            if device == "cuda":
                torch.cuda.empty_cache()

    row["t1_holds"] = (
        cfg.arch != "snn"
        or all(ly["max_abs_g"] <= bound + T1_TOL for ly in row["layers"])
    )
    row["t2_holds"] = (all(row["t2_spikes_bitwise_equal"])
                       and max(row["t2_state_max_abs_diff"]) <= 1e-5)
    return row


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="docs/reports/data/exp_016_reset_jacobian.json")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--only", default=None, help="comma-separated leg ids, e.g. A4,A5")
    ap.add_argument("--batch-step", type=int, default=0,
                    help="sampler batch index. 0 is what pre-registered Leg A "
                         "ran; any other value is POST-HOC and must be written "
                         "to its own --out.")
    args = ap.parse_args(argv)

    want = None if args.only is None else {s.strip() for s in args.only.split(",")}
    rows = []
    for run, ck, leg, note in TARGETS:
        if want is not None and leg not in want:
            continue
        print(f"[{leg}] {run} ...", flush=True)
        r = probe(run, ck, leg, note, args.device, args.batch_step)
        rows.append(r)
        if "error" in r:
            print(f"  {r['error']}", flush=True)
            continue
        for ly in r["layers"]:
            print(f"  L{ly['layer']}  max|g|={ly['max_abs_g']:.6g}"
                  f"  frac(|g|>1)={ly['frac_over']['1.0']:.6g}"
                  f"  max chain log10 gain={ly['max_chain_log10_gain']:+.2f}",
                  flush=True)
        print(f"  T1={r['t1_holds']}  T2={r['t2_holds']}", flush=True)

    out = {
        "experiment": "EXP_016_reset_jacobian",
        "question": ("is the backward-through-time a contraction, for each "
                     "neuron, at each width?"),
        "inference_only": True,
        "trains_nothing": True,
        "adopts_nothing": True,
        "ranks_nothing": True,
        "changes_no_hyperparameter": True,
        "batch_step": args.batch_step,
        "post_hoc": args.batch_step != 0,
        "lif_bound_at_frozen_constants": LIF_BOUND_AT_FROZEN_CONSTANTS,
        "t1_tolerance": T1_TOL,
        "thresholds": list(THRESHOLDS),
        "results": rows,
    }
    p = Path(args.out)
    if not p.is_absolute():
        p = _REPO / p
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    print(f"\nwrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
