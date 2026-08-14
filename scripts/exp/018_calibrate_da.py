"""EXP_018 -- what scale is the reward prediction error, so the squash can be set?

Run BEFORE `experiments/logs/EXP_018_dopamine.md` fixes `tau`, and committed with
its output, so that the squash scale is a calibrated choice rather than a round
number. `013_calibrate_noise_scale.py` is the precedent and the shape.

WHY THIS EXISTS AT ALL
----------------------
The arm broadcasts one scalar per (batch element, timestep):

    S_t   = -log p_{t-1}(x_t)      the surprise: how bad the outcome was   (nats)
    H_t   =  H(p_{t-1})            the model's OWN expected surprise       (nats)
    phi_t =  H_t - S_t             the reward prediction error             (nats)
    DA_t  =  tanh(phi_t / tau)     the bounded neuromodulator

`E_{x~p}[-log p(x)] = H(p)` exactly, so `phi` is zero-mean under the model's own
predictive distribution BY CONSTRUCTION -- no subtracted baseline, no EMA, no
learned critic, and (this is the part that decides the implementation) no
sequential scan over `t`, which would have cost L kernels per layer and broken
CUDA-graph capture.

`tau` is the one free constant and it only means something against the measured
spread of `phi`. A `tau` of 1.0 chosen by taste has the same defect
`013_calibrate_noise_scale.py`'s docstring names: no way to notice it is wrong.

So this measures `phi`'s distribution on the *committed* checkpoints, and the
pre-registration expresses `tau` as the measured standard deviation.

Two properties beyond the scale are measured because both change the design:

  * the **asymmetry**. `phi` is bounded above by `H_t` and unbounded below, so a
    raw `phi` through a multiplicative gain can invert the sign of the current.
    That is what makes `tanh` load-bearing rather than cosmetic.
  * the **lag-1 autocorrelation**. A signal that is white in time is one whose
    time-shuffled control has nearly identical statistics -- which makes the
    shuffle control tight rather than a straw man, and forbids selling the arm
    as a "temporal salience" mechanism without evidence.

`phi` is also measured at INITIALISATION, because a signal that is degenerate
before training is a new parameter with no gradient to ride early on, and that
is a limitation the pre-registration has to state rather than discover.

WHAT IT DOES NOT DO
-------------------
It reads no arm, scores no arm and touches no outcome. `phi` is a property of the
*baseline's own forward pass*. Nothing here can be read as evidence for or
against the hypothesis EXP_018 registers, which is why it is allowed to run
first.

THE SELF-CHECKS, WHICH ARE THE POINT OF F1 IN THIS PROJECT
-----------------------------------------------------------
`phi` is computed from a hand-written causal gather, and a hand-written index
into a committed loss is exactly the kind of second implementation that silently
stops being the first one (EXP_006's L5, EXP_012's S1). Worse, the specific way
it can be wrong -- an off-by-one in the shift -- is the way that would let the
arm read its own target, and it would still train.

So there are two, and both abort:

  C1  The surprise this file gathers must equal `F.cross_entropy` on the
      committed targets, ELEMENTWISE and exactly:

          -log_softmax(logits)[:, :-1].gather(x[:, 1:])  ==  CE(logits[:, :-1], y[:, :-1])

      `snn.data` builds `x = block[:, :-1]` and `y = block[:, 1:]`, so
      `y[:, u] == x[:, u+1]` and the identity holds for the CORRECT shift only.
      An off-by-one fails it by orders of magnitude, not by an ulp.

  C2  The mean surprise over the full val split, converted to bits, must
      reproduce the checkpoint's OWN committed `summary.json` val bpc -- a number
      this file did not produce. `CONTRIBUTING.md` §5: "Make every probe reproduce
      a number it did not produce." The residual is the one position per window
      that C1's slice drops, and it is reported rather than tolerated silently.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "src"))

from snn.config import Config, seed_everything  # noqa: E402
from snn.data import Corpus, windowed_eval_batches  # noqa: E402
from snn.metrics import bits_per_char  # noqa: E402
from snn.model import build_model  # noqa: E402

RUNS = _REPO / "experiments" / "runs"
OUT = _REPO / "docs" / "reports" / "data" / "exp_018_da_calibration.json"

#: Committed checkpoints to measure on. Three ARCHITECTURES rather than three
#: seeds of one, because the question `tau` answers is whether the scale is a
#: property of the task or of the neuron. EXP_005's standing lesson is that a
#: statistic measured on one neuron does not transfer, so it is measured on all
#: three that exist at 735K rather than assumed to.
CKPTS: tuple[tuple[str, str], ...] = (
    ("snn_beta0.5_s0", "snn"),
    ("snn_beta0.5_s1", "snn"),
    ("anchor_twocomp_d512_s0", "twocomp"),
    ("detach_d512_s0", "twocomp_detach"),
)

#: Seeds used for the at-initialisation leg. Untrained models, no checkpoint.
INIT_SEEDS: tuple[int, ...] = (0, 1, 2)

#: Batches for the distribution legs. The full val split is 2560 windows; 8
#: batches is 1024 windows * 255 positions = 261k samples per checkpoint, which
#: resolves a standard deviation far past the two significant figures `tau` is
#: quoted to. The C2 self-check runs the FULL split regardless.
N_BATCHES = 8

#: C1 is an exact identity between two ways of writing the same quantity, so it
#: is asserted at fp32 equality rather than at a tolerance. `log_softmax` then
#: `gather` and `cross_entropy` are not the same kernel, so exact bitwise
#: equality is not available; 1e-5 absolute on a quantity whose scale is ~1.5
#: nats is ~7 significant figures and is four orders of magnitude tighter than
#: any off-by-one could survive.
C1_TOL = 1e-5

#: C2 compares a mean over L-1 of L positions against a committed mean over all
#: L, so a residual of order (1/L) * (per-position spread) is EXPECTED and is not
#: a defect. At L=256 that is ~0.4% of bpc. The bar exists to catch a shift, not
#: to certify agreement to the last digit.
C2_TOL_BPC = 0.05


def _phi(logits: torch.Tensor, idx: torch.Tensor) -> dict[str, torch.Tensor]:
    """The RPE and its two terms, for positions 1..L-1. Causal by construction.

    `logits` [B, L, V] and `idx` [B, L]. The prediction made AT position u is
    scored against the character observed AT u+1, which is `idx[:, u+1]` -- a
    character the model has ALREADY READ by the time position u+1 is processed.
    Nothing here touches `y`, and the returned tensors are indexed so that entry
    `t` may only be consumed at position `t`.
    """
    logp = F.log_softmax(logits.float(), dim=-1)
    p = logp.exp()
    H = -(p * logp).sum(-1)[:, :-1]                                   # [B, L-1]
    S = -logp[:, :-1].gather(-1, idx[:, 1:].unsqueeze(-1)).squeeze(-1)  # [B, L-1]
    return {"H": H, "S": S, "phi": H - S}


def _stats(t: torch.Tensor) -> dict:
    q = torch.tensor([0.001, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 0.999],
                     device=t.device, dtype=torch.float32)
    return {
        "n": int(t.numel()),
        "mean": float(t.mean()), "sd": float(t.std()),
        "min": float(t.min()), "max": float(t.max()),
        "quantiles": {f"{a:.3f}": float(b) for a, b in
                      zip(q.tolist(), torch.quantile(t.float(), q).tolist())},
    }


def _autocorr(x: torch.Tensor, lags: tuple[int, ...] = (1, 2, 4, 8)) -> dict:
    """Lag-k autocorrelation ALONG TIME, computed per window and then pooled.

    Flattening first would wrap the correlation across window boundaries and
    report a number about the batch layout rather than about the signal.
    """
    d = (x - x.mean()).float()
    denom = float(d.pow(2).mean())
    return {str(k): float((d[:, :-k] * d[:, k:]).mean()) / denom for k in lags}


def _measure(model, corpus, cfg, n_batches: int) -> tuple[dict, dict]:
    """Distribution legs plus the C1 self-check, on `n_batches` val batches."""
    das, sur, ent = [], [], []
    ac_accum: list[dict] = []
    c1_max_abs = 0.0
    with torch.no_grad():
        for i, (x, y) in enumerate(
            windowed_eval_batches(corpus.split("val"), cfg.batch_size, cfg.seq_len)
        ):
            if i >= n_batches:
                break
            x = x.to(cfg.device)
            y = y.to(cfg.device)
            logits, _, _ = model(x, None)
            f = _phi(logits, x)

            # --- C1: the gather IS the committed loss, on the correct shift ---
            ce = F.cross_entropy(
                logits[:, :-1].reshape(-1, logits.shape[-1]).float(),
                y[:, :-1].reshape(-1),
                reduction="none",
            ).reshape(f["S"].shape)
            c1_max_abs = max(c1_max_abs, float((ce - f["S"]).abs().max()))

            das.append(f["phi"].flatten())
            sur.append(f["S"].flatten())
            ent.append(f["H"].flatten())
            ac_accum.append(_autocorr(f["phi"]))

    phi = torch.cat(das)
    keys = ac_accum[0].keys()
    out = {
        "phi": _stats(phi),
        "surprise": _stats(torch.cat(sur)),
        "expected_surprise_H": _stats(torch.cat(ent)),
        "phi_autocorr": {k: statistics.mean(a[k] for a in ac_accum) for k in keys},
        "frac_phi_positive": float((phi > 0).float().mean()),
        "n_batches": n_batches,
    }
    c1 = {"max_abs_diff_vs_cross_entropy": c1_max_abs,
          "tol": C1_TOL, "passed": c1_max_abs <= C1_TOL}
    return out, c1


def _c2_full_split(model, corpus, cfg, committed_bpc: float | None) -> dict:
    """Reproduce the checkpoint's own committed val bpc from the gathered surprise.

    A number this file did not produce (`CONTRIBUTING.md` §5). The residual is
    structural -- this sums L-1 of L positions per window -- and is reported.
    """
    total, n = 0.0, 0
    with torch.no_grad():
        for x, _y in windowed_eval_batches(corpus.split("val"), cfg.batch_size,
                                           cfg.seq_len):
            x = x.to(cfg.device)
            logits, _, _ = model(x, None)
            s = _phi(logits, x)["S"]
            total += float(s.sum())
            n += int(s.numel())
    bpc = bits_per_char(total, n)
    row = {"reconstructed_val_bpc_fresh": bpc, "n_positions": n,
           "committed_val_bpc_fresh": committed_bpc, "tol": C2_TOL_BPC}
    if committed_bpc is None or not math.isfinite(committed_bpc):
        # An ABSENT comparison, not a failed one, and the distinction is the
        # `017_detach_results.py` rule that a diverged run resolves NOT RUN and
        # never UNRESOLVED. `anchor_twocomp_d512_s0` is the live case: its
        # `ckpt_best.pt` is a healthy pre-divergence checkpoint and `phi` is
        # measured on it perfectly well, while the `summary.json` val block it
        # would be compared against is NaN because the run died later. Scoring
        # that as a self-check failure would report a defect in this probe that
        # is a fact about a different run. It is recorded, named, and not
        # counted -- which is not the same as widening a tolerance, because no
        # tolerance is involved: there is no finite number on the other side.
        row["passed"] = None
        row["note"] = (
            "no finite committed val bpc to compare against"
            if committed_bpc is None else
            "committed val bpc is non-finite (the run diverged after ckpt_best); "
            "the reference is absent, so C2 is NOT RUN for this checkpoint"
        )
    else:
        row["abs_diff"] = abs(bpc - committed_bpc)
        row["passed"] = row["abs_diff"] <= C2_TOL_BPC
    return row


def _committed_val_bpc(run: str) -> float | None:
    p = RUNS / run / "summary.json"
    if not p.exists():
        return None
    try:
        return float(json.loads(p.read_text(encoding="utf-8"))["val"]["fresh"]["bpc"])
    except (KeyError, TypeError, ValueError):
        return None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--n-batches", type=int, default=N_BATCHES)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--skip-c2", action="store_true",
                    help="skip the full-split reproduction (minutes, not seconds)")
    args = ap.parse_args(argv)

    trained: list[dict] = []
    failures: list[str] = []

    for run, arch in CKPTS:
        path = RUNS / run / "ckpt_best.pt"
        if not path.exists():
            print(f"(missing) {run}", flush=True)
            continue
        ck = torch.load(path, map_location=args.device, weights_only=False)
        cfg = Config(**ck["config"])
        cfg.device = args.device
        cfg.data_dir = str(_REPO / cfg.data_dir)
        seed_everything(cfg.seed, cfg.deterministic)
        model = build_model(cfg)
        model.load_state_dict(ck["model"])
        model.eval()
        corpus = Corpus(cfg.corpus, cfg.data_dir)

        row, c1 = _measure(model, corpus, cfg, args.n_batches)
        row.update({"run": run, "arch": arch, "c1_gather_is_cross_entropy": c1})
        if not c1["passed"]:
            failures.append(f"C1 failed on {run}: {c1['max_abs_diff_vs_cross_entropy']:.3e}")
        if not args.skip_c2:
            c2 = _c2_full_split(model, corpus, cfg, _committed_val_bpc(run))
            row["c2_reproduces_committed_bpc"] = c2
            if c2["passed"] is False:
                failures.append(f"C2 failed on {run}: {c2['abs_diff']:.4f} bpc")
        trained.append(row)
        print(f"{run:<28s} phi sd {row['phi']['sd']:.4f}  mean {row['phi']['mean']:+.4f}  "
              f"min {row['phi']['min']:+.2f}  max {row['phi']['max']:+.2f}  "
              f"frac>0 {row['frac_phi_positive']:.3f}  ac1 {row['phi_autocorr']['1']:+.4f}",
              flush=True)

    at_init: list[dict] = []
    for seed in INIT_SEEDS:
        cfg = Config(arch="snn", device=args.device, seed=seed,
                     data_dir=str(_REPO / "data"))
        seed_everything(cfg.seed, cfg.deterministic)
        corpus = Corpus(cfg.corpus, cfg.data_dir)
        cfg.vocab_size = int(corpus.vocab_size)
        model = build_model(cfg)
        model.eval()
        row, c1 = _measure(model, corpus, cfg, min(2, args.n_batches))
        row.update({"seed": seed, "arch": "snn", "state": "initialisation",
                    "c1_gather_is_cross_entropy": c1})
        if not c1["passed"]:
            failures.append(f"C1 failed at init seed {seed}")
        at_init.append(row)
        print(f"init s{seed:<24d} phi sd {row['phi']['sd']:.4f}  "
              f"mean {row['phi']['mean']:+.4f}", flush=True)

    if not trained:
        raise SystemExit("no committed checkpoint found; nothing to calibrate against")

    sds = [r["phi"]["sd"] for r in trained]
    tau = statistics.median(sds)
    init_sds = [r["phi"]["sd"] for r in at_init]

    summary = {
        "artifact": "exp_018_da_calibration",
        "purpose": "set EXP_018's squash scale tau against the measured spread of "
                   "the reward prediction error it squashes",
        "signal": "phi_t = H(p_{t-1}) - (-log p_{t-1}(x_t)), nats; DA = tanh(phi/tau)",
        "causal": "phi_t depends on x_{<=t} only; the target at t is x_{t+1}",
        "tau_recommended": round(tau, 4),
        "tau_basis": "median over committed checkpoints of sd(phi)",
        "sd_phi_over_checkpoints": {"min": min(sds), "median": tau, "max": max(sds),
                                    "spread_frac": (max(sds) - min(sds)) / tau},
        "sd_phi_at_initialisation": {"seeds": list(INIT_SEEDS),
                                     "values": init_sds,
                                     "median": statistics.median(init_sds)},
        "ratio_trained_to_init_sd": tau / statistics.median(init_sds),
        "self_checks_passed": not failures,
        "self_check_failures": failures,
        "trained": trained,
        "at_initialisation": at_init,
    }
    out = Path(args.out)
    if not out.is_absolute():
        out = _REPO / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print(f"\ntau (median sd of phi over {len(trained)} checkpoints) = {tau:.4f}")
    print(f"spread across checkpoints: {summary['sd_phi_over_checkpoints']['spread_frac']:.1%}")
    print(f"trained : init sd ratio  = {summary['ratio_trained_to_init_sd']:.1f}x")
    print(f"self-checks passed: {summary['self_checks_passed']}")
    for f in failures:
        print(f"  FAIL {f}")
    print(f"written: {out}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
