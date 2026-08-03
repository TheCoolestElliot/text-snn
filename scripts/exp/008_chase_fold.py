"""EXP_008: W1 failed on one seed of three. Chase it before issuing any verdict.

W1 required the folded model to reproduce the arm's own fresh test bpc to
< 1e-3 bpc. Measured: **1.86e-3** on `threshold_s0`, 5.42e-4 on `s1`, 9.47e-4 on
`s2`. EXP_008 §4's rule is that a failed fold means §1.1's algebra or its
implementation is wrong, and that the tolerance is not widened. So this script
does not widen it. It asks a narrower question the bpc comparison cannot answer:

    Is the residual a broken identity, or is it fp32 rounding amplified by a
    discontinuity?

WHY THE bpc COMPARISON CANNOT SETTLE THAT ON ITS OWN
-----------------------------------------------------
Identity 2 says `g*(W·h + b)` and `(g*W)·h + g*b` are the same function. They are
not the same *floating-point program*: the GEMM accumulates 512 products in a
different order once the rows are scaled. That perturbs each current by ~1 ulp.
Between that perturbation and the reported bpc sits `1[v >= thr]` -- a
discontinuity. Any membrane within an ulp of the threshold flips its spike, that
spike is an input to the next layer's GEMM, and the error is no longer at
rounding scale.

This is `EXP_004` §9.2's lesson in a new place: a contract is only guarded where
the quantity it constrains is observed. W1 observes bpc, four transformations
downstream of the quantity the identity is about.

WHAT IS MEASURED, IN THE ORDER THAT SEPARATES THE TWO EXPLANATIONS
-------------------------------------------------------------------
1. **The identity, on the quantity it is actually about.** `g*(W·h + b)` against
   `(g*W)·h + g*b`, on a real batch, in fp32 **and in float64**. If the algebra
   is wrong the fp64 difference stays; if it is rounding, fp64 collapses it by
   ~9 orders of magnitude. This is the decisive leg and it needs no model.
2. **The spike disagreement rate**, per layer -- how many of the ~1.7e7 spikes per
   batch flip, and how far the membrane sat from the threshold when they did.
3. **The logits difference**, and the resulting per-batch bpc difference, so the
   chain from "1 ulp on the current" to "1.9e-3 bpc" is quantified end to end
   rather than asserted.

WHAT THIS SCRIPT DOES NOT DO
----------------------------
It does not change W1's verdict, and it does not propose a tolerance. If the
answer is "rounding through a discontinuity", the right bar for a fold-in check is
a different quantity from the one F1 uses -- F1 compares two evaluations of the
*same* weights, where no spike can flip, so it was the wrong precedent to borrow.
Redefining a pre-registered threshold after seeing it fail is Elliot's call, and
`EXP_005` §9.4's handling of the §6.2 bar is the precedent: report both numbers,
refer the change, apply nothing.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO / "src") not in sys.path:
    sys.path.insert(0, str(_REPO / "src"))

from snn.config import Config  # noqa: E402
from snn.data import Corpus, windowed_eval_batches  # noqa: E402
from snn.metrics import bits_per_char  # noqa: E402
from snn.model import build_model  # noqa: E402

RUNS = _REPO / "experiments" / "runs"


def _load(run: str):
    ck = torch.load(RUNS / run / "ckpt_final.pt", map_location="cpu",
                    weights_only=False)
    known = {f.name for f in dataclasses.fields(Config)}
    cfg = Config(**{k: v for k, v in ck["config"].items() if k in known})
    arm = build_model(cfg)
    arm.load_state_dict(ck["model"])
    arm.eval()
    base_cfg = dataclasses.replace(cfg, arch="snn")
    base = build_model(base_cfg)
    base.load_state_dict(arm.fold_into_spiking_state_dict(), strict=True)
    base.eval()
    return arm, base, cfg


@torch.no_grad()
def identity_on_the_current(arm, base, h: torch.Tensor, layer: int) -> dict:
    """Leg 1. The identity on the quantity it is about, at three precisions.

    Three comparisons, and the point is the *difference between them* -- each one
    removes one source of floating-point error, so the residual that survives all
    three is the only one that could be the algebra.

      A  as shipped         fp32 storage + fp32 GEMM, in both orderings
      B  fp32 fold, fp64 GEMM  the fp32-rounded folded weight, evaluated exactly
      C  fp64 fold, fp64 GEMM  the fold computed in float64 as well

    B still carries error, and that is not a surprise once stated: the folded
    weight `g*W` is *stored* in fp32 by `fold_into_spiking_state_dict`, so B
    measures that rounding with the GEMM's removed. Only C tests the identity
    itself.

    Layer 0 only, deliberately. Layer 1's input is layer 0's spike train, which
    the two models do not agree on bit-for-bit (leg 2 measures by how much), so a
    layer-1 comparison would mix the identity with the disagreement it causes and
    could not answer either question.
    """
    lin_a, lin_b = arm.layers[layer], base.layers[layer]
    g = arm.input_gain(layer)                       # [1, d]

    scaled_after = g * lin_a(h)                     # g*(W·h + b)
    folded_before = lin_b(h)                        # (fp32(g*W))·h + fp32(g*b)
    dA = (scaled_after - folded_before).abs()

    hd, gd = h.double(), g.double()
    a64 = gd * F.linear(hd, lin_a.weight.double(), lin_a.bias.double())
    dB = (a64 - F.linear(hd, lin_b.weight.double(), lin_b.bias.double())).abs()
    exact = F.linear(hd, lin_a.weight.double() * gd.reshape(-1, 1),
                     lin_a.bias.double() * gd.flatten())
    dC = (a64 - exact).abs()

    cur_max = float(scaled_after.abs().max())
    return {
        "layer": layer,
        "current_abs_max": cur_max,
        "A_shipped_fp32_max_abs_diff": float(dA.max()),
        "A_shipped_fp32_mean_abs_diff": float(dA.mean()),
        "A_in_units_of_current_max": float(dA.max()) / cur_max,
        "B_fp32fold_fp64gemm_max_abs_diff": float(dB.max()),
        "B_in_units_of_current_max": float(dB.max()) / cur_max,
        "C_exact_fold_fp64_max_abs_diff": float(dC.max()),
        "C_in_units_of_current_max": float(dC.max()) / cur_max,
        "eps_fp32": float(torch.finfo(torch.float32).eps),
        "eps_fp64": float(torch.finfo(torch.float64).eps),
    }


@torch.no_grad()
def spikes_and_logits(arm, base, idx: torch.Tensor) -> dict:
    """Legs 2 and 3. Spike disagreement per layer, and what it costs downstream."""
    arm.keep_spikes = base.keep_spikes = True
    try:
        la, _, aux_a = arm(idx, None)
        lb, _, aux_b = base(idx, None)
    finally:
        arm.keep_spikes = base.keep_spikes = False

    per_layer = []
    for k, (sa, sb) in enumerate(zip(aux_a["spikes"], aux_b["spikes"])):
        diff = (sa != sb)
        per_layer.append({
            "layer": k,
            "n_spike_sites": int(sa.numel()),
            "n_disagreements": int(diff.sum()),
            "disagreement_fraction": float(diff.float().mean()),
            "arm_firing_rate": float(sa.mean()),
            "folded_firing_rate": float(sb.mean()),
        })
    dl = (la - lb).abs()
    return {
        "per_layer": per_layer,
        "logits_max_abs_diff": float(dl.max()),
        "logits_mean_abs_diff": float(dl.mean()),
        "logits_abs_max": float(la.abs().max()),
    }


@torch.no_grad()
def subset_bpc(model, data, cfg: Config, n_windows: int) -> float:
    total, chars = 0.0, 0
    seen = 0
    for x, y in windowed_eval_batches(data, cfg.batch_size, cfg.seq_len):
        x = x.to(cfg.device)
        y = y.to(cfg.device)
        logits, _, _ = model(x, None)
        total += float(F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]).float(), y.reshape(-1),
            reduction="sum"))
        chars += y.numel()
        seen += int(x.shape[0])
        if seen >= n_windows:
            break
    return bits_per_char(total, chars)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", default="threshold_s0,threshold_s1,threshold_s2")
    ap.add_argument("--windows", type=int, default=512)
    ap.add_argument("--out", default="docs/reports/data/exp_008_fold_residual.json")
    args = ap.parse_args(argv)

    report = {"experiment": "EXP_008_learned_threshold",
              "question": "is W1's residual a broken identity or fp32 rounding "
                          "through a discontinuity?",
              "w1_tolerance_bpc": 1e-3, "runs": {}}

    for run in [r.strip() for r in args.runs.split(",") if r.strip()]:
        arm, base, cfg = _load(run)
        corpus = Corpus(cfg.corpus, cfg.data_dir)
        data = corpus.split("test")
        x, _y = next(iter(windowed_eval_batches(data, cfg.batch_size, cfg.seq_len)))
        idx = x.to(cfg.device)
        h = arm.embed(idx).detach()

        entry = {
            "identity_on_current": [identity_on_the_current(arm, base, h, 0)],
            "spikes": spikes_and_logits(arm, base, idx),
            "subset_windows": args.windows,
            "subset_bpc_arm": subset_bpc(arm, data, cfg, args.windows),
            "subset_bpc_folded": subset_bpc(base, data, cfg, args.windows),
        }
        entry["subset_bpc_residual"] = abs(entry["subset_bpc_arm"]
                                           - entry["subset_bpc_folded"])
        report["runs"][run] = entry

        i0 = entry["identity_on_current"][0]
        print(f"\n{run}   |cur|max {i0['current_abs_max']:.2f}")
        print(f"  leg 1  identity on cur (layer 0), max |diff| as a fraction of "
              f"|cur|max:")
        print(f"           A as shipped (fp32 fold, fp32 GEMM)  "
              f"{i0['A_shipped_fp32_max_abs_diff']:.3e}  "
              f"= {i0['A_in_units_of_current_max']:.2e}   "
              f"({i0['A_in_units_of_current_max'] / i0['eps_fp32']:.1f} fp32 eps)")
        print(f"           B fp32 fold, fp64 GEMM               "
              f"{i0['B_fp32fold_fp64gemm_max_abs_diff']:.3e}  "
              f"= {i0['B_in_units_of_current_max']:.2e}")
        print(f"           C exact fp64 fold                    "
              f"{i0['C_exact_fold_fp64_max_abs_diff']:.3e}  "
              f"= {i0['C_in_units_of_current_max']:.2e}   "
              f"({i0['C_in_units_of_current_max'] / i0['eps_fp64']:.1f} fp64 eps)")
        for ly in entry["spikes"]["per_layer"]:
            print(f"  leg 2  layer {ly['layer']}: {ly['n_disagreements']} spike "
                  f"flips in {ly['n_spike_sites']:,} "
                  f"({ly['disagreement_fraction']:.3e}), rates "
                  f"{ly['arm_firing_rate']:.6f} vs {ly['folded_firing_rate']:.6f}")
        print(f"  leg 3  logits max |diff| "
              f"{entry['spikes']['logits_max_abs_diff']:.3e} on |logits| max "
              f"{entry['spikes']['logits_abs_max']:.1f};  "
              f"{args.windows}-window bpc {entry['subset_bpc_arm']:.6f} vs "
              f"{entry['subset_bpc_folded']:.6f}  "
              f"(residual {entry['subset_bpc_residual']:.2e})")
        del arm, base
        torch.cuda.empty_cache()

    out = _REPO / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nWROTE {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
