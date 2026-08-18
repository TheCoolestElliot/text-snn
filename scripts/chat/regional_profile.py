"""The weight-space regional profile of a two-compartment checkpoint.

    python scripts/chat/regional_profile.py experiments/chat/*/ckpt_best.pt

ZERO TRAINING, ZERO GPU. Reads `beta_s_raw` and `w` off a `state_dict` and
reports, per layer, the realised slow-pole timescale distribution and whether the
slow channels actually carry any weight into the membrane the threshold reads.

Not part of the Phase-1..5 research protocol. No number it prints is a reported
figure.

WHAT THIS IS, AND WHAT IT IS NOT
--------------------------------
It is a *weight-space* profile. It locates a region, in the sense of which
`(layer, channel-block)` rectangle holds long time constants **and** a
non-negligible coefficient into the membrane.

It is **not** evidence that the region does anything. This project's standard for
that is `EXP_006`'s: a matched-size, matched-displacement, disjoint-selection-rule
clamping study with a dose-response curve. This script tells that study where to
point; it does not substitute for it, and a claim of the form "layer 1 is the
working-memory region" may not be sourced to this file alone.

WHY |w| MATTERS AS MUCH AS tau
------------------------------
The neuron is

    v_t = vf_t + w_c * vs_t

so a channel with a 600-character time constant and `w_c = 0` is **inert**: `vs`
never reaches the threshold comparison. `w_c = 0` nests the plain Phase-2 LIF
*exactly* -- `tests/test_twocomp_equivalence.py::test_w_zero_is_the_committed_lif_scan`
asserts it bit-for-bit in both the forward and the backward -- so a slow channel
sitting at `|w| ~ 0.01` has effectively reverted to the baseline neuron and
reporting its tau alone would overstate what the model retains.

WHAT IT FOUND, 2026-08-17
-------------------------
Across 26 committed chat checkpoints with a slow population, spanning two
*independently* trained lineages (`chat-v2` and `chat-v6-scratch`):

  * layer 0's slow channels are **load-bearing in every one** -- median |w| ratio
    4.99 to 10.11 against the fast channels;
  * the deepest two layers' are **switched off in every one** -- 0.07 to 0.21.

The two fresh lineages disagree about *where the slow peak sits* (layer 1 vs
layer 3), so that shape is not a universal outcome and should not be quoted as
one. The load-bearing dissociation is what replicated.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO / "src") not in sys.path:
    sys.path.insert(0, str(_REPO / "src"))

TAU_SLOW = 100.0  # characters; the "is this a long-horizon channel" cut


def _spearman(a: torch.Tensor, b: torch.Tensor) -> float:
    """Rank correlation. Hand-rolled because scipy is not a project dependency."""
    ra = a.argsort().argsort().double()
    rb = b.argsort().argsort().double()
    ra = (ra - ra.mean()) / ra.std()
    rb = (rb - rb.mean()) / rb.std()
    return float((ra * rb).mean())


def profile(ckpt: Path, tau_slow: float = TAU_SLOW) -> dict:
    """Per-layer timescale and load-bearing summary for one checkpoint."""
    d = torch.load(ckpt, map_location="cpu", weights_only=False)
    sd = d.get("model", d)
    cfg = d.get("chat_config") or d.get("config") or {}
    layers = sorted(int(k.split(".")[-1]) for k in sd if k.startswith("beta_s_raw."))
    if not layers:
        raise SystemExit(
            f"{ckpt} has no `beta_s_raw.*` -- it is not a two-compartment "
            f"checkpoint, so it has no slow pole to profile."
        )

    qs = torch.tensor([0.1, 0.25, 0.5, 0.75, 0.9, 0.99], dtype=torch.float64)
    out: dict = {
        "checkpoint": str(ckpt),
        "step": d.get("step"),
        "best_val_bpc": d.get("best_val_bpc"),
        "seed": cfg.get("seed") if isinstance(cfg, dict) else None,
        "tau_slow_cut": tau_slow,
        "layers": [],
    }
    for k in layers:
        beta_s = torch.sigmoid(sd[f"beta_s_raw.{k}"].double().flatten())
        tau = 1.0 / (1.0 - beta_s).clamp_min(1e-12)
        w = sd[f"w.{k}"].double().flatten().abs()
        slow = tau > tau_slow
        q = torch.quantile(tau, qs).tolist()
        rec: dict = {
            "layer": k,
            "d": int(tau.numel()),
            "tau_p10": q[0], "tau_p25": q[1], "tau_p50": q[2],
            "tau_p75": q[3], "tau_p90": q[4], "tau_p99": q[5],
            "tau_max": float(tau.max()),
            "n_slow": int(slow.sum()),
            "frac_slow": float(slow.double().mean()),
            "median_abs_w_slow": float(w[slow].median()) if int(slow.sum()) else None,
            "median_abs_w_fast": float(w[~slow].median()) if int((~slow).sum()) else None,
            "spearman_tau_absw": _spearman(tau, w),
        }
        if rec["median_abs_w_slow"] and rec["median_abs_w_fast"]:
            rec["load_bearing_ratio"] = (
                rec["median_abs_w_slow"] / rec["median_abs_w_fast"]
            )
        out["layers"].append(rec)
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("ckpt", nargs="+", type=Path)
    p.add_argument("--tau-slow", type=float, default=TAU_SLOW,
                   help=f"channels above this tau count as slow (default {TAU_SLOW})")
    p.add_argument("--json", type=Path, default=None)
    a = p.parse_args(argv)

    results = []
    for c in a.ckpt:
        r = profile(c, a.tau_slow)
        results.append(r)
        print(f"\n{c}   seed={r['seed']}  step={r['step']}")
        print("  L | tau p50    p90     p99      max    | n(slow) "
              "| med|w| slow   fast    ratio | rho(tau,|w|)")
        for lyr in r["layers"]:
            ratio = lyr.get("load_bearing_ratio")
            rs = f"{ratio:6.2f}x" if ratio is not None else "    -- "
            ws = (f"{lyr['median_abs_w_slow']:.4f}"
                  if lyr["median_abs_w_slow"] is not None else "  --  ")
            wf = (f"{lyr['median_abs_w_fast']:.4f}"
                  if lyr["median_abs_w_fast"] is not None else "  --  ")
            print(f"  {lyr['layer']} | {lyr['tau_p50']:7.2f} {lyr['tau_p90']:7.2f} "
                  f"{lyr['tau_p99']:8.2f} {lyr['tau_max']:9.1f} |  {lyr['n_slow']:5d}  "
                  f"|   {ws}    {wf} {rs} | {lyr['spearman_tau_absw']:+.3f}")
        peak = max(r["layers"], key=lambda x: x["tau_p50"])["layer"]
        print(f"  -> slowest layer by median tau: L{peak} of {len(r['layers'])}")

    print("\nratio > 1: the slow channels carry MORE mix weight than the fast ones "
          "(load-bearing).\nratio < 1: the slow compartment is switched off in "
          "that layer.")
    print("This is a weight-space profile, NOT a causal ablation -- see the module "
          "docstring.")

    if a.json:
        a.json.parent.mkdir(parents=True, exist_ok=True)
        a.json.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"\nwrote {a.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
