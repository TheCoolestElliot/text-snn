"""EXP_012, POST-HOC: why does the LIF's decision variable pile up at the threshold?

**This diagnostic is post-hoc and is labelled so everywhere it is quoted.** Its
hypothesis was formed *after* reading EXP_012 leg 3's density ladder, which is the
opposite order from the rest of that experiment. It is committed because it
produced a load-bearing number, not because it was planned. `EXP_008` section 9.5
is the precedent -- a chase that follows a result rather than preceding it -- and
`EXP_011`'s divergence rule is the precedent for saying so in the artifact rather
than letting the ordering be inferred.

WHAT THE LADDER SHOWED, AND WHAT IT DID NOT EXPLAIN
----------------------------------------------------
`P(|u| < h)` for the two-compartment arm is linear in `h` across four decades --
a smooth density, `rho_u(0) ~ 0.17` per unit `u`. For the LIF arm it grows only
~9x across the same four decades while sitting 1800-6200x higher at `h = 1e-6`.
That is not a higher smooth density; it is mass concentrated essentially *at* the
threshold. Nothing in leg 3 says why.

THE HYPOTHESIS
--------------
Layer 0's input is one of only `V = 205` distinct embedding rows, so `cur` takes
205 distinct values per channel. The LIF's hard reset sets `v` to **exactly 0**,
so the step after any spike has `v_pre = cur` -- a value drawn from that small
discrete set. Any channel whose `cur` happens to land near `thr` therefore
reproduces the **same** near-threshold `u`, every time that token follows a spike
in that channel. The two-compartment neuron never resets `vs`, so
`u = vf + w*vs` carries a continuously-varying history and cannot land on the
same value twice.

WHAT WOULD FALSIFY IT
---------------------
Near-threshold sites that are (a) not enriched for post-spike steps, (b) spread
over many distinct `u` values rather than a few repeated ones, or (c) spread
evenly over channels. The repeat count is the discriminating one: both neurons
can be enriched for post-spike steps without either replaying a value.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import torch

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO / "src") not in sys.path:
    sys.path.insert(0, str(_REPO / "src"))

from snn.data import Corpus, windowed_eval_batches  # noqa: E402
from snn.prescan import threshold_gain  # noqa: E402

_SPEC = importlib.util.spec_from_file_location(
    "chase012", _REPO / "scripts" / "exp" / "012_chase_fold_gap.py")
chase = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(chase)


@torch.no_grad()
def pileup(run: str, half_width: float) -> dict:
    arm, base, cfg, target = chase._load(run)
    corpus = Corpus(cfg.corpus, cfg.data_dir)
    x, _ = next(iter(windowed_eval_batches(corpus.split("test"),
                                           cfg.batch_size, cfg.seq_len)))
    idx = x.to(cfg.device)
    h = arm.embed(idx)
    cur = threshold_gain(arm._project(arm.layers[0], h), arm.thr_log[0])
    sp, u = chase.decision_variable(cur, **chase._neuron_kwargs(arm, cfg, 0, target))

    near = u.abs() < half_width
    n_near = int(near.sum())
    prev = torch.zeros_like(sp)
    prev[:, 1:] = sp[:, :-1]
    overall = float(prev.mean())
    out = {
        "run": run,
        "neuron": "lif" if target == "snn" else "twocomp",
        "half_width": half_width,
        "n_sites": int(u.numel()),
        "n_near_threshold": n_near,
        "post_spike_fraction_overall": overall,
        "post_spike_fraction_near": (float(prev[near].mean()) if n_near else None),
        "n_distinct_u_values_near": (int(torch.unique(u[near]).numel())
                                     if n_near else 0),
        "n_channels_carrying_them": int((near.sum(dim=(0, 1)) > 0).sum()),
        "max_per_channel": int(near.sum(dim=(0, 1)).max()),
        "d_model": int(u.shape[2]),
    }
    if n_near and out["n_distinct_u_values_near"]:
        out["repeats_per_distinct_value"] = (
            n_near / out["n_distinct_u_values_near"])
        out["post_spike_enrichment"] = (out["post_spike_fraction_near"]
                                        / overall)
    del arm, base, u, sp, cur
    torch.cuda.empty_cache()
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", default="threshold_s0,threshold_s1,threshold_s2,"
                                      "compose_s1,compose_s2")
    ap.add_argument("--half-width", type=float, default=1e-6)
    ap.add_argument("--out", default="docs/reports/data/exp_012_pileup.json")
    args = ap.parse_args(argv)

    report = {
        "experiment": "EXP_012_fold_gap",
        "status": "POST-HOC -- hypothesis formed after reading leg 3's ladder",
        "question": "why is the LIF's near-threshold mass concentrated rather "
                    "than smoothly distributed?",
        "runs": [pileup(r.strip(), args.half_width)
                 for r in args.runs.split(",") if r.strip()],
    }
    for e in report["runs"]:
        print(f"\n{e['run']:14s} ({e['neuron']})")
        print(f"  |u| < {e['half_width']:.0e} at {e['n_near_threshold']:>10,} "
              f"of {e['n_sites']:,} sites")
        if e["n_near_threshold"]:
            print(f"  previous step spiked      {e['post_spike_fraction_near']:.4f}"
                  f"  (baseline {e['post_spike_fraction_overall']:.4f}, "
                  f"{e['post_spike_enrichment']:.2f}x)")
            print(f"  distinct u values         {e['n_distinct_u_values_near']:>10,}"
                  f"  ({e['repeats_per_distinct_value']:.1f} repeats each)")
            print(f"  channels carrying them    {e['n_channels_carrying_them']:>10,}"
                  f" of {e['d_model']}  (top holds {e['max_per_channel']:,})")

    out = _REPO / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nWROTE {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
