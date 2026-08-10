"""EXP_015: is the two-compartment membrane what grows with width?

    python scripts/exp/015_chase_state_growth.py --out docs/reports/data/exp_015_state_growth.json

WHY THIS EXISTS
---------------
`011_chase_compose_divergence.py` bounded both of this project's unexplained
divergences to the same place: **layer 0's backward**, on both the fused and the
eager path, with the forward and the loss finite and the fp32/fp64 gradient norms
in agreement.  `EXP_011` §10.4 stopped there and called the arithmetic unknown.

Running it a second time -- on `arch_twocomp_d1481_s0`, a different arm at a
different width dying 12,460 steps earlier -- produced the *same five tensors*
and one difference worth measuring rather than eyeballing:

    quantity        compose_s0 (d=512)      arch_twocomp_d1481_s0 (d=1481)
    param  max      4.145                   4.226
    logits max      ~226                    ~250
    state  max      [~30, ~66]              [~200, ~700]      <-- ~10x

Weights and logits are comparable between the two.  The membrane is not.  So the
hypothesis this file tests is narrow and falsifiable:

    **The two-compartment state magnitude grows with width, and the plain LIF's
    does not.**

If that holds, the arms that die and the arm that survives are separated by a
measured quantity rather than by an unknown one -- which does not identify the
failing arithmetic, but does say what to instrument next and why the frozen
recipe stopped transferring.  If it fails -- if the plain LIF's state is just as
large at `d = 1481` -- then state magnitude is not the discriminator and this
note says so.

WHAT IT IS NOT
--------------
Inference only, on committed checkpoints, one batch each, no training, no bar and
no verdict.  It cannot say the state growth *causes* the NaN; it can only say
whether the two regimes differ in it.  `EXP_015` §10 refers the causal question.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

import torch

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO / "src") not in sys.path:
    sys.path.insert(0, str(_REPO / "src"))

from snn.config import Config, seed_everything  # noqa: E402
from snn.data import Corpus, RandomWindowSampler  # noqa: E402
from snn.model import build_model  # noqa: E402

RUNS = _REPO / "experiments" / "runs"

#: (run, checkpoint, what it is). The comparison only means something if the
#: surviving and dying runs are read the same way, on the same batch, so the
#: baseline width and the wide width appear for BOTH neurons.
TARGETS = [
    ("snn_beta0.5_s0", "ckpt_final.pt", "plain LIF, 735K -- trains"),
    ("scale_d1481_s0", "ckpt_final.pt", "plain LIF, 5.0M -- trains"),
    ("twocomp_s0", "ckpt_final.pt", "two-compartment, 735K -- trains"),
    ("arch_twocomp_d1481_s0", "ckpt_best.pt",
     "two-compartment, 5.0M -- DIES 138 steps later"),
]


def probe(run: str, ckpt_name: str, note: str, device: str) -> dict:
    ckpt_path = RUNS / run / ckpt_name
    if not ckpt_path.exists():
        return {"run": run, "error": f"missing {ckpt_name}"}

    raw = json.loads((RUNS / run / "config.json").read_text(encoding="utf-8"))
    known = {f.name for f in dataclasses.fields(Config)}
    cfg = Config(**{k: v for k, v in raw.items() if k in known})
    cfg.device = device
    # Eager, so the probe reads the same arithmetic on any machine and the
    # numbers do not depend on whether a graph captured.
    cfg.fused = False
    cfg.cuda_graph = False

    seed_everything(cfg.seed, cfg.deterministic)
    model = build_model(cfg)
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ck["model"])
    model.eval()

    corpus = Corpus(cfg.corpus, cfg.data_dir)
    # Batch 0 of the run's own sampler: the same windows it trained on, so this
    # is the state the model actually sees rather than an arbitrary slice.
    x, _ = RandomWindowSampler(
        corpus.split("train"), cfg.batch_size, cfg.seq_len, cfg.seed).batch(0)
    x = x.to(device)

    with torch.no_grad():
        logits, state, _ = model(x, None)

    row = {
        "run": run,
        "note": note,
        "arch": cfg.arch,
        "d_model": cfg.d_model,
        "step_of_checkpoint": int(ck.get("global_step", -1)),
        "logits_abs_max": float(logits.abs().max()),
        "state_abs_max_per_layer": [float(s.abs().max()) for s in state],
        "state_abs_mean_per_layer": [float(s.abs().mean()) for s in state],
        "param_abs_max": max(float(p.abs().max()) for p in model.parameters()),
    }
    # The two-compartment state is [B, 2d]: fast pole then slow pole. Splitting
    # them is the whole point -- the slow pole is reset-shielded, so if anything
    # accumulates with width it is that half, and a single max over both would
    # hide which.
    if cfg.arch.startswith("twocomp"):
        halves = []
        for s in state:
            d = s.shape[-1] // 2
            halves.append({
                "fast_abs_max": float(s[..., :d].abs().max()),
                "slow_abs_max": float(s[..., d:].abs().max()),
                "fast_abs_mean": float(s[..., :d].abs().mean()),
                "slow_abs_mean": float(s[..., d:].abs().mean()),
            })
        row["twocomp_state_split_per_layer"] = halves
    return row


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="docs/reports/data/exp_015_state_growth.json")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args(argv)

    rows = []
    for run, ck, note in TARGETS:
        print(f"[state] {run} ...", flush=True)
        r = probe(run, ck, note, args.device)
        rows.append(r)
        if "error" in r:
            print(f"  {r['error']}", flush=True)
        else:
            print(f"  d={r['d_model']:<5} arch={r['arch']:<18} "
                  f"|state|max={[round(v, 1) for v in r['state_abs_max_per_layer']]}",
                  flush=True)

    out = {
        "experiment": "EXP_015_architecture_at_width",
        "question": ("does the two-compartment membrane grow with width where "
                     "the plain LIF's does not?"),
        "inference_only": True,
        "adopts_nothing": True,
        "ranks_nothing": True,
        "no_bar": True,
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
