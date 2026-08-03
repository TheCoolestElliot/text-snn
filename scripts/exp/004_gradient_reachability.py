"""EXP_004 entry condition J4: is every new parameter reachable by a gradient?

The screen proposed in `03_phase3_candidates.md` §7.2, run for the first time:

    Build the eager reference at a candidate's proposed init, take one backward,
    report |dL/dp| for every new parameter against |dL/dW|, and flag anything
    exactly zero or ~100x below. Seconds per candidate; it is what identifies
    §6.4's saddle before 0.8 GPU-hours are spent certifying a null.

§6.4 identified the rotational membrane's proposed initialisation as an exact
gradient saddle *from its own backward equations*, and that candidate has not been
run. EXP_004 §2.2 makes the same derivation for this one and predicts (T0) that
`w = 0` -- the initialisation that nests the Phase-2 baseline exactly, and is
therefore the attractive one -- is an exact saddle for the slow decay:

    dL/dvs_t = w * dL/dv_t + beta_s * dL/dvs_{t+1},  dL/dvs_L = 0

so at w = 0 every dL/dvs_t is zero by backward induction, and dL/dbeta_s with it.
The mix itself stays reachable, because dL/dw = sum_t (dL/dv_t) * vs_t and vs
integrates the current whatever w is.

This script measures that rather than trusting it, on the real model, the real
corpus and the real loss -- not on a synthetic batch, because the quantity of
interest is a ratio against `dL/dW` and that reference has to be the one training
will actually see.

WHAT THE RATIO DOES AND DOES NOT MEAN
-------------------------------------
The zero test and the ratio test are not the same kind of evidence, and the
difference is worth stating because it changes how the output should be read.

An **exactly zero** gradient is fatal and unrecoverable: AdamW's update is
`m / (sqrt(v) + eps)`, which is exactly 0 when the gradient is identically 0 at
every step, so the parameter is frozen at its initialisation for the whole run and
nothing in the logs says so.

A **small but nonzero** gradient is a much weaker signal, because Adam is scale-
free per parameter: it divides by the gradient's own second moment, so a uniformly
100x smaller gradient still produces a full-size step. A low ratio here flags a
parameter whose gradient is small *relative to the weights'*, which is worth
recording and is what §7.2 asked for, but it is not on its own a reason to reject
an init. Both numbers are reported; the decision rule below is §7.2's as written,
and this note is an observation about how to read it, not a change to it.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO / "src") not in sys.path:
    sys.path.insert(0, str(_REPO / "src"))

from snn.config import Config, seed_everything  # noqa: E402
from snn.data import Corpus, RandomWindowSampler  # noqa: E402
from snn.model import build_model, count_params, twocomp_param_count  # noqa: E402

# The two candidate initialisations, both named in EXP_004 §2.2 before any of this
# was measured. `w_init = 0.0` is the proposed one (exact nesting); `0.1` is the
# pre-registered fallback.
CANDIDATE_INITS = [
    {"label": "proposed (exact nesting)", "w_init": 0.0, "beta_slow": 0.95},
    {"label": "pre-registered fallback", "w_init": 0.1, "beta_slow": 0.95},
]

# §7.2's rule: "flag anything exactly zero or ~100x below".
RATIO_FLOOR = 1.0 / 100.0

NEW_PARAMS = ("w", "beta_s_raw")


def _grad_stats(t: torch.Tensor | None) -> dict:
    if t is None:
        return {"present": False}
    return {
        "present": True,
        "numel": int(t.numel()),
        "abs_max": float(t.abs().max()),
        "rms": float(t.pow(2).mean().sqrt()),
        "l2": float(t.norm()),
        "exactly_zero": bool(float(t.abs().max()) == 0.0),
    }


def screen_one(cfg: Config, corpus: Corpus, fused: bool) -> dict:
    """One forward/backward at initialisation; gradient statistics per parameter."""
    seed_everything(cfg.seed, cfg.deterministic)
    device = torch.device(cfg.device)
    model = build_model(cfg)
    model.train()

    sampler = RandomWindowSampler(corpus.split("train"), cfg.batch_size,
                                  cfg.seq_len, cfg.seed)
    x, y = sampler.batch(0)
    x, y = x.to(device), y.to(device)

    logits, _state, aux = model(x, None)
    loss = F.cross_entropy(logits.reshape(-1, cfg.vocab_size).float(), y.reshape(-1))
    loss.backward()

    # The reference the ratio is taken against: the layer projections, which are
    # the parameters the baseline arm already trains successfully.
    w_grads = [ly.weight.grad for ly in model.layers]
    w_rms = float(torch.stack([g.pow(2).mean() for g in w_grads]).mean().sqrt())

    per_layer = []
    for k in range(cfg.n_layers):
        row = {
            "layer": k,
            "firing_rate": float(aux["firing_rate"][k]),
            "linear_weight": _grad_stats(model.layers[k].weight.grad),
        }
        for name in NEW_PARAMS:
            g = getattr(model, name)[k].grad
            st = _grad_stats(g)
            st["rms_vs_linear_weight"] = (st["rms"] / w_rms) if st["present"] else None
            st["flagged"] = bool(
                st.get("exactly_zero")
                or (st.get("rms_vs_linear_weight") is not None
                    and st["rms_vs_linear_weight"] < RATIO_FLOOR)
            )
            row[name] = st
        # How much of the membrane the slow compartment actually supplies at
        # init. A parameter can be reachable and still be doing nothing.
        row["slow_share_of_membrane"] = None
        per_layer.append(row)

    flagged = [
        f"layer{r['layer']}.{name}"
        for r in per_layer for name in NEW_PARAMS if r[name]["flagged"]
    ]
    zeros = [
        f"layer{r['layer']}.{name}"
        for r in per_layer for name in NEW_PARAMS if r[name]["exactly_zero"]
    ]
    return {
        "fused": fused,
        "loss": float(loss),
        "train_bpc": float(loss) / 0.6931471805599453,
        "params": count_params(model),
        "linear_weight_grad_rms": w_rms,
        "per_layer": per_layer,
        "flagged": flagged,
        "exactly_zero": zeros,
        "passes_screen": not flagged,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out",
                    default="docs/reports/data/exp_004_gradient_reachability.json")
    args = ap.parse_args(argv)

    if not torch.cuda.is_available():
        raise SystemExit("EXP_004 J4 screen needs CUDA (it screens the real arm)")

    base = Config(arch="twocomp", cuda_graph=False, seed=0)
    corpus = Corpus(base.corpus, base.data_dir)
    base.vocab_size = int(corpus.vocab_size)

    expected = twocomp_param_count(base.vocab_size, base.d_model, base.n_layers)
    results: dict = {
        "experiment": "EXP_004_gradient_reachability",
        "screen": "03_phase3_candidates.md §7.2",
        "rule": "flag a new parameter whose |dL/dp| is exactly zero, or whose "
                "per-element RMS is below 1/100 of the layer weights'",
        "ratio_floor": RATIO_FLOOR,
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(0),
        "expected_params": expected,
        "inits": {},
    }

    for spec in CANDIDATE_INITS:
        cfg = Config(**{**base.to_dict(), "w_init": spec["w_init"],
                        "beta_slow": spec["beta_slow"]})
        key = f"w_init={spec['w_init']}"
        print(f"\n{key}  ({spec['label']})")
        # Both paths, because the screen has to describe the arm that will train
        # (fused) and the arm the R10 gate certifies (eager), and a divergence
        # between them at initialisation would matter more than either number.
        entry = {"label": spec["label"], **spec}
        for fused in (False, True):
            cfg_f = Config(**{**cfg.to_dict(), "fused": fused})
            r = screen_one(cfg_f, corpus, fused)
            entry["eager" if not fused else "fused"] = r
            tag = "fused" if fused else "eager"
            for row in r["per_layer"]:
                ws, bs = row["w"], row["beta_s_raw"]
                print(f"    [{tag}] layer {row['layer']}  rate {row['firing_rate']:.3f}"
                      f"   |dL/dW| rms {row['linear_weight']['rms']:.3e}"
                      f"   |dL/dw| rms {ws['rms']:.3e} (x{ws['rms_vs_linear_weight']:.3f})"
                      f"   |dL/dbeta_s| rms {bs['rms']:.3e} "
                      f"(x{bs['rms_vs_linear_weight']:.3f})")
            print(f"    [{tag}] exactly zero: {r['exactly_zero'] or 'none'}   "
                  f"flagged: {r['flagged'] or 'none'}   "
                  f"{'PASS' if r['passes_screen'] else 'FAIL'}")
        entry["passes_screen"] = (entry["eager"]["passes_screen"]
                                  and entry["fused"]["passes_screen"])
        entry["agrees_fused_vs_eager"] = abs(
            entry["eager"]["loss"] - entry["fused"]["loss"]) < 1e-4
        results["inits"][key] = entry

    # The pre-registered rule: take the proposed init if it passes, otherwise the
    # named fallback. Decided by the screen, not by whichever trains better --
    # nothing has trained yet, which is the entire point of running this first.
    order = [f"w_init={s['w_init']}" for s in CANDIDATE_INITS]
    selected = next((k for k in order if results["inits"][k]["passes_screen"]), None)
    results["selected_init"] = selected
    results["selection_rule"] = (
        "EXP_004 §2.2: the proposed init w=0 is used if it clears §7.2's screen; "
        "otherwise the fallback w=0.1 named in the pre-registration is used.")

    out = _REPO / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print("\n" + "=" * 78)
    print("GRADIENT REACHABILITY -- EXP_004 candidate #1, two new parameters")
    print("=" * 78)
    print(f"{'init':16s} {'w reachable':>13s} {'beta_s reachable':>18s} {'verdict':>10s}")
    for key in order:
        e = results["inits"][key]
        w_ok = not any(r["w"]["exactly_zero"] for r in e["fused"]["per_layer"])
        b_ok = not any(r["beta_s_raw"]["exactly_zero"] for r in e["fused"]["per_layer"])
        print(f"{key:16s} {'yes' if w_ok else 'NO':>13s} {'yes' if b_ok else 'NO':>18s} "
              f"{'PASS' if e['passes_screen'] else 'FAIL':>10s}")
    print(f"\nSELECTED: {selected}")
    if selected is None:
        print("  no candidate init clears the screen -- EXP_004 does not train")
    print(f"\nWROTE {args.out}")
    return 0 if selected else 1


if __name__ == "__main__":
    raise SystemExit(main())
