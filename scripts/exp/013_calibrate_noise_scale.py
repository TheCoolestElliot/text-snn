"""EXP_013 -- what scale is the input current, so the noise ladder can be set?

Run BEFORE `experiments/logs/EXP_013_noise_injection.md` fixes its magnitude
ladder, and committed with its output, so that the ladder is a calibrated choice
rather than three round numbers.

WHY THIS EXISTS AT ALL
----------------------
The arm injects background spike noise into the layer input current `cur`. Its
one hyperparameter is the noise amplitude, and an amplitude only means something
against the scale of the thing it is added to. `03_phase3_candidates.md` 6.2 is
the precedent: the per-context acceptance bar could not be used until it had been
measured, and a flat bar would have been 3x too strict in one place and 30x too
lax in another. A noise ladder of {0.01, 0.1, 1.0} chosen by taste has the same
defect and no way to notice it.

So this script measures `sd(cur)` per layer on the committed Phase-2 baseline
checkpoints, under the *training* data distribution (the noise is a training-time
intervention, so the training current is the relevant one), and the
pre-registration expresses its ladder as a fraction of that number.

WHAT IT DOES NOT DO
-------------------
It reads no bpc, scores no arm and touches no outcome. It measures a property of
the *baseline's* forward pass. Nothing here can be read as evidence for or
against the hypothesis EXP_013 registers, which is why it is allowed to run
first.

THE SELF-CHECK, WHICH IS THE POINT OF F1 IN THIS PROJECT
--------------------------------------------------------
To observe `cur` at all this file re-runs `_CharLMStack.forward`'s loop by hand,
and a hand-rolled replay of a committed forward pass is exactly the kind of
second implementation that silently stops being the first one (EXP_006's L5,
EXP_012's S1). So every replay asserts its logits are **bit-identical**
(`torch.equal`) to `model(idx)`'s, on every checkpoint, and the run aborts if
any pair differs. A statistic gathered from a replay that is not the model is
not a statistic about the model.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

import torch

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "src"))

from snn.config import Config, seed_everything  # noqa: E402
from snn.data import Corpus, RandomWindowSampler  # noqa: E402
from snn.model import build_model  # noqa: E402

RUNS = _REPO / "experiments" / "runs"
OUT = _REPO / "docs" / "reports" / "data" / "exp_013_noise_calibration.json"

#: The five committed Phase-2 baseline seeds. The arm nests this model at
#: amplitude 0, so this is the configuration the ladder must be scaled to.
BASELINE_RUNS = tuple(f"snn_beta0.5_s{i}" for i in range(5))

#: Steps sampled from the training stream. Not step 0: the current's scale at
#: initialisation is not the scale it has for the 20 000 steps that follow, and
#: the noise is injected throughout.
PROBE_STEPS = (0, 1, 2, 4, 8)


def _load(run: str, device: torch.device):
    ck_path = RUNS / run / "ckpt_final.pt"
    if not ck_path.exists():
        raise SystemExit(f"missing checkpoint: {ck_path}")
    ck = torch.load(ck_path, map_location="cpu", weights_only=False)
    known = {f for f in Config().to_dict()}
    cfg = Config(**{k: v for k, v in ck["config"].items() if k in known})
    model = build_model(cfg).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    return cfg, model


@torch.no_grad()
def current_stats(model, idx: torch.Tensor) -> tuple[list[dict], torch.Tensor]:
    """Per-layer statistics of `cur`, plus the logits of the replay.

    A faithful transcription of `_CharLMStack.forward` -- one GEMM per layer over
    the whole sequence, then the scan -- with the intermediate retained. The
    caller checks the logits against the real forward pass.
    """
    h = model.embed(idx)
    state = model.init_state(idx.shape[0], idx.device)
    out: list[dict] = []
    for k, linear in enumerate(model.layers):
        cur = model._project(linear, h)
        out.append({
            "layer": k,
            "sd": float(cur.std()),
            "mean": float(cur.mean()),
            "abs_mean": float(cur.abs().mean()),
            "p99_abs": float(cur.abs().flatten().quantile(0.99)),
            "max_abs": float(cur.abs().max()),
        })
        emitted, v_final = model._scan(cur, state[k], k)
        h = emitted
        state[k] = v_final
    return out, model.head(h)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    device = torch.device(args.device)
    per_run: list[dict] = []
    replay_checks: list[dict] = []

    for run in BASELINE_RUNS:
        cfg, model = _load(run, device)
        seed_everything(cfg.seed, cfg.deterministic)
        corpus = Corpus(cfg.corpus, cfg.data_dir)
        sampler = RandomWindowSampler(corpus.split("train"), cfg.batch_size,
                                      cfg.seq_len, cfg.seed)

        rows: list[list[dict]] = []
        for step in PROBE_STEPS:
            x, _y = sampler.batch(step)
            idx = x.to(device)
            stats, replay_logits = current_stats(model, idx)
            true_logits, _s, _a = model(idx, None)
            same = bool(torch.equal(replay_logits, true_logits))
            replay_checks.append({"run": run, "step": step, "bit_identical": same})
            if not same:
                raise SystemExit(
                    f"ABORT: the replay in {run} at step {step} is not the "
                    "committed forward pass, so its `cur` is not the model's "
                    "`cur`. Fix the replay; do not report the number."
                )
            rows.append(stats)

        per_layer = []
        for k in range(cfg.n_layers):
            sds = [r[k]["sd"] for r in rows]
            per_layer.append({
                "layer": k,
                "sd_median": statistics.median(sds),
                "sd_min": min(sds),
                "sd_max": max(sds),
                "mean_median": statistics.median([r[k]["mean"] for r in rows]),
                "abs_mean_median": statistics.median([r[k]["abs_mean"] for r in rows]),
                "p99_abs_median": statistics.median([r[k]["p99_abs"] for r in rows]),
                "max_abs_median": statistics.median([r[k]["max_abs"] for r in rows]),
            })
        per_run.append({"run": run, "seed": cfg.seed, "threshold": cfg.threshold,
                        "per_layer": per_layer})
        print(f"{run}: " + "  ".join(
            f"L{p['layer']} sd={p['sd_median']:.4f}" for p in per_layer))
        del model
        torch.cuda.empty_cache()

    # The ladder is scaled to the SMALLEST layer sd across seeds, so that a
    # given fraction is at most that fraction of the current anywhere in the
    # model. Scaling to the mean would understate the perturbation in the
    # quieter layer, which is layer 0 -- the one EXP_012 showed is where a
    # perturbation actually reaches the metric.
    all_sd = [p["sd_median"] for r in per_run for p in r["per_layer"]]
    by_layer = {}
    for r in per_run:
        for p in r["per_layer"]:
            by_layer.setdefault(p["layer"], []).append(p["sd_median"])

    summary = {
        "artifact": "exp_013_noise_calibration",
        "purpose": "set EXP_013's noise-amplitude ladder against the measured "
                   "scale of the current it is added to",
        "runs": list(BASELINE_RUNS),
        "probe_steps": list(PROBE_STEPS),
        "split": "train",
        "sd_cur_min_over_layers_and_seeds": min(all_sd),
        "sd_cur_max_over_layers_and_seeds": max(all_sd),
        "sd_cur_median_by_layer": {str(k): statistics.median(v)
                                   for k, v in sorted(by_layer.items())},
        "replay_bit_identical": all(c["bit_identical"] for c in replay_checks),
        "replay_checks": replay_checks,
        "per_run": per_run,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nsd(cur) by layer (median over seeds): "
          f"{summary['sd_cur_median_by_layer']}")
    print(f"min over layers and seeds: {summary['sd_cur_min_over_layers_and_seeds']:.4f}")
    print(f"replay bit-identical on {len(replay_checks)}/{len(replay_checks)}: "
          f"{summary['replay_bit_identical']}")
    print(f"written: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
