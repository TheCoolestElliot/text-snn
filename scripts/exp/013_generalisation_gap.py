"""EXP_013 -- the generalisation gap, measured the same way for every arm.

Pre-registered in `experiments/logs/EXP_013_noise_injection.md`. This file is the
*instrument*, and it is deliberately one file rather than one per arm: EXP_006's
failure mode L5 says a second implementation of a statistic is how two numbers
stop being comparable, and this statistic is the arm's whole decision variable.

THE STATISTIC
-------------
    gap = bpc(test) - bpc(train slice)

so that a POSITIVE gap is the overfitting direction -- the model spends fewer
bits on text it was trained on than on text it was not -- and a regulariser that
works makes the gap SMALLER. The sign is fixed here, once, because a decision
rule stated over a quantity whose sign is ambiguous is a decision rule waiting to
be read backwards.

Computed under both committed protocols ("fresh" and "carried") with
`snn.evaluate.evaluate`, the same function that produced every bpc in this
project. Nothing is reimplemented here; this file chooses the splits and does the
subtraction.

WHY A *SLICE* OF TRAIN, AND WHAT THAT COSTS
-------------------------------------------
The test split is 4 980 736 scored characters in 19 456 windows. The train split
is 18x larger, so scoring all of it would make the two legs of the difference
differ in sample size as well as in role. `max_windows` is therefore set to the
test split's own window count, and `windowed_eval_batches` takes windows in a
fixed order from the start of the split -- so the train leg is the *first*
19 456 windows, the same text for every run.

That is a limitation and it is stated rather than corrected: the train leg is a
fixed head of the split, not a random sample of it, so `gap` is "how much better
does this model do on this particular seen text than on unseen text". Every arm
is measured on the identical text under the identical protocol, so the
*comparison* is paired and clean even though the absolute number is specific to
that slice. A random-sample train leg would need a sampling seed, which would
put an RNG inside the decision variable of an experiment about injected noise.

WHY THIS RUNS ON THE BASELINE BEFORE THE ARM EXISTS
----------------------------------------------------
`EXP_005` established that sigma does not transfer between arms, and
`03_phase3_candidates.md` 6.2 established that a bar has to be calibrated before
it is used. The gap's cross-seed spread has never been measured by this project,
so the five committed Phase-2 baseline seeds are scored here FIRST, the bar is
fixed from that number, and only then does the pre-registration name a threshold.
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

from snn.config import Config  # noqa: E402
from snn.data import Corpus  # noqa: E402
from snn.evaluate import evaluate  # noqa: E402
from snn.model import build_model  # noqa: E402

RUNS = _REPO / "experiments" / "runs"
DATA = _REPO / "docs" / "reports" / "data"

PROTOCOLS = ("fresh", "carried")

#: The test split's window count at the frozen shape (B=128, L=256). Asserted
#: against the measured test leg on every run rather than trusted: if the two
#: legs ever stop being the same size, the difference stops being the statistic
#: this file claims to compute.
EXPECTED_TEST_WINDOWS = 19456


def load_run(run: str, device: torch.device):
    ck_path = RUNS / run / "ckpt_final.pt"
    if not ck_path.exists():
        raise SystemExit(f"missing checkpoint: {ck_path}")
    ck = torch.load(ck_path, map_location="cpu", weights_only=False)
    known = set(Config().to_dict())
    cfg = Config(**{k: v for k, v in ck["config"].items() if k in known})
    cfg.device = str(device)
    model = build_model(cfg).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    return cfg, model


def gap_for_run(run: str, device: torch.device) -> dict:
    cfg, model = load_run(run, device)
    corpus = Corpus(cfg.corpus, cfg.data_dir)

    out: dict = {"run": run, "seed": cfg.seed, "arch": cfg.arch,
                 "noise_amp": float(getattr(cfg, "noise_amp", 0.0)),
                 "noise_p": float(getattr(cfg, "noise_p", 0.0))}
    for protocol in PROTOCOLS:
        test = evaluate(model, corpus, "test", cfg, protocol, max_windows=0)
        if test["n_windows"] != EXPECTED_TEST_WINDOWS:
            raise SystemExit(
                f"ABORT: test leg is {test['n_windows']} windows, expected "
                f"{EXPECTED_TEST_WINDOWS}. The two legs of the gap would not be "
                "matched and the statistic would not be the pre-registered one."
            )
        train = evaluate(model, corpus, "train", cfg, protocol,
                         max_windows=EXPECTED_TEST_WINDOWS)
        if train["n_windows"] != test["n_windows"]:
            raise SystemExit(
                f"ABORT: train leg {train['n_windows']} != test leg "
                f"{test['n_windows']} windows."
            )
        out[protocol] = {
            "train_bpc": train["bpc"],
            "test_bpc": test["bpc"],
            "gap": test["bpc"] - train["bpc"],
            "n_windows": train["n_windows"],
            "train_firing_rate": train["firing_rate"],
            "test_firing_rate": test["firing_rate"],
        }
    del model
    torch.cuda.empty_cache()
    return out


def summarise(rows: list[dict], protocol: str) -> dict:
    gaps = [r[protocol]["gap"] for r in rows]
    tests = [r[protocol]["test_bpc"] for r in rows]
    trains = [r[protocol]["train_bpc"] for r in rows]
    n = len(gaps)
    return {
        "n": n,
        "gap_mean": statistics.fmean(gaps),
        "gap_sd": statistics.stdev(gaps) if n > 1 else None,
        "test_bpc_mean": statistics.fmean(tests),
        "test_bpc_sd": statistics.stdev(tests) if n > 1 else None,
        "train_bpc_mean": statistics.fmean(trains),
        "train_bpc_sd": statistics.stdev(trains) if n > 1 else None,
        "gaps": gaps,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("runs", nargs="+", help="run directory names under experiments/runs")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", required=True, help="artifact path under docs/reports/data")
    ap.add_argument("--label", default="", help="what this group of runs is")
    args = ap.parse_args()

    device = torch.device(args.device)
    rows = []
    for run in args.runs:
        row = gap_for_run(run, device)
        rows.append(row)
        c = row["carried"]
        print(f"{run:24s} carried  train {c['train_bpc']:.5f}  "
              f"test {c['test_bpc']:.5f}  gap {c['gap']:+.5f}")

    payload = {
        "artifact": Path(args.out).stem,
        "label": args.label,
        "statistic": "gap = bpc(test) - bpc(train slice, first 19456 windows); "
                     "positive = overfitting",
        "expected_test_windows": EXPECTED_TEST_WINDOWS,
        "runs": rows,
        "summary": {p: summarise(rows, p) for p in PROTOCOLS},
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    for p in PROTOCOLS:
        s = payload["summary"][p]
        sd = "n/a" if s["gap_sd"] is None else f"{s['gap_sd']:.5f}"
        print(f"\n[{p}] n={s['n']}  gap mean {s['gap_mean']:+.5f}  sd {sd}")
    print(f"written: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
