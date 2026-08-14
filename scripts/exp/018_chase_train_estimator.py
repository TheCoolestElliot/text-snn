"""EXP_018 §10 item 11: why do two train-side estimators disagree by ~0.005?

    python scripts/exp/018_chase_train_estimator.py

§9.5 built a train-side marker from the mean of the last eight logged
single-batch **training** losses. §9.5's own correction then ran `EXP_013`'s
committed instrument -- a 19,456-window slice of the train split scored in
`eval()` -- and the two disagree by ~0.005 bpc **in opposite directions for the
two arms**: the anchor's running loss sits ~0.0028 *above* its fresh train-slice
bpc while the dopamine arm's sits ~0.0026 *below* its own.

That asymmetry is the whole discrepancy, and an unexplained systematic of that
size sitting inside a claim about a 0.006 effect is not something to leave in a
referral if it is cheap to resolve. It is: the sampler is a pure function of
`(seed, step)`, so **the exact batches those eight steps trained on can be
regenerated and re-scored**.

WHAT THIS SEPARATES
-------------------
Three candidate explanations, and the design distinguishes them:

  A. **The batches.** The logged loss is on eight *randomly drawn* windows;
     the instrument's slice is a fixed head of the split. If A is the cause,
     re-scoring the SAME eight batches in `eval()` reproduces the logged number
     and the head-of-split slice is simply different text.
  B. **The weights.** The logged loss at step `s` is computed at the weights
     *before* step `s`'s update; the instrument uses `ckpt_final`. With a cosine
     schedule at ~0 by then this should be negligible, and if it is not, that is
     worth knowing on its own.
  C. **Something arm-specific in the forward.** This is the one that matters,
     because it is the only candidate that could bear on the result: if the
     dopamine arm scores differently on the *same* batch at the *same* weights
     depending on how it is reached, the arm has a state-dependence nobody
     declared.

The probe scores the same eight batches three ways -- logged, `eval()` at final
weights, and `train()` at final weights -- so A shows up as a batch/slice
difference, B as a logged-vs-final difference, and C as an eval-vs-train
difference **on identical inputs and identical weights**.

WHAT IT DOES NOT DO
-------------------
It resolves no bar, adopts nothing and changes no verdict. §9.5's correction
already stands on the committed instrument; this only explains a discrepancy
that correction recorded.
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
if str(_REPO / "src") not in sys.path:
    sys.path.insert(0, str(_REPO / "src"))

from snn.config import Config, seed_everything  # noqa: E402
from snn.data import Corpus, RandomWindowSampler  # noqa: E402
from snn.metrics import bits_per_char  # noqa: E402
from snn.model import build_model  # noqa: E402

RUNS = _REPO / "experiments" / "runs"
OUT = _REPO / "docs" / "reports" / "data" / "exp_018_train_estimator.json"

ARMS = {"da_anchor": 5, "da_mult": 5}
N_TAIL = 8          # the same eight the §9.5 marker averaged


def logged_tail(run: str, n: int) -> list[tuple[int, float]]:
    """(step, bpc) for the last `n` logged training records."""
    recs = []
    for line in (RUNS / run / "log.jsonl").read_text(encoding="utf-8").splitlines():
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if r.get("event") == "train":
            recs.append((int(r["step"]), float(r["bpc"])))
    return recs[-n:]


@torch.no_grad()
def score(model, x, y, vocab: int) -> float:
    logits, _, _ = model(x, None)
    loss = F.cross_entropy(logits.reshape(-1, vocab).float(), y.reshape(-1),
                           reduction="sum")
    return bits_per_char(float(loss), int(y.numel()))


def probe(run: str, device: str) -> dict:
    ck = torch.load(RUNS / run / "ckpt_final.pt", map_location=device,
                    weights_only=False)
    cfg = Config(**ck["config"])
    cfg.device = device
    cfg.data_dir = str(_REPO / cfg.data_dir)
    seed_everything(cfg.seed, cfg.deterministic)
    model = build_model(cfg)
    model.load_state_dict(ck["model"])

    corpus = Corpus(cfg.corpus, cfg.data_dir)
    sampler = RandomWindowSampler(corpus.split("train"), cfg.batch_size,
                                  cfg.seq_len, cfg.seed)

    tail = logged_tail(run, N_TAIL)
    rows = []
    for step, logged in tail:
        x, y = sampler.batch(step)
        x, y = x.to(device), y.to(device)
        model.eval()
        e = score(model, x, y, cfg.vocab_size)
        model.train()
        t = score(model, x, y, cfg.vocab_size)
        rows.append({"step": step, "logged_bpc": logged,
                     "eval_mode_bpc": e, "train_mode_bpc": t})
    model.eval()
    return {
        "run": run, "arch": cfg.arch, "seed": cfg.seed,
        "logged_mean": statistics.fmean(r["logged_bpc"] for r in rows),
        "eval_mode_mean": statistics.fmean(r["eval_mode_bpc"] for r in rows),
        "train_mode_mean": statistics.fmean(r["train_mode_bpc"] for r in rows),
        "per_step": rows,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args(argv)

    out: dict = {"artifact": "exp_018_train_estimator",
                 "question": "why do the running training loss and the train-slice "
                             "bpc disagree by ~0.005, in opposite directions per arm?",
                 "n_tail_steps": N_TAIL, "runs": []}
    for arm, n in ARMS.items():
        for s in range(n):
            run = f"{arm}_s{s}"
            if not (RUNS / run / "ckpt_final.pt").exists():
                continue
            r = probe(run, args.device)
            out["runs"].append(r)
            print(f"{run:<14s} logged {r['logged_mean']:.5f}  "
                  f"eval@final {r['eval_mode_mean']:.5f}  "
                  f"train@final {r['train_mode_mean']:.5f}  "
                  f"(logged-eval {r['logged_mean'] - r['eval_mode_mean']:+.5f}, "
                  f"eval-train {r['eval_mode_mean'] - r['train_mode_mean']:+.5f})",
                  flush=True)

    by = {}
    for arm in ARMS:
        rs = [r for r in out["runs"] if r["run"].startswith(arm)]
        if not rs:
            continue
        by[arm] = {
            "logged_minus_eval": statistics.fmean(
                r["logged_mean"] - r["eval_mode_mean"] for r in rs),
            "eval_minus_train_mode": statistics.fmean(
                r["eval_mode_mean"] - r["train_mode_mean"] for r in rs),
            "n": len(rs),
        }
    out["summary"] = by

    # The one that would matter: eval() and train() must agree on identical
    # inputs at identical weights. Neither arm has dropout or batch norm, and the
    # dopamine arm's `da_active` deliberately does NOT consult `self.training`.
    worst = max((abs(v["eval_minus_train_mode"]) for v in by.values()), default=0.0)
    out["eval_train_mode_agree"] = worst < 1e-9
    out["worst_eval_minus_train_mode"] = worst

    p = Path(args.out)
    if not p.is_absolute():
        p = _REPO / p
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")

    print("\nmean over seeds:")
    for arm, v in by.items():
        print(f"  {arm:<11s} logged - eval@final = {v['logged_minus_eval']:+.5f}   "
              f"eval - train mode = {v['eval_minus_train_mode']:+.2e}  (n={v['n']})")
    print(f"\neval()/train() agree on identical inputs: {out['eval_train_mode_agree']} "
          f"(worst {worst:.2e})")
    print(f"written: {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
