"""The two evaluation protocols, and nothing else.

01_reconnaissance.md 4.3 makes *both* mandatory for every headline number,
because they answer different questions and differ by a lot on a model whose
memory lives entirely in membrane state:

  "fresh"   (protocol A) -- non-overlapping windows, membrane state zeroed at
            every window start.  Matches the training distribution exactly, so
            it is the honest measure of what the model was optimised to do.
  "carried" (protocol B) -- the split is cut into `batch_size` contiguous
            streams and state is carried, detached, across successive windows.
            Every character is scored exactly once, which is the protocol
            published enwik8 bits-per-character figures use.

Reporting only "fresh" would flatter a short-horizon model; reporting only
"carried" would make our number look like a like-for-like comparison with the
literature when the training objective was different.  Both, always.

Accounting notes that matter for reproducibility:

* Losses are summed with ``reduction="sum"`` and accumulated on the host in
  float64.  A running mean-of-means would weight the last partial batch wrongly
  and would make the result depend on how the stream happens to divide.
* ``max_windows`` truncates at a whole-batch boundary.  bits-per-character is
  therefore invariant to ``batch_size`` for the "fresh" protocol whenever
  ``max_windows`` is a multiple of the batch sizes being compared (and when it
  is 0, whenever the window count is).  That is exactly what
  ``tests/test_determinism.py::test_fresh_bpc_invariant_to_batch_size`` checks;
  the residual difference is GPU reduction order, not different characters.
"""

from __future__ import annotations

import warnings

import torch
import torch.nn.functional as F

from snn.config import Config
from snn.data import contiguous_eval_batches, windowed_eval_batches
from snn.metrics import bits_per_char

_PROTOCOLS = ("fresh", "carried")


@torch.no_grad()
def evaluate(model, corpus, split: str, cfg: Config, protocol: str,
             max_windows: int = 0) -> dict:
    """Score `split` under `protocol`.

    Returns {"bpc", "nats", "n_chars", "protocol", "firing_rate", "n_windows"}
    (plus "split", for the benefit of callers writing JSONL).

    Deterministic: no sampling, no dropout, fixed window order.  The model is
    put in eval mode and its previous mode restored, so calling this from
    inside a training loop cannot leave the model in the wrong state.
    """
    if protocol not in _PROTOCOLS:
        raise ValueError(f"protocol must be one of {_PROTOCOLS}, got {protocol!r}")

    data = corpus.split(split)
    device = torch.device(cfg.device)
    batches = (windowed_eval_batches(data, cfg.batch_size, cfg.seq_len)
               if protocol == "fresh"
               else contiguous_eval_batches(data, cfg.batch_size, cfg.seq_len))

    was_training = model.training
    model.eval()

    total_nats = 0.0
    n_chars = 0
    n_windows = 0
    rate_sums: list[float] = []
    rate_weight = 0.0
    state = None

    try:
        for x, y in batches:
            b = int(x.shape[0])
            if max_windows and n_windows + b > max_windows and n_windows > 0:
                break
            if max_windows and n_windows + b > max_windows and n_windows == 0:
                # `max_windows` smaller than one batch used to score NOTHING and
                # return bpc = nan, which then flowed into summary.json and the
                # report table looking like a result. Truncating at a whole-batch
                # boundary is right; truncating to zero batches is not. Score one
                # full batch instead -- the caller asked for a cheap subsample,
                # and the cheapest honest subsample is one batch.
                warnings.warn(
                    f"max_windows={max_windows} is smaller than batch_size={b}; "
                    f"scoring one full batch ({b} windows) instead of none. "
                    "bpc from this call is not comparable with a run whose "
                    "max_windows is a multiple of the batch size.",
                    RuntimeWarning,
                    stacklevel=2,
                )
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            logits, new_state, aux = model(x, state if protocol == "carried" else None)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]).float(),
                y.reshape(-1),
                reduction="sum",
            )
            # float() forces a sync per batch.  Accepted: it keeps the running
            # total in host float64, which makes the result independent of how
            # many batches the split happened to be cut into.
            total_nats += float(loss)
            n_chars += int(y.numel())
            n_windows += b

            if protocol == "carried":
                state = ([s.detach() for s in new_state]
                         if new_state is not None else None)

            rates = aux.get("firing_rate", ()) if isinstance(aux, dict) else ()
            if rates:
                if not rate_sums:
                    rate_sums = [0.0] * len(rates)
                for i, r in enumerate(rates):
                    rate_sums[i] += float(r) * b
                rate_weight += b
    finally:
        model.train(was_training)

    firing_rate = ([s / rate_weight for s in rate_sums] if rate_weight > 0 else [])
    return {
        "bpc": bits_per_char(total_nats, n_chars) if n_chars else float("nan"),
        "nats": total_nats,
        "n_chars": n_chars,
        "protocol": protocol,
        "firing_rate": firing_rate,
        "n_windows": n_windows,
        "split": split,
    }


# `evaluate_both` was removed here. Its docstring said it existed "so that no
# caller can accidentally report one protocol and forget the other -- the single
# most likely way for this repository to publish a number that is not comparable
# with what it claims to be comparable with." **Nothing called it**, so it
# guaranteed nothing, and a dead guard is worse than an absent one: it reads like
# the invariant is enforced somewhere.
#
# The invariant is real and is now carried by `_PROTOCOLS` alone, which both
# callers iterate -- `scripts/evaluate.py` already did, and `snn/train.py`'s
# `_run_eval` had a second hardcoded `("fresh", "carried")` that has been pointed
# at this tuple. One constant, two call sites, no third copy to drift.
