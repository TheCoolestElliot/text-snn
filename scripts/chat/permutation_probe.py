"""Does the model use ORDER, or only RECENCY?

    python scripts/chat/permutation_probe.py --ckpt experiments/chat/chat-v3d-aligned/ckpt_best.pt

POST-HOC AND EXPLORATORY. Not pre-registered, resolves no bar, and no number it
prints is a reported figure. It is a description of a committed checkpoint, in the
same class as reading `beta_s_raw` off a state dict.

THE QUESTION
------------
`score_at_horizon` measures how far back conditioning information helps. It
cannot distinguish

    "the model remembers WHAT was there"   from   "the model remembers roughly
                                                   HOW MUCH was there".

A leaky integrator obeys `vs_t = beta*vs_{t-1} + cur_t`, so two windows holding
the same characters in a different order converge to nearly the same state. If
that is all the model has, then a horizon of 47 characters is a *recency
statistic*, not a memory, and no timescale intervention can change it.

THE DESIGN, AND WHY IT NEEDS TWO PERTURBATIONS
----------------------------------------------
For a band of the window at lag ~c/2..c from the end, positions [L-c, L-c/2):

  PERMUTE  shuffle the order inside the band. Multiset unchanged.
  ROLL     replace the band with the same band from a different window in the
           batch. Content AND order changed. (The batch-roll control shape
           EXP_018 used, so the marginal character statistics are preserved --
           unlike substituting uniform-random ids.)

Loss is scored only at positions >= L-c/2, which are strictly after the band, and
whose own inputs and targets are untouched by either perturbation. So the only
thing that differs is the context.

Reporting `PERMUTE` alone is ambiguous: it goes to zero at large c both when
order stops mattering AND when the band stops mattering at all. The ratio

    order_share(c) = dPERMUTE(c) / dROLL(c)

separates them. dROLL is how much the band is worth; dPERMUTE is how much of that
is its order.

SELF-CHECKS, both of which must be EXACTLY zero
-----------------------------------------------
  S1 identity   applying the identity permutation must change nothing.
  S2 causality  perturbing a band strictly AFTER every scored position must
                change nothing. A non-zero value means the harness is leaking
                future information and the numbers below are meaningless.

A residual that is merely "small" is a failure here, not a pass: both are exact
identities on a causal model, so anything but 0.0 is a bug in this file.
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

from snnchat.data import ChatCorpus, SequentialEval  # noqa: E402
from snnchat.generate import load_chat_checkpoint  # noqa: E402

_LOG2E = 1.4426950408889634


def _perm_band(x: torch.Tensor, lo: int, hi: int, g: torch.Generator) -> torch.Tensor:
    """Shuffle positions [lo, hi) independently per row. Multiset preserved."""
    out = x.clone()
    b, width = x.shape[0], hi - lo
    if width < 2:
        return out
    # One independent permutation per row, so the batch is not one shared shuffle.
    idx = torch.argsort(torch.rand(b, width, generator=g, device=x.device), dim=1)
    out[:, lo:hi] = torch.gather(x[:, lo:hi], 1, idx)
    return out


def _roll_band(x: torch.Tensor, lo: int, hi: int) -> torch.Tensor:
    """Replace [lo, hi) with the same slice from the next row of the batch."""
    out = x.clone()
    if hi - lo < 1:
        return out
    out[:, lo:hi] = torch.roll(x[:, lo:hi], shifts=1, dims=0)
    return out


@torch.no_grad()
def _nats_after(model, x, y, first_scored: int, vocab: int) -> torch.Tensor:
    """Per-row summed nats over positions [first_scored, L). Shape [B]."""
    logits, _, _ = model(x, None)
    loss = F.cross_entropy(
        logits.reshape(-1, vocab).float(), y.reshape(-1), reduction="none"
    ).view(x.shape)
    return loss[:, first_scored:].sum(dim=1).double()


@torch.no_grad()
def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--data-dir", default="data/chat")
    p.add_argument("--source", default="soda")
    p.add_argument("--seq-len", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--max-batches", type=int, default=48)
    p.add_argument("--bands", default="4,8,16,32,64,128")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None)
    a = p.parse_args(argv)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model, cfg, ck = load_chat_checkpoint(a.ckpt, device=dev)
    model.eval()
    vocab, L = cfg.vocab_size, a.seq_len

    corpus = ChatCorpus(a.data_dir, vocab_version=cfg.vocab_version)
    batches = []
    for i, bt in enumerate(SequentialEval(corpus, a.source, a.batch_size, L)):
        if i >= a.max_batches:
            break
        batches.append(bt)
    if not batches:
        raise SystemExit(f"no val data for {a.source!r}")
    n_rows = sum(int(x.shape[0]) for x, _ in batches)

    print(f"checkpoint {a.ckpt}  step {ck.get('step')}")
    print(f"{a.source}: {len(batches)} batches, {n_rows} windows of {L}\n")

    g = torch.Generator(device=dev)
    g.manual_seed(a.seed)

    # ---- self-checks -----------------------------------------------------
    c = 64
    lo, hi, first = L - c, L - c // 2, L - c // 2
    d_ident = d_causal = 0.0
    for x, y in batches[:4]:
        x, y = x.to(dev), y.to(dev)
        base = _nats_after(model, x, y, first, vocab)
        d_ident += float((_nats_after(model, x.clone(), y, first, vocab) - base).abs().sum())
        # perturb a band strictly AFTER every scored position: there is none
        # inside the window, so instead perturb [first, L) and score [0, lo)
        xa = _perm_band(x, first, L, g)
        # Only positions before `first` are causally unaffected by that band.
        lg_b, _, _ = model(x, None)
        lg_p, _, _ = model(xa, None)
        lb = F.cross_entropy(lg_b.reshape(-1, vocab).float(), y.reshape(-1),
                             reduction="none").view(x.shape)[:, :first]
        lp = F.cross_entropy(lg_p.reshape(-1, vocab).float(), y.reshape(-1),
                             reduction="none").view(x.shape)[:, :first]
        d_causal += float((lp - lb).abs().sum())
    print(f"S1 identity permutation      max|delta nats| = {d_ident:.3e}")
    print(f"S2 causality (perturb future) max|delta nats| = {d_causal:.3e}")
    if d_ident != 0.0 or d_causal != 0.0:
        raise SystemExit(
            "SELF-CHECK FAILED: these are exact identities on a causal model. "
            "Chase it rather than widening a tolerance."
        )
    print("both exactly zero\n")

    # ---- the sweep -------------------------------------------------------
    print(f"{'band lag':>10} {'width':>6} {'dPERMUTE':>11} {'dROLL':>11} "
          f"{'order share':>12}")
    print("-" * 56)
    rows = []
    for c in [int(s) for s in a.bands.split(",")]:
        lo, hi = L - c, L - c // 2
        first = hi
        width = hi - lo
        tot_p = tot_r = 0.0
        n_scored = 0
        for x, y in batches:
            x, y = x.to(dev), y.to(dev)
            base = _nats_after(model, x, y, first, vocab)
            perm = _nats_after(model, _perm_band(x, lo, hi, g), y, first, vocab)
            roll = _nats_after(model, _roll_band(x, lo, hi), y, first, vocab)
            tot_p += float((perm - base).sum())
            tot_r += float((roll - base).sum())
            n_scored += int(x.shape[0]) * (L - first)
        dp = tot_p / n_scored * _LOG2E
        dr = tot_r / n_scored * _LOG2E
        share = dp / dr if dr > 1e-9 else float("nan")
        rows.append({"band_lag": c, "width": width, "d_permute_bpc": dp,
                     "d_roll_bpc": dr, "order_share": share,
                     "n_scored": n_scored})
        print(f"{c:>10} {width:>6} {dp:>11.5f} {dr:>11.5f} {share:>12.3f}")

    print("\ndPERMUTE = cost of shuffling the band's ORDER   (multiset kept)")
    print("dROLL    = cost of replacing the band's CONTENT (batch-roll)")
    print("order share = dPERMUTE / dROLL: what fraction of the band's value is order")

    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(
            {"checkpoint": a.ckpt, "step": ck.get("step"), "source": a.source,
             "seq_len": L, "n_windows": n_rows, "rows": rows,
             "self_checks": {"identity": d_ident, "causality": d_causal}},
            indent=2), encoding="utf-8")
        print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
