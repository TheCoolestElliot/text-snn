"""
snn_char_lm.py
==============

A character-level *spiking* neural-network (SNN) language model, built with
PyTorch + snnTorch and sized to train comfortably on an 8 GB GPU.

It demonstrates the pieces requested:

  1. Rate-coded input        -- each character is turned into a stochastic
                                spike train whose firing *rate* encodes the
                                one-hot symbol (snntorch.spikegen.rate).
  2. Recurrent LIF layers    -- stacked `snn.Leaky` leaky integrate-and-fire
                                neurons whose membrane potential is carried
                                across the character sequence. The leak IS the
                                recurrence: the hidden state at step t depends on
                                step t-1, exactly like an RNN, but with spiking
                                LIF dynamics. (See the `# DESIGN: recurrence`
                                note for why this beats an explicit lateral
                                recurrent weight matrix on this task.)
  3. Surrogate-gradient       -- the Heaviside spike is non-differentiable, so
     training                   backprop-through-time uses a smooth surrogate
                                for dS/dU (default: ATan).
  4. Rate-decoded readout    -- the top layer's mean firing rate is decoded by a
                                linear head into next-character logits.

Two more choices make training practical and are commented in the code:

  5. Truncated BPTT (TBPTT)  -- the recurrent state is carried across a long
                                window but the gradient graph is detached every
                                few characters, bounding memory and the length
                                of each backward pass.
  6. Gradient clipping       -- standard insurance for any recurrent net.

The extensive comments explain *why* each SNN design choice was made, not just
what the code does. Search for the tag `# DESIGN:` to jump between them.

Usage
-----
Train on the built-in demo corpus (runs with zero setup)::

    python snn_char_lm.py train --steps 2000

Train on real text -- the shipped ``input.txt`` is ~1.1 MB of public-domain
English prose (build it with ``python build_corpus.py``) -- with a held-out
validation split (the best-validation checkpoint is kept). This writes a scratch
checkpoint so it will not overwrite the pre-trained ``prose.pt`` shipped in the
repo; see the README "Results" section for the full recipe::

    python snn_char_lm.py train --data input.txt --steps 8000 --seq-len 160 \
        --val-split 0.1 --seed 1337 --ckpt my_prose.pt

Sample from a trained checkpoint::

    python snn_char_lm.py sample --ckpt prose.pt --prompt "It was " --length 400

Score a checkpoint's bits-per-character on a corpus (reproduce the metric)::

    python snn_char_lm.py eval --ckpt prose.pt --data input.txt \
        --val-split 0.1 --split val

Run a fast self-test (a short training run + a generation call)::

    python snn_char_lm.py smoke
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import math
import os
import re
import sys
import time
from dataclasses import dataclass, asdict
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

import snntorch as snn
from snntorch import spikegen
from snntorch import surrogate


# ---------------------------------------------------------------------------
# 0. Reproducibility + device
# ---------------------------------------------------------------------------
def set_seed(seed: int) -> None:
    """Seed every RNG we touch so runs are repeatable.

    Rate coding (below) draws Bernoulli samples, so a fixed seed is what makes
    an SNN experiment reproducible at all -- the *stochastic encoder* is part
    of the model, not just the weight init.
    """
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pick_device(pref: str = "auto") -> torch.device:
    """Resolve a device preference to a torch.device.

    "auto" (default) uses CUDA when available, else CPU. "cpu", "cuda", or
    "cuda:N" force a choice; if a CUDA device is requested but unavailable we warn
    and fall back to CPU rather than crashing -- so a sample/eval command still
    runs on a machine whose GPU is absent or busy.
    """
    pref = (pref or "auto").lower()
    if pref == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Validate before handing to torch.device so a typo ("gpu", "0") raises a
    # clean ValueError (main() formats it) rather than an opaque RuntimeError.
    if not (pref in ("cpu", "cuda")
            or (pref.startswith("cuda:") and pref[5:].isdigit())):
        raise ValueError(
            f"invalid --device '{pref}'; expected auto | cpu | cuda | cuda:N")
    if pref.startswith("cuda") and not torch.cuda.is_available():
        print(f"[device] '{pref}' requested but CUDA is unavailable; using CPU",
              file=sys.stderr)
        return torch.device("cpu")
    return torch.device(pref)


# ---------------------------------------------------------------------------
# 1. Data: a character-level vocabulary + window sampler
# ---------------------------------------------------------------------------
# DESIGN: fixed vocabulary + chat role markers (the Tier 7 chat campaign).
# A corpus-derived vocabulary (sorted(set(text))) ties a checkpoint to the exact
# text it was trained on: warm-starting on a second corpus hard-fails the vocab
# match if even one rare character differs. The chat pipeline instead uses one
# FIXED, corpus-independent character set so a model can pretrain on one corpus
# and fine-tune on another. Three control characters frame conversations:
#   \x01 starts a user turn, \x02 starts an assistant turn, \x03 ends an
#   assistant turn (the generation stop symbol). Single-character role markers
#   cost one input neuron and one timestep each, cannot be emitted malformed,
#   and make stopping unambiguous -- the char-level analogue of ChatML's
#   single-token <|im_end|>.
ROLE_USER = "\x01"       # start of a user turn
ROLE_ASSISTANT = "\x02"  # start of an assistant turn
ROLE_END = "\x03"        # end of an assistant turn (generation stop)
# newline + the 3 role markers + printable ASCII 32..126 = 99 symbols.
FIXED_VOCAB = "\n" + ROLE_USER + ROLE_ASSISTANT + ROLE_END + "".join(
    chr(c) for c in range(32, 127))
# A character LM has the tiniest possible vocabulary (the set of distinct
# characters in the corpus). That is a good match for an SNN whose *input layer
# has one neuron per symbol*: rate-coding a one-hot vector of size `vocab_size`
# is cheap and interpretable -- exactly one input neuron is "driven" per char.

# Built-in demo corpus so the script runs end-to-end with no external files.
# It is deliberately small and repetitive so a modest model can visibly learn
# structure (loss drops to a small fraction of a bit/char, samples become
# word-like) within a couple of thousand updates.
_DEMO_CORPUS = (
    "the spiking neuron integrates current until it crosses a threshold. "
    "when the membrane potential crosses the threshold, the neuron fires a "
    "spike and the membrane resets. between spikes the potential leaks toward "
    "rest, so recent input matters more than old input. this leak is the "
    "memory of a leaky integrate and fire neuron. a recurrent layer carries "
    "its membrane potential forward in time, so the network keeps a trace of "
    "the sequence it has seen. rate coding turns a symbol into a train of "
    "spikes whose frequency encodes the value. a surrogate gradient lets us "
    "train the network with backpropagation through time even though the spike "
    "itself is a step function. the model learns to predict the next character "
    "from the spikes of the characters before it.\n"
) * 24


def _encode_fast(text: str, stoi: dict) -> Tuple[torch.Tensor, int]:
    """Encode ``text`` to a long tensor of ids via a 256-entry byte lookup.

    The per-character Python loop costs ~15-20 bytes/char of transient memory
    and minutes of CPU on a 100 MB-class corpus; going bytes -> lookup-table is
    ~100x faster and allocation-light, which is what makes the Tier 7 corpus
    (~150 MB) loadable on a 16 GB machine. Assumes a byte-wide (latin-1-able)
    vocabulary -- true for both the classic corpus-derived vocabs (printable
    ASCII) and FIXED_VOCAB; falls back to the loop for anything wider.
    Characters absent from ``stoi`` are dropped; returns (ids, n_dropped).
    """
    try:
        raw = torch.frombuffer(bytearray(text.encode("latin-1")),
                               dtype=torch.uint8).to(torch.long)
    except UnicodeEncodeError:                       # non-latin-1 corpus: slow path
        ids = [stoi[c] for c in text if c in stoi]
        return torch.tensor(ids, dtype=torch.long), len(text) - len(ids)
    lut = torch.full((256,), -1, dtype=torch.long)
    for c, i in stoi.items():
        b = c.encode("latin-1")
        if len(b) == 1:
            lut[b[0]] = i
    mapped = lut[raw]
    keep = mapped >= 0
    n_dropped = int((~keep).sum())
    ids = mapped[keep].contiguous() if n_dropped else mapped
    # Store ids as uint8 when they fit (any vocab <= 256, i.e. always for the
    # byte-wide vocabs this path handles): 1 byte/char instead of 8 keeps a
    # ~GB-class corpus loadable on a 16 GB machine. get_batch/evaluate cast
    # their (tiny) window slices back to long.
    return ids.to(torch.uint8), n_dropped


class CharDataset:
    """Turns a text string into integer ids and serves random training windows.

    Each sample is a contiguous window of `seq_len + 1` characters. The first
    `seq_len` characters are the input; the same window shifted right by one is
    the target (classic next-character prediction).
    """

    def __init__(self, text: str, seq_len: int, val_split: float = 0.0,
                 fixed_vocab: Optional[str] = None,
                 val_text: Optional[str] = None):
        self.seq_len = seq_len
        # Sorted for determinism so a saved vocab maps ids consistently. The
        # vocabulary is built from the WHOLE text (before splitting) so the model
        # never meets an unseen character at validation time. With
        # ``fixed_vocab`` (e.g. FIXED_VOCAB) the mapping is corpus-INdependent,
        # which is what lets one checkpoint pretrain on one corpus and
        # fine-tune on another without tripping the vocab-match guard.
        chars = sorted(set(fixed_vocab)) if fixed_vocab else sorted(set(text))
        self.stoi = {c: i for i, c in enumerate(chars)}
        self.itos = {i: c for i, c in enumerate(chars)}
        self.vocab_size = len(chars)
        self.data, n_dropped = _encode_fast(text, self.stoi)
        if n_dropped:
            print(f"[data] warning: {n_dropped:,} corpus char(s) outside the "
                  f"fixed vocabulary were dropped", file=sys.stderr)
        if val_text is not None:
            # DESIGN: explicit validation corpus. A multi-source corpus's tail
            # is 100% whichever source was concatenated last, so tail-split
            # validation would score (and select checkpoints on) a single
            # register. A separately built, stratified val file avoids that;
            # it also guarantees train/val doc-disjointness at build time.
            val_data, v_dropped = _encode_fast(val_text, self.stoi)
            if v_dropped:
                print(f"[data] warning: {v_dropped:,} val char(s) outside the "
                      f"fixed vocabulary were dropped", file=sys.stderr)
            self._set_explicit_splits(self.data, val_data)
        else:
            self._set_splits(val_split)

    @classmethod
    def with_vocab(cls, text: str, seq_len: int, stoi: dict, itos: dict,
                   val_split: float = 0.0) -> "CharDataset":
        """Build a dataset over ``text`` using an EXISTING vocabulary (e.g. one
        loaded from a checkpoint) instead of deriving a fresh one, so a saved
        model is scored through the exact character->id mapping it trained with.
        Characters not in ``stoi`` are dropped (the model has no input neuron for
        them); the caller is expected to report how many. Length and val-split
        guards match ``__init__`` (both go through ``_set_splits``).
        """
        ds = cls.__new__(cls)
        ds.seq_len = seq_len
        ds.stoi = dict(stoi)
        ds.itos = {int(k): v for k, v in itos.items()}
        ds.vocab_size = len(ds.stoi)
        ds.data, _ = _encode_fast(text, ds.stoi)
        ds._set_splits(val_split)
        return ds

    def _set_explicit_splits(self, train_data: torch.Tensor,
                             val_data: torch.Tensor) -> None:
        """Install a separately supplied validation corpus (see __init__)."""
        if len(train_data) < self.seq_len + 1:
            raise ValueError(
                f"train corpus has {len(train_data):,} usable chars but "
                f"--seq-len {self.seq_len} needs at least {self.seq_len + 1}.")
        if len(val_data) < self.seq_len + 1:
            raise ValueError(
                f"val corpus has {len(val_data):,} usable chars but "
                f"--seq-len {self.seq_len} needs at least {self.seq_len + 1}.")
        self.splits = {
            "all": torch.cat([train_data, val_data]),
            "train": train_data,
            "val": val_data,
        }

    def _set_splits(self, val_split: float) -> None:
        """Guard corpus length and carve the contiguous validation tail.

        We draw windows of ``seq_len + 1`` chars, so the (post-vocab) corpus must
        contain at least one such window -- fail loudly here rather than with a
        cryptic torch.randint error inside get_batch. The LAST ``val_split``
        fraction is held out for validation; a contiguous tail (not random
        windows) keeps train and val disjoint. An ``all`` split spans everything
        (used by the ``eval`` command).
        """
        n = len(self.data)
        if n < self.seq_len + 1:
            raise ValueError(
                f"corpus has {n:,} usable chars but --seq-len {self.seq_len} "
                f"needs at least {self.seq_len + 1}; use a longer corpus or a "
                f"smaller --seq-len.")
        n_val = int(n * val_split)
        if n_val and n_val < self.seq_len + 1:
            n_val = 0  # too small to sample a window; skip validation silently
        if n - n_val < self.seq_len + 1:
            # A big --val-split on a small corpus can starve the train split.
            raise ValueError(
                f"--val-split {val_split} leaves only {n - n_val:,} chars for "
                f"training, but --seq-len {self.seq_len} needs at least "
                f"{self.seq_len + 1}; use a smaller --val-split, a longer corpus, "
                f"or a smaller --seq-len.")
        self.splits = {
            "all": self.data,
            "train": self.data[: n - n_val],
            "val": self.data[n - n_val:] if n_val else self.data[:0],
        }

    def has_val(self) -> bool:
        return len(self.splits["val"]) >= self.seq_len + 1

    def get_batch(self, batch_size: int, device: torch.device,
                  split: str = "train") -> Tuple[torch.Tensor, torch.Tensor]:
        # Every dataset that can legitimately reach get_batch is built through
        # __init__, which always sets .splits. (A checkpoint-rebuilt dataset from
        # _load_checkpoint carries only the vocab maps and is inference-only, so
        # it never reaches here.)
        d = self.splits[split]
        # The largest valid start is len(d) - seq_len - 1 (its target window ends
        # on the last char), so the exclusive upper bound for randint is
        # len(d) - seq_len -- which also lets the final window be sampled.
        max_start = len(d) - self.seq_len
        ix = torch.randint(0, max_start, (batch_size,))
        x = torch.stack([d[i:i + self.seq_len] for i in ix])                    # (B, L)
        y = torch.stack([d[i + 1:i + 1 + self.seq_len] for i in ix])            # (B, L)
        # .long(): the corpus is stored uint8 (see _encode_fast); the model's
        # one_hot/embedding wants int64, so the cast happens on the small
        # window batch, not the whole corpus.
        return x.long().to(device), y.long().to(device)

    def encode(self, s: str) -> torch.Tensor:
        return torch.tensor([self.stoi[c] for c in s if c in self.stoi],
                            dtype=torch.long)

    def decode(self, ids) -> str:
        return "".join(self.itos[int(i)] for i in ids)


# ---------------------------------------------------------------------------
# 2. The model
# ---------------------------------------------------------------------------
@dataclass
class SNNConfig:
    vocab_size: int = 0          # filled in from the dataset
    arch: str = "snn"            # "snn" (spiking LIF) | "gru" (non-spiking baseline)
    hidden: int = 512            # LIF neurons per recurrent layer
    num_layers: int = 2          # stacked Leaky layers
    num_steps: int = 5           # SNN micro-steps per character (rate-code length T)
    beta: float = 0.9            # membrane decay per micro-step (0=no memory, 1=no leak)
    threshold: float = 1.0       # firing threshold of the LIF neurons
    rate_gain: float = 0.9       # scales one-hot -> spike probability for stochastic coding
    surrogate: str = "atan"      # surrogate-gradient family: atan | fast_sigmoid | sigmoid
    learn_beta: bool = False     # make the membrane time-constant trainable
    learn_threshold: bool = False
    dropout: float = 0.1         # dropout on inter-layer spike currents (regularization)
    seq_len: int = 128           # characters per training window (context length)
    # -- Tier 6 ablation knobs (all off by default; the shipped model uses none) --
    beta_per_neuron: bool = False  # learn a per-neuron [hidden] beta vector, not one scalar
    layernorm: bool = False        # LayerNorm the inter-layer input currents
    input_coding: str = "rate"     # "rate" (stochastic spikes) | "graded" (deterministic current)


def _make_surrogate(name: str):
    """Return a snnTorch surrogate-gradient function.

    # DESIGN: surrogate gradient.
    # A spike is S = Heaviside(U - threshold); its derivative is a Dirac delta
    # (zero almost everywhere), so real backprop would kill all gradients.
    # snnTorch replaces dS/dU on the *backward* pass with a smooth bump centred
    # at threshold. The forward pass still emits hard 0/1 spikes -- only the
    # gradient is faked. ATan is a good default: bounded, symmetric, and it does
    # not vanish as fast as a sigmoid derivative for large deviations.
    """
    name = name.lower()
    if name == "atan":
        return surrogate.atan(alpha=2.0)
    if name in ("fast_sigmoid", "fastsigmoid", "fs"):
        return surrogate.fast_sigmoid(slope=25)
    if name == "sigmoid":
        return surrogate.sigmoid(slope=25)
    raise ValueError(f"unknown surrogate '{name}'")


def prompt_in_vocab(dataset: "CharDataset", prompt: str) -> str:
    """The prompt as the model actually saw it (out-of-vocab chars dropped)."""
    return "".join(c for c in prompt if c in dataset.stoi)


def _has_verbatim_loop(s: str, min_len: int = 16) -> bool:
    """True if ``s`` ends in a verbatim repetition: its tail appears again
    earlier in ``s`` (the signature of a sampling loop, cf. _generate_core)."""
    if len(s) < 2 * min_len:
        return False
    tail = s[-min_len:]
    return tail in s[:-1] and s.count(tail) >= 2


class CharLMBase(nn.Module):
    """Shared machinery for the character-LM architectures.

    Subclasses provide ``init_state(batch, device) -> state`` and
    ``step(char_onehot, state) -> (logits, state)`` plus ``self.cfg.vocab_size``.
    The autoregressive sampler below is architecture-agnostic -- it drives the
    spiking model and the GRU baseline alike.
    """

    @torch.no_grad()
    def generate(self, dataset: "CharDataset", prompt: str, length: int,
                 device: torch.device, temperature: float = 1.0,
                 top_k: Optional[int] = None, top_p: Optional[float] = None,
                 stop_char: Optional[str] = None,
                 ban_chars: Optional[str] = None) -> str:
        """Warm the recurrent state on ``prompt``, then sample ``length`` chars.

        The recurrent state persists across the whole generation, exactly as
        during training, so the memory conditions every new character on
        everything generated so far. See ``_generate_core`` for the sampling
        controls (top-p, stop character, banned characters, loop guard).
        """
        self.eval()
        state = self.init_state(1, device)
        text, _ = self._generate_core(
            dataset, state, prompt, length, device, temperature=temperature,
            top_k=top_k, top_p=top_p, stop_char=stop_char, ban_chars=ban_chars)
        return prompt_in_vocab(dataset, prompt) + text

    @torch.no_grad()
    def _generate_core(self, dataset: "CharDataset", state, prompt: str,
                       length: int, device: torch.device,
                       temperature: float = 1.0, top_k: Optional[int] = None,
                       top_p: Optional[float] = None,
                       stop_char: Optional[str] = None,
                       ban_chars: Optional[str] = None,
                       stream=None) -> Tuple[str, object]:
        """Feed ``prompt`` into ``state``, sample up to ``length`` chars, and
        return (generated_text, state) -- the caller owns the state, which is
        what lets a chat REPL keep one recurrent memory across many turns.

        Sampling controls:
          top_k / top_p     -- truncate the distribution (top_p = nucleus).
          stop_char         -- stop after sampling it (it IS consumed by the
                               state and included in the returned text, so a
                               conversation's memory stays aligned with the
                               training format; the chat UI strips it).
          ban_chars         -- never sample these (e.g. role markers that only
                               the harness may inject).
          stream            -- callable(str) invoked per generated char.

        # DESIGN: loop guard instead of repetition penalty. Char-level
        # repetition penalties punish 'e' and space, wrecking text. Verbatim
        # LOOPS are the actual small-model failure, so when the recent output
        # contains a repeated >=16-char substring we temporarily raise the
        # temperature -- enough randomness to break the cycle, no bias on
        # ordinary characters.
        """
        self.eval()
        vocab = self.cfg.vocab_size
        ban_ids = [dataset.stoi[c] for c in (ban_chars or "") if c in dataset.stoi]
        stop_id = (dataset.stoi.get(stop_char, None)
                   if stop_char is not None else None)

        last_logits = None
        ids = dataset.encode(prompt).to(device)
        for cid in ids:
            oh = F.one_hot(cid.view(1), vocab).float()
            last_logits, state = self.step(oh, state)
        if last_logits is None:
            oh = torch.zeros(1, vocab, device=device)
            last_logits, state = self.step(oh, state)

        out_ids: List[int] = []
        hot_until = 0                      # loop-guard temperature bump window
        for n in range(length):
            # Guard: NaN/inf logits would make torch.multinomial trigger a CUDA
            # device assert; sanitising keeps sampling robust on any checkpoint.
            logits = torch.nan_to_num(last_logits).squeeze(0).float()
            temp = temperature + (0.25 if n < hot_until else 0.0)
            logits = logits / max(temp, 1e-6)
            for b in ban_ids:
                logits[b] = float("-inf")
            if top_k is not None and top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.numel()))
                logits[logits < v[-1]] = float("-inf")
            if top_p is not None and 0.0 < top_p < 1.0:
                probs_sorted, order = torch.sort(F.softmax(logits, dim=-1),
                                                 descending=True)
                keep = torch.cumsum(probs_sorted, dim=-1) - probs_sorted < top_p
                keep[0] = True             # always keep the argmax
                logits[order[~keep]] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            next_id = int(torch.multinomial(probs, 1).item())
            out_ids.append(next_id)
            ch = dataset.itos[next_id]
            if stream is not None:
                stream(ch)
            oh = F.one_hot(torch.tensor([next_id], device=device),
                           vocab).float()
            last_logits, state = self.step(oh, state)
            if stop_id is not None and next_id == stop_id:
                break
            if n >= hot_until and (n + 1) % 16 == 0 and len(out_ids) >= 48:
                if _has_verbatim_loop(dataset.decode(out_ids[-64:])):
                    hot_until = n + 32
        return dataset.decode(out_ids), state

    def detach_state(self, state):
        """Detach the recurrent state between TBPTT chunks (arch-agnostic): the
        SNN carries a list of membrane tensors, the GRU a single hidden tensor.
        """
        if isinstance(state, (list, tuple)):
            return [s.detach() for s in state]
        return state.detach()


class SNNCharLM(CharLMBase):
    """A stack of recurrent leaky integrate-and-fire layers with a linear
    rate-decoding head.

    Two nested time axes are at play, and keeping them straight is the key to
    understanding the model:

      * The *sequence* axis (length L): characters c_0, c_1, ... The LIF membrane
        potential is carried across this axis -- that carried state is the
        language memory (this is what makes the network *recurrent*).
      * The *SNN micro-step* axis (length T = num_steps): within one character we
        present its rate-coded spike train for T steps and let the neurons
        integrate. Reading out the mean spike *rate* over these T steps is what
        "rate decoding" means.

    So one forward pass unrolls L * T LIF updates. We predict the next character
    at every one of the L positions (dense supervision).

    # DESIGN: recurrence via the leaky membrane (not an explicit weight matrix).
    # An `snn.Leaky` neuron obeys  U_t = beta * U_{t-1} + I_t ;  S_t = (U_t > thr).
    # Because U_t depends on U_{t-1}, carrying U across the sequence makes each
    # layer a genuine recurrent unit -- the spiking analogue of a leaky RNN. We
    # deliberately do NOT add an explicit all-to-all lateral recurrent matrix
    # (snnTorch's `RLeaky`). Empirically, on this next-character task, a learned
    # lateral spike-feedback matrix creates a chaotic dynamical system: it makes
    # the model fail to learn (it collapses to predicting character frequencies)
    # and its backprop-through-time gradient explodes past float32. The leak's
    # self-recurrence (spectral radius beta < 1) is contractive, so it is both
    # stable and, with depth, expressive enough to model long-range structure.
    """

    def __init__(self, cfg: SNNConfig):
        super().__init__()
        self.cfg = cfg
        spike_grad = _make_surrogate(cfg.surrogate)

        # Feed-forward ("afferent") projections: one Linear per layer that turns
        # the incoming spikes into an input current for that layer's neurons.
        #   layer 0 sees the rate-coded one-hot character (size vocab_size)
        #   layer l>0 sees the spikes of layer l-1 (size hidden)
        self.fc_in = nn.ModuleList()
        # Leaky integrate-and-fire cells. Their membrane state, threaded across
        # the sequence, is the recurrent memory.
        self.lifs = nn.ModuleList()
        # ABLATION: per-neuron learnable beta. snnTorch accepts a [hidden] beta
        # vector; with learn_beta each neuron gets its own membrane decay, and the
        # forward pass clamps it to [0, 1] so it stays in the contractive regime.
        learn_beta_arg = cfg.learn_beta or cfg.beta_per_neuron
        for l in range(cfg.num_layers):
            in_dim = cfg.vocab_size if l == 0 else cfg.hidden
            self.fc_in.append(nn.Linear(in_dim, cfg.hidden))
            # Fresh beta tensor per layer so learnable per-neuron betas don't share
            # a Parameter across layers.
            beta_arg = (torch.full((cfg.hidden,), float(cfg.beta))
                        if cfg.beta_per_neuron else cfg.beta)
            self.lifs.append(
                snn.Leaky(
                    beta=beta_arg,                 # DESIGN: membrane leak (see below)
                    threshold=cfg.threshold,
                    spike_grad=spike_grad,         # surrogate gradient for dS/dU
                    learn_beta=learn_beta_arg,     # optionally learn the time constant
                    learn_threshold=cfg.learn_threshold,
                    reset_mechanism="subtract",    # DESIGN: soft reset (see below)
                    init_hidden=False,             # we manage the membrane ourselves
                )
            )

        # DESIGN: membrane leak `beta`.
        # After each micro-step U <- beta*U + input. beta in (0,1) makes the
        # neuron a leaky integrator: a low beta forgets quickly (reacts to the
        # latest input), a high beta remembers longer. Across characters this
        # decaying trace is the network's short-term memory. With T micro-steps
        # per character the per-character retention is beta**T.

        # DESIGN: reset mechanism "subtract" (soft reset).
        # On firing we subtract the threshold from U instead of zeroing it, so
        # any "overshoot" above threshold is preserved for the next step. This
        # keeps information that a hard reset-to-zero would discard and tends to
        # train more stably.

        # Dropout applied to the *currents* flowing between layers. We do not
        # drop spikes directly (that would corrupt the binary code); dropping the
        # analog current is the SNN-friendly way to regularize.
        self.drop = nn.Dropout(cfg.dropout)

        # ABLATION: LayerNorm on the per-layer input currents. Normalizing the
        # drive into each LIF can stabilize training of deeper stacks (the README
        # sweep found extra depth hurt). One LayerNorm per layer; off by default.
        self.norms = (nn.ModuleList(nn.LayerNorm(cfg.hidden)
                                    for _ in range(cfg.num_layers))
                      if cfg.layernorm else None)

        # DESIGN: rate-decoding readout.
        # The final prediction is a plain Linear on the top layer's *mean firing
        # rate* over the T micro-steps. Rates are smooth and real-valued, so the
        # classifier head has a clean gradient -- much friendlier than stacking
        # another threshold neuron at the output and counting its spikes. This is
        # the standard "rate decode" readout for SNNs.
        self.readout = nn.Linear(cfg.hidden, cfg.vocab_size)

    # -- state helpers -------------------------------------------------------
    def init_state(self, batch_size: int, device: torch.device
                   ) -> List[torch.Tensor]:
        """Zeroed membrane potential for every layer at the start of a sequence.

        We hold the recurrent state (one membrane tensor per layer) as ordinary
        tensors and thread them through the loops. That gives us explicit control
        over backprop-through-time: to truncate BPTT we just `.detach()` these
        between chunks (see train()).
        """
        return [torch.zeros(batch_size, self.cfg.hidden, device=device)
                for _ in range(self.cfg.num_layers)]

    # -- one character's worth of SNN dynamics -------------------------------
    def _char_rate(self, cur0: torch.Tensor, mems: List[torch.Tensor]
                   ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Run the T micro-steps for one character; return its top-layer rate.

        cur0 : (T, B, hidden) pre-projected layer-0 input current (the rate-coded
               spikes already passed through fc_in[0]). Pre-projecting once
               outside the loop keeps the many tiny per-step kernels off the
               critical path.
        mems : list of per-layer membrane potentials (B, hidden), updated in place
               and returned so they carry to the next character.
        Returns the mean top-layer firing rate (B, hidden), in [0, 1].
        """
        top_spike_sum = torch.zeros_like(mems[-1])
        for t in range(self.cfg.num_steps):
            cur = cur0[t]                                             # (B, hidden)
            for l, lif in enumerate(self.lifs):
                # Layer 0's afferent drive is pre-computed; deeper layers project
                # the spikes of the layer below at every step.
                current = cur if l == 0 else self.drop(self.fc_in[l](cur))
                if self.norms is not None:
                    current = self.norms[l](current)                 # ABLATION: normalize drive
                # Leaky integrates the current into its carried membrane and
                # emits a spike; the returned membrane is the recurrent state.
                spk, mems[l] = lif(current, mems[l])
                cur = spk                                            # feed spikes up
            top_spike_sum = top_spike_sum + cur
        return top_spike_sum / self.cfg.num_steps, mems             # rate in [0, 1]

    # -- one character step (used for autoregressive sampling) ---------------
    def step(self, char_onehot: torch.Tensor, mems: List[torch.Tensor]
             ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Process a single character and return its next-char logits.

        char_onehot : (B, vocab_size) float one-hot input.
        Returns logits (B, vocab_size) and the updated membrane state.

        # DESIGN: rate coding.
        # spikegen.rate draws, for each of T micro-steps, a Bernoulli sample with
        # p = gain * feature. With a one-hot input and gain<1 the "on" neuron
        # fires on ~gain fraction of steps and the rest never fire: a genuine
        # stochastic rate code rather than a constant. Averaging the network's
        # response over these T noisy steps is a lightweight form of ensembling.
        """
        if self.cfg.input_coding == "graded":
            # ABLATION: deterministic graded coding -- inject the one-hot's
            # projection as a constant current for all T micro-steps (no spikes).
            base = self.fc_in[0](char_onehot)                            # (B, hidden)
            cur0 = base.unsqueeze(0).expand(self.cfg.num_steps, -1, -1)  # (T, B, hidden)
        else:
            spike_input = spikegen.rate(char_onehot, num_steps=self.cfg.num_steps,
                                        gain=self.cfg.rate_gain)         # (T, B, V)
            cur0 = self.fc_in[0](spike_input)                            # (T, B, hidden)
        rate, mems = self._char_rate(cur0, mems)
        return self.readout(rate), mems

    # -- a (sub)sequence, carrying recurrent state in and out ----------------
    def forward_seq(self, x: torch.Tensor,
                    mems: Optional[List[torch.Tensor]] = None
                    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Run a chunk of characters and return (logits, new_mems).

        x    : (B, L) character ids.
        mems : incoming membrane state, or None to start from zeros.
        Returns logits (B, L, vocab_size) and the membrane state after the last
        char -- the caller threads that state into the next chunk (see the
        truncated-BPTT loop in train()). Passing state in/out is what lets memory
        span a long sequence while each backward pass stays short.
        """
        B, L = x.shape
        if mems is None:
            mems = self.init_state(B, x.device)
        onehot = F.one_hot(x, self.cfg.vocab_size).float()             # (B, L, V)
        if self.cfg.input_coding == "graded":
            # ABLATION: deterministic graded coding -- inject the one-hot's
            # projection as a constant current for all T micro-steps (no spikes).
            base = self.fc_in[0](onehot)                               # (B, L, hidden)
            cur0_all = base.unsqueeze(0).expand(self.cfg.num_steps, -1, -1, -1)
        else:
            # Encode and project the WHOLE chunk at once: one rate-coding call and
            # one big matmul replace L per-position encodes and L*T micro-GEMMs.
            # Numerically identical (given the same spikes) to encoding per
            # position -- only the sequential recurrent loop below is unavoidable.
            spikes = spikegen.rate(onehot, num_steps=self.cfg.num_steps,
                                   gain=self.cfg.rate_gain)            # (T, B, L, V)
            cur0_all = self.fc_in[0](spikes)                           # (T, B, L, hidden)
        return self._run_core(cur0_all, mems)

    def _run_core(self, cur0_all: torch.Tensor, mems: List[torch.Tensor]
                  ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Sequential recurrent loop over a chunk's pre-encoded input currents.

        cur0_all : (T, B, L, hidden) -- the (possibly RNG-dependent) encoder output
                   computed in forward_seq BEFORE this call. Splitting it out keeps
                   this core RNG-free and fixed-shape, which is exactly what a CUDA
                   graph can capture and replay (see the `bench` command / TIER6.md).
        Returns logits (B, L, vocab_size) and the membrane state after the last char.
        """
        L = cur0_all.shape[2]
        rates = []
        for pos in range(L):
            rate, mems = self._char_rate(cur0_all[:, :, pos, :], mems)
            rates.append(rate)
        rates = torch.stack(rates, dim=1)                              # (B, L, hidden)
        return self.readout(rates), mems                              # (B, L, V) one GEMM

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, L) ids -> logits (B, L, vocab_size), fresh state (for eval)."""
        return self.forward_seq(x, None)[0]

    @torch.no_grad()
    def spike_stats(self, x: torch.Tensor) -> List[float]:
        """Per-layer mean firing rate over a probe batch ``x`` (B, L) in [0, 1].

        This is the health metric of the spiking story: with graded input
        coding the hidden layers' spikes ARE the spiking computation, and
        LayerNorm can silently compensate for a dead (~0%) or saturated
        (~100%) layer while bpc still looks fine. The training loop logs these
        every eval and the campaign alarms outside roughly [2%, 90%].
        (A read-only mirror of forward_seq's loop -- the hot path stays
        untouched so it remains CUDA-graph-capturable.)
        """
        was_training = self.training
        self.eval()
        B, L = x.shape
        mems = self.init_state(B, x.device)
        onehot = F.one_hot(x, self.cfg.vocab_size).float()
        if self.cfg.input_coding == "graded":
            base = self.fc_in[0](onehot)
            cur0_all = base.unsqueeze(0).expand(self.cfg.num_steps, -1, -1, -1)
        else:
            spikes = spikegen.rate(onehot, num_steps=self.cfg.num_steps,
                                   gain=self.cfg.rate_gain)
            cur0_all = self.fc_in[0](spikes)
        sums = [torch.zeros((), device=x.device)
                for _ in range(self.cfg.num_layers)]
        n_micro = 0
        for pos in range(L):
            cur0 = cur0_all[:, :, pos, :]
            for t in range(self.cfg.num_steps):
                cur = cur0[t]
                for l, lif in enumerate(self.lifs):
                    current = cur if l == 0 else self.fc_in[l](cur)
                    if self.norms is not None:
                        current = self.norms[l](current)
                    spk, mems[l] = lif(current, mems[l])
                    sums[l] = sums[l] + spk.mean()
                    cur = spk
                n_micro += 1
        if was_training:
            self.train()
        return [float(s.item()) / n_micro for s in sums]


class GRUCharLM(CharLMBase):
    """Non-spiking GRU character LM -- a matched in-harness baseline for the SNN.

    Same data pipeline, training loop, evaluation, and sampler, and the same
    interface (init_state / forward_seq / step / forward) as SNNCharLM; the only
    differences are a standard GRU core instead of the LIF stack and a learned
    embedding instead of rate coding. It exists to give the README's "above the
    best ANNs" comparison a same-corpus, same-budget control rather than a
    cross-corpus figure. (The SNN-only cfg fields -- num_steps, beta, rate_gain,
    the ablation knobs -- are simply ignored here.)
    """

    def __init__(self, cfg: SNNConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.hidden)
        self.gru = nn.GRU(cfg.hidden, cfg.hidden, num_layers=cfg.num_layers,
                          batch_first=True,
                          dropout=cfg.dropout if cfg.num_layers > 1 else 0.0)
        self.drop = nn.Dropout(cfg.dropout)
        self.readout = nn.Linear(cfg.hidden, cfg.vocab_size)

    def init_state(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(self.cfg.num_layers, batch_size, self.cfg.hidden,
                           device=device)

    def forward_seq(self, x: torch.Tensor, state: Optional[torch.Tensor] = None
                    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, L = x.shape
        if state is None:
            state = self.init_state(B, x.device)
        emb = self.drop(self.embed(x))                     # (B, L, hidden)
        out, state = self.gru(emb, state)                  # (B, L, hidden)
        return self.readout(self.drop(out)), state         # (B, L, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_seq(x, None)[0]

    def step(self, char_onehot: torch.Tensor, state: torch.Tensor
             ) -> Tuple[torch.Tensor, torch.Tensor]:
        # generate() hands us a one-hot; recover the id and embed it (one step).
        ids = char_onehot.argmax(dim=-1)                   # (B,)
        out, state = self.gru(self.embed(ids).unsqueeze(1), state)
        return self.readout(out.squeeze(1)), state


def build_model(cfg: SNNConfig) -> CharLMBase:
    """Construct the model named by cfg.arch ("snn" | "gru")."""
    if cfg.arch == "gru":
        return GRUCharLM(cfg)
    if cfg.arch == "snn":
        return SNNCharLM(cfg)
    raise ValueError(f"unknown arch '{cfg.arch}'; expected snn | gru")


# ---------------------------------------------------------------------------
# 2b. CUDA-graph training capture + chat metrics (Tier 7)
# ---------------------------------------------------------------------------
def _require_graphable(cfg: SNNConfig, device: torch.device, args) -> None:
    """Fail fast (with every reason at once) if the config cannot be captured.

    The captured region must be fixed-shape and RNG-free: graded input coding
    removes the stochastic encoder, dropout 0 removes the last RNG op, and a
    seq_len that divides evenly into chunks keeps every replay's shapes
    identical to the capture's.
    """
    problems = []
    if device.type != "cuda":
        problems.append("a CUDA device")
    if cfg.arch != "snn":
        problems.append("--arch snn")
    if cfg.input_coding != "graded":
        problems.append("--input-coding graded (RNG-free capture)")
    if cfg.dropout != 0.0:
        problems.append("--dropout 0 (RNG-free capture)")
    if args.seq_len % args.tbptt_chunk != 0:
        problems.append("--seq-len divisible by --tbptt-chunk (fixed shapes)")
    if problems:
        raise ValueError("--cuda-graph requires " + ", ".join(problems))


def _capture_chunk_step(model: "SNNCharLM", cfg: SNNConfig, batch_size: int,
                        chunk: int, device: torch.device,
                        warm_lr: float) -> dict:
    """Warm up and capture one TBPTT chunk's forward+loss+backward as a CUDA
    graph; returns the graph plus the static buffers a caller replays through.

    Follows the official whole-iteration capture recipe (see the DESIGN note at
    the call site in train()): 3 full warmup iterations on a side stream, model
    weights snapshotted/restored so warmup does not perturb training, grads set
    to None once so backward allocates them inside the graph's private pool.
    """
    static_x = torch.zeros(batch_size, chunk, dtype=torch.long, device=device)
    static_y = torch.zeros(batch_size, chunk, dtype=torch.long, device=device)
    static_mems = model.init_state(batch_size, device)
    snapshot = {k: v.detach().clone() for k, v in model.state_dict().items()}
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        warm_opt = torch.optim.AdamW(model.parameters(), lr=warm_lr)
        for _ in range(3):
            warm_opt.zero_grad(set_to_none=True)
            w_logits, _ = model.forward_seq(
                static_x, [mm.clone() for mm in static_mems])
            w_loss = F.cross_entropy(
                w_logits.reshape(-1, cfg.vocab_size), static_y.reshape(-1))
            w_loss.backward()
            warm_opt.step()
    torch.cuda.current_stream().wait_stream(side)
    model.load_state_dict(snapshot)       # in-place copy: addresses preserved
    del warm_opt, snapshot, w_logits, w_loss
    for p in model.parameters():
        p.grad = None                     # allocate .grad inside the graph pool
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        static_logits, static_new_mems = model.forward_seq(
            static_x, list(static_mems))
        static_loss = F.cross_entropy(
            static_logits.reshape(-1, cfg.vocab_size), static_y.reshape(-1))
        static_loss.backward()
    return {"graph": graph, "x": static_x, "y": static_y, "mems": static_mems,
            "new_mems": static_new_mems, "loss": static_loss}


def _load_chat_pairs(path: str) -> List[dict]:
    """Load (user, assistant) conditioning pairs from a JSONL file (one object
    with 'user' and 'assistant' keys per line; built by build_chat_corpus.py).
    """
    pairs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                pairs.append(json.loads(line))
    return pairs


@torch.no_grad()
def conditioning_metrics(model: CharLMBase, dataset: CharDataset,
                         pairs: List[dict], device: torch.device,
                         max_pairs: int = 96, shuffle_seed: int = 7) -> dict:
    """The metric that catches fluent-but-unconditioned chat models.

    Scores the bpc of each ASSISTANT span (its characters plus the end-of-turn
    marker, teacher-forced) twice: once conditioned on the TRUE user turn and
    once on a SHUFFLED one (a fixed derangement of the users across pairs).
    ``gain = bpc_shuffled - bpc_true``: a model whose replies actually depend
    on the prompt scores the true pairing distinctly better (positive gain);
    a model that emits generic assistant register regardless of the prompt --
    the most likely tiny-model failure, invisible to val bpc, word-validity
    and cherry-picked transcripts alike -- scores ~0.
    """
    was_training = model.training
    model.eval()
    pairs = pairs[:max_pairs]
    users = [p["user"] for p in pairs]
    assists = [p["assistant"] for p in pairs]
    perm = torch.randperm(
        len(users),
        generator=torch.Generator().manual_seed(shuffle_seed)).tolist()
    for i in range(len(perm)):            # force a derangement: no fixed points
        if perm[i] == i:
            j = (i + 1) % len(perm)
            perm[i], perm[j] = perm[j], perm[i]
    vocab = model.cfg.vocab_size
    pad_id = dataset.stoi.get(" ", 0)

    def spans_bpc(user_list: List[str]) -> float:
        total_nats, total_chars = 0.0, 0
        B = 32
        for i0 in range(0, len(pairs), B):
            us = user_list[i0:i0 + B]
            as_ = assists[i0:i0 + B]
            encs = [dataset.encode(ROLE_USER + u + ROLE_ASSISTANT + a + ROLE_END)
                    for u, a in zip(us, as_)]
            n = len(encs)
            lmax = max(len(e) for e in encs)
            xs = torch.full((n, lmax - 1), pad_id, dtype=torch.long)
            ys = torch.full((n, lmax - 1), pad_id, dtype=torch.long)
            mask = torch.zeros(n, lmax - 1, dtype=torch.bool)
            for k, e in enumerate(encs):
                le = len(e)
                xs[k, :le - 1] = e[:-1]
                ys[k, :le - 1] = e[1:]
                # Loss only over the assistant span + the end marker. In
                # target space (y = seq shifted left by 1) that span starts at
                # index len(\x01 + user) = len(user)+1 and covers
                # len(assistant)+1 characters.
                a_start = len(us[k]) + 1
                mask[k, a_start:a_start + len(as_[k]) + 1] = True
            logits = model(xs.to(device))
            nll = F.cross_entropy(logits.reshape(-1, vocab),
                                  ys.to(device).reshape(-1), reduction="none")
            nll = nll.view(n, -1)[mask.to(device)]
            total_nats += float(nll.sum())
            total_chars += int(mask.sum())
        return total_nats / max(1, total_chars) / math.log(2)

    bpc_true = spans_bpc(users)
    bpc_shuf = spans_bpc([users[p] for p in perm])
    if was_training:
        model.train()
    return {"bpc_true": bpc_true, "bpc_shuffled": bpc_shuf,
            "gain": bpc_shuf - bpc_true, "n_pairs": len(pairs)}


# ---------------------------------------------------------------------------
# 3. Training loop
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate(model: CharLMBase, dataset: CharDataset, split: str,
             batch_size: int, n_batches: int, device: torch.device,
             deterministic: bool = False, encoder_seed: int = 1234,
             max_windows: int = 0) -> float:
    """Mean bits-per-character on held-out windows (lower is better).

    We run full windows with a fresh membrane and no gradient -- the same
    conditions the model trains under -- and average the per-character cross
    entropy. On the `val` split this measures generalization, not memorization.

    The default estimate has two noise sources: which windows are drawn (random)
    and the stochastic rate encoder. With ``deterministic=True`` (Tier 6) we
    instead sweep a FIXED set of contiguous, non-overlapping windows tiling the
    split, under a FIXED encoder RNG -- so the reported bpc is exactly reproducible
    run-to-run and the best-checkpoint comparison is not made against estimator
    noise. ``n_batches`` is ignored in that mode (the whole split is swept once).
    """
    was_training = model.training
    model.eval()
    total_nats, total_chars = 0.0, 0
    vocab = model.cfg.vocab_size
    if deterministic:
        d = dataset.splits[split]
        L = dataset.seq_len
        # Non-overlapping windows tiling the split; the split guard guarantees
        # len(d) >= L + 1, so at least the window at start 0 exists.
        starts = list(range(0, len(d) - L, L)) or [0]
        if max_windows and len(starts) > max_windows:
            # Deterministic SUBSAMPLE: evenly spaced windows across the split.
            # On a 100 MB-class corpus the full sweep costs tens of minutes per
            # eval, which would throttle both the eval and the save-state
            # cadence; the subsample keeps in-run selection reproducible and
            # cheap, and the full sweep is still used for end-of-stage scoring.
            idx = torch.linspace(0, len(starts) - 1, max_windows).round().long()
            starts = [starts[i] for i in torch.unique(idx).tolist()]
        fork_devices = [device] if device.type == "cuda" else []
        with torch.random.fork_rng(devices=fork_devices):
            torch.manual_seed(encoder_seed)
            if device.type == "cuda":
                torch.cuda.manual_seed(encoder_seed)
            for i in range(0, len(starts), batch_size):
                chunk = starts[i:i + batch_size]
                x = torch.stack([d[s:s + L] for s in chunk]).long().to(device)
                y = torch.stack([d[s + 1:s + 1 + L]
                                 for s in chunk]).long().to(device)
                logits = model(x)
                loss = F.cross_entropy(logits.reshape(-1, vocab),
                                       y.reshape(-1), reduction="sum")
                total_nats += loss.item()
                total_chars += y.numel()
    else:
        for _ in range(n_batches):
            x, y = dataset.get_batch(batch_size, device, split=split)
            logits = model(x)                                          # (B, L, V)
            loss = F.cross_entropy(logits.reshape(-1, vocab),
                                   y.reshape(-1), reduction="sum")
            total_nats += loss.item()
            total_chars += y.numel()
    if was_training:
        model.train()
    return total_nats / total_chars / math.log(2)


def train(args) -> None:
    resume_from = getattr(args, "resume", None)
    # On resume we restore the checkpoint's saved RNG below; seeding here would
    # clobber it. A fresh or warm-started run seeds normally for reproducibility.
    if not resume_from:
        set_seed(args.seed)
    device = pick_device(getattr(args, "device", "auto"))

    text = _load_text(args.data)
    fixed = FIXED_VOCAB if getattr(args, "vocab", "corpus") == "fixed" else None
    val_text = None
    if getattr(args, "val_data", None):
        with open(args.val_data, "r", encoding="utf-8") as f:
            val_text = f.read()
    dataset = CharDataset(text, seq_len=args.seq_len, val_split=args.val_split,
                          fixed_vocab=fixed, val_text=val_text)
    n_val = len(dataset.splits["val"])
    print(f"[data] {len(text):,} chars | vocab={dataset.vocab_size}"
          f"{' (fixed)' if fixed else ''} "
          f"| train={len(dataset.splits['train']):,} val={n_val:,}"
          f"{' (separate file)' if val_text is not None else ''} "
          f"| device={device}")

    resume_ckpt = None
    if resume_from:
        # Resume: continue a previous run -- same weights, optimizer state, update
        # counter, best-val, and RNG (weights_only=True: safe load). The LR
        # schedule continues exactly only when --steps/--warmup are unchanged;
        # raising --steps to extend a finished run redefines the cosine horizon
        # and jumps the LR (an SGDR-style warm restart -- see the resume warning).
        resume_ckpt = torch.load(resume_from, map_location=device, weights_only=True)
        if "opt_state" not in resume_ckpt:
            raise ValueError(
                f"--resume {resume_from} has no saved training state (it was not "
                f"written with --save-state); use --init-from for a fresh-optimizer "
                f"warm start instead.")
        _require_matching_vocab(resume_ckpt, dataset, "--resume")
        cfg = SNNConfig(**resume_ckpt["cfg"])
        cfg.dropout = args.dropout
        cfg.seq_len = args.seq_len
        model = build_model(cfg).to(device)
        model.load_state_dict(resume_ckpt["state_dict"])
    elif args.init_from:
        # Warm start: continue from saved WEIGHTS only (fresh optimizer + LR
        # schedule). Architecture comes from the checkpoint -- CLI architecture
        # flags are ignored -- but dropout and seq_len are taken from the CLI so a
        # continuation can retune those. weights_only=True: safe load (pure
        # tensors + dicts; never executes pickle code from a shared .pt).
        ckpt = torch.load(args.init_from, map_location=device, weights_only=True)
        _require_matching_vocab(ckpt, dataset, "--init-from")
        cfg = SNNConfig(**ckpt["cfg"])
        cfg.dropout = args.dropout
        cfg.seq_len = args.seq_len
        model = build_model(cfg).to(device)
        model.load_state_dict(ckpt["state_dict"])
        print(f"[init] warm start from {args.init_from} "
              f"(arch from checkpoint: hidden={cfg.hidden} layers={cfg.num_layers})")
    else:
        cfg = SNNConfig(
            vocab_size=dataset.vocab_size,
            arch=args.arch,
            hidden=args.hidden,
            num_layers=args.layers,
            num_steps=args.num_steps,
            beta=args.beta,
            threshold=args.threshold,
            rate_gain=args.rate_gain,
            surrogate=args.surrogate,
            learn_beta=args.learn_beta,
            learn_threshold=args.learn_threshold,
            dropout=args.dropout,
            seq_len=args.seq_len,
            beta_per_neuron=args.beta_per_neuron,
            layernorm=args.layernorm,
            input_coding=args.input_coding,
        )
        model = build_model(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    arch_bits = (f"T={cfg.num_steps} beta={cfg.beta}" if cfg.arch == "snn"
                 else "(non-spiking GRU baseline)")
    print(f"[model] {n_params/1e6:.2f}M params | arch={cfg.arch} "
          f"hidden={cfg.hidden} layers={cfg.num_layers} {arch_bits}")
    # Resolved-config echo: make every run's stdout fully record what produced it
    # (lr, weight-decay, warmup, dropout, grad-clip, seed, surrogate, ...). cfg
    # holds the EFFECTIVE architecture -- which on --resume/--init-from comes from
    # the checkpoint, not the CLI arch flags -- so overlay it on the run args.
    echo = {k: v for k, v in vars(args).items() if k != "func"}
    echo.update(asdict(cfg))
    echo["layers"] = cfg.num_layers   # arg dest is 'layers'; cfg field is 'num_layers'
    print(f"[config] {echo}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay)
    # DESIGN: LR warmup + cosine decay. `--steps` counts optimizer updates, which
    # is exactly the schedule horizon. A short warmup lets the spiking dynamics
    # settle before cosine-decaying to a fine polish.
    def lr_at(update: int) -> float:
        if update < args.warmup:
            return args.lr * (update + 1) / max(1, args.warmup)
        prog = (update - args.warmup) / max(1, args.steps - args.warmup)
        return args.lr * 0.5 * (1 + math.cos(math.pi * min(prog, 1.0)))

    start_update = 0
    best_val = float("inf")
    if resume_from:
        opt.load_state_dict(resume_ckpt["opt_state"])
        # load_state_dict restored the checkpoint's param-group hyperparameters;
        # re-apply the CLI weight decay so --weight-decay is honored on resume
        # (consistent with --lr, which lr_at re-sets every step).
        for g in opt.param_groups:
            g["weight_decay"] = args.weight_decay
        start_update = int(resume_ckpt.get("update", 0))
        best_val = float(resume_ckpt.get("best_val", float("inf")))
        _restore_rng(resume_ckpt)
        shown_best = f"{best_val:.3f}" if best_val != float("inf") else "n/a"
        print(f"[resume] from {resume_from} at update {start_update}/{args.steps} "
              f"| best val bpc {shown_best}")
        saved_steps = resume_ckpt.get("steps")
        saved_warmup = resume_ckpt.get("warmup")
        if saved_steps is not None and (saved_steps != args.steps
                                        or saved_warmup != args.warmup):
            print(f"[resume] WARNING: LR-schedule horizon changed (steps "
                  f"{saved_steps}->{args.steps}, warmup {saved_warmup}->{args.warmup}); "
                  f"the cosine curve is reshaped and the LR jumps at this boundary "
                  f"(SGDR-style warm restart), so this run won't match the original "
                  f"tail or an uninterrupted run of the new length.", file=sys.stderr)
        if start_update >= args.steps:
            print(f"[resume] already at/over --steps {args.steps}; raise --steps "
                  f"to train further.")

    chunk = args.tbptt_chunk
    print(f"[train] truncated BPTT: window={args.seq_len} chunk={chunk} "
          f"-> backprop depth = chunk*T = {chunk * cfg.num_steps} spiking steps")

    if getattr(args, "tf32", False):
        # TF32 matmuls: measured as a non-lever while the net was launch-bound,
        # but once a CUDA graph removes the launch overhead the matmul share
        # rises, so it is exposed as an opt-in knob (ablated, not assumed).
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print("[train] TF32 matmuls enabled")

    model.train()
    use_graph = bool(getattr(args, "cuda_graph", False))
    graph = static_x = static_y = static_loss = None
    static_mems = static_new_mems = None
    if use_graph:
        _require_graphable(cfg, device, args)
        # DESIGN: CUDA-graph training step (Tier 7). The SNN's training step is
        # a chain of layers*T*chunk tiny sequential kernels, so it is bound by
        # kernel-LAUNCH overhead, not math (TIER6 measured the forward core at
        # 8-13x under graph replay). Capturing one TBPTT chunk's
        # forward+loss+backward and replaying it as a single launch removes
        # that overhead from training itself. The capture recipe follows the
        # official CUDA-graphs notes exactly:
        #   * 3 warmup iterations of the full fwd+loss+bwd on a side stream
        #     (cuBLAS autotuning, workspace allocation), then the model weights
        #     are restored so warmup does not perturb training;
        #   * grads set to None ONCE before capture, so backward allocates the
        #     .grad tensors inside the graph's private pool -- each replay then
        #     REWRITES those same buffers in place (never call zero_grad after
        #     capture: set_to_none=True would free the very addresses the
        #     captured kernels write to);
        #   * clip_grad_norm_ and AdamW.step() stay EAGER between replays --
        #     in-place reads/writes of .grad and params at stable addresses are
        #     safe by construction (the docs' own partial-capture pattern);
        #   * the captured region is RNG-free (graded coding, dropout 0 --
        #     enforced by _require_graphable), sidestepping the whole class of
        #     Philox-replay concerns, and it stays well under WDDM's 2 s TDR.
        # Eval and sampling keep the ordinary eager module: different shapes.
        gs = _capture_chunk_step(model, cfg, args.batch_size, chunk, device,
                                 warm_lr=args.lr)
        graph, static_x, static_y = gs["graph"], gs["x"], gs["y"]
        static_mems, static_new_mems = gs["mems"], gs["new_mems"]
        static_loss = gs["loss"]
        print(f"[graph] captured fwd+loss+bwd of one {args.batch_size}x{chunk} "
              f"chunk ({cfg.num_layers} layers x T={cfg.num_steps} x {chunk} "
              f"chars); clip + optimizer step remain eager")

    csv_writer, csv_file = _open_metrics_csv(getattr(args, "log_csv", None))
    do_val = dataset.has_val()
    # Fixed probe batch for the firing-rate health metric (SNN only): the first
    # up-to-16 non-overlapping val (or train) windows, so the statistic is
    # comparable across the whole run.
    probe_x = None
    if cfg.arch == "snn":
        d_probe = dataset.splits["val" if do_val else "train"]
        n_probe = max(1, min(16, (len(d_probe) - 1) // args.seq_len))
        probe_x = torch.stack(
            [d_probe[i * args.seq_len:(i + 1) * args.seq_len]
             for i in range(n_probe)]).long().to(device)
    chat_pairs = None
    if getattr(args, "chat_pairs", None):
        chat_pairs = _load_chat_pairs(args.chat_pairs)
        print(f"[chat] {len(chat_pairs)} conditioning pairs loaded for in-run "
              f"chat metrics")

    running = torch.zeros((), device=device)   # GPU-side accumulator: one
    n_since_log = 0                            # .item() sync per LOG, not step
    last_gnorm = None
    state_every_s = getattr(args, "state_every_min", 15) * 60.0
    last_state_t = time.time()
    t0 = time.time()
    update = start_update
    interrupted = False
    try:
        while update < args.steps:
            # A fresh random window of the corpus. The membrane is reset to zero
            # at the window start; within the window it persists across chunks.
            x, y = dataset.get_batch(args.batch_size, device)          # (B, L)
            # DESIGN: truncated backpropagation through time (TBPTT).
            # We walk the window in chunks of `chunk` characters. The membrane
            # state is carried from one chunk to the next -- so the network still
            # integrates context across the whole window -- but it is DETACHED at
            # each boundary, so backprop only unrolls chunk*num_steps spiking steps
            # instead of seq_len*num_steps. That bounds both memory and the length
            # of the gradient path, letting --seq-len be long without blowing up.
            # (In graph mode the detach is structural: the static membrane
            # buffers are non-grad inputs of the captured graph.)
            if use_graph:
                for mm in static_mems:
                    mm.zero_()                     # fresh membrane per window
            else:
                mems = model.init_state(args.batch_size, device)
            for c0 in range(0, x.size(1), chunk):
                if update >= args.steps:
                    break
                for g in opt.param_groups:
                    g["lr"] = lr_at(update)
                if use_graph:
                    static_x.copy_(x[:, c0:c0 + chunk])
                    static_y.copy_(y[:, c0:c0 + chunk])
                    graph.replay()                 # fwd+loss+bwd, one launch
                    # DESIGN: gradient clipping -- standard insurance for a
                    # recurrent net; eager between replays (see capture notes).
                    last_gnorm = torch.nn.utils.clip_grad_norm_(
                        model.parameters(), args.grad_clip)
                    opt.step()
                    with torch.no_grad():          # carry membrane to next chunk
                        for sm, nm in zip(static_mems, static_new_mems):
                            sm.copy_(nm)
                    loss_t = static_loss.detach()
                else:
                    mems = model.detach_state(mems)                    # cut the graph
                    xc, yc = x[:, c0:c0 + chunk], y[:, c0:c0 + chunk]
                    logits, mems = model.forward_seq(xc, mems)         # (B, chunk, V)
                    loss = F.cross_entropy(
                        logits.reshape(-1, cfg.vocab_size), yc.reshape(-1))
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    # DESIGN: gradient clipping -- standard insurance for a recurrent
                    # net. The leaky membrane is contractive so gradients stay
                    # well-behaved (norms of order 1-10), but clipping the global norm
                    # cheaply caps the occasional larger step and keeps training smooth.
                    last_gnorm = torch.nn.utils.clip_grad_norm_(
                        model.parameters(), args.grad_clip)
                    opt.step()
                    loss_t = loss.detach()
                update += 1

                running += loss_t                  # GPU-side; no per-step sync
                n_since_log += 1
                if update % args.log_every == 0:
                    avg = (running / n_since_log).item()
                    running.zero_()
                    n_since_log = 0
                    bpc = avg / math.log(2)          # bits-per-character
                    speed = args.log_every / (time.time() - t0)
                    t0 = time.time()
                    mem = (torch.cuda.max_memory_allocated() / 1e9
                           if device.type == "cuda" else 0.0)
                    gn = float(last_gnorm) if last_gnorm is not None else None
                    print(f"upd {update:>6} | loss {avg:6.3f} | bpc {bpc:5.3f} "
                          f"| ppl {math.exp(avg):8.2f} | {speed:4.1f} upd/s "
                          f"| peakGPU {mem:4.2f}GB")
                    _csv_row(csv_writer, csv_file, update, "train", bpc, speed,
                             mem, lr=lr_at(update), grad_norm=gn)

                # Held-out evaluation. We report validation bpc (generalization)
                # and keep the checkpoint with the best val score -- the last step
                # is not necessarily best once the model starts to overfit.
                if do_val and args.eval_every and update % args.eval_every == 0:
                    val_bpc = evaluate(model, dataset, "val", args.batch_size,
                                       args.eval_batches, device,
                                       deterministic=args.deterministic_eval,
                                       max_windows=getattr(args, "eval_max_windows", 0))
                    fire = (model.spike_stats(probe_x)
                            if probe_x is not None else None)
                    tag = ""
                    if val_bpc < best_val:
                        best_val = val_bpc
                        _save_checkpoint(args.ckpt, model, cfg, dataset)
                        tag = "  <- best (saved)"
                    fr = ("" if not fire else " | fire " +
                          "/".join(f"{r:.2f}" for r in fire))
                    print(f"       val bpc {val_bpc:.3f} | ppl {2**val_bpc:6.2f}"
                          f"{fr}{tag}")
                    _csv_row(csv_writer, csv_file, update, "val", val_bpc, None,
                             None, fire_rates=fire)
                    if fire and (min(fire) < 0.02 or max(fire) > 0.90):
                        print("       WARNING: a layer's firing rate left "
                              "[2%, 90%] -- spiking dynamics may be "
                              "degenerating (dead or saturated layer)",
                              file=sys.stderr)
                    if chat_pairs:
                        cm = conditioning_metrics(model, dataset, chat_pairs,
                                                  device, max_pairs=96)
                        print(f"       cond bpc true {cm['bpc_true']:.3f} | "
                              f"shuffled {cm['bpc_shuffled']:.3f} | "
                              f"gain {cm['gain']:+.3f}")
                        _csv_row(csv_writer, csv_file, update, "cond_gain",
                                 cm["gain"], None, None)
                    if args.save_state:
                        _save_state(args.save_state, model, cfg, dataset, opt,
                                    update, best_val, args.steps, args.warmup)
                        last_state_t = time.time()
                    model.train()

                # Wall-clock save-state cadence, decoupled from the eval
                # schedule: a multi-day run must never be more than
                # ~--state-every-min minutes of progress away from a resumable
                # file, no matter how expensive evals are.
                if (args.save_state
                        and time.time() - last_state_t > state_every_s):
                    _save_state(args.save_state, model, cfg, dataset, opt,
                                update, best_val, args.steps, args.warmup)
                    last_state_t = time.time()

                if args.sample_every and update % args.sample_every == 0:
                    preview = model.generate(dataset, prompt=args.prompt or "the ",
                                             length=200, device=device,
                                             temperature=0.8, top_k=args.top_k)
                    print("-" * 60)
                    print(preview)
                    print("-" * 60)
                    model.train()
    except KeyboardInterrupt:
        interrupted = True
        print("\n[interrupt] Ctrl-C caught; saving current progress before exit...",
              file=sys.stderr)
    finally:
        if csv_file is not None:
            csv_file.close()

    # Finalization -- runs whether we finished or were interrupted. Always leave a
    # checkpoint on disk (with validation, keep it only if it beats the running
    # best; this also covers a run whose schedule never triggered a mid-run eval).
    if do_val:
        # Same estimator as the in-run evals (incl. --eval-max-windows): the
        # final number must be comparable with the best-val it is checked
        # against, and a FULL deterministic sweep of a large val split runs at
        # eager speed (~tens of minutes at h2048) -- get the official
        # full-sweep figure post-hoc via `eval --deterministic` instead.
        final_val = evaluate(model, dataset, "val", args.batch_size,
                             args.eval_batches, device,
                             deterministic=args.deterministic_eval,
                             max_windows=getattr(args, "eval_max_windows", 0))
        if final_val < best_val:
            best_val = final_val
            _save_checkpoint(args.ckpt, model, cfg, dataset)
        elif not os.path.exists(args.ckpt):
            # A resumed run can inherit a best_val that no in-run eval beats, so
            # nothing was saved this run. Never leave --ckpt missing (or claim in
            # the line below a file we didn't write) -- but don't clobber a better
            # existing checkpoint at that path.
            _save_checkpoint(args.ckpt, model, cfg, dataset)
        print(f"[done] best val bpc {best_val:.3f} | checkpoint -> {args.ckpt}")
    else:
        _save_checkpoint(args.ckpt, model, cfg, dataset)
        print(f"[done] saved checkpoint -> {args.ckpt}")
    if args.save_state:
        _save_state(args.save_state, model, cfg, dataset, opt, update, best_val,
                    args.steps, args.warmup)
        print(f"[state] resumable state ({update} updates) -> {args.save_state}")
    if interrupted:
        sys.exit(130)


# ---------------------------------------------------------------------------
# 4. Checkpoint I/O
# ---------------------------------------------------------------------------
def _save_checkpoint(path: str, model: SNNCharLM, cfg: SNNConfig,
                     dataset: CharDataset, extra: Optional[dict] = None) -> None:
    payload = {
        "format_version": 1,          # so future schema changes can be detected
        "state_dict": model.state_dict(),
        "cfg": asdict(cfg),
        "stoi": dataset.stoi,
        "itos": dataset.itos,
    }
    if extra:                          # optional resumable training state
        payload.update(extra)
    # DESIGN: atomic save with rotation. torch.save straight onto the target
    # would leave a truncated, unloadable file if the process dies mid-write --
    # fatal when that file is the only resume state of a multi-day run. Write
    # to a temp file, keep the previous good file as .bak, then rename into
    # place (os.replace is atomic on the same volume). A crash at any point
    # leaves at least one loadable file: path, path.bak, or path.tmp.
    tmp = path + ".tmp"
    torch.save(payload, tmp)
    if os.path.exists(path):
        os.replace(path, path + ".bak")
    os.replace(tmp, path)


def _require_matching_vocab(ckpt: dict, dataset: CharDataset, flag: str) -> None:
    """Raise unless a loaded checkpoint's vocabulary matches the current corpus.

    The weights are indexed by character id, so both the vocab SIZE and the exact
    character->id mapping must agree -- otherwise fc_in[0] and the readout would
    be silently indexed by mismatched ids. Used by both --init-from and --resume.
    """
    ckpt_vocab = ckpt["cfg"]["vocab_size"]
    if ckpt_vocab != dataset.vocab_size:
        raise ValueError(
            f"{flag} checkpoint has vocab_size={ckpt_vocab} but the corpus gives "
            f"{dataset.vocab_size}; use the same --data file.")
    ckpt_stoi = ckpt.get("stoi")
    if ckpt_stoi is not None and ckpt_stoi != dataset.stoi:
        n_diff = sum(1 for c in set(ckpt_stoi) | set(dataset.stoi)
                     if ckpt_stoi.get(c) != dataset.stoi.get(c))
        raise ValueError(
            f"{flag} checkpoint's character->id mapping differs from the current "
            f"corpus in {n_diff} symbol(s) (same vocab size, different characters "
            f"or order); the weights would be indexed by mismatched ids. Use the "
            f"same --data file.")


def _save_state(path: str, model: SNNCharLM, cfg: SNNConfig, dataset: CharDataset,
                opt: torch.optim.Optimizer, update: int, best_val: float,
                steps: int, warmup: int) -> None:
    """Write a FULL resumable training state: model + optimizer + progress + RNG.

    Larger than a plain checkpoint (Adam's moments roughly double the size), so it
    is written only when --save-state is given -- kept separate from the lean
    best-val --ckpt deliverable that users sample from. The LR-schedule horizon
    (steps, warmup) is recorded so --resume can warn if it is changed.
    """
    extra = {
        "opt_state": opt.state_dict(),
        "update": update,
        "best_val": best_val,
        "steps": steps,
        "warmup": warmup,
        "rng_state": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        extra["cuda_rng_state"] = torch.cuda.get_rng_state_all()
    _save_checkpoint(path, model, cfg, dataset, extra=extra)


def _restore_rng(ckpt: dict) -> None:
    """Restore CPU (and CUDA) RNG state saved by _save_state, so a resumed run
    continues the same random stream. RNG state must live on the CPU as a
    uint8 ByteTensor even though map_location may have moved it to the GPU.
    """
    rng = ckpt.get("rng_state")
    if rng is not None:
        torch.set_rng_state(rng.cpu() if torch.is_tensor(rng) else rng)
    cuda_rng = ckpt.get("cuda_rng_state")
    if cuda_rng is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([s.cpu() for s in cuda_rng])


def _open_metrics_csv(path: Optional[str]):
    """Open a metrics CSV and write its header; return (writer, file), or
    (None, None) when no --log-csv path was given. Rows are appended by _csv_row.
    """
    if not path:
        return None, None
    # Append mode so a --resume run extends the same curve; write the header only
    # when the file is new/empty (in append mode tell() reports the file size).
    f = open(path, "a", newline="", encoding="utf-8")
    writer = csv.writer(f)
    if f.tell() == 0:
        writer.writerow(["update", "split", "bpc", "ppl", "upd_per_s",
                         "peak_gpu_gb", "ts", "lr", "grad_norm", "fire_rates"])
    return writer, f


def _csv_row(writer, file, update: int, split: str, bpc: float,
             speed: Optional[float], mem: Optional[float],
             lr: Optional[float] = None, grad_norm: Optional[float] = None,
             fire_rates: Optional[List[float]] = None) -> None:
    """Append one long-format metrics row (no-op if CSV logging is off).

    The Tier 7 columns: ``ts`` (unix seconds, so a supervisor can detect a
    stalled run), ``lr``, ``grad_norm``, and ``fire_rates`` (slash-joined
    per-layer mean firing rates -- the spiking-health signal).
    """
    if writer is None:
        return
    writer.writerow([update, split, f"{bpc:.5f}", f"{2 ** bpc:.4f}",
                     "" if speed is None else f"{speed:.3f}",
                     "" if mem is None else f"{mem:.4f}",
                     f"{time.time():.0f}",
                     "" if lr is None else f"{lr:.2e}",
                     "" if grad_norm is None else f"{grad_norm:.3f}",
                     "" if not fire_rates else
                     "/".join(f"{r:.3f}" for r in fire_rates)])
    file.flush()


def _load_checkpoint(path: str, device: torch.device
                     ) -> Tuple[SNNCharLM, CharDataset]:
    # weights_only=True guards against arbitrary-code execution when loading a
    # shared or downloaded checkpoint (the payload is only tensors + plain dicts).
    ckpt = torch.load(path, map_location=device, weights_only=True)
    cfg = SNNConfig(**ckpt["cfg"])
    model = build_model(cfg).to(device)
    model.load_state_dict(ckpt["state_dict"])
    # Rebuild a lightweight, inference-only dataset that carries just the vocab
    # maps (enough for encode/decode/generate). It has no .data/.splits, so it
    # cannot be used for training or evaluate() -- only sampling.
    ds = CharDataset.__new__(CharDataset)
    ds.stoi = ckpt["stoi"]
    ds.itos = {int(k): v for k, v in ckpt["itos"].items()}
    ds.vocab_size = cfg.vocab_size
    ds.seq_len = cfg.seq_len
    return model, ds


# ---------------------------------------------------------------------------
# 5. Sampling entry point
# ---------------------------------------------------------------------------
def _warn_dropped_prompt_chars(dataset: CharDataset, prompt: str) -> None:
    """Warn on stderr if prompt characters are not in the checkpoint's vocab.

    encode() silently drops unknown characters, so "HELLO" against a lowercase
    vocab would quietly condition on fewer chars than typed -- surface it.
    """
    if not prompt:
        return
    dropped = [c for c in prompt if c not in dataset.stoi]
    if not dropped:
        return
    shown = " ".join(repr(c) for c in sorted(set(dropped)))
    if len(dropped) == len(prompt):
        print(f"warning: none of the {len(prompt)} prompt char(s) are in this "
              f"checkpoint's vocab ({shown}); generating from an empty seed",
              file=sys.stderr)
    else:
        print(f"warning: {len(dropped)} of {len(prompt)} prompt char(s) not in "
              f"vocab, ignored: {shown}", file=sys.stderr)


def sample(args) -> None:
    # Sampling is stochastic (the rate encoder and the multinomial draw); seed
    # only when asked, so the default stays fresh-each-run but a run is exactly
    # reproducible on demand.
    if args.seed is not None:
        set_seed(args.seed)
    device = pick_device(getattr(args, "device", "auto"))
    model, ds = _load_checkpoint(args.ckpt, device)
    _warn_dropped_prompt_chars(ds, args.prompt)
    text = model.generate(ds, prompt=args.prompt, length=args.length,
                          device=device, temperature=args.temperature,
                          top_k=args.top_k,
                          top_p=getattr(args, "top_p", None))
    print(text)


def eval_cmd(args) -> None:
    """Score a saved checkpoint's bits-per-character on a corpus.

    Reproduces / re-measures the headline metric without editing source. The
    corpus is mapped through the CHECKPOINT's vocabulary (the mapping the model
    trained with), and any characters outside that vocab are dropped with a
    warning. A fixed --seed makes the (random-window, stochastic-encoder) estimate
    repeatable; raise --eval-batches for a tighter number.
    """
    if args.seed is not None:
        set_seed(args.seed)
    device = pick_device(getattr(args, "device", "auto"))
    model, meta = _load_checkpoint(args.ckpt, device)
    text = _load_text(args.data)
    oov = sorted({c for c in text if c not in meta.stoi})
    if oov:
        n_oov = sum(1 for c in text if c not in meta.stoi)
        shown = " ".join(repr(c) for c in oov[:20]) + (" ..." if len(oov) > 20 else "")
        print(f"warning: {n_oov:,} char(s) in the corpus ({len(oov)} distinct) are "
              f"not in the checkpoint vocab and were dropped: {shown}",
              file=sys.stderr)
    seq_len = args.seq_len if args.seq_len is not None else meta.seq_len
    # --val-split only matters for scoring train|val; ignore it for 'all' so a
    # large val_split can't trip the (irrelevant) train-length guard in _set_splits.
    val_split = args.val_split if args.split in ("train", "val") else 0.0
    ds = CharDataset.with_vocab(text, seq_len=seq_len, stoi=meta.stoi,
                                itos=meta.itos, val_split=val_split)
    if args.split == "val" and not ds.has_val():
        raise ValueError("--split val but the corpus and --val-split yield no "
                         "validation window; use --split all or increase "
                         "--val-split / the corpus size.")
    deterministic = getattr(args, "deterministic", False)
    bpc = evaluate(model, ds, args.split, args.batch_size, args.eval_batches,
                   device, deterministic=deterministic,
                   max_windows=getattr(args, "max_windows", 0))
    where = args.data if args.data else "built-in demo corpus"
    mode = ("deterministic full-split sweep" if deterministic
            else f"{args.eval_batches} x {args.batch_size} random windows")
    print(f"[eval] {args.ckpt} on {where} | split={args.split} seq_len={seq_len} "
          f"| {mode} | bpc {bpc:.4f} | ppl {2 ** bpc:.2f}")


# ---------------------------------------------------------------------------
# 5b. Chat: interactive REPL + scripted one-shot + chat-quality evaluation
# ---------------------------------------------------------------------------
def _checkpoint_can_chat(ds: CharDataset) -> bool:
    return all(c in ds.stoi for c in (ROLE_USER, ROLE_ASSISTANT, ROLE_END))


def chat(args) -> None:
    """Interactive chat with a checkpoint trained on the role-marker format.

    The recurrent membrane state persists ACROSS turns -- the conversation
    lives in the network's spiking dynamics, exactly as during training on
    multi-turn dialogues -- so follow-up turns are conditioned on the whole
    exchange so far. `/reset` zeroes it; `/quit` exits. ``--once`` answers a
    single prompt non-interactively (scriptable) and prints only the reply.
    """
    if args.seed is not None:
        set_seed(args.seed)
    device = pick_device(getattr(args, "device", "auto"))
    model, ds = _load_checkpoint(args.ckpt, device)
    if not _checkpoint_can_chat(ds):
        sys.exit("error: this checkpoint's vocabulary has no chat role "
                 "markers; train one with --vocab fixed on a chat-formatted "
                 "corpus (see build_chat_corpus.py).")
    state = model.init_state(1, device)
    pending_end = False                    # \x03 owed if a reply hit the cap
    gen_kwargs = dict(temperature=args.temperature, top_k=args.top_k,
                      top_p=args.top_p, stop_char=ROLE_END,
                      ban_chars=ROLE_USER + ROLE_ASSISTANT)

    def respond(user_text: str, stream) -> str:
        nonlocal state, pending_end
        prompt = ((ROLE_END if pending_end else "")
                  + ROLE_USER + user_text.strip() + ROLE_ASSISTANT)
        text, state = model._generate_core(
            ds, state, prompt, args.max_chars, device, stream=stream,
            **gen_kwargs)
        pending_end = not text.endswith(ROLE_END)
        return text[:-1] if text.endswith(ROLE_END) else text

    if getattr(args, "once", None) is not None:
        print(respond(args.once, None).strip())
        return

    print(f"[chat] {args.ckpt} | temperature {args.temperature} "
          f"top-p {args.top_p} | /reset clears memory, /quit exits")
    while True:
        try:
            user = input("you> ")
        except (EOFError, KeyboardInterrupt):
            print()
            break
        user = user.strip()
        if not user:
            continue
        if user in ("/quit", "/exit"):
            break
        if user == "/reset":
            state = model.init_state(1, device)
            pending_end = False
            print("[chat] memory cleared")
            continue
        sys.stdout.write("snn> ")
        sys.stdout.flush()

        def _stream(ch: str) -> None:
            if ch not in (ROLE_USER, ROLE_ASSISTANT, ROLE_END):
                sys.stdout.write(ch)
                sys.stdout.flush()

        respond(user, _stream)
        sys.stdout.write("\n")


def _word_set(corpus_path: Optional[str]) -> Optional[set]:
    """Vocabulary of words (>=2 occurrences) from a corpus file, for the
    word-validity metric. None if no corpus was given."""
    if not corpus_path:
        return None
    with open(corpus_path, "r", encoding="utf-8") as f:
        text = f.read(20_000_000).lower()
    counts = collections.Counter(re.findall(r"[a-z']+", text))
    return {w for w, c in counts.items() if c >= 2}


def chateval(args) -> None:
    """Chat-quality report card for a checkpoint.

    Metrics (all defined in CAMPAIGN.md):
      conditioning gain  -- assistant-span bpc with the true vs a shuffled user
                            turn (positive = replies actually depend on the
                            prompt; ~0 = generic-register failure).
      termination rate   -- fraction of sampled replies that emit the
                            end-of-turn marker within --max-chars.
      distinct-3gram     -- unique/total character 3-grams over the replies
                            (low = the model loops).
      word validity      -- fraction of generated words that occur in --data's
                            corpus (needs --data; a proxy for spelling).
    Sampled transcripts are printed (and written to --transcripts if given).
    """
    if args.seed is not None:
        set_seed(args.seed)
    device = pick_device(getattr(args, "device", "auto"))
    model, ds = _load_checkpoint(args.ckpt, device)
    if not _checkpoint_can_chat(ds):
        sys.exit("error: checkpoint has no chat role markers in its vocab.")
    pairs = _load_chat_pairs(args.pairs)
    cm = conditioning_metrics(model, ds, pairs, device, max_pairs=args.n)
    print(f"[chateval] conditioning over {cm['n_pairs']} pairs: "
          f"bpc true {cm['bpc_true']:.3f} | shuffled {cm['bpc_shuffled']:.3f} "
          f"| gain {cm['gain']:+.3f}")

    words = _word_set(getattr(args, "data", None))
    n_gen = min(args.gen_n, len(pairs))
    ended = 0
    lengths = []
    all_text = []
    transcripts = []
    for p in pairs[:n_gen]:
        state = model.init_state(1, device)
        prompt = ROLE_USER + p["user"] + ROLE_ASSISTANT
        text, _ = model._generate_core(
            ds, state, prompt, args.max_chars, device,
            temperature=args.temperature, top_p=args.top_p,
            stop_char=ROLE_END, ban_chars=ROLE_USER + ROLE_ASSISTANT)
        if text.endswith(ROLE_END):
            ended += 1
            text = text[:-1]
        lengths.append(len(text))
        all_text.append(text)
        transcripts.append(f"you> {p['user']}\nsnn> {text}\n")
    joined = " ".join(all_text)
    grams = [joined[i:i + 3] for i in range(len(joined) - 2)]
    distinct3 = len(set(grams)) / max(1, len(grams))
    toks = re.findall(r"[a-z']+", joined.lower())
    validity = (sum(1 for t in toks if t in words) / max(1, len(toks))
                if words is not None else float("nan"))
    print(f"[chateval] {n_gen} sampled replies: terminated {ended}/{n_gen} "
          f"| mean len {sum(lengths)/max(1,len(lengths)):.0f} chars "
          f"| distinct-3gram {distinct3:.3f} "
          f"| word-validity {validity:.3f}")
    print("-" * 60)
    for t in transcripts[:args.show]:
        print(t)
    if getattr(args, "transcripts", None):
        with open(args.transcripts, "w", encoding="utf-8") as f:
            f.write("\n".join(transcripts))
        print(f"[chateval] transcripts -> {args.transcripts}")


# ---------------------------------------------------------------------------
# 5c. CUDA-graph training parity check (the go/no-go gate for --cuda-graph)
# ---------------------------------------------------------------------------
def graphcheck(args) -> None:
    """Train the SAME model twice from identical init -- once eager, once with
    the captured-graph training step -- and compare final weights and losses.

    A subtly wrong capture (stale static buffers, freed grad addresses, baked
    shapes) still produces a decreasing loss curve, so "loss goes down" proves
    nothing; weight-level agreement with eager after N updates is the actual
    correctness gate. Exits non-zero on mismatch.
    """
    device = pick_device(getattr(args, "device", "auto"))
    if device.type != "cuda":
        sys.exit("error: graphcheck needs a CUDA device.")
    B, chunk = args.batch_size, args.chunk
    ds = CharDataset(_DEMO_CORPUS, seq_len=args.seq_len,
                     fixed_vocab=FIXED_VOCAB)
    cfg = SNNConfig(vocab_size=ds.vocab_size, hidden=args.hidden,
                    num_layers=args.layers, num_steps=args.num_steps,
                    seq_len=args.seq_len, dropout=0.0, layernorm=True,
                    input_coding="graded")
    set_seed(999)
    windows = [ds.get_batch(B, device) for _ in range(args.windows)]

    def run(graphed: bool):
        set_seed(123)                      # identical init both runs
        model = SNNCharLM(cfg).to(device).train()
        opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
        gs = (_capture_chunk_step(model, cfg, B, chunk, device, warm_lr=3e-3)
              if graphed else None)
        losses = []
        for x, y in windows:
            if graphed:
                for mm in gs["mems"]:
                    mm.zero_()
            else:
                mems = model.init_state(B, device)
            for c0 in range(0, x.size(1), chunk):
                if graphed:
                    gs["x"].copy_(x[:, c0:c0 + chunk])
                    gs["y"].copy_(y[:, c0:c0 + chunk])
                    gs["graph"].replay()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step()
                    with torch.no_grad():
                        for sm, nm in zip(gs["mems"], gs["new_mems"]):
                            sm.copy_(nm)
                    losses.append(float(gs["loss"]))
                else:
                    mems = model.detach_state(mems)
                    logits, mems = model.forward_seq(x[:, c0:c0 + chunk], mems)
                    loss = F.cross_entropy(
                        logits.reshape(-1, cfg.vocab_size),
                        y[:, c0:c0 + chunk].reshape(-1))
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step()
                    losses.append(float(loss.detach()))
        return model, losses

    eager_model, eager_losses = run(False)
    graph_model, graph_losses = run(True)
    w_diff = max(float((a - b).abs().max())
                 for (_, a), (_, b) in zip(eager_model.state_dict().items(),
                                           graph_model.state_dict().items()))
    l_diff = max(abs(a - b) for a, b in zip(eager_losses, graph_losses))
    n_upd = len(eager_losses)
    print(f"[graphcheck] h{args.hidden} L{args.layers} T={args.num_steps} "
          f"B{B} chunk {chunk} | {n_upd} updates")
    print(f"[graphcheck] loss  eager {eager_losses[0]:.4f}->"
          f"{eager_losses[-1]:.4f} | graph {graph_losses[0]:.4f}->"
          f"{graph_losses[-1]:.4f} | max |diff| {l_diff:.3e}")
    print(f"[graphcheck] max |weight diff| after {n_upd} updates: {w_diff:.3e}")
    ok = w_diff < args.tol and l_diff < args.tol
    print(f"[graphcheck] {'PASS' if ok else 'FAIL'} (tolerance {args.tol:g})")
    if not ok:
        sys.exit(1)


# ---------------------------------------------------------------------------
# 6. Helpers + smoke test
# ---------------------------------------------------------------------------
def _load_text(path: Optional[str]) -> str:
    if path is None:
        return _DEMO_CORPUS
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def corpus_bpc_floors(text: str) -> Tuple[float, float]:
    """(unigram, bigram) bits-per-character entropy floors of ``text``.

    The cross-entropy a model that knows only single-character frequencies
    (unigram) or first-order previous-character statistics (bigram) would reach.
    These are the honest baselines a real language model must beat; the README
    quotes them, and computing them here (rather than hard-coding round numbers)
    keeps the quoted figures pinned to the shipped corpus -- the test suite
    checks the two agree. Reproduce with::

        python -c "import snn_char_lm as m; \\
            print(m.corpus_bpc_floors(open('input.txt', encoding='utf-8').read()))"
    """
    n = len(text)
    if n < 2:
        raise ValueError("need at least 2 characters to estimate entropy floors")
    unigram_counts = collections.Counter(text)
    unigram = -sum((c / n) * math.log2(c / n) for c in unigram_counts.values())
    prev_counts = collections.Counter(text[:-1])
    pair_counts = collections.Counter(zip(text, text[1:]))
    bigram = -sum((c / (n - 1)) * math.log2(c / prev_counts[a])
                  for (a, _b), c in pair_counts.items())
    return unigram, bigram


def smoke(args) -> None:
    """Short end-to-end self-test: prove the chunked-TBPTT path reduces loss and
    that sampling runs, then exit non-zero if the model did not learn.

    This is a self-contained *reimplementation* of the same chunked truncated-BPTT
    update mechanism that train() uses -- it deliberately does NOT call train(),
    and it simplifies for speed and determinism: dropout off, a fixed learning
    rate (no warmup/cosine schedule), and AdamW's default weight decay. Its job is
    a fast smoke test, not a faithful copy of the full training configuration; the
    pytest suite carries the fine-grained correctness assertions.

    The dimensions are parametrizable via the ``smoke`` subcommand flags so CI can
    run a shrunk, fast variant; defaults reproduce the historical self-test.
    """
    set_seed(0)
    device = pick_device(getattr(args, "device", "auto"))
    updates = getattr(args, "updates", 400)
    hidden = getattr(args, "hidden", 256)
    batch = getattr(args, "batch_size", 64)
    seq_len = getattr(args, "seq_len", 64)
    num_steps = getattr(args, "num_steps", 5)
    chunk = getattr(args, "chunk", 16)
    # The demo corpus is highly predictable, so a working model should land well
    # below the ~1.3 bpc bigram floor.
    pass_bpc = 1.5

    ds = CharDataset(_DEMO_CORPUS, seq_len=seq_len)
    cfg = SNNConfig(vocab_size=ds.vocab_size, hidden=hidden, num_layers=2,
                    num_steps=num_steps, seq_len=seq_len, dropout=0.0)
    model = SNNCharLM(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)

    first, last = None, None
    update = 0
    log_every = max(1, updates // 5)
    while update < updates:
        x, y = ds.get_batch(batch, device)
        mems = model.init_state(batch, device)
        for c0 in range(0, x.size(1), chunk):
            if update >= updates:
                break
            mems = model.detach_state(mems)
            logits, mems = model.forward_seq(x[:, c0:c0 + chunk], mems)
            loss = F.cross_entropy(logits.reshape(-1, cfg.vocab_size),
                                   y[:, c0:c0 + chunk].reshape(-1))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            update += 1
            v = loss.item()
            if first is None:
                first = v
            last = v
            if update % log_every == 0:
                print(f"  smoke update {update:3d} | loss {v:.3f} "
                      f"| bpc {v/math.log(2):.3f}")

    finite = all(torch.isfinite(p).all().item() for p in model.parameters())
    print(f"[smoke] loss {first:.3f} -> {last:.3f} "
          f"(random baseline ~{math.log(ds.vocab_size):.3f}) | "
          f"final bpc {last/math.log(2):.3f} | weights_finite={finite}")
    out = model.generate(ds, prompt="the ", length=120, device=device,
                         temperature=0.5)
    print("[smoke] sample:", repr(out))
    # Gate on three things: weights stayed finite, loss actually went DOWN
    # (last < first -- a real regression would trip this even if it still lands
    # under the bpc bar by luck), and the final bpc cleared the learning bar.
    learned = finite and (last < first) and (last / math.log(2)) < pass_bpc
    print("[smoke] PASS" if learned else "[smoke] WARN: did not learn as expected")
    if not learned:
        sys.exit(1)


# ---------------------------------------------------------------------------
# 6b. CUDA-graph benchmark (Tier 6 performance lever)
# ---------------------------------------------------------------------------
def bench(args) -> None:
    """Benchmark the recurrent inner loop: eager vs a CUDA-graph replay.

    The SNN is launch-bound -- a long chain of tiny sequential kernels -- so the
    real throughput lever is cutting per-kernel launch overhead, not faster math.
    We capture the RNG-free, fixed-shape core (SNNCharLM._run_core over
    pre-encoded currents) in a CUDA graph and replay it. Forward-only, no autograd
    -- this is the isolated inner-loop measurement referenced in TIER6.md.
    torch.compile is unavailable here (no Triton on Windows / Python 3.14), so a
    CUDA graph is the Triton-free alternative. The stochastic encoder (spikegen)
    is kept OUTSIDE the graph (RNG in a captured graph is not replay-safe).
    """
    device = pick_device(getattr(args, "device", "auto"))
    if device.type != "cuda":
        print("[bench] CUDA graphs need a GPU; got device=cpu -- nothing to do.")
        return
    import time as _time
    set_seed(0)
    cfg = SNNConfig(vocab_size=args.vocab, hidden=args.hidden,
                    num_layers=args.layers, num_steps=args.num_steps,
                    seq_len=args.seq_len, dropout=0.0)
    model = SNNCharLM(cfg).to(device).eval()
    B, L = args.batch_size, args.seq_len

    # Pre-encode ONCE, outside the timed/captured region (RNG stays out of the graph).
    with torch.no_grad():
        x = torch.randint(0, cfg.vocab_size, (B, L), device=device)
        onehot = F.one_hot(x, cfg.vocab_size).float()
        spikes = spikegen.rate(onehot, num_steps=cfg.num_steps, gain=cfg.rate_gain)
        cur0_all = model.fc_in[0](spikes).contiguous()          # (T, B, L, hidden)

    def eager_once():
        with torch.no_grad():
            model._run_core(cur0_all, model.init_state(B, device))

    def timed(fn, iters):
        for _ in range(args.warmup):
            fn()
        torch.cuda.synchronize()
        t0 = _time.time()
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
        return (_time.time() - t0) / iters * 1e3                 # ms per iteration

    eager_ms = timed(eager_once, args.iters)

    # Capture the core in a CUDA graph. graph_in_mems are the static input buffers
    # and MUST stay alive for replay; passing a shallow copy to _run_core lets it
    # reassign its local list without dropping our references.
    graph_in_mems = model.init_state(B, device)
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):                               # warm up before capture
        for _ in range(3):
            with torch.no_grad():
                model._run_core(cur0_all, [mm.clone() for mm in graph_in_mems])
    torch.cuda.current_stream().wait_stream(side)

    graph = torch.cuda.CUDAGraph()
    with torch.no_grad(), torch.cuda.graph(graph):
        static_out, _ = model._run_core(cur0_all, list(graph_in_mems))
    keep_alive = (cur0_all, graph_in_mems, static_out)          # noqa: keep refs live
    graph_ms = timed(graph.replay, args.iters)

    speedup = eager_ms / graph_ms if graph_ms > 0 else float("nan")
    print(f"[bench] {torch.cuda.get_device_name(0)} | arch=snn hidden={cfg.hidden} "
          f"layers={cfg.num_layers} T={cfg.num_steps} B={B} L={L}")
    print(f"[bench] forward inner-loop core (RNG-free, fixed shape), "
          f"{args.iters} iters:")
    print(f"[bench]   eager       {eager_ms:8.3f} ms/iter")
    print(f"[bench]   cuda-graph  {graph_ms:8.3f} ms/iter")
    print(f"[bench]   speedup     {speedup:8.2f}x")
    del keep_alive


# ---------------------------------------------------------------------------
# 7. CLI
# ---------------------------------------------------------------------------
def _positive_int(s: str) -> int:
    """argparse type for counts that must be >= 1 (avoids a divide-by-zero eval)."""
    v = int(s)
    if v < 1:
        raise argparse.ArgumentTypeError("must be >= 1")
    return v


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    # -- train --
    t = sub.add_parser("train", help="train the SNN language model")
    t.add_argument("--data", type=str, default=None,
                   help="path to a UTF-8 text file (default: built-in demo)")
    t.add_argument("--arch", type=str, default="snn", choices=["snn", "gru"],
                   help="snn (spiking LIF; default) | gru (non-spiking baseline)")
    t.add_argument("--ckpt", type=str, default="snn_char_lm.pt")
    warm = t.add_mutually_exclusive_group()
    warm.add_argument("--init-from", dest="init_from", type=str, default=None,
                      help="warm-start WEIGHTS from this checkpoint (architecture "
                           "is taken from it; fresh optimizer + LR schedule)")
    warm.add_argument("--resume", type=str, default=None,
                      help="resume a run from a --save-state file: restores "
                           "weights, optimizer, step, best-val and RNG, then "
                           "continues toward --steps")
    t.add_argument("--steps", type=_positive_int, default=3000,
                   help="number of optimizer updates (one per TBPTT chunk)")
    t.add_argument("--val-split", dest="val_split", type=float, default=0.1,
                   help="fraction of the corpus (its tail) held out for "
                        "validation; 0 disables held-out eval")
    t.add_argument("--eval-every", dest="eval_every", type=int, default=500,
                   help="run held-out evaluation every N updates")
    t.add_argument("--eval-batches", dest="eval_batches", type=_positive_int,
                   default=20,
                   help="batches averaged per held-out evaluation (>= 1)")
    # DESIGN: the training loop is launch-bound (many tiny sequential spiking
    # kernels), so a big batch costs almost no extra wall-clock but sees far more
    # data per update. 128 peaks well under 1 GB; the 8 GB card can push 256.
    t.add_argument("--batch-size", dest="batch_size", type=_positive_int, default=128)
    t.add_argument("--seq-len", dest="seq_len", type=_positive_int, default=128,
                   help="characters per training window (context length)")
    t.add_argument("--tbptt-chunk", dest="tbptt_chunk", type=_positive_int, default=32,
                   help="truncated-BPTT chunk length; backprop depth = chunk*T")
    t.add_argument("--hidden", type=_positive_int, default=512)
    t.add_argument("--layers", type=_positive_int, default=2)
    t.add_argument("--num-steps", dest="num_steps", type=_positive_int, default=5,
                   help="SNN micro-steps per character (rate-code length T)")
    t.add_argument("--beta", type=float, default=0.9)
    t.add_argument("--rate-gain", dest="rate_gain", type=float, default=0.9)
    t.add_argument("--surrogate", type=str, default="atan",
                   choices=["atan", "fast_sigmoid", "sigmoid"])
    t.add_argument("--learn-beta", dest="learn_beta", action="store_true",
                   help="make the membrane time-constant beta trainable")
    t.add_argument("--threshold", type=float, default=1.0,
                   help="LIF firing threshold")
    t.add_argument("--learn-threshold", dest="learn_threshold",
                   action="store_true",
                   help="make the firing threshold trainable")
    # -- Tier 6 ablation knobs (all off by default) --
    t.add_argument("--beta-per-neuron", dest="beta_per_neuron",
                   action="store_true",
                   help="learn a per-neuron membrane decay (a [hidden] beta "
                        "vector) instead of one shared scalar")
    t.add_argument("--layernorm", action="store_true",
                   help="LayerNorm the inter-layer input currents (ablation to "
                        "help train deeper stacks)")
    t.add_argument("--input-coding", dest="input_coding", type=str,
                   default="rate", choices=["rate", "graded"],
                   help="rate = stochastic spike encoder (default); graded = "
                        "deterministic constant-current injection")
    t.add_argument("--deterministic-eval", dest="deterministic_eval",
                   action="store_true",
                   help="use a fixed-window, seeded-encoder eval for "
                        "best-checkpoint selection (stable, reproducible)")
    t.add_argument("--dropout", type=float, default=0.1)
    t.add_argument("--lr", type=float, default=3e-3)
    t.add_argument("--weight-decay", dest="weight_decay", type=float, default=1e-4)
    t.add_argument("--warmup", type=int, default=100)
    t.add_argument("--grad-clip", dest="grad_clip", type=float, default=1.0)
    t.add_argument("--log-every", dest="log_every", type=_positive_int, default=100)
    t.add_argument("--sample-every", dest="sample_every", type=int, default=750)
    t.add_argument("--prompt", type=str, default="the ")
    t.add_argument("--top-k", dest="top_k", type=int, default=None)
    t.add_argument("--seed", type=int, default=1337)
    t.add_argument("--device", type=str, default="auto",
                   help="auto | cpu | cuda | cuda:N")
    t.add_argument("--save-state", dest="save_state", type=str, default=None,
                   help="also write a full resumable training state "
                        "(model+optimizer+RNG) here, refreshed each eval, every "
                        "--state-every-min minutes, and at exit; load with "
                        "--resume (the previous state is kept as <file>.bak)")
    t.add_argument("--state-every-min", dest="state_every_min", type=float,
                   default=15.0,
                   help="wall-clock minutes between resumable-state saves "
                        "(decoupled from the eval cadence)")
    t.add_argument("--log-csv", dest="log_csv", type=str, default=None,
                   help="append metrics (update,split,bpc,ppl,upd/s,peakGPU,"
                        "ts,lr,gradnorm,fire) to this CSV")
    # -- Tier 7: chat campaign flags --
    t.add_argument("--vocab", type=str, default="corpus",
                   choices=["corpus", "fixed"],
                   help="corpus = vocabulary derived from --data (classic); "
                        "fixed = the corpus-independent FIXED_VOCAB incl. chat "
                        "role markers (lets one model pretrain and fine-tune "
                        "on different corpora)")
    t.add_argument("--val-data", dest="val_data", type=str, default=None,
                   help="separate validation corpus file (overrides "
                        "--val-split tail splitting; used by the multi-source "
                        "chat corpus whose tail would be single-register)")
    t.add_argument("--eval-max-windows", dest="eval_max_windows", type=int,
                   default=0,
                   help="cap deterministic eval at N evenly spaced windows "
                        "(0 = sweep the whole split); keeps in-run evals cheap "
                        "on a large corpus")
    t.add_argument("--cuda-graph", dest="cuda_graph", action="store_true",
                   help="capture the TBPTT chunk's fwd+loss+bwd as a CUDA "
                        "graph and replay it per update (needs --input-coding "
                        "graded, --dropout 0; verify with `graphcheck`)")
    t.add_argument("--tf32", action="store_true",
                   help="allow TF32 matmuls (worth testing once --cuda-graph "
                        "removes the launch-overhead bottleneck)")
    t.add_argument("--chat-pairs", dest="chat_pairs", type=str, default=None,
                   help="conditioning_val.jsonl from build_chat_corpus.py; if "
                        "given, conditioning gain is reported at every eval")
    t.set_defaults(func=train)

    # -- sample --
    s = sub.add_parser("sample", help="generate text from a checkpoint")
    s.add_argument("--ckpt", type=str, default="snn_char_lm.pt")
    s.add_argument("--prompt", type=str, default="the ")
    s.add_argument("--length", type=_positive_int, default=400)
    s.add_argument("--temperature", type=float, default=0.8)
    s.add_argument("--top-k", dest="top_k", type=int, default=None)
    s.add_argument("--top-p", dest="top_p", type=float, default=None,
                   help="nucleus sampling mass (e.g. 0.9)")
    s.add_argument("--seed", type=int, default=None,
                   help="seed RNGs for reproducible generation (default: fresh "
                        "output each run)")
    s.add_argument("--device", type=str, default="auto",
                   help="auto | cpu | cuda | cuda:N")
    s.set_defaults(func=sample)

    # -- eval --
    e = sub.add_parser("eval",
                       help="score a checkpoint's bits-per-character on a corpus")
    e.add_argument("--ckpt", type=str, default="snn_char_lm.pt")
    e.add_argument("--data", type=str, default=None,
                   help="UTF-8 text file to score on (default: built-in demo)")
    e.add_argument("--split", type=str, default="all",
                   choices=["all", "train", "val"],
                   help="which split to score; train|val need --val-split > 0")
    e.add_argument("--val-split", dest="val_split", type=float, default=0.0,
                   help="held-out tail fraction (only for --split train|val)")
    e.add_argument("--seq-len", dest="seq_len", type=_positive_int, default=None,
                   help="window length (default: the checkpoint's training seq_len)")
    e.add_argument("--batch-size", dest="batch_size", type=_positive_int,
                   default=128)
    e.add_argument("--eval-batches", dest="eval_batches", type=_positive_int,
                   default=50,
                   help="random windows averaged; more = tighter estimate")
    e.add_argument("--seed", type=int, default=1337,
                   help="seed so the (random-window, stochastic-encoder) number "
                        "repeats; pass a different value to resample")
    e.add_argument("--deterministic", action="store_true",
                   help="fixed-window + seeded-encoder full-split sweep: an "
                        "exactly reproducible bpc (ignores --eval-batches)")
    e.add_argument("--max-windows", dest="max_windows", type=int, default=0,
                   help="with --deterministic: cap at N evenly spaced windows "
                        "(0 = full sweep); a full sweep of a large corpus runs "
                        "at eager speed and can take tens of minutes")
    e.add_argument("--device", type=str, default="auto",
                   help="auto | cpu | cuda | cuda:N")
    e.set_defaults(func=eval_cmd)

    # -- chat --
    c = sub.add_parser("chat",
                       help="chat with a checkpoint (REPL, or --once for a "
                            "single scripted prompt)")
    c.add_argument("--ckpt", type=str, default="spark.pt")
    c.add_argument("--once", type=str, default=None,
                   help="answer this one prompt and exit (prints only the reply)")
    c.add_argument("--temperature", type=float, default=0.9)
    c.add_argument("--top-p", dest="top_p", type=float, default=0.9,
                   help="nucleus sampling mass (tune this OR --top-k, not both)")
    c.add_argument("--top-k", dest="top_k", type=int, default=None)
    c.add_argument("--max-chars", dest="max_chars", type=_positive_int,
                   default=300,
                   help="reply length cap (short replies stay on-register; "
                        "the model's long replies wander)")
    c.add_argument("--seed", type=int, default=None)
    c.add_argument("--device", type=str, default="auto")
    c.set_defaults(func=chat)

    # -- chateval --
    ce = sub.add_parser("chateval",
                        help="chat-quality report: conditioning gain, "
                             "termination, repetition, word validity")
    ce.add_argument("--ckpt", type=str, default="spark.pt")
    ce.add_argument("--pairs", type=str, default="corpus/conditioning_val.jsonl")
    ce.add_argument("--n", type=_positive_int, default=200,
                    help="pairs scored for conditioning gain")
    ce.add_argument("--gen-n", dest="gen_n", type=_positive_int, default=12,
                    help="prompts sampled for generation metrics")
    ce.add_argument("--show", type=int, default=6,
                    help="transcripts printed to stdout")
    ce.add_argument("--data", type=str, default=None,
                    help="corpus file whose words define the word-validity set")
    ce.add_argument("--transcripts", type=str, default=None,
                    help="write all sampled transcripts to this file")
    ce.add_argument("--temperature", type=float, default=0.9)
    ce.add_argument("--top-p", dest="top_p", type=float, default=0.9)
    ce.add_argument("--max-chars", dest="max_chars", type=_positive_int,
                    default=400)
    ce.add_argument("--seed", type=int, default=7)
    ce.add_argument("--device", type=str, default="auto")
    ce.set_defaults(func=chateval)

    # -- graphcheck --
    gc = sub.add_parser("graphcheck",
                        help="parity gate: eager vs CUDA-graph training must "
                             "produce the same weights")
    gc.add_argument("--hidden", type=_positive_int, default=256)
    gc.add_argument("--layers", type=_positive_int, default=3)
    gc.add_argument("--num-steps", dest="num_steps", type=_positive_int,
                    default=3)
    gc.add_argument("--seq-len", dest="seq_len", type=_positive_int, default=128)
    gc.add_argument("--chunk", type=_positive_int, default=32)
    gc.add_argument("--batch-size", dest="batch_size", type=_positive_int,
                    default=32)
    gc.add_argument("--windows", type=_positive_int, default=6,
                    help="training windows compared (updates = windows * "
                         "seq_len/chunk)")
    gc.add_argument("--tol", type=float, default=1e-3,
                    help="max allowed |weight diff| and |loss diff|")
    gc.add_argument("--device", type=str, default="auto")
    gc.set_defaults(func=graphcheck)

    # -- smoke --
    sm = sub.add_parser("smoke", help="fast self-test")
    sm.add_argument("--updates", type=_positive_int, default=400,
                    help="optimizer updates for the self-test")
    sm.add_argument("--hidden", type=_positive_int, default=256)
    sm.add_argument("--batch-size", dest="batch_size", type=_positive_int,
                    default=64)
    sm.add_argument("--seq-len", dest="seq_len", type=_positive_int, default=64)
    sm.add_argument("--num-steps", dest="num_steps", type=_positive_int,
                    default=5)
    sm.add_argument("--chunk", type=_positive_int, default=16)
    sm.add_argument("--device", type=str, default="auto",
                    help="auto | cpu | cuda | cuda:N")
    sm.set_defaults(func=smoke)

    # -- bench --
    b = sub.add_parser("bench",
                       help="benchmark the recurrent inner loop: eager vs CUDA graph")
    b.add_argument("--hidden", type=_positive_int, default=512)
    b.add_argument("--layers", type=_positive_int, default=2)
    b.add_argument("--num-steps", dest="num_steps", type=_positive_int, default=5)
    b.add_argument("--seq-len", dest="seq_len", type=_positive_int, default=64,
                   help="chunk length L benchmarked (the inner-loop length)")
    b.add_argument("--batch-size", dest="batch_size", type=_positive_int, default=128)
    b.add_argument("--vocab", type=_positive_int, default=77,
                   help="vocabulary size to benchmark (default: the shipped "
                        "corpus's 77 characters)")
    b.add_argument("--iters", type=_positive_int, default=50)
    b.add_argument("--warmup", type=_positive_int, default=10)
    b.add_argument("--device", type=str, default="auto",
                   help="auto | cpu | cuda | cuda:N")
    b.set_defaults(func=bench)

    return p


def main() -> None:
    # NB: we deliberately do NOT enable TF32 (set_float32_matmul_precision) or AMP.
    # This workload is launch-bound (a long chain of tiny sequential spiking
    # kernels), so faster matmuls buy ~nothing while perturbing float32 numerics;
    # the real throughput lever is a CUDA-graph capture of the inner loop (see
    # TIER6.md). Keeping default precision also keeps runs numerically stable.
    args = build_parser().parse_args()
    # Turn the common, expected misuse into a clean one-line message instead of a
    # raw traceback. train() handles its own Ctrl-C (to save); the KeyboardInterrupt
    # catch here covers sample/eval. SystemExit (e.g. train's exit(130)) passes
    # through untouched.
    try:
        args.func(args)
    except KeyboardInterrupt:
        sys.exit(130)
    except FileNotFoundError as e:
        sys.exit(f"error: file not found: {e.filename or e}")
    except IsADirectoryError as e:
        sys.exit(f"error: expected a file but got a directory: {e.filename or e}")
    except UnicodeDecodeError:
        sys.exit("error: --data file is not valid UTF-8; re-save it as UTF-8.")
    except ValueError as e:
        sys.exit(f"error: {e}")


if __name__ == "__main__":
    main()
