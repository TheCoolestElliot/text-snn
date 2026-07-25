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

Train on real text (recommended: Karpathy's tiny-shakespeare, ~1 MB) with a
held-out validation split (the best-validation checkpoint is kept). This writes
a scratch checkpoint so it will not overwrite the pre-trained ``shakespeare.pt``
shipped in the repo; see the README "Results" section for the full recipe that
reproduces the 2.44-bpc model::

    python snn_char_lm.py train --data input.txt --steps 8000 --seq-len 160 \
        --val-split 0.1 --seed 1337 --ckpt my_shakespeare.pt

Sample from a trained checkpoint::

    python snn_char_lm.py sample --ckpt shakespeare.pt --prompt "ROMEO:" --length 400

Run a fast self-test (a short training run + a generation call)::

    python snn_char_lm.py smoke
"""

from __future__ import annotations

import argparse
import collections
import math
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


def pick_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# 1. Data: a character-level vocabulary + window sampler
# ---------------------------------------------------------------------------
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


class CharDataset:
    """Turns a text string into integer ids and serves random training windows.

    Each sample is a contiguous window of `seq_len + 1` characters. The first
    `seq_len` characters are the input; the same window shifted right by one is
    the target (classic next-character prediction).
    """

    def __init__(self, text: str, seq_len: int, val_split: float = 0.0):
        self.seq_len = seq_len
        # We draw windows of `seq_len + 1` chars, so the corpus must contain at
        # least one such window. Fail loudly here rather than with a cryptic
        # torch.randint error inside get_batch.
        if len(text) < seq_len + 1:
            raise ValueError(
                f"corpus has {len(text):,} chars but --seq-len {seq_len} needs "
                f"at least {seq_len + 1}; use a longer --data file or a smaller "
                f"--seq-len."
            )
        # Sorted for determinism so a saved vocab maps ids consistently. The
        # vocabulary is built from the WHOLE text (before splitting) so the model
        # never meets an unseen character at validation time.
        chars = sorted(set(text))
        self.stoi = {c: i for i, c in enumerate(chars)}
        self.itos = {i: c for i, c in enumerate(chars)}
        self.vocab_size = len(chars)
        data = torch.tensor([self.stoi[c] for c in text], dtype=torch.long)
        self.data = data
        # Held-out split: the LAST `val_split` fraction is validation, so we can
        # report generalization, not just how well the model memorized training
        # text. A contiguous tail (not random windows) keeps train/val disjoint.
        n_val = int(len(data) * val_split)
        if n_val and n_val < seq_len + 1:
            n_val = 0  # too small to sample a window; skip validation silently
        if len(data) - n_val < seq_len + 1:
            # A big --val-split on a small corpus can starve the train split of a
            # full window. Fail loudly (as with the length guard above) rather
            # than crash cryptically inside get_batch.
            raise ValueError(
                f"--val-split {val_split} leaves only {len(data) - n_val:,} "
                f"chars for training, but --seq-len {seq_len} needs at least "
                f"{seq_len + 1}; use a smaller --val-split, a longer --data "
                f"file, or a smaller --seq-len."
            )
        self.splits = {
            "train": data[: len(data) - n_val],
            "val": data[len(data) - n_val:] if n_val else data[:0],
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
        return x.to(device), y.to(device)

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


class SNNCharLM(nn.Module):
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
        for l in range(cfg.num_layers):
            in_dim = cfg.vocab_size if l == 0 else cfg.hidden
            self.fc_in.append(nn.Linear(in_dim, cfg.hidden))
            self.lifs.append(
                snn.Leaky(
                    beta=cfg.beta,                 # DESIGN: membrane leak (see below)
                    threshold=cfg.threshold,
                    spike_grad=spike_grad,         # surrogate gradient for dS/dU
                    learn_beta=cfg.learn_beta,     # optionally learn the time constant
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
        spike_input = spikegen.rate(char_onehot, num_steps=self.cfg.num_steps,
                                    gain=self.cfg.rate_gain)             # (T, B, V)
        cur0 = self.fc_in[0](spike_input)                                # (T, B, hidden)
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
        # Encode and project the WHOLE chunk at once: one rate-coding call and
        # one big matmul replace L per-position encodes and L*T micro-GEMMs. This
        # is numerically identical (given the same spikes) to encoding per
        # position -- only the sequential recurrent loop below is unavoidable.
        spikes = spikegen.rate(onehot, num_steps=self.cfg.num_steps,
                               gain=self.cfg.rate_gain)                 # (T, B, L, V)
        cur0_all = self.fc_in[0](spikes)                               # (T, B, L, hidden)
        rates = []
        for pos in range(L):
            rate, mems = self._char_rate(cur0_all[:, :, pos, :], mems)
            rates.append(rate)
        rates = torch.stack(rates, dim=1)                              # (B, L, hidden)
        return self.readout(rates), mems                              # (B, L, V) one GEMM

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, L) ids -> logits (B, L, vocab_size), fresh state (for eval)."""
        return self.forward_seq(x, None)[0]

    # -- autoregressive sampling --------------------------------------------
    @torch.no_grad()
    def generate(self, dataset: "CharDataset", prompt: str, length: int,
                 device: torch.device, temperature: float = 1.0,
                 top_k: Optional[int] = None) -> str:
        """Warm the recurrent state on `prompt`, then sample `length` chars.

        The membrane state persists across the whole generation, exactly as
        during training, so the leaky memory conditions every new character on
        everything generated so far.
        """
        self.eval()
        mems = self.init_state(1, device)
        out_ids: List[int] = []

        # Warm-up: feed the prompt so the membrane state reflects it.
        ids = dataset.encode(prompt).to(device)
        last_logits = None
        for cid in ids:
            oh = F.one_hot(cid.view(1), self.cfg.vocab_size).float()
            last_logits, mems = self.step(oh, mems)
            out_ids.append(int(cid))

        # If the prompt was empty, seed from a zero one-hot so we have logits.
        if last_logits is None:
            oh = torch.zeros(1, self.cfg.vocab_size, device=device)
            last_logits, mems = self.step(oh, mems)

        for _ in range(length):
            # Guard: NaN/inf logits would make torch.multinomial trigger a CUDA
            # device assert; sanitising keeps sampling robust on any checkpoint.
            last_logits = torch.nan_to_num(last_logits)
            logits = last_logits.squeeze(0) / max(temperature, 1e-6)
            if top_k is not None and top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.numel()))
                logits[logits < v[-1]] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            next_id = int(torch.multinomial(probs, 1).item())
            out_ids.append(next_id)
            oh = F.one_hot(torch.tensor([next_id], device=device),
                           self.cfg.vocab_size).float()
            last_logits, mems = self.step(oh, mems)

        return dataset.decode(out_ids)


# ---------------------------------------------------------------------------
# 3. Training loop
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate(model: SNNCharLM, dataset: CharDataset, split: str,
             batch_size: int, n_batches: int, device: torch.device) -> float:
    """Mean bits-per-character on held-out windows (lower is better).

    We run full windows with a fresh membrane and no gradient -- the same
    conditions the model trains under -- and average the per-character cross
    entropy over several random batches. On the `val` split this measures
    generalization, not memorization.
    """
    was_training = model.training
    model.eval()
    total_nats, total_chars = 0.0, 0
    for _ in range(n_batches):
        x, y = dataset.get_batch(batch_size, device, split=split)
        logits = model(x)                                              # (B, L, V)
        loss = F.cross_entropy(logits.reshape(-1, model.cfg.vocab_size),
                               y.reshape(-1), reduction="sum")
        total_nats += loss.item()
        total_chars += y.numel()
    if was_training:
        model.train()
    return total_nats / total_chars / math.log(2)


def train(args) -> None:
    set_seed(args.seed)
    device = pick_device()

    text = _load_text(args.data)
    dataset = CharDataset(text, seq_len=args.seq_len, val_split=args.val_split)
    n_val = len(dataset.splits["val"])
    print(f"[data] {len(text):,} chars | vocab={dataset.vocab_size} "
          f"| train={len(dataset.splits['train']):,} val={n_val:,} "
          f"| device={device}")

    if args.init_from:
        # Warm start: continue training from a saved model's weights (fresh
        # optimizer and LR schedule). The architecture comes from the checkpoint
        # -- CLI architecture flags are ignored -- but dropout and seq_len are
        # taken from the CLI so a continuation can adjust regularization or
        # context length without touching the weights.
        # weights_only=True: the checkpoint is pure data (a tensor state_dict plus
        # plain cfg/stoi/itos dicts), so it loads under torch's safe unpickler --
        # we never execute arbitrary pickle code just to read a shared or
        # downloaded .pt file. (See _load_checkpoint for the same guard.)
        ckpt = torch.load(args.init_from, map_location=device, weights_only=True)
        cfg = SNNConfig(**ckpt["cfg"])
        if cfg.vocab_size != dataset.vocab_size:
            raise ValueError(
                f"--init-from checkpoint has vocab_size={cfg.vocab_size} but the "
                f"corpus gives {dataset.vocab_size}; use the same --data file.")
        # Same vocab SIZE is not enough: the weights are indexed by character id,
        # so a different character->id mapping (same count, different set/order)
        # would silently scramble fc_in[0] and the readout. Enforce the mapping.
        ckpt_stoi = ckpt.get("stoi")
        if ckpt_stoi is not None and ckpt_stoi != dataset.stoi:
            n_diff = sum(1 for c in set(ckpt_stoi) | set(dataset.stoi)
                         if ckpt_stoi.get(c) != dataset.stoi.get(c))
            raise ValueError(
                f"--init-from checkpoint's character->id mapping differs from the "
                f"current corpus in {n_diff} symbol(s) (same vocab size, different "
                f"characters/order); the warm-started weights would be indexed by "
                f"mismatched ids. Use the same --data file.")
        cfg.dropout = args.dropout
        cfg.seq_len = args.seq_len
        model = SNNCharLM(cfg).to(device)
        model.load_state_dict(ckpt["state_dict"])
        print(f"[init] warm start from {args.init_from} "
              f"(arch from checkpoint: hidden={cfg.hidden} layers={cfg.num_layers})")
    else:
        cfg = SNNConfig(
            vocab_size=dataset.vocab_size,
            hidden=args.hidden,
            num_layers=args.layers,
            num_steps=args.num_steps,
            beta=args.beta,
            rate_gain=args.rate_gain,
            surrogate=args.surrogate,
            learn_beta=args.learn_beta,
            dropout=args.dropout,
            seq_len=args.seq_len,
        )
        model = SNNCharLM(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] {n_params/1e6:.2f}M params | hidden={cfg.hidden} "
          f"layers={cfg.num_layers} T={cfg.num_steps} beta={cfg.beta}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay)
    # DESIGN: LR warmup + cosine decay. `--steps` counts optimizer updates, which
    # is exactly the schedule horizon. A short warmup lets the spiking dynamics
    # settle before cosine-decaying to a fine polish.
    def lr_at(update: int) -> float:
        if update < args.warmup:
            return args.lr * (update + 1) / args.warmup
        prog = (update - args.warmup) / max(1, args.steps - args.warmup)
        return args.lr * 0.5 * (1 + math.cos(math.pi * min(prog, 1.0)))

    chunk = args.tbptt_chunk
    print(f"[train] truncated BPTT: window={args.seq_len} chunk={chunk} "
          f"-> backprop depth = chunk*T = {chunk * cfg.num_steps} spiking steps")

    do_val = dataset.has_val()
    best_val = float("inf")
    model.train()
    running = 0.0
    n_since_log = 0
    t0 = time.time()
    update = 0
    while update < args.steps:
        # A fresh random window of the corpus. The membrane is reset to zero at
        # the window start; within the window it persists across chunks.
        x, y = dataset.get_batch(args.batch_size, device)              # (B, L)
        # DESIGN: truncated backpropagation through time (TBPTT).
        # We walk the window in chunks of `chunk` characters. The membrane state
        # is carried from one chunk to the next -- so the network still integrates
        # context across the whole window -- but it is DETACHED at each boundary,
        # so backprop only unrolls chunk*num_steps spiking steps instead of
        # seq_len*num_steps. That bounds both memory and the length of the
        # gradient path, letting --seq-len be long without blowing up either.
        mems = model.init_state(args.batch_size, device)
        for c0 in range(0, x.size(1), chunk):
            if update >= args.steps:
                break
            mems = [m.detach() for m in mems]                          # cut the graph
            for g in opt.param_groups:
                g["lr"] = lr_at(update)
            xc, yc = x[:, c0:c0 + chunk], y[:, c0:c0 + chunk]
            logits, mems = model.forward_seq(xc, mems)                 # (B, chunk, V)
            loss = F.cross_entropy(
                logits.reshape(-1, cfg.vocab_size), yc.reshape(-1))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            # DESIGN: gradient clipping -- standard insurance for a recurrent net.
            # The leaky membrane is contractive so gradients stay well-behaved
            # (norms of order 1-10), but clipping the global norm cheaply caps the
            # occasional larger step and keeps training smooth.
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()
            update += 1

            running += loss.item()
            n_since_log += 1
            if update % args.log_every == 0:
                avg = running / n_since_log
                running = 0.0
                n_since_log = 0
                bpc = avg / math.log(2)          # bits-per-character
                speed = args.log_every / (time.time() - t0)
                t0 = time.time()
                mem = (torch.cuda.max_memory_allocated() / 1e9
                       if device.type == "cuda" else 0.0)
                print(f"upd {update:>6} | loss {avg:6.3f} | bpc {bpc:5.3f} "
                      f"| ppl {math.exp(avg):8.2f} | {speed:4.1f} upd/s "
                      f"| peakGPU {mem:4.2f}GB")

            # Held-out evaluation. We report validation bpc (generalization) and
            # keep the checkpoint with the best val score -- the last step is not
            # necessarily the best once the model starts to overfit.
            if do_val and args.eval_every and update % args.eval_every == 0:
                val_bpc = evaluate(model, dataset, "val", args.batch_size,
                                   args.eval_batches, device)
                tag = ""
                if val_bpc < best_val:
                    best_val = val_bpc
                    _save_checkpoint(args.ckpt, model, cfg, dataset)
                    tag = "  <- best (saved)"
                print(f"       val bpc {val_bpc:.3f} | ppl {2**val_bpc:6.2f}{tag}")

            if args.sample_every and update % args.sample_every == 0:
                sample = model.generate(dataset, prompt=args.prompt or "the ",
                                        length=200, device=device,
                                        temperature=0.8, top_k=args.top_k)
                print("-" * 60)
                print(sample)
                print("-" * 60)
                model.train()

    # Always finish with a checkpoint on disk. With validation we run one final
    # eval and keep it if it beats the running best (this also covers the case
    # where the schedule never triggered a mid-training eval, so best_val is
    # still +inf and nothing has been saved yet).
    if do_val:
        final_val = evaluate(model, dataset, "val", args.batch_size,
                             args.eval_batches, device)
        if final_val < best_val:
            best_val = final_val
            _save_checkpoint(args.ckpt, model, cfg, dataset)
        print(f"[done] best val bpc {best_val:.3f} | checkpoint -> {args.ckpt}")
    else:
        _save_checkpoint(args.ckpt, model, cfg, dataset)
        print(f"[done] saved checkpoint -> {args.ckpt}")


# ---------------------------------------------------------------------------
# 4. Checkpoint I/O
# ---------------------------------------------------------------------------
def _save_checkpoint(path: str, model: SNNCharLM, cfg: SNNConfig,
                     dataset: CharDataset) -> None:
    torch.save({
        "state_dict": model.state_dict(),
        "cfg": asdict(cfg),
        "stoi": dataset.stoi,
        "itos": dataset.itos,
    }, path)


def _load_checkpoint(path: str, device: torch.device
                     ) -> Tuple[SNNCharLM, CharDataset]:
    # weights_only=True guards against arbitrary-code execution when loading a
    # shared or downloaded checkpoint (the payload is only tensors + plain dicts).
    ckpt = torch.load(path, map_location=device, weights_only=True)
    cfg = SNNConfig(**ckpt["cfg"])
    model = SNNCharLM(cfg).to(device)
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
def sample(args) -> None:
    device = pick_device()
    model, ds = _load_checkpoint(args.ckpt, device)
    text = model.generate(ds, prompt=args.prompt, length=args.length,
                          device=device, temperature=args.temperature,
                          top_k=args.top_k)
    print(text)


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
    device = pick_device()
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
            mems = [m.detach() for m in mems]
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
    t.add_argument("--ckpt", type=str, default="snn_char_lm.pt")
    t.add_argument("--init-from", dest="init_from", type=str, default=None,
                   help="warm-start weights from this checkpoint (architecture "
                        "is taken from it; fresh optimizer + LR schedule)")
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
    t.add_argument("--learn-beta", dest="learn_beta", action="store_true")
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
    t.set_defaults(func=train)

    # -- sample --
    s = sub.add_parser("sample", help="generate text from a checkpoint")
    s.add_argument("--ckpt", type=str, default="snn_char_lm.pt")
    s.add_argument("--prompt", type=str, default="the ")
    s.add_argument("--length", type=_positive_int, default=400)
    s.add_argument("--temperature", type=float, default=0.8)
    s.add_argument("--top-k", dest="top_k", type=int, default=None)
    s.set_defaults(func=sample)

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
    sm.set_defaults(func=smoke)

    return p


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
