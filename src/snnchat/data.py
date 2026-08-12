"""Weighted mixture sampling over the packed per-source streams.

The one property worth stating up front, because everything else follows from
it: **`batch(step)` is a pure function of `(seed, step)`**. It draws from a
generator seeded per call rather than advancing a stateful stream, so a run
resumed at step k sees exactly the batch an uninterrupted run would have seen at
step k. That is `snn.data`'s design decision (see its module docstring, item 3)
and it is inherited here rather than reinvented, because it is what makes a
restartable trainer testable instead of hopeful -- and this trainer *must* be
restartable, since a multi-hour run cannot survive the harness's process
lifetime.

The mixture makes that slightly harder than it is for a single corpus: a batch
draws its per-element source from the weights and only then draws an offset
within that source, so the generator is consumed in a fixed order (sources
first, then offsets, in source-id order) regardless of what the weights are.
Consuming it in *any* data-dependent order would make the batch depend on
floating-point comparison outcomes and quietly break the guarantee.
"""

from __future__ import annotations

import json
import os

import numpy as np
from torch import Tensor

# Imported deliberately from the research package rather than copied: `_mix64`
# is the seed-mixing function whose output the resume guarantee is defined
# against, and `_to_int64_cpu` carries a non-obvious contract about page-locked
# host memory and in-flight async copies. Two copies of either would be two
# things to keep in sync. They are private names, and this is the one place
# `snnchat` reaches for one.
from snn.data import _mix64, _to_int64_cpu
from snnchat.tokenizer import BOS, BOT

__all__ = ["ChatCorpus", "MixtureSampler", "load_manifest", "bot_turn_weights"]


def load_manifest(data_dir: str = "data/chat") -> dict:
    path = os.path.join(data_dir, "manifest.json")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. Run:\n"
            f"    python scripts/chat/build_data.py --out-dir {data_dir}"
        )
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


class ChatCorpus:
    """Memory-mapped access to the packed per-source streams.

    Deliberately does not build anything, for the reason `snn.data.Corpus` does
    not download: a training run that silently spends twenty minutes packing a
    corpus it did not announce is a run whose wall-clock means nothing.
    """

    def __init__(self, data_dir: str = "data/chat", vocab_version: int | None = None):
        self.data_dir = data_dir
        self.manifest = load_manifest(data_dir)
        self.vocab_size = int(self.manifest["vocab_size"])
        self.vocab_version = int(self.manifest["vocab_version"])
        if vocab_version is not None and vocab_version != self.vocab_version:
            raise RuntimeError(
                f"corpus was packed under vocab v{self.vocab_version} but this "
                f"code/checkpoint expects v{vocab_version}. The ids mean different "
                f"characters; repack the corpus rather than decoding it anyway."
            )
        self._train: dict[str, np.memmap] = {}
        self._val: dict[str, np.memmap] = {}
        self.available: list[str] = []
        for name in sorted(self.manifest["sources"]):
            path = os.path.join(data_dir, f"{name}.bin")
            if os.path.exists(path) and os.path.getsize(path) > 0:
                self.available.append(name)

    def train(self, name: str) -> np.memmap:
        if name not in self._train:
            path = os.path.join(self.data_dir, f"{name}.bin")
            self._train[name] = np.memmap(path, dtype=np.uint8, mode="r")
        return self._train[name]

    def val(self, name: str) -> np.memmap | None:
        if name not in self._val:
            path = os.path.join(self.data_dir, f"{name}.val.bin")
            if not os.path.exists(path):
                return None
            self._val[name] = np.memmap(path, dtype=np.uint8, mode="r")
        return self._val[name]

    def train_chars(self) -> dict[str, int]:
        return {n: int(len(self.train(n))) for n in self.available}

    def __repr__(self) -> str:
        sizes = ", ".join(f"{n}={len(self.train(n)) / 1e6:,.0f}M" for n in self.available)
        return f"ChatCorpus(V={self.vocab_size}, {sizes})"


class MixtureSampler:
    """Draw `[B, L]` windows from several streams at fixed character ratios.

    `weights` are normalised over the sources that are actually present, so a
    corpus built without (say) `oasst1` trains on the rest at the correct
    relative proportions instead of silently under-filling the batch. The
    realised weights are exposed as `self.weights` and are logged by the trainer,
    because "the mix I asked for" and "the mix I got" differing is exactly the
    kind of thing that is invisible until someone reads a sample and wonders why
    it sounds like a story.
    """

    def __init__(
        self,
        corpus: ChatCorpus,
        weights: dict[str, float],
        batch_size: int,
        seq_len: int,
        seed: int,
        *,
        split: str = "train",
        align_frac: float = 0.0,
        align_lookahead: int = 0,
    ) -> None:
        self.corpus = corpus
        self.batch_size = int(batch_size)
        self.seq_len = int(seq_len)
        self.seed = int(seed)
        self.split = split
        self.align_frac = float(align_frac)
        self.align_lookahead = int(align_lookahead) or int(seq_len)

        getter = corpus.train if split == "train" else corpus.val
        names, arrays, ws = [], [], []
        for name in corpus.available:
            w = float(weights.get(name, 0.0))
            if w <= 0.0:
                continue
            data = getter(name)
            if data is None or len(data) <= seq_len:
                continue
            names.append(name)
            arrays.append(np.asarray(data))
            ws.append(w)
        if not names:
            raise ValueError(
                f"no source in {corpus.available} has a positive weight and enough "
                f"characters for seq_len={seq_len}"
            )

        # Renormalising over the sources that are PRESENT is deliberate and
        # tested (`test_weights_renormalise_over_present_sources_only`) -- it is
        # what lets one mixture be evaluated against a partial corpus. Doing it
        # SILENTLY is not: a requested source that was never packed then trains a
        # mixture that appears nowhere in the repository, inflating every
        # survivor, and the only symptom is a worse model.
        #
        # `DEFAULT_MIX` names `stories_topic`, which `scripts/chat/build_data.py`
        # does not produce -- it comes from `build_topic_stories.py` -- so this
        # warning fires on exactly the path that used to be quiet.
        missing = {n for n, w in weights.items() if w > 0.0} - set(names)
        if missing:
            lost = sum(float(weights[n]) for n in missing)
            print(
                f"  WARNING: {len(missing)} requested source(s) not in the corpus: "
                f"{', '.join(sorted(missing))}\n"
                f"           they carried {lost:.3f} of the requested weight; the "
                f"rest is renormalised by {1.0 / max(1.0 - lost, 1e-9):.3f}x.\n"
                f"           available: {', '.join(sorted(corpus.available))}",
                flush=True,
            )

        total = sum(ws)
        self.names: list[str] = names
        self.arrays: list[np.ndarray] = arrays
        self.weights: list[float] = [w / total for w in ws]
        self._cum = np.cumsum(self.weights)
        # Last legal start: a window reads [o, o+L] inclusive -- L inputs plus
        # the shifted target -- so o <= len - L - 1, giving len - L legal starts.
        # Same accounting as snn.data.RandomWindowSampler, and wrong by one in
        # either direction it either drops the final character of a source or
        # reads past its end.
        self._n_offsets = [len(a) - seq_len for a in arrays]

    def source_of(self, step: int) -> np.ndarray:
        """Which source each of the B elements is drawn from, at `step`."""
        g = np.random.default_rng(_mix64(self.seed, step))
        u = g.random(self.batch_size)
        return np.searchsorted(self._cum, u, side="right").clip(0, len(self.names) - 1)

    def _align(self, s: int, offs: np.ndarray, take: np.ndarray) -> np.ndarray:
        """Nudge the chosen offsets forward to the next conversation start.

        WHY, MEASURED
        -------------
        A window drawn at a uniformly random offset usually begins in the middle
        of somebody's turn. At `seq_len = 256` and turns of up to 400
        characters, the request a reply is conditioned on is frequently OUTSIDE
        the window that reply is trained in -- so the gradient that would teach
        "the story is about what was asked for" is not available in most of the
        windows that contain a story. `docs/chat/QUALITY.md` measures what that
        costs: 7.8 % of topic-conditioned requests were answered on topic.

        Alignment moves each offset to the next `<|bos|>` within
        `align_lookahead` characters, so the window holds a conversation from
        its first character -- which is also exactly the state `ChatSession`
        starts a real conversation in. Offsets with no marker in reach are left
        alone rather than dropped, because dropping them would make the realised
        source mixture depend on how long that source's conversations are.

        The scan is a slice comparison per element and consumes no random
        numbers, so `batch(step)` stays a pure function of `(seed, step)`.
        """
        arr = self.arrays[s]
        limit = self._n_offsets[s]
        for i in take:
            o = int(offs[i])
            end = min(o + self.align_lookahead, limit)
            if end <= o:
                continue
            hit = np.flatnonzero(arr[o:end] == BOS)
            if hit.size:
                offs[i] = o + int(hit[0])
        return offs

    def _windows(self, step: int) -> np.ndarray:
        """`[B, L+1]` uint8. Generator order is fixed; see the module docstring."""
        g = np.random.default_rng(_mix64(self.seed, step))
        u = g.random(self.batch_size)
        src = np.searchsorted(self._cum, u, side="right").clip(0, len(self.names) - 1)
        span = np.arange(self.seq_len + 1, dtype=np.int64)
        out = np.empty((self.batch_size, self.seq_len + 1), dtype=np.uint8)
        # An extra draw ONLY when alignment is on, so a run with align_frac=0
        # consumes the identical generator stream it did before this existed and
        # reproduces earlier runs bit for bit.
        aligned = (
            g.random(self.batch_size) < self.align_frac
            if self.align_frac > 0.0
            else None
        )
        # Sources in id order, never in "order encountered", so the number of
        # generator draws at step k does not depend on which sources happened to
        # be selected.
        for s in range(len(self.names)):
            rows = np.flatnonzero(src == s)
            offs = g.integers(0, self._n_offsets[s], size=self.batch_size, dtype=np.int64)
            if rows.size:
                if aligned is not None:
                    # `offs[i]` is the offset for batch row `rows[i]`, so the
                    # decision for it is `aligned[rows[i]]`.
                    offs = self._align(s, offs, np.flatnonzero(aligned[rows]))
                idx = offs[:rows.size, None] + span[None, :]
                out[rows] = self.arrays[s][idx]
        return out

    def batch(self, step: int) -> tuple[Tensor, Tensor]:
        block = self._windows(step)
        return _to_int64_cpu(block[:, :-1]), _to_int64_cpu(block[:, 1:])

    def batch_weighted(self, step: int, bot_weight: float) -> tuple[Tensor, Tensor, np.ndarray]:
        """`(x, y, w)` where `w` upweights the characters of the model's OWN turns.

        See `bot_turn_weights` for what the third value is and why it is not a
        mask.
        """
        block = self._windows(step)
        w = bot_turn_weights(block, bot_weight)
        return _to_int64_cpu(block[:, :-1]), _to_int64_cpu(block[:, 1:]), w

    def mix_report(self) -> str:
        return ", ".join(f"{n} {w:.0%}" for n, w in zip(self.names, self.weights))


def bot_turn_weights(block: np.ndarray, bot_weight: float) -> np.ndarray:
    """Per-target loss weight: `bot_weight` inside a `<|bot|>` turn, else 1.

    UPWEIGHTING, NOT MASKING -- THE DISTINCTION IS THE WHOLE POINT
    --------------------------------------------------------------
    `snnchat.train`'s module docstring argues against loss masking and the
    argument is right: at this scale the model needs every character of language
    signal it can get, and it has to be able to PREDICT a user turn in order to
    represent where one ends, which is what makes `<|eot|>` mean anything. A
    mask sets the user's turns to zero and throws that away.

    A weight does not. Every character still contributes and still receives
    gradient; the reply's characters simply contribute more. At `bot_weight = 1`
    this function returns all ones and the loss is exactly what it was before --
    which is why the trainer keeps the unweighted code path for that case rather
    than multiplying by a tensor of ones.

    The weight of the prediction made at position `j` is set by the role of
    position `j` -- the CONTEXT the model is speaking from -- and not by the role
    of the character `j + 1` being predicted. The two differ at exactly the two
    ends of a turn, and both differences matter:

    * at the `<|bot|>` marker the model is already in its own turn, so the first
      character of the reply is weighted;
    * at the last character of a reply the model is still in its own turn, so
      the closing `<|eot|>` is weighted -- **knowing when to stop is part of the
      reply**, and it is the one prediction the alternative indexing would drop;
    * predicting the `<|bot|>` marker itself stays at weight 1, which is right:
      it always follows the user's `<|eot|>` and is the most trivial prediction
      in the corpus.

    The role of a character is the last marker at or before it, forward-filled.
    Characters before the window's first marker have no known role and are
    weighted 1: with random window starts that is a real fraction of the batch,
    and guessing would silently weight a user turn as a reply.
    """
    if bot_weight == 1.0:
        return np.ones((block.shape[0], block.shape[1] - 1), dtype=np.float32)
    width = block.shape[1]
    is_marker = block < 4
    idx = np.where(is_marker, np.arange(width, dtype=np.int64)[None, :], 0)
    np.maximum.accumulate(idx, axis=1, out=idx)
    role = np.take_along_axis(block, idx, axis=1)
    seen = np.maximum.accumulate(is_marker, axis=1)
    is_bot = (role == BOT) & seen
    return (1.0 + (bot_weight - 1.0) * is_bot[:, :-1]).astype(np.float32)


class SequentialEval:
    """Non-overlapping windows over one source's val split, in order.

    Protocol A ("fresh state") from `snn.data`, and only that one. The carried
    protocol is not offered here on purpose: this corpus is a mixture of
    unrelated conversations, so carrying membrane state across a window boundary
    would carry it from the end of one stranger's conversation into the start of
    another's. That number would be meaningless, and offering it would invite
    someone to quote it next to the research corpus's carried bpc, which it is
    not comparable with.
    """

    def __init__(self, corpus: ChatCorpus, name: str, batch_size: int, seq_len: int):
        data = corpus.val(name)
        if data is None:
            raise FileNotFoundError(f"no val split packed for {name!r}")
        self.data = np.asarray(data)
        self.batch_size = int(batch_size)
        self.seq_len = int(seq_len)
        self.name = name

    def __iter__(self):
        n_windows = max(len(self.data) - 1, 0) // self.seq_len
        n_batches = n_windows // self.batch_size
        span = np.arange(self.seq_len + 1, dtype=np.int64)
        for b in range(n_batches):
            starts = (np.arange(self.batch_size, dtype=np.int64) + b * self.batch_size) * self.seq_len
            block = self.data[starts[:, None] + span[None, :]]
            yield _to_int64_cpu(block[:, :-1]), _to_int64_cpu(block[:, 1:])

    def n_batches(self) -> int:
        return (max(len(self.data) - 1, 0) // self.seq_len) // self.batch_size


__all__.append("SequentialEval")
