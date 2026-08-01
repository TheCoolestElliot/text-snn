"""Corpus acquisition, verification, memory-mapped access, and samplers.

Spec: docs/reports/02a_phase2_spec.md §4. Gate A5 (reconnaissance §6).

Three decisions in here are worth stating up front, because they are what make
the rest of the project reproducible:

1.  **Everything is checksummed, and the checksums are literals in this file.**
    They were computed on 2026-08-01 from live downloads off mattmahoney.net and
    are pinned so that a future run detects a changed mirror rather than
    silently training on different bytes. §4.1 of the spec calls this gate A5.

2.  **The splits are stored already remapped to contiguous ids.** enwik8 uses
    205 of the 256 byte values, so a raw-byte model would waste 51 embedding
    rows and, worse, the vocabulary would depend on which bytes happened to
    appear in a split. Remapping once at acquisition time -- using the vocabulary
    of the *whole* 100 MB file, which is the standard enwik8/text8 protocol and
    what published bpc numbers use -- means the sampler is a pure slice with no
    per-batch lookup table, and the embedding is exactly V rows.

3.  **The training sampler is a pure function of (seed, step).** It draws its
    offsets from a `torch.Generator` seeded per call rather than from a stateful
    stream, so resuming a checkpoint at step k draws exactly the batch an
    uninterrupted run would have drawn at step k. That is what makes R6
    (resumable checkpoints) testable rather than hopeful.

Host RAM is 15.4 GiB (reconnaissance §2.1, bottleneck B9), so the corpus is
memory-mapped uint8 and never materialised as a tensor.
"""

from __future__ import annotations

import hashlib
import http.client
import json
import os
import shutil
import urllib.error
import urllib.request
import zipfile
from collections.abc import Iterator
from dataclasses import dataclass, field

import numpy as np
import torch
from torch import Tensor

# Imported for its import-time side effect only: snn.config installs
# CUBLAS_WORKSPACE_CONFIG before torch can create a CUDA context, and this
# module queries torch.cuda when deciding whether to page-lock batches. Any
# entry point that reaches CUDA through the data pipeline alone would otherwise
# lose determinism silently. snn.config does not import snn.data, so this is not
# a cycle.
from snn import config as _config  # noqa: F401

# --------------------------------------------------------------------------
# Corpus registry
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class CorpusSpec:
    """Everything needed to acquire one corpus and prove we got the right bytes.

    `raw_sha256` and `vocab_size` are not in the spec's illustrative listing but
    are required by it in prose ("meta.json records ... the extracted-file
    SHA-256"; "Published expectations, asserted"). They are appended after the
    five listed fields so positional construction still matches the spec.
    """

    urls: list[str]
    zip_sha256: str
    member: str
    raw_bytes: int
    split: tuple[int, int, int]
    raw_sha256: str = ""
    vocab_size: int = 0

    def __post_init__(self) -> None:
        if sum(self.split) != self.raw_bytes:
            raise ValueError(
                f"split {self.split} sums to {sum(self.split)}, not raw_bytes={self.raw_bytes}"
            )


#: SHA-256 values measured on 2026-08-01 by downloading from the primary mirror.
#: enwik8.zip 36,445,475 B; text8.zip 31,344,016 B; both unpack to exactly
#: 100,000,000 B. Vocabulary sizes match the published 205 / 27.
CORPORA: dict[str, CorpusSpec] = {
    "enwik8": CorpusSpec(
        urls=[
            "http://mattmahoney.net/dc/enwik8.zip",
            "https://data.deepai.org/enwik8.zip",
        ],
        zip_sha256="547994d9980ebed1288380d652999f38a14fe291a6247c157c3d33d4932534bc",
        member="enwik8",
        raw_bytes=100_000_000,
        split=(90_000_000, 5_000_000, 5_000_000),
        raw_sha256="2b49720ec4d78c3c9fabaee6e4179a5e997302b3a70029f30f2d582218c024a8",
        vocab_size=205,
    ),
    "text8": CorpusSpec(
        urls=["http://mattmahoney.net/dc/text8.zip"],
        zip_sha256="a6640522afe85d1963ad56c05b0ede0a0c000dddc9671758a6cc09b7a38e5232",
        member="text8",
        raw_bytes=100_000_000,
        split=(90_000_000, 5_000_000, 5_000_000),
        raw_sha256="6e890197040d37d85beb962ae1f041ff1d9a9ca8d20c7d99c85027eebf51dca7",
        vocab_size=27,
    ),
}

SPLIT_NAMES = ("train", "val", "test")

#: meta.json layout version. Bump if the on-disk representation changes, so a
#: stale data/ directory is rebuilt instead of misread.
META_VERSION = 1

#: Sentinel stored in `stoi` for byte values that never occur in the corpus.
#: Safe because both corpora have V <= 205 < 255; asserted at build time.
UNUSED_ID = 255


@dataclass
class CorpusPaths:
    """Filesystem locations produced by `ensure_corpus`.

    The spec names the return type but not its fields; this is the shape other
    modules should rely on.
    """

    name: str
    root: str            # data/<name>/
    zip_path: str        # data/<name>.zip
    raw_path: str        # data/<name>/<member>          (the unmodified 100 MB)
    meta_path: str       # data/<name>/meta.json
    splits: dict[str, str] = field(default_factory=dict)  # "train" -> .bin path

    @property
    def train(self) -> str:
        return self.splits["train"]

    @property
    def val(self) -> str:
        return self.splits["val"]

    @property
    def test(self) -> str:
        return self.splits["test"]


# --------------------------------------------------------------------------
# Acquisition
# --------------------------------------------------------------------------


def sha256_file(path: str, chunk: int = 1 << 20) -> str:
    """Streaming SHA-256; the files are 100 MB and host RAM is 15.4 GiB."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def _download(urls: list[str], dest: str, timeout: float = 300.0) -> str:
    """Try each mirror in order; return the URL that worked.

    A partial download is deleted rather than left behind, because a truncated
    zip that fails its checksum on the *next* run is a confusing failure mode.

    `http.client.HTTPException` is caught alongside the socket errors: an
    `IncompleteRead` part-way through a 36 MB transfer is the most likely way
    mattmahoney.net fails, and it is *not* an OSError, so without it the second
    mirror would never be tried and the fallback would be decorative.
    """
    errors: list[str] = []
    tmp = dest + ".part"
    for url in urls:
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "snn-phase2/1.0"})
            with urllib.request.urlopen(req, timeout=timeout) as resp, open(tmp, "wb") as out:
                shutil.copyfileobj(resp, out, length=1 << 20)
            os.replace(tmp, dest)
            return url
        except (
            urllib.error.URLError,
            http.client.HTTPException,
            OSError,
            TimeoutError,
        ) as exc:
            errors.append(f"{url}: {type(exc).__name__}: {exc}")
            if os.path.exists(tmp):
                os.remove(tmp)
    raise RuntimeError("all mirrors failed:\n  " + "\n  ".join(errors))


def _verify(path: str, expected_sha: str, expected_bytes: int, label: str) -> None:
    size = os.path.getsize(path)
    if size != expected_bytes:
        raise RuntimeError(f"{label}: expected {expected_bytes} bytes, got {size} ({path})")
    got = sha256_file(path)
    if got != expected_sha:
        raise RuntimeError(
            f"{label}: SHA-256 mismatch\n  expected {expected_sha}\n  got      {got}\n  {path}"
        )


def _build_vocab(raw_path: str) -> tuple[np.ndarray, np.ndarray]:
    """Vocabulary over the WHOLE file, not per split.

    This is the standard enwik8/text8 protocol: published bpc numbers assume the
    model knows all 205 (resp. 27) symbols, so building the vocabulary from the
    training split alone would make val/test contain unmappable bytes and would
    quietly change the task.
    """
    counts = np.zeros(256, dtype=np.int64)
    with open(raw_path, "rb") as f:
        while True:
            block = f.read(1 << 24)
            if not block:
                break
            counts += np.bincount(np.frombuffer(block, dtype=np.uint8), minlength=256)
    itos = np.flatnonzero(counts).astype(np.uint8)          # sorted byte values
    if itos.size > UNUSED_ID:
        raise RuntimeError(
            f"vocabulary of {itos.size} symbols collides with the {UNUSED_ID} sentinel"
        )
    stoi = np.full(256, UNUSED_ID, dtype=np.uint8)
    stoi[itos] = np.arange(itos.size, dtype=np.uint8)
    return stoi, itos


def _write_splits(raw_path: str, spec: CorpusSpec, root: str, stoi: np.ndarray) -> dict[str, str]:
    """Write train/val/test as raw uint8 ids, in corpus order, streaming.

    The remap is applied here exactly once, so the sampler never needs a lookup.
    """
    paths: dict[str, str] = {}
    offsets = np.cumsum((0,) + spec.split)
    with open(raw_path, "rb") as src:
        for name, start, stop in zip(SPLIT_NAMES, offsets[:-1], offsets[1:]):
            out_path = os.path.join(root, f"{name}.bin")
            remaining = int(stop - start)
            src.seek(int(start))
            with open(out_path, "wb") as out:
                while remaining:
                    block = src.read(min(remaining, 1 << 24))
                    if not block:
                        raise RuntimeError(f"{raw_path}: unexpected EOF building {name}")
                    remaining -= len(block)
                    out.write(stoi[np.frombuffer(block, dtype=np.uint8)].tobytes())
            paths[name] = out_path
    return paths


def ensure_corpus(name: str, data_dir: str = "data") -> CorpusPaths:
    """Download, verify, extract and split a corpus. Idempotent.

    On a second call with everything present this re-verifies every checksum
    (~1 s for 300 MB) and returns without touching the network. A checksum
    mismatch is a hard failure: it means the mirror changed or the disk lied,
    and either way the run must not proceed.
    """
    if name not in CORPORA:
        raise KeyError(f"unknown corpus {name!r}; known: {sorted(CORPORA)}")
    spec = CORPORA[name]

    root = os.path.join(data_dir, name)
    os.makedirs(root, exist_ok=True)
    paths = CorpusPaths(
        name=name,
        root=root,
        zip_path=os.path.join(data_dir, f"{name}.zip"),
        raw_path=os.path.join(root, spec.member),
        meta_path=os.path.join(root, "meta.json"),
        splits={s: os.path.join(root, f"{s}.bin") for s in SPLIT_NAMES},
    )

    meta = _load_meta_if_complete(paths, spec)
    if meta is not None:
        _verify_prepared(paths, spec, meta)
        return paths

    # --- acquire ---------------------------------------------------------
    # An existing zip is trusted only if it hashes correctly; a stale or
    # truncated one is re-fetched rather than diagnosed.
    source_url = "(already present)"
    if not (os.path.exists(paths.zip_path) and sha256_file(paths.zip_path) == spec.zip_sha256):
        source_url = _download(spec.urls, paths.zip_path)
        got = sha256_file(paths.zip_path)
        if got != spec.zip_sha256:
            raise RuntimeError(
                f"{name}.zip from {source_url}: SHA-256 mismatch\n"
                f"  expected {spec.zip_sha256}\n  got      {got}"
            )

    # --- extract ---------------------------------------------------------
    if not (
        os.path.exists(paths.raw_path)
        and os.path.getsize(paths.raw_path) == spec.raw_bytes
        and sha256_file(paths.raw_path) == spec.raw_sha256
    ):
        with zipfile.ZipFile(paths.zip_path) as z:
            if spec.member not in z.namelist():
                raise RuntimeError(f"{name}.zip has no member {spec.member!r}: {z.namelist()}")
            with z.open(spec.member) as src, open(paths.raw_path, "wb") as out:
                shutil.copyfileobj(src, out, length=1 << 20)
    _verify(paths.raw_path, spec.raw_sha256, spec.raw_bytes, f"extracted {spec.member}")

    # --- vocabulary + splits ---------------------------------------------
    stoi, itos = _build_vocab(paths.raw_path)
    vocab_size = int(itos.size)
    if vocab_size != spec.vocab_size:
        raise RuntimeError(
            f"{name}: vocab_size is {vocab_size}, expected the published "
            f"{spec.vocab_size}. Do not adjust the expectation to match; find out why."
        )
    split_paths = _write_splits(paths.raw_path, spec, root, stoi)

    meta = {
        "meta_version": META_VERSION,
        "name": name,
        "member": spec.member,
        "source_url": source_url,
        "zip_sha256": spec.zip_sha256,
        "raw_sha256": spec.raw_sha256,
        "raw_bytes": spec.raw_bytes,
        "vocab": [int(v) for v in itos],
        "vocab_size": vocab_size,
        "split_sizes": {s: int(n) for s, n in zip(SPLIT_NAMES, spec.split)},
        "split_sha256": {s: sha256_file(p) for s, p in split_paths.items()},
    }
    with open(paths.meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, sort_keys=True)
        f.write("\n")
    return paths


def _load_meta_if_complete(paths: CorpusPaths, spec: CorpusSpec) -> dict | None:
    """Return meta.json iff every artifact this build needs already exists."""
    if not os.path.exists(paths.meta_path):
        return None
    if not all(os.path.exists(p) for p in paths.splits.values()):
        return None
    try:
        with open(paths.meta_path, encoding="utf-8") as f:
            meta = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    if meta.get("meta_version") != META_VERSION:
        return None
    if "split_sha256" not in meta:
        return None
    return meta


def _verify_prepared(paths: CorpusPaths, spec: CorpusSpec, meta: dict) -> None:
    """Re-verify an existing data/<name>/ without touching the network."""
    if meta["vocab_size"] != spec.vocab_size:
        raise RuntimeError(
            f"{paths.name}: meta.json vocab_size={meta['vocab_size']}, "
            f"expected {spec.vocab_size}"
        )
    if meta["zip_sha256"] != spec.zip_sha256 or meta["raw_sha256"] != spec.raw_sha256:
        raise RuntimeError(f"{paths.name}: meta.json checksums disagree with CORPORA; delete data/{paths.name}")
    for s, n in zip(SPLIT_NAMES, spec.split):
        p = paths.splits[s]
        size = os.path.getsize(p)
        if size != n:
            raise RuntimeError(f"{p}: expected {n} bytes, got {size}")
        got = sha256_file(p)
        if got != meta["split_sha256"][s]:
            raise RuntimeError(
                f"{p}: SHA-256 mismatch\n  expected {meta['split_sha256'][s]}\n"
                f"  got      {got}\n"
                f"  delete {paths.root} and re-run scripts/data/download_corpus.py"
            )


# --------------------------------------------------------------------------
# Corpus access
# --------------------------------------------------------------------------


class Corpus:
    """Memory-mapped access to a prepared corpus.

    Deliberately does *not* download. Acquisition is an explicit step
    (`scripts/data/download_corpus.py`) so that a training run never blocks for
    minutes on a network fetch it did not announce, and so a typo'd `data_dir`
    fails immediately instead of re-downloading 35 MB into the wrong place.
    """

    def __init__(self, name: str, data_dir: str = "data"):
        if name not in CORPORA:
            raise KeyError(f"unknown corpus {name!r}; known: {sorted(CORPORA)}")
        self.name = name
        self.data_dir = data_dir
        self.root = os.path.join(data_dir, name)
        self.meta_path = os.path.join(self.root, "meta.json")
        if not os.path.exists(self.meta_path):
            raise FileNotFoundError(
                f"{self.meta_path} not found. Run:\n"
                f"    python scripts/data/download_corpus.py --corpus {name} --data-dir {data_dir}"
            )
        with open(self.meta_path, encoding="utf-8") as f:
            self.meta: dict = json.load(f)

        spec = CORPORA[name]
        # ensure_corpus refuses to reuse a directory written by an older layout,
        # but training reads through Corpus, not ensure_corpus. Without the same
        # guard here a stale data/ would be silently *read* rather than rebuilt,
        # which is the failure the version number exists to prevent.
        if self.meta.get("meta_version") != META_VERSION:
            raise RuntimeError(
                f"{self.meta_path}: meta_version={self.meta.get('meta_version')!r}, "
                f"this code expects {META_VERSION}. Delete {self.root} and re-run "
                f"scripts/data/download_corpus.py --corpus {name}"
            )
        self.vocab_size: int = int(self.meta["vocab_size"])
        if self.vocab_size != spec.vocab_size:
            raise RuntimeError(
                f"{name}: prepared vocab_size={self.vocab_size}, expected {spec.vocab_size}"
            )
        # The split ratio is part of the experimental protocol, so a directory
        # prepared under a different one must not be trainable on by accident.
        on_disk = tuple(int(self.meta["split_sizes"][s]) for s in SPLIT_NAMES)
        if on_disk != tuple(spec.split):
            raise RuntimeError(
                f"{name}: prepared split sizes {on_disk} != registry {tuple(spec.split)}; "
                f"delete {self.root} and re-run scripts/data/download_corpus.py"
            )
        self.itos: np.ndarray = np.asarray(self.meta["vocab"], dtype=np.uint8)
        self.stoi: np.ndarray = np.full(256, UNUSED_ID, dtype=np.uint8)
        self.stoi[self.itos] = np.arange(self.vocab_size, dtype=np.uint8)
        self.split_sizes: dict[str, int] = {
            s: int(n) for s, n in self.meta["split_sizes"].items()
        }
        self._cache: dict[str, np.memmap] = {}

    def split(self, which: str) -> np.memmap:
        """uint8 memmap of a split, ids already remapped to 0..V-1."""
        if which not in SPLIT_NAMES:
            raise KeyError(f"unknown split {which!r}; known: {SPLIT_NAMES}")
        if which not in self._cache:
            path = os.path.join(self.root, f"{which}.bin")
            expected = self.split_sizes[which]
            size = os.path.getsize(path)
            if size != expected:
                raise RuntimeError(f"{path}: expected {expected} bytes, got {size}")
            self._cache[which] = np.memmap(path, dtype=np.uint8, mode="r")
        return self._cache[which]

    def decode(self, ids: np.ndarray) -> bytes:
        """ids -> original bytes. Only used for eyeballing samples."""
        return self.itos[np.asarray(ids, dtype=np.int64)].tobytes()

    def __repr__(self) -> str:
        sizes = ", ".join(f"{s}={self.split_sizes[s]:,}" for s in SPLIT_NAMES)
        return f"Corpus({self.name!r}, V={self.vocab_size}, {sizes})"


# --------------------------------------------------------------------------
# Samplers
# --------------------------------------------------------------------------

_PIN_AVAILABLE: bool | None = None


def _pin_available() -> bool:
    """Pinned host memory needs a CUDA context; cache the check.

    Pinning matters because the trainer's static-buffer copy is
    `copy_(..., non_blocking=True)`, which only overlaps with compute if the
    source is page-locked.
    """
    global _PIN_AVAILABLE
    if _PIN_AVAILABLE is None:
        _PIN_AVAILABLE = bool(torch.cuda.is_available())
    return _PIN_AVAILABLE


def _to_int64_cpu(arr: np.ndarray) -> Tensor:
    """uint8 [B,L] -> int64 [B,L], page-locked when CUDA is present.

    A fresh pinned allocation per call is intentional: PyTorch's caching host
    allocator reuses blocks and only hands one back once the outstanding async
    copies that referenced it have completed. Recycling a single buffer
    ourselves would race with `non_blocking=True` copies still in flight.
    """
    out = torch.empty(arr.shape, dtype=torch.int64, pin_memory=_pin_available())
    out.copy_(torch.from_numpy(np.ascontiguousarray(arr)))
    return out


def _mix64(seed: int, step: int) -> int:
    """SplitMix64-style mix of (seed, step) into one 64-bit generator seed.

    A plain `seed + step` would make consecutive steps use adjacent seeds; for
    most RNGs that is fine, but avalanching costs nothing and removes the
    question. Returned as a signed 63-bit value because `Generator.manual_seed`
    rejects anything wider.
    """
    z = (seed * 0x9E3779B97F4A7C15 + step * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
    z = z ^ (z >> 31)
    return z & 0x7FFFFFFFFFFFFFFF


def _gather_windows(data: np.memmap, offsets: np.ndarray, seq_len: int) -> np.ndarray:
    """[n, seq_len+1] uint8 block of windows starting at `offsets`."""
    idx = offsets[:, None] + np.arange(seq_len + 1, dtype=np.int64)[None, :]
    return np.asarray(data)[idx]


class RandomWindowSampler:
    """Training sampler. Deterministic given (seed, step).

    `batch(k)` depends on nothing but `(seed, k)` and the split length, so a run
    resumed from a step-k checkpoint sees exactly the data an uninterrupted run
    would have seen. There is no hidden stream position to checkpoint, which is
    one fewer thing that can silently desynchronise (R6).
    """

    def __init__(self, data: np.memmap, batch_size: int, seq_len: int, seed: int):
        self.data = data
        self.batch_size = int(batch_size)
        self.seq_len = int(seq_len)
        self.seed = int(seed)
        # A window reads data[o : o + seq_len + 1] -- seq_len inputs plus the one
        # shifted target character. It fits iff o + seq_len <= len(data) - 1, so
        # the last legal start is len(data) - seq_len - 1 and there are
        # len(data) - seq_len legal starts. Using one fewer would make the final
        # character of the split unreachable as a target.
        self.n_offsets = len(data) - seq_len
        if self.n_offsets <= 0:
            raise ValueError(
                f"split of {len(data)} chars is too short for seq_len={seq_len}"
            )

    def offsets(self, step: int) -> np.ndarray:
        """The exact start positions batch(step) will read. Exposed for tests."""
        g = torch.Generator()
        g.manual_seed(_mix64(self.seed, step))
        # `high` is exclusive, so the largest start drawn is n_offsets - 1 and the
        # last character touched is (n_offsets - 1) + seq_len = len(data) - 1,
        # i.e. exactly the final character of the split.
        return torch.randint(
            low=0, high=self.n_offsets, size=(self.batch_size,), generator=g, dtype=torch.int64
        ).numpy()

    def batch(self, step: int) -> tuple[Tensor, Tensor]:
        """Returns (inputs [B,L] int64, targets [B,L] int64) on CPU, pinned."""
        block = _gather_windows(self.data, self.offsets(step), self.seq_len)
        return _to_int64_cpu(block[:, :-1]), _to_int64_cpu(block[:, 1:])


# -- evaluation protocols ---------------------------------------------------
#
# Both drop whatever does not fill a full [batch_size, seq_len] block, because
# CUDA-graph capture requires static shapes (bottleneck B4) and because a ragged
# final batch would make the two protocols score different character sets. The
# two helpers below state exactly how many characters survive that drop, so the
# accounting in evaluate.py and in the report is checkable rather than asserted.


def windowed_eval_char_count(n: int, batch_size: int, seq_len: int) -> int:
    """Characters scored by `windowed_eval_batches` over a split of `n` bytes.

    Note the `batch_size` dependence: dropping the trailing partial batch drops
    `n_windows % batch_size` windows, so *which* characters are scored depends on
    `batch_size`. Spec §11 asserts the fresh protocol is "independent of batch
    size in the characters they score"; that holds only when
    `n_windows % batch_size == 0` for every batch size being compared. See the
    docstring of `windowed_eval_batches`.
    """
    n_windows = max(n - 1, 0) // seq_len
    return (n_windows // batch_size) * batch_size * seq_len


def contiguous_eval_char_count(n: int, batch_size: int, seq_len: int) -> int:
    """Characters scored by `contiguous_eval_batches` over a split of `n` bytes."""
    stream_len = max(n - 1, 0) // batch_size
    return (stream_len // seq_len) * batch_size * seq_len


def windowed_eval_batches(
    data: np.memmap, batch_size: int, seq_len: int
) -> Iterator[tuple[Tensor, Tensor]]:
    """Protocol A - fresh state.

    Non-overlapping windows in corpus order; the caller zeroes membrane state at
    every window start. Window w scores target positions
    `w*seq_len + 1 .. w*seq_len + seq_len`, so the windows tile the split with no
    overlap and no gap. The trailing partial batch is dropped; the number of
    characters actually scored is `windowed_eval_char_count(...)`.

    This protocol matches the training distribution (the model never sees state
    older than seq_len during training either), which is why it is reported
    alongside the carried-state number rather than instead of it.

    **Batch-size caveat, stated here because it contradicts a spec claim.**
    Spec §11 says both protocols are "independent of batch size in the characters
    they score". They are not, and cannot be while the trailing partial batch is
    dropped (spec §4.3) and shapes stay static: `n_windows % batch_size` windows
    are discarded, so B=1 scores strictly more characters than B=16 on the same
    split. Comparing bpc across batch sizes is therefore only meaningful when
    `((len(data) - 1) // seq_len) % batch_size` is zero for every batch size in
    the comparison, or when `max_windows` is fixed to a common multiple. The
    window *order* and the per-window content are batch-size independent; only
    the truncation point moves.
    """
    n = len(data)
    n_windows = max(n - 1, 0) // seq_len
    n_batches = n_windows // batch_size
    for b in range(n_batches):
        starts = (np.arange(batch_size, dtype=np.int64) + b * batch_size) * seq_len
        block = _gather_windows(data, starts, seq_len)
        yield _to_int64_cpu(block[:, :-1]), _to_int64_cpu(block[:, 1:])


def contiguous_eval_batches(
    data: np.memmap, batch_size: int, seq_len: int
) -> Iterator[tuple[Tensor, Tensor]]:
    """Protocol B - carried state.

    The split is cut into `batch_size` contiguous streams of equal length; window
    i of every stream is yielded together, so the caller can carry (detached)
    membrane state from one yield to the next and each stream is scored as one
    long unbroken sequence. This is the number comparable with published enwik8
    results.

    Stream b covers target positions `b*S + 1 .. b*S + n_steps*seq_len` where
    `S = (n-1) // batch_size`. Positions are therefore scored exactly once, never
    twice; what is dropped is `(n-1) % batch_size` characters at the very end of
    the split plus `S % seq_len` at the end of each stream -- at B=128, L=256 on
    the 5 MB val split that is under 0.7% of it. `contiguous_eval_char_count`
    returns the exact figure.
    """
    n = len(data)
    stream_len = max(n - 1, 0) // batch_size
    n_steps = stream_len // seq_len
    bases = np.arange(batch_size, dtype=np.int64) * stream_len
    for i in range(n_steps):
        starts = bases + i * seq_len
        block = _gather_windows(data, starts, seq_len)
        yield _to_int64_cpu(block[:, :-1]), _to_int64_cpu(block[:, 1:])
