"""Gate A5: the data pipeline is verifiable, reproducible and complete.

Spec: docs/reports/02a_phase2_spec.md §12, row `test_data.py` --
"Checksums verified; vocab_size 205/27; splits sum to 100 MB; sampler
reproducible from (seed, step); targets are inputs shifted by one".

Two things are tested here beyond that row, because they are the failure modes
that would corrupt every number the project reports without ever raising:

  * **Evaluation coverage.** Both protocols must score exactly the characters
    they claim to, each exactly once. A protocol that quietly double-counts or
    skips a window still produces a plausible bpc. The coverage tests
    reconstruct the scored positions from the yielded tensors on a synthetic
    corpus whose byte values *are* their positions, so the check is on real
    yielded data rather than on a re-derivation of the same arithmetic.

  * **Resume equivalence.** `batch(k)` must be identical whether it is the first
    call or the thousandth. This is what makes R6 (resumable checkpoints)
    meaningful; if it fails, a resumed run silently trains on a different data
    order than the run it claims to continue.

Tests that need the corpus are marked `data` and skip cleanly when `data/` has
not been populated (it is gitignored, so a fresh clone has none).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

import numpy as np
import pytest
import torch

from snn import config as cfgmod
from snn.config import Config, config_from_args, config_hash, seed_everything
from snn.data import (
    CORPORA,
    SPLIT_NAMES,
    UNUSED_ID,
    Corpus,
    RandomWindowSampler,
    contiguous_eval_batches,
    contiguous_eval_char_count,
    ensure_corpus,
    sha256_file,
    windowed_eval_batches,
    windowed_eval_char_count,
)

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(REPO, "data")
CORPUS_NAMES = sorted(CORPORA)

#: SHA-256 of the prepared split files, pinned as literals.
#:
#: Without these, the only split check is `sha256(train.bin) == meta["..."]`,
#: and meta.json was written by the same code that wrote train.bin -- so the
#: assertion holds however wrong the split boundaries or the id remap are. It
#: detects bit-rot and nothing else. These values were recomputed independently
#: on 2026-08-01 from `stoi(raw[lo:hi])`, where `raw` hashes to the pinned
#: `CorpusSpec.raw_sha256`, so they close the loop: split content is now tied to
#: the published corpus rather than to our own record of it.
SPLIT_SHA256: dict[str, dict[str, str]] = {
    "enwik8": {
        "train": "cc72e954362306215f255f54e59d5e1f0bc5bc39e1507502f4dee7e9aacab62d",
        "val": "8b81a1d941d538327f4f38ad0d91605e8ba05c406fdf91fe8cb033efde069173",
        "test": "67fc6dcb3d85434627800084132955f439275558c8240da20a3c13859394e17b",
    },
    "text8": {
        "train": "cac5621c7920f3fea412b8b8e791cd26c0e334737ae587a21597ecdffe1afcff",
        "val": "526d234aa336fc0c89a79dc2a424353be363a113687b19d4b443bc47db412ac0",
        "test": "bd32ed53f5f72cc71c242923b4c5b11a8d2ab635e8b8029cfe062ed77dfb8170",
    },
}


def _prepared(name: str) -> bool:
    return os.path.exists(os.path.join(DATA_DIR, name, "meta.json"))


def _require(name: str) -> None:
    if not _prepared(name):
        pytest.skip(
            f"{name} not prepared; run "
            f"python scripts/data/download_corpus.py --corpus {name}"
        )


@pytest.fixture(scope="module")
def enwik8() -> Corpus:
    _require("enwik8")
    return Corpus("enwik8", DATA_DIR)


# --------------------------------------------------------------------------
# The pinned registry itself
# --------------------------------------------------------------------------


@pytest.mark.parametrize("name", CORPUS_NAMES)
def test_registry_is_internally_consistent(name: str) -> None:
    spec = CORPORA[name]
    assert sum(spec.split) == spec.raw_bytes == 100_000_000
    assert spec.split == (90_000_000, 5_000_000, 5_000_000), "standard 90/5/5 MB split"
    for sha in (spec.zip_sha256, spec.raw_sha256):
        assert len(sha) == 64 and all(c in "0123456789abcdef" for c in sha), (
            "checksums must be pinned literals, not placeholders"
        )
    assert spec.urls, "at least one mirror"


def test_published_vocab_sizes_are_pinned() -> None:
    # If reality ever disagrees, the acquisition code raises rather than these
    # numbers being adjusted downwards to make a run start.
    assert CORPORA["enwik8"].vocab_size == 205
    assert CORPORA["text8"].vocab_size == 27


# --------------------------------------------------------------------------
# Acquisition: checksums, splits, vocabulary
# --------------------------------------------------------------------------


@pytest.mark.data
@pytest.mark.parametrize("name", CORPUS_NAMES)
def test_meta_matches_pinned_checksums(name: str) -> None:
    _require(name)
    spec = CORPORA[name]
    with open(os.path.join(DATA_DIR, name, "meta.json"), encoding="utf-8") as f:
        meta = json.load(f)
    assert meta["zip_sha256"] == spec.zip_sha256
    assert meta["raw_sha256"] == spec.raw_sha256
    assert meta["vocab_size"] == spec.vocab_size
    assert meta["raw_bytes"] == 100_000_000
    assert [meta["split_sizes"][s] for s in SPLIT_NAMES] == list(spec.split)
    assert sum(meta["split_sizes"].values()) == 100_000_000, "splits sum to 100 MB"
    assert len(meta["vocab"]) == spec.vocab_size
    assert meta["vocab"] == sorted(meta["vocab"]), "vocab is the sorted byte values"


@pytest.mark.data
@pytest.mark.parametrize("name", CORPUS_NAMES)
def test_split_files_hash_to_recorded_values(name: str) -> None:
    """The real checksum gate: recompute, and check against a pinned literal.

    Checking only against meta.json would be self-referential -- the same run
    wrote both -- so the pinned SPLIT_SHA256 is what actually gates the content.
    """
    _require(name)
    root = os.path.join(DATA_DIR, name)
    with open(os.path.join(root, "meta.json"), encoding="utf-8") as f:
        meta = json.load(f)
    for s in SPLIT_NAMES:
        path = os.path.join(root, f"{s}.bin")
        digest = sha256_file(path)
        assert os.path.getsize(path) == meta["split_sizes"][s]
        assert digest == meta["split_sha256"][s], "meta.json disagrees with the file"
        assert digest == SPLIT_SHA256[name][s], (
            f"{name}/{s}.bin content changed; if the split protocol was changed "
            f"deliberately, update SPLIT_SHA256 and say so in the report"
        )


@pytest.mark.data
@pytest.mark.parametrize("name", CORPUS_NAMES)
def test_raw_extract_hashes_to_pinned_value(name: str) -> None:
    _require(name)
    spec = CORPORA[name]
    raw = os.path.join(DATA_DIR, name, spec.member)
    if not os.path.exists(raw):
        pytest.skip("raw extract removed; split checksums still cover the content")
    assert os.path.getsize(raw) == spec.raw_bytes
    assert sha256_file(raw) == spec.raw_sha256


@pytest.mark.data
@pytest.mark.parametrize("name", CORPUS_NAMES)
def test_vocab_size_and_id_range(name: str) -> None:
    _require(name)
    corpus = Corpus(name, DATA_DIR)
    assert corpus.vocab_size == CORPORA[name].vocab_size
    assert corpus.itos.shape == (corpus.vocab_size,)
    assert corpus.stoi.shape == (256,)
    # stoi/itos are inverse on the used bytes and sentinel elsewhere.
    assert np.array_equal(corpus.stoi[corpus.itos], np.arange(corpus.vocab_size, dtype=np.uint8))
    unused = np.setdiff1d(np.arange(256, dtype=np.uint8), corpus.itos)
    assert np.all(corpus.stoi[unused] == UNUSED_ID)
    # Every stored id is a legal embedding row, in every split.
    for s in SPLIT_NAMES:
        arr = corpus.split(s)
        sample = np.asarray(arr[:: max(len(arr) // 1_000_000, 1)])
        assert int(sample.max()) < corpus.vocab_size
        assert int(sample.min()) >= 0


@pytest.mark.data
@pytest.mark.parametrize("name", CORPUS_NAMES)
def test_splits_are_the_remapped_corpus_in_order(name: str) -> None:
    """Splits are byte-ranges of the raw file, in order, run through stoi.

    Checked over all 100 MB, not spot-checked at the boundaries: a remap that is
    wrong only for a rare byte value (say, one of enwik8's 80-odd high bytes)
    never appears in a 4 KB window at a split edge, and would silently shift the
    vocabulary the model is trained on. A full comparison costs about a second
    against memmapped data, which is not a reason to test less.
    """
    _require(name)
    spec = CORPORA[name]
    raw_path = os.path.join(DATA_DIR, name, spec.member)
    if not os.path.exists(raw_path):
        pytest.skip("raw extract removed")
    corpus = Corpus(name, DATA_DIR)
    raw = np.memmap(raw_path, dtype=np.uint8, mode="r")
    starts = np.cumsum((0,) + spec.split)
    chunk = 1 << 24
    for s, start, stop in zip(SPLIT_NAMES, starts[:-1], starts[1:]):
        arr = corpus.split(s)
        assert len(arr) == stop - start
        for lo in range(int(start), int(stop), chunk):
            hi = min(lo + chunk, int(stop))
            expected = corpus.stoi[np.asarray(raw[lo:hi])]
            got = np.asarray(arr[lo - start : hi - start])
            assert np.array_equal(got, expected), f"{name}/{s} mismatch at [{lo}:{hi}]"
    # Every symbol of the published vocabulary must actually occur in the
    # remapped data, or the embedding has dead rows the reported parameter count
    # still charges for.
    seen = np.zeros(corpus.vocab_size, dtype=bool)
    for s in SPLIT_NAMES:
        arr = np.asarray(corpus.split(s))
        seen[np.unique(arr)] = True
    assert seen.all(), f"{name}: ids {np.flatnonzero(~seen).tolist()} never occur"


@pytest.mark.data
def test_ensure_corpus_is_idempotent() -> None:
    """A second call verifies and returns; it must not rewrite the splits."""
    _require("enwik8")
    meta_path = os.path.join(DATA_DIR, "enwik8", "meta.json")
    train_path = os.path.join(DATA_DIR, "enwik8", "train.bin")
    before = (os.path.getmtime(meta_path), os.path.getmtime(train_path))
    paths = ensure_corpus("enwik8", DATA_DIR)
    after = (os.path.getmtime(meta_path), os.path.getmtime(train_path))
    assert before == after, "idempotent re-run must not rebuild anything"
    assert os.path.exists(paths.train) and os.path.exists(paths.val) and os.path.exists(paths.test)


def test_corpus_refuses_to_guess() -> None:
    with pytest.raises(KeyError):
        Corpus("ptb", DATA_DIR)
    with pytest.raises(FileNotFoundError):
        Corpus("enwik8", os.path.join(REPO, "does-not-exist"))


@pytest.mark.data
def test_corpus_rejects_a_stale_data_directory(tmp_path) -> None:
    """META_VERSION only protects anything if the *reader* honours it.

    ensure_corpus rebuilds on a version mismatch, but training reads through
    Corpus, which used to accept any meta.json it could parse. The same applies
    to a directory prepared under a different split ratio: silently training on
    an 80/10/10 enwik8 and reporting it as the standard 90/5/5 is exactly the
    kind of error that never raises.
    """
    _require("enwik8")
    root = tmp_path / "enwik8"
    root.mkdir()
    with open(os.path.join(DATA_DIR, "enwik8", "meta.json"), encoding="utf-8") as f:
        good = json.load(f)

    stale = dict(good, meta_version=good["meta_version"] + 1)
    (root / "meta.json").write_text(json.dumps(stale), encoding="utf-8")
    with pytest.raises(RuntimeError, match="meta_version"):
        Corpus("enwik8", str(tmp_path))

    wrong_split = dict(good, split_sizes={"train": 80_000_000, "val": 10_000_000,
                                          "test": 10_000_000})
    (root / "meta.json").write_text(json.dumps(wrong_split), encoding="utf-8")
    with pytest.raises(RuntimeError, match="split sizes"):
        Corpus("enwik8", str(tmp_path))


# --------------------------------------------------------------------------
# Training sampler
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def toy() -> np.ndarray:
    """250 bytes whose value equals their index -- so a scored position is
    recoverable from the yielded tensor rather than re-derived from the same
    arithmetic the implementation used."""
    return np.arange(250, dtype=np.uint8)


def test_sampler_shapes_and_dtypes(toy: np.ndarray) -> None:
    s = RandomWindowSampler(toy, batch_size=4, seq_len=8, seed=0)
    x, y = s.batch(0)
    assert x.shape == (4, 8) and y.shape == (4, 8)
    assert x.dtype is torch.int64 and y.dtype is torch.int64
    assert x.device.type == "cpu"
    if torch.cuda.is_available():
        assert x.is_pinned() and y.is_pinned(), "pinned so non_blocking H2D actually overlaps"


def test_sampler_targets_are_inputs_shifted_by_one(toy: np.ndarray) -> None:
    s = RandomWindowSampler(toy, batch_size=6, seq_len=9, seed=3)
    x, y = s.batch(11)
    assert torch.equal(y[:, :-1], x[:, 1:])
    # On the toy corpus value == position, so the shift is checkable absolutely.
    off = torch.from_numpy(s.offsets(11))
    assert torch.equal(x[:, 0], off)
    assert torch.equal(y[:, 0], off + 1)
    assert torch.equal(y[:, -1], off + 9)


def test_sampler_reproducible_from_seed_and_step(toy: np.ndarray) -> None:
    a = RandomWindowSampler(toy, 4, 8, seed=7)
    b = RandomWindowSampler(toy, 4, 8, seed=7)
    for k in (0, 1, 2, 1000, 999_999):
        xa, ya = a.batch(k)
        xb, yb = b.batch(k)
        assert torch.equal(xa, xb) and torch.equal(ya, yb)


def test_sampler_resume_matches_uninterrupted_run(toy: np.ndarray) -> None:
    """The R6 property: a run resumed at step k draws the batch an uninterrupted
    run would have drawn at step k."""
    uninterrupted = RandomWindowSampler(toy, 5, 8, seed=11)
    seen = [uninterrupted.batch(k)[0] for k in range(20)]
    resumed = RandomWindowSampler(toy, 5, 8, seed=11)
    for k in (13, 17, 19):
        assert torch.equal(resumed.batch(k)[0], seen[k])
    # ...and in the other direction: drawing out of order changes nothing.
    fresh = RandomWindowSampler(toy, 5, 8, seed=11)
    assert torch.equal(fresh.batch(19)[0], seen[19])
    assert torch.equal(fresh.batch(0)[0], seen[0])


def test_sampler_different_seeds_differ(toy: np.ndarray) -> None:
    a = RandomWindowSampler(toy, 16, 8, seed=0)
    b = RandomWindowSampler(toy, 16, 8, seed=1)
    assert not np.array_equal(a.offsets(0), b.offsets(0))


def test_sampler_stays_in_bounds(toy: np.ndarray) -> None:
    s = RandomWindowSampler(toy, 32, 8, seed=5)
    for k in range(50):
        off = s.offsets(k)
        assert off.min() >= 0
        assert off.max() + 8 + 1 <= len(toy)


def test_sampler_can_reach_every_legal_window(toy: np.ndarray) -> None:
    """The bound must be tight in *both* directions.

    `test_sampler_stays_in_bounds` only catches a bound that is too loose. A
    bound that is one too tight -- `len(data) - seq_len - 1` instead of
    `len(data) - seq_len` -- reads perfectly well, passes every shape and dtype
    check, and quietly makes the final character of the split unreachable as a
    target. That is exactly the kind of off-by-one that no other test here sees.
    """
    L = 8
    s = RandomWindowSampler(toy, 64, L, seed=0)
    assert s.n_offsets == len(toy) - L
    last = s.n_offsets - 1
    # The largest legal start really does fit a full seq_len+1 window...
    assert last + L == len(toy) - 1
    # ...and it is drawable, so the last character is a target for some batch.
    # 64 x 200 = 12800 draws from 242 offsets: missing the top one by chance has
    # probability ~1e-23, so a failure here means the bound moved, not bad luck.
    seen = {int(o) for k in range(200) for o in s.offsets(k)}
    assert max(seen) == last, f"largest reachable offset {max(seen)}, expected {last}"
    assert seen == set(range(s.n_offsets)), "every legal window must be drawable"


def test_sampler_accepts_a_split_of_exactly_one_window() -> None:
    """len == seq_len + 1 admits exactly one window; it must not be rejected."""
    s = RandomWindowSampler(np.arange(17, dtype=np.uint8), 2, 16, seed=0)
    assert s.n_offsets == 1
    x, y = s.batch(0)
    assert x.shape == (2, 16) and y.shape == (2, 16)
    assert torch.equal(x[0], torch.arange(16))
    assert torch.equal(y[0], torch.arange(1, 17))


def test_sampler_rejects_too_short_split() -> None:
    with pytest.raises(ValueError):
        RandomWindowSampler(np.arange(9, dtype=np.uint8), 2, 16, seed=0)
    # len == seq_len leaves no room for the shifted target.
    with pytest.raises(ValueError):
        RandomWindowSampler(np.arange(16, dtype=np.uint8), 2, 16, seed=0)


@pytest.mark.data
def test_sampler_on_real_split(enwik8: Corpus) -> None:
    train = enwik8.split("train")
    s = RandomWindowSampler(train, batch_size=8, seq_len=64, seed=0)
    x, y = s.batch(1234)
    assert x.shape == (8, 64)
    assert torch.equal(y[:, :-1], x[:, 1:])
    assert int(x.max()) < enwik8.vocab_size and int(x.min()) >= 0
    x2, _ = RandomWindowSampler(train, 8, 64, seed=0).batch(1234)
    assert torch.equal(x, x2)


# --------------------------------------------------------------------------
# Evaluation protocols: coverage
# --------------------------------------------------------------------------


def _scored_positions(batches, toy_len: int) -> list[int]:
    """Positions scored, recovered from the target values themselves."""
    out: list[int] = []
    for x, y in batches:
        assert torch.equal(y[:, :-1], x[:, 1:]), "targets are inputs shifted by one"
        vals = y.reshape(-1).tolist()
        assert all(0 <= v < toy_len for v in vals)
        out.extend(vals)
    return out


@pytest.mark.parametrize("batch_size,seq_len", [(3, 7), (4, 8), (1, 16), (5, 3)])
def test_windowed_eval_scores_each_character_once(
    toy: np.ndarray, batch_size: int, seq_len: int
) -> None:
    scored = _scored_positions(windowed_eval_batches(toy, batch_size, seq_len), len(toy))
    n = windowed_eval_char_count(len(toy), batch_size, seq_len)
    assert len(scored) == n, "yielded count must match the declared accounting"
    assert len(set(scored)) == n, "no character scored twice"
    # Protocol A tiles the split from the start with non-overlapping windows.
    assert sorted(scored) == list(range(1, n + 1))


@pytest.mark.parametrize("batch_size,seq_len", [(3, 7), (4, 8), (1, 16), (5, 3)])
def test_contiguous_eval_scores_each_character_once(
    toy: np.ndarray, batch_size: int, seq_len: int
) -> None:
    scored = _scored_positions(contiguous_eval_batches(toy, batch_size, seq_len), len(toy))
    n = contiguous_eval_char_count(len(toy), batch_size, seq_len)
    assert len(scored) == n
    assert len(set(scored)) == n, "no character scored twice"
    # Protocol B's streams are contiguous: stream b starts at b * stream_len.
    stream_len = (len(toy) - 1) // batch_size
    n_steps = stream_len // seq_len
    expected = {
        b * stream_len + 1 + i
        for b in range(batch_size)
        for i in range(n_steps * seq_len)
    }
    assert set(scored) == expected


def test_contiguous_eval_yields_streams_in_order(toy: np.ndarray) -> None:
    """Row b of successive yields must continue row b, or carried membrane state
    is being carried across a discontinuity -- which would silently corrupt
    protocol B without changing any shape."""
    batches = list(contiguous_eval_batches(toy, 3, 7))
    for prev, nxt in zip(batches, batches[1:]):
        assert torch.equal(prev[0][:, -1] + 1, nxt[0][:, 0])


@pytest.mark.data
@pytest.mark.parametrize("protocol", ["fresh", "carried"])
def test_eval_coverage_on_real_split(enwik8: Corpus, protocol: str) -> None:
    """The characters actually yielded equal the declared count, and the drop of
    the trailing partial batch costs under 1% of the split."""
    val = enwik8.split("val")
    n, B, L = len(val), 128, 256
    if protocol == "fresh":
        it, expect = windowed_eval_batches(val, B, L), windowed_eval_char_count(n, B, L)
    else:
        it, expect = contiguous_eval_batches(val, B, L), contiguous_eval_char_count(n, B, L)
    total = 0
    n_batches = 0
    for x, y in it:
        assert x.shape == (B, L) and y.shape == (B, L)
        total += y.numel()
        n_batches += 1
    assert total == expect
    assert total == n_batches * B * L
    assert n - 1 - total < B * L, "at most one batch's worth may be dropped"
    assert total / (n - 1) > 0.99


def test_windowed_eval_char_set_depends_on_batch_size(toy: np.ndarray) -> None:
    """Pins a real deviation from spec §11, so evaluate.py cannot assume otherwise.

    §11 says both protocols are "independent of batch size in the characters they
    score" and that a test asserts `bpc` invariance to `batch_size` for "fresh".
    That cannot hold while the trailing partial batch is dropped (§4.3) and
    shapes stay static: dropping `n_windows % batch_size` windows moves the
    truncation point. The precondition under which the §11 claim *is* true is
    asserted below, and it is the one evaluate.py must respect.
    """
    L = 12  # gives n_windows = 20 on the 250-byte toy: composite, so the
            # "b divides n_windows" branch below has something to check.
    n_windows = (len(toy) - 1) // L
    assert n_windows == 20

    def scored(batch_size: int) -> set[int]:
        out: set[int] = set()
        for _, y in windowed_eval_batches(toy, batch_size, L):
            out |= set(y.reshape(-1).tolist())
        return out

    # Not invariant in general: B=16 does not divide n_windows here.
    assert n_windows % 16 != 0
    assert scored(1) != scored(16)
    assert len(scored(1)) > len(scored(16))

    # Invariant exactly when every batch size divides the window count. The
    # window contents themselves never depend on batch_size -- only where the
    # sequence is cut off -- so the smaller set is always a prefix of the larger.
    for b in (1, 2, 4, 8, 16):
        assert scored(b) <= scored(1)
        assert scored(b) == set(range(1, windowed_eval_char_count(len(toy), b, L) + 1))
    divisors = [b for b in (1, 2, 3, 4, 6, 8, 12) if n_windows % b == 0]
    assert len(divisors) >= 2
    ref = scored(divisors[0])
    for b in divisors[1:]:
        assert scored(b) == ref, "invariance must hold when b divides n_windows"


@pytest.mark.data
def test_eval_protocols_score_the_same_characters_at_matched_shape(enwik8: Corpus) -> None:
    """Not required by the spec, but recorded: the two protocols cover almost the
    same characters, so a bpc difference between them is a state-carrying effect
    and not a difference in what was measured."""
    val = enwik8.split("val")
    n, B, L = len(val), 128, 256
    a = windowed_eval_char_count(n, B, L)
    b = contiguous_eval_char_count(n, B, L)
    assert abs(a - b) <= B * L


# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------


def test_cublas_workspace_is_set_at_import_time() -> None:
    """If this ever regresses, determinism degrades to a warning nobody reads."""
    assert os.environ.get("CUBLAS_WORKSPACE_CONFIG") == ":4096:8"


def _run_clean(body: str) -> subprocess.CompletedProcess:
    """Run `body` in a fresh interpreter with src/ on the path.

    json.dumps is used to embed the path because the repository path contains an
    apostrophe; a naive raw-string literal is a syntax error.
    """
    src = json.dumps(os.path.join(REPO, "src"))
    code = f"import sys, os; sys.path.insert(0, {src}); " + body
    return subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, cwd=REPO, timeout=600
    )


def test_cublas_workspace_is_set_before_torch_is_imported() -> None:
    """Import order matters, so assert it in a clean interpreter rather than
    trusting this process, which may have imported torch for other reasons."""
    out = _run_clean(
        "assert 'torch' not in sys.modules; import snn.config; "
        "assert 'torch' in sys.modules; "
        "print(os.environ['CUBLAS_WORKSPACE_CONFIG'])"
    )
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == ":4096:8"


def test_importing_the_package_does_not_touch_torch_or_cuda() -> None:
    """`import snn` must stay free of torch: launch.py and the test collector
    import it in processes that may never use a GPU, and an early CUDA context
    would render the CUBLAS_WORKSPACE_CONFIG setting above inert."""
    out = _run_clean("import snn; print('torch' in sys.modules)")
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "False"


def test_config_defaults_are_the_frozen_baseline() -> None:
    c = Config()
    assert (c.corpus, c.d_model, c.n_layers) == ("enwik8", 512, 2)
    assert (c.batch_size, c.seq_len) == (128, 256)
    assert (c.beta, c.threshold, c.reset) == (0.5, 1.0, "hard")
    assert (c.surrogate, c.surrogate_alpha, c.t_steps) == ("atan", 2.0, 1)
    assert c.dtype == "fp32"


def test_config_hash_is_stable_and_sensitive() -> None:
    a, b = Config(), Config()
    assert config_hash(a) == config_hash(b)
    assert len(config_hash(a)) == 12
    b.seed = 1
    assert config_hash(a) != config_hash(b)


def test_config_from_args_round_trip() -> None:
    c = config_from_args(["--seq-len", "128", "--beta", "0.9", "--no-fused", "--reset", "soft"])
    assert (c.seq_len, c.beta, c.fused, c.reset) == (128, 0.9, False, "soft")
    # underscore spelling is accepted too, so shell scripts cannot fail silently
    assert config_from_args(["--seq_len", "64"]).seq_len == 64
    assert config_from_args([]).fused is True


def test_config_rejects_typos() -> None:
    with pytest.raises(ValueError):
        Config(reset="reset-to-zero")
    with pytest.raises(ValueError):
        Config(dtype="float32")
    with pytest.raises(ValueError):
        Config(seq_len=0)


def test_config_reset_codes_match_spec() -> None:
    assert cfgmod.RESET_CODES == {"hard": 0, "soft": 1, "detached": 2, "none": 3}
    assert Config(reset="detached").reset_code == 2


def test_seed_everything_is_reproducible() -> None:
    # Restore the *previous* global state, not a hard-coded "off". Forcing
    # use_deterministic_algorithms(False) on the way out would silently disable
    # determinism for every test module collected after this one -- including
    # test_determinism.py, whose whole job is to detect that.
    prev = (
        torch.are_deterministic_algorithms_enabled(),
        torch.is_deterministic_algorithms_warn_only_enabled(),
        torch.backends.cudnn.deterministic,
        torch.backends.cudnn.benchmark,
    )
    try:
        seed_everything(0, deterministic=True)
        assert torch.are_deterministic_algorithms_enabled()
        assert torch.is_deterministic_algorithms_warn_only_enabled(), "spec §3 requires warn_only"
        assert torch.backends.cudnn.deterministic and not torch.backends.cudnn.benchmark
        a = torch.randn(8)
        na = np.random.rand(8)
        seed_everything(0, deterministic=True)
        assert torch.equal(a, torch.randn(8))
        assert np.array_equal(na, np.random.rand(8))
        seed_everything(1, deterministic=True)
        assert not torch.equal(a, torch.randn(8))

        # deterministic=False must undo all of it, or --no-deterministic reports
        # a configuration that never actually ran.
        seed_everything(0, deterministic=False)
        assert not torch.are_deterministic_algorithms_enabled()
        assert not torch.backends.cudnn.deterministic
        assert torch.backends.cudnn.benchmark
    finally:
        torch.use_deterministic_algorithms(prev[0], warn_only=prev[1])
        torch.backends.cudnn.deterministic = prev[2]
        torch.backends.cudnn.benchmark = prev[3]
