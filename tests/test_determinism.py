"""Determinism, resumability, and protocol-invariance of the evaluator.

Three claims are checked here, and each of them is load-bearing for a different
part of the programme:

* **Same seed, same trajectory.**  R8 says every ablation conclusion depends on
  knowing the seed-to-seed noise floor.  A noise floor is only measurable if a
  seed is otherwise fully determining, so a run that varies at fixed seed makes
  the whole Phase-4 statistics plan meaningless.  Different seeds must also
  actually differ -- a seeding bug that ignores its argument would pass the
  first half of this and fail the science.

* **Resume is exact, not approximate.**  R6/R7: multi-hour runs get killed, and
  the mitigation is only a mitigation if
  ``train(20) == train(10) + save + load + train(10)`` *exactly*.  Anything
  weaker means a checkpointed run is a different experiment from an
  uninterrupted one, and the two cannot be pooled.

* **The "fresh" protocol's bpc does not depend on batch size.**  Batch size is
  a systems knob (3.10 says it is the cheap axis to spend), so it must not be
  able to move a reported number.  With ``max_windows`` a multiple of the batch
  sizes compared, the same characters are scored either way and the only
  residual is GPU reduction order.
"""

from __future__ import annotations

import dataclasses
import gc
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO / "src") not in sys.path:
    sys.path.insert(0, str(_REPO / "src"))

from snn.config import Config  # noqa: E402
from snn.data import Corpus  # noqa: E402
from snn.evaluate import evaluate  # noqa: E402
from snn.model import build_model  # noqa: E402
from snn.train import Trainer  # noqa: E402

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="needs a GPU"
)

_CORPUS = os.environ.get("SNN_TEST_CORPUS", "enwik8")
_VOCAB = {"enwik8": 205, "text8": 27}


# ----------------------------------------------------------------------
# corpus fixture (see tests/test_graph_equivalence.py for the same helper --
# duplicated rather than shared so neither gate depends on the other's file)
# ----------------------------------------------------------------------

def _real_data_dir(name: str) -> str | None:
    root = os.environ.get("SNN_TEST_DATA_DIR") or str(_REPO / "data")
    p = Path(root) / name
    if (p / "train.bin").exists() and (p / "meta.json").exists():
        return root
    return None


def _synthetic_root(tmp_path_factory) -> Path:
    """Random bytes in the exact shape snn.data.Corpus insists on -- it
    validates split sizes against the registry, so the fake has to be the full
    100 MB.  Written once per session and shared with the other test module."""
    from snn.data import CORPORA
    meta_version = getattr(sys.modules["snn.data"], "META_VERSION", 1)
    spec = CORPORA[_CORPUS]
    root = tmp_path_factory.getbasetemp() / "synthetic_corpus"
    d = root / _CORPUS
    if (d / "meta.json").exists():
        return root
    d.mkdir(parents=True, exist_ok=True)
    vocab_size = int(getattr(spec, "vocab_size", 0) or _VOCAB[_CORPUS])
    rng = np.random.default_rng(0)
    for which, n in zip(("train", "val", "test"), spec.split):
        ids = rng.integers(0, vocab_size, size=int(n), dtype=np.uint8)
        ids[:vocab_size] = np.arange(vocab_size, dtype=np.uint8)
        ids.tofile(d / f"{which}.bin")
    meta = {
        "meta_version": meta_version,
        "corpus": _CORPUS,
        "vocab": list(range(vocab_size)),
        "vocab_size": vocab_size,
        "split_sizes": dict(zip(("train", "val", "test"),
                                (int(x) for x in spec.split))),
        "raw_bytes": int(sum(spec.split)),
        "zip_sha256": getattr(spec, "zip_sha256", ""),
        "raw_sha256": getattr(spec, "raw_sha256", ""),
        "url": "synthetic: random bytes, NOT the real corpus",
        "synthetic": True,
    }
    (d / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return root


@pytest.fixture(scope="session")
def corpus_dir(tmp_path_factory) -> str:
    real = _real_data_dir(_CORPUS)
    if real is not None:
        return real
    try:
        root = _synthetic_root(tmp_path_factory)
        Corpus(_CORPUS, str(root))
    except Exception as exc:
        pytest.skip(f"no usable corpus ({type(exc).__name__}: {exc}); "
                    f"run scripts/data/download_corpus.py")
    print(f"\nWARNING: running against a SYNTHETIC corpus at {root} -- "
          f"losses are meaningless, only their reproducibility is being tested")
    return str(root)


def _tiny_cfg(corpus_dir: str, out_dir, **over) -> Config:
    cfg = Config(
        corpus=_CORPUS, data_dir=corpus_dir,
        seq_len=32, batch_size=8,
        d_model=64, n_layers=2,
        lr=3e-3, warmup_steps=4, max_steps=64,
        device="cuda", seed=0, deterministic=True,
        out_dir=str(out_dir), run_name="t",
        eval_every=0, ckpt_every=0, log_every=0,
        cuda_graph=False,
    )
    for k, v in over.items():
        setattr(cfg, k, v)
    return cfg


def _release() -> None:
    """Reclaim a captured graph's private memory pool.

    The caller must drop its own reference first (`del tr`): a graph's pool is
    only returned when the CUDAGraph object dies, and passing the trainer into a
    helper would move the reference rather than remove it.
    """
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _losses(cfg: Config, n: int) -> tuple[list[float], list[torch.Tensor]]:
    tr = Trainer(cfg)
    out = [tr.step()["loss"] for _ in range(n)]
    params = [p.detach().float().cpu().clone() for p in tr.params]
    del tr
    _release()
    return out, params


# ----------------------------------------------------------------------
# seeding
# ----------------------------------------------------------------------

@pytest.mark.cuda
@requires_cuda
@pytest.mark.parametrize("cuda_graph", [False, True])
def test_same_seed_identical_trajectory(corpus_dir, tmp_path, cuda_graph):
    a, pa = _losses(_tiny_cfg(corpus_dir, tmp_path / "a", seed=0,
                              cuda_graph=cuda_graph), 12)
    b, pb = _losses(_tiny_cfg(corpus_dir, tmp_path / "b", seed=0,
                              cuda_graph=cuda_graph), 12)
    assert a == b, f"same seed produced different losses:\n{a}\n{b}"
    for i, (x, y) in enumerate(zip(pa, pb)):
        assert torch.equal(x, y), f"parameter {i} differs at fixed seed"


@pytest.mark.cuda
@requires_cuda
def test_different_seed_diverges(corpus_dir, tmp_path):
    """The complement of the test above, and not a formality: a seeding helper
    that silently ignores its argument passes 'same seed, same result'."""
    a, _ = _losses(_tiny_cfg(corpus_dir, tmp_path / "s0", seed=0), 8)
    b, _ = _losses(_tiny_cfg(corpus_dir, tmp_path / "s1", seed=1), 8)
    assert any(abs(x - y) > 1e-9 for x, y in zip(a, b)), \
        "seeds 0 and 1 produced the same trajectory; seeding is not wired up"


# ----------------------------------------------------------------------
# resume
# ----------------------------------------------------------------------

@pytest.mark.cuda
@requires_cuda
@pytest.mark.parametrize("cuda_graph", [False, True])
def test_resume_matches_uninterrupted_run(corpus_dir, tmp_path, cuda_graph):
    n, half = 20, 10

    ref, ref_params = _losses(
        _tiny_cfg(corpus_dir, tmp_path / "ref", cuda_graph=cuda_graph), n)

    tb = Trainer(_tiny_cfg(corpus_dir, tmp_path / "part", cuda_graph=cuda_graph))
    first = [tb.step()["loss"] for _ in range(half)]
    ckpt = tmp_path / "ckpt.pt"
    tb.save_checkpoint(ckpt)
    del tb
    _release()

    tc = Trainer(_tiny_cfg(corpus_dir, tmp_path / "resumed", cuda_graph=cuda_graph))
    tc.load_checkpoint(ckpt)
    assert tc.global_step == half, "checkpoint did not restore the step counter"
    second = [tc.step()["loss"] for _ in range(n - half)]
    resumed_params = [p.detach().float().cpu().clone() for p in tc.params]
    del tc
    _release()

    assert first == ref[:half]
    assert second == ref[half:], (
        "resume diverged from the uninterrupted run:\n"
        f"  uninterrupted {ref[half:]}\n  resumed       {second}"
    )
    for i, (x, y) in enumerate(zip(ref_params, resumed_params)):
        assert torch.equal(x, y), f"parameter {i} differs after resume"


@pytest.mark.cuda
@requires_cuda
def test_checkpoint_records_every_rng_stream(corpus_dir, tmp_path):
    """Phase 2 draws no random numbers after initialisation, so none of this is
    load-bearing *today*.  It is recorded anyway because the day someone adds
    dropout or stochastic depth, an unrecorded generator turns 'resume is
    exact' into 'resume is approximately right' with no failing test."""
    tr = Trainer(_tiny_cfg(corpus_dir, tmp_path / "ck"))
    tr.step()
    path = tmp_path / "c.pt"
    tr.save_checkpoint(path)
    del tr
    _release()

    ck = torch.load(path, map_location="cpu", weights_only=False)
    for key in ("step", "model", "optimizer", "config", "config_hash", "rng"):
        assert key in ck, f"checkpoint is missing {key!r}"
    for key in ("python", "numpy", "torch", "torch_cuda"):
        assert key in ck["rng"], f"checkpoint is missing RNG state {key!r}"
    assert ck["step"] == 1
    assert ck["config"]["seed"] == 0
    assert ck["config"]["d_model"] == 64


@pytest.mark.cuda
@requires_cuda
def test_checkpoint_write_is_atomic(corpus_dir, tmp_path, monkeypatch):
    """torch.save straight to the destination leaves a truncated file if the
    process dies mid-write, which on a multi-hour run is exactly when it dies.

    Overwriting and tmp-file cleanup are only the visible half of that claim.
    The half that matters -- a failed save must leave the *previous* checkpoint
    loadable -- is asserted by making torch.save raise, which is the only way to
    distinguish `tmp + os.replace` from a direct write without killing a
    process.
    """
    tr = Trainer(_tiny_cfg(corpus_dir, tmp_path / "at"))
    tr.step()
    path = tmp_path / "a.pt"
    tr.save_checkpoint(path)
    tr.save_checkpoint(path)  # overwriting must work on Windows too
    assert path.exists()
    assert not list(path.parent.glob("*.tmp")), "temporary file left behind"
    good = torch.load(path, map_location="cpu", weights_only=False)
    assert good["step"] == 1

    tr.step()  # so a successful in-place write would visibly change the file

    def _die(*_a, **_k):
        raise RuntimeError("simulated crash during torch.save")

    monkeypatch.setattr(torch, "save", _die)
    with pytest.raises(RuntimeError):
        tr.save_checkpoint(path)
    monkeypatch.undo()
    del tr
    _release()

    survivor = torch.load(path, map_location="cpu", weights_only=False)
    assert survivor["step"] == 1, (
        "a save that died mid-write damaged the previous checkpoint; "
        "save_checkpoint must write to a temporary file and os.replace it"
    )


# ----------------------------------------------------------------------
# evaluator
# ----------------------------------------------------------------------

def test_evaluate_rejects_unknown_protocol():
    cfg = Config()
    with pytest.raises(ValueError):
        evaluate(None, None, "val", cfg, "whatever")


@pytest.fixture(scope="session")
def eval_model(corpus_dir):
    if not torch.cuda.is_available():
        pytest.skip("needs a GPU")
    # batch_size is small on purpose: max_windows below truncates at a whole
    # batch, so a default batch of 128 would score zero windows and the
    # assertions would pass vacuously.
    cfg = Config(corpus=_CORPUS, data_dir=corpus_dir, seq_len=32, batch_size=8,
                 d_model=64, n_layers=2, device="cuda", seed=0)
    corpus = Corpus(_CORPUS, corpus_dir)
    cfg.vocab_size = int(corpus.vocab_size)
    model = build_model(cfg).to(torch.device(cfg.device))
    model.eval()
    return model, corpus, cfg


@pytest.mark.cuda
@requires_cuda
def test_fresh_bpc_invariant_to_batch_size(eval_model):
    """max_windows is a multiple of every batch size compared, so all three
    runs score exactly the same characters.  Any surviving difference is float
    reduction order, which is why this is a tolerance and not an equality."""
    model, corpus, base = eval_model
    results = {}
    for bs in (4, 8, 12):
        cfg = dataclasses.replace(base, batch_size=bs)
        results[bs] = evaluate(model, corpus, "val", cfg, "fresh", max_windows=24)

    n_chars = {bs: r["n_chars"] for bs, r in results.items()}
    assert len(set(n_chars.values())) == 1, \
        f"different batch sizes scored different numbers of characters: {n_chars}"
    ref = results[8]["bpc"]
    for bs, r in results.items():
        assert abs(r["bpc"] - ref) <= 1e-5 * max(1.0, abs(ref)), (
            f"fresh bpc moved with batch size: bs={bs} -> {r['bpc']!r} "
            f"vs bs=8 -> {ref!r}"
        )


@pytest.mark.cuda
@requires_cuda
@pytest.mark.parametrize("protocol", ["fresh", "carried"])
def test_evaluate_is_deterministic(eval_model, protocol):
    model, corpus, cfg = eval_model
    a = evaluate(model, corpus, "val", cfg, protocol, max_windows=32)
    b = evaluate(model, corpus, "val", cfg, protocol, max_windows=32)
    assert a["n_chars"] == b["n_chars"] > 0
    assert a["n_windows"] == b["n_windows"] > 0
    assert a["nats"] == b["nats"], "repeated evaluation gave a different total"
    assert a["bpc"] == b["bpc"]


@pytest.mark.cuda
@requires_cuda
def test_evaluate_returns_the_documented_keys(eval_model):
    model, corpus, cfg = eval_model
    r = evaluate(model, corpus, "val", cfg, "carried", max_windows=16)
    for key in ("bpc", "nats", "n_chars", "protocol", "firing_rate", "n_windows"):
        assert key in r
    assert r["protocol"] == "carried"
    assert r["bpc"] > 0.0 and np.isfinite(r["bpc"])
    assert isinstance(r["firing_rate"], list)


@pytest.mark.cuda
@requires_cuda
def test_evaluate_leaves_model_mode_untouched(eval_model):
    """evaluate() is called from inside the training loop; leaving the model in
    eval mode there would silently disable anything mode-dependent added later."""
    model, corpus, cfg = eval_model
    model.train()
    evaluate(model, corpus, "val", cfg, "fresh", max_windows=8)
    assert model.training is True
    model.eval()
    evaluate(model, corpus, "val", cfg, "fresh", max_windows=8)
    assert model.training is False
