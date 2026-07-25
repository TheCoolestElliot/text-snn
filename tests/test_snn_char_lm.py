"""Unit tests for snn_char_lm.

These cover what the `smoke` integration self-test does not: the data plumbing,
the checkpoint round-trip, chunked-vs-single forward equivalence, generation
shape/prompt handling, the CLI guards, and the reproducible entropy floors.

Everything runs on CPU so it is deterministic and CI-friendly. The default rate
encoder is *stochastic* (rate_gain < 1), which would make equivalence and
round-trip comparisons flaky; the fixtures build models with rate_gain=1.0 and
dropout=0.0 in eval mode, the one configuration under which the encoder is
deterministic (verified: forward_seq chunked == single-pass, maxdiff 0.0).
"""

import argparse
import math
import os

import pytest
import torch

import snn_char_lm as m

CPU = torch.device("cpu")
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _deterministic_model(seq_len=16, hidden=32, num_steps=3):
    """A small model + dataset whose forward pass is deterministic on CPU."""
    text = "hello world. the quick brown fox jumps. " * 20
    ds = m.CharDataset(text, seq_len=seq_len, val_split=0.0)
    cfg = m.SNNConfig(
        vocab_size=ds.vocab_size, hidden=hidden, num_layers=2,
        num_steps=num_steps, rate_gain=1.0, dropout=0.0, seq_len=seq_len,
    )
    model = m.SNNCharLM(cfg)
    model.eval()
    return ds, cfg, model


# --------------------------------------------------------------------------- #
# CharDataset
# --------------------------------------------------------------------------- #
def test_get_batch_shapes_and_range():
    ds = m.CharDataset("abcdefghij" * 100, seq_len=12, val_split=0.0)
    x, y = ds.get_batch(8, CPU)
    assert x.shape == (8, 12)
    assert y.shape == (8, 12)
    assert x.dtype == torch.long
    assert int(x.max()) < ds.vocab_size and int(x.min()) >= 0
    assert int(y.max()) < ds.vocab_size and int(y.min()) >= 0


def test_val_split_is_contiguous_disjoint_tail():
    text = "".join(chr(97 + (i % 10)) for i in range(1000))
    ds = m.CharDataset(text, seq_len=16, val_split=0.1)
    assert ds.has_val()
    n_tr = len(ds.splits["train"])
    n_val = len(ds.splits["val"])
    assert n_tr + n_val == len(text)          # partition, no overlap/gap
    assert torch.equal(ds.splits["val"], ds.data[len(text) - n_val:])  # the tail


def test_seq_len_too_long_raises():
    with pytest.raises(ValueError):
        m.CharDataset("short text", seq_len=100)


def test_val_split_too_big_raises():
    # 50 chars, seq_len 20 needs 21; val_split 0.9 -> 45 val leaves 5 for train.
    with pytest.raises(ValueError):
        m.CharDataset("a" * 50, seq_len=20, val_split=0.9)


def test_encode_decode_roundtrip_and_oov_drop():
    ds = m.CharDataset("abcabc abc.\n" * 10, seq_len=8, val_split=0.0)
    assert ds.decode(ds.encode("abc abc")) == "abc abc"
    # Characters outside the vocab are silently dropped by encode().
    assert ds.decode(ds.encode("abcZZZ")) == "abc"


# --------------------------------------------------------------------------- #
# Model: checkpoint round-trip and chunked equivalence
# --------------------------------------------------------------------------- #
def test_checkpoint_roundtrip_identical_logits(tmp_path):
    ds, cfg, model = _deterministic_model()
    x, _ = ds.get_batch(2, CPU)
    with torch.no_grad():
        before = model(x)

    path = str(tmp_path / "ckpt.pt")
    m._save_checkpoint(path, model, cfg, ds)
    model2, ds2 = m._load_checkpoint(path, CPU)
    model2.eval()
    with torch.no_grad():
        after = model2(x)

    assert torch.equal(before, after)          # deterministic encoder -> exact
    assert ds2.stoi == ds.stoi
    assert ds2.itos == ds.itos                 # int keys restored, not str


def test_forward_seq_chunked_equals_single_pass():
    ds, cfg, model = _deterministic_model(seq_len=24)
    x, _ = ds.get_batch(4, CPU)
    with torch.no_grad():
        full, _ = model.forward_seq(x, None)
        mems, parts = None, []
        for c0 in range(0, x.size(1), 8):
            logits, mems = model.forward_seq(x[:, c0:c0 + 8], mems)
            parts.append(logits)
        chunked = torch.cat(parts, dim=1)
    assert torch.allclose(full, chunked, atol=1e-6)


def test_loaded_dataset_is_inference_only(tmp_path):
    # _load_checkpoint rebuilds a vocab-only dataset with no .splits; get_batch
    # must fail loudly rather than silently doing the wrong thing.
    ds, cfg, model = _deterministic_model()
    path = str(tmp_path / "ckpt.pt")
    m._save_checkpoint(path, model, cfg, ds)
    _, ds2 = m._load_checkpoint(path, CPU)
    assert not hasattr(ds2, "splits")
    with pytest.raises(AttributeError):
        ds2.get_batch(2, CPU)


# --------------------------------------------------------------------------- #
# Generation
# --------------------------------------------------------------------------- #
def test_generate_length_and_prompt_prefix():
    ds, cfg, model = _deterministic_model()
    out = model.generate(ds, prompt="hello", length=20, device=CPU,
                         temperature=0.5)
    assert isinstance(out, str)
    assert out.startswith("hello")            # every prompt char is in-vocab
    assert len(out) == len("hello") + 20


def test_generate_top_k_zero_does_not_crash():
    # Regression for the top_k <= 0 guard (previously indexed an empty topk).
    ds, cfg, model = _deterministic_model()
    out = model.generate(ds, prompt="he", length=5, device=CPU, top_k=0)
    assert len(out) == len("he") + 5


def test_generate_empty_prompt():
    ds, cfg, model = _deterministic_model()
    out = model.generate(ds, prompt="", length=10, device=CPU)
    assert len(out) == 10


# --------------------------------------------------------------------------- #
# CLI guards and helpers
# --------------------------------------------------------------------------- #
def test_positive_int_rejects_non_positive():
    assert m._positive_int("5") == 5
    for bad in ("0", "-3"):
        with pytest.raises(argparse.ArgumentTypeError):
            m._positive_int(bad)


def test_make_surrogate_unknown_raises():
    with pytest.raises(ValueError):
        m._make_surrogate("bogus")
    assert m._make_surrogate("atan") is not None


def test_cli_requires_subcommand():
    with pytest.raises(SystemExit):
        m.build_parser().parse_args([])


def test_cli_rejects_zero_num_steps():
    with pytest.raises(SystemExit):
        m.build_parser().parse_args(["train", "--num-steps", "0"])


# --------------------------------------------------------------------------- #
# Reproducible entropy floors (pins the README's quoted numbers to the corpus)
# --------------------------------------------------------------------------- #
def test_corpus_bpc_floors_small_case():
    uni, big = m.corpus_bpc_floors("abab")
    # 'a' and 'b' each appear twice -> unigram is exactly 1 bit/char.
    assert abs(uni - 1.0) < 1e-9
    # Every 'a' is followed by 'b' and vice-versa -> bigram floor is 0.
    assert abs(big) < 1e-9


def test_corpus_bpc_floors_match_readme():
    path = os.path.join(REPO_ROOT, "input.txt")
    if not os.path.exists(path):
        pytest.skip("input.txt not present")
    with open(path, encoding="utf-8") as f:
        uni, big = m.corpus_bpc_floors(f.read())
    # These are the values quoted in README.md; keep them in lock-step.
    assert round(uni, 2) == 4.78
    assert round(big, 2) == 3.54
