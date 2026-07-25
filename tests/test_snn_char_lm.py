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


# --------------------------------------------------------------------------- #
# Tier 5: device, eval, with_vocab, resume state, CSV logging
# --------------------------------------------------------------------------- #
def test_pick_device_cpu_and_auto():
    assert m.pick_device("cpu").type == "cpu"
    assert m.pick_device("auto").type in ("cpu", "cuda")


def test_pick_device_cuda_unavailable_falls_back(monkeypatch, capsys):
    monkeypatch.setattr(m.torch.cuda, "is_available", lambda: False)
    assert m.pick_device("cuda").type == "cpu"
    assert "CUDA is unavailable" in capsys.readouterr().err


def test_pick_device_invalid_raises():
    # A typo of the new --device flag must raise a clean ValueError (main()
    # formats it) rather than an opaque RuntimeError from torch.device.
    for bad in ("gpu", "0", "foo", "cuda:x"):
        with pytest.raises(ValueError):
            m.pick_device(bad)


def test_with_vocab_uses_given_vocab_and_drops_oov():
    base = m.CharDataset("abcdef " * 50, seq_len=8, val_split=0.0)
    ds = m.CharDataset.with_vocab("abcZabc def " * 20, seq_len=8,
                                  stoi=base.stoi, itos=base.itos)
    assert ds.stoi == base.stoi          # vocab preserved, not re-derived
    assert "Z" not in ds.stoi            # the OOV 'Z' has no id
    assert int(ds.data.max()) < ds.vocab_size
    assert "all" in ds.splits and len(ds.splits["all"]) == len(ds.data)


def test_all_split_spans_everything():
    ds = m.CharDataset("abc " * 100, seq_len=8, val_split=0.1)
    assert len(ds.splits["all"]) == len(ds.data)


def test_require_matching_vocab():
    ds = m.CharDataset("abcdef " * 50, seq_len=8)
    m._require_matching_vocab(
        {"cfg": {"vocab_size": ds.vocab_size}, "stoi": ds.stoi}, ds, "--init-from")
    with pytest.raises(ValueError):
        m._require_matching_vocab(
            {"cfg": {"vocab_size": ds.vocab_size + 1}, "stoi": ds.stoi}, ds, "x")
    scrambled = dict(ds.stoi)
    k = list(scrambled)
    scrambled[k[0]], scrambled[k[1]] = scrambled[k[1]], scrambled[k[0]]
    with pytest.raises(ValueError):
        m._require_matching_vocab(
            {"cfg": {"vocab_size": ds.vocab_size}, "stoi": scrambled}, ds, "x")


def test_save_state_roundtrip(tmp_path):
    ds, cfg, model = _deterministic_model()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    x, y = ds.get_batch(2, CPU)
    logits, _ = model.forward_seq(x, None)
    loss = torch.nn.functional.cross_entropy(
        logits.reshape(-1, cfg.vocab_size), y.reshape(-1))
    loss.backward()
    opt.step()                                       # give the optimizer state
    path = str(tmp_path / "s.state")
    m._save_state(path, model, cfg, ds, opt, update=7, best_val=1.23,
                  steps=3000, warmup=100)

    ck = torch.load(path, map_location="cpu", weights_only=True)   # safe load
    assert ck["format_version"] == 1
    assert ck["update"] == 7 and abs(ck["best_val"] - 1.23) < 1e-9
    assert ck["steps"] == 3000 and ck["warmup"] == 100
    assert "opt_state" in ck
    m._restore_rng(ck)                               # must not raise
    opt2 = torch.optim.AdamW(model.parameters(), lr=1e-3)
    opt2.load_state_dict(ck["opt_state"])            # reloadable into a fresh opt


def test_eval_cmd_scores_checkpoint(tmp_path, capsys):
    ds, cfg, model = _deterministic_model(seq_len=16)
    ckpt = str(tmp_path / "e.pt")
    m._save_checkpoint(ckpt, model, cfg, ds)
    corpus = "hello world. the quick brown fox jumps. " * 20
    data = str(tmp_path / "c.txt")
    with open(data, "w", encoding="utf-8") as f:
        f.write(corpus)
    args = argparse.Namespace(ckpt=ckpt, data=data, split="all", val_split=0.0,
                              seq_len=16, batch_size=8, eval_batches=2,
                              seed=0, device="cpu")
    m.eval_cmd(args)
    out = capsys.readouterr().out
    assert "[eval]" in out and "bpc" in out


def test_eval_split_all_ignores_val_split(tmp_path, capsys):
    # A large --val-split must NOT abort `--split all` (val_split is irrelevant
    # there); scoring 'all' uses the whole corpus regardless.
    ds, cfg, model = _deterministic_model(seq_len=16)
    ckpt = str(tmp_path / "e.pt")
    m._save_checkpoint(ckpt, model, cfg, ds)
    corpus = "hello world. the quick brown fox jumps. " * 20
    data = str(tmp_path / "c.txt")
    with open(data, "w", encoding="utf-8") as f:
        f.write(corpus)
    args = argparse.Namespace(ckpt=ckpt, data=data, split="all", val_split=0.99,
                              seq_len=16, batch_size=8, eval_batches=2,
                              seed=0, device="cpu")
    m.eval_cmd(args)                                  # must not raise
    assert "[eval]" in capsys.readouterr().out


def test_warn_dropped_prompt_chars(capsys):
    ds = m.CharDataset("abcdef " * 50, seq_len=8)
    m._warn_dropped_prompt_chars(ds, "abc")          # all in vocab -> silent
    assert capsys.readouterr().err == ""
    m._warn_dropped_prompt_chars(ds, "abZ")          # partial
    assert "not in vocab" in capsys.readouterr().err
    m._warn_dropped_prompt_chars(ds, "ZZZ")          # none in vocab
    assert "none of the" in capsys.readouterr().err


def test_metrics_csv_single_header_on_reopen(tmp_path):
    p = str(tmp_path / "m.csv")
    w, f = m._open_metrics_csv(p)
    m._csv_row(w, f, 1, "train", 2.0, 3.0, 0.1)
    f.close()
    w2, f2 = m._open_metrics_csv(p)                  # resume: append, no re-header
    m._csv_row(w2, f2, 2, "val", 1.5, None, None)
    f2.close()
    lines = open(p, encoding="utf-8").read().strip().splitlines()
    assert sum(1 for ln in lines if ln.startswith("update,split")) == 1
    assert len(lines) == 3                           # header + 2 rows


def test_cli_eval_parses():
    args = m.build_parser().parse_args(["eval", "--ckpt", "x.pt", "--split", "val"])
    assert args.func is m.eval_cmd and args.split == "val"


def test_cli_resume_and_init_from_mutually_exclusive():
    with pytest.raises(SystemExit):
        m.build_parser().parse_args(
            ["train", "--init-from", "a.pt", "--resume", "b.pt"])


def test_cli_learn_threshold_flag():
    args = m.build_parser().parse_args(["train", "--learn-threshold",
                                        "--threshold", "0.8"])
    assert args.learn_threshold is True and args.threshold == 0.8
