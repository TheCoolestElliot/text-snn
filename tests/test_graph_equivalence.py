"""R5 gate: a captured graph must train the same model as the eager path.

01_reconnaissance.md rates R5 "Medium likelihood / Critical impact" because the
failure mode is silent: a graph that reads a stale static buffer, or that baked
in a learning rate, keeps producing plausible losses.  Nothing crashes.  The
run finishes.  The number is wrong.

So this file asserts three separate things, and each of them has a distinct way
of going wrong:

1.  *Numerical equivalence* -- graphed and eager training produce the same loss
    trajectory over 20 steps, and the same parameters at the end.  Catches
    static-buffer aliasing, stale inputs, and a captured region that quietly
    dropped an operation.

2.  *The learning rate is live* -- capture bakes Python scalars in.  A
    float LR therefore freezes the schedule while `param_group['lr']` and the
    log keep changing, which is the single most deceptive bug in the trainer.
    ``test_lr_tensor_is_live_in_graph`` replays at lr=0 and requires the
    parameters not to move; ``test_captured_step_scales_with_lr`` requires the
    update to be exactly proportional to the LR.  A frozen LR fails both.

3.  *The captured region is capture-safe by construction* -- checked by parsing
    ``Trainer._step_body`` rather than by running it, so it fails on a CPU-only
    machine too, at review time rather than at capture time.

The warm-up-restore behaviour is tested as well: the capture recipe runs three
real optimiser steps, and a run that reported "step 0" while actually being at
step 3 would make everything above meaningless.
"""

from __future__ import annotations

import ast
import gc
import inspect
import json
import os
import re
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest
import torch

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO / "src") not in sys.path:
    sys.path.insert(0, str(_REPO / "src"))

from snn.config import Config  # noqa: E402
from snn.data import Corpus  # noqa: E402
from snn.train import Trainer  # noqa: E402

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA-graph capture needs a GPU"
)

_CORPUS = os.environ.get("SNN_TEST_CORPUS", "enwik8")
_VOCAB = {"enwik8": 205, "text8": 27}
_N_STEPS = 20  # spec 12: the R5 gate compares >= 20 steps


# ----------------------------------------------------------------------
# corpus fixture: prefer the real thing, synthesise one if it is absent
# ----------------------------------------------------------------------

def _real_data_dir(name: str) -> str | None:
    root = os.environ.get("SNN_TEST_DATA_DIR") or str(_REPO / "data")
    p = Path(root) / name
    if (p / "train.bin").exists() and (p / "meta.json").exists():
        return root
    return None


def _synthetic_root(tmp_path_factory) -> Path:
    """Random bytes in the exact shape snn.data.Corpus insists on.

    Corpus validates split sizes against the registry, so a small stand-in is
    rejected -- the fake has to be the full 100 MB.  It is written once per
    pytest session and shared between test modules.  The numbers it produces
    are meaningless; what these gates check (graphed == eager, resume == no
    resume) does not depend on the data being real, only on it being fixed.
    """
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
    except Exception as exc:  # data.py validates more than we can necessarily fake
        pytest.skip(f"no usable corpus ({type(exc).__name__}: {exc}); "
                    f"run scripts/data/download_corpus.py")
    print(f"\nWARNING: running against a SYNTHETIC corpus at {root} -- "
          f"losses are meaningless, only their agreement is being tested")
    return str(root)


def _tiny_cfg(corpus_dir: str, out_dir, **over) -> Config:
    """Small enough to capture in under a second, large enough to be a real
    model: two spiking layers, a real GEMM, a real backward pass."""
    cfg = Config(
        corpus=_CORPUS, data_dir=corpus_dir,
        seq_len=32, batch_size=8,
        d_model=64, n_layers=2,
        lr=3e-3, warmup_steps=4, max_steps=64, lr_schedule="cosine",
        grad_clip=1.0,
        device="cuda", seed=0, deterministic=True,
        out_dir=str(out_dir), run_name="t",
        eval_every=0, ckpt_every=0, log_every=0,
    )
    for k, v in over.items():
        setattr(cfg, k, v)
    return cfg


def _release() -> None:
    """Reclaim a captured graph's private memory pool.

    The caller must drop its own reference first (``del tr``): a graph's pool is
    only returned when the CUDAGraph object dies, and passing the trainer into a
    helper would just move the reference, not remove it.
    """
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _run(cfg: Config, n_steps: int):
    tr = Trainer(cfg)
    losses = [tr.step()["loss"] for _ in range(n_steps)]
    params = [p.detach().float().cpu().clone() for p in tr.params]
    captured = tr.graph is not None
    del tr
    _release()
    return losses, params, captured


# ----------------------------------------------------------------------
# 3. static check -- runs without a GPU
# ----------------------------------------------------------------------

def _step_body_statements() -> tuple[str, list[ast.stmt]]:
    src = textwrap.dedent(inspect.getsource(Trainer._step_body))
    fn = ast.parse(src).body[0]
    body = list(fn.body)
    if (body and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)):
        body = body[1:]  # the docstring names the banned calls; do not scan it
    return "\n".join(ast.unparse(n) for n in body), body


def test_step_body_contains_no_host_sync():
    """Anything that reads a tensor's value on the host aborts capture.

    Parsed rather than grepped so that a mention in the docstring or a comment
    cannot trip it, and unparsed so formatting cannot hide a call.
    """
    code, _ = _step_body_statements()
    banned_substrings = [".item()", ".tolist()", ".numpy()", ".cpu(",
                         "torch.cuda.synchronize", "print("]
    for bad in banned_substrings:
        assert bad not in code, f"{bad!r} inside the captured region: {code}"
    # float(t)/int(t)/bool(t) sync; .float()/.int() do not, hence the lookbehind.
    for name in ("float", "int", "bool", "len"):
        assert not re.search(rf"(?<![.\w]){name}\s*\(", code), \
            f"{name}(...) inside the captured region: {code}"


def test_step_body_has_no_python_control_flow():
    """A Python branch is resolved once, at capture time, and then frozen.

    Even a branch on a config value is a hazard here: it makes the captured
    program depend on state that the caller can still mutate afterwards, which
    is the same class of bug as the baked-in learning rate.
    """
    _, body = _step_body_statements()
    for node in body:
        for sub in ast.walk(node):
            assert not isinstance(sub, (ast.If, ast.While, ast.IfExp, ast.Try)), \
                f"{type(sub).__name__} inside the captured region"


def test_step_body_zeroes_grads_without_freeing_them():
    """set_to_none=True hands the gradients back to the allocator, so the next
    backward may place them somewhere else -- and the graph replays writes to
    the old addresses."""
    code, _ = _step_body_statements()
    assert "set_to_none=False" in code


# ----------------------------------------------------------------------
# 1. graphed vs eager
# ----------------------------------------------------------------------

@pytest.mark.cuda
@requires_cuda
def test_graphed_matches_eager_losses(corpus_dir, tmp_path):
    graphed, gparams, captured = _run(
        _tiny_cfg(corpus_dir, tmp_path / "g", cuda_graph=True), _N_STEPS)
    assert captured, "cuda_graph=True did not produce a captured graph"
    eager, eparams, not_captured = _run(
        _tiny_cfg(corpus_dir, tmp_path / "e", cuda_graph=False), _N_STEPS)
    assert not not_captured

    assert len(graphed) == len(eager) == _N_STEPS
    worst = 0.0
    identical = 0
    for i, (a, b) in enumerate(zip(graphed, eager)):
        assert np.isfinite(a) and np.isfinite(b), f"non-finite loss at step {i}"
        identical += int(a == b)
        rel = abs(a - b) / max(abs(b), 1e-12)
        worst = max(worst, rel)
        assert rel <= 1e-6, (
            f"step {i}: graphed loss {a!r} vs eager {b!r} (rel {rel:.3e}). "
            "The two paths run the same _step_body, so any divergence is the "
            "graph reading state the eager path does not."
        )
    print(f"\nR5: {identical}/{_N_STEPS} steps bit-identical, worst rel {worst:.3e}")

    for i, (a, b) in enumerate(zip(gparams, eparams)):
        assert torch.allclose(a, b, rtol=1e-5, atol=1e-7), \
            f"parameter tensor {i} diverged after {_N_STEPS} steps"


@pytest.mark.cuda
@requires_cuda
def test_graphed_and_eager_agree_on_firing_rates(corpus_dir, tmp_path):
    """R4 says firing rate is a first-class metric from step 0, so it has to
    survive capture too -- it is published out of the graph by copy_ into a
    static buffer, which is exactly the mechanism R5 is about."""
    tg = Trainer(_tiny_cfg(corpus_dir, tmp_path / "g", cuda_graph=True))
    te = Trainer(_tiny_cfg(corpus_dir, tmp_path / "e", cuda_graph=False))
    for _ in range(3):
        mg, me = tg.step(), te.step()
        assert len(mg["firing_rate"]) == len(me["firing_rate"])
        for a, b in zip(mg["firing_rate"], me["firing_rate"]):
            assert abs(a - b) <= 1e-6 * max(1.0, abs(b))
        assert np.isfinite(mg["grad_norm"]) and mg["grad_norm"] >= 0.0
    del tg, te
    _release()


@pytest.mark.cuda
@requires_cuda
def test_static_input_buffers_never_move(corpus_dir, tmp_path):
    tr = Trainer(_tiny_cfg(corpus_dir, tmp_path / "s", cuda_graph=True))
    tr.step()
    ptrs = (tr.static_x.data_ptr(), tr.static_y.data_ptr(),
            tr.static_metrics.data_ptr())
    grads = [p.grad.data_ptr() for p in tr.params]
    for _ in range(5):
        tr.step()
    assert (tr.static_x.data_ptr(), tr.static_y.data_ptr(),
            tr.static_metrics.data_ptr()) == ptrs
    assert [p.grad.data_ptr() for p in tr.params] == grads, \
        "gradient storage moved; the replayed graph is writing to freed memory"
    del tr
    _release()


@pytest.mark.cuda
@requires_cuda
def test_capture_restores_the_pre_warmup_state(corpus_dir, tmp_path):
    """The capture recipe runs three real steps.  If they are not undone, the
    run silently begins at step 3 with warmed-up Adam moments, and every
    graphed-vs-eager comparison in this file is comparing different models."""
    tr = Trainer(_tiny_cfg(corpus_dir, tmp_path / "w", cuda_graph=True))
    before = [p.detach().clone() for p in tr.params]
    tr._ensure_ready()
    assert tr.graph is not None
    for i, (p, b) in enumerate(zip(tr.params, before)):
        assert torch.equal(p.detach(), b), f"warm-up left parameter {i} moved"
    assert tr.global_step == 0
    for p in tr.params:
        st = tr.opt.state[p]
        assert float(st["exp_avg"].abs().max()) == 0.0
        assert float(st["exp_avg_sq"].abs().max()) == 0.0
        assert float(torch.as_tensor(st["step"]).max()) == 0.0
    del tr
    _release()


# ----------------------------------------------------------------------
# 2. the learning-rate trap
# ----------------------------------------------------------------------

@pytest.mark.cuda
@requires_cuda
def test_lr_tensor_is_live_in_graph(corpus_dir, tmp_path):
    """THE test this module exists for.

    Capture bakes Python floats into the kernels.  If the optimiser's LR is a
    float, replaying after setting lr=0 still moves the parameters -- by
    exactly the amount the LR that was live at capture time dictates.  If it is
    a tensor written in place, lr=0 means AdamW multiplies its update (and its
    decoupled weight decay) by zero and the parameters do not move at all.
    """
    cfg = _tiny_cfg(corpus_dir, tmp_path / "lr", cuda_graph=True,
                    warmup_steps=0, lr_schedule="constant", lr=1e-2)
    tr = Trainer(cfg)
    assert tr.graph is None, "capture must not happen before the first step"
    tr.step()
    assert tr.graph is not None

    lr_entry = tr.opt.param_groups[0]["lr"]
    assert torch.is_tensor(lr_entry), (
        "param_group['lr'] is a Python float. capturable AdamW must hold it as "
        "a CUDA tensor, otherwise the value is a compile-time constant of the "
        "captured graph and the schedule stops working the moment we capture."
    )

    before = [p.detach().clone() for p in tr.params]
    tr._set_lr(0.0)
    tr.graph.replay()
    torch.cuda.synchronize()
    moved = max(float((p.detach() - b).abs().max())
                for p, b in zip(tr.params, before))
    assert moved == 0.0, (
        f"replaying at lr=0 moved parameters by {moved:.3e}. The graph is using "
        f"the learning rate that was live at capture time ({cfg.lr}); the "
        "schedule is decorative."
    )

    before = [p.detach().clone() for p in tr.params]
    tr._set_lr(cfg.lr)
    tr.graph.replay()
    torch.cuda.synchronize()
    moved = max(float((p.detach() - b).abs().max())
                for p, b in zip(tr.params, before))
    assert moved > 0.0, "restoring a non-zero lr did not move the parameters"
    del tr
    _release()


@pytest.mark.cuda
@requires_cuda
def test_captured_step_scales_with_lr(corpus_dir, tmp_path):
    """A stronger form of the same check: AdamW's update is exactly linear in
    the learning rate, so replaying the *same* state and inputs at 2x the LR
    must move the parameters exactly twice as far.  A frozen LR gives a ratio
    of 1, which this catches even if some other term happens to be zero.

    weight_decay is switched off here for numerical reasons, not scientific
    ones: decoupled decay is applied as ``p *= (1 - lr*wd)``, and for a
    parameter with no gradient the whole delta is then the difference of two
    nearly-equal fp32 numbers, which is proportional to lr only to about 1e-3.
    The zero-LR test above covers the decay path.
    """
    cfg = _tiny_cfg(corpus_dir, tmp_path / "sc", cuda_graph=True,
                    warmup_steps=0, lr_schedule="constant", weight_decay=0.0)
    tr = Trainer(cfg)
    tr.step()
    snapshot = tr._snapshot_train_state()

    def delta(lr: float):
        tr._restore_train_state(snapshot)
        before = [p.detach().clone() for p in tr.params]
        tr._set_lr(lr)
        tr.graph.replay()
        torch.cuda.synchronize()
        return [(p.detach() - b).clone() for p, b in zip(tr.params, before)]

    d1 = delta(1e-3)
    d2 = delta(2e-3)
    num = sum(float((2.0 * a - b).abs().sum()) for a, b in zip(d1, d2))
    den = sum(float(b.abs().sum()) for b in d2)
    assert den > 0.0, "no parameter moved at all; the test is not measuring anything"
    assert num / den < 1e-3, (
        f"doubling the LR did not double the update (residual {num / den:.3e}); "
        "the replayed graph is not reading the LR tensor"
    )
    del tr
    _release()


@pytest.mark.cuda
@requires_cuda
def test_schedule_is_written_every_step(corpus_dir, tmp_path):
    """Plumbing check to sit beside the two above: the value the optimiser
    holds after each step must be the value the schedule prescribes for that
    step, and warm-up must actually vary it."""
    cfg = _tiny_cfg(corpus_dir, tmp_path / "sch", cuda_graph=True,
                    warmup_steps=4, lr_schedule="cosine")
    tr = Trainer(cfg)
    seen = []
    for _ in range(6):
        m = tr.step()
        seen.append(m["lr"])
        # fp32 storage, so compare relatively rather than to the last bit
        assert abs(tr._current_lr() - m["lr"]) <= 1e-6 * max(1e-12, abs(m["lr"]))
        assert abs(m["lr"] - tr._lr_at(m["step"])) <= 1e-12
    assert len(set(seen)) > 1, "the schedule never changed the learning rate"
    assert seen[0] < seen[3], "warm-up is not ramping"
    del tr
    _release()
