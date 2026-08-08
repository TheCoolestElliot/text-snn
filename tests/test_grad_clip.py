"""Decision #7 (rev 6 of ``04_phase4_interim.md``): the gradient clip's fp32
accumulator silently zeroes the update on a large-but-finite gradient instead
of clipping it, because ``clip_grad_norm_``'s foreach path squares each
gradient in its own (fp32) dtype before summing -- ``EXP_009`` Sec 9.4 hit a
gradient of 5.46e31, which squares to 2.98e63 and overflows fp32's ~3.4e38
range on its own, before any cross-parameter sum runs.

This file has two jobs:

1.  Prove ``snn.train._clip_grad_norm_fp64`` reproduces the *bug*, on exactly
    the numbers ``EXP_009`` measured, against the stock implementation it
    replaces -- so the regression this guards is not hypothetical.
2.  Prove the *fix* rescales instead of zeroing on that same input, and
    otherwise agrees with the stock implementation wherever the stock
    implementation was already correct (ordinary gradients never exercised
    the bug, so the two must not diverge there).

No GPU is required: the bug and the fix are both properties of fp32 vs fp64
arithmetic, reproducible on CPU tensors, which is also what keeps this test
meaningful on a CPU-only review machine (the same reasoning
``test_graph_equivalence.py``'s static checks use).
"""

from __future__ import annotations

import ast
import inspect
import re
import sys
import textwrap
from pathlib import Path

import pytest
import torch

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO / "src") not in sys.path:
    sys.path.insert(0, str(_REPO / "src"))

from snn.train import _clip_grad_norm_fp64  # noqa: E402

# EXP_009 Sec 9.4's own measured numbers, reused verbatim so this test is
# anchored to the finding rather than to a round number chosen for effect.
_EXP_009_BAD_GRAD = 5.46e31
# ||g|| in fp64, as EXP_009's own table reports it -- across *all* of the
# distilled model's parameters, not the three synthetic ones below. Used
# only as an order-of-magnitude cross-check (this file does not reproduce
# EXP_009's full parameter set), never as an equality target.
_EXP_009_FP64_NORM = 5.833e31


def _params_with_grads(*grad_rows: list[float]) -> list[torch.Tensor]:
    """One leaf fp32 parameter per row, ``.grad`` set directly -- matching
    how the trainer's own ``self.params`` look immediately after
    ``loss.backward()``, one tensor per named parameter."""
    params = []
    for row in grad_rows:
        p = torch.zeros(len(row), dtype=torch.float32, requires_grad=True)
        p.grad = torch.tensor(row, dtype=torch.float32)
        params.append(p)
    return params


# ----------------------------------------------------------------------
# 1. the bug reproduces on EXP_009's own gradient
# ----------------------------------------------------------------------

def test_stock_clip_zeroes_the_update_on_exp_009s_gradient():
    """Ground truth: reproduce the failure the fix exists for, on the stock
    implementation, so a change to PyTorch's own behaviour would surface here
    rather than only inside this file's other assertions."""
    params = _params_with_grads([1.0], [2.0], [_EXP_009_BAD_GRAD])
    gnorm = torch.nn.utils.clip_grad_norm_(
        params, 1.0, norm_type=2.0, error_if_nonfinite=False, foreach=True,
    )
    assert not torch.isfinite(gnorm), "expected the fp32 accumulator to overflow"
    # clamp(1.0 / (inf + 1e-6), max=1.0) == 0.0 exactly -- IEEE division by
    # inf, not a NaN -- so the update is zeroed, not merely shrunk.
    assert (params[0].grad == 0).all() and (params[2].grad == 0).all()


# ----------------------------------------------------------------------
# 2. the fix rescales on the exact same input
# ----------------------------------------------------------------------

def test_fp64_clip_rescales_where_stock_zeroes():
    params = _params_with_grads([1.0], [2.0], [_EXP_009_BAD_GRAD])
    total_norm = _clip_grad_norm_fp64(params, 1.0)

    assert torch.isfinite(total_norm), "fp64 accumulation must not overflow here"
    # With only one gradient anywhere near this magnitude, the L2 norm is
    # dominated by it (1**2 and 2**2 are ~60 orders of magnitude smaller than
    # (5.46e31)**2) -- so the fp64-accumulated norm should equal the bad
    # gradient itself to high precision. This is a *different* number from
    # EXP_009's own 5.833e31 (that norm is across the whole distilled
    # model's parameters, not these three synthetic ones); both being within
    # the same order of magnitude is the only claim made about the pair.
    assert total_norm.item() == pytest.approx(_EXP_009_BAD_GRAD, rel=1e-6)
    assert total_norm.item() / _EXP_009_FP64_NORM == pytest.approx(1.0, rel=0.2)

    for p in params:
        assert torch.isfinite(p.grad).all(), "clipped gradient must stay finite"
    # the huge gradient is scaled down, not erased: clip_coef ~= 1/5.46e31,
    # so the result is tiny but strictly nonzero and finite.
    assert params[2].grad.abs().item() > 0.0


# ----------------------------------------------------------------------
# 3. the fix agrees with the stock implementation wherever the bug never
#    fires -- the fp64 accumulation must not change ordinary behaviour
# ----------------------------------------------------------------------

def test_fp64_clip_matches_stock_for_ordinary_gradients():
    torch.manual_seed(0)
    rows = [torch.randn(17).tolist(), torch.randn(5).tolist(), torch.randn(64).tolist()]

    stock_params = _params_with_grads(*rows)
    stock_gnorm = torch.nn.utils.clip_grad_norm_(
        stock_params, 1.0, norm_type=2.0, error_if_nonfinite=False, foreach=True,
    )

    fp64_params = _params_with_grads(*rows)
    fp64_gnorm = _clip_grad_norm_fp64(fp64_params, 1.0)

    assert fp64_gnorm.item() == pytest.approx(stock_gnorm.item(), rel=1e-5)
    for sp, fp in zip(stock_params, fp64_params):
        assert torch.allclose(sp.grad, fp.grad, rtol=1e-5, atol=1e-7)


def test_fp64_clip_is_a_noop_below_max_norm():
    """max_norm=inf (grad_clip<=0's committed sentinel, train.py's
    ``self._max_norm``) must leave gradients untouched, same as the stock
    implementation's own no-clip path."""
    params = _params_with_grads([0.1, -0.2, 0.3])
    original = [p.grad.clone() for p in params]
    _clip_grad_norm_fp64(params, float("inf"))
    for p, orig in zip(params, original):
        assert torch.equal(p.grad, orig)


# ----------------------------------------------------------------------
# 4. capture-safety, checked by parsing rather than running -- same method
#    test_graph_equivalence.py uses for Trainer._step_body, applied to the
#    helper _step_body now calls from inside the captured region
# ----------------------------------------------------------------------

def _fn_body() -> tuple[str, list[ast.stmt]]:
    src = textwrap.dedent(inspect.getsource(_clip_grad_norm_fp64))
    fn = ast.parse(src).body[0]
    body = list(fn.body)
    if (body and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)):
        body = body[1:]  # the docstring names fp32/fp64 by number; don't scan it
    return "\n".join(ast.unparse(n) for n in body), body


def test_clip_helper_contains_no_host_sync():
    code, _ = _fn_body()
    banned_substrings = [".item()", ".tolist()", ".numpy()", ".cpu(",
                         "torch.cuda.synchronize", "print("]
    for bad in banned_substrings:
        assert bad not in code, f"{bad!r} inside the captured region: {code}"
    for name in ("float", "int", "bool", "len"):
        assert not re.search(rf"(?<![.\w]){name}\s*\(", code), \
            f"{name}(...) inside the captured region: {code}"


def test_clip_helper_has_no_value_dependent_control_flow():
    """A comprehension's ``if`` filter is not ``ast.If`` -- it is a
    structural filter over which parameters exist, resolved identically on
    every call (the docstring's capture-safety note), not a branch on a
    tensor's runtime value. This asserts the stronger, value-dependent forms
    are absent."""
    _, body = _fn_body()
    for node in body:
        for sub in ast.walk(node):
            assert not isinstance(sub, (ast.If, ast.While, ast.IfExp, ast.Try)), \
                f"{type(sub).__name__} inside the captured region"
