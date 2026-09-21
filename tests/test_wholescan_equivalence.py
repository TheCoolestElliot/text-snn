"""Bit-identity gate: the whole-scan kernels ARE the per-timestep path, to the last bit.

`snn.wholescan` replaces L jiterator launches per layer with one NVRTC kernel. It
is outside the Phase 1-5 protocol and is allowed to exist on one condition: that no
number anywhere moves. Not "within tolerance" -- a gradient that differs by one ulp
is inside every tolerance in this repository and changes every trained weight, so a
tolerance cannot gate this. Everything below compares BIT PATTERNS (`_same_bits`
views the floats as int32, because `torch.equal` calls -0.0 and +0.0 equal).

What is compared, and against what:

  1. the ORACLE is the per-timestep jiterator Function, reached the way the model
     reaches it: through `lif_scan` / `twocomp_scan` / `twocomp_detach_scan` with
     `use_scan_impl("jiterator")`. It stays in the tree for exactly this;
  2. every output (spikes, final state), every SAVED INTERMEDIATE (`v_pre`,
     `vs_seq` -- asserted at the kernel boundary on the tensor itself, because a
     mutation once escaped a 24-test gate by corrupting a quantity that reached no
     assertion), and every gradient (`cur`, `v0`, and `w`, `beta_s` where they
     exist), for all six variants;
  3. across shapes (an odd tiny one, the 735K and 5.0M research shapes, the L=1 and
     B=1 edges the chat decoder uses), firing regimes from ~3% to ~40%, non-zero
     initial state, eager and under CUDA-graph replay, with grad and without;
  4. the FALLBACK rules: what the raw-pointer kernel cannot be trusted with goes to
     the per-timestep path and gives the same answer, never a mis-read;
  5. a real `Trainer`, a few dozen steps: parameters and AdamW state bit-equal
     between the two settings of `SNN_SCAN_IMPL`.

THIS FILE CARRIES ITS OWN PROOF THAT IT CAN FAIL. The reason the kernels need a
contraction barrier (`snn.wholescan`, "Why it is bit-identical, and the one way it
was not") is that without it NVRTC re-pairs one FMA per timestep and three of the
six backwards come out one ulp different. `test_without_the_barrier_it_is_not_bit_
identical` builds exactly that kernel and asserts the comparison used everywhere
above REJECTS it -- while `allclose` at the R10 tolerances accepts it, which is the
point. If a future NVRTC stops re-pairing, that test fails, and the right response
is to find a new negative control, not to delete the assertion: a bitwise gate that
has never been seen to fail is a claim, not a gate. The wrapper's own mutation
campaign is `scripts/audit/12_wholescan_mutations.py`.

The R10 question -- is either path RIGHT -- is not asked here. It is asked by
`test_neuron_equivalence.py`, `test_twocomp_equivalence.py` and
`test_twocomp_detach_equivalence.py`, each of which now runs every CUDA test under
both implementations (conftest.py, `scan_impl`).
"""

from __future__ import annotations

import gc
import json
import os
import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest
import torch

_REPO = Path(__file__).resolve().parents[1]
_SRC = str(_REPO / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import snn.wholescan as ws  # noqa: E402
from snn.kernels import (  # noqa: E402
    RESET_CODES,
    RESET_MODES,
    backward_source,
    forward_source,
    jiterator_available,
)
from snn.neuron import lif_scan, lif_scan_eager  # noqa: E402
from snn.twocomp import (  # noqa: E402
    twocomp_backward_source,
    twocomp_forward_source,
    twocomp_scan,
)
from snn.twocomp_detach import (  # noqa: E402
    twocomp_detach_backward_source,
    twocomp_detach_scan,
)

requires_wholescan = pytest.mark.skipif(
    not (torch.cuda.is_available() and jiterator_available() and ws.wholescan_available()),
    reason="needs a CUDA device with working jiterator AND NVRTC whole-scan kernels",
)

VARIANTS = ws.VARIANTS
#: The three whose backward ADDS the carried adjoint directly, which is where
#: NVRTC re-pairs an FMA across timesteps if nothing stops it.
HAZARD_VARIANTS = ("lif_none", "twocomp", "twocomp_detach")

# (current scale, beta, thr, alpha). The scale sets the firing rate (~3% / ~25% /
# ~40%), and only the first row is the spec §13 baseline: beta = 0.5 is a power of
# two and thr = 1.0 is 1.0, and both make a wrong rounding accidentally right.
REGIMES = [(0.5, 0.5, 1.0, 2.0), (1.0, 0.9, 0.7, 3.3), (3.0, 0.95, 1.5, 2.0)]

# odd and tiny (no dimension a multiple of anything); the 735K research shape; the
# 5.0M one, which is also where B*L*d first passes 2^25 elements per tensor.
SHAPES = [(3, 17, 5), (128, 256, 512), (128, 256, 1481)]

# L=1 is one chat decode step, B=1 a replay feed, d=1 a degenerate channel axis.
EDGE_SHAPES = [(1, 1, 1), (2, 3, 1), (256, 1, 1024), (1, 400, 1024), (5, 1, 3)]


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def _same_bits(a: torch.Tensor, b: torch.Tensor) -> bool:
    """Equal as bit patterns: distinguishes -0.0 from +0.0, and NaN from NaN'."""
    assert a.dtype is torch.float32 and b.dtype is torch.float32, (a.dtype, b.dtype)
    if a.shape != b.shape:
        return False
    return bool((a.contiguous().view(torch.int32) == b.contiguous().view(torch.int32)).all())


def _describe(a: torch.Tensor, b: torch.Tensor) -> str:
    if a.shape != b.shape:
        return f"shape {tuple(a.shape)} vs {tuple(b.shape)}"
    n = int((a.contiguous().view(torch.int32) != b.contiguous().view(torch.int32)).sum())
    return f"{n}/{a.numel()} elements differ, max|d|={float((a - b).abs().max()):.3e}"


def _assert_bits(got: dict, want: dict, where) -> None:
    assert got.keys() == want.keys(), (where, sorted(got), sorted(want))
    for name in want:
        assert _same_bits(got[name], want[name]), (
            f"{where}: {name} is not bit-identical -- {_describe(got[name], want[name])}")


def _poison(like: torch.Tensor, n: int = 6) -> None:
    """Fill the caching allocator's free blocks of this size with NaN.

    The whole-scan launchers allocate their outputs with `torch.empty`, and the
    allocator hands back the most recently freed block of the right size -- which,
    in a test that has just run the oracle, holds the oracle's CORRECT answer. A
    kernel that failed to write an output would then be rescued by stale memory
    and pass. Poisoning first means an unwritten output reads as NaN.
    """
    junk = [torch.full_like(like, float("nan")) for _ in range(n)]
    del junk


def _is_tc(variant: str) -> bool:
    return not variant.startswith("lif_")


def _inputs(variant: str, shape, scale: float, seed: int) -> dict:
    """Random current, NON-ZERO initial state, random cotangents for both outputs."""
    B, L, d = shape
    g = torch.Generator(device="cuda").manual_seed(seed)

    def rnd(*s, k=1.0):
        return torch.randn(*s, device="cuda", generator=g) * k

    width = 2 * d if _is_tc(variant) else d
    t = {"cur": rnd(B, L, d, k=scale), "v0": rnd(B, width, k=0.5),
         "gs": rnd(B, L, d, k=1e-3), "gv": rnd(B, width, k=1e-3)}
    if _is_tc(variant):
        t["w"] = 0.1 + rnd(1, d, k=0.3)
        t["beta_s"] = torch.sigmoid(2.9 + rnd(1, d))
    return t


def _scan(variant: str, leaves: dict, beta, thr, alpha):
    """The MODEL-FACING call for `variant`, fused. Which kernels run is the selector's."""
    if variant.startswith("lif_"):
        return lif_scan(leaves["cur"], leaves["v0"], beta, thr, alpha,
                        variant[len("lif_"):], fused=True)
    fn = twocomp_scan if variant == "twocomp" else twocomp_detach_scan
    return fn(leaves["cur"], leaves["v0"], leaves["w"], leaves["beta_s"],
              beta, thr, alpha, fused=True)


_LEAF_NAMES = ("cur", "v0", "w", "beta_s")
_WHOLE = {"lif": "WholeScanLIF", "twocomp": "WholeScanTwoCompartment",
          "twocomp_detach": "WholeScanTwoCompDetach"}
_PER_STEP = {"lif": "FusedLIFScan", "twocomp": "FusedTwoCompartmentScan",
             "twocomp_detach": "FusedTwoCompDetachScan"}


def _node_name(variant: str, impl: str) -> str:
    table = _WHOLE if impl == "wholescan" else _PER_STEP
    return table["lif" if variant.startswith("lif_") else variant] + "Backward"


def _run(variant: str, impl: str, t: dict, beta, thr, alpha) -> dict:
    """Forward + backward through the dispatcher under `impl`. Everything it made.

    Asserts WHICH autograd node was built. Without that, a whole-scan path that
    silently fell back would compare the oracle against itself and every test in
    this file would pass while checking nothing.
    """
    leaves = {k: t[k].detach().clone().requires_grad_(True) for k in _LEAF_NAMES if k in t}
    if impl == "wholescan":
        _poison(t["cur"])
    with ws.use_scan_impl(impl):
        s, vf = _scan(variant, leaves, beta, thr, alpha)
    assert type(s.grad_fn).__name__ == _node_name(variant, impl), (
        variant, impl, type(s.grad_fn).__name__)
    saved = s.grad_fn.saved_tensors
    out = {"spikes": s.detach(), "v_final": vf.detach(), "saved_v_pre": saved[0]}
    if _is_tc(variant):
        out["saved_vs_seq"] = saved[1]
    torch.autograd.backward([s, vf], [t["gs"], t["gv"]])
    for k, leaf in leaves.items():
        out[f"grad_{k}"] = leaf.grad
    return out


def _run_nograd(variant: str, impl: str, t: dict, beta, thr, alpha) -> dict:
    if impl == "wholescan":
        _poison(t["cur"])
    with ws.use_scan_impl(impl), torch.no_grad():
        s, vf = _scan(variant, t, beta, thr, alpha)
    assert s.grad_fn is None and not s.requires_grad
    return {"spikes": s, "v_final": vf}


def _release() -> None:
    gc.collect()
    torch.cuda.empty_cache()


# --------------------------------------------------------------------------
# 1. every output, every saved intermediate, every gradient
# --------------------------------------------------------------------------

@pytest.mark.cuda
@requires_wholescan
@pytest.mark.parametrize("shape", SHAPES, ids=lambda s: "x".join(map(str, s)))
@pytest.mark.parametrize("variant", VARIANTS)
def test_every_output_and_gradient_is_bit_identical(variant, shape):
    """The whole claim, on the grad path: old vs new, every tensor, every bit."""
    _release()
    for i, (scale, beta, thr, alpha) in enumerate(REGIMES):
        t = _inputs(variant, shape, scale, seed=1000 + 17 * i + shape[2])
        want = _run(variant, "jiterator", t, beta, thr, alpha)
        got = _run(variant, "wholescan", t, beta, thr, alpha)
        _assert_bits(got, want, (variant, shape, scale, beta, thr))

        # The test is only meaningful if the neuron fires and the gradients are
        # live: bit-identical zeros would satisfy everything above.
        rate = float(want["spikes"].mean())
        if t["cur"].numel() > 10_000:
            assert 0.01 < rate < 0.9, (variant, shape, scale, rate)
        assert float(want["grad_cur"].abs().max()) > 0.0
        if _is_tc(variant):
            assert float(want["grad_w"].abs().max()) > 0.0
            assert float(want["grad_beta_s"].abs().max()) > 0.0
        del t, want, got
    _release()


@pytest.mark.cuda
@requires_wholescan
@pytest.mark.parametrize("shape", SHAPES, ids=lambda s: "x".join(map(str, s)))
@pytest.mark.parametrize("variant", VARIANTS)
def test_the_no_grad_path_is_bit_identical(variant, shape):
    """The inference forward: no `v_pre`, no `vs_seq`, same spikes, same state.

    Against the per-timestep path under `no_grad` AND against the grad path's own
    outputs, so the `save` flag cannot change what the forward computes. `lif` and
    `twocomp_detach` have no per-timestep inference path at all; this is theirs.
    """
    _release()
    for i, (scale, beta, thr, alpha) in enumerate(REGIMES):
        t = _inputs(variant, shape, scale, seed=2000 + 17 * i + shape[2])
        want = _run_nograd(variant, "jiterator", t, beta, thr, alpha)
        got = _run_nograd(variant, "wholescan", t, beta, thr, alpha)
        _assert_bits(got, want, (variant, shape, scale, "no-grad"))
        grad_path = _run(variant, "wholescan", t, beta, thr, alpha)
        _assert_bits(got, {k: grad_path[k] for k in got}, (variant, shape, "save flag"))
        del t, want, got, grad_path
    _release()


@pytest.mark.cuda
@requires_wholescan
@pytest.mark.parametrize("shape", EDGE_SHAPES, ids=lambda s: "x".join(map(str, s)))
@pytest.mark.parametrize("variant", VARIANTS)
def test_edge_shapes_are_bit_identical(variant, shape):
    """L = 1, B = 1, d = 1.

    L = 1 is the only length at which the backward's `t == 0` branch is ALSO its
    first iteration, so `vs_prev` comes from `v0` and never from `vs_seq` -- the
    chat decoder runs that shape once per character.
    """
    scale, beta, thr, alpha = REGIMES[1]
    t = _inputs(variant, shape, scale, seed=3000 + sum(shape))
    _assert_bits(_run(variant, "wholescan", t, beta, thr, alpha),
                 _run(variant, "jiterator", t, beta, thr, alpha), (variant, shape))
    _assert_bits(_run_nograd(variant, "wholescan", t, beta, thr, alpha),
                 _run_nograd(variant, "jiterator", t, beta, thr, alpha), (variant, shape))


@pytest.mark.cuda
@requires_wholescan
@pytest.mark.parametrize("variant", VARIANTS)
def test_the_launchers_match_the_oracle_called_by_hand(variant):
    """Kernel boundary, no dispatcher and no autograd engine in between.

    `wholescan.self_check` is what decides at run time whether a variant may be
    used at all, so it is held to the same standard as the tests: it must say yes
    here, where everything above says yes.
    """
    assert ws.self_check(variant) is True


# --------------------------------------------------------------------------
# 2. NEGATIVE CONTROL -- the gate can see one ulp
# --------------------------------------------------------------------------

@pytest.mark.cuda
@requires_wholescan
@pytest.mark.parametrize("variant", HAZARD_VARIANTS)
def test_without_the_barrier_it_is_not_bit_identical(variant):
    """The kernel this project must never ship, built on purpose, and REJECTED.

    Same functor text, same wrapper, same flags; the only change is that the
    per-step trampoline is `inline` instead of `noinline`, so NVRTC sees the
    carried adjoint across the timestep boundary and re-pairs an FMA onto it.

    Three things are asserted, and the second is the one that matters:
      * the forward is untouched (the barrier is a backward-only concern);
      * `grad_cur` is NOT bit-identical to the oracle -- so `_same_bits` can see
        the defect this whole file exists to exclude;
      * and yet it IS `allclose` at the R10 gate's own tolerances (rtol 1e-4,
        atol 1e-6), so no other test in the repository could have seen it.
    """
    scale, beta, thr, alpha = REGIMES[1]
    shape = (16, 64, 48)
    t = _inputs(variant, shape, scale, seed=4242)
    want = _run(variant, "jiterator", t, beta, thr, alpha)

    gs, gv = t["gs"].contiguous(), t["gv"].contiguous()
    if _is_tc(variant):
        s, v_pre, vs_seq, vf = ws.twocomp_forward_scan(
            t["cur"], t["v0"], t["w"], t["beta_s"], beta, thr, save=True)

        def backward(barrier):
            return ws.twocomp_backward_scan(
                variant, gs, gv, v_pre, vs_seq, t["v0"], t["w"], t["beta_s"],
                beta, thr, alpha, barrier=barrier)
    else:
        reset = variant[len("lif_"):]
        s, v_pre, vf = ws.lif_forward_scan(t["cur"], t["v0"], beta, thr, reset, save=True)

        def backward(barrier):
            return ws.lif_backward_scan(gs, gv, v_pre, beta, thr, alpha, reset,
                                        barrier=barrier)

    assert _same_bits(s, want["spikes"]) and _same_bits(vf, want["v_final"])
    assert _same_bits(v_pre, want["saved_v_pre"])

    # positive control first: WITH the barrier these very calls are bit-identical,
    # so what follows is the barrier's absence and not a mistake in this test
    with_barrier = backward(True)
    assert _same_bits(with_barrier[0], want["grad_cur"])
    assert _same_bits(with_barrier[1], want["grad_v0"])

    without = backward(False)
    assert not _same_bits(without[0], want["grad_cur"]), (
        f"{variant}: the barrier-less backward came out BIT-IDENTICAL. NVRTC is no "
        "longer re-pairing the FMA this control relies on. The gate has lost its "
        "proof that it can fail -- find a new negative control before trusting it.")
    assert torch.allclose(without[0], want["grad_cur"], rtol=1e-4, atol=1e-6), (
        "the barrier-less kernel is wrong by more than rounding; this control is "
        "meant to be the one-ulp case")
    # and the run-time guard draws the same line
    assert ws.self_check(variant, barrier=False) is False
    assert ws.self_check(variant, barrier=True) is True


# --------------------------------------------------------------------------
# 3. under CUDA-graph replay
# --------------------------------------------------------------------------

def _capture(step):
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(3):
            step()
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        step()
    return graph


@pytest.mark.cuda
@requires_wholescan
@pytest.mark.parametrize("variant", VARIANTS)
def test_bit_identical_under_cuda_graph_replay(variant):
    """Forward + backward captured, replayed on data the graph has never seen.

    The trainer captures the scan, so this is the path every trained weight
    actually comes from. Replay rather than capture is compared: a launch that
    baked in a stale pointer, or ran on the wrong stream, is right during capture
    and wrong on every replay after it (risk R5 in miniature).
    """
    scale, beta, thr, alpha = REGIMES[1]
    shape = (8, 64, 48)
    static = _inputs(variant, shape, scale, seed=51)
    leaves = {k: static[k].detach().clone().requires_grad_(True)
              for k in _LEAF_NAMES if k in static}
    held = {}

    def step():
        for leaf in leaves.values():
            if leaf.grad is not None:
                leaf.grad.zero_()
        s, vf = _scan(variant, leaves, beta, thr, alpha)
        torch.autograd.backward([s, vf], [static["gs"], static["gv"]])
        held["spikes"], held["v_final"] = s.detach(), vf.detach()

    with ws.use_scan_impl("wholescan"):
        graph = _capture(step)

    fresh = _inputs(variant, shape, scale, seed=97)
    with torch.no_grad():
        for k, leaf in leaves.items():
            leaf.copy_(fresh[k])
    static["gs"].copy_(fresh["gs"])
    static["gv"].copy_(fresh["gv"])
    graph.replay()
    torch.cuda.synchronize()

    got = {"spikes": held["spikes"], "v_final": held["v_final"]}
    got.update({f"grad_{k}": leaf.grad for k, leaf in leaves.items()})
    want = _run(variant, "jiterator", fresh, beta, thr, alpha)
    _assert_bits(got, {k: want[k] for k in got}, (variant, "graph replay"))


@pytest.mark.cuda
@requires_wholescan
@pytest.mark.parametrize("variant", VARIANTS)
def test_no_grad_forward_bit_identical_under_cuda_graph_replay(variant):
    """What `snnchat.stepper` captures: the no-grad forward, state fed back in."""
    scale, beta, thr, alpha = REGIMES[1]
    shape = (8, 1, 48)   # one decode step
    static = _inputs(variant, shape, scale, seed=61)
    held = {}

    def step():
        with torch.no_grad():
            held["spikes"], held["v_final"] = _scan(variant, static, beta, thr, alpha)

    with ws.use_scan_impl("wholescan"):
        graph = _capture(step)

    fresh = _inputs(variant, shape, scale, seed=71)
    for k in ("cur", "v0"):
        static[k].copy_(fresh[k])
    graph.replay()
    torch.cuda.synchronize()
    fresh.update({k: static[k] for k in ("w", "beta_s") if k in static})
    want = _run_nograd(variant, "jiterator", fresh, beta, thr, alpha)
    _assert_bits(dict(held), want, (variant, "no-grad graph replay"))


@pytest.mark.cuda
@requires_wholescan
def test_a_capture_that_comes_first_records_the_per_timestep_path():
    """The self-check syncs, so it must never run inside a capture -- and does not.

    Every capture site in the tree warms up eagerly first, which is where the
    check really happens. This is the case where one does not: the verdict is
    unknown when the capture begins, so that call takes the per-timestep path
    (bit-identical, merely slower), says so, and the capture SUCCEEDS. The
    alternative -- compiling, loading a module and syncing inside `torch.cuda.
    graph` -- aborts the capture or poisons it.
    """
    scale, beta, thr, alpha = REGIMES[1]
    t = _inputs("lif_soft", (4, 16, 8), scale, seed=5)
    want = _run_nograd("lif_soft", "jiterator", t, beta, thr, alpha)   # and compiles it
    held = {}
    ws.clear_wholescan_cache()
    try:
        graph = torch.cuda.CUDAGraph()
        with ws.use_scan_impl("wholescan"), warnings.catch_warnings(record=True) as seen:
            warnings.simplefilter("always")
            with torch.no_grad(), torch.cuda.graph(graph):
                held["spikes"], held["v_final"] = _scan("lif_soft", t, beta, thr, alpha)
        graph.replay()
        torch.cuda.synchronize()
        assert any("being captured" in str(w.message) for w in seen), [
            str(w.message) for w in seen]
        assert ws.scan_impl_record()["verified"] == {}, "the check ran inside the capture"
        _assert_bits(dict(held), want, "capture-first")
    finally:
        ws.clear_wholescan_cache()
    # and the very next eager call verifies and takes the whole-scan path
    _assert_bits(_run_nograd("lif_soft", "wholescan", t, beta, thr, alpha), want, "after")
    assert ws.scan_impl_record()["verified"] == {"lif_soft": True}


# --------------------------------------------------------------------------
# 4. the fallback rules: never a mis-read
# --------------------------------------------------------------------------

@pytest.mark.cuda
@requires_wholescan
@pytest.mark.parametrize("variant", VARIANTS)
def test_non_contiguous_current_takes_the_per_timestep_path(variant):
    """`_CudaKernel` ignores strides. A strided `cur` must not reach it.

    The view here is a transposed buffer: same values as its contiguous copy,
    different memory order, so a kernel that read it raw would return the scan of
    a DIFFERENT current and nothing would raise. The dispatcher sends it to the
    per-timestep path instead, and the answer is the contiguous one, bit for bit.
    """
    scale, beta, thr, alpha = REGIMES[1]
    B, L, d = 6, 20, 14
    t = _inputs(variant, (B, L, d), scale, seed=88)
    strided = t["cur"].transpose(0, 1).contiguous().transpose(0, 1)
    assert not strided.is_contiguous() and torch.equal(strided, t["cur"])
    want = _run(variant, "jiterator", t, beta, thr, alpha)

    leaves = {k: t[k].detach().clone().requires_grad_(True) for k in ("v0", "w", "beta_s")
              if k in t}
    leaves["cur"] = strided.detach().requires_grad_(True)
    with ws.use_scan_impl("wholescan"):
        assert not ws.wholescan_selected(variant, leaves["cur"], leaves["v0"])
        s, vf = _scan(variant, leaves, beta, thr, alpha)
    assert type(s.grad_fn).__name__ == _node_name(variant, "jiterator")
    torch.autograd.backward([s, vf], [t["gs"], t["gv"]])
    assert _same_bits(s.detach(), want["spikes"])
    assert _same_bits(leaves["cur"].grad, want["grad_cur"])

    # and the raw launch REFUSES it rather than mis-reading it
    with pytest.raises(ValueError, match="contiguous"):
        if _is_tc(variant):
            ws.twocomp_forward_scan(strided, t["v0"], t["w"], t["beta_s"], beta, thr, True)
        else:
            ws.lif_forward_scan(strided, t["v0"], beta, thr, variant[4:], True)


@pytest.mark.cuda
@requires_wholescan
@pytest.mark.parametrize("variant", VARIANTS)
def test_strided_small_tensors_and_expanded_cotangents_are_repaired(variant):
    """`v0`, `w`, `beta_s` and the cotangents are COPIED when strided, not refused.

    They are small, a copy is bit-neutral, and refusing them would send carried-
    state evaluation down the slow path for nothing. The cotangents are the case
    that is easy to forget: `s.sum().backward()` hands the backward an EXPANDED
    tensor whose stride is 0 in every dimension.
    """
    scale, beta, thr, alpha = REGIMES[1]
    B, L, d = 5, 12, 10
    t = _inputs(variant, (B, L, d), scale, seed=123)
    wide = {k: torch.repeat_interleave(t[k], 2, dim=1) for k in ("v0", "w", "beta_s") if k in t}
    strided = {k: v[:, ::2] for k, v in wide.items()}
    assert all(not v.is_contiguous() or v.shape[1] == 1 for v in strided.values())
    assert all(torch.equal(strided[k], t[k]) for k in strided)

    def run(impl):
        leaves = {k: strided.get(k, t[k]).detach().requires_grad_(True)
                  for k in _LEAF_NAMES if k in t}
        with ws.use_scan_impl(impl):
            s, vf = _scan(variant, leaves, beta, thr, alpha)
        assert type(s.grad_fn).__name__ == _node_name(variant, impl)
        (s.sum() + vf.sum()).backward()          # expanded, stride-0 cotangents
        out = {"spikes": s.detach(), "v_final": vf.detach()}
        out.update({f"grad_{k}": leaf.grad for k, leaf in leaves.items()})
        return out

    _assert_bits(run("wholescan"), run("jiterator"), (variant, "strided small tensors"))


@pytest.mark.cuda
@requires_wholescan
def test_fp64_is_refused_on_both_paths_and_never_narrowed():
    """fp64 never reaches a raw kernel: it would be read as twice as many floats."""
    cur = torch.randn(2, 8, 4, device="cuda", dtype=torch.float64)
    v0 = torch.zeros(2, 4, device="cuda", dtype=torch.float64)
    for impl in ws.SCAN_IMPLS:
        with ws.use_scan_impl(impl), pytest.raises(TypeError):
            lif_scan(cur, v0, 0.9, 1.0, 2.0, "hard", fused=True)
    assert not ws.wholescan_selected("lif_hard", cur, v0)
    with pytest.raises(TypeError, match="fp32"):
        ws.WholeScanLIF.apply(cur, v0, 0.9, 1.0, 2.0, RESET_CODES["hard"])
    # ... while the eager reference still takes it, which is what the R10 gate's
    # float64 leg relies on
    s, _ = lif_scan_eager(cur, v0, 0.9, 1.0, 2.0, "hard")
    assert s.dtype is torch.float64


def test_cpu_tensors_take_the_eager_path_whatever_is_selected():
    """`fused=True` on CPU has always meant the eager reference. It still does."""
    g = torch.Generator().manual_seed(0)
    cur = torch.randn(3, 9, 5, generator=g)
    v0 = torch.randn(3, 5, generator=g)
    want = lif_scan_eager(cur, v0, 0.9, 0.7, 3.3, "hard")
    for impl in ws.SCAN_IMPLS:
        with ws.use_scan_impl(impl):
            assert not ws.wholescan_selected("lif_hard", cur, v0)
            got = lif_scan(cur, v0, 0.9, 0.7, 3.3, "hard", fused=True)
        assert _same_bits(got[0], want[0]) and _same_bits(got[1], want[1])


def test_shapes_beyond_the_launch_limits_are_not_selected():
    """32-bit shape arguments and a 2^31-block grid: past either, fall back."""
    def meta(*shape):
        return torch.empty(*shape, device="meta")

    assert ws._fits(meta(128, 256, 512))
    assert ws._fits(meta(1, 1, 1))
    assert not ws._fits(meta(0, 4, 4))                 # nothing to launch
    assert not ws._fits(meta(1, 2**31, 1))             # L does not fit an int
    assert not ws._fits(meta(2**31, 1, 300))           # nor B; nor the grid
    assert ws._fits(meta(2**20, 1, 2**11))             # B*d = 2^31 is fine: 2^23 blocks
    assert not ws.wholescan_selected("lif_hard", meta(2, 3, 4), meta(2, 4))


@pytest.mark.cuda
@requires_wholescan
def test_double_backward_raises_rather_than_lying():
    """Same guard, same reason, as the per-timestep Functions."""
    c = torch.randn(2, 8, 4, device="cuda", requires_grad=True)
    v0 = torch.zeros(2, 4, device="cuda", requires_grad=True)
    with ws.use_scan_impl("wholescan"):
        s, vf = lif_scan(c, v0, 0.9, 1.0, 2.0, "hard", fused=True)
    assert type(s.grad_fn).__name__ == "WholeScanLIFBackward"
    loss = (s * s).sum() + (vf * vf).sum() + (c * c * c).sum()
    with pytest.raises(RuntimeError, match="double backward"):
        torch.autograd.grad(loss, c, create_graph=True)


# --------------------------------------------------------------------------
# 5. selection, routing, and what a run records
# --------------------------------------------------------------------------

def test_the_environment_variable_selects_and_a_typo_raises(monkeypatch):
    """`SNN_SCAN_IMPL`: unset means whole-scan, `jiterator` is the exact old path.

    A value that is set and unknown RAISES. Falling back to the default would be
    a run that believes it used one implementation and used the other.
    """
    try:
        for raw, want in ((None, "wholescan"), ("", "wholescan"), ("wholescan", "wholescan"),
                          ("jiterator", "jiterator"), ("  JITerator ", "jiterator")):
            ws._impl_from_environment.cache_clear()
            if raw is None:
                monkeypatch.delenv("SNN_SCAN_IMPL", raising=False)
            else:
                monkeypatch.setenv("SNN_SCAN_IMPL", raw)
            assert ws.scan_impl() == want, (raw, ws.scan_impl())
            assert ws.scan_impl_record()["requested"] == want

        # read ONCE: a change after first use is not seen
        monkeypatch.setenv("SNN_SCAN_IMPL", "wholescan")
        assert ws.scan_impl() == "jiterator"

        ws._impl_from_environment.cache_clear()
        monkeypatch.setenv("SNN_SCAN_IMPL", "jiterater")
        with pytest.raises(ValueError, match="jiterater"):
            ws.scan_impl()
    finally:
        monkeypatch.undo()
        ws._impl_from_environment.cache_clear()


def test_use_scan_impl_nests_and_restores():
    before = ws.scan_impl()
    with ws.use_scan_impl("jiterator"):
        assert ws.scan_impl() == "jiterator"
        with ws.use_scan_impl("wholescan"):
            assert ws.scan_impl() == "wholescan"
        assert ws.scan_impl() == "jiterator"
    assert ws.scan_impl() == before
    with pytest.raises(ValueError):
        with ws.use_scan_impl("eager"):
            pass
    assert ws.scan_impl() == before


@pytest.mark.cuda
@requires_wholescan
@pytest.mark.parametrize("variant", VARIANTS)
def test_the_dispatchers_route_on_grad_mode(variant, monkeypatch):
    """With grad: the Function, saving. Without: the no-save forward, and no node.

    A no-save path taken while a gradient is wanted produces a tensor with no
    `grad_fn` and a loss that never learns; a saving path taken without grad is
    the 2-3x transient memory this path exists to remove.
    """
    ws.verify_all()      # so the self-check's own launches are not counted below
    saves = []
    launchers = ("twocomp_forward_scan",) if _is_tc(variant) else ("lif_forward_scan",)
    for name in launchers:
        real = getattr(ws, name)

        def counting(*a, _real=real, **k):
            saves.append(k.get("save", a[-1]))
            return _real(*a, **k)

        monkeypatch.setattr(ws, name, counting)

    scale, beta, thr, alpha = REGIMES[1]
    t = _inputs(variant, (2, 16, 8), scale, seed=7)
    _run_nograd(variant, "wholescan", t, beta, thr, alpha)
    assert saves == [False]
    _run(variant, "wholescan", t, beta, thr, alpha)
    assert saves == [False, True]
    with ws.use_scan_impl("jiterator"):
        _run(variant, "jiterator", t, beta, thr, alpha)
    assert saves == [False, True], "SNN_SCAN_IMPL=jiterator still reached a whole-scan kernel"


@pytest.mark.cuda
@requires_wholescan
def test_the_no_grad_path_actually_allocates_less():
    """Measured, not read off the source: one [B, L, d] tensor instead of three."""
    scale, beta, thr, alpha = REGIMES[1]
    t = _inputs("twocomp_detach", (16, 256, 256), scale, seed=11)

    def peak(fn):
        _release()
        torch.cuda.synchronize()
        base = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        out = fn()
        torch.cuda.synchronize()
        used = torch.cuda.max_memory_allocated() - base
        del out
        return used

    with ws.use_scan_impl("wholescan"):
        with_grad = peak(lambda: _scan("twocomp_detach", t, beta, thr, alpha))
        with torch.no_grad():
            without = peak(lambda: _scan("twocomp_detach", t, beta, thr, alpha))
    assert without < 0.5 * with_grad, (without, with_grad)


@pytest.mark.cuda
@requires_wholescan
def test_every_fused_arm_reaches_the_whole_scan_kernels(monkeypatch):
    """No call site changed, so every arm gets it -- asserted, not assumed.

    An arm whose prescan handed the scan a strided current would fall back
    silently: correct, and as slow as before, with nothing to say so.
    """
    from snn.config import Config
    from snn.model import build_model

    launched = []
    real = ws._launch

    def counting(name, *a, **k):
        launched.append(name)
        return real(name, *a, **k)

    monkeypatch.setattr(ws, "_launch", counting)
    ws.verify_all()
    lif_arms = ("snn", "tokenshift", "threshold", "noise", "dopamine", "interface",
                "localdopamine")
    for arch in (*lif_arms, "twocomp", "twocomp_threshold", "twocomp_detach"):
        cfg = Config(arch=arch, d_model=32, n_layers=2, vocab_size=205, device="cuda",
                     seq_len=16, batch_size=4)
        model = build_model(cfg).cuda()
        allocate = getattr(model, "allocate_noise", None)
        if allocate is not None:
            allocate(4, 16)
        idx = torch.randint(0, 205, (4, 16), device="cuda")
        stem = "wholescan_lif" if arch in lif_arms else "wholescan_tc"
        launched.clear()
        with ws.use_scan_impl("wholescan"):
            logits, _, _ = model(idx, None)
            logits.sum().backward()
        assert launched.count(f"{stem}_fwd") == 2 and launched.count(f"{stem}_bwd") == 2, (
            arch, launched)
        launched.clear()
        with ws.use_scan_impl("jiterator"):
            model(idx, None)[0].sum().backward()
        assert launched == [], (arch, launched)


# --------------------------------------------------------------------------
# 6. source assembly -- runs without a GPU
# --------------------------------------------------------------------------

def _backward_sources(barrier: bool = True):
    for reset in RESET_MODES:
        yield (f"lif_{reset}", "lif_backward", "lif_step", backward_source(reset),
               ws.lif_backward_scan_source(reset, barrier))
    yield ("twocomp", "tc_backward", "tc_step", twocomp_backward_source(),
           ws.twocomp_backward_scan_source("twocomp", barrier))
    yield ("twocomp_detach", "tcd_backward", "tcd_step", twocomp_detach_backward_source(),
           ws.twocomp_backward_scan_source("twocomp_detach", barrier))


def test_the_run_time_comparison_is_bitwise_not_numeric():
    """`self_check` decides with `_bits_equal`, so it has to see what `==` cannot.

    -0.0 == +0.0 is True and their bit patterns differ; a NaN is unequal to itself
    and bit-identical to itself. `torch.equal` gets both backwards for this purpose.
    """
    pz, nz = torch.tensor([0.0, 1.5]), torch.tensor([-0.0, 1.5])
    assert torch.equal(pz, nz) and not ws._bits_equal(pz, nz)
    assert not _same_bits(pz, nz)
    nan = torch.tensor([float("nan"), 2.0])
    assert not torch.equal(nan, nan) and ws._bits_equal(nan, nan.clone())
    assert not ws._bits_equal(torch.zeros(2, 3), torch.zeros(3, 2))


def test_the_functor_text_is_the_committed_text():
    """Pasted, never retyped: the committed source appears verbatim, exactly once.

    This is what makes `scripts/audit/08_mutation_campaign.py`'s functor mutations
    reach the whole-scan kernels without a second list of them.
    """
    for reset in RESET_MODES:
        assert ws.lif_forward_scan_source(reset).count(forward_source(reset)) == 1
    assert ws.twocomp_forward_scan_source().count(twocomp_forward_source()) == 1
    for name, _functor, _step, committed, assembled in _backward_sources():
        assert assembled.count(committed) == 1, name
        assert assembled.startswith(committed), name


def test_every_backward_is_behind_the_noinline_trampoline():
    """ALL SIX, not only the three that need it today.

    The kernel body must call the trampoline and must not call the functor: a
    direct call is an inlined call, and an inlined step is the one-ulp defect.
    """
    for name, functor, step, _committed, assembled in _backward_sources():
        kernel = assembled[assembled.index('extern "C"'):]
        assert f"{step}(" in kernel, name
        assert functor not in kernel, f"{name}: the time loop calls the functor directly"
        assert assembled.count("__attribute__((noinline))") == 1, name
        assert f"__attribute__((noinline)) void {step}(" in assembled, name
        assert "@" not in assembled, f"{name}: an unfilled template token"


def test_the_negative_control_really_is_barrier_less():
    for name, _functor, step, _committed, assembled in _backward_sources(barrier=False):
        assert "noinline" not in assembled, name
        assert f"inline void {step}(" in assembled, name


def test_the_forward_kernels_keep_the_rounding_contract_and_index_in_64_bits():
    sources = [ws.lif_forward_scan_source(r) for r in RESET_MODES]
    sources += [ws.twocomp_forward_scan_source()]
    sources += [assembled for *_rest, assembled in _backward_sources()]
    for src in sources:
        assert "const long long k = base + (long long)t * d;" in src
        assert "(long long)blockIdx.x * blockDim.x" in src
        assert "@" not in src
    for src in sources[:5]:
        assert "__fadd_rn(__fmul_rn(" in src and "noinline" not in src
    # "detached" costs no second forward compile: the cache is keyed on the text
    assert ws.lif_forward_scan_source("detached") == ws.lif_forward_scan_source("hard")


def test_the_compile_flags_are_jiterators_and_the_arch_is_read_from_the_device(monkeypatch):
    """The flags are part of the identity argument, so they are pinned.

    No `--fmad` in either direction: contraction is ON by NVRTC's default in both
    paths, and that shared default is why the functor text compiles to the same
    arithmetic in both.
    """
    for cc, arch in (((8, 6), "sm_86"), ((12, 0), "sm_120")):
        monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None, cc=cc: cc)
        assert ws._nvrtc_flags(0) == [
            "--std=c++20", f"--gpu-architecture={arch}", "-default-device", "-DNDEBUG"]


@pytest.mark.cuda
@requires_wholescan
def test_a_compile_failure_carries_the_nvrtc_log():
    with pytest.raises(RuntimeError) as err:
        ws._compile_ptx("this is not CUDA C++;", "broken", torch.cuda.current_device())
    assert "broken.cu" in str(err.value) and "error" in str(err.value).lower()


@pytest.mark.cuda
@requires_wholescan
@pytest.mark.parametrize("cuda_home", [None, "Z:/no/such/toolkit"])
def test_it_needs_no_cuda_toolkit_and_leaves_the_environment_alone(cuda_home, tmp_path):
    """`torch.cuda._compile_kernel` raises here without CUDA_HOME; this must not care.

    A fresh process, because the question is about process-wide state: with the
    variable unset and with it pointing at nothing, all six variants must verify,
    and `os.environ` must come out exactly as it went in.
    """
    env = {k: v for k, v in os.environ.items() if k not in ("CUDA_HOME", "CUDA_PATH")}
    if cuda_home is not None:
        env["CUDA_HOME"] = cuda_home
    code = (
        "import os, sys, json\n"
        f"sys.path.insert(0, {_SRC!r})\n"
        "from snn.config import seed_everything\n"
        "before = dict(os.environ)\n"
        "import snn.wholescan as ws\n"
        "out = {'verified': ws.verify_all(), 'env_unchanged': dict(os.environ) == before,\n"
        "       'cuda_home': os.environ.get('CUDA_HOME')}\n"
        "print('RESULT' + json.dumps(out))\n"
    )
    proc = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True,
                          text=True, timeout=300, cwd=str(tmp_path))
    assert proc.returncode == 0, proc.stderr[-2000:]
    out = json.loads(proc.stdout.split("RESULT", 1)[1])
    assert out["verified"] == dict.fromkeys(VARIANTS, True), out
    assert out["env_unchanged"] is True and out["cuda_home"] == cuda_home, out


@pytest.mark.cuda
@requires_wholescan
def test_the_self_check_does_not_touch_the_global_rng():
    """Enabling the path must not change any later random number in the process.

    The noise arm draws from the CUDA stream and the chat sampler from both. A
    self-check that drew its inputs from either would make `SNN_SCAN_IMPL` change
    results with no kernel involved at all.
    """
    ws.clear_wholescan_cache()
    cpu, cuda = torch.get_rng_state().clone(), torch.cuda.get_rng_state().clone()
    assert ws.verify_all() == dict.fromkeys(VARIANTS, True)
    assert torch.equal(cpu, torch.get_rng_state())
    assert torch.equal(cuda, torch.cuda.get_rng_state())


# --------------------------------------------------------------------------
# 7. a real Trainer: the weights themselves
# --------------------------------------------------------------------------

_CORPUS = os.environ.get("SNN_TEST_CORPUS", "enwik8")
_N_STEPS = 40


def _real_data_dir(name: str) -> str | None:
    root = os.environ.get("SNN_TEST_DATA_DIR") or str(_REPO / "data")
    p = Path(root) / name
    if (p / "train.bin").exists() and (p / "meta.json").exists():
        return root
    return None


def _synthetic_root(tmp_path_factory) -> Path:
    """Random bytes in the shape `snn.data.Corpus` insists on; see
    `tests/test_graph_equivalence.py`, whose session-wide copy this shares. What is
    checked here is that two runs AGREE, which does not need the data to be real.
    """
    from snn.data import CORPORA
    meta_version = getattr(sys.modules["snn.data"], "META_VERSION", 1)
    spec = CORPORA[_CORPUS]
    root = tmp_path_factory.getbasetemp() / "synthetic_corpus"
    d = root / _CORPUS
    if (d / "meta.json").exists():
        return root
    d.mkdir(parents=True, exist_ok=True)
    vocab_size = int(getattr(spec, "vocab_size", 0) or {"enwik8": 205, "text8": 27}[_CORPUS])
    rng = np.random.default_rng(0)
    for which, n in zip(("train", "val", "test"), spec.split):
        ids = rng.integers(0, vocab_size, size=int(n), dtype=np.uint8)
        ids[:vocab_size] = np.arange(vocab_size, dtype=np.uint8)
        ids.tofile(d / f"{which}.bin")
    meta = {
        "meta_version": meta_version, "corpus": _CORPUS,
        "vocab": list(range(vocab_size)), "vocab_size": vocab_size,
        "split_sizes": dict(zip(("train", "val", "test"), (int(x) for x in spec.split))),
        "raw_bytes": int(sum(spec.split)),
        "zip_sha256": getattr(spec, "zip_sha256", ""),
        "raw_sha256": getattr(spec, "raw_sha256", ""),
        "url": "synthetic: random bytes, NOT the real corpus", "synthetic": True,
    }
    (d / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return root


@pytest.fixture(scope="session")
def corpus_dir(tmp_path_factory) -> str:
    real = _real_data_dir(_CORPUS)
    if real is not None:
        return real
    try:
        from snn.data import Corpus
        root = _synthetic_root(tmp_path_factory)
        Corpus(_CORPUS, str(root))
    except Exception as exc:
        pytest.skip(f"no usable corpus ({type(exc).__name__}: {exc}); "
                    f"run scripts/data/download_corpus.py")
    return str(root)


def _train(cfg, impl: str, monkeypatch) -> dict:
    """`_N_STEPS` real optimiser steps under `impl`; everything that was learned."""
    from snn.train import Trainer

    launched = []
    real = ws._launch
    monkeypatch.setattr(ws, "_launch", lambda name, *a, **k: (
        launched.append(name), real(name, *a, **k))[1])
    with ws.use_scan_impl(impl):
        tr = Trainer(cfg)
        losses = [tr.step()["loss"] for _ in range(_N_STEPS)]
        out = {
            "losses": losses,
            "captured": tr.graph is not None,
            "params": [p.detach().clone() for p in tr.params],
            "optim": [{k: v.detach().clone() for k, v in tr.opt.state[p].items()
                       if torch.is_tensor(v)} for p in tr.params],
            "record": json.loads((tr.run_dir / "environment.json").read_text("utf-8")),
            "scan_launches": [n for n in launched if n != "wholescan_probe"],
        }
    monkeypatch.undo()
    del tr
    _release()
    return out


@pytest.mark.cuda
@requires_wholescan
@pytest.mark.parametrize("cuda_graph", [True, False], ids=["graphed", "eager"])
@pytest.mark.parametrize("arch,reset", [("snn", "hard"), ("snn", "none"),
                                        ("twocomp", "hard"), ("twocomp_detach", "hard")])
def test_a_real_trainer_learns_the_same_weights(arch, reset, cuda_graph, corpus_dir,
                                                tmp_path, monkeypatch):
    """The end of the argument: same seed, same data, both settings, SAME WEIGHTS.

    Forty optimiser steps through the real `Trainer` -- warm-up, capture, the fp64
    clip, AdamW -- and then every parameter and every AdamW moment compared as bit
    patterns. One differing ulp in one gradient at step 1 would be amplified by
    every step after it, so this is the test with the least room to be fooled.
    `reset="none"` is here because it is the plain LIF's hazard variant and no
    committed run uses it, so nothing else would ever train through it.
    """
    from snn.config import Config

    def cfg(tag):
        return Config(
            corpus=_CORPUS, data_dir=corpus_dir, arch=arch, reset=reset,
            seq_len=32, batch_size=8, d_model=64, n_layers=2,
            lr=3e-3, warmup_steps=4, max_steps=64, lr_schedule="cosine", grad_clip=1.0,
            device="cuda", seed=0, deterministic=True, cuda_graph=cuda_graph,
            out_dir=str(tmp_path / tag), run_name="t",
            eval_every=0, ckpt_every=0, log_every=0,
        )

    old = _train(cfg("jiterator"), "jiterator", monkeypatch)
    new = _train(cfg("wholescan"), "wholescan", monkeypatch)

    assert old["captured"] is cuda_graph and new["captured"] is cuda_graph
    # the comparison is only a comparison if the two runs really took two paths
    assert old["scan_launches"] == [], old["scan_launches"][:4]
    assert new["scan_launches"], "the whole-scan Trainer never launched a whole-scan kernel"
    assert old["record"]["scan_impl"]["requested"] == "jiterator"
    assert new["record"]["scan_impl"]["requested"] == "wholescan"
    assert new["record"]["scan_impl"]["verified"] == dict.fromkeys(VARIANTS, True)

    assert [repr(x) for x in new["losses"]] == [repr(x) for x in old["losses"]]
    assert new["losses"][-1] < new["losses"][0], "nothing was learned; the test is vacuous"
    for i, (a, b) in enumerate(zip(new["params"], old["params"])):
        assert _same_bits(a, b), f"parameter {i}: {_describe(a, b)}"
    for i, (sa, sb) in enumerate(zip(new["optim"], old["optim"])):
        assert sa.keys() == sb.keys() and sa, i
        for k in sa:
            assert _same_bits(sa[k].float(), sb[k].float()), (
                f"AdamW {k} of parameter {i}: {_describe(sa[k].float(), sb[k].float())}")
