"""Assumption A1: kernel count is the cost model, and the fused scan hits its floor.

01_reconnaissance.md §3.4 measured wall-clock to be dead linear in the number of
CUDA launches at 14.5 us +/- 0.7 us each, across a 32x range, with tensor size
almost absent from the equation. That promoted "kernels issued per timestep" from
an implementation detail to *the* performance metric of this architecture
(§8.1.1), and the whole design of `snn/neuron.py` follows from it.

A claim that load-bearing should not be left as a comment. These tests measure
the counts with `torch.profiler` and assert them, so that a future edit which
quietly adds an elementwise op to the time loop -- a clamp, a cast, a detach that
materialises -- fails here instead of showing up as a 30% slower training run
that nobody attributes to anything.

Counts are read from the profiler's CUDA-side events, excluding memcpy and
memset, so they are launches that actually reached the device rather than ATen
op invocations.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest
import torch

# See note in test_surrogate.py: src/ on sys.path, spec §1.
_SRC = str(Path(__file__).resolve().parents[1] / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from snn.kernels import jiterator_available  # noqa: E402
from snn.neuron import lif_scan, lif_scan_eager  # noqa: E402

requires_fused = pytest.mark.skipif(
    not (torch.cuda.is_available() and jiterator_available()),
    reason="needs a CUDA device with a working jiterator",
)

BETA, THR, ALPHA, RESET = 0.5, 1.0, 2.0, "hard"

# The cost model's constant (§3.4). Used only as a sanity band, not as a target:
# the point of A1 is that per-kernel cost is roughly size-independent, not that
# it is 14.5 us to three figures on any given driver build.
US_PER_KERNEL = 14.5

# Reported so the numbers in the Phase-2 write-up come from a run, not a memory.
MEASURED: dict[str, float] = {}


def count_cuda_kernels(fn, warmup: int = 3) -> dict:
    """{'total': n, 'by_name': {...}} for one call of `fn`.

    Mirrors the signature `snn.metrics.count_cuda_kernels` is specified to have
    (spec §9) but is implemented locally, so this gate does not depend on another
    module being finished. `test_matches_metrics_helper` below checks the two
    agree once `snn.metrics` exists.
    """
    from torch.autograd import DeviceType
    from torch.profiler import ProfilerActivity, profile

    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        fn()
        torch.cuda.synchronize()

    by_name: dict[str, int] = {}
    for evt in prof.key_averages():
        if evt.device_type != DeviceType.CUDA:
            continue
        if "emcpy" in evt.key or "emset" in evt.key:
            continue
        by_name[evt.key] = by_name.get(evt.key, 0) + evt.count
    return {"total": sum(by_name.values()), "by_name": by_name}


def _timeit(fn, iters=10, warmup=3) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters


def _tensors(B, L, d, requires_grad=False):
    cur = torch.randn(B, L, d, device="cuda", requires_grad=requires_grad)
    v0 = torch.zeros(B, d, device="cuda", requires_grad=requires_grad)
    return cur, v0


# --------------------------------------------------------------------------
# forward
# --------------------------------------------------------------------------

@pytest.mark.cuda
@requires_fused
def test_fused_forward_is_one_kernel_per_timestep():
    """One kernel per timestep is the floor for a hard-threshold neuron.

    The scan must issue exactly L fused kernels plus a small constant for the two
    `torch.stack` calls at the end -- constant in L, because `torch.cat` batches
    its inputs rather than launching once per tensor. If that constant ever
    became O(L) the whole architecture argument in spec §0 would collapse.
    """
    B, L, d = 8, 128, 128
    cur, v0 = _tensors(B, L, d)
    # Precondition, not decoration: see test_fill_uninitialized_memory_is_off.
    # With the fill on, this test measures 4.05 kernels/timestep and fails, and
    # the cause is three tensor allocations rather than anything in the scan.
    assert torch.utils.deterministic.fill_uninitialized_memory is False, (
        "fill_uninitialized_memory leaked back on; kernel counts are unmeasurable"
    )
    with torch.no_grad():
        got = count_cuda_kernels(
            lambda: lif_scan(cur, v0, BETA, THR, ALPHA, RESET, fused=True)
        )
    per_step = got["total"] / L
    MEASURED["fused_forward_per_step"] = per_step
    MEASURED["fused_forward_total_L128"] = got["total"]

    lif_kernels = sum(v for k, v in got["by_name"].items() if "lif_forward" in k)
    assert lif_kernels == L, (L, got["by_name"])
    # the residue is the stacks; allow a generous constant but not a per-step one
    assert got["total"] - L <= 8, got["by_name"]
    assert per_step < 1.25, per_step


@pytest.mark.cuda
@requires_fused
def test_fill_uninitialized_memory_is_off_under_determinism():
    """The determinism setting that silently costs 2x, pinned so it cannot return.

    `torch.use_deterministic_algorithms(True)` also enables
    `torch.utils.deterministic.fill_uninitialized_memory`, which emits one extra
    fill kernel per uninitialised allocation. Our fused LIF is a THREE-output
    jiterator kernel called once per timestep, so the flag turns the project's
    one-launch-per-timestep floor into four -- on the exact path spec §0 exists to
    protect.

    Measured on this box (B=128, L=256, d=512, fused fwd+bwd):
        fill on   4.05 kernels/timestep   29.99 ms
        fill off  1.03 kernels/timestep   15.18 ms

    `seed_everything(deterministic=True)` therefore turns the fill off, which is
    sound because every kernel here is elementwise over its full output shape:
    nothing is ever read before it is written. This test asserts both halves --
    that the flag is off after seeding, and that it really is what moves the count
    -- so that a future torch upgrade flipping the default back is caught here
    rather than as an unexplained 2x slowdown.
    """
    from snn.config import seed_everything

    prev_det = torch.are_deterministic_algorithms_enabled()
    prev_fill = torch.utils.deterministic.fill_uninitialized_memory
    try:
        seed_everything(0, deterministic=True)
        assert torch.are_deterministic_algorithms_enabled() is True
        assert torch.utils.deterministic.fill_uninitialized_memory is False

        B, L, d = 8, 128, 128
        cur, v0 = _tensors(B, L, d)

        def scan():
            with torch.no_grad():
                lif_scan(cur, v0, BETA, THR, ALPHA, RESET, fused=True)

        off = count_cuda_kernels(scan)["total"] / L
        torch.utils.deterministic.fill_uninitialized_memory = True
        on = count_cuda_kernels(scan)["total"] / L

        MEASURED["fill_off_per_step"] = off
        MEASURED["fill_on_per_step"] = on

        assert off < 1.25, off
        # The claim is that the fill is the cause. If a torch release stops
        # filling jiterator outputs this becomes ~1.0 and the test fails -- at
        # which point the mitigation is no longer needed and this test should be
        # deleted, deliberately, rather than the flag being re-enabled by accident.
        assert on > 3.0, (
            f"expected ~4 kernels/timestep with the fill on, measured {on}. "
            "If torch no longer fills jiterator outputs, drop the mitigation in "
            "snn.config.seed_everything and delete this test."
        )
    finally:
        torch.utils.deterministic.fill_uninitialized_memory = prev_fill
        torch.use_deterministic_algorithms(prev_det, warn_only=True)


@pytest.mark.cuda
@requires_fused
def test_eager_forward_issues_at_least_six_kernels_per_timestep():
    """The reference path costs ~13 kernels/step, as §3.4 predicted.

    Spec §12 only requires >= 6. The measured figure is reported so the Phase-2
    write-up can quote the real number and so a change that accidentally made the
    reference *cheaper* -- which would mean it stopped computing the surrogate --
    is visible.
    """
    B, L, d = 8, 128, 128
    cur, v0 = _tensors(B, L, d)
    got = count_cuda_kernels(
        lambda: lif_scan_eager(cur, v0, BETA, THR, ALPHA, RESET)
    )
    per_step = got["total"] / L
    MEASURED["eager_forward_per_step"] = per_step
    MEASURED["eager_forward_total_L128"] = got["total"]
    assert per_step >= 6.0, (per_step, got["by_name"])


@pytest.mark.cuda
@requires_fused
def test_fused_beats_eager_by_at_least_six_times_on_kernel_count():
    """The ratio is what the cost model converts into wall-clock.

    Measured here rather than read out of `MEASURED`. An earlier version pulled
    both figures from the module global and called `pytest.skip` if they were
    absent, which meant the assertion silently vanished under `-k`, `--lf`, or
    any xdist sharding that did not happen to put all three tests in one worker.
    A gate that skips itself when run alone is not a gate.
    """
    B, L, d = 8, 128, 128
    cur, v0 = _tensors(B, L, d)
    with torch.no_grad():
        fused = count_cuda_kernels(
            lambda: lif_scan(cur, v0, BETA, THR, ALPHA, RESET, fused=True)
        )["total"] / L
        eager = count_cuda_kernels(
            lambda: lif_scan_eager(cur, v0, BETA, THR, ALPHA, RESET)
        )["total"] / L
    MEASURED["forward_kernel_ratio"] = eager / fused
    assert eager / fused >= 6.0, (eager, fused)


# --------------------------------------------------------------------------
# forward + backward
# --------------------------------------------------------------------------

def _fwd_bwd(cur, v0, fused: bool):
    if cur.grad is not None:
        cur.grad = None
        v0.grad = None
    scan = lif_scan_eager
    s, v = (
        lif_scan(cur, v0, BETA, THR, ALPHA, RESET, fused=True)
        if fused
        else scan(cur, v0, BETA, THR, ALPHA, RESET)
    )
    (s.sum() + v.sum()).backward()


@pytest.mark.cuda
@requires_fused
def test_fused_backward_is_one_kernel_per_timestep():
    """Backward must also be one launch per step, or half the win is lost.

    BPTT through the scan is another L-step sequential loop, so the same floor
    applies to it. Two kernels per timestep for the whole fwd+bwd pair is the
    target; the eager path needs ~27.
    """
    B, L, d = 8, 128, 128
    cur, v0 = _tensors(B, L, d, requires_grad=True)
    fused = count_cuda_kernels(lambda: _fwd_bwd(cur, v0, True))
    eager = count_cuda_kernels(lambda: _fwd_bwd(cur, v0, False))
    MEASURED["fused_fwdbwd_per_step"] = fused["total"] / L
    MEASURED["eager_fwdbwd_per_step"] = eager["total"] / L
    MEASURED["fwdbwd_kernel_ratio"] = eager["total"] / fused["total"]

    lif_bwd = sum(v for k, v in fused["by_name"].items() if "lif_backward" in k)
    assert lif_bwd == L, (L, fused["by_name"])
    assert fused["total"] / L < 2.5, fused["by_name"]
    assert eager["total"] / L >= 12.0, eager["by_name"]


# --------------------------------------------------------------------------
# A1 itself: does the 14.5 us/kernel constant hold for this workload?
# --------------------------------------------------------------------------

@pytest.mark.cuda
@requires_fused
def test_cost_model_predicts_eager_wall_clock():
    """A1: wall-clock / kernels should land near the §3.4 launch constant.

    Asserted as a band (5-40 us) rather than a point. The claim under test is
    that the eager scan is launch-bound -- that its cost is set by how many
    kernels it issues and not by how much arithmetic they do -- so the same
    per-kernel figure must hold at two widths that differ by 16x in work. A
    narrow assertion on 14.5 us would be measuring the driver build and the clock
    state, which is not the claim.
    """
    for d in (128, 2048):
        B, L = 8, 64
        cur, v0 = _tensors(B, L, d)
        with torch.no_grad():
            n = count_cuda_kernels(
                lambda: lif_scan_eager(cur, v0, BETA, THR, ALPHA, RESET)
            )["total"]
            dt = _timeit(lambda: lif_scan_eager(cur, v0, BETA, THR, ALPHA, RESET))
        us = dt * 1e6 / n
        MEASURED[f"eager_us_per_kernel_d{d}"] = us
        assert 5.0 < us < 40.0, (d, us, n)

    lo = MEASURED["eager_us_per_kernel_d128"]
    hi = MEASURED["eager_us_per_kernel_d2048"]
    # 16x the work per kernel must not cost 2x the time, or it is not launch-bound
    assert hi / lo < 2.0, (lo, hi)


@pytest.mark.cuda
@requires_fused
def test_fused_speedup_at_the_baseline_configuration():
    """Assumption A9, measured at the spec §13 headline shape (B=128, L=256, d=512).

    §3.11 C1 measured 4.0-4.9x for a fused LIF in isolation. This is the same
    measurement on the real scan, forward-only and forward+backward, and it is
    the number the Phase-2 report should quote for the neuron path.
    """
    B, L, d = 128, 256, 512
    cur, v0 = _tensors(B, L, d)
    with torch.no_grad():
        tf = _timeit(lambda: lif_scan(cur, v0, BETA, THR, ALPHA, RESET, fused=True))
        te = _timeit(lambda: lif_scan_eager(cur, v0, BETA, THR, ALPHA, RESET))
    MEASURED["fwd_ms_fused"] = tf * 1e3
    MEASURED["fwd_ms_eager"] = te * 1e3
    MEASURED["fwd_speedup"] = te / tf

    cur_g, v0_g = _tensors(B, L, d, requires_grad=True)
    tf2 = _timeit(lambda: _fwd_bwd(cur_g, v0_g, True), iters=5, warmup=2)
    te2 = _timeit(lambda: _fwd_bwd(cur_g, v0_g, False), iters=5, warmup=2)
    MEASURED["fwdbwd_ms_fused"] = tf2 * 1e3
    MEASURED["fwdbwd_ms_eager"] = te2 * 1e3
    MEASURED["fwdbwd_speedup"] = te2 / tf2

    assert te / tf > 2.0, MEASURED
    assert te2 / tf2 > 2.0, MEASURED


# --------------------------------------------------------------------------
# integration with snn.metrics, once that module lands
# --------------------------------------------------------------------------

@pytest.mark.cuda
@requires_fused
def test_matches_metrics_helper():
    """`snn.metrics.count_cuda_kernels` must agree with the local counter.

    Skipped until `snn/metrics.py` exists. Kept because spec §9 makes that helper
    the thing `scripts/bench/kernel_count.py` uses to produce the report's table,
    and a table that disagrees with the gate would be worse than no table.
    """
    metrics = pytest.importorskip("snn.metrics")
    if not hasattr(metrics, "count_cuda_kernels"):
        pytest.skip("snn.metrics.count_cuda_kernels not implemented yet")
    B, L, d = 8, 64, 64
    cur, v0 = _tensors(B, L, d)

    def run():
        with torch.no_grad():
            lif_scan(cur, v0, BETA, THR, ALPHA, RESET, fused=True)

    mine = count_cuda_kernels(run)["total"]
    theirs = metrics.count_cuda_kernels(run)["total"]
    assert abs(mine - theirs) <= 2, (mine, theirs)


def test_report_measured_numbers():
    """Not an assertion: prints the measured table with `pytest -s`.

    Ordered last so every other test has populated MEASURED.
    """
    if not MEASURED:
        pytest.skip("no measurements collected (CPU-only run)")
    print("\n--- measured kernel counts and timings ---")
    for k in sorted(MEASURED):
        print(f"  {k:34s} {MEASURED[k]:.3f}")
