"""Phase-2 metrics: bits-per-character, firing rates, and CUDA kernel counting.

Two of these are ordinary bookkeeping. The third is not: on this machine
wall-clock is ~14.5 us x (CUDA kernels issued) and is nearly independent of
tensor size (01_reconnaissance.md §3.3, §3.4). That makes "kernels per timestep"
a *measurable prediction* of the architecture rather than an implementation
detail, so `count_cuda_kernels` exists to turn the cost model into something a
test can assert (assumption A1, gate in tests/test_kernel_count.py) and a report
table can quote (scripts/bench/kernel_count.py).
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from typing import Any

import torch
from torch import Tensor

__all__ = [
    "bits_per_char",
    "count_cuda_kernels",
    "summarise_firing_rates",
]

_LN2 = math.log(2.0)


def bits_per_char(total_nats: float, n_chars: int) -> float:
    """Convert an accumulated natural-log loss into bits per character.

    `total_nats` is the SUM of the per-character cross-entropies in nats (i.e.
    `F.cross_entropy(..., reduction="sum")` accumulated over the split), not a
    mean. Keeping the sum and the count separate is what makes the two evaluation
    protocols comparable: protocol B scores every character exactly once, and a
    running mean of per-batch means would silently reweight the trailing batch.
    """
    if n_chars <= 0:
        raise ValueError(f"n_chars must be positive, got {n_chars}")
    return float(total_nats) / n_chars / _LN2


def summarise_firing_rates(rates: Sequence[Tensor] | Sequence[float]) -> list[float]:
    """Per-layer firing rates as plain floats, for JSONL logging.

    Calls `.item()`, so this is a host synchronisation: never call it inside a
    captured CUDA graph or in the hot path of a training step. The trainer copies
    the rate tensors into a static buffer and only summarises on logging steps.
    """
    out: list[float] = []
    for r in rates:
        out.append(float(r.item()) if isinstance(r, Tensor) else float(r))
    return out


# ---------------------------------------------------------------------------
# Kernel counting
# ---------------------------------------------------------------------------

# kineto tags every device-side activity. These are the three that appear for
# this workload; anything else on the CUDA device is treated as a kernel so an
# unexpected activity type inflates the count rather than hiding from it.
_MEMCPY = "gpu_memcpy"
_MEMSET = "gpu_memset"
_KERNEL = "kernel"

# Host-side launch calls. jiterator kernels go out through the *driver* API
# (cuLaunchKernel), not the runtime one, so counting only cudaLaunchKernel would
# undercount exactly the kernels this project cares most about.
_LAUNCH_CALLS = frozenset(
    {
        "cudaLaunchKernel",
        "cudaLaunchKernelExC",
        "cudaLaunchKernelC",
        "cudaGraphLaunch",
        "cuLaunchKernel",
        "cuLaunchKernelEx",
        "cuGraphLaunch",
    }
)


def _mangled_components(name: str) -> list[str]:
    """Length-prefixed identifiers out of an Itanium-mangled symbol.

    `_ZN2at6native29vectorized_elementwise_kernelI...` encodes each name part as
    <length><chars>, so walking the digits recovers `at`, `native`,
    `vectorized_elementwise_kernel`. Template arguments produce some junk
    components; the caller filters.
    """
    out: list[str] = []
    i, n = 0, len(name)
    while i < n:
        if not name[i].isdigit():
            i += 1
            continue
        j = i
        while j < n and name[j].isdigit():
            j += 1
        length = int(name[i:j])
        if 0 < length <= n - j:
            out.append(name[j : j + length])
            i = j + length
        else:
            i = j
    return out


def _short_kernel_name(name: str) -> str:
    """Best-effort readable label for a mangled device symbol.

    Kineto reports the raw C++ symbol. The full name is kept in `by_name`; this
    is cosmetic, for report tables, and is allowed to fall back to the raw name.
    """
    if not name.startswith("_Z"):
        return name
    parts = _mangled_components(name)
    for part in parts:
        if "kernel" in part:
            return part
    for part in parts:
        if part in ("at", "native", "cuda", "detail") or part.startswith("_GLOBAL__N_"):
            continue
        return part
    return name


def count_cuda_kernels(fn: Callable[[], Any], warmup: int = 3) -> dict:
    """Count the CUDA kernels one call to `fn` issues.

    `warmup` calls run first, outside the profiled region, so that lazy
    compilation (jiterator's ~0.17 s NVRTC compile, §3.11 C1), cuBLAS handle
    creation and the caching allocator's first-touch behaviour are not charged to
    the measurement. Then exactly one call is profiled.

    Returns {'total': n, 'by_name': {...}} plus diagnostics:

        total         device kernels launched (memcpy/memset excluded)
        by_name       {mangled kernel symbol: count}
        by_short_name {readable label: count}, for report tables
        memcpy/memset device-side copies and fills, counted but not in `total`
        launch_calls  host-side cudaLaunchKernel* calls, a cross-check on `total`
        warmup        echoed back so a recorded result is self-describing

    memcpy is excluded from `total` deliberately: it is billed at the same
    ~14.5 us launch cost, but a *copy* appearing in the inner loop is a different
    bug from a *kernel* appearing there, and collapsing them would hide it.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("count_cuda_kernels requires CUDA")

    from torch.autograd import DeviceType
    from torch.profiler import ProfilerActivity, profile

    for _ in range(max(0, int(warmup))):
        fn()
    torch.cuda.synchronize()

    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        fn()
        # Inside the profiled region so every launched kernel has completed and
        # been flushed by the time the profiler stops. A sync is a runtime call,
        # not a kernel, so it does not perturb the count.
        torch.cuda.synchronize()

    by_name: dict[str, int] = {}
    by_short: dict[str, int] = {}
    total = memcpy = memset = launches = 0

    for evt in prof.events():
        activity = getattr(evt, "activity_type", None)
        if evt.device_type == DeviceType.CUDA:
            name = evt.name
            if activity == _MEMCPY or name.startswith("Memcpy"):
                memcpy += 1
            elif activity == _MEMSET or name.startswith("Memset"):
                memset += 1
            else:
                total += 1
                by_name[name] = by_name.get(name, 0) + 1
                short = _short_kernel_name(name)
                by_short[short] = by_short.get(short, 0) + 1
        elif evt.name in _LAUNCH_CALLS:
            launches += 1

    return {
        "total": total,
        "by_name": by_name,
        "by_short_name": by_short,
        "memcpy": memcpy,
        "memset": memset,
        "launch_calls": launches,
        "warmup": int(warmup),
    }
