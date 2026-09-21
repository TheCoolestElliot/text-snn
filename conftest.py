"""Pytest bootstrap: make `src/` importable and register shared markers.

The package lives under `src/snn/` and is not pip-installed, so `import snn`
only works if `src/` is on sys.path. Doing it here rather than in each test file
means the tests import the same way the scripts do.

`src/` is prepended, not appended, so a stale copy of `snn` elsewhere on the path
cannot shadow the working tree -- which is exactly the sort of thing that makes a
"reproducible" result irreproducible.
"""

from __future__ import annotations

import os
import sys

import pytest

SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

#: The two implementations of the fused scan (`snn.wholescan.SCAN_IMPLS`), spelled
#: out rather than imported so that collecting the suite does not import torch.
SCAN_IMPLS = ("jiterator", "wholescan")


def pytest_configure(config) -> None:
    config.addinivalue_line(
        "markers", "cuda: test requires a working CUDA device; skipped on CPU-only machines"
    )
    config.addinivalue_line(
        "markers", "data: test requires the prepared corpus under data/"
    )


@pytest.fixture
def scan_impl(request):
    """Run a test under one named implementation of the fused scan.

    The three R10 gates opt in with `pytestmark = pytest.mark.usefixtures(
    "scan_impl")`, and `pytest_generate_tests` below then runs each of their
    CUDA tests once per implementation. That is not thoroughness for its own
    sake. The dispatchers default to the whole-scan kernels, so a gate that
    called `lif_scan(..., fused=True)` once would exercise ONLY that path, and
    the per-timestep Functions -- which stay in the tree as the bitwise oracle and
    the fallback, and whose Python `scripts/audit/08_mutation_campaign.py` mutates
    (T19, T20, D11, D12) -- would be left guarded by nothing. Run under both, every
    leg of every gate holds each path to the eager reference directly.

    Unparametrised (a CPU test, or any test outside those files) it selects
    nothing and yields None.
    """
    impl = getattr(request, "param", None)
    if impl is None:
        yield None
        return
    from snn.wholescan import use_scan_impl, wholescan_available

    if impl == "wholescan" and not wholescan_available():
        # Without it the dispatcher falls back, and the test would pass by
        # checking the per-timestep path a second time under the wrong name.
        pytest.skip("the whole-scan NVRTC kernels are unavailable on this stack")
    with use_scan_impl(impl):
        yield impl


def pytest_generate_tests(metafunc) -> None:
    # CUDA tests only: on CPU `fused=True` is the eager path whichever
    # implementation is named, and running it twice would prove nothing twice.
    if "scan_impl" in metafunc.fixturenames and metafunc.definition.get_closest_marker("cuda"):
        metafunc.parametrize("scan_impl", SCAN_IMPLS, indirect=True)
