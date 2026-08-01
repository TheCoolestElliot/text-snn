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

SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)


def pytest_configure(config) -> None:
    config.addinivalue_line(
        "markers", "cuda: test requires a working CUDA device; skipped on CPU-only machines"
    )
    config.addinivalue_line(
        "markers", "data: test requires the prepared corpus under data/"
    )
