"""snn -- character-level spiking language model (Phase 2).

Deliberately empty of re-exports. Importing `snn` must not import torch, touch
CUDA, or compile anything: `scripts/launch.py` and the test collector both import
this package in processes that may never use a GPU, and creating a CUDA context
early would defeat the CUBLAS_WORKSPACE_CONFIG determinism setting that
`snn.config` installs at its own import time (see that module's docstring).

Import the submodules explicitly:

    from snn.config import Config, seed_everything
    from snn.data import Corpus, ensure_corpus
"""

# PEP 440. The previous spelling, "2.0.0-phase2", is NOT a valid version and
# setuptools refuses to build metadata from it, so the package could not be
# installed at all; pyproject.toml reads this attribute as the single source of
# truth. The local segment after "+" carries the phase, which is what the old
# suffix was for, and it is now current: the tree is on Phase 4.
#
# This attribute is read by pyproject.toml and by nothing else. It has never been
# stamped into a run manifest or a report artifact -- every version recorded in
# docs/reports/data/ is torch's, numpy's or snntorch's -- so moving it changes no
# committed number and invalidates no provenance.
__version__ = "2.0.0+phase4"

__all__ = ["__version__"]
