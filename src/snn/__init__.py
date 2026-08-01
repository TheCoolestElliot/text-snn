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

__version__ = "2.0.0-phase2"

__all__ = ["__version__"]
