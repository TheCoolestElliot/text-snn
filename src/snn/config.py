"""Configuration, seeding and determinism for the Phase-2 baseline.

Spec: docs/reports/02a_phase2_spec.md §3.

WHY THE ENVIRONMENT VARIABLE IS SET AT IMPORT TIME
--------------------------------------------------
cuBLAS chooses its workspace allocation strategy once, when the first cuBLAS
handle is created for a device. If ``CUBLAS_WORKSPACE_CONFIG`` is not already in
the environment at that moment, cuBLAS may reuse workspace across streams and
some GEMMs become run-to-run non-deterministic; worse,
``torch.use_deterministic_algorithms(True)`` will then *raise* on the first
affected GEMM, and with ``warn_only=True`` (which the spec requires) it will
merely warn and silently give up determinism.

Setting the variable from inside ``seed_everything`` is therefore not sufficient
on its own: by the time a caller gets around to calling it, another module may
already have touched CUDA. The only reliable place is module import, before
anyone has a chance to create a context -- so it happens at the top of this file,
before ``import torch``. ``snn/__init__.py`` deliberately imports nothing, so
``import snn.config`` is the first CUDA-relevant event in any of our entry
points.

``:4096:8`` means "8 workspaces of 4096 KiB", the value PyTorch's own
determinism documentation prescribes for CUDA >= 10.2.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import random
import warnings
from dataclasses import dataclass, fields
from typing import Any

# --- must precede `import torch` and any CUDA context creation. See docstring.
CUBLAS_WORKSPACE_CONFIG = ":4096:8"
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", CUBLAS_WORKSPACE_CONFIG)

import numpy as np  # noqa: E402  (deliberately after the env var)
import torch  # noqa: E402


# --------------------------------------------------------------------------
# The configuration object
# --------------------------------------------------------------------------

CORPUS_CHOICES = ("enwik8", "text8")
ARCH_CHOICES = ("snn", "analogue", "twocomp", "gru")
RESET_CHOICES = ("hard", "soft", "detached", "none")
SURROGATE_CHOICES = ("atan",)
DTYPE_CHOICES = ("fp32", "bf16", "fp16")
SCHEDULE_CHOICES = ("cosine", "constant")

#: reset rule -> integer code passed to FusedLIFScan (spec §7). Kept here rather
#: than in neuron.py so that config validation and the kernel cache agree on one
#: table.
RESET_CODES = {"hard": 0, "soft": 1, "detached": 2, "none": 3}

#: cfg.dtype -> torch dtype for the GEMMs only. Membrane state is always fp32
#: (spec §2, reconnaissance §3.11 C2).
TORCH_DTYPES = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}


@dataclass(frozen=False)
class Config:
    """Every knob of a Phase-2 run. Field order and names are normative."""

    # --- data ------------------------------------------------------------
    corpus: str = "enwik8"          # "enwik8" | "text8"
    data_dir: str = "data"
    seq_len: int = 256              # L
    batch_size: int = 128           # B

    # --- model -----------------------------------------------------------
    d_model: int = 512
    n_layers: int = 2               # K
    vocab_size: int = 0             # filled from the corpus; 0 means "unset"
    arch: str = "snn"               # "snn" | "analogue" | "gru"

    # --- neuron (SpikeGPT reference values, 01_reconnaissance.md §4.1) ----
    beta: float = 0.5               # membrane decay
    threshold: float = 1.0
    reset: str = "hard"             # "hard" | "soft" | "detached" | "none"
    surrogate: str = "atan"
    surrogate_alpha: float = 2.0
    t_steps: int = 1                # T

    # --- two-compartment neuron (arch="twocomp" only; EXP_004 §2) ---------
    # Ignored by every other arm. They are Config fields rather than constants in
    # model.py so that a run's config.json records the neuron it actually trained
    # -- the initialisation is the thing EXP_004's §7.2 screen selects, and an
    # unlogged choice there would be an unfalsifiable one.
    beta_slow: float = 0.95         # slow-pole decay at init; sigmoid-parameterised
    w_init: float = 0.1             # initial fast/slow mix; 0.0 nests the baseline

    # --- optimisation ----------------------------------------------------
    lr: float = 3e-3
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    warmup_steps: int = 200
    max_steps: int = 20_000
    lr_schedule: str = "cosine"     # "cosine" | "constant"

    # --- systems ---------------------------------------------------------
    fused: bool = True              # jiterator LIF; False = eager reference
    cuda_graph: bool = True
    dtype: str = "fp32"             # GEMM dtype: "fp32" | "bf16" | "fp16"
    device: str = "cuda"
    seed: int = 0
    deterministic: bool = True

    # --- run bookkeeping -------------------------------------------------
    run_name: str = ""
    out_dir: str = "experiments/runs"
    eval_every: int = 500
    eval_max_windows: int = 0       # 0 = full split
    ckpt_every: int = 1000
    log_every: int = 50

    def __post_init__(self) -> None:
        # Catch typos at construction rather than three hours into a run.
        self._check_choice("corpus", CORPUS_CHOICES)
        self._check_choice("arch", ARCH_CHOICES)
        self._check_choice("reset", RESET_CHOICES)
        self._check_choice("surrogate", SURROGATE_CHOICES)
        self._check_choice("dtype", DTYPE_CHOICES)
        self._check_choice("lr_schedule", SCHEDULE_CHOICES)
        for name in ("seq_len", "batch_size", "d_model", "n_layers", "t_steps"):
            if getattr(self, name) < 1:
                raise ValueError(f"Config.{name} must be >= 1, got {getattr(self, name)!r}")
        if self.vocab_size < 0:
            raise ValueError("Config.vocab_size must be >= 0 (0 means 'unset')")

    def _check_choice(self, name: str, choices: tuple[str, ...]) -> None:
        value = getattr(self, name)
        if value not in choices:
            raise ValueError(f"Config.{name}={value!r} not in {choices}")

    # -- convenience -------------------------------------------------------

    @property
    def torch_dtype(self) -> torch.dtype:
        """GEMM dtype. Membrane state ignores this and stays fp32 (spec §2)."""
        return TORCH_DTYPES[self.dtype]

    @property
    def reset_code(self) -> int:
        """Integer reset code for FusedLIFScan (spec §7)."""
        return RESET_CODES[self.reset]

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    def to_json(self) -> str:
        """Canonical serialisation: sorted keys, no incidental whitespace."""
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Config":
        known = {f.name for f in fields(cls)}
        unknown = set(d) - known
        if unknown:
            raise ValueError(f"unknown Config fields: {sorted(unknown)}")
        return cls(**d)


# --------------------------------------------------------------------------
# Hashing
# --------------------------------------------------------------------------

def config_hash(cfg: Config) -> str:
    """First 12 hex chars of sha256 over the sorted-key JSON of the config.

    Caveat, recorded deliberately: the spec defines this over the *whole*
    dataclass, so bookkeeping fields (`run_name`, `out_dir`, `log_every`, ...)
    change the hash even though they change no science. It is a run fingerprint,
    not a scientific-identity hash. Do not use it to decide whether two runs are
    the same experiment.
    """
    return hashlib.sha256(cfg.to_json().encode("utf-8")).hexdigest()[:12]


# --------------------------------------------------------------------------
# Seeding / determinism
# --------------------------------------------------------------------------

def seed_everything(seed: int, deterministic: bool) -> None:
    """Seed every RNG this project can reach, and optionally pin determinism.

    `deterministic=True` uses `warn_only=True` as the spec requires: a handful of
    ops (notably some backward kernels) have no deterministic implementation, and
    we would rather have the run continue with a warning on the record than die
    at step 4000. Any warning it emits belongs in the report.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        want = CUBLAS_WORKSPACE_CONFIG
        have = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
        if have != want and torch.cuda.is_initialized():
            # Too late to matter: cuBLAS has already picked its strategy.
            warnings.warn(
                f"CUBLAS_WORKSPACE_CONFIG is {have!r}, not {want!r}, and a CUDA "
                "context already exists. Determinism of cuBLAS GEMMs is not "
                "guaranteed for this process. Import snn.config first.",
                RuntimeWarning,
                stacklevel=2,
            )
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = want
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # MEASURED, NOT ASSUMED -- see 02_baseline_report.md.
        #
        # use_deterministic_algorithms(True) also flips
        # torch.utils.deterministic.fill_uninitialized_memory on, which emits an
        # extra fill kernel for every uninitialised allocation. Our fused LIF is
        # a 3-output jiterator kernel called once per timestep, so that is THREE
        # extra kernel launches per timestep on the one code path the whole
        # architecture exists to keep at one launch per timestep.
        #
        # Measured on this box at the baseline config (B=128, L=256, d=512):
        #     kernels/timestep   1.03  ->  4.05   (3.9x)
        #     fused fwd+bwd     15.18  -> 29.99 ms (1.98x)
        #
        # Leaving it on would silently hand back half of the fusion win in every
        # deterministic run -- which is every run we report.
        #
        # Turning it off is SOUND here, not merely convenient: the fill exists to
        # make *reads of uninitialised memory* reproducible. Our kernels are
        # elementwise over the full output shape, so every element of every
        # output is written before it is read. Nothing in this project reads an
        # uninitialised tensor, and tests/test_determinism.py asserts bit-exact
        # reproducibility with the fill disabled, which is the property that
        # actually matters.
        torch.utils.deterministic.fill_uninitialized_memory = False
    else:
        # Symmetric with the branch above: a process that turns determinism off
        # must not keep cuDNN pinned to its deterministic algorithm set, or
        # `--no-deterministic` measures something that is neither fully
        # deterministic nor fully fast, and the report would describe a
        # configuration that never ran.
        torch.use_deterministic_algorithms(False)
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

_HELP: dict[str, str] = {
    "corpus": "corpus name",
    "data_dir": "directory holding data/<corpus>/{train,val,test}.bin",
    "seq_len": "sequence length L (characters per window)",
    "batch_size": "batch size B",
    "d_model": "model width d",
    "n_layers": "number of spiking layers K",
    "vocab_size": "vocabulary size V; 0 means 'take it from the corpus'",
    "arch": "architecture arm",
    "beta": "membrane decay",
    "threshold": "firing threshold (comparison is >=)",
    "reset": "membrane reset rule",
    "surrogate": "surrogate gradient family",
    "surrogate_alpha": "surrogate width alpha; derivative(0) = alpha/2",
    "t_steps": "micro-steps per character T",
    "beta_slow": "twocomp only: slow-pole decay at init, in (0, 1)",
    "w_init": "twocomp only: initial fast/slow mix; 0.0 nests the Phase-2 baseline",
    "lr": "peak learning rate",
    "weight_decay": "AdamW weight decay",
    "beta1": "AdamW beta1",
    "beta2": "AdamW beta2",
    "grad_clip": "global grad-norm clip",
    "warmup_steps": "linear warmup steps",
    "max_steps": "total optimisation steps",
    "lr_schedule": "post-warmup schedule",
    "fused": "use the jiterator-fused LIF scan (--no-fused = eager reference)",
    "cuda_graph": "capture the training step into a CUDA graph",
    "dtype": "GEMM dtype; membrane state is fp32 regardless",
    "device": "torch device",
    "seed": "master seed",
    "deterministic": "pin deterministic algorithms and cuBLAS workspace",
    "run_name": "run identifier; empty means auto from arch/corpus/hash",
    "out_dir": "root directory for run artifacts",
    "eval_every": "steps between evaluations",
    "eval_max_windows": "cap on eval windows; 0 = full split",
    "ckpt_every": "steps between checkpoints",
    "log_every": "steps between JSONL log records",
}

_CHOICES: dict[str, tuple[str, ...]] = {
    "corpus": CORPUS_CHOICES,
    "arch": ARCH_CHOICES,
    "reset": RESET_CHOICES,
    "surrogate": SURROGATE_CHOICES,
    "dtype": DTYPE_CHOICES,
    "lr_schedule": SCHEDULE_CHOICES,
}


def build_parser() -> argparse.ArgumentParser:
    """One flag per Config field, generated from the dataclass.

    Both spellings are accepted (`--seq-len` and `--seq_len`) so that neither
    convention silently fails in a shell script. Booleans use
    `argparse.BooleanOptionalAction`, which gives the `--no-fused` /
    `--no-cuda-graph` forms the entry checklist requires.
    """
    p = argparse.ArgumentParser(
        description="Phase-2 spiking char-LM configuration",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    for f in fields(Config):
        dashed = "--" + f.name.replace("_", "-")
        flags = [dashed]
        if "_" in f.name:
            flags.append("--" + f.name)
        help_text = _HELP.get(f.name, "")
        if f.type is bool or f.type == "bool":
            p.add_argument(
                *flags,
                dest=f.name,
                action=argparse.BooleanOptionalAction,
                default=f.default,
                help=help_text,
            )
        else:
            kind = {"int": int, "float": float, "str": str}[
                f.type if isinstance(f.type, str) else f.type.__name__
            ]
            p.add_argument(
                *flags,
                dest=f.name,
                type=kind,
                default=f.default,
                choices=_CHOICES.get(f.name),
                help=help_text,
            )
    return p


def config_from_args(argv: list[str] | None = None) -> Config:
    """Parse argv (default `sys.argv[1:]`) into a validated Config."""
    ns = build_parser().parse_args(argv)
    return Config(**vars(ns))


def default_run_name(cfg: Config) -> str:
    """`<arch>-<corpus>-d<d>-L<K>-s<seed>-<hash>`; used when run_name is empty.

    Lives here rather than in train.py so that launch.py and train.py cannot
    disagree about where a run's artifacts are.
    """
    if cfg.run_name:
        return cfg.run_name
    return (
        f"{cfg.arch}-{cfg.corpus}-d{cfg.d_model}-L{cfg.n_layers}"
        f"-s{cfg.seed}-{config_hash(cfg)}"
    )
