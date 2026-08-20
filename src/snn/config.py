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
ARCH_CHOICES = ("snn", "analogue", "twocomp", "twocomp_threshold", "twocomp_detach",
                "tokenshift", "threshold", "noise", "dopamine", "interface",
                "localdopamine", "gru")
IFACE_READ_CHOICES = ("last", "all")
#: Kept in step with `snn.neuromod.NM_SOURCES` by
#: tests/test_neuromod.py::test_config_choices_match_the_module -- two tuples
#: naming the same thing is how they stop agreeing.
NM_SOURCE_CHOICES = ("off", "const", "pos", "rolled", "local")
DA_MODE_CHOICES = ("mult", "add")
DA_SOURCE_CHOICES = ("off", "rpe", "rolled")
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

    # --- two-compartment neuron (arch="twocomp"/"twocomp_threshold"/
    #     "twocomp_detach"; EXP_004 §2) --------------------------------------
    # Ignored by every other arm. They are Config fields rather than constants in
    # model.py so that a run's config.json records the neuron it actually trained
    # -- the initialisation is the thing EXP_004's §7.2 screen selects, and an
    # unlogged choice there would be an unfalsifiable one.
    #
    # NOTE ON `reset` AND `twocomp_detach` (EXP_017): all three two-compartment
    # arms require `reset="hard"` and their forward passes are identical. The
    # `_detach` arm's difference is in the BACKWARD -- the reset factor is treated
    # as a constant -- which is why it is named by `arch` and not by `reset`.
    # `reset` continues to describe the forward rule, which is what it has always
    # meant, and `config.json` therefore still records the neuron that ran.
    beta_slow: float = 0.95         # slow-pole decay at init; sigmoid-parameterised
    w_init: float = 0.1             # initial fast/slow mix; 0.0 nests the baseline

    # --- pre-scan arms (EXP_007, EXP_008) --------------------------------
    # Ignored by every other arm, and Config fields rather than constants for the
    # same reason `beta_slow` and `w_init` are: the initialisation is what the
    # §7.2 screen selects, and an unlogged choice there is an unfalsifiable one.
    # Both defaults are the value at which the arm IS the Phase-2 baseline, so a
    # run that forgets to set them trains the baseline rather than something
    # undocumented.
    mu_init: float = 1.0            # tokenshift only: 2-tap mix; 1.0 nests the baseline
    # EXP_011 composes the threshold arm onto the two-compartment neuron, so
    # thr_log_init is read by `twocomp_threshold` too. At 0.0 that arm nests the
    # ADOPTED arm bitwise, not the Phase-2 baseline -- a different nesting from the
    # one the comment above describes, and the reason this line is not "threshold
    # only" any more.
    thr_log_init: float = 0.0       # threshold/twocomp_threshold: 0.0 nests the parent

    # --- injected background spike noise (arch="noise"; EXP_013) ----------
    # Ignored by every other arm. `noise_amp` is the standard deviation of the
    # injected current in threshold units -- the Bernoulli process is centred and
    # normalised so that it is, which is EXP_013 §1.2's whole design. 0.0 is the
    # value at which the arm IS the Phase-2 baseline, bitwise and by code path,
    # for the same reason `mu_init` and `thr_log_init` default to their nesting
    # values: a run that forgets to set it trains the baseline rather than
    # something undocumented.
    noise_amp: float = 0.0          # noise only: injected current sd; 0.0 nests Phase 2
    noise_p: float = 0.1            # noise only: Bernoulli event rate; held fixed by EXP_013

    # --- dopamine: a broadcast reward prediction error (arch="dopamine"; EXP_018)
    # Ignored by every other arm, and Config fields rather than constants for the
    # reason `beta_slow`, `mu_init` and `noise_amp` are: the initialisation and
    # the squash scale are what the §7.2 screen and the calibration select, and an
    # unlogged choice there is an unfalsifiable one.
    #
    # `da_source="off"` is the value at which the arm IS the Phase-2 baseline --
    # by code path, not by float identity, so the nesting holds for the additive
    # mode too (`x + 0.0` is not `x` for `x = -0.0`). It is the default for the
    # same reason `noise_amp` defaults to 0.0: a run that forgets to set it trains
    # the baseline rather than something undocumented. The driver sets it
    # explicitly and G5 in EXP_018 asserts the run's config.json says so.
    da_mode: str = "mult"           # "mult" (gain) | "add" (excitability)
    da_source: str = "off"          # "off" nests Phase 2 | "rpe" | "rolled" (the control)
    # tau, the squash scale. MEASURED, not chosen: the median sd of phi across
    # four committed checkpoints spanning three architectures, which agree to
    # 3.1%. Provenance: docs/reports/data/exp_018_da_calibration.json.
    da_scale: float = 1.1291
    da_gain_init: float = 0.0       # dopamine only: per-channel sensitivity; 0.0 nests the baseline

    # --- EXP_020: the interface (arch="interface") ----------------------
    # Every default here nests the Phase-2 baseline BITWISE, forward and
    # backward -- not "to a tolerance". That is what makes `arch="interface"`
    # with no other flag a legitimate anchor rather than a fourth thing to
    # control for, and tests/test_interface.py::test_nests_spiking asserts it.
    # The same reason `noise_amp` and `da_gain_init` default the way they do: a
    # run that forgets a flag trains the baseline, not something undocumented.
    iface_fold: bool = False        # remove layers.0's GEMM; the table IS the current
    iface_binary_input: bool = False  # layer 0 receives {0,1}, completing I1
    iface_read_layers: str = "last"   # "last" | "all": which layers the head reads
    iface_read_lags: int = 1          # how many taps t, t-1, ...; 1 nests the baseline
    # The input code's firing threshold. 0.0 against a zero-mean shadow is
    # density 0.5, which is the value `snn.interface`'s gain derivation assumes;
    # it is NOT swept, and moving it moves the derived gain with it.
    iface_thr_in: float = 0.0
    # EXP_022: the input code's PLASTICITY axis, orthogonal to its density.
    # The realised density is a function of `iface_thr_in / iface_code_sd`, so
    # at the default threshold this moves no density, no gain and no centre --
    # only how far each bit's shadow sits from its own threshold at step 0, and
    # therefore how much of `atan_spike`'s surrogate gradient it receives.
    # 1.0 is the value EXP_020's `binin` ran at.
    iface_code_sd: float = 1.0

    # --- EXP_021: neuromodulation --------------------------------------
    # Two independent forms, deliberately separable: `nm_*` moves the RPE from
    # the head to the previous layer (arch="localdopamine", a forward-pass
    # modulator that decision #3 ADMITS); `tf_kappa` moves it off the forward
    # pass entirely and onto the learning rule, and applies to EVERY arch
    # because it changes no function class. Running both at once would compose
    # two arms, which no pre-registration here does.
    # EXP_023 turns this into a LADDER of decreasing signal content in one
    # fixed algebraic form: off < const < pos < rolled < local. See
    # snn.neuromod.NM_SOURCES -- `const` is a learned per-channel gain (which
    # EXP_008 Identity 1 makes a learned threshold), `pos` is a learned
    # function of window position and nothing else, and both exist because
    # EXP_021 could not say whether its winner won by being time-varying or
    # merely by being a gain.
    nm_source: str = "off"          # "off" | "const" | "pos" | "rolled" | "local"
    # How many window positions `nm_source='pos'` carries. Must cover seq_len:
    # a positional gain that wrapped would be a different arm that still trains.
    nm_pos_len: int = 256
    # tau for the LOCAL signal. Different quantity and different units from
    # da_scale (nats over d channels, not over V symbols), so it is calibrated
    # separately and never inherited. Provenance:
    # docs/reports/data/exp_021_nm_calibration.json.
    nm_tau: float = 1.0
    nm_gain_init: float = 0.0       # per-channel sensitivity; 0.0 nests the baseline
    nm_a_init: float = 0.0          # diagonal predictor slope
    # Predictor prior. sigmoid(-0.7) = 0.332, which is the CONVERGED population
    # rate (EXP_000 F3: 0.345-0.392), read off layer 0 -- the predictor consumes
    # layer k-1's spikes, so at K = 2 the only modulated layer's predictor reads
    # LAYER 0, not layer 1 as this comment used to say.
    #
    # THIS IS NOT THE VALUE EXP_021 SS2.6 CERTIFIES THE ARM ON, and the
    # difference is not cosmetic. That section certifies the squash unsaturated
    # at `b = -2.97 = logit(0.0488)`, layer 0's rate AT INITIALISATION, measuring
    # `mean sech^2(phi/tau) = 0.777`. `LocalDopamineCharLM.__init__` still
    # carries -2.97 as its own default, but `build_model` forwards
    # `cfg.nm_b_init`, so THIS value is what every committed nm_* run trained
    # with -- verified in each run's own config.json.
    #
    # With `nm_a_init = 0.0` the init logit is `b` exactly, so
    # `phi = (rbar - sigmoid(b)) * b`. At layer 0's init rate 0.0488:
    #     b = -2.97 -> phi = -0.0000, phi/tau = -0.00, sech^2 = 1.000
    #     b = -0.70 -> phi = +0.1981, phi/tau = +3.46, sech^2 = 0.004
    # so the shipped value starts SATURATED and desaturates as the rate climbs,
    # while the certified value starts unsaturated and saturates. Each is right
    # at one end of training. `nm_tau` is learned (SS2.6), which is the design's
    # answer to the 7x rate rise, so the saturation need not be permanent -- but
    # the gradient into (a, b) at step 0 is attenuated ~190x against the point
    # SS2.6 certifies. Recorded, referred, and NOT changed: -0.7 is what every
    # committed number used, and CONTRIBUTING.md SS4 forbids moving a baseline to
    # make a diagnostic look better. Pinned by
    # tests/test_neuromod.py::test_nm_b_init_is_the_value_the_runs_actually_used.
    nm_b_init: float = -0.7
    # The three-factor loss weight. 0.0 nests the unweighted objective BY CODE
    # PATH. |kappa| < 1 is enforced so a weight can never reach zero and delete
    # a position from the gradient. Not swept: 0.5 is the value at which the
    # extreme weight ratio is exactly 3.0.
    tf_kappa: float = 0.0
    tf_rolled: bool = False         # batch-roll the weights: THE CONTROL
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
        # Caught here rather than inside the training step: `noise_scale` divides
        # by sqrt(p*(1-p)), and p at 0 or 1 would produce an inf amplitude three
        # hours into a run instead of at construction.
        if not 0.0 < self.noise_p < 1.0:
            raise ValueError(f"Config.noise_p must be in (0, 1), got {self.noise_p!r}")
        if self.noise_amp < 0.0:
            raise ValueError(f"Config.noise_amp must be >= 0, got {self.noise_amp!r}")
        self._check_choice("iface_read_layers", IFACE_READ_CHOICES)
        if self.iface_read_lags < 1:
            raise ValueError(
                f"Config.iface_read_lags must be >= 1, got {self.iface_read_lags!r}"
            )
        if self.iface_read_lags > self.seq_len:
            # A readout tap longer than the window reads nothing but the zero
            # padding for every position, which trains and looks plausible.
            raise ValueError(
                f"Config.iface_read_lags ({self.iface_read_lags}) exceeds "
                f"seq_len ({self.seq_len}); every position would read only pad"
            )
        if self.iface_code_sd <= 0.0:
            raise ValueError(
                f"Config.iface_code_sd must be > 0, got {self.iface_code_sd!r}"
            )
        self._check_choice("nm_source", NM_SOURCE_CHOICES)
        if self.nm_tau <= 0.0:
            raise ValueError(f"Config.nm_tau must be > 0, got {self.nm_tau!r}")
        if self.nm_pos_len < 1:
            raise ValueError(
                f"Config.nm_pos_len must be >= 1, got {self.nm_pos_len!r}"
            )
        if (self.arch == "localdopamine" and self.nm_source == "pos"
                and self.nm_pos_len < self.seq_len):
            # The forward raises too, but three hours later. A positional gain
            # that does not cover the window is a typo every time.
            raise ValueError(
                f"Config.nm_pos_len ({self.nm_pos_len}) is shorter than "
                f"seq_len ({self.seq_len}); nm_source='pos' would raise mid-run"
            )
        if not -1.0 < self.tf_kappa < 1.0:
            # At |kappa| >= 1 a weight can reach 0 or go negative, deleting a
            # position from the gradient or rewarding the model for getting it
            # wrong. Caught at construction, not three hours into a run.
            raise ValueError(
                f"Config.tf_kappa must be in (-1, 1), got {self.tf_kappa!r}"
            )
        if self.tf_rolled and self.tf_kappa == 0.0:
            # The rolled control of an inactive weighting is the unweighted
            # objective, so the run would be an anchor wearing a control's name.
            raise ValueError(
                "tf_rolled=True with tf_kappa=0.0 is the unweighted objective; "
                "the control would silently BE the anchor"
            )
        if self.tf_rolled and self.batch_size < 2:
            raise ValueError(
                f"tf_rolled needs batch_size >= 2; got {self.batch_size}. At "
                "B = 1 the roll is the identity and the control would be the arm"
            )
        if self.arch == "localdopamine" and self.nm_source == "rolled"                 and self.batch_size < 2:
            raise ValueError(
                f"nm_source='rolled' needs batch_size >= 2; got {self.batch_size}"
            )
        self._check_choice("da_mode", DA_MODE_CHOICES)
        self._check_choice("da_source", DA_SOURCE_CHOICES)
        # Caught here rather than inside the forward pass: `dopamine()` divides by
        # tau, and a zero or negative tau would produce an inf or a sign-flipped
        # neuromodulator three hours into a run instead of at construction.
        if self.da_scale <= 0.0:
            raise ValueError(f"Config.da_scale must be > 0, got {self.da_scale!r}")
        # The rolled control is a roll along the batch axis, which is the identity
        # at B = 1 -- the control would silently BE the arm. Caught at
        # construction rather than at the first batch.
        if self.arch == "dopamine" and self.da_source == "rolled" and self.batch_size < 2:
            raise ValueError(
                "da_source='rolled' needs batch_size >= 2; at B = 1 the roll is "
                f"the identity and the control would be the arm. Got {self.batch_size}"
            )

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
    "beta_slow": "twocomp/twocomp_threshold: slow-pole decay at init, in (0, 1)",
    "w_init": "twocomp/twocomp_threshold: initial fast/slow mix; 0.0 nests Phase 2",
    "mu_init": "tokenshift only: 2-tap input mix; 1.0 nests the Phase-2 baseline",
    "thr_log_init": "threshold/twocomp_threshold: log per-channel threshold; "
                    "0.0 nests the arm's own parent",
    "noise_amp": "noise only: sd of the injected background spike current, in "
                 "threshold units; 0.0 nests the Phase-2 baseline",
    "noise_p": "noise only: Bernoulli event rate of the background spike train; "
               "sets sparsity, not variance",
    "da_mode": "dopamine only: how the broadcast RPE reaches the current -- "
               "'mult' (gain) or 'add' (excitability)",
    "da_source": "dopamine only: 'off' nests the Phase-2 baseline by code path; "
                 "'rpe' is the real signal; 'rolled' is the batch-roll control",
    "da_scale": "dopamine only: tau, the squash scale in nats; measured, see "
                "docs/reports/data/exp_018_da_calibration.json",
    "da_gain_init": "dopamine only: initial per-channel sensitivity k; 0.0 nests "
                    "the Phase-2 baseline",
    "iface_fold": "interface only: fold layers.0 into the embedding table "
                  "(an identity on the function class; frees d*d params)",
    "iface_binary_input": "interface only: binarise the input code through "
                          "atan_spike, so layer 0 receives {0,1} like every "
                          "other layer",
    "iface_read_layers": "interface only: 'last' or 'all' -- which layers' "
                         "spikes the head reads",
    "iface_read_lags": "interface only: number of readout taps (t, t-1, ...); "
                       "1 nests the baseline",
    "iface_thr_in": "interface only: input-code firing threshold; 0.0 is "
                    "density 0.5, which the gain derivation assumes",
    "iface_code_sd": "interface only: sd of the input code's shadow at init. "
                     "Density depends on iface_thr_in/iface_code_sd, so at the "
                     "default threshold this changes plasticity and not "
                     "density; 1.0 is EXP_020's value",
    "nm_pos_len": "localdopamine only: window positions carried by "
                  "nm_source='pos'; must be >= seq_len",
    "nm_source": "localdopamine only: 'off' nests Phase 2; 'local' is the"
                 "previous layer's Bernoulli RPE; 'rolled' is the control",
    "nm_tau": "localdopamine only: squash scale in nats over d channels; "
              "calibrated, see docs/reports/data/exp_021_nm_calibration.json",
    "nm_gain_init": "localdopamine only: initial per-channel sensitivity; "
                    "0.0 nests the Phase-2 baseline",
    "nm_a_init": "localdopamine only: diagonal predictor slope at init",
    "nm_b_init": "localdopamine only: predictor prior logit at init",
    "tf_kappa": "three-factor loss weight w = 1 + kappa*tanh(phi/tau); 0.0 "
                "nests the unweighted objective by code path",
    "tf_rolled": "three-factor: roll the weights along the batch axis (control)",
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
    "da_mode": DA_MODE_CHOICES,
    "da_source": DA_SOURCE_CHOICES,
    "iface_read_layers": IFACE_READ_CHOICES,
    "nm_source": NM_SOURCE_CHOICES,
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
