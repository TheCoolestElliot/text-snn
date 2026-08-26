"""The chat model: the committed two-compartment spiking arm, scaled up.

WHAT IS AND IS NOT NEW HERE
---------------------------
The network is `snn.model.TwoCompThresholdCharLM`, unmodified -- the arm EXP_011
measured, with the two-compartment neuron EXP_004 adopted and EXP_008's learned
per-channel threshold composed onto it. No new layer type, no new kernel, no new
backward, and therefore nothing here that needs an R10 gate. A checkpoint this
module writes loads into `snn.model` and evaluates under `snn.evaluate` without
a translation step.

Exactly two things differ from a research run, and both are ordinary
hyperparameters rather than architecture:

1.  **Size.** Wider and deeper than the 735 K-parameter Phase-2 baseline, because
    nothing about conversation is measured against that budget. The baseline's
    width is fixed at d=512 so that arms are comparable to each other; this model
    is not being compared to anything.

2.  **The initialisation of `beta_s_raw`** -- see `spread_slow_poles` below.
    This changes no function class, adds no parameter, and touches no forward
    pass: it is the value an existing parameter starts at.

A third became available on 2026-08-22 and is **opt-in and off by default**:
`arch="twocomp_threshold_detach"`, which trains the same network through
`EXP_017`'s bounded gradient estimator. It is not a hyperparameter and it is not
a new network either; see `TwoCompThresholdDetachCharLM` below, which states
exactly what it changes and what it provably does not.

WHY (2) IS THE INTERESTING ONE
------------------------------
Phase 4 measured the model's memory horizon -- how far back context still helps
-- at 7 characters for the Phase-2 LIF and 47-48 for the two-compartment neuron.
Forty-eight characters is about eight words. A conversational model with an
eight-word memory has, by the time it is half way through a reply, already lost
the question.

The horizon is set by the slow pole: a channel with decay `beta_s` integrates
over roughly `tau = 1/(1 - beta_s)` characters, and the committed initialisation
gives **every channel the same** `beta_s = 0.95`, i.e. `tau = 20`. So the model
starts with every neuron in the network sharing one timescale, and has to
discover a spread by gradient descent from a point where none exists.

EXP_006 found that ~3 % of channels carry the horizon and 94.9 % of the arm's
gain, and that clamping every channel to the median `beta_s` is *worse than the
Phase-2 baseline* -- the tail is the mechanism. `spread_slow_poles` gives the
model that tail at initialisation instead of asking it to grow one: `tau` is
laid out log-uniformly from a few characters to several hundred, so some
channels are integrating over a word and others over a whole turn from step zero.

**This is a bet, not a result.** It is the kind of claim the research protocol
would require pre-registering, running against the committed initialisation at
matched seeds, and reporting either way. That has not been done, and nothing in
this package should be read as evidence that it works. It is here because a
conversational model needs a longer horizon than 48 characters and this is the
cheapest thing that might supply one. `scripts/chat/horizon.py` measures what the
trained model actually got.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field

import torch
import torch.nn as nn

from snn.model import TwoCompartmentCharLM, TwoCompThresholdCharLM, count_params
from snn.prescan import threshold_gain
from snn.twocomp_detach import twocomp_detach_scan
from snnchat.tokenizer import VOCAB_VERSION, ChatTokenizer

__all__ = ["ChatConfig", "build_chat_model", "spread_slow_poles", "describe_model",
           "TwoCompThresholdDetachCharLM", "ARCHS"]

#: Every `arch` string `build_chat_model` accepts. The single-pole arms are
#: absent on purpose -- `EXP_014`/`EXP_015` measure their memory horizon at 7-8
#: characters, which cannot hold a turn.
ARCHS = ("twocomp_threshold", "twocomp", "twocomp_threshold_detach")


@dataclass
class ChatConfig:
    """Every knob of a chat training run.

    Deliberately NOT `snn.config.Config`. That dataclass is the normative record
    of a research run -- its field order is specified, its hash identifies an
    experiment, and its `corpus` field admits only `enwik8` and `text8`. Writing
    a `config.json` that claims `corpus="enwik8"` for a model trained on SODA
    would put a false statement into a file whose whole purpose is to be
    believed. This is a separate type for a separate thing.
    """

    # --- data ---------------------------------------------------------
    data_dir: str = "data/chat"
    seq_len: int = 384
    batch_size: int = 64
    mix: dict[str, float] = field(default_factory=dict)
    #: Loss weight on characters inside a `<|bot|>` turn; 1.0 = every character
    #: weighted equally, which is what every checkpoint before this field
    #: existed was trained with, and which the trainer keeps a separate code
    #: path for so that those runs still reproduce exactly.
    #:
    #: Above 1.0 the model spends proportionally more of its gradient on
    #: producing the reply and less on predicting the prompt. This is NOT
    #: masking -- see `snnchat.data.bot_turn_weights` for why the distinction
    #: matters at this scale.
    bot_loss_weight: float = 1.0
    #: Fraction of training windows nudged forward to start at a conversation
    #: boundary. 0.0 = uniformly random offsets, as before.
    #:
    #: The point is that a reply can only learn to depend on a request that is
    #: inside its own window: at `seq_len = 256` with turns of up to 400
    #: characters, a randomly-placed window often holds a reply whose request it
    #: does not contain. Aligned windows are also the state a real conversation
    #: begins in, so training and `ChatSession` see the same thing.
    #:
    #: THIS IS A REQUEST, NOT THE REALISED RATE. An offset is only moved to a
    #: conversation start it can reach within `MixtureSampler.align_lookahead`,
    #: which defaults to `seq_len`; a conversation longer than that is left where
    #: it fell. Measured on the 2026-08-06 mixture at `seq_len = 256`,
    #: `align_frac = 0.75` realises **38.3 %** aligned windows against a baseline
    #: of 0.2 % (`experiments/chat/_quality/alignment_realised.json`). Read it the
    #: way `MixtureSampler` asks its own weights to be read: the rate you asked
    #: for and the rate you got are different numbers.
    align_frac: float = 0.0
    #: How far forward to look for a conversation start. 0 = `seq_len`.
    #:
    #: Raising it is how the realised rate is made to approach the requested one.
    #: It should NOT simply be set as high as possible: at 100 % alignment the
    #: model only ever sees the first `seq_len` characters of a conversation, and
    #: on a source whose conversations average 757 characters that is two thirds
    #: of every story it would never train on. The windows left unaligned are
    #: what covers the rest.
    align_lookahead: int = 0

    # --- model --------------------------------------------------------
    d_model: int = 1024
    n_layers: int = 4
    vocab_size: int = 0            # 0 = take it from the tokenizer
    arch: str = "twocomp_threshold"

    # --- neuron -------------------------------------------------------
    beta: float = 0.5              # fast pole, as committed
    threshold: float = 1.0
    surrogate_alpha: float = 2.0
    w_init: float = 0.1
    thr_log_init: float = 0.0
    #: `beta_slow` is ignored when `spread_tau` is on; kept so a run that turns
    #: the spread off is still fully specified by this record.
    beta_slow: float = 0.95
    spread_tau: bool = True
    tau_min: float = 3.0
    tau_max: float = 600.0

    # --- optimisation -------------------------------------------------
    lr: float = 2.0e-3
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    warmup_steps: int = 500
    max_steps: int = 60_000
    lr_schedule: str = "cosine"
    lr_final_frac: float = 0.05

    # --- systems ------------------------------------------------------
    dtype: str = "fp32"
    fused: bool = True
    cuda_graph: bool = True
    device: str = "cuda"
    seed: int = 0
    deterministic: bool = False

    # --- bookkeeping --------------------------------------------------
    run_name: str = "chat"
    out_dir: str = "experiments/chat"
    eval_every: int = 2000
    eval_batches: int = 40
    #: Evaluation batch size, separate from the training one and smaller by
    #: default. This is not a preference, it is a fix: an evaluation at the
    #: TRAINING batch size runs alongside a still-resident captured graph and its
    #: static buffers, and on an 8 GiB card that combination spills into host
    #: memory. Under Windows' WDDM the spill does not fail, it just gets ~50x
    #: slower -- one probe arm sat in a 30-second evaluation for eighteen
    #: minutes before it was killed. 0 means "use batch_size".
    eval_batch_size: int = 48
    ckpt_every: int = 1000
    log_every: int = 100
    sample_every: int = 2000
    vocab_version: int = VOCAB_VERSION

    def to_dict(self) -> dict:
        return asdict(self)


def spread_slow_poles(
    model: TwoCompartmentCharLM, tau_min: float, tau_max: float
) -> list[float]:
    """Lay each layer's `beta_s` out log-uniformly in its time constant.

    Channel `c` of `d` gets `tau = tau_min * (tau_max/tau_min)**(c/(d-1))` and
    `beta_s = 1 - 1/tau`, written into `beta_s_raw` as its logit so the realised
    decay is that value to fp32 rather than approximately it -- the same care
    `TwoCompartmentCharLM.__init__` takes with the scalar case.

    Log-uniform rather than linear because what matters is the *ratio* between
    adjacent timescales: linear spacing from 3 to 600 would put 95 % of the
    channels above tau=30 and leave the short end, where most of the useful
    signal is, with almost no coverage.

    Every layer gets the same layout. Giving deeper layers slower poles is the
    obvious alternative and is not done here, because it is a second untested
    choice stacked on an untested one, and there would be no way to tell which
    of the two did anything.

    Returns the realised tau of the first and last channel, for the log.
    """
    if not (0.0 < tau_min < tau_max):
        raise ValueError(f"need 0 < tau_min < tau_max, got {tau_min}, {tau_max}")
    if tau_min <= 1.0:
        # beta_s = 1 - 1/tau is <= 0 at tau <= 1, and the sigmoid's logit is
        # undefined there. Caught here rather than as a NaN in the first forward.
        raise ValueError(f"tau_min must exceed 1 character, got {tau_min}")

    with torch.no_grad():
        for layer in range(model.n_layers):
            raw = model.beta_s_raw[layer]
            d = raw.shape[-1]
            frac = torch.linspace(0.0, 1.0, d, dtype=torch.float64)
            tau = tau_min * (tau_max / tau_min) ** frac
            beta_s = 1.0 - 1.0 / tau
            raw.copy_(torch.log(beta_s / (1.0 - beta_s)).to(raw.dtype).view_as(raw))
    return [float(tau_min), float(tau_max)]


class TwoCompThresholdDetachCharLM(TwoCompThresholdCharLM):
    """The chat arm, trained through `EXP_017`'s **bounded** gradient estimator.

    NOT A RESEARCH ARM AND NOT PRE-REGISTERED. It lives here rather than in
    `snn.model` because `snnchat` is outside the Phase-1..5 protocol
    (`snnchat/__init__.py`) and because putting it in the research package would
    add an `arch` value that no pre-registration has asked for. Nothing under
    `snn/` is modified by this class: it imports `snn.twocomp_detach` and
    `snn.prescan` and composes them, which is the one-way dependency this
    package is allowed.

    WHAT IT CHANGES, EXACTLY
    ------------------------
    One thing: the **backward**. `TwoCompThresholdCharLM._scan` calls
    `snn.twocomp.twocomp_scan`; this calls `snn.twocomp_detach.twocomp_detach_scan`
    on the identical arguments. The two share the same compiled *forward* kernel
    -- `twocomp_forward_kernel` is imported by `snn.twocomp_detach`, not
    re-declared -- so under `model.eval()` the two classes are the same network,
    and `test_chat_detach_forward_is_bitwise_the_shipped_arm` asserts that at
    `== 0.0` rather than at a tolerance.

    It adds **no parameter**, no state variable, no function, no kernel, no
    hand-written backward of its own, and no inference cost. A checkpoint trained
    here has a `twocomp_threshold` state dict verbatim, so it loads into
    `snn.model.build_model(arch="twocomp_threshold")` and evaluates bit-identically
    -- the property `tests/test_snnchat.py::test_chat_model_is_a_research_arm_unchanged`
    exists to protect, extended to this arch by
    `test_a_detached_chat_checkpoint_still_loads_as_the_research_arm`.

    WHY IT IS HERE
    --------------
    `EXP_016` proved the two-compartment backward-through-time has a per-step
    multiplier `beta_f*dv` bounded by **nothing** once `|w*vs| > 1`, because the
    spike test is on the mixed membrane `vf + w*vs` while the reset lands on `vf`
    alone and `vs` is never reset. The plain LIF's same multiplier is provably
    bounded by 0.5123596 at any width. `EXP_015` is the run that died of the
    difference; `EXP_017` is the fix, and `EXP_025`'s H3 held at 6 of 6 runs
    completing 20,000 steps with zero non-finite losses -- **3/3 at each of two
    rungs**, 735 K and 5.0 M, under the unchanged frozen recipe, where the
    undetached arm at 5.0 M diverges deterministically. Quoted that way rather
    than as "6/6 at 5.0 M", which `04_phase4_interim.md` §23.4 compresses it to
    and which reads as six runs at one width.

    **The chat model is on the wrong side of that line, measured rather than
    assumed.** `scripts/chat/reset_jacobian_probe.py` reads the multiplier off
    the checkpoints already on disk, gated on the local scan reproducing the
    committed one bitwise. On `chat-v3d-aligned/ckpt_best.pt` -- the checkpoint
    `experiments/chat/SHIPPED` names -- every one of the four layers exceeds 1.0,
    layer 0 reaches **9.776, i.e. 19.1x the plain LIF's hard ceiling**, and
    `max|w*vs|` reaches **14,454** against a crossover at ~1. Five checkpoints
    were probed (`chat-v2`, `chat-v3d-aligned`, `chat-v6-scratch`,
    `chat-v6-inst-a-s1`, `chat-v12-rare`) and **all five are inside the region**,
    at every layer.

    Detaching the reset sends `d vf_out / d v` to zero, which removes `vf` from
    both entries of the 2x2 per-step Jacobian, leaving `diag(beta_f*(1 - s),
    beta_s)` -- diagonal, so no non-normal transient, and bounded by
    `beta_f = 0.5`. That is stricter than the plain LIF's own ceiling and it does
    not depend on `w`, on `vs`, or on width.

    THE COST, AND IT IS NOT HIDDEN
    -------------------------------
    The gradient is **biased**: the pathway "firing now lowers my own future
    membrane" is dropped from the backward. `EXP_017` H3 measured what that costs
    in bpc at 735 K on the research corpus and `EXP_025` measured it at 5.0 M;
    **neither figure transfers to this corpus at this size, and none is quoted
    here.** Whether it costs anything on the chat model is unmeasured, which is
    why this arch is opt-in and why `twocomp_threshold` remains the default.
    """

    def _scan(self, cur, v0, layer):
        return twocomp_detach_scan(
            threshold_gain(cur, self.thr_log[layer]),
            v0,
            self.w[layer],
            self.slow_decay(layer),   # one kernel per layer, NOT per timestep
            self.beta,                # beta_f: the baseline's fixed fast pole
            self.threshold,
            self.surrogate_alpha,
            self.fused,
        )


def build_chat_model(cfg: ChatConfig) -> nn.Module:
    """Construct the arm named by `cfg.arch`, on `cfg.device`.

    Parameters are initialised on CPU and then moved, so the initial weights
    depend only on the seed and not on which device the run lands on -- the
    property `snn.model.build_model` documents and this mirrors.
    """
    if cfg.vocab_size <= 0:
        cfg.vocab_size = ChatTokenizer().vocab_size
    common = dict(
        vocab_size=cfg.vocab_size,
        d_model=cfg.d_model,
        n_layers=cfg.n_layers,
        beta=cfg.beta,
        threshold=cfg.threshold,
        reset="hard",           # EXP_004 §2 fixes this for the two-compartment neuron
        surrogate_alpha=cfg.surrogate_alpha,
        dtype=cfg.dtype,
        fused=cfg.fused,
        t_steps=1,
    )
    if cfg.arch == "twocomp_threshold":
        model: nn.Module = TwoCompThresholdCharLM(
            beta_slow=cfg.beta_slow, w_init=cfg.w_init,
            thr_log_init=cfg.thr_log_init, **common
        )
    elif cfg.arch == "twocomp_threshold_detach":
        # Same network, same forward kernel, same state dict; the single
        # difference is the gradient estimator. See the class docstring.
        model = TwoCompThresholdDetachCharLM(
            beta_slow=cfg.beta_slow, w_init=cfg.w_init,
            thr_log_init=cfg.thr_log_init, **common
        )
    elif cfg.arch == "twocomp":
        model = TwoCompartmentCharLM(
            beta_slow=cfg.beta_slow, w_init=cfg.w_init, **common
        )
    else:
        raise ValueError(
            f"snnchat supports {ARCHS}; the single-pole arms have a 7-character "
            f"memory horizon and cannot hold a turn. Got {cfg.arch!r}"
        )

    if cfg.spread_tau:
        spread_slow_poles(model, cfg.tau_min, cfg.tau_max)

    return model.to(cfg.device)


def describe_model(model: nn.Module, cfg: ChatConfig) -> dict:
    """A dict for the run header: size, and the realised timescale spread.

    The timescales are *read back off the parameter* rather than recomputed from
    `cfg.tau_min`/`tau_max`, so a run whose spread silently failed to apply
    reports the uniform 0.95 it actually has instead of the spread it asked for.
    """
    out: dict = {
        "params": count_params(model),
        "d_model": getattr(model, "d_model", None),
        "n_layers": getattr(model, "n_layers", None),
        "vocab_size": getattr(model, "vocab_size", None),
        "arch": cfg.arch,
    }
    if hasattr(model, "slow_decay"):
        taus = []
        with torch.no_grad():
            for k in range(model.n_layers):
                beta_s = model.slow_decay(k).flatten().double()
                tau = 1.0 / (1.0 - beta_s).clamp_min(1e-12)
                taus.append(
                    {
                        "layer": k,
                        "tau_min": round(float(tau.min()), 2),
                        "tau_median": round(float(tau.median()), 2),
                        "tau_max": round(float(tau.max()), 2),
                    }
                )
        out["slow_pole_tau"] = taus
    return out


def param_count(vocab_size: int, d_model: int, n_layers: int, arch: str) -> int:
    """Closed form, for sizing a run before building it."""
    base = (
        vocab_size * d_model
        + n_layers * (d_model * d_model + d_model)
        + d_model * vocab_size
        + vocab_size
    )
    # `twocomp_threshold_detach` shares its row with `twocomp_threshold` because
    # it IS that parameterisation -- the detached estimator adds no parameter, so
    # a closed form that gave it its own number would be describing a network
    # that does not exist.
    per_channel = {"twocomp": 2, "twocomp_threshold": 3,
                   "twocomp_threshold_detach": 3}[arch]
    return base + per_channel * n_layers * d_model


__all__.append("param_count")
_ = math
