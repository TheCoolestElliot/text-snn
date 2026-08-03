"""The §7.2 gradient-reachability screen, as a committed tool.

`03_phase3_candidates.md` §7.2 specified this screen and did not build it:

    Build the eager reference at a candidate's proposed init, take one backward,
    report |dL/dp| for every new parameter against |dL/dW|, and flag anything
    exactly zero or ~100x below. Seconds per candidate; it is what identifies
    §6.4's saddle before 0.8 GPU-hours are spent certifying a null.

`scripts/exp/004_gradient_reachability.py` ran that rule for EXP_004's candidate
and only for it -- two hard-coded inits of one neuron that already had a kernel,
an arch and a `Config` field. This file is the screen the section actually
described: a candidate registry, an eager prototype per candidate, and one rule
applied to all of them, runnable **before** any kernel exists. That ordering is
the entire economic argument for the screen. A candidate that needs a jiterator
kernel, a hand-written backward and a mutation-tested R10 gate before it can be
screened has already spent the cost the screen exists to avoid.

WHAT IT SCREENS
---------------
Every §6.3 candidate that introduces a learnable parameter, at the
initialisation its own proposal names. Candidates carrying no new parameter
(#3 depth, #7 reset ablation, #8 initialisation, #9 rate set-point, #11-#14) are
outside the screen's scope and are listed as such rather than silently omitted.

Prototypes here are **screening prototypes, not neurons.** They live in this
script rather than in `src/snn/` on purpose:

  * a neuron in `src/snn/` is covered by the R10 gate and the mutation campaign,
    and these are not -- they are plain-autograd references whose only job is to
    have the right *backward graph shape*. Shipping them next to gated code would
    imply a guarantee they do not carry.
  * `scripts/audit/08_mutation_campaign.py` requires each mutation's anchor text
    to appear exactly once in its target file. More neurons in `src/snn/` means
    more duplicated anchor lines and a campaign that fails on its own guard --
    the reason `snn/twocomp.py` is a separate module in the first place.

DERIVE FIRST, THEN MEASURE
--------------------------
§7.2's own precedent (§6.4) is that a saddle is checkable from the backward
equations *before* it is measured. So every candidate below carries a
`derived_zero` field written from its own adjoint recursion, and the screen
reports whether the measurement AGREES with the derivation. A disagreement in
either direction is loud:

  * derived zero, measured nonzero -> the derivation is wrong, or the prototype
    does not implement the candidate that was derived.
  * derived nonzero, measured zero -> a saddle nobody predicted. That is the
    outcome the screen exists for.

Neither is a reason to edit the derivation afterwards. `derived_zero` is fixed in
source before the run, which is the same discipline the experiment logs use.

CAN THIS SCREEN FAIL?
---------------------
A screen nothing has ever tripped is indistinguishable from a screen that cannot
trip. Three synthetic controls run on every invocation and the run **aborts** if
any of them behaves wrongly:

  * `ctl_live`  -- a per-channel gain on the current. Must PASS.
  * `ctl_dead`  -- a parameter entering the scan multiplied by exactly 0.0. Must
                   be flagged, and specifically by the ZERO leg.
  * `ctl_faint` -- a parameter whose contribution is scaled by 1e-9. Must be
                   flagged, and specifically by the RATIO leg, with a gradient
                   that is small but *not* zero.

`ctl_faint` exists because the rule has two legs and a control that only trips
the zero leg would leave the ratio leg unverified -- the `EXP_004` §9.2 lesson
that a contract is only guarded where the quantity it constrains is observed.

REPRODUCING A NUMBER IT DID NOT PRODUCE
---------------------------------------
Candidate #1 is screened through the real `arch="twocomp"` model, so its two
inits must reproduce `docs/reports/data/exp_004_gradient_reachability.json` --
committed by a different script, before this one existed. The check is on by
default and the run aborts if it fails. It is the same discipline as `EXP_001`'s
F1 self-check, which is what caught a contaminated training run in Phase 3.

WHAT THE TWO LEGS MEAN, WHICH IS NOT THE SAME THING
---------------------------------------------------
Carried over from `scripts/exp/004_gradient_reachability.py`, because it governs
how the output should be read and is easy to lose.

An **exactly zero** gradient is fatal and unrecoverable: AdamW's update is
`m / (sqrt(v) + eps)`, which is exactly 0 when the gradient is identically 0 at
every step. The parameter is frozen at its initialisation for the whole run and
nothing in the logs says so.

A **small but nonzero** gradient is much weaker evidence, because Adam is
scale-free per parameter: it divides by the gradient's own second moment, so a
uniformly 100x smaller gradient still takes a full-size step. A low ratio flags a
parameter worth watching; on its own it is not a reason to reject an init. Both
numbers are reported separately, and `verdict` distinguishes `SADDLE` from
`FAINT` rather than collapsing them into one FAIL.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import types
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO / "src") not in sys.path:
    sys.path.insert(0, str(_REPO / "src"))

from snn.config import Config, seed_everything  # noqa: E402
from snn.data import Corpus, RandomWindowSampler  # noqa: E402
from snn.model import build_model  # noqa: E402
from snn.surrogate import atan_spike  # noqa: E402

# §7.2 as written: "flag anything exactly zero or ~100x below".
RATIO_FLOOR = 1.0 / 100.0

# The committed artifact candidate #1 must reproduce, and the tolerance it must
# do it to. Same seed, same batch, same eager code path, same device -- so the
# honest expectation is bit-identity, and the tolerance is set just off it so
# that a real drift is chased rather than absorbed.
EXP_004_JSON = "docs/reports/data/exp_004_gradient_reachability.json"
REPRO_RTOL = 1e-6

_LOG = math.log


def _logit(p: float) -> float:
    return _LOG(p / (1.0 - p))


# ==========================================================================
# Screening prototypes -- plain autograd, no kernels, no gate.
#
# Each takes (self, cur [B, L, d], v0 [B, state_mult*d], layer) and returns
# (spikes [B, L, d], v_final). `self` is the host SpikingCharLM, so `self.beta`,
# `self.threshold` and `self.surrogate_alpha` are the frozen Phase-2 values and
# no candidate can silently move one of them.
# ==========================================================================


def _lif_baseline(self, cur: Tensor, v0: Tensor, layer: int):
    """The Phase-2 neuron, re-stated here as the zero-new-parameter reference."""
    v = v0
    spikes = []
    for t in range(cur.shape[1]):
        v_pre = v * self.beta + cur[:, t]
        s = atan_spike(v_pre - self.threshold, self.surrogate_alpha)
        v = v_pre * (1.0 - s)
        spikes.append(s)
    return torch.stack(spikes, dim=1), v


def _scan_learned_decay(self, cur: Tensor, v0: Tensor, layer: int):
    """Candidate #5: one learned per-channel decay, sigmoid-parameterised.

    `beta_raw = logit(0.5) = 0` nests the Phase-2 baseline exactly, and unlike
    candidate #1's `w = 0` that nesting is NOT a saddle: the parameter multiplies
    `v_{t-1}`, which is a nonzero *state*, whereas #1's `beta_s` multiplies a
    state whose *adjoint* is what vanishes. The screen is here to confirm that
    distinction is real rather than plausible.
    """
    beta = torch.sigmoid(self.beta_raw[layer])
    v = v0
    spikes = []
    for t in range(cur.shape[1]):
        v_pre = v * beta + cur[:, t]
        s = atan_spike(v_pre - self.threshold, self.surrogate_alpha)
        v = v_pre * (1.0 - s)
        spikes.append(s)
    return torch.stack(spikes, dim=1), v


def _scan_rms_current(self, cur: Tensor, v0: Tensor, layer: int):
    """Candidate #2: RMSNorm on the input current, outside the time loop.

    The gain is applied once to the whole [B, L, d] current before the scan
    starts, which is what "outside the time loop" means and why this candidate
    costs O(1) kernels per layer rather than O(L).
    """
    rms = cur.pow(2).mean(dim=-1, keepdim=True).add(1e-5).sqrt()
    cur = (cur / rms) * self.rms_gain[layer]
    return _lif_baseline(self, cur, v0, layer)


def _scan_learned_threshold(self, cur: Tensor, v0: Tensor, layer: int):
    """`EXP_004` §10.6: a static learned per-channel threshold, one parameter.

    §10.6 showed by algebra that at zero context the two-compartment neuron *is*
    the baseline with a per-channel threshold `thr/(1 + w_c)`, and that this
    accounts for 40.6 % of its gain. The multiplicative form is used here to match
    that algebra. `thr_gain = 1` nests the baseline exactly.
    """
    g = self.thr_gain[layer]
    v = v0
    spikes = []
    for t in range(cur.shape[1]):
        v_pre = v * self.beta + cur[:, t]
        s = atan_spike(v_pre - self.threshold * g, self.surrogate_alpha)
        v = v_pre * (1.0 - s)
        spikes.append(s)
    return torch.stack(spikes, dim=1), v


def _scan_adaptive_threshold(self, cur: Tensor, v0: Tensor, layer: int):
    """Candidate #6: spike-driven adaptive threshold (adLIF), two parameters.

    State is [B, 2d]: the membrane, then the adaptation variable.

        a_t     = rho_c * a_{t-1} + s_{t-1}
        thr_t   = thr + b_c * a_t
        v_t     = beta * v_{t-1} + cur_t
        s_t     = 1[v_t >= thr_t]                 hard reset on v

    `b_c = 0` nests the baseline exactly and is therefore the attractive init. It
    is also a saddle for `rho_c`, by exactly the argument `EXP_004` §2.2 makes for
    the two-compartment mix:

        dL/da_t = b_c * dL/dthr_t + rho_c * dL/da_{t+1},   dL/da_L = 0

    so at `b_c = 0` every `dL/da_t` is zero by backward induction, and
    `dL/drho_c = sum_t (dL/da_t) * a_{t-1}` with it. `b_c` itself stays reachable
    because `a` integrates the spike train whatever `b_c` is.
    """
    d = cur.shape[2]
    v, a = v0[:, :d], v0[:, d:]
    b = self.adapt_b[layer]
    rho = torch.sigmoid(self.adapt_rho_raw[layer])
    spikes = []
    for t in range(cur.shape[1]):
        v_pre = v * self.beta + cur[:, t]
        s = atan_spike(v_pre - (self.threshold + b * a), self.surrogate_alpha)
        v = v_pre * (1.0 - s)
        a = rho * a + s
        spikes.append(s)
    return torch.stack(spikes, dim=1), torch.cat([v, a], dim=1)


def _scan_rotational(self, cur: Tensor, v0: Tensor, layer: int):
    """Candidate #10: rotational / complex membrane, two parameters.

    State is [B, 2d]: the real part, then the imaginary part.

        (re_t, im_t) = r_c * R(theta_c) @ (re_{t-1}, im_{t-1}) + (cur_t, 0)
        s_t          = 1[re_t >= thr]             fire on the real part
        re_t        <- re_t * (1 - s_t)           hard reset on the real part only

    §6.4 derived, on paper and without running anything, that the proposed
    `theta = 0` with `im_0 = 0` and no direct readout of `im` is an **exact**
    gradient saddle for `theta`: `im` stays identically zero, so `dL/dre`'s
    theta-path is multiplied by `im = 0`, while `dL/dim` is itself zero because
    `im` reaches the loss only through `sin(theta) = 0`. That derivation is what
    §6.4 called "the clearest case in the phase of a candidate whose ROI column
    would otherwise have been fiction", and it has never been measured. This is
    the measurement.
    """
    d = cur.shape[2]
    re, im = v0[:, :d], v0[:, d:]
    r = torch.sigmoid(self.rot_r_raw[layer])
    th = self.rot_theta[layer]
    cos_t, sin_t = torch.cos(th), torch.sin(th)
    spikes = []
    for t in range(cur.shape[1]):
        re_p = r * (cos_t * re - sin_t * im) + cur[:, t]
        im_p = r * (sin_t * re + cos_t * im)
        s = atan_spike(re_p - self.threshold, self.surrogate_alpha)
        re = re_p * (1.0 - s)
        im = im_p
        spikes.append(s)
    return torch.stack(spikes, dim=1), torch.cat([re, im], dim=1)


# -- synthetic controls: the screen's own gate --------------------------------

def _scan_ctl_live(self, cur: Tensor, v0: Tensor, layer: int):
    """Reachable by construction: a per-channel gain straight onto the current."""
    return _lif_baseline(self, cur * self.ctl_gain[layer], v0, layer)


def _scan_ctl_dead(self, cur: Tensor, v0: Tensor, layer: int):
    """Unreachable by construction: multiplied by exactly 0.0.

    In the graph, so `.grad` is populated; multiplied by zero, so it is populated
    with exactly zero. If the screen does not flag this, the zero leg is dead.
    """
    return _lif_baseline(self, cur + 0.0 * self.ctl_dead[layer], v0, layer)


def _scan_ctl_faint(self, cur: Tensor, v0: Tensor, layer: int):
    """Reachable, but ~1e-9 of the scale. Trips the RATIO leg and not the zero leg.

    Written as `(p - 1)` so the forward is the baseline's exactly at the init and
    the control cannot be flagged for having perturbed the model instead.
    """
    return _lif_baseline(
        self, cur + 1e-9 * (self.ctl_faint[layer] - 1.0), v0, layer
    )


# ==========================================================================
# The registry
# ==========================================================================

class Candidate:
    """One screenable arm: new parameters, an eager prototype, a derivation.

    `derived_zero` is the set of parameter names the candidate's own backward
    equations say will be *exactly* zero at this init. It is written before the
    run and is never edited to match a measurement.
    """

    def __init__(self, key: str, candidate: str, init_label: str, *,
                 params: tuple, scan, state_mult: int = 1,
                 derived_zero: tuple = (), derivation: str = "",
                 fallback: str | None = None, arch: str = "snn",
                 cfg_overrides: dict | None = None, control: str | None = None,
                 post_hoc_zero: tuple = (), post_hoc_reason: str = ""):
        self.key = key
        self.candidate = candidate
        self.init_label = init_label
        self.params = params            # ((name, init_value), ...)
        self.scan = scan
        self.state_mult = state_mult
        self.derived_zero = derived_zero
        self.derivation = derivation
        self.fallback = fallback
        self.arch = arch
        self.cfg_overrides = cfg_overrides or {}
        self.control = control          # "pass" | "zero" | "ratio" | None
        # Saddles the pre-run derivation MISSED, added after a measurement and
        # labelled as such. `derived_zero` above is never edited to match a
        # result -- the experiment logs' rule, applied to source. A candidate
        # carrying a `post_hoc_zero` is one the screen taught us something about,
        # and the split between the two fields is the record of that.
        self.post_hoc_zero = post_hoc_zero
        self.post_hoc_reason = post_hoc_reason
        if bool(post_hoc_zero) != bool(post_hoc_reason):
            raise ValueError(f"{key}: post_hoc_zero and post_hoc_reason go together")

    @property
    def expected_zero(self) -> list:
        return sorted(set(self.derived_zero) | set(self.post_hoc_zero))

    @property
    def param_names(self) -> tuple:
        if self.arch == "twocomp":
            return ("w", "beta_s_raw")
        return tuple(name for name, _ in self.params)


CONTROLS = [
    Candidate(
        "ctl_live", "control (synthetic)", "per-channel gain on the current",
        params=(("ctl_gain", 1.0),), scan=_scan_ctl_live,
        derivation="reachable by construction", control="pass",
    ),
    Candidate(
        "ctl_dead", "control (synthetic)", "parameter multiplied by exactly 0.0",
        params=(("ctl_dead", 1.0),), scan=_scan_ctl_dead,
        derived_zero=("ctl_dead",),
        derivation="dL/dp = 0.0 * upstream, exactly", control="zero",
    ),
    Candidate(
        "ctl_faint", "control (synthetic)", "contribution scaled by 1e-9",
        params=(("ctl_faint", 1.0),), scan=_scan_ctl_faint,
        derivation="reachable, but ~1e-9 of the layer weights' scale",
        control="ratio",
    ),
]

CANDIDATES = [
    # -- #1, through the real arch, so it reproduces the committed EXP_004 JSON --
    Candidate(
        "c01_w0", "#1 two-compartment neuron", "proposed (exact nesting), w = 0",
        params=(), scan=None, arch="twocomp",
        cfg_overrides={"w_init": 0.0, "beta_slow": 0.95},
        derived_zero=("beta_s_raw",),
        derivation="dL/dvs_t = w*dL/dv_t + beta_s*dL/dvs_{t+1}, dL/dvs_L = 0; "
                   "at w = 0 every dL/dvs_t is zero by backward induction",
        fallback="c01_w01",
    ),
    Candidate(
        "c01_w01", "#1 two-compartment neuron", "pre-registered fallback, w = 0.1",
        params=(), scan=None, arch="twocomp",
        cfg_overrides={"w_init": 0.1, "beta_slow": 0.95},
        derivation="w != 0 opens the mix path, so both parameters carry gradient",
    ),
    # -- #2 -------------------------------------------------------------------
    Candidate(
        "c02", "#2 current normalisation (RMSNorm on cur)", "gain = 1",
        params=(("rms_gain", 1.0),), scan=_scan_rms_current,
        derivation="the gain multiplies the current on every path to the loss",
    ),
    # -- #5 -------------------------------------------------------------------
    Candidate(
        "c05", "#5 learned per-channel decay", "beta_raw = logit(0.5), nests baseline",
        params=(("beta_raw", 0.0),), scan=_scan_learned_decay,
        derivation="dL/dbeta_c = sum_t (dL/dv_pre_t) * v_{t-1}; v_{t-1} is a "
                   "nonzero state and its adjoint is not suppressed, so the "
                   "exact-nesting init is reachable",
    ),
    # -- #6, both inits -------------------------------------------------------
    Candidate(
        "c06_b0", "#6 adaptive threshold (adLIF)", "proposed (exact nesting), b = 0",
        params=(("adapt_b", 0.0), ("adapt_rho_raw", _logit(0.9))),
        scan=_scan_adaptive_threshold, state_mult=2,
        derived_zero=("adapt_rho_raw",),
        derivation="dL/da_t = b*dL/dthr_t + rho*dL/da_{t+1}, dL/da_L = 0; at "
                   "b = 0 every dL/da_t is zero, and dL/drho with it",
        fallback="c06_b01",
    ),
    Candidate(
        "c06_b01", "#6 adaptive threshold (adLIF)", "fallback, b = 0.1",
        params=(("adapt_b", 0.1), ("adapt_rho_raw", _logit(0.9))),
        scan=_scan_adaptive_threshold, state_mult=2,
        derivation="b != 0 opens the adaptation path",
        post_hoc_zero=("adapt_b", "adapt_rho_raw"),
        post_hoc_reason=(
            "MEASURED, NOT DERIVED -- the first general run of this screen "
            "(2026-08-03) disagreed with the line above, and the disagreement was "
            "the real finding. Both parameters are exactly zero in LAYER 1, and "
            "not because of the b = 0 saddle: layer 1's firing rate at this init "
            "is exactly 0.0, so the spike-driven adaptation variable "
            "a_t = rho*a_{t-1} + s_{t-1} is identically zero and takes both of its "
            "parameters' gradients with it. Raising b from 0 to 0.1 lifts the "
            "threshold, which drops layer 0's rate from 0.0518 to 0.0447, which "
            "tips layer 1 from 2.98e-07 to silent. This is the initialisation "
            "pathology of `audit_07_init_pathology.json` and EXP_003 R4, not a "
            "property of the candidate: a SPIKE-DRIVEN parameter in a silent layer "
            "is unreachable at every init of its own. Candidate #6 is therefore "
            "not screenable independently of candidate #8 (variance-scaled "
            "initialisation), which #6.3 ranked only as a prerequisite for deep "
            "arms. The pre-run derivation is left above, unedited."),
    ),
    # -- #10, both inits ------------------------------------------------------
    Candidate(
        "c10_th0", "#10 rotational / complex membrane", "proposed, theta = 0, im_0 = 0",
        params=(("rot_r_raw", _logit(0.5)), ("rot_theta", 0.0)),
        scan=_scan_rotational, state_mult=2,
        derived_zero=("rot_theta",),
        derivation="§6.4: at theta = 0 with im_0 = 0 the imaginary component has "
                   "no downstream influence, so dL/dim_t = 0 for all t and "
                   "dL/dtheta = sum_t dL/dre_t * (-r*im_{t-1}) = 0 exactly",
        fallback="c10_thpi8",
    ),
    Candidate(
        "c10_thpi8", "#10 rotational / complex membrane", "fallback, theta = pi/8",
        params=(("rot_r_raw", _logit(0.5)), ("rot_theta", math.pi / 8)),
        scan=_scan_rotational, state_mult=2,
        derivation="theta != 0 couples re and im, so both parameters carry gradient",
    ),
    # -- EXP_004 §10.6's follow-up -------------------------------------------
    Candidate(
        "thr", "EXP_004 §10.6 learned per-channel threshold", "gain = 1, nests baseline",
        params=(("thr_gain", 1.0),), scan=_scan_learned_threshold,
        derivation="dL/dg = -thr * sum_t (dL/d(v_pre - thr*g)); the surrogate "
                   "derivative is nonzero, so an exact-nesting init is reachable "
                   "here even though #1's is not",
    ),
]

#: Ranked candidates the screen does not apply to, listed rather than omitted.
OUT_OF_SCOPE = {
    "#3":  "depth K = 4 -- no new parameter; width is re-derived, not added",
    "#4":  "readout capacity -- new parameters, but ordinary Linear layers "
           "outside the time loop with no adjoint recursion to collapse",
    "#7":  "reset ablation -- a free parameter of the existing kernel, not a "
           "learnable one",
    "#8":  "variance-scaled initialisation -- rescales existing parameters",
    "#9":  "firing-rate set-point -- a loss term plus a threshold constant; the "
           "learnable-threshold half is screened as `thr`",
    "#11": "parameter-scaling pilot -- no new parameter kind",
    "#12": "surrogate width sweep -- alpha is a hyperparameter, not learned",
    "#13": "training budget -- no parameter",
    "#14": "re-measure sigma -- a measurement, not an arm",
}


# ==========================================================================
# Model construction
# ==========================================================================

def _make_init_state(mult: int):
    def init_state(self, batch_size: int, device=None):
        if device is None:
            device = self.embed.weight.device
        return [
            torch.zeros(batch_size, mult * self.d_model, device=device,
                        dtype=torch.float32)
            for _ in range(self.n_layers)
        ]
    return init_state


def build_screen_model(cfg: Config, cand: Candidate) -> nn.Module:
    """Build the host stack and graft the candidate's parameters and scan onto it.

    The host is the ordinary Phase-2 `SpikingCharLM` (or, for #1, the real
    `TwoCompartmentCharLM`), constructed by `build_model` from the same seed as
    every other arm. The embedding, the layer projections and the head therefore
    hold *the baseline's* initial weights, and the ratio each candidate is scored
    against is a gradient the baseline arm is known to train on successfully --
    not one produced by a synthetic batch or a randomly-initialised stand-in.

    Grafting rather than subclassing keeps every candidate on `_CharLMStack`'s one
    forward loop, which is the same reason the analogue control is identical to
    the spiking arm by construction rather than by careful bookkeeping.
    """
    seed_everything(cfg.seed, cfg.deterministic)
    model = build_model(cfg)
    if cand.arch == "twocomp":
        return model              # already owns w / beta_s_raw

    d = cfg.d_model
    for name, init in cand.params:
        setattr(model, name, nn.ParameterList([
            nn.Parameter(torch.full((1, d), float(init)))
            for _ in range(cfg.n_layers)
        ]))
    model.to(cfg.device)
    if cand.state_mult != 1:
        model.init_state = types.MethodType(_make_init_state(cand.state_mult), model)
    model._scan = types.MethodType(cand.scan, model)
    return model


# ==========================================================================
# The screen itself
# ==========================================================================

def _grad_stats(t: Tensor | None) -> dict:
    if t is None:
        return {"present": False}
    return {
        "present": True,
        "numel": int(t.numel()),
        "abs_max": float(t.abs().max()),
        "rms": float(t.pow(2).mean().sqrt()),
        "l2": float(t.norm()),
        "exactly_zero": bool(float(t.abs().max()) == 0.0),
    }


def screen(cfg: Config, corpus: Corpus, cand: Candidate) -> dict:
    """One forward/backward at the candidate's proposed init.

    Deliberately a single step on the real corpus, the real batch and the real
    cross-entropy. The quantity of interest is a *ratio* against `dL/dW`, and that
    reference only means anything if it is the one training will actually see.
    """
    model = build_screen_model(cfg, cand)
    model.train()

    sampler = RandomWindowSampler(corpus.split("train"), cfg.batch_size,
                                  cfg.seq_len, cfg.seed)
    x, y = sampler.batch(0)
    x, y = x.to(cfg.device), y.to(cfg.device)

    logits, _state, aux = model(x, None)
    loss = F.cross_entropy(logits.reshape(-1, cfg.vocab_size).float(),
                           y.reshape(-1))
    loss.backward()

    # The reference: the layer projections, the parameters the baseline arm
    # already trains successfully. Pooled over layers, so one candidate cannot be
    # scored generously by a layer whose weights happen to have a small gradient.
    w_grads = [ly.weight.grad for ly in model.layers]
    if any(g is None for g in w_grads):
        raise RuntimeError(f"{cand.key}: a layer projection received no gradient; "
                           "the prototype has disconnected the graph")
    w_rms = float(torch.stack([g.pow(2).mean() for g in w_grads]).mean().sqrt())

    names = cand.param_names
    per_layer = []
    for k in range(cfg.n_layers):
        row = {
            "layer": k,
            "firing_rate": float(aux["firing_rate"][k]),
            "linear_weight": _grad_stats(model.layers[k].weight.grad),
        }
        for name in names:
            st = _grad_stats(getattr(model, name)[k].grad)
            st["rms_vs_linear_weight"] = (st["rms"] / w_rms) if st["present"] else None
            st["zero_leg"] = bool(st.get("exactly_zero"))
            st["ratio_leg"] = bool(
                not st.get("exactly_zero")
                and st.get("rms_vs_linear_weight") is not None
                and st["rms_vs_linear_weight"] < RATIO_FLOOR
            )
            st["flagged"] = bool(st["zero_leg"] or st["ratio_leg"])
            row[name] = st
        per_layer.append(row)

    # A layer that never fires makes every SPIKE-DRIVEN parameter in it trivially
    # unreachable, whatever that parameter's own initialisation is. Recorded
    # explicitly, because a verdict that is really about the baseline's known
    # initialisation pathology must not be read as a property of the candidate.
    silent = [r["layer"] for r in per_layer if r["firing_rate"] == 0.0]

    zeros = [f"layer{r['layer']}.{n}" for r in per_layer for n in names
             if r[n]["zero_leg"]]
    faint = [f"layer{r['layer']}.{n}" for r in per_layer for n in names
             if r[n]["ratio_leg"]]
    for r in per_layer:
        for n in names:
            r[n]["in_silent_layer"] = bool(r["layer"] in silent)

    # Measurement against the derivation written before the run. Reported at the
    # parameter level, because "some layer was zero" is not the same claim as
    # "this parameter is unreachable".
    measured_zero = sorted({n for r in per_layer for n in names if r[n]["zero_leg"]})
    derived = sorted(cand.derived_zero)
    verdict = "SADDLE" if zeros else ("FAINT" if faint else "PASS")

    return {
        "key": cand.key,
        "candidate": cand.candidate,
        "init": cand.init_label,
        "derivation": cand.derivation,
        "derived_zero": derived,
        "post_hoc_zero": sorted(cand.post_hoc_zero),
        "post_hoc_reason": cand.post_hoc_reason,
        "expected_zero": cand.expected_zero,
        "measured_zero": measured_zero,
        "derivation_agrees": cand.expected_zero == measured_zero,
        "predicted_before_any_run": derived == measured_zero,
        "silent_layers": silent,
        "fallback": cand.fallback,
        "new_params": list(names),
        "loss": float(loss.detach()),
        "train_bpc": float(loss.detach()) / math.log(2.0),
        "linear_weight_grad_rms": w_rms,
        "per_layer": per_layer,
        "exactly_zero": zeros,
        "below_ratio_floor": faint,
        "verdict": verdict,
        "passes_screen": verdict == "PASS",
    }


# ==========================================================================
# The screen's own gates
# ==========================================================================

def check_controls(results: list[dict]) -> dict:
    """A screen nothing has tripped is a screen nobody has tested.

    Each control names in advance which leg of the rule must fire. The run aborts
    on any disagreement -- a broken screen that reports PASS on every candidate is
    strictly worse than no screen, because it would be quoted as evidence.
    """
    want = {c.key: c.control for c in CONTROLS}
    rows, ok = [], True
    for r in results:
        expect = want.get(r["key"])
        if expect is None:
            continue
        if expect == "pass":
            got = r["verdict"] == "PASS"
        elif expect == "zero":
            got = bool(r["exactly_zero"]) and not r["below_ratio_floor"]
        else:  # "ratio"
            got = bool(r["below_ratio_floor"]) and not r["exactly_zero"]
        ok &= got
        rows.append({"control": r["key"], "expected_leg": expect,
                     "verdict": r["verdict"], "held": bool(got)})
    return {"all_held": bool(ok), "controls": rows}


def _walk(node, path=""):
    """Flatten the numeric leaves of a committed JSON block for comparison."""
    if isinstance(node, dict):
        for k, v in node.items():
            yield from _walk(v, f"{path}.{k}" if path else k)
    elif isinstance(node, list):
        for i, v in enumerate(node):
            yield from _walk(v, f"{path}[{i}]")
    elif isinstance(node, (int, float)) and not isinstance(node, bool):
        yield path, float(node)


def check_reproduction(results: list[dict]) -> dict:
    """Candidate #1 must reproduce a number this script did not produce.

    `docs/reports/data/exp_004_gradient_reachability.json` was written by a
    different script before this one existed. Same seed, same batch, same eager
    path, same device -- so this is a genuine re-derivation of a committed
    quantity, and it is the check that would notice if the model, the sampler or
    the corpus had drifted underneath the screen.

    Only the quantities both scripts define are compared, and they are compared by
    name rather than by position.
    """
    path = _REPO / EXP_004_JSON
    if not path.exists():
        return {"ran": False, "reason": f"{EXP_004_JSON} not present"}
    committed = json.loads(path.read_text(encoding="utf-8"))["inits"]
    by_key = {r["key"]: r for r in results}

    rows, worst, ok = [], 0.0, True
    for key, ckey in (("c01_w0", "w_init=0.0"), ("c01_w01", "w_init=0.1")):
        if key not in by_key or ckey not in committed:
            continue
        ours, theirs = by_key[key], committed[ckey]["eager"]
        fields = [("loss", ours["loss"], theirs["loss"]),
                  ("linear_weight_grad_rms", ours["linear_weight_grad_rms"],
                   theirs["linear_weight_grad_rms"])]
        for k in range(len(ours["per_layer"])):
            a, b = ours["per_layer"][k], theirs["per_layer"][k]
            for name in ("w", "beta_s_raw"):
                for stat in ("rms", "abs_max", "rms_vs_linear_weight"):
                    fields.append((f"layer{k}.{name}.{stat}", a[name][stat],
                                   b[name][stat]))
            fields.append((f"layer{k}.firing_rate", a["firing_rate"],
                           b["firing_rate"]))
        for name, got, ref in fields:
            denom = abs(ref) if abs(ref) > 0 else 1.0
            rel = abs(got - ref) / denom
            worst = max(worst, rel)
            if rel > REPRO_RTOL:
                ok = False
                rows.append({"init": ckey, "field": name, "ours": got,
                             "committed": ref, "rel": rel})
    return {"ran": True, "source": EXP_004_JSON, "rtol": REPRO_RTOL,
            "worst_rel_residual": worst, "mismatches": rows, "held": bool(ok)}


# ==========================================================================
# CLI
# ==========================================================================

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="§7.2 gradient-reachability screen (03_phase3_candidates.md)")
    ap.add_argument("--out",
                    default="docs/reports/data/audit_09_gradient_reachability.json")
    ap.add_argument("--only", default="",
                    help="comma-separated candidate keys; default is all")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip-reproduction", action="store_true",
                    help="skip the EXP_004 cross-check (it needs CUDA and the "
                         "baseline shape)")
    args = ap.parse_args(argv)

    base = Config(cuda_graph=False, fused=False, seed=args.seed,
                  device="cuda" if torch.cuda.is_available() else "cpu")
    corpus = Corpus(base.corpus, base.data_dir)
    base.vocab_size = int(corpus.vocab_size)

    wanted = set(filter(None, args.only.split(",")))
    todo = [c for c in (CONTROLS + CANDIDATES) if not wanted or c.key in wanted]

    print(f"device {base.device}   B={base.batch_size} L={base.seq_len} "
          f"d={base.d_model} K={base.n_layers}   ratio floor 1/{int(1/RATIO_FLOOR)}")

    results = []
    for cand in todo:
        cfg = Config(**{**base.to_dict(), "arch": cand.arch, **cand.cfg_overrides})
        r = screen(cfg, corpus, cand)
        results.append(r)
        mark = {"PASS": "pass", "FAINT": "FAINT", "SADDLE": "SADDLE"}[r["verdict"]]
        print(f"\n  {cand.key:12s} {cand.candidate} -- {cand.init_label}")
        for row in r["per_layer"]:
            bits = "  ".join(
                f"|dL/d{n}| {row[n]['rms']:.3e} "
                f"(x{row[n]['rms_vs_linear_weight']:.3f})"
                for n in r["new_params"])
            print(f"      layer {row['layer']}  rate {row['firing_rate']:.4f}  "
                  f"|dL/dW| {row['linear_weight']['rms']:.3e}   {bits}")
        agree = "agrees" if r["derivation_agrees"] else "DISAGREES WITH DERIVATION"
        if r["post_hoc_zero"]:
            agree += f"; {r['post_hoc_zero']} added post-hoc, see JSON"
        if r["silent_layers"]:
            agree += f"; layer(s) {r['silent_layers']} SILENT at init"
        print(f"      -> {mark}   expected zero {r['expected_zero'] or '[]'}, "
              f"measured {r['measured_zero'] or '[]'} ({agree})")

    controls = check_controls(results)
    repro = ({"ran": False, "reason": "--skip-reproduction"}
             if args.skip_reproduction else check_reproduction(results))

    payload = {
        "audit": "09_gradient_reachability",
        "screen": "03_phase3_candidates.md §7.2",
        "rule": "flag a new parameter whose |dL/dp| is exactly zero (fatal: AdamW "
                "cannot move it), or whose per-element RMS is below 1/100 of the "
                "layer weights' (worth recording; Adam is scale-free, so this is "
                "not on its own a reason to reject an init)",
        "ratio_floor": RATIO_FLOOR,
        "torch": torch.__version__,
        "device": base.device,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "shape": {"batch_size": base.batch_size, "seq_len": base.seq_len,
                  "d_model": base.d_model, "n_layers": base.n_layers,
                  "seed": args.seed},
        "self_test": controls,
        "reproduction": repro,
        "out_of_scope": OUT_OF_SCOPE,
        "results": results,
    }
    out = _REPO / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("\n" + "=" * 78)
    print("§7.2 GRADIENT REACHABILITY")
    print("=" * 78)
    print(f"{'key':13s} {'candidate / init':52s} {'verdict':>9s}")
    for r in results:
        label = f"{r['candidate']} -- {r['init']}"
        print(f"{r['key']:13s} {label[:52]:52s} {r['verdict']:>9s}")

    print(f"\nself-test (can the screen fail?):  "
          f"{'all controls held' if controls['all_held'] else 'BROKEN'}")
    for row in controls["controls"]:
        print(f"    {row['control']:12s} expected {row['expected_leg']:5s} leg -> "
              f"{row['verdict']:6s} {'ok' if row['held'] else 'WRONG'}")

    if repro.get("ran"):
        print(f"\nreproduction of {EXP_004_JSON}: "
              f"{'held' if repro['held'] else 'FAILED'}   "
              f"worst relative residual {repro['worst_rel_residual']:.2e}")
        for m in repro["mismatches"][:8]:
            print(f"    {m['init']} {m['field']}: {m['ours']!r} vs "
                  f"{m['committed']!r} ({m['rel']:.2e})")
    else:
        print(f"\nreproduction: not run ({repro.get('reason')})")

    disagree = [r["key"] for r in results if not r["derivation_agrees"]]
    print(f"\nderivation vs measurement: "
          f"{'all agree' if not disagree else 'DISAGREE: ' + ', '.join(disagree)}")

    taught = [r["key"] for r in results if r["post_hoc_zero"]]
    if taught:
        print(f"    saddles the screen found that no derivation predicted: "
              f"{', '.join(taught)}")
    silent = [(r["key"], r["silent_layers"]) for r in results if r["silent_layers"]]
    if silent:
        print("    layers silent at init (spike-driven parameters there are "
              "unreachable at ANY init of their own):")
        for key, layers in silent:
            print(f"        {key:12s} layer(s) {layers}")
    print(f"\nWROTE {args.out}")

    # The run fails if the screen cannot be trusted -- a broken screen reporting
    # PASS is worse than no screen. A candidate FAILING the screen is a result,
    # not an error, and does not fail the run.
    bad = (not controls["all_held"]) or bool(disagree) or (
        repro.get("ran") and not repro["held"])
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
