"""Phase-2 models: the spiking baseline and the two controls it is measured against.

The architecture is dictated by the Phase-1 cost model, not by aesthetics. On this
machine wall-clock is ~14.5 us x (CUDA kernels issued), so kernel *count* is the
performance metric (01_reconnaissance.md §3.4). The naive spiking-LM loop puts a
GEMM inside the time loop and issues K*L GEMMs plus ~13*K*L elementwise kernels
per forward pass.

Because invariant I5 forbids a lateral recurrent matrix and there is no
cross-layer feedback, layer k's input current at time t depends only on layer
k-1's output at time t. The layers can therefore be evaluated sequentially in
*depth*, each computing its entire input current for all L timesteps in ONE GEMM
before its time loop starts:

    h = embed(x)                    # [B, L, d]
    for k in range(K):
        cur  = linear_k(h)          # [B, L, d]   one GEMM for the whole sequence
        h, v = lif_scan(cur, v0)    # [B, L, d]   L fused elementwise kernels
    logits = head(h)                # [B, L, V]

That is K+2 GEMMs plus K*L fused elementwise kernels -- and the time loop contains
exactly one kernel per step per layer, the floor for a hard-threshold neuron with
reset. This is exact, not an approximation: nothing at layer k, time t depends on
layer k+1 at any time, nor on layer k-1 at any time > t. See 02a_phase2_spec.md §0.

It is also the first place invariant I5 *pays* for itself computationally rather
than costing.

Three arms live here:

  SpikingCharLM   the model under test. Satisfies I1-I5.
  AnalogueCharLM  identical parameterisation, continuous emission instead of a
                  binary spike. Isolates I1+I3 as the single variable.
  GRUCharLM       external anchor at matched parameter count. Deliberately
                  violates I5; labelled as such wherever it appears.

Precision policy (01_reconnaissance.md §3.11 C2, bottleneck B8): membrane state
and the threshold comparison are ALWAYS fp32. `cfg.dtype` selects the dtype of the
hidden GEMMs only. The embedding and the output head are fp32 and never spike
(§4.3 item 2 -- every competitive spiking LM keeps them in floating point).
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn
from torch import Tensor

from snn.neuron import lif_scan
from snn.surrogate import atan_value
from snn.twocomp import twocomp_scan

if TYPE_CHECKING:  # avoid a hard import cycle / hard dependency at runtime
    from snn.config import Config

__all__ = [
    "SpikingCharLM",
    "AnalogueCharLM",
    "TwoCompartmentCharLM",
    "GRUCharLM",
    "build_model",
    "count_params",
    "spiking_param_count",
    "twocomp_param_count",
    "gru_param_count",
    "match_gru_width",
    "resolve_gemm_dtype",
    "analogue_scan",
    "RESET_MODES",
]

RESET_MODES = ("hard", "soft", "detached", "none")

_GEMM_DTYPES = {
    "fp32": torch.float32,
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
}


def resolve_gemm_dtype(name: str) -> torch.dtype:
    """Map a config dtype string onto a torch dtype for the hidden GEMMs only."""
    try:
        return _GEMM_DTYPES[name]
    except KeyError:
        raise ValueError(
            f"dtype must be one of {sorted(_GEMM_DTYPES)}, got {name!r}"
        ) from None


# ---------------------------------------------------------------------------
# Parameter accounting
# ---------------------------------------------------------------------------


def count_params(model: nn.Module) -> int:
    """Total number of parameter elements.

    Counts every registered parameter, trainable or not, so that the reported
    figure is the model's size and not an optimiser-dependent quantity. None of
    the Phase-2 arms freeze anything, so the two coincide here.
    """
    return sum(p.numel() for p in model.parameters())


def spiking_param_count(vocab_size: int, d_model: int, n_layers: int) -> int:
    """Closed form for SpikingCharLM / AnalogueCharLM.

    V*d (embedding) + K*(d*d + d) (layer projections) + d*V + V (head).
    At the Phase-2 baseline V=205, d=512, K=2 this is 735_437 (spec §13).
    """
    return (
        vocab_size * d_model
        + n_layers * (d_model * d_model + d_model)
        + d_model * vocab_size
        + vocab_size
    )


def gru_param_count(vocab_size: int, hidden_size: int, n_layers: int) -> int:
    """Closed form for GRUCharLM, whose embedding width equals its hidden width.

    torch's nn.GRU carries both b_ih and b_hh, so each layer is
    3H*in + 3H*H + 6H; with in == H that is 6H^2 + 6H.
    """
    h = hidden_size
    n = vocab_size * h  # embedding
    for _ in range(n_layers):
        n += 3 * h * h + 3 * h * h + 6 * h  # W_ih, W_hh, b_ih, b_hh
    n += h * vocab_size + vocab_size  # head
    return n


def match_gru_width(
    vocab_size: int, n_layers: int, target_params: int, tol: float = 0.02
) -> int:
    """Smallest-error GRU hidden width whose parameter count matches `target_params`.

    The GRU control only means something if it is the same size as the arm it is
    controlling for, so the width is derived rather than chosen. `gru_param_count`
    is strictly increasing in the width, so scanning up to the crossover and
    comparing the two neighbours is exact.

    Raises if no width lands within `tol` (relative), because silently shipping a
    mismatched control is worse than failing to build one.
    """
    if target_params <= 0:
        raise ValueError("target_params must be positive")
    h = 1
    while (
        gru_param_count(vocab_size, h, n_layers) < target_params and h < 1 << 16
    ):
        h += 1
    candidates = [h] if h == 1 else [h - 1, h]
    best = min(
        candidates,
        key=lambda w: abs(gru_param_count(vocab_size, w, n_layers) - target_params),
    )
    err = abs(gru_param_count(vocab_size, best, n_layers) - target_params) / target_params
    if err > tol:
        raise ValueError(
            f"no GRU width matches {target_params} params to within {tol:.1%}; "
            f"best is width {best} at {gru_param_count(vocab_size, best, n_layers)} "
            f"({err:.2%} off)"
        )
    return best


# ---------------------------------------------------------------------------
# Shared feed-forward stack
# ---------------------------------------------------------------------------


class _CharLMStack(nn.Module):
    """embedding -> [Linear -> neuron] x K -> Linear head.

    Private base shared by SpikingCharLM and AnalogueCharLM. The two differ in
    exactly one method, `_scan`, which is the point of the analogue control: the
    parameterisation is *identical by construction* rather than identical by
    careful bookkeeping, so a future edit cannot silently desynchronise them.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_layers: int,
        *,
        beta: float = 0.5,
        threshold: float = 1.0,
        reset: str = "hard",
        surrogate_alpha: float = 2.0,
        dtype: str = "fp32",
        fused: bool = True,
        t_steps: int = 1,
    ) -> None:
        super().__init__()
        if vocab_size <= 0:
            raise ValueError("vocab_size must be set (>0) before building a model")
        if n_layers < 1:
            raise ValueError("n_layers must be >= 1")
        if reset not in RESET_MODES:
            raise ValueError(f"reset must be one of {RESET_MODES}, got {reset!r}")
        if t_steps != 1:
            # T is a declared free parameter (01_reconnaissance.md §1.3) but the
            # frozen Phase-2 baseline is T=1 (spec §13) and T>1 changes the shape
            # of the readout, which is not specified. Fail loudly rather than
            # ignore the flag.
            raise NotImplementedError(
                "t_steps > 1 is a Phase-3/4 ablation; the Phase-2 baseline is T=1"
            )

        self.vocab_size = int(vocab_size)
        self.d_model = int(d_model)
        self.n_layers = int(n_layers)
        self.beta = float(beta)
        self.threshold = float(threshold)
        self.reset = str(reset)
        self.surrogate_alpha = float(surrogate_alpha)
        self.dtype_name = str(dtype)
        self.gemm_dtype = resolve_gemm_dtype(dtype)
        self.fused = bool(fused)
        self.t_steps = int(t_steps)

        # Opt-in diagnostic: keeps a reference to each layer's emitted tensor in
        # aux["spikes"]. Off by default because under no_grad it would keep
        # K * B*L*d floats alive that would otherwise be freed. Tests and any
        # future firing-rate regulariser need the un-reduced tensors.
        self.keep_spikes = False

        # fp32, non-spiking (Phase-1 report §4.3 item 2).
        self.embed = nn.Embedding(vocab_size, d_model)
        # One GEMM per layer for the WHOLE sequence -- never inside the time loop.
        self.layers = nn.ModuleList(
            [nn.Linear(d_model, d_model, bias=True) for _ in range(n_layers)]
        )
        # fp32, non-spiking.
        self.head = nn.Linear(d_model, vocab_size, bias=True)

    # -- state ------------------------------------------------------------

    def init_state(
        self, batch_size: int, device: torch.device | str | None = None
    ) -> list[Tensor]:
        """Zeroed membrane state, one [B, d] fp32 tensor per layer.

        fp32 unconditionally: the threshold comparison must not be quantised
        (§3.11 C2 -- bf16's spacing at v=1.0 is 0.78% of the threshold).
        """
        if device is None:
            device = self.embed.weight.device
        return [
            torch.zeros(batch_size, self.d_model, device=device, dtype=torch.float32)
            for _ in range(self.n_layers)
        ]

    # -- the one GEMM per layer ------------------------------------------

    def _project(self, linear: nn.Linear, x: Tensor) -> Tensor:
        """Apply `linear` to the entire [B, L, d] sequence in a single GEMM.

        Parameters stay fp32 (they are the optimiser's master weights); only the
        GEMM operands are cast, and the result is cast straight back to fp32
        before it can touch a membrane. Casting costs three kernels per layer per
        forward pass, against K*L in the time loop -- irrelevant.

        CAVEAT, recorded because it is invisible at the call site: the non-fp32
        branch calls `F.linear` directly rather than `linear(x)`, so nothing
        registered on the module fires -- no forward hook, no parametrisation, no
        `__torch_function__` interception of the module call. Anything that
        instruments the layers (including
        `test_one_gemm_per_layer_regardless_of_sequence_length`, which counts
        `nn.Linear` forward hooks) is only measuring the fp32 path. There is no
        way to keep the module call *and* cast the operands without also casting
        the stored parameters, which §2 forbids.
        """
        if self.gemm_dtype is torch.float32:
            return linear(x)
        bias = linear.bias
        out = torch.nn.functional.linear(
            x.to(self.gemm_dtype),
            linear.weight.to(self.gemm_dtype),
            None if bias is None else bias.to(self.gemm_dtype),
        )
        return out.float()

    # -- the neuron, supplied by the subclass -----------------------------

    def _scan(self, cur: Tensor, v0: Tensor, layer: int) -> tuple[Tensor, Tensor]:
        """`layer` is the index of the layer being scanned.

        The Phase-2 arms ignore it: their neuron owns no parameters, which is the
        structural form of I5 (`snn.neuron.lif_scan` takes tensors and scalars
        only). `TwoCompartmentCharLM` needs it, because its per-channel decay and
        mix are per layer. Passing the index rather than letting the subclass
        override `forward` keeps every arm on one forward loop -- the same reason
        the analogue control is identical to the spiking arm *by construction*
        rather than by careful bookkeeping.
        """
        raise NotImplementedError

    # -- forward ----------------------------------------------------------

    def forward(
        self, idx: Tensor, state: list[Tensor] | None = None
    ) -> tuple[Tensor, list[Tensor], dict[str, Any]]:
        """idx [B, L] int64 -> (logits [B, L, V], new_state, aux).

        `state` is the per-layer membrane potential carried in from the previous
        window (evaluation protocol B); None means zeroed (protocol A). The
        returned state is NOT detached -- carried-state callers detach it, which
        keeps truncated BPTT available without a second code path.

        aux["firing_rate"] is a list of K scalar tensors, the mean emission rate
        of each layer for this batch. Detached, because it is a diagnostic (R4)
        and must never leak into the loss by accident; a firing-rate regulariser
        would build on aux["spikes"] instead.
        """
        if idx.dim() != 2:
            raise ValueError(f"idx must be [B, L], got shape {tuple(idx.shape)}")
        batch_size = idx.shape[0]

        if state is None:
            state = self.init_state(batch_size, idx.device)
        elif len(state) != self.n_layers:
            raise ValueError(
                f"state must hold {self.n_layers} tensors, got {len(state)}"
            )

        h = self.embed(idx)  # [B, L, d] fp32 -- analogue direct coding, T=1
        new_state: list[Tensor] = []
        rates: list[Tensor] = []
        spikes: list[Tensor] = []

        for k, linear in enumerate(self.layers):
            cur = self._project(linear, h)  # ONE GEMM, whole sequence
            emitted, v_final = self._scan(cur, state[k], k)
            h = emitted
            new_state.append(v_final)
            rates.append(emitted.detach().mean())
            if self.keep_spikes:
                spikes.append(emitted)

        logits = self.head(h)  # fp32, non-spiking

        aux: dict[str, Any] = {"firing_rate": rates}
        if self.keep_spikes:
            aux["spikes"] = spikes
        return logits, new_state, aux


# ---------------------------------------------------------------------------
# Arm 1: the model under test
# ---------------------------------------------------------------------------


class SpikingCharLM(_CharLMStack):
    """embedding -> [Linear -> LIF] x K -> Linear head.

    Invariants: I1 binary spikes between layers, I2 leaky integration,
    I3 hard threshold, I4 surrogate BPTT, I5 no lateral recurrent matrix.
    Embedding and head are fp32 and non-spiking (Phase-1 report §4.3 item 2).

    I5 holds structurally: the only [d, d] parameters are `layers.k.weight`, and
    each is applied to the *previous* layer's output before the scan begins. The
    scan itself (`snn.neuron.lif_scan`) takes tensors and scalars only -- it owns
    no weights -- so the temporal recurrence is diagonal in the channel axis and
    no layer's output can be mixed back into itself. tests/test_model.py asserts
    both the structural and the functional (channel-diagonal) form of this.

    Layer 0's input is the embedding (analogue direct coding, T=1); layers
    1..K-1 receive binary spikes, valued {0.0, 1.0} in fp32 because they feed a
    GEMM directly.
    """

    def _scan(self, cur: Tensor, v0: Tensor, layer: int) -> tuple[Tensor, Tensor]:
        return lif_scan(
            cur,
            v0,
            self.beta,
            self.threshold,
            self.surrogate_alpha,
            self.reset,
            self.fused,
        )


# ---------------------------------------------------------------------------
# Arm 2: the analogue control (isolates I1 + I3)
# ---------------------------------------------------------------------------


def analogue_scan(
    cur: Tensor, v0: Tensor, beta: float, thr: float, alpha: float, reset: str
) -> tuple[Tensor, Tensor]:
    """Leaky integration with a *continuous* emission instead of a spike.

    Identical to `lif_scan_eager` except that the hard threshold is removed: the
    neuron emits `atan_value(v_pre - thr, alpha)` -- the surrogate's own value
    function, which is the smooth counterpart of the step it stands in for -- and
    the reset consumes that continuous value rather than a binary spike.

    Using the surrogate's value function (rather than, say, a sigmoid or a ReLU)
    is deliberate: it makes the control differ from the spiking arm in exactly the
    thing under test. Where the spiking forward uses `s_hard` and the backward
    uses `d(atan_value)/dx`, this arm uses `atan_value` in both, so the *local*
    emission derivative d(emitted)/d(v_pre) is the same function `atan_grad` in
    both arms. The single variable is therefore binarity + the hard threshold
    (I1 + I3), not the shape of the nonlinearity.

    Stated precisely, because the weaker claim "only the forward differs" is
    false: the reset feeds the emitted value back into the membrane, so under
    `hard` reset the spiking arm's dv_next/dv_pre is `(1 - s_hard) - v_pre*sg`
    while this arm's is `(1 - a) - v_pre*sg`. Those differ exactly to the extent
    that `a` differs from `s_hard`, i.e. exactly by the variable under test.

    Range is (0, 1) with value 1/2 at threshold, so the emission is a soft spike
    and firing-rate bookkeeping stays comparable. Note the fp32 caveat, measured:
    `atan_value` saturates to *exactly* 1.0 for `v_pre - thr >= ~1e7` and to
    exactly 0.0 below ~-1e7, because atan itself saturates to float32(pi/2) long
    before that. A healthy run never gets near it -- and under `hard` reset the
    scan self-limits, since v_next = v_pre*(1-a) -> 0 as a -> 1 -- but under
    `soft` or `none` reset a diverging run can, and if it does, this control
    stops being strictly non-binary. Treat an emission of exactly 0.0 or 1.0 as a
    divergence signal, not as a spike.

    This path is ordinary autograd -- there is no fused kernel for it and it is
    not needed: the control arm is not the arm whose throughput matters.
    """
    if reset not in RESET_MODES:
        raise ValueError(f"reset must be one of {RESET_MODES}, got {reset!r}")
    if cur.dtype != torch.float32 or v0.dtype != torch.float32:
        raise TypeError("membrane arithmetic is fp32 only")
    # Same shape contract `snn.neuron._check_shapes` enforces on the spiking
    # path. Without it a carried state of the wrong batch size ([1, d], say)
    # broadcasts silently here while raising in the snn arm -- the two arms would
    # then differ in something other than the variable under test, which is
    # exactly what a control must not do.
    if cur.dim() != 3:
        raise ValueError(f"cur must be [B, L, d]; got shape {tuple(cur.shape)}")
    if v0.dim() != 2:
        raise ValueError(f"v0 must be [B, d]; got shape {tuple(v0.shape)}")
    if v0.shape[0] != cur.shape[0] or v0.shape[1] != cur.shape[2]:
        raise ValueError(
            f"v0 shape {tuple(v0.shape)} does not match cur {tuple(cur.shape)}"
        )

    length = cur.shape[1]
    v = v0
    out: list[Tensor] = []
    for t in range(length):
        v_pre = v * beta + cur[:, t]
        a = atan_value(v_pre - thr, alpha)
        if reset == "hard":
            v = v_pre * (1.0 - a)
        elif reset == "soft":
            v = v_pre - thr * a
        elif reset == "detached":
            v = v_pre * (1.0 - a.detach())
        else:  # "none"
            v = v_pre
        out.append(a)
    return torch.stack(out, dim=1), v


class AnalogueCharLM(_CharLMStack):
    """Identical to SpikingCharLM in every respect except that the neuron emits
    `atan_value(v_pre - thr, alpha)` continuously instead of a binary spike, and
    the reset uses that continuous value. Isolates I1+I3 (binarity and the hard
    threshold) as the single variable. Parameter count is *identical*, not merely
    matched.

    "Identical" is enforced by construction: both arms are `_CharLMStack` and
    differ only in `_scan`, which owns no parameters. Same modules, same shapes,
    same initialisation order -- so with the same seed the two arms even start
    from the same weights.

    Violates I1 and I3 by design; preserves I2 (leaky integration) and I5 (no
    lateral recurrent matrix). I4 -- "surrogate BPTT" -- does not apply here and
    must not be claimed: with the hard threshold removed the emission is exactly
    differentiable, so there is no surrogate standing in for anything. What is
    shared with the spiking arm is the *function* d(emission)/d(v_pre), which is
    `atan_grad` in both; that is why the control isolates I1+I3 rather than
    confounding them with the shape of the nonlinearity. Label it accordingly in
    every table.
    """

    def _scan(self, cur: Tensor, v0: Tensor, layer: int) -> tuple[Tensor, Tensor]:
        return analogue_scan(
            cur, v0, self.beta, self.threshold, self.surrogate_alpha, self.reset
        )


# ---------------------------------------------------------------------------
# Arm 4: the Phase-4 candidate (EXP_004, candidate #1)
# ---------------------------------------------------------------------------


def twocomp_param_count(vocab_size: int, d_model: int, n_layers: int) -> int:
    """Closed form for TwoCompartmentCharLM.

    The spiking count plus two `[1, d]` per-channel parameters per layer. At
    V=205, d=512, K=2 this is 735_437 + 2_048 = 737_485, which is **+0.28%** over
    the Phase-2 baseline. EXP_004 §6.1 names that as a limitation rather than
    correcting for it: shedding 2_048 parameters means d = 511, which changes the
    width -- a second variable, introduced to control for a smaller one.
    """
    return spiking_param_count(vocab_size, d_model, n_layers) + 2 * n_layers * d_model


class TwoCompartmentCharLM(_CharLMStack):
    """embedding -> [Linear -> two-compartment LIF] x K -> Linear head.

    The Phase-4 candidate ranked #1 in `03_phase3_candidates.md` §6.3, pre-
    registered in `experiments/logs/EXP_004_two_compartment.md`. The neuron and the
    reasons for its shape live in `snn.twocomp`; this class is only the wiring.

    Invariants: I1 binary spikes between layers, I2 leaky integration (twice over),
    I3 hard threshold, I4 surrogate BPTT, **I5 upheld** -- the §4.6 ruling admits
    multiple learned timescales per neuron (row 2) because they are O(1) parameters
    per neuron and introduce no cross-neuron mixing. The scan owns `w` and
    `beta_s`, both `[1, d]`, both diagonal in the channel axis; there is still no
    `[d, d]` parameter anywhere except `layers.k.weight`, which is applied to the
    *previous* layer's output before the scan begins.

    Per-layer state is one `[B, 2d]` tensor -- the fast half then the slow half --
    rather than two `[B, d]` tensors, so that `train.py`, `evaluate.py`'s carried
    protocol, and `EXP_001`'s horizon probe keep one code path for every arm.

    `beta_s` is stored unconstrained and squashed with a sigmoid **outside the time
    loop**: one kernel per layer per forward pass, against K*L inside it. The mix
    `w` is stored directly and is deliberately unconstrained -- nothing requires it
    to be positive, and forcing it would be an unlogged hyperparameter choice.
    """

    def __init__(self, *args, beta_slow: float = 0.95, w_init: float = 0.1,
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if not 0.0 < beta_slow < 1.0:
            raise ValueError(f"beta_slow must lie in (0, 1), got {beta_slow!r}")
        self.beta_slow_init = float(beta_slow)
        self.w_init = float(w_init)

        d = self.d_model
        # logit(beta_slow): the sigmoid's own inverse, so the realised initial
        # decay is beta_slow to fp32 rather than approximately it.
        raw = math.log(beta_slow / (1.0 - beta_slow))
        self.w = nn.ParameterList(
            [nn.Parameter(torch.full((1, d), float(w_init))) for _ in range(self.n_layers)]
        )
        self.beta_s_raw = nn.ParameterList(
            [nn.Parameter(torch.full((1, d), float(raw))) for _ in range(self.n_layers)]
        )

    def init_state(
        self, batch_size: int, device: torch.device | str | None = None
    ) -> list[Tensor]:
        """One [B, 2d] fp32 tensor per layer: `[:, :d]` fast, `[:, d:]` slow.

        fp32 unconditionally, for the same reason as every other arm: the
        threshold comparison must not be quantised (§3.11 C2).
        """
        if device is None:
            device = self.embed.weight.device
        return [
            torch.zeros(batch_size, 2 * self.d_model, device=device,
                        dtype=torch.float32)
            for _ in range(self.n_layers)
        ]

    def slow_decay(self, layer: int) -> Tensor:
        """The realised `beta_s` for one layer, `[1, d]` in (0, 1).

        A method rather than an inline expression so that the reachability screen,
        the gradient gate and the results scripts all read the same quantity the
        forward pass uses, instead of each re-deriving it from `beta_s_raw`.
        """
        return torch.sigmoid(self.beta_s_raw[layer])

    def _scan(self, cur: Tensor, v0: Tensor, layer: int) -> tuple[Tensor, Tensor]:
        return twocomp_scan(
            cur,
            v0,
            self.w[layer],
            self.slow_decay(layer),   # one kernel per layer, NOT per timestep
            self.beta,                # beta_f: the baseline's fixed fast pole
            self.threshold,
            self.surrogate_alpha,
            self.fused,
        )


# ---------------------------------------------------------------------------
# Arm 3: the external anchor (deliberately violates I5)
# ---------------------------------------------------------------------------


class GRUCharLM(nn.Module):
    """External anchor: nn.GRU at matched parameter count. Deliberately violates
    I5 and is labelled as such in every table it appears in. Exists so that the
    text8 5M-param literature anchors (LSTM 1.50, TCN 1.45) can be connected to
    something trained under our exact protocol.

    The I5 violation is concrete and nameable: `gru.weight_hh_l{k}` is a
    [3H, H] lateral matrix that maps a layer's own previous output back into
    itself. That is exactly the construction I5 forbids, which is why this arm is
    an anchor and not a result.

    The hidden width is *derived*, not chosen: `match_gru_width` picks the width
    whose parameter count is closest to the spiking arm's, and construction
    asserts the realised count is within tolerance. At V=205, d_model=512, K=2
    the spiking arm has 735_437 parameters and this arm lands at width 231 /
    738_019 parameters, +0.35%.

    The embedding width equals the hidden width so the arm has a single free
    size, the same way the spiking arm does.

    fp32 only. `cfg.dtype` is a Phase-4 ablation of the *spiking* arm's GEMMs;
    silently running the control at a different precision would confound it.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_layers: int,
        *,
        hidden_size: int | None = None,
        target_params: int | None = None,
        tol: float = 0.02,
    ) -> None:
        super().__init__()
        if vocab_size <= 0:
            raise ValueError("vocab_size must be set (>0) before building a model")
        if n_layers < 1:
            raise ValueError("n_layers must be >= 1")

        self.target_params = (
            spiking_param_count(vocab_size, d_model, n_layers)
            if target_params is None
            else int(target_params)
        )
        if hidden_size is None:
            hidden_size = match_gru_width(vocab_size, n_layers, self.target_params, tol)

        self.vocab_size = int(vocab_size)
        self.d_model = int(d_model)
        self.n_layers = int(n_layers)
        self.hidden_size = int(hidden_size)
        self.param_tol = float(tol)

        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.gru = nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=n_layers,
            bias=True,
            batch_first=True,
        )
        self.head = nn.Linear(hidden_size, vocab_size, bias=True)

        realised = count_params(self)
        rel = abs(realised - self.target_params) / self.target_params
        if rel > tol:
            raise AssertionError(
                f"GRU control missed its parameter budget: {realised} vs target "
                f"{self.target_params} ({rel:.2%} > {tol:.1%})"
            )
        self.param_error = (realised - self.target_params) / self.target_params

    def init_state(
        self, batch_size: int, device: torch.device | str | None = None
    ) -> list[Tensor]:
        """One [B, H] fp32 tensor per layer, matching the spiking arms' convention.

        nn.GRU wants a single [K, B, H] tensor; keeping the public shape a list of
        per-layer states means evaluate.py and train.py can carry state across
        windows with one code path for all three arms.
        """
        if device is None:
            device = self.embed.weight.device
        return [
            torch.zeros(batch_size, self.hidden_size, device=device, dtype=torch.float32)
            for _ in range(self.n_layers)
        ]

    def forward(
        self, idx: Tensor, state: list[Tensor] | None = None
    ) -> tuple[Tensor, list[Tensor], dict[str, Any]]:
        """idx [B, L] int64 -> (logits [B, L, V], new_state, aux).

        aux["firing_rate"] is EMPTY for this arm. There are no spikes to count,
        and emitting a plausible-looking number for a non-spiking control is how
        a log turns into a lie. aux["hidden_abs_mean"] carries the analogous
        activity measure for anyone who wants one.
        """
        if idx.dim() != 2:
            raise ValueError(f"idx must be [B, L], got shape {tuple(idx.shape)}")
        if state is not None and len(state) != self.n_layers:
            raise ValueError(
                f"state must hold {self.n_layers} tensors, got {len(state)}"
            )

        h = self.embed(idx)  # [B, L, H]
        h0 = None if state is None else torch.stack(state, dim=0).contiguous()
        out, h_n = self.gru(h, h0)  # [B, L, H], [K, B, H]
        logits = self.head(out)

        aux: dict[str, Any] = {
            "firing_rate": [],
            "hidden_abs_mean": [out.detach().abs().mean()],
        }
        return logits, list(h_n.unbind(0)), aux


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def build_model(cfg: "Config") -> nn.Module:
    """Build the arm named by `cfg.arch` and place it on `cfg.device`.

    Parameters are initialised on CPU and then moved, so the initial weights
    depend only on the seed and not on which device the run lands on.
    """
    if cfg.vocab_size <= 0:
        raise ValueError(
            "cfg.vocab_size is unset; fill it from the corpus before building"
        )
    if cfg.t_steps != 1:
        raise NotImplementedError(
            "t_steps > 1 is a Phase-3/4 ablation; the Phase-2 baseline is T=1"
        )

    arch = cfg.arch
    if arch in ("snn", "analogue", "twocomp"):
        if cfg.surrogate != "atan":
            # §4.3 item 3: surrogate shape is robust, scale is fragile. Phase 2
            # locks the shape and sweeps width instead.
            raise NotImplementedError(
                "only the arctangent surrogate is implemented in Phase 2, got "
                f"{cfg.surrogate!r}"
            )
        common = dict(
            vocab_size=cfg.vocab_size,
            d_model=cfg.d_model,
            n_layers=cfg.n_layers,
            beta=cfg.beta,
            threshold=cfg.threshold,
            reset=cfg.reset,
            surrogate_alpha=cfg.surrogate_alpha,
            dtype=cfg.dtype,
            fused=cfg.fused,
            t_steps=cfg.t_steps,
        )
        if arch == "twocomp":
            if cfg.reset != "hard":
                # EXP_004 §2 fixes the reset rule: hard on the fast pole, shielded
                # on the slow one. `cfg.reset` still selects the fast pole's rule
                # for the baseline arms, and silently ignoring it here would let a
                # run's config.json describe a neuron that never ran. The reset
                # ablation is candidate #7 and is a separate kernel.
                raise NotImplementedError(
                    "the two-compartment neuron is pre-registered with hard reset "
                    f"on its fast pole (EXP_004 §2); got reset={cfg.reset!r}. "
                    "Reset variants are candidate #7 and need their own kernel "
                    "and their own R10 gate."
                )
            model: nn.Module = TwoCompartmentCharLM(
                beta_slow=cfg.beta_slow, w_init=cfg.w_init, **common
            )
        else:
            cls = SpikingCharLM if arch == "snn" else AnalogueCharLM
            model = cls(**common)
    elif arch == "gru":
        if cfg.dtype != "fp32":
            raise NotImplementedError(
                "the GRU control runs in fp32 only; cfg.dtype ablates the spiking "
                "arm's GEMMs and changing the control's precision would confound it"
            )
        model = GRUCharLM(
            vocab_size=cfg.vocab_size,
            d_model=cfg.d_model,
            n_layers=cfg.n_layers,
        )
    else:
        raise ValueError(
            f"arch must be 'snn', 'analogue', 'twocomp' or 'gru', got {arch!r}"
        )

    return model.to(cfg.device)
