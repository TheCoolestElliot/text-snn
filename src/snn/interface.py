"""The two ends of the model: how a character becomes current, and how spikes
become logits.

Pre-registered in `experiments/logs/EXP_020_interface.md`. Read that first --
every init constant here is DERIVED there before it was measured, and none of
them is tuned.

Nothing in this file is a new neuron. There is no new scan, no new recurrence
and no hand-written backward, so there is no new R10 gate: `snn.neuron.lif_scan`
and its 59-mutation campaign are untouched, and G1 asserts that. What this file
changes is what happens **before** the first scan starts and **after** the last
one ends -- both of which are O(1) kernels per layer against the scan's `K*L`,
which is the same slot `snn.prescan`, `snn.noise` and `snn.dopamine` occupy and
for the same reason.

THE FOLD, WHICH IS AN IDENTITY AND NOT AN APPROXIMATION
--------------------------------------------------------
`_CharLMStack.forward` computes

    h    = E[x]                      the embedding, [B, L, d]
    cur0 = h @ W0^T + b0             layer 0's input current

`E[x]` is a *lookup*: it is one of `V` rows, never a mixture. A linear map
applied to a lookup is a lookup of the mapped table, so with `T = E @ W0^T`

    cur0[b, l] = T[x[b,l]] + b0                                   EXACTLY

**`layers.0.weight` therefore spans no function the embedding table does not
already span.** At the committed shape (`V = 205`, `d = 512`, `K = 2`) that
matrix is 262,144 of the model's 735,437 parameters -- **35.7 %** -- and it is
the *only* parameter block in the stack that is provably redundant with another
one. Measured on `snn_beta0.5_s0`: max abs difference between the two-step and
the folded current is **1.9e-06**, relative **5.2e-08**, i.e. fp32 rounding of a
`d`-term dot product and nothing else.

This is `EXP_008`'s Identity 2 one layer earlier, and the precedent cuts both
ways, which is why the fold is measured rather than assumed:
**`EXP_008` found that the redundantly-parameterised form trains BETTER**, by
0.0590 bpc, on a function class it had proved identical. A reparameterisation
that provably cannot change what the model can express can still change what the
optimiser finds. So `fold` at unchanged width is not an efficiency claim, it is
the control that prices the factorisation; `fold_wide` at matched parameters is
the candidate.

WHAT THE FOLD FREES, AND THE WIDTH IT BUYS
-------------------------------------------
The closed form at `K = 2` is `d^2 + 412d + 205` once `layers.0` is gone
against `2d^2 + 412d + 205` with it. Holding the total at or below the committed
735,437 takes `d` from **512 to 675** -- 733,930 parameters, 99.80 % of the
anchor -- a **1.32x** width at (very slightly under) the same budget.
`match_width_for_params` returns 675 and not 676 because 676 costs 735,693,
which is 256 parameters OVER the anchor; the arm is width-matched, not
parameter-identical, and `EXP_020` G2 reports the realised counts rather than
claiming an identity it does not have. `EXP_014` is the standing result that
width is worth more than every architectural arm this project has built
(-0.25238 bpc at 54.75 sigma).
`EXP_014` also says what that gain is: **per-step capacity, not reach** (horizon
7 -> 8). This arm therefore predicts a within-reach gain and no horizon change,
and §4 of the pre-registration puts the bar there rather than on the mean.

BINARY INPUT CODING, AND WHY IT IS NOT RATE CODING
---------------------------------------------------
I1 says binary spikes carry all information *between* layers, and the committed
model honours it everywhere except at its first layer, which receives
`nn.Embedding`'s real-valued row -- "analogue direct coding, T=1"
(`model.py:349`). `BinaryInputCode` closes that exception: the character's code
is itself passed through the project's own `atan_spike`, so

    B = 1[shadow_c(x) >= thr_in]        exactly {0, 1}, same nonlinearity,
                                        same surrogate, same alpha as a neuron

and layer 0 then receives a spike pattern through `layers.0`'s synapses exactly
as layer 1 receives layer 0's. **Every inter-unit signal in the model is then
binary, and the only real numbers left are membranes, weights and logits** --
which is what I1 was always about, and what the literature keeps in floating
point too (`01_reconnaissance.md` §4.3 item 2).

This is NOT the Poisson rate coding `01a_literature_review.md` §3.1 reports
losing to direct coding, and that finding does not transfer: rate coding spends
`T` timesteps per character to represent one analogue value in a spike count,
which is why it needs 150 of them. This spends **one** timestep, `T = 1`, and
represents the character as a *spatial* pattern across `d` fibres. The axis
being binarised is the channel axis, not the time axis. The cost of rate coding
-- `T`x the kernels -- is exactly the cost this does not pay.

`shadow` is a real-valued parameter and stays one. So do the synapses. That is
the same arrangement as everywhere else in the model: analogue weights,
analogue membranes, binary activations.

THE INITS ARE DERIVED, NOT TUNED
---------------------------------
`02_baseline_report.md` §5.2 measured the one initialisation pathology this
architecture has, and it is a statement about exactly this interface: layer 0
sees dense `N(0,1)` and fires at 4.9 %, layer 1 sees a spike train at density
`p` where `E[s^2] = p` rather than 1, so its current's sd is `sqrt(p/3)` and its
threshold is **7.8 sigma** away. **Layer 1 never fires at init.** Anything that
changes what layer 0 is handed must therefore state what it does to that
arithmetic, in advance, or it is changing two things at once.

    baseline layer 0:   E[x^2] = 1        =>  sd(cur) = sqrt(1/3) = 0.5774

Every init below is chosen to land on that same 0.5774 and on nothing else:

    fold        T ~ N(0, 1/3).  Var(E W0^T) = d * Var(E) * Var(W0)
                = d * 1 * 1/(3d) = 1/3 exactly, so the folded table is
                initialised to the distribution the product it replaces had.

    binary_in   cur = W0 @ g*(B - q).  Matching the second moment the GEMM was
                calibrated for, E[x^2] = g^2 q(1-q) = 1, gives
                g = 1/sqrt(q(1-q)), which at q = 0.5 is 2.

    fold_binary the code IS the current, so it matches the current's own
                variance rather than the GEMM's input moment:
                Var = g^2 q(1-q) = 1/3  =>  g = 1/sqrt(3 q(1-q)) = 1.1547.

**CENTRING IS A REPARAMETERISATION OF `layers.0.bias`, NOT AN ANALOGUE
SIGNAL.** This has to be established, because `snn.noise`'s docstring draws
exactly the distinction that would otherwise sink this arm: a centred Bernoulli
is "two-valued but NOT {0,1}", i.e. analogue. Here it is not, and the algebra
says why:

    W0 @ (g*(B - q)) + b0  =  (g*W0) @ B  +  (b0 - g*q*W0 @ 1)
                              ^^^^^^^^^^^     ^^^^^^^^^^^^^^^^
                              a GEMM on the   a TOKEN-INDEPENDENT
                              raw {0,1} code  constant

and `b0` is a free learned parameter, so the second term is representable
exactly. **Layer 0 therefore receives the `{0,1}` pattern `B` through synapses
`g*W0` plus a bias — which is precisely the arrangement under which layer 1
receives layer 0's emission.** The gain and the offset are properties of the
synapse, not of the signal. Verified: max relative difference between the two
forms is 5.8e-07, fp32 rounding of a 512-term dot product.

The consequence is the whole point of the arm. Counting inter-unit signals per
position at `d = 512`, `K = 2` — token to layer 0, layer 0 to layer 1, layer 1
to the head — the committed model is binary on **1024 of 1536** wires, 66.7 %,
and this arm is binary on **1536 of 1536**. What stays real-valued is exactly
what I2 requires (membranes), what synapses are (weights), and what
`01_reconnaissance.md` §4.3 item 2 keeps in floating point on purpose (the head).

**The uncentred alternative is a real trap and is recorded as one.** An uncentred
`g*B` has the right marginal second moment and the wrong structure: writing
`S_c = sum_j W0[c,j]`, the per-channel current is

    cur_c(x) = g * sum_{j : B[x,j]=1} W0[c,j]
             = g*q*S_c            + (a token-dependent term)
               ^^^^^^^^^ a FIXED per-channel offset, sd 0.408 at q = 0.5

so half the channels are biased toward firing at every character and half away
from it, before a single token is read. Measured at d = 512: the uncentred code
fires layer 0 at **0.082** against the baseline's 0.0482, a 1.7x mismatch that
would have travelled through the whole arm as an unstated hyperparameter.
Centring removes the offset exactly, because `E[B - q] = 0` by construction.
`snn.noise` centres its Bernoulli draw by the identical construction and for a
related reason, and `noise_scale`'s `1/sqrt(p(1-p))` is the same factor as
`binary_in`'s gain.

None of these constants was measured first and none is swept. If an arm fails,
the failure is the arm's and not an unstated hyperparameter's.

THE READOUT, AND WHAT IT IS NOT ALLOWED TO BE
-----------------------------------------------
`logits = self.head(h)` reads the last layer's spike vector at `t` and nothing
else. Two things are discarded there, and both are already in memory:

  * every other layer's emission at `t` -- layer 0's spikes drive layer 1 and
    are then dropped;
  * the same layer's emission at `t-1, t-2, ...` -- a strided view of a tensor
    that already exists.

`readout` concatenates the requested `(layer, lag)` blocks. Both extensions cost
**zero** kernels inside the time loop and one `cat` outside it, and -- this is
the constraint that shapes the whole function -- **both keep the head's input
strictly binary.** Reading the membrane instead would be a real-valued signal on
the inter-unit path, which is the thing I1 exists to forbid; it is measured in
`020_readout_probe.py` as a reference and is not an arm.

A lag is padded with zeros INSIDE the window, exactly as `snn.prescan.shift_by_one`
pads, so position `t` of one window can never see a character from the window
before it. That matters more here than in the pre-scan: this tensor feeds the
head directly, so a leak would be a leak of the *answer*.

Causality: `s[t-r]` depends on `x[<= t-r]`, and the target at `t` is `x[t+1]`.
A lagged readout therefore reads strictly less of the future than the unlagged
one does, which is to say none. `tests/test_interface.py`'s C1 asserts it by
perturbation rather than by argument.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor

from snn.surrogate import atan_spike

__all__ = [
    "fold_input_table",
    "fold_residual",
    "BinaryInputCode",
    "readout",
    "readout_width",
    "folded_param_count",
    "match_width_for_params",
]


# ---------------------------------------------------------------------------
# the fold
# ---------------------------------------------------------------------------

def fold_input_table(embed_weight: Tensor, w0_weight: Tensor,
                     w0_bias: Tensor | None = None) -> tuple[Tensor, Tensor]:
    """`(E, W0, b0) -> (T, b0)` with `T = E @ W0^T`. Returns `(T [V, d], b [d])`.

    The identity `E[x] @ W0^T + b0 == T[x] + b0` holds because `E[x]` is a
    lookup, not a mixture. In fp32 it holds to the rounding of a `d`-term dot
    product and no better, which is why `fold_residual` exists and why no
    tolerance is asserted here: decision #6 established that a fold-in tolerance
    **cannot be one project constant** -- the same identity through the same fold
    costs 2.0e-03 bpc on the baseline neuron and 1.4e-05 on the two-compartment
    one. The residual is measured and reported per arm; it is not inherited.
    """
    if embed_weight.dim() != 2:
        raise ValueError(f"embed_weight must be [V, d]; got {tuple(embed_weight.shape)}")
    if w0_weight.dim() != 2:
        raise ValueError(f"w0_weight must be [d, d]; got {tuple(w0_weight.shape)}")
    if w0_weight.shape[0] != w0_weight.shape[1]:
        raise ValueError(f"w0_weight must be square; got {tuple(w0_weight.shape)}")
    if embed_weight.shape[1] != w0_weight.shape[1]:
        raise ValueError(
            f"embed_weight {tuple(embed_weight.shape)} does not compose with "
            f"w0_weight {tuple(w0_weight.shape)}"
        )
    table = embed_weight @ w0_weight.t()
    bias = torch.zeros(table.shape[1], dtype=table.dtype, device=table.device)
    if w0_bias is not None:
        bias = w0_bias.clone()
    return table, bias


@torch.no_grad()
def fold_residual(embed_weight: Tensor, w0_weight: Tensor, w0_bias: Tensor,
                  idx: Tensor) -> dict:
    """Measure the fold's fp32 residual on real token ids. Reported, not asserted.

    Returns max/mean absolute difference and the max relative difference between
    the two-step current and the folded one, plus the fraction of elements that
    are bitwise equal.
    """
    table, bias = fold_input_table(embed_weight, w0_weight, w0_bias)
    two_step = F.linear(F.embedding(idx, embed_weight), w0_weight, w0_bias)
    folded = F.embedding(idx, table) + bias
    diff = (two_step - folded).abs()
    scale = two_step.abs().max().clamp_min(torch.finfo(two_step.dtype).tiny)
    return {
        "max_abs": float(diff.max()),
        "mean_abs": float(diff.mean()),
        "max_rel": float(diff.max() / scale),
        "frac_bitwise_equal": float((two_step == folded).float().mean()),
        "n_elements": int(diff.numel()),
    }


# ---------------------------------------------------------------------------
# the binary input code
# ---------------------------------------------------------------------------

class BinaryInputCode(torch.nn.Module):
    """A learned `{0,1}` code per character, through the project's own spike.

        B[v, c] = 1[shadow[v, c] >= thr_in]          hard, exactly {0.0, 1.0}
        out     = gain * B[x]                        [B, L, d]

    The forward is `snn.surrogate.atan_spike`, unchanged and unre-implemented, so
    the input code fires by exactly the rule a neuron fires by and carries
    exactly the surrogate a neuron carries. That is not a convenience: a second
    threshold nonlinearity with its own shape would be a second free
    hyperparameter, and `01_reconnaissance.md` §4.3 item 3 froze the surrogate
    shape precisely so that it is not one.

    The table is binarised ONCE per forward -- `[V, d]` with `V = 205`, not
    `[B, L, d]` -- and then gathered. Two kernels for the whole sequence, outside
    the time loop.

    `gain` is a single scalar buffer, not a parameter, and `centre` is a Python
    float. Both are DERIVED in the module docstring, and a per-channel learned
    gain is deliberately not offered here: by `EXP_008` Identity 1 a per-channel
    gain on the current is exactly a per-channel threshold, which is an adopted
    arm in its own right (#5), and folding it in silently would compose two arms
    while claiming to measure one.

    TWO KNOBS, AND THEY ARE DELIBERATELY ORTHOGONAL (`EXP_022`)
    -------------------------------------------------------------
    `EXP_020` measured this code at `q = 0.5` only, and found it costs 0.0448
    bpc with **83 % of that at zero context** -- a per-character expressiveness
    cost, not a memory cost. It is NOT an information-capacity cost: at
    `d = 512, q = 0.5` the code carries 507 bits against the alphabet's
    `log2(205) = 7.7`, and even `q = 0.05` carries 145. So the cost is about how
    the code interacts with the GEMM and with its own threshold, and this class
    exposes exactly the two axes that can move that:

      `thr_in`   sets the DENSITY `q`. Sparser codes are more localist: at
                 `q = 0.10` one bit flip moves layer 0's current by `g` times a
                 single column of `W0` against a sum of ~51 terms, where at
                 `q = 0.5` it moves it against a sum of ~256 -- ~8x the relative
                 leverage per bit. `q = 0.34` additionally makes the input code
                 statistically **identical to what every other layer receives**,
                 since that is the converged firing rate the stack runs at
                 (`EXP_000` F3), which is the strongest form of the I1
                 completeness argument: the input boundary then looks like every
                 other boundary rather than merely being binary at it.

      `code_sd`  sets the PLASTICITY, at fixed density. The code trains through
                 `atan_spike`'s surrogate, whose gradient falls off as
                 `1/(1 + (pi*alpha*x/2)^2)` in the distance `x` from threshold,
                 so a shadow spread over `N(0, 1)` has most of its bits already
                 attenuated at step 0 and the far ones effectively frozen.
                 Shrinking `code_sd` moves every bit closer to its threshold
                 without moving `q` at all -- `q` depends on `thr_in/code_sd`
                 alone, so at `thr_in = 0` it is 0.5 for every `code_sd`, and
                 the derived `gain` and `centre` are therefore unchanged too.

    **The two knobs are needed together because `thr_in` alone confounds them.**
    Raising `thr_in` to sparsify also pushes the typical bit FURTHER from its
    threshold (at `q = 0.10` the typical shadow sits 1.28 away), so a density
    ladder run on its own moves sparsity and plasticity in opposite directions
    and cannot say which one it measured. `code_sd` is the axis that separates
    them, at `q = 0.5` where the density is held exactly at `EXP_020`'s measured
    point.

    **`code_sd` is implemented and tested but NOT RUN.** `EXP_022` §2.3, §2.5 and
    §6 item 1 all say so, and its driver freezes the field; the axis is priced
    there at ~0.35 GPU-h and referred. This docstring previously claimed
    `EXP_022` runs it, which is the opposite of what that pre-registration says.

    Neither knob costs a parameter, and neither changes the second moment layer
    0 receives: `gain` is re-derived from the realised `q` on every path.
    """

    def __init__(self, vocab_size: int, d_model: int, *, thr_in: float = 0.0,
                 alpha: float = 2.0, is_current: bool = False,
                 code_sd: float = 1.0) -> None:
        super().__init__()
        if vocab_size <= 0 or d_model <= 0:
            raise ValueError("vocab_size and d_model must be positive")
        if code_sd <= 0.0:
            raise ValueError(f"code_sd must be > 0, got {code_sd!r}")
        self.vocab_size = int(vocab_size)
        self.d_model = int(d_model)
        self.thr_in = float(thr_in)
        self.alpha = float(alpha)
        self.is_current = bool(is_current)
        self.code_sd = float(code_sd)

        # N(0, code_sd^2) shadow against thr_in gives density
        # q = P(shadow >= thr_in) = 1 - Phi(thr_in / code_sd). At the defaults
        # (thr_in = 0, code_sd = 1) that is 0.5. The shadow is the parameter;
        # the code is its threshold image.
        self.shadow = torch.nn.Parameter(
            torch.randn(vocab_size, d_model) * code_sd
        )

        z = thr_in / code_sd
        q = 0.5 if z == 0.0 else float(
            1.0 - 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))
        )
        self.init_density = q
        var = q * (1.0 - q)
        # Centred on BOTH paths -- see the module docstring on the per-channel
        # offset an uncentred code carries. The two paths differ only in WHICH
        # moment they match: a code feeding a GEMM matches the unit second
        # moment the GEMM's init was calibrated for; a code that IS the current
        # matches the current's own variance, 1/3.
        gain = (math.sqrt(1.0 / 3.0) if is_current else 1.0) / math.sqrt(
            max(var, 1e-12)
        )
        self.register_buffer("gain", torch.tensor(gain, dtype=torch.float32))
        self.register_buffer("centre", torch.tensor(q, dtype=torch.float32))

    def code(self) -> Tensor:
        """The `[V, d]` binary code table. Exactly {0.0, 1.0}."""
        return atan_spike(self.shadow - self.thr_in, self.alpha)

    def density(self) -> Tensor:
        """Mean code density -- a diagnostic, detached, never in the loss."""
        with torch.no_grad():
            return self.code().mean()

    def forward(self, idx: Tensor) -> Tensor:
        if idx.dim() != 2:
            raise ValueError(f"idx must be [B, L]; got {tuple(idx.shape)}")
        return F.embedding(idx, self.gain * (self.code() - self.centre))


# ---------------------------------------------------------------------------
# the readout
# ---------------------------------------------------------------------------

def _lag(x: Tensor, r: int) -> Tensor:
    """Delay by `r` inside the window, zero-padded in front.

    `F.pad` rather than an in-place write into a zeros buffer, for the reason
    `snn.prescan.shift_by_one` and `snn.dopamine.rpe` both give: the in-place
    form mutates a buffer whose address a CUDA-graph capture bakes in, and this
    runs inside the captured training step.
    """
    if r == 0:
        return x
    return F.pad(x[:, :-r], (0, 0, r, 0))


def readout(spikes: list[Tensor], layers: tuple[int, ...],
            lags: tuple[int, ...]) -> Tensor:
    """Concatenate `(layer, lag)` spike blocks into the head's input.

    `spikes[k]` is layer `k`'s emission `[B, L, d]`. Block order is
    `(layer, lag)` lexicographic and is FIXED: the head's columns must mean the
    same thing on every forward pass and in every checkpoint, and an order that
    depended on argument order would make two checkpoints with the same config
    hash describe different models.

    Returns `[B, L, len(layers)*len(lags)*d]`, binary wherever its inputs are.
    """
    if not spikes:
        raise ValueError("spikes must be non-empty")
    if not layers or not lags:
        raise ValueError("layers and lags must both be non-empty")
    n = len(spikes)
    for k in layers:
        if not -n <= k < n:
            raise IndexError(f"layer {k} out of range for {n} layers")
    if any(r < 0 for r in lags):
        raise ValueError(f"lags must be >= 0; got {lags}")
    # UNIQUENESS IS CHECKED ON THE NORMALISED INDICES, not on the spelling.
    # `-1` and `n-1` are the same layer, so checking `set(layers)` before the
    # `% n` wrap below let two spellings of one layer through the guard and emit
    # the same block twice -- exactly the rank-deficiency the guard exists to
    # forbid, reached by the one route it did not look at.
    norm_layers = [k % n for k in layers]
    if len(set(norm_layers)) != len(norm_layers) or len(set(lags)) != len(lags):
        # A repeated block is exactly collinear with itself, which makes the
        # head rank-deficient by construction and is never intended.
        raise ValueError(
            f"layers {layers} (normalised {norm_layers}) and lags {lags} must "
            "each be unique")

    # SORTED, not argument order. The order is a property of the checkpoint --
    # the head's columns must mean the same thing on every forward pass -- so it
    # may not depend on how a caller happened to spell its arguments. Every call
    # `Config` can produce is already ascending, so this changes no committed
    # run; it makes the paragraph above true of every call rather than of the
    # careful ones.
    order = [(k, r) for k in norm_layers for r in lags]
    blocks = [_lag(spikes[k], r) for k, r in sorted(order)]
    return blocks[0] if len(blocks) == 1 else torch.cat(blocks, dim=-1)


def readout_width(d_model: int, n_layers_read: int, n_lags: int) -> int:
    return int(d_model * n_layers_read * n_lags)


# ---------------------------------------------------------------------------
# parameter accounting
# ---------------------------------------------------------------------------

def folded_param_count(vocab_size: int, d_model: int, n_layers: int, *,
                       folded: bool, read_blocks: int = 1,
                       binary_input: bool = False) -> int:
    """Parameters of an interface arm. Mirrors `snn.model.spiking_param_count`.

    `folded`       layer 0's GEMM is gone; the table IS the current, plus a bias
    `read_blocks`  how many `[B, L, d]` blocks the head reads
    `binary_input` the input table is a shadow of the same shape (so the count
                   is unchanged by binarising it -- which is the point: the arm
                   costs nothing in parameters and changes only the code's
                   alphabet)
    """
    V, d, K = int(vocab_size), int(d_model), int(n_layers)
    table = V * d + (d if folded else 0)          # +d for the folded bias
    gemms = (K - 1 if folded else K) * (d * d + d)
    head = read_blocks * d * V + V
    return table + gemms + head


def match_width_for_params(target: int, vocab_size: int, n_layers: int, *,
                           folded: bool, read_blocks: int = 1,
                           binary_input: bool = False) -> int:
    """Largest `d` whose arm does not exceed `target` parameters.

    Integer search rather than the quadratic root, because the root is only the
    right answer for `K = 2` and the caller must not have to know that. Width
    matching IS parameter matching here only up to the integer grid; the gate
    that matters (`G2`) asserts the realised counts, and the driver records both.
    """
    lo, hi = 1, 1
    while folded_param_count(vocab_size, hi, n_layers, folded=folded,
                             read_blocks=read_blocks,
                             binary_input=binary_input) < target:
        hi *= 2
        if hi > 1 << 20:
            raise ValueError("target parameter count is unreachable")
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if folded_param_count(vocab_size, mid, n_layers, folded=folded,
                              read_blocks=read_blocks,
                              binary_input=binary_input) <= target:
            lo = mid
        else:
            hi = mid - 1
    return lo
