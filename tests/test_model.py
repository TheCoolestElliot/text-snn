"""Model gates (spec 02a §12, `test_model.py` row).

What this file is responsible for asserting:

  * the parameter counts that appear in the report are the counts the code
    actually builds (spec §13: 735_437 for the headline arm);
  * the `analogue` control has an *identical* parameter count, not a matched one;
  * spikes are strictly {0.0, 1.0}, and that both values actually occur (an
    all-zero layer satisfies every binarity predicate, so non-vacuity is
    asserted separately);
  * the `snn` arm contains no lateral recurrent matrix (I5);
  * the GEMM-hoisting architecture of spec §0 is real -- one GEMM per layer for
    the whole sequence, never one per timestep;
  * carried state round-trips for all three arms, which is the model-level half
    of evaluation protocol B;
  * the step-0 firing rates of the frozen baseline are what the report says they
    are, including the fact that layer 1 is silent at initialisation.

Everything here runs on CPU with `fused=False`, which exercises the eager
reference scan; `snn.neuron.lif_scan` falls back to it automatically when the
tensors are not CUDA. Two CUDA smoke tests are marked `cuda` and skip cleanly.

## How I5 is asserted, honestly

I5 -- "no explicit lateral recurrent weight matrix" -- is a statement about what
a module *does*, not only about what parameters it owns, so no single structural
check settles it. Three complementary assertions are made and their limits are
stated:

1. `test_i5_parameter_inventory_is_exhaustive` enumerates every parameter by name
   and shape. If a lateral matrix were added it would have to appear here, so an
   exact-set assertion (not a subset one) catches it. It cannot, on its own, tell
   a lateral matrix from an input projection.

2. `test_i5_no_recurrent_modules` asserts no `nn.RNNBase` subclass is present.
   Narrow, but it is the failure mode that would arise from someone "just
   swapping in an nn.GRU".

3. `test_i5_temporal_recurrence_is_channel_diagonal` is the real one, and it is
   functional rather than structural. A lateral recurrent matrix is precisely a
   mechanism by which channel j's output at time t-1 reaches channel i != j at
   time t. So the Jacobian of layer k's emitted sequence with respect to layer
   k's own bias is computed and asserted to be exactly diagonal in the channel
   axis: perturbing the current injected into channel j can change no other
   channel, at any time. That is the property I5 exists to guarantee, and it
   would fail for any lateral mixing, learned or hard-coded, anywhere in the
   layer -- including inside the scan, which this file does not otherwise see.

   Its limit: it is checked at one random initialisation on a small model, so it
   is a strong empirical assertion rather than a proof. A cross-channel term with
   an exactly zero coefficient at this initialisation would evade it. Nothing
   weaker is available without inspecting the scan's source, and nothing stronger
   is available without a symbolic argument.
"""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn as nn

from snn.metrics import bits_per_char, summarise_firing_rates
from snn.model import (
    AnalogueCharLM,
    GRUCharLM,
    SpikingCharLM,
    build_model,
    count_params,
    gru_param_count,
    match_gru_width,
    spiking_param_count,
)

# --- the frozen baseline (spec §13) ---------------------------------------
BASE_VOCAB = 205
BASE_D = 512
BASE_K = 2
# 205*512 + 2*(512^2 + 512) + 512*205 + 205
SPEC_PARAMS = 735_437

# --- the derived GRU control ----------------------------------------------
# Reported so that a change in the matching rule is a test failure, not a silent
# change in what "matched parameter count" means.
GRU_WIDTH = 231
GRU_PARAMS = 738_019

# --- small model used for the behavioural tests ---------------------------
# vocab != d_model on purpose, so an embedding matrix can never be mistaken for
# a square [d, d] recurrent one.
TINY = dict(vocab_size=11, d_model=6, n_layers=2)
TINY_B, TINY_L = 3, 7

RESETS = ("hard", "soft", "detached", "none")


def tiny_snn(reset: str = "hard", seed: int = 0, **kw) -> SpikingCharLM:
    torch.manual_seed(seed)
    return SpikingCharLM(**TINY, reset=reset, fused=False, **kw)


def tiny_idx(seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, TINY["vocab_size"], (TINY_B, TINY_L), generator=g)


def tiny_gru(seed: int = 0) -> GRUCharLM:
    """A GRU control small enough to run fast.

    `tol` is loosened only here: at ~900 parameters one unit of hidden width is
    already ~8% of the budget, so the 2% rule is arithmetically unreachable. The
    2% rule is asserted where it actually matters, on the reported baseline, in
    `test_gru_width_is_derived_and_matched`.
    """
    torch.manual_seed(seed)
    return GRUCharLM(vocab_size=11, d_model=16, n_layers=2, tol=0.10)


# ===========================================================================
# Parameter counts
# ===========================================================================


def test_spiking_param_count_matches_spec():
    """The headline arm must be the size the report claims it is."""
    model = SpikingCharLM(vocab_size=BASE_VOCAB, d_model=BASE_D, n_layers=BASE_K)
    n = count_params(model)
    assert n == SPEC_PARAMS, f"expected {SPEC_PARAMS} params, built {n}"
    # The closed form used for GRU matching must agree with the built model.
    assert spiking_param_count(BASE_VOCAB, BASE_D, BASE_K) == n
    # And with the arithmetic written out in spec §13.
    assert n == (
        BASE_VOCAB * BASE_D
        + BASE_K * (BASE_D * BASE_D + BASE_D)
        + BASE_D * BASE_VOCAB
        + BASE_VOCAB
    )
    assert 0.73e6 < n < 0.74e6  # "~0.735 M"


#: EXP_014 §2.2's ladder. The parameter count is that experiment's x-axis, so a
#: silent change to it would move every point without moving any prose. These
#: are the exact-error-minimising integer widths for the 2.5M and 5M targets --
#: within 0.06 % of nominal, which is EXP_003 §4's own standard for deriving a
#: width from a parameter target rather than picking a round one.
EXP014_LADDER = ((512, 735_437), (1020, 2_501_245), (1481, 4_997_099))


def test_exp014_ladder_widths_have_the_preregistered_parameter_counts():
    """The pre-registered x-axis, checked against a built model at each rung.

    `014_run_scaling_ladder.py`'s G3 makes the same assertion against each run's
    own `summary.json` at run time; this is its offline half, so a width that
    stopped meaning what EXP_014 §2.2 says it means fails in CI rather than
    three GPU-hours into a ladder.
    """
    for d, expected in EXP014_LADDER:
        closed_form = spiking_param_count(BASE_VOCAB, d, BASE_K)
        assert closed_form == expected, (
            f"d={d}: closed form gives {closed_form:,}, EXP_014 §2.2 "
            f"pre-registered {expected:,}"
        )
        built = count_params(
            SpikingCharLM(vocab_size=BASE_VOCAB, d_model=d, n_layers=BASE_K))
        assert built == expected, (
            f"d={d}: built model has {built:,} parameters, closed form "
            f"predicted {expected:,}"
        )


def test_exp014_ladder_hits_its_nominal_targets():
    """The rungs are labelled 0.735M / 2.5M / 5M and must deserve those labels.

    Guards the direction of the ladder too: EXP_014 §2.2 quotes 1.766 then 0.998
    octaves and says explicitly that no slope may be read as if the steps
    matched, so the asymmetry is pinned rather than left to drift.
    """
    import math

    nominal = (735_437, 2_500_000, 5_000_000)
    for (d, actual), target in zip(EXP014_LADDER, nominal):
        rel = abs(actual - target) / target
        assert rel < 1e-3, f"d={d}: {actual:,} is {rel:.2%} from {target:,}"

    counts = [c for _, c in EXP014_LADDER]
    octaves = [math.log2(counts[i + 1] / counts[i]) for i in range(len(counts) - 1)]
    assert round(octaves[0], 3) == 1.766
    assert round(octaves[1], 3) == 0.998
    assert octaves[0] > octaves[1], "the ladder's steps are unequal, by design"


def test_analogue_param_count_is_identical_not_merely_matched():
    """The control isolates I1+I3, so its parameterisation must be identical."""
    torch.manual_seed(0)
    snn = SpikingCharLM(vocab_size=BASE_VOCAB, d_model=BASE_D, n_layers=BASE_K)
    torch.manual_seed(0)
    ana = AnalogueCharLM(vocab_size=BASE_VOCAB, d_model=BASE_D, n_layers=BASE_K)

    assert count_params(ana) == count_params(snn) == SPEC_PARAMS

    snn_shapes = {k: tuple(v.shape) for k, v in snn.named_parameters()}
    ana_shapes = {k: tuple(v.shape) for k, v in ana.named_parameters()}
    assert snn_shapes == ana_shapes, "same names, same shapes, or it is not a control"

    # Same construction order under the same seed => same initial weights. This
    # is not required by the spec, but it removes initialisation as a confound
    # from the I1+I3 comparison, so it is worth locking in.
    for name, p in snn.named_parameters():
        torch.testing.assert_close(p, dict(ana.named_parameters())[name])


def test_gru_width_is_derived_and_matched():
    """The external anchor is only an anchor if it is the same size."""
    assert match_gru_width(BASE_VOCAB, BASE_K, SPEC_PARAMS) == GRU_WIDTH
    assert gru_param_count(BASE_VOCAB, GRU_WIDTH, BASE_K) == GRU_PARAMS

    model = GRUCharLM(vocab_size=BASE_VOCAB, d_model=BASE_D, n_layers=BASE_K)
    assert model.hidden_size == GRU_WIDTH
    n = count_params(model)
    assert n == GRU_PARAMS, f"expected {GRU_PARAMS} params, built {n}"

    rel = (n - SPEC_PARAMS) / SPEC_PARAMS
    assert abs(rel) <= 0.02, f"GRU control is {rel:+.2%} off the snn arm"
    assert math.isclose(model.param_error, rel, rel_tol=1e-9)


def test_count_params_counts_everything():
    model = tiny_snn()
    assert count_params(model) == sum(p.numel() for p in model.parameters())
    assert count_params(model) == spiking_param_count(
        TINY["vocab_size"], TINY["d_model"], TINY["n_layers"]
    )


# ===========================================================================
# Forward contract
# ===========================================================================


@pytest.mark.parametrize("reset", RESETS)
def test_forward_shapes_dtypes_and_state(reset):
    model = tiny_snn(reset=reset)
    idx = tiny_idx()
    logits, state, aux = model(idx)

    assert logits.shape == (TINY_B, TINY_L, TINY["vocab_size"])
    assert logits.dtype is torch.float32
    assert len(state) == TINY["n_layers"]
    for v in state:
        assert v.shape == (TINY_B, TINY["d_model"])
        # Membrane state is fp32 always, whatever the GEMM dtype (spec §2, B8).
        assert v.dtype is torch.float32
    assert set(aux) >= {"firing_rate"}


@pytest.mark.parametrize("arm", ["snn", "analogue", "gru"])
def test_state_is_accepted_back_and_carries_exactly(arm):
    """Protocol B relies on this: two windows with carried state == one window.

    Parametrised over all three arms, not just `snn`, because every headline
    number is reported under both protocols and each arm has its own state
    plumbing. `GRUCharLM` in particular converts between the public convention
    (a list of K [B, H] tensors) and nn.GRU's native [K, B, H]; a transposed or
    mis-ordered conversion there would be invisible in protocol A and would
    corrupt only the anchor's protocol-B bpc.
    """
    if arm == "snn":
        model = tiny_snn()
    elif arm == "analogue":
        torch.manual_seed(0)
        model = AnalogueCharLM(**TINY, fused=False)
    else:
        model = tiny_gru()

    idx = tiny_idx(seed=3)
    with torch.no_grad():
        full, _, _ = model(idx)
        first, state, _ = model(idx[:, :3])
        second, _, _ = model(idx[:, 3:], state)
    torch.testing.assert_close(
        torch.cat([first, second], dim=1), full, rtol=1e-5, atol=1e-6
    )
    # A zeroed carried state must be indistinguishable from no state at all, or
    # protocol A and the first window of protocol B disagree by construction.
    with torch.no_grad():
        explicit, _, _ = model(idx, model.init_state(TINY_B))
    torch.testing.assert_close(explicit, full, rtol=0, atol=0)


def test_wrong_state_length_is_rejected():
    model = tiny_snn()
    with pytest.raises(ValueError):
        model(tiny_idx(), [torch.zeros(TINY_B, TINY["d_model"])])


@pytest.mark.parametrize("cls", [SpikingCharLM, AnalogueCharLM])
def test_wrong_state_batch_is_rejected_by_both_arms(cls):
    """A [1, d] carried state must raise, not broadcast, in *both* arms.

    The spiking path gets this from `snn.neuron._check_shapes`. The analogue
    path has to enforce it itself, and if it did not, `v * beta + cur[:, t]`
    would broadcast a single-row state across the batch and quietly produce a
    different model -- a difference between the arms that is not the variable
    under test.
    """
    torch.manual_seed(0)
    model = cls(**TINY, fused=False)
    bad = [torch.zeros(1, TINY["d_model"]) for _ in range(TINY["n_layers"])]
    with pytest.raises(ValueError):
        model(tiny_idx(), bad)


def test_init_state_is_zeroed_fp32():
    model = tiny_snn()
    state = model.init_state(TINY_B)
    assert len(state) == TINY["n_layers"]
    for v in state:
        assert v.dtype is torch.float32
        assert torch.count_nonzero(v) == 0


# ===========================================================================
# I1: spikes are binary
# ===========================================================================


@pytest.mark.parametrize("reset", RESETS)
def test_spikes_are_strictly_zero_or_one(reset):
    """I1. fp32-valued, not bool -- they feed the next layer's GEMM directly."""
    model = tiny_snn(reset=reset, seed=1)
    model.keep_spikes = True
    # Push the membrane well past threshold so both values actually occur.
    with torch.no_grad():
        for layer in model.layers:
            layer.bias.add_(0.9)
    _, _, aux = model(tiny_idx(seed=1))

    spikes = aux["spikes"]
    assert len(spikes) == TINY["n_layers"]
    for k, s in enumerate(spikes):
        assert s.dtype is torch.float32, f"layer {k} spikes must be fp32"
        assert torch.equal(s, s * s), f"layer {k} emitted a value outside {{0, 1}}"
        assert bool(((s == 0.0) | (s == 1.0)).all())
        uniq = torch.unique(s)
        assert set(uniq.tolist()) <= {0.0, 1.0}
        # Non-vacuity. Every assertion above is satisfied by an all-zero tensor,
        # and a silent layer is a real failure mode of this model (see
        # test_second_layer_is_silent_at_initialisation), so a run in which no
        # neuron ever fires must not be allowed to certify "spikes are binary".
        assert set(uniq.tolist()) == {0.0, 1.0}, (
            f"layer {k} emitted only {sorted(uniq.tolist())}; the binarity "
            f"assertion above is vacuous unless both values occur"
        )


def test_analogue_control_is_not_binary():
    """The control is only a control if it actually removes the hard threshold."""
    torch.manual_seed(1)
    model = AnalogueCharLM(**TINY, fused=False)
    model.keep_spikes = True
    _, _, aux = model(tiny_idx(seed=1))
    for a in aux["spikes"]:
        assert bool(((a > 0.0) & (a < 1.0)).all()), "atan_value maps into (0, 1)"
        assert not bool(((a == 0.0) | (a == 1.0)).any())


# ===========================================================================
# R4: firing rate from step 0
# ===========================================================================


def test_firing_rate_is_available_from_step_zero():
    """R4: a per-layer rate, at initialisation, before any training happens."""
    model = tiny_snn(seed=2)
    _, _, aux = model(tiny_idx(seed=2))
    rates = aux["firing_rate"]

    assert isinstance(rates, list)
    assert len(rates) == TINY["n_layers"]
    for r in rates:
        assert isinstance(r, torch.Tensor)
        assert r.ndim == 0
        assert r.dtype is torch.float32
        assert 0.0 <= float(r) <= 1.0
        assert math.isfinite(float(r))
        # Detached: a diagnostic must not be able to leak into the loss.
        assert not r.requires_grad

    floats = summarise_firing_rates(rates)
    assert floats == [float(r) for r in rates]


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_second_layer_is_silent_at_initialisation(seed):
    """A measured property of the frozen baseline, pinned so it cannot be lost.

    This test does not assert that the model is *correct*; it records what the
    baseline of spec §13 actually does at step 0, because the number is
    surprising and it goes in the report:

      * layer 0 fires at roughly 5% -- the embedding is N(0,1), the projection is
        the default nn.Linear init (U(+/-1/sqrt(d)), so sigma ~ 0.0255), the
        current has sigma ~ sqrt(d)*0.0255 ~ 0.58 and beta = 0.5 gives a membrane
        sigma ~ 0.67 against a threshold of 1.0;
      * layer 1 fires at exactly 0% -- its input is that ~5%-dense binary vector,
        so its current has sigma ~ sqrt(0.05*d)*0.0255 ~ 0.13 and the threshold
        sits ~7 sigma out. It never fires, for any seed.

    The consequences are real and belong in the R4 log and the report: at step 0
    the head's input is identically zero, `logits == head.bias`, the loss is
    exactly ln(V), and `head.weight.grad` is exactly zero. The arm is not dead --
    the arctangent surrogate has heavy tails, so layers 0 and 1 do receive
    gradient -- but the reported firing rate from step 0 is [~0.05, 0.00] and
    that is a fact about the frozen configuration, not a bug in the code.

    If a future change to the initialisation makes layer 1 fire, this test fails.
    That is the intent: the report's step-0 numbers must then be re-measured.
    """
    torch.manual_seed(seed)
    model = SpikingCharLM(vocab_size=BASE_VOCAB, d_model=BASE_D, n_layers=BASE_K)
    g = torch.Generator().manual_seed(seed)
    idx = torch.randint(0, BASE_VOCAB, (8, 32), generator=g)
    y = torch.randint(0, BASE_VOCAB, (8, 32), generator=g)

    logits, _, aux = model(idx)
    rates = [float(r) for r in aux["firing_rate"]]
    assert 0.01 < rates[0] < 0.20, f"layer 0 rate moved: {rates}"
    assert rates[1] == 0.0, f"layer 1 is no longer silent at init: {rates}"

    # logits == head.bias broadcast, so the step-0 loss is exactly uniform.
    torch.testing.assert_close(
        logits, model.head.bias.expand_as(logits), rtol=0, atol=0
    )
    loss = torch.nn.functional.cross_entropy(
        logits.reshape(-1, BASE_VOCAB), y.reshape(-1)
    )
    assert float(loss.detach()) == pytest.approx(math.log(BASE_VOCAB), abs=5e-3)

    loss.backward()
    assert torch.count_nonzero(model.head.weight.grad) == 0
    # ...but the surrogate tails still deliver gradient to the spiking layers,
    # which is what distinguishes "silent" from "dead".
    for k in range(BASE_K):
        assert torch.count_nonzero(model.layers[k].weight.grad) > 0


def test_gru_reports_no_firing_rate():
    """A non-spiking control must not emit a plausible-looking spike rate."""
    model = tiny_gru()
    _, _, aux = model(tiny_idx())
    assert aux["firing_rate"] == []
    assert len(aux["hidden_abs_mean"]) == 1


# ===========================================================================
# I5: no lateral recurrent matrix in the snn arm
# ===========================================================================


def _expected_param_inventory(vocab: int, d: int, k: int) -> dict[str, tuple[int, ...]]:
    inv = {"embed.weight": (vocab, d)}
    for i in range(k):
        inv[f"layers.{i}.weight"] = (d, d)
        inv[f"layers.{i}.bias"] = (d,)
    inv["head.weight"] = (vocab, d)
    inv["head.bias"] = (vocab,)
    return inv


@pytest.mark.parametrize("cls", [SpikingCharLM, AnalogueCharLM])
def test_i5_parameter_inventory_is_exhaustive(cls):
    """Every parameter is named and accounted for; there is nothing else.

    An exact-set assertion, so a newly added lateral matrix cannot hide in the
    gaps. The square [d, d] parameters are exactly the K input projections.
    """
    model = cls(**TINY, fused=False)
    got = {k: tuple(v.shape) for k, v in model.named_parameters()}
    want = _expected_param_inventory(
        TINY["vocab_size"], TINY["d_model"], TINY["n_layers"]
    )
    assert got == want

    d = TINY["d_model"]
    square = [k for k, s in got.items() if s == (d, d)]
    assert sorted(square) == [f"layers.{i}.weight" for i in range(TINY["n_layers"])]


@pytest.mark.parametrize("cls", [SpikingCharLM, AnalogueCharLM])
def test_i5_no_recurrent_modules(cls):
    model = cls(**TINY, fused=False)
    for name, mod in model.named_modules():
        assert not isinstance(mod, nn.RNNBase), f"{name} is a recurrent module"
    kinds = {type(m) for _, m in model.named_modules()}
    assert kinds <= {cls, nn.ModuleList, nn.Embedding, nn.Linear}


@pytest.mark.parametrize("reset", RESETS)
def test_i5_temporal_recurrence_is_channel_diagonal(reset):
    """The functional I5 assertion (see this module's docstring).

    d(layer k's emitted sequence)/d(layer k's own bias) must be diagonal in the
    channel axis. Any lateral recurrent matrix -- the mechanism by which a
    layer's own previous output re-enters itself across channels -- makes an
    off-diagonal entry non-zero.
    """
    model = tiny_snn(reset=reset, seed=4)
    model.keep_spikes = True
    with torch.no_grad():
        for layer in model.layers:
            layer.bias.add_(0.4)  # keep the neurons near threshold, not silent
    _, _, aux = model(tiny_idx(seed=4))

    d = TINY["d_model"]
    for k, emitted in enumerate(aux["spikes"]):
        bias = model.layers[k].bias
        jac = torch.zeros(d, d)
        for i in range(d):
            (g,) = torch.autograd.grad(
                emitted[:, :, i].sum(), bias, retain_graph=True, allow_unused=False
            )
            jac[i] = g
        off = jac - torch.diag(torch.diagonal(jac))
        assert torch.count_nonzero(off) == 0, (
            f"layer {k}: bias of channel j changed channel i != j -- that is "
            f"lateral mixing, which I5 forbids"
        )
        # Guard against a vacuous pass: the Jacobian must not be all zeros.
        assert torch.count_nonzero(torch.diagonal(jac)) > 0


def test_gru_control_violates_i5_and_says_so():
    """The anchor's I5 violation is nameable, so its label in tables is truthful."""
    model = tiny_gru()
    assert isinstance(model.gru, nn.RNNBase)
    h = model.hidden_size
    for k in range(model.n_layers):
        w_hh = getattr(model.gru, f"weight_hh_l{k}")
        assert tuple(w_hh.shape) == (3 * h, h), "the lateral matrix I5 forbids"
    assert "I5" in (GRUCharLM.__doc__ or "")


# ===========================================================================
# Spec §0: the GEMM is hoisted out of the time loop
# ===========================================================================


@pytest.mark.parametrize("length", [4, 32])
def test_one_gemm_per_layer_regardless_of_sequence_length(length):
    """Spec §0. The whole architecture rests on this, so it is asserted directly.

    Counting module invocations rather than CUDA kernels lets this run on CPU;
    tests/test_kernel_count.py asserts the CUDA-side consequence.
    """
    model = tiny_snn(seed=5)
    counts = {"linear": 0, "embed": 0}

    def bump(key):
        def hook(_mod, _inp, _out):
            counts[key] += 1

        return hook

    handles = []
    for mod in model.modules():
        if isinstance(mod, nn.Linear):
            handles.append(mod.register_forward_hook(bump("linear")))
        elif isinstance(mod, nn.Embedding):
            handles.append(mod.register_forward_hook(bump("embed")))
    try:
        g = torch.Generator().manual_seed(5)
        idx = torch.randint(0, TINY["vocab_size"], (TINY_B, length), generator=g)
        with torch.no_grad():
            model(idx)
    finally:
        for h in handles:
            h.remove()

    # K layer projections + 1 head, independent of L. A GEMM inside the time
    # loop would make this K*L + 1.
    assert counts["linear"] == TINY["n_layers"] + 1
    assert counts["embed"] == 1


# ===========================================================================
# Precision policy
# ===========================================================================


@pytest.mark.parametrize("dtype_name", ["fp32", "bf16", "fp16"])
def test_gemm_dtype_never_leaks_into_state_or_parameters(dtype_name):
    """cfg.dtype selects the GEMM dtype only (spec §2; §3.11 C2, bottleneck B8)."""
    torch.manual_seed(6)
    model = SpikingCharLM(**TINY, dtype=dtype_name, fused=False)
    model.keep_spikes = True
    idx = tiny_idx(seed=6)
    logits, state, aux = model(idx)

    for name, p in model.named_parameters():
        assert p.dtype is torch.float32, f"{name} must stay an fp32 master weight"
    for v in state:
        assert v.dtype is torch.float32
    for s in aux["spikes"]:
        assert s.dtype is torch.float32
        assert bool(((s == 0.0) | (s == 1.0)).all())
    assert logits.dtype is torch.float32

    # Positive half of the claim. Everything above is a "nothing low-precision
    # escaped" assertion and would pass unchanged if `_project` ignored
    # `gemm_dtype` altogether, so the reduced-precision GEMM is also shown to
    # have actually happened: its result must differ from the fp32 one, and it
    # must still hand fp32 forward.
    with torch.no_grad():
        h = model.embed(idx)
        projected = model._project(model.layers[0], h)
        reference = model.layers[0](h)
    assert projected.dtype is torch.float32
    if dtype_name == "fp32":
        assert torch.equal(projected, reference)
    else:
        assert not torch.equal(projected, reference), (
            f"dtype={dtype_name} produced a bit-identical GEMM to fp32; the "
            f"cast path is not being taken"
        )


def test_bad_dtype_and_reset_are_rejected():
    with pytest.raises(ValueError):
        SpikingCharLM(**TINY, dtype="int8")
    with pytest.raises(ValueError):
        SpikingCharLM(**TINY, reset="none-ish")
    with pytest.raises(NotImplementedError):
        SpikingCharLM(**TINY, t_steps=2)


# ===========================================================================
# build_model
# ===========================================================================


@pytest.mark.parametrize(
    "arch,cls", [("snn", SpikingCharLM), ("analogue", AnalogueCharLM), ("gru", GRUCharLM)]
)
def test_build_model_dispatch(arch, cls):
    Config = pytest.importorskip("snn.config").Config
    cfg = Config(
        arch=arch,
        vocab_size=BASE_VOCAB,
        d_model=BASE_D,
        n_layers=BASE_K,
        device="cpu",
        fused=False,
    )
    model = build_model(cfg)
    assert isinstance(model, cls)
    # AnalogueCharLM must not be an instance of SpikingCharLM, or arch dispatch
    # downstream would silently mislabel the control.
    if arch == "analogue":
        assert not isinstance(model, SpikingCharLM)
    expected = SPEC_PARAMS if arch != "gru" else GRU_PARAMS
    assert count_params(model) == expected
    # build_model places the model on cfg.device. The spec does not say so, so
    # it is pinned here: a trainer that relies on it and a trainer that also
    # calls .to() must both keep working.
    assert all(p.device.type == "cpu" for p in model.parameters())


def test_build_model_rejects_unset_vocab_and_bad_arch():
    """Each guard is exercised through the path that can actually reach it.

    `Config.__post_init__` already rejects an out-of-range `arch`, so passing
    `arch="transformer"` to the constructor tests *config.py* and never calls
    build_model at all -- the previous form of this test would have passed even
    if build_model's dispatch fell through and returned None. Config is
    `frozen=False`, so the field is mutated after construction to put the bad
    value where build_model has to deal with it.
    """
    Config = pytest.importorskip("snn.config").Config

    with pytest.raises(ValueError):
        build_model(Config(vocab_size=0, device="cpu"))

    bad_arch = Config(vocab_size=11, device="cpu", fused=False)
    bad_arch.arch = "transformer"
    with pytest.raises(ValueError, match="arch"):
        build_model(bad_arch)

    bad_surrogate = Config(vocab_size=11, device="cpu", fused=False)
    bad_surrogate.surrogate = "sigmoid"
    with pytest.raises(NotImplementedError):
        build_model(bad_surrogate)

    with pytest.raises(NotImplementedError):
        build_model(Config(vocab_size=11, t_steps=2, device="cpu", fused=False))
    with pytest.raises(NotImplementedError):
        build_model(Config(vocab_size=11, arch="gru", dtype="bf16", device="cpu"))


# ===========================================================================
# metrics
# ===========================================================================


def test_bits_per_char():
    # A model that assigns uniform probability over 256 symbols costs 8 bits.
    n = 1000
    assert bits_per_char(n * math.log(256), n) == pytest.approx(8.0)
    assert bits_per_char(n * math.log(2), n) == pytest.approx(1.0)
    with pytest.raises(ValueError):
        bits_per_char(1.0, 0)


# ===========================================================================
# CUDA smoke tests -- skipped on a CPU-only machine
# ===========================================================================


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_baseline_arm_runs_on_cuda():
    torch.manual_seed(0)
    model = SpikingCharLM(
        vocab_size=BASE_VOCAB, d_model=BASE_D, n_layers=BASE_K
    ).cuda()
    idx = torch.randint(0, BASE_VOCAB, (4, 32), device="cuda")
    logits, state, aux = model(idx)
    assert logits.shape == (4, 32, BASE_VOCAB)
    assert all(v.dtype is torch.float32 for v in state)
    assert all(math.isfinite(float(r)) for r in aux["firing_rate"])
    assert count_params(model) == SPEC_PARAMS


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("reset", RESETS)
def test_fused_and_eager_arms_agree_on_cuda(reset):
    """Model-level restatement of the R10 gate: --fused must not change the model."""
    torch.manual_seed(7)
    fused = SpikingCharLM(**TINY, reset=reset, fused=True).cuda()
    torch.manual_seed(7)
    eager = SpikingCharLM(**TINY, reset=reset, fused=False).cuda()
    fused.keep_spikes = eager.keep_spikes = True

    idx = torch.randint(0, TINY["vocab_size"], (TINY_B, TINY_L), device="cuda")
    with torch.no_grad():
        lf, sf, af = fused(idx)
        le, se, ae = eager(idx)

    for a, b in zip(af["spikes"], ae["spikes"]):
        assert torch.equal(a, b), "spikes must be bit-identical (R10)"
    for a, b in zip(sf, se):
        torch.testing.assert_close(a, b, rtol=0, atol=1e-5)
    torch.testing.assert_close(lf, le, rtol=1e-5, atol=1e-6)
