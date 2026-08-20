"""Gates for the interface arms (EXP_020).

`snn.interface` adds no neuron, no scan and no hand-written backward: every
switch here changes what happens **before** the first scan starts or **after**
the last one ends. So this is deliberately NOT an R10 gate --
`tests/test_prescan.py` and `tests/test_dopamine.py` state that standard for
exactly this arm shape and this file follows it. `snn.neuron.lif_scan` and its
mutation campaign are untouched, and N1 is what asserts that they are.

What has to be checked here is different and smaller, and it is five things:

  1. **the nesting is exact.** `arch="interface"` with every switch at its
     default is `SpikingCharLM` -- logits AND every parameter gradient
     **bitwise**, on a shared state dict whose key sets are identical. That is
     what makes the arm its own anchor instead of a fourth thing to control for,
     and `EXP_011` K4 is the precedent for asserting it bitwise rather than to
     1e-12.
  2. **the fold is an identity, and its residual is measured rather than
     assumed.** Decision #6 says a fold-in tolerance cannot be one project
     constant, so F1 asserts the algebra (a lookup of a mapped table IS the map
     of a lookup) and *reports* the fp32 residual on the committed checkpoint
     instead of inheriting a threshold from another arm.
  3. **the code is exactly binary, and the gradient still reaches the shadow.**
     A code that drifted off {0, 1} would violate I1 silently -- the model would
     still train -- and a straight-through path that did not reach `shadow`
     would leave `V*d` parameters frozen at their init, which also trains and
     also looks plausible.
  4. **the readout's column order is fixed.** The head's columns must mean the
     same thing in every checkpoint with the same config hash, and the padding
     must stay INSIDE the window: this tensor feeds the head directly, so a leak
     across the window boundary is a leak of the answer.
  5. **causality**, asserted by perturbation rather than by argument, exactly as
     `tests/test_dopamine.py`'s G2 asserts the head-driven signal's.

TWO NAMES IN THIS FILE ARE LOAD-BEARING. `snn.config` names
`tests/test_interface.py::test_nests_spiking` and `snn.model` names
`::test_param_count_nests`. Both are spelled here exactly as those docstrings
spell them, rather than carrying the `n1_`/`f1_` prefix the rest of the file
uses, so that the two cross-references actually resolve.

ON THE MUTATION LEG, AND WHY IT READS LAYER 0 RATHER THAN A LAG BLOCK.
`02_baseline_report.md` §5.2's pathology is a statement about exactly this
interface: **layer 1 never fires at init.** Measured here at `d = 16` over six
seeds, layer 1's rate is exactly 0.0 in five of them and 0.0020 in the sixth,
against layer 0's 0.035-0.080. A mutation leg that switched on a LAG block of
the *last* layer would therefore have been comparing zeros against zeros and
would have passed by being unable to fail -- the defect
`tests/test_dopamine.py`'s G1 leg exists to rule out, met again here for a
different reason. The leg switches on the layer-0 block instead.
"""

from __future__ import annotations

import inspect
import math
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from snn.config import Config
from snn.interface import (BinaryInputCode, fold_input_table, fold_residual,
                           folded_param_count, match_width_for_params, readout,
                           readout_width)
from snn.model import (InterfaceCharLM, SpikingCharLM, build_model,
                       count_params, interface_param_count,
                       spiking_param_count)

V, D, K, B, L = 37, 16, 2, 4, 8

#: The committed Phase-2 baseline. The fold's residual is a property of REAL
#: weights -- `E @ W0^T` on a trained embedding is not the same rounding problem
#: as on `randn` -- so F1 measures it there when the run is present and falls
#: back to random tensors when it is not, rather than silently measuring only
#: the easy case.
CKPT = (Path(__file__).resolve().parents[1] / "experiments" / "runs"
        / "snn_beta0.5_s0" / "ckpt_final.pt")

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(),
                                   reason="needs a GPU")
requires_ckpt = pytest.mark.skipif(
    not CKPT.exists(), reason=f"no committed checkpoint at {CKPT}")

SWITCHES = [(fold, binary, read, lags)
            for fold in (False, True)
            for binary in (False, True)
            for read in ("last", "all")
            for lags in (1, 3)]


def _idx(seed: int = 3, b: int = B, length: int = L) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, V, (b, length), generator=g)


def _arm(seed: int = 0, **kw) -> InterfaceCharLM:
    torch.manual_seed(seed)
    return InterfaceCharLM(V, D, K, fused=False, **kw)


def _plain(seed: int = 0) -> SpikingCharLM:
    torch.manual_seed(seed)
    return SpikingCharLM(V, D, K, fused=False)


def _pair(**kw):
    """An InterfaceCharLM and a SpikingCharLM carrying bit-identical weights.

    The state dict is *copied across* rather than merely seeded the same way:
    `InterfaceCharLM.__init__` allocates in a different order on some switch
    settings, and a nesting gate that depended on RNG ordering would be
    asserting the seed rather than the arm.
    """
    arm, plain = _arm(**kw), _plain()
    plain.load_state_dict({k: v.clone() for k, v in arm.state_dict().items()})
    return arm, plain


def _grads(model, idx) -> dict:
    logits, _, _ = model(idx, None)
    logits.float().square().mean().backward()
    return {n: p.grad for n, p in model.named_parameters()}


# ---------------------------------------------------------------------------
# N1 -- the nesting, bitwise, forward and backward
# ---------------------------------------------------------------------------

def test_nests_spiking():
    """N1 forward. Named as `snn.config` names it -- see the header.

    At `fold=False, binary_input=False, read_layers='last', read_lags=1` every
    kernel on the path is the baseline's own kernel in the baseline's own order,
    so the two logit tensors are equal BITWISE and not to a tolerance. A
    tolerance here would admit an arm that had quietly become a different model
    within 1e-12, which is precisely what `arch="interface"` may not be if it is
    to serve as its own anchor.
    """
    arm, plain = _pair()
    arm.train(); plain.train()
    idx = _idx()
    a, _, _ = arm(idx, None)
    b, _, _ = plain(idx, None)
    assert torch.equal(a, b)


def test_n1_the_state_dict_key_sets_are_identical():
    """A checkpoint written by either arm must load into the other.

    This is the half of the nesting a forward comparison cannot see: an unused
    parameter left behind on the interface path would still be counted by
    `count_params`, still decayed by AdamW, and would still make every
    parameter-matching gate in the experiment describe a model that is not the
    one that ran.
    """
    arm, plain = _pair()
    assert set(arm.state_dict()) == set(plain.state_dict())
    assert sorted(arm.state_dict()) == [
        "embed.weight", "head.bias", "head.weight",
        "layers.0.bias", "layers.0.weight", "layers.1.bias", "layers.1.weight",
    ]


def test_n1_nesting_is_bitwise_in_the_backward_too():
    """Every parameter's gradient, matched BY NAME.

    By name rather than by zipping two sorted lists: `tests/test_dopamine.py`'s
    header records that a positional zip silently compared 3 of 7 parameters
    once already. The key sets are equal here (the test above), so the by-name
    match is total -- and the count is asserted so that a future switch which
    dropped a parameter could not shrink the comparison silently.
    """
    arm, plain = _pair()
    arm.train(); plain.train()
    idx = _idx()
    ga, gb = _grads(arm, idx), _grads(plain, idx)
    assert set(ga) == set(gb) and len(ga) == 7
    bad = [n for n in sorted(ga) if not torch.equal(ga[n], gb[n])]
    assert bad == [], f"gradients differ at {bad}"


def test_n1_the_default_readout_is_the_last_layer_at_lag_zero():
    """The nesting stated structurally, so a default change cannot be silent."""
    arm = _arm()
    assert arm.read_layer_idx == (K - 1,)
    assert arm.read_lag_idx == (0,)
    assert arm.head.in_features == D
    assert (Config().iface_fold, Config().iface_binary_input) == (False, False)
    assert (Config().iface_read_layers, Config().iface_read_lags) == ("last", 1)


def _all_layers_pair(*, block0_live: bool):
    """A `read_layers='all'` arm whose head is the baseline's, widened.

    Block 1 (layer `K-1`, the block the baseline reads) carries the baseline's
    own head; block 0 (layer 0) carries either zeros or a copy of it. The two
    models cannot share a state dict directly -- the arm's head is `[V, 2d]` --
    so the widening happens here rather than in `_pair`.
    """
    arm, plain = _arm(read_layers="all"), _plain()
    sd = {k: v.clone() for k, v in plain.state_dict().items()}
    wide = torch.zeros(V, 2 * D)
    wide[:, D:] = sd["head.weight"]
    if block0_live:
        wide[:, :D] = sd["head.weight"]
    sd["head.weight"] = wide
    arm.load_state_dict(sd)
    arm.eval(); plain.eval()
    return arm, plain


def test_n1_a_zeroed_extra_read_block_reproduces_the_baseline():
    """`read_layers='all'` with layer 0's head columns zeroed IS the baseline.

    Asserted to a tolerance rather than bitwise, and the reason is stated rather
    than hidden: the head's GEMM now accumulates `2d` terms instead of `d`, so
    the zeros are added in a different order even though they are zeros. It came
    out bitwise equal on this stack; the contract is the tolerance.
    """
    arm, plain = _all_layers_pair(block0_live=False)
    idx = _idx()
    a, _, _ = arm(idx, None)
    b, _, _ = plain(idx, None)
    assert torch.allclose(a, b, rtol=0.0, atol=1e-6), \
        f"max|d| = {float((a - b).detach().abs().max()):.3e}"


def test_n1_mutation_leg_a_live_extra_read_block_does_change_the_forward():
    """The gate above must be capable of failing. Switch layer 0's block on.

    Layer 0 fires at 0.035-0.080 at init (measured; see the header) so this leg
    really does move the logits -- by 0.54 here. Placed on the LAST layer it
    would have moved them by exactly nothing, because that layer does not fire
    at init at all.
    """
    arm, plain = _all_layers_pair(block0_live=True)
    idx = _idx()
    a, _, _ = arm(idx, None)
    b, _, _ = plain(idx, None)
    assert not torch.allclose(a, b, rtol=0.0, atol=1e-6)


# ---------------------------------------------------------------------------
# G2 -- parameter accounting
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("shape", [(37, 16, 2), (97, 32, 3), (11, 8, 1)])
@pytest.mark.parametrize("switch", SWITCHES)
def test_param_count_nests(shape, switch):
    """The closed form equals the REALISED count, for every switch combination.

    Named as `snn.model.interface_param_count` names it -- see the header. The
    closed form is what the driver width-matches on, so a form that disagreed
    with the model it describes would produce an experiment whose "matched" arms
    are not matched. Realised rather than derived: `count_params` walks the
    module tree, so a parameter that `__init__` forgot to remove is counted here
    even though no formula mentions it.

    `binary_input` is deliberately absent from the closed form's signature and
    is asserted here to change nothing: the shadow is `[V, d]`, exactly the
    shape of the `embed`/`table` it replaces, which is the whole point -- the
    arm costs zero parameters and changes only the code's alphabet.
    """
    vocab, d, layers = shape
    fold, binary, read, lags = switch
    cfg = Config(arch="interface", vocab_size=vocab, d_model=d, n_layers=layers,
                 device="cpu", fused=False, deterministic=False,
                 iface_fold=fold, iface_binary_input=binary,
                 iface_read_layers=read, iface_read_lags=lags)
    model = build_model(cfg)
    want = interface_param_count(vocab, d, layers, fold=fold,
                                 read_layers=read, read_lags=lags)
    assert count_params(model) == want


@pytest.mark.parametrize("shape", [(37, 16, 2), (97, 32, 3), (11, 8, 1),
                                   (205, 512, 2)])
def test_g2_the_closed_form_is_spiking_param_count_at_the_defaults(shape):
    """Exactly, for every (V, d, K) -- not just at the committed shape."""
    vocab, d, layers = shape
    assert interface_param_count(vocab, d, layers, fold=False,
                                 read_layers="last", read_lags=1) == \
        spiking_param_count(vocab, d, layers)


def test_g2_the_fold_frees_the_documented_share_of_the_committed_model():
    """35.7 % of 735,437. `snn.interface`'s docstring prices the whole arm on
    that figure, and it is arithmetic rather than measurement, so it is asserted
    rather than reported.

    `layers.0.weight` leaves (`d*d`) and `layers.0.bias` stays, renamed to
    `table_bias` -- so the net saving is the GEMM alone, 262,144 parameters.
    """
    committed = spiking_param_count(205, 512, 2)
    assert committed == 735_437
    freed = committed - folded_param_count(205, 512, 2, folded=True)
    assert freed == 512 * 512
    assert 0.356 < freed / committed < 0.357


def test_g2_read_blocks_multiply_only_the_head():
    """A block is `d*V` more head, and nothing else moves."""
    one = folded_param_count(V, D, K, folded=False, read_blocks=1)
    two = folded_param_count(V, D, K, folded=False, read_blocks=2)
    assert two - one == D * V
    assert readout_width(D, 2, 3) == 6 * D


# ---------------------------------------------------------------------------
# F1 -- the fold, which is an identity and not an approximation
# ---------------------------------------------------------------------------

def _fold_case_random():
    g = torch.Generator().manual_seed(7)
    embed = torch.randn(V, D, generator=g)
    w0 = torch.randn(D, D, generator=g) / math.sqrt(3.0 * D)
    b0 = torch.randn(D, generator=g)
    return embed, w0, b0


def _assert_fold_identity(embed, w0, b0, idx, *, atol: float) -> dict:
    table, bias = fold_input_table(embed, w0, b0)
    two_step = F.linear(F.embedding(idx, embed), w0, b0)
    folded = F.embedding(idx, table) + bias
    assert table.shape == embed.shape
    assert torch.equal(bias, b0) and bias is not b0
    assert torch.allclose(two_step, folded, rtol=0.0, atol=atol), \
        f"max|d| = {float((two_step - folded).abs().max()):.3e}"
    return fold_residual(embed, w0, b0, idx)


def test_f1_the_fold_is_an_identity_on_random_tensors():
    """`E[x] @ W0^T + b0 == T[x] + b0`, because `E[x]` is a lookup.

    The residual is REPORTED and not gated on any inherited constant: decision
    #6 established that a fold-in tolerance cannot be one project number -- the
    same identity through the same fold costs 2.0e-03 bpc on the baseline neuron
    and 1.4e-05 on the two-compartment one. What is asserted is fp32 rounding of
    a `d`-term dot product, which is what the identity is allowed to cost.
    """
    embed, w0, b0 = _fold_case_random()
    res = _assert_fold_identity(embed, w0, b0, _idx(), atol=1e-6)
    assert res["max_rel"] < 1e-5
    assert res["n_elements"] == B * L * D
    assert 0.0 <= res["frac_bitwise_equal"] <= 1.0


def test_f1_the_fold_with_no_bias_is_the_same_identity():
    """`w0_bias=None` must give a zero bias of the right dtype and device, not
    a `None` the caller then has to special-case."""
    embed, w0, _ = _fold_case_random()
    table, bias = fold_input_table(embed, w0)
    assert torch.equal(bias, torch.zeros(D))
    idx = _idx()
    assert torch.allclose(F.linear(F.embedding(idx, embed), w0),
                          F.embedding(idx, table) + bias, rtol=0.0, atol=1e-6)


@requires_ckpt
def test_f1_the_fold_residual_on_the_committed_checkpoint():
    """The same identity on REAL trained weights, at the committed shape.

    A trained embedding is not `randn`: its rows have structure and its scale is
    not 1, so this is the case the arm actually runs on. Measured here at
    max|d| ~ 3.8e-06 and max relative ~ 9.1e-08 -- one ulp of a 512-term dot
    product, and the same order as the 1.9e-06 / 5.2e-08 `snn.interface`'s
    docstring reports for this run.
    """
    sd = torch.load(CKPT, map_location="cpu", weights_only=False)["model"]
    embed, w0, b0 = (sd["embed.weight"], sd["layers.0.weight"],
                     sd["layers.0.bias"])
    assert embed.shape == (205, 512)
    idx = torch.randint(0, 205, (2, 256),
                        generator=torch.Generator().manual_seed(0))
    res = _assert_fold_identity(embed, w0, b0, idx, atol=1e-4)
    assert res["max_rel"] < 1e-5, res
    assert res["mean_abs"] < 1e-6, res
    # Reported, not gated: the fraction that is bitwise equal is a property of
    # the accumulation order and of nothing this project controls.
    print(f"\nfold residual on {CKPT.parent.name}: {res}")


def test_f1_a_folded_model_reproduces_the_two_step_one_end_to_end():
    """The identity through the whole stack, not just through the first GEMM.

    Layer 0's spike train is a *threshold* of the folded current, so an identity
    that held to 1e-6 in the current but flipped one spike would be visible here
    and invisible above. This is the tested form of the claim that `fold` adds
    and removes no function -- for which `EXP_008`'s Identity 2 is the
    precedent, one layer later, and which that experiment then found the
    optimiser does not treat as free.
    """
    plain = _plain(seed=1)
    arm = _arm(seed=2, fold=True)
    table, bias = fold_input_table(plain.embed.weight.detach(),
                                   plain.layers[0].weight.detach(),
                                   plain.layers[0].bias.detach())
    sd = arm.state_dict()
    sd["table"] = table
    sd["table_bias"] = bias
    sd["layers.0.weight"] = plain.layers[1].weight.detach().clone()
    sd["layers.0.bias"] = plain.layers[1].bias.detach().clone()
    sd["head.weight"] = plain.head.weight.detach().clone()
    sd["head.bias"] = plain.head.bias.detach().clone()
    arm.load_state_dict(sd)
    arm.eval(); plain.eval()
    idx = _idx()
    a, _, _ = arm(idx, None)
    b, _, _ = plain(idx, None)
    assert torch.allclose(a, b, rtol=0.0, atol=1e-5), \
        f"max|d| = {float((a - b).abs().max()):.3e}"


def test_f1_the_folded_arm_really_has_lost_layer_zeros_gemm():
    """Otherwise the identity above would hold between two of the same thing,
    and the 35.7 % would not have been freed at all."""
    arm = _arm(fold=True)
    assert len(arm.layers) == K - 1
    assert arm.embed is None
    assert sorted(arm.state_dict()) == [
        "head.bias", "head.weight", "layers.0.bias", "layers.0.weight",
        "table", "table_bias",
    ]


@pytest.mark.parametrize("args", [
    (torch.zeros(5), torch.zeros(4, 4)),
    (torch.zeros(5, 4), torch.zeros(4)),
    (torch.zeros(5, 4), torch.zeros(4, 3)),
    (torch.zeros(5, 6), torch.zeros(4, 4)),
])
def test_f1_fold_input_table_rejects_shapes_that_do_not_compose(args):
    with pytest.raises(ValueError):
        fold_input_table(*args)


# ---------------------------------------------------------------------------
# F2 -- the fold, backwards
# ---------------------------------------------------------------------------

def test_f2_as_spiking_state_dict_round_trips_through_the_committed_path():
    """A folded checkpoint must be evaluable by `SpikingCharLM` itself.

    `EXP_017`'s precedent: the detached arm's checkpoint evaluates as
    `arch='twocomp'`. The inverse picks `E = T, W0 = I`, which is the one choice
    that needs no measurement -- and `I @ T[x]` is a real GEMM, so the contract
    is fp32 tolerance and not bitwise even though the algebra is exact.
    """
    arm = _arm(fold=True)
    arm.eval()
    sd = arm.as_spiking_state_dict()
    spiking = _plain()
    assert set(sd) == set(spiking.state_dict())
    spiking.load_state_dict(sd)
    spiking.eval()
    idx = _idx()
    a, _, _ = arm(idx, None)
    b, _, _ = spiking(idx, None)
    assert torch.allclose(a, b, rtol=0.0, atol=1e-5), \
        f"max|d| = {float((a - b).abs().max()):.3e}"
    assert count_params(spiking) == spiking_param_count(V, D, K)


@pytest.mark.parametrize("kw", [
    {"fold": False},                        # already spiking
    {"fold": True, "binary_input": True},   # no real-valued preimage
    {"fold": True, "read_lags": 2},         # no SpikingCharLM counterpart
    {"fold": True, "read_layers": "all"},
])
def test_f2_the_inverse_refuses_the_arms_it_cannot_invert(kw):
    """It raises rather than inventing structure. A silently wrong inverse would
    produce a `SpikingCharLM` that evaluates a model nobody trained."""
    with pytest.raises(NotImplementedError):
        _arm(**kw).as_spiking_state_dict()


# ---------------------------------------------------------------------------
# B1 -- the binary input code
# ---------------------------------------------------------------------------

def test_b1_the_code_is_exactly_binary():
    """Bitwise {0.0, 1.0}, asserted by counting the elements that are neither.

    Not `allclose(c, c.round())`: `atan_spike`'s own docstring records that the
    spec's parenthesisation produced 1 +/- 5.96e-08 on 7.2 % of elements, which
    every tolerance-based check passes and which violates I1. The count is the
    only form of this assertion that catches it.
    """
    torch.manual_seed(0)
    code = BinaryInputCode(V, D).code()
    assert code.shape == (V, D)
    assert int(((code != 0.0) & (code != 1.0)).sum()) == 0


def test_b1_the_straight_through_gradient_reaches_the_shadow():
    """`shadow` is `V*d` parameters -- 35.7 % of the committed model at the
    committed shape. A code whose gradient did not reach it would train the rest
    of the model around a frozen random input alphabet and would look entirely
    healthy."""
    torch.manual_seed(0)
    code = BinaryInputCode(V, D)
    code.code().sum().backward()
    assert code.shadow.grad is not None
    assert float(code.shadow.grad.abs().max()) > 0.0
    assert int((code.shadow.grad == 0.0).sum()) == 0


def test_b1_the_gains_are_the_derived_ones_and_the_code_is_centred():
    """Both gains are derived in `snn.interface`'s docstring before measurement.

    `is_current=False` matches the unit second moment layer 0's GEMM was
    calibrated for: `g = 1/sqrt(q(1-q)) = 2`. `is_current=True` matches the
    current's OWN variance instead, 1/3: `g = 1/sqrt(3q(1-q)) = 1.1547`. Both
    land layer 0's current on sd 0.5774, which is the one number
    `02_baseline_report.md` §5.2 says an input-side change may not move.
    """
    torch.manual_seed(0)
    feeds_gemm = BinaryInputCode(V, D, is_current=False)
    is_current = BinaryInputCode(V, D, is_current=True)
    assert feeds_gemm.init_density == 0.5
    assert float(feeds_gemm.centre) == 0.5
    assert float(feeds_gemm.gain) == 2.0
    assert abs(float(is_current.gain) - math.sqrt(1.0 / 3.0) / 0.5) < 1e-6
    # Centred: the emitted code takes the two values +/- g*q and nothing else,
    # so the fixed per-channel offset an uncentred code carries is exactly zero.
    out = feeds_gemm(_idx()).detach()
    assert sorted(torch.unique(out).tolist()) == [-1.0, 1.0]
    assert abs(float(out.mean())) < 0.1


@pytest.mark.parametrize("thr,q", [
    (0.0, 0.50), (0.4124631294, 0.34), (1.2815515655, 0.10),
    (1.6448536270, 0.05),
])
def test_b1_the_density_ladder_lands_on_its_targets_and_derives_its_own_gain(thr, q):
    """EXP_022's input leg. `thr_in` is the DENSITY axis: `q = 1 - Phi(thr/sd)`,
    and the gain that keeps layer 0's current on its calibrated second moment
    follows from `q` alone -- `g = 1/sqrt(q(1-q))`. Nothing on this ladder is
    tuned; each rung's gain is the one its density forces.

    The realised density is checked against the target too, because
    `init_density` is a closed form and the code table is a finite draw: a
    formula that is right about a distribution the constructor does not actually
    sample would pass every algebraic check here and train a different arm."""
    torch.manual_seed(0)
    code = BinaryInputCode(V, 512, thr_in=thr)
    assert abs(code.init_density - q) < 5e-5
    assert abs(float(code.gain) - 1.0 / math.sqrt(q * (1.0 - q))) < 1e-4
    assert abs(float(code.density()) - q) < 0.02
    assert abs(float(code.centre) - code.init_density) < 1e-6


def test_b1_code_sd_moves_the_backward_and_provably_not_the_forward():
    """`iface_code_sd` is the PLASTICITY axis, and at `thr_in = 0` it is a pure
    backward intervention: scaling a shadow by `c > 0` cannot change
    `sign(shadow)`, so from the same seed the code table, the gain, the centre
    and the emitted current are all BITWISE identical and only the surrogate
    gradient reaching the shadow differs.

    That is what makes it the axis that separates plasticity from density --
    raising `thr_in` to sparsify also pushes every bit further from its own
    threshold, so a density ladder run alone cannot say which it measured.
    EXP_022 does not run this axis; the switch exists so that §6 can price it."""
    torch.manual_seed(7)
    wide = BinaryInputCode(V, D, thr_in=0.0, code_sd=1.0)
    torch.manual_seed(7)
    tight = BinaryInputCode(V, D, thr_in=0.0, code_sd=0.5)
    assert torch.equal(wide.code(), tight.code())
    assert float(wide.gain) == float(tight.gain)
    assert float(wide.centre) == float(tight.centre)
    idx = _idx()
    assert torch.equal(wide(idx), tight(idx))
    assert wide.init_density == tight.init_density == 0.5
    # ... and the shadows themselves are not equal, so the two models are
    # genuinely different objects that happen to agree on every forward value.
    assert not torch.equal(wide.shadow, tight.shadow)
    assert abs(float(tight.shadow.detach().std())
               - 0.5 * float(wide.shadow.detach().std())) < 1e-5


def test_b1_code_sd_and_thr_in_set_density_only_through_their_ratio():
    """`q = 1 - Phi(thr_in / code_sd)`. Asserted because the pre-EXP_022 formula
    read `thr_in` alone, which is right at `code_sd = 1` and silently wrong
    everywhere else -- the gain would then be derived from a density the code
    does not have, and layer 0's current would miss the one moment
    `02_baseline_report.md` §5.2 says an input-side change may not move."""
    a = BinaryInputCode(V, D, thr_in=0.8249262588, code_sd=2.0)
    b = BinaryInputCode(V, D, thr_in=0.4124631294, code_sd=1.0)
    assert abs(a.init_density - b.init_density) < 1e-9
    assert abs(float(a.gain) - float(b.gain)) < 1e-9
    with pytest.raises(ValueError, match="code_sd must be > 0"):
        BinaryInputCode(V, D, code_sd=0.0)


def test_b1_the_gain_and_centre_are_buffers_and_not_parameters():
    """A learned per-channel gain on the current is exactly a learned
    per-channel threshold by `EXP_008` Identity 1 -- an adopted arm (#5).
    Folding one in here would compose two arms while claiming to measure one."""
    code = BinaryInputCode(V, D)
    assert {n for n, _ in code.named_parameters()} == {"shadow"}
    assert {n for n, _ in code.named_buffers()} == {"gain", "centre"}


def test_b1_the_code_costs_no_parameters():
    """The arm changes the alphabet and not the budget -- which is what makes it
    separable from every width result this project has."""
    plain = _plain()
    arm = _arm(binary_input=True)
    assert count_params(arm) == count_params(plain)
    assert arm.embed is None and arm.code is not None


def test_b1_density_is_a_detached_diagnostic():
    code = BinaryInputCode(V, D)
    assert code.density().requires_grad is False
    assert 0.0 <= float(code.density()) <= 1.0


def test_b1_the_code_rejects_a_degenerate_shape():
    with pytest.raises(ValueError, match="must be positive"):
        BinaryInputCode(0, D)
    with pytest.raises(ValueError, match=r"idx must be \[B, L\]"):
        BinaryInputCode(V, D)(torch.zeros(B, dtype=torch.long))


# ---------------------------------------------------------------------------
# R1 -- the readout
# ---------------------------------------------------------------------------

def _blocks(n_layers: int = 3, length: int = 4, width: int = 2):
    """Layer `k` emits the constant `k+1` everywhere -- so every block in the
    concatenation is identifiable by value alone, and its lag by position."""
    return [torch.full((1, length, width), float(k + 1))
            for k in range(n_layers)]


def test_r1_block_order_is_layer_major_lag_minor():
    """The order is (layer, lag) lexicographic and it is FIXED.

    The head's columns must mean the same thing in every checkpoint that shares
    a config hash. Asserted by construction rather than by reading the source:
    each layer emits a distinct constant, so the block boundaries are visible in
    the values, and the lag is visible in how much zero padding leads each one.
    """
    spikes = _blocks(n_layers=2, length=3, width=2)
    got = readout(spikes, (0, 1), (0, 1))
    assert got.shape == (1, 3, 2 * 2 * 2)
    # row 1 is past the padding, so every block shows its own layer's constant
    assert got[0, 1].tolist() == [1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0]
    # row 0 is inside the padding, so only the lag-0 blocks are populated
    assert got[0, 0].tolist() == [1.0, 1.0, 0.0, 0.0, 2.0, 2.0, 0.0, 0.0]


def test_r1_the_padding_is_zero_and_inside_the_window():
    """Position `t` may never see a character from the window before it.

    `snn.prescan.shift_by_one` pads the same way and for the same reason, and it
    matters more here: this tensor feeds the head directly, so a leak across the
    boundary would be a leak of the answer rather than of a feature.
    """
    spikes = _blocks(n_layers=1, length=5, width=2)
    got = readout(spikes, (0,), (0, 1, 3))
    lag0, lag1, lag3 = got[..., 0:2], got[..., 2:4], got[..., 4:6]
    assert torch.equal(lag0, spikes[0])
    assert torch.equal(lag1[:, 0], torch.zeros(1, 2))
    assert torch.equal(lag1[:, 1:], spikes[0][:, :-1])
    assert torch.equal(lag3[:, :3], torch.zeros(1, 3, 2))
    assert torch.equal(lag3[:, 3:], spikes[0][:, :2])


def test_r1_the_readout_stays_binary_wherever_its_inputs_are():
    """I1 is the constraint that shapes the whole function: the head's input is
    a spike pattern, extended, and never a membrane."""
    g = torch.Generator().manual_seed(2)
    spikes = [(torch.rand(2, 6, D, generator=g) < 0.3).float() for _ in range(2)]
    got = readout(spikes, (0, 1), (0, 2))
    assert int(((got != 0.0) & (got != 1.0)).sum()) == 0


def test_r1_a_single_block_is_returned_without_a_copy():
    """The nesting is a code path here too: no `cat`, hence nothing to round."""
    spikes = _blocks(n_layers=2)
    assert readout(spikes, (1,), (0,)) is spikes[1]


def test_r1_negative_layer_indices_address_from_the_end():
    spikes = _blocks(n_layers=3)
    assert torch.equal(readout(spikes, (-1,), (0,)), spikes[2])


@pytest.mark.parametrize("layers,lags,exc", [
    ((0, 0), (0,), ValueError),   # a repeated block is collinear with itself
    ((0,), (0, 0), ValueError),
    ((5,), (0,), IndexError),     # out of range
    ((-4,), (0,), IndexError),
    ((0,), (-1,), ValueError),    # a negative lag would read the future
    ((), (0,), ValueError),
    ((0,), (), ValueError),
])
def test_r1_malformed_requests_raise(layers, lags, exc):
    with pytest.raises(exc):
        readout(_blocks(n_layers=3), layers, lags)


def test_r1_an_empty_spike_list_raises():
    with pytest.raises(ValueError, match="spikes must be non-empty"):
        readout([], (0,), (0,))


# ---------------------------------------------------------------------------
# C1 -- causality, by perturbation
# ---------------------------------------------------------------------------

def test_c1_a_lagged_readout_is_bitwise_invariant_to_the_future():
    """Perturb `x[:, t+1:]`; `logits[:, :t+1]` must not move by a single bit.

    `s[t-r]` depends on `x[<= t-r]` and the target at `t` is `x[t+1]`, so a
    lagged readout reads strictly LESS of the future than the unlagged one does
    -- which is to say none. Asserted by perturbation rather than by that
    argument, exactly as `tests/test_dopamine.py`'s G2 is.
    """
    arm = _arm(read_lags=4)
    arm.eval()
    idx = _idx()
    base, _, _ = arm(idx, None)
    for t in range(L - 1):
        future = idx.clone()
        future[:, t + 1:] = (future[:, t + 1:] + 1) % V
        got, _, _ = arm(future, None)
        assert torch.equal(got[:, :t + 1], base[:, :t + 1]), (
            f"the readout leaked the future at t={t}: max|d|="
            f"{float((got[:, :t + 1] - base[:, :t + 1]).abs().max()):.3e}")


def test_c1_mutation_leg_a_lag_that_padded_at_the_end_would_be_caught():
    """The gate above must be able to fire. Pad behind instead of in front.

    `F.pad(x[:, r:], (0, 0, 0, r))` is the same one-character edit
    `tests/test_dopamine.py`'s G2 leg makes: it turns a delay into an ADVANCE,
    so position `t` reads the spike emitted at `t+r` -- a spike that saw the
    character the head at `t` is being asked to predict.

    **Run on LAYER 0 and at 32x64 rather than on the last layer at 4x8**, for
    the reason the header gives and `tests/test_dopamine.py`'s G1 leg gives: the
    last layer does not fire at init, so a leg placed there would compare zeros
    against zeros and pass by being unable to fail. The real `readout` is
    checked on the same spikes in the same test, so the leg cannot pass by the
    perturbation simply being too small either.
    """
    def leaking_lag(x, r):
        return x if r == 0 else F.pad(x[:, r:], (0, 0, 0, r))

    arm = _arm(read_lags=4)
    arm.eval()
    arm.keep_spikes = True
    lags = arm.read_lag_idx
    idx = _idx(b=32, length=64)

    def blocks(x):
        _, _, aux = arm(x, None)
        s0 = aux["spikes"][0]                       # the layer that fires
        honest = readout([s0], (0,), lags)
        leaking = torch.cat([leaking_lag(s0, r) for r in lags], dim=-1)
        return honest, leaking

    base_honest, base_leaking = blocks(idx)
    caught = False
    for t in range(idx.shape[1] - 1):
        future = idx.clone()
        future[:, t + 1:] = (future[:, t + 1:] + 1) % V
        honest, leaking = blocks(future)
        assert torch.equal(honest[:, :t + 1], base_honest[:, :t + 1])
        if not torch.equal(leaking[:, :t + 1], base_leaking[:, :t + 1]):
            caught = True
            break
    assert caught, "the causality gate cannot detect a lag that reads forward"


def test_c1_forward_takes_idx_alone():
    """The strongest form of the no-leak claim: `y` is not an argument, so no
    index arithmetic inside the arm can reach it."""
    assert list(inspect.signature(InterfaceCharLM.forward).parameters) == \
        ["self", "idx", "state"]
    assert list(inspect.signature(InterfaceCharLM.input_current).parameters) == \
        ["self", "idx"]


# ---------------------------------------------------------------------------
# W1 -- width matching
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("target", [735_437, 100_000, 2_000_000])
@pytest.mark.parametrize("folded", [False, True])
def test_w1_match_width_is_the_largest_width_that_fits(target, folded):
    """The bracket, asserted on both sides: `d` fits and `d+1` does not.

    Width matching IS parameter matching only up to the integer grid, which is
    why the driver records the realised counts and why this states a bracket
    rather than an equality.
    """
    d = match_width_for_params(target, 205, 2, folded=folded)
    assert d >= 1
    assert folded_param_count(205, d, 2, folded=folded) <= target
    assert folded_param_count(205, d + 1, 2, folded=folded) > target


def test_w1_the_fold_buys_the_documented_width():
    """512 -> 675 at the committed budget, and 676 is the first width over it.

    `snn.interface`'s prose rounds this to "512 to 676"; the function returns
    the largest width that does NOT exceed the target, which is 675 (733,930
    parameters, against 735,693 at 676). Recorded here so that the 1.32x claim
    is anchored to the width a driver would actually build.
    """
    d = match_width_for_params(735_437, 205, 2, folded=True)
    assert d == 675
    assert folded_param_count(205, 675, 2, folded=True) == 733_930
    assert folded_param_count(205, 676, 2, folded=True) == 735_693
    assert 1.31 < d / 512 < 1.33


def test_w1_an_unreachable_target_raises_rather_than_looping():
    with pytest.raises(ValueError, match="unreachable"):
        match_width_for_params(10 ** 20, 205, 2, folded=True)


def test_w1_more_read_blocks_buy_less_width():
    """A wider head is paid for out of the same budget, so the matched width
    falls -- which is the trade the readout arms are actually making."""
    one = match_width_for_params(735_437, 205, 2, folded=True, read_blocks=1)
    four = match_width_for_params(735_437, 205, 2, folded=True, read_blocks=4)
    assert four < one


# ---------------------------------------------------------------------------
# G3 -- dispatch, configuration and state
# ---------------------------------------------------------------------------

def test_g3_build_model_dispatches_and_records_its_switches():
    cfg = Config(arch="interface", vocab_size=V, d_model=D, n_layers=K,
                 device="cpu", fused=False, deterministic=False,
                 iface_fold=True, iface_binary_input=True,
                 iface_read_layers="all", iface_read_lags=2, iface_thr_in=0.25)
    model = build_model(cfg)
    assert isinstance(model, InterfaceCharLM)
    assert (model.fold, model.binary_input) == (True, True)
    assert (model.read_layers_mode, model.read_lags) == ("all", 2)
    assert model.thr_in == 0.25
    assert model.code.init_density != 0.5    # thr_in moved the density with it


@pytest.mark.parametrize("kw,match", [
    ({"read_layers": "middle"}, "read_layers must be"),
    ({"read_lags": 0}, "read_lags must be >= 1"),
    ({"read_lags": D + 1}, "implausible"),
])
def test_g3_the_module_rejects_impossible_switches(kw, match):
    with pytest.raises(ValueError, match=match):
        _arm(**kw)


def test_g3_config_rejects_a_lag_ladder_longer_than_the_window():
    """Every position would read only pad -- which trains and looks plausible."""
    with pytest.raises(ValueError, match="exceeds"):
        Config(seq_len=4, iface_read_lags=5)
    with pytest.raises(ValueError, match="iface_read_lags must be >= 1"):
        Config(iface_read_lags=0)
    with pytest.raises(ValueError, match="iface_read_layers"):
        Config(iface_read_layers="middle")


@pytest.mark.parametrize("kw", [
    {}, {"fold": True}, {"binary_input": True},
    {"fold": True, "binary_input": True},
])
def test_g3_state_round_trips_under_protocol_b(kw):
    """Carried state must come back in the same shape, as for every arm --
    including on the folded and binary paths, where `init_state` cannot read
    `embed` because there is no longer an `embed` to read."""
    arm = _arm(**kw)
    arm.eval()
    idx = _idx()
    _, state, _ = arm(idx, None)
    assert len(state) == K and all(s.shape == (B, D) for s in state)
    logits, state2, _ = arm(idx, [s.detach() for s in state])
    assert logits.shape == (B, L, V) and len(state2) == K


def test_g3_the_code_density_is_reported_only_when_there_is_a_code():
    """And only when `keep_spikes` asks for it.

    `code_density` re-binarises the whole `[V, d]` table to produce one scalar.
    On the hot path that is `V*d` threshold work per FORWARD -- which at decode
    is per CHARACTER, inside a captured graph, for a number nothing on that path
    reads. It therefore takes `keep_spikes`' contract, the same one
    `aux["spikes"]` has: off by default, on when a caller wants diagnostics.

    A code that collapsed to all-ones is a constant current and would train and
    score plausibly, so the diagnostic has to remain reachable -- which is what
    the second half of this test asserts.
    """
    plain = _arm()
    coded = _arm(binary_input=True)
    assert "code_density" not in plain(_idx(), None)[2]
    # off by default, on both arms
    assert "code_density" not in coded(_idx(), None)[2]

    coded.keep_spikes = True
    coded_aux = coded(_idx(), None)[2]
    assert 0.0 <= float(coded_aux["code_density"]) <= 1.0
    assert coded_aux["code_density"].requires_grad is False

    plain.keep_spikes = True
    assert "code_density" not in plain(_idx(), None)[2]


# ---------------------------------------------------------------------------
# CUDA -- the fused kernel is the one the arms actually run
# ---------------------------------------------------------------------------

@pytest.mark.cuda
@requires_cuda
def test_cuda_the_nesting_is_bitwise_on_the_fused_kernel_too():
    """N1 on CPU exercises the eager reference. The runs use the jiterator."""
    torch.manual_seed(0)
    arm = InterfaceCharLM(V, D, K, fused=True).cuda()
    torch.manual_seed(0)
    plain = SpikingCharLM(V, D, K, fused=True).cuda()
    plain.load_state_dict({k: v.clone() for k, v in arm.state_dict().items()})
    idx = _idx().cuda()
    a, _, _ = arm(idx, None)
    b, _, _ = plain(idx, None)
    assert torch.equal(a, b)


@pytest.mark.cuda
@requires_cuda
def test_cuda_fused_and_eager_agree_on_the_folded_binary_arm():
    """R10 is inherited rather than new -- the arm's own path is asserted anyway,
    because the input code is the one place a new nonlinearity enters."""
    torch.manual_seed(0)
    fused = InterfaceCharLM(V, D, K, fused=True, fold=True,
                            binary_input=True, read_lags=2).cuda()
    torch.manual_seed(0)
    eager = InterfaceCharLM(V, D, K, fused=False, fold=True,
                            binary_input=True, read_lags=2).cuda()
    eager.load_state_dict(fused.state_dict())
    idx = _idx().cuda()
    a, _, _ = fused(idx, None)
    b, _, _ = eager(idx, None)
    assert torch.equal(a, b)


def test_a2_a_multi_tap_readout_refuses_a_window_shorter_than_its_deepest_lag():
    """The trap this raise exists to close, and why raising is the right fix.

    `_lag` reads `s[t - r]` from INSIDE the call's window and no lag state is
    carried across calls, so at `L <= max(lag)` every lagged block is the zero
    padding and nothing else. A model trained at L = 256 -- where those blocks
    carry real spikes -- would then decode one character at a time with half its
    readout permanently zero.

    That failure is silent in every way this project usually catches things: it
    does not raise, it does not fail CUDA-graph capture, and the eager and
    captured paths agree with each other while both are wrong, so
    `test_graph_and_eager_draw_the_same_text` still passes.

    Nothing measured is affected -- `EXP_020` leg 6 was NOT RUN on leg A's
    evidence and `iface_read_lags` is 1 in every committed run -- but the switch
    exists, so the failure is made loud.
    """
    arm = _arm(read_lags=3)          # lags (0, 1, 2), so the deepest is 2

    # L > max_lag is fine: at L = 3 position 2's lag-2 block is s[0], a real
    # spike. The boundary is exactly here and is asserted rather than assumed.
    arm(_idx(b=2, length=3), None)
    arm(_idx(b=2, length=4), None)

    # L <= max_lag makes the DEEPEST block all-pad at every position
    for bad_len in (1, 2):
        with pytest.raises(ValueError, match="needs L >"):
            arm(_idx(b=2, length=bad_len), None)

    # and the single-tap default is unaffected at every length, including L = 1,
    # which is what an ordinary decode loop passes
    plain = _arm()
    for length in (1, 2, 8):
        plain(_idx(b=2, length=length), None)
