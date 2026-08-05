"""Gate for the composed arm: two-compartment neuron + learned threshold (EXP_011).

EXP_011 entry condition §8 is that this file passes before a single training step
runs, and its C5/K5 is the decisive leg: **Identity 1 must hold for the
two-compartment neuron**, not merely for the single-compartment LIF that
`EXP_008` §1.1 proved it for.

That distinction is the reason this file exists at all. A reset-shielded second
state variable and a learned mix are exactly the structure that breaks a
substitution argument, so re-using EXP_008's conclusion here would be inheriting
a proof that was never about this neuron. EXP_011 §1.1 redoes the induction; this
file checks it numerically, at a precision where "it is the algebra" and "it is
fp32 rounding" cannot be confused.

THE LEGS
--------
  1. **Identity 1 in float64, at a NON-ZERO per-channel theta.** The composed
     path -- `threshold_gain(cur, theta)` into the scan at the committed scalar
     threshold -- is checked against a per-channel-threshold reference
     transcribed *in this file* from EXP_011 §1.1's equations. Transcribed here
     rather than imported for `test_twocomp_equivalence.py` leg 2's reason: a
     shared misreading by `prescan.py` and `twocomp.py` cannot pass a check whose
     reference was written from the derivation instead of from the code.

     The non-zero theta is K5/M3. At `theta = 0` the check passes vacuously and
     would tell nobody anything -- EXP_007's V4 failure mode, which cleared a
     ceiling the mechanism never approached and was reported as vacuous.

  2. **The nesting leg (K4): at `theta = 0` the composed model IS the adopted
     arm**, bit-for-bit in forward AND backward, asserted at `== 0.0` rather than
     at a tolerance. `exp(0) = 1.0` exactly in fp32 and multiplying by 1.0 is
     exact, so there is no reason to accept a tolerance here, and an equality
     that holds bitwise cannot hide a small systematic error. This ties the
     composed arm to `snn.twocomp.twocomp_scan`, which already carries its own
     mutation campaign, exactly as that scan's leg 5 ties it to `lif_scan`.

  3. **Identity 2 (the fold): `w` and `beta_s_raw` are carried through
     untouched.** That is a claim the derivation makes -- the two parameters
     enter the induction only through a linear combination, which commutes with
     the scaling -- and it is checked rather than assumed. The folded dict must
     also have `twocomp_param_count` exactly and load into `arch="twocomp"`.

  4. **The §7.2 reachability screen at init**: `dL/dtheta` must be non-zero at
     `theta = 0`. EXP_011 §2.1 derives that it should be, and derives what it
     equals; the protocol's rule is that a prediction is not a measurement.

WHAT IS DELIBERATELY NOT ASSERTED
---------------------------------
The fold's *bpc* residual. EXP_011 C4 reports it and does not gate it, because
the fold-in tolerance is Elliot's decision #6 and is still open. Asserting a bar
here would be this project choosing a threshold it has said twice it has no
authority to choose (`EXP_008` §9.5, Phase-4 report §8).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

_SRC = str(Path(__file__).resolve().parents[1] / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from snn.model import (  # noqa: E402
    TwoCompartmentCharLM,
    TwoCompThresholdCharLM,
    twocomp_param_count,
    twocomp_threshold_param_count,
)
from snn.prescan import threshold_gain  # noqa: E402
from snn.twocomp import twocomp_scan_eager  # noqa: E402

ALPHA = 2.0


# ---------------------------------------------------------------------------
# The reference, transcribed from EXP_011 §1.1 rather than from the source
# ---------------------------------------------------------------------------


def _twocomp_per_channel_threshold(cur, v0, w, beta_s, beta_f, thr_c, alpha):
    """The two-compartment neuron with a PER-CHANNEL threshold `thr_c` [1, d].

    Written from EXP_011 §1.1's equation block, in the order it states them:

        vf_t = beta_f·vf_{t-1} + cur_t
        vs_t = beta_s·vs_{t-1} + cur_t
        v_t  = vf_t + w·vs_t
        s_t  = 1[v_t >= thr_c]
        vf_t <- vf_t·(1 - s_t)          hard reset, fast pole only
        vs_t <- vs_t                    reset-shielded

    This is the form Identity 1 says the composed arm implements. `snn.twocomp`
    has no per-channel-threshold entry point -- its `thr` is a scalar -- which is
    precisely why the identity needs a reference the library cannot provide.

    The spike is the hard comparison, not the surrogate: this leg checks the
    FORWARD identity, which is where §1.1's induction lives. The backward is not
    invariant and §1.1 says so explicitly.
    """
    d = cur.shape[2]
    vf = v0[:, :d].clone()
    vs = v0[:, d:].clone()
    spikes = []
    for t in range(cur.shape[1]):
        vf = vf * beta_f + cur[:, t]
        vs = vs * beta_s + cur[:, t]
        s = (vf + w * vs >= thr_c).to(cur.dtype)
        spikes.append(s)
        vf = vf * (1.0 - s)
    return torch.stack(spikes, dim=1), torch.cat([vf, vs], dim=1)


def _inputs(B, L, d, seed, dtype=torch.float64):
    g = torch.Generator().manual_seed(seed)
    cur = torch.randn(B, L, d, generator=g, dtype=dtype)
    v0 = torch.zeros(B, 2 * d, dtype=dtype)
    w = torch.randn(1, d, generator=g, dtype=dtype) * 0.3
    beta_s = torch.sigmoid(torch.randn(1, d, generator=g, dtype=dtype))
    return cur, v0, w, beta_s


# ---------------------------------------------------------------------------
# Leg 1 -- C5/K5: Identity 1 holds for the two-compartment neuron, in fp64
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("beta_f,thr", [(0.5, 1.0), (0.9, 1.0), (0.5, 2.0), (0.95, 0.5)])
def test_identity_1_holds_for_the_two_compartment_neuron(beta_f, thr):
    """`threshold_gain(cur, theta)` at scalar `thr` == the neuron at `thr·exp(theta)`.

    EXP_011 §1.1, in float64 and at a per-channel theta that is nowhere zero.
    Spikes must agree EXACTLY -- the identity is about a spike train, and the
    induction it rests on is exact in real arithmetic, so anything short of exact
    agreement in fp64 means the derivation is wrong rather than that the tolerance
    was too tight (the protocol's "chase the residual, never widen the tolerance").

    The membranes agree only up to the scaling `v' = g·v`, which is the content of
    the induction rather than an artefact of it, so the fast half is compared
    after undoing the gain. The slow half is compared the same way; it is
    reset-shielded, so if the shielding were what broke the substitution this is
    the leg that would catch it.
    """
    B, L, d = 3, 64, 24
    cur, v0, w, beta_s = _inputs(B, L, d, seed=1101)

    gen = torch.Generator().manual_seed(4242)
    theta = torch.randn(1, d, generator=gen, dtype=torch.float64) * 0.6
    assert float(theta.abs().min()) > 1e-3, "K5/M3: theta must be nowhere zero"

    # Composed path: the committed scalar threshold, driven by a scaled current.
    s_comp, v_comp = twocomp_scan_eager(
        threshold_gain(cur, theta), v0, w, beta_s, beta_f, thr, ALPHA
    )
    # Reference: the unscaled current at a per-channel threshold thr·exp(theta).
    s_ref, v_ref = _twocomp_per_channel_threshold(
        cur, v0, w, beta_s, beta_f, thr * torch.exp(theta), ALPHA
    )

    assert bool((s_comp == s_ref).all()), (
        f"Identity 1 fails for the two-compartment neuron at beta_f={beta_f}, "
        f"thr={thr}: {int((s_comp != s_ref).sum())} of {s_comp.numel()} spikes differ"
    )

    g = torch.exp(-theta).squeeze(0)          # the gain applied to the current
    resid_f = float((v_comp[:, :d] / g - v_ref[:, :d]).abs().max())
    resid_s = float((v_comp[:, d:] / g - v_ref[:, d:]).abs().max())
    assert resid_f < 1e-12, f"fast pole breaks the scaling: {resid_f:.3e}"
    assert resid_s < 1e-12, f"shielded slow pole breaks the scaling: {resid_s:.3e}"


def test_identity_1_is_not_vacuous_at_the_theta_it_is_checked_at():
    """The check in leg 1 must be capable of failing. Mutate theta and watch it.

    The protocol's standing rule is that a test is not trusted until it has been
    seen to fail. Here the failure is injected by giving the reference a threshold
    that is wrong by one channel -- if the assertion still passed, leg 1 would be
    comparing something other than what it claims to.
    """
    B, L, d = 3, 64, 24
    cur, v0, w, beta_s = _inputs(B, L, d, seed=1101)
    gen = torch.Generator().manual_seed(4242)
    theta = torch.randn(1, d, generator=gen, dtype=torch.float64) * 0.6

    s_comp, _ = twocomp_scan_eager(
        threshold_gain(cur, theta), v0, w, beta_s, 0.5, 1.0, ALPHA
    )
    bad = (1.0 * torch.exp(theta)).clone()
    bad[0, 0] *= 1.5                                   # one channel, wrong
    s_bad, _ = _twocomp_per_channel_threshold(cur, v0, w, beta_s, 0.5, bad, ALPHA)
    assert not bool((s_comp == s_bad).all()), (
        "a 1.5x error on one channel's threshold changed no spike -- leg 1 cannot fail"
    )


# ---------------------------------------------------------------------------
# Leg 2 -- K4: at theta = 0 the composed model IS the adopted arm, bitwise
# ---------------------------------------------------------------------------


def _models(seed=7, vocab=19, d=16, layers=2):
    torch.manual_seed(seed)
    composed = TwoCompThresholdCharLM(
        vocab, d, layers, beta=0.5, threshold=1.0, reset="hard",
        surrogate_alpha=ALPHA, fused=False, thr_log_init=0.0,
    )
    torch.manual_seed(seed)
    adopted = TwoCompartmentCharLM(
        vocab, d, layers, beta=0.5, threshold=1.0, reset="hard",
        surrogate_alpha=ALPHA, fused=False,
    )
    # Same weights by construction, not by luck: the composed arm's extra
    # ParameterList is created after the shared ones, so the two share an RNG
    # prefix -- but that is an implementation detail and this makes it explicit.
    sd = {k: v for k, v in composed.state_dict().items()
          if not k.startswith("thr_log.")}
    adopted.load_state_dict(sd)
    return composed, adopted


def test_theta_zero_is_the_adopted_arm_bitwise():
    """K4: the composed arm nests the ADOPTED arm at `theta = 0`, forward and back.

    Asserted at `== 0.0`. `exp(0)` is 1.0 exactly in fp32 and `cur * 1.0` is exact,
    so the composed forward performs the same arithmetic in the same order; a
    tolerance here would be accepting an error that has no reason to exist.

    This is the composition's counterpart to
    `test_twocomp_equivalence.py::test_w_zero_is_the_committed_lif_scan`, and it
    matters for the same reason: it ties a new arm to one that is already trusted
    rather than only to a reference written alongside it.
    """
    composed, adopted = _models()
    idx = torch.randint(0, 19, (2, 12), generator=torch.Generator().manual_seed(3))

    lc = composed(idx)[0]
    la = adopted(idx)[0]
    assert float((lc - la).abs().max().detach()) == 0.0, (
        "forward is not bitwise the adopted arm"
    )

    lc.sum().backward()
    la.sum().backward()
    for (name, pc), (_, pa) in zip(composed.named_parameters(),
                                   adopted.named_parameters()):
        if name.startswith("thr_log."):
            continue
        assert pc.grad is not None and pa.grad is not None, name
        assert float((pc.grad - pa.grad).abs().max()) == 0.0, (
            f"backward is not bitwise the adopted arm at {name}"
        )


# ---------------------------------------------------------------------------
# Leg 3 -- Identity 2: the fold, and what it must leave alone
# ---------------------------------------------------------------------------


def test_fold_carries_the_mix_and_the_slow_decay_through_untouched():
    """EXP_011 §1.1: `w` and `beta_s` do not participate in the substitution.

    They enter the induction only through `v = vf + w·vs`, a linear combination
    that commutes with the scaling, so the fold must not touch them. Checked
    rather than assumed -- a fold that quietly rescaled the mix would still
    produce a loadable state dict and a plausible bpc.
    """
    composed, _ = _models()
    with torch.no_grad():
        for k in range(composed.n_layers):
            composed.thr_log[k].copy_(torch.randn(1, composed.d_model) * 0.4)

    before = {k: v.detach().clone() for k, v in composed.state_dict().items()}
    folded = composed.fold_into_twocomp_state_dict()

    for k in range(composed.n_layers):
        assert f"thr_log.{k}" not in folded, "the gain did not fold away"
        assert float((folded[f"w.{k}"] - before[f"w.{k}"]).abs().max()) == 0.0
        assert float(
            (folded[f"beta_s_raw.{k}"] - before[f"beta_s_raw.{k}"]).abs().max()
        ) == 0.0
        # And the rows that SHOULD have moved, did.
        assert float((folded[f"layers.{k}.weight"]
                      - before[f"layers.{k}.weight"]).abs().max()) > 0.0


def test_folded_state_dict_loads_into_the_adopted_arm():
    """The fold's output must have `twocomp_param_count` exactly and load clean.

    `EXP_011` C4 reports the resulting bpc residual and does not gate it (decision
    #6 is open), so this leg checks the structural claim only: the composed arm's
    weights, folded, ARE a two-compartment model.
    """
    composed, adopted = _models()
    with torch.no_grad():
        for k in range(composed.n_layers):
            composed.thr_log[k].copy_(torch.randn(1, composed.d_model) * 0.4)

    folded = composed.fold_into_twocomp_state_dict()
    adopted.load_state_dict(folded)          # strict=True: no missing, no extra
    assert sum(v.numel() for v in folded.values()) == sum(
        p.numel() for p in adopted.parameters()
    )


def test_param_counts_are_the_closed_forms():
    """The composed arm is the adopted arm plus one `[1, d]` per layer."""
    V, d, K = 205, 512, 2
    assert twocomp_threshold_param_count(V, d, K) == twocomp_param_count(V, d, K) + K * d
    assert twocomp_threshold_param_count(V, d, K) == 738_509

    composed, _ = _models(vocab=V, d=d, layers=K)
    assert sum(p.numel() for p in composed.parameters()) == 738_509


# ---------------------------------------------------------------------------
# Leg 4 -- the §7.2 reachability screen at init
# ---------------------------------------------------------------------------


def test_theta_is_gradient_reachable_at_init():
    """EXP_011 §2.1 predicts `theta = 0` is not a saddle. A prediction is not a
    measurement, and EXP_004's most attractive init was caught here as an exact
    saddle that would have trained a frozen parameter in every seed, silently.

    The derived gradient is
        dL/dtheta_c = -sum_t (dL/dcur'_{t,c})·cur'_{t,c}
    which vanishes only if the incoming gradient is orthogonal to the scaled
    current across the whole window -- not a structural zero.
    """
    composed, _ = _models()
    idx = torch.randint(0, 19, (2, 12), generator=torch.Generator().manual_seed(11))
    composed(idx)[0].sum().backward()

    for k in range(composed.n_layers):
        gtheta = composed.thr_log[k].grad
        assert gtheta is not None, f"thr_log.{k} received no gradient at all"
        assert float(gtheta.abs().max()) > 0.0, (
            f"thr_log.{k} is an exact saddle at init -- EXP_011 §2.1 predicted "
            f"otherwise and §6 K-screen aborts the experiment"
        )
