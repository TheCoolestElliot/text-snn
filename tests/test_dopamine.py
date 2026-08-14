"""Gates for the dopamine arm (EXP_018 §7).

**This is deliberately NOT an R10 gate, and the reason is the shape of the arm.**
R10 is about a hand-written jiterator backward being silently wrong. This arm has
none: it transforms `cur` with ordinary autograd ops outside the time loop and
hands the result to `snn.neuron.lif_scan` -- the committed Phase-2 kernel, still
guarded by its own gate and its 25/25 mutation campaign, untouched here.
`tests/test_prescan.py` states that standard for exactly this arm shape and this
file follows it.

What this arm has instead, and what most of this file is about, is a **causality
contract**. `DA_t` is built from the model's own prediction at `t-1` scored
against the character at `t`. Shift it by one in the wrong direction and the
model is handed the surprise of its own *target*: it would still train, the bpc
would still look plausible, and it would be a leak. G2 is that gate, and G8's
mutation leg proves G2 can actually fire -- "a screen nothing has ever tripped is
indistinguishable from one that cannot trip" (`tests/test_noise.py`).

One structural note that is itself a caught bug. G1 compares the arm's gradients
against the baseline's **by name**, not by `zip`ping two sorted lists as
`test_noise.py` does. That works there because `NoisyCharLM` adds no parameter.
Here `k.0` and `k.1` sort between `head.weight` and `layers.0.bias`, so a
positional zip silently compares **3 of 7** shared parameters and passes. The
first draft of this file did exactly that.
"""

from __future__ import annotations

import inspect

import pytest
import torch
import torch.nn.functional as F

from snn.config import Config
from snn.dopamine import (apply_dopamine, dopamine, roll_across_batch, rpe)
from snn.model import (DopamineCharLM, SpikingCharLM, build_model, count_params,
                       dopamine_param_count, spiking_param_count)

V, D, K, B, L = 37, 16, 2, 4, 8
TAU = 1.1291

MODES = ("mult", "add")
SOURCES = ("off", "rpe", "rolled")

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(),
                                   reason="needs a GPU")


def _pair(mode: str = "mult", source: str = "rpe", k_init: float = 0.0,
          seed: int = 0):
    """A DopamineCharLM and a SpikingCharLM with bit-identical shared weights."""
    torch.manual_seed(seed)
    arm = DopamineCharLM(V, D, K, fused=False, da_mode=mode, da_source=source,
                         da_scale=TAU, da_gain_init=k_init)
    torch.manual_seed(seed)
    plain = SpikingCharLM(V, D, K, fused=False)
    plain.load_state_dict({k: v.clone() for k, v in arm.state_dict().items()
                           if not k.startswith("k.")})
    return arm, plain


def _idx(seed: int = 3, b: int = B, length: int = L):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, V, (b, length), generator=g)


def _shared_grads(arm, plain) -> list[str]:
    """Names of shared parameters whose gradients differ. Matched BY NAME."""
    a = dict(arm.named_parameters())
    b = dict(plain.named_parameters())
    return [n for n in sorted(set(a) & set(b))
            if not torch.equal(a[n].grad, b[n].grad)]


# ---------------------------------------------------------------------------
# G1 -- nesting
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("source", SOURCES)
def test_g1_k_zero_is_the_baseline_bitwise_forward(mode, source):
    """At k = 0 the arm IS SpikingCharLM. Not a tolerance -- torch.equal.

    `EXP_011` K4 is the precedent for asserting a nesting bitwise rather than to
    1e-12. It holds for the ADDITIVE mode too, which is not obvious: `cur + 0.0`
    is not `cur` for `cur = -0.0`. It survives because the only way the two
    differ is the sign of a zero current, and `v*beta + (-0.0)` and
    `v*beta + (+0.0)` differ only when `v*beta` is itself exactly `-0.0` -- at
    which point the membrane is zero either way, `-0.0 >= 1.0` is False either
    way, and the spike train and every downstream product are identical.
    """
    arm, plain = _pair(mode, source)
    arm.train(); plain.train()
    idx = _idx()
    a, _, _ = arm(idx, None)
    b, _, _ = plain(idx, None)
    assert torch.equal(a, b)


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("source", SOURCES)
def test_g1_k_zero_is_the_baseline_bitwise_backward(mode, source):
    """Every SHARED parameter's gradient, bitwise. Matched by name -- see header."""
    arm, plain = _pair(mode, source)
    arm.train(); plain.train()
    idx = _idx()
    for m in (arm, plain):
        logits, _, _ = m(idx, None)
        logits.square().mean().backward()
    assert _shared_grads(arm, plain) == []
    # ...and the arm really does have the extra parameters, so the comparison
    # above was over 7 shared names and not over a coincidentally equal set.
    extra = sorted(set(dict(arm.named_parameters())) - set(dict(plain.named_parameters())))
    assert extra == [f"k.{i}" for i in range(K)]


@pytest.mark.parametrize("mode", MODES)
def test_g1_mutation_leg_a_real_gain_does_change_the_forward(mode):
    """The nesting check must be capable of failing. At k != 0 it must.

    **Run at 32x64 rather than at this file's 4x8, and the reason is a real
    property of the arm rather than a testing convenience.** A spiking forward is
    quantised by its own threshold: a perturbation of `cur` that flips no spike
    anywhere produces a *bitwise identical* output, because the only thing a
    layer emits is a binary. Measured at 4x8x16 = 512 sites, an untrained model
    at `k = 1.0` flips nothing at all and the forward is `torch.equal` to the
    baseline's; the first flip appears near `k = 5`. At 32x64 the same `k = 0.35`
    flips spikes comfortably.

    So a mutation leg placed at the small shape would have "passed" by being
    unable to fail, which is the exact defect this leg exists to rule out.
    """
    arm, plain = _pair(mode, "rpe", k_init=0.35)
    arm.train(); plain.train()
    idx = _idx(b=32, length=64)
    a, _, _ = arm(idx, None)
    b, _, _ = plain(idx, None)
    assert not torch.equal(a, b)


def test_g1_a_subthreshold_gain_is_invisible_and_that_is_i3_not_a_bug():
    """Guards the property the leg above had to work around.

    A perturbation too small to cross any threshold changes nothing observable,
    because I1 says the only thing crossing between layers is a binary. This is
    recorded as a gate so that a future edit which makes the arm's effect
    *continuous* -- an analogue leak, say -- fails here loudly instead of
    quietly changing what the arm is.
    """
    arm, plain = _pair("mult", "rpe", k_init=1.0)
    arm.eval(); plain.eval()
    idx = _idx()                                  # the small 4x8 shape
    assert float(arm.dopamine_signal(idx, None).abs().max()) > 0.0
    a, _, _ = arm(idx, None)
    b, _, _ = plain(idx, None)
    assert torch.equal(a, b)


def test_g1_source_off_does_not_run_a_second_pass():
    """`off` nests by CODE PATH, which is what makes it free as well as identical."""
    arm, _ = _pair("mult", "off")
    assert arm.da_active is False
    calls = []
    real = torch.nn.functional.log_softmax

    def counting(*a, **kw):
        calls.append(1)
        return real(*a, **kw)

    torch.nn.functional.log_softmax = counting
    try:
        arm(_idx(), None)
    finally:
        torch.nn.functional.log_softmax = real
    assert calls == []


# ---------------------------------------------------------------------------
# G2 -- causality. The one thing that must not be wrong.
# ---------------------------------------------------------------------------

def test_g2_da_at_t_is_bitwise_invariant_to_the_future():
    """Perturb x[:, t+1:]; DA[:, :t+1] must not move by a single bit.

    This is the leak gate. `phi_t` is allowed to depend on `x_{<=t}` and on
    nothing else; the target at position `t` is `x_{t+1}`.
    """
    arm, _ = _pair("mult", "rpe")
    arm.eval()
    idx = _idx()
    base = arm.dopamine_signal(idx, None)
    for t in range(L - 1):
        future = idx.clone()
        # Change every character strictly after t to something else.
        future[:, t + 1:] = (future[:, t + 1:] + 1) % V
        got = arm.dopamine_signal(future, None)
        assert torch.equal(got[:, :t + 1], base[:, :t + 1]), (
            f"DA leaked the future at t={t}: "
            f"max|d|={float((got[:, :t + 1] - base[:, :t + 1]).abs().max()):.3e}"
        )


def test_g2_mutation_leg_an_off_by_one_shift_is_caught():
    """The gate above must be able to fire. Pad at the END instead of the front.

    That single-character mutation makes entry `t` the surprise of `x_{t+1}` --
    the model's own target, handed to it at the position that must predict it.
    It is the exact bug G2 exists for, and it must be caught at position 0.
    """
    def leaking_rpe(logits, idx):
        logp = F.log_softmax(logits.float(), dim=-1)
        prev = logp[:, :-1]
        expected = -(prev.exp() * prev).sum(-1)
        observed = -prev.gather(-1, idx[:, 1:].unsqueeze(-1)).squeeze(-1)
        return F.pad(expected - observed, (0, 1))   # MUTANT: (1, 0) -> (0, 1)

    # The logits come from a PLAIN model: reaching them through the arm would
    # trip its own "no dopamine signal is live" guard, which is a different gate.
    _, plain = _pair("mult", "rpe")
    plain.eval()
    idx = _idx()

    def signal(fn, x):
        logits, _, _ = plain(x, None)
        return dopamine(fn(logits, x), TAU)

    caught = False
    for t in range(L - 1):
        future = idx.clone()
        future[:, t + 1:] = (future[:, t + 1:] + 1) % V
        if not torch.equal(signal(leaking_rpe, future)[:, :t + 1],
                           signal(leaking_rpe, idx)[:, :t + 1]):
            caught = True
            break
    assert caught, "the causality gate cannot detect an off-by-one shift"


def test_g2_forward_takes_idx_alone():
    """The targets are not reachable from the model at all. By signature.

    The strongest possible form of the no-leak claim: `y` is not an argument, so
    no amount of index arithmetic inside the arm can reach it. Asserted on both
    the arm's `forward` and on the pass-1 helper the causality gate exercises.
    """
    assert list(inspect.signature(DopamineCharLM.forward).parameters) == \
        ["self", "idx", "state"]
    assert list(inspect.signature(DopamineCharLM.dopamine_signal).parameters) == \
        ["self", "idx", "state"]
    assert list(inspect.signature(rpe).parameters) == ["logits", "idx"]


def test_g2_rpe_is_the_committed_cross_entropy_on_the_correct_shift():
    """The gathered surprise IS `F.cross_entropy` on the committed targets.

    `snn.data` builds `x = block[:, :-1]`, `y = block[:, 1:]`, so
    `y[:, u] == x[:, u+1]` and this identity holds for the CORRECT shift only.
    An off-by-one fails it by orders of magnitude, not by an ulp. This is
    `018_calibrate_da.py`'s C1, as a unit test that needs no checkpoint.
    """
    torch.manual_seed(0)
    logits = torch.randn(B, L, V)
    block = _idx(seed=11, length=L + 1)
    x, y = block[:, :-1], block[:, 1:]

    logp = F.log_softmax(logits.float(), dim=-1)
    observed = -logp[:, :-1].gather(-1, x[:, 1:].unsqueeze(-1)).squeeze(-1)
    ce = F.cross_entropy(
        logits[:, :-1].reshape(-1, V), y[:, :-1].reshape(-1), reduction="none"
    ).reshape(observed.shape)
    assert torch.allclose(observed, ce, atol=1e-6)


def test_g2_da_is_zero_at_position_zero():
    """Position 0 has no preceding prediction, so it is unmodulated exactly."""
    arm, _ = _pair("mult", "rpe")
    arm.eval()
    da = arm.dopamine_signal(_idx(), None)
    assert torch.equal(da[:, 0], torch.zeros(B))


# ---------------------------------------------------------------------------
# G3 / G4 -- the new parameter
# ---------------------------------------------------------------------------

def test_g3_the_new_parameter_is_in_the_graph_and_reachable():
    """Exactly K gradients named k.*, none None, none exactly zero at k = 0.

    §7.2's rule is "flag anything exactly zero or ~100x below". Exactly zero is
    fatal and unrecoverable -- AdamW's update is identically 0 forever and
    nothing in the logs says so. This asserts the fatal half; the ratio half is
    `scripts/audit/09_gradient_reachability.py`'s and is reported, not gated.
    """
    arm, _ = _pair("mult", "rpe")
    arm.train()
    logits, _, _ = arm(_idx(), None)
    logits.square().mean().backward()
    grads = {n: p.grad for n, p in arm.named_parameters() if n.startswith("k.")}
    assert len(grads) == K
    for n, g in grads.items():
        assert g is not None, f"{n} is not in the graph"
        assert float(g.abs().max()) > 0.0, f"{n} is an exact saddle at k_init = 0"


def test_g4_k_is_per_channel_and_not_broadcast_from_one_element():
    """Recovered functionally, not by shape. A [1,d] and a [d] broadcast alike."""
    cur = torch.randn(B, L, D)
    da = torch.randn(B, L)
    k = torch.zeros(1, D)
    base = apply_dopamine(cur, k, da, "add")
    k2 = k.clone()
    k2[0, 0] = 1.0
    got = apply_dopamine(cur, k2, da, "add")
    assert not torch.equal(got[..., 0], base[..., 0])
    assert torch.equal(got[..., 1:], base[..., 1:])


# ---------------------------------------------------------------------------
# G5 -- shape contracts
# ---------------------------------------------------------------------------

def test_g5_apply_rejects_a_broadcastable_but_wrong_k():
    with pytest.raises(ValueError, match=r"k must be \[1, d\]"):
        apply_dopamine(torch.randn(B, L, D), torch.zeros(D), torch.randn(B, L), "mult")


def test_g5_apply_rejects_a_broadcastable_but_wrong_da():
    """A [1, L] signal broadcasts fine and shares one row's dopamine across the
    batch -- a different experiment that looks exactly like this one."""
    with pytest.raises(ValueError, match="matching cur"):
        apply_dopamine(torch.randn(B, L, D), torch.zeros(1, D),
                       torch.randn(1, L), "mult")


def test_g5_unknown_mode_is_rejected():
    with pytest.raises(ValueError, match="mode must be one of"):
        apply_dopamine(torch.randn(B, L, D), torch.zeros(1, D),
                       torch.randn(B, L), "divide")


def test_g5_scan_outside_forward_raises_rather_than_training_the_baseline():
    """A silently unmodulated dopamine arm is indistinguishable from a null."""
    arm, _ = _pair("mult", "rpe")
    with pytest.raises(RuntimeError, match="no dopamine signal is live"):
        arm._scan(torch.randn(B, L, D), torch.zeros(B, D), 0)


# ---------------------------------------------------------------------------
# G6 -- the cost structure IS the argument
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("length", (4, 16))
def test_g6_the_stack_runs_twice_and_the_transform_once_per_layer_per_pass(length):
    """2K scans and K transforms per forward, INDEPENDENT of L.

    If either scaled with L the whole cost model would be gone: a transform
    inside the time loop is `K*L` kernels against the scan's own `K*L`, and this
    architecture exists to keep that loop at one kernel per timestep per layer.
    """
    import snn.model as M

    scans, applies = [], []
    real_scan, real_apply = M.lif_scan, M.apply_dopamine
    M.lif_scan = lambda cur, *a, **kw: (scans.append(cur.shape[1]), real_scan(cur, *a, **kw))[1]
    M.apply_dopamine = lambda cur, *a, **kw: (applies.append(cur.shape[1]), real_apply(cur, *a, **kw))[1]
    try:
        arm, _ = _pair("mult", "rpe")
        arm(_idx(length=length), None)
    finally:
        M.lif_scan, M.apply_dopamine = real_scan, real_apply

    assert scans == [length] * (2 * K), f"expected {2 * K} scans, got {len(scans)}"
    assert applies == [length] * K, f"expected {K} transforms, got {len(applies)}"


def test_g6_source_off_runs_the_stack_once():
    import snn.model as M

    scans = []
    real = M.lif_scan
    M.lif_scan = lambda cur, *a, **kw: (scans.append(1), real(cur, *a, **kw))[1]
    try:
        arm, _ = _pair("mult", "off")
        arm(_idx(), None)
    finally:
        M.lif_scan = real
    assert len(scans) == K


# ---------------------------------------------------------------------------
# G8 -- the control, and mutation legs on it
# ---------------------------------------------------------------------------

def test_g8_the_roll_preserves_the_marginal_exactly():
    """Not "in distribution" -- the same multiset of values."""
    da = torch.randn(B, L)
    rolled = roll_across_batch(da)
    assert torch.equal(torch.sort(da.flatten()).values,
                       torch.sort(rolled.flatten()).values)


def test_g8_the_roll_preserves_each_rows_time_series_exactly():
    """Row b gets row b-1's series INTACT -- that is what a time shuffle would
    destroy, and why the control is a batch roll on a signal whose measured
    lag-1 autocorrelation is only 0.046."""
    da = torch.randn(B, L)
    rolled = roll_across_batch(da)
    for b in range(B):
        assert torch.equal(rolled[b], da[(b - 1) % B])


def test_g8_mutation_leg_a_roll_along_time_would_leak_the_future():
    """Why the control rolls the BATCH axis. A time roll wraps `da[L-1]` to
    position 0 and hands the model a quantity built from the end of its own
    window. Asserted so the choice is guarded, not merely documented."""
    da = torch.arange(B * L, dtype=torch.float32).reshape(B, L)
    time_rolled = torch.roll(da, shifts=1, dims=1)
    assert torch.equal(time_rolled[:, 0], da[:, -1])          # the leak
    batch_rolled = roll_across_batch(da)
    for b in range(B):
        assert not torch.equal(batch_rolled[b], da[b])        # genuinely moved
        assert torch.equal(batch_rolled[b], da[(b - 1) % B])  # and only across b


def test_g8_the_roll_refuses_a_batch_of_one():
    """At B = 1 the roll is the identity and the control would BE the arm."""
    with pytest.raises(ValueError, match="batch_size >= 2"):
        roll_across_batch(torch.randn(1, L))


def test_g8_rolled_really_differs_from_rpe():
    """The control must not accidentally be the arm."""
    torch.manual_seed(0)
    a = DopamineCharLM(V, D, K, fused=False, da_source="rpe", da_scale=TAU)
    torch.manual_seed(0)
    s = DopamineCharLM(V, D, K, fused=False, da_source="rolled", da_scale=TAU)
    idx = _idx()
    assert not torch.equal(a.dopamine_signal(idx, None), s.dopamine_signal(idx, None))


def test_g8_tanh_bounds_the_signal_and_is_exactly_zero_at_zero():
    """The squash is load-bearing: measured phi reaches -12.97, and an unsquashed
    -12.97 through a multiplicative gain inverts the sign of the current."""
    phi = torch.tensor([[-12.97, -1.0, 0.0, 1.0, 2.15]])
    da = dopamine(phi, TAU)
    # Bounded, and at the measured extreme it saturates to EXACTLY -1.0 in fp32
    # (tanh(-11.49) rounds to -1). That is the intended behaviour and is what
    # "bounded" has to mean here -- the point is that the modulation can never
    # exceed +/-k, not that it approaches the bound asymptotically.
    assert float(da.abs().max()) <= 1.0
    assert float(da[0, 0]) == -1.0
    assert float(da[0, 2]) == 0.0           # exactly zero RPE is exactly no drive
    assert -1.0 < float(da[0, 1]) < 0.0     # the bulk is not saturated
    assert 0.0 < float(da[0, 4]) < 1.0
    # ...and an unsquashed phi would invert the sign of the current at a gain of
    # only 0.1, which is why the squash is load-bearing rather than decorative.
    assert 1.0 + 0.1 * float(phi[0, 0]) < 0.0


def test_g8_mutation_leg_a_sign_flip_on_phi_is_visible():
    """phi > 0 must mean "better than predicted". A sign flip must not be a no-op."""
    torch.manual_seed(0)
    logits = torch.randn(B, L, V)
    idx = _idx()
    phi = rpe(logits, idx)
    assert not torch.equal(phi, -phi)
    assert float(phi[:, 1:].abs().max()) > 0.0


# ---------------------------------------------------------------------------
# G9 -- accounting, dispatch, config, state
# ---------------------------------------------------------------------------

def test_g9_parameter_count_is_the_baseline_plus_one_per_channel_per_layer():
    arm, plain = _pair()
    assert count_params(arm) == count_params(plain) + K * D
    assert count_params(arm) == dopamine_param_count(V, D, K)


def test_g9_parameter_count_at_the_committed_configuration():
    assert dopamine_param_count(205, 512, 2) == 736_461
    assert dopamine_param_count(205, 512, 2) == spiking_param_count(205, 512, 2) + 1024


def test_g9_build_model_dispatches_and_records_its_init():
    cfg = Config(arch="dopamine", vocab_size=V, d_model=D, n_layers=K,
                 device="cpu", fused=False, da_mode="add", da_source="rolled",
                 da_scale=0.5, da_gain_init=0.25)
    m = build_model(cfg)
    assert isinstance(m, DopamineCharLM)
    assert (m.da_mode, m.da_source, m.da_scale, m.da_gain_init) == \
           ("add", "rolled", 0.5, 0.25)
    assert all(float(p.detach().min()) == 0.25 and float(p.detach().max()) == 0.25
               for p in m.k)


def test_g9_config_rejects_a_degenerate_tau():
    with pytest.raises(ValueError, match="da_scale must be > 0"):
        Config(da_scale=0.0)


def test_g9_config_rejects_a_batch_of_one_under_the_rolled_control():
    with pytest.raises(ValueError, match="batch_size >= 2"):
        Config(arch="dopamine", da_source="rolled", batch_size=1)


def test_g9_config_rejects_unknown_modes_and_sources():
    with pytest.raises(ValueError, match="da_mode"):
        Config(da_mode="divide")
    with pytest.raises(ValueError, match="da_source"):
        Config(da_source="shuffle")


def test_g9_unknown_arch_still_names_every_arm():
    """`build_model`'s error message must list the new arm, or a typo sends a
    reader looking for an arch that exists."""
    cfg = Config(vocab_size=V, d_model=D, n_layers=K, device="cpu")
    cfg.arch = "nonesuch"          # bypasses __post_init__ deliberately
    with pytest.raises(ValueError, match="dopamine"):
        build_model(cfg)


def test_g9_state_round_trips_under_protocol_b():
    """Carried state must be accepted back unchanged in shape, as for every arm."""
    arm, _ = _pair("mult", "rpe")
    arm.eval()
    idx = _idx()
    _, state, _ = arm(idx, None)
    assert len(state) == K and all(s.shape == (B, D) for s in state)
    logits2, state2, _ = arm(idx, [s.detach() for s in state])
    assert len(state2) == K and logits2.shape == (B, L, V)


def test_g9_the_default_config_nests_the_baseline():
    """A run that forgets to set da_source trains the baseline, not something
    undocumented. Same rule as noise_amp = 0.0 and mu_init = 1.0."""
    assert Config().da_source == "off"
    assert Config().da_gain_init == 0.0


# ---------------------------------------------------------------------------
# G7 -- capture (GPU only)
# ---------------------------------------------------------------------------

@pytest.mark.cuda
@requires_cuda
def test_g7_survives_cuda_graph_capture():
    """Forward AND backward, inside a CUDAGraph, with the no_grad pass nested.

    The trainer captures the full step, so an arm that cannot be captured runs at
    ~1/11 throughput and is a different experiment. This is the one structural
    risk the two-pass design carried, so it is proven on the real module rather
    than argued.
    """
    torch.manual_seed(0)
    arm = DopamineCharLM(V, D, K, fused=True, da_mode="mult", da_source="rpe",
                         da_scale=TAU, da_gain_init=0.05).cuda()
    arm.train()
    static_x = _idx().cuda()

    def step():
        for p in arm.parameters():
            if p.grad is not None:
                p.grad.zero_()
        logits, _, _ = arm(static_x, None)
        logits.square().mean().backward()

    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(3):
            step()
    torch.cuda.current_stream().wait_stream(side)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        step()

    # Replay against data the graph has never seen -- what would catch a captured
    # kernel reading a stale address rather than the static buffer (risk R5).
    static_x.copy_(_idx(seed=99).cuda())
    graph.replay()
    torch.cuda.synchronize()
    got = {n: p.grad.clone() for n, p in arm.named_parameters()}

    for p in arm.parameters():
        p.grad = None
    logits, _, _ = arm(static_x, None)
    logits.square().mean().backward()
    for n, p in arm.named_parameters():
        assert torch.allclose(got[n], p.grad, rtol=1e-4, atol=1e-6), (
            f"{n} after replay max|d|={float((got[n] - p.grad).abs().max()):.3e}")


@pytest.mark.cuda
@requires_cuda
def test_g7_fused_and_eager_agree():
    """R10 is inherited, not new -- but the arm's own path is asserted anyway."""
    torch.manual_seed(0)
    fused = DopamineCharLM(V, D, K, fused=True, da_source="rpe", da_scale=TAU,
                           da_gain_init=0.2).cuda()
    torch.manual_seed(0)
    eager = DopamineCharLM(V, D, K, fused=False, da_source="rpe", da_scale=TAU,
                           da_gain_init=0.2).cuda()
    eager.load_state_dict(fused.state_dict())
    idx = _idx().cuda()
    a, _, _ = fused(idx, None)
    b, _, _ = eager(idx, None)
    assert torch.equal(a, b)


# ---------------------------------------------------------------------------
# The L = 1 boundary: EXP_001's horizon probe sweeps k down to 1
# ---------------------------------------------------------------------------

def test_rpe_at_length_one_is_zero_not_an_error():
    """At L = 1 every position is position 0, which has no preceding prediction.

    Zeros is the consistent extension of `phi_0 = 0`, and it is what makes D5's
    k = 1 rung computable at all: `EXP_001`'s probe reshapes `[B, L]` to
    `[B*L//k, k]` and would otherwise hand `rpe` a single position.
    """
    torch.manual_seed(0)
    phi = rpe(torch.randn(B, 1, V), _idx(length=1))
    assert phi.shape == (B, 1)
    assert torch.equal(phi, torch.zeros(B, 1))


def test_the_arm_is_the_baseline_at_zero_context_by_construction():
    """A structural property worth guarding, not an accident of the boundary.

    With one character of context the dopamine signal does not exist, so the arm
    IS `SpikingCharLM` there -- bitwise, at any `k`, including a large one. Any
    gain this arm produces must therefore come from longer contexts.
    """
    arm, plain = _pair("mult", "rpe", k_init=2.0)
    arm.eval(); plain.eval()
    one = _idx(b=B, length=1)
    a, _, _ = arm(one, None)
    b, _, _ = plain(one, None)
    assert torch.equal(a, b)


def test_the_l1_extension_changes_nothing_at_the_training_length():
    """The boundary branch must be inert everywhere a run actually lives.

    Training and both evaluation protocols run at L = 256; the branch above is
    reachable only from the horizon probe. Asserted so that adding it cannot have
    moved a training run.
    """
    torch.manual_seed(0)
    logits = torch.randn(B, L, V)
    idx = _idx()
    phi = rpe(logits, idx)
    logp = F.log_softmax(logits.float(), dim=-1)
    prev = logp[:, :-1]
    expect = F.pad(-(prev.exp() * prev).sum(-1)
                   + prev.gather(-1, idx[:, 1:].unsqueeze(-1)).squeeze(-1), (1, 0))
    assert torch.equal(phi, expect)
