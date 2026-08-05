"""EXP_012: why is EXP_011's fold ~1000x tighter than EXP_008's?

Pre-registered in `experiments/logs/EXP_012_fold_gap.md`. This script implements
that file's four legs and resolves Y1-Y6 against the bars fixed in its section 3.
It applies no tolerance to any fold and states no verdict about decision #6.

WHAT IS ALREADY EXCLUDED, AND IS NOT MEASURED HERE
---------------------------------------------------
EXP_012 section 0 records two exclusions made by reading the committed source
before this script existed: the two `fold_into_*_state_dict` methods have the
same six-line body, and the two `fold_check` functions compute the same quantity
by the same code path. So neither the fold's arithmetic nor the metric can be the
explanation, and what remains is the neuron and the operating point.

THE FOUR LEGS, IN THE ORDER THAT SEPARATES THOSE TWO
-----------------------------------------------------
1. **The identity and the injected perturbation** (EXP_008 section 9.5's three
   precisions, generalised to dispatch on `arch`). Y1 is the algebra control; Y2
   is the load-bearing one -- if the perturbation entering the neuron already
   differs, the answer is upstream and legs 3 and 4 are moot.
2. **The cascade**: spike flips per layer, the per-layer amplification ratio, the
   logits difference and a subset bpc. The like-for-like table EXP_008 has and
   EXP_011 does not.
3. **The two factors of the flip model, measured separately.** `flips ~
   N * rho_u(0) * E|du|`. Neither factor is observable from any scan's return
   value, so the recursion is recomputed here -- which is exactly why S1 exists.
4. **The cross-over**, which is the decisive leg: the same current and the same
   perturbation through *both* neurons, at matched firing rate. Leg 3 can only
   say the two arms differ; leg 4 is what says whether the neuron is why.

WHY LEG 3 NEEDS A SELF-CHECK MORE THAN THE OTHERS
--------------------------------------------------
Legs 1, 2 and 4 call committed code. Leg 3 cannot: `u`, the decision variable, is
consumed inside the scan and never returned, so observing it means recomputing
the recursion here. A probe that reimplements a neuron will happily agree with
itself and measure nothing -- EXP_004 section 9.2's lesson, that a contract is
only guarded where the quantity it constrains is actually observed. S1 therefore
requires this file's recomputed `u` to yield spikes **bit-identical** to the
committed `lif_scan` / `twocomp_scan`, on every run and both layers, and aborts
otherwise.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO / "src") not in sys.path:
    sys.path.insert(0, str(_REPO / "src"))

from snn.config import Config  # noqa: E402
from snn.data import Corpus, windowed_eval_batches  # noqa: E402
from snn.metrics import bits_per_char  # noqa: E402
from snn.model import build_model  # noqa: E402
from snn.neuron import lif_scan  # noqa: E402
from snn.prescan import threshold_gain  # noqa: E402
from snn.twocomp import twocomp_scan  # noqa: E402

RUNS = _REPO / "experiments" / "runs"

# --- EXP_012 section 3's bars, transcribed. Nothing below reads a number that is
# --- not fixed here, and none of these is a fold-in tolerance.
Y1_LEG_C_MAX = 1e-14           # relative to |cur|max
Y2_LEG_A_RANGE = (2.3e-07, 3.0e-06)
Y3_L0_FLIP_FRACTION_MAX = 1.6e-05
Y4_AMPLIFICATION_MAX = 5.0
Y5_CROSSOVER_RATIO_MIN = 10.0
Y6_MODEL_FACTOR = 3.0

# EXP_008's committed figures, from docs/reports/data/exp_008_fold_residual.json,
# quoted here so the resolver's comparisons are readable at the point of use.
EXP008_LEG_A = (6.902e-07, 7.281e-07, 9.872e-07)
EXP008_L0_FRACTION = (3.182e-04, 1.600e-04, 1.699e-04)
EXP008_AMPLIFICATION = (14.60, 15.89, 17.92)

FOLD_TARGET = {"threshold": "snn", "twocomp_threshold": "twocomp"}


def _load(run: str):
    """The arm, its folded counterpart, and the config -- for either arch.

    `008_chase_fold.py`'s `_load` is hard-wired to `fold_into_spiking_state_dict`
    and `arch="snn"`. This one dispatches, so the two arms go through one code
    path and a difference between them cannot be a difference between scripts.
    """
    ck = torch.load(RUNS / run / "ckpt_final.pt", map_location="cpu",
                    weights_only=False)
    known = {f.name for f in dataclasses.fields(Config)}
    cfg = Config(**{k: v for k, v in ck["config"].items() if k in known})
    if cfg.arch not in FOLD_TARGET:
        raise SystemExit(f"{run}: arch={cfg.arch!r} has no fold; expected one of "
                         f"{sorted(FOLD_TARGET)}")

    arm = build_model(cfg)
    arm.load_state_dict(ck["model"])
    arm.eval()

    target = FOLD_TARGET[cfg.arch]
    folded_sd = (arm.fold_into_spiking_state_dict() if target == "snn"
                 else arm.fold_into_twocomp_state_dict())
    base_cfg = dataclasses.replace(cfg, arch=target, run_name=f"{run}_folded")
    base = build_model(base_cfg)
    base.load_state_dict(folded_sd, strict=True)
    base.eval()
    return arm, base, cfg, target


# ---------------------------------------------------------------------------
# leg 1 -- the identity, at three precisions (EXP_008 section 9.5, unchanged)
# ---------------------------------------------------------------------------

@torch.no_grad()
def identity_on_the_current(arm, base, h: torch.Tensor, layer: int) -> dict:
    """A / B / C on layer 0's current. Arch-agnostic: it touches only the Linear.

    A  as shipped         fp32 storage + fp32 GEMM, in both orderings
    B  fp32 fold, fp64 GEMM  the fp32-rounded folded weight, evaluated exactly
    C  fp64 fold, fp64 GEMM  the fold computed in float64 as well

    Layer 0 only, for EXP_008's reason: layer 1's input is layer 0's spike train,
    which the two models do not agree on, so a layer-1 comparison would mix the
    identity with the disagreement it causes.
    """
    lin_a, lin_b = arm.layers[layer], base.layers[layer]
    g = arm.input_gain(layer)                       # [1, d]

    scaled_after = g * lin_a(h)                     # g*(W·h + b)
    folded_before = lin_b(h)                        # (fp32(g*W))·h + fp32(g*b)
    dA = (scaled_after - folded_before).abs()

    hd, gd = h.double(), g.double()
    a64 = gd * F.linear(hd, lin_a.weight.double(), lin_a.bias.double())
    dB = (a64 - F.linear(hd, lin_b.weight.double(), lin_b.bias.double())).abs()
    exact = F.linear(hd, lin_a.weight.double() * gd.reshape(-1, 1),
                     lin_a.bias.double() * gd.flatten())
    dC = (a64 - exact).abs()

    cur_max = float(scaled_after.abs().max())
    return {
        "layer": layer,
        "current_abs_max": cur_max,
        "A_shipped_fp32_max_abs_diff": float(dA.max()),
        "A_in_units_of_current_max": float(dA.max()) / cur_max,
        "A_mean_abs_diff": float(dA.mean()),
        "B_fp32fold_fp64gemm_max_abs_diff": float(dB.max()),
        "B_in_units_of_current_max": float(dB.max()) / cur_max,
        "C_exact_fold_fp64_max_abs_diff": float(dC.max()),
        "C_in_units_of_current_max": float(dC.max()) / cur_max,
        "eps_fp32": float(torch.finfo(torch.float32).eps),
        "eps_fp64": float(torch.finfo(torch.float64).eps),
    }


# ---------------------------------------------------------------------------
# leg 2 -- the cascade
# ---------------------------------------------------------------------------

@torch.no_grad()
def spikes_and_logits(arm, base, idx: torch.Tensor) -> dict:
    """Spike disagreement per layer, the amplification ratio, and the logits."""
    arm.keep_spikes = base.keep_spikes = True
    try:
        la, _, aux_a = arm(idx, None)
        lb, _, aux_b = base(idx, None)
    finally:
        arm.keep_spikes = base.keep_spikes = False

    per_layer = []
    for k, (sa, sb) in enumerate(zip(aux_a["spikes"], aux_b["spikes"])):
        diff = (sa != sb)
        n = int(sa.numel())
        per_layer.append({
            "layer": k,
            "n_spike_sites": n,
            "n_disagreements": int(diff.sum()),
            "disagreement_fraction": float(diff.float().mean()),
            "arm_firing_rate": float(sa.mean()),
            "folded_firing_rate": float(sb.mean()),
            # limitation 2: one batch cannot resolve a fraction below ~1/n.
            "resolution_floor_fraction": 1.0 / n,
        })
    amp = None
    if len(per_layer) >= 2 and per_layer[0]["disagreement_fraction"] > 0:
        amp = (per_layer[1]["disagreement_fraction"]
               / per_layer[0]["disagreement_fraction"])
    dl = (la - lb).abs()
    return {
        "per_layer": per_layer,
        "amplification_layer1_over_layer0": amp,
        "logits_max_abs_diff": float(dl.max()),
        "logits_abs_max": float(la.abs().max()),
    }


# ---------------------------------------------------------------------------
# leg 3 -- the decision variable, recomputed so it can be observed at all
# ---------------------------------------------------------------------------

@torch.no_grad()
def decision_variable(cur: torch.Tensor, *, kind: str, thr: float,
                      beta: float, w=None, beta_s=None):
    """Recompute the membrane recursion and return (spikes, u).

    `u` is the argument `atan_spike` sees -- `v_pre - thr` for the LIF and
    `vf + w*vs - thr` for the two-compartment neuron -- and a spike is emitted
    iff `u >= 0` (`surrogate.atan_spike`: `s_hard = (x >= 0)`).

    The order of operations is transcribed from `lif_scan_eager` and
    `twocomp_scan_eager` rather than rearranged, because S1 requires the spikes
    this produces to be bit-identical to the committed kernel's and floating-point
    addition is not associative. `twocomp.py`'s own docstring makes the same point
    about the two places the fused kernel suppresses FMA contraction.
    """
    B, L, d = cur.shape
    us, spikes = [], []
    if kind == "twocomp":
        vf = cur.new_zeros(B, d)
        vs = cur.new_zeros(B, d)
        for t in range(L):
            vf = vf * beta + cur[:, t]
            vs = vs * beta_s + cur[:, t]
            u = vf + w * vs - thr
            s = (u >= 0).to(cur.dtype)
            us.append(u)
            spikes.append(s)
            vf = vf * (1.0 - s)
    else:
        v = cur.new_zeros(B, d)
        for t in range(L):
            v_pre = v * beta + cur[:, t]
            u = v_pre - thr
            s = (u >= 0).to(cur.dtype)
            us.append(u)
            spikes.append(s)
            v = v_pre * (1.0 - s)
    return torch.stack(spikes, dim=1), torch.stack(us, dim=1)


def _neuron_kwargs(model, cfg: Config, layer: int, target: str) -> dict:
    """The scan parameters for one layer, read from the model, not re-derived."""
    if target == "twocomp":
        return {"kind": "twocomp", "thr": cfg.threshold, "beta": cfg.beta,
                "w": model.w[layer], "beta_s": model.slow_decay(layer)}
    return {"kind": "lif", "thr": cfg.threshold, "beta": cfg.beta}


@torch.no_grad()
def _effective_currents(arm, base, h_arm, h_base, layer: int):
    """The current each model's scan actually receives at `layer`.

    The arm's `_scan` applies `threshold_gain` internally, so the arm's effective
    current is `g * (W·h + b)`; the folded model has no `thr_log` and its
    effective current is its Linear's output as-is. These are the same two
    tensors leg 1 compares at layer 0.
    """
    cur_arm = threshold_gain(arm._project(arm.layers[layer], h_arm),
                             arm.thr_log[layer])
    cur_base = base._project(base.layers[layer], h_base)
    return cur_arm, cur_base


@torch.no_grad()
def flip_model(arm, base, cfg: Config, target: str, idx: torch.Tensor,
               ladder: list[float]) -> dict:
    """Leg 3, plus S1. Density at the threshold, realised perturbation, prediction.

    The predicted flip count is `N * rho_u(0) * E|du|`, and `rho_u(0)` is
    estimated at the perturbation's own scale -- `rho ~ P(|u| < E|du|) / (2*E|du|)`
    -- which makes the prediction exactly `N * P(|u| < E|du|) / 2`. The ladder is
    reported alongside because a histogram estimate is bin-width dependent
    (EXP_012 limitation 4) and the single-bin number should not be read alone.
    """
    h_arm = arm.embed(idx)
    h_base = base.embed(idx)
    out, s1 = [], []
    for layer in range(cfg.n_layers):
        cur_arm, cur_base = _effective_currents(arm, base, h_arm, h_base, layer)
        kw = _neuron_kwargs(arm, cfg, layer, target)
        sp_arm, u_arm = decision_variable(cur_arm, **kw)
        kw_b = _neuron_kwargs(base, cfg, layer, target)
        sp_base, u_base = decision_variable(cur_base, **kw_b)

        # --- S1: the probe must reproduce the committed kernel, bit for bit.
        ref_arm, _ = arm._scan(arm._project(arm.layers[layer], h_arm),
                               arm.init_state(idx.shape[0], idx.device)[layer],
                               layer)
        ref_base, _ = base._scan(base._project(base.layers[layer], h_base),
                                 base.init_state(idx.shape[0], idx.device)[layer],
                                 layer)
        s1.append({
            "layer": layer,
            "arm_bit_identical": bool(torch.equal(sp_arm, ref_arm)),
            "folded_bit_identical": bool(torch.equal(sp_base, ref_base)),
        })

        du = (u_arm - u_base).abs()
        mean_du = float(du.mean())
        n = int(u_arm.numel())
        near = {f"{h:.0e}": float((u_arm.abs() < h).float().mean())
                for h in ladder}
        p_at_scale = float((u_arm.abs() < mean_du).float().mean())
        measured = int((sp_arm != sp_base).sum())
        out.append({
            "layer": layer,
            "n_sites": n,
            "u_abs_mean": float(u_arm.abs().mean()),
            "u_std": float(u_arm.std()),
            "du_mean_abs": mean_du,
            "du_max_abs": float(du.max()),
            "fraction_within_ladder": near,
            "density_at_threshold_per_unit_u": (p_at_scale / (2 * mean_du)
                                                if mean_du > 0 else None),
            "predicted_flips": n * p_at_scale / 2.0,
            "measured_flips": measured,
            "model_ratio_predicted_over_measured": (
                (n * p_at_scale / 2.0) / measured if measured else None),
        })
        # the two [B, L, d] fp32 membranes per layer are the memory ceiling here
        del cur_arm, cur_base, u_arm, u_base, du, sp_arm, sp_base, ref_arm, ref_base
        torch.cuda.empty_cache()

        # layer k+1's input is layer k's spikes, and the two models disagree on
        # them -- which is the cascade leg 2 measures, and is why each model must
        # advance on its own output rather than on a shared one.
        h_arm, _, _ = _advance(arm, h_arm, layer)
        h_base, _, _ = _advance(base, h_base, layer)
    return {"per_layer": out, "s1_probe_reproduces_kernel": s1}


@torch.no_grad()
def _advance(model, h, layer):
    cur = model._project(model.layers[layer], h)
    emitted, v = model._scan(cur, model.init_state(h.shape[0], h.device)[layer],
                             layer)
    return emitted, v, cur


# ---------------------------------------------------------------------------
# leg 4 -- the cross-over: one current, one perturbation, two neurons
# ---------------------------------------------------------------------------

@torch.no_grad()
def _scan_spikes(cur, *, kind, thr, cfg: Config, w=None, beta_s=None):
    """One scan through the committed kernel. `alpha` and `reset` come from the
    run's own config rather than from a literal, so leg 4 cannot silently drift
    from the arm it is drawn from."""
    v0 = (cur.new_zeros(cur.shape[0], 2 * cur.shape[2]) if kind == "twocomp"
          else cur.new_zeros(cur.shape[0], cur.shape[2]))
    if kind == "twocomp":
        sp, _ = twocomp_scan(cur, v0, w, beta_s, cfg.beta, thr,
                             cfg.surrogate_alpha, cfg.fused)
    else:
        sp, _ = lif_scan(cur, v0, cfg.beta, thr, cfg.surrogate_alpha,
                         cfg.reset, cfg.fused)
    return sp


@torch.no_grad()
def _match_threshold(cur, target_rate: float, *, kind, cfg: Config, w=None,
                     beta_s=None, iters: int = 30) -> tuple:
    """Bisect the scalar threshold until the firing rate matches `target_rate`.

    Firing rate is monotone non-increasing in the threshold, so bisection is
    sound. Returns (threshold, achieved_rate) and the caller reports the achieved
    rate rather than assuming the match is exact.
    """
    lo, hi = 1e-4, 200.0
    thr, rate = 1.0, float("nan")
    for _ in range(iters):
        thr = 0.5 * (lo + hi)
        rate = float(_scan_spikes(cur, kind=kind, thr=thr, cfg=cfg, w=w,
                                  beta_s=beta_s).mean())
        if rate > target_rate:
            lo = thr
        else:
            hi = thr
    return thr, rate


@torch.no_grad()
def crossover(cur_a: torch.Tensor, cur_b: torch.Tensor, cfg: Config,
              tc_w, tc_beta_s) -> dict:
    """Both neurons, on the identical current pair leg 1 compares.

    `cur_a` is the arm's effective layer-0 current and `cur_b` the folded model's,
    so the perturbation between them is exactly the one the fold injects. The only
    thing that varies across the rows below is the neuron and its threshold.

    Firing rate is the confound -- a neuron that fires less has fewer sites at
    risk -- so each neuron is run twice: once at the committed threshold, and once
    at the threshold that matches the *other* neuron's rate on the unperturbed
    current.
    """
    def flips(kind, thr, w=None, beta_s=None):
        sa = _scan_spikes(cur_a, kind=kind, thr=thr, cfg=cfg, w=w, beta_s=beta_s)
        sb = _scan_spikes(cur_b, kind=kind, thr=thr, cfg=cfg, w=w, beta_s=beta_s)
        n = int(sa.numel())
        d = int((sa != sb).sum())
        return {"threshold": thr, "firing_rate": float(sa.mean()),
                "n_sites": n, "n_flips": d, "flip_fraction": d / n}

    tc = flips("twocomp", cfg.threshold, w=tc_w, beta_s=tc_beta_s)
    lif = flips("lif", cfg.threshold)

    thr_lif, rate_lif = _match_threshold(cur_a, tc["firing_rate"], kind="lif",
                                         cfg=cfg)
    lif_matched = flips("lif", thr_lif)
    lif_matched["achieved_rate_vs_target"] = [rate_lif, tc["firing_rate"]]

    thr_tc, rate_tc = _match_threshold(cur_a, lif["firing_rate"], kind="twocomp",
                                       cfg=cfg, w=tc_w, beta_s=tc_beta_s)
    tc_matched = flips("twocomp", thr_tc, w=tc_w, beta_s=tc_beta_s)
    tc_matched["achieved_rate_vs_target"] = [rate_tc, lif["firing_rate"]]

    def ratio(a, b):
        return (a["flip_fraction"] / b["flip_fraction"]
                if b["flip_fraction"] > 0 else None)

    return {
        "twocomp_committed_thr": tc,
        "lif_committed_thr": lif,
        "lif_rate_matched_to_twocomp": lif_matched,
        "twocomp_rate_matched_to_lif": tc_matched,
        "ratio_lif_over_twocomp_committed": ratio(lif, tc),
        "ratio_lif_over_twocomp_rate_matched": ratio(lif_matched, tc),
        "ratio_lif_over_twocomp_rate_matched_other_way": ratio(lif, tc_matched),
    }


# ---------------------------------------------------------------------------
# subset bpc, for the end of the chain
# ---------------------------------------------------------------------------

@torch.no_grad()
def subset_bpc(model, data, cfg: Config, n_windows: int) -> float:
    total, chars, seen = 0.0, 0, 0
    for x, y in windowed_eval_batches(data, cfg.batch_size, cfg.seq_len):
        x, y = x.to(cfg.device), y.to(cfg.device)
        logits, _, _ = model(x, None)
        total += float(F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]).float(), y.reshape(-1),
            reduction="sum"))
        chars += y.numel()
        seen += int(x.shape[0])
        if seen >= n_windows:
            break
    return bits_per_char(total, chars)


# ---------------------------------------------------------------------------
# EXP_012 section 3 and section 4, applied
# ---------------------------------------------------------------------------

def resolve(report: dict) -> dict:
    """Y1-Y6 and the section 4 truth table, applied mechanically to the measured
    context. No number below is chosen after the fact; every bar is a constant at
    the top of this file, transcribed from the pre-registration."""
    thr_runs = [r for r, e in report["runs"].items() if e["arch"] == "threshold"]
    cmp_runs = [r for r, e in report["runs"].items()
                if e["arch"] == "twocomp_threshold"]
    R = report["runs"]

    def legA(runs):
        return [R[r]["identity"]["A_in_units_of_current_max"] for r in runs]

    def legC(runs):
        return [R[r]["identity"]["C_in_units_of_current_max"] for r in runs]

    def l0frac(runs):
        return [R[r]["cascade"]["per_layer"][0]["disagreement_fraction"]
                for r in runs]

    def amp(runs):
        return [R[r]["cascade"]["amplification_layer1_over_layer0"] for r in runs]

    p = {}
    c_vals = legC(cmp_runs)
    p["Y1"] = {
        "statement": "leg C on the composed arm is <= 1e-14 of |cur|max "
                     "(the identity is exact there too)",
        "bar": Y1_LEG_C_MAX, "measured": c_vals,
        "held": bool(c_vals) and all(v <= Y1_LEG_C_MAX for v in c_vals),
    }
    a_vals = legA(cmp_runs)
    p["Y2"] = {
        "statement": "leg A on the composed arm is within 3x of EXP_008's "
                     "6.902e-07 - 9.872e-07 (the injected perturbation matches)",
        "bar": list(Y2_LEG_A_RANGE), "measured": a_vals,
        "exp_008_reference": list(EXP008_LEG_A),
        "held": bool(a_vals) and all(Y2_LEG_A_RANGE[0] <= v <= Y2_LEG_A_RANGE[1]
                                     for v in a_vals),
        "load_bearing": "if this fails the gap is upstream of the neuron and "
                        "Y3-Y6 are moot (EXP_012 section 4 row 3)",
    }
    f_vals = l0frac(cmp_runs)
    p["Y3"] = {
        "statement": "layer-0 flip fraction on the composed arm is < 1.6e-05, "
                     "i.e. >= 10x below EXP_008's 1.600e-04 - 3.182e-04",
        "bar": Y3_L0_FLIP_FRACTION_MAX, "measured": f_vals,
        "exp_008_reference": list(EXP008_L0_FRACTION),
        "held": bool(f_vals) and all(v < Y3_L0_FLIP_FRACTION_MAX for v in f_vals),
    }
    amp_vals = [v for v in amp(cmp_runs) if v is not None]
    p["Y4"] = {
        "statement": "layer-1 / layer-0 flip fraction on the composed arm is "
                     "< 5x, against EXP_008's 14.6 / 15.9 / 17.9",
        "bar": Y4_AMPLIFICATION_MAX, "measured": amp(cmp_runs),
        "exp_008_reference": list(EXP008_AMPLIFICATION),
        "held": (all(v < Y4_AMPLIFICATION_MAX for v in amp_vals)
                 if amp_vals else None),
        "note": None if amp_vals else "layer-0 flip count was 0, so the ratio is "
                                      "undefined; reported as null, not as a hit",
    }
    xr = [R[r]["crossover"]["ratio_lif_over_twocomp_rate_matched"]
          for r in report["runs"] if R[r].get("crossover")]
    xr = [v for v in xr if v is not None]
    p["Y5"] = {
        "statement": "at matched firing rate, on the identical current and "
                     "perturbation, the two-compartment neuron flips >= 10x "
                     "fewer spikes than the LIF",
        "bar": Y5_CROSSOVER_RATIO_MIN, "measured": xr,
        "held": bool(xr) and all(v >= Y5_CROSSOVER_RATIO_MIN for v in xr),
        "decisive": "this is what separates the neuron from the operating point",
    }
    ratios = []
    for r, e in R.items():
        for ly in e["flip_model"]["per_layer"]:
            v = ly["model_ratio_predicted_over_measured"]
            if v is not None:
                ratios.append({"run": r, "layer": ly["layer"], "ratio": v})
    p["Y6"] = {
        "statement": "N * rho_u(0) * E|du| predicts the measured flip count to "
                     "within 3x, on both arms and both layers",
        "bar": Y6_MODEL_FACTOR, "measured": ratios,
        "held": bool(ratios) and all(
            1.0 / Y6_MODEL_FACTOR <= x["ratio"] <= Y6_MODEL_FACTOR
            for x in ratios),
    }

    # --- EXP_012 section 4's truth table, resolved on Y2 then Y5.
    if not p["Y2"]["held"]:
        verdict = ("the gap is UPSTREAM of the neuron: leg A differs, so the "
                   "fold's inputs differ in conditioning and section 1's "
                   "derivation is set aside")
        floor_scoped = False
    elif p["Y5"]["held"]:
        verdict = ("the two-compartment neuron is INTRINSICALLY less sensitive "
                   "to a reparameterisation of its own weights; the ~2e-3 floor "
                   "is a property of the baseline LIF neuron and does not "
                   "transfer to the adopted arm")
        floor_scoped = True
    else:
        verdict = ("the gap is real and is an OPERATING-POINT effect, not a "
                   "neuron effect; a fold-in tolerance cannot be a project "
                   "constant because it moves with where the arm trained")
        floor_scoped = False

    return {
        "predictions": p,
        "verdict": verdict,
        "section_6_5_noise_floor_needs_scope_qualifier": floor_scoped,
        "decision_6": "UNTOUCHED AND OPEN. This experiment sets no tolerance and "
                      "proposes no number for W1 (EXP_012 section 4).",
        "threshold_runs": thr_runs, "composed_runs": cmp_runs,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", default="threshold_s0,threshold_s1,threshold_s2,"
                                      "compose_s1,compose_s2")
    ap.add_argument("--windows", type=int, default=512)
    ap.add_argument("--crossover-from", default="compose_s1,threshold_s0",
                    help="runs whose layer-0 current pair drives leg 4, both "
                         "directions (EXP_012 section 2 leg 4)")
    ap.add_argument("--twocomp-params-from", default="twocomp_s0",
                    help="the adopted arm's w/beta_s, used for leg 4's "
                         "two-compartment neuron when the current comes from a "
                         "run that has none")
    ap.add_argument("--out",
                    default="docs/reports/data/exp_012_fold_gap.json")
    args = ap.parse_args(argv)

    ladder = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2]
    report = {
        "experiment": "EXP_012_fold_gap",
        "question": "why is EXP_011's fold ~1000x tighter than EXP_008's, given "
                    "the same identity, the same fold arithmetic and the same "
                    "metric?",
        "sets_no_tolerance": True,
        "density_ladder": ladder,
        "runs": {},
    }

    crossover_runs = [r.strip() for r in args.crossover_from.split(",") if r.strip()]

    # the adopted arm's two-compartment parameters, for leg 4's LIF-side current
    ck = torch.load(RUNS / args.twocomp_params_from / "ckpt_final.pt",
                    map_location="cpu", weights_only=False)
    known = {f.name for f in dataclasses.fields(Config)}
    tc_cfg = Config(**{k: v for k, v in ck["config"].items() if k in known})
    tc_ref = build_model(tc_cfg)
    tc_ref.load_state_dict(ck["model"])
    tc_ref.eval()
    report["twocomp_params_from"] = args.twocomp_params_from

    for run in [r.strip() for r in args.runs.split(",") if r.strip()]:
        arm, base, cfg, target = _load(run)
        corpus = Corpus(cfg.corpus, cfg.data_dir)
        data = corpus.split("test")
        x, _y = next(iter(windowed_eval_batches(data, cfg.batch_size, cfg.seq_len)))
        idx = x.to(cfg.device)
        h = arm.embed(idx).detach()

        entry = {
            "arch": cfg.arch,
            "folds_to": target,
            "threshold_multiplier_exp_theta": [
                float(arm.threshold_multiplier(k).mean())
                for k in range(cfg.n_layers)],
            "identity": identity_on_the_current(arm, base, h, 0),
            "cascade": spikes_and_logits(arm, base, idx),
            "flip_model": flip_model(arm, base, cfg, target, idx, ladder),
            "subset_windows": args.windows,
        }
        entry["subset_bpc_arm"] = subset_bpc(arm, data, cfg, args.windows)
        entry["subset_bpc_folded"] = subset_bpc(base, data, cfg, args.windows)
        entry["subset_bpc_residual"] = abs(entry["subset_bpc_arm"]
                                           - entry["subset_bpc_folded"])

        if run in crossover_runs:
            h_arm = arm.embed(idx)
            cur_a, cur_b = _effective_currents(arm, base, h_arm, base.embed(idx), 0)
            w = (arm.w[0] if target == "twocomp" else tc_ref.w[0])
            bs = (arm.slow_decay(0) if target == "twocomp"
                  else tc_ref.slow_decay(0))
            entry["crossover"] = crossover(cur_a, cur_b, cfg, w, bs)
            entry["crossover"]["twocomp_params_from"] = (
                run if target == "twocomp" else args.twocomp_params_from)
            del cur_a, cur_b, h_arm

        report["runs"][run] = entry
        _print_run(run, entry)
        del arm, base
        torch.cuda.empty_cache()

    report["resolution"] = resolve(report)

    s1_bad = [(r, s) for r, e in report["runs"].items()
              for s in e["flip_model"]["s1_probe_reproduces_kernel"]
              if not (s["arm_bit_identical"] and s["folded_bit_identical"])]
    report["S1_all_layers_bit_identical"] = not s1_bad
    if s1_bad:
        print("\nS1 FAILED -- the probe does not reproduce the committed kernel:")
        for r, s in s1_bad:
            print(f"   {r} layer {s['layer']}: {s}")
        print("EXP_012 section 6 aborts here; no density is reported.")
        return 2

    out = _REPO / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    _print_resolution(report["resolution"])
    print(f"\nWROTE {args.out}")
    return 0


def _print_run(run: str, e: dict) -> None:
    i = e["identity"]
    print(f"\n{run}  arch={e['arch']} -> {e['folds_to']}   "
          f"|cur|max {i['current_abs_max']:.2f}   "
          f"exp(theta) {['%.3f' % v for v in e['threshold_multiplier_exp_theta']]}")
    print(f"  leg 1  A {i['A_in_units_of_current_max']:.3e} of |cur|max "
          f"({i['A_in_units_of_current_max'] / i['eps_fp32']:.1f} fp32 eps)   "
          f"C {i['C_in_units_of_current_max']:.3e} "
          f"({i['C_in_units_of_current_max'] / i['eps_fp64']:.1f} fp64 eps)")
    for ly in e["cascade"]["per_layer"]:
        print(f"  leg 2  layer {ly['layer']}: {ly['n_disagreements']} flips in "
              f"{ly['n_spike_sites']:,} ({ly['disagreement_fraction']:.3e}), "
              f"rate {ly['arm_firing_rate']:.6f}")
    amp = e["cascade"]["amplification_layer1_over_layer0"]
    print(f"  leg 2  amplification layer1/layer0: "
          f"{'n/a' if amp is None else '%.2fx' % amp};  logits max |diff| "
          f"{e['cascade']['logits_max_abs_diff']:.3e} on "
          f"{e['cascade']['logits_abs_max']:.1f};  "
          f"{e['subset_windows']}-window bpc residual "
          f"{e['subset_bpc_residual']:.2e}")
    for ly in e["flip_model"]["per_layer"]:
        r = ly["model_ratio_predicted_over_measured"]
        tail = (f"(x{r:.2f})" if r is not None
                else "(measured 0 -- ratio undefined)")
        print(f"  leg 3  layer {ly['layer']}: E|du| {ly['du_mean_abs']:.3e}, "
              f"u sd {ly['u_std']:.3f}, predicted "
              f"{ly['predicted_flips']:.1f} vs measured "
              f"{ly['measured_flips']} {tail}")
    if e.get("crossover"):
        c = e["crossover"]
        print(f"  leg 4  same current + same perturbation, both neurons:")
        for k in ("twocomp_committed_thr", "lif_committed_thr",
                  "lif_rate_matched_to_twocomp", "twocomp_rate_matched_to_lif"):
            r = c[k]
            print(f"           {k:34s} thr {r['threshold']:8.4f}  rate "
                  f"{r['firing_rate']:.6f}  flips {r['n_flips']:>8,} "
                  f"({r['flip_fraction']:.3e})")
        print(f"           LIF / twocomp flip ratio: committed "
              f"{c['ratio_lif_over_twocomp_committed']}, rate-matched "
              f"{c['ratio_lif_over_twocomp_rate_matched']}")


def _print_resolution(res: dict) -> None:
    print("\n" + "=" * 72)
    for k, v in res["predictions"].items():
        held = v["held"]
        mark = "HELD" if held else ("FAILED" if held is False else "NULL")
        print(f"  {k}  {mark:6s}  {v['statement'][:70]}")
    print("=" * 72)
    print(f"  VERDICT: {res['verdict']}")
    print(f"  section 6.5 needs a scope qualifier: "
          f"{res['section_6_5_noise_floor_needs_scope_qualifier']}")
    print(f"  {res['decision_6']}")


if __name__ == "__main__":
    raise SystemExit(main())
