"""Phase-2 gate R10, third opinion: our LIF against snnTorch's, not against itself.

The R10 test in tests/test_neuron_equivalence.py compares our fused CUDA kernel
against our own eager reference. That catches a wrong kernel. It cannot catch the
more embarrassing failure: both of our implementations sharing the same wrong
assumption about what a LIF neuron *is*. For that we need an implementation we did
not write.

snnTorch 1.0.0 is installed and is the reference library for surrogate-gradient
SNNs. It is used here for nothing except this cross-check -- the model imports
none of it.

Three things are checked:

  X1  Surrogate gradient. Our atan_grad(x, alpha) against snntorch.surrogate.ATan.
      NOTE: snnTorch's own docstring and its code disagree. The docstring says
      dS/dU = (1/pi) / (1 + (pi*U*alpha/2)^2); the code computes
      (alpha/2) / (1 + (pi/2*alpha*U)^2). These differ by a factor of alpha*pi/2.
      The code is what actually trains every snnTorch model ever published, so the
      code is the reference. Our spec (02a_phase2_spec.md, S5) matches the code.
      This discrepancy is recorded rather than quietly resolved.

  X2  Forward semantics, soft reset ("subtract") and hard reset ("zero"), over a
      real multi-step sequence. Membrane trajectory and spike train.

  X3  Backward semantics. Gradients of a scalar loss w.r.t. the input current,
      through the full BPTT chain. This is the one that matters: a forward that
      agrees while the backward does not is exactly the R10 failure mode.

RESULT: the mapping between our four reset modes and snnTorch's parameterisation
is not one-to-one, and the difference is not cosmetic.

  ours        snnTorch                        agreement
  ----------  ------------------------------  ----------------------------------
  hard        zero,     reset_delay=False     exact (spikes bit-identical)
  detached    zero,     reset_delay=True      exact  <-- snnTorch's DEFAULT
  none        none                            exact
  soft        subtract, reset_delay=False     exact ONLY while the membrane stays
                                              below 2*threshold (see X4)

Two consequences worth stating plainly:

  * **snnTorch's default is our `detached`, not our `hard`.** `reset_delay`
    defaults to True, which applies the reset at the start of the next step using
    a DETACHED signal. Forward trajectories are identical, so this is invisible
    unless you look at gradients -- and it changes them. Any comparison against a
    default-configured snnTorch model is a comparison against detached reset.

  * **`soft` diverges above 2*threshold, and snnTorch is the odd one out.**
    snnTorch's `_base_sub` decides the spike on `v_pre - reset_prev*thr`, carrying
    a residual subtraction from the previous step whenever the post-reset membrane
    was itself still above threshold. The textbook definition -- and ours --
    decides on `v_pre`. X4 isolates this: hold the membrane under 2*thr and the
    two agree to 2.4e-07; let it cross and they diverge.

KNOWN, DELIBERATE DIVERGENCE
  snnTorch fires on  mem >  threshold  (strict).
  This project fires on  v_pre >= threshold, matching the Phase-1 verified kernel
  in scripts/audit/05_verify_literature_claims.py. The difference is a measure-zero
  set under continuous input and cannot be reached by the random inputs used here;
  it is counted rather than assumed to be irrelevant.
"""

import json
import math
import sys

import torch

sys.path.insert(0, "src")

RESULTS = {}


# ---------------------------------------------------------------------------
# X1 -- surrogate gradient
# ---------------------------------------------------------------------------

def check_surrogate():
    from snntorch import surrogate as sur

    from snn.surrogate import atan_grad, atan_value

    out = {}
    alpha = 2.0
    x = torch.linspace(-3, 3, 4001, dtype=torch.float64, requires_grad=True)

    # snnTorch: run its autograd.Function and read the gradient it produces.
    spk = sur.atan(alpha=alpha)(x.float())
    (g_ref,) = torch.autograd.grad(spk.sum(), x)

    g_ours = atan_grad(x.detach().float(), alpha)
    out["max_abs_diff_vs_snntorch"] = float((g_ours - g_ref.float()).abs().max())
    out["our_grad_at_0"] = float(atan_grad(torch.zeros(1), alpha))
    out["snntorch_grad_at_0"] = float(g_ref[torch.argmin(x.detach().abs())])

    # Our own value/derivative pair must be internally consistent: the derivative
    # of atan_value must BE atan_grad, or the straight-through construction in the
    # eager reference silently carries the wrong gradient.
    xv = torch.linspace(-3, 3, 4001, requires_grad=True)
    v = atan_value(xv, alpha)
    (g_auto,) = torch.autograd.grad(v.sum(), xv)
    out["max_abs_diff_value_vs_derivative"] = float(
        (g_auto - atan_grad(xv.detach(), alpha)).abs().max()
    )

    # The documented-vs-implemented discrepancy inside snnTorch itself.
    docstring_form = (1 / math.pi) / (1 + (math.pi * x.detach() * alpha / 2) ** 2)
    out["snntorch_docstring_vs_code_ratio_at_0"] = float(
        g_ref[torch.argmin(x.detach().abs())] / docstring_form[torch.argmin(x.detach().abs())]
    )
    out["expected_ratio_alpha_pi_over_2"] = alpha * math.pi / 2
    return out


# ---------------------------------------------------------------------------
# X2 / X3 -- forward and backward semantics against snntorch.Leaky
# ---------------------------------------------------------------------------

def _snntorch_scan(cur, beta, thr, alpha, reset_mechanism, reset_delay=False):
    """Reference trajectory from snnTorch, driven one timestep at a time.

    cur is [B, L, d]. snnTorch's Leaky is stateless-per-call when mem is passed
    explicitly, which is how it is used here so that nothing hidden persists
    between calls.
    """
    import snntorch as snn
    from snntorch import surrogate as sur

    B, L, d = cur.shape
    cell = snn.Leaky(
        beta=beta,
        threshold=thr,
        spike_grad=sur.atan(alpha=alpha),
        reset_mechanism=reset_mechanism,
        init_hidden=False,
        learn_beta=False,
        learn_threshold=False,
        reset_delay=reset_delay,
    ).to(cur.device)

    mem = torch.zeros(B, d, device=cur.device, dtype=cur.dtype)
    spikes = []
    for t in range(L):
        spk, mem = cell(cur[:, t], mem)
        spikes.append(spk)
    return torch.stack(spikes, dim=1), mem


def check_semantics(device):
    from snn.neuron import lif_scan_eager

    out = {}
    beta, thr, alpha = 0.9, 1.0, 2.0
    B, L, d = 8, 64, 32

    # The mapping is NOT one-to-one: snnTorch's reset_delay flag selects between
    # our `hard` and our `detached` while leaving the forward trajectory
    # identical. Getting this wrong makes a correct implementation look broken.
    for ours, theirs, delay in [
        ("hard", "zero", False),
        ("detached", "zero", True),        # snnTorch's DEFAULT configuration
        ("soft", "subtract", False),
        ("none", "none", False),
    ]:
        torch.manual_seed(0)
        base = torch.randn(B, L, d, device=device) * 0.5

        cur_a = base.clone().requires_grad_(True)
        cur_b = base.clone().requires_grad_(True)

        v0 = torch.zeros(B, d, device=device)
        spk_ours, _ = lif_scan_eager(cur_a, v0, beta, thr, alpha, ours)
        spk_ref, _ = _snntorch_scan(cur_b, beta, thr, alpha, theirs, delay)

        # X2 forward
        spike_mismatch = int((spk_ours != spk_ref).sum())

        # X3 backward -- a non-trivial scalar so every timestep contributes
        w = torch.randn(B, L, d, device=device, generator=torch.Generator(
            device=device).manual_seed(1))
        (spk_ours * w).sum().backward()
        (spk_ref * w).sum().backward()
        g_ours, g_ref = cur_a.grad, cur_b.grad
        denom = g_ref.abs().max().clamp_min(1e-12)

        # The measure-zero threshold case: count it rather than assume it away, so
        # the >= vs > divergence cannot be silently absorbing a real difference.
        # Recomputed from an independent eager trajectory under no_grad.
        with torch.no_grad():
            v = torch.zeros(B, d, device=device)
            v_exact_hits = 0
            for t in range(L):
                v = v * beta + base[:, t]
                v_exact_hits += int((v == thr).sum())
                s = (v >= thr).to(v.dtype)
                if ours == "soft":
                    v = v - thr * s
                elif ours == "hard":
                    v = v * (1 - s)

        out[ours] = {
            "snntorch_reset_mechanism": theirs,
            "snntorch_reset_delay": delay,
            "snntorch_default_config": (theirs == "zero" and delay is True),
            "spike_mismatches": spike_mismatch,
            "spike_total": int(spk_ref.numel()),
            "spike_rate_ours": round(float(spk_ours.detach().mean()), 6),
            "spike_rate_snntorch": round(float(spk_ref.detach().mean()), 6),
            "grad_max_abs_diff": float((g_ours - g_ref).abs().max()),
            "grad_max_rel_diff": float(((g_ours - g_ref).abs().max() / denom)),
            "grad_norm_ours": float(g_ours.norm()),
            "grad_norm_snntorch": float(g_ref.norm()),
            "exact_threshold_hits": v_exact_hits,
        }
    return out


def check_soft_boundary(device):
    """X4 -- isolate the one real semantic divergence, at the 2*threshold boundary.

    Claim under test: our `soft` and snnTorch's `subtract` are the same neuron
    while the post-reset membrane stays below threshold, and different once it
    does not. If that is right, shrinking the input scale until the membrane can
    never reach 2*thr must make the two agree exactly -- a sharp, falsifiable
    prediction rather than a tolerance argument.
    """
    from snn.neuron import lif_scan_eager

    out = {}
    beta, thr, alpha = 0.9, 1.0, 2.0
    B, L, d = 8, 64, 32

    for scale in (0.5, 0.15):
        torch.manual_seed(0)
        base = torch.randn(B, L, d, device=device) * scale

        # How often does the POST-reset membrane remain at or above threshold?
        # That is the exact condition under which snnTorch carries a residual.
        with torch.no_grad():
            v = torch.zeros(B, d, device=device)
            residual = 0
            for t in range(L):
                v_pre = v * beta + base[:, t]
                s = (v_pre >= thr).to(v.dtype)
                v = v_pre - thr * s
                residual += int((v >= thr).sum())

        cur_a = base.clone().requires_grad_(True)
        cur_b = base.clone().requires_grad_(True)
        spk_o, _ = lif_scan_eager(cur_a, torch.zeros(B, d, device=device),
                                  beta, thr, alpha, "soft")
        spk_r, _ = _snntorch_scan(cur_b, beta, thr, alpha, "subtract", False)
        w = torch.randn(B, L, d, device=device,
                        generator=torch.Generator(device=device).manual_seed(1))
        (spk_o * w).sum().backward()
        (spk_r * w).sum().backward()

        out[f"input_sigma={scale}"] = {
            "post_reset_membrane_at_or_above_threshold": residual,
            "spike_mismatches": int((spk_o.detach() != spk_r.detach()).sum()),
            "grad_max_abs_diff": float((cur_a.grad - cur_b.grad).abs().max()),
        }
    return out


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    RESULTS["device"] = device
    RESULTS["torch"] = torch.__version__
    import snntorch
    RESULTS["snntorch"] = snntorch.__version__

    print("== X1 surrogate gradient vs snnTorch ==", flush=True)
    RESULTS["X1_surrogate"] = check_surrogate()
    print(json.dumps(RESULTS["X1_surrogate"], indent=2), flush=True)

    print("\n== X2/X3 forward + backward semantics vs snntorch.Leaky ==", flush=True)
    RESULTS["X23_semantics"] = check_semantics(device)
    print(json.dumps(RESULTS["X23_semantics"], indent=2), flush=True)

    print("\n== X4 the soft-reset divergence, isolated at the 2*thr boundary ==",
          flush=True)
    RESULTS["X4_soft_boundary"] = check_soft_boundary(device)
    print(json.dumps(RESULTS["X4_soft_boundary"], indent=2), flush=True)

    out_path = sys.argv[1] if len(sys.argv) > 1 else "neuron_cross_check.json"
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(RESULTS, fh, indent=2)
    print(f"\nWROTE {out_path}")


if __name__ == "__main__":
    main()
