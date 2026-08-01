"""Phase-2 discrepancy: with PyTorch's default init, every layer past the first is silent.

Found while running the Phase-2 gates. At the frozen baseline config (d=512, K=2,
threshold 1.0, beta 0.5, default `nn.Linear` init) the measured per-layer firing
rates at initialisation are:

    [0.0489, 0.0]        seed 0
    [0.0484, 0.0]        seed 1
    [0.0482, 0.0]        seed 2

Layer 1 never fires. The head therefore receives an identically-zero input, the
logits are the head bias broadcast, and `head.weight.grad` is exactly zero. The
same holds at K=3 and K=4: only the first layer is ever alive.

WHY -- and this is arithmetic, not a guess.

`nn.Linear`'s default init draws w ~ U(-1/sqrt(d), 1/sqrt(d)), so Var(w) = 1/(3d).
The pre-activation variance for an input x is

    Var(Wx) = d * Var(w) * E[x^2] = E[x^2] / 3

Layer 0's input is the embedding: dense, N(0,1), so E[x^2] = 1 and sigma = 0.577.
Against a threshold of 1.0 that is 1.73 sigma, giving a firing rate of about 4.2%
-- which is what is measured.

Layer 1's input is a BINARY SPIKE VECTOR at rate p. For s in {0,1}, s^2 = s, so
E[s^2] = p, not 1. With p = 0.049 that is

    sigma = sqrt(0.049/3) = 0.128     ->   threshold is 7.8 sigma away

The default init is calibrated for dense unit-variance activations. A spike train
at 5% density carries twenty times less energy, and the mismatch compounds with
depth. This is a property of the initialisation, not of the architecture.

WHAT THIS SCRIPT ESTABLISHES

  P1  The pathology, measured across seeds, depths, widths and thresholds, with
      the predicted sigma compared against the observed one.
  P2  Whether training ESCAPES it. The surrogate gradient has heavy tails, so a
      silent layer is not necessarily a dead one. This is the question that
      decides whether the default init is fatal or merely a bad start, and it is
      answered by running it rather than by reasoning about it.
  P3  A principled fix and its effect: rescale each spiking layer so the membrane's
      steady-state standard deviation equals the threshold. Derivation in
      `variance_scaled_init` below.

The Phase-2 report's discrepancy section is written from this script's JSON.
"""

import json
import math
import sys

import torch
import torch.nn as nn

sys.path.insert(0, "src")

RESULTS = {}


# ---------------------------------------------------------------------------
# The fix
# ---------------------------------------------------------------------------

def variance_scaled_init(model, cfg, sample_idx, target_sigma_ratio=1.0):
    """Rescale each spiking layer so its membrane sigma matches the threshold.

    Derivation. For the reset-free leaky recurrence v_t = beta*v_{t-1} + i_t with
    i_t iid of variance s_i^2, the steady-state membrane variance is

        Var(v) = s_i^2 / (1 - beta^2)

    so aiming the MEMBRANE at the threshold means aiming the current at
    s_i = thr * sqrt(1 - beta^2). Reset only removes energy, so this is an upper
    bound on what the layer will actually see -- deliberately, because
    over-shooting into saturation is easier to diagnose than silence.

    Rather than assume a firing rate p and solve the resulting fixed point, the
    scale is MEASURED layer by layer on a real batch: run the layer, look at the
    current it actually produced, and rescale to hit the target. That is the LSUV
    idea (Mishkin & Matas, 2015) specialised to a spiking stack, and it needs no
    assumption about p at all -- which matters, because p is exactly the quantity
    the default init gets wrong.

    Deterministic given the seed and the batch. Returns the per-layer scale factors.
    """
    from snn.neuron import lif_scan

    thr, beta = cfg.threshold, cfg.beta
    target = thr * math.sqrt(max(1.0 - beta * beta, 1e-6)) * target_sigma_ratio

    scales = []
    with torch.no_grad():
        h = model.embed(sample_idx)
        for linear in model.layers:
            cur = linear(h)
            sigma = float(cur.std())
            scale = target / max(sigma, 1e-12)
            linear.weight.mul_(scale)
            linear.bias.mul_(scale)
            scales.append(scale)
            cur = linear(h)
            v0 = torch.zeros(cur.shape[0], cur.shape[2], device=cur.device)
            h, _ = lif_scan(cur, v0, beta, thr, cfg.surrogate_alpha,
                            cfg.reset, fused=cfg.fused)
    return scales


# ---------------------------------------------------------------------------
# P1 -- the pathology
# ---------------------------------------------------------------------------

def measure_pathology():
    from snn.config import Config, seed_everything
    from snn.model import build_model

    out = {}
    for d, K, thr in [(512, 2, 1.0), (512, 4, 1.0), (256, 2, 1.0),
                      (1024, 2, 1.0), (512, 2, 0.5), (512, 2, 0.25)]:
        key = f"d={d},K={K},thr={thr}"
        rates_by_seed = []
        for seed in (0, 1, 2):
            seed_everything(seed, True)
            cfg = Config(vocab_size=205, d_model=d, n_layers=K, arch="snn",
                         beta=0.5, threshold=thr, reset="hard",
                         device="cuda", fused=True)
            model = build_model(cfg).cuda()
            idx = torch.randint(0, 205, (32, 128), device="cuda")
            with torch.no_grad():
                _, _, aux = model(idx)
            rates_by_seed.append([round(float(r), 6) for r in aux["firing_rate"]])
        out[key] = {
            "firing_rate_by_seed": rates_by_seed,
            "silent_layers": sum(1 for r in rates_by_seed[0][1:] if r == 0.0),
        }
        print(f"  {key:26s} -> {rates_by_seed[0]}", flush=True)

    # predicted vs observed sigma, which is the whole causal claim
    from snn.config import Config as C, seed_everything as se
    se(0, True)
    cfg = C(vocab_size=205, d_model=512, n_layers=2, arch="snn", beta=0.5,
            threshold=1.0, reset="hard", device="cuda", fused=True)
    model = build_model(cfg).cuda()
    idx = torch.randint(0, 205, (32, 128), device="cuda")
    from snn.neuron import lif_scan
    with torch.no_grad():
        h = model.embed(idx)
        sigma_check = []
        for k, linear in enumerate(model.layers):
            cur = linear(h)
            e_x2 = float((h ** 2).mean())
            predicted = math.sqrt(e_x2 / 3.0)
            sigma_check.append({
                "layer": k,
                "input_E[x^2]": round(e_x2, 6),
                "predicted_current_sigma": round(predicted, 6),
                "observed_current_sigma": round(float(cur.std()), 6),
                "threshold_in_sigmas": round(1.0 / max(float(cur.std()), 1e-12), 3),
            })
            v0 = torch.zeros(cur.shape[0], cur.shape[2], device=cur.device)
            h, _ = lif_scan(cur, v0, 0.5, 1.0, 2.0, "hard", fused=True)
    out["sigma_analysis"] = sigma_check
    return out


# ---------------------------------------------------------------------------
# P2 -- does training escape it?
# ---------------------------------------------------------------------------

def measure_escape(steps=400):
    """Train the real model briefly and watch layer 1's firing rate.

    The surrogate's tails are heavy, so gradient does reach a silent layer. The
    question is whether it is enough to revive it within a budget anyone would
    actually run. Uses random targets deliberately: this measures the OPTIMISER's
    ability to break the symmetry, not the corpus.
    """
    from snn.config import Config, seed_everything
    from snn.model import build_model

    out = {}
    for label, use_fix in [("default_init", False), ("variance_scaled", True)]:
        seed_everything(0, True)
        cfg = Config(vocab_size=205, d_model=512, n_layers=2, arch="snn",
                     beta=0.5, threshold=1.0, reset="hard", device="cuda",
                     fused=True)
        model = build_model(cfg).cuda()
        idx = torch.randint(0, 205, (32, 128), device="cuda")
        scales = None
        if use_fix:
            scales = variance_scaled_init(model, cfg, idx)

        opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.1)
        x = torch.randint(0, 205, (32, 129), device="cuda")
        inp, tgt = x[:, :-1], x[:, 1:]

        traj = []
        for step in range(steps + 1):
            logits, _, aux = model(inp)
            loss = nn.functional.cross_entropy(
                logits.reshape(-1, logits.shape[-1]), tgt.reshape(-1))
            if step % 50 == 0:
                traj.append({
                    "step": step,
                    "loss": round(float(loss), 5),
                    "rates": [round(float(r), 6) for r in aux["firing_rate"]],
                    "head_grad_norm": (round(float(model.head.weight.grad.norm()), 8)
                                       if model.head.weight.grad is not None else None),
                })
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

        final = traj[-1]
        out[label] = {
            "init_scales": [round(s, 5) for s in scales] if scales else None,
            "trajectory": traj,
            "layer1_alive_at_end": final["rates"][-1] > 1e-6,
            "steps_to_first_spike_in_layer1": next(
                (t["step"] for t in traj if t["rates"][-1] > 1e-6), None),
            "ln_vocab": round(math.log(205), 5),
        }
        print(f"  {label:16s} final loss {final['loss']:.4f}  rates {final['rates']}",
              flush=True)
    return out


def main():
    assert torch.cuda.is_available()
    RESULTS["torch"] = torch.__version__

    print("== P1: the pathology across seeds, depths, widths, thresholds ==", flush=True)
    RESULTS["P1_pathology"] = measure_pathology()

    print("\n== P2/P3: does training escape it, and does the fix work? ==", flush=True)
    RESULTS["P23_escape_and_fix"] = measure_escape()

    out_path = sys.argv[1] if len(sys.argv) > 1 else "init_pathology.json"
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(RESULTS, fh, indent=2)
    print(f"\nWROTE {out_path}")


if __name__ == "__main__":
    main()
