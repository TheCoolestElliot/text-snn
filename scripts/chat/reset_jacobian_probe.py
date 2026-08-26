"""Is the chat model training inside the region `EXP_016` proved unbounded?

NOT PART OF THE RESEARCH PROTOCOL. `src/snnchat/` and `scripts/chat/` sit
outside the Phase-1..5 study (`snnchat/__init__.py`), so no number this script
produces is a reported figure, and nothing here pre-registers, adopts or ranks
anything. It is a diagnostic on a checkpoint that is already on disk. It trains
nothing, writes nothing under `experiments/`, and touches no research artifact.

THE QUESTION, AND WHY IT IS WORTH ASKING HERE
----------------------------------------------
`EXP_016` (`docs/reports/04_phase4_interim.md` §15) proved that both neurons
thread an adjoint over all `L` timesteps with a per-step multiplier `beta_f*dv`:

    plain LIF   dv = (1 - s)  - v_pre * sg(v_pre - thr)     SAME variable twice
    twocomp     dv = (1 - sh) - vf    * sgd(v_pre - thr)    DIFFERENT variables

The LIF's is self-limiting and bounded by **0.5123596 < 1** at the frozen
constants, at every width. The two-compartment neuron's is bounded by nothing,
because the spike test is on the *mixed* membrane `v_pre = vf + w*vs` while the
reset lands on `vf` alone and `vs` is never reset. The crossover is at
`|w*vs| ~ 1`. `EXP_016` measured `max|beta_f*dv| = 8.96` at `d = 1481` with
`max|w*vs| = 279.66` in that layer and **696.28 in layer 1**, the largest value
in `exp_016_reset_jacobian.json` -- quote 696.28 as that artifact's maximum, not
the 279.66 the report's §15.2 table carries, which is layer 0's. `EXP_015` is the
run that died of it: NaN at step 5138, deterministically, at 5.0M parameters.

**The chat model is a two-compartment arm at 4,417,637 parameters** -- `d = 1024`,
`K = 4`, `arch = "twocomp_threshold"` -- i.e. inside the size range where the
research side's identically-shaped neuron does not train. Three facts sit
beside that and have never been connected to it:

  * `chat-v6-inst-a-s1` **diverged**: `{"event": "divergence", "step": 4859,
    "loss": NaN}` (`experiments/chat/chat-v6-inst-a-s1/stdout.log:70`), and
    `snnchat.train.Trainer._recover_from_divergence` exists to absorb exactly
    that -- roll back, bump the sampler seed, halve the learning rate;
  * every chat run since `chat-v2` trains at **lr 5e-4**, four times below the
    2e-3 `chat-v1`/`chat-v2` used, and `docs/chat/BUILD_NOTES.md` line 449
    records that this was **inherited by accident** from a fine-tune command
    line, not chosen: *"nothing tested whether 5e-4 is sensible from scratch"*;
  * `chat-v6-scratch`'s eval trace was **still falling when the cosine ran out**
    (weighted 1.2440 at 78,000 -> 1.2399 at 84,000) -- the arm was not converged.

A low learning rate is what a gradient explosion looks like from the outside
once someone has tuned around it. This script asks whether that is what happened
here, or whether the chat model sits comfortably inside the bound and its lr is
merely conservative. **Both answers are useful and the script is written not to
prefer either**: it reports the measured multiplier against the LIF's ceiling and
against 1.0, and it reports the drive `|w*vs|` that decides which side of the
crossover the arm is on.

WHAT IT DOES NOT ESTABLISH
---------------------------
An expanding multiplier is a **necessary, not sufficient** condition for the
failure -- `016_posthoc_window.py` says so in terms, and this script inherits
that caveat rather than restating it as a finding. The adjoint is injected at
every timestep, so a product over the whole unroll averages the mechanism away;
the informative statistic is the best CONTIGUOUS window (Kadane), and the
longest consecutive run of `|g| > 1`, which is what `EXP_017` H1 reported. Both
are computed here for the same reason.

Nor does it say what to do. When this script was written `snnchat` could not
express the arm that bounds this (`snn.twocomp_detach`, `EXP_017`) --
`build_chat_model` admitted `twocomp_threshold` and `twocomp` and raised on
anything else. It now also admits `twocomp_threshold_detach`, opt-in and off by
default, and `docs/chat/PREDICTION_v13.md` is the round that measures whether
that arm costs bits. **Adding the option did not make this script's numbers an
argument for using it**: `GRADIENT_BOUND_NOTE.md` §2.2 measures the two
estimators' gradients as differing by under 10 % in norm at the shipped weights,
which is evidence against the urgent reading. This script's job is to supply the
measurement, not the conclusion.

METHOD
------
The scan is re-implemented locally, exactly as `scripts/exp/016_reset_jacobian.py`
does and for the same reason: the committed scan does not expose `g` per
timestep, and a probe that computed a *different* quantity would be worthless.
The local scan is therefore checked against `snn.twocomp.twocomp_scan_eager` --
bitwise on the spikes, and on the final state -- before any statistic it emits is
believed. That check is T2 below, and it is the whole basis for trusting the row.

`threshold_gain` is applied to the current for `twocomp_threshold`, exactly as
`snn.model.TwoCompThresholdCharLM._scan` applies it, so the quantity measured is
the one the arm actually backpropagates through. `EXP_011` §1.1 established that
the learned threshold does not change the function class; it does change *where
the surrogate is evaluated*, which is precisely the argument of `sgd` here, so
dropping it would measure a different neuron.

fp64 for the accumulators, `fused = False` for the arithmetic: the same two
choices `016` makes, so the numbers are readable next to it.

    python scripts/chat/reset_jacobian_probe.py
    python scripts/chat/reset_jacobian_probe.py --run chat-v12-rare --batch-step 4859
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

import torch  # noqa: E402

from snn.prescan import threshold_gain  # noqa: E402
from snn.surrogate import atan_grad  # noqa: E402
from snn.twocomp import twocomp_scan_eager  # noqa: E402
from snnchat.data import ChatCorpus, MixtureSampler  # noqa: E402
from snnchat.model import ChatConfig, build_chat_model  # noqa: E402

CHAT = REPO / "experiments" / "chat"

#: The same ladder `016` reports against, so the two artifacts read side by side.
#: The first entry is the plain LIF's closed-form ceiling and is re-derived at
#: the run's own constants rather than hard-coded -- `EXP_016` §6 item 5 records
#: that the bound is exact only at the frozen `(alpha, thr, beta)` and must be
#: re-derived if any of them moves.
THRESHOLDS = (1.0, 2.0, 10.0, 100.0)

HIST_BINS, HIST_LO, HIST_HI = 400, -20.0, 20.0


def lif_bound(alpha: float, thr: float, beta: float) -> float:
    """`max_x |beta * ((1 - H(x)) - v_pre*atan_grad(x))|` with `v_pre = x + thr`.

    Re-derived, not inherited. This is the number the two-compartment arm is
    being asked whether it exceeds, and at the committed constants
    (`alpha = 2, thr = 1, beta = 0.5`) it is 0.5123596.
    """
    x = torch.linspace(-2000.0, 2000.0, 8_000_001, dtype=torch.float64)
    u = x * (math.pi / 2 * alpha)
    sg = (0.5 * alpha) / (1.0 + u * u)
    v_pre = x + thr
    dv = (1.0 - (x >= 0).to(torch.float64)) - v_pre * sg
    return float((beta * dv).abs().max())


class Acc:
    """Online accumulators for one layer. Nothing `[B, L, d]` is materialised.

    `best`/`run` are Kadane's maximum-subarray over `log|g_t|`, which is
    `016_posthoc_window.py`'s statistic and the one that survives the objection
    that the whole-unroll product averages the mechanism away. `run_over`/
    `longest_over` are `EXP_017` H1's: the longest CONSECUTIVE stretch of
    timesteps whose multiplier exceeds 1, which is what an explosion looks like
    when it is actually happening.
    """

    def __init__(self, batch: int, d: int, device: str) -> None:
        z = dict(device=device, dtype=torch.float64)
        self.max_abs = torch.zeros((), **z)
        self.n_total = 0
        self.counts = torch.zeros(len(THRESHOLDS), **z)
        self.count_over_bound = torch.zeros((), **z)
        self.log_prod = torch.zeros(batch, d, **z)
        self.best = torch.zeros(batch, d, **z)
        self.run = torch.zeros(batch, d, **z)
        self.run_over = torch.zeros(batch, d, **z)
        self.longest_over = torch.zeros(batch, d, **z)
        self.hist = torch.zeros(HIST_BINS, **z)
        self.max_drive = torch.zeros((), **z)
        #: `beta_f*(1 - sh)`, the DETACHED rule's own multiplier. Bounded by
        #: `beta_f` by construction, so it is a check on that construction and
        #: not a measurement -- see `scan_twocomp`'s docstring.
        self.max_abs_detached = torch.zeros((), **z)

    def add(self, g: torch.Tensor, drive: torch.Tensor, bound: float,
            g_detached: torch.Tensor | None = None) -> None:
        a = g.detach().abs().to(torch.float64)
        self.max_abs = torch.maximum(self.max_abs, a.max())
        self.n_total += a.numel()
        for i, t in enumerate(THRESHOLDS):
            self.counts[i] += (a > t).sum()
        self.count_over_bound += (a > bound).sum()

        # log(0) is -inf and would poison every chain it touches; a chain
        # containing an exact zero has a true product of exactly 0, so flooring
        # is conservative in the direction that matters.
        la = torch.log(a.clamp_min(1e-300))
        self.log_prod += la
        self.run = torch.clamp(self.run + la, min=0.0)
        self.best = torch.maximum(self.best, self.run)

        over = (a > 1.0).to(torch.float64)
        self.run_over = (self.run_over + over) * over
        self.longest_over = torch.maximum(self.longest_over, self.run_over)

        idx = (((torch.log10(a.clamp_min(1e-300)) - HIST_LO)
                / ((HIST_HI - HIST_LO) / HIST_BINS)).long().clamp_(0, HIST_BINS - 1))
        self.hist += torch.bincount(idx.reshape(-1), minlength=HIST_BINS).to(torch.float64)
        self.max_drive = torch.maximum(
            self.max_drive, drive.detach().abs().max().to(torch.float64))
        if g_detached is not None:
            self.max_abs_detached = torch.maximum(
                self.max_abs_detached,
                g_detached.detach().abs().max().to(torch.float64))

    def quantiles(self, qs: tuple[float, ...]) -> dict[str, float]:
        c = torch.cumsum(self.hist, 0) / max(float(self.hist.sum()), 1.0)
        width = (HIST_HI - HIST_LO) / HIST_BINS
        out = {}
        for q in qs:
            j = int(torch.searchsorted(c, torch.tensor(q, device=c.device, dtype=c.dtype)))
            out[f"q{q}"] = float(10.0 ** (HIST_LO + (min(j, HIST_BINS - 1) + 0.5) * width))
        return out

    def report(self, bound: float) -> dict:
        n = float(self.n_total)
        return {
            "max_abs_g": float(self.max_abs),
            "n_sites": self.n_total,
            "frac_over": {str(t): float(self.counts[i]) / n
                          for i, t in enumerate(THRESHOLDS)},
            "frac_over_lif_bound": float(self.count_over_bound) / n,
            "n_over_1": int(self.counts[THRESHOLDS.index(1.0)]),
            "quantiles_abs_g": self.quantiles((0.5, 0.9, 0.99, 0.999, 0.9999)),
            "max_chain_log10_gain": float(self.log_prod.max()) / math.log(10.0),
            "n_chains_expanding": int((self.log_prod > 0).sum()),
            "max_window_log10_gain": float(self.best.max()) / math.log(10.0),
            "n_chains_with_expanding_window": int((self.best > 0).sum()),
            "longest_consecutive_over_1": int(self.longest_over.max()),
            "n_chains": int(self.log_prod.numel()),
            "max_abs_w_times_vs": float(self.max_drive),
            "max_abs_g_detached_rule": float(self.max_abs_detached),
        }


def scan_twocomp(cur, v0, w, beta_s, beta_f, thr, alpha, acc, bound):
    """Local `twocomp_scan_eager`, emitting `g = beta_f*dv` per timestep.

    `dv` uses the eager `vf` rather than the fused kernel's reconstruction
    `v_pre - w*vs`; they are algebraically identical and the eager path is the
    one `EXP_016` reproduced the failure on.

    ON A CHECKPOINT TRAINED WITH `twocomp_threshold_detach`, THIS `g` IS A
    COUNTERFACTUAL, AND DELIBERATELY SO. The detached estimator's own per-step
    multiplier is `beta_f*(1 - sh)`, which is bounded by `beta_f = 0.5` **by
    construction** and therefore measures nothing -- it would read 0.5 on every
    checkpoint ever trained and could not distinguish two of them. What this
    reports instead is the UNDETACHED multiplier those weights would carry: the
    same yardstick applied to both arms, which is the only way "did the bounded
    estimator move the model out of the region?" has an answer.
    So read `max_abs_g` on a detached run as *"where in weight space it landed"*,
    never as *"the gradient it actually experienced"*. `beta_f*(1-sh)` is
    reported alongside as `max_abs_g_detached_rule` purely so the contrast is
    visible in the artifact rather than needing this docstring.
    """
    d = cur.shape[-1]
    vf, vs = v0[:, :d], v0[:, d:]
    spikes = []
    for t in range(cur.shape[1]):
        vf = vf * beta_f + cur[:, t]
        vs = vs * beta_s + cur[:, t]
        v_pre = vf + w * vs
        x = v_pre - thr
        sh = (x >= 0).to(v_pre.dtype)
        sgd = atan_grad(x, alpha)
        acc.add(beta_f * ((1.0 - sh) - vf * sgd), w * vs, bound,
                beta_f * (1.0 - sh))
        spikes.append(sh)
        vf = vf * (1.0 - sh)
    return torch.stack(spikes, dim=1), torch.cat([vf, vs], dim=1)


def probe(run: str, ckpt_name: str, device: str, batch_step: int) -> dict:
    run_dir = CHAT / run
    ckpt_path = run_dir / ckpt_name
    if not ckpt_path.exists():
        return {"run": run, "error": f"missing {ckpt_path}"}

    raw = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    known = {f.name for f in dataclasses.fields(ChatConfig)}
    cfg = ChatConfig(**{k: v for k, v in raw.items() if k in known})
    cfg.device = device
    cfg.fused = False        # read the same arithmetic on any machine
    # The spread is baked into the checkpoint being loaded; re-applying it at
    # build time would overwrite the trained `beta_s_raw` on the way in.
    cfg.spread_tau = False

    model = build_chat_model(cfg)
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ck["model"])
    model.eval()

    corpus = ChatCorpus(cfg.data_dir)
    sampler = MixtureSampler(
        corpus, cfg.mix, cfg.batch_size, cfg.seq_len, cfg.seed,
        align_frac=cfg.align_frac, align_lookahead=cfg.align_lookahead,
    )
    x, _ = sampler.batch(batch_step)
    x = x.to(device)

    bound = lif_bound(cfg.surrogate_alpha, cfg.threshold, cfg.beta)
    row = {
        "run": run,
        "arch": cfg.arch,
        "checkpoint": ckpt_name,
        "step_of_checkpoint": int(ck.get("global_step", -1)),
        "batch_step": batch_step,
        "d_model": cfg.d_model,
        "n_layers": cfg.n_layers,
        "params": sum(p.numel() for p in model.parameters()),
        "seq_len": cfg.seq_len,
        "batch_size": cfg.batch_size,
        "lr": cfg.lr,
        "constants": {"alpha": cfg.surrogate_alpha, "thr": cfg.threshold,
                      "beta_f": cfg.beta},
        "lif_closed_form_bound": bound,
        "layers": [],
        "t2_spikes_bitwise_equal": [],
        "t2_state_max_abs_diff": [],
    }

    with torch.no_grad():
        h = model.embed(x)
        state = model.init_state(x.shape[0], device)
        for k, linear in enumerate(model.layers):
            cur = model._project(linear, h)
            # EXACTLY what `TwoCompThresholdCharLM._scan` feeds the scan. See
            # the module docstring on why dropping it measures a different
            # neuron.
            # KEYED ON THE MODEL, NOT ON THE ARCH STRING. This read
            # `cfg.arch == "twocomp_threshold"` until 2026-08-22, which silently
            # dropped the learned gain for every OTHER arch that carries
            # `thr_log` -- `twocomp_threshold_detach` was the first, and the
            # first v13 probe reported its input gain as exactly 1.00 and 2,628
            # pinned channels because the current it scanned was ~20x too small.
            # Same defect class as the mutant that escaped this session's first
            # test pass: a path where the gain is absent is invisible whenever
            # the gain happens to be 1.
            if hasattr(model, "thr_log"):
                cur = threshold_gain(cur, model.thr_log[k])
            w, bs = model.w[k], model.slow_decay(k)
            acc = Acc(cur.shape[0], cur.shape[-1], device)
            mine, v_end = scan_twocomp(cur, state[k], w, bs, cfg.beta,
                                       cfg.threshold, cfg.surrogate_alpha,
                                       acc, bound)
            ref, v_ref = twocomp_scan_eager(cur, state[k], w, bs, cfg.beta,
                                            cfg.threshold, cfg.surrogate_alpha)
            # T2: the local scan IS the committed scan, or nothing below counts.
            row["t2_spikes_bitwise_equal"].append(bool(torch.equal(mine, ref)))
            row["t2_state_max_abs_diff"].append(float((v_end - v_ref).abs().max()))

            layer = acc.report(bound)
            layer["layer"] = k
            layer["max_abs_g_over_lif_bound"] = layer["max_abs_g"] / bound
            with torch.no_grad():
                tau = 1.0 / (1.0 - model.slow_decay(k).double()).clamp_min(1e-12)
                layer["slow_pole_tau"] = {
                    "min": round(float(tau.min()), 2),
                    "median": round(float(tau.median()), 2),
                    "max": round(float(tau.max()), 2),
                }
                layer["max_abs_w"] = float(w.abs().max())
            row["layers"].append(layer)
            h = ref
            del cur, mine, ref
            if device == "cuda":
                torch.cuda.empty_cache()

    row["t2_holds"] = (all(row["t2_spikes_bitwise_equal"])
                       and max(row["t2_state_max_abs_diff"]) == 0.0)
    row["any_layer_exceeds_lif_bound"] = any(
        ly["max_abs_g"] > bound for ly in row["layers"])
    row["any_layer_exceeds_one"] = any(ly["max_abs_g"] > 1.0 for ly in row["layers"])
    return row


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run", default="chat-v3d-aligned",
                    help="run directory under experiments/chat/ (default: the "
                         "checkpoint named by experiments/chat/SHIPPED)")
    ap.add_argument("--ckpt", default="ckpt_best.pt")
    ap.add_argument("--batch-step", type=int, default=0,
                    help="which training batch to read; `batch(s)` is a pure "
                         "function of (seed, s), so this names a real batch the "
                         "run actually saw")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default="")
    args = ap.parse_args(argv)

    row = probe(args.run, args.ckpt, args.device, args.batch_step)
    if "error" in row:
        print(f"ERROR: {row['error']}")
        return 1

    bound = row["lif_closed_form_bound"]
    print(f"{row['run']}  {row['arch']}  d={row['d_model']} K={row['n_layers']}  "
          f"{row['params']:,} params  step {row['step_of_checkpoint']}  "
          f"lr {row['lr']:g}  batch {row['batch_step']}")
    print(f"  plain-LIF closed-form ceiling: {bound:.7f}   "
          f"T2 (local scan == committed scan): {row['t2_holds']}")
    for ly in row["layers"]:
        print(f"  L{ly['layer']}  max|beta*dv| = {ly['max_abs_g']:.5f} "
              f"({ly['max_abs_g_over_lif_bound']:.2f}x the LIF ceiling)  "
              f"max|w*vs| = {ly['max_abs_w_times_vs']:.3f}")
        print(f"        sites over 1.0: {ly['n_over_1']:,}/{ly['n_sites']:,} "
              f"({ly['frac_over']['1.0']:.3e})   longest consecutive run "
              f"over 1.0: {ly['longest_consecutive_over_1']}")
        print(f"        best contiguous window: 10^{ly['max_window_log10_gain']:+.2f}   "
              f"chains with an expanding window: "
              f"{ly['n_chains_with_expanding_window']:,}/{ly['n_chains']:,}")
        print(f"        slow-pole tau: {ly['slow_pole_tau']}   "
              f"max|w| = {ly['max_abs_w']:.4f}   "
              f"detached-rule max|g| = {ly['max_abs_g_detached_rule']:.4f}")
    verdict = ("EXCEEDS 1.0 -- inside EXP_016's unbounded region"
               if row["any_layer_exceeds_one"] else
               "exceeds the LIF ceiling but not 1.0"
               if row["any_layer_exceeds_lif_bound"] else
               "within the plain LIF's own ceiling")
    print(f"  VERDICT: {verdict}")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps({
            "script": "scripts/chat/reset_jacobian_probe.py",
            "not_a_research_artifact": True,
            "question": ("is the chat model's backward-through-time inside the "
                         "region EXP_016 proved unbounded?"),
            "trains_nothing": True,
            "adopts_nothing": True,
            "result": row,
        }, indent=1), encoding="utf-8")
        print(f"  wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
