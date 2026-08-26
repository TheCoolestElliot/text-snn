"""What did the chat model do with the timescales it was given?

NOT PART OF THE RESEARCH PROTOCOL. `src/snnchat/` and `scripts/chat/` sit outside
the Phase-1..5 study, so nothing here is pre-registered and no number it produces
is a reported figure. It trains nothing and writes nothing under `experiments/`.

THE QUESTION
------------
`snnchat.model.spread_slow_poles` is the chat model's one deliberate departure
from a committed research initialisation: instead of giving every channel the
same `beta_s = 0.95` (`tau = 20`), it lays `tau` out log-uniformly from
`tau_min = 3` to `tau_max = 600`, so that some channels integrate over a word and
others over a whole turn from step zero. Its own docstring calls this
**"a bet, not a result"** and says plainly that nothing in the package should be
read as evidence that it works.

`scripts/chat/horizon.py` measures what the trained model got in *bits*. This
measures what it got in *channels*, which is a different question and the one the
bet was actually about. Three things are asked of every channel:

  1.  **What timescale did it end up at, against the one it started at?** A
      channel that was handed `tau = 400` and trained down to `tau = 2` has had
      its long timescale taken away, and the model has that much less machinery
      for holding a turn. `docs/chat/WEIGHT_DECAY_NOTE.md` gives one mechanism
      for that drift and is the reason this is worth counting rather than
      assuming.

  2.  **Is the channel still doing anything?** A spiking channel that fires on
      ~0 % or ~100 % of timesteps carries no information about its input,
      whatever its decay constant says. The two-compartment membrane is
      `v = vf + w*vs`, and `vs` is **never reset** -- so a channel whose `w*vs`
      has drifted far from the threshold is pinned, and its `tau` is then a
      description of a dead parameter rather than of a live memory.

  3.  **Where is the drive coming from?** `EXP_016` showed that `|w*vs|` is the
      quantity that decides whether the backward-through-time is bounded, and
      `scripts/chat/reset_jacobian_probe.py` measures it at the extremes. This
      script reports its *distribution* against `tau`, so the question "is it the
      long-timescale channels that are unstable?" gets an answer rather than an
      argument.

`twocomp_threshold` composes `EXP_008`'s learned per-channel threshold onto the
adopted neuron, which is exactly a learned per-channel gain on the current
(`EXP_008` Identity 1). That gain multiplies what `vs` integrates, so it is
reported here too: a channel whose input gain has trained to 20x is feeding its
un-reset slow compartment twenty times what the initialisation calibrated for.

RELATION TO `regional_profile.py`, WHICH CAME FIRST
----------------------------------------------------
`scripts/chat/regional_profile.py` reads `beta_s_raw` and `w` straight off a
`state_dict` -- zero training, zero GPU, no forward pass -- and answers *where*
the slow channels are and whether they carry weight into the membrane. It is the
right tool for that and this does not replace it; `REGIONAL_NOTE.md` is its
write-up and its 26-checkpoint census is the standing result.

This script exists for the three quantities that a forward pass is the only way
to get: the **realised firing rate** (a channel's `tau` and `|w|` say nothing
about whether it is pinned), what the learned threshold does to `mean|cur|`, and
`|w*vs|` itself -- which is the value, not the parameter, and is the quantity
`EXP_016`'s bound is stated in. It costs a GPU-minute where `regional_profile`
costs none, so prefer that one unless a realised value is what is wanted.

`REGIONAL_NOTE.md`'s finding that the checkpoints span **two independently
trained lineages** (`chat-v2` and `chat-v6-scratch`) is the reason this script's
results are read as n = 2 and not as n = 6.

WHAT THIS IS NOT
----------------
It is a census, not a comparison. There is no anchor arm, no seed, and no bar; it
describes one checkpoint. `docs/chat/CONVENTIONS.md`'s rule that a rate carries
an interval and a denominator applies to anything read *out* of it -- the
denominators are printed for that reason. Nothing here proposes a change to
`spread_slow_poles`, to `tau_max`, or to any other value.

    python scripts/chat/slow_channel_census.py
    python scripts/chat/slow_channel_census.py --run chat-v6-scratch --out census.json
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

import torch  # noqa: E402

from snn.prescan import threshold_gain  # noqa: E402
from snn.twocomp import twocomp_scan_eager  # noqa: E402
from snnchat.data import ChatCorpus, MixtureSampler  # noqa: E402
from snnchat.model import ChatConfig, build_chat_model  # noqa: E402

CHAT = REPO / "experiments" / "chat"

#: Channel is called PINNED if it fires on fewer than this fraction of timesteps
#: or on more than (1 - this). A pinned channel emits (almost) the same bit at
#: every position, so it carries (almost) no information about its input --
#: which is the property being counted, and the reason the cut is on the
#: realised firing rate rather than on the membrane.
PINNED = 0.01

#: Buckets of `tau = 1/(1 - beta_s)`, in characters. The edges are the
#: interesting scales for text and are fixed here rather than derived from the
#: data: roughly a character, a word, a clause, a sentence, a turn, longer.
TAU_EDGES = (0.0, 2.0, 8.0, 32.0, 128.0, 512.0, float("inf"))
TAU_NAMES = ("<2", "2-8", "8-32", "32-128", "128-512", ">512")


def _q(t: torch.Tensor, qs=(0.01, 0.25, 0.5, 0.75, 0.99)) -> dict[str, float]:
    t = t.flatten().double()
    return {f"q{q}": round(float(torch.quantile(t, q)), 6) for q in qs}


def census(run: str, ckpt_name: str, device: str, batch_step: int) -> dict:
    run_dir = CHAT / run
    ckpt_path = run_dir / ckpt_name
    if not ckpt_path.exists():
        return {"run": run, "error": f"missing {ckpt_path}"}

    raw = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    known = {f.name for f in dataclasses.fields(ChatConfig)}
    cfg = ChatConfig(**{k: v for k, v in raw.items() if k in known})
    cfg.device = device
    cfg.fused = False
    cfg.spread_tau = False        # the spread is baked into the checkpoint

    model = build_chat_model(cfg)
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ck["model"])
    model.eval()

    corpus = ChatCorpus(cfg.data_dir)
    x, _ = MixtureSampler(
        corpus, cfg.mix, cfg.batch_size, cfg.seq_len, cfg.seed,
        align_frac=cfg.align_frac, align_lookahead=cfg.align_lookahead,
    ).batch(batch_step)
    x = x.to(device)

    row = {
        "run": run, "arch": cfg.arch, "checkpoint": ckpt_name,
        "step_of_checkpoint": int(ck.get("global_step", -1)),
        "d_model": cfg.d_model, "n_layers": cfg.n_layers,
        "seq_len": cfg.seq_len, "batch_size": cfg.batch_size,
        "tau_init_requested": (
            [cfg.tau_min, cfg.tau_max] if raw.get("spread_tau") else None),
        "beta_slow_init": cfg.beta_slow,
        "pinned_cut": PINNED,
        "layers": [],
    }

    with torch.no_grad():
        h = model.embed(x)
        state = model.init_state(x.shape[0], device)
        for k, linear in enumerate(model.layers):
            cur_raw = model._project(linear, h)
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
                gain = model.input_gain(k).flatten()          # exp(-thr_log)
                cur = threshold_gain(cur_raw, model.thr_log[k])
            else:
                gain = torch.ones(cur_raw.shape[-1], device=device)
                cur = cur_raw

            w = model.w[k].flatten()
            beta_s = model.slow_decay(k).flatten()
            tau = 1.0 / (1.0 - beta_s.double()).clamp_min(1e-12)

            # Replay the committed eager scan to get the spikes, then a second
            # pass for `vs` alone. `vs` is a pure linear filter of `cur` and does
            # not depend on the spikes at all -- it is never reset -- so it can
            # be recomputed without re-deriving anything the scan owns.
            spikes, _ = twocomp_scan_eager(
                cur, state[k], model.w[k], model.slow_decay(k), cfg.beta,
                cfg.threshold, cfg.surrogate_alpha)

            vs = torch.zeros_like(cur[:, 0])
            abs_vs_sum = torch.zeros(cur.shape[-1], device=device, dtype=torch.float64)
            max_abs_vs = torch.zeros(cur.shape[-1], device=device, dtype=torch.float64)
            for t in range(cur.shape[1]):
                vs = vs * model.slow_decay(k) + cur[:, t]
                a = vs.abs().double()
                abs_vs_sum += a.mean(0)
                max_abs_vs = torch.maximum(max_abs_vs, a.max(0).values)
            mean_abs_vs = abs_vs_sum / cur.shape[1]

            rate = spikes.mean(dim=(0, 1)).double()           # [d]
            pinned = (rate < PINNED) | (rate > 1.0 - PINNED)
            drive = (w.double().abs() * mean_abs_vs)          # typical |w*vs|

            buckets = []
            for i in range(len(TAU_NAMES)):
                m = (tau >= TAU_EDGES[i]) & (tau < TAU_EDGES[i + 1])
                n = int(m.sum())
                buckets.append({
                    "tau": TAU_NAMES[i],
                    "n_channels": n,
                    "frac_of_layer": round(n / tau.numel(), 4),
                    "n_pinned": int((pinned & m).sum()),
                    "frac_pinned": round(float((pinned & m).float().mean()), 4) if n else None,
                    "median_firing_rate": round(float(rate[m].median()), 5) if n else None,
                    "median_abs_w": round(float(w[m].abs().median()), 4) if n else None,
                    "median_input_gain": round(float(gain[m].median()), 4) if n else None,
                    "median_typical_drive": round(float(drive[m].median()), 4) if n else None,
                    "max_typical_drive": round(float(drive[m].max()), 2) if n else None,
                })

            row["layers"].append({
                "layer": k,
                "d": int(tau.numel()),
                "tau": _q(tau),
                "tau_max": round(float(tau.max()), 1),
                "firing_rate": _q(rate),
                "n_pinned": int(pinned.sum()),
                "frac_pinned": round(float(pinned.float().mean()), 4),
                "abs_w": _q(w.abs()),
                "input_gain_exp_minus_thr_log": _q(gain),
                "mean_abs_cur_pre_gain": round(float(cur_raw.abs().mean()), 4),
                "mean_abs_cur_post_gain": round(float(cur.abs().mean()), 4),
                "typical_abs_w_vs": _q(drive),
                "max_abs_vs": round(float(max_abs_vs.max()), 2),
                "by_tau": buckets,
            })
            h = spikes
            del cur, cur_raw, spikes
            if device == "cuda":
                torch.cuda.empty_cache()
    return row


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run", default="chat-v3d-aligned")
    ap.add_argument("--ckpt", default="ckpt_best.pt")
    ap.add_argument("--batch-step", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default="")
    args = ap.parse_args(argv)

    row = census(args.run, args.ckpt, args.device, args.batch_step)
    if "error" in row:
        print(f"ERROR: {row['error']}")
        return 1

    req = row["tau_init_requested"]
    print(f"{row['run']}  {row['arch']}  d={row['d_model']} K={row['n_layers']}  "
          f"step {row['step_of_checkpoint']}  L={row['seq_len']}  B={row['batch_size']}")
    print(f"  tau requested at init: {req}   (pinned = fires <{PINNED:.0%} or "
          f">{1 - PINNED:.0%} of timesteps)")
    for ly in row["layers"]:
        print(f"\n  L{ly['layer']}  tau q01/q50/q99 = {ly['tau']['q0.01']:.2f} / "
              f"{ly['tau']['q0.5']:.2f} / {ly['tau']['q0.99']:.1f}   max {ly['tau_max']}")
        print(f"        pinned channels: {ly['n_pinned']}/{ly['d']} "
              f"({ly['frac_pinned']:.1%})   firing rate q50 = "
              f"{ly['firing_rate']['q0.5']:.4f}")
        print(f"        input gain exp(-thr_log) q01/q50/q99 = "
              f"{ly['input_gain_exp_minus_thr_log']['q0.01']:.3f} / "
              f"{ly['input_gain_exp_minus_thr_log']['q0.5']:.3f} / "
              f"{ly['input_gain_exp_minus_thr_log']['q0.99']:.3f}   "
              f"mean|cur| {ly['mean_abs_cur_pre_gain']:.3f} -> "
              f"{ly['mean_abs_cur_post_gain']:.3f}")
        print(f"        typical |w*vs| q50 = {ly['typical_abs_w_vs']['q0.5']:.3f}, "
              f"q99 = {ly['typical_abs_w_vs']['q0.99']:.2f}   max|vs| = {ly['max_abs_vs']}")
        print(f"        {'tau':>8} {'chans':>6} {'pinned':>8} {'rate':>8} "
              f"{'|w|':>7} {'gain':>7} {'|w*vs|':>9}")
        for b in ly["by_tau"]:
            if not b["n_channels"]:
                continue
            print(f"        {b['tau']:>8} {b['n_channels']:>6} "
                  f"{b['frac_pinned']:>7.1%} {b['median_firing_rate']:>8.4f} "
                  f"{b['median_abs_w']:>7.3f} {b['median_input_gain']:>7.3f} "
                  f"{b['median_typical_drive']:>9.3f}")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps({
            "script": "scripts/chat/slow_channel_census.py",
            "not_a_research_artifact": True,
            "question": ("what did the chat model do with the timescales "
                         "spread_slow_poles gave it?"),
            "trains_nothing": True,
            "adopts_nothing": True,
            "result": row,
        }, indent=1), encoding="utf-8")
        print(f"\n  wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
