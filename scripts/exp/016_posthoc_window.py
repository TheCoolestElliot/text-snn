"""EXP_016, POST-HOC: the gradient does not traverse all 256 steps, so P3 asked
the wrong question. What is the best CONTIGUOUS window's gain?

**This diagnostic is post-hoc and is labelled so everywhere it is quoted.** Its
quantity was chosen *after* reading Leg A, which is the opposite order from the
rest of this experiment. `012_posthoc_pileup.py` is the precedent -- a probe that
follows a result rather than preceding it -- and it is committed because it
bears on a bar that fired, not because it rescues one.

**P3 IS NOT AMENDED BY THIS FILE.** It resolved REFUTED AS SUFFICIENT on the
quantity it named, that verdict stands in `EXP_016` §9, and `CONTRIBUTING.md` §3
forbids repairing a pre-registered rule retroactively. This measures a
*different* quantity and reports it under its own name.

WHAT P3 GOT WRONG, AND WHY IT WAS FORESEEABLE
---------------------------------------------
P3 placed its bar on `max_(b,c) sum_t log|g_t|` -- the product over ALL 256
timesteps. But the adjoint is not injected once at t = L and carried to t = 0.
`grad_spike` enters at EVERY timestep from the layer above, so the cotangent that
reaches a parameter is a SUM over injection points, and the term injected at
timestep `t` traverses only the steps between `t` and wherever it is consumed.
A run of consecutive expanding steps anywhere in the unroll amplifies its own
term, and the 250 contracting steps elsewhere in the same chain do not undo it --
they multiply a DIFFERENT term.

Summing over the whole unroll therefore averages the mechanism away by
construction. Leg A measured exactly that: 0 of 189,568 chains expand net, while
max|g| is 8.96 and 0.06 % of sites expand.

`EXP_016` §4.0 and §6 item 4 both state that `sum_t log|g_t|` "bounds the
homogeneous gain and is NOT the gradient" -- the limitation was written down and
the bar was placed on the limited quantity anyway. That is `EXP_012` Y5's shape
and `EXP_013` N1's, and `EXP_016` §9 records it as the fourth occurrence in
Phase 4 rather than the first.

WHAT THIS MEASURES INSTEAD
--------------------------
Per `(batch, channel)` chain, the maximum over all contiguous windows of
`sum_(t in window) log|g_t|` -- Kadane's maximum-subarray, run online over the
forward sweep so nothing `[B, L, d]` is materialised. That is the largest
homogeneous amplification any single injected term can receive.

WHAT IT STILL CANNOT SAY
------------------------
It is an upper bound on the homogeneous part alone: it ignores the magnitude of
what is injected, the sign cancellation in `grad_cur = gf + gs`, and the
`grad_spike * sgd` term. A large window gain is a NECESSARY condition for the
mechanism, not a sufficient one. And this reads `ckpt_best.pt` at step 5000; the
failure is at 5138. Leg B is what observes the real step.
"""

from __future__ import annotations

import argparse
import dataclasses
import importlib.util
import json
import math
import sys
from pathlib import Path

import torch

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO / "src") not in sys.path:
    sys.path.insert(0, str(_REPO / "src"))

from snn.config import Config, seed_everything  # noqa: E402
from snn.data import Corpus, RandomWindowSampler  # noqa: E402
from snn.model import build_model  # noqa: E402
from snn.surrogate import atan_grad  # noqa: E402

_SPEC = importlib.util.spec_from_file_location(
    "rj016", _REPO / "scripts" / "exp" / "016_reset_jacobian.py")
rj = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(rj)

#: The same legs Leg A ran, so the two artifacts are read against each other.
TARGETS = rj.TARGETS


class Window:
    """Kadane's maximum-subarray, per (batch, channel), accumulated online.

    `best` is the largest contiguous sum seen so far; `run` is the best sum of a
    window ENDING at the current timestep. Both in nats, fp64.
    """

    def __init__(self, batch: int, d: int, device: str) -> None:
        z = torch.zeros(batch, d, device=device, dtype=torch.float64)
        self.run = z.clone()
        self.best = z.clone()
        self.longest = torch.zeros(batch, d, device=device, dtype=torch.float64)
        self.cur_len = torch.zeros(batch, d, device=device, dtype=torch.float64)

    def add(self, g: torch.Tensor) -> None:
        x = torch.log(g.detach().abs().to(torch.float64).clamp_min(1e-300))
        cont = self.run + x > x            # extending beats restarting
        self.run = torch.where(cont, self.run + x, x)
        self.cur_len = torch.where(cont, self.cur_len + 1.0,
                                   torch.ones_like(self.cur_len))
        self.best = torch.maximum(self.best, self.run)
        # how long a run of consecutive EXPANDING steps this chain has managed
        expanding = x > 0
        self.longest = torch.where(
            expanding, torch.maximum(self.longest, self.cur_len), self.longest)

    def report(self) -> dict:
        b = float(self.best.max())
        return {
            "max_window_log_gain": b,
            "max_window_log10_gain": b / math.log(10.0),
            "mean_window_log_gain": float(self.best.mean()),
            "n_chains_with_expanding_window": int((self.best > 0).sum()),
            "n_chains": int(self.best.numel()),
            "longest_expanding_run_steps": float(self.longest.max()),
        }


def probe(run: str, ckpt_name: str, leg: str, note: str, device: str,
          batch_step: int = 0) -> dict:
    ckpt_path = rj.RUNS / run / ckpt_name
    if not ckpt_path.exists():
        return {"leg": leg, "run": run, "error": f"missing {ckpt_name}"}

    raw = json.loads((rj.RUNS / run / "config.json").read_text(encoding="utf-8"))
    known = {f.name for f in dataclasses.fields(Config)}
    cfg = Config(**{k: v for k, v in raw.items() if k in known})
    cfg.device, cfg.fused, cfg.cuda_graph = device, False, False

    seed_everything(cfg.seed, cfg.deterministic)
    model = build_model(cfg)
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ck["model"])
    model.eval()

    corpus = Corpus(cfg.corpus, cfg.data_dir)
    x, _ = RandomWindowSampler(
        corpus.split("train"), cfg.batch_size, cfg.seq_len, cfg.seed
    ).batch(batch_step)
    x = x.to(device)

    row = {"leg": leg, "run": run, "note": note, "arch": cfg.arch,
           "d_model": cfg.d_model, "checkpoint": ckpt_name,
           "batch_step": batch_step,
           "step_of_checkpoint": int(ck.get("global_step", -1)), "layers": []}

    with torch.no_grad():
        h = model.embed(x)
        state = model.init_state(x.shape[0], device)
        for k, linear in enumerate(model.layers):
            cur = model._project(linear, h)
            d = cur.shape[-1]
            win = Window(cur.shape[0], d, device)
            beta_f, thr, alpha = cfg.beta, cfg.threshold, cfg.surrogate_alpha

            if cfg.arch == "twocomp":
                w, bs = model.w[k], model.slow_decay(k)
                vf, vs = state[k][:, :d], state[k][:, d:]
                spikes = []
                for t in range(cur.shape[1]):
                    vf = vf * beta_f + cur[:, t]
                    vs = vs * bs + cur[:, t]
                    v_pre = vf + w * vs
                    xg = v_pre - thr
                    sh = (xg >= 0).to(v_pre.dtype)
                    win.add(beta_f * ((1.0 - sh) - vf * atan_grad(xg, alpha)))
                    spikes.append(sh)
                    vf = vf * (1.0 - sh)
                emitted = torch.stack(spikes, dim=1)
            elif cfg.arch == "snn":
                v = state[k]
                spikes = []
                for t in range(cur.shape[1]):
                    v_pre = v * beta_f + cur[:, t]
                    xg = v_pre - thr
                    s = (xg >= 0).to(v_pre.dtype)
                    win.add(beta_f * ((1.0 - s) - v_pre * atan_grad(xg, alpha)))
                    spikes.append(s)
                    v = v_pre * (1.0 - s)
                emitted = torch.stack(spikes, dim=1)
            else:
                return {"leg": leg, "run": run, "error": f"arch {cfg.arch!r}"}

            rep = win.report()
            rep["layer"] = k
            row["layers"].append(rep)
            h = emitted
            del cur
            if device == "cuda":
                torch.cuda.empty_cache()
    return row


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="docs/reports/data/exp_016_posthoc_window.json")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--batch-step", type=int, default=0,
                    help="sampler batch index; 5138 is the batch the dying run "
                         "actually saw at the step it died on")
    args = ap.parse_args(argv)

    rows = []
    for run, ck, leg, note in TARGETS:
        print(f"[{leg}] {run} ...", flush=True)
        r = probe(run, ck, leg, note, args.device, args.batch_step)
        rows.append(r)
        if "error" in r:
            print(f"  {r['error']}", flush=True)
            continue
        for ly in r["layers"]:
            print(f"  L{ly['layer']}  max window log10 gain="
                  f"{ly['max_window_log10_gain']:+.2f}"
                  f"  longest expanding run={ly['longest_expanding_run_steps']:.0f}"
                  f"  chains with an expanding window="
                  f"{ly['n_chains_with_expanding_window']}/{ly['n_chains']}",
                  flush=True)

    out = {
        "experiment": "EXP_016_reset_jacobian",
        "post_hoc": True,
        "batch_step": args.batch_step,
        "amends_no_bar": True,
        "p3_stands_as_it_fired": True,
        "question": ("what is the largest gain any single CONTIGUOUS window of "
                     "the reverse recursion can contribute?"),
        "inference_only": True,
        "trains_nothing": True,
        "adopts_nothing": True,
        "changes_no_hyperparameter": True,
        "necessary_not_sufficient": True,
        "results": rows,
    }
    p = Path(args.out)
    if not p.is_absolute():
        p = _REPO / p
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    print(f"\nwrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
