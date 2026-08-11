"""EXP_016 Leg B: where in the backward chain is the first non-finite value made?

    python scripts/exp/016_chase_grad_cur.py --out docs/reports/data/exp_016_grad_cur.json

WHAT LEG A LEFT OPEN
--------------------
Leg A measured the PREMISE on committed checkpoints and split:

    P1 HELD              max|g| = 8.96 at layer 0, d=1481 -- 17.5x the plain
                         LIF's hard ceiling of 0.5124, which no LIF leg exceeded
                         at any width.
    P3 REFUTED AS SUFF.  and decisively: 0 of 189,568 chains expand net over the
                         full 256-step unroll. The homogeneous product cannot
                         reach the observed 7.79e29.

So the per-step factor is unbounded and the whole-unroll product still contracts.
Leg A cannot say which of these is why, because it reads `ckpt_best.pt` at step
5000 and the failure is at 5138 (`EXP_016` §6 item 3):

  (i)  the mechanism is real but not sufficient, and something else overflows; or
  (ii) the state at 5138 is not the state at 5000.

WHAT THIS MEASURES
------------------
The actual failing step, on the actual dying run, instrumented at every boundary
of the backward chain rather than only at the parameters:

    loss -> head -> [spikes_1] -> scan_1 -> [cur_1] -> Linear_1
                 -> [spikes_0] -> scan_0 -> [cur_0] -> Linear_0 -> embed

`cur_k` is the output of `_project` and the input to `_scan`, so a hook on it
reads the cotangent LEAVING layer k's 256-step reverse recursion -- on the fused
path and the eager path alike, because it is the same tensor either way.
`spikes_k` is the cotangent ENTERING it. The ratio across each scan is the
amplification the recursion itself contributes, and the ratio across each Linear
is what the GEMM contributes. That decides, in one run and without editing any
file under `src/snn/`, whether the first non-finite value is BORN IN THE
RECURSION or ARRIVES from above already ruined.

`EXP_016` P6 is binary and has no bar: both outcomes are results.

DISCIPLINE
----------
G4: the hooks add no arithmetic to the trajectory. They write into a
preallocated device tensor and there is exactly one device-to-host copy, after
`backward()` returns -- no `bool(...)` inside a hook, and no data-dependent
shape (the finite max is taken with `torch.where`, not by boolean indexing).
The step-5138 loss must reproduce 1.4197630882263184 bitwise or the leg is void.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO / "src") not in sys.path:
    sys.path.insert(0, str(_REPO / "src"))

_SPEC = importlib.util.spec_from_file_location(
    "chase011", _REPO / "scripts" / "exp" / "011_chase_compose_divergence.py")
chase011 = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(chase011)

#: The loss the committed chase recorded at the failing step, on BOTH dispatch
#: paths (`exp_015_divergence.json`). G4 checks against it bitwise.
EXPECTED_LOSS_AT_5138 = 1.4197630882263184

N_FIELDS = 4      # n_nan, n_inf, max_abs_finite, n_total


def _stats_into(buf: torch.Tensor, row: int):
    """A backward hook that records into a preallocated buffer, without syncing."""
    def hook(g: torch.Tensor) -> None:
        finite = torch.isfinite(g)
        buf[row, 0] = torch.isnan(g).sum()
        buf[row, 1] = (~finite & ~torch.isnan(g)).sum()
        buf[row, 2] = torch.where(finite, g.abs(), torch.zeros_like(g)).max()
        buf[row, 3] = g.numel()
    return hook


def _read(buf: torch.Tensor, row: int, name: str) -> dict:
    n_nan, n_inf, max_finite, n_total = (float(v) for v in buf[row].tolist())
    return {
        "tensor": name,
        "n_nan": int(n_nan),
        "n_inf": int(n_inf),
        "n_nonfinite": int(n_nan + n_inf),
        "n_total": int(n_total),
        "max_abs_finite": max_finite,
        "is_nonfinite": bool(n_nan + n_inf > 0),
        "hook_fired": bool(n_total > 0),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", default="arch_twocomp_d1481_s0")
    ap.add_argument("--from-ckpt", default="ckpt_best.pt")
    ap.add_argument("--bad-step", type=int, default=5138)
    ap.add_argument("--lead-in", type=int, default=6,
                    help="ungraphed steps before the bad step, matching "
                         "011's pass 2 so the trajectory into it is the one "
                         "that produced the recorded loss")
    ap.add_argument("--eager", action="store_true",
                    help="also run the failing step on the eager scan")
    ap.add_argument("--out", default="docs/reports/data/exp_016_grad_cur.json")
    args = ap.parse_args(argv)

    ckpt = chase011.RUNS / args.run / args.from_ckpt
    report: dict = {
        "experiment": "EXP_016_reset_jacobian",
        "leg": "B",
        "chased_by": "scripts/exp/016_chase_grad_cur.py",
        "reuses": "scripts/exp/011_chase_compose_divergence.py::_fresh_trainer",
        "run": args.run,
        "from_checkpoint": str(ckpt.relative_to(_REPO)),
        "bad_step": args.bad_step,
        "expected_loss": EXPECTED_LOSS_AT_5138,
        "adopts_nothing": True,
        "changes_no_hyperparameter": True,
        "passes": [],
    }

    for eager in ([False, True] if args.eager else [False]):
        tag = "eager" if eager else "fused"
        t = chase011._fresh_trainer(args.run, f"legB_{tag}", "016")
        t.load_checkpoint(ckpt)
        model = t.model

        # `fused` is flipped AFTER the replay, not before -- exactly as 011's
        # pass 3 does it (011:280-285). Two reasons, and the first is the one
        # that matters: it makes the dispatch of the FINAL backward the single
        # variable, which is what the R10 separation means. The second is
        # measured -- replaying 132 steps on the eager path under CUDA-graph
        # capture is ~13 kernels per timestep per layer and does not finish in
        # any useful time on this card.

        # `keep_spikes` stays OFF for the replay. Retaining two [B, L, d]
        # tensors per step through 132 graphed steps costs ~388 MiB of live
        # allocation each, and measurement showed it drives this card into the
        # WDDM spill the project has hit before -- a ~50x slowdown that never
        # raises. It is switched on for the instrumented step alone, which does
        # not go through the captured graph.
        model.keep_spikes = False

        # Graphed replay to just before the bad step, exactly as 011 pass 2
        # does: the graphed path to `bad_step - lead_in`, then ungraphed.
        inspect_from = max(t.global_step, args.bad_step - args.lead_in)
        print(f"[{tag}] resumed at {t.global_step}; replaying (graphed) to "
              f"{inspect_from}", flush=True)
        while t.global_step < inspect_from:
            t.step()
        print(f"[{tag}] replayed to {t.global_step}, stepping ungraphed to "
              f"{args.bad_step}", flush=True)

        n_layers = model.n_layers
        rows = 2 * n_layers
        while t.global_step <= args.bad_step:
            step = t.global_step
            instrumented = step == args.bad_step
            buf = torch.zeros(rows, N_FIELDS, device=t.static_x.device,
                              dtype=torch.float64)
            handles: list = []

            if instrumented:
                seen = {"k": 0}
                orig = model._project

                def patched(linear, x, _orig=orig, _seen=seen, _h=handles, _b=buf):
                    out = _orig(linear, x)
                    k = _seen["k"]
                    _seen["k"] += 1
                    if out.requires_grad:
                        _h.append(out.register_hook(_stats_into(_b, n_layers + k)))
                    return out
                model._project = patched

            model.keep_spikes = instrumented
            if instrumented and eager:
                model.fused = False          # the single variable, final step only
            t._fill_inputs(step)
            t._set_lr(t._lr_at(step))
            t.opt.zero_grad(set_to_none=False)
            logits, state, aux = model(t.static_x, None)

            if instrumented:
                for k, sp in enumerate(aux["spikes"]):
                    if sp.requires_grad:
                        handles.append(sp.register_hook(_stats_into(buf, k)))

            flat = logits.reshape(-1, t._vocab).float()
            loss = F.cross_entropy(flat, t.static_y.reshape(-1))
            loss.backward()

            if instrumented:
                host = buf.cpu()                      # the ONE sync
                for h in handles:
                    h.remove()
                model._project = orig
                gbad = chase011._bad(
                    [(n, p.grad) for n, p in model.named_parameters()])
                boundaries = (
                    [_read(host, k, f"grad_spikes_layer{k}") for k in range(n_layers)]
                    + [_read(host, n_layers + k, f"grad_cur_layer{k}")
                       for k in range(n_layers)]
                )
                loss_v = float(loss.detach())
                cur0 = next(b for b in boundaries
                            if b["tensor"] == "grad_cur_layer0")
                report["passes"].append({
                    "path": tag,
                    "fused": not eager,
                    "step": step,
                    "loss": loss_v,
                    "g4_loss_reproduces_bitwise": loss_v == EXPECTED_LOSS_AT_5138,
                    "nonfinite_params": gbad,
                    "boundaries": boundaries,
                    "verdict": ("BORN IN THE RECURSION" if cur0["is_nonfinite"]
                                else "BORN IN A DOWNSTREAM REDUCTION"),
                })
                print(f"[{tag}] step {step} loss {loss_v!r} "
                      f"(bitwise={loss_v == EXPECTED_LOSS_AT_5138})", flush=True)
                for b in boundaries:
                    print(f"    {b['tensor']:24s} nan={b['n_nan']:>10d} "
                          f"inf={b['n_inf']:>10d} "
                          f"max|finite|={b['max_abs_finite']:.6e}", flush=True)
                print(f"    non-finite params: {gbad}", flush=True)
                break

            torch.nn.utils.clip_grad_norm_(model.parameters(), t._max_norm)
            t.opt.step()
            t.global_step += 1

    if report["passes"]:
        report["verdict"] = report["passes"][0]["verdict"]
        report["g4_holds"] = all(p["g4_loss_reproduces_bitwise"]
                                 for p in report["passes"])
    p = Path(args.out)
    if not p.is_absolute():
        p = _REPO / p
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    print(f"\nwrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
