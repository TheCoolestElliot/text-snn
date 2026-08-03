"""EXP_009: chase the NaN in `twocomp_distill_s1` rather than reporting around it.

`twocomp_distill_s1` trained cleanly to step 12 250 -- loss 1.5495, grad-norm
0.27, firing rates 0.389/0.396, and a checkpoint at step 10 000 whose parameters
sit inside the range the seven un-distilled seeds occupy (`|w|` max 1.48 against
their 1.73, `beta_s` max 0.9863 against their 0.9855). By step 12 500 the loss,
the grad-norm and both firing rates were NaN, and stayed NaN for the remaining
7 500 steps.

The protocol's rule is to chase an unexpected residual rather than widen a
tolerance, and "one seed diverged, cause unknown" is a tolerance widened in prose.
This script replays the run from its last healthy checkpoint and reports **which
step, and which quantity first**, which is a different claim from "it went NaN".

TWO PASSES, AND WHY
-------------------
Pass 1 replays on the **graphed** path -- the path the run actually took -- and
steps until the loss metric is non-finite. That gives the exact step at ~37
steps/s. It cannot say *what* went first, because every intermediate lives inside
the captured region.

Pass 2 replays from the same checkpoint to `bad_step - 1` on the same graphed
path, then executes that one step **eagerly and instrumented**, watching the
logits, the carried state, the two loss terms, the teacher's log-probabilities and
every parameter gradient. `Trainer._step_body`'s two dispatch paths run identical
arithmetic (its own docstring, and the R5 gate exists to keep that true), so the
step being inspected is the step that happened.

WHAT THE ANSWER SEPARATES
-------------------------
  * **forward** -- the two-compartment membrane overflows. `vs` is reset-shielded
    and integrates without bound; at tau up to 73 characters, amplified into
    `v = vf + w*vs` by `|w|` up to 1.5, a batch that drives one channel hard can
    reach fp32's ceiling. Follow-up: a membrane bound -- a change to an ADOPTED
    arm, and therefore Elliot's call, not a diagnostic's.
  * **loss / backward** -- the forward is finite and the KD term or its gradient is
    not. Follow-up touches only this experiment's own recipe.
  * **does not reproduce** -- a much more serious finding than the NaN: the
    divergence would not be a deterministic function of (checkpoint, step), and
    the project's determinism guarantee would be in question rather than the run.

WHY THE PERMANENCE IS ALREADY EXPLAINED AND IS NOT THE QUESTION
---------------------------------------------------------------
Once any gradient is non-finite, `_step_body` calls `clip_grad_norm_(...,
error_if_nonfinite=False)`, whose branchless implementation multiplies **every**
gradient by `clamp(max_norm/(total_norm + 1e-6), max=1.0)`. With `total_norm =
NaN` that factor is NaN, so every parameter is NaN after the next optimiser step
and the run cannot recover. That is read off the committed code, not measured
here, and it explains why one bad step killed 7 500 more. It does not explain the
bad step. `error_if_nonfinite=False` is deliberate and documented -- the True path
calls `bool()` on a tensor and would break capture -- and this script takes no
position on that trade.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO / "src") not in sys.path:
    sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

_distill = __import__("009_distill")
RUNS = _REPO / "experiments" / "runs"


def _bad(named) -> list[str]:
    return [n for n, t in named if t is not None and not torch.isfinite(t).all()]


def _fresh_trainer(seed: int, tag: str):
    cfg = _distill._student_config(seed)
    cfg.run_name = f"_chase_{tag}"
    cfg.eval_every = 0
    cfg.ckpt_every = 0
    cfg.log_every = 0
    return _distill.DistillTrainer(
        cfg, _distill.TEACHER_RUN.format(seed=seed), _distill.LAMBDA)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", default="twocomp_distill_s1")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--from-ckpt", default="ckpt_best.pt")
    ap.add_argument("--max-steps", type=int, default=4000)
    ap.add_argument("--inspect-steps", type=int, default=12,
                    help="instrumented eager steps from the first bad metric")
    ap.add_argument("--out", default="docs/reports/data/exp_009_divergence.json")
    args = ap.parse_args(argv)

    ckpt = RUNS / args.run / args.from_ckpt
    report: dict = {
        "experiment": "EXP_009_distillation_ceiling",
        "run": args.run, "from_checkpoint": str(ckpt.relative_to(_REPO)),
        "lambda": _distill.LAMBDA, "reproduced": False,
        "first_nonfinite_step": None, "origin": None, "trace": [],
    }

    # ---- pass 1: the graphed path, to the exact step ------------------------
    t = _fresh_trainer(args.seed, "pass1")
    t.load_checkpoint(ckpt)
    start = t.global_step
    report["resumed_at_step"] = start
    print(f"pass 1 (graphed): replaying {args.run} from step {start}", flush=True)
    bad_step = None
    for _ in range(args.max_steps):
        m = t.step()
        finite = all(map(lambda v: v == v and abs(v) != float("inf"),
                         (m["loss"], m["grad_norm"], m["objective"],
                          m["kl_to_teacher"])))
        if m["step"] % 250 == 0:
            print(f"  step {m['step']:6d}  ce {m['loss']:.4f}  "
                  f"kl {m['kl_to_teacher']:.4f}  |g| {m['grad_norm']:.4f}",
                  flush=True)
            report["trace"].append({k: m[k] for k in
                                    ("step", "loss", "grad_norm", "objective",
                                     "kl_to_teacher", "firing_rate")})
        if not finite:
            bad_step = m["step"]
            report["trace"].append({k: m[k] for k in
                                    ("step", "loss", "grad_norm", "objective",
                                     "kl_to_teacher", "firing_rate")})
            print(f"\n  first non-finite METRIC at step {bad_step}: "
                  f"ce {m['loss']} kl {m['kl_to_teacher']} |g| {m['grad_norm']}",
                  flush=True)
            break
    del t
    torch.cuda.empty_cache()

    if bad_step is None:
        report["note"] = (
            f"did not reproduce within {args.max_steps} steps of {start}; the "
            "divergence would then not be a deterministic function of "
            "(checkpoint, step), which is a larger finding than the NaN")
        print("\nDID NOT REPRODUCE. " + report["note"], flush=True)
        (_REPO / args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
        return 1

    report["reproduced"] = True
    report["first_nonfinite_step"] = bad_step
    report["steps_after_resume"] = bad_step - start

    # ---- pass 2: the same step, eagerly, instrumented -----------------------
    print(f"\npass 2: replaying to {bad_step - 1}, then one instrumented step",
          flush=True)
    t = _fresh_trainer(args.seed, "pass2")
    t.load_checkpoint(ckpt)
    while t.global_step < bad_step:
        t.step()

    # The first non-finite *metric* need not be the first non-finite *tensor*:
    # `clip_grad_norm_` accumulates the sum of squares in fp32, so a gradient of
    # ~1e20 -- perfectly representable -- squares to an overflow. So the
    # instrumented replay does not stop at `bad_step`; it steps on, eagerly and
    # fully watched, until something is genuinely non-finite or the budget runs
    # out. Each step records the fp32 norm the clip actually computes AND the
    # same norm in float64, because the gap between them is the whole question.
    model = t.model
    finite_max = (lambda x: float(x[torch.isfinite(x)].abs().max())
                  if torch.isfinite(x).any() else None)
    steps: list[dict] = []
    origin = "not reached"
    for _ in range(args.inspect_steps):
        step = t.global_step
        t._fill_inputs(step)
        t._set_lr(t._lr_at(step))
        t.opt.zero_grad(set_to_none=False)
        logits, state, _ = model(t.static_x, None)
        fwd = _bad([("logits", logits)] +
                   [(f"state_layer{i}", s) for i, s in enumerate(state)])
        flat = logits.reshape(-1, t._vocab).float()
        ce = F.cross_entropy(flat, t.static_y.reshape(-1))
        kl = F.kl_div(F.log_softmax(flat, dim=-1),
                      t.static_teacher_logp.reshape(-1, t._vocab),
                      log_target=True, reduction="batchmean")
        loss = (1.0 - t.lam) * ce + t.lam * kl
        lbad = _bad([("teacher_logp", t.static_teacher_logp), ("ce", ce),
                     ("kl", kl), ("loss", loss)])
        loss.backward()
        gbad = _bad([(n, p.grad) for n, p in model.named_parameters()])

        grads = [p.grad for p in model.parameters() if p.grad is not None]
        norm32 = float(torch.linalg.vector_norm(
            torch.stack([torch.linalg.vector_norm(g) for g in grads])))
        norm64 = float(torch.linalg.vector_norm(
            torch.stack([torch.linalg.vector_norm(g.double()) for g in grads])))
        gmax = max(float(g.abs().max()) for g in grads)
        pmax = max(float(p.abs().max()) for p in model.parameters())
        row = {
            "step": step,
            "ce": float(ce), "kl": float(kl), "loss": float(loss),
            "grad_norm_fp32": norm32, "grad_norm_fp64": norm64,
            "grad_abs_max": gmax, "param_abs_max": pmax,
            "logits_abs_max_finite": finite_max(logits),
            "state_abs_max_finite": [finite_max(s) for s in state],
            "nonfinite_forward": fwd, "nonfinite_loss": lbad,
            "nonfinite_grads": gbad,
        }
        steps.append(row)
        print(f"  step {step:6d}  ce {row['ce']:.4f}  kl {row['kl']:.4f}  "
              f"|g|max {gmax:.3e}  ||g||fp32 {norm32:.3e}  ||g||fp64 {norm64:.3e}  "
              f"|logits|max {row['logits_abs_max_finite']}", flush=True)
        if fwd or lbad or gbad:
            origin = "forward" if fwd else "loss" if lbad else "backward"
            print(f"\n  first non-finite TENSOR at step {step}: origin={origin}\n"
                  f"    forward {fwd}\n    loss {lbad}\n"
                  f"    grads ({len(gbad)}) {gbad[:8]}", flush=True)
            break

        torch.nn.utils.clip_grad_norm_(t.params, t._max_norm, norm_type=2.0,
                                       error_if_nonfinite=False, foreach=True)
        t.opt.step()
        t.global_step += 1

    report.update({
        "origin": origin,
        "fp32_max": float(torch.finfo(torch.float32).max),
        "fp32_sqrt_max": float(torch.finfo(torch.float32).max) ** 0.5,
        "instrumented_steps": steps,
        "n_parameters_with_grad": sum(1 for p in model.parameters()
                                      if p.grad is not None),
    })
    print(f"\nORIGIN: {origin}")

    (_REPO / args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"WROTE {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
