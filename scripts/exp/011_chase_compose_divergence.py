"""EXP_011: chase the NaN in `compose_s0` rather than reporting around it.

`compose_s0` trained cleanly to step 17 500 -- loss 1.3954, grad-norm 0.2256,
firing rates 0.380/0.385, and a `ckpt_best` at that step whose every tensor is
finite. By step 17 750 the loss, the grad-norm and both firing rates were NaN,
and stayed NaN for the remaining 2 250 steps. This is the **second** seed in
Phase 4 to die this way; `EXP_009`'s was the first.

The protocol's rule is to chase an unexpected residual rather than widen a
tolerance, and "one seed diverged, cause unknown" is a tolerance widened in
prose. This script replays the run from its last healthy checkpoint and reports
**which step, and which quantity first**, which is a different claim from "it
went NaN".

WHY THIS IS NOT `009_chase_divergence.py` WITH A FLAG
------------------------------------------------------
That script builds a `DistillTrainer` -- it imports `009_distill`, loads a
teacher checkpoint and watches a KL term against the teacher's
log-probabilities. None of that exists here: the composed arm trains under the
plain `snn.train.Trainer` on cross-entropy alone. Bending 009's script to run
both arms would put a teacher-shaped branch through a chase that has no teacher,
and the instrumented step is the one place in this project where an untaken
branch is most likely to be wrong. The **two-pass structure and every
instrument** are 009's, deliberately, so the two chases produce comparable
answers; the trainer and the loss are this arm's.

THE QUESTION THIS SEPARATES
----------------------------
`EXP_009` §9.4 found its divergence was NOT an exploding forward. The forward and
the loss were finite, every gradient was finite, the largest was 5.46e31 -- and
`clip_grad_norm_`'s **fp32 sum of squares overflowed**, so the clip computed an
infinite norm, produced a zero scale factor, and destroyed the update. That is a
*numerical* bug in a guard rail, not a divergent model.

So the decisive instrument is the gradient norm computed **twice, in fp32 and in
float64, on the same gradients**. If fp32 is `inf` while fp64 is finite, this is
EXP_009's mechanism a second time -- and the second occurrence is on the
**adopted** arm's backward, which raises what decision #7 is worth. If both are
finite, or both infinite, it is a different mechanism and the difference is the
finding.

WHAT THIS SCRIPT DOES NOT DO
----------------------------
It does not apply a fix. The fixes were named in `EXP_009` §9.4 and deliberately
not applied, because they touch committed Phase-2 code and applying them would
flatter this work; that is decision #7 and it is Elliot's. It does not replace
the dead seed either -- `EXP_011` reports n=2 with every verdict provisional,
for `EXP_009`'s reason: swapping a diverged seed for a fresh one is how a
failure leaves the record.
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
from snn.train import Trainer  # noqa: E402

RUNS = _REPO / "experiments" / "runs"


def _bad(named) -> list[str]:
    return [n for n, t in named if t is not None and not torch.isfinite(t).all()]


def _fresh_trainer(run: str, tag: str) -> Trainer:
    """A Trainer on the dead run's own config, writing nothing that matters.

    The run name is `_chase011_*` rather than `_chase_*` so this cannot overwrite
    `EXP_009`'s committed chase artifacts, which are still the evidence for its
    §9.4.
    """
    raw = json.loads((RUNS / run / "config.json").read_text(encoding="utf-8"))
    known = {f.name for f in dataclasses.fields(Config)}
    cfg = Config(**{k: v for k, v in raw.items() if k in known})
    cfg.run_name = f"_chase011_{tag}"
    cfg.eval_every = 0
    cfg.ckpt_every = 0
    cfg.log_every = 0
    return Trainer(cfg)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", default="compose_s0")
    ap.add_argument("--from-ckpt", default="ckpt_best.pt")
    ap.add_argument("--max-steps", type=int, default=4000)
    ap.add_argument("--inspect-steps", type=int, default=12,
                    help="instrumented eager steps from the first bad metric")
    ap.add_argument("--lead-in", type=int, default=6,
                    help="instrumented steps to run BEFORE the first bad metric. "
                         "EXP_009's divergence had two events -- a finite but "
                         "enormous gradient, then a NaN one step later -- and a "
                         "chase that starts at the bad step can only see the "
                         "second. This is what makes the two comparable.")
    ap.add_argument("--out", default="docs/reports/data/exp_011_divergence.json")
    args = ap.parse_args(argv)

    ckpt = RUNS / args.run / args.from_ckpt
    report: dict = {
        "experiment": "EXP_011_composition",
        "run": args.run,
        "from_checkpoint": str(ckpt.relative_to(_REPO)),
        "compares_with": "docs/reports/data/exp_009_divergence.json",
        "reproduced": False,
        "first_nonfinite_step": None,
        "origin": None,
        "trace": [],
    }

    # ---- pass 1: the graphed path, to the exact step ------------------------
    t = _fresh_trainer(args.run, "pass1")
    t.load_checkpoint(ckpt)
    start = t.global_step
    report["resumed_at_step"] = start
    print(f"pass 1 (graphed): replaying {args.run} from step {start}", flush=True)
    bad_step = None
    for _ in range(args.max_steps):
        m = t.step()
        finite = all(v == v and abs(v) != float("inf")
                     for v in (m["loss"], m["grad_norm"]))
        if m["step"] % 50 == 0:
            print(f"  step {m['step']:6d}  loss {m['loss']:.4f}  "
                  f"|g| {m['grad_norm']:.4f}", flush=True)
            report["trace"].append(
                {k: m[k] for k in ("step", "loss", "grad_norm", "firing_rate")})
        if not finite:
            bad_step = m["step"]
            report["trace"].append(
                {k: m[k] for k in ("step", "loss", "grad_norm", "firing_rate")})
            print(f"\n  first non-finite METRIC at step {bad_step}: "
                  f"loss {m['loss']} |g| {m['grad_norm']}", flush=True)
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
    inspect_from = max(start, bad_step - args.lead_in)
    print(f"\npass 2: replaying to {inspect_from}, then instrumented steps "
          f"through {bad_step}", flush=True)
    t = _fresh_trainer(args.run, "pass2")
    t.load_checkpoint(ckpt)
    while t.global_step < inspect_from:
        t.step()
    report["instrumented_from_step"] = inspect_from

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
    for _ in range(args.lead_in + args.inspect_steps):
        step = t.global_step
        t._fill_inputs(step)
        t._set_lr(t._lr_at(step))
        t.opt.zero_grad(set_to_none=False)
        logits, state, _ = model(t.static_x, None)
        fwd = _bad([("logits", logits)] +
                   [(f"state_layer{i}", s) for i, s in enumerate(state)])
        flat = logits.reshape(-1, t._vocab).float()
        loss = F.cross_entropy(flat, t.static_y.reshape(-1))
        lbad = _bad([("loss", loss)])
        loss.backward()
        gbad = _bad([(n, p.grad) for n, p in model.named_parameters()])

        grads = [p.grad for p in model.parameters() if p.grad is not None]
        norm32 = float(torch.linalg.vector_norm(
            torch.stack([torch.linalg.vector_norm(g) for g in grads])))
        norm64 = float(torch.linalg.vector_norm(
            torch.stack([torch.linalg.vector_norm(g.double()) for g in grads])))
        # Which parameter carries the largest gradient, so the origin is
        # localised to a tensor rather than to "the model".
        per_param = sorted(
            ((float(g.abs().max()), n)
             for (n, p), g in zip(model.named_parameters(), grads)
             if torch.isfinite(g).any()),
            reverse=True)
        row = {
            "step": step,
            "loss": float(loss.detach()),
            "grad_norm_fp32": norm32,
            "grad_norm_fp64": norm64,
            "fp32_overflowed_while_fp64_finite": (
                norm32 == float("inf") and norm64 == norm64
                and abs(norm64) != float("inf")),
            "grad_abs_max": per_param[0][0] if per_param else None,
            "grad_abs_max_param": per_param[0][1] if per_param else None,
            "top_grads": [{"param": n, "abs_max": v} for v, n in per_param[:5]],
            "param_abs_max": max(float(p.abs().max())
                                 for p in model.parameters()
                                 if torch.isfinite(p).any()),
            "logits_abs_max_finite": finite_max(logits),
            "state_abs_max_finite": [finite_max(s) for s in state],
            "nonfinite_forward": fwd,
            "nonfinite_loss": lbad,
            "nonfinite_grads": gbad,
        }
        steps.append(row)
        print(f"  step {step:6d}  loss {row['loss']:.4f}  "
              f"|g|max {row['grad_abs_max']:.3e} ({row['grad_abs_max_param']})  "
              f"||g||fp32 {norm32:.3e}  ||g||fp64 {norm64:.3e}  "
              f"|logits|max {row['logits_abs_max_finite']}", flush=True)
        if fwd or lbad or gbad:
            origin = "forward" if fwd else "loss" if lbad else "backward"
            print(f"\n  first non-finite TENSOR at step {step}: origin={origin}\n"
                  f"    forward {fwd}\n    loss {lbad}\n"
                  f"    grads ({len(gbad)}) {gbad[:8]}", flush=True)
            break

        # `t._max_norm`, not `t.cfg.grad_clip`: the Trainer maps a non-positive
        # grad_clip to inf, and the instrumented step has to be the step that
        # happened rather than a re-reading of the config.
        torch.nn.utils.clip_grad_norm_(model.parameters(), t._max_norm)
        t.opt.step()
        t.global_step += 1

    # ---- pass 3: the SAME step on the eager path ---------------------------
    # The one question passes 1 and 2 cannot answer. `twocomp_scan` dispatches to
    # a hand-written jiterator backward when fused, and to ordinary autograd when
    # not. R10 -- "a hand-written backward that is silently wrong" -- is this
    # project's worst bug class, and its R10 gate compares the two paths on
    # healthy inputs. It has never compared them on THIS input.
    #
    #   * eager finite, fused NaN  -> the fused backward is the origin, and the
    #     R10 gate has a gap at exactly the inputs that matter;
    #   * both NaN                 -> the arithmetic itself, on both paths, and
    #     the kernel is exonerated.
    #
    # Same checkpoint, same step, same batch, one variable.
    eager = None
    if origin == "backward":
        print("\npass 3: the same step, on the EAGER path (R10 separation)",
              flush=True)
        t3 = _fresh_trainer(args.run, "pass3")
        t3.load_checkpoint(ckpt)
        while t3.global_step < bad_step:
            t3.step()
        m3 = t3.model
        m3.fused = False                     # the single variable
        t3._fill_inputs(bad_step)
        t3._set_lr(t3._lr_at(bad_step))
        t3.opt.zero_grad(set_to_none=False)
        lg3, st3, _ = m3(t3.static_x, None)
        fwd3 = _bad([("logits", lg3)])
        loss3 = F.cross_entropy(lg3.reshape(-1, t3._vocab).float(),
                                t3.static_y.reshape(-1))
        loss3.backward()
        gbad3 = _bad([(n, p.grad) for n, p in m3.named_parameters()])
        eager = {
            "step": bad_step,
            "fused": False,
            "loss": float(loss3.detach()),
            "nonfinite_forward": fwd3,
            "nonfinite_grads": gbad3,
            "grad_abs_max_finite": max(
                (float(g[torch.isfinite(g)].abs().max())
                 for g in (p.grad for p in m3.parameters()) if torch.isfinite(g).any()),
                default=None),
        }
        print(f"  eager loss {eager['loss']:.4f}  "
              f"non-finite grads ({len(gbad3)}) {gbad3[:8]}", flush=True)
        del t3
        torch.cuda.empty_cache()

    report["eager_replay_of_bad_step"] = eager
    if eager is not None:
        fused_bad = set(steps[-1]["nonfinite_grads"])
        eager_bad = set(eager["nonfinite_grads"])
        report["r10_separation"] = (
            "BOTH paths produce NaN -- the fused kernel is exonerated and the "
            "arithmetic is the origin on either path."
            if eager_bad else
            "EAGER IS FINITE WHERE FUSED IS NaN. The hand-written two-compartment "
            "backward is the origin. This is an R10-class finding and the R10 "
            "gate does not cover these inputs.")
        report["fused_only_nonfinite_params"] = sorted(fused_bad - eager_bad)

    report["origin"] = origin
    report["instrumented_steps"] = steps
    overflow = [s for s in steps if s["fp32_overflowed_while_fp64_finite"]]
    report["clip_fp32_overflow_reproduced"] = bool(overflow)
    report["verdict"] = (
        "EXP_009 §9.4's mechanism, a SECOND time and on the adopted arm's "
        "backward: the fp32 sum of squares inside clip_grad_norm_ overflowed "
        "while the same norm in float64 was finite."
        if overflow else
        "NOT EXP_009's mechanism: the fp32 clip norm did not overflow while "
        "fp64 stayed finite. A different origin; see `origin` and the trace."
    )

    (_REPO / args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\norigin: {origin}")
    print(f"verdict: {report['verdict']}")
    print(f"\nWROTE {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
