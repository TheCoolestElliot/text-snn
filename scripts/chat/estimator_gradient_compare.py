"""Does the bounded estimator actually change the gradient this model gets?

NOT PART OF THE RESEARCH PROTOCOL. `src/snnchat/` and `scripts/chat/` sit outside
the Phase-1..5 study, so nothing here is pre-registered and no number it produces
is a reported figure. It trains nothing and writes nothing outside `--out`.

THE QUESTION, AND WHY IT IS THE ONE THAT MATTERS
--------------------------------------------------
`scripts/chat/reset_jacobian_probe.py` establishes that every chat checkpoint on
disk sits inside the region `EXP_016` proved unbounded -- the shipped model's
layer 0 runs at 19.1x a ceiling the plain LIF provably cannot exceed. That is a
fact about the **per-step multiplier** `beta_f * dv`.

It is not yet a fact about the **gradient**. `EXP_016`'s own P3 resolved
"REFUTED AS SUFFICIENT" and §15 records why: the adjoint is injected at every
timestep, so a product over the unroll averages the mechanism away, and an
expanding window is a **necessary, not sufficient** condition for a failure.
A tiny fraction of sites expanding for 5-9 consecutive steps can leave the
parameter gradient entirely ordinary.

So this script asks the question the probe cannot: load ONE set of weights into
both arches, run the same batch through both, and compare what `backward()`
actually delivers to the optimiser. It is written to be able to return "no
difference worth caring about", and `GRADIENT_BOUND_NOTE.md` §2.2 reports that it
largely does.

WHAT IT CHECKS BEFORE IT COMPARES ANYTHING
--------------------------------------------
The loss must be **bit-identical** between the two arches. `snn.twocomp_detach`
imports `twocomp_forward_kernel` from `snn.twocomp` rather than re-declaring it,
so the forward is the same compiled kernel and the same network; if the losses
ever differ, the comparison below is between two models rather than between two
estimators and every number in it is meaningless. Reported as `loss_bitwise_equal`
and printed loudly.

`w` and `beta_s_raw` are singled out from the parameter sweep because they are
the two the mechanism runs through: `EXP_016`'s bound breaks at `|w * vs| ~ 1`,
and `beta_s_raw` is what sets how far `vs` integrates before it gets there.

READ THE RESULT WITH ITS DENOMINATOR
--------------------------------------
Four batches of one checkpoint is four batches of one checkpoint. It says what
the estimators do **at this point in weight space**, which is a converged one
reached by training with the *undetached* estimator. It says nothing about what
they do to a trajectory, which is the only thing that would settle whether the
arch is worth training with, and which costs a training run rather than a minute.

    python scripts/chat/estimator_gradient_compare.py
    python scripts/chat/estimator_gradient_compare.py --run chat-v6-scratch --out cmp.json
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
import torch.nn.functional as F  # noqa: E402

from snnchat.data import ChatCorpus, MixtureSampler  # noqa: E402
from snnchat.model import ChatConfig, build_chat_model  # noqa: E402

CHAT = REPO / "experiments" / "chat"
ARCHES = ("twocomp_threshold", "twocomp_threshold_detach")


def _grad_stats(model, n_layers: int) -> dict:
    g = {n: p.grad for n, p in model.named_parameters() if p.grad is not None}
    return {
        "global_norm": math.sqrt(sum(float((v.double() ** 2).sum()) for v in g.values())),
        "max_abs": max(float(v.abs().max()) for v in g.values()),
        "n_nonfinite": sum(int((~torch.isfinite(v)).sum()) for v in g.values()),
        "n_params_with_grad": len(g),
        "w_norm": math.sqrt(sum(float((g[f"w.{k}"].double() ** 2).sum())
                                for k in range(n_layers))),
        "w_max": max(float(g[f"w.{k}"].abs().max()) for k in range(n_layers)),
        "beta_s_raw_norm": math.sqrt(sum(float((g[f"beta_s_raw.{k}"].double() ** 2).sum())
                                         for k in range(n_layers))),
        "beta_s_raw_max": max(float(g[f"beta_s_raw.{k}"].abs().max())
                              for k in range(n_layers)),
    }


def compare(run: str, ckpt_name: str, device: str, batches: tuple[int, ...]) -> dict:
    run_dir = CHAT / run
    ckpt_path = run_dir / ckpt_name
    if not ckpt_path.exists():
        return {"run": run, "error": f"missing {ckpt_path}"}

    raw = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    known = {f.name for f in dataclasses.fields(ChatConfig)}
    base = {k: v for k, v in raw.items() if k in known}

    probe_cfg = ChatConfig(**base)
    corpus = ChatCorpus(probe_cfg.data_dir)
    sampler = MixtureSampler(
        corpus, probe_cfg.mix, probe_cfg.batch_size, probe_cfg.seq_len,
        probe_cfg.seed, align_frac=probe_cfg.align_frac,
        align_lookahead=probe_cfg.align_lookahead,
    )
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)

    per_arch: dict[str, list] = {}
    for arch in ARCHES:
        cfg = ChatConfig(**base)
        cfg.arch, cfg.device = arch, device
        cfg.spread_tau = False       # the spread is baked into the checkpoint
        model = build_chat_model(cfg)
        model.load_state_dict(ck["model"])
        model.train()

        rows = []
        for s in batches:
            x, y = sampler.batch(s)
            x, y = x.to(device), y.to(device)
            model.zero_grad(set_to_none=True)
            logits, _, _ = model(x, None)
            loss = F.cross_entropy(
                logits.reshape(-1, cfg.vocab_size).float(), y.reshape(-1))
            loss.backward()
            rows.append({"batch": s, "loss": float(loss.detach()),
                         **_grad_stats(model, cfg.n_layers)})
        per_arch[arch] = rows
        del model
        if device == "cuda":
            torch.cuda.empty_cache()

    ratios = []
    for i, s in enumerate(batches):
        a, b = per_arch[ARCHES[0]][i], per_arch[ARCHES[1]][i]
        ratios.append({
            "batch": s,
            # The gate. If this is ever False, nothing below means anything.
            "loss_bitwise_equal": a["loss"] == b["loss"],
            "global_norm_ratio": a["global_norm"] / b["global_norm"],
            "max_abs_ratio": a["max_abs"] / b["max_abs"],
            "w_norm_ratio": a["w_norm"] / b["w_norm"],
            "beta_s_raw_max_ratio": a["beta_s_raw_max"] / b["beta_s_raw_max"],
        })

    return {
        "run": run, "checkpoint": ckpt_name,
        "step_of_checkpoint": int(ck.get("global_step", -1)),
        "d_model": probe_cfg.d_model, "n_layers": probe_cfg.n_layers,
        "seq_len": probe_cfg.seq_len, "batch_size": probe_cfg.batch_size,
        "batches": list(batches),
        "numerator": ARCHES[0], "denominator": ARCHES[1],
        "per_arch": per_arch, "ratios": ratios,
        "loss_bitwise_equal_on_every_batch": all(r["loss_bitwise_equal"] for r in ratios),
        "any_nonfinite_gradient": any(
            r["n_nonfinite"] for rows in per_arch.values() for r in rows),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run", default="chat-v3d-aligned")
    ap.add_argument("--ckpt", default="ckpt_best.pt")
    ap.add_argument("--batches", type=int, nargs="+", default=[0, 1000, 5000, 13999])
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default="")
    args = ap.parse_args(argv)

    r = compare(args.run, args.ckpt, args.device, tuple(args.batches))
    if "error" in r:
        print(f"ERROR: {r['error']}")
        return 1

    print(f"{r['run']}  d={r['d_model']} K={r['n_layers']}  L={r['seq_len']} "
          f"B={r['batch_size']}  step {r['step_of_checkpoint']}")
    print(f"  forward identical on every batch: {r['loss_bitwise_equal_on_every_batch']}"
          f"   any non-finite gradient: {r['any_nonfinite_gradient']}")
    print(f"\n  {'arch':>26} {'batch':>6} {'loss':>8} {'|g|':>9} {'max|g|':>9} "
          f"{'|dw|':>9} {'max|dbeta_raw|':>15}")
    for arch, rows in r["per_arch"].items():
        for row in rows:
            print(f"  {arch:>26} {row['batch']:>6} {row['loss']:>8.4f} "
                  f"{row['global_norm']:>9.4f} {row['max_abs']:>9.4f} "
                  f"{row['w_norm']:>9.4f} {row['beta_s_raw_max']:>15.5f}")
    print(f"\n  ratios, {r['numerator']} / {r['denominator']}:")
    for x in r["ratios"]:
        print(f"    batch {x['batch']:>6}  |g| {x['global_norm_ratio']:.3f}   "
              f"max|g| {x['max_abs_ratio']:.3f}   |dw| {x['w_norm_ratio']:.3f}   "
              f"max|dbeta_raw| {x['beta_s_raw_max_ratio']:.3f}")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps({
            "script": "scripts/chat/estimator_gradient_compare.py",
            "not_a_research_artifact": True,
            "question": ("at one checkpoint's weights, how much does the detached "
                         "estimator change the gradient the optimiser receives?"),
            "trains_nothing": True,
            "adopts_nothing": True,
            "result": r,
        }, indent=1), encoding="utf-8")
        print(f"\n  wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
