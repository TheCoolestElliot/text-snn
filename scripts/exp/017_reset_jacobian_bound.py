"""EXP_017 Leg A: the same membranes, both reset rules, side by side.

    python scripts/exp/017_reset_jacobian_bound.py \
        --out docs/reports/data/exp_017_jacobian_bound.json

WHAT IT MEASURES, AND WHY IT IS A COUNTERFACTUAL RATHER THAN A COMPARISON
-------------------------------------------------------------------------
`EXP_016` measured the per-step chain factor `g = beta_f * dv` of the adopted
arm's backward and found it bounded by nothing: 8.96 at `d = 1481`, 17.5x the
plain LIF's hard ceiling, with 85 consecutive expanding timesteps on the batch
the run actually died on.

`EXP_017` §2.2 derives that detaching the fast-pole reset sends `dv` to `1 - s`,
which is in {0, 1}, so `|g| <= beta_f` at every width, every `w`, every `vs`.

**`detach()` changes no forward value.** So both rules can be evaluated on
*exactly the same membranes*, from the same checkpoint, on the same batch, inside
one scan -- and the only thing that differs between the two columns of every
table this file writes is the backward rule itself. That is a strictly stronger
statement than probing two separately-trained models, and it costs one pass
instead of two.

It is also, deliberately, the *hardest* case for the new rule: these are weights
trained BY the hard rule. `EXP_017` §6 item 9 states that limitation in advance;
after Leg B the same probe is pointed at `detach_d1481_s0`'s own checkpoint (leg
A6, picked up automatically once it exists) and both are reported.

WHAT IS REUSED, AND WHY REUSE IS THE POINT
------------------------------------------
`Acc`, `Window`, `lif_bound`, `THRESHOLDS` and `TARGETS` are imported from
`016_reset_jacobian.py` and `016_posthoc_window.py` rather than re-implemented.
Two implementations of one statistic is how two numbers stop being comparable --
the same reason `001_memory_horizon.py` is reused unmodified by six experiments.

**`Window` is `EXP_016`'s POST-HOC quantity, and it is pre-registered here.**
`EXP_016` §9.2 recorded that its P3 placed a bar on `sum over ALL 256 steps`,
which its own §4.0 had already said "is NOT the gradient", and that the maximum
contiguous window was the quantity that should have been asked for. `EXP_017` §3
asks for the corrected one from the start. That is what a referral is for.

WHAT IT IS NOT
--------------
One checkpoint and one batch per cell. No sigma, no seeds, no interval, no bpc.
Every number is a marker. Nothing here trains, nothing is adopted, and no
hyperparameter is read from anywhere but each run's own `config.json`.
"""

from __future__ import annotations

import argparse
import dataclasses
import importlib.util
import json
import sys
from pathlib import Path

import torch

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO / "src") not in sys.path:
    sys.path.insert(0, str(_REPO / "src"))

from snn.config import Config, seed_everything  # noqa: E402
from snn.data import Corpus, RandomWindowSampler  # noqa: E402
from snn.model import build_model  # noqa: E402
from snn.neuron import lif_scan_eager  # noqa: E402
from snn.surrogate import atan_grad  # noqa: E402
from snn.twocomp import twocomp_scan_eager  # noqa: E402
from snn.twocomp_detach import MAX_CHAIN_FACTOR, twocomp_detach_scan_eager  # noqa: E402


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(
        name, _REPO / "scripts" / "exp" / filename)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_j16 = _load("exp016_jac", "016_reset_jacobian.py")
_w16 = _load("exp016_win", "016_posthoc_window.py")

Acc = _j16.Acc
Window = _w16.Window
lif_bound = _j16.lif_bound
T1_TOL = _j16.T1_TOL

RUNS = _REPO / "experiments" / "runs"
EXP016_ARTIFACT = _REPO / "docs" / "reports" / "data" / "exp_016_reset_jacobian.json"

#: H1's tolerance. `dv` is exactly 0 or 1 and `beta_f` is a power of two, so the
#: product is exact in fp32 and this could be asserted at equality. 1e-6 is
#: carried anyway because `beta_f` arrives from a JSON round-trip.
H1_TOL = 1e-6

#: `EXP_016`'s five legs, unchanged, plus this experiment's own arm once it
#: exists. A6 is skipped silently until Leg B has run, which is what makes this
#: file runnable both before and after it.
TARGETS = list(_j16.TARGETS) + [
    ("detach_d1481_s0", "ckpt_final.pt", "A6",
     "detached reset, 5.0M -- THIS ARM's own trajectory (EXP_017 Sec 6 item 9)"),
]

#: The batch the pre-registered Leg A reads, and the batch the dying run actually
#: saw at the step it died on. `EXP_016` Sec 9.4 found that reading batch 0 alone
#: was a defect in ITS Leg A design, so both are pre-registered here rather than
#: one being added post hoc.
BATCH_STEPS = (0, 5138)


# --------------------------------------------------------------------------
# the dual-rule scans
# --------------------------------------------------------------------------

def scan_twocomp_both(cur, v0, w, beta_s, beta_f, thr, alpha, accs, wins):
    """One forward trajectory; `g` under BOTH reset rules at every timestep.

    The forward is identical for the two rules -- `detach()` changes no number --
    so `vf`, `vs` and `v_pre` below are shared, and the two `g` expressions are
    evaluated on the same tensors. That identity is the whole design: nothing in
    the comparison can be attributed to the two arms having reached different
    states, because there is only one state.

        hard      g = beta_f * ((1 - sh) - vf*sgd)      EXP_016 Sec 2.3
        detach    g = beta_f *  (1 - sh)                EXP_017 Sec 2.2

    `dv` uses the eager `vf` (the real fast membrane) rather than the fused
    kernel's reconstruction `v_pre - w*vs`; they are algebraically identical and
    the eager path is the one that reproduces the failure. Inherited verbatim
    from `016_reset_jacobian.scan_twocomp`, whose numbers this file's `hard`
    column must reproduce (gate T1b).
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
        drive = w * vs
        g_hard = beta_f * ((1.0 - sh) - vf * sgd)
        g_detach = beta_f * (1.0 - sh)
        accs["hard"].add(g_hard, drive)
        accs["detach"].add(g_detach, drive)
        wins["hard"].add(g_hard)
        wins["detach"].add(g_detach)
        spikes.append(sh)
        vf = vf * (1.0 - sh)
    return torch.stack(spikes, dim=1), torch.cat([vf, vs], dim=1)


def scan_lif_both(cur, v0, beta, thr, alpha, accs, wins):
    """The plain LIF, for the instrument check.

    The `detach` column here is `snn.kernels`' own `"detached"` mode -- a
    first-class Phase-2 reset rule with its own compiled kernel and its own
    mutation coverage (M04, M05), not something this experiment invented. Its
    bound is the same `beta` as the two-compartment case, for the same reason.
    """
    v = v0
    spikes = []
    for t in range(cur.shape[1]):
        v_pre = v * beta + cur[:, t]
        x = v_pre - thr
        s = (x >= 0).to(v_pre.dtype)
        sg = atan_grad(x, alpha)
        g_hard = beta * ((1.0 - s) - v_pre * sg)
        g_detach = beta * (1.0 - s)
        accs["hard"].add(g_hard, None)
        accs["detach"].add(g_detach, None)
        wins["hard"].add(g_hard)
        wins["detach"].add(g_detach)
        spikes.append(s)
        v = v_pre * (1.0 - s)
    return torch.stack(spikes, dim=1), v


# --------------------------------------------------------------------------

def probe(run: str, ckpt_name: str, leg: str, note: str, device: str,
          batch_step: int) -> dict:
    ckpt_path = RUNS / run / ckpt_name
    if not ckpt_path.exists():
        return {"leg": leg, "run": run, "batch_step": batch_step,
                "error": f"missing {ckpt_name}", "skipped": True}

    raw = json.loads((RUNS / run / "config.json").read_text(encoding="utf-8"))
    known = {f.name for f in dataclasses.fields(Config)}
    cfg = Config(**{k: v for k, v in raw.items() if k in known})
    cfg.device = device
    cfg.fused = False       # read the same arithmetic on any machine
    cfg.cuda_graph = False

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

    bound = lif_bound(cfg.surrogate_alpha, cfg.threshold, cfg.beta)
    row = {
        "leg": leg, "run": run, "note": note, "arch": cfg.arch,
        "batch_step": batch_step,
        "d_model": cfg.d_model, "n_layers": cfg.n_layers,
        "step_of_checkpoint": int(ck.get("global_step", -1)),
        "checkpoint": ckpt_name,
        "seq_len": cfg.seq_len, "batch_size": cfg.batch_size,
        "constants": {"alpha": cfg.surrogate_alpha, "thr": cfg.threshold,
                      "beta_f": cfg.beta, "reset": cfg.reset},
        "lif_closed_form_bound": bound,
        "detach_closed_form_bound": cfg.beta,
        "layers": [],
        "t2_spikes_bitwise_equal": [],
        "t2_state_max_abs_diff": [],
    }

    with torch.no_grad():
        h = model.embed(x)
        state = model.init_state(x.shape[0], device)
        for k, linear in enumerate(model.layers):
            cur = model._project(linear, h)
            B, d = cur.shape[0], cur.shape[-1]
            accs = {r: Acc(B, d, device) for r in ("hard", "detach")}
            wins = {r: Window(B, d, device) for r in ("hard", "detach")}

            if cfg.arch in ("twocomp", "twocomp_detach"):
                w, bs = model.w[k], model.slow_decay(k)
                mine, v_end = scan_twocomp_both(
                    cur, state[k], w, bs, cfg.beta, cfg.threshold,
                    cfg.surrogate_alpha, accs, wins)
                eager = (twocomp_detach_scan_eager
                         if cfg.arch == "twocomp_detach" else twocomp_scan_eager)
                ref, v_ref = eager(cur, state[k], w, bs, cfg.beta, cfg.threshold,
                                   cfg.surrogate_alpha)
            elif cfg.arch == "snn":
                mine, v_end = scan_lif_both(
                    cur, state[k], cfg.beta, cfg.threshold,
                    cfg.surrogate_alpha, accs, wins)
                ref, v_ref = lif_scan_eager(
                    cur, state[k], cfg.beta, cfg.threshold,
                    cfg.surrogate_alpha, cfg.reset)
            else:
                return {"leg": leg, "run": run,
                        "error": f"arch {cfg.arch!r} has no scan of this form"}

            # T2: the local scan IS the committed scan, or nothing below counts.
            row["t2_spikes_bitwise_equal"].append(bool(torch.equal(mine, ref)))
            row["t2_state_max_abs_diff"].append(float((v_end - v_ref).abs().max()))

            entry = {"layer": k}
            for rule in ("hard", "detach"):
                rep = accs[rule].report()
                rep.update(wins[rule].report())
                rep["max_abs_g_over_lif_bound"] = rep["max_abs_g"] / bound
                entry[rule] = rep
            entry["max_abs_g_ratio_hard_over_detach"] = (
                entry["hard"]["max_abs_g"] / max(entry["detach"]["max_abs_g"], 1e-300))
            row["layers"].append(entry)
            h = ref
            del cur, mine, ref
            if device == "cuda":
                torch.cuda.empty_cache()

    # ---- the bars, evaluated here so the artifact carries them ------------
    row["t1_holds"] = (
        cfg.arch != "snn"
        or all(ly["hard"]["max_abs_g"] <= bound + T1_TOL for ly in row["layers"])
    )
    row["t2_holds"] = (all(row["t2_spikes_bitwise_equal"])
                       and max(row["t2_state_max_abs_diff"]) <= 1e-5)
    row["h1_holds"] = all(
        ly["detach"]["max_abs_g"] <= MAX_CHAIN_FACTOR + H1_TOL
        and ly["detach"]["n_over_1"] == 0
        and ly["detach"]["longest_expanding_run_steps"] == 0
        for ly in row["layers"]
    )
    return row


def check_reproduces_exp016(rows: list[dict]) -> dict:
    """T1b: this probe's `hard` column must reproduce `EXP_016`'s committed one.

    `CONTRIBUTING.md` §5: *make every probe reproduce a number it did not
    produce.* T1 checks the plain LIF against a closed form, which validates the
    arithmetic but not the plumbing -- the checkpoint loading, the sampler, the
    layer loop and the state threading are all shared with `EXP_016` and none of
    them is constrained by a bound on `dv`.

    So the `hard` column at `batch_step = 0` is compared element-for-element
    against `docs/reports/data/exp_016_reset_jacobian.json`, which was committed
    before this file existed. Exact equality is required: both paths run the same
    fp64 accumulation over the same fp32 forward on the same batch, so anything
    non-zero here is a plumbing difference, not rounding.
    """
    if not EXP016_ARTIFACT.exists():
        return {"checked": False, "why": "EXP_016 artifact not on disk"}
    prev = json.loads(EXP016_ARTIFACT.read_text(encoding="utf-8"))
    ref = {}
    for r in prev.get("results", []):
        if "error" in r or r.get("batch_step", 0) != 0:
            continue
        for ly in r.get("layers", []):
            ref[(r["leg"], ly["layer"])] = ly["max_abs_g"]
    if not ref:
        return {"checked": False, "why": "no comparable rows in the EXP_016 artifact"}

    diffs, compared = [], 0
    for r in rows:
        if r.get("error") or r.get("batch_step") != 0:
            continue
        for ly in r["layers"]:
            key = (r["leg"], ly["layer"])
            if key not in ref:
                continue
            compared += 1
            d = abs(ly["hard"]["max_abs_g"] - ref[key])
            if d != 0.0:
                diffs.append({"leg": key[0], "layer": key[1],
                              "exp_016": ref[key],
                              "exp_017_hard": ly["hard"]["max_abs_g"],
                              "abs_diff": d})
    return {"checked": True, "n_compared": compared, "n_differing": len(diffs),
            "differences": diffs, "holds": compared > 0 and not diffs}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="docs/reports/data/exp_017_jacobian_bound.json")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--only", default=None, help="comma-separated leg ids, e.g. A4,A5")
    ap.add_argument("--batch-steps", default=",".join(str(b) for b in BATCH_STEPS))
    args = ap.parse_args(argv)

    want = None if args.only is None else {s.strip() for s in args.only.split(",")}
    batches = [int(b) for b in args.batch_steps.split(",") if b.strip()]

    rows = []
    for batch_step in batches:
        for run, ck, leg, note in TARGETS:
            if want is not None and leg not in want:
                continue
            print(f"[{leg}] {run} batch={batch_step} ...", flush=True)
            r = probe(run, ck, leg, note, args.device, batch_step)
            rows.append(r)
            if r.get("error"):
                print(f"  {r['error']}", flush=True)
                continue
            for ly in r["layers"]:
                h, dd = ly["hard"], ly["detach"]
                print(f"  L{ly['layer']}  hard: max|g|={h['max_abs_g']:.6g} "
                      f"frac>1={h['frac_over']['1.0']:.3g} "
                      f"run={h['longest_expanding_run_steps']:.0f} "
                      f"win=10^{h['max_window_log10_gain']:.2f}   ||   "
                      f"detach: max|g|={dd['max_abs_g']:.6g} "
                      f"frac>1={dd['frac_over']['1.0']:.3g} "
                      f"run={dd['longest_expanding_run_steps']:.0f}", flush=True)

    scored = [r for r in rows if not r.get("error")]
    out = {
        "experiment": "EXP_017_bounded_reset_jacobian",
        "leg": "A",
        "prereg": "experiments/logs/EXP_017_bounded_reset_jacobian.md",
        "batch_steps": batches,
        "detach_closed_form_bound": MAX_CHAIN_FACTOR,
        "h1_tolerance": H1_TOL,
        "trains_nothing": True,
        "adopts_nothing": True,
        "ranks_nothing": True,
        "changes_no_hyperparameter": True,
        "legs": rows,
        "t1_holds": all(r.get("t1_holds", True) for r in scored),
        "t2_holds": all(r.get("t2_holds", False) for r in scored),
        "h1_holds": all(r.get("h1_holds", False) for r in scored),
        "t1b_reproduces_exp_016": check_reproduces_exp016(rows),
    }
    p = Path(args.out)
    if not p.is_absolute():
        p = _REPO / p
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")

    print(f"\nwrote {p}")
    print(f"  T1  (snn legs reproduce the closed form) : "
          f"{'HOLDS' if out['t1_holds'] else 'FAILS'}")
    t1b = out["t1b_reproduces_exp_016"]
    print(f"  T1b (hard column reproduces EXP_016)     : "
          f"{'HOLDS' if t1b.get('holds') else ('FAILS' if t1b.get('checked') else 'NOT CHECKED')}"
          f"  ({t1b.get('n_compared', 0)} cells, {t1b.get('n_differing', '?')} differing)")
    print(f"  T2  (local scan == committed eager)      : "
          f"{'HOLDS' if out['t2_holds'] else 'FAILS'}")
    print(f"  H1  (detached rule bounded by beta_f)    : "
          f"{'HOLDS' if out['h1_holds'] else 'REFUTED'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
