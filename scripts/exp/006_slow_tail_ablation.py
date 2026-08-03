"""EXP_006 -- do ~3% of channels carry the horizon?

Pre-registered in experiments/logs/EXP_006_slow_tail_ablation.md. Read that
first: the interventions, the thresholds and the decision rule were all fixed
before this file existed.

What this does
--------------
Loads the seven trained two-compartment checkpoints, edits `beta_s_raw` in a
COPY of each state dict, and re-runs EXP_001's memory-horizon probe on the
edited model. No training, no new kernel, no new parameter.

Why the intervention is on beta_s and not on w
----------------------------------------------
EXP_004 10.6: at zero context both compartments start from zero, so

    v_0 = vf_0 + w_c*vs_0 = cur_0 + w_c*cur_0 = cur_0*(1 + w_c)

which makes `w` a learned per-channel THRESHOLD as well as a mix, and 40.6% of
the arm's gain is that threshold. Ablating `w` would move the timescale and the
threshold together and could not answer a question about either.

`beta_s` does not appear in that expression -- `vs_0 = 0*beta_s + cur_0` for any
finite decay -- so clamping it cannot change anything the network computes at
zero context, exactly and bitwise. That is the isolation the experiment needs
AND a self-check it did not have to invent: at k = 1 every scored character has
zero context, so the probe's k = 1 point must be BIT-IDENTICAL to the intact
model's under every intervention here (H2). A non-zero residual means this file
is not doing what the pre-registration says it does.

Why the probe is imported rather than reimplemented
---------------------------------------------------
Failure mode L5. `score_at_horizon`, `paired_context_curve` and
`horizon_from_excess` are the instrument that produced every horizon number in
this project. A second implementation of one statistic is how two numbers stop
being comparable, so this file imports them, adds no entry to the probe's `ARMS`
registry, and never writes to a committed artifact.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import statistics
import sys
import time
from pathlib import Path

import torch

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO / "src") not in sys.path:
    sys.path.insert(0, str(_REPO / "src"))


def _load_probe():
    """Import scripts/exp/001_memory_horizon.py, whose name is not an identifier."""
    path = _REPO / "scripts" / "exp" / "001_memory_horizon.py"
    spec = importlib.util.spec_from_file_location("exp001_memory_horizon", path)
    if spec is None or spec.loader is None:          # pragma: no cover
        raise SystemExit(f"cannot import the EXP_001 probe from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


PROBE = _load_probe()

from snn.config import seed_everything                          # noqa: E402
from snn.data import Corpus                                     # noqa: E402
from snn.model import build_model, count_params                 # noqa: E402

SEEDS = tuple(f"twocomp_s{i}" for i in range(7))

# Pre-registered configuration list (EXP_006 2). A(0) is the intact reference and
# MUST come first: H2 and the bpc deltas are all measured against it.
CONFIGS: tuple[tuple[str, int], ...] = (
    ("A", 0),
    ("A", 4), ("A", 8), ("A", 16), ("A", 32), ("A", 64), ("A", 128), ("A", 512),
    ("B", 16), ("B", 64),
    ("C", 16), ("C", 64),
    ("E", 16),
)

# EXP_006 7, L7: fixed here so the random control's draw cannot be chosen after
# seeing a result.
RNG_SEED = 20260803

# EXP_006 3: thresholds are evaluated at the inherited bar, at which the intact
# arm's committed horizons are 38/47/45/57/48/47/53 and the baseline's are 7.
# EXP_005's two-compartment family bar is reported beside every number and is
# never mixed into a threshold (failure mode L4).
TOL_INHERITED = 0.00922
TOL_FAMILY = 0.00627

# EXP_006 6, H2: forced by algebra, so the tolerance is numerical noise and not
# a modelling allowance.
H2_TOLERANCE_BPC = 1e-12

# E's eligibility cut (EXP_006 2): "channels with intact beta_s < 0.8", i.e. ones
# with no long timescale to lose.
E_ELIGIBLE_BELOW = 0.8


# --------------------------------------------------------------------------
# the interventions
# --------------------------------------------------------------------------

def _logit(x: torch.Tensor) -> torch.Tensor:
    return torch.log(x / (1.0 - x))


def layer_facts(base_sd: dict, layer: int) -> dict:
    """Rankings and the clamp target, computed ONCE from the intact checkpoint.

    Failure mode L6: if these were recomputed from an edited model, one
    configuration's edit would contaminate the next one's ranking.
    """
    beta = torch.sigmoid(base_sd[f"beta_s_raw.{layer}"].flatten().double())
    w = base_sd[f"w.{layer}"].flatten().double()
    return {
        "beta": beta,
        "w": w,
        "median": float(beta.median()),
        "order_beta": torch.argsort(beta, descending=True),
        "order_w": torch.argsort(w.abs(), descending=True),
        "eligible_for_E": torch.nonzero(beta < E_ELIGIBLE_BELOW).flatten(),
        "n_above_0.9": int((beta > 0.9).sum()),
    }


def plan_layer(kind: str, n: int, facts: dict, gen: torch.Generator) -> dict:
    """Return {index -> new beta} for one layer under one configuration.

    A -- clamp the top n by beta_s to the layer median.
    B -- clamp everything EXCEPT the top n by beta_s (keep only the tail).
    C -- clamp the top n by |w|.
    E -- n random channels with intact beta_s < 0.8, shifted DOWN by the mean
         displacement A(n) applies in this layer, clamped to (0.01, 0.99). Same
         count and same mean parameter displacement as A(n), different channels.
    """
    beta, median = facts["beta"], facts["median"]
    if kind == "A":
        idx = facts["order_beta"][:n]
        return {int(i): median for i in idx}
    if kind == "B":
        idx = facts["order_beta"][n:]
        return {int(i): median for i in idx}
    if kind == "C":
        idx = facts["order_w"][:n]
        return {int(i): median for i in idx}
    if kind == "E":
        tail = facts["order_beta"][:n]
        displacement = float((beta[tail] - median).abs().mean())
        pool = facts["eligible_for_E"]
        if pool.numel() < n:
            raise SystemExit(f"E({n}): only {pool.numel()} eligible channels")
        pick = pool[torch.randperm(pool.numel(), generator=gen)[:n]]
        return {int(i): min(max(float(beta[int(i)]) - displacement, 0.01), 0.99)
                for i in pick}
    raise SystemExit(f"unknown intervention {kind!r}")


def apply_plan(base_sd: dict, plans: list[dict]) -> tuple[dict, list[list[int]]]:
    """A fresh copy of the checkpoint with beta_s_raw edited and nothing else.

    Returns the new state dict and, per layer, the indices that actually changed
    -- which H5 checks against the indices the plan asked for. They can differ by
    one: the channel that IS the median is already at the target.
    """
    sd = {k: v.clone() for k, v in base_sd.items()}
    changed: list[list[int]] = []
    for layer, plan in enumerate(plans):
        key = f"beta_s_raw.{layer}"
        raw = sd[key]
        before = raw.clone()
        for i, target in plan.items():
            t = torch.tensor(float(target), dtype=torch.float64)
            raw[0, i] = _logit(t).to(raw.dtype)
        changed.append(torch.nonzero(raw != before).flatten().tolist()
                       if raw.dim() == 1 else
                       torch.nonzero((raw != before).flatten()).flatten().tolist())
    return sd, changed


# --------------------------------------------------------------------------
# one probe run
# --------------------------------------------------------------------------

def run_probe(model, data, cfg, device) -> dict:
    """The full k-sweep, then EXP_001's own paired statistic. Nothing new here."""
    curve = {}
    for k in PROBE.K_VALUES:
        curve[str(k)] = PROBE.score_at_horizon(
            model, data, cfg.batch_size, cfg.seq_len, k, device, max_windows=0)
    pc = PROBE.paired_context_curve(curve, cfg.seq_len)
    ex = pc["excess_bpc_at_context"]
    return {
        "bpc_fresh": curve[str(cfg.seq_len)]["bpc"],
        "bpc_k1": curve["1"]["bpc"],
        "nats_k1": curve["1"]["nats"],
        "n_chars": curve[str(cfg.seq_len)]["n_chars"],
        "horizon_inherited": PROBE.horizon_from_excess(ex, TOL_INHERITED),
        "horizon_family": PROBE.horizon_from_excess(ex, TOL_FAMILY),
        "excess_bpc_at_context": ex,
    }


# `horizon_from_excess` returns None when the excess curve never flattens inside
# the probe's range, which means "horizon > 127" -- the LONGEST outcome, not a
# missing one. Encoding it as 128 keeps the median meaningful; silently dropping
# those seeds would bias every median downward, and dropping them from a heavily
# ablated arm would bias it downward exactly where the experiment is looking.
HORIZON_UNRESOLVED = 128


def _median_horizon(rows: list[dict], key: str) -> float | None:
    vals = [HORIZON_UNRESOLVED if r[key] is None else r[key] for r in rows]
    return statistics.median(vals) if vals else None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", default="test", choices=["val", "test"])
    ap.add_argument("--runs", default="", help="comma-separated subset of seeds")
    ap.add_argument("--out", default="docs/reports/data/exp_006_slow_tail_ablation.json")
    args = ap.parse_args(argv)

    wanted = [r.strip() for r in args.runs.split(",") if r.strip()] or list(SEEDS)

    # H4's reference: EXP_005's committed per-seed horizons, READ from the
    # artifact rather than retyped, so a typo cannot turn a failing check into a
    # passing one.
    exp005 = json.loads((_REPO / "docs" / "reports" / "data"
                         / "exp_005_memory_horizon.json").read_text(encoding="utf-8"))
    committed_horizon = {r: e["paired_context"]["horizon_2sigma"]
                         for r, e in exp005["arms"].items()}

    out: dict = {
        "experiment": "EXP_006_slow_tail_ablation",
        "split": args.split,
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "k_values": list(PROBE.K_VALUES),
        "tolerances": {"inherited": TOL_INHERITED, "family_exp005": TOL_FAMILY},
        "rng_seed": RNG_SEED,
        "configs": [f"{k}({n})" for k, n in CONFIGS],
        "seeds": {},
        "self_checks": {"H1": [], "H2": [], "H3": [], "H4": [], "H5": []},
    }
    failures: list[str] = []

    for run in wanted:
        ckpt = _REPO / "experiments" / "runs" / run / "ckpt_final.pt"
        if not ckpt.exists():
            raise SystemExit(f"{run}: no checkpoint at {ckpt}")

        ck = torch.load(ckpt, map_location="cpu", weights_only=False)
        cfg = PROBE._config_from_checkpoint(ck)
        if cfg.arch != "twocomp":
            raise SystemExit(f"{run}: arch is {cfg.arch!r}, not 'twocomp'")
        seed_everything(cfg.seed, cfg.deterministic)
        device = torch.device(cfg.device)
        corpus = Corpus(cfg.corpus, cfg.data_dir)
        data = corpus.split(args.split)
        model = build_model(cfg).to(device)

        base_sd = {k: v.clone() for k, v in ck["model"].items()}
        facts = [layer_facts(base_sd, L) for L in range(cfg.n_layers)]
        gen = torch.Generator().manual_seed(RNG_SEED + cfg.seed)

        # Recorded because EXP_004 10.5 says |w| is uncorrelated with the tail and
        # C is the control built on that claim.
        overlap = [len(set(f["order_beta"][:16].tolist())
                       & set(f["order_w"][:16].tolist())) for f in facts]

        entry = {
            "seed": cfg.seed,
            "params": count_params(model),
            "checkpoint_step": ck.get("step"),
            "n_above_0.9_per_layer": [f["n_above_0.9"] for f in facts],
            "median_beta_s_per_layer": [f["median"] for f in facts],
            "top16_beta_vs_top16_w_overlap": overlap,
            "results": {},
        }
        print(f"\n{run}  seed={cfg.seed}  beta_s>0.9 per layer "
              f"{entry['n_above_0.9_per_layer']}  median "
              f"{[round(f['median'], 4) for f in facts]}  "
              f"top16 overlap {overlap}")

        ref: dict | None = None
        for kind, n in CONFIGS:
            tag = f"{kind}({n})"
            plans = [plan_layer(kind, n, f, gen) for f in facts]
            sd, changed = apply_plan(base_sd, plans)

            # ---- H1: beta_s_raw and nothing else -------------------------
            other = [k for k in base_sd
                     if not k.startswith("beta_s_raw.")
                     and not torch.equal(sd[k], base_sd[k])]
            h1 = not other
            out["self_checks"]["H1"].append({"run": run, "config": tag, "pass": h1,
                                             "tensors_changed": other})
            if not h1:
                failures.append(f"H1 {run} {tag}: also changed {other}")

            # ---- H5: the edit landed, on exactly the planned channels -----
            #
            # Two legs. The strict one is containment: NOTHING outside the plan
            # may move, which is what would catch an off-by-one in a ranking or a
            # broadcast that wrote a whole row. The count is checked to within
            # one, not exactly: the channel that IS the layer median is already
            # at the target, so clamping it is a legitimate no-op. Demanding
            # exact equality there would make the check fail on arithmetic that
            # is correct.
            subset = all(set(c) <= set(plan) for c, plan in zip(changed, plans))
            counts_ok = all(len(plan) - 1 <= len(c) <= len(plan)
                            for c, plan in zip(changed, plans))
            out["self_checks"]["H5"].append({
                "run": run, "config": tag,
                "indices_match": subset and counts_ok,
                "changed_is_subset_of_planned": subset,
                "n_planned": [len(p) for p in plans],
                "n_changed": [len(c) for c in changed],
            })
            if not subset:
                failures.append(f"H5 {run} {tag}: channels outside the plan moved")
            elif not counts_ok:
                failures.append(
                    f"H5 {run} {tag}: changed {[len(c) for c in changed]} of "
                    f"planned {[len(p) for p in plans]}")

            model.load_state_dict(sd)
            model.eval()
            t0 = time.perf_counter()
            row = run_probe(model, data, cfg, device)
            row["wall_clock_s"] = round(time.perf_counter() - t0, 1)
            entry["results"][tag] = row

            if kind == "A" and n == 0:
                ref = row
                # ---- H3 (EXP_001's F1) -------------------------------------
                comm = PROBE.committed_fresh_bpc(run, args.split)
                res = abs(row["bpc_fresh"] - comm) if comm is not None else None
                h3 = res is not None and res < PROBE.F1_TOLERANCE_BPC
                out["self_checks"]["H3"].append({
                    "run": run, "committed": comm, "measured": row["bpc_fresh"],
                    "residual": res, "pass": h3})
                if not h3:
                    failures.append(f"H3 {run}: F1 residual {res}")
                # ---- H4: EXP_005's committed horizon ------------------------
                h4 = row["horizon_inherited"] == committed_horizon.get(run)
                out["self_checks"]["H4"].append({
                    "run": run, "committed": committed_horizon.get(run),
                    "measured": row["horizon_inherited"], "pass": h4})
                if not h4:
                    failures.append(
                        f"H4 {run}: horizon {row['horizon_inherited']} != "
                        f"committed {committed_horizon.get(run)}")
            else:
                # ---- H2: zero context cannot depend on beta_s --------------
                assert ref is not None
                res = abs(row["bpc_k1"] - ref["bpc_k1"])
                h2 = res < H2_TOLERANCE_BPC
                out["self_checks"]["H2"].append({
                    "run": run, "config": tag, "residual_bpc": res, "pass": h2,
                    "nats_identical": row["nats_k1"] == ref["nats_k1"]})
                if not h2:
                    failures.append(f"H2 {run} {tag}: k=1 residual {res:.3e}")
                # ---- H5's second leg: a real ablation moves the score -------
                if n >= 4 and row["bpc_fresh"] == ref["bpc_fresh"]:
                    failures.append(f"H5 {run} {tag}: bpc identical to A(0)")

            d = row["bpc_fresh"] - (ref["bpc_fresh"] if ref else float("nan"))
            print(f"   {tag:8s} horizon {str(row['horizon_inherited']):>4s} "
                  f"({str(row['horizon_family']):>4s} @family)   "
                  f"fresh {row['bpc_fresh']:.5f}  d {d:+.5f}   "
                  f"{row['wall_clock_s']:.0f}s")

        out["seeds"][run] = entry
        del model
        torch.cuda.empty_cache()

    # ---- aggregation ---------------------------------------------------
    agg: dict = {}
    for kind, n in CONFIGS:
        tag = f"{kind}({n})"
        rows = [out["seeds"][r]["results"][tag] for r in out["seeds"]]
        refs = [out["seeds"][r]["results"]["A(0)"] for r in out["seeds"]]
        deltas = [x["bpc_fresh"] - y["bpc_fresh"] for x, y in zip(rows, refs)]
        agg[tag] = {
            "n_seeds": len(rows),
            "horizon_inherited_per_seed": [r["horizon_inherited"] for r in rows],
            "horizon_inherited_median": _median_horizon(rows, "horizon_inherited"),
            "horizon_family_per_seed": [r["horizon_family"] for r in rows],
            "horizon_family_median": _median_horizon(rows, "horizon_family"),
            "n_seeds_horizon_unresolved": sum(
                1 for r in rows if r["horizon_inherited"] is None),
            "bpc_fresh_mean": sum(r["bpc_fresh"] for r in rows) / len(rows),
            "delta_bpc_fresh_mean": sum(deltas) / len(deltas),
            "delta_bpc_fresh_per_seed": deltas,
        }
    out["by_config"] = agg

    # ---- the pre-registered predictions, resolved as written -------------
    def med(tag: str) -> float | None:
        return agg[tag]["horizon_inherited_median"]

    def dbpc(tag: str) -> float:
        return agg[tag]["delta_bpc_fresh_mean"]

    preds = {
        "U1": {"claim": "median horizon A(16) <= 27",
               "measured": med("A(16)"), "threshold": 27,
               "held": med("A(16)") is not None and med("A(16)") <= 27},
        "U2": {"claim": "median horizon B(16) >= 27",
               "measured": med("B(16)"), "threshold": 27,
               "held": med("B(16)") is not None and med("B(16)") >= 27},
        "U3": {"claim": "|median horizon A(32) - A(512)| <= 3",
               "measured": (None if med("A(32)") is None or med("A(512)") is None
                            else abs(med("A(32)") - med("A(512)"))),
               "threshold": 3,
               "held": (med("A(32)") is not None and med("A(512)") is not None
                        and abs(med("A(32)") - med("A(512)")) <= 3)},
        "U4": {"claim": "A(16) costs >= 0.05 bpc fresh",
               "measured": dbpc("A(16)"), "threshold": 0.05,
               "held": dbpc("A(16)") >= 0.05},
        "U5": {"claim": "median horizon C(16) >= 38",
               "measured": med("C(16)"), "threshold": 38,
               "held": med("C(16)") is not None and med("C(16)") >= 38},
        "U6": {"claim": "median horizon E(16) >= 38",
               "measured": med("E(16)"), "threshold": 38,
               "held": med("E(16)") is not None and med("E(16)") >= 38},
    }
    out["predictions"] = preds

    # N-half: the smallest swept N at which half the horizon gain is gone.
    intact, base = med("A(0)"), 7.0
    half = (intact + base) / 2 if intact is not None else None
    n_half = None
    if half is not None:
        for _, n in [c for c in CONFIGS if c[0] == "A"]:
            m = med(f"A({n})")
            if m is not None and m <= half:
                n_half = n
                break
    out["n_half"] = {"intact_median": intact, "baseline_committed": base,
                     "half_point": half, "smallest_swept_N_at_or_below": n_half}
    out["self_check_failures"] = failures

    dest = _REPO / args.out
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=2), encoding="utf-8")

    print("\n" + "=" * 78)
    print("EXP_006 -- horizon and fresh bpc under beta_s ablation "
          f"(n = {len(out['seeds'])} seeds)")
    print("=" * 78)
    print("config    horizon(2s=0.00922)  horizon(0.00627)   fresh bpc   d vs A(0)")
    for tag, a in agg.items():
        print(f"{tag:9s} {str(a['horizon_inherited_median']):>10s}  "
              f"{str(a['horizon_inherited_per_seed']):<28s} "
              f"{str(a['horizon_family_median']):>6s}   "
              f"{a['bpc_fresh_mean']:.5f}  {a['delta_bpc_fresh_mean']:+.5f}")
    print("\nPRE-REGISTERED PREDICTIONS")
    for k, p in preds.items():
        m = p["measured"]
        print(f"  {k}  {p['claim']:42s} measured "
              f"{('%.5f' % m) if isinstance(m, float) else m:>10}   "
              f"{'held' if p['held'] else 'FAILED'}")
    print(f"\n  N-half (smallest swept N at or below "
          f"{out['n_half']['half_point']}): {n_half}")
    print("\nSELF-CHECKS")
    for name in ("H1", "H2", "H3", "H4", "H5"):
        rows = out["self_checks"][name]
        ok = sum(1 for r in rows
                 if r.get("pass", r.get("indices_match", False)))
        print(f"  {name}: {ok}/{len(rows)} passed")
    if out["self_checks"]["H2"]:
        worst = max(r["residual_bpc"] for r in out["self_checks"]["H2"])
        allid = all(r["nats_identical"] for r in out["self_checks"]["H2"])
        print(f"       H2 worst k=1 residual {worst:.3e} bpc; "
              f"nats bit-identical on every config: {allid}")
    if failures:
        print("\n  SELF-CHECK FAILURES:")
        for f in failures:
            print(f"    {f}")
    print(f"\nWROTE {args.out}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
