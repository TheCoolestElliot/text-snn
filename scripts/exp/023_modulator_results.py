"""EXP_023 resolver: applies §4's bars mechanically to §3's runs.

    python scripts/exp/023_modulator_results.py --out docs/reports/data/exp_023_arm_results.json

COMMITTED BEFORE ANY LEG'S BPC IS READ.

WHICH TEST, AND WHY IT IS DECIDED HERE RATHER THAN IN §9
---------------------------------------------------------
A seed fixes both the initial weights and the data order, so arms sharing seeds
are PAIRED, and `EXP_018` D1 is this project's own record of what happens when
that is got wrong: an unpaired Welch test on paired data resolved UNRESOLVED by
3.6e-06 bpc, and the corrected paired test -- run because it hurt -- found the
arm cost 0.0062 bpc on 5 of 5 seeds.

This ladder has cells at n = 6 and cells at n = 3, so both tests appear, and
which one applies to which bar is fixed in §4 and encoded here:

    H2, H5   Welch, unequal n           (n = 6 arm against an n = 3 rung)
    H1       Welch                      (n = 3 rung against the pooled anchor)
    H3       paired on seeds 0,1,2
    H4       paired on all six seeds    -- EXP_021 §11 item 2's measurement
    H6       paired on seeds 3,4,5 ONLY -- the out-of-sample replication

**H6 IS PRINTED FIRST AND SEPARATELY.** A replication and a power increase are
not the same number, and pooling them would make the one that could fail
invisible inside the one that cannot.

WHAT IT REFUSES TO DO
---------------------
  * It does not adopt, rank, or recommend. `nm_const` is a RUNG OF THIS LADDER
    and is never reported as a re-measurement of adopted arm #5 -- §6 item 1
    says why that distinction is real and where it is thin, and #6 is open.
  * It does not repair a bar that misfired; a corrected rule is referred.
  * It does not substitute a seed.
  * It does not compare any number with a committed figure (decision #10).
  * It does not use `n_contexts_significantly_worse` as a bar (EXP_020 §9.5).
  * **It does not read a train-side number from `log.jsonl`.** `EXP_018` §10
    item 11 refers exactly that, and `EXP_021` §9.5 is the second experiment it
    caught: for a weighted objective the last logged loss is not a train bpc.
    Any train-side figure here comes from `013_generalisation_gap.py`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO / "src") not in sys.path:
    sys.path.insert(0, str(_REPO / "src"))

import torch  # noqa: E402

from snn.dopamine import roll_across_batch  # noqa: E402


def _import_010():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "_r010", _REPO / "scripts" / "exp" / "010_phase4_arm_results.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


RUNS = _REPO / "experiments" / "runs"
PREREG = _REPO / "experiments" / "logs" / "EXP_023_modulator_ladder.md"
MANIFEST = _REPO / "docs" / "reports" / "data" / "exp_023_run_manifest.json"

SIGMA_TRANSFERRED = 0.00461
BAR_2SIGMA = 2 * SIGMA_TRANSFERRED

POOLED = (0, 1, 2, 3, 4, 5)
FRESH = (3, 4, 5)
NEW = (0, 1, 2)

NM_GAIN_INIT = 0.1

#: The ladder, in order of INCREASING signal content. The order is the finding:
#: §4 H7 says that if `rms(kappa)` is flat across it, the optimiser is buying
#: the FORM and not the signal, whichever way the bpc goes.
LADDER = ("nm_const", "nm_pos", "nm_rolled", "nm_local")

ARMS = {
    "if_anchor": [f"if_anchor_s{s}" for s in POOLED],
    "nm_const":  [f"nm_const_s{s}" for s in NEW],
    "nm_pos":    [f"nm_pos_s{s}" for s in NEW],
    "nm_rolled": [f"nm_rolled_s{s}" for s in POOLED],
    "nm_local":  [f"nm_local_s{s}" for s in POOLED],
}

#: What each rung's DA knows. Printed with every table so a reader never has to
#: reconstruct the ladder from run names.
KNOWS = {
    "nm_const": "nothing (a static per-channel gain = EXP_008 Identity 1)",
    "nm_pos": "WHEN only -- window position. DIAGNOSTIC ONLY, non-deployable",
    "nm_rolled": "when, and what surprise looks like in general",
    "nm_local": "when, and what surprised THIS sequence",
}


def _final(run: str) -> dict | None:
    p = RUNS / run / "final_test.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def _bpc(run: str, protocol: str) -> float | None:
    j = _final(run)
    if j is None:
        return None
    v = j.get("results", {}).get(protocol, {}).get("bpc")
    return None if v is None else float(v)


def _summary(run: str) -> dict:
    p = RUNS / run / "summary.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def _diverged(run: str) -> bool:
    p = RUNS / run / "log.jsonl"
    if not p.exists():
        return True
    for line in p.read_text(encoding="utf-8").splitlines():
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        v = rec.get("loss")
        if v is not None and (v != v or v in (float("inf"), float("-inf"))):
            return True
    b = _bpc(run, "fresh")
    return b is None or b != b


def arm_stats(runs: list[str], protocol: str,
              gate_failed: set[str] | None = None) -> dict:
    """One cell's bpc, with gate failures handled the way divergences are.

    `gate_failed` is §7 G8's second clause made mechanical: *"if any `nm_pos`
    seed finishes with `nm_gain` still at its 0.1 init, that run is reported as
    a failed unlock and **its bpc is not read as an arm result**."* Written into
    the pre-registration, and until now recorded by the driver and ignored here
    -- so the gate said one thing and the resolver did another.

    A gate-failed run is treated EXACTLY as a diverged one: dropped from the
    cell, named in the artifact, and the cell marked provisional. It is never
    substituted, per `CONTRIBUTING.md` §4.
    """
    gate_failed = gate_failed or set()
    vals, dead, kept, gated = [], [], [], []
    for r in runs:
        if r in gate_failed:
            gated.append(r)
            continue
        if _diverged(r):
            dead.append(r)
            continue
        b = _bpc(r, protocol)
        if b is None:
            dead.append(r)
        else:
            vals.append(b)
            kept.append(r)
    return {"runs": runs, "kept": kept, "n": len(vals),
            "n_preregistered": len(runs), "diverged": dead,
            "gate_failed_G8": gated,
            "provisional": len(vals) < len(runs), "bpc": vals,
            "mean": statistics.fmean(vals) if vals else None,
            "sd": statistics.stdev(vals) if len(vals) > 1 else None}


def paired(a: dict, b: dict) -> dict:
    if a["n"] != a["n_preregistered"] or b["n"] != b["n_preregistered"]:
        return {"paired": False,
                "reason": "a cell lost a seed; pairing across unequal n would "
                          "silently drop the partner"}
    if len(a["bpc"]) != len(b["bpc"]):
        return {"paired": False,
                "reason": f"unequal n ({len(a['bpc'])} vs {len(b['bpc'])}); "
                          "this comparison is Welch by §4 and not paired"}
    d = [x - y for x, y in zip(a["bpc"], b["bpc"])]
    n = len(d)
    m = statistics.fmean(d)
    sd = statistics.stdev(d) if n > 1 else None
    se = (sd / math.sqrt(n)) if sd else None
    return {"paired": True, "diffs": d, "mean_diff": m, "sd_diff": sd,
            "se_diff": se, "t": (m / se) if se else None,
            "n_negative": sum(1 for x in d if x < 0)}


def welch(a: dict, b: dict) -> dict:
    """Two-sample Welch. Used ONLY where §4 says so -- cells of unequal n.

    Named rather than defaulted: `EXP_018` D1's error was reaching for this test
    on data that was paired, and the way to not repeat it is for every call site
    to be a §4 citation.
    """
    xa, xb = a["bpc"], b["bpc"]
    if len(xa) < 2 or len(xb) < 2:
        return {"welch": False, "reason": "a cell has fewer than 2 seeds"}
    ma, mb = statistics.fmean(xa), statistics.fmean(xb)
    va, vb = statistics.variance(xa), statistics.variance(xb)
    se = math.sqrt(va / len(xa) + vb / len(xb))
    return {"welch": True, "mean_diff": ma - mb, "se_diff": se,
            "t": ((ma - mb) / se) if se else None,
            "n": [len(xa), len(xb)]}


def subset(stats_runs: list[str], protocol: str, seeds: tuple) -> dict:
    """One cell restricted to a seed set -- H6's fresh-seeds-only replication."""
    want = {f"_s{s}" for s in seeds}
    return arm_stats([r for r in stats_runs
                      if any(r.endswith(w) for w in want)], protocol)


def gain_rms(run: str) -> float | None:
    """rms of `nm_gain` from the run's own final checkpoint -- H7's quantity.

    `EXP_018`'s reusable result was exactly this measurement on the analogous
    parameter: 0.195 aligned against 0.0075 rolled, a factor of 26. `EXP_021`
    got 1.077 / 0.958 -- a factor of 1.12. This ladder puts four rungs on the
    same axis, and §4 says a flat column here IS the headline whichever way the
    bpc goes.
    """
    p = RUNS / run / "ckpt_final.pt"
    if not p.exists():
        return None
    ck = torch.load(p, map_location="cpu", weights_only=False)
    sd = ck.get("model", ck)
    vals = [v.float().pow(2).mean() for k, v in sd.items()
            if k.startswith("nm_gain")]
    if not vals:
        return None
    return float(torch.stack(vals).mean().sqrt())


def da_stats(run: str, n_windows: int = 32, batch: int = 8) -> dict | None:
    """The DRIVING SIGNAL's own distribution, per rung. `[mean, sd, saturation]`.

    ADDED BEFORE ANY EXP_023 BPC WAS READ, and it exists because H7 cannot be
    interpreted without it. H7 compares `rms(kappa)` across rungs, but the
    modulation is `1 + kappa_c * DA_t`, so a rung whose `DA` barely varies has a
    STATIC per-channel gain however large its `kappa` is -- and by `EXP_008`
    Identity 1 a static per-channel gain on the current is a learned per-channel
    THRESHOLD, i.e. adopted arm #5. `rms(kappa)` alone cannot tell those apart.

    **Measured on EXP_021's own committed checkpoints before this experiment
    ran:** `nm_local` has `tau` learned down to 0.0071-0.0080 from its
    calibrated 0.057321, `mean(DA) = +0.86`, `sd(DA) = 0.23`; `nm_rolled` has
    `tau` at 0.0038-0.0039, `mean(DA) = +0.996` and **`sd(DA) = 0.044`, with
    98.9 % of positions at `|DA| > 0.99`**. So the misaligned control
    degenerated into a static gain and the aligned arm did not -- a 5.2x
    difference in `sd(DA)` that `rms(kappa)` reports as 1.12.

    That is the mechanism behind `EXP_021`'s H2 (UNRESOLVED at -0.00577) and H3
    (ratio 1.12 against `EXP_018`'s 26): the control was not a misaligned-RPE
    arm, it was a learned-gain arm, so the near-tie was structural rather than
    evidence about alignment. `EXP_021` is not retracted -- its bars fired as
    written -- but this is what its §11 item 5 asked for.

    CPU by construction (`cfg.device = "cpu"`), because a timed run may be in
    flight and nothing else may touch the GPU. 32 windows is ample: the
    quantities above separate by 5x.

    Returns None for a rung that builds no `DA` (`if_anchor`), or if the run is
    missing.
    """
    d = RUNS / run
    if not (d / "ckpt_final.pt").exists() or not (d / "config.json").exists():
        return None
    from snn.config import Config
    from snn.data import Corpus, windowed_eval_batches
    from snn.model import build_model
    from snn.neuromod import bernoulli_rpe

    cfg = Config.from_dict(json.loads((d / "config.json").read_text(encoding="utf-8")))
    if cfg.arch != "localdopamine" or cfg.nm_source == "off":
        return None
    cfg.device = "cpu"
    corpus = Corpus(cfg.corpus, "data")
    cfg.vocab_size = corpus.vocab_size
    model = build_model(cfg)
    ck = torch.load(d / "ckpt_final.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(ck.get("model", ck))
    model.eval()
    model.keep_spikes = True

    vals, taken = [], 0
    with torch.no_grad():
        for x, _ in windowed_eval_batches(corpus.split("val"), batch, cfg.seq_len):
            _, _, aux = model(x, None)
            # Rebuild DA the way the forward does, per rung, from layer 0's
            # emission -- `_driving_da` is not reused because it needs `h` in the
            # forward's own dtype and this must not depend on a private method.
            h = aux["spikes"][0].float()
            if cfg.nm_source == "const":
                da = torch.ones(h.shape[0], h.shape[1])
            elif cfg.nm_source == "pos":
                da = torch.tanh(model.nm_pos[0][:, :h.shape[1]]).expand(
                    h.shape[0], h.shape[1])
            else:
                phi = bernoulli_rpe(h, model.nm_a[0], model.nm_b[0])
                da = torch.tanh(phi / torch.exp(model.nm_log_tau[0]))
                if cfg.nm_source == "rolled":
                    da = roll_across_batch(da)
            vals.append(da.reshape(-1))
            taken += x.shape[0]
            if taken >= n_windows:
                break
    if not vals:
        return None
    da = torch.cat(vals)
    tau = (float(torch.exp(model.nm_log_tau[0]).detach())
           if len(model.nm_log_tau) else None)
    return {
        "nm_source": cfg.nm_source,
        "tau_learned": None if tau is None else round(tau, 6),
        "tau_calibrated": 0.057321,
        "mean": round(float(da.mean()), 5),
        # THE quantity: the modulation's time-variation is kappa_c * sd(DA), so
        # sd(DA) ~ 0 means a static per-channel gain whatever kappa is.
        "sd": round(float(da.std()), 5),
        "frac_abs_over_0p99": round(float((da.abs() > 0.99).float().mean()), 5),
        "frac_abs_over_0p9": round(float((da.abs() > 0.9).float().mean()), 5),
        "n": int(da.numel()),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--protocol", default="carried", choices=("fresh", "carried"))
    ap.add_argument("--out", default="docs/reports/data/exp_023_arm_results.json")
    ap.add_argument("--no-da-stats", action="store_true",
                    help="skip the CPU forward passes that measure sd(DA)")
    args = ap.parse_args(argv)

    out: dict = {
        "experiment": "EXP_023", "protocol": args.protocol,
        "sigma_transferred": SIGMA_TRANSFERRED,
        "sigma_source": "04_phase4_interim.md §236, baseline n=5, DIFFERENT tree",
        "bar_2sigma": BAR_2SIGMA,
        "prereg_sha256": hashlib.sha256(PREREG.read_bytes()).hexdigest(),
        "ladder_knows": KNOWS,
    }
    if MANIFEST.exists():
        man = json.loads(MANIFEST.read_text(encoding="utf-8"))
        out["manifest_prereg_sha256_before"] = man.get("prereg_sha256_before")
        out["prereg_matches_manifest"] = (
            man.get("prereg_sha256_before") == out["prereg_sha256"])
        out["pooled_with_exp_021"] = man.get("pooled_with_exp_021")
        # G2 is asserted over EVERY run. `all()` of an empty generator is True,
        # so filtering on `"G2_params_match" in r` made this gate PASS on a
        # manifest with no runs and DROP a run with no `summary.json` -- the
        # case where the parameter count is most in doubt -- instead of failing
        # it. The denominator travels with the verdict (CONTRIBUTING.md §3).
        _g2_runs = man.get("runs", [])
        _g2_seen = [r for r in _g2_runs if "G2_params_match" in r]
        out["G2_all_params_match"] = bool(_g2_runs) and len(_g2_seen) == len(_g2_runs) \
            and all(r["G2_params_match"] for r in _g2_seen)
        out["G2_denominator"] = {"runs_in_manifest": len(_g2_runs),
                                 "runs_carrying_the_gate": len(_g2_seen),
                                 "expected": 15}
        out["G2_missing_the_gate"] = [r.get("run") for r in _g2_runs
                                      if "G2_params_match" not in r]
        # §2.2's closed forms, CHECKED here rather than trusted from the driver.
        # An arm whose realised count is described by a closed form that is not
        # its own is the failure `local_dopamine_param_count`'s docstring records
        # being caught at, one digit, and G2 in the driver is the only other
        # place it is tested.
        out["G2_closed_form_check"] = {
            r["run"]: {"params": r.get("params"),
                       "expected": r.get("expected_params"),
                       "matches": r.get("params") == r.get("expected_params")}
            for r in _g2_runs}
        out["G2_closed_form_all_match"] = bool(_g2_runs) and all(
            v["matches"] for v in out["G2_closed_form_check"].values())
        out["G5_violations"] = {r["run"]: r["G5_frozen_violations"]
                                for r in man.get("runs", [])
                                if r.get("G5_frozen_violations")}
        out["K1_unexpected"] = {r["run"]: r["K1_unexpected"]
                                for r in man.get("runs", [])
                                if r.get("K1_unexpected")}
        # G8's second clause, read back off the manifest rather than recomputed.
        out["G8_pos_unlocked"] = {r["run"]: r.get("G8_pos_unlocked")
                                  for r in man.get("runs", [])
                                  if "G8_pos_unlocked" in r}

    # G8's second clause, applied rather than merely recorded. Read from the
    # manifest the driver wrote, so the resolver and the driver cannot disagree
    # about which runs unlocked.
    g8_failed = {r for r, ok in (out.get("G8_pos_unlocked") or {}).items()
                 if ok is False}
    if g8_failed:
        print()
        print(f"  !! G8: {sorted(g8_failed)} finished with nm_gain at its "
              f"{NM_GAIN_INIT} init. Their bpc is NOT read as an arm result "
              "(§7 G8), the nm_pos cell is PROVISIONAL, and any bar resting "
              "on it is reported at reduced n.")
        print()
    out["G8_failed_runs_excluded"] = sorted(g8_failed)

    stats = {k: arm_stats(v, args.protocol, g8_failed) for k, v in ARMS.items()}
    out["arms"] = stats

    # The on-tree sigma this project has never had at n = 6. A MARKER: the bars
    # above stay on the transferred sigma, exactly as EXP_020/021 used them.
    out["sigma_on_tree_n6"] = {
        "anchor_sd": stats["if_anchor"]["sd"],
        "n": stats["if_anchor"]["n"],
        "transferred": SIGMA_TRANSFERRED,
        "ratio": (None if not stats["if_anchor"]["sd"]
                  else stats["if_anchor"]["sd"] / SIGMA_TRANSFERRED),
        "note": "EXP_020 measured 0.004683 at n = 3 on this tree, a x1.02 "
                "transfer. This is the same check at n = 6. It is a marker and "
                "does not move any bar in this experiment.",
    }

    bars: dict = {}

    def _register(key, arm, ref, test, statement, note=""):
        a, b = stats[arm], stats[ref]
        delta = (None if a["mean"] is None or b["mean"] is None
                 else a["mean"] - b["mean"])
        detail = paired(a, b) if test == "paired" else welch(a, b)
        bars[key] = {"statement": statement, "note": note, "test": test,
                     "arm": arm, "reference": ref, "delta": delta,
                     "detail": detail,
                     "verdict": ("NOT RUN" if delta is None else
                                 "HELD" if delta < -BAR_2SIGMA else
                                 "FAILED" if delta > BAR_2SIGMA else
                                 "UNRESOLVED")}

    _register("H1", "nm_const", "if_anchor", "welch",
              f"mean(nm_const) < mean(if_anchor) - {BAR_2SIGMA}",
              "the rung that could deflate EXP_021's headline: a constant DA "
              "makes this a learned per-channel gain, which EXP_008 Identity 1 "
              "makes a learned threshold. NOT a re-measurement of arm #5 -- §6 "
              "item 1, and #6 is open.")
    _register("H2", "nm_local", "nm_const", "welch",
              f"mean(nm_local) < mean(nm_const) - {BAR_2SIGMA}",
              "does time-variation buy ANYTHING over a static gain?")
    _register("H3", "nm_pos", "nm_const", "paired",
              f"mean(nm_pos) < mean(nm_const) - {BAR_2SIGMA}",
              "is the time-variation positional? window position predicts "
              "surprise and no rung below `pos` can use it")
    _register("H4", "nm_local", "nm_rolled", "paired",
              f"mean(nm_local) < mean(nm_rolled) - {BAR_2SIGMA}",
              "EXP_021 §11 item 2's measurement, at the n = 6 it priced: "
              "-0.00577 with paired se 0.0018 at n = 3")
    _register("H5", "nm_local", "nm_pos", "welch",
              f"mean(nm_local) < mean(nm_pos) - {BAR_2SIGMA}",
              "does data dependence beat position? H4's question against a "
              "control that cannot leak content at all")

    # H6 -- the out-of-sample replication, on FRESH seeds only. Computed on its
    # own subsets so that it can fail while the pooled estimate holds.
    a6 = subset(ARMS["nm_local"], args.protocol, FRESH)
    b6 = subset(ARMS["if_anchor"], args.protocol, FRESH)
    d6 = (None if a6["mean"] is None or b6["mean"] is None
          else a6["mean"] - b6["mean"])
    bars["H6"] = {
        "statement": f"mean(nm_local s3-5) < mean(if_anchor s3-5) - {BAR_2SIGMA}",
        "note": "OUT-OF-SAMPLE replication of EXP_021's headline (-0.03371). "
                "Reported FIRST and separately: a replication and a power "
                "increase are not the same number.",
        "test": "paired, fresh seeds only", "delta": d6,
        "exp_021_headline_for_reference": -0.03371,
        "arm": a6, "reference": b6, "detail": paired(a6, b6),
        "verdict": ("NOT RUN" if d6 is None else
                    "HELD" if d6 < -BAR_2SIGMA else
                    "FAILED" if d6 > BAR_2SIGMA else "UNRESOLVED")}

    # H7 -- findability across the whole ladder.
    rms = {arm: [gain_rms(r) for r in ARMS[arm]] for arm in LADDER}
    # sd(DA) per rung -- without it H7's rms(kappa) column cannot distinguish a
    # time-varying gain from a static one. See `da_stats`.
    das = ({arm: da_stats(ARMS[arm][0]) for arm in LADDER}
           if not args.no_da_stats else {})
    means = {arm: (statistics.fmean([v for v in vs if v is not None])
                   if any(v is not None for v in vs) else None)
             for arm, vs in rms.items()}
    live = [v for v in means.values() if v]
    bars["H7"] = {
        "statement": "rms(nm_gain) per rung; a FLAT column means the optimiser "
                     "buys the FORM and not the signal",
        "per_run": rms, "per_arm_mean": means,
        "max_over_min_ratio": (max(live) / min(live) if len(live) > 1 and
                               min(live) > 0 else None),
        "exp_018_ratio_on_the_same_quantity": 26.0,
        "exp_021_ratio_on_the_same_quantity": 1.12,
        "gain_init": NM_GAIN_INIT,
        "da_stats_seed0": das,
        "da_note": "the modulation is 1 + kappa_c*DA_t, so sd(DA) ~ 0 is a "
                   "STATIC per-channel gain however large kappa is -- EXP_008 "
                   "Identity 1, i.e. adopted arm #5. rms(kappa) alone cannot "
                   "tell those apart, which is why this column exists.",
        "note": "a marker, not a bar. EXP_021 §2.7 named the FAINT "
                "reachability ratio in advance as the risk that would surface "
                "here, and it did.",
    }
    out["bars"] = bars

    # -- cost, H9 --------------------------------------------------------------
    ref_w = statistics.fmean(
        [w for w in (_summary(r).get("wall_clock_s") for r in ARMS["if_anchor"])
         if w] or [1.0])
    out["cost"] = {
        arm: {"wall_clock_ratio_vs_if_anchor": round(
                  statistics.fmean([w for w in
                                    (_summary(r).get("wall_clock_s") for r in runs)
                                    if w] or [0.0]) / ref_w, 3),
              "peak_vram_gib": [_summary(r).get("peak_vram_gib") for r in runs],
              "params": [_summary(r).get("params") for r in runs]}
        for arm, runs in ARMS.items()
    }

    # -- the decomposition CONTRIBUTING.md §3 requires before a gain is quoted --
    hzp = _REPO / "docs" / "reports" / "data" / "exp_023_memory_horizon.json"
    if not hzp.exists():
        # H8 is §4's decomposition marker. An absent horizon artifact means it
        # was NOT MEASURED, and the artifact says so in words rather than
        # leaving a reader to infer it from an empty table.
        out["decomposition"] = {
            "measured": False,
            "artifact": str(hzp.relative_to(_REPO)),
            "H8": "NOT RUN — the horizon probe has not been run over these "
                  "checkpoints; run scripts/exp/001_memory_horizon.py first",
        }
    else:
        hz = json.loads(hzp.read_text(encoding="utf-8"))
        r010 = _import_010()
        ac = hz["absolute_context_curves"]
        _p2s = hz.get("per_context_2sigma", {})
        # The wrapper: bars are one level down. Passing the wrapper silently
        # falls back to the flat bar -- EXP_020 §9.8 item 3.
        ctx_bars = _p2s.get("bars", _p2s) if isinstance(_p2s, dict) else {}
        # NAME THE GAPS rather than skipping them. Every rung is differenced
        # against `if_anchor`, so an absent reference takes the WHOLE ladder with
        # it, and a bare `continue` would leave `measured: true` beside an empty
        # table -- EXP_020's stale-artifact defect from the other side. The 022
        # resolver names its gaps for the same reason.
        missing = [a for a in ("if_anchor",) + tuple(LADDER) if a not in ac]
        dec = {}
        if "if_anchor" not in ac:
            out["decomposition"] = {
                "measured": False,
                "artifact": str(hzp.relative_to(_REPO)),
                "arms_missing_from_horizon": missing,
                "H8": "NOT RUN — `if_anchor` is absent from the horizon "
                      "artifact and every rung is differenced against it",
            }
        else:
            ref = ac["if_anchor"]["bpc_at_context"]
            for arm in LADDER:
                if arm not in ac:
                    continue
                dec[arm] = r010.decompose(ref, ac[arm]["bpc_at_context"])
                dec[arm]["reference"] = "if_anchor"
                dec[arm]["context_comparison"] = r010.context_comparison(
                    ac[arm]["bpc_at_context"], ref, ctx_bars)
                # G8's second clause reaches the bpc cells through `arm_stats`,
                # but the horizon artifact aggregates every seed of an arm, so a
                # G8-failed run's curve is INSIDE this row and cannot be removed
                # here. Say so on the row rather than letting a caveat that
                # applies to one table silently not apply to its neighbour.
                if arm == "nm_pos" and g8_failed:
                    dec[arm]["G8_caveat"] = (
                        f"{sorted(g8_failed)} failed G8's unlock clause and are "
                        "excluded from this arm's bpc cells, but the horizon "
                        "artifact aggregates all seeds, so this decomposition "
                        "row STILL CONTAINS THEM. It is not a clean arm result.")
                    dec[arm]["G8_contaminated"] = True
            out["decomposition"] = {
                "measured": True, "arms": dec,
                "arms_missing_from_horizon": missing,
                "complete": not missing,
                "note": "EXP_020 §9.5's bitwise-copy noise floor: -0.00499 total, "
                        "-0.01395 at c=0, +0.00944 within reach. EXP_021 found "
                        "nm_local's gain ENTIRELY within reach (+0.0559) and "
                        "nothing beyond the horizon (-0.0005); §4 H8 predicts a "
                        "rung that knows less keeps that shape and loses the rest."}

    p = _REPO / args.out
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")

    print(f"\nEXP_023 — protocol {args.protocol}, "
          f"2 sigma = {BAR_2SIGMA} (TRANSFERRED, different tree)\n")

    # The gates, PRINTED. A gate whose result only reaches the JSON is a gate a
    # reader of this summary never sees -- and G2 in particular used to be
    # computed and then never mentioned.
    _g2d = out.get("G2_denominator") or {}
    print(f"  gates: prereg_matches_manifest={out.get('prereg_matches_manifest')}"
          f"  G2={out.get('G2_all_params_match')} "
          f"({_g2d.get('runs_carrying_the_gate')}/{_g2d.get('runs_in_manifest')} "
          f"runs, expected {_g2d.get('expected')})"
          f"  G2_closed_form={out.get('G2_closed_form_all_match')}")
    if out.get("G5_violations") or out.get("K1_unexpected"):
        print(f"    !! G5={out.get('G5_violations')}  "
              f"K1={out.get('K1_unexpected')}")
    _decomp_measured = (out.get("decomposition") or {}).get("measured")
    if not _decomp_measured:
        print("    !! DECOMPOSITION NOT MEASURED — H8 is NOT RUN. Run "
              "scripts/exp/001_memory_horizon.py before this resolver.")
    print()

    # H6 FIRST, as the header of this file requires.
    h6 = bars["H6"]
    dd = h6["detail"]
    print(f"  H6 (REPLICATION, fresh seeds 3-5 only): {h6['verdict']}")
    print(f"      delta={h6['delta']}   EXP_021 headline was "
          f"{h6['exp_021_headline_for_reference']}")
    if dd.get("paired"):
        # `t` is None when the paired sd is exactly zero (identical differences)
        # or when a cell holds one seed. Printed as "--" rather than crashing:
        # a resolver that dies on a degenerate statistic loses the whole ladder
        # after it has already been paid for.
        tt = "--" if dd.get("t") is None else f"{dd['t']:.2f}"
        print(f"      paired t={tt}  "
              f"{dd['n_negative']}/{len(dd['diffs'])} seeds negative")
    print()

    for arm, s in stats.items():
        print(f"  {arm:10s} n={s['n']}/{s['n_preregistered']} mean={s['mean']} "
              f"sd={s['sd']} "
              f"{'PROVISIONAL ' + str(s['diverged']) if s['provisional'] else ''}")

    print("\n  the ladder, in order of INCREASING signal content:")
    print(f"    {'rung':10s} {'mean bpc':>10s} {'vs anchor':>10s} "
          f"{'rms(kappa)':>11s} {'sd(DA)':>8s}   what DA knows")
    for arm in LADDER:
        m = stats[arm]["mean"]
        d = (None if m is None or stats["if_anchor"]["mean"] is None
             else m - stats["if_anchor"]["mean"])
        g = bars["H7"]["per_arm_mean"].get(arm)
        ds = (das or {}).get(arm) or {}
        sdda = ds.get("sd")
        print(f"    {arm:10s} {(m if m is not None else float('nan')):10.5f} "
              f"{(d if d is not None else float('nan')):+10.5f} "
              f"{(g if g is not None else float('nan')):11.4f} "
              f"{(sdda if sdda is not None else float('nan')):8.4f}   {KNOWS[arm]}")
    r = bars["H7"]["max_over_min_ratio"]
    print(f"    H7 max/min rms(kappa) = {r}  "
          f"(EXP_018 got 26 on this quantity, EXP_021 got 1.12)")

    print()
    for k in ("H1", "H2", "H3", "H4", "H5"):
        b = bars[k]
        dt = b["detail"]
        extra = ""
        if dt.get("paired") and dt.get("t") is not None:
            extra = (f"  paired t={dt['t']:.2f}  "
                     f"{dt['n_negative']}/{len(dt['diffs'])} seeds negative")
        elif dt.get("welch") and dt.get("t") is not None:
            extra = f"  Welch t={dt['t']:.2f}  n={dt['n']}"
        print(f"  {k}: {b['verdict']:12s} delta={b['delta']}{extra}")
        print(f"      {b['statement']}")

    if out["decomposition"].get("measured"):
        print("\n  EXP_004 §10.3 decomposition (positive = arm BETTER), "
              "cut at the baseline horizon 7:")
        print(f"    {'rung':11s} {'total':>9s} {'c=0':>9s} {'within':>9s} "
              f"{'beyond':>9s}")
        for arm, d in out["decomposition"]["arms"].items():
            print(f"    {arm:11s} {d['total_gain_bpc']:+9.5f} "
                  f"{d['zero_context_gain_bpc']:+9.5f} "
                  f"{d['within_reach_gain_bpc']:+9.5f} "
                  f"{d['beyond_horizon_gain_bpc']:+9.5f}")
        print("    (EXP_020's bitwise-identical control reads -0.00499 / "
              "-0.01395 / +0.00944 / -0.00047)")

    print(f"\nWROTE {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
