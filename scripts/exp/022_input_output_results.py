"""EXP_022 resolver: applies §4's bars mechanically to §3's runs.

    python scripts/exp/022_input_output_results.py --out docs/reports/data/exp_022_arm_results.json

COMMITTED BEFORE ANY LEG'S BPC IS READ.

Every bar here is on a **paired per-seed difference**, and that is not a style
choice: `EXP_018` D1 used an unpaired Welch test on paired data, resolved
UNRESOLVED by 3.6e-06 bpc, and the corrected paired test -- run because it hurt
-- found the arm cost 0.0062 bpc on 5 of 5 seeds. A seed fixes both the initial
weights and the data order, so arms sharing seeds are paired. The per-seed signs
are reported next to every mean.

WHAT IT REFUSES TO DO
---------------------
  * It does not adopt, rank, or recommend.
  * It does not repair a bar that misfired; a corrected rule is referred.
  * It does not substitute a seed.
  * It does not compare any number with a committed figure (decision #10).
  * **It does not use `n_contexts_significantly_worse` as a bar.** `EXP_020`
    §9.5 measured that statistic against a BITWISE copy of the anchor and it
    read 120 of 128 contexts "significantly worse" -- bars up to 80x below
    single-seed variation at n = 3. It is printed as a marker and nothing here
    turns on it.

THE THREE COMPARISONS AND WHY THEY ARE DIFFERENT SHAPES
---------------------------------------------------------
  leg A   each sparse rung against `if_binin`, the q = 0.50 rung of its OWN
          ladder. Parameter-identical (735,437 at all four rungs), so this is a
          single-variable comparison in the strongest sense the protocol has.
  leg B   `io_mall` against `if_anchor` -- NOT single-variable, it carries
          14.3 % more parameters, and §6 item 5 says so;
          `io_mall` against `io_wide` -- the same 104,960 extra parameters spent
          on a wider head against a wider stack. **That is the decisive bar**,
          and it is the only form of "is the readout worth it" that `EXP_014`
          leaves open.
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


def _import_010():
    """`decompose` and `context_comparison`, IMPORTED not reimplemented.

    `EXP_011`'s resolver set the precedent: one implementation of each
    statistic, not one file. `010_phase4_arm_results.py` is committed and closed,
    so it is imported by path rather than edited.
    """
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "_r010", _REPO / "scripts" / "exp" / "010_phase4_arm_results.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


RUNS = _REPO / "experiments" / "runs"
PREREG = _REPO / "experiments" / "logs" / "EXP_022_input_output.md"
MANIFEST = _REPO / "docs" / "reports" / "data" / "exp_022_run_manifest.json"


def gain_is_at_zero_context(zero: float, total: float,
                            share: float = 0.5) -> bool:
    """H4's post-hoc replacement statistic. Module level SO IT CAN BE TESTED.

    `zero` and `total` come from `010.decompose`, whose contract is
    **positive = the arm is better** (`total = ref_curve[CMAX] - arm_curve[CMAX]`).
    A GAIN is therefore `total > 0`, and this was written `total < 0` -- the
    file's local "negative = better" delta convention carried into the one block
    that consumes the opposite one. Inverted, it was True only when a rung was
    WORSE, and False for every genuine recovery of `if_binin`'s cost.

    Kept as a function rather than an inline expression because the inline
    version is what let the sign go unobserved: nothing could assert on it.
    `tests/test_exp022_resolver.py` now does.
    """
    return total > 0 and zero > 0 and abs(zero) >= share * abs(total)


def loss_is_at_zero_context(zero: float, total: float,
                            share: float = 0.5) -> bool:
    """The mirror of `gain_is_at_zero_context`, so a rung that moves the wrong
    way is reported rather than merely absent from the gain column."""
    return total < 0 and zero < 0 and abs(zero) >= share * abs(total)

SIGMA_TRANSFERRED = 0.00461
BAR_2SIGMA = 2 * SIGMA_TRANSFERRED

SEEDS = (0, 1, 2)

#: `if_anchor` and `if_binin` are EXP_020 cells, reused rather than re-run --
#: same session, same tree, same recipe, same seeds. `if_binin` IS the q = 0.50
#: rung of leg A's ladder, which is why it appears here as a ladder row and not
#: only as a reference.
ARMS = {
    "if_anchor": [f"if_anchor_s{s}" for s in SEEDS],
    "binin_q50": [f"if_binin_s{s}" for s in SEEDS],
    "io_sp34":   [f"io_sp34_s{s}" for s in SEEDS],
    "io_sp10":   [f"io_sp10_s{s}" for s in SEEDS],
    "io_sp05":   [f"io_sp05_s{s}" for s in SEEDS],
    "io_mall":   [f"io_mall_s{s}" for s in SEEDS],
    "io_wide":   [f"io_wide_s{s}" for s in SEEDS],
}

#: The ladder, in density order, with the threshold each rung ran at. Used for
#: the dose-response print and for §6 item 3's drift diagnostic.
LADDER = (("binin_q50", 0.50, 0.0000000000),
          ("io_sp34", 0.34, 0.4124631294),
          ("io_sp10", 0.10, 1.2815515655),
          ("io_sp05", 0.05, 1.6448536270))

#: EXP_020 §5's rule for which EXP_022 arms would run, and what it fired.
#: Recorded by the resolver rather than by prose, so §9 reports a rule that was
#: applied rather than a rule that was remembered.
EXP_020_RULE = {
    "composed": {"required": "EXP_020 H2 AND H4 both held",
                 "fired": "NOT RUN", "because": "H2 UNRESOLVED (+0.01326), "
                                                "H4 UNRESOLVED (+0.00530)"},
    "composed_bin": {"required": "additionally EXP_020 H3 held",
                     "fired": "NOT RUN", "because": "H3 FAILED (+0.04569)"},
    "sparse34": {"required": "EXP_020 H3 held or failed within 4 sigma "
                             "(0.01844)",
                 "fired": "NOT RUN under EXP_020's rule",
                 "because": "H3 failed at +0.04569, which is 2.5x the threshold"},
    "sparse10": {"required": "as sparse34", "fired": "NOT RUN under EXP_020's "
                 "rule", "because": "as sparse34"},
    "tied": {"required": "unconditional", "fired": "AUTHORISED",
             "because": "EXP_022 does not run it: EXP_020 §9 measured the fold "
                        "at 1.63x more than its parameters are worth and 86 % "
                        "of that as a REACH cost, which is evidence against a "
                        "further fold, not for one. Referred with its price."},
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


def arm_stats(runs: list[str], protocol: str) -> dict:
    vals, dead = [], []
    for r in runs:
        if _diverged(r):
            dead.append(r)
            continue
        b = _bpc(r, protocol)
        if b is None:
            dead.append(r)
        else:
            vals.append(b)
    return {"runs": runs, "n": len(vals), "n_preregistered": len(runs),
            "diverged": dead, "provisional": len(vals) < len(runs),
            "bpc": vals,
            "mean": statistics.fmean(vals) if vals else None,
            "sd": statistics.stdev(vals) if len(vals) > 1 else None}


def paired(a: dict, b: dict) -> dict:
    if a["n"] != a["n_preregistered"] or b["n"] != b["n_preregistered"]:
        return {"paired": False,
                "reason": "a cell lost a seed; pairing across unequal n would "
                          "silently drop the partner"}
    d = [x - y for x, y in zip(a["bpc"], b["bpc"])]
    n = len(d)
    m = statistics.fmean(d)
    sd = statistics.stdev(d) if n > 1 else None
    se = (sd / math.sqrt(n)) if sd else None
    return {"paired": True, "diffs": d, "mean_diff": m, "sd_diff": sd,
            "se_diff": se, "t": (m / se) if se else None,
            "n_negative": sum(1 for x in d if x < 0)}


def code_stats(run: str) -> dict | None:
    """§6 item 3's diagnostic, and what it turned into.

    `q` is an INIT density -- `shadow` is a parameter, so a rung's realised
    density drifts. §6 item 3 pre-registered this readback for the case where
    all four rungs converge to one density, which would mean the ladder measured
    an initialisation and not a code.

    **THEY DO NOT CONVERGE, AND THE DRIFT HAS A CLOSED FORM.** Measured on the
    first two completed checkpoints, from the CHECKPOINT and before any bpc in
    this experiment was read: `if_binin_s0` (thr = 0) sits at density 0.4983 and
    `io_sp34_s0` (thr = 0.4125, target 0.34) has fallen to **0.1358**. Weight
    decay shrinks `shadow` toward zero -- sd 1.000 -> 0.368/0.397 -- while
    `thr_in` is a CONSTANT, so

        q_final = 1 - Phi(thr_in / sd(shadow))

    which predicts 0.5000 and 0.1493 against 0.4983 and 0.1358 realised.
    **`thr_in = 0` is a FIXED POINT of density under weight decay** -- shrinking
    a shadow cannot change its sign -- and **every `thr_in > 0` is not**: the
    code gets monotonically sparser for the whole 20,000 steps.

    So this function reports three things rather than one, because the second
    and third are the mechanism and the first is only its symptom:

      `density`   realised, from the checkpoint
      `silent`    characters whose code is ALL ZERO. Their input current is
                  `W0·(g·(0 - q)) + b0`, a constant -- so every silent character
                  is mapped to the SAME current as every other silent one.
      `distinct`  distinct code rows. A collision is two characters the model
                  cannot tell apart at its input, at any context. **6 silent and
                  200/205 distinct on `io_sp34_s0`, against 0 and 205/205 on
                  `if_binin_s0`.**

    Collisions are a *countable, per-character* mechanism for a *zero-context*
    cost, which is where `EXP_020` §9.4 located 83 % of the binary input's
    price. Reported as a marker: no bar was pre-registered on it and none is
    invented now.
    """
    ck = RUNS / run / "ckpt_final.pt"
    cf = RUNS / run / "config.json"
    if not ck.exists() or not cf.exists():
        return None
    sd = torch.load(ck, map_location="cpu", weights_only=False)
    sd = sd.get("model", sd)
    shadow = next((v for k, v in sd.items() if k.endswith("code.shadow")), None)
    if shadow is None:
        return None
    thr = float(json.loads(cf.read_text(encoding="utf-8")).get("iface_thr_in", 0.0))
    shadow = shadow.float()
    B = (shadow >= thr)
    per_char = B.sum(1)
    return {
        "density": round(float(B.float().mean()), 5),
        "shadow_sd": round(float(shadow.std()), 5),
        # The closed form above, evaluated on this checkpoint's own shadow.
        "density_predicted_by_weight_decay": round(float(
            1.0 - 0.5 * (1.0 + math.erf(
                (thr / max(float(shadow.std()), 1e-12)) / math.sqrt(2.0)))), 5),
        "silent_chars": int((per_char == 0).sum()),
        "distinct_codes": len({tuple(r.tolist()) for r in B.to(torch.uint8)}),
        "vocab": int(B.shape[0]),
        "min_active_fibres": int(per_char.min()),
        "max_active_fibres": int(per_char.max()),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--protocol", default="carried", choices=("fresh", "carried"))
    ap.add_argument("--out", default="docs/reports/data/exp_022_arm_results.json")
    args = ap.parse_args(argv)

    out: dict = {
        "experiment": "EXP_022", "protocol": args.protocol,
        "sigma_transferred": SIGMA_TRANSFERRED,
        "sigma_source": "04_phase4_interim.md §236, baseline n=5, DIFFERENT tree",
        "bar_2sigma": BAR_2SIGMA,
        "prereg_sha256": hashlib.sha256(PREREG.read_bytes()).hexdigest(),
        "exp_020_arm_selection_rule": EXP_020_RULE,
    }
    if MANIFEST.exists():
        man = json.loads(MANIFEST.read_text(encoding="utf-8"))
        out["manifest_prereg_sha256_before"] = man.get("prereg_sha256_before")
        out["prereg_matches_manifest"] = (
            man.get("prereg_sha256_before") == out["prereg_sha256"])
        out["reused_from_exp_020"] = man.get("reused_from_exp_020")
        # G2 is asserted over EVERY run, not over the runs that happen to carry
        # the key. `all()` of an empty generator is True, so filtering on
        # `"G2_params_match" in r` made the gate PASS for a manifest with no
        # runs, and dropped a run with no `summary.json` -- the case where the
        # parameter count is most in doubt -- rather than failing it. The
        # denominator is reported beside the verdict, per CONTRIBUTING.md §3.
        _g2_runs = man.get("runs", [])
        _g2_seen = [r for r in _g2_runs if "G2_params_match" in r]
        out["G2_all_params_match"] = bool(_g2_runs) and len(_g2_seen) == len(_g2_runs) \
            and all(r["G2_params_match"] for r in _g2_seen)
        out["G2_denominator"] = {"runs_in_manifest": len(_g2_runs),
                                 "runs_carrying_the_gate": len(_g2_seen),
                                 "expected": 15}
        out["G2_missing_the_gate"] = [r.get("run") for r in _g2_runs
                                      if "G2_params_match" not in r]
        out["G5_violations"] = {r["run"]: r["G5_frozen_violations"]
                                for r in man.get("runs", [])
                                if r.get("G5_frozen_violations")}
        out["K1_unexpected"] = {r["run"]: r["K1_unexpected"]
                                for r in man.get("runs", [])
                                if r.get("K1_unexpected")}

    stats = {k: arm_stats(v, args.protocol) for k, v in ARMS.items()}
    out["arms"] = stats

    bars: dict = {}

    def _one_sided(key, arm, ref, statement, note=""):
        p = paired(stats[arm], stats[ref])
        delta = (None if stats[arm]["mean"] is None or stats[ref]["mean"] is None
                 else stats[arm]["mean"] - stats[ref]["mean"])
        bars[key] = {"statement": statement, "note": note, "delta": delta,
                     "arm": arm, "reference": ref, "paired": p,
                     "verdict": ("NOT RUN" if delta is None else
                                 "HELD" if delta < -BAR_2SIGMA else
                                 "FAILED" if delta > BAR_2SIGMA else
                                 "UNRESOLVED")}

    _one_sided("H1", "io_sp34", "binin_q50",
               f"mean(io_sp34) < mean(if_binin) - {BAR_2SIGMA}",
               "q = 0.34 is the stack's OWN converged firing rate, so the input "
               "boundary is then statistically the same boundary as every other")
    _one_sided("H2", "io_sp10", "binin_q50",
               f"mean(io_sp10) < mean(if_binin) - {BAR_2SIGMA}",
               "predicted BETTER and by MORE than sp34, if leverage-per-bit is "
               "the mechanism")
    def _two_sided(key, arm, ref, statement, note=""):
        """A MAGNITUDE bar, resolved without imposing a direction.

        §4 states H3's direction is UNSTATED and that it is "reported as it
        fires". Running it through `_one_sided` would record a >= 2 sigma
        difference in the worse direction as "FAILED" -- which is a direction,
        and it is the one §4 refuses to state. The verdict vocabulary here is
        therefore BAR MET / UNRESOLVED, and the sign is reported beside it as
        an observation rather than as a pass or a fail.
        """
        p = paired(stats[arm], stats[ref])
        delta = (None if stats[arm]["mean"] is None or stats[ref]["mean"] is None
                 else stats[arm]["mean"] - stats[ref]["mean"])
        bars[key] = {"statement": statement, "note": note, "delta": delta,
                     "arm": arm, "reference": ref, "paired": p,
                     "two_sided": True,
                     "direction": (None if delta is None else
                                   "arm BETTER" if delta < 0 else "arm WORSE"),
                     "verdict": ("NOT RUN" if delta is None else
                                 "BAR MET" if abs(delta) >= BAR_2SIGMA else
                                 "UNRESOLVED")}

    _two_sided("H3", "io_sp05", "binin_q50",
               f"|mean(io_sp05) - mean(if_binin)| >= {BAR_2SIGMA}",
               "DIRECTION DELIBERATELY UNSTATED in §4: 26 active fibres is where "
               "a capacity floor and a plasticity floor would both first appear "
               "and predicting a sign would be predicting which arrives first. "
               "If sp05 LOSES, §5's fourth cell applies and this experiment may "
               "not say which axis caused it (§2.3's confound).")
    _one_sided("H5", "io_mall", "if_anchor",
               f"mean(io_mall) < mean(if_anchor) - {BAR_2SIGMA}",
               "NOT single-variable: io_mall carries 14.3 % more parameters. "
               "§6 item 5.")
    _one_sided("H6", "io_mall", "io_wide",
               f"mean(io_mall) < mean(io_wide) - {BAR_2SIGMA}",
               "THE DECISIVE BAR: the same 104,960 extra parameters spent on a "
               "wider head against a wider stack")

    # H2's second clause is a comparison BETWEEN rungs, not against the bar, and
    # it is reported as a marker: §4 predicts sp10 beats sp34 and no bar was
    # pre-registered on that difference.
    if stats["io_sp10"]["mean"] is not None and stats["io_sp34"]["mean"] is not None:
        bars["H2"]["sp10_minus_sp34"] = (
            stats["io_sp10"]["mean"] - stats["io_sp34"]["mean"])
        bars["H2"]["monotone_in_1_over_q_as_predicted"] = (
            bars["H2"]["sp10_minus_sp34"] < 0)

    out["bars"] = bars

    # -- the density ladder, as a dose-response --------------------------------
    out["density_ladder"] = [
        {"arm": a, "q_target": q, "thr_in": thr,
         "mean_bpc": stats[a]["mean"],
         "delta_vs_binin": (None if stats[a]["mean"] is None
                            or stats["binin_q50"]["mean"] is None
                            else stats[a]["mean"] - stats["binin_q50"]["mean"]),
         "delta_vs_anchor": (None if stats[a]["mean"] is None
                             or stats["if_anchor"]["mean"] is None
                             else stats[a]["mean"] - stats["if_anchor"]["mean"]),
         # §6 item 3: an init density that all four rungs drifted away from
         # would mean this ladder measured an initialisation.
         "code_stats": [code_stats(r) for r in ARMS[a]]}
        for a, q, thr in LADDER
    ]

    # -- cost, H8 --------------------------------------------------------------
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
    hzp = _REPO / "docs" / "reports" / "data" / "exp_022_memory_horizon.json"
    if not hzp.exists():
        # AN ABSENT MEASUREMENT MUST NOT READ AS A PASSING ONE, and it must not
        # read as an ABSENT PREDICTION either. Writing only `measured: false`
        # here left `bars` with five of §4's eight predictions and nothing in the
        # artifact recording that H4 and H7 had not been measured -- the same
        # defect the EXP_017 resolver was fixed for ("an absent H5 measurement
        # must not read as a passing one"), one level up: there the verdict was
        # missing, here the whole bar was.
        out["decomposition"] = {
            "measured": False,
            "artifact": str(hzp.relative_to(_REPO)),
            "why": "the horizon probe has not been run over these checkpoints; "
                   "H4 and H7 are SHAPE claims on the decomposition and cannot "
                   "be resolved without it",
        }
        for _k, _st in (("H4", "AS PRE-REGISTERED: for any rung that clears its "
                               "bar, the c = 0 component moves by MORE than the "
                               "total"),
                        ("H7", f"|io_mall beyond-horizon gain| < {BAR_2SIGMA}")):
            bars[_k] = {"statement": _st, "verdict": "NOT RUN",
                        "why_not_run": f"{hzp.name} absent — run "
                                       "scripts/exp/001_memory_horizon.py first"}
    else:
        hz = json.loads(hzp.read_text(encoding="utf-8"))
        r010 = _import_010()
        ac = hz["absolute_context_curves"]
        _p2s = hz.get("per_context_2sigma", {})
        # NOTE the wrapper: per_context_2sigma is
        # {n_seeds, arm, bars, whole_split_2sigma_for_reference} and the bars are
        # one level down. Passing the wrapper silently falls back to the flat bar
        # at every context -- EXP_020 §9.8 item 3 is that defect, caught there
        # only because two implementations of one statistic disagreed.
        ctx_bars = _p2s.get("bars", _p2s) if isinstance(_p2s, dict) else {}
        dec = {}
        # Leg A decomposes against `if_binin` (its own ladder's q = 0.50 rung),
        # leg B against `if_anchor`. Decomposing a sparse rung against the
        # ANALOGUE anchor would fold the whole binarity cost into every row and
        # hide the thing leg A is measuring.
        against = {"io_sp34": "if_binin", "io_sp10": "if_binin",
                   "io_sp05": "if_binin", "if_binin": "if_anchor",
                   "io_mall": "if_anchor", "io_wide": "if_anchor"}
        # WHAT WAS EXPECTED AND WHAT WAS FOUND, recorded rather than inferred.
        # Skipping a missing arm silently while still reporting
        # `measured: true` is how leg A's ENTIRE decomposition could vanish --
        # every sparse rung is differenced against `if_binin`, so one absent
        # curve takes all three with it and the artifact still looks complete.
        # That is EXP_020's stale-artifact defect (its resolver ran before the
        # horizon probe and shipped `measured: false` with an empty table) met
        # from the other side, and it is the reason this block names its gaps.
        missing = {arm: ref for arm, ref in against.items()
                   if arm not in ac or ref not in ac}
        for arm, refname in against.items():
            if arm in missing:
                continue
            ref = ac[refname]["bpc_at_context"]
            dec[arm] = r010.decompose(ref, ac[arm]["bpc_at_context"])
            dec[arm]["reference"] = refname
            dec[arm]["context_comparison"] = r010.context_comparison(
                ac[arm]["bpc_at_context"], ref, ctx_bars)
            dec[arm]["n_worse_probe_stored"] = ac[arm].get(
                "n_contexts_significantly_worse")
        out["decomposition"] = {
            "measured": True, "arms": dec,
            "arms_expected": {a: r for a, r in against.items()},
            "arms_missing_from_horizon": missing,
            "complete": not missing,
            "note": "EXP_020 §9.5 measured this statistic's noise floor with a "
                    "BITWISE copy of the anchor: -0.00499 total, -0.01395 at "
                    "c=0, +0.00944 within reach, 120/128 contexts 'worse'. Read "
                    "every row against that."}
        # H4 and H7 are SHAPE claims from §4 and are resolved here, on the
        # decomposition, rather than being left as prose.
        h4 = {}
        for arm in ("io_sp34", "io_sp10", "io_sp05"):
            d = dec.get(arm)
            if d:
                z, t = d["zero_context_gain_bpc"], d["total_gain_bpc"]
                h4[arm] = {
                    "total": t,
                    "zero_context": z,
                    "within_reach": d["within_reach_gain_bpc"],
                    "beyond_horizon": d["beyond_horizon_gain_bpc"],
                    # AS PRE-REGISTERED, and it is reported as it fires.
                    "zero_moves_more_than_total": abs(z) > abs(t),
                    # THE CORRECTED STATISTIC, added post-hoc and labelled. See
                    # the note below for why the pre-registered one cannot do
                    # the job §4 gave it.
                    "zero_context_share_of_gain": (z / t) if t else None,
                    "gain_is_at_zero_context": gain_is_at_zero_context(z, t),
                    "loss_is_at_zero_context": loss_is_at_zero_context(z, t),
                }
        bars["H4"] = {
            "statement": "AS PRE-REGISTERED: for any rung that clears its bar, "
                         "the c = 0 component moves by MORE than the total",
            "note": "83 % of binin's cost is at c = 0, so a fix for that cost "
                    "must appear there",
            "per_rung": h4,
            # ------------------------------------------------------------------
            # A DEFECT IN THE PRE-REGISTERED STATISTIC, found by audit BEFORE any
            # leg-A bpc was read, reported rather than repaired.
            #
            # The decomposition is ADDITIVE: total = c0 + within + beyond. So
            # "|c0| > |total|" is equivalent to "(within + beyond) has the
            # OPPOSITE sign to total", i.e. the rung must LOSE outside zero
            # context to pass. A rung that exactly undoes binin's cost -- the
            # canonical success this bar was written for -- FAILS it.
            #
            # Worse, binin's OWN cost, which the 83 % motivation is taken from,
            # has |c0| = 0.0371 < |total| = 0.0448. The bar's own motivating
            # example does not satisfy the bar.
            #
            # And the implementation compounds it: abs() on both sides means a
            # rung whose c = 0 component moved the WRONG WAY can still pass.
            #
            # H4 is a MARKER carrying no adoption weight (§4: "no bar;
            # reported"), so nothing turns on it -- but §5 does branch on it, and
            # that branch is now known to be unreliable. Both statistics are
            # emitted: the pre-registered one as it fires, and
            # `gain_is_at_zero_context` (sign-aware, >= 50 % of a real gain
            # sitting at c = 0) as the post-hoc correction.
            #
            # THIS IS THE THIRD INSTANCE OF OPEN DECISION #9's DEFECT CLASS --
            # a pre-registered bar that needed a guard-clause review before it
            # was committed -- after EXP_012's Y5 and EXP_013's N1. Referred.
            #
            # AND THE CORRECTION ITSELF SHIPPED WITH ITS SIGN INVERTED, caught by
            # a second audit before this file was committed and before any leg-A
            # bpc was read. `gain_is_at_zero_context` was written
            # `(t < 0 and z < 0 and ...)`, carrying this file's local
            # "negative = better" delta convention into the one block that
            # consumes `010.decompose`'s "positive = better" one. It was
            # therefore True only when a rung was WORSE, and False for every
            # genuine gain; fed §2.1's own motivating row (binin, -0.0448 total,
            # -0.0371 at c = 0, header "positive = better") it fired
            # "the gain is at zero context" on a pure 0.0448 bpc COST.
            #
            # The fix is the sign only; the >= 50 % threshold and the
            # sign-awareness are unchanged from what was written before any run.
            # Recorded here rather than silently corrected, because a marker that
            # was wrong in one direction is evidence about how it was reviewed --
            # and because decision #9 is about exactly this.
            # ------------------------------------------------------------------
            "preregistered_statistic_is_defective": True,
            "corrected_statistic_sign_was_inverted_and_fixed_before_commit": True,
            "defect": "total = c0 + within + beyond, so |c0| > |total| requires "
                      "the rung to LOSE outside zero context; binin's own "
                      "83 %-at-c0 cost has |c0| < |total| and would fail this "
                      "bar. The abs() on both terms additionally admits a rung "
                      "whose c = 0 component moved the wrong way.",
            "corrected_statistic": "gain_is_at_zero_context: total > 0 AND "
                                   "c0 > 0 AND |c0| >= 0.5*|total| (sign-aware; "
                                   "010.decompose returns positive = better)",
            "refers_to_decision": 9,
        }
        dm = dec.get("io_mall")
        bars["H7"] = {
            "statement": f"|io_mall beyond-horizon gain| < {BAR_2SIGMA}",
            "note": "a multi-layer readout adds no temporal information; if "
                    "it gains beyond the horizon the mechanism is not the "
                    "one claimed in §2.4",
            "beyond_horizon_gain_bpc": (dm["beyond_horizon_gain_bpc"]
                                        if dm else None),
            # A missing `io_mall` row is NOT RUN, never a silent absence: the arm
            # is differenced against `if_anchor`, so either curve going missing
            # takes H7 with it.
            "verdict": ("NOT RUN" if not dm else
                        "HELD" if abs(dm["beyond_horizon_gain_bpc"]) < BAR_2SIGMA
                        else "FAILED"),
            "why_not_run": (None if dm else
                            "io_mall or if_anchor absent from the horizon "
                            "artifact — see decomposition.arms_missing_from_horizon"),
        }

    p = _REPO / args.out
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")

    print(f"\nEXP_022 — protocol {args.protocol}, "
          f"2 sigma = {BAR_2SIGMA} (TRANSFERRED, different tree)\n")
    for arm, s in stats.items():
        print(f"  {arm:10s} n={s['n']}/{s['n_preregistered']} mean={s['mean']} "
              f"sd={s['sd']} "
              f"{'PROVISIONAL ' + str(s['diverged']) if s['provisional'] else ''}")

    _decomp = out.get("decomposition", {})
    _dm = _decomp.get("arms_missing_from_horizon") or {}
    if not _decomp.get("measured"):
        # The warning used to be guarded on a key that exists only in the
        # measured branch, so the ENTIRELY-ABSENT case printed nothing at all --
        # the loudest case was the silent one.
        print("\n  !! DECOMPOSITION NOT MEASURED: the horizon artifact is "
              "absent. H4 and H7 are recorded NOT RUN. Run "
              "scripts/exp/001_memory_horizon.py BEFORE this resolver.")
    elif _dm:
        print(f"\n  !! DECOMPOSITION INCOMPLETE: {sorted(_dm)} absent from the "
              "horizon artifact. Every row differenced against a missing "
              "reference is DROPPED, and H4/H7 rest on those rows. Run the "
              "horizon probe BEFORE this resolver.")

    print("\n  the density ladder (delta vs if_binin, negative = BETTER):")
    print(f"    {'rung':10s} {'q':>6s} {'mean bpc':>10s} {'vs binin':>10s} "
          f"{'vs anchor':>10s}  final density")
    for row in out["density_ladder"]:
        cs = [c for c in row["code_stats"] if c]
        dens = [f"{c['density']:.3f}" for c in cs] or ["--"]
        coll = [f"{c['vocab'] - c['distinct_codes']}/{c['silent_chars']}"
                for c in cs] or ["--"]
        print(f"    {row['arm']:10s} {row['q_target']:6.2f} "
              f"{(row['mean_bpc'] if row['mean_bpc'] is not None else float('nan')):10.5f} "
              f"{(row['delta_vs_binin'] if row['delta_vs_binin'] is not None else float('nan')):+10.5f} "
              f"{(row['delta_vs_anchor'] if row['delta_vs_anchor'] is not None else float('nan')):+10.5f}"
              f"  {dens}  collisions/silent {coll}")

    print()
    for k in ("H1", "H2", "H3", "H5", "H6"):
        b = bars.get(k)
        if b is None:
            continue
        pr = b["paired"]
        extra = ""
        if pr.get("paired") and pr.get("t") is not None:
            extra = (f"  paired t={pr['t']:.2f}  "
                     f"{pr['n_negative']}/{len(pr['diffs'])} seeds negative")
        print(f"  {k}: {b['verdict']:12s} delta={b['delta']}{extra}")
        print(f"      {b['statement']}")
    # H4 and H7 are printed unconditionally, including when they are NOT RUN.
    # Every one of §4's eight predictions appears in this summary or the summary
    # is not a summary.
    _h4 = bars.get("H4")
    if _h4 is not None:
        print(f"\n  H4: {_h4.get('verdict', 'REPORTED (marker, no bar)'):12s} "
              f"{_h4['statement'][:58]}")
        for _arm, _r in (_h4.get("per_rung") or {}).items():
            print(f"      {_arm:10s} total={_r['total']:+.5f} "
                  f"c0={_r['zero_context']:+.5f} "
                  f"prereg={_r['zero_moves_more_than_total']!s:5s} "
                  f"corrected={_r['gain_is_at_zero_context']!s}")
        if _h4.get("why_not_run"):
            print(f"      {_h4['why_not_run']}")
    _h7 = bars.get("H7")
    if _h7 is not None:
        _bh = _h7.get("beyond_horizon_gain_bpc")
        print(f"\n  H7: {_h7['verdict']:12s} beyond-horizon="
              + (f"{_bh:+.5f}" if _bh is not None else "NOT MEASURED"))
        if _h7.get("why_not_run"):
            print(f"      {_h7['why_not_run']}")
    print("\n  EXP_020 §5's arm-selection rule, as it fired:")
    for arm, r in EXP_020_RULE.items():
        print(f"    {arm:14s} {r['fired']:26s} {r['because']}")
    print(f"\nWROTE {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
