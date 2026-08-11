"""EXP_017 resolver -- every bar in §4, scored exactly as it was written.

    python scripts/exp/017_detach_results.py \
        --out docs/reports/data/exp_017_results.json

Committed **before any run's bpc is read**, which is the standard `010`, `011`,
`014` and `015` set. It reads artifacts and resolves the pre-registered
predictions; it trains nothing, adopts nothing and ranks nothing.

WHAT THIS FILE IS CAREFUL ABOUT, AND WHY
----------------------------------------
Phase 4 has now had **four** pre-registered bars misfire, all the same way: a bar
specified without the guard the experiment's own limitations section had already
written down (`EXP_012` Y5, `EXP_013` N1, `EXP_015` P1/P2, `EXP_016` P3). Three
specific defences are therefore built into this file rather than left to prose:

  * **`UNRESOLVED` is a first-class verdict**, returned whenever `|delta|` falls
    inside the two-sample band -- never silently rendered as "no difference".
    `EXP_015` P1/P2 resolved UNRESOLVED for legs that produced no number at all,
    so the word is also kept distinct from `NOT RUN`.
  * **H3 prints both arms' absolute means, always.** `EXP_013`'s N1 fired because
    a difference was constrained without its subtrahend; a delta favourable to
    the arm because the *anchor* collapsed is not a win, and the only way to see
    that is to print both.
  * **A diverged run resolves `DIVERGED`, not `UNRESOLVED`.** `EXP_015` §10 item
    3 referred exactly this correction and it is applied here for this
    experiment's own bars -- which is what a referral being *taken up* looks
    like, as opposed to a closed log being rewritten.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
RUNS = _REPO / "experiments" / "runs"
DATA = _REPO / "docs" / "reports" / "data"

#: The adopted arm's committed carried test bpc at 735K, n = 7 (report §5).
#: H3a compares the FRESH anchor against it as a MARKER -- §4's H3a states in
#: advance that a failure here does not void H3, because H3 compares two arms
#: trained on one tree, which is what Elliot's ruling on decision #10 prescribes.
TWOCOMP_COMMITTED_CARRIED = 2.11869
H3A_MARKER_BAND = 0.02

#: `EXP_015`'s parameter-identical table at `d_model = 1481`. Quoted beside H2b,
#: which carries NO verdict (§4.0): these are n = 1 numbers and a difference
#: between two of them resolves nothing.
EXP015_AT_WIDTH = {
    "snn": 2.00073,
    "twocomp": None,               # NaN -- diverged at step 5138
    "twocomp_threshold": None,     # NaN -- diverged at ~2000
    "gru": 1.54699,
}

#: Wall-clock of the adopted arm at each width, for H5's ratio. The denominator
#: is named here rather than in a caption because `EXP_011` had a ratio column
#: whose reference row was wrong (report rev 2) and this is the cheap fix.
ADOPTED_WALL_CLOCK_S = {512: 536.69, 1481: 2432.24}
H5_BAND = 1.10

#: The plain LIF's committed 2-sigma memory horizon at 735K, and the adopted
#: arm's. H4's LOST verdict is "indistinguishable from the plain LIF".
LIF_HORIZON = 7
TWOCOMP_HORIZON = 47


def _read(p: Path) -> dict | None:
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def _carried(run: str) -> float | None:
    """Carried test bpc, or None if the run has no scoreable number.

    A NaN is returned as None rather than as a float, deliberately: a NaN that
    flows into a mean produces a NaN mean, which is how a diverged leg silently
    poisons an arm's summary instead of being reported as a divergence.
    """
    j = _read(RUNS / run / "final_test.json")
    if not j:
        return None
    v = j.get("results", {}).get("carried", {}).get("bpc")
    if v is None or not math.isfinite(float(v)):
        return None
    return float(v)


def _mean_sd(xs: list[float]) -> tuple[float, float, int]:
    n = len(xs)
    if n == 0:
        return float("nan"), float("nan"), 0
    m = statistics.fmean(xs)
    sd = statistics.stdev(xs) if n > 1 else 0.0
    return m, sd, n


# --------------------------------------------------------------------------
# T1, T1b, T2, H1 -- Leg A and the gates
# --------------------------------------------------------------------------

def resolve_leg_a(jac: dict | None, campaign: dict | None) -> dict:
    if jac is None:
        return {"status": "NOT RUN", "why": "Leg A artifact absent"}
    t1b = jac.get("t1b_reproduces_exp_016", {})

    # T2 has two halves and both are abort-on-fail: the probe's local scan must
    # be the committed scan (Leg A's own check), and the R10 gate must have been
    # mutation-tested before any training step (§4's T2).
    mut = {"checked": False}
    if campaign is not None:
        muts = campaign.get("mutations", {})
        d_ids = sorted(k for k in muts if k.startswith("D"))
        d_caught = [k for k in d_ids if muts[k].get("caught")]
        other = sorted(k for k in muts if not k.startswith("D"))
        other_caught = [k for k in other if muts[k].get("caught")]
        mut = {
            "checked": True,
            "n_new_mutations": len(d_ids),
            "n_new_caught": len(d_caught),
            "new_escaped": [k for k in d_ids if k not in d_caught],
            "n_preexisting": len(other),
            "n_preexisting_caught": len(other_caught),
            "preexisting_escaped": [k for k in other if k not in other_caught],
            "holds": (len(d_ids) > 0 and len(d_caught) == len(d_ids)
                      and len(other_caught) == len(other)),
        }

    rows = []
    for r in jac.get("legs", []):
        if r.get("error"):
            continue
        for ly in r["layers"]:
            rows.append({
                "leg": r["leg"], "run": r["run"], "arch": r["arch"],
                "d_model": r["d_model"], "batch_step": r["batch_step"],
                "layer": ly["layer"],
                "hard_max_abs_g": ly["hard"]["max_abs_g"],
                "hard_frac_over_1": ly["hard"]["frac_over"]["1.0"],
                "hard_longest_expanding_run": ly["hard"]["longest_expanding_run_steps"],
                "hard_max_window_log10_gain": ly["hard"]["max_window_log10_gain"],
                "detach_max_abs_g": ly["detach"]["max_abs_g"],
                "detach_frac_over_1": ly["detach"]["frac_over"]["1.0"],
                "detach_longest_expanding_run": ly["detach"]["longest_expanding_run_steps"],
                "detach_max_window_log10_gain": ly["detach"]["max_window_log10_gain"],
                "max_abs_w_times_vs": ly["hard"]["max_abs_w_times_vs"],
                "ratio_hard_over_detach": ly["max_abs_g_ratio_hard_over_detach"],
            })

    return {
        "T1": {"verdict": "HOLDS" if jac.get("t1_holds") else "FAILS",
               "what": "the snn legs reproduce the 0.5123596 closed form"},
        "T1b": {"verdict": ("HOLDS" if t1b.get("holds")
                            else ("FAILS" if t1b.get("checked") else "NOT CHECKED")),
                "n_compared": t1b.get("n_compared"),
                "n_differing": t1b.get("n_differing"),
                "differences": t1b.get("differences"),
                "what": "the hard column reproduces EXP_016's committed artifact"},
        "T2_probe": {"verdict": "HOLDS" if jac.get("t2_holds") else "FAILS",
                     "what": "the local scan is bitwise the committed eager scan"},
        "T2_gate": {"verdict": ("HOLDS" if mut.get("holds")
                                else ("FAILS" if mut.get("checked") else "NOT CHECKED")),
                    **mut,
                    "what": "the R10 gate is mutation-tested before any training step"},
        "H1": {"verdict": "HELD" if jac.get("h1_holds") else "REFUTED",
               "bound": jac.get("detach_closed_form_bound"),
               "what": ("under the detached rule, max|g| <= beta_f, frac(|g|>1) = 0 "
                        "and the longest expanding run is 0, on every leg, "
                        "every layer, both batches")},
        "table": rows,
    }


# --------------------------------------------------------------------------
# H2, H2b -- the headline
# --------------------------------------------------------------------------

def resolve_h2(manifest: dict | None) -> dict:
    if manifest is None:
        return {"verdict": "NOT RUN", "why": "run manifest absent"}
    row = next((r for r in manifest.get("runs", []) if r["run"] == "detach_d1481_s0"),
               None)
    if row is None:
        return {"verdict": "NOT RUN", "why": "detach_d1481_s0 not in the manifest"}
    scan = row.get("divergence_scan", {})
    if not scan.get("log_present"):
        return {"verdict": "NOT RUN", "why": "no log; an absent log is not a clean log"}

    n_bad = scan.get("n_nonfinite_loss_records") or 0
    last = scan.get("last_logged_step")
    if n_bad:
        first = scan.get("first_nonfinite_step")
        verdict = f"DIVERGED at step {first}"
        # §4's guard: a divergence at a DIFFERENT step is a different finding and
        # is not "the bound failed". Said by the resolver so nobody has to
        # remember it while reading a table.
        same = (first == 5138)
        note = ("same step as the adopted arm -- §5 row 4: §2.2 REFUTED AS "
                "SUFFICIENT, report prominently"
                if same else
                "a DIFFERENT step from the adopted arm's 5138 -- §5 row 5: a NEW "
                "divergence, to be chased. §2.2 stands and this is not evidence "
                "against it")
    else:
        verdict = "SURVIVES"
        note = ("n = 1 seed. The available sentence is 'the arm did not exhibit "
                "the adopted arm's deterministic divergence', NOT 'the arm "
                "trains at width' -- §4's H2 guards.")

    carried = _carried("detach_d1481_s0")
    return {
        "verdict": verdict,
        "note": note,
        "passed_step_5138": scan.get("passed_step_5138"),
        "last_logged_step": last,
        "n_nonfinite_loss_records": n_bad,
        "first_nonfinite_step": scan.get("first_nonfinite_step"),
        "max_grad_norm": scan.get("max_grad_norm"),
        "n_logged_steps_over_clip": scan.get("n_logged_steps_over_clip"),
        "adopted_arm_at_this_width": {
            "run": "arch_twocomp_d1481_s0", "outcome": "DIVERGED at step 5138",
        },
        "H2b_marker": {
            "carried_bpc": carried,
            "verdict": "MARKER -- no bar, no verdict (§4.0)",
            "exp_015_parameter_identical_table": EXP015_AT_WIDTH,
        },
    }


# --------------------------------------------------------------------------
# H3 -- the cost at 735K
# --------------------------------------------------------------------------

def resolve_h3(seeds: tuple[int, ...]) -> dict:
    det = [(s, _carried(f"detach_d512_s{s}")) for s in seeds]
    anc = [(s, _carried(f"anchor_twocomp_d512_s{s}")) for s in seeds]
    dv = [v for _s, v in det if v is not None]
    av = [v for _s, v in anc if v is not None]

    out: dict = {
        "detach_per_seed": {str(s): v for s, v in det},
        "anchor_per_seed": {str(s): v for s, v in anc},
    }
    if len(dv) < 2 or len(av) < 2:
        out["verdict"] = "NOT RUN"
        out["why"] = (f"{len(dv)} scoreable detach seeds and {len(av)} anchor "
                      "seeds; a diverged or missing run resolves NOT RUN, not "
                      "UNRESOLVED (EXP_015 §10 item 3)")
        return out

    dm, dsd, dn = _mean_sd(dv)
    am, asd, an = _mean_sd(av)
    delta = dm - am
    se = math.sqrt(dsd * dsd / dn + asd * asd / an)   # Welch, unpooled

    if se == 0.0:
        verdict = "UNRESOLVED"
    elif delta <= -2.0 * se:
        verdict = "THE BOUND IS FREE OR BETTER"
    elif delta >= 2.0 * se:
        verdict = "THE BOUND COSTS"
    else:
        verdict = "UNRESOLVED"

    anchor_off = abs(am - TWOCOMP_COMMITTED_CARRIED)
    out.update({
        # BOTH absolute means, always. EXP_013 N1's lesson.
        "detach_mean": dm, "detach_sd": dsd, "detach_n": dn,
        "anchor_mean": am, "anchor_sd": asd, "anchor_n": an,
        "delta_detach_minus_anchor": delta,
        "welch_se": se,
        "delta_in_se": (delta / se) if se else None,
        "verdict": verdict,
        "resolution_floor_bpc": 2.0 * se,
        "note": ("UNRESOLVED means the design lacked the power to separate the "
                 "arms, NOT that they are equal (§4.0). This design resolves "
                 f"{2.0 * se:.5f} bpc and no better."),
        "H3a_anchor_marker": {
            "committed_n7": TWOCOMP_COMMITTED_CARRIED,
            "fresh_n3_mean": am,
            "abs_difference": anchor_off,
            "band": H3A_MARKER_BAND,
            "verdict": "WITHIN BAND" if anchor_off <= H3A_MARKER_BAND else "OUTSIDE BAND",
            "consequence": ("none for H3's verdict -- §4's H3a fixes in advance "
                            "that H3 still resolves, because it compares two arms "
                            "on ONE tree, which is what decision #10's ruling "
                            "prescribes. The discrepancy is quoted beside every "
                            "H3 number."),
        },
        # sigma, re-measured for a structurally new arm (CONTRIBUTING.md §4).
        "sigma_markers": {
            "sigma_detach": dsd, "df_detach": max(dn - 1, 0),
            "sigma_anchor": asd, "df_anchor": max(an - 1, 0),
            "committed_sigma_twocomp_exp005": 0.00313,
            "warning": ("2 degrees of freedom. These are MARKERS and must not be "
                        "promoted to project constants -- CONTRIBUTING.md §4, and "
                        "the error EXP_005 already recorded twice."),
        },
    })
    return out


# --------------------------------------------------------------------------
# H4 -- reach
# --------------------------------------------------------------------------

def resolve_h4(horizon: dict | None, seeds: tuple[int, ...]) -> dict:
    if horizon is None:
        return {"verdict": "NOT RUN", "why": "horizon artifact absent"}
    arms = horizon.get("arms", {})

    def h(run: str) -> int | None:
        e = arms.get(run)
        if not e:
            return None
        return e.get("paired_context", {}).get("horizon_2sigma")

    pairs, det_h, anc_h = [], [], []
    for s in seeds:
        a, b = h(f"detach_d512_s{s}"), h(f"anchor_twocomp_d512_s{s}")
        pairs.append({"seed": s, "detach": a, "anchor": b,
                      "delta": (None if a is None or b is None else a - b)})
        if a is not None:
            det_h.append(a)
        if b is not None:
            anc_h.append(b)

    deltas = [p["delta"] for p in pairs if p["delta"] is not None]
    if not deltas or not det_h:
        return {"verdict": "NOT RUN", "pairs": pairs,
                "why": "no paired horizon available"}

    med_delta = statistics.median(deltas)
    med_det = statistics.median(det_h)
    # The tie is decided in §4, before it was seen: median delta == 0 -> RETAINED.
    if med_delta >= 0:
        verdict = "REACH RETAINED"
    elif med_det > LIF_HORIZON:
        verdict = "REACH REDUCED"
    else:
        verdict = "REACH LOST"

    return {
        "verdict": verdict,
        "pairs": pairs,
        "median_delta": med_delta,
        "median_detach_horizon": med_det,
        "median_anchor_horizon": statistics.median(anc_h) if anc_h else None,
        "reference_lif_horizon": LIF_HORIZON,
        "reference_committed_twocomp_horizon": TWOCOMP_HORIZON,
        "note": ("n = 3, no sigma_horizon exists anywhere in this project, no "
                 "interval. This resolves a DIRECTION. A median delta of -1 and "
                 "of -20 both resolve REDUCED, which is why every per-seed value "
                 "is printed above. It may not be used alone to adopt or reject "
                 "the arm (§4's H4 guards)."),
    }


# --------------------------------------------------------------------------
# H5 -- the systems column
# --------------------------------------------------------------------------

def resolve_h5(manifest: dict | None) -> dict:
    if manifest is None:
        return {"verdict": "NOT RUN"}
    rows = []
    breach = False
    for r in manifest.get("runs", []):
        if r.get("arch") != "twocomp_detach":
            continue
        d = r.get("d_model")
        ref = ADOPTED_WALL_CLOCK_S.get(d)
        wc = r.get("wall_clock_s")
        ratio = (wc / ref) if (wc and ref) else None
        ok = ratio is not None and ratio <= H5_BAND
        breach = breach or (ratio is not None and not ok)
        rows.append({"run": r["run"], "d_model": d, "wall_clock_s": wc,
                     "adopted_arm_wall_clock_s": ref,
                     "ratio_vs_adopted_arm": ratio,
                     "peak_vram_gib": r.get("peak_vram_gib"),
                     "within_band": ok})
    # An ABSENT measurement is not a passing one. With no rows, `breach` is False
    # and the naive verdict would be "WITHIN BAND" -- the same shape of defect
    # `EXP_015` §9.10 records twice (a driver that marked dead legs complete, and
    # an absent log that read as a clean log). Caught here by a dry run against
    # partial artifacts, before it could report on nothing.
    if not rows:
        return {"verdict": "NOT RUN", "band": H5_BAND, "rows": [],
                "why": "no twocomp_detach run has a wall-clock figure yet"}
    if any(r["ratio_vs_adopted_arm"] is None for r in rows):
        return {"verdict": "PARTIAL", "band": H5_BAND, "rows": rows,
                "why": "at least one run has no comparable wall-clock figure"}
    return {
        "verdict": "BREACH -- investigate" if breach else "WITHIN BAND",
        "band": H5_BAND,
        "denominator": "the ADOPTED arm at the same width, from its summary.json",
        "rows": rows,
        "note": "marker; a breach raises a systems question rather than resolving a bar",
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="docs/reports/data/exp_017_results.json")
    ap.add_argument("--seeds", default="0,1,2")
    args = ap.parse_args(argv)
    seeds = tuple(int(s) for s in args.seeds.split(",") if s.strip())

    jac = _read(DATA / "exp_017_jacobian_bound.json")
    manifest = _read(DATA / "exp_017_run_manifest.json")
    horizon = _read(DATA / "exp_017_horizon.json")
    campaign = _read(DATA / "audit_08_mutation_campaign.json")

    out = {
        "experiment": "EXP_017_bounded_reset_jacobian",
        "prereg": "experiments/logs/EXP_017_bounded_reset_jacobian.md",
        "prereg_sha256_from_manifest": (
            manifest.get("prereg_sha256_before_first_run") if manifest else None),
        "prereg_unchanged": manifest.get("prereg_unchanged") if manifest else None,
        "adopts_nothing": True,
        "ranks_nothing": True,
        "decides_no_open_decision": True,
        "leg_a": resolve_leg_a(jac, campaign),
        "H2": resolve_h2(manifest),
        "H3": resolve_h3(seeds),
        "H4": resolve_h4(horizon, seeds),
        "H5": resolve_h5(manifest),
    }

    la = out["leg_a"]
    aborts = [k for k in ("T1", "T1b", "T2_probe", "T2_gate")
              if isinstance(la.get(k), dict) and la[k].get("verdict") == "FAILS"]
    out["abort"] = bool(aborts)
    out["abort_reason"] = aborts or None

    p = Path(args.out)
    if not p.is_absolute():
        p = _REPO / p
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")

    print(f"wrote {p}\n")
    if isinstance(la, dict) and "T1" in la:
        for k in ("T1", "T1b", "T2_probe", "T2_gate", "H1"):
            print(f"  {k:9s} {la[k]['verdict']}")
    print(f"  {'H2':9s} {out['H2']['verdict']}")
    print(f"  {'H2b':9s} carried = {out['H2'].get('H2b_marker', {}).get('carried_bpc')}"
          "   (MARKER, no verdict)")
    h3 = out["H3"]
    print(f"  {'H3':9s} {h3['verdict']}")
    if "delta_detach_minus_anchor" in h3:
        print(f"            detach {h3['detach_mean']:.5f} +/- {h3['detach_sd']:.5f} "
              f"(n={h3['detach_n']})   anchor {h3['anchor_mean']:.5f} +/- "
              f"{h3['anchor_sd']:.5f} (n={h3['anchor_n']})")
        print(f"            delta = {h3['delta_detach_minus_anchor']:+.5f} bpc "
              f"= {h3['delta_in_se']:+.2f} se   (resolves "
              f"{h3['resolution_floor_bpc']:.5f} and no better)")
        print(f"            H3a anchor vs committed 2.11869: "
              f"{h3['H3a_anchor_marker']['verdict']} "
              f"({h3['H3a_anchor_marker']['abs_difference']:.5f})")
    print(f"  {'H4':9s} {out['H4']['verdict']}")
    print(f"  {'H5':9s} {out['H5']['verdict']}")
    if out["abort"]:
        print(f"\n  ABORT: {out['abort_reason']} -- §5 row 1. "
              "No H may be reported.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
