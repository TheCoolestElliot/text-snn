"""EXP_016 resolver: T1, T2 and P1-P5 against the pre-registration.

    python scripts/exp/016_jacobian_results.py --print-bars      # before any result
    python scripts/exp/016_jacobian_results.py --out docs/reports/data/exp_016_results.json

Committed BEFORE any leg's output is read, per `CONTRIBUTING.md` §2 and the
standard `010`/`011`/`014`/`015` already set.  It reads
`exp_016_reset_jacobian.json` and resolves each pre-registered prediction to the
vocabulary fixed in `EXP_016` §4-§5, including the two that can only ABORT and
the one whose power is deliberately one-sided.

It adopts nothing, ranks nothing and changes no hyperparameter.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]

#: The largest FINITE gradient recorded in the eager replay of the failing step
#: (`docs/reports/data/exp_015_divergence.json` -> eager_replay_of_bad_step ->
#: grad_abs_max_finite).  P3 asks whether the homogeneous product can reach it.
OBSERVED_FINITE_GRAD = 7.789977971209904e29
P3_BAR_LOG = math.log(OBSERVED_FINITE_GRAD)

#: Leg ids, fixed by EXP_016 §3.
CONTROLS = ("A1", "A2", "A3")     # arch = "snn"
ARM_NARROW, ARM_WIDE, CONTROL_WIDE = "A4", "A5", "A3"

BARS = f"""
EXP_016 -- pre-registered bars, printed from the resolver before any result

  T1  (abort-on-fail)  every "snn" leg (A1, A2, A3), every layer:
                       max|g| <= lif_bound(alpha, thr, beta) + 1e-5
                       This tests the PROBE, not the theory: EXP_016 §2.2 derives
                       the bound for every finite membrane at every width, so a
                       violation means the probe computes something else.

  T2  (abort-on-fail)  every leg, every layer: the local scan's spikes are
                       BITWISE equal to the committed eager reference, and the
                       final state agrees to <= 1e-5.

  P1  (existence)      A5 (twocomp, d=1481), layer 0:  max|g| > 1
                       -> HELD / REFUTED.  REFUTED means §2.3's mechanism did not
                       fire in this model and Leg B does not run.

  P2  (direction only) frac(|g| > 1) at layer 0 is strictly larger on A5 than on
                       A4 -> DIRECTION HELD / DIRECTION REVERSED / INDISTINGUISHABLE.
                       Not step-matched, n=1, no interval.  Never "width causes it".

  P3  (ONE-SIDED)      A5, layer 0:  max_(b,c) sum_t log|g_t| >= {P3_BAR_LOG:.5f}
                       ( = log({OBSERVED_FINITE_GRAD:.6g}), the largest FINITE gradient
                       in the eager replay of step 5138 )
                       FAILS  -> REFUTED AS SUFFICIENT.  Decisive.
                       HOLDS  -> CONSISTENT WITH.  NOT proven -- the sum ignores
                                 the grad_spike injections and sign cancellation,
                                 and A5's checkpoint is 138 steps early.
                                 The word "proven" may not be used of a held P3.

  P4  (direction only) A5: layer 0's frac(|g|>1) AND max chain gain both exceed
                       layer 1's -> DIRECTION HELD / DIRECTION REVERSED / SPLIT.

  P5  (existence)      A3 (snn, d=1481 -- same width, trains cleanly), every
                       layer:  frac(|g| > 1) == 0 exactly -> HELD / REFUTED.
                       This is the sentence that answers EXP_015's constraint (f).

No bar here adopts anything, changes a hyperparameter, or decides decision #11.
"""


def _layer(row: dict, k: int) -> dict:
    for ly in row["layers"]:
        if ly["layer"] == k:
            return ly
    raise KeyError(f"leg {row['leg']} has no layer {k}")


def resolve(data: dict) -> dict:
    rows = {r["leg"]: r for r in data["results"] if "error" not in r}
    missing = [lg for lg in (*CONTROLS, ARM_NARROW, ARM_WIDE) if lg not in rows]
    out: dict = {
        "experiment": "EXP_016_reset_jacobian",
        "adopts_nothing": True,
        "ranks_nothing": True,
        "changes_no_hyperparameter": True,
        "decides_no_decision_row": True,
        "missing_legs": missing,
        "p3_bar_log": P3_BAR_LOG,
        "observed_finite_grad": OBSERVED_FINITE_GRAD,
    }

    # -- T1 / T2: abort-on-fail ------------------------------------------
    t1_fail, t2_fail = [], []
    for lg, r in sorted(rows.items()):
        if r["arch"] == "snn":
            for ly in r["layers"]:
                if ly["max_abs_g"] > r["lif_closed_form_bound"] + data["t1_tolerance"]:
                    t1_fail.append(
                        {"leg": lg, "layer": ly["layer"],
                         "max_abs_g": ly["max_abs_g"],
                         "bound": r["lif_closed_form_bound"]})
        if not r["t2_holds"]:
            t2_fail.append({"leg": lg,
                            "bitwise": r["t2_spikes_bitwise_equal"],
                            "state_diff": r["t2_state_max_abs_diff"]})
    out["T1"] = {"verdict": "HOLDS" if not t1_fail else "FAILED", "violations": t1_fail}
    out["T2"] = {"verdict": "HOLDS" if not t2_fail else "FAILED", "violations": t2_fail}
    if t1_fail or t2_fail:
        out["verdict"] = "ABORT -- instrument defect; no prediction resolved"
        return out

    # -- P1 ---------------------------------------------------------------
    a5_l0 = _layer(rows[ARM_WIDE], 0)
    out["P1"] = {
        "verdict": "HELD" if a5_l0["max_abs_g"] > 1.0 else "REFUTED",
        "max_abs_g": a5_l0["max_abs_g"],
        "lif_bound_for_reference": rows[ARM_WIDE]["lif_closed_form_bound"],
        "ratio_to_lif_bound": a5_l0["max_abs_g_over_lif_bound"],
        "max_abs_w_times_vs": a5_l0["max_abs_w_times_vs"],
    }

    # -- P2 ---------------------------------------------------------------
    f_wide = a5_l0["frac_over"]["1.0"]
    f_narrow = _layer(rows[ARM_NARROW], 0)["frac_over"]["1.0"]
    out["P2"] = {
        "verdict": ("DIRECTION HELD" if f_wide > f_narrow else
                    "DIRECTION REVERSED" if f_wide < f_narrow else
                    "INDISTINGUISHABLE"),
        "frac_over_1_d1481": f_wide,
        "frac_over_1_d512": f_narrow,
        "note": ("direction only -- not step-matched, n=1 checkpoint per cell, "
                 "no interval; no functional form may be fitted to two points"),
    }

    # -- P3: one-sided ----------------------------------------------------
    s = a5_l0["max_chain_log_gain"]
    out["P3"] = {
        "verdict": ("CONSISTENT WITH" if s >= P3_BAR_LOG else "REFUTED AS SUFFICIENT"),
        "max_chain_log_gain": s,
        "max_chain_log10_gain": a5_l0["max_chain_log10_gain"],
        "bar_log": P3_BAR_LOG,
        "n_chains_expanding": a5_l0["n_chains_expanding"],
        "n_chains": a5_l0["n_chains"],
        "power": ("ONE-SIDED. A failure is decisive; a hold is CONSISTENCY, not "
                  "proof -- sum_t log|g_t| ignores the grad_spike injections and "
                  "the sign cancellation in grad_cur = gf + gs, and this "
                  "checkpoint is 138 steps before the failing step."),
        "may_not_be_called_proven": True,
    }

    # -- P4 ---------------------------------------------------------------
    if rows[ARM_WIDE]["n_layers"] > 1:
        a5_l1 = _layer(rows[ARM_WIDE], 1)
        f_ok = a5_l0["frac_over"]["1.0"] > a5_l1["frac_over"]["1.0"]
        s_ok = a5_l0["max_chain_log_gain"] > a5_l1["max_chain_log_gain"]
        out["P4"] = {
            "verdict": ("DIRECTION HELD" if (f_ok and s_ok) else
                        "DIRECTION REVERSED" if not (f_ok or s_ok) else "SPLIT"),
            "layer0": {"frac_over_1": a5_l0["frac_over"]["1.0"],
                       "max_chain_log_gain": a5_l0["max_chain_log_gain"]},
            "layer1": {"frac_over_1": a5_l1["frac_over"]["1.0"],
                       "max_chain_log_gain": a5_l1["max_chain_log_gain"]},
            "note": ("direction only; a held P4 does not establish that the "
                     "layer-0 tail CAUSES the layer-0 locus"),
        }

    # -- P5 ---------------------------------------------------------------
    ctrl = rows[CONTROL_WIDE]
    over = [ly["frac_over"]["1.0"] for ly in ctrl["layers"]]
    out["P5"] = {
        "verdict": "HELD" if all(v == 0.0 for v in over) else "REFUTED",
        "frac_over_1_per_layer": over,
        "max_abs_g_per_layer": [ly["max_abs_g"] for ly in ctrl["layers"]],
        "bound": ctrl["lif_closed_form_bound"],
    }

    out["controls_summary"] = [
        {"leg": lg, "arch": rows[lg]["arch"], "d_model": rows[lg]["d_model"],
         "max_abs_g_per_layer": [ly["max_abs_g"] for ly in rows[lg]["layers"]]}
        for lg in sorted(rows)
    ]
    out["verdict"] = _headline(out)
    return out


def _headline(out: dict) -> str:
    if out["P1"]["verdict"] == "REFUTED":
        return ("P1 REFUTED -- the factor does not exceed 1 at the width the arm "
                "died at. §2 stands as algebra and is not this divergence's "
                "explanation. Leg B does not run.")
    if out["P3"]["verdict"] == "REFUTED AS SUFFICIENT":
        return ("P1 HELD, P3 REFUTED AS SUFFICIENT -- the factor exceeds 1 but "
                "the homogeneous product cannot reach the observed magnitude. "
                "Reported as a falsified derivation.")
    return ("P1 HELD and P3 CONSISTENT WITH -- the premise holds. Leg B is what "
            "decides whether the first non-finite value is born in the recursion "
            "or in a downstream reduction; until it runs, nothing here is proven.")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="docs/reports/data/exp_016_reset_jacobian.json")
    ap.add_argument("--out", default="docs/reports/data/exp_016_results.json")
    ap.add_argument("--print-bars", action="store_true")
    args = ap.parse_args(argv)

    if args.print_bars:
        print(BARS)
        return 0

    p = Path(args.data)
    if not p.is_absolute():
        p = _REPO / p
    if not p.exists():
        print(f"missing {p}", file=sys.stderr)
        return 1

    out = resolve(json.loads(p.read_text(encoding="utf-8")))
    print(json.dumps(out, indent=1))

    q = Path(args.out)
    if not q.is_absolute():
        q = _REPO / q
    q.parent.mkdir(parents=True, exist_ok=True)
    q.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    print(f"\nwrote {q}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
