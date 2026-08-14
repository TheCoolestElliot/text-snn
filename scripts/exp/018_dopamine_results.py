"""EXP_018: resolve every pre-registered bar, exactly as written.

    python scripts/exp/018_dopamine_results.py

Committed **before any run's bpc is read**, which is the standard `010`, `011`,
`014`, `015` and `017` set. It reads artifacts and resolves the predictions; it
trains nothing, adopts nothing and ranks nothing.

WHAT THIS FILE MAY NOT DO
-------------------------
It may not adopt anything, may not rank anything, and **may not widen a bar that
fails**. A bar is reported as it fired; a corrected rule is referred to the next
phase, never applied retroactively (`CONTRIBUTING.md` §3).

THREE DEFENCES ENCODED IN CODE RATHER THAN IN PROSE
----------------------------------------------------
Phase 4 has had four bars misfire — `EXP_012` Y5, `EXP_013` N1, `EXP_015` P1/P2,
`EXP_016` P3 — and the three guards those failures bought are structural here:

  1. `UNRESOLVED` is a first-class verdict, **distinct from `NOT RUN`**. The
     first says the design lacked the power to separate the arms; the second
     says there was nothing to separate. They are different sentences.
  2. **Both absolute means are always printed**, never the difference alone.
     `EXP_013`'s N1 fired on a difference whose two terms were both catastrophic.
  3. **A diverged run resolves `DIVERGED` / `NOT RUN`, never `UNRESOLVED`.**

THE STATISTICS, AND WHY THEY ARE HAND-ROLLED
---------------------------------------------
`EXP_018` §4 sets its bars at a **95 % Welch confidence interval**, not at the
±2·se band `017_detach_results.py` uses. At n = 5 v 5 those differ by 15 %
(t = 2.306 against 2.000 at df = 8), which is not negligible against a bar, so
the interval is computed properly with Welch–Satterthwaite degrees of freedom.

`scipy` is not in this project's audited stack, so the Student-t quantile is
implemented here from `math` alone: the regularised incomplete beta by Lentz's
continued fraction, then a bisection for the quantile. **A hand-rolled special
function is exactly the kind of second implementation that is silently wrong**,
so `_selfcheck_t()` reproduces five published critical values it did not
produce, to 1e-4, and the script **aborts** if any of them misses.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO / "src") not in sys.path:
    sys.path.insert(0, str(_REPO / "src"))

RUNS = _REPO / "experiments" / "runs"
DATA = _REPO / "docs" / "reports" / "data"
MANIFEST = DATA / "exp_018_run_manifest.json"
CALIB = DATA / "exp_018_da_calibration.json"
HORIZON = DATA / "exp_018_horizon.json"

# ---------------------------------------------------------------------------
# Bars, TRANSCRIBED from experiments/logs/EXP_018_dopamine.md. Nothing below
# recomputes them.
# ---------------------------------------------------------------------------

#: §4.0. The confidence level every bar is set at.
ALPHA = 0.05

#: §4.0's design-sizing sigma -- the committed Phase-2 baseline figure. Used
#: ONLY to reproduce the pre-registered MDE. Every verdict below uses the sigma
#: measured on these runs' own seeds (CONTRIBUTING.md §4: sigma is not a project
#: constant).
SIGMA_DESIGN = 0.00461

#: §4.0's pre-registered minimum detectable differences, at 80 % power.
MDE_D1 = 0.0093
MDE_D2 = 0.0113

#: §4 D4's alarm threshold on the learned sensitivity.
K_ENGAGED_FLOOR = 1e-3

#: §2.5's measured cost ratios, for the record. Not a bar.
COST_RATIO = {"da_mult": 1.640, "da_rolled": 1.644, "da_add": 1.509}

#: §4 D5's scale markers. Quoted, never a bar -- the horizon is an integer on a
#: dense lattice and CONTRIBUTING.md §3 forbids a threshold on a reachable value.
LIF_HORIZON = 7
TWOCOMP_HORIZON = 47

ARMS: dict[str, tuple[str, ...]] = {
    "da_anchor": tuple(f"da_anchor_s{s}" for s in range(5)),
    "da_mult":   tuple(f"da_mult_s{s}" for s in range(5)),
    "da_rolled": tuple(f"da_rolled_s{s}" for s in range(3)),
    "da_add":    tuple(f"da_add_s{s}" for s in range(3)),
}
PROTOCOLS = ("carried", "fresh")

#: The headline protocol. Both are always reported; this is the one the bars are
#: set on, because it is the protocol published enwik8 figures use.
HEADLINE = "carried"


# ---------------------------------------------------------------------------
# Student-t, from math alone
# ---------------------------------------------------------------------------

def _betacf(a: float, b: float, x: float) -> float:
    """Continued fraction for the incomplete beta (Lentz). NR §6.4."""
    tiny, eps, itmax = 1e-30, 3e-16, 300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    if abs(d) < tiny:
        d = tiny
    d = 1.0 / d
    h = d
    for m in range(1, itmax + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            break
    return h


def _betai(a: float, b: float, x: float) -> float:
    """Regularised incomplete beta I_x(a, b)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lbeta = (math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
             + a * math.log(x) + b * math.log1p(-x))
    front = math.exp(lbeta)
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


def t_sf(t: float, df: float) -> float:
    """P(T > t) for Student's t with `df` degrees of freedom."""
    if df <= 0:
        return float("nan")
    p_two = _betai(df / 2.0, 0.5, df / (df + t * t))
    return p_two / 2.0 if t > 0 else 1.0 - p_two / 2.0


def t_ppf(p: float, df: float) -> float:
    """Inverse CDF by bisection. Monotone, so bisection is exact to tolerance."""
    if df <= 0:
        return float("nan")
    lo, hi = -300.0, 300.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if (1.0 - t_sf(mid, df)) < p:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


#: Published two-sided 97.5 % critical values. This file did not produce them;
#: reproducing them is what earns the right to use the implementation above
#: (CONTRIBUTING.md §5, "make every probe reproduce a number it did not produce").
_T_TABLE = {2: 4.30265, 6: 2.44691, 8: 2.30600, 30: 2.04227, 1000: 1.96234}


def _selfcheck_t() -> dict:
    worst, rows = 0.0, {}
    for df, want in _T_TABLE.items():
        got = t_ppf(0.975, float(df))
        rows[str(df)] = {"published": want, "computed": got, "abs_diff": abs(got - want)}
        worst = max(worst, abs(got - want))
    ok = worst < 1e-4
    if not ok:
        raise SystemExit(
            f"the hand-rolled Student-t is wrong: worst |diff| = {worst:.2e} "
            "against published critical values. No bar may be resolved with it.")
    return {"passed": ok, "worst_abs_diff": worst, "rows": rows}


def welch(a: list[float], b: list[float]) -> dict:
    """`mean(a) - mean(b)` with a Welch–Satterthwaite 95 % CI. BOTH means kept."""
    na, nb = len(a), len(b)
    ma, mb = statistics.fmean(a), statistics.fmean(b)
    va = statistics.variance(a) if na > 1 else 0.0
    vb = statistics.variance(b) if nb > 1 else 0.0
    se = math.sqrt(va / na + vb / nb) if (na and nb) else float("nan")
    if se > 0:
        num = (va / na + vb / nb) ** 2
        den = ((va / na) ** 2 / max(na - 1, 1)) + ((vb / nb) ** 2 / max(nb - 1, 1))
        df = num / den if den > 0 else float(na + nb - 2)
        tcrit = t_ppf(1.0 - ALPHA / 2.0, df)
        half = tcrit * se
    else:
        df, tcrit, half = float("nan"), float("nan"), 0.0
    delta = ma - mb
    return {
        "mean_a": ma, "sd_a": math.sqrt(va), "n_a": na,
        "mean_b": mb, "sd_b": math.sqrt(vb), "n_b": nb,
        "delta": delta, "welch_se": se, "df": df, "t_crit": tcrit,
        "ci95": [delta - half, delta + half],
        "excludes_zero": bool(se > 0 and (delta - half) * (delta + half) > 0),
        "delta_in_se": (delta / se) if se else None,
        "resolution_floor_bpc": 2.0 * half if se > 0 else None,
    }


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def _load_manifest() -> dict:
    return json.loads(MANIFEST.read_text(encoding="utf-8")) if MANIFEST.exists() else {}


def load_run(run: str) -> dict:
    """One run's test scores and status. Missing/diverged is a STATUS, not a gap."""
    d = RUNS / run
    row: dict = {"run": run, "present": d.exists(), "status": "NOT RUN"}
    ft, sm = d / "final_test.json", d / "summary.json"
    if not ft.exists() or not sm.exists():
        return row
    res = json.loads(ft.read_text(encoding="utf-8"))["results"]
    summary = json.loads(sm.read_text(encoding="utf-8"))
    row.update({
        "params": summary.get("params"),
        "wall_clock_s": summary.get("wall_clock_s"),
        "steps": summary.get("steps"),
        "bpc": {p: res[p]["bpc"] for p in PROTOCOLS},
        "firing_rate": {p: res[p].get("firing_rate") for p in PROTOCOLS},
    })
    finite = all(math.isfinite(v) for v in row["bpc"].values())
    row["status"] = "OK" if finite else "DIVERGED"
    return row


def _k_stats(run: str) -> dict:
    """D4: the learned sensitivity at step 20,000. CPU-only load."""
    ck = RUNS / run / "ckpt_final.pt"
    if not ck.exists():
        return {"present": False}
    import torch
    sd = torch.load(ck, map_location="cpu", weights_only=False)["model"]
    ks = {n: t for n, t in sd.items() if n.startswith("k.")}
    if not ks:
        return {"present": False, "note": "no k.* in state dict (not a dopamine run)"}
    per_layer = {n: {"abs_max": float(t.abs().max()), "rms": float(t.pow(2).mean().sqrt()),
                     "mean": float(t.mean()), "frac_nonzero": float((t != 0).float().mean())}
                 for n, t in sorted(ks.items())}
    return {"present": True, "per_layer": per_layer,
            "max_abs_over_layers": max(v["abs_max"] for v in per_layer.values())}


# ---------------------------------------------------------------------------
# Resolution
# ---------------------------------------------------------------------------

def _values(rows: dict, arm: str, protocol: str) -> tuple[list[float], list[str]]:
    """Finite bpc values for an arm, plus the names of the runs that are absent."""
    good, missing = [], []
    for run in ARMS[arm]:
        r = rows[run]
        if r["status"] == "OK":
            good.append(r["bpc"][protocol])
        else:
            missing.append(f"{run}:{r['status']}")
    return good, missing


def resolve_d1(rows: dict, protocol: str) -> dict:
    anchor, a_missing = _values(rows, "da_anchor", protocol)
    arm, m_missing = _values(rows, "da_mult", protocol)
    out = {"bar": "D1", "protocol": protocol,
           "statement": "delta = mean bpc(anchor) - mean bpc(da_mult) > 0 and the "
                        "95% Welch CI on delta excludes 0",
           "missing": a_missing + m_missing,
           "n_preregistered": [5, 5]}
    if len(anchor) < 2 or len(arm) < 2:
        out["verdict"] = "NOT RUN"
        out["note"] = ("fewer than two usable seeds on one side; a diverged or "
                       "absent run resolves NOT RUN, never UNRESOLVED")
        return out
    w = welch(anchor, arm)
    out.update(w)
    out["anchor_mean"] = w["mean_a"]
    out["arm_mean"] = w["mean_b"]
    if w["excludes_zero"] and w["delta"] > 0:
        out["verdict"] = "THE DOPAMINE SIGNAL HELPS"
    elif w["excludes_zero"] and w["delta"] < 0:
        out["verdict"] = "IT HURTS"
    else:
        out["verdict"] = "UNRESOLVED"
    out["provisional"] = bool(out["missing"])
    out["note"] = (
        f"this design resolves {w['resolution_floor_bpc']:.5f} bpc and no better; "
        "UNRESOLVED means the difference was not established, NOT that it is zero "
        "(EXP_018 §4.0)")
    return out


def resolve_d2(rows: dict, protocol: str, d1: dict) -> dict:
    arm, m_missing = _values(rows, "da_mult", protocol)
    ctrl, c_missing = _values(rows, "da_rolled", protocol)
    out = {"bar": "D2", "protocol": protocol,
           "statement": "D1 fires in the 'helps' direction AND "
                        "delta_align = mean bpc(rolled) - mean bpc(da_mult) > 0 "
                        "with a 95% Welch CI excluding 0",
           "algebra": "delta(A) - delta(S) against a shared anchor IS bpc(S) - bpc(A); "
                      "the anchor cancels exactly, so this does not inherit its variance",
           "missing": m_missing + c_missing,
           "n_preregistered": [5, 3]}
    if len(arm) < 2 or len(ctrl) < 2:
        out["verdict"] = "NOT RUN"
        return out
    w = welch(ctrl, arm)
    out.update(w)
    out["rolled_mean"] = w["mean_a"]
    out["arm_mean"] = w["mean_b"]
    out["d1_verdict"] = d1.get("verdict")
    if d1.get("verdict") != "THE DOPAMINE SIGNAL HELPS":
        out["verdict"] = "NOT RUN"
        out["note"] = ("D1 did not fire in the 'helps' direction, so there is no "
                       "gain to attribute. The alignment comparison is reported "
                       "below for the record and carries no verdict.")
        return out
    if w["excludes_zero"] and w["delta"] > 0:
        out["verdict"] = "THE ALIGNMENT IS THE MECHANISM"
    elif w["excludes_zero"] and w["delta"] < 0:
        out["verdict"] = "THE CONTROL BEATS THE ARM"
        out["note"] = ("the most informative outcome in §4's table: whatever the "
                       "arm is doing, it is not the RPE")
    else:
        out["verdict"] = "THE MECHANISM IS NOT ESTABLISHED"
        out["note"] = ("the gain may be a perturbation effect; report it as "
                       "unattributed, not as the RPE's")
    return out


def resolve_d3(rows: dict, protocol: str) -> dict:
    add, add_missing = _values(rows, "da_add", protocol)
    anchor, _ = _values(rows, "da_anchor", protocol)
    mult, _ = _values(rows, "da_mult", protocol)
    out = {"bar": "D3", "protocol": protocol, "type": "MARKER, no verdict",
           "missing": add_missing,
           "note": "no direction was predicted and no bar was set; n = 3 is "
                   "underpowered by construction and EXP_018 §4 says so"}
    if len(add) < 2:
        out["verdict"] = "NOT RUN"
        return out
    out["add_mean"] = statistics.fmean(add)
    out["add_sd"] = statistics.stdev(add) if len(add) > 1 else 0.0
    out["add_n"] = len(add)
    if len(anchor) >= 2:
        out["vs_anchor"] = welch(anchor, add)
    if len(mult) >= 2:
        out["vs_mult"] = welch(mult, add)
    out["verdict"] = "MARKER"
    return out


def resolve_d4(rows: dict) -> dict:
    out = {"bar": "D4", "type": "MARKER + ALARM", "floor": K_ENGAGED_FLOOR,
           "statement": "report max|k| per run; if max|k| < 1e-3 the arm never "
                        "engaged and D1's null is about the optimiser, not the "
                        "mechanism",
           "runs": {}}
    worst = None
    for arm in ("da_mult", "da_rolled", "da_add"):
        for run in ARMS[arm]:
            if rows[run]["status"] == "NOT RUN":
                continue
            st = _k_stats(run)
            out["runs"][run] = st
            if st.get("present"):
                m = st["max_abs_over_layers"]
                worst = m if worst is None else min(worst, m)
    out["min_max_abs_over_runs"] = worst
    if worst is None:
        out["verdict"] = "NOT RUN"
    elif worst < K_ENGAGED_FLOOR:
        out["verdict"] = "ALARM: THE ARM NEVER ENGAGED"
        out["consequence"] = ("a D1 null is a statement about the initialisation "
                              "and the optimiser, not about the mechanism. The "
                              "mechanism is recorded as NOT MEASURED.")
    else:
        out["verdict"] = "ENGAGED"
        out["consequence"] = "a D1 null is about the mechanism, not the optimiser"
    return out


def resolve_d5() -> dict:
    out = {"bar": "D5", "type": "MARKER, no verdict",
           "scale_markers": {"lif_horizon": LIF_HORIZON,
                             "twocomp_horizon": TWOCOMP_HORIZON},
           "note": "the horizon is an integer on a dense lattice, so no threshold "
                   "is placed on it (CONTRIBUTING.md §3). The continuous "
                   "per-context bpc curve is the thing a future bar may use."}
    if not HORIZON.exists():
        out["verdict"] = "NOT RUN"
        out["reason"] = f"{HORIZON.relative_to(_REPO)} absent"
        return out
    hz = json.loads(HORIZON.read_text(encoding="utf-8"))
    out["verdict"] = "MARKER"
    out["horizon"] = hz.get("horizons", hz.get("horizon"))
    out["per_context"] = hz.get("per_context", hz.get("curves"))
    return out


def sigma_markers(rows: dict) -> dict:
    """sigma, RE-MEASURED on these seeds. CONTRIBUTING.md §4."""
    out = {"design_sigma_used_only_for_power": SIGMA_DESIGN,
           "preregistered_mde": {"D1": MDE_D1, "D2": MDE_D2}, "measured": {}}
    for arm in ARMS:
        for protocol in PROTOCOLS:
            vals, _ = _values(rows, arm, protocol)
            if len(vals) > 1:
                out["measured"][f"{arm}/{protocol}"] = {
                    "sd": statistics.stdev(vals), "n": len(vals), "df": len(vals) - 1,
                    "mean": statistics.fmean(vals),
                    "caveat": ("a 3-seed sigma is a marker quoted with its df and "
                               "is never promoted" if len(vals) < 5 else None),
                }
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(DATA / "exp_018_results.json"))
    args = ap.parse_args(argv)

    selfcheck = _selfcheck_t()
    manifest = _load_manifest()
    rows = {run: load_run(run) for arm in ARMS for run in ARMS[arm]}

    results: dict = {
        "experiment": "EXP_018",
        "prereg": "experiments/logs/EXP_018_dopamine.md",
        "adopts_nothing": True, "ranks_nothing": True,
        "prereg_unchanged": manifest.get("prereg_unchanged"),
        "student_t_selfcheck": selfcheck,
        "cost_ratio_measured": COST_RATIO,
        "tau": json.loads(CALIB.read_text(encoding="utf-8"))["tau_recommended"]
               if CALIB.exists() else None,
        "runs": rows,
        "sigma": sigma_markers(rows),
        "bars": {},
    }

    for protocol in PROTOCOLS:
        d1 = resolve_d1(rows, protocol)
        d2 = resolve_d2(rows, protocol, d1)
        d3 = resolve_d3(rows, protocol)
        results["bars"][protocol] = {"D1": d1, "D2": d2, "D3": d3}
    results["bars"]["D4"] = resolve_d4(rows)
    results["bars"]["D5"] = resolve_d5()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2, default=str) + "\n", encoding="utf-8")

    # ---- the scoreboard -------------------------------------------------
    print("=" * 78)
    print("EXP_018 -- a broadcast reward prediction error")
    print("=" * 78)
    ok = sum(r["status"] == "OK" for r in rows.values())
    print(f"runs usable: {ok}/{len(rows)}   prereg unchanged: "
          f"{results['prereg_unchanged']}   t self-check worst "
          f"{selfcheck['worst_abs_diff']:.2e}")
    print()
    for arm in ARMS:
        vals, missing = _values(rows, arm, HEADLINE)
        if vals:
            sd = statistics.stdev(vals) if len(vals) > 1 else float("nan")
            print(f"  {arm:<11s} n={len(vals)}  carried {statistics.fmean(vals):.5f} "
                  f"(sd {sd:.5f})" + (f"   MISSING {missing}" if missing else ""))
        else:
            print(f"  {arm:<11s} NOT RUN  {missing}")
    print()
    for protocol in PROTOCOLS:
        print(f"-- {protocol} " + "-" * 60)
        for name in ("D1", "D2", "D3"):
            b = results["bars"][protocol][name]
            extra = ""
            if "delta" in b:
                lo, hi = b["ci95"]
                extra = (f"  delta {b['delta']:+.5f}  95% CI [{lo:+.5f}, {hi:+.5f}]"
                         f"  df {b['df']:.1f}")
            print(f"  {name:<3s} {b.get('verdict', '-'):<32s}{extra}")
    print()
    d4 = results["bars"]["D4"]
    print(f"  D4  {d4['verdict']:<32s}  min over runs of max|k| = "
          f"{d4['min_max_abs_over_runs']}")
    d5 = results["bars"]["D5"]
    print(f"  D5  {d5['verdict']:<32s}  {d5.get('reason', '')}")
    print(f"\nwritten: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
