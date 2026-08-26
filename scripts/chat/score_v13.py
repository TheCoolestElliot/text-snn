"""Score the v13 round against the bars `PREDICTION_v13.md` fixed in advance.

NOT PART OF THE RESEARCH PROTOCOL. `snnchat` is outside the Phase-1..5 study.
Trains nothing; adopts nothing; edits `experiments/chat/SHIPPED` under no
outcome.

ORDERING, RECORDED RATHER THAN LEFT TO A COMMIT MESSAGE
--------------------------------------------------------
`04_phase4_interim.md` §20.7 records `EXP_020`'s and `EXP_021`'s entry condition
-- the resolver committed before any bpc is read -- being missed, and that no
later commit can supply the ordering. So the ordering here is stated plainly:

  * every bar below was fixed in `PREDICTION_v13.md` **before either arm was
    trained**, and that file's sha256 was stamped at `027542a6...` before launch;
  * **`chat-v13-ctl`'s bpc WAS read before this file was written.** It had to be:
    `PREDICTION_v13.md` §3 makes the control landing inside
    `[1.1721929, 1.1757542]` an entry condition for P1 resolving at all, and a
    guard that is only checked afterwards is not a guard. It read 1.1745948,
    in band;
  * **`chat-v13-det`'s bpc did not exist when this file was written.** That is
    the arm P1's verdict actually turns on.

Nothing in this file computes a threshold. Every constant below is transcribed
from `PREDICTION_v13.md` and the transcription is asserted against the four
committed replicates it was derived from, so a typo fails loudly rather than
moving a bar.

    python scripts/chat/score_v13.py                 # P1 only, no GPU
    python scripts/chat/score_v13.py --probes        # P1 + P2 + P3, needs the GPU
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
CHAT = REPO / "experiments" / "chat"

CTL, DET = "chat-v13-ctl", "chat-v13-det"

#: The four committed replicates of this recipe, seeds 0-3. `SD_seed` is derived
#: from THESE, not asserted as a literal, so the bar cannot drift from its
#: evidence.
REPLICATES = {
    "chat-v3d-aligned": 1.1742902997363112,
    "chat-v3d-aligned-s1": 1.1750910985023768,
    "chat-v3d-aligned-s2": 1.1732632780631018,
    "chat-v3d-aligned-s3": 1.1732494564146139,
}

#: Transcribed from `PREDICTION_v13.md` §3. Checked against REPLICATES below.
PREREG = {
    "sd_seed": 0.0008903,
    "se_diff": 0.0012591,
    "bar_2sigma": 0.0025182,
    "control_band": (1.1721929, 1.1757542),
    "prediction_sha256": "027542a63da7db72c029e01d4f19759d7050dc4cbad0052ba99b01925f8f3107",
}

#: `GRADIENT_BOUND_NOTE.md` §2 / §4, on `chat-v3d-aligned/ckpt_best.pt`.
SHIPPED_REFERENCE = {
    "max_abs_g_L0": 9.77616, "max_abs_w_times_vs": 14453.56,
    "l0_tau_median": 2.672562, "l0_frac_under_tau8": 0.888671875,
    "l0_input_gain_median": 20.928251,
}


def _check_transcription() -> dict:
    """The bar must equal what its stated evidence implies. Fails loudly."""
    vals = list(REPLICATES.values())
    sd = statistics.stdev(vals)
    se = sd * math.sqrt(2.0)
    mean = statistics.mean(vals)
    checks = {
        "sd_seed": (round(sd, 7), PREREG["sd_seed"]),
        "se_diff": (round(se, 7), PREREG["se_diff"]),
        "bar_2sigma": (round(2 * se, 7), PREREG["bar_2sigma"]),
        "band_lo": (round(mean - 2 * sd, 7), PREREG["control_band"][0]),
        "band_hi": (round(mean + 2 * sd, 7), PREREG["control_band"][1]),
    }
    bad = {k: v for k, v in checks.items() if v[0] != v[1]}
    if bad:
        raise SystemExit(
            f"TRANSCRIPTION ERROR: PREDICTION_v13.md's constants do not match "
            f"the replicates they are derived from: {bad}")
    return {"n_replicates": len(vals), "mean": mean, "sd_seed": sd,
            "se_diff": se, "recomputed_ok": True}


def _prediction_hash() -> dict:
    import hashlib
    p = REPO / "docs" / "chat" / "PREDICTION_v13.md"
    got = hashlib.sha256(p.read_bytes()).hexdigest()
    return {"path": str(p.relative_to(REPO)), "sha256": got,
            "matches_stamp": got == PREREG["prediction_sha256"]}


def _summary(run: str) -> dict:
    p = CHAT / run / "summary.json"
    if not p.exists():
        raise SystemExit(f"{p} does not exist -- run not finished")
    return json.loads(p.read_text(encoding="utf-8"))


def _n_divergences(run: str) -> int:
    p = CHAT / run / "log.jsonl"
    if not p.exists():
        return -1
    n = 0
    for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            if json.loads(line).get("event") == "divergence":
                n += 1
        except Exception:
            continue
    return n


def p1() -> dict:
    ctl, det = _summary(CTL), _summary(DET)
    c, d = ctl["val"]["weighted"], det["val"]["weighted"]
    delta = d - c
    lo, hi = PREREG["control_band"]
    bar = PREREG["bar_2sigma"]

    # THE GUARD FIRST. A bar on a difference whose other term is unconstrained is
    # the defect EXP_012's Y5, EXP_013's N1 and EXP_022's H4 all fired.
    in_band = lo <= c <= hi
    if not in_band:
        verdict = "NOT RUN"
    elif delta > bar:
        verdict = "worse"
    elif delta < -bar:
        verdict = "better"
    else:
        verdict = "unresolved"

    return {
        "bar": "P1 -- held-out weighted bpc, det minus ctl, negative is better",
        "control_weighted": c,
        "detached_weighted": d,
        "delta": delta,
        "delta_in_sd_seed": delta / PREREG["sd_seed"],
        "delta_in_se": delta / PREREG["se_diff"],
        "bar_2sigma": bar,
        "control_band": [lo, hi],
        "control_in_band": in_band,
        "verdict": verdict,
        "steps": {CTL: ctl["steps"], DET: det["steps"]},
        "wall_clock_s": {CTL: ctl["wall_clock_s"], DET: det["wall_clock_s"]},
        "n_divergences": {CTL: _n_divergences(CTL), DET: _n_divergences(DET)},
        "per_source": {
            k: {"ctl": ctl["val"].get(k), "det": det["val"].get(k),
                "delta": (det["val"].get(k, float("nan"))
                          - ctl["val"].get(k, float("nan")))}
            for k in sorted(set(ctl["val"]) | set(det["val"]))
        },
    }


def _run_probe(script: str, run: str, out: Path) -> dict:
    out.parent.mkdir(parents=True, exist_ok=True)
    r = subprocess.run(
        [sys.executable, f"scripts/chat/{script}", "--run", run,
         "--out", str(out)], cwd=REPO, capture_output=True, text=True)
    if r.returncode != 0:
        return {"error": r.stderr[-2000:]}
    return json.loads(out.read_text(encoding="utf-8"))["result"]


def probes(outdir: Path) -> dict:
    """P2 and P3. Markers, no bars -- see PREDICTION_v13.md §4 and §5."""
    out: dict = {"p2": {}, "p3": {}}
    for run in (CTL, DET):
        j = _run_probe("reset_jacobian_probe.py", run,
                       outdir / f"jacobian_{run}.json")
        if "error" in j:
            out["p2"][run] = j
            continue
        ls = j["layers"]
        out["p2"][run] = {
            "t2_holds": j["t2_holds"],
            "max_abs_g_L0": ls[0]["max_abs_g"],
            "max_abs_g_over_lif_bound_L0": ls[0]["max_abs_g_over_lif_bound"],
            "max_abs_w_times_vs": max(x["max_abs_w_times_vs"] for x in ls),
            "layers_over_1": sum(1 for x in ls if x["max_abs_g"] > 1.0),
            "n_layers": len(ls),
        }
        c = _run_probe("slow_channel_census.py", run,
                       outdir / f"census_{run}.json")
        if "error" in c:
            out["p3"][run] = c
            continue
        l0 = c["layers"][0]
        under = sum(b["n_channels"] for b in l0["by_tau"]
                    if b["tau"] in ("<2", "2-8"))
        out["p3"][run] = {
            "l0_tau_median": l0["tau"]["q0.5"],
            "l0_frac_under_tau8": under / l0["d"],
            "l0_input_gain_median": l0["input_gain_exp_minus_thr_log"]["q0.5"],
            "pinned_all_layers": sum(x["n_pinned"] for x in c["layers"]),
        }
    # P2 is unreadable unless the local scan reproduces the committed one.
    out["p2_readable"] = all(
        isinstance(v, dict) and v.get("t2_holds") is True
        for v in out["p2"].values())
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--probes", action="store_true",
                    help="also run P2/P3 (needs the GPU; do not use while a "
                         "timed run is in flight)")
    ap.add_argument("--out",
                    default="experiments/chat/_probe/v13/score_v13.json")
    args = ap.parse_args(argv)

    transcription = _check_transcription()
    phash = _prediction_hash()
    res = p1()

    print("PREDICTION_v13 -- results\n")
    print(f"  pre-registration sha256 matches the pre-launch stamp: "
          f"{phash['matches_stamp']}")
    print(f"  bar recomputed from its own evidence: "
          f"{transcription['recomputed_ok']}  "
          f"(SD_seed {transcription['sd_seed']:.7f}, n=4)\n")
    print(f"  {'':22} {'ctl':>12} {'det':>12} {'delta':>12}")
    for k, v in res["per_source"].items():
        mark = "  <-- P1" if k == "weighted" else ""
        print(f"  {k:22} {v['ctl']:>12.7f} {v['det']:>12.7f} "
              f"{v['delta']:>+12.7f}{mark}")
    print(f"\n  control in band {res['control_band']}: {res['control_in_band']}")
    print(f"  delta = {res['delta']:+.7f} = {res['delta_in_se']:+.2f} SE "
          f"against a 2-sigma bar of +-{res['bar_2sigma']:.7f}")
    print(f"  divergences/rollbacks: {res['n_divergences']}")
    print(f"\n  P1 VERDICT: {res['verdict'].upper()}")

    payload = {
        "round": "v13", "not_a_research_artifact": True,
        "prediction": phash, "bar_transcription": transcription,
        "P1": res,
    }
    if args.probes:
        payload["probes"] = probes(Path(args.out).parent)
        pr = payload["probes"]
        print(f"\n  P2 (marker) readable: {pr['p2_readable']}")
        for run, v in pr["p2"].items():
            if "error" in v:
                print(f"    {run}: ERROR")
                continue
            print(f"    {run:16} L0 max|b*dv| {v['max_abs_g_L0']:.5f} "
                  f"({v['max_abs_g_over_lif_bound_L0']:.2f}x)  "
                  f"max|w*vs| {v['max_abs_w_times_vs']:.0f}  "
                  f"layers>1 {v['layers_over_1']}/{v['n_layers']}")
        print(f"    {'shipped (ref)':16} L0 max|b*dv| "
              f"{SHIPPED_REFERENCE['max_abs_g_L0']:.5f}  max|w*vs| "
              f"{SHIPPED_REFERENCE['max_abs_w_times_vs']:.0f}")
        print("\n  P3 (marker): layer 0")
        for run, v in pr["p3"].items():
            if "error" in v:
                print(f"    {run}: ERROR")
                continue
            print(f"    {run:16} tau q50 {v['l0_tau_median']:.3f}  "
                  f"frac<tau8 {v['l0_frac_under_tau8']:.1%}  "
                  f"gain q50 {v['l0_input_gain_median']:.2f}  "
                  f"pinned {v['pinned_all_layers']}")
        print(f"    {'shipped (ref)':16} tau q50 "
              f"{SHIPPED_REFERENCE['l0_tau_median']:.3f}  frac<tau8 "
              f"{SHIPPED_REFERENCE['l0_frac_under_tau8']:.1%}  gain q50 "
              f"{SHIPPED_REFERENCE['l0_input_gain_median']:.2f}")

    p = Path(args.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    print(f"\n  wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
