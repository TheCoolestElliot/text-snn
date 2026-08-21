"""Every reading the beyond-horizon component has ever produced, from committed
artifacts only. Zero GPU, no checkpoint, no model.

    python scripts/audit/11_beyond_horizon_census.py
    python scripts/audit/11_beyond_horizon_census.py --out docs/reports/data/audit_11_beyond_horizon_census.json

WHY THIS EXISTS
---------------
`EXP_024` section 4 states, as the sentence that sets H1's expectation:

    "The beyond-horizon component has never been observed outside +/-0.0016 on
     any arm, so an UNRESOLVED H1 is the modal outcome and is not a
     disappointment."

**The premise is sound as scoped and false as generalised, and the difference
matters.** `EXP_024`'s own H1 row says *"Six arms have moved this component by at
most 0.0015 (section 0)"*, section 6 item 5 says *"across six arms"*, and section
0 names those six -- all true, all inside the band. Only the unqualified prose
restatement says *"on any arm"*, and on the wider committed record that is false.
The same statistic -- the `EXP_004` section 10.3 three-way cut, positive = better
-- reads outside the band in seven distinct committed arm-gains, one of them
`EXP_020`'s, which is the very experiment `EXP_024` section 2.4 draws H1's noise
floor from. Nothing here is a new measurement: every row is read out of a file
under `docs/reports/data/` that was committed when its experiment closed.

**The cleanest counterexamples are not the largest ones.** `tokenshift` reads
**+0.00565 on an arm that WON** (total +0.04847) -- 3.5x the band, 12x the
bitwise-copy floor, and with no question of compression attached to it -- and
`io_sp05` reads **+0.00312** on a winning arm (+0.03560). Neither is in section
0's table of six.

THREE CLASSES OF BLOCK, AND ONLY ONE OF THEM IS WHAT THE SENTENCE IS ABOUT
--------------------------------------------------------------------------
Sweeping both spellings finds **41 blocks across 12 artifacts**, and most do not
bear on the claim. They are classified from the JSON path, by a rule stated here
rather than guessed per row:

* `arm_gain` -- `...decomposition` and `...decomposition_vs_*`. **This is an
  arm's gain against a reference, which is the quantity section 4's sentence is
  about**, and the only class scored below.
* `gap_composition` -- `remaining_gap_vs_gru` and `i5_gap_decomposition`. These
  measure how the **gap to the GRU anchor** divides up, not any arm's gain:
  `010_phase4_arm_results.py:569` computes the first as `decompose(curve,
  gru_curve)` -- the same function as the arm's own block with the arm in the
  *reference* slot, so it is the GRU's gain over the arm. Scoring it against
  section 4's sentence would score the gap `EXP_024` section 0 already cites as
  its motivation. **Excluded**, and not because it is large.
* `bar` -- blocks under `bars.`. A pre-registered bar's stored value, duplicating
  an arm row. **Excluded.**

Within `arm_gain`, one arm can appear twice: once against its own anchor
(`decomposition`) and once against the Phase-2 baseline (`decomposition_vs_*`).
The second is marked `secondary_reference` and not counted as an independent
arm. **For `tokenshift` and `threshold` the two blocks are bit-identical**,
because those arms' reference already *is* the Phase-2 baseline; the only genuine
second references are `twocomp_distill` and `compose`, which the family filter
removes anyway, so this exclusion moves no tally. `if_binin` is stored
bit-identically in `EXP_020` and `EXP_022` and is de-duplicated across sources by
`duplicate_of`.

WHAT THE CENSUS DOES **NOT** SAY, AND THIS IS THE POINT
-------------------------------------------------------
**Finding readings outside +/-0.0016 is not finding arms that extended reach.**
Two of the three bar-clearing readings are arms that **LOST** overall --
`if_fold` (total -0.09453, beyond +0.01400) and `if_foldwide` (total -0.01366,
beyond +0.01708) -- so a positive beyond-horizon number on them cannot be read
as a win, and `compression_suspect` flags exactly that pattern.

**But the sign pattern is not itself diagnostic, and this file will not pretend
otherwise.** From `scripts/exp/010_phase4_arm_results.py:190-196` the quantity is

    beyond = (ref[CMAX] - arm[CMAX]) - (ref[h] - arm[h]) = g(CMAX) - g(h)

-- a **difference of gains**, invariant under any constant shift of the arm's
curve, so an arm uniformly worse by any amount scores `beyond = 0`. A negative
total does not mechanically induce a positive beyond. The negative
`beyond_horizon_share` is `beyond/total` restating the two signs it is built
from, and is **not** independent corroboration. And the pattern fails its own
control: `if_binin` lost **more** than `if_foldwide` (-0.04478 against -0.01366)
and read beyond **-0.00477**, with its horizon *falling* to `[6, 6, 7]`.

**`compression_suspect` is therefore a flag to be checked, not a verdict**, and
this script checks it against the horizon rather than leaving it standing.

`exp_020_memory_horizon.json` records both arms' 2-sigma memory
horizon at `[9, 10, 9]` and `[9, 8, 9]` against the anchor's `[7, 7, 7]` and a
**bitwise copy of the anchor** at `[7]`. Two markers that a compression artifact
would be expected to separate instead **agree**, on 3 of 3 seeds each. Every such
row carries `compression_suspect_contradicted_by_horizon: true` so that no
reader takes the flag as settled.

**That is not evidence of reach either**, and this script asserts none: the
integer horizon has **no sigma anywhere in this project** (`EXP_017` section 10
item 8), `EXP_020` section 4 pre-registered H5 with no bar on it, `EXP_024`'s own
H9 carries it as a marker with no verdict, and n = 3 cannot resolve it. **Two
markers agreeing is two markers agreeing.**

So the census supports a narrower claim than "the six-arm count is wrong", and
the narrower claim is the useful one: **the beyond-horizon statistic is noisier
and more artifact-prone than +/-0.0016 implies, `EXP_024`'s H1 bar of
2 sigma = 0.00922 is not protected against the compression case, and the two
committed arms that clear it while losing bits are exactly the two whose horizon
marker also moved.** `EXP_024`'s H4 -- the `EXP_001` section 8.4 anti-signature,
on near-context excess at `c = 1..4` -- is a partial guard, because a compressed
arm is worse near context; **whether it is a sufficient guard is not this
script's to say and is not asserted here.**

The two-compartment family's large readings (+0.132, +0.131) are **not** a
counterexample `EXP_024` overlooked: section 0 names the slow pole as "the one
thing that has ever moved reach". They are reported for completeness and
labelled `twocomp_family`.

WHAT IS AND IS NOT THIS SCRIPT'S BUSINESS
------------------------------------------
It **rules nothing**. Report section 8's open decisions are the author's,
`EXP_024` section 8 item 1 is the author's, and this file neither ranks an arm
nor recommends an experiment. It does not edit `EXP_024_gated_decay.md`: that log
is **committed** at `aff0d49`, and a commit -- not a stamp -- is what fixes it.
`EXP_024` has no run manifest yet (its own section 8 item 7 is an unmet entry
condition), so `verify_prereg_hash.py` reports `NO MANIFEST` for it and `git
diff` is the only check available for this log today. Either way
`CONTRIBUTING.md` section 2's "never rewrite the hypothesis" applies, so the
correction belongs in report section 8's prose, which is where this report has
recorded every other correction to a standing claim since rev 3.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
DATA = _REPO / "docs" / "reports" / "data"

#: BOTH committed spellings. `beyond_horizon_bpc` appears in five
#: `*_memory_horizon.json` files as `i5_gap_decomposition` -- the baseline-vs-GRU
#: gap, identical (+0.22794) in all five. It is gap-composition class and is
#: excluded from scoring either way, but sweeping one spelling and calling the
#: result "every reading" would be the same single-lookup defect this census's
#: own report note faults the audit tool for.
KEYS = ("beyond_horizon_gain_bpc", "beyond_horizon_bpc")
KEY = KEYS[0]

#: `EXP_024` section 4's stated band, and section 4's H1 bar. Both are quoted
#: from the pre-registration; neither is invented here.
EXP_024_BAND = 0.0016
EXP_024_H1_BAR = 0.00922

#: `EXP_020` section 9.5's floor, measured with a BITWISE copy of the anchor --
#: the first honest floor this statistic has had. `EXP_024` section 2.4 quotes it.
BITWISE_FLOOR = -0.0004747071187929386

#: Arms built on the second, slower state variable. `EXP_024` section 0 already
#: credits this family with the only movement in reach the project has measured,
#: so its readings are not offered as something the pre-registration missed.
TWOCOMP_FAMILY = ("twocomp", "compose", "detach", "distill")

#: The anchor an experiment's own horizon file measures its arms against. The
#: first name present in a given file wins.
ANCHOR_NAMES = ("if_anchor", "snn_beta0.5", "da_anchor", "anchor_twocomp_d512")


def _horizons(source: str) -> dict:
    """`{arm: [per-seed 2-sigma horizon]}` from the sibling horizon artifact.

    Joined so the compression flag can be CHECKED rather than trusted: an arm
    flagged `compression_suspect` whose horizon marker also moved is not
    obviously an artifact, and the JSON should say so without a reader having to
    open a second file. Empty when no sibling exists.
    """
    prefix = source.split("_arm_results")[0].split("_scaling_results")[0]
    for name in (f"{prefix}_memory_horizon.json", f"{prefix}_horizon.json"):
        p = DATA / name
        if not p.exists():
            continue
        try:
            doc = json.loads(p.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            return {}
        out = {}
        for arm, v in (doc.get("by_arm") or {}).items():
            paired = v.get("_paired", v) if isinstance(v, dict) else {}
            h = paired.get("horizon_2sigma_per_seed")
            if h:
                out[arm] = h
        return out
    return {}


def _median(xs):
    s = sorted(xs)
    n = len(s)
    return None if not n else (s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2)


def blocks(obj, path=""):
    """Yield `(json_path, block)` for every dict carrying the decomposition."""
    if isinstance(obj, dict):
        if any(k in obj for k in KEYS):
            yield path, obj
            return
        for k, v in obj.items():
            yield from blocks(v, f"{path}.{k}" if path else k)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from blocks(v, f"{path}[{i}]")


def _segments(jpath: str) -> list[str]:
    return [s for s in re.split(r"[.\[\]]", jpath) if s]


def _is_decomp(jpath: str) -> bool:
    """True if any SEGMENT is the decomposition container.

    Checking only the tail was wrong: `EXP_020` nests the arm below it
    (`H5_horizon.decomposition.if_fold`) and `EXP_021`-`EXP_023` nest it twice
    (`decomposition.arms.nm_local`), so a tail test classified every arm in four
    experiments as `other` and scored none of them.
    """
    return any(s == "decomposition" or s.startswith("decomposition_vs")
               for s in _segments(jpath))


def classify(jpath: str) -> str:
    if "remaining_gap" in jpath or "gap_decomposition" in jpath:
        return "gap_composition"
    if jpath.startswith("bars") or ".bars." in jpath:
        return "bar"
    if _is_decomp(jpath):
        return "arm_gain"
    return "other"


def identify(jpath: str, block: dict, doc) -> tuple[str, str]:
    """`(arm, reference_label)` recovered from the path and the block itself."""
    tail = jpath.rsplit(".", 1)[-1]
    ref = block.get("reference")
    if not ref:
        ref = ("phase2_baseline" if tail.startswith("decomposition_vs")
               else "own anchor")

    # `arms.<name>.<kind>` and `decomposition.arms.<name>` name the arm just
    # after an `arms` segment; `H5_horizon.decomposition.<name>` names it just
    # after the `decomposition` segment, with no `arms` container at all.
    segs = _segments(jpath)
    for i, s in enumerate(segs):
        if s == "arms" and i + 1 < len(segs):
            return segs[i + 1], ref
    for i, s in enumerate(segs):
        if (s == "decomposition" or s.startswith("decomposition_vs")) \
                and i + 1 < len(segs):
            return segs[i + 1], ref

    # `EXP_014`'s ladder identifies a leg by its width, not by a name.
    m = re.match(r"S4\.legs\[(\d+)\]", jpath)
    if m:
        leg = doc["S4"]["legs"][int(m.group(1))]
        return f"width d={leg.get('d_model')}", ref

    # `EXP_011` stores one arm at the top level.
    if segs[0] == "arm":
        return "compose (threshold x twocomp)", ref
    if segs[0] == "H5_horizon" and "noise_floor" in jpath:
        return "if_nest (bitwise copy)", ref
    return jpath, ref


def census() -> list[dict]:
    rows = []
    # `audit_*.json` is EXCLUDED, and this census's own output is one of them.
    # Without it the documented `--out` path writes back into the directory the
    # glob reads, so the second run ingests the first run's rows and reports 72
    # blocks, the third 180, compounding every time. A number its own generator
    # cannot reproduce is worse than an uncited one.
    for f in sorted(p for p in DATA.glob("*.json")
                    if not p.name.startswith("audit_")):
        try:
            doc = json.loads(f.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        for jpath, b in blocks(doc):
            beyond = next((b[k] for k in KEYS if k in b), None)
            if beyond is None:
                continue
            total = b.get("total_gain_bpc")
            arm, ref = identify(jpath, b, doc)
            kind = classify(jpath)
            secondary = any(s.startswith("decomposition_vs")
                        for s in _segments(jpath))
            family = any(t in arm.lower() for t in TWOCOMP_FAMILY)
            # A net-losing arm reading positive beyond the horizon is the
            # compression signature. Detected from the two gains directly, so a
            # block that stores no share is still classified.
            suspect = beyond > 0 and total is not None and total < 0

            # Join the horizon marker so the compression flag is checkable.
            # `width d=1481` is this script's label; the horizon file calls the
            # same run `scale_d1481`.
            hz = _horizons(f.name)
            key = arm.replace("width d=", "scale_d")
            arm_h = hz.get(key)
            anchor_h = next((hz[a] for a in ANCHOR_NAMES if a in hz), None)
            am, cm = _median(arm_h or []), _median(anchor_h or [])
            contradicted = bool(suspect and am is not None and cm is not None
                                and am > cm)

            rows.append({
                "source": f.name,
                "json_path": jpath,
                "arm": arm,
                "reference": ref,
                "block_class": kind,
                "secondary_reference": secondary,
                "twocomp_family": family,
                "cut_at_context": b.get("cut_at_context"),
                "total_gain_bpc": total,
                "zero_context_gain_bpc": b.get("zero_context_gain_bpc"),
                "within_reach_gain_bpc": b.get("within_reach_gain_bpc"),
                "beyond_horizon_gain_bpc": beyond,
                "beyond_horizon_share": b.get("beyond_horizon_share"),
                "abs_beyond": abs(beyond),
                "outside_exp024_band": abs(beyond) > EXP_024_BAND,
                "clears_exp024_h1_bar": abs(beyond) >= EXP_024_H1_BAR,
                "compression_suspect": suspect,
                "horizon_2sigma_per_seed": arm_h,
                "anchor_horizon_2sigma_per_seed": anchor_h,
                # The compression reading says the curve was squeezed, not
                # lengthened. A horizon marker that ALSO moved is evidence
                # against that reading -- not evidence for reach, since the
                # horizon has no sigma anywhere in this project (EXP_017 s10
                # item 8), but enough that the flag must not be read as settled.
                "compression_suspect_contradicted_by_horizon": contradicted,
            })
    rows.sort(key=lambda r: -r["abs_beyond"])

    # `if_binin` is stored BIT-IDENTICALLY in `EXP_020` and `EXP_022`. Counting
    # it twice inflates every tally by one, and the de-duplication this file
    # already declares for `secondary_reference` has to apply across sources
    # too. The first occurrence keeps the count; later ones point at it.
    seen: dict[tuple, str] = {}
    for r in rows:
        sig = (r["arm"], r["total_gain_bpc"], r["beyond_horizon_gain_bpc"])
        first = seen.get(sig)
        r["duplicate_of"] = first
        if first is None:
            seen[sig] = r["source"]
    return rows


def _fmt(v, spec=">+10.5f", width=10):
    """Sign goes BEFORE width in a format spec; ">10+.5f" is a ValueError."""
    return f"{'n/a':>{width}}" if v is None else f"{v:{spec}}"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="", help="write the census JSON here")
    args = ap.parse_args(argv)

    rows = census()
    scored = [r for r in rows if r["block_class"] == "arm_gain"
              and not r["secondary_reference"] and not r["duplicate_of"]]
    plain = [r for r in scored if not r["twocomp_family"]]
    outside = [r for r in plain if r["outside_exp024_band"]]
    clears = [r for r in plain if r["clears_exp024_h1_bar"]]
    clean = [r for r in clears if not r["compression_suspect"]]

    print(f"\n{len(rows)} decomposition blocks in "
          f"{len({r['source'] for r in rows})} committed artifacts.")
    print(f"{len(scored)} are an arm's gain against its own reference; "
          f"{len(plain)} of those are outside the two-compartment family.\n")

    hdr = (f"  {'arm':<28} {'total':>10} {'beyond':>10} {'share':>9}  "
           f"{'flags':<26} source")
    print(hdr)
    print("  " + "-" * (len(hdr) + 8))
    for r in rows:
        flags = []
        if r["block_class"] != "arm_gain":
            flags.append(r["block_class"])
        elif r["secondary_reference"]:
            flags.append("2nd reference")
        elif r["twocomp_family"]:
            flags.append("twocomp family")
        if r["block_class"] == "arm_gain" and r["clears_exp024_h1_bar"]:
            flags.append("CLEARS H1 BAR")
        elif r["block_class"] == "arm_gain" and r["outside_exp024_band"]:
            flags.append("outside band")
        if r["compression_suspect"]:
            flags.append("compression? BUT HORIZON MOVED"
                         if r["compression_suspect_contradicted_by_horizon"]
                         else "compression?")
        print(f"  {r['arm'][:28]:<28} {_fmt(r['total_gain_bpc'])} "
              f"{_fmt(r['beyond_horizon_gain_bpc'])} "
              f"{_fmt(r['beyond_horizon_share'], '>+9.4f', 9)}  "
              f"{', '.join(flags)[:26]:<26} {r['source']}")

    print(f"\nEXP_024 section 4: the component 'has never been observed outside "
          f"+/-{EXP_024_BAND} on any arm'.")
    print(f"  Scored: {len(plain)} primary arm-gain readings outside the "
          f"two-compartment family.")
    print(f"  {len(outside)} are outside the stated band.")
    print(f"  {len(clears)} clear EXP_024's own H1 bar of 2 sigma = "
          f"{EXP_024_H1_BAR}: "
          f"{', '.join(r['arm'] for r in clears) or 'none'}")
    print(f"  Of those, {len(clean)} are NOT compression-suspect: "
          f"{', '.join(r['arm'] for r in clean) or 'none'}")
    contra = [r for r in clears if r["compression_suspect_contradicted_by_horizon"]]
    if contra:
        print("  And the compression flag is CONTRADICTED by the horizon marker "
              "for:")
        for r in contra:
            print(f"    {r['arm']:<14} horizon {r['horizon_2sigma_per_seed']} "
                  f"vs anchor {r['anchor_horizon_2sigma_per_seed']}")
        print("  The horizon has no sigma anywhere in this project (EXP_017 s10 "
              "item 8).")
        print("  Two markers agreeing is two markers agreeing; it settles "
              "nothing on its own.")
    print(f"\n  Bitwise-copy floor (EXP_020 section 9.5, quoted by EXP_024 "
          f"section 2.4): {BITWISE_FLOOR:+.5f}")
    print("\n  A positive beyond-horizon reading on a net-LOSING arm may be curve")
    print("  compression rather than extended reach. This census FLAGS it, joins")
    print("  the horizon marker so the flag can be checked, and nets nothing out:")
    print("  which readings are artifacts is a question about the instrument that")
    print("  no committed artifact answers, and this script does not answer it.\n")

    if args.out:
        payload = {
            "what": ("every committed reading of the EXP_004 section 10.3 "
                     "beyond-horizon component"),
            "audit": "11_beyond_horizon_census",
            "generated_by": "scripts/audit/11_beyond_horizon_census.py",
            "gpu_used": False,
            "exp_024_stated_band": EXP_024_BAND,
            "exp_024_h1_bar_2sigma": EXP_024_H1_BAR,
            "bitwise_copy_floor": BITWISE_FLOOR,
            "n_blocks": len(rows),
            "n_scored_primary_arm_gains": len(scored),
            "n_scored_outside_twocomp_family": len(plain),
            "n_outside_stated_band": len(outside),
            "n_clearing_h1_bar": len(clears),
            "arms_clearing_h1_bar": [r["arm"] for r in clears],
            "arms_clearing_h1_bar_not_compression_suspect": [r["arm"] for r in clean],
            "arms_whose_compression_flag_the_horizon_contradicts": [
                r["arm"] for r in clears
                if r["compression_suspect_contradicted_by_horizon"]],
            "caveat": (
                "A positive beyond-horizon gain on an arm whose total gain is "
                "negative is the curve-compression signature. Such rows are "
                "flagged compression_suspect and are NOT evidence that any arm "
                "moved reach -- but where the arm's 2-sigma memory horizon also "
                "moved against the anchor, compression_suspect_contradicted_by"
                "_horizon is set and the flag must not be read as settled. The "
                "horizon itself has no sigma anywhere in this project (EXP_017 "
                "s10 item 8), so neither marker resolves the other. Excluded "
                "from scoring: "
                "remaining_gap_vs_gru blocks (a gap's composition, not an arm's "
                "gain), bars.* blocks (duplicates of an arm row), and "
                "decomposition_vs_* blocks (the same arm against a second "
                "reference)."
            ),
            "rows": rows,
        }
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"  wrote {out}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
