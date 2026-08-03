"""Check the Phase-3 candidate spec against the report it was reconstructed from.

`docs/reports/data/phase3_candidates.json` enumerates the 18 candidates §6 of
`03_phase3_candidates.md` describes. A hand-written JSON next to a hand-written
report is just the same prose twice, and would drift the first time either is
edited -- so this script parses the report's own tables and asserts the two
agree, cell by cell.

WHY THE SPEC EXISTS AT ALL
--------------------------
Every other Phase-3 claim traces to a script, a machine-readable output and an
experiment log. The candidate set did not: it was generated, adversarially
verified and completeness-audited in a session whose transcript was never
exported, and survived only as report prose. That is the one place in the phase
where a reader who wanted to check a claim had nothing to check it against.

WHAT THIS SCRIPT CAN AND CANNOT ESTABLISH
-----------------------------------------
It establishes that the committed spec faithfully records what the report says,
and that the report's arithmetic closes. It does **not** establish that the
original generation was sound -- the reasoning trail for that is gone, the spec
says so in its `provenance` block, and no script can recover it. Claims the
record cannot support are collected into `unverifiable_claims` and printed, so
that they stay visible instead of being laundered by the existence of a JSON file.

It also cross-links the spec to `audit_09_gradient_reachability.json`, in both
directions: every candidate that claims a screen result must have one, and every
non-control screen must belong to a candidate. That is what stops the two
artifacts from quietly describing different candidate sets.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]

REPORT = _REPO / "docs/reports/03_phase3_candidates.md"
SPEC = _REPO / "docs/reports/data/phase3_candidates.json"
SCREEN = _REPO / "docs/reports/data/audit_09_gradient_reachability.json"
OUT = _REPO / "docs/reports/data/audit_10_candidate_spec.json"

#: Screens that exist for a reason other than a ranked candidate, and so are not
#: expected to map back to one.
SCREEN_EXEMPT = {"ctl_live", "ctl_dead", "ctl_faint", "thr"}


def norm(s: str) -> str:
    """Compare report text and spec text on content, not on markdown emphasis.

    Bold, italic and code spans are typography; a spec that had to reproduce them
    byte-for-byte would fail on a purely cosmetic edit to the report and would
    train the reader to ignore this script's output.
    """
    s = s.replace("*", "").replace("`", "")
    return re.sub(r"\s+", " ", s).strip()


def section(text: str, start: str, end: str | None) -> str:
    i = text.index(start)
    j = text.index(end, i) if end else len(text)
    return text[i:j]


def parse_ranked_table(block: str) -> list[dict]:
    """The §6.3 shortlist, one dict per row, keyed by the header's own columns."""
    rows = []
    for line in block.splitlines():
        line = line.strip()
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) != 7 or set(cells[0]) <= set("-: "):
            continue
        if not re.fullmatch(r"\**\d+\**", cells[0]):
            continue
        rows.append({
            "rank": int(norm(cells[0])),
            "name": cells[1],
            "attacks": cells[2],
            "params_per_neuron": cells[3],
            "i5": cells[4],
            "gpu_hours_3_seeds": cells[5],
            "note": cells[6],
        })
    return rows


def parse_referred(block: str) -> list[str]:
    """The bolded lead of each 'Referred to Elliot' bullet."""
    out = []
    for line in block.splitlines():
        m = re.match(r"\s*\*\s+\*\*(.+?)\*\*", line)
        if m:
            out.append(norm(m.group(1)))
    return out


def main() -> int:
    text = REPORT.read_text(encoding="utf-8")
    spec = json.loads(SPEC.read_text(encoding="utf-8"))
    failures: list[str] = []
    checks: list[dict] = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        checks.append({"check": name, "ok": bool(ok), "detail": detail})
        if not ok:
            failures.append(f"{name}: {detail}")

    by_id = {c["id"]: c for c in spec["candidates"]}
    ranked = [c for c in spec["candidates"] if c["status"] == "ranked"]
    refuted = [c for c in spec["candidates"] if c["status"] == "refuted"]
    referred = [c for c in spec["candidates"] if c["status"] == "referred_to_elliot"]

    # -- 1. the ranked shortlist, cell by cell ------------------------------
    shortlist = section(text, "### 6.3 The ranked shortlist", "### 6.4")
    report_rows = parse_ranked_table(shortlist)
    check("ranked row count", len(report_rows) == len(ranked) == 14,
          f"report {len(report_rows)}, spec {len(ranked)}, expected 14")

    ranked_by_rank = {c["rank"]: c for c in ranked}
    for row in report_rows:
        cand = ranked_by_rank.get(row["rank"])
        if cand is None:
            check(f"rank {row['rank']} present in spec", False, "missing")
            continue
        for field in ("name", "attacks", "params_per_neuron", "i5", "note"):
            got, want = norm(str(cand[field])), norm(row[field])
            check(f"rank {row['rank']} {field}", got == want,
                  f"spec {got!r} != report {want!r}")
        try:
            want_h = float(norm(row["gpu_hours_3_seeds"]))
        except ValueError:
            want_h = None
        check(f"rank {row['rank']} gpu_hours",
              want_h is not None and abs(cand["gpu_hours_3_seeds"] - want_h) < 1e-9,
              f"spec {cand['gpu_hours_3_seeds']} != report {row['gpu_hours_3_seeds']}")

    # -- 2. the budget line has to be the sum of the rows it totals ----------
    total = round(sum(c["gpu_hours_3_seeds"] for c in ranked), 6)
    check("spec budget equals the sum of the ranked rows",
          abs(spec["budget"]["ranked_gpu_hours_total"] - total) < 1e-9,
          f"spec says {spec['budget']['ranked_gpu_hours_total']}, rows sum to {total}")
    m = re.search(r"Total:\s*\*\*~?([\d.]+)\s*GPU-hours\*\*", shortlist)
    check("report total is present", m is not None, "no 'Total: ~N GPU-hours' line")
    if m:
        check("report total equals the sum of its own rows",
              abs(float(m.group(1)) - total) < 0.05,
              f"report says ~{m.group(1)}, its rows sum to {total}")

    # -- 3. refuted and referred -------------------------------------------
    check("exactly one refuted candidate", len(refuted) == 1, f"{len(refuted)}")
    if refuted:
        blk = section(text, "**Refuted and dropped:**", "**Referred to Elliot")
        check("refuted candidate named in the report",
              norm(refuted[0]["name"]).lower() in norm(blk).lower(),
              f"{refuted[0]['name']!r} not found in the refutation paragraph")

    ref_blk = section(text, "**Referred to Elliot, not decided here.**",
                      "### 6.4")
    leads = parse_referred(ref_blk)
    check("three referred candidates", len(referred) == len(leads) == 3,
          f"report {len(leads)}, spec {len(referred)}")
    for lead in leads:
        check(f"referred {lead!r} in spec",
              any(norm(c["name"]).startswith(lead) for c in referred),
              f"no spec entry whose name starts with {lead!r}")

    # -- 4. counts and merges ----------------------------------------------
    counts = spec["counts"]
    check("counts.ranked", counts["ranked"] == len(ranked), str(counts["ranked"]))
    check("counts.refuted", counts["refuted"] == len(refuted), str(counts["refuted"]))
    check("counts.referred_to_elliot", counts["referred_to_elliot"] == len(referred),
          str(counts["referred_to_elliot"]))
    check("counts arithmetic closes",
          counts["ranked"] + counts["refuted"] + counts["referred_to_elliot"]
          == counts["candidates_in_final_set"] == len(spec["candidates"]) == 18,
          f"{counts}")
    check("pre-merge proposal count",
          counts["proposals_before_merging"]
          == counts["candidates_in_final_set"] + counts["merged_pairs"],
          f"{counts['proposals_before_merging']} != 18 + {counts['merged_pairs']}")

    merge_blk = section(text, "### 6.1 Two pairs that are the same lever", "### 6.2")
    n_bullets = len(re.findall(r"^\s*\*\s+\*\*", merge_blk, re.M))
    check("merge count matches the report", n_bullets == counts["merged_pairs"] == 2,
          f"report {n_bullets}, spec {counts['merged_pairs']}")
    for mg in spec["merges"]:
        check(f"merge target {mg['into']} exists", mg["into"] in by_id, mg["into"])
        check(f"merge target {mg['into']} is flagged in its candidate entry",
              by_id.get(mg["into"], {}).get("merged_from_pair") is True,
              "merged_from_pair is not true")

    # -- 5. the §9 exit-criteria line has to agree with the spec ------------
    # The summary is the bolded run containing "generated". Anchoring on it
    # matters: the same bullet also says "10-15 ranked hypotheses", and a loose
    # search for "N ranked" finds the 15 in the target range instead of the 14 in
    # the tally.
    ex = section(text, "## 9. Exit criteria", "### 9.1")
    tally = next((b for b in re.findall(r"\*\*(.+?)\*\*", ex, re.S)
                  if "generated" in b), None)
    check("§9 carries a candidate tally", tally is not None,
          "no bolded run mentioning 'generated' in §9")
    if tally:
        tally = norm(tally)
        for label, want in (("generated", counts["candidates_in_final_set"]),
                            ("refuted", counts["refuted"]),
                            ("pairs merged", counts["merged_pairs"]),
                            ("ranked", counts["ranked"])):
            m = re.search(rf"(\d+)\s+{re.escape(label)}", tally)
            check(f"§9 tally says '{want} {label}'",
                  m is not None and int(m.group(1)) == want,
                  f"found {m.group(1) if m else None!r}, spec says {want}")

    # -- 6. cross-link to the reachability screen, in both directions -------
    screen_ok = SCREEN.exists()
    check("reachability screen artifact present", screen_ok, str(SCREEN))
    if screen_ok:
        screened = {r["key"]: r for r in
                    json.loads(SCREEN.read_text(encoding="utf-8"))["results"]}
        claimed: set[str] = set()
        for c in spec["candidates"]:
            keys = c["reachability"]["screen_keys"]
            claimed |= set(keys)
            for k in keys:
                check(f"{c['id']} screen key {k!r} exists", k in screened,
                      "not in audit_09")
            if c["reachability"]["applies"] and not keys:
                # Legitimate only when the spec says why. A silent gap here is
                # the failure mode: a candidate that looks screened because it
                # sits in a file full of screened ones.
                check(f"{c['id']} explains why it is unscreened",
                      "NOT SCREENED" in c["reachability"]["outcome"],
                      "applies=true, no keys, and no stated reason")
        orphans = sorted(set(screened) - claimed - SCREEN_EXEMPT)
        check("no screened arm is missing from the spec", not orphans,
              f"screened but unclaimed: {orphans}")

    # -- 7. what cannot be checked, kept visible ----------------------------
    attributable = [c["id"] for c in spec["candidates"]
                    if c.get("added_by_completeness_audit") is True]
    unverifiable = [
        {"claim": "eight of the eighteen candidates were added by a completeness "
                  "audit",
         "status": f"only {len(attributable)} are attributable from the committed "
                   f"record ({', '.join(attributable)}), via §5",
         "recoverable": False,
         "why": "the generating session's transcript was never exported"},
        {"claim": "the two merged pairs were the same lever under different names",
         "status": "the merge algebra is given in §6.1 and is checkable on its own "
                   "terms; the original proposal wordings are not in the repository, "
                   "so the merge cannot be checked against what was proposed",
         "recoverable": False,
         "why": "same lost transcript"},
        {"claim": "the candidate set spans the six areas the Phase-1 plan names",
         "status": "the spec's `family_reconstructed` is this reconstruction's "
                   "inference from candidate text, not a recovered value; four "
                   "candidates are left null rather than guessed",
         "recoverable": False,
         "why": "same lost transcript"},
    ]

    payload = {
        "audit": "10_candidate_spec",
        "spec": str(SPEC.relative_to(_REPO)).replace("\\", "/"),
        "report": str(REPORT.relative_to(_REPO)).replace("\\", "/"),
        "checks_run": len(checks),
        "checks_failed": len(failures),
        "passed": not failures,
        "failures": failures,
        "checks": checks,
        "unverifiable_claims": unverifiable,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("=" * 78)
    print("PHASE-3 CANDIDATE SPEC vs THE REPORT IT RECORDS")
    print("=" * 78)
    print(f"  candidates   {len(spec['candidates'])}  "
          f"({len(ranked)} ranked, {len(refuted)} refuted, {len(referred)} referred)")
    print(f"  ranked GPU-h {total}")
    print(f"  checks       {len(checks) - len(failures)}/{len(checks)} passed")
    for f in failures:
        print(f"    FAIL  {f}")
    print("\n  claims the committed record cannot support (kept visible on purpose):")
    for u in unverifiable:
        print(f"    - {u['claim']}")
        print(f"      -> {u['status']}")
    print(f"\nWROTE {OUT.relative_to(_REPO)}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
