"""Reconstruct a pre-registration's PRE-RESULTS bytes and check them against the
SHA-256 its run manifest stamped before the first run.

    python scripts/exp/verify_prereg_hash.py                # every stamped experiment
    python scripts/exp/verify_prereg_hash.py --exp 022      # one

WHY THIS EXISTS
---------------
`CONTRIBUTING.md` §2 allows a stamped hash to stand in for a pre-run commit, but
only on a condition most of this project's logs have not met:

    "stamp the log's SHA-256 into the run manifest before the first run and
     re-check it after the last. `EXP_013` does this and records the recipe for
     reconstructing the pre-results bytes, so the guarantee survives appending
     results."

The recipe is the second half, and without it the stamp is not independently
checkable: once §9 is appended the live file no longer hashes to the stamp, and a
reader has no way to tell a correct stamp from a wrong one. Report §8's rev-14
note records that `EXP_020` and `EXP_021` supply the first half and not the
second. This file is the recipe, executable, so the guarantee survives appending
results for every experiment that follows the same layout.

THE RECIPE, AND IT IS THE WHOLE CONTRACT
----------------------------------------
A pre-registration is stamped in the state it had when §9 held nothing but its
placeholder. Two things change when an experiment closes, and only two:

  1. everything from the `## 9. Results` heading onward is replaced by the
     appended results;
  2. the one-line `**Status: OPEN.**` becomes `**Status: CLOSED ...**`.

So the pre-results bytes are recovered by truncating at the heading, restoring
the placeholder line, and restoring the status line. **Nothing else may differ.**
If a log has been edited above §9 in any other way this check fails, which is
exactly what it is for -- `CONTRIBUTING.md` §2's "never rewrite the hypothesis"
has had no mechanical enforcement until now.

WHAT IT FOUND WHEN IT WAS FIRST RUN, 2026-08-20
------------------------------------------------
`EXP_022` and `EXP_023` **VERIFY**: their stamped bytes reconstruct exactly, so
their hypotheses are provably the ones the runs were scored against.

`EXP_020` and `EXP_021` **DO NOT**, and a deliberate search says the recipe is
unrecoverable rather than merely unlisted. Every prefix of each live log was
hashed under both line-ending conventions, crossed with six plausible status
lines, five §9 layouts and two truncation points; nothing reproduces either
stamp. **That is not a finding that a hypothesis was edited** — the far more
likely explanation is a cosmetic change above §9 after stamping, which is
innocuous and unverifiable in exactly the same way. What it does mean is that
for those two experiments **the stamp is no longer independent evidence of
anything**, which is the guarantee `CONTRIBUTING.md` §2 was written to preserve.
Report §8's rev-14 note records it. `CONTRIBUTING.md` §3 — *"where a reasoning
trail is unrecoverable, say so in the artifact"* — is why it is stated here
rather than quietly omitted.

The older logs (`EXP_005`–`EXP_018`) predate this layout and read UNVERIFIED for
that reason; they are not evidence of anything either way and the tool says so.

It is deliberately NOT a pytest gate: it reads `experiments/runs/`-derived
manifests under `docs/reports/data/`, and it is a provenance audit rather than a
unit test. Run it when closing an experiment and when reviewing one.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
LOGS = _REPO / "experiments" / "logs"
DATA = _REPO / "docs" / "reports" / "data"

HEADING = "## 9. Results"

#: The placeholder line differs between logs -- `EXP_021` uses an em-dash where
#: `EXP_022` uses a semicolon -- and that is a formatting difference, NOT
#: evidence that a hypothesis was edited. Every known variant is tried, and the
#: one that matched is reported, so a reader can see which layout was assumed.
PLACEHOLDERS = (
    "*(appended after the run; nothing above this line is rewritten)*\n",
    "*(appended after the run — nothing above this line is rewritten)*\n",
    "*(appended after the run - nothing above this line is rewritten)*\n",
    "",
)
STATUS_OPEN = "**Status: OPEN.**\n"
STATUS_RE = re.compile(r"\*\*Status: CLOSED.*?\*\*\n", re.S)


def candidates(text: str):
    """Every plausible reconstruction of the pre-results bytes, with a label.

    Yields `(label, bytes)`. A log that verifies under ANY of these has an intact
    pre-registration; one that verifies under NONE is **unverified**, which is a
    weaker statement than "edited" and is reported as the weaker one. This tool
    can only ever confirm, never convict: the stamp fixes what the file WAS, and
    a failure to reconstruct it may mean the hypothesis moved or may mean the
    results were appended in a layout not listed above.
    """
    if HEADING not in text:
        return
    head = text[:text.index(HEADING)]
    heads = [("as-is", head)]
    restored, n = STATUS_RE.subn(STATUS_OPEN, head, count=1)
    if n:
        heads.append(("status restored to OPEN", restored))
    for hlabel, h in heads:
        for i, ph in enumerate(PLACEHOLDERS):
            body = h + HEADING + ("\n\n" + ph if ph else "\n")
            # BOTH LINE ENDINGS. `Path.write_text` translates "\n" to os.linesep
            # on Windows, so a log edited by a Python tool on this box acquires
            # CRLF while the same file stamped by a driver that read raw bytes
            # may have been LF. Hashing only one of them turns a newline
            # convention into an apparent hypothesis edit.
            yield f"{hlabel}, placeholder #{i}, LF", body
            yield f"{hlabel}, placeholder #{i}, CRLF", body.replace("\n", "\r\n")


def check(exp: str) -> dict:
    man_path = DATA / f"exp_{exp}_run_manifest.json"
    if not man_path.exists():
        return {"experiment": exp, "status": "NO MANIFEST"}
    man = json.loads(man_path.read_text(encoding="utf-8"))
    stamp = man.get("prereg_sha256_before")
    after = man.get("prereg_sha256_after")
    matches = [p for p in LOGS.glob(f"EXP_{exp}_*.md")]
    if len(matches) != 1:
        return {"experiment": exp, "status": f"AMBIGUOUS LOG ({len(matches)} matches)"}
    log = matches[0]
    text = log.read_text(encoding="utf-8")
    live = hashlib.sha256(text.encode("utf-8")).hexdigest()
    out = {
        "experiment": exp, "log": log.name,
        "manifest_stamp_before": stamp, "manifest_stamp_after": after,
        "before_equals_after": stamp == after,
        "live_hash": live,
        "live_equals_stamp": live == stamp,
    }
    if live == stamp:
        # An OPEN experiment, or one whose results live elsewhere.
        out["status"] = "VERIFIED"
        out["matched_layout"] = "live file, unmodified"
        return out
    for label, body in candidates(text):
        if hashlib.sha256(body.encode("utf-8")).hexdigest() == stamp:
            out["status"] = "VERIFIED"
            out["matched_layout"] = label
            return out
    # NOT "mismatch", and deliberately not. See `candidates`.
    out["status"] = "UNVERIFIED — no known layout reconstructs the stamp"
    out["matched_layout"] = None
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exp", default="", help="one experiment id, e.g. 022")
    args = ap.parse_args(argv)

    ids = ([args.exp] if args.exp else
           sorted(p.name[4:7] for p in DATA.glob("exp_*_run_manifest.json")))
    rows = [check(e) for e in ids]
    ok = sum(r["status"] == "VERIFIED" for r in rows)
    stamped = sum(r["status"] != "NO MANIFEST" for r in rows)
    print()
    for r in rows:
        print(f"  EXP_{r['experiment']}  {r['status']}")
        if r.get("matched_layout"):
            print(f"      stamp {r['manifest_stamp_before'][:32]}…  "
                  f"before==after {r['before_equals_after']}")
            print(f"      via   {r['matched_layout']}")
    print(f"\n{ok}/{stamped} stamped pre-registrations verified.")
    print("UNVERIFIED means no layout this tool knows reconstructs the stamp. It is\n"
          "NOT a finding that a hypothesis was edited -- the placeholder line and the\n"
          "status line vary between logs, and a variant not listed in PLACEHOLDERS\n"
          "reads the same way as a real edit. Add the variant, or check by hand.\n")
    # Exit 0: this is a provenance report, not a gate. A gate that cannot tell a
    # formatting variant from an edited hypothesis would be a gate nobody reads.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
