"""Reconstruct a pre-registration's PRE-RESULTS bytes and check them against the
SHA-256 its run manifest stamped before the first run.

    python scripts/exp/verify_prereg_hash.py                # every stamped experiment
    python scripts/exp/verify_prereg_hash.py --exp 022      # one
    python scripts/exp/verify_prereg_hash.py --json         # machine-readable

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
reader has no way to tell a correct stamp from a wrong one. This file is the
recipe, executable.

TWO ROUTES TO THE PRE-RESULTS BYTES, AND THE FIRST IS STRONGER
--------------------------------------------------------------
1. **BY COMMIT.** These logs are committed. If any commit in a log's history
   holds bytes that hash to the stamp, the pre-registration is verified against
   an object in the repository rather than against a guess about layout. History
   is walked **oldest first**, so the commit named is the earliest that held the
   stamped bytes. This route needs no PLACEHOLDERS table and names the commit so
   a reader can `git show` it. **It is tried first, and where it fires the recipe
   route is not consulted.**

   **What it does NOT establish, and why there is a third check.** A blob that
   matches the stamp proves the STAMP is honest. It says nothing about the file
   on disk today, which is the guarantee §2 actually wants. So every
   commit-verified row also reports `live_file_agrees_above_results`: the blob
   and the live file, both truncated at the heading, compared line by line with
   the status BLOCK removed from each. The status is the one change above the
   heading that closing an experiment may make, and it is not one line --
   `EXP_022`'s wraps over six and summarises its verdicts, so a first-line-only
   filter reported five continuation lines as an edited hypothesis. All six
   commit-verified logs read `True`.

2. **BY RECIPE.** Where history does not hold the pre-results bytes -- a log
   whose only commit already contained its results -- the bytes are reconstructed
   from the live file. A pre-registration is stamped in the state it had when §9
   held nothing but its placeholder, and exactly two things change when an
   experiment closes:

     a. everything from the `## 9. Results` heading onward is replaced;
     b. the one-line `**Status: OPEN.**` becomes `**Status: CLOSED ...**`.

   So the bytes are recovered by truncating at the heading and restoring the
   placeholder and status lines. **Nothing else may differ.** A log edited above
   §9 in any other way fails, which is exactly what it is for -- §2's "never
   rewrite the hypothesis" had no mechanical enforcement before this file.

FOUR STATES, NOT TWO, AND THE DENOMINATOR IS PART OF THE FINDING
-----------------------------------------------------------------
An earlier revision of this tool reported "2/12 verified". Both halves were
wrong, and the way they were wrong is the reason this docstring is long:

* **Three of those twelve carry no stamp at all.** `exp_005`, `exp_009` and
  `exp_011` have no `prereg_*` key in their manifests. Counting them in the
  denominator of "stamped pre-registrations" reports an absent stamp as a failed
  one. They are now `NO STAMP`, and they are excluded. **A fourth manifest,
  `exp_007_008`, also carries no stamp but never reached that denominator** --
  the id defect below had already excluded it. The two defects are interlocked,
  and the twelve is `3 unstamped + 9 stamped`, of which 2 verified.
* **Five more stamp under a different key.** `EXP_013`, `014`, `015`, `017` and
  `018` write `prereg_sha256_before_first_run`, not `prereg_sha256_before`. The
  tool read only the latter, got `None`, and compared every candidate against
  `None`. Both spellings are now accepted.
* **The id parse split the wrong filename.** `p.name[4:7]` turns
  `exp_007_008_run_manifest.json` into `007`, and the tool then looked for a
  manifest named `exp_007_run_manifest.json`, which does not exist, and reported
  `NO MANIFEST` for a manifest it had just enumerated. Ids are now taken whole.
* **`PLACEHOLDERS` omitted the variant used by the one log `CONTRIBUTING.md` §2
  names.** `EXP_013` records its own reconstruction stub at `:424-431` and the
  table did not carry it, so the exemplar read UNVERIFIED. That is the
  single-variant-lookup defect again, one entry away from the key defect above.

WHAT IT FINDS AT `593baaa`, WITH BOTH ROUTES AND THE KEYS READ CORRECTLY
------------------------------------------------------------------------
  VERIFIED BY COMMIT   EXP_014, EXP_015, EXP_017, EXP_018, EXP_022, EXP_023
  VERIFIED BY RECIPE   EXP_013
  UNVERIFIED           EXP_020, EXP_021
  NO STAMP             EXP_005, EXP_007_008, EXP_009, EXP_011

**7/9, against the 2/12 the previous revision printed.** Five experiments it
reported as UNVERIFIED are provably unedited: four against a named commit, and
`EXP_013` against the recipe its own log records. `EXP_022` and `EXP_023` verify
by **both** routes, which is a check on the recipe itself -- the reconstruction
agrees with the object database wherever both exist.

The two that remain unverified share one cause and it is not suspicion: each has
exactly **one** commit touching its log, and that commit already contained its
results, so no pre-results bytes exist in history to compare against. For
`EXP_020` and `EXP_021` an exhaustive prefix search under both line endings,
crossed with every known status and placeholder variant, also reproduces
neither stamp. **That is not a finding that a hypothesis was edited** -- a
cosmetic change above §9 after stamping produces exactly this and is far more
likely. What it means is that for those two the stamp is no longer independent
evidence of anything, which is the guarantee §2 exists to preserve.
`CONTRIBUTING.md` §3 -- *"where a reasoning trail is unrecoverable, say so in the
artifact"* -- is why that is stated here rather than quietly omitted.

WHAT THIS TOOL REFUSES TO DO
----------------------------
**It reports UNVERIFIED, never MISMATCH.** The placeholder line varies between
logs -- `EXP_021` uses an em-dash where `EXP_022` uses a semicolon -- and an
unlisted variant reads identically to a real edit. This tool can confirm; it
cannot convict.

**It hashes both line endings.** `Path.write_text` translates "\n" to os.linesep
on Windows, so a log edited by a Python tool on this box acquires CRLF while a
driver that read raw bytes stamped LF. Hashing only one turns a newline
convention into an apparent hypothesis edit. `EXP_015` verifies only under the
CRLF reading of its committed blob, so this is load-bearing and not theoretical.

**It refuses the headline when git is unavailable.** `git_history` returns `None`
for "could not consult git" and `[]` for "consulted, no history". Collapsing
those printed `EXP_014  0 commit(s) touching the log` beneath a heading that
explains UNVERIFIED as "one commit, and it already contained results" -- an
affirmatively false sentence about a log with four commits, produced by a tool
whose whole purpose is provenance. The two are now distinguished and a
recipe-only run says so above the ratio.

It is deliberately NOT a pytest gate: it reads `experiments/runs/`-derived
manifests under `docs/reports/data/`, and it is a provenance audit rather than a
unit test. Run it when closing an experiment and when reviewing one.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
LOGS = _REPO / "experiments" / "logs"
DATA = _REPO / "docs" / "reports" / "data"

HEADING = "## 9. Results"

#: Both spellings appear in committed manifests. `EXP_013`-`EXP_018` write the
#: `_first_run`/`_last_run` pair; `EXP_020` onward write the short pair. Reading
#: only one of them reports five intact pre-registrations as unverified.
STAMP_KEYS_BEFORE = ("prereg_sha256_before", "prereg_sha256_before_first_run")
STAMP_KEYS_AFTER = ("prereg_sha256_after", "prereg_sha256_after_last_run")

#: The placeholder line differs between logs -- `EXP_021` uses an em-dash where
#: `EXP_022` uses a semicolon -- and that is a formatting difference, NOT
#: evidence that a hypothesis was edited. Every known variant is tried, and the
#: one that matched is reported, so a reader can see which layout was assumed.
PLACEHOLDERS = (
    "*(appended after the run; nothing above this line is rewritten)*\n",
    "*(appended after the run — nothing above this line is rewritten)*\n",
    "*(appended after the run - nothing above this line is rewritten)*\n",
    # `EXP_013` is the log `CONTRIBUTING.md` §2 names as the one that "records
    # the recipe", and it records this stub verbatim at its own :424-431. A
    # single-variant table that omits it reports the exemplar as UNVERIFIED --
    # which is the same defect as reading a single manifest key, one paragraph
    # apart, and it is why this entry is quoted from the log rather than guessed.
    ("*(Empty at pre-registration. Appended after the runs; predictions are "
     "resolved as\nwritten, including the ones that fail.)*\n"),
    "",
)
#: The status strings a pre-registration may have carried WHEN IT WAS STAMPED.
#: The recipe route reconstructs the pre-results bytes from the live file, so a
#: closed log's status has to be put back to whatever it was before closure --
#: and this project uses more than one shape. `EXP_013` stamps
#: `**Status:** PRE-REGISTERED ...` (bold closed after the label) while
#: `EXP_015`-`EXP_017` stamp `**Status: PRE-REGISTERED ...**`.
#:
#: A SINGLE ENTRY HERE IS THE DEFECT THIS FILE KEEPS REDISCOVERING. It has now
#: bitten three times: the `prereg_sha256_before` key, the `PLACEHOLDERS` table,
#: and the status-block parser. Restoring only `**Status: OPEN.**` meant that
#: correcting a stale header -- which eight closed logs need -- silently cost
#: `EXP_013` its only verification route. These are SHAPES, not per-log entries,
#: which is what keeps the list short enough to stay right.
#:
#: Adding candidates cannot manufacture a false positive: every one must still
#: hash to the stamp under SHA-256.
STATUS_RESTORES = (
    "**Status: OPEN.**\n",
    "**Status:** OPEN.\n",
    "**Status: PRE-REGISTERED \u2014 no results yet.**\n",
    "**Status:** PRE-REGISTERED \u2014 no results yet.\n",
)
STATUS_RE = re.compile(r"\*\*Status:.*?\*\*\n", re.S)


def _first(mapping: dict, keys) -> str | None:
    """The first key present and non-empty, or None. Order is precedence."""
    for k in keys:
        if mapping.get(k):
            return mapping[k]
    return None


def _variants(raw: bytes):
    """`(label, bytes)` for the byte-identical, all-CRLF and all-LF readings.

    Deduplicated, so a file already in one convention yields one entry and the
    label a match reports is the convention that actually applied to it.
    """
    seen: dict[bytes, str] = {}
    for label, b in (("as-committed", raw),
                     ("CRLF", raw.replace(b"\r\n", b"\n").replace(b"\n", b"\r\n")),
                     ("LF", raw.replace(b"\r\n", b"\n"))):
        seen.setdefault(b, label)
    return [(label, b) for b, label in seen.items()]


def git_history(relpath: str) -> list[str] | None:
    """Commits touching `relpath`, newest first.

    **`None` means git could not be consulted; `[]` means it was and the file has
    no history.** Collapsing those two into `[]` made the summary print
    "EXP_014  0 commit(s) touching the log" under a heading that explains an
    UNVERIFIED result as "one commit, and it already contained results" -- an
    affirmatively false sentence about a log with four commits. When git is
    unavailable the headline is refused rather than restated without it.
    """
    try:
        r = subprocess.run(["git", "log", "--follow", "--format=%H", "--", relpath],
                           cwd=_REPO, capture_output=True, timeout=60)
    except (OSError, subprocess.SubprocessError):
        return None
    if r.returncode != 0:
        return None
    return r.stdout.decode("utf-8", "replace").split()


def git_blob(sha: str, relpath: str) -> bytes:
    """The file's bytes at `sha`, or empty if it did not exist there."""
    try:
        r = subprocess.run(["git", "show", f"{sha}:{relpath}"],
                           cwd=_REPO, capture_output=True, timeout=60)
    except (OSError, subprocess.SubprocessError):
        return b""
    return r.stdout if r.returncode == 0 else b""


def by_commit(relpath: str, stamp: str) -> tuple[str, str] | None:
    """`(short_sha, line_ending_label)` of the first commit whose blob hashes to
    `stamp`, or None.

    This is the strong route: it compares the stamp against an object in the
    repository instead of against a guess about how results were appended.

    History is walked OLDEST FIRST. `git log` emits newest first, and returning
    the newest match would name the last commit that happened to hold the
    stamped bytes rather than the first -- for a still-OPEN experiment whose live
    file equals its stamp, that is HEAD, which is provenance-free. The earliest
    match is the claim worth making: these bytes existed at least this early.
    """
    for sha in reversed(git_history(relpath) or []):
        raw = git_blob(sha, relpath)
        if not raw:
            continue
        for label, b in _variants(raw):
            if hashlib.sha256(b).hexdigest() == stamp:
                return sha[:8], label
    return None


def lines_above_results(s: str) -> list[str]:
    """Lines above the heading, with the whole status BLOCK removed.

    The status is the one change above the heading that closing an
    experiment is allowed to make, and it is not one line: `EXP_022`'s
    `**Status: CLOSED ...**` wraps over six, summarising the verdicts. A
    first-line-only filter reported those five continuation lines as an
    edited hypothesis -- precisely the false alarm this tool exists not to
    raise.

    THE BLOCK IS A PARAGRAPH, NOT A BOLD SPAN, AND THAT CORRECTION MATTERS
    ---------------------------------------------------------------------
    This skipped from `**Status:` to the line closing its **bold span**,
    which silently mis-parsed the second status shape the project's own logs
    use. `EXP_013`, `EXP_014` and `EXP_018` write

        **Status:** PRE-REGISTERED -- no results yet.

    -- the bold closes after the label, so the line does not end in `**` and
    the scan ran on, swallowing the pre-registration blurb, the first real
    heading, and every line down to the next bold-terminated one. On a
    probe of that shape **five of seven lines vanished from the
    comparison**, `## 1. The question` among them.

    It read `True` anyway, because the blob and the live file mis-parsed
    *identically* -- so the guarantee held **by coincidence, not by
    correctness**, and any edit to those three status lines would have
    silently changed what the check compares. That is the same
    single-variant defect report §8's rev-15 note found in `PLACEHOLDERS`,
    in the same file, arriving by a second route.

    A status is a **paragraph**: it ends at the blank line, whatever its
    bold shape. The blank line itself is kept, so a one-line status and a
    six-line one both collapse to the same thing on both sides.
    """
    s = s.replace("\r\n", "\n")
    head = s[:s.index(HEADING)] if HEADING in s else s
    out, skipping = [], False
    for ln in head.split("\n"):
        t = ln.strip()
        if skipping:
            if t:
                continue          # still inside the status paragraph
            skipping = False      # the blank line ends it, and is kept
        elif t.startswith("**Status:"):
            skipping = True
            continue
        out.append(ln)
    return out


def live_matches_blob(sha: str, relpath: str, text: str) -> bool | None:
    """Does the LIVE file still agree with the verified blob above `## 9. Results`?

    The commit route returns before the live file is ever read, so on its own it
    is strong evidence that the STAMP is honest and weak evidence that the
    working file is unedited -- the guarantee this tool exists to give. This is
    the cheap third check: truncate both at the heading and compare. `None` when
    the blob cannot be re-read.
    """
    raw = git_blob(sha, relpath)
    if not raw:
        return None

    return (lines_above_results(raw.decode("utf-8", "replace"))
            == lines_above_results(text))


def nl_of(s: str) -> str:
    """The newline convention `s` uses, so a restored status keeps it."""
    return "\r\n" if "\r\n" in s else "\n"


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
    for st in STATUS_RESTORES:
        want = st.replace(chr(10), nl_of(head))
        restored, n = STATUS_RE.subn(want, head, count=1)
        if n and restored != head:
            heads.append((f"status restored to {st.strip()!r}", restored))
    for hlabel, h in heads:
        for i, ph in enumerate(PLACEHOLDERS):
            body = h + HEADING + ("\n\n" + ph if ph else "\n")
            yield f"{hlabel}, placeholder #{i}, LF", body
            yield f"{hlabel}, placeholder #{i}, CRLF", body.replace("\n", "\r\n")


def check(exp: str) -> dict:
    """`exp` is the manifest's own id -- "022", or "007_008" -- taken whole."""
    man_path = DATA / f"exp_{exp}_run_manifest.json"
    if not man_path.exists():
        return {"experiment": exp, "status": "NO MANIFEST"}
    man = json.loads(man_path.read_text(encoding="utf-8"))
    stamp = _first(man, STAMP_KEYS_BEFORE)
    after = _first(man, STAMP_KEYS_AFTER)

    # The log id is the manifest id's first segment: `exp_007_008` covers
    # `EXP_007` and `EXP_008`, and its log glob must not carry the second.
    matches = sorted(LOGS.glob(f"EXP_{exp.split('_')[0]}_*.md"))
    if len(matches) != 1:
        return {"experiment": exp, "status": f"AMBIGUOUS LOG ({len(matches)} matches)"}
    log = matches[0]
    out = {
        "experiment": exp, "log": log.name,
        "manifest_stamp_before": stamp, "manifest_stamp_after": after,
        "before_equals_after": bool(stamp) and stamp == after,
    }
    if not stamp:
        # An absent stamp is not a failed one. Reporting it as UNVERIFIED, and
        # counting it in the denominator, is how "2/12" was arrived at.
        out["status"] = "NO STAMP"
        out["matched_layout"] = None
        return out

    relpath = f"experiments/logs/{log.name}"
    history = git_history(relpath)
    out["git_available"] = history is not None
    text = log.read_text(encoding="utf-8")
    hit = by_commit(relpath, stamp)
    if hit:
        sha, ending = hit
        out["status"] = "VERIFIED BY COMMIT"
        out["commit"] = sha
        out["matched_layout"] = f"blob at {sha} ({ending})"
        out["live_file_agrees_above_results"] = live_matches_blob(sha, relpath, text)
        return out

    if hashlib.sha256(text.encode("utf-8")).hexdigest() == stamp:
        # An OPEN experiment, or one whose results live elsewhere.
        out["status"] = "VERIFIED BY RECIPE"
        out["matched_layout"] = "live file, unmodified"
        return out
    for label, body in candidates(text):
        if hashlib.sha256(body.encode("utf-8")).hexdigest() == stamp:
            out["status"] = "VERIFIED BY RECIPE"
            out["matched_layout"] = label
            return out
    # NOT "mismatch", and deliberately not. See `candidates`.
    out["status"] = "UNVERIFIED"
    out["matched_layout"] = None
    out["history_commits"] = None if history is None else len(history)
    return out


def manifest_ids() -> list[str]:
    """Ids taken whole from the filename, so `exp_007_008` stays `007_008`."""
    return sorted(p.name[len("exp_"):-len("_run_manifest.json")]
                  for p in DATA.glob("exp_*_run_manifest.json"))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exp", default="", help="one experiment id, e.g. 022")
    ap.add_argument("--json", action="store_true", help="emit the rows as JSON")
    args = ap.parse_args(argv)

    rows = [check(e) for e in ([args.exp] if args.exp else manifest_ids())]
    if args.json:
        print(json.dumps(rows, indent=2))
        return 0

    n_commit = sum(r["status"] == "VERIFIED BY COMMIT" for r in rows)
    n_recipe = sum(r["status"] == "VERIFIED BY RECIPE" for r in rows)
    unverified = [r for r in rows if r["status"] == "UNVERIFIED"]
    nostamp = [r for r in rows if r["status"] == "NO STAMP"]
    stamped = n_commit + n_recipe + len(unverified)

    if any(r.get("git_available") is False for r in rows):
        print()
        print("  WARNING: git could not be consulted, so the commit route did "
              "not run.")
        print("  The ratio below is the RECIPE ROUTE ONLY and is not this "
              "tool's verdict.")
    print()
    for r in rows:
        print(f"  EXP_{r['experiment']:<8} {r['status']}")
        if r.get("matched_layout"):
            print(f"      stamp {r['manifest_stamp_before'][:32]}...  "
                  f"before==after {r['before_equals_after']}")
            print(f"      via   {r['matched_layout']}")
            live = r.get("live_file_agrees_above_results")
            if live is not None:
                print(f"      live  file agrees with that blob above "
                      f"'{HEADING}': {live}")

    print(f"\n{n_commit + n_recipe}/{stamped} stamped pre-registrations verified "
          f"({n_commit} by commit, {n_recipe} by recipe).")
    if nostamp:
        print(f"{len(nostamp)} manifest(s) carry NO STAMP and are excluded from that "
              f"denominator: {', '.join('EXP_' + r['experiment'] for r in nostamp)}.")
    if unverified:
        print("\nUNVERIFIED means neither route reconstructs the stamp. It is NOT a")
        print("finding that a hypothesis was edited -- a cosmetic edit above section 9")
        print("after stamping reads identically. Where the log has one commit and it")
        print("already contained results, no pre-results bytes exist to compare:")
        for r in unverified:
            n = r.get("history_commits")
            print(f"  EXP_{r['experiment']:<8} "
                  + ("git unavailable -- history not consulted" if n is None
                     else f"{n} commit(s) touching the log"))
    print()
    # Exit 0: this is a provenance report, not a gate. A gate that cannot tell a
    # formatting variant from an edited hypothesis would be a gate nobody reads.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
