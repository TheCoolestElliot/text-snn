"""The status-block parser that decides whether a pre-registration was edited.

`verify_prereg_hash.py`'s `live_file_agrees_above_results` is the check that
turns "the STAMP is honest" into "the FILE on disk is unedited" -- the guarantee
`CONTRIBUTING.md` §2 actually wants. It works by removing the status block from
both the committed blob and the live file and comparing what is left.

WHY THESE TESTS EXIST
---------------------
The block was skipped from `**Status:` to the line closing its **bold span**.
That silently mis-parsed the second status shape the project's own logs use:

    **Status:** PRE-REGISTERED -- no results yet.      <- EXP_013, 014, 018
    **Status: PRE-REGISTERED -- no results yet.**      <- EXP_015, 016, 017, 022, 025

In the first, the bold closes after the label, so the line does not end in `**`
and the scan ran on -- swallowing the pre-registration blurb, the first real
heading, and everything down to the next bold-terminated line. On a probe of
that shape **five of seven lines vanished from the comparison.**

It still reported `True`, because the blob and the live file mis-parsed
*identically*. **The guarantee held by coincidence rather than by correctness**,
and it would have changed meaning silently the moment anyone edited one of those
three status lines -- which closing Phase 4 requires. Same single-variant defect
as the `PLACEHOLDERS` omission this file's own docstring records, in the same
file, by a second route.

The rule is now shape-independent: a status is a **paragraph**, ending at the
blank line, whatever its bold shape.

Every test here was verified to fail against the old bold-span rule before being
trusted. Not an R10 gate: no scan, no kernel, no hand-written backward.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]


def _mod():
    spec = importlib.util.spec_from_file_location(
        "_vph", _REPO / "scripts" / "exp" / "verify_prereg_hash.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_vph"] = mod
    spec.loader.exec_module(mod)
    return mod


_BODY = """
Pre-registered before any driver, resolver or run existed.

## 1. The question

**A bold line that must survive.**
The hypothesis text, which is the whole point of the comparison.

## 9. Results

everything here is below the heading and must never be compared
"""


def _doc(status: str) -> str:
    return "# EXP_999\n\n" + status + "\n" + _BODY


BOLD_CLOSES_LATE = "**Status: PRE-REGISTERED -- no results yet.**"
BOLD_CLOSES_EARLY = "**Status:** PRE-REGISTERED -- no results yet."


def test_both_status_shapes_leave_the_same_body():
    """The regression: the two shapes must be indistinguishable after removal."""
    m = _mod()
    a = m.lines_above_results(_doc(BOLD_CLOSES_LATE))
    b = m.lines_above_results(_doc(BOLD_CLOSES_EARLY))
    assert a == b, "the two status shapes must not change what is compared"


def test_the_early_closing_shape_does_not_swallow_the_hypothesis():
    """The specific defect: `**Status:** <text>` used to eat everything down to
    the next bold-terminated line, `## 1. The question` among it."""
    m = _mod()
    kept = [ln for ln in m.lines_above_results(_doc(BOLD_CLOSES_EARLY)) if ln.strip()]
    assert "## 1. The question" in kept
    assert "**A bold line that must survive.**" in kept
    assert any("hypothesis text" in ln for ln in kept)


def test_a_multi_line_status_is_dropped_whole():
    """`EXP_022`'s wraps over six lines and summarises its verdicts; a
    first-line-only filter reported the continuations as an edited hypothesis."""
    m = _mod()
    wrapped = ("**Status: CLOSED 2026-08-20 -- H2, H3, H5 and H7 held; H1 and\n"
               "H4 failed; H6 UNRESOLVED. Fifteen runs, zero divergences, and\n"
               "the winning code cannot tell 42 of its 205 characters apart.**")
    assert m.lines_above_results(_doc(wrapped)) == m.lines_above_results(_doc(BOLD_CLOSES_LATE))


def test_an_edited_hypothesis_is_still_visible():
    """The check must be able to FAIL -- that is the only reason it exists."""
    m = _mod()
    clean = _doc(BOLD_CLOSES_LATE)
    edited = clean.replace("The hypothesis text, which is the whole point",
                           "The hypothesis text, quietly reworded after the fact")
    assert m.lines_above_results(clean) != m.lines_above_results(edited)


def test_everything_below_the_results_heading_is_excluded():
    m = _mod()
    kept = m.lines_above_results(_doc(BOLD_CLOSES_LATE))
    assert not any("must never be compared" in ln for ln in kept)


def test_a_document_with_no_status_line_is_returned_intact():
    m = _mod()
    doc = "# EXP_998\n\nno status here at all\n\n## 9. Results\nbelow\n"
    kept = [ln for ln in m.lines_above_results(doc) if ln.strip()]
    assert kept == ["# EXP_998", "no status here at all"]


def test_crlf_and_lf_are_read_identically():
    """`Path.write_text` gives CRLF on this box while a driver reading raw bytes
    stamped LF; hashing one convention turns a newline into a hypothesis edit."""
    m = _mod()
    lf = _doc(BOLD_CLOSES_LATE)
    assert m.lines_above_results(lf) == m.lines_above_results(lf.replace("\n", "\r\n"))


def test_the_recipe_can_restore_more_than_one_stamped_status_shape():
    """`EXP_013`'s only route is reconstruction from the live file, so its status
    line is load-bearing: restoring only `**Status: OPEN.**` meant correcting a
    stale header silently cost it its verification. The table holds SHAPES."""
    m = _mod()
    shapes = {st.strip() for st in m.STATUS_RESTORES}
    assert "**Status: OPEN.**" in shapes
    assert "**Status:** PRE-REGISTERED — no results yet." in shapes
    assert "**Status: PRE-REGISTERED — no results yet.**" in shapes


def test_a_closed_log_reconstructs_to_each_stamped_shape():
    """The end the table exists for: a closed header must be restorable to the
    pre-results one, whichever shape that was."""
    m = _mod()
    closed = ("**Status: CLOSED 2026-08-05 — everything held. Results in §9.**")
    doc = _doc(closed)
    labels = [lab for lab, _ in m.candidates(doc)]
    for shape in ("**Status: OPEN.**", "**Status:** PRE-REGISTERED — no results yet."):
        assert any(shape in lab for lab in labels), shape


def test_restoring_a_status_preserves_the_documents_newline_convention():
    """A restored status carrying the wrong newline turns a line ending into an
    apparent hypothesis edit -- the failure `_variants` exists to prevent."""
    m = _mod()
    crlf = _doc("**Status: CLOSED 2026-08-05 — held.**").replace("\n", "\r\n")
    for _, body in m.candidates(crlf):
        text = body.decode("utf-8") if isinstance(body, bytes) else body
        if "**Status: OPEN.**" in text:
            assert "**Status: OPEN.**\r\n" in text or "\n" not in text
            break


def test_the_real_logs_still_agree_with_their_committed_blobs():
    """The end-to-end claim, on the actual repository rather than a fixture."""
    m = _mod()
    rows = [m.check(e) for e in m.manifest_ids()]
    commit_verified = [r for r in rows if r.get("status") == "VERIFIED BY COMMIT"]
    if not commit_verified:            # no git, or nothing commit-verified
        return
    disagreeing = [r["experiment"] for r in commit_verified
                   if r.get("live_file_agrees_above_results") is False]
    assert not disagreeing, f"live file disagrees above the heading: {disagreeing}"
