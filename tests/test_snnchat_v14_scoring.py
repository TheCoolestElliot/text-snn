"""The v14 evidence scripts: the pool writer and the zero-GPU scorer.

`scripts/chat/score_v14.py` decides whether `RerankParams.subject_tier` is
adopted, from a replay. A replay that is subtly not the shipped selector still
produces a table, and the table still looks like a result -- so what is tested
here is the replay against answers worked out BY HAND, the abort that fires when
a stored pool and the replay disagree, and `scripts/chat/v14_pools.py`'s
recording of the REAL `snnchat.rerank.select` on CPU, which the replay must then
reproduce index for index.

Every test here was seen to fail once, against a deliberate break of the thing
it guards; the break is named in each docstring.

CPU only. No checkpoint, no corpus, no GPU.
"""
from __future__ import annotations

import copy
import gzip
import json
import os
import sys

import pytest
import torch

from snnchat.generate import SamplingParams
from snnchat.model import ChatConfig, build_chat_model
from snnchat.prime import tier_subject
from snnchat.rerank import (
    Candidate,
    RerankParams,
    echo_weight,
    echoes,
    prompt_content_words,
    select,
)
from snnchat.tokenizer import EOT, ChatTokenizer

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# The two scripts import each other (and `echo_holdout`) by bare name, the way
# `prime_probe.py` imports `echo_holdout`; importing them the same way here means
# the test and the scorer share ONE `v14_pools` module object.
sys.path.insert(0, os.path.join(REPO, "scripts", "chat"))

import score_v14 as scorer  # noqa: E402
import v14_pools as pools  # noqa: E402

PROMPT = "tell me a story about a penguin"
_FILL = "It was a nice day and they played in the park. " * 5
#: Echoes the request FRAME (tell + story) and names no penguin; its definite
#: subject "The pig" was never introduced, so UER@200 flags it.
FRAME = "Let me tell you a story. The pig was sad. " + _FILL
#: Names the subject; nothing unintroduced.
NAMES = "Once there was a penguin. The penguin was happy. " + _FILL
NAMES_TOO = "A penguin went to the sea. " + _FILL
#: Names the subject but is below `min_chars`: the length partition sets it aside.
SHORT = "Penguin."
PLAIN = _FILL


def _cand(text: str, logp_cond: float, logp_null: float, subject: str | None = "penguin",
          prompt: str = PROMPT) -> dict:
    """A stored candidate whose derived fields are what `select` would write."""
    return {"text": text, "n_chars": len(text), "n_scored": len(text) + 1, "closed": True,
            "logp_cond": logp_cond, "logp_null": logp_null,
            "echo_weight": echo_weight(text, prompt_content_words(prompt)),
            "subject_hit": bool(subject) and echoes(subject, text.lower())}


def _hand_pool() -> list[dict]:
    """Four candidates, worked by hand in `test_replay_matches_a_hand_computed_answer`."""
    return [
        _cand(FRAME, -110.0, -100.0),       # 0: tell + story = 11.14, the top weighted tier
        _cand(NAMES, -60.0, -100.0),        # 1: penguin = 10.34; score (-60 + 60) / 285 = 0
        _cand(NAMES_TOO, -50.0, -10.0),     # 2: penguin; score (-50 + 6) / 268 = -0.164
        _cand(SHORT, -0.1, -0.2),           # 3: 8 chars; score (+0.02) / 9 would beat all
    ]


def _drafts(pool: list[dict]) -> list:
    return scorer.drafts_v14({"pool": pool})


def _replay(pool, *, lam=0.6, subject_tier, has_subject=True):
    return scorer.replay(_drafts(pool), lam=lam, min_chars=12, subject_tier=subject_tier,
                         has_subject=has_subject)


def _blob(draws: list[dict], probe_set: str = "heldout", lam: float = 0.6,
          ckpt: str = "experiments/chat/none/ckpt_best.pt",
          ckpt_sha256: str | None = None) -> dict:
    return {"format": pools.FORMAT, "ckpt": ckpt, "ckpt_sha256": ckpt_sha256,
            "probe_set": probe_set, "n": 4, "lam": lam, "min_chars": 12, "draws": draws}


def _hand_draw(seed: int = 6) -> dict:
    return {"kind": "story", "expect": ["penguin"], "prompt": PROMPT, "seed": seed,
            "subject": "penguin", "shipped_index": 0, "subject_index": 1,
            "pool": _hand_pool()}


# ---------------------------------------------------------------------------
# the replay, against answers worked out by hand
# ---------------------------------------------------------------------------


def test_replay_matches_a_hand_computed_answer():
    """Both rules, lambda on and off, on one four-candidate pool.

    Shipped: the weighted tier's maximum is 11.14 (tell + story) and only
    candidate 0 has it, so the tier is one draft and it wins with no score --
    the defect the subject rule is aimed at, in miniature. Subject rule: the
    drafts that name the penguin among those long enough, {1, 2}; at lambda 0.6
    candidate 1 scores 0 against -0.164, at lambda 0 it is -60/285 = -0.2105
    against -50/268 = -0.1866 and candidate 2 wins instead.

    Mutations that turned this red: dropping the length partition from `replay`
    (the 8-character candidate 3 wins the subject tier); passing
    `subject_tier=False` through to `echo_tier` regardless (the subject rule
    returns 0).
    """
    pool = _hand_pool()
    shipped = _replay(pool, subject_tier=False)
    assert (shipped.index, shipped.tier_size, shipped.decided) == (0, 1, True)
    subject = _replay(pool, subject_tier=True)
    assert (subject.index, subject.tier_size, subject.decided) == (1, 2, False)
    assert _replay(pool, lam=0.0, subject_tier=True).index == 2
    # No subject known: the subject rule IS the old rule.
    assert _replay(pool, subject_tier=True, has_subject=False).index == 0


def test_replay_scores_with_n_scored_not_n_chars():
    """Two drafts whose order flips between the two denominators.

    Draft `a` is 10 characters and closed its turn, so 11 ids were scored;
    draft `b` is 11 characters and did not. By `n_scored`, `a` wins: -10.0/11 =
    -0.909 against -10.9/11 = -0.991. By `n_chars`, `b` would: -10.0/10 = -1.000
    against -0.991. Mutation: `max(d.n_chars, 1)` in `replay`'s key.
    """
    a = _cand("x" * 10 + " aa", 0.0, 0.0)
    b = _cand("y" * 10 + " bb", 0.0, 0.0)
    a.update(n_chars=10, n_scored=11, logp_cond=-10.0, echo_weight=0.0, subject_hit=False)
    b.update(n_chars=11, n_scored=11, logp_cond=-10.9, echo_weight=0.0, subject_hit=False)
    got = scorer.replay(_drafts([b, a]), lam=0.0, min_chars=5, subject_tier=False,
                        has_subject=False)
    assert got.index == 1


def test_a_one_member_tier_wins_without_a_score_and_without_a_null_term():
    """The shipped `decided` rule, and what it means for an unmeasured pool.

    With no `logp_null` anywhere the shipped rule is still replayable (its tier
    is one draft) and the subject rule is NOT (two drafts, a score is needed) --
    which is exactly why the committed v12 pools only support a lambda replay on
    a subset. At lambda 0 no null term is needed and both replay. When only one
    draft names the subject the subject rule is decided too.

    Mutation: removing `not decided` from `replay`'s replayability test makes
    the shipped pick None here.
    """
    drafts = _drafts(_hand_pool())
    for d in drafts:
        d.null_measured = False
    kw = {"min_chars": 12, "has_subject": True}
    assert scorer.replay(drafts, lam=0.6, subject_tier=False, **kw).index == 0
    assert scorer.replay(drafts, lam=0.6, subject_tier=True, **kw).index is None
    assert scorer.replay(drafts, lam=0.0, subject_tier=True, **kw).index == 2

    lone = [d for i, d in enumerate(drafts) if i != 2]
    got = scorer.replay(lone, lam=0.6, subject_tier=True, **kw)
    assert (got.index, got.decided) == (1, True)


def test_when_no_draft_names_the_subject_the_tier_is_everyone_not_the_frame_tier():
    """The subject rule's fallback, which is the half of it that does the work.

    Nobody names the penguin. Shipped still forms a frame-word tier and hands
    the turn to candidate 0; the subject rule lets the score choose among the
    whole length-partitioned pool, and the plain story scores best.

    Mutation: `replay` applying the subject rule INSIDE the weighted tier
    instead of to the length pool, which is the fallback `echo_tier`'s docstring
    rules out; both rules then return 0.
    """
    pool = [_cand(FRAME, -110.0, -100.0), _cand(PLAIN, -40.0, -50.0), _cand(SHORT, -0.1, -0.2)]
    pool[2].update(text="Hi there.", echo_weight=0.0, subject_hit=False, n_chars=9, n_scored=10)
    assert _replay(pool, subject_tier=False).index == 0
    got = _replay(pool, subject_tier=True)
    assert (got.index, got.whole_pool, got.tier_size) == (1, True, 2)


def test_every_story_prompt_has_its_noun_as_subject_and_no_dodge_prompt_has_one():
    """GUARD 1 is an identity only because `tier_subject` is None on the guard set.

    Mutation: `probes_for("dodge")` also admitting the battery's `topic` probes,
    which are story requests and do have a subject.
    """
    dodge = pools.probes_for("dodge")
    assert len(dodge) == 12
    assert [tier_subject(p) for p, _, _ in dodge] == [None] * 12
    assert {k for _, _, k in dodge} == {"list", "fact"}
    for name in ("heldout", "fresh", "wide"):
        for prompt, expect, kind in pools.probes_for(name):
            assert tier_subject(prompt) == expect[0]
            assert kind == "story"


# ---------------------------------------------------------------------------
# the confirmatory abort
# ---------------------------------------------------------------------------


def _write(tmp_path, name: str, blob: dict):
    path = tmp_path / name
    pools.write_pool(path, blob)
    return path


def test_a_faithful_pool_loads_and_the_metrics_are_the_hand_computed_ones(tmp_path):
    """End to end on the hand pool: P1, P2, P3, the floor and the tier sizes.

    Shipped returns FRAME (no penguin; "The pig" unintroduced; -110/278 =
    -0.3957, under the -0.3934 floor). The subject rule returns NAMES (penguin;
    clean; -60/285 = -0.2105).

    Mutations: swapping `fixed` and `broken` in `compare`; reading
    `d.drafts[label]["shipped"]` for both arms.
    """
    path = _write(tmp_path, "v14_heldout_x.json.gz", _blob([_hand_draw()]))
    info, draws = scorer.load_file(path, exploratory=False)
    assert info["checks"]["stored_echo_weight_mismatch"] == 0
    c = scorer.compare(draws, "lam0.6")
    # Bare mention is reported and is NOT a bar: it carries no verdict to read.
    assert (c["topic_mention"]["shipped_only"], c["topic_mention"]["subject_only"]) == (0, 1)
    assert c["topic_mention"]["identity_violations"] == 0
    assert "verdict" not in c["topic_mention"]
    # NAMES introduces "a penguin" and says "The penguin" again; FRAME has none.
    p1 = c["P1_anchored_topic"]
    assert (p1["shipped"]["k"], p1["subject"]["k"]) == (0, 1)
    assert (p1["shipped_only"], p1["subject_only"], p1["verdict"]) == (0, 1, "pass")
    assert p1["mean_reply_chars"] == {"note": p1["mean_reply_chars"]["note"],
                                      "shipped": len(FRAME), "subject": len(NAMES)}
    assert c["P2_logp_per_char"]["pooled"]["delta"] == pytest.approx(-60 / 285 + 110 / 278)
    # Both picks HAVE a definite subject ("The pig was", "The penguin was"), so
    # this draw is inside the bar's denominator, and it is a real fix.
    assert (c["P3_uer"]["fixed"], c["P3_uer"]["broken"]) == (1, 0)
    assert c["P3_uer"]["both_qualify"]["k"] == 1
    assert (c["P3_uer"]["both_exposed"]["k"], c["P3_uer"]["both_exposed"]["n"]) == (1, 1)
    aq = c["P3_uer"]["all_qualifying_draws"]
    assert (aq["fixed"], aq["broken"]) == (1, 0)
    assert aq["fixed_where_the_subject_pick_has_no_definite_subject"] == 0
    assert c["below_floor"]["shipped"]["k"] == 1 and c["below_floor"]["subject"]["k"] == 0
    assert c["tier_size"]["shipped"]["one_member"]["k"] == 1
    assert c["tier_size"]["subject"]["buckets"]["2-4"] == 1
    assert c["no_draft_names_subject"]["k"] == 0


def _exposure_draw(seed: int) -> dict:
    """Shipped returns FRAME ("The pig was", unintroduced). The subject rule
    returns NAMES_TOO, which names the penguin and contains NO definite subject
    at all in its first 200 characters -- clean because there is nothing to flag."""
    return {"kind": "story", "expect": ["penguin"], "prompt": PROMPT, "seed": seed,
            "subject": "penguin", "shipped_index": 0, "subject_index": 1,
            "pool": [_cand(FRAME, -110.0, -100.0), _cand(NAMES_TOO, -50.0, -10.0)]}


def test_p3_is_not_passed_by_picks_that_merely_contain_no_definite_subject(tmp_path):
    """The review's finding, as a pool. Eight draws on which the subject pick
    "fixes" an unintroduced entity only by containing no definite subject: the
    all-draws count reads fixed 8, broken 0, McNemar p < 0.05 -- and the BAR
    must not, because not one entity was introduced any more carefully.

    Mutation: reading the bar over `qual` instead of `both` in `compare`.
    """
    assert scorer.definite_subjects(FRAME) == [("pig", False)]
    assert scorer.definite_subjects(NAMES_TOO) == []
    path = _write(tmp_path, "v14_heldout_x.json.gz",
                  _blob([_exposure_draw(seed) for seed in range(6, 14)]))
    _, draws = scorer.load_file(path, exploratory=False)
    c = scorer.compare(draws, "lam0.6")
    p3 = c["P3_uer"]

    # NAMES_TOO says "penguin" ONCE: a mention, which is what the tier selected
    # it for, and not an anchored topic, which needs the noun referred back to.
    assert (c["topic_mention"]["subject"]["k"], c["topic_mention"]["subject_only"]) == (8, 8)
    assert (c["P1_anchored_topic"]["subject"]["k"],
            c["P1_anchored_topic"]["subject_only"]) == (0, 0)

    aq = p3["all_qualifying_draws"]
    assert (aq["fixed"], aq["broken"]) == (8, 0) and aq["mcnemar_p"] < scorer.P3_ALPHA
    assert aq["fixed_where_the_subject_pick_has_no_definite_subject"] == 8
    assert "verdict" not in aq

    assert (p3["both_exposed"]["k"], p3["fixed"], p3["broken"]) == (0, 0, 0)
    assert p3["verdict"] == "unresolved"

    # The decomposition says what happened: exposure fell to nothing, and the
    # rate among exposed replies has no subject-arm reading at all.
    dec = p3["decomposition"]
    assert (dec["shipped"]["exposure"]["k"], dec["shipped"]["exposure"]["n"]) == (8, 8)
    assert (dec["subject"]["exposure"]["k"], dec["subject"]["exposure"]["n"]) == (0, 8)
    assert dec["shipped"]["uer_given_exposed"]["rate"] == 1.0
    assert dec["subject"]["uer_given_exposed"]["rate"] is None


def test_sameness_is_reported_with_its_withdrawn_bar_and_is_not_in_overall():
    """Amendment A (`PREDICTION_v14.md` section 12): the owner ruled that sameness
    is not a bar. GUARD 2 is still measured, reads `reported` and nothing else,
    and says what the withdrawn 0.10 bar WOULD have read through the unchanged
    `guard2_verdict`; `overall` no longer reads it. P1, P2, P3 and GUARD 1 are
    what `overall` reads, and bare mention is still not.

    Mutations: `pooled_drop > bar` to `pooled_drop >= bar` in `guard2_verdict`;
    "GUARD2_sameness" put back into `IN_OVERALL`; `guard2_reported` returning
    `guard2_verdict`'s own verdict as its `verdict`; `would_have_read` hard-coded
    to a pass; adding "topic_mention" to `IN_OVERALL`; dropping "P3_uer" from it.
    """
    bar = scorer.GUARD2_MAX_DISTINCT2_DROP
    every = {name: bar / 2 for name in scorer.P2_LISTS}
    assert scorer.distinct2_drop({"value": 0.25}, {"value": 0.20}) == pytest.approx(0.2)
    assert scorer.distinct2_drop({"value": None}, {"value": None}) is None
    assert scorer.guard2_verdict(bar / 2, every)["verdict"] == "pass"
    assert scorer.guard2_verdict(bar, every)["verdict"] == "pass"          # "at most"
    assert scorer.guard2_verdict(bar * 1.01, every)["verdict"] == "fail"
    assert scorer.guard2_verdict(-0.05, every)["verdict"] == "pass"        # it ROSE
    assert scorer.guard2_verdict(bar / 2, {**every, "wide": bar * 2})["verdict"] == "unresolved"
    assert scorer.guard2_verdict(bar / 2, {"heldout": 0.0})["verdict"] == "unresolved"
    assert scorer.guard2_verdict(None, {})["verdict"] == "unresolved"

    # What the block carries now: never a pass/fail/unresolved of its own, and
    # the old reading beside it, whichever way the old reading goes.
    for drop, lists, old in ((bar * 3, every, "fail"), (bar / 2, every, "pass"),
                             (bar / 2, {**every, "wide": bar * 2}, "unresolved"),
                             (None, {}, "unresolved")):
        got = scorer.guard2_reported(drop, lists)
        assert got["verdict"] == "reported"
        assert got["withdrawn_bar"] == bar == 0.10
        assert got["would_have_read"] == scorer.guard2_verdict(drop, lists)
        assert got["would_have_read"]["verdict"] == old
        assert "As many as needed" in got["why"] and "2026-09-21" in got["why"]

    assert scorer.IN_OVERALL == ("P1_anchored_topic", "P2_logp_per_char", "P3_uer",
                                 "GUARD1_identity")
    passing = {name: {"verdict": "pass"} for name in scorer.IN_OVERALL}
    g1 = {"verdict": "pass"}
    assert scorer.overall(passing, g1) == "pass"
    # The ruling removes a bar: a sameness reading of any kind moves nothing ...
    for verdict in ("fail", "unresolved", "reported"):
        assert scorer.overall({**passing, "GUARD2_sameness": {"verdict": verdict}}, g1) == "pass"
    # ... and rescues nothing: every other bar still decides.
    for name in ("P1_anchored_topic", "P2_logp_per_char", "P3_uer"):
        assert scorer.overall({**passing, name: {"verdict": "fail"}}, g1) == "fail"
    assert scorer.overall({**passing, "P3_uer": {"verdict": "unresolved"}}, g1) == "unresolved"
    assert scorer.overall(passing, {"verdict": "fail"}) == "fail"
    # A failing bare-mention row, if anyone ever gave it a verdict, is not read.
    assert scorer.overall({**passing, "topic_mention": {"verdict": "fail"}}, g1) == "pass"


def test_identical_picks_on_every_draw_are_reported_with_what_the_old_bar_read(tmp_path):
    """End to end: the subject rule returning ONE reply for every draw while the
    shipped rule returns varied ones is a fall in distinct-2 that `compare` must
    still MEASURE and show -- as `reported`, with the withdrawn bar's `fail`
    beside it -- and that `overall` must not read (Amendment A).

    Mutation: `compare` spreading `guard2_verdict` again instead of
    `guard2_reported`."""
    draws = []
    for i, word in enumerate(("red", "blue", "green", "pink")):
        varied = f"Let me tell you a story about the {word} kite and a {word} hat. " + _FILL
        draws.append({"kind": "story", "expect": ["penguin"], "prompt": PROMPT, "seed": 6 + i,
                      "subject": "penguin", "shipped_index": 0, "subject_index": 1,
                      "pool": [_cand(varied, -110.0, -100.0), _cand(NAMES, -60.0, -100.0)]})
    path = _write(tmp_path, "v14_heldout_x.json.gz", _blob(draws))
    _, loaded = scorer.load_file(path, exploratory=False)
    g2 = scorer.compare(loaded, "lam0.6")["GUARD2_sameness"]
    assert g2["distinct_2"]["subject"]["value"] < g2["distinct_2"]["shipped"]["value"]
    assert g2["relative_drop"] == pytest.approx(
        1 - g2["distinct_2"]["subject"]["value"] / g2["distinct_2"]["shipped"]["value"])
    assert g2["relative_drop"] > scorer.GUARD2_MAX_DISTINCT2_DROP
    assert g2["relative_drop_per_list"] == {"heldout": g2["relative_drop"]}
    assert g2["verdict"] == "reported"
    assert (g2["withdrawn_bar"], g2["would_have_read"]["verdict"]) == (0.10, "fail")
    assert (g2["top_opening"]["shipped"]["k"], g2["top_opening"]["subject"]["k"]) == (1, 4)

    # Through `main`: the fall is in the artifact and `overall` is not `fail`
    # because of it. One list of four draws leaves P2 short of its three lists.
    out = tmp_path / "scored.json"
    assert scorer.main([str(path), "--out", str(out)]) == 0
    blob = json.loads(out.read_text(encoding="utf-8"))
    assert "GUARD2_sameness" not in blob["in_overall"]
    assert blob["bars"]["GUARD2_withdrawn_max_distinct2_drop"] == 0.10
    got = blob["subsets"]["lam0.6_all"]
    assert got["GUARD2_sameness"]["would_have_read"]["verdict"] == "fail"
    assert blob["GUARD1_identity"]["verdict"] == "unresolved"
    assert blob["overall"] == scorer.overall(got, blob["GUARD1_identity"])
    assert "fail" not in (got[name]["verdict"] for name in
                          ("P1_anchored_topic", "P2_logp_per_char", "P3_uer"))
    assert blob["overall"] == "unresolved"


def test_the_corpus_reference_is_read_from_the_committed_artifact(tmp_path):
    """`score_v14.py` prints each arm's exposure next to the corpus's, and takes
    the corpus's from `coherence_v14.json` rather than recomputing it. Absent or
    older artifact: None, not a crash and not a zero."""
    ref = scorer.corpus_reference()
    assert ref is not None and ref["source"] == "coherence_v14.json"
    for key in ("exposure", "uer_given_exposed", "unintroduced_per_subject"):
        assert set(ref[key]) == {"rate", "k", "n", "ci"}
    # uer = exposure * uer_given_exposed, exactly, against the artifact's own UER.
    committed = json.loads((scorer.QUALITY / "coherence_v14.json").read_text(encoding="utf-8"))
    whole = committed["corpus"]["with_stoplist"]
    assert ref["uer_given_exposed"]["k"] == whole["k"]
    assert ref["exposure"]["n"] == whole["n"]
    assert ref["uer_given_exposed"]["n"] == ref["exposure"]["k"]
    assert scorer.corpus_reference(tmp_path) is None
    (tmp_path / "coherence_v14.json").write_text('{"corpus": {"file": "x"}}', encoding="utf-8")
    assert scorer.corpus_reference(tmp_path) is None


@pytest.mark.parametrize("field,wrong", [("shipped_index", 1), ("subject_index", 2)])
def test_a_wrong_recorded_index_aborts_the_scorer(tmp_path, field, wrong):
    """The confirmatory assertion. A table is never printed over a disagreement.

    Mutation: deleting the index comparison in `load_file` lets both cases load.
    """
    draw = _hand_draw()
    draw[field] = wrong
    path = _write(tmp_path, "v14_heldout_x.json.gz", _blob([draw]))
    with pytest.raises(scorer.ReplayMismatch, match=field.split("_")[0]):
        scorer.load_file(path, exploratory=False)
    with pytest.raises(scorer.ReplayMismatch):
        scorer.main([str(path), "--out", str(tmp_path / "out.json")])
    assert not (tmp_path / "out.json").exists()


def test_a_pool_whose_stored_fields_do_not_describe_its_text_aborts(tmp_path):
    """`subject_hit` says candidate 1 does not name the penguin; its text does.

    Left alone that flips the subject tier while every index still "agrees" with
    a replay of the corrupted fields, so the fields are re-derived from the text.
    Mutation: dropping the `stored_*_mismatch` check from `load_file`.
    """
    draw = _hand_draw()
    draw["pool"][1]["subject_hit"] = False
    draw["subject_index"] = 2           # what the corrupted fields replay to
    path = _write(tmp_path, "v14_heldout_x.json.gz", _blob([draw]))
    with pytest.raises(scorer.ReplayMismatch, match="stored text"):
        scorer.load_file(path, exploratory=False)


def test_confirmatory_mode_refuses_a_committed_echo_holdout_pool(tmp_path):
    """Those pools carry no recorded subject winner; only --exploratory reads them.

    Mutation: removing the `not is_v14 and not exploratory` refusal.
    """
    path = tmp_path / "v12_like.json"
    path.write_text(json.dumps({"ckpt": "x", "probe_set": "heldout", "n": 4, "lam": 0.6,
                                "rows": []}), encoding="utf-8")
    with pytest.raises(SystemExit, match="exploratory"):
        scorer.load_file(path, exploratory=False)
    info, draws = scorer.load_file(path, exploratory=True)
    assert info["format"] == "echo_holdout" and draws == []


def test_exploratory_mode_recovers_n_scored_and_reproduces_the_recorded_picks(tmp_path):
    """An `echo_holdout.py`-format row, as that script would have written it.

    A draft shorter than `max_new` closed its turn, so `len(scored)` is
    `n_chars + 1`; one at `max_new` did not. The file's own `echo_logp` and
    `base_logp` are the check. Mutation: `n_scored_rule` returning `n_chars`
    (one `n_scored_mismatch` and two pick mismatches appear).
    """
    assert scorer.n_scored_rule(299, 300) == 300
    assert scorer.n_scored_rule(300, 300) == 300
    hand = _hand_pool()
    row_pool = [{k: c[k] for k in ("text", "n_chars", "logp_cond", "logp_null",
                                   "echo_weight")} for c in hand]
    for c in row_pool:
        c["logp_null"] = 0.0              # a decided draw: the null pass was skipped
    row = {"prompt": PROMPT, "seed": 0, "expect": ["penguin"], "pool": row_pool,
           # shipped pick: candidate 0, decided. Score-only pick at an unmeasured
           # null: the best logp_cond / len(scored) among the long ones, 2.
           "echo_text": FRAME[:300], "echo_logp": -110.0 / (len(FRAME) + 1),
           "base_text": NAMES_TOO[:300], "base_logp": -50.0 / (len(NAMES_TOO) + 1)}
    path = tmp_path / "v12_like.json"
    path.write_text(json.dumps({"ckpt": "x", "probe_set": "heldout", "n": 4, "lam": 0.6,
                                "rows": [row]}), encoding="utf-8")
    info, draws = scorer.load_file(path, exploratory=True)
    checks = info["checks"]
    assert checks["n_scored_checked_closed"] == 2 and checks["n_scored_mismatch"] == 0
    assert checks["shipped_pick_mismatch"] == 0 and checks["base_pick_mismatch"] == 0
    (d,) = draws
    assert not d.null_whole_pool
    assert d.picks["lam0.6"]["shipped"].index == 0
    assert d.picks["lam0.6"]["subject"].index is None       # needs a null term it lacks
    assert d.picks["lam0"]["subject"].index == 2
    assert d.names_subject == 2                             # SHORT is not in the length pool

    # The same bytes gzip-compressed, which is how the chat-v6-scratch pools are
    # committed (`v6scratch_*_n256.json.gz`), read the same. Mutation:
    # `read_pool` no longer decompressing on the gzip magic.
    zipped = tmp_path / "v12_like.json.gz"
    zipped.write_bytes(gzip.compress(path.read_bytes(), mtime=0))
    info_gz, draws_gz = scorer.load_file(zipped, exploratory=True)
    assert info_gz["checks"] == checks and info_gz["format"] == "echo_holdout"
    assert scorer.draw_rows(draws_gz) == [
        {**row, "source": zipped.name} for row in scorer.draw_rows(draws)]


# ---------------------------------------------------------------------------
# Amendment A's two refusals: the stamp, and one checkpoint per verdict
# ---------------------------------------------------------------------------


def test_the_stamp_is_over_lf_bytes_and_a_changed_pre_registration_is_refused(tmp_path):
    """`check_stamp` hashes with CRLF normalised to LF, so the committed LF blob
    and a CRLF checkout agree -- `score_v13.py`'s raw-bytes recipe does not --
    and anything else about the text changes the hash and stops the scorer.

    The committed `PREDICTION_v14.md` must hash to `PREDICTION_STAMP`: editing
    that file (appending results to it, say) turns this red, and the fix is to
    put results in `QUALITY_v14.md`, not to restamp.

    Mutations: `prediction_stamp` hashing the raw bytes (the CRLF copy no longer
    agrees); `check_stamp` comparing `got` with `got` (the edited copy passes).
    """
    import hashlib

    text = b"# a pre-registration\n\nthe bar is +0.05\n"
    lf, crlf, edited = (tmp_path / n for n in ("lf.md", "crlf.md", "edited.md"))
    lf.write_bytes(text)
    crlf.write_bytes(text.replace(b"\n", b"\r\n"))
    edited.write_bytes(text.replace(b"+0.05", b"+0.04"))
    stamp = hashlib.sha256(text).hexdigest()
    assert scorer.prediction_stamp(lf) == scorer.prediction_stamp(crlf) == stamp
    assert scorer.check_stamp(crlf, stamp) == stamp
    with pytest.raises(SystemExit, match="nothing is scored"):
        scorer.check_stamp(edited, stamp)
    with pytest.raises(SystemExit, match="cannot read"):
        scorer.check_stamp(tmp_path / "absent.md", stamp)

    assert scorer.check_stamp() == scorer.PREDICTION_STAMP
    # The two checkpoints the scorer recognises are the two the document names.
    registered = scorer.PREDICTION.read_text(encoding="utf-8")
    assert sorted(scorer.REGISTERED_CHECKPOINTS) == ["chat-v3d-aligned", "chat-v6-scratch"]
    for name, sha in scorer.REGISTERED_CHECKPOINTS.items():
        assert len(sha) == 64 and sha in registered and name in registered


def test_confirmatory_mode_scores_nothing_against_an_unstamped_pre_registration(
        tmp_path, monkeypatch):
    """Through `main`: a wrong stamp stops a confirmatory run before any file is
    written, and does not stop `--exploratory`, which tests nothing.

    Mutation: `main` not calling `check_stamp` (the confirmatory run writes its
    artifact).
    """
    path = _write(tmp_path, "v14_heldout_x.json.gz", _blob([_hand_draw()]))
    out = tmp_path / "out.json"
    monkeypatch.setattr(scorer, "PREDICTION_STAMP", "0" * 64)
    with pytest.raises(SystemExit, match="not the stamp"):
        scorer.main([str(path), "--out", str(out)])
    assert not out.exists()
    assert scorer.main([str(path), "--exploratory", "--out", str(out)]) == 0
    assert json.loads(out.read_text(encoding="utf-8"))["prediction"]["stamp"] is None


def test_pools_from_two_checkpoints_are_refused_and_the_output_names_the_one(
        tmp_path, monkeypatch):
    """Amendment A draws the same sets from two checkpoints into one directory,
    and `v14_*.json.gz` matches both. One verdict is about one checkpoint.

    Mutations: `one_checkpoint` never raising (the glob is pooled and scored);
    keying on the run name alone (two files with one name and two hashes pass);
    the default output path back to `v14_confirmatory.json`.
    """
    old, new = "a" * 64, "b" * 64
    kw_old = {"ckpt": "C:/x/experiments/chat/chat-old/ckpt_best.pt", "ckpt_sha256": old}
    kw_new = {"ckpt": "C:/x/experiments/chat/chat-new/ckpt_best.pt", "ckpt_sha256": new}
    _write(tmp_path, "v14_heldout_chat-old.json.gz", _blob([_hand_draw()], **kw_old))
    _write(tmp_path, "v14_fresh_chat-old.json.gz",
           _blob([_hand_draw(7)], probe_set="fresh", **kw_old))
    _write(tmp_path, "v14_heldout_chat-new.json.gz", _blob([_hand_draw()], **kw_new))
    monkeypatch.setattr(scorer, "QUALITY", tmp_path / "quality")

    with pytest.raises(SystemExit, match="2 checkpoints"):
        scorer.main([str(tmp_path / "v14_*.json.gz")])
    assert not (tmp_path / "quality").exists()

    # One name, two hashes: a checkpoint retrained in place is two checkpoints.
    _write(tmp_path / "again", "v14_heldout_chat-old.json.gz",
           _blob([_hand_draw()], **{**kw_old, "ckpt_sha256": new}))
    with pytest.raises(SystemExit, match="2 checkpoints"):
        scorer.main([str(tmp_path / "v14_heldout_chat-old.json.gz"),
                     str(tmp_path / "again" / "v14_heldout_chat-old.json.gz")])

    assert scorer.main([str(tmp_path / "v14_*_chat-old.json.gz")]) == 0
    assert [f.name for f in (tmp_path / "quality").iterdir()] == [
        "v14_confirmatory_chat-old.json"]
    blob = json.loads((tmp_path / "quality" / "v14_confirmatory_chat-old.json")
                      .read_text(encoding="utf-8"))
    assert blob["checkpoint"] == {"name": "chat-old", "sha256": old,
                                  "pre_registered": False, "decides_the_default": False}
    assert blob["prediction"]["stamp"] == scorer.PREDICTION_STAMP
    assert blob["n_story_draws"] == 2

    # What the two registered checkpoints read as: only SHIPPED's decides.
    def described(name, sha):
        return scorer.one_checkpoint([{"ckpt": f"C:/t/{name}/ckpt_best.pt", "ckpt_sha256": sha}])

    reg = scorer.REGISTERED_CHECKPOINTS
    got = described("chat-v6-scratch", reg["chat-v6-scratch"])
    assert (got["pre_registered"], got["decides_the_default"]) == (True, True)
    got = described("chat-v3d-aligned", reg["chat-v3d-aligned"])
    assert (got["pre_registered"], got["decides_the_default"]) == (True, False)
    got = described("chat-v6-scratch", reg["chat-v3d-aligned"])
    assert (got["pre_registered"], got["decides_the_default"]) == (False, False)
    got = described("chat-v6-scratch", None)
    assert (got["pre_registered"], got["decides_the_default"]) == (False, False)


def test_a_confirmatory_file_list_must_be_the_draws_the_design_registers(tmp_path):
    """Section 2: sampler seeds 6-11 on every prompt of the five sets, once --
    600 story draws, 72 dodge, 72 clause, "no more and no fewer". Nothing in a
    pool file enforces that: `v14_pools.py --allow-committed-seeds` draws a
    v14-format pool at a READ seed, and chat-v6-scratch's seeds 0-5 are read.

    Mutations: the `spent` refusal deleted (a seed-5 pool is scored); `twice`
    keyed on `(source, prompt, seed)` (the same file under two paths is counted
    twice); the design refusal deleted, or `pre_registered` ignored (one draw
    decides the default); `CONFIRMATORY_SEEDS` ending at 12; `given` built
    without the set's name (a draw filed under the wrong list passes).
    """
    from types import SimpleNamespace as D

    other = {"name": "chat-tiny", "pre_registered": False}
    deciding = {"name": "chat-v6-scratch", "pre_registered": True}
    design = [D(source=f"v14_{name}.json.gz", probe_set=name, prompt=prompt, seed=seed)
              for name in pools.SET_NAMES for prompt, _, _ in pools.probes_for(name)
              for seed in range(6, 12)]
    assert sum(1 for d in design if d.probe_set in scorer.P2_LISTS) == 600
    assert scorer.check_design(design, deciding) == {
        "sampler_seeds": [6, 7, 8, 9, 10, 11], "n_draws": 744, "registered_n_draws": 744,
        "missing": 0, "not_in_the_design": 0, "is_the_registered_design": True}

    first = design[0]
    beyond = D(**{**vars(first), "seed": 12})
    misfiled = D(**{**vars(first), "probe_set": "wide" if first.probe_set != "wide" else "fresh"})
    for wrong, missing, extra in (
            (design[1:], 1, 0),                                       # one draw short
            (design + [beyond], 0, 1),                                # a seed past 11
            ([d for d in design if d.probe_set == "clause"], 672, 0),  # one set alone
            ([misfiled] + design[1:], 1, 1)):                         # under another list
        with pytest.raises(SystemExit, match="not the registered design"):
            scorer.check_design(wrong, deciding)
        got = scorer.check_design(wrong, other)
        assert (got["missing"], got["not_in_the_design"]) == (missing, extra)
        assert got["is_the_registered_design"] is False

    # Read seeds and repeats are refused whatever the checkpoint.
    with pytest.raises(SystemExit, match="confirm nothing"):
        scorer.check_design([D(**{**vars(first), "seed": 5})], other)
    again = D(**{**vars(first), "source": "a_copy.json.gz"})
    with pytest.raises(SystemExit, match="more than once"):
        scorer.check_design([first, again], other)

    # Through `main`, where each refusal must come before any artifact.
    out = tmp_path / "out.json"
    read = _write(tmp_path, "v14_heldout_read.json.gz", _blob([_hand_draw(5)]))
    with pytest.raises(SystemExit, match="confirm nothing"):
        scorer.main([str(read), "--out", str(out)])
    fresh = _write(tmp_path, "v14_heldout_x.json.gz", _blob([_hand_draw()]))
    copied = _write(tmp_path / "copy", "v14_heldout_x.json.gz", _blob([_hand_draw()]))
    with pytest.raises(SystemExit, match="more than once"):
        scorer.main([str(fresh), str(copied), "--out", str(out)])
    sha = scorer.REGISTERED_CHECKPOINTS["chat-v6-scratch"]
    alone = _write(tmp_path, "v14_heldout_chat-v6-scratch.json.gz", _blob(
        [_hand_draw()], ckpt="C:/x/experiments/chat/chat-v6-scratch/ckpt_best.pt",
        ckpt_sha256=sha))
    with pytest.raises(SystemExit, match="not the registered design"):
        scorer.main([str(alone), "--out", str(out)])
    assert not out.exists()

    assert scorer.main([str(fresh), "--out", str(out)]) == 0
    got = json.loads(out.read_text(encoding="utf-8"))["design"]
    assert (got["n_draws"], got["sampler_seeds"], got["is_the_registered_design"]) == (
        1, [6], False)


def test_exploratory_mode_does_not_overwrite_the_committed_artifact_by_default(
        tmp_path, monkeypatch):
    """`v14_exploratory.json` is the record section 1 was sized from, and section
    12.3 says it is not to be overwritten; `--exploratory` with no `--out` used
    to rewrite it in the amended format, and the test that re-derives it reads
    inputs and rows only, so nothing would have noticed.

    Mutation: the `exists()` refusal deleted (the sentinel is overwritten).
    """
    path = _write(tmp_path, "v14_heldout_x.json.gz", _blob([_hand_draw()]))
    monkeypatch.setattr(scorer, "QUALITY", tmp_path / "quality")
    record = tmp_path / "quality" / "v14_exploratory.json"
    record.parent.mkdir()
    record.write_text("the committed record\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="committed record"):
        scorer.main([str(path), "--exploratory"])
    assert record.read_text(encoding="utf-8") == "the committed record\n"
    # Naming it is still allowed, and so is the default when nothing is there.
    assert scorer.main([str(path), "--exploratory", "--out", str(record)]) == 0
    assert json.loads(record.read_text(encoding="utf-8"))["mode"] == "exploratory"
    record.unlink()
    assert scorer.main([str(path), "--exploratory"]) == 0 and record.exists()


# ---------------------------------------------------------------------------
# the real selector on CPU, recorded by v14_pools and reproduced by the replay
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def tiny_model():
    torch.manual_seed(0)
    cfg = ChatConfig(d_model=32, n_layers=2, vocab_size=ChatTokenizer().vocab_size,
                     device="cpu", spread_tau=True, seed=0)
    model = build_chat_model(cfg)
    model.eval()
    return model


def _candidates(tok, texts_and_logps) -> list[Candidate]:
    out = []
    for text, logp in texts_and_logps:
        ids = [int(i) for i in tok.encode(text)]
        out.append(Candidate(ids=ids, scored=[*ids, EOT], logp_cond=logp))
    return out


_POOLS = {
    "subject tier has two members": (PROMPT, [(FRAME, -110.0), (NAMES, -60.0),
                                              (NAMES_TOO, -50.0), (SHORT, -0.1)]),
    "both rules decided, no null pass in either": (PROMPT, [(FRAME, -110.0), (NAMES, -60.0),
                                                            (SHORT, -0.1)]),
    "nobody names the subject": (PROMPT, [(FRAME, -110.0), (PLAIN, -40.0),
                                          ("Let me tell you a story. " + PLAIN, -90.0)]),
    "a non-story prompt": ("name three animals", [("I can name three animals.", -9.0),
                                                  ("A dog, a cat, and a bird.", -8.0),
                                                  ("Three animals I can name.", -7.0)]),
}


@pytest.mark.parametrize("lam", [0.6, 0.0])
def test_recorded_pools_from_the_real_selector_replay_exactly(tiny_model, tmp_path, lam):
    """`record` over hand-written candidates, then the scorer's replay of them.

    The winners recorded are the REAL `select`'s, with the null term measured by
    a real (tiny, untrained) model, so nothing about the expected indices is
    assumed here -- the assertion is that `load_file` does not raise, i.e. the
    replay reproduces every one of them, and that GUARD 1 holds on the
    subject-less pool.

    Also holds `v14_pools`' promise that EVERY stored `logp_null` was measured:
    the "both rules decided" pool never runs a null pass inside `select`, and at
    lambda = 0 nothing does. Mutations: deleting the `fill_null_scores` /
    `_score_null` block from `record` stores 0.0 there; deleting only the forced
    `_score_null` fails the lambda = 0 case alone; `record` calling `select`
    with the shipped `rp` records an index the replay does not reproduce.
    """
    tok = ChatTokenizer()
    rp = RerankParams(n=4, lam=lam, graph=False)
    story, dodge = [], []
    # One sampler seed per pool: three of these share a prompt, and the scorer
    # refuses a (prompt, seed) it is given twice.
    for seed, (prompt, spec) in enumerate(_POOLS.values(), start=6):
        cands = _candidates(tok, spec)
        shipped = select(tiny_model, cands, rp, device="cpu", tok=tok,
                         echo_words=prompt_content_words(prompt))
        row = pools.record(tiny_model, tok, prompt, rp, cands, shipped, device="cpu")
        assert all(c["logp_null"] != 0.0 for c in row["pool"])
        assert all(c["n_scored"] == c["n_chars"] + 1 and c["closed"] for c in row["pool"])
        assert row["timing_pass_agrees"]
        subject = tier_subject(prompt)
        assert row["subject"] == subject
        full = {"kind": "story" if subject else "list", "prompt": prompt, "seed": seed,
                "expect": [subject or "dog"], "rerank_seconds": 0.0, **row}
        (story if subject else dodge).append(full)

    assert story[0]["shipped_index"] == 0 and story[0]["subject_index"] in (1, 2)
    assert (story[1]["shipped_index"], story[1]["subject_index"]) == (0, 1)
    assert dodge[0]["shipped_index"] == dodge[0]["subject_index"]

    a = _write(tmp_path, "v14_heldout_x.json.gz", _blob(story, lam=lam))
    b = _write(tmp_path, "v14_dodge_x.json.gz", _blob(dodge, probe_set="dodge", lam=lam))
    out = tmp_path / "scored.json"
    assert scorer.main([str(a), str(b), "--out", str(out)]) == 0
    blob = json.loads(out.read_text(encoding="utf-8"))
    assert blob["mode"] == "confirmatory" and blob["n_story_draws"] == 3
    assert blob["GUARD1_identity"]["verdict"] == "pass"
    assert blob["GUARD1_identity"]["n"] == 1
    # One list of three is not the three lists P2 is fixed on.
    assert blob["subsets"][f"lam{lam:g}_all"]["P2_logp_per_char"]["verdict"] in (
        "unresolved", "fail")
    assert blob["overall"] != "pass"


def test_guard_1_fails_when_a_subjectless_draw_was_selected_differently(tmp_path):
    """An identity, so one exception is a failure.

    The pool is built so that the two recorded indices differ on a prompt with no
    subject; the replay (correctly) says both rules pick 0, so this must abort as
    a mismatch rather than be scored -- and `guard1` itself, handed draws whose
    picks differ, must say fail. Mutation: `guard1` comparing `shipped` with
    `shipped`.
    """
    prompt = "name three animals"
    pool = [_cand("I can name three animals.", -9.0, -5.0, None, prompt),
            _cand("A dog, a cat, and a bird.", -8.0, -5.0, None, prompt)]
    draw = {"kind": "list", "expect": ["dog"], "prompt": prompt, "seed": 6, "subject": None,
            "shipped_index": 0, "subject_index": 0, "pool": pool}
    path = _write(tmp_path, "v14_dodge_x.json.gz", _blob([draw], probe_set="dodge"))
    _, draws = scorer.load_file(path, exploratory=False)
    assert scorer.guard1(draws, lambda d: "lam0.6")["verdict"] == "pass"

    forged = copy.deepcopy(draws)
    forged[0].picks["lam0.6"]["subject"].index = 1
    forged[0].drafts["lam0.6"]["subject"] = scorer.drafts_v14(draw)[1]
    got = scorer.guard1(forged, lambda d: "lam0.6")
    assert got["verdict"] == "fail" and got["identical"] == 0
    assert scorer.guard1([], lambda d: "lam0.6")["verdict"] == "unresolved"


def test_draw_samples_a_pool_on_cpu_and_stores_every_field(tiny_model):
    """The sampling path itself: `rerank` called once, fields consistent.

    An untrained model drafts noise, so this checks structure rather than
    selection: a draft closed iff it stopped before `max_new`, which is the rule
    exploratory mode uses to recover `n_scored`. Mutation: storing `c.n_chars`
    as `n_scored` in `record`.
    """
    tok = ChatTokenizer()
    rp = RerankParams(n=6, lam=0.6, graph=False)
    row = pools.draw(tiny_model, tok, PROMPT, 6, rp, max_new=40, device="cpu", timing=False)
    assert len(row["pool"]) == 6 and row["seed"] == 6 and row["subject"] == "penguin"
    assert "select_seconds_shipped" not in row
    for c in row["pool"]:
        assert c["n_scored"] == c["n_chars"] + int(c["closed"])
        assert c["closed"] == (c["n_chars"] < 40)
        assert c["n_scored"] == scorer.n_scored_rule(c["n_chars"], 40)
        assert c["logp_null"] != 0.0
    with pytest.raises(ValueError, match="pool of one"):
        pools.draw(tiny_model, tok, PROMPT, 6, RerankParams(n=1), max_new=8, device="cpu")


def test_the_command_line_draws_a_pool_the_scorer_accepts(tiny_model, tmp_path, capsys):
    """`v14_pools.py` to `score_v14.py`, end to end, as the lead will run them.

    A saved (untrained) checkpoint named by ABSOLUTE path, the real `rerank` on
    sampled drafts, the dodge set and a story set, then the confirmatory scorer
    over both files: it must find nothing to disagree with, GUARD 1 must hold on
    all twelve non-story draws, and the seeds must be the fresh ones.

    Mutation: `record` storing `shipped_index + 1` (modulo the pool) -- the
    scorer aborts with `ReplayMismatch` instead of scoring.
    """
    cfg = ChatConfig(d_model=32, n_layers=2, vocab_size=ChatTokenizer().vocab_size,
                     device="cpu", spread_tau=True)
    ckpt = tmp_path / "chat-tiny" / "ckpt_best.pt"
    ckpt.parent.mkdir()
    torch.save({"format": 1, "kind": "snnchat", "step": 0, "model": tiny_model.state_dict(),
                "optimizer": {}, "chat_config": cfg.to_dict(), "vocab_version": 1}, ckpt)

    outs = {}
    for name in ("dodge", "fresh"):
        outs[name] = tmp_path / f"v14_{name}_chat-tiny.json.gz"
        assert pools.main(["--ckpt", str(ckpt), "--set", name, "--seeds", "1", "--n", "6",
                           "--max-new", "32", "--device", "cpu", "--no-graph",
                           "--out", str(outs[name])]) == 0
    blob = pools.read_pool(outs["dodge"])
    assert blob["format"] == pools.FORMAT and blob["sampler_seeds"] == [6]
    assert blob["n_draws"] == 12 and len(blob["draws"][0]["pool"]) == 6
    assert blob["timing"]["n_draws"] == 12
    assert blob["timing"]["timing_pass_disagreements"] == 0
    assert len(blob["ckpt_sha256"]) == 64
    # The header says what was drawn AND what the REPL would have drawn, so a
    # pool at another length cannot be mistaken for one at the REPL's.
    assert (blob["max_new"], blob["repl_max_new"]) == (32, SamplingParams().max_new)
    assert "is not the REPL's" in capsys.readouterr().out

    # A wildcard is expanded by the scorer, sorted: PowerShell passes it through.
    assert scorer.expand([str(tmp_path / "v14_*_chat-tiny.json.gz")]) == sorted(outs.values())
    with pytest.raises(SystemExit, match="no pool file"):
        scorer.expand([str(tmp_path / "v15_*.json.gz")])
    scored = tmp_path / "scored.json"
    assert scorer.main([str(tmp_path / "v14_*_chat-tiny.json.gz"), "--out", str(scored)]) == 0
    result = json.loads(scored.read_text(encoding="utf-8"))
    assert result["n_story_draws"] == 20
    # What the REAL writer drew is inside the design the scorer holds a registered
    # checkpoint to: 32 of its 744 draws, none outside it. Mutation: `check_design`
    # keying `given` on `d.kind` instead of the set's name (32 outside).
    assert result["design"] == {
        "sampler_seeds": [6], "n_draws": 32, "registered_n_draws": 744, "missing": 712,
        "not_in_the_design": 0, "is_the_registered_design": False}
    g1 = result["GUARD1_identity"]
    assert (g1["verdict"], g1["n"], g1["identical"]) == ("pass", 12, 12)
    assert set(result["latency"]) == {p.name for p in outs.values()}
    assert "OVERALL" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# the pools describe the REPL, and the clause set
# ---------------------------------------------------------------------------


def test_a_pool_is_drawn_at_the_repl_defaults():
    """`python scripts/chat.py` with no flags is the configuration a confirmatory
    pool claims to describe. The first version of `v14_pools.py` drafted 300
    characters where the REPL drafts 400, which changes both tiers' membership.

    Mutation: `--max-new` defaulting to 300 again.
    """
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "chat_repl_for_v14", os.path.join(REPO, "scripts", "chat.py"))
    repl = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(repl)
    theirs = repl.build_parser().parse_args([])
    ours = pools.build_parser().parse_args([])

    assert ours.max_new == theirs.max_new == pools.REPL_MAX_NEW
    assert (ours.n, ours.lam) == (theirs.rerank, theirs.rerank_lambda)
    # What `main` does not expose it takes from the two dataclasses' defaults;
    # the REPL passes its own flags, so those must be the same numbers.
    sp, rp = SamplingParams(), RerankParams()
    assert (sp.temperature, sp.top_p, sp.top_k, sp.min_p, sp.max_new) == (
        theirs.temperature, theirs.top_p, theirs.top_k, theirs.min_p, theirs.max_new)
    assert (rp.min_chars, rp.temperature_spread, rp.steer_every, rp.echo) == (
        theirs.rerank_min_chars, theirs.rerank_spread, theirs.steer_every, theirs.rerank_echo)
    assert theirs.prime is False


def test_the_clause_set_is_half_cut_and_half_declined_and_never_bare():
    """The set exists because every other probe list is a bare "about a X", on
    which the subject extractor cannot be wrong. Mutation: `probes_for` giving
    the clause set kind "story", which would pool it into the fixed bars."""
    from snnchat.prime import prime_topic

    rows = pools.probes_for("clause")
    assert len(rows) == 12 and {kind for _, _, kind in rows} == {"story_clause"}
    assert "clause" in pools.SET_NAMES
    subjects = [tier_subject(prompt) for prompt, _, _ in rows]
    assert sum(s is not None for s in subjects) == 6 == sum(s is None for s in subjects)
    for (prompt, expect, _), subject in zip(rows, subjects):
        assert subject in (None, expect[0]), prompt
        # The last-word extractor is WRONG on every one: that is the set's point.
        assert prime_topic(prompt) not in (None, expect[0]), prompt


def test_a_recorded_pool_is_selected_on_the_subject_the_session_would_pass(tiny_model):
    """`record` must tier on `tier_subject`, not on the last word after "about":
    on "a penguin who loves fish" a tier built on "fish" picks the draft about a
    fish, and on "a penguin and make it funny" there is no subject at all, so
    the two recorded winners must be the same draft."""
    tok = ChatTokenizer()
    rp = RerankParams(n=3, lam=0.0, graph=False)
    fish = "Let me tell you a story. A fish swam in the sea. " + _FILL
    funny = "Let me tell you a story. It was a funny day. " + _FILL

    def recorded(prompt, off_topic):
        cands = _candidates(tok, [(FRAME, -110.0), (NAMES, -60.0), (off_topic, -20.0)])
        shipped = select(None, cands, rp, device="cpu", tok=tok,
                         echo_words=prompt_content_words(prompt))
        return pools.record(tiny_model, tok, prompt, rp, cands, shipped, device="cpu",
                            timing=False)

    row = recorded("tell me a story about a penguin who loves fish", fish)
    assert row["subject"] == "penguin"
    assert [c["subject_hit"] for c in row["pool"]] == [False, True, False]
    assert row["subject_index"] == 1

    row = recorded("tell me a story about a penguin and make it funny", funny)
    assert row["subject"] is None
    assert not any(c["subject_hit"] for c in row["pool"])
    assert row["subject_index"] == row["shipped_index"]


def test_clause_draws_are_scored_apart_from_the_pooled_bars(tmp_path):
    """A clause pool handed to the scorer beside a story pool gets its own
    subset, and `overall` and the `_all` subset do not read it."""
    prompt = "tell me a story about a penguin who loves fish"
    assert tier_subject(prompt) == "penguin"
    clause = {"kind": "story_clause", "expect": ["penguin"], "prompt": prompt, "seed": 6,
              "subject": "penguin", "shipped_index": 0, "subject_index": 1,
              "pool": [_cand(FRAME, -110.0, -100.0, prompt=prompt),
                       _cand(NAMES, -60.0, -100.0, prompt=prompt)]}
    a = _write(tmp_path, "v14_heldout_x.json.gz", _blob([_hand_draw()]))
    b = _write(tmp_path, "v14_clause_x.json.gz", _blob([clause], probe_set="clause"))
    out = tmp_path / "scored.json"
    assert scorer.main([str(a), str(b), "--out", str(out)]) == 0
    blob = json.loads(out.read_text(encoding="utf-8"))
    assert blob["n_story_draws"] == 1
    assert blob["subsets"]["lam0.6_all"]["n_draws"] == 1
    got = blob["subsets"]["lam0.6_clause"]
    assert (got["n_draws"], got["of"], got["draws_with_a_subject"]) == (1, 1, 1)
    assert got["P1_anchored_topic"]["subject_only"] == 1
    # The same run without the clause file says the same thing overall.
    alone = tmp_path / "alone.json"
    assert scorer.main([str(a), "--out", str(alone)]) == 0
    assert json.loads(alone.read_text(encoding="utf-8"))["overall"] == blob["overall"]
    assert "lam0.6_clause" not in json.loads(alone.read_text(encoding="utf-8"))["subsets"]

    # A stored subject that is the clause's LAST word is refused: the pool was
    # written by a selector that was told the wrong thing.
    wrong = dict(clause, subject="fish")
    c = _write(tmp_path, "v14_clause_y.json.gz", _blob([wrong], probe_set="clause"))
    with pytest.raises(scorer.ReplayMismatch, match="stored subject"):
        scorer.load_file(c, exploratory=False)


# ---------------------------------------------------------------------------
# statistics helpers
# ---------------------------------------------------------------------------


def test_mcnemar_and_the_verdicts_on_tiny_inputs():
    """Mutations: `<=` to `<` in `p1_verdict`; `p2_verdict` passing on the point
    estimate alone; `p3_verdict` ignoring `p`."""
    assert scorer.mcnemar(0, 0) == 1.0
    assert scorer.mcnemar(0, 3) == pytest.approx(0.25)
    assert scorer.mcnemar(1, 5) == pytest.approx(2 * (1 + 6) / 64)

    assert scorer.p1_verdict(0, 0, 10)["verdict"] == "pass"
    assert scorer.p1_verdict(1, 0, 10)["verdict"] == "fail"
    assert scorer.p1_verdict(0, 0, 0)["verdict"] == "unresolved"

    def pooled(delta, se):
        return {"n": 600, "delta": delta, "se_paired": se, "se_by_prompt": se / 2}

    lists = {name: {"n": 10, "delta": 0.01} for name in scorer.P2_LISTS}
    assert scorer.p2_verdict(pooled(0.08, 0.01), lists)["verdict"] == "pass"
    # A near-miss is a fail, however wide the interval.
    assert scorer.p2_verdict(pooled(0.0499, 0.5), lists)["verdict"] == "fail"
    # The point estimate clears the bar and the interval does not.
    assert scorer.p2_verdict(pooled(0.06, 0.01), lists)["verdict"] == "unresolved"
    negative = {**lists, "heldout": {"n": 10, "delta": -0.001}}
    assert scorer.p2_verdict(pooled(0.08, 0.01), negative)["verdict"] == "fail"
    absent = {k: v for k, v in lists.items() if k != "wide"}
    assert scorer.p2_verdict(pooled(0.08, 0.01), absent)["verdict"] == "unresolved"

    assert scorer.p3_verdict(20, 4, scorer.mcnemar(4, 20), 100)["verdict"] == "pass"
    assert scorer.p3_verdict(5, 1, scorer.mcnemar(1, 5), 100)["verdict"] == "unresolved"
    assert scorer.p3_verdict(4, 4, 1.0, 100)["verdict"] == "fail"
    assert scorer.p3_verdict(0, 0, 1.0, 0)["verdict"] == "unresolved"
    # No draw with a definite subject in BOTH picks: nothing was compared,
    # whatever the counts handed in say.
    assert scorer.p3_verdict(20, 0, 1e-6, 0)["verdict"] == "unresolved"


def test_sameness_helpers_on_tiny_inputs():
    """Mutations: letting bigrams run across replies (total becomes 5);
    `top_opening` using `min`."""
    d = scorer.distinct_2(["a b a b", "A b"])
    # a-b, b-a, a-b | a-b : 4 bigrams, 2 distinct, none across the boundary
    assert (d["distinct"], d["total"], d["value"]) == (2, 4, 0.5)
    assert scorer.distinct_2([])["value"] is None

    top = scorer.top_opening(["xx1", "xy2", "xx3"], chars=2)
    assert (top["opening"], top["k"], top["n"]) == ("xx", 2, 3)
    assert top["share"] == pytest.approx(2 / 3)
    assert scorer.top_opening(["b", "a"], chars=1)["opening"] == "b"     # tie: first met
    assert scorer.top_opening([])["share"] is None


def test_paired_delta_reports_the_cluster_standard_error():
    """Differences 1, 1, 3, 3 in two prompts: mean 2; SE over draws
    sd/sqrt(4) = 0.5774; SE over the two prompt means (1 and 3) is
    1.4142/sqrt(2) = 1. Mutation: clustering by draw instead of by prompt."""
    got = scorer.paired_delta([0.0] * 4, [1.0, 1.0, 3.0, 3.0], ["p", "p", "q", "q"])
    assert got["delta"] == 2.0 and got["n_prompts"] == 2
    assert got["se_paired"] == pytest.approx(0.57735, abs=1e-5)
    assert got["se_by_prompt"] == pytest.approx(1.0)
    assert scorer.paired_delta([], [], [])["delta"] is None


def test_tier_size_buckets_and_rate():
    """Mutation: the `5-16` bucket's upper edge moved to 17, so a tier of 17 is
    counted twice."""
    picks = [scorer.Pick(0, s, s == 1, False) for s in (1, 1, 3, 16, 17, 256)]
    got = scorer.tier_sizes(picks)
    assert got["buckets"] == {"1": 2, "2-4": 1, "5-16": 1, "17-64": 1, "65+": 1}
    assert (got["one_member"]["k"], got["one_member"]["n"]) == (2, 6)
    r = scorer.rate(6, 48)
    assert r["rate"] == 0.125 and r["ci"] == pytest.approx([0.0586, 0.2470], abs=5e-4)


# ---------------------------------------------------------------------------
# the pool file, and the seeds it may be drawn at
# ---------------------------------------------------------------------------


def test_pool_files_round_trip_through_gzip_and_are_a_function_of_their_content(tmp_path):
    """Mutations: dropping `mtime=0` (the gzip header's timestamp is no longer
    zero); `ensure_ascii=True`, which the round trip survives and the file size
    does not -- hence the check on the decompressed BYTES."""
    blob = _blob([_hand_draw()])
    blob["draws"][0]["pool"][0]["text"] = "café — \"quoted\"\nnew line"
    path = tmp_path / "deep" / "pool.json.gz"
    pools.write_pool(path, blob)
    raw = path.read_bytes()
    assert raw[:2] == b"\x1f\x8b"
    assert raw[4:8] == b"\x00\x00\x00\x00"              # gzip MTIME
    assert "café".encode() in gzip.decompress(raw)
    assert pools.read_pool(path) == blob
    assert len(raw) < len(json.dumps(blob))

    plain = tmp_path / "plain.json"
    plain.write_text(json.dumps(blob), encoding="utf-8")
    assert pools.read_pool(plain) == blob


def test_a_seed_offset_that_would_redraw_committed_pools_is_refused(tmp_path):
    """Sampler seeds 0-5 are the committed v12 pools'. Mutation: deleting the
    `offset < COMMITTED_SEEDS` test in `sampler_seeds`."""
    assert pools.sampler_seeds(6, 6) == [6, 7, 8, 9, 10, 11]
    for offset in (0, 5):
        with pytest.raises(ValueError, match="overlaps"):
            pools.sampler_seeds(6, offset)
    assert pools.sampler_seeds(6, 0, allow_committed=True) == [0, 1, 2, 3, 4, 5]
    with pytest.raises(ValueError):
        pools.sampler_seeds(0, 6)

    # And from the command line, BEFORE any checkpoint is opened: the path below
    # does not exist, so reaching the loader would be a different error.
    out = tmp_path / "never.json.gz"
    with pytest.raises(SystemExit, match="overlaps"):
        pools.main(["--ckpt", str(tmp_path / "no_such.pt"), "--seed-offset", "5",
                    "--device", "cpu", "--out", str(out)])
    assert not out.exists()
    assert pools.resolve(str(tmp_path / "a.pt")) == tmp_path / "a.pt"
    assert pools.resolve("experiments/x.pt") == pools.ROOT / "experiments" / "x.pt"


# ---------------------------------------------------------------------------
# the committed exploratory artifact
# ---------------------------------------------------------------------------


def test_the_committed_exploratory_artifact_is_what_the_scorer_says_about_a_committed_pool():
    """`v14_exploratory.json` feeds a pre-registration, so it may not drift.

    Re-reads ONE of the twelve committed pools (the whole dozen is 25 s of JSON
    parsing; this is 4) and requires the artifact's per-draw rows and check
    counters for that file to be what the scorer computes today, and the
    artifact's headline subset to be the draws those rows say it is. Changing
    `replay`, `echoes`, the word table or the instrument turns this red, and
    the fix is to re-run `score_v14.py --exploratory` and commit what it says.

    Mutations: `replay` never applying the subject rule; `V12_MIN_CHARS` moved
    to 200. NOT killed by `n_scored_rule` returning `n_chars` -- no recorded pick
    in this one file closed its own turn, so that rule is held by
    `test_exploratory_mode_recovers_n_scored_and_reproduces_the_recorded_picks`
    and by the 14 closed picks the full run checks, not here.
    """
    quality = os.path.join(REPO, "experiments", "chat", "_quality")
    with open(os.path.join(quality, "v14_exploratory.json"), encoding="utf-8") as f:
        artifact = json.load(f)
    assert artifact["mode"] == "exploratory"
    assert [i["path"] for i in artifact["inputs"]] == list(scorer.EXPLORATORY_FILES)

    name = "v12_heldout_chat-v3d-aligned.json"
    info, draws = scorer.load_file(scorer.Path(quality) / name, exploratory=True)
    (recorded,) = [i for i in artifact["inputs"] if i["path"] == name]
    assert info == recorded
    assert scorer.draw_rows(draws) == [r for r in artifact["rows"] if r["source"] == name]

    rows = artifact["rows"]
    assert len(rows) == artifact["n_story_draws"] == 2400
    head = artifact["subsets"]["lam0.6_null_measured"]
    assert head["n_draws"] == sum(1 for r in rows if r["null_whole_pool"])
    assert artifact["shipped_tier_all_draws"]["one_member"]["k"] == sum(
        1 for r in rows if r["lam0.6_shipped"][1] == 1)
    # Every check the exploratory recovery rests on came back clean.
    assert {k: v for k, v in artifact["checks"].items() if "mismatch" in k
            or k == "shipped_not_replayable"} == {
        "n_scored_mismatch": 0, "shipped_pick_mismatch": 0, "base_pick_mismatch": 0,
        "shipped_not_replayable": 0, "stored_echo_weight_mismatch": 0}
