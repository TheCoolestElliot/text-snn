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


def _blob(draws: list[dict], probe_set: str = "heldout", lam: float = 0.6) -> dict:
    return {"format": pools.FORMAT, "ckpt": "experiments/chat/none/ckpt_best.pt",
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
    assert (c["P1_topic"]["shipped_only"], c["P1_topic"]["subject_only"]) == (0, 1)
    assert c["P1_topic"]["verdict"] == "pass"
    assert c["P2_logp_per_char"]["pooled"]["delta"] == pytest.approx(-60 / 285 + 110 / 278)
    assert (c["P3_uer"]["fixed"], c["P3_uer"]["broken"]) == (1, 0)
    assert c["P3_uer"]["both_qualify"]["k"] == 1
    assert c["below_floor"]["shipped"]["k"] == 1 and c["below_floor"]["subject"]["k"] == 0
    assert c["tier_size"]["shipped"]["one_member"]["k"] == 1
    assert c["tier_size"]["subject"]["buckets"]["2-4"] == 1
    assert c["no_draft_names_subject"]["k"] == 0


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
    for prompt, spec in _POOLS.values():
        cands = _candidates(tok, spec)
        shipped = select(tiny_model, cands, rp, device="cpu", tok=tok,
                         echo_words=prompt_content_words(prompt))
        row = pools.record(tiny_model, tok, prompt, rp, cands, shipped, device="cpu")
        assert all(c["logp_null"] != 0.0 for c in row["pool"])
        assert all(c["n_scored"] == c["n_chars"] + 1 and c["closed"] for c in row["pool"])
        assert row["timing_pass_agrees"]
        subject = tier_subject(prompt)
        assert row["subject"] == subject
        full = {"kind": "story" if subject else "list", "prompt": prompt, "seed": 6,
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

    # A wildcard is expanded by the scorer, sorted: PowerShell passes it through.
    assert scorer.expand([str(tmp_path / "v14_*_chat-tiny.json.gz")]) == sorted(outs.values())
    with pytest.raises(SystemExit, match="no pool file"):
        scorer.expand([str(tmp_path / "v15_*.json.gz")])
    scored = tmp_path / "scored.json"
    assert scorer.main([str(tmp_path / "v14_*_chat-tiny.json.gz"), "--out", str(scored)]) == 0
    result = json.loads(scored.read_text(encoding="utf-8"))
    assert result["n_story_draws"] == 20
    g1 = result["GUARD1_identity"]
    assert (g1["verdict"], g1["n"], g1["identical"]) == ("pass", 12, 12)
    assert set(result["latency"]) == {p.name for p in outs.values()}
    assert "OVERALL" in capsys.readouterr().out


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
