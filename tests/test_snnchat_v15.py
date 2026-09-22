"""The v15 recall round's instruments: both probes, the scorer and the driver.

CPU only, no corpus, no checkpoint, no GPU query: `git` and `nvidia-smi` are
replaced by stand-ins and every child process by a fake that writes the artifact
its command line names. What is held up here is the PRE-REGISTRATION -- that the
gating stratum's phrasings really are absent from the generator, that a value is
in or out of the trained lists as its stratum claims, that the bars sit where
`PREDICTION_v15.md` put them (15/80 fails, 16/80 passes), that a straddle is
`unresolved`, and that the driver will not start on an uncommitted scorer or a
busy GPU.

Every test here was run against a deliberately broken implementation and seen
to fail; the mutation each one caught is in the chat-v15 B2 hand-off.

Not part of the research protocol.
"""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import pathlib
import shutil
import subprocess
import sys
import types

import pytest
import torch

from snnchat import recall
from snnchat.model import ChatConfig, build_chat_model
from snnchat.rerank import RerankParams
from snnchat.tokenizer import ChatTokenizer

ROOT = pathlib.Path(__file__).resolve().parents[1]

#: `git rev-parse 4446951:scripts/chat/memory_probe.py` -- the commit the v15
#: branch was cut from. The committed floor was measured with these bytes.
MEMORY_PROBE_BLOB = "a524c3d7eceeaf40b7862cc1f5d1d33ec84eb560"


def _load_script(name: str):
    path = ROOT / "scripts" / "chat" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"_{name}_v15_under_test", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def mp1():
    return _load_script("memory_probe")


@pytest.fixture(scope="module")
def mp2():
    return _load_script("memory_probe_v2")


@pytest.fixture(scope="module")
def binding():
    return _load_script("binding_probe")


@pytest.fixture(scope="module")
def score():
    return _load_script("score_v15")


@pytest.fixture(scope="module")
def driver():
    return _load_script("session15_driver")


@pytest.fixture(scope="module")
def tiny_model():
    torch.manual_seed(0)
    cfg = ChatConfig(d_model=32, n_layers=2, vocab_size=ChatTokenizer().vocab_size,
                     device="cpu", spread_tau=True, seed=0)
    model = build_chat_model(cfg)
    model.eval()
    return model


# --------------------------------------------------------------------------
# the committed probe is the committed probe
# --------------------------------------------------------------------------


def test_the_committed_memory_probe_is_byte_unchanged():
    data = (ROOT / "scripts" / "chat" / "memory_probe.py").read_bytes()
    blob = hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()
    assert blob == MEMORY_PROBE_BLOB


def test_the_statistics_are_the_committed_probes_own(mp1, mp2, score):
    # Imported, not re-implemented: same function objects' behaviour, to the bit.
    for b in range(0, 9):
        for c in range(0, 9):
            assert mp2.mcnemar(b, c) == mp1.mcnemar(b, c) == score.mcnemar(b, c)
    assert mp2.wilson(0, 80) == mp1.wilson(0, 80)
    assert mp2.hit("Tom likes tomatoes", ("tomato",)) == mp1.hit("x tomato", ("tom",)) == 0.0
    # The arithmetic the RESOLVED tier's "6/80 vs 0" rests on.
    assert score.mcnemar(0, 6) == 0.03125 < score.ALPHA
    assert score.mcnemar(0, 5) == 0.0625 > score.ALPHA


# --------------------------------------------------------------------------
# items: phrasing and value membership, stratum by stratum
# --------------------------------------------------------------------------


def _user_turns_of(mp2, stratum):
    out = set()
    for item in mp2.ITEM_SETS[stratum]:
        for cond in ("told", "swapped" if stratum == "S4" else "untold"):
            for dist in mp2.DISTANCES[stratum]:
                turns, _ = mp2.user_turns(item, cond, dist)
                out |= set(turns)
    return out


def test_gating_phrasings_are_absent_from_the_generators_templates(mp1, mp2):
    assert len(mp2.S1_ITEMS) == 10
    for item in mp2.S1_ITEMS:
        assert item.statement in recall.HELDOUT_STATEMENTS[item.slot]
        assert item.question in recall.HELDOUT_QUESTIONS[item.slot]
        assert item.statement not in recall.TRAIN_STATEMENTS[item.slot]
        assert item.question not in recall.TRAIN_QUESTIONS[item.slot]

    # By a route that does not go through the tables: nothing S1 or S5 sends
    # is a user turn the generator ever writes, whole or as an " and " half.
    emitted = set()
    for d in recall.build_recall_dialogues(20_000, seed=0):
        for role, text in d.turns:
            if role == "user":
                emitted |= {text, *text.split(" and ")}
    fillers = set(mp1.FILLERS)
    for stratum in ("S1", "S5"):
        sent = _user_turns_of(mp2, stratum) - fillers
        assert sent and not (sent & emitted), sorted(sent & emitted)
        assert sent <= recall.heldout_phrases() | {
            recall.fill(i.statement, i.value, i.aux) for i in mp2.S5_ITEMS}


def test_reference_strata_use_trained_phrasings(mp2):
    for stratum in ("S2", "S3"):
        for item in mp2.ITEM_SETS[stratum]:
            assert item.statement in recall.TRAIN_STATEMENTS[item.slot]
            assert item.question in recall.TRAIN_QUESTIONS[item.slot]
    for two in mp2.S4_ITEMS:
        for fact in (two.asked, two.other):
            assert fact.statement in recall.TRAIN_STATEMENTS[fact.slot]
        assert two.asked.question in recall.TRAIN_QUESTIONS[two.asked.slot]


def test_values_are_in_or_out_of_list_as_each_stratum_claims(mp1, mp2):
    committed = {t for _e, _q, targets in mp1.ITEMS for t in targets}
    for stratum in ("S1", "S2"):
        for item in mp2.ITEM_SETS[stratum]:
            assert item.value in recall.VALUES[item.slot]
            assert item.value not in committed          # a NEW item set
            assert item.aux is None or item.aux in recall.AUX[item.slot]
    for item in mp2.S3_ITEMS:
        assert item.value in recall.HELDOUT_VALUES[item.slot]
        assert item.value not in recall.VALUES[item.slot]
        assert item.aux is None or item.aux in recall.AUX[item.slot]   # only the VALUE is unseen
    # S1 against S2 is one factor: same facts, different wording.
    def facts(items):
        return [(i.slot, i.value, i.aux) for i in items]

    assert facts(mp2.S1_ITEMS) == facts(mp2.S2_ITEMS)
    assert [(i.slot, i.aux) for i in mp2.S3_ITEMS] == [(i.slot, i.aux) for i in mp2.S2_ITEMS]
    for two in mp2.S4_ITEMS:
        for fact in (two.asked, two.other):
            assert fact.value in recall.VALUES[fact.slot]
        assert {two.asked.slot, two.other.slot} in ({"name", "pet"}, {"colour", "object"})
        assert two.asked.value != two.other.value
    assert sum(t.asked_first for t in mp2.S4_ITEMS) == 5
    markers = [m for _s, _q, ms in recall.HELDOUT_FRAMES for m in ms]
    for item in mp2.S5_ITEMS:
        assert item.slot is None and item.value not in committed
        assert any(m in recall.fill(item.statement, item.value) for m in markers)

    # The /80 lattice every bar is set on, and seeds the committed probe never drew.
    assert all(len(v) == 10 for v in mp2.ITEM_SETS.values()) and mp2.SEEDS == 8
    assert mp2.SEED_BASE >= 8
    assert mp2.DISTANCES == {"S1": (0,), "S2": (0, 1, 2), "S3": (0,), "S4": (0,), "S5": (0,)}
    assert "elliot" in recall.VALUES["name"]
    assert [s for s, _a in mp2.COMMITTED_SLOTS].count(None) == 2


def test_the_control_is_the_told_conversation_minus_its_evidence(mp1, mp2):
    for stratum in ("S1", "S2", "S3", "S5"):
        for item in mp2.ITEM_SETS[stratum]:
            for dist in mp2.DISTANCES[stratum]:
                told, est = mp2.user_turns(item, "told", dist)
                untold, none = mp2.user_turns(item, "untold", dist)
                assert est == [0] and none == []
                assert told[1:] == untold and len(told) == dist + 2
                assert told[1:-1] == list(mp1.FILLERS[:dist])
                assert item.value in told[0] and item.value not in " ".join(untold)


def test_the_swapped_roles_control_exchanges_values_and_nothing_else(mp2):
    for two in mp2.S4_ITEMS:
        told, est = mp2.user_turns(two, "told")
        swapped, est2 = mp2.user_turns(two, "swapped")
        assert est == est2 == [0, 1] and told[-1] == swapped[-1]
        a, o = two.asked, two.other
        asked_at = 0 if two.asked_first else 1
        assert told[asked_at] == recall.fill(a.statement, a.value, a.aux)
        assert swapped[asked_at] == recall.fill(a.statement, o.value, a.aux)
        assert swapped[1 - asked_at] == recall.fill(o.statement, a.value, o.aux)
        # The target word is in context in BOTH conversations.
        assert a.value in " ".join(told) and a.value in " ".join(swapped)
    with pytest.raises(ValueError):
        mp2.user_turns(mp2.S4_ITEMS[0], "untold")
    with pytest.raises(ValueError):
        mp2.user_turns(mp2.S1_ITEMS[0], "swapped")


# --------------------------------------------------------------------------
# scoring categories, on hand-written replies
# --------------------------------------------------------------------------


@pytest.mark.parametrize("reply, slot, targets, others, category, flags", [
    ("Your name is Alice.", "name", ("alice",), (), "hit", {"hit", "any_value"}),
    ("Your name is Tom.", "name", ("alice",), (), "wrong_value", {"wrong_value", "any_value"}),
    # whole words only: 'tom' is not in 'tomato', 'ben' is not in 'bench'
    ("I like tomatoes on a bench.", "name", ("tom",), (), "other", set()),
    ("Your name is Tomas.", "name", ("tom",), (), "other", set()),
    ("You haven't told me your name yet.", "name", ("alice",), (), "refusal", {"refusal"}),
    ("I don't know your name. You haven't told me.", "name", ("alice",), (), "refusal",
     {"refusal"}),
    ("I don\u2019t know what your dog is called.", "pet", ("buddy",), (), "refusal", {"refusal"}),
    ("I don't really have a name. I'm a spiking neural network.", "name", ("alice",), (),
     "persona_capture", {"persona_capture"}),
    ("No name - just a spiking char-LM. You can call me whatever you like.", "name",
     ("alice",), (), "persona_capture", {"persona_capture"}),
    # a held-out value counts as a value of the slot: it is a confabulated name
    ("Your name is Oscar.", "name", ("alice",), (), "wrong_value", {"wrong_value", "any_value"}),
    # two-fact: the OTHER told value is a swap, and is not double-counted as wrong
    ("Your favourite colour is black.", "colour", ("pink",), ("black",), "swap",
     {"swap", "any_value"}),
    ("It is pink, and your car is black.", "colour", ("pink",), ("black",), "hit",
     {"hit", "swap", "any_value"}),
    ("Your name is Tom. Or Alice.", "name", ("alice",), (), "hit",
     {"hit", "wrong_value", "any_value"}),
    # a held-out frame has no vocabulary to be wrong in
    ("You live in Tom.", None, ("milton",), (), "other", set()),
    ("You are 9 years old.", None, ("nine", "9"), (), "hit", {"hit"}),
    ("Once upon a time there was a little girl.", "food", ("soup",), (), "other", set()),
])
def test_reply_categories(mp2, reply, slot, targets, others, category, flags):
    got = mp2.classify(reply, slot, targets, other_values=others)
    assert got["category"] == category
    assert {f for f in mp2.FLAGS if got[f]} == flags


def test_the_echo_off_winner_ignores_the_echo_tier(mp2):
    rp = RerankParams(n=4, lam=0.6)
    def C(score, n_chars, w):
        return types.SimpleNamespace(score=score, n_chars=n_chars, echo_weight=w)

    short_best = C(9.0, 3, 0.0)
    echoing = C(1.0, 40, 5.0)
    plain_best = C(2.0, 40, 0.0)
    assert mp2.echo_off_winner([short_best, echoing, plain_best], rp) is plain_best
    assert mp2.echo_off_winner([short_best], rp) is short_best      # all short: ranked anyway


def _row(told_hit, control_hit, *, ack_on=False, ack_off=False, **told_extra):
    flags = dict.fromkeys(("hit", "swap", "wrong_value", "refusal", "persona_capture",
                           "any_value"), False)
    told = {**flags, "hit": told_hit, "category": "hit" if told_hit else "other", **told_extra}
    control = {**flags, "hit": control_hit, "category": "hit" if control_hit else "other"}
    return {"control_kind": "untold", "told_flags": told, "control_flags": control,
            "acks": [{"on_hit": ack_on, "off_hit": ack_off}]}


def test_summarise_counts_pairs_and_the_acknowledgement_column(mp2):
    rows = ([_row(True, False, ack_on=True)] * 6 + [_row(True, True, ack_on=True, ack_off=True)]
            + [_row(False, True)] * 2 + [_row(False, False, wrong_value=True)] * 71)
    s = mp2.summarise(rows)
    assert s["n"] == 80
    assert (s["told"]["hit"]["k"], s["control"]["hit"]["k"]) == (7, 3)
    assert s["discordant"] == {"control_only": 2, "told_only": 6}
    assert s["mcnemar_p"] == mp2.mcnemar(2, 6)
    assert s["told"]["wrong_value"]["k"] == 71 and s["told"]["category"] == {"hit": 7, "other": 73}
    lo, hi = mp2.wilson(7, 80)
    assert s["told"]["hit"]["ci"] == [round(lo, 4), round(hi, 4)]
    assert s["ack_echo"]["echo_on"]["k"] == 7 and s["ack_echo"]["echo_off"]["k"] == 1
    assert s["ack_echo"]["discordant"] == {"on_only": 6, "off_only": 0}
    assert s["told_hit_by_ack"]["ack_carried_value"]["k"] == 7
    assert s["told_hit_by_ack"]["ack_did_not"] == {"k": 0, "n": 73, "rate": 0.0,
                                                   "ci": [0.0, round(mp2.wilson(0, 73)[1], 4)]}


def test_the_probe_runs_on_cpu_and_pairs_every_row(mp2, tiny_model):
    tok = ChatTokenizer()
    seeds = [mp2.SEED_BASE, mp2.SEED_BASE + 1]
    rp = RerankParams(n=2, lam=0.6, graph=False)
    rows = mp2.run_stratum(tiny_model, tok, "cpu", "S2", seeds=seeds, rp=rp, max_new=6,
                           distances=(0, 1), items=mp2.S2_ITEMS[:2])
    assert len(rows) == 2 * 2 * 2
    assert [r["seed"] for r in rows[:2]] == seeds
    for r in rows:
        assert r["control_kind"] == "untold" and r["control_turns"] == r["told_turns"][1:]
        assert len(r["told_turns"]) == r["distance"] + 2
        assert len(r["acks"]) == 1 and set(r["acks"][0]) == {"on", "off", "on_hit", "off_hit"}
        assert set(r["told_flags"]) == set(mp2.FLAGS) | {"category"}
    # Same seed, same conversation -> same reply: the pairing is on the sampler seed.
    again = mp2.run_stratum(tiny_model, tok, "cpu", "S2", seeds=seeds[:1], rp=rp, max_new=6,
                            distances=(0,), items=mp2.S2_ITEMS[:1])
    assert again[0]["told"] == rows[0]["told"] and again[0]["control"] == rows[0]["control"]

    two = mp2.run_stratum(tiny_model, tok, "cpu", "S4", seeds=seeds[:1], rp=rp, max_new=6,
                          items=mp2.S4_ITEMS[:1])
    assert two[0]["control_kind"] == "swapped" and len(two[0]["acks"]) == 2
    assert two[0]["other_values"] == ["rocky"]
    assert mp2.summarise(two)["n"] == 1


def test_each_arms_flags_come_from_that_arms_reply_at_the_same_sampler_seed(mp2, monkeypatch):
    """The one assignment that decides the DIRECTION of every McNemar test, and
    the one argument that makes the two arms a pair. A parrot stands in for the
    model: it replies with exactly what it was sent, so the told arm names the
    value, the untold arm cannot, and nothing about the real sampler is needed
    to see which reply each flag was computed from."""
    calls = []

    def parrot(model, tok, device, params, rp, turns, establishing, targets):
        calls.append((params.seed, tuple(turns), tuple(establishing)))
        return {"reply": " / ".join(turns), "acks": [{"on": "", "off": "", "on_hit": False,
                                                      "off_hit": False}] * len(establishing)}

    monkeypatch.setattr(mp2, "converse", parrot)
    seeds = [mp2.SEED_BASE, mp2.SEED_BASE + 1, mp2.SEED_BASE + 2]
    rows = mp2.run_stratum(None, None, "cpu", "S1", seeds=seeds, rp=None, distances=(0, 2))
    assert len(rows) == 2 * len(mp2.S1_ITEMS) * len(seeds) and len(calls) == 2 * len(rows)
    for i, r in enumerate(rows):
        told_call, control_call = calls[2 * i], calls[2 * i + 1]
        assert told_call[0] == control_call[0] == r["seed"]          # paired on the seed
        assert told_call[1] == tuple(r["told_turns"]) and told_call[2] == (0,)
        assert control_call[1] == tuple(r["control_turns"]) and control_call[2] == ()
        assert r["told"] == " / ".join(r["told_turns"])
        assert r["control"] == " / ".join(r["control_turns"])
        assert r["told_flags"]["hit"] and not r["control_flags"]["hit"], r
    s = mp2.summarise([r for r in rows if r["distance"] == 0])
    assert s["discordant"] == {"control_only": 0, "told_only": s["n"]}
    assert s["told"]["hit"]["k"] == s["n"] and s["control"]["hit"]["k"] == 0

    # S4's control holds the same words in exchanged roles, so a hit cannot tell
    # the arms apart there; the reply text and the recomputed flags can.
    calls.clear()
    for r in mp2.run_stratum(None, None, "cpu", "S4", seeds=seeds[:1], rp=None):
        assert r["told"] == " / ".join(r["told_turns"]) != r["control"]
        assert r["control"] == " / ".join(r["control_turns"])
        for arm in ("told", "control"):
            assert r[arm + "_flags"] == mp2.classify(r[arm], r["slot"], tuple(r["targets"]),
                                                     other_values=tuple(r["other_values"]))
    assert {c[0] for c in calls} == {seeds[0]}


def test_the_shipped_decoder_pass_really_uses_the_shipped_n(mp2, monkeypatch, tmp_path):
    """`--shipped-decoder` is the only thing separating the n=256 artifact from
    the plain one, and the scorer files whatever it is handed under `n256`."""
    seen = {}

    def fake_run(model, tok, device, stratum, *, seeds, rp, max_new, distances, items):
        seen[stratum] = (rp.n, tuple(distances))
        return []

    monkeypatch.setattr(mp2, "run_stratum", fake_run)
    monkeypatch.setattr(mp2, "summarise", lambda rows: {"n": 0})
    monkeypatch.setattr(mp2, "_print", lambda *a: None)
    monkeypatch.setattr(mp2, "load_chat_checkpoint",
                        lambda path, device: (torch.nn.Identity(), None, None))
    for flag, n in ((["--shipped-decoder"], mp2.SHIPPED_N), ([], mp2.PLAIN_N)):
        out = tmp_path / f"n{n}.json"
        assert mp2.main(["--ckpt", str(tmp_path / "x.pt"), "--device", "cpu",
                         "--out", str(out), "--strata", "S1", "S2", *flag]) == 0
        art = json.loads(out.read_text(encoding="utf-8"))
        assert art["decoder"]["n"] == n == seen["S1"][0]
        assert art["decoder"]["shipped_decoder"] is bool(flag)
        assert seen["S2"][1] == ((0,) if flag else mp2.DISTANCES["S2"])
    assert (mp2.SHIPPED_N, mp2.PLAIN_N) == (256, 8)


# --------------------------------------------------------------------------
# the binding probe
# --------------------------------------------------------------------------


def test_binding_items_reproduce_the_committed_turns(mp1, binding):
    items = binding.committed_items()
    trained = [it for it, (slot, _a) in zip(mp1.ITEMS, binding.COMMITTED_SLOTS) if slot]
    assert len(items) == len(trained) == 8
    for it, (establish, question, targets) in zip(items, trained):
        assert recall.fill(it["statement"], it["value"], it["aux"]) == establish
        assert it["question"] == question and it["value"] == targets[0]
    assert [i["group"] for i in binding.probe_items()].count("S1") == 10
    assert len(binding.probe_items()) == 38


def test_binding_contexts_differ_only_in_what_was_told(binding):
    tok = ChatTokenizer()
    for item in binding.probe_items():
        true = item["value"]
        told = tok.decode(binding.context_ids(tok, item, true))
        untold = tok.decode(binding.context_ids(tok, item, None))
        lead, shown = binding.answer_split(item["slot"], true, item["aux"])
        assert lead + shown + "." == recall.expected_answer(item["slot"], true, item["aux"])
        assert told.endswith("<|bot|>" + lead) and untold.endswith("<|bot|>" + lead)
        assert true in told and binding.ACK in told
        assert true not in untold and binding.ACK not in untold
        swaps = binding.swap_values(item["slot"], true)
        assert len(set(swaps)) == binding.N_SWAPS and true not in swaps
        assert set(swaps) <= set(recall.VALUES[item["slot"]])
        for other in swaps:
            ctx = tok.decode(binding.context_ids(tok, item, other))
            assert other in ctx and true not in ctx and ctx.endswith("<|bot|>" + lead)


def test_binding_differences_have_the_documented_sign(binding, monkeypatch):
    tok = ChatTokenizer()
    item = binding.probe_items()[0]
    true = item["value"]

    def fake(model, prefix_ids, sequences, device):
        assert tok.decode(sequences[0]) == recall.display(item["slot"], true)
        ctx = tok.decode(prefix_ids)
        nats = 1.0 if true in ctx else (2.0 if binding.ACK not in ctx else 3.0)
        return [-nats * len(sequences[0])]

    monkeypatch.setattr(binding, "score_under_prefix", fake)
    r = binding.score_item(None, tok, "cpu", item)
    ln2 = 0.6931471805599453
    assert r["bpc_told"] == pytest.approx(1 / ln2) and r["bpc_untold"] == pytest.approx(2 / ln2)
    assert r["bpc_swapped"] == pytest.approx(3 / ln2)
    assert r["told_minus_untold"] == pytest.approx(-1 / ln2)
    assert r["told_minus_swapped"] == pytest.approx(-2 / ln2)
    p = binding.pooled([r, {**r, "value_chars": 3 * r["value_chars"], "bpc_told": 0.0,
                            "told_minus_swapped": 1.0}])
    assert p["bpc_told"] == pytest.approx(r["bpc_told"] / 4)      # character-weighted
    assert p["items_told_below_swapped"] == 1


def test_the_binding_probe_scores_a_real_model_on_cpu(binding, tiny_model):
    r = binding.score_item(tiny_model, ChatTokenizer(), "cpu", binding.probe_items()[2])
    assert r["value_chars"] == len("Buddy") and len(r["bpc_swapped_each"]) == binding.N_SWAPS
    assert all(0.0 < r[k] < 20.0 for k in ("bpc_told", "bpc_untold", "bpc_swapped"))


# --------------------------------------------------------------------------
# score_v15: tiers, guards, the lattice, the straddle
# --------------------------------------------------------------------------


def _cell(told=20, control=0, *, told_only=None, control_only=0, wrong=2, swap=0,
          any_untold=1, refusal_untold=40):
    flags = ("hit", "swap", "wrong_value", "refusal", "persona_capture", "any_value")

    def rate(k):
        return {"k": k, "n": 80, "rate": k / 80, "ci": [0.0, 1.0]}

    t = {f: rate(0) for f in flags}
    c = {f: rate(0) for f in flags}
    t.update(hit=rate(told), wrong_value=rate(wrong), swap=rate(swap), category={"hit": told})
    c.update(hit=rate(control), any_value=rate(any_untold), refusal=rate(refusal_untold),
             category={"hit": control})
    return {"n": 80, "control_kind": "untold", "told": t, "control": c,
            "discordant": {"told_only": told if told_only is None else told_only,
                           "control_only": control_only},
            "mcnemar_p": 0.0, "ack_echo": {}, "told_hit_by_ack": {}}


def _artifacts(score, run, **over):
    """A seed that clears everything; `over` replaces one piece at a time."""
    ref = score.INCUMBENT_VAL[score.REFERENCE[run]]
    # One dict PER pick. `[{...}] * 20` is twenty names for one dict, and a test
    # that flips "one" pick then flips them all -- which is how the social and
    # identity bars first escaped a mutation.
    picks = ([{"kind": "social", "hit": 1.0, "text": "Hello there, friend!"} for _ in range(20)]
             + [{"kind": "identity", "hit": 1.0, "text": "I'm a small model."} for _ in range(15)]
             + [{"kind": "identity", "hit": 0.0, "text": "Hmm, let me think."}]
             + [{"kind": "topic", "hit": 0.0, "text": "Once upon a time there was a cat."}
                for _ in range(64)])
    art = {
        "committed": {"summary": {"0": {"n": 80, "told": [16, 80], "untold": [4, 80],
                                        "discordant": [0, 12]}},
                      "rows": [{"distance": 0, "targets": ["elliot"], "told_hit": 1.0}] * 8},
        "v2": {"complete": True, "decoder": {"n": 8}, "summary": {
            "S1": {"0": _cell(told=8)}, "S4": {"0": _cell(told=30, control=5, swap=3)},
            "S2": {"0": _cell(told=40), "1": _cell(told=30), "2": _cell(told=25)},
            "S3": {"0": _cell(told=1)}, "S5": {"0": _cell(told=0)}}},
        "v2_shipped": None, "binding": None,
        "battery": {"report": {"baseline_picks": picks}},
        "heldout": {"n": 256, "rows": [{"echo_hit": 1.0}] * 60 + [{"echo_hit": 0.0}] * 60},
        "summary": {"steps": 14000, "val": {**ref, "recall": 0.2, "weighted": 1.1}},
        "n_divergences": 0,
    }
    art.update(over)
    return art


def _with_s1(score, run, **cell):
    art = _artifacts(score, run)
    art["v2"]["summary"]["S1"]["0"] = _cell(**cell)
    return art


def test_a_seed_that_clears_everything_is_a_ship_candidate(score):
    run = score.RUNS[0]
    got = score.score_seed(run, _artifacts(score, run))
    assert got["verdict"] == "ship-candidate"
    assert all(g["status"] == "pass" for g in {**got["ship_bars"], **got["guards"]}.values())
    assert set(got["guards"]) == {"wrong_value", "confabulation", "swap", "bpc_soda",
                                  "bpc_stories_topic", "bpc_tinystories", "heldout",
                                  "social", "identity"}
    assert got["reported"]["committed_item1_elliot_seen_value"].startswith("1.0000 (8/8)")
    assert set(got["reported"]["strata"]) == {"S1_d0", "S2_d0", "S2_d1", "S2_d2", "S3_d0",
                                              "S4_d0", "S5_d0"}


def test_the_bars_sit_between_lattice_points(score):
    run = score.RUNS[0]

    def art(told, untold):
        return _artifacts(score, run, committed={
            "summary": {"0": {"n": 80, "told": [told, 80], "untold": [untold, 80],
                              "discordant": [0, told]}}})

    assert score.score_seed(run, art(16, 4))["verdict"] == "ship-candidate"
    near = score.score_seed(run, art(15, 4))
    assert near["verdict"] == "resolved"                       # a near-miss is a miss
    assert near["ship_bars"]["i_committed_probe"]["status"] == "FAIL"
    assert score.score_seed(run, art(16, 5))["verdict"] == "resolved"
    # (ii): 7/80 fails, 8/80 passes -- and 7 vs 0 is still RESOLVED.
    seven = score.score_seed(run, _with_s1(score, run, told=7))
    assert seven["verdict"] == "resolved"
    assert seven["ship_bars"]["ii_unseen_phrasing"]["status"] == "FAIL"
    assert score.score_seed(run, _with_s1(score, run, told=8))["verdict"] == "ship-candidate"
    for thr in (score.COMMITTED_TOLD_MIN, score.COMMITTED_UNTOLD_MAX, score.S1_TOLD_MIN,
                score.SOCIAL_MIN[0], score.IDENTITY_MIN[0]):
        assert thr % 1 == 0.5


def test_resolved_needs_significance_in_the_told_direction(score):
    run = score.RUNS[0]
    assert score.score_seed(run, _with_s1(score, run, told=6))["verdict"] == "resolved"
    assert score.score_seed(run, _with_s1(score, run, told=5))["verdict"] == "floor"
    # p < 0.05 with the CONTROL ahead is not "resolved".
    lost = score.score_seed(run, _with_s1(score, run, told=1, control=12, told_only=0,
                                          control_only=11))
    assert lost["verdict"] == "floor" and lost["primary_S1"]["verdict"] == "below"
    # hits that are all concordant establish nothing
    tied = score.score_seed(run, _with_s1(score, run, told=30, control=30, told_only=2,
                                          control_only=2))
    assert tied["verdict"] == "floor"


def _fails_only(score, run, art, guard):
    got = score.score_seed(run, art)
    bad = {k for k, g in {**got["ship_bars"], **got["guards"]}.items() if g["status"] != "pass"}
    assert bad == {guard}, bad
    assert got["verdict"] == "resolved"


def test_each_guard_alone_stops_a_ship_candidate(score):
    run = score.RUNS[1]
    _fails_only(score, run, _with_s1(score, run, told=8, wrong=8), "wrong_value")
    assert score.score_seed(run, _with_s1(score, run, told=8, wrong=7))["verdict"] \
        == "ship-candidate"
    _fails_only(score, run, _with_s1(score, run, told=8, any_untold=41), "confabulation")
    assert score.score_seed(run, _with_s1(score, run, told=8, any_untold=40))["verdict"] \
        == "ship-candidate"

    art = _artifacts(score, run)
    art["v2"]["summary"]["S4"]["0"] = _cell(told=30, control=5, swap=30)
    _fails_only(score, run, art, "swap")
    art = _artifacts(score, run)
    art["v2"]["summary"]["S4"]["0"] = _cell(told=30, control=28, told_only=4, control_only=2)
    _fails_only(score, run, art, "iii_two_fact")

    ref = score.INCUMBENT_VAL[score.REFERENCE[run]]
    for source in score.GUARDED_SOURCES:
        band = score.BPC_BAND[source]
        worse = _artifacts(score, run)
        worse["summary"]["val"][source] = ref[source] + band + 1e-6
        _fails_only(score, run, worse, f"bpc_{source}")
        edge = _artifacts(score, run)
        edge["summary"]["val"][source] = ref[source] + band - 1e-6
        assert score.score_seed(run, edge)["verdict"] == "ship-candidate"
        better = _artifacts(score, run)
        better["summary"]["val"][source] = ref[source] - 10 * band
        got = score.score_seed(run, better)
        assert got["verdict"] == "ship-candidate"
        assert got["guards"][f"bpc_{source}"]["side"] == "better"
    # alpaca is the donor: it may rise without failing anything.
    donor = _artifacts(score, run)
    donor["summary"]["val"]["alpaca"] += 0.5
    assert score.score_seed(run, donor)["verdict"] == "ship-candidate"

    low = [{"echo_hit": 1.0}] * 40 + [{"echo_hit": 0.0}] * 80
    _fails_only(score, run, _artifacts(score, run, heldout={"n": 256, "rows": low}), "heldout")
    # 50/120 is below 60 but its interval reaches 0.5: NOT resolved below, so it passes.
    mid = [{"echo_hit": 1.0}] * 50 + [{"echo_hit": 0.0}] * 70
    assert score.score_seed(run, _artifacts(score, run, heldout={"n": 256, "rows": mid}))[
        "verdict"] == "ship-candidate"

    for kind, guard in (("social", "social"), ("identity", "identity")):
        art = _artifacts(score, run)
        picks = copy.deepcopy(art["battery"]["report"]["baseline_picks"])
        flip = [p for p in picks if p["kind"] == kind and p["hit"] >= 1.0][0]
        flip["hit"] = 0.0
        art["battery"]["report"]["baseline_picks"] = picks
        assert sum(p["hit"] for p in picks if p["kind"] == kind) == {"social": 19,
                                                                     "identity": 14}[kind]
        _fails_only(score, run, art, guard)      # 19/20, and 14/16


def test_a_missing_or_partial_artifact_is_never_a_pass(score):
    run = score.RUNS[0]
    assert score.score_seed(run, _artifacts(score, run, v2=None))["verdict"] == "not run"
    partial = _artifacts(score, run)
    partial["v2"]["complete"] = False
    assert score.score_seed(run, partial)["verdict"] == "not run"
    wrong_decoder = _artifacts(score, run)
    wrong_decoder["v2"]["decoder"]["n"] = 256
    assert score.score_seed(run, wrong_decoder)["verdict"] == "not run"
    short = _artifacts(score, run)
    short["summary"]["steps"] = 9000
    assert score.score_seed(run, short)["verdict"] == "not run"
    for key in ("battery", "heldout", "committed"):
        got = score.score_seed(run, _artifacts(score, run, **{key: None}))
        assert got["verdict"] == "resolved", key
    rolled = score.score_seed(run, _artifacts(score, run, n_divergences=1))
    assert rolled["rolled_back"] and rolled["verdict"] == "ship-candidate"   # reported, not replaced


def test_s1_is_reported_per_item_and_without_its_near_trained_items(score, mp2):
    """The S1 bar is 8/80 and an item is 8 cells. Three S1 phrasings are a word
    or two from a trained template; the bar stays where it was pre-registered
    and the report says which items cleared it."""
    run = score.RUNS[0]
    art = _artifacts(score, run)
    hits = {1: 8, 4: 3, 10: 1}                        # 1-based item -> told hits of 8
    art["v2"]["rows"] = [
        {"stratum": "S1", "distance": 0, "item": i, "seed": seed,
         "told_flags": {"hit": seed < hits.get(i + 1, 0)}}
        for i in range(len(mp2.S1_ITEMS)) for seed in range(8)]
    art["v2"]["rows"] += [{"stratum": "S2", "distance": 0, "item": 0, "seed": 0,
                           "told_flags": {"hit": True}},
                          {"stratum": "S1", "distance": 2, "item": 1, "seed": 0,
                           "told_flags": {"hit": True}}]
    got = score.score_seed(run, art)["reported"]["S1_d0_per_item"]
    assert got["told"]["1"] == score.fmt(8, 8) and got["told"]["4"] == score.fmt(3, 8)
    assert got["told"]["2"] == score.fmt(0, 8) and len(got["told"]) == 10
    assert got["near_trained_items"] == [1, 3, 10]
    assert got["told_without_near_trained"] == score.fmt(3, 56)
    # The three are what the comment says they are, in the probe's own tables.
    statements = [st for st, _q in mp2._HELDOUT_PHRASING]
    assert statements[0] == "everyone calls me {v}"
    assert statements[2].replace("i've got", "i have") in recall.TRAIN_STATEMENTS["pet"]
    assert statements[9].replace("i've", "i have") in recall.TRAIN_STATEMENTS["object"]
    # Reported only: it gates nothing.
    assert "S1_d0_per_item" not in score.score_seed(run, art)["guards"]


def test_the_n256_column_is_only_filled_from_an_n256_artifact(score):
    """Reported, not gated -- but a column headed "n256" that holds the plain
    n=8 pass a second time is a wrong number with a decoder's name on it."""
    run = score.RUNS[0]
    cells = {"S1": {"0": _cell(told=11)}, "S2": {"0": _cell(told=44)}}
    shipped = {"complete": True, "decoder": {"n": 256, "shipped_decoder": True},
               "summary": cells}
    got = score.score_seed(run, _artifacts(score, run, v2_shipped=shipped))["reported"]
    assert got["n256_pass"] == {"S1_d0": score.fmt(11, 80), "S2_d0": score.fmt(44, 80)}

    plain = {**shipped, "decoder": {"n": 8, "shipped_decoder": False}}
    got = score.score_seed(run, _artifacts(score, run, v2_shipped=plain))["reported"]
    assert list(got["n256_pass"]) == ["not_the_shipped_decoder"]
    assert "n=8" in got["n256_pass"]["not_the_shipped_decoder"]
    assert score.score_seed(run, _artifacts(score, run))["reported"]["n256_pass"] == {}
    assert score.SHIPPED_DECODER_N == 256


def test_both_seeds_must_clear_and_a_straddle_is_unresolved(score):
    o = score.overall
    assert o(["ship-candidate", "ship-candidate"]) == "ship-candidate"
    assert o(["ship-candidate", "resolved"]) == o(["resolved", "resolved"]) == "resolved"
    assert o(["floor", "floor"]) == "floor"
    assert o(["resolved", "floor"]) == o(["floor", "ship-candidate"]) == "unresolved"
    assert o(["ship-candidate", "not run"]) == o(["floor", "not run"]) == "unresolved"
    assert o(["ship-candidate"]) == "unresolved"       # n = 2 was fixed in advance


def test_the_decision_rule_fires_only_where_it_was_pre_registered(score):
    def seeds(**kw):
        return [score.score_seed(r, _mut(score, r, **kw)) for r in score.RUNS]

    def _mut(score, run, s1=8, s2=(40, 30, 25), committed_c=12):
        art = _artifacts(score, run)
        art["v2"]["summary"]["S1"]["0"] = _cell(told=s1)
        art["v2"]["summary"]["S2"] = {str(d): _cell(told=k) for d, k in enumerate(s2)}
        art["committed"]["summary"]["0"]["discordant"] = [0, committed_c]
        return art

    assert score.decision_rule("resolved", seeds())["fired"] is None       # 25 >= 40/2
    assert "carried-state" in score.decision_rule("resolved", seeds(s2=(40, 30, 19)))["fired"]
    stop = score.decision_rule("floor", seeds(s1=0, s2=(0, 0, 0), committed_c=0))
    assert "stop the memory programme" in stop["fired"]
    # S1 at the floor but trained phrasings resolve: recitation, not the stop rule.
    assert score.decision_rule("floor", seeds(s1=0, s2=(30, 0, 0)))["fired"] is None
    assert score.decision_rule("unresolved", seeds())["fired"] is None


def test_the_bpc_bands_are_derived_from_their_evidence(score, monkeypatch):
    assert score._check_transcription()["recomputed_ok"]
    assert score.BPC_SIGMAS == 2.83 and set(score.BPC_BAND) == set(score.GUARDED_SOURCES)
    monkeypatch.setitem(score.BPC_BAND, "soda", 0.0030000)
    with pytest.raises(SystemExit, match="TRANSCRIPTION"):
        score._check_transcription()


def test_template_capture_covers_the_recall_sources_own_lines(score):
    for text in ("Your name is Tom.", "you told me your dog is called Rex. Anything else?",
                 "You haven't told me your favourite food yet.", "Got it.",
                 "An orange hat sounds nice.", "I'm good, thank you.",
                 "Your favourite food is ice cream."):
        assert score.is_recall_template(text), text
    for text in ("Once upon a time there was a dog named Rex.", "Red, blue and green.",
                 "I don't really have a name. I'm a spiking neural network.", ""):
        assert not score.is_recall_template(text), text
    # every answer the generator can write is recognised
    for d in recall.build_recall_dialogues(300, seed=3):
        assert score.is_recall_template(d.turns[d.answer_turn][1])


def test_score_v15_reads_the_drivers_layout_end_to_end(score, tmp_path, capsys):
    names = {"committed": "memory_probe.json", "v2": "memory_probe_v2_n8.json",
             "battery": "battery.json", "heldout": "heldout_n256.json"}
    for i, run in enumerate(score.RUNS):
        art = _artifacts(score, run) if i == 0 else _with_s1(score, run, told=2)
        (tmp_path / "scores" / run).mkdir(parents=True)
        (tmp_path / "runs" / run).mkdir(parents=True)
        for key, name in names.items():
            (tmp_path / "scores" / run / name).write_text(json.dumps(art[key]), encoding="utf-8")
        (tmp_path / "runs" / run / "summary.json").write_text(
            json.dumps(art["summary"]), encoding="utf-8")
        (tmp_path / "runs" / run / "log.jsonl").write_text(
            '{"event": "run_start"}\n' + ('{"event": "divergence", "step": 5}\n' * i),
            encoding="utf-8")
    assert score.main(["--out-root", str(tmp_path)]) == 0
    blob = json.loads((tmp_path / "score_v15.json").read_text(encoding="utf-8"))
    assert [s["verdict"] for s in blob["seeds"]] == ["ship-candidate", "floor"]
    assert blob["verdict"] == "unresolved"
    assert [s["rolled_back"] for s in blob["seeds"]] == [False, True]
    assert blob["bars"]["committed_told_min"] == 15.5
    out = capsys.readouterr().out
    assert "OVERALL: UNRESOLVED" in out and "ROLLED BACK" in out


# --------------------------------------------------------------------------
# the driver
# --------------------------------------------------------------------------

_REF_CFG = {
    "data_dir": "data/chat", "seq_len": 256, "batch_size": 160,
    "mix": {"stories_topic": 0.4, "soda": 0.2, "alpaca": 0.15, "tinystories": 0.09,
            "persona": 0.08, "dolly": 0.06, "oasst1": 0.02},
    "bot_loss_weight": 3.0, "align_frac": 0.75, "align_lookahead": 1024, "d_model": 1024,
    "n_layers": 4, "vocab_size": 101, "arch": "twocomp_threshold", "beta": 0.5,
    "threshold": 1.0, "surrogate_alpha": 2.0, "w_init": 0.1, "thr_log_init": 0.0,
    "beta_slow": 0.95, "spread_tau": True, "tau_min": 3.0, "tau_max": 600.0, "lr": 0.0005,
    "weight_decay": 0.1, "beta1": 0.9, "beta2": 0.95, "grad_clip": 1.0, "warmup_steps": 200,
    "max_steps": 14000, "lr_schedule": "cosine", "lr_final_frac": 0.05, "dtype": "fp32",
    "fused": True, "cuda_graph": True, "device": "cuda", "seed": 0, "deterministic": False,
    "run_name": "chat-v3d-aligned", "out_dir": "experiments/chat", "eval_every": 3500,
    "eval_batches": 30, "eval_batch_size": 48, "ckpt_every": 1000, "log_every": 250,
    "sample_every": 3500, "vocab_version": 1,
}


@pytest.fixture()
def world(tmp_path, driver):
    """A main tree, a packed corpus and an out-root, none of them real."""
    chat = tmp_path / "main" / "experiments" / "chat"
    (chat / "chat-v3d-aligned").mkdir(parents=True)
    (chat / "chat-v2-anneal").mkdir()
    (chat / "SHIPPED").write_text("chat-v3d-aligned/ckpt_best.pt   # a comment\n# more\n",
                                  encoding="utf-8")
    (chat / "chat-v3d-aligned" / "ckpt_best.pt").write_bytes(b"")
    (chat / "chat-v2-anneal" / "ckpt_best.pt").write_bytes(b"")
    (chat / "chat-v3d-aligned" / "config.json").write_text(json.dumps(_REF_CFG),
                                                           encoding="utf-8")
    data = tmp_path / "corpus"
    data.mkdir()
    names = [*_REF_CFG["mix"], "recall"]
    for name in names:
        (data / f"{name}.bin").write_bytes(b"")
        (data / f"{name}.val.bin").write_bytes(b"")
    (data / "manifest.json").write_text(json.dumps(
        {"sources": {n: {"chars_train": 1} for n in names}}), encoding="utf-8")
    out = tmp_path / "out"
    argv = ["--out-root", str(out), "--data-dir", str(data), "--chat-root", str(chat)]
    return types.SimpleNamespace(chat=chat, data=data, out=out, argv=argv, tmp=tmp_path)


def _fake_run(*, porcelain="", code_porcelain="", uncommitted=(), smi="", smi_rc=0):
    """Stands in for `git` and `nvidia-smi`. Records what it was asked, and how."""
    calls = []
    kwargs = []

    def run(cmd, **kw):
        calls.append(cmd)
        kwargs.append(kw)

        def done(out="", rc=0, err=""):
            return subprocess.CompletedProcess(cmd, rc, out, err)

        if cmd[0] == "nvidia-smi":
            return done(smi, smi_rc, "boom" if smi_rc else "")
        if cmd[:2] == ["git", "status"]:
            return done(code_porcelain if cmd[4:] == ["src/snnchat", "scripts/chat"]
                        else porcelain)
        if cmd[:2] == ["git", "rev-parse"]:
            if cmd[2] == "HEAD":
                return done("f" * 40 + "\n")
            path = cmd[2].split(":", 1)[1]
            return done("", 128, "fatal") if path in uncommitted else \
                done(hashlib.sha1(path.encode()).hexdigest() + "\n")
        raise AssertionError(f"unexpected command {cmd}")

    run.calls = calls
    run.kwargs = kwargs
    return run


def test_dry_run_resolves_every_path_and_orders_the_stages(driver, world, capsys):
    assert driver.main([*world.argv, "--dry-run"]) == 0
    out = capsys.readouterr().out
    stages = [ln[1:-1] for ln in out.splitlines() if ln.startswith("[") and ln.endswith("]")]
    assert stages == ["preflight", "floors", "smoke", "train:chat-v15-recall-s0",
                      "train:chat-v15-recall-s1", "score:chat-v15-recall-s0",
                      "score:chat-v15-recall-s1", "verdict"]
    assert not world.out.exists()                                  # it ran nothing
    commands = [ln.strip()[2:] for ln in out.splitlines() if ln.strip().startswith("$ ")]
    assert len(commands) == 3 + 1 + 2 + 6 + 6 + 1
    trains = [c for c in commands if "train.py" in c]
    for c in trains:
        assert f"--data-dir {world.data}" in c and f"--init-from {world.chat}" in c
        assert "alpaca=0.09" in c and "recall=0.06" in c and "stories_topic=0.4" in c
        assert "--lr 0.0005" in c and "--batch-size 160" in c and "--no-resume" in c
    assert "--max-steps 20 " in trains[0] and "--seed 0 " in trains[1] and "--seed 1 " in trains[2]
    assert all("--max-steps 14000 " in c for c in trains[1:])
    # every path a child is given is absolute
    for c in commands:
        toks = c.split(" ")
        for flag in ("--ckpt", "--out", "--out-root", "--out-dir", "--data-dir", "--init-from"):
            for i, t in enumerate(toks):
                if t == flag:
                    assert pathlib.Path(toks[i + 1]).is_absolute(), (flag, toks[i + 1])
    floors = [c for c in commands if str(world.out / "floors") in c]
    assert len(floors) == 3 and all(str(world.chat / "chat-v3d-aligned") in c for c in floors)
    assert sum("--shipped-decoder" in c for c in commands) == 2
    assert sum("--set heldout --seeds 6 --n 256" in c for c in commands) == 2


def test_the_driver_never_writes_into_the_tree_it_reads(driver, world):
    inside = world.chat.parents[1] / "scratch"
    with pytest.raises(SystemExit, match="inside"):
        driver.main(["--out-root", str(inside), "--data-dir", str(world.data),
                     "--chat-root", str(world.chat), "--dry-run"])


def test_preflight_refuses_an_uncommitted_or_dirty_pre_registration(driver):
    clean = driver.check_prereg(_fake_run())
    assert clean["head"] == "f" * 40 and set(clean["blobs"]) == set(driver.PREREGISTERED)
    for must in ("docs/chat/PREDICTION_v15.md", "src/snnchat/recall.py",
                 "scripts/chat/memory_probe_v2.py", "scripts/chat/score_v15.py"):
        assert must in driver.PREREGISTERED
    with pytest.raises(driver.Refusal, match="not committed as it stands"):
        driver.check_prereg(_fake_run(porcelain=" M scripts/chat/score_v15.py\n"))
    with pytest.raises(driver.Refusal, match="not committed as it stands"):
        driver.check_prereg(_fake_run(porcelain="?? scripts/chat/memory_probe_v2.py\n"))
    with pytest.raises(driver.Refusal, match="PREDICTION_v15.md is not committed"):
        driver.check_prereg(_fake_run(uncommitted=("docs/chat/PREDICTION_v15.md",)))
    # the status query is scoped to the pre-registered paths, not the whole tree
    run = _fake_run()
    driver.check_prereg(run)
    assert run.calls[0][:4] == ["git", "status", "--porcelain", "--"]
    assert run.calls[0][4:] == list(driver.PREREGISTERED)


def test_preflight_refuses_a_busy_gpu(driver):
    import os
    idle = "1880, C:\\Windows\\System32\\dwm.exe\n4092, C:\\NVIDIA\\NVIDIA Overlay.exe\n"
    assert driver.check_gpu(force=False, run=_fake_run(smi=idle)) == []
    busy = idle + "4242, C:\\Users\\E\\Python314\\python.exe\n"
    with pytest.raises(driver.Refusal, match="4242"):
        driver.check_gpu(force=False, run=_fake_run(smi=busy))
    assert driver.check_gpu(force=True, run=_fake_run(smi=busy)) == []
    assert driver.gpu_python_processes(_fake_run(smi="77, /usr/bin/python3.14\n")) \
        == [(77, "/usr/bin/python3.14")]
    # this process and its parent are not "another" process
    mine = f"{os.getpid()}, C:\\py\\python.exe\n{os.getppid()}, C:\\py\\pythonw.exe\n"
    assert driver.check_gpu(force=False, run=_fake_run(smi=mine)) == []
    # not being able to ask is not the same as an idle GPU
    with pytest.raises(driver.Refusal, match="nvidia-smi exited"):
        driver.check_gpu(force=False, run=_fake_run(smi_rc=9))


def test_the_recipe_gate_passes_the_generated_command_and_nothing_else(driver, world):
    cmd = driver.train_command(_REF_CFG, "chat-v15-recall-s1", 1, data_dir=world.data,
                               out_dir=world.out / "runs",
                               parent=world.chat / driver.PARENT)
    diff = driver.check_recipe(_REF_CFG, cmd)
    assert set(diff["differs"]) == {"mix", "run_name", "seed", "out_dir", "data_dir"}
    assert set(diff["differs"]) <= driver.MAY_DIFFER and not diff["violations"]
    assert driver.v15_mix(_REF_CFG["mix"]) == {**_REF_CFG["mix"], "alpaca": 0.09, "recall": 0.06}

    for flag, value in (("--lr", "0.001"), ("--weight-decay", "0.0"), ("--max-steps", "7000"),
                        ("--bot-loss-weight", "1.0")):
        bad = list(cmd)
        bad[bad.index(flag) + 1] = value
        with pytest.raises(driver.Refusal, match=flag[2:].replace("-", "_")):
            driver.check_recipe(_REF_CFG, bad)
    # a flag nobody passed is a default nobody chose: dropping one is caught too
    i = cmd.index("--batch-size")
    with pytest.raises(driver.Refusal, match="batch_size"):
        driver.check_recipe(_REF_CFG, cmd[:i] + cmd[i + 2:])
    # the mix: only alpaca may move, and only by the recall weight
    for old, new in (("soda=0.2", "soda=0.14"), ("alpaca=0.09", "alpaca=0.15"),
                     ("recall=0.06", "recall=0.08")):
        with pytest.raises(driver.Refusal, match="mix|new source"):
            driver.check_recipe(_REF_CFG, [new if c == old else c for c in cmd])

    written = {**_REF_CFG, "run_name": "x", "mix": driver.v15_mix(_REF_CFG["mix"])}
    path = world.tmp / "config.json"
    path.write_text(json.dumps(written), encoding="utf-8")
    driver.check_written_config(_REF_CFG, path)
    path.write_text(json.dumps({**written, "vocab_size": 205}), encoding="utf-8")
    with pytest.raises(driver.Refusal, match="vocab_size"):
        driver.check_written_config(_REF_CFG, path)
    path.write_text(json.dumps({**written, "brand_new_field": 0}), encoding="utf-8")
    with pytest.raises(driver.Refusal, match="brand_new_field"):
        driver.check_written_config(_REF_CFG, path)


def test_preflight_refuses_a_corpus_without_the_recall_source(driver, world):
    mix = driver.v15_mix(_REF_CFG["mix"])
    assert driver.check_corpus(world.data, mix)["sources"] == sorted(mix)
    (world.data / "recall.val.bin").unlink()
    with pytest.raises(driver.Refusal, match="recall.val.bin"):
        driver.check_corpus(world.data, mix)
    manifest = json.loads((world.data / "manifest.json").read_text(encoding="utf-8"))
    del manifest["sources"]["recall"]
    (world.data / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(driver.Refusal, match="'recall' is not in"):
        driver.check_corpus(world.data, mix)


class _FakeChild:
    """A child process that writes what its command line says it will write."""

    fail: set = set()
    diverge: set = set()
    bad_exit: set = set()
    launched: list = []

    def __init__(self, cmd, *, stdout, **kw):
        type(self).launched.append(cmd)
        type(self).kwargs = kw

        def arg(flag):
            return pathlib.Path(cmd[cmd.index(flag) + 1])

        self.rc = 0
        if cmd[2].endswith("train.py"):
            name = cmd[cmd.index("--run-name") + 1]
            run_dir = arg("--out-dir") / name
            steps = int(cmd[cmd.index("--max-steps") + 1])
            stdout.write("  mix: alpaca 9%, dolly 6%, oasst1 2%, persona 8%, recall 6%, "
                         "soda 20%, stories_topic 40%, tinystories 9%\n")
            stdout.flush()
            mix = dict(kv.split("=") for kv in cmd[cmd.index("--mix") + 1:])
            cfg = {**_REF_CFG, "run_name": name, "out_dir": str(arg("--out-dir")),
                   "data_dir": str(arg("--data-dir")), "max_steps": steps,
                   "seed": int(cmd[cmd.index("--seed") + 1]),
                   "mix": {k: float(v) for k, v in mix.items()}}
            (run_dir / "config.json").write_text(json.dumps(cfg), encoding="utf-8")
            events = ['{"event": "run_start"}']
            if name in self.diverge:
                events.append('{"event": "divergence", "step": 700, "loss": NaN, '
                              '"action": "rollback"}')
            (run_dir / "log.jsonl").write_text("\n".join(events) + "\n", encoding="utf-8")
            if name in self.fail:
                (run_dir / "ckpt_last.pt").write_bytes(b"partial")
                self.rc = 3
                return
            (run_dir / "ckpt_best.pt").write_bytes(b"")
            (run_dir / "summary.json").write_text(json.dumps({"steps": steps, "val": {}}),
                                                  encoding="utf-8")
        else:
            out = arg("--out-root") / "score_v15.json" if "--out-root" in cmd else arg("--out")
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps({"complete": True}), encoding="utf-8")
            if out.name in self.bad_exit:
                self.rc = 5          # the artifact is there; the exit code is not 0

    def wait(self, timeout=None):
        return self.rc


def _execute(driver, world, *, run=None, fail=(), diverge=(), bad_exit=(), **flags):
    _FakeChild.fail, _FakeChild.diverge, _FakeChild.launched = set(fail), set(diverge), []
    _FakeChild.bad_exit = set(bad_exit)
    args = types.SimpleNamespace(out_root=str(world.out), data_dir=str(world.data),
                                 chat_root=str(world.chat), device="cpu", force=False,
                                 preflight_only=False, retrain_diverged=False)
    vars(args).update(flags)
    plan = driver.build_plan(args, _REF_CFG)
    status = driver.Status(world.out / "status.json")
    rc = driver.execute(args, plan, _REF_CFG, status, run=run or _fake_run(), popen=_FakeChild)
    blob = json.loads((world.out / "status.json").read_text(encoding="utf-8"))
    return rc, blob, list(_FakeChild.launched)


def test_the_driver_trains_sequentially_scores_after_and_is_restartable(driver, world):
    rc, status, launched = _execute(driver, world)
    assert rc == 0 and status["state"] == "RUNNING"          # `main` stamps DONE on exit
    assert [s["stage"] for s in status["stages"]] == [p for p in (
        "preflight", "floors", "smoke", "train:chat-v15-recall-s0", "train:chat-v15-recall-s1",
        "score:chat-v15-recall-s0", "score:chat-v15-recall-s1", "verdict")]
    assert all(s["result"] == "ok" and s["finished"] for s in status["stages"])
    scripts = [pathlib.Path(c[2]).name for c in launched]
    first_score = scripts.index("quality.py")
    assert scripts.index("train.py") > scripts.index("binding_probe.py")     # floors first
    assert all(s != "train.py" for s in scripts[first_score:])               # then no training
    assert scripts.count("train.py") == 3 and scripts[-1] == "score_v15.py"
    record = json.loads((world.out / "driver_record.json").read_text(encoding="utf-8"))
    assert record["prereg"]["head"] == "f" * 40
    assert record["mix"]["recall"] == 0.06 and record["parent"].endswith("ckpt_best.pt")
    assert "recall 6%" in status["stages"][2]["smoke"]["mix_line"]

    # A second start retrains nothing and re-probes nothing; only the zero-GPU verdict reruns.
    rc, status, launched = _execute(driver, world)
    names = [(pathlib.Path(c[2]).name, c[c.index("--run-name") + 1] if "--run-name" in c else "")
             for c in launched]
    assert names == [("train.py", "chat-v15-smoke"), ("score_v15.py", "")]
    assert [s["result"] for s in status["stages"]][3:5] == ["skipped-complete"] * 2


def test_a_partial_run_is_deleted_not_resumed(driver, world):
    partial = world.out / "runs" / "chat-v15-recall-s0"
    partial.mkdir(parents=True)
    (partial / "ckpt_last.pt").write_bytes(b"stale")
    (partial / "summary.json").write_text('{"steps": 9000}', encoding="utf-8")
    rc, status, _ = _execute(driver, world)
    assert rc == 0 and not (partial / "ckpt_last.pt").exists()
    assert json.loads((partial / "summary.json").read_text(encoding="utf-8"))["steps"] == 14000
    with pytest.raises(driver.Refusal):
        driver._safe_rmtree(world.chat, world.out)               # never outside --out-root


def test_a_failed_seed_is_reported_and_the_other_still_runs(driver, world):
    rc, status, launched = _execute(driver, world, fail={"chat-v15-recall-s0"})
    assert rc == 1 and status["state"] == "FAILED"
    results = {s["stage"]: s for s in status["stages"]}
    assert results["train:chat-v15-recall-s0"]["result"] == "failed"
    assert "rc=3" in results["train:chat-v15-recall-s0"]["error"]
    assert results["train:chat-v15-recall-s1"]["result"] == "ok"
    assert results["score:chat-v15-recall-s0"]["result"] == "skipped"
    assert results["score:chat-v15-recall-s1"]["result"] == "ok"
    scored = [c[c.index("--ckpt") + 1] for c in launched if "--ckpt" in c]
    assert not any("recall-s0" in c for c in scored)


def test_an_artifact_from_a_child_that_exited_nonzero_is_not_a_result(driver, world):
    rc, status, launched = _execute(driver, world, bad_exit={"battery.json"})
    results = {s["stage"]: s for s in status["stages"]}
    assert rc == 1 and status["state"] == "FAILED"
    assert results["score:chat-v15-recall-s0"]["result"] == "failed"
    assert "rc=5" in results["score:chat-v15-recall-s0"]["error"]
    assert results["train:chat-v15-recall-s1"]["result"] == "ok"
    # one probe dying does not cost the stage its other readouts
    after = [pathlib.Path(c[2]).name for c in launched if "recall-s0" in " ".join(c)]
    assert after[-2:] == ["quality.py", "echo_holdout.py"]
    assert str(world.out / "scores" / "chat-v15-recall-s0" / "heldout_n256.json")         in results["score:chat-v15-recall-s0"]["artifacts"]

    # ... and a probe that cannot read SHIPPED stops the round before it trains.
    shutil.rmtree(world.out)
    rc, status, launched = _execute(driver, world, bad_exit={"binding_probe.json"})
    assert rc == 1 and [s["stage"] for s in status["stages"]] == ["preflight", "floors"]
    assert not any(c[2].endswith("train.py") for c in launched)


def test_a_divergence_event_is_surfaced_in_the_status_file(driver, world):
    rc, status, _ = _execute(driver, world, diverge={"chat-v15-recall-s1"})
    assert rc == 0 and status["state"] == "DIVERGED"
    assert [a["state"] for a in status["alerts"]] == ["DIVERGED"]
    assert "chat-v15-recall-s1" in status["alerts"][0]["message"]
    assert "700" in status["alerts"][0]["message"]


def test_nothing_trains_past_a_refused_preflight_or_a_bad_smoke(driver, world, monkeypatch):
    busy = _fake_run(smi="4242, C:\\py\\python.exe\n")
    rc, status, launched = _execute(driver, world, run=busy)
    assert rc == 2 and launched == [] and status["state"] == "FAILED"
    assert "4242" in status["stages"][0]["error"]

    rc, status, launched = _execute(driver, world, run=_fake_run(porcelain=" M x\n"))
    assert rc == 2 and launched == []

    # A smoke run whose banner shows a different mixture stops everything after it.
    real = _FakeChild.__init__

    def wrong_mix(self, cmd, *, stdout, **kw):
        real(self, cmd, stdout=stdout, **kw)
        if "chat-v15-smoke" in cmd:
            stdout.write("  WARNING: 1 requested source(s) not in the corpus: recall\n"
                         "  mix: alpaca 10%, soda 21%\n")
            stdout.flush()

    monkeypatch.setattr(_FakeChild, "__init__", wrong_mix)
    rc, status, launched = _execute(driver, world)
    assert rc == 2 and status["state"] == "FAILED"
    assert [pathlib.Path(c[2]).name for c in launched].count("train.py") == 1
    assert "recall" in status["stages"][-1]["error"]


def test_a_run_without_its_best_checkpoint_is_not_complete(driver, tmp_path):
    """Every probe loads `ckpt_best.pt`. A run skipped as complete on its
    `summary.json` alone would fail all six of them an hour later."""
    (tmp_path / "summary.json").write_text('{"steps": 14000}', encoding="utf-8")
    assert not driver.complete(tmp_path, 14000)
    (tmp_path / "ckpt_best.pt").write_bytes(b"")
    assert driver.complete(tmp_path, 14000) and not driver.complete(tmp_path, 13999)
    (tmp_path / "summary.json").write_text("{not json", encoding="utf-8")
    assert not driver.complete(tmp_path, 14000)


def test_no_child_is_given_a_console_window(driver, world):
    """The driver runs detached, with no console; a console child of such a
    parent gets a visible window of its own, and closing it kills the child."""
    run = _fake_run()
    rc, _status, launched = _execute(driver, world, run=run)
    assert rc == 0 and launched
    assert _FakeChild.kwargs["creationflags"] == driver._NO_WINDOW
    assert {c[0] for c in run.calls} == {"git", "nvidia-smi"}
    assert all(kw["creationflags"] == driver._NO_WINDOW for kw in run.kwargs)
    if sys.platform == "win32":
        assert driver._NO_WINDOW == subprocess.CREATE_NO_WINDOW != 0
        # ... and a real child started this way still runs and is still captured.
        r = subprocess.run([sys.executable, "-c", "print('alive')"], capture_output=True,
                           text=True, creationflags=driver._NO_WINDOW)
        assert (r.returncode, r.stdout.strip()) == (0, "alive")


def test_preflight_only_runs_the_checks_in_the_foreground_and_nothing_else(driver, world,
                                                                          monkeypatch, capsys):
    rc, status, launched = _execute(driver, world, preflight_only=True)
    assert rc == 0 and launched == []
    assert [(s["stage"], s["result"]) for s in status["stages"]] == [("preflight", "ok")]
    assert (world.out / "driver_record.json").exists()

    refused = _fake_run(uncommitted=("docs/chat/PREDICTION_v15.md",))
    rc, status, launched = _execute(driver, world, run=refused, preflight_only=True)
    assert rc == 2 and launched == [] and status["state"] == "FAILED"
    assert "PREDICTION_v15.md is not committed" in status["stages"][0]["error"]

    # Through `main`: success must not read DONE, which is what a finished ROUND reads.
    real = driver.execute
    monkeypatch.setattr(driver, "execute", lambda args, plan, cfg, st: real(
        args, plan, cfg, st, run=_fake_run(), popen=_FakeChild))
    assert driver.main([*world.argv, "--preflight-only"]) == 0
    blob = json.loads((world.out / "status.json").read_text(encoding="utf-8"))
    assert blob["state"] == "PREFLIGHT_OK" and blob["finished"] is True
    assert "PREFLIGHT_OK" in capsys.readouterr().out


def test_preflight_refuses_modified_code_outside_the_pre_registration(driver):
    assert driver.CODE_DIRS == ("src/snnchat", "scripts/chat")
    with pytest.raises(driver.Refusal, match="HEAD does not describe"):
        driver.check_prereg(_fake_run(code_porcelain=" M src/snnchat/data.py\n"))
    with pytest.raises(driver.Refusal, match="train.py"):
        driver.check_prereg(_fake_run(code_porcelain="M  scripts/chat/train.py\n"))
    # An untracked file cannot be imported by tracked code that has not itself
    # changed: recorded, not refused.
    got = driver.check_prereg(_fake_run(code_porcelain="?? scripts/chat/scratch.py\n"))
    assert got["untracked_code"] == ["?? scripts/chat/scratch.py"]
    run = _fake_run()
    assert driver.check_prereg(run)["untracked_code"] == []
    assert run.calls[1] == ["git", "status", "--porcelain", "--", *driver.CODE_DIRS]


def test_a_restart_keeps_what_the_attempt_before_it_reported(driver, world):
    rc, first, _ = _execute(driver, world, fail={"chat-v15-recall-s0"},
                            diverge={"chat-v15-recall-s0"})
    assert rc == 1 and first["state"] == "FAILED" and first["previous"] == []
    assert [a["state"] for a in first["alerts"]] == ["DIVERGED"]
    head_then = json.loads((world.out / "driver_record.json")
                           .read_text(encoding="utf-8"))["recorded"]

    rc, second, _ = _execute(driver, world, fail={"chat-v15-recall-s0"},
                             diverge={"chat-v15-recall-s0"})
    assert len(second["previous"]) == 1
    before = second["previous"][0]
    assert before["state"] == "FAILED" and "previous" not in before
    assert [a["state"] for a in before["alerts"]] == ["DIVERGED"]
    rc, third, _ = _execute(driver, world, fail={"chat-v15-recall-s0"})
    assert [p["started"] for p in third["previous"]] == [first["started"], second["started"]]
    record = json.loads((world.out / "driver_record.json").read_text(encoding="utf-8"))
    assert [h["head"] for h in record["history"]] == ["f" * 40] * 2
    assert record["history"][0]["recorded"] == head_then


def test_a_partial_run_that_diverged_is_not_silently_replaced(driver, world):
    """The power rule: a rolled-back seed is reported and NOT replaced. The
    recipe trains with `deterministic` off, so a re-train is a different seed."""
    assert _REF_CFG["deterministic"] is False
    rc, status, _ = _execute(driver, world, fail={"chat-v15-recall-s0"},
                             diverge={"chat-v15-recall-s0"})
    assert rc == 1
    partial = world.out / "runs" / "chat-v15-recall-s0"
    assert (partial / "ckpt_last.pt").read_bytes() == b"partial"

    # Restarted with a trainer that would now succeed: the seed must stay failed.
    rc, status, launched = _execute(driver, world)
    results = {s["stage"]: s for s in status["stages"]}
    assert rc == 1 and status["state"] == "FAILED"
    assert "replace a failed seed" in results["train:chat-v15-recall-s0"]["error"]
    assert (partial / "ckpt_last.pt").read_bytes() == b"partial"          # left as it was
    assert results["score:chat-v15-recall-s0"]["result"] == "skipped"
    assert results["train:chat-v15-recall-s1"]["result"] == "skipped-complete"
    assert not any("chat-v15-recall-s0" in c for c in launched if c[2].endswith("train.py"))

    # The override exists, and says what it did.
    rc, status, launched = _execute(driver, world, retrain_diverged=True)
    results = {s["stage"]: s for s in status["stages"]}
    assert rc == 0 and results["train:chat-v15-recall-s0"]["result"] == "ok"
    assert results["train:chat-v15-recall-s0"]["retrained_after_divergence"][0]["step"] == 700
    assert any("--retrain-diverged" in a["message"] for a in status["alerts"])
    # ... while a partial run that never diverged is still simply re-trained.
    shutil.rmtree(world.out)
    _execute(driver, world, fail={"chat-v15-recall-s1"})
    rc, status, _ = _execute(driver, world)
    assert rc == 0 and {s["result"] for s in status["stages"]} <= {"ok", "skipped-complete"}


def test_a_plan_that_cannot_be_built_still_leaves_a_status_file(driver, world):
    (world.chat / "SHIPPED").write_text("nowhere/ckpt_best.pt\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="cannot build the plan"):
        driver.main(world.argv)
    blob = json.loads((world.out / "status.json").read_text(encoding="utf-8"))
    assert blob["state"] == "FAILED" and blob["finished"] is True
    assert "nowhere" in blob["crash"]
    # ... and a dry run still writes nothing at all.
    shutil.rmtree(world.out)
    with pytest.raises(SystemExit):
        driver.main([*world.argv, "--dry-run"])
    assert not world.out.exists()


def test_a_changed_scorer_cannot_judge_a_checkpoint_that_predates_it(driver, world):
    rc, _status, _ = _execute(driver, world)
    assert rc == 0

    def moved(cmd, **kw):
        done = _fake_run()(cmd, **kw)
        if cmd[:2] == ["git", "rev-parse"] and cmd[2].endswith("score_v15.py"):
            return subprocess.CompletedProcess(cmd, 0, "0" * 40 + "\n", "")
        return done

    rc, status, launched = _execute(driver, world, run=moved)
    assert rc == 2 and launched == []
    assert "score_v15.py" in status["stages"][0]["error"]
