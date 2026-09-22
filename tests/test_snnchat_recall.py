"""`snnchat.recall` and `scripts/chat/build_recall.py`: the user-fact recall source.

CPU only, no corpus, no checkpoint. What is held up here is mostly a set of
PROMISES the generator makes to a probe that does not exist yet
(`scripts/chat/memory_probe_v2.py`): that a held-out value, a held-out phrasing
and a held-out frame are really held out; that a dialogue really fits a training
window; that the acknowledgement really is not a second copy of the answer. The
generator asserts most of them on itself at generation time, so each test below
checks the property by a route that does NOT go through the generator's own
guard -- a guard and a test that share an implementation share its bugs.

Every test here was run against a deliberately broken generator and seen to
fail; the mutation each one caught is in the chat-v15 B1 and B4 hand-offs.

Not part of the research protocol.
"""

from __future__ import annotations

import dataclasses
import hashlib
import importlib.util
import json
import pathlib
import re
import statistics
import sys

import numpy as np
import pytest

from snnchat import recall
from snnchat.persona import TOPICS
from snnchat.tokenizer import BOS, ChatTokenizer

ROOT = pathlib.Path(__file__).resolve().parents[1]

#: Big enough that every shape share is known to about a quarter of a percent,
#: small enough to build in under a second.
N = 20_000


def _load_script(name: str):
    path = ROOT / "scripts" / "chat" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"_{name}_under_test", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def dialogues():
    return recall.build_recall_dialogues(N, seed=0)


@pytest.fixture(scope="module")
def build_recall():
    return _load_script("build_recall")


def _words(text: str) -> set[str]:
    return set(re.findall(r"[a-z]+(?:'[a-z]+)?", text.lower()))


def _has(text: str, value: str) -> bool:
    """Whole-word, case-insensitive -- the rule `memory_probe.hit` scores by."""
    return re.search(r"\b" + re.escape(value) + r"\b", text.lower()) is not None


# ---------------------------------------------------------------------------
# determinism
# ---------------------------------------------------------------------------

def test_recall_is_a_pure_function_of_n_and_seed():
    a = recall.build_recall_conversations(500, seed=3)
    b = recall.build_recall_conversations(500, seed=3)
    assert a == b
    assert a != recall.build_recall_conversations(500, seed=4)
    # The prefix property the docstring promises: a longer build extends a
    # shorter one, so reasoning about N here says something about the pack.
    assert recall.build_recall_conversations(700, seed=3)[:500] == a


def test_conversations_have_the_persona_structure(dialogues):
    """`_pack_source` renders these with `render_conversation`, which accepts
    exactly what `build_persona_conversations` returns: a list of lists of
    `(role, text)` with roles alternating from `user`."""
    convs = recall.build_recall_conversations(200, seed=0)
    assert convs == [list(d.turns) for d in dialogues[:200]]
    for conv in convs:
        assert isinstance(conv, list) and len(conv) % 2 == 0
        assert [role for role, _ in conv] == ["user", "bot"] * (len(conv) // 2)
        assert all(isinstance(text, str) and text for _, text in conv)


# ---------------------------------------------------------------------------
# the reservations
# ---------------------------------------------------------------------------

def test_no_heldout_value_is_ever_emitted(dialogues):
    held = {w for words in recall.HELDOUT_VALUES.values() for w in words}
    held |= {w for words in recall.HELDOUT_AUX.values() for w in words}
    assert held and all(" " not in w for w in held)     # so a word set suffices
    for d in dialogues:
        for _role, text in d.turns:
            assert not (_words(text) & held), text


def test_heldout_and_trained_values_are_disjoint_everywhere():
    trained = {w for s in recall.SLOTS for w in recall.VALUES[s]}
    trained |= {w for words in recall.AUX.values() for w in words}
    held = {w for s in recall.SLOTS for w in recall.HELDOUT_VALUES[s]}
    held |= {w for words in recall.HELDOUT_AUX.values() for w in words}
    assert not (trained & held)
    assert "elliot" in recall.VALUES["name"]            # the owner's own demo
    assert recall.VALUES["object"] == recall.VALUES["colour"]


def _pattern(template: str, slot: str) -> re.Pattern:
    """A held-out template as a regex over KNOWN values -- never a wildcard,
    which would match the trained "what is my name" against "{v} is my name"."""
    values = recall.VALUES[slot] + recall.HELDOUT_VALUES[slot]
    auxes = recall.AUX.get(slot, ()) + recall.HELDOUT_AUX.get(slot, ())
    body = re.escape(template)
    body = body.replace(re.escape("{v}"), "(?:" + "|".join(map(re.escape, values)) + ")")
    body = body.replace(re.escape("{x}"), "(?:" + "|".join(map(re.escape, auxes)) + ")")
    body = body.replace(re.escape("{a}"), "an?")
    return re.compile(r"(?<![a-z])" + body.rstrip(r"\?") + r"(?![a-z])")


def test_no_heldout_phrasing_is_ever_emitted(dialogues):
    patterns = [_pattern(t, slot) for slot in recall.SLOTS
                for t in recall.HELDOUT_STATEMENTS[slot] + recall.HELDOUT_QUESTIONS[slot]]
    patterns += [re.compile(re.escape(f.rstrip("?"))) for f in recall.HELDOUT_FILLERS]
    markers = [m for _s, _q, ms in recall.HELDOUT_FRAMES for m in ms]
    user_turns = {text for d in dialogues for role, text in d.turns if role == "user"}
    assert len(user_turns) > 1000
    for text in user_turns:
        for pat in patterns:
            assert pat.search(text) is None, (text, pat.pattern)
        for marker in markers:
            assert marker not in text, (text, marker)


def test_heldout_phrasings_are_not_trained_phrasings_in_disguise():
    for slot in recall.SLOTS:
        train = set(recall.TRAIN_STATEMENTS[slot]) | set(recall.TRAIN_QUESTIONS[slot])
        held = set(recall.HELDOUT_STATEMENTS[slot]) | set(recall.HELDOUT_QUESTIONS[slot])
        assert len(held) >= 6
        assert not (train & held)
        for h in held:
            for t in train:
                bare_h, bare_t = (re.sub(r"[^a-z{} ]", "", x) for x in (h, t))
                assert bare_h not in bare_t and bare_t not in bare_h, (h, t)
    users = {u for u, _b in recall.TRAIN_FILLERS}
    assert not (users & set(recall.HELDOUT_FILLERS))


@pytest.mark.parametrize("turns", [
    [("user", "my name is oscar"), ("bot", "Got it.")],                  # held-out value
    [("user", "my name is tom"), ("bot", "Hello, Hannah!")],             # ... in a bot turn
    [("user", "i have a kitten called rex"), ("bot", "Got it.")],        # held-out aux
    [("user", "who am i?"), ("bot", "Your name is Tom.")],               # held-out question
    [("user", "i go by tom"), ("bot", "Got it.")],                       # held-out statement
    [("user", "my name is tom and i go by tom"), ("bot", "Got it.")],    # ... joined
    [("user", "do you like music?"), ("bot", "Yes.")],                   # held-out filler
    [("user", "i am seven years old"), ("bot", "Got it.")],              # held-out frame
    [("user", "my name is tom " + "x" * 230), ("bot", "Got it.")],       # too long
    [("user", "my name is tom" + chr(233)), ("bot", "Got it.")],         # outside vocab v1
])
def test_the_generators_own_guard_fires(turns):
    """The guard runs on every dialogue at generation time. It is only worth
    that if it can fail, so feed it one of everything it exists to refuse."""
    with pytest.raises(AssertionError):
        recall._assert_clean(turns)
    recall._assert_clean([("user", "my name is tom"), ("bot", "Hello, Tom!")])


# ---------------------------------------------------------------------------
# the window
# ---------------------------------------------------------------------------

def test_every_dialogue_fits_a_window_and_round_trips_the_tokenizer(dialogues):
    tok = ChatTokenizer()
    longest = 0
    for d in dialogues:
        ids = tok.render_conversation(list(d.turns))
        assert ids[0] == BOS
        assert len(ids) == recall.rendered_length(d.turns)      # the stdlib count is exact
        assert len(ids) <= recall.MAX_RENDERED_CHARS
        longest = max(longest, len(ids))
        for _role, text in d.turns:
            enc = tok.encode(text)
            assert len(enc) == len(text)                        # nothing dropped or folded
            assert tok.decode_visible(enc) == text
    # 230 is a ceiling the generator REACHES, not one the templates merely
    # happen to sit under: if nothing comes near it the redraw loop is dead code
    # and this test would pass with the ceiling deleted.
    assert longest > recall.MAX_RENDERED_CHARS - 5
    assert recall.MAX_RENDERED_CHARS < recall.DOSE_RECIPE["seq_len"]


def test_user_turns_are_lower_case(dialogues):
    for d in dialogues:
        for role, text in d.turns:
            if role == "user":
                assert text == text.lower(), text


# ---------------------------------------------------------------------------
# shapes
# ---------------------------------------------------------------------------

def test_shape_shares_are_within_tolerance(dialogues):
    # Pinned, because the dose arithmetic and the probe's strata were sized
    # against these and a quiet edit to the table would otherwise pass this test
    # by moving the target along with the draw.
    assert recall.SHAPE_SHARES == {"single": 0.35, "filler": 0.15, "two_fact": 0.20,
                                   "untold": 0.15, "contrast": 0.15}
    assert tuple(recall.SHAPE_SHARES) == recall.SHAPES
    for shape in recall.SHAPES:
        share = sum(d.shape == shape for d in dialogues) / len(dialogues)
        # Binomial sd at N = 20,000 is below 0.0034 for every share; 0.015 is
        # over four of them.
        assert share == pytest.approx(recall.SHAPE_SHARES[shape], abs=0.015), shape


def test_shapes_have_the_turn_structure_they_claim(dialogues):
    fillers = {u for u, _b in recall.TRAIN_FILLERS}
    for d in dialogues:
        users = [t for r, t in d.turns if r == "user"]
        if d.shape == "single":
            assert len(d.turns) == 4 and d.statement_turn == 0 and d.answer_turn == 3
        elif d.shape == "filler":
            assert d.fillers in (1, 2) and len(d.turns) == 4 + 2 * d.fillers
            between = users[1:-1]
            assert len(between) == d.fillers == len(set(between)) and set(between) <= fillers
            assert d.statement_turn == 0 and d.answer_turn == len(d.turns) - 1
        elif d.shape == "two_fact":
            assert len(d.turns) in (4, 6) and len(d.told) == 2
            assert d.told[0][0] != d.told[1][0]                 # two DIFFERENT slots
        elif d.shape == "untold":
            assert len(d.turns) in (2, 4, 6) and d.value is None and d.statement_turn is None
        else:
            assert d.shape == "contrast" and len(d.turns) == 6
        if d.shape not in ("filler", "untold"):
            assert d.fillers == 0 and not (set(users) & fillers)
        assert d.turns[d.answer_turn][0] == "bot"
    # The probe sweeps distances 0, 1 and 2, and its decision rule reads distance
    # 2 against distance 0. A distance the generator never emits cannot be told
    # apart from a distance the model cannot reach, so BOTH are trained, evenly.
    filler = [d for d in dialogues if d.shape == "filler"]
    assert 0.45 < sum(d.fillers == 2 for d in filler) / len(filler) < 0.55


def test_told_answers_carry_the_asked_value_and_only_that_one(dialogues):
    asked_second = 0
    two_fact = [d for d in dialogues if d.shape == "two_fact"]
    for d in dialogues:
        if d.value is None:
            continue
        answer = d.turns[d.answer_turn][1]
        assert _has(answer, d.value), (d.turns, d.value)
        assert _has(d.turns[d.statement_turn][1], d.value)
        assert recall.UNTOLD_MARKER not in answer
        # The probe scores the VALUE, and prints `expected_answer` as what a pass
        # looks like; the canonical form must be one the generator really trains.
        forms = {recall._say(t, d.slot, d.value, d.aux) for t in recall.ANSWERS[d.slot]}
        assert answer in forms
        assert recall.expected_answer(d.slot, d.value, d.aux) in forms
    for d in two_fact:
        other = next(f for f in d.told if f[:2] != (d.slot, d.value))
        assert other[1] != d.value
        assert not _has(d.turns[d.answer_turn][1], other[1]), d.turns   # a swap is a miss
        asked_second += d.told[1][:2] == (d.slot, d.value)
    # Asking only about the most recent fact would make "say the last value" a
    # complete solution to the one shape that exists to defeat it.
    assert 0.45 < asked_second / len(two_fact) < 0.55


def test_acknowledgement_repeats_the_value_in_at_most_half(dialogues):
    def echoed(d) -> bool:
        acks = [t for i, (r, t) in enumerate(d.turns) if r == "bot" and i != d.answer_turn]
        return any(_has(a, value) for a in acks for _slot, value, _aux in d.told)

    told = [d for d in dialogues if d.told]
    rate = sum(map(echoed, told)) / len(told)
    assert rate <= 0.5
    assert rate > 0.3                       # and it does happen: the probe scores it
    # Per CONVERSATION, which is the pre-registered unit. Two independent draws
    # would pass the pooled bound above and still put a value in an
    # acknowledgement of most two-fact dialogues.
    for shape in ("single", "filler", "two_fact", "contrast"):
        sub = [d for d in told if d.shape == shape]
        assert sum(map(echoed, sub)) / len(sub) <= 0.5, shape
    assert recall.ACK_ECHO_FRACTION <= 0.5


def test_untold_dialogues_never_contain_a_value_for_the_asked_slot(dialogues):
    untold = [d for d in dialogues if d.shape == "untold"]
    fillers = {u for u, _b in recall.TRAIN_FILLERS}
    cold = 0
    for d in untold:
        legal = recall.VALUES[d.slot] + recall.HELDOUT_VALUES[d.slot]
        for _role, text in d.turns:
            assert not any(_has(text, v) for v in legal), (d.slot, d.turns)
        reply = d.turns[d.answer_turn][1]
        assert recall.UNTOLD_MARKER in reply and d.answer_turn == len(d.turns) - 1
        assert all(slot != d.slot for slot, _v, _a in d.told)
        cold += len(d.turns) == 2
        if d.fillers:
            # The probe's untold control at distances 1 and 2: filler exchanges,
            # then the question, and NO statement anywhere.
            users = [t for r, t in d.turns if r == "user"]
            assert not d.told and len(d.turns) == 2 + 2 * d.fillers
            assert set(users[:-1]) <= fillers and len(set(users[:-1])) == d.fillers
        else:
            assert len(d.turns) == 2 + 2 * len(d.told) and len(d.told) <= 1
    # Half after a fact about something else -- so "a fact is in context" is not
    # the cue for a value -- a quarter cold (the probe's control at distance 0)
    # and a quarter after fillers, so that "a filler is in context" is not the
    # cue either: in the `filler` shape it is ALWAYS followed by a told answer.
    assert (recall.UNTOLD_AFTER_FACT, recall.UNTOLD_COLD) == (0.5, 0.25)
    assert 0.20 < cold / len(untold) < 0.30
    assert 0.45 < sum(bool(d.told) for d in untold) / len(untold) < 0.55
    after = [d.fillers for d in untold if d.fillers]
    assert 0.20 < len(after) / len(untold) < 0.30
    assert 0.4 < after.count(2) / len(after) < 0.6
    told_after_filler = sum(d.shape == "filler" for d in dialogues)
    assert len(after) / (len(after) + told_after_filler) > 0.15


def test_my_and_your_are_contrasted_with_the_personas_own_lines(dialogues):
    persona = set(TOPICS["name"][1]) | set(TOPICS["favourites"][1])
    mine_first = 0
    contrast = [d for d in dialogues if d.shape == "contrast"]
    assert {d.slot for d in contrast} == set(recall.CONTRAST_SLOTS)
    for d in contrast:
        yours = [i for i, (r, t) in enumerate(d.turns)
                 if r == "user" and t in recall.YOUR_QUESTIONS[d.slot]]
        assert len(yours) == 1
        assert d.turns[yours[0] + 1][1] in persona          # "your" -> the persona line
        mine = d.turns[d.answer_turn - 1][1]
        assert mine in _instances(recall.TRAIN_QUESTIONS[d.slot], d.slot)
        assert "your" not in _words(mine) and mine not in recall.YOUR_QUESTIONS[d.slot]
        assert _has(d.turns[d.answer_turn][1], d.value)     # "my" -> the told value
        mine_first += d.answer_turn < yours[0]
    assert 0.4 < mine_first / len(contrast) < 0.6
    # Same words either side of the possessive, so the possessive is the cue.
    assert "what is my name?" in recall.TRAIN_QUESTIONS["name"]
    assert "what is your name?" in recall.YOUR_QUESTIONS["name"]


# ---------------------------------------------------------------------------
# dose
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def replay(dialogues, build_recall, tmp_path_factory):
    """The module's build packed, and the REAL sampler replayed over it."""
    out = tmp_path_factory.mktemp("recall_replay")
    assert build_recall.main(["--out-dir", str(out), "--conversations", str(N)]) == 0
    hazard = build_recall.window_hazard(str(out), "recall", dialogues, steps=150,
                                        batch_size=160, seq_len=256, align_frac=0.75,
                                        align_lookahead=1024)
    return out, hazard


def test_every_slot_is_asked_for_at_least_the_preregistered_floor(dialogues, replay):
    """Read from a replay of the sampler, NOT from characters / mean length.

    The arithmetic counts every dialogue whole. The sampler does not deliver
    that: the answer is a dialogue's last turn, and the dialogue a window's far
    edge truncates loses exactly its answer. The first version of this test
    asserted the floor against the arithmetic and passed while the replay had
    every pet name under it.

    The floor is held on the per-slot MEAN here because 150 steps cannot resolve
    a single value (a count near a hundred, scaled up); the per-value minimum on
    the default build is printed by `build_recall.py --hazard-steps` and quoted
    in `snnchat.recall`.
    """
    r = recall.DOSE_RECIPE
    assert (r["max_steps"], r["batch_size"], r["seq_len"]) == (14_000, 160, 256)
    assert r["mix_weight"] == 0.06 and recall.MIN_EXPOSURES == 600
    _out, hazard = replay
    dose = hazard["dose"]
    assert dose["seq_len_is_the_recipes"] and dose["min_exposures"] == 600
    assert dose["recipe_windows"] == 14_000 * 160 * 0.06
    assert dose["scale"] == pytest.approx(dose["recipe_windows"] / (150 * 160))

    mean = sum(recall.rendered_length(d.turns) for d in dialogues) / len(dialogues)
    upper = recall.expected_exposures(mean)
    seen = r["max_steps"] * r["batch_size"] * r["seq_len"] * r["mix_weight"] / mean
    for slot in recall.SLOTS:
        row = dose["per_slot"][slot]
        assert row["with_evidence"]["mean"] >= recall.MIN_EXPOSURES, (slot, row)
        assert row["with_evidence"]["mean"] < row["asked"]["mean"]
        # `expected_exposures` is what it says it is: the dialogue count, by a
        # route that does not call it, and an UPPER estimate of the real dose.
        asks = sum(d.slot == slot and d.value is not None for d in dialogues) / len(dialogues)
        # `asks` is a binomial share of N dialogues; four of its standard
        # deviations, which for the least-asked slot (`object`) is about 0.11.
        rel = 4 * ((1 - asks) / (len(dialogues) * asks)) ** 0.5
        assert upper[slot] == pytest.approx(seen * asks / len(recall.VALUES[slot]), rel=rel)
        per_value = dose["per_value"][slot]
        assert list(per_value) == list(recall.VALUES[slot])
        assert statistics.fmean(a for a, _e in per_value.values()) == \
            pytest.approx(row["asked"]["mean"])
        assert all(e <= a for a, e in per_value.values())
    # Over all values the sampler delivers well under the dialogue count. Pooled,
    # because `name` alone sits within a few per cent of its own upper estimate
    # (half its `contrast` answers are mid-dialogue, where no far edge cuts them)
    # and 150 steps cannot resolve that.
    delivered = sum(dose["per_slot"][s]["asked"]["mean"] * len(recall.VALUES[s])
                    for s in recall.SLOTS)
    assert delivered < 0.9 * sum(upper[s] * len(recall.VALUES[s]) for s in recall.SLOTS)


def test_the_replayed_dose_counts_exactly_the_answers_that_are_targets(dialogues, replay,
                                                                        build_recall):
    """`window_hazard`'s window edges, by a route that does not use them: mark
    every answer position in the stream and sum the marks over each window's
    TARGET slice, `offset + 1 .. offset + seq_len` inclusive."""
    from snnchat.data import ChatCorpus

    out, hazard = replay
    corpus = ChatCorpus(str(out))
    sampler = build_recall._RecordingSampler(corpus, {"recall": 1.0}, 160, 256, 0,
                                             align_frac=0.75, align_lookahead=1024)
    stream = sampler.arrays[0]
    answer, statement, _echo = build_recall._spans(dialogues)
    keep = answer < len(stream)
    is_answer = np.zeros(len(stream) + 1, dtype=np.int64)
    is_answer[answer[keep]] = 1
    starts_at = np.full(len(stream) + 1, -1, dtype=np.int64)
    starts_at[answer[keep]] = statement[keep]

    answers = cut = 0
    for step in range(150):
        sampler._windows(step)
        for o in sampler.last_offsets:
            targets = slice(int(o) + 1, int(o) + 256 + 1)
            answers += int(is_answer[targets].sum())
            here = starts_at[targets]
            cut += int(((here >= 0) & (here < o)).sum())
    assert (answers, cut) == (hazard["total"]["answers"], hazard["total"]["cut"])
    asked = sum(a for slot in recall.SLOTS for a, _e in hazard["dose"]["per_value"][slot].values())
    assert asked == pytest.approx(answers * hazard["dose"]["scale"])


def test_slot_weights_are_the_ones_the_dose_was_tuned_with(dialogues):
    # Pinned for the reason the shape shares are: the replayed dose in
    # `snnchat.recall` was measured under exactly these.
    assert recall.SLOT_WEIGHTS == {"name": 0.19, "pet": 0.24, "colour": 0.10,
                                   "food": 0.18, "animal": 0.20, "object": 0.09}
    plain = [d for d in dialogues if d.shape != "contrast"]
    for slot in recall.SLOTS:
        share = sum(d.slot == slot for d in plain) / len(plain)
        assert share == pytest.approx(recall.SLOT_WEIGHTS[slot], abs=0.012), slot
    contrast = [d for d in dialogues if d.shape == "contrast"]
    total = sum(recall.SLOT_WEIGHTS[s] for s in recall.CONTRAST_SLOTS)
    for slot in recall.CONTRAST_SLOTS:
        share = sum(d.slot == slot for d in contrast) / len(contrast)
        assert share == pytest.approx(recall.SLOT_WEIGHTS[slot] / total, abs=0.03), slot


def test_two_fact_pairs_the_two_slots_that_share_a_value_list(dialogues):
    """colour + object is the only pairing a model cannot solve by knowing which
    list the asked value is in. Under a uniform partner draw it was the rarest
    pairing there was."""
    two_fact = [d for d in dialogues if d.shape == "two_fact"]
    sharing = [d for d in two_fact if d.slot in ("colour", "object")]
    paired = [d for d in sharing if {f[0] for f in d.told} == {"colour", "object"}]
    # BINDING_PAIR_FRACTION, plus the uniform draw's one-in-five.
    assert recall.BINDING_PAIR_FRACTION == 0.5
    assert 0.5 < len(paired) / len(sharing) < 0.7
    assert len(paired) / len(two_fact) > 0.08
    for d in paired:
        assert d.told[0][1] != d.told[1][1]                     # two DIFFERENT colours
    # ... and only there: any other asked slot still draws its partner evenly.
    rest = [d for d in two_fact if d.slot not in ("colour", "object")]
    for slot in recall.SLOTS:
        eligible = [d for d in rest if d.slot != slot]
        share = sum(slot in {f[0] for f in d.told} for d in eligible) / len(eligible)
        assert share == pytest.approx(0.2, abs=0.03), slot


def test_a_joined_two_fact_acknowledgement_does_not_point_at_the_asked_fact(dialogues):
    """One user turn states both facts and ONE acknowledgement follows. If the
    echoing acknowledgement always repeated the asked value, "copy from the
    acknowledgement" would solve the shape that exists to defeat it -- and the
    pooled echo rate would not move."""
    asked = other = 0
    for d in dialogues:
        if d.shape != "two_fact" or len(d.turns) != 4 or not d.echo:
            continue
        ack = d.turns[1][1]
        mine = _has(ack, d.value)
        theirs = any(_has(ack, v) for s, v, _a in d.told if (s, v) != (d.slot, d.value))
        assert mine != theirs, d.turns                          # exactly one of the two
        asked += mine
        other += theirs
    assert asked + other > 500
    assert 0.42 < asked / (asked + other) < 0.58


def test_the_length_ceiling_redraws_the_wording_and_never_the_facts(monkeypatch):
    """What makes a per-slot dose a per-VALUE dose. If a too-long dialogue were
    redrawn whole, "whiskers" and "spaghetti" would be thinned out relative to
    "pip" and "ham" and the least-dosed value would be below the figure the
    arithmetic gives. Forced here rather than waited for: every new set of facts
    gets one artificially over-long phrasing first."""
    calls: list[tuple] = []
    forced: list[tuple] = []
    real = recall._phrase

    def too_long_once(rng, shape, asked, other, echo, fillers):
        args = (shape, asked, other, echo, fillers)
        calls.append(args)
        d = real(rng, shape, asked, other, echo, fillers)
        if not forced or forced[-1] != args:
            forced.append(args)
            d = dataclasses.replace(d, turns=d.turns + (("user", "x" * 300),))
        return d

    monkeypatch.setattr(recall, "_phrase", too_long_once)
    ds = recall.build_recall_dialogues(300, seed=1)
    assert len(forced) == len(ds) == 300 and len(calls) >= 600
    for d, (shape, asked, _other, _echo, fillers) in zip(ds, forced):
        assert (d.shape, d.slot, d.aux, d.fillers) == (shape, asked[0], asked[2], fillers)
        assert d.value in (asked[1], None)
        assert recall.rendered_length(d.turns) <= recall.MAX_RENDERED_CHARS


def test_values_are_decided_by_their_first_characters():
    for slot, width in (("name", 2), ("pet", 2), ("food", 2), ("animal", 2), ("colour", 3)):
        heads = [v[:width] for v in recall.VALUES[slot]]
        assert len(set(heads)) == len(heads), slot


# ---------------------------------------------------------------------------
# the committed probe
# ---------------------------------------------------------------------------

def _instances(templates, slot) -> set[str]:
    auxes = recall.AUX.get(slot, (None,))
    return {recall.fill(t, v, a) for t in templates for v in recall.VALUES[slot] for a in auxes}


def test_the_committed_probes_trained_frames_are_train_templates(dialogues):
    """`scripts/chat/memory_probe.py` stays byte-unchanged and its distance-0
    told rate is ship-candidate bar (i). Eight of its ten items are frames this
    source trains, in the probe's exact words and with in-list values; the town
    and the age are `HELDOUT_FRAMES` and stay untrained on purpose."""
    probe = _load_script("memory_probe")
    statements = {s for slot in recall.SLOTS
                  for s in _instances(recall.TRAIN_STATEMENTS[slot], slot)}
    questions = {q for slot in recall.SLOTS
                 for q in _instances(recall.TRAIN_QUESTIONS[slot], slot)}
    frames = {(s, q) for s, q, _m in recall.HELDOUT_FRAMES}
    trained = 0
    for establish, question, targets in probe.ITEMS:
        if establish in statements:
            assert question in questions, question
            assert any(t in v for t in targets for v in recall.VALUES.values())
            trained += 1
        else:
            assert any(question == q and re.fullmatch(s.replace("{v}", "[a-z]+"), establish)
                       for s, q in frames), establish
    assert trained == 8
    assert {u for u, _b in recall.TRAIN_FILLERS} >= set(probe.FILLERS)
    # The probe's distance-2 condition sends FILLERS[0] then FILLERS[1]. That
    # exact pair, in that order, is trained on both sides of the told/untold
    # pairing -- not merely each filler somewhere.
    for shape in ("filler", "untold"):
        between = {tuple(t for r, t in d.turns if r == "user")[-3:-1]
                   for d in dialogues if d.shape == shape and d.fillers == 2}
        assert tuple(probe.FILLERS[:2]) in between, shape


def test_expected_answer_is_the_canonical_form():
    assert recall.expected_answer("name", "elliot") == "Your name is Elliot."
    assert recall.expected_answer("pet", "rex", "dog") == "Your dog is called Rex."
    assert recall.expected_answer("object", "red", "bicycle") == "Your bicycle is red."
    assert recall.expected_answer("animal", "owl") == "Your favourite animal is the owl."
    assert recall.fill("i have {a} {v} {x}", "orange", "hat") == "i have an orange hat"
    with pytest.raises(ValueError):
        recall.expected_answer("pet", "rex")
    with pytest.raises(KeyError):
        recall.expected_answer("town", "ashby")


# ---------------------------------------------------------------------------
# scripts/chat/build_recall.py
# ---------------------------------------------------------------------------

def _fake_corpus(path: pathlib.Path) -> dict:
    """A packed directory with two sources already in it, one of them with
    provenance fields a careless rewrite would drop."""
    rng = np.random.default_rng(0)
    for name in ("alpaca", "persona"):
        body = rng.integers(4, 101, size=6000, dtype=np.uint8)
        body[::50] = BOS
        (path / f"{name}.bin").write_bytes(body.tobytes())
        (path / f"{name}.val.bin").write_bytes(body[:1500].tobytes())
    manifest = {
        "built_at": "2026-08-12T10:11:18", "vocab_size": 101, "vocab_version": 1,
        "sources": {
            "alpaca": {"chars_train": 6000, "chars_val": 1500, "reconstructed": True},
            "persona": {"chars_train": 6000, "chars_val": 1500, "conversations": 60000},
            "soda": {"chars_train": 123, "note": "listed, not present on this disk"},
        },
    }
    (path / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def _digests(path: pathlib.Path) -> dict[str, str]:
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(path.iterdir()) if p.name != "manifest.json"}


def test_build_recall_refuses_to_run_without_an_explicit_directory(build_recall, tmp_path,
                                                                   monkeypatch):
    monkeypatch.chdir(tmp_path)             # so a default, if one crept in, lands here
    with pytest.raises(SystemExit) as err:
        build_recall.main(["--conversations", "50"])
    assert "--out-dir" in str(err.value)
    assert list(tmp_path.iterdir()) == []


def test_build_recall_adds_a_source_and_drops_nothing(build_recall, tmp_path, capsys):
    from snnchat.data import ChatCorpus, MixtureSampler

    before = _fake_corpus(tmp_path)
    files_before = _digests(tmp_path)
    assert build_recall.main(["--data-dir", str(tmp_path), "--conversations", "3000",
                              "--seed", "5"]) == 0

    after = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert {k: after[k] for k in before if k != "sources"} == \
           {k: before[k] for k in before if k != "sources"}
    for name, entry in before["sources"].items():
        assert after["sources"][name] == entry              # merged, not replaced
    assert set(after["sources"]) == set(before["sources"]) | {"recall"}
    files_after = _digests(tmp_path)
    assert {k: files_after[k] for k in files_before} == files_before   # nothing repacked
    assert set(files_after) - set(files_before) == {"recall.bin", "recall.val.bin"}

    entry = after["sources"]["recall"]
    train = np.fromfile(tmp_path / "recall.bin", dtype=np.uint8)
    val = np.fromfile(tmp_path / "recall.val.bin", dtype=np.uint8)
    assert entry["conversations"] == 3000 and entry["seed"] == 5
    assert (entry["chars_train"], entry["chars_val"]) == (train.size, val.size)
    assert val.size >= 4096                                  # held out like the others
    # The file IS the generator's output: same ids, same order, train then val.
    tok = ChatTokenizer()
    ids = [i for conv in recall.build_recall_conversations(3000, seed=5)
           for i in tok.render_conversation(conv)]
    assert np.array_equal(np.concatenate([train, val]), np.asarray(ids, dtype=np.uint8))
    assert entry["report"]["rendered_chars"]["max"] <= recall.MAX_RENDERED_CHARS

    # ... and `snnchat.data` trains from it without complaint.
    capsys.readouterr()
    corpus = ChatCorpus(str(tmp_path), vocab_version=1)
    sampler = MixtureSampler(corpus, {"alpaca": 0.09, "persona": 0.08, "recall": 0.06},
                             8, 256, seed=0, align_frac=0.75, align_lookahead=1024)
    assert "recall" in sampler.mix_report()
    assert "WARNING" not in capsys.readouterr().out
    x, y = sampler.batch(0)
    assert tuple(x.shape) == (8, 256) and int(x.max()) < 101


@pytest.mark.parametrize("leftover", ["manifest", "bin", "val"])
def test_build_recall_never_overwrites_an_existing_source(build_recall, tmp_path, leftover):
    _fake_corpus(tmp_path)
    if leftover == "manifest":
        # `soda` is listed and has no file on this disk. Still not ours to take.
        name = "soda"
    else:
        # ... and the converse: a file the manifest has never heard of.
        name = "stray"
        (tmp_path / ("stray.bin" if leftover == "bin" else "stray.val.bin")).write_bytes(b"x")
    manifest_before = (tmp_path / "manifest.json").read_bytes()
    files_before = _digests(tmp_path)
    with pytest.raises(SystemExit) as err:
        build_recall.main(["--out-dir", str(tmp_path), "--name", name,
                           "--conversations", "3000"])
    assert "never overwrites" in str(err.value)
    assert (tmp_path / "manifest.json").read_bytes() == manifest_before
    assert _digests(tmp_path) == files_before
    assert "--force" not in (ROOT / "scripts" / "chat" / "build_recall.py").read_text(
        encoding="utf-8").split('"""', 2)[2]


def test_build_recall_refuses_a_second_pack_of_its_own_source(build_recall, tmp_path):
    assert build_recall.main(["--out-dir", str(tmp_path), "--conversations", "3000"]) == 0
    first = _digests(tmp_path)
    with pytest.raises(SystemExit):
        build_recall.main(["--out-dir", str(tmp_path), "--conversations", "3000",
                           "--seed", "1"])
    assert _digests(tmp_path) == first


def test_build_recall_refuses_a_manifest_from_another_vocabulary(build_recall, tmp_path):
    manifest = _fake_corpus(tmp_path)
    manifest["vocab_version"] = 2
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(SystemExit) as err:
        build_recall.main(["--out-dir", str(tmp_path), "--conversations", "3000"])
    assert "vocab" in str(err.value)
    assert not (tmp_path / "recall.bin").exists()


def test_spans_point_at_the_value_in_the_packed_stream(build_recall):
    """The hazard count is only as good as these positions, so they are checked
    against the tokenizer's own rendering rather than against the arithmetic
    that produced them."""
    ds = recall.build_recall_dialogues(400, seed=2)
    tok = ChatTokenizer()
    stream = np.asarray([i for d in ds for i in tok.render_conversation(list(d.turns))])
    answer, statement, echo = build_recall._spans(ds)
    told = [d for d in ds if d.value is not None]
    assert len(answer) == len(statement) == len(echo) == len(told)
    assert any(e >= 0 for e in echo) and any(e < 0 for e in echo)
    for d, a, s, e in zip(told, answer, statement, echo):
        n = len(d.value)
        assert tok.decode_visible(stream[a:a + n]).lower() == d.value
        assert tok.decode_visible(stream[s:s + n]) == d.value
        assert s < a
        if e >= 0:
            assert s < e < a and tok.decode_visible(stream[e:e + n]).lower() == d.value


def test_window_hazard_counts_cut_evidence_and_alignment_removes_it(build_recall, tmp_path):
    assert build_recall.main(["--out-dir", str(tmp_path), "--conversations", "3000"]) == 0
    ds = recall.build_recall_dialogues(3000, seed=0)
    kw = dict(steps=40, batch_size=32, seq_len=256, align_lookahead=1024)
    shipped = build_recall.window_hazard(str(tmp_path), "recall", ds, align_frac=0.75, **kw)
    aligned = build_recall.window_hazard(str(tmp_path), "recall", ds, align_frac=1.0, **kw)

    tot = shipped["total"]
    assert tot["windows"] == 40 * 32 and tot["answers"] > tot["windows"]
    assert tot["no_evidence"] <= tot["cut"] <= tot["answers"]
    # A window that starts on <|bos|> cannot have cut a statement off; one that
    # starts anywhere else often has. Both directions, so a flipped comparison
    # or a mislabelled row fails.
    rand, algn = shipped["by_window"]["random"], shipped["by_window"]["aligned"]
    assert 0.15 < rand["windows"] / tot["windows"] < 0.35
    assert rand["cut"] / rand["answers"] > 0.2
    assert algn["cut"] / algn["answers"] < 0.01
    assert aligned["by_window"]["random"]["windows"] == 0
    assert aligned["cut_rate"] < 0.01 < 0.05 < shipped["cut_rate"] < 0.25
    with pytest.raises(ValueError):
        build_recall.window_hazard(str(tmp_path), "recall", ds, align_frac=0.0, **kw)
