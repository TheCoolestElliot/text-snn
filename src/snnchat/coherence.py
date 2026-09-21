"""Unintroduced-entity rate (UER@200): does a story talk about things it never set up?

WHAT IT IS, AND THE ONE FAILURE IT WAS BUILT FOR
------------------------------------------------
The chat model's stories drift: *"there was a pretty purple sheep. The pig was
very big... the penguin was very fit."* Until this module the package's only
statement about a reply's content was `snnchat.quality`'s topic column -- does
the reply contain the requested noun -- and `docs/chat/QUALITY_v8.md` §7 records
what that cannot see, in one of its own winning replies: *"Once upon a time,
there was a big oak tree. The boat was very happy." On topic, and not good.*

English marks that failure grammatically. A story introduces a thing with "a"
and refers back to it with "the"; a DEFINITE SUBJECT -- "The pig was ..." --
whose head noun has not occurred anywhere earlier in the reply is a reference to
something the reader was never given. This module counts replies that contain at
least one, inside a FIXED window of the reply's first `window` characters:

    UER@200 = (qualifying replies with >= 1 unintroduced definite subject)
              / (qualifying replies),    qualifying = len(reply) >= 200

IT IS NOT A COHERENCE MEASURE
-----------------------------
The file is called `coherence` because that is the question that motivated it.
The instrument is narrower than its filename and must be reported under its own
name, "unintroduced-entity rate (UER@200)", never as "coherence":

* It has NEVER been validated against a human rating. Nobody has checked that a
  reader finds a low-UER reply more coherent than a high-UER one.
* It is a countable proxy for ONE drift mechanism, in the sense
  `snnchat.quality`'s docstring uses the word: chosen so that a change in it is
  hard to produce by accident, not because it measures the thing itself.

WHAT IT CANNOT SEE -- READ THIS BEFORE QUOTING A NUMBER
-------------------------------------------------------
1.  **Interleaved threads.** Two complete stories shuffled sentence by sentence
    (A1, B1, A2, B2, ...) read as CLEAN, because each story introduces its own
    entities with "a" before it says "the". The reply is incoherent and the
    instrument is blind to it. This is pinned, not hoped away:
    `tests/test_snnchat_coherence.py::test_plain_interleave_is_not_detected_known_blindness`.
    What it does detect is the REPLACE mutation -- keep story A's odd sentences,
    substitute story B's even ones -- because that deletes introductions while
    keeping the references to them. `replace_alternate_sentences` and
    `interleave_sentences` below are those two mutations, kept in the module so
    the test and `scripts/chat/coherence_score.py` apply the identical ones.
2.  **Pronoun drift.** "He" changing referent mid-story has no definite noun
    phrase in it.
3.  **Name swaps.** "Tim" becoming "Tom" is two proper nouns and no "the".
4.  **Anything past the window.** A reply that holds together for 200 characters
    and falls apart at 250 scores clean.
5.  **Definites that are not subjects.** "She kicked the ball" is not counted:
    the pattern requires the noun phrase to be followed by one of a small closed
    list of story verbs. That is deliberate (object position is where English
    puts bridging definites -- "went to the park" -- that need no introduction)
    and it means the rate is a floor on the phenomenon, not a census of it.
6.  **Verbs outside the list, and noun phrases with more than two modifiers.**
7.  **An entity the PROMPT introduced.** "tell me a story about a pig" followed
    by "The pig was big." is a perfectly good definite, and by default it is
    flagged, because by default this module is given the reply alone. Pass
    `context=` to count the prompt's words as introduced. Both readings are
    legitimate and they answer different questions; say which one a number is.

8.  **A reply with NO definite subject in the window.** It scores clean, because
    there is nothing in it to flag -- so UER is a product of two things, how
    often a reply contains a definite subject at all (EXPOSURE) and how often
    such a reply has an unintroduced one, and it falls when either does. An arm
    that writes more formulaic openings ("Once upon a time, there was a little
    boy named Tim. Tim loved ...") has fewer "the X was" constructions in its
    first 200 characters and reads better without introducing one entity more
    carefully. This is not hypothetical: it is what the v14 subject tier did on
    the committed pools (`experiments/chat/_quality/v14_exploratory.json`,
    `P3_uer.decomposition`: exposure fell, the rate among exposed replies did
    not). And any selector that prefers likelier text lowers exposure, so UER is
    NOT independent of a per-character log-probability measured on the same
    picks. `uer_decomposition` reports the two factors and the per-subject rate
    separately; a UER difference between two arms is never quoted without them.

And it over-counts in one known way: a BRIDGING definite in subject position
("They went to school. The teacher was kind.") is flagged although no reader
would object. The corpus reference rate is mostly this, which is why a model's
rate is only ever meaningful NEXT TO the corpus rate from the same window and
the same stoplist, never alone.

THE WINDOW IS MANDATORY, NOT A DEFAULT TO BE RELAXED
----------------------------------------------------
A longer reply has more sentences and so more chances to contain one
unintroduced subject. Scored over whole replies, an arm that merely writes
SHORTER replies would look better purely by length -- the confound this package
has met repeatedly (`scripts/chat/topic_by_length.py` exists because of it). So
every reply is read over exactly its first `window` characters, and a reply
shorter than the window does not QUALIFY and is excluded from the denominator
rather than scored over less text. The excluded share is reported
(`qualifying_fraction`), because a rate over a shrinking subset is its own way
of flattering an arm.

THIS MUST NEVER BE USED INSIDE `snnchat.rerank`
-----------------------------------------------
The rule is absolute and it is the reason this module re-implements its tiny
whole-word matcher instead of importing `snnchat.rerank`'s. The echo partition
selects replies that contain the requested noun, and the battery's topic column
scores whether the reply contains the requested noun: once the selector and the
metric were the same function, a gain on that column became partly definitional
(`scripts/chat/echo_holdout.py`'s docstring, `docs/chat/QUALITY_v8.md` §4).
A reranker that filtered on unintroduced entities would do the same thing to
this instrument on the day it shipped -- the rate would fall to zero and mean
nothing. It measures selection rules; it is never one.

PURE PYTHON, AND THAT IS A CONTRACT
-----------------------------------
No torch, no numpy, nothing from `snnchat` that imports either
(`snnchat.rerank` and `snnchat.quality` both import torch, so the Wilson interval
is inlined here in the same shape as `snnchat.quality.wilson_interval` rather
than imported). A scorer of committed JSON should run on a machine with no GPU
wheel at all; `test_importing_coherence_does_not_import_torch` holds that up.

Not part of the research protocol. No number this module produces is a reported
figure.
"""
from __future__ import annotations

import re
from collections.abc import Iterable, Sequence

__all__ = [
    "WINDOW",
    "definite_subjects",
    "unintroduced_entities",
    "uer",
    "uer_decomposition",
    "anchored_topic",
    "wilson_interval",
    "split_sentences",
    "replace_alternate_sentences",
    "interleave_sentences",
]

#: The fixed reading window, in characters. Part of the instrument's NAME
#: (UER@200): a rate at another window is a different instrument and is not
#: comparable with one at this window.
WINDOW = 200

#: Two-sided 95 %. Same constant as `snnchat.quality.Z95`.
Z95 = 1.959963984540054

#: The closed list of verbs that make a noun phrase a story SUBJECT. Closed and
#: small on purpose: an open verb class needs a tagger, and a tagger is a model
#: whose errors would differ between the corpus and a small model's broken
#: output -- exactly the comparison this exists to make. These are the past-tense
#: narrative verbs of the TinyStories register. The list is the prototype's,
#: unchanged, so that the prototype's reading can be reproduced.
_VERBS = (
    "was|were|is|had|has|saw|said|went|wanted|loved|liked|felt|looked|came|ran|"
    "did|could|would|got|took|made|found|asked|smiled|laughed|cried|played|"
    "jumped|flew|lived|thought|knew|tried|started|decided"
)

#: Words that are never a head noun and never a modifier inside a noun phrase.
#:
#: WHY THIS EXISTS: THE PROTOTYPE'S FIRST DEFECT
#: ---------------------------------------------
#: The pattern is "the" + up to two words + HEAD + verb, and a regex cannot tell
#: a noun from any other word. On the corpus reference the prototype's most
#: common "unintroduced entity" was the word "and" -- "the hanger AND went",
#: where the real noun phrase had ended a word earlier -- followed by pronouns
#: reached through a subordinate clause ("the cupboard because IT was"). None of
#: those is an entity. They are excluded INSIDE the pattern (a negative
#: lookahead on every MODIFIER as well as on the head) rather than by filtering
#: the returned heads afterwards, because a head filter cannot see a function
#: word in modifier position: in "He opened the box and Tim was glad" the
#: unbarred pattern reads "the [box and] Tim was" as one noun phrase and reports
#: "tim", a perfectly good proper-noun subject of the NEXT clause. A noun phrase
#: does not span a conjunction, so the bar on the modifier ends it there. The
#: price is that a coordinated subject ("The boy and girl were") is not read at
#: all, which is one more way the rate is a floor.
#:
#: Admitted by CLASS (articles, conjunctions, pronouns, relativisers,
#: prepositions, quantifiers and adverbs), not by how often a word fired.
_FUNCTION_WORDS = frozenset("""
a an the
and or but so then when while because if as than though although until
that who whom whose which where what why how
i you he she it we they me him her us them his hers its their our your my
this these those there here
one ones all both each some any none every other another
in on at of to for with from by into onto over under up down out off about
not no very too just only also always never again still now soon even
""".split())

#: Nouns that are legitimately definite at FIRST mention, because the story world
#: holds exactly one of them and every reader already has it: "The sun was
#: shining" introduces nothing and needs no introduction.
#:
#: WHY THE LIST IS SHORT AND STAYS SHORT
#: -------------------------------------
#: This is the second half of the prototype's first defect ("sun" and "sky" were
#: its second and third most common corpus heads). It is also the place where
#: the instrument could be tuned until the corpus reads clean, and the contrast
#: with the model manufactured. So the list is admitted by KIND -- the sky and
#: what is in it, the weather, the ground underfoot, the time of day -- and NOT
#: by reading the corpus's flagged heads and deleting the frequent ones. Bridging
#: definites the corpus really does use without introduction ("the kids were",
#: "the town was", "the animals were") are deliberately NOT here: they stay
#: flagged, and they are most of the corpus reference's floor.
_UNIQUE_REFERENCE = frozenset("""
sun moon sky stars wind rain snow weather air ground world
day night morning evening
""".split())


def _subject_pattern(excluded: frozenset[str]) -> re.Pattern[str]:
    """ "the" + up to two modifiers + HEAD + story verb, with `excluded` barred.

    The modifiers are lazy, so the head is the word immediately before the verb
    -- the last word of an English noun phrase is its head, the same choice
    `snnchat.prime.prime_topic` and `snnchat.topics` make.
    """
    if excluded:
        alt = "|".join(sorted(excluded, key=len, reverse=True))
        mod_bar = rf"(?!(?:{'|'.join(sorted(_FUNCTION_WORDS, key=len, reverse=True))})\b)"
        head_bar = rf"(?!(?:{alt})\b)"
    else:
        mod_bar = head_bar = ""
    return re.compile(
        rf"\bthe\s+((?:{mod_bar}[a-z]+\s+){{0,2}}?){head_bar}([a-z]+)\s+(?:{_VERBS})\b",
        re.IGNORECASE,
    )


_SUBJECT = _subject_pattern(_FUNCTION_WORDS | _UNIQUE_REFERENCE)
#: The pattern with NO stoplist at all, modifiers included. Exists so the
#: stoplist's effect is a measured difference (`stoplist=False`) rather than a
#: claim, and so the prototype's pre-stoplist reading can be checked.
_SUBJECT_RAW = _subject_pattern(frozenset())


def _stems(word: str) -> tuple[str, ...]:
    """`word` and its plausible singulars, for a plural-tolerant lookup.

    "birds" -> ("birds", "bird"); "foxes" -> ("foxes", "foxe", "fox"). The junk
    stem is harmless: it is only ever matched as a whole word. A stem shorter
    than three letters is dropped, so "bus" never degrades to a search for "bu".
    "-ss" is not a plural ("grass", "dress").
    """
    out = [word]
    if word.endswith("s") and not word.endswith("ss"):
        out.append(word[:-1])
        if word.endswith("es"):
            out.append(word[:-2])
    return tuple(s for s in out if len(s) >= 3 or s == word)


def _occurs(word: str, low: str) -> bool:
    """Is `word` in `low` as a whole word, singular or plural, either direction?

    WHY THIS IS NOT `snnchat.rerank`'s MATCHER
    ------------------------------------------
    `snnchat.rerank` has a whole-word, plural-tolerant matcher already, and
    importing it would (a) import torch into a module whose contract is that it
    does not, and (b) couple this instrument to the selector it exists to
    measure -- a change to how the echo partition matches a noun would silently
    change what this module calls "introduced". Ten lines are cheaper than
    either. `low` must already be lower-cased.

    Tolerant in BOTH directions -- "a bird ... The birds were" and "two birds
    ... The bird was" are both introduced -- which the prototype was not: it
    looked up the head as written, so a plural head never matched its own
    singular introduction.
    """
    return any(
        re.search(rf"\b{re.escape(stem)}(?:s|es)?\b", low) is not None
        for stem in _stems(word)
    )


def definite_subjects(text: str, window: int = WINDOW, *, context: str = "",
                      stoplist: bool = True) -> list[tuple[str, bool]]:
    """`(head, introduced)` for EVERY definite subject in the window, in order.

    The census `unintroduced_entities` filters. It exists because a rate of
    flagged replies cannot tell "fewer unintroduced subjects" from "fewer
    subjects": see "WHAT IT CANNOT SEE" item 8 in the module docstring, and
    `uer_decomposition`. Arguments are `unintroduced_entities`'s.
    """
    pattern = _SUBJECT if stoplist else _SUBJECT_RAW
    low = text.lower()
    ctx = context.lower()
    out: list[tuple[str, bool]] = []
    for m in pattern.finditer(text):
        if m.end() > window:
            break
        head = m.group(2).lower()
        introduced = _occurs(head, low[: m.start()]) or bool(ctx and _occurs(head, ctx))
        out.append((head, introduced))
    return out


def unintroduced_entities(text: str, window: int = WINDOW, *, context: str = "",
                          stoplist: bool = True) -> list[str]:
    """Head nouns of definite subjects in `text[:window]` that were never set up.

    A DEFINITE SUBJECT is "the" + up to two modifiers + HEAD + a verb from the
    closed story-verb list. Its head is UNINTRODUCED if it has no earlier
    whole-word occurrence, singular or plural, anywhere in `text` before the
    "the" (nor in `context`). Returned lower-cased, in order of appearance. "The
    pig was big. The pig was pink." with no earlier pig returns ["pig"] once: the
    second mention has the first one before it.

    `window`    Only a subject that ENDS at or before character `window` counts.
                The pattern is run over the whole text and then cut, rather than
                over `text[:window]`, so a verb is never recognised in the first
                three letters of a longer word the cut happened to split.
    `context`   Text the reader saw BEFORE the reply -- the user's prompt. Its
                words count as introduced. Empty by default: see "WHAT IT CANNOT
                SEE" item 7 in the module docstring, and say which reading a
                number is.
    `stoplist`  False disables BOTH stoplists (function words and
                unique-reference nouns). For measuring what they do; never for a
                reported rate.

    What this cannot see is the module docstring's list and is not repeated
    here: pronoun drift, name swaps, interleaved threads, anything past the
    window, definites that are not subjects.
    """
    return [head for head, introduced in
            definite_subjects(text, window, context=context, stoplist=stoplist)
            if not introduced]


def wilson_interval(k: int, n: int, *, z: float = Z95) -> tuple[float, float]:
    """Wilson score interval. The same arithmetic as `snnchat.quality`'s.

    Inlined because `snnchat.quality` imports torch; the reasons for Wilson over
    the normal approximation, and for clamping the closed end against `p`, are
    documented there and apply unchanged. `n = 0` returns the whole interval.
    """
    if n <= 0:
        return (0.0, 1.0)
    p = k / n
    d = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / d
    return (min(max(0.0, centre - half), p), max(min(1.0, centre + half), p))


def uer(texts: Iterable[str], window: int = WINDOW, *,
        contexts: Sequence[str] | None = None, stoplist: bool = True) -> dict:
    """Unintroduced-entity rate over `texts`, reported the way a rate has to be.

    Returns
        n_total              every text given
        n                    the QUALIFYING ones, `len(text) >= window`
        qualifying_fraction  n / n_total -- report it next to the rate, always:
                             two arms with different reply lengths are scored on
                             different-sized and differently-selected subsets,
                             and this is the only place that shows
        k                    qualifying texts with >= 1 unintroduced entity
        rate                 k / n, or None when n == 0. None and not 0.0: no
                             qualifying reply is no measurement, and 0.0 would
                             read as a perfect score
        ci                   Wilson 95 % on k/n, `[low, high]`
        window               the window, so the artifact names its instrument

    THE INTERVAL ASSUMES INDEPENDENT REPLIES. 256 candidates drawn for one
    prompt are not independent, so on a rerank POOL the interval is too narrow
    and the yardstick for a difference between two trained models is the spread
    across training seeds, not this interval.

    `contexts`, if given, is one prompt per text (see `unintroduced_entities`).
    """
    texts = list(texts)
    if contexts is not None and len(contexts) != len(texts):
        raise ValueError(
            f"contexts has {len(contexts)} entries for {len(texts)} texts; "
            "it is one prompt per text"
        )
    n = k = 0
    for i, text in enumerate(texts):
        if len(text) < window:
            continue
        n += 1
        ctx = contexts[i] if contexts is not None else ""
        if unintroduced_entities(text, window, context=ctx, stoplist=stoplist):
            k += 1
    lo, hi = wilson_interval(k, n)
    return {
        "rate": (k / n) if n else None,
        "k": k,
        "n": n,
        "ci": [lo, hi],
        "qualifying_fraction": (n / len(texts)) if texts else None,
        "n_total": len(texts),
        "window": window,
    }


def _rate(k: int, n: int) -> dict:
    lo, hi = wilson_interval(k, n)
    return {"rate": (k / n) if n else None, "k": k, "n": n, "ci": [lo, hi]}


def uer_decomposition(texts: Iterable[str], window: int = WINDOW, *,
                      contexts: Sequence[str] | None = None,
                      stoplist: bool = True) -> dict:
    """UER split into the two things that move it, plus the per-subject rate.

    Over the QUALIFYING texts (`len(text) >= window`, as in `uer`):

        exposure                  replies with >= 1 definite subject in the
                                  window, introduced or not        / qualifying
        uer_given_exposed         replies with >= 1 UNINTRODUCED one / exposed
        unintroduced_per_subject  unintroduced definite subjects
                                                    / all definite subjects

    `uer = exposure * uer_given_exposed` EXACTLY -- a flagged reply is an exposed
    one, so the numerators and denominators cancel -- which is what makes this a
    decomposition and not three more numbers. A change in UER with
    `uer_given_exposed` and `unintroduced_per_subject` unmoved is a change in
    how often the arm writes "the X was" at all, and says nothing about whether
    entities are introduced (module docstring, "WHAT IT CANNOT SEE" item 8).

    Each is `{"rate", "k", "n", "ci"}` with a Wilson 95 % interval. The caveat
    on `uer`'s interval applies to all three, and `unintroduced_per_subject`
    has one more: two subjects in ONE reply are not independent draws, so its
    interval is too narrow even over independent replies.
    """
    texts = list(texts)
    if contexts is not None and len(contexts) != len(texts):
        raise ValueError(
            f"contexts has {len(contexts)} entries for {len(texts)} texts; "
            "it is one prompt per text"
        )
    n = exposed = flagged = subjects = unintroduced = 0
    for i, text in enumerate(texts):
        if len(text) < window:
            continue
        n += 1
        ctx = contexts[i] if contexts is not None else ""
        found = definite_subjects(text, window, context=ctx, stoplist=stoplist)
        missing = sum(1 for _head, introduced in found if not introduced)
        exposed += bool(found)
        flagged += bool(missing)
        subjects += len(found)
        unintroduced += missing
    return {
        "exposure": _rate(exposed, n),
        "uer_given_exposed": _rate(flagged, exposed),
        "unintroduced_per_subject": _rate(unintroduced, subjects),
        "n_total": len(texts),
        "window": window,
    }


# ---------------------------------------------------------------------------
# a stricter "on topic" than bare mention
# ---------------------------------------------------------------------------

def anchored_topic(text: str, noun: str) -> bool:
    """Is `noun` something the reply is ABOUT, rather than something it mentions?

    The battery's topic column asks whether the requested noun occurs at all,
    and a drifting story satisfies it in passing: "...there was a sheep. The pig
    was big... the PENGUIN was very fit" is a hit for "penguin". Two conditions,
    both required, both countable:

    1.  ANCHORED. The noun is introduced the way a story introduces its subject
        -- with "a"/"an" and at most two modifiers ("a little brown penguin") --
        or it occurs in the first sentence at all.
    2.  SUSTAINED. It occurs at least twice. A subject is referred back to.

    Whole-word and plural-tolerant, like everything else here. Kept this simple
    on purpose: it is a PROPOSED replacement for bare mention and has not been
    validated against a reader any more than UER has. Unlike UER it is not
    windowed, so it inherits the length confound -- a longer reply has more
    chances at a second mention. Compare it only between arms of matched reply
    length, or alongside `scripts/chat/topic_by_length.py`.

    The same rule as the module's applies: never inside `snnchat.rerank`.
    """
    low = text.lower()
    stems = _stems(noun.lower())
    alt = "|".join(re.escape(s) for s in stems)
    mentions = re.findall(rf"\b(?:{alt})(?:s|es)?\b", low)
    if len(mentions) < 2:
        return False
    if re.search(rf"\ban?\s+(?:[a-z]+\s+){{0,2}}?(?:{alt})(?:s|es)?\b", low):
        return True
    sentences = split_sentences(low)
    return bool(sentences) and _occurs(noun.lower(), sentences[0])


# ---------------------------------------------------------------------------
# the two validation mutations
# ---------------------------------------------------------------------------
#
# WHY THESE LIVE IN THE MODULE AND NOT IN THE TEST
# ------------------------------------------------
# An instrument nobody has tried to break is a number generator. These two
# mutations are how this one was broken: one it detects and one it provably does
# not. The test suite applies them to a committed fixture and
# `scripts/chat/coherence_score.py` applies them to the full corpus reference,
# and the two readings are only comparable if they are the same code.

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


def split_sentences(text: str) -> list[str]:
    """Split after `.`, `!` or `?` followed by whitespace. Crude and stable.

    A closing quote after the stop ('"Hello!" she said.') does not split, which
    keeps a quotation with its attribution. Good enough for a mutation that only
    needs sentence-sized units; not a tokenizer.
    """
    return [s for s in _SENTENCE_END.split(text.strip()) if s]


def replace_alternate_sentences(a: str, b: str) -> str:
    """Story `a` with every SECOND sentence replaced by story `b`'s at that index.

    Keeps a[0], a[2], ...; substitutes b[1], b[3], ... where `b` is long enough.
    The length stays close to `a`'s, so the mutant qualifies when `a` does, and
    introductions are DELETED while later references to them survive -- the
    shape of the model's real drift. This is the mutation UER detects.
    """
    sa, sb = split_sentences(a), split_sentences(b)
    return " ".join(
        sb[j] if (j % 2 == 1 and j < len(sb)) else sa[j] for j in range(len(sa))
    )


def interleave_sentences(a: str, b: str) -> str:
    """a[0], b[0], a[1], b[1], ... -- two complete stories shuffled together.

    Nothing is deleted, so every entity in either story still has its own
    introduction ahead of its first definite mention. UER does NOT detect this,
    and a reader would call it the less coherent of the two mutants. It is here
    so that blindness stays measured.
    """
    out: list[str] = []
    for x, y in zip(split_sentences(a), split_sentences(b), strict=False):
        out += [x, y]
    return " ".join(out)
