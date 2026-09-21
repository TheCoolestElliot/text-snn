"""Start the model's turn inside the sentence the corpus would have used.

WHAT THIS IS, STATED PLAINLY FIRST
----------------------------------
This puts words in the model's mouth. Given "tell me a story about a penguin"
it begins the bot turn with "Once upon a time, there was a little penguin" and
lets the model continue from there. The topic then appears in the reply because
**this module wrote it**, not because the model produced it.

So the metric the rest of this package uses -- does the reply contain the
requested noun -- is 1.0 by construction here and is worthless as evidence. Any
honest measurement of priming has to score the CONTINUATION: whether the model
keeps writing about the thing after the prime ends, and at what cost in fluency.
`scripts/chat/prime_probe.py` does exactly that and reports nothing else.

WHY THIS PARTICULAR SENTENCE, AND NOT ONE THAT SOUNDS GOOD
----------------------------------------------------------
The prime is not invented; it is measured off the training corpus, so that the
model is continuing from a state its own data would have put it in. Over 4,319
topic-conditioned conversations decoded from `data/chat/stories_topic.bin`:

* the requested noun appears in the reply **99.5 %** of the time -- the corpus is
  not the problem, reproducing it is;
* its first occurrence sits at a median of 81 characters in (p10 33, p90 182),
  so it is not an opening word and a prime that puts it first is already off
  distribution;
* the eighteen characters before it are dominated by three frames --
  "..., there was a big X" (104), "...time, there was a X" (103) and
  "...here was a little X" (56).

`_TEMPLATE` is the second and third of those, which is the frame that both opens
a story and introduces the subject in one clause.

A prime is only produced for a request that actually asks for a story about
something. Anything else returns None and the turn is generated normally: this
must not fire on "hello", on "name three animals", or on a question.

Not part of the research protocol. No number here is a reported figure.
"""
from __future__ import annotations

import re

__all__ = ["story_prime", "prime_topic", "tier_subject"]

#: The corpus's own opening-and-introducing clause. Measured, see the module
#: docstring. The trailing space matters: the model continues mid-sentence.
_TEMPLATE = "Once upon a time, there was a little {topic}"

#: A request for a narrative. "tale" and "bedtime story" are in the corpus's
#: request vocabulary (`snnchat.build_corpus._STORY_PROMPTS_ABOUT`).
_STORY_RE = re.compile(r"\b(story|stories|tale)\b", re.I)

#: The subject of the request. Only the `about X` frame is accepted -- the
#: corpus builds its topic requests that way and a looser rule starts priming
#: "tell me a story" with whatever noun happens to be in the sentence.
_ABOUT_RE = re.compile(
    r"\babout\s+(?:a|an|the)?\s*([a-z][a-z\- ]{1,30}?)\s*(?:[,.!?]|$|\bplease\b)",
    re.I,
)

#: Words that are never the subject even when they follow "about".
_NOT_A_SUBJECT = frozenset("""
it that this them something anything everything nothing me you us him her
yourself myself life today tomorrow yesterday
""".split())


def prime_topic(prompt: str) -> str | None:
    """The noun a story was requested about, or None.

    Returns the LAST word of the captured phrase, so "a boy and his kite" gives
    "kite" and "a birthday cake" gives "cake" -- the head of an English noun
    phrase is its last word, and the corpus's own subject extractor
    (`snnchat.topics`) makes the same choice.
    """
    if not _STORY_RE.search(prompt):
        return None
    m = _ABOUT_RE.search(prompt)
    if not m:
        return None
    words = [w for w in re.findall(r"[a-z]+", m.group(1).lower()) if len(w) > 2]
    if not words:
        return None
    head = words[-1]
    if head in _NOT_A_SUBJECT:
        return None
    return head


#: Words that open a clause MODIFYING the noun before them: "a penguin who loves
#: fish", "a dragon named Pip". Everything from the first of these on describes
#: the subject and is not the subject, so `tier_subject` cuts the phrase there.
_MODIFIES_THE_SUBJECT = frozenset("""
who whom whose which that where when named called
""".split())

#: Words that mean the captured phrase is NOT one simple noun phrase, so its
#: last word cannot be trusted to be its head. Admitted by CLASS, the way
#: `snnchat.coherence._FUNCTION_WORDS` is, not by which prompts were tried:
#: conjunctions ("a penguin and make it funny", "a boy and his kite" -- two
#: candidates for the subject, or a second request), prepositions ("a day at
#: the beach" -- the grammatical head and the topical word are different nouns),
#: pronouns and possessives, a determiner in mid-phrase (a second noun phrase has
#: begun), auxiliaries (a clause with no relativiser: "a penguin is sad"), and
#: the trailing adverbs a request ends on ("... a penguin tonight").
_NOT_ONE_NOUN_PHRASE = frozenset("""
and or but so then nor yet
in on at of to for with from by into onto over under near inside outside behind
through without across around after before
it he she they him her his hers its their them my your our me you us we i
a an the this these those some any
is are was were be been being has have had can could will would do does did
now tonight too again instead also
""".split())


def tier_subject(prompt: str) -> str | None:
    """The one word `RerankParams.subject_tier` may build a tier on, or None.

    WHY THIS IS NOT `prime_topic`
    -----------------------------
    `prime_topic` returns the LAST word of everything after "about", which is
    the head when the phrase is one noun phrase and is something else entirely
    when it is not: "a penguin and make it funny" gives "funny", "a penguin who
    loves fish" gives "fish", "a dragon who lives in a cave" gives "cave". For a
    PRIME that is a poor opening sentence. For the subject tier it is worse than
    no subject at all, because the tier is built from this ONE word: every
    draft that says "funny" outranks every draft about a penguin, which is the
    defect the subject rule exists to remove, reintroduced for any request with
    a trailing clause. The weighted tier has no such failure -- "penguin" still
    carries the largest weight in the sum -- so the safe answer to "I am not
    sure what the subject is" is None, and None is the weighted tier
    (`snnchat.rerank.echo_tier`).

    So this is deliberately CONSERVATIVE, and wrong only toward None:

    1.  The phrase is cut at the first word that opens a clause modifying the
        noun (`_MODIFIES_THE_SUBJECT`): "a penguin who loves fish" -> "penguin".
    2.  If what is left contains any word of `_NOT_ONE_NOUN_PHRASE`, the answer
        is None. "a boy and his kite" is None here although `prime_topic` says
        "kite": there are two candidates and the tier takes exactly one.
    3.  Otherwise the head of what is left, exactly as `prime_topic` takes it.

    On a bare "story about a X" request -- every prompt of every probe list in
    `scripts/chat/echo_holdout.py` -- the two functions agree, and
    `tests/test_snnchat.py::test_the_tier_subject_is_the_prime_topic_on_every_bare_request`
    holds that, so nothing measured on those lists depends on which was called.

    WHAT IT STILL GETS WRONG: a closed word list cannot see an open-class word
    in the wrong place. "a story about a penguin quickly" gives "quickly". And a
    multi-word name gives its last word ("New York" -> "york"), which is harmless
    for a whole-word match -- a draft that names New York names "york" -- and
    would not be for a prime.

    `prime_topic` is left exactly as it was: `story_prime` and
    `scripts/chat/prime_probe.py` measured it as it is.
    """
    if not _STORY_RE.search(prompt):
        return None
    m = _ABOUT_RE.search(prompt)
    if not m:
        return None
    # EVERY word, including the one- and two-letter ones `prime_topic` drops:
    # "in", "at", "it", "or" and "is" are exactly the evidence being looked for.
    phrase: list[str] = []
    for w in re.findall(r"[a-z]+", m.group(1).lower()):
        if w in _MODIFIES_THE_SUBJECT:
            break
        phrase.append(w)
    if any(w in _NOT_ONE_NOUN_PHRASE for w in phrase):
        return None
    words = [w for w in phrase if len(w) > 2]
    if not words:
        return None
    head = words[-1]
    if head in _NOT_A_SUBJECT:
        return None
    return head


def story_prime(prompt: str) -> str | None:
    """The text to begin the bot turn with, or None to generate normally."""
    topic = prime_topic(prompt)
    if topic is None:
        return None
    return _TEMPLATE.format(topic=topic)
