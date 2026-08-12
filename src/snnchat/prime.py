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

__all__ = ["story_prime", "prime_topic"]

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


def story_prime(prompt: str) -> str | None:
    """The text to begin the bot turn with, or None to generate normally."""
    topic = prime_topic(prompt)
    if topic is None:
        return None
    return _TEMPLATE.format(topic=topic)
