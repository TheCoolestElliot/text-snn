"""Short, self-contained topic-conditioned conversations.

WHY THIS EXISTS
---------------
`docs/chat/QUALITY.md` §4.1 measured the binding constraint on responsiveness and
it is not the model: **a reply can only learn to depend on a request that is
inside its own training window.** `stories_topic` conversations average 757
characters against a 256-character window, so at uniformly random offsets only
4.6 % of windows carry the request together with any of the story it conditions.

`MixtureSampler.align_frac` attacks that from the sampler side by nudging window
starts onto conversation boundaries, and it worked: topic propensity moved
0.0518 -> 0.0908 across window coverage 4.6 % -> 27 % -> 76 %, monotonically and
without flattening. But alignment buys coverage of the *request* only, and it
buys it at a price §3.2 of that document states plainly -- an aligned window
holds the first 256 characters of a 757-character story, so the other ~500 are
only ever seen in unaligned windows, with the request out of frame.

Count what that leaves. Of the characters of a reply that the model is asked to
produce, the fraction trained with the request actually present is about

    0.76 (aligned) x 215/700 (of the story that fits) ~ 23 %

The remaining three quarters of every story is still being learned as
unconditioned narrative. **That is the dose that has not been paid**, and
QUALITY.md §6 item 2 names the fix: shorter conversations, so coverage is a
property of the corpus rather than of a sampler flag.

WHAT THIS DOES
--------------
`short_story` truncates a TinyStories narrative at a sentence boundary to a
sampled target near 160 characters. A conversation is then

    <|bos|><|user|> a short story about a rabbit <|eot|><|bot|> ...161 chars... <|eot|>

about 200 characters end to end, which fits inside a 256-character window whole.
Two things follow, and the second is the one worth having:

1.  A conversation start falls inside almost every window, so the request is in
    frame **without** needing `align_frac` at all.
2.  When it is in frame, **all** of the reply is in frame with it. The share of
    reply characters trained against their own request goes from ~23 % to ~100 %
    of whatever the coverage is -- roughly a four-fold increase in the dose that
    the measured series says is the thing that works.

It also shortens the recurrent distance. This is a network with no context
window: the request survives in membrane state or not at all, and Phase 4 put
the two-compartment neuron's memory horizon at 47-48 characters. Asking it to
still be conditioned on a request 500 characters back is asking for something the
architecture is measured not to have. At 161 characters the whole reply sits
inside the range where context is measured to still pay (`docs/chat/README.md`
§1, bpc still falling at k = 1024, most of the fall by k = 256).

THE THREE JUDGEMENTS IN HERE, STATED AS JUDGEMENTS
--------------------------------------------------
*   **Truncation loses the ending.** A story cut after four sentences stops
    without resolving, and a model trained on it will stop without resolving. The
    alternative -- splicing the final sentence back on -- was tried on samples
    and rejected: TinyStories endings refer to objects introduced in the middle
    ("The shiny stone made everyone happy"), so splicing manufactures
    non-sequiturs, which is a worse thing to teach than brevity. `tinystories`
    and `stories_topic` stay in the mixture and both carry complete stories with
    real endings, so the corpus as a whole is not short-only.
*   **The target length is sampled, not fixed.** Uniform on [140, 235] rather
    than a constant, so the model learns "a short story is a few sentences and
    then it stops" instead of "a story is exactly four sentences".
*   **Roughly half the requests say so.** Phrasings that contain "short" or
    "little" are over-represented here and under-represented in the long
    sources, which makes reply length something the prompt can influence. That is
    a capability the model did not previously have, and it is also the reason
    this file does not simply reuse `topics.story_request`.

WHAT IS MEASURED AND WHAT IS NOT
--------------------------------
Measured on 20,000 stories before anything trained: mean kept length 161
characters (median 161, p90 207), and **88.9 %** of truncations still contain a
topic chosen from the full story, which is what `topic_for` requires before it
will ask for one. Whether any of this moves a reply is not measured here; that is
what an arm is for.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass

from snnchat.topics import _ABOUT, _GENERIC, _MIDSENTENCE_CAP, _article, topic_of

__all__ = [
    "Truncation", "SHORT", "SHORT300",
    "short_story", "topic_for", "short_request", "short_conversation",
]

#: Sentence boundary: terminal punctuation followed by whitespace. Deliberately
#: crude -- TinyStories has no abbreviations, no decimals and no ellipses, so the
#: failure modes a real sentence splitter exists for do not occur in this corpus.
#: A closing quote counts as terminal so dialogue turns are not split mid-line.
_SENTENCE = re.compile(r'(?<=[.!?"])\s+')

@dataclass(frozen=True)
class Truncation:
    """How long a kept story is allowed to be.

    ``lo``/``hi`` bound the uniformly sampled target; ``cap`` is the hard ceiling
    no kept story exceeds whatever its sentence lengths are. A single sentence
    longer than ``cap`` is truncated at a word boundary rather than dropped,
    because dropping it would silently bias the corpus against stories that open
    with a long sentence.

    This is a parameter rather than three module constants because
    `docs/chat/QUALITY_v4.md` §7 item 2 asks for a **second** source built by the
    same shaper at a longer target, so that the two differ in truncation length
    and in nothing else. Making it an argument is what keeps that claim true: the
    sentence splitter, the topic rule, the request phrasings, the raw fraction
    and the draw order are shared code, not copied code.
    """

    lo: int
    hi: int
    cap: int

    def __post_init__(self) -> None:
        if not 0 < self.lo <= self.hi <= self.cap:
            raise ValueError(f"need 0 < lo <= hi <= cap, got {self}")


#: The 2026-08-06 short-conversation round's target (`stories_short`,
#: `chat-v4a-short`). The upper end is chosen so that request + markers + reply
#: stays under 256 even at the tail: a request is ~34 characters and there are 4
#: marker ids, so 235 + 38 = 273 would overflow -- which is why the cap exists
#: and why p90 lands at 207 rather than 235.
SHORT = Truncation(lo=140, hi=235, cap=235)

#: The 2026-08-07 length round's target (`stories_short300`,
#: `chat-v5a-short300`). `QUALITY_v4.md` §7 item 2 asks for "~300 characters" and
#: says why: *"which would overlap the shipped arm's length distribution"*. So the
#: quantity that has to land on 300 is the **kept-length mean**, not the sampled
#: target's midpoint -- the shaper stops at the last sentence boundary *before*
#: the target, so kept runs 7-14 % under mid-target and choosing the midpoint
#: would overshoot. [242, 407] was picked by measuring six candidate ranges on
#: 20,000 stories and taking the one nearest 300 (298.5, median 298);
#: `docs/chat/PREDICTION_v5.md` §2 records that sweep, and it read no model.
#:
#: **This deliberately does not fit a 256 window**, which `SHORT` was built to do.
#: That is the manipulation, not an oversight -- see `PREDICTION_v5.md` §3 for the
#: coupling it forces and why the round accepts it.
SHORT300 = Truncation(lo=242, hi=407, cap=407)

#: Back-compatible aliases. `SHORT` is the default everywhere, so every caller
#: written before this parameter existed keeps its exact behaviour and
#: `stories_short.bin` repacks bit-identically from seed 11.
_TARGET_LO, _TARGET_HI = SHORT.lo, SHORT.hi
_HARD_CAP = SHORT.cap

#: Request phrasings that name the length. Kept separate from `topics._ABOUT`
#: rather than merged into it, so that "short" is a signal the model can learn
#: to act on: this source is ~50 % these phrasings and the long sources are 0 %.
_SHORT_ABOUT = (
    "tell me a short story about {}",
    "tell me a little story about {}",
    "a short story about {}, please",
    "can you tell me a short story about {}",
    "i want a short story about {}",
    "write me a short story about {}",
    "make up a little story about {}",
    "a quick story about {}?",
    "please tell me a short story about {}",
    "give me a short story about {}",
)
_SHORT_GENERIC = (
    "tell me a short story", "tell me a little story", "a short story please",
    "can you tell me a quick story", "i want a short story",
    "make up a short story", "tell me a short tale", "give me a short story",
)


def short_story(story: str, rng: random.Random,
                trunc: Truncation = SHORT) -> str:
    """The opening of `story`, cut at a sentence boundary near a sampled target.

    At least two sentences are always kept, even if that overshoots the target:
    a one-sentence "story" is not a story, and the alternative -- dropping those
    stories -- would bias the corpus toward whatever kind of story opens with two
    short sentences.
    """
    target = rng.randint(trunc.lo, trunc.hi)
    parts = [p.strip() for p in _SENTENCE.split(story) if p.strip()]
    if not parts:
        return ""
    kept: list[str] = []
    used = 0
    for part in parts:
        if kept and used + 1 + len(part) > target:
            break
        kept.append(part)
        used += len(part) + 1
    if len(kept) < 2 and len(parts) >= 2:
        kept = parts[:2]
    text = " ".join(kept)
    if len(text) > trunc.cap:
        # Cut at the last word boundary rather than mid-word; a character model
        # shown half a word learns that half words happen.
        cut = text.rfind(" ", 0, trunc.cap)
        text = text[: cut if cut > 0 else trunc.cap].rstrip()
    return text


def topic_for(full_story: str, kept: str) -> list[str]:
    """Topics of the whole story that survive into the kept opening.

    The topic is chosen from the **full** story, not from the truncation. That is
    the point: `topic_of` needs the word to recur to know it is the subject, and
    161 characters is rarely enough to see a word twice -- measured, choosing from
    the truncation alone yields a topic for 32 % of stories against 89 % this way.
    Requiring the chosen word to then appear in the kept text is what keeps the
    request honest, since a request for a story about something the reply never
    mentions is precisely the training signal this whole round exists to remove.
    """
    low = kept.lower()
    return [w for w in topic_of(full_story) if w in low]


def short_request(topics: list[str], names: list[str], rng: random.Random) -> str:
    """A user turn the short story is a plausible answer to.

    The proportions mirror `topics.story_request` -- mostly noun topics, some
    names so the capability that already worked is not trained away, a few
    two-topic requests, the rest generic -- with one addition: within each shape,
    a coin decides whether the phrasing names the length.
    """
    short_form = rng.random() < 0.5
    about = _SHORT_ABOUT if short_form else _ABOUT
    generic = _SHORT_GENERIC if short_form else _GENERIC
    r = rng.random()

    if topics and r < 0.60:
        w = topics[0]
        return rng.choice(about).format(f"{_article(w)}{w}")
    if names and r < 0.75:
        return rng.choice(about).format(names[0])
    if len(topics) >= 2 and r < 0.83:
        a, b = topics[0], topics[1]
        return rng.choice(about).format(f"{_article(a)}{a} and {_article(b)}{b}")
    if topics and r < 0.90:
        return rng.choice(about).format(f"{_article(topics[0])}{topics[0]}")
    return rng.choice(generic)


def short_conversation(story: str, rng: random.Random,
                       trunc: Truncation = SHORT) -> list[tuple[str, str]] | None:
    """`[(user, request), (bot, short story)]`, or None if the story is unusable.

    The random draws happen in a fixed order regardless of which branch is taken,
    for the same reason `MixtureSampler._windows` fixes its generator order: a
    packing whose stream position depends on the content of the story it just
    read cannot be reproduced from a seed alone. That property is what makes each
    source individually reproducible from its seed, and it is unaffected by
    `trunc`.

    **`trunc` does NOT make two sources story-for-story comparable, and an earlier
    version of this docstring wrongly claimed it did.** `rng.randint` draws a
    different number of Mersenne-Twister words for different range widths --
    `SHORT`'s width of 96 takes `getrandbits(7)` and `SHORT300`'s 166 takes
    `getrandbits(8)`, with different rejection probabilities -- so the two streams
    diverge after the first story and the *n*th story of `stories_short` is not
    the *n*th story of `stories_short300`. The claim is corrected here rather than
    deleted, and `docs/chat/PREDICTION_v5.md` §7 records that nothing in that
    round's design rested on it: the comparison is distributional over ~10^6
    conversations, not paired.
    """
    kept = short_story(story, rng, trunc)
    if len(kept) < 40:
        return None
    topics = topic_for(story, kept)
    names = [n for n in _MIDSENTENCE_CAP.findall(story[:200]) if n.lower() in kept.lower()]
    return [("user", short_request(topics, names, rng)), ("bot", kept)]
