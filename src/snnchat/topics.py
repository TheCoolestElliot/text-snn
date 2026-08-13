"""Turning a story into the request it is an answer to -- by its SUBJECT.

WHY THIS MODULE EXISTS, AND WHAT MEASUREMENT PRODUCED IT
--------------------------------------------------------
`docs/chat/RESULTS.md` records the shipped model's characteristic failure: asked
for "a story about a rabbit" it wrote a story about a bird. `scripts/chat/
quality.py` put a number on it -- **7.8 %** of topic-conditioned requests got a
reply containing the topic -- and, more usefully, showed the shape of the
failure per probe: "a little girl named Mia" landed 50 % of the time and every
common noun landed 0 %.

That split is not a coincidence, and it is not a capacity limit. It is what the
corpus taught. `build_corpus._story_conversation` builds a topic-conditioned
request by finding a capitalised word in the middle of a sentence
(`_MIDSENTENCE_CAP`), which in TinyStories is a CHARACTER NAME -- Lily, Tim,
Max. So the model was trained on thousands of examples of "a story about Lily"
answered by a story about Lily, and on **no** examples of "a story about a
rabbit" answered by a story about a rabbit. It learned the capability it was
shown, and was then measured on one it was not.

WHAT THIS DOES INSTEAD
----------------------
`topic_of` picks the story's subject the way a reader would: the content word it
keeps coming back to. No part-of-speech tagger and no curated noun list --
frequency within the story, a stop list of function words and narrative verbs,
and a requirement that the word occur at least twice and start early enough to
fall inside a training window.

Frequency rather than a lexicon is the deliberate choice. A curated noun list
would cap the topics the model can learn at the size of the list and would bake
this file's author's idea of what a story is about into 700 MB of training data.
Counting words in the story cannot do that: the topic vocabulary is whatever
TinyStories talks about, which is the distribution the model has to serve.

THE HONEST CAVEAT
-----------------
The probe that measures this asks for stories about common nouns, and this
module manufactures training data of exactly that form. The *form* is the thing
being taught and it is the form `docs/chat/README.md` was already advertising
before any of this work started -- "tell me a story about a rabbit" is the
canonical example in the shipped documentation. But the specific topics come
from the corpus, not from the probe list, and nothing here was written by
reading which probes failed. A reader who wants the strict version of the
result should read the per-probe table in `docs/chat/QUALITY.md`, where topics
absent from TinyStories (a robot, a pirate) are reported separately from topics
common in it.
"""

from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass
from pathlib import Path

__all__ = ["topic_of", "story_request", "STOP_WORDS",
           "SubjectPolicy", "POLICIES", "order_subjects"]

#: Function words, plus the verbs and adjectives TinyStories uses in almost
#: every story. Without the second group the most frequent content word in a
#: story is "said" or "played" and every prompt would be "a story about play".
STOP_WORDS: frozenset[str] = frozenset(
    """
    the and but for with from that this these those there here then than when
    while because before after into onto over under about again very really
    just also too all some any each every both other another none
    was were are was been being have has had having will would can could
    should shall must may might did does done doing
    say says said saying tell tells told telling ask asks asked asking
    answer answers answered reply replies replied
    want wants wanted like likes liked love loves loved need needs needed
    go goes went going come comes came coming get gets got getting give gives
    gave giving take takes took taking see sees saw seeing look looks looked
    looking find finds found finding make makes made making know knows knew
    play plays played playing run runs ran running walk walks walked walking
    put puts putting keep keeps kept let lets letting try tries tried trying
    think thinks thought start starts started help helps helped feel feels felt
    day days time times little big small good bad happy sad new old nice
    one two three four five many much more most lot lots
    her his its their our your my mine hers ours yours
    she he they them him you they what which who whom whose how why where
    mom mommy mum dad daddy papa mama boy girl man woman kid kids child children
    friend friends home house name named called once upon end
    not now soon still even ever always never together back away down around
    hop hops hopped jump jumps jumped fly flies flew swim swims swam eat eats
    ate drink drank sleep slept cry cries cried laugh laughs laughed smile
    smiles smiled shout shouted sing sang dance danced climb climbed hide hid
    share shared learn learned decide decided remember forgot forget understand
    promise promised thank thanked agree wonder wondered hope hoped wish wished
    dream dreamed work worked live lived stay stayed stop stopped wait waited
    watch watched listen listened hear heard touch touched taste tasted smell
    smelled push pushed pull pulled throw threw catch caught carry carried
    open opened close closed build built fix fixed clean cleaned wash washed
    cook cooked bake baked draw drew paint painted read write wrote count
    counted buy bought sell sold use used become became seem seemed show showed
    turn turned bring brought leave left move moved fall fell sit sat stand
    stood talk talked meet met call calls
    great best better worse worst fun funny silly pretty shiny soft fluffy
    scary brave kind friendly angry hungry tired excited amazing beautiful
    colorful colourful bright dark loud quiet warm cold hot wet dry dirty
    strong weak fast slow deep high tall short round sharp heavy curious
    gentle proud careful special favorite favourite magic magical delicious
    sweet sour salty fresh young wild safe lucky grumpy jealous worried afraid
    scared surprised sorry ready busy empty full broken lost tiny huge giant
    enormous normal weird strange mean rude polite honest wise clever smart
    lazy helpful useful unique perfect terrible awful wonderful smooth
    first last next same different only own real true sure right wrong
    """.split()
)

#: A word that has been preceded by one of these at least once is being used as
#: a noun. Two words of context is not a parser, but in TinyStories -- where
#: almost every noun phrase is `determiner noun` -- it is the difference between
#: "a story about a vase" and "a story about hopped".
_DETERMINERS = frozenset(
    "a an the his her their its my your our one two three four five some this "
    "that another each every other no".split()
)

#: A capitalised word that is not sentence-initial is almost always a character
#: name in TinyStories. Same expression `build_corpus` uses; kept here so the
#: two ways of asking for a story share one definition of "name".
_MIDSENTENCE_CAP = re.compile(r"(?<=[a-z,] )([A-Z][a-z]{2,11})\b")
_WORD = re.compile(r"[a-z]+")

#: Requests whose answer is a story about `{}`. Several phrasings, because the
#: model is character-level and a single phrasing would be learned as a string
#: rather than as an intent.
_ABOUT = (
    "tell me a story about {}",
    "tell me a story about {}, please",
    "can you tell me a story about {}",
    "i want a story about {}",
    "make up a story about {}",
    "a story about {}?",
    "write me a little story about {}",
    "tell me a short story about {}",
    "please tell me a story about {}",
    "could you make up a story about {}",
    "story about {} please",
    "i'd like a story about {}",
)
_GENERIC = (
    "tell me a story", "tell me a short story", "can you tell me a story",
    "i want a story", "story please", "make up a story", "tell me a tale",
    "read me a story", "tell me a bedtime story", "tell me another story",
)

_VOWELS = "aeiou"


def _article(word: str) -> str:
    """`a`/`an`, or nothing for a word that already reads as plural."""
    if word.endswith("s") and not word.endswith("ss"):
        return ""
    return "an " if word[0] in _VOWELS else "a "


def topic_of(story: str, *, min_count: int = 2, window: int = 220,
             max_rank: int = 3) -> list[str]:
    """The content words this story keeps coming back to, best first.

    `window` is the reason this is not simply "the most frequent word": a topic
    the model is asked for must be one it can still be reading about while the
    request is inside its training window, so a word whose first mention is 900
    characters in is no use as a conditioning target however often it recurs.
    Only words first seen within `window` characters of the start are eligible.

    Two filters beyond the stop list, both from reading what the first version
    produced:

    * **Character names are excluded.** They are the most repeated word in
      almost every TinyStories story, so a pure frequency count returns "a
      story about a tim" every time -- and the name form is already handled,
      separately and correctly, by `story_request`.
    * **A topic must have been used as a noun.** At least one occurrence has to
      follow a determiner, which is what stops "a story about hopped".
    """
    lower = story.lower()
    names = {n.lower() for n in _MIDSENTENCE_CAP.findall(story)}
    counts: dict[str, int] = {}
    first: dict[str, int] = {}
    noun: set[str] = set()
    prev = ""
    for m in _WORD.finditer(lower):
        w = m.group(0)
        if len(w) >= 3 and w not in STOP_WORDS and w not in names:
            counts[w] = counts.get(w, 0) + 1
            first.setdefault(w, m.start())
            if prev in _DETERMINERS:
                noun.add(w)
        prev = w
    eligible = [
        w for w, c in counts.items()
        if c >= min_count and first[w] < window and w in noun
    ]
    eligible.sort(key=lambda w: (-counts[w], first[w]))
    return eligible[:max_rank]


#: How often each word is available as a subject, and how often the shipped rule
#: actually requests it. Built by `scripts/chat/subject_table.py` over the raw
#: corpus with `topic_of` itself, so the counts describe exactly the candidates
#: a packer chooses among. Absent file degrades every policy to `head`, which is
#: the shipped behaviour, rather than to nothing.
_FREQ_PATH = Path(__file__).with_name("subject_freq.json")
try:
    _blob = json.loads(_FREQ_PATH.read_text(encoding="utf-8"))
    _SUBJECT_COUNTS: dict[str, int] = _blob["counts"]
except (OSError, ValueError, KeyError):       # pragma: no cover - packaging guard
    _SUBJECT_COUNTS = {}


@dataclass(frozen=True)
class SubjectPolicy:
    """Which of a story's eligible subjects goes into the request.

    THE PROBLEM THIS EXISTS FOR
    ---------------------------
    `topic_of` returns up to three eligible subjects -- each already in-window,
    used as a noun, and not a character name -- and `story_request` has always
    taken `topics[0]`, the most frequent. TinyStories is about balls, birds and
    cats, so that is what the request slot fills with: measured over the raw
    corpus, **140 subjects carry 49.3 % of all requests**, and "penguin" is
    asked 151 times against "bird"'s 31,076.

    `QUALITY_v11.md` §1 measured why that matters. Whether this model can draft
    a reply about a noun is predicted by how often it was ASKED for the noun
    (partial rho +0.328, p = 0.042) and not at all by how often it has READ it
    (partial rho -0.027, p = 0.87). So the request slot is the lever, and its
    distribution is a choice this file makes rather than a property of the data.

    `alpha` is the exponent on inverse availability: a candidate is drawn with
    weight `1 / (1 + count)**alpha`. **`alpha = 0` is the shipped rule exactly**
    -- equal weights, and `order_subjects` short-circuits to the identity
    without drawing at all, so `stories_topic.bin` stays reproducible character
    for character.

    WHAT THIS CANNOT BREAK, AND WHY THAT IS THE SAFETY PROPERTY
    -----------------------------------------------------------
    Every candidate it chooses among has already passed `topic_of`'s filters, so
    a flattened request still names something the story is genuinely about --
    just less centrally. The arm's failure mode is therefore a *weaker*
    request-to-story correspondence, not a wrong one, and that is the thing to
    watch in the held-out numbers rather than nonsense requests.
    """

    name: str
    alpha: float = 0.0
    #: "head" keeps `topic_of`'s own order; "weighted" samples by
    #: `1/(1+count)**alpha`; "rarest" sorts ascending by availability and draws
    #: nothing. Only the middle one is stochastic.
    mode: str = "head"

    def weight(self, word: str) -> float:
        if self.alpha == 0.0:
            return 1.0
        return 1.0 / (1.0 + _SUBJECT_COUNTS.get(word, 0)) ** self.alpha

    def count(self, word: str) -> int:
        return _SUBJECT_COUNTS.get(word, 0)


#: Named rather than loose floats, for the reason `shortform._TARGETS` is named:
#: a packed corpus's manifest entry then records which pre-registered policy it
#: was built under, and a later reader does not have to infer it.
#:
#: BOTH NON-DEFAULT POLICIES ARE PARAMETER-FREE, WHICH IS WHY THERE ARE TWO
#: -----------------------------------------------------------------------
#: `inverse` is the gentler reading of "flatten": every candidate keeps some
#: chance, in proportion to how rare it is. `rarest` is the extreme: the least
#: available eligible subject always wins. There is a continuum between them
#: (`alpha = 2`, `alpha = 3`, ...) and it is deliberately not exposed, because
#: picking a value off it would be fitting a constant to the corpus.
POLICIES: dict[str, SubjectPolicy] = {
    "head": SubjectPolicy("head", 0.0, "head"),
    "inverse": SubjectPolicy("inverse", 1.0, "weighted"),
    "rarest": SubjectPolicy("rarest", 0.0, "rarest"),
}


def order_subjects(topics: list[str], rng: random.Random,
                   policy: SubjectPolicy | None = None) -> list[str]:
    """Reorder a story's eligible subjects under `policy`. Best first.

    `None` and `alpha = 0` both return the input unchanged **and draw no random
    numbers**. That is not an optimisation: `build_topic_stories.py` threads one
    generator through the raw-narrative gate and every request, so an extra draw
    here would shift the whole downstream stream and `stories_topic.bin` would
    no longer rebuild to the bytes eight committed checkpoints were trained on.
    `tests/test_snnchat.py::test_the_head_policy_draws_no_random_numbers` holds
    that.

    Weighted sampling WITHOUT replacement, so the two-topic branch still gets
    two distinct subjects and gets them under the same policy as the one-topic
    branch. Under a policy the draw count is `len(topics) - 1`, which is
    data-dependent -- expected, since a policy pack is a new corpus by
    construction and is packed under a new name.
    """
    if policy is None or policy.mode == "head" or len(topics) < 2:
        return list(topics)
    if policy.mode == "rarest":
        # Ascending availability, ties broken by `topic_of`'s own order so the
        # result is a pure function of the story. Draws nothing.
        order = sorted(range(len(topics)), key=lambda i: (policy.count(topics[i]), i))
        return [topics[i] for i in order]
    remaining = list(topics)
    out: list[str] = []
    while len(remaining) > 1:
        weights = [policy.weight(w) for w in remaining]
        total = sum(weights)
        if total <= 0.0:                       # every candidate unknown: keep order
            break
        target = rng.random() * total
        acc = 0.0
        idx = len(remaining) - 1
        for i, w in enumerate(weights):
            acc += w
            if target < acc:
                idx = i
                break
        out.append(remaining.pop(idx))
    out.extend(remaining)
    return out


def story_request(story: str, rng: random.Random, *,
                  policy: SubjectPolicy | None = None) -> str | None:
    """A user turn this story is a plausible answer to. None = leave it raw.

    The mixture of phrasings is a judgement and is recorded as one:

    * ~55 % ask for a NOUN topic, which is the capability that was missing;
    * ~20 % ask for a NAME, which is the one that already worked and must not be
      trained away;
    * ~10 % ask for two topics at once, so "about a dog and a ball" is not an
      unseen shape;
    * the rest stay generic, because "tell me a story" with no topic must keep
      working and is what most users type first.

    Read those percentages as the nominal reading of one `rng.random()` against
    cumulative thresholds, not as independent probabilities: the 0.85-0.90 slice
    is a *second* single-topic branch, so the realised noun share is **60 %**
    when a name is available and higher when one is not.

    `policy` chooses WHICH eligible subject fills the slot; the default is the
    shipped behaviour and is bit-reproducible. See `SubjectPolicy`.
    """
    topics = order_subjects(topic_of(story), rng, policy)
    names = _MIDSENTENCE_CAP.findall(story[:200])
    r = rng.random()

    if topics and r < 0.55:
        w = topics[0]
        return rng.choice(_ABOUT).format(f"{_article(w)}{w}")
    if names and r < 0.75:
        return rng.choice(_ABOUT).format(names[0])
    if len(topics) >= 2 and r < 0.85:
        a, b = topics[0], topics[1]
        return rng.choice(_ABOUT).format(
            f"{_article(a)}{a} and {_article(b)}{b}"
        )
    if topics and r < 0.90:
        w = topics[0]
        return rng.choice(_ABOUT).format(f"{_article(w)}{w}")
    return rng.choice(_GENERIC)
