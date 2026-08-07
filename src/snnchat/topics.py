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

import random
import re

__all__ = ["topic_of", "story_request", "STOP_WORDS"]

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


def story_request(story: str, rng: random.Random) -> str | None:
    """A user turn this story is a plausible answer to. None = leave it raw.

    The mixture of phrasings is a judgement and is recorded as one:

    * ~55 % ask for a NOUN topic, which is the capability that was missing;
    * ~20 % ask for a NAME, which is the one that already worked and must not be
      trained away;
    * ~10 % ask for two topics at once, so "about a dog and a ball" is not an
      unseen shape;
    * the rest stay generic, because "tell me a story" with no topic must keep
      working and is what most users type first.
    """
    topics = topic_of(story)
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
