"""The exchanges that teach "my" from "your": the user states a fact, then asks for it back.

WHY THIS FILE EXISTS
--------------------
`scripts/chat/memory_probe.py` tells the shipped model a fact, asks for it back,
and scores the reply against a control that was never told. The committed
artifact (`experiments/chat/_quality/memory_probe.json`) reads 0/80 told and
0/80 untold at every distance, including distance 0, where the fact sits well
inside the 256-character training window. `docs/chat/QUALITY_v8.md` section 10
records the shape of the failure: asked "what is my name?" the model answers
about ITS OWN name, from `snnchat.persona` -- "I don't really have a name. I'm a
spiking neural network." The possessive is not resolved, so the question is
routed to the identity intent rather than to memory, and that section's closing
sentence is the hypothesis this file acts on: a corpus that taught the
distinction is cheaper than more reach.

Nothing in the corpus teaches it. `snnchat.persona` has no my-side template at
all and drills the your-side answer at 8 % of the mix; SODA's speakers state
facts about themselves but are never quizzed on them; Alpaca's prompts are
tasks. So the exchanges are written here, as templates, exactly as the persona
ones are -- and with the same caveat that file makes about itself: these
replies are *stipulated*. A model that passes a probe built from this file has
been taught a closed-vocabulary routine, not discovered memory.

THE GOAL IS CLOSED-VOCABULARY RECALL, AND THE HELD-OUT LISTS ARE HOW IT IS KEPT HONEST
--------------------------------------------------------------------------------------
The pre-registered goal is recall of TRAINED values inside one conversation.
Copying a value the model has never seen is a reported secondary and never a
gate. Three things are reserved here so that a probe can tell recitation from
recall, and the generator refuses to emit any of them (`_assert_clean` raises,
it does not warn):

* `HELDOUT_VALUES` / `HELDOUT_AUX` -- values disjoint from every trained list;
* `HELDOUT_STATEMENTS` / `HELDOUT_QUESTIONS` / `HELDOUT_FILLERS` -- phrasings of
  the trained frames that appear in no dialogue, reserved for
  `scripts/chat/memory_probe_v2.py`. The committed probe's phrasings are, on
  purpose, a subset of the TRAIN templates (its result is then a recitation
  floor), which is exactly why a second, unseen-phrasing stratum is needed;
* `HELDOUT_FRAMES` -- whole attributes (a town, an age, ...) that are never
  trained in any phrasing. Two of the committed probe's ten items are such
  frames and are left that way deliberately.

HOW THE VALUES WERE CHOSEN
--------------------------
A 4.4 M-parameter character model has to (a) hold which of N values it was told
and (b) spell it. (b) is made as cheap as the lists allow:

* short, common, lower-case-friendly words the rest of the corpus already
  spells -- TinyStories-register names, pets, foods and animals;
* within `name`, `pet`, `food` and `animal` no two trained values share their
  first two characters, so the reply is decided by its first two characters and
  the rest is ordinary spelling (checked at import by `_check_tables`; the
  colour list cannot satisfy it -- blue/black -- and is held to three);
* no value is a common English word of another part of speech where an
  alternative existed (no "will", "mark", "rose" among the names), because the
  probe's `hit` is a whole-word match and "I will" must not score as a name;
* every list is disjoint from every other list (the `object` slot's values ARE
  the colour list, by construction), and the pet species are disjoint from the
  favourite animals, so that in a two-fact dialogue a reply containing the
  OTHER fact's value is unambiguously a role swap;
* few enough per slot that each clears `MIN_EXPOSURES` under the pre-registered
  recipe -- see `expected_exposures`. The committed
  `experiments/chat/_quality/subject_frequency.json` is why the dose is counted
  in times ASKED: at matched exposure, being asked for predicts a hit and being
  read does not (`partial_asked_given_seen` against `partial_seen_given_asked`).

`elliot` is in the trained names on purpose: the owner's own demo must not be
the one name the model cannot say.

WHY EVERY DIALOGUE FITS IN 230 CHARACTERS
-----------------------------------------
`snnchat.data.MixtureSampler` aligns 75 % of windows (the shipped
`align_frac`) to a conversation start, and a window is 256 characters. A
dialogue longer than the window has its answer trained in a window that may not
hold its establishing turn -- which teaches the model to produce a name from
nothing, the opposite of the lesson. `MAX_RENDERED_CHARS` is therefore a hard
ceiling on the RENDERED length (markers included), enforced by redrawing the
phrasing, never the value, so the dose per value stays uniform.

The other 25 % of windows start at a random offset and some of those do cut an
establishing turn off (`MixtureSampler._windows` draws the alignment decision
once per batch row, for every source alike; there is no per-source setting).
That hazard is real and it is measured rather than argued. On the default
250,000-dialogue build (seed 0), replaying 2,000 steps of the real sampler at
B160 x L256, `align_frac` 0.75, `align_lookahead` 1024:

    trained answers whose establishing value is outside the window
        52,686 / 424,599 = 0.1241
    ... with no value-repeating acknowledgement inside it either
        49,061 / 424,599 = 0.1155
    of the 52,686, in random-offset windows: 52,685 (of 126,814 answers there)

    python scripts/chat/build_recall.py --out-dir <an empty dir> --hazard-steps 2000

The counts are over windows that overlap, so no interval is attached; the
figure is a description of the sampler, not an estimate compared with a bar.
Two things bound the damage without removing it. Every such answer lies BEFORE
its window's first `<|bos|>` (a dialogue is shorter than a window, so a cut
statement means the window opened inside that same dialogue), and no inference
context lacks a `<|bos|>`; whether the model uses that cue is not measured
here. And it is not fixable from this file: it needs per-source alignment in
`snnchat.data`.

WHY THE ACKNOWLEDGEMENT USUALLY DOES NOT REPEAT THE VALUE
---------------------------------------------------------
"Nice to meet you, Tom." puts the value twenty characters closer to the
question, and a model trained on it every time learns to copy from the
acknowledgement. At probe time the acknowledgement is the model's OWN sample,
which may hold no name or the wrong one. So the echo is drawn once per
conversation at `ACK_ECHO_FRACTION`, below one half, and in most dialogues the
only place the value exists is the user's turn.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass

from snnchat.persona import TOPICS as _PERSONA_TOPICS

__all__ = [
    "ACK_ECHO_FRACTION",
    "ANSWERS",
    "AUX",
    "CONTRAST_SLOTS",
    "DOSE_RECIPE",
    "HELDOUT_AUX",
    "HELDOUT_FILLERS",
    "HELDOUT_FRAMES",
    "HELDOUT_QUESTIONS",
    "HELDOUT_STATEMENTS",
    "HELDOUT_VALUES",
    "MAX_RENDERED_CHARS",
    "MIN_EXPOSURES",
    "RecallDialogue",
    "SHAPES",
    "SHAPE_SHARES",
    "SLOTS",
    "TRAIN_FILLERS",
    "TRAIN_QUESTIONS",
    "TRAIN_STATEMENTS",
    "UNTOLD_ANSWERS",
    "UNTOLD_MARKER",
    "VALUES",
    "YOUR_QUESTIONS",
    "build_recall_conversations",
    "build_recall_dialogues",
    "display",
    "expected_answer",
    "expected_exposures",
    "fill",
    "heldout_phrases",
    "rendered_length",
]

#: The attributes a user can state and ask back. `object` is the
#: object-attribute form the committed probe carries ("i have a red bicycle" /
#: "what colour is my bicycle?"): its VALUE is the colour and its auxiliary noun
#: is the object, exactly as `pet`'s value is the pet's name and its auxiliary is
#: the species.
SLOTS: tuple[str, ...] = ("name", "pet", "colour", "food", "animal", "object")

_NAMES = (
    "adam", "alice", "amy", "anna", "ben", "billy", "carl", "chloe", "david",
    "elliot", "emma", "eva", "fred", "george", "greg", "harry", "helen", "ian",
    "ivy", "jack", "jenny", "john", "julia", "kate", "kevin", "laura", "leo",
    "lily", "lucy", "matt", "megan", "mia", "molly", "nick", "noah", "oliver",
    "paul", "peter", "ron", "ryan", "sarah", "sophie", "steve", "tim", "tom",
    "victor", "wendy", "zoe",
)
_PETS = (
    "alfie", "archie", "bella", "bruno", "buddy", "charlie", "coco", "daisy",
    "dexter", "duke", "felix", "fluffy", "gizmo", "goldie", "jasper", "kiki",
    "lola", "luna", "max", "mittens", "monty", "murphy", "nala", "nemo",
    "ollie", "otis", "pip", "poppy", "ralph", "rex", "rocky", "rusty", "simba",
    "smokey", "teddy", "tilly", "toby", "whiskers", "winnie", "ziggy",
)
_COLOURS = (
    "red", "blue", "green", "yellow", "purple", "pink", "black", "white",
    "brown", "orange", "gold", "silver",
)
_FOODS = (
    "apples", "bananas", "beans", "bread", "burgers", "cake", "cheese",
    "cookies", "curry", "donuts", "dumplings", "eggs", "grapes", "ham",
    "ice cream", "jam", "lasagna", "lemons", "mango", "melon", "muffins",
    "noodles", "nuts", "olives", "pancakes", "peas", "pizza", "plums",
    "popcorn", "pretzels", "rice", "salad", "soup", "spaghetti", "stew",
    "sushi", "tacos", "toast", "waffles", "yogurt",
)
_ANIMALS = (
    "badger", "bear", "butterfly", "camel", "cow", "crocodile", "deer",
    "dolphin", "duck", "eagle", "elephant", "fox", "frog", "giraffe", "goat",
    "hedgehog", "hippo", "horse", "jaguar", "kangaroo", "koala", "lion",
    "llama", "monkey", "otter", "owl", "panda", "penguin", "pig", "rhino",
    "seal", "shark", "snake", "spider", "squirrel", "swan", "tiger", "whale",
    "wolf", "zebra",
)

#: slot -> the values the generator trains. `object` shares the colour tuple:
#: the thing to be recalled about "my bicycle" is its colour.
VALUES: dict[str, tuple[str, ...]] = {
    "name": _NAMES,
    "pet": _PETS,
    "colour": _COLOURS,
    "food": _FOODS,
    "animal": _ANIMALS,
    "object": _COLOURS,
}

_HELDOUT_COLOURS = ("grey", "violet", "navy", "lilac")

#: slot -> values NEVER emitted, in any turn of any dialogue, as a whole word.
#: They feed the probe's unseen-value stratum, which is reported and not gated.
HELDOUT_VALUES: dict[str, tuple[str, ...]] = {
    "name": ("oscar", "hannah", "simon", "nina", "frank", "derek", "abby", "wanda"),
    "pet": ("bandit", "marley", "bonnie", "hazel", "rufus", "pixie"),
    "colour": _HELDOUT_COLOURS,
    "food": ("pasta", "chocolate", "cereal", "sandwiches", "bacon", "pears"),
    "animal": ("mouse", "sheep", "lizard", "beaver", "goose", "leopard"),
    "object": _HELDOUT_COLOURS,
}

#: The auxiliary noun a slot's templates take as `{x}`: the pet's species, the
#: coloured object. The species are disjoint from `VALUES["animal"]` so that
#: "dog" in a reply is never a favourite-animal answer.
AUX: dict[str, tuple[str, ...]] = {
    "pet": ("dog", "cat", "rabbit", "hamster", "parrot", "pony", "turtle", "goldfish"),
    "object": ("bicycle", "car", "hat", "ball", "bag", "coat", "kite", "boat",
               "cup", "chair", "scarf", "umbrella"),
}

#: Auxiliary nouns never emitted, for the same stratum as `HELDOUT_VALUES`.
HELDOUT_AUX: dict[str, tuple[str, ...]] = {
    "pet": ("kitten", "puppy"),
    "object": ("jumper", "tent", "skateboard"),
}

#: slot -> the ways a user STATES the fact. `{v}` is the value, `{x}` the
#: auxiliary noun, `{a}` the indefinite article that agrees with the value
#: ("an orange hat"). Lower-case and unpunctuated: that is how the owner types
#: and how the committed probe asks, and an unpunctuated statement can be joined
#: to another with " and " for the one-turn form of the two-fact shape.
#:
#: The committed probe's statements for the six trained frames are all
#: instances of these (asserted by `tests/test_snnchat_recall.py`).
TRAIN_STATEMENTS: dict[str, tuple[str, ...]] = {
    "name": ("my name is {v}", "i am {v}", "i'm {v}", "call me {v}",
             "hi, my name is {v}", "hello, i am {v}"),
    "pet": ("i have a {x} called {v}", "i have a {x} named {v}",
            "my {x} is called {v}", "my {x}'s name is {v}", "my {x} is named {v}"),
    "colour": ("my favourite colour is {v}", "my favorite color is {v}",
               "{v} is my favourite colour", "i love the colour {v}",
               "i like {v} more than any other colour"),
    "food": ("my favourite food is {v}", "my favorite food is {v}",
             "my favourite thing to eat is {v}", "i love eating {v}",
             "i like {v} more than any other food"),
    "animal": ("my favourite animal is the {v}", "my favorite animal is the {v}",
               "the {v} is my favourite animal", "my favourite animal is {a} {v}",
               "i love the {v} more than any other animal"),
    "object": ("i have {a} {v} {x}", "my {x} is {v}", "i own {a} {v} {x}",
               "i have got {a} {v} {x}", "my new {x} is {v}"),
}

#: slot -> the ways a user ASKS for the fact back.
TRAIN_QUESTIONS: dict[str, tuple[str, ...]] = {
    "name": ("what is my name?", "what's my name?", "whats my name",
             "what is my name", "do you know my name?", "can you tell me my name?"),
    "pet": ("what is my {x} called?", "what's my {x} called?",
            "what is my {x}'s name?", "what is the name of my {x}?",
            "do you know my {x}'s name?"),
    "colour": ("what is my favourite colour?", "what's my favourite colour?",
               "what is my favorite color?", "whats my favourite colour",
               "do you know my favourite colour?", "which colour do i like most?"),
    "food": ("what is my favourite food?", "what's my favourite food?",
             "what is my favorite food?", "whats my favourite food",
             "do you know my favourite food?", "which food do i like most?"),
    "animal": ("what is my favourite animal?", "what's my favourite animal?",
               "what is my favorite animal?", "whats my favourite animal",
               "do you know my favourite animal?", "which animal do i like most?"),
    "object": ("what colour is my {x}?", "what color is my {x}?",
               "what colour is my {x}", "do you know what colour my {x} is?",
               "what colour {x} do i have?"),
}

#: slot -> statement phrasings RESERVED for `scripts/chat/memory_probe_v2.py`.
#: Never emitted. No reserved phrasing contains a trained one or is contained in
#: one (`_check_tables`), so "unseen phrasing" is not a trained phrasing with a
#: word in front of it.
HELDOUT_STATEMENTS: dict[str, tuple[str, ...]] = {
    "name": ("everyone calls me {v}", "i go by {v}", "{v} is my name"),
    "pet": ("i've got a {x} called {v}", "{v} is the name of my {x}",
            "we call our {x} {v}"),
    "colour": ("the colour i like best is {v}", "i think {v} is the best colour",
               "{v} has always been my colour"),
    "food": ("the food i like best is {v}", "i think {v} is the best food",
             "nothing beats {v} for me"),
    "animal": ("the animal i like best is the {v}",
               "i think the {v} is the best animal",
               "no animal is better than the {v}"),
    "object": ("i've got {a} {v} {x}", "the {x} i have is {v}",
               "i just bought {a} {v} {x}"),
}

#: slot -> question phrasings reserved for the probe. Never emitted.
HELDOUT_QUESTIONS: dict[str, tuple[str, ...]] = {
    "name": ("do you remember my name?", "what am i called?", "who am i?"),
    "pet": ("do you remember what my {x} is called?", "what did i name my {x}?",
            "who is my {x}?"),
    "colour": ("do you remember my favourite colour?", "which colour is my favourite?",
               "what colour do i like?"),
    "food": ("do you remember my favourite food?", "which food is my favourite?",
             "what food do i like?"),
    "animal": ("do you remember my favourite animal?", "which animal is my favourite?",
               "what animal do i like?"),
    "object": ("do you remember the colour of my {x}?", "which colour is my {x}?",
               "tell me the colour of my {x}"),
}

#: Attributes that are never trained in ANY phrasing: (statement, question,
#: substrings that must appear in no user turn). The first two are the committed
#: probe's items 7 and 8; they are left untrained so the probe has a stratum
#: that says whether anything beyond the drilled frames moved.
HELDOUT_FRAMES: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("i live in a town called {v}", "where do i live?", ("i live in", "where do i")),
    ("i am {v} years old", "how old am i?", ("years old", "how old")),
    ("my sister is called {v}", "what is my sister called?", ("sister",)),
    ("my favourite sport is {v}", "what is my favourite sport?", ("sport",)),
)

#: slot -> how the model answers a told question. The FIRST form is canonical
#: and is what `expected_answer` returns. Two forms, not six: the answer's job is
#: to carry the value, and every extra frame is dose taken from the binding.
#: Both forms share their opening "You" with `UNTOLD_ANSWERS`, so whether a fact
#: was told has to be decided by the third character of the reply at the latest.
ANSWERS: dict[str, tuple[str, ...]] = {
    "name": ("Your name is {v}.", "You told me your name is {v}."),
    "pet": ("Your {x} is called {v}.", "You told me your {x} is called {v}."),
    "colour": ("Your favourite colour is {v}.",
               "You told me your favourite colour is {v}."),
    "food": ("Your favourite food is {v}.", "You told me your favourite food is {v}."),
    "animal": ("Your favourite animal is the {v}.",
               "You told me your favourite animal is the {v}."),
    "object": ("Your {x} is {v}.", "You told me your {x} is {v}."),
}

#: Every untold answer contains this, so a probe can count "declined" replies
#: with one substring test instead of a list of templates.
UNTOLD_MARKER = "haven't told me"

#: slot -> how the model answers a question about a fact it was NOT told. This
#: is the half of the lesson the confabulation guard reads: without it, "answer
#: `what is my name?` with a name" is learnable with no memory at all.
UNTOLD_ANSWERS: dict[str, tuple[str, ...]] = {
    "name": ("You haven't told me your name yet.",
             "I don't know your name. You haven't told me."),
    "pet": ("You haven't told me about your {x} yet.",
            "I don't know. You haven't told me what your {x} is called."),
    "colour": ("You haven't told me your favourite colour yet.",
               "I don't know your favourite colour. You haven't told me."),
    "food": ("You haven't told me your favourite food yet.",
             "I don't know your favourite food. You haven't told me."),
    "animal": ("You haven't told me your favourite animal yet.",
               "I don't know your favourite animal. You haven't told me."),
    "object": ("You haven't told me about your {x} yet.",
               "I don't know. You haven't told me what colour your {x} is."),
}

#: Acknowledgements that REPEAT the value. Used in `ACK_ECHO_FRACTION` of
#: conversations; see the module docstring for why that is below one half.
_ACKS_ECHO: dict[str, tuple[str, ...]] = {
    "name": ("Nice to meet you, {v}.", "Hello, {v}!", "Hi {v}. Good to meet you."),
    "pet": ("{v} is a lovely name for a {x}.", "I bet {v} is a good {x}.",
            "Say hello to {v} for me."),
    "colour": ("{v} is a nice colour.", "I like {v} too.", "Ah, {v}. Good choice."),
    "food": ("Yum, {v} sounds good.", "I hear {v} is tasty.", "Mmm, {v}. Good choice."),
    "animal": ("The {v} is a great animal.", "I like the {v} too.",
               "Ah, the {v}. Good choice."),
    "object": ("{a} {v} {x} sounds nice.", "Your {v} {x} sounds lovely.",
               "I bet the {v} {x} looks good."),
}

#: Acknowledgements that carry no value, for any slot.
_ACKS_PLAIN: tuple[str, ...] = (
    "Okay, I'll remember that.", "Got it.", "Thanks for telling me.",
    "Good to know.", "That's nice to know.",
)
#: ... and two that only make sense after a name.
_ACKS_PLAIN_NAME: tuple[str, ...] = ("Nice to meet you.", "Hello! Nice to meet you.")

#: Probability that a conversation's acknowledgements repeat the value. ONE draw
#: per conversation, not one per acknowledgement: a two-fact dialogue has two
#: acknowledgements, and two independent draws at this rate would put a value in
#: some acknowledgement of most two-fact dialogues.
ACK_ECHO_FRACTION = 0.4

#: (user, bot) exchanges that carry no fact, placed between the statement and
#: the question in the `filler` shape. Short on purpose -- the dialogue has to
#: stay under `MAX_RENDERED_CHARS`. The first three user turns are the committed
#: probe's `FILLERS`, so its distance-1 and distance-2 conditions are phrased in
#: distribution even though this file only ever trains one filler exchange.
TRAIN_FILLERS: tuple[tuple[str, str], ...] = (
    ("how are you?", "I'm good, thank you."),
    ("that is interesting", "I'm glad you think so."),
    ("tell me something else", "I like short sentences."),
    ("how are you today?", "I'm running well, thanks."),
    ("ok", "Sure. What next?"),
    ("cool", "Thanks!"),
    ("what can you do?", "I can chat and tell short stories."),
    ("are you a robot?", "I'm a small spiking neural network."),
    ("thanks", "You're welcome!"),
    ("nice", "I'm glad."),
)

#: Filler user turns reserved for the probe. Never emitted.
HELDOUT_FILLERS: tuple[str, ...] = (
    "what is the weather like?", "do you like music?", "i am a bit tired today",
    "that sounds fun",
)

#: The slots `snnchat.persona` has a your-side answer for, which are therefore
#: the slots the `contrast` shape can be built on. The persona has no pet and
#: no bicycle.
CONTRAST_SLOTS: tuple[str, ...] = ("name", "colour", "food", "animal")

#: slot -> the YOUR-side question, answered with the persona's own line so that
#: this source reinforces `snnchat.persona` instead of fighting it.
YOUR_QUESTIONS: dict[str, tuple[str, ...]] = {
    "name": ("what is your name?", "what's your name?", "whats your name",
             "what is your name"),
    "colour": ("what is your favourite colour?", "what's your favourite colour?",
               "whats your favorite color"),
    "food": ("what is your favourite food?", "what's your favourite food?"),
    "animal": ("what is your favourite animal?", "what's your favourite animal?"),
}

#: The persona's replies, imported rather than copied: a second copy would drift
#: the first time someone edits the persona's name line.
_YOUR_ANSWERS: dict[str, tuple[str, ...]] = {
    "name": tuple(_PERSONA_TOPICS["name"][1]),
    "colour": tuple(_PERSONA_TOPICS["favourites"][1]),
    "food": tuple(_PERSONA_TOPICS["favourites"][1]),
    "animal": tuple(_PERSONA_TOPICS["favourites"][1]),
}

#: Dialogue shapes, in the order the shape draw walks them.
#:
#: * `single`   -- statement, acknowledgement, question, answer: the committed
#:   probe's distance 0.
#: * `filler`   -- the same with one short unrelated exchange in between.
#: * `two_fact` -- two facts from different slots (as two turns, or as one turn
#:   joined with " and "), then a question about ONE of them, first or second
#:   with equal probability. This is the only shape in which "say the value that
#:   is in context" is wrong half the time, so it is what teaches binding a value
#:   to its slot.
#: * `untold`   -- the question with no establishing fact, answered with an
#:   `UNTOLD_ANSWERS` line. Half are asked cold (the probe's control condition);
#:   half follow a fact about a DIFFERENT slot, so "a fact is in context" does not
#:   become the cue to produce a value.
#: * `contrast` -- a told fact, then "what is YOUR name" (persona line) and "what
#:   is MY name" (the told value) in either order: the distinction section 10 of
#:   QUALITY_v8 found unresolved, in one window.
SHAPES: tuple[str, ...] = ("single", "filler", "two_fact", "untold", "contrast")

#: Probability of each shape. A judgement, not a measurement. `single` is the
#: largest because it is the pre-registered primary; `untold` and `contrast`
#: carry no more than they need to keep the two guards (confabulation, persona
#: identity) trained, because every `untold` dialogue is dose the values do not
#: get.
SHAPE_SHARES: dict[str, float] = {
    "single": 0.35,
    "filler": 0.15,
    "two_fact": 0.20,
    "untold": 0.15,
    "contrast": 0.15,
}

#: Hard ceiling on a dialogue's RENDERED length: `<|bos|>` plus, per turn, a role
#: marker, the text and `<|eot|>`. See the module docstring.
MAX_RENDERED_CHARS = 230

#: The pre-registered recipe's terms in the dose arithmetic: the shipped
#: `experiments/chat/chat-v3d-aligned/config.json` values for the first three and
#: the round's mix weight for the fourth. They live here, not in the trainer,
#: because the value lists above are sized against them: change the recipe and
#: `expected_exposures` says whether the lists are still affordable.
DOSE_RECIPE: dict[str, float] = {
    "max_steps": 14_000,
    "batch_size": 160,
    "seq_len": 256,
    "mix_weight": 0.06,
}

#: The least number of times each trained value must be the ASKED-FOR answer
#: over one training run. Pre-registered by the design panel from the committed
#: `subject_frequency.json`, whose rarely-asked nouns score at the floor however
#: often they are read.
#:
#: Measured on the default build (250,000 dialogues, seed 0; mean rendered
#: length 137.5, which makes the recipe's share of this source 1.00 passes over
#: it): `expected_exposures` gives name 804, pet 730, food 965, animal 965,
#: colour 3,215, object 2,433 per value, and the least-asked single value in the
#: realised build is a pet name at 664. `scripts/chat/build_recall.py` prints
#: both for whatever it packs and records them in the manifest entry.
MIN_EXPOSURES = 600


@dataclass(frozen=True)
class RecallDialogue:
    """One generated dialogue with what a test or a probe needs to know about it.

    `turns` is what gets packed. Everything else is bookkeeping that could be
    recovered from the text but should not have to be: `slot`/`value`/`aux` are
    the ASKED fact (`value` is None for `untold`), `told` lists every fact a
    user turn states, `statement_turn` indexes the user turn that states the
    asked value (None for `untold`) and `answer_turn` the bot turn that answers
    the my-side question.
    """

    shape: str
    slot: str
    value: str | None
    aux: str | None
    turns: tuple[tuple[str, str], ...]
    told: tuple[tuple[str, str, str | None], ...]
    echo: bool
    statement_turn: int | None
    answer_turn: int


def _article(word: str) -> str:
    return "an" if word[:1] in "aeiou" else "a"


def fill(template: str, value: str, aux: str | None = None) -> str:
    """A USER-side template with its value, auxiliary noun and article filled in."""
    if "{x}" in template and not aux:
        raise ValueError(f"template {template!r} needs an auxiliary noun (species / object)")
    return template.format(v=value, x=aux or "", a=_article(value))


def display(slot: str, value: str) -> str:
    """How the BOT writes a value: names and pet names capitalised, the rest as typed.

    The user types "my name is elliot"; a reply of "Your name is elliot." would
    be the only lower-case proper noun in the whole corpus.
    """
    return value.capitalize() if slot in ("name", "pet") else value


def _say(template: str, slot: str, value: str, aux: str | None) -> str:
    """A BOT-side template filled in, with its first letter capitalised."""
    text = template.format(v=display(slot, value), x=aux or "", a=_article(value))
    return text[:1].upper() + text[1:]


def expected_answer(slot: str, value: str, aux: str | None = None) -> str:
    """The canonical reply to a told question: `ANSWERS[slot][0]`, filled in.

    For a probe to score against. A probe should score the VALUE (whole word,
    case-insensitive, as `memory_probe.hit` does) rather than demand this exact
    sentence -- the generator also trains `ANSWERS[slot][1]` -- but this is the
    sentence to print next to a transcript as "what a pass looks like".
    """
    if slot not in ANSWERS:
        raise KeyError(f"unknown slot {slot!r}; expected one of {SLOTS}")
    if slot in AUX and not aux:
        raise ValueError(f"slot {slot!r} needs aux (one of {AUX[slot]} or a held-out noun)")
    return _say(ANSWERS[slot][0], slot, value, aux)


def rendered_length(turns) -> int:
    """Length in ids of `ChatTokenizer.render_conversation(turns)`, without numpy.

    `<|bos|>`, then per turn a role marker, one id per character and `<|eot|>`.
    Exact only because every template and value is printable ASCII, which
    `_check_tables` enforces; the test suite checks it against the tokenizer.
    """
    return 1 + sum(len(text) + 2 for _role, text in turns)


def heldout_phrases() -> frozenset[str]:
    """Every reserved user turn, fully instantiated.

    Held-out templates are expanded over ALL values and auxiliary nouns, trained
    and held out alike, and compared by exact equality. A wildcard would be the
    obvious implementation and it is wrong: "{v} is my name" as a pattern
    matches the trained question "what is my name", with `v = "what"`.
    """
    out: set[str] = set(HELDOUT_FILLERS)
    for slot in SLOTS:
        values = VALUES[slot] + HELDOUT_VALUES[slot]
        auxes = (AUX[slot] + HELDOUT_AUX[slot]) if slot in AUX else (None,)
        for template in HELDOUT_STATEMENTS[slot] + HELDOUT_QUESTIONS[slot]:
            for aux in auxes:
                if "{v}" not in template and "{a}" not in template:
                    out.add(fill(template, "", aux))
                    continue
                for value in values:
                    out.add(fill(template, value, aux))
    for _statement, question, _markers in HELDOUT_FRAMES:
        out.add(question)
    return frozenset(out)


def _norm(template: str) -> str:
    return re.sub(r"[^a-z{} ]", "", template.lower()).strip()


def _check_tables() -> None:
    """Refuse to import with tables that break a promise the docstring makes."""
    for slot in SLOTS:
        for table in (VALUES, HELDOUT_VALUES, TRAIN_STATEMENTS, TRAIN_QUESTIONS,
                      HELDOUT_STATEMENTS, HELDOUT_QUESTIONS, ANSWERS, UNTOLD_ANSWERS,
                      _ACKS_ECHO):
            if slot not in table:
                raise AssertionError(f"slot {slot!r} missing from a template table")
        if set(VALUES[slot]) & set(HELDOUT_VALUES[slot]):
            raise AssertionError(f"{slot}: a value is both trained and held out")
        train = [_norm(t) for t in TRAIN_STATEMENTS[slot] + TRAIN_QUESTIONS[slot]]
        held = [_norm(t) for t in HELDOUT_STATEMENTS[slot] + HELDOUT_QUESTIONS[slot]]
        for h in held:
            for t in train:
                if h in t or t in h:
                    raise AssertionError(
                        f"{slot}: held-out phrasing {h!r} overlaps trained {t!r}")
        for answer in UNTOLD_ANSWERS[slot]:
            if UNTOLD_MARKER not in answer:
                raise AssertionError(f"{slot}: untold answer without {UNTOLD_MARKER!r}")
    if abs(sum(SHAPE_SHARES.values()) - 1.0) > 1e-12 or tuple(SHAPE_SHARES) != SHAPES:
        raise AssertionError("SHAPE_SHARES must cover SHAPES in order and sum to 1")

    # Disjoint across slots, so a reply holding another slot's value is a swap
    # and nothing else. `object` is exempt by construction: it IS the colours.
    seen: dict[str, str] = {}
    lists = [(s, VALUES[s] + HELDOUT_VALUES[s]) for s in SLOTS if s != "object"]
    lists += [(f"aux:{s}", AUX[s] + HELDOUT_AUX[s]) for s in AUX]
    for label, words in lists:
        for word in words:
            if word in seen:
                raise AssertionError(f"{word!r} is in both {seen[word]} and {label}")
            seen[word] = label
            if not all(0x20 <= ord(ch) < 0x7F for ch in word) or word != word.lower():
                raise AssertionError(f"{word!r} is not lower-case printable ASCII")

    # The first-characters promise. Two for the big lists; the colours cannot
    # (blue/black) and are held to three.
    for slot, width in (("name", 2), ("pet", 2), ("food", 2), ("animal", 2), ("colour", 3)):
        heads = [v[:width] for v in VALUES[slot]]
        if len(set(heads)) != len(heads):
            clash = sorted(h for h in set(heads) if heads.count(h) > 1)
            raise AssertionError(f"{slot}: values share their first {width} chars: {clash}")


_check_tables()

_HELDOUT_PHRASES = heldout_phrases()
_HELDOUT_WORDS = re.compile(
    r"\b(?:" + "|".join(sorted(
        {re.escape(w) for s in SLOTS for w in HELDOUT_VALUES[s]}
        | {re.escape(w) for s in HELDOUT_AUX for w in HELDOUT_AUX[s]}
    )) + r")\b"
)
_FRAME_MARKERS: tuple[str, ...] = tuple(m for _s, _q, ms in HELDOUT_FRAMES for m in ms)


def _assert_clean(turns) -> None:
    """Raise if a dialogue breaks a reservation or the length ceiling.

    Called on every dialogue, at generation time, rather than once in a test:
    the reservations are what the probe's held-out strata mean, and a generator
    edited later must not be able to void them silently.
    """
    n = rendered_length(turns)
    if n > MAX_RENDERED_CHARS:
        raise AssertionError(f"dialogue renders to {n} > {MAX_RENDERED_CHARS} chars: {turns}")
    for role, text in turns:
        if not all(0x20 <= ord(ch) < 0x7F for ch in text):
            raise AssertionError(f"turn is not printable ASCII: {text!r}")
        low = text.lower()
        found = _HELDOUT_WORDS.search(low)
        if found:
            raise AssertionError(f"held-out value {found.group(0)!r} emitted: {text!r}")
        if role != "user":
            continue
        for part in [text, *text.split(" and ")]:
            if part in _HELDOUT_PHRASES:
                raise AssertionError(f"held-out phrasing emitted: {part!r}")
        for marker in _FRAME_MARKERS:
            if marker in low:
                raise AssertionError(f"held-out frame marker {marker!r} emitted: {text!r}")


def _draw_fact(rng: random.Random, slot: str) -> tuple[str, str, str | None]:
    aux = rng.choice(AUX[slot]) if slot in AUX else None
    return (slot, rng.choice(VALUES[slot]), aux)


def _ack(rng: random.Random, fact: tuple[str, str, str | None], echo: bool) -> str:
    slot, value, aux = fact
    if echo:
        return _say(rng.choice(_ACKS_ECHO[slot]), slot, value, aux)
    plain = _ACKS_PLAIN + _ACKS_PLAIN_NAME if slot == "name" else _ACKS_PLAIN
    return rng.choice(plain)


def _phrase(rng: random.Random, shape: str, asked: tuple[str, str, str | None],
            other: tuple[str, str, str | None] | None, echo: bool) -> RecallDialogue:
    """Choose the WORDING of a dialogue whose shape, facts and echo are fixed.

    Split from the draw of the facts so that the length ceiling can redraw this
    and only this. Redrawing the value too would thin out the long values
    ("whiskers", "spaghetti") and the dose per value would stop being uniform.
    """
    slot, value, aux = asked
    statement = fill(rng.choice(TRAIN_STATEMENTS[slot]), value, aux)
    question = fill(rng.choice(TRAIN_QUESTIONS[slot]), "", aux)
    turns: list[tuple[str, str]] = []
    told: list[tuple[str, str, str | None]] = []
    statement_turn: int | None = 0

    if shape == "untold":
        statement_turn = None
        if other is not None:
            o_slot, o_value, o_aux = other
            turns += [("user", fill(rng.choice(TRAIN_STATEMENTS[o_slot]), o_value, o_aux)),
                      ("bot", _ack(rng, other, echo))]
            told.append(other)
        turns += [("user", question),
                  ("bot", _say(rng.choice(UNTOLD_ANSWERS[slot]), slot, "", aux))]
        return RecallDialogue(shape, slot, None, aux, tuple(turns), tuple(told), echo,
                              None, len(turns) - 1)

    answer = _say(rng.choice(ANSWERS[slot]), slot, value, aux)
    if shape == "two_fact":
        o_slot, o_value, o_aux = other
        o_statement = fill(rng.choice(TRAIN_STATEMENTS[o_slot]), o_value, o_aux)
        asked_first = rng.random() < 0.5
        first, second = (asked, other) if asked_first else (other, asked)
        s_first, s_second = (statement, o_statement) if asked_first else (o_statement, statement)
        told += [first, second]
        if rng.random() < 0.5:
            # A greeting cannot open the second clause ("... and hi, my name is
            # tom"). Dropping it leaves another trained template, not a new one.
            s_second = re.sub(r"^(?:hi|hello), ", "", s_second)
            turns += [("user", f"{s_first} and {s_second}"),
                      ("bot", _ack(rng, rng.choice((first, second)), echo))]
            statement_turn = 0
        else:
            turns += [("user", s_first), ("bot", _ack(rng, first, echo)),
                      ("user", s_second), ("bot", _ack(rng, second, echo))]
            statement_turn = 0 if asked_first else 2
    else:
        turns += [("user", statement), ("bot", _ack(rng, asked, echo))]
        told.append(asked)

    if shape == "filler":
        turns += [(role, text) for role, text in zip(("user", "bot"), rng.choice(TRAIN_FILLERS))]

    mine = [("user", question), ("bot", answer)]
    if shape == "contrast":
        yours = [("user", rng.choice(YOUR_QUESTIONS[slot])),
                 ("bot", rng.choice(_YOUR_ANSWERS[slot]))]
        if rng.random() < 0.5:
            turns += yours + mine
            answer_turn = len(turns) - 1
        else:
            turns += mine
            answer_turn = len(turns) - 1
            turns += yours
    else:
        turns += mine
        answer_turn = len(turns) - 1
    return RecallDialogue(shape, slot, value, aux, tuple(turns), tuple(told), echo,
                          statement_turn, answer_turn)


def _draw(rng: random.Random) -> RecallDialogue:
    u, shape = rng.random(), SHAPES[-1]
    for name in SHAPES:
        u -= SHAPE_SHARES[name]
        if u < 0.0:
            shape = name
            break
    slot = rng.choice(CONTRAST_SLOTS if shape == "contrast" else SLOTS)
    asked = _draw_fact(rng, slot)
    other = None
    if shape == "two_fact" or (shape == "untold" and rng.random() < 0.5):
        # `untold` never pairs `colour` with `object`: they share a value list,
        # and "my favourite colour is blue" followed by an untold bicycle would
        # put a legal bicycle answer in context. `two_fact` DOES pair them, with
        # different colours -- that is the hardest binding item there is.
        banned = {slot}
        if shape == "untold" and slot in ("colour", "object"):
            banned |= {"colour", "object"}
        other = _draw_fact(rng, rng.choice([s for s in SLOTS if s not in banned]))
        while other[1] == asked[1]:
            other = _draw_fact(rng, other[0])
    echo = rng.random() < ACK_ECHO_FRACTION

    for _ in range(1000):
        dialogue = _phrase(rng, shape, asked, other, echo)
        if rendered_length(dialogue.turns) <= MAX_RENDERED_CHARS:
            _assert_clean(dialogue.turns)
            return dialogue
    raise AssertionError(
        f"no phrasing of {shape} {asked} {other} fits in {MAX_RENDERED_CHARS} chars")


def build_recall_dialogues(n: int = 250_000, seed: int = 0) -> list[RecallDialogue]:
    """`n` dialogues with their bookkeeping. Deterministic in `(n, seed)`.

    Sequential draws from one `random.Random(seed)`, so the first k of a longer
    build are the k-dialogue build -- the same prefix property
    `build_persona_conversations` has, and what lets a test reason about a small
    `n` and a pack use a large one.
    """
    rng = random.Random(seed)
    return [_draw(rng) for _ in range(n)]


def build_recall_conversations(
    n: int = 250_000, seed: int = 0
) -> list[list[tuple[str, str]]]:
    """`n` dialogues as `[(role, text), ...]`, the structure
    `snnchat.persona.build_persona_conversations` returns and
    `ChatTokenizer.render_conversation` packs."""
    return [list(d.turns) for d in build_recall_dialogues(n, seed)]


def expected_exposures(mean_chars: float, recipe: dict[str, float] | None = None
                       ) -> dict[str, float]:
    """slot -> expected times EACH trained value is the asked-for answer in one run.

    The arithmetic the value lists are sized against, from the constants and one
    measured input:

        characters trained on   = max_steps * batch_size * seq_len
        of which this source    = ... * mix_weight
        dialogues seen          = ... / mean_chars        (rendered, measured)
        ... asking for `slot`   = ... * sum over told shapes of
                                        SHAPE_SHARES[shape] / (slots eligible)
        ... for one value       = ... / len(VALUES[slot])

    `untold` contributes nothing: it asks and no value is the answer. A
    two-fact dialogue counts once, for the fact that is ASKED -- the other is
    read, and read is not what predicts a hit. Dialogues cut by a window edge
    are counted whole, so this is an upper estimate by roughly the share of a
    window its last, truncated dialogue takes.
    """
    r = DOSE_RECIPE if recipe is None else recipe
    chars = r["max_steps"] * r["batch_size"] * r["seq_len"] * r["mix_weight"]
    dialogues = chars / mean_chars
    out: dict[str, float] = {}
    for slot in SLOTS:
        share = 0.0
        for shape in SHAPES:
            if shape == "untold":
                continue
            eligible = CONTRAST_SLOTS if shape == "contrast" else SLOTS
            if slot in eligible:
                share += SHAPE_SHARES[shape] / len(eligible)
        out[slot] = dialogues * share / len(VALUES[slot])
    return out
