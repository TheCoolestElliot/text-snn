"""Decoding that spends the model's spare batch capacity on being ON TOPIC.

WHY THIS EXISTS
---------------
`docs/chat/RESULTS.md` names the shipped model's failure exactly: it is *fluent*
without being *responsive*. It reliably produces well-formed English of the
right shape for the turn, and only sometimes English that is about the question.
"Tell me a story about a rabbit" produced a story about a bird.

That is not a sampling temperature problem. It is what a maximum-likelihood
decoder does when one reply is high-probability under the model *whatever* was
asked: a generic story, a greeting, the identity answer. Sampling from
`P(reply | prompt)` gives those replies a share of the mass proportional to how
generic they are, and generic is exactly what they are.

THE FIX, AND WHY IT IS ALMOST FREE HERE
---------------------------------------
Draw N replies instead of one and keep the reply that the prompt did the most
work for. Concretely, rank candidates by a per-character mutual-information
score

    score(y) = [ log P(y | prompt) - lambda * log P(y | no prompt) ] / len(y)

which is Li et al. (2016)'s anti-LM objective. `log P(y | no prompt)` is scored
under the identical model given an EMPTY user turn, so the second term measures
"how much would this reply have been said anyway". A generic story scores high
on both terms and nets out low; a reply that only makes sense as an answer to
*this* prompt keeps its score. lambda = 0 degenerates to plain best-of-N by
likelihood, which is a fluency filter and nothing more.

The reason this is worth doing on THIS model rather than being a generic trick:
`docs/chat/BUILD_NOTES.md` §1 measured the per-timestep scan to be latency-bound
at these widths -- ~26 us per timestep per layer, largely independent of batch
size up to `B*d ~ 200k`. At d=1024 that leaves a factor of ~200 of unused batch.
**So N candidates cost approximately what 1 candidate costs**, and the
measurement in `docs/chat/QUALITY.md` bears that out. A transformer would pay N
times for this; a recurrent spiking net at this width very nearly does not.

That was true of the SCAN and false of this module, until 2026-08-12. The loop
around the scan read `2n` values back from the device per character to ask which
rows had finished, at 36 us each, so the cost of a pool grew linearly in the
pool -- the one axis the argument above says is free. It has been paid off
(`docs/chat/QUALITY_v10.md` §1): one device read per character, and the whole
step captured as a CUDA graph in `snnchat.stepper`. A turn at n = 128 went from
4.21 s to 0.57 s with **bit-identical** output, which is what made 128 the
default rather than 32.

THE ECHO PARTITION, AND WHY THE SCORE ALONE LEAVES MOST OF THE POOL ON THE FLOOR
--------------------------------------------------------------------------------
The score above ranks by log-probability and nothing else, and measured on the
committed draws it is **at chance** for picking the on-topic candidate. Replayed
over `experiments/chat/_quality/chat-v3d-aligned.json`, an oracle that always
picks the best of the 8 candidates the model already drew scores 0.359 on the
`topic` probes where the shipped score picks 0.141. The pool is not the problem;
the selector is. 23 of 64 topic draws already contain a reply that mentions what
was asked for, and the score throws it away.

So candidates are additionally partitioned by how many DISTINCT content words of
the user's prompt they echo, highest tier first, with the score above breaking
ties inside the tier. "Content word" means a word that is not in a closed-class
function-word list -- see `_FUNCTION_WORDS`, which was written from English
grammar and not from the probe battery, a distinction `docs/chat/QUALITY_v8.md`
§3 measures rather than asserts.

A partition rather than an additive bonus, for the same reason `min_chars` is a
partition: the score is a mean log-probability per character, an echo count is a
count, and there is no principled exchange rate between them. A partition needs
no such rate and no tuning constant.

**When no candidate echoes any content word the partition is the identity** --
`tests/test_snnchat.py::test_echo_partition_is_the_identity_when_nothing_echoes`.

Read that narrowly. It does NOT say "a pool with no on-topic reply is decided as
before", because the request frame survives the function-word list: `tell`,
`story`, `write`, `want` and `make` are all content words, so a draft can enter a
nonzero tier without being about anything. Measured on
`experiments/chat/_quality/echo_holdout.json`, 16 of the 25 draws where the
selection changed had no topic hit under either rule -- the tier was decided by
frame vocabulary. Over the whole run those sideways moves cost nothing (+9/-0 on
the metric) but they are movement, not inertia.

THE SUBJECT TIER, AND WHY THE WEIGHTED SUM STILL LETS THE FRAME WIN
-------------------------------------------------------------------
The tier above is the maximum over the pool of the weighted SUM over *all* the
prompt's content words, and a sum can be won by two cheap words against one dear
one. The weights are in the comment on `_FREQ_PATH`: for "tell me a story about
a penguin" a draft that echoes `tell` (5.93) and `story` (5.20) carries 11.13
and OUTRANKS a draft that echoes only `penguin` (10.34). The inverse document
frequency separated the frame from the subject word by word; it did not stop the
frame from adding up.

Two consequences follow from the construction, neither of which needs a
measurement to state. When the top tier holds one draft the anti-LM score
decides nothing -- the tier has already decided -- so the reply is whichever
draft happened to collect the most frame vocabulary, chosen with no regard to
how well the model thinks it reads. And when no draft in the pool names the
subject at all, a frame-word tier still forms and still hands the turn to a
draft for saying "tell" and "story", which is evidence of nothing.

`RerankParams.subject_tier` replaces the tier, for a turn whose subject is known,
with a single question: does the returned text name the subject? If any draft
does, those drafts are the tier and the score chooses among them. **If none
does, the tier is the whole length-partitioned pool** -- not the frame-word tier
-- so that the score chooses among everyone, which is what it would have done
before the partition existed. The subject is supplied by the caller;
`ChatSession` takes it from `snnchat.prime.prime_topic`, which returns None for
anything that is not a "story about X" request, and None is the old rule.

Off by default, and off nests the old behaviour exactly:
`tests/test_snnchat.py::test_the_subject_tier_is_off_by_default_and_off_is_the_old_rule`.

ONE FUNCTION DECIDES THE TIER, EVERYWHERE
-----------------------------------------
`echo_tier` is that function. It reads `.echo_weight` and `.subject_hit` off
whatever it is handed and touches no tensor, so a replay script can call it on
stand-in objects built from a stored pool. That is deliberate: a scorer that
re-implements the partition by hand measures a selector that is only *believed*
to be the shipped one, and the two drift the first time this file changes.
`select` is everything `rerank` does after sampling, split out for the same
reason -- one pool can be selected from twice, under two rules, without drawing
it twice.

WHAT IT CANNOT DO
-----------------
It selects among the replies the model would have produced anyway. If every one
of N candidates is off topic, reranking returns the least-bad of N off-topic
replies. It buys no knowledge, no arithmetic, and no memory across turns -- and
the numbers in `docs/chat/QUALITY.md` show exactly that shape: topicality moves
a lot, the factual battery does not move at all.

The echo partition does not change that. It makes the selector able to FIND an
on-topic reply that was drawn; it cannot cause one to be drawn. Its measured
gain is therefore bounded by the oracle above, and on the arms where the pool is
emptier the gain is smaller.

Not part of the research protocol. No number here is a reported figure.
"""

from __future__ import annotations

import json as _json
import math as _math
import re
import re as _re
from dataclasses import dataclass
from pathlib import Path as _Path

import torch
import torch.nn.functional as F

# `_filter` and `_loop_penalty` are the sampler's own truncation and cycle
# breaker. Imported rather than reimplemented so that a candidate drawn here is
# drawn from the identical distribution the single-candidate path uses -- two
# copies of a nucleus filter that disagree by one index would make every
# comparison in docs/chat/QUALITY.md meaningless.
from snnchat.generate import SamplingParams, _filter, loop_penalty_target
from snnchat.tokenizer import BOS, BOT, EOT, USER

__all__ = [
    "RerankParams",
    "Candidate",
    "Steering",
    "sample_candidates",
    "score_under_prefix",
    "fill_null_scores",
    "null_prefix_ids",
    "trim_to_sentence",
    "final_ids",
    "prompt_content_words",
    "echo_count",
    "echoes",
    "echo_weight",
    "word_weight",
    "steer_word",
    "echo_tier",
    "select",
    "rerank",
]

_STOP_IDS = (EOT, USER, BOS)


# ---------------------------------------------------------------------------
# the echo partition
# ---------------------------------------------------------------------------

#: Words that carry grammar rather than subject matter. A prompt's content words
#: are what is left after removing these.
#:
#: THIS LIST IS CLOSED-CLASS ENGLISH AND NOTHING ELSE, ON PURPOSE
#: -------------------------------------------------------------
#: Determiners, pronouns, auxiliaries, modals, prepositions, conjunctions,
#: wh-words and a few degree adverbs. It contains no verbs of asking ("tell",
#: "write"), no task nouns ("story", "list") and no numerals -- none of the
#: vocabulary that the probe battery in `snnchat.quality` happens to use.
#:
#: That restraint is the point, and it was measured rather than assumed:
#: `docs/chat/QUALITY_v8.md` §3 replays three lists over the committed draws --
#: this one, this one plus request verbs, and this one plus the battery's own
#: task nouns -- and the tuned list is **worse** (`list` 0.100 against 0.233 on
#: the shipped arm) while the two untuned ones are identical to four decimals.
#: There was nothing to gain by tuning it, so it is not tuned, and a reader can
#: check that claim without trusting this comment.
_FUNCTION_WORDS = frozenset("""
a an the this that these those my your his her its our their whose
i me you he she it we us them him
is am are was were be been being do does did doing done have has had having
will would shall should can could may might must
of to in on at for with about from by as into onto over under after before
and or but if then than so because while when where what which who whom how why
not no nor none very just only also too more most some any all each every
there here
""".split())

#: Below this many characters a word is too short to be evidence of anything --
#: "cat" is kept, "at" would be noise even if it survived the list above.
_MIN_CONTENT_CHARS = 3

_WORD_RE = re.compile(r"[a-z]+")


def prompt_content_words(text: str) -> list[str]:
    """The subject-matter words of a user turn, deduplicated and sorted.

    Sorted rather than in prompt order so that the result is a set with a stable
    representation: nothing downstream depends on the order, and a stable one
    keeps a logged `echo_words` comparable between runs.
    """
    seen = {
        w for w in _WORD_RE.findall(text.lower())
        if len(w) >= _MIN_CONTENT_CHARS and w not in _FUNCTION_WORDS
    }
    return sorted(seen)


#: Word counts over 4,004,093 words of `stories_topic.bin`, used to weight an
#: echoed word by how surprising it is. Shipped as a file rather than computed at
#: import because `data/` is gitignored and a decoder must work on a fresh clone.
#:
#: WHY WEIGHTING, MEASURED
#: -----------------------
#: Counting DISTINCT echoed words treats "story" and "penguin" as equal, and a
#: large candidate pool then fills the top tier with drafts that echo the request
#: FRAME while being about nothing. That is not hypothetical: it is why the
#: partition saturates at 0.300 held-out while the oracle keeps climbing to
#: 0.450 (`QUALITY_v8.md` §13), and why 16 of the 25 draws where selection
#: changed had no topic hit under either rule.
#:
#: Inverse document frequency separates the two cleanly, and the separation is a
#: property of the corpus rather than a choice: tell 5.93, story 5.20, make 6.02,
#: want 6.27, little 5.05 against penguin 10.34, castle 8.86, turtle 9.13,
#: whale 9.70, lighthouse 15.20 (absent). No threshold is needed -- the sum is
#: simply dominated by the rare word.
_FREQ_PATH = _Path(__file__).with_name("word_freq.json")
try:
    _freq_blob = _json.loads(_FREQ_PATH.read_text(encoding="utf-8"))
    _WORD_COUNTS: dict[str, int] = _freq_blob["counts"]
    _WORD_TOTAL: int = int(_freq_blob["total_words"])
except (OSError, ValueError, KeyError):       # pragma: no cover - packaging guard
    _WORD_COUNTS, _WORD_TOTAL = {}, 1


def word_weight(word: str) -> float:
    """How much echoing `word` counts for. Inverse document frequency.

    Falls back to 1.0 for every word if the table is missing, which degrades the
    weighted tier to the unweighted count tier rather than to nothing.
    """
    if not _WORD_COUNTS:
        return 1.0
    return _math.log(_WORD_TOTAL / (1 + _WORD_COUNTS.get(word, 0)))


def echoes(word: str, lowered: str) -> bool:
    """Does `lowered` use `word`? Whole word, plural tolerated either way.

    ONE FUNCTION, BECAUSE TWO COPIES OF THIS DRIFTED ONCE ALREADY
    -------------------------------------------------------------
    `echo_count` and `echo_weight` must agree about WHICH words were echoed and
    differ only in what each is worth. They used to say so in a docstring and
    implement it twice.

    WHOLE WORD, NOT A PREFIX -- MEASURED, 2026-08-13
    ------------------------------------------------
    This was `re.search(r"\\b" + stem, ...)`, a prefix match, and the prefix was
    doing real damage on any prompt that is not a story request. `"name three
    animals"` has `name` as a content word, `\\bname` matches **"named"**, and
    `" named "` is the literal signature of the register this decoder is
    supposed to be avoiding -- `snnchat.quality._is_story` keys on that exact
    substring. So the echo tier promoted "Three years old girl named Emma" over
    "A dog, a cat, and a bird", because the story echoed the request verb and
    the correct answer echoed nothing.

    Measured on 72 non-narrative draws (`scripts/chat/lambda_dodge.py`): the
    partition took `story_dodge` from 0.0000 to 0.2778, and `name` was the
    echoed word in **16 of the 20 dodges**, at an IDF weight of 8.66 -- which
    puts a request verb in the same weight band as a subject noun.

    The `(s|es)?` suffix keeps every behaviour the prefix rule was documented
    for: "rabbit" is still found in "rabbits" and "dragon" in "dragons". What it
    stops is an unrelated inflection of a request verb counting as subject
    matter. No constant was tuned to get this.
    """
    stem = word[:-1] if len(word) > 4 and word.endswith("s") else word
    return _re.search(r"\b" + _re.escape(stem) + r"(s|es)?\b", lowered) is not None


def echo_weight(text: str, words) -> float:
    """Summed `word_weight` of the DISTINCT prompt words the reply uses.

    The weighted counterpart of `echo_count`, and the quantity the tier is
    actually built on.
    """
    if not words:
        return 0.0
    lowered = text.lower()
    return sum(word_weight(w) for w in words if echoes(w, lowered))


def echo_count(text: str, words) -> int:
    """How many DISTINCT words of `words` the reply uses.

    Distinct, not total: a reply that says "boat" nine times is about a boat
    exactly as much as one that says it once, and counting repeats would hand
    the tier to whichever candidate looped -- the failure `_loop_penalty`
    already exists to suppress.

    Matching is `echoes` -- whole word with a crude plural stem, so "rabbit" is
    found in "rabbits" and "dragon" in "dragons". A false match is not as cheap
    as it looks and the docstring used to say it was; see `echoes` for the one
    that cost 0.2778 of `story_dodge`.
    """
    if not words:
        return 0
    lowered = text.lower()
    return sum(1 for w in words if echoes(w, lowered))


def temperature_ladder(base: float, n: int, spread: float) -> list[float] | None:
    """`n` temperatures spanning `base * [1-spread, 1+spread]`, spread-preserving.

    Returns None when the ladder is off, and None means the scalar temperature
    path, not a ladder of equal values -- so a run with `spread = 0` reproduces
    every draw made before this function existed, bit for bit.

    THE ORDER IS NOT ASCENDING, AND THAT IS THE WHOLE DESIGN
    -------------------------------------------------------
    `quality.score` compares reranking settings by taking the **first n** of a
    larger collected pool, which is sound only because "the rows are independent
    draws, so the first n of them are a sample of size n" (its own docstring). An
    ascending ladder breaks that silently: the first 8 rows of a 16-row ascending
    ladder are the 8 coldest, so a table comparing n=8 against n=16 would be
    comparing a cold pool against a full-range one and reporting the difference
    as an effect of pool size.

    The values are therefore emitted in van der Corput (bit-reversed) order, for
    which **every** prefix is spread across the range rather than clustered at one
    end: for n = 16 the first four temperatures are the 1st, 9th, 5th and 13th
    rungs. This is the standard low-discrepancy trick and it is used here for
    exactly the property it is named for.
    """
    if spread <= 0.0 or n < 2:
        return None
    rungs = [base * (1.0 - spread + 2.0 * spread * r / (n - 1)) for r in range(n)]
    return [rungs[i] for i in _van_der_corput_order(n)]


def _van_der_corput_order(n: int) -> list[int]:
    """A permutation of `range(n)` whose every prefix is spread over the range.

    Sorting by the bit-reversal of the index is the finite version of the van der
    Corput sequence. `n` need not be a power of two: indices beyond `n` are
    generated and dropped, which keeps the property for the prefix lengths that
    are powers of two and degrades gracefully for the rest.
    """
    bits = max(1, (n - 1).bit_length())
    keyed = sorted(range(n), key=lambda i: int(f"{i:0{bits}b}"[::-1], 2))
    return keyed


@dataclass
class RerankParams:
    """How many candidates to draw and how to choose between them.

    `n = 1` disables the whole mechanism and costs nothing: `rerank` returns the
    single candidate without ever building the null-context pass.
    """

    #: Candidates drawn per turn. Batch is nearly free on this scan; the cost
    #: that does grow with N is the null-context rescoring pass, which is one
    #: extra forward over N * len(reply) characters.
    n: int = 8
    #: Weight on the null-context term. 0 = best-of-N by likelihood alone.
    #: 1 = pure pointwise mutual information. Above ~0.8 the score starts
    #: rewarding rare text for being rare; measured in docs/chat/QUALITY.md.
    lam: float = 0.6
    #: Candidates shorter than this many characters are set aside unless every
    #: candidate is that short. Without it the mean-per-character score is won
    #: every time by the empty reply, whose single `<|eot|>` the model assigns a
    #: very high probability to -- a length pathology of the normaliser, not a
    #: judgement about the reply.
    min_chars: int = 12
    #: When a candidate ran out of `max_new` mid-sentence, end the turn at its
    #: last complete sentence instead. Only ever applied to a reply the model did
    #: NOT close itself -- a turn the model ended is left exactly as it ended it.
    #:
    #: This is done here rather than at display time on purpose. The trimmed
    #: characters are the only ones fed to the membrane, so what the reader sees
    #: and what the model is carrying are the same string. Trimming the display
    #: alone would leave the session's state holding half a sentence nobody was
    #: shown, which is the incoherence `ChatSession.rewind` exists to avoid.
    trim_to_sentence: bool = True
    #: The context the anti-LM term is scored under. "empty_user" keeps the
    #: conversation format byte-identical and removes only the prompt's TEXT,
    #: which is the ablation the score is supposed to be measuring. "bot_only"
    #: drops the user turn entirely and therefore also measures the format.
    null: str = "empty_user"
    #: Spread the candidates over a range of temperatures instead of drawing all
    #: of them at `SamplingParams.temperature`. Row `r` of `n` is drawn at
    #: `T * (1 - s + 2s*r/(n-1))`, so `s = 0.3` at T = 0.85 runs 0.60 .. 1.11.
    #:
    #: WHY THIS AND NOT A HIGHER TEMPERATURE
    #: `docs/chat/QUALITY.md` §6 records the sharpest null result in the package:
    #: reranking moved topicality by *exactly nothing* on a model whose candidate
    #: pool contained no on-topic reply. Selection cannot invent what was never
    #: proposed, so the lever that is left is the pool. Eight draws at one
    #: temperature are eight samples from one distribution; a ladder is eight
    #: samples from eight, and the conservative end and the exploratory end fail
    #: in different ways, which is the point.
    #:
    #: It is safe to mix temperatures in one pool for a reason specific to how
    #: this scorer is built: `sample_candidates` accumulates `logp_cond` under the
    #: model's OWN distribution, before temperature and nucleus truncation touch
    #: it. So a candidate is never scored under the sampler that proposed it, and
    #: a draw at T = 1.1 is compared with one at T = 0.6 on identical terms.
    #: A per-row temperature changes what is proposed and nothing about how it is
    #: judged.
    #:
    #: 0.0 is off and is bit-identical to every draw made before this existed --
    #: not "a ladder with zero width", but the scalar code path itself.
    temperature_spread: float = 0.0
    #: Partition candidates by how many distinct content words of the prompt they
    #: echo before ranking them by `score`. See the module docstring.
    #:
    #: On by default because this is a front end whose entire purpose is that
    #: the model can be talked to, and because the one measurement that is not
    #: circular supports it: on 20 story topics the probe battery has never
    #: contained, 6 seeds each, both selectors choosing from ONE shared pool, the
    #: hit rate goes 1/120 -> 10/120, exact McNemar p = 0.0039, nine draws better
    #: and none worse (`docs/chat/QUALITY_v8.md` §4,
    #: `scripts/chat/echo_holdout.py`). Replayed across four arms at n = 4, 8 and
    #: 16, no probe kind on any arm moves down.
    #:
    #: It costs one pass over N short strings and no forward pass at all.
    #:
    #: THREE HONEST LIMITS, all measured in `docs/chat/QUALITY_v8.md`:
    #: (1) On the battery's own `topic` column this partition IS the oracle --
    #: both score 0.2969 at n=8 -- because the probe asks whether the reply
    #: contains the requested noun and this rule prefers replies containing it.
    #: That column is circular and is not evidence; §4's held-out set is.
    #: (2) The `list` gain is mostly reply length: a plain longest-candidate
    #: baseline beats this partition there (0.317 against 0.233).
    #: (3) The gain is bounded by the pool and the pool is thin -- on held-out
    #: topics the oracle itself is only 0.0917, because the model rarely drafts
    #: a reply about an unfamiliar noun at all.
    #:
    #: The `fact` gain (0.071 -> 0.179, where the echoed word is "france" and
    #: the scored word is "paris") is the one battery column that is not
    #: circular, and it is a single arm at 28 draws.
    echo: bool = True
    #: Build the echo tier from ONE question -- does the returned text name the
    #: turn's subject? -- instead of from the weighted sum over every content
    #: word of the prompt.
    #:
    #: THE DEFECT THIS IS AIMED AT
    #: The tier is the pool's maximum of a SUM, and the request frame survives
    #: `_FUNCTION_WORDS`, so two frame words can outweigh the subject. By the
    #: weights quoted at `_FREQ_PATH`, a draft that echoes tell (5.93) + story
    #: (5.20) = 11.13 OUTRANKS a draft that echoes only penguin (10.34). Two
    #: things follow by construction. A top tier with one member is decided
    #: before `score` is consulted, so the reply is the draft that collected the
    #: most frame vocabulary and the anti-LM term -- the thing this module
    #: exists for -- chose nothing. And when NO draft names the subject a
    #: frame-word tier still forms, and hands the turn to a draft for having
    #: said "tell" and "story".
    #:
    #: WHAT IT DOES INSTEAD
    #: Drafts whose returned text names the subject are the tier. If there are
    #: none the tier is the whole length-partitioned pool, NOT the frame-word
    #: tier, so that `score` chooses among everyone. See `echo_tier`.
    #:
    #: False nests the old behaviour exactly -- not "a subject tier that matches
    #: nothing", but the old code path. It also has no effect unless `echo` is
    #: on and the caller passed a `subject`, so a prompt with no recognisable
    #: subject is decided by the weighted tier as before.
    subject_tier: bool = False
    #: Route the per-character forward through a captured CUDA graph. Pure
    #: speedup, asserted identical, and silently ignored where a graph cannot be
    #: captured -- see `snnchat.stepper`. Off is the old code path, kept because
    #: "turn the optimisation off and see" is the first thing anyone wants when a
    #: number looks wrong.
    graph: bool = True
    #: Resample the pool toward drafts the subject is imminent in, every this
    #: many characters. 0 is off and off is the identity. See `Steering` for the
    #: mechanism, and for why the candidates stop being independent when it is on.
    steer_every: int = 0
    #: Fraction of live rows replaced at each steering checkpoint.
    steer_frac: float = 0.25

    def __post_init__(self) -> None:
        if self.n < 1:
            raise ValueError(f"n must be >= 1, got {self.n}")
        if self.steer_every < 0:
            raise ValueError(f"steer_every must be >= 0, got {self.steer_every}")
        if self.null not in ("empty_user", "bot_only"):
            raise ValueError(f"null must be 'empty_user' or 'bot_only', got {self.null!r}")
        if not 0.0 <= self.temperature_spread < 1.0:
            raise ValueError(
                f"temperature_spread must be in [0, 1), got {self.temperature_spread}"
            )

    def temperature_ladder(self, base: float) -> list[float] | None:
        """Per-candidate temperatures, or None for the unchanged scalar path."""
        return temperature_ladder(base, self.n, self.temperature_spread)

    def describe(self) -> str:
        if self.n <= 1:
            return "off (n=1)"
        out = f"n={self.n} lambda={self.lam:g} min_chars={self.min_chars} null={self.null}"
        if self.temperature_spread > 0.0:
            out += f" spread={self.temperature_spread:g}"
        out += f" echo={'on' if self.echo else 'off'}"
        if self.subject_tier:
            out += " subject-tier"
        if self.steer_every:
            out += f" steer={self.steer_every}/{self.steer_frac:g}"
        return out


@dataclass
class Candidate:
    """One drawn reply and the two log-probabilities it is ranked by."""

    ids: list[int]              #: visible characters only, no stop id
    scored: list[int]           #: what both log-probabilities cover: ids + stop id if any
    logp_cond: float            #: log P(scored | conversation so far)
    logp_null: float = 0.0      #: log P(scored | null context)
    score: float = 0.0
    #: Distinct prompt content words this reply uses. 0 when the echo partition
    #: is off, which is why `/candidates` labels the column rather than printing
    #: a bare number.
    echo: int = 0
    #: The inverse-document-frequency sum of those words, and the quantity the
    #: tier is actually built on. A field rather than an attribute assigned from
    #: outside, so a candidate that was never scored reads 0.0 instead of
    #: raising on access.
    echo_weight: float = 0.0
    #: Does the text this draft is RETURNED as name the turn's subject? Matched
    #: by `echoes` on the same `final_ids` string the two fields above read, so
    #: the three cannot disagree about which text was judged. False when no
    #: subject was passed or the echo partition is off -- "not asked", exactly
    #: as `echo` reads 0 -- and written whether or not
    #: `RerankParams.subject_tier` is on, so a stored pool records what the
    #: subject tier WOULD have seen under either rule.
    subject_hit: bool = False
    #: False when the null-context pass was skipped because it could not change
    #: the winner. `logp_null` is then 0.0 because it was not measured, not
    #: because it was measured to be zero -- `fill_null_scores` is what turns
    #: one into the other, and `/candidates` calls it before displaying.
    null_scored: bool = True
    #: The `RerankParams.null` that `logp_null` was measured under, or None if it
    #: never was. Written by `_score_null` and by nothing else. `null_scored`
    #: cannot serve: it defaults to True and is True at lambda = 0, so on a
    #: candidate fresh from the sampler it says "measured" about a 0.0 nobody
    #: measured. This is what lets `select` run twice on one pool and pay for
    #: the null pass at most once per null context.
    null_context: str | None = None

    @property
    def n_chars(self) -> int:
        return len(self.ids)

    @property
    def closed(self) -> bool:
        """True if the model ended its own turn rather than running out of budget."""
        return len(self.scored) > len(self.ids)


#: Characters a sentence may end on. The closing quote and bracket are included
#: so that a reply ending `..." he said."` is not cut back to the word before the
#: quotation mark.
_SENTENCE_END = '.!?'
_TRAILING = '"\')]'


def trim_to_sentence(ids: list[int], tok, *, min_keep: int = 12) -> list[int]:
    """Cut `ids` back to its last complete sentence. Identity if there is none.

    `ids` holds no marker (stop ids are never appended to a candidate's text), so
    one id is one character and an index into the decoded string is an index into
    the list. That is what makes this safe to do on ids rather than on text.

    Returns the input unchanged when trimming would leave less than `min_keep`
    characters -- a reply cut to three words to satisfy a punctuation rule is
    worse than one that ends mid-sentence, which is the thing being fixed.
    """
    if not ids:
        return ids
    text = tok.decode_visible(ids)
    cut = -1
    for i, ch in enumerate(text):
        if ch in _SENTENCE_END:
            j = i
            while j + 1 < len(text) and text[j + 1] in _TRAILING:
                j += 1
            cut = j
    if cut < 0 or cut + 1 < min_keep:
        return ids
    return ids[:cut + 1]


def final_ids(cand: "Candidate", tok, rp: "RerankParams") -> list[int]:
    """The ids this candidate will actually be returned as.

    THE SELECTOR MUST RANK THE REPLY, NOT THE DRAFT
    -----------------------------------------------
    `trim_to_sentence` cuts an unclosed draft back to its last complete
    sentence, and it runs *after* selection. So for the first four rounds of
    this work the echo tier read `c.ids` -- text including a tail that the
    reader never sees. A draft whose only mention of the subject was in that
    tail won the tier for a word its reply does not contain.

    Measured 2026-08-12 on the committed pools at n = 128: 0.7 % of drafts lose
    echo weight to trimming, and tiering on the returned text instead is
    **+3 / -0** on the twenty held-out nouns and **0 / 0** on the twenty fresh
    ones. Small, and one-directional by construction rather than by luck --
    ranking the string you return cannot be worse than ranking one you discard.

    It also settles a disagreement rather than creating one. `QUALITY_v9.md`
    §2's n = 128 row reports 0.4167, which was replayed off the *stored* pool
    text and is therefore the trimmed reading; the shipped code scored 0.3917.
    Both numbers were right about different selectors. This is the one function
    both now use, which is what stops that recurring.
    """
    ids = list(cand.ids)
    if rp.trim_to_sentence and not cand.closed:
        ids = trim_to_sentence(ids, tok, min_keep=rp.min_chars)
    return ids


def null_prefix_ids(kind: str = "empty_user") -> list[int]:
    """The id sequence the anti-LM term is conditioned on.

    `empty_user` is `<|bos|><|user|><|eot|><|bot|>` -- a conversation that has
    opened, a user who said nothing, and the model's turn beginning. Every
    structural cue the real prefix carries is present and only the prompt text
    is missing, so the difference of the two log-probabilities is attributable
    to the text rather than to the format.
    """
    if kind == "bot_only":
        return [BOS, BOT]
    return [BOS, USER, EOT, BOT]


@torch.no_grad()
def _feed(model, ids_2d: torch.Tensor, state):
    logits, state, _ = model(ids_2d, state=state)
    return logits[:, -1, :].float(), state


def _expand_state(state, n: int):
    """Copy a B=1 state into B=n rows.

    `repeat` rather than `expand`: the fused scan writes its final membrane into
    the tensor it was handed, and a stride-0 broadcast view would have N rows
    aliasing one allocation.
    """
    if state is None:
        return None
    return [s.repeat(n, *([1] * (s.dim() - 1))) if s.shape[0] == 1 else s for s in state]


# ---------------------------------------------------------------------------
# steering the pool: resampling toward drafts the subject is imminent in
# ---------------------------------------------------------------------------


@dataclass
class Steering:
    """Reallocate the candidate pool mid-draft toward drafts about the subject.

    WHY A DECODER CAN DO THIS AT ALL, AND WHY THIS MODEL ESPECIALLY
    ---------------------------------------------------------------
    Best-of-N spends its whole budget before it learns anything: `n` drafts are
    run to completion and only then compared. Every draft that went off topic in
    its first clause consumed a full reply's worth of scan anyway.

    The alternative is the standard sequential-Monte-Carlo move -- periodically
    score the partial drafts, then copy promising ones over unpromising ones and
    let them diverge. On a transformer that copy means duplicating a KV cache
    that grows with the reply; here a draft's entire history is `[2d]` of
    membrane per layer, so cloning one is an `index_select` over a fixed-size
    tensor and costs the same at character 300 as at character 3. This is the
    same O(1)-state property `snnchat.generate`'s docstring opens with, spent on
    search instead of on session length.

    THE POTENTIAL IS A LOOKAHEAD, NOT AN ECHO COUNT
    -----------------------------------------------
    The obvious score is the one the selector already uses -- how much of the
    prompt the draft has echoed so far. It is useless here, and the reason is
    worth stating: over 4,319 topic conversations the corpus puts the subject
    noun a median of **81 characters** into the reply, so a draft that has
    already said "penguin" has already succeeded and one that has not carries no
    lexical evidence either way. Resampling on it would be a no-op until the
    moment it stopped mattering.

    So the potential is `log P(" penguin" | draft so far)` -- one teacher-forced
    pass over the word from each row's current membrane, which asks the model
    itself whether the subject is *about to* arrive. A draft that has just
    written "Once upon a time, there was a little " scores high on it before
    committing to any animal at all.

    Rows that have already said the word are given `0.0`, the top of the scale:
    the quantity is "will this reply be about the subject", and for them the
    answer is settled. That also stops the pressure from compounding into a row
    that says the word over and over.

    AND IT IS AIMED AT A MEASURED FAILURE, NOT A GUESSED ONE
    --------------------------------------------------------
    Phase 4 measured this neuron's memory horizon at **47-48 characters**. The
    corpus's own subject position is 81. The model is therefore being asked to
    recall the requested noun from *beyond its own reach* at exactly the moment
    the story form demands it, which is a mechanism for the topic failure that
    no amount of reranking at the end can repair. Resampling every `every`
    characters -- comfortably inside the horizon -- is an external memory of the
    prompt, applied while the draft can still act on it.

    WHAT IT COSTS AND WHAT IT BREAKS
    --------------------------------
    One forward over `len(ids) - 1` characters per checkpoint, from a cloned
    state, plus an `index_select`. Nothing per character.

    What it breaks is independence: with steering on, the returned candidates
    are a resampled particle system, not `n` i.i.d. draws. `snnchat.quality`'s
    `score` takes the first `n` rows of a larger pool and calls that a sample of
    size `n` -- sound for independent draws and NOT sound for these. Steering is
    off by default partly for that reason.
    """

    #: The subject word, lowercased, as it is searched for in a draft.
    word: str
    #: Ids of the continuation whose log-probability is the potential -- the
    #: word with a leading space, so the model is asked about a word boundary
    #: rather than about a suffix it might be in the middle of.
    ids: list[int]
    #: Characters between resampling checkpoints. Below the 47-48 character
    #: horizon by construction; see above.
    every: int = 32
    #: Fraction of the live rows replaced at a checkpoint. The rest are left
    #: alone, which is what keeps the pool from collapsing onto one lineage.
    frac: float = 0.25

    def __post_init__(self) -> None:
        if self.every < 1:
            raise ValueError(f"every must be >= 1, got {self.every}")
        if not 0.0 <= self.frac < 0.5:
            # At frac >= 0.5 the donor and victim halves meet and a row can be
            # asked to clone itself; below it the two are always disjoint.
            raise ValueError(f"frac must be in [0, 0.5), got {self.frac}")


def steer_word(
    words, tok, *, every: int = 32, frac: float = 0.25, min_weight: float = 0.0
) -> Steering | None:
    """Build a `Steering` for the rarest content word of a prompt, or None.

    Rarest by the same inverse-document-frequency table the echo tier is built
    on, so the two agree about which word of "tell me a story about a penguin"
    is the subject -- penguin at 10.34 against tell at 5.93 -- without a second
    notion of subjecthood to keep in step.

    `min_weight` is 0 by default, i.e. no threshold. A cut somewhere around 7
    would separate the request frame from the subject on the words measured in
    `docs/chat/QUALITY_v9.md` §2, but that is a constant fitted to twenty
    prompts and this file has been careful not to acquire any. Without it the
    rule on a subject-less prompt ("hello") is to steer toward the model
    answering in kind, which is harmless.
    """
    if not words:
        return None
    word = max(words, key=word_weight)
    if word_weight(word) < min_weight:
        return None
    ids = [int(i) for i in tok.encode(" " + word)]
    if not ids:
        return None
    return Steering(word=word, ids=ids, every=every, frac=frac)


@torch.no_grad()
def _topic_imminence(model, stepper, state, logprobs, steer, out, tok, device) -> list[float]:
    """Per row, `log P(steer.ids | draft so far)`, or 0.0 if it is already there.

    The first character is scored from `logprobs`, which the caller already has,
    so the forward pass covers only the remaining `len(ids) - 1`.
    """
    n = len(out)
    total = logprobs[:, steer.ids[0]]
    if len(steer.ids) > 1:
        # A CLONE of the state: this is a question about a hypothetical
        # continuation, and the drafts must go on from where they actually are.
        st = stepper.read_state() if stepper is not None else [s.clone() for s in state]
        feed = torch.tensor(steer.ids[:-1], dtype=torch.int64, device=device)
        feed = feed.unsqueeze(0).expand(n, -1).contiguous()
        lg, _, _ = model(feed, state=st)
        lp = F.log_softmax(lg.float(), dim=-1)
        tgt = torch.tensor(steer.ids[1:], dtype=torch.int64, device=device)
        tgt = tgt.unsqueeze(0).expand(n, -1)
        total = total + lp.gather(2, tgt[:, :, None]).squeeze(2).sum(dim=1)

    vals = total.tolist()
    if tok is not None:
        for r in range(n):
            if steer.word in tok.decode_visible(out[r]).lower():
                vals[r] = 0.0
    return vals


@torch.no_grad()
def _steer_indices(model, stepper, state, logprobs, steer, out, alive_cpu, tok, device):
    """The row permutation a resampling checkpoint asks for, or None for none.

    Only LIVE rows take part. A finished draft is a completed candidate and
    overwriting it would throw away a reply the pool has already paid for --
    including, quite possibly, the one the selector was going to return.
    """
    n = len(out)
    rows = [r for r in range(n) if alive_cpu[r]]
    k = int(len(rows) * steer.frac)
    if k < 1:
        return None
    pot = _topic_imminence(model, stepper, state, logprobs, steer, out, tok, device)
    order = sorted(rows, key=lambda r: pot[r], reverse=True)

    idx = list(range(n))
    changed = False
    for i in range(k):
        donor, victim = order[i], order[-1 - i]
        # Strictly worse, not merely lower: at the start of a reply every row
        # scores the same and a tie-broken permutation would churn the pool for
        # nothing.
        if pot[victim] < pot[donor] - 1e-4:
            idx[victim] = donor
            changed = True
    return torch.tensor(idx, dtype=torch.int64, device=device) if changed else None


@torch.no_grad()
def sample_candidates(
    model,
    logits: torch.Tensor,
    state,
    params: SamplingParams,
    n: int,
    *,
    stop_ids=_STOP_IDS,
    temperatures: list[float] | None = None,
    graph: bool = True,
    steer: "Steering | None" = None,
    tok=None,
) -> list[Candidate]:
    """Draw `n` replies from one context, in one batch.

    `logits` is the model's distribution over the first character of the reply
    ([V] or [n, V]); `state` is the membrane state that produced it (B=1 or
    B=n). Rows are advanced together and a row that has emitted a stop id is
    frozen -- it keeps being fed a filler character so the batch stays
    rectangular, and its output is not read again.

    `logp_cond` accumulates the log-probability of each sampled character under
    the model's OWN distribution, before temperature, `top_p` and the loop
    breaker touch it. That matters: the score is meant to compare replies under
    the model, not under the sampler's truncated view of it, and a candidate
    that survived a nucleus cut is not thereby more probable. It is also what
    makes `temperatures` legitimate -- see `RerankParams.temperature_spread`.

    `temperatures`, when given, is one temperature per row and replaces
    `params.temperature` for the proposal distribution only. `None` runs the
    scalar path unchanged.

    ONE HOST SYNCHRONISATION PER CHARACTER, NOT `2n`
    ------------------------------------------------
    Only `nxt` is read back. Liveness is tracked in a plain Python list derived
    from the same ids and the same stop set, so it cannot disagree with the
    device-side `alive` mask that the arithmetic uses.

    The version this replaced asked `bool(just_stopped[r])` and `bool(alive[r])`
    per row per character. Each of those is a device read, measured at 36 us on
    this box, so a 300-character turn at n = 128 paid ~2.8 s to learn what it
    already knew -- and it grew *linearly in the pool size*, which is precisely
    the axis `docs/chat/QUALITY_v9.md` §3 identifies as the one worth spending
    on. Draw cost is now flat in `n` up to the batch where the scan itself
    stops being latency-bound.

    `graph` routes the forward, the truncation and both softmaxes through a
    captured CUDA graph (`snnchat.stepper`) when one can be captured. It is a
    pure speedup: `tests/test_snnchat.py::test_graph_and_eager_draw_the_same_text`
    asserts the two paths produce identical ids.

    `steer` turns the pool from `n` independent draws into a resampled particle
    system -- see `Steering`. It is off unless asked for, and when it is on the
    returned candidates are **no longer independent**, which matters to any
    caller that treats a prefix of them as a smaller pool.
    """
    device = logits.device
    if logits.dim() == 1:
        logits = logits.unsqueeze(0)
    if logits.shape[0] == 1 and n > 1:
        logits = logits.repeat(n, 1)
        state = _expand_state(state, n)
    n = logits.shape[0]

    temps = None
    if temperatures is not None:
        if len(temperatures) != n:
            raise ValueError(f"got {len(temperatures)} temperatures for {n} rows")
        temps = torch.tensor(
            [max(float(t), 1e-6) for t in temperatures], device=device, dtype=logits.dtype
        )[:, None]

    stepper = None
    if graph:
        from snnchat.stepper import stepper_for

        stepper = stepper_for(model, n, params, temps=temps, device=device)
        if stepper is not None:
            stepper.prime(state)

    gen = None
    if params.seed is not None:
        gen = torch.Generator(device=device)
        gen.manual_seed(int(params.seed))

    out: list[list[int]] = [[] for _ in range(n)]
    recent: list[list[int]] = [[] for _ in range(n)]
    stopped: list[int | None] = [None] * n
    logp = torch.zeros(n, device=device, dtype=torch.float32)
    alive = torch.ones(n, dtype=torch.bool, device=device)
    alive_cpu = [True] * n
    n_alive = n
    stop_set = set(stop_ids)
    stop_t = torch.tensor(sorted(stop_set), device=device)
    zero = torch.zeros(n, device=device, dtype=torch.float32)
    eot = torch.full((n,), EOT, dtype=torch.int64, device=device)

    # The first distribution comes from the caller's logits rather than from a
    # step, so it is built here with the identical arithmetic the loop uses
    # below. `recent` is empty at this point, so there is no penalty to apply.
    probs = F.softmax(_filter(logits.clone(), params, temps), dim=-1)
    logprobs = F.log_softmax(logits, dim=-1)

    for t in range(params.max_new):
        if steer is not None and t and t % steer.every == 0 and n_alive > 1:
            idx = _steer_indices(
                model, stepper, state, logprobs, steer, out, alive_cpu, tok, device
            )
            if idx is not None:
                probs = probs.index_select(0, idx)
                logprobs = logprobs.index_select(0, idx)
                logp = logp.index_select(0, idx)
                if stepper is not None:
                    stepper.permute(idx)
                else:
                    state = [s.index_select(0, idx) for s in state]
                order = idx.tolist()
                out = [list(out[s]) for s in order]
                recent = [list(recent[s]) for s in order]

        nxt = torch.multinomial(probs, 1, generator=gen).squeeze(-1)      # [n]
        raw = logprobs.gather(1, nxt[:, None]).squeeze(1)
        logp = logp + torch.where(alive, raw, zero)

        is_stop = (nxt[:, None] == stop_t[None, :]).any(dim=1)
        alive = alive & ~is_stop

        nxt_cpu = nxt.tolist()                     # THE ONE SYNC PER CHARACTER
        for r in range(n):
            if not alive_cpu[r]:
                continue
            c = nxt_cpu[r]
            if c in stop_set:
                stopped[r] = c
                alive_cpu[r] = False
                n_alive -= 1
                continue
            out[r].append(c)
            recent[r].append(c)
            if len(recent[r]) > 2 * params.loop_max_block:
                del recent[r][0]
        if n_alive == 0:
            break

        # Dead rows are fed EOT: harmless, since their state is never read
        # again, and it keeps the batch rectangular so the scan stays one call.
        feed = torch.where(alive, nxt, eot)
        pairs = _penalty_pairs(recent, params) if params.loop_penalty > 0.0 else []
        if stepper is not None:
            stepper.penalise(pairs, params.loop_penalty)
            probs, logprobs = stepper.step(feed)
        else:
            logits, state = _feed(model, feed[:, None], state)
            work = logits.clone()
            for r, tok_id in pairs:
                work[r, tok_id] -= params.loop_penalty
            probs = F.softmax(_filter(work, params, temps), dim=-1)
            logprobs = F.log_softmax(logits, dim=-1)

    lp = logp.tolist()
    return [
        Candidate(
            ids=out[r],
            scored=out[r] + ([stopped[r]] if stopped[r] is not None else []),
            logp_cond=float(lp[r]),
        )
        for r in range(n)
    ]


def _penalty_pairs(recent: list[list[int]], params: SamplingParams):
    """`[(row, token_id), ...]` the loop breaker wants suppressed next character.

    A list rather than a tensor write per row: in non-degenerate text it is
    empty, which is the property `snnchat.generate`'s module docstring claims
    for the loop breaker and which a per-row device write would have quietly
    cost anyway.
    """
    pairs = []
    for r, rec in enumerate(recent):
        target = loop_penalty_target(rec, params)
        if target is not None:
            pairs.append((r, target))
    return pairs


@torch.no_grad()
def score_under_prefix(
    model,
    prefix_ids,
    sequences: list[list[int]],
    device,
    *,
    filler: int = EOT,
) -> list[float]:
    """Teacher-forced `log P(seq | prefix)` for several sequences, one call.

    Rows are right-padded to a common length and a mask selects, per row, only
    the positions that predict one of that row's own characters -- so the pad is
    computed and discarded rather than scored. Padding on the right is safe in a
    way padding on the left would not be: this is a recurrent network with no
    positional encoding, and everything before a position is part of its
    context, while everything after it is not.
    """
    if not sequences:
        return []
    prefix = list(prefix_ids)
    p = len(prefix)
    lengths = [len(s) for s in sequences]
    width = p + max(max(lengths), 1)
    rows = torch.full((len(sequences), width), filler, dtype=torch.int64)
    mask = torch.zeros(len(sequences), width - 1, dtype=torch.bool)
    pref_t = torch.tensor(prefix, dtype=torch.int64)
    for r, seq in enumerate(sequences):
        rows[r, :p] = pref_t
        if seq:
            rows[r, p:p + len(seq)] = torch.tensor(seq, dtype=torch.int64)
            # logits at index j predict rows[:, j+1]; the first reply character
            # sits at index p and is therefore predicted at index p-1.
            mask[r, p - 1:p - 1 + len(seq)] = True

    rows = rows.to(device)
    mask = mask.to(device)
    logits, _, _ = model(rows, state=None)
    logprobs = F.log_softmax(logits[:, :-1, :].float(), dim=-1)
    picked = logprobs.gather(2, rows[:, 1:, None]).squeeze(2)
    return (picked * mask).sum(dim=1).tolist()


def echo_tier(pool, *, subject_tier: bool, has_subject: bool) -> list:
    """The members of `pool` that `score` is allowed to choose between.

    `pool` is the LENGTH-partitioned pool; this is the partition inside it. It
    reads `.echo_weight` and `.subject_hit` and nothing else, and touches no
    tensor, so it can be called on any stand-in object carrying those two
    attributes -- a row of a stored pool, say.

    ONE FUNCTION, BECAUSE A HAND COPY OF THIS IS A KNOWN HAZARD
    -----------------------------------------------------------
    `scripts/chat/echo_holdout.py` re-implements the weighted tier inline and
    says of its own copy: "If this copy and `rerank` ever disagree, every number
    in QUALITY_v8 describes a selector the REPL does not use." The tier now has
    two rules and a fallback between them, which is more than a comment can keep
    in step. `select` decides the tier by calling this and in no other way, so a
    scorer that calls it too is measuring the REPL's selector by construction.

    THE SUBJECT RULE (`subject_tier` and `has_subject`)
    ---------------------------------------------------
    The drafts with `subject_hit`. If there are none, `pool` UNCHANGED -- the
    whole length-partitioned pool, and deliberately NOT the weighted tier below.
    Falling back to the weighted tier would keep exactly the behaviour this rule
    exists to remove: when nobody drafted the noun, the only nonzero weights in
    the pool are frame words.

    THE WEIGHTED RULE (otherwise)
    -----------------------------
    `best == 0` means no candidate echoed any content word, and then this is
    the identity. Note that the frame counts as content ("tell", "story"), so
    `best > 0` is NOT the same as "something on topic was drawn". Tiered on the
    WEIGHTED sum, not the count: `Candidate.echo` is retained for `/candidates`
    and for the artifacts, but it is not what decides.
    """
    if subject_tier and has_subject:
        return [c for c in pool if c.subject_hit] or pool
    best = max((c.echo_weight for c in pool), default=0.0)
    if best > 0.0:
        pool = [c for c in pool if c.echo_weight >= best - 1e-9]
    return pool


def select(
    model,
    cands: list[Candidate],
    rp: RerankParams,
    *,
    device,
    tok=None,
    echo_words=None,
    subject: str | None = None,
) -> Candidate:
    """Choose the winner among candidates that have ALREADY been drawn.

    Everything `rerank` does after sampling: the echo annotation, the length
    partition, `echo_tier`, the null pass if it can matter, the score. Split out
    so that one pool can be selected from under two rules without being drawn
    twice -- a paired comparison, with the draw removed as a source of variance.

    SAFE TO CALL AGAIN ON THE SAME CANDIDATES WITH A DIFFERENT `rp`
    ---------------------------------------------------------------
    Nothing here trusts what an earlier call left behind. `echo`, `echo_weight`
    and `subject_hit` are rewritten every call, and rewritten to their defaults
    when this call's settings do not ask for them, so an `echo=False` pass over
    a pool is not decided by the previous call's weights. `score` is recomputed.
    `logp_null` is the one expensive field and is measured at most once per null
    context -- see `Candidate.null_context`.

    `model` is only used by the null pass, so it may be None at `rp.lam == 0`.

    With fewer than two candidates there is nothing to select; `rerank` handles
    that case itself and never gets here.
    """
    # `echo_words` alone is the condition this had before `subject` existed, so
    # a caller that passes no subject annotates exactly when it used to.
    annotate = bool(rp.echo and tok is not None and (echo_words or subject))
    for c in cands:
        if annotate:
            # The text this draft BECOMES, not the draft. See `final_ids`.
            text = tok.decode_visible(final_ids(c, tok, rp))
            c.echo = echo_count(text, echo_words)
            c.echo_weight = echo_weight(text, echo_words)
            c.subject_hit = bool(subject) and echoes(subject, text.lower())
        else:
            c.echo, c.echo_weight, c.subject_hit = 0, 0.0, False

    # The length guard is applied by PARTITION rather than by penalty: a reply
    # of two characters is not a slightly worse reply, it is a different event
    # (the model closing its turn immediately), and the mean-per-character score
    # cannot compare the two. If every candidate is short, they are ranked among
    # themselves rather than the turn being forced to produce text.
    long_enough = [c for c in cands if c.n_chars >= rp.min_chars]
    pool = long_enough or cands

    # The echo partition is applied INSIDE the length partition, not before it.
    # The other order would let a nine-character fragment that happens to name
    # the topic beat a whole reply that also names it, which is the length
    # pathology `min_chars` exists to prevent, reintroduced one level up. That
    # holds for the subject rule exactly as for the weighted one.
    pool = echo_tier(
        pool, subject_tier=rp.subject_tier, has_subject=bool(annotate and subject)
    )

    # THE PARTITIONS ARE RESOLVED BEFORE THE NULL PASS, AND SOMETIMES INSTEAD OF IT
    # -----------------------------------------------------------------------------
    # `score` only ever decides between members of the surviving tier, so when
    # the tier holds one candidate the anti-LM term cannot change the answer --
    # and it is the second-largest cost in a turn, a teacher-forced pass over
    # the full pool that runs the scan once per character all over again. Skipped
    # rather than computed and discarded.
    #
    # This is an ordering change and not a behavioural one: the tier is built
    # from lengths and echo weights, neither of which has ever read `score`.
    # `test_skipping_the_null_pass_picks_the_same_winner` holds that.
    decided = len(pool) == 1
    if (rp.lam != 0.0 and not decided
            and any(c.null_context != rp.null for c in cands)):
        _score_null(model, cands, rp, device)
    for c in cands:
        c.score = (c.logp_cond - rp.lam * c.logp_null) / max(len(c.scored), 1)
        # At lambda = 0 the term is not part of the score and was never measured
        # before this change either, so there is nothing outstanding to fill.
        # Otherwise the question is whether it HAS been measured, not whether
        # this call measured it: an earlier `select` on the same pool may have.
        c.null_scored = rp.lam == 0.0 or c.null_context == rp.null

    return max(pool, key=lambda c: c.score)


def rerank(
    model,
    logits: torch.Tensor,
    state,
    params: SamplingParams,
    rp: RerankParams,
    *,
    device=None,
    stop_ids=_STOP_IDS,
    tok=None,
    echo_words=None,
    subject: str | None = None,
) -> tuple[Candidate, list[Candidate]]:
    """Draw `rp.n` candidates and return `(winner, all_candidates)`.

    With `rp.n == 1` the null-context pass is skipped entirely, so the ordinary
    single-sample path pays nothing for this module existing.

    `echo_words` is the prompt's content words, from `prompt_content_words`, and
    `tok` decodes a candidate to look for them. Both are optional: without them
    the echo partition is skipped and the selection is exactly what it was
    before the partition existed, which is what makes every caller that has not
    been updated behave identically rather than subtly differently.

    `subject` is the one word the turn is about, if the caller knows it, and is
    optional in the same sense: None is the weighted tier, whatever
    `rp.subject_tier` says. See `echo_tier`.

    This is `sample_candidates` followed by `select`, and nothing else.
    """
    steer = None
    if rp.steer_every and echo_words and tok is not None:
        steer = steer_word(echo_words, tok, every=rp.steer_every, frac=rp.steer_frac)

    cands = sample_candidates(
        model, logits, state, params, rp.n, stop_ids=stop_ids,
        temperatures=rp.temperature_ladder(params.temperature),
        graph=rp.graph, steer=steer, tok=tok,
    )
    if len(cands) == 1:
        cands[0].score = cands[0].logp_cond / max(cands[0].n_chars, 1)
        return cands[0], cands

    winner = select(model, cands, rp, device=device or logits.device, tok=tok,
                    echo_words=echo_words, subject=subject)
    return winner, cands


def _score_null(model, cands, rp: RerankParams, device) -> None:
    """Fill `logp_null` for every candidate, in one batched pass."""
    nulls = score_under_prefix(
        model, null_prefix_ids(rp.null), [c.scored for c in cands], device
    )
    for c, v in zip(cands, nulls):
        c.logp_null = float(v)
        c.null_context = rp.null


@torch.no_grad()
def fill_null_scores(model, cands, rp: RerankParams, device) -> None:
    """Measure `logp_null` for candidates whose turn skipped it, and rescore.

    Exists so `/candidates` can print a real number instead of a zero that looks
    like one. Safe to call at any time and from any state: the anti-LM term is
    conditioned on a fixed four-id prefix and on nothing about the session, so
    it is as computable after the turn as during it.
    """
    if not cands or all(c.null_scored for c in cands):
        return
    _score_null(model, cands, rp, device)
    for c in cands:
        c.score = (c.logp_cond - rp.lam * c.logp_null) / max(len(c.scored), 1)
        c.null_scored = True
