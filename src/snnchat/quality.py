"""Measuring whether a reply is ABOUT the prompt, without a human in the loop.

WHY A HARNESS AND NOT A READ-THROUGH
------------------------------------
`docs/chat/RESULTS.md` says the shipped model is "fluent without being
responsive", and that judgement was made by reading `docs/chat/transcript.md`.
That is fine as a description and useless as a target: two checkpoints cannot be
ordered by reading twenty replies each, because the sampler is stochastic, the
reader knows which one is supposed to be better, and twenty replies is not
enough to see a ten-point difference in anything.

So this module scores replies mechanically. Nothing here is a language-quality
judgement -- a mechanical scorer cannot make one. Every metric is a countable
proxy, chosen so that a change in it is hard to produce by accident:

1.  **Prompt dependence** (`prompt_dependence`). Held-out (user, reply) pairs
    from the val splits, scored twice: once given their own user turn and once
    given a DIFFERENT conversation's user turn. The difference, in bits per
    character, is how much the real prompt was worth to the model. It needs no
    generation, no sampling seed and no lexicon, and it is the metric to trust
    when two others disagree. A model that ignored the prompt entirely would
    score 0.

2.  **Topicality** (`PROBES`, kind `topic`). Ask for a story about a rabbit,
    check whether the word "rabbit" occurs. Crude, and it only detects the
    coarsest kind of on-topic-ness -- but it is exactly the failure
    `docs/chat/RESULTS.md` records ("a story about a rabbit produced a story
    about a bird"), and it cannot be gamed except by mentioning the topic.

3.  **Fallback rate.** How often a substantive prompt is answered with the
    greeting or the identity line. This is the model's characteristic evasion
    and it is invisible to bpc, which likes those replies.

4.  **Degeneration and format.** Repeated blocks, word-bigram diversity, whether
    the turn was closed by the model or cut off by the character budget.

5.  **The expected failures.** Arithmetic and facts, scored the same way and
    reported alongside. They are here so that an arm which improves topicality
    while quietly wrecking something else is visible, and because a battery that
    only shows what a model is good at is an advertisement.

CANDIDATES ARE COLLECTED ONCE AND SCORED MANY TIMES
---------------------------------------------------
`collect` draws N candidates per (probe, seed) and records each one's two
log-probabilities. `score` then selects a winner for a given `(n, lambda)` and
computes the metrics in pure Python. A whole sweep over reranking settings is
therefore one pass on the GPU and a few seconds of arithmetic -- and, more
usefully, every setting is scored on the IDENTICAL candidate pool, so a
difference between two settings is a difference in selection and cannot be a
difference in what happened to be drawn.

Not part of the research protocol. No number this module produces is a reported
figure, there are no control arms, and the probe set is a judgement.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field

import numpy as np
import torch
import torch.nn.functional as F

from snnchat.generate import SamplingParams
from snnchat.rerank import (
    null_prefix_ids,
    sample_candidates,
    score_under_prefix,
    temperature_ladder,
)
from snnchat.tokenizer import BOS, BOT, EOT, USER, ChatTokenizer

__all__ = [
    "Probe",
    "PROBES",
    "collect",
    "score",
    "prompt_dependence",
    "extract_pairs",
    "wilson_interval",
    "rate_ci",
    "resolves_against",
]

_LOG2E = 1.4426950408889634

#: Two-sided 95 % normal quantile. Named rather than inlined because every
#: interval this module reports must be at ONE confidence level -- a table that
#: mixes 90 % and 95 % intervals is worse than a table with none.
Z95 = 1.959963984540054


# ---------------------------------------------------------------------------
# reporting a rate against a threshold
# ---------------------------------------------------------------------------
#
# WHY A POINT ESTIMATE IS NOT A RESULT
# ------------------------------------
# `story_dodge`, `list_strict` and every `by_kind` entry are proportions over a
# countable number of draws, and each is compared in `docs/chat/` against a
# threshold fixed in advance. A bare fraction hides two different things:
#
# 1.  **Resolution.** The metric can only land on multiples of 1/N. With the
#     4-seed battery `story_dodge` runs over 48 draws, so it is quantised at
#     0.0208 -- and `PREDICTION_v5.md`'s 0.125 band edge is exactly 6/48, a
#     lattice point. A threshold that sits ON the lattice cannot be missed
#     narrowly; it is hit exactly, and the verdict then turns on which side of
#     "below" the wording fell. That is what happened to P1.
# 2.  **Sampling noise.** 6/48 and 150/1200 are the same number and not the same
#     evidence. Without an interval a reader cannot tell them apart.
#
# So: wherever a rate is compared against a pre-registered threshold, report
# `rate_ci` alongside it and let `resolves_against` say whether the comparison
# was actually decided. See `docs/chat/CONVENTIONS.md`.


def wilson_interval(k: int, n: int, *, z: float = Z95) -> tuple[float, float]:
    """Wilson score interval for `k` successes in `n` Bernoulli draws.

    Wilson rather than the textbook `p +/- z*sqrt(p(1-p)/n)`: at the rates these
    metrics live at (0.03 - 0.17) and the sample sizes available, the normal
    approximation undercovers badly and can put the lower bound below zero,
    which is not an interval so much as an apology. Wilson is bounded in [0, 1]
    by construction, does not collapse to zero width at `k = 0`, and is the
    interval Brown, Cai & DasGupta (2001) recommend for exactly this regime.

    `n = 0` returns the whole interval, because no draws is no information.
    """
    if n <= 0:
        return (0.0, 1.0)
    p = k / n
    d = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / d
    # The interval contains `p` by construction; at k = 0 and k = n the
    # arithmetic leaves ~1e-17 of rounding on the closed end, which would make
    # `ci_low <= rate <= ci_high` false in a report for no reason at all. Clamped
    # against `p` rather than rounded, so the bracket is exact at every n.
    return (min(max(0.0, centre - half), p), max(min(1.0, centre + half), p))


def rate_ci(k: int, n: int, *, z: float = Z95) -> dict:
    """A proportion reported the way a proportion has to be reported.

    Returns the point estimate, its numerator and denominator (so a reader can
    recompute anything, including the lattice spacing `1/n`), the binomial
    standard error, and the Wilson interval.
    """
    lo, hi = wilson_interval(k, n, z=z)
    p = (k / n) if n else 0.0
    return {
        "k": int(k),
        "n": int(n),
        "rate": round(p, 4),
        "se": round(((p * (1 - p) / n) ** 0.5) if n else 0.0, 4),
        "ci_low": round(lo, 4),
        "ci_high": round(hi, 4),
        "lattice": round(1.0 / n, 5) if n else None,
        "z": z,
    }


def resolves_against(k: int, n: int, threshold: float, *, z: float = Z95) -> str:
    """Did `k/n` actually settle its comparison with `threshold`?

    Returns `"below"`, `"above"` or `"unresolved"`. The point of the third value
    is that it is the honest answer far more often than a point estimate makes
    it look, and it is the answer a threshold sitting inside the interval
    deserves whichever side the fraction happens to fall on.
    """
    lo, hi = wilson_interval(k, n, z=z)
    if hi < threshold:
        return "below"
    if lo > threshold:
        return "above"
    return "unresolved"


# ---------------------------------------------------------------------------
# the probe set
# ---------------------------------------------------------------------------


@dataclass
class Probe:
    """One prompt and the mechanical check applied to its reply."""

    prompt: str
    kind: str
    #: Words any of which counts as a hit. Matched on word boundaries, with a
    #: trailing "s" allowed, so "rabbits" satisfies "rabbit" -- a character model
    #: that pluralises is not off topic.
    expect: tuple[str, ...] = ()
    #: For `kind == "list"`: how many DISTINCT members of `expect` are wanted.
    want: int = 1


#: Animals, colours and fruits that a TinyStories-trained model plausibly knows.
#: Deliberately short and common: the point is to count whether the reply named
#: any at all, not to test breadth of vocabulary.
_ANIMALS = ("dog", "cat", "bird", "fish", "cow", "horse", "pig", "duck", "frog",
            "bear", "lion", "mouse", "rabbit", "sheep", "chicken", "elephant",
            "monkey", "tiger", "goat", "fox", "owl", "bee", "snake", "turtle")
_COLOURS = ("red", "blue", "green", "yellow", "orange", "purple", "pink",
            "brown", "black", "white", "grey", "gray")
_FRUITS = ("apple", "banana", "orange", "pear", "grape", "strawberry", "lemon",
           "peach", "cherry", "melon", "plum", "mango")

#: Single-turn probes. Grouped by `kind`; every kind is reported separately
#: because they fail for different reasons and averaging them hides that.
PROBES: tuple[Probe, ...] = (
    # -- topicality: the headline failure ---------------------------------
    Probe("tell me a story about a rabbit", "topic", ("rabbit", "bunny")),
    Probe("tell me a story about a dragon", "topic", ("dragon",)),
    Probe("tell me a story about a robot", "topic", ("robot",)),
    Probe("tell me a story about a boat", "topic", ("boat", "ship")),
    Probe("tell me a story about a snowman", "topic", ("snowman", "snow")),
    Probe("tell me a story about a pirate", "topic", ("pirate",)),
    Probe("tell me a story about a little girl named Mia", "topic", ("mia",)),
    Probe("tell me a story about a boy and his kite", "topic", ("kite",)),
    Probe("tell me a story about a frog in a pond", "topic", ("frog",)),
    Probe("tell me a story about a birthday cake", "topic", ("cake",)),
    Probe("write me a little story about a train", "topic", ("train",)),
    Probe("can you tell me a story about a farmer", "topic", ("farmer", "farm")),
    Probe("i want a story about the moon", "topic", ("moon",)),
    Probe("make up a story about a spider", "topic", ("spider",)),
    Probe("tell me a story about a teacher", "topic", ("teacher", "school")),
    Probe("tell me a story about a bicycle", "topic", ("bicycle", "bike")),

    # -- instruction following: name/list N things -------------------------
    Probe("name three animals", "list", _ANIMALS, want=3),
    Probe("name three colours", "list", _COLOURS, want=3),
    Probe("list three fruits", "list", _FRUITS, want=3),
    Probe("can you name an animal?", "list", _ANIMALS, want=1),
    Probe("tell me the name of a colour", "list", _COLOURS, want=1),

    # -- the register the persona data stipulates --------------------------
    Probe("hello", "social", ("hi", "hello", "hey")),
    Probe("hey there", "social", ("hi", "hello", "hey")),
    Probe("how are you?", "social", ("good", "well", "fine", "great", "thanks",
                                     "thank", "ok", "okay")),
    # The acknowledgement and farewell lists were WIDENED after reading the
    # baseline's own draws, and the change is disclosed in docs/chat/QUALITY.md
    # with the number it moved. "Happy to help." and "See you. Have a good one."
    # were being scored as failures, which is a scorer defect and not a model
    # one -- but widening a metric after seeing what a model produced is exactly
    # how a metric gets gamed, so: the additions are ordinary English forms any
    # list written in advance should have had, they were applied to every arm
    # equally by re-scoring saved draws, and the baseline is reported under both
    # versions.
    Probe("thanks!", "social", ("welcome", "any time", "anytime", "pleasure",
                                "glad", "no problem", "sure", "help", "course")),
    Probe("goodbye", "social", ("bye", "goodbye", "care", "later", "night",
                                "see you", "farewell", "so long")),
    Probe("what are you?", "identity", ("spiking", "network", "neural", "model")),
    Probe("are you a human?", "identity", ("not", "no", "spiking", "network")),
    Probe("what can you do?", "identity", ("story", "stories", "talk",
                                           "conversation", "chat", "small")),
    Probe("do you remember our last conversation?", "identity",
          ("not", "no", "cannot", "can't", "don't")),

    # -- expected failures, retained ---------------------------------------
    Probe("what is 17 plus 25?", "fact", ("42",)),
    Probe("what is 2 plus 2?", "fact", ("4", "four")),
    Probe("what is the capital of France?", "fact", ("paris",)),
    Probe("what colour is the sky?", "fact", ("blue",)),
    Probe("what colour is grass?", "fact", ("green",)),
    Probe("how many legs does a dog have?", "fact", ("4", "four")),
    Probe("write a haiku about autumn leaves falling in a quiet garden",
          "fact", ("leaf", "leaves", "autumn", "fall", "garden")),
)

#: A reply to "name three animals" longer than this is a story, whatever words
#: it contains. Used only by the `list_strict` diagnostic.
_LIST_MAX_CHARS = 120

#: Kinds whose replies should be ABOUT something. A greeting or identity line
#: here is the model dodging the prompt, which is what `fallback_rate` counts.
_SUBSTANTIVE = ("topic", "list", "fact")

#: Phrases that mark a reply as the model's stock evasion. Drawn from what
#: `snnchat.persona` stipulates plus the two openers the model actually reaches
#: for; matched at the START of the reply for the greetings, because "hi" inside
#: a story is not an evasion.
_FALLBACK_PREFIXES = ("hi ", "hi!", "hi,", "hi.", "hello", "hey ", "hey!",
                      "hey.", "hey,")
_FALLBACK_ANYWHERE = ("spiking neural network", "i'm a spiking", "i am a spiking",
                      "trained on characters", "i don't really have a name")


def _word_hit(text: str, words) -> list[str]:
    """Which of `words` occur in `text`, on word boundaries, plural allowed."""
    low = text.lower()
    found = []
    for w in words:
        if re.search(r"\b" + re.escape(w.lower()) + r"(s|es)?\b", low):
            found.append(w)
    return found


#: A reply to "name three animals" that opens like a TinyStories narrative is
#: not an answer, whatever else it is. Added after reading `chat-v3d-aligned`'s
#: transcript, where the greeting evasion the `fallback_rate` was built to catch
#: had been REPLACED by a story evasion that it does not catch -- the rate fell
#: to zero while the model was dodging just as often, in a new shape. A metric
#: that only counts the old failure will always report progress.
_STORY_OPENERS = ("once upon a time", "one day", "there was a", "there once",
                  "one sunny day", "long ago")


def _is_story_opener(text: str) -> bool:
    """Branch one of `_is_story`, alone. Length-independent by construction.

    Split out rather than inlined because `PREDICTION_v5.md` §4 declared, before
    that round's numbers existed, that the two branches have different length
    behaviour and that the opener branch would be reported as a labelled
    diagnostic. A decomposition that is recomputed by hand in a document is a
    decomposition nobody can check.
    """
    low = text.strip().lower()
    return any(low.startswith(o) for o in _STORY_OPENERS)


def _is_story(text: str) -> bool:
    low = text.strip().lower()
    if _is_story_opener(low):
        return True
    # "a little boy named Tim" is the register's signature and survives a
    # non-standard opening.
    return " named " in low and len(low) > 80


#: Every reply string `snnchat.persona` stipulates, normalised for comparison.
#:
#: WHY `_is_fallback` IS NOT ENOUGH, MEASURED
#: -----------------------------------------
#: `_FALLBACK_ANYWHERE` is five hand-picked phrases, and it misses most of what
#: the model actually reaches for. On `chat-v3d-aligned`'s committed draws,
#: `_is_fallback` fires on 12 of 1,792 substantive candidates (0.0067) while
#: ~150 of them (0.084) are persona reply strings; at the pinned `n=1, λ=0`
#: reading row, **7 of the 20 `list` picks are verbatim persona text** --
#: "name three animals" -> "That's beyond me, I'm afraid.", "list three fruits"
#: -> "You can call me SNN. That's what I am." -- and `fallback_rate` reports
#: **0.0000** on that same row.
#:
#: That matters more than its size suggests: `list` is 20 draws at weight 0.3
#: and carries ~89 % of the headline's between-training-seed variance, so the
#: column that decides the ship gate is about a third memorised boilerplate,
#: scored by a penalty term that cannot see it. It is the failure
#: `QUALITY.md` §5 already named once -- "a metric that counts only the failure
#: a model used to have will always report progress" -- recurring in a new
#: shape, which is exactly what that section predicted would happen next.
def _normalise_for_match(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


#: A prefix match shorter than this is not evidence of recitation.
#:
#: CALIBRATED, NOT CHOSEN. At 15 characters this detector fired on **53 of the
#: 64 `topic` picks** of the shipped arm, because 15 characters of persona's
#: `story_request` narratives is "once upon a time" and so is the opening of
#: every story the model tells. At 20 it fires on 0 of 64 `topic` and 0 of 28
#: `fact` picks while holding 8 of 20 on `list`, and that is stable at 24 and 30
#: -- so the surviving matches are not a threshold artefact. The calibration is
#: in `docs/chat/QUALITY_v8.md` §9 with the table.
_CANNED_MIN_CHARS = 20


def _persona_replies() -> frozenset[str]:
    """Stipulated persona replies, minus the ones that are stories.

    `persona.TOPICS["story_request"]` maps "tell me something" onto a handful of
    written-out narratives. Those are excluded for a reason of division of
    labour rather than convenience: answering "name three animals" with a story
    is already counted, by `story_dodge`, and a reply that shares an opening
    with a stock narrative is evidence about the REGISTER the model fell into,
    not about it reciting a specific stipulated answer. Counting it here would
    double-count one failure and hide the other.
    """
    from snnchat import persona

    out = {r for _prompts, replies in persona.TOPICS.values() for r in replies}
    out |= {r for _q, r in persona._FOLLOW_UPS}
    out = {r for r in out if not _is_story(r)}
    return frozenset(
        _normalise_for_match(r) for r in out if len(r.strip()) >= _CANNED_MIN_CHARS
    )


_PERSONA_REPLIES = _persona_replies()


def _is_canned(text: str) -> bool:
    """True if the reply is (a prefix of, or prefixed by) stipulated persona text.

    Matched in BOTH directions because both failures occur: the model emits a
    persona line and stops, or emits one and carries on into something else. A
    prefix either way over at least `_CANNED_MIN_CHARS` normalised characters is
    recitation, not an answer.

    Deliberately NOT in the headline. `headline` has to stay comparable with the
    21 committed arms scored before this existed, and a term added now would
    silently rewrite five rounds of ship decisions. `canned_rate` is reported
    beside it as its own column; what to do about it is a decision for a
    pre-registration, not for a scorer.
    """
    low = _normalise_for_match(text)
    if len(low) < _CANNED_MIN_CHARS:
        return False
    for r in _PERSONA_REPLIES:
        if low.startswith(r[:_CANNED_MIN_CHARS]) or r.startswith(low[:_CANNED_MIN_CHARS]):
            return True
    return False


def _is_fallback(text: str) -> bool:
    low = text.strip().lower()
    if any(low.startswith(p) for p in _FALLBACK_PREFIXES):
        return True
    return any(p in low for p in _FALLBACK_ANYWHERE)


def _has_repeated_block(text: str, block: int = 12) -> bool:
    """True if some block of `block` characters is immediately repeated."""
    for i in range(len(text) - 2 * block + 1):
        if text[i:i + block] == text[i + block:i + 2 * block]:
            return True
    return False


def _distinct2(text: str) -> float:
    """Distinct word bigrams / total word bigrams. 1.0 = nothing repeats."""
    words = re.findall(r"[a-z']+", text.lower())
    if len(words) < 3:
        return 1.0
    grams = [(words[i], words[i + 1]) for i in range(len(words) - 1)]
    return len(set(grams)) / len(grams)


# ---------------------------------------------------------------------------
# generation: draw once, score many times
# ---------------------------------------------------------------------------


@dataclass
class ProbeDraw:
    """Every candidate drawn for one (probe, seed), with its two scores."""

    prompt: str
    kind: str
    seed: int
    texts: list[str] = field(default_factory=list)
    logp_cond: list[float] = field(default_factory=list)
    logp_null: list[float] = field(default_factory=list)
    n_scored: list[int] = field(default_factory=list)
    closed: list[bool] = field(default_factory=list)
    #: The temperature each candidate was proposed at. Constant unless the draw
    #: used a ladder; saved per candidate so a later analysis can ask which end of
    #: the ladder the winners came from, which is the only way to tell "the
    #: spread helped" from "the spread happened to be on".
    temperature: list[float] = field(default_factory=list)


@torch.no_grad()
def collect(
    model,
    *,
    probes=PROBES,
    seeds=(0, 1, 2, 3),
    n: int = 16,
    params: SamplingParams | None = None,
    null: str = "empty_user",
    device=None,
    tok: ChatTokenizer | None = None,
    progress: bool = True,
    temperature_spread: float = 0.0,
    probe_index: tuple[int, ...] | None = None,
) -> dict:
    """Draw `n` candidates for every (probe, seed) and score them both ways.

    One GPU pass. The result is a plain dict, so it serialises to JSON and a
    reranking sweep afterwards is arithmetic on a file rather than a rerun --
    which is what makes every setting comparable on the identical draws.

    `temperature_spread` proposes the candidates over a range of temperatures
    instead of all at `params.temperature`; see `RerankParams.temperature_spread`
    for why that is a change to the pool and not to the scoring. It changes the
    draws, so it cannot be swept over a saved file the way `(n, lam)` can -- an
    arm drawn with a spread and one drawn without are two runs of this function,
    and `compare_quality.py` shows them as two rows.

    `probe_index` overrides the index that enters each probe's sampling seed. By
    default a probe's seed is its position in `probes`, which means **drawing a
    SUBSET of the battery does not reproduce that subset's draws from the full
    battery** -- the positions shift and every seed with them. Passing each
    probe's index in `PROBES` restores them exactly, which is what lets a metric
    defined over 12 of the 37 probes be resampled at many seeds without paying
    for the other 25 and without the result being a different experiment.
    `None` is the original behaviour, bit for bit.
    """
    tok = tok or ChatTokenizer()
    device = device or next(model.parameters()).device
    base = params or SamplingParams()
    ladder = temperature_ladder(base.temperature, n, temperature_spread)
    temps = ladder or [base.temperature] * n
    if probe_index is None:
        probe_index = tuple(range(len(probes)))
    elif len(probe_index) != len(probes):
        raise ValueError(
            f"probe_index has {len(probe_index)} entries for {len(probes)} probes"
        )
    draws: list[ProbeDraw] = []
    was_training = model.training
    model.eval()
    try:
        for pi, probe in enumerate(probes):
            prefix = [BOS, *tok.render_turn("user", probe.prompt), BOT]
            idx = torch.tensor([prefix], dtype=torch.int64, device=device)
            for seed in seeds:
                logits, state, _ = model(idx, state=None)
                sp = base.copy(seed=int(seed) * 10_007 + probe_index[pi])
                cands = sample_candidates(model, logits[:, -1, :].float(), state, sp, n,
                                          temperatures=ladder)
                nulls = score_under_prefix(
                    model, null_prefix_ids(null), [c.scored for c in cands], device
                )
                draws.append(ProbeDraw(
                    prompt=probe.prompt,
                    kind=probe.kind,
                    seed=int(seed),
                    texts=[tok.decode_visible(c.ids).strip() for c in cands],
                    logp_cond=[c.logp_cond for c in cands],
                    logp_null=[float(v) for v in nulls],
                    n_scored=[max(len(c.scored), 1) for c in cands],
                    closed=[len(c.scored) > len(c.ids) for c in cands],
                    temperature=list(temps),
                ))
            if progress:
                print(f"  [{pi + 1}/{len(probes)}] {probe.prompt!r}", flush=True)
    finally:
        if was_training:
            model.train()
    return {
        "n_candidates": n,
        "seeds": list(seeds),
        "null": null,
        "sampling": repr(base),
        "temperature_spread": float(temperature_spread),
        "probe_index": list(probe_index),
        "probes": [asdict(p) for p in probes],
        "draws": [asdict(d) for d in draws],
    }


# ---------------------------------------------------------------------------
# scoring
# ---------------------------------------------------------------------------


def score(collected: dict, *, n: int | None = None, lam: float = 0.0,
          min_chars: int = 12, probes=None) -> dict:
    """Select a winner per (probe, seed) under `(n, lam)` and score the set.

    `n` may be any value up to the number of candidates collected: the rows are
    independent draws, so the first `n` of them are a sample of size `n`.

    `probes` overrides the expectations stored in the draw file. It exists so a
    correction to the scorer can be applied to arms that were DRAWN before the
    correction, without regenerating anything -- every arm is then scored by one
    rule, which is the only way a comparison between them means anything.
    """
    stored = {p["prompt"]: p for p in collected["probes"]}
    if probes is not None:
        current = {p.prompt: asdict(p) for p in probes}
        missing = set(stored) - set(current)
        if missing:
            raise ValueError(
                f"the draw file has probes this scorer does not: {sorted(missing)[:3]}. "
                f"Re-collect rather than scoring two probe sets as one."
            )
        stored = {k: current[k] for k in stored}
    by_prompt = stored
    n = n or collected["n_candidates"]
    kinds: dict[str, list[float]] = {}
    per_probe: dict[str, list[float]] = {}
    picks: list[dict] = []
    fallback, repeated, closed, lengths, d2, strict = [], [], [], [], [], []
    story_dodge: list[bool] = []
    story_opener: list[bool] = []
    #: Recitation of stipulated persona text where an answer was asked for. See
    #: `_is_canned`. Reported beside `fallback_rate`, never folded into it: the
    #: two count different things and `fallback_rate` is load-bearing in the
    #: headline, which must stay comparable with the arms already scored.
    canned: list[bool] = []
    #: The same, over the `list` probes alone -- the 20-draw column that carries
    #: ~89 % of the headline's between-training-seed variance, and where the
    #: contamination is concentrated.
    canned_list: list[bool] = []
    #: Mean per-character log-probability of the SELECTED reply under the model
    #: itself. This is the quantity the anti-LM term trades away, so it is the
    #: one that has to be watched while trading it: a reply the model considers
    #: very unlikely is, at this scale, usually word salad. It costs nothing --
    #: the numerator was recorded during sampling.
    selected_logp: list[float] = []
    #: Every candidate's hit, not just the selected one. See `pool_by_kind`.
    pool: dict[str, list[float]] = {}

    for d in collected["draws"]:
        k = min(n, len(d["texts"]))
        scores = [
            (d["logp_cond"][i] - lam * d["logp_null"][i]) / d["n_scored"][i]
            for i in range(k)
        ]
        order = [i for i in range(k) if len(d["texts"][i]) >= min_chars] or list(range(k))
        best = max(order, key=lambda i: scores[i])
        text = d["texts"][best]
        probe = by_prompt[d["prompt"]]
        hit = _score_one(probe, text)
        pool.setdefault(d["kind"], []).extend(
            _score_one(probe, t) for t in d["texts"]
        )
        kinds.setdefault(d["kind"], []).append(hit)
        per_probe.setdefault(d["prompt"], []).append(hit)
        picks.append({"prompt": d["prompt"], "kind": d["kind"], "seed": d["seed"],
                      "text": text, "hit": hit, "index": best})
        lengths.append(len(text))
        selected_logp.append(d["logp_cond"][best] / max(d["n_scored"][best], 1))
        closed.append(bool(d["closed"][best]))
        repeated.append(_has_repeated_block(text))
        d2.append(_distinct2(text))
        if d["kind"] in _SUBSTANTIVE:
            fallback.append(_is_fallback(text))
            canned.append(_is_canned(text))
            if d["kind"] == "list":
                canned_list.append(_is_canned(text))
        if d["kind"] in ("list", "fact"):
            # Neither of these asks for a narrative, so a narrative is a dodge.
            story_dodge.append(_is_story(text))
            story_opener.append(_is_story_opener(text))
        if d["kind"] == "list":
            # `list` counts lexicon members anywhere in the reply, and a STORY
            # that happens to mention a dog and a fish satisfies "name three
            # animals" without following the instruction at all -- observed in
            # `chat-v3a-data`'s own draws. The strict variant additionally
            # requires the reply to be short enough to be a list rather than a
            # narrative. It is a diagnostic column, not part of the headline,
            # which stays as it was pre-registered.
            strict.append(hit if len(text) <= _LIST_MAX_CHARS else 0.0)

    out = {
        "n": n,
        "lambda": lam,
        "by_kind": {k: round(float(np.mean(v)), 4) for k, v in sorted(kinds.items())},
        "list_strict": round(float(np.mean(strict)), 4) if strict else 0.0,
        # How often a prompt that asked for no narrative got one anyway.
        "story_dodge": round(float(np.mean(story_dodge)), 4) if story_dodge else 0.0,
        # The same number with its evidence attached: numerator, denominator,
        # lattice spacing and Wilson interval. `story_dodge` is compared against
        # a pre-registered threshold in every round that reports it, and
        # `docs/chat/CONVENTIONS.md` requires the comparison carry an interval.
        # Emitted unconditionally so that an arm scored before that convention
        # and one scored after are still one file format.
        "story_dodge_ci": rate_ci(int(sum(story_dodge)), len(story_dodge)),
        # The length-independent half of `_is_story`, which `PREDICTION_v5.md`
        # §4 required be reported alongside as a labelled diagnostic: the
        # " named " branch fires more often on a longer reply for arithmetic
        # reasons, the opener branch does not.
        "story_dodge_opener_ci": rate_ci(int(sum(story_opener)), len(story_opener)),
        "mean_logp": round(float(np.mean(selected_logp)), 4) if selected_logp else 0.0,
        # The same hit rate over EVERY candidate drawn rather than over the one
        # selected. It is 16x the sample size and it is not a measure of the
        # decoder at all -- it is the model's underlying propensity to mention
        # what it was asked about. That resolution matters: the selected-reply
        # `topic` column has 64 samples per arm, so its binomial standard error
        # near p = 0.08 is 0.034 and nothing below a 0.07 difference is
        # resolvable. This column has ~1,024 and resolves to about 0.02.
        "pool_by_kind": {k: round(float(np.mean(v)), 4) for k, v in sorted(pool.items())},
        "pool_n": {k: len(v) for k, v in sorted(pool.items())},
        "fallback_rate": round(float(np.mean(fallback)), 4) if fallback else 0.0,
        "canned_rate": round(float(np.mean(canned)), 4) if canned else 0.0,
        "canned_rate_list": round(float(np.mean(canned_list)), 4) if canned_list else 0.0,
        "canned_n": [int(np.sum(canned)), len(canned)],
        "canned_n_list": [int(np.sum(canned_list)), len(canned_list)],
        "repeat_rate": round(float(np.mean(repeated)), 4),
        "closed_rate": round(float(np.mean(closed)), 4),
        "mean_chars": round(float(np.mean(lengths)), 1),
        "distinct2": round(float(np.mean(d2)), 4),
        "per_probe": {k: round(float(np.mean(v)), 4) for k, v in per_probe.items()},
        "picks": picks,
    }
    # One headline number, so a sweep can be ordered. Topicality and instruction
    # following are what this work is trying to move; the fallback rate is
    # subtracted because answering "name three animals" with the identity line
    # is the specific behaviour being attacked. `fact` is NOT in it -- nothing
    # here can teach the model that Paris is the capital of France, and putting
    # it in the objective would just add noise.
    out["headline"] = round(
        0.5 * out["by_kind"].get("topic", 0.0)
        + 0.3 * out["by_kind"].get("list", 0.0)
        + 0.2 * out["by_kind"].get("social", 0.0)
        - 0.2 * out["fallback_rate"],
        4,
    )
    return out


def _score_one(probe: dict, text: str) -> float:
    words = probe["expect"]
    if not words:
        return 0.0
    found = _word_hit(text, words)
    want = max(int(probe.get("want", 1)), 1)
    if want == 1:
        return 1.0 if found else 0.0
    return min(len(set(found)), want) / want


# ---------------------------------------------------------------------------
# prompt dependence: the metric with no lexicon and no sampler in it
# ---------------------------------------------------------------------------


def extract_pairs(ids: np.ndarray, *, max_pairs: int = 256,
                  user_range=(12, 200), bot_range=(24, 320)) -> list[tuple[list[int], list[int]]]:
    """Pull `(user, bot)` turn pairs out of a packed id stream.

    A turn is `<|user|> text <|eot|>` immediately followed by `<|bot|> text
    <|eot|>`, which is exactly what `ChatTokenizer.render_conversation` writes,
    so this is a parse rather than a heuristic -- and it cannot be fooled by
    text, because `encode` cannot emit a marker id
    (`tests/test_snnchat.py::test_encode_cannot_forge_a_turn_marker`).
    """
    marks = np.flatnonzero(ids < 4)
    pairs: list[tuple[list[int], list[int]]] = []
    for a in range(len(marks) - 3):
        i, j, k, m = marks[a], marks[a + 1], marks[a + 2], marks[a + 3]
        if ids[i] != USER or ids[j] != EOT or ids[k] != BOT or ids[m] != EOT:
            continue
        if k != j + 1:
            continue
        user = ids[i + 1:j]
        bot = ids[k + 1:m]
        if not (user_range[0] <= len(user) <= user_range[1]):
            continue
        if not (bot_range[0] <= len(bot) <= bot_range[1]):
            continue
        pairs.append(([int(v) for v in user], [int(v) for v in bot]))
        if len(pairs) >= max_pairs:
            break
    return pairs


@torch.no_grad()
def _score_rows(model, prefixes: list[list[int]], sequences: list[list[int]],
                device, *, filler: int = EOT) -> list[float]:
    """`log P(seq | prefix)` with a DIFFERENT prefix per row, one batched call."""
    widths = [len(p) + len(s) for p, s in zip(prefixes, sequences)]
    width = max(widths)
    b = len(prefixes)
    rows = torch.full((b, width), filler, dtype=torch.int64)
    mask = torch.zeros(b, width - 1, dtype=torch.bool)
    for r, (pre, seq) in enumerate(zip(prefixes, sequences)):
        p = len(pre)
        rows[r, :p] = torch.tensor(pre, dtype=torch.int64)
        rows[r, p:p + len(seq)] = torch.tensor(seq, dtype=torch.int64)
        mask[r, p - 1:p - 1 + len(seq)] = True
    rows = rows.to(device)
    mask = mask.to(device)
    logits, _, _ = model(rows, state=None)
    lp = F.log_softmax(logits[:, :-1, :].float(), dim=-1)
    picked = lp.gather(2, rows[:, 1:, None]).squeeze(2)
    return (picked * mask).sum(dim=1).tolist()


@torch.no_grad()
def prompt_dependence(
    model,
    corpus,
    *,
    sources=("soda", "alpaca", "dolly", "oasst1"),
    per_source: int = 128,
    batch_size: int = 32,
    device=None,
    check: bool = True,
    user_range=(12, 200),
    bot_range=(24, 320),
) -> dict:
    """How many bits per character the model's reply owes to its own prompt.

    Each held-out `(user, bot)` pair is scored twice under identical code: once
    behind its own user turn, once behind the NEXT pair's user turn. The shift
    is a fixed rotation rather than a shuffle so the control has the same
    length and register distribution as the treatment and does not depend on a
    seed.

    Returns bits per character for both, and `delta = shuffled - matched`. A
    model that ignored the prompt would score `delta = 0`; a larger delta means
    more of the reply is attributable to what was asked.

    `check=True` re-derives one batch's matched log-probability by feeding the
    conversation through the model ONE CHARACTER AT A TIME -- the path
    `ChatSession` actually uses -- and fails if the two disagree. The masked
    batched scorer above is the kind of index arithmetic that is wrong by one
    and still produces a plausible number, so it is made to reproduce a number
    it did not produce (`snn-v2-research-protocol`).
    """
    device = device or next(model.parameters()).device
    was_training = model.training
    model.eval()
    out: dict = {"per_source": {}}
    tot_m, tot_s, tot_c = 0.0, 0.0, 0
    try:
        for name in sources:
            data = corpus.val(name)
            if data is None:
                continue
            pairs = extract_pairs(np.asarray(data), max_pairs=per_source,
                                  user_range=user_range, bot_range=bot_range)
            if len(pairs) < 8:
                continue
            m_nats, s_nats, chars = 0.0, 0.0, 0
            for start in range(0, len(pairs), batch_size):
                chunk = pairs[start:start + batch_size]
                if len(chunk) < 2:
                    continue
                pre_m = [[BOS, USER, *u, EOT, BOT] for u, _ in chunk]
                rot = chunk[1:] + chunk[:1]
                pre_s = [[BOS, USER, *u, EOT, BOT] for u, _ in rot]
                seqs = [[*b, EOT] for _, b in chunk]
                lm = _score_rows(model, pre_m, seqs, device)
                ls = _score_rows(model, pre_s, seqs, device)
                if check and start == 0 and name == sources[0]:
                    out["check"] = _sequential_check(model, pre_m[0], seqs[0], lm[0], device)
                m_nats -= sum(lm)
                s_nats -= sum(ls)
                chars += sum(len(s) for s in seqs)
            if chars:
                out["per_source"][name] = {
                    "pairs": len(pairs),
                    "chars": chars,
                    "bpc_matched": round(m_nats / chars * _LOG2E, 5),
                    "bpc_shuffled": round(s_nats / chars * _LOG2E, 5),
                    "delta": round((s_nats - m_nats) / chars * _LOG2E, 5),
                }
                tot_m += m_nats
                tot_s += s_nats
                tot_c += chars
    finally:
        if was_training:
            model.train()
    if tot_c:
        out["bpc_matched"] = round(tot_m / tot_c * _LOG2E, 5)
        out["bpc_shuffled"] = round(tot_s / tot_c * _LOG2E, 5)
        out["delta"] = round((tot_s - tot_m) / tot_c * _LOG2E, 5)
        out["chars"] = tot_c
    return out


@torch.no_grad()
def _sequential_check(model, prefix, seq, batched_logp, device, tol=2e-3) -> dict:
    """Re-score one row one character at a time; raise if it disagrees."""
    state = None
    logits, state, _ = model(
        torch.tensor([prefix], dtype=torch.int64, device=device), state=state
    )
    total = 0.0
    last = logits[:, -1, :].float()
    for ch in seq:
        total += float(F.log_softmax(last, dim=-1)[0, ch])
        last, state, _ = model(
            torch.tensor([[ch]], dtype=torch.int64, device=device), state=state
        )
        last = last[:, -1, :].float()
    rel = abs(total - batched_logp) / max(abs(total), 1.0)
    if rel > tol:
        raise RuntimeError(
            f"the batched masked scorer and the one-character-at-a-time path "
            f"disagree: {batched_logp:.6f} vs {total:.6f} nats "
            f"(relative {rel:.2e} > {tol:.0e}). The mask or the off-by-one in "
            f"_score_rows is wrong; do not widen this tolerance."
        )
    return {"batched": round(batched_logp, 5), "sequential": round(total, 5),
            "relative": float(f"{rel:.3e}"), "chars": len(seq)}


