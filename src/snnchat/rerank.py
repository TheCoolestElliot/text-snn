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

WHAT IT CANNOT DO
-----------------
It selects among the replies the model would have produced anyway. If every one
of N candidates is off topic, reranking returns the least-bad of N off-topic
replies. It buys no knowledge, no arithmetic, and no memory across turns -- and
the numbers in `docs/chat/QUALITY.md` show exactly that shape: topicality moves
a lot, the factual battery does not move at all.

Not part of the research protocol. No number here is a reported figure.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

# `_filter` and `_loop_penalty` are the sampler's own truncation and cycle
# breaker. Imported rather than reimplemented so that a candidate drawn here is
# drawn from the identical distribution the single-candidate path uses -- two
# copies of a nucleus filter that disagree by one index would make every
# comparison in docs/chat/QUALITY.md meaningless.
from snnchat.generate import SamplingParams, _filter, _loop_penalty
from snnchat.tokenizer import BOS, BOT, EOT, USER

__all__ = [
    "RerankParams",
    "Candidate",
    "sample_candidates",
    "score_under_prefix",
    "null_prefix_ids",
    "trim_to_sentence",
    "rerank",
]

_STOP_IDS = (EOT, USER, BOS)


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

    def __post_init__(self) -> None:
        if self.n < 1:
            raise ValueError(f"n must be >= 1, got {self.n}")
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
        return out


@dataclass
class Candidate:
    """One drawn reply and the two log-probabilities it is ranked by."""

    ids: list[int]              #: visible characters only, no stop id
    scored: list[int]           #: what both log-probabilities cover: ids + stop id if any
    logp_cond: float            #: log P(scored | conversation so far)
    logp_null: float = 0.0      #: log P(scored | null context)
    score: float = 0.0

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
) -> list[Candidate]:
    """Draw `n` independent replies from one context, in one batch.

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

    gen = None
    if params.seed is not None:
        gen = torch.Generator(device=device)
        gen.manual_seed(int(params.seed))

    out: list[list[int]] = [[] for _ in range(n)]
    recent: list[list[int]] = [[] for _ in range(n)]
    stopped: list[int | None] = [None] * n
    logp = torch.zeros(n, device=device, dtype=torch.float32)
    alive = torch.ones(n, dtype=torch.bool, device=device)
    stop_set = set(stop_ids)
    stop_t = torch.tensor(sorted(stop_set), device=device)

    for _ in range(params.max_new):
        work = logits.clone()
        if params.loop_penalty > 0.0:
            for r in range(n):
                if recent[r]:
                    _loop_penalty(recent[r], work[r], params)
        probs = F.softmax(_filter(work, params, temps), dim=-1)
        nxt = torch.multinomial(probs, 1, generator=gen).squeeze(-1)      # [n]
        raw = F.log_softmax(logits, dim=-1).gather(1, nxt[:, None]).squeeze(1)
        logp = logp + torch.where(alive, raw, torch.zeros_like(raw))

        is_stop = (nxt[:, None] == stop_t[None, :]).any(dim=1)
        just_stopped = alive & is_stop
        alive = alive & ~is_stop

        nxt_cpu = nxt.tolist()
        for r in range(n):
            if bool(just_stopped[r]):
                stopped[r] = nxt_cpu[r]
            elif bool(alive[r]):
                out[r].append(nxt_cpu[r])
                recent[r].append(nxt_cpu[r])
                if len(recent[r]) > 2 * params.loop_max_block:
                    del recent[r][0]
        if not bool(alive.any()):
            break
        # Dead rows are fed EOT: harmless, since their state is never read
        # again, and it keeps the batch rectangular so the scan stays one call.
        feed = torch.where(alive, nxt, torch.full_like(nxt, EOT))
        logits, state = _feed(model, feed[:, None], state)

    lp = logp.tolist()
    return [
        Candidate(
            ids=out[r],
            scored=out[r] + ([stopped[r]] if stopped[r] is not None else []),
            logp_cond=float(lp[r]),
        )
        for r in range(n)
    ]


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


def rerank(
    model,
    logits: torch.Tensor,
    state,
    params: SamplingParams,
    rp: RerankParams,
    *,
    device=None,
    stop_ids=_STOP_IDS,
) -> tuple[Candidate, list[Candidate]]:
    """Draw `rp.n` candidates and return `(winner, all_candidates)`.

    With `rp.n == 1` the null-context pass is skipped entirely, so the ordinary
    single-sample path pays nothing for this module existing.
    """
    cands = sample_candidates(
        model, logits, state, params, rp.n, stop_ids=stop_ids,
        temperatures=rp.temperature_ladder(params.temperature),
    )
    if len(cands) == 1:
        cands[0].score = cands[0].logp_cond / max(cands[0].n_chars, 1)
        return cands[0], cands

    device = device or logits.device
    if rp.lam != 0.0:
        nulls = score_under_prefix(
            model, null_prefix_ids(rp.null), [c.scored for c in cands], device
        )
        for c, v in zip(cands, nulls):
            c.logp_null = float(v)

    for c in cands:
        n_scored = max(len(c.scored), 1)
        c.score = (c.logp_cond - rp.lam * c.logp_null) / n_scored

    # The length guard is applied by PARTITION rather than by penalty: a reply
    # of two characters is not a slightly worse reply, it is a different event
    # (the model closing its turn immediately), and the mean-per-character score
    # cannot compare the two. If every candidate is short, they are ranked among
    # themselves rather than the turn being forced to produce text.
    long_enough = [c for c in cands if c.n_chars >= rp.min_chars]
    pool = long_enough or cands
    winner = max(pool, key=lambda c: c.score)
    return winner, cands
