"""Sampling: how a spiking recurrent net is made to hold a conversation.

THE ONE STRUCTURAL ADVANTAGE THIS MODEL HAS
-------------------------------------------
It is a recurrent network, not a transformer, so there is no context window and
no re-reading. The conversation lives in `[B, 2d]` of membrane state per layer,
and a new character costs the same whether it is the tenth of the session or the
ten-thousandth. `ChatSession` exploits that directly: the state is fed forward
and never reset, so an hour-long conversation runs at constant cost and constant
memory.

The corresponding disadvantage is the one that shapes everything else here. What
the state can still *resolve* is bounded by the slow pole's time constant --
Phase 4 measured the memory horizon of the committed two-compartment neuron at
47-48 characters, and `snnchat.model.spread_slow_poles` is an attempt to buy
more. There is no mechanism, at any temperature, by which this model recalls
something said four turns ago. Sampling settings cannot fix that and this module
does not pretend to.

WHY THERE IS A LOOP BREAKER AND NOT A REPETITION PENALTY
---------------------------------------------------------
The standard repetition penalty divides the logits of previously-emitted tokens.
On a *character* model that is close to nonsense: penalising every character
already seen penalises `e`, space and `t` within a dozen characters of any
sentence, and the text degrades into deliberate misspelling. The failure mode it
is aimed at is real, though -- small models fall into exact cycles ("that's
great! that's great! that's great!") and never leave.

`_loop_penalty` attacks the cycle specifically: it looks for a block of length
`n` that has just repeated back to back, and suppresses only the single
character that would extend it to a third copy. In non-degenerate text no such
block exists and the sampler is untouched, so this costs nothing where it is not
needed -- which is the property a blanket penalty does not have.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence

import torch
import torch.nn.functional as F

from snnchat.tokenizer import BOS, BOT, EOT, USER, ChatTokenizer

__all__ = [
    "SamplingParams",
    "ChatSampler",
    "ChatSession",
    "load_chat_checkpoint",
    "loop_penalty_target",
]


class SamplingParams:
    """Decoding settings, with defaults chosen for a small char model.

    `temperature=0.85` and `top_p=0.92`: a model this size puts a long, ragged
    tail on nearly every position, and sampling from it produces the
    characteristic char-LM word salad. Nucleus truncation removes the tail
    without the hard vocabulary cut that `top_k` imposes on positions where the
    model is genuinely uncertain between many plausible letters.

    `min_p=0.02` is the one that matters most in practice here. After a space at
    the start of a common word the model is often 90 % confident; `top_p` still
    admits the next dozen characters, `min_p` cuts everything below 2 % of the
    peak and lets a confident model be confident. It is cheap insurance against
    a single unlucky draw derailing a sentence, which at char granularity is
    ten times more likely than at token granularity.
    """

    __slots__ = ("temperature", "top_k", "top_p", "min_p", "loop_penalty",
                 "loop_max_block", "max_new", "seed")

    def __init__(
        self,
        temperature: float = 0.85,
        top_k: int = 0,
        top_p: float = 0.92,
        min_p: float = 0.02,
        loop_penalty: float = 6.0,
        loop_max_block: int = 24,
        max_new: int = 400,
        seed: int | None = None,
    ) -> None:
        self.temperature = float(temperature)
        self.top_k = int(top_k)
        self.top_p = float(top_p)
        self.min_p = float(min_p)
        self.loop_penalty = float(loop_penalty)
        self.loop_max_block = int(loop_max_block)
        self.max_new = int(max_new)
        self.seed = seed

    def copy(self, **overrides) -> "SamplingParams":
        kw = {name: getattr(self, name) for name in self.__slots__}
        kw.update(overrides)
        return SamplingParams(**kw)

    def __repr__(self) -> str:
        return (
            f"temperature={self.temperature:g} top_k={self.top_k} "
            f"top_p={self.top_p:g} min_p={self.min_p:g} "
            f"loop_penalty={self.loop_penalty:g} max_new={self.max_new}"
        )


def loop_penalty_target(recent: list[int], params: SamplingParams) -> int | None:
    """The one character a just-repeated block would extend, or None.

    For each block length `n`, if `recent[-n:] == recent[-2n:-n]` then the tail
    has already been said twice, and `recent[-n]` is the character that starts
    the third copy. Only that one id is reported.

    Shortest block first, and return at the first hit: a cycle of period 2 is
    also a cycle of period 4, 6, 8..., and penalising all of them would suppress
    several distinct characters on the strength of one repetition.

    The `recent[-n - 1] != last` line is an exact early-out, not an
    approximation. Taking `i = n - 1` in the element-wise reading of the slice
    comparison gives `recent[-1] == recent[-n - 1]` as a *necessary* condition,
    so a block length that fails it cannot match and the slice never has to be
    built. That matters at the batch sizes `rerank` now draws: the naive form
    builds two slices per block length per row per character, which is ~10^6
    list copies for one turn at n = 128 and was a measurable share of the loop.
    """
    if params.loop_penalty <= 0.0 or len(recent) < 4:
        return None
    limit = min(params.loop_max_block, len(recent) // 2)
    last = recent[-1]
    for n in range(2, limit + 1):
        if recent[-n - 1] != last:
            continue
        if recent[-n:] == recent[-2 * n:-n]:
            return recent[-n]
    return None


def _loop_penalty(recent: list[int], logits: torch.Tensor, params: SamplingParams) -> None:
    """Suppress the character that would extend a just-repeated block. In place."""
    target = loop_penalty_target(recent, params)
    if target is not None:
        logits[target] -= params.loop_penalty


def _filter(
    logits: torch.Tensor, params: SamplingParams, temps: torch.Tensor | None = None
) -> torch.Tensor:
    """temperature -> top_k -> min_p -> top_p, in that order.

    Order matters and this one is deliberate. `min_p` is a threshold relative to
    the *peak* probability, so it must be applied before `top_p` removes mass and
    renormalises -- otherwise its threshold is being taken against a distribution
    that has already been truncated, and the same `min_p` value means something
    different at every position.

    `temps` is an optional `[rows, 1]` tensor of per-row temperatures, used by
    `rerank.sample_candidates` to draw a candidate pool at several temperatures at
    once. When it is None the scalar path runs unchanged, which is what keeps
    every earlier draw reproducible: `None` is not "a tensor of equal values", it
    is the same arithmetic those runs used.
    """
    logits = logits / (temps if temps is not None else max(params.temperature, 1e-6))

    if params.top_k > 0:
        k = min(params.top_k, logits.shape[-1])
        kth = torch.topk(logits, k).values[..., -1, None]
        logits = logits.masked_fill(logits < kth, float("-inf"))

    if params.min_p > 0.0:
        probs = F.softmax(logits, dim=-1)
        threshold = params.min_p * probs.max(dim=-1, keepdim=True).values
        logits = logits.masked_fill(probs < threshold, float("-inf"))

    if 0.0 < params.top_p < 1.0:
        ordered, order = torch.sort(logits, descending=True, dim=-1)
        cumulative = F.softmax(ordered, dim=-1).cumsum(dim=-1)
        # Shift by one so the token that carries the distribution *past* top_p is
        # itself kept. Without the shift a distribution whose top character is
        # already above top_p has every candidate removed.
        drop = cumulative - F.softmax(ordered, dim=-1) > params.top_p
        drop[..., 0] = False
        logits = logits.masked_fill(drop.scatter(-1, order, drop), float("-inf"))

    return logits


class ChatSampler:
    """Stateless helper: prime on a conversation, generate one reply.

    Used for the training loop's periodic samples and for one-shot generation.
    `ChatSession` is the stateful counterpart and is what the REPL uses.
    """

    def __init__(self, model, tokenizer: ChatTokenizer | None = None, *, device=None):
        self.model = model
        self.tok = tokenizer or ChatTokenizer()
        self.device = torch.device(device) if device is not None else next(model.parameters()).device
        #: Off by default and free when off: recording keeps a per-character
        #: reference to each layer's firing-rate scalar, which is K live tensors
        #: per generated character.
        self.record_spikes = False
        self._spike_trace: list[list[torch.Tensor]] = []

    @torch.no_grad()
    def _feed(self, ids: Sequence[int], state):
        """Run `ids` through the model, returning (last logits, new state).

        The whole prefix goes through in ONE call, which is the reason priming a
        long conversation is fast: the model evaluates its per-layer GEMM over
        the entire `[1, L, d]` prefix at once and only the elementwise scan is
        sequential. Feeding character by character would issue L times as many
        GEMMs for an identical result.
        """
        idx = torch.tensor([list(ids)], dtype=torch.int64, device=self.device)
        logits, state, aux = self.model(idx, state=state)
        if self.record_spikes:
            # Kept as device tensors and transferred once at the end of the
            # turn. Calling .item() here would be a host sync per layer per
            # CHARACTER, which on a 300-character reply is 1200 synchronisations
            # and turns an instant reply into a visibly slow one.
            self._spike_trace.append(aux.get("firing_rate", []))
        return logits[:, -1, :].squeeze(0).float(), state

    def start_recording(self) -> None:
        self.record_spikes = True
        self._spike_trace = []

    def collect_spikes(self) -> list[list[float]]:
        """`[character][layer]` mean firing rate, in one host transfer."""
        trace = self._spike_trace
        self._spike_trace = []
        if not trace or not trace[0]:
            return []
        flat = torch.stack([r.reshape(()) for step in trace for r in step]).cpu()
        n_layers = len(trace[0])
        return flat.reshape(len(trace), n_layers).tolist()

    @torch.no_grad()
    def stream(
        self,
        prefix_ids: Sequence[int],
        params: SamplingParams | None = None,
        *,
        state=None,
        stop_ids: Sequence[int] = (EOT, USER, BOS),
    ) -> Iterator[tuple[int, object]]:
        """Prime on `prefix_ids`, then delegate to `stream_from`.

        A caller that needs the post-prefix state even when nothing is generated
        -- `ChatSession.send`, whose user turn is in that prefix -- must feed the
        prefix itself and call `stream_from`. This wrapper cannot hand that state
        back: if the very first sample is a stop id it yields once and stops, and
        the state it yields is the post-prefix one, but a caller reading only the
        loop body would never see it.
        """
        logits, state = self._feed(prefix_ids, state)
        yield from self.stream_from(logits, state, params, stop_ids=stop_ids)

    @torch.no_grad()
    def stream_from(
        self,
        logits: torch.Tensor,
        state,
        params: SamplingParams | None = None,
        *,
        stop_ids: Sequence[int] = (EOT, USER, BOS),
    ) -> Iterator[tuple[int, object]]:
        """Yield `(id, state)` per generated character until a stop id or budget.

        **The state yielded with a text character has already consumed it.** That
        is the contract callers rely on, and getting it backwards is a bug that
        hides: a session whose state lags the text by one character behaves
        almost correctly, and only diverges from a replay of its own transcript
        (`tests/test_snnchat.py::test_rewind_reproduces_the_state_of_a_fresh_replay`
        is what catches it).

        A **stop id is yielded with the state unchanged** and is not fed. It is a
        control signal, not part of the turn, and the caller decides how to close
        the turn -- `ChatSession.send` always closes with `EOT`, whichever stop id
        actually fired.
        """
        params = params or SamplingParams()
        gen = None
        if params.seed is not None:
            gen = torch.Generator(device=self.device)
            gen.manual_seed(int(params.seed))

        recent: list[int] = []
        stop = set(stop_ids)
        for _ in range(params.max_new):
            work = logits.clone()
            _loop_penalty(recent, work, params)
            probs = F.softmax(_filter(work, params), dim=-1)
            nxt = int(torch.multinomial(probs, 1, generator=gen).item())
            if nxt in stop:
                yield nxt, state
                return
            recent.append(nxt)
            if len(recent) > 2 * params.loop_max_block:
                del recent[0]
            logits, state = self._feed([nxt], state)
            yield nxt, state

    def reply(
        self,
        turns: Sequence[tuple[str, str]],
        *,
        max_new: int = 400,
        **overrides,
    ) -> str:
        """One-shot: render `turns`, open a bot turn, and return the text of it."""
        params = SamplingParams(max_new=max_new, **overrides)
        prefix = self.tok.render_conversation(turns) + [BOT]
        out = [i for i, _ in self.stream(prefix, params)]
        return self.tok.decode_visible(out).strip()


class ChatSession:
    """A live conversation. Membrane state persists; nothing is ever re-fed.

    This is what makes the model feel like a recurrent network rather than a
    context window: `send` costs time proportional to the length of *this* turn,
    not to the length of the conversation, and a session that has been running
    for an hour is exactly as fast as one that just started.

    `rewind` exists because that same property makes an ordinary undo impossible
    -- the state is a running summary and there is no way to subtract a turn from
    it. So a rewind replays the retained transcript from scratch. It is the one
    operation whose cost grows with the conversation, and the reason the
    transcript is retained at all.
    """

    def __init__(
        self,
        model,
        tokenizer: ChatTokenizer | None = None,
        *,
        device=None,
        params: SamplingParams | None = None,
        rerank=None,
    ) -> None:
        self.sampler = ChatSampler(model, tokenizer, device=device)
        self.tok = self.sampler.tok
        self.params = params or SamplingParams()
        #: A `snnchat.rerank.RerankParams`, or None for one candidate. Kept as a
        #: plain attribute so `/rerank` in the REPL can change it mid-conversation
        #: -- it affects only how the next reply is chosen and nothing that has
        #: already been fed to the membrane.
        self.rerank = rerank
        #: Every candidate considered for the last reply, for `/candidates`.
        self.last_candidates: list = []
        #: The candidate `rerank` actually returned, and whether the echo
        #: partition was live for that draw. RECORDED rather than recomputed:
        #: `/candidates` renders a decision already taken, and the settings it
        #: was taken under can have changed since (`/echo off`, `/rerank 1`).
        #: Recomputing also silently drops the `min_chars` partition, which put
        #: the star on a draft the selector could never return on 4.8 % of the
        #: committed draws.
        self.last_winner = None
        self.last_echo_on = False
        #: A COPY of the `RerankParams` the last pool was drawn under. Recorded
        #: for the same reason `last_winner` is: `/candidates` renders a decision
        #: already taken, and `self.rerank` is a live object that `/lambda` and
        #: `/rerank` mutate. Scoring a stored pool with a lambda it was never
        #: ranked under prints numbers no selector ever saw.
        self.last_rerank = None
        self.turns: list[tuple[str, str]] = []
        self._state = None
        self._primed = False
        self.chars_fed = 0
        #: Every id ever fed to the model, and the index in it at which each
        #: exchange began. `rewind` replays a prefix of this rather than
        #: re-rendering `self.turns`, because the reply text a caller sees has
        #: been stripped for display and re-rendering it would feed the model a
        #: slightly different byte string than it originally produced. Replaying
        #: the ids is exact by construction.
        self._fed: list[int] = []
        self._marks: list[int] = []
        self.last_spikes: list[list[float]] = []
        #: How many times the CURRENT turn has been regenerated, and which turn
        #: that is. Only read when a seed is fixed; see `regenerate`.
        self._regen = 0
        self._regen_key: tuple | None = None

    # -- the conversation -------------------------------------------------

    @torch.no_grad()
    def _prime(self) -> None:
        if not self._primed:
            _, self._state = self.sampler._feed([BOS], None)
            self._primed = True
            self._fed = [BOS]
            self.chars_fed = 1

    @torch.no_grad()
    def send(
        self,
        text: str,
        *,
        on_char: Callable[[str], None] | None = None,
        params: SamplingParams | None = None,
        prime: str | None = None,
    ) -> str:
        """Append a user turn, generate the reply, return it.

        `on_char` is called with each character as it is produced, which is what
        makes the REPL stream. Special ids are never passed to it -- a turn
        marker is a control signal, not something to print.

        `prime` begins the model's turn with text the CALLER supplies, and the
        model continues from there. See `snnchat.prime` for what that is for and
        for the honesty problem it creates: the primed characters are part of
        the returned reply, so any metric that asks whether the reply contains
        the requested word is answering a question about the caller.

        The prime is appended to `self._fed` and fed through the membrane like
        any other characters, so `rewind` replays it and the session's state is
        the state a transcript containing it would reach. Priming only the
        display would leave the model carrying a turn nobody saw -- the
        incoherence `rewind` exists to avoid.
        """
        self._prime()
        params = params or self.params
        self._marks.append(len(self._fed))
        prefix = self.tok.render_turn("user", text) + [BOT]
        primed_ids: list[int] = []
        if prime:
            # Encoded, so it cannot forge a turn marker -- `encode` is incapable
            # of producing one (`test_encode_cannot_forge_a_turn_marker`), which
            # is what makes it safe to accept caller text here at all.
            # `list(...)`: `encode` returns an ndarray and `prefix` is a list,
            # so `+` would broadcast rather than concatenate.
            primed_ids = [int(i) for i in self.tok.encode(prime)]
            prefix = list(prefix) + primed_ids
        self._fed.extend(prefix)
        self.chars_fed += len(prefix)

        # The prefix is fed HERE, not inside `stream`, so that the user's turn
        # reaches the membrane even when the model ends its own turn immediately
        # and the loop below never runs. Letting `stream` own the priming loses
        # exactly that case, and it is not a rare one -- an empty reply is what a
        # model does when it is confident the turn is over.
        logits, self._state = self.sampler._feed(prefix, self._state)
        if self.sampler.record_spikes:
            # Cleared AFTER the prefix: that call reports one firing rate
            # averaged over the whole prompt, which would sit in the trace as a
            # spurious first "character" and flatten the sparkline.
            self.sampler.start_recording()

        out: list[int] = []
        rp = self.rerank
        if rp is not None and rp.n > 1:
            out = self._reranked(logits, params, rp, on_char, text)
        else:
            for i, state in self.sampler.stream_from(logits, self._state, params):
                if i < self.tok.n_special:
                    break        # a stop id: `stream` left the state untouched
                self._state = state
                out.append(i)
                if on_char is not None:
                    on_char(self.tok.decode_visible([i]))
        self._fed.extend(out)
        self.chars_fed += len(out)
        if self.sampler.record_spikes:
            # Before the EOT feed below, so the trace is exactly the reply.
            self.last_spikes = self.sampler.collect_spikes()

        # The model's turn is closed explicitly whether or not it chose to emit
        # EOT itself. Without this, a reply cut off by `max_new` leaves the state
        # mid-utterance and the next user turn arrives as an interruption -- the
        # model carries on its own sentence instead of answering.
        _, self._state = self.sampler._feed([EOT], self._state)
        self._fed.append(EOT)

        # The primed characters ARE part of the reply -- they were fed to the
        # membrane and they are what the reader sees on the screen. Prepending
        # them here rather than at display time keeps `turns`, `_fed` and the
        # returned string describing the same utterance.
        reply = self.tok.decode_visible(primed_ids + out).strip()
        self.turns.append(("user", text))
        self.turns.append(("bot", reply))
        return reply

    @torch.no_grad()
    def _reranked(self, logits, params, rp, on_char, prompt: str = "") -> list[int]:
        """Draw `rp.n` candidates, keep one, and walk the real state through it.

        THE STATE IS ADVANCED BY A REPLAY, NOT BY THE WINNING BRANCH
        -----------------------------------------------------------
        The candidates are generated in a batch of N, each row carrying its own
        copy of the membrane. The session's own state is the one that existed
        *before* they were drawn, and the reply is fed through it afterwards.

        The alternative -- keeping the winning row's state -- looks equivalent
        and is a trap: those rows are advanced together, a finished row goes on
        being fed a filler character to keep the batch rectangular, and reading a
        state back out of that batch means depending on which rows were still
        alive. Re-feeding the chosen characters costs one forward pass over a
        few hundred characters and is exact by construction, which is the same
        reason `rewind` replays ids rather than re-rendering text.
        """
        import dataclasses

        from snnchat.prime import prime_topic
        from snnchat.rerank import final_ids, prompt_content_words
        from snnchat.rerank import rerank as _rerank

        recording = self.sampler.record_spikes
        # Spike recording traces ONE reply, and the candidate pass is N replies
        # at once. It is turned off for the draw and the winner is then re-fed
        # character by character, which is slower and is what the user asked for
        # by turning `/spikes` on.
        self.sampler.record_spikes = False
        # The words come from what the user TYPED, not from the rendered ids:
        # the echo partition is a statement about the prompt's subject matter,
        # and `render_turn` normalises punctuation and case that this does not
        # want to depend on.
        echo_words = prompt_content_words(prompt) if rp.echo else None
        # The subject tier needs to be told what the subject IS, and it is told
        # by the same extractor `snnchat.prime` builds a story prime from, so
        # the two cannot disagree about what a request was about. It returns
        # None for anything that is not a "story about X" request, and None is
        # the weighted tier -- so every other prompt is decided as before.
        subject = prime_topic(prompt) if rp.echo and rp.subject_tier else None
        winner, cands = _rerank(self.sampler.model, logits, self._state, params, rp,
                                device=self.sampler.device,
                                tok=self.tok, echo_words=echo_words, subject=subject)
        self.sampler.record_spikes = recording
        self.last_candidates = cands
        self.last_winner = winner
        self.last_echo_on = bool(rp.echo and echo_words)
        self.last_rerank = dataclasses.replace(rp)

        # `final_ids` and not an inline trim: the echo tier ranks candidates by
        # this exact string, and two copies of the rule that drifted apart would
        # mean the selector ranked one reply and the session fed another.
        out = final_ids(winner, self.tok, rp)
        if out:
            if recording:
                self.sampler.start_recording()
                for i in out:
                    _, self._state = self.sampler._feed([i], self._state)
            else:
                _, self._state = self.sampler._feed(out, self._state)
        if on_char is not None:
            on_char(self.tok.decode_visible(out))
        return out

    # -- what the neurons did ---------------------------------------------

    def show_spikes(self, on: bool = True) -> None:
        """Record per-layer firing rates for each generated character."""
        self.sampler.record_spikes = bool(on)
        if on:
            self.sampler.start_recording()
        self.last_spikes = []

    def spike_summary(self, width: int = 48) -> str:
        """Mean firing rate per layer, plus a sparkline over the reply.

        The sparkline is the layer-mean rate per generated character, which is
        the only view of the network's activity that fits on one line. It is a
        diagnostic and a curiosity, not a measurement: `aux["firing_rate"]` is a
        mean over channels, so a flat line means the *average* was steady and
        says nothing about which channels moved.
        """
        if not self.last_spikes:
            return "  (no spike record -- /spikes on, then say something)"
        blocks = " .:-=+*#%@"
        n_layers = len(self.last_spikes[0])
        lines = []
        for k in range(n_layers):
            series = [row[k] for row in self.last_spikes]
            mean = sum(series) / len(series)
            # One column per character would overflow the terminal on a long
            # reply, so the series is bucketed to `width` columns.
            step = max(1, len(series) / width)
            spark = "".join(
                blocks[min(len(blocks) - 1, int(
                    (sum(series[int(i * step):max(int(i * step) + 1, int((i + 1) * step))])
                     / max(1, max(int((i + 1) * step), int(i * step) + 1) - int(i * step)))
                    * len(blocks)))]
                for i in range(min(width, len(series)))
            )
            lines.append(f"  layer {k}  mean {mean:5.1%}  |{spark}|")
        total = sum(sum(r) for r in self.last_spikes) / (len(self.last_spikes) * n_layers)
        lines.append(f"  {len(self.last_spikes)} characters, "
                     f"{total:.1%} of neurons fired per character on average")
        return "\n".join(lines)

    def reset(self) -> None:
        self.turns.clear()
        self._state = None
        self._primed = False
        self.chars_fed = 0
        self._fed = []
        self._marks = []
        self._regen = 0
        self._regen_key = None
        self.last_candidates = []
        self.last_winner = None
        self.last_echo_on = False
        self.last_rerank = None

    @torch.no_grad()
    def rewind(self, n_exchanges: int = 1) -> None:
        """Drop the last `n_exchanges` exchanges and replay the ids that remain.

        There is no way to subtract a turn from a membrane -- the state is a
        running summary, not a stack -- so undo is replay, and it is the one
        operation whose cost grows with the conversation.

        The replay reads `self._fed`, the exact ids the model consumed, not a
        re-render of `self.turns`: the reply text in `turns` has been stripped
        for display, so re-rendering it would feed a byte string the model never
        produced and land on a state no session was ever in.
        """
        n = min(n_exchanges, len(self._marks))
        if n == 0:
            return
        cut = self._marks[-n]
        kept_ids = self._fed[:cut]
        kept_marks = self._marks[:-n]
        kept_turns = self.turns[: max(0, len(self.turns) - 2 * n)]

        self.reset()
        if not kept_ids:
            return
        _, self._state = self.sampler._feed(kept_ids, None)
        self._primed = True
        self._fed = kept_ids
        self._marks = kept_marks
        self.turns = kept_turns
        self.chars_fed = len(kept_ids)

    @torch.no_grad()
    def regenerate(self, **overrides) -> str:
        """Re-answer the last user turn. Rewinds one exchange and resends it.

        UNDER A FIXED SEED THIS ADVANCES THE SEED, AND MUST
        --------------------------------------------------
        `send` seeds a fresh generator from `params.seed` at the top of every
        turn, so replaying the same user text under the same seed draws the
        identical reply, character for character. That made `/again` a silent
        no-op for anyone who had set `/seed` -- it reprinted the reply the user
        had just rejected, which is the one situation `/again` exists for.

        The offset is a COUNT of consecutive regenerations of this turn rather
        than a random re-seed, so the sequence stays reproducible: the same
        conversation with the same seed and the same number of `/again`s
        produces the same text. The counter is keyed on the turn, so answering
        a new turn starts it over.
        """
        if not self.turns:
            raise RuntimeError("nothing to regenerate")
        last_user = next(
            (text for role, text in reversed(self.turns) if role == "user"), None
        )
        if last_user is None:
            raise RuntimeError("no user turn to regenerate from")

        params = overrides.pop("params", None) or self.params
        bump = 0
        if params.seed is not None:
            key = (len(self._marks), last_user)
            bump = self._regen + 1 if key == self._regen_key else 1
            params = params.copy(seed=int(params.seed) + bump)

        self.rewind(1)
        # AFTER the rewind, not before: `rewind` replays from scratch and goes
        # through `reset`, which clears the counter along with everything else.
        # Setting it first is the version of this that silently answers every
        # `/again` with the second reply.
        if bump:
            self._regen = bump
            self._regen_key = (len(self._marks) + 1, last_user)
        return self.send(last_user, params=params, **overrides)

    def transcript(self) -> str:
        lines = []
        for role, text in self.turns:
            lines.append(f"{'you' if role == 'user' else 'snn'}: {text}")
        return "\n".join(lines)


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------


def load_chat_checkpoint(path, device: str | None = None, *, fused: bool | None = None):
    """Rebuild the model a checkpoint describes and load its weights.

    Handles both kinds of checkpoint this repository produces:

    * `kind == "snnchat"` -- written by `snnchat.train`, carries a `chat_config`.
    * anything else -- a research checkpoint from `snn.train`, carrying a
      `config` block. Those are trained on enwik8's 205-symbol vocabulary and
      have no role markers, so they can be *sampled* but not *chatted with*;
      `scripts/chat.py` detects this and runs in continuation mode instead of
      refusing, because watching the research baseline continue a prompt is a
      legitimate thing to want.
    """
    import dataclasses

    ck = torch.load(path, map_location="cpu", weights_only=False)
    if ck.get("kind") == "snnchat":
        from snnchat.model import ChatConfig, build_chat_model

        known = {f.name for f in dataclasses.fields(ChatConfig)}
        raw = {k: v for k, v in ck["chat_config"].items() if k in known}
        cfg = ChatConfig(**raw)
        if device:
            cfg.device = device
        if fused is not None:
            cfg.fused = fused
        # Turn the spread off before building: the weights are about to be
        # overwritten anyway, and running the initialiser is pure cost.
        cfg.spread_tau = False
        model = build_chat_model(cfg)
        model.load_state_dict(ck["model"])
        model.eval()
        return model, cfg, ck

    from snn.config import Config
    from snn.model import build_model

    raw_cfg = ck.get("config")
    if not raw_cfg:
        raise SystemExit(f"{path}: not a checkpoint this repository wrote")
    known = {f.name for f in dataclasses.fields(Config)}
    cfg = Config(**{k: v for k, v in raw_cfg.items() if k in known})
    if device:
        cfg.device = device
    if fused is not None:
        cfg.fused = fused
    model = build_model(cfg)
    model.load_state_dict(ck["model"])
    model.eval()
    return model, cfg, ck
