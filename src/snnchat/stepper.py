"""One captured CUDA graph for the whole per-character decode step.

WHY THIS EXISTS
---------------
`docs/chat/BUILD_NOTES.md` §1 measured the per-timestep scan to be *latency*
bound at these widths, and everything in `snnchat.rerank` is built on the
consequence: a pool of N drafts costs about what one draft costs, so the
decoder can afford to propose widely. That reasoning is sound and the
measurement behind it is real -- but the constant it was measured with is not
the constant the REPL actually pays.

Measured on this box (RTX 5060, torch 2.12, `d_model=1024`, `K=4`), one
generated character:

    scan alone, any batch 1..512      ~1580 us      <- flat in N, as documented
    ... of which the arithmetic is        ~50 us    (one [B,1,1024] GEMM)
    sampling arithmetic (filter etc)   ~690 us
    a captured graph of the same step   ~205 us

So ~97 % of a decode step was dispatch, not work. The jiterator kernels the
fused scan is built from are launched one per layer per timestep, and at L = 1
there is nothing to amortise them against: the training scan issues them inside
a loop over hundreds of timesteps, and decoding issues four of them and then
returns to Python.

A CUDA graph is the standard fix and it fits this model unusually well, because
a decode step is *the same kernels with the same shapes every time* -- there is
no context window growing underneath it, which is exactly the structural
property `snnchat.generate`'s docstring opens with. The graph is captured once
per (batch size, sampling settings) and replayed per character.

WHAT IS AND IS NOT INSIDE THE GRAPH
-----------------------------------
Inside: the embedding, the K projections, the K scans, the head, the loop-break
penalty add, `_filter`'s truncation, and both softmaxes. All of it deterministic
given its inputs.

Outside: `torch.multinomial` and everything that reads a value on the host.
Deliberately. Capturing the draw would mean capturing RNG state, which changes
which random numbers a given seed produces -- and every committed artifact in
`experiments/chat/_quality/` was drawn with the generator being advanced the way
it is advanced today. A speedup is not worth invalidating the package's own
reproducibility, and `test_graph_and_eager_draw_the_same_text` is what holds the
line.

BIT-IDENTITY
------------
A replay issues the identical kernels with the identical arguments, so it
returns the identical bits, and `tests/test_snnchat.py` asserts that directly
rather than trusting the argument.

One deliberate difference, stated because "identical" should mean it: the eager
path applies the loop-break penalty by subtracting in place from the logit it
targets, and this path adds a `[n, V]` bias that is zero everywhere else.
`x - p` and `x + (-p)` agree exactly in IEEE754, and `x + 0.0` returns `x` for
every float except `-0.0`, which it returns as `+0.0`. `exp(-0.0) == exp(0.0)`,
so the two paths cannot disagree about a probability even in that case.

Not part of the research protocol. This module writes nothing under `snn/` and
imports only to call a model that is already built.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from snnchat.generate import SamplingParams, _filter

__all__ = ["DecodeStepper", "stepper_for", "graph_capture_available", "clear_stepper_cache"]


def graph_capture_available(device) -> bool:
    """True iff a graph can be captured on `device`.

    Checked by property rather than by try/except at the call site: a failed
    capture leaves the allocator in a state that is awkward to reason about, and
    the answer is the same for every turn of a session.
    """
    return torch.device(device).type == "cuda" and torch.cuda.is_available()


class DecodeStepper:
    """A captured single-character step for a fixed batch size and sampler.

    The state lives in the graph's own static buffers. `prime` writes a state in,
    `step` advances it by one character, `read_state` copies it back out. Nothing
    else may touch those buffers, which is why they are private and why
    `permute` exists rather than callers indexing them.
    """

    def __init__(
        self,
        model,
        n: int,
        params: SamplingParams,
        *,
        temps: torch.Tensor | None = None,
        device=None,
    ) -> None:
        self.model = model
        self.n = int(n)
        self.device = torch.device(device) if device is not None else next(model.parameters()).device
        self.vocab = int(model.head.out_features)
        #: The settings baked into the capture. `stepper_for` compares against
        #: these before handing a cached stepper back: a graph captured at
        #: top_p = 0.92 silently ignores a caller who has since said 0.5.
        self.key = _sampler_key(params, temps)

        d2 = _state_width(model)
        self._feed = torch.zeros(self.n, 1, dtype=torch.int64, device=self.device)
        self._penalty = torch.zeros(self.n, self.vocab, dtype=torch.float32, device=self.device)
        self._state = [
            torch.zeros(self.n, d2, dtype=torch.float32, device=self.device)
            for _ in range(model.n_layers)
        ]
        #: Rows whose penalty row is nonzero, so the buffer is cleared by row
        #: rather than in full. A cycle is rare and clearing all `n * V` of it
        #: every character would cost more than the penalty saves.
        self._dirty: list[int] = []
        self._graph: torch.cuda.CUDAGraph | None = None
        self._probs: torch.Tensor | None = None
        self._logprobs: torch.Tensor | None = None
        self._capture(params, temps)

    # -- construction ------------------------------------------------------

    def _capture(self, params: SamplingParams, temps: torch.Tensor | None) -> None:
        model = self.model
        # Warm up on a side stream first. This is not optional: capture records
        # allocations, and the first call through a jiterator kernel COMPILES it,
        # which cannot happen inside a capture.
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side), torch.no_grad():
            for _ in range(3):
                _run_step(model, self._feed, self._state, self._penalty, params, temps)
        torch.cuda.current_stream().wait_stream(side)

        graph = torch.cuda.CUDAGraph()
        with torch.no_grad(), torch.cuda.graph(graph):
            probs, logprobs, new_state = _run_step(
                model, self._feed, self._state, self._penalty, params, temps
            )
            # The state update is INSIDE the capture, so a replay is
            # self-advancing and one character costs exactly one replay. Copying
            # afterwards would put K more launches back into the loop this
            # module exists to empty.
            for k, s in enumerate(new_state):
                self._state[k].copy_(s)
        self._graph = graph
        self._probs = probs
        self._logprobs = logprobs

    # -- the state ---------------------------------------------------------

    @torch.no_grad()
    def prime(self, state) -> None:
        """Load a membrane state into the graph's buffers. Broadcasts B=1 to n."""
        if state is None:
            for s in self._state:
                s.zero_()
            return
        for k, s in enumerate(state):
            self._state[k].copy_(s)          # copy_ broadcasts a [1, 2d] row

    @torch.no_grad()
    def read_state(self, rows=None) -> list[torch.Tensor]:
        """A private copy of the current state, optionally of selected rows."""
        if rows is None:
            return [s.clone() for s in self._state]
        idx = torch.as_tensor(rows, dtype=torch.int64, device=self.device)
        return [s.index_select(0, idx) for s in self._state]

    @torch.no_grad()
    def permute(self, idx: torch.Tensor) -> None:
        """Reorder the state rows. `index_select` first, so nothing aliases."""
        for s in self._state:
            s.copy_(s.index_select(0, idx))

    # -- one character -----------------------------------------------------

    @torch.no_grad()
    def penalise(self, pairs, amount: float) -> None:
        """Subtract `amount` from logit `t` of row `r` for each `(r, t)` in `pairs`.

        Cleared and refilled per character. Both loops are no-ops in
        non-degenerate text, so the common case costs two empty Python loops and
        no launch at all.
        """
        if not pairs and not self._dirty:
            return
        for r in self._dirty:
            self._penalty[r].zero_()
        self._dirty = [r for r, _ in pairs]
        for r, tok_id in pairs:
            self._penalty[r, tok_id] = -amount

    @torch.no_grad()
    def step(self, feed: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Feed one character per row; return `(probs, raw logprobs)`, both [n, V].

        The returned tensors are the graph's own outputs and are **overwritten by
        the next call**. Callers read them before stepping again, which the
        sampling loop does naturally; anything that needs to keep one clones it.
        """
        self._feed.copy_(feed.reshape(self.n, 1))
        self._graph.replay()
        return self._probs, self._logprobs


def _state_width(model) -> int:
    """The per-layer state width: `2d` for the two-compartment arms, `d` otherwise."""
    probe = model.init_state(1, next(model.parameters()).device)
    return int(probe[0].shape[-1])


def _run_step(model, feed, state, penalty, params, temps):
    """The body that is captured. Pure function of its tensor inputs."""
    logits, new_state, _ = model(feed, state=state)
    last = logits[:, -1, :].float()
    probs = F.softmax(_filter(last + penalty, params, temps), dim=-1)
    # The RAW log-probabilities, before temperature and truncation: the score in
    # `snnchat.rerank` compares replies under the model, not under the sampler's
    # truncated view of it. Computed here so the loop does not launch it.
    logprobs = F.log_softmax(last, dim=-1)
    return probs, logprobs, new_state


def _sampler_key(params: SamplingParams, temps: torch.Tensor | None):
    """Everything baked into a capture. Not the seed: the RNG stays outside."""
    return (
        round(float(params.temperature), 12),
        int(params.top_k),
        round(float(params.top_p), 12),
        round(float(params.min_p), 12),
        None if temps is None else tuple(round(float(t), 12) for t in temps.flatten().tolist()),
    )


#: Captures are cached because one costs a warmup plus an NVRTC-warm capture
#: (~40 ms here) and a REPL turn is otherwise ~0.2 s. Two entries: a session
#: alternates between its pool size and, when `/spikes` is on, a single row.
_CACHE: list[DecodeStepper] = []
_CACHE_MAX = 2


def stepper_for(model, n, params, *, temps=None, device=None) -> DecodeStepper | None:
    """A stepper for this batch size and these settings, or None if unavailable.

    None is a normal answer, not an error: on CPU, or on a build where capture
    fails, the caller runs its eager loop, which is the same arithmetic. That is
    what keeps this module optional rather than load-bearing.
    """
    device = device if device is not None else next(model.parameters()).device
    if not graph_capture_available(device):
        return None
    key = _sampler_key(params, temps)
    for i, st in enumerate(_CACHE):
        if st.model is model and st.n == n and st.key == key:
            _CACHE.append(_CACHE.pop(i))       # most recently used last
            return st
    try:
        st = DecodeStepper(model, n, params, temps=temps, device=device)
    except Exception:                          # noqa: BLE001 - capture is best-effort
        # Anything from an unsupported op to an allocator refusal. The eager path
        # is correct and only slower, so a failure here must not end the turn.
        return None
    _CACHE.append(st)
    while len(_CACHE) > _CACHE_MAX:
        _CACHE.pop(0)
    return st


def clear_stepper_cache() -> None:
    """Drop cached graphs. For tests, and for anything that rebuilds a model."""
    _CACHE.clear()
