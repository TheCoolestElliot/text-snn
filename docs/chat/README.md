# Talking to the spiking network

This directory documents `snnchat`, a conversational front end for the spiking
char-LM. **It is not part of the Phase-1..5 research protocol.** Nothing in it is
pre-registered, no number it produces is a reported figure, and it must not be
cited as evidence for or against any research candidate. It exists so the model
can be talked to.

```
python scripts/chat.py
```

---

## 1. What this is

A conversation with a **4.4-million-parameter spiking neural network** that reads
and writes one character at a time. Its neurons integrate charge and fire binary
spikes; between layers, nothing but spikes is transmitted. It is the same
two-compartment neuron the research side of this repository adopted in `EXP_004`
and composed with a learned threshold in `EXP_011` — `snnchat` changes the size,
the corpus and one initialisation, and changes nothing about the architecture.

`tests/test_snnchat.py::test_chat_model_is_a_research_arm_unchanged` is what
holds that claim up: it loads a chat model's `state_dict` into a model built by
`snn.model.build_model` and asserts the logits are bit-identical.

### It has no context window

This is a recurrent network, so the conversation lives in `[B, 2d]` of membrane
state per layer rather than in a re-read prefix. A character costs the same
whether it is the tenth of the session or the ten-thousandth, and an hour-long
conversation uses exactly as much memory as a one-line one. `/state` shows how
many characters have been fed through the membrane.

The flip side is that **nothing is stored losslessly**. There is no window that
older text falls out of — it decays. What the state can still resolve is bounded
by the slow pole's time constant, and no sampling setting changes that.

Measured on the trained model: context still lowers bits-per-character out to
**1,024 characters**, with the gain halving at every doubling and no cutoff.
That is a statement about *how far back statistical context helps*, not about
recall — the same model, told your name, cannot repeat it one turn later.

Re-measured on the current checkpoint (`docs/chat/horizon_soda_1024_v3d.json`),
where the shape is unchanged — context still pays at 1,024 characters, the gain
still halves at every doubling, and the probe's `k = L` self-check reproduces an
independently computed bpc to a residual of exactly zero:

| k | 1 | 8 | 64 | 256 | 1024 |
| --- | --- | --- | --- | --- | --- |
| bpc | 3.523 | 2.160 | 1.346 | 1.186 | 1.144 |

The absolute numbers are worse than the 2026-08-05 model's on this source and
that is not a regression in the horizon: the mixture moved 20 points of weight
away from SODA, so the model is being scored on a corpus it now sees less of.
Compare the *shape* across the row, not the level against the older table.

---

## 2. What to expect

It is a small model trained for a few hours on one consumer GPU. Concretely:

| It does this reasonably | It does this badly |
| --- | --- |
| Holds the turn format: it answers and then stops | Anything requiring facts |
| Fluent, grammatical short English sentences | Arithmetic, dates, places |
| Greetings, identity and small talk | Following a multi-step instruction |
| Short simple stories in a consistent register | Naming three of something |
| Continuing ("what happened next?") | Carrying a fact across turns |
| | Being *about* what you asked — **usually** |

Measured, not predicted — see `docs/chat/QUALITY.md` for the numbers and
`docs/chat/transcript.md` for the full battery. The last row is the honest
headline and it is only half-fixed. Asked for a story about a named character it
gets it right nearly every time; asked for a story about a *dragon*, a *robot* or
a *bicycle*, it still writes about something else.

That failure had a specific cause, and finding it is most of what
`docs/chat/QUALITY.md` is about: the corpus built its topic-conditioned requests
from a regex that finds capitalised mid-sentence words — which in TinyStories are
**character names**. The model was trained on "a story about Lily" and measured
on "a story about a rabbit", a form it had never once seen. Fixing the corpus and
the window placement raised the rate at which it mentions your topic by **75 %
relative** (0.052 → 0.091 over ~1,000 drawn replies). That is a real improvement
and it is still a small number: nine of sixteen test topics land at zero.

Told "my name is Elliot" and then asked for the name, it still cannot. Nothing
here changed that and nothing at this scale will.

**It will confidently say false things.** It has no knowledge base, no retrieval,
and no capacity to check anything. The one topic on which it is reliably accurate
is itself, and only because those answers are stipulated in
`src/snnchat/persona.py` rather than learned.

The best things to type at it are the things it was built for: `hello`,
`what are you?`, `tell me a story about a rabbit`.

---

## 3. Running it

```bash
# the checkpoint experiments/chat/SHIPPED names, else the newest one
python scripts/chat.py

# a specific one
python scripts/chat.py --ckpt experiments/chat/chat-v1/ckpt_best.pt

# one-shot, no REPL
python scripts/chat.py --prompt "tell me a story about a fox"

# a RESEARCH checkpoint -- continuation mode, see below
python scripts/chat.py --ckpt experiments/runs/compose_s1/ckpt_best.pt

# what is available, with the default marked
python scripts/chat.py --list
```

`experiments/chat/SHIPPED` is a one-line file naming the default checkpoint.
Without it the default is whichever `ckpt_best.pt` is newest, which stops being
the right rule as soon as a run is trained that is *not* meant to be talked to —
the responsiveness round ended with a deliberately-worse control arm, and by
modification time that would have become the model everyone loaded.

### Commands

`/help` `/new` `/again` `/back [n]` `/temp <x>` `/topp <x>` `/topk <n>`
`/minp <x>` `/len <n>` `/rerank <n>` `/lambda <x>` `/spread <x>` `/candidates`
`/seed <n|off>` `/params` `/spikes` `/state` `/transcript` `/save <file>`
`/quit`

`/new` clears the membrane state — the only true reset. `/back` cannot subtract a
turn from a membrane, so it replays the retained transcript; it is the one
command whose cost grows with the conversation.

`/spikes` turns on a per-reply readout of what the network actually did:

```
you > tell me a story about a fox
snn > Once upon a time there was a little fox named Fern...
  layer 0  mean 28.6%  |=+=*+==+*=+=*==+=*+==*=+=*+=+*=+==*+=*+=+*=+=*=|
  layer 1  mean 24.1%  |=+==*=+==*+=+=*=+==*=+=*=+==*+=*=+==*=+=*+==*=+=|
  layer 2  mean 31.2%  |+=*+=*+=*+==*+=*+=*+=*+==*+=*+=*+==*+=*+=*+=*+==|
  212 characters, 27.9% of neurons fired per character on average
```

Each column is one character of the reply and the bar is that layer's mean firing
rate. It is a curiosity rather than a measurement — `aux["firing_rate"]` is a
mean over channels, so a flat line says the average was steady and says nothing
about *which* neurons moved. Recording is off by default because it keeps one
tensor per layer per character alive.

### Best of N: how a reply is chosen

By default the model drafts several replies to each turn and keeps the one your
prompt best explains, scoring candidates by

```
[ log P(reply | your prompt) - lambda * log P(reply | an empty prompt) ] / length
```

The second term is the same model asked the same way with your words removed, so
it measures "would it have said this anyway" — which is precisely the failure
this model has. A generic story is likely whatever you asked for and loses; a
reply that only makes sense as an answer to *your* question keeps its score.

This is cheap here and nowhere else. The per-timestep scan is latency-bound at
this width (`docs/chat/BUILD_NOTES.md` §1), so extra drafts ride along in the
same batch: **sixteen drafts measured at 2.1× the wall clock of one**, a marginal
cost of about 0.07× per draft. Cheap, not free — a transformer would pay sixteen
times over.

`lambda` is a real trade and the default sits where it was measured to be worth
making. At 0 you get best-of-N by likelihood, which on a model trained 40 % on
stories answers "name three animals" with a story 17 % of the time. Above ~1.0
the model starts preferring text it thinks is *unlikely*, which at this scale
means word salad. 0.6 is the setting that is more on-topic **and** more fluent
than not reranking at all.

```
/rerank 1       one draft, chosen by sampling alone (the pre-2026-08-06 behaviour)
/rerank 8       eight drafts
/lambda 0       best of N by likelihood alone — a fluency filter, nothing more
/spread 0.3     draw the drafts across a range of temperatures, not all at one
/candidates     what the last reply was chosen from, with scores
```

It selects among replies the model would have produced anyway. If all eight
drafts are off topic it returns the least-bad of eight off-topic replies, and it
buys no knowledge, no arithmetic and no memory across turns. That is the reason
`/spread` exists: selection cannot invent a reply that was never proposed, so
once reranking is on, the only remaining lever is the pool it selects from.
`/spread 0.3` at temperature 0.85 proposes the drafts at 0.60 through 1.11 —
the cautious end and the adventurous end fail differently, and the score decides.
Mixing temperatures in one pool is sound here because a candidate's
log-probability is accumulated under the model's *own* distribution, before
temperature and nucleus truncation touch it, so a draft is never scored under
the sampler that proposed it.

### Sampling

Defaults are `temperature 0.85`, `top_p 0.92`, `min_p 0.02`. If replies ramble,
lower the temperature to ~0.6. If they are dull and repetitive, raise it to ~1.0
and set `/minp 0`. There is a loop breaker on by default that suppresses only the
single character which would extend a just-repeated block — it is silent on
ordinary text, which `tests/test_snnchat.py::test_loop_penalty_is_silent_on_ordinary_text`
checks.

### Continuation mode

Pointing `scripts/chat.py` at a checkpoint from `experiments/runs/` loads a
research model. Those are trained on enwik8 and have never seen a conversation,
so there is nothing to chat with — the REPL detects this and switches to
continuation mode, where what you type is a prefix the model carries on from.
It reads the *research* corpus's 205-symbol vocabulary off disk, because that
checkpoint's embedding rows are indexed by it.

---

## 4. Building it from scratch

```bash
python scripts/chat/build_data.py         # ~1.7 GB download, ~7 min packing
python scripts/chat/size_probe.py         # optional: pick a config by measurement
python scripts/chat/train.py --run-name chat-v1 --budget-minutes 300
```

`train.py` is restartable: re-running the identical command resumes from
`ckpt_last.pt`. The sampler is a pure function of `(seed, step)`, so a resumed
run draws exactly the batches an uninterrupted one would have.

That is also how the **final chat-focused anneal** is done — no extra machinery.
Re-run the same command with the same `--run-name` and a different `--mix`, and
it picks up where it left off with the learning rate already near its floor:

```bash
python scripts/chat/train.py --run-name chat-v1 ... --budget-minutes 30 \
    --mix tinystories=0.15 soda=0.40 alpaca=0.20 dolly=0.08 oasst1=0.02 persona=0.15
```

The point of the anneal is to spend the end of the budget on the shape of the
task rather than on general English: less raw narrative, more turn-taking and
question-answering, at a learning rate low enough that it sharpens the format
without disturbing what the model already knows.

### The corpus

`scripts/chat/build_data.py` downloads five public sources and adds one written
here, renders everything into a single character-level transcript format, and
packs each source into its own `uint8` id stream under `data/chat/`.

| source | training characters | what it teaches |
| --- | --- | --- |
| `tinystories` | 747 M | simple, correct English — the grammar teacher |
| `soda` | 636 M | short natural two-party dialogue — turn-taking |
| `stories_topic` | 398 M | the reply should be *about the thing that was asked for* |
| `stories_short` | 243 M | the same, in conversations that fit inside one training window |
| `alpaca` | 8.2 M | answering a question rather than continuing it |
| `persona` | 5.3 M | greetings, identity, limits — written, not collected |
| `dolly` | 1.8 M | human-written instruction/response |
| `oasst1` | 0.3 M | real assistant register |

`stories_short` is the third wrapping of the same TinyStories text, and it exists
because of an arithmetic problem rather than a linguistic one. A reply can only
learn to depend on a request that is inside its own training window, and a
757-character conversation against a 256-character window delivers that
dependency for about a quarter of the reply's characters even with window
alignment turned on. Truncating each story at a sentence boundary near 161
characters makes the whole conversation fit, which takes the share of reply
characters trained against their own request from **0.26 to 0.88**
(`scripts/chat/window_coverage.py`). See `snnchat.shortform` for the three
judgements involved — chiefly that a truncated story has no ending, which is why
the two long story sources stay in the mixture.

`stories_topic` is the same TinyStories text as `tinystories`, wrapped in a
different request. The original wrapping asks for a story about a **capitalised
mid-sentence word**, which in TinyStories is the character's *name* — so the
corpus taught "a story about Lily" and never "a story about a rabbit", and the
model was then measured on the second. `snnchat.topics` picks the story's
**subject** instead. It is packed as a separate file, added by
`scripts/chat/build_topic_stories.py`, so `tinystories.bin` is untouched and the
balance between the two wrappings is a mixture weight rather than a fork. See
`docs/chat/QUALITY.md` §2 for how that was found.

**One file per source, mixed at sampling time.** The mixture weights are the most
uncertain thing in the build, and this way changing one is three numbers in a
config rather than a repack. It also means a 2 MB source can be 10 % of a 1.4 G
corpus without writing 70 copies of it to disk.

Every turn longer than 400 characters is **dropped, not truncated** — a truncated
turn teaches the model to stop mid-sentence. That filter is why `oasst1`
contributes only 796 conversations, and why its weight is 0.02 rather than the
0.06 the other instruction sources get: at 0.06 it would be 400 passes over 796
examples, which is memorisation rather than training.

### The format

Four ids are not characters: `<|bos|>`, `<|user|>`, `<|bot|>`, `<|eot|>`. The rest
of the vocabulary is a closed 97-character printable-ASCII alphabet, fixed in
`src/snnchat/tokenizer.py` and versioned — a corpus or checkpoint from another
`VOCAB_VERSION` is refused rather than decoded into garbage.

The markers are ids rather than literal strings such as `"User: "` for one
reason: **`encode` cannot produce them.** Whatever a user types, and whatever the
model emits, a turn boundary is an exact integer comparison and not a string
search. `tests/test_snnchat.py::test_encode_cannot_forge_a_turn_marker` is that
guarantee.

---

## 5. The one thing that is a bet

`snnchat.model.spread_slow_poles` initialises each layer's slow-pole decay
`beta_s` **log-uniformly across channels**, so time constants run from ~3 to ~600
characters instead of every channel starting at the committed `beta_slow = 0.95`
(`tau = 20`).

The motivation is measured; the fix is not. Phase 4 put the memory horizon of the
committed two-compartment neuron at 47–48 characters — about eight words — and
`EXP_006` found that ~3 % of channels carry that horizon and 94.9 % of the arm's
gain, and that clamping every channel to the median `beta_s` is *worse than the
Phase-2 baseline*. The tail is the mechanism. This gives the model that tail at
initialisation rather than asking it to grow one from a point where no spread
exists.

**It has not been tested against the committed initialisation at matched seeds.**
It changes no function class and adds no parameter, so it cannot make the model
unable to represent something; but whether it helps, hurts or does nothing is
unmeasured, and nothing here should be read as evidence that it works.
`scripts/chat/horizon.py` measures what the trained model actually ended up with,
which is a different question from whether the spread caused it.

If it is worth pursuing, it belongs on the research side as a pre-registered
candidate with a control arm — not as a claim inherited from a chat model.

It is no longer the *only* untested choice here. The responsiveness round of
2026-08-06 added several more — the mixture weight of the new story source, the
loss weight on the model's own turns, the fraction of aligned windows, and the
stop list that decides what a story is about. `docs/chat/BUILD_NOTES.md` §9 lists
which of them were measured and which were judged, and
`docs/chat/QUALITY.md` §6 lists what the round did **not** fix.

---

## 6. Layout

```
src/snnchat/
  tokenizer.py     the closed alphabet and the four role markers
  sources.py       downloads; one reader per dataset -> (role, text) turns
  persona.py       the authored greeting/identity exchanges
  build_corpus.py  renders and packs; one .bin per source
  data.py          weighted mixture sampler, pure in (seed, step); loss weights
  model.py         ChatConfig, and the slow-pole spread
  train.py         CUDA-graph captured, restartable training loop
  generate.py      sampling filters, ChatSampler, ChatSession
  topics.py        what a story is about, for asking for it by subject
  shortform.py     stories short enough that request and reply share a window
  rerank.py        draw N replies, keep the one the prompt explains
  quality.py       the metrics: prompt dependence, and the probe battery
scripts/chat.py            the REPL
scripts/chat/build_data.py download + pack
scripts/chat/build_topic_stories.py  pack the subject-conditioned story source
scripts/chat/build_short_stories.py  pack the short-conversation story source
scripts/chat/window_coverage.py      how much of a reply is trained with its request
scripts/chat/train.py      training
scripts/chat/size_probe.py matched-wall-clock config bake-off
scripts/chat/horizon.py    how far back context still helps
scripts/chat/demo.py       a fixed prompt battery, for comparing checkpoints
scripts/chat/quality.py    score a checkpoint, and sweep decoding settings
scripts/chat/compare_quality.py  several arms side by side, as markdown
scripts/chat/story_dodge_resample.py  story_dodge at a resolution that can miss
tests/test_snnchat.py      CPU-only; runs while the GPU is busy
docs/chat/README.md        this file
docs/chat/CONVENTIONS.md   how a rate is reported against a threshold -- READ FIRST
docs/chat/BUILD_NOTES.md   what was measured, what was guessed, what broke
docs/chat/RESULTS.md       the trained model's numbers
docs/chat/QUALITY.md       the responsiveness gap: measuring it, and closing it
docs/chat/transcript.md    the demo battery's output
```

**Before writing a `PREDICTION_*.md` or reading a number in a `QUALITY_*.md`
against a threshold, read [`CONVENTIONS.md`](CONVENTIONS.md).** Every rate here is
a proportion over a countable number of draws, and one of them — `story_dodge` —
spent a whole round's prediction on a threshold placed exactly on its lattice.
The rule is that a rate compared against a threshold is reported with its
numerator, its denominator and a 95 % interval, and that `unresolved` is one of
the three verdicts.

Nothing in `snn/`, `experiments/runs/`, `experiments/logs/` or `docs/reports/` is
read for anything but model construction, and none of it is written.
