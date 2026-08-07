# chat-v2: what was trained, and what it does

Results for the chat model. **Not part of the research protocol** — no number
here is pre-registered or a reported figure, there are no control arms, and n=1
throughout. `docs/chat/BUILD_NOTES.md` records how the configuration was chosen
and which parts of it are guesses; `docs/chat/WEIGHT_DECAY_NOTE.md` records the
one observation from this work that bears on the research side.

## The run

| | |
| --- | --- |
| architecture | `snn.model.TwoCompThresholdCharLM` (EXP_004's neuron + EXP_008's threshold) |
| parameters | 4,417,637 |
| width / depth | d = 1024, K = 4 spiking layers |
| vocabulary | 101 (97 printable-ASCII characters + 4 role markers) |
| batch / window | B = 160, L = 256 |
| slow-pole init | log-uniform, tau 3 – 600 characters |
| weight decay | 0.1, **excluding** `beta_s_raw`, `w`, `thr_log` and biases |
| steps | 76,000 (full cosine, 2e-3 → 1e-4) + 8,000 anneal |
| characters seen | 3.11 G (≈ 2.2 passes over the corpus) |
| corpus | 1.40 G characters, six sources |
| hardware | one RTX 5060, 8 GiB, ~3.7 hours |

## Held-out bits per character

> **These numbers are not comparable with anything in `docs/reports/`.** The
> research figures are bits per character of **enwik8** over a **205-symbol**
> vocabulary; these are bits per character of a mixture of **children's stories,
> chit-chat and templated persona text** over a **101-symbol** vocabulary. The
> chat corpus is enormously more predictable than Wikipedia markup — TinyStories
> is written to a 1,500-word vocabulary on purpose — so a lower number here says
> nothing whatsoever about the architecture. Putting the two in one table would
> be the single easiest way to draw a false conclusion from this work.
>
> Within this file the numbers are comparable to each other, and that is all they
> are for.

Fresh-state bpc on each source's held-out tail, at the end of phase 1
(step 76,000):

| source | bpc | what it is |
| --- | --- | --- |
| `persona` | 0.2157 | templated; effectively memorised, and meant to be |
| `tinystories` | 1.0496 | simple narrative, 1,500-word vocabulary |
| `soda` | 1.1415 | natural two-party dialogue |
| `alpaca` | 1.6563 | instruction/response |
| `oasst1` | 1.8119 | real assistant register |
| `dolly` | 1.9260 | human-written instruction/response |
| **weighted** | **1.1521** | under the training mixture |

### The training curve, and the run it replaced

Both runs start from the identical initialisation and differ in one thing: the
first decayed the neuron's own parameters, the second did not. Scored on the four
sources both runs evaluated, renormalised — `chat-v2` added two more sources to
its evaluation, so its own weighted column is on a different scale.

| step | `chat-v1` (decayed) | `chat-v2` (exempt) |
| --- | --- | --- |
| 5,000 | 1.3508 | 1.3843 |
| 10,000 | 1.3556 | 1.3003 |
| 15,000 | **1.2821** (its best) | 1.2666 |
| 20,000 | 1.4007 ← regressed | 1.2461 |
| 40,000 | — | 1.1772 |
| 60,000 | — | 1.1154 |
| 76,000 | — | **1.0872** |

`chat-v1` was stopped at step 22,000. It is retained under `experiments/chat/` as
the evidence for the weight-decay note, not as a candidate.

**`chat-v2` was behind at its first evaluation** (+0.0336 at step 5,000). Acting
on that would have reverted a change that ended 0.15 bpc ahead — recorded because
it is the third time in this build that an early reading pointed the wrong way.

### What the slow poles did

The measurement that motivated the whole detour, on the two runs' own
checkpoints. Both initialised at tau 3–600 log-uniform, median 42.3:

| | tau median | tau max | % > 10 | % > 100 |
| --- | --- | --- | --- | --- |
| initialisation | 42.3 | 600 | 62.5 % | — |
| `chat-v1` @ 22,000 (decayed) | 2.46 | 63.0 | 1.9 % | 0.0 % |
| `chat-v2` @ 5,000 (exempt) | 23.69 | 778.6 | 68.9 % | 23.0 % |

With decay removed the maximum rises **above** its own initialisation. See
`docs/chat/WEIGHT_DECAY_NOTE.md`.

The shipped model is the anneal (`chat-v2-anneal/ckpt_best.pt`, step 7,874). It
beat phase 1 on the four-source scale (**1.0806** vs 1.0872) and on five of six
sources individually, costing only `tinystories` (1.0496 → 1.0672), and it
regressed neither greetings nor identity in the demo battery — which was the
acceptance criterion fixed before it was run.

## The slow poles it ended with

| quantile | tau (characters) |
| --- | --- |
| median | 5.8 |
| 75 % | 22.4 |
| 90 % | 56.8 |
| 99 % | 240.3 |
| max | **8,506** |

40.3 % of channels above tau = 10, 5.0 % above 100, 0.3 % above 600 — the last
being *above the longest time constant in the initialisation*. Compare
`chat-v1`, which decayed its neuron parameters: median 2.46, max 63.0, 1.9 %
above tau = 10.

## Memory horizon

`scripts/chat/horizon.py`, on SODA's held-out tail. The state is forced to zero
every `k` characters, so `k` is exactly how much context the model may use;
`k = L` is the identity and must reproduce an independently computed bpc, which
it does to **residual 0.00e+00** at both window sizes.

| k | 1 | 2 | 4 | 8 | 16 | 32 | 64 | 128 | 256 | 512 | 1024 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bpc | 3.416 | 3.017 | 2.493 | 2.017 | 1.653 | 1.404 | 1.253 | 1.165 | 1.118 | 1.093 | 1.081 |
| gain | | .399 | .524 | .476 | .365 | .248 | .152 | .088 | .047 | .024 | .013 |

**The gain halves with every doubling of context, and is still positive at 1,024
characters.** There is no cutoff — the model uses context as far out as the probe
looks, with smoothly diminishing returns. Note it was trained at L = 256, so
everything beyond that is extrapolation for a network with no positional encoding.

**Do not compare this to `EXP_001`'s 47–48 characters.** That is a different
corpus, a different vocabulary, a 6× smaller model, two fewer layers, a different
slow-pole initialisation *and* a different weight-decay treatment. At least five
things differ; nothing here isolates any of them. What the number means is "this
model, on this data, still benefits from a kilobyte of context" — and nothing more.

SODA is also the friendliest possible corpus for a long-horizon result: it is
dialogue with explicit turn markers, so distant context carries genuine
information about conversational state.

## Demo battery

Full output in `docs/chat/transcript.md` — same prompts, same seed, so two
checkpoints can be compared without the prompts moving.

```
you > what are you?
snn > I'm a spiking neural network trained on characters, one at a time.

you > tell me a story
snn > One day, a little dog named Max was walking in the middle of the dark dog.
      He wanted to play with the dog in the big tree...

you > what can you do?
snn > I'm quite small, so: conversation and stories. Nothing that needs real knowledge.
```

and the failures, which are the more informative half:

```
you > name three animals
snn > I don't really have a name. I'm a spiking neural network.

you > my name is Elliot
snn > Yeah, I wish this can be found you.
you > what is my name?
snn > I am Nothing to see anyone with my driver...

you > what is 17 plus 25?
snn > Item 3 degrees.
```

`name three animals` is the clearest illustration of what a character model is:
the prompt shares the string "name" with the identity intent, and that is enough.

## Honest assessment

**What it does.** Holds the turn format reliably and stops when it is done.
Greets, says what it is, and states its limits — correctly, because those answers
are stipulated in `snnchat.persona` rather than learned. Tells fluent short
stories in the TinyStories register. Follows "what happened next?" and keeps
going in the same register. Runs at ~420 characters/second on this GPU.

**What it does not do.** Topic conditioning is unreliable — "a story about a
rabbit" produced a story about a bird. It does not carry a fact across turns:
told a name, it cannot repeat it. Arithmetic, facts and verse all fail. A
question it has no template for often gets answered with a greeting.

**The honest summary:** it is a 4.4-million-parameter character model, and it
behaves like one. It is *fluent* without being *responsive* — it reliably
produces well-formed English of the right shape for the turn, and only sometimes
produces English that is about what you asked. The gap between those two is the
whole remaining distance, and it is not a gap that another few hours of the same
training would close.

## Wall clock

| | |
| --- | --- |
| corpus download + pack | ~9 min |
| size bake-off (4 arms, 2 completed) | ~25 min |
| `chat-v1` (abandoned — weight decay) | ~65 min |
| `chat-v2` phase 1, 76,000 steps | 211 min |
| `chat-v2` anneal, 8,000 steps | 22 min |
| peak VRAM | 3.83 GiB |

---

# The responsiveness round (2026-08-06)

Everything above describes `chat-v2-anneal`, the model shipped on 2026-08-05.
The "Honest assessment" section ends by naming the remaining gap — *fluent
without being responsive* — and saying it "is not a gap that another few hours of
the same training would close". That turned out to be true in a more specific
way than intended: it was not a training-length problem at all.

**`docs/chat/QUALITY.md` is the write-up.** In one paragraph: the corpus's
topic-conditioned story requests were built from a regex that finds capitalised
mid-sentence words, which in TinyStories are character *names*, so the model was
trained on "a story about Lily" and measured on "a story about a rabbit" — a
form it had never seen. Topicality was 7.8 % for common nouns and 50 % for names,
and that split is the whole finding.

Two things were done about it: a story corpus asked for by **subject**
(`snnchat.topics`, `scripts/chat/build_topic_stories.py`), and a decoder that
draws several replies and keeps the one the prompt best explains
(`snnchat.rerank`). The numbers for both, the arms that separate them, and what
neither of them fixes are in `docs/chat/QUALITY.md`.

## The model now shipped

`experiments/chat/chat-v3d-aligned/ckpt_best.pt`, named by
`experiments/chat/SHIPPED`, which is what `python scripts/chat.py` loads.

| | |
| --- | --- |
| initialised from | `chat-v2-anneal/ckpt_best.pt` (the model above) |
| steps | 14,000 at B=160, L=256, lr 5e-4 cosine, 39.2 min |
| corpus | as before **plus** `stories_topic` at 0.40 of the mixture |
| objective | `bot_loss_weight = 3.0` (the reply's characters weigh more than the prompt's) |
| windows | `align_frac = 0.75`, `align_lookahead = 1024` — 74 % start at a conversation |
| decoder | best of 8 drafts at `lambda = 0.6`; `/rerank 1` restores the old sampler exactly |

What moved, and by how much:

| | shipped 2026-08-05 | shipped 2026-08-06 |
| --- | ---: | ---: |
| mentions your topic (over ~1,024 drafts) | 0.0518 | **0.0908** |
| bits a held-out reply owes to its own prompt | +0.0474 | **+0.0619** |
| bits a held-out story owes to its own request | +0.0012 | **+0.0027** |
| answers a substantive prompt with the greeting/identity line | 0.036 | **0.018** |
| strict instruction following ("name three animals") | 0.000 | 0.000 **(regressed from 0.050 in an intermediate arm)** |

Held-out bpc is **not** comparable across these two: the mixture changed, so the
number is measuring a different corpus. `docs/chat/QUALITY.md` §6 is the list of
what this did not fix, and it is longer than the list of what it did.

