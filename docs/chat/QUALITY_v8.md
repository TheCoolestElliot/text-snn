# QUALITY v8 — the selector, not the pool

**2026-08-11. A decoding round: no training, no new checkpoint, no arm.**
`experiments/chat/SHIPPED` still names `chat-v3d-aligned/ckpt_best.pt` and this
round does not propose changing it. What changed is how a reply is *chosen* from
the drafts that checkpoint already produces.

Not part of the Phase-1..5 research protocol. No number here is a reported
figure. Rates follow [`CONVENTIONS.md`](CONVENTIONS.md): numerator, denominator
and a 95 % Wilson interval, and `unresolved` is a real verdict.

**This document was written after its measurements, not before them.** There is
no `PREDICTION_v8.md`. That is a genuine departure from how rounds v4–v7 were
run and it is stated here rather than glossed: what follows is exploratory, and
the one confirmatory-shaped claim in it (§4) is the one measurement that was
designed, on held-out prompts, before its numbers were read.

---

## 1. The finding this round starts from

The candidate pool already contains on-topic replies. The score throws them
away.

Replaying `experiments/chat/_quality/chat-v3d-aligned.json` — the committed
draws, no new generation — at the shipped decoding row (`n = 8`, `λ = 0.6`):

| | `topic` | `list` | `fact` |
| --- | ---: | ---: | ---: |
| shipped selector | 0.1406 | 0.0500 | 0.0714 |
| **oracle over the same 8 drafts** | **0.2969** | **0.4833** | 0.2857 |

19 of the 64 `topic` draws already contain a draft that mentions what was asked
for. On `chat-v6-scratch` the same oracle is 0.4062.

This is a **factor of 2.1 sitting in a pool that has already been paid for**.
`docs/chat/BUILD_NOTES.md` §1 measured the scan to be latency-bound at this
width, so the drafts cost ~0.07× each; nothing about harvesting them costs a
forward pass.

> **Corrected 2026-08-11, before this document was committed.** An earlier draft
> of this table printed 0.3594 / 0.5333 / "23 of the 64" / 0.4844 against a row
> labelled "the same 8 drafts". Those are the **n = 16** oracle, read out of a
> file whose pool is 16 candidates wide, while the selector row beneath them is
> genuinely n = 8. The table compared an 8-draft selector against a 16-draft
> oracle and inflated the headroom by a third. Every figure above is now n = 8
> on both rows. The held-out measurement in §4 was always n = 8 throughout and
> is unaffected.

The reason the score does not find them: measured as a within-draw ranker for a
topic hit, the shipped score's AUC over the 8-candidate pool is 0.571 at
`λ = 0.6` — barely above chance. It ranks by log-probability, and being about
the right thing is not what the model assigns probability to.

## 2. What was changed

`snnchat.rerank` gains an **echo partition**, applied inside the existing
`min_chars` partition: candidates are grouped by how many *distinct* content
words of the user's prompt they use, the highest group wins, and the existing
score breaks ties inside it.

A partition rather than an additive term, for the reason `min_chars` is a
partition: the score is a mean log-probability per character and an echo count
is a count, and there is no principled exchange rate. A partition needs none.

**When no draft echoes any content word the partition is the identity**
(`tests/test_snnchat.py::test_echo_partition_is_the_identity_when_nothing_echoes`).

That is weaker than "every draw with no on-topic reply is decided as before",
which an earlier draft of this document claimed and which is **false**. Content
words include the request frame — `tell`, `story`, `write`, `want`, `make`,
`little` are not closed-class and are not in `_FUNCTION_WORDS` — so a draft can
enter a nonzero tier by saying "story" while being about nothing in particular.

Measured on the held-out run's own artifact: selection changed on 25 of 120
draws, and **16 of those 25 changed while neither selector produced a topic
hit**, the tier having been decided by `make` (4), `tell` (3), `story` (3) and
`want` (3). Those 16 are not gains; they are the partition moving the pick
sideways on frame vocabulary. They are also not losses — the net over all 120 is
+9/−0 (§4) — but the mechanism is not the clean one the first draft described,
and the honest summary is that the partition fires more often than it helps and
has so far never fired against the reader.

`/echo off` restores the previous selection. `/candidates` now ranks by the rule
that actually chose the winner, and shows the echo count.

## 3. The function-word list is untuned, and that was measured

The obvious objection is that the list was written by someone who had seen the
probe battery. Three lists were therefore replayed over the identical committed
draws:

| list | `topic` | `list` | `social` | `identity` |
| --- | ---: | ---: | ---: | ---: |
| shipped selector (no echo) | 0.1406 | 0.0500 | 0.9000 | 0.9375 |
| **A** — closed-class English only (111 words) | **0.2969** | **0.2333** | 0.9000 | 0.9375 |
| B — A + request verbs (131) | 0.2969 | 0.2333 | 0.9000 | 0.9375 |
| C — B + the battery's own task nouns (142) | 0.2969 | 0.1000 | 0.9000 | 0.9375 |

A and B are identical to four decimals on every kind, and the **tuned** list C
is *worse* (`list` 0.1000 against 0.2333). There was nothing to gain by tuning,
so the shipped list is A: determiners, pronouns, auxiliaries, modals,
prepositions, conjunctions, wh-words and a few degree adverbs. It contains no
verb of asking, no task noun and no numeral. A reader can check that claim
against `snnchat.rerank._FUNCTION_WORDS` without trusting this paragraph.

Replayed across four arms (`chat-v3d-aligned`, `chat-v6-scratch`,
`chat-v5a-short300`, `chat-v6-inst-a`) at n = 4, 8 and 16, **no probe kind on
any arm moved down.** `social` is unchanged to four decimals on all four.

`identity` is **not** inert, and an earlier draft of this section wrongly said it
was: it moves *up*, on `chat-v5a-short300` (0.9375 → 1.0000 at n = 8 and n = 16)
and on `chat-v6-scratch` (0.8750 → 0.9375 at n = 16). The n = 8 change is a
single draw — "are you a human?" at seed 3, where the partition swaps *"I'm an
arrival and much simpler."* for *"Not a human. I'm a spiking language model, and
a fairly small one."* One draw is not a result, and it is reported here because
the claim it replaces was false, not because it is evidence.

The mechanism originally given for the `social`/`identity` inertia — "a greeting
has no content words to echo" — is also false and is withdrawn. "are you a
human?" yields the content word `human`; "what can you do?" yields nothing only
because every word in it is closed-class. The correct statement is narrower:
these probes are stable *in practice* because their pools are near-saturated
already (`social` reads 0.9–1.0 before any partition), not because the partition
cannot fire on them.

## 4. Held-out topics: the measurement that is not definitional

§1 and §3 overstate the case and this section is why.

The `topic` probe asks whether the reply contains the requested noun. The echo
partition prefers replies containing the requested noun. On those probes the two
are not merely similar, they coincide: the partition scores **0.2969** and the
n = 8 oracle is **0.2969**. The selector attains the ceiling exactly, because
"pick a draft containing the noun" is what both the rule and the metric say.

**So the battery's `topic` column is not evidence for this change and is not
offered as any.** Any rule that reads the probe's own answer key would score the
same. §1's factor of 2.1 measures how much the *log-probability* score was
leaving behind — a fact about the old selector — and nothing about whether the
new one generalises.

`scripts/chat/echo_holdout.py` was written to answer the question that is not:
20 story requests whose nouns appear nowhere in `quality.PROBES` (penguin,
lighthouse, wizard, turtle, violin, castle, squirrel, balloon, whale, mountain,
bakery, mermaid, clock, bear, garden, firefighter, butterfly, snail, rainbow,
elephant), 6 seeds each, **both selectors choosing from one shared pool** so the
comparison is paired and the draw is not a source of variance.

| | rate | 95 % CI |
| --- | ---: | --- |
| shipped selector | 0.0083 (1/120) | [0.0015, 0.0457] |
| **+ echo partition** | **0.0833 (10/120)** | [0.0459, 0.1466] |
| oracle over the same pool | 0.0917 (11/120) | — |

Discordant pairs: **9 echo-only, 0 shipped-only. Exact McNemar p = 0.0039.**
Selection changed on 25 of 120 draws and never changed for the worse. The
partition recovers **10 of the 11** hits the pool contained.

**Verdict: `above`, resolved.** It is also, as far as this package's record goes,
the second cleanly resolved comparison the chat side has ever produced — the
first being `story_dodge` at 301 seeds — and it cost no GPU-hours of training.

### The same table is the strongest evidence yet for the corpus diagnosis

The oracle on held-out topics is **0.0917**. On battery topics it is 0.2969.
The model essentially never *drafts* a reply about a penguin, so there is almost
nothing for any selector to find. The selector is now near its ceiling and the
ceiling is the pool.

Read together: **selection was the binding constraint and is no longer; coverage
is.** That is a stronger statement of `QUALITY.md` §6's diagnosis than anything
in v4–v7, and it was obtained without training an arm.

## 5. Costs, and the two things this does not fix

* **Fluency.** Mean per-character `logP` moves −0.2785 → −0.2998 on the held-out
  set and −0.3317 → −0.3531 on the battery. Both sit above the −0.3934 floor
  `scripts/chat.py` states as the constraint on this trade ("maximise topicality
  subject to not making the model less fluent than it is without reranking").
* **Length is not the mechanism.** On held-out topics mean reply length is
  277.3 → 276.7 characters. On the battery's `topic` probes every draft is at the
  300-character cap, so length is controlled by construction — and a plain
  longest-candidate baseline scores 0.0781 there, *below* the shipped selector's
  0.1406.
* **`list` is a length effect and is reported as one.** The battery `list` gain
  (0.0500 → 0.2333) is **not** claimed: a longest-candidate baseline beats the
  echo partition on that column (0.3167 against 0.2333). `list` prompts ask for
  animal *names* while the echoed word is "animals", so the partition is doing
  something indirect there and length explains most of it.
* **It buys no knowledge, no arithmetic and no memory across turns.** `fact`'s
  0.0714 → 0.1786 is a real, non-definitional gain (the echoed word is "france",
  the scored word is "paris") and it is still a model that does not know the
  answer.

## 6. Two defects fixed alongside

* **`/again` was a silent no-op under `/seed`.** `send` seeds a fresh generator
  from `params.seed` every turn, so replaying the same user text at the same seed
  reproduced the reply the user had just rejected — the one situation `/again`
  exists for. `regenerate` now offsets the seed by a *count* of consecutive
  regenerations, so the fix does not cost reproducibility.
* **`build_corpus.DEFAULT_MIX` was still the `chat-v1` mixture**, which predates
  `stories_topic` entirely. Any run launched without an explicit `--mix` silently
  trained on a corpus with no subject-conditioned story source in it — the single
  change `QUALITY.md` attributes the responsiveness gain to. It now holds the
  shipped checkpoint's own `chat_config["mix"]`.

## 7. What this round does not claim

* **It does not displace `SHIPPED`.** No checkpoint changed. A decoding default
  changed.
* **It does not fix the measurement problem.** The headline is still 89 %
  `list`-variance at 20 draws, `fallback_rate` still reads 0.0000 while 7 of the
  20 `list` picks at the reading row are verbatim `snnchat.persona` strings, and
  no comparison in this document touches the headline. That defect is real, is
  independent of this round, and is untouched by it.
* **It does not test whether the partition helps a human reader.** It tests
  whether the reply mentions what was asked about. On a 4.4 M-parameter
  character-level model those are not the same thing — §4's own winning replies
  include *"Once upon a time, there was a big oak tree. The boat was very happy."*
  On topic, and not good.

## 8. `lambda` after the partition: the knob still works, its reason does not

`scripts/chat.py` fixed `λ = 0.6` on 2026-08-06 under a stated rule — *maximise
topicality subject to not making the model less fluent than it is without
reranking*. At that time the anti-LM score was the only thing pushing a reply
towards the prompt, so one constant served two purposes. §2 took the first
purpose away: inside a tier every draft echoes the prompt equally, so `λ` now
decides only which of several equally-echoing drafts is kept.

`scripts/chat/lambda_sweep.py` replays the committed draws across a `λ` grid,
partition on and off (no generation, four arms pooled, n = 8):

| λ | topic (echo off) | logp (echo off) | topic (echo **on**) | logp (echo **on**) |
| ---: | ---: | ---: | ---: | ---: |
| 0.00 | 0.0781 | −0.2372 | 0.2344 | −0.2869 |
| 0.60 | 0.1172 | −0.3126 | 0.2539 | −0.3291 |
| 1.00 | 0.1211 | −0.4360 | 0.2539 | −0.4146 |
| 1.50 | 0.1250 | −0.4792 | 0.2500 | −0.4441 |

With the partition **off** the old trade is plainly visible: topicality climbs
by 0.047 across the range and fluency falls by 0.242. With it **on**, topicality
is flat to within a few draws while fluency falls just as far.

**On held-out topics the flatness is exact.** Replaying the §4 pool (the run now
stores every candidate, so any selection rule can be tried against one
generation) gives **10/120 at every λ from 0.00 to 1.50** — not one draw of 120
changes — while mean per-character `logP` runs −0.2951 → −0.4325.

**Verdict: `λ = 0.6` is unchanged, and `unresolved` is why.** Paired McNemar on
the battery against λ = 0.6, partition on: lowering to 0.0/0.2/0.4 loses 5 topic
draws and gains 0 (p = 0.0625); raising to 0.8 is 6 better and 3 worse
(p = 0.508); 1.0 is 7 and 7. Nothing displaces the incumbent, which is the same
rule that has kept `SHIPPED` in place for four rounds.

Two things are now known that were not:

* **Above 0.6 is pure loss.** Held-out topicality does not move at all and
  fluency falls monotonically. The README's old guidance — "above ~1.0 the model
  starts preferring text it thinks is unlikely" — understates it: with the
  partition on there is no longer any range in which raising `λ` buys anything.
* **`λ = 0.6` now stands only as an incumbent.** The measurement that justified
  it was a trade the partition has since removed. If a future round wants to
  lower it for fluency, the battery's 5-versus-0 topic loss at p = 0.0625 is the
  thing to resolve, and 120 held-out draws are not enough to do it.

## 9. `canned_rate`: the ship gate's dominant column is 38 % recitation

`fallback_rate` exists to count the model dodging a substantive prompt with its
stock line. It is five hand-picked phrases and it does not work.

`snnchat.quality._is_canned` compares a reply against all ~104 non-story replies
`snnchat.persona` stipulates, by prefix match in either direction over ≥ 20
normalised characters. `scripts/chat/canned_rate.py` applies it to every
committed arm's `baseline_picks` and **rewrites none of the 21 files** — those
are the record five rounds of documents quote.

Pooled over 17 arms, at the pinned `n = 1, λ = 0` reading row:

| | k/n | rate | 95 % CI |
| --- | ---: | ---: | --- |
| `list` picks that are persona recitation | 130/340 | **0.3824** | [0.3323, 0.4350] |
| substantive picks `_is_fallback` fires on | 28/1904 | 0.0147 | — |

Per arm the `list` rate runs 0.30 (`chat-v6-scratch`, `chat-v3d-aligned-s3`,
`chat-v6-inst-a-s2`) to 0.50 (`chat-v3d-aligned-s1`, `chat-v6-inst-a-s1`,
`chat-v6-inst-b`). The shipped arm is 8/20. **Almost every canned pick is in the
`list` column** — 8 of the 8 substantive ones on the shipped arm — which is the
column carrying ~89 % of the headline's between-training-seed variance and
therefore the column that has decided every ship gate since 2026-08-06.

### The detector was calibrated, and its first version was wrong

At a 15-character threshold it fired on **53 of 64 `topic` picks**, because
persona's `story_request` topic contains written-out narratives and 15
characters of those is "once upon a time" — the opening of nearly every story
the model tells. Two independent corrections each remove it:

| exclude story replies | min chars | topic | list | fact |
| --- | ---: | ---: | ---: | ---: |
| no | 15 | **53/64** | 9/20 | 1/28 |
| no | 20 | 0/64 | 8/20 | 0/28 |
| yes | 20 | 0/64 | 8/20 | 0/28 |
| yes | 30 | 0/64 | 7/20 | 0/28 |

Both are applied: story replies are excluded (that failure belongs to
`story_dodge`, and counting it here would double-count one failure while hiding
another) and the threshold is 20. The result is stable at 24 and 30, so the
surviving matches are not a threshold artefact.

**`canned_rate` is emitted beside `fallback_rate`, never folded into it, and is
not in the headline.** A term added now would silently rewrite five rounds of
ship decisions. What to do about a 38 % contamination is a decision for a
pre-registration; this round's job was to make it visible.

## 10. Cross-turn recall: 0/240, with a control

*"Told your name, it cannot repeat it a turn later"* appears in
`docs/chat/README.md` §2, in `RESULTS.md`, and in five QUALITY documents. **No
committed artifact contained that measurement**, and `snnchat.quality`'s battery
has no multi-turn probe at all — its `fact` kind is single-turn world knowledge,
a different capability.

`scripts/chat/memory_probe.py` builds one. 10 items (a name, a pet, a colour, a
town, an age, a food, a bicycle), 8 sampler seeds, 3 distances, each run twice
at the same seed: **told** (establishing turn, then filler turns, then the
question) and **untold** (the identical question with the establishing turn
removed). The control is the design — asked "what is my name?" the model might
say "Elliot" because it was told or because it is a likely name, and one
condition cannot tell those apart.

| distance | told | untold (control) | McNemar p |
| ---: | ---: | ---: | ---: |
| 0 | 0/80 | 0/80 | 1.000 |
| 1 | 0/80 | 0/80 | 1.000 |
| 2 | 0/80 | 0/80 | 1.000 |

**0/240 in both conditions. 95 % CI [0, 0.0158].** Cross-turn recall on this
battery is below 1.6 %.

The probe is not silently broken: replies average 51.5 characters and none is
empty. The failure has a shape worth recording — asked *"what is my name?"* the
model answers about **its own** name, from persona: *"I don't really have a
name. I'm a spiking neural network."* The possessive is not resolved, so the
question is routed to the identity intent rather than to memory. A corpus that
taught the distinction is a cheaper hypothesis than more reach.

**This is a floor for later work, and that is the point of running it now.**
Every proposal to lengthen the training window or widen the model is argued
partly on reach; none could be evaluated before this existed. Anything above
zero is now a measurable improvement.

## 11. Reproducing

```bash
python scripts/chat/echo_holdout.py --seeds 6     # §4, ~2 GPU-minutes
python scripts/chat/lambda_sweep.py               # §8, no GPU
python scripts/chat/canned_rate.py                # §9, no GPU
python scripts/chat/memory_probe.py --seeds 8     # §10, ~9 GPU-minutes
python -m pytest tests/test_snnchat.py -q
```

Artifacts, all under `experiments/chat/_quality/`:

| file | what is in it |
| --- | --- |
| `echo_holdout.json` | §4 — per draw: both selectors' chosen text and, since this round, **every candidate** with its two log-probabilities, so a later selection rule can be replayed against this generation instead of drawing its own |
| `lambda_sweep_v8.json` | §8 — the grid, partition on and off, four arms |
| `canned_rate.json` | §9 — per arm, with the pooled interval. Written as a new file; the 21 committed `_quality/*.json` are untouched |
| `memory_probe.json` | §10 — per item, per seed, per distance, both conditions, with the replies |

## 12. What this round changed in the code

| file | change |
| --- | --- |
| `src/snnchat/rerank.py` | the echo partition, `prompt_content_words`, `echo_count`, `RerankParams.echo` |
| `src/snnchat/generate.py` | threads the prompt into reranking; records the winner; fixes `/again` under a fixed seed |
| `src/snnchat/quality.py` | `_is_canned`, `canned_rate`, `canned_rate_list` — **`headline` is unchanged** |
| `src/snnchat/data.py` | warns when a mixture names a source the corpus does not contain |
| `src/snnchat/build_corpus.py` | `DEFAULT_MIX` is the shipped recipe, not `chat-v1`'s |
| `scripts/chat.py` | `/echo`, `--no-echo`, and a `/candidates` that stars the reply that was actually chosen |
