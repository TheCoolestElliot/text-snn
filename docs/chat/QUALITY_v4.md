# Making the reply fit the window: the 2026-08-06 short-conversation round

`docs/chat/QUALITY.md` closed with a list of what would finish the job. This file
is items 1, 2 and 3 of that list, attempted. `docs/chat/PREDICTION_v4.md` was
written before either arm trained and is what the results below are scored
against.

**Not part of the research protocol.** Nothing here is pre-registered in the
protocol's sense, there are no control arms in the protocol's sense, n = 1 per
arm, and no number here may be cited for or against a research candidate.

---

## 1. What the last round left on the table

QUALITY.md established the mechanism and then ran out of dose. Four arms ordered
by "how often a training window contained the request" moved topic propensity
0.0518 → 0.0635 → 0.0752 → 0.0908, monotonically, with a control arm at exactly
0.0518 proving the movement was the corpus and not the extra training. §6 named
the next three things to try:

1. more of the dose that worked;
2. **shorter conversations**, so coverage is a property of the corpus rather than
   of a sampler flag;
3. a mixture ladder for the instruction-following regression the round shipped.

This round does 2 and 3, and 2 is how it does 1.

## 2. The dose was being measured with the wrong quantity

Before building anything, the mechanism metric was rebuilt
(`scripts/chat/window_coverage.py`), and the previous one turns out to have been
two different things at once.

* §4.1 reported **4.6 %** of windows carrying the request, derived as "the
  request sits in the first ~34 characters of a 757-character conversation, so
  only a window starting inside that span contains it". That excludes every
  window that starts in the *previous* conversation and rolls forward into this
  one — which is an entirely ordinary window and does carry the dependency. The
  corrected figure for the same source is **0.248**.
* More importantly, neither version measures the **dose**. A window holding a
  request plus the first 215 characters of a 726-character reply teaches the
  dependency for 215 characters and not for the other 511. The quantity that
  matters is the share of *reply characters* whose gradient is conditioned on
  their own request:

  > for a reply character `p` in a conversation starting at `s`, a window of
  > length `L` containing `p` also contains `s` with probability
  > `clip(1 - (p - s)/L, 0, 1)`

  averaged over every reply character in the source. It is 1.0 only when replies
  are short relative to `L`, and that is the whole point.

Measured before any arm trained, at `seq_len = 256`:

| source | conversation | reply | §4.1's "strict" | carries | dose, unaligned | dose, aligned | **dose at 0.74** |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `stories_topic` — the shipped arm's source | 757 | 726 | 0.039 | 0.248 | 0.132 | 0.304 | **0.260** |
| `stories_short` — new | 241 | 161 | 0.139 | 0.840 | 0.531 | 0.998 | **0.877** |

The realised alignment rate is **0.741, 0.743 and 0.744** for the shipped
mixture and the two new ones respectively — measured over 1,920 windows of each
arm's own mixture, so the two columns are being compared at the same alignment
and the difference is entirely the corpus.

Two readings, and the second is the one QUALITY.md §6 item 2 asked for:

1. The dose rises **3.4×**, a larger single step than any in the series that
   produced the 0.0518 → 0.0908 movement.
2. `stories_short` **unaligned** (0.531) beats `stories_topic` **aligned**
   (0.304). Window coverage has stopped being a property of a sampler flag. That
   also dissolves the trade §3.2 recorded — aligning every window means never
   training on a story's tail — because a 161-character reply has no tail.

## 3. What was built

**`snnchat.shortform`** truncates a TinyStories narrative at a sentence boundary
to a sampled target near 161 characters, chooses the topic from the **full**
story and then requires that word to survive into the kept text (choosing from
the truncation alone yields a topic for 32 % of stories against **88.9 %** this
way), and wraps it in a request. Packed by
`scripts/chat/build_short_stories.py` as a new source, so `tinystories.bin` and
`stories_topic.bin` are untouched and every earlier checkpoint still has its
corpus.

Three judgements, stated as judgements in the module and repeated here because
they are the parts a reader should distrust:

* **A truncated story has no ending**, and a model trained on one will stop
  without resolving. Splicing the original final sentence back on was tried on
  samples and rejected — TinyStories endings refer to objects introduced in the
  middle ("The shiny stone made everyone happy"), so splicing manufactures
  non-sequiturs, which is worse to teach than brevity. Both long story sources
  stay in the mixture.
* **The target length is sampled**, uniform on [140, 235], so the model learns
  "a short story is a few sentences and then it stops" rather than "a story is
  exactly four sentences".
* **About half the requests say "short" or "little"**, and the long sources say
  it never, which makes reply length something the prompt can influence. That is
  a capability the model did not have.

**`RerankParams.temperature_spread`** proposes the best-of-N drafts across a
range of temperatures instead of all at one. The justification is QUALITY.md's
own sharpest null result: reranking moved topicality by *exactly nothing* on a
model whose candidate pool contained no on-topic reply. Selection cannot invent
what was never proposed, so with the scorer already tuned, the pool is the lever
left. It is legitimate to mix temperatures in one pool for a reason specific to
how this scorer is built — `sample_candidates` accumulates `logp_cond` under the
model's own distribution, before temperature and nucleus truncation touch it, so
a draft is never scored under the sampler that proposed it.

The ladder is emitted in van der Corput order rather than ascending. That is not
decoration: `quality.score` compares reranking settings by taking the **first n**
of a larger collected pool, so an ascending ladder would make the `n = 4` row a
cold-only pool and report the difference as an effect of pool size.
`tests/test_snnchat.py::test_every_prefix_of_the_ladder_spans_the_range` is that
guard, and `test_a_zero_spread_draw_is_bit_identical_to_no_ladder` is what keeps
every arm drawn before this comparable — `spread = 0` selects the scalar code
path, not a ladder of equal values.

## 4. Results

### 4.1 The prediction failed, and the failure is informative

**P1 said `chat-v4a-short` would score above `chat-v3d-aligned`'s 0.0908 on topic
propensity. It scored 0.0723.** Not inside the error bar in the right direction —
below it, and above only the two-rounds-old baseline.

| arm | dose | topic propensity | 2 s.e. |
| --- | ---: | ---: | ---: |
| `chat-v2-anneal` | — | 0.0518 | ±0.0139 |
| `chat-v3d-aligned` (shipped) | 0.260 | **0.0908** | ±0.0180 |
| `chat-v4a-short` | 0.877 | 0.0723 | ±0.0162 |

A 3.4× increase in the mechanism the previous round's series was ordered by moved
the outcome **backwards**. Scored as written, the prediction is wrong, and the
"more of the dose that worked" reading of QUALITY.md §6 item 1 does not survive
contact with a larger dose.

**P2 also failed**, by less. Prompt dependence — the metric with no sampler, no
lexicon and no probe set in it — went from **+0.0619 to +0.0572**. The confound
P2 flagged does not arise, and checking that was worth the minute it took:
`prompt_dependence` is computed on `soda`, `alpaca`, `dolly` and `oasst1` only,
never on a story source, and `chat-v4a-short` carries `soda` and `alpaca` at
exactly the weights `chat-v3d-aligned` did. The two numbers are measured on the
same text at the same weights.

The per-source breakdown localises the loss completely, and it is not where the
intervention was:

| source | `chat-v3d-aligned` | `chat-v4a-short` |
| --- | ---: | ---: |
| `alpaca` | +0.0664 | +0.0663 |
| `dolly` | +0.0933 | +0.0923 |
| `soda` | **+0.0494** | **+0.0383** |

Instruction-following dependence is unchanged to the fourth decimal place. The
entire deficit is on `soda` — multi-turn dialogue — at an identical mixture
weight of 0.20 in both arms. A mixture that is 40 % two-turn, 200-character
exchanges appears to cost something on conversations that are neither, and that
is a cost the dose calculation in §2 does not see, because it only counts story
sources.

### 4.2 The one confound worth taking seriously, and it does not rescue the arm

`topic` is scored as "does the reply contain the word asked for", and this round
cut reply length nearly in half. A shorter reply has fewer chances to contain any
given word, so some of the decline is arithmetic rather than behavioural.
`scripts/chat/topic_by_length.py` asks how much:

| arm | candidates | mean chars | topic (raw) | per 100 chars |
| --- | ---: | ---: | ---: | ---: |
| `chat-v2-anneal` | 1,024 | 288 | **0.0527** | 0.0183 |
| `chat-v3d-aligned` | 1,024 | 300 | **0.0908** | 0.0303 |
| `chat-v4a-short` | 1,024 | 166 | **0.0723** | 0.0437 |

Per 100 characters of reply, the new arm is **44 % more on topic than the shipped
one** and 2.4× the old baseline. That is the strongest reading available and it
is not strong enough, because per-100-characters assumes hit probability is
linear in length and it is not — it saturates, which flatters short replies. The
honest version is to compare within a length band, and the honest version
declines to answer:

| arm | 0–100 | 100–200 | 200–300 | 300+ |
| --- | ---: | ---: | ---: | ---: |
| `chat-v2-anneal` | 0.000 (34) | 0.000 (18) | 0.062 (226) | 0.054 (746) |
| `chat-v3d-aligned` | — (0) | — (0) | 0.106 (207) | 0.087 (817) |
| `chat-v4a-short` | 0.125 (16) | 0.073 (846) | 0.064 (156) | 0.000 (6) |

**The two arms barely share a length band.** `chat-v3d-aligned` drew not one
candidate under 200 characters; `chat-v4a-short` put 846 of 1,024 there. The only
band with both is 200–300, where the shipped arm scores 0.106 on 207 candidates
and the new arm 0.064 on 156 — but those 156 are the new arm's *tail*, the draws
where it failed to stop, so that cell compares one arm's typical reply with the
other's atypical one. The difference (0.042) is also inside the combined 2 s.e.
of about 0.058.

So: the length confound is real, it plausibly accounts for a large part of the
decline, and **these draws cannot settle it**. Settling it needs an arm whose
reply-length distribution overlaps the shipped one, which is a fourth arm this
round did not have the budget for. What can be said without qualification is that
the raw rate — which is what a person typing at the model experiences — went
down.

### 4.3 What the arm did buy, which is not nothing

| | `chat-v3d-aligned` | `chat-v4a-short` |
| --- | ---: | ---: |
| instruction following, `list (strict)` | 0.000 | **0.033** |
| identity | 0.938 | **1.000** |
| fallback (greeting/identity evasion) | 0.000 | 0.000 |
| **turn closed by the model itself** | — | **0.94 at n=1, ~1.00 at n≥8** |
| model's own log-prob per character | −0.3934 | **−0.3454** |
| mean reply characters | 189 | 136 |
| story dodge | 0.083 | **0.167** |
| topic (selected reply, n=1) | 0.094 | 0.047 |
| headline | **0.297** | 0.248 |

Two of those matter and pull in opposite directions.

**The reply now ends itself.** `closed` — the model emitting `<|eot|>` rather
than running out of `max_new` — is 0.94 at one candidate and reaches 1.000 at
eight. Combined with a mean per-character log-probability that improved by 0.048,
this is a model that writes shorter, better-formed, more confident text and stops
when it is done. That is a real improvement in "the text it gives back" and it is
the direct consequence of training on 161-character replies that end at a
sentence.

**`story_dodge` doubled, 0.083 → 0.167.** Asked for something that is not a
story, the new arm answers with a narrative twice as often. This is the same
failure QUALITY.md §5 caught in the shipped arm — an evasion replaced rather than
removed — and it is worse here. The likely cause is that `stories_short` makes a
story *cheap*: at 161 characters a story is a plausible answer to almost anything,
where a 700-character one at least had to commit. That is a hypothesis and this
round did not test it.

**The pre-registered headline prefers the shipped arm**, 0.297 to 0.248, and it
was fixed before any model in this package was measured. `chat-v4a-short` does
not ship.

### 4.4 The prediction scorecard

Scored against `docs/chat/PREDICTION_v4.md` as written, not as one would like to
have written it.

| | claim | outcome |
| --- | --- | --- |
| **P1** | `chat-v4a-short` beats 0.0908 topic propensity | **FAILED.** 0.0723. The dose–response relation did not continue. |
| **P2** | `chat-v4a-short` beats +0.0619 prompt dependence | **FAILED.** +0.0572. The confound P2 flagged does not apply — the metric never touches a story source. |
| **P3** | `chat-v4b-balance` beats 0.000 on strict `list` | **CORRECT.** 0.083, the best in the package, and §4.5 closes the length confound on it. |
| **P4** | replies get shorter, and a `list` gain that arrives with a length collapse is an artefact | **CORRECT, and it fired.** 189 → 136 mean characters, `list (strict)` 0.000 → 0.033. The warning applies to `chat-v4a-short`'s own best number, and §4.5 is how it was answered rather than argued around. |
| **P5** | `fact` and cross-turn memory do not move | **CORRECT.** `fact` 0.036 → 0.036 on `chat-v4a-short`. |

Three of five correct, and the two that failed are the two the round was built
around. The value of having written them down first is in §4.2 and §4.5: the
length confound was named as a risk *before* any arm produced a result it would
have been used to excuse — so against `chat-v3d-aligned` it stands unresolved and
is reported as unresolved, and between the two new arms, which happen to land one
character apart in mean reply length, it is closed and the gain is real.

### 4.5 The mixture ladder: P3 confirmed, and the length confound cannot explain it

`chat-v4b-balance` cuts total story weight 0.49 → 0.34 and raises instruction
sources 0.23 → 0.36. **P3 said it would beat 0.000 on strict `list`. It scored
0.083**, the best instruction following any checkpoint in this package has had.

| | `chat-v3d-aligned` | `chat-v4a-short` | `chat-v4b-balance` |
| --- | ---: | ---: | ---: |
| `list` (loose) | — | 0.083 | **0.183** |
| **`list` (strict, <120 chars)** | 0.000 | 0.033 | **0.083** |
| prompt dependence | +0.0619 | +0.0572 | **+0.0658** |
| story dodge | **0.083** | 0.167 | 0.146 |
| social register | **1.000** | **1.000** | 0.900 |
| topic propensity | **0.0908** | 0.0723 | 0.0645 |
| mean chars, selected reply | 189 | 136 | 127 |
| **mean chars, all 1,024 drawn** | 300 | **166** | **167** |
| headline | **0.297** | 0.248 | 0.257 |

**The last-but-one row is what makes this round's most useful comparison
possible.** P4 warned that a `list (strict)` gain could be an artefact of shorter
replies, since the metric requires a reply under 120 characters. Against
`chat-v3d-aligned` that warning stands and cannot be dismissed. But
`chat-v4a-short` and `chat-v4b-balance` draw candidates averaging **166 and 167
characters** — a difference of one character over 1,024 draws. Between those two
arms the length confound is closed by construction, and strict `list` still goes
**0.033 → 0.083**. The mixture is doing the work, not the length.

The same matched-length reading applies to what it cost, and this is a trade
rather than a free lunch:

* **Topic propensity falls again**, 0.0723 → 0.0645, at identical reply length.
  Less story weight buys less topic conditioning, exactly as the mechanism
  predicts, and this pair is the cleanest measurement of that trade in the
  package because nothing else differs.
* **Social register regresses**, 1.000 → 0.900 selected and 0.9812 → 0.9094 over
  the pool. Raising instruction weight to 0.36 costs greetings and farewells.
  That is a new regression this round introduced and it belongs in the ledger
  next to the `list` gain.

**Prompt dependence is the best in the package at +0.0658**, and the per-source
breakdown says where it came from:

| source | `chat-v3d-aligned` | `chat-v4b-balance` |
| --- | ---: | ---: |
| `alpaca` | +0.0664 | **+0.0770** |
| `dolly` | +0.0933 | **+0.1099** |
| `soda` | **+0.0494** | +0.0424 |

Instruction sources up substantially, dialogue still down. **Read this one more
carefully than §4.1's**, though: `chat-v4b-balance` changes the weights of
`alpaca` and `soda` themselves, so unlike `chat-v4a-short` this is not a
measurement at matched mixture. It says the model got better at conditioning on
instructions it saw more of, which is a weaker claim than it looks.

**`story_dodge` moved 0.167 → 0.146 and that is the round's clearest miss.** The
arm exists partly to test the hypothesis that a 161-character story is a cheap
answer to anything; cutting story weight by a third moved the symptom by 0.021,
which is inside the noise of a 1,024-draw rate at this level. Both new arms
remain far worse than the shipped 0.083. **The story-weight hypothesis is not
supported**, and whatever makes these models answer non-story prompts with
narrative is something the short-conversation corpus introduced that a mixture
weight does not remove.

**Nothing ships.** The pre-registered headline is 0.297 for `chat-v3d-aligned`
against 0.257 and 0.248, and it was fixed before any model in this package was
measured. `experiments/chat/SHIPPED` is unchanged and carries a note saying why.

### 4.6 The temperature spread: a null on the thing it was built for

The claim in §3 was specific and therefore falsifiable: reranking cannot invent a
reply that was never proposed, so widening the pool should raise the pool's own
topicality. **It does not.** Measured on three checkpoints, each against its own
no-spread draw of the identical probes and seeds:

| model | topic propensity, pool of 1,024 | with spread 0.3 |
| --- | ---: | ---: |
| `chat-v3d-aligned` (shipped) | 0.0908 | 0.0830 |
| `chat-v4a-short` | 0.0723 | 0.0781 |
| `chat-v4b-balance` | 0.0645 | 0.0664 |

Two up, one down, every move far inside a 2 s.e. of about ±0.017, and mean
candidate length unchanged to within one character in all three. Including the
shipped model as a control is what makes this readable: if the spread worked, the
model with the most room — the one whose pool is most often off topic — should
have shown it, and it moved the wrong way.

**What it does do, consistently, is something it was not built for.** The model's
own mean per-character log-probability of the reply it selects improves in
**all six** comparisons:

| | no spread | spread 0.3 |
| --- | ---: | ---: |
| `chat-v3d-aligned`, n=1 / n=8 | −0.3934 / −0.3251 | **−0.3018 / −0.2867** |
| `chat-v4a-short`, n=1 / n=8 | −0.3454 / −0.2605 | **−0.2649 / −0.2337** |
| `chat-v4b-balance`, n=1 / n=8 | −0.3417 / −0.2865 | **−0.2757 / −0.2806** |

which is what a ladder containing colder drafts should do to a score with a
likelihood term in it. On the shipped model it also takes `story_dodge` from
0.062 to **0.021** and `list (strict)` from 0.000 to **0.117** at n=8.

**The default stays at 0.0 anyway**, and the reason is a rule that predates this
round rather than a reading of these rows. QUALITY.md §5 fixed the decoder by
"maximise topicality subject to the model not finding its own replies less likely
than without reranking". At n=8, λ=0.6 the spread takes the shipped model's
selected-reply `topic` from 0.141 to 0.078. It fails the first clause. The
`story_dodge` and `list` movements are 64-sample columns, they were noticed after
the fact rather than predicted, and promoting them to a default would be choosing
an objective after seeing which one the intervention won — the exact failure
QUALITY.md §5 documents three separate instances of.

So: the mechanism is built, tested, wired into `scripts/chat.py` as `/spread`,
and **off**. What it is now is a measured hypothesis with a plausible next
experiment — it looks like a *fluency* intervention rather than a topicality one,
and the way to test that is a fluency-first objective with a topicality floor,
which is the mirror image of the rule above and would need its own
pre-registration.

### 4.7 One measurement where the short corpus wins outright

`scripts/chat/topic_dependence.py` scores a real story from `stories_topic`'s
held-out tail behind its own request and behind another story's. No sampler, no
seed, no lexicon, no probe list — and, for the two new arms, **on a source
neither of them ever trained on**:

| arm | bpc, own request | bpc, another request | delta |
| --- | ---: | ---: | ---: |
| `chat-v2-anneal` | 1.0111 | 1.0122 | +0.0012 |
| `chat-v3d-aligned` | 0.9820 | 0.9847 | +0.0027 |
| `chat-v4a-short` | 1.0269 | 1.0304 | **+0.0035** |
| `chat-v4b-balance` | 1.0578 | 1.0601 | +0.0023 |

`chat-v4a-short` extracts the most from a request of any checkpoint in the
package, on long stories it has never seen, having been trained only on short
ones. The absolute columns are worse for exactly the reason the script's
docstring warns about — it did not train on this source — but the delta is a
within-model quantity.

This is the one place the short-conversation corpus is unambiguously ahead, and
it is worth weighing against §4.1 rather than instead of it. The numbers are
small (a 34-character request against a 700-character story cannot be worth much)
and the ordering is not resolved by any error bar computed here. Taken with §4.2,
the fair summary is that **the conditioning mechanism plausibly did improve and
the reply the user actually receives did not**, and this round could not close the
gap between those two statements.

### 4.8 Where the numbers live

Every table above regenerates from `experiments/chat/_quality/`:

```bash
python scripts/chat/compare_quality.py experiments/chat/_quality/chat-v*.json
python scripts/chat/topic_by_length.py experiments/chat/_quality/chat-v*.json
```

## 5. What this round does not do, and what to distrust in it

* **It buys no knowledge and no cross-turn memory.** Neither does anything else
  at 4.4 M parameters with a 47-character neuron horizon. `fact` stays where it
  was and the model still cannot repeat a name you gave it one turn ago.
* **Every topic probed was taught**, exactly as QUALITY.md §6 records. The
  request form is manufactured by `snnchat.shortform` and the probe set asks for
  that form. The topics themselves come from the corpus's own word counts and
  nothing here was written by reading which probes failed — but this is "the
  model learned a form it was shown", not "the model generalises to unseen
  topics".
* **n = 1 per arm.** `EXP_005`'s lesson — that σ is not a project constant and
  must be re-measured on every structurally new arm — applies here and was not
  paid for. Two arms separated by less than a few points on the 64-sample
  battery are not ordered, which is why the training comparison is read off the
  ~1,024-sample propensity column.
* **`chat-v4b-balance` moves two things at once.** Story weight down, instruction
  weight up. A result on it is attributable to the pair.
* **The `list (strict)` metric is partly a length detector.** It requires a reply
  under 120 characters, and this round deliberately shortens replies. `mean
  chars` is in the battery table for exactly that reason and the two columns must
  be read together — which is why `compare_quality.py` now prints it.
* **There is no control arm for the corpus change in this round.** The previous
  round's `chat-v3c-control` established that 14,000 further steps on the old
  mixture buy nothing, and both arms here are matched to `chat-v3d-aligned` in
  step count, schedule, base checkpoint and objective flags. That is a matched
  baseline, not a control, and the distinction is the one QUALITY.md §4.3 was
  careful about.
* **The temperature spread was tested at one value.** 0.3, against 0, on three
  checkpoints. It is a knob with a plausible optimum somewhere and nothing here
  found it -- what §4.6 establishes is that 0.3 does not move the pool's
  topicality on any of the three, not that no spread ever could.
* **§4.6's fluency and `story_dodge` movements were noticed after the fact.**
  They are consistent across six comparisons and two of them are large, and they
  are still 64-sample columns found by reading a table the intervention did not
  predict. They are written down as a hypothesis for a future measurement and
  they did not change a default.

## 6. Reproducing it

```bash
# the short-conversation story source (adds files, changes nothing existing)
python scripts/chat/build_short_stories.py --limit-chars 320000000

# the dose, before anything trains
python scripts/chat/window_coverage.py stories_short stories_topic \
    --seq-len 256 --align-frac 0.74

# an arm
python scripts/chat/train.py --run-name chat-v4a-short \
    --init-from experiments/chat/chat-v2-anneal/ckpt_best.pt --no-resume \
    --max-steps 14000 --batch-size 160 --seq-len 256 --lr 5e-4 \
    --bot-loss-weight 3.0 --align-frac 0.75 --align-lookahead 1024 \
    --budget-minutes 46 \
    --mix stories_short=0.40 soda=0.20 alpaca=0.15 tinystories=0.09 \
          persona=0.08 dolly=0.06 oasst1=0.02

# score it; --temperature-spread changes the DRAWS, so it is a separate file
python scripts/chat/quality.py --ckpt experiments/chat/chat-v4a-short/ckpt_best.pt \
    --out experiments/chat/_quality/chat-v4a-short.json
python scripts/chat/quality.py --ckpt experiments/chat/chat-v4a-short/ckpt_best.pt \
    --temperature-spread 0.3 --label chat-v4a-short.spread \
    --out experiments/chat/_quality/chat-v4a-short.spread.json

# every arm side by side
python scripts/chat/compare_quality.py experiments/chat/_quality/chat-v*.json

# and the question §4.2 exists to ask
python scripts/chat/topic_by_length.py experiments/chat/_quality/chat-v*.json
```

## 7. What this leaves for the next round

0. **Two interventions are now measured and dead**, which is the round's main
   contribution: the window-coverage dose (§4.1) and the candidate-pool
   temperature spread (§4.6). Neither should be tried again in its current form.

1. **The dose is not the binding constraint any more.** It went 0.260 → 0.877 and
   the outcome went down. Whatever now limits topicality at ~9 %, more windows
   carrying the request is not it, and the next round should stop spending GPU
   hours on that axis. QUALITY.md §6 item 1 is closed, answered "no".
2. **The length confound needs one arm to settle it**, and it is the cheapest
   informative arm available: `stories_short` with the truncation target raised to
   ~300 characters, which would overlap the shipped arm's length distribution and
   make §4.2's band table answer instead of decline. It would also separate "short
   replies hurt topicality" from "short *training* conversations hurt topicality",
   which nothing here can.
3. **`story_dodge` is the worst number in the package** at 0.167 and 0.146, and
   its named suspect has been **eliminated**: cutting story weight by a third
   moved it 0.021, inside the noise. Something the short-conversation corpus
   introduced makes these models answer non-story prompts with narrative, and it
   is not the mixture weight. Next suspect worth testing is reply *length* itself
   — at 161 characters a story costs the model almost nothing to start — which
   the ~300-character truncation arm in item 2 would test at the same time.
4. **Instruction weight and social register trade against each other**, newly
   measured: 0.23 → 0.36 instruction weight bought `list (strict)`
   0.033 → 0.083 and cost `social` 1.000 → 0.900 at matched reply length. Neither
   end of that trade has been laddered; two points is a direction, not a curve.
5. **What actually improved is worth keeping and is orthogonal to all of the
   above.** Replies that end themselves (`closed` 0.94 → ~1.00), a better
   per-character log-probability, `identity` at 1.000 and `fallback` at 0.000 came
   from the short-conversation corpus and cost nothing but length. If the next
   round finds a way to keep those without the topicality cost, that is the
   product.
