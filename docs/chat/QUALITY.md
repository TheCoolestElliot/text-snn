# Making the replies better, and measuring whether they got better

`docs/chat/RESULTS.md` ends with a judgement: the model is **fluent without being
responsive** — "it reliably produces well-formed English of the right shape for
the turn, and only sometimes produces English that is about what you asked". This
file is the attempt to close that gap, and the record of what closing it took.

**Not part of the research protocol.** Nothing here is pre-registered, no number
is a reported figure, there are no control arms in the protocol's sense, and n=1
per arm throughout. It must not be cited for or against any research candidate.

---

## 1. The gap had to be measured before it could be closed

Reading twenty replies cannot order two checkpoints: the sampler is stochastic,
the prompts move, and the reader knows which model is supposed to be better. So
the first thing built was `scripts/chat/quality.py`, which scores replies
mechanically. Two metrics carry the weight.

**Prompt dependence** (`snnchat.quality.prompt_dependence`) is the one to trust.
Take held-out `(user, reply)` pairs from the val splits and score each real reply
twice under the model: once behind its own user turn, once behind a different
conversation's. The difference, in bits per character, is what the real prompt
was worth. No sampler, no seed, no lexicon, no probe set — a model that ignored
the prompt entirely would score zero.

The scorer that computes it pads many pairs into one batch and selects each row's
own positions with a mask, which is exactly the kind of index arithmetic that is
wrong by one and still returns a plausible number. So it is made to reproduce a
number it did not produce: one batch is re-scored by feeding the conversation
through the model **one character at a time**, the path `ChatSession` actually
uses, and the probe fails hard if the two disagree. On every run in this file the
two agree to a relative 3e-08. `tests/test_snnchat.py::test_the_scorer_check_can_actually_fail`
mutation-tests that guard, because a check that cannot fail is decoration.

**The battery** is 37 single-turn probes at 4 seeds, scored by kind:

| kind | what it asks | how it is scored |
| --- | --- | --- |
| `topic` | "tell me a story about a rabbit" | does the reply contain the topic word |
| `list` | "name three animals" | how many distinct members of a lexicon |
| `social` | "hello", "thanks!" | does the reply use the expected register |
| `identity` | "what are you?" | does it mention what it is |
| `fact` | "what is 17 plus 25?" | expected to fail, retained anyway |

plus a **fallback rate** — how often a substantive prompt is answered with the
greeting or the identity line, which is this model's characteristic evasion and
is invisible to bits per character, because bpc *likes* those replies.

The single `headline` number the sweep is ordered by is
`0.5·topic + 0.3·list + 0.2·social − 0.2·fallback`. It is a judgement, and the
one thing worth saying about it is *when* it was fixed: the formula was written
into `snnchat.quality.score` before any model was measured, so no arm was chosen
by an objective built after seeing it. `fact` is deliberately absent — nothing in
this round can teach a 4.4 M-parameter character model a fact, and putting it in
the objective would only add noise.

---

## 2. What the measurement said, and the diagnosis it forced

The shipped model, `chat-v2-anneal`:

| | |
| --- | --- |
| prompt dependence | **+0.0474 bpc** |
| topicality | **0.078** |
| instruction following (`list`) | **0.017** |
| social register | 0.950 |
| identity | 0.875 |
| facts | 0.000 |

The per-probe breakdown is where the diagnosis is, and it is not a smooth
failure:

```
0.50  tell me a story about a rabbit
0.50  tell me a story about a little girl named Mia
0.25  tell me a story about a snowman
0.00  tell me a story about a dragon / robot / boat / pirate / kite / frog
0.00  ... every other common noun
```

A **name** works half the time. A **noun** never works. That split is not a
capacity limit, and finding it took one table.

`build_corpus._story_conversation` builds its topic-conditioned requests with
`_MIDSENTENCE_CAP`, a regex for a capitalised word in the middle of a sentence.
In TinyStories that is the character's name — Lily, Tim, Max. So the corpus
contains thousands of examples of *"tell me a story about Lily"* answered by a
story about Lily, and **no examples at all** of *"tell me a story about a
rabbit"* answered by a story about a rabbit.

The model learned the capability it was shown, and `docs/chat/README.md` then
advertised, and `docs/chat/transcript.md` then measured, a capability it was
never shown. The 7.8 % is not the model being small. It is the model being
right.

---

## 3. Two interventions, kept separable

### 3.1 Decoding: best-of-N by mutual information (`snnchat.rerank`)

Draw N replies instead of one and keep the one the prompt did the most work for:

```
score(y) = [ log P(y | prompt) − λ · log P(y | empty prompt) ] / len(y)
```

The second term is scored under the identical model given a user turn with no
text in it, so it measures "would it have said this anyway". λ = 0 is best-of-N
by likelihood alone; λ = 1 is pointwise mutual information.

This is worth trying **on this network specifically** because
`docs/chat/BUILD_NOTES.md` §1 measured the per-timestep scan to be latency-bound
at these widths — ~26 µs per timestep per layer, nearly independent of batch size
up to `B·d ≈ 200 k`. At d = 1024 that is a factor of ~200 of unused batch, so
extra drafts ride along in the same batch instead of being paid for one at a
time. §4.4 measures what that actually comes to: **16 drafts for 2.1× the wall
clock**, not the 1.0× an idealised reading of the scan would predict and not the
16× a transformer would pay.

`RerankParams(n=1)` is exactly the old sampler, character for character at the
same seed, and
`tests/test_snnchat.py::test_a_single_candidate_rerank_is_the_ordinary_sampler`
is what makes that a guarantee rather than an intention.

### 3.2 Data and objective: `snnchat.topics`, `bot_loss_weight`, `align_frac`

Three changes, packed into two arms so the data change can be told apart from the
training-objective change:

1. **A story corpus asked for by subject.** `snnchat.topics.topic_of` picks the
   content word a story keeps coming back to — frequency within the story, a stop
   list of function words and narrative verbs, character names excluded, and a
   determiner test so the topic has to have been used as a noun ("a story about
   hopped" is what that filter is for). 95 % of TinyStories yields a usable
   subject and 6,000 stories yield 1,197 distinct ones.
   `scripts/chat/build_topic_stories.py` packs it as a **new** source, so
   `tinystories.bin` is untouched and every earlier run still has its corpus.

2. **`bot_loss_weight`** — the characters of the model's own turn weigh more in
   the loss than the characters of the prompt. This is *upweighting, not
   masking*: `snnchat.train`'s docstring argues against masking and the argument
   is right — at this scale the model needs every character of language signal,
   and it must be able to predict a user turn in order to represent where one
   ends. A weight keeps all of that and re-apportions it. At 1.0 the unweighted
   code path runs, unchanged, and every earlier run reproduces bit for bit.

3. **`align_frac`** — a fraction of training windows are nudged forward to start
   at a conversation boundary. At `seq_len = 256` with turns up to 400
   characters, a uniformly-placed window frequently contains a reply whose
   request is outside it, so the gradient that would teach "the story is about
   what was asked for" is not available in most of the windows that contain a
   story. Aligned windows are also the state `ChatSession` begins a real
   conversation in.

   The requested fraction and the realised one differ, and the difference is
   worth stating rather than discovering later: an offset is only moved to a
   conversation start it can reach within `align_lookahead` characters, which
   defaults to `seq_len`, and many conversations are longer than 256 characters.
   Measured over 12 batches of the arm's own mixture
   (`experiments/chat/_quality/alignment_realised.json`):

   | `align_frac` | `align_lookahead` | windows that begin a conversation |
   | ---: | ---: | ---: |
   | 0.0 | 256 | 0.2 % |
   | 0.75 | 256 (= `seq_len`, the default) | **38.2 %** |
   | 0.75 | 1024 | 74.0 % |
   | 1.0 | 1024 | 99.4 % |

   So the arm labelled 0.75 is an arm at 38 %. `MixtureSampler`'s docstring makes
   the same point about mixture weights — "the mix I asked for and the mix I got
   differing is exactly the kind of thing that is invisible until someone reads a
   sample and wonders why it sounds like a story".

   The last row is in the table to be argued against. Aligning *every* window
   would mean the model only ever trains on the first 256 characters of a
   757-character story, so two thirds of the corpus's prose would never be seen.
   The windows left unaligned are what covers the tails.

---

## 4. Results

Three arms, all initialised from `chat-v2-anneal/ckpt_best.pt`, all 14,000 steps
at B = 160, L = 256, lr 5e-4 cosine to a 5 % floor, ~40 minutes each on one RTX
5060. They differ in exactly the two things being tested:

| arm | corpus | objective | steps |
| --- | --- | --- | ---: |
| `chat-v2-anneal` | — | the previously shipped checkpoint, not retrained | — |
| `chat-v3c-control` | the **old** mixture | unweighted, unaligned | 14,000 |
| `chat-v3a-data` | + `stories_topic` at 0.40 | unweighted, unaligned | 13,258 |
| `chat-v3b-objective` | + `stories_topic` at 0.40 | `bot_loss_weight 3.0`, `align_frac 0.75` (realised 38 %) | 14,000 |
| `chat-v3d-aligned` **(shipped)** | + `stories_topic` at 0.40 | as above, `align_lookahead 1024` (realised 74 %) | 14,000 |

`chat-v3c-control` is the arm that makes the rest readable. The shipped
checkpoint is not a matched-length baseline, so "shipped vs `chat-v3a-data`"
would confound the new corpus with 14,000 extra steps of training on any
mixture. The control removes that: same base, same steps, same schedule, the old
mixture. Whatever it moves is what more training moves.

### 4.1 A prediction, written down before the arm that tests it

`chat-v3a-data` finished first, and **it did not move topicality at all**: 0.078
before, 0.078 after. The `list` and `fallback` columns moved; the column the
whole corpus change was built for did not.

Rather than explain that after the fact, here is the explanation and the
prediction it makes, recorded at 09:55 on 2026-08-06 while `chat-v3b-objective`
was still training and before any battery had been run against it.

**The explanation.** A reply can only learn to depend on a request that is inside
its own training window. Measured on the packed source
(`experiments/chat/_quality/window_coverage.json`):

| | |
| --- | --- |
| mean conversation length in `stories_topic` | **757 characters** |
| median | 737 |
| characters before the `<\|bot\|>` marker (the request) | mean 34 |
| training window | 256 |
| **random-offset windows carrying the request AND its story** | **4.6 %** |

So `chat-v3a-data` spent 40 % of its mixture on subject-conditioned stories and
saw the subject *attached to* its story in about one window in twenty. The other
nineteen were bare narrative with the request out of frame, which teaches
fluency and nothing about conditioning. **The corpus was right and it was being
delivered in a form that could not teach the thing it was built to teach.**

Note that a longer window would not fix this. The constraint is where a window
*starts*, not how long it is: the request sits in the first ~34 characters of a
conversation, so only a window starting inside that span contains it, and
`L = 512` has exactly the same 35 good offsets out of 757 that `L = 256` has.
Alignment is the only lever.

**The prediction.** `chat-v3b-objective` turns on `align_frac = 0.75`, which
raises the share of `stories_topic` windows starting at a conversation boundary
to roughly `0.75 x 256/757 ~ 25 %`, plus the 4.6 % that land there by chance --
a **~6x increase** in windows that carry the dependency at all. If the diagnosis
is right, its topicality should be materially above 0.078. If it is not -- if
`chat-v3b-objective` also sits at 0.078 -- then window coverage is not the
binding constraint either, and the honest conclusion is that 14,000 steps at 4.4
M parameters cannot install this mechanism no matter how the data is presented.

The arm also changes `bot_loss_weight`, so a positive result is attributable to
the pair and not to alignment alone.

**A resolution problem, and the fix for it.** The `topic` column scores the one
reply selected per (probe, seed): 16 probes x 4 seeds = **64 samples**. Near
p = 0.08 that is a binomial standard error of 0.034, so nothing below a ~0.07
difference between two arms is resolvable and every arm above is inside the
noise band of every other. Saying "topicality did not move" on 64 samples is
therefore a statement about the measurement as much as about the model.

The fix costs no GPU time at all: score **every candidate drawn**, not just the
selected one. That is ~1,024 samples per arm, it resolves to about 0.02, and it
is a cleaner quantity anyway — it measures the model's own propensity to mention
what it was asked about, with the decoder taken out of it entirely. It is
reported as `topic propensity` below and it is the column the training
comparison should be read from. The selected-reply column stays, because it is
what a user actually receives.

### 4.2 The prediction, scored as it was written

**As literally stated, it failed on the arm it named.** §4.1 said
`chat-v3b-objective`'s topicality "should be materially above 0.078", and the
column that sentence refers to — the selected-reply `topic` score — came back at
**0.062**, which is below it. Scored strictly: the bar was missed.

Scored usefully: that column has 64 samples and a standard error of 0.034, so it
could not have resolved the effect either way, and saying "the prediction failed"
without saying that would be reporting a coin flip as a result. The
better-powered measurement — the same hit rate over all ~1,024 drawn candidates —
was **introduced after the prediction was written**, in response to noticing the
resolution problem, and it supports the prediction cleanly. Both facts belong in
the record, and a reader should weight the second knowing it was chosen after the
first disappointed.

What is *not* post-hoc is the ordering. Four arms, ranked by a mechanism measured
before any of them trained, come out in the predicted order on a metric with the
sample size to see it.

### 4.3 The tables

Regenerate any of these with
`python scripts/chat/compare_quality.py experiments/chat/_quality/*.json`.

**Topic propensity — the hit rate over every one of ~1,024 drawn candidates.**
This is the column the training comparison should be read from, and it is
ordered exactly as window coverage is:

| arm | windows carrying the request | topic propensity | 2 s.e. |
| --- | ---: | ---: | ---: |
| `chat-v2-anneal` (previous ship) | — | 0.0518 | ±0.0139 |
| `chat-v3c-control` (14,000 steps, **old corpus**) | — | 0.0518 | ±0.0139 |
| `chat-v3a-data` | 4.6 % | 0.0635 | ±0.0152 |
| `chat-v3b-objective` | ~27 % | 0.0752 | ±0.0165 |
| `chat-v3d-aligned` | ~76 % | **0.0908** | ±0.0180 |

**The control moved it by nothing at all** — 0.0518 against 0.0518, 53 hits in
1,024 both times. Fourteen thousand further steps at the same learning rate on
the same schedule, differing only in which corpus they were spent on, buy exactly
zero topicality. That is what makes the rest of the column a comparison rather
than a trend: the gain is attributable to the corpus and the window placement,
and not to having trained for longer.

`chat-v3d-aligned` is **+75 % relative** on the baseline, a gap of 0.039 against
a combined 2 s.e. of about 0.023. Every arm in between is ordered by how often a
training window actually contained the request, which is what §4.1 predicted in
writing before the last arm was trained.

**Prompt dependence — no sampler, no lexicon, no probe set.**

| arm | step | bpc, own prompt | bpc, another prompt | delta |
| --- | ---: | ---: | ---: | ---: |
| `chat-v2-anneal` | 7,874 | 1.4789 | 1.5263 | **+0.0474** |
| `chat-v3c-control` | 14,000 | 1.4819 | 1.5283 | **+0.0464** |
| `chat-v3a-data` | 13,258 | 1.5074 | 1.5509 | **+0.0435** |
| `chat-v3b-objective` | 14,000 | 1.4961 | 1.5583 | **+0.0622** |
| `chat-v3d-aligned` | 14,000 | 1.5157 | 1.5776 | **+0.0619** |

The control is flat here too (+0.0464 against +0.0474). The two arms that changed
the *objective* gain ~31 %; the arm that changed only the corpus does not.
That is a different cut of the same finding: `bot_loss_weight` and window
alignment are what made the prompt matter, and the corpus is what gave it
something to say.

Measured on the story source itself
(`scripts/chat/topic_dependence.py`), where a request is ~34 characters and the
story it conditions is ~700:

| arm | bpc, own request | bpc, another request | delta |
| --- | ---: | ---: | ---: |
| `chat-v2-anneal` | 1.0111 | 1.0122 | +0.0012 |
| `chat-v3a-data` | 0.9509 | 0.9525 | +0.0016 |
| `chat-v3b-objective` | 0.9720 | 0.9738 | +0.0017 |
| `chat-v3d-aligned` | 0.9820 | 0.9847 | **+0.0027** |

Small numbers, because most of a 700-character story is not determined by a
34-character request — but the shipped arm extracts **2.3x** as much from it as
the baseline did. Three independent measurements, one of which has no sampler
and no word list in it, all move the same way.

**The battery, every arm at the same decoder setting.**

| arm | n | lam | topic | list (strict) | social | identity | fact | fallback | story dodge | logp/char | headline |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `chat-v2-anneal` | 1 | 0 | 0.078 | 0.000 | 0.950 | 0.875 | 0.000 | 0.036 | 0.021 | -0.4395 | 0.227 |
| `chat-v2-anneal` | 8 | 0.6 | 0.047 | 0.017 | 1.000 | 0.938 | 0.036 | 0.045 | 0.021 | -0.3493 | 0.220 |
| `chat-v3a-data` | 1 | 0 | 0.078 | 0.050 | 1.000 | 0.875 | 0.000 | 0.018 | 0.083 | -0.4029 | 0.266 |
| `chat-v3b-objective` | 1 | 0 | 0.062 | 0.000 | 1.000 | 0.875 | 0.000 | 0.009 | 0.042 | -0.3999 | 0.255 |
| `chat-v3d-aligned` | 1 | 0 | 0.094 | 0.000 | 1.000 | 0.938 | 0.036 | 0.000 | 0.083 | -0.3934 | **0.297** |
| `chat-v3d-aligned` | 8 | 0.6 | **0.141** | 0.000 | 0.900 | 0.938 | 0.071 | 0.018 | 0.062 | -0.3251 | 0.262 |

### 4.4 What it costs, measured

The README claimed N candidates were "close to free". They are not free; they
are **cheap**, and the difference is worth stating because it was asserted before
it was measured. The same 20-prompt battery, same seed, same model:

| decoder | wall clock | characters | rate |
| --- | ---: | ---: | ---: |
| one candidate | 8.3 s | 3,624 | 437 char/s |
| sixteen candidates | 17.5 s | 3,598 | 206 char/s |

**Sixteen drafts cost 2.1x**, not 16x — a marginal cost of about 0.07x per extra
draft, which is the latency-bound scan of `BUILD_NOTES.md` §1 showing up exactly
where it was predicted to. It is still a factor of two, and a claim of "free"
would have been wrong.

### 4.5 The honest size of the win

The propensity is up 75 % relative. It is up **from 5 % to 9 %**, and the
per-probe table says what that means:

| probe | shipped | `chat-v3d-aligned` |
| --- | ---: | ---: |
| a story about a little girl named **Mia** | 1.00 | 1.00 |
| a story about a **birthday cake** | 0.00 | 0.25 |
| a story about a **frog** in a pond | 0.25 | 0.25 |
| a story about a **dragon / robot / pirate / kite / train / moon / spider / teacher / bicycle** | 0.00 | 0.00 |

Nine of sixteen topics still land at zero for every arm. **The model is more
likely to be about what you asked; it is still usually not.**

> **Correction, 2026-08-08 — appended, not applied, because this is a closed
> round record.** "Nine" could not be reproduced from any committed arm at any
> reading row. Recomputing from `experiments/chat/_quality/*.json` at the pinned
> `n = 1, λ = 0` row: **thirteen** of sixteen score zero for `chat-v2-anneal` and
> **thirteen** for `chat-v3d-aligned`, and the set that is zero in *both* — which
> is what "for every arm" would mean — is **eleven**, not nine. Across all
> twenty-one arms the project has scored, the per-arm count ranges 11–15 and
> **none is nine**. So this is not, as `TOPIC_DOSE_NOTE.md` §2 supposed, a figure
> read off an earlier arm's table; it matches no arm. The sentence is left as
> written because a round record reports what it reported. The two arms' claims
> above and the conclusion drawn from them are unaffected — the direction and the
> magnitude of the topic gain are measured elsewhere in this file and stand. The mechanism has
been identified, the fix has been shown to work in the direction predicted and to
scale with the dose, and the dose available in a 40-minute anneal is not enough
to finish the job. Section 6 says what would be.

---

## 5. A correction to the scorer, disclosed

The acknowledgement and farewell word lists were **widened after reading the
baseline's own draws**. "Happy to help." and "See you. Have a good one." were
being counted as failures on `thanks!` and `goodbye`, which is a defect in the
scorer and not in the model — but widening a metric after seeing what a model
produced is exactly how a metric gets gamed, so the change is recorded here
rather than absorbed:

| | `social` at n=1 | `social` at n=8, λ=1 | best headline |
| --- | ---: | ---: | ---: |
| as first written | 0.950 | 0.650 | 0.244 |
| corrected | 0.950 | **1.000** | 0.274 |

The narrow list made reranking look like it was *damaging* the social register,
by 30 points, when what it had actually done was pick a different correct
farewell. Every arm in this file is scored by the corrected version, applied by
re-running the sweep over saved draws (`quality.py --rescore`) so that arms drawn
before the correction and after it are judged by one rule. The pre-correction
file is kept at `experiments/chat/_quality/chat-v2-anneal.narrow-social.json`.

This is the second time in this package that a first reading pointed the wrong
way (`docs/chat/RESULTS.md` records the first, where `chat-v2` was behind at its
first evaluation and ended 0.15 bpc ahead).

**A second correction, in the other direction.** `list` counts lexicon members
anywhere in the reply, and `chat-v3a-data`'s own draws showed why that is too
generous: a *story* beginning "One day, a big fish named Buddy ... a dog named
..." satisfies "name three animals" while following no instruction at all. The
`list (strict)` column additionally requires the reply to be under 120
characters — short enough to be a list rather than a narrative. It roughly halves
the arm's apparent instruction following:

| | `list` | `list` (strict) |
| --- | ---: | ---: |
| `chat-v2-anneal`, n=1 | 0.017 | **0.000** |
| `chat-v3a-data`, n=1 | 0.100 | **0.050** |

The strict column is a **diagnostic and is not in the headline**, which stays as
it was pre-registered. Reporting only the loose column would have credited this
round with roughly twice the instruction-following gain it earned.

**A third correction, which changed what shipped.** The `fallback_rate` counts
one specific evasion: answering a substantive prompt with the greeting or the
identity line. On `chat-v3d-aligned` it fell to **0.000**, and reading the
transcript showed why that was not the victory it looked like — the evasion had
been *replaced*, not removed. "Name three animals" now got a story about a bird.
A metric that counts only the failure a model used to have will always report
progress.

`story_dodge` counts the new shape (a narrative in reply to a prompt that asked
for none), and `mean_logp` — the model's own mean per-character log-probability
of the reply it selected — was added alongside it, because the anti-LM term
trades fluency for specificity and nothing was watching the thing being traded.
Both are free: the numbers were already in the saved draws. Together they moved
the shipped decoder from `n=16, λ=0` (the headline's argmax, and a 16.7 %
story-dodge rate) to `n=8, λ=0.6`:

| setting | topic | story dodge | logp/char | headline |
| --- | ---: | ---: | ---: | ---: |
| no reranking | 0.094 | 0.083 | −0.3934 | 0.297 |
| `n=16, λ=0` — headline argmax | 0.094 | **0.167** | −0.2557 | **0.308** |
| `n=8, λ=0.6` — shipped | **0.141** | 0.062 | −0.3251 | 0.262 |
| `n=8, λ=1.0` | 0.109 | 0.000 | **−0.4277** | 0.238 |

The shipped row is chosen by a rule stated before the row was picked: **maximise
topicality subject to the model not finding its own replies less likely than it
does without reranking**. λ = 1.0 breaks that floor, and the transcript at that
setting contains "There offer the bank is for a dominator", which is what
−0.4277 looks like in prose.

**This is the third time in this file that the first version of a measurement
pointed the wrong way**, and the pattern is worth naming: every one of them was
caught by reading actual replies next to the number that was supposed to
summarise them. The harness is what makes a comparison possible; it is not what
makes it correct.

---

## 6. What this does not do

* **It buys no knowledge.** The `fact` column moves from 0.000 to about 0.07,
  which is a handful of lucky draws out of 28 and not a capability. Nothing here
  can teach a 4.4-million-parameter character model that Paris is the capital of
  France, and the arithmetic probes stay at zero.
* **It buys no memory across turns.** Told a name, the model still cannot repeat
  it one turn later. That is the slow pole's time constant, and no amount of data
  or decoding changes it.
* **It does not make the model reliably on topic.** 0.052 to 0.091 is a 75 %
  relative gain and an 91-in-1000 absolute one. Nine of sixteen test topics still
  land at zero for every arm. *(Corrected 2026-08-08: thirteen per arm, eleven in
  the intersection, and no arm in the project's history scores nine — see the
  appended note in §5 above.)*
* **It cost instruction following.** `chat-v3a-data` reaches the best strict
  `list` score of the round (0.050–0.067); the shipped arm scores **0.000**. A
  mixture that is 40 % stories makes a model that tells stories, and "name three
  animals" is not a story. This is a real regression and it is in the shipped
  model.
* **Reranking cannot invent an on-topic reply.** It selects among the N the model
  would have produced anyway, and on the 2026-08-05 model — where essentially no
  draw mentions the topic — it moved topicality by nothing at all. That null
  result is the cleanest evidence in this file that the failure was upstream of
  the decoder.
* **The two objective changes cannot be separated from each other.**
  `bot_loss_weight` and `align_frac` moved together in every arm that has them,
  so they are reported jointly and nothing here says which one did the work.
* **Every topic probed was taught.** All sixteen topic probes occur as a chosen
  training topic between 72 and 1,518 times per 200,000 stories
  (`experiments/chat/_quality/topic_frequency.json`), so this is "the model
  learned a form it was shown", **not** "the model generalises to unseen topics".
  The probe set was frozen before any arm trained and was not extended
  afterwards, which is why the weakest-taught topic is still in it.
* **n=1 per arm, no repeated seeds.** `EXP_005`'s lesson — that σ is not a
  project constant and must be re-measured on every structurally new arm —
  applies here and was not paid for. Two arms differing by less than a few points
  on a 37-probe, 4-seed battery are "not separated", not ordered. This is why the
  training comparison is made on the 1,024-sample propensity column and not on
  the 64-sample one.

**What would finish the job**, in the order a next session should try it:

1. **More of the dose that worked.** Propensity scaled monotonically with window
   coverage across 4.6 % → 27 % → 76 % and had not flattened. A run **from
   scratch** on the corrected corpus, rather than a 40-minute anneal on top of a
   model trained the old way, is the obvious next arm — the anneal is arguing
   with 76,000 steps of "the request does not matter".
2. **Shorter conversations, not just aligned windows.** `stories_topic` averages
   757 characters against a 256-character window. Packing one story per
   conversation with a cap nearer the window length would make coverage a
   property of the corpus rather than of a sampler flag, and would remove the
   trade in item 1 of §3.2's table.
3. **A mixture ladder for the `list` regression.** The 40 % story weight is the
   suspect; nothing here measured an alternative.
4. **Separate `bot_loss_weight` from `align_frac`**, which would need two arms
   and about 80 minutes.

## 7. Reproducing it

```bash
# the subject-conditioned story source (adds files, changes nothing existing)
python scripts/chat/build_topic_stories.py --limit-chars 400000000

# an arm
python scripts/chat/train.py --run-name chat-v3a-data \
    --init-from experiments/chat/chat-v2-anneal/ckpt_best.pt --no-resume \
    --max-steps 14000 --batch-size 160 --seq-len 256 --lr 5e-4 \
    --budget-minutes 40 \
    --mix stories_topic=0.40 soda=0.20 alpaca=0.15 tinystories=0.09 \
          persona=0.08 dolly=0.06 oasst1=0.02
# ... and the objective arm adds:  --bot-loss-weight 3.0 --align-frac 0.75

# score it, and sweep the decoder over the saved draws
python scripts/chat/quality.py --ckpt experiments/chat/chat-v3a-data/ckpt_best.pt \
    --out experiments/chat/_quality/chat-v3a-data.json

# the tables in this file
python scripts/chat/compare_quality.py experiments/chat/_quality/*.json
```

The probe battery is stochastic in the sampler only: `collect` fixes a seed per
(probe, seed index), so two runs of `quality.py` against the same checkpoint draw
the same candidates. The sweep afterwards is pure arithmetic over a saved file
and can be re-run with `--rescore` at no cost.
