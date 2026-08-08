# A note on topic dose, and the resampler it justifies

**Status: an observation plus a power calculation, not a result.** No arm was
trained, `data/chat/stories_topic.bin` was not rebuilt, and nothing under
`src/snn/`, `experiments/runs/`, `experiments/logs/` or `docs/reports/` was
touched. Two things were built: a measurement of *why* the topic probe
battery's zero/nonzero split looks the way it does, and
`scripts/chat/probe_resample.py`, the generic resampler `QUALITY_v6.md` §7
named as missing. Whether to spend GPU time on the training arm this note
sets up is Elliot's call and needs its own budget decision, the same way
every prior chat round's training has.

---

## 1. What was measured

`docs/chat/README.md` has stood since 2026-08-06 saying "nine of sixteen test
topics land at zero." The obvious question this had never been checked
against: are those nine **absent** from the corpus, or just **rare**?

`snnchat.topics.topic_of` extracts a story's subject by frequency, with a
stop list and a noun test (`src/snnchat/topics.py`). Running it over the
first 200,000 raw TinyStories records (`data/chat_raw/tinystories.txt`,
read-only, no files changed) and counting which subject each story's top
pick is:

- 188,945 of 200,000 stories yielded a subject
- **2,559 distinct subjects**
- the corpus's actual most common subjects — ball, bird, cat, tree, toy,
  park, dog, box, car — **are not in the probe battery at all**. The probe
  battery happens to sample the corpus's long tail.

Every probe topic word *is* present in that tail, at ranks from 11 (`cake`)
to 770 (`pirate`) out of 2,559 and frequencies from 0.36 to 6.74 occurrences
per 1,000 stories — a **19× spread**. Nothing tested is categorically absent
the way "a story about a rabbit" was before the 2026-08-06 corpus fix
(`docs/chat/QUALITY.md` §2); this is a dose question, not a form question.

### 1.1 Confirmed on the actual packed pipeline, not just the raw scan

The number that matters is training **dose** — how often a probe's word
actually appears in a *request* the model is trained to answer — not raw
subject frequency. `scripts/chat/build_topic_stories.py`'s own
`_conversations()` was re-run exactly as it runs when packing
`stories_topic.bin` (`--seed 7`, `RAW_FRACTION 0.15`, stopped at the same
400,000,000-character `--limit-chars` default, 531,802 conversations,
451,880 of them carrying a request turn), counting every probe word that
appears in the *generated request text* (covering all of `story_request`'s
branches — the single-topic form, the two-topic form, and the leftover
single-topic form, not just the primary 55% branch):

| probe | requests naming it | per 1,000 conversations |
| --- | ---: | ---: |
| cake | 2,306 | 4.336 |
| boat | 2,150 | 4.043 |
| frog | 1,547 | 2.909 |
| rabbit | 1,501 | 2.822 |
| train | 1,016 | 1.910 |
| farmer | 514 | 0.967 |
| kite | 477 | 0.897 |
| dragon | 415 | 0.780 |
| robot | 406 | 0.763 |
| spider | 395 | 0.743 |
| moon | 298 | 0.560 |
| bicycle | 243 | 0.457 |
| teacher | 231 | 0.434 |
| snowman | 228 | 0.429 |
| pirate | 144 | 0.271 |

This confirms §1's raw-scan ranking (same relative order, same rough
magnitudes) on the code path that actually produces training data, not an
approximation of it.

## 2. The ground truth on the committed checkpoint, and why it's noisier than it looks

Reading `experiments/chat/_quality/chat-v3d-aligned.json` (the incumbent's
own committed draws, zero GPU — `snnchat.quality.score` is pure arithmetic
over already-generated text) at the pinned `n=1, λ=0` row, per topic probe,
4 seeds each:

| probe | dose (/1k conv) | hit rate (k/4) |
| --- | ---: | :-- |
| little girl named **Mia** | n/a (name form) | 0.75 (3/4) |
| boat | 4.04 | 0.50 (2/4) |
| frog | 2.91 | 0.25 (1/4) |
| cake | 4.34 | 0.00 (0/4) |
| rabbit | 2.82 | 0.00 (0/4) |
| train | 1.91 | 0.00 (0/4) |
| farmer | 0.97 | 0.00 (0/4) |
| kite | 0.90 | 0.00 (0/4) |
| dragon | 0.78 | 0.00 (0/4) |
| robot | 0.76 | 0.00 (0/4) |
| spider | 0.74 | 0.00 (0/4) |
| moon | 0.56 | 0.00 (0/4) |
| bicycle | 0.46 | 0.00 (0/4) |
| teacher | 0.43 | 0.00 (0/4) |
| snowman | 0.43 | 0.00 (0/4) |
| pirate | 0.27 | 0.00 (0/4) |

(pooled `by_kind.topic` = 0.0938, matching `QUALITY_v7.md`'s own table for
this checkpoint exactly — this is a re-read of already-committed numbers,
not a new draw.)

**13 of 16 are literally 0/4, not the "nine" `README.md` has stood on** —
that prose figure was reading an earlier, different arm's table. **This
table is NOT a clean confirmation of the dose hypothesis and it would be
dishonest to present it as one.** `boat`, the second-highest-dose probe,
is the best-scoring word-topic probe (0.50) — consistent. But `cake`, the
*highest*-dose probe, is 0/4, and `rabbit` (4th-highest dose) is also 0/4.
At n=4 draws per probe, Wilson's own interval on 0/4 is **[0, 0.49]** — wide
enough that "0/4" and "1/4" are barely distinguishable, and every zero in
this table could just as easily be a low-probability event that didn't
land in four tries. **No existing measurement in this project has enough
seeds per individual topic probe to tell a real zero from a rare hit
sampled unluckily.** That gap, not a confirmed correlation, is what this
note's second half is for.

## 3. What this might mean (inference, not measurement)

- **The dose spread is real and large (19×) and plausibly explains at least
  part of the zero/nonzero split** — the shape is the same as the one
  intervention in this project's history that resolved by a lot (fixing a
  categorical absence, §1 of `docs/chat/QUALITY.md`), just continuous
  instead of binary. But §2 shows the per-probe evidence for it, as
  currently measured, is not strong enough to act on. `cake` scoring worse
  than `boat` despite higher dose could be genuine (frequency isn't the
  only thing that varies between prompts — word length, its distribution
  within the 220-character eligibility window in `topic_of`, its
  collision rate with the stop list) or could be four coin flips. Nothing
  here distinguishes those.
- **`Mia` (0.75) and the name-form generally remain the strongest signal
  in the battery**, as they have been every round since `QUALITY.md`
  first split character-name requests from common-noun ones. Names are
  both far more frequent (every story has one) and reached by a separate,
  well-populated branch of `story_request`.
- **Given the noise at n=4, the honest next step is not "train an arm and
  see" — training compounds a hyperparameter change with the exact same
  measurement-resolution problem `story_dodge_resample.py` was built to
  fix for a different metric.** Any future round that trains a
  frequency-flattened `stories_topic` arm needs to read the result at a
  seed count that can actually tell "still zero" from "meaningfully
  nonzero" apart — which is what §5 sizes.

## 4. What it would take to establish any of it

The design this note sets up, for a future round with its own GPU-budget
decision (not run this session):

1. Rebuild `stories_topic.bin` with a per-topic sampling weight (e.g.
   `1 / count(topic)^0.5` — partial flattening, not full inversion) so
   `pirate` (0.27/1k) gets a dose closer to `cake` (4.34/1k) without letting
   a topic that occurs twice dominate the corpus.
2. Train one anneal arm, matched to `chat-v3d-aligned`'s exact recipe
   (`chat-v2-anneal` init, `bot_loss_weight 3.0`, `align_frac 0.75`,
   `align_lookahead 1024`, 14,000 steps), seeds 0–3 — reusing the
   incumbent's **already-trained** four seeds rather than retraining them.
3. Compare with §5's tool, at the seed count §5 recommends, not at the
   standard 4-seed `quality.py` battery's resolution — §2 is the
   demonstration of why that resolution isn't enough for this question.
4. Guardrail on the full `quality.py` battery (list/social/identity/
   fallback) the way every prior round has, so a topic-dosing win that
   costs something else is visible rather than laundered.

Pre-register the direction and the possibility that it does nothing or
hurts, the same discipline `WEIGHT_DECAY_NOTE.md` §4 asked of its own
follow-up experiment. There is a real chance frequency isn't the load-bearing
variable — e.g. reply length, or how early a topic's first mention falls
within `topic_of`'s 220-character window (§ of `topics.py`), could matter
as much or more, and this note has not ruled those out.

## 5. The tool this round built, and its power calculation

`scripts/chat/probe_resample.py` generalises
`scripts/chat/story_dodge_resample.py`'s pattern (Wilson intervals, the
superset-reproduction gate against committed draws, a fixed seed count
decided before any number is read) from the hard-coded `list`+`fact` kind
pair to **any** `--kind` and/or `--probes` subset. Verified this session
against the committed `chat-v3d-aligned.json`: a 2-probe, 4-seed resample of
`dragon`/`robot` reproduced those checkpoints' committed draws character for
character (`superset check: 8 committed draws reproduced identical`), and
the fractional-score guard correctly refuses `--kind list` (`want > 1`
probes, which need a sample-SE report per `CONVENTIONS.md` §2, not Wilson).

### Power calculation, for the nine lowest-dose probes (`pirate`, `snowman`,
`teacher`, `bicycle`, `moon`, `spider`, `robot`, `dragon`, `kite` — all
under 1/1,000 conversations by §1.1's table)

Wilson upper bound at `k = 0` (the "still indistinguishable from zero"
case), computed by `snnchat.quality.rate_ci`:

| seeds/probe (S) | pooled draws (9S) | pooled lattice | pooled CI @ k=0 | per-probe CI @ k=0 | ≈ GPU time/arm |
| ---: | ---: | ---: | --- | --- | ---: |
| 16 | 144 | 0.00694 | [0, 0.0260] | [0, 0.1936] | 3.1 min |
| 65 | 585 | 0.00171 | [0, 0.0065] | [0, 0.0558] | 12.7 min |
| **101** | **909** | **0.00110** | **[0, 0.0042]** | **[0, 0.0366]** | **19.7 min** |
| 201 | 1,809 | 0.00055 | [0, 0.0021] | [0, 0.0188] | 39.2 min |
| 301 | 2,709 | 0.00037 | [0, 0.0014] | [0, 0.0126] | 58.7 min |

(GPU time uses `story_dodge_resample.py`'s own measured ~1.3 s/draw.)

**Recommendation: S = 101 seeds/probe**, the same count this project has
already used for arms "not being tested against a threshold, only reported
alongside" (`CONVENTIONS.md` §5). At S=101 a pooled null reads with an
upper bound of 0.0042 — comfortably below every effect size this project
has ever established as real, including the whole-battery pool figure of
0.0908 (`by_kind`/`pool_by_kind` on `chat-v3d-aligned`, §2 above) — so a
future "still zero" result at this resolution is a real answer, not a
resolution failure. Per-probe resolution at S=101 (CI half-width ≈0.037) is
adequate to see whether an individual long-tail probe moved into the
range `frog`/`boat` already occupy (0.25/0.50 at n=4), though not tight
enough to finely rank the nine against each other — S=301 would tighten
per-probe resolution to ±0.013 at roughly 3× the cost, and is available if
a future round wants that.

**This recommendation is not a pre-registration.** The actual bar (if any)
and the exact probe set belong in that future round's own
`PREDICTION_v8.md`, written before the rebuilt corpus trains anything, per
`CONTRIBUTING.md` §2. `probe_resample.py` prints its own lattice warning
once a real `--threshold` is supplied, so whoever writes that
pre-registration doesn't have to re-derive the divisibility check by hand.

## 6. What was not done

No training, no corpus rebuild, no `experiments/chat/SHIPPED` change. Nothing
under `src/snn/`, `experiments/runs/`, `experiments/logs/` or `docs/reports/`
was read for anything but confirming which functions to reuse, and none of
it was modified. `data/chat/stories_topic.bin` was read implicitly through
`build_topic_stories.py`'s own construction logic (§1.1), not rebuilt.
