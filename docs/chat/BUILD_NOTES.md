# Build notes: how the chat model was chosen and trained

A record of the decisions taken while building `snnchat`, written so that the
ones that were guesses are identifiable as guesses. **None of this is part of the
research protocol.** No number here is pre-registered, none is a reported figure,
and the selection experiment below is n=1 per arm with no error bars.

---

## 1. What was measured before anything was trained

Throughput on this box (RTX 5060, 8 GiB) at several sizes, `fwd+bwd+AdamW` inside
a captured CUDA graph:

| d | K | B | L | params | ms/step | Mchar/s | peak VRAM |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 768 | 4 | 64 | 384 | 2.53 M | 79.2 | 0.31 | 1.80 |
| 1024 | 4 | 64 | 384 | 4.42 M | 116.5 | 0.21 | 2.44 |
| 1024 | 6 | 48 | 384 | 6.52 M | 138.9 | 0.13 | 2.56 |
| 1536 | 4 | 32 | 384 | 9.77 M | 117.2 | 0.10 | 2.12 |
| 1024 | 4 | 32 | 768 | 4.42 M | 114.2 | 0.22 | 2.63 |
| 768 | 3 | 256 | 256 | 1.93 M | 132.5 | 0.49 | 3.82 |
| 1024 | 4 | 160 | 256 | 4.42 M | 163.7 | 0.25 | 3.96 |
| 1024 | 2 | 320 | 256 | 2.31 M | 171.9 | 0.48 | 5.15 |
| 1536 | 3 | 128 | 256 | 7.41 M | 200.0 | 0.16 | 4.16 |

The row that mattered: **at d=768, K=4, tripling the batch from 64 to 192 cost
only 1.63× the time.** The per-timestep scan kernel is latency-bound at these
widths — `B·d = 49k` elements is nowhere near saturating the device — so batch
size is close to free and sequence length is close to irrelevant to throughput
(`chars/s ≈ B / (K · t_kernel)`, with `L` cancelling). Every subsequent
configuration uses a large batch and a short-ish window for that reason.

This also says something about the research side's cost model that is worth
recording: `01_reconnaissance.md` measured ~14.5 µs of CPU-side launch cost per
kernel, and under a captured graph that cost is gone — but the measured
per-timestep cost here is ~26 µs and does **not** fall with graph capture. The
scan is bound by something other than launch overhead at these shapes. It was not
chased further, because nothing in this deliverable depends on knowing what.

**Addendum 2026-08-12: this paragraph is right, and it was read too widely.**
Everything above was measured on the TRAINING step, where the scan loops over
L = 384 timesteps inside one captured graph. Decoding calls the same scan with
L = 1 and, until v10, captured nothing at all -- so the four jiterator launches
per character had nothing to amortise against and the REPL was paying ~1580 us
per character where the arithmetic is ~50. Capturing the decode step brings it
to ~205 us, i.e. ~51 us per timestep per layer, which is the same order as the
26 us floor above: the floor is real, and the decoder simply was not standing on
it. See `QUALITY_v10.md` §1 and `src/snnchat/stepper.py`.

## 2. Choosing the size

`scripts/chat/size_probe.py`, four arms, five minutes each, constant learning
rate, strictly sequential, nothing else on the GPU. The question is not "which
model is best" but "which is best **per hour**", which is a different ordering.

| arm | params | steps in 5 min | chars seen | weighted val bpc |
| --- | --- | --- | --- | --- |
| `small_K3` d=768, K=3, B=256 | 1.93 M | 2,140 | 140 M | **1.5808** |
| `mid_K4` d=1024, K=4, B=160 | 4.42 M | 1,663 | 68 M | 1.6048 |
| `large_K3` d=1536, K=3, B=128 | 7.41 M | 1,386 | 45 M | *abandoned, see below* |
| `mid_K4_bf` d=1024, K=4, bf16 | 4.42 M | — | — | *not run* |

**Two arms did not finish, and the reason is itself a result.** `large_K3` hit
its step budget normally and then sat in a 30-second evaluation for eighteen
minutes. Its training footprint is 4.16 GiB and the evaluation ran at the
*training* batch size alongside a still-resident captured graph; on an 8 GiB card
that spills to host memory, and under Windows' WDDM a spill does not fail, it
just becomes ~50× slower. It was killed, and `eval_batch_size` was added as a
separate config field so the real run cannot hit the same wall. The bf16 arm was
dropped with it — by then the ranking question had been answered and the
remaining budget was better spent training.

**The choice: `mid_K4` (d=1024, K=4, B=160, 4.42 M parameters).** Not the arm
that won the probe, and the reasoning is the probe's known weakness. Five minutes
is 1/55th of the real run and small models always look better at small budgets;
what the table actually shows is `mid_K4` sitting **0.024 bpc behind while having
seen less than half the data**. Per character it is well ahead, and the real run
is long enough for that to be the term that matters. `large_K3` might have been
better still, and there is no measurement here that rules it out — it was passed
over for the VRAM headroom, not for its numbers.

`experiments/chat/_probe/` holds the two arms that completed. No
`results.json` was written: the script writes it after the last arm, and the last
arm was killed.

## 3. What is a guess

* The **mixture weights** (`snnchat.build_corpus.DEFAULT_MIX`). Nothing was
  measured. `persona` at 0.10 is the aggressive one: ~5 MB of authored text
  against ~1.4 G of corpus, so the model sees it hundreds of times and will
  effectively memorise it. For the twenty-odd intents it covers that is the
  desired behaviour; the cost is a pull on the model's general register toward
  short apologetic sentences.
* The **slow-pole spread** (`snnchat.model.spread_slow_poles`). Motivated by
  `EXP_006`, untested against the committed uniform initialisation. See §5 of
  `docs/chat/README.md`.
* The **400-character turn limit** and dropping rather than truncating. The
  reasoning is sound (a truncated turn teaches the model to stop mid-sentence);
  the specific number is not measured.
* **Anything that would have needed a control arm to establish.** There are no
  control arms anywhere in this package.

## 4. What is not a guess

* The chat model is `snn.model.TwoCompThresholdCharLM` and nothing else. Asserted
  by `tests/test_snnchat.py::test_chat_model_is_a_research_arm_unchanged`, which
  loads a chat `state_dict` into a research-built model and compares logits.
* `batch(step)` is a pure function of `(seed, step)`, so the run is restartable
  without a data-order desynchronisation. Asserted by
  `test_batch_is_a_pure_function_of_seed_and_step`.
* A user cannot forge a turn marker, and neither can the model. Asserted by
  `test_encode_cannot_forge_a_turn_marker`.
* The horizon probe reproduces an independently-computed bpc at `k = L`, where
  its reshape is the identity — `scripts/chat/horizon.py` fails hard if it does
  not, rather than reporting a number nothing checked.

## 5. Two bugs the tests caught

Recorded because both were invisible in output and would have degraded every
reply slightly:

1. **The last generated character was never fed back into the membrane.**
   `stream` yielded the state from *before* consuming each character, so a reply
   ended one character short of its own state. Symptom: none, until a session's
   state was compared against a replay of its own transcript
   (`test_rewind_reproduces_the_state_of_a_fresh_replay`).
2. **An empty reply dropped the user's turn.** The prompt was fed inside the
   generation loop, so when the model closed its turn immediately the loop body
   never ran and the user's message never reached the membrane — which looks
   like the model ignoring you, not like a bug
   (`test_an_empty_reply_still_feeds_the_user_turn`).

## 6. `CUDA_VISIBLE_DEVICES=""` does not hide the GPU on this box

Found by accident and worth carrying over to the research side, because it
defeats the obvious way of enforcing a rule the protocol already has.

`03_phase3_candidates.md` §8 requires that nothing else touch the GPU while a
timed run is in flight. The natural way to guarantee that for a test suite is to
run it with `CUDA_VISIBLE_DEVICES=""`. **On this machine that does nothing:**

```
CUDA_VISIBLE_DEVICES=""    ->  torch.cuda.is_available() == True
CUDA_VISIBLE_DEVICES="-1"  ->  torch.cuda.is_available() == False
unset                      ->  torch.cuda.is_available() == True
```

The empty string is evidently dropped rather than treated as "no devices", so a
suite launched that way runs on the card at full speed alongside whatever else is
there.

How it surfaced: `tests/test_kernel_count.py::test_cost_model_predicts_eager_wall_clock`
failed during the chat training run, measuring **119.9 µs per kernel against a
5–40 µs band**. It is a wall-clock test whose tensors are hardcoded to
`device="cuda"`, so it had been running on the busy GPU the whole time. 336 of
337 tests passed; that one failure is contention, not breakage — nothing under
`src/snn/` was modified by this work at all.

The practical consequence for anything timed: **use `-1`, not `""`**. The
consequence here: this run's wall-clock figures are mildly contaminated by a
couple of minutes of test-suite and CPU-demo traffic. Recorded rather than
corrected, because no wall-clock number in this package is a reported figure —
had this been an experiment, `EXP_013`'s standard applies and the affected timing
would have been thrown away and re-measured.

## 7. An observation for the open Phase-4 question, offered as a pointer only

At **step 14,500** of the chat run the global gradient norm was
**1,729,672.38** — about six orders of magnitude above the run's median of
0.288 (max of the other 30 logged steps: 2.204). The loss stayed finite
(bpc 1.3504), `clip_grad_norm_` absorbed it, and the next logged step was back to
0.20 with every source's validation bpc improving.

This is the *shape* of `EXP_009`'s failure mode — a finite but enormous gradient,
not an overflow in the forward pass — appearing in the same neuron family
(`twocomp_threshold`) at **4.42 M parameters, six times the scale Phase 4 ran it
at**, on a completely different corpus. `EXP_009`'s was 5.46e31, large enough that
the fp32 norm overflowed to `inf` inside the clip and produced the NaN. 1.7e6 is
nowhere near that, so this one cost nothing.

**It is not evidence.** n=1, no control arm, no matched seed, a different corpus,
a different vocabulary and a different width; the gradient was logged only because
step 14,500 happened to fall on the 500-step logging cadence, so the true
frequency of these spikes is unmeasured and could be far higher. It is recorded
because the report's open question is *which arithmetic in layer 0's backward
produces the NaN*, and "the spike shape reproduces at 6× scale on unrelated data"
is a cheap pointer for whoever chases that — not an answer, and not a result.

## 8. Divergence handling

`ChatTrainer._recover_from_divergence` rolls back to `ckpt_last.pt`, bumps the
sampler seed and halves the learning rate, up to five times. The seed bump is the
load-bearing part: `batch(step)` is pure, so a plain restart replays the batch
that killed the run and dies at the same step.

This is a **recovery, not a diagnosis**, and it is only defensible because this is
not an experiment. Phase 4 has two unexplained divergences in this neuron family
(`EXP_009`'s fp32 overflow inside the clip, and `compose_s0` at step 17,598 with
gradients of 0.03 and everything finite), and the research protocol is explicit
that the correct response to a NaN is to chase it. Here there is no hypothesis to
protect, so the run carries on — loudly, with the event in `log.jsonl`.

---

## 9. The responsiveness round (2026-08-06)

A second pass over the same model, aimed at one thing: the gap
`docs/chat/RESULTS.md` named as "fluent without being responsive". Full write-up
and every number in `docs/chat/QUALITY.md`; this section records only what was
measured, what was guessed, and what broke.

### Measured

* **The failure is in the corpus, not in the model.** Topic-conditioned
  requests were answered on topic 7.8 % of the time, but split by probe the
  failure is binary: a request naming a *character* landed 50 %, a request
  naming a *common noun* landed 0 %. `build_corpus._story_conversation` builds
  topic requests with `_MIDSENTENCE_CAP`, which in TinyStories finds the
  character's name — so "a story about a rabbit" answered by a story about a
  rabbit appears nowhere in the training data. The model was measured on a
  capability it was never shown.
* **Prompt dependence is a usable metric and needs no sampler.** The same
  held-out reply scored behind its own user turn and behind another
  conversation's differs by a measurable number of bits per character. It has no
  seed, no lexicon and no probe set in it.
* **The batched masked scorer agrees with the one-character-at-a-time path to a
  relative 3e-08**, and the guard that checks it has been mutation-tested.
* **The round has a control arm**, which no other part of this package does, and
  **it came back flat.** `chat-v3c-control` is the same base checkpoint trained
  for the same 14,000 steps at the same learning rate on the *old* mixture. Its
  topic propensity is **0.0518 against the baseline's 0.0518** — 53 hits in 1,024
  both times — and its prompt dependence is +0.0464 against +0.0474. Fourteen
  thousand further steps buy no responsiveness at all when they are spent on the
  old corpus, which is what turns the rest of `docs/chat/QUALITY.md` §4.3 from a
  trend into an attribution.
* **Every topic probed occurs in the training data** between 72 and 1,518 times
  per 200,000 stories (`experiments/chat/_quality/topic_frequency.json`), so the
  result is about learning a form that was taught and says nothing about
  generalising to unseen topics.

### Guessed

* **The mixture weights for the new source.** `stories_topic` at 0.40 against
  `tinystories` at 0.09 is a judgement; no ladder was run.
* **The learning rate (5e-4) and the 14,000-step length** were chosen to fit a
  wall-clock budget. What is *not* a guess is that they are the same in every
  arm, including the control — so they cannot be the reason one arm beat
  another.
* **`bot_loss_weight = 3.0` and `align_frac = 0.75`.** Both plausible, neither
  tuned. They were varied together in one arm, so the two cannot be told apart
  from each other — only from the data change.
* **`lambda` and `N` for reranking** are *not* guesses — they were swept — but
  the sweep is on one probe set at four seeds, and the probe set is a judgement.
* **The stop list in `snnchat.topics`.** Hand-written, and it is the one place
  where an author's idea of what a story is about enters 400 MB of training
  data.

### What broke

1. **The mean-per-character rerank score is won by the empty reply.** `<|eot|>`
   alone is a very probable "reply" and scores beautifully per character. Caught
   in the first three prompts of the first draw, before any training; the guard
   is `RerankParams.min_chars`, applied as a partition rather than a penalty.
2. **A method fell out of its class.** `MixtureSampler.mix_report` ended up
   nested inside a module-level function after an edit and became silently
   unreachable — the run died at its first log line rather than at import, which
   is the noisy failure and the lucky one.
3. **Two training runs shared the card for fourteen minutes.** A queued control
   arm was supposed to be cancelled; the kill hit the wrong PID (a monitor's
   `tail`, not the script's shell), the script survived, and it launched its run
   six seconds after the arm it was meant to yield to. Each then ran at **1.9
   steps/s against 5.95** — the tell, and the only reason it was noticed, was
   that the slower arm's step rate did not recover when every other job was
   stopped.

   Nothing about the optimisation was corrupted: `batch(step)` is a pure
   function of `(seed, step)`, so both runs took exactly the steps they would
   have taken alone. What was corrupted was the **budget**: a 40-minute
   wall-clock budget at a third of the speed buys a third of the steps, and a
   shorter arm is a confounded arm. Both were deleted and rerun rather than
   explained away, which is `EXP_013`'s standard for a self-inflicted overlap
   applied to a self-inflicted overlap.

   The lesson worth carrying: **verify a cancellation, do not assume it.** `ps`
   showed the surviving shell the whole time.

4. **The loss weight was one position out.** The first version weighted the
   `<|bot|>` marker (trivial, always follows the user's `<|eot|>`) and *not* the
   `<|eot|>` that closes the reply — that is, it dropped the model's own
   decision to stop from the thing being upweighted. Found by printing the
   weight next to the character it applies to, which is the only way this class
   of bug is visible.

## 10. The short-conversation round (2026-08-06, later the same day)

Aimed at items 1–3 of `docs/chat/QUALITY.md` §6's "what would finish the job".
Full write-up in `docs/chat/QUALITY_v4.md`; the prediction it is scored against
is in `docs/chat/PREDICTION_v4.md` and was written before either arm trained.

### Measured

* **The metric the previous round ordered its arms by was two different
  quantities.** §4.1's 4.6 % counts only windows that *begin inside* the request,
  which excludes every window starting in the previous conversation and rolling
  forward into this one — an ordinary window that does carry the dependency. The
  corrected figure for the same source is **0.248**. Neither version measures the
  **dose**: the share of *reply characters* whose gradient is conditioned on
  their own request, which for `stories_topic` at the shipped alignment is
  **0.260**. `scripts/chat/window_coverage.py` computes all three, and reports
  the old one alongside so the two rounds stay comparable.
* **Shortening conversations raises the dose 3.4×, to 0.877**, and does most of
  it without the sampler: `stories_short` *unaligned* (0.531) beats
  `stories_topic` *aligned* (0.304). That is QUALITY.md §6 item 2's claim —
  coverage becomes a property of the corpus — measured rather than asserted.
* **The realised alignment rate is identical across all three mixtures**
  (0.741 / 0.743 / 0.744 over 1,920 windows each), so the dose comparison is at
  matched alignment and the difference is entirely the corpus.
* **Choosing a story's topic from its truncation does not work; choosing it from
  the full story and requiring it to survive does.** 32 % against **88.9 %**,
  measured on 20,000 stories. 161 characters is rarely enough to see a word
  twice, which is what `topic_of` needs in order to know it is the subject.
* **`stories_short` yields 1,560 distinct noun topics over 30,000 requests.** The
  topic vocabulary is the corpus's, not an author's list.
* **The dose lever is exhausted.** 3.4x the dose moved topic propensity
  0.0908 -> 0.0723 and prompt dependence +0.0619 -> +0.0572, i.e. backwards on
  both. QUALITY.md 6.1 is answered "no" and no further GPU should go on window
  coverage.
* **The two new arms are a matched-length pair, which is the round's luckiest
  fact.** They draw candidates averaging 166 and 167 characters -- one character
  apart over 1,024 draws -- so between them the reply-length confound on `topic`
  and on `list (strict)` is closed by construction. That is what makes the
  mixture result attributable: strict `list` 0.033 -> 0.083 at matched length.
* **Instruction weight trades against social register.** 0.23 -> 0.36 bought
  `list (strict)` 0.033 -> 0.083 and cost `social` 1.000 -> 0.900 (pool 0.9812 ->
  0.9094), at matched reply length. Two points, not a curve.
* **The story-weight explanation for `story_dodge` is eliminated**, not merely
  unsupported: a third off the story weight moved it 0.167 -> 0.146.

### Guessed

* **The truncation target**, uniform on [140, 235] characters. Sampled rather
  than fixed so the model does not learn "a story is exactly four sentences", but
  the range itself is a judgement and no ladder was run.
* **That a truncated story is a better thing to teach than a spliced one.**
  Splicing the original final sentence back on would restore an ending;
  TinyStories endings refer to objects introduced in the middle, so splicing
  manufactures non-sequiturs. Judged from reading samples, not measured.
* **That ~50 % of short-form requests should say "short"**, which is what makes
  reply length promptable. Nothing measured whether the model picks it up.
* **`chat-v4b-balance`'s mixture.** It moves story weight down and instruction
  weight up *together*, so a result on it is attributable to the pair. There was
  not GPU budget for a four-arm ladder and that is the honest reason.
* **The temperature spread of 0.3.** One value, tested against zero on three
  checkpoints. Not swept.

### What broke

1. **The first version of the candidate-temperature ladder was ascending, which
   would have quietly corrupted every row of the reranking sweep.**
   `quality.score` compares settings by taking the *first n* of a larger pool —
   sound only because the rows are exchangeable. An ascending ladder makes the
   first 4 of 16 the four coldest, so the `n = 4` row would have been a cold pool
   and the `n = 16` row a full-range one, and the table would have reported the
   difference as an effect of pool size. Fixed by emitting the ladder in van der
   Corput order; the guard is
   `test_every_prefix_of_the_ladder_spans_the_range`, and its first version
   asserted the wrong property (that a prefix of 2 spans the full range, which no
   low-discrepancy sequence does) and had to be restated as the property actually
   needed: centred prefixes at the lengths the sweep uses.
2. **`spread = 0` had to be made mean the scalar code path, not a ladder of equal
   values.** Dividing by a `[n, 1]` tensor of 0.85 is not bit-identical to
   dividing by the Python float 0.85, and every arm drawn before this knob
   existed is compared against arms drawn after it.

## 11. The reply-length round (2026-08-06, that evening)

One arm, `chat-v5a-short300`, aimed at the single thing `QUALITY_v4.md` left open: every result in §10 was measured on arms whose replies averaged 136 characters against the shipped arm's 189, and both `topic` and `list (strict)` depend on reply length. Pre-registration in `docs/chat/PREDICTION_v5.md`, write-up in `docs/chat/QUALITY_v5.md`. It closed the confound, lost the headline, and shipped nothing. Its most durable output is not the arm but the defect it exposed in `story_dodge`.

### Measured

* **The truncation target [242, 407] was fixed by measurement, before the prediction was written.** Six candidate ranges over 20,000 stories, kept-length means 268.7, 279.5, 289.2, **298.5**, 308.8, 323.0 — [242, 407] misses 300 by 1.5 against the next-nearest miss of 8.8 (`experiments/chat/_quality/truncation_calibration.json`). The measurement reads `data/chat_raw/tinystories.txt` and nothing else: no model, no checkpoint, no held-out split, so there is no outcome in it to have peeked at. Frozen as `snnchat.shortform.SHORT300` (`src/snnchat/shortform.py:148`) and gated by `tests/test_snnchat.py::test_the_long_target_is_pinned_to_what_was_pre_registered`.
* **Targeting the kept-length *mean* rather than the sampled midpoint is what made the length match happen.** `short_story` stops at the last sentence boundary *before* its sampled target, so kept length runs under it: `kept_over_midpoint` is 0.858 at `SHORT` and 0.920 at `SHORT300`. Aiming the midpoint at 300 would have overshot by 25–40 characters. The `SHORT` row of the same file reproduces the figures `snnchat.shortform`'s docstring committed before `chat-v4a-short` ever trained (160.8 mean, 161 median, 0.8882 topic survival), which is what makes the new row believable rather than merely computed.
* **The cost was priced before the run, not discovered after it.** `conv_fits_256` falls 0.9647 → 0.0545 and the reply-character dose falls 0.877 → 0.623 (`_quality/window_coverage_v5.json`, `_quality/truncation_calibration.json`), both measured before training started. The round accepted the loss because §10 had already closed the dose as a lever, and was vindicated: pool topic propensity 0.0957 at dose 0.623 beats `chat-v4a-short`'s 0.0723 at dose 0.877.
* **The topic-survival rise was disclosed rather than controlled, because it favours the arm on the metric being scored.** 0.8882 → 0.9466 topic survival is a 6.6 % relative improvement in the training signal for `topic`, and it is not separable from the manipulation — a longer truncation *is* more surviving topics. Survival is flat at 0.9459–0.9466 across all five long candidates, so this is a property of "longer than `SHORT`", not of [242, 407].
* **Story-request density is a third confound, named before the run.** `conversations / chars_train` from `data/chat/manifest.json`: `stories_short` 1,012,158 / 242,868,982 = 4,168 per M characters; `stories_short300` 869,777 / 318,720,158 = 2,729; `stories_topic` 528,616 / 398,400,031 = 1,327. The ladder is monotone in reply length (161 → 299 → 726), dose (0.877 → 0.623 → 0.260) and request density at once, so any monotone result is attributable to the **set**, not to length. This confound was never named in `QUALITY_v4.md` and applies retroactively to §10's comparison.
* **The "one variable changed" claim is literally true, checked by diffing configs rather than reading prose.** `experiments/chat/chat-v5a-short300/config.json` and `chat-v4a-short/config.json` differ in exactly two keys, `mix` (`stories_short` → `stories_short300`, both at 0.40) and `run_name`. Everything else is identical: 14,000 steps, B=160, L=256, lr 5e-4 cosine to `lr_final_frac` 0.05, warmup 500, `bot_loss_weight` 3.0, `align_frac` 0.75, `align_lookahead` 1024, weight_decay 0.1, grad_clip 1.0, fp32, seed 0, d=1024, K=4, 4,417,637 params.
* **The deliverable landed: reply length matched.** Mean selected reply 188.3 characters against `chat-v3d-aligned`'s 188.9, measured reply length in the corpus 298.8 against the 298.5 calibrated for. The band table then splits — 300+ band 0.102 (716 candidates) to 0.087 (817) in the new arm's favour, 200–300 band 0.081 (307) to 0.106 (207) against it.
* **The 300+ band was declared a point mass at the decoder ceiling before the numbers existed** (`PREDICTION_v5.md` §7 item 2), and it is: every one of the 716 candidates in that band is exactly 300 characters with `closed` unset, and so are all 817 of the shipped arm's. The meaningful cell is the 200–300 band, which both models end themselves and which the arm loses. Which band should dominate was **not** pre-registered, so that reading is a judgement made after the split appeared.
* **Pinning the reading row to `n = 1, λ = 0` before the checkpoint was scored is the amendment that mattered, and a committed number justifies it.** Across the 19 rows of `DEFAULT_GRID`, `chat-v5a-short300`'s `story_dodge` runs 0.0 to 0.375 and its headline 0.2241 to 0.3053; `chat-v4a-short`'s `story_dodge` runs 0.0208 to 0.3958 on the very checkpoint that anchors P1's bands. Left unpinned, the row choice and not the model would have decided P1. The commit carrying the pin (`1c3e348`, 18:10) precedes the commit carrying the quality JSON (`fc607ad`, 18:30), and `PREDICTION_v5.md` has no commits after 18:10.
* **The length-independent `_is_story` opener branch was declared in advance so it could not be promoted afterwards.** Opener rate 0.1042 (5/48) for `chat-v5a-short300`, 0.1667 (8/48) for `chat-v4a-short`, 0.1250 (6/48) for `chat-v4b-balance`, 0.0417 (2/48) for `chat-v3d-aligned` (`QUALITY_v5.md` §4.4). Raising reply length removed ~37 % of the short-form corpus's excess story-dodging on the branch length arithmetic cannot move, and left it at 2.5× the shipped arm's. Length is part of it; length is not most of it.
* **The headline stayed as pre-registered and the arm did not ship.** 0.2779 vs 0.2969, recomputable to the digit: 0.5(0.1094) + 0.3(0.0833) + 0.2(1.0) − 0.2(0.0089) = 0.27791. The arm beats the shipped one on the dominant term (topic 0.1094 vs 0.0938) and loses entirely on `list` (0.0833 vs 0.1667). `experiments/chat/SHIPPED` was updated with a note explaining why it did *not* move.
* **The prompt-dependence rule was never tested, because the arm lost on it.** Delta +0.05425 against `chat-v3d-aligned`'s +0.06195 and `chat-v4a-short`'s +0.05717 (soda 0.04233, alpaca 0.05796, dolly 0.08684). §10's finding that the short corpus extracts more from a request does not survive raising the truncation length — and it was the one measurement on which that corpus was unambiguously ahead. The batched-vs-sequential scorer guard held at relative 8.996e-08.
* **The run itself was uneventful and the wall-clock budget never fired.** 2,496.1 s = 41.6 minutes against a 55-minute ceiling, so this is a completed cosine and not a stopped one. Peak VRAM 3.831 GiB. Max logged gradient norm 0.362 at step 500 against a median of 0.244 — nothing resembling §7's 1.7e6 spike, no `divergence` event. Weighted val bpc fell monotonically 1.19371 → 1.12179 across all seven evals, so `ckpt_best.pt` is simply step 14,000.

### Guessed

* **P1's threshold, `story_dodge` below 0.125.** Chosen as the midpoint of `chat-v4a-short`'s 0.167 and `chat-v3d-aligned`'s 0.083, which is the right instinct applied to the wrong quantity. `story_dodge` is scored over 12 probes × 4 seeds = 48 draws, quantised at 1/48 = 0.02083, and **0.125 is exactly 6/48**. On that lattice the arm could not miss narrowly; it could only hit exactly, and it returned exactly 6/48. Reported UNRESOLVED under the pre-registration's own middle band rather than nudged. What would have tested it: stating the denominator and lattice next to the threshold before running. A threshold at 0.135 resolves cleanly on the identical draws.
* **Keeping `--max-new 300` for an arm trained to write ~298-character replies.** Deliberate and reasoned — raising it makes this arm's draws incomparable to every arm it is measured against — but never tested. The predicted cost arrived: `closed_rate` 0.9392 → 0.5676, i.e. 43 % of selected replies never emitted `<|eot|>`. `closed` was disqualified as a quality metric at this reply length **in advance**, so it can be read neither as a loss for this arm nor as a win for `chat-v4a-short`. What would test it: draw a second pool at `--max-new 600` for both arms and see whether the 300+ band's ordering survives decensoring.
* **Not repacking `stories_short300` at a higher character limit.** It hit the 320 M pack limit (`chars_total` 320,000,158) while `stories_short` exhausted `tinystories.txt` below it (243,844,359), so the new source is built from roughly the first 86 % of the file — 869,777 conversations against 1,012,158. Repacking and retraining was priced at ~45 minutes and not done; the confound is disclosed and judged small on the grounds that TinyStories is not ordered by any property this round measures. **That grounds statement is itself untested.** Found by the pre-run audit, not by the round's own design.
* **Skipping the `--temperature-spread` companion run every §10 arm has.** Deliberate — §10 recorded the spread as measured and dead, and re-running a dead intervention to fill a column spends GPU on table symmetry. Confirmed by absence: there is no `chat-v5a-short300.spread.json`. The cost is that this arm's numbers are one draw where §10's arms have two, and nothing checks that spread = 0.3 is still inert at 300-character replies. For the record, the `chat-v4a-short` spread run moved `story_dodge` 0.1667 → 0.2083 and `mean_chars` 136.4 → 124.0, so "dead" was never quite zero.

### Inherited

New category from this round on, and it earns its own heading: **carried from an earlier round without re-checking at this round's settings.** Not a guess (someone measured it once) and not a measurement (nobody measured it *here*).

* **The whole training recipe, unchanged from `chat-v4a-short`.** lr 5e-4 cosine to a 5 % floor, warmup 500, B=160, L=256, 14,000 steps, `bot_loss_weight` 3.0, `align_frac` 0.75, `align_lookahead` 1024, weight_decay 0.1, grad_clip 1.0, fp32, `deterministic` false, seed 0, initialised from `chat-v2-anneal/ckpt_best.pt` (step 7,874, val 1.17947). None was re-tuned at 300-character replies. This is the correct choice for the one-variable comparison and simultaneously the reason nothing here says whether 5e-4 is right for a corpus with 1.85× the conversation length. `bot_loss_weight` and `align_frac` remain §9's untuned judgements, now two rounds old.
* **Measuring the dose at `align_frac` 0.74 while training at 0.75.** `window_coverage.py` was run with `--align-frac 0.74` because 0.74 is the *realised* rate §10 measured (0.741/0.743/0.744 over 1,920 windows per mixture), and `window_coverage_v5.json` records `align_frac_realised: 0.74`. The trainer's config requests 0.75. The gap was never re-verified at this round's conversation length — a 337-character conversation gives the sampler a different set of boundaries to snap to than a 241-character one, so the realised rate is an assumption here, not a measurement.
* **Comparing against `chat-v3d-aligned` as if it were a matched arm.** It is not, and the difference is named nowhere in the round. Diffing configs: `chat-v3d-aligned` used `warmup_steps` 200 against this arm's 500, `eval_every` 3500 against 2000, `eval_batches` 30 against 40, `log_every` 250 against 100. The warmup difference is probably immaterial; the eval cadence is not obviously so, because `ckpt_best.pt` is selected on weighted val bpc, so the shipped arm's checkpoint was picked from 4 candidate evaluations and this arm's from 7.
* **Selecting the scored checkpoint on weighted validation bpc, when the weighted mean's *composition differs between the arms being compared*.** Each mixture contains its own story source at 0.40: `stories_topic` 1.0452 for the shipped arm, `stories_short` 0.8052 for `chat-v4a-short`, `stories_short300` 0.9098 for this one, giving weighted means of 1.1743 / 1.0801 / 1.1218. Those three numbers are not comparable to each other and the round never quotes them, which is right — but the checkpoint each arm is judged on was chosen by an arm-specific criterion, and that is not stated anywhere. Here it costs nothing: val bpc fell monotonically in all three, so `ckpt_best` is the last step.
* **The frozen 37-probe battery, 4 seeds, 16 candidates and the existing `_is_story` rule.** Not extended, reworded or reweighted; `min_chars` 12, `_LIST_MAX_CHARS` 120 and the `snnchat.topics` stop list all untouched. That freeze is what makes the arms comparable and it is also the ceiling on everything the round can say: `topic` rests on 64 selected draws per arm, `story_dodge` on 48, `list_strict` on 16. The 0.0049 pool-topic gain over the shipped arm (0.0957 vs 0.0908, n=1,024) sits inside a ±0.018 binomial bar and is reported as not a result.
* **`stories_short.bin` still repacks bit-identically after `Truncation` was parameterised — asserted, not demonstrated.** `test_the_default_truncation_is_the_one_stories_short_was_packed_at` checks `(SHORT.lo, SHORT.hi, SHORT.cap) == (140, 235, 235)` and that `short_story(story, Random(seed))` equals `short_story(story, Random(seed), SHORT)` — over 20 seeds, on one hard-coded story, at the `short_story` level only. It does not repack, does not hash a `.bin`, and does not exercise `short_conversation` or `build_short_stories.py`. Corroborating evidence that the source was in fact never rebuilt: `data/chat/manifest.json`'s entry for `stories_short` has no `truncation` key at all, while `stories_short300`'s records `{name: short300, lo: 242, hi: 407, cap: 407}`.
* **`RAW_FRACTION` left at 0.08 in both sources — identical by rate, not by character.** Solving `0.08·R + 0.92·conv_mean = chars_total/conversations` against the manifest gives R ≈ 725 for both, so bare untruncated narrative is ~24 % of `stories_short` by character and ~16 % of `stories_short300`. `chat-v4a-short` trained on half again as much narrative-with-no-request per story character as this arm did — a plausible driver of `story_dodge` in its own right that moves *with* the manipulation. Named only in the audit amendment, not in the round's original §3. Cheapest untested suspect this round leaves behind.

### What broke

1. **Forty-eight draws were never going to resolve any `story_dodge` threshold, and it was invisible until the flip landed on the line.** The 95 % Wilson interval on 6/48 is **[0.0586, 0.2470]** — a factor of four wide, spanning `chat-v3d-aligned` (0.0833), `chat-v4b-balance` (0.1458) and `chat-v4a-short` (0.1667) simultaneously. Every `story_dodge` comparison in the v4 and v5 packages sits inside one interval. The metric had been carrying threshold verdicts it could not support and would have gone on doing so silently had the arm returned 0.1042. `docs/chat/CONVENTIONS.md` was written the same night in response (`31e49fa`, 23:45, five hours after the results commit), adding `wilson_interval` / `rate_ci` / `resolves_against` to `snnchat.quality`, a third verdict value `unresolved`, and a rule against placing a threshold on a lattice point.
2. **P1 was then resolved by resampling rather than by retraining or reinterpreting, and that part worked.** 301 seeds, so the denominator 12 × 301 = 3,612 is not divisible by 8 and 0.125 is unreachable; seed count fixed in committed code with no stopping rule; seeds 0–3 must reproduce the committed 48 draws character-for-character or the script aborts (`superset_check: identical`). Result **374/3612 = 0.1035, CI [0.0940, 0.1139]**, entirely below 0.125 — verdict `below`, after 3,933.0 s of GPU. Opener branch 273/3612 = 0.0756. Three context arms at 101 seeds: `chat-v3d-aligned` 0.0611 `below`, `chat-v4a-short` 0.1733 `above`, `chat-v4b-balance` 0.1304 `unresolved`. Note the small-sample estimates moved by up to 0.022 (v3d 0.0833 → 0.0611).
3. **`CONVENTIONS.md` §5 quotes the wrong lattice spacing for the 301-seed row.** The table says "12 × 301 = 3,612 draws; 3,612/8 = 451.5, so 0.125 is off the lattice (spacing 0.000825)". 0.000825 is 1/1212 — the *101-seed* spacing from the row below it. `story_dodge_resample_v5a.json` records `lattice: 0.000277` = 1/3612. The conclusion is right; the number attached to it was copied from the wrong row. Recorded because the file's entire purpose is to make denominators and lattices explicit, and this is the one number in it that is neither.
4. **The promised `QUALITY_v5.md` §8 does not exist.** `CONVENTIONS.md`'s preamble states that "`QUALITY_v5.md` §8 applies the rule as an appended amendment and leaves §§1–7 untouched", but the file is 358 lines and ends at §7 "Reproducing it". The resolution landed in `QUALITY_v6.md` §3 instead. A reader of `QUALITY_v5.md` alone sees P1 UNRESOLVED at 0.1250 with no pointer to the 3,612-draw measurement that resolved it. Leaving §§1–7 untouched was correct; the forward pointer was not delivered.
5. **The §4.6 headline table's `fallback` column is mis-transcribed for two of three rows.** It reads 0.009 for `chat-v3d-aligned` and 0.027 for `chat-v4a-short`; the committed `n=1, λ=0` rows are **0.0** for both, and `chat-v4a-short` has `fallback_rate` 0.0 in *all nineteen* rows of its sweep, so 0.027 appears nowhere in that arm's file. Only `chat-v5a-short300`'s 0.0089 is right. The headlines are unaffected — 0.2969, 0.2484 and 0.2779 all recompute exactly from the true values — so this is a transcription error in a reported column, not an arithmetic one. Separately, the same table's header says `list (strict)` (0.000 / 0.033 / 0.000, which is `list_strict`) while the sentence beneath it quotes the *loose* `list` (0.083 vs 0.167, which is what the headline weights): two metrics in one paragraph under one name.
6. **A false comparability claim in committed code, corrected in place.** `short_conversation`'s docstring said `trunc` "changes the *value* of the first draw and not the number or order of draws, so two sources built at different targets from the same seed stay comparable story-for-story." The second half is false: `rng.randint` consumes a different number of Mersenne-Twister words for different range widths (`SHORT`'s width 96 takes `getrandbits(7)`, `SHORT300`'s 166 takes `getrandbits(8)`, with different rejection probabilities), so the streams diverge after the first story. Nothing in the round's design rested on it — the comparison is distributional over ~10⁶ conversations, not paired — but the claim was load-bearing-*looking*. Corrected rather than deleted. Found by the pre-run adversarial audit.
7. **The fix that made the six-range calibration reproducible is itself incomplete.** The original claim ("six candidate ranges were measured") was true but unverifiable: the committed script hard-coded two, so a reader could confirm "[242, 407] gives 298.5" and could *not* confirm it was the nearest of six to 300 — the part that selects the number. `--candidates` was added and the full table committed. But the flag's help text says "The search that chose SHORT300 is the default below" while `--candidates` defaults to `None`, so running `scripts/chat/calibrate_truncation.py` bare still measures only the two named targets. Reproducing the ranking requires copying the five ranges out of `QUALITY_v5.md` §7. Same class of defect as the one it was fixing, one layer down.
8. **§6's `requests / M chars` column was credited to the wrong script.** The pre-registration credited its whole §6 table to `scripts/chat/window_coverage.py`, which emits no such quantity — the column is `conversations / chars_train` from `data/chat/manifest.json`. Numbers correct, provenance line not. Caught by the audit before any result existed and corrected as an amendment rather than by editing §6. Small, and recorded because it is exactly the failure these notes exist to catch: a number that looks measured by a named tool and is not.

## 12. The from-scratch and instruction-ladder round (2026-08-06 → 08-07)

Four training arms — `chat-v6-scratch` (84,000 steps, fresh init), the instruction-weight ladder `chat-v6-inst-a` / `chat-v6-inst-b`, and the seed replicate `chat-v6-inst-a-s1` — plus the no-retraining `story_dodge` resample above. Pre-registration `docs/chat/PREDICTION_v6.md`, write-up `docs/chat/QUALITY_v6.md`. Round total ≈8.28 GPU-hours: training 21,573.8 s (5.99 h), resampling 7,731.5 s (2.15 h), quality batteries 502.5 s.

Every number `QUALITY_v6.md` quotes reproduces from the committed draw files under the committed scorer. What does not survive contact with the artifacts is the **framing** around the round's flagship finding.

### Measured

* **The branch was pre-registered before the number that decided it.** `PREDICTION_v6.md` §0 said the from-scratch arm would use whichever of `chat-v5a-short300` / `chat-v3d-aligned` won §1, and §1 came back z = −0.478 (diff −0.0190, propagated SE 0.0397), unresolved, so the incumbent's recipe ran. Component SEs reproduce too (topic 0.0533, list 0.0979, fallback 0.0089). This is the round's cleanest piece of process: a rule written down before the number, and a branch that can be checked.
* **`chat-v6-scratch` is worse than the incumbent on held-out bpc, on the identical mixture, and the round did not make the comparison.** `QUALITY_v6.md` §2 reports weighted val 1.2399 and declares it "not comparable to `chat-v2-anneal`'s 1.1795 — different mixture". True of `chat-v2-anneal`; but `chat-v3d-aligned`'s mixture is *identical* to `chat-v6-scratch`'s (stories_topic 0.40, soda 0.20, alpaca 0.15, tinystories 0.09, persona 0.08, dolly 0.06, oasst1 0.02), so its weighted **1.1743 is the comparable number** and the fresh run is 0.0656 bpc worse. It is worse on every source: alpaca 1.7617 vs 1.6660, dolly 2.0039 vs 1.9443, oasst1 1.9253 vs 1.8701, persona 0.2649 vs 0.2294, soda 1.2888 vs 1.1948, stories_topic 1.0962 vs 1.0452, tinystories 1.1059 vs 1.0545. That is a seed-free, sampler-free, 40-batch-per-source corroboration of P0's failure, far better powered than the 148-draw battery that decided it. Caveat: `eval_batches` is 40 for scratch and 30 for `chat-v3d-aligned`, so the two cover different-length prefixes of the same held-out tails.
* **The from-scratch arm's small sources were seen 50–244 times over, and nothing in the round notes it.** 84,000 × 160 × 256 = 3.441 G characters. Against the corpus sizes in the run's own `run_start` event: oasst1 68.8 M / 0.28 M = 243.6 epochs; dolly 116.2; alpaca 63.3; persona 51.6. `stories_topic` gets 3.5, soda 1.1, tinystories 0.4. The 14,000-step anneal arms get roughly a sixth of that exposure. The fresh arm's held-out bpc on exactly those over-exposed sources is the worst of any arm in the package — which is what over-exposure plus under-training looks like, and is worth a line in any future from-scratch attempt.
* **The 240-minute ceiling was cleared by about 87 seconds.** §2 calls it "just under budget"; the margin is thinner than that reads. At step 83,900 in-loop elapsed was 14,296.87 s against a 14,400 s ceiling at 5.868 steps/s — ~103 s, and the run needed ~17 s of it to reach 84,000. Roughly 0.6 % of the budget, or ~510 steps. Any of §6's or §9's known contention hazards would have turned P0 into a budget-stopped run scored under `PREDICTION_v6.md` §3's ceiling clause. Nothing was wrong; the margin simply was not what "just under budget" implies.
* **The from-scratch arm ends with a materially narrower slow-pole spread than every arm continuing from `chat-v2-anneal`.** Read off the checkpoints rather than the summaries (see "what broke" below): `chat-v6-scratch` layer 0 has tau min/median/max **2.03 / 12.85 / 799.68**; `chat-v6-inst-a`'s is 1.38 / 2.69 / 7,339.11 and `chat-v3d-aligned`'s 1.38 / 2.67 / 7,928.74. The incumbent lineage carries a slow-pole tail an order of magnitude longer, grown during `chat-v2`/`chat-v2-anneal` and inherited by every `--init-from` arm. So "identical architecture" is true of the parameter count and false of the trained time constants — the horizon `spread_slow_poles` exists to buy. P0 compared a model with a ~800-character tail against one with a ~7,900-character tail and attributed the whole difference to stale pre-training.
* **`chat-v6-inst-a`'s 0.3101 is the highest `n=1, λ=0` headline the project has produced**, and §6 hedges it correctly: against the incumbent the margin is +0.0132, SE 0.0450, z = +0.293, unresolved.
* **The resample's sample sizes were fixed before any number existed and the superset gate held on all four arms.** `CONVENTIONS.md` §5 tabulates 301/101 with the reason while the script was still running. Each arm's record carries `superset_check: {checked: 48, status: "identical"}`, so the enlarged samples *extend* `QUALITY_v5.md`'s measurement rather than replacing it with a differently-seeded one. The mechanism is `collect(probe_index=...)` pinning each probe's index in the full 37-probe battery, because the sampling seed derives from probe position.
* **The resample's true cost was 2.15 GPU-hours; the write-up reports 65.5 minutes of it.** `chat-v5a-short300` 3,933.0 s (the figure §3 quotes) plus `chat-v3d-aligned` 1,278.7 s, `chat-v4a-short` 1,274.0 s and `chat-v4b-balance` 1,245.8 s = 7,731.5 s — about a quarter of the round, and the reason the ladder could not afford the same treatment. Worth stating because §7 prices future resamples off the 65.5-minute number.
* **The two-sample z-test that decides every verdict in the round exists in no committed code and has no test.** Every z reproduces to the digit under the method §1 describes — Wald component SEs, `list` by sample SD/√n, headline SE by the fixed weights combined across independent groups: −2.406 (§8), −0.478 (§1), −0.738 (§2), +0.293 and −1.034 (§6). So the prose is a faithful description. But `snnchat.quality` ships only `wilson_interval`, `rate_ci` and `resolves_against` (one rate against one threshold), and `scripts/chat/compare_quality.py` prints only a ±2√(p(1−p)/n) pool bar. The machinery was computed ad hoc. Contrast the `story_dodge` resample, which is committed code with a superset gate: **the round's decision rule is the least-defended part of its toolchain.**
* **On held-out bpc — seed-independent, sampler-free — the two seeds differ by 0.0023.** `chat-v6-inst-a` weighted 1.146746, `chat-v6-inst-a-s1` 1.144428, identical mixture and identical eval settings; per source they agree to ~0.003 everywhere (alpaca 1.6217/1.6209, stories_short 0.8124/0.8114, tinystories 1.1021/1.0979). The training-seed change moved the language model by 0.2 % and the sampler-scored headline by 32 % — despite s1 also running 10,000 steps at half LR. That is a strong hint §8's 0.099 lives mostly in the 148-draw battery rather than in the model, and it is the cheapest available cross-check on the round's headline finding.
* **The four training runs were strictly sequential on the card, with no overlap.** From `launch.json` start times plus `summary.json` wall clocks: `chat-v6-scratch` 23:51:57 → 03:50:39, `chat-v6-inst-a` 06:15:21 → 06:54:57, `chat-v6-inst-b` 07:05:39 → 07:45:08, `chat-v6-inst-a-s1` 08:02:12 → 08:43:59. No two intervals intersect; the 144.7-minute gap after the from-scratch run is where the resampling fits. This is §9 item 3's standard met by construction rather than after a rerun. Nothing in `QUALITY_v6.md` claims it, so it is recorded here.
* **Round cost and throughput.** `chat-v6-scratch` 14,322.32 s at 5.868 steps/s; `chat-v6-inst-a` 2,375.87 s; `chat-v6-inst-b` 2,368.63 s; `chat-v6-inst-a-s1` 2,507.02 s (the extra ~131 s over inst-a is the 859 steps redone after its rollback). Peak VRAM 3.831 GiB on every arm, unchanged since `chat-v3d-aligned`, so nothing came near the 8 GiB wall that killed `large_K3` in §2. The ladder arms used 39.5–41.8 min of their 46-minute ceilings.

### Guessed

* **`chat-v6-scratch` was run at lr 5e-4 with warmup 500 — the *fine-tune* recipe applied to a from-scratch run.** The only other fresh-init run in the project, `chat-v2`, used lr 2e-3 and warmup 1000 for 76,000 steps. `chat-v6-scratch` inherited 5e-4/500 from the 14,000-step anneal arms because `PREDICTION_v6.md` §2 copied `chat-v3d-aligned`'s command line and deleted `--init-from`. A 4× lower peak LR, and nothing tested whether 5e-4 is sensible from scratch at 84,000 steps. Note the eval trace was still falling at the end (weighted 1.2440 at 78,000 → 1.2399 at 84,000), so the arm was not converged when the cosine ran out. What would test it: one more fresh-init arm at 2e-3 with the same mixture and step count.
* **`chat-v6-inst-a`'s mixture, where soda alone absorbs the 0.13 that instruction weight gains.** `stories_short` pinned at 0.40, tinystories 0.09 and persona 0.08 held at v4a's values, alpaca/dolly/oasst1 raised to v4b's exact 0.26/0.08/0.02, and soda cut 0.20 → 0.07 to balance. The justification in §4 is that soda "is dialogue register, not implicated in either hypothesis". That is an assumption, and it is the one that matters: `social` is precisely the register component the headline weights at 0.2, and the sibling arm that cut soda further (`inst-b`, 0.03) is the one whose `social` fell. So inst-a vs v4a isolates instruction weight from story weight while confounding it with a 65 % cut to the dialogue source. What would test it: an arm at v4a's instruction weight with soda at 0.07.
* **"A second occurrence … consistent with the divergence being seed-triggered" rests on one event in one run.** n=1. It has since been tested by accident: across rounds v6 and v7 there are **eight** 14,000-step `--init-from chat-v2-anneal` runs, and only `chat-v6-inst-a-s1` logs a `divergence` event. So the observed rate is 1 in 8 in this family, and the single diverged run is also the one that produced the round's headline outlier. What would test the seed hypothesis properly: rerun seed 1 with the recovery disabled, or log the gradient norm at every step rather than every 100.
* **§5's decision not to spend the budget on a powered resample, on an advisor-checked power calculation.** §5 declines a resample of inst-a vs the incumbent on the grounds that resolving it needs "~180 seeds per arm (~108 min each, ~3.6 h for the pair)". The inputs to that calculation are not shown and cannot be checked from any artifact; the observed per-seed cost that does exist is the `story_dodge` resample's ~1.3 s per (probe, seed), over 12 probes, not 37. The decision was probably right — it bought the seed replicate instead — but the number that justified it is unsourced. What would check it: the same calculation written out against the committed component SEs (topic 0.0364, list 0.0914, social 0.0000, fallback 0.0089 for inst-a).
* **Comparing `prompt_dependence` deltas across arms of different absolute fit.** §2 and §4 rank arms by the matched-vs-shuffled bpc delta (`chat-v6-scratch` 0.07292, inst-b 0.06697, inst-a 0.06243, `chat-v3d-aligned` 0.06195). Nothing has checked that the delta is comparable between models with different absolute bpc, and across the eight arms in the package the Pearson correlation between `bpc_matched` and `delta` is +0.622 (n=8) — the arm with the largest delta is also the worst-fitting by a wide margin (`bpc_matched` 1.59751 against 1.50–1.53 for everything else). The metric's internal guard is sound (batched vs sequential agree to relative 8e-09 to 2e-07 on every arm), so this is about interpretation, not implementation. What would test it: normalise the delta by matched bpc, or measure it on two checkpoints of the same run at different steps.
* **The qualitative reply-quality claim about `chat-v6-inst-b` rests on a single training-loop sample.** §4 contrasts inst-b's "Hello there, what can I add it for a natural language?" with inst-a's "Hi there. What can I do for you?". Both are the step-14,000 `sample` events from `_sample_report`, drawn at temperature 0.8, top_p 0.92 and `seed=self.global_step` — one draw each, from a code path whose documented purpose is "the only honest progress signal", not measurement. At step 12,000 the same probe gives inst-b "Hi there! How can I help you today?", which reads no worse than inst-a's. The quantitative version of the claim is the `social` component (0.9000 vs 1.0000, 20 draws each, |z| < 1.5, unresolved), and §4 already reports it.
* **Only the four pre-v6 arms were resampled; the round's own three new arms were not.** `chat-v6-scratch`, `-inst-a` and `-inst-b` stayed at 48 draws. So §2's "`story_dodge` (0.0833, identical to four decimal places on both arms)" is 4/48 against 4/48 — the same lattice point, on the very denominator §3 spent 65 minutes of GPU proving cannot resolve anything (95 % CI [0.0329, 0.1955] on both). "Identical to four decimal places" is a property of the lattice, not an agreement of two measurements. Resampling `chat-v6-scratch` would have cost about what `chat-v3d-aligned`'s context arm cost (1,278.7 s).

### Inherited

* **The architecture, size and every neuron hyperparameter, untouched for all four arms.** d_model 1024, n_layers 4, vocab 101, arch `twocomp_threshold`, beta 0.5, threshold 1.0, surrogate_alpha 2.0, w_init 0.1, thr_log_init 0.0, beta_slow 0.95, spread_tau true, tau_min 3.0, tau_max 600.0, weight_decay 0.1, beta1/beta2 0.9/0.95, grad_clip 1.0, fp32, fused, cuda_graph — identical across all four configs and identical to `chat-v4a-short`. All four report 4,417,637 params, matching the pre-registration's dry-run claim. Not re-checked at this round's settings; carried.
* **The probe battery, seeds, candidate count, sampler string and 19-row reranking grid, frozen and reused unchanged.** All eight arms compared this round were drawn at 37 probes (16 topic / 7 fact / 5 list / 5 social / 4 identity), seeds (0,1,2,3), `n_candidates` 16, null `empty_user`, `temperature=0.85 top_k=0 top_p=0.92 min_p=0.02 loop_penalty=6 max_new=300`, and scored over `DEFAULT_GRID`'s 19 rows with (1, 0.0) first. Component denominators follow: topic 64, list 20, social 20, substantive/fallback 112, `story_dodge` 48. `temperature_spread` is 0.0 on every v4/v5/v6 file and absent from `chat-v3d-aligned`'s, which predates the knob — comparable only because §10 item 2 made spread = 0 mean the scalar code path. Re-scoring every file with the committed scorer reproduces every stored row, so no scorer drift entered between rounds.

### What broke

1. **P0's 84,000-step budget was declared "the same total step budget" as the incumbent's lineage. It is 14,000 steps short of it.** `PREDICTION_v6.md` §1 matches 84,000 to `chat-v2` (76,000) + `chat-v2-anneal` (8,000). But the arm it is scored against, `chat-v3d-aligned`, is those 84,000 steps **plus** its own 14,000-step anneal — 98,000 in total, at a 4× higher peak LR through the bulk of them. The fresh arm got 86 % of the incumbent's steps and a quarter of its peak LR, and P0 was written as though the only difference were staleness of the pre-training. A clean test of the stale-pretraining hypothesis needs 98,000 fresh steps, or a step-matched incumbent.
2. **`chat-v6-inst-a-s1` is NOT `chat-v6-inst-a` with a different batch order. It ran 10,000 of its 14,000 steps at half the learning rate.** This is the round's largest single problem. The log records `{"event": "divergence", "step": 4859, "loss": NaN, "recovery": 1, "action": "rollback", "resumed_at": 4000, "sampler_seed": 2, "lr_multiplier": 0.5}`. `_recover_from_divergence` halves the LR for the remainder, and the trace confirms it: at step 4,000 inst-a logs lr 4.2548e-04 and s1 logs 2.1274e-04; at step 10,000, 1.2067e-04 against 6.0337e-05. So from step 4,000 onward s1 trained at half the peak LR *and* on sampler seed 2. `QUALITY_v6.md` §8 and `PREDICTION_v6.md` §5 both assert "the only thing a different seed changes is which batches the 14,000-step fine-tune sees, in what order". That is false for the run actually executed. §8 does describe the rollback — in a closing "genuine bonus, not scored" paragraph — and never reconciles it with the claim two paragraphs above. **The finding survives as "something about a same-recipe reroll moved the headline 0.099". It does not survive as a measurement of seed variance.**
3. **The s1 divergence is attached to the wrong precedent.** §8 calls it "the same gradient-norm divergence shape `BUILD_NOTES.md` §9 already flagged". The event in question is §7's — a finite but enormous global gradient norm, 1,729,672.38 against a median of 0.288. `chat-v6-inst-a-s1`'s **maximum** logged gradient norm over the entire run is 0.4166 (median 0.2451), and the nearest logged step before the NaN is 4,800 with |g| = 0.2664 and loss 0.7148. The NaN at 4,859 fell between two 100-step log points with nothing anomalous on either side. If anything this matches Phase 4's *other* divergence (`compose_s0` at step 17,598, "gradients of 0.03 and everything finite"), not §7's spike. §7's own caveat applies twice over: at `log_every=100` the true frequency of whatever happened is unmeasured.
4. **The 0.099 swing is read at the one row of nineteen where it is largest, and where its sign is the minority sign.** At the pinned row s1 − inst-a = −0.0992. Across the full 19-row sweep the difference is positive on 9 rows and negative on 10, mean −0.0073; at every *other* λ=0 row seed 1 is **higher** (n=4: +0.0580, n=8: +0.0136, n=16: +0.0172), and the pinned row is the extreme of the set in either direction. The 19 rows are re-selections from the same 148 draws, not independent replicates, so this is not a significance argument — but it does mean "seed 1 lower" is a property of one decoder setting rather than of the two checkpoints. §8's defence ("all four components moved the same direction at once") is only true at that row.
5. **On the one metric with no sampler in it, seed 1 moved the other way, and §8 does not mention it.** Prompt dependence — which `scripts/chat/quality.py`'s own docstring calls "the number to believe", with no seed, no lexicon and no sampler — reads 0.07142 for `chat-v6-inst-a-s1` against 0.06243 for `chat-v6-inst-a`: s1 is 14 % **higher**, on the same 26,328 held-out characters. §2 leaned on exactly this metric to say something encouraging about `chat-v6-scratch` (0.07292 vs 0.06195). It cannot be the number to believe in §2 and absent from §8.
6. **`chat-v6-inst-b` is described as "one clean step up the instruction axis". It moves six sources at once, two of them the register sources.** Reading inst-a → inst-b from the configs: alpaca 0.26 → 0.32, dolly 0.08 → 0.10, oasst1 0.02 → 0.03 (instruction +0.09), funded by soda 0.07 → 0.03, persona 0.08 → 0.05, tinystories 0.09 → 0.07. §4 attributes inst-b's `social` drop (1.0000 → 0.9000) to instruction weight and calls the step clean — but persona, the authored social/register source, was cut 38 % and soda 57 % in the same step. The `social` cost is at least as attributable to the ballast, and P1b's answer ("plateau/reverse") therefore closes the instruction axis on an arm that also closed the register axis. Only `stories_short` (0.40) is genuinely pinned across the ladder.
7. **P1a was scored as "a recall error, not a re-derivation". It is neither — it is two aggregations in one rule.** §4's P1a asks whether "topic propensity stays nearer v4a's 0.0723 than v4b's 0.0645". Those are not invented numbers: they are exactly `report.best.pool_by_kind["topic"]` for `chat-v4a-short` (0.0723) and `chat-v4b-balance` (0.0645), the 1,024-draw column `QUALITY_v4.md` tabulates. `QUALITY_v6.md` §4 declared the rule inapplicable by reading `by_kind["topic"]` at the pinned row instead (both arms 0.0469, 3/64) — a different, 64-draw quantity. The real defect is that P1a pairs a **pool** metric with a **pinned-row** metric in one conditional. Applied at the aggregation its own reference values came from, it resolves: `chat-v6-inst-a`'s pool topic is 0.0693, nearer v4a's 0.0723 (Δ0.0030) than v4b's 0.0645 (Δ0.0048), and `list_strict` moved 0.0333 → 0.0500 toward v4b's 0.0833 — i.e. "credit shifts to instruction weight", the same conclusion §4 reached on the plain data. Nothing was lost; the diagnosis on the record is wrong.
8. **The ladder arms are compared to the incumbent across a 63-character reply-length gap, with no mention of the confound the previous two rounds were built around.** At the pinned row: `chat-v6-inst-a` 126.1 chars, `-inst-b` 134.7, `-inst-a-s1` 123.9, against `chat-v3d-aligned` 188.9 and `chat-v6-scratch` 187.3. §11 exists *because* `topic` and `list (strict)` depend on reply length. §6's ship table nonetheless ranks inst-a's 0.3101 against the incumbent's 0.2969 without noting that the two arms differ by a third in reply length. The within-round comparisons are fine — inst-a vs inst-b and inst-a vs s1 are length-matched, and so, incidentally, is `chat-v6-scratch` vs the incumbent (187.3 vs 188.9), the round's own matched pair, also unremarked.
9. **`summary.json`'s `slow_pole_tau` block describes the model at LOAD time, not at the end of training — in every arm this project has.** `ChatTrainer.description` is built at construction and refreshed only in `load_checkpoint` and `init_from`, never before the `run_end` summary is written (`src/snnchat/train.py` lines 125, 512, 538, 545–557, 651). Consequence: a from-scratch arm reports its initialisation (`chat-v6-scratch` and `chat-v2` both claim 3.00 / 42.32 / 600.02 on all four layers, while `chat-v6-scratch`'s checkpoint actually holds 2.03 / 12.85 / 799.68); an `--init-from` arm reports the checkpoint it loaded (`chat-v6-inst-a` claims 8,507.72 where its own weights hold 7,339.11). The only reason `chat-v6-inst-a-s1`'s summary differs at all is that its rollback called `_refresh_description()` at step 4,000 — so its 7,921.25 is the one genuine mid-training reading in the package and the others are not. Cosmetic: it touched no training and no reported metric. The fix is one `_refresh_description()` before the summary is assembled.
10. **`CONVENTIONS.md` §5's lattice error (§11 item 3) is compounded by the resample script's own docstring**, whose section "THE SEED COUNT IS FIXED BEFORE ANY NUMBER IS READ" argues in detail for `--seeds 101` while the P1 arm was run at 301. Neither error touches a verdict — [0.0940, 0.1139] sits below 0.125 either way — but the pre-commitment record is the artifact whose whole value is being exactly right, and `CONVENTIONS.md` is the authority the script's docstring now contradicts.

## 13. The n=4-training-seed round (2026-08-07, committed as `c0ea307`)

An unattended session that bought the one thing five prior rounds never had: more than one training seed per arm. `chat-v3d-aligned` (incumbent) and `chat-v6-inst-a` (challenger) raised to n=4 seeds each, n fixed in advance so a null could not be seeded away, compared with a stdlib Welch t-test. Pre-registration `docs/chat/PREDICTION_v7.md`, write-up `docs/chat/QUALITY_v7.md`. Every row came back unresolved, which `PREDICTION_v7.md` §3 said would happen before any of it trained, and `SHIPPED` did not move.

**The round's real product is not the comparison. It is the first measured between-seed SDs, 0.0141 and 0.0419.** And the largest thing the write-up does not say is that the two groups are not four independent draws each.

### Measured

* **n = 4 seeds per recipe, fixed before the first run and honoured.** 5/5 planned runs completed and were scored; the fixed-n rule was not renegotiated once interim results were visible. The driver logged `SESSION7_ALL_DONE 4/5` on the first pass and `5/5` after the backfill. `PREDICTION_v7.md` §4's `insufficient n` escape clause was never needed. This is the first recipe-vs-recipe comparison in the chat lineage with a pre-declared sample size.
* **The pooled Welch comparison, unresolved on all five rows.** Headline: incumbent 0.2940 (sd 0.0141) vs challenger 0.2694 (sd 0.0419), diff −0.0246, se 0.0221, t = 1.111, df = 3.67, p = 0.3341. Components: topic p = 0.1373, list p = 0.7788, social p = 0.2522, fallback p = 0.6198. Re-running `scripts/chat/session7_stats.py` against the committed JSONs reproduces §2's per-seed table and §3's statistics digit for digit.
* **One identical battery across all eight checkpoints.** Same draw settings in all eight `_quality/*.json`: `temperature=0.85 top_k=0 top_p=0.92 min_p=0.02 loop_penalty=6 max_new=300`, `n_candidates: 16`, `seeds: [0,1,2,3]`, 37 probes, `temperature_spread: 0.0`. Scoring cost 117.5–137.2 s per checkpoint. Nothing about the decoding path differs between the arms being compared.
* **Reading `ckpt_best.pt` rather than the final checkpoint is inert here, and demonstrably so.** All eight runs have a monotonically falling eval trace, so `ckpt_best.pt` **is** the step-14,000 checkpoint in every one of them. That is what rescues the eval-cadence mismatch below from being a confound.
* **Both recipes are the same architecture at the same size from the same starting weights.** All eight configs record 4,417,637 params, d_model 1024, n_layers 4, `twocomp_threshold`, 14,000 × 160 × 256 = 573,440,000 characters, and `--init-from experiments/chat/chat-v2-anneal/ckpt_best.pt`. `PREDICTION_v7.md` §2's dry-run claim is confirmed after the fact by the committed summaries even though no dry-run transcript was committed.
* **`--budget-minutes 75` was a safety ceiling, not a duration, and it never bound.** The five new runs took 2,372.32, 2,372.87, 2,373.77, 2,378.22 and 2,404.23 s — 39.5 to 40.1 minutes. No run was truncated mid-cosine, so no arm is short. §5's phrasing "~40–75 minutes each" implies a spread that does not exist in the artifacts.
* **The whole campaign cost ~3.49 GPU-hours of an ~8-hour budget.** 11,901.41 s of training plus 650.2 s of scoring = 12,551.6 s. Driver wall-clock agrees: 10:11:13 → 12:59:18 plus a 12:59:29 → 13:41:26 backfill. Roughly 4.5 GPU-hours went unspent.
* **The divergence-recovery seed scored best on validation bpc and worst on the headline.** `chat-v6-inst-a-s1` has the **lowest** weighted val bpc of its group (1.1444, against 1.1467 / 1.1480 / 1.1479) and the **lowest** headline (0.2109, against 0.3101 / 0.2819 / 0.2749). The half-LR tail after its rollback bought the best language-model number and the worst quality number in the same run. Anything reaching for val bpc as a cheap proxy for the headline should be pointed at this pair of columns first.
* **A stdlib Welch t-test with a hand-rolled regularized incomplete beta, because there is no scipy on this box.** `session7_stats.py` computes Welch–Satterthwaite df and gets the p-value from `betai(df/2, 0.5, df/(df+t²))` (Numerical Recipes `betacf`/`betai`). It reads the committed `_quality/*.json` and nothing else, so no hand-transcribed number enters the table. Two nits, neither numerical: the `if se > 0 else float('inf')` guard would silently return t = inf if both groups had zero variance (the `social` row came within one group of that), and `QUALITY_v7.md` §3 cites `report["baseline_picks"]` as the source row when the script reads `report["sweep"]` where `n == 1` — `baseline_picks` holds reply texts, not metrics. Same row, wrong citation.
* **The driver's restart rule — skip a run whose `summary.json` shows 14,000 steps, delete a partial directory, never resume — was exercised by a real failure and held.** The backfill pass retrained only `chat-v3d-aligned-s1` and logged `SKIP` + `SKIP_EVAL` for the other four. No artifact on disk describes two runs at once, and no half-run got resumed into a completed one. This is the one piece of session-7 infrastructure that a failure actually tested.
* **Not attempted, and the reason is a measurement rather than a preference: a corpus expansion.** Elliot asked for a session that "trains on all the data the project has"; `PREDICTION_v7.md`'s preamble scopes that down to GPU-hours on the existing chat mixture, because `data/enwik8` / `data/text8` use a 205-symbol vocabulary against chat's 101 and a research checkpoint's embedding rows are indexed by a different alphabet. Every config records `vocab_size 101, vocab_version 1`, and `init_from` hard-fails on a `vocab_version` mismatch. The scope reduction is enforced by code, not by argument.

### Guessed

* **Seeds 0–3 per recipe, reusing the three checkpoints that already existed** (`chat-v3d-aligned`, `chat-v6-inst-a`, `chat-v6-inst-a-s1`) to save ~2 GPU-hours. Consecutive small integers. Nothing tests whether seeds 0–3 behave like four draws from the seed distribution — `_mix64(seed, step)` should decorrelate them, but that is an argument, not a measurement. What would test it: four more seeds at arbitrary large values and a comparison of the two SDs.
* **`SD_seed ≈ 0.070` as the power calculation's input, extrapolated from the single 0.099 pair.** `PREDICTION_v7.md` §3 says so itself and warns it is a guess: `SD_diff ≈ 0.099` for two draws implies `SD_seed ≈ 0.099/√2 ≈ 0.070`. The round then measured 0.0141 and 0.0419, both smaller. **The guess being labelled a guess in advance is the part that worked.**
* **α = 0.05 two-tailed on five rows with no multiplicity correction.** Never justified, and structurally odd: `headline` is a deterministic linear function of the other four rows (0.5·topic + 0.3·list + 0.2·social − 0.2·fallback_rate, verified — it reproduces 0.2969 and 0.2109 exactly), so §3's table is not five independent pieces of evidence and Bonferroni would be the wrong instrument anyway. Moot this round because nothing resolved; it becomes load-bearing the first time one row does.
* **One battery per checkpoint, with no attempt to separate battery noise from training-seed noise.** The round's headline deliverable is called `SD_seed`, but what it measures is total per-checkpoint variance: training seed **and** the battery's own finite-sample wobble on 37 probes × 4 scoring seeds. Nothing partitions them. What would test it: score one checkpoint at battery seeds 4–7 as well as 0–3 and read the spread of a fixed checkpoint. That measurement costs ~2 minutes of GPU and was not made.
* **The tie-break rule for a challenger win — "the seed whose `ckpt_best` scored closest to the resolved group mean", not the best of four.** Pre-registered in §3 for good reasons (picking the highest of four correlated draws moves the peeking problem from stopping time to selection time) and never exercised, because nothing resolved. Untested machinery, with an unexamined edge: proximity-to-mean would deliberately ship a mediocre checkpoint, which is statistically clean and may not be what anyone wants from a deliverable.
* **Deliberately not attempted: §11's two `story_dodge` suspects and the `bot_loss_weight` / `align_frac` isolation.** Named in `QUALITY_v6.md` §7, deferred by `PREDICTION_v7.md` §5 on the judgement that each needs its own resample budget and its own pre-registration. The judgement is sound and it was honoured under pressure — §5 says explicitly that the fixed-n commitment was kept "rather than redirecting mid-campaign toward a result that looked more decisive". Note what it cost: ~4.5 of the ~8 GPU-hours went unspent, and §5's own escape hatch (append a §6 at the time if hours remain) was available and not used. **Fixing n at 4 rather than at what the budget could afford was itself the unexamined choice.**

### Inherited

* **The headline formula and the 37-probe battery, carried in unchanged and unexamined.** `headline = 0.5·topic + 0.3·list + 0.2·social − 0.2·fallback_rate`, with `list` resting on 5 probes × 4 scoring seeds = **20 picks**. Decomposing this round's own eight numbers: the variance of the `0.3·list` term alone is 1.729e-4 against the incumbent group's headline variance of 1.997e-4 — **86.6 % of the round's flagship `SD_seed = 0.0141` is the wobble of a 20-pick column.** Nobody stated that, and it is the first thing to check before quoting the SD anywhere else.
* **Initialising all eight runs from the same `chat-v2-anneal/ckpt_best.pt`.** Carried from every prior chat round without re-examination, and it silently scopes the deliverable: `seed_everything(cfg.seed)` runs, then `init_from` overwrites every weight with the identical checkpoint, so **initialisation variance is excluded by construction** and the measured SD covers data order plus CUDA nondeterminism (`deterministic: false`) only. 0.0141 / 0.0419 are therefore a *lower bound* on the seed variance a from-scratch recipe would show. `QUALITY_v6.md` §8 said this for its pair; `QUALITY_v7.md` does not repeat the caveat while generalising the number.
* **Reusing `chat-v3d-aligned`'s quality JSON, whose draws predate the temperature-spread knob.** Its report carries `"rescored": true` and has **no** `temperature_spread` field, unlike the other seven — its candidate texts were drawn by an older generate path and only the sweep was recomputed under current probes. Comparability rests entirely on §10 item 2's guarantee that `spread = 0` was made to mean the scalar code path rather than a ladder of equal values. That guarantee was not re-checked here; it was taken on trust. Similarly `chat-v6-inst-a` and `-s1` were scored in session 6 and not rescored — no `src/snnchat/` file changed in `c0ea307`, so this is defensible, but it was never stated as a premise.
* **`--budget-minutes 46` for the challenger's seeds 0–1 against 75 for seeds 2–3.** A launch-flag mismatch inside the challenger group, carried from round v6. Inert: the longest of those four runs is `chat-v6-inst-a-s1` at 2,507.02 s = 41.8 min, under even the 46-minute ceiling. Recorded because `PREDICTION_v7.md` §2 asserts the runs share "the same step count and budget" and one of those two is not true.
* **Trusting `summary.json`'s `description.slow_pole_tau` as a description of the trained model.** §12 item 9's bug, and this is where it becomes actively misleading: seven of the eight summaries report the taus of the *shared init checkpoint* (layer 0 `tau_max` 8507.72, identical across all seven), and the only run reporting different numbers (`chat-v6-inst-a-s1`, 7921.25) does so because its rollback called `load_checkpoint` at step 4,000. The field reads like an s1 anomaly and is the exact opposite: s1 is the only run whose taus are post-training at all.
* **`unresolved` leaves `SHIPPED` unchanged.** The rule from every prior round, applied again and stated in both directions: §4 explicitly refuses to read the null as a quiet win for the incumbent, and the `SHIPPED` file records the round with its numbers rather than just the outcome. What the round does **not** note is that across §§11–13 alone the incumbent has now survived five consecutive unresolved challenges (z = −0.478, z = −0.738, z = +0.293, z = −1.034, p = 0.3341). A rule that can only retain and never displace, on a test this underpowered, is a ratchet, and nothing in the lineage has priced that.
* **Deliberately not attempted: a longer anneal (> 14,000 steps), and the `spread_slow_poles` A/B.** The longer anneal is refused as a genuinely new arm — conflating "more steps" with "more seeds" in one round would make neither measurement clean, which is right. `spread_slow_poles` stays research-side-only per `docs/chat/README.md`'s boundary, as in every prior round; the untested-since-§3 status of that initialisation is now five rounds old and unchanged.

### What broke

1. **Treating the two groups as four independent draws each. They are not, inside the challenger group.** `chat-v6-inst-a-s1` diverged at step 4,859, and `_recover_from_divergence` sets `self.sampler.seed = old_seed + 1` — 1 → 2. `chat-v6-inst-a-s2`'s sampler seed is 2 for its whole run, and `batch(step)` is a pure function of `(seed, step)` (`np.random.default_rng(_mix64(self.seed, step))`, `src/snnchat/data.py:187`). **So s1 and s2 drew the identical batch sequence for steps 4,000–14,000: 10,000 of 14,000 steps, 71 % of training, same order, same mixture.** The Welch test's independence assumption is violated for the very group whose 3× wider SD §3 offers as its secondary lead. Trivially avoidable: bump the seed by a large constant, or by the recovery count × 1000, instead of by 1.
2. **Leaving the divergence-recovery machinery armed inside a variance measurement.** `chat-v6-inst-a-s1` is not the same recipe as its three group-mates: `lr_multiplier *= 0.5` persists for the remainder, so at step 10,000 its LR is 6.034e-05 against 1.207e-04 for s0/s2/s3 — exactly half, for 10,000 of 14,000 steps — and its log replays steps 4,000–4,800 a second time. One quarter of the challenger's "n = 4 same-recipe" group ran a different LR schedule on a different data stream, and it is the seed that produces the group's 3× SD. §3 names the rollback as the driver of that SD but does not say the run therefore is not a replicate of its own recipe.
3. **The pre-registered power figure overstates this design's sensitivity by about 25 %.** §3 says the design "resolves a gap of roughly 0.14 or larger, Cohen's d ≈ 2.0 at 80 % power". That is the large-sample normal approximation (n = 15.7/d² → d = 1.98 at n = 4), not the small-sample truth. Monte Carlo through the round's own `session7_stats.py::welch` at n = 4 per group, α = 0.05 two-tailed, 40,000 reps per point: d = 2.00 gives power **0.610**, d = 2.40 gives 0.763, d = 2.50 gives 0.792. Eighty percent power needs d ≈ 2.5, i.e. a gap of ~0.175 at SD 0.070. Harmless to the conclusion — the round predicted `unresolved` and was even less able to resolve than it thought — but it is the number future rounds will reuse.
4. **Reporting `SD_seed` = 0.0141 for the incumbent without reconciling it against session 6's own noise estimate.** `QUALITY_v6.md` §8 put the propagated sampler SE on a headline *difference* at 0.0412, i.e. ~0.029 per checkpoint. This round's incumbent group spans an SD of 0.0141 — half that. Both numbers cannot be right about the same quantity, and neither round reconciles them. Either the v6 propagation is inflated (its own §1 already flagged an se = 0 boundary bias in it) or four seeds landed improbably tight. Unaddressed.
5. **Running the same Welch test on the `social` row, where it is degenerate.** All four incumbent seeds score exactly 1.0000 — a hard ceiling, sd = 0.0000 — so v1 = 0, the Welch df collapses to exactly n2 − 1 = 3.00 (which is why that row alone reports df 3.00), and the test reduces to a one-sample t-test of the challenger group against the constant 1.0. Its p = 0.2522 is not the two-sample quantity the column headings claim. `QUALITY_v6.md` §1 flagged precisely this class of boundary artefact for the propagated SE one day earlier; v7 does not flag it in its own table.
6. **`PREDICTION_v7.md` §2's claim that "the only thing that differs within a recipe is `--seed`" is true of the challenger group and false of the incumbent's.** `chat-v3d-aligned/config.json` records `eval_every` 3500, `eval_batches` 30, `log_every` 250, `sample_every` 3500; its three replicates record the current `ChatConfig` defaults 2000 / 40 / 100 / 2000. Seed 0's `ckpt_best.pt` was selected over 4 eval points on 30 batches and its replicates' over 7 on 40. §2 caught exactly one of the five stale fields (`--warmup-steps 200`, correctly forced on the command line) and missed the other four. Inert as it happens — every run's best eval is its last — but inert by luck, not by design, and nothing in the round checked.
7. **The stated evidence for the `chat-v3d-aligned-s1` 0xC000013A death does not support the inference drawn from it.** The observation is real: the run died 7 seconds in with rc 3221225786 and nothing in the driver log. But `session7_driver.py:91` calls `subprocess.run(cmd, cwd=REPO)` with no `stdout=`, and the driver itself was launched detached by `launch.py` — **no child's stdout is captured, for any run.** The driver log contains zero `train.py` output for the four runs that completed normally, exactly as for the one that died. By contrast `chat-v6-inst-a/stdout.log` (launched straight through `launch.py` in session 6) carries the full banner and even the `!! loss nan at step 4859` line. So "no banner, not even the pre-CUDA one" says nothing about how far s1 got. Worth noting alongside: 0xC000013A is `STATUS_CONTROL_C_EXIT`, and `launch.py` sets `CREATE_NEW_PROCESS_GROUP` on the driver — a console-control event delivered to that group at 10:11:20 is a candidate the round did not consider. The fix that would have made this diagnosable is one keyword argument: capture the child's stdout to the run directory.
8. **The 0xC000013A death was retried rather than chased.** Recorded here as the round recorded it: the retry ran the identical command from a cold start and completed in 39.6 minutes, the failure did not recur across five other attempts, and it left no partial checkpoint and no traceback. §1 and §4 both name it as unexplained rather than laundering it out of the log, and the backfill driver's log is committed next to the original so the record shows both passes. That is the right disposal for a non-experiment; it is still an unexplained process death sitting inside the round's data collection.
9. **§3's characterisation of the rollback seed as "dropping every component at once" overstates by one column.** `chat-v6-inst-a-s1` is worst-in-group on `list` (0.0500), `social` (0.8500) and `fallback` (0.0268), but **not** on `topic` — `chat-v6-inst-a-s3` is lower at 0.0469 against s1's 0.0625. The seed-stability lead is still worth chasing; the sentence supporting it is not accurate as written.
10. **§2's per-seed table reports bare rates, without numerators, denominators or intervals.** `QUALITY_v6.md` §8's equivalent table gave "0.0938 (6/64)" and "1.0000 (20/20)"; §2 gives 0.0938 and 1.0000 alone. `CONVENTIONS.md` §1's requirement is triggered by comparison against a pre-registered *threshold*, and this round compares group means to each other, so it is a format regression rather than a violation — but the denominators are exactly what would have made the `social` ceiling problem and the 20-pick `list` column visible on the face of the table.

## 14. The decoding round (2026-08-11)

The first chat round that trained nothing. No checkpoint changed and
`experiments/chat/SHIPPED` still names `chat-v3d-aligned`; what changed is how a
reply is chosen from the drafts that checkpoint already produces, plus four
measurements that did not exist. Full write-up in `docs/chat/QUALITY_v8.md`.

**There is no `PREDICTION_v8.md`, and this round is exploratory.** That is a
real departure from v4–v7 and it is recorded here rather than glossed. One
measurement was designed before its numbers were read — the held-out comparison
in §4 — and it is the only one this round treats as confirmatory.

### Measured

* **The pool was already better than the selector.** An oracle over the eight
  drafts the model produces anyway scores 0.2969 on the `topic` battery where
  the shipped score picks 0.1406 (`experiments/chat/_quality/chat-v3d-aligned.json`,
  n = 8, λ = 0.6). Selection, not proposal, was the binding constraint.
* **The echo partition works on topics the battery has never contained.** 20
  held-out story nouns × 6 seeds, both selectors choosing from one shared pool:
  1/120 → 10/120, exact McNemar p = 0.0039, nine draws better and none worse,
  against an oracle of 11/120 (`_quality/echo_holdout.json`). This is the second
  cleanly resolved comparison in the chat package's history.
* **The function-word list did not need tuning, and tuning made it worse.**
  Three lists replayed over the committed draws: closed-class-only and
  closed-class-plus-request-verbs agree to four decimals on every kind; adding
  the battery's own task nouns *drops* `list` from 0.233 to 0.100. The shipped
  list is closed-class only.
* **`λ` no longer buys topicality.** On the held-out pool the selected reply is
  identical at every λ from 0.00 to 1.50 — 10/120 throughout, not one draw of
  120 changes — while mean per-character `logP` falls from −0.2951 to −0.4325
  (`_quality/lambda_sweep_v8.json`). The trade λ = 0.6 was chosen to balance no
  longer exists.
* **38 % of the `list` column is recitation of `snnchat.persona`.** 130/340
  pooled over 17 arms, 95 % CI [0.3323, 0.4350], while `_is_fallback` — the term
  written to catch exactly this — fires on 28/1904 substantive picks
  (`_quality/canned_rate.json`). `list` is 20 draws at weight 0.3 and carries
  ~89 % of the headline's between-training-seed variance.
* **Cross-turn recall is 0 of 240, with a control at 0 of 240.** 10 facts × 8
  seeds × 3 conversational distances, each run twice at the same seed with and
  without the establishing turn. 95 % CI [0, 0.0158]
  (`_quality/memory_probe.json`). Asserted in `README.md` and `RESULTS.md` since
  the first round; measured for the first time here.

### Guessed

* **`echo` defaults to on.** Supported by the held-out result, but that is one
  battery of 20 nouns on one checkpoint. Whether a *reader* prefers the echoed
  reply is untested — §4's own winning replies include "Once upon a time, there
  was a big oak tree. The boat was very happy.", which is on topic and not good.
* **The tier is a hard partition rather than a soft score.** Chosen because
  `min_chars` is already a partition and because a count and a mean
  log-probability have no principled exchange rate. A soft rule was never tried.
* **Distinct words, not total, and a crude singular stem** (`w[:-1]` when longer
  than four characters and ending in `s`). Both are judgements. The stem's
  false-match rate was never measured; the argument for tolerating it is that a
  false match costs one tier place rather than a wrong answer.
* **`_CANNED_MIN_CHARS = 20` was calibrated, not derived.** 15 was wrong (see
  below); 20, 24 and 30 all give the same answer on the shipped arm, which is
  the evidence that 20 is not an artefact — not evidence that it is optimal.
* **The memory probe's ten items and three distances.** A null at 0/240 is not
  sensitive to the battery's composition in the way a positive rate would be,
  but a different set of facts — shorter targets, or targets the corpus uses
  more often — could in principle read differently.

### Inherited

* **λ = 0.6 stands as an incumbent, not as a result.** Every alternative tested
  came back `unresolved` (lowering loses 5 topic draws and gains 0 at p = 0.0625;
  raising to 0.8 is 6 better and 3 worse at p = 0.508), and an unresolved
  comparison does not displace an incumbent. But the measurement that originally
  justified 0.6 was a topicality/fluency trade the echo partition has removed.
* **`min_chars = 12`, `null = "empty_user"`, `temperature_spread = 0`** — all
  carried unchanged and none re-checked with the partition on.
* **`headline` is untouched.** `canned_rate` is emitted beside `fallback_rate`
  and folded into nothing. A term added now would silently rewrite five rounds
  of ship decisions.

### Broke

* **The first version of `_is_canned` fired on 53 of 64 `topic` picks.**
  `persona.TOPICS["story_request"]` holds written-out narratives, and 15
  characters of one of those is "once upon a time" — the opening of nearly every
  story the model tells. Caught by checking the flagged texts rather than the
  rate. Fixed twice over: story replies are excluded from the comparison set
  (that failure belongs to `story_dodge`) and the threshold is 20.
  **The first draft of the metric written to expose a bad metric was a worse
  one**, and it would have reported a 51 % contamination rate.
* **An `Edit` silently deleted a test.** Writing a new test over the `def` line
  of `test_topic_prefers_the_repeated_noun_over_the_character_name` reparented
  its body into the new function; the suite still passed. Caught by counting
  `^def test_` against `HEAD` (85 + 5 expected, 89 present). **A green suite does
  not prove no test was lost.**
* **`/candidates` starred a reply the selector could not have returned.** It
  recomputed the winner from `last_candidates` and omitted the `min_chars`
  partition, disagreeing with `rerank` on 298 of 6,216 committed draw/N
  combinations, and it read the *current* settings against a pool drawn under
  the old ones, so `/echo off` moved the star onto the draft the partition had
  just rejected. Both fixed by recording the decision at selection time instead.
  The defect predates this round; the round's own comment claimed it was fixed
  when it was half-fixed.
* **`/rerank <n>` silently undid `/echo off`**, because it rebuilds
  `RerankParams` and read only `lam` and `temperature_spread` off the old one.
* **`build_corpus.DEFAULT_MIX` was the `chat-v1` mixture**, naming none of
  `stories_topic`, `stories_short` or `stories_short300` — so any run launched
  without an explicit `--mix` trained a two-rounds-obsolete recipe. Updating it
  to the shipped mixture then exposed a second, worse defect: `stories_topic` is
  built by `build_topic_stories.py`, which `build_data.py` does not call, and
  `MixtureSampler` renormalises over the sources actually present. A fresh clone
  would therefore have trained a mixture that appears nowhere in this repository
  and said nothing about it. The renormalisation is deliberate and tested; the
  *silence* was not, and it now prints the missing source and the weight it
  carried. `docs/chat/README.md` §4's quickstart was missing the same line.

### What this round did not touch

The headline's own defects. It is still 89 % `list` variance at 20 draws, still
gated on a column that is 38 % recitation, and no comparison in `QUALITY_v8.md`
uses it. `canned_rate` makes that visible and decides nothing about it.
