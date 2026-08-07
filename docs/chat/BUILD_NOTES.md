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
