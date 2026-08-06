# Chat quality, round 5 — is it reply *length*?

**Date:** 2026-08-06
**Author:** Elliot Asher Caudill
**Status:** **CLOSED** 2026-08-06. **P2, P4, P5 and P6 held; P3 held as written
and split against itself; P1 is UNRESOLVED on the pre-registered boundary.**
`QUALITY_v4.md` §7 item 2 is closed — the length confound is removed by
construction and the band table answers. **Nothing ships:** `SHIPPED` stays
`chat-v3d-aligned`, which beats this arm on the pre-registered headline 0.2969 to
0.2779.
**Predicts:** `docs/chat/PREDICTION_v5.md`, committed at `c26dbd5` before the arm
trained.
**Does not modify:** `docs/chat/QUALITY_v4.md`, which is closed. `SHIPPED` still
points at `chat-v3d-aligned` and v4's scorecard still reads 3 of 5; nothing in
this round edits either.

---

## 1. The one question this round exists to answer

`QUALITY_v4.md` §7 left five items. Four are directions and would each cost a
round. **Item 2 is different: it is a measurement the previous round was unable
to make**, and it turns out that item 3's open question is answered by the same
arm.

**§4.2 declined to answer.** The `topic` metric asks "does the reply contain the
word requested", so a longer reply has more chances to contain it. The honest
correction is to compare within a length band, and the band table could not:

| arm | 0–100 | 100–200 | 200–300 | 300+ |
| --- | ---: | ---: | ---: | ---: |
| `chat-v3d-aligned` (shipped) | — (0) | — (0) | 0.106 (207) | 0.087 (817) |
| `chat-v4a-short` | 0.125 (16) | 0.073 (846) | 0.064 (156) | 0.000 (6) |

The two arms shared one band, and the 156 candidates the short arm put in it are
its *tail* — the draws where it failed to stop. §4.2's own verdict: settling it
"needs an arm whose reply-length distribution overlaps the shipped one, which is
a fourth arm this round did not have the budget for."

**§7 item 3 named reply length as `story_dodge`'s last standing suspect.**
`story_dodge` — a narrative in reply to a prompt that asked for none — is the
worst number in the v4 package at 0.167 and 0.146, and its *first* suspect was
eliminated there: `chat-v4b-balance` cut story weight by a third and moved it
0.021, inside the noise. What was left was the hypothesis that at 161 characters
a story is cheap enough to be a plausible answer to anything.

One arm tests both, and §7 item 2 specifies it exactly: `stories_short` **with
the truncation target raised to ~300 characters**.

## 2. What was built

**`snnchat.shortform.Truncation`** turns the truncation target from three module
constants into a parameter, with two named values. The point of a parameter
rather than a copied module is that it makes the round's central claim *true*:
the two sources differ in truncation length and in nothing else, because the
sentence splitter, the topic rule, the request phrasings, the raw fraction and
the draw order are shared code.

`SHORT` is the default everywhere, so `stories_short.bin` repacks bit-identically
and every `QUALITY_v4.md` number stays reproducible.
`tests/test_snnchat.py::test_the_default_truncation_is_the_one_stories_short_was_packed_at`
is the gate on that, and four more gate the new target.

**The target was fixed by measurement, before the predictions were written.**
`short_story` stops at the last sentence boundary *before* its sampled target, so
kept length runs under the target — 161 against a midpoint of 187.5. The quantity
§7 item 2 wants on 300 is therefore the **kept-length mean**, not the midpoint,
because its stated purpose is to overlap the shipped arm's *reply* length.
`scripts/chat/calibrate_truncation.py` measured six ranges on 20,000 stories;
[242, 407] was taken as the one nearest 300.

| target | range | mid | kept mean | median | p90 | topic survival | conv mean | fits 256 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `SHORT` (v4a) | [140, 235] | 187.5 | **160.8** | 161 | 205 | 0.888 | 198.8 | **0.965** |
| `SHORT300` (new) | [242, 407] | 324.5 | **298.5** | 298 | 368 | 0.947 | 336.7 | **0.054** |

The `SHORT` row reproduces the figures `snnchat.shortform`'s docstring committed
before `chat-v4a-short` ever trained — 161 mean, 161 median, 88.9 % topic
survival — which is what makes the second row believable. The calibration read
`tinystories.txt` and no model, checkpoint, arm or held-out split.

## 3. The arm, and the three couplings named before it ran

`chat-v5a-short300` is **exactly** `chat-v4a-short`'s recipe with
`stories_short300` 0.40 in place of `stories_short` 0.40 — 14,000 steps, B = 160,
L = 256, lr 5e-4 cosine to a 5 % floor, `bot_loss_weight 3.0`, `align_frac 0.75`,
initialised from `chat-v2-anneal/ckpt_best.pt`. One variable.

It fills the cell the package was missing:

| | short-form shaper | `stories_topic` (whole stories) |
| --- | --- | --- |
| **~161-char replies** | `chat-v4a-short` | — |
| **~300-char replies** | **`chat-v5a-short300`** | `chat-v3d-aligned` (shipped) |

* **vs `chat-v4a-short`** — same shaper, isolates reply length. The causal test.
* **vs `chat-v3d-aligned`** — length-matched, different corpus. Makes §4.2's band
  table answer.

**Three things move with truncation length and cannot be held fixed. All three
were named in `PREDICTION_v5.md` §3 and §6 before any result existed.**

| source | conv | reply | strict | carries | **dose @0.74** | **requests / M chars** |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `stories_short` (v4a) | 241 | 161 | 0.139 | 0.840 | **0.877** | **4,168** |
| `stories_short300` (new) | 368 | 299 | 0.092 | 0.549 | **0.623** | **2,729** |
| `stories_topic` (shipped) | 757 | 726 | 0.039 | 0.248 | **0.260** | **1,327** |

1. **Window coverage falls, and it is meant to.** A 337-character conversation
   does not fit a 256-character window; the dose goes 0.877 → 0.623. The round
   accepts this because §7 item 1 already closed the dose as a lever — it went
   0.260 → 0.877 and the outcome went *down*.
2. **Topic survival rises 0.888 → 0.947, in this arm's favour**, on the exact
   metric being scored. A longer truncation keeps more of the story, so the
   requested word survives into the reply more often. Not separable from the
   manipulation; disclosed rather than controlled.
3. **Story-request density moves with reply length.** `stories_short` packs 4,168
   conversations per M characters against `stories_topic`'s 1,327 — so at the
   same 0.40 weight, **`chat-v4a-short` saw ~3.1× more story requests per
   training character than the shipped arm did**. The new source lands between
   them at 2,729.

**The ladder is monotone in all three at once.** Reply length 161 → 299 → 726 runs
with dose 0.877 → 0.623 → 0.260 and request density 4,168 → 2,729 → 1,327. So a
result that is merely monotone across the ladder is attributable to **the set**,
not to length. What the arm *can* do without that ambiguity is the **within-band**
comparison, which holds length fixed by construction — and that is precisely what
§7 item 2 asked for.

*(Confound 3 was not named in `QUALITY_v4.md` and applies retroactively to that
round's own comparison. It is recorded here rather than by editing that file.)*

## 4. Results

**Run 2026-08-06.** 14,000 steps in 41 minutes on an RTX 5060; the 55-minute
budget never fired, so this is a completed cosine and not a stopped one.
**Every number below is the `n = 1, λ = 0` row**, pinned by `PREDICTION_v5.md`
§7 item 1 before the checkpoint was scored. Raw JSON:
`experiments/chat/_quality/chat-v5a-short300.json`.

### 4.1 The arm hit its length target almost exactly, which is what makes the rest readable

| | `chat-v3d-aligned` | `chat-v4a-short` | **`chat-v5a-short300`** |
| --- | ---: | ---: | ---: |
| mean **candidate** chars | 300 | 166 | **295** |
| mean **selected reply** chars | 188.9 | 136.4 | **188.3** |

188.3 against the shipped arm's 188.9. **The length confound §4.2 left open is
closed** — not argued around, not corrected for, but removed by construction.
Everything in §4.2 follows from that and from nothing else.

### 4.2 The band table answers

| arm | 0–100 | 100–200 | 200–300 | 300+ |
| --- | ---: | ---: | ---: | ---: |
| `chat-v3d-aligned` | — (0) | — (0) | 0.106 (207) | 0.087 (817) |
| `chat-v4a-short` | 0.125 (16) | 0.073 (846) | 0.064 (156) | 0.000 (6) |
| **`chat-v5a-short300`** | — (0) | 0.000 (1) | **0.081 (307)** | **0.102 (716)** |

Both of the shipped arm's bands now hold hundreds of candidates for both arms.
**The comparison §4.2 declined to make is now available, and it splits:**

* in the **300+ band**, `chat-v5a-short300` is **ahead**, 0.102 to 0.087;
* in the **200–300 band**, it is **behind**, 0.081 to 0.106.

**That split is the honest result and it is not a tie being spun as one.** The two
bands are not equivalent. `PREDICTION_v5.md` §7 item 2 established before the run
that the 300+ band is a **point mass at the `--max-new 300` ceiling** — every
candidate in it is a reply that failed to stop — so the band where this arm wins
is the band of *truncated* replies, and the band where it loses is the one where
both models end their own sentences. Weighted by which band is the better proxy
for a reply a person would actually receive, **the 200–300 band is the more
meaningful cell, and the shipped arm wins it.**

### 4.3 On the raw column the new arm is ahead, and the dose fell to get there

| arm | propensity (1,024 candidates) | dose @0.74 |
| --- | ---: | ---: |
| `chat-v2-anneal` | 0.0518 | 0.046 |
| `chat-v3d-aligned` (shipped) | 0.0908 | 0.260 |
| `chat-v4a-short` | 0.0723 | 0.877 |
| `chat-v4b-balance` | 0.0645 | — |
| **`chat-v5a-short300`** | **0.0957** | **0.623** |

**0.0957 is the best topic propensity any checkpoint in this package has
produced**, and it was produced at a *lower* dose than the arm it beats by 32 %
(`chat-v4a-short`). That is the third independent confirmation of `QUALITY_v4.md`
§7 item 1 — the window-coverage dose is not the binding constraint — and the
first one where the dose moved *down* and the outcome moved *up*.

It is a **0.0049 gain over the shipped arm against a ±0.0180 error bar**, so on
its own it is inside the noise and is reported as such. What is not inside the
noise is the 0.0234 over `chat-v4a-short` at the same shaper.

### 4.4 `story_dodge` landed exactly on the pre-registered boundary

| arm | `story_dodge` | opener branch | " named " branch |
| --- | ---: | ---: | ---: |
| `chat-v3d-aligned` | 0.0833 | 0.0417 | 0.0417 |
| `chat-v4a-short` | **0.1667** | 0.1667 | 0.0000 |
| `chat-v4b-balance` | 0.1458 | 0.1250 | 0.0208 |
| **`chat-v5a-short300`** | **0.1250** | 0.1042 | 0.0208 |

**P1 predicted "below 0.125". The measurement is 0.1250.** It is not below it.
Under the band rule fixed in §4 of the pre-registration — below 0.125 → length is
a driver; at or above 0.146 → length is eliminated; **between → unresolved, and
"no story is to be told about the middle band"** — this round resolves P1 as
**UNRESOLVED**, and that is where it stays.

**The threshold was badly chosen and that is this round's fault, not the data's.**
`story_dodge` is measured over 48 probe draws, so it is quantised at 1/48 =
0.0208, and **0.125 is exactly 6/48** — a lattice point. "Below 0.125" therefore
required ≤ 5/48 and the arm returned exactly 6/48. A threshold placed between
lattice points (0.135, say) would have resolved cleanly. It was not, and the rule
is applied as written rather than nudged.

**What can be said without touching P1.** The pre-registration declared before the
run that `_is_story`'s `" named "` branch is length-sensitive and therefore biases
this arm's `story_dodge` **upward**. The **opener branch is length-independent**,
and on it the arm reads **0.1042 against `chat-v4a-short`'s 0.1667 and
`chat-v3d-aligned`'s 0.0417.** So: raising reply length from 161 to 299 removed
**about 37 % of the short-form corpus's excess story-dodging, and left it at
2.5× the shipped arm's.** Length is part of it. Length is not most of it.

And per §3, even that much is attributable to **{length, dose, request density}**
as a set, because all three move together across this ladder.

### 4.5 The prediction scorecard

Scored against `docs/chat/PREDICTION_v5.md` as written, at the row §7 item 1
pinned, not as one would like to have written it.

| | claim | outcome |
| --- | --- | --- |
| **P1** | `story_dodge` below 0.125 | **UNRESOLVED.** 0.1250 — exactly the boundary, and exactly a 6/48 lattice point. Reported as the pre-registered middle band demands. |
| **P2** | ≥ 150 candidates in 300+ and ≥ 100 in 200–300, so the band table answers | **CORRECT.** 716 and 307. The round's deliverable, and it did not depend on P1. |
| **P3** | in the 300+ band, beats 0.087 | **CORRECT as written** — 0.102. But §7 item 2 also required the 200–300 band be reported, and there it **loses**, 0.081 to 0.106. A win in the band that is a truncation artefact and a loss in the band that is not. |
| **P4** | beats `chat-v4a-short`'s 0.0723 propensity | **CORRECT.** 0.0957, and it also passes `chat-v3d-aligned`'s 0.0908 — which P4 explicitly declined to predict. |
| **P5** | `list (strict)` falls back toward 0.000 | **CORRECT, and it retro-scores v4.** 0.0000, with `mean_chars` back at 188.3. `chat-v4a-short`'s 0.033 was the 120-character cut-off, exactly as `QUALITY_v4.md`'s P4 warned. That warning is now confirmed rather than merely plausible. |
| **P6** | `fact` ~0.036, `identity` ≥ 0.938, `closed` degrades and is disqualified | **CORRECT.** `fact` 0.036, `identity` 1.000, `closed` 0.939 → **0.568** — the predicted censoring, and it is not read as a regression. |

**Four correct, one correct-but-split, one unresolved.** The two that carried the
round's purpose — P2 and P5 — both held.

### 4.6 The headline, and what ships

| arm | headline | topic | list (strict) | social | fallback |
| --- | ---: | ---: | ---: | ---: | ---: |
| `chat-v3d-aligned` (shipped) | **0.2969** | 0.0938 | 0.000 | 1.000 | 0.009 |
| `chat-v4a-short` | 0.2484 | 0.0469 | 0.033 | 1.000 | 0.027 |
| **`chat-v5a-short300`** | **0.2779** | 0.1094 | 0.000 | 1.000 | 0.009 |

**`chat-v5a-short300` does not beat the shipped arm on the pre-registered headline
and therefore does not ship.** 0.2779 to 0.2969. It recovers most of the ground
`chat-v4a-short` lost (0.2484) and beats it on the headline's dominant term, but
the headline weights `list` at 0.3 and this arm reads 0.083 there against the
shipped arm's 0.167. **`SHIPPED` is unchanged.**

### 4.7 The measurement this round was forbidden from promoting — and did not need to

`PREDICTION_v5.md` §5 committed in advance that `topic_dependence`'s
prompt-dependence delta would not be promoted to the objective, because
`QUALITY_v4.md` §4.7's rule is that a post-hoc measurement does not become the
objective by being the one the intervention won on.

| arm | dependence delta (total) | soda | alpaca | dolly |
| --- | ---: | ---: | ---: | ---: |
| `chat-v3d-aligned` | **+0.0619** | 0.0494 | 0.0664 | 0.0933 |
| `chat-v4a-short` | +0.0572 | 0.0383 | 0.0663 | 0.0923 |
| `chat-v5a-short300` | **+0.0542** | 0.0423 | 0.0580 | 0.0868 |

**The rule was never tested, because the arm lost on it.** `chat-v5a-short300` is
last of the three, and the shipped arm is first. That is worth recording for its
own sake: §4.7's finding — that the short corpus extracts more from a request —
**does not survive raising the truncation length**, and it was the one measurement
on which that corpus was unambiguously ahead.

## 5. What this round does not do, and what to distrust in it

1. **It does not resolve P1, and no amount of reading will make it.** 0.1250
   against a "below 0.125" threshold on a 1/48 lattice. The direction (down from
   0.1667, against a length arithmetic that pushes up) and the length-independent
   opener branch (0.1667 → 0.1042) both point the same way, and **neither is the
   pre-registered test.** They are reported in §4.4 as what they are.
2. **It cannot separate length from dose from request density.** §3's ladder is
   monotone in all three. The **within-band** results in §4.2 are the only ones
   free of this, which is why they are the deliverable.
3. **n = 1 per arm, no seeds, no control arms.** Same as every round in this
   package. A 0.0049 propensity gain over the shipped arm sits inside a ±0.0180
   bar and is not a result.
4. **Four disclosed defects sit under these numbers**, all in
   `PREDICTION_v5.md` §7 and all named before the numbers existed: the 300+ band
   is a point mass at the decoder ceiling; the raw-narrative share differs 24 % vs
   16 % **by character** between this arm and `chat-v4a-short` though
   `RAW_FRACTION` is identical; `stories_short300` hit the 320 M pack limit while
   `stories_short` exhausted the source, so the two are not drawn from the same
   story pool; and `closed` is censored at this reply length and must not be read.
5. **The one thing that would have gone wrong silently, did not.** Without §7 item
   1's pinned row, `story_dodge` for this arm reads **0.1250 at n = 1** and the
   sweep contains rows spanning far wider. The row was fixed before the JSON
   existed. That is the only reason §4.4 can say "unresolved" rather than pick.

## 6. What this leaves for the next round

1. **`QUALITY_v4.md` §7 item 2 is closed.** The band table answers. Reply length
   is no longer a confound anyone has to argue about, and the arm that closed it
   is committed and scorable.
2. **§7 item 3 is half-answered and should not be re-run as posed.** Reply length
   is *a* driver of `story_dodge` — 37 % of the excess on the length-independent
   branch — and is **not the main one**. Two named suspects remain and neither has
   been tested: the **ending-less truncation** (a corpus of stories that stop
   without resolving may teach that a narrative is an acceptable shape for any
   reply) and the **~50 % "short"/"little" request phrasings**. Of the two, the
   raw-narrative character share in §7 item 3 of the pre-registration is the
   cheapest to test, because it is a `RAW_FRACTION` change and no new shaper.
3. **Topic propensity is at 0.0957 and the dose is not what is holding it.** Three
   arms now, across dose 0.260 → 0.877 → 0.623, land between 0.072 and 0.096. The
   dose axis is closed twice over.
4. **The shipped arm's advantage is `list`, not `topic`.** 0.167 against 0.083, and
   it is the whole headline gap. Instruction following — not topicality — is what
   the next round should be aimed at, and `chat-v4b-balance`'s 0.36 instruction
   weight at strict 0.083 is the only point on that curve that has ever moved it.

## 7. Reproducing it

```bash
# the calibration that fixed the target (reads tinystories.txt, no model)
python scripts/chat/calibrate_truncation.py --stories 20000 \
    --candidates "261,438;250,420;235,395;228,383;220,370" \
    --out experiments/chat/_quality/truncation_calibration.json

# the corpus (adds files; stories_short.bin is untouched)
python scripts/chat/build_short_stories.py --name stories_short300 \
    --trunc short300 --limit-chars 320000000

# the dose, before anything trains
python scripts/chat/window_coverage.py stories_short300 stories_short stories_topic \
    --seq-len 256 --align-frac 0.74

# the arm
python scripts/chat/train.py --run-name chat-v5a-short300 \
    --init-from experiments/chat/chat-v2-anneal/ckpt_best.pt --no-resume \
    --max-steps 14000 --batch-size 160 --seq-len 256 --lr 5e-4 \
    --bot-loss-weight 3.0 --align-frac 0.75 --align-lookahead 1024 \
    --budget-minutes 55 \
    --mix stories_short300=0.40 soda=0.20 alpaca=0.15 tinystories=0.09 \
          persona=0.08 dolly=0.06 oasst1=0.02

# score it, and the question §4.2 exists to ask
python scripts/chat/quality.py --ckpt experiments/chat/chat-v5a-short300/ckpt_best.pt \
    --out experiments/chat/_quality/chat-v5a-short300.json
python scripts/chat/topic_by_length.py experiments/chat/_quality/chat-v*.json
```

The `--temperature-spread` companion run every v4 arm has is **deliberately not
run**: `QUALITY_v4.md` §7 item 0 records the candidate-pool temperature spread as
measured and dead, and re-running a dead intervention to fill a column would be
spending GPU hours to make a table look symmetric.
