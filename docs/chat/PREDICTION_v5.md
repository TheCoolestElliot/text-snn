# What the 2026-08-06 reply-length round predicts, written before it trains

This file exists so that the next round's result can be scored against a claim
made in advance rather than explained afterwards. `docs/chat/QUALITY.md` §4.1 did
it, `QUALITY_v4.md` §4.4 scored it honestly including the two predictions the
round was built around and which both failed, and this is the same discipline
applied to the arm below.

**Not part of the research protocol.** No control arms in the protocol's sense,
n = 1 per arm, and nothing here is a reported figure.

**Why a new document rather than a §8 of `QUALITY_v4.md`.** That file is closed:
`SHIPPED` still points at `chat-v3d-aligned` and its scorecard reads 3 of 5.
Folding this round into it would edit a closed record to accommodate a later
result, which is the failure mode `04_phase4_interim.md` §8 exists to prevent on
the SNN side. `QUALITY_v4.md` is not touched by this round except to be cited.

---

## 1. The one question, and why it is the only one worth 40 minutes

`QUALITY_v4.md` §7 leaves five items. Four of them are directions. **Item 2 is a
measurement that is currently impossible**, and items 2 and 3 turn out to be the
same arm:

* **§4.2 declined to answer.** `topic` is scored as "does the reply contain the
  word asked for", so a longer reply has more chances to contain it. The honest
  correction is to compare within a length band, and the band table could not:
  `chat-v3d-aligned` drew **not one** candidate under 200 characters and
  `chat-v4a-short` put 846 of 1,024 there. The only shared band was 200–300,
  where the 156 candidates the short arm contributed are its *tail*. §4.2's
  verdict was that settling it "needs an arm whose reply-length distribution
  overlaps the shipped one, which is a fourth arm this round did not have the
  budget for."
* **§7 item 3 names reply length as `story_dodge`'s remaining suspect.**
  `story_dodge` is the worst number in the v4 package (0.167 and 0.146) and its
  first suspect is **eliminated**: `chat-v4b-balance` cut story weight by a third
  and moved it 0.021, inside the noise. The named next suspect is that at 161
  characters a story is cheap enough to be a plausible answer to anything.

One arm tests both, and §7 item 2 specifies it: `stories_short` **with the
truncation target raised to ~300 characters**.

## 2. Fixing the target by measurement, before the prediction

`snnchat.shortform.short_story` stops at the last sentence boundary *before* a
sampled target, so kept length runs under the target it was given — 161 against
a midpoint of 187.5. The quantity §7 item 2 wants on 300 is the **kept-length
mean**, because its stated purpose is to overlap the shipped arm's *reply* length
distribution, so choosing the midpoint would overshoot by ~40 characters.

`scripts/chat/calibrate_truncation.py` measured six candidate ranges on 20,000
stories and [242, 407] was taken as the one nearest 300. **This read
`data/chat_raw/tinystories.txt` and nothing else — no model, no checkpoint, no
arm, no held-out split** — so there is no outcome in it to have peeked at. It is
disclosed here for the same reason `EXP_013` §2 discloses its two calibration
measurements: a number chosen before a pre-registration is still a choice.

| target | range | mid | kept mean | median | p90 | kept/mid | topic survival | conv mean | fits 256 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `SHORT` (`stories_short`, v4a) | [140, 235] | 187.5 | **160.8** | 161 | 205 | 0.858 | 0.888 | 198.8 | **0.965** |
| `SHORT300` (`stories_short300`, new) | [242, 407] | 324.5 | **298.5** | 298 | 368 | 0.920 | 0.947 | 336.7 | **0.054** |

The `SHORT` row reproduces the figures `snnchat.shortform`'s docstring committed
before `chat-v4a-short` trained (161 mean, 161 median, 88.9 % topic survival),
which is what makes the second row believable.

Frozen as `snnchat.shortform.SHORT300`. `SHORT` is the default everywhere, so
`stories_short.bin` still repacks bit-identically and every `QUALITY_v4.md`
number stays reproducible — `tests/test_snnchat.py::test_the_default_truncation_is_the_one_stories_short_was_packed_at`
is that gate.

## 3. The arm, and the two couplings it cannot avoid

`chat-v5a-short300` is **exactly** `chat-v4a-short`'s recipe with
`stories_short300` 0.40 in place of `stories_short` 0.40: 14,000 steps,
B = 160, L = 256, lr 5e-4 cosine to a 5 % floor, `bot_loss_weight 3.0`,
`align_frac 0.75`, `align_lookahead 1024`, initialised from
`chat-v2-anneal/ckpt_best.pt`. One variable, and it is truncation length.

That places it in the cell the package is missing:

| | short-form shaper (truncated, no ending, ~50 % "short" phrasings) | `stories_topic` (whole stories) |
| --- | --- | --- |
| **~161-character replies** | `chat-v4a-short` | — |
| **~300-character replies** | **`chat-v5a-short300`** ← this arm | `chat-v3d-aligned` (shipped) |

* **against `chat-v4a-short`** — same shaper, same weight, same everything;
  isolates reply length. This is the causal test for §7 item 3.
* **against `chat-v3d-aligned`** — length-matched, different corpus construction.
  This is what makes §4.2's band table answer.

**Two things move with the truncation length and cannot be held fixed, and both
are named here rather than discovered afterwards.**

1. **Window coverage falls, and it is meant to.** A 337-character conversation
   does not fit a 256-character window: `fits 256` goes 0.965 → 0.054. The dose
   `PREDICTION_v4.md` §1 measured at 0.877 will drop back toward
   `stories_topic`'s 0.260. This round **accepts** that, because §7 item 1 already
   closed the dose as a lever — it went 0.260 → 0.877 and the outcome went *down*.
   Giving some back is the price of the length manipulation, not a second
   intervention. The exact figure is measured before training and recorded in §6.
2. **Topic survival rises, 0.888 → 0.947, and it favours this arm.** A longer
   truncation keeps more of the story, so the word the request asks for survives
   into the reply more often. That is a **6.6 % relative improvement in the
   training signal for the exact metric being scored**, and any topic gain must
   be read against it. It is not separable from the manipulation — a longer
   truncation *is* more surviving topics — so it is disclosed rather than
   controlled.

**A third confound, which this round cannot break and which also applies
retroactively to `QUALITY_v4.md`'s own comparison.** Truncating stories changes
how many *story requests* a training character buys. `stories_short` packs
1,012,158 conversations into 243.8 M characters — 4,151 requests per M — against
`stories_topic`'s 528,616 in 398.4 M, or 1,327 per M. **At the same 0.40 mixture
weight, `chat-v4a-short` saw ~3.1× more "tell me a story" requests per training
character than the shipped arm did.** `stories_short300` will land between them,
so **story-request density and reply length move together across this ladder and
this arm cannot separate them.** If `story_dodge` tracks the ladder, the finding
is "one of length or request density", not "length". Stated now, so it cannot be
quietly dropped from the reading later.

## 4. The predictions

Written 2026-08-06, before the arm existed and before any window-coverage number
for the new source was computed.

**A hazard that has to be declared first, because it makes one prediction hard to
win and another easy.** `snnchat.quality._is_story` fires on either a story
opener *or* `" named " in text and len(text) > 80`. **The second branch is
length-sensitive**: a longer reply is arithmetically more likely to trip it. So
the length arithmetic pushes this arm's `story_dodge` **up**, against P1. Two
consequences, fixed now:

* a `story_dodge` **decrease** on a longer-replying arm is evidence *against* the
  arithmetic and is therefore strong;
* a `story_dodge` **increase** is ambiguous between behaviour and arithmetic, and
  will be reported as ambiguous rather than as a finding.
* The **opener-only sub-rate** (branch one alone, which is length-independent)
  will be reported alongside, as a **labelled diagnostic decomposition of an
  existing metric**. It is not a new objective, it does not enter the headline,
  and §4.7's rule applies to it in advance: it is being declared *before* the
  result, precisely so it cannot be promoted afterwards because the arm won on it.

**P1 — `story_dodge` comes back down.** `chat-v5a-short300` scores `story_dodge`
**below 0.125**, the midpoint of `chat-v4a-short`'s 0.167 and `chat-v3d-aligned`'s
0.083. The midpoint rule is fixed here so it cannot be chosen later. Reading:

* **below 0.125** → reply length (or request density, per §3) is a driver, and
  §7 item 3's suspect survives its first test;
* **at or above 0.146** (`chat-v4b-balance`'s value, where story *weight* was cut)
  → length is eliminated alongside story weight, §7 item 3's suspect list is
  **exhausted**, and the cause is something else the short-form corpus
  introduced — the ending-less truncation or the "short"/"little" phrasings are
  what would be left;
* **between 0.125 and 0.146** → unresolved, and reported as unresolved. No story
  is to be told about the middle band.

**P2 — the band table answers. This is the round's deliverable and it does not
depend on which way P1 goes.** `chat-v5a-short300` puts **≥ 150 of its 1,024
candidates in the 300+ band and ≥ 100 in the 200–300 band**, so both of the two
bands `chat-v3d-aligned` occupies hold ≥ 100 candidates for both arms and
`topic_by_length.py` stops declining to answer. This is a prediction about the
*instrument*, not about the model. If it fails, the round produced an arm that
still cannot be compared and the honest report says so.

**P3 — topicality, compared within a band.** In the **300+ band**,
`chat-v5a-short300`'s topic rate is **above `chat-v3d-aligned`'s 0.087**.
Rationale: §4.2's per-100-character reading put the short corpus 44 % ahead
(0.0437 vs 0.0303) and §4.7 found it extracts more from a request than any
checkpoint in the package. If either is real rather than an artefact of
saturation, a length-matched comparison is where it shows. **If it lands below,
the §4.1 decline was behavioural and not arithmetic**, the per-100-character
reading was flattery, and the short-form corpus is simply worse at topicality —
which would close §4.2 in the direction that costs this line of work its main
remaining defence.

**P4 — raw topic propensity, as an arithmetic check and not as a hope.**
`chat-v5a-short300` scores above `chat-v4a-short`'s **0.0723** on the
~1,024-candidate propensity column. This nearly follows from longer replies
alone, so it is weak by construction and is here to check that the arithmetic
behaves; if it *fails*, something behavioural is pushing the other way and P3 is
the number to read. **It is explicitly not predicted to beat `chat-v3d-aligned`'s
0.0908** — the dose falls this round and §7 item 1 already closed the dose as a
lever, so predicting a win there would be predicting against this project's own
evidence.

**P5 — `list (strict)` falls back, which retro-scores `QUALITY_v4.md`'s P4.**
v4's P4 warned that a `list (strict)` gain arriving with a length collapse is an
artefact of the metric's 120-character cut-off. This round **reverses** the length
change, so that warning predicts `list (strict)` returns toward
`chat-v3d-aligned`'s **0.000** from `chat-v4a-short`'s 0.033. If it does, v4's P4
is confirmed retrospectively and the 0.033 was never instruction following. If
`list (strict)` **holds at ≥ 0.033 with `mean_chars` back near the shipped arm's**,
then the short-form corpus bought a real instruction-following gain that v4 could
not tell apart from the artefact, and that is the one way this round produces
something worth keeping on that axis.

**P6 — what will not move, and one metric that must not be read.** `fact` stays
at ~0.036. `identity` stays ≥ 0.938. Cross-turn memory stays absent. And
**`closed` is expected to degrade** from `chat-v4a-short`'s ~1.00 at n ≥ 8,
because a ~298-character trained reply against a `--max-new 300` ceiling has
almost no room to emit `<|eot|>` before the harness cuts it. **That is censoring
by the scorer, not a regression in the model**, and `closed` is therefore
**disqualified as a quality metric at this reply length** — declared now, so it
cannot be reported either as a loss for this arm or as a win for `chat-v4a-short`.

## 5. How it will be scored, and the rule that is not being broken

Identical to every arm already in `experiments/chat/_quality/`:

```bash
python scripts/chat/quality.py --ckpt experiments/chat/chat-v5a-short300/ckpt_best.pt \
    --out experiments/chat/_quality/chat-v5a-short300.json
```

**The probe battery is frozen and was frozen two rounds ago.** It is not
extended, reworded or reweighted here. `--max-new` stays at **300** for this arm
as for every other, even though §4's P6 hazard is a direct consequence of that
ceiling — raising it for the arm that needs the room would make this arm's draws
incomparable to every arm it is being compared against, which is a worse defect
than the censoring. The `topics` stop list is untouched.

**Scored against `QUALITY.md` §5's metrics only: `topic`, `story_dodge`,
`list (strict)`.** The headline stays as pre-registered
(`0.5·topic + 0.3·list + 0.2·social − 0.2·fallback`) and is not redefined.

**`QUALITY_v4.md` §4.7's rule stands and is the one most at risk this round.**
`topic_dependence.py`'s prompt-dependence delta is the single measurement the
short-form corpus wins outright (+0.0035 for `chat-v4a-short`, the best in the
package). It is **not** promoted to the objective, not put in the headline, and
not used to rescue the arm if P1 and P3 both fail. It will be reported in a
labelled diagnostic column because withholding it would be its own kind of
dishonesty — but a post-hoc objective does not become the objective by being
the one the intervention won.

**What ships.** Nothing, unless `chat-v5a-short300` beats `chat-v3d-aligned` on
the pre-registered headline. `SHIPPED` stays where it is otherwise, and a
within-band topicality win that does not carry the headline is a **finding**, not
a promotion.

## 6. The dose, measured before training

Columns through `dose @0.74` are from `scripts/chat/window_coverage.py`; the
**`requests / M chars` column is `conversations / chars_train` from
`data/chat/manifest.json`**, which that script does not emit *(provenance
corrected — §7 item 6)*. Both were run after `stories_short300` was packed and
**before `chat-v5a-short300` started**, so that §3's coupling 1 is a number
rather than an expectation. **The predictions in §4 were written before this was
computed and are not conditioned on it** — that ordering is the only reason the
table can be quoted next to them.

| source | conv | reply | strict | carries | dose @0 | dose @1 | **dose @0.74** | **requests / M chars** |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `stories_short` (v4a) | 241 | 161 | 0.139 | 0.840 | 0.531 | 0.998 | **0.877** | **4,168** |
| `stories_short300` (**new**) | 368 | 299 | 0.092 | 0.549 | 0.316 | 0.731 | **0.623** | **2,729** |
| `stories_topic` (shipped) | 757 | 726 | 0.039 | 0.248 | 0.132 | 0.304 | **0.260** | **1,327** |

The measured reply length is **299**, against the 298.5 §2 calibrated for.

**The ladder is monotone in all three quantities at once, which is the design's
limitation stated as a number.** Reply length 161 → 299 → 726 runs with dose
0.877 → 0.623 → 0.260 and request density 4,168 → 2,729 → 1,327. The new arm sits
between the other two on every one of them. So a `story_dodge` or `topic` result
that is monotone across the ladder is attributable to **the set** {length, dose,
request density} and not to length alone — §3's third confound, now priced. What
the arm *can* do without that ambiguity is P2 and P3: a **within-band** topicality
comparison against `chat-v3d-aligned` holds length fixed by construction, which
is the whole reason §7 item 2 asked for this arm.

**One bookkeeping note.** `stories_short300` packs 318.7 M characters against
`stories_short`'s 242.9 M from the same 320 M-character source limit, because a
longer truncation keeps more of each story. At 0.40 mixture weight the run
consumes ~229 M characters from this source over 14,000 steps, so **neither
source repeats within the run** and the difference in packed size does not become
a difference in epochs seen. *(Amended — see §7 item 4. This note is left as
written; the correction is below it, not inside it.)*

## 7. Amendments, made after an audit and before any number was read

`chat-v5a-short300` was still training when §§1–6 were put through a structured
adversarial audit (40 agents, four lenses: post-hoc leakage, falsifiability,
unnamed confounds, code correctness; every finding then given to an independent
verifier instructed to refute it by default). **Seven findings survived
refutation.** They are recorded here as amendments with their reasons rather than
edited into §§1–6, because a pre-registration that is quietly rewritten is not one.

**Every amendment below was written and committed before `quality.py` was run on
this arm and before any of its numbers existed.** That ordering is the only thing
that makes them amendments rather than excuses, and it is checkable: the commit
that carries this section precedes the commit that carries
`experiments/chat/_quality/chat-v5a-short300.json`.

**1. The reading row is now pinned, and this is the one that mattered.** §4's
thresholds were quoted as if `story_dodge`, `list (strict)`, `identity`, `fact`
and `headline` were one number per arm. **They are not.** `quality.py` scores
every arm at **19 (n, λ) reranking settings** and each of those metrics is
computed from the *selected* reply, so each changes per row. Across that grid
`chat-v4a-short`'s `story_dodge` runs **0.021 – 0.396** — below P1's 0.125 band in
8 of 19 rows and at or above its 0.146 band in 9 — **on the very checkpoint that
anchors both bands.** `chat-v3d-aligned`'s headline is 0.297 at n=1, 0.262 at the
shipped setting and 0.308 at its own argmax, a 0.047 spread wider than every
inter-arm gap in the package. Left unpinned, **the row choice and not the model
would have decided P1**, after the numbers existed.

> **Reading rule, fixed now.** Every threshold in §4 — P1's 0.125/0.146 bands,
> P3's 0.087, P4's 0.0723, P5's 0.033/0.000, P6's `fact` ~0.036 and `identity`
> ≥ 0.938 — and the ship gate's headline comparison are read at the **n = 1,
> λ = 0** row, which is the row every figure quoted from `QUALITY_v4.md` §4.3 and
> §4.4 comes from. The shipped-decoder row (n = 8, λ = 0.6) and each arm's
> headline argmax are reported alongside as **labelled diagnostic columns and
> cannot resolve any prediction.**

**2. The 300+ band is a point mass, so P2 and P3 mean something narrower than
they say.** `--max-new 300` caps a candidate at 300 characters, so every candidate
in the "300+" band is *exactly* at the cap — `chat-v3d-aligned`'s 817 are all
cap-hits that never emitted `<|eot|>`. P2's "≥ 150 in the 300+ band" is therefore
a prediction that the arm **fails to stop**, not that it writes long replies, and
P3's within-band comparison is **between two sets of truncated replies**. That is
still a genuine length-matched comparison — both sides are censored identically,
which is the point — but it is a comparison of cap-hits and is to be reported as
one. **The 200–300 band is the one where both arms end their own replies**, and
P3 will be reported there as well as at 300+.

**3. The story source's raw-narrative share differs between the two arms by
character, not by record.** `RAW_FRACTION = 0.08` emits whole untruncated stories
at the same *rate* in both sources, but a whole story is ~880 characters against a
truncated 161 or 299 — so by **character** the raw, unconditioned narrative is
**~24 % of `stories_short` and ~16 % of `stories_short300`.** `chat-v4a-short`
therefore trained on half again as much bare narrative-with-no-request, per
character of story source, as this arm does. That is a plausible driver of
`story_dodge` in its own right, it moves with the manipulation, and **it was not
named in §3.** It is named now.

**4. The two sources are not drawn from the same story pool.**
`stories_short300` hit the 320 M-character pack limit (`chars_total` 320,000,158)
while `stories_short` **exhausted** `tinystories.txt` below it (243,844,359). So
`stories_short300` is built from roughly the **first 86 %** of that file —
869,777 conversations against 1,012,158 — and any ordering structure in its final
~14 % is confounded with truncation length. Repacking at a higher limit and
retraining was priced at ~45 minutes and **not done**; the confound is disclosed
instead, and it is judged small because TinyStories is not ordered by any
property this round measures. Recorded as a limitation, not as a null.

**5. §2's "six candidate ranges" was not reproducible from anything committed.**
The claim was true — six ranges were measured — but the committed script
hard-coded two, so a reader could verify "[242, 407] gives kept mean 298.5" and
could **not** verify that it was the nearest of six to 300, which is the part
that selects the number. `calibrate_truncation.py` now takes `--candidates`, and
the full six-row table is committed at
`experiments/chat/_quality/truncation_calibration.json`.

**6. §6's `requests / M chars` column was mis-attributed.** §6 credits the whole
table to `scripts/chat/window_coverage.py`, which emits no such quantity — the
column is `conversations / chars_train` from `data/chat/manifest.json`. The
numbers are correct; the provenance line was not.

**7. A false claim in committed code, corrected in place.**
`short_conversation`'s docstring said `trunc` "changes the *value* of the first
draw and not the number or order of draws, so two sources built at different
targets from the same seed stay comparable story-for-story." **The second half is
false**: `rng.randint` consumes a different number of Mersenne-Twister words for
different range widths, so the streams diverge after the first story. **Nothing in
this round's design rested on it** — the comparison is distributional over ~10⁶
conversations, not paired — but the claim was load-bearing-looking and is now
corrected in `src/snnchat/shortform.py` rather than deleted.

**What the audit did not find, which is worth as much.** It did not find that the
calibration read a model, a checkpoint, an arm or a held-out split; it verified
that `read_source` opens only `tinystories.txt`, that `ChatTokenizer` reads no
file, and that the val split is the last `VAL_FRACTION` by position while the
calibration takes the first 20,000 stories. §2's central disclosure holds.
