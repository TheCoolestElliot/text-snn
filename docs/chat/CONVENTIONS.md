# Standing conventions for `docs/chat/` measurements

**Status: in force from 2026-08-06.** Applies to every `PREDICTION_*.md` and
`QUALITY_*.md` written after this date, and to any older number that is re-read
against a threshold.

**Not part of the Phase-1..5 research protocol.** Nothing here is pre-registered
research, no number `snnchat` produces is a reported figure, and this file adds a
reporting discipline to the chat side rather than claiming one.

Closed documents are **not** rewritten to comply. `QUALITY_v4.md` and
`QUALITY.md` stay as they were reported; `QUALITY_v5.md` §8 applies the rule as
an appended amendment and leaves §§1-7 untouched. That is the same treatment
`PREDICTION_v5.md` §7 gave its own audit findings, and for the same reason: a
record edited to accommodate a later result is not a record.

---

## 1. The rule

> **Wherever a rate is compared against a pre-registered threshold, report an
> interval with it.** The point estimate alone is not a verdict.

Concretely, every such number is written as

```
story_dodge 0.1250 (6/48), 95% CI [0.059, 0.247]
```

— the fraction, **its numerator and denominator**, and a 95 % interval. The
denominator is not decoration: it is what tells a reader the metric's resolution,
and the resolution is what P1 died of.

A comparison then has **three** outcomes, not two: `below`, `above`, and
**`unresolved`**. `unresolved` is a real verdict and is reported as one. It is
the right answer whenever the threshold lies inside the interval, *whichever side
the point estimate happens to fall on* — a fraction of 0.11 against a 0.125
threshold has not established anything if its interval runs to 0.14.

### Why this rule exists

`PREDICTION_v5.md` P1 asked whether `chat-v5a-short300`'s `story_dodge` came in
**below 0.125**. It measured **0.1250**, and `QUALITY_v5.md` §4.4 resolved P1 as
UNRESOLVED under the pre-registration's middle band.

Two separate defects produced that, and the fraction alone showed neither:

1. **The threshold was on the metric's lattice.** `story_dodge` is scored over
   the `list` and `fact` probes — 12 of the battery's 37 — at 4 seeds each, so it
   is a proportion over **48 draws**, quantised at 1/48 = 0.0208. **0.125 is
   exactly 6/48.** On that lattice the arm could not miss the threshold narrowly.
   It could only hit it exactly, and it did. A threshold at 0.135 would have
   resolved cleanly on the identical draws.
2. **48 draws could not have resolved it anyway.** The 95 % interval on 6/48 is
   **[0.059, 0.247]** — nearly a factor of four wide, spanning `chat-v3d-aligned`
   (0.083), `chat-v4b-balance` (0.146) and `chat-v4a-short` (0.167) all at once.
   Every `story_dodge` comparison in the v4 and v5 packages sits inside one
   interval. **The round read a coin flip as a boundary case**, and the only
   reason that was visible at all is that the flip landed exactly on the line.

The second defect is the serious one. The first is merely unlucky; the second
means the metric had been carrying threshold verdicts it was never able to
support, and would have gone on doing so silently had 0.1250 landed at 0.1042.

## 2. How to compute it

`src/snnchat/quality.py` implements this; do not re-derive it in a document.

```python
from snnchat.quality import rate_ci, resolves_against, wilson_interval

rate_ci(6, 48)              # -> {"k": 6, "n": 48, "rate": 0.125, "se": 0.0477,
                            #     "ci_low": 0.0586, "ci_high": 0.247,
                            #     "lattice": 0.02083, "z": 1.95996}
resolves_against(6, 48, 0.125)   # -> "unresolved"
```

`score()` emits `story_dodge_ci` and `story_dodge_opener_ci` on every row of
every sweep, so the interval is in the JSON whether or not a document quotes it.

**Wilson, not `p ± z·sqrt(p(1-p)/n)`.** At the rates these metrics live at
(0.03–0.17) and the sample sizes available, the normal approximation undercovers
badly and puts the lower bound below zero, which is not an interval. Wilson is
bounded in [0, 1] by construction, does not collapse to zero width at `k = 0`,
and is what Brown, Cai & DasGupta (2001) recommend for exactly this regime.

**One confidence level, everywhere: 95 %.** `snnchat.quality.Z95` is the single
constant. A table that mixes 90 % and 95 % intervals is worse than one with none.

### For a metric that is not 0/1

`list` scores `min(distinct hits, want) / want`, so it takes values in
{0, 1/3, 2/3, 1} and Wilson does not apply. Use the sample standard error,
`s / sqrt(n)`, and say which was used. Everything else in the battery
(`topic`, `social`, `identity`, `fact`, `story_dodge`, `list_strict`,
`fallback_rate`) is Bernoulli and takes Wilson.

## 3. What the interval does *not* cover

Stating this is part of the convention, because an interval invites more trust
than it has earned.

* **The probe battery is treated as fixed, not sampled.** It was frozen two
  rounds ago and is the same 37 prompts for every arm, so probe-set variability
  is not a source of error *for this estimand* — but the estimand is then "the
  rate over these 12 prompts", not "over story-free requests in general". A
  battery of 12 different prompts could give a materially different number and
  the interval says nothing about that.
* **Because the battery is fixed and seeds are allocated equally across it, the
  sample is stratified and the binomial interval is conservative.** The
  stratified variance is `binomial × (1 − Var(p_j)/(p̄(1−p̄)))`, which is never
  larger. Reporting the wider one is deliberate: a verdict that rides on an
  interval should ride on the conservative one.
* **`n = 1` per arm, so training-seed variability is not in it at all.** The
  interval is over the sampler, not over the run. Two arms trained from different
  seeds could differ by more than these intervals suggest, and nothing in the
  chat package has ever measured that.
* **It is a marginal interval, not a simultaneous one.** Six predictions read at
  95 % each are not a 95 % package.

## 4. Rules for writing the *next* `PREDICTION_*.md`

1. **State the threshold's denominator and lattice next to the threshold.** Write
   "below 0.135 (`story_dodge` runs over 12 probes × S seeds; at S = 4 the
   lattice is 0.0208)". If the author cannot state the denominator, the author
   does not yet know what is being predicted.
2. **Never place a threshold on a lattice point.** `story_dodge`'s denominator is
   `12 × seeds`, which is divisible by 8 — and therefore hits 0.125 exactly — if
   and only if the seed count is even. Put the bar strictly between two
   attainable values.
3. **State the prediction's power before running it**, per `CONTRIBUTING.md` §2.
   "What is the smallest sample at which this threshold could be missed rather
   than hit?" is now a question with a one-line answer, and it should appear in
   the pre-registration. The v5 answer was 48 draws and an interval of ±0.09, and
   writing that down would have moved the bar or the seed count before the arm
   trained.
4. **Fix the sample size in advance and honour it.** Enlarging a sample until an
   estimate crosses a threshold, then stopping, converts a null into a finding
   with no new evidence. `scripts/chat/story_dodge_resample.py` fixes its seed
   count in committed code and reports at that count whatever it says.
5. **A near-miss is still a miss** (`CONTRIBUTING.md` §3) and an interval does not
   change that. The interval decides whether the comparison was *decided*; it
   never rescues a bar that fired against the arm.

## 5. What was pre-committed for the P1 resample, before its numbers existed

Recorded here while `scripts/chat/story_dodge_resample.py` was still running, so
that the choice of sample size is on the record ahead of the result it produced.

| | value | why |
| --- | ---: | --- |
| seeds, `chat-v5a-short300` | **301** | the arm P1 is about. 12 × 301 = **3,612** draws; 3,612/8 = 451.5, so 0.125 is **off the lattice** (spacing 0.000825) |
| seeds, the three context arms | **101** | 1,212 draws each, also odd-seeded and off-lattice. They are not being tested against a threshold, only reported alongside |
| candidates per (probe, seed) | 16 | unused at the pinned `n = 1` row, but it sets the sampler's RNG stream and must match what the committed arms were drawn at |
| reading row | `n = 1, λ = 0` | the row `PREDICTION_v5.md` §7 item 1 pinned before the checkpoint was scored. Not swept |
| everything else | unchanged | same checkpoints, same frozen battery, same `_is_story`, same sampler settings, `--max-new 300`. **No retraining** |

**The odd seed count is the whole trick** and it is worth stating plainly: the
denominator `12 × seeds` is divisible by 8 exactly when the seed count is even,
so an even number of seeds — *however large* — leaves 0.125 an attainable value.
`story_dodge_resample.py` warns when the denominator it is about to use is
divisible by 8.

**The 4-seed rerun is a gate, not a result.** Seeds 0–3 of the resample must
reproduce the committed 48 draws character for character, or the script aborts:
`collect` derives a probe's sampling seed from its *position* in the probe tuple,
so drawing 12 of 37 probes would silently be a different experiment.
`collect(probe_index=...)` pins each probe's index in `PROBES` instead, and the
check confirms the enlarged sample is a strict **superset** of the one
`QUALITY_v5.md` reported. This is `CONTRIBUTING.md` §5's "make every probe
reproduce a number it did not produce", applied to a resample.

---

## 6. Where this is enforced

| | |
| --- | --- |
| `snnchat.quality.wilson_interval` | the interval |
| `snnchat.quality.rate_ci` | fraction + k/n + SE + interval + lattice |
| `snnchat.quality.resolves_against` | `below` / `above` / **`unresolved`** |
| `snnchat.quality.score` | emits `story_dodge_ci`, `story_dodge_opener_ci` |
| `snnchat.quality._is_story_opener` | the length-independent branch, split out so a document need not recompute it |
| `scripts/chat/story_dodge_resample.py` | the resample, its fixed seed count and its superset gate |
| `tests/test_snnchat.py` | `test_wilson_*`, `test_rate_ci_*`, `test_resolves_against_*`, `test_probe_index_*` |
| `CONTRIBUTING.md` §3 | the one-line pointer here |

## 7. Training rounds report the unintroduced-entity rate next to bpc

**In force from 2026-09-21, for every round that trains a checkpoint.** Next to
weighted bpc, report the **pool** unintroduced-entity rate, UER@200
(`snnchat.coherence`, scored by `scripts/chat/coherence_score.py` on the
round's `echo_holdout.py`-format pools, zero GPU), written as §1 requires and
always with its **qualifying fraction** — the share of replies that reach the
200-character window — because a rate over a shrinking subset flatters an arm
that writes shorter replies. Read it against two things and never alone: the
corpus reference from the same window and stoplist, **0.0230 (42/1824), 95 % CI
[0.0171, 0.0310]**, and the incumbent recipe's spread across training seeds,
**mean 0.2258, SD_seed 0.0147 (n = 4, ddof 1)** — not the pool's Wilson
interval, which treats 256 drafts of one prompt as independent
(`experiments/chat/_quality/coherence_v14.json`; regenerate with `python
scripts/chat/coherence_score.py --data-dir <tree with data/chat>`). Quote a
difference between two arms only with `uer_decomposition`'s two factors beside
it, because a reply with no definite subject scores clean and likelier text has
fewer of them. The instrument is named "unintroduced-entity rate (UER@200)" and
never "coherence": it has not been validated against a human rating and is blind
to a plain interleave. **UER must NEVER enter `snnchat/rerank.py`**, as a
filter, a tier or a term of the score. The precedent is `QUALITY_v8.md` §4 and
§7: once the echo partition selected on the matcher the `topic` column scores,
a gain on that column became definitional and stopped being evidence. A selector
that read UER would do the same to this instrument on the day it shipped.
`test_importing_coherence_does_not_import_torch` holds one direction of that
boundary (the instrument imports nothing from `snnchat.rerank`); the other
direction is held by review, and a reviewer who sees `coherence` imported in
`rerank.py` should stop the change there.
</content>
