# Chat quality, round 6 — a from-scratch run, the instruction-weight ladder, and closing out P1

**Date:** 2026-08-06 – 2026-08-07 (session ran overnight)
**Author:** Elliot Asher Caudill
**Status: CLOSED** 2026-08-07. **P1 resolved and correct; P0 fails cleanly;
P1a is a caught pre-registration defect; P1b answered as designed; the §5
seed-variance measurement resolved when it was pre-registered as not needing
to (§8) — the round's most consequential finding.** `SHIPPED` is unchanged:
`chat-v3d-aligned/ckpt_best.pt`. This file was written incrementally as each
part of the round completed; sections were appended, not edited once landed,
mirroring `QUALITY_v5.md`'s own treatment of `QUALITY_v4.md`.
**Predicts:** `docs/chat/PREDICTION_v6.md` (written before `chat-v6-scratch`
launches).
**Does not modify:** `QUALITY.md`, `QUALITY_v4.md`, `QUALITY_v5.md`, or
`PREDICTION_v4.md`/`PREDICTION_v5.md`, all closed.

---

## 1. Is the incumbent actually ahead of its strongest challenger? (a free check, no GPU)

`QUALITY_v5.md`'s own verdict was that `chat-v3d-aligned` "beats this arm on the
pre-registered headline 0.2969 to 0.2779" and `SHIPPED` was left unchanged on
that basis. That comparison has never been read with an interval. Before
spending any GPU-hour building on top of "what's shipped," this section reads
it the way `docs/chat/CONVENTIONS.md` requires.

**Method.** `docs/chat/CONVENTIONS.md` gives Wilson CIs for a single rate
against a threshold. A headline is a linear combination of four component
rates over four *disjoint* probe groups (`0.5·topic + 0.3·list + 0.2·social −
0.2·fallback`, `src/snnchat/quality.py::score`) at the pinned `n=1, λ=0` row
(`report["baseline_picks"]`, i.e. the first row of `DEFAULT_GRID`, which
`compare_quality.py` and every prior round have used as the "plain sampler"
reference row). `topic`, `social` and `fallback` are Bernoulli and get a Wilson
`rate_ci`; `list` is not (values in {0, ⅓, ⅔, 1}) and gets a sample SE per
CONVENTIONS.md §2. The headline's SE is the components' SEs combined by their
fixed weights, **treating the four groups as independent** — an approximation,
not exact, because `fallback` is computed on the same draws as `topic`+`list`
(the union of `topic`/`list`/`fact`, per `_SUBSTANTIVE`), so a hit and a
fallback flag on the same draw are not independent events. This is the same
approximation an informal check made in the prior (uncommitted, unexported)
session; it is formalised here, not improved on, and the caveat is the point
of writing it down.

**A second, one-directional caveat, present everywhere in this document that
`social` reads 20/20 or 0/20 (which is every table in §1, §2 and §4):** the
component `se` used in the propagated headline SE is a Wald
`sqrt(p(1-p)/n)`, which is exactly `0` at `p = 0` or `p = 1` — the same
degeneracy `docs/chat/CONVENTIONS.md`'s own case for Wilson over Wald exists
to avoid, re-introduced here because the headline's propagation formula needs
a single `se` number per component, not a full interval. A `social` term of
`0` makes the propagated headline SE an **underestimate**, never an
overestimate — so it does not manufacture a resolved verdict anywhere in this
document; it can only have made an unresolved call look slightly more
confident than it should. Every verdict in §1, §2 and §4 was already
`unresolved` under the understated SE, so correcting it would only widen
margins that were already wide enough to not matter — but it is a real
direction of bias and is recorded here rather than left implicit.

Two-sample z = (rate₂ − rate₁) / sqrt(se₁² + se₂²); |z| < 1.96
is `unresolved`, matching `resolves_against`'s own boundary.

**Result**, on the committed `experiments/chat/_quality/chat-v3d-aligned.json`
and `chat-v5a-short300.json`, `n=1, λ=0`:

| component | `chat-v3d-aligned` (shipped) | `chat-v5a-short300` (challenger) | diff (challenger − shipped) | SE | z | verdict |
| --- | ---: | ---: | ---: | ---: | ---: | :-- |
| `topic` | 0.0938 (6/64) [0.044, 0.190] | 0.1094 (7/64) [0.054, 0.209] | +0.0156 | 0.0533 | 0.292 | unresolved |
| `list` | 0.1667 (n=20), se 0.0820 | 0.0833 (n=20), se 0.0534 | −0.0834 | 0.0979 | −0.852 | unresolved |
| `social` | 1.0000 (20/20) | 1.0000 (20/20) | 0.0000 | 0 / 0 | — | **tied at ceiling — not a comparison** (both arms saturate 20/20; a Wald SE of 0 here is a boundary artefact, not evidence of certainty) |
| `fallback` | 0.0000 (0/112) [0, 0.033] | 0.0089 (1/112) [0.002, 0.049] | +0.0089 | 0.0089 | 1.000 | unresolved |
| **headline** | **0.2969** | **0.2779** | **−0.0190** | **0.0397** | **−0.479** | **unresolved** |

**Verdict: UNRESOLVED.** The 0.0190 headline gap `QUALITY_v5.md` read as a loss
for `chat-v5a-short300` does not survive a 95 % read (z = −0.479, far inside
±1.96); every component gap is individually unresolved too, and `social` is a
tie at the metric's ceiling for both arms. **This does not mean the arms are
equal — it means 20 draws per component cannot tell them apart, in either
direction.** `chat-v3d-aligned` remains `SHIPPED` because the standing rule is
that an unresolved comparison does not displace the incumbent (§0 below), not
because this round found it superior. This is the first time that distinction
has been written down rather than implied by a bare fraction.

**What this decides.** Per the standing rule, Phase 1's from-scratch run trains
on the **incumbent's** corpus recipe (`stories_topic` at 0.40, `chat-v3d-aligned`'s
mixture), not the challenger's, because the comparison did not resolve in the
challenger's favour. See `PREDICTION_v6.md` §0.

**What this does not decide.** This says nothing about `story_dodge`'s P1
boundary case specifically (`chat-v5a-short300` vs. its own pre-registered
threshold, not vs. another arm) — that is Phase 4's resample, run at 301 seeds
because 20-and 48-draw samples are visibly too small to resolve differences
this size, as this section's own numbers now show directly.

---

## 2. The from-scratch run: `chat-v6-scratch` — P0 FAILS

**`chat-v6-scratch` completed its full 84,000-step cosine schedule** (it did
not need its `--budget-minutes 240` ceiling — `wall_clock_s = 14322.3`,
≈3.98 h, just under budget), fresh-init, on `chat-v3d-aligned`'s exact recipe
(`PREDICTION_v6.md` §0/§2): 4,417,637 params (unchanged architecture), peak
VRAM 3.831 GiB, weighted held-out val bpc 1.2399 under its own training
mixture (not comparable to `chat-v2-anneal`'s 1.1795 — different mixture, per
`snn-chat-subsystem`'s standing rule).

**Result at the pinned `n=1, λ=0` row**, same method as §1:

| component | `chat-v3d-aligned` (shipped) | `chat-v6-scratch` (fresh-init) | diff | SE | z | verdict |
| --- | ---: | ---: | ---: | ---: | ---: | :-- |
| `topic` | 0.0938 (6/64) | 0.1094 (7/64) | +0.0156 | 0.0533 | 0.292 | unresolved |
| `list` | 0.1667 (n=20), se 0.0820 | 0.0500 (n=20), se 0.0500 | −0.1167 | 0.0960 | −1.215 | unresolved |
| `social` | 1.0000 (20/20) | 1.0000 (20/20) | 0.0000 | — | — | tied at ceiling |
| `fallback` | 0.0000 (0/112) | 0.0089 (1/112) | +0.0089 | 0.0089 | 1.000 | unresolved |
| **headline** | **0.2969** | **0.2679** | **−0.0290** | **0.0393** | **−0.738** | **unresolved** |

**Verdict on P0: FAILS**, cleanly, on the pre-registered threshold —
`PREDICTION_v6.md` §3 required headline > 0.2969 *and* a resolved margin;
`chat-v6-scratch`'s point estimate (0.2679) does not clear 0.2969 at all, so
this is a miss on the first, simpler condition, not one that needed the
significance machinery to decide (`docs/chat/CONVENTIONS.md` §4 rule 5: a
near-miss is still a miss, and this is not even a near-miss on the point
estimate — though the *margin* against the incumbent is itself statistically
unresolved, so "worse" is not established either, only "not better"). **The
hypothesis that stale pre-training was suppressing quality is not supported by
this experiment.** A fresh 84,000-step run on the identical corpus, mixture,
and alignment settings a 14,000-step anneal was already using lands at a
headline statistically indistinguishable from the anneal's, and if anything
lower on point estimate, mainly on `list` (0.050 vs 0.167 — but both are 1 or
3 hits out of 20 draws, and neither number resolves against the other).

**One measurement moved in the encouraging direction, and it is not in the
headline.** Prompt dependence (`prompt_dependence`, no sampler, no lexicon —
the "number to believe" per `scripts/chat/quality.py`'s own docstring) is
**0.0729 for `chat-v6-scratch` against 0.0620 for `chat-v3d-aligned`** — higher,
in the direction that matters, on a metric immune to decoder tuning. This is
one arm, not a trend, and it does not move the ship decision (P0's threshold
was fixed on the sampler-based headline, not on this number), but it is worth
naming: the fresh-init run's replies depend somewhat *more* on their own
prompt than the anneal's do, even though the sampler-scored battery does not
show it. `closed_rate` (0.480 vs 0.493) and `story_dodge` (0.0833, identical
to four decimal places on both arms) show no difference at all.

**Decision: `SHIPPED` is unchanged — stays `chat-v3d-aligned`.** Per the
pre-registered rule, this is not a comparison that needed the unresolved-margin
clause invoked; the point estimate itself did not clear the bar. Committed as
a plain negative result, matching two of this project's last three rounds.

## 3. P1: does `story_dodge` actually clear 0.125? — RESOLVED, below

`docs/chat/PREDICTION_v5.md` P1 asked whether `chat-v5a-short300`'s
`story_dodge` comes in below 0.125. At the committed 48 draws it measured
**exactly** 0.1250 — a lattice point the metric could not miss narrowly — and
`QUALITY_v5.md` §4.4 could only report UNRESOLVED. `docs/chat/CONVENTIONS.md`
§5 pre-committed the fix before this round's numbers existed: resample at 301
seeds (odd, so the denominator 3,612 is off the lattice), same checkpoint,
same frozen battery, same pinned `n=1, λ=0` row, no retraining.

**Superset check passed**: all 48 committed draws reproduced character for
character, so this is confirmably a superset of `QUALITY_v5.md`'s measurement,
not a second, differently-seeded one that happens to be nearby.

**Result:** `story_dodge` = **0.1035** (374/3612), **95 % CI [0.0940, 0.1139]**
— the entire interval sits below 0.125. `resolves_against(374, 3612, 0.125)`
→ **`below`**. Opener-only branch (length-independent):
0.0756 (273/3612), 95 % CI [0.0674, 0.0847].

**P1: RESOLVED, below — CORRECT**, after two rounds unresolved. Took 65.5
minutes on the GPU, no retraining. `docs/chat/CONVENTIONS.md`'s own thesis
(the 2026-08-06 unresolved verdict was a resolution failure, not a fact about
the model) is now confirmed on the exact case it was written to fix, not just
argued in the abstract. Written to
`experiments/chat/_quality/story_dodge_resample_v5a.json`.

**The three context arms** (101 seeds each, 1,212 draws, not tested against a
threshold — reported for reference, not as a P1-style verdict):

| arm | `story_dodge` | 95% CI | opener-only | vs. 0.125 (reference only) |
| --- | ---: | --- | ---: | :-- |
| `chat-v3d-aligned` (shipped) | 0.0611 (74/1212) | [0.0489, 0.0760] | 0.0470 | below |
| `chat-v4b-balance` | 0.1304 (158/1212) | [0.1126, 0.1505] | 0.1139 | unresolved |
| `chat-v4a-short` | 0.1733 (210/1212) | [0.1530, 0.1956] | 0.1502 | above |

At this resolution the three arms now separate cleanly from each other
(`chat-v3d-aligned`'s upper bound, 0.0760, sits below `chat-v4b-balance`'s
lower bound, 0.1126, which sits below `chat-v4a-short`'s), even though
`chat-v4b-balance` itself still straddles the *specific* 0.125 threshold —
this is a real, informative interval landing near a boundary, not a
resolution failure like the original 48-draw measurement, since 1,212 draws
puts the lattice at 0.00083. `chat-v3d-aligned` remains the best of the four
arms measured on `story_dodge` at every sample size this project has used.
Written to `experiments/chat/_quality/story_dodge_resample_context.json`.

## 4. The instruction-weight ladder: `chat-v6-inst-a`, `chat-v6-inst-b`

**A defect in `PREDICTION_v6.md` §4, found before scoring, corrected here
rather than silently worked around.** P1a's attribution rule was written
against `chat-v4a-short` topic 0.0723 and `chat-v4b-balance` topic 0.0645 —
those numbers do not match the committed data. Read directly from the JSON
files (not a prose recap, the same discipline §4's own table used for the
mixture weights): **both arms score topic = 0.0469 (3/64) at `n=1, λ=0`,
identical to four decimal places.** The reference values in the
pre-registration were a recall error, not a re-derivation, and the rule as
written cannot be applied — it asks which of two numbers `chat-v6-inst-a`
lands nearer to, and the two numbers are the same number. This is reported as
a pre-registration defect, per the standing practice of amending rather than
silently reinterpreting, and P1a is scored on the plain data below instead of
through the broken conditional.

**`chat-v6-inst-a` result** (story weight pinned at `v4a`'s 0.40, instruction
raised to `v4b`'s exact 0.36, `soda` absorbing the difference), all at the
pinned `n=1, λ=0` row:

| arm | `stories_short` | instruction | `topic` | `list` | `list_strict` |
| --- | ---: | ---: | ---: | ---: | ---: |
| `chat-v4a-short` | 0.40 | 0.23 | 0.0469 (3/64) | 0.0833 | 0.0333 |
| `chat-v4b-balance` | 0.26 | 0.36 | 0.0469 (3/64) | 0.1833 | 0.0833 |
| `chat-v6-inst-a` | **0.40** | **0.36** | **0.0938 (6/64)** | **0.2167** | **0.0500** |
| `chat-v3d-aligned` (shipped) | — | 0.23 | 0.0938 (6/64) | 0.1667 | 0.0000 |

Holding story weight fixed at 0.40 and raising instruction weight alone to
0.36 **doubled the topic point estimate relative to both `v4a` and `v4b`**
(0.0469 → 0.0938) and lifted `list`/`list_strict` above `v4a`'s (though not
cleanly above `v4b`'s `list_strict`). **None of these differences resolve**:
two-sample checks (same method as §1/§2) put `chat-v6-inst-a` vs. each of
`v4a-short`, `v4b-balance`, and `chat-v3d-aligned` all at `|z| < 1.4` on both
`topic` and `list` — every comparison in this table is `unresolved` at the
standard 4-seed battery's 64-topic/20-list sample sizes.

**The honest reading:** the point estimates are consistent with instruction
weight (not the story cut) driving `list_strict`'s gain, and even hint that
holding story weight fixed does *better* on topic than either confounded
arm — but "consistent with" is not "established". This ladder needed the
same treatment `story_dodge` got in §3 to actually resolve, and did not get
it (a resample at these `n` values would cost per-arm what §3's context
arms cost, ~20+ min each, not budgeted this round).

**This round's own pattern, stated plainly:** every topic/list comparison in
`QUALITY_v6.md` — §1, §2, and this section — has been `unresolved` at the
standard battery's sample sizes. The one comparison that *did* resolve
(`story_dodge`, §3) is the one place this round spent a resample on it. This
is a structural observation about the measurement tool, not about any one
arm, and belongs in §7 as a concrete next step: the standard 4-seed
`quality.py` battery is underpowered for `topic`/`list`/headline comparisons
in general, the same way it was for `story_dodge` before `PREDICTION_v5.md`'s
P1 forced the issue.

**`chat-v6-inst-b` result** (same shape, story weight still pinned at 0.40,
instruction pushed further to 0.45):

| arm | instruction | `topic` | `list` | `list_strict` | `social` |
| --- | ---: | ---: | ---: | ---: | ---: |
| `chat-v6-inst-a` | 0.36 | 0.0938 (6/64) | 0.2167 | 0.0500 | 1.0000 (20/20) |
| `chat-v6-inst-b` | **0.45** | 0.0938 (6/64) | **0.1000** | **0.0167** | **0.9000 (18/20)** |

**P1b answer: plateau/reverse, not a continued rise.** `topic` sat exactly flat
(6/64 both arms); `list_strict` fell back from 0.050 to 0.017, below even
`chat-v4a-short`'s 0.033; `social` dropped to 0.900, matching `chat-v4b-balance`'s
cost exactly. None of the `chat-v6-inst-a` vs. `chat-v6-inst-b` differences
resolve either (`|z| < 1.5` on `topic`/`list`/`social`), but the *direction* —
every one of them moved the wrong way at once, at the same story weight, from
one clean step up the instruction axis — is consistent with a real cost rather
than noise landing badly, and mirrors this project's other closed lever
(window-coverage dose, `snn-chat-window-dose-ceiling`): pushing an axis 25%
further (0.36→0.45, a smaller step than dose's 3.4×) was enough to reverse the
gain it was chasing. **Instruction weight around 0.36, on a corpus that also
holds story weight at 0.40, is now this project's best-supported point on this
axis** — not because 0.45 is disproven (the reversal is itself unresolved),
but because it is the only point with anything pointing toward it rather than
away. `chat-v6-inst-b`'s reply quality also degraded qualitatively: its
`hello` demo reply ("Hello there, what can I add it for a natural language?")
reads less coherent than `chat-v6-inst-a`'s ("Hi there. What can I do for you?"),
consistent with the `social` drop. Prompt dependence (no sampler): 0.0670,
between `chat-v3d-aligned`'s 0.0620 and `chat-v6-scratch`'s 0.0729.

## 5. Prediction scorecard

| # | statement | verdict |
| --- | --- | --- |
| P0 | fresh-init `chat-v6-scratch` beats `chat-v3d-aligned`'s headline (>0.2969) with a resolved margin | **FAILS** — 0.2679, below the threshold outright (§2) |
| P1 | `chat-v5a-short300`'s `story_dodge` comes in below 0.125 | **CORRECT, RESOLVED** — 0.1035 (374/3612), 95% CI [0.0940, 0.1139], entirely below (§3) |
| P1a | attribution rule for `chat-v6-inst-a`'s `list_strict` gain (nearer `v4a`'s topic or `v4b`'s) | **DEFECT — not applicable as written.** Reference values (0.0723/0.0645) do not match the committed data (both 0.0469); reported on the plain data instead (§4) |
| P1b | does `list_strict` keep rising (0.23→0.36→0.45) or plateau/reverse | **ANSWERED: reverses.** Both pre-registered outcomes counted as a result; this is the one that happened (§4) |
| §5 (unnumbered — explicitly not a threshold test) | measure training-seed variance via `chat-v6-inst-a-s1` (identical recipe, `--seed 1`) | **MEASURED: 0.099 headline swing, RESOLVED (z=−2.406)** despite every individual component being unresolved (§8) |

**Scorecard: 1 resolved-correct (P1), 1 clean fail (P0), 1 pre-registration
defect caught before scoring (P1a), 1 answered-as-designed (P1b), 1
variance measurement that came back resolved when it was explicitly
pre-registered as not needing to (§8).** No prediction was silently reworded
after its number existed; P1a's defect is reported as a defect, not quietly
patched.

## 6. What ships

**`SHIPPED` is unchanged: `chat-v3d-aligned/ckpt_best.pt`.** Every new arm
this round was checked against the incumbent's headline (0.2969) with the same
propagated-SE method:

| arm | headline | vs. incumbent | verdict |
| --- | ---: | ---: | :-- |
| `chat-v6-scratch` | 0.2679 | z = −0.738 | unresolved, and below threshold regardless (§2) |
| `chat-v6-inst-a` | 0.3101 | z = +0.294 | unresolved — highest point estimate of the round, not a resolved win |
| `chat-v6-inst-b` | 0.2533 | z = −1.035 | unresolved |

None of the ladder arms had a formal pre-registered ship threshold the way
`chat-v6-scratch` did (§4 was framed as attribution, not a ship candidate) —
this table applies the standing rule (§0: an unresolved comparison does not
displace an incumbent) uniformly anyway, rather than letting an arm's numeric
rank stand in for a decision it was never tested against. `chat-v6-inst-a`'s
0.3101 is the best point estimate any arm has ever scored at `n=1, λ=0` in
this project's history — worth a follow-up round with a resample behind it,
not a promotion on the strength of a single unresolved measurement.

## 7. What this round leaves open

- **The standard 4-seed `quality.py` battery is underpowered for
  `topic`/`list`/headline comparisons, not just for `story_dodge`.** Every
  such comparison in this document — §1, §2, and §4 — came back `unresolved`;
  the one comparison that *did* resolve is the one this round spent a
  301-seed resample on. A generic resample tool for `topic`/`list` (parallel
  to `story_dodge_resample.py`, currently hard-coded to the `list`+`fact`
  kinds) does not exist yet and would be the highest-leverage infrastructure
  investment for a future round, ahead of any specific new arm.
- **`chat-v6-inst-a`'s ~0.36 instruction weight is this project's
  best-supported point on that axis**, not proven superior to the incumbent,
  but the only new arm this round with more evidence pointing toward it than
  away. A resampled, properly-powered head-to-head against `chat-v3d-aligned`
  is the natural next step.
- **`bot_loss_weight`/`align_frac` isolation** — still always varied together
  since `chat-v3b`, never attempted this round (named in `PREDICTION_v6.md`
  §4 as out of scope, priced above a plain 40-minute arm).
- **The two named `story_dodge` suspects** — ending-less truncation, and the
  "short"/"little" phrasing bias — still untested. Now that `story_dodge` can
  actually be measured with a resample, these are cheaper to chase than they
  were: the resample tooling exists and works.
- **`spread_slow_poles` A/B** (log-uniform vs. committed uniform `beta_s`
  init, matched seeds) remains flagged research-side-only, per `README.md`'s
  own boundary — not acted on from the chat side this round either.
- **Cross-turn factual memory / recall** — architecturally out of reach at
  4.4M parameters and a ~47-character neuron horizon. Nothing tried in five
  rounds has moved it and nothing at this scale will.
- **A stretch arm was deliberately not attempted this round.** The originally
  planned candidate (a lowered `RAW_FRACTION` for `story_dodge` via
  `snnchat.shortform`) was never pre-registered with its own resample budgeted,
  and improvising an unregistered experiment in the session's closing hours
  would trade this project's core discipline for one more data point. Named
  here as a next-round candidate instead.

## 8. Training-seed variance: the headline moves 0.099, and RESOLVES

Pre-registered in `PREDICTION_v6.md` §5, after remaining budget made building
an underpowered resample tool the wrong move (§7, advisor-checked power
calculation). Not a ship test — `chat-v6-inst-a-s1` is `chat-v6-inst-a`'s
identical recipe at `--seed 1` instead of `0`; both initialise from the same
`chat-v2-anneal` checkpoint, so only training batch order differs.

| component | `chat-v6-inst-a` (seed 0) | `chat-v6-inst-a-s1` (seed 1) | diff | SE | z | verdict |
| --- | ---: | ---: | ---: | ---: | ---: | :-- |
| `topic` | 0.0938 (6/64) | 0.0625 (4/64) | −0.0313 | 0.0474 | −0.661 | unresolved |
| `list` | 0.2167 (n=20) | 0.0500 (n=20) | −0.1667 | 0.0984 | −1.694 | unresolved |
| `social` | 1.0000 (20/20) | 0.8500 (17/20) | −0.1500 | 0.0798 | −1.880 | unresolved |
| `fallback` | 0.0089 (1/112) | 0.0268 (3/112) | +0.0179 | 0.0177 | 1.011 | unresolved |
| **headline** | **0.3101** | **0.2109** | **−0.0992** | **0.0412** | **−2.406** | **RESOLVED — seed 1 lower** |

**Every individual component is unresolved. The headline is not.** All four
components moved the same direction at once (topic down, list down, social
down, fallback up — every one costs the headline), and the headline pools
that correlated shift into a signal none of its parts carries alone. This is
the mechanism, not a contradiction: §1's independence caveat means the
propagated SE is an *approximation* of the headline's true variance, not a
recomputation of a quantity already measured directly — and the headline
*is* measured directly here, from the same 37-probe draw, the same way every
other headline in this document was.

**The magnitude is the finding.** A **0.099** swing from changing nothing but
which batches 14,000 steps of fine-tuning happened to see is larger than
every arm-vs-arm difference this round measured (§1: 0.019, unresolved; §2:
0.029, unresolved; §4: 0.013 to 0.044, all unresolved) — **and this is the one
that resolves.** Two conclusions, both load-bearing for how to read this
project's history:

1. **`chat-v6-inst-a`'s 0.3101 does not, on its own, support ranking it above
   `chat-v3d-aligned`'s 0.2969** (§6) — not only because that specific
   comparison was already unresolved, but because a same-recipe reroll alone
   can move the headline by more than that gap.
2. **Every arm comparison in `QUALITY.md` through this document — five
   rounds, dozens of pairwise reads — was made at `n = 1` training seed per
   arm**, exactly the limitation `docs/chat/CONVENTIONS.md` §3 states in the
   abstract ("training-seed variability is not in it at all... two arms
   trained from different seeds could differ by more than these intervals
   suggest"). This measurement turns that abstract statement into a concrete
   number for the first time: **the training-seed contribution to headline
   variance is large enough, on its own, to resolve against a same-recipe
   reroll** — meaning a resolved-looking arm-vs-arm gap smaller than ~0.10
   cannot yet be attributed to the recipe change with confidence, since a
   same-recipe seed change alone can produce a gap that size. No comparison
   in this project's history has been re-run at a second seed before this.

**One data point, not a distribution — say that plainly.** `n = 2` seeds
establishes that seed variance is *nonzero and large*, not what its
distribution is, whether 0.099 is typical or an outlier, or whether it holds
at other recipes. Do not treat 0.099 as "the" seed-noise floor for this
package; treat it as a lower bound on how large that floor can be, established
once.

**A genuine bonus, not scored**: `chat-v6-inst-a-s1` hit the same
gradient-norm divergence shape `BUILD_NOTES.md` §9 already flagged (loss
`nan` at step 4859, six seeds and two arms after the first sighting),
recovered automatically via the trainer's rollback+seed-bump+halved-LR
mechanism (`log.jsonl`), and finished its full 14,000 steps regardless. A
second occurrence on the *same* recipe as a run that did not diverge
(`chat-v6-inst-a`, seed 0, clean) is a small additional data point that the
divergence is seed-triggered, consistent with `EXP_009`'s general shape
(research side) — not chased further here, outside this round's scope.

This changes the round's headline framing, not its `SHIPPED` decision:
`chat-v3d-aligned` was already the incumbent on an unresolved-does-not-displace
basis (§6); it still is. What changes is the confidence attached to *every*
number in this document and its five predecessors — now stated once, plainly,
rather than left as CONVENTIONS.md's abstract caveat.
