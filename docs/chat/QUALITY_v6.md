# Chat quality, round 6 — a from-scratch run, the instruction-weight ladder, and closing out P1

**Date:** 2026-08-06
**Author:** Elliot Asher Caudill
**Status:** IN PROGRESS. This file is written incrementally as each part of the
round completes; sections below are appended, not edited once landed, mirroring
`QUALITY_v5.md`'s own treatment of `QUALITY_v4.md`.
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
session; it is formalised here, not improved on, and the caveat is the point of
writing it down. Two-sample z = (rate₂ − rate₁) / sqrt(se₁² + se₂²); |z| < 1.96
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

*(`chat-v6-inst-b` appended once it trains and scores.)*

## 5. Prediction scorecard

*(P0, P1a, P1b scored exactly as `PREDICTION_v6.md` states them — CORRECT /
FAILED / UNRESOLVED, no post-hoc rewording, per `docs/chat/CONVENTIONS.md` §4
rule 5: a near-miss is still a miss.)*

## 6. What ships

*(the `SHIPPED` decision, restated with the rule that produced it.)*

## 7. What this round leaves open

*(named explicitly, not silently dropped: `bot_loss_weight`/`align_frac`
isolation; the two `story_dodge` suspects — ending-less truncation and the
"short"/"little" phrasing bias; `spread_slow_poles` A/B, still flagged
research-side-only; cross-turn factual memory, architecturally out of reach at
this model size.)*
