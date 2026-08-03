# EXP_006 — Do ~3 % of channels carry the horizon?

**Pre-registered:** 2026-08-03, before any ablated number was computed.
**Status:** **CLOSED** 2026-08-03. **All six predictions held**; the reading they
support is narrower than `EXP_004` §10.11 item 4's, and §9.6 says why.
**Phase:** 4 (controlled experiments). The causal question `EXP_004` §10.5 left
open and §10.11 item 4 called "the highest-information single run available".

**A note on which phase this belongs to.** `EXP_004` §10.5 calls this "a Phase-5
experiment". `01_reconnaissance.md` §8.2 defines Phase 5 as *final training and
docs* and Phase 4 as *pre-registered, single-variable, ≥3-seed controlled
experiments*, which is what this is. It is filed as Phase 4 and the
inconsistency is recorded here rather than resolved by relabelling `EXP_004`.
Nothing about the design depends on which name it carries.

---

## 1. Why this experiment

`EXP_004` §10.5 found that `beta_s`, initialised at 0.95 for every channel, went
**bimodal**: about 3 % of channels per layer kept `beta_s > 0.9` (time constants
18–26 characters) and the other 97 % pulled their "slow" pole down to ~0.5–0.6.
`EXP_005` confirmed the arm's horizon at n = 7 (median 47, per-seed 38–57)
against the baseline's 7.

The inference §10.11 item 4 draws from that is the most actionable thing in
Phase 4: *if those ~15 channels per layer carry the horizon, the capacity this
project devotes to long memory is 3 % of its width and is nowhere near
saturated.* Two things stop it being usable as written:

* **It is correlational.** The experiment did not intervene. `EXP_004` §10.5
  says so, and adds the fact that argues against the naive reading: `|w|` on the
  tail channels is **not** larger than on the rest (0.99× in layer 0, 1.16× in
  layer 1), which is not what one would expect if those channels were carrying
  the load.
* **It says nothing about bits.** A channel set can carry the *horizon*
  statistic and carry very few *bits*, and `EXP_005` §9.6 makes that live: the
  beyond-horizon component is now only 28 % of the remaining gap while the
  within-reach component is 51.8 %. Whether the horizon is where the bits are is
  a separate measurement and this experiment can make it.

The whole thing runs on the seven checkpoints already on disk. No training, no
new kernel, no new parameter, no gradient gate, and no σ that has not already
been measured.

### 1.1 Why the intervention is on `beta_s` and not on `w`

The obvious ablation — zero `w_c` on the tail channels — is **confounded**, and
the confound is `EXP_004` §10.6's own algebra. At zero context both compartments
start from zero, so

```
v_0 = vf_0 + w_c·vs_0 = cur_0 + w_c·cur_0 = cur_0·(1 + w_c)
```

and `w_c` is therefore a *learned per-channel threshold* as well as a mix.
Zeroing it removes the channel's long memory **and** its threshold shift at
once, and §10.6 attributes 40.6 % of the arm's whole gain to the threshold
effect. An ablation that moves both cannot answer a question about either.

`beta_s` moves only one of them. The expression above has no `beta_s` in it —
`vs_0 = 0·beta_s + cur_0 = cur_0` for any finite decay — so **clamping `beta_s`
cannot change anything the network computes at zero context, exactly and
bitwise.** That gives the experiment a self-check it did not have to invent
(§6, H2) and an intervention that isolates timescale from threshold.

Every intervention below therefore writes to `beta_s_raw` and to nothing else.

---

## 2. Design

Seven trained checkpoints (`twocomp_s0…s6`), inference only. Per layer, all
rankings and the clamp target are computed **once from the intact checkpoint**
and reused at every `N`, so no intervention's ranking is contaminated by another
intervention's edit (failure mode L6).

`clamp(c) := beta_s[c] ← median(beta_s of that layer, intact)`, applied by
setting `beta_s_raw[c] ← logit(median)`. The median is ~0.548 in layer 0 and
~0.573 in layer 1, and is the population the 97 % actually chose.

| Arm | Intervention | N |
|---|---|---|
| **A(N)** | clamp the top `N` channels **ranked by `beta_s` descending** | 0, 4, 8, 16, 32, 64, 128, 512 |
| **B(N)** | clamp **every channel except** the top `N` by `beta_s` — keep only the tail | 16, 64 |
| **C(N)** | clamp the top `N` channels **ranked by `\|w\|` descending** | 16, 64 |
| **E(N)** | `N` channels drawn at random from those with intact `beta_s < 0.8`, each shifted **down by the mean displacement A(N) applied in that layer**, clamped to (0.01, 0.99) | 16 |

`A(0)` is the intact model and is the reference every other row is compared
against. `A(512)` clamps every channel to the median — the arm with no bimodality
left at all.

**Why these four and not more.** `A` is the dose–response curve and answers "how
many channels". `B` is its converse and is what turns a necessity result into a
dissociation. `C` asks whether a different selection rule — the mix magnitude,
which §10.5 noted is uncorrelated with the tail — picks out the same behaviour.
`E` asks whether it is the *identity* of the channels or merely the *size* of the
parameter change, because clamping a tail channel moves `beta_s` by ~0.35 while
clamping a median channel moves it by ~0, and a control that does not match the
displacement is not a control.

91 configurations. At `EXP_005`'s measured 15.6 s per twocomp checkpoint for the
full k-sweep this is ~0.45 GPU-hours.

### 2.1 What is measured

1. **The horizon**, per seed, per configuration, via
   `scripts/exp/001_memory_horizon.py`'s own `paired_context_curve` and
   `horizon_from_excess` — imported, not reimplemented (failure mode L5).
2. **Whole-split fresh test bpc**, per seed, per configuration: the probe's
   `k = 256` point over the full 4 980 736-character test split.
3. **The paired excess curve**, c = 0…127, per seed, per configuration.
4. The overlap between the top-16-by-`beta_s` and top-16-by-`|w|` channel sets.

### 2.2 What is deliberately NOT measured

**Carried bpc.** The probe produces the `fresh` protocol by construction and
`U4`'s threshold is stated in fresh bpc accordingly. Running `scripts/evaluate.py`
as well would add a second instrument for one number and invite the two to
disagree; if the fresh answer is close to the threshold, that is a reason to run
carried afterwards and say so, not a reason to run it now.

**Any re-training.** Nothing here shows what a network *trained* with more slow
channels would do. See §5 item 1 — that limitation is the point of this
experiment's scope, not an oversight in it.

---

## 3. Hypotheses

All horizon thresholds are evaluated at the **inherited 2σ = 0.00922 tolerance**,
at which the intact arm's committed per-seed horizons are 38 / 47 / 45 / 57 / 48
/ 47 / 53 (median 47) and the baseline's are 7, 7, 7, 7, 7. The
two-compartment family's own bar from `EXP_005` (2σ = 0.00627) is reported beside
every number but is not what the thresholds are set against, because the
published 7-and-47 that make these thresholds interpretable were measured at
0.00922. `EXP_005` §9.4's warning applies: the horizon is defined against a
tolerance, so it moves when the tolerance moves. Every comparison here is
**within-seed, at one fixed tolerance.**

| # | Prediction | Threshold |
|---|---|---|
| **U1** | **The tail carries the horizon.** Clamping the top 16 channels per layer (3.1 % of width) removes at least half the horizon: median horizon at `A(16)` ≤ **27** | midpoint of the baseline's 7 and the intact arm's 47 |
| **U2** | **The tail is sufficient as well as necessary.** Keeping only the top 16 slow poles and clamping the other 496 leaves median horizon at `B(16)` ≥ **27** | the same midpoint, from the other side |
| **U3** | **The effect saturates.** \|median horizon `A(32)` − median horizon `A(512)`\| ≤ **3** characters | if it does not saturate, the capacity is distributed and "3 %" is the wrong description |
| **U4** | **The bits follow the horizon.** `A(16)` costs ≥ **0.05 bpc** of whole-split fresh test bpc against `A(0)` | ~40 % of the arm's 0.1240 bpc fresh gain. *Stated in the direction that flatters the plan this experiment exists to support* |
| **U5** | **Selection by `\|w\|` is not the same thing.** Median horizon at `C(16)` ≥ **38** | the lowest intact per-seed horizon |
| **U6** | **It is the channels, not the size of the edit.** Median horizon at `E(16)` ≥ **38** | the same |

**On U4, explicitly.** The reading `EXP_004` §10.11 item 4 invites — long-memory
capacity is 3 % of width and nowhere near saturated, so widen it — is flattered
by U4 holding and damaged by U4 failing. It is stated in the flattering
direction on purpose, so that a failure is a real result and cannot be presented
as one. **If U4 fails, the horizon is nearly free to remove in bits, and
horizon-widening should be demoted in the Phase-5 ranking regardless of what U1
does** — which would corroborate §10.11 item 3 (short-range fidelity, not
horizon, is where the remaining gap lives) from a second direction.

**On U1's power.** This is a within-seed paired intervention on n = 7, so the
comparison does not depend on the seed-to-seed σ that `EXP_005` measured. What it
does depend on is the horizon's own coarseness as a statistic: it is an integer
read off a curve crossing a fixed tolerance, and the intact arm's per-seed values
already span 38–57. A median shift of 20 characters is far outside that spread; a
shift of 5 would not be, and would not be claimed.

---

## 4. Decision rule, fixed now

| U1 | U2 | U5 ∧ U6 | Verdict |
|---|---|---|---|
| **holds** | **holds** | **hold** | **The horizon is carried by ~3 % of channels, causally.** §10.5's association is upgraded from correlational to causal at the level of `beta_s`, and a "widen the slow tail" candidate is added to the ranking — **priced against U4's answer, not against the horizon result.** |
| **holds** | fails | hold | **Necessary but not sufficient.** Both legs reported; the capacity claim is not made. |
| **fails** | either | — | **The association is not causal at N = 16.** The dose–response curve and `N½` (the smallest N at which half the horizon is lost) are reported, and `EXP_004` §10.11 item 4 gets an appended correction. |
| — | — | **either fails** | **The intervention is non-specific.** Nothing is concluded from U1, and this experiment reports a failed control rather than a result. This leg dominates the others. |

Two riders, also fixed now:

* **U4 fails** ⇒ recorded as the headline finding regardless of U1's outcome, and
  horizon-widening is demoted in the Phase-5 ranking. A mechanism that can be
  removed for less than 0.05 bpc is not where 0.3398 bpc of remaining gap is
  going to come from.
* **`B(16)`'s bpc is not interpretable as a cost** and will not be quoted as one.
  It clamps 97 % of channels; of course it is bad. Only its **horizon** is used,
  and only as the converse leg of U1.

---

## 5. Known limitations, stated in advance

1. **This ablates a trained network. It cannot establish headroom.** Showing that
   ~15 channels are necessary for the horizon in a network that chose to have ~15
   is not the same as showing that 50 would buy more. §10.11 item 4's "nowhere
   near saturated" needs a *training* arm, which this experiment prices rather
   than settles — and which it may argue against, if U4 fails.
2. **The clamp target is a choice.** Clamping to the layer's own median is
   defensible (it is where 97 % of the population sits) and it is fixed in
   advance, but a different target — 0.5, or `beta_f` — would give different
   magnitudes. The result is a statement about this intervention.
3. **The ablated configurations have no σ of their own,** and measuring one would
   mean retraining, which is the whole cost this experiment avoids. Every
   comparison is within-seed at a fixed stated tolerance, and no cross-arm
   horizon comparison is made.
4. **The horizon is a coarse statistic** — an integer where a curve crosses a
   tolerance. Small movements in it mean little; §3's power note says how small.
5. **`E`'s random draw is one draw**, seeded and recorded, not a distribution over
   draws. If `E(16)` lands near its threshold, the honest response is more draws,
   not a verdict.

---

## 6. Self-checks — aborting, not advisory

These are not hypotheses. Each one reproduces a number this harness did not
produce, and the run aborts if any fails.

| # | Check | Why it can fail |
|---|---|---|
| **H1** | Every intervention modifies `beta_s_raw` **only**. `w` and every other tensor are bitwise identical to the checkpoint | catches an ablation that silently moves the threshold as well as the timescale — the confound §1.1 exists to avoid |
| **H2** | The probe's **k = 1 point is bit-identical** to `A(0)`'s, for every intervention and every seed, residual < 1e-12 bpc | this is forced by algebra (§1.1: nothing at zero context depends on `beta_s`), so a non-zero residual means the intervention is not doing what this file says it does |
| **H3** | `A(0)`'s k = 256 bpc reproduces each run's committed `final_test.json` fresh bpc to < 1e-3 | `EXP_001`'s F1 — the check that caught a contaminated run in Phase 3 |
| **H4** | `A(0)`'s per-seed horizon reproduces `EXP_005`'s committed **38 / 47 / 45 / 57 / 48 / 47 / 53** exactly | catches a re-derivation of the horizon that is not the committed one |
| **H5** | The edit actually landed: exactly `N` entries of `beta_s_raw` differ per layer (512 − N for `B`), and for N ≥ 4 the k = 256 bpc differs from `A(0)`'s | a no-op ablation produces a perfect null and looks like a clean negative result |

---

## 7. Failure modes

| # | Failure | Detection |
|---|---|---|
| **L1** | The clamp is a silent no-op | H5 |
| **L2** | The clamp writes to the wrong tensor | H1, and H2 would fail for a `w` edit |
| **L3** | A checkpoint is contaminated (phase-3 report §8) | H3 on every intact run; no mutation campaign runs while this does |
| **L4** | Horizons compared across different tolerances | one tolerance fixed in §3; the second bar reported beside, never mixed into a threshold |
| **L5** | The probe is edited and stops being the instrument that produced every other horizon in this project | the statistic, the k-sweep and the F1 tolerance are **imported** from `scripts/exp/001_memory_horizon.py`; this experiment adds no entry to its `ARMS` registry and never overwrites a committed artifact |
| **L6** | Ranking or clamp target computed from an already-edited model, so N values contaminate each other | both computed once from the intact state dict and cached; every configuration is applied to a **fresh** copy of the checkpoint |
| **L7** | The random controls' draw is chosen after seeing a result | RNG seeded from a constant fixed in this file (`20260803`), recorded in the output |

---

## 8. Entry conditions

- [x] Existing test suite green — **242 passed**, matching `EXP_005` §8.1
- [x] No mutation campaign running, and none started while this runs
- [x] Nothing in `src/snn/` modified by this experiment — it is inference-only and
      edits a state dict in memory
- [x] This file committed **before** the harness is written — `bd3939f`, against
      the harness's `8bcad97`

### 8.1 Two refinements made before the run, recorded rather than absorbed

Both are in the harness's commit message and neither was made after seeing a
result.

* **H5 checks containment strictly and the count to within one**, not exact
  equality. The channel that *is* the layer median is already at the clamp
  target, so clamping it is a legitimate no-op; demanding an exact count would
  have failed the check on arithmetic that is correct. The strict leg — nothing
  outside the plan may move — is the one with the content, and it passed 91/91.
* **An unresolved horizon is encoded as 128, not dropped.** `horizon_from_excess`
  returns `None` when the excess curve never flattens inside the probe's range,
  which means *horizon > 127* — the longest outcome, not a missing one. Dropping
  those seeds would bias every median downward, and would do it hardest on the
  heavily ablated arms this experiment is looking at. **In the event it never
  fired:** every horizon resolved, on all 91 configurations.

---

## 9. Results

**Run 2026-08-03.** Seven seeds × 13 configurations = **91 probe runs**, full test
split (4 980 736 characters) at all nine `k`. **~27 minutes ≈ 0.45 GPU-hours**,
against §2's 0.45 estimate. No training. Raw JSON:
`docs/reports/data/exp_006_slow_tail_ablation.json`.

### 9.1 The headline: ~3 % of channels carry the horizon, and they carry the bits too

Median over 7 seeds, at the inherited 2σ = 0.00922 tolerance. "% horizon gone" is
against the intact arm's 47 and the baseline's committed 7; "% gain gone" is
against the arm's 0.1240 bpc fresh gain over the Phase-2 baseline's 2.2697.

| Config | Horizon | % horizon gone | Fresh bpc | % gain gone | vs baseline |
|---|---:|---:|---:|---:|---:|
| **A(0)** *(intact)* | **47** | 0.0 % | **2.14566** | 0.0 % | −0.12404 |
| A(4) | 31 | 40.0 % | 2.17763 | 25.8 % | −0.09207 |
| A(8) | 22 | 62.5 % | 2.21652 | 57.1 % | −0.05318 |
| **A(16)** | **12** | **87.5 %** | **2.26340** | **94.9 %** | −0.00630 |
| A(32) | 8 | 97.5 % | 2.32203 | 142.2 % | **+0.05233** |
| A(64) | 8 | 97.5 % | 2.36057 | 173.3 % | +0.09087 |
| A(128) | 7 | 100.0 % | 2.40012 | 205.1 % | +0.13042 |
| A(512) | 6 | 102.5 % | 2.44868 | 244.3 % | +0.17898 |

**Clamping the 16 highest-`beta_s` channels per layer — 3.1 % of the width —
takes the horizon from 47 to 12 and removes 94.9 % of the arm's bits-per-character
gain.** Clamping 32 leaves the arm **worse than the single-compartment Phase-2
baseline it was built to beat.**

`N½`, the smallest swept `N` at which half the horizon gain is gone, is **8** —
1.6 % of the width.

### 9.2 The dissociation, which is what makes it causal rather than correlational

| Config | What it does | Horizon | % horizon gone | % gain gone |
|---|---|---:|---:|---:|
| **A(16)** | clamp the top 16 by `beta_s` | **12** | **87.5 %** | 94.9 % |
| **B(16)** | clamp all **except** the top 16 by `beta_s` | **45** | **5.0 %** | 148.7 % |
| **C(16)** | clamp the top 16 by `\|w\|` | **47** | **0.0 %** | 28.7 % |
| **E(16)** | 16 random non-tail channels, **same mean displacement** | **48** | **−2.5 %** | 16.6 % |

Four interventions of identical size — 16 channels per layer — and only one of
them touches the horizon. Removing the tail destroys it; keeping *only* the tail
and flattening the other 496 preserves it at 45 of 47; selecting the same number
of channels by mix magnitude, or by a matched-displacement random draw, leaves it
where it was.

**The two selection rules pick disjoint sets.** The overlap between the top 16 by
`beta_s` and the top 16 by `|w|` is **0, 0, 1, 0, 2, 1, 1** channels of 16 across
the seven seeds — which quantifies `EXP_004` §10.5's observation that `|w|` on the
tail is not larger than elsewhere, and is why `C` is a control rather than a
second look at the same channels.

**E(16) also prices the null intervention.** Clamping 16 arbitrary channels by the
same amount costs 16.6 % of the gain. So of A(16)'s 94.9 %, about **78 points are
specific to the tail's timescale** and the rest is the generic cost of clamping
sixteen channels. The horizon result carries no such correction: E(16) removes
none of it.

### 9.3 The predictions, resolved as written

| # | Prediction | Threshold | Measured | Verdict |
|---|---|---|---|---|
| **U1** | median horizon `A(16)` ≤ 27 | 27 | **12** | **held** |
| **U2** | median horizon `B(16)` ≥ 27 | 27 | **45** | **held** |
| **U3** | \|`A(32)` − `A(512)`\| ≤ 3 | 3 | **2** | **held** |
| **U4** | `A(16)` costs ≥ 0.05 bpc fresh | 0.05 | **0.1177** | **held** |
| **U5** | median horizon `C(16)` ≥ 38 | 38 | **47** | **held** |
| **U6** | median horizon `E(16)` ≥ 38 | 38 | **48** | **held** |

**Decision cell (§4): U1 ∧ U2 ∧ U5 ∧ U6 all hold ⇒ the horizon is carried by ~3 %
of channels, causally.** `EXP_004` §10.5's association is upgraded from
correlational to causal at the level of `beta_s`.

**U4 held, so §4's demotion rider does not fire.** The bits do follow the
horizon-carrying channels, and horizon-widening is **not** demoted on this
evidence. §9.6 is where the caution actually belongs, and it is a different
caution from the one §4 anticipated.

**Every prediction holding is itself worth a remark.** Six for six is not what
this project's pre-registrations have usually produced — `EXP_004` falsified T2,
`EXP_005` failed S1, S3 and S4 — and a clean sweep is the pattern that should
attract suspicion rather than satisfaction. What makes it credible here is §9.4:
the interventions are four independent selection rules of identical size whose
results dissociate, and the harness proved its own null on 91 configurations
before any of them were read.

### 9.4 Self-checks — 280 of 280, and one of them is exact

| # | Check | Result |
|---|---|---|
| **H1** | `beta_s_raw` and nothing else modified | **91 / 91** |
| **H2** | the k = 1 point is bit-identical to `A(0)`'s | **84 / 84**, worst residual **0.000e+00 bpc**, nats bit-identical on every configuration |
| **H3** | `A(0)` reproduces the committed `final_test.json` fresh bpc (`EXP_001`'s F1) | **7 / 7** |
| **H4** | `A(0)` reproduces `EXP_005`'s committed per-seed horizons | **7 / 7** — 38 / 47 / 45 / 57 / 48 / 47 / 53, exactly |
| **H5** | the edit landed on exactly the planned channels | **91 / 91** |

**H2 is the one worth dwelling on.** Across all 91 configurations there are
exactly **seven distinct zero-context bpc values — one per seed**, unchanged by
every intervention. That is forced by algebra (§1.1) and it is the check that
distinguishes this ablation from the `w`-zeroing one it replaced: the confound
`EXP_004` §10.6 identified is not merely argued away here, it is measured at zero
to the bit, 84 times.

### 9.5 An internal consistency check nobody asked for, which the design happened to permit

`A(N)` and `B(N)` clamp complementary channel sets, so their costs should sum to
`A(512)`'s — clamping everything. They do, to a precision the experiment did not
target:

| | A | B | sum | A(512) | residual |
|---|---:|---:|---:|---:|---:|
| N = 16 | 0.11774 | 0.18439 | **0.30213** | **0.30302** | 0.00089 |
| N = 64 | 0.21491 | 0.08825 | **0.30316** | **0.30302** | 0.00014 |

The slow pole's total contribution decomposes **additively** between any channel
set and its complement, to within 0.0009 bpc on a 0.303 bpc quantity. Nothing in
the design required this and no prediction rests on it; it is reported because an
un-forced identity that closes is evidence the interventions are doing what they
are described as doing.

It also reframes the numerator. **The slow pole is worth 0.303 bpc, not 0.124** —
`A(512)`, with every slow pole flattened to the population median, is 0.179 bpc
*worse than the Phase-2 baseline*. A second compartment running at ~0.55 is not a
neutral addition that the tail then improves on; it is actively harmful, and the
tail is what pays for it and then some. Of that 0.303, the top 16 channels carry
**39 %** and the other 496 carry 61 %.

### 9.6 What this does NOT establish, and why it cuts against §10.11 item 4

`EXP_004` §10.11 item 4 reads: *"~3 % of channels appear to carry the horizon,
which if true means the long-memory capacity is far from saturated."* **The first
half is now established. The second does not follow, and this experiment supplies
an argument against it.**

`beta_s` was initialised at **0.95 on all 512 channels** (`EXP_004` §10.5). The
network began with the maximum possible slow capacity and **pulled 97 % of it
down**, keeping 11–19 channels per layer above 0.9. So the tail's size is a
**learned outcome of a network that had 512 slow channels available and declined
them**, not a ceiling it ran into. "Nowhere near saturated" describes a constraint
that the training run does not appear to have been under.

This is an argument, not a measurement, and two things could defeat it: the
descent might be an optimisation artifact rather than a preference (a slow pole
may hurt early training in a way that has nothing to do with its final value), and
nothing here rules out a *different* count being better under a different
initialisation. **The experiment that would settle it is cheap and is named
here rather than left implicit: freeze or regularise `beta_s` toward 0.95 on N
channels per layer, train at N ∈ {16, 64, 256}, and read the bpc.** Three seeds at
`EXP_004`'s measured 525 s is ~1.3 GPU-hours for the ladder.

Until that runs, the correct statement is the narrow one: **~3 % of channels carry
the horizon and ~95 % of the arm's gain; whether more of them would help is
unmeasured, and the initialisation history is evidence that it would not.**

Two further limits, restating §5 rather than discovering them:

* **`B(16)`'s bpc is not a cost and is not quoted as one** (§4's rider). It clamps
  97 % of channels and lands 0.060 bpc *worse than the Phase-2 baseline*. Only its
  horizon is used, and only as U2's leg.
* **`C(64)`'s partial horizon loss (25 %) is consistent with the `|w|` ranking
  reaching into the tail as N grows, but the overlap was only measured at N = 16.**
  Stated as consistency, not as a measurement.

### 9.7 Status

**CLOSED. All six pre-registered predictions held; 280 of 280 self-checks passed.**

Three things change, and one that pointedly does not:

1. **`EXP_004` §10.5's association is causal.** ~3 % of channels per layer carry
   the horizon: removing 16 takes it from 47 to 12, keeping only those 16 holds it
   at 45, and two matched controls of identical size move it not at all.
2. **They carry the bits as well as the statistic** — 94.9 % of the arm's fresh
   gain, or ~78 points of that after subtracting the matched control's generic
   cost. The horizon is not a decorative statistic on this arm.
3. **The slow pole is worth 0.303 bpc and is harmful without its tail** (§9.5).
   `A(512)` is 0.179 bpc worse than the Phase-2 baseline.
4. **`EXP_004` §10.11 item 4's "far from saturated" is NOT established, and this
   experiment argues against it** (§9.6). The ranking input it was being used to
   justify — widen the slow tail — should not be taken from this result. The
   frozen-`beta_s` ladder in §9.6 is what would justify it.
