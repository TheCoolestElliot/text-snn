# Phase 4 — controlled experiments: interim report

**Revision 1, 2026-08-03.** Interim, not final: Phase 4 is open and this document
is the running record of it. It exists because Phase 4 had produced six closed
experiments and no report, and a phase whose findings live only in its experiment
logs cannot be reviewed as a phase.

**Nothing in this document is a decision.** Four decisions are Elliot's and are
listed unresolved in §8, exactly as they stood before this report was written.

---

## 1. What Phase 4 is for, and where it stands

`01_reconnaissance.md` §8.2 defines Phase 4 as *pre-registered, single-variable,
≥3-seed controlled experiments* against the ranked candidate list in
`03_phase3_candidates.md` §6.3, with a budget of ~30 GPU-hours.

| | |
|---|---|
| Experiments closed | `EXP_004`, `EXP_005`, `EXP_006`, and `EXP_007`–`EXP_009` in §6 |
| Ranked candidates closed | 3 of 14 (#1, #14, and the horizon question §10.5 opened) |
| Arms adopted | **1** — the two-compartment neuron (`EXP_004`) |
| GPU-hours spent | ~3.5 of ~30 |
| Phase-3 §9 sign-off | **still unticked**; Phase 4 was opened without it on instruction |

The phase has moved the project's answer to its own question twice, and both
moves were away from the thing the ranking was built around.

---

## 2. The one adopted arm, and the shape of its gain

`EXP_004` adopted the two-compartment neuron — a fast pole at the baseline's
`beta = 0.5` and a per-channel learned slow pole, reset-shielded, mixed additively.
`EXP_005` carried it to seven seeds.

| | baseline (n=5) | **two-compartment (n=7)** | GRU anchor (n=3) |
|---|---:|---:|---:|
| test bpc, carried | 2.25311 | **2.11869** | 1.76741 |
| test bpc, fresh | 2.26969 | **2.14566** | 1.80443 |
| whole-split sd | 0.00461 | **0.00313** | 0.00223 |
| memory horizon (2σ) | 7 (5/5 seeds) | **47** (median of 7; 38–57) | 57–60 |

**42.9 σ at the family's own bar.** The result is not in question. What the phase
spent its remaining effort on is the fact that the gain is not uniform:

| Component of the gain | bpc | share |
|---|---:|---:|
| At zero context | +0.0506 | **+41.0 %** |
| Within the baseline's own 7-character reach | **−0.0598** | **−48.4 %** |
| Beyond the horizon | +0.1327 | +107.4 % |
| **Total** | **+0.1235** | 100 % |

Quoted as one number this reads as a uniform win while nearly half its magnitude
is handed back inside the seven characters the baseline already reached. §6.2's
criterion caught it: the arm is worse than the baseline at 5 of 128 context
lengths and they are contiguous — c = 3, 4, 5, 6, 7 — worst at c = 4 where it is
6.5× that context's own noise bar.

**The composition of what is left changed more than its size.**

| Component of the remaining gap to the anchor | baseline | **two-compartment** |
|---|---:|---:|
| At zero context | 0.1193 (25.8 %) | 0.0687 (20.2 %) |
| **Within the 7-character reach** | 0.1161 (25.0 %) | **0.1759 (51.8 %)** |
| Beyond the horizon | 0.2279 (49.2 %) | 0.0952 (28.0 %) |
| **Total** | **0.4633** | **0.3398** |

The candidate recovered 59 % of the component it was ranked to attack and made the
largest remaining component larger. **Short-range fidelity, not horizon, is where
the gap now lives** — and until `EXP_007` no arm this project has run had ever
been aimed at it.

---

## 3. σ is not a project constant

`EXP_005` re-measured the noise floor on the adopted neuron because every 2σ
verdict in the project assumed one measured at a single configuration.

* σ = **0.00313** on the two-compartment arm against **0.00461** on the baseline —
  0.680×. The failure is in the safe direction, and n = 7 cannot *prove* the
  difference: F(6,4) resolves only outside [0.16, 9.20]. **Candidate #14 is closed
  for this neuron and not in general.** Any structurally different arm needs its
  own σ, and `EXP_007`/`EXP_008` report theirs rather than inheriting.
* The §6.2 per-context bars are **arm-specific in shape, not only in scale**: the
  baseline's excess-sd collapses past its 7-character horizon while the
  two-compartment arm's persists to c = 127 and is 21× larger at c = 16.
* The bar in use is **built on the wrong scale** — 2× the baseline's per-seed sd
  of the excess curve, compared against a difference of two arm means, and 45×
  too small at c = 127. The count of significantly-worse contexts is unchanged at
  5/128 under all three constructions. **A change to §6.2's definition is referred
  to Elliot in §8 and is not applied here.**

---

## 4. `EXP_006` — three per cent of the channels carry the horizon, and the bits with it

This is the experiment `EXP_004` §10.11 called "the highest-information single run
available", and it is the one that most changes what Phase 5 should do.

### 4.1 What was measured

`beta_s` was initialised at 0.95 on all 512 channels and went **bimodal**: ~3 % of
channels per layer kept `beta_s > 0.9` (time constants 18–26 characters) and the
other 97 % pulled their "slow" pole down to ~0.5–0.6. `EXP_004` §10.5 observed
that and could not act on it: the observation was correlational, and it said
nothing about bits.

`EXP_006` intervened, on the seven checkpoints already on disk — 91 inference-only
configurations, ~0.45 GPU-hours, no training. **The intervention writes to
`beta_s_raw` and to nothing else**, because zeroing the mix `w` would have removed
a channel's long memory *and* its learned threshold at once (§10.6's algebra), and
an ablation that moves both cannot answer a question about either.

### 4.2 The dose–response curve

Clamping the top `N` channels per layer to that layer's median `beta_s`:

| N clamped | % of width | Horizon | Fresh bpc | % of the arm's gain gone |
|---:|---:|---:|---:|---:|
| 0 *(intact)* | 0 % | **47** | 2.14566 | 0 % |
| 4 | 0.8 % | 31 | 2.17763 | 25.8 % |
| 8 | 1.6 % | 22 | 2.21652 | 57.1 % |
| **16** | **3.1 %** | **12** | **2.26340** | **94.9 %** |
| 32 | 6.3 % | 8 | 2.32203 | 142.2 % *(worse than the baseline)* |
| 512 | 100 % | 6 | 2.44868 | 244.3 % |

**Clamping 3.1 % of the width takes the horizon from 47 to 12 and removes 94.9 %
of the arm's bits-per-character gain.** `N½` — the smallest swept `N` at which
half the horizon gain is gone — is **8**, or 1.6 % of the width.

### 4.3 The dissociation, which is what makes it causal

Four interventions of identical size, 16 channels per layer:

| Intervention | Horizon | % horizon gone |
|---|---:|---:|
| clamp the top 16 by `beta_s` | **12** | **87.5 %** |
| clamp everything **except** the top 16 by `beta_s` | **45** | 5.0 % |
| clamp the top 16 by `\|w\|` | 47 | 0.0 % |
| 16 random non-tail channels at **matched displacement** | 48 | −2.5 % |

Only one of the four touches the horizon. The two selection rules pick **disjoint
sets** — the overlap between the top 16 by `beta_s` and the top 16 by `|w|` is
0, 0, 1, 0, 2, 1, 1 channels of 16 across the seven seeds.

The matched-displacement control also prices the null intervention: clamping 16
arbitrary channels by the same amount costs 16.6 % of the gain, so of the 94.9 %,
about **78 points are specific to the tail's timescale**. The horizon result needs
no such correction — the control removes none of it.

All six pre-registered predictions held and **280 of 280 self-checks passed**,
including one that is exact: because nothing at zero context can depend on
`beta_s` (`vs_0 = 0·beta_s + cur_0`), the probe's k = 1 point must be bit-identical
under every intervention. Across all 91 configurations there are exactly seven
distinct zero-context bpc values, one per seed, **nats identical to the bit**.

### 4.4 The slow pole is worth 0.303 bpc, and is harmful without its tail

`A(N)` and `B(N)` clamp complementary channel sets, so their costs should sum to
`A(512)`'s. They do, to 0.0009 bpc on a 0.303 bpc quantity — an identity nothing
in the design required and no prediction rests on.

It reframes the numerator. **`A(512)`, with every slow pole flattened to the
population median, is 0.179 bpc *worse than the Phase-2 baseline*.** A second
compartment running at ~0.55 is not a neutral addition that the tail improves on;
it is actively harmful, and the tail pays for it and then some. Of the 0.303, the
top 16 channels carry 39 % and the other 496 carry 61 %.

### 4.5 What it does **not** establish — and this is the part that changes Phase 5

`EXP_004` §10.11 item 4 read: *"~3 % of channels appear to carry the horizon,
which if true means the long-memory capacity is far from saturated."*

**The first half is now established causally. The second does not follow, and
`EXP_006` supplies an argument against it.** `beta_s` began at 0.95 on all 512
channels: the network started with the maximum available slow capacity and
**pulled 97 % of it down**. The tail's size is a learned outcome of a network that
had 512 slow channels and declined them, not a ceiling it ran into.

That is an argument, not a measurement, and two things could defeat it — the
descent could be an optimisation artifact rather than a preference, and nothing
rules out a different count being better under a different initialisation. **The
experiment that would settle it is the frozen-`beta_s` ladder named in `EXP_006`
§9.6: freeze or regularise `beta_s` toward 0.95 on N channels per layer, train at
N ∈ {16, 64, 256}, read the bpc. Three seeds, ~1.3 GPU-hours.**

**Until that runs, "widen the slow tail" must not be ranked on this result.** The
correct statement is the narrow one: *~3 % of channels carry the horizon and ~95 %
of the arm's gain; whether more of them would help is unmeasured, and the
initialisation history is evidence that it would not.*

### 4.6 Why six-for-six should attract suspicion rather than satisfaction

`EXP_004` falsified T2 and `EXP_005` failed three of five, so a clean sweep is the
pattern that deserves scrutiny. What carries the credibility here is not the
count: it is that the four interventions are independent selection rules of
identical size whose results dissociate, over a harness that proved its own null
on 91 configurations — 84 of them bit-exactly — before any of them were read.

One control is weaker than its verdict suggests, and is recorded as such:
`C(64)`'s partial horizon loss (25 %) is consistent with the `|w|` ranking
reaching into the tail as N grows, but the overlap was only measured at N = 16.

---

## 5. What §4 changes about the plan

1. **The horizon question is closed for this neuron.** It is carried by ~3 % of
   channels, it is causal, and the bits follow it. There is nothing left to
   establish about *whether* the mechanism works.
2. **The obvious follow-up is blocked, not opened.** "Widen the slow tail" was the
   ranking input §10.11 item 4 was being used to justify, and §4.5 withdraws it
   pending a 1.3 GPU-hour ladder.
3. **The remaining gap is 51.8 % within-reach and only 28 % beyond the horizon.**
   The Phase-3 ranking, which put beyond-horizon candidates first, was right for
   Phase 4 and is wrong for what follows.
4. **So the next arms have to point somewhere else** — at short-range fidelity, at
   the zero-context component, or at the training signal. That is exactly what
   `EXP_007`, `EXP_008` and `EXP_009` do, and §6 reports them.
