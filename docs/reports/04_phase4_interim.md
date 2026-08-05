# Phase 4 — controlled experiments: interim report

**Revision 2, 2026-08-04.** Interim, not final: Phase 4 is open and this document
is the running record of it. It exists because Phase 4 had produced six closed
experiments and no report, and a phase whose findings live only in its experiment
logs cannot be reviewed as a phase.

**Nothing in this document is a decision.** Eight decisions are Elliot's and are
listed unresolved in §8. Rev 2 adds one of them and decides none.

**What rev 2 does.** It reads `EXP_007`, `EXP_008` and `EXP_009` *together* rather
than one after another (§6.7), states a verdict for every Phase-4 arm against its
own pre-registered rule (§6.8), re-runs every gate in §9 from a clean tree, and
corrects five numbers rev 1 got wrong (§10). **No verdict, threshold or decision
changes.** Two of the corrections make an arm look worse, one makes the adopted
arm look slightly better, and all five restore a committed measurement rather than
redefining anything — the redefinitions on the table are still in §8, still
unapplied.

---

## 1. What Phase 4 is for, and where it stands

`01_reconnaissance.md` §8.2 defines Phase 4 as *pre-registered, single-variable,
≥3-seed controlled experiments* against the ranked candidate list in
`03_phase3_candidates.md` §6.3, with a budget of ~30 GPU-hours.

| | |
|---|---|
| Experiments closed | `EXP_004`, `EXP_005`, `EXP_006`; `EXP_007`, `EXP_008` and `EXP_009` in §6 |
| Ranked candidates closed | 5 of 14 (#1, #14, the horizon question §10.5 opened, and two of the three I5 referrals run as diagnostics) |
| Arms adopted | **1** — the two-compartment neuron (`EXP_004`). A second is **recommended** in §8 and not taken |
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

The candidate recovered **58.2 %** of the component it was ranked to attack
(+0.1327 of the 0.2279 available beyond the horizon) and made the largest
remaining component larger. **Short-range fidelity, not horizon, is where
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

---

## 6. `EXP_007`, `EXP_008`, `EXP_009` — three arms aimed away from the horizon

Pre-registered together at `1f05bab`, before the harness (`5d6f88a`) and before
the resolver (`f9c509c`); every threshold, every self-check and every
direction-of-statement was fixed then. Nine training runs, **~1.6 GPU-hours**.

### 6.1 The scoreboard

| arm | n | test bpc, carried | Δ vs its reference | se | wall-clock | peak VRAM |
|---|---:|---:|---:|---:|---:|---:|
| Phase-2 baseline | 5 | 2.25311 | — | — | 1.00× | 0.609 GiB |
| **token-shift** *(diagnostic)* | 3 | **2.21444 ± 0.00103** | **−0.03867** | −18.0 | 1.49× | 0.859 GiB |
| **learned threshold** | 3 | **2.19416 ± 0.00244** | **−0.05896** | −23.6 | **1.12×** | 0.734 GiB |
| two-compartment *(adopted)* | 7 | 2.11869 | −0.13442 | — | 1.32× | 0.980 GiB |
| **distilled twocomp** *(diagnostic)* | **2** | **2.10921 ± 0.00190** | −0.00948 | −5.3 | 2.13× | 1.10 GiB |
| GRU anchor *(violates I5)* | 3 | 1.76741 | — | — | 1.11× | 0.864 GiB |

**The wall-clock column is against `snn_beta0.5_s0`'s 398.9 s** — `EXP_004`'s
convention and the run manifest's `reference_wall_clock_s` — and every arm's entry
is the mean over the seeds that were *scored*. Rev 1 had two entries that were not:
the adopted arm at 1.35× was `twocomp_s0` alone against `EXP_004`'s committed
three-seed **1.32×** (all seven seeds give 1.30×), and the distilled arm's 1.58×
was measured against the two-compartment arm — correct in `EXP_009` §9.1, wrong in
a column whose reference row is the baseline, where it is **2.13×**. §10 records
both.

Both new arms clear the adoption bar — token-shift by **4.2×** and the threshold
arm by **6.4×** (8.4σ and 12.8σ at the inherited noise floor; rev 1 said "an order
of magnitude", which the numbers do not support) — and **neither regresses a
single one of the 128 context lengths**, which no previous Phase-4 arm has
managed.

### 6.2 Where each arm's gain lives, which is the finding

Decomposed on `EXP_004` §10.3's basis, cut at the baseline's horizon of 7.
Positive = better.

| arm | zero context | within the 7-char reach | beyond the horizon |
|---|---:|---:|---:|
| two-compartment | +0.0506 (41.0 %) | **−0.0598 (−48.4 %)** | +0.1327 (107.4 %) |
| **token-shift** | **+0.0493 (101.8 %)** | −0.0065 (−13.4 %) | +0.0057 (11.7 %) |
| **learned threshold** | +0.0291 (49.7 %) | **+0.0288 (49.1 %)** | +0.0007 (1.2 %) |
| **distilled** *(vs twocomp)* | +0.0243 (226 %) | **−0.0131 (−122 %)** | −0.0004 (−4 %) |

**Three of the four gains are dominated by the zero-context component, and the
project has now measured that component four independent ways.** That is the
strongest pattern in Phase 4.

**The token-shift's result is the one that explains the others.** At the first
character of a window `cur_{−1} = 0`, so `cur'_0 = mu_c·cur_0` — the two-tap
filter **degenerates to a per-channel gain**, which is precisely `EXP_004`
§10.6's learned threshold and precisely what `EXP_008` implements. The arm ranked
to attack short-range memory delivered 102 % of its gain through the *other*
arm's mechanism, and −0.0065 through its own. `EXP_007` §9.2 has the algebra.

**And the arm nobody aimed at short range is the only one that moved it.** The
learned threshold recovers **+0.0288 bpc within the baseline's own reach — 24.8 %
of the 0.1161 available there** — where the two-compartment neuron gave back
0.0598 and the token-shift gave back 0.0065. A static per-channel threshold is
not a zero-context trick: it changes when every channel fires, and therefore when
it resets, at every context length.

### 6.3 The threshold arm buys 44 % of the adopted arm's gain for nothing

| | two-compartment | **learned threshold** |
|---|---|---|
| test bpc gained | 0.1344 | **0.0590 (43.9 % of it)** |
| new state variable | yes | **no** |
| new CUDA kernel + hand-written backward | yes | **no** |
| new R10 gate + mutation campaign | yes (22 mutations) | **no** |
| parameters | +2 048 (+0.28 %) | +1 024 (+0.14 %), **and redundant** |
| wall-clock | 1.32× | **1.12×** |
| contexts regressed (§6.2 criterion) | 5 / 128 | **0 / 128** |
| cost at inference after folding | — | **zero: it folds into `W`** |

**`EXP_008` §1.1's Identity 2 is confirmed to 6.8–8.3 units in the last place of
float64** (`EXP_008` §9.5 quotes the ≈7 of its first two seeds; seed 2's leg C is
8.3 eps): `g·(W·h + b)` and `(g·W)·h + g·b` are the same function, so the arm's
1 024 parameters span no new functions and fold into the projection they multiply.
The gain is therefore an **optimisation effect** — a different trajectory under
AdamW, not a larger reachable set — and it can be shipped at the baseline's exact
parameter count and cost.

That also settles a question `EXP_004` §10.6 left open. It offered two live
explanations for the two-compartment arm's zero-context gain — the learned
threshold, or the +0.28 % parameters — and could not separate them. **Identity 2
collapses them: neither is capacity.** The baseline could always have represented
the threshold by scaling its own rows; what it could not do was *find* it.

### 6.4 Distillation closes 3.2 % of the gap, and one seed died proving something else

`EXP_009` was the diagnostic that would say whether the remaining 0.3398 bpc is
representational or optimisational. It gives a **lower bound of 3.2 %** at 5.3 se
— real, and far under the 25 % that would have changed Phase 5's shape.

**§1.1 of that file forbids the tempting inversion.** X2 failing does *not* show
the gap is representational: λ, temperature, schedule and teacher were all unswept
by design, so a null is a null *of the recipe*. Only the positive part is
load-bearing, and the positive part is 3.2 %.

**The more transferable result is the failure.** `twocomp_distill_s1` diverged at
step 12 497 and `009_chase_divergence.py` reproduces it deterministically:

* the **forward never overflowed** — logits max 38.9, loss a healthy 1.5047;
* the first symptom is **not a NaN**: every gradient was finite, the largest at
  **5.46e31**, and `clip_grad_norm_`'s fp32 sum-of-squares overflowed, so the clip
  reported `inf` and *zeroed* the update instead of rescaling it;
* one step later, the NaN appears in exactly `embed.weight`, `layers.0.*`, `w.0`
  and `beta_s_raw.0` — layer 0 and below only — so it was **generated inside layer
  0's two-compartment backward**, not inherited from above.

So the **adopted** neuron has a BPTT gradient-explosion mode that the project's
gradient clip cannot see, because the clip's own accumulator overflows first.
Seven un-distilled seeds never hit it, so nothing here shows the committed
configuration is unstable — but the clip's blind spot is real and is not a
property of distillation. Two one-line changes suggest themselves (accumulate the
clip norm in fp64; abort or skip on a non-finite norm rather than silently zeroing
it) and **neither is applied**: both touch committed Phase-2 code and both would
flatter the work. §8 refers them.

### 6.5 A noise floor the project did not know it had

Chasing `EXP_008`'s failed W1 produced a number worth more than the prediction:

> **This architecture's reported bpc is reproducible to only ~2e-3 across
> mathematically-equivalent reparameterisations of its own weights.**

A one-ulp change in the current — from folding `g` into `W` and letting the GEMM
accumulate in a different order — flips 1.6–3.2e-4 of layer 0's spikes, which the
next layer amplifies to 2.5–4.6e-3, until the logits differ by up to 27 on a scale
of 264 and the bpc by up to 0.002.

This does **not** contradict `tests/test_determinism.py`: for fixed weights and a
fixed code path, evaluation stays bit-exact. It applies exactly when weights are
rewritten into an equivalent form — folding, quantisation, weight normalisation.
**The adoption bar 2σ = 0.00922 is only 4.6× this floor**, and the §6.2
per-context bars at c ≥ 16 (≈ 0.0003) are an order of magnitude *below* it. Any
future claim that rests on a reparameterisation needs this floor beside it rather
than the seed σ.

### 6.6 Predictions resolved, including the ones that failed

| experiment | held | failed | notes |
|---|---|---|---|
| `EXP_007` | V1, V3, V5, V6 | **V2** | V4 held **vacuously** and is not counted as a hit — it bounded the mechanism from above and the mechanism went the other way |
| `EXP_008` | W2, W4, W5, W6 | **W1, W3** | W3 missed by 0.3 points (49.7 % vs 50 %) and is reported as failed. W1's chase confirmed the algebra and indicted the tolerance |
| `EXP_009` | X1, X3, X5 | **X2, X4** | provisional at n = 2 |

Five of seventeen predictions failed and two more are qualified. After `EXP_006`'s
six-for-six — which §4.6 flagged as a pattern deserving suspicion — that is a more
normal distribution for this project, and the three that were **stated against the
plan** (`EXP_008`'s W5 and W6, `EXP_009`'s X5) all held.

### 6.7 The three read together, which is not how they were designed

The three experiments were pointed at three different things: short-range fidelity
(`EXP_007`), the zero-context component (`EXP_008`), and the training signal
(`EXP_009`). **All three landed on the same component, and for two of them it is
provably the same mechanism.**

| arm | aimed at | where the gain landed | what the mechanism reduces to at c = 0 |
|---|---|---|---|
| token-shift | within-reach | **101.8 % zero-context** | `cur'_0 = mu_c·cur_0` — a per-channel gain |
| learned threshold | zero-context | 49.7 % zero, 49.1 % within | `cur' = exp(−θ_c)·cur` — a per-channel gain |
| distilled twocomp | the training signal | **226 % zero-context** | the teacher's marginals, no mechanism claimed |

The first two rows are **the same mechanism arrived at from opposite directions**,
and `EXP_007` §9.2 has the line of algebra that says so: at the first character of
a window the shift register is zero, so the two-tap filter *is* a per-channel gain
on the current — which is what `EXP_008` implements deliberately and what
`EXP_004` §10.6 found implicitly inside the adopted neuron. Counting the adopted
arm, **four arms have now measured one quantity**, and they disagree about its
size: the two-compartment neuron reaches an effective 0.69/0.79, token-shift's
`mu` settles at 0.712/1.147, the explicit parameter goes to **0.342/0.428**.

**Three consequences, and the third is the one that should reorganise Phase 5.**

**1. Only the arm that added the least structure moved the component the project
cares about.** Against the 0.1161 bpc available within the baseline's own reach:

| arm | what it added | within-reach recovered |
|---|---|---:|
| two-compartment | a second state variable, a kernel, a backward, a gate | **−51.6 %** |
| token-shift | a two-tap FIR and four `[B,L,d]` intermediates per layer | −5.6 % |
| **learned threshold** | **one redundant scalar per channel** | **+24.8 %** |

The ordering is inverse to the amount of machinery, and nothing in the project
predicted it. §7 item 2 keeps *why* as an open question rather than answering it.

**2. Token-shift is dominated, which drains the stake out of decision #3.** The two
arms were not run head to head, so this is a descriptive contrast against a common
baseline and neither file pre-registered it — but it is not close: the threshold
arm is **0.0203 bpc better at 13.3 se**, at 1.12× wall-clock against 1.49×, 0.734
GiB against 0.859, with no new intermediates, and it folds away at inference.
Token-shift is worse on every measured axis and its one distinguishing feature —
the second tap — is worth **−0.0009 bpc**: the two components where the tap is
live (within-reach −0.0065, beyond-horizon +0.0057) very nearly cancel. **Ruling
token-shift inside or outside I5 therefore changes nothing about what to build.**
The ruling is still Elliot's and §8 still carries it; what §6.7 removes is the cost
of getting it wrong.

**3. Phase 4's bits are being won by optimisation, and the ranked list is entirely
architectural.** Take the three gains in turn against the function class each arm
can reach:

* the **threshold** arm adds no functions at all — Identity 2, measured (§6.3);
* the **distilled** arm is the adopted architecture, unchanged, trained against a
  different signal;
* **token-shift** genuinely does add functions — a two-tap FIR over time is not in
  the baseline's class — and the parts of the curve where those functions are live
  are worth the −0.0009 bpc above. Its 0.0387 is bought at c = 0 by the part that
  is a reparameterisation.

**So essentially none of the 0.0387 / 0.0590 / 0.0095 bpc these three arms bought
is attributable to a larger function class.** Together with `EXP_009`'s
constructive lower bound — ≥ 3.2 % of the remaining gap is reachable without any
architectural change — that is Phase 4's second-strongest pattern, and
`03_phase3_candidates.md` §6.3 ranks fourteen architectural candidates and no
optimisation ones.

**What reading them together does not license.** The three arms act on one
component, so **their numbers may not be added**, and neither may any of them be
added to the adopted arm's (§7 item 4). Each is n = 3 against the baseline's n = 5
— n = 2 for the distilled arm — no arm was re-run, and each reports its own σ. And
`EXP_009`'s divergence is not a property of distillation: it is in the **adopted**
neuron's backward (§6.4), so it survives every decision about these three arms.

**One cross-log correction.** `EXP_007` §9.6 item 2 closes with "the within-reach
component remains unattacked by any arm this project has run". `EXP_008`, closed
the same day, moved it by +0.0288. Both logs are closed and neither is rewritten;
the joint claim is the one in §6.2 and in item 1 above.

### 6.8 The verdict on every Phase-4 arm, against its own rule

Each arm is resolved only against the thresholds its own pre-registration fixed.
`scripts/exp/010_phase4_arm_results.py` applies them mechanically and was
committed at `f9c509c`, before any of these numbers were read.

| arm | pre-registered cell reached | verdict as written | status now |
|---|---|---|---|
| **two-compartment** `EXP_004`/`005`/`006` | T-rules; 42.9σ at the family's bar | **adopted in Phase 4** | **the only adopted arm.** Unchanged by rev 2. Carries a gradient-explosion mode the clip cannot see → decision #7 |
| **token-shift** `EXP_007` | **V1 holds, V2 fails** | "It buys bits, but not the ones this experiment aimed at. The within-reach claim is **not made**." | **null for its purpose, retained.** −0.0387 bpc is real; the arm is dominated (§6.7 item 2). **Not recommended even if I5 admits it** |
| **learned threshold** `EXP_008` | **W1 fails** → "stop and chase it" | chased (`008_chase_fold.py`): the algebra is right to 6.8–8.3 fp64 eps, the **bar** was borrowed from the wrong precedent. W2 held at 23.6 se | **recommended, not adopted** — decision #5, and coupled to #6 because the recommendation rests on a check that **failed as written** |
| **distilled twocomp** `EXP_009` | **X1 holds, X2 fails**, provisional at n = 2 | "Some of the gap is optimisational; less than a quarter of it by this route. The architecture ranking stands." | **diagnostic, never a headline** (`EXP_009` §4's rider). Not an arm to adopt — it is a training recipe, and the recipe was one unswept point |

**Nothing here is adopted by this report.** `EXP_008` §4's rider is explicit that
the adoption is recommended *to Elliot* and that Phase 3's §9 is still unticked;
§8 items 1 and 5 are where that lives.

**And the threshold arm's headline is quoted under a caveat, every time.** Its
decision cell is `W1 fails`, whose text is "no bpc verdict is issued until the
residual is explained". The residual was explained and the tolerance was **not**
widened, so what is reported is: W1 failed as written; Identity 2 holds; the
question of what tolerance a fold-in should carry is #6 and is open. An experiment
does not get to redefine its own threshold after watching it fail.

---

## 7. What Phase 5 should be aimed at now

Phase 4 has moved the target twice, and §6 moves it a third time.

1. **The zero-context component is the one that keeps paying, and it is cheap.**
   Four arms have now attacked it and all four succeeded; the cheapest of them
   (`EXP_008`) delivers 44 % of the adopted arm's whole gain for one foldable
   parameter per channel. **Nothing in the ranked list was pointed here.**
2. **The within-reach component — 51.8 % of the remaining gap — is still barely
   attacked.** The arm designed for it moved it by −0.0065; the arm not designed
   for it moved it by +0.0288, which is 24.8 % of what is there. What made the
   difference is not understood, and understanding it is worth more than another
   arm.
3. **The horizon question is closed and its follow-up is blocked** (§4.5). Do not
   rank tail-widening without the frozen-`beta_s` ladder.
4. **Composition is now the biggest unmeasured thing in the project.** The
   adopted arm and the threshold arm both act on the zero-context component, and
   `EXP_004` §10.11 item 2 warned in advance that "in this arm it is already
   present and would not add twice". The two-compartment neuron reaches an
   effective threshold of 0.69/0.79; the explicit arm reaches 0.342/0.428. **The
   numbers may not be added.**
5. **The phase's bits came from optimisation and the candidate list is
   architectural** (§6.7 item 3). Three arms, essentially none of whose gain is
   attributable to a larger function class, and a constructive lower bound that
   ≥ 3.2 % of the remaining gap needs no architecture at all — against a ranked
   list of fourteen architectural candidates and zero optimisation ones. Phase 5
   should carry at least one arm that changes only how the model is trained.

### 7.1 A revised ranking, offered and not applied

`03_phase3_candidates.md` §6.3's list is a Phase-3 artifact and is not edited
here. This is what Phase 4's evidence implies about it, for Elliot to accept or
reject:

| rank | arm | why | GPU-h |
|---:|---|---|---:|
| **1** | **threshold × two-compartment composition** | the only way to know whether Phase 4's two best arms add. §10.11 item 2 asked for it in advance | 0.5 |
| **2** | **#2 current normalisation (RMSNorm on `cur`)** | the real §6.3 #2, passes the Phase-3 §7.2 screen, attacks the same components by a different route, still unrun | 0.4 |
| **3** | **why the threshold arm helps within reach** | a mechanism study, not an arm. The only measurement pointed at the largest remaining component | 0.3 |
| **4** | **frozen-`beta_s` ladder**, N ∈ {16, 64, 256} | the only thing that can license tail-widening (§4.5) | 1.3 |
| **5** | #13 training budget (40k steps) | three arms' gains are optimisational; the budget is the cheapest optimisation lever and is unmeasured | 0.7 |
| — | #5 learned per-channel decay | partly subsumed by the adopted arm | — |
| — | #6 adaptive threshold | **blocked**: both its inits are saddles, and its fallback needs #8 first (`03_phase3_candidates.md` §7.3) | — |
| — | #10 rotational membrane | screened init is a saddle; θ = π/8 fallback passes but the mechanism is unmotivated by any Phase-4 finding | — |

**Demoted on evidence, not on taste:** every beyond-horizon candidate. That
component is 28 % of what is left, the adopted arm already took 58.2 % of it, and
`EXP_006` withdrew the argument that more is available.

### 7.2 The order it should run in, and what has to be settled first

Recommended, not scheduled. The whole of §7.1 is ~3.2 GPU-hours against the ~26.5
still unspent, so **the constraint is review, not budget**.

**Before anything in §7.1 starts, three of §8 need answers**, and they are cheap
ones: **#1** (Phase-3 §9) because `EXP_008` §4's rider makes adoption conditional
on it; **#5** because rank 1 measures the composition of an arm that is only
*recommended*, and if #5 is "no" the composition run answers a question about an
arm nobody is shipping; **#4** because it decides whether any of this is Phase 4
or Phase 5 — which is the new decision #8.

Given those, the order below is not arbitrary — each step's result changes whether
the next is worth running:

1. **Composition (§7.1 rank 1, 0.5 GPU-h).** Threshold on top of the adopted
   neuron. It is first because every other ranking depends on the answer, and
   because §7 item 4 says the two gains may not be added — so today the project
   cannot state what its best model scores. **Pre-register the non-additive
   prediction before running it**, or a partial gain will read as a disappointment
   rather than as the measurement `EXP_004` §10.11 item 2 asked for.
2. **RMSNorm on `cur` (rank 2, 0.4 GPU-h)** — `03_phase3_candidates.md` §6.3's
   actual #2, unrun, attacks the same components by a different route, and its
   result is only interpretable against step 1's.
3. **Why the threshold arm helps within reach (rank 3, 0.3 GPU-h).** A mechanism
   study, not an arm. §6.7 item 1 is the strongest unexplained result in the
   phase, and it points at the largest remaining component.
4. **The frozen-`beta_s` ladder (rank 4, 1.3 GPU-h)** — the only thing that can
   license tail-widening (§4.5), and the only reason to run it is to *close* that
   question rather than leave it open into another phase.
5. **A training-signal arm (rank 5, 0.7 GPU-h)** — §7 item 5. `EXP_009`'s λ was
   one unswept point and §9.3 of that file forbids reading its null as a ceiling.

**Two things Phase 5 should not do.** It should not open on the beyond-horizon
candidates the Phase-3 list still ranks first — §7.1's demotion is on evidence.
And it should not carry `EXP_008`'s gain into a headline until #6 is ruled: the
number rests on a check that failed as written, and §6.8 says so wherever it is
quoted.

---

## 8. What is Elliot's, and is still open

Nothing in this list has been decided. Seven items stood open at rev 1 and all
seven still do; **#8 is new at rev 2**. Rev 2 sharpened the evidence under #3, #4
and #5 without moving any of them.

| # | Decision | Status | Evidence, as of rev 2 |
|---:|---|---|---|
| 1 | **Phase-3 §9 sign-off** | **open** — still unticked | none; nothing is left for it to wait on. It now **gates #5**: `EXP_008` §4's rider makes the adoption conditional on it |
| 2 | **§6.2 bar definition** (SE-on-the-absolute-curve, per arm) | **open** | `EXP_005` §9.4. Verified since: the per-arm sd it needs is already computed from the ≥3 seeds the adoption rule requires, so the cost `EXP_005` flagged is already paid |
| 3 | **The three I5 boundary questions** | **open**, and **now cheap** | token-shift and distillation ran as labelled diagnostics. §6.7 item 2: token-shift is dominated by the threshold arm on bits (0.0203 bpc, 13.3 se), wall-clock (1.49× vs 1.12×), VRAM and inference cost, and its own second tap is worth −0.0009 bpc. **Ruling it either way changes nothing about what to build** |
| 4 | **Phase 5 authorisation** | **open** | recommend **not yet**: 5 of 14 candidates closed, ~3.5 of ~30 GPU-hours, and §7.1 item 1 — whether Phase 4's two best arms compose — is unmeasured and costs 0.5 GPU-hours |
| 5 | **Adopt the learned per-channel threshold?** | **open** | −0.0590 bpc at 23.6 se (6.4× the bar), 0/128 contexts regressed, 1.12× wall-clock, folds to zero cost. **Recommended for adoption**, with §6.5's ~2e-3 drift caveat and §7 item 4's composition warning. **Coupled to #6**: its own load-bearing check, W1, failed as written (§6.8) |
| 6 | **The fold-in tolerance** | **open** | `EXP_008` W1 failed at 1e-3, a bar borrowed from F1, which compares two evaluations of the *same* weights. §6.5 measures why that was the wrong precedent; the algebra holds at 6.8–8.3 fp64 eps. **Referred, not redefined** |
| 7 | **The gradient clip's fp32 accumulator** | **open** | `EXP_009` §9.4: the clip overflows at ‖g‖ = 5.5e31 and silently zeroes the update. Two one-line changes named, **neither applied** — both touch committed Phase-2 code and both would flatter the work. Independent of every decision about the three new arms: the exploding gradient is in the **adopted** neuron's backward |
| **8** | **Does the composition experiment run inside Phase 4, or as Phase 5's first arm?** | **NEW, open** | §7.1 rank 1 is 0.5 GPU-h and answers a question `EXP_004` §10.11 item 2 asked in advance. Rev 1 implied it in #4 without separating it. Inside Phase 4 it keeps the phase open past its report; as Phase 5's opener it closes Phase 4 with the composition unmeasured — **and today the project cannot state what its best model scores either way**. Recommendation: **inside Phase 4**, because the arm it composes is the one #5 is being asked about |

---

## 9. Gates, at the close of this revision

**Every row was re-run for rev 2 on 2026-08-04**, from the clean tree at `dcc8f82`,
rather than carried over from rev 1.

| Gate | State |
|---|---|
| Test suite | **262 passed**, 1 warning, 42 s — the 242 from `EXP_006` unchanged, plus 20 in `tests/test_prescan.py`, four of which test Identity 2 directly |
| **Resolver reproduces its own artifact** | `010_phase4_arm_results.py` re-run to a scratch path: **0 differences** against the committed `exp_007_009_arm_results.json` — every reference, decomposition, prediction and the three GPU fold-in evaluations, bit-for-bit |
| Committed references recomputed | the resolver's own guard: baseline **2.25311**, twocomp **2.11869**, GRU **1.76741** carried, each recomputed from per-run `final_test.json` and asserted against the figure the reports quote (worst residual **1.5e-5**, tolerance 5e-4) |
| Phase-3 §7.2 reachability screen | re-run: `thr` **PASS**, `ctl_live` **PASS**, `ctl_dead` SADDLE and `ctl_faint` FAINT as expected, all three self-tests held, and it reproduces `exp_004_gradient_reachability.json` at **0.00e+00** worst relative residual |
| R10 gradient gates | untouched: neither new arm has a hand-written backward, and both call the committed `lif_scan` |
| `EXP_001` F1 (probe reproduces committed bpc) | **23 / 23** runs, `f1_failures` empty in the horizon artifact |
| Mutation campaign | none run during any training run; lockfile checked before each |
| Working tree | clean at `dcc8f82` before this revision; **rev 2 modifies exactly one file — this one.** No script, artifact, experiment log or run directory is touched, and nothing is committed |

**Phase 4 remains open.** Rev 1 closed `EXP_007`, `EXP_008` and `EXP_009`; rev 2
synthesises them, corrects five numbers and states a verdict for every arm. **It
authorises nothing, adopts nothing, and redefines no threshold.**

---

## 10. Changelog

| Rev | Change |
|---|---|
| 1 | 2026-08-03. First Phase-4 report: the adopted arm and the shape of its gain, σ's non-transfer, `EXP_006`'s causal tail result and what it withdraws, `EXP_007`/`EXP_008`/`EXP_009` resolved against their pre-registrations, the reparameterisation noise floor, a revised ranking offered and not applied, and seven open decisions. |
| 2 | 2026-08-04. **Synthesis and corrections. No verdict, threshold or decision changes.** (a) §6.7 added: the three arms read together — two of them are the same mechanism, only the arm that added the least structure moved the within-reach component, token-shift is dominated at 13.3 se, and essentially none of the three gains is attributable to a larger function class. (b) §6.8 added: the verdict on every Phase-4 arm against its own rule, including that `EXP_008`'s cell is `W1 fails` and its headline is quoted under that caveat. (c) §7.2 added: the order Phase 5's arms should run in and what must be settled first. (d) §8 item 8 added; #3, #4, #5 re-evidenced; **all eight remain open**. (e) §9 re-run from a clean tree, with the resolver's 0-difference reproduction as a new row. **Corrections:** (f) §6.1's wall-clock for the adopted arm was **1.35×**, which is `twocomp_s0` alone — `EXP_004`'s committed three-seed figure is **1.32×** (1.30× over all seven), and `EXP_004` §10 item 6 said to quote it; *this correction flatters the adopted arm and is made because it restores a committed measurement*. (g) §6.1's distilled row was **1.58×**, correct against the two-compartment arm in `EXP_009` §9.1 and wrong in a column referenced to the baseline, where it is **2.13×**. (h) "clear the adoption bar by an order of magnitude" → **4.2×** and **6.4×** (8.4σ and 12.8σ). (i) the adopted arm recovered **58.2 %** of the beyond-horizon component, not 59 %, in §2 and §7.1. (j) Identity 2 holds to **6.8–8.3** fp64 eps; `EXP_008` §9.5's "≈7" is its first two seeds. Closed experiment logs are **not** rewritten for (f)–(j); the corrected numbers live here. (k) adding §7.2 made three bare "§7.2"/"§7.3" references read as self-references when they meant `03_phase3_candidates.md`; all three are now qualified — the same slip the Phase-3 report's own changelog records as (d). |
