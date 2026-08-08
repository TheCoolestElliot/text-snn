# Phase 4 — controlled experiments: interim report

**Revision 7, 2026-08-08.** Interim, not final: Phase 4 is open and this document
is the running record of it. It exists because Phase 4 had produced six closed
experiments and no report, and a phase whose findings live only in its experiment
logs cannot be reviewed as a phase.

> **Rev 7 runs no experiment, closes nothing, and moves no verdict.** It records
> that rev 6's working-tree state is now committed, corrects three figures that
> had drifted, and flags one ranked cost estimate as unsupported by any
> measurement. **No decision row is edited and all five open decisions stay
> open.** See §13.

**Nothing in this document is a decision — every decision in it is Elliot's, and
four of what is now nine have been made by Elliot rather than here.** #8 was a
scheduling question — *does the composition run inside Phase 4?* — answered
**inside Phase 4**. **#1 is Phase-3 §9's sign-off**, signed off 2026-08-05 and
recorded as retrospective. **#5 is the learned per-channel threshold: adopted
standalone**, at rev 6 — the composed arm (`EXP_011`) stays separately
**provisional at n = 2 of a pre-registered 3** and is not thereby "the project's
best model." **#7 is the gradient clip's fp32 accumulator: fixed**, at rev 6. §8
marks all four resolved in their rows; **the other five rows — #2, #3, #4, #6, and
a new #9 — are open.**

**What rev 6 does.** Four things:

1. **Folds in `EXP_013`** (noise injection) as new §12, pushing the changelog to
   §13. The arm itself is a null — it cost bits at every amplitude and closed no
   gap — but its calibration measurement is the transferable result: this
   735,437-parameter model's generalisation gap is +0.00594 bpc (fresh) / −0.00180
   bpc (carried, indistinguishable from zero), against ~655M training characters
   and ~7.3 epochs. **The model is not overfitting.** That reframes §6.7 item 3 and
   §7 item 5's own "Phase 5 should carry a training-only arm" ask (§12 answers it,
   with a null) and removes the objection that had kept the parameter-scaling pilot
   (`03_phase3_candidates.md` §6.3 #11) off §7.1's ranking — it is added there now.
2. **Decides #5**, per Elliot's instruction: the threshold is adopted
   **standalone** — −0.0590 bpc at 23.6 se, 0/128 contexts regressed, a
   reparameterisation that folds to zero inference cost. The composed arm is
   explicitly **not** adopted by this — it remains `EXP_011`'s provisional n = 2
   result, and §7.2's gate for it is now #4 alone.
3. **Decides #7**, per Elliot's instruction: `src/snn/train.py`'s
   `_clip_grad_norm_fp64` replaces `clip_grad_norm_`, accumulating the sum of
   squares in fp64 rather than fp32 — the first of `EXP_009` §9.4's two named
   fixes. Verified against the exact failure (a synthetic 5.46e31 gradient, which
   the stock implementation still zeroes and the replacement rescales instead —
   `tests/test_grad_clip.py`) and against the real R5 capture gate
   (`tests/test_graph_equivalence.py`, 10/10, unchanged). It does **not** explain
   or prevent `compose_s0`'s divergence (§10.4) — a different mechanism, already
   established — so #7 closes one bug, not both.
4. **Adds decision #9** — whether Phase 4 pre-registrations need an explicit
   guard-clause review before a bar is committed — per `EXP_012`'s Y5 and
   `EXP_013`'s N1 firing the identical defect twice. The *protocol* rule this
   implies is **already ratified**: `CONTRIBUTING.md` §2 has carried it since
   `51b44c6` (2026-08-05, authored by Elliot, citing both experiments by name).
   What was missing was this table's own row; #9 records that the report was the
   thing lagging, not the practice.

**What rev 5 does, and it is small.** It records **decision #1** and nothing else.
It runs no experiment, moves no number, re-runs no gate and does not claim to have.
Its one consequence for the rest of the report is that `EXP_008` §4's rider —
which made the threshold arm's adoption *"recommended to Elliot, not taken"*
partly because Phase-3 §9 was unticked — **no longer applies, so #5 is ungated by
#1 and still undecided on its own merits.** Its one deliberate non-consequence:
the sign-off does **not** close **#2**, whose defect sits under a §9 box that is
now ticked, and both files say so in place rather than letting the tick absorb it.

**What rev 4 does.** It adds §11: `EXP_012`, which chases the one thing §10.5
named as unexplained and **not** offered as evidence for decision #6 — why the
composed arm's fold-in residual is ~1000× tighter than `EXP_008`'s, given the
same identity, the same six-line fold and the same metric. The answer is that
**the baseline LIF's decision variable is degenerate near its threshold**: ~11 000
sites sit within 1e-6 of it and they are **twenty distinct values replayed ~548
times each**, because layer 0 sees one of 205 embedding rows and the hard reset
zeroes the membrane exactly. The two-compartment neuron's unreset slow pole
destroys that replay — 5 sites, 4 values — and flips **1 342–5 339× fewer**
spikes under an identical perturbation.

**Three of `EXP_012`'s six predictions failed, and two of the failures are at the
resolution floor.** Its own decision rule, applied verbatim, returns a verdict
(*"an operating-point effect, not a neuron effect"*) that §11.4 explains must not
be read as the finding — because the leg that carried the failure compares **0
flips against 1**, which that experiment's limitations section had declared
unresolvable *before* it ran. The rule is **reported as it fired and is not
repaired.** In consequence **§6.5's noise-floor sentence stands exactly as
written** (`EXP_012` §4's trigger did not fire), and the scope qualifier its
evidence supports is **referred, not applied**.

**Rev 4 changes no earlier number, verdict or threshold**, adopts nothing, and
moves none of §8's seven open rows.

**What rev 3 does.** It adds §10: `EXP_011`, the composition of the adopted
two-compartment neuron with `EXP_008`'s learned per-channel threshold. **The arms
compose** — 2.08326 carried against the adopted arm's 2.11869, at −28.7 se, with
0/128 contexts regressed — and the result is **provisional at n = 2 of a
pre-registered 3**, because one seed diverged and was not replaced. Two findings
in it are worth more than the headline: **the composition's entire gain is the
within-reach component**, which confirms `EXP_004` §10.6's overlap argument in
its strong form while relocating where the value is; and **the dead seed did not
die of `EXP_009`'s bug**, which decision #7 would have fixed.

**Rev 3 changes no earlier number, verdict or threshold**, and it adopts nothing.

**What rev 2 did.** It read `EXP_007`, `EXP_008` and `EXP_009` *together* rather
than one after another (§6.7), stated a verdict for every Phase-4 arm against its
own pre-registered rule (§6.8), re-ran every gate in §9 from a clean tree, and
corrected five numbers rev 1 got wrong (§13, renumbered from §12 at rev 6 — see
this changelog's own rev-6 entry). **No verdict, threshold or decision
changed.** Two of the corrections make an arm look worse, one makes the adopted
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
| Experiments closed | `EXP_004`, `EXP_005`, `EXP_006`; `EXP_007`, `EXP_008` and `EXP_009` in §6; **`EXP_011` in §10**; **`EXP_012` in §11**; **`EXP_013` in §12** |
| Ranked candidates closed | 5 of 14 (#1, #14, the horizon question §10.5 opened, and two of the three I5 referrals run as diagnostics). **`EXP_011`, `EXP_012` and `EXP_013` close no ranked candidate** — the first composes two arms already counted here, the second is a mechanism study and trains nothing, the third is not on the 18-item list at all (`EXP_013` §0) |
| Arms adopted | **2** — the two-compartment neuron (`EXP_004`) and, **at rev 6, the learned per-channel threshold standalone** (`EXP_008`, decision #5). The composed arm (§10) is **not** adopted — provisional at n = 2 |
| GPU-hours spent | **~6.5 of ~30** (`EXP_012` cost ~0.2 and trained nothing; `EXP_013` cost ~1.9, its §9 measured total rather than its §0 pre-registration estimate of ~1.6) |
| Unexplained divergences | **2** — `twocomp_distill_s1` (§6.4) and `compose_s0` (§10.4), chased to **different** causes. Neither is `EXP_013`'s: the noise arm trained cleanly at every amplitude |
| Phase-3 §9 sign-off | **signed off 2026-08-05** (decision #1), and **recorded as retrospective**: Phase 4 was opened without it on instruction and eight experiments ran before it |
| Decisions open | **5 of 9** (§8). #8 resolved at rev 3, #1 at rev 5, #5 and #7 at rev 6, and rev 6 adds a new #9 |

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
a column whose reference row is the baseline, where it is **2.13×**. §13 records
both (renumbered from §12 at rev 6).

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

*(Rev 6: this is "neither of **these** is capacity" — `EXP_013` §2.2 measures a
different, complementary sense at §12.2: the model as a whole is not
capacity-constrained in the overfitting sense, so nothing here should be read as
evidence that added capacity would go to waste. None of Phase 4's arms adding no
function class is not the same claim as the model having no room to grow.)*

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

> **Rev 4 — a scope qualifier is REFERRED here and deliberately not applied.**
> The sentence above says *"this architecture"*. `EXP_012` (§11) measured the same
> fold on the **adopted** neuron and got **1.4e-05 / 9.3e-06** on the same
> 512-window metric that gives 2.0e-03 / 6.2e-04 / 1.6e-03 here — and traced the
> difference to a degeneracy that exists in the baseline LIF's decision variable
> and not in the two-compartment one (§11.3, §11.6). On that evidence the floor
> looks like a property of **the neuron it was measured on**, not of the
> architecture family, which would be `EXP_005`'s σ-non-transfer lesson in a new
> place.
>
> **It is not applied, for a reason that is not a formality.** `EXP_012` §4
> pre-registered exactly this correction behind a trigger — `Y2 ∧ Y5` — and
> **Y5 failed** (§11.4). Applying the qualifier anyway would be adopting a
> correction whose own pre-registered gate did not open, immediately after
> watching it not open. **§6.5 therefore stands as written**, and the qualifier is
> §11.7's referral (`EXP_012` §9.7 item 9), for Elliot.

> **Rev 6 — a fourth data point on floor-vs-signal, and it is the one this section
> warned would be needed.** `EXP_013` §2.2's whole generalisation-gap measurement
> is **+0.00594 bpc (fresh)**, only ~3× this ~2e-3 floor, and **−0.00180 bpc
> (carried)** sits *below* it entirely — indistinguishable from the floor itself.
> `EXP_013` §5 item 7 and §9.2 already carry this guard internally (any `Δgap`
> under the floor is reported as "below what this instrument resolves," not as a
> signed number — see §12.2/§12.4). This is the first time a headline figure this
> project computed, rather than a diagnostic residual, has landed *inside* the
> floor rather than merely near it; nothing here is a correction to §6.5's own
> claim, which is about reparameterisation residuals specifically.

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

*(Rev 6: this item is what `EXP_013` was run to answer — its own §0 item 2 quotes
this exact paragraph as its justification for existing. §12 reports a null: the
one training-only arm tried costs bits at every amplitude and closes no gap.
That does not fill the gap this item names — a null does not — but the
calibration `EXP_013` measured on the way there does something more useful for
Phase 5's own ranking: it removes the standing objection to the
parameter-scaling pilot (§7.1 rank 2, newly added at rev 6), which had no
evidence there was room to grow until now.)*

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
§8 items 1 and 5 are where that lives. *(Rev 5: **§9 is now ticked** — #1 is
signed off — so the second half of that rider is discharged and #5 is ungated by
#1. **Nothing here is adopted by this report** is unchanged: #5 is Elliot's and is
still undecided, and this section's verdicts are untouched.)*

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
   *(Rev 6: tried, once — `EXP_013`, §12. Null result, transferable calibration:
   this model is not overfitting, which is item 6 below.)*
6. **Added at rev 6. The model has room to grow, and nothing has tested it.**
   `EXP_013` §2.2 measured this model's generalisation gap directly for the first
   time in the project's history: +0.00594 bpc (fresh, 4.96 se from zero but below
   the 2σ adoption bar) and −0.00180 bpc (carried, indistinguishable from zero).
   735,437 parameters against ~655M training characters is ~7.3 epochs — an
   underfitting regime, not an overfitting one. This is a *referral*, not a
   ranking made here (`EXP_013` §10 item 4 refers it the same way): it removes the
   standing objection to `03_phase3_candidates.md` §6.3 #11 (the parameter-scaling
   pilot) and argues against ranking further regularisation-flavoured candidates
   ahead of capacity-adding ones.

### 7.1 A revised ranking, offered and not applied

`03_phase3_candidates.md` §6.3's list is a Phase-3 artifact and is not edited
here. This is what Phase 4's evidence implies about it, for Elliot to accept or
reject:

| rank | arm | why | GPU-h |
|---:|---|---|---:|
| ~~1~~ | ~~threshold × two-compartment composition~~ | **DONE at rev 3 — this is `EXP_011` (§10).** Kept in the table struck through rather than deleted, so the ranking still reads as the list rev 2 offered. The arms compose; the gain is within-reach, not zero-context | ~~0.5~~ → 0.86 actual |
| **2** | **#11 parameter-scaling pilot** (0.735M / 2.5M / 5M params, 1 seed each) | **added at rev 6, and newly the best-motivated item here.** Absent from this table through rev 5 for lack of evidence there was room to grow; `EXP_013` §2.2 supplies exactly that evidence (no overfitting signal at 735K params / ~7.3 epochs). Tells Phase 5 what size to run and bounds every other ceiling | 0.35 |
| **3** | **#2 current normalisation (RMSNorm on `cur`)** | the real §6.3 #2, passes the Phase-3 §7.2 screen, attacks the same components by a different route, still unrun | 0.4 |
| **4** | **why the threshold arm helps within reach** | a mechanism study, not an arm. The only measurement pointed at the largest remaining component | 0.3 |
| **5** | **frozen-`beta_s` ladder**, N ∈ {16, 64, 256} | the only thing that can license tail-widening (§4.5) | 1.3 |
| **6** | #13 training budget (40k steps) | the cheapest pure-optimisation lever, unmeasured, and — per `EXP_013` §2.2 — no longer competing against a regularisation reading of the same evidence | 0.7 |
| — | #5 learned per-channel decay | partly subsumed by the adopted arm | — |
| — | #6 adaptive threshold | **blocked**: both its inits are saddles, and its fallback needs #8 first (`03_phase3_candidates.md` §7.3) | — |
| — | #10 rotational membrane | screened init is a saddle; θ = π/8 fallback passes but the mechanism is unmotivated by any Phase-4 finding | — |

**Demoted on evidence, not on taste:** every beyond-horizon candidate. That
component is 28 % of what is left, the adopted arm already took 58.2 % of it, and
`EXP_006` withdrew the argument that more is available.

### 7.2 The order it should run in, and what has to be settled first

Recommended, not scheduled. The whole of §7.1 is ~3.2 GPU-hours against the ~26.5
still unspent, so **the constraint is review, not budget**.

> **Rev 7 corrects both figures, and flags that one of them rests on nothing.**
> The arithmetic first: ranks 2–6 sum to **3.05**, not 3.2, and §1 records ~6.5
> of ~30 spent, so **~23.5** remains, not ~26.5 — a rev-2 figure carried forward
> through four revisions. The conclusion is unchanged and strengthened: the
> constraint is review, not budget.
>
> The substantive correction is that **rank 2's 0.35 GPU-h is not a measurement
> and no measurement in this tree supports it.** It is a Phase-3 estimate made
> when the only width evidence available was `01_reconnaissance.md` §3.5's
> *eager, forward-only, `no_grad`* sweep ("a 16× width increase is free") — which
> §6.2 of the next report withdrew, in a section titled *"The 14.5 µs constant
> does not transfer"* whose closing instruction is that Phase 3's ROI estimates
> must not reuse it. `EXP_003`'s depth ladder is the only other multi-width
> evidence and it holds `K·d²` nearly constant, so `K`, `K·d` and `K·d²` are
> collinear over its four points and cannot price width at all. A GEMM-FLOP model
> built from `audit_01_gpu_capability.json`'s measured 11.91 TFLOP/s puts the
> pilot at **~0.9–1.4 GPU-h, i.e. 2.6–4× the ranked figure**. That model is
> itself unvalidated at width, so **no corrected number is written into the table
> here**: `EXP_014` opens with a pre-flight calibration that measures it, and a
> later revision records what it measured. The other five rows in §7.1 rest on
> estimates of the same vintage and should be read with the same suspicion until
> each is measured.

**Before anything in §7.1 starts, three of §8 need answers**, and they are cheap
ones: **#1** (Phase-3 §9) because `EXP_008` §4's rider makes adoption conditional
on it; **#5** because rank 1 measures the composition of an arm that is only
*recommended*, and if #5 is "no" the composition run answers a question about an
arm nobody is shipping; **#4** because it decides whether any of this is Phase 4
or Phase 5 — which is the new decision #8.

> **Rev 3, recorded rather than quietly dropped: rank 1 ran without #1 or #5
> being answered.** Decision #8 was resolved "inside Phase 4" and `EXP_011` ran on
> that instruction, so the caveat this paragraph raised is now live rather than
> hypothetical — **the composition is measured on top of an arm that is
> recommended and not adopted, and against a Phase-3 sign-off that is still
> unticked.** That does not weaken §10's numbers, which stand on their own
> pre-registration; it means §10 cannot be read as "the project's best model" until
> #1 and #5 are answered. The remaining items in §7.1 still have this paragraph's
> prerequisites in front of them.
>
> **Rev 5: #1 is answered — signed off 2026-08-05 — and #5 is not.** One of this
> paragraph's two conditions is met, so §10 still cannot be read as "the project's
> best model", and the retrospective character of the sign-off does not change what
> rev 3 recorded here: rank 1 *did* run before either was answered, and this
> paragraph is left standing as the record of that. **§7.2's gate is now #5 and
> #4.**
>
> **Rev 6: #5 is answered too — the threshold is adopted standalone.** Both of
> this paragraph's original conditions are now met; the one still open is **#4**,
> Phase-5 authorisation. Elliot has separately instructed items below to start
> inside Phase 4 rather than wait for #4 — the same scheduling precedent decision
> #8 set for the composition (§10). **This revision records that instruction and
> the ranking it acts on; it does not itself report a closed experiment against
> it.** A future revision closes whatever runs.

Given those, the order below is not arbitrary — each step's result changes whether
the next is worth running:

1. **Composition (§7.1 rank 1, 0.5 GPU-h).** Threshold on top of the adopted
   neuron. It is first because every other ranking depends on the answer, and
   because §7 item 4 says the two gains may not be added — so today the project
   cannot state what its best model scores. **Pre-register the non-additive
   prediction before running it**, or a partial gain will read as a disappointment
   rather than as the measurement `EXP_004` §10.11 item 2 asked for. **Done —
   `EXP_011`, §10.**
2. **Parameter-scaling pilot (rank 2, 0.35 GPU-h)** — added at rev 6. Three
   sizes, one seed each; `EXP_013` §2.2 is the motivating evidence and §7 item 6
   states it. Run before RMSNorm and the mechanism study below, since its answer
   (does capacity move the number at all) bears on how the other two are read.
3. **RMSNorm on `cur` (rank 3, 0.4 GPU-h)** — `03_phase3_candidates.md` §6.3's
   actual #2, unrun, attacks the same components by a different route, and its
   result is only interpretable against step 1's.
4. **Why the threshold arm helps within reach (rank 4, 0.3 GPU-h).** A mechanism
   study, not an arm. §6.7 item 1 is the strongest unexplained result in the
   phase, and it points at the largest remaining component.
5. **The frozen-`beta_s` ladder (rank 5, 1.3 GPU-h)** — the only thing that can
   license tail-widening (§4.5), and the only reason to run it is to *close* that
   question rather than leave it open into another phase.
6. **A training-signal arm (rank 6, 0.7 GPU-h)** — §7 item 5. `EXP_009`'s λ was
   one unswept point and §9.3 of that file forbids reading its null as a ceiling;
   §7 item 5 notes the training-*signal* half of this ask was tried once
   (`EXP_013`, a null) and training-*budget* remains unmeasured.

**Two things Phase 5 should not do.** It should not open on the beyond-horizon
candidates the Phase-3 list still ranks first — §7.1's demotion is on evidence.
And it should not carry `EXP_008`'s gain into a headline until #6 is ruled: the
number rests on a check that failed as written, and §6.8 says so wherever it is
quoted.

---

## 8. What is Elliot's, and is still open

**Seven of these eight are Elliot's, are undecided, and are unchanged by rev 3 —
not one of rows 1–7 has been edited.** Rev 2 sharpened the evidence under #3, #4
and #5 without moving any of them; rev 3 moves none of them either. *(Rev 5: **#1
is now signed off**, so the count is **six** open, and row 1 **is** edited — the
first row in this table to be edited since #8. This paragraph records what was
true at rev 3 and is left as it was; the rev-5 note below the table carries the
change, and rev 4 likewise edited nothing. Rev 6: **#5 and #7 are now also
decided** and their rows edited, and a new row **#9** is added open — the table
now has nine rows, four resolved (#1, #5, #7, #8) and five open (#2, #3, #4, #6,
#9). This paragraph is left exactly as rev 3 wrote it; the rev-6 note below the
table carries this change, as the rev-5 note carried its own.)*

**#8 is resolved, and it is the only one.** It was the single *scheduling*
question in the list — whether the composition ran inside Phase 4 or opened
Phase 5 — and Elliot directed it to run inside Phase 4. It did, as `EXP_011`
(§10). **Resolving it decides nothing about the arm it measured**: adopting the
composed arm is #5's and #1's business, and both are still open.

| # | Decision | Status | Evidence, as of rev 2 |
|---:|---|---|---|
| 1 | **Phase-3 §9 sign-off** | **RESOLVED at rev 5 — signed off 2026-08-05** | Elliot's instruction, 2026-08-05. Ticked in `03_phase3_candidates.md` §9 and **labelled there as retrospective**: Phase 4 opened without it and eight experiments ran before it, so the box's "Phase 4 does not begin until" was not honoured in sequence. Rev 2's evidence cell said *"nothing is left for it to wait on"* and that was still true at rev 4 — `EXP_011` and `EXP_012` are both Phase-4 work and neither touches a Phase-3 deliverable. **This releases the `EXP_008` §4 rider, so #5 is no longer gated by #1.** It does **not** decide #5, and it does not close #2, which is a defect under a ticked §9 box and is named as such in that file's rev 3 |
| 2 | **§6.2 bar definition** (SE-on-the-absolute-curve, per arm) | **open** | `EXP_005` §9.4. Verified since: the per-arm sd it needs is already computed from the ≥3 seeds the adoption rule requires, so the cost `EXP_005` flagged is already paid |
| 3 | **The three I5 boundary questions** | **open**, and **now cheap** | token-shift and distillation ran as labelled diagnostics. §6.7 item 2: token-shift is dominated by the threshold arm on bits (0.0203 bpc, 13.3 se), wall-clock (1.49× vs 1.12×), VRAM and inference cost, and its own second tap is worth −0.0009 bpc. **Ruling it either way changes nothing about what to build** |
| 4 | **Phase 5 authorisation** | **open** | recommend **not yet**: 5 of 14 candidates closed, ~3.5 of ~30 GPU-hours, and §7.1 item 1 — whether Phase 4's two best arms compose — is unmeasured and costs 0.5 GPU-hours |
| 5 | **Adopt the learned per-channel threshold?** | **RESOLVED at rev 6 — adopted standalone** | Elliot's instruction, 2026-08-08. −0.0590 bpc at 23.6 se (6.4× the bar), 0/128 contexts regressed, 1.12× wall-clock, folds to zero cost — a reparameterisation, confirmed to 6.8–8.3 fp64 eps (§6.3). **The composed arm (`EXP_011`, §10) is explicitly not adopted by this**: it stays provisional at n = 2 of a pre-registered 3 per `CONTRIBUTING.md` §4's provisional-promotion rule (ratified 2026-08-08 — seeds are *added*, never substituted, up to the pre-registered n before "provisional" becomes "adopted"), and is not "the project's best model" until reseeded |
| 6 | **The fold-in tolerance** | **open** | `EXP_008` W1 failed at 1e-3, a bar borrowed from F1, which compares two evaluations of the *same* weights. §6.5 measures why that was the wrong precedent; the algebra holds at 6.8–8.3 fp64 eps. **Referred, not redefined** |
| 7 | **The gradient clip's fp32 accumulator** | **RESOLVED at rev 6 — fixed** | Elliot's instruction, 2026-08-08. `src/snn/train.py`'s `_clip_grad_norm_fp64` accumulates the sum of squares in fp64, the first of `EXP_009` §9.4's two named fixes. Verified on the exact failing gradient (5.46e31, stock zeroes / fix rescales — `tests/test_grad_clip.py`) and against the real R5 capture gate (`tests/test_graph_equivalence.py`, 10/10). **Does not explain or prevent `compose_s0`'s divergence** (§10.4) — a different, still-open mechanism |
| **8** | **Does the composition experiment run inside Phase 4, or as Phase 5's first arm?** | **RESOLVED at rev 3 — inside Phase 4** | Elliot's instruction, 2026-08-04, matching rev 2's recommendation. Ran as `EXP_011` (§10): the arms compose at 2.08326 carried, −28.7 se, 0/128 contexts regressed, **provisional at n = 2** after one seed diverged and was not replaced. The composed arm is a **Phase-5 candidate; adopting it is #5's and #1's business and neither has moved.** ~0.86 GPU-hours including the probe and the divergence chase |
| **9** | **Should a pre-registered bar require a guard-clause review before it's committed?** | **open — added at rev 6** | `EXP_012`'s Y5 and `EXP_013`'s N1 (§12.4) fired the identical defect: a bar specified without a guard the experiment's own limitations section had already written down. `EXP_013` §10 item 2 named this report by section as the place lacking a row for it. **The protocol rule is already ratified** — `CONTRIBUTING.md` §2, since `51b44c6` (2026-08-05, Elliot, citing both experiments) — so this row tracks the report's own bookkeeping, not an open question of substance |

**Two rows above now carry evidence `EXP_011` has overtaken, and the rows are
deliberately left as they were.** Editing an open decision's evidence is how a
decision gets made quietly, so the correction is recorded here instead:

* **#4 (Phase 5 authorisation)** reads *"§7.1 item 1 — whether Phase 4's two best
  arms compose — is unmeasured and costs 0.5 GPU-hours."* **It is now measured**
  (§10) and it cost ~0.86 GPU-hours. That removes rev 2's stated reason for
  recommending "not yet"; the other reasons stand — 5 of 14 candidates closed and
  ~4.4 of ~30 GPU-hours spent — and **#4 is still open and still Elliot's.**
* **#7 (the gradient clip's fp32 accumulator)** reads that the fix is independent
  of the three new arms. It still is. But §10.4 shows the clip overflow is **not**
  what killed `compose_s0`, so **fixing it would not have prevented Phase 4's
  second divergence.** That is an argument for chasing the second mechanism, not
  against fixing the first, and **#7 is unchanged and still open.**

Rev 3 adds no new decision. If the composed arm is to be adopted, that is #5 and
#1, already on the list.

**Rev 4 adds no new decision either, and edits no row. It sharpens #6, which was
the one item on this list that could not be decided at all**, and the correction
is again recorded here rather than in the row:

* **#6 (the fold-in tolerance)** reads that `EXP_008`'s W1 failed at 1e-3 and that
  §6.5 measures why the precedent was wrong. Both still stand. What was missing is
  that the project's only *other* fold-in measurement — `EXP_011` C4 at 1.05e-06 —
  disagreed with it by three orders of magnitude, and §10.5 explicitly withheld it
  as evidence "until understood". **`EXP_012` (§11) understands it**: the two
  residuals differ because the two neurons put wildly different amounts of
  probability mass at the firing threshold, and `EXP_011`'s number is now
  available to #6 rather than withheld from it.
* **What that does to the decision is narrow and worth stating plainly.** It does
  **not** propose a tolerance — `EXP_012` sets none, by design. It establishes
  that **a fold-in tolerance cannot be one project constant**: the same identity,
  through the same fold, costs 2.0e-03 bpc on the baseline neuron and 1.4e-05 on
  the adopted one. Any single number for W1 will be far too loose for one arm or
  far too tight for the other, and **that is the shape of the choice #6 is**.
* **#4 (Phase 5 authorisation)** now reads against **~4.6** of ~30 GPU-hours, not
  the ~3.5 in its row or the ~4.4 rev 3 corrected it to; `EXP_012` cost ~0.2 and
  trained nothing. **5 of 14 candidates closed is unchanged** — `EXP_012` is a
  mechanism study and closes none. **#4 is still open and still Elliot's.**

**Rev 5 edits exactly one row — #1 — because #1 is now decided, and a decided row
is edited rather than annotated.** That is `#8`'s precedent at rev 3, not a new
one: the rule this report follows is that an *open* decision's evidence is never
edited (the correction goes in prose beneath the table, as rev 3 and rev 4 both
did), while a decision that has actually been made is recorded in its own row so
the table cannot be read as still asking the question.

* **#1 (Phase-3 §9 sign-off)** — **signed off by Elliot on 2026-08-05.** The tick
  is in `03_phase3_candidates.md` §9 and is **labelled retrospective in place**,
  because Phase 4 opened without it and `EXP_004`–`EXP_009`, `EXP_011` and
  `EXP_012` all ran before it. Nothing in `EXP_011` or `EXP_012` bore on it either
  way; both are Phase-4 work and neither touches a Phase-3 deliverable, so rev 2's
  *"nothing is left for it to wait on"* was still accurate when it was ticked.
* **What #1 does to #5, and it is only this.** `EXP_008` §4's rider made the
  threshold arm's adoption *"recommended to Elliot, not taken"* on the ground that
  "§9 of the Phase-3 report is still unticked". That ground is gone. **#5 is
  therefore ungated by #1 and is still undecided** — its own substance, and its
  coupling to #6, are untouched by this tick. **Six decisions remain open.**
* **What #1 does not do.** It does not close **#2**, and #2 is a defect sitting
  under a §9 box that is now ticked: §9's "acceptance criterion derived from
  measurement and calibrated per-context" is ticked, and `EXP_005` §9.4 has since
  shown that bar built on the wrong scale. The sign-off does not launder it. It is
  named in `03_phase3_candidates.md` rev 3, it remains Elliot's, and the count it
  governs is unchanged at 5/128 under all three constructions.
* **§7.2's first prerequisite is met, and its other two are not.** That paragraph
  requires #1, #5 and #4 before anything in §7.1 starts. One of three.

**Rev 6 edits two rows — #5 and #7 — because both are now decided, and adds one
new row, #9, for a decision that did not exist before.**

* **#5 (adopt the learned per-channel threshold?)** — **adopted standalone**, per
  Elliot's instruction. §7's item 4 composition warning and §6.5's ~2e-3 drift
  caveat are carried into the row rather than dropped: adopting the threshold
  does not adopt the composition, which stays `EXP_011`'s provisional n = 2 result
  until reseeded. **§7.2's second prerequisite is now met; only #4 remains.**
* **#7 (the gradient clip's fp32 accumulator)** — **fixed**, per Elliot's
  instruction. `_clip_grad_norm_fp64` (`src/snn/train.py`) replaces the stock
  `clip_grad_norm_` call in `Trainer._step_body`, upcasting each gradient to fp64
  before squaring rather than after — the exact computation
  `009_chase_divergence.py` used to diagnose the bug, now load-bearing instead of
  diagnostic. `tests/test_grad_clip.py` reproduces `EXP_009` §9.4's numbers
  directly: a synthetic 5.46e31 gradient, which the stock implementation still
  zeroes (`clamp(1.0/(inf+1e-6), max=1.0) = 0`, exactly) and the replacement
  rescales to a finite, nonzero update. `tests/test_graph_equivalence.py`'s full
  10-test R5 gate — both the static capture-safety AST checks and the real
  graphed-vs-eager equivalence tests on this GPU — pass unchanged. **This closes
  one bug, not two**: §10.4 already established `compose_s0`'s divergence is a
  different mechanism (ordinary-magnitude gradients, fp32/fp64 norms equal,
  identical NaN on the eager path), so this fix would not have prevented it and
  does not claim to.
* **#9 (guard-clause review before a bar is committed) — added, and open.**
  `EXP_012`'s Y5 (§11.4) and `EXP_013`'s N1 (§12.4) are the same defect twice: a
  pre-registered bar on a difference, committed without constraining what the
  subtrahend is allowed to do, satisfiable by a sufficiently damaged model
  without any real improvement. `EXP_013` §10 item 2 asked this report by name to
  gain a row for it. **The protocol side of this is not open** —
  `CONTRIBUTING.md` §2 has carried the rule since `51b44c6` (2026-08-05, authored
  by Elliot, citing `EXP_012`'s Y5 and `EXP_013`'s N1 by name, predating this
  report's own row for it). Row #9 exists so this table stops being the one place
  that had not caught up.
* **Two further `EXP_013` referrals, noted rather than tabled.** §10 item 3
  (whether noise injection belongs on `03_phase3_candidates.md`'s ranked list at
  all) and item 4 (the no-overfitting finding's bearing on future candidates,
  captured above in §6.7 item 3, §7 item 6 and §7.1's new rank 2) are Elliot's to
  rule on but do not need their own numbered row — neither is a yes/no this table
  format fits, and both are fully stated where they already live.

---

## 9. Gates, at the close of this revision

**Every row was re-run for rev 3 on 2026-08-04**, and the suite and the two
`EXP_012` rows again for **rev 4 on 2026-08-05**. Rev 2's rows were re-run from
the clean tree at `dcc8f82`; rev 3 re-ran them with `EXP_011` in the tree and
added four rows; rev 4 adds three.

| Gate | State |
|---|---|
| Test suite | **394 passed**, 1 warning, 47 s (full: `pytest`, GPU + corpus). Rev 4's 280, fully accounted rather than assumed: **+85** `tests/test_snnchat.py` (the chat track's own suite, committed `31e49fa` between rev 4 and rev 6 and untouched by this revision), **+23** `tests/test_noise.py` (`EXP_013`'s gates), **+6** `tests/test_grad_clip.py` (decision #7's fix, new at this revision) — 280 + 85 + 23 + 6 = 394 exactly. Fast subset (`pytest -m "not cuda and not data"`, no GPU/corpus): **256 passed, 1 skipped, 8 s** |
| **`EXP_013` G1 — the nesting** | `arch="noise"` at `noise_amp = 0` reproduces `SpikingCharLM` **bitwise, forward and backward** (`torch.equal` on logits and every gradient) — licenses scoring the five committed baseline seeds as the noise-off control instead of training three more |
| **`EXP_013` G3 — the draw is centred and correctly scaled** | on a large draw, sample mean within 4 se of 0 and sample sd within 1 % of `amp`; the mutation leg (an uncentred draw) is shown to fail it |
| **`EXP_013` G4/G5 — resumability and independence** | the noise at `(seed, step)` is a pure function of `(seed, step)` — refilling twice at the same step gives bitwise-identical buffers; the salted generator is not reproducible from the data sampler's stream at the same step |
| **`EXP_013` F1** | **9 / 9**, residuals 1.45e-09 – 1.18e-08, `f1_failures` empty — every gap evaluation's test leg reproduces the committed 2.26969 / 2.25311 |
| **`EXP_013` K1** | every noise run's `config.json` against `snn_beta0.5_s0`'s, field-by-field: only `arch`, `noise_amp`, `run_name` differ; `beta_slow`, `w_init`, `mu_init`, `thr_log_init`, `noise_p` excused as absent-from-reference at the `Config` default |
| **`EXP_013` calibration self-check** | `013_calibrate_noise_scale.py` replays `_CharLMStack.forward`'s loop by hand to observe `cur`, and every replay's logits are asserted bit-identical to `model(idx)`'s: **25 / 25** |
| **`EXP_013` pre-registration hash** | §0–§8 verified byte-identical to what was hashed before the first training run: SHA-256 prefix `285eb661…82ef8d`. No hypothesis, bar or decision cell moved after the numbers arrived |
| **Decision #7 — `_clip_grad_norm_fp64` reproduces the fix, not just the intent** | `tests/test_grad_clip.py`: a synthetic 5.46e31 gradient (`EXP_009` §9.4's own number) — stock `clip_grad_norm_` returns `inf` and zeroes every gradient exactly; the replacement returns a finite ~5.46e31 and rescales instead. Static AST checks (mirroring `test_step_body_has_no_python_control_flow`) confirm no host sync and no value-dependent branch in the new helper. The real R5 capture gate, unchanged: **10 / 10** |
| **`EXP_012` S3 — the environment has not drifted** | `008_chase_fold.py` re-run to a scratch path reproduces the committed `exp_008_fold_residual.json` at **0 differences**, every field, bit-for-bit. Run **before** any cross-arm comparison was formed, because the comparison is the experiment |
| **`EXP_012` S1 — the probe reproduces the committed kernel** | leg 3 must recompute the membrane recursion to observe `u` at all, so its spikes are asserted **bit-identical** (`torch.equal`) to `lif_scan` / `twocomp_scan` on every run and both layers: **10/10 pass**. Guarded in the suite by `tests/test_fold_gap_probe.py` across three firing rates, with **two mutation legs** — resetting `vs`, and a soft reset — so the check has been seen to fail |
| **`EXP_012` sets no tolerance** | asserted structurally: `resolve()` reads its bars from constants transcribed at the top of the file, and the artifact carries `"sets_no_tolerance": true` and `"decision_6": "UNTOUCHED AND OPEN"`. No fold in this experiment is gated |
| **`EXP_011` C5/K5 — Identity 1 on the two-compartment neuron** | float64, at a nowhere-zero per-channel `θ`, against a reference transcribed from `EXP_011` §1.1 rather than from the source: **spikes exact, membranes ≤1e-12** on four (`beta_f`, `thr`) regimes. A mutation leg asserts the check can fail. Run as the driver's entry condition **before** any training step |
| **`EXP_011` K4 — the composed arm nests the adopted arm** | at `θ = 0`, forward and backward **bitwise** (`== 0.0`, not a tolerance) against `TwoCompartmentCharLM` |
| **`EXP_011` K1 — the arm is the adopted arm plus one thing** | every composed run's `config.json` field-by-field against **`twocomp_s0`** (not the Phase-2 baseline): only `arch`, `run_name`, `seed`, `thr_log_init` differ; `mu_init` excused at the `Config` default and recorded in the manifest |
| **Resolvers reproduce their own artifacts** | `010_phase4_arm_results.py`: **0 differences** against the committed `exp_007_009_arm_results.json`. `011_compose_results.py` re-run to a scratch path: **0 differences** against `exp_011_arm_results.json` — every reference, decomposition, prediction and both GPU fold evaluations, bit-for-bit |
| Committed references recomputed | the resolver's own guard: baseline **2.25311**, twocomp **2.11869**, GRU **1.76741** carried, each recomputed from per-run `final_test.json` and asserted against the figure the reports quote (worst residual **1.5e-5**, tolerance 5e-4) |
| Phase-3 §7.2 reachability screen | re-run: `thr` **PASS**, `ctl_live` **PASS**, `ctl_dead` SADDLE and `ctl_faint` FAINT as expected, all three self-tests held, and it reproduces `exp_004_gradient_reachability.json` at **0.00e+00** worst relative residual |
| R10 gradient gates | untouched. `EXP_007`/`EXP_008`'s arms call the committed `lif_scan`; **`EXP_011`'s calls the committed `twocomp_scan`** and adds no hand-written backward. Independently, §10.4's pass 3 replayed `compose_s0`'s failing step on the **eager** path and got an identical NaN, which exonerates the fused two-compartment backward rather than merely leaving it unaccused |
| `EXP_001` F1 (probe reproduces committed bpc) | **23 / 23** in the `EXP_007`–`EXP_009` artifact; **17 / 17** in `exp_011_memory_horizon.json` (worst residual 1.3e-08), `f1_failures` empty in both |
| Mutation campaign | none run during any training run; lockfile checked before each, including all three `EXP_011` runs |
| Sequential GPU use | the `EXP_011` driver blocks on each child in turn, and the horizon probe and divergence chase ran only **after** the last training run exited — so no wall-clock figure in §10.2 overlaps other GPU work |
| Working tree | rev 2 modified exactly one file and committed nothing. **Rev 3 is different and says so:** it adds `EXP_011`'s pre-registration, arm, gate, driver, resolver and chase, a registry entry in `001_memory_horizon.py`, four artifacts under `docs/reports/data/`, three run directories, and this file. Every piece was **committed before the numbers it produces were read** — the pre-registration at `f0eebe4`, the arm and its gate at `ca94f01`, the driver at `5e3d5e5`, the resolver at `bc0dd08` |

**Phase 4 remains open.** Rev 1 closed `EXP_007`, `EXP_008` and `EXP_009`; rev 2
synthesised them; **rev 3 closes `EXP_011` and resolves the one scheduling
decision that authorised it.** It authorises no phase, adopts no arm, and
redefines no threshold — and seven decisions in §8 are untouched.

**Rev 4 closes `EXP_012`**, which trains nothing, adopts nothing and sets no
tolerance. It leaves all seven decisions open and all seven rows unedited; what
it changes is that **#6 is now a decision that can be made** — §10.5's withheld
evidence is released to it, and the shape of the choice is stated in §8's rev-4
note. It also **declines to apply** the §6.5 scope qualifier its own evidence
supports, because the trigger it pre-registered for that correction did not fire.

**Rev 5 runs no experiment, adds no gate row and changes no number.** It records
Elliot's sign-off of Phase-3 §9 — decision **#1**, the first of the seven to be
decided — in that file's §9 and changelog and in §8 row 1 here. **Six decisions
remain open**, Phase 4 remains open, no arm is adopted, no tolerance is set and no
phase is authorised. The gate table above was **not** re-run for rev 5, and is not
claimed to have been: rev 5 touches two report files and no code, no artifact and
no run directory, so every row still describes the tree at rev 4's commit
`677793d`.

**Rev 6 closes `EXP_013`, decides #5 and #7, and re-runs the full suite for
real** — both marker sets above are literal `pytest` output, not carried forward.
It adopts the threshold arm standalone (#5) and fixes the gradient clip (#7); it
does **not** adopt the composed arm, does **not** set a fold-in tolerance (#6
unchanged), and does **not** authorise Phase 5 (#4 unchanged, still "not yet").
The mutation campaign was **not** re-run for rev 6 — nothing in `snn/*.py`'s
kernels changed, only `Trainer._step_body`'s clip call, which the mutation
campaign does not target — and this row says so rather than implying otherwise.

---

## 10. `EXP_011` — the two best arms compose, and the seed that died proving something else

Pre-registered in `experiments/logs/EXP_011_composition.md` and closed the same
day. It is §7.1's rank-1 candidate and the measurement `EXP_004` §10.11 item 2
asked for in advance. **Three training runs, ~0.51 GPU-hours**, plus ~0.35 for
the 17-checkpoint horizon probe and the divergence chase.

### 10.1 The arm, and why its parameters add nothing

`arch="twocomp_threshold"`: the adopted neuron with `snn.prescan.threshold_gain`
on the current before the scan. `EXP_011` §1.1 derives, **before any code
existed**, that `EXP_008`'s Identity 1 extends to the two-compartment neuron —
both poles scale linearly with the current, the mix is linear, and the fast reset
is multiplicative, so `cur → g·cur` gives `v → g·v` and firing at `g·v ≥ thr` is
firing at `v ≥ thr/g`. The reset-shielded slow pole carries the induction
untouched, which is the step that had to be redone rather than inherited.

`tests/test_compose_equivalence.py` checks it in float64 against a
per-channel-threshold reference transcribed from the derivation rather than from
the source, at a nowhere-zero `θ` so it cannot pass vacuously: **spikes exact,
membranes ≤1e-12.** So the composed arm's 1 024 extra parameters are **exactly
redundant** over the adopted arm — +0.14 % in parameters, **+0.00 % in
function-space dimension** — and whatever it buys is an optimisation effect.
*(Rev 6, same caveat as §6.3: this is a statement about function class, not about
whether the model has room for more parameters at all — `EXP_013` §12.2 measures
that separately, and it is not "no".)*

### 10.2 The scoreboard

| | n | test bpc, carried | vs adopted | se | wall-clock | peak VRAM |
|---|---:|---:|---:|---:|---:|---:|
| Phase-2 baseline | 5 | 2.25311 | — | — | 1.00× | 0.609 GiB |
| learned threshold | 3 | 2.19416 | — | — | 1.12× | 0.734 GiB |
| two-compartment *(adopted)* | 7 | 2.11869 | — | — | 1.32× | 0.980 GiB |
| **composed** | **2** | **2.08326 ± 0.00049** | **−0.03543** | **−28.7** | **1.41×** | 1.105 GiB |
| GRU anchor *(violates I5)* | 3 | 1.76741 | — | — | 1.11× | 0.864 GiB |

Wall-clock is against `snn_beta0.5_s0`'s 398.9 s, the standing denominator;
against the adopted arm's own three-seed 525.3 s the composed arm is **1.074×**.
Seeds: 2.08291 and 2.08360. **`compose_s0` diverged and is excluded, not
replaced** (§10.4). The gap to the GRU anchor narrows 0.35128 → 0.31585.

**60.1 % of the threshold arm's 0.05896 carried over.** `EXP_004` §10.11 item 2
predicted the numbers "would not add twice" and they did not — the arm's
pre-registered C3 (sub-additivity) held. The design cannot separate 60 % from
55 % or 70 %, and `EXP_011` §3.1 said so before it ran.

### 10.3 The gain is entirely within-reach, which confirms §10.6 by relocating it

`EXP_004` §10.3's decomposition, **against the adopted arm**:

| component | bpc | share |
|---|---:|---:|
| Zero context | **+0.00050** | **1.4 %** |
| Within the baseline's reach | **+0.03597** | **102.4 %** |
| Beyond the horizon | −0.00134 | −3.8 % |

**`EXP_004` §10.6 argued the two mechanisms are the same mechanism at zero
context.** If so, stacking a real per-channel threshold on the adopted neuron
should buy nothing there. It buys **0.00050 bpc — 1.4 %, and below the ~2e-3
noise floor §6.5 measured.** The prediction is confirmed about as cleanly as this
project can confirm anything.

And that is *why* the composition is worth having. Its entire gain is the
**within-reach** component — the one the adopted arm made **worse** (§6.2:
−48.4 % inside the baseline's own reach) and the one the threshold arm was the
first arm in this project to move (+0.0288 against the baseline). Composed, it
moves that component by **+0.03597 — more than it managed against the baseline**,
because the adopted arm had left more room there to recover.

Against the Phase-2 baseline the composed arm decomposes as zero **+0.05113**
(32.2 %), within-reach **−0.02386** (−15.0 %), beyond **+0.13139** (82.8 %): it
still gives back short-range ground, but **less than the adopted arm alone**.

**The two arms compose because they act on different components than their
headline numbers advertise.** Phase 4's ranked list is built on the assumption
that arms attack the component they were ranked for; this is the second time in
Phase 4 that assumption has failed (§6.7 was the first), and it is the finding
here that most affects Phase 5.

Supporting rows: **§6.2 against the adopted arm — 0 / 128 contexts significantly
worse**, worst delta −0.0005 at c = 0; against the baseline 3 / 128, where the
adopted arm alone is 5 / 128. **Median horizon 48** against the adopted arm's 47 —
the composition is not a horizon effect and was not expected to be. **The fold
(C4) executes**: residuals 1.05e-06 and 1.89e-07, 738 509 → 737 485 parameters,
loading into `arch="twocomp"` — so **the composed arm costs the adopted arm
exactly at inference.**

### 10.4 `compose_s0` — chased, and decision #7 would not have saved it

The seed trained cleanly to step 17 500 and was NaN by 17 750.
`scripts/exp/011_chase_compose_divergence.py` reproduces it **deterministically
at step 17 598** from the last healthy checkpoint:

| | `EXP_009`'s dead seed | **`compose_s0`** |
|---|---|---|
| forward / loss at the bad step | finite | **finite** (logits 223.0, loss 1.4622) |
| gradients before it | finite, largest **5.46e31** | **1.2e-2 – 3.1e-2, no run-up** |
| ‖g‖ fp32 vs fp64 | **inf vs finite** — the clip overflowed | **both NaN; equal at every healthy step** |
| eager path, same step | not tested | **identical NaN — kernel exonerated** |
| origin | backward, layer 0 + embedding | backward, layer 0 + embedding |

**This is not `EXP_009` §9.4's mechanism, and that matters for decision #7.** That
seed died to an fp32 overflow inside `clip_grad_norm_` on a genuinely enormous
gradient. This one died with gradients of 0.03, every quantity in the range of the
six preceding healthy steps, and parameters at its last healthy checkpoint sitting
**inside** the range the two surviving seeds occupy — `compose_s2` reached a
*larger* maximum gain (7.71 vs 5.40) and lived. Pass 3 replays the step on the
eager path and gets an identical NaN in the identical six tensors, which
**exonerates the hand-written two-compartment backward and rules out an R10
defect.**

Two divergences that look identical in a training log have different causes.
Phase 4 now has **two unexplained ones**, and the fix named in §8 row 7 addresses
only the first.

### 10.5 What this does not settle

**The verdict is provisional at n = 2**, and C2 (cross-seed sd ≤ 0.005) is the
prediction that costs most: 0.00049 says the two surviving seeds agreed and
little else, and it is **not** evidence of stability against whatever killed seed
0. Not settled either: which arithmetic in layer 0's backward makes the NaN
(bounded, not localised); **why this fold is ~1000× tighter than `EXP_008` W1's
1.86e-3** — same identity, same class of fp32 reordering, and this experiment
does not explain it, so it is **not** offered as evidence for decision #6 *(rev 4:
`EXP_012` explains it in §11, and the number is released to #6 there; this
paragraph records what `EXP_011` did and did not settle and is left as it was)*;
and
whether either parent's initialisation is right for the composition, which is one
unswept point in a two-dimensional space.

---

## 11. `EXP_012` — the fold gap is a degeneracy, and the rule that measured it misfired

Pre-registered in `experiments/logs/EXP_012_fold_gap.md` and closed the same day.
It is the second of the two things §10.5 left standing, and the one this report
had **refused to use** as evidence for decision #6 until something explained it.
**No training, ~0.2 GPU-hours**, five existing checkpoints.

### 11.1 Two explanations were removed before anything was measured

Read from the committed source and recorded in that file's §0, so that neither
could later be reported as a discovery: `fold_into_spiking_state_dict` and
`fold_into_twocomp_state_dict` have the **same six-line body**, and `fold_check`
and `fold_check_composed` compute the **same quantity by the same code path**.
Neither the fold's arithmetic nor the metric can be the explanation, which leaves
exactly two candidates — **the neuron** and **the operating point** — and the
experiment is built to separate them.

### 11.2 The controls held: it is not the algebra and not the perturbation

| | `EXP_008` arm (→ `snn`) | composed arm (→ `twocomp`) |
|---|---|---|
| leg C — the algebra, fp64 fold and fp64 GEMM | 6.9 / 6.8 / 8.3 fp64 eps | **3.6 / 7.4 fp64 eps** |
| leg A — the perturbation actually injected | 5.8 / 6.1 / 8.3 fp32 eps | **6.4 / 8.2 fp32 eps** |

Identity 2 is exact on **both** neurons, and the fold perturbs both currents by
the same relative amount to within the spread of `EXP_008`'s own three seeds.
**Whatever separates the two folds happens after the GEMM.**

### 11.3 Where the gap is, and what sets it

| | layer-0 flips | fraction | 512-window bpc residual |
|---|---:|---:|---:|
| `threshold_s0/s1/s2` | 5 339 / 2 684 / 2 851 | 3.18e-04 / 1.60e-04 / 1.70e-04 | 2.03e-03 / 6.23e-04 / 1.62e-03 |
| `compose_s1/s2` | **1 / 2** | **5.96e-08 / 1.19e-07** | **1.44e-05 / 9.29e-06** |

**1 342–5 339× fewer flips.** And the flip count tracks the probability mass the
decision variable puts *at* the threshold — `P(|u| < 1e-6)` is **5.5e-04 – 7.4e-04**
on the baseline neuron against **1.2e-07 – 3.0e-07** on the two-compartment one,
a factor of **1 843–6 218**, while the membrane's overall spread differs by only
~1.19× (sd 5.42 vs 4.55). **It is not a wider distribution. It is a concentration
at the threshold**, and the ladder's shape says so: the composed arm's
`P(|u| < h)` is **linear in `h`** across four decades — a smooth density — while
the baseline's grows only **~9× across the same four decades**.

**Three ratios, three different quantities — and this report quotes all three
rather than the most impressive.** Flip count 1 342–5 339×; 512-window bpc
residual **43–219×**; the committed full-test residuals (`EXP_008` W1 against
`EXP_011` C4) **515× at closest approach, 1 770× headline-to-headline and 9 841×
at the extremes**. The chain from a flipped spike to a bpc is signed and
cancels, and it cancels differently at different sample sizes. **"~1000×" is a
fair headline for the committed numbers and is not the flip-count ratio.**

**`EXP_012` §1's derivation was wrong in both of its terms, in opposite
directions, and the log says so.** It predicted the unreset slow pole would make
the perturbation *larger*; `E|du|` is ~100× **smaller**. It predicted the density
would fall because the spread widens; **the spread does not widen.** The `du`
result is a **consequence of the flip count, not a cause of it** — a flipped
spike moves the LIF's state by an O(1) reset rather than an O(ulp) rounding, so
every later step of that channel carries a macroscopic perturbation. Quoting
`du/dcur` as the mechanism would have inverted the causation.

### 11.4 The decision rule fired on a leg with no power, and is reported unrepaired

The cross-over drives **both** neurons with the **identical** current and
perturbation. On `threshold_s0`'s current — where the LIF's 5 339 flips sit far
above the resolution floor — the two-compartment neuron flips **2**:

| | thr | firing rate | flips |
|---|---:|---:|---:|
| LIF | 1.0000 | 0.344880 | **5 339** |
| LIF, rate-matched to twocomp | 0.3256 | 0.398950 | **205** |
| two-compartment | 1.0000 | 0.398939 | **2** |
| two-compartment, rate-matched to LIF | 1.8887 | 0.344880 | **1** |

**102.5× at matched rate 0.399, 5 339× at matched rate 0.345.** But Y5 required
*every* cross-over leg to reach 10×, and the second leg — on `compose_s1`'s
current — compares **0 flips against 1**. Ratio 0.0. **Y5 FAILED**, and
`EXP_012` §4's truth table therefore returns *"an operating-point effect, not a
neuron effect"*.

**That verdict is not this report's finding, and the reason is not
after-the-fact.** `EXP_012`'s own limitation 2 and failure mode P3 stated in
advance that a zero count would be reported as *"below what one batch resolves"*,
not as a number — and Y5 was then specified without that guard. **The rule is
recorded as it fired and is not rewritten**, because rewriting a bar after seeing
which way it fell is precisely what `EXP_005` §9.4, `EXP_008` §9.4 and `EXP_011`
§0 established the practice of refusing. The 102.5× is a **marker with no verdict
attached**; the corrected wording for Y5 is referred, not adopted.

### 11.5 Three of six predictions failed, and one failure is the mechanism showing

Y1, Y2, Y3 held. Y4 failed (34.0× amplification against a `< 5×` prediction — but
that is 34/1 and 68/2, and **an amplification ratio built on a denominator of 1
is not a measurement of one**, so it is not offered as evidence in the other
direction either). Y5 failed as above.

**Y6 — the flip model — failed, and the shape of its failure is informative.**
Predicted ÷ measured is **1.11 / 1.19 / 1.17** on the baseline's layer 1 and
**0.97 / 0.84** on the composed arm's, but **3.50 / 8.82 / 5.69** on the
baseline's layer 0. The model assumes a *smooth* density; it works wherever §11.3
says the density is smooth and over-predicts by 3.5–8.8× exactly where §11.6 says
there is a point mass. Per `EXP_012` §3, **no quantitative mechanistic claim is
made** from Y3 or Y5 on the back of a failed Y6.

### 11.6 What the concentration is made of — post-hoc, and labelled so

**This subsection is post-hoc**: the hypothesis was formed after reading §11.3's
ladder, and `scripts/exp/012_posthoc_pileup.py` carries the same label.
`EXP_008` §9.5 is the precedent for a chase that follows a result.

Sites with `|u| < 1e-6` at layer 0:

| | count | distinct `u` values | repeats each | channels |
|---|---:|---:|---:|---:|
| `threshold_s0/s1/s2` | 10 962 / 12 436 / 9 213 | **20 / 23 / 23** | **548 / 541 / 401** | 60 / 68 / 69 of 512 |
| `compose_s1/s2` | 5 / 2 | 4 / 2 | 1.2 / 1.0 | 5 / 2 of 512 |

**The baseline's ten thousand near-threshold sites are twenty numbers.** Layer 0's
input is one of only **205 embedding rows**, so `cur` takes 205 values per
channel; the hard reset sets `v` to **exactly 0**, so the step after a spike has
`v_pre = cur`, drawn from that small set. A channel whose `cur` lands within an
ulp of `thr` replays the *same* `u` every time that token follows a spike —
87–91 % of these sites are post-spike steps. The two-compartment neuron never
resets `vs`, so `u = vf + w·vs` carries a continuously-varying history and lands
on a repeated value 1.0–1.2 times.

**Post-spike enrichment does not discriminate** — 2.61× on the baseline, 2.11× on
the composed arm — so an analysis that stopped there would have found a false
positive. **The repeat count is the discriminator**, and it was named as the
falsifier in the script's docstring before it ran.

### 11.7 What this releases, and what it refuses to apply

**Released to decision #6.** §10.5 withheld `EXP_011` C4's 1.05e-06 as evidence
"until understood". It is understood, and it is released. What it establishes is
**not** a tolerance — `EXP_012` proposes none by design — but the **shape of the
choice**: the same identity, through the same fold, costs **2.0e-03 bpc on the
baseline neuron and 1.4e-05 on the adopted one**, for a structural reason. **A
single W1-style constant will be far too loose for one arm or far too tight for
the other.**

**Refused.** The §6.5 scope qualifier — that the ~2e-3 floor belongs to the
baseline LIF rather than to "this architecture" — is supported by §11.3 and
§11.6 and is **not applied**, because `EXP_012` §4 pre-registered it behind
`Y2 ∧ Y5` and Y5 failed. §6.5 stands as written and carries a referral note.
**Adopting a correction whose own gate did not open, seconds after watching it
not open, is the failure mode this protocol exists to prevent** — and it would
have been an easy one to commit, because the correction is almost certainly
right.

**Still unexplained after this revision:** which arithmetic in layer 0's backward
makes `compose_s0`'s NaN (§10.4), and whether any of §11 survives at another
scale or under another reparameterisation — quantisation and weight
normalisation are the two §6.5's floor was written to cover, and `EXP_012`
measured one fold on one architecture pair.

---

## 12. `EXP_013` — the model does not overfit, and the noise arm that tested it is a null

Pre-registered in `experiments/logs/EXP_013_noise_injection.md`, 2026-08-05,
after two calibration measurements that read only the committed Phase-2 baseline
and no arm. Closed the same day. **Not on `03_phase3_candidates.md`'s
18-candidate list** — the file opens a candidate rather than closing one, and
says so in its own §0. Ten training runs (nine plus one re-run, whose two results
are bit-identical — an operational note, not a finding), fourteen gap
evaluations, nine horizon probes, two calibrations: **~1.9 GPU-hours** (its own
§0 pre-registration estimated ~1.6; the measured total is what §1's table above
uses).

### 12.1 What was measured, and why

Inject a zero-mean, unit-variance-scaled Bernoulli background current into every
layer's input, during training only:

```
cur'_{t,c} = cur_{t,c} + amp · (B_{t,c} − p)/sqrt(p(1−p)),   B ~ Bernoulli(p)
```

The `− p` term is derived, not chosen: an uncentred draw adds a constant to every
channel's current, which is `EXP_008`'s Identity 1 in its uniform form — a
threshold shift, not a regulariser — and `EXP_008` already measured a per-channel
version of exactly that mechanism at 0.0590 bpc, an order of magnitude larger
than the whole quantity this experiment targets. Centring removes the confound
at the cost of one kernel. At inference the term is absent; the arm costs the
Phase-2 baseline exactly, with nothing to fold.

Built on the Phase-2 baseline rather than the adopted neuron, for the same
reason `EXP_007`/`EXP_008` were: the noise-off control already exists at n = 5,
and its noise floor is measured on those same five seeds and nothing else. The
composition onto `arch="twocomp"` is the obvious follow-up and is not run here.

### 12.2 The calibration is the finding: this model does not overfit

Measured **before the arm existed** (§2.2), on the five committed baseline
seeds, train-slice bpc against test bpc under both committed protocols:

| protocol | train bpc | test bpc | **gap** | cross-seed sd | bar (2σ_gap) |
|---|---:|---:|---:|---:|---:|
| fresh | 2.26375 | 2.26969 | **+0.00594** | 0.00268 | 0.00536 |
| carried | 2.25491 | 2.25311 | **−0.00180** | 0.00306 | 0.00612 |

Both test legs reproduce the committed Phase-2 baseline figures to five decimals
— `EXP_001`'s F1 discipline applied to a new instrument. **Under `carried` there
is no gap at all** (three of five seeds negative). **Under `fresh` there is a
small, consistent gap — all five seeds positive, 4.96 se from zero — and it is
still smaller than this project's own 2σ = 0.00922 adoption bar**, and only ~3×
the reparameterisation noise floor §6.5 measured.

**Why: 735,437 parameters against ~655M training characters and a 90M-character
split — ~7.3 epochs, ~122 characters per parameter.** This project's problem is
underfitting (2.25311 bpc against the GRU anchor's 1.76741), and a regulariser
is a treatment for the opposite disease. This is the measurement §7 item 6 and
§7.1 rank 2 act on — it does not by itself say anything a noise arm bought, only
that there was very little room for one to work in.

### 12.3 The noise arm itself: costs bits at every amplitude, buys none

Three amplitudes, geometric ×4 (0.1, 0.4, 1.6 — 3.2–113.6 % of `sd(cur)`,
measured per-layer in §2.1), n = 3 seeds each, against the baseline's n = 5:

| arm | n | test bpc, carried | Δ vs baseline | se | cross-seed sd | wall-clock | peak VRAM |
|---|---:|---:|---:|---:|---:|---:|---:|
| Phase-2 baseline | 5 | 2.25311 | — | — | 0.00461 | 1.00× | 0.609 GiB |
| noise, amp 0.1 | 3 | 2.25731 | +0.00419 | +1.2 | 0.00167 | 1.068× | 0.796 GiB |
| noise, amp 0.4 | 3 | 2.30290 | +0.04978 | +14.8 | 0.00325 | 1.089× | 0.796 GiB |
| noise, amp 1.6 | 3 | 2.89287 | +0.63975 | +190.0 | 0.01817 | 1.073× | 0.796 GiB |

Every amplitude costs bits; none buys any. The memory horizon does not move
(6–7 against the baseline's 5/5 seeds at exactly 7), so whatever the noise does,
it is not a horizon effect.

### 12.4 N1 holds and must not be believed — the same defect as `EXP_012`'s Y5

The primary pre-registered rule — `Δgap ≤ −2σ_gap` under `fresh` — fired at two
of three amplitudes. The decomposition says why that must not be read as
regularisation:

| amp | Δ train bpc | Δ test bpc | **Δgap** | resolved? |
|---:|---:|---:|---:|---|
| 0.1 | +0.00468 | +0.00387 | −0.00081 | no — below the ~2e-3 floor |
| 0.4 | +0.05878 | +0.05013 | **−0.00865** | yes |
| 1.6 | +0.67083 | +0.64130 | **−0.02953** | yes |

`Δgap = −(Δtrain − Δtest)` exactly. **The gap closed because the model got
worse at the training text faster than at the test text — a uniformly damaged
model, not a better-generalising one.** At the two amplitudes where N1 fires,
test bpc is 14.8 and 190 se worse than baseline. The bar is satisfiable by
damage because it constrained a difference without constraining what its
subtrahend was allowed to do — precisely `EXP_012`'s Y5 (§11.4), independently,
in a different experiment, six days later. **The bar is reported as it fired and
is not rewritten**, per this project's four standing precedents; the corrected
form — count a reduction only when `Δtest ≤ 0` — is referred, not applied
(§12.6).

**Read against the substance rather than the fired cell, the verdict is the
pre-registration's own second cell: NULL, reported as "there was no gap to
close" — never as "noise injection does not regularise."** N3 held (amp 0.1
inside the null band both ways). Not recommended for adoption; no adoption is
made here.

### 12.5 Predictions, and the rest of the ladder

| | verdict | note |
|---|---|---|
| N1 | **HELD** | wrong reason — §12.4 |
| N2 | **HELD** | monotone in amplitude; +0.640 at the top against a 0.00922 bar |
| N3 | **HELD** | amp 0.1 inside ±0.00922 both ways |
| N4 | **FAILED** | layer 1 monotone 0.323→0.658; layer 0 not (0.337/0.351/0.339/0.361) — no mechanism claimed |
| N5 | **FAILED as written**, held as glossed | the pre-registration stated opposite-direction hypotheses as if complementary; both are the same measurement with opposite signs, so this is one event, not two |
| N6 | **FAILED, both directions** | cross-seed sd 0.36× baseline at amp 0.1, 3.94× at amp 1.6 — σ is a property of the hyperparameter here, the same lesson `EXP_005` established across architectures |

Three of six failed and a fourth is repudiated by its own drafting error — after
`EXP_012`'s three-of-six, the source log calls this "a consistent picture: this
project's pre-registered bars are failing often enough to be doing work."

### 12.6 What this releases, and what it does not

**Released to §7's ranking** (§6.7 item 3, §7 items 5–6, §7.1 rank 2 — this
report's own use of the finding, cross-referenced rather than repeated here).
The transferable result is §12.2: this model is not overfitting, which reframes
regularisation-flavoured candidates downward and capacity-flavoured ones upward,
and specifically un-blocks the parameter-scaling pilot.

**Referred, not applied.** N1's own construction — a gap-reduction bar without a
`Δtest ≤ 0` guard — is referred rather than self-corrected, per `EXP_005` §9.4,
`EXP_008` §9.4, `EXP_011` §0 and `EXP_012` §11.4's precedent. The corrected form
is decision #9's substance (§8), whose protocol-level rule is separately already
ratified (`CONTRIBUTING.md` §2, `51b44c6`).

**Not answered, and not claimed to be.** Whether noise injection regularises a
model that *does* overfit (this one doesn't — nothing here bears on that);
whether `p` matters (held at 0.1, unswept); whether any of this transfers to the
adopted two-compartment neuron (not run there, and `EXP_005`/`EXP_012` are two
separate demonstrations that this project's statistics do not transfer between
those two neurons); and whether the arm belongs on `03_phase3_candidates.md`'s
ranked list at all (§0's own question, Elliot's to rule on).

---

## 13. Changelog

| Rev | Change |
|---|---|
| 7 | 2026-08-08. **Bookkeeping only: the tree is committed, three figures are corrected, and one ranked cost is flagged as unsupported.** No experiment ran, no arm is adopted, no tolerance is set, no phase is authorised, and **no §8 row is edited — five decisions remain open.** (a) **Rev 6's uncommitted working tree is now committed**, in seven commits on `main`: `EXP_013`'s pre-registration and arm, decision #7's fp64 clip (deliberately split from the first so a future bisect can separate a change to *every* arm's training numerics from one experiment's wiring), `EXP_013`'s drivers/resolver/evidence, rev 6 itself, chat session 7, chat Track A, and an orphan sampler. `main` was 74 commits behind at Phase 1 and is fast-forwarded to the tip; a `git bundle` of every ref is written outside the repository. **`EXP_013`'s pre-registration was verified before and after committing**: §0–§8 hash to `285eb661…82ef8d` and the reconstructed pre-results file to `2e9ee41e…`, matching both stamps in `exp_013_run_manifest.json`. That does **not** retroactively supply the pre-run commit its own §0 promised and this entry does not pretend otherwise — the hash chain is what discharges the guarantee, and it held. Full suite re-run on the committed tree: **394 passed, 44.9 s**; fast subset 256 passed / 1 skipped; `ruff` clean. (b) **Rev 6's changelog inventory of what remained uncommitted was incomplete** and is corrected here rather than by rewriting that entry: it lists `EXP_013`'s log, four `exp_013_*.json`, `src/snn/noise.py`, `tests/test_noise.py` and `scripts/exp/013_*.py`, and **omits `src/snn/model.py`, `src/snn/config.py` and `scripts/exp/001_memory_horizon.py`** — two of them in `src/`, so a reader planning the commit from that list would have shipped the arm without the class that implements it. It also undercounts the `exp_013_*.json` artifacts, which are six. (c) **§7.2's budget sentence corrected**: ranks 2–6 sum to **3.05**, not 3.2, and ~**23.5** GPU-hours remain, not ~26.5 — the latter a rev-2 figure carried through four revisions against §1's own ~6.5-of-~30. (d) **§7.2 gains a standing caveat on rank 2's 0.35 GPU-h**: it descends from `01_reconnaissance.md` §3.5's eager forward-only width sweep, which `02_baseline_report.md` §6.2 withdrew in a section that closes by instructing Phase 3's ROI estimates not to reuse it, and `EXP_003`'s ladder is collinear in `K`, `K·d` and `K·d²` so it cannot price width either. A GEMM-FLOP model on `audit_01`'s measured 11.91 TFLOP/s puts it at ~0.9–1.4 GPU-h. **No corrected number is written into §7.1's table**, because that model is itself unvalidated at width; `EXP_014` opens with a calibration that measures it. (e) `README.md`'s status table corrected — it still read 1 adopted arm / 6 of 8 open / ~4.6 GPU-hours, three revisions stale, because rev 6 (g) corrected only the test-suite figures in that file. |
| 6 | 2026-08-08. **`EXP_013` folds in, #5 and #7 are decided, and #9 is added.** (a) §12 added: `EXP_013` (noise injection) closes as a null — costs bits at every amplitude (+0.00419/+0.04978/+0.63975 bpc, 1.2/14.8/190.0 se), buys none — but its §2.2 calibration is the transferable result: this 735,437-parameter model's generalisation gap is +0.00594 bpc (fresh) / −0.00180 (carried, indistinguishable from zero) against ~655M training characters, ~7.3 epochs. **The model is not overfitting.** N1 (the primary rule) holds at two amplitudes and must not be believed — the decomposition shows a uniformly damaged model, not better generalisation, the same defect as `EXP_012`'s Y5 (§12.4). (b) §6.3, §6.5, §6.7 item 3, §10.1 and §7 item 5 each gain a rev-6 cross-reference to §12, none rewritten. (c) §7.1 gains a new rank 2, **the parameter-scaling pilot** (`03_phase3_candidates.md` §6.3 #11, 0.35 GPU-h) — absent from this table through rev 5 for lack of evidence there was room to grow; `EXP_013` supplies exactly that evidence. §7 gains item 6 stating the referral. Ranks 2–5 renumbered 3–6; no prior rank's own evidence changed. (d) §8: **row 5 edited to RESOLVED** — the learned per-channel threshold is **adopted standalone**, per Elliot's instruction; the composed arm (`EXP_011`) is explicitly **not** adopted by this and stays provisional at n = 2. **Row 7 edited to RESOLVED** — `src/snn/train.py`'s `_clip_grad_norm_fp64` replaces `clip_grad_norm_`, accumulating the sum of squares in fp64; verified on `EXP_009` §9.4's own failing gradient (`tests/test_grad_clip.py`) and against the unchanged R5 capture gate (10/10). Does **not** address `compose_s0`'s divergence — a different, still-open mechanism (§10.4). **Row 9 added, open**: whether a pre-registered bar needs a guard-clause review before commit, per `EXP_012`'s Y5 and `EXP_013`'s N1 firing the identical defect twice — the *protocol* rule is already ratified in `CONTRIBUTING.md` §2 (`51b44c6`, 2026-08-05, Elliot); this row is the report catching up to it, not a new decision. **Table now nine rows, four resolved, five open.** (e) `CONTRIBUTING.md` §4 gains two ratified amendments: the fold-in tolerance is architecture-specific and re-derived per arm, never inherited (decision #6's substance); a provisional result is promoted only by seeds *added*, never substituted, up to the pre-registered n. (f) §9: full suite re-run for real, **394 passed** (280 + 85 `test_snnchat.py`, uncommitted-to-this-revision + 23 `test_noise.py` + 6 `test_grad_clip.py`); fast subset 256 passed/1 skipped/8s. Gate rows added for `EXP_013`'s G1/G3/G4-5/F1/K1/calibration/hash checks and for decision #7's fix. Mutation campaign **not** re-run — nothing in `snn/kernels.py`/`surrogate.py`/`twocomp.py` changed. (g) `README.md` and `CONTRIBUTING.md`'s quoted test-suite figures corrected to the numbers this revision measured, both stale independently of each other (README's fast count and CONTRIBUTING's full count had drifted by different amounts). (h) the changelog renumbered §12 → §13 as `EXP_013`'s new §12 was inserted, the same slip rev 3's item (f) and rev 4's item (g) each record for their own insertions; two bare "§12" cross-references (§1's rev-2 summary, §6.1's wall-clock correction note) meant the changelog and were retargeted to §13, caught in this same pass rather than left for a rev 7 to find. **GPU-hours: ~4.6 → ~6.5** (using `EXP_013` §9's measured 1.9, not its §0 estimate of 1.6). **Working tree, recorded rather than resolved:** `EXP_013`'s own files (log, four `docs/reports/data/exp_013_*.json` beyond the two already read at rev 4, `src/snn/noise.py`, `tests/test_noise.py`, `scripts/exp/013_*.py`) remain uncommitted, as they were at rev 5 and at `31e49fa`'s deliberate exclusion; this revision's own edits (this file, `CONTRIBUTING.md`, `README.md`, `src/snn/train.py`, `tests/test_grad_clip.py`) are likewise left uncommitted — staging is Elliot's, not this revision's. |
| 5 | 2026-08-05. **Decision #1 is decided, and it is the only thing in this revision.** Elliot signed off Phase-3 §9. (a) `03_phase3_candidates.md` §9's last box is ticked and **labelled retrospective in place** — Phase 4 opened without it and eight experiments ran before it, so the box's "Phase 4 does not begin until" was not honoured in sequence and the tick says so rather than letting the record read as though it had been; that file gains a rev 3 carrying the provenance. (b) §8 **row 1 edited to RESOLVED** — the first of rows 1–7 ever edited, following #8's rev-3 precedent that a *decided* row is edited while an *open* row is only annotated in prose. (c) A rev-5 note beneath §8 states the one consequence and two non-consequences: it **releases `EXP_008` §4's rider so #5 is ungated by #1**, it does **not** decide #5, and it does **not** launder **#2** — a defect `EXP_005` §9.4 found under a §9 box that is now ticked, named in both files rather than absorbed. (d) §8's rev-3 preamble is left as written with a rev-5 parenthetical, per the §10.5 precedent. (e) §9 gains a closing paragraph stating that the gate table was **not** re-run and is not claimed to have been, because rev 5 touches two report files and no code, artifact or run directory. **Six decisions remain open. No experiment ran, no number moved, no arm is adopted, no tolerance is set, and no phase is authorised.** |
| 4 | 2026-08-05. **`EXP_012`: the fold gap explained, and a rule that misfired reported rather than repaired.** (a) §11 added: the ~1000× gap between `EXP_008` W1 and `EXP_011` C4 is **not** the algebra (leg C exact on both neurons) and **not** the injected perturbation (leg A 6.4–8.2 fp32 eps against 5.8–8.3), but the **layer-0 flip count** — 1/2 flips against 2 684–5 339, a factor of **1 342–5 339** — which tracks the probability mass the decision variable puts at the threshold: **1 843–6 218× more** within 1e-6 of it, at a membrane spread that differs by only 1.19×. (b) §11.6, **post-hoc and labelled so**: the baseline's ~11 000 near-threshold sites are **20–23 distinct values replayed 400–548 times each**, because layer 0 sees one of 205 embedding rows and the hard reset zeroes `v` exactly; the unreset slow pole destroys the replay (4 values, 1.2 repeats). Post-spike enrichment does **not** discriminate (2.61× vs 2.11×) and is reported as a near-miss for the analysis. (c) **Three of six predictions failed.** Y5's rule resolved on a leg comparing **0 flips against 1**, which `EXP_012`'s own limitation 2 had declared unresolvable before it ran; the rule is **recorded as it fired and not rewritten**, and the powered leg's 102.5× is a marker with no verdict. Y4's 34× is 34/1 and 68/2 and is not offered as evidence either way. Y6 failed, and its failure pattern — accurate wherever the density is smooth, 3.5–8.8× over wherever there is an atom — is consistent with (b). (d) §6.5 **stands exactly as written**; the scope qualifier its own evidence supports is **referred, not applied**, because the trigger `EXP_012` §4 pre-registered for it (`Y2 ∧ Y5`) did not fire. (e) §8: **no row edited**; #6's evidence is sharpened in prose beneath the table — §10.5's withheld number is **released** to it, and the finding is that a fold-in tolerance **cannot be one project constant**. #4 re-evidenced at ~4.6 of ~30 GPU-hours. (f) §9 adds three gate rows; test suite **272 → 280**. (g) the changelog renumbered §11 → §12 and the header's rev-2 reference retargeted — the same slip rev 3's item (f) records. **No earlier number, verdict or threshold changes; nothing is adopted and no phase is authorised.** |
| 1 | 2026-08-03. First Phase-4 report: the adopted arm and the shape of its gain, σ's non-transfer, `EXP_006`'s causal tail result and what it withdraws, `EXP_007`/`EXP_008`/`EXP_009` resolved against their pre-registrations, the reparameterisation noise floor, a revised ranking offered and not applied, and seven open decisions. |
| 3 | 2026-08-04. **`EXP_011`, and the resolution of the one decision that was a scheduling question.** (a) §10 added: the composed arm (adopted two-compartment neuron + `EXP_008`'s learned per-channel threshold) at **2.08326 ± 0.00049 carried, −0.03543 against the adopted arm at −28.7 se, 0/128 contexts regressed, folding to the adopted arm's exact inference cost** — and **provisional at n = 2 of a pre-registered 3**. (b) §10.3: the composition's gain is **102.4 % within-reach and 1.4 % zero-context**, which confirms `EXP_004` §10.6's overlap argument in its strong form and relocates the value to the component the adopted arm damaged — the second time in Phase 4 that an arm has not moved the component it was ranked for. (c) §10.4: `compose_s0` diverged, was chased to step 17 598 and is **not** `EXP_009`'s mechanism — gradients of 0.03, fp32 and fp64 norms equal, an identical NaN on the eager path — so **decision #7's fix would not have prevented it** and the fused backward is exonerated. The seed was excluded and **not replaced**. (d) §8 row 8 marked **RESOLVED (inside Phase 4)**; **rows 1–7 are not edited**, and the two whose evidence `EXP_011` overtakes (#4, #7) are corrected in prose beneath the table rather than in the rows. (e) §9 re-run with four new gate rows; test suite **262 → 272**. (f) the header's §10 reference retargeted to §11 as the changelog renumbered. **No earlier number, verdict or threshold changes.** |
| 2 | 2026-08-04. **Synthesis and corrections. No verdict, threshold or decision changes.** (a) §6.7 added: the three arms read together — two of them are the same mechanism, only the arm that added the least structure moved the within-reach component, token-shift is dominated at 13.3 se, and essentially none of the three gains is attributable to a larger function class. (b) §6.8 added: the verdict on every Phase-4 arm against its own rule, including that `EXP_008`'s cell is `W1 fails` and its headline is quoted under that caveat. (c) §7.2 added: the order Phase 5's arms should run in and what must be settled first. (d) §8 item 8 added; #3, #4, #5 re-evidenced; **all eight remain open**. (e) §9 re-run from a clean tree, with the resolver's 0-difference reproduction as a new row. **Corrections:** (f) §6.1's wall-clock for the adopted arm was **1.35×**, which is `twocomp_s0` alone — `EXP_004`'s committed three-seed figure is **1.32×** (1.30× over all seven), and `EXP_004` §10 item 6 said to quote it; *this correction flatters the adopted arm and is made because it restores a committed measurement*. (g) §6.1's distilled row was **1.58×**, correct against the two-compartment arm in `EXP_009` §9.1 and wrong in a column referenced to the baseline, where it is **2.13×**. (h) "clear the adoption bar by an order of magnitude" → **4.2×** and **6.4×** (8.4σ and 12.8σ). (i) the adopted arm recovered **58.2 %** of the beyond-horizon component, not 59 %, in §2 and §7.1. (j) Identity 2 holds to **6.8–8.3** fp64 eps; `EXP_008` §9.5's "≈7" is its first two seeds. Closed experiment logs are **not** rewritten for (f)–(j); the corrected numbers live here. (k) adding §7.2 made three bare "§7.2"/"§7.3" references read as self-references when they meant `03_phase3_candidates.md`; all three are now qualified — the same slip the Phase-3 report's own changelog records as (d). |
