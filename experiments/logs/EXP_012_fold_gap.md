# EXP_012 — Why is `EXP_011`'s fold ~1000× tighter than `EXP_008`'s?

**Pre-registered:** 2026-08-05, before any measurement on the composed arm's fold
beyond the two residuals `EXP_011` C4 already committed.
**Status:** OPEN.
**Phase:** 4 (controlled experiments). This is the second of the two things
`EXP_011` §10 named as unsettled, and the one the Phase-4 report §10.5 says is
**not** offered as evidence for decision #6 *until understood*.

**What it costs:** no training. Four forward-pass legs over five existing
checkpoints, ~0.2 GPU-hours.

**What it is for:** decision #6 (the fold-in tolerance) is currently
**undecidable as posed**, because the only two fold-in measurements this project
has disagree by three orders of magnitude and nothing explains why. This
experiment does not decide #6. It establishes what the residual is a property
*of*, so that a tolerance can be set against a mechanism instead of against a
precedent.

---

## 0. What is already established, and is therefore not a finding of this experiment

Two things were checked by reading the committed source **before** this file was
written, and both remove candidate explanations a priori. They are recorded here
so that neither can be reported later as though this experiment discovered it.

1. **The fold arithmetic is identical.** `LearnedThresholdCharLM.fold_into_spiking_state_dict`
   and `TwoCompThresholdCharLM.fold_into_twocomp_state_dict` (`src/snn/model.py`)
   have the same six-line body: clone the state dict, and for each layer multiply
   `layers.k.weight` by `g.unsqueeze(1)`, multiply `layers.k.bias` by `g`, and
   delete `thr_log.k`, where `g = input_gain(k) = exp(-theta)` in both classes.
   Same operations, same order, same dtype. **The fold is not the difference.**
2. **The metric is identical.** `fold_check` in `scripts/exp/010_phase4_arm_results.py`
   and `fold_check_composed` in `scripts/exp/011_compose_results.py` both compute
   `abs(folded_fresh_bpc - committed_arm_fresh_bpc)`, both build the folded model
   through `build_model` and score it with `snn.evaluate.evaluate(..., "fresh")`.
   **The residual is the same quantity on both arms**, so the gap is not a units
   or protocol artifact.

What differs between the two folds is therefore exactly two things: **the neuron
the perturbed current is fed into**, and **the operating point the arm trained
to**. Separating those is the whole of this experiment.

### 0.1 The gap, stated precisely

| | `EXP_008` W1 (`arch="snn"`) | `EXP_011` C4 (`arch="twocomp"`) |
|---|---|---|
| residual, bpc | **1.86e-3** / 5.42e-4 / 9.47e-4 | **1.053e-06** / 1.890e-07 |
| n | 3 | 2 (`compose_s0` diverged, excluded, not replaced) |
| verdict | **FAILED** its 1e-3 bar on 1 of 3 | no bar applied — reported |

Closest approach is 5.42e-4 against 1.053e-06 = **515×**. Headline against
headline is **1770×**. Extremes are 9840×. "~1000×" is the fair summary and is
what this file means by it.

---

## 1. The derivation, before any code

`EXP_008` §9.5 established the chain on the baseline neuron: the fold perturbs
each current by ~1 ulp; any membrane sitting within an ulp of the threshold flips
its spike; flipped spikes are the *input* to the next layer's GEMM, so the
disagreement is amplified per layer until it reaches the logits and the bpc. The
question here is which link in that chain behaves differently under the
two-compartment neuron.

**The two decision variables.** From `lif_scan_eager` and `twocomp_scan_eager`:

| | baseline LIF | two-compartment |
|---|---|---|
| state update | `v_pre = v·beta + cur` | `vf = vf·beta_f + cur`; `vs = vs·beta_s + cur` |
| decision variable `u` | `v_pre - thr` | `vf + w·vs - thr` |
| reset | `v = v_pre·(1-s)` — **the whole state** | `vf = vf·(1-s)` — **`vs` is never reset** |

**The flip model.** A spike flips wherever the perturbation carries `u` across
zero. To first order, over `N` spike sites,

>  `flips  ≈  N · rho_u(0) · E|du|`

where `rho_u(0)` is the density of the decision variable at the threshold and
`du` is the perturbation reaching it. The two neurons differ in **both** factors,
and **they differ in opposite directions**:

* **`du` should be LARGER in the two-compartment neuron.** `vs` integrates the
  perturbed current with `beta_s` near 1 and is never reset, so `du` accumulates
  as `sum_k beta_s^(t-k) · dcur_k` over the whole window. The LIF's `v` is zeroed
  at every spike, so its `du` has a memory of only a few timesteps at a ~0.35
  firing rate.
* **`rho_u(0)` should be SMALLER in the two-compartment neuron**, for the same
  reason: an unreset slow pole with a long time constant gives `u` a much wider
  spread, and a wider spread at fixed `thr` means a lower density at the
  threshold.

**These fight, and the derivation does not say which wins.** That is what makes
this an experiment rather than a write-up. What the committed residuals already
tell us is only the *net* outcome at the far end of the chain; they say nothing
about which factor produced it, nor whether the cause is the neuron at all rather
than the operating point the two arms happened to train to.

### 1.1 What is genuinely open, and what is not

**Not open:** that the composed arm's fold residual is smaller. That is committed
in `exp_011_arm_results.json` and this experiment cannot and does not re-predict
it. Predictions below that merely restate it would be scoring a hit for knowing
the answer, which `EXP_007`'s V4 is this project's precedent for refusing.

**Open, and each can fail:** *where* in the chain the gap appears (leg A? the
flip count? the per-layer amplification?); whether the cause survives holding the
operating point fixed; and whether the flip model above predicts the flip counts
it did not produce.

---

## 2. Design — four legs, in the order that separates the explanations

**Runs.** `threshold_s0`, `threshold_s1`, `threshold_s2` (the `EXP_008` arm, all
three, including the seed that failed W1) and `compose_s1`, `compose_s2` (the
`EXP_011` arm; `compose_s0` diverged, its `final_test.json` is NaN at firing rate
0.0, and it is excluded for `EXP_011` §9's reason and **not replaced**).

**Leg 1 — the identity and the injected perturbation, on both arms.**
`EXP_008` §9.5's three precisions, generalised to dispatch on `arch`:

| | what it measures |
|---|---|
| **A** as shipped: fp32 fold, fp32 GEMM | storage **and** GEMM reordering — the perturbation actually injected |
| **B** fp32 fold, fp64 GEMM | the fold's own fp32 storage alone |
| **C** exact fp64 fold, fp64 GEMM | the algebra |

Layer 0 only, for `EXP_008`'s stated reason: layer 1's input is layer 0's spike
train, which the two models do not agree on, so a layer-1 comparison would mix
the identity with the disagreement it causes.

**Leg 2 — the cascade, on both arms.** Spike disagreement per layer, the
per-layer amplification ratio, the logits difference, and a subset bpc. This is
the like-for-like table `EXP_008` has and `EXP_011` does not.

**Leg 3 — the two factors of the flip model, measured separately.** For each arm
and each layer: the distribution of `u`, the density `rho_u(0)` estimated over a
ladder of half-widths, the realised `E|du|`, and the **predicted** flip count
`N · rho_u(0) · E|du|` against the flip count leg 2 measured independently.

**Leg 4 — the cross-over, which is the decisive leg.** Take one arm's layer-0
current `cur` and its own folded counterpart `cur'` — the exact pair leg A
compares — and drive **both** neurons with **both**:

* `twocomp_scan(cur)` vs `twocomp_scan(cur')` → flips under the adopted neuron
* `lif_scan(cur)` vs `lif_scan(cur')` → flips under the baseline neuron

Identical input, identical perturbation, identical fp32 path: the only difference
is the neuron. Firing rate is the obvious confound — a neuron that fires less has
fewer sites at risk — so the LIF leg is run twice, once at the committed
`threshold`, and once at the threshold that **matches the two-compartment arm's
firing rate** on the unperturbed current. Both are reported.

---

## 3. Hypotheses, fixed now

| | statement | bar | if it fails |
|---|---|---|---|
| **Y1** | **The identity is exact on the composed arm too.** Leg C, in units of `\|cur\|max`, on `compose_s1`/`s2` | **≤ 1e-14** (`EXP_008` measured 1.507e-15 – 1.839e-15, ≈ 7 fp64 eps) | the two folds are not the same algebra and §0 item 1 is wrong. Everything below is void |
| **Y2** | **The injected perturbation is the same size on both arms.** Leg A, in units of `\|cur\|max` | within **3×** of `EXP_008`'s 6.902e-07 – 9.872e-07, i.e. in **[2.3e-07, 3.0e-06]** | **load-bearing**: the gap is upstream of the neuron, Y3–Y6 are moot, and the answer is about the fold's inputs, not about spiking |
| **Y3** | **The two-compartment neuron flips far fewer spikes.** Layer-0 disagreement fraction on the composed arm | **< 1.6e-05**, i.e. ≥ 10× below `EXP_008`'s 1.600e-04 – 3.182e-04 | the gap is not in the flip count and must be in the amplification or the readout — which would be a more interesting answer, not a null |
| **Y4** | **It also amplifies less between layers.** Layer-1 ÷ layer-0 flip fraction | **< 5×**, against `EXP_008`'s 14.6 / 15.9 / 17.9 | the per-layer amplification is a property of the depth and the readout, not of the neuron |
| **Y5** | **The cause is the neuron, not the operating point.** Leg 4, at **matched firing rate**, identical current and perturbation | two-compartment flips **≥ 10×** fewer than LIF | the cause is where the arms trained to — `theta`, `\|W\|`, firing rate — and Y3's result is a fact about these five checkpoints rather than about the neuron. This is a real possible outcome and is not a failure of the experiment |
| **Y6** | **The flip model predicts the flip counts.** `N · rho_u(0) · E\|du\|` against leg 2's measured flips, on **both** arms | within **3×**, both arms, both layers | §1's mechanism is wrong or incomplete. Y3/Y5 may still stand as measurements, but no mechanistic claim is made from them |

**Y3 and Y4 are not restatements of the committed result.** The committed result
is a bpc residual at the end of a four-stage chain; these are the flip counts at
stages one and two, which have never been measured on the composed arm. It is
entirely possible for the bpc gap to be 1000× while the layer-0 flip gap is 2×,
with the rest arriving later in the chain — in which case Y3 fails and the answer
is somewhere else.

**Y5 is the one that matters** and it is the one most likely to fail. The
composed arm reaches `exp(theta)` 0.449/0.529 and 0.456/0.523; `EXP_008`'s arm
reaches 0.342/0.428. Those are different operating points, and a difference in
flip counts is exactly what different operating points would also produce.

### 3.1 The power of this design, stated before it runs

The composed side is **n = 2** and every number from it inherits that. Leg 4 does
not: it is a controlled comparison inside a single checkpoint, so its verdict
does not rest on the seed count. **If Y3 holds and Y5 fails, the honest report is
"the gap is real and is not explained by the neuron"** — and that is a result
this file is willing to write.

---

## 4. Decision rule, fixed now

Resolved on Y2 first, then Y5, because those two partition the outcome space.

| Y2 (perturbation matched) | Y5 (cross-over) | verdict |
|---|---|---|
| **holds** | **holds** | **The two-compartment neuron is intrinsically less sensitive to a reparameterisation of its own weights.** The ~2e-3 noise floor is a property of the *baseline LIF neuron* and does not transfer to the adopted arm. Decision #6 gains the thing it lacks: a mechanism |
| **holds** | **fails** | **The gap is real and is an operating-point effect, not a neuron effect.** Reported as such. Decision #6 gains a warning instead of a mechanism: a fold-in tolerance cannot be a project constant, because it moves with where the arm trained |
| **fails** | — | **The gap is upstream of the neuron.** Leg A differs, so the fold's inputs differ in conditioning; the spiking story is not the explanation and §1's derivation is set aside |

**This experiment sets no tolerance and proposes no number for W1.** Decision #6
is Elliot's and stays open whatever the outcome. `EXP_005` §9.4 and `EXP_008`
§9.4 are the precedent: report the numbers, refer the change, apply nothing.

**One consequence is pre-registered here so it cannot be claimed post-hoc.** If
Y2 and Y5 both hold, then the Phase-4 report §6.5's sentence — *"this
architecture's reported bpc is reproducible to only ~2e-3 across
mathematically-equivalent reparameterisations of its own weights"* — is **too
broad as written**, because it was measured on one neuron and the adopted arm is
a different one. The correction is a **scope qualifier on a Phase-4 claim**, it
is written into the report and not backdated into `EXP_008`, and it is this
project's `EXP_005` σ-non-transfer lesson in a new place. If Y5 fails, §6.5
stands as written and this paragraph is reported as not triggered.

---

## 5. Known limitations, stated in advance

1. **n = 2 on the composed side**, and no replacement seed. Every cross-arm
   number inherits it. Leg 4 does not.
2. **One batch, one corpus, one config.** Leg 2's flip fractions are over
   ~1.7e7 spike sites, which is ample for a fraction of 1e-4 but resolves a
   fraction of 1e-8 poorly. If the composed arm's flip count is 0, the report
   says "below what one batch resolves", not "zero".
3. **Leg 4 is layer-0 only**, for leg 1's reason.
4. **`rho_u(0)` is a histogram estimate, not a density.** It is bin-width
   dependent, so leg 3 reports a ladder of half-widths and the report quotes the
   ladder, not one bin.
5. **The flip model is first-order.** It assumes `du` is small relative to the
   scale over which `rho_u` varies. Y6 is what checks that assumption rather than
   asserting it.
6. **`compose_s0` is excluded and not replaced**, so this experiment inherits
   `EXP_011`'s provisional status wherever it uses the composed arm.

---

## 6. Self-checks — aborting, not advisory

* **S1 — the probe reproduces the committed kernel.** Leg 3 recomputes the
  membrane recursion in the probe in order to observe `u`, which no scan returns.
  The recomputed `u` must produce spikes **bit-identical** to the committed
  `lif_scan` / `twocomp_scan` on the same input, on every run, both layers. A
  mismatch aborts before any density is reported. This is `EXP_004` §9.2's
  lesson: the contract is asserted on the intermediate itself, at the kernel
  boundary, and a probe that agrees only with itself measures nothing.
* **S2 — the identity control.** Y1's leg C on both arms, run before legs 2–4.
* **S3 — the environment has not drifted.** `scripts/exp/008_chase_fold.py`
  re-run to a scratch path must reproduce `docs/reports/data/exp_008_fold_residual.json`.
  Anything else invalidates the cross-arm comparison before it starts, and the
  comparison is the experiment.
* **S4 — the mechanism check can fail.** Y6 compares a predicted flip count to an
  independently measured one. It is in the self-check list and not only in the
  prediction table because a mechanism that cannot reproduce a number it did not
  produce is prose.

---

## 7. Failure modes this design is trying to avoid

| | failure mode | what stops it |
|---|---|---|
| **P1** | A probe that reimplements the neuron and therefore agrees with itself | **S1**, bit-exact against the committed scan |
| **P2** | Comparing two arms at two operating points and calling the difference a neuron effect | **leg 4**, identical current and perturbation through both neurons, at matched firing rate |
| **P3** | Reading a small flip count as zero when one batch cannot resolve it | limitation 2, and the report states the resolution floor beside the count |
| **P4** | A density that is an artifact of one bin width | limitation 4, a ladder of half-widths |
| **P5** | Explaining the gap with a mechanism that predicts the wrong magnitude | **Y6**, within 3×, on both arms |
| **P6** | Quietly becoming a decision about the fold-in tolerance | §4's closing paragraph; the script reports and gates nothing |

---

## 8. Entry conditions

1. Working tree clean, and **this file committed before the chase script exists**.
2. The chase script committed **before any of its numbers are read**.
3. S3 reproduces `exp_008_fold_residual.json`.
4. No mutation campaign running; lockfile checked (protocol: a tool that writes
   wrong code into the tree owns the machine).

---

## 9. Results

**Closed 2026-08-05. Y1, Y2 and Y3 held; Y4, Y5 and Y6 failed.** S1 held on every
run and both layers, S3 reproduced `exp_008_fold_residual.json` at **0
differences**. Artifacts: `docs/reports/data/exp_012_fold_gap.json` and
`docs/reports/data/exp_012_pileup.json`.

**The one-line answer: the gap is in the flip count, the flip count is set by how
much probability mass the decision variable puts *at* the threshold, and the
LIF's mass there is 20 distinct values replayed ~548 times each — a degeneracy
the two-compartment neuron's unreset slow pole destroys.**

### 9.0 The mechanical resolution, as `resolve()` printed it

| | verdict | measured | bar |
|---|---|---|---|
| **Y1** identity exact on the composed arm | **HELD** | 7.937e-16, 1.639e-15 (3.6 / 7.4 fp64 eps) | ≤ 1e-14 |
| **Y2** injected perturbation matches | **HELD** | 7.575e-07, 9.833e-07 (6.4 / 8.2 fp32 eps) | [2.3e-07, 3.0e-06] |
| **Y3** far fewer layer-0 flips | **HELD** | 5.960e-08, 1.192e-07 | < 1.6e-05 |
| **Y4** less per-layer amplification | **FAILED** | 34.0, 34.0 | < 5 |
| **Y5** the neuron, not the operating point | **FAILED** | 102.5 and **0.0** | ≥ 10, all legs |
| **Y6** the flip model predicts the counts | **FAILED** | 0.84 – 8.82 | within 3× |

§4's truth table, resolved on Y2 then Y5, returns: *"the gap is real and is an
**operating-point** effect, not a neuron effect."* **§9.4 is why that verdict must
not be read as this experiment's finding**, and §9.7 says what is reported
instead. The rule is recorded as it fired; it is not rewritten.

### 9.1 The two controls held, so the gap is neither the algebra nor the input

Leg C lands at 3.6 and 7.4 fp64 eps on the composed arm against `EXP_008`'s
6.8–8.3. **Identity 2 is exact on both neurons** and §0 item 1's reading of the
source is confirmed numerically.

Leg A — the perturbation the fold actually injects — is **6.4 and 8.2 fp32 eps**
against `EXP_008`'s **5.8, 6.1, 8.3**. The two arms are perturbed by the same
relative amount, by the same mechanism, to within the spread of `EXP_008`'s own
three seeds. **Y2 is the load-bearing control and it removes the upstream
explanation entirely**: whatever separates the two folds happens after the GEMM.

### 9.2 Y3: the gap is in the flip count, and it is enormous

| | layer-0 flips | fraction | layer-1 flips | logits max \|diff\| | 512-window bpc residual |
|---|---:|---:|---:|---:|---:|
| `threshold_s0` | 5 339 | 3.182e-04 | 77 953 | 27.13 / 263.6 | 2.03e-03 |
| `threshold_s1` | 2 684 | 1.600e-04 | 42 653 | 20.58 / 272.4 | 6.23e-04 |
| `threshold_s2` | 2 851 | 1.699e-04 | 51 081 | 23.52 / 288.3 | 1.62e-03 |
| **`compose_s1`** | **1** | **5.960e-08** | 34 | 6.23 / 232.6 | **1.44e-05** |
| **`compose_s2`** | **2** | **1.192e-07** | 68 | 8.68 / 232.7 | **9.29e-06** |

**1 342× to 5 339× fewer layer-0 flips.** Y3 asked for 10× and got three orders
of magnitude more than that.

**Three different ratios, and they are not the same number — say so.** The flip
count differs by 1 342–5 339×; the 512-window bpc residual by **43–219×**; the
committed full-test residual (`EXP_008` W1 against `EXP_011` C4) by 515–1 770×.
The chain from a flipped spike to a bpc is signed and cancels, and it cancels
differently at different sample sizes. **"~1000×" is a fair headline for the
committed numbers and is not the flip-count ratio**, and no row above should be
quoted as though it were another.

### 9.3 Why: the density at the threshold, not the spread of the membrane

`u`'s overall spread is *comparable* on the two neurons — sd **5.27 / 5.42 /
5.30** against **4.59 / 4.55**. So §1's "the slow pole widens the distribution"
is not what happens. What differs is the mass *at* the threshold:

`P(|u| < h)`, layer 0:

| h | `threshold_s0/s1/s2` | `compose_s1/s2` |
|---|---|---|
| 1e-6 | 6.53e-04 / 7.41e-04 / 5.49e-04 | **2.98e-07 / 1.19e-07** |
| 1e-5 | 1.34e-03 / 1.43e-03 / 1.09e-03 | 3.34e-06 / 4.17e-06 |
| 1e-4 | 2.06e-03 / 2.76e-03 / 1.92e-03 | 3.45e-05 / 3.34e-05 |
| 1e-3 | 3.17e-03 / 3.79e-03 / 2.99e-03 | 2.99e-04 / 3.05e-04 |
| 1e-2 | 6.00e-03 / 6.42e-03 / 5.48e-03 | 2.85e-03 / 2.86e-03 |

**Read the shape, not only the level.** The composed arm's row is **linear in
`h`** across four decades — a smooth density, `rho_u(0) ≈ 0.17` per unit `u`. The
threshold arm's grows only **~9× across the same four decades** while sitting
**1 843–6 218× higher at h = 1e-6**. At h = 1e-2 the two are within 2×. **The LIF
does not have a higher density; it has an atom.** That is the whole finding, and
§9.6 is what it is made of.

**§1's derivation was wrong in both of its terms, and in opposite directions.**
It predicted `du` would be *larger* under the two-compartment neuron because the
slow pole integrates without reset; measured, `E|du|` is **1.48e-06 / 1.52e-06
against 1.05e-04 – 1.99e-04**, i.e. ~100× *smaller*. And it predicted the density
would fall because the spread widens; the spread does not widen. Both halves are
recorded as falsified.

The `du` result is a **consequence, not a cause**, and the direction of causation
matters: mean |Δcur| is nearly identical across all five runs (**1.30e-06 –
1.38e-06** vs **9.12e-07 – 9.20e-07**), but the membrane multiplies it by
**76–148×** in the LIF and **1.6×** in the two-compartment neuron. That factor is
the hard reset feeding back: once a spike flips, `v ← v_pre·(1−s)` moves the
state by an O(1) amount rather than an O(ulp) one, so every subsequent step of
that channel carries a macroscopic `du`. **Few flips ⇒ little feedback ⇒ small
`E|du|`.** Reporting `du/dcur` as a cause would have inverted the mechanism.

### 9.4 Y5 failed, on a leg this design had already declared unresolvable

Leg 4, layer 0, identical current and identical perturbation through both
neurons. Two-compartment parameters for the `threshold_s0` direction come from
`twocomp_s0`, which has none of its own (`--twocomp-params-from`).

**Direction 1 — `threshold_s0`'s current (the leg with power):**

| neuron | thr | firing rate | flips | fraction |
|---|---:|---:|---:|---:|
| two-compartment | 1.0000 | 0.398939 | **2** | 1.192e-07 |
| LIF | 1.0000 | 0.344880 | **5 339** | 3.182e-04 |
| LIF, rate-matched to twocomp | 0.3256 | 0.398950 | **205** | 1.222e-05 |
| twocomp, rate-matched to LIF | 1.8887 | 0.344880 | **1** | 5.960e-08 |

Matched at rate 0.399: **205 vs 2 = 102.5×.** Matched at rate 0.345: **5 339 vs
1 = 5 339×.** At the committed thresholds: 2 669×.

**Direction 2 — `compose_s1`'s current:** two-compartment 1 flip; LIF **0**;
LIF rate-matched **0**. Ratio **0.0**.

**Y5's rule required every leg to reach 10×, so Y5 FAILED.** The failure is
carried by direction 2, where the counts are **0 and 1 out of 16 777 216**, at a
resolution floor of 5.96e-08. That comparison cannot distinguish a ratio of 0.1
from a ratio of 100.

**The pre-registration said this in advance and the rule ignored it.**
Limitation 2: *"If the composed arm's flip count is 0, the report says 'below
what one batch resolves', not 'zero'."* P3 names the same failure mode. Y5 was
specified without the guard its own limitations section had already written down
— it should have required the reference leg's count to clear the floor by some
margin before a ratio was formed.

**That is a defect in the rule, and it is being reported, not repaired.**
Rewriting a bar after seeing which way it fell is the thing this protocol exists
to prevent, and `EXP_005` §9.4, `EXP_008` §9.4 and `EXP_011` §0 are three
precedents for referring instead. **Y5 is FAILED.** The 102.5× is reported as a
**marker with no verdict attached**, in the `EXP_011` §0 sense.

What the marker says, stated without a verdict: **on the one current pair where
the measurement has power, feeding the identical perturbation through the
two-compartment neuron instead of the LIF removes 99.0–99.98 % of the flips at
matched firing rate.**

### 9.5 Y4 and Y6 also failed, and Y6's failure is the mechanism showing through

**Y4 — amplification.** Composed arm **34.0×** on both seeds against `EXP_008`'s
14.6 / 15.9 / 17.9. Predicted `< 5×`; the direction is **opposite** to the
prediction — the composed arm amplifies *more* per layer. But both figures are
34/1 and 68/2: **an "amplification ratio" built from a denominator of 1 and 2 is
not a measurement of one**, and limitation 2 applies to it exactly as it applies
to Y5. Reported as **FAILED with its counts beside it**, and not offered as
evidence that the two-compartment neuron amplifies more.

**Y6 — the flip model.** Predicted ÷ measured:

| | layer 0 | layer 1 |
|---|---:|---:|
| `threshold_s0/s1/s2` | **3.50 / 8.82 / 5.69** | 1.11 / 1.19 / 1.17 |
| `compose_s1/s2` | 4.50 / 1.75 | **0.97 / 0.84** |

Bar was 3× on every cell, so **Y6 FAILED**. Per §3's Y6 row, **no mechanistic
claim is made at the quantitative level** from Y3 or Y5.

**The pattern of the failure is itself informative and is reported as such.** The
model assumes a *smooth* density — it estimates `rho_u(0)` from `P(|u| < E|du|)`
and multiplies. It predicts within 1.2× exactly where §9.3 says the density is
smooth (both composed layers; the LIF's layer 1, whose input is 512 binary spikes
rather than 205 embedding rows) and over-predicts by 3.5–8.8× exactly where §9.3
says there is an atom (the LIF's layer 0). **A first-order density model cannot
predict flips through a point mass**, and Y6 failing on precisely those four
cells is consistent with §9.6 rather than with §1's mechanism being absent.

### 9.6 POST-HOC — the LIF's near-threshold mass is 20 values replayed 548 times

**This subsection is post-hoc.** Its hypothesis was formed after reading §9.3's
ladder, `scripts/exp/012_posthoc_pileup.py` carries the same label in its
docstring, and it is separated from §9.0–9.5 so that nothing here is counted as
a pre-registered hit. `EXP_008` §9.5 is the precedent for a chase that follows a
result rather than preceding it.

Sites with `|u| < 1e-6` at layer 0:

| | count | distinct `u` values | repeats per value | previous step spiked | channels |
|---|---:|---:|---:|---:|---:|
| `threshold_s0` | 10 962 | **20** | **548.1** | 0.898 (2.61× baseline) | 60 / 512 |
| `threshold_s1` | 12 436 | **23** | **540.7** | 0.908 (2.53×) | 68 / 512 |
| `threshold_s2` | 9 213 | **23** | **400.6** | 0.870 (2.47×) | 69 / 512 |
| `compose_s1` | 5 | 4 | 1.2 | 0.800 (2.11×) | 5 / 512 |
| `compose_s2` | 2 | 2 | 1.0 | 0.000 | 2 / 512 |

**The LIF's ten thousand near-threshold sites are twenty numbers.** Layer 0's
input is one of only `V = 205` embedding rows, so `cur` takes 205 values per
channel; the hard reset sets `v` to **exactly 0**, so the step after a spike has
`v_pre = cur` — a value from that small set. A channel whose `cur` happens to
land within an ulp of `thr` therefore reproduces the *same* `u` every time that
token follows a spike, in 60–69 channels of 512. The two-compartment neuron never
resets `vs`, so `u = vf + w·vs` carries a continuously-varying history and lands
on the same value 1.0–1.2 times.

**Note what does *not* discriminate.** Post-spike enrichment is 2.61× on the LIF
and 2.11× on the composed arm — *both* neurons' near-threshold sites are enriched
for the step after a spike, so enrichment alone would have been a false positive.
**The repeat count is the discriminator**, and it is the falsifier the script's
docstring named in advance of running it.

### 9.7 What is established, what is not, and what is referred

**Established:**

1. The gap is **not** the algebra (Y1) and **not** the injected perturbation
   (Y2). Both controls held on both arms.
2. The gap is in the **layer-0 flip count**, by 1 342–5 339× (Y3).
3. The flip count tracks the **mass of the decision variable at the threshold**,
   which differs by 1 843–6 218× at `h = 1e-6` while the membrane's overall
   spread differs by only ~1.19× (§9.3).
4. The LIF's near-threshold mass is a **degeneracy, not a density**: 20–23
   distinct values repeated 400–548 times each (§9.6, post-hoc).

**Not established:**

5. **That the neuron causes it.** The powered cross-over leg says so at
   102.5–5 339×, but **Y5 as written failed** and this file does not overturn its
   own rule. §9.4's number is a marker.
6. **The quantitative mechanism.** Y6 failed; §9.5's pattern is consistent with
   §9.6 but is not a validated model.
7. **That any of it holds at another scale, on another reparameterisation
   (quantisation, weight normalisation), or on `compose_s0`**, which diverged and
   was not replaced (§10 items 3–4, and `EXP_011`'s provisional n = 2).

**Referred to Elliot, applied nowhere:**

8. **Decision #6 is untouched and still open.** This experiment set no tolerance
   and proposes no number for W1. What it adds is that **a fold-in tolerance
   cannot be one project constant**: the same identity through the same fold
   costs 2.0e-03 bpc on one neuron and 1.4e-05 on another, and the difference is
   structural rather than incidental.
9. **§4's pre-registered consequence did NOT fire.** The trigger was `Y2 ∧ Y5`,
   and Y5 failed. **So the Phase-4 report §6.5's noise-floor sentence stands as
   written**, exactly as §4 said it would in that case. Items 2–4 above are
   nonetheless evidence that the ~2e-3 floor is a property of the *baseline LIF
   neuron* and not of "this architecture", and **that scope qualifier is referred
   as a recommendation, not applied.** Redefining it is Elliot's, and the trigger
   this file wrote for itself did not fire.
10. **Y5's specification.** It should have required the reference leg to clear
    the resolution floor before forming a ratio. The corrected wording is *not*
    adopted here.

### 9.8 Status

**CLOSED.** Y1, Y2, Y3 held; Y4, Y5, Y6 failed. Three of six predictions failed
and two of the three failures are at the resolution floor — which is a design
weakness this file names in §9.4 and §9.5 rather than a property of the arms.
Nothing is adopted, no tolerance is set, no phase is authorised, and the §6.5
scope qualifier is **not** applied.

---

## 10. What this experiment does not answer

1. **What the fold-in tolerance should be.** Decision #6, Elliot's, untouched.
2. **Which arithmetic in layer 0's backward makes `compose_s0`'s NaN.** A
   different open chase; this one is a forward-pass question entirely.
3. **Whether the two-compartment neuron's insensitivity survives quantisation or
   weight normalisation**, which are the other reparameterisations §6.5's floor
   was written to cover. This experiment measures one fold on one architecture
   pair.
4. **Whether any of it holds at another scale**, which is the third thing the
   Phase-4 report §10.5 lists as unmeasured.
