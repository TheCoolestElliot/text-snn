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

*(appended after the run; hypotheses above are not edited)*

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
