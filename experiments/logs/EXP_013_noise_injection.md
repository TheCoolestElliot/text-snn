# EXP_013 — Does injected background spike noise regularise a model that does not overfit?

**Pre-registered:** 2026-08-05, after the two calibration measurements in §2.1
and §2.2 — both of which read only the *committed Phase-2 baseline* and no arm —
and **before any noise arm was trained**. The arm's code (`src/snn/noise.py`,
`NoisyCharLM`) was written alongside this file; no training run starts until this
file is committed. That sequencing is stated rather than claimed as
"before any code existed", because it was not.
**Status: CLOSED 2026-08-05 — predictions resolved exactly as §3 wrote them, including the three that FAILED and the one that held for a reason that does not support it.** A null: this model does not overfit. Results in §9; folded into `docs/reports/04_phase4_interim.md` §12.**
**Phase:** 4 (controlled experiments). **This candidate is not on
`03_phase3_candidates.md` §6.3's list, and is not on the 18-candidate set in
`docs/reports/data/phase3_candidates.json` at all** — see §0.

**What it costs:** nine training runs at the committed recipe (3 amplitudes × 3
seeds) plus fourteen inference-only gap evaluations, **~1.6 GPU-hours**. The
noise-off control is trained **zero** times: it is the five committed Phase-2
baseline seeds, and §6 gate G1 is what earns the right to say so.

---

## 0. What this experiment is, and what it is not, on the candidate record

**Stochastic noise injection appears nowhere in this project's candidate set.**
`docs/reports/data/phase3_candidates.json` enumerates all eighteen candidates —
fourteen ranked, one refuted, three referred — and none of them is this. It is
not the refuted one (micro-step refinement at T > 1) and it is not one of the
three referred to Elliot. There is no prior EXP log, no ruling, and no adoption
decision. The only occurrences of the concept anywhere in the tree are three that
are not about this project's model: a literature-review line about a *cited*
LSTM baseline's grid-searched recurrent dropout, `snn/evaluate.py`'s docstring
promising "no sampling, no dropout", and a comment in `snn/train.py` observing
that reproducibility "breaks once anyone adds dropout".

So this file opens a candidate rather than running one, and that is worth
stating plainly for two reasons.

1. **It sits in a named area that the Phase-3 generation step missed.**
   `03_phase3_candidates.md` §6 says the eighteen span "the six areas the
   Phase-1 plan names", of which one is *optimisation*. §6.5 of that report
   already records that the generation step's provenance is the thinnest in the
   document — six of the eight candidates its completeness audit claims to have
   added are unattributable from the committed record — so a gap in that area is
   a gap in exactly the place the report itself flags as least trustworthy. This
   file is evidence for §6.5's caution and is recorded as such.
2. **It is the kind of arm the Phase-4 report asks for by name and the ranked
   list cannot supply.** `04_phase4_interim.md` §6.7 item 3 and §7 item 5:
   *"Phase 4's bits are being won by optimisation, and the ranked list is
   entirely architectural… Phase 5 should carry at least one arm that changes
   only how the model is trained."* This arm changes only how the model is
   trained. It adds no parameter, no state variable, no kernel and no function to
   the model's class, and at inference it does not exist.

**Adopting it, ranking it, or adding it to §6.3 is Elliot's, not this file's.**
This file measures it.

---

## 1. Why this experiment, and what it is really testing

### 1.1 The mechanism

Inject a zero-mean background spike train into every layer's input current during
training, and only during training:

```
cur'_{t,c} = cur_{t,c} + amp · (B_{t,c} − p)/sqrt(p(1−p)),   B ~ Bernoulli(p)
```

i.i.d. over batch, time and channel. `B` is a literal background event train:
each (step, channel) either receives a background input event or does not. The
affine rescaling makes the process zero-mean and unit-variance, so **`amp` is the
standard deviation of the injected current, in the same units as the firing
threshold** (which is 1.0), and `p` sets only how sparse the events are.

At inference the term is absent, so every committed evaluation protocol applies
unchanged and the arm costs the Phase-2 baseline exactly. There is nothing to
fold: unlike `EXP_008`'s gain, this is not a reparameterisation that has to be
absorbed — it is simply not there when `model.eval()` is set.

### 1.2 The `− p` is the whole design, and it is derived rather than chosen

The obvious implementation of "inject background spikes" is `cur + a·B`, and it
is **wrong for this experiment in a way that would not have shown up in the
result**.

Uncentred background spikes add a constant `a·p` to every channel's current at
every step. A constant added to the current is a uniform reduction of the
effective firing threshold: in the steady state `v ≈ cur/(1−beta)`, so firing at
`v ≥ thr` becomes firing at `v ≥ thr − a·p/(1−beta)`. That is `EXP_008` §1.1's
Identity 1 in its uniform form — and `EXP_008` **measured** a per-channel version
of exactly that mechanism to be worth **0.0590 bpc**, 44 % of the adopted arm's
entire gain.

So an uncentred arm would confound "regularisation" with "threshold shift", and
the confound would be **an order of magnitude larger than the entire quantity
this experiment is aimed at** (§2.2: the whole generalisation gap is 0.0059 bpc).
It would very likely have produced a bpc improvement, and that improvement would
have been `EXP_008`'s result arriving through a door marked "regularisation".

Subtracting `p` costs one kernel and removes it. The arm is then a pure
*second-moment* intervention: it changes the variance of the current and nothing
about its mean. This is the project's standing rule that a derivation done before
implementing is allowed to change what the experiment is (`EXP_008` §1.1 is the
precedent), and here it changed it from a confounded arm to a clean one.

### 1.3 Why it is built on the Phase-2 baseline and not on the adopted arm

`EXP_007` and `EXP_008` are both *baseline plus one thing*, and `EXP_011`
composed onto the adopted neuron afterwards. This file follows that order for
three reasons: the noise-off control already exists at **n = 5** rather than
needing to be trained; the gap statistic's own noise floor has been measured on
those same five seeds (§2.2) and on nothing else; and `EXP_005`'s lesson is that
a statistic measured on one neuron does not transfer, so the arm should first be
run where its control's statistics are known.

**The composition onto `arch="twocomp"` is the obvious follow-up and is not run
here.** `src/snn/noise.py` is written against `cur` rather than against a neuron,
so that follow-up is one subclass, exactly as `TwoCompThresholdCharLM` was.

---

## 2. Design

### 2.1 Calibration 1 — the scale of the current the noise is added to

An amplitude means nothing without the scale of the thing it perturbs, and
`03_phase3_candidates.md` §6.2 is this project's precedent for measuring a bar
before using it: the acceptance bar could not be used until calibrated, and a
flat one would have been 3× too strict in one place and 30× too lax in another. A
noise ladder of three round numbers has the same defect and no way to notice it.

`scripts/exp/013_calibrate_noise_scale.py` measures `sd(cur)` per layer on the
five committed Phase-2 baseline checkpoints under the **training** data
distribution (the noise is a training-time intervention, so the training current
is the relevant one). Artifact: `docs/reports/data/exp_013_noise_calibration.json`.

| | layer 0 | layer 1 |
|---|---:|---:|
| `sd(cur)`, median over 5 seeds | **3.167** | **1.409** |
| range over seeds | 3.155 – 3.202 | 1.390 – 1.611 |

**Self-check, and it is the F1 rule applied to an intermediate:** to observe
`cur` at all the script re-runs `_CharLMStack.forward`'s loop by hand, which is
the kind of second implementation `EXP_006`'s L5 and `EXP_012`'s S1 exist to
distrust. Every replay's logits are asserted **bit-identical** (`torch.equal`) to
`model(idx)`'s, and the run aborts otherwise. **25 of 25 replays are
bit-identical.**

### 2.2 Calibration 2 — the gap the arm is aimed at, and it is almost not there

This is the measurement that determines what this experiment can and cannot
establish, and it was taken **before** the arm existed.

`scripts/exp/013_generalisation_gap.py` scores the five committed baseline seeds
on a train slice and on test under both committed protocols. Artifact:
`docs/reports/data/exp_013_gap_baseline.json`.

```
gap = bpc(test) − bpc(train slice)          positive = the overfitting direction
```

The train leg is the **first 19 456 windows** of the train split, matched exactly
to the test split's own window count, so the two legs of the difference are the
same size. Both legs use `snn.evaluate.evaluate`, the function that produced
every bpc in this project; nothing is reimplemented.

| protocol | train bpc | test bpc | **gap** | cross-seed sd | per-seed gaps |
|---|---:|---:|---:|---:|---|
| fresh | 2.26375 | 2.26969 | **+0.00594** | **0.00268** | +0.00874, +0.00450, +0.00635, +0.00214, +0.00797 |
| carried | 2.25491 | 2.25311 | **−0.00180** | **0.00306** | +0.00163, −0.00336, −0.00244, −0.00574, +0.00093 |

**The probe reproduces two numbers it did not produce.** Its test legs return
2.26969 fresh and 2.25311 carried — the committed Phase-2 baseline figures in
`02_baseline_report.md` and `04_phase4_interim.md` §6.1, to five decimals. That
is `EXP_001`'s F1 discipline applied to a new instrument, and it is why the gap
numbers beside them are trustworthy.

**Three things follow, and the third is a ceiling.**

1. **Under `carried` there is no gap at all.** −0.00180 against a cross-seed sd
   of 0.00306, with three of five seeds *negative*. It is indistinguishable from
   zero and points the wrong way.
2. **Under `fresh` there is a small gap and it is consistent** — all five seeds
   positive, mean +0.00594, which is 4.96 se from zero. It is nonetheless
   **smaller than this project's own adoption bar** (2σ = 0.00922) and only ~3×
   the reparameterisation noise floor `04_phase4_interim.md` §6.5 measured.
3. **The absolute gap is confounded, and the difference between arms is not.**
   The train slice and the test split are different text, so `gap` mixes
   memorisation of seen text with the intrinsic difficulty difference between two
   fixed regions — which is what the fresh/carried disagreement is most likely
   made of. The difficulty term is **the same constant for every arm**, because
   every arm is scored on the identical two regions under the identical protocol.
   So it cancels in `Δgap = gap(arm) − gap(baseline)`, and **`Δgap` is the
   decision variable**, exactly as `EXP_001` §2.2 made text difficulty cancel by
   pairing.

### 2.3 The ceiling, stated before the run because that is when it counts

`03_phase3_candidates.md` §5 is the precedent: *"no candidate proposed in this
phase quoted a ceiling that acknowledged it"*, and quoting one is now house
practice.

**A regulariser can only remove the memorisation component of the gap, and the
whole gap — memorisation plus region difficulty — is 0.00594 bpc under `fresh`
and indistinguishable from zero under `carried`.** Against a bar of
2σ_gap = 0.00536 (fresh), detecting a reduction requires the arm to remove
**≥ 90 % of the entire fresh gap including the part no regulariser can touch.**
Under `carried` there is nothing to remove at all.

**So this experiment is underpowered by construction for its own primary success
criterion, and that is known now rather than discovered in §9.** Stating a
prediction's power before running it is `EXP_005` S1's rule, and the reason it
exists: a null on N1 must be reported as *"there was no gap to close"*, never as
*"noise injection does not regularise"*.

The reason the model does not overfit is not mysterious. 735 K parameters see
~655 M training characters against a 90 M-character split — ~7.3 epochs at ~122
characters per parameter. **This project's problem is underfitting** (2.25 bpc
against the GRU anchor's 1.77), and a regulariser is a treatment for the opposite
disease.

**What the experiment is still powered to answer**, and these are not small
questions: whether noise injection *costs* bits, at what amplitude it starts to,
and whether it moves `Δgap` in the wrong direction. All three are measured
against the standing 2σ = 0.00922 bar, which is 1.55× the whole gap.

### 2.4 The amplitude ladder

Three amplitudes, geometric at ×4, fixed now:

| `noise_amp` | as % of `sd(cur)` L0 / L1 | injected membrane sd, in thresholds |
|---:|---|---:|
| **0.1** | 3.2 % / 7.1 % | 0.115 |
| **0.4** | 12.6 % / 28.4 % | 0.462 |
| **1.6** | 50.5 % / 113.6 % | 1.848 |

The membrane column is `amp/sqrt(1−beta²) = amp·1.1547`, the stationary sd of the
noise after the membrane integrates it at `beta = 0.5`, in units of the threshold.

**`amp = 1.6` is expected to be destructive and is included deliberately.**
`EXP_006`'s ladder ran to N = 512, where the intervention was 244 % of the arm's
gain, for the same reason: a ladder that only contains points where nothing
happens cannot tell a null apart from a dose too small to matter.

**`p = 0.1` is held fixed and is an unswept point.** It controls only the
sparsity of the events, not the injected variance, but sparsity at fixed variance
changes the kurtosis of the perturbation and nothing here bounds that effect.
`EXP_009` §9.3's rule applies: a null at one `p` is a null *of that `p`*.

### 2.5 The single variable

`arch="noise"` differs from `arch="snn"` in exactly one thing: `noise_amp > 0`.
Same optimiser, same schedule, same 20 000 steps, same data order at a given
seed, same seeds (0, 1, 2), same evaluation. `Config.noise_p` is held at 0.1
throughout. The driver's sameness check asserts every other field against
`snn_beta0.5_s0`'s committed `config.json`, which is `EXP_011`'s K1 gate.

### 2.6 What is measured

Per run: test bpc and train-slice bpc under both protocols, `Δgap` against the
baseline, per-layer firing rate on both splits, wall-clock, peak VRAM, and the
per-context bpc curve required by `03_phase3_candidates.md` §6.2.

---

## 3. Hypotheses

Every threshold below is fixed by this file before any noise arm is trained.
σ_gap is **measured** (§2.2) and not inherited, per `EXP_005`.

* **σ = 0.00461**, the Phase-2 baseline whole-split sd; bar **2σ = 0.00922** for
  test bpc, the standing project bar for this neuron.
* **σ_gap = 0.00268 (fresh), 0.00306 (carried)**; bar **2σ_gap = 0.00536 (fresh),
  0.00612 (carried)** for `Δgap`.

| | prediction | resolves |
|---|---|---|
| **N1** | *(primary, and predicted to fail — §2.3)* At least one amplitude reduces the gap: `Δgap ≤ −2σ_gap` under `fresh`. | the tasked success criterion |
| **N2** | Test bpc degrades monotonically in amplitude across {0.1, 0.4, 1.6}, and at `amp = 1.6` is worse than the baseline by more than 2σ = 0.00922. | is the ladder wide enough to bracket the effect |
| **N3** | At `amp = 0.1` test bpc is within 2σ of the baseline in **both** directions — neither help nor harm. | is the bottom of the ladder in the null region |
| **N4** | Mean firing rate rises monotonically in amplitude, on both splits and in both layers. | mechanism: a symmetric perturbation of a sub-threshold membrane adds crossings |
| **N5** | `Δgap ≥ 0` at every amplitude — the train leg degrades at least as much as the test leg. | the direction opposite to N1, stated so a null is not read as "no effect" |
| **N6** | The arm's own cross-seed sd of test bpc is within a factor of 2 of the baseline's 0.00461. | `EXP_005`: σ does not transfer; n = 3 estimates it to ±40 %, which is stated now |

**N1 and N5 are complementary and both are pre-registered.** N1 is what the arm
is being run to test; N5 is what §2.3's ceiling argument predicts instead. Naming
both in advance is what stops the result being narrated in whichever direction it
lands. **N2, N4 and N5 are stated against the arm's own interest.**

---

## 4. Decision rule, fixed now

Applied mechanically by `scripts/exp/013_noise_results.py`.

| cell | verdict |
|---|---|
| **N1 holds** | The arm regularises a model that barely overfits. **Report the effect and refer adoption to Elliot**, with §2.3's ceiling quoted beside it, because a reduction ≥ 90 % of a gap that includes an irreducible region-difficulty term demands an explanation before it is believed. |
| **N1 fails ∧ N3 holds** | **Null, and reported as "there was no gap to close"** — never as "noise injection does not regularise". The arm is free at small amplitude and buys nothing measurable. Not recommended for adoption. The result that transfers is §2.2's: **this model does not overfit**, which bears on every future regularisation candidate and on the parameter-scaling pilot (`03_phase3_candidates.md` §6.3 #11). |
| **N1 fails ∧ N3 fails, harm direction** | The arm costs bits even at 3 % of the current's sd. Report as a **negative result with the amplitude at which harm begins**, which bounds every future arm that perturbs the current — including quantisation, which `04_phase4_interim.md` §6.5 names as unmeasured. |
| **N1 fails ∧ N3 fails, help direction** | `amp = 0.1` improves test bpc without touching the gap. **Stop and chase it** before reporting anything: an improvement from a zero-mean perturbation with no gap to close would most likely be §1.2's confound leaking back in, and the first thing to check is whether the injected mean is zero to machine precision on the actual draws. |

**No adoption is made by this file** and no candidate is added to
`03_phase3_candidates.md` §6.3. `EXP_008` §4's rider is the precedent.

**A bar failing does not get the bar rewritten.** `EXP_005` §9.4, `EXP_008` §9.4,
`EXP_011` §0 and `EXP_012` §11.4 are four consecutive precedents; if N1's bar
turns out to be the wrong construction, the corrected construction is *referred*,
and both numbers are reported.

---

## 5. Known limitations, stated in advance

1. **The gap is confounded by text region** (§2.2 item 3). `Δgap` cancels it;
   the absolute gap is not a pure memorisation measure and is never quoted as one.
2. **The train leg is a fixed head of the split, not a random sample.** A random
   sample would need a sampling seed, which would put an RNG inside the decision
   variable of an experiment about injected randomness.
3. **`p = 0.1` is unswept** (§2.4).
4. **One backbone.** The arm is not run on the adopted two-compartment neuron
   (§1.3), so nothing here says what it would do there — and `EXP_005` and
   `EXP_012` are two separate demonstrations that this project's statistics do not
   transfer between those two neurons.
5. **n = 3 per amplitude** against the baseline's n = 5. N6 exists because that
   estimates the arm's own σ to about ±40 %.
6. **The experiment is underpowered for N1 by construction** (§2.3). This is a
   limitation of the *model*, not of the design, and no amount of seeds fixes it:
   the gap is 0.0059 bpc because the model has 735 K parameters.
7. **Zero counts and near-zero differences.** `EXP_012`'s Y5 resolved on a leg
   comparing 0 flips against 1, because a bar was written without the guard its
   own limitations section had already named. Applying that lesson here: **any
   `Δgap` whose magnitude is below the reparameterisation noise floor of ~2e-3
   is reported as "below what this instrument resolves", not as a number with a
   sign.** N1, N3 and N5 all carry that guard, and it is written into the bar
   rather than left to the write-up.

---

## 6. Self-checks — aborting, not advisory

| | check |
|---|---|
| **G1** | **The nesting.** `arch="noise"` at `noise_amp = 0` must reproduce `SpikingCharLM` **bitwise, forward and backward** — `torch.equal` on logits and on every gradient, not a tolerance. This is what licenses using the five committed baseline seeds as the noise-off control instead of training three more. `EXP_011` K4's standard. |
| **G2** | **The noise is off at inference.** Under `model.eval()` the arm's logits must be bitwise identical to the baseline's at every amplitude. If they are not, the arm is not a training-only intervention and every bpc in §9 is measuring the wrong thing. |
| **G3** | **The draw is centred and correctly scaled.** On a large draw, the sample mean must be within 4 se of 0 and the sample sd within 1 % of `amp`. §1.2 makes this the difference between this experiment and `EXP_008`'s. |
| **G4** | **Resumability (R6).** The noise at `(seed, step)` is a pure function of `(seed, step)`: refilling twice at the same step gives bitwise identical buffers, and at different steps gives different ones. |
| **G5** | **Independence from the data stream.** The noise generator's seed is salted, so the noise at step k must not be reproducible from the sampler's generator at step k. Asserted by construction and tested. |
| **G6** | **The mutation legs.** G1, G3 and G4 must each be shown to fail under a deliberate corruption — an uncentred draw, a shared-across-batch buffer, and a step-independent seed. A check nothing has ever tripped is indistinguishable from one that cannot trip (`03_phase3_candidates.md` §7.3). |
| **G7** | **F1.** Every gap evaluation's test leg must reproduce the committed bpc of the arm it scores, and the baseline legs must reproduce 2.26969 / 2.25311. |

**No R10 gradient gate and no mutation campaign are required**, and the reason is
structural rather than convenient: the arm adds no recurrence, so it calls the
committed `lif_scan` with its committed backward, and the gradient through an
additive constant is the identity. This is the same argument `snn/prescan.py`
makes for `EXP_007` and `EXP_008`, and `03_phase3_candidates.md` §7.1 prices it
as most of why those two rows were cheap.

**The §7.2 gradient-reachability screen does not apply**: the arm introduces no
learnable parameter, so there is no initialisation for a saddle to sit in. This
is the same disposition the candidate spec records for #9 (`applies: false`).

---

## 7. Failure modes

* **P1 — the confound returns.** If the draw's empirical mean is not zero, §1.2's
  threshold shift is back and any bpc improvement is `EXP_008`'s result in
  disguise. G3 is the guard; §4's fourth cell routes an improvement to a chase
  rather than to a headline.
* **P2 — capture breaks.** An RNG op inside the captured region would make R6
  untestable for this arm. The buffers are filled outside it; if capture fails,
  the documented fallback is `cfg.cuda_graph=False`, which changes wall-clock and
  must be reported if used.
* **P3 — the null is narrated as a finding.** §4's second cell fixes the wording
  in advance, because "noise injection does not help" and "this model has no
  overfitting to regularise" are different claims and only the second is
  supported.
* **P4 — VRAM.** Two `[128, 256, 512]` fp32 buffers are 0.125 GiB on top of the
  baseline's 0.609 GiB. If peak VRAM exceeds ~1.2 GiB the buffers are being kept
  alive by autograd, which would mean the add is not the constant-addend it is
  supposed to be.

---

## 8. Entry conditions

1. G1 and G2 pass, before any training run starts.
2. The test suite passes with `tests/test_noise.py` added.
3. `docs/reports/data/exp_013_noise_calibration.json` and
   `exp_013_gap_baseline.json` are committed — the ladder and the bar both rest
   on them, and a bar whose calibration is not committed is a bar chosen after
   the fact.
4. No mutation campaign is running (lockfile check), per `03_phase3_candidates.md` §8.
5. **This file's SHA-256 is recorded in `exp_013_run_manifest.json` before the
   first training run starts, and re-checked after the last one.** Every other
   Phase-4 experiment says "committed before the numbers it produces were read"
   (`04_phase4_interim.md` §9); this file does not make a git commit, because
   committing is Elliot's decision and not this experiment's. The hash is what
   discharges the obligation the commit was discharging: it is a cryptographic
   commitment that no threshold in §3 or §4 moved after the results arrived, and
   it is checkable by anyone with the file. If the hash differs at the end of the
   run, the driver says so and the experiment is void.

---

## 9. Results

Appended 2026-08-05, after all runs. **Predictions are resolved exactly as §3
wrote them, including the three that failed and the one that held for a reason
that does not support it.**

**§8 item 5's hash, and how to check it now that this section exists.**
`exp_013_run_manifest.json` records
`2e9ee41e98bbde93f9dc8595e053bb816862202b91960b5f73e32c168c68fa58` both before
the first run and after the last. Appending §9 and §10 necessarily changes the
whole-file hash — that is what "results are appended" means — so the recorded
value no longer matches this file, and pretending otherwise would be worse than
saying so.

**The guarantee is still checkable, exactly.** Replace everything from
`## 9. Results` onward with the two-line stub this section replaced —

```
## 9. Results

*(Empty at pre-registration. Appended after the runs; predictions are resolved as
written, including the ones that fail.)*
```

— and the SHA-256 of the result is `2e9ee41e…`, **verified**. So §0–§8 are
byte-identical to what was hashed before the first training run: no hypothesis,
no bar and no decision cell moved after the numbers arrived. The prefix through
the end of §8 hashes to `285eb661aa454a0dcbd8fd7b35f64f11c56e4c1624baa6631bd3f8dc6782ef8d`,
which is the value a future revision of this section should be checked against
instead.

**Hardware and versions.** NVIDIA GeForce RTX 5060, 8 151 MiB, driver via CUDA
13.0; torch 2.12.1+cu130; Python 3.14; Windows 11. Seeds 0, 1, 2 at every
amplitude. ~1.9 GPU-hours: ten training runs at ~428 s each (nine plus one
re-run, §9.8), fourteen gap evaluations, nine horizon probes and two
calibrations.

**Gates.** K1 passed on all nine runs — the only fields differing from
`snn_beta0.5_s0` are `arch`, `noise_amp` and `run_name`, with `beta_slow`,
`w_init`, `mu_init`, `thr_log_init` and `noise_p` excused as absent-from-reference
and left at the `Config` default. K2 (no mutation campaign) checked before every
run. F1: **9 of 9**, residuals 1.45e-09 – 1.18e-08, `f1_failures` empty.

### 9.1 The scoreboard

| arm | n | test bpc, carried | Δ vs baseline | se | cross-seed sd | wall-clock | peak VRAM |
|---|---:|---:|---:|---:|---:|---:|---:|
| Phase-2 baseline | 5 | 2.25311 | — | — | 0.00461 | 1.00× | 0.609 GiB |
| **noise, amp 0.1** | 3 | **2.25731** | **+0.00419** | +1.2 | **0.00167** | 1.068× | 0.796 GiB |
| **noise, amp 0.4** | 3 | **2.30290** | **+0.04978** | +14.8 | 0.00325 | 1.089× | 0.796 GiB |
| **noise, amp 1.6** | 3 | **2.89287** | **+0.63975** | +190.0 | **0.01817** | 1.073× | 0.796 GiB |

**The arm costs bits at every amplitude and buys none at any.** Peak VRAM is
+0.187 GiB, exactly the two `[128, 256, 512]` fp32 buffers (0.125 GiB) plus one
extra `[B, L, d]` intermediate, so P4's diagnosis — that a larger figure would
mean autograd was keeping the buffers alive — does not fire.

### 9.2 N1 held, and the decomposition says it must not be believed

`Δgap ≤ −2σ_gap` was reached at **two of three amplitudes**, so §4's cell is
**"N1 holds"**. That cell required "an explanation before it is believed", and
the explanation is available and is decisive:

| amp | Δ train bpc | Δ test bpc | **Δgap** | resolved? |
|---:|---:|---:|---:|---|
| 0.1 | +0.00468 | +0.00387 | −0.00081 | **no** — below the 2e-3 floor |
| 0.4 | +0.05878 | +0.05013 | **−0.00865** | yes |
| 1.6 | +0.67083 | +0.64130 | **−0.02953** | yes |

`Δgap = −(Δtrain − Δtest)` to every digit shown. **The gap closed because the
model got worse at the training text faster than it got worse at the test text.**
Nothing generalised better; the train leg simply fell further. At the two
amplitudes where N1 fires, test bpc is **14.8 se and 190 se worse** than the
baseline.

**So N1's bar is satisfied by any sufficiently damaged model**, and that is a
defect in the bar, not a property of the arm. A generalisation gap is a
difference of two quantities, and this experiment pre-registered a threshold on
the difference without pre-registering that the subtrahend must not move — which
§2.3's ceiling argument had already implied and §3 did not encode.

**The bar is reported as it fired and is not rewritten.** `EXP_005` §9.4,
`EXP_008` §9.4, `EXP_011` §0 and `EXP_012` §11.4 are four consecutive precedents,
and `EXP_012`'s is the closest: a rule fired on a leg with no power, the powered
number pointed the other way, and the rule was recorded rather than repaired. The
corrected form — *a gap reduction counts only when Δtest ≤ 0* — is **referred, not
applied**, and it is §10's first item.

**What this leaves as the actual finding is §2.2's, and it was measured before
the arm existed:** this model does not overfit — the gap is +0.00594 bpc under
`fresh` and indistinguishable from zero under `carried` — so there was nothing
here for a regulariser to close.

### 9.3 N5 contradicts its own gloss, and the error is in the pre-registration

§3 states N5 as **"`Δgap ≥ 0` at every amplitude"** and glosses it in the same
cell as **"the train leg degrades at least as much as the test leg"**. Those are
opposite: if the train leg degrades more, `test − train` *falls* and Δgap goes
negative. The resolver implemented the formal statement, and **N5 FAILED** at
amps 0.4 and 1.6.

**The gloss is what happened, at all three amplitudes.** The formal statement is
what was scored. Both are reported and neither is retro-fitted; the drafting
error is mine and it is named here rather than resolved by choosing the reading
that looks better after the fact.

**The consequence is worth more than the correction.** N1 and N5 were presented
in §3 as *complementary* predictions — "N1 is what the arm is being run to test;
N5 is what §2.3's ceiling argument predicts instead". They are not complementary.
They are the same measurement with opposite signs, so N1 holding and N5 failing
is **one event, not two**, and §3's claim to have pre-registered both directions
bought less independence than it appeared to.

### 9.4 N4 failed, and the failure is layer-specific

Mean firing rate on test, `carried`:

| arm | layer 0 | layer 1 |
|---|---:|---:|
| baseline | 0.3372 | 0.3228 |
| amp 0.1 | 0.3511 | 0.3536 |
| amp 0.4 | **0.3390** | 0.4673 |
| amp 1.6 | 0.3614 | 0.6583 |

**Layer 1 rises monotonically and steeply** — 0.323 → 0.354 → 0.467 → 0.658 —
which is the predicted mechanism: a symmetric perturbation of a sub-threshold
membrane adds crossings. **Layer 0 does not**: it rises, falls at amp 0.4, then
rises. N4 required every layer, so it **FAILED**.

No mechanism is claimed for layer 0's non-monotonicity. It is recorded because
`EXP_012` §11.6 found layer 0's decision variable structurally unlike layer 1's —
its input is one of only 205 embedding rows and its near-threshold mass is a
degeneracy rather than a density — and a layer-0-specific anomaly in a
perturbation experiment is the kind of thing that should be on the record when
the next experiment perturbs that layer. **That is a pointer, not a finding**: this
experiment did not measure the density and cannot attribute the effect.

### 9.5 N6 failed in both directions, which is more informative than failing in one

Cross-seed sd of test bpc, `carried`, against the band [0.00231, 0.00922]:

| arm | sd | vs baseline 0.00461 |
|---|---:|---|
| amp 0.1 | **0.00167** | 0.36× — **below** the band |
| amp 0.4 | 0.00325 | 0.71× — inside |
| amp 1.6 | **0.01817** | 3.94× — **above** the band |

**σ is not a property of the neuron; here it is a property of the
hyperparameter.** `EXP_005` established that σ does not transfer between two
*architectures*; this is the same lesson inside one architecture, moved by a
scalar that changes no parameter and no function. At small amplitude the noise
*averages seeds together* and tightens the spread by 2.8×; at large amplitude it
tears them apart by 3.9×.

**n = 3 estimates a standard deviation to about ±40 %** (§5 item 5, stated in
advance), so the direction of each departure is better supported than its size —
but a 0.36× and a 3.94× cannot both be explained by ±40 %.

### 9.6 The per-context curve, and what the noise-floor guard stopped being claimed

`03_phase3_candidates.md` §6.2's criterion, against the five baseline seeds'
per-context 2σ bars:

| arm | contexts significantly worse | significantly better | worst | horizon (3 seeds) |
|---|---:|---:|---:|---|
| amp 0.1 | **5 / 128** | 3 / 256 | +0.0267 at c = 0 | 7, 6, 7 |
| amp 0.4 | 126 / 128 | 2 / 256 | +0.4311 at c = 0 | 6, 7, 7 |
| amp 1.6 | 128 / 128 | 0 / 256 | +1.9797 at c = 0 | 6, 6, 6 |

**The memory horizon does not move**: 6–7 everywhere against the baseline's
5/5 seeds at exactly 7. Whatever the noise does, it is not a horizon effect.

**Amp 0.1 is not a flat null, and this nearly became an over-claim.** Its curve is
significantly *worse* at c = 0 (+0.0267 against a 0.0140 bar) and significantly
*better* at c = 2, 3 and 4 (−0.0311, −0.0378, −0.0103). Read as a per-context
table that is a real, structured effect pointing at the within-reach component —
the one `04_phase4_interim.md` §7 item 2 calls 51.8 % of the remaining gap and
"still barely attacked".

**Weighted by the characters at each context, it is not resolvable.** On
`EXP_004` §10.3's basis (cut at c = 0 | 1 ≤ c ≤ 7 | c ≥ 8; positive = better):

| amp | total | zero context | within reach | beyond horizon |
|---:|---:|---:|---:|---:|
| 0.1 | **−0.00387** | −0.00010 | **+0.00028** | −0.00405 (105 %) |
| 0.4 | −0.05013 | −0.00168 | −0.00034 | −0.04811 (96 %) |
| 1.6 | −0.64130 | −0.00773 | −0.01416 | −0.61940 (97 %) |

The within-reach term at amp 0.1 is **+0.00028 bpc — an eighth of the
reparameterisation noise floor**, so §5 item 7's guard applies and it is reported
as **below what this instrument resolves**, not as a gain. The decomposition sums
to the measured fresh Δ of +0.00387, which is the self-check that it is the same
quantity.

*(This subsection's decomposition is **post-hoc**: §2.6 pre-registered publishing
the curve, not cutting it this way. `EXP_012` §11.6 is the precedent for labelling
a chase that follows a result. The cut is a reconstruction of §10.3's basis, not
the identical computation, and the component weights are uniform over positions.)*

### 9.7 What the arm costs, for the record

Whatever else is true, the arm's price is now measured: **1.07–1.09× wall-clock
and +0.187 GiB**, for `K·B·L·d` Bernoulli draws per step outside the captured
region. That is cheaper than token-shift's 1.49× (`EXP_007`) and comparable to
the threshold arm's 1.12×, so `03_phase3_candidates.md` §7.1's warning that
"outside the time loop is not free" is priced here rather than assumed.

### 9.8 An operational mistake, reported rather than absorbed

**`noise_a0p1_s0` was trained twice, and the first result was discarded.** The
test suite was launched on the same GPU roughly two minutes into that run, so its
wall-clock overlapped other GPU work — the condition under which
`03_phase3_candidates.md` §8 declines to report a wall-clock at all. The run
directory was deleted and the seed retrained on an idle GPU.

**Everything scored is bit-identical between the two runs**: test bpc
2.2726652738779745 fresh and 2.25643173415706 carried, the nats, and both layers'
firing rates, all equal under `==`. Only wall-clock moved, 437.8 s → **426.9 s**
(1.098× → 1.070×), and the discarded figure was not even an outlier — the nine
clean runs span 425.3–449.3 s.

**The re-run is therefore a stronger result than the incident it corrects.** It is
an end-to-end demonstration that 20 000 steps of a *stochastic* training
intervention reproduce bit-exactly from the seed alone, which is what §1.2's
per-`(seed, step)` generator was built for and what G4 could only check one draw
at a time. `tests/test_determinism.py` guarantees fixed-weight evaluation; this
guarantees the whole trajectory, for this arm.

### 9.9 The decision rule, and why the verdict is not the cell that fired

`scripts/exp/013_noise_results.py` returns **"N1 holds"** and its text —
*"report the effect and refer adoption to Elliot"*. That is recorded, and it is
not the verdict of this experiment.

The cell fired because §3's N1 is satisfied by a uniformly worse model (§9.2).
Its own escape clause — "demands an explanation before it is believed" — was
written in advance for exactly this, and the explanation refutes it. **Read
against the substance rather than the arithmetic, the outcome is §4's second
cell:**

> **NULL, and it is reported as "there was no gap to close" — never as "noise
> injection does not regularise".**

N3 held: at amp 0.1 the arm is inside the 2σ null band. **Not recommended for
adoption**, and no adoption is made here.

**This is a rule failing, and it is reported as one.** The honest summary is that
`EXP_013` pre-registered a decision variable that could be satisfied the wrong
way, its ceiling section anticipated the direction, and its threshold section did
not encode what the ceiling section knew. That is the same failure `EXP_012`'s Y5
made — a bar written without the guard its own limitations list had already
named — occurring twice in a row in this project, and §10 refers it as a
*protocol* item rather than an experiment-local one.

### 9.10 Predictions, resolved as written

| | verdict | note |
|---|---|---|
| **N1** | **HELD** | at amps 0.4 and 1.6 — and §9.2 shows the bar is satisfiable by damage. Predicted in advance to fail; it "passed" for the wrong reason, which is worse |
| **N2** | **HELD** | monotone in amplitude, and +0.640 at the top against a 0.00922 bar |
| **N3** | **HELD** | +0.00419 at amp 0.1, inside ±0.00922 both ways |
| **N4** | **FAILED** | layer 1 monotone 0.323 → 0.658; layer 0 not |
| **N5** | **FAILED as written**, held as glossed | §9.3 — the contradiction is in the pre-registration |
| **N6** | **FAILED** | 0.36× at amp 0.1 and 3.94× at amp 1.6, outside the band in *both* directions |

**Three of six failed and a fourth is repudiated by its own experiment.** After
`EXP_012`'s three-of-six that is a consistent picture: this project's
pre-registered bars are failing often enough to be doing work.

### 9.11 Status

**CLOSED as an experiment; the arm is not adopted, not ranked and not added to
`03_phase3_candidates.md` §6.3.** Adoption, ranking and the §10 referrals are
Elliot's.

### 9.12 What this experiment does not answer

1. **Whether noise injection regularises a model that overfits.** It cannot:
   §2.2 measured that this model does not. Every statement here is conditional on
   735 K parameters over ~7.3 epochs.
2. **Whether `p` matters.** Held at 0.1 throughout; a null at one `p` is a null
   of that `p` (`EXP_009` §9.3's rule).
3. **Whether any of it survives on the adopted neuron.** Not run there.
   `EXP_005` and `EXP_012` are two demonstrations that this project's statistics
   do not transfer between those two neurons.
4. **What layer 0's non-monotone firing rate is** (§9.4).
5. **Whether an amplitude between 0.1 and 0.4 is different in kind.** The ladder
   is ×4 and the interval where the arm goes from harmless to 14.8 se harmful is
   unsampled.

---

## 10. Referred to Elliot, and not decided here

1. **N1's construction.** A gap-reduction bar must require `Δtest ≤ 0`, or it
   scores damage as success (§9.2). This is a defect in *this file's* §3 and the
   correction is referred rather than applied, per four standing precedents.
2. **The same defect as a protocol item.** `EXP_012`'s Y5 and this file's N1 are
   the same mistake twice: a bar specified without a guard the experiment's own
   limitations section had already written down. `04_phase4_interim.md` §8 has no
   row for "how bars are checked before they are committed", and on this evidence
   it may need one.
3. **Whether this arm belongs on the candidate list at all** (§0). It is
   uncovered work in a named area, and `03_phase3_candidates.md` §6.5 already
   records that area's provenance as the weakest in the report.
4. **§2.2's measurement is the transferable result and it bears on other
   candidates.** That this model does not overfit is an argument about the
   parameter-scaling pilot (§6.3 #11) and about every future regularisation
   proposal, and it was measured on the committed baseline rather than on
   anything this experiment trained.
