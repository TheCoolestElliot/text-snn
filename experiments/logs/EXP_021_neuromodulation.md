# EXP_021 — EXP_018's signal was real and its slot was wrong. Is either other slot right?

**Pre-registered:** 2026-08-18, after the derivations in §2, the `tau`
calibration in §2.5, the §7.2 reachability screen in §7 G8 and the cost
pre-flight in §3.6, and **before any arm in §3 was trained and before any bpc
existed for any of them.** **Committed before the first run of §3.**

**Status: CLOSED 2026-08-18. H1, H4, H5 and H6 held; H2 UNRESOLVED; H3 below threshold. The local modulator wins by 3.7x the bar and its control wins nearly as much.**

---

## 0. What this experiment is, and what it is not

`EXP_018` built a reward prediction error, broadcast it into every layer's input
current, and closed with three findings that this experiment is built on rather
than around:

> 1. **The optimiser finds the signal and wants it.** `rms(k) = 0.195` for the
>    aligned RPE against **0.0075** for the batch-misaligned control — a factor
>    of **26**.
> 2. **Paying attention to it costs 0.0062 bpc on 5 of 5 seeds** (t = −7.48,
>    p = 1.7e-03). It buys no extra training fit while losing at test, so the
>    train→test gap *widens* by 0.0057. A capability loss, not overfitting.
> 3. **It is head-driven, so decision #3 makes it permanently a diagnostic** —
>    the signal cannot be supplied inline and needs a second forward pass,
>    measured at **1.71x** wall-clock, at inference as well as training.

So the arm did not fail for lack of signal. **It failed because of where the
signal was injected**, and there are exactly two other slots. This experiment
tries both, separately, and neither is a re-tuning of `EXP_018`.

**What this is not:**

* **It adopts nothing and ranks nothing.** Adoption is Elliot's (report §8).
* **It changes no hyperparameter.** G5 reads the frozen Phase-2 recipe back out
  of each run's own `config.json`.
* **It does not decide #3.** Form 1 is built inside the shape decision #3
  already **admits**; form 2 is argued in §2.4 to be outside decision #3's scope
  entirely, and that argument is **referred** in §10, not relied on.
* **It does not edit any adopted arm.** `src/snn/neuron.py`, `kernels.py`,
  `surrogate.py`, `twocomp.py`, `twocomp_detach.py` untouched; G1 asserts it.
  **No new scan, no new recurrence, no hand-written backward**, so R10 is
  untouched and the 59-mutation campaign is not re-run.
* **It does not re-baseline.** Decision #10. The anchor is `EXP_020` leg 1,
  trained on this same tree in the same session; no figure here may be compared
  with a committed one.
* **It reuses `EXP_018`'s control unchanged.** `roll_across_batch` is imported,
  not re-implemented, so this arm's control and `EXP_018`'s are the same
  function and not two functions with the same name.

---

## 1. The question

> **Dopamine's best-established action is gating plasticity, not moment-to-moment
> gain, and its driving signal in cortex is local. `EXP_018` tested neither. Does
> moving the RPE to the previous layer, or off the forward pass altogether, do
> what moving it into the current did not?**

---

## 2. The derivation, done before any measurement

### 2.1 Form 1 — the same construction, driven by the previous layer

Decision #3 (ruled 2026-08-14) **admits** a modulator driven by layer `k-1`'s
output, because that activity is already computed before layer `k`'s time loop
begins, so the signal costs O(1) kernels per layer and preserves the
depth-sequential evaluation. `EXP_018` §10's own referral names a feedforward
data-dependent arm as the obvious candidate. This is that arm, built with
`EXP_018`'s derivation rather than a new one.

Layer `k-1` emits a binary vector `s_t`. Give each channel a **diagonal**
one-step predictor of its own next spike, `p_{t,c} = sigmoid(a_c s_{t-1,c} + b_c)`,
and the identical three quantities follow over the channel axis instead of the
vocabulary axis:

    S_t   = -mean_c [ s log p + (1-s) log(1-p) ]    the surprise   (nats/channel)
    H_t   =  mean_c H_bernoulli(p_{t,c})            its OWN expectation
    phi_t =  H_t - S_t                              zero-mean by construction
    DA_t  =  tanh(phi_t / tau)
    cur^k =  cur^k * (1 + kappa_k * DA_t)

**The entropy is again the right baseline and again decides the implementation.**
`E_{s~Bern(p)}[-log P(s)] = H_b(p)` exactly, so there is no EMA, no critic, no
batch statistic and nothing to tune — and the alternative, a running mean over
`t`, is a **sequential scan**, `L` kernels on the one code path this architecture
exists to keep at one kernel per timestep.

**The predictor is diagonal, and the constraint picked it.** A `[d, d]`
predictor applied to `s_{t-1}` is `W_rec s_{t-1}`, exactly the form I5 forbids.
`a_c` and `b_c` are per-channel and O(1) per neuron, which the 2026-08-01 I5
ruling explicitly admits.

**Layer 0 is unmodulated**, structurally: it has no previous layer.

### 2.2 The reduction is a MEAN, and a sum would have measured an adopted arm

Summed over channels, `phi` is O(d): measured at `d = 512` with the predictor at
its prior it is **+101 nats**, and `tanh(101/tau)` is `1.0` to fp32 for any `tau`
of order one. A saturated squash kills the gradient into `(a, b)` **and** makes
`DA` a constant, so `cur*(1 + kappa*DA)` becomes a constant per-channel gain —
which by `EXP_008` Identity 1 is exactly a learned per-channel threshold, an arm
this project has **already adopted** (#5). The saturated arm would have trained,
scored plausibly, and been measuring #5 under a new name.

Meaned, `phi` is nats per channel, O(1) in the width, so `tau` is a property of
the task rather than of `d`.

### 2.3 The form it is computed in, forced by a measurement

Written literally, `H - S` needs `log p`, `log(1-p)`, `p` and two weighted
reductions — about eight elementwise passes over `[B, L, d]` per layer plus their
backward. At 735K parameters this model is memory-bandwidth-bound, and the
pre-flight measured that literal form at **2.01x the anchor's wall-clock**:
**worse than the 1.71x of the head-driven arm this experiment exists to beat**,
which would have destroyed the arm's entire cost argument before it ran.

The two terms share their `log(1-p)`, and it cancels. With `log p = log(1-p) + logit`:

    S = -[log(1-p) + s*logit],   H = -[log(1-p) + p*logit]
    phi = H - S = (s - p) * logit                              EXACTLY

so `phi_t = mean_c (s_{t,c} - p_{t,c}) * logit_{t,c}` — four passes, no
logarithm at all, and the general exponential-family identity
`H - S = (s - E[s]) . eta` written for a Bernoulli. Re-measured: **1.49x**.
It also reads as what it is: **the RPE is the prediction error projected onto
the predictor's own confidence.** A channel the predictor is unsure about
(`logit ~ 0`) contributes nothing however wrong it turns out to be.

The rewrite is verified against an independently written literal `H - S`:
max abs difference **1.6e-07**, max relative **5.1e-07**, and the calibrated
`tau` is **unchanged to all reported digits** (0.057321 before and after).

### 2.4 Form 2 — dopamine on the learning rule, not on the forward pass

`EXP_018` put a neuromodulator on the forward current. In the brain the
best-established action of dopamine is a **third factor gating plasticity**,
`dw ~ pre x post x DA`. Nothing in this project has tested that.

    S_t   = -log p_t(y_t)                    the per-position cross-entropy
    H_t   =  H(p_t)                          the model's OWN expected surprise
    phi_t =  H_t - S_t
    w_t   =  1 + kappa * tanh(phi_t / tau)                       DETACHED
    loss  =  sum_t w_t S_t / sum_t w_t

Under BPTT the per-position gradient **is** the eligibility trace, so weighting
the per-position loss is the three-factor rule written in the only form this
trainer can express. `kappa < 0` up-weights positions that went **worse** than
the model expected; `kappa > 0` up-weights positions that went **better**. Both
are real dopaminergic stories, neither is derivable from the other, and both are
arms — exactly as `EXP_018` pre-registered `mult` and `add`.

**ALIGNMENT, which is the one thing here that must not be wrong.**
`snn.dopamine.rpe` deliberately builds `phi_u` from `logits[:, u-1]` and
`idx[:, u]`, because that signal is consumed **inside the forward pass at
position u** and must not contain the target. This one is consumed **by the loss
at position t**, whose target is `y_t` by definition, so the RPE describing that
prediction is built from `logits[:, t]` and `y_t` — **no shift**. Using
`snn.dopamine.rpe`'s indexing here would weight position `t` by the error made
at `t-1`; it would still train, still look plausible, and be a different arm.
The two are the same algebra deliberately aligned two ways, which is why they
are separate functions and not a shared helper.

**There is no leak.** `w` is detached, so nothing about `y` reaches a parameter
except through `S_t`, which is the objective.

**Four properties follow, and they are why this is the sharper test:**

* **The forward pass is not touched at all.** At eval the model is the Phase-2
  baseline **bitwise, by code path** — so unlike `EXP_018` this arm
  *structurally cannot* cost capability at inference. Whatever it does, it does
  to the optimiser.
* **There is no inference cost.** The weighting exists only inside
  `Trainer._step_body`.
* **The function class is unchanged**, so this is arguably not a decision-#3
  question at all — it is a training objective, and this project already ships
  one: `snnchat.train` weights `<|bot|>` turns by `bot_loss_weight` with the
  identical `(per_char * w).sum() / w.sum()` normalisation. **That argument is
  referred to Elliot in §10 and is not relied on by any bar.**
* **`w` is DETACHED and that is load-bearing.** Undetached, the model could lower
  its loss by manipulating the weights rather than by predicting better — and it
  would, because `w` is a function of its own logits.

`sum_t w_t` in the denominator rather than `L`, so the arm cannot change the
effective learning rate as a side effect and be a learning-rate change wearing a
neuromodulator's clothes.

### 2.5 `tau` is measured, and the measurement rejected the first design

`scripts/exp/021_calibrate_nm.py`, run before this file existed, over four
committed checkpoints spanning two architectures, 256 val windows each.

Calibrating at the arm's own init prior (`b = -2.97 = logit(0.0488)`, layer 0's
rate **at initialisation**) returned `phi mean = -0.879 to -1.018` with
**100.0 %** of values negative — because those checkpoints have converged to
rates of 0.345–0.392. A `tau` fitted to that is fitted to a constant offset, and
§2.2 says what a constant offset makes this arm.

Calibrated instead at the predictor's **fixed point**, `b_c = logit(r_c)` with
`r_c` the measured per-channel rate:

| checkpoint | phi mean | phi sd | frac positive | frac beyond 2 sd |
|---|---:|---:|---:|---:|
| `snn_beta0.5_s0` | +0.00000 | 0.06752 | 0.421 | 0.026 |
| `snn_beta0.5_s1` | +0.00000 | 0.05284 | 0.417 | 0.048 |
| `snn_beta0.5_s2` | +0.00000 | 0.06180 | 0.413 | 0.046 |
| `twocomp_s0` | −0.00000 | 0.04640 | 0.513 | 0.030 |

**`nm_tau = 0.057321`**, the median sd. **C1 passed at exactly 0.000000**, which
is the derivation checking out rather than the data cooperating: with `p_c = r_c`,
`E[S] = mean_c H_b(r_c) = E[H]` identically.

**The relative spread is 36.8 %, against `EXP_018`'s 3.1 %, and that is recorded
as a limitation rather than smoothed.** This scale is **less** transferable than
the head-driven one, so §6.2 states it and §2.6 is the design response.

### 2.6 `tau` is initialised from that calibration and then LEARNED

One scalar per modulated layer, log-parameterised. This departs from `EXP_018`,
which froze `tau`, and the reason is a property of the signal rather than a
preference: `EXP_018`'s surprise is over a 205-way vocabulary whose scale is
near-stationary, while this one is over a population whose firing rate rises
**7x** during training — 0.049 at init (`audit_07`) to 0.34 at convergence
(`EXP_000` F3). A frozen `tau` right at step 0 is wrong at step 20,000, and §2.2
says a saturated `tanh` is the one failure this arm cannot survive.

Measured at init with the arm's own scalar `b = -2.97`: `phi` mean −0.0044,
sd 0.0446, **mean `sech^2(phi/tau) = 0.777`** at `tau = 0.057321` — unsaturated,
so the arm starts where it can learn.

### 2.7 The saddle, derived before the screen was run

`dL/dkappa_c` is nonzero at `kappa = 0` because the factor `(1 + kappa DA)` is
then 1 and every downstream adjoint is the baseline's. But the predictor reaches
the loss **only** through that factor: `dL/da = (dL/dDA)(dDA/dphi)(dphi/da)` and
`dL/dDA` is proportional to `kappa`. **At `kappa = 0`, `nm_a`, `nm_b` and
`nm_log_tau` are exactly zero.** That is `EXP_004` §2.2's pattern, and it takes
`EXP_004`'s remedy: a **pre-registered nonzero fallback**, `nm_gain_init = 0.1`,
which is what the arms below use.

The §7.2 screen confirmed the derivation exactly — `nm_g0` **SADDLE** on exactly
the three derived parameters, `nm_g01` clear of zero on all four — with all three
synthetic controls holding, so the screen is demonstrably able to fail.

**One thing the screen returned that was NOT predicted in advance, labelled
POST-HOC:** at the fallback init `nm_a` and `nm_b` sit at **2.6e-09** and
**5.3e-09** against `|dL/dW|` of 6.5e-05 — ratio ~4e-05, well under the screen's
1/100 floor, so `nm_g01` reports **FAINT**. The explanation is structural and is
offered as such rather than as a prediction: `phi` is a **mean** over `d`
channels (§2.2), so `dphi/da_c` carries a `1/d` factor, and `d = 512`. A faint
ratio is not fatal — **Adam is scale-free per parameter**, which is the reading
`EXP_018`'s `da_mult` entry already applies to its own 46x — and an exactly zero
one would be. **It is a risk this experiment carries and names, not one it
resolves**, and H3 is the bar that would detect it having mattered.

---

## 3. The legs

All at `V = 205`, `d = 512`, `K = 2`, enwik8, 20,000 steps, seeds 0/1/2, one
variable per leg against a named reference, driven by
`scripts/exp/021_run_neuromod_arms.py`.

**The anchor is `EXP_020` leg 1** (`if_anchor_s0/1/2`), trained on this tree in
this session. It is not re-run.

### Leg 1 — `nm_local` (3 runs, ~30 GPU-min)
`arch="localdopamine"`, `nm_source="local"`, `nm_tau=0.057321`,
`nm_gain_init=0.1`. 736,974 params (+0.21 % over the anchor: `(3d+1)(K-1)`).

### Leg 2 — `nm_rolled` (3 runs, ~30 GPU-min)  **the control**
Identical but `nm_source="rolled"`. The roll is along the **batch** axis, which
preserves the marginal distribution and the temporal autocorrelation **exactly**
and destroys only the alignment — and cannot leak the future, because it moves
nothing along time.

### Leg 3 — `tf_neg` (3 runs, ~21 GPU-min)
`arch="snn"`, `tf_kappa = -0.5`. Learn more from what went worse than expected.

### Leg 4 — `tf_pos` (3 runs, ~21 GPU-min)
`arch="snn"`, `tf_kappa = +0.5`. Learn more from what went better.

### Leg 5 — `tf_roll` (3 runs, ~21 GPU-min)  **the control**
`tf_kappa = -0.5`, `tf_rolled = True`.

**`kappa = 0.5` is not swept.** It is the value at which the extreme weight ratio
is exactly `1.5/0.5 = 3.0`, and `|kappa| < 1` is enforced at construction so a
weight can never reach zero and delete a position from the gradient.

### 3.6 Cost, measured before this file was committed

Marginal ms/step from a 400-vs-1600-step difference:

| arm | ms/step | 20k | x anchor |
|---|---:|---:|---:|
| anchor (`EXP_020` leg 1) | 18.38 | 368 s | 1.00 |
| `nm_local`, literal `H - S` | 36.90 | 738 s | **2.01** — rejected, §2.3 |
| `nm_local`, cancelled form | 27.41 | 548 s | **1.49** |
| `tf_*` | 19.09 | 382 s | **1.04** |

**Budgeted: ~2.3 GPU-hours, x1.0.** 15 training runs at 7,326 s plus 15 test
evaluations. Combined with `EXP_020`'s ~2.1, this session spends **~4.4 of the
~14.2 GPU-hours remaining** in Phase 4's ~30.

---

## 4. Pre-registered predictions

### 4.0 What this design can and cannot resolve, stated before the run

**n = 3 per cell**, sigma **transferred**: 0.00461, the whole-split baseline sd
over 5 seeds, measured on a different tree. The se of a difference of two 3-seed
means is 0.00376, so the 2-sigma bar of 0.00922 is **2.45 se**. `EXP_020` leg 1
re-measures sigma on this tree and §9 reports whether the transfer held.

**A miss resolves UNRESOLVED, never FAILED.** A remedy is seeds **added**, never
substituted. **A diverged run resolves DIVERGED / NO RESULT.**

**`EXP_018` D1's defect is not repeated.** It used an unpaired Welch test on
paired data and resolved UNRESOLVED by 3.6e-06 bpc; the corrected paired test —
run *because it hurt* — found the arm costs 0.0062 on 5/5 seeds. Every bar below
is on a **paired per-seed difference**, because a seed fixes both the initial
weights and the data order, and §9 reports the per-seed signs alongside the mean.

---

### T1 — the instrument  *(abort-on-fail)*
> `EXP_020` T1 passed, so the anchor these arms are read against is the
> committed Phase-2 baseline and not an arm.

### T2 — the gates  *(abort-on-fail)*
> G1, G2 and G8 hold as written in §7 before leg 1 starts.

---

### H1 — the local modulator  *(difference, BOTH terms constrained; one-sided)*
> `mean bpc(nm_local) < mean bpc(anchor) − 0.00922`.

**Mechanism, stated so a miss is informative.** `DA_t` multiplies the current, so
the arm computes a product of the previous layer's *population summary* with
layer `k`'s current — a second-order interaction that no GEMM in the stack can
express, and a strictly larger function class than `EXP_008`'s constant
per-channel gain (which bought 0.0590 on this architecture).

**Failure mode:** `EXP_018` found a time-varying gain **costs** 0.0062, so this
bar is predicting the opposite sign from the nearest precedent, on the argument
that the slot rather than the signal was the problem. If it misses in the
direction of harm, that argument is wrong and §9 says so.
**Guards:** (a) both terms are 3-seed means from this session's tree; (b) paired
per-seed differences reported, not only the mean; (c) H2 must also hold for a hit
to mean anything — §5 makes that explicit.

### H2 — is it a mechanism or a perturbation?  *(difference, BOTH terms constrained; one-sided)*
> `mean bpc(nm_local) < mean bpc(nm_rolled) − 0.00922`.

**This is the bar that matters most**, and it is the one `EXP_018` had. Without
it, a gain from `nm_local` is indistinguishable from a gain from *any* bounded
time-varying multiplicative perturbation with that marginal.

**Failure mode:** both arms gaining equally, which would say the benefit is
stochastic-gain regularisation. `EXP_013` measured noise injection as a null at
every amplitude, so that outcome would also contradict `EXP_013`, and §9 must
report the tension rather than pick a side.

### H3 — is the signal findable?  *(MARKER with a threshold, no verdict)*
> `rms(nm_gain)` for `nm_local` is at least **3x** that for `nm_rolled`.

`EXP_018`'s reusable result was a factor of **26** on exactly this quantity. 3x
is deliberately far below it: this arm's `phi` is a different signal with a
different scale, and §2.7's FAINT ratio is a named risk that would show up
**here** first. A ratio near 1 means the optimiser cannot tell the aligned signal
from the misaligned one, which would explain a null on H1/H2 without anything
else needing to be true.

### H4 — the three-factor rule on the mean  *(difference, BOTH terms constrained; one-sided)*
> `mean bpc(tf_neg)` is **NOT** below `mean bpc(anchor) − 0.00922`.

**A prediction of no improvement, made in advance and on an argument.** bpc is
the *mean* of `S`; reweighting the training objective away from that mean means
no longer minimising the metric reported, so the arm can only win if the
optimisation benefit exceeds the objective mismatch. `EXP_013` established this
model is **underfitting** (gap +0.00594 fresh), which cuts both ways: there is
headroom, and there is no overfitting for a hard-example emphasis to fix.

**Failure mode:** a bar that predicts a null is unfalsifiable if written
loosely. This one is not: it is failed by any improvement exceeding 2 sigma, and
§5 records that outcome as the interesting one.

### H5 — the three-factor alignment control  *(difference, BOTH terms constrained; two-sided)*
> `|mean bpc(tf_neg) − mean bpc(tf_roll)| > 0.00922`.

If the aligned and misaligned weightings are indistinguishable, the arm is a
bounded random reweighting of the loss and its RPE is decorative.

### H6 — the sign is the scientific content  *(difference, BOTH terms constrained; one-sided)*
> `mean bpc(tf_neg) < mean bpc(tf_pos) − 0.00922`.

Learning more from what went **worse** beats consolidating what went better, on
the mechanism that an underfitting model has more to gain on positions it has
not learned than on positions it has.

### H7 — where the effect is  *(MARKER, no verdict)*
> The train→test gap per arm, against the anchor's.

`EXP_018`'s clearest structural finding was that its arm bought **no extra
training fit** while losing at test — gap **+0.0057** — which is what made it a
capability loss rather than overfitting. The same quantity is reported here for
all five arms. A three-factor arm that *narrows* the gap is doing something
different from one that widens it, and neither is visible in the mean alone.

### H8 — cost  *(MARKER, no verdict)*
> Realised wall-clock ratio and peak VRAM per arm.

---

## 5. Decision rule, fixed now

| cell | verdict |
|---|---|
| H1 hit, H2 hit | "a previous-layer-driven RPE pays, and its alignment is load-bearing" — the result this experiment was built for. **Referred, adopted by nobody here.** |
| H1 hit, H2 miss | "a bounded time-varying gain pays and the RPE is not why" — recorded as such, and the tension with `EXP_013`'s null stated |
| H1 miss, H2 hit | alignment matters but not enough to clear the bar — UNRESOLVED on H1, and H3 is quoted as the reason to believe the signal was found at all |
| H1 miss, H2 miss, H3 near 1 | "the optimiser could not distinguish the aligned signal from its control" — the §2.7 FAINT risk realised, and §9 says so plainly |
| H1 miss, H2 miss, H3 large | "the signal is findable and the previous-layer slot does not pay either" — with `EXP_018`, that is **two of the three slots ruled out on this architecture** |
| H4 holds (no improvement) | "the three-factor rule does not improve mean bpc", quoted with the realised sign and H7's gap |
| H4 fails (improvement > 2 sigma) | the most surprising outcome in this file, and §9 leads with it; H5 then decides whether it is the RPE or the reweighting |
| H5 miss | the three-factor arms are a bounded random reweighting; H4 and H6 are reported but carry no mechanism |
| H6 hit / miss | the sign of dopaminergic plasticity gating on this task, recorded either way |
| any leg diverges | **DIVERGED / NO RESULT**; n reduced, verdict provisional, **no seed substituted** |

**No verdict in this table adopts anything, changes any hyperparameter, decides
#3, #4 or #11, or authorises Phase 5.**

---

## 6. Known limitations, stated in advance

1. **n = 3, sigma transferred from a different tree.** §4.0.
2. **`nm_tau`'s relative spread is 36.8 %**, against `EXP_018`'s 3.1 % for the
   analogous constant. This scale is **not** established as a property of the
   task, and §2.6's learned `tau` is a design response to that, not evidence
   against it.
3. **`kappa`, `nm_gain_init`, `nm_a_init` and `nm_b_init` are not swept**, and
   `nm_gain_init = 0.1` is `EXP_004`'s fallback value reused rather than
   re-derived for this arm.
4. **§2.7's FAINT ratio is unresolved.** `nm_a` and `nm_b` may be effectively
   frozen at the observed 4e-05 ratio; H3 would detect it having mattered, and
   nothing here would distinguish "the predictor did not learn" from "the
   predictor learned and the signal does not pay".
5. **Form 2's decision-#3 status is argued, not ruled.** §10.
6. **One width, one depth, one corpus, plain LIF only.** Nothing transfers to
   the adopted two-compartment arm or to `snnchat`.
7. **The two forms are never composed.** Running both at once would compose two
   arms while claiming to measure one; `Config` permits it and no leg here does
   it.
8. **H7 is a marker on one number per arm**, not the per-context decomposition
   `EXP_004` §10.3 established as this project's standard. That decomposition
   costs a horizon sweep per arm and is not budgeted here.

---

## 7. Gates — aborting, not advisory

* **G1** — `git diff` empty over `src/snn/neuron.py`, `kernels.py`,
  `surrogate.py`, `twocomp.py`, `twocomp_detach.py`. **R10 untouched, mutation
  campaign not re-run.**
* **G2** — realised parameter counts equal the closed form **exactly**. The
  `tf_*` arms are `arch="snn"` and must be **735,437 — identical to the anchor.**
* **G3 / K1** — one field changed per arm against its named reference.
* **G4** — `tf_kappa = 0.0` returns `F.cross_entropy` by **code path**, asserted
  bitwise in `tests/test_neuromod.py`, so every non-`tf` run in this session used
  the unchanged objective.
* **G5** — the frozen recipe read back out of each run's own `config.json`.
* **G6** — VRAM alarm at 4.0 GiB.
* **G7** — divergence read from `log.jsonl`, not the exit code; an absent log
  reads as diverged.
* **G8** — the §7.2 screen, green before the first training step:
  `nm_g0` SADDLE **exactly as derived**, `nm_g01` clear of zero on all four
  parameters, all three synthetic controls holding. Artifact:
  `docs/reports/data/exp_020_021_reachability.json`.
* **G9** — the mutation-campaign lockfile absent before every run.

---

## 8. Entry conditions

* This file committed **before the first run**, SHA-256 stamped into
  `exp_021_run_manifest.json` before and after.
* `docs/reports/data/exp_021_nm_calibration.json` exists and C1 passed — `tau` is
  a measured constant and the ladder does not start without it.
* `docs/reports/data/exp_020_021_reachability.json` exists and is green.
* `EXP_020` leg 1 complete: the anchor exists and T1 passed.
* `tests/test_neuromod.py` green, `ruff` clean.
* The resolver `scripts/exp/021_neuromod_results.py` committed **before any
  leg's bpc is read**.
* GPU idle; nothing else runs while any leg is timed.

---

## 9. Results

*(appended after the run — nothing above this line is rewritten)*

**Closed 2026-08-18, ~2.0 GPU-hours** (15 training runs at 7,125 s, plus the
`tau` calibration, the §7.2 screen, the cost pre-flight, a 18-checkpoint horizon
sweep and 12 generalisation-gap evaluations). Pre-registration SHA-256 stamped
into `exp_021_run_manifest.json` before the first run and re-checked after the
last: **unchanged**. All 15 runs completed, **none diverged**, G2 exact on 15/15,
no K1 or G5 violation, no VRAM alarm.

### 9.1 The scoreboard

| bar | verdict | one line |
|---|---|---|
| **H1** the local modulator | **HELD** | `nm_local` beats the anchor by **−0.03371** bpc, paired t = **−8.95**, **3/3** seeds, 3.7× the bar |
| **H2** mechanism or perturbation | **UNRESOLVED** | −0.00577 against `nm_rolled`, t = −3.22, 3/3 seeds — below the 0.00922 bar |
| **H3** is the signal findable | **BELOW threshold** | `rms(nm_gain)` **1.077 aligned / 0.958 rolled — ratio 1.12**, against a 3.0 threshold and `EXP_018`'s 26 |
| **H4** three-factor on the mean | **HELD** (it predicted no improvement) | `tf_neg` is **+0.04826** *worse*, t = +29.34, 0/3 seeds better |
| **H5** three-factor alignment | **HELD** | \|`tf_neg` − `tf_roll`\| = **0.04736** > bar, t = 19.35 |
| **H6** the sign | **HELD** | `tf_neg` beats `tf_pos` by **−0.03770**, t = −30.26, 3/3 seeds |
| **H7** train→test gap | measured, §9.5 — **and the pre-registered statistic was invalid for two arms** | |
| **H8** cost | measured, §9.6 | |

### 9.2 The headline, and the sentence §5 fixed for it in advance

`nm_local` is **the first arm in this session to beat its anchor**, and it is in
the slot decision #3 explicitly **admits** as adoptable: driven by layer `k-1`'s
own emission, no second forward pass. Per-seed differences −0.04042, −0.02739,
−0.03331 — all three the same sign, none marginal.

But H2 did not hold and H3 came in at **1.12**, so §5's cell is

> **H1 hit, H2 miss — "a bounded time-varying gain pays and the RPE is not why"**

and that is the verdict, written before the run and not softened now. The
optimiser grew `kappa` from its 0.1 init to **rms ≈ 1.0 for the aligned signal
and ≈ 0.96 for the batch-rolled one**. It wants a large modulation, and it cannot
tell the two apart. `EXP_018`'s factor of 26 on the identical quantity does not
reproduce here, and §2.7 named the FAINT reachability ratio in advance as the
risk that would surface exactly here.

**What is established** is that a bounded, learned, **rank-1 time-varying
multiplicative gain on the input current** is worth ~0.028–0.034 bpc on this
architecture. **What is not established** is that the reward prediction error is
why: the alignment adds at most 0.0058, which this design cannot resolve.

### 9.3 The tension with `EXP_013`, stated rather than reconciled

`EXP_013` measured injected noise as a **null at every amplitude** (+0.00419 /
+0.04978 / +0.63975). Here a signal the model cannot distinguish from a
misaligned one is worth −0.028. Those are not the same intervention and the
differences are structural, not a matter of degree:

| | `EXP_013` noise | this arm |
|---|---|---|
| form | **additive** | **multiplicative** |
| structure | i.i.d. over (batch, time, channel) | **rank-1**: one scalar per (batch, time), shared across channels |
| amplitude | a swept constant | **learned per channel** (`kappa`, grown 10× from init) |
| source | an RNG | an activation the forward pass already produced |

No claim is made here about which difference matters. **The tension is recorded
because a reader is entitled to notice it**, and separating those four factors is
four arms, not one.

### 9.4 The decomposition, and it locates the gain precisely

`EXP_004` §10.3's split, cut at the baseline horizon of 7, positive = better.
`EXP_020` §9.5 measured this statistic's noise floor with a **bitwise copy of the
anchor**: −0.00499 total, −0.01395 at c = 0, +0.00944 within reach. Read every
row against that.

| arm | total | c = 0 | within reach | beyond horizon |
|---|---:|---:|---:|---:|
| `nm_local` | **+0.03353** | −0.02183 | **+0.05588** | −0.00052 |
| `nm_rolled` | +0.02737 | **−0.08103** | **+0.10955** | −0.00115 |
| `tf_neg` | −0.04765 | −0.02909 | −0.01824 | −0.00032 |
| `tf_pos` | −0.08712 | **−0.21159** | +0.12552 | −0.00105 |
| `tf_roll` | −0.00084 | +0.00983 | −0.00925 | −0.00143 |

**`nm_local`'s entire gain is WITHIN REACH.** +0.0559 inside the baseline's own
7-character horizon, −0.0218 *paid* at zero context, and **−0.0005 beyond the
horizon — nothing at all.** The arm does not extend reach; it improves the use of
context the model already had.

That matters more than the headline. `EXP_015` §14.3 located the entire
matched-size gap to the GRU anchor in the model's failure to *use* context —
"0.051 apart at zero context, 0.408 apart by c = 64" — and every arm in §7.1's
ranking targets bits. **This is a within-reach gain, which is the component
`EXP_004` §10.3 found the two-compartment neuron also took most of.** Whether
that makes it complementary to the adopted arm or redundant with it is **not
measured here** — the composition is `EXP_018` §10 item 4's referral and is not
run.

**Both `nm_local` and `nm_rolled` trade zero-context capacity for within-reach
use, and the aligned signal trades less of it.** `nm_rolled` pays −0.081 at
c = 0 to gain +0.110; `nm_local` pays −0.022 to gain +0.056. The totals are close
and the *shapes* are not, which is the one place in this experiment where
alignment visibly does something. It is a marker: no bar was pre-registered on
the decomposition's components and none is invented now.

`tf_pos` costs **−0.2116 at zero context** — by far the largest single component
in this experiment. Up-weighting positions the model already predicts well
down-weights exactly the surprising ones, and at `c = 0` every position is
surprising.

### 9.5 H7 fired on an invalid statistic, and the referral that predicted it

H7 was pre-registered as the train→test gap read from the last logged training
loss. **For the `tf_*` arms that quantity is the WEIGHTED objective**, not a
train bpc, so it returned −0.354 and +0.673 — numbers that mean nothing.

`EXP_018` §10 item 11 refers exactly this: *"a running training loss is not a
stand-in for a train-split bpc, and any future experiment that wants a train-side
number should score a slice with `snn.evaluate.evaluate` rather than read
`log.jsonl`."* **This experiment did the thing that referral warns against, and
it produced the failure the referral predicts.** Reported as it fired.

Re-measured with the committed instrument (`scripts/exp/013_generalisation_gap.py`,
4 full-split evaluations per run, 19,456 windows each):

| arm | gap, fresh | gap, carried |
|---|---:|---:|
| `if_anchor` | +0.00472 ± 0.00104 | −0.00388 ± 0.00156 |
| `nm_local` | +0.00835 ± 0.00220 | −0.00105 ± 0.00226 |
| `nm_rolled` | +0.01100 ± 0.00030 | +0.00151 ± 0.00124 |
| `tf_neg` | −0.00283 ± 0.00173 | −0.01004 ± 0.00182 |

**`nm_local`'s gain is not a generalisation effect.** Its gap is within ~0.004 of
the anchor's on both protocols, so the arm is better on held-out text because it
is better, not because it overfits less. That is the **opposite** of `EXP_018`'s
head-driven arm, whose gap widened by 0.0057 while it lost — and it is the
cleanest evidence in this experiment that the forward slot and the head slot are
genuinely different places to put the same signal.

### 9.6 Cost

Wall-clock against `if_anchor_s0`: `nm_local` **1.49×**, `nm_rolled` 1.49×,
`tf_neg` **1.05×**, `tf_pos` 1.04×, `tf_roll` 1.02×.

`nm_local`'s 1.49× is **below `EXP_018`'s 1.71×** and, unlike it, is a
**training-only** cost in the sense that matters: there is no second forward
pass, so the arm's inference model is one pass, not two. §2.3 records that the
literal `H − S` form measured **2.01×** and that the exact cancellation
`phi = (s − p)·logit` brought it to 1.49× before a single run — the pre-flight
paying for itself.

The three-factor arms cost **1.02–1.05×** and are **bitwise the Phase-2 baseline
at inference by code path**, which is what §2.4 claimed for them. That claim
survives; the arm does not.

### 9.7 Gates

| gate | result |
|---|---|
| G1 adopted arms untouched | **PASS** — no new kernel, no hand-written backward; R10 untouched |
| G2 realised = closed form, exactly | **PASS**, 15/15 — including 736,974 for the modulated arm, which a defect found before the ladder had as 736,973 |
| G3 / K1 one thing changed | **PASS**, no unexpected field |
| G4 `tf_kappa = 0` nests by code path | **PASS**, asserted bitwise in `tests/test_neuromod.py` |
| G5 frozen recipe | **PASS** |
| G6 VRAM alarm | **PASS**, 0 alarms |
| G7 divergence from the log | **PASS**, 0 diverged, 15/15 complete |
| G8 §7.2 screen | **PASS** before the first step; `nm_g0` SADDLE exactly as derived, `nm_g01` clear on all four |
| G9 mutation lockfile | **PASS** |

### 9.8 Cost, and the project total

~2.0 GPU-hours against a budgeted ~2.3. **Nothing bought no number.** Project
total: **~19.7 of ~30 GPU-hours.**

### 9.9 The pre-registration guarantee

Nothing above §9 was rewritten. Every bar is reported as it fired, including H7's
invalid statistic and H3's failure to reproduce `EXP_018`'s factor of 26.

---

## 11. Referred to Elliot, and not decided here

*(added after the run; created by §9 and marked as such)*

1. **Whether `nm_local` is adopted.** It is the shape decision #3 admits, it wins
   by 3.7× the bar on 3/3 seeds, its gain is not a generalisation artefact, and
   it costs 1.49×. **Against it:** H2 is unresolved and H3 is 1.12, so the
   *mechanism* claimed in its name is not established. Adopting it would be
   adopting "a learned rank-1 time-varying gain", not "a dopamine signal", and
   §9.2 says so.
2. **The measurement that would resolve H2**, named with its price: `nm_rolled`
   at n = 5–7 seeds. The effect is −0.0058 with a paired se of 0.0018, so ~6
   seeds per cell would put 2 se below the bar. **~0.6 GPU-h.** Not run here
   because n = 3 was pre-registered and `CONTRIBUTING.md` §4 forbids adding seeds
   to a cell after seeing its result *within* the same experiment.
3. **Which of the four structural differences from `EXP_013` matters** (§9.3):
   additive vs multiplicative, i.i.d. vs rank-1, swept vs learned amplitude, RNG
   vs activation. Four arms, ~2.4 GPU-h, and none is designed here.
4. **Whether the composition onto `twocomp_detach` is worth running** —
   `EXP_018` §10 item 4's standing referral, now with a reason: both this arm and
   the two-compartment neuron take their gain **within reach**, so they may be
   redundant rather than complementary, and that is cheap to find out.
5. **Whether `EXP_018`'s conclusion should be restated.** Its finding was that
   the RPE is findable and does not pay. This experiment finds the same signal in
   a different slot pays −0.034 — but also that a misaligned control pays
   −0.027. So the correct restatement may be neither about slots nor about
   signals but about **modulation structure**, and that is not this experiment's
   to make.
6. **The `EXP_018` §10 item 11 referral should be ratified into
   `CONTRIBUTING.md`.** §9.5 is the second experiment to be caught by it and the
   first to be caught *after* it was written down.

---

## 10. Referred to Elliot, and not decided here

1. **Whether form 2 is inside decision #3 at all.** §2.4 argues it is not: the
   weighting is computed *after* the forward pass from logits already produced,
   needs no second pass, and changes no function class — so the structural cost
   decision #3 draws its line on does not arise. **The argument is offered, not
   applied**, and no bar in this file depends on it. If Elliot rules the other
   way, legs 3–5 become labelled diagnostics exactly as `EXP_018` is, and none
   of their numbers change.
2. **Whether either form is adopted**, on any outcome.
3. **Whether a learned `tau` is admissible where `EXP_018` froze one.** §2.6
   gives the reason; it is a departure from a committed precedent and is flagged
   as one rather than absorbed.
4. **Whether `EXP_018`'s conclusion should be restated** if H1 and H2 both hold.
   Its finding would then be about a slot rather than about a signal, which is a
   change to how a closed experiment reads and is not this experiment's to make.
