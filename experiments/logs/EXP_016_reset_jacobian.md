# EXP_016 — Why the two-compartment neuron cannot be trained at width

**Pre-registered:** 2026-08-11, after the derivation in §2 and **before any probe
was run against any checkpoint**. **Committed before the first run.**

**Status: PRE-REGISTERED — no results yet.**

---

## 0. What this experiment is, and what it is not

`EXP_015` found that both two-compartment arms **fail to train at 5.0M
parameters** under the frozen Phase-2 recipe, deterministically, at steps 5138
and ~2000, while the plain LIF and the GRU at the identical width, seed, recipe
and tree trained cleanly. That result blocks Phase 5 as conceived and is open
decision **#11**.

`EXP_015` §10 referred five things. This experiment takes **referral 4** — the
divergence itself — and takes it because `CONTRIBUTING.md` §4 makes it a standing
obligation rather than a ranked candidate:

> **Chase a NaN to its origin.** "One seed diverged" is a tolerance widened in
> prose.

**It deliberately does not take referral 1.** The recipe question is Elliot's,
`EXP_015` §10 says so explicitly ("it changes a frozen hyperparameter either
way"), and **nothing here changes, proposes or probes a hyperparameter value.**

**What this is not:**

* **It adopts nothing, ranks nothing and fixes nothing.** No arm is modified. No
  file under `src/snn/` is edited by this experiment. `adopts_nothing`,
  `ranks_nothing` and `changes_no_hyperparameter` are carried as structural
  fields in the resolver's artifact, not as prose.
* **It does not decide #11, and it is not evidence *for* a recipe change.** It
  supplies the cause that #11 should be ruled against. §2.4 states in advance the
  one consequence the derivation carries for #11, so that consequence is on the
  record as *derived* rather than discovered convenient later — and it is
  **referred**, not applied.
* **It trains nothing.** Every leg of §3 is a forward pass over committed
  checkpoints that already exist on disk. See §6 for the cost.
* **It produces no bpc, no σ and no seeds.** Every quantity below is a **marker**
  measured at n = 1 checkpoint per cell. §4.0 states what that forbids.
* **It is not a fix.** Naming an unbounded quantity is not bounding it. Any
  remedy is a new arm, needs its own pre-registration, and is out of scope.

---

## 1. The question

`EXP_015` §14.5 (report) and §9.7 (log) closed the chase at this boundary:

> "Always layer 0 and the embedding; never layer 1." … "**and not a gradient
> explosion** — the largest gradient in the six steps before death is 2.0e-02 and
> the clip never fires in the whole run."

Both sentences are about quantities measured **between** optimiser steps. Neither
constrains what happens **inside** the single failing backward. The question this
experiment asks is the one no instrument in this project has yet been pointed at:

> **Is the two-compartment neuron's backward-through-time a contraction?**

`EXP_016` is stated as a **derivation with a measured premise**, not as a
hypothesis search. §2 derives a closed-form answer for both neurons before any
GPU is touched; §3 measures the one quantity the derivation leaves open.

---

## 2. The derivation, done before any measurement

### 2.1 The per-step chain factor

Both neurons thread an adjoint backwards over all `L = 256` timesteps of the
unroll (`src/snn/twocomp.py:423-428`, `src/snn/kernels.py:146`). In both, the
fast/only membrane's adjoint is carried by

```
grad_v_prev = beta * (grad_v_next * dv + grad_spike * sg)
```

so **`beta * dv` is the per-step multiplier of the homogeneous part of the
backward recurrence.** Over `L` steps the homogeneous gain of one `(batch,
channel)` chain is `prod_t |beta * dv_t|`.

### 2.2 The plain LIF is a strict contraction, at any width

`src/snn/kernels.py:129`, hard reset:

```
dv = (1 - s) - v_pre * sg      with   x = v_pre - thr,  s = H(x),  sg = atan_grad(x, alpha)
```

**The same variable `v_pre` appears in the multiplier and inside the surrogate.**
That makes the product self-limiting: `sg` decays as `1/(1 + (pi/2 * alpha * x)^2)`
exactly as fast as `v_pre` grows. At this project's frozen constants
(`alpha = 2.0`, `thr = 1.0`, `beta = 0.5`; `src/snn/config.py:86,87,90`),

> `sg(x) = 1 / (1 + (pi x)^2)`, `v_pre = x + 1`, so `|v_pre * sg|` has extrema at
> `x* = -1 ± sqrt(1 + 1/pi^2)`, giving
>
> **`max_x |dv| = 1.0247193`  and  `max_x |beta * dv| = 0.5123596 < 1`.**

This is a **theorem, not a measurement**: it holds for every finite `v_pre`,
every batch, every channel, every step, and — critically — **at every width**,
because no term in it depends on `d_model`. The plain LIF's BPTT is a strict
contraction by construction. **This is the first available explanation of why
`scale_d1481_s0` trains 20,000 steps cleanly**, and it predicts the plain LIF
would train at any width the hardware allows.

### 2.3 The two-compartment neuron's is not bounded at all

`src/snn/twocomp.py:179-180`:

```
vf = v_pre - w_c * vs
dv = (1 - sh) - vf * sgd       with   x = v_pre - thr,  sgd = atan_grad(x, alpha)
```

and `v_pre` is the **mixed** membrane `vmix = vf + w_c * vs`
(`src/snn/twocomp.py:113,125`), while `vs` is the **reset-shielded** compartment —
never reset, by design (`src/snn/twocomp.py:128`, and the eager path's comment at
:324-329, *"vs is NOT reset — that is the whole point of the candidate"*).

**The multiplier and the surrogate's argument are different variables.**
Expanding,

```
vf * sgd = v_pre * sgd  -  w_c * vs * sgd
           \__________/    \____________/
            bounded by      bounded by NOTHING:
            1.0247 as §2.2  |w_c * vs| * sgd, and vs is an
                            unreset accumulator
```

The self-limiting that bounds the LIF is broken by exactly the structural feature
that defines the arm. A channel near threshold (`sgd ~ 1`) with `|w_c * vs| > 1`
gives `|beta_f * dv| > 1`, and the reverse recurrence **multiplies rather than
contracts** — independently per `(batch, channel)` chain, over up to 256 steps,
with no clamp, clip or normalisation anywhere in `twocomp.py` or `model.py` to
arrest it.

Evaluated at the same frozen constants, the crossover is **`|w_c * vs| ~ 1`**:

| `w_c * vs` | `max|dv|` | `max|beta_f * dv|` | |
|---:|---:|---:|---|
| 0 | 1.0247 | 0.5124 | contracts — identical to the LIF, as `w = 0` nesting requires |
| −1 | 2.0126 | **1.0063** | expands |
| −10 | 11.0023 | 5.5012 | expands |
| −100 | 101.0003 | 50.5001 | expands |

**This is why `w = 0` is bit-identical to the committed LIF** (the kernel comment
at `twocomp.py:146-156` says the grouping was chosen for exactly that) — and it
is also why the arm stops being a contraction as soon as `w` does anything.

### 2.4 The one consequence this carries for decision #11, stated in advance

`dv` is a function of `(v_pre, vs, w_c, alpha, thr, beta_f)`. **It contains no
learning rate, no schedule, no weight decay and no gradient clip.** If §3
confirms the premise, then:

> **No value of any Phase-2 recipe term bounds this quantity.** A lower learning
> rate changes *how fast training reaches* states with `|w_c * vs| > 1`; it does
> not change the multiplier once there. A gradient clip acts on the accumulated
> gradient **after** `backward()` returns and cannot act on a product formed
> inside it.

This is written here, before the run, so that it reads as a consequence of §2.3
rather than as a conclusion reached after seeing a result. **It is referred to
Elliot under #11 and is not applied, and it is not a recommendation to change
anything.** It bears on #11's *form* — whether "re-derive the recipe" is even the
right question — and that is #11's owner's to weigh.

---

## 3. What is measured

Five **forward-only** legs over checkpoints already committed to disk. No leg
trains. `arch` and `d_model` are the only things that vary.

| leg | run | arch | `d_model` | params | role |
|---|---|---|---:|---:|---|
| A1 | `snn_beta0.5_s0` | `snn` | 512 | 735,437 | control, narrow |
| A2 | `scale_d1020_s0` | `snn` | 1020 | 2,501,245 | control, middle |
| A3 | `scale_d1481_s0` | `snn` | 1481 | 4,997,099 | **control at the width that kills the arm** |
| A4 | `twocomp_s0` | `twocomp` | 512 | 737,485 | the arm, at a width where it trains |
| A5 | `arch_twocomp_d1481_s0` | `twocomp` | 1481 | 5,003,023 | **the arm that died, `ckpt_best.pt` = step 5000, 138 steps before death** |

For each leg, on a fixed batch, at each layer, over every `(batch, timestep,
channel)` site, the probe records the per-step chain factor `g = beta * dv` of
§2.1 and accumulates, **online** (no `[B, L, d]` tensor is materialised):

1. `max |g|` and the full quantile set of `|g|`;
2. `frac(|g| > 1)` — the fraction of sites at which the recurrence expands;
3. per-`(batch, channel)` chain, `S = sum_t log|g_t|` — the log of the
   homogeneous gain over the whole 256-step unroll — and `max_{b,c} S`;
4. `max |w_c * vs|`, the quantity §2.3 identifies as the driver.

### Leg B — where the first non-finite value is actually made (~2 GPU-min)

Legs A1–A5 measure the **premise**. Leg B measures the **event**: extend
`011_chase_compose_divergence.py` with a hook that records, at the true failing
step 5138 on the already-eager, already-outside-CUDA-graph instrumented step,
whether layer 0's `grad_cur` is already non-finite when it leaves the scan.

* `grad_cur` non-finite ⟹ the first non-finite value was **born inside the
  256-step reverse recursion**, which is §2.3's mechanism.
* `grad_cur` finite (even at 1e29–1e35) ⟹ the recursion did **not** overflow and
  the first non-finite value was made in an fp32 **reduction downstream** of it.

Leg B runs **only if** T1 passes and P1 resolves (§5). It changes no
hyperparameter and adds no arithmetic to the trajectory.

---

## 4. Hypotheses

### 4.0 What n = 1 checkpoint per cell means, stated before the run

`CONTRIBUTING.md` §2 requires a prediction's power to be stated in advance.

* **There is no σ on any quantity here and this experiment produces none.** One
  checkpoint per cell, one batch. Every number below is a **marker**.
* **No bar is placed on a difference of two measured quantities**, because
  neither term has an interval. Every prediction below is either (i) a comparison
  against a **closed form derived in §2**, or (ii) a **direction**, or (iii) an
  **existence** claim. That is the most these data can carry and it is fixed here
  rather than discovered afterwards.
* **The legs are not step-matched.** `ckpt_best.pt` is at a different training
  step in every run. This is stated as a limitation in §5 and is why P2 is a
  direction and not a magnitude.
* **`sum_t log|g_t|` is not the gradient.** It is the log of the *homogeneous*
  product. The true adjoint also carries per-step `grad_spike * sgd` injections
  and sign cancellation across the sum `grad_cur = gf + gs`. **The asymmetry this
  forces on P3 is pre-registered in P3 itself.**

### T1 — the theorem, and the instrument's own validation *(abort-on-fail)*

> On **A1, A2 and A3** (`arch = "snn"`), at every layer,
> **`max |g| <= 0.5123596 + 1e-5`.**

§2.2 derives this for every finite membrane at every width. It is therefore
**not a test of the theory — it is a test of the probe.** A violation means the
probe does not compute what §2.1 says it computes, and **the experiment aborts
and reports nothing else** (§6). This is `EXP_015` F1's role, one level down.

### T2 — the local scan reproduces the committed forward *(abort-on-fail)*

The probe re-implements the scan locally rather than editing `src/snn/`
(repo convention; `011`'s and `015_chase`'s docstrings both state it). That is
only sound if it is the same scan.

> For every leg, the local scan's spike tensor is **bitwise equal** to the
> committed model's on the same batch, and the final state matches to
> `<= 1e-5` relative.

Bitwise on spikes, not a tolerance: spikes are a comparison against a threshold,
so equality is the honest bar and `EXP_012` is the standing demonstration that a
tolerance borrowed for a threshold comparison hides exactly the thing that
matters.

### P1 — the discriminator *(existence)*

> On **A5** (`twocomp`, `d = 1481`), at **layer 0**, **`max |g| > 1`.**

If `max |g| <= 1` at the width where the arm died, §2.3's mechanism did not fire
in this model and the derivation, though still true as algebra, is **not** the
explanation of the divergence. That outcome is reported as **REFUTED** and Leg B
does not run.

### P2 — width *(direction only)*

> `frac(|g| > 1)` at layer 0 is **strictly larger on A5 (`d = 1481`) than on A4
> (`d = 512`)**.

*Guard:* not step-matched (§4.0), n = 1, no interval. A reversal is informative;
a small same-direction difference is **not** evidence of a width law, and no
functional form may be fitted to two points. Resolves **DIRECTION HELD** /
**DIRECTION REVERSED** / **INDISTINGUISHABLE**, never "width causes it".

### P3 — can the mechanism reach the magnitude actually observed? *(one-sided)*

The eager replay of step 5138 recorded a largest **finite** gradient of
**7.79e29** (`docs/reports/data/exp_015_divergence.json`).

> On **A5**, at layer 0, **`max_{b,c} sum_t log|g_t| >= log(7.79e29) = 68.83`.**

**The power of this prediction is asymmetric and that asymmetry is the point:**

* **Fails** (`max_{b,c} S < 68.83`) ⟹ the homogeneous product **cannot** produce
  the observed magnitude ⟹ §2.3 is **refuted as sufficient**. This direction is
  decisive.
* **Holds** ⟹ the mechanism is **CONSISTENT WITH** the observation. It is **not
  proven**, because `S` ignores the injections and the sign cancellation of
  §4.0, and because A5's checkpoint is 138 steps before the death step.
  **The word "proven" may not be used of a held P3.** Leg B is what upgrades it.

### P4 — locus *(direction only)*

> On **A5**, layer 0's `frac(|g| > 1)` and `max_{b,c} S` both **exceed layer
> 1's**.

Every non-finite gradient in the record is at layer 0 or the embedding and none
is at layer 1. *Guard:* direction only, no interval, and a held P4 does not
establish that the layer-0 tail *causes* the layer-0 locus.

### P5 — the control, which is what makes P1 mean anything *(existence)*

> On **A3** (`snn`, `d = 1481` — same width, same recipe, same tree, trains
> cleanly), at every layer, **`frac(|g| > 1) = 0` exactly.**

T1 makes this a corollary rather than an independent test, and it is listed
separately because it is the sentence that answers `EXP_015`'s constraint (f).
If T1 holds and P5 does not, the probe is inconsistent with itself and §6 aborts.

### P6 — Leg B, the event itself *(binary, no bar)*

> At step 5138, layer 0's `grad_cur` **is already non-finite as it leaves the
> scan.**

Reported as **BORN IN THE RECURSION** or **BORN IN A DOWNSTREAM REDUCTION**.
Both are results; neither is a failure. There is no bar because there is no
statistic — it is one observation of one deterministic event.

---

## 5. Decision rule, fixed now

| | verdict |
|---|---|
| T1 or T2 fails | **ABORT.** Report the instrument defect and nothing else. No P is resolved. |
| P1 holds, P3 holds, P6 = BORN IN THE RECURSION | **The mechanism is identified**: the two-compartment reset Jacobian is unbounded, and the first non-finite value is made inside the reverse recursion. |
| P1 holds, P3 holds, P6 = BORN IN A DOWNSTREAM REDUCTION | **The premise holds and the event is elsewhere.** Report both. The unbounded factor is real and is the amplifier; the overflow is made in an fp32 reduction. Do not call either one "the cause" alone. |
| P1 holds, P3 fails | **§2.3 REFUTED AS SUFFICIENT.** The factor exceeds 1 but cannot reach the observed magnitude. Report as a falsified derivation, prominently, per `CONTRIBUTING.md` §3. |
| P1 fails | **REFUTED.** Leg B does not run. §2 stands as algebra and is not the explanation of this divergence. |

**No verdict in this table adopts anything, changes any hyperparameter, or
decides #11.**

---

## 6. Known limitations, stated in advance

1. **n = 1 checkpoint, 1 batch per cell.** No σ, no seeds, no interval. Markers.
2. **Not step-matched.** §4.0. P2 and P4 are directions for this reason.
3. **A5's checkpoint is step 5000; death is at 5138.** The probe measures the
   state 138 steps *before* the event. It can show the mechanism was already
   available; it cannot show it was what fired. **Leg B exists because of this
   gap and is the only leg that observes the actual failing step.**
4. **`sum_t log|g_t|` bounds the homogeneous gain, and is not the gradient.**
   §4.0, and P3's asymmetry.
5. **The derivation is exact only at the frozen constants** `alpha = 2.0`,
   `thr = 1.0`, `beta = 0.5`. The bound 0.5123596 is not a project constant and
   must be re-derived if any of the three moves — the same error
   `CONTRIBUTING.md` §4 records for fold-in tolerances.
6. **This says nothing about `twocomp_threshold`**, which died at ~2000 and has
   an additional `thr_log` parameter and therefore an additional
   `exp()`. It is not measured here and no claim is made about it.
7. **A held P1 does not make the arm bad.** `twocomp` is the adopted arm and is
   0.134 bpc ahead of the baseline at 735K across 7 seeds. The finding, if it
   holds, is about a **training-stability boundary at width**, not about the
   neuron's value at the width it was adopted at.

---

## 7. Gates

* **T1, T2** — abort-on-fail, above.
* **G1 — no `src/snn/` file is modified by this experiment.** `git diff --stat
  src/snn/` is empty at close. The probe re-implements the scan locally.
* **G2 — determinism.** Each leg is a pure function of `(checkpoint, batch
  index)`; re-running any leg to a scratch `--out` reproduces its artifact at
  **0 differences**, the standard `010`/`011`/`014`/`015` already set.
* **G3 — no hyperparameter is read from anywhere but the run's own
  `config.json`**, and none is overridden.
* **G4 — Leg B adds no arithmetic to the trajectory.** Hooks write into a
  preallocated device tensor with a single D2H copy after `backward()` returns;
  no `bool(...)` sync inside a hook. Leg B's step-5138 loss must reproduce the
  recorded **1.4197630882263184** bitwise, or Leg B is void.

## 8. Entry conditions

* All five checkpoints in §3 exist on disk. **Verified 2026-08-11.**
* GPU idle; nothing else is run while any leg is timed.
* The resolver is committed **before any leg's output is read**.
* This file is committed **before the first probe runs**, and its SHA-256 is
  stamped into the run manifest before and after.

---

## 9. Results

*(appended after the run — nothing above this line is rewritten)*

**Closed 2026-08-11, ~0.8 GPU-hours**, of which **~0.37 bought no number** and is
recorded as spent rather than netted off (§9.8). Pre-registration committed at
`b8887ea` before the first probe; probe, resolver and tests at `6eed5d8` before
any leg was read; `--print-bars` run and read first.

### 9.1 The scoreboard

Per-step chain factor `g = beta * dv`, over every `(batch, timestep, channel)`
site — 48,529,408 sites per layer at `d = 1481`:

| leg | arch | `d` | layer | `max\|g\|` | ÷ LIF bound | `frac(\|g\| > 1)` | `max\|w·vs\|` |
|---|---|---:|---:|---:|---:|---:|---:|
| A1 | `snn` | 512 | 0 | **0.51236** | 1.00 | **0** | — |
| A1 | `snn` | 512 | 1 | **0.51236** | 1.00 | **0** | — |
| A2 | `snn` | 1020 | 0 | **0.51236** | 1.00 | **0** | — |
| A2 | `snn` | 1020 | 1 | **0.51236** | 1.00 | **0** | — |
| A3 | `snn` | 1481 | 0 | **0.51236** | 1.00 | **0** | — |
| A3 | `snn` | 1481 | 1 | **0.51236** | 1.00 | **0** | — |
| A4 | `twocomp` | 512 | 0 | 2.87015 | 5.60 | 5.80e-05 | 22.91 |
| A4 | `twocomp` | 512 | 1 | 2.44904 | 4.78 | 4.53e-06 | 16.21 |
| A5 | `twocomp` | 1481 | 0 | **8.95778** | **17.48** | 5.97e-04 | 279.66 |
| A5 | `twocomp` | 1481 | 1 | 7.89901 | 15.42 | 3.31e-03 | 696.28 |

**The three `snn` legs return 0.51236 — the derived bound, to five decimals — at
every width and every layer, and not one site of 48.5 million exceeds it.** §2.2
is not merely consistent with the data; the data saturate it and stop.

| bar | verdict | |
|---|---|---|
| **T1** | **HOLDS** | every `snn` leg ≤ 0.5123596 + 1e-5. The probe measures what §2.1 says. |
| **T2** | **HOLDS** | local scans bitwise equal to the committed eager references, all five legs, both layers. |
| **P1** | **HELD** | `max\|g\| = 8.95778 > 1` at layer 0, `d = 1481` — **17.48×** the plain LIF's hard ceiling. |
| **P2** | **DIRECTION HELD** | `frac(\|g\| > 1)` at layer 0: 5.80e-05 → 5.97e-04, **10.3×** with width. |
| **P3** | **REFUTED AS SUFFICIENT** | **0 of 189,568 chains expand net** over the full unroll; `max_(b,c) Σ_t log\|g_t\| = −171.91` against a bar of +68.83. |
| **P4** | **DIRECTION REVERSED** | layer **1** has the heavier tail — 3.31e-03 against layer 0's 5.97e-04 — and the higher chain gain. The opposite of the prediction. |
| **P5** | **HELD** | `frac(\|g\| > 1) = 0` **exactly**, both layers, at the width that kills the arm. |

### 9.2 P3 failed, and the failure was the experiment's own design

P3 placed its bar on the product over **all 256** timesteps. The adjoint is not
injected once and carried the length of the unroll: `grad_spike` enters at *every*
timestep, so the cotangent reaching a parameter is a **sum over injection points**,
and each term traverses only its own window. Summing over the whole unroll
averages the mechanism away by construction, which is exactly what the number
says — `max|g| = 8.96` and 0.06 % of sites expanding, with **not one chain in
189,568** expanding net.

**§4.0 and §6 item 4 of this file both state that `Σ_t log|g_t|` "bounds the
homogeneous gain and is NOT the gradient". The limitation was written down and
the bar was placed on the limited quantity anyway.** That is `EXP_012` Y5's shape
and `EXP_013` N1's and `EXP_015` P1/P2's — **the fourth occurrence in Phase 4**,
and the first one authored in full knowledge of the previous three. It is
reported unrepaired per `CONTRIBUTING.md` §3; the corrected quantity is measured
in §9.4 under its own name and **P3's verdict stands as it fired**.

### 9.3 Leg B: the first non-finite value is born inside layer 0's recursion

At step 5138, on the dying run, **both dispatch paths**, with the loss
reproducing **1.4197630882263184 bitwise** (G4):

| boundary, in backward order | `max\|finite\|` | non-finite |
|---|---:|---|
| `grad_spikes_layer1` — enters layer 1's scan from the head | **1.4639e−04** | none |
| `grad_cur_layer1` — leaves layer 1's scan | **1.1027e+15** | none |
| `grad_spikes_layer0` — enters layer 0's scan | 5.6801e+14 | none |
| `grad_cur_layer0` — leaves layer 0's scan | **6.1173e+37** | **140 NaN + 1 inf** |

| stage | amplification |
|---|---:|
| layer 1's 256-step reverse recursion | **× 10^18.88** |
| layer 1's `Linear` backward | × 10^−0.29 |
| layer 0's 256-step reverse recursion | **× 10^23.03** |
| **one backward pass, end to end** | **× 10^41.62** |

**P6 = BORN IN THE RECURSION.** The eager path gives the identical boundaries to
seven significant figures and 141 NaN where the fused path gives 140 NaN + 1 inf,
so **R10 separation now holds one level deeper than `EXP_015` could take it** —
the fused kernel is exonerated at the *boundary*, not merely at the parameters.

### 9.4 POST-HOC: the run does the damage, and Leg A read the wrong batch

Two post-hoc measurements, **labelled post-hoc everywhere they are quoted**, on
`012_posthoc_pileup.py`'s precedent. Neither amends a bar.

The first replaces P3's whole-unroll product with the **maximum contiguous
window** (Kadane, online). The second re-runs both probes on **the batch the run
actually died on** — `--batch-step 5138` — because Leg A read batch 0, which is
an arbitrary batch and not the one that killed it. **That was a defect in Leg A's
design, not only in P3's quantity**, and it is why §6 item 3's limitation turned
out to be load-bearing:

| leg | layer | longest expanding **run** | window gain |
|---|---:|---:|---:|
| | | batch 0 → **batch 5138** | batch 0 → **batch 5138** |
| A1–A3 `snn`, all widths | 0, 1 | **0 → 0** | **10^0.00 → 10^0.00** |
| A4 `twocomp` 735K | 0 | 2 → 3 | 10^0.66 → 10^0.66 |
| A5 `twocomp` 5.0M | 1 | 18 → **83** | 10^1.84 → **10^7.26** |
| A5 `twocomp` 5.0M | 0 | 8 → **85** | 10^1.73 → **10^16.00** |

**On the batch that killed the run, layer 0 has a contiguous run of 85 timesteps
in which the reverse recurrence expands**, worth 10^16 on its own, against 3
steps and 10^0.66 for the same arm at 735K. `max|w·vs|` at layer 0 goes 44.62 →
**356.61** between the two widths on that batch, and `max|g|` reaches **9.84**.

**The plain LIF is at exactly zero in every cell of that table** — no window, at
any width, on either batch, ever expands. It cannot: §2.2 forbids it.

### 9.5 What is now accounted for, and what is not

The measured amplification across layer 0's scan is **10^23.03**. The homogeneous
window supplies **10^16.00**. The coupling term `w_c · gv`, with
`gv = sgd·(grad_spike − grad_vf_next·vf)`, supplies at most
`max|w·vs| · 1/(1 − max β_s) ≈ 356.6 × 71 ≈ 10^4.4`.

**That leaves a residual of ~2.6 orders of magnitude this experiment does not
resolve**, and it is stated rather than absorbed. Three candidates, none
measured here: the window maximum and the injection maximum need not occur at the
same site, so the product of maxima is not a path any single element takes; the
`1/(1 − β_s)` figure is a DC gain and understates an accumulator whose input is
itself growing geometrically backward in time; and **the weights are from step
5000 while the failure is at 5138** — Leg A's checkpoint is 138 steps stale and
only the *batch* was corrected in §9.4, not the parameters.

**No claim is made that §2.3 is the whole cause.** What is established is that the
plain LIF's recurrence provably cannot expand and does not, that the
two-compartment one demonstrably does, that on the fatal batch it does so for 85
consecutive steps, and that the first non-finite value is made inside it.

### 9.6 What this says about `EXP_015` §14.5 — a correction, not a refinement

`EXP_015` §9.7 and report §14.5 both state, of this divergence:

> "**and not a gradient explosion** — the largest gradient in the six steps
> before death is 2.0e-02 and the clip never fires in the whole run."

**That sentence is wrong, and its own committed artifact contains the
refutation.** `exp_015_divergence.json` records `layers.1.weight` at
**1.676e+15** and `layers.1.bias` at **3.70e+13** at step 5138, both finite, and
the eager replay's `grad_abs_max_finite` at **7.79e+29**. §9.3 measures the
explosion directly at **10^41.62 within one backward pass**.

**Both cited facts are true and neither supports the conclusion**, because both
are measured *between* optimiser steps:

* "the largest gradient in the six steps before death is 2.0e-02" is a statement
  about steps 5132–5137, and the explosion happens *inside* the backward at 5138;
* "**the clip never fires**" is not evidence against an explosion — it is a
  **consequence** of one. `clip_grad_norm_` acts on the accumulated gradient
  after `backward()` returns. Here the overflow occurs inside `backward()`, so by
  the time the clip computes a norm the tensor already holds NaN and the norm is
  NaN. A run that died of a 10^41.6 gradient explosion logs
  `max_grad_norm = 0.663` and `n_logged_steps_over_clip = 0` **for exactly that
  reason**.

"Always layer 0 and the embedding; never layer 1" stands as written about
*non-finiteness* and is misleading about *magnitude*: layer 1 carries 1.10e+15
where it carried 1.46e−04 one boundary earlier. It is not spared — **it has not
yet run out of exponent.**

Per `CONTRIBUTING.md` §3 the closed `EXP_015` log is **not** rewritten. The
correction is carried in the report at rev 10 with the rev-9 sentence left
standing beneath it, the same treatment rev 9 (i) gave rev 8's gap arithmetic.

### 9.7 What this does not answer, and what it must not be read as

* **It is not a fix.** Naming an unbounded quantity is not bounding it. No remedy
  is proposed, designed or costed here.
* **It does not say the arm is bad.** `twocomp` is the adopted arm, 0.134 bpc
  ahead of the baseline at 735K across 7 seeds, where its longest expanding run
  is **3 steps**. This is a *training-stability boundary at width*, not a verdict
  on the neuron at the width it was adopted at.
* **It says nothing about `twocomp_threshold`** (died ~2000), which was not
  probed and has an additional `exp()`.
* **It does not establish a width law.** Two widths for the arm, n = 1
  checkpoint, no σ. `frac(|g| > 1)` rising 10.3× is a direction.
* **It does not decide #11.** §9.8 states the consequence; the ruling is Elliot's.

### 9.8 Cost, and the ~0.37 GPU-hours that bought no number

~0.8 GPU-hours total. **Two Leg B attempts were aborted and are recorded as spent
rather than netted off**, following `EXP_015` §7.1's precedent:

1. `keep_spikes = True` was set before the 132-step replay rather than for the
   instrumented step alone, retaining two `[128, 256, 1481]` tensors per step;
   the card reached 7.7 of 8.1 GiB and entered the WDDM spill this project has
   hit before — **a ~50× slowdown that never raises**.
2. `fused = False` was set before the replay rather than after it, so 132 steps
   replayed on the eager path (~13 kernels per timestep per layer) under graph
   capture. `011`'s pass 3 already does this correctly at `011:280-285`; the fix
   was to copy it. **The reusable lesson is the one `EXP_015` §9.10 recorded in
   different words: the instrument's own defects cost more than the measurement.**

A stray child process survived the first abort and held the GPU; it was killed
before the re-run. No leg here is timed, so no wall-clock figure is affected.

### 9.9 Gates

| gate | result |
|---|---|
| **T1** | HOLDS — §9.1 |
| **T2** | HOLDS — bitwise, five legs, both layers |
| **G1** — no `src/snn/` file modified | **PASS**, `git diff --stat src/snn/` empty |
| **G2** — determinism / 0-difference reproduction | **PASS** — both Leg B passes re-derive the same boundaries; Leg A is a pure function of (checkpoint, batch index) |
| **G3** — no hyperparameter read from outside each run's `config.json` | **PASS** |
| **G4** — Leg B adds no arithmetic; loss reproduces bitwise | **PASS**, `1.4197630882263184` on both paths |
| tests | `tests/test_reset_jacobian.py` **10 passed**; `ruff` clean |

### 9.10 The pre-registration guarantee

`experiments/logs/EXP_016_reset_jacobian.md` §0–§8 was committed at **`b8887ea`**,
before the first probe ran, and its SHA-256 over the working-tree bytes at that
commit is

> `28c5e92625dfba93503560153b3e25453569837e47dcf953402533eacf95ce39`

**The recipe is CRLF-dependent**, exactly as `EXP_015` §9.12 records for this
tree: `.gitattributes` sets `* text=auto eol=lf`, so git stores LF and checks out
CRLF here, and the digest above is over the checked-out bytes. Reconstructing
with `\n` gives a different value and would wrongly read as an edited
pre-registration.

---

## 10. Referred to Elliot, and not decided here

1. **Decision #11's form, not its answer.** §2.4 was written before the run and
   §9 does not weaken it: `dv` is a function of `(v_pre, vs, w_c, alpha, thr,
   beta_f)` and **contains no learning rate, no schedule, no weight decay and no
   gradient clip.** §9.6 shows the clip *cannot* act on this failure — it reads a
   norm that is already NaN. So "may the frozen recipe be re-derived at a new
   size?" may be the wrong question to rule on: no value of any recipe term
   bounds this quantity. **Nothing is proposed and no term is changed.**
2. **Whether a remedy is a ranked candidate.** Anything that bounds the
   recurrence — clamping the slow compartment, resetting it, normalising `cur`,
   or truncating BPTT — is a **new arm**, needs its own pre-registration and its
   own σ, and would change the adopted arm's function class. Not designed here.
3. **The residual 2.6 orders of magnitude** (§9.5). The cheapest next measurement
   is Leg A's probe run against the step-**5137** weights rather than step 5000's,
   which needs a checkpoint the dying run never wrote — one replay with a dump,
   ~2 GPU-minutes.
4. **The verdict vocabulary, a fourth time** (§9.2). `EXP_015` §10 item 3 referred
   **DIVERGED / NO RESULT**; this file adds that a bar can also be placed on a
   quantity whose own limitations section already says it is the wrong one. That
   is decision **#9**'s substance again, and the guard `CONTRIBUTING.md` §2
   requires plainly did not catch it here.
5. **Whether `EXP_014` §9.10's width finding should be re-read.** Firing rates
   fall with width and `max|w·vs|` rises with it. Nothing here connects them, and
   the connection is not this file's to assert.

