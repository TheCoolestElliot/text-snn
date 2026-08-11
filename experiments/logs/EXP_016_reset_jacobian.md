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
