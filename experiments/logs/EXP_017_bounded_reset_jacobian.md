# EXP_017 — Bounding the reset Jacobian: can the adopted arm be trained at width?

**Pre-registered:** 2026-08-11, after the derivation in §2 and **before any leg
was run, any checkpoint was probed, and any bpc existed for this arm.**
**Committed before the first run.**

**Status: PRE-REGISTERED — no results yet.**

---

## 0. What this experiment is, and what it is not

`EXP_015` found that both two-compartment arms **fail to train at 5.0M
parameters** under the frozen Phase-2 recipe, deterministically, while the plain
LIF and the GRU at the identical width, seed, recipe and tree train cleanly.
`EXP_016` found out why: the adopted arm's backward-through-time is **bounded by
nothing**, the plain LIF's is **provably a strict contraction at any width**, and
the difference is one line of arithmetic. `EXP_016` §10 item 2 referred the
remedy:

> **Whether a remedy is a ranked candidate.** Anything that bounds the recurrence
> — clamping the slow compartment, resetting it, normalising `cur`, or truncating
> BPTT — is a **new arm**, needs its own pre-registration and its own σ, and would
> change the adopted arm's function class. Not designed here.

**This is that new arm, and it is the one member of that list that does *not*
change the function class** — because it changes nothing in the forward at all.
§2 derives why before anything is measured.

**What this is not:**

* **It adopts nothing and ranks nothing.** Adoption is Elliot's (report §8) and
  this arm is not proposed for it. `adopts_nothing` and `ranks_nothing` are
  carried as structural fields in the driver's manifest, not as prose.
* **It changes no hyperparameter.** `lr = 3e-3`, `wd = 0.1`, `grad_clip = 1.0`,
  cosine over 20,000, `warmup = 200`, `beta_slow = 0.95`, `w_init = 0.1` are all
  inherited **unchanged**. G5 verifies that from each run's own `config.json`.
  **That is what makes H2 mean anything**: if the arm survives, it survives under
  the recipe that killed the adopted arm, not under a recipe chosen to let it.
* **It does not decide #11.** `EXP_016` §10 item 1 put #11's *form* in question
  because `dv` contains no recipe term. This experiment does not answer that; it
  supplies a second data point of the same shape — that the failure is fixable
  **without touching the recipe at all** — and refers it.
* **It does not edit the adopted arm.** `src/snn/twocomp.py`, `kernels.py`,
  `neuron.py` and `surrogate.py` are untouched (G1). The new neuron is a new
  module with its own gate and its own mutations, which is the precedent
  `EXP_004` set when it added the second neuron.
* **It measures a cost as well as a benefit, and the cost leg is not optional.**
  H3 and H4 exist because a change that makes an arm trainable by damaging it is
  the failure mode `EXP_013`'s N1 walked into, and §4 encodes the guard rather
  than trusting §2's argument.

---

## 1. The question

> **`EXP_016` proved the adopted arm's reverse recurrence is unbounded. Can it be
> bounded without changing what the network computes — and if so, what does the
> bound cost?**

Two halves, and both are pre-registered, because half of this question has an
answer that flatters the work and half does not.

---

## 2. The derivation, done before any measurement

### 2.1 The per-step chain factor (restated from `EXP_016` §2.1)

Both neurons thread an adjoint over all `L = 256` steps of the unroll with
`grad_vf_prev = beta_f * (grad_vf_next * dv + grad_spike * sgd)`, so **`beta_f *
dv` is the per-step multiplier of the homogeneous reverse recurrence.**

| | `dv` | bound |
|---|---|---|
| plain LIF (`kernels.py:130`, `"hard"`) | `(1 − s) − v_pre·sg(v_pre − thr)` | **0.5123596** — `EXP_016` §2.2 |
| `twocomp` (`twocomp.py:180`) | `(1 − sh) − vf·sgd(v_pre − thr)`, `vf = v_pre − w·vs` | **none** |
| **this arm** (`twocomp_detach.py`) | **`(1 − sh)`** | **`beta_f` = 0.5** |

### 2.2 What the detached reset does to the *whole* Jacobian, not just to `dv`

`dv` is only the diagonal. Writing the per-step homogeneous backward as a 2×2
matrix acting on `(grad_vf_next, grad_vs_next)` — read directly off
`twocomp.py:179-187` — the adopted arm is

```
J_hard   = [[ beta_f·dv            ,  0      ]        dv = (1 − sh) − vf·sgd
            [ −beta_s·w·sgd·vf     ,  beta_s ]]
```

and **`vf` appears twice**: once in the diagonal and once in the off-diagonal
coupling, which injects the fast compartment's cotangent into the slow one. The
crossover for the diagonal is `|w·vs| ≈ 1`; `EXP_016` measured `max|w·vs| = 279.7`
at `d = 1481` and `max|beta_f·dv| = 8.96`.

Detaching the reset sends `∂vf_out/∂v` to zero, which removes `vf` from **both**
entries at once:

```
J_detach = [[ beta_f·(1 − s) ,  0      ]
            [ 0              ,  beta_s ]]
```

Three consequences, each stated as a claim that could be false:

1. **`(1 − s) ∈ {0, 1}`, so `|beta_f·dv| ≤ beta_f = 0.5` exactly** — for every
   finite membrane, every `w`, every `vs`, every batch, every channel, every step
   and **every width**. Nothing in it depends on `d_model`, on the mix, or on the
   slow pole. This is a **theorem, not a measurement**, and it is *strictly
   tighter* than the plain LIF's own 0.5123596.
2. **The off-diagonal is exactly zero**, so the Jacobian is **diagonal** and the
   product over any window is exactly the product of the diagonals. There is no
   non-normal transient either — a stronger statement than "bounded", and one the
   plain LIF gets for free only because it has one compartment.
3. **The slow pole's contraction rate is `beta_s`, unchanged.** `beta_s =
   sigmoid(beta_s_raw)` lies in `(0, 1)` by construction and weight decay pulls
   `beta_s_raw` toward 0, i.e. toward *more* damping — `EXP_015` ruled out a
   runaway slow pole on exactly this argument, without a GPU. **So the mechanism
   the arm exists for is not touched**: what is removed is only the two terms that
   carry the reset back through the spike.

### 2.3 The forward is not merely equivalent — it is the same kernel

`detach()` changes no number. `snn/twocomp_detach.py` **imports**
`twocomp_forward_kernel` from `snn.twocomp` rather than declaring a forward source
of its own, so there is no second copy of the text to drift and no second NVRTC
compile. Therefore:

* the arm adds **no parameter** (`twocomp_param_count` applies unchanged — it is
  parameter-**identical** to the adopted arm, not parameter-matched);
* it adds **no state variable, no new function and no inference cost**;
* under `model.eval()` there is no backward at all, so a checkpoint trained here
  **is** an adopted-arm checkpoint. `as_twocomp_state_dict` is a plain copy and
  the logits match at **`== 0.0`**.

That last point is worth one sentence of contrast, because this project has twice
had to report a fold-in residual. `EXP_008` W1 folded a gain into a `Linear` and
landed at 1.9e-03 bpc; `EXP_011` C4 at 1.05e-06. **Here there is nothing to fold**
— same parameter set, same kernel — so the identity is exact and decision #6's
tolerance question does not arise for this arm at all.

### 2.4 The cost, stated in advance and not buried

**This is a biased gradient estimator.** The pathway "firing now lowers my own
future membrane" is dropped from the backward. The model still learns *through*
the spike — `grad_spike · sgd` is untouched, and `EXP_017`'s gate asserts both
per-channel gradients stay live in every channel — but the reset's own
contribution to `∂vf_out/∂v` is gone.

**It is not novel and this file does not claim it is.** Detaching the reset is
standard in the surrogate-gradient literature (`detach_reset=True` in
spikingjelly) and has been a first-class mode of *this project's own* LIF kernel
since Phase 2, with its own compiled kernel and its own mutation coverage (M04,
M05). What §2.2 contributes is the observation that on a **two-compartment**
neuron it stops being a stylistic choice about a gradient and becomes **the one
change that restores the contraction the second compartment destroyed** — and
that it does so while leaving the forward bit-identical.

**Whether the bias costs bpc is an empirical question and §4's H3 is it.** §2
gives no reason to expect it to be free, and a §2 that argued it was free would be
the post-hoc rationalisation this protocol exists to prevent.

### 2.5 The alternative that is NOT run here, named so it is on the record

A **soft** reset on the fast pole (`vf_out = vf − thr·s`) also bounds the
Jacobian: `dv = 1 − thr·sgd ∈ [0, 1)`, so `|beta_f·dv| ≤ 0.5`. It is a weaker
result — the off-diagonal becomes `−beta_s·w·thr·sgd`, bounded by `|w|` rather
than zero, and the **forward changes**, so the arm would have its own function
class, its own inference model and a real fold-in question. It is named here, with
its derivation, so that a reader can see the choice was made rather than
overlooked, and it is **referred, not run** (§10).

---

## 3. What is measured

### Leg A — the bound, on the membranes that actually killed the run (~3 GPU-min)

Forward-only, over checkpoints already on disk. **No leg trains.** The probe is
`EXP_016`'s, extended with the detached rule; both rules are evaluated **on the
same membranes from the same checkpoint on the same batch**, so the only thing
that varies is the backward rule.

| leg | run | arch | `d` | role |
|---|---|---|---:|---|
| A1 | `snn_beta0.5_s0` | `snn` | 512 | instrument check (T1) |
| A2 | `scale_d1020_s0` | `snn` | 1020 | instrument check (T1) |
| A3 | `scale_d1481_s0` | `snn` | 1481 | instrument check (T1), at the killing width |
| A4 | `twocomp_s0` | `twocomp` | 512 | the arm where it trains |
| A5 | `arch_twocomp_d1481_s0` | `twocomp` | 1481 | **the arm that died**, `ckpt_best` = step 5000 |

Batches: **0 and 5138**, the second being the batch the run actually died on —
`EXP_016` §9.4 recorded that reading batch 0 alone was a defect in *its* Leg A
design, and this file inherits the correction rather than repeating the mistake.

Recorded per leg, per layer, per rule, accumulated online (no `[B, L, d]` tensor
materialised): `max|g|`, `frac(|g| > 1)`, the **longest contiguous expanding run**
and its window gain (Kadane, `EXP_016` §9.4's corrected quantity — *not* the
whole-unroll sum whose bar misfired there), and `max|w·vs|`.

**A5 under the detached rule is the counterfactual that matters**: on the exact
weights and the exact batch that produced 85 consecutive expanding timesteps worth
10^16, what does the bounded rule give?

### Leg B — the headline: does it train at width? (~42 GPU-min)

One run. `arch = "twocomp_detach"`, `d_model = 1481`, `seed = 0`, 20,000 steps,
**every other flag identical to `arch_twocomp_d1481_s0`'s** — the run that died at
step 5138 — and scored on test.

### Leg C — the cost at the width the arm was adopted at (~55 GPU-min)

Six runs at `d_model = 512`, 20,000 steps each:

* `detach_d512_s{0,1,2}` — the arm;
* `anchor_twocomp_d512_s{0,1,2}` — **a freshly trained `twocomp` anchor on today's
  tree.**

The anchor is trained rather than read off the committed 2.11869 because
[`snn-committed-baseline-not-reproducible`] / `EXP_014` §9.12 established that
decision #7's fp64 clip moved the trajectory, and **Elliot ruled on 2026-08-10
that this project does not re-baseline: every new experiment trains its own
anchor.** Spending 0.45 GPU-h on it is that ruling being obeyed, not redundancy.

### Leg D — reach: the thing the arm exists for (~12 GPU-min)

`scripts/exp/001_memory_horizon.py`, unmodified but for two registry entries,
over all six Leg-C checkpoints. Same statistic, same `k`-sweep, same F1 tolerance
as every horizon number in the project, with an explicit `--out` so no committed
artifact is overwritten. **Both arms are probed**, so the comparison is
anchor-to-arm on one tree rather than arm-to-committed-figure.

---

## 4. Hypotheses

### 4.0 What this design can and cannot resolve, stated before the run

`CONTRIBUTING.md` §2 requires a prediction's power to be stated in advance, and
§3 requires every bar to be checked against §6 before it is committed. Both were
done, and the results are these:

* **Leg B is n = 1 seed.** It can resolve a *deterministic* binary event and
  nothing else. It cannot establish "this arm is stable at width" and no verdict
  below says so.
* **Leg B's bpc is a marker.** n = 1, no σ, no interval. It may be compared with
  `EXP_015`'s table because the runs are parameter-identical and share width,
  seed, recipe and tree — but a difference of any size between two n = 1 numbers
  resolves nothing and §5 attaches no verdict to it.
* **Leg C resolves ~0.005 bpc and no better.** σ on the two-compartment neuron is
  not established for this arm and is re-measured here (`CONTRIBUTING.md` §4);
  taking `EXP_005`'s two-state figure 0.00313 as the planning value, three seeds
  per arm give a two-sample se of ≈0.00256 and a 2-se bar of **≈0.0051 bpc**.
  **A |Δ| below that is UNRESOLVED, which is not the same sentence as "the arms
  are equal", and H3 says so in its own verdict table.**
* **σ itself is reported with 2 degrees of freedom.** Three seeds give a very
  noisy σ. It is a marker, is quoted with its df, and **must not be promoted to a
  project constant** — which is the error `EXP_005` and `CONTRIBUTING.md` §4
  already record twice.
* **The horizon has no error bar and this project has never established
  σ_horizon.** H4 is a direction; all six per-seed values are reported so a reader
  can see the spread rather than trust a median.

### T1 — the instrument, checked against a number it did not produce *(abort-on-fail)*

> Run with the **hard** rule, legs A1–A3 (`arch = "snn"`) return
> `max|g| = 0.51236` to 5 dp and `frac(|g| > 1) = 0` at every layer, reproducing
> `EXP_016` §9.1 exactly.

This is `CONTRIBUTING.md` §5's standing rule — *make every probe reproduce a
number it did not produce*. A violation means this probe is not `EXP_016`'s probe,
and **the experiment aborts and reports nothing else.**

### T2 — R10, before a single training step *(abort-on-fail)*

> `tests/test_twocomp_detach_equivalence.py` passes in full, **and**
> `scripts/audit/08_mutation_campaign.py` catches **12 of 12** new D-mutations
> with the pre-existing 47 still caught.

`EXP_002` §8.6 priced a candidate neuron's real cost as "its own mutation-tested
gradient gate before it may run", and this is that cost being paid. **A wrong
backward still trains and still produces a plausible bpc**, so if any D-mutation
escapes, no number from Legs B, C or D may be reported at all.

### H1 — the bound, on real membranes *(closed form; a test of the probe)*

> Under the **detached** rule, on **every** leg A1–A5, every layer, both batches:
> **`max|g| ≤ 0.5 + 1e-6`** and **`frac(|g| > 1) = 0` exactly**, and the longest
> expanding run is **0**.

§2.2 derives this for every finite membrane at every width, so — exactly as
`EXP_016`'s T1 was — **this is more a test of the probe than of the theory.** Its
informative content is entirely in A5, where the *same membranes* under the hard
rule give `max|g| = 8.96` and an 85-step expanding run. **A violation would mean
the implementation does not compute what §2.2 says**, and resolves as REFUTED with
Legs B–D reported but explicitly disconnected from §2's explanation.

### H2 — the headline *(binary, deterministic event, n = 1)*

> `detach_d1481_s0` completes **20,000 steps with zero non-finite logged losses**,
> under the unchanged frozen recipe — and in particular passes **step 5138**,
> where `arch_twocomp_d1481_s0` died.

Resolves **SURVIVES** / **DIVERGED at step _k_**.

*Guards, fixed now:*

* The adopted arm's failure is **deterministic** and has been reproduced three
  times (`EXP_011` `compose_s0` at 17,598; `EXP_015` at 5138; `EXP_015` §9's
  replay of the same). A single non-divergence is therefore genuinely informative
  **against that failure** — and against nothing else. n = 1 buys no claim about
  stability in general, and **"the arm trains at width" is not an available
  sentence from this leg**; "the arm did not exhibit the adopted arm's
  deterministic divergence" is.
* **If it diverges at a different step, that is a different finding.** It is
  reported as a new divergence, chased per `CONTRIBUTING.md` §4, and **not**
  reported as "the bound failed" — §2.2's bound would still hold and the cause
  would be elsewhere.
* A SURVIVES verdict says nothing about whether the arm is *good* at width. That
  is H2b's number and H2b is a marker.

### H2b — what it scores at width *(MARKER, no verdict)*

> `detach_d1481_s0`'s carried test bpc, reported beside `EXP_015`'s parameter-
> identical table (`snn` 2.00073, `twocomp` NaN, `twocomp_threshold` NaN, `gru`
> 1.54699).

**No bar and no verdict attached**, per §4.0. It is reported because a reader's
first question after a SURVIVES will be "and is it any good", and leaving the
number out would invite it to be inferred from something worse.

### H3 — the cost at 735K *(difference, BOTH terms constrained)*

> `Δ = mean bpc_carried(detach, n=3) − mean bpc_carried(anchor, n=3)` at
> `d_model = 512`, with a Welch two-sample se.

| | verdict |
|---|---|
| `Δ ≤ −2 se` | **THE BOUND IS FREE OR BETTER** at 735K |
| `Δ ≥ +2 se` | **THE BOUND COSTS `Δ` bpc** at 735K |
| otherwise | **UNRESOLVED at n = 3** — *and this is a verdict, not a failure to report one* (`CONTRIBUTING.md` §3) |

**The guard that makes this a bar on a difference rather than on one term.**
`EXP_013`'s N1 fired at two amplitudes whose test bpc was catastrophically worse,
because it constrained a difference without constraining its subtrahend. So:

* **H3a (the anchor's own validity — a MARKER with a stated consequence, not a
  gate).** The fresh anchor's mean carried bpc is compared against the committed
  n=7 figure **2.11869**. If `|mean − 2.11869| > 0.02`, **H3 is still resolved** —
  it compares two arms trained on one tree, which is the comparison the ruling on
  #10 prescribes — but every H3 number is quoted with the anchor discrepancy
  beside it and the report says the tree moved. 0.02 is ≈6× the sum of `EXP_014`
  G1's measured tree shift (1.97e-03) and three seeds' se (≈0.0018), chosen loose
  **because it is a sanity marker and a tight marker would masquerade as a gate.**
* **Both arms' absolute numbers are reported, not only Δ.** A Δ favourable to the
  detached arm because the *anchor* collapsed is not a win, and the only way to
  see that is to print both.
* `Δ = 0` exactly is measure-zero on a continuous bpc, so no verdict boundary here
  sits on a value the metric can land on.

### H4 — reach *(paired direction; the tie is decided now)*

> Paired per-seed 2σ horizon difference `Δh_i = h(detach, s_i) − h(anchor, s_i)`
> for `i ∈ {0,1,2}`, from `001_memory_horizon.py` at `tol = 0.00922`.

| | verdict |
|---|---|
| median `Δh ≥ 0` | **REACH RETAINED** |
| median `Δh < 0` and median `h(detach) > 7` | **REACH REDUCED** |
| median `h(detach) ≤ 7` | **REACH LOST** — indistinguishable from the plain LIF |

**`median Δh == 0` resolves RETAINED**, stated here so the tie is decided before
it is seen — the horizon is an integer on a dense lattice and `CONTRIBUTING.md`
§3 forbids leaving a reachable boundary value undecided.

*Guards:* n = 3, no σ_horizon anywhere in this project, no interval. **A median
Δh of −1 and of −20 both resolve REDUCED and the raw six values must be quoted.**
This resolves a direction and is evidence about the mechanism; **it may not be
used on its own to adopt or reject the arm.**

### H5 — the systems column must not discriminate *(marker + alarm)*

> Wall-clock per step and peak VRAM within **1.10×** of the adopted arm's at the
> same width (`twocomp_s0` 536.69 s, `arch_twocomp_d1481_s0` 2432.24 s).

The forward is the same kernel and the backward is one kernel per timestep with
one *fewer* input, so anything outside that band is a systems finding to chase,
not a result. Reported as a marker; a breach raises rather than resolves.

---

## 5. Decision rule, fixed now

| | verdict |
|---|---|
| T1 or T2 fails | **ABORT.** Report the instrument or gate defect and nothing else. No H is resolved. |
| H1 refuted | The implementation does not realise §2.2. Legs B–D reported if already run, **explicitly disconnected from §2's explanation**, and the discrepancy chased. |
| H1 held, H2 SURVIVES | **The mechanism `EXP_016` identified is sufficient to explain the divergence, and bounding it is sufficient to prevent it** — under one seed, one width, one recipe. H3/H4 then say what it cost. |
| H1 held, H2 DIVERGED at 5138 | **§2.2 REFUTED AS SUFFICIENT.** The Jacobian is bounded and the run still dies at the same step, so `EXP_016`'s mechanism is not the cause. Report prominently per `CONTRIBUTING.md` §3 — this is the outcome that would most change the project. |
| H1 held, H2 DIVERGED elsewhere | A **new** divergence. Chased, not absorbed. §2.2 stands. |
| H3 = COSTS, H4 = LOST | The bound is bought by destroying the mechanism. **Report as a negative result**; the arm is not worth ranking and §10 says so. |

**No verdict in this table adopts anything, changes any hyperparameter, decides
#11, or authorises Phase 5.**

---

## 6. Known limitations, stated in advance

1. **Leg B is n = 1 seed at one width.** §4.0, and H2's guards.
2. **Leg C resolves ~0.005 bpc.** A smaller true difference is invisible to this
   design, and UNRESOLVED must not be read as zero.
3. **σ is re-measured here at 3 seeds (2 df) and is a marker**, not a project
   constant. `CONTRIBUTING.md` §4.
4. **The anchor is n = 3 against a committed n = 7.** The comparison's power is
   the fresh pair's, not the committed figure's.
5. **`beta_slow` and `w_init` are NOT re-tuned for this arm.** Both frozen inits
   are inherited, exactly as `EXP_011` inherited both parents'. An arm with a
   different backward may well want a different mix at init, and this experiment
   does not look.
6. **Nothing here is measured on `twocomp_threshold`**, which died at ~2000 and
   has an additional `exp()`. No claim is made about it.
7. **The bias is measured only through bpc and horizon.** This experiment does not
   measure what the biased estimator is approximating, or how far the two
   trajectories separate — only where each one lands.
8. **The bound `0.5` is `beta_f` and is exact only at the frozen `beta_f = 0.5`.**
   It is not a project constant and must be re-derived if `beta` moves — the same
   error `CONTRIBUTING.md` §4 records for fold-in tolerances.
9. **Leg A probes the detached rule on weights trained by the HARD rule.** It is a
   counterfactual on real membranes, which is the strongest available statement
   before Leg B exists, and it is not the same thing as the arm's own trajectory.
   After Leg B the probe is re-run on `detach_d1481_s0`'s own checkpoint and both
   are reported.
10. **A SURVIVES at n = 1 cannot exclude that the arm dies at 40,000 steps, at
    another width, or at another seed.** The budget buys one run.

---

## 7. Gates

* **T1, T2** — abort-on-fail, above.
* **G1 — the adopted arm is not edited.** `git diff` over `src/snn/twocomp.py`,
  `src/snn/kernels.py`, `src/snn/neuron.py`, `src/snn/surrogate.py` is **empty**
  against the commit preceding this experiment. `EXP_004` failure mode J9.
* **G2 — parameter identity.** Every run's realised `params` equals
  `twocomp_param_count` exactly, at both widths. Not "matched to 0.18%" —
  **equal**, and the driver aborts otherwise.
* **G3 — K1, one thing changed.** Each run's `config.json` is its reference run's
  with `arch`, `run_name` and (Leg C only) `seed` changed and **nothing else**.
  Reference: `arch_twocomp_d1481_s0` for Leg B, `anchor_twocomp_d512_s0` for
  Leg C. The driver aborts on any other differing field.
* **G4 — determinism.** Leg A is a pure function of (checkpoint, batch index) and
  re-running it to a scratch `--out` reproduces its artifact at **0 differences**.
* **G5 — no hyperparameter moved.** `lr`, `weight_decay`, `grad_clip`,
  `warmup_steps`, `max_steps`, `lr_schedule`, `beta`, `beta_slow`, `w_init`,
  `threshold`, `surrogate_alpha` read from each run's own `config.json` and
  asserted equal to the frozen values. This is G3 restated on the fields that
  matter most, deliberately, because H2's whole meaning depends on it.
* **G6 — VRAM alarm at 4.0 GiB**, `EXP_015`'s figure at this width (measured
  peak there: 2.61 GiB). Under WDDM a spill is a ~50× slowdown that never raises.
* **G7 — a diverged leg is recorded as diverged.** The driver reads the log for
  non-finite losses rather than trusting the exit code, and requires
  `log_present`; `EXP_015` §9.10 records both halves of that defect.

---

## 8. Entry conditions

* All five Leg-A checkpoints exist on disk. **Verified 2026-08-11.**
* `tests/test_twocomp_detach_equivalence.py` green and the mutation campaign at
  12/12 on the new mutations, **before** the first training step (T2).
* GPU idle; nothing else runs while any leg is timed
  (`CONTRIBUTING.md` §4, and [`snn-gpu-isolation-gotcha`]).
* The resolver is committed **before any leg's bpc is read**.
* This file is committed **before the first leg runs**, and its SHA-256 is
  stamped into the run manifest before and after. Per `EXP_015` §9.12 the digest
  is over the **working-tree (CRLF)** bytes; reconstructing with `\n` gives a
  different value and would wrongly read as an edited pre-registration.

---

## 9. Results

*(appended after the run — nothing above this line is rewritten)*

**Closed 2026-08-11, ~2.0 GPU-hours.** Pre-registration committed at `50b3904`
before the first leg; the arm, its gate and its 12 mutations at `6107975`; the
probe, driver and resolver at `34f07c9`, before any bpc was read. SHA-256 over
the working-tree (CRLF) bytes at `50b3904`:

> `a07ba5c3c5e03580ec8ae04f7f820faa8b45f47b606df0431c6a4c33b3240f6f`

### 9.1 The scoreboard

| bar | verdict | |
|---|---|---|
| **T1** | **HOLDS** | the three `snn` legs return `max\|g\| = 0.51236` at every width and layer |
| **T1b** | **HOLDS** | the `hard` column reproduces `EXP_016`'s committed artifact, **10/10 cells, 0 differing** |
| **T2 (probe)** | **HOLDS** | local scan bitwise equal to the committed eager scan, six legs, both layers |
| **T2 (gate)** | **HOLDS** | **59/59** mutations caught — 12 new, 47 pre-existing — before a single training step |
| **H1** | **HELD** | under the detached rule `max\|g\| = 0.5`, `frac(\|g\|>1) = 0`, longest expanding run **0** — every leg, every layer, both batches |
| **H2** | **SURVIVES** | 20,000 steps, **0 non-finite losses**, passed step 5138 |
| **H2b** | **MARKER** | carried test bpc **1.86290** at 5.0M parameters |
| **H3** | **NOT RUN** | 3 scoreable detach seeds, **1** anchor seed — see §9.4 |
| **H4** | **REACH REDUCED** | on **one** available pair, `Δh = −2` — see §9.6 |
| **H5** | **BREACH — investigate** | 1.127× at `d = 1481` against a denominator that is itself a diverged run — see §9.7 |

### 9.2 Leg A: the same membranes, both rules

`g = beta_f · dv`, over every `(batch, timestep, channel)` site. **Both columns
come from one forward pass on one checkpoint** — `detach()` changes no forward
value, so nothing here can be attributed to two arms having reached different
states. Batch 5138 is the batch the dying run actually saw at the step it died.

| leg | arch | `d` | layer | hard `max\|g\|` | hard run | hard window | **detach `max\|g\|`** | **detach run** |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| A1 | `snn` | 512 | 0,1 | 0.51236 | 0 | 10^0.00 | **0.5** | **0** |
| A2 | `snn` | 1020 | 0,1 | 0.51236 | 0 | 10^0.00 | **0.5** | **0** |
| A3 | `snn` | 1481 | 0,1 | 0.51236 | 0 | 10^0.00 | **0.5** | **0** |
| A4 | `twocomp` | 512 | 0 | 2.87016 | 3 | 10^0.66 | **0.5** | **0** |
| A5 | `twocomp` | 1481 | 0 | **9.84309** | **85** | **10^16.00** | **0.5** | **0** |
| A5 | `twocomp` | 1481 | 1 | 7.73804 | 83 | 10^7.26 | **0.5** | **0** |
| A6 | `twocomp_detach` | 1481 | 0 | 5.92227 | **84** | **10^27.58** | **0.5** | **0** |
| A6 | `twocomp_detach` | 1481 | 1 | 6.06335 | 17 | 10^4.44 | **0.5** | **0** |

**A6 is the row worth reading twice, and it is not the row that was expected.**
It is this arm's *own* trained checkpoint. Under the rule it actually trains
with, its recurrence never expands. Under the hard rule those same weights would
expand for **84 consecutive timesteps with a window gain of 10^27.58** — an order
of magnitude *worse* than the adopted arm's own 10^16.00 at the step it died.

> **So the fix does not work by steering training away from the dangerous region
> of weight space. It trains happily inside it, because its gradient is
> indifferent to `w·vs`.** Nothing in §2 predicted this and nothing in §2
> forbids it: the derivation bounds `dv`, and says nothing about where the
> forward goes. It is the single most useful thing Leg A measured, and it is
> **post-registered as an observation, not a bar** — no prediction was placed on
> A6's `hard` column.

`max|w·vs|` at A6 layer 0 is **445.9** against A5's 356.6 on the same batch: the
bounded arm's slow compartment grows *more*, not less.

**G4 (determinism):** the probe was re-run in full after Leg B. Across A1–A5 ×
2 batches, **200 quantities compared, 0 differing.**

### 9.3 Leg B: it trains, under the recipe that killed the adopted arm

| | `arch_twocomp_d1481_s0` | **`detach_d1481_s0`** |
|---|---|---|
| outcome | **DIVERGED, step 5138** | **20,000 steps, 0 non-finite** |
| carried test bpc | NaN | **1.86290** |
| parameters | 5,003,023 | **5,003,023** (identical, G2) |
| peak VRAM | 2.6138 GiB | **2.6138 GiB** (identical) |
| `max_grad_norm` logged | 0.663 | 1.0775 |

Every frozen recipe term was read back out of the run's own `config.json` and
asserted (G5, 24 fields). **No hyperparameter moved.**

**H2b, beside `EXP_015`'s parameter-identical table** — all five at
`d_model = 1481`, seed 0, same recipe, same tree. **No verdict is attached to any
comparison in this table** (§4.0):

| arm @ 5.0M | carried bpc |
|---|---:|
| `gru` anchor (violates I5) | 1.54699 |
| **`twocomp_detach`** | **1.86290** |
| `snn` control | 2.00073 |
| `twocomp` | NaN — diverged 5138 |
| `twocomp_threshold` | NaN — diverged ~2000 |

Marker: the spiking-to-anchor gap at matched size goes 0.45374 → **0.31591**.
`EXP_015` measured width closing **6.6 %** of that gap; this closes **30.4 %** of
it. **n = 1 on both sides and no bar is placed on it.**

### 9.4 Leg C, and the result nobody pre-registered: the anchor mostly died

**Two of the three fresh `twocomp` anchors diverged at `d = 512`** — the width
the arm was adopted at, where the committed record holds **seven** clean seeds.

| run | outcome | carried | `max grad_norm` |
|---|---|---:|---:|
| `anchor_twocomp_d512_s0` | **DIVERGED, step 11500** | NaN | 0.815 |
| `anchor_twocomp_d512_s1` | trains | **2.11766** | 0.564 |
| `anchor_twocomp_d512_s2` | **DIVERGED, step 12500** | NaN | 16.730 |
| `detach_d512_s0` | trains | 2.11769 | — |
| `detach_d512_s1` | trains | 2.11296 | — |
| `detach_d512_s2` | trains | 2.11532 | — |

**H3 therefore resolves NOT RUN**, by the guard §4 wrote for it: a diverged run
resolves NOT RUN, not UNRESOLVED. The seeds are **not replaced**
(`CONTRIBUTING.md` §4). This is `EXP_011`'s `compose_s0` situation a second time,
and this time it took the *control*.

**What may still be said, as markers with no bar:**

* detach at `d = 512`, n = 3: **2.11533 ± 0.00237** (sd), se 0.00137.
* the one surviving anchor: **2.11766**. The committed n = 7 mean: **2.11869**.
* so the detached arm sits **−0.00234** from the surviving anchor and
  **−0.00337** from the committed mean — both inside this design's own stated
  0.005 resolution floor, and **H3a's marker would have read WITHIN BAND** had
  H3 been resolvable (the surviving anchor is 0.00103 from the committed figure,
  which is the tightest agreement any fresh/committed pair in this project has
  produced).
* **σ re-measured** for this structurally new arm: **0.00237, 2 df**, against
  `EXP_005`'s 0.00313 on the two-compartment neuron. A **marker**; 2 df, and
  `CONTRIBUTING.md` §4 forbids promoting it to a project constant.

**Two of three is a rate and it carries its denominator and an interval**
(`CONTRIBUTING.md` §3): the adopted arm diverged **0.667 (2/3), 95 % CI
[0.208, 0.939]**; this arm **0.000 (0/4) across both widths, 95 % CI
[0.000, 0.490]**. Fisher's exact on the 3-vs-3 table at `d = 512` gives
**p = 0.40**. **The stability difference is NOT established by this design**, and
saying so is not a formality — the intervals overlap heavily and four runs cannot
separate them. What *is* established is that the adopted arm produced two more
divergences, bringing Phase 4's total to **five**.

**The post-hoc probe on both dead anchors** (own artifact,
`exp_017_posthoc_anchor_jacobian.json`, labelled post-hoc in the file):

| run | layer | hard `max\|g\|` | hard run | hard window | detach |
|---|---:|---:|---:|---:|---|
| `anchor_..._s0` @ its batch 11500 | 0 | 4.045 | 3 | 10^0.61 | 0.5 / run 0 |
| `anchor_..._s2` @ its batch 11500 | 0 | 4.966 | **21** | **10^4.21** | 0.5 / run 0 |
| `anchor_..._s2` @ its batch 12500 | 0 | 6.183 | 19 | 10^2.39 | 0.5 / run 0 |

**`EXP_016`'s mechanism is present at 735K on the runs that died** — the
expanding window is smaller than at 5.0M (10^4.21 against 10^16.00) but it is
there, and it is not there at all for the plain LIF at any width.

### 9.5 What this says about the adopted arm, and it is not comfortable

`EXP_016` §9.7 said, correctly at the time: *"It does not say the arm is bad.
`twocomp` is the adopted arm, 0.134 bpc ahead of the baseline at 735K across 7
seeds, where its longest expanding run is 3 steps."* §9.4 above narrows that. The
seven clean seeds were trained **before decision #7 replaced the gradient clip**;
retrained on today's tree, the same configuration at the same width lost two of
three. **This is not a claim that the clip caused it** — that is a specific,
testable hypothesis this experiment did not test and it is referred (§10 item 5),
one 9-minute run each way. What is measured is only that the adopted arm's
stability at 735K is **marginal enough to be moved by something**, and that the
bounded arm's, over four runs at two widths, was not moved by the same something.

### 9.6 Leg D: reach, and a verdict that rests on less than it sounds like

| seed | detach | anchor | `Δh` |
|---:|---:|---:|---:|
| 0 | 47 | *(diverged)* | — |
| 1 | 50 | 52 | **−2** |
| 2 | 48 | *(diverged)* | — |
| median | **48** | 52 | **−2** |

**H4 = REACH REDUCED**, reported exactly as the rule fired. And the rule fired on
**one pair**, at `Δh = −2` characters, on a metric with no σ anywhere in this
project — while the three detach seeds alone span **47–50**, a spread larger than
the difference the verdict rests on. §4's guard said in advance that a median Δh
of −1 and of −20 both resolve REDUCED and that the raw values must be quoted;
they are quoted above, and this is what that guard was for.

For context and **not as part of the verdict**: the committed adopted arm's
horizon is **47** (n = 7) and the plain LIF's is **7**. The detached arm's median
48 is one character above the adopted arm's committed figure and nearly seven
times the baseline's. F1 passed on all four runs (residual 6.06e-10 on the one
printed).

**Report a near-miss as a miss** (`CONTRIBUTING.md` §3): the verdict is REDUCED
and is not negotiable. The near-ness is what §10 item 2 refers.

### 9.7 H5 breached, and the breach is in the denominator

| run | `d` | wall-clock | ÷ adopted arm | VRAM |
|---|---:|---:|---:|---:|
| `detach_d512_s0` | 512 | 531.47 s | 0.990 | 0.9801 GiB |
| `detach_d512_s1` | 512 | 541.85 s | 1.010 | 0.9801 GiB |
| `detach_d512_s2` | 512 | 556.19 s | 1.036 | 0.9801 GiB |
| `detach_d1481_s0` | 1481 | 2740.11 s | **1.127** | 2.6138 GiB |

**The bar fired as written and the verdict is BREACH.** Its own wording — *"a
breach raises rather than resolves"* — required the investigation, which is:

1. **H5's pre-registered denominator at `d = 1481` is 2432.24 s, the wall-clock
   of a run that spent 14,862 of its 20,000 steps computing on NaN.** That is not
   a denominator. **This is a defect in H5's design**, written by me, in the same
   file that quotes `CONTRIBUTING.md`'s rule that a ratio needs its reference
   stated — the rule turns out to need a second clause: the reference must also
   be a *valid measurement*.
2. **A controlled paired re-measurement** (`exp_017_posthoc_cost_pair.json`,
   POST-HOC): `014_calibrate_cost.py`, 400 steps per arm, one session, no eval or
   checkpointing inside the timed window, `twocomp` against `twocomp_detach` at
   `d = 1481`. Projected 20k: **2432.2 s vs 2515.7 s → 1.034×**, inside the band.
3. **At `d = 512`, where three fresh anchors were measured in the same session as
   the three detach runs**, the detached arm is **faster**: 543.2 s mean against
   the anchors' 577.1 s (0.941×).
4. **Peak VRAM is identical to the byte at both widths** — 0.9801 and 2.6138 GiB
   — which is the strongest available confirmation that the arm saves the same
   tensors, and the gate holds the kernel count under 1.05 per timestep.

**The verdict stands as BREACH and is not repaired.** The corrected denominator
is **referred** (§10 item 3), on `EXP_012`'s precedent: a pre-registered rule
that returns an unhelpful verdict is reported as it fired.

### 9.8 A mutation escaped, and why it escaped is worth more than the patch

The first campaign run was **58/59**. **D09** — `>` for `>=` in the backward
kernel — **ESCAPED**.

**The bound cannot see this mutation.** At `v_pre == thr` exactly, `>=` gives
`sh = 1` and `g = 0`; `>` gives `sh = 0` and `g = beta_f`. *Both values are inside
`{0, beta_f}`*, so every assertion about `|g| ≤ beta_f` holds exactly as before
while the derivative is wrong. **A bound is a statement about a set, and an
off-by-one in a comparison moves a point within that set.**

That is `CONTRIBUTING.md` §5's rule — *a numerical contract is only guarded where
the quantity it constrains is actually observed* — arriving from a direction it
had not arrived from before: here the unguarded quantity was not an intermediate
nobody read, but *which branch* a value came from. The adopted arm's gate catches
the same mutation (T15) only because it carries an explicitly **constructed**
equality case; random inputs cannot supply one, since exact equality has measure
zero. This file's gate was missing it.

Two legs were added — the branch asserted at the kernel boundary at `v_pre == thr`
and one ulp below it, and the adopted arm's end-to-end construction mirrored — and
**the whole campaign re-run at 59/59**, not just the D block: `model.py` and
`config.py` both changed for this arm, and "those gates do not import them" is an
argument, not a measurement.

### 9.9 Gates

| gate | result |
|---|---|
| **T1, T1b, T2** | HOLD — §9.1 |
| **G1** — the adopted arm is not edited | **PASS**, `git diff` empty over `twocomp.py`, `kernels.py`, `neuron.py`, `surrogate.py` |
| **G2** — parameter identity with `twocomp` | **PASS**, exactly equal at both widths, 7/7 runs |
| **G3** — K1, one thing changed | **PASS**, 7/7 runs |
| **G4** — determinism, 0-difference reproduction | **PASS**, 200 quantities, 0 differing |
| **G5** — no hyperparameter moved | **PASS**, 24 fields per run, read from each run's own `config.json` |
| **G6** — VRAM alarm 4.0 GiB | **PASS**, peak 2.6138 |
| **G7** — a diverged leg is recorded as diverged | **PASS**, both dead anchors marked `completed: false` |
| tests | `tests/test_twocomp_detach_equivalence.py` **30 passed**; fast suite **281+4**; `ruff` clean |

### 9.10 Cost

**~2.0 GPU-hours.** Legs B and C 1.60 h of training; Leg A 0.10 h and its re-run
0.10 h; Leg D 0.05 h; the post-hoc probes and the cost pair 0.05 h; the mutation
campaign twice, 0.12 h. **Nothing was aborted and no GPU-hour bought no number** —
the first campaign run is counted in full and it bought D09.

Project total: **~12.9 of ~30 GPU-hours.**

## 10. Referred to Elliot, and not decided here

*(written before the run; the referrals are structural and do not depend on which
way the bars fall)*

1. **Adoption.** Whatever H3 and H4 return, adopting or ranking this arm is
   Elliot's (report §8). Note that adoption here is unusually cheap to reverse:
   the arm's inference model **is** the adopted arm, so nothing downstream of a
   checkpoint would change.
2. **The soft-reset alternative** (§2.5), derived and not run.
3. **Decision #11's form, a second time.** `EXP_016` §10 item 1 put it in question
   because `dv` contains no recipe term. If H2 SURVIVES, that is a second, more
   direct data point: the failure was fixable **without touching the recipe**.
   Still not an answer to #11 — and still Elliot's.
4. **Whether Phase 5's plan is unblocked.** `EXP_015` said the divergence "blocks
   Phase 5 as conceived" because the plan was to scale the adopted arm. Nothing
   here authorises Phase 5 (#4 is still "not yet"); what it may do is remove one
   stated reason it could not proceed.
5. **Whether `twocomp_threshold` deserves the same treatment.** It died at ~2000
   and is out of scope here (§6 item 6). The composition is the project's best
   model at 735K and its width behaviour is unexplained.

---

### Added after the run — referrals the results created

These are **not** pre-registered and are marked as such. None is decided here.

6. **The two dead anchors are the most important loose end in this experiment,
   and there is a cheap test.** The hypothesis is specific: decision #7's
   `_clip_grad_norm_fp64` fires 3 times in 20,000 steps and changes the
   trajectory (`EXP_014` §9.12 measured +1.97e-03 bpc on the plain LIF), and the
   adopted arm's stability at 735K is marginal enough for that to flip a seed.
   **Testable at ~9 GPU-minutes per run** by retraining `twocomp` at `d = 512`,
   seed 0, with the stock `clip_grad_norm_` monkeypatched back — machinery
   `1450796` already built. **If it is the clip, then the committed seven-seed
   evidence for the adopted arm describes a tree that no longer exists**, which
   is decision **#10**'s substance and bears on **#5** and **#1**. If it is not
   the clip, the arm's 735K stability is worse than the record shows for a
   reason nobody has named. **Not run here: it is a different question from the
   one §1 asks, and running it under this pre-registration would be scope the
   log did not declare.**
7. **H5's denominator (§9.7).** The corrected rule is "the adopted arm's
   wall-clock at the same width, **from a run that did not diverge**" — and at
   `d = 1481` no such run exists, so the fallback has to be a controlled paired
   calibration. Referred, not applied; the breach stands.
8. **H4's bar (§9.6).** A paired horizon difference over 3 seeds, resolved on
   whichever pairs survive, has no power and this experiment now has the number
   to prove it: one pair, `Δh = −2`, against a within-arm spread of 3. Whether
   the horizon deserves a σ — this project has never measured one, at any width,
   for any arm — is a real gap and is not this file's to close.
9. **What A6 means, if anything (§9.2).** The bounded arm trains *inside* the
   region of weight space that kills the unbounded one, and further into it
   (`max|w·vs|` 445.9 against 356.6). Two readings are available and this file
   picks neither: that the reset gradient was the only thing keeping `w·vs`
   small and it was doing so at a cost, or that `w·vs` is simply unconstrained in
   both arms and only one of them notices. **`EXP_006`'s frozen-`beta_s` ladder
   is the instrument that would separate them** and it is still unranked.
10. **What "the adopted arm" now means.** §9.5 is deliberately narrow, but the
    combination — the arm cannot be trained at width, and lost two of three fresh
    seeds at the width it was adopted at — is the sort of thing decision #5's
    owner would want to know before anything is built on top of it. **No
    recommendation is made and no row is edited.**
