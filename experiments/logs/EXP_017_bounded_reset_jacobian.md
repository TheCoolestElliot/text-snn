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
