# EXP_009 — Is the remaining gap representational or optimisational?

**Pre-registered:** 2026-08-03, before any distilled student was trained.
**Status:** **CLOSED PROVISIONALLY** 2026-08-03, at **n = 2 of a pre-registered
3**: one seed diverged and §9.4 is what happened. X1, X3 and X5 held; **X2 and X4
failed**. The recipe closes **3.2 %** of the remaining gap — a lower bound, and
§1.1 forbids reading its smallness as a representational ceiling.
**Phase:** 4 (controlled experiments).

**This is a labelled diagnostic and never the headline.** Distillation from the
trained GRU is the third of the three I5 boundary questions
`03_phase3_candidates.md` §6.3 referred to Elliot, and it is **still referred**.
It touches no invariant — I1–I5 constrain the architecture, not the training
signal — but it changes the project's headline claim from *trained from scratch*
to *distilled*, which is a scope decision this experiment has no authority over.

Every number here is therefore a **bound**, reported as a bound. No bpc produced
by this experiment may be quoted as the spiking model's score.

---

## 1. Why this experiment

After `EXP_004`, `EXP_005` and `EXP_006`, the adopted two-compartment arm sits
**0.3398 bpc** short of the parameter-matched GRU anchor on the absolute
context curve (0.3513 on the whole-split carried score). Phase 5's entire shape
depends on one fact nobody has measured:

> Is that gap something the architecture **cannot represent**, or something
> gradient descent on hard targets **did not find**?

The two answers point opposite ways. Representational ⇒ Phase 5 should chase
architecture, and the ranked candidate list is the right instrument.
Optimisational ⇒ Phase 5 should chase the training signal — schedule, budget,
targets, initialisation — and the candidate list is largely beside the point.

Distillation bounds it. If the *same* two-compartment network, trained with the
GRU's own output distribution as an additional target, closes a large part of the
gap, then the architecture could represent that much all along and the shortfall
was optimisational. The teacher is already on disk (`gru_s0…s2`), so this costs
three training runs and no new model.

### 1.1 The asymmetry, stated before the run because it decides how to read the result

**A positive result is strong. A null is weak.** If the distilled student closes
a quarter of the gap, that is a constructive demonstration: a network of this
architecture, with these weights available, reaching a score the from-scratch
recipe did not reach. Nothing about the distillation hyperparameters can make
that false.

If it closes nothing, the honest conclusion is *"this distillation recipe did not
help"*, **not** *"the gap is representational"*. λ, the temperature, the schedule
and the teacher choice are all unswept here, and any of them could be the reason.
This experiment therefore **cannot establish a representational ceiling** and the
decision rule in §4 does not let it try.

That asymmetry is why the ceiling language in the title is qualified everywhere
below as a *lower bound on what is optimisational*.

---

## 2. Design

| | |
|---|---|
| Student | `arch="twocomp"`, the adopted arm, **unchanged** — same kernel, same init, same 20 000 steps |
| Teacher | `gru_s{i}` `ckpt_final.pt`, frozen, `eval()`, no grad |
| Pairing | seed `i` student ↔ seed `i` teacher, `i ∈ {0, 1, 2}` |
| Loss | `L = (1 − λ)·CE(student, y) + λ·KL(teacher ‖ student)`, `λ = 0.5`, temperature 1 |
| Reported metric | ordinary cross-entropy test bpc, both protocols — the same number every other arm in this project reports |
| Cost | 3 × ~700 s ≈ **0.6 GPU-hours** |

Both loss terms are in nats per character, so λ mixes two quantities on one
scale. Temperature is 1, which means `KL(teacher ‖ student)` is exactly the extra
information in the teacher's full distribution over the one-hot target — the
thing distillation is supposed to transfer — with no temperature hyperparameter
to tune or to be accused of tuning.

**Seed-matched teachers, not one teacher.** A single teacher for all three
students would make the three runs share an error pattern and would turn three
seeds into something closer to one. Pairing costs nothing (the checkpoints
exist) and keeps the seeds independent.

### 2.1 How the teacher gets into a captured training step

`snn.train.Trainer` captures its whole step into a CUDA graph, and a cuDNN GRU
forward inside that capture is a risk with no upside. The teacher therefore runs
**outside** the captured region, under `no_grad`, writing its log-probabilities
into a pre-allocated static buffer that the captured step reads — the same
discipline `Trainer` already uses for `static_x` and `static_y`. The teacher
forward is a deterministic function of the batch, so nothing about it needs to be
inside the graph.

**This is done by subclassing `Trainer` inside the experiment script, not by
editing `src/snn/train.py`.** A diagnostic that is not going to be adopted must
not modify the trainer every adopted arm shares (protocol: never change the
baseline to make a diagnostic look better). The subclass overrides exactly two
methods and adds one buffer.

### 2.2 What is measured

1. Whole-split test bpc, both protocols, per seed (`scripts/evaluate.py`).
2. The fraction of the 0.3398 asymptotic gap closed.
3. The paired excess curve, the horizon, and the absolute bpc-at-context curve.
4. The gain decomposed by context against **the un-distilled two-compartment
   arm**, cut at the baseline's horizon 7 — so that "where did distillation help"
   is answered on the same basis as every other decomposition in Phase 4.
5. The §6.2 per-context comparison against the baseline **and** against the
   un-distilled arm.

### 2.3 What is deliberately NOT measured

**A λ or temperature sweep.** One recipe, fixed in advance. Sweeping would price
a hyperparameter and would also destroy §1.1's asymmetry argument by making a
null look like a searched-and-failed result when it would still be a
three-point search.

**Distillation into the Phase-2 baseline.** Interesting, and a different
experiment: the question here is about the *adopted* arm's remaining gap.

---

## 3. Hypotheses

Thresholds are on **carried** whole-split test bpc against the committed
seven-seed two-compartment mean **2.11869** (sd 0.00313), at **the
two-compartment family's own bar, 2σ = 0.00627** (`EXP_005`, which closed
candidate #14 for this neuron). The inherited whole-split 2σ = 0.00922 is
reported beside every number and is never mixed into a threshold.

**Power, stated before the run.** 3 distilled seeds against 7 committed ones. If
the distilled arm's σ matches the un-distilled arm's 0.00313, `se(Δ) =
sqrt(0.00313²/7 + 0.00313²/3) = 0.00216`, so the 0.00627 bar sits at **2.9 se**.
The student is the same neuron trained on a different signal, so inheriting the
family's σ is the defensible choice `EXP_005` §9.5 licenses — but the distilled
arm's own 3-seed sd is reported beside it, and a σ *difference* is not claimable
at n = 3.

| # | Prediction | Threshold | Direction |
|---|---|---|---|
| **X1** | **Distillation buys bits.** Mean carried test bpc over 3 seeds ≤ **2.11242** | un-distilled mean − 2σ_family | flatters |
| **X2** | **It buys a material share of what is left.** ≥ **25 %** of the 0.3398 asymptotic gap closed | a quarter is the smallest share that would change Phase 5's shape | flatters |
| **X3** | **It does not buy it by shortening reach.** Median horizon over 3 seeds ≥ **38** | the lowest per-seed horizon among the seven committed un-distilled seeds | flatters |
| **X4** | **The help is not concentrated beyond the horizon.** within-reach gain over the un-distilled arm > beyond-horizon gain | the un-distilled arm's remaining gap is 51.8 % within-reach, so a teacher that helps should help there | **stated against the plan** |
| **X5** | **It does not close the gap.** Mean carried test bpc ≥ **1.85** | ~0.08 above the GRU's own 1.76741; a student that matched its teacher would mean the anchor is reachable and the I5 question is answered, which no result in this project anticipates | **stated against the plan** |

**On X5.** It is written down because it is the outcome that would matter most
and is least expected, and because a project that only pre-registers thresholds
it expects to clear is not pre-registering anything. If X5 fails, that is the
Phase-4 headline and everything else in this file is a footnote to it.

---

## 4. Decision rule, fixed now

| X1 | X2 | Verdict |
|---|---|---|
| **holds** | **holds** | **At least 25 % of the remaining gap is optimisational, demonstrated constructively.** Phase 5 must carry a training-signal arm, and the ranked architecture candidates are re-priced against a denominator that is smaller than 0.3398. The number is a **lower bound**: it is what *this* recipe reached, not what is reachable. |
| **holds** | fails | **Some of the gap is optimisational; less than a quarter of it by this route.** Both numbers reported. The architecture ranking stands. |
| **fails** | — | **This recipe did not help.** Recorded as a **null of the recipe, not of the hypothesis** (§1.1). It is explicitly **not** evidence that the gap is representational, and §9 will say so in those words. The follow-up — a λ sweep, or a longer budget — is named and not run. |

**Rider, fixed now.** Whatever this experiment produces, no distilled bpc enters
the project's headline table, the Phase-4 report's adopted-arm row, or any
comparison against the literature anchors. It appears only in a section labelled
as a diagnostic, beside the un-distilled number.

---

## 5. Known limitations, stated in advance

1. **A null is uninformative about the hypothesis** (§1.1). This is the single
   most important thing to carry out of this file.
2. **One recipe.** λ = 0.5, T = 1, no schedule on λ, same 20 000 steps.
3. **The teacher is a 738 019-parameter GRU trained under this protocol**, not a
   strong external model. The bound is relative to *this* anchor.
4. **Three seeds**, and σ inherited from the family rather than re-measured for
   the distilled arm (§3). The inheritance is licensed by the student being the
   same neuron; it is stated, not assumed silently.
5. **The teacher runs at fp32 in `eval()` on the same batches**, so it sees no
   dropout and no augmentation — there is none in this project — and its outputs
   are a deterministic function of the batch. Nothing here re-derives the
   teacher's own score except Y2, which requires it to reproduce.
6. **`EXP_002`'s systems numbers do not cover this.** A distilled run costs a
   teacher forward per step and is measured, not assumed, in §9.

---

## 6. Self-checks — aborting, not advisory

| # | Check | Why it can fail |
|---|---|---|
| **Y1** | With `λ = 0`, one training step of the subclassed trainer is **bitwise identical** to `snn.train.Trainer`'s on the same seed and batch — same loss, same grad-norm, same parameters afterwards | the subclass must add a term, not change the trainer; a drifted step would make the comparison against the committed un-distilled seeds meaningless |
| **Y2** | The teacher path reproduces `gru_s{i}`'s committed `final_test.json` **fresh** bpc to < 1e-3, scored through this harness's own teacher forward | reproduces a number this harness did not produce; catches a mis-loaded teacher, a vocab mismatch, or a teacher accidentally left in `train()` |
| **Y3** | The teacher's parameters have `requires_grad = False` and are bitwise unchanged between the first and last step | a teacher that is silently being trained is a different experiment |
| **Y4** | The student's `config.json` differs from `twocomp_s{i}`'s only in `seed`, `run_name` and the distillation fields | "the adopted arm on a different signal" is a claim about sameness |
| **Y5** | The horizon probe's `k = L` point reproduces each distilled run's committed `final_test.json` fresh bpc to < 1e-3 (`EXP_001`'s F1) | the Phase-3 contamination check |
| **Y6** | No mutation campaign lockfile at the start of **every** run | Phase-3 report §8 |

---

## 7. Failure modes

| # | Failure | Detection |
|---|---|---|
| **Q1** | Teacher and student see different batches | the teacher is fed `static_x`, the identical buffer the student's forward reads; Y2 would not catch this, so it is structural rather than checked |
| **Q2** | The KL is computed against logits instead of log-probabilities, or with the arguments swapped | `KL(teacher ‖ student)` is asserted against a hand-computed reference on a small tensor in the tests |
| **Q3** | The teacher leaks gradients into the student's optimiser | Y3, and the teacher's parameters are never passed to the optimiser |
| **Q4** | The distilled score is quoted as the project's result | §4's rider, and the results table labels every row |
| **Q5** | The teacher forward is captured into the CUDA graph and replays stale logits | the buffer is filled outside the captured region on every step; Y1 with λ = 0 would not catch it, so a dedicated test asserts the buffer changes between consecutive steps |
| **Q6** | A null is read as a representational ceiling | §1.1, §4's third row, and §9 will restate it |

---

## 8. Entry conditions

- [x] Existing test suite green — **242 passed** (2026-08-03)
- [x] Working tree clean at `1e5b239`, branch `phase-4-experiments`
- [x] No mutation campaign running
- [x] Nothing in `src/snn/` modified by this experiment (§2.1) — the trainer is
      subclassed in `scripts/exp/009_distill.py`
- [x] This file committed **before** the harness is written — `1f05bab`, against
      the harness's `5d6f88a`

### 8.1 The smoke run, 2026-08-03 — all four checks discharged before the sweep

Three steps at `lambda = 0.5` on seed 0, plus the two checks that cannot be made
by the sweep itself:

| Check | Result |
|---|---|
| **Y1** | at `lambda = 0` the subclass steps **bitwise** like `Trainer` — identical loss and grad-norm on all three steps, **and identical parameters afterwards** |
| **Y2** | the teacher scores **1.802617** through this harness against `gru_s0`'s committed **1.802617** — residual **0.00e+00** |
| **Y3** | the teacher's parameters unchanged |
| **Q5** | the teacher log-probability buffer differs on every consecutive step, so it is not a stale value baked into the captured graph |

Y1 was checked on parameters as well as on losses deliberately: two trainers can
report the same loss and still have stepped differently, and it is the parameters
that carry the difference into the next 19 997 steps.

### 8.2 One refinement, made before any result was read

The driver skips runs that are already complete and deletes partial ones, on the
same rule and for the same reason as `EXP_007` §8.1 — the first launch of the
chain was killed by a harness timeout. Completeness here additionally requires
`distill.json`, which is written only after Y2 and Y3 have passed for that run, so
a run can never be skipped on the strength of artifacts that predate its own
self-checks.

---

## 9. Results

**Run 2026-08-03.** Three seeds launched, **two completed**; `twocomp_distill_s1`
diverged at step 12 497 of 20 000 and is §9.4. **~0.71 GPU-hours.** Raw JSON:
`docs/reports/data/exp_009_run_manifest.json`,
`docs/reports/data/exp_009_divergence.json`,
`docs/reports/data/exp_007_009_arm_results.json`.

**Every verdict below is PROVISIONAL at n = 2.** §3 fixed n = 3 and it was not
met. §9.4 records why the shortfall was not made up.

### 9.1 The bound, which is small

| | un-distilled twocomp (n=7) | **distilled (n=2)** | Δ |
|---|---:|---:|---:|
| test bpc, carried | 2.11869 | **2.10921 ± 0.00190** | **−0.00948** |
| test bpc, fresh | 2.14566 | **2.13562 ± 0.00178** | −0.01005 |
| memory horizon | 47 (median of 7) | 40, 47 | — |
| wall-clock, 20 000 steps | 536.7 s | **849 s (1.58×)** | |

−0.00948 clears the family's own bar (2σ = 0.00627) at **5.3 standard errors**, so
distillation does buy bits. What it does not buy is much of the gap:

| | bpc |
|---|---:|
| asymptotic gap to the GRU anchor, before | **0.3398** |
| asymptotic gap to the GRU anchor, after | **0.3291** |
| **share closed** | **3.2 %** |

**X2 required 25 %.** The teacher's own distribution, handed to the student on
every one of 20 000 batches, moved the student 3.2 % of the way toward it.

### 9.2 The predictions, resolved as written

| # | Prediction | Threshold | Measured | Verdict |
|---|---|---|---|---|
| **X1** | mean carried improves by > 2σ_family | 2.11242 | **2.10921** | **held** |
| **X2** | closes ≥ 25 % of the asymptotic gap | 0.25 | **0.032** | **FAILED** |
| **X3** | median horizon ≥ 38 | 38 | **43.5** | **held** |
| **X4** | within-reach gain > beyond-horizon gain | — | **−0.0131 vs −0.0004** | **FAILED** |
| **X5** | mean carried ≥ 1.85 | 1.85 | **2.10921** | **held** |

**Decision cell (§4): X1 holds, X2 fails ⇒ "Some of the gap is optimisational;
less than a quarter of it by this route. The architecture ranking stands."**

**X4 failed, and it was one of the two stated against the plan.** Decomposing the
distilled arm's gain over the un-distilled one:

| Component | bpc | share |
|---|---:|---:|
| At zero context | +0.0243 | 226.3 % |
| Within the 7-character reach | **−0.0131** | −122.3 % |
| Beyond the horizon | −0.0004 | −4.0 % |
| **Total** | **+0.0107** | 100 % |

So the teacher's signal helped the student at the one context where the teacher
has no memory advantage at all, and **hurt it inside the reach** — the component
that is 51.8 % of the gap and the reason this project cares. That is the third
Phase-4 arm in a row whose gain turns out to live at zero context (`EXP_007`
§9.1, `EXP_008` §9.2), and the pattern is now the most consistent thing in the
phase.

### 9.3 What §1.1 forbids concluding, restated because it is the whole point

X2 failed. **That does not establish that the remaining gap is
representational**, and this file's §1.1 fixed that reading before the run:

> If it closes nothing, the honest conclusion is *"this distillation recipe did
> not help"*, **not** *"the gap is representational"*. λ, the temperature, the
> schedule and the teacher choice are all unswept here, and any of them could be
> the reason.

What **is** established, constructively and at 5.3 se, is a **lower bound**: at
least 3.2 % of the asymptotic gap is reachable by this architecture under a
different training signal, with no architectural change. A lower bound of 3.2 %
is a weak bound. It is not zero.

The follow-up that would tighten it is named and **not run**: a λ sweep
(λ ∈ {0.1, 0.9}), a λ schedule, or a longer budget. §2.3 excluded a sweep
deliberately, and running one now — after seeing the answer — would convert a
pre-registered null into a searched-until-it-worked result.

### 9.4 One seed of three diverged, and it is a finding rather than an accident

`twocomp_distill_s1` trained cleanly to step 12 250 — loss 1.5495, grad-norm
0.2725, firing rates 0.389/0.396 — and by step 12 500 every logged quantity was
NaN and stayed NaN for the remaining 7 500 steps. Its `final_test.json` is NaN.

`scripts/exp/009_chase_divergence.py` replays the run from its last healthy
checkpoint (step 10 000, whose parameters sit **inside** the range the seven
un-distilled seeds occupy: `|w|` max 1.48 against their 1.73, `beta_s` max 0.9863
against their 0.9855). **The divergence reproduces exactly**, so it is a
deterministic function of (checkpoint, step) and not a hardware event.

| step | forward | ce | kl | max \|grad\| | ‖g‖ in fp32 | ‖g‖ in fp64 |
|---:|---|---:|---:|---:|---:|---:|
| 12 497 | finite, \|logits\| max 38.9 | 1.5047 | 0.2886 | **5.46e31** | **inf** | 5.833e31 |
| 12 498 | finite, \|logits\| max 39.7 | 1.5034 | 0.3035 | **NaN** | NaN | NaN |

**Three things follow, and none of them is what the log looked like.**

1. **The forward never overflowed.** At both steps the logits and the carried
   state are finite and the loss is *healthy*. This is not a membrane blow-up.
2. **The first symptom is not a NaN.** At 12 497 every individual gradient is
   finite and the largest is 5.46e31. `clip_grad_norm_` accumulates the sum of
   squares in fp32, and 5.46e31 squared overflows — so the norm is `inf` in fp32
   and 5.833e31 in fp64. **The gradient clip that exists to prevent exactly this
   is the component that fails first**, and because the branchless implementation
   computes `clamp(max_norm/(inf + 1e-6), max=1.0) = 0`, it *zeroes* the update
   instead of rescaling it.
3. **The NaN is generated in layer 0's two-compartment backward.** At 12 498 the
   non-finite gradients are exactly `embed.weight`, `layers.0.weight`,
   `layers.0.bias`, `w.0` and `beta_s_raw.0`. Layer 1's and the head's are finite.
   Backpropagation runs head → layer 1 → layer 0 → embed, so a NaN confined to
   layer 0 and below **was produced inside layer 0's scan**, not inherited.

Once any gradient is non-finite the same clip multiplies every gradient by NaN,
so one bad step is permanent. That is read off the committed trainer, not measured
here, and it explains the 7 500 dead steps rather than the first one.

**What this says about the adopted arm, carefully.** The exploding quantity is the
BPTT gradient through the reset-shielded two-compartment scan — a property of the
**adopted neuron**, surfaced by a change of training signal rather than caused by
one. Seven un-distilled seeds did not hit it, so nothing here shows the adopted
arm is unstable in its committed configuration. What it does show is that **the
arm has a gradient-explosion mode that the project's gradient clip cannot catch,
because the clip's own fp32 accumulator overflows before the gradient does.**

**Two changes suggest themselves and neither is applied**, both because they touch
committed Phase-2 code and both because they would flatter the work:

* accumulate the clip norm in fp64, so the clip does its job at 5e31 rather than
  reporting `inf`;
* make a non-finite gradient norm abort or skip the step rather than silently
  zeroing it, which would have turned 7 500 dead steps into one logged event.

Both are **referred to Elliot**. `EXP_005` §9.4's precedent applies: measure it,
report it, refer the change.

### 9.5 Why the diverged seed was not replaced

The obvious repair is to train a fourth seed. It was not done, for two reasons
fixed before looking at what a fourth seed would have said:

* §2 pairs student seed `i` with teacher `gru_s{i}` and **there is no `gru_s3`.**
  A replacement would reuse a teacher and break the design that keeps the seeds
  independent — a second variable introduced to repair a first.
* **Swapping a diverged seed for a fresh one is how a failure leaves the record.**
  The divergence is the most transferable thing this experiment produced; a
  three-seed table with it quietly absent would be worth less than a two-seed
  table with it in §9.4.

So: n = 2, said plainly, with every verdict marked provisional, and the
`excluded_runs` rule written into `scripts/exp/010_phase4_arm_results.py`'s
registry with its reasoning rather than applied silently.

### 9.6 Self-checks

| # | Check | Result |
|---|---|---|
| **Y1** | at λ = 0 the subclass steps bitwise like `Trainer` | **held** — identical losses, grad-norms and parameters over 3 steps |
| **Y2** | the teacher reproduces its committed fresh test bpc | **3 / 3**, residual **0.00e+00** on every seed |
| **Y3** | the teacher is unchanged after training | **3 / 3** |
| **Y4** | the student's config differs from `twocomp_s{i}`'s only in the allowed fields | **3 / 3** |
| **Y5** | the horizon probe's k = L point reproduces each run's committed fresh bpc | **2 / 2** (the diverged run was not probed) |
| **Y6** | no mutation campaign lockfile before every run | **3 / 3** |

**Y2 is the one worth noting.** The teacher scored **1.802617** through this
harness against `gru_s0`'s independently committed **1.802617** — residual exactly
zero, on all three seeds. Whatever went wrong in seed 1, it was not the teacher.

### 9.7 Status

**CLOSED, PROVISIONALLY, at n = 2. X1, X3 and X5 held; X2 and X4 failed.**

1. **A lower bound: ≥ 3.2 % of the remaining 0.3398 bpc gap is optimisational**,
   demonstrated constructively at 5.3 se. Small, and not zero.
2. **X2's failure does not establish a representational ceiling** (§1.1, §9.3).
   The recipe was one point in a space this experiment deliberately did not sweep.
3. **Distillation's help is at zero context and it *hurts* within the reach**
   (§9.2) — the third Phase-4 arm in a row to land at zero context.
4. **One seed in three diverged, deterministically, and the clip is what failed
   first** (§9.4). The exploding gradient is in the adopted neuron's backward, not
   in this recipe's forward, and the project's gradient clip cannot see it.
5. **No number in this file may be quoted as the spiking model's score** (§4's
   rider). The distilled arm does not appear in the Phase-4 report's adopted row.
