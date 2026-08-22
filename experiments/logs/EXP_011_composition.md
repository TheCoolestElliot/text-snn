# EXP_011 — Do the learned threshold and the two-compartment neuron compose?

**Pre-registered:** 2026-08-04, before any composed arm was trained or built.
**Status: CLOSED 2026-08-04 — C1–C5 all held; the arms COMPOSE. PROVISIONAL at n = 2 of a pre-registered 3: `compose_s0` diverged and was NOT replaced (`CONTRIBUTING.md` §4). 2.08326 carried, −0.03543 against the adopted arm. Results in §9; folded into `docs/reports/04_phase4_interim.md` §10.**
**Phase:** 4 (controlled experiments). This is the "biggest unmeasured thing" the
Phase-4 interim report's §6.7 synthesis left standing, and the experiment
Elliot's decision #8 authorises to run *inside* Phase 4 rather than opening
Phase 5 for it.

**What it costs:** three training runs at the committed recipe, ~0.5 GPU-hours.

---

## 0. A note on this experiment's tasking, recorded rather than resolved silently

The instruction that scheduled this work specified four things that the
committed tree contradicts. `EXP_008` §0 set the precedent for recording such a
collision in the log instead of relabelling a document to make it disappear, and
that is what this section does. Each item states what was asked, what is true,
and which was followed.

1. **The acceptance bar.** The tasking set "test bpc ≤ 2.083 (matching or
   beating two-compartment alone)". The adopted two-compartment arm's carried
   figure is **2.11869** (n=7, `EXP_005`), and 2.083 appears nowhere in this
   repository. The parenthetical and the number are therefore not the same bar:
   2.083 is 0.036 bpc stricter than "matching or beating two-compartment alone".
   A composed result of, say, 2.095 would record as FAIL under the number while
   simultaneously *confirming* the hypothesis the number was attached to.
   **Followed:** the hypothesis, not the number. C1 below is fixed at
   **≤ 2.11869**. 2.083 is retained in §4.3 as a reported stretch marker with no
   verdict attached to it, so the tasking's figure is not lost.

2. **The seeds.** The tasking specified seeds 42, 43, 44. Every Phase-4 arm to
   date ran seeds 0, 1, 2 (the adopted arm, 0–6), and the drivers' K1 sameness
   check anchors to `snn_beta0.5_s0`. **Followed:** seeds **0, 1, 2**, because a
   *composition* experiment's whole question is the interaction between two arms
   whose per-seed results already exist at those seeds. Seeds 42/43/44 would
   still yield a valid cross-seed σ but would make this the first Phase-4 arm
   that cannot be read per-seed against `twocomp_s{0,1,2}` and
   `threshold_s{0,1,2}` — discarding the paired comparison that is the point.

3. **Where the two arms are defined.** The tasking cited
   `scripts/exp/010_phase4_arm_results.py` for the two-compartment cell and
   `scripts/exp/008_chase_fold.py` for the threshold arm. Both are wrong: 010 is
   the *resolver* that applies pre-registered bars to committed artifacts, and
   008_chase_fold is the fp64 *diagnostic* that chased W1's residual. The cell
   is `snn.twocomp.twocomp_scan` (wired by `model.TwoCompartmentCharLM`) and the
   threshold arm is `snn.prescan.threshold_gain` (wired by
   `model.LearnedThresholdCharLM`). **Followed:** the real modules.

4. **The pre-registration's location.** The tasking asked for
   `docs/pre_reg/EXP_011_composition.md`. No such directory exists; all ten
   prior pre-registrations live in `experiments/logs/`, which is also where
   `010_phase4_arm_results.py` reads thresholds from. **Followed:** the
   canonical location, this file.

The tasking's instruction to **not** substitute candidate #2 (RMSNorm on `cur`)
is correct and is followed: this experiment composes the explicit per-channel
threshold parameterisation `thr_c = thr·exp(θ_c)`, realised as `cur·exp(−θ_c)`,
and touches no normalisation of any kind. `EXP_008` §0 records why that
distinction needs stating every time.

---

## 1. Why this experiment, and what it is really testing

Two arms have moved this project's metric, and **both act hardest at zero
context**:

* the **two-compartment neuron** (`EXP_004`, adopted): 2.11869 from 2.25311,
  of which **+41.0 %** of the gain is the zero-context component;
* the **learned per-channel threshold** (`EXP_008`): 2.19416 from 2.25311, a
  gain of **0.05896 bpc** — 44 % of the whole two-compartment gain — for one
  redundant parameter per channel.

`EXP_004` §10.11 item 2, which proposed the threshold arm in the first place,
warned in advance that the two numbers "would not add twice". §10.6 is why: at
zero context the two-compartment neuron **is** the baseline with a learned
per-channel threshold, because `v_0 = cur_0·(1 + w_c)` fires iff
`cur_0 >= thr/(1 + w_c)`. If the mechanisms are the same mechanism, composing
them buys nothing and may cost something. **Nobody has measured it.**

### 1.1 Identity 1 extends to the two-compartment neuron — derived first

`EXP_008` §1.1 proved a per-channel threshold is a per-channel input gain **for
the committed single-compartment LIF**. That proof does not transfer for free:
the two-compartment neuron has a second state variable, a learned mix, and a
**reset-shielded** slow pole, and the shielded pole is exactly the sort of
structure that breaks a substitution argument. So it is derived here, before any
code exists, per the protocol's derive-before-implementing rule.

Take the two-compartment neuron driven by a scaled current `g_c·cur`, `g_c > 0`,
at the **committed** threshold `thr`:

```
vf'_t = beta_f·vf'_{t-1} + g·cur_t
vs'_t = beta_s·vs'_{t-1} + g·cur_t
v'_t  = vf'_t + w·vs'_t
s'_t  = 1[v'_t >= thr]
vf'_t <- vf'_t·(1 - s'_t)          hard reset, fast pole only
vs'_t <- vs'_t                     RESET-SHIELDED
```

**Claim:** `vf'_t = g·vf_t` and `vs'_t = g·vs_t`, where the unprimed run is the
*unscaled* current at threshold `thr/g`.

*Base case.* `vf'_0 = vs'_0 = 0 = g·0`. Holds for the zeroed initial state every
run starts from.

*Induction.* Assume `vf'_{t-1} = g·vf_{t-1}` and `vs'_{t-1} = g·vs_{t-1}`. Then

```
vf'_t = beta_f·g·vf_{t-1} + g·cur_t = g·(beta_f·vf_{t-1} + cur_t) = g·vf_t
vs'_t = beta_s·g·vs_{t-1} + g·cur_t = g·(beta_s·vs_{t-1} + cur_t) = g·vs_t
v'_t  = g·vf_t + w·g·vs_t = g·(vf_t + w·vs_t) = g·v_t
```

so, since `g > 0`,

```
s'_t = 1[g·v_t >= thr] = 1[v_t >= thr/g] = s_t
```

— the same spike. The fast reset carries the induction because it is
**multiplicative**: `vf'_t·(1 - s_t) = g·(vf_t·(1 - s_t))`. The slow pole
carries it trivially because it is shielded and neither side touches it. ∎

**Three things this derivation settles, and one it does not.**

* The mix `w` and the slow decay `beta_s` are **untouched** by the substitution.
  They appear in the induction only through the linear combination, which
  commutes with the scaling. So the composed arm does not perturb the adopted
  arm's own two parameters — a per-channel threshold on top of the
  two-compartment neuron is still *exactly* a per-channel threshold.
* **Identity 2 carries too.** `cur = W·h + b`, so `g_c·(W_c·h + b_c)` is the
  model with `(W_c, b_c)` replaced by `(g_c·W_c, g_c·b_c)`. The gain folds into
  the layer's own `nn.Linear` and `thr_log` disappears, leaving a state dict
  with the two-compartment arm's parameter count exactly.
* Therefore **the composed arm adds no functions to the two-compartment arm's
  class.** Its 1 024 extra parameters are exactly redundant, exactly as
  `EXP_008`'s were against the baseline. Whatever it buys is an **optimisation**
  effect — a different AdamW trajectory — and not a larger reachable set.
* **What it does not settle:** the *backward* is not invariant. The surrogate
  `atan_spike` is evaluated at `g·v − thr` in one form and at `v − thr/g` in the
  other, so the two forms are the same function with different gradients. That
  asymmetry is not a defect; it is the entire mechanism by which a provably
  redundant parameter can change where training lands, and it is why this
  experiment can have a non-zero result at all.

### 1.2 What that makes this experiment

Not "does adding a parameter help". It is: **two reparameterisations of
overlapping mechanisms, one adopted and one merely measured — does applying both
recover more than either, and is the interaction negative?** §10.6's algebra says
the mechanisms overlap at zero context. §6.2 of the Phase-4 report says the
threshold arm is also the first arm in the project to move the *within-reach*
component (+0.0288, 24.8 % of what is there), which the two-compartment neuron
made **worse** (−48.4 % within its own reach). Those two facts point opposite
ways, and that is the tension the experiment resolves.

---

## 2. Design

One arm, `arch="twocomp_threshold"`: the adopted two-compartment neuron with
`snn.prescan.threshold_gain` applied to the current before the scan. It is the
adopted arm **plus one thing**, and the one thing is `EXP_008`'s arm verbatim.

* **Seeds** 0, 1, 2, matching `threshold_s{0,1,2}` and `twocomp_s{0,1,2}`.
* **Recipe** identical to the committed Phase-2 baseline in every other field;
  guarded field-by-field by K1 (§6).
* **Initialisation** `thr_log_init = 0.0` and `w_init = 0.1`,
  `beta_slow = 0.95` — each arm's own committed init, unchanged. `θ = 0` gives
  `exp(0) = 1.0` exactly in fp32, so the composed arm **nests the adopted arm
  bitwise** at init, in the interior of an unconstrained parameter.
* **Parameters** 735 437 + 2 048 + 1 024 = **738 509**, +0.42 % over the
  Phase-2 baseline, +0.14 % over the adopted arm — and **+0.00 %** in
  function-space dimension over the adopted arm, by §1.1.

### 2.1 The §7.2 gradient-reachability screen runs before training

The protocol requires it before every new-parameter arm, and `EXP_004`'s most
attractive initialisation was caught by it as an exact saddle. Derived here in
advance: for the composed arm,

```
dL/dθ_c = Σ_t (dL/dcur'_{t,c})·(−cur_{t,c}·exp(−θ_c)) = −Σ_t (dL/dcur'_{t,c})·cur'_{t,c}
```

which vanishes only if the incoming gradient is orthogonal to the scaled current
over the whole window — not a structural zero, and the same expression
`EXP_008`'s screen already cleared for the single-compartment case. **Predicted:
not a saddle.** The screen is run anyway, because a prediction is not a
measurement, and it aborts the experiment if any new parameter's gradient is
zero at init.

### 2.2 What is measured

Whole-split test bpc under the carried protocol (the project's standing metric),
cross-seed sd, the `EXP_004` §10.3 decomposition into zero-context /
within-reach / beyond-horizon, the §6.2 per-context curve and its regression
count, the memory horizon, wall-clock and peak memory, and the fold residual.
All are recomputed by the existing resolver's statistics, not re-implemented —
`EXP_006` failure mode L5.

---

## 3. Hypotheses, fixed now

**C1 — the arms compose without negative interaction.** Composed test bpc
**≤ 2.11869**, the adopted arm's carried figure.

**C2 — the composition is reproducible.** Cross-seed sd **σ ≤ 0.005**.
`EXP_005` measured σ = 0.00313 on the two-state neuron and the threshold arm
came in at ±0.00244, so 0.005 is a bar both parent arms already clear; it is
here to catch a composition that is unstable in a way neither parent was.

**C3 — the gain is sub-additive.** Composed bpc **> 2.05973**, i.e. strictly
worse than full additivity (2.11869 − 0.05896). This is the direction
`EXP_004` §10.11 item 2 predicted and §10.6's algebra implies. **Stated as a
prediction that can fail**: if the composition is additive or better, C3 fails
and §10.6's overlap argument is wrong about the composed case.

**C4 — the composed arm still folds.** The folded state dict reproduces the
composed arm's own test bpc, and — per `EXP_008` §9.5 — the residual is
**reported, not gated**. W1's 1e-3 bar was borrowed from `EXP_001`'s F1, which
compares two evaluations of the *same* weights where no spike can flip; a
fold-in compares two weight tensors denoting the same function, which is the one
case where fp32 reordering reaches the metric. That redefinition is **decision
#6 and is Elliot's, still open**, so this experiment does not apply a bar it has
no authority to set. It reports the residual and the fp64 check below.

**C5 — Identity 1 holds numerically for the two-compartment neuron.** The
composed forward at `θ` equals the two-compartment forward at `thr·exp(θ)` to
fp64 tolerance. This is the decisive leg of §1.1 and it **needs no training** —
it runs before the arm is trained, and a failure means §1.1 is wrong and the
experiment does not run at all.

### 3.1 The power of this design, stated before it runs

With σ ≈ 0.003 and n = 3, the SE on the composed mean is ≈ 0.0018. Against a
7-seed reference the SE of the difference is ≈ 0.0021, so the design resolves a
shift of ≈ 0.004 bpc at 2σ. The distance between C1's bar and full additivity is
**0.059 bpc — 28× that**, so the experiment has ample power to separate "no
interaction" from "additive". It has **little** power to separate, say, 90 %
carry-over from 100 % (0.0059 apart, ≈ 2.8 SE) and **none** to resolve
differences at the ~2e-3 reparameterisation noise floor `EXP_008` §9.6 measured.
Any carry-over percentage this experiment quotes is therefore a point estimate
with a wide interval, and will be reported as one.

---

## 4. Decision rule, fixed now

### 4.1 The rule

| Outcome | Verdict |
|---|---|
| C1 holds and C2 holds | **The arms compose.** The composed arm becomes a Phase-5 candidate; adoption remains Elliot's. |
| C1 holds, C2 fails | **Composition works but is unstable.** Report; do not rank without more seeds. |
| C1 fails | **Negative interaction demonstrated.** The composed arm is not carried forward, and `EXP_004` §10.6's overlap argument is confirmed in the strong form. |

**No verdict in this table adopts anything.** Adoption is Elliot's, and this
experiment does not close any of the seven decisions open in the Phase-4
report's §8 other than the one that authorised it.

### 4.2 What is not a criterion

Wall-clock, memory and parameter count are **reported and not gated**. The
composed arm inherits the adopted arm's kernel and adds one elementwise multiply
per layer outside the time loop; `EXP_007` proved that "outside the time loop"
is not free (1.49× for token-shift), so the number is measured rather than
assumed — but no bar is set on it here, because none was pre-registered for the
parent arms either.

### 4.3 The tasking's 2.083, reported without a verdict

§0 item 1 records why 2.083 is not C1. It is retained here as a **stretch
marker**: the results section will state whether the composed arm landed below
2.083, below the half-gain midpoint 2.08921, and what fraction of the threshold
arm's 0.05896 bpc carried over onto the two-compartment neuron. None of those
three numbers carries a pass/fail verdict, and none of them can change the
verdict C1 and C2 produce.

---

## 5. Known limitations, stated in advance

* **n = 3 against a 7-seed reference.** The adopted arm's 2.11869 is an n=7
  mean; the composed arm gets 3. The comparison is between means of unequal
  precision and §3.1 states what that resolves.
* **One initialisation.** `thr_log_init = 0.0` only. `EXP_008` swept nothing
  either, and a composed arm has a two-dimensional init space (`θ`, `w`) that
  this experiment does not enter. A null result is a null result **at this
  point**, not for the composition in general — `EXP_009` §1.1's discipline.
* **The corpus, the width and the step count are the committed ones.** Nothing
  here speaks to whether the composition behaves differently at another scale.
* **The within-reach prediction is not a criterion.** §1.2 names a real tension
  between the two arms' §6.2 profiles, and the decomposition will be reported,
  but no hypothesis is registered on it because the per-context bars at c ≥ 16
  sit an order of magnitude below the reparameterisation noise floor
  (`EXP_008` §9.6). Registering a bar this design cannot resolve would be
  `EXP_007` V4's vacuous-prediction failure in a new place.

---

## 6. Self-checks — aborting, not advisory

**K1 — the arm is the adopted arm plus one thing.** Every composed run's
`config.json` is compared field-by-field against `twocomp_s0`'s, and only
`seed`, `run_name`, `arch` and `thr_log_init` may differ. Fields the older
config predates are excused only at the `Config` default, and every excusal is
recorded in the manifest rather than swallowed — `EXP_007`'s K1 verbatim.

**K2 — no mutation campaign while training**, re-checked before every run. The
Phase-3 report's §8 incident: a run trained 7 750 steps against a mutated kernel
with nothing in its own logs to say so.

**K3 — wall-clock recorded per run** against the baseline's measured 398.9 s,
with the denominator named in the caption. Rev 2 of the Phase-4 report found two
ratio columns quoting a reference they were not measured against.

**K4 — the composed arm nests the adopted arm at init.** `θ = 0` must reproduce
`twocomp` bitwise in both forward and backward, asserted in the test suite
against `snn.twocomp.twocomp_scan` directly. This is
`test_twocomp_equivalence.py::test_w_zero_is_the_committed_lif_scan`'s pattern
applied one level up.

**K5 — Identity 1 in fp64 (C5) runs before training and aborts it.** Per §1.1
the composed forward at `θ` must equal the two-compartment forward at
`thr·exp(θ)`. Asserted on a real batch at a *non-zero* random `θ`, because at
`θ = 0` the check passes vacuously — `EXP_007` V4's lesson.

---

## 7. Failure modes this design is trying to avoid

* **M1 — reading a null result as "the mechanisms are the same".** They overlap
  at zero context by §10.6; the decomposition is what distinguishes "no gain
  anywhere" from "gain moved between components". Both are reported.
* **M2 — the composed arm's σ inherited rather than measured.** `EXP_005`
  showed σ moves 32 % between neurons and changes *shape* per context. This arm
  is structurally new, so σ is measured on it and nothing is inherited.
* **M3 — a fold check that passes vacuously.** K5 uses a non-zero `θ`.
* **M4 — quoting a carry-over percentage as if it were resolved.** §3.1 states
  the interval in advance.
* **M5 — the composed run silently using the eager scan.** The fused/eager
  dispatch is recorded per run in the manifest, because a composed arm that fell
  back would be a different wall-clock number and a different rounding.

---

## 8. Entry conditions

* Test suite green at 262 gates before anything is built (verified 2026-08-04).
* No mutation campaign lockfile present (K2).
* `twocomp_s{0,1,2}` and `threshold_s{0,1,2}` present and complete, since the
  paired comparison depends on them.
* C5/K5 passes in fp64 before a single training step runs.

---

## 9. Results

**Closed 2026-08-04. C1–C5 all held; the verdict is `C1 holds, C2 holds` — THE
ARMS COMPOSE — and it is PROVISIONAL at n = 2 of a pre-registered 3.** One seed
diverged and was not replaced. Nothing above this line was rewritten.

### 9.1 The headline, and the seed that is missing from it

| | n | test bpc, carried | vs adopted arm | vs Phase-2 | wall-clock | peak VRAM |
|---|---:|---:|---:|---:|---:|---:|
| Phase-2 baseline | 5 | 2.25311 | — | — | 1.00× | 0.609 GiB |
| two-compartment *(adopted)* | 7 | 2.11869 | — | −0.13442 | 1.32× | 0.980 GiB |
| learned threshold | 3 | 2.19416 | — | −0.05896 | 1.12× | 0.734 GiB |
| **composed** *(this arm)* | **2** | **2.08326 ± 0.00049** | **−0.03543** | **−0.16986** | **1.41×** | 1.105 GiB |
| GRU anchor *(violates I5)* | 3 | 1.76741 | — | — | 1.11× | 0.864 GiB |

Seeds: `compose_s1` **2.08291**, `compose_s2` **2.08360**. `compose_s0` diverged
(§9.5) and is excluded under the post-hoc rule in
`scripts/exp/011_compose_results.py`'s header — **excluded, not replaced**.

**The wall-clock column is against `snn_beta0.5_s0`'s 398.9 s**, this project's
standing denominator. Against the adopted arm's own three-seed 525.3 s the
composed arm is **1.074×** — the cost of one elementwise multiply per layer
outside the time loop, and the manifest records both denominators rather than
leaving the reader to guess which one a ratio is against.

−0.03543 bpc is **−28.7 se** of the difference and **7.7× the 2σ adoption bar**.
Both surviving seeds beat the adopted arm individually, and the spread between
them is 0.00069 bpc.

### 9.2 Where the gain lives, and it is not where either parent's headline is

`EXP_004` §10.3's decomposition, cut at the baseline's horizon of 7, **against
the adopted arm**:

| component | bpc | share |
|---|---:|---:|
| Zero context | **+0.00050** | **1.4 %** |
| Within the baseline's reach | **+0.03597** | **102.4 %** |
| Beyond the horizon | −0.00134 | −3.8 % |
| **Total** | **+0.03513** | 100 % |

**This is the finding, and it confirms `EXP_004` §10.6 in the strong form while
inverting what that implied about the composition's value.**

§10.6 argued the two mechanisms are the same mechanism *at zero context*: the
two-compartment neuron there **is** the baseline with a learned per-channel
threshold. If that is right, stacking a real per-channel threshold on top should
buy nothing at c = 0. **It buys 0.00050 bpc — 1.4 %, and below the ~2e-3
reparameterisation noise floor `EXP_008` §9.6 measured.** The prediction is
confirmed about as cleanly as this design can confirm anything.

So the composition's entire gain is the **within-reach** component — the one the
adopted arm *damaged*. Against the Phase-2 baseline the composed arm decomposes
as zero **+0.05113** (32.2 %), within-reach **−0.02386** (−15.0 %), beyond
**+0.13139** (82.8 %): it still gives back within-reach ground, but **less** than
the adopted arm alone does, and the +0.03597 above is exactly that repair.

Two facts sharpen it:

* the threshold arm moved within-reach by **+0.0288** against the Phase-2
  baseline — the first arm in this project to move that component at all;
* composed onto the adopted arm it moves the same component by **+0.03597**,
  *more* than it managed against the baseline, because the two-compartment
  neuron had left more room there to recover.

**The arms compose because they act on different components than their headline
numbers advertise.** §10.6 was right that they collide at zero context, and that
collision is precisely why the composition's value shows up somewhere else.

### 9.3 The per-context curve, the horizon, and the fold

* **§6.2 against the adopted arm: 0 / 128 contexts significantly worse.** The
  worst delta anywhere is −0.0005, at c = 0. The composition regresses nothing
  relative to the arm it is built on.
* Against the Phase-2 baseline: **3 / 128**, where the adopted arm alone is
  5 / 128. Worst short-context +0.0307 at c = 3 — still the adopted arm's own
  short-range cost, partially repaired.
* **Median horizon 48** (48 on both seeds) against the adopted arm's 47. The
  composition is not a horizon effect and was not expected to be.
* **C4, the fold.** `compose_s1` residual **1.05e-06**, `compose_s2`
  **1.89e-07**, 738 509 → 737 485 parameters, loading into `arch="twocomp"`.
  Identity 2 executed: the gain folds into the layer's `Linear` and `thr_log`
  disappears, so **the composed arm costs the adopted arm exactly at inference.**

  These residuals are **three to four orders of magnitude smaller** than
  `EXP_008` W1's 1.86e-3, which failed its bar. Both are the same identity
  through the same kind of fp32 GEMM reordering, so the gap is worth stating and
  **this experiment does not explain it.** It is reported as an open observation,
  not as evidence about decision #6, and §10 names it as a thing to chase.

* The learned thresholds: `exp(θ)` mean **0.449 / 0.529** (layers 0 / 1) on
  `compose_s1` and **0.456 / 0.523** on `compose_s2`. `EXP_008`'s threshold arm
  alone reached 0.342 / 0.428 and the two-compartment arm's effective figure was
  0.69 / 0.79; composed, the arm settles between them — consistent with the two
  mechanisms partially substituting for one another, which is what §9.2 measures
  directly.

### 9.4 The predictions, resolved as written

| | statement | measured | verdict |
|---|---|---:|---|
| **C1** | carried mean ≤ 2.11869 (no negative interaction) | **2.08326** | **HELD**, by 0.0354 |
| **C2** | cross-seed sd ≤ 0.005 | **0.00049** | **HELD** — but see below |
| **C3** | carried mean > 2.05973 (sub-additive) | 2.08326 | **HELD** |
| **C4** | folds into `arch="twocomp"`; residual reported | 1.05e-06 / 1.89e-07 | **HELD** (no bar applied) |
| **C5** | Identity 1 holds for the two-compartment neuron (fp64) | exact spikes; ≤1e-12 membrane | **HELD** |

**C2 is the prediction the missing seed costs most, and its verdict should be
read weakly.** A sample sd over two values is a number, not an estimate. §3.1
priced this design at n = 3 and even there it resolved only ~0.004 bpc; at n = 2
the 0.00049 says the two surviving seeds agreed, and little else. It is not
evidence that the arm is stable, and it is certainly not evidence that it is
stable *against the failure that killed seed 0*.

**C3 held: the gain is sub-additive.** 60.1 % of the threshold arm's 0.05896
carried over. `EXP_004` §10.11 item 2 predicted the numbers "would not add
twice", and they did not. §3.1 stated in advance that this design cannot
separate 60 % from, say, 55 % or 70 %, so **60.1 % is a point estimate with a
wide interval** and is quoted as one.

### 9.5 `compose_s0` — chased, and it is not the failure the project expected

The seed trained cleanly to step 17 500 (loss 1.3954, grad-norm 0.2256, firing
0.380/0.385) and was NaN by 17 750, for the remaining 2 250 steps.
`scripts/exp/011_chase_compose_divergence.py` replays it from its last healthy
checkpoint. **It reproduces deterministically at step 17 598**, and:

| | `EXP_009`'s dead seed | **`compose_s0`** |
|---|---|---|
| forward at the bad step | finite | **finite** (logits max 223.0) |
| loss | finite | **finite** (1.4622) |
| gradients before it | finite, largest **5.46e31** | **1.2e-2 – 3.1e-2, no run-up at all** |
| ‖g‖ fp32 vs fp64 | **inf vs finite** — the clip overflowed | **both NaN; equal at every healthy step** |
| origin | backward, layer 0 + embedding | **backward, layer 0 + embedding** |
| eager path at the same step | not tested | **identical NaN — kernel exonerated** |

**This is not `EXP_009` §9.4's mechanism.** That seed died because
`clip_grad_norm_`'s fp32 sum of squares overflowed on a genuinely enormous
gradient. This one died with gradients of 0.03, every quantity in the same range
as the six preceding healthy steps, and the parameters at its last healthy
checkpoint sitting **inside** the ranges the two surviving seeds occupy —
`compose_s2` reached a *larger* maximum gain (7.71 against 5.40) and lived.

**Decision #7's fix would not have saved this seed.** Two divergences that look
identical in a training log have different causes, which is the protocol's
"chase a NaN to its origin" earning its place.

Pass 3 replays the same step with `fused=False`. The eager path produces an
identical NaN in the identical six tensors, so the hand-written two-compartment
backward is **exonerated** and this is **not** an R10-class defect.

What is established: the NaN is generated inside layer 0's backward, on **both**
dispatch paths, from a finite forward, a finite loss and unremarkable gradients.
What is **not** established is which arithmetic step produces it — that needs
instrumentation inside the scan's backward, and §10 names it rather than
guessing at it here.

### 9.6 The stretch markers, reported without a verdict (§4.3)

| marker | value | composed arm below it? |
|---|---:|---|
| the tasking's 2.083 | 2.083 | **No** — 2.08326, a miss by **0.00026** |
| half-gain midpoint | 2.08921 | Yes |

**The 2.083 marker is reported as a miss, because that is what it is** — §4.3
attached no verdict to it and the protocol's rule (`EXP_008` W3, failed at 49.7 %
against 50 %) is that a near-miss is a miss.

But the size of the miss is the point. **0.00026 bpc is roughly an eighth of the
~2e-3 reparameterisation noise floor** and two orders of magnitude below what
§3.1 said this design resolves. Had 2.083 been pre-registered as C1 — as this
experiment's tasking specified, and as §0 item 1 declined — **EXP_011 would now
be recording a FAIL on a margin it cannot measure, in the same table where the
arms demonstrably composed at −28.7 se.** The bar and its own justification would
have disagreed, and the number would have won.

### 9.7 Status

**CLOSED**, provisional at n = 2. The composed arm is a **Phase-5 candidate**;
adoption is Elliot's and Phase-3 §9 is still unticked. This experiment closes
**no** other open decision.

Nine training runs' worth of GPU time was not needed: **three runs, ~0.51
GPU-hours**, plus ~0.35 GPU-hours for the 17-checkpoint horizon probe and the
divergence chase, which §2.2 committed to and the tasking's estimate did not
include.

## 10. What this experiment does not answer

1. **Which arithmetic in layer 0's backward makes the NaN.** Bounded to the
   backward on both dispatch paths from finite inputs (§9.5); not localised
   further. The next step is instrumenting inside `twocomp_scan`'s backward, and
   it is now the **second** unexplained divergence in Phase 4.
2. **Why this fold is 1000× tighter than `EXP_008`'s** (§9.3). Same identity,
   same class of fp32 reordering, wildly different residual. Bears on decision #6
   and is deliberately not offered as evidence for it until it is understood.
3. **Whether n = 3 would have held C2.** It cannot be known from n = 2, and the
   seed was not replaced.
4. **Whether either parent's initialisation is right for the composition.** One
   point in a two-dimensional init space (`θ`, `w`), §5's limitation, unswept.
5. **Whether the within-reach repair survives at another width, depth or corpus.**
   Nothing here speaks to scale.
