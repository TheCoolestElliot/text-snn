# EXP_007 — What does a trivial 2-tap input mix buy inside the reach?

**Pre-registered:** 2026-08-03, before any token-shift arm was trained.
**Status:** OPEN. Results go in §9.
**Phase:** 4 (controlled experiments).

**This is a labelled diagnostic, not a candidate for adoption.** Token-shift is
one of the three I5 boundary questions `03_phase3_candidates.md` §6.3 referred to
Elliot and that are **still referred**. This experiment does not rule on the
boundary and may not be read as ruling on it. What it produces is a **price**:
how much of the within-reach component of the I5 gap — now the largest single
piece at 51.8 % — a two-tap FIR on the input current recovers. That number is a
Phase-5 ranking input whether or not the arm is ever admissible.

If Elliot rules token-shift **inside** I5, everything here is already in the form
an adoption decision needs (≥3 seeds, the §6.2 curve, the horizon). If he rules
it **outside**, the numbers stay as the control they were run as. Neither reading
is chosen here.

---

## 1. Why this experiment

`EXP_004` §10.3 and `EXP_005` §9.6 decomposed the gap to the GRU anchor by
context length. Against the **adopted** two-compartment arm at n = 7:

| Component | bpc | share of the remaining 0.3398 |
|---|---:|---:|
| At zero context | 0.0687 | 20.2 % |
| **Within the baseline's 7-character reach** | **0.1759** | **51.8 %** |
| Beyond the horizon | 0.0952 | 28.0 % |

**No arm this project has run has ever been aimed at the middle row.** Every
candidate ranked in Phase 3 §6.3 attacks the beyond-horizon component or the
zero-context one. `EXP_006` closed the horizon question in the direction that
makes this worse, not better: ~3 % of channels already carry the horizon and 95 %
of the adopted arm's gain, and the initialisation history argues against widening
the tail buying more.

A two-tap mix is the cheapest possible attack on the middle row and the one whose
mechanism is least ambiguous. Per neuron `c`, applied to the layer's input
current **outside the time loop**:

```
cur'_t = mu_c · cur_t + (1 - mu_c) · cur_{t-1}
```

One parameter per neuron. No new state variable, no new kernel, no change to the
scan. It gives each neuron a length-2 FIR of its own input, which the Phase-2 LIF
— an exponential IIR at fixed `beta = 0.5` with a hard reset — cannot represent
at any weight. So a gain here is a **capacity** gain and not a reparameterisation
(contrast `EXP_008`, whose whole point is that it is one).

### 1.1 Why the mix is on `cur` and not on `h`

`03_phase3_candidates.md` §6.3 states the referral as `x'_t = mu_c·x_t +
(1−mu_c)·x_{t−1}`, "one parameter per neuron". A neuron indexes an **output**
channel of the layer's GEMM, so `mu` indexes `cur`, and the mix goes after the
projection. Shifting the layer's *input* `h` instead would index the previous
layer's channels and would need its own GEMM to stay per-neuron; it is a
different arm and is not this one.

The mix is applied once to the whole `[B, L, d]` current before the scan starts —
two kernels per layer per forward pass against `K·L` inside the time loop. The
one-kernel-per-timestep floor `01_reconnaissance.md` §3.4 makes the performance
metric is untouched.

### 1.2 Why `mu` is unconstrained and initialised at 1.0

At `mu = 1` the mix is `1.0·cur_t + 0.0·cur_{t−1}`, which is `cur_t` **bitwise**
for every finite current. The arm therefore nests the Phase-2 baseline exactly,
and — following `EXP_004` §2's reasoning for preferring the additive mix over the
convex one — it nests it in the **interior** of an unconstrained parameter rather
than at the boundary of a squashed one. Constraining `mu` to `[0, 1]` via a
sigmoid would put the nesting at `logit(1) = +inf`, which is not an
initialisation.

`mu = 1` is **not** a saddle. `dL/dmu_c = sum_t (dL/dcur'_{t,c})·(cur_{t,c} −
cur_{t−1,c})`, which vanishes only if the current is constant in time. That is
derived here, before measurement, as §7.2 requires; G5 measures it.

### 1.3 The one thing the shift cannot do, stated now

`cur_{−1}` is zero at the start of every evaluation window. Under the `carried`
protocol the membrane crosses window boundaries but **the shift register does
not** — so at exactly one character in 256 the arm is handed a zero where a real
previous current existed. This understates the arm by at most one character's
worth of context per window and it is not corrected, because correcting it means
widening the state tensor and changing the contract every other arm shares.
The horizon probe scores with fresh state, so the paired context curve — where
V2, V3 and V5 are decided — is unaffected.

---

## 2. Design

| | |
|---|---|
| Arm | `arch="tokenshift"`, `mu_init = 1.0`, everything else the Phase-2 baseline |
| Seeds | 0, 1, 2 (the adoption rule's minimum, and the seeds `EXP_004` used) |
| Reference | the five committed `snn_beta0.5` seeds — **not retrained** |
| Cost | 3 × ~400 s training + 3 × ~30 s scoring ≈ **0.4 GPU-hours** |

Parameter count is `735_437 + K·d = 736_461`, **+0.14 %**. That is half the
two-compartment arm's +0.28 % and it is a limitation, not a control: shedding
1 024 parameters means `d = 511`, which changes the width — a second variable
introduced to control for a smaller one (`EXP_004` §6.1's ruling, followed here).

### 2.1 What is measured

1. **Whole-split test bpc, both protocols**, per seed, via `scripts/evaluate.py`.
2. **The paired excess curve and the horizon**, per seed, via
   `scripts/exp/001_memory_horizon.py` — imported and run as-is, with an explicit
   `--out`, adding only registry entries. The statistic, the k-sweep and the F1
   tolerance are untouched (`EXP_005`'s failure mode K5).
3. **The absolute bpc-at-context curve** and the §6.2 per-context comparison.
4. **The gain decomposed by context** on `EXP_004` §10.3's basis: zero-context,
   within the baseline's 7-character reach, and beyond it.
5. `|1 − mu|` per layer, the analogue of `EXP_004`'s T4.

### 2.2 What is deliberately NOT measured

**A sweep of `mu_init`.** One initialisation, fixed in advance, chosen because it
nests the baseline. A sweep would price a hyperparameter, not the mechanism.

**A second tap.** Three-tap and learned-lag variants are a different arm. If the
two-tap result is interesting, the follow-up is named in §9 and not run here.

---

## 3. Hypotheses

All bpc thresholds are on **carried** whole-split test bpc against the committed
five-seed baseline mean **2.25311** (sd 0.00461), at the inherited whole-split
bar **2σ = 0.00922**. Horizon thresholds are at the inherited **2σ = 0.00922**
tolerance, at which the baseline's five seeds all return exactly 7.

**Power, stated before the run** (protocol: `EXP_005` §S1's lesson). This is a
3-seed arm against a 5-seed reference. If the arm's own σ matches the baseline's,
`se(Δ) = sqrt(0.00461²/5 + 0.00461²/3) = 0.0034`, so the 0.00922 bar sits at
**2.7 se** — resolvable. A true difference of 0.005 bpc would **not** be, and
would not be claimed. σ is **not** inherited for this arm: it is a structurally
new arm, so `EXP_005`'s rule applies and the arm's own 3-seed sd is reported
beside every verdict. n = 3 cannot establish that the two σ differ and no such
claim will be made.

| # | Prediction | Threshold | Direction |
|---|---|---|---|
| **V1** | **The mix buys bits.** Mean carried test bpc over 3 seeds ≤ **2.24389** | baseline − 2σ | flatters the diagnostic |
| **V2** | **The bits come from inside the reach.** within-reach gain > beyond-horizon gain, both in bpc | strict inequality | **flatters the plan this experiment exists to serve** |
| **V3** | **It does not buy horizon.** Median horizon over 3 seeds ≤ **14** | 2× the baseline's 7 — `EXP_004`'s T1 bar | a 2-tap FIR reaching past 14 is not doing what §1 says |
| **V4** | **It recovers less than half of what is there.** within-reach gain < **0.0581 bpc** | half of the baseline-vs-GRU within-reach component 0.1161 | **stated against the plan** |
| **V5** | **It does not degrade short contexts.** Zero contexts `c ≤ 8` worse than the baseline by more than that context's own 2σ bar | §6.2, applied as a prediction rather than only as a report | flatters |
| **V6** | **The mechanism is used.** Mean `\|1 − mu\|` ≥ **0.05** in both layers | `EXP_004` T4's shape | flatters |

**On V4, explicitly.** V1, V2, V5 and V6 all flatter the diagnostic. V4 is the
one stated so that a *good* result falsifies it: if a two-tap FIR recovers more
than half of the within-reach component, the component is far softer than four
experiments have implied and the Phase-5 ranking changes more than a null here
would change it. It is written down now so that outcome cannot be presented as
having been expected.

**On V3's power.** The horizon is an integer read off a curve crossing a fixed
tolerance (`EXP_006` §3). The baseline's five seeds agree exactly at 7, so a
median of 8 or 9 would be a real movement; a median of 14 is the bar and is
deliberately loose, because the prediction being tested is "this is not a horizon
mechanism", not "this changes the horizon by nothing".

---

## 4. Decision rule, fixed now

| V1 | V2 | Verdict |
|---|---|---|
| **holds** | **holds** | **A two-tap input mix buys bits inside the reach.** The within-reach component is attackable by a mechanism this project has not been ranking, and the recovered fraction is reported as the price. The I5 ruling stays Elliot's; the number does not depend on it. |
| **holds** | fails | **It buys bits, but not the ones this experiment was aimed at.** Both components reported; the within-reach claim is not made, and the arm is re-described as whatever the decomposition says it is. |
| **fails** | — | **Null, retained.** A two-tap FIR does not beat the baseline at 3 seeds. Reported as a negative result with the decomposition and the horizon beside it, and the within-reach component keeps its status as *unattacked* rather than *attacked and hard*. |

Two riders, also fixed now:

* **V5 failing does not veto anything** — §6.2 is a reporting requirement, not a
  veto (Phase-3 report §6.2). It is stated as a prediction here because a
  short-range mechanism that *degrades* short range would be a genuine surprise,
  and a surprise that is not written down in advance is an anecdote.
* **Nothing here is an adoption.** Even at V1 ∧ V2 ∧ V5 the arm is not adopted,
  because its I5 status is not this experiment's to decide.

---

## 5. Known limitations, stated in advance

1. **The shift register does not cross window boundaries** (§1.3). One character
   in 256 under the `carried` protocol sees a zero where a real current existed.
2. **+0.14 % parameters, uncontrolled** (§2). The zero-context part of any gain
   has the same two live explanations `EXP_004` §10.6 named, and this arm cannot
   separate them either. `EXP_008` can, and does.
3. **One initialisation, one tap count, three seeds.** The result is a statement
   about `mu_init = 1.0` at K = 2, d = 512, 20 000 steps.
4. **The within-reach component is defined against the baseline's horizon of 7.**
   If this arm's own horizon moves, its decomposition is still cut at 7, because
   the components have to stay comparable with `EXP_004` §10.3's. V3 is what
   checks that the cut still makes sense.
5. **No σ of its own beyond 3 seeds.** Reported, not inherited; not claimed to
   differ from the baseline's.

---

## 6. Self-checks — aborting, not advisory

| # | Check | Why it can fail |
|---|---|---|
| **G1** | At `mu = 1` the arm's logits **and** its gradients w.r.t. every shared parameter are **bitwise identical** to `SpikingCharLM`'s at the same seed, on real data | this is the nesting the whole design rests on; a non-bitwise result means the mix is not the identity at init and V1 measures two changes |
| **G2** | Parameter count is exactly **736 461** = `735_437 + 2·512` | catches a mix that accidentally became `[1, d]` per *layer pair*, or a second copy |
| **G3** | The horizon probe's `k = L` point reproduces each run's committed `final_test.json` fresh bpc to < 1e-3 (`EXP_001`'s F1) | the check that caught a contaminated run in Phase 3 |
| **G4** | Every new run's `config.json` differs from `snn_beta0.5_s0`'s only in `seed`, `run_name`, `arch` and `mu_init` (`EXP_005`'s K1) | "the baseline plus one thing" is a claim about sameness |
| **G5** | The §7.2 reachability screen reports `\|dL/dmu\|` non-zero and within 100× of `\|dL/dW\|` at the `mu = 1` init | §1.2 derives that it is non-zero; the screen is what turns the derivation into a number |
| **G6** | No mutation campaign lockfile at the start of **every** run (`EXP_005`'s K2) | Python imports a module once; a run started mid-campaign trains against a mutated kernel and its log cannot tell |

---

## 7. Failure modes

| # | Failure | Detection |
|---|---|---|
| **M1** | The mix is a no-op — `mu` never leaves 1.0 | V6, and G5 before the run |
| **M2** | The shift is off by one, or leaks across the batch | a dedicated test asserting the shifted tensor equals `cur` rolled by one with a zero first column, and G1 at `mu = 1` |
| **M3** | The arm is not the baseline plus one thing | G4 |
| **M4** | A gain is read as within-reach because the decomposition was cut at the wrong horizon | V3, and the cut is fixed at the baseline's 7 in advance (§5 item 4) |
| **M5** | The probe is edited to accommodate a new arm | registry entries only; `--out` to a new file; no committed artifact overwritten |
| **M6** | The new arch breaks an existing arm | the full suite is re-run and must stay at **242 passed** plus the new tests, with none of the 242 changed |

---

## 8. Entry conditions

- [x] Existing test suite green — **242 passed** (2026-08-03, matching `EXP_006` §8)
- [x] Working tree clean at `1e5b239`, branch `phase-4-experiments`
- [x] No mutation campaign running
- [x] This file committed **before** the arm's implementation is written
