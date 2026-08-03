# EXP_007 — What does a trivial 2-tap input mix buy inside the reach?

**Pre-registered:** 2026-08-03, before any token-shift arm was trained.
**Status:** **CLOSED** 2026-08-03. **V1 held and V2 failed**: the two-tap mix buys
0.0387 bpc, and **101.8 % of it is at zero context**, where the mix is not a mix.
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
- [x] This file committed **before** the arm's implementation is written — `1f05bab`,
      against the harness's `5d6f88a` and the resolver's `f9c509c`
- [x] Suite green **after** the implementation — **262 passed**: the same 242,
      none of them changed, plus 20 new in `tests/test_prescan.py`

### 8.1 One refinement, made before any result was read

**The driver skips runs that are already complete, and deletes partial ones.**
The first launch was killed by a harness timeout part-way through
`tokenshift_s1`, leaving `tokenshift_s0` finished and scored and `tokenshift_s1`
holding a `ckpt_last.pt` from step 10 000 of 20 000. `is_complete` was added to
`scripts/exp/007_run_prescan_arms.py` at that point — before any bpc from this
experiment had been looked at — and it is recorded here rather than absorbed
into the harness silently.

Two properties of it matter:

* **Completeness requires both halves**: `summary.json` (written only after the
  trainer finishes `max_steps`) *and* `final_test.json` (written only by
  `scripts/evaluate.py`), with the step count checked against the config's. A
  directory holding an interrupted run satisfies neither.
* **A partial run is deleted and retrained, not resumed.** `Trainer` supports
  resume and `tests/test_determinism.py` asserts it is bit-identical for the
  Phase-2 arm, but that has not been checked for this one — and a partial
  directory would in any case append to `log.jsonl` and leave artifacts
  describing two runs at once. Ten minutes of GPU is cheaper than an ambiguous
  provenance.

`tokenshift_s0` is therefore the only run in this experiment that predates the
refinement, and it was complete before it was made.

---

## 9. Results

**Run 2026-08-03.** Three seeds, 20 000 steps each, strictly sequential.
**~0.50 GPU-hours** of training against §2's 0.4 estimate. Raw JSON:
`docs/reports/data/exp_007_008_run_manifest.json` and
`docs/reports/data/exp_007_009_arm_results.json`; horizons in
`docs/reports/data/exp_007_009_memory_horizon.json`.

### 9.1 The headline: it works, and not for the reason it was built

| | baseline (n=5) | **token-shift (n=3)** | Δ |
|---|---:|---:|---:|
| test bpc, carried | 2.25311 | **2.21444 ± 0.00103** | **−0.03867** |
| test bpc, fresh | 2.26969 | **2.22123 ± 0.00132** | −0.04846 |
| memory horizon | 7 (5/5) | **8 (3/3)** | +1 |

−0.03867 is **18.0 standard errors** of the difference and 8.4× the inherited σ.
V1 is not close.

**And then the decomposition, which is the result.**

| Component of the gain | bpc | share |
|---|---:|---:|
| **At zero context** | **+0.0493** | **+101.8 %** |
| Within the baseline's 7-character reach | **−0.0065** | −13.4 % |
| Beyond the horizon | +0.0057 | +11.7 % |
| **Total** | **+0.0485** | 100 % |

**Everything the two-tap mix buys, it buys at the one context where it has nothing
to mix.** Within the reach it was built to attack, it is slightly *worse* than the
baseline.

### 9.2 Why, in one line of algebra that was in §1.2 all along

At the first character of a window `cur_{-1} = 0`, so

```
cur'_0 = mu_c·cur_0 + (1 - mu_c)·0 = mu_c·cur_0
```

**At zero context the token-shift is not a filter. It is a per-channel gain on the
current — which is exactly `EXP_004` §10.6's learned per-channel threshold, and
exactly what `EXP_008` tests directly.** The `1 − mu` tap contributes nothing at
c = 0 for the same reason `EXP_006` §1.1's `beta_s` contributes nothing there.

So the arm that was ranked to attack the within-reach component turns out to
deliver ~102 % of its gain through the *other* candidate's mechanism, arriving at
it sideways. That is what V2 failing means, and it is worth more than V2 holding
would have been: **it is independent corroboration of §10.6 from an arm that was
not built to test it.**

The learned `mu` is consistent with that reading and is not uniform between layers:

| | layer 0 | layer 1 |
|---|---:|---:|
| mean `mu` (3 seeds) | **0.712 ± 0.003** | **1.147 ± 0.003** |
| 10th–90th percentile | 0.63 – 0.82 | 0.24 – 2.02 |

Layer 0 settles tightly just below 1 in every seed; layer 1 spreads across
`mu > 1` — a *negative* weight on the previous character — with a 10-90 band eight
times wider. The two layers are not doing the same thing, and this experiment does
not establish what layer 1 is doing.

### 9.3 The predictions, resolved as written

| # | Prediction | Threshold | Measured | Verdict |
|---|---|---|---|---|
| **V1** | mean carried improves by > 2σ | 2.24389 | **2.21444** | **held** |
| **V2** | within-reach gain > beyond-horizon gain | — | **−0.0065 vs +0.0057** | **FAILED** |
| **V3** | median horizon ≤ 14 | 14 | **8** | **held** |
| **V4** | within-reach gain < 0.0581 | 0.0581 | **−0.0065** | held *(vacuously — see below)* |
| **V5** | no context c ≤ 8 worse than its bar | 0 | **0 of 128 at any c** | **held** |
| **V6** | mean \|1 − mu\| ≥ 0.05 in both layers | 0.05 | **0.29 / 0.62** | **held** |

**Decision cell (§4): V1 holds, V2 fails ⇒ "It buys bits, but not the ones this
experiment aimed at."** The within-reach claim is **not made**.

**V4 held vacuously and is reported as such rather than as a pass.** It was stated
against the plan — "a two-tap FIR recovers less than half of the within-reach
component" — and a *negative* within-reach gain clears a positive ceiling without
telling anyone anything. The prediction was mis-specified: it bounded the
mechanism from above and never considered that the mechanism might go the wrong
way. Counting it as a hit would be scoring a miss as a hit.

**V5 is the one clean piece of good news.** The arm is worse than the baseline at
**zero** of 128 context lengths — the first Phase-4 arm to manage that. The
two-compartment arm is worse at 5, and every β-raising arm in Phase 3 at ~126.

### 9.4 What it cost, which is not what §1 implied

| | baseline | **token-shift** | two-compartment |
|---|---:|---:|---:|
| wall-clock, 20 000 steps | 398.9 s | **595 s (1.49×)** | 536.7 s (1.35×) |
| peak VRAM | 0.609 GiB | **0.859 GiB (1.41×)** | 0.980 GiB |

**The "cheap" arm is more expensive in wall-clock than the two-state neuron it was
supposed to undercut.** §1 priced it as O(1) kernels per layer against the scan's
O(L), which is true and is the wrong metric: the transform allocates four
`[B, L, d]` intermediates per layer — 67 MiB each at B=128, L=256, d=512 — and all
of them stay alive for the backward. On a memory-bound box that costs 1.49×.

This is `EXP_002` §10.9's mistake in a new place: a forward-only kernel-count
argument that omits what the backward has to keep. Recorded here rather than in a
systems appendix because it changes the arm's ROI, and because the same reasoning
was about to be applied to every other "outside the time loop" candidate.

### 9.5 What this does and does not settle for I5

**Nothing.** The boundary question is Elliot's and this experiment took no
position on it (§header). What it supplies is the price it promised, and the price
is now known to be for a different thing than expected:

* If token-shift is ruled **inside** I5, it is an adoptable arm at −0.0387 bpc
  with 0/128 contexts regressed — but its mechanism is ~102 % zero-context, so it
  is largely **redundant with `EXP_008`'s one-parameter arm**, which delivers more
  (−0.0590) for less (1.12× wall-clock against 1.49×). Adopting both without
  measuring their interaction would be double-counting.
* If it is ruled **outside**, nothing is lost that `EXP_008` does not already
  provide, and the within-reach component stays unattacked.

### 9.6 Status

**CLOSED. V1, V3, V5, V6 held; V2 failed; V4 held vacuously and is not counted.**

1. **A two-tap input mix buys 0.0387 bpc over the Phase-2 baseline at 3 seeds**,
   with no context regressed beyond its own noise bar.
2. **It does not attack the within-reach component.** It is 0.0065 bpc *worse*
   there. **The within-reach component — 51.8 % of the remaining gap — remains
   unattacked by any arm this project has run**, and this experiment was the one
   that was supposed to attack it.
3. **101.8 % of its gain is at zero context, where the mix degenerates to a
   per-channel gain** (§9.2). The arm corroborates `EXP_004` §10.6 sideways and is
   mechanistically close to `EXP_008` rather than complementary to it.
4. **It costs 1.49× wall-clock**, more than the two-compartment neuron, for the
   reason §9.4 gives. "Outside the time loop" is not the same as "free", and the
   kernel-count argument that said otherwise is the one `EXP_002` already got
   wrong once.
