# EXP_008 — A learned per-channel threshold, and the fact that it adds nothing

**Pre-registered:** 2026-08-03, before any threshold arm was trained.
**Status:** OPEN. Results go in §9.
**Phase:** 4 (controlled experiments). The arm `EXP_004` §10.11 item 2 named as
"one parameter, no second state variable, no new kernel, and no gradient gate",
and which nothing has run.

**A note on the number in its name.** The user request that scheduled this work
called it "Candidate #2". `03_phase3_candidates.md` §6.3 ranks **#2 = current
normalisation (RMSNorm on `cur`)**; the learned per-channel threshold is not on
the §6.3 list at all — it is **item 2 of `EXP_004` §10.11**, proposed after the
ranking was frozen. The two are different arms that attack overlapping
components. This file runs the one that was named in words, records the
collision rather than resolving it by relabelling either document, and §10 ranks
RMSNorm as the follow-up.

---

## 1. Why this experiment, and what it is really testing

`EXP_004` §10.6 established by algebra that at zero context the two-compartment
neuron **is** the Phase-2 baseline with a learned per-channel threshold:

```
v_0 = vf_0 + w_c·vs_0 = cur_0·(1 + w_c)      so the neuron fires iff
cur_0 >= thr / (1 + w_c)
```

and that this accounts for **41.0 %** of the adopted arm's gain (`EXP_005`, n=7:
+0.0506 of 0.1235 bpc). §10.6 called a one-parameter threshold arm "the natural
candidate to capture that 40 % on its own", and named a second thing it would
settle: whether the zero-context gain is the threshold or is simply the +0.28 %
extra parameters, which in the two-compartment arm are *the same 2 048 numbers*.

**This experiment can settle both, and the reason is that the arm is provably a
reparameterisation of the baseline.** That is derived below, before it is run.

### 1.1 Two identities, derived first

**Identity 1 — a per-channel threshold is a per-channel input gain.**
For `g_c > 0`, consider the committed LIF with threshold `thr·g_c`:

```
v_t = beta·v_{t-1} + cur_t ,   s_t = 1[v_t >= thr·g_c] ,   v_t <- v_t·(1 - s_t)
```

Substituting `u = v / g_c` gives

```
u_t = beta·u_{t-1} + cur_t/g_c ,   s_t = 1[u_t >= thr] ,   u_t <- u_t·(1 - s_t)
```

— the **committed** neuron at the **committed** threshold, driven by a scaled
current. The reset is multiplicative, so it commutes with the scaling; that is
what makes the substitution exact rather than approximate. The two forms produce
the identical spike train in exact arithmetic.

This experiment therefore implements the **second** form: `cur' = cur·exp(-θ_c)`
applied once to the whole `[B, L, d]` current outside the time loop, with the
committed `lif_scan` and the committed `thr = 1.0` untouched. **No new kernel, no
new hand-written backward, no new R10 gate** — the claim `EXP_004` §10.11 item 2
made about this arm's cost is met by construction rather than by care.

**Identity 2 — the arm adds no functions.** The current is `cur = W·h + b`, so
scaling output channel `c` by `γ_c` is exactly the model with `(W_c, b_c)`
replaced by `(γ_c·W_c, γ_c·b_c)`. Every layer's gain folds into that layer's own
`nn.Linear`, whose rows are already free parameters. **The function class of this
arm is identical to the Phase-2 baseline's.** The `K·d = 1 024` parameters it
adds span no new functions; they are exactly redundant.

### 1.2 What that means, and why the experiment is still worth running

Identity 2 has a consequence that sharpens `EXP_004` §10.6 rather than confirming
it:

* **The zero-context gain cannot be a capacity effect.** Whatever effective
  threshold the two-compartment neuron learned, the Phase-2 baseline could
  already have represented it by scaling the rows of its own projection. So
  §10.6's two "live explanations" — the threshold, or the +0.28 % parameters —
  **collapse into one**: neither is capacity. What is left is *optimisation*.
* **Any gain this arm shows is therefore an optimisation effect** — a different
  trajectory under AdamW, not a larger reachable set. The two parameterisations
  are not equivalent to the optimiser: Adam normalises per parameter, so a gain
  gets its own effective step size, and decoupled weight decay pulls `θ` toward 0
  (gain toward 1) while it pulls `W` toward 0. That is precisely the mechanism by
  which a redundant parameter can still change where training lands.
* **And it would be free.** A gain that folds into `W` costs nothing at inference
  and can be folded away before shipping: zero parameters, zero kernels, zero
  latency. An optimisation-only win is worth exactly as much as a capacity win in
  bits and rather more in engineering.

The experiment's load-bearing prediction is therefore **W1, the fold-in**, which
turns Identity 2 from algebra into a measurement.

---

## 2. Design

| | |
|---|---|
| Arm | `arch="threshold"`, `thr_log_init = 0.0`, everything else the Phase-2 baseline |
| Parameterisation | `thr_c = thr·exp(θ_c)`, realised as `cur' = cur·exp(−θ_c)`; `θ` unconstrained, init 0 |
| Seeds | 0, 1, 2 |
| Reference | the five committed `snn_beta0.5` seeds — **not retrained** |
| Cost | 3 × ~400 s training + scoring ≈ **0.4 GPU-hours** |

`exp(0) = 1.0` exactly in fp32 and `cur·1.0 == cur` bitwise, so the arm nests the
baseline **exactly** at init — in the interior of an unconstrained parameter, not
at a boundary. `exp` also keeps `g > 0`, which Identity 1 requires: a gain that
crossed zero would invert the threshold comparison and is not the arm §10.6
described.

Parameter count `735_437 + K·d = 736_461`, **+0.14 %** — and, by Identity 2,
+0.00 % in function-space dimension. Both are reported.

### 2.1 The §7.2 screen is already committed for this arm

`scripts/audit/09_gradient_reachability.py` screens this candidate as `thr` and
**it passes** — `docs/reports/data/audit_09_gradient_reachability.json`, verdict
`PASS`, `thr_gain` gradient rms 1.124e-4 (layer 0) and 5.801e-4 (layer 1) against
`|dL/dW|` rms of 5.763e-5 and 8.143e-5. The screen's own synthetic control
`ctl_live` — "a per-channel gain straight onto the current" — is *literally* this
arm's implemented form, and also passes. `03_phase3_candidates.md` §7.3 item 3
already names this arm as one of the two exact-nesting inits that are **not**
saddles, "because their parameter multiplies a live state".

So reachability is settled by an artifact committed before this experiment
existed. H5 re-runs the screen and requires it to reproduce that artifact.

### 2.2 What is measured

1. Whole-split test bpc, both protocols, per seed (`scripts/evaluate.py`).
2. The paired excess curve, the horizon, and the absolute bpc-at-context curve
   (`scripts/exp/001_memory_horizon.py`, registry entries only).
3. The gain decomposed by context on `EXP_004` §10.3's basis.
4. **The fold-in**: `γ_c` multiplied into each layer's `Linear` rows and bias,
   the result loaded as a plain `arch="snn"` model, and scored.
5. The learned effective threshold `exp(θ_c)` per layer — mean, sd, and the
   10th–90th percentile band, to compare with §10.6's measured `thr × 0.69` and
   `thr × 0.79`.

---

## 3. Hypotheses

bpc thresholds are on **carried** whole-split test bpc against the committed
five-seed baseline mean **2.25311**, at the inherited **2σ = 0.00922**. Context
components use `EXP_004` §10.3's decomposition, cut at the baseline's horizon 7.

**Power, stated before the run.** Same as `EXP_007`: 3 seeds against 5, so if
this arm's σ matches the baseline's 0.00461 the bar sits at 2.7 se. A true
difference of 0.005 bpc is **not** resolvable at n = 3 and will not be claimed.
σ is reported from the arm's own three seeds and is **not** inherited
(`EXP_005`'s rule) — though note that this arm has more claim to the baseline's σ
than any arm yet run, since by Identity 2 it trains inside the baseline's own
function class.

| # | Prediction | Threshold | Direction |
|---|---|---|---|
| **W1** | **The arm is a reparameterisation, and it is measurable.** For every seed, folding `γ_c` into the layer's `Linear` and scoring as `arch="snn"` reproduces the arm's own fresh test bpc to < **1e-3 bpc** | `EXP_001`'s F1 tolerance, which is this project's standing bar for "one number, two code paths" | **load-bearing; a failure falsifies §1.1's algebra, not the arm** |
| **W2** | **The reparameterisation buys bits anyway.** Mean carried test bpc over 3 seeds ≤ **2.24389** | baseline − 2σ | flatters |
| **W3** | **The gain is concentrated at zero context.** zero-context gain ≥ 50 % of the total gain | §10.6's attribution, which predicts the mechanism is a threshold and not a memory | flatters |
| **W4** | **The network lowers its threshold.** Mean `exp(θ_c)` < **0.95** in both layers | §10.6 measured 0.69 and 0.79 for the equivalent quantity in the two-compartment arm | flatters |
| **W5** | **It does not exceed the two-compartment arm's zero-context gain.** zero-context gain ≤ **0.0506 bpc** | `EXP_005`'s n=7 measurement of the quantity §10.6 attributes *to this mechanism* | **stated against the plan** |
| **W6** | **It does not buy horizon.** Median horizon over 3 seeds ≤ **8** | one character above the baseline's 7; by Identity 2 the arm cannot change the reachable set, so a horizon gain would need explaining | **stated against the plan** |

**W3 is conditional and says so now.** A share of a gain is meaningless if there
is no gain. If W2 fails, W3 is reported as **not evaluable**, with the three
components printed in bpc, and it is not scored as either held or failed.

**On W5 and W6, explicitly.** Four of these six predictions flatter the arm. W5
and W6 are the two that a good result falsifies, and they are the two that would
mean §10.6's account is incomplete: if a bare threshold beats the two-compartment
arm's zero-context gain, then the thing §10.6 attributed to the threshold is not
only the threshold; if it lengthens the horizon, then something is reaching
further inside a function class that provably has not grown, and that is a
finding about the optimiser rather than about the neuron.

---

## 4. Decision rule, fixed now

| W1 | W2 | Verdict |
|---|---|---|
| **holds** | **holds** | **A free win, and a correction to §10.6.** The zero-context component is reachable by the baseline's own parameterisation and is an *optimisation* effect, not a capacity one. The arm is recommended for adoption **and** for folding away at inference — zero parameters, zero kernels. §10.6's "two live explanations" are reported as collapsed into one. |
| **holds** | fails | **Null, retained, and still informative.** The threshold effect §10.6 measured inside the two-compartment neuron does not transfer to the baseline on its own. Since the arm cannot lack the *capacity* (Identity 2), a null localises the two-compartment arm's zero-context gain in the interaction with the slow pole rather than in the threshold alone — which is a correction to §10.6 in the other direction, and is reported as one. |
| **fails** | — | **Stop and chase it.** A failed fold-in means §1.1's algebra or its implementation is wrong. No bpc verdict is issued until the residual is explained. The tolerance is not widened (protocol: chase an unexpected residual). |

**Rider.** If W2 holds, the adoption is recommended **to Elliot**, not taken:
this is the first arm in Phase 4 whose adoption would change the shipped model,
and §9 of the Phase-3 report is still unticked.

---

## 5. Known limitations, stated in advance

1. **Identity 2 is exact in function space and inexact in floating point.**
   `γ·(W·h + b)` and `(γ·W)·h + γ·b` round differently — the GEMM accumulates in
   a different order. W1's tolerance is 1e-3 bpc for that reason and not because
   the algebra is doubted; the measured residual is reported.
2. **A null does not show the mechanism is worthless**, it shows it is worthless
   *at this initialisation, this optimiser and this budget*. The arm is a
   parameterisation change, and parameterisation effects are budget-dependent.
3. **Three seeds.** See §3's power note.
4. **The fold-in is checked on `fresh`, not `carried`.** One protocol, one
   number, one comparison; running both would invite two checks of one identity
   to disagree at the 1e-4 level and turn a clean check into an argument.
5. **This experiment does not test RMSNorm** (§6.3's actual #2), which attacks
   the same zero-context component by a different route and is a separate arm.

---

## 6. Self-checks — aborting, not advisory

| # | Check | Why it can fail |
|---|---|---|
| **H1** | At `θ = 0` the arm's logits **and** its gradients w.r.t. every shared parameter are **bitwise identical** to `SpikingCharLM`'s at the same seed, on real data | the nesting the design rests on; also proves `exp(0)·cur == cur` bitwise on real values rather than in principle |
| **H2** | Parameter count is exactly **736 461** | catches a `[1, d]` that became `[d, d]`, or a per-layer-pair parameter |
| **H3** | The horizon probe's `k = L` point reproduces each run's committed `final_test.json` fresh bpc to < 1e-3 (`EXP_001`'s F1) | the Phase-3 contamination check |
| **H4** | Every new run's `config.json` differs from `snn_beta0.5_s0`'s only in `seed`, `run_name`, `arch` and `thr_log_init` | "the baseline plus one thing" is a claim about sameness |
| **H5** | `scripts/audit/09_gradient_reachability.py --only thr,ctl_live,ctl_dead,ctl_faint` reproduces the committed verdicts in `audit_09_gradient_reachability.json`, and the real `arch="threshold"` model reports non-zero `\|dL/dθ\|` at init | a number produced by a different script, before this experiment existed |
| **H6** | No mutation campaign lockfile at the start of **every** run | Phase-3 report §8 |

---

## 7. Failure modes

| # | Failure | Detection |
|---|---|---|
| **P1** | The gain is applied inside the time loop, costing `K·L` kernels | a test asserting one `exp` and one multiply per layer per forward pass, and the kernel-count gate |
| **P2** | `θ` is not trained (frozen, or excluded from the optimiser) | W4, and H5's non-zero gradient at init |
| **P3** | The fold-in is written in a way that cannot fail — e.g. re-using the arm's own forward | the folded model is built as `arch="snn"` through `build_model` and scored by `snn.evaluate.evaluate`, the same code path the committed baseline numbers came from |
| **P4** | The arm is not the baseline plus one thing | H4 |
| **P5** | A gain is attributed to the threshold when it is the extra 1 024 parameters | Identity 2: there is no such distinction to make. This failure mode is *dissolved* by the design rather than detected by it, which is the reason for choosing this arm over a free per-channel bias |
| **P6** | The new arch breaks an existing arm | the full suite stays at **242 passed** plus the new tests, none of the 242 changed |

---

## 8. Entry conditions

- [x] Existing test suite green — **242 passed** (2026-08-03)
- [x] Working tree clean at `1e5b239`, branch `phase-4-experiments`
- [x] No mutation campaign running
- [x] This file committed **before** the arm's implementation is written
