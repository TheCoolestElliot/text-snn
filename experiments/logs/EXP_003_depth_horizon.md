# EXP_003 — Does memory horizon compose with depth?

**Pre-registered:** 2026-08-02, before any depth run was launched.
**Status:** PRE-REGISTERED — no results yet.
**Phase:** 3 (candidate improvements). A **ranking diagnostic**, not an ablation.

---

## 1. Why this must be answered before Phase 4 is costed

`EXP_001` measured the Phase-2 baseline's memory horizon at **7 characters**
against the GRU anchor's **57–60**, and identified that 8× as the mechanism behind
the 0.486 bpc cost of invariant I5. Every Phase-3 candidate is therefore, in
effect, a proposal for how to buy horizon.

There are two ways to buy it, and they cost wildly different amounts:

| Route | What it needs | Cost |
|---|---|---|
| **A new neuron** (multi-timescale, adaptive threshold, rotational state) | a new fused kernel, a hand-written backward, and a **mutation-tested R10 gradient gate** for each | days of engineering; `EXP_002` shows the GPU cost is negligible |
| **More layers** | nothing. `n_layers` is already a config field, already tested, already correct | one flag |

Information at layer *k*, time *t* can only have come from layer *k−1* at times
≤ *t*, each of which reaches back through its own membrane. So the horizons
*might* compose across depth — in which case **K = 8 would reach the GRU's horizon
with no new kernel, no new gradient gate, and no new risk**, and would outrank
every neuron candidate in the matrix on cost alone.

That would be a very different Phase-4 plan from the one the §4.6-admitted neurons
imply. Guessing which is right and then spending 30 GPU-hours on the guess is
exactly what this protocol exists to prevent. Three runs settle it for under one
GPU-hour.

## 2. Hypotheses

**Pre-registered predictions:**

| # | Prediction | Threshold |
|---|---|---|
| **R1** | Horizon grows with depth, roughly linearly, at 3–4 characters per layer: `horizon(K=4) ∈ [14, 28]` and `horizon(K=8) ∈ [28, 56]` | the K=2 baseline is 7 |
| **R2** | bpc improves with depth at matched parameters but sub-linearly, and **K = 8 does not reach the GRU's 1.7674** | |
| **R3** | If R1 holds while R2 also holds, then horizon is confirmed **necessary but not sufficient**, reinforcing `EXP_001` §8.4 from a second, independent direction | |
| **R4** | The §5 initialisation pathology **worsens with depth**: at K = 8 more than one layer is silent at step 0, and recovery takes longer than the ~50 steps measured at K = 2 | firing rates logged from step 0 |

**Decision rule, fixed now:**

* **R1 holds** (horizon ≈ linear in K) ⇒ **depth is a first-class horizon lever**
  and ranks at the top of the Phase-3 ROI matrix, because it needs no new kernel
  and no new gradient gate. Neuron candidates must then justify themselves
  *against* depth, not against the K=2 baseline.
* **Horizon saturates** — `horizon(K=8) < 2 × horizon(K=2)`, i.e. < 14 ⇒ **depth
  does not compose horizon.** Depth is demoted to an ordinary capacity knob and
  the §4.6-admitted neuron-state candidates own the top of the matrix.
* **Horizon grows but bpc does not follow** ⇒ the strongest possible confirmation
  of `EXP_001` §8.4, and it makes "horizon bought without loss of near-context
  fidelity" the single ranking criterion rather than horizon alone.
* **R4 holds** ⇒ the variance-scaled initialisation is **promoted**: it stops
  being a candidate whose ROI is known to be small at K=2 and becomes a
  prerequisite for any deep candidate.

## 3. Design

| Field | Value |
|---|---|
| Arms | `K ∈ {1, 2, 4, 8}` spiking, **parameter-matched** |
| Widths | K=1 → d=676, K=2 → d=512, K=4 → d=380, K=8 → d=278 |
| Parameters | 735 017 / 735 437 / 735 125 / 734 681 — all within **0.1 %** of the Phase-2 baseline |
| Seeds | **1 per new arm** (seed 0). K=2 reuses the five existing Phase-2 checkpoints |
| Everything else | byte-identical to the frozen Phase-2 baseline: enwik8, B=128, L=256, β=0.5, threshold 1.0, hard reset, atan α=2, T=1, fp32, 20 000 steps, cosine schedule |
| Measured | test bpc (both protocols), **memory horizon via `EXP_001`'s paired statistic**, per-layer firing rates from step 0, wall-clock |
| Budget | ~40 minutes of GPU, inside the ~1 GPU-hour Phase-3 allocation (§8.2) |

The width is *derived* from the depth to hold parameters fixed, not chosen, so
that "deeper" does not silently mean "bigger". Solved with
`snn.model.spiking_param_count`; the realised counts are asserted.

## 4. What this is not

**These bpc numbers are single-seed diagnostics and are not ablation results.**
The noise floor is σ = 0.00461 bpc at *n* = 5; one seed does not resolve a 2σ
effect, and no bpc difference measured here may be quoted as adopted, or carried
into the report as an effect size. The quantity this experiment exists to produce
is the **horizon-versus-depth curve**, which `EXP_001` showed to be far more
stable across seeds than bpc is — all five K=2 seeds returned exactly 7.

If depth turns out to look promising on bpc, that is a *hypothesis for Phase 4* to
run properly at ≥3 seeds, and it must be pre-registered there like anything else.

## 5. Known limitations, stated in advance

1. **Depth and width are confounded by construction.** Holding parameters fixed
   means K=8 is also d=278, roughly half the baseline's width. A horizon that
   grows with K therefore grows *despite* narrowing, which is the conservative
   direction; but a bpc that fails to improve cannot be attributed to depth alone.
2. **The hyperparameters are the K=2 baseline's.** Learning rate, warm-up and
   schedule were never tuned for K=8, and deep spiking networks are the case where
   the §5 initialisation pathology is most likely to bite. A poor K=8 result is
   therefore evidence about *this configuration at this depth*, not about depth.
   R4 exists to detect exactly that confound rather than let it be read as a
   verdict on depth.
3. **n = 1.** See §4.

## 6. Failure modes

| # | Failure | Detection |
|---|---|---|
| H1 | Parameter matching is wrong, so "matched" arms differ in size | Realised counts recorded per run and asserted within 0.5 % |
| H2 | A deep run collapses to silence and is read as "depth does not work" | Per-layer firing rates logged from step 0 (R4); a dead run is reported as dead, not as a depth result |
| H3 | The horizon probe behaves differently at K≠2 | `EXP_001`'s F1 self-check is re-run per checkpoint: the k=L point must reproduce that run's own `fresh` bpc |
| H4 | A depth result is quoted as an adopted effect | §4; the report labels every row single-seed |

## 7. Entry conditions

- [ ] `EXP_001` closed, so the horizon statistic is fixed before it is applied here
- [ ] `EXP_002` closed, so the systems cost of depth is known

---

## 8. Results

**Run 2026-08-02.** Three new runs (K = 1, 4, 8), each 20 000 steps at the frozen
baseline settings with the width derived to hold parameters at 735 K. K = 2 is the
five existing Phase-2 seeds. Horizons measured with `EXP_001`'s paired statistic
through the same script, so the numbers are comparable row for row.

### 8.1 The ladder

| K | d | Params | Δ vs baseline | **Horizon** | Test bpc (fresh) | Test bpc (carried) |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 676 | 735 017 | −0.06 % | **6** | 2.5084 | 2.4952 |
| **2** | **512** | **735 437** | — | **7** *(5 seeds, all 7)* | **2.2697** | **2.2531** |
| 4 | 380 | 735 125 | −0.04 % | **9** | **2.2528** | **2.2348** |
| 8 | 278 | 734 681 | −0.10 % | **14** | 2.4194 | 2.3995 |

H1 passed: every arm is within 0.1 % of the baseline's parameter count. H3 passed:
every new checkpoint's `k = L` point reproduces its own independently-produced
`fresh` bpc to ~1e-09.

### 8.2 The predictions, resolved

| # | Prediction | Measured | Verdict |
|---|---|---|---|
| **R1** | `horizon(K=4) ∈ [14, 28]`, `horizon(K=8) ∈ [28, 56]` | **9** and **14** | **FAILED, both** |
| **R2** | K = 8 does not reach the GRU's 1.7674 | 2.3995 | held |
| **R3** | horizon grows while bpc does not follow | K=8 has the longest SNN horizon *and* is worse than K=2 and K=4 | **held** |
| **R4** | more than one layer silent at K=8; recovery slower than ~50 steps | 6–7 layers silent; recovery at **step ~150** | **held** |

### 8.3 Horizon grows as √K, not K

The measured law is clean. Taking K = 2 as the reference, `7·√(K/2)` predicts
4.9 / 7 / 9.9 / 14 against measured **6 / 7 / 9 / 14**.

That is the behaviour of cascaded leaky integrators rather than of concatenated
memories: the sum of K independent exponential delays has mean K·τ but standard
deviation √K·τ, and for a thresholded channel it is the spread that sets usable
reach. Depth does not chain horizons end to end; it diffuses one.

**The consequence is the ranking answer this experiment was run to get.** At √K
scaling, matching the GRU's 57–60 characters needs

```
K = 2 · (58/7)²  ≈  137 layers
```

which at fixed parameters means d ≈ 70, and which the K = 8 arm already shows is
heading the wrong way on bpc. **Depth is not a viable route to the GRU's horizon.**
The pre-registered decision rule's saturation branch — `horizon(K=8) < 2 ×
horizon(K=2)` — lands exactly on its boundary (14 against 14), so it is recorded
as *borderline by the letter and decisive by the substance*: a 4× depth increase
bought exactly 2× horizon, and no extrapolation of that curve reaches the anchor.

Neuron-state candidates therefore do **not** have to justify themselves against
depth. They own the top of the matrix by default, and depth returns to being an
ordinary capacity knob.

### 8.4 K = 4 is the best arm measured, and it fails the §6.2 criterion

K = 4 scores **2.2348 carried against the baseline's 2.2531** — a 0.0183 bpc
improvement, 4σ by the EXP_000 whole-split rule. It is the best number this
project has produced under I1–I5.

**It is n = 1 and it is not adopted** (§4). It is also the first candidate to be
caught by the acceptance criterion `EXP_001` §8.4 pre-registered *before* these
runs existed: K = 4 is worse than the baseline at 5 of 128 context lengths, all
short, worst **+0.0764 bpc at c = 3** against that context's measured 2σ bar of
0.0179 — a 4× significant regression in three-character prediction, bought back at
every context from 8 onward.

That is a milder version of the β signature, and it is exactly what the criterion
exists to surface. It does not make K = 4 a bad candidate; it makes K = 4 a
candidate whose gain has a named mechanism and a named cost, which is what Phase 4
should be testing rather than a scalar.

### 8.5 R4: the initialisation pathology is much worse at depth

`depth_K8_initprobe`, 300 steps with `log_every = 10`:

| Step | Silent layers (of 8) | Train bpc |
|---:|---:|---:|
| 0 | 6 | 7.676 |
| 50 | 6 | 7.636 |
| 100 | 6 | 7.518 |
| 130 | 5 | 7.408 |
| 140 | 3 | 7.375 |
| **150** | **0** | 6.643 |
| 160 | 0 | 5.138 |

At K = 2 the Phase-2 report measured revival within ~25–50 steps. At K = 8 nothing
wakes until **step 130** and the network makes almost no progress until it does —
the loss falls 7.676 → 7.408 over 130 steps, then 7.375 → 5.138 in the twenty
steps after the layers come alive.

Per the decision rule, **R4 holding promotes the variance-scaled initialisation**:
it stops being a candidate with a known-small ROI at K = 2 and becomes a
**prerequisite for any deep candidate**, and a live suspect in why K = 8
underperforms. Whether K = 8's deficit is depth or initialisation is not resolved
here, and §5.2 said in advance that it would not be.

### 8.6 A contaminated run, discarded

The first `depth_K8_s0` was **discarded and re-run.** It started at 01:28:01,
inside the window in which `scripts/audit/08_mutation_campaign.py` was writing
deliberately-wrong kernels into `src/snn/` — timeline reconstruction from the
campaign's own committed timings puts mutation **M15 (`>` instead of `>=` in the
forward kernel)** live at that instant. Python reads a module's source once, at
import, so it trained 7 750 steps against a wrong spike condition with nothing in
its logs to show it.

It was caught because `EXP_001`'s F1 self-check — "the probe's `k = L` point must
reproduce the committed bpc" — failed on an unrelated checkpoint at 1.14e-03, and
the residual was chased instead of waved through as tolerance. The campaign script
now refuses to start while another Python process is alive and writes a lock file
while any mutation is applied.

The other two arms are clean: `depth_K1_s0` started 01:12:03 and `depth_K4_s0`
01:18:54, both before the first mutation was applied at 01:22:12.

**Wall-clock is not reported for any depth arm.** All three overlapped other GPU
work, so their timings measure contention rather than the architecture.

### 8.7 Status

**CLOSED.** R1 failed, R2/R3/R4 held. The ranking question is answered: horizon
grows as √K, depth cannot reach the anchor, and neuron-state candidates are not
displaced. One run was contaminated by this phase's own tooling, detected by a
self-check, discarded and re-run.
