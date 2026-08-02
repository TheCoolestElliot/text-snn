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

*(appended after the run; nothing above this line is edited)*
