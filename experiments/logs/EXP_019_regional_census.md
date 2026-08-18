# EXP_019 — Is the model's regional structure functional, or only visible in its weights?

**Pre-registered:** 2026-08-17, **after** the two post-hoc observations in §2 and
**before any causal leg in §3 was run**. **Committed before the first run of §3.**

**Status: PRE-REGISTERED — no results yet.**

---

## 0. What this experiment is, and what it is not

The project's chat model carries a per-channel slow pole whose time constants are
initialised identically in every layer (`snnchat.model.spread_slow_poles`) and
which, after training, are not identical at all. `EXP_006` established **causally**
that ~3 % of channels carry the memory horizon and 94.9 % of the two-compartment
arm's gain, on the *research* neuron at 735K. Nothing has ever asked the same
question per **layer**, and nothing has asked it of the chat model.

That matters now because a regional/"brain-areas" architecture has been proposed
(see the §8 referral) and every mechanism in it is a claim about a
`(layer, channel-block)` rectangle. **A rectangle that no measurement can locate
is not a region, it is a diagram.**

**What this is not:**

* **It trains nothing.** Every leg of §3 is a forward pass over checkpoints
  already on disk. `EXP_016` is the precedent: ~0.8 GPU-h, no training, and it
  closed decision #11's mechanism.
* **It adopts nothing, ranks nothing, and changes no hyperparameter.** No file
  under `src/snn/` is edited by this experiment.
* **It is not a Phase-5 arm and does not open Phase 5.** Decision #4 stands at
  "recommend not yet".
* **It produces no bpc that may be compared with a committed figure.** Decision
  #10: this project does not re-baseline. Every number here is measured on the
  current tree, in one session, and is internal to this experiment.
* **It does not decide whether a regional arm is worth building.** It supplies
  the instrument that a regional arm would have to be scored on, and one result
  that could kill the programme outright (§4, L2).

---

## 1. The question

> **Does zeroing one layer's recurrent state cost more than zeroing another's,
> and does the cost track the weight-space profile?**

Stated as a dissociation rather than a description, because a description is what
§2 already has and it is not enough.

---

## 2. What is already measured — POST-HOC, and labelled so

Both of the following were run on 2026-08-17 **before this file existed**. They
are **not** pre-registered, they resolve no bar, and they are recorded here so
that the predictions in §4 are visibly downstream of them rather than
independent. This is `EXP_016` §15.5's precedent (`POST-HOC: it is the run of
consecutive steps`) applied in advance.

### 2.1 The weight-space profile (`scripts/chat/regional_profile.py`)

Across 26 committed chat checkpoints with a slow population, two independently
trained lineages:

| | layer 0 | layer 1 | layer 2 | layer 3 |
|---|---:|---:|---:|---:|
| median \|w\| ratio, slow vs fast channels | **4.99–10.11** | 0.09–5.58 | **0.07–0.17** | **0.07–0.21** |

Layer 0's slow channels are load-bearing in every checkpoint; the deepest two
layers' are switched off in every checkpoint, at `|w| ≈ 0.01`, which is
essentially the `w = 0` boundary at which a two-compartment channel provably
reverts to the plain Phase-2 LIF.

**The two fresh lineages disagree about where the slow peak sits** — `chat-v2`
puts it at layer 1 (median tau 13.23), `chat-v6-scratch` at layer 3 (24.94). That
shape is therefore **not** a universal outcome and is not predicted below.

### 2.2 The order/recency probe (`scripts/chat/permutation_probe.py`)

`order share = ΔPERMUTE / ΔROLL` — what fraction of a context band's value is its
order rather than its character multiset. 3,072 windows, soda val, both
self-checks exactly 0.0:

| band lag | 4 | 8 | 16 | 32 | 64 | 128 |
|---|---:|---:|---:|---:|---:|---:|
| shipped | 0.343 | 0.638 | 0.744 | 0.773 | 0.732 | 0.724 |
| `chat-v6-scratch` | 0.335 | 0.637 | 0.743 | 0.766 | 0.733 | 0.750 |

It does not decay. **The model is not a bag of characters**, and the horizon is a
memory rather than a recency statistic. The lag-4 point is a width-2 artifact
(half of all random permutations on two elements are the identity).

---

## 3. The legs

All on committed checkpoints. `L = 256`, soda val, `state = None`.

* **Leg A — the per-layer lesion.** Zero **layer k's** recurrent state alone
  every *j* characters, leaving every other layer's state to carry, and measure
  end-to-end bpc. Sweep `k ∈ {0..K-1}` and `j ∈ divisors of L`. This is the
  causal counterpart of §2.1 and it is the instrument the regional programme
  needs. It **replaces** the per-layer linear probe considered and rejected:
  that would need K trained decoders whose capacity varies with block width, has
  no power analysis, and invites comparing a probe number against a model number
  — the error `docs/chat/CONVENTIONS.md` exists to catch.
* **Leg B — the block lesion.** Within the layer Leg A finds most costly, zero
  only the **slow half** of the state (`[:, d:]`), and separately only the
  channels above `tau = 100`, against a **matched-size, matched-displacement**
  random-channel control. `EXP_006`'s four-way dissociation design, unchanged in
  method.
* **Leg C — the beta-inflation pre-flight.** Bin `max|w·vs|` by the channel's own
  `beta_s` on committed checkpoints and extrapolate to the retention floors a
  future arm would use. This is a **safety measurement, not a result**: `EXP_016`
  put the crossover at `|w·vs| = 1` and measured 8.96 on the adopted arm, and the
  within-window sd of `vs` scales as `sqrt((1-beta^(2L))/(1-beta^2))`, i.e. 2.67×
  at `beta = 0.95` and 5.90× at `beta = 0.99` against the learned median 0.5506.

---

## 4. Pre-registered predictions

Stated with the quantity, the direction and the bar, before any leg of §3 runs.
Verdicts are reported as counts with exact-binomial intervals; `unresolved` is an
available verdict (`docs/chat/CONVENTIONS.md`).

* **L1.** Lesioning layer 0's state costs **more** bpc than lesioning layer 3's,
  in ≥ 5 of the 6 checkpoints tested. *Sourced from §2.1: layer 0 is the layer
  whose slow compartment is switched on in every checkpoint.*
  **FALSIFIED if** layer 3 costs more in ≥ 2.
* **L2 — the one that can kill the programme.** Lesioning the **slow half** of
  the most-costly layer costs ≥ 0.01 bpc more than the matched random control.
  **If it does not, the slow compartment is not doing load-bearing work in the
  chat model at all**, and every timescale-based regional proposal is refuted
  before it is built. This bar is placed where it can fire against the
  programme's own interest, which is the point.
* **L3.** The rank correlation across layers between "cost of lesioning layer k"
  and "layer k's median |w| ratio from §2.1" is positive. *This is what makes
  §2.1 a locator rather than a curiosity.* **FALSIFIED if** it is negative.
* **L4 (guard, both terms).** L1 and L3 count only if Leg A's **no-lesion** arm
  reproduces the ordinary fresh-state bpc. `EXP_014`'s S3 is the precedent for
  guarding a difference by constraining its subtrahend; four Phase-4 bars fired
  and misled for want of exactly this.
* **L5.** Leg C's extrapolated `max|w·vs|` at `beta = 0.95` exceeds 1 by more
  than 20×. *If it does not, the stability argument against a raised floor is
  weaker than §3 assumes and the referral in §8 must say so.*

**No prediction is made about which layer holds the slow peak.** §2.1 measured
two independent lineages disagreeing, so a prediction there would be a
coin-flip dressed as a hypothesis.

---

## 5. Self-checks, which must be exact

* **S1 (mandatory).** Lesioning **every** layer every *j* characters must
  reproduce `scripts/chat/horizon.py`'s `bpc_at_k(j)` to a residual of
  **exactly 0.0** — the two are the same computation reached by different code.
  A non-zero residual means the lesion instrument is not doing what this log
  says, and the experiment **stops** rather than widening a tolerance.
* **S2.** Lesioning no layer must reproduce the direct fresh evaluation, residual
  `< 1e-6` (float-associativity only).
* **S3.** A lesion applied only to positions after every scored position must
  change nothing, exactly. The permutation probe already carries this check and
  it returned 0.0.

---

## 6. Cost

`EXP_005` measured 15.6 s per checkpoint for a full k-sweep; `EXP_006` did 91
configurations in ~0.45 GPU-h. This is ~70 configurations plus a Jacobian sweep
in `016_reset_jacobian.py`'s shape. **Budgeted 0.7 GPU-h, ×1.0** (no new arch, no
new kernel, no training). A 400-step pre-flight is not required because nothing
here trains.

---

## 7. What this cannot establish

* **It cannot attribute anything to `spread_slow_poles`.** There is no control
  arm with the committed uniform initialisation at matched seeds. That comparison
  is the never-run experiment `docs/chat/README.md` §5 has flagged since
  `chat-v1`, and it is not this experiment.
* **It is n = 1 per checkpoint per cell**, and chat checkpoints sharing a parent
  are **not** independent seeds — 23 of the 27 on disk descend from
  `chat-v2-anneal` at step 7,874. Counts below 6 are counts of *lineages*, not of
  draws, wherever that distinction bites.
* **It measures the model the project has, not the model it could have.** A null
  on L2 refutes the slow compartment as *currently trained*; it does not refute a
  floored or gated one.

---

## 8. What it refers

If L2 holds and L3 is positive, the instrument exists and the referral is a
**floored-retention arm** (`beta_s = beta_min(k) + (1-beta_min(k))·sigmoid(raw)`),
which moves weight decay's fixed point from tau = 2 to a designed per-layer value
without touching the frozen recipe — and which, by Leg C, must ride on
`twocomp_detach`. **Whether an arm may hold `twocomp_detach` fixed as a
co-variable is Elliot's**, is a departure from single-variable discipline, and is
not decided here.

If L2 fails, the referral is that the timescale route is closed on this model and
the gating route (`cur' = g·cur`, rank-1, O(1) per neuron on any reading of
decision #3) should be tried first instead.
