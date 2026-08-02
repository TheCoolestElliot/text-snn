# EXP_001 — The memory horizon of the Phase-2 baseline, and of the anchor it loses to

**Pre-registered:** 2026-08-02, before the probe was written or run.
**Status:** PRE-REGISTERED — no results yet.
**Phase:** 3 (candidate improvements). Answers `01_reconnaissance.md` §4.5 item 4,
the open question named there as directly measurable.

---

## 1. Why this runs first in Phase 3

Phase 2 produced one number that defines the project's remaining work:

> The SNN is **0.486 bpc (105σ) worse than a GRU** at matched parameter count,
> corpus, schedule and seeds, while being **0.0117 bpc better** than its own
> identical-parameter analogue control (`02_baseline_report.md` §7.2).

Binarity is not the cost. Invariant I5 is. But "I5 costs 0.486 bpc" is a
statement about a *constraint*, not about a *mechanism*, and Phase 3 has to rank
candidates against a mechanism. There are two rival explanations and they imply
opposite rankings:

| # | Explanation | What Phase 3 should then rank highest |
|---|---|---|
| **M1** | **Horizon.** The lateral matrix stores context over tens or hundreds of characters; a leaky membrane stores a few. The gap is *length of memory*. | Multi-timescale membranes, rotational state, adaptive thresholds — the §4.6-admitted dynamics that lengthen or restructure the horizon |
| **M2** | **Per-step expressivity.** A GRU's gate performs an input-dependent multiplicative update every step; a LIF performs a fixed linear decay plus a threshold. The gap is *what one step can compute*, not how far back it reaches. | Input-dependent per-neuron dynamics (adaptive threshold, learned decay driven by the neuron's own state), and depth/width reallocation — not longer timescales |

The §7.3 β result already leans against a naive version of M1: raising β from 0.5
to 0.95 lengthens the membrane's horizon and monotonically **hurts**, by 11σ and
20σ. But that is evidence about a *low-pass filter's* horizon, not about whether
horizon matters at all. This experiment separates the two directly, on the trained
models we already have, without training anything.

## 2. Hypotheses, and what each outcome forces

The probe forcibly zeroes every model's recurrent state every `k` characters at
evaluation time and scores the full test split. `bpc(k)` traces how much of each
model's prediction quality depends on state older than `k` characters.

**Pre-registered predictions:**

| # | Prediction | Threshold |
|---|---|---|
| **P1** | The SNN's score is unaffected by truncation at k = 8: `bpc_snn(8) − bpc_snn(256) < 0.0092` (2σ) | the EXP_000 adoption threshold |
| **P2** | The probe is *sensitive*: at k = 1 the SNN is badly degraded, `bpc_snn(1) > 3.0` | rules out a flat curve caused by a broken probe |
| **P3** | The GRU is materially degraded at the same truncation: `bpc_gru(8) − bpc_gru(256) > 0.10` bpc | > 10× the SNN's degradation |
| **P4** | Horizon to within 2σ of untruncated: **SNN ≤ 8 characters, GRU ≥ 64 characters** | |

**Decision rule, fixed now:**

* **P1 ∧ P3 hold** ⇒ mechanism **M1**. The I5 cost is a horizon cost. Phase 3
  ranks candidates primarily by *how much horizon they buy per parameter*, and the
  §4.6-admitted multi-timescale / rotational forms move to the top of the matrix.
* **P1 holds, P3 fails** (the GRU is also short-horizon) ⇒ mechanism **M2**.
  Multi-timescale candidates are **demoted**: lengthening memory cannot recover a
  gap that is not made of memory. Ranking shifts to input-dependent per-step
  dynamics and to depth/width reallocation.
* **P1 fails** (the SNN degrades materially at k = 8) ⇒ the baseline already
  exploits a horizon longer than the leak arithmetic predicts, which sits badly
  with §7.3's β result. That is a **discrepancy**, and it is handled under the
  protocol's discrepancy rule — quantified and root-caused before any Phase-4 arm
  is costed — not absorbed into the ranking.
* **P2 fails** (k = 1 is not degraded) ⇒ the probe is invalid, the run is
  discarded, and the failure is reported. No ranking conclusion may be drawn from
  it.

These are written down before the run precisely because M1 is the comfortable
answer — it is the one that makes the §4.6 ruling's admitted extensions look like
the obvious plan — and a ranking that assumed it would be unfalsifiable.

## 3. Design

| Field | Value |
|---|---|
| Trains anything? | **No.** Evaluation only, on the Phase-2 final checkpoints |
| Split | enwik8 **test**, full split, both under the training-matched window |
| Base protocol | `fresh` (state zeroed at each 256-character window start) — this is the training distribution, so it is the honest reference for a truncation curve |
| Truncation `k` | 1, 2, 4, 8, 16, 32, 64, 128, 256 characters |
| Reference point | `k = 256` is the untruncated within-window score and **must reproduce the committed `fresh` bpc** in `docs/reports/data/phase2_final_scores.json` |
| Arms | SNN β=0.5 (seeds 0–4), GRU (seeds 0–2), analogue (seed 0), SNN β=0.9 (seed 0), SNN β=0.95 (seed 0) |
| Output | `docs/reports/data/exp_001_memory_horizon.json`, committed |

**Mechanism of the probe.** Every arm's only cross-time carrier is its recurrent
state — per-layer membrane `v` for the spiking and analogue arms, hidden `h` for
the GRU — and all three zero it when `state=None`. Truncating every `k`
characters is therefore *exactly* equivalent to reshaping the input window
`[B, L] → [B·L/k, k]` and running with fresh state. No model code is modified and
no state-zeroing hook is introduced, which removes the class of bug where the
probe measures its own implementation. `k` divides `L = 256` in every case, so no
window is ragged.

The same (context, target) character pairs are scored at every `k`; only the
length of context available to predict each one changes. Character counts are
therefore identical across the sweep by construction and are asserted, not assumed.

## 4. What is recorded regardless of outcome

Per arm and per `k`: bpc, summed nats, character count, per-layer firing rate,
and wall-clock. Per arm: the `k = 256` self-check residual against the committed
Phase-2 number. Curves are reported for **every** arm including the ones that make
no argument, and a flat GRU curve — which would falsify the project's working
assumption about where the gap comes from — is reported as prominently as an
informative one.

## 5. Known limitations, stated in advance

1. **Truncated evaluation is off-distribution for k < 256.** The models were
   trained with 256-character windows. `bpc(k)` therefore measures how much the
   *trained solution* depends on old state, not how well a model *trained* at
   horizon `k` would do. That is the intended question — it is a probe of the
   learned function, not a claim about attainable performance at short horizon —
   and no result here may be quoted as the latter.
2. **This cannot separate "does not use long context" from "cannot use long
   context."** A model that never learned to use a horizon it possesses looks
   identical to one that lacks it. The candidate ranking must treat a short
   measured horizon as necessary-but-not-sufficient evidence for the M1 reading.
3. **The GRU arm violates I5 and is an anchor, not a target.** Its curve bounds
   what a lateral matrix buys at this scale; it is not a specification for what a
   compliant neuron should achieve.
4. **n = 1 for the analogue and β-variant arms.** Those curves are shape evidence
   only; no effect below 2σ may be read off them.

## 6. Failure modes that would invalidate this experiment

| # | Failure | Detection |
|---|---|---|
| F1 | The probe's `k = 256` point does not reproduce the committed `fresh` bpc | Asserted directly against `phase2_final_scores.json`; a residual above 1e-3 bpc aborts the run |
| F2 | Reshaping changes which characters are scored | `n_chars` compared across every `k`; must be identical |
| F3 | A checkpoint is loaded with the wrong config (e.g. the β variant scored as β=0.5) | Config hash read from the checkpoint and recorded per row |
| F4 | The GRU's state convention differs from the spiking arms', so truncation means something different for it | Both are "zero the state at sequence start"; the reshape path is identical for all arms and exercises each arm's own `state=None` branch |
| F5 | Batch-shape change alters numerics enough to matter | Residual at `k = 256` (F1) bounds it; the comparison threshold is 0.0092 bpc and reduction-order effects are ~1e-6 |

## 7. Entry conditions

- [ ] Phase-2 checkpoints present for every named arm
- [ ] `docs/reports/data/phase2_final_scores.json` present, for the F1 self-check
- [ ] Test suite green on this branch before the probe is trusted

---

## 8. Results

*(appended after the run; nothing above this line is edited)*
