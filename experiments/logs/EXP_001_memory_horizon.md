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

**Run 2026-08-02.** 11 checkpoints, full enwik8 test split (4 980 736 characters
each), 9 truncation lengths, ~25 s of GPU per checkpoint. Raw JSON in
`docs/reports/data/exp_001_memory_horizon.json`.

**F1 passed on all 11 checkpoints.** The `k = 256` point reproduces the committed
Phase-2 `fresh` bpc to between 4.4e-10 and 1.3e-08 bpc — the probe is the same
measurement as the baseline, reached by a different route. F2 passed: 4 980 736
characters scored at every `k`, for every arm.

### 8.1 An amendment, made before the verdicts and labelled as one

The pre-registered statistic is **confounded**, and this was found during the
first smoke run, before the full sweep. `bpc(k)` averages over every position
inside a chunk, so a fraction `1/k` of all scored characters sit at position 0
with no context at all. A model with a *short* horizon but expensive first
characters therefore still traces a smooth curve improving all the way out to
`k = L`, which looks exactly like long memory.

That is not a hypothesis about the confound; it is measurable, and it was
measured. If the whole truncation penalty is boundary amortisation then

```
bpc(k) − bpc(L)  =  (L/k − 1)/L  ·  Σ_{c<k} excess(c)
```

**exactly, with no free parameter.** Reconstructing the pre-registered curve from
the paired one this way reproduces it to a maximum absolute residual of **0.0072
to 0.0282 bpc**, against truncation penalties of 0.47–0.83 bpc — i.e. **97–99 %
of the pre-registered signal is boundary amortisation.** The statistic carries
almost no independent information about memory.

So a second statistic was added: the **paired context curve** (§3 of
`scripts/exp/001_memory_horizon.py`). For a fixed position `p`, the model, the
window and the target character are identical between the truncated and
untruncated runs and only the available context differs, so text difficulty —
which dominates the ~0.02 bpc per-position standard error of the naive
alternative — cancels in the difference.

**This addition was made after seeing the smoke-test curve.** It is recorded as
post-hoc, the pre-registered predictions are resolved below against the statistic
they were written for, and both curves are reported in full.

### 8.2 The pre-registered predictions, resolved as written

| # | Prediction | Measured | Verdict |
|---|---|---|---|
| **P1** | `bpc_snn(8) − bpc_snn(256) < 0.0092` | **+0.5232** | **FAILED as written** |
| **P2** | `bpc_snn(1) > 3.0` | 4.1462 | held |
| **P3** | `bpc_gru(8) − bpc_gru(256) > 0.10` | **+0.8347** | held |
| **P4** | horizon: SNN ≤ 8, GRU ≥ 64 | SNN **7**, GRU **57–60** | SNN half held; **GRU half failed** (57–60 < 64) |

P1 failed on the confounded statistic and its *substance* — that the SNN's
membrane carries nothing usable past ~8 characters — holds on the corrected one,
at +0.0030 bpc of excess at 8 characters of context, a third of the 2σ bar. Both
facts are recorded. P4's GRU half missed its threshold by 4–7 characters; it is
recorded as failed rather than rounded into a pass.

### 8.3 The number this experiment exists to produce

Memory horizon = the shortest context beyond which more context is worth less
than 2σ = 0.00922 bpc, measured on the paired curve.

| Arm | n | **Horizon (chars)** | Excess at 8 chars | Excess at 16 chars | Test bpc |
|---|---:|---:|---:|---:|---:|
| Analogue control (β=0.5) | 1 | **6** | +0.0007 | +0.0000 | 2.2806 |
| **SNN baseline (β=0.5)** | 5 | **7** *(7,7,7,7,7)* | +0.0030 | −0.0000 | 2.2697 |
| SNN β=0.9 | 1 | **24** | +0.1040 | +0.0262 | 2.3276 |
| SNN β=0.95 | 1 | **35** | +0.1246 | +0.0451 | 2.3702 |
| **GRU anchor** *(violates I5)* | 3 | **57–60** | +0.1970 | +0.0800 | 1.8044 |

All five SNN seeds return **exactly 7**. All three GRU seeds return 57–60.

### 8.4 What it means — and the part that was not predicted

**The GRU reaches roughly 8× further back than the spiking baseline can.** That is
the mechanism behind the 0.486 bpc that `02_baseline_report.md` §7.2 attributes to
invariant I5, and it is now measured rather than inferred.

**But the β arms falsify the naive reading of that, inside this same experiment.**
Raising β from 0.5 to 0.95 *does* buy horizon — 7 → 24 → 35 characters, a 5×
increase — and bpc gets monotonically **worse** over exactly that range (2.2697 →
2.3276 → 2.3702). Horizon is **necessary but not sufficient**, and the pre-
registered M1/M2 dichotomy in §1 was too coarse: this is neither "the gap is
length of memory" nor "length of memory is irrelevant".

The resolution the data supports: a scalar leak buys horizon by **low-pass
filtering**, which lengthens memory by *blurring* it — the same state variable
that holds the distant past also has to hold the present character, and at β=0.95
the present is 1/20th of the signal. The GRU is not merely *longer* than the
spiking baseline; it is longer **while keeping recent context sharp**, because a
gate lets each unit choose per step whether to retain or overwrite.

**The Phase-3 consequence is sharper than "buy horizon".** A candidate must
lengthen memory *without* degrading the fidelity of recent context — which means
**separating** the fast and slow paths rather than slowing the single existing
one. That is precisely the distinction between the §4.6-admitted multi-timescale
and rotational forms (separate state variables) and the β knob (one state
variable, slower), and it converts `02_baseline_report.md` §7.3's "structure, not
magnitude" from a slogan into a quantitative, testable prediction:

> **A two-compartment neuron should reach a horizon materially longer than 7
> characters while its excess at 1–4 characters of context stays at the
> baseline's level. Raising β lengthens the horizon and raises the near-context
> excess together (β=0.9: +0.4660 at 3 chars vs the baseline's +0.2913). Any
> candidate that shows the β signature is buying horizon the losing way.**

That is a pre-registerable acceptance criterion for Phase 4, and it is carried
into `03_phase3_candidates.md` as one.

Two further observations, recorded because they were not asked for:

* **The analogue control's horizon is 6 — statistically the same as the spiking
  arm's 7.** Binarity and the hard threshold do not shorten memory. That is an
  independent confirmation, on a different axis, of §7.2's finding that binarity
  is nearly free: the two arms differ in emission, not in reach.
* **The GRU's cost of *zero* context is higher than the SNN's** (+2.2379 vs
  +1.8920 bpc at c = 0). It is not that the GRU is uniformly better at every
  context length — it is that the GRU has more to lose, because it has more
  invested in context. At 0–2 characters the two arms are much closer than their
  headline gap suggests.

### 8.5 Failure modes, checked

| # | Failure | Outcome |
|---|---|---|
| F1 | `k = L` does not reproduce the committed number | Passed, 11/11, residual 4.4e-10 to 1.3e-08 bpc |
| F2 | Reshaping changes which characters are scored | Passed: 4 980 736 characters at every k, every arm |
| F3 | Wrong config loaded for a checkpoint | Config hash recorded per row; β=0.9 and β=0.95 arms reproduce their own committed bpc under F1, so they are the models they claim to be |
| F4 | GRU state convention differs | Both arms exercise their own `state=None` branch through one shared code path; the GRU's F1 residual is 1.6e-09 to 1.0e-08 |
| F5 | Batch-shape change alters numerics | Bounded by F1 at ~1e-08 bpc, six orders below the 0.00922 threshold |

### 8.6 Status

**CLOSED.** Deliverable produced. P1 failed as written and its substance holds
under the corrected statistic; P4's GRU half failed; the confound in the
pre-registered design was found, quantified, and reported rather than absorbed.
The unpredicted β result in §8.4 is the finding that most changes Phase 3.
