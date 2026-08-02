# EXP_000 — Seed-to-seed variance of the Phase-2 baseline

**Pre-registered:** 2026-08-01, before any training run was launched.
**Status:** PRE-REGISTERED — no results yet.
**Phase:** 2 (baseline reproduction). Required by `01_reconnaissance.md` §8.3
("Science") and risk **R8**.

---

## 1. Why this runs first

Every later conclusion in this project is a comparison. "Reset-free beats hard
reset by 0.04 bpc" is a *claim about a difference*, and a difference is only
meaningful against the spread of runs that differ by nothing at all.

The Phase-1 literature survey found (§4.5 item 6) that **nobody has published the
seed-to-seed standard deviation of bits-per-character for surrogate-gradient SNN
training.** Everyone imports a figure from word-level LSTM work. That number does
not transfer: SNN training has a discrete, non-differentiable forward pass and a
surrogate backward, which is a materially different noise process.

So the noise floor is measured here, before any ablation is believed. Until this
number exists, the honest statement about any Phase-3/4 result is "we do not know
whether that is signal."

## 2. Hypothesis

**H0 (null, and the one we expect to fail to reject):** identical configurations
differing only in `seed` produce test bpc drawn from a single distribution whose
standard deviation σ is small relative to the effect sizes Phase 4 will chase.

This experiment does not test H0 against an alternative. It **estimates σ**. The
deliverable is a number and a decision threshold, not a p-value.

## 3. Design

| Field | Value |
|---|---|
| Arms | 1 (identical config) |
| Replicates | 5 — seeds `{0, 1, 2, 3, 4}` |
| Varied | `cfg.seed` only. Every other field is byte-identical, asserted by comparing `config_hash` with `seed` excluded |
| Corpus | enwik8, standard 90/5/5 MB byte splits |
| Config | The frozen Phase-2 baseline, `02a_phase2_spec.md` §13: `d_model=512`, `n_layers=2`, `batch=128`, `seq_len=256`, `beta=0.5`, `threshold=1.0`, `reset=hard`, `atan(α=2)`, `T=1`, fp32 |
| Budget | `max_steps` fixed in advance (not early-stopped), identical for all seeds |
| Measured | test bpc under **both** evaluation protocols (fresh-windowed and carried-contiguous), val bpc, per-layer firing rate, wall-clock, peak VRAM |

`seed` controls: parameter init, training batch order, and nothing else. Data
order is a deterministic function of `(seed, step)`, so seeds differ in *which*
windows they see, which is part of the noise being measured and is deliberately
not held fixed.

## 4. What is recorded, regardless of outcome

Per seed: the full `log.jsonl` trajectory, final val and test bpc under both
protocols, per-layer firing rates, wall-clock, peak VRAM, and the resolved config.
Curated summary lands in `docs/reports/data/exp_000_seed_variance.json` and is
committed. **Negative or ugly results are retained** — in particular, if the
baseline turns out to be weak (see §6), that is the finding, not a failure.

## 5. Decision rule — fixed now, so it cannot be chosen later

Let σ be the sample standard deviation of test bpc across the 5 seeds (carried
protocol, the headline number).

* An ablation in Phase 3/4 is reported as a **real effect** only if its mean over
  ≥3 seeds differs from the baseline mean by **more than 2σ**.
* An effect between 1σ and 2σ is reported as **"within noise; not adopted"** and
  the numbers are still published.
* An effect below 1σ is reported as **no effect**.
* If σ > 0.05 bpc, the ablation budget must rise to 5 seeds per arm and the
  Phase-4 plan is re-costed against §9's budget model before proceeding. This
  branch is written down now precisely so that discovering a large σ cannot be
  quietly absorbed.

## 6. Declared expectation (so it cannot be rationalised afterwards)

`beta = 0.5` gives a membrane half-life below one character, so the baseline's
memory horizon is expected to be very short and its bpc correspondingly weak —
plausibly worse than the text8 5 M-parameter LSTM anchor of 1.50, on a corpus
where no sub-17 M-parameter spiking result exists at all (§4.2). **That is the
honest reference point.** It is written here in advance so that a weak baseline is
recorded as the predicted consequence of the reference hyperparameters rather than
re-described after the fact as a bug or a success.

A separate, clearly-labelled `beta ∈ {0.5, 0.9, 0.95}` check runs as *baseline
establishment* — confirming the arm is not accidentally crippled by one
hyperparameter. It is **not** an ablation result and does not change the headline
configuration.

## 7. Failure modes that would invalidate this experiment

| # | Failure | Detection |
|---|---|---|
| F1 | Runs are not actually identical apart from the seed | `config_hash` compared across all 5, seed field excluded |
| F2 | Non-determinism inside a single seed inflates σ | `tests/test_determinism.py` must pass first: same seed ⇒ identical loss trajectory |
| F3 | A seed diverges or collapses to silence (zero firing) | Per-layer firing rate logged from step 0 (R4); a dead-network run is reported, not silently re-rolled |
| F4 | σ estimated from 5 samples is itself noisy | Acknowledged. σ is reported with its own uncertainty; the 2σ rule is deliberately conservative to absorb this |
| F5 | Fused and eager paths disagree, so "the baseline" is ambiguous | R10 gate (`test_neuron_equivalence.py`) must pass before this experiment runs |

## 8. Entry conditions — none of this runs until all are green

- [ ] R10 gate passes: fused vs eager forward bit-identical, backward within tolerance, all four reset modes
- [ ] R5 gate passes: graphed vs eager losses agree over ≥20 steps
- [ ] `test_determinism.py` passes: same seed ⇒ identical trajectory; checkpoint-resume is exact
- [ ] `test_data.py` passes: corpus checksums verified, vocab 205, every character scored exactly once by both eval protocols

---

## 9. Results

**Run 2026-08-01.** 5 runs, 0 failures, 374–399 s each, 0.61 GiB peak VRAM.
Scored on the full enwik8 test split from the final checkpoint. Raw JSON in
`docs/reports/data/phase2_final_scores.json`.

### 9.1 The number

| Seed | Test bpc (carried) | Test bpc (fresh) |
|---:|---:|---:|
| 0 | 2.255477 | 2.27186 |
| 1 | 2.254172 | 2.27058 |
| 2 | 2.253759 | 2.27049 |
| 3 | 2.245182 | 2.26188 |
| 4 | 2.256985 | 2.27353 |
| **mean** | **2.25311** | 2.26967 |
| **σ** | **0.00461** | 0.00447 |

> **σ = 0.00461 bits/character** over 5 identical-except-seed runs.

### 9.2 The decision rule, now resolved to numbers

| Effect size | Verdict |
|---|---|
| > **0.00922 bpc** (2σ) | real effect, may be adopted |
| 0.00461 – 0.00922 | within noise; published, not adopted |
| < 0.00461 | no effect |

§5's replan trigger (σ > 0.05) is **not triggered** — σ is roughly 11× below it.
The Phase-4 budget stands at **3 seeds per arm** and needs no re-costing.

Worth stating positively rather than as a mere pass: at 0.0046 bpc the instrument
resolves **0.01 bpc effects at three seeds**. That is finer than the campaign was
designed assuming, so Phase 4 can afford to ask sharper questions.

### 9.3 Failure modes, checked

| # | Failure | Outcome |
|---|---|---|
| F1 | Runs not identical apart from seed | Config hashes differ only where seed does; all five share every other field |
| F2 | Non-determinism inflating σ | `test_determinism.py` passed before the campaign: same seed ⇒ identical trajectory, resume bit-exact |
| F3 | A seed collapsing to silence | None did. Final firing rates 0.337–0.353 (layer 0) and 0.321–0.335 (layer 1) across all five |
| F4 | σ from 5 samples is itself noisy | Acknowledged and unresolved by design; the 2σ rule absorbs it. Seed 3 is the outlier at 2.2452, 1.7σ below the mean |
| F5 | Fused/eager disagreement making "the baseline" ambiguous | R10 gate passed, and was mutation-tested 21/21, before this ran |

### 9.4 On §6's declared expectation — it was wrong

§6 predicted, in advance, that β = 0.5's sub-one-character memory horizon would
make the baseline weak, and asked that a weak result be recorded as the predicted
consequence of the reference hyperparameters rather than re-described afterwards.

The prediction failed, and it is recorded here as failed. β = 0.5 is the **best**
of the three values checked: β = 0.9 costs +0.0511 bpc (11.1σ) and β = 0.95 costs
+0.0929 bpc (20.1σ). The baseline is not crippled by its decay constant, and a
longer membrane time constant makes things monotonically worse.

The mechanism visible in the logs: as β rises the second layer goes quieter
(0.33 → 0.23 → 0.22 mean firing rate), i.e. each neuron's output depends more on
accumulated past current and less on the present character. For next-character
prediction that is a losing trade. Membrane leak is a low-pass filter, and a
low-pass filter is a poor place to store linguistic context.

This is a more useful result than the one predicted: it says Phase 3 should pursue
the *structure* of neuron memory (the multi-timescale, adaptive-threshold and
rotational forms admitted by the §4.6 ruling), not its *length*.

### 9.5 Status

**CLOSED.** Deliverable produced; decision rule resolved; §6's expectation
falsified and recorded. Full analysis in `docs/reports/02_baseline_report.md` §7.
