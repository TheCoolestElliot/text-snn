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

*Not yet run. This section is appended to, never rewritten.*
