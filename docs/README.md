# Documentation and evidence index

Two kinds of thing live under `docs/`:

* **`docs/reports/*.md`** — the phase reports. Prose, arguments, verdicts.
* **`docs/reports/data/*.json`** — the evidence. Every number quoted in a report
  traces to a tracked file here.

That second rule is the reason this directory exists in the shape it does.
Training runs and checkpoints are bulky and are *not* committed; the curated
artifacts extracted from them are. A report is therefore auditable without
shipping gigabytes, and a number that appears in prose but in no artifact is a
defect.

> The leading slash in `.gitignore`'s `/data/` entry is load-bearing. A bare
> `data/` would also ignore `docs/reports/data/` and silently stop committing all
> of the evidence below.

---

## Reports, in reading order

| Report | Phase | What it establishes |
|---|---|---|
| [`01_reconnaissance.md`](reports/01_reconnaissance.md) | 1 | audited environment, cost model, risk register, benchmarking standard |
| [`01a_literature_review.md`](reports/01a_literature_review.md) | 1 | the literature survey, and which of its claims survived independent re-measurement |
| [`02_baseline_report.md`](reports/02_baseline_report.md) | 2 | the baseline, the σ = 0.00461 noise floor, the analogue control, the GRU anchor |
| [`02a_phase2_spec.md`](reports/02a_phase2_spec.md) | 2 | the frozen baseline specification |
| [`03_phase3_candidates.md`](reports/03_phase3_candidates.md) | 3 | the 18-candidate set, its ranking, and the acceptance criteria |
| [`04_phase4_interim.md`](reports/04_phase4_interim.md) | 4 | **current, open** — running record, findings, and the open decisions in §8 |

`04_phase4_interim.md` is an **interim** report and says so. Phase 4 is open; the
document is the running record of it, and its §8 decision table is the live list
of what is waiting on the author.

## Pre-registered experiment logs

One file per experiment in [`../experiments/logs/`](../experiments/logs/), written
**before** the run and never edited afterwards — corrections are appended, not
applied in place.

`EXP_000`, `EXP_001` … `EXP_009`, `EXP_011`, `EXP_012`, `EXP_013`.

**There is no `EXP_010`.** The id is unused. The `010_` prefix on
`scripts/exp/010_phase4_arm_results.py` is the results resolver for `EXP_007`,
`EXP_008` and `EXP_009`, not a tenth experiment.

---

## Evidence artifacts, and what produced each

Producers below were derived from the source tree; where a script takes its
destination as an argument rather than hard-coding it, the canonical invocation is
given. Audit commands `01`–`05` are quoted from
[`01_reconnaissance.md`](reports/01_reconnaissance.md) §10, which is their
canonical source.

### Phase 1 — audits

| Artifact | Produced by |
|---|---|
| `audit_01_gpu_capability.json` | `scripts/audit/01_gpu_capability.py <out>` |
| `audit_02_snn_cost_model.json` | `scripts/audit/02_snn_cost_model.py <out>` |
| `audit_03_kernel_count.json` | `scripts/audit/03_kernel_count.py <out>` |
| `audit_04_graphed_scaling.json` | `scripts/audit/04_graphed_scaling.py <out>` |
| `audit_05_verify_claims.json` | `scripts/audit/05_verify_literature_claims.py <out>` |
| `audit_06_neuron_cross_check.json` | `scripts/audit/06_neuron_cross_check.py` |
| `audit_07_init_pathology.json` | `scripts/audit/07_init_pathology.py` |
| `audit_08_mutation_campaign.json` | `scripts/audit/08_mutation_campaign.py` |
| `audit_09_gradient_reachability.json` | `scripts/audit/09_gradient_reachability.py` |
| `audit_10_candidate_spec.json` | `scripts/audit/10_candidate_spec.py` |

`06_neuron_cross_check.py` is the one that checks the hand-written LIF against
**snnTorch**, which is why snnTorch is a dependency of the repository and not of
the model.

`08_mutation_campaign.py` is the gate-of-gates: it mutates the kernels and
asserts the test suite catches every mutant (25/25). It **must own the machine
while it runs** — Python reads a module once at import, so anything that starts
mid-campaign keeps the mutated version for its whole life.

### Phase 2 — baseline

| Artifact | Produced by |
|---|---|
| `phase2_campaign.json` | `scripts/run_phase2_campaign.py` |
| `phase2_final_scores.json` | `scripts/score_phase2.py` |
| `phase2_kernel_bench.json` | `scripts/exp/002_candidate_neuron_cost.py` |

### Phase 3 — candidates

| Artifact | Produced by |
|---|---|
| `phase3_candidates.json` | `scripts/audit/10_candidate_spec.py` |

This artifact is a **reconstruction** and is labelled as one. It leaves
unattributable fields `null` rather than guessing, and carries an
`unverifiable_claims` list that its validator prints on every run.

### Phase 4 — experiments

| Artifact | Produced by |
|---|---|
| `exp_001_memory_horizon.json` | `scripts/exp/001_memory_horizon.py` |
| `exp_002_candidate_neuron_cost.json` | `scripts/exp/002_candidate_neuron_cost.py` |
| `exp_003_depth_ladder.json` | `scripts/exp/001_memory_horizon.py --out …` — EXP_003's horizons went through EXP_001's paired statistic deliberately, so the rows are comparable |
| `exp_004_gradient_reachability.json` | `scripts/exp/004_gradient_reachability.py` |
| `exp_004_memory_horizon.json`, `exp_004_twocomp_results.json` | `scripts/exp/004_twocomp_results.py` |
| `exp_005_run_manifest.json` | `scripts/exp/005_run_seeds.py` |
| `exp_005_memory_horizon.json`, `exp_005_sigma_transfer.json` | `scripts/exp/005_sigma_transfer.py` |
| `exp_006_slow_tail_ablation.json` | `scripts/exp/006_slow_tail_ablation.py` |
| `exp_007_008_run_manifest.json` | `scripts/exp/007_run_prescan_arms.py` |
| `exp_007_009_arm_results.json`, `exp_007_009_memory_horizon.json` | `scripts/exp/010_phase4_arm_results.py` |
| `exp_008_fold_residual.json` | `scripts/exp/008_chase_fold.py` |
| `exp_009_run_manifest.json` | `scripts/exp/009_distill.py` |
| `exp_009_divergence.json` | `scripts/exp/009_chase_divergence.py` |
| `exp_011_run_manifest.json` | `scripts/exp/011_compose.py` |
| `exp_011_arm_results.json`, `exp_011_memory_horizon.json` | `scripts/exp/011_compose_results.py` |
| `exp_011_divergence.json` | `scripts/exp/011_chase_compose_divergence.py` |
| `exp_012_fold_gap.json` | `scripts/exp/012_chase_fold_gap.py` |
| `exp_012_pileup.json` | `scripts/exp/012_posthoc_pileup.py` |

Several resolvers write **more than one** artifact, and two of them *rewrite* an
artifact an earlier experiment produced — `012_chase_fold_gap.py` writes
`exp_008_fold_residual.json`, and both `010_` and `011_chase_compose_divergence.py`
write `exp_009_divergence.json`. That is intentional (a later experiment extends an
earlier record), and it is the reason drivers take an explicit `--out`: so that
re-running one cannot silently overwrite a committed artifact belonging to another.

---

## Reproducing a number

1. **Acquire the corpus.** `python scripts/data/download_corpus.py --corpus enwik8`.
   Idempotent — a second run re-verifies the SHA-256 on disk and touches no
   network. That re-verification *is* gate A5.
2. **Re-run the producing script** from the table above.
3. **Compare against the committed JSON**, which is tracked and therefore
   diffable.

Absolute timings are hardware- and driver-specific. The **relationships** are the
reproducible claims: flat-in-size launch cost, linear-in-kernel-count wall-clock,
~11× CUDA-graph speed-up, ~4.9× jiterator fusion.

For anything that trains, launch it detached (`scripts/launch.py`) and expect
GPU-hours, not minutes. See [`../CONTRIBUTING.md`](../CONTRIBUTING.md) §6.
