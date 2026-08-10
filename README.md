# A character-level spiking language model

A from-scratch spiking neural network trained as a character language model on
enwik8, built as a **phased, pre-registered study** rather than as a codebase that
happens to have experiments in it. Every number in every report is backed by a
committed JSON artifact; every experiment was written down, with its decision
threshold, before it was run.

The research question is narrow and stated plainly: **how much of the gap between
a spiking model and a comparable non-spiking one is architectural, and which
mechanisms actually close it?**

---

## Status

**Phase 4 of 5, open.** This is a live study, not a finished result.

| | |
|---|---|
| Phases 1–3 | complete |
| Phase 4 | **open** — running record in [`docs/reports/04_phase4_interim.md`](docs/reports/04_phase4_interim.md) |
| Arms adopted | **2** — the two-compartment neuron, and the learned per-channel threshold (standalone) |
| Arms recommended but *not* taken | 1 — the composition, provisional at n = 2 |
| Open decisions | **6 of 10**, all the author's (§8 of the Phase-4 report) |
| Budget | ~8.0 of ~30 GPU-hours spent |

Two consequences worth stating before anyone reads a number out of this
repository:

* **Nothing here is a recommendation to adopt anything you have not been shown
  the caveats for.** Three arms have beaten the baseline; two are adopted. The
  third — the composition — is provisional and the reports say so in every place
  it appears. The threshold arm's own headline still carries the `W1 fails`
  caveat until decision #6 is ruled.
* **The composition result is provisional at n = 2** of a pre-registered 3. One
  seed diverged, and it was **not** replaced — replacing it would have removed a
  failure from the record.

## Results so far

Full enwik8 test split, scored under **both** evaluation protocols. "fresh"
zeroes membrane state at each window (matches the training distribution);
"carried" carries state across contiguous streams (the protocol published enwik8
figures use). Reporting only one would flatter the model in one direction or the
other, so both are always reported.

| | baseline (n=5) | **two-compartment** (n=7) | GRU anchor (n=3) |
|---|---:|---:|---:|
| test bpc, carried | 2.25311 | **2.11869** | 1.76741 |
| test bpc, fresh | 2.26969 | **2.14566** | 1.80443 |
| sd over seeds | 0.00461 | 0.00313 | 0.00223 |
| memory horizon | 7 | **47** | 57–60 |

735 437 parameters. The GRU anchor is a **reference point, not a competitor** — it
violates the study's own spiking constraint (I5) and exists to say how far there
is to go.

The headline is not the mean improvement. **Decomposed by context length, the
two-compartment neuron's gain is three effects of opposite sign**: +41 % at zero
context, **−48 % inside the 7 characters the baseline already reached**, and
+107 % beyond that horizon. Quoted as one number it reads as a uniform win while
nearly half its magnitude is handed back. That decomposition is the single most
useful thing this project has produced about how to read an SNN result.

**The largest single effect measured in Phase 4 is not an arm — it is width.**
`EXP_014` trained the *plain* baseline neuron at 2.5M and 5.0M parameters and
took **2.25311 → 2.00073 carried**, a −0.252 bpc move at 54.75 σ. That is more
than every architectural arm in the table above, at **6.8× the parameters** — so
it **refutes none of them**; it says the project had been spending its budget on
a 0.03–0.17 bpc axis with an unmeasured 0.25 bpc axis beside it. Read it with
three caveats the report states in full: it is **n = 1 per size** and adopts
nothing, it is **not parameter-matched** against any arm, and at 6.8× the
parameters the model is **still 0.233 bpc behind the GRU anchor at 1×**.

Selected findings that are expensive to re-derive are collected in
[`docs/reports/04_phase4_interim.md`](docs/reports/04_phase4_interim.md); the
short version is that **the model underfits at 735K parameters and stops doing so
somewhere between 2.5M and 5M** (the gap is within noise of zero at 735K and
+0.055 at 5.0M, while test bpc improves by 0.25 — memorisation alongside a real
gain, not damage), **the learned threshold is provably a reparameterisation**
(so its gain is an optimisation effect, not a capacity one), and **two training
divergences that looked identical in the logs had different causes**.

One more, because it constrains how any number here may be compared: the fp64
gradient-clip fix adopted as decision #7 is **not numerically inert**. It fires
about three times in a 20,000-step run, and those three steps are worth
+1.97e-03 bpc by the end. Every figure above predates it, so the arms remain
mutually comparable with each other — but a **newly trained** run cannot be
compared bitwise against them, and new experiments train their own anchor
instead.

---

## Install

**Order matters.** PyTorch must come from NVIDIA's index first, because the pin is
load-bearing twice over: the wheel has to be the CUDA 13.0 / sm_120 build for
Blackwell, and the fused kernels use `torch.cuda.jiterator`, a private API only
guaranteed at this version.

```bash
pip install torch==2.12.1 --index-url https://download.pytorch.org/whl/cu130
pip install -r requirements.txt
pip install -e .                     # makes `import snn` work anywhere
```

`pip install -e .` is optional — every script under `scripts/` prepends `src/` to
`sys.path` itself, so the repository works without it. It exists so that editors,
type checkers and a bare `python -c "import snn"` resolve the same package the
tests do.

Audited environment: **Python 3.14.6**, torch 2.12.1 (CUDA 13.0, sm_120), numpy
2.5.0. It is the only interpreter this project has been measured on; see
[`docs/reports/01_reconnaissance.md`](docs/reports/01_reconnaissance.md) §2.2 for
the full audited stack.

> **`triton` is deliberately absent.** It has no Windows wheel for this stack, so
> `torch.compile(backend="inductor")` is unavailable. Fusion is *not* unavailable:
> NVRTC ships inside the torch wheel and `jiterator` compiles elementwise CUDA
> against it at runtime.

## Quickstart

```bash
# 1. Acquire and verify the corpus (idempotent; re-runs verify checksums only)
python scripts/data/download_corpus.py --corpus enwik8

# 2. Train the Phase-2 baseline
python scripts/train.py --corpus enwik8 --d_model 512 --n_layers 2

# 3. Score a checkpoint under BOTH protocols
python scripts/evaluate.py --ckpt experiments/runs/<run>/ckpt_best.pt --split test
```

**For any run longer than a few minutes, launch it detached:**

```bash
python scripts/launch.py -- --corpus enwik8 --max_steps 20000
```

`scripts/launch.py` is not a convenience. v1 of this project lost long runs
because they were children of an interactive shell that went away (risk R6);
`launch.py` starts the run as a detached OS process so it survives its parent.

## Tests

```bash
pytest                              # everything the machine can run
pytest -m "not cuda and not data"   # no GPU, no corpus — 259 tests, ~8s
ruff check .                        # lint gate
```

`pytest` reports skips by name (`-ra`) on purpose. A suite that silently skips
half of itself and prints "passed" is a failure mode this project has already
met once, in a training run that had imported a mutated kernel.

The test suite is not only unit tests. It includes **equivalence gates** — fused
kernel against eager reference, CUDA-graph path against non-graph, folded weights
against unfolded — and the gates are themselves mutation-tested
(`scripts/audit/08_mutation_campaign.py`, 25/25 mutants caught). A test that has
never been shown capable of failing is not trusted here.

---

## Repository layout

| Path | What is in it |
|---|---|
| `src/snn/` | the model. `neuron.py` (LIF), `twocomp.py`, `prescan.py`, `kernels.py` (fused), `train.py`, `evaluate.py`, `data.py`, `config.py` |
| `scripts/train.py`, `evaluate.py`, `launch.py` | entry points |
| `scripts/audit/` | Phase-1 environment and capability audits, numbered `01`–`10` |
| `scripts/exp/` | one driver per experiment, numbered by `EXP` id |
| `tests/` | unit tests **and** the equivalence gates |
| `experiments/logs/` | **pre-registered** hypotheses, one `EXP_<ID>.md` each |
| `docs/reports/` | the phase reports |
| `docs/reports/data/` | committed JSON evidence — every number in a report traces to a file here |

`data/`, `experiments/runs/` and checkpoints are **not** committed. Curated
evidence extracted from them is, which is what makes a report auditable without
shipping gigabytes.

## Reports

Read them in order; each assumes the last.

| Report | What it establishes |
|---|---|
| [`01_reconnaissance.md`](docs/reports/01_reconnaissance.md) | audited environment, cost model, risk register |
| [`01a_literature_review.md`](docs/reports/01a_literature_review.md) | claims from the literature, and which survived checking |
| [`02_baseline_report.md`](docs/reports/02_baseline_report.md) | the Phase-2 baseline, the σ = 0.00461 noise floor, the analogue control |
| [`02a_phase2_spec.md`](docs/reports/02a_phase2_spec.md) | the frozen baseline specification |
| [`03_phase3_candidates.md`](docs/reports/03_phase3_candidates.md) | the 18-candidate set and its ranking |
| [`04_phase4_interim.md`](docs/reports/04_phase4_interim.md) | **current** — the running Phase-4 record and the open decisions |

## How this project is run

The protocol is the point, and it is not optional. In short:

* **Pre-register or it does not count.** Hypotheses and decision thresholds are
  written to `experiments/logs/EXP_<ID>.md` *before* the run. Results are
  appended; the hypothesis is never rewritten.
* **Negative results stay, prominently.** A Phase-2 prediction about β was wrong
  and the report says so in its own voice.
* **A bar is reported as it fired.** When a pre-registered rule returns an
  unhelpful verdict, the verdict is reported and the *corrected* rule is referred
  to the next phase — not applied retroactively.
* **Never widen a tolerance to make a residual go away.** Chase it.
* **Phases end at a human gate.** Deliverables are completed, exit criteria are
  documented, and then the phase stops and waits for review.

Full version: [`CONTRIBUTING.md`](CONTRIBUTING.md).

## Licence

MIT — see [`LICENSE`](LICENSE).
