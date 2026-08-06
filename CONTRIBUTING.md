# Working in this repository

This is a research repository run under a protocol, and the protocol is the
deliverable as much as the model is. Most of what follows is about **how a claim
becomes trustworthy**, not about code style.

---

## 1. The phase gate

The study runs in five phases. **A phase ends at a human gate.** Complete the
deliverables, document the exit criteria, then *stop and request review*. Never
auto-advance.

This exists to prevent post-hoc rationalisation. The rules only work if decisions
are written down **before** the evidence that would bias them arrives.

Current state and the open decisions live in
[`docs/reports/04_phase4_interim.md`](docs/reports/04_phase4_interim.md) §8.
**Decisions in that table are the author's.** An open decision's evidence may be
corrected in prose beneath the table; the row itself is edited only when the
decision is actually made.

## 2. Pre-registration

Write `experiments/logs/EXP_<ID>.md` **before** running anything, including the
decision threshold. Append results. **Never rewrite the hypothesis.**

* If committing before the run is not possible, stamp the log's SHA-256 into the
  run manifest before the first run and re-check it after the last. `EXP_013`
  does this and records the recipe for reconstructing the pre-results bytes, so
  the guarantee survives appending results.
* **Check every pre-registered bar against the experiment's own limitations
  section before committing it.** This has failed twice, the same way both times:
  `EXP_012`'s Y5 resolved on a leg comparing 0 flips against 1, and `EXP_013`'s
  N1 fired at two amplitudes whose test bpc was catastrophically *worse*, because
  it constrained a difference without constraining both of its terms.
* **State a prediction's power before running it.** "The band was failed" and
  "the difference was not established" are different sentences and both may be
  true.
* **Read every prediction's prose against its formula.** `EXP_013` pre-registered
  two predictions as opposite-direction hypotheses that were the same measurement
  with opposite signs.

## 3. Reporting

* **Negative results and falsified predictions stay, prominently.**
* **Report a near-miss as a miss.** `EXP_008`'s W3 failed at 49.7 % against a
  50 % bar. The near-ness is worth discussing; the verdict is not negotiable.
* **A bar is reported as it fired.** If a pre-registered rule returns an unhelpful
  verdict, report the verdict and *refer* the corrected rule to the next phase.
  Do not repair it retroactively.
* **A pre-registered correction whose trigger did not fire is not applied**, even
  when it is probably right.
* **When a correction would flatter the work, refer it; when it would hurt,
  measure it.**
* **Decompose a gain by context before reporting it.** `EXP_004`'s mean was three
  effects of opposite sign that nearly cancelled.
* **A ratio needs its reference stated; a systems figure needs its seeds.** Quote
  ratios recomputed from `summary.json` and name the denominator in the caption.
* **Where a reasoning trail is unrecoverable, say so in the artifact.** Leave
  unattributable fields null rather than guessing.
* **Prose that describes an artifact must name one that exists.**

## 4. Measurement

* **Measure rather than assume**, and re-measure anything inherited from a
  previous phase or reported by an agent.
* **σ is not a project constant.** Re-measure it on every structurally new arm.
  It is 0.00461 on the baseline, 0.00313 on the two-state neuron, and it varies
  with a *hyperparameter* that changes no parameter and no function.
* **Never change the baseline to make a diagnostic look better.** Document the
  discrepancy and ship the fix as a ranked candidate.
* **Never replace a failed seed to restore n.** Report the reduced n and mark the
  verdict provisional.
* **Chase a NaN to its origin.** "One seed diverged" is a tolerance widened in
  prose. Replay the failing step on *both* dispatch paths — an identical NaN on
  the eager path exonerates a hand-written backward.
* **Never widen a tolerance to make a residual go away**, and never borrow one
  from a check that constrains a different thing.
* **Check a mechanism's direction of causation before reporting it.**
* **Derive before implementing.** `EXP_008`'s two identities turned "add a
  threshold" into "prove the arm adds no functions, then measure what a redundant
  parameter buys" — and removed the need for a new kernel entirely.
* **Run the §7.2 gradient-reachability screen before every new-parameter arm.** It
  costs seconds and it caught an initialisation that was an exact saddle.
* **Never run anything else on the GPU while a timed run is in flight.**

## 5. Testing

* **Verify a test can actually fail before trusting it.** The standard is
  `scripts/audit/08_mutation_campaign.py` (25/25 mutants caught).
* **A numerical contract is only guarded where the quantity it constrains is
  actually observed.** A mutation once escaped a 24-test gate because the
  quantity it corrupted reached no assertion. Assert at the kernel boundary, on
  the intermediate itself.
* **Make every probe reproduce a number it did not produce.** The horizon probe's
  "k = L must equal the committed bpc" check is what caught a run that had
  silently imported a mutated kernel — a contaminated run looks exactly like a
  clean one.
* **A tool that writes wrong code into the source tree must own the machine.**
  Python reads a module once at import, so anything starting mid-campaign keeps
  the mutated version for its whole life.

## 6. Running things

**Long runs die under a 10-minute interactive timeout.** Launch detached:

```bash
python scripts/launch.py -- --corpus enwik8 --max_steps 20000
```

Make drivers restartable: skip runs that are complete (**both** `summary.json` at
full step count *and* `final_test.json`) and **delete** partial ones rather than
resuming, so that no artifact ever describes two runs at once.

---

## 7. Engineering conventions

### Gates

```bash
ruff check .                        # must pass
pytest -m "not cuda and not data"   # 165 tests, ~7s, no GPU or corpus needed
pytest                              # 303 tests, ~42s, needs GPU + corpus
```

CI runs the first two. **CI cannot catch a regression in the fused kernels, the
CUDA-graph capture path, or determinism under replay** — those are `cuda`-marked
and deselected on the runner. Run the full suite on the GPU machine before
closing a phase.

### What is committed and what is not

`data/`, `experiments/runs/`, checkpoints and `*.log` are **not** committed.
Curated evidence extracted from them **is**, under `docs/reports/data/`, so that
every number in a report traces to a tracked file. `experiments/logs/` is tracked
and must stay so.

### Imports

Scripts prepend `src/` to `sys.path` themselves and carry `# noqa: E402` for the
imports that follow. `conftest.py` does the same for tests, and *prepends* so the
working tree always wins over an installed copy. `pip install -e .` is supported
and optional; it does not replace either mechanism.

**Import sorting (`ruff`'s `I` rules) is not enabled and must not be.** It would
hoist imports above the `sys.path.insert` lines they depend on.

### Lint policy, and every exemption in it

The ruff configuration in `pyproject.toml` is **curated, not default**. The broad
default set reports 388 findings here and essentially none are defects.

`src/` is held to the full set and **is clean**. The exemptions are:

| Rule | Where | Why it is exempt |
|---|---|---|
| `B023` | everywhere | Lambdas closing over a loop variable, consumed by `timeit` *inside that same iteration*. The late-binding hazard cannot occur when the closure never outlives the iteration. 37 sites, all benign. |
| `B905` | everywhere | `zip(..., strict=)`. 46 sites, all equal-length by construction. |
| `F821` | `tests/test_graph_equivalence.py`, `scripts/*` | A **false positive**: `del a, b` / `del tr` frees GPU memory after a benchmark, and ruff then treats the name as unbound inside a lambda that was already evaluated. Not a defect — see `scripts/audit/01_gpu_capability.py:49`. |
| `E702` | `tests/*` | `a = 1; b = 2` setup pairs. |
| `E731`, `E741`, `F541`, `B007`, `F841`, `F401` | `scripts/*` | Style only. **These scripts produced numbers committed under `docs/reports/data/`. They are evidence and are not edited for cosmetics.** The exemption records that decision somewhere a machine will check, rather than in a comment nobody runs. |
| `E402` | `conftest.py` | Imports follow the `sys.path` insert, necessarily. |
| `NPY002` | not selected | It would rewrite legacy numpy RNG calls and **change the random stream**, invalidating reproduction of committed numbers. |

`ruff format` is **not** enabled and is not run: it would reformat 60 of 84 files
for zero behavioural gain and bury every future real change in the diff.

If you add a rule to `select`, the tree must be clean under it on the same
commit. A gate with a backlog is a gate nobody reads, which is the same as no
gate at all.

### Versioning

`src/snn/__init__.py` holds `__version__` and `pyproject.toml` reads it — one
source of truth. It must stay **PEP 440-valid** (`2.0.0+phase4`, not
`2.0.0-phase2`) or the package cannot be built. It is deliberately *not* stamped
into any run manifest; run provenance records torch, numpy and snntorch versions
instead.
