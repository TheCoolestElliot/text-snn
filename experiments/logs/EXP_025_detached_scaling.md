# EXP_025 — Width, on the neuron the project actually adopted

**Status: PRE-REGISTERED — no results yet.**

Pre-registered before any driver, resolver or run existed. `## 9. Results` holds
nothing but its placeholder until the ladder is complete.

---

## 0. What this experiment is, and what it is not

`EXP_014` bought **−0.25238 bpc at 54.75 transferred σ** — more than every
architectural arm Phase 4 built. Report §13.6 recorded what it could not do:
**it ran on the plain LIF.** The project's adopted neuron is the two-compartment
one (`EXP_004`), and at the time `EXP_014` ran, that neuron could not be trained
at 5.0M parameters at all — `EXP_015` killed both two-compartment legs,
deterministically, at steps 5138 and ~2000, and that failure became open decision
**#11**.

`EXP_017` removed the blocker. Detaching the fast pole's reset factor makes the
per-step backward Jacobian **diagonal and bounded by `beta_f = 0.5`**, strictly
tighter than the plain LIF's own 0.5123596 at every width; it changes **nothing
in the forward** — the same compiled kernel, and the checkpoint evaluates as
`arch="twocomp"` bit-identically; and the arm then trained **20,000 steps at 5.0M
with zero non-finite losses under the recipe that killed the adopted arm**, with
no hyperparameter moved and 24 frozen fields read back out of each run's own
`config.json` to prove it.

**What `EXP_017` could not do is put a bar on it.** Its §9.3 H2b table is
explicitly *"n = 1 on both sides, no bar, and the report attaches no verdict to
it"*, and its stability finding — 3/3 detached seeds train at `d = 512` while 2/3
freshly trained `twocomp` anchors diverge — resolved **H3 as NOT RUN at Fisher
p = 0.40**.

**This experiment is that bar, and nothing else.**

### What it is not

* **It changes no code.** `arch="twocomp_detach"` has existed since `6107975` and
  `Config.d_model` since Phase 2. `EXP_025` adds **zero lines to `src/`** — the
  second Phase-4 experiment to do so, after `EXP_014`. There is consequently **no
  new parameter, no new arch, no new kernel, no new hand-written backward, no R10
  gradient gate, and no mutation-campaign re-run**. The R10 tax that gates
  `EXP_024` §8 item 1 **is not payable here and is not being paid by the back
  door**.
* **It adds no `Config` field, so it needs no K1 exemption of its own.** The
  standing gotcha — adding any field makes every pre-existing reference run fail
  K1 — is not triggered. §7's K1b still has to exempt the 24 fields added
  *between* the committed reference run and today's tree, and it lists them by
  name rather than exempting a count.
* **The §7.2 gradient-reachability screen does not apply**, on the same basis
  `EXP_014` §0 recorded: no new parameter kind. Both arms' parameters are
  screened already.
* **It does not engage decision #11, and that is deliberate.** #11 asks whether
  the frozen Phase-2 recipe may be re-derived at a new size. **Every run here uses
  it unchanged** — `lr = 3e-3`, cosine over 20,000 steps, `grad_clip = 1.0`,
  `weight_decay = 0.1` — which is precisely the condition under which `EXP_017`'s
  5.0M leg trained. **If a run diverges it is reported as a divergence, the seed
  is not replaced, and no recipe term moves.** Nothing in this file proposes a
  value for any recipe term, and G5 is the mechanism that holds that rather than a
  promise in prose. `EXP_016` §10's referral — that `dv` contains no learning
  rate, no schedule, no weight decay and no gradient clip, so #11's *form* may be
  the wrong question — is why this is worth stating: **this experiment neither
  rules #11 nor needs it ruled.**
* **It adopts nothing and ranks nothing.** Adopting the detached arm is decision
  **#5**'s shape and is Elliot's. Whether `EXP_014`'s ladder should be re-read in
  light of this one is report §7.1's business, which is *offered and not applied*.
* **It authorises no phase.** `EXP_014` and `EXP_015` were both scaling ladders
  that ran inside Phase 4; this runs on that precedent and does not touch decision
  **#4**.

---

## 1. The question

Three. The first is the one `EXP_014` was structurally unable to ask.

**Q1 — does the two-compartment advantage survive width?** The project's central
architectural claim is that its adopted neuron is better than the plain LIF. That
claim has only ever been measured at 735,437 parameters. `EXP_014` then showed
that width buys more than the entire architectural programme did — which raises
the possibility that the project has spent Phase 4 defending an advantage
capacity simply absorbs. **If the advantage shrinks with width, the ranked
candidate list is aimed at the wrong thing.** Nothing in this project has
measured it.

**Q2 — what is σ at width?** Every bar in `EXP_014` was placed against
**σ = 0.00461**, measured at `d = 512, arch = "snn"`, n = 5, and its own §3.0
named the transfer as the design's largest risk: `EXP_013` §9.5 measured σ moving
**0.36× to 3.94×** under a scalar hyperparameter that changed no parameter and no
function, and a 6.8× parameter change is a far larger perturbation than that.
**No experiment in this project has ever measured σ above 735,437 parameters.**
n = 3 per cell produces one, at both rungs, for both arms.

**Q3 — does width buy REACH on this neuron?** `EXP_014` S4 held: width buys
per-step capacity and **essentially no reach**. Report §14.3 located the *entire*
matched-size gap to the GRU anchor in the failure to use context. So if the
two-compartment neuron's advantage is a reach advantage, its decomposition at
width should not look like the plain LIF's. **If it looks the same, that is the
more important answer**, and it is the one this file expects (H5).

---

## 2. Design

### 2.1 The ladder

**Two rungs × two arms × three seeds = 12 runs.**

| | `d_model` | parameters, `snn` | parameters, `twocomp_detach` |
|---|---:|---:|---:|
| low rung | 512 | 735,437 | 737,485 |
| high rung | 1481 | 4,997,099 | 5,003,023 |

The widths are **`EXP_014`'s own end points and are not newly chosen**: 512 is the
width every committed figure in this project is measured at, and 1481 is the
exact-error-minimising integer for 5M parameters that `EXP_014` §2.2 fixed and
`EXP_015` and `EXP_017` both reused. The parameter counts are read back and
asserted (G3), not assumed.

**Why there is no middle rung.** `d = 1020` would add ~1.0 GPU-h (§2.3) to buy a
monotonicity check `EXP_014` already ran on the plain LIF, and three points at
n = 3 cannot resolve curvature anyway. The two ends buy the difference-of-
differences that Q1 *is*. **The cost of that choice is stated in §5 rather than
left implicit: no per-octave slope may be quoted from this experiment at all.**

### 2.2 The single variable, and it is unusually single

At a fixed `d_model` and seed, `Config(arch="snn")` and
`Config(arch="twocomp_detach")` differ in **exactly one field — `arch`** — out of
58. This was verified before this file was committed and is re-asserted per run by
K1a. No Phase-4 experiment has had a cleaner single-variable claim: the two arms
share every recipe term, every initialisation constant and every evaluation
setting by construction rather than by care.

`MAY_DIFFER = {arch, d_model, seed, run_name}`. Everything else aborts the ladder.

### 2.3 Cost, measured before this file was committed

`scripts/exp/014_calibrate_cost.py`, 400 steps per cell, six cells, denominator
`snn_beta0.5_s0` — committed as `docs/reports/data/exp_025_cost_calibration.json`.
**This is a measurement and not an estimate**, because report §7.1's one
previously-checked estimate was low by **3.1×** and the constant its column
descended from was formally withdrawn.

| arch | `d` | s/step | 20,000 steps | peak VRAM | clip fires |
|---|---:|---:|---:|---:|---:|
| `snn` | 512 | 0.01904 | 380.8 s | 0.6089 GiB | 0/17 |
| `snn` | 1481 | 0.10698 | 2139.7 s | 1.6434 GiB | 3/17 |
| `twocomp_detach` | 512 | 0.02489 | 497.8 s | 0.9801 GiB | 2/17 |
| `twocomp_detach` | 1481 | 0.12626 | 2525.3 s | 2.6138 GiB | 8/17 |

The committed `snn_beta0.5_s0` ran 20,000 steps in **398.88 s** against this
calibration's 380.8 s projection for the same configuration, so a **realisation
factor of 1.0475** is applied to every projection below rather than quoting the
400-step figure as if it were the run.

| rung | runs | projected |
|---|---:|---:|
| `d = 512` (both arms, 3 seeds) | 6 | **0.77 GPU-h** |
| `d = 1481` (both arms, 3 seeds) | 6 | **4.07 GPU-h** |
| **ladder total** | **12** | **4.84 GPU-h** |
| evaluations, horizon sweep, gap probes | — | ~0.25 GPU-h |
| **total** | | **~5.1 GPU-h** |

**This is ~81 % of the ~6.3 GPU-h remaining in Phase 4's ~30-hour budget, and that
is stated here rather than discovered afterwards.** The mitigation is structural,
not optimistic: **the rungs run low-first**, the driver is restartable, and §8
item 6 makes the high rung a separate authorisation. If the `d = 512` rung comes
back wrong, 0.77 GPU-h has been spent and not 5.1.

**VRAM.** The widest cell peaks at 2.6138 GiB on a 7.96 GiB card. G4 alarms at 3.0
GiB — an alarm and not a budget — because a spill under Windows' WDDM is a ~50×
slowdown that does not raise, which would corrupt a wall-clock figure while
looking merely slow.

**The clip binds at the high rung on both arms and binds harder on the detached
one** (8/17 against 3/17 at 400 steps). That is not a confound this experiment
introduces — `EXP_014` §2.1 measured the same thing — but `EXP_014` §9.12
established that `_clip_grad_norm_fp64` changes the trajectory wherever it fires,
so it is carried as **D1** and it constrains how H1 may be read.

---

## 3. Hypotheses

### 3.0 What n = 3 per cell can and cannot resolve, stated before the run

`CONTRIBUTING.md` §2 requires a prediction's power to be stated in advance.

**σ used for sizing.** The only σ measured on this exact arm is from `EXP_017`'s
three committed `detach_d512` seeds: **0.002363** (mean 2.115325). The plain LIF's
committed σ at the same width is **0.00461**, n = 5. Sizing below uses the
**larger** of the two, 0.00461, so every bar quoted here is the conservative one.

| quantity | se at n = 3 | 2σ bar |
|---|---:|---:|
| one arm's mean at one rung | 0.00266 | — |
| difference between arms at one rung | 0.00376 | **0.00753** |
| difference-of-differences across rungs | 0.00532 | **0.01065** |

**Bars are scored against σ measured in this experiment, not against these
numbers.** The pre-registered rule is fixed now: **each bar is a Welch two-sample
test on that rung's 3 + 3 seeds**, and the sizing table exists so a reader can see
what the design was expected to resolve. **A verdict that flips depending on
whether measured or transferred σ is used is reported as UNRESOLVED**, and both
are published. That guard descends directly from `EXP_014`'s transferred-σ caveat
and it is in the bar, not in the write-up.

**A miss resolves UNRESOLVED, never FAILED**, except where a hypothesis is
explicitly two-sided (H2). **The remedy for any unresolved cell is seeds *added*,
never substituted.**

**Divergence.** A run that goes non-finite is **not replaced** and its cell drops
to the reduced n, which is reported everywhere the verdict appears. If any cell
falls below n = 3 the affected verdict is marked **provisional**, per
`CONTRIBUTING.md` §4 and `EXP_011`'s standing example.

---

### H1 — the advantage exists at 5.0M, with seeds *(primary; a sanity bar, and said so in advance)*

> Mean carried test bpc of `twocomp_detach` at `d = 1481` is **below** `snn`'s at
> the same width, and the difference resolves at ≥ 2σ on a Welch test over the
> 3 + 3 seeds.

**Power, stated honestly: this is the hypothesis most likely to hold and least
likely to be informative.** `EXP_017`'s committed n = 1 reading is **−0.13783**,
which is **18×** the 2σ bar of 0.00753. H1 exists so that the headline number this
experiment reports **carries a verdict** instead of being quoted from a table its
own author labelled *"no bar"*. **If H1 fails, the ladder is broken and §6's
aborts should have fired first** — a failure here is a systems finding, not a
scientific one, and §4 routes it that way.

**Guards.** (a) H1 may not be reported as held unless **D1** is published beside
it — if the clip binds hard and asymmetrically at that rung, *"the neuron helped"*
and *"the recipe throttled one arm less"* are not separated by this design. (b) H1
may not be reported as held unless **H5**'s per-context decomposition is published
beside it: report §6.2 shows arms regressing 126–128 of 128 contexts while moving
a mean.

### H2 — the advantage is FLAT in width *(the informative one)*

> `|Δ(d = 1481) − Δ(d = 512)| < 2σ_DoD`, where `Δ(d)` is the detached arm's mean
> carried test bpc minus the plain LIF's at that width. The arm's advantage
> **neither grows nor erodes** across 2.76 octaves of parameters.

**Registered as an equivalence claim, and scored as one.** The committed n = 1
readings are **−0.13778** at `d = 512` (against the five-seed `snn` mean) and
**−0.13783** at `d = 1481` — **5e-05 bpc apart, against a resolution of 0.01065**.
So the prediction is FLAT, and this file says so before the run rather than
discovering flatness afterwards and calling it a result.

**An equivalence claim that "passes" by failing to reject is worthless**, which is
why H2 is scored by **two one-sided tests against a pre-declared equivalence
margin of 2σ_DoD** and not by a non-significant difference:

| verdict | condition |
|---|---|
| **FLAT** | both one-sided tests reject — the DoD interval lies inside ±2σ_DoD |
| **GROWS** / **ERODES** | the DoD resolves at ≥ 2σ in one direction |
| **UNRESOLVED** | neither — the confidence interval straddles the margin |

**UNRESOLVED is the honest modal outcome at n = 3 and is not a disappointment** —
and that sentence is scoped to *this design at this n*, deliberately, because
report §8's rev-15 note records what happens when such a sentence is generalised
past the evidence supporting it.

**Guard.** There is no middle rung, so **no per-octave slope may be quoted from
H2**, and "flat" means flat *across this interval*, never linear within it. This
restriction is restated wherever H2 is quoted.

### H3 — stability, at an n `EXP_017` could not reach

> **6 of 6** `twocomp_detach` runs complete 20,000 steps with zero non-finite
> losses.

Reported as a rate with its interval and its denominator — `6/6, 95 % CI [0.61,
1.00]` — per `CONTRIBUTING.md` §3, and **scored on this experiment's own runs
alone**.

**The comparison against `twocomp` is secondary and is marked cross-tree.** The
committed contrast is 1/3 trained at `d = 512` (`EXP_017` §9.4) and 0/1 at
`d = 1481` (`EXP_015`); Fisher's exact on 6/6 against 1/4 gives **p = 0.0048**,
which would establish for the first time the difference `EXP_017` could not
(p = 0.40). **But those runs are a different experiment on a different tree, and
decision #10's ruling is that new experiments train their own anchor** — so that
p-value is reported as a **secondary, cross-tree comparison explicitly labelled as
such**, and H3's own verdict does not depend on it.

**No `twocomp` control is trained here.** It would cost between 0.58 and 2.24
GPU-h depending on whether the runs die, and buying a same-tree denominator for a
secondary comparison is not worth that share of the remaining budget. **That is a
budget decision and it is recorded as one**, not dressed as a design choice.

### H4 — σ at width, measured for the first time in this project

> The pooled within-arm sd at `d = 1481` differs from the transferred **0.00461**
> by more than a factor of **1.5** in either direction.

**Whatever the verdict, the deliverable is the number**, not the bar: every
`EXP_014` bar that used the transferred value is re-reported against the measured
one in §9. `EXP_013` §9.5's 0.36×–3.94× range is why 1.5× is the threshold and why
either outcome is publishable.

**This hypothesis cannot fail to produce its measurement**, which is stated so a
HELD/UNRESOLVED verdict on it is not read as the point of it.

### H5 — where the bits are, decomposed by context

`EXP_004` §10.3's three-way cut — zero-context / within-reach / beyond-horizon,
positive = better — applied to `twocomp_detach`'s gain over `snn` at `d = 1481`.

> The gain is **not** predominantly a beyond-horizon gain: the zero-context and
> within-reach components together carry **≥ 50 %** of it.

**No bar is placed on the beyond-horizon component**, and the reason is recent and
specific: report §8's rev-15 note established that this component has read outside
±0.0016 on **7 of 24** committed primary arm-gain readings, three of them clearing
a 2σ bar, and that its sign carries a compression artifact on losing arms. It is
reported as a **marker with the integer horizon beside it**, and the note's own
conclusion is inherited verbatim: **two markers agreeing is two markers
agreeing.**

### D1 — diagnostic, not a bar: does the clip bind, and asymmetrically?

Per-run `n_over_clip`, `frac_over_clip`, `max_grad_norm`, and the count of
non-finite loss records. **Not scored.** §2.3's calibration says the clip fires
2.7× more often on the detached arm at the high rung, and `EXP_014` §9.12
established the clip changes the trajectory wherever it fires. **H1's guard (a) is
enforced from this row.**

---

## 4. Decision rule, fixed now

1. **H1 held + H2 FLAT** — the two-compartment advantage is real at 5.0M and is a
   constant offset that width neither amplifies nor absorbs. **This does not adopt
   the arm** (decision #5) and does not rank anything; it is handed to report §7.1
   as evidence about what the ranked list is aimed at.
2. **H1 held + H2 ERODES** — capacity absorbs the architecture. This is the result
   that would most change the project's direction, and the one this file is least
   expecting; it would be reported first and most prominently.
3. **H1 held + H2 GROWS** — the strongest possible case for the arm, and the one
   most likely to be a seed artifact at n = 3. It would be reported as
   **provisional pending seeds added**, not as a headline.
4. **H1 UNRESOLVED** — a systems finding. §6's aborts are re-read, the ladder is
   audited before any scientific reading is offered, and no bpc from it is quoted
   as a figure.
5. **Any cell below n = 3** — every verdict touching it is **provisional**, the
   reduced n is cited wherever it appears, and the seed is not replaced.

**This experiment rules no open decision, adopts no arm, ranks no candidate,
changes no hyperparameter, authorises no phase, and proposes no value for any
recipe term.** Anything it implies about #5, #4 or #11 is referred in §10 and is
Elliot's.

---

## 5. Known limitations, stated in advance

1. **Two rungs cannot see a curve.** No per-octave slope, no curvature, no
   extrapolation beyond 5.0M. §2.1 records the budget reason.
2. **n = 3 is the floor of §6.2's adoption rule, not a comfortable n.** The DoD
   bar is 0.01065 bpc — larger than every arm effect Phase 4 has measured except
   width itself. **H2 is therefore powered to detect only a large change in the
   advantage**, and a FLAT verdict means "no change larger than ~0.01 bpc", not
   "no change".
3. **The GRU anchor is not retrained here.** Any gap-to-anchor figure is quoted
   from `EXP_015`'s committed n = 1 run at `d = 1481` and is labelled cross-tree
   and n = 1 wherever it appears. It is context, never a bar.
4. **The clip binds asymmetrically at the high rung** (§2.3, D1) and this design
   does not separate a capacity effect from a differential throttling effect.
5. **`twocomp` itself is absent** (H3), so "the detached arm is more stable" rests
   on a cross-tree denominator and is reported as such.
6. **Carried is the primary protocol**; fresh is reported for both arms at both
   rungs, always, per `src/snn/evaluate.py`'s standing rule that reporting one
   flatters the model in one direction.
7. **This measures `twocomp_detach`, not the adopted arm.** `EXP_017` §9.2's A6
   row is the caveat that matters and it is inherited here in full: the detached
   rule **trains further inside the dangerous region of weight space**, because
   its gradient is indifferent to `w·vs`. A stability result here is a statement
   about the detached rule, **not** a reassurance about `twocomp`.

---

## 6. Self-checks — aborting, not advisory

* **A1.** Any run's realised parameter count differs from the pre-registered one →
  abort (G3).
* **A2.** Any run's `config.json` differs outside `MAY_DIFFER` → abort (K1a).
* **A3.** Peak VRAM > 3.0 GiB on any run → abort (G4).
* **A4.** Realised wall-clock differs from §2.3's projection by > 1.5× → **halt and
  report**, do not silently continue: it means the calibration does not describe
  the ladder and every budget figure here is wrong.
* **A5.** The mutation-campaign lockfile is present → abort before every child,
  not once at the top.
* **A6.** `ruff check .` or the CPU test subset fails on the tree the ladder runs
  from → abort before the first run.

---

## 7. Gates

* **K1a — config sameness, internal, zero exemptions.** Every one of the 12 runs is
  diffed field-by-field against **this experiment's own** `e25_snn_d512_s0`. Only
  `MAY_DIFFER` may differ. All 12 runs are produced by one tree, so there is **no
  new-field exemption anywhere in K1a** — the strongest form of the gate the
  project has run.
* **K1b — config sameness against the committed reference**, `snn_beta0.5_s0`.
  **24 fields have been added to `Config` since that run** and every one is at its
  default in every run here. They are exempted **by name, listed in the manifest**,
  not by count: `beta_slow`, `w_init`, `mu_init`, `thr_log_init`, `noise_amp`,
  `noise_p`, `da_mode`, `da_source`, `da_scale`, `da_gain_init`, `iface_fold`,
  `iface_binary_input`, `iface_read_layers`, `iface_read_lags`, `iface_thr_in`,
  `iface_code_sd`, `nm_source`, `nm_pos_len`, `nm_tau`, `nm_gain_init`,
  `nm_a_init`, `nm_b_init`, `tf_kappa`, `tf_rolled`.
* **G3 — parameter count asserted**, per run, against §2.1's table. The parameter
  count is this experiment's x-axis and an off-by-one width would move it
  silently.
* **G4 — VRAM alarm** at 3.0 GiB (§2.3).
* **K3 — the cost model is checked, not assumed.** Realised wall-clock per run
  against §2.3's projection, reported as a ratio with the calibration named as the
  denominator.
* **G5 — the recipe is verified frozen**, not asserted: the recipe fields are read
  back out of every run's own `config.json` and compared to `02a_phase2_spec.md`'s
  frozen values. **This is the gate that keeps decision #11 unengaged** and it is
  the one whose failure would matter most.

---

## 8. Entry conditions

1. This file committed, before the driver and before the resolver.
2. `scripts/exp/025_run_detached_scaling.py` committed, restartable, deleting
   partials rather than resuming.
3. `scripts/exp/025_detached_scaling_results.py` — the resolver — **committed
   before any run's bpc is read.** `EXP_020` and `EXP_021` did not meet this
   condition and report §20.7 records that; `EXP_022` and `EXP_023` did.
4. Pre-registration SHA-256 stamped into `exp_025_run_manifest.json` before the
   first run and re-checked after the last. Because this file is **committed**
   before the run, `scripts/exp/verify_prereg_hash.py`'s commit route can verify it
   against an object in the repository rather than a guess about layout.
5. Cost pre-flight complete — `exp_025_cost_calibration.json`, §2.3. **MET.**
6. **The `d = 1481` rung is a separate authorisation.** The driver runs the
   `d = 512` rung first and stops. The high rung costs 4.07 GPU-h — 65 % of the
   phase's remaining budget — and it is not started until the low rung's six runs
   are complete and their gates green.
7. Nothing else runs on the GPU while a timed run is in flight.

---

## 9. Results

*(placeholder — no results yet)*
