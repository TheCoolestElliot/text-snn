# EXP_023 — EXP_021's winner beat its anchor by 0.034. What part of the signal did that?

**Pre-registered:** 2026-08-18, after the arithmetic in §2, the §7.2
reachability screen in §2.5 and the cost pre-flight in §2.6, and **before any
arm in §3 was trained, any checkpoint was written, and any bpc existed for any
arm in §3.** **Committed before the first run of §3.**

**Status: CLOSED 2026-08-20. H6, H1 and H5 held; H3 FAILED; H2 and H4 UNRESOLVED; H7 flat at 1.1176. The headline replicates out of sample at -0.02535 — and the rung that knows NOTHING is worth -0.02975 against the rung that knows this sequence's own prediction error at -0.02953. The optimiser buys the form, not the signal.**

---

## 0. What this experiment is, and what it is not

`EXP_021` produced the first arm in this session to beat its anchor:
`nm_local` at **−0.03371 bpc, paired t = −8.95, 3/3 seeds, 3.7× the bar**, in
the slot decision #3 explicitly **admits** as adoptable. And it produced two
numbers that say the mechanism in the arm's name is not established:

* **H2 UNRESOLVED.** The batch-rolled control `nm_rolled` gains −0.0279 on its
  own; the alignment adds only −0.00577, below the 0.00922 bar.
* **H3 at 1.12.** `rms(nm_gain)` is 1.077 aligned against 0.958 rolled —
  against `EXP_018`'s **26** on the identical quantity. **The optimiser grows
  `kappa` ~10× from its init for either signal and cannot tell them apart.**

`EXP_021` §9.2 therefore wrote, from a cell §5 fixed before the run,
*"a bounded time-varying gain pays and the RPE is not why"*, and §11 left two
questions with prices on them. This experiment is those two questions, plus the
rung that decides whether the headline survives at all.

**THE RUNG THAT COULD DEFLATE THE HEADLINE.** `cur * (1 + kappa_c · DA_t)` with
`DA` held **constant** is `cur * (1 + kappa_c)` — a learned per-channel gain on
the input current, which by `EXP_008` Identity 1 is **exactly a learned
per-channel threshold**. That is adopted arm #5, and `EXP_008` measured it at
**0.0590 bpc** — *larger* than `EXP_021`'s headline. `if_anchor` is
`arch="snn"`, which does not carry it. **So until the constant rung is run, this
project cannot say whether its feedforward modulator won by being a modulator or
by being a gain it already knew about.** Nothing in `EXP_021` is retracted; this
is the control it did not have.

**What this is not:**

* **It adopts nothing and ranks nothing.** Adoption is Elliot's (report §8), and
  no arm here may be ranked in §7.1.
* **It changes no hyperparameter.** The Phase-2 recipe is frozen and G5 reads it
  back out of each run's own `config.json`. `nm_tau = 0.057321` is `EXP_021`'s
  calibrated constant, **reused unchanged and not re-fitted**.
* **It does not decide #11 and does not touch #6.** `EXP_008`'s gain may not
  enter a headline until #6 is ruled; §5 reports `nm_const` as a **rung of this
  ladder**, never as a re-measurement of arm #5, and §6 says why that
  distinction is real and where it is thin.
* **It does not edit any adopted arm.** G1 by `git diff`. **No new scan, no new
  recurrence, no hand-written backward.**
* **It violates no invariant.** `nm_a`, `nm_b`, `nm_gain`, `nm_pos` are all
  O(1) per neuron or O(1) per position; no `[d, d]` lateral matrix anywhere, so
  I5 holds in the form the 2026-08-01 ruling admits.
* **It does not re-baseline.** Decision #10. `if_anchor` seeds 3–5 are **fresh
  seeds of the same configuration on the same tree**, added under this
  pre-registration.
* **Adding seeds is legal here and would not have been there.**
  `CONTRIBUTING.md` §4 forbids adding seeds to a cell after seeing its result
  **within one experiment**, which is exactly why `EXP_021` §11 item 2 priced
  this at ~0.6 GPU-h and did not run it. §4 below pre-registers **both** the
  fresh-seed-only replication and the pooled n = 6 estimate, so the powered
  number and the out-of-sample one are separable.

---

## 1. The question

> **A rank-1 time-varying multiplicative gain on the input current is worth
> ~0.03 bpc. How much of that survives when the signal driving it is replaced by
> something that knows less?**

Four rungs of **one fixed algebraic form**, in order of decreasing signal
content. Only what drives `DA` changes:

    cur^k  =  cur^k * (1 + kappa_k · DA^k_t)              IDENTICAL at every rung

| rung | `DA_t` | knows |
|---|---|---|
| `const` | `1` | nothing. A static per-channel gain = `EXP_008` Identity 1. |
| `pos` | `tanh(w_t)`, learned per window position | **when**, and nothing else. |
| `rolled` | another sequence's aligned RPE | when, and what surprise looks like in general |
| `local` | this sequence's own RPE | when, and what surprised **this** sequence |

**The rung that gives the ladder its point.** `nm_rolled` rolls across the
**batch**, which preserves the time index exactly — so a misaligned RPE still
carries *"how far into the window am I"*, and within-window position is the
strongest single predictor of surprise there is (early positions have less
context). **`pos` is that leak, isolated.** If `pos` recovers most of the gain,
then `EXP_021`'s H2 was small because both of its arms were riding a positional
signal, and the answer to "is it the RPE?" is no for a reason neither arm could
show.

---

## 2. The derivation, done before any measurement

### 2.1 The four rungs are one code path with one branch

`snn.model.LocalDopamineCharLM._driving_da` resolves the rung on a string fixed
at construction, so each is a **distinct code path** and none rests on a float
identity — the same construction `nm_source="off"` uses for its bitwise nesting.
`const` builds a real `[B, L]` ones tensor rather than the Python float `1.0`,
**deliberately**, so the multiply, the broadcast and the kernel count are
identical to the other rungs' and §2.6's cost comparison is not measuring a
removed kernel.

### 2.2 Each rung allocates only what it reads

An unused parameter is still counted by `count_params` and still decayed by
AdamW, so a rung that carried a predictor it never reads would be described by a
closed form that is not its own — the failure `local_dopamine_param_count`'s own
docstring records being caught at, one digit, before `EXP_021`'s G2 aborted on
it. Per modulated layer (there are `K − 1 = 1`):

| rung | per-layer parameters | realised at d = 512 |
|---|---|---:|
| `const` | `d` — `kappa` alone | **735,949** |
| `pos` | `d + 256` — `kappa` and the positional table | **736,205** |
| `rolled` / `local` | `3d + 1` — `a`, `b`, `kappa`, `log_tau` | **736,974** |
| *(`off` keeps `3d + 1` on purpose — its nesting claim is about the code path, and carrying the parameters is what lets G2 assert they reach nothing)* | | 736,974 |

All four are within **0.14 %** of each other and within **0.21 %** of
`if_anchor`'s 735,437, so **no rung on this ladder is a width arm in disguise**.
Asserted, not assumed, by `tests/test_neuromod.py::
test_g4_every_rung_realises_its_own_closed_form_exactly`.

### 2.3 `pos` is diagnostic-only, and that is the point rather than a caveat

Window position is a **training artefact**: under the carried protocol the state
crosses window boundaries, so position `t` means nothing about how much context
the model actually has, and at decode (`L = 1`) it means nothing at all. `pos`
is therefore **structurally non-deployable** and is labelled so wherever it
appears. That is exactly what makes it the control: an arm that cannot be
shipped, recovering a gain an arm that can be shipped also recovers, is
evidence about the shipped arm.

The forward **raises** rather than wrapping when the window outruns the table
(`nm_pos_len < L`), because a silently tiled positional gain is a different arm
that trains and scores plausibly — the same defect class as `EXP_020`'s
`iface_read_lags > 1` decoding all-zero blocks at `L = 1`.

### 2.4 `pos` has a two-step sequential unlock, and it is not `EXP_004`'s saddle

At `w = 0`, `DA ≡ 0`, so `dL/dkappa_c = Σ (dL/dcur') · cur · DA` is **exactly
zero**. That is the mirror image of `nm_g0`, where `kappa` was the reachable
parameter and the predictor was not. But

    dL/dw_t = Σ_c kappa_c · (dL/dcur'_{t,c}) · cur_{t,c} · sech²(0)

is **not** zero at `kappa = 0.1`, so `w` moves at step 1 and `kappa` becomes
reachable at step 2 — **two steps out of 20,000**, against `EXP_004` §2.2's
saddle, which had no reachable parameter at all.

**Measured, not argued**, twice, before this file was committed:

* `tests/test_neuromod.py::test_g4_the_pos_rung_unlocks_kappa_in_two_steps_rather_than_never`
  asserts `dL/dkappa = 0.0` exactly at step 0 and `> 0` at step 1.
* A real 400-step GPU run at the arm's own recipe:
  `rms(nm_gain) = 0.2559` (from a 0.1 init) and `rms(nm_pos) = 0.2841` (from
  0). **Both parameters train.**

`nm_gain_init = 0.1` is `EXP_004`'s pre-registered fallback, reused unchanged —
the same value `EXP_021` ran every `nm_*` arm at, so the ladder's rungs are
init-matched to each other and to `EXP_021`'s.

### 2.5 The §7.2 reachability screen — run before this file was committed

`docs/reports/data/exp_022_023_reachability.json`, through the **real arch**:

| candidate | init | verdict | derived_zero | measured zero |
|---|---|---|---|---|
| `nm_const` | `kappa = 0.1`, `DA = 1` | **PASS** | — | — (agrees) |
| `nm_pos` | `kappa = 0.1`, `w = 0` | **SADDLE** | `("nm_gain",)` | `nm_gain` (agrees) |

`nm_pos` reads SADDLE and **that is the derivation's own prediction, declared in
`derived_zero` before the screen ran.** It is a one-sided, self-resolving
saddle: §2.4's measurement is the evidence that it opens, and G8 below is
written to accept it on exactly that evidence and on nothing else. The screen's
controls held and it reproduced `exp_004_gradient_reachability.json` at worst
relative residual 0.00e+00.

`nm_const` has **no saddle to fall back from** — unlike the RPE rungs there is
no second parameter whose only path to the loss runs through `kappa`.

### 2.6 The cost pre-flight — run before this file was committed

`docs/reports/data/exp_022_023_cost_preflight.json`, 400 steps per row,
denominator `snn_beta0.5_s0` at 398.88 s / 20,000 steps.

| row | steps/s | ratio | projected 20k | peak VRAM |
|---|---:|---:|---:|---:|
| `snn` d = 512 *(base)* | 54.63 | 1.0000 | 398.9 s | 0.609 GiB |
| `nm_const` | 47.54 | **1.1491** | 458.4 s | 0.734 GiB |
| `nm_pos` | 46.51 | **1.1747** | 468.6 s | 0.734 GiB |
| `nm_local` | 36.43 | **1.4995** | 598.1 s | 0.984 GiB |

**`nm_local` re-measures at 1.4995 against `EXP_021` §9.6's realised 1.49×** —
an independent cross-check of that experiment's cost claim on a different
instrument, and it holds to three digits.

The two new rungs cost **1.15–1.17×**, i.e. the modulation's broadcast multiply
and its backward account for ~0.15 of `nm_local`'s 0.50 and the Bernoulli
predictor accounts for the other ~0.35. That decomposition is a by-product and
is reported, not adopted.

---

## 3. The legs

`nm_tau = 0.057321`, `nm_gain_init = 0.1`, `nm_pos_len = 256` throughout.
`nm_a_init`, `nm_b_init` at `EXP_021`'s values wherever a predictor exists.

| leg | run names | arch | `nm_source` | realised params | reference for K1 |
|---|---|---|---|---:|---|
| **1** | `if_anchor_s{3,4,5}` | `snn` | — | 735,437 | `if_anchor_s0` |
| **2** | `nm_const_s{0,1,2}` | `localdopamine` | `const` | 735,949 | `nm_local_s0` |
| **3** | `nm_pos_s{0,1,2}` | `localdopamine` | `pos` | 736,205 | `nm_local_s0` |
| **4** | `nm_local_s{3,4,5}` | `localdopamine` | `local` | 736,974 | `nm_local_s0` |
| **5** | `nm_rolled_s{3,4,5}` | `localdopamine` | `rolled` | 736,974 | `nm_rolled_s0` |

Legs 4 and 5 are **fresh seeds of `EXP_021`'s own cells**, at the same recipe on
the same tree, and legs 1, 4, 5 exist to take three cells from n = 3 to n = 6.

**Order is load-bearing:** leg 1 runs first because legs 2 and 3 are scored
against the pooled anchor, and a reference must exist before the run diffed
against it.

---

## 4. Pre-registered predictions

σ transferred: **2σ = 0.00922**, as `EXP_020` and `EXP_021` used it.
`EXP_020` measured 0.004683 on this tree at n = 3; **§9 reports the n = 6
on-tree σ, which this project has never had**, and reports it as a marker, not
as a re-derivation of the bar.

Comparisons are **paired by seed** where both cells hold the same seeds.
Where n differs (`const`/`pos` at n = 3 against a pooled n = 6 anchor), the test
is **Welch two-sample**, stated here so §9 cannot choose it later.

| # | claim | cells | bar | direction predicted |
|---|---|---|---|---|
| **H1** | **the constant rung against the anchor** | `nm_const` (3) vs `if_anchor` (6), Welch | ≥ 2σ | **const BETTER.** `EXP_008` measured this mechanism at 0.0590 bpc on a different tree; it should not be a null here. |
| **H2** | **does time-variation buy anything at all?** | `nm_local` (6) vs `nm_const` (3), Welch | ≥ 2σ | **local BETTER.** If it is not, `EXP_021`'s headline is `EXP_008` under a new name and §5 says so. |
| **H3** | **is the time-variation positional?** | `nm_pos` (3) vs `nm_const` (3), paired | ≥ 2σ | **pos BETTER.** §1's leak argument: window position predicts surprise, and no rung below `pos` can use it. |
| **H4** | **the measurement `EXP_021` §11 item 2 priced** | `nm_local` (6) vs `nm_rolled` (6), paired | ≥ 2σ | **local BETTER.** `EXP_021` measured −0.00577 with paired se 0.0018; at n = 6 that lands ~2 se from the bar, which is exactly why 6 was the number priced. |
| **H5** | **does data dependence beat position?** | `nm_local` (6) vs `nm_pos` (3), Welch | ≥ 2σ | **local BETTER.** This is H4's question asked against a control that cannot leak content at all. |
| **H6** | **out-of-sample replication of `EXP_021`'s headline** | `nm_local` s3–5 vs `if_anchor` s3–5, paired, **fresh seeds only** | ≥ 2σ | **local BETTER by ≈ 0.034.** Reported separately from the pooled estimate so a replication and a power increase are never the same number. |
| **H7** | **findability across the ladder** | `rms(nm_gain)` per rung | none; reported | `EXP_021` got 1.077 / 0.958 — ratio 1.12 against `EXP_018`'s 26. **If all four rungs land within ~1.2× of each other, the optimiser is buying the FORM and not the signal**, and that is the ladder's headline whichever way the bpc goes. |
| **H8** | decomposition by context, every rung | `EXP_004` §10.3 split | none; reported | `EXP_021` found `nm_local`'s gain **entirely within reach** (+0.0559) and **nothing beyond the horizon** (−0.0005). A rung that knows less should keep the within-reach shape and lose the rest. |
| **H9** | realised wall-clock | none; reported | §2.6: 1.15 / 1.17 / 1.50. |

**H7 IS THE ONE TO READ FIRST.** A bpc ladder that decreases monotonically with
signal content and a `rms(kappa)` that is flat across it are the same finding
stated twice, and it is the finding `EXP_021` §9.2 could not reach with two
points.

**`n_contexts_significantly_worse` is not a bar anywhere here** — `EXP_020`
§9.5's bitwise-copy control read 120/128 on it. Reported as a marker only.

---

## 5. Decision rule, fixed now

| outcome | what is written in §9 |
|---|---|
| H1 holds and H2 fails | **"EXP_021's arm is a learned per-channel gain, and the time-variation is free"** — the headline is restated, `EXP_021` is not retracted, and #6 becomes load-bearing for both |
| H1 and H2 hold, H3 holds, H5 fails | **"the gain is time-varying and positional, and the data dependence adds nothing"** — `EXP_021`'s H2 was small because both arms rode the position leak |
| H1, H2, H5 hold and H4 holds | **"the aligned prediction error is worth something, and here is how much"** — the first positive statement about alignment this project has, reported with its size |
| H1, H2, H5 hold and H4 fails | **"data dependence pays and ALIGNMENT still does not"** — the two controls disagree and §9 says which |
| H6 fails | **"EXP_021's headline does not replicate out of sample"** — reported at the top of §9, before every other bar, whatever the pooled number says |
| H7 flat across all four rungs | **"the optimiser buys the form, not the signal"** — recorded whatever the bpc ladder does |

Every cell above is written before the first run and none is softened afterwards.

---

## 6. Known limitations, stated in advance

1. **`nm_const` is not a clean re-measurement of adopted arm #5, and must not be
   read as one.** `EXP_008`'s arm is a learned per-channel *threshold* on the
   neuron; this is a learned per-channel gain on the *current*, which Identity 1
   makes equivalent **at fixed decay** and which this arm additionally carries at
   only `K − 1 = 1` of the 2 layers (layer 0 is unmodulated by construction). So
   `nm_const` is a **lower bound** on what arm #5 would give here. That
   asymmetry is stated now because it cuts against §0's own argument: if
   `nm_const` gains, arm #5 would gain **at least** as much, and H2 becomes
   harder to pass rather than easier.
2. **#6 is open and `EXP_008`'s gain may not enter a headline until it is
   ruled.** §5's cells are written to report rungs of a ladder, and §9 will not
   quote `EXP_008`'s 0.0590 as a result.
3. **Pooling n = 3 + n = 3 across two pre-registrations.** The seeds are fresh
   and the recipe, tree and instrument are identical, but `EXP_021`'s three were
   observed before this file was written. **H6 exists so that the out-of-sample
   replication is reported separately and first**, and the pooled figure is
   never the only one quoted.
4. **`pos` is non-deployable** (§2.3) and is labelled so in every table.
5. **n = 3 for `const` and `pos`.** Their comparisons are Welch against a
   pooled n = 6 anchor and are less powered than the paired ones. Effects
   between 1σ and 2σ read UNRESOLVED.
6. **Layer 0 is unmodulated at every rung**, so the whole ladder measures a
   modulator on 1 of 2 layers. `EXP_021` §6 carries the same limitation and this
   experiment does not lift it.
7. **`nm_tau` is `EXP_021`'s calibrated constant, reused.** `const` and `pos` do
   not use it at all (no squash of an RPE), which is one more respect in which
   the rungs are not a single-variable ladder — they are a single-*form* ladder,
   and §1's table says exactly what varies.
8. **Everything is at `d = 512, K = 2`.** `EXP_015` is the standing reason a
   735K result need not survive to 5.0M.

---

## 7. Gates — aborting, not advisory

| gate | what it asserts | on failure |
|---|---|---|
| **G1** | adopted arms untouched, by `git diff --stat` | abort the ladder |
| **G2** | realised parameter count **equals** the rung's own closed form, exactly, for all 15 runs | abort that run |
| **G3 / K1** | one field changed per arm against its named reference in §3 | abort the ladder |
| **G4** | `nm_source="off"` nests Phase 2 bitwise, forward and backward | asserted in `tests/test_neuromod.py` |
| **G5** | the frozen recipe, and `nm_tau = 0.057321` to 5 significant figures, read back out of each run's own `config.json` | abort that run |
| **G6** | peak VRAM alarm (pre-flight max 0.984 GiB) | report, do not abort |
| **G7** | divergence read from `log.jsonl`, **not** the exit code | record, evaluate anyway, continue |
| **G8** | §7.2 screen run **before** the first step. `nm_pos`'s SADDLE is accepted **only** on §2.4's two-step-unlock evidence, and §9 reports `rms(nm_gain)` for every `nm_pos` run — **if any `nm_pos` seed finishes with `nm_gain` still at its 0.1 init, that run is reported as a failed unlock and its bpc is not read as an arm result** | abort if any other candidate is a saddle with no fallback |
| **G9** | mutation lockfile absent | refuse to start |

**G8's second clause is the gate this experiment adds**, and it is written
because a rung whose gain never unlocked would train, score plausibly, and be
`nm_source="off"` wearing an arm's name.

---

## 8. Entry conditions

* Pre-registration SHA-256 stamped into `exp_023_run_manifest.json` **before**
  the first run and re-checked after the last.
* `scripts/exp/023_modulator_results.py` committed **before any bpc is read.**
* **Budget, from §2.6 and not guessed:**
  3 × 398.9 + 3 × 458.4 + 3 × 468.6 + 3 × 598.1 + 3 × 598.1 = **7,566 s =
  2.10 GPU-h** of training, plus evaluation, the horizon probe and the
  generalisation-gap probe. **~2.5 GPU-h budgeted.**
* `EXP_022` completes first. Nothing else runs on the GPU while a timed run is
  in flight.

---

## 9. Results

*(appended after the run — nothing above this line is rewritten)*

**Closed 2026-08-20, ~2.1 GPU-hours** (15 training runs at 7,177 s, plus
evaluation, a 24-checkpoint horizon sweep and the §2.5/§2.6 pre-flights already
recorded above). Pre-registration SHA-256 stamped into `exp_023_run_manifest.json`
before the first run and re-checked after the last: **unchanged**, and re-verified
at resolution (`prereg_matches_manifest: true`). All 15 runs completed, **none
diverged**, G2 exact on 15/15 **and** every rung's realised count equal to its own
closed form, no G5 violation, no K1 unexpected field, no VRAM alarm. The horizon
probe's F1 self-check passed on all 24 checkpoints (`f1_failures: []`).

**G8's second clause passed on all three `nm_pos` runs** — `rms(nm_gain)` =
0.9927 / 0.9765 / 0.9825 against the 0.1 init, so the two-step unlock §2.4
derived actually happened and no run is excluded. **The resolver was committed
before any bpc was read** (`98d3f62`), which is §8's entry condition.

### 9.1 H6 FIRST, and separately: the headline replicates out of sample

§4 required this before every other bar, because a replication and a power
increase are not the same number.

> **H6 HELD.** `nm_local` seeds 3–5 against `if_anchor` seeds 3–5, paired,
> **fresh seeds only**: **−0.02535 bpc, paired t = −8.45, 3/3 seeds negative**,
> 2.7× the bar. `EXP_021`'s headline was −0.03371.

Per-seed: 2.22010 / 2.22285 / 2.22315 against 2.24518 / 2.25353 / 2.24344.
**The arm is real and it replicates**, at **75 % of the size** `EXP_021`
reported. The three seeds that produced the original number were not the reason
for it.

### 9.2 The scoreboard

| bar | verdict | one line |
|---|---|---|
| **H6** out-of-sample replication | **HELD** | −0.02535, t = −8.45, 3/3 fresh seeds |
| **H1** the constant rung vs the anchor | **HELD** | **−0.02975**, Welch t = −11.50, n = [3, 6] |
| **H2** does time-variation buy anything at all? | **UNRESOLVED**, and the sign is wrong | **+0.00022**, Welch t = 0.11 — 42× *below* the bar |
| **H3** is the time-variation positional? | **FAILED** | +0.01099, paired t = +11.27, **0/3** seeds. Predicted BETTER; `pos` is WORSE than `const` |
| **H4** `local` vs `rolled` at n = 6 | **UNRESOLVED** | −0.00221, t = −1.07, 5/6 seeds |
| **H5** does data dependence beat position? | **HELD** | −0.01077, Welch t = −6.70 |
| **H7** findability across the ladder | reported — **FLAT** | `rms(kappa)` max/min = **1.1176** across four rungs |
| **H8** decomposition by context | reported, §9.6 | four rungs, one total, four different routes |
| **H9** cost | reported, §9.7 | 1.12 / 1.14 / 1.47 / 1.47 realised |

### 9.3 The ladder, in order of increasing signal content

| rung | what `DA` knows | mean bpc | vs anchor | `rms(kappa)` | `sd(DA)` |
|---|---|---:|---:|---:|---:|
| `if_anchor` (n = 6) | — | 2.24992 | — | — | — |
| **`nm_const`** (n = 3) | **nothing** | **2.22016** | **−0.02975** | 0.9822 | 0.0000 |
| `nm_pos` (n = 3) *(non-deployable)* | **when**, only | 2.23116 | −0.01876 | 0.9839 | 0.0794 |
| `nm_rolled` (n = 6) | when, and what surprise looks like in general | 2.22260 | −0.02732 | 0.9616 | 0.0442 |
| `nm_local` (n = 6) | when, and what surprised **this** sequence | 2.22038 | −0.02953 | **1.0747** | **0.2283** |

**Read the first and last rows together. A rung that knows NOTHING is worth
−0.02975 and a rung that knows this sequence's own prediction error is worth
−0.02953.** They are 0.00022 apart, and the one that knows nothing is nominally
ahead.

### 9.4 §5's cell, and it is the one §0 named as the risk

§0 opened by naming the rung that could deflate the headline:

> `cur * (1 + kappa_c · DA_t)` with `DA` held **constant** is
> `cur * (1 + kappa_c)` — a learned per-channel gain on the input current, which
> by `EXP_008` Identity 1 is **exactly a learned per-channel threshold** … until
> the constant rung is run, this project cannot say whether its feedforward
> modulator won by being a modulator or by being a gain it already knew about.

**It won by being a gain it already knew about.** H1 held and H2 did not, so §5's
cell is:

> **"`EXP_021`'s arm is a learned per-channel gain, and the time-variation is
> free"** — the headline is restated, `EXP_021` is not retracted, and #6 becomes
> load-bearing for both.

**A vocabulary gap, reported rather than smoothed.** §5's cell says "H2 fails"
and the resolver returned **UNRESOLVED**, which is the honest verdict for a
+0.00022 difference against a 0.00922 bar: this experiment establishes that
time-variation is **not worth 2σ**, not that it is worth zero. The substance of
the cell applies — H2 did not hold, and its point estimate points the wrong way —
but the verdict is UNRESOLVED and is recorded as such. **This is the same defect
`EXP_022`'s H6 hit on the same day**: a decision table with cells for HELD and
FAILED and none for the verdict a 2σ bar most often returns. Referred to
decision **#9**, not repaired here.

**§6 item 1 said in advance which way this would cut, and it cut that way.**
`nm_const` is a gain on the *current* at 1 of 2 layers, so it is a **lower bound**
on what adopted arm #5 would give here; if it gains, #5 gains at least as much,
and H2 becomes harder to pass rather than easier. That asymmetry was written
before the run and it is exactly what happened.

**`EXP_008`'s figure is NOT quoted as a result**, per §6 item 2 and the standing
rule that its gain may not enter a headline until #6 is ruled. What is said here
is structural: the winning rung's *mechanism* is one this project has already
adopted, and that is a statement about identity, not about size.

### 9.5 H3 and H4: neither control explains the arm, and one of them is refuted

**H3 FAILED, and it failed in the informative direction.** §1's argument was
that `nm_rolled` preserves the time index, so a misaligned RPE still carries
"how far into the window am I", and that `pos` isolates that leak. If the leak
were the mechanism, `pos` should recover most of the gain. Instead **`nm_pos` is
0.01099 bpc WORSE than `nm_const`, on 0 of 3 seeds better, at paired t = +11.27**
— a clean, large, wrong-signed result. Giving the modulator *only* the position
is worse than giving it nothing at all.

**So the positional leak is not what `nm_rolled` was riding**, and §5's cell
"the gain is time-varying and positional" is **not** the one that fires.

**H4 UNRESOLVED at n = 6, and the effect got smaller rather than clearer.**
`EXP_021` measured `local − rolled` at −0.00577 with paired se 0.0018 and §11
item 2 priced n = 6 at ~0.6 GPU-h precisely because that lands ~2 se from the
bar. The power was bought. **At n = 6 the difference is −0.00221, t = −1.07,
5/6 seeds** — under half the n = 3 estimate and further from the bar than
before. Alignment is not worth 2σ, and the extra seeds did not rescue it; they
shrank it.

**H5 HELD at −0.01077**, and it must be read against H3 rather than as a win.
`nm_local` beats `nm_pos` because `nm_pos` is the worst rung on the ladder, not
because data dependence pays: `nm_local` does not beat `nm_const`, which knows
strictly less than `nm_pos` does about time.

### 9.6 H8 — four rungs, one total, four different routes

`EXP_004` §10.3's split, cut at the baseline horizon of 7, positive = better,
against `if_anchor`. `EXP_020`'s bitwise-copy noise floor is −0.00499 total,
−0.01395 at `c = 0`, +0.00944 within reach, −0.00047 beyond.

| rung | total | `c = 0` | within reach | beyond horizon |
|---|---:|---:|---:|---:|
| `nm_const` | +0.02946 | −0.00773 | **+0.03683** | +0.00036 |
| `nm_pos` | +0.02628 | **+0.03072** | −0.00455 | +0.00011 |
| `nm_rolled` | +0.02696 | **−0.06324** | **+0.09045** | −0.00025 |
| `nm_local` | +0.02940 | −0.01376 | **+0.04370** | −0.00054 |

**The totals span 0.0032 and the zero-context components span 0.0940.** These
four arms are not doing the same thing, and the bpc column cannot tell. `pos` is
the mirror image of `rolled`: it *gains* +0.031 at zero context and loses within
reach, while `rolled` pays −0.063 at zero context to buy +0.090 within reach.
`const` and `local` sit between them and land at the same place.

**H8's prediction was that a rung that knows less keeps the within-reach shape
and loses the rest. It does not.** `nm_const` knows the least of the deployable
rungs and has the *cleanest* within-reach gain (+0.0368 with only −0.0077 paid
at `c = 0`); `nm_rolled` knows more and has the most violent trade. The
prediction is reported as it fired and is **wrong**.

**Nothing anywhere on this ladder is beyond the horizon.** All four `beyond`
components are within 0.0006 of zero and inside `EXP_020`'s −0.00047 floor.
`EXP_021` §9.4 found the same for `nm_local` alone; four rungs now say it.
**A rank-1 time-varying gain on the input current does not extend reach, whatever
drives it** — and that is the most transferable sentence in this experiment.

### 9.7 H7 — the optimiser buys the form, not the signal

| rung | `rms(nm_gain)` | `sd(DA)` |
|---|---:|---:|
| `nm_const` | 0.9822 | 0.0000 |
| `nm_pos` | 0.9839 | 0.0794 |
| `nm_rolled` | 0.9616 | 0.0442 |
| `nm_local` | **1.0747** | **0.2283** |

**max/min = 1.1176.** `EXP_018` measured **26** on the identical quantity;
`EXP_021` measured 1.12 with two rungs and could not say why. §4 fixed the
reading in advance — *"if all four rungs land within ~1.2× of each other, the
optimiser is buying the FORM and not the signal, and that is the ladder's
headline whichever way the bpc goes"* — and §5 fixed its sentence:

> **"the optimiser buys the form, not the signal."**

That is now measured across four rungs whose driving signals differ by every
axis available: one has **no variance at all** (`sd(DA) = 0` by construction),
one varies 5× more than another, and the optimiser grows `kappa` ~10× from its
0.1 init in every case and cannot separate them. **The bpc ladder and the
`rms(kappa)` ladder are the same finding stated twice.**

**`sd(DA)` is what rules out the trivial explanation.** `nm_local`'s driving
signal really does vary — 0.2283, 5.2× `nm_rolled`'s and 2.9× `nm_pos`'s — so
the four rungs are not accidentally the same arm. They receive genuinely
different signals and produce the same number.

### 9.8 THE CONFOUND THIS LADDER INHERITED, found after the runs and stated here

**Every rung on this ladder trained at `Config.nm_b_init = -0.7`, and `EXP_021`
§2.6 certifies the arm at `b = -2.97`.** `LocalDopamineCharLM.__init__` defaults
to −2.97; `build_model` forwards `cfg.nm_b_init`; the Config value wins.
Verified in each run's own `config.json`.

With `nm_a_init = 0.0` the init logit is `b` exactly, so
`phi = (rbar − sigmoid(b))·b`. At layer 0's measured init rate of 0.0488:

| `b` | `sigmoid(b)` | `phi` | `phi/tau` | `sech²` |
|---:|---:|---:|---:|---:|
| **−2.97** *(certified)* | 0.0488 | −0.0000 | −0.00 | **1.000** |
| **−0.70** *(shipped)* | 0.3318 | +0.1981 | +3.46 | **0.004** |

**The squash starts saturated**, so at step 0 the RPE rungs are algebraically the
`const` rung, and the gradient into `(a, b)` is attenuated ~190× against the
point §2.6 certifies. The same arithmetic reproduces `EXP_021` §2.5's own
rejected number (−0.8797 against its reported −0.879), which is what confirms
the formula. Pinned by
`tests/test_neuromod.py::test_nm_b_init_is_the_value_the_runs_actually_used`,
verified to fail against −2.97 before being trusted.

**What it bears on, stated narrowly.** H2 and H5 are the bars it touches: if the
RPE rungs begin as `const` and desaturate over training, then part of what H2
compares is a rung against its own early self. **It does not explain the
result**, for a reason this experiment can measure: `sd(DA) = 0.2283` at the
final checkpoint, so `nm_local` did desaturate and was genuinely data-dependent
by the end. And it cannot touch H1 or H3 at all — `const` has no squash and
`pos` does not carry an RPE.

**It is NOT a limitation §6 wrote**, and this section is where it is recorded
rather than in §6, which is above the line. `nm_tau` is learned (§2.6), which is
the design's own answer to the 7× rate rise, so the saturation is not
permanent — but the value was never the one the certification measured.

**Nothing is retracted and nothing is re-run.** −0.7 is what every committed
figure in `EXP_021` and here used, and `CONTRIBUTING.md` §4 forbids moving a
baseline to make a diagnostic look better. **The re-run at −2.97 is referred with
a price in §10 and is Elliot's.**

### 9.9 H9 — cost, and the practical sentence that falls out of it

| rung | pre-flighted §2.6 | realised | params |
|---|---:|---:|---:|
| `nm_const` | 1.149 | **1.116** | 735,949 |
| `nm_pos` | 1.175 | **1.143** | 736,205 |
| `nm_rolled` | 1.500 | **1.474** | 736,974 |
| `nm_local` | 1.500 | **1.470** | 736,974 |

The pre-flight was accurate to ~3 % on all four rungs — the fourth time
`014_calibrate_cost.py` has transferred. And the decomposition §2.6 offered as a
by-product is confirmed: the broadcast multiply and its backward account for
~0.12–0.14 of `nm_local`'s 0.47, and the Bernoulli predictor accounts for the
other ~0.33.

**Which gives the sentence: the same −0.0295 bpc is available at 1.12× or at
1.47×.** The predictor costs 32 % of a training run and, on this ladder, buys
+0.00022 bpc in the wrong direction.

### 9.10 `n_contexts_significantly_worse`, reported as a marker and used as nothing

§4 fixed this in advance and it is honoured. `EXP_020` §9.5's bitwise-copy
control read 120 of 128 on this statistic. Fresh evidence for open decision
**#2**; no redefinition is proposed here.

---

## 10. Referred to Elliot, and not decided here

1. **The cheapest rung is the best rung, and it is a mechanism already adopted.**
   `nm_const` is −0.02975 at 1.12×; `nm_local` is −0.02953 at 1.47×. Whether
   that changes anything about adopted arm #5 is **#6's business and #5's**, and
   both are Elliot's. Nothing here re-measures arm #5 — §6 item 1 is why
   `nm_const` is a lower bound on it and not a replacement for it.
2. **The `nm_b_init` re-run** (§9.8). Three seeds of `nm_local` and three of
   `nm_rolled` at `b = -2.97`, the value `EXP_021` §2.6 certifies, at ~0.6
   GPU-h by §2.6's measured 598 s per run. It is the only measurement that can
   say whether alignment is worth nothing or was merely started where it could
   not be learned. **When a correction would flatter the work, refer it; when it
   would hurt, measure it** — this one could go either way, which is why it is
   referred with a price rather than run on the resolver's own authority.
3. **H4 was priced at n = 6 and n = 6 did not resolve it.** The effect halved.
   A further seed increase is not obviously the answer, and `EXP_021` §11 item
   2's pricing method — se from n = 3, extrapolated — is what should be
   re-examined before another cell is bought that way.
4. **§5 and `EXP_022` §5 both lack a cell for UNRESOLVED**, which is the verdict
   a 2σ bar most often returns. Two experiments closed on the same day hit it.
   This is decision **#9**'s substance, and the corrected rule is referred
   forward rather than applied.
5. **H8's prediction was wrong** (§9.6) and the decomposition is the most
   informative table in this experiment: four arms with the same total and a
   zero-context spread of 0.094. **No bar was pre-registered on the components**
   and none is invented. An experiment aimed at the c = 0 / within-reach trade
   is not proposed here; §7.1 is the report's.
6. **Nothing on this ladder extends reach**, and that now rests on four rungs
   rather than one. Combined with `EXP_022` §9.6 — where a wider *head* moved
   within-reach by +0.0431 and beyond-horizon by +0.0015 — the phase has yet to
   move the beyond-horizon component with anything.

