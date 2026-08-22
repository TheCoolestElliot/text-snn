# EXP_014 — Does this model have room to grow, and does the frozen recipe scale?

**Pre-registered:** 2026-08-08, after the cost calibration in §2.1 — which reads
only wall-clock, VRAM and firing rates, and no bpc against any bar — and
**before any ladder run is trained**. This file is committed before the first
run, which `EXP_013` §0 promised and could not deliver; the hash mechanism in §8
is kept anyway, because a commit proves ordering only if nobody amends it.
**Status: CLOSED 2026-08-08 — S1, S2, S3 and S4 all held, and the gate expected to pass is the one that FAILED (G1).** Width bought −0.25238 bpc at 54.75 transferred σ — more than every architectural arm of the phase — and it ran on the plain LIF. Results in §9; folded into `docs/reports/04_phase4_interim.md` §13.**
**Phase:** 4 (controlled experiments). Candidate **#11** on
`03_phase3_candidates.md` §6.3's ranked list, and **rank 2** of
`04_phase4_interim.md` §7.1 since rev 6.

**What it costs:** two training runs at new widths plus one retrain at the
committed width, **~1.10 GPU-hours measured** (§2.1), against §6.3 #11's
estimate of 0.35. The estimate was never a measurement; §2.1 is.

---

## 0. What this experiment is, and what it is not

`EXP_013` §2.2 measured this model's generalisation gap for the first time:
**+0.00594 bpc fresh** (4.96 se from zero, still below the project's own 2σ =
0.00922 adoption bar) and **−0.00180 carried** (three of five seeds negative).
735,437 parameters against ~655M training characters is ~7.3 epochs and ~122
characters per parameter. **The model underfits.** `04_phase4_interim.md` §7
item 6 turned that into a referral — it removes the standing objection to
candidate #11 — and rev 6 placed #11 at rank 2 on the strength of it.

This experiment is the pilot that referral asks for. It is **a pilot and not an
arm**:

* **It cannot adopt anything.** §6.2's Phase-4 acceptance rule requires ≥ 3
  seeds; this is 1 seed per rung, by #11's own specification.
* **It produces no σ at the new widths**, and does not pretend to. Every bar
  below is placed against σ = 0.00461, measured at `d = 512, arch = "snn"`,
  n = 5 — a **transferred** statistic, which `CONTRIBUTING.md` §4 forbids
  relying on quietly. §3.0 states what that costs and how each bar is sized
  against it.
* **It changes no code.** `Config.d_model` already exists and `scripts/train.py`
  already forwards it. `EXP_014` adds **zero lines to `src/`** — the first
  Phase-4 experiment that does. There is consequently no new parameter, no new
  arch, no new kernel, no R10 gradient gate, and no mutation-campaign re-run.
* **The §7.2 gradient-reachability screen does not apply**, and this is on the
  candidate record rather than asserted here:
  `docs/reports/data/phase3_candidates.json`'s C11 carries
  `reachability: {applies: false, screen_keys: [], outcome: "no new parameter kind"}`.

## 1. The question, and the second question the calibration forced into it

**Q1 — does capacity move the number at all, at the frozen recipe?** Nothing in
this project has ever varied width at fixed depth on the committed
architecture. `EXP_003` varied depth and found horizon ∝ √K; `01_reconnaissance.md`
§3.5's width sweep was eager, forward-only and under `no_grad`, and
`02_baseline_report.md` §6.2 withdrew its cost model. So the answer is unknown,
and every capacity-flavoured candidate rev 6 promoted is ranked on an
extrapolation from a gap measurement rather than on a scaling measurement.

**Q2 — does the frozen recipe still train these widths?** §2.1 measured that
`grad_clip = 1.0` **never binds at `d = 512` and binds at every larger width
tested**. That is not a confound this experiment introduces; it is what the
frozen recipe does when it meets a wider model. But it means an underperforming
wide rung has **two** available explanations, and §4 fixes in advance which one
may be reported.

### 1.1 Width, not depth, and why the choice is not free

Depth is ruled out by measurement: `EXP_003` R1 failed and its own arithmetic
puts the horizon at ~137 layers for a useful gain. Width at `n_layers = 2` is
therefore the only capacity axis this pilot can move while keeping every other
frozen field of `02a_phase2_spec.md` §9 intact.

**This does not answer "what size should Phase 5 run", and §9 must not claim it
does.** Phase 5's backbone is the two-compartment neuron plus the now-adopted
standalone threshold, and `EXP_005` and `EXP_012` are two independent
demonstrations that this project's statistics do not transfer between those two
neurons. The ladder is built on the Phase-2 baseline because that is the only
place `EXP_013`'s motivating gap measurement, σ = 0.00461 at n = 5, the §6.2
per-context bars, and a horizon of 7 at 5/5 seeds all exist. Running it on
`twocomp` would put the pilot on an arm whose generalisation gap has never been
measured — reinstating on that arm exactly the objection rev 6 lifted on this
one. The `twocomp` ladder is named in §10 as a referral, not folded in here.

## 2. Design

### 2.1 Calibration — width is priced before the ladder is fixed

`scripts/exp/014_calibrate_cost.py`, committed and run before this file existed;
artifact `docs/reports/data/exp_014_cost_calibration.json`. Six candidate widths,
400 steps each, no evaluation and no checkpointing inside the timed window, every
run directory deleted afterwards. It reads **no bpc against any bar**.

| `d_model` | params | s/step ÷ d=512 | projected 20k-step s | peak VRAM GiB | layer-1 escape (≥1 %) | logged steps over clip |
|---:|---:|---:|---:|---:|---:|---:|
| 512 | 735,437 | 1.000 | 398.9 | 0.609 | 25 | **0 / 17** |
| 1020 | 2,501,245 | 2.948 | 1,175.8 | 1.142 | 25 | 1 / 17 |
| 1024 | 2,519,245 | 2.912 | 1,161.6 | 1.143 | 25 | 1 / 17 |
| 1480 | 4,990,765 | 5.773 | 2,302.8 | 1.643 | 25 | 4 / 17 |
| **1481** | **4,997,099** | 5.938 | 2,368.3 | 1.643 | 25 | 3 / 17 |
| 1488 | 5,041,549 | 5.872 | 2,342.4 | 1.646 | 25 | 2 / 17 |

Three things it settles:

1. **Cost is ~1.10 GPU-hours, 3.1× the ranked 0.35.** Reported as a correction
   to §7.1's column, not absorbed.
2. **There is no GEMM alignment cliff at the odd width.** `d = 1481` is 4-byte
   aligned only and this was an open risk; it is 2.9 % slower than `d = 1480`
   and 1.1 % slower than `d = 1488`. So the ladder uses the
   **exact-error-minimising** widths rather than retreating to round ones —
   `EXP_003` §4's own convention.
3. **The clip binds at width and not at the anchor**, which becomes Q2 and the
   §3 diagnostic D1.

**Why calibrating first does not contaminate the pre-registration.** The widths
are selected on **wall-clock**; the experiment decides on **bpc**. The two
quantities are disjoint, no run scored here survives, and no bar below was
written before this table existed *except* in the sense that matters — none of
these numbers is a bpc. Choosing a configuration from a measurement is normally
what pre-registration forbids, so the argument is recorded rather than assumed.

### 2.2 The ladder

| rung | `d_model` | parameters | error vs nominal | × baseline | log₂ | chars/param |
|---|---:|---:|---:|---:|---:|---:|
| 0.735M | **512** | **735,437** | — (committed) | 1.000 | 0.000 | 122.4 |
| 2.5M | **1020** | **2,501,245** | +0.050 % | 3.401 | 1.766 | 36.0 |
| 5M | **1481** | **4,997,099** | −0.058 % | 6.795 | 2.764 | 18.0 |

Counts computed with the committed `snn.model.spiking_param_count`, which
reproduces 735,437 at `d = 512` — the F1-shaped check that makes the other two
worth quoting.

**The rungs are not equally spaced and no slope may be read as if they were:**
1.766 octaves then 0.998. Stated here so §9 cannot quote a per-octave figure off
the pair.

### 2.3 The single variable

Every field of the frozen Phase-2 recipe is held: `arch = "snn"`, `beta = 0.5`,
`n_layers = 2`, `batch_size = 128`, `seq_len = 256`, `lr = 3e-3`,
`warmup_steps = 200`, cosine, `max_steps = 20000`, `grad_clip = 1.0`,
`weight_decay = 0.1`, `reset = "hard"`, `surrogate = "atan"`, seed 0.
**`d_model` and `run_name` are the only fields that differ**, and K1 (§7)
asserts it field-by-field rather than trusting it. Note what is *not* in the
may-differ set: `seed`. All three rungs are seed 0, so K1 here is a stronger
claim than any previous Phase-4 experiment's.

**The recipe is single-variable; the optimiser's behaviour is not** (§2.1 item
3, and D1).

## 3. Hypotheses

### 3.0 What n = 1 per rung means, stated before the run rather than after

`CONTRIBUTING.md` §2 requires a prediction's power to be stated in advance.

* **There is no σ at 2,501,245 or 4,997,099 parameters and this experiment does
  not produce one.** Every bar is placed against σ = 0.00461, transferred from
  `d = 512`. `EXP_013` §9.5 measured σ moving **0.36× to 3.94×** under a scalar
  hyperparameter that changed no parameter and no function; a 6.8× parameter
  change is a far larger perturbation than that, so the transfer is a real risk
  and not a formality.
* **The guard against it is bar magnitude, not confidence.** S1 is placed at
  **10σ = 0.0461 bpc**, which still clears 2.5σ if σ at the top rung is 3.94×
  the baseline's — the largest inflation this project has ever measured.
* **A miss resolves UNRESOLVED, never FAILED.** At n = 1 this design can show an
  effect *exceeds* a threshold; it can never show one is *absent*. "The band was
  failed" and "the difference was not established" are different sentences.
* **The remedy for any unresolved rung is seeds *added*, never substituted**
  (`CONTRIBUTING.md` §4; `EXP_011`'s `compose_s0` is the standing example).

### S1 — capacity buys bits, by more than a 4× σ error could explain

> `bpc_carried(d = 1481) ≤ 2.25311 − 0.0461 = 2.20701`.

**The anchor is the committed five-seed mean 2.25311, not seed 0's own
2.2554769668134713.** Both numbers exist and they differ by 0.00237, half the
transferred σ, so which one anchors the bar has to be stated rather than left to
whoever reads the artifact. The mean is chosen because it is the figure every
report quotes and because n = 5 determines it better than n = 1; the cost is that
the comparison is an n = 1 point against an n = 5 mean, whose difference has
se ≈ σ·√(1 + 1/5) = 1.095σ, so the 10σ bar is ~9.1 se rather than 10. Seed 0's
own value is used **only** by G1, which compares seed 0 against seed 0.

**Failure mode:** (i) σ at the top rung is unmeasured and the bar is
transferred; (ii) **a one-sided improvement bar is satisfiable by a model that
is worse where it matters** — §6.2 shows `SNN β=0.9` and `K=8` each regressing
126–128 of 128 contexts while moving a mean.

**Guards:** (a) 10σ, sized against the worst σ inflation this project has
measured; (b) a miss is UNRESOLVED, not FAILED; (c) **S1 may not be reported as
held unless S4's per-context decomposition is published beside it**; (d) **S1
may not be read as a statement about capacity unless D1 is reported with it** —
if the clip is binding hard at that rung, "capacity helped" and "the recipe
throttled it less than expected" are not separated by this design.

### S2 — monotone in capacity, scored only where the instrument resolves

> `bpc_carried` is non-increasing across 512 → 1020 → 1481, **scored only on
> adjacent pairs whose |Δ| ≥ 2σ = 0.00922.** A pair below that is recorded as
> *tied at this instrument's resolution* and neither confirms nor breaks it.

**Failure mode:** monotonicity over three n = 1 points is breakable by cross-seed
noise alone; without the resolution gate an inversion of 0.001 bpc would score as
a FAIL of a hypothesis the instrument cannot address.

**Guards:** the ≥2σ scoring gate is **in the bar**, not in the write-up; and the
unequal octave spacing (§2.2) is restated wherever S2 is quoted, so "monotone" is
never read as "constant returns".

### S3 — the generalisation gap, extended along the axis it was blocked on

Characters per parameter falls 122.4 → 36.0 → 18.0 across the ladder.

> `gap(fresh)` rises with parameter count, and at `d = 1481` exceeds the
> baseline's +0.00594 by more than 2σ_gap(fresh) = 0.00536 — **counted at a rung
> only where `Δtest ≤ 0` at that rung.**

**Failure mode:** this is `EXP_012`'s Y5 and `EXP_013`'s N1 exactly — a bar on a
**difference** with neither term constrained. `Δgap = −(Δtrain − Δtest)`, so a
gap grows either because train improved (memorisation, the thing being measured)
or because test degraded (damage). `EXP_013`'s N1 fired at two amplitudes whose
test bpc was 14.8 and 190 se **worse**.

**Guards:** (a) **the `Δtest ≤ 0` condition is in the bar** — this is precisely
the correction `EXP_013` §10 item 1 referred and decision #9 tracks, applied here
at the first opportunity; (b) any `Δgap` below the **2e-3 bpc** reparameterisation
noise floor (§6.5) is reported as *below what this instrument resolves*, never as
a signed number; (c) σ_gap is transferred from `d = 512`, n = 5, and that is
named in the same sentence as the bar.

### S4 — where the bits come from, cut on `EXP_004` §10.3's basis, fixed now

> At any rung where |Δbpc| exceeds 10σ, **more than half of Δbpc lands in the
> zero-context and within-reach components** (`c = 0` and `1 ≤ c ≤ 7`) rather
> than beyond the horizon (`c ≥ 8`); and the **memory horizon does not exceed 8**
> at any rung.

**Failure mode:** the horizon is an integer read off a curve against a 0.00922
tolerance and has **no error bar at n = 1** — `EXP_013` §9.6 saw 6, 7 and 7
across nine runs it concluded did not move the horizon, against a baseline of 5/5
seeds at exactly 7.

**Guards:** (a) the horizon bar is `≤ 8` and **6, 7 and 8 all count as "did not
move"**, with `EXP_013` §9.6's spread quoted inside the bar as the evidence for
that resolution; (b) the three-way cut and its uniform position weights are fixed
**here, before any run** — `EXP_013` had to label its own version post-hoc; (c)
the decomposition must sum to the measured Δ, which is its own self-check; (d)
the share bar is scored only above 10σ, because the shares of a Δ smaller than the
noise are meaningless.

### D1 — a diagnostic, not a bar: does the frozen recipe still train this width?

> At every rung, record: any non-finite loss at any logged step; final test
> firing rate per layer; the step at which layer 1 first exceeds a 1 % rate; and
> **the fraction of logged steps whose `grad_norm` exceeds `grad_clip = 1.0`.**

**This is deliberately not a pass/fail bar**, because a health check that passes
tells you nothing about whether the optimiser is the binding constraint. It gates
*interpretation* (S1 guard (d), and §4), and it is reported whatever it says.

`audit_07_init_pathology.json` already carries a `d = 1024` row whose rates match
`d = 512`'s, and its own variance argument — `Var(Wx) = d·Var(w)·E[x²] = E[x²]/3`,
**independent of d** — is why. §2.1 confirms it empirically to `d = 1488`:
layer 1 escapes at step 25 at every width. So D1's init leg is expected to be
dull; its **clip leg is not**, and that is the one to read.

## 4. Decision rule, fixed now

| cell | verdict |
|---|---|
| **S1 holds ∧ D1's clip engagement at the top rung is comparable to the anchor's** | Capacity moves the number at the frozen recipe. Report the slope, S4's decomposition, and the measured cost. **No size is adopted and no candidate is ranked** — n = 1 cannot clear §6.2's ≥3-seed rule. Refer to Elliot: which size Phase 5 runs, and whether the top rung is reseeded to n = 3 before any number from it is quoted as a result. |
| **S1 holds ∧ the clip is binding materially harder at the top rung** | Capacity moved the number, **and this design cannot attribute how much of it to capacity**. Report both, refer the disentangling run (a clip/lr sweep at fixed width — candidate #13's territory), and do **not** quote the Δ as a capacity coefficient. |
| **S1 unresolved (\|Δ\| < 0.0461)** | **"Capacity was not shown to move the number at this recipe and this n"** — never *"capacity does not help"*. The remedy is seeds added, not a wider ladder. This wording is fixed in advance, as `EXP_013` §4's second cell fixed its own. |
| **S1 holds in the harm direction (top rung ≥ 10σ worse)** | **A finding, not a null.** The frozen recipe does not scale, which bounds every capacity-flavoured candidate §7.1 has just promoted and says §7 item 6's referral was acted on prematurely. Report with the width at which harm begins and with D1 beside it. |
| **any rung shows a non-finite loss** | That rung is a **recipe failure**, is excluded with the exclusion named, and **no seed is replaced** (`CONTRIBUTING.md` §4). |

**A bar that fails does not get the bar rewritten.** `EXP_005` §9.4, `EXP_008`
§9.4, `EXP_011` §0, `EXP_012` §11.4 and `EXP_013` §9.9 are five consecutive
precedents.

## 5. Known limitations, stated in advance

1. **n = 1 per rung.** No σ, no adoption, no per-rung error bar. §3.0.
2. **σ, σ_gap and the §6.2 per-context bars are all transferred from `d = 512`.**
   `CONTRIBUTING.md` §4 says σ is not a project constant; this design cannot
   re-measure it and instead sizes its bars to survive a 4× error.
3. **The clip binds at the new widths and not at the anchor** (§2.1). D1 records
   it; §4 fixes what may be concluded when it matters.
4. **It does not answer what size Phase 5 should run** (§1.1) — wrong neuron.
5. **Two rungs cannot establish a functional form.** No power law is fitted and
   none may be quoted; the octave steps are unequal (§2.2).
6. **Width is not the only capacity axis**, and the readout, the embedding and
   the two `Linear`s all scale together here. This pilot cannot attribute the
   effect to any one of them; candidate #4 (readout capacity) is the arm that
   would.
7. **`d = 1481` is 4-byte aligned only.** §2.1 measured no cliff on this box,
   with this driver, at this batch size. It is not a general claim.

## 6. Self-checks — aborting, not advisory

* **G3 — the x-axis is the pre-registered one.** Each run's
  `final_test.json["params"]` must equal `spiking_param_count(205, d, 2)` and the
  value in §2.2, exactly. An off-by-one width would silently move the axis.
* **G4 — VRAM ceiling.** Peak > 3.0 GiB at any rung aborts and is investigated
  rather than absorbed. The measured top rung is 1.643 GiB, and
  `docs/chat/BUILD_NOTES.md` §2 records that a WDDM spill is a ~50× slowdown
  that does **not** raise — so it would corrupt K3's wall-clock while looking
  like a slow run.
* **G2 — capture.** CUDA-graph capture must succeed at every width; if the
  documented `cuda_graph = False` fallback is used it changes wall-clock and must
  be reported (`EXP_013` P2's precedent).
* **K2 — no mutation campaign.** `src/snn/.MUTATION_CAMPAIGN_RUNNING` re-checked
  **before every child process**, not once.

## 7. Gates

### G1 — does today's tree still produce the committed baseline?

**This is the gate this experiment would not have had without an audit of what
changed under it.** Before today, `src/snn/train.py` had exactly one commit
(`b5f71a9`, Phase 2), so **every committed baseline was trained with the stock
`torch.nn.utils.clip_grad_norm_`**. Decision #7 replaced it with
`_clip_grad_norm_fp64` (rev 6). The bottom rung of this ladder is the committed
baseline, and it would otherwise be the only rung trained on a different
`train.py`.

> **G1: retrain `d = 512, seed 0` as `scale_d512_s0` and require its
> `final_test.json` bpc to equal `snn_beta0.5_s0`'s bitwise** (`==`, not a
> tolerance).

The update is unchanged **iff** `clip_coef = clamp(1.0/(‖g‖+1e-6), max=1.0)` is
exactly 1.0 at every step, i.e. iff `‖g‖ < 0.999999` always. §2.1 measured
`max ‖g‖ = 0.600` over 17 logged steps at this width and the committed run's 81
logged steps peak at 0.3844 — so it is **expected to pass**, but 19,919 steps are
unlogged and a sample of 81 is not a proof. Bitwise is the right standard because
`EXP_013` §9.8 demonstrated 20,000 steps of a *stochastic* arm reproducing
bit-exactly.

* **Passes** → the clip change is inert for this recipe; the committed n = 5
  baseline is a legitimate anchor and a legitimate σ. `scale_d512_s0` is kept as
  the ladder's own bottom rung so all three rungs come from one tree, and its
  identity to the committed run is the evidence.
* **Fails** → **the committed baseline was trained on a different code path, and
  that is a finding that must be reported**, not routed around.
  `scale_d512_s0` becomes the anchor and σ = 0.00461 carries that caveat
  wherever it is quoted.

### K1 — config sameness

Reference `snn_beta0.5_s0/config.json`, field by field.
**`MAY_DIFFER = {"d_model", "run_name"}`** — `seed` deliberately excluded (§2.3).
Fields absent from the reference are excused **only** at the `Config` default, and
every excusal is recorded in the manifest rather than swallowed.

### F1 — a probe reproduces a number it did not produce

* **F1a** — the horizon probe's `k = L` point reproduces each rung's own
  `final_test.json` fresh bpc, written by `scripts/evaluate.py`, a different code
  path. Tolerance 1e-3; `EXP_013`'s residuals were 1.45e-09 – 1.18e-08 at 9/9.
* **F1b** — the gap probe's test leg for each run must reproduce **that run's
  own** committed `final_test.json` bpc, written by `scripts/evaluate.py`. For
  `snn_beta0.5_s0` that is **2.2718995629892054 fresh / 2.2554769668134713
  carried**. The report's **2.26969 / 2.25311 are the five-seed means**, not any
  single run's, and stating a per-run leg against a mean would be a check that
  cannot pass — the kind of borrowed bar `CONTRIBUTING.md` §4 exists to stop.
  Where all five baseline seeds are scored, their mean must reproduce the means
  to 5 dp, which it does (2.269692045956405 / 2.253114842808576).

### K3 — the cost model is checked, not assumed

Wall-clock and peak VRAM per rung against `snn_beta0.5_s0`'s 398.88 s / 0.6089
GiB **and** against §2.1's 400-step projection. This is the project's first
width-cost measurement, so K3 is a measurement rather than a sameness check, and
§2.1's projection ignores capture cost and the eight in-training evaluations —
the discrepancy is the point.

## 8. Entry conditions

1. Full suite green on the GPU **before the first launch, never during one** —
   `EXP_013` §9.8 discarded a run because a test suite started two minutes into
   it. `ruff check .` clean.
2. `src/snn/.MUTATION_CAMPAIGN_RUNNING` absent.
3. This file committed **before** the first ladder run, and its SHA-256 stamped
   into `exp_014_run_manifest.json` before the first run and re-checked after the
   last. §9's stub-reconstruction recipe is recorded there so the guarantee
   survives appending results, exactly as `EXP_013` §8 item 5 did.
4. `exp_014_cost_calibration.json` written, committed, and its widths transcribed
   into §2.2.
5. The resolver `scripts/exp/014_scaling_results.py` written and committed
   **before any rung's bpc is read**, with its bars transcribed from this file
   and `--print-bars` available for eyeball comparison. It may not adopt, may not
   rank, and may not widen a bar that fails.

---

## 9. Results

**CLOSED 2026-08-08. Every prediction held, and the gate that was expected to
pass is the one that failed.** ~1.03 GPU-hours of training (378.6 + 1,112.4 +
2,191.4 s) plus three evaluations, one horizon sweep over eight checkpoints and
one gap sweep over four.

### 9.1 The scoreboard

| rung | params | test bpc fresh | test bpc carried | wall-clock | peak VRAM | firing rate (test, carried) |
|---|---:|---:|---:|---:|---:|---|
| `scale_d512_s0` | 735,437 | 2.27387 | 2.25747 | 378.6 s | 0.6089 GiB | 0.339 / 0.320 |
| `scale_d1020_s0` | 2,501,245 | 2.08760 | **2.06947** | 1,112.4 s | 1.1418 GiB | 0.293 / 0.267 |
| `scale_d1481_s0` | 4,997,099 | 2.02007 | **2.00073** | 2,191.4 s | 1.6434 GiB | 0.209 / 0.185 |

Committed references, all on the **old** tree and all at 735,437 parameters
except where noted: Phase-2 baseline **2.25311** (n=5), the adopted
two-compartment arm **2.11869** (n=7), the composed arm **2.08326** (n=2,
738,509 params), the GRU anchor **1.76741** (n=3).

### 9.2 S1 — HELD, at 54.75 transferred σ

`bpc_carried(d=1481) = 2.00073` against the anchor 2.25311: **−0.25238 bpc**,
against a bar of −0.0461. The margin is 5.5× the bar and **54.75×** the
transferred σ, so the σ-transfer risk §3.0 sized the bar against is not close to
load-bearing — even a 3.94× inflation leaves this at 13.9 σ.

**Against `scale_d512_s0` instead** — the anchor §7's G1-failure branch
pre-registered — it is **−0.25674**. The verdict does not depend on which anchor
is used, which is worth stating because G1 did fail.

### 9.3 S2 — HELD, and returns are diminishing per octave

| step | Δ bpc carried | octaves | Δ per octave | resolves at 2σ |
|---|---:|---:|---:|---|
| d=512 → d=1020 | **−0.18800** | 1.766 | −0.1064 | yes |
| d=1020 → d=1481 | **−0.06874** | 0.998 | −0.0689 | yes |

Both steps resolve by more than an order of magnitude over the 0.00922 gate, so
neither is scored as tied. **Per octave the second step buys 65 % of the first**
— diminishing, but still enormous against anything else this project has
measured. No functional form is fitted and none may be quoted (§5 item 5).

### 9.4 S3 — HELD, and this is the ceiling the pilot existed to find

| rung | gap (fresh) | Δgap vs baseline | Δtest (fresh) | Δtest ≤ 0 | verdict |
|---|---:|---:|---:|---|---|
| d=512 | +0.00592 | −0.00002 | +0.00418 | no | below what this instrument resolves |
| d=1020 | +0.03524 | +0.02930 | −0.18209 | **yes** | **counts** |
| d=1481 | +0.05509 | +0.04915 | −0.24962 | **yes** | **counts** |

**The guard did real work and the bar passed it honestly.** `EXP_013`'s N1 fired
on models whose test bpc was 14.8 and 190 se *worse*; here the gap grows while
the test leg improves by 0.18 and 0.25 bpc. That is memorisation appearing
alongside a real gain, not damage — the distinction the `Δtest ≤ 0` condition
exists to draw, drawn for the first time.

**So the underfitting regime ends between 2.5M and 5M parameters**, i.e. between
36 and 18 characters per parameter. `EXP_013` §2.2's "the model does not
overfit" is confirmed as a statement about **735K parameters** and is now bounded
above.

**An independent confirmation worth more than the bar it serves:** the d=512
leg's fresh gap is **+0.00592**, against `EXP_013` §2.2's **+0.00594** measured
on five different runs, on a different tree, at a different time. Two
measurements agreeing to 2e-05 on a quantity whose 2σ bar is 0.00536 is the
strongest check in this experiment, and it was not designed as one.

### 9.5 S4 — HELD. Width buys per-step capacity, and essentially no reach

| rung | horizon | total Δ | zero-context | within-reach | beyond-horizon | near share |
|---|---:|---:|---:|---:|---:|---:|
| d=512 | 7 | −0.00421 | +0.00712 | −0.01154 | +0.00021 | not scored (below 10σ) |
| d=1020 | **8** | +0.18212 | +0.09679 | +0.08212 | +0.00321 | **98.2 %** |
| d=1481 | **8** | +0.24961 | +0.10926 | +0.12538 | +0.01497 | **94.0 %** |

The horizon moves 7 → 8 → 8, **inside the pre-registered ceiling**, where 6, 7
and 8 all count as "did not move". It sits *at* the ceiling at both wide rungs
and that is reported rather than smoothed: this is a marker, not a demonstration
that width buys one character of reach, because at n = 1 the horizon has no error
bar (§3, S4's guard (a)).

**94–98 % of the gain is at or inside the baseline's own reach, and 6 % or less
lies beyond it.** `03_phase3_candidates.md` §5 *assumed* width was per-step
capacity when it assigned the zero-context and within-reach components to "the
readout, normalisation, width, scale". This is the first measurement of that
assumption, and it holds.

**Note what this does to §7.1's premise.** The within-reach component — 51.8 % of
the remaining gap, the thing §7 item 2 called "still barely attacked" — moves by
**+0.12538** here. The best any architectural arm managed against the baseline
was the threshold arm's +0.0288.

### 9.6 D1 — the recipe trains these widths, with one blind spot named

No non-finite loss anywhere. Firing rates stay in band and in fact **fall
monotonically with width** — 0.339/0.320 → 0.293/0.267 → 0.209/0.185 — so the
network gets sparser as it gets wider, which is an energy-relevant observation
nothing in this project had measured and which no bar here anticipated.

**The clip: 0 of 81 logged steps over 1.0 at every rung, including both wide
ones.** So §4's clean cell is the one that fires.

**But 0/81 is "not at the logged steps", not "never", and the difference
matters here.** §2.1's calibration logged every **25** steps over the first 400
and saw the clip bind **1–4 times** at exactly these widths. These runs log every
**250**, so they cannot see that early transient at all. The honest statement is
that the clip does not bind at width **in steady state**, and that whether it
binds during warm-up is unresolved by this measurement while being *positively
indicated* by the calibration. The interpretation guard in §4 is therefore
satisfied on a measurement with a known hole in it, and S1's margin — 54.75 σ —
is far too large for the transient to be a competing explanation.

### 9.7 G1 — FAILED. Today's tree does not reproduce the committed baseline

| protocol | committed `snn_beta0.5_s0` | retrain `scale_d512_s0` | delta |
|---|---:|---:|---:|
| fresh | 2.2718995629892054 | 2.273872258304913 | **+1.973e-03** |
| carried | 2.2554769668134713 | 2.257471102255831 | **+1.994e-03** |

**Established:** the forward is bit-identical (step 0's loss matches exactly);
`environment.json` is identical field-for-field — same torch, driver, CPU and
`CUBLAS_WORKSPACE_CONFIG`; `config.json` differs only in `run_name` (K1 passed);
step 0's *logged* grad-norm differs by ~1.5e-08, about one ulp; the trajectories
are materially apart by step 250.

**Not established, and not claimed: the mechanism.** The obvious account — the
clip fires at some unlogged step and the fp32/fp64 coefficients differ — **failed
its reproduction**: on CPU, `_clip_grad_norm_fp64` and stock `clip_grad_norm_`
return **bitwise-identical** gradients both below `max_norm` and when the clip
fires. The delta sits at §6.5's ~2e-3 reparameterisation noise floor, which is
suggestive and is not evidence of a cause.

**The chase was then run rather than referred, and it closed. See §9.12.**

**What it does not invalidate:** every committed arm was trained on the old tree,
so the arms remain mutually comparable and the adopted effects (0.03–0.06 bpc)
are 15–30× this delta. **What it removes** is "reproduce a committed number
bitwise" as an available gate, and it means any *new* run must be compared
against a newly trained baseline. This experiment did exactly that (§9.2 reports
both anchors).

### 9.8 Gates

| gate | state |
|---|---|
| **K1** | 3/3. Every rung differs from `snn_beta0.5_s0` in `d_model` and `run_name` only — `seed` deliberately not in the may-differ set |
| **G3** | 3/3 exact: 735,437 / 2,501,245 / 4,997,099, equal to `spiking_param_count` and to §2.2 |
| **G4** | peak 1.6434 GiB at the widest rung against a 3.0 alarm and a 7.96 card |
| **G2** | capture succeeded at every width; no `cuda_graph=False` fallback used |
| **K2** | no mutation-campaign lockfile before any of the six child processes |
| **K3** | realised/projected 0.949 / 0.946 / 0.925 — the 400-step calibration over-predicts by 5–8 %, consistently, which is the in-training evaluations it excludes |
| **F1** | `f1_failures` **empty**; every horizon probe's `k = L` leg reproduced its own run's committed bpc |
| **pre-registration hash** | unchanged across the ladder (`prereg_unchanged: true`) |
| **G1** | **FAILED** — §9.7 |

### 9.9 Predictions, resolved as written

| | statement | verdict |
|---|---|---|
| S1 | capacity buys ≥10σ | **HELD** at 54.75σ |
| S2 | monotone, scored at ≥2σ | **HELD**, both pairs resolve |
| S3 | gap rises, only where Δtest ≤ 0 | **HELD** at both wide rungs; d=512 below resolution |
| S4 | gain is near-context; horizon ≤ 8 | **HELD**, 94–98 % near, horizon 8 |
| D1 | diagnostic | recipe trains these widths; clip blind spot named |
| G1 | tree reproduces the baseline | **FAILED** |

### 9.10 What this does not answer, and what it must not be read as

* **Nothing is adopted and nothing is ranked.** n = 1 per rung; §6.2 needs three
  seeds. `adopts_nothing` and `ranks_nothing` are structural fields in the
  artifact.
* **The comparison against the architectural arms is not parameter-matched.**
  `d=1481` beats the adopted two-compartment arm by 0.118 bpc and the composed
  arm by 0.083 — **at 6.8× the parameters**. It does not show width is a better
  *mechanism* than either; it shows the project has been spending its GPU-hours
  on a 0.03–0.17 bpc axis while an unmeasured 0.25 bpc axis sat beside it, which
  is §7 item 6's referral vindicated rather than any arm refuted.
* **It does not say what size Phase 5 should run** (§1.1) — wrong neuron, and
  `EXP_005`/`EXP_012` are two demonstrations that statistics do not transfer
  between these two neurons.
* **σ at the new widths is still unmeasured**, and S3 now gives a reason to
  expect it to differ: a model that has begun to overfit has a different noise
  structure from one that has not.
* Whether the gap keeps opening past 5M, where the optimum sits, and whether any
  of this survives on `twocomp` are all open.

### 9.12 G1 chased to its origin: decision #7's clip, confirmed by substitution

Run the same day, ~15 GPU-minutes. **`CONTRIBUTING.md` §4 says chase a residual
to its origin and never widen a tolerance around it; this is that chase, and it
reverses the "mechanism not established" reading §9.7 was written with.**

**Four measurements, in the order they were made:**

1. **Today's tree reproduces itself bitwise.** `scale_d512_s0` retrained under a
   different `run_name`: all 81 logged steps and both val bpc identical to full
   precision. So training *is* bit-reproducible across processes, and G1's
   failure is a code change rather than nondeterminism.
2. **The clip fires exactly three times in 20,000 steps** — steps **32, 33, 34**,
   grad norms **1.0329 / 1.1198 / 1.0957**, measured by retraining at
   `log_every = 1`. Every one falls *between* the 250-step logging cadence, which
   is why §9.6 reads 0/81 and why §2.1's 25-step calibration saw it.
3. **The Phase-2 tree reproduces the committed baseline bitwise.** A worktree at
   `b5f71a9` — the commit that last touched `src/snn/train.py`, `kernels.py`,
   `neuron.py`, `surrogate.py`, `data.py` and `evaluate.py`, none of which have
   changed since — retrained `d = 512, seed 0`: **0 of 81 logged steps differ**
   from `snn_beta0.5_s0`, val bpc identical to full precision. The committed
   baseline was never irreproducible; the regression is entirely between
   `b5f71a9` and HEAD.
4. **Substituting the clip back restores bitwise agreement.** Monkeypatching
   `snn.train._clip_grad_norm_fp64` to the stock `clip_grad_norm_` and running
   250 steps on **today's** tree: **no differing loss at any step** against the
   Phase-2 tree's own 250. With the fp64 clip in place, the first differing loss
   is step **47** — the identical step at which the two trees diverge.

**Verdict: `_clip_grad_norm_fp64` is the whole cause.** The three clipped steps
are the only ones on which it can act; there the fp32 and fp64 sums-of-squares
differ by ~1 ulp, so the clip coefficients differ in their last bits and the
updates differ. The perturbation stays below the loss's own fp32 resolution for
thirteen more steps and first becomes visible at step 47, then amplifies over the
remaining ~19,950 steps to **+1.97e-03 bpc** — which lands on §6.5's ~2e-3
reparameterisation noise floor, now with a mechanism attached rather than as a
coincidence.

**A false negative on the way, recorded because it is the more useful half.** The
substitution test was first run for **40 steps** and returned "no divergence",
which was read as exonerating the clip. It stopped **seven steps short** of the
divergence. The probe had been validated as *sensitive* — the two paths' reported
grad norms differ from step 0 — so it looked trustworthy, and a validated probe
run for too short a window is exactly the shape of `EXP_007`'s V4: a check that
clears its bar while telling nobody anything. **The horizon of a null result is
part of the null result.** The corrected test states its own: 250 steps, with
the divergence at 47 well inside it.

**What this changes.** Decision #7's fix is **not numerically inert**, and
`src/snn/train.py`'s docstring and `04_phase4_interim.md` §8 row 7 both say
something close to that — the fix "does not change behaviour for any gradient
that was already representable in fp32". That is true of the **gradients** and
false of the **trajectory**: on any step where the clip actually fires, the
coefficient differs in its last bits and the run diverges. The fix is still the
right fix — it repairs a failure mode that silently zeroes updates forever — but
it is a **tree change that invalidates bitwise comparison against every number
committed before it**, and it should be described that way. Referred to Elliot
(§10 item 1), not applied here.

This file was committed before the first run (`d59f98e`), so git history is the
primary evidence. The hash chain is kept as well, because a commit proves
ordering only if nobody amends it.

`exp_014_run_manifest.json` stamps
`7703641f57a098b7d18c9ae9be4e4fce818dd649a2e2b907e17350918469f22b` both before
the first run and after the last. To reproduce it from this file as it now
stands: take everything up to and including the first byte of `## 9. Results`'s
line, and append the two-line stub this section replaced —

```
## 9. Results

*(Empty at pre-registration. Appended after the runs; predictions are resolved as
written, including the ones that fail.)*
```

— and the SHA-256 of the result is that value, **verified 2026-08-08**. §0–§8 are
byte-identical to what was hashed before the first training run: no hypothesis,
no bar and no decision cell moved after the numbers arrived. The prefix through
the end of §8 hashes to
`e9e148d5fb85c3eb67c384a618834b4ef962fd8933e2da55310aa90d01d21a78`, which is the
value a future revision of this section should be checked against instead.

## 10. Referred to Elliot, and not decided here

1. **G1's failure is explained (§9.12) and what to do about it is not.**
   Decision #7's clip is the cause, confirmed by substitution. Three things are
   Elliot's: whether §8 row 7 and `train.py`'s docstring are corrected to say the
   fix changes the trajectory wherever the clip fires rather than being
   numerically inert; whether the project re-baselines (every committed figure
   predates the change, and arms trained on either side of it are no longer
   bitwise comparable); and whether a standing rule is wanted that a change to
   `src/snn` which alters any committed number must be recorded as such before
   it is merged. **Nothing here argues for reverting the fix** — it repairs a
   failure that silently zeroes updates for the rest of a run.
2. **Whether the top rung is reseeded to n = 3** before any number from it is
   quoted as a result. At ~2,191 s per seed that is ~1.2 GPU-hours.
3. **Whether §7.1's ranking survives this.** Every remaining architectural
   candidate is ranked against gains of 0.03–0.17 bpc; width bought 0.25 at a
   cost the report priced at 0.35 GPU-h and which is really ~1.1.
4. **Whether the `twocomp` ladder is now worth its ~1.2–1.8 GPU-hours**, since
   the sizing answer Phase 5 actually needs is on that neuron and this pilot
   cannot supply it.
5. **§7.1's GPU-h column generally** — one item was measured and came in at
   3.1× its estimate, and every other row descends from the same withdrawn cost
   model (§2.1).

## 11. Corrections appended after closure

Per `CONTRIBUTING.md` §2 a closed log is never rewritten; corrections are
appended. Nothing below changes a hypothesis, a bar, a number or a verdict, and
nothing below is inside the hashed prefix.

1. **§9 has no §9.11, and one heading in it is doing two jobs.** The commit that
   added the G1 chase (`1450796`) replaced the heading
   `### 9.11 The pre-registration guarantee, checkable after the fact` with
   `### 9.12 G1 chased to its origin: decision #7's clip, confirmed by
   substitution`, and kept the paragraphs that sat beneath the old heading. So
   §9 runs 9.1 … 9.10, **9.12**, and the pre-registration-guarantee text —
   the hash chain, the reconstruction recipe, and the
   `e9e148d5…d21a78` prefix hash — now sits **unheaded at the tail of §9.12**,
   where it reads as if it were part of the chase. It is not; it predates it.
   **Neither heading is renumbered here.** `0dce054`'s commit message cites
   "§9.11" and a commit message cannot be corrected, so renumbering would break
   the one reference that is now the only pointer to that text. This note is the
   correction.
2. **The reconstruction recipe in §9.12 is unaffected by this file growing.** It
   hashes a *prefix* — everything up to and including the `## 9. Results` line,
   plus a two-line stub — so appending §11 leaves both stamped values intact.
   Re-verified on the tree that added this section: prefix-through-§8
   `e9e148d5…d21a78` and reconstructed pre-results file
   `7703641f…8469f22b`, both unchanged.
3. **§10 item 1 is partly discharged.** `04_phase4_interim.md` rev 8 (2026-08-10)
   corrects `src/snn/train.py`'s docstring, which claimed the fp64 clip "does not
   change behaviour for any gradient that was already representable in fp32" —
   true of the returned norm, false of the trajectory. Elliot has ruled the
   re-baselining question: **the project does not re-baseline**, and every new
   experiment trains its own anchor on the current tree. The remaining half —
   a standing rule for `src/snn/` changes that move a committed number — is
   **report decision #10, open**. §8 row 7 is deliberately *not* edited: it is a
   decided row, and a correction to a decided row's evidence goes in prose
   beneath the table.
4. **§10 items 3 and 5 are answered in the report, not here.** §7.1's rev-8 note
   offers a re-ranking and applies none, and records that the one ranked cost
   ever measured came in at 3.1×. **Items 2 and 4 remain open and are Elliot's.**
