# EXP_014 — Does this model have room to grow, and does the frozen recipe scale?

**Pre-registered:** 2026-08-08, after the cost calibration in §2.1 — which reads
only wall-clock, VRAM and firing rates, and no bpc against any bar — and
**before any ladder run is trained**. This file is committed before the first
run, which `EXP_013` §0 promised and could not deliver; the hash mechanism in §8
is kept anyway, because a commit proves ordering only if nobody amends it.
**Status:** PRE-REGISTERED — no results yet.
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

*(Empty at pre-registration. Appended after the runs; predictions are resolved as
written, including the ones that fail.)*
