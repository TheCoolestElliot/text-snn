# EXP_005 — Does σ transfer to a two-state neuron?

**Pre-registered:** 2026-08-03, before any additional seed was launched.
**Status:** PRE-REGISTERED — no results yet.
**Phase:** 4 (controlled experiments). Candidate **#14** of
`03_phase3_candidates.md` §6.3, **promoted to blocking** by `EXP_004` §4.

---

## 1. Why this experiment, and why it is next

`EXP_004`'s decision cell fired `T1 holds ∧ T2 falsified ⇒ ADOPT`, and that cell
carries a rider written before the result was known:

> Candidate **#14** — re-measure σ on a two-state neuron — is promoted to
> *blocking*, because every subsequent 2σ verdict would then be resting on a σ
> measured on a one-state neuron.

So this is not a candidate competing for a slot in the ranking. It is the thing
that has to happen before any other Phase-4 arm's verdict means anything.
`03_phase3_candidates.md` §9.1 item 6 says the same: *"Re-measuring it on the
first structurally different neuron Phase 4 adopts is a ranked item, not an
optional extra."*

**σ = 0.00461 bpc is a whole-split standard deviation measured on the one-state
baseline over five seeds** (`EXP_000`). The two-compartment neuron carries a
second state variable, 2 048 more parameters, a different reset rule on one pole
and a bimodal learned decay (`EXP_004` §10.5). Nothing about that guarantees the
seed-to-seed spread is the same.

`EXP_004` §10.7 gave a first reading from its own three seeds — carried sd
**0.00487** against the baseline's **0.00461** — and said in advance that it does
not substitute for this experiment: *"n = 3 estimates a standard deviation to
about ±40 %."* That statement is the reason this file exists rather than a
citation of §10.7.

### 1.1 There are two bars, not one, and the second is the fragile one

The project uses σ in two different places and they are not the same quantity.

* **The whole-split bar, 2σ = 0.00922 bpc.** Used for adoption verdicts. This is
  the one §9.1 item 6 names.
* **The per-context bars** (§6.2), ranging from 0.0284 at c = 1 to 0.0003 beyond
  c = 16. Used for the §6.2 acceptance criterion. `EXP_004` §10.4 applied these
  to the two-compartment arm to flag **5 of 128 contexts** — and it applied the
  **baseline's** bars, measured on the baseline's five seeds, to a difference
  involving a different arm. §6.2 of the phase-3 report says explicitly that a
  bar measured at one configuration was wrong by up to 30× when moved to another;
  that lesson has not yet been applied to moving a bar between *arms*.

This experiment measures both, because the second is where an unexamined
assumption is doing the most work.

### 1.2 A construction question this experiment also has to settle

The per-context bar is currently **2 × the baseline's per-seed standard
deviation**, and it is compared against a **difference of two arm means**
(`mean of 3 twocomp − mean of 5 baseline`). Those are not the same scale. The
noise on a difference of means is

```
se(Δ_c) = sqrt( s_base,c² / n_base  +  s_arm,c² / n_arm )
```

which is *smaller* than a per-seed sd, so the bar as built is **conservative** —
it under-reports how many contexts are significantly worse. §10.4's "5 of 128" is
therefore a floor, not an estimate, and this experiment can say by how much.

**This is flagged, measured and reported here; it is not unilaterally applied.**
§6.2's criterion is a Phase-3 deliverable and changing its definition is Elliot's
call, not this experiment's. Both bars are reported side by side, the old one
first, so nothing in `EXP_004`'s table becomes uncomparable.

---

## 2. Design

| Field | Value |
|---|---|
| New arms | `twocomp_s3`, `twocomp_s4`, `twocomp_s5`, `twocomp_s6` — **4 new seeds** |
| Total | **n = 7** two-compartment seeds (3 existing + 4 new) |
| Config | **byte-identical to the adopted `EXP_004` arm**: `arch=twocomp`, `w_init=0.1`, `beta_slow=0.95`, and every other field as `experiments/runs/twocomp_s0/config.json`. Only `seed` and `run_name` differ |
| Control | the **five existing** `snn_beta0.5_s*` Phase-2 checkpoints, unretrained and untouched |
| Execution | **sequentially**, one run at a time, so no arm measures contention (the phase-3 report §8's reason for not quoting wall-clock on the depth ladder) |
| Budget | ~0.6 GPU-hours. §6.3 priced #14 at 0.55; four runs at `EXP_004`'s measured 525 s is 0.58, plus the probe |

### 2.1 What is measured

1. **Whole-split test bpc**, both protocols, full split (4 980 736 characters),
   for each new seed. `scripts/evaluate.py`, unmodified — the same instrument
   that produced every other number in this project.
2. **Sample sd over n = 7**, fresh and carried, against the baseline's n = 5.
3. **Per-context excess curves**, c = 0…127, per seed, via
   `scripts/exp/001_memory_horizon.py` — the same probe, the same paired
   statistic. Its **F1 self-check is mandatory** on every new checkpoint: the
   `k = L` point must reproduce that run's own independently committed `fresh`
   bpc to better than 1e-3.
4. **Per-context sd of the two-compartment arm at n = 7**, against the
   baseline's at n = 5.
5. **`EXP_004` §10.4's five flagged contexts, re-evaluated** against (a) the bar
   as built, now with the arm's own seeds contributing, and (b) the
   standard-error bar of §1.2.
6. **The horizon at n = 7**, against `EXP_004`'s median of 45 (38 / 47 / 45).

### 2.2 What is deliberately NOT measured

Nothing about the arm is re-tuned, and no new arm is introduced. Four extra seeds
of an already-adopted configuration is the entire intervention. Any temptation to
report the n = 7 mean as an improved headline is answered in §3, S5, which is
stated in the direction that hurts.

---

## 3. Hypotheses

**Pre-registered predictions.**

| # | Prediction | Threshold |
|---|---|---|
| **S1** | Whole-split σ transfers: carried sample sd over 7 seeds lies within **[0.7×, 1.4×]** of 0.00461, i.e. **[0.00323, 0.00645]** | point estimate; see the power note below |
| **S2** | **No verdict `EXP_004` already issued changes** when re-evaluated at the newly measured whole-split bar: the −0.1357 bpc adoption stays beyond 2σ, and T1/T3/T4/T5 keep their recorded outcomes | the new bar, applied to `EXP_004` §10.2 |
| **S3** | **The trade is worse than reported.** Against the §1.2 standard-error bar the arm is worse than the baseline at **more than 5** of 128 contexts | *stated in the direction that hurts the adopted arm* |
| **S4** | The per-context sd of the two-compartment arm has the same **shape** as the baseline's: largest at c ≤ 4, and at least **10× smaller** by c ≥ 16 | §6.2's calibration finding is a property of the statistic, not of the arm |
| **S5** | **The headline was not seed-luck.** The n = 7 carried mean stays within 2σ = 0.00922 of `EXP_004`'s n = 3 value of **2.1174** | *stated so that it can fail*; four fresh seeds are a real replication test of a three-seed result |

**On S1's power, stated in advance so that a null is not over-read.** At n = 7 the
95 % confidence interval for σ spans roughly **[0.64×, 2.20×]** of the point
estimate (χ²₆). A ±40 % band therefore sits *inside* the measurement's own
resolution: **S1 failing would be strong evidence that σ does not transfer, and
S1 holding is weak evidence that it does.** It is stated because a point estimate
outside the band would be actionable, and because the alternative — quoting a
number with no pre-registered expectation attached — is how a measurement becomes
whatever the author needed it to be. The experiment's real deliverable is a
**better estimate and a corrected bar**, not a hypothesis test, and §4 is written
accordingly.

**On S3, explicitly.** The correction in §1.2 makes the *adopted* arm look worse,
which is the only reason it is safe for this experiment to propose it: the party
that benefits from a laxer bar is the party proposing the arm. Had the correction
gone the other way it would have been referred rather than measured.

---

## 4. Decision rule, fixed now

The two load-bearing predictions are **S5** (did the headline survive more seeds?)
and **S2** (does any verdict change?).

| S5 replication | S2 verdicts stable | Verdict and consequence |
|---|---|---|
| **holds** | **holds** | **#14 is CLOSED.** The whole-split bar for the two-compartment family is set to the n = 7 measurement and quoted as such from here on. `EXP_004`'s adoption stands unchanged and Phase 4 may issue further 2σ verdicts. |
| **holds** | **fails** | **#14 is closed, and `EXP_004` gets an appended correction** naming the verdict that changed. The adoption itself is safe by inspection at 29.4σ, so a change here would land on T3 or on §10.4's context flags, not on the headline. |
| **fails** | either | **`EXP_004`'s headline was measured on three lucky seeds.** The arm is re-reported at n = 7, the adoption is re-opened at the new mean, and every downstream statement — including "the new backbone at 2.1174" — is restated. This is the outcome that would matter most and is the reason four seeds are being run rather than two. |

Two riders, also fixed now:

* **S3 holds** ⇒ `EXP_004` §10.4's "5 of 128" is a floor, the corrected count is
  reported beside it, and **a change to §6.2's bar definition is referred to
  Elliot** with both numbers on the table. This experiment does not redefine a
  Phase-3 acceptance criterion on its own authority.
* **S4 fails** ⇒ the per-context bars are arm-specific in *shape* and not only in
  scale, which would mean §6.2's calibration must be redone per arm rather than
  once. That is a materially more expensive criterion and would be reported as a
  cost, not absorbed.

---

## 5. Known limitations, stated in advance

1. **n = 7 is still a small sample for a variance.** §3's power note gives the
   interval. This experiment improves the estimate from ±40 % to roughly
   −36 %/+120 %; it does not settle σ to three figures and will not be quoted as
   if it had.
2. **The baseline stays at n = 5.** Every comparison is therefore between a
   7-seed and a 5-seed estimate, and the F-test that would compare them formally
   has (6, 4) degrees of freedom — it can only detect a σ ratio outside about
   [0.40, 3.03]. Reported, not hidden behind a p-value.
3. **σ is measured at one training length and one corpus**, exactly as the
   baseline's was. Candidate #13 (training budget) is unrun, so nothing here
   speaks to whether σ transfers across step counts.
4. **The new seeds share the sampler's seeding scheme** with the existing three.
   Seeds 3–6 are fresh values, but the window sampler is derived from the same
   `_mix64` construction, so these are new draws of the same generator rather
   than an independent randomisation of the data order.
5. **This experiment cannot separate a σ that genuinely differs from one that
   differs because the two-compartment arm has a different mean.** It measures
   spread at the configuration that was adopted, which is the configuration the
   bar will be used on, and that is the honest scope.

---

## 6. Failure modes

| # | Failure | Detection |
|---|---|---|
| **K1** | A new seed is not the same arm — a config field drifts, and the "extra seeds" measure a different neuron | Every new run's `config.json` is compared field-by-field against `twocomp_s0`'s; **only `seed` and `run_name` may differ**, and the results script aborts otherwise |
| **K2** | A run silently imports a mutated kernel (phase-3 report §8) | **No mutation campaign runs while any training process is alive.** `EXP_001`'s F1 check runs on every new checkpoint: the probe's `k = L` point must reproduce that run's own committed `fresh` bpc to < 1e-3 |
| **K3** | Runs overlap and measure contention rather than the arm | Launched strictly sequentially; wall-clock recorded per run and compared against `EXP_004`'s 525 s |
| **K4** | σ is computed on a different split or protocol from the baseline's | The same `scripts/evaluate.py`, full test split, both protocols, unmodified |
| **K5** | The probe is edited to accommodate the new seeds and stops being the same instrument | Only the `ARMS` registry gains entries — a data change. The statistic, the k-sweep and the F1 tolerance are untouched, and the Phase-3 artifact is never overwritten (explicit `--out`) |
| **K6** | The n = 7 mean is quietly reported as an improved headline | S5 is stated so that it can fail, and §2.2 forbids it in advance |

---

## 7. Entry conditions

- [ ] Existing test suite green before any run starts (the baseline path unregressed)
- [ ] No mutation campaign running, and none started until every training run has finished (K2)
- [ ] `twocomp_s0`'s config read and pinned as the field-by-field reference (K1)
- [ ] Nothing in `src/snn/` modified by this experiment

---

## 8. What this is not

**This is not a re-run of `EXP_004`.** The arm, the instrument and the analysis
are unchanged; only the number of seeds moves. If the headline shifts, that is a
result about `EXP_004`'s sample size and is to be reported as one.

**It is also not a new candidate.** #14 is a measurement of the project's own
measuring apparatus. It buys no bits per character and is ranked because every
other verdict depends on it.
