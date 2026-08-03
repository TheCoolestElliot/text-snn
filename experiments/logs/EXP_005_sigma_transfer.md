# EXP_005 — Does σ transfer to a two-state neuron?

**Pre-registered:** 2026-08-03, before any additional seed was launched.
**Status:** **CLOSED** 2026-08-03. S2 and S5 held; **S1, S3 and S4 failed.**
Candidate #14 is closed and no longer blocks. Results in §9.
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

---

## 8.1 Entry conditions, resolved — 2026-08-03, before the first training run

| # | Condition | Outcome |
|---|---|---|
| — | existing suite green | **242 passed**, matching `EXP_004` §9.1 |
| K2 | no mutation campaign running | held — no lockfile, and the driver re-checks before **every** run, not once |
| K1 | `twocomp_s0`'s config pinned as reference | held — the train command is lifted from its `launch.json` rather than retyped |
| — | nothing in `src/snn/` modified | held — `git status src/` clean throughout |

### 8.2 A methods deviation, recorded rather than absorbed

`scripts/launch.py` exists to satisfy risk R6: unattended runs are launched as
**detached** OS processes so that a dying parent cannot take them with it. On this
machine, in this session, **it does not work** — a detached child is killed at
roughly 60 seconds regardless of `DETACHED_PROCESS` and
`CREATE_NEW_PROCESS_GROUP`, because the harness the parent runs under places it in
a job object the child cannot escape.

This was not assumed. The first launch of `twocomp_s3` died at step ~1 750 with
nothing in any log, which is exactly the silent-failure signature the project
worries about, so it was tested directly: a detached heartbeat process writing one
line per second stopped at **line 64** and vanished. The partial run was deleted
and the four seeds were re-run under the harness's own durable background
mechanism instead.

**Consequence for the record:** these four runs were *not* launched the way every
previous run in this project was. The sequencing guarantee §2 asks for is
unaffected — it comes from the driver, which blocks on each child in turn, and the
measured wall-clocks (513–515 s against `EXP_004`'s 525.3 s, 0.98×) show no
contention. But `launch.py`'s R6 protection was not in force, and a reader
comparing these runs with the Phase-2 campaign should know that.

---

## 9. Results

**Run 2026-08-03.** Four seeds, 20 000 steps each, strictly sequential.
**2 113 s total = 0.59 GPU-hours**, against §6.3's 0.55 estimate for candidate #14.
Raw JSON: `docs/reports/data/exp_005_sigma_transfer.json`,
`docs/reports/data/exp_005_memory_horizon.json`,
`docs/reports/data/exp_005_run_manifest.json`.

### 9.1 The headline: σ does not transfer, and it is *smaller*

| Arm | n | Test bpc (fresh) | Test bpc (carried) | σ ratio (carried) |
|---|---:|---|---|---:|
| SNN baseline (β=0.5) | 5 | 2.2697 ± 0.00450 | 2.2531 ± 0.00461 | 1.000 |
| **Two-compartment** | **7** | **2.1457 ± 0.00296** | **2.1187 ± 0.00313** | **0.680** |

Per-seed carried: 2.1216, 2.1121, 2.1184, **2.1199, 2.1208, 2.1191, 2.1189**.

**The two-compartment neuron's seed-to-seed spread is about two-thirds of the
baseline's** — 0.680× carried, 0.658× fresh. `EXP_004`'s n = 3 reading of 0.00487
was inflated by a single low seed (`s1` at 2.1121); the four new seeds cluster
between 2.1189 and 2.1208 and pull the estimate down to 0.00313.

**This fails S1 as written, and the failure is in the safe direction.** A smaller σ
means the 0.00922 bar every verdict has been held to is *conservative* for this
family, not lax. Nothing already adopted is at risk; if anything the adoptions were
under-claimed.

### 9.2 The predictions, resolved as written

| # | Prediction | Measured | Verdict |
|---|---|---|---|
| **S1** | carried sd within [0.7×, 1.4×] of 0.00461 | **0.680×** | **FAILED** |
| **S2** | no `EXP_004` verdict changes at the new bar | none changed | **held** |
| **S3** | worse at **more** than 5 of 128 contexts under the SE bar | **5** — unchanged | **FAILED** |
| **S4** | per-context sd keeps the baseline's shape | shape differs; see §9.5 | **FAILED** |
| **S5** | n = 7 carried mean within 2σ of 2.1174 | **2.1187**, Δ = +0.0013 | **held** |

**Decision cell (§4): S5 holds ∧ S2 holds ⇒ #14 is CLOSED.** The whole-split bar
for the two-compartment family is **2σ = 0.00627** and is quoted as such from here
on. Phase 4 may issue further 2σ verdicts in this family.

### 9.3 What S1's failure does and does not license

§3's power note was written for exactly this outcome and it constrains the reading.
The point estimate is outside the pre-registered band; the **measurement cannot
demonstrate that it is outside**. The variance ratio is **0.462**, and an F(6,4)
test only resolves ratios outside **[0.1605, 9.20]**, so equality is not rejected.
Equivalently, the 95 % interval for the arm's σ is roughly [0.00202, 0.00689], and
the baseline's 0.00461 sits inside it.

**So the honest claim is the weak one: the best estimate of σ for this arm is
0.00313, it is 32 % below the inherited value, and n = 7 is not enough to prove the
difference is real.** The band was failed; the difference was not established. Both
statements are true and only reporting the first would be an overclaim.

### 9.4 `EXP_004`'s verdicts, re-resolved at the new bar

| Check | Old bar (2σ = 0.00922) | New bar (2σ = 0.00627) | Held |
|---|---|---|---|
| Adoption: gain 0.1344 bpc | 29.2 σ | **42.9 σ** | yes |
| **T1** median horizon ≥ 14 | median **47** (38/47/45/57/48/47/53) | median **63** | yes |
| **T5** gain < 0.2279 bpc | 0.1344 | 0.1344 | yes |

Two notes on the horizon, because the second is a trap:

* At n = 7 the horizon median is **47**, against `EXP_004`'s 45 at n = 3. The
  spread across seeds is wide — 38 to 57 — and that spread is itself the subject
  of §9.5.
* **The horizon is defined against a 2σ tolerance, so a smaller σ lengthens it
  mechanically.** At the new bar the arm's median horizon is 63, which is *past*
  the GRU anchor's committed 57–60 — and that comparison is **not valid** and is
  not being made. The anchor's horizon was measured at the baseline's σ, and
  re-measuring σ on the GRU is a separate experiment that has not been run. The
  63 is recorded only to show that the bar change moves this statistic, which is a
  reason to quote the horizon with its tolerance attached from now on.

### 9.5 S3 and S4: the bar's construction was wrong, and it did not matter — yet

§1.2 predicted the per-context bar is built on the wrong scale, and it is. At
c = 127 the bar as built is **0.0001 bpc** while the correct standard-error bar is
**0.0047** — **45× too small**, because the excess curve is measured relative to
each seed's own asymptote and therefore excludes the asymptote's own 0.0045 bpc
spread.

**And the count does not move.** Under all three bars the two-compartment arm is
worse than the baseline at exactly **5 of 128 contexts, c = 3…7** — the same five
`EXP_004` §10.4 reported.

| Bar | contexts worse | which |
|---|---:|---|
| as built (`EXP_004` §10.4) | **5 / 128** | c = 3…7 |
| SE on the excess curve | **5 / 128** | c = 3…7 |
| **SE on the absolute curve** *(correct)* | **5 / 128** | c = 3…7 |

**S3 was stated in the direction that hurts the adopted arm, and it was wrong.**
`EXP_004` §10.4's trade is not broader than reported; it is exactly as reported,
and it is now robust to the bar's construction rather than merely asserted under
one. That is a better outcome for the arm than the pre-registration expected, and
it is recorded as a failed prediction rather than as a vindication.

The defect is still real and still prospective: it is harmless here **only because
this arm is far better than the baseline at long contexts and is nowhere near
being flagged there.** An arm that is slightly worse beyond c = 16 would be flagged
spuriously by a bar 45× too small. Per §4's rider, **a change to §6.2's bar
definition is referred to Elliot** with both numbers on the table; this experiment
does not redefine a Phase-3 acceptance criterion.

**S4 failed, and it is the most consequential failure here.** The per-context noise
bar is arm-dependent in *shape*, not only in scale:

| | baseline (n=5) | two-compartment (n=7) | ratio |
|---|---:|---:|---:|
| sd of excess at c = 16 | 0.00013 | **0.00273** | **21×** |
| sd of excess at c = 64 | 0.00014 | 0.00192 | 14× |
| max sd at c ≤ 4 | 0.01420 | 0.00802 | 0.56× |
| max-short ÷ mean-long | **~70×** | **~4×** | — |

The mechanism is not mysterious and it is worth stating because it generalises:
**the baseline's excess-sd collapses beyond c ≈ 7 because it has no memory left to
vary.** The two-compartment arm still has memory out to c ≈ 47, and *how much*
differs by seed (horizons 38 to 57), so its excess keeps varying all the way to
c = 127.

Per §4's rider, this means **§6.2's per-context calibration must be redone per arm
rather than measured once on the baseline** — and that is a materially more
expensive criterion, reported as a cost rather than absorbed. Concretely: any arm
whose horizon differs from the baseline's needs its own bars, which means the
criterion costs ≥3 seeds of the *candidate* before its curve can be judged, not
just 3 seeds to establish its mean.

### 9.6 An observation that was not asked for

Not pre-registered; recorded because it is free and because a reader will want it.
`EXP_004` §10.3's gain decomposition survives four more seeds essentially
unchanged:

| Component | `EXP_004`, n = 3 | **EXP_005, n = 7** |
|---|---:|---:|
| At zero context | +40.6 % | **+41.0 %** |
| Within the baseline's 7-char reach | −47.7 % | **−48.4 %** |
| Beyond the horizon | +107.2 % | **+107.4 %** |

And the remaining gap against the anchor: 0.0687 / 0.1759 / 0.0952, total **0.3398**
(against `EXP_004`'s 0.0685 / 0.1759 / 0.0936 / 0.3380). **The within-reach
component is 51.8 % of what is left**, confirming `EXP_004` §10.11 item 3 on seven
seeds rather than three.

### 9.7 Reproduction checks

Everything this experiment re-measured came back unchanged:

* **F1 passed on 15 of 15 checkpoints**, residuals down to 4.8e-09 bpc — the check
  that caught a contaminated run in Phase 3.
* **All five baseline seeds returned horizon 7 again**, and the GRU 60 / 58 / 57.
* **The §5 I5-gap decomposition reproduced exactly**: 0.4633 total, 0.1193 at zero
  context, 0.1161 within reach, 0.2279 beyond, **51 % not addressable by horizon.**
* **The analysis script re-derives `EXP_004`'s committed three-seed numbers on
  every run** — 2.1174 / 2.1443 / 0.00487 / 0.00445 — and aborts if it cannot.
  Worst residual **3.35e-05 bpc**, which is the precision they were quoted to.
* **K1:** all four runs differ from `twocomp_s0` in `seed` and `run_name` only.
* **K3:** 513 / 513 / 513 / 515 s against `EXP_004`'s 525.3 s — **0.98×**, no
  contention.

### 9.8 Status

**CLOSED. S2 and S5 held; S1, S3 and S4 failed and are recorded as failures.**

**Candidate #14 is closed and no longer blocks.** Three things change:

1. **The two-compartment family's bar is 2σ = 0.00627**, from σ = 0.00313 at n = 7.
   The inherited 0.00922 was conservative, so no adopted verdict moves.
2. **σ is not a project constant.** It was measured on one neuron and is 32 %
   different on the next one tried. Any future structurally-different arm needs its
   own σ before its verdicts mean anything — #14 is closed for *this* neuron, not
   in general.
3. **§6.2's per-context bars are arm-specific in shape** (§9.5), and the criterion
   costs more than Phase 3 priced it at. The bar-definition fix is **referred to
   Elliot**, not applied here.
