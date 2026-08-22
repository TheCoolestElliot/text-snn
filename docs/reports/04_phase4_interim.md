# Phase 4 — controlled experiments: interim report

**Revision 16, 2026-08-22.** Interim, not final: Phase 4 is open and this document
is the running record of it. It exists because Phase 4 had produced six closed
experiments and no report, and a phase whose findings live only in its experiment
logs cannot be reviewed as a phase.

> **Rev 16 folds in `EXP_025` as §22: the neuron buys REACH, the width buys
> CAPACITY, and the plain LIF stops reading at nine characters.** The detached
> two-compartment arm is **−0.13426 bpc** below the plain LIF at 5.0M on 3
> seeds — **79.4 se** — under the **unchanged** frozen recipe, verified
> field-by-field on all twelve runs. **H5 FAILED and its failure is the
> result**: the pre-registered prediction was that the gain is *not*
> predominantly beyond-horizon, and it is **118.6 %** beyond-horizon, with the
> arm **paying −0.02504 within the baseline's own reach**. The curves cross at
> **context 9** and the integer horizon moves **9 → 61** on 3 of 3 seeds. The
> sharpest number is neither: from c = 11 to c = 127 the plain LIF at 5.0M
> improves **+0.00032 bpc** — flat to sd 0.000156 over 116 characters — against
> the detached arm's **+0.09956**, **311×** as much. With §13's S4 that makes
> the pair explicit and measured on one tree: **width buys per-step capacity and
> no reach; the neuron buys reach and pays capacity.** **σ above 735K
> parameters is measured for the first time** at **0.002070**, **0.449×** the
> value `EXP_014` transferred — so §13's S1 was **~122 σ**, not 54.75, and its
> conclusion is *strengthened*. **H2 is UNRESOLVED because its own guard
> fired**, the two σ readings disagreeing. Four `d = 512` runs reproduce
> committed runs **bit-for-bit**, which nothing predicted. **12 runs, 0
> divergences, ~5.17 GPU-h against ~5.1 projected**, zero lines added to `src/`.
> **No arm is adopted, no open decision is ruled — and none is engaged — and
> rev 16 edits no row of §8.**

> **Rev 15 folds in no experiment and corrects four things, the first of them
> rev 14's own note.** The pre-registration audit was reading one wrong key, one
> mis-parsed filename and a single-variant placeholder table, and it had no
> second route; with all four repaired and the object database consulted the
> record reads **7/9, not 2/12** — six experiments verified against a **committed
> blob**, `EXP_013` against the recipe its own log states, and all six
> commit-verified logs also agree with the live file above `## 9. Results`.
> **`EXP_020` and `EXP_021` are unchanged**: one commit each, and it already
> carried results. Then `EXP_024`'s pre-registration: §4's prose generalises to
> *"any arm"* a band the same document scopes correctly to six — **7 of 24
> committed primary arm-gain readings sit outside it and three clear H1's own 2σ
> bar** (`scripts/audit/11_beyond_horizon_census.py`, 41 blocks, 12 artifacts,
> **zero GPU**); its one modulated layer is **layer 1**, which the screen's own
> artifact records at a firing rate of **2.98e-07** against an **exact**-equality
> silence guard, a regime the project has already measured away by step 50 and
> which §§2.2, 2.6 and 6 do not mention; and §8's `1.0117` is `EXP_002`'s
> **kernels-per-timestep**, identical to the N0 baseline's, not a step-time
> ratio — the honest reading is **parity**, and the budget does not move.
> **`EXP_024_gated_decay.md` is not edited**; it is committed at `aff0d49` and a
> commit, not a stamp, is what fixes it. **The four corrections cut in different
> directions and this note takes no position on the R10 tax**, which is decision
> #4's. **No row of §8 is edited, nothing is adopted, and no bar is proposed,
> widened or repaired.** See §8's rev-15 note.

> **Rev 14 folds in `EXP_022` and `EXP_023` as §20 and §21: the cost of a binary
> input was a cost of its DENSITY, and the modulator that won §19 wins just as
> well when its signal is a constant.** A fully binary input code at `q = 0.05`
> is **0.00955 bpc** from the analogue anchor against **0.04569** at `q = 0.50` —
> 79 % of the binarity cost recovered **for no parameters and at 0.97×
> wall-clock** — and 82.8 % of that cost sits at zero context, exactly where
> every sparse rung recovers it. The winning code cannot tell **42 of its 205
> characters apart** and beats the code that distinguishes all of them.
> Meanwhile §19's headline **replicates out of sample** at −0.02535 on three
> fresh seeds, and then a four-rung ladder takes its mechanism away for the
> second time: the rung whose driving signal is a **constant** is worth
> **−0.02975** against the rung that knows this sequence's own prediction error
> at **−0.02953**, `rms(kappa)` is flat across all four at **1.12×** against
> `EXP_018`'s 26, and **nothing anywhere on the ladder reaches beyond the
> horizon**. Two arms in this revision moved the within-reach component more than
> anything before them — a wider *head* by **+0.0431** — and neither is ranked
> here. **No arm is adopted, no open decision is ruled, and rev 14 edits no row
> of §8.**

> **Rev 11 folds in `EXP_017` as §16: the thing §15 proved unbounded can be
> bounded without changing the forward at all, and the arm then trains at width.**
> Detaching the fast pole's reset factor removes `vf` from **both** entries of the
> two-compartment per-step backward Jacobian at once, leaving it **diagonal** and
> bounded by `beta_f = 0.5` — **strictly tighter than the plain LIF's own
> 0.5123596**, at every width, every mix and every slow-pole value. The forward is
> the **same compiled kernel**, so the arm is parameter-*identical* to the adopted
> one and its checkpoint evaluates as `arch="twocomp"` **bit-identically**. On the
> batch that killed `EXP_015`'s run, the adopted rule expands for **85 consecutive
> timesteps worth 10^16**; the detached rule, on the *same membranes from the same
> forward pass*, gives 0.5 and a longest run of **0**. It then trained **20,000
> steps with zero non-finite losses under the unchanged recipe**, scoring
> **1.86290** at 5.0M against the plain LIF's 2.00073 and the GRU's 1.54699.
> **Three things cut the other way and are reported as prominently:** two of three
> *freshly trained* `twocomp` controls **diverged at 735K**, so H3 resolves **NOT
> RUN** and the stability difference is **not established** (Fisher p = 0.40); H4
> resolved **REACH REDUCED** on a single surviving pair; and H5 **breached** — on a
> denominator that was itself a diverged run. **Nothing is adopted, no
> hyperparameter moved, no §8 row is edited, and Phase 5 is not authorised.**
> See §17.

> **Rev 12 folds in `EXP_018` as §17: the model can find its own reward
> prediction error, and conditioning on it costs bits.** A dopamine signal —
> `phi_t = H(p_{t-1}) + log p_{t-1}(x_t)`, zero-mean under the model's own belief
> *by construction*, broadcast as one scalar per `(batch, timestep)` with a
> learned per-channel sensitivity. **D1 is UNRESOLVED by 3.6e-06 bpc**, in the
> direction of harm; the design used an **unpaired** Welch test on **paired**
> data, and the corrected test — measured, not referred, because it hurts —
> says the arm costs **0.0062 bpc on 5/5 seeds** (t = −7.48, p = 1.7e-03).
> **The reusable result is a factor of 26**: the optimiser assigns the aligned
> RPE `rms(k) = 0.195` and the same signal misaligned across the batch
> **0.0075**, so the prediction error *is* information this model can find and
> wants — it just does not pay. It buys **no** extra training fit (+0.00043,
> p = 0.63) while losing 0.0061 at test, so the **train→test gap widens by
> 0.0057**, roughly doubling the 0.0059 §12.2 measured. The per-context
> decomposition is two effects of opposite sign: **−0.0246 at c = 0, where the
> mechanism is structurally inactive**, and **+0.0427 at c = 3**, seven times the
> whole-split mean. Sixteen runs, zero diverged, ~2.9 GPU-hours. **Adopts
> nothing and ranks nothing; it hands #3 a fourth I5 boundary case, and #3 is then
> RESOLVED on Elliot's instruction — feedforward-driven modulators admitted,
> head-driven ones diagnostic-only.**

> **Rev 10 folds in `EXP_016` as §15, and it explains rev 9's divergence — then
> corrects rev 9 for having called it something it was not.** The two neurons'
> backward-through-time differs in one line: the plain LIF's per-step multiplier
> uses **the same variable** in its reset factor and inside its surrogate, which
> is self-limiting and bounds it at **0.5123596 < 1 for every finite membrane at
> every width**; the two-compartment neuron uses **different variables**, and the
> bound breaks at `|w·vs| ≈ 1` because the slow compartment is never reset.
> Measured over 48.5 M sites per layer, the plain LIF returns that bound at three
> widths and **never exceeds it**, while the adopted arm reaches **8.96**. At the
> failing step the cotangent grows from **1.46e−04** to **6.12e+37** across two
> scans — **10^41.62 inside one backward pass** — and the first non-finite value
> is made **inside layer 0's recursion**, on both dispatch paths. **§14.5's "not a
> gradient explosion" is corrected in place** (the correction sits beneath the
> original, which is left standing). Nothing is adopted, no remedy is proposed,
> and **no recipe term is changed** — but `dv` contains no learning rate, no
> schedule, no decay and no clip, which bears on the *form* of decision #11.
> See §16.

> **Rev 9 folds in `EXP_015` as §14, and it changes what Phase 5's problem is.**
> Three things, none of them the one the experiment was designed to measure.
> **(1) The adopted arm does not train at 5.0M parameters** under the frozen
> Phase-2 recipe — both arms containing the two-compartment neuron diverged, the
> plain LIF and the GRU at the identical width did not. **(2) Parameter-matched,
> the anchor scales as well as the spiking model**: the gap moves 0.48570 →
> **0.45374**, so width closes **6.6 %** of it, not the ~52 % the unmatched
> comparison suggested. **(3) The difference is reach, not capacity** — with
> width the GRU's memory horizon goes 57–60 → **97** and the spiking model's goes
> 7 → **8**, while at zero context the two are nearly equal. **Rev 9 also
> corrects an arithmetic error rev 8 introduced** (§16, item (i) — retargeted at
> rev 10, which also corrects the item letter: rev 9 wrote "(f)", and (f) is the
> falsified state-growth hypothesis). No arm is adopted, no size is chosen, no
> phase is authorised; one row is *added* — **#11, open** — for the recipe
> question that now blocks scaling. See §16.

> **Rev 8 folds in `EXP_014` as §13 — the largest single effect this phase has
> measured, and it is not an arm.** Plain width buys **−0.25238 bpc** at 54.75
> transferred σ, more than every architectural arm this project has built,
> at 6.8× the parameters. It **refutes no arm** (§13.6) and **adopts nothing**
> (n = 1 per rung against §6.2's three-seed rule). Its G1 gate failed, and the
> chase found the cause is **decision #7's own fp64 gradient clip** — which is
> therefore **not numerically inert**, correcting a claim `src/snn/train.py`'s
> docstring made in as many words (§13.5). **No verdict moves, no arm is
> adopted, no tolerance is set, no phase is authorised, and no existing decision
> row is edited.** One row is *added* — **#10, open** — for the consequence of
> that clip finding. See §16.

> **Rev 7 runs no experiment, closes nothing, and moves no verdict.** It records
> that rev 6's working-tree state is now committed, corrects three figures that
> had drifted, and flags one ranked cost estimate as unsupported by any
> measurement. **No decision row is edited and all five open decisions stay
> open.** See §15 (renumbered from §13 at rev 8, and from §14 at rev 9).

**Nothing in this document is a decision — every decision in it is Elliot's, and
four of what is now ten have been made by Elliot rather than here.** #8 was a
scheduling question — *does the composition run inside Phase 4?* — answered
**inside Phase 4**. **#1 is Phase-3 §9's sign-off**, signed off 2026-08-05 and
recorded as retrospective. **#5 is the learned per-channel threshold: adopted
standalone**, at rev 6 — the composed arm (`EXP_011`) stays separately
**provisional at n = 2 of a pre-registered 3** and is not thereby "the project's
best model." **#7 is the gradient clip's fp32 accumulator: fixed**, at rev 6. §8
marks all four resolved in their rows; **the other six rows — #2, #3, #4, #6, #9,
and a new #10 added at rev 8 — are open.**

**What rev 6 does.** Four things:

1. **Folds in `EXP_013`** (noise injection) as new §12, pushing the changelog to
   §13 *(rev 8 pushed it again to §14, inserting `EXP_014` as the new §13; rev 9
   pushed it to §15, inserting `EXP_015` as the new §14)*. The arm itself is a null — it cost bits at every amplitude and closed no
   gap — but its calibration measurement is the transferable result: this
   735,437-parameter model's generalisation gap is +0.00594 bpc (fresh) / −0.00180
   bpc (carried, indistinguishable from zero), against ~655M training characters
   and ~7.3 epochs. **The model is not overfitting.** That reframes §6.7 item 3 and
   §7 item 5's own "Phase 5 should carry a training-only arm" ask (§12 answers it,
   with a null) and removes the objection that had kept the parameter-scaling pilot
   (`03_phase3_candidates.md` §6.3 #11) off §7.1's ranking — it is added there now.
2. **Decides #5**, per Elliot's instruction: the threshold is adopted
   **standalone** — −0.0590 bpc at 23.6 se, 0/128 contexts regressed, a
   reparameterisation that folds to zero inference cost. The composed arm is
   explicitly **not** adopted by this — it remains `EXP_011`'s provisional n = 2
   result, and §7.2's gate for it is now #4 alone.
3. **Decides #7**, per Elliot's instruction: `src/snn/train.py`'s
   `_clip_grad_norm_fp64` replaces `clip_grad_norm_`, accumulating the sum of
   squares in fp64 rather than fp32 — the first of `EXP_009` §9.4's two named
   fixes. Verified against the exact failure (a synthetic 5.46e31 gradient, which
   the stock implementation still zeroes and the replacement rescales instead —
   `tests/test_grad_clip.py`) and against the real R5 capture gate
   (`tests/test_graph_equivalence.py`, 10/10, unchanged). It does **not** explain
   or prevent `compose_s0`'s divergence (§10.4) — a different mechanism, already
   established — so #7 closes one bug, not both.
4. **Adds decision #9** — whether Phase 4 pre-registrations need an explicit
   guard-clause review before a bar is committed — per `EXP_012`'s Y5 and
   `EXP_013`'s N1 firing the identical defect twice. The *protocol* rule this
   implies is **already ratified**: `CONTRIBUTING.md` §2 has carried it since
   `51b44c6` (2026-08-05, authored by Elliot, citing both experiments by name).
   What was missing was this table's own row; #9 records that the report was the
   thing lagging, not the practice.

**What rev 5 does, and it is small.** It records **decision #1** and nothing else.
It runs no experiment, moves no number, re-runs no gate and does not claim to have.
Its one consequence for the rest of the report is that `EXP_008` §4's rider —
which made the threshold arm's adoption *"recommended to Elliot, not taken"*
partly because Phase-3 §9 was unticked — **no longer applies, so #5 is ungated by
#1 and still undecided on its own merits.** Its one deliberate non-consequence:
the sign-off does **not** close **#2**, whose defect sits under a §9 box that is
now ticked, and both files say so in place rather than letting the tick absorb it.

**What rev 4 does.** It adds §11: `EXP_012`, which chases the one thing §10.5
named as unexplained and **not** offered as evidence for decision #6 — why the
composed arm's fold-in residual is ~1000× tighter than `EXP_008`'s, given the
same identity, the same six-line fold and the same metric. The answer is that
**the baseline LIF's decision variable is degenerate near its threshold**: ~11 000
sites sit within 1e-6 of it and they are **twenty distinct values replayed ~548
times each**, because layer 0 sees one of 205 embedding rows and the hard reset
zeroes the membrane exactly. The two-compartment neuron's unreset slow pole
destroys that replay — 5 sites, 4 values — and flips **1 342–5 339× fewer**
spikes under an identical perturbation.

**Three of `EXP_012`'s six predictions failed, and two of the failures are at the
resolution floor.** Its own decision rule, applied verbatim, returns a verdict
(*"an operating-point effect, not a neuron effect"*) that §11.4 explains must not
be read as the finding — because the leg that carried the failure compares **0
flips against 1**, which that experiment's limitations section had declared
unresolvable *before* it ran. The rule is **reported as it fired and is not
repaired.** In consequence **§6.5's noise-floor sentence stands exactly as
written** (`EXP_012` §4's trigger did not fire), and the scope qualifier its
evidence supports is **referred, not applied**.

**Rev 4 changes no earlier number, verdict or threshold**, adopts nothing, and
moves none of §8's seven open rows.

**What rev 3 does.** It adds §10: `EXP_011`, the composition of the adopted
two-compartment neuron with `EXP_008`'s learned per-channel threshold. **The arms
compose** — 2.08326 carried against the adopted arm's 2.11869, at −28.7 se, with
0/128 contexts regressed — and the result is **provisional at n = 2 of a
pre-registered 3**, because one seed diverged and was not replaced. Two findings
in it are worth more than the headline: **the composition's entire gain is the
within-reach component**, which confirms `EXP_004` §10.6's overlap argument in
its strong form while relocating where the value is; and **the dead seed did not
die of `EXP_009`'s bug**, which decision #7 would have fixed.

**Rev 3 changes no earlier number, verdict or threshold**, and it adopts nothing.

**What rev 2 did.** It read `EXP_007`, `EXP_008` and `EXP_009` *together* rather
than one after another (§6.7), stated a verdict for every Phase-4 arm against its
own pre-registered rule (§6.8), re-ran every gate in §9 from a clean tree, and
corrected five numbers rev 1 got wrong (§15, renumbered from §12 at rev 6, from
§13 at rev 8 and from §14 at rev 9 — see this changelog's own entries). **No
verdict, threshold or decision
changed.** Two of the corrections make an arm look worse, one makes the adopted
arm look slightly better, and all five restore a committed measurement rather than
redefining anything — the redefinitions on the table are still in §8, still
unapplied.

---

## 1. What Phase 4 is for, and where it stands

`01_reconnaissance.md` §8.2 defines Phase 4 as *pre-registered, single-variable,
≥3-seed controlled experiments* against the ranked candidate list in
`03_phase3_candidates.md` §6.3, with a budget of ~30 GPU-hours.

| | |
|---|---|
| Experiments closed | `EXP_004`, `EXP_005`, `EXP_006`; `EXP_007`, `EXP_008` and `EXP_009` in §6; **`EXP_011` in §10**; **`EXP_012` in §11**; **`EXP_013` in §12**; **`EXP_014` in §13**; **`EXP_015` in §14**; **`EXP_016` in §15**; **`EXP_017` in §16**; **`EXP_018` in §17**; **`EXP_020` in §18**; **`EXP_021` in §19**; **`EXP_022` in §20**; **`EXP_023` in §21**; **`EXP_025` in §22**. **`EXP_019` is pre-registered and NOT RUN** — its zero-GPU instruments produced the regional census committed at `19cdd4a` and its training ladder never started, so it has no section |
| Ranked candidates closed | **6 of 14** (#1, #14, **#11 at rev 8**, the horizon question §10.5 opened, and two of the three I5 referrals run as diagnostics; **#3 itself is resolved at rev 12**). **`EXP_011`, `EXP_012` and `EXP_013` close no ranked candidate** — the first composes two arms already counted here, the second is a mechanism study and trains nothing, the third is not on the 18-item list at all (`EXP_013` §0). **`EXP_014` does close one — #11, the parameter-scaling pilot** — and closing it is the only sense in which it settles anything: it adopts no size and ranks no arm. **`EXP_017` closes none either** — it is a new arm answering `EXP_016` §10's referral, not an item on the 18-item list. **`EXP_018` closes none either** and is not on the 18-item list at all, for the same reason `EXP_013` was not. **`EXP_020` and `EXP_021` close none either** — neither is on the 18-item list: the first measures two ends the ranking never named, the second answers `EXP_018` §10's referral in the slot #3 admits. **`EXP_022` and `EXP_023` close none either**, for the same reason: the first is §18's own follow-up and the second is §19's. **`EXP_025` closes none either** — it is §16's referral carried to a bar, not an item on the 18-item list, and §13 already closed #11 |
| Arms adopted | **2** — the two-compartment neuron (`EXP_004`) and, **at rev 6, the learned per-channel threshold standalone** (`EXP_008`, decision #5). The composed arm (§10) is **not** adopted — provisional at n = 2. **`EXP_014` adopts nothing**: width is not an arm, and n = 1 per rung cannot clear §6.2. **`EXP_015` adopts nothing either, and adds a caveat to both adopted arms**: neither has been shown to train above 735K parameters, and §14 is where both failed to. **`EXP_016` adopts nothing and proposes nothing**; it explains why §14's legs died. **`EXP_017` adopts nothing and is not proposed for adoption**, though it is the first arm this phase has built that trains at 5.0M — and §16.4 puts a caveat on the *adopted* arm rather than lifting one: two of three freshly trained `twocomp` controls diverged at 735K. **`EXP_018` adopts nothing and is the first Phase-4 arm that is worse than its own anchor on every seed** (§17.3). **`EXP_020` adopts nothing and EVERY arm in it lost** (§18.2). **`EXP_021` adopts nothing** — but it is the first arm since `EXP_008` to beat its anchor (−0.03371, 3/3 seeds, §19.1), and §19.2 is why that is not the same as having a mechanism: adopting it would be adopting *a learned rank-1 time-varying gain*, not *a dopamine signal*. **`EXP_022` adopts nothing** and prices closing I1's last exception at 0.00955 bpc for no parameters (§20.1) — whether that is cheap enough is #5's shape and is Elliot's. **`EXP_023` adopts nothing** and narrows §19's arm further still: §21.2's constant rung is worth what the aligned one is worth, and by `EXP_008` Identity 1 that mechanism is one the project has already adopted. **`EXP_025` adopts nothing** — it is the first bar this project has put on the two-compartment advantage at 5.0M (−0.13426, 79.4 se, 3 seeds) and adopting the detached arm is decision #5's shape and Elliot's; §16.2's A6 caveat stands, and this measures `twocomp_detach` rather than the adopted `twocomp` |
| GPU-hours spent | **~29.7 of ~30** (`EXP_012` cost ~0.2 and trained nothing; `EXP_013` cost ~1.9; **`EXP_014` cost ~1.5**; **`EXP_015` cost ~2.1** — 1.79 of ladder training (§14.1), ~0.09 of arch calibration, ~0.08 for the divergence chase, and ~0.1 for the evaluations, horizon, gap and state probes. **~0.7 of `EXP_015`'s training bought no number**: two legs ran 20,000 steps of which ~14,750 and ~18,000 computed NaN, and that is recorded as spent rather than netted off. **`EXP_016` cost ~0.8 and trained nothing** — five forward passes over committed checkpoints and two instrumented replays of one step — of which **~0.37 bought no number**, two aborted Leg B attempts whose defects are recorded in §15.5's source log. **`EXP_017` cost ~2.0** — 1.60 of training across seven runs, ~0.25 of probes and the horizon, and ~0.12 for running the mutation campaign **twice**, which is recorded as spent because the first run is what caught D09); **`EXP_018` cost ~2.9** — 2.63 of ladder training across sixteen runs, none of which diverged, plus ~0.3 of calibration, the systems probe, the §7.2 screen and the horizon sweep); **`EXP_020` cost ~1.9** — 16 runs at 6,658 s, none diverged, plus the frozen-feature probe, the §7.2 screen, the cost pre-flight and a 16-checkpoint horizon sweep; **`EXP_021` cost ~2.0** — 15 runs at 7,125 s, none diverged, plus the `tau` calibration, the screen, the pre-flight, an 18-checkpoint horizon sweep and 12 generalisation-gap evaluations. **Nothing in either bought no number.** 31 runs across the two, **zero divergences**; **`EXP_022` cost ~1.9** — 15 runs at 6,246 s, none diverged, plus a 21-checkpoint horizon sweep and 15 generalisation-gap evaluations; **`EXP_023` cost ~2.1** — 15 runs at 7,177 s, none diverged, plus a 24-checkpoint horizon sweep and 12 gap evaluations. **61 runs across the four, zero divergences, and nothing that bought no number**; **`EXP_025` cost ~5.17** — 4.858 of ladder across 12 runs, none diverged, plus a 12-checkpoint horizon sweep and the six-cell cost pre-flight, **against ~5.1 projected, the first Phase-4 estimate right to 0.4 %**. **73 runs across the five, zero divergences.** **The ~30 GPU-h budget is now essentially spent, and that is itself for Elliot** |
| Unexplained divergences | **4, sharing 2 causes** — `twocomp_distill_s1` (§6.4) has its own; `compose_s0` (§10.4) and **both of `EXP_015`'s dead legs (§14.5)** share a signature: origin in layer 0's backward, forward and loss finite, fp32 and fp64 gradient norms equal, and the eager path producing an identical NaN in identical tensors. **It is now reproducible 138 steps from a committed checkpoint rather than 17,598**, and it blocks scaling the adopted arm. **At rev 10 the shared signature has a mechanism (§15)**: the first non-finite value is made *inside* layer 0's reverse recursion, on both dispatch paths, after an amplification of **10^41.62 within one backward pass** — and the plain LIF's recurrence is provably incapable of it. **At rev 11 there are 6, sharing 2 causes**: two *freshly trained* `twocomp` seeds at `d = 512` (§16.4) joined the shared signature, at the width the arm was adopted at, where the committed record holds seven clean seeds |
| Phase-3 §9 sign-off | **signed off 2026-08-05** (decision #1), and **recorded as retrospective**: Phase 4 was opened without it on instruction and eight experiments ran before it |
| Decisions open | **6 of 11** (§8). #8 resolved at rev 3, #1 at rev 5, #5 and #7 at rev 6; rev 6 adds #9, rev 8 adds #10 and **rev 9 adds #11**, all three open. **Rev 11 adds none and edits none**, and sets out beneath the §8 table the case for a twelfth it deliberately did not add. **Rev 12 adds none and RESOLVES #3** — `EXP_018` handed it a fourth boundary case (a rank-1 broadcast from the head) and it was ruled on Elliot's instruction: **feedforward-driven modulators admitted, head-driven ones diagnostic-only**, extended in `01_reconnaissance.md` §4.6. **Rev 13 adds none and edits none**, and records fresh measured evidence for **#2** and a second independent measurement bearing on **#6** in prose beneath the table. **Rev 14 adds none and edits none either**, and records beneath the table a third measurement bearing on **#6**, a third instance of **#9**'s defect class, and a protocol deviation in §18's and §19's entry conditions. **Rev 15 adds none and edits none either.** It corrects rev 14's own note on the pre-registration audit — the record is **7/9, not 2/12** — and three things in `EXP_024`'s pre-registration, which is **not edited**. It bears on **#4** only by improving the record that decision will be taken on, and **takes no position on it**. **Rev 16 adds none and edits none either**, and `EXP_025` **engaged none of the six**: gate **G5** read the frozen recipe back out of all twelve `config.json`s, including all six at 5.0M, so **#11 was never engaged — mechanically rather than by assertion** — at the exact width where the adopted `twocomp` arm diverges |

The phase has moved the project's answer to its own question twice, and both
moves were away from the thing the ranking was built around.

---

## 2. The one adopted arm, and the shape of its gain

`EXP_004` adopted the two-compartment neuron — a fast pole at the baseline's
`beta = 0.5` and a per-channel learned slow pole, reset-shielded, mixed additively.
`EXP_005` carried it to seven seeds.

| | baseline (n=5) | **two-compartment (n=7)** | GRU anchor (n=3) |
|---|---:|---:|---:|
| test bpc, carried | 2.25311 | **2.11869** | 1.76741 |
| test bpc, fresh | 2.26969 | **2.14566** | 1.80443 |
| whole-split sd | 0.00461 | **0.00313** | 0.00223 |
| memory horizon (2σ) | 7 (5/5 seeds) | **47** (median of 7; 38–57) | 57–60 |

**42.9 σ at the family's own bar.** The result is not in question. What the phase
spent its remaining effort on is the fact that the gain is not uniform:

| Component of the gain | bpc | share |
|---|---:|---:|
| At zero context | +0.0506 | **+41.0 %** |
| Within the baseline's own 7-character reach | **−0.0598** | **−48.4 %** |
| Beyond the horizon | +0.1327 | +107.4 % |
| **Total** | **+0.1235** | 100 % |

Quoted as one number this reads as a uniform win while nearly half its magnitude
is handed back inside the seven characters the baseline already reached. §6.2's
criterion caught it: the arm is worse than the baseline at 5 of 128 context
lengths and they are contiguous — c = 3, 4, 5, 6, 7 — worst at c = 4 where it is
6.5× that context's own noise bar.

**The composition of what is left changed more than its size.**

| Component of the remaining gap to the anchor | baseline | **two-compartment** |
|---|---:|---:|
| At zero context | 0.1193 (25.8 %) | 0.0687 (20.2 %) |
| **Within the 7-character reach** | 0.1161 (25.0 %) | **0.1759 (51.8 %)** |
| Beyond the horizon | 0.2279 (49.2 %) | 0.0952 (28.0 %) |
| **Total** | **0.4633** | **0.3398** |

The candidate recovered **58.2 %** of the component it was ranked to attack
(+0.1327 of the 0.2279 available beyond the horizon) and made the largest
remaining component larger. **Short-range fidelity, not horizon, is where
the gap now lives** — and until `EXP_007` no arm this project has run had ever
been aimed at it.

---

## 3. σ is not a project constant

`EXP_005` re-measured the noise floor on the adopted neuron because every 2σ
verdict in the project assumed one measured at a single configuration.

* σ = **0.00313** on the two-compartment arm against **0.00461** on the baseline —
  0.680×. The failure is in the safe direction, and n = 7 cannot *prove* the
  difference: F(6,4) resolves only outside [0.16, 9.20]. **Candidate #14 is closed
  for this neuron and not in general.** Any structurally different arm needs its
  own σ, and `EXP_007`/`EXP_008` report theirs rather than inheriting.
* The §6.2 per-context bars are **arm-specific in shape, not only in scale**: the
  baseline's excess-sd collapses past its 7-character horizon while the
  two-compartment arm's persists to c = 127 and is 21× larger at c = 16.
* The bar in use is **built on the wrong scale** — 2× the baseline's per-seed sd
  of the excess curve, compared against a difference of two arm means, and 45×
  too small at c = 127. The count of significantly-worse contexts is unchanged at
  5/128 under all three constructions. **A change to §6.2's definition is referred
  to Elliot in §8 and is not applied here.**

---

## 4. `EXP_006` — three per cent of the channels carry the horizon, and the bits with it

This is the experiment `EXP_004` §10.11 called "the highest-information single run
available", and it is the one that most changes what Phase 5 should do.

### 4.1 What was measured

`beta_s` was initialised at 0.95 on all 512 channels and went **bimodal**: ~3 % of
channels per layer kept `beta_s > 0.9` (time constants 18–26 characters) and the
other 97 % pulled their "slow" pole down to ~0.5–0.6. `EXP_004` §10.5 observed
that and could not act on it: the observation was correlational, and it said
nothing about bits.

`EXP_006` intervened, on the seven checkpoints already on disk — 91 inference-only
configurations, ~0.45 GPU-hours, no training. **The intervention writes to
`beta_s_raw` and to nothing else**, because zeroing the mix `w` would have removed
a channel's long memory *and* its learned threshold at once (§10.6's algebra), and
an ablation that moves both cannot answer a question about either.

### 4.2 The dose–response curve

Clamping the top `N` channels per layer to that layer's median `beta_s`:

| N clamped | % of width | Horizon | Fresh bpc | % of the arm's gain gone |
|---:|---:|---:|---:|---:|
| 0 *(intact)* | 0 % | **47** | 2.14566 | 0 % |
| 4 | 0.8 % | 31 | 2.17763 | 25.8 % |
| 8 | 1.6 % | 22 | 2.21652 | 57.1 % |
| **16** | **3.1 %** | **12** | **2.26340** | **94.9 %** |
| 32 | 6.3 % | 8 | 2.32203 | 142.2 % *(worse than the baseline)* |
| 512 | 100 % | 6 | 2.44868 | 244.3 % |

**Clamping 3.1 % of the width takes the horizon from 47 to 12 and removes 94.9 %
of the arm's bits-per-character gain.** `N½` — the smallest swept `N` at which
half the horizon gain is gone — is **8**, or 1.6 % of the width.

### 4.3 The dissociation, which is what makes it causal

Four interventions of identical size, 16 channels per layer:

| Intervention | Horizon | % horizon gone |
|---|---:|---:|
| clamp the top 16 by `beta_s` | **12** | **87.5 %** |
| clamp everything **except** the top 16 by `beta_s` | **45** | 5.0 % |
| clamp the top 16 by `\|w\|` | 47 | 0.0 % |
| 16 random non-tail channels at **matched displacement** | 48 | −2.5 % |

Only one of the four touches the horizon. The two selection rules pick **disjoint
sets** — the overlap between the top 16 by `beta_s` and the top 16 by `|w|` is
0, 0, 1, 0, 2, 1, 1 channels of 16 across the seven seeds.

The matched-displacement control also prices the null intervention: clamping 16
arbitrary channels by the same amount costs 16.6 % of the gain, so of the 94.9 %,
about **78 points are specific to the tail's timescale**. The horizon result needs
no such correction — the control removes none of it.

All six pre-registered predictions held and **280 of 280 self-checks passed**,
including one that is exact: because nothing at zero context can depend on
`beta_s` (`vs_0 = 0·beta_s + cur_0`), the probe's k = 1 point must be bit-identical
under every intervention. Across all 91 configurations there are exactly seven
distinct zero-context bpc values, one per seed, **nats identical to the bit**.

### 4.4 The slow pole is worth 0.303 bpc, and is harmful without its tail

`A(N)` and `B(N)` clamp complementary channel sets, so their costs should sum to
`A(512)`'s. They do, to 0.0009 bpc on a 0.303 bpc quantity — an identity nothing
in the design required and no prediction rests on.

It reframes the numerator. **`A(512)`, with every slow pole flattened to the
population median, is 0.179 bpc *worse than the Phase-2 baseline*.** A second
compartment running at ~0.55 is not a neutral addition that the tail improves on;
it is actively harmful, and the tail pays for it and then some. Of the 0.303, the
top 16 channels carry 39 % and the other 496 carry 61 %.

### 4.5 What it does **not** establish — and this is the part that changes Phase 5

`EXP_004` §10.11 item 4 read: *"~3 % of channels appear to carry the horizon,
which if true means the long-memory capacity is far from saturated."*

**The first half is now established causally. The second does not follow, and
`EXP_006` supplies an argument against it.** `beta_s` began at 0.95 on all 512
channels: the network started with the maximum available slow capacity and
**pulled 97 % of it down**. The tail's size is a learned outcome of a network that
had 512 slow channels and declined them, not a ceiling it ran into.

That is an argument, not a measurement, and two things could defeat it — the
descent could be an optimisation artifact rather than a preference, and nothing
rules out a different count being better under a different initialisation. **The
experiment that would settle it is the frozen-`beta_s` ladder named in `EXP_006`
§9.6: freeze or regularise `beta_s` toward 0.95 on N channels per layer, train at
N ∈ {16, 64, 256}, read the bpc. Three seeds, ~1.3 GPU-hours.**

**Until that runs, "widen the slow tail" must not be ranked on this result.** The
correct statement is the narrow one: *~3 % of channels carry the horizon and ~95 %
of the arm's gain; whether more of them would help is unmeasured, and the
initialisation history is evidence that it would not.*

### 4.6 Why six-for-six should attract suspicion rather than satisfaction

`EXP_004` falsified T2 and `EXP_005` failed three of five, so a clean sweep is the
pattern that deserves scrutiny. What carries the credibility here is not the
count: it is that the four interventions are independent selection rules of
identical size whose results dissociate, over a harness that proved its own null
on 91 configurations — 84 of them bit-exactly — before any of them were read.

One control is weaker than its verdict suggests, and is recorded as such:
`C(64)`'s partial horizon loss (25 %) is consistent with the `|w|` ranking
reaching into the tail as N grows, but the overlap was only measured at N = 16.

---

## 5. What §4 changes about the plan

1. **The horizon question is closed for this neuron.** It is carried by ~3 % of
   channels, it is causal, and the bits follow it. There is nothing left to
   establish about *whether* the mechanism works.
2. **The obvious follow-up is blocked, not opened.** "Widen the slow tail" was the
   ranking input §10.11 item 4 was being used to justify, and §4.5 withdraws it
   pending a 1.3 GPU-hour ladder.
3. **The remaining gap is 51.8 % within-reach and only 28 % beyond the horizon.**
   The Phase-3 ranking, which put beyond-horizon candidates first, was right for
   Phase 4 and is wrong for what follows.
4. **So the next arms have to point somewhere else** — at short-range fidelity, at
   the zero-context component, or at the training signal. That is exactly what
   `EXP_007`, `EXP_008` and `EXP_009` do, and §6 reports them.

---

## 6. `EXP_007`, `EXP_008`, `EXP_009` — three arms aimed away from the horizon

Pre-registered together at `1f05bab`, before the harness (`5d6f88a`) and before
the resolver (`f9c509c`); every threshold, every self-check and every
direction-of-statement was fixed then. Nine training runs, **~1.6 GPU-hours**.

### 6.1 The scoreboard

| arm | n | test bpc, carried | Δ vs its reference | se | wall-clock | peak VRAM |
|---|---:|---:|---:|---:|---:|---:|
| Phase-2 baseline | 5 | 2.25311 | — | — | 1.00× | 0.609 GiB |
| **token-shift** *(diagnostic)* | 3 | **2.21444 ± 0.00103** | **−0.03867** | −18.0 | 1.49× | 0.859 GiB |
| **learned threshold** | 3 | **2.19416 ± 0.00244** | **−0.05896** | −23.6 | **1.12×** | 0.734 GiB |
| two-compartment *(adopted)* | 7 | 2.11869 | −0.13442 | — | 1.32× | 0.980 GiB |
| **distilled twocomp** *(diagnostic)* | **2** | **2.10921 ± 0.00190** | −0.00948 | −5.3 | 2.13× | 1.10 GiB |
| GRU anchor *(violates I5)* | 3 | 1.76741 | — | — | 1.11× | 0.864 GiB |

**The wall-clock column is against `snn_beta0.5_s0`'s 398.9 s** — `EXP_004`'s
convention and the run manifest's `reference_wall_clock_s` — and every arm's entry
is the mean over the seeds that were *scored*. Rev 1 had two entries that were not:
the adopted arm at 1.35× was `twocomp_s0` alone against `EXP_004`'s committed
three-seed **1.32×** (all seven seeds give 1.30×), and the distilled arm's 1.58×
was measured against the two-compartment arm — correct in `EXP_009` §9.1, wrong in
a column whose reference row is the baseline, where it is **2.13×**. §15 records
both (renumbered from §12 at rev 6, §13 at rev 8 and §14 at rev 9).

Both new arms clear the adoption bar — token-shift by **4.2×** and the threshold
arm by **6.4×** (8.4σ and 12.8σ at the inherited noise floor; rev 1 said "an order
of magnitude", which the numbers do not support) — and **neither regresses a
single one of the 128 context lengths**, which no previous Phase-4 arm has
managed.

### 6.2 Where each arm's gain lives, which is the finding

Decomposed on `EXP_004` §10.3's basis, cut at the baseline's horizon of 7.
Positive = better.

| arm | zero context | within the 7-char reach | beyond the horizon |
|---|---:|---:|---:|
| two-compartment | +0.0506 (41.0 %) | **−0.0598 (−48.4 %)** | +0.1327 (107.4 %) |
| **token-shift** | **+0.0493 (101.8 %)** | −0.0065 (−13.4 %) | +0.0057 (11.7 %) |
| **learned threshold** | +0.0291 (49.7 %) | **+0.0288 (49.1 %)** | +0.0007 (1.2 %) |
| **distilled** *(vs twocomp)* | +0.0243 (226 %) | **−0.0131 (−122 %)** | −0.0004 (−4 %) |

**Three of the four gains are dominated by the zero-context component, and the
project has now measured that component four independent ways.** That is the
strongest pattern in Phase 4.

**The token-shift's result is the one that explains the others.** At the first
character of a window `cur_{−1} = 0`, so `cur'_0 = mu_c·cur_0` — the two-tap
filter **degenerates to a per-channel gain**, which is precisely `EXP_004`
§10.6's learned threshold and precisely what `EXP_008` implements. The arm ranked
to attack short-range memory delivered 102 % of its gain through the *other*
arm's mechanism, and −0.0065 through its own. `EXP_007` §9.2 has the algebra.

**And the arm nobody aimed at short range is the only one that moved it.** The
learned threshold recovers **+0.0288 bpc within the baseline's own reach — 24.8 %
of the 0.1161 available there** — where the two-compartment neuron gave back
0.0598 and the token-shift gave back 0.0065. A static per-channel threshold is
not a zero-context trick: it changes when every channel fires, and therefore when
it resets, at every context length.

### 6.3 The threshold arm buys 44 % of the adopted arm's gain for nothing

| | two-compartment | **learned threshold** |
|---|---|---|
| test bpc gained | 0.1344 | **0.0590 (43.9 % of it)** |
| new state variable | yes | **no** |
| new CUDA kernel + hand-written backward | yes | **no** |
| new R10 gate + mutation campaign | yes (22 mutations) | **no** |
| parameters | +2 048 (+0.28 %) | +1 024 (+0.14 %), **and redundant** |
| wall-clock | 1.32× | **1.12×** |
| contexts regressed (§6.2 criterion) | 5 / 128 | **0 / 128** |
| cost at inference after folding | — | **zero: it folds into `W`** |

**`EXP_008` §1.1's Identity 2 is confirmed to 6.8–8.3 units in the last place of
float64** (`EXP_008` §9.5 quotes the ≈7 of its first two seeds; seed 2's leg C is
8.3 eps): `g·(W·h + b)` and `(g·W)·h + g·b` are the same function, so the arm's
1 024 parameters span no new functions and fold into the projection they multiply.
The gain is therefore an **optimisation effect** — a different trajectory under
AdamW, not a larger reachable set — and it can be shipped at the baseline's exact
parameter count and cost.

That also settles a question `EXP_004` §10.6 left open. It offered two live
explanations for the two-compartment arm's zero-context gain — the learned
threshold, or the +0.28 % parameters — and could not separate them. **Identity 2
collapses them: neither is capacity.** The baseline could always have represented
the threshold by scaling its own rows; what it could not do was *find* it.

*(Rev 6: this is "neither of **these** is capacity" — `EXP_013` §2.2 measures a
different, complementary sense at §12.2: the model as a whole is not
capacity-constrained in the overfitting sense, so nothing here should be read as
evidence that added capacity would go to waste. None of Phase 4's arms adding no
function class is not the same claim as the model having no room to grow.)*

### 6.4 Distillation closes 3.2 % of the gap, and one seed died proving something else

`EXP_009` was the diagnostic that would say whether the remaining 0.3398 bpc is
representational or optimisational. It gives a **lower bound of 3.2 %** at 5.3 se
— real, and far under the 25 % that would have changed Phase 5's shape.

**§1.1 of that file forbids the tempting inversion.** X2 failing does *not* show
the gap is representational: λ, temperature, schedule and teacher were all unswept
by design, so a null is a null *of the recipe*. Only the positive part is
load-bearing, and the positive part is 3.2 %.

**The more transferable result is the failure.** `twocomp_distill_s1` diverged at
step 12 497 and `009_chase_divergence.py` reproduces it deterministically:

* the **forward never overflowed** — logits max 38.9, loss a healthy 1.5047;
* the first symptom is **not a NaN**: every gradient was finite, the largest at
  **5.46e31**, and `clip_grad_norm_`'s fp32 sum-of-squares overflowed, so the clip
  reported `inf` and *zeroed* the update instead of rescaling it;
* one step later, the NaN appears in exactly `embed.weight`, `layers.0.*`, `w.0`
  and `beta_s_raw.0` — layer 0 and below only — so it was **generated inside layer
  0's two-compartment backward**, not inherited from above.

So the **adopted** neuron has a BPTT gradient-explosion mode that the project's
gradient clip cannot see, because the clip's own accumulator overflows first.
Seven un-distilled seeds never hit it, so nothing here shows the committed
configuration is unstable — but the clip's blind spot is real and is not a
property of distillation. Two one-line changes suggest themselves (accumulate the
clip norm in fp64; abort or skip on a non-finite norm rather than silently zeroing
it) and **neither is applied**: both touch committed Phase-2 code and both would
flatter the work. §8 refers them.

### 6.5 A noise floor the project did not know it had

Chasing `EXP_008`'s failed W1 produced a number worth more than the prediction:

> **This architecture's reported bpc is reproducible to only ~2e-3 across
> mathematically-equivalent reparameterisations of its own weights.**

A one-ulp change in the current — from folding `g` into `W` and letting the GEMM
accumulate in a different order — flips 1.6–3.2e-4 of layer 0's spikes, which the
next layer amplifies to 2.5–4.6e-3, until the logits differ by up to 27 on a scale
of 264 and the bpc by up to 0.002.

This does **not** contradict `tests/test_determinism.py`: for fixed weights and a
fixed code path, evaluation stays bit-exact. It applies exactly when weights are
rewritten into an equivalent form — folding, quantisation, weight normalisation.
**The adoption bar 2σ = 0.00922 is only 4.6× this floor**, and the §6.2
per-context bars at c ≥ 16 (≈ 0.0003) are an order of magnitude *below* it. Any
future claim that rests on a reparameterisation needs this floor beside it rather
than the seed σ.

> **Rev 4 — a scope qualifier is REFERRED here and deliberately not applied.**
> The sentence above says *"this architecture"*. `EXP_012` (§11) measured the same
> fold on the **adopted** neuron and got **1.4e-05 / 9.3e-06** on the same
> 512-window metric that gives 2.0e-03 / 6.2e-04 / 1.6e-03 here — and traced the
> difference to a degeneracy that exists in the baseline LIF's decision variable
> and not in the two-compartment one (§11.3, §11.6). On that evidence the floor
> looks like a property of **the neuron it was measured on**, not of the
> architecture family, which would be `EXP_005`'s σ-non-transfer lesson in a new
> place.
>
> **It is not applied, for a reason that is not a formality.** `EXP_012` §4
> pre-registered exactly this correction behind a trigger — `Y2 ∧ Y5` — and
> **Y5 failed** (§11.4). Applying the qualifier anyway would be adopting a
> correction whose own pre-registered gate did not open, immediately after
> watching it not open. **§6.5 therefore stands as written**, and the qualifier is
> §11.7's referral (`EXP_012` §9.7 item 9), for Elliot.

> **Rev 6 — a fourth data point on floor-vs-signal, and it is the one this section
> warned would be needed.** `EXP_013` §2.2's whole generalisation-gap measurement
> is **+0.00594 bpc (fresh)**, only ~3× this ~2e-3 floor, and **−0.00180 bpc
> (carried)** sits *below* it entirely — indistinguishable from the floor itself.
> `EXP_013` §5 item 7 and §9.2 already carry this guard internally (any `Δgap`
> under the floor is reported as "below what this instrument resolves," not as a
> signed number — see §12.2/§12.4). This is the first time a headline figure this
> project computed, rather than a diagnostic residual, has landed *inside* the
> floor rather than merely near it; nothing here is a correction to §6.5's own
> claim, which is about reparameterisation residuals specifically.

### 6.6 Predictions resolved, including the ones that failed

| experiment | held | failed | notes |
|---|---|---|---|
| `EXP_007` | V1, V3, V5, V6 | **V2** | V4 held **vacuously** and is not counted as a hit — it bounded the mechanism from above and the mechanism went the other way |
| `EXP_008` | W2, W4, W5, W6 | **W1, W3** | W3 missed by 0.3 points (49.7 % vs 50 %) and is reported as failed. W1's chase confirmed the algebra and indicted the tolerance |
| `EXP_009` | X1, X3, X5 | **X2, X4** | provisional at n = 2 |

Five of seventeen predictions failed and two more are qualified. After `EXP_006`'s
six-for-six — which §4.6 flagged as a pattern deserving suspicion — that is a more
normal distribution for this project, and the three that were **stated against the
plan** (`EXP_008`'s W5 and W6, `EXP_009`'s X5) all held.

### 6.7 The three read together, which is not how they were designed

The three experiments were pointed at three different things: short-range fidelity
(`EXP_007`), the zero-context component (`EXP_008`), and the training signal
(`EXP_009`). **All three landed on the same component, and for two of them it is
provably the same mechanism.**

| arm | aimed at | where the gain landed | what the mechanism reduces to at c = 0 |
|---|---|---|---|
| token-shift | within-reach | **101.8 % zero-context** | `cur'_0 = mu_c·cur_0` — a per-channel gain |
| learned threshold | zero-context | 49.7 % zero, 49.1 % within | `cur' = exp(−θ_c)·cur` — a per-channel gain |
| distilled twocomp | the training signal | **226 % zero-context** | the teacher's marginals, no mechanism claimed |

The first two rows are **the same mechanism arrived at from opposite directions**,
and `EXP_007` §9.2 has the line of algebra that says so: at the first character of
a window the shift register is zero, so the two-tap filter *is* a per-channel gain
on the current — which is what `EXP_008` implements deliberately and what
`EXP_004` §10.6 found implicitly inside the adopted neuron. Counting the adopted
arm, **four arms have now measured one quantity**, and they disagree about its
size: the two-compartment neuron reaches an effective 0.69/0.79, token-shift's
`mu` settles at 0.712/1.147, the explicit parameter goes to **0.342/0.428**.

**Three consequences, and the third is the one that should reorganise Phase 5.**

**1. Only the arm that added the least structure moved the component the project
cares about.** Against the 0.1161 bpc available within the baseline's own reach:

| arm | what it added | within-reach recovered |
|---|---|---:|
| two-compartment | a second state variable, a kernel, a backward, a gate | **−51.6 %** |
| token-shift | a two-tap FIR and four `[B,L,d]` intermediates per layer | −5.6 % |
| **learned threshold** | **one redundant scalar per channel** | **+24.8 %** |

The ordering is inverse to the amount of machinery, and nothing in the project
predicted it. §7 item 2 keeps *why* as an open question rather than answering it.

**2. Token-shift is dominated, which drains the stake out of decision #3.** The two
arms were not run head to head, so this is a descriptive contrast against a common
baseline and neither file pre-registered it — but it is not close: the threshold
arm is **0.0203 bpc better at 13.3 se**, at 1.12× wall-clock against 1.49×, 0.734
GiB against 0.859, with no new intermediates, and it folds away at inference.
Token-shift is worse on every measured axis and its one distinguishing feature —
the second tap — is worth **−0.0009 bpc**: the two components where the tap is
live (within-reach −0.0065, beyond-horizon +0.0057) very nearly cancel. **Ruling
token-shift inside or outside I5 therefore changes nothing about what to build.**
The ruling is still Elliot's and §8 still carries it; what §6.7 removes is the cost
of getting it wrong.

**3. Phase 4's bits are being won by optimisation, and the ranked list is entirely
architectural.** Take the three gains in turn against the function class each arm
can reach:

* the **threshold** arm adds no functions at all — Identity 2, measured (§6.3);
* the **distilled** arm is the adopted architecture, unchanged, trained against a
  different signal;
* **token-shift** genuinely does add functions — a two-tap FIR over time is not in
  the baseline's class — and the parts of the curve where those functions are live
  are worth the −0.0009 bpc above. Its 0.0387 is bought at c = 0 by the part that
  is a reparameterisation.

**So essentially none of the 0.0387 / 0.0590 / 0.0095 bpc these three arms bought
is attributable to a larger function class.** Together with `EXP_009`'s
constructive lower bound — ≥ 3.2 % of the remaining gap is reachable without any
architectural change — that is Phase 4's second-strongest pattern, and
`03_phase3_candidates.md` §6.3 ranks fourteen architectural candidates and no
optimisation ones.

*(Rev 6: this item is what `EXP_013` was run to answer — its own §0 item 2 quotes
this exact paragraph as its justification for existing. §12 reports a null: the
one training-only arm tried costs bits at every amplitude and closes no gap.
That does not fill the gap this item names — a null does not — but the
calibration `EXP_013` measured on the way there does something more useful for
Phase 5's own ranking: it removes the standing objection to the
parameter-scaling pilot (§7.1 rank 2, newly added at rev 6), which had no
evidence there was room to grow until now.)*

**What reading them together does not license.** The three arms act on one
component, so **their numbers may not be added**, and neither may any of them be
added to the adopted arm's (§7 item 4). Each is n = 3 against the baseline's n = 5
— n = 2 for the distilled arm — no arm was re-run, and each reports its own σ. And
`EXP_009`'s divergence is not a property of distillation: it is in the **adopted**
neuron's backward (§6.4), so it survives every decision about these three arms.

**One cross-log correction.** `EXP_007` §9.6 item 2 closes with "the within-reach
component remains unattacked by any arm this project has run". `EXP_008`, closed
the same day, moved it by +0.0288. Both logs are closed and neither is rewritten;
the joint claim is the one in §6.2 and in item 1 above.

### 6.8 The verdict on every Phase-4 arm, against its own rule

Each arm is resolved only against the thresholds its own pre-registration fixed.
`scripts/exp/010_phase4_arm_results.py` applies them mechanically and was
committed at `f9c509c`, before any of these numbers were read.

| arm | pre-registered cell reached | verdict as written | status now |
|---|---|---|---|
| **two-compartment** `EXP_004`/`005`/`006` | T-rules; 42.9σ at the family's bar | **adopted in Phase 4** | **the only adopted arm.** Unchanged by rev 2. Carries a gradient-explosion mode the clip cannot see → decision #7 |
| **token-shift** `EXP_007` | **V1 holds, V2 fails** | "It buys bits, but not the ones this experiment aimed at. The within-reach claim is **not made**." | **null for its purpose, retained.** −0.0387 bpc is real; the arm is dominated (§6.7 item 2). **Not recommended even if I5 admits it** |
| **learned threshold** `EXP_008` | **W1 fails** → "stop and chase it" | chased (`008_chase_fold.py`): the algebra is right to 6.8–8.3 fp64 eps, the **bar** was borrowed from the wrong precedent. W2 held at 23.6 se | **recommended, not adopted** — decision #5, and coupled to #6 because the recommendation rests on a check that **failed as written** |
| **distilled twocomp** `EXP_009` | **X1 holds, X2 fails**, provisional at n = 2 | "Some of the gap is optimisational; less than a quarter of it by this route. The architecture ranking stands." | **diagnostic, never a headline** (`EXP_009` §4's rider). Not an arm to adopt — it is a training recipe, and the recipe was one unswept point |

**Nothing here is adopted by this report.** `EXP_008` §4's rider is explicit that
the adoption is recommended *to Elliot* and that Phase 3's §9 is still unticked;
§8 items 1 and 5 are where that lives. *(Rev 5: **§9 is now ticked** — #1 is
signed off — so the second half of that rider is discharged and #5 is ungated by
#1. **Nothing here is adopted by this report** is unchanged: #5 is Elliot's and is
still undecided, and this section's verdicts are untouched.)*

**And the threshold arm's headline is quoted under a caveat, every time.** Its
decision cell is `W1 fails`, whose text is "no bpc verdict is issued until the
residual is explained". The residual was explained and the tolerance was **not**
widened, so what is reported is: W1 failed as written; Identity 2 holds; the
question of what tolerance a fold-in should carry is #6 and is open. An experiment
does not get to redefine its own threshold after watching it fail.

---

## 7. What Phase 5 should be aimed at now

Phase 4 has moved the target twice, and §6 moves it a third time.

1. **The zero-context component is the one that keeps paying, and it is cheap.**
   Four arms have now attacked it and all four succeeded; the cheapest of them
   (`EXP_008`) delivers 44 % of the adopted arm's whole gain for one foldable
   parameter per channel. **Nothing in the ranked list was pointed here.**
2. **The within-reach component — 51.8 % of the remaining gap — is still barely
   attacked.** The arm designed for it moved it by −0.0065; the arm not designed
   for it moved it by +0.0288, which is 24.8 % of what is there. What made the
   difference is not understood, and understanding it is worth more than another
   arm.
3. **The horizon question is closed and its follow-up is blocked** (§4.5). Do not
   rank tail-widening without the frozen-`beta_s` ladder.
4. **Composition is now the biggest unmeasured thing in the project.** The
   adopted arm and the threshold arm both act on the zero-context component, and
   `EXP_004` §10.11 item 2 warned in advance that "in this arm it is already
   present and would not add twice". The two-compartment neuron reaches an
   effective threshold of 0.69/0.79; the explicit arm reaches 0.342/0.428. **The
   numbers may not be added.**
5. **The phase's bits came from optimisation and the candidate list is
   architectural** (§6.7 item 3). Three arms, essentially none of whose gain is
   attributable to a larger function class, and a constructive lower bound that
   ≥ 3.2 % of the remaining gap needs no architecture at all — against a ranked
   list of fourteen architectural candidates and zero optimisation ones. Phase 5
   should carry at least one arm that changes only how the model is trained.
   *(Rev 6: tried, once — `EXP_013`, §12. Null result, transferable calibration:
   this model is not overfitting, which is item 6 below.)*
6. **Added at rev 6. The model has room to grow, and nothing has tested it.**
   `EXP_013` §2.2 measured this model's generalisation gap directly for the first
   time in the project's history: +0.00594 bpc (fresh, 4.96 se from zero but below
   the 2σ adoption bar) and −0.00180 bpc (carried, indistinguishable from zero).
   735,437 parameters against ~655M training characters is ~7.3 epochs — an
   underfitting regime, not an overfitting one. This is a *referral*, not a
   ranking made here (`EXP_013` §10 item 4 refers it the same way): it removes the
   standing objection to `03_phase3_candidates.md` §6.3 #11 (the parameter-scaling
   pilot) and argues against ranking further regularisation-flavoured candidates
   ahead of capacity-adding ones.

### 7.1 A revised ranking, offered and not applied

`03_phase3_candidates.md` §6.3's list is a Phase-3 artifact and is not edited
here. This is what Phase 4's evidence implies about it, for Elliot to accept or
reject:

| rank | arm | why | GPU-h |
|---:|---|---|---:|
| ~~1~~ | ~~threshold × two-compartment composition~~ | **DONE at rev 3 — this is `EXP_011` (§10).** Kept in the table struck through rather than deleted, so the ranking still reads as the list rev 2 offered. The arms compose; the gain is within-reach, not zero-context | ~~0.5~~ → 0.86 actual |
| ~~2~~ | ~~#11 parameter-scaling pilot~~ (0.735M / 2.5M / 5M params, 1 seed each) | **DONE at rev 8 — this is `EXP_014` (§13).** Kept struck through rather than deleted, on rank 1's precedent, so the table still reads as the ranking rev 6 offered. Added at rev 6 as "newly the best-motivated item here", on `EXP_013` §2.2's no-overfitting evidence. **It bought −0.25238 bpc — more than every arm in this phase — and it did not tell Phase 5 what size to run**, because it ran on the wrong neuron (§13.6) | ~~0.35~~ → **~1.5 actual** |
| **3** | **#2 current normalisation (RMSNorm on `cur`)** | the real §6.3 #2, passes the Phase-3 §7.2 screen, attacks the same components by a different route, still unrun | 0.4 |
| **4** | **why the threshold arm helps within reach** | a mechanism study, not an arm. The only measurement pointed at the largest remaining component | 0.3 |
| **5** | **frozen-`beta_s` ladder**, N ∈ {16, 64, 256} | the only thing that can license tail-widening (§4.5) | 1.3 |
| **6** | #13 training budget (40k steps) | the cheapest pure-optimisation lever, unmeasured, and — per `EXP_013` §2.2 — no longer competing against a regularisation reading of the same evidence | 0.7 |
| — | #5 learned per-channel decay | partly subsumed by the adopted arm | — |
| — | #6 adaptive threshold | **blocked**: both its inits are saddles, and its fallback needs #8 first (`03_phase3_candidates.md` §7.3) | — |
| — | #10 rotational membrane | screened init is a saddle; θ = π/8 fallback passes but the mechanism is unmotivated by any Phase-4 finding | — |

**Demoted on evidence, not on taste:** every beyond-horizon candidate. That
component is 28 % of what is left, the adopted arm already took 58.2 % of it, and
`EXP_006` withdrew the argument that more is available.

> **Rev 8: rank 2 is done, and its result argues that the rest of this table is
> ranked against the wrong axis. Offered, and — as with every revision of this
> table — not applied.** `EXP_014` (§13) is the first row here ever measured
> rather than estimated, and it came in at **3.1× its ranked cost**. Three things
> follow, and all three are Elliot's:
>
> 1. **The remaining rows are ranked against gains of 0.03–0.17 bpc. Width bought
>    0.25.** Ranks 3 and 4 (RMSNorm, the mechanism study) both target the
>    zero-context and within-reach components — and §13.3 measured width moving
>    within-reach by **+0.12538**, against the best architectural arm's +0.0288.
>    Neither is refuted by that; both are now candidates whose *scale* matters as
>    much as their mechanism, and neither has been measured at any width but 735K.
> 2. **The measurement §7.2 said would reorder this table is not the one that ran.**
>    §7.2 step 2 justified running rank 2 first because *"its answer bears on how
>    the other two are read"*. It does — but §13.6 states plainly that the pilot
>    **cannot** supply Phase 5's sizing answer, because it ran on the plain LIF and
>    `EXP_005`/`EXP_012` are two separate demonstrations that this project's
>    statistics do not transfer between its two neurons. **The item that would
>    answer it is a parameter-matched architecture comparison at width** —
>    `EXP_014` §10 item 4, priced there at ~1.2–1.8 GPU-hours. It is not written
>    into this table, because §7.1 is offered and not applied, and because ranking
>    it here would be this report deciding its own next experiment.
> 3. **Every GPU-h in this column is now suspect in a specific, measurable way.**
>    Rev 7 flagged the whole column as descending from a withdrawn cost model.
>    Rev 8 can say more than "suspect": the one row that was measured was **low by
>    3.1×**. Ranks 3–6 sum to 2.70 as written; if they carry the same error they
>    sum to ~8.4, which is a different conversation against ~22 remaining. **The
>    cheap fix is the one `EXP_014` used** — a 400-step pre-flight calibration
>    (`scripts/exp/014_calibrate_cost.py`) costs minutes and replaces the estimate
>    with a measurement before anything is pre-registered against it.
>
> **No rank is renumbered and no row's evidence is rewritten**, on the same
> principle rev 6 followed when it inserted rank 2: this table records what the
> evidence implies, and what to do about it is not this document's to say.

> **Rev 9: the item rev 8 said would answer the sizing question ran, and it
> changed the question. Offered, not applied.**
>
> 1. **Ranks 3, 4 and 5 are all about the two-compartment family or the
>    components width already moved, and the family does not currently train at
>    the size that matters.** `EXP_015` (§14) is the parameter-matched comparison
>    rev 8's note said was missing; both two-compartment legs diverged at 5.0M
>    while the plain LIF and the GRU did not. Rank 5 — the frozen-`beta_s` ladder
>    — is a study *of* that neuron, so its result at 735K would not transfer to a
>    size at which the neuron cannot be trained. **None of these rows is
>    demoted here**; what has changed is that a prerequisite appeared underneath
>    three of them, and it is now decision **#11**.
> 2. **The measurement that reorders this table is no longer an arm.** §14.3:
>    with width the anchor's memory horizon goes 57–60 → **97** and the spiking
>    model's goes 7 → **8**, while at zero context the two are nearly equal
>    (4.0524 against 4.0014). **The gap is reach, and capacity does not buy it.**
>    Every remaining row here targets bits; none targets reach except rank 5,
>    which is blocked behind #11. That is a statement about the ranking's *axis*,
>    not about any row's evidence.
> 3. **The GPU-h column has a second measured row.** `EXP_015` cost ~2.1 against
>    a pre-flight projection of 1.93 for its training — the calibration method
>    rev 8 recommended transferred across *arms*, not just widths, with K3 ratios
>    0.926 / 0.915 / 0.972. **~0.7 GPU-hours of it bought no number**, because two
>    legs ran 20,000 steps of which most computed NaN. A cost model cannot predict
>    that, and §1 records it as spent rather than netting it off.

> **Rev 10: the prerequisite rev 9 put underneath three of these rows now has a
> mechanism, and it is not a hyperparameter. Offered, not applied.**
>
> 1. **Ranks 3, 4 and 5 are still not demoted, and the reason to wait is now
>    sharper.** `EXP_016` (§15) shows the two-compartment neuron's
>    backward-through-time has a per-step multiplier bounded by nothing once
>    `|w·vs| > 1`, while the plain LIF's is **provably** bounded by 0.5123596 at
>    any width. Rank 5 — the frozen-`beta_s` ladder — is a study of the parameter
>    that shields the compartment doing this, so its result at 735K would not
>    transfer to a size where the arm cannot be trained.
> 2. **A row that is not on this table has appeared underneath it.** Anything that
>    *bounds* the recurrence — clamping or resetting the slow compartment,
>    normalising `cur`, truncating BPTT — is a **new arm**: new function class,
>    new σ, its own pre-registration, and for two of those a new kernel. None is
>    ranked here, because §7.1 is offered and not applied and because ranking it
>    would be this report choosing its own next experiment. **It is referred
>    under #11.**
> 3. **The GPU-h column gains a third measured row, and it is the cheapest yet.**
>    `EXP_016` cost **~0.8** and trained nothing — it ran on checkpoints already
>    on disk. **~0.37 of it bought no number**, both times because the instrument
>    was wrong rather than the measurement. Against ranks 3–6's estimates, the
>    lesson rev 8 drew still holds: a pre-flight is cheap, and so is reading how
>    an existing script already solved the same problem (`011:280-285`).

> **Rev 14: two arms moved the component this table barely attacks, and neither
> is on it. Offered, and — as with every revision of this table — not applied.**
>
> 1. **The within-reach component has a new best, and it came from the head.**
>    §7 item 2 records within-reach as 51.8 % of the remaining gap and barely
>    attacked: the arm designed for it moved it −0.0065, and the best any arm had
>    managed was +0.0288. **§20.5 measures a wider all-layer READOUT at +0.0431
>    within reach** — paid for with −0.0235 at zero context, which is why the
>    headline is only −0.0216 and why H6 could not separate it from a width arm
>    that buys +0.0119 and +0.0123 evenly. **The two arms have the same total and
>    are not the same intervention.** No row is added for it here.
> 2. **The zero-context component now has an intervention that costs nothing.**
>    §20.1: sparsifying the binary input code recovers 79 % of the binarity cost
>    at **identical parameter count and 0.97× wall-clock**, and §20.3 shows the
>    recovery lands at `c = 0`, which is where §7 item 1 says the component that
>    keeps paying is. This table's remaining rows are all ranked in GPU-hours
>    against gains of 0.03–0.17; this one is free and it is not ranked here.
> 3. **The beyond-horizon component remains untouched by everything.** §21.5
>    puts all four modulator rungs within 0.0006 of zero beyond the horizon, and
>    §20.4's H7 puts the all-layer readout at +0.0015. Added to §19.3's single
>    point, **the phase has now measured six arms that do not extend reach.**
>    §7.1's demotion of the beyond-horizon candidates was made on evidence and
>    this is more of it — but §14.3's finding that *the gap is reach* is
>    unchanged, so what this strengthens is that **no arm built so far attacks
>    it**, not that it should not be attacked.
> 4. **The GPU-h column gains two more measured rows, and the pre-flight
>    transferred twice more.** `EXP_022` cost ~1.9 against a §2.7 budget of ~2.2,
>    `EXP_023` ~2.1 against ~2.5, and `014_calibrate_cost.py`'s 400-step
>    projections came in accurate to ~4 % and ~3 % respectively. That is the
>    third and fourth transfer, and it continues to be the cheapest correction
>    available to the estimates rev 8 flagged as suspect.

### 7.2 The order it should run in, and what has to be settled first

Recommended, not scheduled. The whole of §7.1 is ~3.2 GPU-hours against the ~26.5
still unspent, so **the constraint is review, not budget**.

> **Rev 7 corrects both figures, and flags that one of them rests on nothing.**
> The arithmetic first: ranks 2–6 sum to **3.05**, not 3.2, and §1 records ~6.5
> of ~30 spent, so **~23.5** remains, not ~26.5 — a rev-2 figure carried forward
> through four revisions. The conclusion is unchanged and strengthened: the
> constraint is review, not budget.
>
> The substantive correction is that **rank 2's 0.35 GPU-h is not a measurement
> and no measurement in this tree supports it.** It is a Phase-3 estimate made
> when the only width evidence available was `01_reconnaissance.md` §3.5's
> *eager, forward-only, `no_grad`* sweep ("a 16× width increase is free") — which
> §6.2 of the next report withdrew, in a section titled *"The 14.5 µs constant
> does not transfer"* whose closing instruction is that Phase 3's ROI estimates
> must not reuse it. `EXP_003`'s depth ladder is the only other multi-width
> evidence and it holds `K·d²` nearly constant, so `K`, `K·d` and `K·d²` are
> collinear over its four points and cannot price width at all. A GEMM-FLOP model
> built from `audit_01_gpu_capability.json`'s measured 11.91 TFLOP/s puts the
> pilot at **~0.9–1.4 GPU-h, i.e. 2.6–4× the ranked figure**. That model is
> itself unvalidated at width, so **no corrected number is written into the table
> here**: `EXP_014` opens with a pre-flight calibration that measures it, and a
> later revision records what it measured. The other five rows in §7.1 rest on
> estimates of the same vintage and should be read with the same suspicion until
> each is measured.

**Before anything in §7.1 starts, three of §8 need answers**, and they are cheap
ones: **#1** (Phase-3 §9) because `EXP_008` §4's rider makes adoption conditional
on it; **#5** because rank 1 measures the composition of an arm that is only
*recommended*, and if #5 is "no" the composition run answers a question about an
arm nobody is shipping; **#4** because it decides whether any of this is Phase 4
or Phase 5 — which is the new decision #8.

> **Rev 3, recorded rather than quietly dropped: rank 1 ran without #1 or #5
> being answered.** Decision #8 was resolved "inside Phase 4" and `EXP_011` ran on
> that instruction, so the caveat this paragraph raised is now live rather than
> hypothetical — **the composition is measured on top of an arm that is
> recommended and not adopted, and against a Phase-3 sign-off that is still
> unticked.** That does not weaken §10's numbers, which stand on their own
> pre-registration; it means §10 cannot be read as "the project's best model" until
> #1 and #5 are answered. The remaining items in §7.1 still have this paragraph's
> prerequisites in front of them.
>
> **Rev 5: #1 is answered — signed off 2026-08-05 — and #5 is not.** One of this
> paragraph's two conditions is met, so §10 still cannot be read as "the project's
> best model", and the retrospective character of the sign-off does not change what
> rev 3 recorded here: rank 1 *did* run before either was answered, and this
> paragraph is left standing as the record of that. **§7.2's gate is now #5 and
> #4.**
>
> **Rev 6: #5 is answered too — the threshold is adopted standalone.** Both of
> this paragraph's original conditions are now met; the one still open is **#4**,
> Phase-5 authorisation. Elliot has separately instructed items below to start
> inside Phase 4 rather than wait for #4 — the same scheduling precedent decision
> #8 set for the composition (§10). **This revision records that instruction and
> the ranking it acts on; it does not itself report a closed experiment against
> it.** A future revision closes whatever runs.

Given those, the order below is not arbitrary — each step's result changes whether
the next is worth running:

1. **Composition (§7.1 rank 1, 0.5 GPU-h).** Threshold on top of the adopted
   neuron. It is first because every other ranking depends on the answer, and
   because §7 item 4 says the two gains may not be added — so today the project
   cannot state what its best model scores. **Pre-register the non-additive
   prediction before running it**, or a partial gain will read as a disappointment
   rather than as the measurement `EXP_004` §10.11 item 2 asked for. **Done —
   `EXP_011`, §10.**
2. **Parameter-scaling pilot (rank 2, 0.35 GPU-h)** — added at rev 6. Three
   sizes, one seed each; `EXP_013` §2.2 is the motivating evidence and §7 item 6
   states it. Run before RMSNorm and the mechanism study below, since its answer
   (does capacity move the number at all) bears on how the other two are read.
3. **RMSNorm on `cur` (rank 3, 0.4 GPU-h)** — `03_phase3_candidates.md` §6.3's
   actual #2, unrun, attacks the same components by a different route, and its
   result is only interpretable against step 1's.
4. **Why the threshold arm helps within reach (rank 4, 0.3 GPU-h).** A mechanism
   study, not an arm. §6.7 item 1 is the strongest unexplained result in the
   phase, and it points at the largest remaining component.
5. **The frozen-`beta_s` ladder (rank 5, 1.3 GPU-h)** — the only thing that can
   license tail-widening (§4.5), and the only reason to run it is to *close* that
   question rather than leave it open into another phase.
6. **A training-signal arm (rank 6, 0.7 GPU-h)** — §7 item 5. `EXP_009`'s λ was
   one unswept point and §9.3 of that file forbids reading its null as a ceiling;
   §7 item 5 notes the training-*signal* half of this ask was tried once
   (`EXP_013`, a null) and training-*budget* remains unmeasured.

**Two things Phase 5 should not do.** It should not open on the beyond-horizon
candidates the Phase-3 list still ranks first — §7.1's demotion is on evidence.
And it should not carry `EXP_008`'s gain into a headline until #6 is ruled: the
number rests on a check that failed as written, and §6.8 says so wherever it is
quoted.

---

## 8. What is Elliot's, and is still open

**Seven of these eight are Elliot's, are undecided, and are unchanged by rev 3 —
not one of rows 1–7 has been edited.** Rev 2 sharpened the evidence under #3, #4
and #5 without moving any of them; rev 3 moves none of them either. *(Rev 5: **#1
is now signed off**, so the count is **six** open, and row 1 **is** edited — the
first row in this table to be edited since #8. This paragraph records what was
true at rev 3 and is left as it was; the rev-5 note below the table carries the
change, and rev 4 likewise edited nothing. Rev 6: **#5 and #7 are now also
decided** and their rows edited, and a new row **#9** is added open — the table
now has nine rows, four resolved (#1, #5, #7, #8) and five open (#2, #3, #4, #6,
#9). This paragraph is left exactly as rev 3 wrote it; the rev-6 note below the
table carries this change, as the rev-5 note carried its own. Rev 8: a new row
**#10** is added open, for what follows from decision #7 changing every
trajectory it clips — **ten rows, four resolved, six open**. **No row is edited
at rev 8**, including #7's, whose evidence is corrected in the rev-8 note beneath
the table instead. Rev 9: a new row **#11** is added open, for whether the frozen
recipe may be re-derived at a new size — **eleven rows, four resolved, seven
open** — and again no row is edited.)*

**#8 is resolved, and it is the only one.** It was the single *scheduling*
question in the list — whether the composition ran inside Phase 4 or opened
Phase 5 — and Elliot directed it to run inside Phase 4. It did, as `EXP_011`
(§10). **Resolving it decides nothing about the arm it measured**: adopting the
composed arm is #5's and #1's business, and both are still open.

| # | Decision | Status | Evidence, as of rev 2 |
|---:|---|---|---|
| 1 | **Phase-3 §9 sign-off** | **RESOLVED at rev 5 — signed off 2026-08-05** | Elliot's instruction, 2026-08-05. Ticked in `03_phase3_candidates.md` §9 and **labelled there as retrospective**: Phase 4 opened without it and eight experiments ran before it, so the box's "Phase 4 does not begin until" was not honoured in sequence. Rev 2's evidence cell said *"nothing is left for it to wait on"* and that was still true at rev 4 — `EXP_011` and `EXP_012` are both Phase-4 work and neither touches a Phase-3 deliverable. **This releases the `EXP_008` §4 rider, so #5 is no longer gated by #1.** It does **not** decide #5, and it does not close #2, which is a defect under a ticked §9 box and is named as such in that file's rev 3 |
| 2 | **§6.2 bar definition** (SE-on-the-absolute-curve, per arm) | **open** | `EXP_005` §9.4. Verified since: the per-arm sd it needs is already computed from the ≥3 seeds the adoption rule requires, so the cost `EXP_005` flagged is already paid |
| 3 | **The I5 boundary questions** (three at rev 2, **four** from rev 12) | **RESOLVED at rev 12 — ruled 2026-08-14** | Elliot's instruction, 2026-08-14. **The boundary is extended where the cost model already draws the line**, and the extension is ratified in `01_reconnaissance.md` §4.6 beside the original: a modulator driven by the **previous layer's** output is **admitted** (it is computed before that layer's time loop, so it preserves the depth-sequential evaluation and costs O(1) kernels per layer — `01a_literature_review.md` §4.4's "permission slip"); a modulator driven by the **head** is **not adoptable** (it is downstream of every layer, so it needs a second forward pass), though it may be **built and measured as a labelled diagnostic**, which is what `EXP_018` is. A learned `[d, d]` lateral matrix stays forbidden outright and the O(1)-per-neuron test still applies on top. **Ruled AFTER the evidence existed, and that is stated rather than hidden** — what makes it defensible is that the criterion is structural and would read the same had the dopamine arm won. It costs nothing the project wanted: the one head-driven arm ever built is worse than its anchor on every seed at 1.71× (§17), while the feedforward case it admits is the untested lever aimed at the reach gap §14 identified. **Consequences:** token-shift (`EXP_007`) and dopamine (`EXP_018`) are permanently diagnostic and neither was ever proposed for adoption; the distilled GRU (`EXP_009`) is a teacher, not an arm, and is unaffected; **a feedforward data-dependent decay is now a legal arm and is the obvious Phase-5 candidate.** `scripts/exp/010_phase4_arm_results.py:98` still labels token-shift "I5 status not ruled" — **stale as of this revision, and deliberately not edited**, because that script produced committed evidence under `docs/reports/data/` |
| 4 | **Phase 5 authorisation** | **open** | recommend **not yet**: 5 of 14 candidates closed, ~3.5 of ~30 GPU-hours, and §7.1 item 1 — whether Phase 4's two best arms compose — is unmeasured and costs 0.5 GPU-hours |
| 5 | **Adopt the learned per-channel threshold?** | **RESOLVED at rev 6 — adopted standalone** | Elliot's instruction, 2026-08-08. −0.0590 bpc at 23.6 se (6.4× the bar), 0/128 contexts regressed, 1.12× wall-clock, folds to zero cost — a reparameterisation, confirmed to 6.8–8.3 fp64 eps (§6.3). **The composed arm (`EXP_011`, §10) is explicitly not adopted by this**: it stays provisional at n = 2 of a pre-registered 3 per `CONTRIBUTING.md` §4's provisional-promotion rule (ratified 2026-08-08 — seeds are *added*, never substituted, up to the pre-registered n before "provisional" becomes "adopted"), and is not "the project's best model" until reseeded |
| 6 | **The fold-in tolerance** | **open** | `EXP_008` W1 failed at 1e-3, a bar borrowed from F1, which compares two evaluations of the *same* weights. §6.5 measures why that was the wrong precedent; the algebra holds at 6.8–8.3 fp64 eps. **Referred, not redefined** |
| 7 | **The gradient clip's fp32 accumulator** | **RESOLVED at rev 6 — fixed** | Elliot's instruction, 2026-08-08. `src/snn/train.py`'s `_clip_grad_norm_fp64` accumulates the sum of squares in fp64, the first of `EXP_009` §9.4's two named fixes. Verified on the exact failing gradient (5.46e31, stock zeroes / fix rescales — `tests/test_grad_clip.py`) and against the real R5 capture gate (`tests/test_graph_equivalence.py`, 10/10). **Does not explain or prevent `compose_s0`'s divergence** (§10.4) — a different, still-open mechanism |
| **8** | **Does the composition experiment run inside Phase 4, or as Phase 5's first arm?** | **RESOLVED at rev 3 — inside Phase 4** | Elliot's instruction, 2026-08-04, matching rev 2's recommendation. Ran as `EXP_011` (§10): the arms compose at 2.08326 carried, −28.7 se, 0/128 contexts regressed, **provisional at n = 2** after one seed diverged and was not replaced. The composed arm is a **Phase-5 candidate; adopting it is #5's and #1's business and neither has moved.** ~0.86 GPU-hours including the probe and the divergence chase |
| **9** | **Should a pre-registered bar require a guard-clause review before it's committed?** | **open — added at rev 6** | `EXP_012`'s Y5 and `EXP_013`'s N1 (§12.4) fired the identical defect: a bar specified without a guard the experiment's own limitations section had already written down. `EXP_013` §10 item 2 named this report by section as the place lacking a row for it. **The protocol rule is already ratified** — `CONTRIBUTING.md` §2, since `51b44c6` (2026-08-05, Elliot, citing both experiments) — so this row tracks the report's own bookkeeping, not an open question of substance |
| **10** | **What follows from decision #7 changing every trajectory it clips?** | **open — added at rev 8; the re-baselining half is ruled** | `EXP_014`'s G1 gate failed and §13.5 chased it to `_clip_grad_norm_fp64`: the clip fires 3 times in 20,000 steps on the committed recipe, the fp32/fp64 coefficients differ in their last bits there, and the run ends **+1.97e-03 bpc** apart. **Elliot has ruled that the project does not re-baseline** — new experiments train their own anchor on the current tree rather than reproducing a committed number. **Still open: whether a standing rule is wanted that a `src/snn/` change altering any committed figure must be recorded as such before merge.** Rev 8 corrected `train.py`'s docstring, which claimed the fix changed no behaviour; it does not revert the fix and nothing here argues for reverting it |
| **11** | **May the frozen Phase-2 recipe be re-derived at a new size, and how?** | **open — added at rev 9** | `EXP_015` (§14) trained the adopted two-compartment arm and the composed arm at 5.0M parameters and **both diverged**, at steps 5138 and ~2000, while the plain LIF and the GRU at the identical width, seed and recipe trained cleanly. The recipe — `lr = 3e-3`, cosine over 20,000 steps, `grad_clip = 1.0`, `weight_decay = 0.1` — was fixed at **735K** parameters and has never been re-derived at any other size. **This blocks Phase 5 as currently conceived**, whose plan was to scale the adopted arm. What is Elliot's: whether a recipe term may be changed at a new size at all (it is frozen by `02a_phase2_spec.md`), and if so whether that runs as a **diagnostic** — the arm dies at step 5138, so an LR probe is ~11 GPU-minutes, not a full run — or as a pre-registered arm. **Nothing here proposes a value**, and no term is changed at this revision |

**Two rows above now carry evidence `EXP_011` has overtaken, and the rows are
deliberately left as they were.** Editing an open decision's evidence is how a
decision gets made quietly, so the correction is recorded here instead:

* **#4 (Phase 5 authorisation)** reads *"§7.1 item 1 — whether Phase 4's two best
  arms compose — is unmeasured and costs 0.5 GPU-hours."* **It is now measured**
  (§10) and it cost ~0.86 GPU-hours. That removes rev 2's stated reason for
  recommending "not yet"; the other reasons stand — 5 of 14 candidates closed and
  ~4.4 of ~30 GPU-hours spent — and **#4 is still open and still Elliot's.**
* **#7 (the gradient clip's fp32 accumulator)** reads that the fix is independent
  of the three new arms. It still is. But §10.4 shows the clip overflow is **not**
  what killed `compose_s0`, so **fixing it would not have prevented Phase 4's
  second divergence.** That is an argument for chasing the second mechanism, not
  against fixing the first, and **#7 is unchanged and still open.**

Rev 3 adds no new decision. If the composed arm is to be adopted, that is #5 and
#1, already on the list.

**Rev 4 adds no new decision either, and edits no row. It sharpens #6, which was
the one item on this list that could not be decided at all**, and the correction
is again recorded here rather than in the row:

* **#6 (the fold-in tolerance)** reads that `EXP_008`'s W1 failed at 1e-3 and that
  §6.5 measures why the precedent was wrong. Both still stand. What was missing is
  that the project's only *other* fold-in measurement — `EXP_011` C4 at 1.05e-06 —
  disagreed with it by three orders of magnitude, and §10.5 explicitly withheld it
  as evidence "until understood". **`EXP_012` (§11) understands it**: the two
  residuals differ because the two neurons put wildly different amounts of
  probability mass at the firing threshold, and `EXP_011`'s number is now
  available to #6 rather than withheld from it.
* **What that does to the decision is narrow and worth stating plainly.** It does
  **not** propose a tolerance — `EXP_012` sets none, by design. It establishes
  that **a fold-in tolerance cannot be one project constant**: the same identity,
  through the same fold, costs 2.0e-03 bpc on the baseline neuron and 1.4e-05 on
  the adopted one. Any single number for W1 will be far too loose for one arm or
  far too tight for the other, and **that is the shape of the choice #6 is**.
* **#4 (Phase 5 authorisation)** now reads against **~4.6** of ~30 GPU-hours, not
  the ~3.5 in its row or the ~4.4 rev 3 corrected it to; `EXP_012` cost ~0.2 and
  trained nothing. **5 of 14 candidates closed is unchanged** — `EXP_012` is a
  mechanism study and closes none. **#4 is still open and still Elliot's.**

**Rev 5 edits exactly one row — #1 — because #1 is now decided, and a decided row
is edited rather than annotated.** That is `#8`'s precedent at rev 3, not a new
one: the rule this report follows is that an *open* decision's evidence is never
edited (the correction goes in prose beneath the table, as rev 3 and rev 4 both
did), while a decision that has actually been made is recorded in its own row so
the table cannot be read as still asking the question.

* **#1 (Phase-3 §9 sign-off)** — **signed off by Elliot on 2026-08-05.** The tick
  is in `03_phase3_candidates.md` §9 and is **labelled retrospective in place**,
  because Phase 4 opened without it and `EXP_004`–`EXP_009`, `EXP_011` and
  `EXP_012` all ran before it. Nothing in `EXP_011` or `EXP_012` bore on it either
  way; both are Phase-4 work and neither touches a Phase-3 deliverable, so rev 2's
  *"nothing is left for it to wait on"* was still accurate when it was ticked.
* **What #1 does to #5, and it is only this.** `EXP_008` §4's rider made the
  threshold arm's adoption *"recommended to Elliot, not taken"* on the ground that
  "§9 of the Phase-3 report is still unticked". That ground is gone. **#5 is
  therefore ungated by #1 and is still undecided** — its own substance, and its
  coupling to #6, are untouched by this tick. **Six decisions remain open.**
* **What #1 does not do.** It does not close **#2**, and #2 is a defect sitting
  under a §9 box that is now ticked: §9's "acceptance criterion derived from
  measurement and calibrated per-context" is ticked, and `EXP_005` §9.4 has since
  shown that bar built on the wrong scale. The sign-off does not launder it. It is
  named in `03_phase3_candidates.md` rev 3, it remains Elliot's, and the count it
  governs is unchanged at 5/128 under all three constructions.
* **§7.2's first prerequisite is met, and its other two are not.** That paragraph
  requires #1, #5 and #4 before anything in §7.1 starts. One of three.

**Rev 6 edits two rows — #5 and #7 — because both are now decided, and adds one
new row, #9, for a decision that did not exist before.**

* **#5 (adopt the learned per-channel threshold?)** — **adopted standalone**, per
  Elliot's instruction. §7's item 4 composition warning and §6.5's ~2e-3 drift
  caveat are carried into the row rather than dropped: adopting the threshold
  does not adopt the composition, which stays `EXP_011`'s provisional n = 2 result
  until reseeded. **§7.2's second prerequisite is now met; only #4 remains.**
* **#7 (the gradient clip's fp32 accumulator)** — **fixed**, per Elliot's
  instruction. `_clip_grad_norm_fp64` (`src/snn/train.py`) replaces the stock
  `clip_grad_norm_` call in `Trainer._step_body`, upcasting each gradient to fp64
  before squaring rather than after — the exact computation
  `009_chase_divergence.py` used to diagnose the bug, now load-bearing instead of
  diagnostic. `tests/test_grad_clip.py` reproduces `EXP_009` §9.4's numbers
  directly: a synthetic 5.46e31 gradient, which the stock implementation still
  zeroes (`clamp(1.0/(inf+1e-6), max=1.0) = 0`, exactly) and the replacement
  rescales to a finite, nonzero update. `tests/test_graph_equivalence.py`'s full
  10-test R5 gate — both the static capture-safety AST checks and the real
  graphed-vs-eager equivalence tests on this GPU — pass unchanged. **This closes
  one bug, not two**: §10.4 already established `compose_s0`'s divergence is a
  different mechanism (ordinary-magnitude gradients, fp32/fp64 norms equal,
  identical NaN on the eager path), so this fix would not have prevented it and
  does not claim to.
* **#9 (guard-clause review before a bar is committed) — added, and open.**
  `EXP_012`'s Y5 (§11.4) and `EXP_013`'s N1 (§12.4) are the same defect twice: a
  pre-registered bar on a difference, committed without constraining what the
  subtrahend is allowed to do, satisfiable by a sufficiently damaged model
  without any real improvement. `EXP_013` §10 item 2 asked this report by name to
  gain a row for it. **The protocol side of this is not open** —
  `CONTRIBUTING.md` §2 has carried the rule since `51b44c6` (2026-08-05, authored
  by Elliot, citing `EXP_012`'s Y5 and `EXP_013`'s N1 by name, predating this
  report's own row for it). Row #9 exists so this table stops being the one place
  that had not caught up.
* **Two further `EXP_013` referrals, noted rather than tabled.** §10 item 3
  (whether noise injection belongs on `03_phase3_candidates.md`'s ranked list at
  all) and item 4 (the no-overfitting finding's bearing on future candidates,
  captured above in §6.7 item 3, §7 item 6 and §7.1's new rank 2) are Elliot's to
  rule on but do not need their own numbered row — neither is a yes/no this table
  format fits, and both are fully stated where they already live.

**Rev 8 edits no row and adds one — #10 — for a consequence that did not exist as
a question until `EXP_014`'s G1 gate failed.**

* **#10 (what follows from decision #7 not being numerically inert) — added, and
  half of it is already ruled.** §13.5 establishes by substitution that
  `_clip_grad_norm_fp64` changes the training trajectory wherever the clip
  actually fires — three steps out of 20,000 on the committed baseline recipe,
  worth **+1.97e-03 bpc** of final test loss. **Elliot has ruled the first half:
  the project does not re-baseline.** Every new experiment trains its own anchor
  on the current tree instead of reproducing a committed figure — which is what
  `EXP_014` itself did once G1 failed, and §13.2 quotes S1 against both anchors
  precisely so the verdict does not depend on the choice. **The second half is
  open**: whether a standing rule is wanted that a change to `src/snn/` altering
  any committed number must be recorded as such before it is merged.
* **What rev 8 did about it without deciding anything.** `src/snn/train.py`'s
  docstring said the fix *"does not change behaviour for any gradient that was
  already representable in fp32"*. That sentence is true of the returned norm and
  **false of the trajectory**, and it is corrected in place with §13.5's
  mechanism and the +1.97e-03 figure named. **`tests/test_grad_clip.py` needed no
  change**, which is worth recording: its bitwise assertion covers only the
  `max_norm=inf` sentinel path, where no clip happens at all, and the test that
  does exercise a firing clip asserts `allclose(rtol=1e-5)` rather than equality.
  **No test ever asserted trajectory equivalence — the docstring did.**
* **Row #7 is not edited.** It is a *decided* row and this is a correction to its
  evidence rather than a new decision, so it is recorded here, on the rule rev 5
  set out. **Nothing in §13.5 argues for reverting the fix** — it repairs a
  failure that silently zeroes every update for the rest of a run.
* **`CONTRIBUTING.md` gains nothing at rev 8.** The standing-rule half of #10 is
  precisely what would be added to it, and that half is open.

**Rev 9 edits no row and adds one — #11 — for a question that did not exist until
the adopted arm failed to train.**

* **#11 (may the frozen recipe be re-derived at a new size?) — added, and open.**
  `EXP_015` §14.5 chased the divergence to the same boundary `EXP_011` §10.4
  reached and no further: layer 0's backward, on both the fused and the eager
  path, with the forward and the loss finite and the fp32/fp64 norms in
  agreement. So the report cannot say the recipe is *at fault* — only that under
  it, two arms do not train at a size at which two others do. **That distinction
  is why this is a decision and not a finding.**
* **What rev 9 did about it without deciding anything.** Nothing. No recipe term
  is changed, no value is proposed, and `02a_phase2_spec.md` is untouched. The
  cost of the cheapest discriminating measurement is recorded (~11 GPU-minutes
  per LR probe, because the arm dies at step 5138 rather than at 20,000) so that
  the decision is made against a price rather than a guess.
* **#9 is re-evidenced, not edited.** `EXP_015`'s P1 and P2 are the **third**
  pre-registered rules in this phase to fire and mislead — UNRESOLVED, which
  §3.0 defines as a statement about statistical power, returned for two runs that
  produced no number at all, with seeds recommended as a remedy that cannot work
  against a deterministic divergence. The corrected form is a **DIVERGED / NO
  RESULT** verdict; per `CONTRIBUTING.md` §3 it is referred, not applied, and #9's
  row is left exactly as rev 6 wrote it.
* **#4 (Phase 5 authorisation) is re-evidenced in prose, and the row is not
  touched.** It reads against ~3.5 of ~30 GPU-hours; §1 now records ~10.1. More
  substantively, §14 changes what Phase 5 would be authorising: at matched size
  the anchor is 0.45374 ahead, its reach is **97** against **8**, the two models
  are nearly equal at zero context, and the arm Phase 5 planned to scale does not
  currently train at scale. **#4 is still open and still Elliot's.**

**Rev 11 edits no row and adds none, and the second half of that sentence is a
judgement call the reader should be able to overturn** — `EXP_017` produced two
findings that could each justify a twelfth row, and the case for and against is
set out below rather than settled here.

* **#11 (may the frozen recipe be re-derived at a new size?) — re-evidenced, not
  edited, and its *form* is now in question twice over.** `EXP_016` §2.4 observed
  before its run that `dv` contains no learning rate, no schedule, no weight decay
  and no clip. §16 supplies the stronger version of the same point: the failure
  was **fixed without touching the recipe at all**, by a change that alters no
  forward value. So the answer to "may the recipe be re-derived at a new size?"
  may be *"it need not be"* — which is not an answer to the question as asked, and
  is why the row is left exactly as rev 9 wrote it. **No recipe term is changed
  and no value is proposed.**
* **#10 (what follows from decision #7 changing every trajectory it clips?) —
  re-evidenced, and the new instance is much sharper than G1's.** Rev 8's evidence
  was a **+1.97e-03 bpc** shift on the plain LIF. §16.4's is that the *adopted
  arm*, retrained at the width it was adopted at, **lost two of three seeds to
  divergence** where the committed record holds seven clean ones. **The clip is
  not shown to be the cause** — `EXP_017` §10 item 6 refers a ~9-GPU-minute test
  either way and deliberately did not run it under a pre-registration that had not
  declared it. If it *is* the cause, then the standing-rule half of #10 — whether
  a `src/snn/` change that alters a committed figure must be recorded as such
  before merge — stops being bookkeeping and starts being the reason seven seeds
  of evidence describe a tree that no longer exists.
* **#5 and #1 are re-evidenced in prose, and neither row is touched.** Both rest
  in part on the two-compartment neuron's behaviour at 735K. §16.4 does not
  overturn that evidence — the surviving fresh anchor reproduces the committed
  figure to **0.00103 bpc** — but it does show the configuration is close enough to
  a stability boundary that two of three fresh seeds fell off it. **Whether that
  changes what "adopted" should mean is #5's owner's, and this revision does not
  argue a direction.**
* **The case for a twelfth row, stated so it can be taken up.** A reader could
  reasonably hold that "**is the adopted arm's 735K evidence still current?**" is a
  new open question rather than #10's or #5's substance, and that it deserves its
  own row. Rev 11 does not add it because every fact underneath it is already
  attached to an existing row and because adding a row is the one thing this table
  treats as consequential. **If Elliot wants it as #12, nothing in §16 needs to
  change for it to be added.**
* **#9 is re-evidenced, and for the first time the referred correction was applied
  *prospectively rather than retroactively*.** `EXP_017`'s H3 was written with the
  **DIVERGED / NO RESULT** distinction built into its decision rule before any run
  started, so when two anchors died the bar returned **NOT RUN** instead of
  `EXP_015`'s misleading UNRESOLVED. **The rule is still referred, not ratified,**
  and #9's row is left exactly as rev 6 wrote it — but a referred correction that a
  later experiment adopts voluntarily is evidence about the correction that a
  closed log cannot supply.

---

**A rev-13 note on two open rows, added in prose because neither is decided
and the rev-5 rule keeps an open row's evidence out of the table.**

**#2 — the per-context machinery — now has the negative control it never had.**
`EXP_005` called §6.2's scale wrong; §18.5 measured how wrong. `if_nest_s0` is
**bitwise** its own anchor over a full 20,000-step run — T1 puts their final bpc
exactly 0.0 apart — and the statistic still reports **120 of 128 contexts
"significantly worse"**. The per-context 2σ bars fall to 2.4e-05 beyond `c ≈ 70`
where a bit-identical run differs from the three-seed mean by ~0.005: **up to
80× below single-seed variation.** `n_contexts_significantly_worse` is therefore
**unusable at n = 3**, and every number this project has quoted from it at n = 3
should be read with that floor beside it. **No redefinition is proposed here** —
that is #2's owner's, and this revision does not pre-empt it. What it does is
supply the measurement #2 has been open for want of since rev 2.

**#6 — whether a provable reparameterisation may enter a headline — now has two
independent measurements of the same phenomenon, pointing the same way.**
`EXP_008` found the *over*-parameterised form of a threshold worth **−0.0590
bpc** on a function class it had proved identical. §18.2 finds that removing a
*provably redundant* factorisation of the input path costs **+0.0946 bpc**, at
**1.63×** what §13's slope prices those parameters at, and that spending the
freed 35.7 % on width does not recover it. The two arms differ in direction, in
layer and in sign of intervention, and they agree: **on this architecture,
redundant parameterisation of the input path is worth real bits to the
optimiser, and the function class does not predict how many.** That does not
decide #6 — whether such a gain may be *quoted as a result* is exactly what is
open — but it removes the reading under which `EXP_008`'s number was an
artefact of one arm. **`EXP_008`'s gain still does not enter a headline in this
report**, and §19's own pre-registration holds itself to the same rule.


---

**A rev-14 note on three open rows and one protocol failure. No row is edited;
the rev-5 rule keeps an open decision's evidence out of the table.**

**#6 — whether a provable reparameterisation may enter a headline — now has a
third measurement, and it is the sharpest of the three.** §18.2 found removing a
*provably redundant* factorisation costs +0.0946; `EXP_008` found the
*over*-parameterised form of a threshold worth −0.0590 on a function class it had
proved identical. §21.2 adds the cleanest case yet: **the modulator arm with its
signal held CONSTANT is `cur·(1 + kappa_c)`, which by `EXP_008` Identity 1 is
exactly a learned per-channel gain — and it is worth −0.02975 against its anchor,
which is what the *aligned* time-varying version is worth (−0.02953).** The three
measurements differ in layer, in direction and in sign of intervention, and they
agree: **on this architecture, redundant parameterisation is worth real bits to
the optimiser and the function class does not predict how many.** That does not
decide #6 — whether such a gain may be *quoted as a result* is exactly what is
open — and `EXP_023` §6 item 2 held itself to the rule, quoting no figure of
`EXP_008`'s as a result. **What is new is that #6 is now load-bearing for an arm
that beat its anchor**, not only for one that was measured and set aside.

**#9 — guard-clause review before a bar is committed — has a third and a fourth
instance, and one of them was caught in time.** `EXP_022`'s H4 is the third: a
bar on `|c0| > |total|` over an **additive** decomposition, which by construction
requires the rung to *lose* outside zero context, so the cleanest possible
success fails it — and the motivating example the bar was built from does not
satisfy it either. That is `EXP_012`'s Y5 and `EXP_013`'s N1 again. The fourth is
different in kind and worth separating: **`EXP_022` §5 and `EXP_023` §5 both
have decision cells for HELD and FAILED and none for UNRESOLVED**, which is the
verdict a 2σ bar most often returns — and on 2026-08-20 it returned it four
times across the two experiments (H1, H6, H2, H4). Both are **referred, not
repaired**. What is genuinely new is that **the H4 defect was caught by audit
BEFORE the resolver was committed and before any bpc was read**, so the
correction was free rather than retroactive; the pre-registered statistic is
still emitted exactly as written beside it. That is the first time this project
has caught a #9-class defect inside the window where catching it costs nothing.

**#2 — the per-context machinery — is untouched and both new experiments honoured
its standing caveat.** Neither used `n_contexts_significantly_worse` as a bar
anywhere; both report it as a marker, against §18.5's bitwise-copy control that
read 120 of 128. No redefinition is proposed.

**A PROTOCOL DEVIATION IN §18 AND §19, recorded here because nothing else in the
report records it.** `EXP_020` §8 and `EXP_021` §8 each require the resolver
**committed before any bpc is read**. Neither was: both resolvers reached git for
the first time on 2026-08-20, after their numbers had been read and written into
rev 13. The pre-registration SHA-256 stamps hold byte-for-byte, so the
*hypothesis-against-runs* ordering is evidenced and the *resolver-against-numbers*
ordering is not — and no commit made later can supply it. Both logs additionally
state they were committed before the first run while having been untracked, and
supply `CONTRIBUTING.md` §2's stamped hash **without** the recorded
reconstruction recipe that is the second half of the substitute for a commit.

**That second half is now measured rather than asserted, and the result is worse
than "undocumented".** `scripts/exp/verify_prereg_hash.py` reconstructs a
pre-registration's pre-results bytes and hashes them against the stamp.
**`EXP_022` and `EXP_023` VERIFY** — their stamped bytes reproduce exactly, so
their hypotheses are provably the ones their runs were scored against.
**`EXP_020` and `EXP_021` do not**, and a deliberate search says the recipe is
unrecoverable rather than merely unlisted: every prefix of each live log was
hashed under both line-ending conventions, crossed with six plausible status
lines, five §9 layouts and two truncation points, and **nothing reproduces
either stamp**. **This is NOT a finding that either hypothesis was edited** — a
cosmetic change above §9 after stamping would produce exactly this and is far
more likely — but it means that for those two experiments **the stamp is no
longer independent evidence of anything**, which is the guarantee §2 exists to
preserve. `CONTRIBUTING.md` §3's rule for an unrecoverable reasoning trail is to
say so in the artifact, and this is that. The older logs (`EXP_005`–`EXP_018`)
predate the layout and read UNVERIFIED for that reason; they are evidence
neither way and the tool says so rather than implying otherwise.

**`EXP_022` and `EXP_023` met the resolver condition too** (`858c48a`,
`98d3f62`), which is what makes the deviation visible rather than invisible. **Nothing in §18 or §19 is retracted**;
the runs are what they are and the bars fired as they fired. Whether the project
wants a standing gate that refuses to run a resolver whose file is untracked is
**Elliot's**, and it is the same shape as #10's still-open half.

**A rev-15 note. No row is edited, no decision is ruled, no bar is proposed, and
`EXP_024`'s pre-registration is not touched. Four corrections, and the first of
them corrects rev 14's own note above.**

**1. The pre-registration audit was reading one key, one filename and one
placeholder, and it had no second route. With all four repaired the record is
7/9, not 2/12.** The tool printed **"2/12"** when rev 14 ran it; rev 14 itself
records its verdicts qualitatively and states no ratio, so the figure is quoted
here from the tool's stdout rather than from the note above.
`scripts/exp/verify_prereg_hash.py` now also tries the **object database** — if
any commit in a log's history holds bytes that hash to the stamp, the
pre-registration is verified against a committed object rather than against a
guess about layout. The result, with the tree at `593baaa` and the revised tool
in the working copy:

| verdict | experiments | route |
|---|---|---|
| **VERIFIED BY COMMIT** | `EXP_014` (`d59f98e`), `EXP_015` (`d4469d6`), `EXP_017` (`50b3904`), `EXP_018` (`dd652bf`), `EXP_022` (`858c48a`), `EXP_023` (`98d3f62`) | a committed blob hashes to the stamp |
| **VERIFIED BY RECIPE** | `EXP_013` | the reconstruction stub the log records at its own `:424-431` |
| **UNVERIFIED** | `EXP_020`, `EXP_021` | one commit each, and it already contained results |
| **NO STAMP** | `EXP_005`, `EXP_007_008`, `EXP_009`, `EXP_011` | no `prereg_*` key in the manifest at all |

**Three defects moved the denominator and one absent route moved the
numerator**, and separating them matters because only the first three were
defects. The tool read `prereg_sha256_before` while `EXP_013`–`EXP_018` stamp
`prereg_sha256_before_first_run`. It parsed `exp_007_008_run_manifest.json` into
a nonexistent id `007`. And it counted **three** unstamped manifests — `EXP_005`,
`EXP_009`, `EXP_011` — in the denominator of "stamped pre-registrations", which
reports an absent stamp as a failed one; the fourth unstamped manifest,
`exp_007_008`, never reached that denominator at all, because the id defect had
already excluded it, so the two are interlocked rather than independent and the
twelve is `3 + 9`. **Correcting all three and keeping the recipe route alone
gives 2/9.** The numerator moves only when the object database is consulted.

**A fourth defect, and it is the same species as the first.** `PLACEHOLDERS` is a
single-variant lookup table and it omitted the variant used by `EXP_013` — the
one log `CONTRIBUTING.md` §2 names as the exemplar that *"records the recipe"*,
and which states that recipe verbatim at `:424-431`. The exemplar read UNVERIFIED
for want of a table entry the log itself supplies. That is recorded here rather
than quietly repaired, because a tool written to catch this class contained two
instances of it.

**The four newly commit-verified blobs are the pre-registration commits
themselves** — the oldest commit touching each log, subject *"Pre-register
EXP_0NN: …"*, status `PRE-REGISTERED — no results yet`, §9 empty. **No false
positive is possible in that set:** §9 in all six matched blobs holds nothing but
its heading and a placeholder, so no stamp was computed after results existed.
So **rev 14's "evidence neither way" is false for four of `EXP_005`–`EXP_018`**,
and `EXP_013` verifies on top of that.

**One qualification the commit route needs, and the tool now closes it rather
than leaving it open.** A stamp that matches a historical blob is strong evidence
that the **stamp** is honest and no evidence at all that the **working file** is
unedited — the guarantee §2 actually wants. So every commit-verified row now also
reports `live_file_agrees_above_results`: the blob and the live file, both
truncated at `## 9. Results`, compared line by line. **All six read `True`.**

That check needed one correction before it could be trusted, and it is the kind
this tool exists not to make. Dropping only the *first* `**Status:` line left
`EXP_022`'s five continuation lines — its `**Status: CLOSED …**` wraps over six
lines and summarises the verdicts — looking like an edited hypothesis. The whole
status block is the one permitted change above the heading, so the whole block is
dropped from both sides. A first-line filter would have convicted a clean log.

**Nothing changes for `EXP_020` and `EXP_021`** — rev 14's finding stands exactly
as written, and the reason is now mechanical: their logs have one commit each and
it already carried §9.

**2. `EXP_024` §4's prose generalises a claim the same document scopes correctly
three times, and the generalisation is false on the wider record.** §4's own H1
row says *"**Six arms** have moved this component by at most 0.0015 (§0)"*, §6
item 5 says *"across **six arms**"*, and §0's table names those six — all true,
all inside the band, maximum `|0.00154|`. **The premise H1 rests on is sound as
scoped.** The unqualified restatement is not:

> *The beyond-horizon component has never been observed outside ±0.0016 on any
> arm, so an UNRESOLVED H1 is the modal outcome and is not a disappointment.*

`scripts/audit/11_beyond_horizon_census.py` reads every committed block carrying
the component under **either** of its two spellings — `beyond_horizon_gain_bpc`
and the `beyond_horizon_bpc` of five `*_memory_horizon.json` files — **41 blocks
across 12 artifacts, zero GPU** — and classifies them, because most do not bear
on the claim. Gap blocks (`remaining_gap_vs_gru`, `i5_gap_decomposition`) are
excluded: `010_phase4_arm_results.py:569` computes the first as
`decompose(curve, gru_curve)`, the same function with the arm in the *reference*
slot, so they measure the gap to the anchor that §0 already cites as its
motivation rather than an arm's gain. `bars.*` blocks duplicate an arm row, and
`if_binin` — stored **bit-identically** in `EXP_020` and `EXP_022` — is counted
once. `docs/reports/data/audit_11_beyond_horizon_census.json` carries all 41 rows
with the classification on each.

**The cleanest counterexamples are not the large ones**, and neither is in §0's
table: **`tokenshift` reads +0.00565 on an arm that WON** (total +0.04847) — 3.5×
the band, 12× the bitwise-copy floor, with no question of compression attached —
and **`io_sp05` reads +0.00312** on a winning arm (+0.03560). Of **24 distinct
primary arm-gain readings outside the two-compartment family, 7 are outside
±0.0016** and three clear `EXP_024`'s own H1 bar of 2σ = 0.00922:

| reading | total | beyond horizon | share | source |
|---|---:|---:|---:|---|
| `if_foldwide` | **−0.01366** | **+0.01708** | **−1.2508** | `exp_020_arm_results.json` |
| width `d=1481` *(a width leg, not an arm — §1: "`EXP_014` adopts nothing")* | +0.24961 | **+0.01497** | n/a | `exp_014_scaling_results.json` |
| `if_fold` | **−0.09453** | **+0.01400** | **−0.1481** | `exp_020_arm_results.json` |

**Two of the three are arms that LOST**, so a positive beyond-horizon reading on
them cannot be read as a win — and the census flags both `compression_suspect`.
**But the sign pattern is not itself diagnostic, and this note will not claim it
is.** From `scripts/exp/010_phase4_arm_results.py:190-196` the quantity is
`beyond = (ref[CMAX] − arm[CMAX]) − (ref[h] − arm[h]) = g(CMAX) − g(h)` — a
**difference of gains**, invariant under any constant shift of the arm's curve,
so an arm uniformly worse by any amount scores `beyond = 0`. A negative total
does not mechanically induce a positive beyond. The negative
`beyond_horizon_share` is `beyond/total` restating the two signs it is built
from, not independent corroboration. And the pattern fails its own control:
**`if_binin` lost more than `if_foldwide`** (−0.04478 against −0.01366) and read
beyond **−0.00477**, its horizon *falling* to `[6, 6, 7]`.

**So the flag is checked against the horizon rather than left standing**, and the
horizon declines to support it. `exp_020_memory_horizon.json`,
`by_arm.<arm>._paired.horizon_2sigma_per_seed`: `if_fold` **`[9, 10, 9]`**,
`if_foldwide` **`[9, 8, 9]`**, against `if_anchor` `[7, 7, 7]` and a **bitwise
copy of the anchor** at `[7]`. Both move on 3 of 3 seeds, past the 6/7/8 band
§13.3 treats as "did not move". The two markers **agree** for exactly the two
arms in question.

**None of which is new to this report, and that is worth saying plainly.** §18.4
already concluded, in bold, that *"the fold's cost is a REACH cost"*. What is new
is only that the same two readings sit outside a band `EXP_024` §4 states as
universal, and that the horizon corroborates them.

**`EXP_024`'s H4 answers the compression question on these cases, and it fires.**
`exp_020_arm_results.json`'s `short_context_table` stores the multiples outright:
`if_fold` at c = 3 is **7.90×** its bar and at c = 4 **10.22×**; `if_foldwide`
**5.53×** and **6.09×**. Both would route to §5's second row — *"it bought reach
the losing way"* — so the anti-signature is not silent here. **Whether H4 is a
sufficient guard in general remains open**, and no bar is proposed, widened or
repaired by this note. *(The width leg's apparent H4 failure is confounded by its
much lower asymptote and is not cited alongside them.)*

**What the census establishes is narrower than "the premise is falsified" and
more useful than it.** H1's 2σ bar is cleared by three committed readings, two of
them on arms that lost bits, and those two are precisely the ones whose horizon
marker also moved. **Whether `if_fold` and `if_foldwide` bought reach and paid
for it elsewhere is unresolved on the committed record** — the integer horizon
has no σ anywhere in this project (`EXP_017` §10 item 8; `EXP_024`'s own H9
carries it as a marker with no verdict), `EXP_020` §4 pre-registered H5 with no
bar on it, and n = 3 cannot resolve it. **Two markers agreeing is two markers
agreeing.** It is worth recording only that these are arms carrying **no new
kernel and no new hand-written backward** (`f90c8c5`: the fold axis "sits outside
the time loop"). **Nothing here proposes running anything.**

**3. `EXP_024` modulates exactly one layer, that layer is silent at
initialisation, the screen's guard for that condition misses it by 3e-07, and the
pre-registration mentions neither — nor that the project has already measured the
silence away by step 50.**

§2.6's table carries both layers (`0.116 / 0.399` for `beta_raw`, `0.041 / 0.042`
for `k_gate`), though nothing in its headers says the "/" separates layers; the
prose is the only place that pairing is explained, and it explains it correctly
for **layer 0**, reporting `k_gate`'s 0.041 as 2.8× below `beta_raw`'s 0.116. §3
states outright that *"`K = 2`, so there is one modulated layer"*, and §2.2 and
`model.py:1640` put it at **layer 1**, where the ratios are `k_gate` **0.041750**
against `beta_raw` **0.399252** — **9.56×**. **That widening is `beta_raw`
moving, not `k_gate` failing:** `k_gate` reads 0.041118 at layer 0 and 0.041750
at layer 1, a **1.5 %** difference clearing the screen's 0.01 floor by ~4.1× at
both, while `beta_raw` moves **3.45×**. The gate is no harder to find at the
layer that carries it; its co-parameter is far easier.

**The layer-1 reading is measured in a regime the pre-registration never names.**
`exp_024_reachability.json` records layer 1's firing rate at this init as
**2.98e-07** — layer 0 is at **5.2 %** in the same screen — and
`audit_07_init_pathology.json` records layer 1 at `d=512, K=2, thr=1.0` as
**exactly 0.0 on 3 of 3 seeds, `silent_layers: 1`**. The screen's own guard is
`silent = [... if r["firing_rate"] == 0.0]`
(`scripts/audit/09_gradient_reachability.py:888`), an **exact** equality, so
2.98e-07 clears it and every layer-1 parameter is stamped `in_silent_layer:
false`. That the margin is real and known is in the same file at `:576`, which
describes a threshold change that *"tips layer 1 from 2.98e-07 to silent"*.

**And the same artifacts record the escape, which is why this is a gap in §6's
limitations list rather than a hazard to the arm.** `02_baseline_report` §5.3 —
titled *"It self-corrects — measured, not assumed"* — tabulates layer 1 at
`0.000` at step 0, `0.066` at step 25 and `0.452` at step 50 on the real corpus;
`audit_07_init_pathology.json`'s `P23_escape_and_fix` records
`steps_to_first_spike_in_layer1: 50`; and §5.4 carries the project's standing
ruling that the pathology *"self-corrects within ~50 of 20 000 steps"* and that
changing an initialisation because a resolved diagnostic looked bad *"would be
exactly the kind of unforced modification that makes a baseline non-comparable"*.
**Nothing here overturns §2.6's PASS, and nothing here says the gate is in the
wrong place.** What §2.2, §2.6 and §6 omit is that the arm's one modulated layer
spends its first ~50 steps in that regime and that the screen measures exactly
it.

**A third thing the same artifact shows is not an open question, and the tree
already answers it.** The screen reports a nonzero `k_gate` gradient at **layer
0** (0.041118) for a design that says layer 0 carries no gate — because
`build_screen_model` (`scripts/audit/09_gradient_reachability.py:800-806`)
allocates every *grafted* candidate's parameters `for _ in range(cfg.n_layers)`
and grafts the scan onto all of them, which is why `c24_gated` shows layer-0
entries where the real-arch `nm_*` candidates read `present: false`. So layer 0's
column describes a topology the arm will not have — a property of the screening
harness, visible in the committed script. It does not disturb the layer-1
numbers. **The gate that will hold `gateddecay.py` to one modulated layer is
`G2`**, whose closed form is §3's `735,437 + 2d·(K−1)` — not `G4`, which asserts
*nesting* rather than topology. `model.py:1640` already names that same mechanism
for the modulator's unused layer-0 parameters: *"the parameters still exist, and
**G2** asserts they reach nothing."*

**4. §8's cost sentence misreads `EXP_002`'s artifact, and the honest replacement
is parity rather than a saving.** §8 states that *"`EXP_002` N1 priced a
per-channel decay's **systems** cost at a relative step time of 1.0117"*. In
`exp_002_candidate_neuron_cost.json`, `1.0117` is N1's `kernels_per_timestep`
and is **identical to the N0 baseline's** — both `kernels = 259`, and
259/256 = 1.01171875. It is the one-kernel-per-timestep result, not a step-time
ratio, and both committed write-ups label that column "Kernels/timestep". N1's
committed `ms_vs_baseline` is **0.972** — 8.898 ms against 9.158 — **but that is
a median of three**, with N1's own `ms_spread_rel` at **7.3 %** and N0's whole
range `[9.096, 9.279]` sitting inside N1's `[8.725, 9.378]`. Per `CLAUDE.md`'s
standing rule that a figure compared against a threshold carries an interval and
its denominator, the reading is **parity**: not 1.17 % slower, and not faster.
**It moves nothing** — 10 runs × 398.9 s is 1.108 h, ×1.0117 is 1.121 h, ×0.972
is 1.077 h, and §8's *"~1.1 GPU-h"* survives either way — and it does not touch
§8 item 5's requirement that the budget be pre-flighted rather than quoted.

**What this note does not do.** It edits no row, resolves nothing, ranks no arm
and proposes no bar. It does **not** edit `EXP_024_gated_decay.md`: that log is
**committed** at `aff0d49`, and a commit — not a stamp — is what fixes it;
`EXP_024` has no run manifest yet (its own §8 item 7 is an unmet entry
condition), so `verify_prereg_hash.py` reports `NO MANIFEST` for it and `git
diff` is the only check available for that log today. The correction goes in
prose here because the decision it bears on — the R10 tax — is §8 item 1's, and
routes to open decision **#4**.

**It takes no position on that item**, and the four corrections cut in different
directions rather than one:

* **1 is unambiguously good news for the record**, and its only bearing on
  `EXP_024` is that the project's provenance is in better shape than rev 14 said.
* **2 cuts both ways.** It weakens §0's *necessity* argument — the statistic has
  moved further, on more arms, than §4's generalisation allows — while also
  weakening H1's *bar*, since three committed readings clear 2σ and two of them
  sit on arms that lost bits.
* **3 is a documentation gap, not bad news for the arm.** The layer is silent for
  ~50 of 20 000 steps by this project's own measurement, §5.4 already ruled that
  this does not warrant changing an initialisation, and the PASS stands. What is
  missing is any mention of it in §2.2, §2.6 or §6.
* **4 corrects a label without moving the budget.**

**The point of the note is that whichever way that decision goes, it goes on the
real record.** Corrections 3 and 4 were found only because the census in
correction 2 was built and run; none of the four required a GPU.

**The three places elsewhere in this report that read "revision 14" have moved
with this note**: line 3's masthead, §22's changelog table, and §1's "Decisions
open" cell, which has carried a per-revision sentence since rev 11. **A masthead
blockquote is added as a fourth**, which is not one of those three and is
recorded here for that reason: it is the convention every revision since 7 has
followed, and rev 14's changelog row (b) is where the one revision that skipped
it is named. **The gate table in §9 is not re-run**, on the precedent revs 5 and
7 both state explicitly: this revision closes no experiment, adopts nothing, and
touches no code that any row in it describes. `ruff check .` passes and the CPU
suite is **540 passed, 1 skipped** on this tree; both tools this note relies on
were re-run from the committed artifacts and
`docs/reports/data/audit_11_beyond_horizon_census.json` reproduces **byte-for-byte**.

---

## 9. Gates, at the close of this revision

**Every row was re-run for rev 3 on 2026-08-04**, and the suite and the two
`EXP_012` rows again for **rev 4 on 2026-08-05**. Rev 2's rows were re-run from
the clean tree at `dcc8f82`; rev 3 re-ran them with `EXP_011` in the tree and
added four rows; rev 4 adds three. **Rev 16 adds six `EXP_025` rows and re-runs none of
the others** — they describe experiments and code this revision does not touch,
and re-running them would be re-reporting rather than gating. What *was* run on
this tree, at this revision: `ruff check .` clean, the CPU subset **557 passed,
1 skipped**, and every gate in the six rows below, on all twelve runs.

| Gate | State |
|---|---|
| **`EXP_025` G5 — the frozen recipe, read back rather than asserted** *(new at rev 16)* | ten recipe fields — `lr`, `lr_schedule`, `max_steps`, `warmup_steps`, `grad_clip`, `weight_decay`, `beta1`, `beta2`, `batch_size`, `seq_len` — read out of **every one of the twelve** runs' own `config.json` and compared to `02a_phase2_spec.md`'s frozen values: **12/12 all_frozen**, including all six at 5.0M. **This is the gate that keeps decision #11 unengaged**, and it is checkable rather than a promise in prose |
| **`EXP_025` K1a — config sameness with ZERO exemptions** *(new at rev 16)* | every run diffed field-by-field against **this experiment's own** `e25_snn_d512_s0`, not a committed reference, so all twelve came off one tree and `absent_from_reference` is **empty on all twelve**. Only `arch`, `d_model`, `seed`, `run_name` differ. **The strongest form of this gate the project has run.** K1b keeps the committed comparison against `snn_beta0.5_s0` and exempts the 24 post-dating fields **by name**, not by count |
| **`EXP_025` G3/G4/K3** *(new at rev 16)* | realised parameter counts **735,437 / 737,485 / 4,997,099 / 5,003,023, all exact**; peak VRAM reproduces the cost calibration **to four decimals** against a 3.0 GiB alarm; wall-clock ratios **0.933–1.043**, so A4's 1.5× halt was never approached |
| **`EXP_025` — four runs reproduce committed runs BITWISE** *(new at rev 16)* | `e25_detach_d512_s0/1/2` and `e25_snn_d512_s0` against `EXP_017`'s and `EXP_014`'s: **2.1176906639749444 / 2.1129641077769445 / 2.1153212748345145 / 2.257471102255831**, bit-for-bit. Unpredicted, carried as an observation. The mirror of §13.7's **G1 failure**, and the empirical justification for K1b's exemption list |
| **`EXP_025` pre-registration hash** *(new at rev 16)* | **VERIFIED BY COMMIT** via the blob at `d9f8d45`, `before == after` across both rungs, and `live_file_agrees_above_results` **True**. It read **False** first — appending §9 had also rewritten a sentence above the results heading, which the recipe does not permit — so **the tool rev 15 repaired caught a live violation within a day of shipping** |
| **`EXP_025` resolver — statistics that can fail** *(new at rev 16)* | `fisher_exact_one_sided`, `wilson_ci`, `tost` and `non_beyond_share` are module-level with **17 tests**, verified to fail before trusted: **9 mutations, 9 caught**. Three defects were found this way, all decision #9's class — and one of them had already reached a **committed** pre-registration, which is the first time that has happened |
| Test suite | **394 passed**, 1 warning, 47 s (full: `pytest`, GPU + corpus). Rev 4's 280, fully accounted rather than assumed: **+85** `tests/test_snnchat.py` (the chat track's own suite, committed `31e49fa` between rev 4 and rev 6 and untouched by this revision), **+23** `tests/test_noise.py` (`EXP_013`'s gates), **+6** `tests/test_grad_clip.py` (decision #7's fix, new at this revision) — 280 + 85 + 23 + 6 = 394 exactly. Fast subset (`pytest -m "not cuda and not data"`, no GPU/corpus): **256 passed, 1 skipped, 8 s** |
| **`EXP_013` G1 — the nesting** | `arch="noise"` at `noise_amp = 0` reproduces `SpikingCharLM` **bitwise, forward and backward** (`torch.equal` on logits and every gradient) — licenses scoring the five committed baseline seeds as the noise-off control instead of training three more |
| **`EXP_013` G3 — the draw is centred and correctly scaled** | on a large draw, sample mean within 4 se of 0 and sample sd within 1 % of `amp`; the mutation leg (an uncentred draw) is shown to fail it |
| **`EXP_013` G4/G5 — resumability and independence** | the noise at `(seed, step)` is a pure function of `(seed, step)` — refilling twice at the same step gives bitwise-identical buffers; the salted generator is not reproducible from the data sampler's stream at the same step |
| **`EXP_013` F1** | **9 / 9**, residuals 1.45e-09 – 1.18e-08, `f1_failures` empty — every gap evaluation's test leg reproduces the committed 2.26969 / 2.25311 |
| **`EXP_013` K1** | every noise run's `config.json` against `snn_beta0.5_s0`'s, field-by-field: only `arch`, `noise_amp`, `run_name` differ; `beta_slow`, `w_init`, `mu_init`, `thr_log_init`, `noise_p` excused as absent-from-reference at the `Config` default |
| **`EXP_013` calibration self-check** | `013_calibrate_noise_scale.py` replays `_CharLMStack.forward`'s loop by hand to observe `cur`, and every replay's logits are asserted bit-identical to `model(idx)`'s: **25 / 25** |
| **`EXP_013` pre-registration hash** | §0–§8 verified byte-identical to what was hashed before the first training run: SHA-256 prefix `285eb661…82ef8d`. No hypothesis, bar or decision cell moved after the numbers arrived |
| **Decision #7 — `_clip_grad_norm_fp64` reproduces the fix, not just the intent** | `tests/test_grad_clip.py`: a synthetic 5.46e31 gradient (`EXP_009` §9.4's own number) — stock `clip_grad_norm_` returns `inf` and zeroes every gradient exactly; the replacement returns a finite ~5.46e31 and rescales instead. Static AST checks (mirroring `test_step_body_has_no_python_control_flow`) confirm no host sync and no value-dependent branch in the new helper. The real R5 capture gate, unchanged: **10 / 10** |
| **`EXP_012` S3 — the environment has not drifted** | `008_chase_fold.py` re-run to a scratch path reproduces the committed `exp_008_fold_residual.json` at **0 differences**, every field, bit-for-bit. Run **before** any cross-arm comparison was formed, because the comparison is the experiment |
| **`EXP_012` S1 — the probe reproduces the committed kernel** | leg 3 must recompute the membrane recursion to observe `u` at all, so its spikes are asserted **bit-identical** (`torch.equal`) to `lif_scan` / `twocomp_scan` on every run and both layers: **10/10 pass**. Guarded in the suite by `tests/test_fold_gap_probe.py` across three firing rates, with **two mutation legs** — resetting `vs`, and a soft reset — so the check has been seen to fail |
| **`EXP_012` sets no tolerance** | asserted structurally: `resolve()` reads its bars from constants transcribed at the top of the file, and the artifact carries `"sets_no_tolerance": true` and `"decision_6": "UNTOUCHED AND OPEN"`. No fold in this experiment is gated |
| **`EXP_011` C5/K5 — Identity 1 on the two-compartment neuron** | float64, at a nowhere-zero per-channel `θ`, against a reference transcribed from `EXP_011` §1.1 rather than from the source: **spikes exact, membranes ≤1e-12** on four (`beta_f`, `thr`) regimes. A mutation leg asserts the check can fail. Run as the driver's entry condition **before** any training step |
| **`EXP_011` K4 — the composed arm nests the adopted arm** | at `θ = 0`, forward and backward **bitwise** (`== 0.0`, not a tolerance) against `TwoCompartmentCharLM` |
| **`EXP_011` K1 — the arm is the adopted arm plus one thing** | every composed run's `config.json` field-by-field against **`twocomp_s0`** (not the Phase-2 baseline): only `arch`, `run_name`, `seed`, `thr_log_init` differ; `mu_init` excused at the `Config` default and recorded in the manifest |
| **Resolvers reproduce their own artifacts** | `010_phase4_arm_results.py`: **0 differences** against the committed `exp_007_009_arm_results.json`. `011_compose_results.py` re-run to a scratch path: **0 differences** against `exp_011_arm_results.json` — every reference, decomposition, prediction and both GPU fold evaluations, bit-for-bit |
| Committed references recomputed | the resolver's own guard: baseline **2.25311**, twocomp **2.11869**, GRU **1.76741** carried, each recomputed from per-run `final_test.json` and asserted against the figure the reports quote (worst residual **1.5e-5**, tolerance 5e-4) |
| Phase-3 §7.2 reachability screen | re-run: `thr` **PASS**, `ctl_live` **PASS**, `ctl_dead` SADDLE and `ctl_faint` FAINT as expected, all three self-tests held, and it reproduces `exp_004_gradient_reachability.json` at **0.00e+00** worst relative residual |
| R10 gradient gates | untouched. `EXP_007`/`EXP_008`'s arms call the committed `lif_scan`; **`EXP_011`'s calls the committed `twocomp_scan`** and adds no hand-written backward. Independently, §10.4's pass 3 replayed `compose_s0`'s failing step on the **eager** path and got an identical NaN, which exonerates the fused two-compartment backward rather than merely leaving it unaccused |
| `EXP_001` F1 (probe reproduces committed bpc) | **23 / 23** in the `EXP_007`–`EXP_009` artifact; **17 / 17** in `exp_011_memory_horizon.json` (worst residual 1.3e-08), `f1_failures` empty in both |
| Mutation campaign | none run during any training run; lockfile checked before each, including all three `EXP_011` runs |
| Sequential GPU use | the `EXP_011` driver blocks on each child in turn, and the horizon probe and divergence chase ran only **after** the last training run exited — so no wall-clock figure in §10.2 overlaps other GPU work |
| Working tree | rev 2 modified exactly one file and committed nothing. **Rev 3 is different and says so:** it adds `EXP_011`'s pre-registration, arm, gate, driver, resolver and chase, a registry entry in `001_memory_horizon.py`, four artifacts under `docs/reports/data/`, three run directories, and this file. Every piece was **committed before the numbers it produces were read** — the pre-registration at `f0eebe4`, the arm and its gate at `ca94f01`, the driver at `5e3d5e5`, the resolver at `bc0dd08` |

**Phase 4 remains open.** Rev 1 closed `EXP_007`, `EXP_008` and `EXP_009`; rev 2
synthesised them; **rev 3 closes `EXP_011` and resolves the one scheduling
decision that authorised it.** It authorises no phase, adopts no arm, and
redefines no threshold — and seven decisions in §8 are untouched.

**Rev 4 closes `EXP_012`**, which trains nothing, adopts nothing and sets no
tolerance. It leaves all seven decisions open and all seven rows unedited; what
it changes is that **#6 is now a decision that can be made** — §10.5's withheld
evidence is released to it, and the shape of the choice is stated in §8's rev-4
note. It also **declines to apply** the §6.5 scope qualifier its own evidence
supports, because the trigger it pre-registered for that correction did not fire.

**Rev 5 runs no experiment, adds no gate row and changes no number.** It records
Elliot's sign-off of Phase-3 §9 — decision **#1**, the first of the seven to be
decided — in that file's §9 and changelog and in §8 row 1 here. **Six decisions
remain open**, Phase 4 remains open, no arm is adopted, no tolerance is set and no
phase is authorised. The gate table above was **not** re-run for rev 5, and is not
claimed to have been: rev 5 touches two report files and no code, no artifact and
no run directory, so every row still describes the tree at rev 4's commit
`677793d`.

**Rev 6 closes `EXP_013`, decides #5 and #7, and re-runs the full suite for
real** — both marker sets above are literal `pytest` output, not carried forward.
It adopts the threshold arm standalone (#5) and fixes the gradient clip (#7); it
does **not** adopt the composed arm, does **not** set a fold-in tolerance (#6
unchanged), and does **not** authorise Phase 5 (#4 unchanged, still "not yet").
The mutation campaign was **not** re-run for rev 6 — nothing in `snn/*.py`'s
kernels changed, only `Trainer._step_body`'s clip call, which the mutation
campaign does not target — and this row says so rather than implying otherwise.

**Rev 7 did not re-run this table either**, for the same reason rev 5 did not: it
touches report and README prose and no code. Its own commit-time gates are
recorded in §16's rev-7 entry (394 passed, 44.9 s, on the committed tree).

**Rev 8 re-ran the suite for real and it is literal output: `ruff check .` clean,
fast subset `258 passed, 1 skipped` in 8.2 s, full suite `396 passed` in 38.7 s
on this GPU.** The full count moves 394 → 396 because `EXP_014` added two ladder
tests; nothing was removed. **The gate rows above are not re-run and are not
claimed to have been** — rev 8's only code change is a *docstring* in
`src/snn/train.py` (§8's rev-8 note), which alters no kernel, no captured graph
and no numeric path, so every row still describes the tree it was measured on.
**The mutation campaign was not re-run**, for the same reason rev 6 gave.

`EXP_014`'s own gates are recorded in its log rather than duplicated here: K1
3/3, G3 3/3 exact against `spiking_param_count`, G4 peak 1.6434 GiB against a
3.0 alarm on a 7.96 GiB card, G2 capture succeeded at every width with no
`cuda_graph=False` fallback, K2 no mutation lockfile before any of the six child
processes, K3 realised/projected 0.949 / 0.946 / 0.925, F1 zero failures, and the
pre-registration hash unchanged across the ladder. **G1 failed** (§13.5).

**Rev 9 re-ran the suite: `ruff` clean, fast subset 267 passed / 1 skipped, full
suite green on this GPU.** The fast count moves 258 → 267 for
`tests/test_calibrate_cost.py`, which pins the shared calibration tool's argv and
parameter closed forms, the arms' parameter match at `d = 1481` *and its
`O(d)`-vs-`O(d²)` mechanism*, and the two completion-test defects `EXP_015`
§9.10 found. **The gate rows above are not re-run** — rev 9 changes no kernel, no
captured graph and no numeric path in `src/snn`.

`EXP_015`'s own gates: K1 3/3 with `absent_from_reference_and_left_at_default`
**empty** at every leg (§14's stricter K1, possible because `scale_d1481_s0`
carries every field of the current `Config`), G3 3/3 exact, G4 3/3 against a
**4.0 GiB** alarm derived from a measured 2.9767 GiB peak rather than inherited,
K2 clean, K3 0.9262 / 0.9147 / 0.9720, F1 zero failures, pre-registration hash
unchanged. **No G1 gate exists**, and that is deliberate: every leg is anchored
to `scale_d1481_s0` on this tree, and entry condition E1 discharged the tree
question by measurement — the calibration's `snn` leg reproduced `EXP_014`'s own
calibration to full precision eleven commits later.

---

## 10. `EXP_011` — the two best arms compose, and the seed that died proving something else

Pre-registered in `experiments/logs/EXP_011_composition.md` and closed the same
day. It is §7.1's rank-1 candidate and the measurement `EXP_004` §10.11 item 2
asked for in advance. **Three training runs, ~0.51 GPU-hours**, plus ~0.35 for
the 17-checkpoint horizon probe and the divergence chase.

### 10.1 The arm, and why its parameters add nothing

`arch="twocomp_threshold"`: the adopted neuron with `snn.prescan.threshold_gain`
on the current before the scan. `EXP_011` §1.1 derives, **before any code
existed**, that `EXP_008`'s Identity 1 extends to the two-compartment neuron —
both poles scale linearly with the current, the mix is linear, and the fast reset
is multiplicative, so `cur → g·cur` gives `v → g·v` and firing at `g·v ≥ thr` is
firing at `v ≥ thr/g`. The reset-shielded slow pole carries the induction
untouched, which is the step that had to be redone rather than inherited.

`tests/test_compose_equivalence.py` checks it in float64 against a
per-channel-threshold reference transcribed from the derivation rather than from
the source, at a nowhere-zero `θ` so it cannot pass vacuously: **spikes exact,
membranes ≤1e-12.** So the composed arm's 1 024 extra parameters are **exactly
redundant** over the adopted arm — +0.14 % in parameters, **+0.00 % in
function-space dimension** — and whatever it buys is an optimisation effect.
*(Rev 6, same caveat as §6.3: this is a statement about function class, not about
whether the model has room for more parameters at all — `EXP_013` §12.2 measures
that separately, and it is not "no".)*

### 10.2 The scoreboard

| | n | test bpc, carried | vs adopted | se | wall-clock | peak VRAM |
|---|---:|---:|---:|---:|---:|---:|
| Phase-2 baseline | 5 | 2.25311 | — | — | 1.00× | 0.609 GiB |
| learned threshold | 3 | 2.19416 | — | — | 1.12× | 0.734 GiB |
| two-compartment *(adopted)* | 7 | 2.11869 | — | — | 1.32× | 0.980 GiB |
| **composed** | **2** | **2.08326 ± 0.00049** | **−0.03543** | **−28.7** | **1.41×** | 1.105 GiB |
| GRU anchor *(violates I5)* | 3 | 1.76741 | — | — | 1.11× | 0.864 GiB |

Wall-clock is against `snn_beta0.5_s0`'s 398.9 s, the standing denominator;
against the adopted arm's own three-seed 525.3 s the composed arm is **1.074×**.
Seeds: 2.08291 and 2.08360. **`compose_s0` diverged and is excluded, not
replaced** (§10.4). The gap to the GRU anchor narrows 0.35128 → 0.31585.

**60.1 % of the threshold arm's 0.05896 carried over.** `EXP_004` §10.11 item 2
predicted the numbers "would not add twice" and they did not — the arm's
pre-registered C3 (sub-additivity) held. The design cannot separate 60 % from
55 % or 70 %, and `EXP_011` §3.1 said so before it ran.

### 10.3 The gain is entirely within-reach, which confirms §10.6 by relocating it

`EXP_004` §10.3's decomposition, **against the adopted arm**:

| component | bpc | share |
|---|---:|---:|
| Zero context | **+0.00050** | **1.4 %** |
| Within the baseline's reach | **+0.03597** | **102.4 %** |
| Beyond the horizon | −0.00134 | −3.8 % |

**`EXP_004` §10.6 argued the two mechanisms are the same mechanism at zero
context.** If so, stacking a real per-channel threshold on the adopted neuron
should buy nothing there. It buys **0.00050 bpc — 1.4 %, and below the ~2e-3
noise floor §6.5 measured.** The prediction is confirmed about as cleanly as this
project can confirm anything.

And that is *why* the composition is worth having. Its entire gain is the
**within-reach** component — the one the adopted arm made **worse** (§6.2:
−48.4 % inside the baseline's own reach) and the one the threshold arm was the
first arm in this project to move (+0.0288 against the baseline). Composed, it
moves that component by **+0.03597 — more than it managed against the baseline**,
because the adopted arm had left more room there to recover.

Against the Phase-2 baseline the composed arm decomposes as zero **+0.05113**
(32.2 %), within-reach **−0.02386** (−15.0 %), beyond **+0.13139** (82.8 %): it
still gives back short-range ground, but **less than the adopted arm alone**.

**The two arms compose because they act on different components than their
headline numbers advertise.** Phase 4's ranked list is built on the assumption
that arms attack the component they were ranked for; this is the second time in
Phase 4 that assumption has failed (§6.7 was the first), and it is the finding
here that most affects Phase 5.

Supporting rows: **§6.2 against the adopted arm — 0 / 128 contexts significantly
worse**, worst delta −0.0005 at c = 0; against the baseline 3 / 128, where the
adopted arm alone is 5 / 128. **Median horizon 48** against the adopted arm's 47 —
the composition is not a horizon effect and was not expected to be. **The fold
(C4) executes**: residuals 1.05e-06 and 1.89e-07, 738 509 → 737 485 parameters,
loading into `arch="twocomp"` — so **the composed arm costs the adopted arm
exactly at inference.**

### 10.4 `compose_s0` — chased, and decision #7 would not have saved it

The seed trained cleanly to step 17 500 and was NaN by 17 750.
`scripts/exp/011_chase_compose_divergence.py` reproduces it **deterministically
at step 17 598** from the last healthy checkpoint:

| | `EXP_009`'s dead seed | **`compose_s0`** |
|---|---|---|
| forward / loss at the bad step | finite | **finite** (logits 223.0, loss 1.4622) |
| gradients before it | finite, largest **5.46e31** | **1.2e-2 – 3.1e-2, no run-up** |
| ‖g‖ fp32 vs fp64 | **inf vs finite** — the clip overflowed | **both NaN; equal at every healthy step** |
| eager path, same step | not tested | **identical NaN — kernel exonerated** |
| origin | backward, layer 0 + embedding | backward, layer 0 + embedding |

**This is not `EXP_009` §9.4's mechanism, and that matters for decision #7.** That
seed died to an fp32 overflow inside `clip_grad_norm_` on a genuinely enormous
gradient. This one died with gradients of 0.03, every quantity in the range of the
six preceding healthy steps, and parameters at its last healthy checkpoint sitting
**inside** the range the two surviving seeds occupy — `compose_s2` reached a
*larger* maximum gain (7.71 vs 5.40) and lived. Pass 3 replays the step on the
eager path and gets an identical NaN in the identical six tensors, which
**exonerates the hand-written two-compartment backward and rules out an R10
defect.**

Two divergences that look identical in a training log have different causes.
Phase 4 now has **two unexplained ones**, and the fix named in §8 row 7 addresses
only the first.

### 10.5 What this does not settle

**The verdict is provisional at n = 2**, and C2 (cross-seed sd ≤ 0.005) is the
prediction that costs most: 0.00049 says the two surviving seeds agreed and
little else, and it is **not** evidence of stability against whatever killed seed
0. Not settled either: which arithmetic in layer 0's backward makes the NaN
(bounded, not localised); **why this fold is ~1000× tighter than `EXP_008` W1's
1.86e-3** — same identity, same class of fp32 reordering, and this experiment
does not explain it, so it is **not** offered as evidence for decision #6 *(rev 4:
`EXP_012` explains it in §11, and the number is released to #6 there; this
paragraph records what `EXP_011` did and did not settle and is left as it was)*;
and
whether either parent's initialisation is right for the composition, which is one
unswept point in a two-dimensional space.

---

## 11. `EXP_012` — the fold gap is a degeneracy, and the rule that measured it misfired

Pre-registered in `experiments/logs/EXP_012_fold_gap.md` and closed the same day.
It is the second of the two things §10.5 left standing, and the one this report
had **refused to use** as evidence for decision #6 until something explained it.
**No training, ~0.2 GPU-hours**, five existing checkpoints.

### 11.1 Two explanations were removed before anything was measured

Read from the committed source and recorded in that file's §0, so that neither
could later be reported as a discovery: `fold_into_spiking_state_dict` and
`fold_into_twocomp_state_dict` have the **same six-line body**, and `fold_check`
and `fold_check_composed` compute the **same quantity by the same code path**.
Neither the fold's arithmetic nor the metric can be the explanation, which leaves
exactly two candidates — **the neuron** and **the operating point** — and the
experiment is built to separate them.

### 11.2 The controls held: it is not the algebra and not the perturbation

| | `EXP_008` arm (→ `snn`) | composed arm (→ `twocomp`) |
|---|---|---|
| leg C — the algebra, fp64 fold and fp64 GEMM | 6.9 / 6.8 / 8.3 fp64 eps | **3.6 / 7.4 fp64 eps** |
| leg A — the perturbation actually injected | 5.8 / 6.1 / 8.3 fp32 eps | **6.4 / 8.2 fp32 eps** |

Identity 2 is exact on **both** neurons, and the fold perturbs both currents by
the same relative amount to within the spread of `EXP_008`'s own three seeds.
**Whatever separates the two folds happens after the GEMM.**

### 11.3 Where the gap is, and what sets it

| | layer-0 flips | fraction | 512-window bpc residual |
|---|---:|---:|---:|
| `threshold_s0/s1/s2` | 5 339 / 2 684 / 2 851 | 3.18e-04 / 1.60e-04 / 1.70e-04 | 2.03e-03 / 6.23e-04 / 1.62e-03 |
| `compose_s1/s2` | **1 / 2** | **5.96e-08 / 1.19e-07** | **1.44e-05 / 9.29e-06** |

**1 342–5 339× fewer flips.** And the flip count tracks the probability mass the
decision variable puts *at* the threshold — `P(|u| < 1e-6)` is **5.5e-04 – 7.4e-04**
on the baseline neuron against **1.2e-07 – 3.0e-07** on the two-compartment one,
a factor of **1 843–6 218**, while the membrane's overall spread differs by only
~1.19× (sd 5.42 vs 4.55). **It is not a wider distribution. It is a concentration
at the threshold**, and the ladder's shape says so: the composed arm's
`P(|u| < h)` is **linear in `h`** across four decades — a smooth density — while
the baseline's grows only **~9× across the same four decades**.

**Three ratios, three different quantities — and this report quotes all three
rather than the most impressive.** Flip count 1 342–5 339×; 512-window bpc
residual **43–219×**; the committed full-test residuals (`EXP_008` W1 against
`EXP_011` C4) **515× at closest approach, 1 770× headline-to-headline and 9 841×
at the extremes**. The chain from a flipped spike to a bpc is signed and
cancels, and it cancels differently at different sample sizes. **"~1000×" is a
fair headline for the committed numbers and is not the flip-count ratio.**

**`EXP_012` §1's derivation was wrong in both of its terms, in opposite
directions, and the log says so.** It predicted the unreset slow pole would make
the perturbation *larger*; `E|du|` is ~100× **smaller**. It predicted the density
would fall because the spread widens; **the spread does not widen.** The `du`
result is a **consequence of the flip count, not a cause of it** — a flipped
spike moves the LIF's state by an O(1) reset rather than an O(ulp) rounding, so
every later step of that channel carries a macroscopic perturbation. Quoting
`du/dcur` as the mechanism would have inverted the causation.

### 11.4 The decision rule fired on a leg with no power, and is reported unrepaired

The cross-over drives **both** neurons with the **identical** current and
perturbation. On `threshold_s0`'s current — where the LIF's 5 339 flips sit far
above the resolution floor — the two-compartment neuron flips **2**:

| | thr | firing rate | flips |
|---|---:|---:|---:|
| LIF | 1.0000 | 0.344880 | **5 339** |
| LIF, rate-matched to twocomp | 0.3256 | 0.398950 | **205** |
| two-compartment | 1.0000 | 0.398939 | **2** |
| two-compartment, rate-matched to LIF | 1.8887 | 0.344880 | **1** |

**102.5× at matched rate 0.399, 5 339× at matched rate 0.345.** But Y5 required
*every* cross-over leg to reach 10×, and the second leg — on `compose_s1`'s
current — compares **0 flips against 1**. Ratio 0.0. **Y5 FAILED**, and
`EXP_012` §4's truth table therefore returns *"an operating-point effect, not a
neuron effect"*.

**That verdict is not this report's finding, and the reason is not
after-the-fact.** `EXP_012`'s own limitation 2 and failure mode P3 stated in
advance that a zero count would be reported as *"below what one batch resolves"*,
not as a number — and Y5 was then specified without that guard. **The rule is
recorded as it fired and is not rewritten**, because rewriting a bar after seeing
which way it fell is precisely what `EXP_005` §9.4, `EXP_008` §9.4 and `EXP_011`
§0 established the practice of refusing. The 102.5× is a **marker with no verdict
attached**; the corrected wording for Y5 is referred, not adopted.

### 11.5 Three of six predictions failed, and one failure is the mechanism showing

Y1, Y2, Y3 held. Y4 failed (34.0× amplification against a `< 5×` prediction — but
that is 34/1 and 68/2, and **an amplification ratio built on a denominator of 1
is not a measurement of one**, so it is not offered as evidence in the other
direction either). Y5 failed as above.

**Y6 — the flip model — failed, and the shape of its failure is informative.**
Predicted ÷ measured is **1.11 / 1.19 / 1.17** on the baseline's layer 1 and
**0.97 / 0.84** on the composed arm's, but **3.50 / 8.82 / 5.69** on the
baseline's layer 0. The model assumes a *smooth* density; it works wherever §11.3
says the density is smooth and over-predicts by 3.5–8.8× exactly where §11.6 says
there is a point mass. Per `EXP_012` §3, **no quantitative mechanistic claim is
made** from Y3 or Y5 on the back of a failed Y6.

### 11.6 What the concentration is made of — post-hoc, and labelled so

**This subsection is post-hoc**: the hypothesis was formed after reading §11.3's
ladder, and `scripts/exp/012_posthoc_pileup.py` carries the same label.
`EXP_008` §9.5 is the precedent for a chase that follows a result.

Sites with `|u| < 1e-6` at layer 0:

| | count | distinct `u` values | repeats each | channels |
|---|---:|---:|---:|---:|
| `threshold_s0/s1/s2` | 10 962 / 12 436 / 9 213 | **20 / 23 / 23** | **548 / 541 / 401** | 60 / 68 / 69 of 512 |
| `compose_s1/s2` | 5 / 2 | 4 / 2 | 1.2 / 1.0 | 5 / 2 of 512 |

**The baseline's ten thousand near-threshold sites are twenty numbers.** Layer 0's
input is one of only **205 embedding rows**, so `cur` takes 205 values per
channel; the hard reset sets `v` to **exactly 0**, so the step after a spike has
`v_pre = cur`, drawn from that small set. A channel whose `cur` lands within an
ulp of `thr` replays the *same* `u` every time that token follows a spike —
87–91 % of these sites are post-spike steps. The two-compartment neuron never
resets `vs`, so `u = vf + w·vs` carries a continuously-varying history and lands
on a repeated value 1.0–1.2 times.

**Post-spike enrichment does not discriminate** — 2.61× on the baseline, 2.11× on
the composed arm — so an analysis that stopped there would have found a false
positive. **The repeat count is the discriminator**, and it was named as the
falsifier in the script's docstring before it ran.

### 11.7 What this releases, and what it refuses to apply

**Released to decision #6.** §10.5 withheld `EXP_011` C4's 1.05e-06 as evidence
"until understood". It is understood, and it is released. What it establishes is
**not** a tolerance — `EXP_012` proposes none by design — but the **shape of the
choice**: the same identity, through the same fold, costs **2.0e-03 bpc on the
baseline neuron and 1.4e-05 on the adopted one**, for a structural reason. **A
single W1-style constant will be far too loose for one arm or far too tight for
the other.**

**Refused.** The §6.5 scope qualifier — that the ~2e-3 floor belongs to the
baseline LIF rather than to "this architecture" — is supported by §11.3 and
§11.6 and is **not applied**, because `EXP_012` §4 pre-registered it behind
`Y2 ∧ Y5` and Y5 failed. §6.5 stands as written and carries a referral note.
**Adopting a correction whose own gate did not open, seconds after watching it
not open, is the failure mode this protocol exists to prevent** — and it would
have been an easy one to commit, because the correction is almost certainly
right.

**Still unexplained after this revision:** which arithmetic in layer 0's backward
makes `compose_s0`'s NaN (§10.4), and whether any of §11 survives at another
scale or under another reparameterisation — quantisation and weight
normalisation are the two §6.5's floor was written to cover, and `EXP_012`
measured one fold on one architecture pair.

---

## 12. `EXP_013` — the model does not overfit, and the noise arm that tested it is a null

Pre-registered in `experiments/logs/EXP_013_noise_injection.md`, 2026-08-05,
after two calibration measurements that read only the committed Phase-2 baseline
and no arm. Closed the same day. **Not on `03_phase3_candidates.md`'s
18-candidate list** — the file opens a candidate rather than closing one, and
says so in its own §0. Ten training runs (nine plus one re-run, whose two results
are bit-identical — an operational note, not a finding), fourteen gap
evaluations, nine horizon probes, two calibrations: **~1.9 GPU-hours** (its own
§0 pre-registration estimated ~1.6; the measured total is what §1's table above
uses).

### 12.1 What was measured, and why

Inject a zero-mean, unit-variance-scaled Bernoulli background current into every
layer's input, during training only:

```
cur'_{t,c} = cur_{t,c} + amp · (B_{t,c} − p)/sqrt(p(1−p)),   B ~ Bernoulli(p)
```

The `− p` term is derived, not chosen: an uncentred draw adds a constant to every
channel's current, which is `EXP_008`'s Identity 1 in its uniform form — a
threshold shift, not a regulariser — and `EXP_008` already measured a per-channel
version of exactly that mechanism at 0.0590 bpc, an order of magnitude larger
than the whole quantity this experiment targets. Centring removes the confound
at the cost of one kernel. At inference the term is absent; the arm costs the
Phase-2 baseline exactly, with nothing to fold.

Built on the Phase-2 baseline rather than the adopted neuron, for the same
reason `EXP_007`/`EXP_008` were: the noise-off control already exists at n = 5,
and its noise floor is measured on those same five seeds and nothing else. The
composition onto `arch="twocomp"` is the obvious follow-up and is not run here.

### 12.2 The calibration is the finding: this model does not overfit

Measured **before the arm existed** (§2.2), on the five committed baseline
seeds, train-slice bpc against test bpc under both committed protocols:

| protocol | train bpc | test bpc | **gap** | cross-seed sd | bar (2σ_gap) |
|---|---:|---:|---:|---:|---:|
| fresh | 2.26375 | 2.26969 | **+0.00594** | 0.00268 | 0.00536 |
| carried | 2.25491 | 2.25311 | **−0.00180** | 0.00306 | 0.00612 |

Both test legs reproduce the committed Phase-2 baseline figures to five decimals
— `EXP_001`'s F1 discipline applied to a new instrument. **Under `carried` there
is no gap at all** (three of five seeds negative). **Under `fresh` there is a
small, consistent gap — all five seeds positive, 4.96 se from zero — and it is
still smaller than this project's own 2σ = 0.00922 adoption bar**, and only ~3×
the reparameterisation noise floor §6.5 measured.

**Why: 735,437 parameters against ~655M training characters and a 90M-character
split — ~7.3 epochs, ~122 characters per parameter.** This project's problem is
underfitting (2.25311 bpc against the GRU anchor's 1.76741), and a regulariser
is a treatment for the opposite disease. This is the measurement §7 item 6 and
§7.1 rank 2 act on — it does not by itself say anything a noise arm bought, only
that there was very little room for one to work in.

### 12.3 The noise arm itself: costs bits at every amplitude, buys none

Three amplitudes, geometric ×4 (0.1, 0.4, 1.6 — 3.2–113.6 % of `sd(cur)`,
measured per-layer in §2.1), n = 3 seeds each, against the baseline's n = 5:

| arm | n | test bpc, carried | Δ vs baseline | se | cross-seed sd | wall-clock | peak VRAM |
|---|---:|---:|---:|---:|---:|---:|---:|
| Phase-2 baseline | 5 | 2.25311 | — | — | 0.00461 | 1.00× | 0.609 GiB |
| noise, amp 0.1 | 3 | 2.25731 | +0.00419 | +1.2 | 0.00167 | 1.068× | 0.796 GiB |
| noise, amp 0.4 | 3 | 2.30290 | +0.04978 | +14.8 | 0.00325 | 1.089× | 0.796 GiB |
| noise, amp 1.6 | 3 | 2.89287 | +0.63975 | +190.0 | 0.01817 | 1.073× | 0.796 GiB |

Every amplitude costs bits; none buys any. The memory horizon does not move
(6–7 against the baseline's 5/5 seeds at exactly 7), so whatever the noise does,
it is not a horizon effect.

### 12.4 N1 holds and must not be believed — the same defect as `EXP_012`'s Y5

The primary pre-registered rule — `Δgap ≤ −2σ_gap` under `fresh` — fired at two
of three amplitudes. The decomposition says why that must not be read as
regularisation:

| amp | Δ train bpc | Δ test bpc | **Δgap** | resolved? |
|---:|---:|---:|---:|---|
| 0.1 | +0.00468 | +0.00387 | −0.00081 | no — below the ~2e-3 floor |
| 0.4 | +0.05878 | +0.05013 | **−0.00865** | yes |
| 1.6 | +0.67083 | +0.64130 | **−0.02953** | yes |

`Δgap = −(Δtrain − Δtest)` exactly. **The gap closed because the model got
worse at the training text faster than at the test text — a uniformly damaged
model, not a better-generalising one.** At the two amplitudes where N1 fires,
test bpc is 14.8 and 190 se worse than baseline. The bar is satisfiable by
damage because it constrained a difference without constraining what its
subtrahend was allowed to do — precisely `EXP_012`'s Y5 (§11.4), independently,
in a different experiment, six days later. **The bar is reported as it fired and
is not rewritten**, per this project's four standing precedents; the corrected
form — count a reduction only when `Δtest ≤ 0` — is referred, not applied
(§12.6).

**Read against the substance rather than the fired cell, the verdict is the
pre-registration's own second cell: NULL, reported as "there was no gap to
close" — never as "noise injection does not regularise."** N3 held (amp 0.1
inside the null band both ways). Not recommended for adoption; no adoption is
made here.

### 12.5 Predictions, and the rest of the ladder

| | verdict | note |
|---|---|---|
| N1 | **HELD** | wrong reason — §12.4 |
| N2 | **HELD** | monotone in amplitude; +0.640 at the top against a 0.00922 bar |
| N3 | **HELD** | amp 0.1 inside ±0.00922 both ways |
| N4 | **FAILED** | layer 1 monotone 0.323→0.658; layer 0 not (0.337/0.351/0.339/0.361) — no mechanism claimed |
| N5 | **FAILED as written**, held as glossed | the pre-registration stated opposite-direction hypotheses as if complementary; both are the same measurement with opposite signs, so this is one event, not two |
| N6 | **FAILED, both directions** | cross-seed sd 0.36× baseline at amp 0.1, 3.94× at amp 1.6 — σ is a property of the hyperparameter here, the same lesson `EXP_005` established across architectures |

Three of six failed and a fourth is repudiated by its own drafting error — after
`EXP_012`'s three-of-six, the source log calls this "a consistent picture: this
project's pre-registered bars are failing often enough to be doing work."

### 12.6 What this releases, and what it does not

**Released to §7's ranking** (§6.7 item 3, §7 items 5–6, §7.1 rank 2 — this
report's own use of the finding, cross-referenced rather than repeated here).
The transferable result is §12.2: this model is not overfitting, which reframes
regularisation-flavoured candidates downward and capacity-flavoured ones upward,
and specifically un-blocks the parameter-scaling pilot.

**Referred, not applied.** N1's own construction — a gap-reduction bar without a
`Δtest ≤ 0` guard — is referred rather than self-corrected, per `EXP_005` §9.4,
`EXP_008` §9.4, `EXP_011` §0 and `EXP_012` §11.4's precedent. The corrected form
is decision #9's substance (§8), whose protocol-level rule is separately already
ratified (`CONTRIBUTING.md` §2, `51b44c6`).

**Not answered, and not claimed to be.** Whether noise injection regularises a
model that *does* overfit (this one doesn't — nothing here bears on that);
whether `p` matters (held at 0.1, unswept); whether any of this transfers to the
adopted two-compartment neuron (not run there, and `EXP_005`/`EXP_012` are two
separate demonstrations that this project's statistics do not transfer between
those two neurons); and whether the arm belongs on `03_phase3_candidates.md`'s
ranked list at all (§0's own question, Elliot's to rule on).

---

## 13. `EXP_014` — width buys more than every arm this phase built, and the gate expected to pass is the one that failed

Pre-registered in `experiments/logs/EXP_014_parameter_scaling.md`, 2026-08-08,
and **committed before its first run** (`d59f98e`) — the guarantee `EXP_013`
could only discharge with a hash. Closed the same day. Three training runs, three
evaluations, one horizon sweep over eight checkpoints, one gap sweep over four,
a pre-flight calibration at six widths run twice, and a five-measurement chase:
**~1.5 GPU-hours** against §7.1's ranked **0.35**.

This section is the largest single effect in Phase 4 and the one that should be
read most carefully, because **it closes a ranked candidate (#11) while adopting
nothing and refuting nothing.** §13.6 is the part that bears on what runs next.

### 13.1 The entry condition was a measurement, and it changed the design twice

`03_phase3_candidates.md` §6.3 #11 priced this pilot at **0.35 GPU-hours**, a
figure rev 7 had already flagged as descending from a cost model
`02_baseline_report.md` §6.2 formally withdrew. Rather than pre-register against
it, `EXP_014` §2.1 ran a 400-step probe at six widths first
(`scripts/exp/014_calibrate_cost.py`, `docs/reports/data/exp_014_cost_calibration.json`),
and **committed that measurement before the pre-registration cited it**
(`41a6c07`).

| `d_model` | params | s/step | × d=512 | projected 20k | peak VRAM |
|---:|---:|---:|---:|---:|---:|
| 512 | 735,437 | 0.01861 | 1.000 | 398.9 s | 0.6089 GiB |
| 1020 | 2,501,245 | 0.05486 | 2.948 | 1,175.8 s | 1.1418 GiB |
| 1481 | 4,997,099 | 0.11050 | 5.938 | 2,368.3 s | 1.6434 GiB |

**Cost is ~1.10 GPU-hours, 3.1× the ranked 0.35** — and the probe earned its
keep twice over beyond that. It found no GEMM alignment cliff at the odd width
(`d = 1481` is 2.9 % slower than `d = 1480` and 1.1 % slower than `d = 1488`, so
the awkward parameter-matched width costs nothing structural), and it found the
**gradient clip binding at the wide rungs and never at the anchor** — 1 of 17
logged steps at `d = 1020`, 3 of 17 at `d = 1481`, against 0 of 17 at `d = 512`.
That last observation is what made G1 (§13.5) worth gating on at all.

### 13.2 The ladder: −0.25238 bpc at 54.75 transferred σ

Seed 0, `arch="snn"`, the frozen Phase-2 recipe, `d_model` the only field
changed — K1 passed 3/3, with `run_name` the only other permitted difference.

| rung | params | test bpc fresh | test bpc **carried** | wall-clock | peak VRAM | firing rate |
|---|---:|---:|---:|---:|---:|---|
| `scale_d512_s0` | 735,437 | 2.27387 | 2.25747 | 378.6 s | 0.6089 GiB | 0.339 / 0.320 |
| `scale_d1020_s0` | 2,501,245 | 2.08760 | **2.06947** | 1,112.4 s | 1.1418 GiB | 0.293 / 0.267 |
| `scale_d1481_s0` | 4,997,099 | 2.02007 | **2.00073** | 2,191.4 s | 1.6434 GiB | 0.209 / 0.185 |

**S1 held at 54.75 σ** against a pre-registered 10 σ bar: −0.25238 bpc against
the 5-seed anchor's 2.25311, a margin 5.5× the bar. The σ used is **transferred**
from `d = 512` and §3.0 of the log says so before the run; even the 3.94×
inflation `EXP_013` §9.5 measured leaves this at 13.9 σ. Against the alternative
anchor the log pre-registered for a G1 failure — `scale_d512_s0`'s own 2.25747 —
it is **−0.25674**, so **the verdict does not depend on which anchor is used**,
which matters because G1 did fail.

**S2 held, and returns diminish per octave.** 512 → 1020 is **−0.18800** over
1.766 octaves (−0.1064/octave); 1020 → 1481 is **−0.06874** over 0.998 octaves
(−0.0689/octave). The second step buys 65 % of the first. Both resolve by more
than an order of magnitude over the 2 σ = 0.00922 gate. **The rungs are not
equally spaced and no functional form is fitted** — §5 item 5 forbids quoting
a power law from two unequal steps, and none is quoted here.

**Unanticipated, and no bar predicted it: firing rates fall monotonically with
width** — 0.339/0.320 → 0.293/0.267 → 0.209/0.185. The network gets *sparser* as
it gets wider. Nothing in this project had measured that, and it bears directly
on the energy argument, which has always assumed sparsity is a fixed property of
the neuron rather than something width moves.

### 13.3 Width is per-step capacity, and buys essentially no reach

S4's three-way cut, on the absolute context curve at the `EXP_004` §10.3 basis:

| rung | horizon | total Δ | zero-context | within-reach | beyond-horizon | near share |
|---|---:|---:|---:|---:|---:|---:|
| d=512 | 7 | −0.00421 | +0.00712 | −0.01154 | +0.00021 | not scored |
| d=1020 | 8 | +0.18212 | +0.09679 | +0.08212 | +0.00321 | **98.2 %** |
| d=1481 | 8 | +0.24961 | +0.10926 | **+0.12538** | +0.01497 | **94.0 %** |

**94–98 % of the gain is at or inside the baseline's own reach; 6 % or less lies
beyond it.** The memory horizon moves **7 → 8 → 8**, inside the pre-registered
ceiling where 6, 7 and 8 all count as "did not move" — it sits *at* the ceiling
at both wide rungs, and that is reported rather than smoothed, because at n = 1
the horizon has no error bar and this is a marker, not a demonstration that
width bought one character of reach.

**`03_phase3_candidates.md` §5 assumed exactly this** when it assigned the
zero-context and within-reach components to *"the readout, normalisation, width,
scale"*. This is the first measurement of that assumption, and it holds.

### 13.4 The underfitting regime ends between 2.5M and 5M parameters

S3, measured with `scripts/exp/013_generalisation_gap.py` **unchanged and
reused**, not reimplemented:

| rung | gap (fresh) | Δ vs baseline's +0.00594 | Δtest (fresh) | counts? |
|---|---:|---:|---:|---|
| d=512 | +0.00592 | −0.00002 | +0.00418 | no — below what this instrument resolves |
| d=1020 | +0.03524 | +0.02930 | **−0.18209** | **yes** |
| d=1481 | +0.05509 | +0.04915 | **−0.24962** | **yes** |

**The `Δtest ≤ 0` guard did real work, and this is its first application.** It is
the correction `EXP_013`'s N1 forced and §8 row #9 tracks: a bar on a *difference*
must constrain both of its terms. `EXP_013`'s N1 fired on models whose test bpc
was 14.8 and 190 se *worse*; here the gap grows while the test leg improves by
0.18 and 0.25 bpc. That is memorisation appearing **alongside** a real gain, not
damage — the distinction the guard exists to draw, drawn for the first time.

**So `EXP_013` §2.2's "the model does not overfit" is confirmed as a statement
about 735K parameters, and is now bounded above**: between 36 and 18 characters
per parameter, the regime changes.

**The strongest check in the experiment was not designed as one.** The `d = 512`
leg's fresh gap is **+0.00592** against `EXP_013` §2.2's **+0.00594** — five
different runs, a different tree, a different day, agreeing to **2e-05** on a
quantity whose 2 σ bar is 0.00536.

### 13.5 G1 failed, and the chase closed it — decision #7's clip is not numerically inert

**G1 existed only because an audit asked what had changed underneath the
baseline.** Before rev 6, `src/snn/train.py` had exactly one commit (`b5f71a9`,
Phase 2), so **every committed figure in this project was trained with the stock
`torch.nn.utils.clip_grad_norm_`**, and decision #7 replaced it afterwards. The
gate: retrain `d = 512, seed 0` and require **bitwise** equality with
`snn_beta0.5_s0`. It was expected to pass, because the clip had never been
observed to fire at this width.

**It failed:** fresh 2.273872258304913 against 2.2718995629892054 (**+1.973e-03**),
carried 2.257471102255831 against 2.2554769668134713 (**+1.994e-03**).

The log first reported the mechanism as *unestablished*, because the obvious
explanation failed its own reproduction on CPU. **`CONTRIBUTING.md` §4 says chase
a residual to its origin, and the chase was then run rather than referred.** Five
measurements, in the order made (`EXP_014` §9.12):

| test | result |
|---|---|
| today's tree against itself | **bitwise identical**, 81/81 → a code change, not nondeterminism |
| clip firings in 20,000 steps, at `log_every=1` | exactly **three** — steps 32/33/34, norms 1.0329/1.1198/1.0957 |
| a worktree at `b5f71a9` against the committed baseline | **bitwise identical**, 0/81 → the baseline was never irreproducible |
| stock clip monkeypatched into today's tree, 250 steps | **zero differing losses** against the Phase-2 tree |
| fp64 clip on today's tree | first differing loss at step **47** — the step at which the trees diverge |

**Mechanism.** At the three clipped steps the fp32 and fp64 sums-of-squares
differ by ~1 ulp, so the clip coefficients differ in their last bits. The
perturbation stays below the loss's own fp32 resolution for thirteen steps,
surfaces at step 47, and amplifies over the remaining ~19,950 — landing on
§6.5's ~2e-3 reparameterisation noise floor **with a mechanism attached rather
than as a coincidence.** All three firings fall *between* the runs' 250-step
logging cadence, which is why §13.2's own diagnostic reads 0 of 81 while §13.1's
25-step calibration saw it.

**A false negative worth keeping in the record.** The substitution test was first
run for **40 steps** and read as exonerating the clip. It stopped **seven steps
short** of the divergence. The probe was demonstrably *sensitive* — the two
paths' reported grad norms differ from step 0 — which is exactly what made the
false negative convincing. **The horizon of a null result is part of the null
result**, and `EXP_007`'s V4 is the same shape.

**What this changes, and what it does not.** Decision #7's fix is **not
numerically inert** — true of the *gradients*, false of the *trajectory* —
and `src/snn/train.py`'s docstring said in as many words that it was. **Rev 8
corrects that docstring** and adds decision **#10** (§8) for the rest.
It invalidates **no committed verdict**: every arm was trained on the old tree,
so the arms remain mutually comparable, and the adopted effects (0.03–0.06 bpc)
are 15–30× this delta. What it removes is *"reproduce a committed number
bitwise"* as an available gate. **Nothing here argues for reverting the fix** —
it repairs a failure that silently zeroes every subsequent update.

### 13.6 What this does to §7.1's premise — and what it must not be read as

At 4,997,099 parameters the **plain Phase-2 baseline LIF** scores 2.00073
carried. Every committed arm sits at ~735K:

| | params | test bpc carried | vs baseline |
|---|---:|---:|---:|
| Phase-2 baseline LIF | 735,437 | 2.25311 (n=5) | — |
| learned threshold *(adopted)* | 736,461 | 2.19416 (n=3) | −0.05896 |
| two-compartment *(adopted)* | 737,485 | 2.11869 (n=7) | −0.13442 |
| composed *(provisional n=2)* | 738,509 | 2.08326 | −0.16985 |
| **plain LIF, wide** | **4,997,099** | **2.00073 (n=1)** | **−0.25238** |
| GRU anchor *(violates I5)* | 738,019 | 1.76741 (n=3) | −0.48570 |

**This refutes no arm, and it is important that it is not read as doing so.** The
comparison is **not parameter-matched** — 6.8× — so it does not show width is a
better *mechanism* than the two-compartment neuron or the threshold. What it
shows is that **this phase spent its GPU-hours on a 0.03–0.17 bpc axis while an
unmeasured 0.25 bpc axis sat beside it.** That is §7 item 6's referral
vindicated, not any arm refuted.

**The sharpest version of it:** the within-reach component — 51.8 % of the
remaining gap, the thing §7 item 2 called *"still barely attacked"* — moves by
**+0.12538** here. The best any architectural arm managed against the baseline
was the threshold arm's **+0.0288**.

**And the sobering version, which belongs in the same paragraph:** at 6.8× the
parameters the SNN is **still 0.2333 bpc behind a GRU at 1×**, which closes
**52.0 %** of the 735K gap.

> **Rev 9 corrects this paragraph twice.** As written at rev 8 it said width
> closed *"about a quarter of the remaining gap"*; the figure is **52.0 %**, and
> the error was reading the absolute 0.25238 bpc gain as a fraction. **And the
> corrected figure is still the wrong comparison** — it prices a 5M spiking model
> against a **738K** anchor. `EXP_015` §14.2 trained the anchor at matched size:
> the gap is **0.45374**, so width closes **6.6 %**, not 52 %. The sentence above
> is left standing with this note beneath it, because it is what rev 8 asserted.

**What it does not answer, quoted from §9.10 rather than paraphrased.** It does
not say what size Phase 5 should run — *"wrong neuron, and `EXP_005`/`EXP_012`
are two demonstrations that statistics do not transfer between these two
neurons"*. σ at the new widths is unmeasured, and S3 supplies a reason to expect
it to differ. Whether the gap keeps opening past 5M, where the optimum sits, and
**whether any of this survives on `twocomp`** are all open.

### 13.7 What this releases, and what it refers

**Released.** Ranked candidate #11 is closed (§1). §7.1's rank 2 is done, at a
**measured** 1.1 GPU-hours of training against a ranked 0.35 — the first row in
that table ever measured, and it came in at 3.1×. §7.1's rev-8 note records what
that implies for the other five rows.

**Referred to Elliot, and not decided here** (`EXP_014` §10, verbatim in
substance):

1. **The consequence of §13.5.** Whether the project re-baselines, and whether a
   standing rule is wanted that a `src/snn` change altering a committed number
   must be recorded as such before merge. **This is new decision #10 (§8).**
   *(Rev 8: Elliot has ruled the first half — the project does **not**
   re-baseline; every new experiment trains its own anchor on the current tree.
   The standing-rule half stays open, and #10's row carries both.)*
2. **Whether the top rung is reseeded to n = 3** before any number from it is
   quoted as a result — ~1.2 GPU-hours at 2,191 s per seed.
3. **Whether §7.1's ranking survives this.** Offered in §7.1's rev-8 note, not
   applied there.
4. **Whether the `twocomp` ladder is now worth its ~1.2–1.8 GPU-hours**, since
   the sizing answer Phase 5 actually needs is on that neuron and this pilot
   cannot supply it.
5. **§7.1's GPU-h column generally**, since every remaining row descends from the
   cost model §13.1 withdrew.

**Not claimed, structurally.** `exp_014_scaling_results.json` carries
`adopts_nothing: true` and `ranks_nothing: true` as fields rather than as prose,
and the resolver was committed (`44051fb`) **before any rung's bpc was read**.
The pre-registration's §0–§8 hash to
`7703641f…8469f22b` both before the first run and after the last.

---

## 14. `EXP_015` — the adopted arm does not train at width, and the anchor scales with it

Pre-registered in `experiments/logs/EXP_015_architecture_at_width.md`, **committed
before the first run** (`d4469d6`), hash-verified after the last. Closed
2026-08-10, ~2.1 GPU-hours. It exists because `EXP_014` §9.10 stated its own
limit — the width comparison was **not parameter-matched** and ran on the **wrong
neuron** — and §10 item 4 referred exactly this.

**The design was cheap for a structural reason.** At `d_model = 1481` every arm
lands within **0.18 %** of the plain LIF's 4,997,099 parameters, because each
arm's extras are one or two vectors per layer — `O(d)` against an `O(d²)` stack —
so the mismatch falls with width (0.28 % for `twocomp` at `d = 512`, 0.12 % at
1481). Width-matching *is* parameter-matching here, no arm needs a derived width,
and the GRU is matched **by construction** since `GRUCharLM` sizes its hidden
width from `spiking_param_count`. `tests/test_calibrate_cost.py` pins both the
tolerance and the mechanism.

### 14.1 The scoreboard

| leg | arch | params | test bpc carried | wall-clock | peak VRAM | outcome |
|---|---|---:|---:|---:|---:|---|
| control | `snn` | 4,997,099 | **2.00073** | 2,191.4 s | 1.6434 GiB | trains (`EXP_014`) |
| A | `twocomp` | 5,003,023 | **NaN** | 2,432.2 s | 2.6138 GiB | **diverged, step 5138** |
| B | `twocomp_threshold` | 5,005,985 | **NaN** | 2,551.0 s | 2.9767 GiB | **diverged, step ~2000** |
| anchor | `gru` | 4,997,829 | **1.54699** | 1,481.3 s | 2.1293 GiB | trains |

**Both arms containing the two-compartment neuron failed to train at 5.0M
parameters under the frozen Phase-2 recipe.** The plain LIF at the identical
width, seed, recipe and tree does not, and neither does the GRU. At 735K these
same arms trained across 7 and 2 seeds. **This is the finding that bears on
Phase 5**, whose plan was to scale the adopted arm.

### 14.2 The anchor at matched size — the leg that carried no bar

P3 was pre-registered with **no bar**, because the GRU violates I5 and is not a
competitor. It was trained so the project's gap statement would stop comparing a
5M spiking model against a 738K GRU:

| comparison | gap | fraction of the 735K gap closed |
|---|---:|---:|
| 735K vs 735K | 0.48570 | — |
| 5M SNN vs **735K** GRU | 0.23332 | 52.0 % |
| **5M vs 5M** | **0.45374** | **6.6 %** |

Width bought the spiking model **−0.25238** and the anchor **−0.22042**. Both
improve; the distance barely moves. **Capacity is not what separates this
architecture from its anchor**, and §13.6's reading — accurate as far as it went —
does not survive matching parameters on both sides.

### 14.3 Where the difference actually lives: reach, not capacity

| arm | params | horizon (2σ) |
|---|---:|---:|
| `snn_beta0.5` (n=5) | 735,437 | 7 |
| `scale_d1481` | 4,997,099 | **8** |
| `arch_gru_d1481` | 4,997,829 | **97** |
| `twocomp` *(committed)* | 737,485 | 47 |
| `gru` *(committed)* | 738,019 | 57–60 |

**Width buys the anchor reach and buys the spiking model none.** The same
statement without any horizon definition, from the absolute per-context curves:

| context c | 0 | 2 | 4 | 8 | 16 | 64 |
|---|---:|---:|---:|---:|---:|---:|
| `arch_gru_d1481` | 4.0014 | 2.7013 | 2.1781 | 1.8766 | 1.7222 | 1.6126 |
| `scale_d1481` | 4.0524 | 2.8761 | 2.2261 | 2.0291 | 2.0194 | 2.0201 |

**At zero context the two 5M models are nearly equal — 0.051 apart.** By `c = 8`
it is 0.152; by `c = 64` it is **0.408**, and the spiking curve has been flat
since `c ≈ 8`. **The entire matched-size gap is a failure to use context, not a
failure to model the marginal** — and the only spiking arm that ever bought reach
(`twocomp`, horizon 47) is the arm that will not train at width.

*Guard, pre-registered:* n = 1 at both wide points and the lattice is coarse, so
these are **markers with no verdict**. 8 against 97 is why they are reported.

### 14.4 P1 and P2 fired, and said the wrong thing — for the third time

The resolver returns **UNRESOLVED** for both dead legs, and its decision cell
recommends seeds. **Both are wrong for what happened**: §3.0 defines UNRESOLVED
as a statement about *power*, and these runs produced no number at all; and the
divergence is a **deterministic** function of (checkpoint, step), so a second seed
measures a different trajectory rather than the same one again.

`CONTRIBUTING.md` §3 says report the verdict and refer the correction, so P1 and
P2 stand as **UNRESOLVED, as fired**. This is the third pre-registered rule in
Phase 4 to fire and mislead — after `EXP_012`'s Y5 and `EXP_013`'s N1 — which is
decision **#9**'s substance a third time. The corrected form is a
**DIVERGED / NO RESULT** verdict distinct from UNRESOLVED; **referred, not
applied.**

### 14.5 The divergence, and it is `compose_s0`'s

`011_chase_compose_divergence.py` **reused unchanged** but for a `--tag`, replayed
from `ckpt_best.pt` and **reproduced the NaN deterministically at step 5138** —
138 steps after the checkpoint, inside the 250-step logging cadence that made the
run's own log read 5250.

| | `compose_s0` (735K) | `arch_twocomp_d1481_s0` (5.0M) |
|---|---|---|
| step | 17,598 | **5,138** |
| origin | backward | **backward** |
| forward, loss | finite | **finite** (1.4198) |
| fp32 vs fp64 norm | equal | **equal** |
| eager path | identical NaN | **identical NaN** |
| non-finite grads | embed, layers.0.{weight,bias}, w.0, beta_s_raw.0, thr_log.0 | **the same five, minus thr_log** |

**The same mechanism, and the first time on the plain adopted arm.** Not
`EXP_009`'s clip overflow (norms agree), not a fused-kernel defect (R10 clean,
`fused_only_nonfinite_params` empty), and not a gradient explosion — the largest
gradient in the six steps before death is **2.0e-02** and the clip never fires in
the whole run. Always layer 0 and the embedding; never layer 1.

> **CORRECTION, rev 10: "not a gradient explosion" is wrong, and this revision's
> own artifact already contained the refutation.** `EXP_016` §9.3 measures the
> backward at step 5138 boundary by boundary: the cotangent entering layer 1's
> scan is **1.4639e−04** and the one leaving layer 0's scan is **6.1173e+37** —
> **an amplification of 10^41.62 inside a single backward pass**, on both dispatch
> paths, with the loss bit-identical. It is a gradient explosion, and a large one.
>
> The paragraph above is left standing per `CONTRIBUTING.md` §3, and the two facts
> it cites are both true. **Neither supports the conclusion, because both are
> measured *between* optimiser steps.** "The largest gradient in the six steps
> before death is 2.0e-02" describes steps 5132–5137; the explosion happens
> *inside* the backward at 5138. And **"the clip never fires" is a consequence of
> the explosion, not evidence against it** — `clip_grad_norm_` reads the
> accumulated gradient after `backward()` returns, by which point the tensor holds
> NaN and the norm is NaN. A run that died of a 10^41.6 explosion logs
> `max_grad_norm = 0.663` and `n_logged_steps_over_clip = 0` for exactly that
> reason.
>
> `exp_015_divergence.json`, committed at rev 9, records `layers.1.weight` at
> **1.676e+15** and the eager replay's largest finite gradient at **7.79e+29** at
> that step. Those numbers were in the artifact and were not read against the
> sentence.
>
> **"Never layer 1" stands as written about non-finiteness and misleads about
> magnitude.** Layer 1 carries 1.10e+15 where it carried 1.46e−04 one boundary
> earlier. It is not spared; it has not yet run out of exponent. See §15.

**A hypothesis was stated and falsified.** The two chases' state magnitudes differ
~10×, so: does the two-compartment membrane grow with width where the plain LIF's
does not? Measured on four committed checkpoints — layer 1 grows **13.30×** for
the plain LIF and **14.96×** for `twocomp`. The *surviving* arm grows more, and a
plain LIF carrying `|state|` up to 556 trains 20,000 steps without incident.
**State magnitude is not the discriminator.** What survives is a lead only: at
layer 0, the locus of every non-finite gradient, `twocomp` is **2.84×** the plain
LIF at the same width, carried by the reset-shielded slow pole. Separately,
"the slow pole ran away" was removed without spending GPU — `beta_s =
sigmoid(beta_s_raw)` is bounded in (0, 1) and weight decay pulls it toward *more*
damping.

### 14.6 What this releases, and what it refers

**Released.** `EXP_014` §10 item 4 is answered: the `twocomp` ladder was worth
running, and what it bought was not a bpc. §13.6's gap statement is superseded by
§14.2's matched one. `EXP_011` §10.4's mechanism is reproducible in 138 steps
from a committed checkpoint instead of 17,598.

**Referred** (`EXP_015` §10): the recipe question, which is new decision **#11**;
whether the arm is retried at an intermediate width (`d = 1020`, one run, where
`EXP_014` already has a control); the verdict-vocabulary correction (#9 again);
what the now-cheap third divergence is worth chasing to; and whether the anchor's
result restates what Phase 5 is for.

**Closes no ranked candidate.** `EXP_015` is not on `03_phase3_candidates.md`
§6.3's list, like `EXP_012` and `EXP_013` before it.

---

## 15. `EXP_016` — the plain neuron's backward provably cannot explode, and the adopted one's demonstrably does

`EXP_015` closed with the third divergence bounded to "layer 0's backward" and
its arithmetic unknown, and `CONTRIBUTING.md` §4 makes chasing that a standing
obligation rather than a ranked candidate. `EXP_016` took it. **It trains
nothing, changes no hyperparameter, adopts nothing and touches no file under
`src/snn/`** — five forward passes over checkpoints already on disk, plus two
instrumented replays of one step. ~0.8 GPU-hours, of which ~0.37 bought no
number (§15.5).

### 15.1 A closed form, derived before any measurement

Both neurons thread an adjoint backwards over all `L = 256` timesteps with
`grad_v_prev = beta * (grad_v_next * dv + grad_spike * sg)`, so **`beta * dv` is
the per-step multiplier of the homogeneous reverse recurrence**. The two neurons
write `dv` differently, and the difference is the whole result:

| | `dv` | |
|---|---|---|
| plain LIF (`kernels.py:129`) | `(1 − s) − v_pre · sg(v_pre − thr)` | **the same variable** in the multiplier and inside the surrogate |
| `twocomp` (`twocomp.py:179-180`) | `(1 − sh) − vf · sgd(v_pre − thr)`, `vf = v_pre − w·vs` | **different variables** — the multiplier is the *fast* membrane, the surrogate's argument is the *mixed* one |

The LIF's form is self-limiting: `sg` decays as `1/(1 + (πα/2·x)²)` exactly as
fast as `v_pre` grows. At the frozen constants (`α = 2`, `thr = 1`, `β = 0.5`)
the extrema are at `x* = −1 ± √(1 + 1/π²)` and

> **`max|β·dv| = 0.5123596 < 1`, for every finite membrane, at every width.**

**The plain LIF's backward-through-time is a strict contraction by construction**,
and nothing in that bound depends on `d_model`. This is the first available
explanation of why `scale_d1481_s0` trains 20,000 steps cleanly.

In `twocomp` the self-limiting is broken by the reset-shielded compartment:
`vf·sgd = v_pre·sgd − w·vs·sgd`, whose second term is bounded by nothing because
`vs` is never reset. The crossover is at **`|w·vs| ≈ 1`**.

### 15.2 Measured: the LIF saturates its bound and stops; the arm does not

Over 48,529,408 sites per layer, on five committed checkpoints:

| arm | `d` | layer 0 `max\|g\|` | ÷ LIF bound | `frac(\|g\| > 1)` | `max\|w·vs\|` |
|---|---:|---:|---:|---:|---:|
| `snn` | 512 | **0.51236** | 1.00 | **0** | — |
| `snn` | 1020 | **0.51236** | 1.00 | **0** | — |
| `snn` | 1481 | **0.51236** | 1.00 | **0** | — |
| `twocomp` | 512 | 2.87015 | 5.60 | 5.80e-05 | 22.91 |
| `twocomp` | **1481** | **8.95778** | **17.48** | 5.97e-04 | 279.66 |

The three `snn` legs return the derived bound to five decimals at every width and
every layer, and **not one site of 48.5 million exceeds it** — which is the
sentence that answers `EXP_015`'s constraint (f). T1 (the probe's own validation
against the closed form) and T2 (local scans bitwise equal to the committed eager
references) both hold.

### 15.3 The bar that mattered was placed on the wrong quantity, and said so in advance

**P3 asked whether the product over all 256 steps could reach the observed
7.79e29, and it cannot: 0 of 189,568 chains expand net.** REFUTED AS SUFFICIENT,
decisively — which is what a one-sided bar is for.

The quantity was wrong. `grad_spike` is injected at *every* timestep, so the
cotangent reaching a parameter is a **sum over injection points**, each traversing
its own window; summing over the whole unroll averages the mechanism away by
construction. **`EXP_016` §4.0 and §6 item 4 both say `Σ_t log|g_t|` "is NOT the
gradient", and the bar was placed on it anyway.** That is `EXP_012` Y5's shape,
`EXP_013` N1's and `EXP_015` P1/P2's — **the fourth occurrence in Phase 4**, and
the first authored in full knowledge of the previous three. Reported unrepaired
per `CONTRIBUTING.md` §3.

**P4 also failed, in the informative direction**: layer **1** carries the heavier
per-step tail (3.31e-03 against layer 0's 5.97e-04), the opposite of the
prediction, and §15.4 explains why that is consistent with every non-finite
gradient landing at layer 0.

### 15.4 Leg B: the explosion, measured boundary by boundary

At step 5138 on the dying run, **both dispatch paths**, loss reproducing
`1.4197630882263184` **bitwise**:

| boundary, backward order | `max\|finite\|` | non-finite |
|---|---:|---|
| entering layer 1's scan | **1.4639e−04** | none |
| leaving layer 1's scan | **1.1027e+15** | none |
| entering layer 0's scan | 5.6801e+14 | none |
| leaving layer 0's scan | **6.1173e+37** | **140 NaN + 1 inf** |

**Layer 1's recursion amplifies by 10^18.88, layer 0's by a further 10^23.03 —
10^41.62 end to end, inside one backward pass.** The eager path gives the
identical boundaries to seven significant figures, so **R10 separation now holds
one level deeper than `EXP_015` could take it**: the fused kernel is exonerated
at the boundary, not merely at the parameters.

`EXP_015` §14.5's "never layer 1" is therefore true of *non-finiteness* and
misleading about *magnitude* — layer 1 has not been spared, it has not yet run
out of exponent. **And "not a gradient explosion" is wrong**; see the rev-10
correction under §14.5.

### 15.5 POST-HOC: it is the run of consecutive steps, and Leg A read the wrong batch

Two post-hoc measurements, labelled so everywhere, on `012_posthoc_pileup.py`'s
precedent; neither amends a bar. The first replaces P3's whole-unroll product
with the **maximum contiguous window**. The second re-runs both probes on **the
batch the run actually died on**, because Leg A read batch 0 — **a defect in Leg
A's design, not only in P3's quantity**:

| arm | layer | longest expanding **run** (batch 0 → **5138**) | window gain |
|---|---:|---:|---:|
| `snn`, all three widths | 0, 1 | **0 → 0** | **10^0.00 → 10^0.00** |
| `twocomp` 735K | 0 | 2 → 3 | 10^0.66 → 10^0.66 |
| `twocomp` 5.0M | 0 | 8 → **85** | 10^1.73 → **10^16.00** |

**On the fatal batch, layer 0's reverse recurrence expands for 85 consecutive
timesteps.** At 735K the same arm manages 3. The plain LIF is at exactly zero in
every cell — it cannot be otherwise.

**What is not accounted for is stated rather than absorbed** (`EXP_016` §9.5): the
window supplies 10^16.00 and the coupling term at most ~10^4.4 against a measured
10^23.03, leaving **~2.6 orders of magnitude unresolved**. Candidates named, none
measured: the window and injection maxima need not share a site; `1/(1 − β_s)`
understates an accumulator whose input is itself growing; and the probe's weights
are 138 steps stale. **No claim is made that this is the whole cause.**

### 15.6 What this releases, and what it refers

**Released.** `EXP_011` §10.4's "which arithmetic is unknown and is the next thing
to chase" is answered as far as this evidence goes: the first non-finite value is
made inside layer 0's reverse recursion, on both paths, and the plain LIF's
immunity is a theorem rather than an observation. §14.5's explosion claim is
corrected in place beneath the original.

**Referred** (`EXP_016` §10), and the first is the one that matters: **decision
#11's *form*.** `dv` is a function of `(v_pre, vs, w_c, α, thr, β_f)` and contains
**no learning rate, no schedule, no weight decay and no gradient clip** — and
§15.4 shows the clip cannot act on this failure at all, because it reads a norm
that is already NaN. So "may the frozen recipe be re-derived at a new size?" may
be the wrong question to rule on. **Nothing is proposed and no term is changed**;
#11's row is not edited.

Also referred: whether a remedy is a ranked candidate (anything bounding the
recurrence is a **new arm** with its own σ); the residual 2.6 orders (~2 GPU-min);
the verdict vocabulary a fourth time (#9); and whether `EXP_014`'s
falling-firing-rate-with-width finding connects to `max|w·vs|` rising with it.

**Closes no ranked candidate**, like `EXP_012`, `EXP_013` and `EXP_015` before it.

---

## 16. `EXP_017` — the bound holds, the arm trains at width, and the control mostly died

`EXP_016` §10 item 2 referred the remedy as a new arm: *"anything that bounds the
recurrence — clamping the slow compartment, resetting it, normalising `cur`, or
truncating BPTT — is a **new arm**, needs its own pre-registration and its own σ,
and would change the adopted arm's function class."* `EXP_017` is that arm, and it
is the one member of that list that **does not change the function class**,
because it changes nothing in the forward at all.

Pre-registered at `50b3904` before any leg ran; ~2.0 GPU-hours;
`experiments/logs/EXP_017_bounded_reset_jacobian.md`. **Nothing is adopted,
nothing is ranked, no hyperparameter is changed, and no row of §8 is edited.**

### 16.1 The change is one token, and the derivation is a stronger theorem than §15's

Detaching the fast pole's reset factor sends `∂vf_out/∂v` to zero. Writing the
per-step homogeneous backward as a 2×2 matrix on `(grad_vf_next, grad_vs_next)`:

```
adopted    J = [[ beta_f·dv          ,  0      ]     dv = (1 − sh) − vf·sgd
                [ −beta_s·w·sgd·vf   ,  beta_s ]]

this arm   J = [[ beta_f·(1 − s)     ,  0      ]     dv = 1 − s
                [ 0                  ,  beta_s ]]
```

`vf` appears **twice** in the adopted arm's Jacobian — in the diagonal *and* in
the off-diagonal coupling that injects the fast compartment's cotangent into the
slow one — and the detach removes it from both at once. So:

* `(1 − s) ∈ {0, 1}`, hence **`|beta_f·dv| ≤ beta_f = 0.5`** — for every finite
  membrane, every `w`, every `vs`, every batch, every channel, every step and
  **every width**. That is **strictly tighter than the plain LIF's own
  0.5123596** (§15.1);
* the off-diagonal is **exactly zero**, so the Jacobian is **diagonal** — no
  non-normal transient either, which the plain LIF gets for free only because it
  has one compartment;
* the slow pole's rate is `beta_s`, **unchanged**. The mechanism the arm exists
  for is not touched.

**And the forward is not merely equivalent — it is the same compiled kernel.**
`snn/twocomp_detach.py` imports `twocomp_forward_kernel` rather than declaring a
forward source. So the arm adds **no parameter** (parameter-*identical*, not
parameter-matched), no state, no function and **no inference cost**: a checkpoint
trained here loads into `arch="twocomp"` and evaluates **bit-identically**,
asserted at `== 0.0`. §6.3 and §10.1 both had to report a fold-in *residual*
(1.9e-03 and 1.05e-06 bpc); **this arm folds nothing, so decision #6's tolerance
question does not arise for it at all.**

**The cost is a biased gradient and the log does not hide it**: the pathway
"firing now lowers my own future membrane" is dropped. Detaching a reset is not
novel — it is `reset="detached"` in this project's own Phase-2 kernel, with its
own compiled kernel and its own mutation coverage (M04, M05). What is new is that
on a *two-compartment* neuron it stops being a stylistic choice about a gradient
and becomes the one change that restores the contraction the second compartment
destroyed.

### 16.2 The counterfactual, on the membranes that actually killed the run

Both rules were evaluated **in one forward pass on one checkpoint** — `detach()`
changes no forward value — so nothing below can be attributed to two arms having
reached different states. Batch 5138 is the batch the dying run actually saw.

| leg | arch | `d` | layer | hard `max\|g\|` | hard run | hard window | **detach** |
|---|---|---:|---:|---:|---:|---:|---|
| A1–A3 | `snn` | 512/1020/1481 | 0,1 | 0.51236 | 0 | 10^0.00 | **0.5 / 0** |
| A4 | `twocomp` | 512 | 0 | 2.87016 | 3 | 10^0.66 | **0.5 / 0** |
| A5 | `twocomp` | 1481 | 0 | **9.84309** | **85** | **10^16.00** | **0.5 / 0** |
| A6 | **this arm** | 1481 | 0 | 5.92227 | **84** | **10^27.58** | **0.5 / 0** |

**T1** holds (the `snn` legs return the derived bound), **T1b** holds (the `hard`
column reproduces `EXP_016`'s committed artifact, **10/10 cells, 0 differing** —
the bound validates the arithmetic, this validates the plumbing), **T2** holds
bitwise, **G4** holds (200 quantities re-measured, 0 differing), and **H1 is
HELD**: `max|g| = 0.5`, `frac(|g|>1) = 0`, longest expanding run **0**, on every
leg, every layer, both batches.

**A6 is the row nothing predicted.** It is *this arm's own* trained checkpoint.
Under the rule it trains with, its recurrence never expands. Under the hard rule
those same weights would expand for **84 consecutive timesteps worth 10^27.58** —
an order of magnitude worse than the adopted arm's own 10^16.00 at the step it
died, with `max|w·vs|` **445.9 against 356.6**.

> **The fix does not steer training away from the dangerous region of weight
> space. It trains further inside it, because its gradient is indifferent to
> `w·vs`.** No prediction was placed on this and it is carried as an observation,
> not a bar. It bears on §4's unranked frozen-`beta_s` ladder, which is the
> instrument that would say whether `w·vs` was ever being *held* small.

### 16.3 H2 — it trains at width, under the recipe that killed the adopted arm

| | `arch_twocomp_d1481_s0` | **`detach_d1481_s0`** |
|---|---|---|
| outcome | **DIVERGED, step 5138** | **20,000 steps, 0 non-finite** |
| carried test bpc | NaN | **1.86290** |
| parameters | 5,003,023 | **5,003,023** — identical |
| peak VRAM | 2.6138 GiB | **2.6138 GiB** — identical |

**No hyperparameter moved**, and that is verified rather than asserted: 24 frozen
fields read back out of each run's own `config.json` (gate G5). H2's whole meaning
depends on it.

**H2b, a marker with no verdict**, beside §14's parameter-identical table — all
five at `d = 1481`, seed 0, same recipe, same tree:

| arm @ 5.0M | carried bpc |
|---|---:|
| `gru` anchor (violates I5) | 1.54699 |
| **`twocomp_detach`** | **1.86290** |
| `snn` control | 2.00073 |
| `twocomp` | NaN — diverged 5138 |
| `twocomp_threshold` | NaN — diverged ~2000 |

The matched-size gap to the anchor goes 0.45374 → **0.31591**. §14.2 measured
width closing **6.6 %** of that gap; this closes **30.4 %**. **n = 1 on both
sides, no bar, and the report attaches no verdict to it.**

### 16.4 The result nobody pre-registered: two of three fresh anchors diverged

**At `d = 512` — the width the adopted arm was adopted at, where §2 rests on seven
clean seeds — two of three freshly trained `twocomp` anchors went non-finite**, at
steps 11500 and 12500. Seeds are **not replaced** (`CONTRIBUTING.md` §4). This is
§10.4's `compose_s0` a second time, and this time it took the *control*.

| run | outcome | carried |
|---|---|---:|
| `anchor_twocomp_d512_s0` | **DIVERGED, step 11500** | NaN |
| `anchor_twocomp_d512_s1` | trains | 2.11766 |
| `anchor_twocomp_d512_s2` | **DIVERGED, step 12500** | NaN |
| `detach_d512_s0/1/2` | **all three train** | 2.11769 / 2.11296 / 2.11532 |

**H3 therefore resolves NOT RUN**, by the guard its own §4 wrote: a diverged run
resolves NOT RUN, not UNRESOLVED — §14's referred **DIVERGED / NO RESULT**
correction (decision #9's substance) applied prospectively for the first time.

**Markers, with no bar:** the detached arm is **2.11533 ± 0.00237** (n = 3) —
**−0.00234** from the surviving anchor and **−0.00337** from §2's committed n = 7
mean of 2.11869, both inside the design's own stated 0.005 resolution floor. σ
re-measured for the new arm: **0.00237, 2 df**, against `EXP_005`'s 0.00313. The
surviving anchor lands **0.00103** from the committed figure, the tightest
fresh-vs-committed agreement this project has produced.

**The stability difference is NOT established, and the log says so.** The
divergence rate carries its denominator and an interval (`CONTRIBUTING.md` §3):
adopted arm **0.667 (2/3), 95 % CI [0.208, 0.939]**; this arm **0.000 (0/4) across
both widths, 95 % CI [0.000, 0.490]**; Fisher's exact on the 3-vs-3 table
**p = 0.40**. Four runs cannot separate them.

**A post-hoc probe on both dead anchors** found `EXP_016`'s mechanism present at
735K on the runs that died — layer 0 expanding for up to **21 consecutive steps**
worth **10^4.21**, smaller than 5.0M's 10^16.00 but present, and **absent for the
plain LIF at every width**.

**What this narrows.** §15.7 said, correctly then, *"it does not say the arm is
bad… where its longest expanding run is 3 steps."* The seven clean seeds were
trained **before decision #7 replaced the gradient clip**. Retrained on today's
tree, the same configuration at the same width lost two of three. **This is not a
claim that the clip caused it** — that is a specific hypothesis, ~9 GPU-minutes
each way, and `EXP_017` §10 item 6 refers it rather than testing it under a
pre-registration that did not declare it. What is measured is that the adopted
arm's 735K stability is **marginal enough to be moved by something**, and that
this arm's, over four runs at two widths, was not.

### 16.5 H4 and H5, both reported as they fired

**H4 = REACH REDUCED**, on **one** surviving pair, at `Δh = −2` characters, on a
metric for which this project has never measured a σ — while the three detach
seeds alone span **47–50**, a spread larger than the difference the verdict rests
on. For context and not as part of the verdict: the committed adopted arm's
horizon is **47** and the plain LIF's is **7**; this arm's median is **48**.
**Report a near-miss as a miss** — the verdict is REDUCED and is not negotiable;
the near-ness is referred.

**H5 = BREACH** at 1.127× at `d = 1481`. **The breach is in the denominator, and
the denominator was mine**: H5's pre-registered reference is the wall-clock of a
run that spent **14,862 of its 20,000 steps computing on NaN**. `CONTRIBUTING.md`'s
rule that a ratio needs its reference stated turns out to need a second clause —
the reference must also be a *valid measurement*. The bar stands as it fired; the
investigation its own wording demanded gives a controlled paired re-measurement of
**1.034×**, three `d = 512` runs **faster** than three same-session anchors
(0.941×), and peak VRAM **identical to the byte** at both widths.

### 16.6 A mutation escaped, and it teaches something new

The first campaign run was **58/59**. **D09** — `>` for `>=` in the backward
kernel — **escaped**. **A bound cannot see that mutation**: at `v_pre == thr`
exactly, `>=` gives `g = 0` and `>` gives `g = beta_f`, and *both are inside
`{0, beta_f}`*, so every assertion about the bound holds while the derivative is
wrong. **A bound is a statement about a set, and an off-by-one in a comparison
moves a point within that set.** §11's rule — *a numerical contract is only
guarded where the quantity it constrains is actually observed* — from a direction
it had not arrived from: the unguarded quantity was not an unread intermediate but
*which branch* a value came from, and random inputs cannot supply it because exact
equality has measure zero. Two legs added, **whole campaign re-run at 59/59.**

### 16.7 What this changes, and what it does not

**Changes:** Phase 5's stated blocker was that the adopted arm cannot be scaled
(§14.1). A parameter-identical arm with the same forward, the same inference
model and the same inference cost now trains at 5.0M under the unchanged recipe
and scores 1.86290. **And `EXP_016` §2.4's consequence for decision #11 gains a
second, more direct data point: the failure was fixable without touching the
recipe at all.**

**Does not change:** nothing is adopted (#5, #1), no size is chosen, **Phase 5 is
not authorised** (#4 is still "not yet"), and #11 is not decided. H3 produced no
comparison; H4 resolved REDUCED; the stability difference is unestablished at
p = 0.40; and one seed at one width is one seed at one width.

**Closes no ranked candidate** — like `EXP_012`, `EXP_013`, `EXP_015` and
`EXP_016` before it. **Referred to Elliot** (log §10, ten items): adoption; the
soft-reset alternative, derived and not run; **the two dead anchors and the
9-minute clip test that would explain them**; H5's denominator; H4's power; what
A6 means; and what "the adopted arm" now denotes.

---

## 17. `EXP_018` — the model can find its own prediction error, and using it costs bits

Pre-registered in [`experiments/logs/EXP_018_dopamine.md`](../../experiments/logs/EXP_018_dopamine.md),
committed before the first training step; SHA-256 stamped into the run manifest
and verified unchanged after the last. Sixteen runs, **sixteen completed, zero
diverged**, ~2.9 GPU-hours.

**This is a negative result and it is reported as one.** The arm adopts nothing,
ranks nothing, and closes no ranked candidate — it is not on the 18-item list, for
the same reason `EXP_013` was not.

### 17.1 The arm

A **dopamine signal**: the model's own reward prediction error, broadcast to every
neuron as one scalar per `(batch, timestep)`, with a learned per-channel
sensitivity `k` (`[1, d]` per layer, `+K·d` = **+0.14 %**, `k_init = 0`).

```
S_t   = -log p_{t-1}(x_t)      the surprise                              (nats)
H_t   =  H(p_{t-1})            the model's OWN expected surprise
phi_t =  H_t - S_t             the RPE.     DA_t = tanh(phi_t / tau)
cur'  =  cur * (1 + k_c*DA_t)   [mult]   or   cur + k_c*DA_t   [add]
```

Three properties decided the design and are worth carrying forward.

**`E_{x~p}[-log p(x)] = H(p)` exactly**, so `phi` is zero-mean under the model's
own belief *by construction* — no baseline, no EMA, no critic. That is not only
tidy: an EMA over `t` is a sequential scan, `L` kernels per layer, inside the
region `snn.train` captures into a CUDA graph. The entropy is one parallel
reduction. **The derivation removed the arm's only other free hyperparameter.**

**It is not a reparameterisation, and that is the scientific content.** `EXP_008`'s
constant per-channel gain folds into `layers.k.weight` (§6.3), so its 0.0590 bpc
is an *optimisation* effect on an unchanged function class. `DA_t` varies with `t`
and folds into no weight, so this arm's function class is strictly larger. There
is deliberately no `fold_into_*` method and **decision #6 does not arise for it.**

**It costs two forward passes.** `DA_t` needs the head at `t-1`, and the head is
downstream of every layer, so the depth-sequential evaluation (§0 of the spec)
cannot supply it inline. Measured, not estimated: **1.708x** wall clock against a
same-session anchor, at training **and at inference** — worse than `EXP_007`'s
1.49x, which §6.7 item 2 already records as dominated on cost.

**I5 was NOT RULED when this arm was pre-registered, and it is now.** The
parameters pass §4.6's test (`[1, d]`, channel-diagonal); the *pathway* is a
rank-1 all-to-one-to-all coupling, which was a **fourth** boundary question for
decision #3. **Ruled 2026-08-14 on Elliot's instruction** (§8 row 3, extension
ratified in `01_reconnaissance.md` §4.6): a modulator driven by the **previous
layer** is admitted, one driven by the **head** is **diagnostic-only**. This arm
is head-driven, so **it is permanently a diagnostic and was never a candidate for
adoption** — which is what `EXP_018` labelled it throughout, before the ruling
existed.

### 17.2 D1 — UNRESOLVED by 3.6 millionths of a bpc, in the direction of harm

| carried, test | n | mean | sd |
|---|---:|---:|---:|
| `da_anchor` | 5 | **2.25121** | 0.00475 |
| `da_mult` | 5 | **2.25743** | 0.00359 |

`Δ = −0.0062148`, Welch 95 % CI **[−0.0124333, +0.0000036]**, t = −2.33480 against
t_crit = 2.33617. **The interval includes zero by 3.6e-06 bpc and the statistic
misses by 0.0014.** `CONTRIBUTING.md` §3: report a near-miss as a miss.
**D1 is UNRESOLVED.**

σ **re-measured** on these seeds: 0.00475 carried, so the pre-registered power
holds with the measured number substituted — MDE 0.0096 rather than 0.0093.

### 17.3 The correction, measured because it hurts

**The design used an unpaired Welch test on paired data.** A seed in this project
fixes both the initial weights and the data order, so the two arms are genuinely
paired. `CONTRIBUTING.md` §3 — *"when a correction would hurt, measure it"*:

> paired, `anchor − arm`, carried: **5 / 5 seeds worse**, mean **−0.006215**
> (sd 0.001859), t = **−7.477** on df 4, 95 % CI **[−0.008523, −0.003907]**,
> **p = 1.7e-03**.

**The arm costs ~0.0062 bpc on every seed and the pre-registered instrument was
too blunt to say so.** D1's verdict stands as it fired; the corrected *rule* is
referred (`EXP_018` §10 item 6) and is a candidate standing rule for the phase.

### 17.4 The 26x — the finding that is not about bpc

`rms(k)` at step 20,000, over layers and seeds:

| | mean rms(k) | mean max\|k\| |
|---|---:|---:|
| `da_mult` — the aligned RPE | **0.1950** | 2.32 |
| `da_rolled` — the same signal, misaligned across the batch | **0.0075** | 0.045 |

**A factor of 26, stable to three significant figures across seeds.** The
optimiser can tell the aligned prediction error from a misaligned one and turns
the gain up 26x on the real one. **The reward prediction error is information this
model can find and wants to use.**

And it does not help. Paired, `arm − anchor`, n = 5:

| | mean | p |
|---|---:|---:|
| train bpc | **+0.00043** | **0.63** |
| test bpc, fresh | **+0.00611** | **0.0017** |

That was `EXP_018` §9.5's reading, and **the committed instrument overturned
it.** §9.5 used the mean of the last eight logged single-batch *training* losses.
`scripts/exp/013_generalisation_gap.py` — §12's own instrument, a 19 456-window
train slice scored in `eval()` — was then run over all sixteen checkpoints, and
says something different:

| paired, arm − anchor, carried | Δ | t (df 4) | p | arm worse |
|---|---:|---:|---:|---|
| **train slice** | **+0.00522** | +7.23 | **0.0019** | **5/5** |
| test | +0.00621 | +7.48 | 0.0017 | 5/5 |
| **gap** | **+0.00100** | +1.71 | **0.16** | 4/5 |

**The arm is worse on text it was trained on by almost exactly as much as on text
it was not, and the generalisation gap does not widen resolvably.** "The gap
roughly doubles" is **withdrawn**; **this is a capability loss, not a
generalisation failure** — the arm is simply a worse model everywhere. The
self-referential-channel hypothesis that reading motivated is **not supported**,
and `EXP_018` §9.5 carries the correction beneath the standing original.

**Why the two estimators disagreed is now known**, and chasing it produced the
more durable finding. The sampler is a pure function of `(seed, step)`, so the
exact eight batches were regenerated and re-scored at `ckpt_final`: the arm is
worse on **those very batches** by **+0.00377** (t 4.22, p 0.0135, 5/5), against
+0.00043 as logged. **A logged training loss is measured at pre-update weights**,
and over the logged tail the anchor was still improving by 0.00619 against the
arm's 0.00285 — an asymmetry of **+0.00334 ± 0.00096** (p 0.026), almost exactly
the effect size, which cancelled it.

`eval()` and `train()` agree **bit-for-bit** (`0.00e+00`) on identical inputs at
identical weights, so there is no mode-dependence and no undeclared state in the
arm. The arm is therefore worse on the **very batches it trained on, at the
weights it finished with** — a third independent slice agreeing with the other
two. **Standing lesson: a running training loss is not a stand-in for a
train-split bpc, and a train-side number should be scored with
`snn.evaluate.evaluate` rather than read out of `log.jsonl`.**

### 17.5 The decomposition, which is two effects of opposite sign

`EXP_001`'s probe, all 16 checkpoints, `--baseline-arm da_anchor` (today's tree,
never `snn_beta0.5` — decision #10). **F1 held on all sixteen**, residuals 2.0e-11
to 1.2e-08.

| Δ vs anchor, + = worse | c=0 | c=2 | **c=3** | c=4 | c=8 | c=32 |
|---|---:|---:|---:|---:|---:|---:|
| `da_mult` | **−0.0246** | +0.0217 | **+0.0427** | +0.0322 | +0.0079 | +0.0050 |
| `da_rolled` | −0.0004 | −0.0168 | +0.0029 | +0.0011 | +0.0029 | +0.0039 |
| 2σ bar | 0.0296 | 0.0260 | 0.0144 | 0.0145 | 0.0040 | 0.0010 |

**The −0.0062 mean is two effects of opposite sign**, which is §2's lesson applied
to a fourth arm.

* **At c = 0 the arm is 0.0246 better — and the mechanism is structurally
  inactive there.** `phi_0 = 0`, so at one character of context no modulation is
  applied at all; that difference is what training left in the *weights*. It is
  also inside the 2σ bar and is not established.
* **The damage peaks at +0.0427 at c = 3**, three times the local bar and **seven
  times the whole-split mean**, where the predictive distribution is still nearly
  uninformative so `phi` is mostly noise — and the arm has learned a large gain
  on it.
* **It settles to ~+0.006 for c ≥ 8**, resolved many times over.

`da_mult` is worse at **125 of 128** contexts; `nowhere_worse` is **False**, so the
`EXP_001` dominance criterion is failed, at short contexts, which is the failure
mode it exists to catch. The misaligned control shows **neither** end of the
shape, so both track alignment rather than the perturbation.

**D5, marker, no verdict**: 2σ horizon `da_anchor` [7,7,7,7,7], `da_mult`
[8,8,8,7,7]. A nominal 7 → 8 on three of five seeds while the absolute curve is
worse almost everywhere is exactly the trade `EXP_001` was built to expose.

### 17.6 What this changes about the plan

1. **A global broadcast is not free, and this is the first measurement of what it
   costs here.** §7 has been asking for arms that are not purely architectural;
   this was one, and it is the first Phase-4 arm to be **worse than its anchor on
   every seed**.
2. **The signal is real even though the arm is not.** The 26x is the reusable
   result: an RPE is findable and usable by this model — and the harm it does is
   a **capability** loss on seen and unseen text alike, not overfitting. §10 item 9 of the
   experiment log points at the obvious follow-up — spend it **training-only**
   (weighting the loss, say), which is `EXP_013`'s shape: no parameter, no
   inference cost, nothing to fold, and none of the 1.71x.
3. **Short contexts are where a modulation can do damage**, and the per-context
   bar is what showed it. A whole-split mean would have called this a 0.006 null.
4. **Nothing here touches decisions #3, #6, #10 or #11**, and no row of §8 is
   edited. The I5 question this arm raises is a *fourth* boundary case for #3.

---

## 18. `EXP_020` — the two ends, measured for the first time, and every arm lost

Every arm this phase built changed the middle of the model. `EXP_020` is the
first to change its **two ends**: layer 0's input had been `nn.Embedding` since
Phase 2 and the head one `nn.Linear` on the last layer's spikes at `t`, both
inherited from `02a_phase2_spec.md` §4 and never measured. Those two blocks hold
**28.6 %** of the parameters at the committed shape, and a third — `layers.0.weight`
— turns out to be provably redundant with the first.

Sixteen runs, ~1.9 GPU-hours, **none diverged**, all nine gates passed. The
pre-registration's SHA-256 was stamped into `exp_020_run_manifest.json` before
the first run and re-checked after the last: unchanged.

### 18.1 The algebra, and it is exact

`E[x]` is a *lookup*: one of `V` rows, never a mixture. A linear map applied to a
lookup is a lookup of the mapped table, so with `T = E @ W0^T`

    cur0[b, l] = W0 · E[x[b,l]] + b0 = T[x[b,l]] + b0                  EXACTLY

Measured on `snn_beta0.5_s0`: max relative difference **5.19e-08**, 62.8 % of
elements bitwise equal — fp32 rounding of a `d`-term dot product and nothing
else. **`layers.0.weight` spans no function the embedding table does not, and it
is 262,144 of 735,437 parameters — 35.7 %**, the only parameter block in the
stack provably redundant with another one.

### 18.2 And removing it costs 0.0946 bpc — 1.63× what the parameters are worth

| bar | verdict | number |
|---|---|---|
| **T1** whole-run nesting | **PASS at exactly 0.0** | `if_nest_s0` reproduced `if_anchor_s0` to 0.0 in both protocols over a full 20,000-step run |
| **H1** the fold | **FAILED (magnitude)** | +0.09459 bpc, paired t = 24.42, 0/3 seeds better, against a predicted ceiling of +0.0582 |
| **H2** the width it buys | **UNRESOLVED**, toward harm | `foldwide` at `d = 675` is +0.01326 *worse*; t = 3.45, 0/3 better |
| **H3** a binary input | **FAILED (arm worse)** | +0.04569 bpc at **identical** parameters; t = 14.24, 0/3 better |
| **H4** the multi-layer readout | **UNRESOLVED** | +0.00530 worse, inside 2σ; t = 1.86, 1/3 better |

**Every arm lost.** Two of the four lost by more than the bar, and spending the
freed 35.7 % on width does not recover it.

This is **§6.5's `EXP_008` result one layer earlier and larger**. There, a
provable reparameterisation was worth −0.0590 bpc in the *over*-parameterised
direction; here, removing a provably redundant factorisation costs +0.0946. Both
say the same thing: **on this architecture, redundant parameterisation of the
input path is worth real bits to the optimiser, and the function class does not
predict how many.** That is now two independent measurements of one phenomenon,
and it bears directly on open decision **#6**.

### 18.3 σ transferred, and for the first time it was checked and held

Transferred 0.00461 (§3, five baseline seeds on a **different** tree) against
**0.004683** measured on this tree's three anchor seeds: **×1.02.** §3 says σ is
not a project constant and every previous transfer went unchecked. This one was
checked and it held — which strengthens §3's rule rather than weakening it,
because the check is the point.

### 18.4 The decomposition inverts two of the four readings

`EXP_004` §10.3's split, cut at the baseline's horizon of 7, positive = better:

| arm | total | c = 0 | within reach | beyond horizon |
|---|---:|---:|---:|---:|
| `if_fold` | −0.09453 | −0.02710 | **−0.08143** | +0.01400 |
| `if_foldwide` | −0.01366 | **+0.03973** | **−0.07047** | +0.01708 |
| `if_binin` | −0.04478 | **−0.03710** | −0.00291 | −0.00477 |
| `if_mlayer` | −0.00535 | −0.03695 | +0.03128 | +0.00032 |
| `if_nest` *(bitwise the anchor)* | −0.00499 | −0.01395 | +0.00944 | −0.00047 |

* **`foldwide`'s two effects have opposite signs and nearly cancel.** The freed
  width buys **+0.0397 at zero context** — 2.8× §18.5's noise floor, and exactly
  the per-step capacity §13 says width buys — while the fold costs **−0.0705
  within the baseline's own 7-character reach**. The headline −0.0137 is a real
  gain plus a larger real loss, and the sum alone hides both.
* **The fold's cost is a REACH cost.** 86 % of it is within-reach and only 29 %
  at zero context. Removing a redundant *input* parameterisation damages the
  model's use of context, which no argument from parameter counting predicts.
* **Binarising the input is almost purely a zero-context cost** — 83 % at
  `c = 0`, −0.0029 within reach, −0.0048 beyond. It is a per-character
  *expressiveness* price and costs essentially nothing in memory.

### 18.5 The instrument failed its own control, and that is the finding

`if_nest_s0` is **bitwise** `if_anchor_s0` (T1 puts their final bpc exactly 0.0
apart), so every number in its row is one seed against a three-seed mean and
nothing else. It reads **−0.00499 total, −0.01395 at `c = 0`, and 120 of 128
contexts "significantly worse"**. The per-context 2σ bars fall to 2.4e-05 beyond
`c ≈ 70` while a bit-identical run differs from the three-seed mean by ~0.005
there — **up to 80× below single-seed variation.**

**`n_contexts_significantly_worse` is unusable at n = 3**, and `mlayer`'s
−0.00535 is indistinguishable from a bitwise copy of the anchor. That is fresh,
measured evidence for open decision **#2**, which has stood since `EXP_005`
called §6.2's per-context machinery "built on the wrong scale". **No
redefinition is proposed here**, and every later experiment in this session
reports the statistic as a marker with this floor quoted beside it.

### 18.6 Binarity, counted

At `d = 512, K = 2` the inter-unit wires per position are token→L0, L0→L1,
L1→head. The committed model is binary on **1024 of 1536 — 66.7 %**; `binin` is
**1536/1536** at identical parameter count. And it is a *true* spike input, not a
two-valued analogue one:

    W0·(g(B − q)) + b0  =  (g·W0)·B  +  (b0 − g·q·W0·1)

so the centring is a **reparameterisation of `layers.0.bias`** (verified
5.8e-07) — the distinction `snn.noise`'s docstring draws about its own centred
Bernoulli, met here and cleared.

### 18.7 Three defects found, none touching a number

* `local_dopamine_param_count` omitted the learned `nm_log_tau` — 736,973
  against a realised 736,974. Caught by an `xfail(strict=True)` test **before**
  `EXP_021`'s G2 gate would have aborted on it.
* `iface_read_lags > 1` decoded all-zero lag blocks at `L = 1` with no raise and
  no capture failure, eager and graph paths agreeing with each other while both
  were wrong. Now raises.
* The resolver read the per-context bars from a **wrapper** object, so it
  silently fell back to the flat bar and reported 2/128 where the probe's own
  figure was 120/128. **Caught only because two implementations of one statistic
  disagreed** — which is the argument `EXP_011`'s resolver made for importing
  §10's helpers rather than reimplementing them, met again.

### 18.8 What it does not do

Adopts nothing, ranks nothing, closes no ranked candidate, and is not on the
18-item list. It re-baselines nothing (decision #10): every figure above is
internal to the experiment, against an anchor trained in the same session on the
same tree.

---

## 19. `EXP_021` — the modulator that won, and the control that took its mechanism away

§17 established that the head-driven reward prediction error is **findable** —
the optimiser assigns it 26× the sensitivity of a batch-rolled control — and
that **paying attention to it costs 0.0062 bpc on 5 of 5 seeds**. Decision #3,
ruled at rev 12, makes that slot permanently diagnostic and **admits** a
modulator driven by the previous layer. `EXP_021` is that arm, plus a second,
independent form: dopamine on the **learning rule** rather than on the forward
pass.

Fifteen runs, ~2.0 GPU-hours, **none diverged**, all gates passed, hash
unchanged.

### 19.1 H1 held — the first Phase-4 arm since `EXP_008` to beat its anchor

`nm_local` gives each channel of layer `k−1` a diagonal one-step predictor of its
own next spike, and multiplies layer `k`'s current by `1 + kappa_c·tanh(phi/tau)`:

> **−0.03371 bpc, paired t = −8.95, 3/3 seeds, 3.7× the bar**, at **1.49×**
> wall-clock and **no second forward pass**.

Per-seed −0.04042, −0.02739, −0.03331 — same sign, none marginal. It is in the
shape decision #3 admits, so unlike §17's arm it is a **candidate** and not a
labelled diagnostic.

### 19.2 And H2 and H3 take the mechanism away

| bar | verdict | number |
|---|---|---|
| **H2** mechanism or perturbation | **UNRESOLVED** | −0.00577 against `nm_rolled`; t = −3.22, 3/3 seeds — below the 0.00922 bar |
| **H3** is the signal findable | **BELOW threshold** | `rms(nm_gain)` **1.077 aligned / 0.958 rolled — ratio 1.12**, against a 3.0 threshold and §17's **26** |

The batch-rolled control gains −0.0279 on its own, and the optimiser grows
`kappa` ~10× from its init **for either signal, and cannot tell them apart**.
§17's factor of 26 on the identical quantity does not reproduce, and the
pre-registration named the FAINT reachability ratio in advance as the risk that
would surface exactly here.

§5 of the pre-registration fixed the sentence for this cell before the run, and
it is not softened now:

> **What is established:** a bounded, learned, **rank-1 time-varying
> multiplicative gain on the input current** is worth ~0.028–0.034 bpc on this
> architecture. **What is not:** that the reward prediction error is why.

### 19.3 The gain is entirely within reach — which is where §14 said the gap is

`EXP_004` §10.3's split, positive = better. Read against §18.5's bitwise-copy
noise floor (−0.00499 / −0.01395 / +0.00944).

| arm | total | c = 0 | within reach | beyond horizon |
|---|---:|---:|---:|---:|
| `nm_local` | **+0.03353** | −0.02183 | **+0.05588** | **−0.00052** |
| `nm_rolled` | +0.02737 | **−0.08103** | **+0.10955** | −0.00115 |
| `tf_neg` | −0.04765 | −0.02909 | −0.01824 | −0.00032 |
| `tf_pos` | −0.08712 | **−0.21159** | +0.12552 | −0.00105 |
| `tf_roll` | −0.00084 | +0.00983 | −0.00925 | −0.00143 |

**`nm_local`'s entire gain is within reach** — +0.0559 inside the baseline's own
7-character horizon, −0.0218 *paid* at zero context, and **−0.0005 beyond the
horizon, which is nothing.** The arm does not extend reach; it improves the use
of context the model already had.

That matters more than the headline. §14.3 located the **entire** matched-size
gap to the GRU anchor in exactly that failure — "0.051 apart at zero context,
0.408 apart by `c = 64`" — while every row in §7.1 is ranked against bits. And
§18's decomposition puts the two-compartment neuron's gain in the same place, so
whether these two are complementary or redundant is **not measured** and is
referred.

**Both arms trade zero-context capacity for within-reach use, and the aligned
signal trades less of it** (−0.022 against −0.081). The totals are close and the
*shapes* are not: that is the one place in the experiment where alignment
visibly does something, and no bar was pre-registered on it.

### 19.4 It is not a generalisation effect, and that is the opposite of §17

Measured with §12's committed instrument, not the logged loss — four full-split
evaluations per run, 19,456 windows each:

| arm | gap, fresh | gap, carried |
|---|---:|---:|
| `if_anchor` | +0.00472 ± 0.00104 | −0.00388 ± 0.00156 |
| `nm_local` | +0.00835 ± 0.00220 | −0.00105 ± 0.00226 |
| `nm_rolled` | +0.01100 ± 0.00030 | +0.00151 ± 0.00124 |

`nm_local`'s gap is within ~0.004 of the anchor's on both protocols: **the arm is
better on held-out text because it is better.** §17's head-driven arm widened
its gap by 0.0057 while losing, so this is the cleanest evidence yet that the
forward slot and the head slot are genuinely different places to put one signal.

### 19.5 The learning-rule slot is actively harmful, and the harm is
### alignment-dependent

`w_t = 1 + kappa·tanh(phi_t/tau)`, detached, `sum w·S / sum w` — the three-factor
rule `dw ~ pre × post × DA` written in the only form this trainer can express.
The forward pass is untouched, so at eval the model is the Phase-2 baseline
**bitwise, by code path**, and the arm costs 1.02–1.05×.

* **H4 held** (it predicted no improvement): `tf_neg` is **+0.04826 worse**,
  t = 29.34, 0/3 seeds better.
* **H5 held**: the harm IS alignment-dependent — `tf_roll` is neutral (−0.0008)
  while the aligned form is +0.048 worse.
* **H6 held**: learn-from-errors beats consolidate-success by **0.03770**
  (t = −30.26, 3/3), though both lose. `tf_pos` costs **−0.2116 at zero
  context**, the largest single component in the experiment: up-weighting
  positions the model already predicts well down-weights exactly the surprising
  ones, and at `c = 0` every position is surprising.

### 19.6 A pre-registered statistic was invalid, and §17's own referral predicted it

H7 read the train→test gap from the last logged training loss. **For the `tf_*`
arms that quantity is the WEIGHTED objective**, so it returned −0.354 and +0.673
— numbers that mean nothing. `EXP_018` §10 item 11 refers exactly this: *"a
running training loss is not a stand-in for a train-split bpc."* **This is the
second experiment that referral has caught and the first caught after it was
written down**, and §19.4's table is the re-measurement with the committed
instrument. **It should be ratified into `CONTRIBUTING.md`.**

### 19.7 Two derivations were forced by measurement, before any run

* **`phi` summed over channels is O(d)** — +101 nats at `d = 512` — which
  saturates `tanh` to exactly 1.0. That kills the predictor's gradient *and*
  makes `DA` constant, which by `EXP_008` Identity 1 is a learned per-channel
  threshold: **the saturated arm would have trained, scored plausibly, and been
  measuring an already-adopted arm (#5) under a new name.** Fixed by using the
  per-channel mean, which also makes `tau` width-independent.
* **The literal `H − S` measured 2.01×** — worse than the 1.71× arm it exists to
  beat. The two terms share a `log(1−p)` that cancels exactly:
  `phi = mean_c (s − p)·logit`, the exponential-family identity
  `H − S = (s − E[s])·eta`. Four passes, no logarithm, **1.49×**. The pre-flight
  paid for itself before a single run.
* `nm_tau = 0.057321`, calibrated at the predictor's **fixed point**
  (`b_c = logit(r_c)`). Calibrating at the arm's own init prior returned 100 %
  negative `phi`, because `nm_b_init` describes the *init* firing rate while
  checkpoints converge to 0.345–0.392. **Relative spread 36.8 % against §17's
  3.1 %**, so this scale is NOT established as a property of the task and `tau`
  is therefore learned rather than frozen.

### 19.8 A tension with §12, recorded rather than reconciled

§12 measured injected noise as a null at every amplitude. Here a signal the model
cannot distinguish from a misaligned one is worth −0.028. Four structural
differences, none isolated: **additive vs multiplicative**; i.i.d. over
(b, t, c) vs **rank-1**; a swept constant vs **learned per channel**; an RNG vs
**an activation the forward pass already produced**. Separating them is four
arms, and none is designed here.

### 19.9 What it does not do

Adopts nothing, ranks nothing, closes no ranked candidate. **Adopting `nm_local`
would be adopting "a learned rank-1 time-varying gain", not "a dopamine signal"**,
and §19.2 says so. Six post-run referrals, of which two carry prices: resolving
H2 at n = 5–7 seeds (~0.6 GPU-h) and composing onto `twocomp_detach` (`EXP_018`
§10 item 4, now with a reason — both take their gain within reach, so they may be
redundant rather than complementary).

---

## 20. `EXP_022` — the binarity cost was a density cost, and closing the input boundary is free

§18 measured a fully binary input code at **+0.04569 bpc** against the analogue
anchor and could not say whether that was the price of *binarity* or of the one
density nobody chose. `EXP_022` ran the density ladder. Fifteen runs,
~1.9 GPU-hours, **none diverged**, all gates passed, hash unchanged and
re-verified at resolution.

**It is the first experiment this session whose §8 entry condition — the
resolver committed before any bpc is read — was actually met** (`858c48a`).
§18's and §19's were not, and §20.7 below records that.

### 20.1 The ladder, and it is steep below q = 0.34

Density `q` is set by one field, `iface_thr_in`, with the gain re-derived from
it so the second moment layer 0 was calibrated for never moves. **All four rungs
are 735,437 parameters**, asserted exactly by G2 — changing a code's density
changes its alphabet and not its shape.

| rung | `q` init | mean bpc | vs `if_binin` | vs `if_anchor` | binarity cost recovered |
|---|---:|---:|---:|---:|---:|
| `if_binin` | 0.50 | 2.29813 | — | +0.04569 | 0 % |
| `io_sp34` | 0.34 | 2.30530 | +0.00717 | +0.05286 | −16 % |
| `io_sp10` | 0.10 | 2.27247 | **−0.02566** | +0.02002 | **56 %** |
| `io_sp05` | 0.05 | 2.26199 | **−0.03614** | **+0.00955** | **79 %** |

> **At `q = 0.05` a true spike input costs 0.00955 bpc against the analogue
> anchor, for no parameters and at 0.97× wall-clock.** H2 held at 2.8× its bar
> and H3 met it at 3.9×, both on 3 of 3 seeds.

**H1 failed, and it is the rung §2.2 argued hardest for.** `q = 0.34` is the
stack's own converged firing rate, where the input boundary is statistically the
same boundary as every other one — and it is the only rung that does not beat
`q = 0.50`. The ladder is monotone between sp10 and sp05 and not over its whole
range, so §2.2's leverage argument does not describe its top end.

### 20.2 The realised densities are not the pre-registered ones, and the winning code is lossy

`EXP_022` §6 item 3 pre-registered this diagnostic before the run. `shadow` is a
parameter, so density drifts:

| rung | `q` init | `q` final | drift | active fibres | distinct codes | silent |
|---|---:|---:|---:|---:|---|---:|
| `if_binin` | 0.50 | 0.499 | 1.00× | 255 | 205/205 | 0 |
| `io_sp34` | 0.34 | 0.136 | 0.40× | 70 | 200–201 | 5–6 |
| `io_sp10` | 0.10 | 0.025 | 0.25× | 13 | 184–187 | 12–14 |
| `io_sp05` | 0.05 | **0.013** | 0.25× | **6.5** | **163–166** | **26–28** |

The rungs did **not** converge to one density, so the pre-registered failure
condition did not fire and the ladder is still a density ladder — but every
figure in §20.1 belongs against the realised column. **The `q = 0.50` rung is the
only one that does not drift**, and the reason is structural: at `thr_in = 0` the
shadow is symmetric about its threshold, so weight decay leaves the density at
0.5 exactly; at `thr_in > 0` the same pull moves every bit toward the inactive
side. That is `docs/chat/WEIGHT_DECAY_NOTE.md`'s fixed-point argument on a
different parameter, and it was not anticipated.

**And the winning code cannot tell 42 of its 205 characters apart.** At
`io_sp05`, 39–42 characters share a code with another and 26–28 emit the
all-zero code, which is indistinguishable from no input at all. §2.2's argument
that the ladder is not information-limited is about *capacity*; the realised
random code is lossy, and it beats the code that distinguishes every character
by 0.036 bpc on 3/3 seeds. **That is better evidence for §2.2's mechanism — a
claim about the code's geometry and its leverage per bit — than the capacity
arithmetic §2.2 actually offered**, and it is not an argument that
pre-registration made.

### 20.3 The fix appears exactly where the cost was

`EXP_004` §10.3's decomposition, positive = better, against the noise floor §18.5
measured with a bitwise copy of the anchor (−0.00499 / −0.01395 / +0.00944):

| arm | reference | total | `c = 0` | within reach | beyond horizon |
|---|---|---:|---:|---:|---:|
| `if_binin` | `if_anchor` | −0.04478 | **−0.03710** | −0.00291 | −0.00477 |
| `io_sp34` | `if_binin` | −0.00688 | +0.02465 | **−0.03137** | −0.00016 |
| `io_sp10` | `if_binin` | +0.02546 | **+0.04127** | −0.01633 | +0.00052 |
| `io_sp05` | `if_binin` | +0.03560 | **+0.03256** | −0.00009 | +0.00312 |

82.8 % of binin's cost sits at `c = 0` — reproducing §18's 83 % — and **all three
sparse rungs recover at exactly that component**. What separates them is what
they give back within reach, and *that* is monotone where the totals are not:
−0.0314, −0.0163, −0.0001. Sparsity buys the same thing at zero context on every
rung and stops charging for it as `q` falls. **No bar was pre-registered on that
gradient and none is invented.**

### 20.4 Leg B: the readout is worth its parameters, and width is worth the same

| arm | arch | `d` | params | mean bpc | vs anchor |
|---|---|---:|---:|---:|---:|
| `if_anchor` | `snn` | 512 | 735,437 | 2.25245 | — |
| `io_mall` | `interface`, all-layer readout | 512 | 840,397 | 2.23088 | **−0.02157** |
| `io_wide` | `snn` | 553 | 839,659 | 2.22812 | −0.02392 |

**H5 held** at −0.02157, t = −8.49, 3/3. **H6 — the decisive bar — returned
UNRESOLVED** at +0.00276, and the point estimate favours width. §5's table has
cells for HELD and FAILED and none for this, so the verdict is reported as it
fired: the readout is **not better than width by 2σ**, and this experiment does
not establish that width is better either.

**H7 held at +0.00154.** §2.4 predicted before the run that a multi-layer readout
adds no temporal reach — layer 0's spikes at `t` carry nothing about `t' < t`
that layer 1's do not already carry — and it gains nothing beyond the horizon.

### 20.5 What H6 could not see, and it is the largest within-reach movement of the phase

The two arms are 0.0028 apart in total and are not doing the same thing:

* **`io_mall` pays −0.02348 at zero context to buy +0.04310 within reach.**
* **`io_wide` buys +0.01186 and +0.01229** — evenly.

§14.3 located the entire matched-size gap to the GRU anchor in the failure to
*use* context, and §7 item 2 records the within-reach component as the one barely
attacked: the arm designed for it moved it by −0.0065 and the best any arm had
managed was +0.0288. **`io_mall` moves it by +0.0431**, and it is invisible in
the headline because it is paid for at `c = 0`.

This is a **marker**. No bar was pre-registered on the decomposition's components,
none is invented here, and **nothing in this section ranks an arm** — §7.1 is
offered and not applied, and ranking it would be this report choosing its own
next experiment.

`io_mall`'s seed sd is **0.000301**, fifteen times below the anchor's 0.004683 on
the same tree. §3's rule that σ is not a project constant, with an unusually
large instance; any later use of this arm must re-measure rather than transfer.

### 20.6 The gap, and an arm that generalises best while scoring worst

| arm | gap, fresh | gap, carried | test bpc |
|---|---:|---:|---:|
| `if_anchor` | +0.00472 ± 0.00104 | −0.00388 ± 0.00156 | 2.25245 |
| `if_binin` | **−0.00156 ± 0.00344** | **−0.01043 ± 0.00324** | **2.29813** |
| `io_sp05` | +0.00357 ± 0.00171 | −0.00579 ± 0.00266 | 2.26199 |
| `io_mall` | +0.00800 ± 0.00221 | −0.00135 ± 0.00272 | 2.23088 |
| `io_wide` | +0.00752 ± 0.00192 | −0.00203 ± 0.00218 | 2.22812 |

`if_anchor`'s row reproduces §19.5's to five decimals — the same three runs, the
same instrument. **`io_sp05`'s gain is not a generalisation effect**: its gap is
inside the seed spread of the anchor's on both protocols. And **the arm with the
smallest gap has the worst test bpc**, which is §12's no-overfitting finding
arriving from a new direction: on this model at this size, moving the gap does
not move the number.

### 20.7 Two protocol facts recorded here rather than left to a commit message

**§18's and §19's resolver-commit entry condition was not met.** Both
pre-registrations require the resolver committed before any bpc is read; both
resolvers reached git for the first time on 2026-08-20, after their numbers had
been read and written into rev 13. The pre-registration SHA-256 stamps hold
byte-for-byte, so the *hypothesis-against-runs* ordering is evidenced; the
*resolver-against-numbers* ordering is not, and no commit made later can supply
it. Both logs additionally assert a pre-run commit while having been untracked,
and carry `CONTRIBUTING.md` §2's stamped hash without the reconstruction recipe
that is the second half of the substitute for a commit. **Recorded as a
deviation, not repaired.** `EXP_022` and `EXP_023` met the condition.

**A defect in `EXP_022`'s own resolver was caught before it was committed**, and
the fix is therefore not a repaired bar. The post-hoc replacement for H4,
`gain_is_at_zero_context`, had its **sign inverted** — it consumed
`010.decompose`'s *positive = better* convention through this file's local
*negative = better* one, so it was True only when a rung was **worse**. Fed §2.1's
own motivating row it announced "the gain is at zero context" on a pure 0.0448
bpc cost. It is now a module-level function with a four-test gate, three of which
were verified to fail against the original expression. **The pre-registered H4 is
emitted exactly as written and is still defective for the reason the resolver
already recorded** — the decomposition is additive, so `|c0| > |total|` requires
the rung to *lose* outside zero context, which is why the cleanest rung fails it.
**Third instance of decision #9's class**, after `EXP_012`'s Y5 and `EXP_013`'s
N1. Referred.

## 21. `EXP_023` — four rungs, one number, and the optimiser buys the form

§19 produced the first arm of the session to beat its anchor and two numbers
saying the mechanism in its name was not established. `EXP_023` is the ladder
that separates them: **one fixed algebraic form,
`cur' = cur·(1 + kappa_c·DA_t)`, with only what drives `DA` changing.** Fifteen
runs, ~2.1 GPU-hours, none diverged, all gates passed including G8's unlock
clause on all three `nm_pos` runs, hash unchanged.

### 21.1 H6 first: the headline replicates out of sample

§4 required this before every other bar, because a replication and a power
increase are not the same number.

> **Fresh seeds 3–5 only, paired: −0.02535 bpc, t = −8.45, 3/3 seeds**, against
> §19's −0.03371. **The arm is real and it replicates, at 75 % of the size.**

### 21.2 And the rung that knows nothing is the rung that wins

| rung | what `DA` knows | bpc | vs anchor | `rms(kappa)` | `sd(DA)` | cost |
|---|---|---:|---:|---:|---:|---:|
| **`nm_const`** | **nothing** | **2.22016** | **−0.02975** | 0.9822 | 0.0000 | **1.12×** |
| `nm_pos` *(non-deployable)* | **when**, only | 2.23116 | −0.01876 | 0.9839 | 0.0794 | 1.14× |
| `nm_rolled` | general surprise | 2.22260 | −0.02732 | 0.9616 | 0.0442 | 1.47× |
| `nm_local` | **this sequence's** surprise | 2.22038 | −0.02953 | **1.0747** | **0.2283** | 1.47× |

**H1 held** (Welch t = −11.50). **H2 — "does time-variation buy anything at
all?" — came back at +0.00022**, forty-two times below the bar and pointing the
wrong way. §0 named this rung in advance as the one that could deflate the
headline, because `DA` held constant makes the arm `cur·(1 + kappa_c)`, which by
`EXP_008` Identity 1 is exactly a learned per-channel gain — the mechanism of
adopted arm #5. §5's cell, fixed before the run:

> **"`EXP_021`'s arm is a learned per-channel gain, and the time-variation is
> free."**

**The verdict is UNRESOLVED, not FAILED**, and that distinction is kept: this
establishes that time-variation is not worth 2σ, not that it is worth zero.
§6 item 1 stated in advance that `nm_const` is a **lower bound** on arm #5 — it
is a gain on the current at 1 of 2 layers — and that this cuts against §0's own
argument. It did. **`EXP_008`'s figure is not quoted as a result here**, per that
experiment's §6 item 2 and decision #6.

### 21.3 Neither control explains it, and one of them is refuted

**H3 FAILED, in the informative direction.** §1's argument was that `nm_rolled`
preserves the time index, so a misaligned RPE still carries "how far into the
window am I", and that `pos` isolates that leak. **`nm_pos` is 0.011 bpc WORSE
than `nm_const`, on 0 of 3 seeds, at t = +11.27.** Giving the modulator only the
position is worse than giving it nothing. The positional leak is not what
`nm_rolled` was riding.

**H4 UNRESOLVED at n = 6, and the effect halved.** §19 measured `local − rolled`
at −0.00577 and §11 item 2 priced n = 6 at ~0.6 GPU-h precisely to resolve it.
The power was bought; the difference came in at **−0.00221, t = −1.07, 5/6
seeds** — under half the n = 3 estimate. **Alignment is not worth 2σ, and more
seeds shrank it rather than sharpening it.**

### 21.4 H7 is the headline, and it is the second time this quantity has read flat

`rms(nm_gain)` across four rungs: **0.982 / 0.984 / 0.962 / 1.075 —
max/min = 1.1176**, against §17's **26** on the identical quantity and §19's 1.12
with two rungs. §4 fixed the reading in advance and §5 fixed its sentence:

> **"the optimiser buys the form, not the signal."**

**`sd(DA)` is what rules out the trivial explanation.** One rung has no variance
at all by construction and another has 0.2283 — 5.2× a third's. The signals
genuinely differ; the optimiser grows `kappa` ~10× from its 0.1 init in every
case and cannot separate them, and the bpc column agrees. **The bpc ladder and
the `rms(kappa)` ladder are the same finding stated twice**, which is what §19
could not reach with two points.

### 21.5 Four rungs, one total, four different routes — and none of them reach

| rung | total | `c = 0` | within reach | beyond horizon |
|---|---:|---:|---:|---:|
| `nm_const` | +0.02946 | −0.00773 | +0.03683 | +0.00036 |
| `nm_pos` | +0.02628 | **+0.03072** | −0.00455 | +0.00011 |
| `nm_rolled` | +0.02696 | **−0.06324** | **+0.09045** | −0.00025 |
| `nm_local` | +0.02940 | −0.01376 | +0.04370 | −0.00054 |

**The totals span 0.0032 and the zero-context components span 0.0940.** `pos` is
the mirror image of `rolled`. **H8's prediction — that a rung which knows less
keeps the within-reach shape and loses the rest — is wrong**, and is reported as
it fired: `nm_const` knows the least of the deployable rungs and has the
cleanest within-reach gain.

**Nothing on this ladder is beyond the horizon.** All four components are within
0.0006 of zero, inside §18.5's bitwise-copy floor. §19.3 found this for
`nm_local` alone; four rungs now say it. **A rank-1 time-varying gain on the input
current does not extend reach, whatever drives it** — and that is the most
transferable sentence in the experiment.

Which gives the practical one: **the same −0.0295 bpc is available at 1.12× or at
1.47×.** The Bernoulli predictor costs 32 % of a training run and buys +0.00022
in the wrong direction.

### 21.6 A confound `EXP_021` did not name, found by audit after both experiments ran

**Every `localdopamine` run ever trained — §19's and §21's — used
`Config.nm_b_init = -0.7`. `EXP_021` §2.6 certifies the arm at `b = -2.97`.**
The model class still defaults to −2.97; `build_model` forwards the Config value,
so the class default is dead code. Verified in each run's own `config.json`.

With `nm_a_init = 0.0` the init logit is `b` exactly, so
`phi = (rbar − sigmoid(b))·b`. At layer 0's measured init rate of 0.0488:

| `b` | `phi` | `phi/tau` | `sech²` |
|---:|---:|---:|---:|
| −2.97 *(certified)* | −0.0000 | −0.00 | **1.000** |
| −0.70 *(shipped)* | +0.1981 | +3.46 | **0.004** |

**The squash starts saturated**, so at step 0 the RPE rungs are algebraically the
`const` rung and the gradient into `(a, b)` is attenuated ~190× against the point
§2.6 certifies. The same arithmetic reproduces §2.5's own rejected number
(−0.8797 against its reported −0.879), which is what confirms the formula. It
also has the right size and direction for §2.7's unexplained FAINT residual,
which that section attributed to the `1/d` channel mean.

**What it bears on, narrowly.** H2 and H5 are the bars it touches. **It does not
explain the result**: `sd(DA) = 0.2283` at the final checkpoint, so `nm_local`
did desaturate and was genuinely data-dependent by the end. It cannot touch H1 or
H3 at all — `const` has no squash and `pos` carries no RPE. And `nm_tau` is
learned, which is §2.6's own answer to the 7× rate rise.

**Nothing is retracted and the value is not changed.** −0.7 is what every
committed figure used, and `CONTRIBUTING.md` §4 forbids moving a baseline to make
a diagnostic look better. It is pinned by a test verified to fail against −2.97,
and **the re-run at −2.97 is referred with a price (~0.6 GPU-h) and is Elliot's.**

### 21.7 The gap: a fourth axis on which the ladder is the same

| arm | gap, fresh | gap, carried |
|---|---:|---:|
| `if_anchor` (s3–5) | +0.00303 ± 0.00224 | −0.00473 ± 0.00220 |
| `nm_const` | **+0.01006 ± 0.00126** | **+0.00060 ± 0.00150** |
| `nm_pos` | +0.00875 ± 0.00199 | −0.00014 ± 0.00060 |
| `nm_local` (s3–5) | **+0.01004 ± 0.00216** | **+0.00107 ± 0.00164** |

**`nm_const` and `nm_local` differ by 0.00002 fresh and 0.00047 carried.** The
ladder is now indistinguishable on **four** measurements: test bpc, `rms(kappa)`,
the beyond-horizon component, and the generalisation gap. Only §21.5's
`c = 0`/within-reach split separates them, and no bar was pre-registered on it.

Every modulated rung widens the gap against the anchor by ~0.006–0.007 while
scoring better held-out — which, in the underfitting regime §12 established, is
what an increase in usable capacity looks like rather than a regularisation
story.

### 21.8 What the two experiments do not do

Neither adopts, ranks, or recommends anything. Neither rules an open decision.
Neither changes a hyperparameter, edits an adopted arm, or re-baselines. Twelve
referrals between them, all in their own §10s, all Elliot's.

---

## 22. `EXP_025` — the neuron buys reach, the width buys capacity, and the plain LIF stops reading at nine characters

Pre-registered at `d9f8d45` **before any driver, resolver or run existed**;
closed 2026-08-22. 12 runs, **0 divergences**, **~5.17 GPU-h**.
`experiments/logs/EXP_025_detached_scaling.md`. **Nothing is adopted, nothing is
ranked, no hyperparameter is changed, and no row of §8 is edited.**

§13.6 recorded that `EXP_014` — which bought more than every architectural arm
this phase built — **ran on the plain LIF**, because the adopted two-compartment
neuron could not be trained at 5.0M at all (§14). §16 removed that blocker at
n = 1 with no bar. This is the bar.

### 22.1 The scoreboard

| bar | verdict | the number |
|---|---|---|
| **H1** — the advantage exists at 5.0M | **HELD** | −0.13426 bpc, **79.4 se**, 95 % CI [−0.13914, −0.12939] |
| **H2** — the advantage is FLAT in width | **UNRESOLVED** | DoD **+0.00286**; the two σ readings disagree and §3.0's guard fires |
| **H3** — stability at n = 6 | **HELD** | **6/6**, 95 % CI [0.610, 1.000] |
| **H4** — σ at width | **HELD** | **0.002070**, **0.449×** the transferred 0.00461 |
| **H5** — the gain is not predominantly beyond-horizon | **FAILED** | zero + within carry **−18.6 %** |

Carried test bpc, n = 3 per cell, `docs/reports/data/exp_025_scaling_results.json`:

| arch | `d = 512` | `d = 1481` |
|---|---:|---:|
| `snn` | 2.25245 (sd 0.004683) | 1.99920 (sd 0.001721) |
| `twocomp_detach` | **2.11533** (sd 0.002363) | **1.86494** (sd 0.002368) |
| **Δ** | **−0.13712** | **−0.13426** |

### 22.2 H5 FAILED, and the failure is the phase's clearest result

The pre-registered prediction was that zero-context and within-reach together
carry ≥ 50 % of the gain. They carry **−18.6 %**. At `EXP_004` §10.3's cut of 7:

| component | bpc | share |
|---|---:|---:|
| total | +0.12182 | — |
| zero context | +0.00243 | +2.0 % |
| within reach | **−0.02504** | **−20.6 %** |
| beyond horizon | **+0.14443** | **+118.6 %** |

**The arm's entire gain is reach, and it pays for it at short context.** The
curves cross at **context 9**; below it the detached arm is *worse*, worst
**−0.04305 at c = 5**. The integer horizon agrees on 3 of 3 seeds:

| arm | per-seed horizon (2σ) | median |
|---|---|---:|
| `snn` `d = 512` | [7, 7, 7] | 7 |
| `snn` `d = 1481` | [8, 9, 9] | **9** |
| `twocomp_detach` `d = 512` | [47, 50, 48] | 48 |
| `twocomp_detach` `d = 1481` | [61, 52, 61] | **61** |

**The sharpest number is not in either table.** From c = 11 to c = 127 the plain
LIF at 5.0M improves by **+0.00032 bpc** — its curve is flat to sd 0.000156 over
116 characters. The detached arm improves **+0.09956** over the same span,
**311×** as much. *The plain LIF stops reading at about ten characters and stays
stopped, and 6.8× the parameters moves its horizon 7 → 9.* §14.3 located the
entire matched-size gap to the GRU in the failure to use context; this is that
failure measured directly.

**§8's rev-15 note is why this reading is allowed to stand.** That note flagged
positive beyond-horizon readings on arms that *lost* as compression-suspect, and
said the committed record could not adjudicate them. **This arm wins by 0.12182
and its horizon marker moves 9 → 61**, so the flag does not apply — the census's
open case, closed on a new measurement rather than by argument.

**The 118.6 % share is a property of the CUT, and is reported as one.** The cut
is at the *baseline's* horizon of 7 — `EXP_007` §5 item 4's standing limitation:

| cut | within reach | beyond horizon | beyond share |
|---:|---:|---:|---:|
| **7** *(pre-registered)* | −0.02504 | +0.14443 | **+118.6 %** |
| 9 *(the crossover)* | +0.00400 | +0.11540 | +94.7 % |
| 16 | +0.05866 | +0.06073 | +49.9 % |
| 61 *(the arm's own horizon)* | +0.11364 | +0.00576 | +4.7 % |

**The share is cut-dependent; the finding is not.** At every cut the gain lives
past the plain LIF's horizon, and the crossover at c = 9 is a fact of the curve.
`d = 512` decomposes the same way: total +0.12651, within reach **−0.00964**,
beyond horizon **+0.13099** (+103.5 %).

### 22.3 What this says together with `EXP_014`, and it is the point of the phase

§13.5's S4 held: **width buys per-step capacity and essentially no reach.** §22.2
measures the complement on the same tree, at the same widths, under the same
recipe: **the neuron buys reach and pays capacity.**

They are not competing explanations of one gain. They are two different gains,
and at 5.0M this arm has both. **What follows for §7.1's ranked list is offered
there and is not ruled here** — §7 item 2 recorded the within-reach component as
the one barely attacked, and this experiment says the adopted lineage attacks the
*other* one.

### 22.4 H2 UNRESOLVED — the guard fired, and it cost the cleanest headline

The difference-of-differences is **+0.00286 bpc**: the advantage *eroded* by a
quarter of the pre-declared margin across 2.76 octaves of parameters.

| reading | margin | verdict |
|---|---:|---|
| primary (§3.0's pre-declared constant) | 0.010648 | **FLAT** — CI₉₀ [−0.00422, +0.00994] lies inside |
| alternate (2 × measured se) | 0.006936 | **UNRESOLVED** |

They disagree, so §3.0's rule fires: *"a verdict that flips depending on whether
measured or transferred σ is used is reported as UNRESOLVED, and both are
published."* **The alternate margin is tighter only because H4 found σ smaller
than transferred** — the measurement that makes H4 hold is what denies H2 its
verdict. **No per-octave slope may be quoted from this row**; there is no middle
rung. Resolving it needs seeds *added*, never substituted.

### 22.5 H4 — σ above 735K parameters, and it re-scores `EXP_014`

**σ at 5.0M, measured for the first time in this project: 0.002070**, pooled
across both arms' `d = 1481` cells — **0.449×** the 0.00461 `EXP_014` transferred
to every one of its bars.

| cell | n | sd |
|---|---:|---:|
| `snn` `d = 512` | 3 | 0.004683 |
| `snn` `d = 1481` | 3 | **0.001721** |
| `twocomp_detach` `d = 512` | 3 | 0.002363 |
| `twocomp_detach` `d = 1481` | 3 | **0.002368** |

`snn` `d = 512`'s 0.004683 lands **0.5 % from the committed five-seed 0.00461** —
the instrument agreeing with itself on the one cell where a committed value
exists. **σ falls with width on the plain LIF (0.37×) and is flat on the detached
arm (1.00×);** no bar was placed on that and none is invented here.

**Re-reporting §13's bars against it, which H4 pre-registered as its
deliverable:** `EXP_014` S1 measured −0.25238 against a 10σ = 0.0461 bar and
reported **54.75 transferred σ**. Against the measured σ the same margin is
0.0207 and the same measurement is **~122 σ**. **`EXP_014`'s conclusion is
strengthened, not weakened — its transferred σ was conservative by 2.2×.** Its S2
pair gate moves 0.00922 → 0.00414.

### 22.6 H1 and H3

**H1 HELD** at −0.13426 bpc, 79.4 se, against a 2σ bar of 0.00753. §3 predicted
it would hold and predicted it would be uninformative, and both were right; it
exists so the headline carries a verdict rather than being quoted from §16.3's
`n = 1, no bar` table. Its guards are discharged — D1 and H5 are published beside
it.

**H3 HELD**: 6 of 6 runs completed 20,000 steps with zero non-finite losses, 3/3
at each rung, `6/6, 95 % CI [0.610, 1.000]`.

**`EXP_025` §3's stated Fisher p for the cross-tree comparison is wrong, and the
resolver computes it instead of quoting it.** The one-sided probability for 6/6
against `twocomp`'s committed 1/4 is `C(7,6)·C(3,0)/C(10,6)` = **7/210 = 0.0333**,
not 0.0048 — a factor of **6.9**. The pre-registration is **not edited**;
§22.8 records it. The direction survives and the force does not, and it was
always a secondary comparison H3's verdict did not rest on.

### 22.7 The result nobody pre-registered: four runs reproduce bitwise

`e25_detach_d512_s0/1/2` and `e25_snn_d512_s0` reproduce `EXP_017`'s and
`EXP_014`'s committed runs **bit-for-bit** — 2.1176906639749444,
2.1129641077769445, 2.1153212748345145 and 2.257471102255831.

No prediction was placed on this and it is carried as an observation. It
establishes that the **24 `Config` fields added since `EXP_017`** and
`three_factor_loss` entering the captured step at `tf_kappa = 0` changed
**nothing** for these arms — `train.py`'s "unchanged BY CODE PATH" comment
measured rather than asserted, and the K1b exemption justified empirically rather
than by inspection. **It is the mirror of §13.7's G1 failure.** It says nothing
about `twocomp`, which was not trained here.

### 22.8 Three defects in the experiment's own instruments, and a fourth caught by rev 15's tool

All three are decision **#9**'s class — a bar whose guard clause was written and
whose arithmetic was not — and all three lived in inline expressions where
nothing could assert on them.

1. **H3's Fisher p, in the *committed* pre-registration** (§22.6). **The first
   instance of this class to reach a committed log** rather than a resolver.
   Not repaired; computed forward.
2. **H2's equivalence margin**, which read literally makes TOST degenerate:
   equivalence would require `|DoD| < 0.1·se`, so UNRESOLVED would have been
   pre-determined by arithmetic rather than measured. Both readings emitted.
   Found before any bpc was read.
3. **H5's share was computed from ABSOLUTE components** — 3.393, a **339 %
   "share"**, on an `io_mall`-shaped decomposition, and it would have cleared its
   own 50 % bar on sign disagreement alone. Fixed **after** the `d = 512` bpc was
   read and **before H5 had ever fired**, and recorded in the artifact's own
   `resolver_defects_fixed_in_this_file` rather than only in a commit message.

The three statistics are now module-level functions with **17 tests**, verified to
fail before being trusted: **9 mutations, 9 caught.**

**The fourth was caught by the tool rev 15 repaired.** Appending §9 to the
pre-registration, a sentence *above* the results heading was also rewritten, and
`verify_prereg_hash.py` reported `live_file_agrees_above_results: False`. The
status block is the only change the recipe permits above that heading. Reverted;
it now reads **VERIFIED BY COMMIT** via the blob at `d9f8d45`, with the live-file
check **True**. **Rev 15's repair caught a live violation within a day of
shipping.**

### 22.9 Gates and cost

**All twelve runs green on K1a, K1b, G3, G4, K3 and G5**
(`docs/reports/data/exp_025_run_manifest.json`).

* **G5 — every run trained under the frozen recipe**, ten fields read back out of
  each `config.json`, including all six at 5.0M. **Decision #11 is not engaged,
  mechanically rather than by assertion**, at the exact width where the adopted
  `twocomp` arm diverges.
* **K1a carries zero exemptions** across all twelve — its reference is this
  experiment's own `e25_snn_d512_s0`, so every run came off one tree. **The
  strongest form of that gate the project has run.**
* **G3** — 735,437 / 737,485 / 4,997,099 / 5,003,023, all exact. **G4** — peak
  VRAM reproduces the calibration to four decimals against a 3.0 GiB alarm.
  **K3** — ratios 0.933–1.043; A4 never approached its 1.5 halt.

| item | projected | realised |
|---|---:|---:|
| ladder, 12 runs | 4.84 GPU-h | **4.858** |
| horizon sweep + pre-flight | ~0.25 | ~0.31 |
| **total** | ~5.1 | **~5.17 GPU-h** |

**The cost model was right to 0.4 %** — the first Phase-4 estimate that was, and
it is a measurement (`exp_025_cost_calibration.json`) rather than a descendant of
the constant §7.1 withdrew.

### 22.10 What this does not do

It adopts nothing — **decision #5**'s shape, and §5 item 7's caveat stands: this
measures `twocomp_detach`, and §16.2's A6 row says the detached rule trains
*further* inside the dangerous region of weight space. It ranks nothing. It rules
no open decision, and **engaged none of the six** — G5 is the mechanical proof for
#11. It authorises no phase. Six referrals are in `EXP_025` §10, all Elliot's.

---

## 23. Changelog

| Rev | Change |
|---|---|
| 16 | 2026-08-22. **`EXP_025` folds in as §22: the neuron buys reach, the width buys capacity, and the plain LIF stops reading at nine characters.** (a) **§22 added, pushing the changelog §22 → §23** — the same insertion slip revs 3 (f), 4 (g), 6 (h), 8 (f), 9, 10, 11, 12 (a), 13 (a) and 14 (a) each record for their own. (b) **H1 HELD at −0.13426 bpc, 79.4 se**, 95 % CI [−0.13914, −0.12939], on 3 seeds per cell — the first bar this project has put on the two-compartment advantage at 5.0M, against §16.3's `n = 1, no bar`. (c) **H5 FAILED, and the failure is the result.** Zero-context and within-reach carry **−18.6 %** of the gain against a pre-registered ≥ 50 %; beyond-horizon carries **+118.6 %**, and the arm **pays −0.02504 within the baseline's own reach**. The curves cross at **context 9**, worst −0.04305 at c = 5. (d) **The integer horizon agrees on 3 of 3 seeds**: `snn` 7 → 9 across 6.8× the parameters, `twocomp_detach` 48 → **61**. §8's rev-15 note flagged positive beyond-horizon readings on *losing* arms as compression-suspect and said the record could not adjudicate them; **this arm wins by 0.12182**, so the flag does not apply. (e) **From c = 11 to c = 127 the plain LIF at 5.0M improves +0.00032 bpc**, flat to sd 0.000156 over 116 characters, against the detached arm's **+0.09956** — **311×**. (f) **The 118.6 % share is a property of the CUT and is reported as one** (94.7 % at cut 9, 49.9 % at 16, **4.7 %** at the arm's own horizon of 61); the share is cut-dependent, the finding is not. (g) **With §13's S4, the pair is now explicit and measured on one tree: width buys per-step capacity and no reach; the neuron buys reach and pays capacity.** (h) **H4 measures σ above 735,437 parameters for the first time in this project** — **0.002070**, **0.449×** the transferred 0.00461 — so §13's S1 was **~122 σ** rather than 54.75 and **`EXP_014`'s conclusion is strengthened**; σ falls with width on the plain LIF (0.37×) and is flat on the detached arm (1.00×). (i) **H2 UNRESOLVED because its own guard fired**: DoD **+0.00286**, primary margin says FLAT, the measured-σ margin says UNRESOLVED, and §3.0's disagreement rule resolves it UNRESOLVED — **the measurement that makes H4 hold is what denies H2 its verdict**. (j) **H3 HELD, 6/6, 95 % CI [0.610, 1.000]**, and the pre-registration's stated cross-tree Fisher p of 0.0048 is **wrong — 7/210 = 0.0333**, computed forward rather than repaired. (k) **Four `d = 512` runs reproduce committed runs BIT-FOR-BIT**, which nothing predicted: the 24 `Config` fields added since `EXP_017` and `three_factor_loss` at `tf_kappa = 0` changed nothing, the mirror of §13.7's G1 *failure*. (l) **Three defects in the experiment's own instruments**, all decision #9's class, **the first of them reaching a COMMITTED log**; and **a fourth was caught by the tool rev 15 repaired** — appending §9 also rewrote a sentence above it and `live_file_agrees_above_results` read False until reverted. (m) **All twelve runs green on K1a, K1b, G3, G4, K3 and G5**; **G5 is the mechanical proof that decision #11 was never engaged**, at the exact width where the adopted `twocomp` arm diverges; **K1a carries zero exemptions**. (n) **12 runs, 0 divergences, ~5.17 GPU-h against ~5.1 projected — the cost model was right to 0.4 %**, the first Phase-4 estimate that was, and **zero lines were added to `src/`**. **No arm is adopted, no candidate ranked, no hyperparameter changed, no phase authorised, and no §8 row edited.** |
| 15 | 2026-08-21. **No experiment folds in; four corrections, and the first of them corrects rev 14's own §8 note.** (a) **The pre-registration audit was reading one key, one filename and one placeholder, and it had no second route.** `scripts/exp/verify_prereg_hash.py` read `prereg_sha256_before` while `EXP_013`–`EXP_018` stamp `prereg_sha256_before_first_run`; it parsed `exp_007_008_run_manifest.json` into a nonexistent id `007`; it counted three **unstamped** manifests in the denominator of *stamped* pre-registrations; and its `PLACEHOLDERS` table omitted the variant used by `EXP_013` — the one log `CONTRIBUTING.md` §2 names as the exemplar, which states that recipe verbatim at its own `:424-431`. **A tool written to catch this class contained two instances of it, and that is recorded rather than quietly repaired.** (b) **A second route is added and it is the stronger one**: if any commit in a log's history holds bytes that hash to the stamp, the pre-registration verifies against **an object in the repository** rather than a guess about layout, and the earliest such commit is named. **The record reads 7/9, not 2/12** — correcting the three defects alone gives 2/9, so the denominator and the numerator moved for different reasons. The four newly commit-verified blobs are the **pre-registration commits themselves**, §9 empty in all six, so **no false positive is possible in that set**. (c) **`EXP_020` and `EXP_021` are unchanged and rev 14's finding stands**, now with a mechanical reason: one commit each, and it already carried §9. (d) **Every commit-verified row also reports `live_file_agrees_above_results`, and all six read `True`** — a matching blob proves the *stamp* is honest and says nothing about the file on disk, which is the guarantee §2 wants. That check needed one fix first: `EXP_022`'s `**Status: CLOSED …**` wraps over **six** lines, so a first-line-only filter reported five continuation lines as an edited hypothesis and **would have convicted a clean log**. (e) **`EXP_024` §4's prose generalises to "any arm" a band the same document scopes correctly to six**, and on the wider committed record that is false: `scripts/audit/11_beyond_horizon_census.py` reads **41 blocks across 12 artifacts under both spellings, zero GPU**, and of **24 primary arm-gain readings 7 sit outside ±0.0016**, three clearing H1's own 2σ bar. **The cleanest counterexamples are not the largest**: `tokenshift` reads **+0.00565 on an arm that WON**. (f) **Two of the three largest are arms that LOST**, so the census flags them `compression_suspect` — **and checks the flag rather than leaving it standing**: the horizon marker moves for both (`[9, 10, 9]` and `[9, 8, 9]` against `[7, 7, 7]`), while `if_binin` lost more and read **−0.00477**. **Two markers agreeing is two markers agreeing**; the integer horizon has no σ anywhere in this project, and **it is left unresolved**. (g) **`EXP_024` modulates layer 1, that layer's firing rate at init is 2.98e-07, and the screen's silence guard is an exact equality** — so it clears, and every layer-1 parameter is stamped `in_silent_layer: false`. **§2.6's PASS is not disturbed**: `02_baseline_report` §5.3 measures the escape by step 50 and §5.4 already ruled that a resolved diagnostic does not warrant changing an init. What is missing is any mention of it in §§2.2, 2.6 or 6. The screen's layer-0 `k_gate` entry is a **harness** property — `build_screen_model` grafts every candidate onto all layers — and the gate holding the arm to one modulated layer is **G2**, not G4. (h) **§8's `1.0117` is `EXP_002`'s kernels-per-timestep, identical to the N0 baseline's** (both `kernels = 259`), not a step-time ratio; N1's `ms_vs_baseline` is 0.972 but is a **median of three** with 7.3 % spread and overlapping ranges, so **the honest reading is parity** and the ~1.1 GPU-h survives either way. (i) **`EXP_024_gated_decay.md` is NOT edited** — it is committed at `aff0d49`, and a commit rather than a stamp is what fixes it; it has no run manifest yet, so `git diff` is the only check available for that log today. (j) **The §9 gate table is not re-run**, on revs 5's and 7's precedent: this revision closes no experiment, adopts nothing, and touches no code any row in it describes. **No §8 row is edited, no decision is ruled, no bar is proposed, widened or repaired, and corrections 3 and 4 were found only because the census in correction 2 was built and run. None of the four required a GPU.** |
| 14 | 2026-08-20. **`EXP_022` and `EXP_023` fold in as §20 and §21: the cost of a binary input was a cost of its DENSITY, and the modulator that won §19 wins just as well when its signal is a constant.** (a) **§20 and §21 added**, pushing the changelog **§20 → §22** — the same insertion slip revs 3 (f), 4 (g), 6 (h), 8 (f), 9, 10, 11, 12 (a) and 13 (a) each record for their own. (b) **The masthead is corrected from Revision 12 to Revision 14**: rev 13 shipped a rev-13 changelog row, rev-13 status text and two new sections behind a masthead still reading *Revision 12, 2026-08-14*, and it is the only revision since 7 that added no masthead blockquote. Recorded rather than quietly fixed. (c) **A fully binary input code costs 0.00955 bpc at `q = 0.05` against 0.04569 at `q = 0.50`** — 79 % of §18's binarity cost recovered at **identical parameter count** and **0.97× wall-clock**, on 3/3 seeds, H2 at 2.8× its bar and H3 at 3.9×. (d) **82.8 % of that cost sits at zero context and all three sparse rungs recover at exactly that component**; what separates them is what they give back within reach (−0.0314 / −0.0163 / −0.0001), and no bar was pre-registered on that gradient. (e) **The winning code is LOSSY** — 39–42 of 205 characters share a code and 26–28 emit the all-zero one — and it beats the code that distinguishes every character. (f) **`EXP_022` §6 item 3's diagnostic fired**: realised densities are 0.50 / 0.14 / 0.025 / 0.013, not the 0.50 / 0.34 / 0.10 / 0.05 asked for, with a named mechanism (weight decay against a one-sided threshold). (g) **H6 UNRESOLVED**: the all-layer readout is worth its parameters (H5, −0.0216, t = −8.49) and **not** worth more than the same parameters spent on width — but §20.5 shows the two reach the same total by opposite routes, and the readout's **+0.0431 within reach is the largest such movement of the phase**. (h) **§19's headline REPLICATES out of sample** at −0.02535 on three fresh seeds, t = −8.45. (i) **And a four-rung ladder takes its mechanism away for the second time**: `nm_const`, whose driving signal is a constant, is worth −0.02975 against `nm_local`'s −0.02953 — H2 at **+0.00022**, forty-two times below the bar and pointing the wrong way — so §5's pre-written cell fires: *EXP_021's arm is a learned per-channel gain, and the time-variation is free*. (j) **H3 FAILED and refutes the positional-leak explanation**: `nm_pos` is 0.011 WORSE than `nm_const`, 0/3 seeds. (k) **H4 was priced at n = 6 and n = 6 halved it** — −0.00221 against the n = 3 estimate of −0.00577, still unresolved. (l) **H7 flat at 1.1176 across four rungs** against `EXP_018`'s 26, with `sd(DA)` spanning 0.000 to 0.2283 — the optimiser buys the form, not the signal. (m) **Nothing on either ladder reaches beyond the horizon.** (n) **A confound found by audit after both experiments ran**: every `localdopamine` run used `nm_b_init = -0.7` (init `sech²` = 0.004) against the −2.97 `EXP_021` §2.6 certifies (0.777). Pinned by a test verified to fail, recorded in §21.6, **not changed**, and the re-run referred at ~0.6 GPU-h. (o) **A protocol deviation recorded beneath §8**: §18's and §19's resolvers were never committed before their bpc was read; §20's and §21's were. (p) **A resolver defect caught BEFORE commit** — H4's post-hoc replacement had its sign inverted and would have announced a gain on a pure cost; fixed free, gated by four tests, three verified to fail against the original. (q) **No row of §8 is edited, no rank is applied, no arm is adopted and no open decision is ruled.** |
| 13 | 2026-08-19. **`EXP_020` and `EXP_021` fold in as §18 and §19: the model's two ends are measured for the first time and every arm loses, and a feedforward modulator beats its anchor for a reason its own control takes away.** (a) **§18 and §19 added**, pushing the changelog **§18 → §20** — the same insertion slip revs 3 (f), 4 (g), 6 (h), 8 (f), 9, 10, 11 and 12 (a) each record for their own. (b) **`layers.0.weight` is provably redundant with the embedding** — `cur0 = W0·E[x] + b0 = T[x] + b0` exactly, measured at 5.19e-08 relative — and it is **35.7 % of the model**. **Removing it costs +0.09459 bpc against the +0.0582 §13's slope predicts for a generic cut that size: 1.63× more than the parameters are worth**, and spending the freed budget on width does not recover it (`foldwide` +0.01326 *worse*). This is **§6.5's `EXP_008` result one layer earlier and larger**, and it is now two independent measurements bearing on open decision **#6**. (c) **T1 PASSED at exactly 0.0** — a full 20,000-step nesting run reproducing its anchor bitwise in both protocols. (d) **A binary input code costs +0.04569 bpc at IDENTICAL parameter count, and 83 % of that is at zero context** — a per-character expressiveness price, not a memory price (−0.0029 within reach, −0.0048 beyond). The committed model is binary on **1024 of 1536** inter-unit wires; the arm is 1536/1536, and the centring is a provable reparameterisation of `layers.0.bias`, so it is a **true spike input** and not a two-valued analogue one. (e) **The fold's cost is a REACH cost** — 86 % within-reach — which no argument from parameter counting predicts, and `foldwide`'s headline is a real +0.0397 zero-context gain plus a larger −0.0705 within-reach loss that nearly cancel. (f) **σ transfer CHECKED and HELD for the first time**: 0.004683 on this tree against the transferred 0.00461, ×1.02. §3's rule is unchanged — the point is that the check was run. (g) **The per-context instrument failed its own negative control, and that is a finding.** `if_nest_s0` is **bitwise** its anchor and still reads **120 of 128 contexts "significantly worse"**; the bars fall to 2.4e-05 where single-seed variation is ~0.005 — **up to 80× below it**. `n_contexts_significantly_worse` is **unusable at n = 3**. Fresh measured evidence for open decision **#2**; **no redefinition is proposed** and every later experiment quotes this floor beside the statistic. (h) **`EXP_021`'s `nm_local` beats its anchor by −0.03371 bpc, t = −8.95, 3/3 seeds, 3.7× the bar, at 1.49× with no second forward pass** — the first arm since `EXP_008` to do so, and in the slot **#3 admits**. (i) **And its own controls take the mechanism away**: H2 UNRESOLVED at −0.00577 against the batch-rolled control, and H3's `rms(nm_gain)` ratio is **1.12** against §17's **26** on the identical quantity. §5's pre-fixed cell stands: *a bounded time-varying gain pays and the RPE is not why*. (j) **The gain is ENTIRELY within reach** — +0.0559 inside the baseline's 7-character horizon, **−0.0005 beyond it** — which is exactly where §14.3 located the whole matched-size gap to the GRU, and where §18's decomposition puts the two-compartment neuron's gain too. Complementary or redundant is **not measured** and is referred. (k) **Not a generalisation effect**, measured with §12's committed instrument: the arm's gap is within ~0.004 of the anchor's on both protocols — the **opposite** of §17, whose arm widened its gap by 0.0057 while losing. (l) **Dopamine on the LEARNING RULE is actively harmful**: `tf_neg` +0.04826 worse (t 29.34, 0/3), the harm IS alignment-dependent (`tf_roll` neutral at −0.0008), and learn-from-errors beats consolidate-success by 0.0377 though both lose. (m) **A pre-registered statistic was INVALID and §17's own referral predicted it** — H7 read the train→test gap from a logged loss that is the *weighted* objective for the `tf_*` arms, returning −0.354 and +0.673. **Second experiment caught by `EXP_018` §10 item 11, first caught after it was written down**; re-measured with the committed instrument. **It should be ratified into `CONTRIBUTING.md`.** (n) **Two derivations forced by measurement before any run**: a summed `phi` is +101 nats at `d = 512` and saturates `tanh` to exactly 1.0, which would have made the arm a constant per-channel gain — i.e. **adopted arm #5 under a new name, training and scoring plausibly**; and the literal `H − S` measured **2.01×**, worse than the arm it exists to beat, until the exact cancellation `phi = mean_c (s − p)·logit` brought it to 1.49×. (o) **Three defects found, none touching a number**: a closed form missing `nm_log_tau` (caught by strict-xfail *before* the gate that would have aborted on it), `iface_read_lags > 1` silently decoding zeros at `L = 1`, and a resolver reading per-context bars from a wrapper — **caught only because two implementations of one statistic disagreed**. (p) §1's status table updated: experiments closed gains `EXP_020` and `EXP_021`; **ranked candidates closed unchanged at 6 of 14** — neither is on the 18-item list; **arms adopted unchanged at 2**; **GPU-hours ~15.8 → ~19.7**. (q) **`EXP_019` is pre-registered and NOT RUN**, and is recorded here rather than given a section: its zero-GPU instruments produced the regional census committed at `19cdd4a`, and its training ladder never started. (r) **No §8 row is edited and none is added.** #2 gains fresh evidence in prose beneath the table, per the rev-5 rule; #6 gains a second independent measurement, also in prose. **Six decisions remain open.** (s) **No arm is adopted, no rank is renumbered, no tolerance is set, no phase is authorised, and no recipe term moves.** |
| 12 | 2026-08-14. **`EXP_018` folds in as §17: a broadcast reward prediction error is findable, wanted, and harmful.** (a) **§17 added**, pushing the changelog to §18 — the same insertion slip revs 3 (f), 4 (g), 6 (h), 8 (f), 9, 10 and 11 each record for their own. (b) **The derivation removed a hyperparameter before the arm existed**: `E[-log p] = H(p)` exactly, so the RPE is zero-mean under the model's own belief with no baseline, no EMA and no critic — and, decisively for the implementation, no sequential scan over `t` inside a CUDA-graph capture. (c) **Not a reparameterisation**, unlike `EXP_008`: `DA_t` varies with `t` and folds into no weight, so decision #6 does not arise for it. (d) **D1 UNRESOLVED by 3.6e-06 bpc** and reported as a miss (`CONTRIBUTING.md` §3), in the direction of harm. (e) **The pre-registration's own defect, measured rather than referred because it hurts**: an unpaired Welch test on paired data — a seed here fixes both the initial weights and the data order — and the paired test gives 5/5 seeds worse, −0.006215, t = −7.477, p = 1.7e-03. The verdict still stands as it fired. (f) **The 26x**: `rms(k)` 0.1950 for the aligned RPE against 0.0075 for the same signal rolled across the batch, stable across seeds — the optimiser can tell them apart and wants the real one. (g) **Identical train fit (+0.00043, p 0.63), worse test fit (+0.00611, p 0.0017)**; the gap widens 0.0057, roughly doubling §12.2's 0.0059. Hypothesis offered and not established: `phi` is a function of the model's own parameters, so conditioning on it may open a self-referential channel. (h) **The decomposition is two effects of opposite sign** — −0.0246 at c = 0 where the mechanism is structurally inactive, +0.0427 at c = 3, ~+0.006 asymptotically; worse at 125 of 128 contexts, so `EXP_001`'s dominance criterion fails at short contexts. (i) **Cost 1.708x at training AND inference**, measured against a same-session anchor, against `EXP_007`'s already-dominated 1.49x. (j) **Adopts nothing, ranks nothing, closes no ranked candidate.** It raises a **fourth** I5 boundary case for #3 — and **#3 is then RESOLVED at this revision on Elliot's instruction of 2026-08-14**, the first §8 row a revision has closed since rev 6: feedforward-driven modulators admitted, head-driven ones diagnostic-only, extension ratified in `01_reconnaissance.md` §4.6 beside the 2026-08-01 original. Ruled after the evidence existed and stated as such; the criterion is structural and costs the project nothing it wanted. Six post-run referrals. (k) **A claim §17.4 made was withdrawn inside the same revision.** `EXP_018` §10 item 7 referred the proper generalisation-gap measurement to a later experiment; it was run here instead, with §12's own committed instrument over all sixteen checkpoints. It says the arm is worse on the **train slice** by **+0.00522** (t 7.23, p 0.0019, 5/5 seeds) against **+0.00621** on test, so **the gap does NOT widen** (p 0.16 carried, 0.74 fresh). "The gap roughly doubles" is withdrawn: **this is a capability loss on seen and unseen text alike, not a generalisation failure**, and the self-referential-channel hypothesis loses the observation that motivated it. The correction sits beneath the standing original, per §14.5's precedent. (l) **Two train-side estimators disagree by ~0.005 in opposite directions for the two arms** — the anchor's running loss sits 0.0028 above its fresh train-slice bpc, the arm's 0.0026 below — **chased and resolved rather than referred**: a logged training loss is measured at pre-update weights, and the anchor was still improving over the logged tail by 0.00619 against the arm's 0.00285 — an asymmetry of +0.00334 ± 0.00096 (p 0.026) that cancelled the effect. Re-scored on the SAME batches at final weights the arm is worse by **+0.00377** (t 4.22, p 0.0135, 5/5), and `eval()`/`train()` agree **bit-for-bit**, so there is no mode-dependence and no undeclared state in the arm — a third independent slice agreeing with the other two. **Standing lesson: a running training loss is not a stand-in for a train-split bpc.** |
| 11 | 2026-08-11. **`EXP_017` folds in as §16: the reset Jacobian can be bounded without changing the forward, the bounded arm trains at 5.0M, and two of three fresh controls died.** (a) **§16 added**, pushing the changelog to §17 — the same insertion slip revs 3 (f), 4 (g), 6 (h), 8 (f), 9 and 10 each record for their own. (b) **The derivation is a stronger theorem than §15's.** Detaching the fast pole's reset sends `∂vf_out/∂v` to zero, which removes `vf` from **both** entries of the 2×2 per-step Jacobian at once, leaving `diag(beta_f·(1−s), beta_s)` — **diagonal**, so no non-normal transient, and bounded by `beta_f = 0.5`, **strictly tighter than the plain LIF's own 0.5123596**. The slow pole's rate is unchanged, so the memory mechanism is untouched. (c) **The forward is the same compiled kernel**, imported rather than re-declared, so the arm is parameter-**identical** to the adopted one and its checkpoint evaluates as `arch="twocomp"` **bit-identically** — asserted at `== 0.0` where §6.3 and §10.1 could only report residuals of 1.9e-03 and 1.05e-06 bpc. **Decision #6's tolerance question does not arise for this arm.** (d) **H1 HELD**: on the batch that killed the run, the adopted rule expands for **85 consecutive timesteps worth 10^16.00** and the detached rule, on the **same membranes from the same forward pass**, gives `max|g| = 0.5` and a longest run of **0** — every leg, every layer, both batches. T1, T1b (10/10 cells against `EXP_016`'s committed artifact), T2 and G4 (200 quantities, 0 differing) all hold. (e) **H2 SURVIVES**: 20,000 steps, **0 non-finite losses**, past step 5138, under the **unchanged** frozen recipe — 24 fields verified per run. Carried **1.86290** at parameter-identical 5,003,023, against the plain LIF's 2.00073 and the GRU's 1.54699; the matched-size gap closes **30.4 %** where §14.2's width closed 6.6 %. **Marker, n = 1, no verdict.** (f) **A6, unpredicted and post-registered as an observation**: this arm's *own* weights would, under the hard rule, expand for **84 steps worth 10^27.58** — worse than the arm that died — with `max|w·vs|` 445.9 against 356.6. **The fix does not avoid the dangerous region of weight space; it is indifferent to it.** (g) **Two of three fresh `twocomp` anchors diverged at `d = 512`**, steps 11500 and 12500, at the width §2's seven-seed evidence comes from. Seeds not replaced. **H3 resolves NOT RUN** by its own guard — the first prospective application of §14's referred DIVERGED/NO RESULT correction. Markers: detach **2.11533 ± 0.00237** (n = 3), **−0.00337** from the committed 2.11869; σ re-measured at 0.00237, 2 df. **The stability difference is NOT established** — 2/3 vs 0/4, Fisher **p = 0.40**, intervals quoted. A post-hoc probe finds the §15 mechanism present at 735K on both dead runs (up to 21 steps, 10^4.21). (h) **H4 REACH REDUCED** on **one** surviving pair at Δh = −2, against a within-arm spread of 3 and no σ_horizon anywhere in this project; median 48 against the committed 47 and the baseline's 7. Reported as it fired. (i) **H5 BREACH at 1.127× — against a denominator that is itself a diverged run**, 14,862 of whose 20,000 steps were non-finite. The bar stands unrepaired; a controlled paired re-measurement gives 1.034×, `d = 512` runs are **faster** than same-session anchors, and VRAM is identical to the byte. **`CONTRIBUTING.md`'s ratio rule needs a second clause: the reference must be a valid measurement.** (j) **D09 escaped the first mutation campaign** — `>` for `>=` in a backward kernel, invisible to a bound because **both branches land inside the bounded set**. Closed with a constructed equality case; **whole campaign re-run at 59/59**. (k) **No row of §8 is edited**, nothing is adopted or ranked, no hyperparameter moved, and Phase 5 is not authorised. ~2.0 GPU-hours; **~12.9 of ~30** spent. |
| 10 | 2026-08-11. **`EXP_016` folds in as §15: the plain neuron's backward provably cannot explode, the adopted one's demonstrably does, and rev 9's "not a gradient explosion" is corrected.** (a) **§15 added**, pushing the changelog to §16 — the same insertion slip revs 3 (f), 4 (g), 6 (h), 8 (f) and 9 each record for their own, caught in this pass rather than left for a rev 11. (b) **A closed form, derived before any GPU was touched.** Both neurons carry a per-step reverse-recurrence multiplier `beta*dv`. The plain LIF writes `dv = (1 − s) − v_pre·sg(v_pre − thr)` — **the same variable** in the multiplier and inside the surrogate — which is self-limiting, so at the frozen constants `max|beta·dv| = 0.5123596 < 1` **for every finite membrane at every width**. `twocomp` writes `dv = (1 − sh) − vf·sgd(v_pre − thr)` with `vf = v_pre − w·vs`: **different variables**, and the bound breaks at `|w·vs| ≈ 1` because `vs` is never reset. **This is the first explanation this project has had of why the plain LIF trains at 5.0M.** (c) **Measured over 48,529,408 sites per layer on five committed checkpoints**: the three `snn` legs return **0.51236 at every width and every layer** and **not one site exceeds the bound**; `twocomp` reaches **8.95778 at `d = 1481`**, 17.48× that ceiling, with `frac(|g| > 1)` rising **10.3×** with width and `max|w·vs|` reaching 279.66. T1 (the probe validated against the closed form) and T2 (local scans **bitwise** equal to the committed eager references) both hold. (d) **Leg B measured the failing step boundary by boundary, on both dispatch paths, with the loss bit-identical at `1.4197630882263184`**: the cotangent entering layer 1's scan is **1.4639e−04** and the one leaving layer 0's is **6.1173e+37** with 140 NaN + 1 inf — **layer 1's recursion amplifies by 10^18.88, layer 0's by 10^23.03, 10^41.62 end to end inside ONE backward pass**. **P6 = BORN IN THE RECURSION**, and R10 separation now holds one level deeper than `EXP_015` could take it. (e) **CORRECTION to rev 9.** §14.5's *"and not a gradient explosion — the largest gradient in the six steps before death is 2.0e-02 and the clip never fires"* is **wrong**, and rev 9's own committed artifact contained the refutation (`layers.1.weight` at **1.676e+15**, eager `grad_abs_max_finite` at **7.79e+29**, both at step 5138). Both cited facts are true and neither supports the conclusion, because both are measured *between* optimiser steps: the explosion is *inside* the backward, and **"the clip never fires" is a CONSEQUENCE of it** — `clip_grad_norm_` reads a norm that is already NaN. The rev-9 sentence is left standing with the correction beneath it, per `CONTRIBUTING.md` §3 and rev 9 (i)'s precedent. **"Never layer 1" stands about non-finiteness and misleads about magnitude**: layer 1 carries 1.10e+15 where it carried 1.46e−04 one boundary earlier. (f) **A pre-registered bar misfired for the fourth time in this phase, and this one was authored today.** P3 placed its bar on the product over all 256 timesteps and resolved **REFUTED AS SUFFICIENT** — 0 of 189,568 chains expand net — but the adjoint is injected at *every* timestep, so the quantity averages the mechanism away by construction. **`EXP_016` §4.0 and §6 item 4 both say that quantity "is NOT the gradient", and the bar was placed on it anyway.** Reported unrepaired. **P4 also failed**, informatively: layer 1 carries the *heavier* per-step tail, which (d) explains. (g) **POST-HOC, labelled so** (`012_posthoc_pileup.py`'s precedent), amending no bar: replacing the whole-unroll product with the maximum contiguous window, **and re-running on the batch the run actually died on** — Leg A read batch 0, **a defect in its design, not only in P3's quantity**. On the fatal batch layer 0's recurrence **expands for 85 consecutive timesteps** (10^16.00) against 3 steps (10^0.66) for the same arm at 735K, while the plain LIF is at **exactly zero in every cell, both batches, all three widths**. (h) **The residual is stated, not absorbed**: window (10^16.00) plus coupling (≤10^4.4) against a measured 10^23.03 leaves **~2.6 orders of magnitude unresolved**, with three candidates named and none measured. **No claim is made that this is the whole cause.** (i) **§8: no row edited and no row added.** #11's *form* is re-evidenced in prose: `dv` contains **no learning rate, no schedule, no weight decay and no gradient clip**, and §15.4 shows the clip cannot act on this failure — so "may the frozen recipe be re-derived at a new size?" may be the wrong question to rule on. **No term is changed and no remedy is proposed.** Table stays eleven rows, four resolved, seven open. (j) **§7.1 gains a rev-10 note, offered and not applied**: ranks 3–5 are still not demoted; a row that is not on the table has appeared underneath it (anything bounding the recurrence is a **new arm**); and the GPU-h column gains a third measured row at ~0.8, of which ~0.37 bought no number. (k) §1: experiments closed gains `EXP_016`; **GPU-hours ~10.1 → ~10.9**; the divergence row records that the shared signature now has a mechanism. (l) §9 re-run: `ruff` clean, fast subset **277 passed / 1 skipped**, full suite **415 passed** (396 → 415 for `tests/test_reset_jacobian.py` and `EXP_015`'s additions); `README.md` and `CONTRIBUTING.md`'s quoted counts corrected **259 → 278** (fast, which counts the skip) and **396 → 415** (full) — the fast figure had been stale since rev 9, which measured 267/1 and did not propagate it. **Gate rows not re-run**: rev 10 changes no kernel, no captured graph and no numeric path, and G1 (`git diff --stat src/snn/` empty) is checked and recorded. **No earlier number, verdict or threshold changes beyond the correction at (e). No arm is adopted, no size is chosen, no tolerance is set, no phase is authorised, and no recipe term is changed.** |
| 9 | 2026-08-10. **`EXP_015` folds in as §14: the adopted arm does not train at width, the anchor scales with the spiking model, and the difference between them is reach rather than capacity.** (a) **§14 added**, pushing the changelog to §15. Four legs at `d_model = 1481`, seed 0, `arch` the only variable, all within **0.18 %** of 4,997,099 parameters — width-matching *is* parameter-matching at this width because each arm's extras are `O(d)` against an `O(d²)` stack, and the GRU is matched by construction. **Both arms containing the two-compartment neuron DIVERGED** — `twocomp` at step 5138, `twocomp_threshold` at ~2000 — while the plain LIF and the GRU at the identical width, seed, recipe and tree trained cleanly; at 735K these same arms trained across 7 and 2 seeds. (b) **P3, the leg pre-registered with no bar, is the result.** The anchor at matched size scores **1.54699** carried against the spiking model's 2.00073, so the gap is **0.45374** where at 735K it was 0.48570 — **width closes 6.6 % of it**. Width bought the spiking model −0.25238 and the anchor −0.22042. **Capacity is not what separates this architecture from its anchor.** (c) **§14.3 locates the difference**: with width the anchor's 2σ memory horizon goes 57–60 → **97** and the spiking model's goes 7 → **8**, and the absolute per-context curves say it without any horizon definition — the two 5M models are **0.051 apart at zero context**, 0.152 apart by `c = 8`, and **0.408 apart by `c = 64`**, with the spiking curve flat since `c ≈ 8`. The entire matched-size gap is a failure to *use* context. The one spiking arm that ever bought reach (`twocomp`, horizon 47) is the arm that will not train at width. (d) **P1 and P2 fired and misled, for the third time in this phase.** Both resolve UNRESOLVED — which §3.0 defines as a statement about statistical *power* — for two runs that produced no number at all, with seeds recommended as a remedy that cannot work against a deterministic divergence. Reported unrepaired per `CONTRIBUTING.md` §3; the corrected **DIVERGED / NO RESULT** verdict is referred, and #9's row is left as rev 6 wrote it. (e) **§14.5: the divergence is `compose_s0`'s mechanism**, reproduced deterministically at step 5138 by `011_chase_compose_divergence.py` **reused unchanged** but for a `--tag`. Origin backward; forward and loss finite; fp32 and fp64 norms equal; the eager path producing an **identical** NaN in the **identical** five tensors (R10 clean); always layer 0 and the embedding, never layer 1; largest gradient in the six steps before death **2.0e-02**, and the clip never fires in the whole run. First occurrence on the plain adopted arm rather than the composed one, and now reproducible **138 steps from a committed checkpoint instead of 17,598**. (f) **A hypothesis was stated and falsified in place**: state magnitude is *not* the discriminator — layer 1's state grows **13.30×** for the plain LIF and **14.96×** for `twocomp` across the same width change, the *surviving* arm growing more, and a plain LIF carrying `|state|` up to 556 trains 20,000 steps without incident. What survives is a lead only: at layer 0, `twocomp` is 2.84× the plain LIF, carried by the reset-shielded slow pole. "The slow pole ran away" was separately removed **without spending GPU** — `beta_s = sigmoid(beta_s_raw)` is bounded in (0, 1) and weight decay pulls it toward *more* damping. (g) **§8: no row edited; row #11 added, open** — may the frozen Phase-2 recipe be re-derived at a new size, and how? The recipe was fixed at 735K and has never been re-derived; this blocks Phase 5 as conceived. **No term is changed and no value is proposed**, and the price of the cheapest discriminating measurement is recorded (~11 GPU-minutes per LR probe, because the arm dies at 5138 rather than at 20,000) so the decision is made against a price. **Table now eleven rows, four resolved, seven open.** #4 and #9 are re-evidenced in prose only. (h) **§7.1 gains a rev-9 note, offered and not applied**: three of the remaining rows sit on the two-compartment family or on components width already moved, and a prerequisite has appeared underneath them; the measurement that reorders the table is no longer an arm but reach; and the GPU-h column gains a second measured row, with the calibration method transferring across *arms* (K3 0.926 / 0.915 / 0.972). **~0.7 GPU-hours bought no number** and is recorded as spent rather than netted off. (i) **CORRECTION to rev 8.** §13.6 said width closed *"about a quarter of the remaining gap to the anchor"*. The figure against the 735K anchor is **52.0 %** — the error was reading the absolute 0.25238 bpc gain as a fraction — and the corrected figure is *still the wrong comparison*, because it prices a 5M model against a 738K anchor. §14.2's matched figure, **6.6 %**, supersedes both. The rev-8 sentence is left standing with the correction beneath it. (j) §1: experiments closed gains `EXP_015`; **GPU-hours ~8.0 → ~10.1**; **unexplained divergences 2 → 4, sharing 2 causes**; decisions open **6 of 10 → 7 of 11**; the adopted-arms row gains the caveat that neither adopted arm has been shown to train above 735K. (k) §9 re-run: `ruff` clean, fast subset **267 passed / 1 skipped** (258 → 267 for `tests/test_calibrate_cost.py`), full suite green; **gate rows not re-run**, since no kernel, captured graph or numeric path changed. (l) `EXP_015` §9.10 records **three defects the experiment found in its own instruments** — the driver marked both dead legs `completed: true` (a trainer that goes non-finite exits 0 and writes a checkpoint), the same defect one level down where an absent log read as a clean log, and the resolver crashing on P4 by guessing an artifact's shape. All three fixed with tests; **no threshold, bar or verdict moved**, and the committed manifest still reads `completed: true` because that is what the driver did. (m) `EXP_015` §9.12 records that the pre-registration hash recipe is **CRLF-dependent** — reconstructing with LF gives a different digest, so a future reader checking that way would wrongly conclude the pre-registration had been edited. **No earlier number, verdict or threshold changes beyond the correction at (i). No arm is adopted, no size is chosen, no tolerance is set, no phase is authorised, and no recipe term is changed.** |
| 8 | 2026-08-10. **`EXP_014` folds in as §13 — the largest effect this phase has measured, and it is not an arm — and decision #7 turns out not to be numerically inert.** (a) **§13 added**, pushing the changelog to §14: the parameter-scaling ladder at `d ∈ {512, 1020, 1481}` (735,437 / 2,501,245 / 4,997,099 params), seed 0, `arch="snn"`, `d_model` the only field changed. **−0.25238 bpc carried at 54.75 transferred σ** against a pre-registered 10 σ bar, and **−0.25674** against the alternative anchor the log pre-registered for a G1 failure, so the verdict does not turn on the anchor. S1–S4 all held. **94–98 % of the gain is at or inside the baseline's own reach** (horizon 7 → 8 → 8, inside the pre-registered ceiling); the within-reach component moves **+0.12538** against the best architectural arm's +0.0288. **The underfitting regime ends between 2.5M and 5M parameters** — fresh gap +0.00592 / +0.03524 / +0.05509 — and the `Δtest ≤ 0` guard (decision #9's substance) did real work for the first time, distinguishing memorisation-alongside-a-real-gain from `EXP_013` N1's damaged model. **Unanticipated and unpredicted by any bar: firing rates fall monotonically with width** (0.339/0.320 → 0.293/0.267 → 0.209/0.185), which bears on the energy argument. An unplanned check worth more than the bar it served: the `d=512` leg's fresh gap is +0.00592 against `EXP_013` §2.2's +0.00594, five different runs on a different tree agreeing to 2e-05 on a quantity whose 2σ bar is 0.00536. **Nothing is adopted and nothing is ranked** — n = 1 per rung against §6.2's three-seed rule, carried as structural fields in the artifact rather than as prose. (b) **§13.5: G1 — the one gate expected to pass — FAILED, and was chased to its origin the same day.** Today's tree does not reproduce the committed baseline (+1.973e-03 fresh / +1.994e-03 carried). Five measurements identify **decision #7's own `_clip_grad_norm_fp64`** as the whole cause: the tree reproduces *itself* bitwise; the clip fires exactly three times in 20,000 steps (32/33/34, norms 1.0329/1.1198/1.0957, all between the 250-step logging cadence); a worktree at `b5f71a9` reproduces the committed baseline bitwise; monkeypatching the stock clip back gives zero differing losses over 250 steps; the fp64 clip first differs at step 47. **The fix is therefore not numerically inert** — true of the gradients, false of the trajectory — and `src/snn/train.py`'s docstring said in as many words that it was. **That docstring is corrected**; `tests/test_grad_clip.py` needed no change, and §8's rev-8 note records why (its bitwise assertion covers only the `max_norm=inf` sentinel path; the test that exercises a firing clip already asserts `allclose`, not equality). The record also keeps the **false negative**: the substitution test was first run for 40 steps, read as exonerating the clip, and stopped seven steps short of the divergence — a *validated* probe run over too short a window, `EXP_007` V4's shape. (c) **§8: no row edited; row #10 added, open** — what follows from #7 changing every trajectory it clips. **Elliot has ruled the re-baselining half: the project does not re-baseline**, and every new experiment trains its own anchor on the current tree. The standing-rule half — whether a `src/snn/` change altering a committed figure must be recorded as such before merge — **stays open**, and `CONTRIBUTING.md` is therefore **not** amended at this revision. **Row #7 is not edited**: it is decided, and this is a correction to its evidence, which the rev-5 rule puts in prose beneath the table. **Table now ten rows, four resolved (#1, #5, #7, #8) and six open (#2, #3, #4, #6, #9, #10).** (d) **§7.1: rank 2 struck through as DONE**, on rank 1's precedent, with its cost corrected **0.35 → ~1.5 actual**. A rev-8 note offers three consequences and applies none: the remaining rows are ranked against 0.03–0.17 bpc gains while width bought 0.25; the sizing answer §7.2 said rank 2 would supply **cannot** come from it, because it ran on the plain LIF and `EXP_005`/`EXP_012` are two demonstrations that this project's statistics do not transfer between its two neurons; and the GPU-h column is now suspect in a *measurable* way rather than a general one — the one row ever measured was low by **3.1×**, and ranks 3–6 sum to 2.70 as written against ~8.4 if they carry the same error. **No rank is renumbered and no row's evidence is rewritten.** (e) §1's status table updated: experiments closed gains `EXP_014`; **ranked candidates closed 5 → 6 of 14** (`EXP_014` closes #11, and that is the only thing it settles); **GPU-hours ~6.5 → ~8.0**, itemised — 1.02 ladder training, ~0.17 calibration run twice, ~0.09 evaluations and probes, ~0.25 for the chase; decisions open **5 of 9 → 6 of 10**. (f) the changelog renumbered §13 → §14 as `EXP_014`'s new §13 was inserted — the same slip rev 3 (f), rev 4 (g) and rev 6 (h) each record for their own insertions — and the three bare "§13" cross-references that meant the changelog (the header's rev-7 blockquote, §1's rev-6 summary of what rev 6 did, §6.1's wall-clock correction note) were retargeted in this same pass rather than left for a rev 9 to find. (g) §9 re-run for real: `ruff` clean, fast subset **258 passed / 1 skipped** in 8.2 s, full suite **396 passed** in 38.7 s — 394 → 396 because `EXP_014` added two ladder tests. `CONTRIBUTING.md` and `README.md`'s quoted counts corrected **257 → 259** (fast, which counts the skip) and **394 → 396** (full); the **gate rows themselves are not re-run and not claimed to be**, since rev 8's only code change is a docstring. (h) `README.md`'s results section gains the width result and the caveats it must be read under, and its *"the model underfits"* line — true at 735K and now bounded above — is qualified rather than deleted. **No earlier number, verdict or threshold changes. No arm is adopted, no tolerance is set, no phase is authorised, and no size is adopted.** |
| 7 | 2026-08-08. **Bookkeeping only: the tree is committed, three figures are corrected, and one ranked cost is flagged as unsupported.** No experiment ran, no arm is adopted, no tolerance is set, no phase is authorised, and **no §8 row is edited — five decisions remain open.** (a) **Rev 6's uncommitted working tree is now committed**, in seven commits on `main`: `EXP_013`'s pre-registration and arm, decision #7's fp64 clip (deliberately split from the first so a future bisect can separate a change to *every* arm's training numerics from one experiment's wiring), `EXP_013`'s drivers/resolver/evidence, rev 6 itself, chat session 7, chat Track A, and an orphan sampler. `main` was 74 commits behind at Phase 1 and is fast-forwarded to the tip; a `git bundle` of every ref is written outside the repository. **`EXP_013`'s pre-registration was verified before and after committing**: §0–§8 hash to `285eb661…82ef8d` and the reconstructed pre-results file to `2e9ee41e…`, matching both stamps in `exp_013_run_manifest.json`. That does **not** retroactively supply the pre-run commit its own §0 promised and this entry does not pretend otherwise — the hash chain is what discharges the guarantee, and it held. Full suite re-run on the committed tree: **394 passed, 44.9 s**; fast subset 256 passed / 1 skipped; `ruff` clean. (b) **Rev 6's changelog inventory of what remained uncommitted was incomplete** and is corrected here rather than by rewriting that entry: it lists `EXP_013`'s log, four `exp_013_*.json`, `src/snn/noise.py`, `tests/test_noise.py` and `scripts/exp/013_*.py`, and **omits `src/snn/model.py`, `src/snn/config.py` and `scripts/exp/001_memory_horizon.py`** — two of them in `src/`, so a reader planning the commit from that list would have shipped the arm without the class that implements it. It also undercounts the `exp_013_*.json` artifacts, which are six. (c) **§7.2's budget sentence corrected**: ranks 2–6 sum to **3.05**, not 3.2, and ~**23.5** GPU-hours remain, not ~26.5 — the latter a rev-2 figure carried through four revisions against §1's own ~6.5-of-~30. (d) **§7.2 gains a standing caveat on rank 2's 0.35 GPU-h**: it descends from `01_reconnaissance.md` §3.5's eager forward-only width sweep, which `02_baseline_report.md` §6.2 withdrew in a section that closes by instructing Phase 3's ROI estimates not to reuse it, and `EXP_003`'s ladder is collinear in `K`, `K·d` and `K·d²` so it cannot price width either. A GEMM-FLOP model on `audit_01`'s measured 11.91 TFLOP/s puts it at ~0.9–1.4 GPU-h. **No corrected number is written into §7.1's table**, because that model is itself unvalidated at width; `EXP_014` opens with a calibration that measures it. (e) `README.md`'s status table corrected — it still read 1 adopted arm / 6 of 8 open / ~4.6 GPU-hours, three revisions stale, because rev 6 (g) corrected only the test-suite figures in that file. |
| 6 | 2026-08-08. **`EXP_013` folds in, #5 and #7 are decided, and #9 is added.** (a) §12 added: `EXP_013` (noise injection) closes as a null — costs bits at every amplitude (+0.00419/+0.04978/+0.63975 bpc, 1.2/14.8/190.0 se), buys none — but its §2.2 calibration is the transferable result: this 735,437-parameter model's generalisation gap is +0.00594 bpc (fresh) / −0.00180 (carried, indistinguishable from zero) against ~655M training characters, ~7.3 epochs. **The model is not overfitting.** N1 (the primary rule) holds at two amplitudes and must not be believed — the decomposition shows a uniformly damaged model, not better generalisation, the same defect as `EXP_012`'s Y5 (§12.4). (b) §6.3, §6.5, §6.7 item 3, §10.1 and §7 item 5 each gain a rev-6 cross-reference to §12, none rewritten. (c) §7.1 gains a new rank 2, **the parameter-scaling pilot** (`03_phase3_candidates.md` §6.3 #11, 0.35 GPU-h) — absent from this table through rev 5 for lack of evidence there was room to grow; `EXP_013` supplies exactly that evidence. §7 gains item 6 stating the referral. Ranks 2–5 renumbered 3–6; no prior rank's own evidence changed. (d) §8: **row 5 edited to RESOLVED** — the learned per-channel threshold is **adopted standalone**, per Elliot's instruction; the composed arm (`EXP_011`) is explicitly **not** adopted by this and stays provisional at n = 2. **Row 7 edited to RESOLVED** — `src/snn/train.py`'s `_clip_grad_norm_fp64` replaces `clip_grad_norm_`, accumulating the sum of squares in fp64; verified on `EXP_009` §9.4's own failing gradient (`tests/test_grad_clip.py`) and against the unchanged R5 capture gate (10/10). Does **not** address `compose_s0`'s divergence — a different, still-open mechanism (§10.4). **Row 9 added, open**: whether a pre-registered bar needs a guard-clause review before commit, per `EXP_012`'s Y5 and `EXP_013`'s N1 firing the identical defect twice — the *protocol* rule is already ratified in `CONTRIBUTING.md` §2 (`51b44c6`, 2026-08-05, Elliot); this row is the report catching up to it, not a new decision. **Table now nine rows, four resolved, five open.** (e) `CONTRIBUTING.md` §4 gains two ratified amendments: the fold-in tolerance is architecture-specific and re-derived per arm, never inherited (decision #6's substance); a provisional result is promoted only by seeds *added*, never substituted, up to the pre-registered n. (f) §9: full suite re-run for real, **394 passed** (280 + 85 `test_snnchat.py`, uncommitted-to-this-revision + 23 `test_noise.py` + 6 `test_grad_clip.py`); fast subset 256 passed/1 skipped/8s. Gate rows added for `EXP_013`'s G1/G3/G4-5/F1/K1/calibration/hash checks and for decision #7's fix. Mutation campaign **not** re-run — nothing in `snn/kernels.py`/`surrogate.py`/`twocomp.py` changed. (g) `README.md` and `CONTRIBUTING.md`'s quoted test-suite figures corrected to the numbers this revision measured, both stale independently of each other (README's fast count and CONTRIBUTING's full count had drifted by different amounts). (h) the changelog renumbered §12 → §13 as `EXP_013`'s new §12 was inserted, the same slip rev 3's item (f) and rev 4's item (g) each record for their own insertions; two bare "§12" cross-references (§1's rev-2 summary, §6.1's wall-clock correction note) meant the changelog and were retargeted to §13, caught in this same pass rather than left for a rev 7 to find. **GPU-hours: ~4.6 → ~6.5** (using `EXP_013` §9's measured 1.9, not its §0 estimate of 1.6). **Working tree, recorded rather than resolved:** `EXP_013`'s own files (log, four `docs/reports/data/exp_013_*.json` beyond the two already read at rev 4, `src/snn/noise.py`, `tests/test_noise.py`, `scripts/exp/013_*.py`) remain uncommitted, as they were at rev 5 and at `31e49fa`'s deliberate exclusion; this revision's own edits (this file, `CONTRIBUTING.md`, `README.md`, `src/snn/train.py`, `tests/test_grad_clip.py`) are likewise left uncommitted — staging is Elliot's, not this revision's. |
| 5 | 2026-08-05. **Decision #1 is decided, and it is the only thing in this revision.** Elliot signed off Phase-3 §9. (a) `03_phase3_candidates.md` §9's last box is ticked and **labelled retrospective in place** — Phase 4 opened without it and eight experiments ran before it, so the box's "Phase 4 does not begin until" was not honoured in sequence and the tick says so rather than letting the record read as though it had been; that file gains a rev 3 carrying the provenance. (b) §8 **row 1 edited to RESOLVED** — the first of rows 1–7 ever edited, following #8's rev-3 precedent that a *decided* row is edited while an *open* row is only annotated in prose. (c) A rev-5 note beneath §8 states the one consequence and two non-consequences: it **releases `EXP_008` §4's rider so #5 is ungated by #1**, it does **not** decide #5, and it does **not** launder **#2** — a defect `EXP_005` §9.4 found under a §9 box that is now ticked, named in both files rather than absorbed. (d) §8's rev-3 preamble is left as written with a rev-5 parenthetical, per the §10.5 precedent. (e) §9 gains a closing paragraph stating that the gate table was **not** re-run and is not claimed to have been, because rev 5 touches two report files and no code, artifact or run directory. **Six decisions remain open. No experiment ran, no number moved, no arm is adopted, no tolerance is set, and no phase is authorised.** |
| 4 | 2026-08-05. **`EXP_012`: the fold gap explained, and a rule that misfired reported rather than repaired.** (a) §11 added: the ~1000× gap between `EXP_008` W1 and `EXP_011` C4 is **not** the algebra (leg C exact on both neurons) and **not** the injected perturbation (leg A 6.4–8.2 fp32 eps against 5.8–8.3), but the **layer-0 flip count** — 1/2 flips against 2 684–5 339, a factor of **1 342–5 339** — which tracks the probability mass the decision variable puts at the threshold: **1 843–6 218× more** within 1e-6 of it, at a membrane spread that differs by only 1.19×. (b) §11.6, **post-hoc and labelled so**: the baseline's ~11 000 near-threshold sites are **20–23 distinct values replayed 400–548 times each**, because layer 0 sees one of 205 embedding rows and the hard reset zeroes `v` exactly; the unreset slow pole destroys the replay (4 values, 1.2 repeats). Post-spike enrichment does **not** discriminate (2.61× vs 2.11×) and is reported as a near-miss for the analysis. (c) **Three of six predictions failed.** Y5's rule resolved on a leg comparing **0 flips against 1**, which `EXP_012`'s own limitation 2 had declared unresolvable before it ran; the rule is **recorded as it fired and not rewritten**, and the powered leg's 102.5× is a marker with no verdict. Y4's 34× is 34/1 and 68/2 and is not offered as evidence either way. Y6 failed, and its failure pattern — accurate wherever the density is smooth, 3.5–8.8× over wherever there is an atom — is consistent with (b). (d) §6.5 **stands exactly as written**; the scope qualifier its own evidence supports is **referred, not applied**, because the trigger `EXP_012` §4 pre-registered for it (`Y2 ∧ Y5`) did not fire. (e) §8: **no row edited**; #6's evidence is sharpened in prose beneath the table — §10.5's withheld number is **released** to it, and the finding is that a fold-in tolerance **cannot be one project constant**. #4 re-evidenced at ~4.6 of ~30 GPU-hours. (f) §9 adds three gate rows; test suite **272 → 280**. (g) the changelog renumbered §11 → §12 and the header's rev-2 reference retargeted — the same slip rev 3's item (f) records. **No earlier number, verdict or threshold changes; nothing is adopted and no phase is authorised.** |
| 1 | 2026-08-03. First Phase-4 report: the adopted arm and the shape of its gain, σ's non-transfer, `EXP_006`'s causal tail result and what it withdraws, `EXP_007`/`EXP_008`/`EXP_009` resolved against their pre-registrations, the reparameterisation noise floor, a revised ranking offered and not applied, and seven open decisions. |
| 3 | 2026-08-04. **`EXP_011`, and the resolution of the one decision that was a scheduling question.** (a) §10 added: the composed arm (adopted two-compartment neuron + `EXP_008`'s learned per-channel threshold) at **2.08326 ± 0.00049 carried, −0.03543 against the adopted arm at −28.7 se, 0/128 contexts regressed, folding to the adopted arm's exact inference cost** — and **provisional at n = 2 of a pre-registered 3**. (b) §10.3: the composition's gain is **102.4 % within-reach and 1.4 % zero-context**, which confirms `EXP_004` §10.6's overlap argument in its strong form and relocates the value to the component the adopted arm damaged — the second time in Phase 4 that an arm has not moved the component it was ranked for. (c) §10.4: `compose_s0` diverged, was chased to step 17 598 and is **not** `EXP_009`'s mechanism — gradients of 0.03, fp32 and fp64 norms equal, an identical NaN on the eager path — so **decision #7's fix would not have prevented it** and the fused backward is exonerated. The seed was excluded and **not replaced**. (d) §8 row 8 marked **RESOLVED (inside Phase 4)**; **rows 1–7 are not edited**, and the two whose evidence `EXP_011` overtakes (#4, #7) are corrected in prose beneath the table rather than in the rows. (e) §9 re-run with four new gate rows; test suite **262 → 272**. (f) the header's §10 reference retargeted to §11 as the changelog renumbered. **No earlier number, verdict or threshold changes.** |
| 2 | 2026-08-04. **Synthesis and corrections. No verdict, threshold or decision changes.** (a) §6.7 added: the three arms read together — two of them are the same mechanism, only the arm that added the least structure moved the within-reach component, token-shift is dominated at 13.3 se, and essentially none of the three gains is attributable to a larger function class. (b) §6.8 added: the verdict on every Phase-4 arm against its own rule, including that `EXP_008`'s cell is `W1 fails` and its headline is quoted under that caveat. (c) §7.2 added: the order Phase 5's arms should run in and what must be settled first. (d) §8 item 8 added; #3, #4, #5 re-evidenced; **all eight remain open**. (e) §9 re-run from a clean tree, with the resolver's 0-difference reproduction as a new row. **Corrections:** (f) §6.1's wall-clock for the adopted arm was **1.35×**, which is `twocomp_s0` alone — `EXP_004`'s committed three-seed figure is **1.32×** (1.30× over all seven), and `EXP_004` §10 item 6 said to quote it; *this correction flatters the adopted arm and is made because it restores a committed measurement*. (g) §6.1's distilled row was **1.58×**, correct against the two-compartment arm in `EXP_009` §9.1 and wrong in a column referenced to the baseline, where it is **2.13×**. (h) "clear the adoption bar by an order of magnitude" → **4.2×** and **6.4×** (8.4σ and 12.8σ). (i) the adopted arm recovered **58.2 %** of the beyond-horizon component, not 59 %, in §2 and §7.1. (j) Identity 2 holds to **6.8–8.3** fp64 eps; `EXP_008` §9.5's "≈7" is its first two seeds. Closed experiment logs are **not** rewritten for (f)–(j); the corrected numbers live here. (k) adding §7.2 made three bare "§7.2"/"§7.3" references read as self-references when they meant `03_phase3_candidates.md`; all three are now qualified — the same slip the Phase-3 report's own changelog records as (d). |
