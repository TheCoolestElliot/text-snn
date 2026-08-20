# EXP_020 — What does this model pay to read a character in, and to write one out?

**Pre-registered:** 2026-08-18, after the algebra in §2, the frozen-feature
probe in §2.6, the §7.2 reachability screen in §7 G8 and the cost pre-flight in
§3.5, and **before any arm was trained, any checkpoint was written, and any bpc
existed for any arm in §3.** **Committed before the first run of §3.**

**Status: CLOSED 2026-08-18. T1 held; H1 and H3 failed; H2 and H4 UNRESOLVED. Every arm lost.**

---

## 0. What this experiment is, and what it is not

Every arm this project has built changes the middle of the model — the neuron,
its reset, its decay, the current it integrates. **Nothing has ever changed its
two ends.** Layer 0's input has been `nn.Embedding` since Phase 2 and the head
has been one `nn.Linear` on the last layer's spikes at `t`, and both were
inherited from `02a_phase2_spec.md` §4 rather than measured.

Those two ends hold **28.6 %** of the model's parameters at the committed shape
(`embed` 104,960 + `head` 105,165 of 735,437), and §2.1 shows that a third
block, `layers.0.weight`, is provably redundant with the first of them.

**What this is not:**

* **It adopts nothing and ranks nothing.** Adoption is Elliot's (report §8).
* **It changes no hyperparameter.** The Phase-2 recipe — `lr = 3e-3`, cosine
  over 20,000 steps, `grad_clip = 1.0`, `weight_decay = 0.1`, `beta = 0.5`,
  `threshold = 1.0`, hard reset, `atan` at `alpha = 2` — is frozen, and G5 reads
  all of it back out of each run's own `config.json`.
* **It does not decide #11.** Every arm here is at 735K. Nothing is run at 5.0M,
  so nothing here says whether the recipe may be re-derived at a new size.
* **It does not edit any adopted arm.** `src/snn/neuron.py`, `kernels.py`,
  `surrogate.py`, `twocomp.py` and `twocomp_detach.py` are untouched; G1 asserts
  it by `git diff`. This experiment adds **no new scan, no new recurrence and no
  hand-written backward**, so the R10 gate and the 59-mutation campaign are not
  re-run and the gate table says "untouched" rather than implying otherwise.
* **It violates no invariant.** I1 binary spikes between layers — *strengthened*
  by leg 4, which removes the one exception the committed model has. I2, I3, I4
  untouched. I5: the only `[d, d]` parameters are still `layers.k.weight`,
  each applied to the previous layer's output before a scan begins.
* **It does not re-baseline.** Decision #10: this project does not re-baseline,
  so leg 1 trains a **fresh anchor on today's tree** and every number in §9 is
  internal to this experiment. No figure here may be compared with a committed
  one.
* **It measures a cost as well as a benefit, and the cost leg is not optional.**
  §3.5 prices every arm in wall-clock before it runs and §4 H5 reports the
  realised ratio whatever it is.

---

## 1. The question

> **The input current is a lookup, the readout is a linear map on one binary
> vector, and neither has ever been measured. What does each cost?**

Two sub-questions, deliberately separated into legs rather than composed:

1. `layers.0.weight` spans no function the embedding does not. What does
   carrying it anyway buy, and what does spending it on width buy instead?
2. The head reads the last layer at `t` and discards every other layer's
   emission at `t`. What is discarded worth?

---

## 2. The derivation, done before any measurement

### 2.1 The fold is an identity, and it is 35.7 % of the model

`_CharLMStack.forward` computes `h = E[x]` then `cur0 = h W0^T + b0`. `E[x]` is
a **lookup**: it is one of `V` rows, never a mixture. A linear map of a lookup is
a lookup of the mapped table, so with `T = E W0^T`

    cur0[b, l] = T[x[b,l]] + b0                                      EXACTLY

for every `E`, every `W0` and every input. Conversely any table `T` is `E W0^T`
with `E = T`, `W0 = I`. **The function classes are equal.**

At `V = 205`, `d = 512`, `K = 2`, `layers.0.weight` is **262,144 of 735,437
parameters — 35.7 %** — and it is the only parameter block in the stack provably
redundant with another. Measured on `snn_beta0.5_s0`, over 4×32 real tokens:
max abs difference **1.907e-06**, max relative **5.186e-08**, 62.8 % of elements
bitwise equal. That is the rounding of a 512-term dot product and nothing else.

**This is `EXP_008`'s Identity 2, one layer earlier, and its precedent is a
warning rather than an encouragement.** `EXP_008` proved a per-channel threshold
folds into `layers.k.weight`, and then measured the redundantly-parameterised
form training **better by 0.0590 bpc at 23.6 se**. A reparameterisation that
provably cannot change what a model expresses can still change what the
optimiser finds, and on this architecture it has, once, in the direction that
favours the redundant form. So the fold is not asserted to be free. Leg 2 prices
it at unchanged width; leg 3 is the arm.

### 2.2 What the fold frees, and the width it buys

With `layers.0` gone the closed form at `K = 2` is `d^2 + 412d + 205` against
`2d^2 + 412d + 205`. Holding the total at the anchor's 735,437 gives

| arm | d | parameters | of anchor |
|---|---:|---:|---:|
| anchor, binin | 512 | 735,437 | 100.00 % |
| fold | 512 | 473,293 | 64.36 % |
| foldwide | 675 | 733,930 | 99.80 % |
| mlayer | 471 | 734,494 | 99.87 % |

**1.32x the width at the same parameter count.** `EXP_014` is the standing
reason to care: width bought −0.25238 bpc at 54.75 transferred sigma, more than
every architectural arm this project has built, and what it bought was
**per-step capacity and essentially no reach** (horizon 7 → 8). §4 therefore
puts H2's bar on the mean and predicts explicitly that the horizon does not move.

### 2.3 Binary input coding, and why the rate-coding literature does not apply

I1 says binary spikes carry all information *between* layers. The committed
model honours it everywhere except its first layer, which receives a real-valued
embedding row — `model.py:349` calls it "analogue direct coding, T=1". Leg 4
closes that exception: the code is passed through the project's own `atan_spike`,
so layer 0 receives a `{0,1}` pattern through `layers.0`'s synapses exactly as
layer 1 receives layer 0's. **Every inter-unit signal in the model is then
binary**, and the only real numbers left are membranes, weights and logits.

`01a_literature_review.md` §3.1 records that direct coding beats **Poisson rate
coding**, and that finding does not transfer, for a reason of shape rather than
of degree. Rate coding spends `T` timesteps per character encoding one analogue
value as a spike count — which is why DIET-SNN needs 150 of them — and its cost
is `T`x the kernels. This spends **one** timestep, `T = 1`, and represents the
character as a *spatial* pattern across `d` fibres. **The axis being binarised is
the channel axis, not the time axis.** The cost rate coding pays is exactly the
cost this does not.

Leg 4 is also the only leg that is **exactly parameter-identical** to the
anchor: binarising a code changes its alphabet, not its shape.

### 2.4 The inits are derived, and every one lands on the same number

`02_baseline_report.md` §5.2 measured this architecture's one initialisation
pathology, and it is a statement about exactly this interface: layer 0 sees
dense `N(0,1)`, so `sd(cur) = sqrt(E[x^2]/3) = 0.5774` and it fires at 4.9 %;
layer 1 sees a spike train where `E[s^2] = p`, not 1, so its threshold is
**7.8 sigma** away and **it never fires at init**. Anything that changes what
layer 0 is handed must state what it does to that arithmetic in advance, or it
is changing two things at once.

Every init below is chosen to land on `sd(cur) = 0.5774` and on nothing else:

| arm | derivation | measured init rate, layer 0 |
|---|---|---:|
| anchor | `E ~ N(0,1)`, `Var(cur) = E[x^2]/3 = 1/3` | 0.0490 |
| fold, foldwide | `T ~ N(0, 1/3)`, because `Var(E W0^T) = d * 1 * 1/(3d) = 1/3` exactly | 0.0481 |
| binin | `cur = W0 g(B - q)`, `E[x^2] = g^2 q(1-q) = 1` ⇒ `g = 1/sqrt(q(1-q)) = 2` | 0.0493 |

The code is **centred**, and that is not cosmetic. An uncentred `g B` has the
right marginal second moment and the wrong structure: with `S_c = sum_j W0[c,j]`,
`cur_c` carries a fixed per-channel offset `g q S_c` of sd 0.408, so half the
channels are biased toward firing at every character before a single token is
read. Measured: the uncentred code fires layer 0 at **0.082** against the
baseline's 0.0490, a 1.7x mismatch that would have travelled through the whole
arm as an unstated hyperparameter. `snn.noise` centres its Bernoulli draw by the
identical construction, and its `1/sqrt(p(1-p))` is the same factor.

**None of these three constants was measured first and none is swept.**

### 2.5 The readout, and the one thing it is not allowed to be

The head reads the last layer's spikes at `t`. Two things are discarded and both
are already in memory: every other layer's emission at `t`, and the same layer's
emission at `t-1, t-2, ...`. Concatenating them costs **zero** kernels inside the
time loop and one `cat` outside it, and — the constraint that shapes the whole
design — **both keep the head's input strictly binary.**

Reading the **membrane** instead would put a real-valued signal on the
inter-unit path, which is what I1 exists to forbid. It is measured in §2.6 as a
reference and **is not an arm.**

### 2.6 Leg A, already run, and what it authorises — POST-HOC and labelled so

`scripts/exp/020_readout_probe.py`, committed and run 2026-08-18 **before this
file existed**. It fits a fresh ridge-selected multinomial logistic head on
**frozen** features from committed checkpoints, 1024 val windows, 60/20/20 split
**by window**, ridge chosen on dev and reported on test. Two seeds:

| feature set | `snn_beta0.5_s0` | `s1` | vs R0 |
|---|---:|---:|---:|
| R0 last layer at `t` (the committed readout) | 2.13268 | 2.14477 | — |
| R1 last layer at `t`, `t-1` | 2.13299 | 2.14230 | **+0.0003 / −0.0025** |
| R2 **all layers at `t`** | 2.11414 | 2.12280 | **−0.0185 / −0.0220** |
| R5 membrane at `t` (analogue reference, NOT an arm) | 2.51384 | 2.52326 | **+0.381 / +0.378** |

**The probe's asymmetry runs one way only and it is the reason to run it first.**
The representation was optimised end to end *for R0's readout*, so every other
feature set is handicapped: its features were never shaped to be read that way.
A gain measured here is therefore a **LOWER BOUND** on what joint training would
give, and a null here **does not bound the joint-training gain from above.** It
can authorise an arm cheaply; it cannot veto one.

Three things follow, and all three are acted on in §3 rather than admired:

1. **R2 authorises leg 5.** Both seeds clear 2 sigma on frozen features alone.
2. **R1 does not authorise a tap ladder, and the tap ladder is therefore NOT
   RUN.** This is the probe paying for itself: three runs not spent.
3. **R5 prices the binarity of the readout at −0.38 bpc in the model's favour.**
   A linear head reads the *thresholded decision* far better than the membrane
   it was computed from. That is a marker on one checkpoint family and not a
   general claim, but it is the opposite sign from the intuition that binarity
   is a concession.

**T1 FAILED and is reported as it fired.** The refit R0 beats the committed head
by 0.031 and 0.028 bpc against a 0.02 tolerance. The tolerance was wrong, not
the probe: the refit is fitted on held-in val windows and the committed head was
fitted on the train split, so the refit is expected to win and the bar should
never have been two-sided at 0.02. **Reported unrepaired**; no delta above is
adjusted, and the deltas are between feature sets fitted identically, which is
the comparison T1 does not bear on.

---

## 3. The legs

All at `V = 205`, `K = 2`, enwik8, 20,000 steps, seeds 0/1/2, one variable per
leg against a named reference run, driven by
`scripts/exp/020_run_interface_arms.py`.

### Leg 1 — the anchor (3 runs, ~20 GPU-min)
`arch="snn"`, `d = 512`. Fresh, on today's tree, per decision #10.

### Leg 1b — the nesting gate (1 run, ~7 GPU-min)
`arch="interface"` at every default. `tests/test_interface.py` N1 asserts this is
`arch="snn"` **bitwise**, forward and backward, so this run must reproduce
`if_anchor_s0` at **exactly 0 difference** in final bpc. One seed, because one
seed either reproduces it or does not. **A gate, not an arm.**

### Leg 2 — the fold at unchanged width (3 runs, ~16 GPU-min)
`iface_fold`, `d = 512`, 473,293 params. Prices the factorisation. Deliberately
**not** parameter-matched — matching it would confound the two things this
experiment separates.

### Leg 3 — the fold, reinvested (3 runs, ~24 GPU-min)  **the candidate**
`iface_fold`, `d = 675`, 733,930 params.

### Leg 4 — the binary input code (3 runs, ~20 GPU-min)
`iface_binary_input`, `d = 512`, **735,437 params — identical to the anchor.**

### Leg 5 — the multi-layer binary readout (3 runs, ~24 GPU-min)
`iface_read_layers="all"`, `d = 471`, 734,494 params.

### Leg 6 — the tap ladder. **NOT RUN.** §2.6 R1 did not authorise it.

### 3.5 Cost, measured before this file was committed

Marginal ms/step from a 400-vs-1600-step difference, which removes the ~10 s of
fixed start-up that made a naive 400-step extrapolation over-predict the anchor
by 2.3x:

| arm | ms/step | 20k steps | x anchor |
|---|---:|---:|---:|
| anchor | 18.38 | 368 s | 1.00 |
| fold | 14.28 | 286 s | 0.78 |
| foldwide | 22.30 | 446 s | 1.21 |
| binin | 18.55 | 371 s | 1.01 |
| mlayer | 21.92 | 438 s | 1.19 |

The anchor's 368 s against its committed 398.9 s is the **8 % under-prediction
`EXP_014` §9 measured for this method** (realised/projected 0.925–0.972), and it
is not corrected for — it is the reason the budget below carries 1.09x.

**Budgeted: ~2.1 GPU-hours, x1.0.** 16 training runs at 6,644 s plus 16 test
evaluations. `§7.1`'s own audit says every GPU-h estimate in this project is
suspect and the one row ever checked was low by 3.1x; this one descends from a
measurement on the arms themselves rather than from a constant, which is the
remedy `EXP_014` §9 proposed and not a claim that it cannot be wrong.

---

## 4. Pre-registered predictions

### 4.0 What this design can and cannot resolve, stated before the run

**n = 3 per cell.** The sigma is **transferred**: 0.00461, the whole-split
baseline sd over 5 seeds (`04_phase4_interim.md` §236), measured on a **different
tree** than the one these runs use — decision #10 records that the fp64 clip
moves a run by +1.97e-03 bpc, which is itself 0.43 sigma. The transfer is named
here in the same sentence as every bar that uses it, per `EXP_014` §4.

The se of a difference of two 3-seed means at that sigma is
`0.00461 * sqrt(2/3) = 0.00376`, so the 2-sigma bar of 0.00922 is **2.45 se**.
This design can show an effect exceeds that bar. **It resolves a miss as
UNRESOLVED, never as FAILED**, and a remedy is seeds **added**, never one
substituted for a failure. Leg 1 re-measures sigma on this tree and §9 reports
whether the transferred value held; a transferred sigma that turns out 2x too
small makes every bar below 2x too lax, and that would be a limitation of this
experiment rather than a correction to apply retroactively.

**A diverged run resolves DIVERGED / NO RESULT, not UNRESOLVED** (the correction
`EXP_017` H3 applied prospectively).

---

### T1 — the instrument, before any arm is read  *(abort-on-fail)*
> `if_nest_s0`'s final test bpc equals `if_anchor_s0`'s at **exactly 0
> difference**, both protocols.

`arch="interface"` at its defaults is `arch="snn"` bitwise by construction, and
the trainer is deterministic, so a whole 20,000-step run either reproduces it
exactly or the nesting claim is false somewhere the unit test does not reach.

**Failure mode:** a switch that is not actually off by default; a parameter that
exists and is decayed but reaches nothing; an ordering difference in the readout
concat at one block.
**Guards:** (a) equality is on the committed `final_test.json`, not on a
re-evaluation; (b) both protocols, because a fresh-only match would miss a state
handling difference.

### T2 — the gates, before a single training step  *(abort-on-fail)*
> G1, G2 and G8 hold as written in §7 before leg 1 starts.

---

### H1 — what the redundant block was buying  *(difference, BOTH terms constrained; one-sided)*
> `mean bpc(fold, d=512) − mean bpc(anchor)` is **positive and below +0.0582**.

Positive because leg 2 has 35.7 % fewer parameters and this model is
underfitting (`EXP_013`: the generalisation gap is +0.00594 fresh, so it is not
overfitting and parameters are not free). The **+0.0582** is `EXP_014`'s scaling
slope — 0.0913 bpc per doubling, from 2.25311 → 2.00073 over 6.8x — evaluated at
0.6436x parameters.

**This bar's reference is indicative, not exact, and that is stated rather than
discovered:** `EXP_014` scaled *width*, which scales every block; leg 2 removes
*one specific block*. The two are not the same cut and the transfer is the
weakest in this file. A miss above +0.0582 says the redundant factorisation was
carrying more than a generic parameter of its size, which is `EXP_008`'s result
repeating; a hit says it was carrying less.

**Failure mode:** reading a negative difference as "the fold is free" when it
would mean the anchor is undertrained.
**Guards:** (a) both terms are 3-seed means from this ladder, never a committed
figure; (b) the sign is predicted separately from the magnitude, and a negative
difference resolves H1 FAILED rather than being reported as a win.

### H2 — the candidate  *(difference, BOTH terms constrained; one-sided)*
> `mean bpc(foldwide, d=675) < mean bpc(anchor) − 0.00922`  (2 transferred sigma)

**Failure mode:** the 0.20 % parameter shortfall flattering the anchor. It does
not: the shortfall is against the arm, so a hit is conservative.
**Guards:** (a) realised parameter counts reported by G2 and quoted with the
result; (b) the horizon is measured too — see H4 — so a gain cannot be
attributed to reach without evidence.

### H3 — the price of a binary input  *(equivalence; power stated, one-sided both ways)*
> `|mean bpc(binin) − mean bpc(anchor)| < 0.00922`, at **identical** parameters.

**This is the one bar in this file that can only be failed, never proved.** An
interval that contains zero at n = 3 is a statement about power, not about
equality, and §4.0's 2.45 se is stated so the reader can price it. The prior is
`02_baseline_report.md` §6.2's I1 result — the analogue control, which removes
binarity from *every* inter-layer signal, costs only −0.0117 at 2.5 sigma — so
removing the *last analogue signal* is predicted to be similarly cheap.

**Failure mode:** an UNRESOLVED read as "binarity is free".
**Guards:** (a) the verdict wording in §5 forbids that sentence; (b) the
realised code density is logged per run (`aux["code_density"]`), because a code
that collapsed to all-ones is a constant current and would nest the baseline for
the wrong reason.

### H4 — the multi-layer readout  *(difference, BOTH terms constrained; one-sided)*
> `mean bpc(mlayer) < mean bpc(anchor) − 0.00922`.

§2.6's frozen-feature probe measured −0.0185 and −0.0220 as a lower bound on two
seeds, so this is the best-supported bar in the file.

**Failure mode:** the 41-point width reduction (512 → 471) eating the gain, which
would make a miss uninformative about the readout.
**Guards:** (a) parameter counts matched to 0.13 % and reported; (b) if H4 misses,
§9 reports the width confound explicitly rather than concluding against the
readout.

### H5 — reach, not capacity  *(MARKER, no verdict)*
> The 2-sigma memory horizon of `foldwide` and `mlayer`, against the anchor's.

`EXP_014` found width buys per-step capacity and no reach (7 → 8). If either arm
moves the horizon, that is a different mechanism from the one predicted and is
recorded as such. **No bar**, because n = 3 at one width cannot resolve a
horizon difference of one character.

### H6 — cost  *(MARKER, no verdict)*
> Realised wall-clock ratio per arm against `if_anchor_s0`, and realised peak VRAM.

---

## 5. Decision rule, fixed now

| cell | verdict |
|---|---|
| T1 at exactly 0 | the ladder proceeds |
| T1 nonzero | **ABORT.** The nesting claim is false; no arm below is interpretable and none is reported as a result |
| H2 hit, H1 hit | "the redundant block was cheap, and the width it frees pays" — **referred to Elliot, adopted by nobody here** |
| H2 hit, H1 miss | "the width pays despite the factorisation being load-bearing" — the stronger result, and §9 says so |
| H2 miss, H1 hit | "the block is redundant and removing it is free, but the width does not pay at this size" — a null on the candidate, retained |
| H2 miss, H1 miss | "the factorisation is load-bearing and the fold is a regression" — `EXP_008`'s result repeating one layer earlier |
| H3 hit | "binarising the input costs less than 2 transferred sigma at identical parameters" — **NOT** "binarity is free" |
| H3 miss, arm worse | "completing I1 at the input costs `X` bpc", quoted with X |
| H3 miss, arm better | recorded, and **not** claimed as an improvement from binarity without a mechanism |
| H4 hit | "the emission the head discards is worth more than 2 sigma" — referred |
| H4 miss | reported with the 512 → 471 width confound named, and the frozen-feature lower bound restated |
| any leg diverges | **DIVERGED / NO RESULT** for that cell; n is reduced and the verdict marked provisional; **no seed is substituted** |

**No verdict in this table adopts anything, changes any hyperparameter, decides
#4 or #11, or authorises Phase 5.**

---

## 6. Known limitations, stated in advance

1. **n = 3, and the sigma is transferred from a different tree.** §4.0.
2. **Width matching is on the integer grid.** `foldwide` is 99.80 % and `mlayer`
   99.87 % of the anchor, not identical. Only `binin` is exact.
3. **`iface_thr_in` is not swept.** It is 0.0, which is code density 0.5, which
   is what the gain derivation in §2.4 assumes. A different density needs a
   different gain and would be a different arm.
4. **The fold's init matches a distribution, not a draw.** `T ~ N(0, 1/3)` is the
   distribution `E W0^T` has at init; it is not any particular product.
5. **Leg A cannot veto.** §2.6. Leg 6's non-run rests on a probe that is
   structurally unable to establish absence, and that is a choice made against a
   price, not a finding.
6. **One width, one depth, one corpus.** `K = 2`, `d ≈ 512`, enwik8. `EXP_015`
   is the standing reason that a 735K result may not survive at 5.0M, and
   nothing here is run there.
7. **The fold is not tested on the two-compartment neuron.** Every leg uses the
   plain LIF, so nothing here transfers to the adopted arm or to `snnchat`.
8. **H1's reference slope is borrowed from a different kind of cut.** §4 H1.
9. **This experiment cannot separate the fold's two effects at leg 3.** `foldwide`
   changes the parameterisation *and* the width. Leg 2 is what makes the
   decomposition possible at all, and it is one arm rather than a ladder.

---

## 7. Gates — aborting, not advisory

* **G1** — no adopted arm edited: `git diff` empty over `src/snn/neuron.py`,
  `kernels.py`, `surrogate.py`, `twocomp.py`, `twocomp_detach.py` against the
  commit preceding this experiment. **No new kernel, no new hand-written
  backward, so R10 is untouched and the 59-mutation campaign is not re-run.**
* **G2** — realised parameter counts equal the closed forms **exactly**
  (`interface_param_count`, `spiking_param_count`), not to a tolerance.
* **G3 / K1** — one field changed per arm against its named reference run,
  checked field by field in `config.json`; anything outside the declared
  `may_differ` set aborts the ladder.
* **G4** — leg 1b, T1's whole-run nesting reproduction at 0 difference.
* **G5** — the frozen recipe read back out of each run's own `config.json`.
* **G6** — VRAM alarm at 4.0 GiB against a measured 0.61 GiB baseline. Under
  WDDM a spill is a ~50x slowdown that never raises.
* **G7** — a diverged leg is recorded as diverged by reading `log.jsonl` for
  non-finite losses, **not** by trusting the exit code, and an **absent** log
  reads as diverged rather than as clean (`EXP_015` §9.10).
* **G8** — the §7.2 gradient-reachability screen, run **before** the first
  training step and already green: `iface_fold` **PASS**, `iface_binary`
  **PASS**, with all three synthetic controls holding (`ctl_live` PASS,
  `ctl_dead` SADDLE, `ctl_faint` FAINT) so the screen is demonstrably able to
  fail. Artifact: `docs/reports/data/exp_020_021_reachability.json`.
* **G9** — the mutation-campaign lockfile is absent before every run.

---

## 8. Entry conditions

* This file is committed **before the first run**, and its SHA-256 is stamped
  into `exp_020_run_manifest.json` before the first run and re-checked after the
  last. A mismatch **voids the experiment** and the driver raises rather than
  writing a manifest that would look complete.
* `docs/reports/data/exp_020_readout_probe.json` exists — leg A authorises leg 5
  and de-authorises leg 6, and the ladder does not start without it.
* `docs/reports/data/exp_020_021_reachability.json` exists and is green (G8).
* `tests/test_interface.py` green, `ruff` clean, before the first training step.
* The resolver `scripts/exp/020_interface_results.py` is committed **before any
  leg's bpc is read**.
* GPU idle; nothing else runs while any leg is timed.

---

## 9. Results

*(appended after the run — nothing above this line is rewritten)*

**Closed 2026-08-18, ~1.9 GPU-hours** (16 training runs at 6,658 s, plus the
leg-A probe, the §7.2 screen, the cost pre-flight and a 16-checkpoint horizon
sweep). Pre-registration SHA-256, stamped into `exp_020_run_manifest.json`
before the first run and re-checked after the last, **unchanged**.

### 9.1 The scoreboard

| bar | verdict | one line |
|---|---|---|
| **T1** whole-run nesting | **PASS at exactly 0.0** | `if_nest_s0` reproduced `if_anchor_s0` to **0.0 in both protocols** over a full 20,000-step run |
| **T2** gates before the first step | **PASS** | G1, G2, G8 held; the §7.2 screen was green with all three synthetic controls firing |
| **H1** what the redundant block was buying | **FAILED (magnitude)** | `fold` costs **+0.09459** bpc against a predicted ceiling of +0.0582; paired t = 24.42, 0/3 seeds better |
| **H2** the candidate | **UNRESOLVED**, in the direction of harm | `foldwide` is **+0.01326** *worse* than the anchor; paired t = 3.45, 0/3 seeds better |
| **H3** the price of a binary input | **FAILED (arm worse)** | `binin` costs **+0.04569** bpc at **identical** parameters; paired t = 14.24, 0/3 seeds better |
| **H4** the multi-layer readout | **UNRESOLVED** | `mlayer` is +0.00530 worse, inside 2σ; paired t = 1.86, 1/3 seeds better |
| **H5** reach | measured, §9.4 | |
| **H6** cost | measured, §9.6 | |

**Every arm lost.** No arm in this experiment beat its anchor, and two of the
four lost by more than the bar. That is the result.

### 9.2 σ transferred, and re-measured

The transferred 0.00461 (5 baseline seeds, a different tree) against **0.004683**
measured on this tree's three anchor seeds: **×1.02**. The transfer held, which
is the first time this project has checked one and found that it did.

### 9.3 H1 — the redundant block is load-bearing, and by more than its size

`layers.0.weight` spans no function the embedding does not (§2.1, verified at
5.2e-08 relative). Removing it costs **+0.09459 bpc**, against the +0.0582 that
`EXP_014`'s scaling slope predicts for a generic parameter cut of the same size:
**1.63× more than the parameters are worth.**

This is `EXP_008`'s result repeating one layer earlier and larger. There, a
provable reparameterisation was worth −0.0590 bpc in the *over*-parameterised
direction. Here, removing a provably redundant factorisation costs +0.0946. Both
say the same thing: **on this architecture, redundant parameterisation of the
input path is worth real bits to the optimiser, and the function class does not
predict how many.**

### 9.4 The decomposition, and it inverts two of the four readings

`EXP_004` §10.3's split, cut at the baseline's horizon of 7, positive = better:

| arm | total | c = 0 | within reach | beyond horizon |
|---|---:|---:|---:|---:|
| `if_fold` | −0.09453 | −0.02710 | **−0.08143** | +0.01400 |
| `if_foldwide` | −0.01366 | **+0.03973** | **−0.07047** | +0.01708 |
| `if_binin` | −0.04478 | **−0.03710** | −0.00291 | −0.00477 |
| `if_mlayer` | −0.00535 | −0.03695 | +0.03128 | +0.00032 |
| `if_nest` *(bitwise the anchor)* | −0.00499 | −0.01395 | +0.00944 | −0.00047 |

Three things the mean alone does not say:

* **`foldwide`'s two effects have opposite signs and nearly cancel.** The width
  the fold frees buys **+0.0397 at zero context** — 2.8× the noise floor in
  §9.5, and exactly the per-step capacity `EXP_014` says width buys — while the
  fold costs **−0.0705 within the baseline's own 7-character reach**. The
  headline −0.0137 is the sum of a real gain and a larger real loss, and
  reporting only the sum would have hidden both.
* **The fold's cost is a REACH cost, not a capacity cost.** 86 % of `if_fold`'s
  loss is within-reach; only 29 % is at zero context. Removing a redundant
  *input* parameterisation damages the model's use of context, which no argument
  from parameter counting predicts.
* **Binarising the input is almost purely a zero-context cost.** 83 % of
  `if_binin`'s −0.0448 sits at `c = 0`; it is −0.0029 within reach and −0.0048
  beyond. A `{0,1}` code with a derived fixed gain is a less expressive
  *per-character* representation than an arbitrary real vector, and that is the
  whole of its price. It costs essentially nothing in memory.

### 9.5 The instrument failed a control, and the control is the finding

`if_nest_s0` is **bitwise** `if_anchor_s0` — T1 puts their final bpc exactly 0.0
apart — so every number in its row is one seed against a three-seed mean and
nothing else. It reads **−0.00499 total, −0.01395 at c = 0, and 120 of 128
contexts "significantly worse"**.

**`n_contexts_significantly_worse` is therefore unusable at n = 3 seeds**, and no
arm's count in this artifact is evidence of anything. The per-context bars fall
to 2.4e-05–6.0e-05 beyond c ≈ 70, while a bit-identical run differs from the
three-seed mean by ~0.005 there — the bars are up to **80× below** single-seed
variation.

This is fresh evidence on **decision #2**, open since `EXP_005` §9.4 found §6.2's
bar "built on the wrong scale": measured on this tree, with a bit-identical
negative control, the per-context construction saturates. **No redefinition is
proposed and §6.2 is not edited** — #2 is Elliot's.

`if_mlayer`'s −0.00535 total and `if_nest`'s −0.00499 are indistinguishable.
**H4's UNRESOLVED is a statement about power, and the decomposition does not
rescue it**: whatever the readout is worth, the 512 → 471 width cut that paid for
the head is the same size as the effect. §4 H4 named that failure mode in advance
and required §9 to say so, which this does.

### 9.6 Cost

Wall-clock against `if_anchor_s0`: `fold` **0.79×**, `binin` **1.07×**,
`foldwide` **1.20×**, `mlayer` **1.24×**. No arm exceeded the 4.0 GiB VRAM alarm.

### 9.7 Gates

| gate | result |
|---|---|
| G1 adopted arms untouched | **PASS** — no new kernel, no hand-written backward; R10 untouched, the 59-mutation campaign not re-run |
| G2 realised = closed form, exactly | **PASS**, 16/16 |
| G3 / K1 one thing changed | **PASS**, no unexpected field on any run |
| G4 whole-run nesting | **PASS at 0.0** |
| G5 frozen recipe | **PASS**, no violation on any run |
| G6 VRAM alarm | **PASS**, 0 alarms |
| G7 divergence read from the log | **PASS**, 0 diverged, 16/16 complete |
| G8 §7.2 screen before the first step | **PASS** |
| G9 mutation lockfile absent | **PASS** |

### 9.8 Defects this experiment found in its own instruments

Three, all found after the ladder, none affecting a number above:

1. **`local_dopamine_param_count` omitted the learned `nm_log_tau`** — 736,973
   against a realised 736,974. Caught by `tests/test_neuromod.py`'s
   `xfail(strict=True)` **before** `EXP_021`'s G2 would have aborted its ladder
   on the first run.
2. **`iface_read_lags > 1` was silently wrong at `L = 1`.** `_lag` pads inside
   the window and no lag state is carried, so a multi-tap readout decodes with
   every lagged block permanently zero — no raise, no capture failure, and the
   eager and captured paths agreeing with each other while both are wrong. It
   now raises. **No committed run is affected**: leg 6 was NOT RUN and
   `iface_read_lags` is 1 everywhere.
3. **The resolver read the per-context bars from a wrapper object**, so
   `context_comparison` fell back to the flat bar and reported 2/128 where the
   probe's own count is 120/128. Caught only because two implementations of one
   statistic disagreed — which is the reason `EXP_011`'s resolver imports the
   statistic instead of reimplementing it, and it still took a wrapper to make
   the point.

### 9.9 Cost, and the project total

~1.9 GPU-hours against a budgeted ~2.1. **Nothing here bought no number**: all 16
runs completed and were scored. Project total: **~17.7 of ~30 GPU-hours.**

### 9.10 The pre-registration guarantee

Nothing above §9 was rewritten. Every bar is reported as it fired, including
T1's failed tolerance in §2.6 — a two-sided 0.02 on a refit that is *expected* to
win, **reported unrepaired**.

---

## 11. Referred to Elliot, and not decided here

*(added after the run; created by §9 and marked as such)*

1. **Whether the per-context §6.2 construction survives its own negative
   control.** §9.5 is the first time this project has run a bit-identical arm
   through it, and it failed 120/128. Evidence for **#2**; no redefinition is
   proposed.
2. **Whether "the fold costs reach" deserves a mechanism study.** §9.4's finding
   — that removing a *redundant input parameterisation* puts 86 % of its loss
   within reach — is explained by nothing in this report, and `EXP_012`'s
   fold-gap machinery is the instrument that could ask.
3. **Whether `mlayer` is worth re-running unmatched.** H4 is confounded by the
   width cut that paid for the head; three runs at `d = 512` (+14.3 % parameters)
   would separate "the readout does not help" from "the width cut ate it". Not
   run here, because it is not the arm that was pre-registered.
4. **What the binarity price means for the project.** §9.4 puts it at 0.0448 bpc
   at identical parameters, 83 % of it at zero context, against the analogue
   control's 0.0117 for removing binarity from *every inter-layer* signal
   (`02_baseline_report.md` §6.2). **The input boundary is ~4× more expensive to
   binarise than all the inter-layer ones together.** Whether I1 should be
   extended to the input at that price is Elliot's, and nothing here recommends
   it.

---

## 10. Referred to Elliot, and not decided here

1. **Whether the fold is adopted**, at either width. §5 produces a sentence; it
   does not produce an adoption.
2. **Whether a multi-layer readout is adopted.** It costs `(K-1) d V`
   parameters, which is 14.3 % of the model at `K = 2` and grows with depth, so
   it is a different trade at `K = 4` than at `K = 2` and this experiment
   measures only the latter.
3. **Whether "the model's input is binary" is worth anything to the project
   beyond I1's letter.** Leg 4 can say what it costs. It cannot say what it is
   for, and no claim is made that it is for anything.
4. **Whether §7.1's ranking should gain a row for the interface.** Ranking it
   here would be this experiment choosing its own successor, which
   `04_phase4_interim.md` §7.1 forbids.
