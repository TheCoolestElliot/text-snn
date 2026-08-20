# EXP_024 — Six arms have now failed to extend reach. This one changes the only coefficient that sets it.

**Pre-registered:** 2026-08-20, after the arithmetic in §2 and the §7.2
reachability screen in §2.6, and **before any line of `src/snn/` for this arm
exists, before any checkpoint is written, and before any bpc exists for any arm
in §3.**

**Status: PRE-REGISTERED — NOT BUILT AND NOT RUN. §8's entry conditions are NOT
met.** This file is committed at the phase gate. Nothing in §3 may be trained
until every item in §8 is signed off, and §8 item 1 is Elliot's.

---

## 0. What this experiment is, and what it is not

`EXP_015` §14.3 located the whole matched-size gap to the GRU anchor in **reach**:
at 5.0M the anchor's memory horizon is **97** and the spiking model's is **8**,
while at zero context the two are 0.051 apart. Since then this phase has measured
**six arms against the beyond-horizon component and moved it with none of them**:

| arm | beyond-horizon gain | source |
|---|---:|---|
| `nm_local` | −0.00052 | `EXP_021` §9.4 |
| `nm_const` | +0.00036 | `EXP_023` §9.6 |
| `nm_pos` | +0.00011 | `EXP_023` §9.6 |
| `nm_rolled` | −0.00025 | `EXP_023` §9.6 |
| `io_mall` | +0.00154 | `EXP_022` §9.4 |
| `io_wide` | −0.00024 | `EXP_022` §9.4 |
| *(bitwise copy of the anchor — the floor)* | −0.00047 | `EXP_020` §9.5 |

**Every one of them is inside the noise floor a bit-identical run reads.** Width
does not buy it (`EXP_014`: horizon 7 → 8 for 7× the parameters), a rank-1 gain
on the current does not (four rungs, `EXP_023`), and a wider readout does not
(`EXP_022`, and §2.4 of that experiment *predicted* it would not).

**The one thing that has ever moved reach is a second, slower state variable** —
`twocomp`'s slow pole, 7 → 47 — and `EXP_006` proves causally that ~3 % of its
channels carry ~95 % of it. That arm is behind decision #11.

**THE RUNG THAT COULD DEFLATE THIS EXPERIMENT, NAMED BEFORE IT RUNS.** With the
gate held at zero this arm is a **learned per-channel decay, constant in time** —
which is ranked candidate #5, admitted by `01_reconnaissance.md` §4.6 row 1, and
**never run in this project's history**. `EXP_023` is the standing warning: its
`nm_const` rung, which knows *nothing*, was worth −0.02975 against `nm_local`'s
−0.02953, and §9.7 concluded *"the optimiser buys the form, not the signal."*
**If that repeats here, this experiment measures candidate #5 and nothing else**,
and §5 says so in a cell written now. The constant rung runs **first** for that
reason.

**What this is not:**

* **It adopts nothing and ranks nothing.** Adoption is Elliot's (report §8), and
  no arm here may be ranked in §7.1.
* **It changes no hyperparameter.** The Phase-2 recipe — `lr = 3e-3`, cosine over
  20,000 steps, `grad_clip = 1.0`, `weight_decay = 0.1`, `beta = 0.5`,
  `threshold = 1.0`, hard reset, `atan` at `alpha = 2` — is frozen, and G5 reads
  all of it back out of each run's own `config.json`.
* **It does not decide #11 and does not touch the two-compartment family.**
  Every arm here is a **plain LIF** at 735K. Nothing is run at 5.0M.
* **It does not edit any adopted arm.** The new neuron gets its own module —
  `src/snn/gateddecay.py` — exactly as `twocomp.py` and `twocomp_detach.py` did,
  and for the mechanical reason `08_mutation_campaign.py:414-420` gives: a
  mutation's anchor text must appear exactly once in its target file, and a
  second neuron inside `kernels.py` would duplicate lines like
  `const T s = (v >= thr) ? T(1) : T(0);`.
* **It DOES carry a new fused kernel and a new hand-written backward**, which is
  the one thing every arm since `EXP_017` has avoided. §2.5 states that bill in
  full and §7's T2 is the gate that must pass **before a single training step**.
* **It violates no invariant.** I1, I2, I3, I4 untouched. I5: the new parameters
  are `beta_raw` and `k_gate`, both `[1, d]`, so the O(1)-per-neuron test holds;
  there is no `[d, d]` lateral matrix and no free-form FIR kernel.
* **It does not re-baseline.** Decision #10. The anchor is `if_anchor` seeds 0–5,
  trained in this session on this tree.

---

## 1. The question

> **A constant leak lengthens memory by blurring it. A gain on the current does
> not lengthen it at all. Does a decay that varies per step with what just
> arrived lengthen it while keeping recent context sharp — and is that worth
> anything the same decay learned as a constant is not?**

---

## 2. The derivation, done before any measurement

### 2.1 Why a *constant* decay is the wrong instrument, measured rather than argued

`EXP_001` §8.3 swept the scalar `beta` and the result is the reason this arm has
the shape it has:

| `beta` | horizon | test bpc |
|---:|---:|---:|
| **0.5** | **7** | **2.2697** |
| 0.9 | 24 | 2.3276 |
| 0.95 | 35 | 2.3702 |

**Raising `beta` buys horizon — 7 → 24 → 35, a 5× increase — and loses bpc
monotonically over exactly that range.** `EXP_001` §8.4 names the mechanism: a
scalar leak lengthens memory by **blurring** it, and at `beta = 0.95` the present
is 1/20th of the signal. That section then wrote the acceptance criterion this
experiment must be judged against, and H4 below is it:

> *Any candidate that shows the `beta` signature is buying horizon the losing
> way.*

**A gate is the one admitted mechanism that can avoid that signature**, because
it lets each channel choose per step whether to retain or overwrite instead of
retaining everything equally. That is what the GRU has and the LIF does not, and
`EXP_001` §8.4 says so in as many words.

### 2.2 The form, and why it is free

    beta_{t,c} = sigmoid( logit(beta_c) + k_c * cur_{t,c} )
    v_t        = beta_{t,c} * v_{t-1} + cur_t

`cur = W_k · s^{k-1} + b_k` is **layer `k−1`'s output, already projected**, and it
is computed **before** the time loop. So:

* **It is the shape decision #3 admits.** Driven by the previous layer, computed
  before that layer's time loop, depth-sequential evaluation preserved.
* **It needs no extra GEMM and no extra kernel launch.** The literature form
  (`01a_literature_review.md` §4.4) is `beta_t = sigmoid(W_b · s^{l-1})`, which
  costs one extra `d × d` GEMM per layer **and** is O(d) parameters per neuron,
  which the I5 O(1)-per-neuron test forbids. Driving from `cur` reuses a tensor
  the forward pass already holds and adds **two `[1, d]` parameters per modulated
  layer**.
* **The `beta_t` tensor is `[B, L, d]` and is built outside the time loop**, so
  the one-fused-kernel-per-timestep floor survives. `EXP_002` §8.6's conclusion
  stands: every §4.6-admitted neuron is systems-free, and the real cost is the
  gradient gate.

**Layer 0 is unmodulated**, matching `EXP_021` and `EXP_023`. Whether layer 0's
`cur` — which comes from the embedding rather than from a spiking layer — counts
as "the previous layer's output" is a **boundary question and is Elliot's**;
`EXP_019`'s regional census (L0's slow channels load-bearing in 26 of 26
checkpoints) is the reason someone would want it, and it is **not asked here**.

### 2.3 The nesting ladder is exact at every rung, by construction

| rung | `beta_raw` | `k_gate` | what it is |
|---|---|---|---|
| `off` | `logit(0.5)`, frozen | `0`, frozen | **Phase 2, bitwise** |
| `const` | learned | `0`, frozen | **ranked candidate #5**, never run |
| `gated` | learned | learned | the arm |
| `rolled` | learned | learned | the arm, drive rolled across the batch |

At `k_gate = 0` the sigmoid argument is `logit(beta_c)` and `beta_{t,c} = beta_c`
exactly — **a float identity, and therefore not good enough on its own.** G4
asserts the nesting through the *code path*, as `nm_source="off"` does: the
`off` rung must reproduce a Phase-2 run **bitwise, forward and backward**, and
`const` must reproduce a `_scan_learned_decay` reference.

### 2.4 Where the answer must appear, and why bpc is not the bar

**The project has never measured a σ for the horizon at any width for any arm**
(`EXP_017` §10 item 8), and `EXP_017`'s H4 is what a bar on a horizon
*difference* is worth: it fired **REACH REDUCED at Δh = −2** while the three
seeds it rested on spanned **47–50**. So the integer horizon is reported here as
a **marker with no verdict**, exactly as `EXP_015` P4 was.

The bar is on the **`EXP_004` §10.3 decomposition**, against the noise floor
`EXP_020` §9.5 measured with a **bitwise copy of the anchor** — the first honest
floor that statistic has ever had:

| component | bitwise-copy floor |
|---|---:|
| total | −0.00499 |
| `c = 0` | −0.01395 |
| within reach | +0.00944 |
| **beyond horizon** | **−0.00047** |

**H1 is on the beyond-horizon component**, which is 20× that floor at the 2σ bar
and is the component §0 shows six arms have failed to move.

### 2.5 What this costs to build, stated before anything is built

**This is the expensive class of change and the first since `EXP_017`.** `beta`
is a *scalar* kwarg of the compiled kernel (`kernels.py:112,227`) and jiterator
binds every tensor argument before the scalars — the incident `twocomp.py:94-103`
records is a kernel that compiled, ran, produced sensible spikes and priced
within 2 % while silently binding a per-channel tensor into a scalar slot. So a
per-element `beta` means a new signature, new source text and a new compiled
kernel. And `neuron.py:257` returns `None` for `beta`'s gradient today; the new
backward gains

    dL/dbeta_t = g * v_{t-1},        g = grad_v_next * dv + grad_spike * sg

**Roughly half the arithmetic already exists and was verified.**
`scripts/exp/002_candidate_neuron_cost.py:80-90` is the forward for a per-channel
decay (max abs err 2.4e-07, spikes bit-identical) and `:159-173` is its backward,
already emitting `grad_beta_elem = v_prev * g`. Unlike `twocomp`'s `[1, d]`
decay, a per-element `beta_t` gradient is `[B, L, d]` and needs **no reduction**,
so `EXP_002`'s Q5 penalty does not apply in that form. `v_{t-1}` needs no new
saved tensor: under hard reset it is `v_pre_{t-1} · (1 − s_{t-1})`, recoverable
from the already-saved `v_pre` stack shifted by one with `v0` in front — the
trick `twocomp.py:489` uses.

Arity: 3 tensors in / 3 out forward, 4–5 in / 3 out backward, against jiterator's
8/8. `twocomp` already runs 7/5 and calls itself the widest in the project.

**The bill, itemised, and it is engineering rather than GPU:**

| item | precedent | budget |
|---|---|---|
| `src/snn/gateddecay.py` | `twocomp.py` | ~500 lines |
| 4-leg equivalence gate: fused-vs-eager (spikes **bit-identical**), an implementation transcribed in the test file from the spec tables, fp32-vs-fp64, `gradcheck` + a third-party cross-check | `twocomp_detach` = 30 tests | ~30 tests |
| mutation block, **whole campaign re-run** | 59/59 today (25 + 22 + 12) | **+15–22 → 75–81** |
| Config fields, validation, help, choices, K1 exemption | `localdopamine` added 6 | 3 |
| §7.2 screen | **DONE — §2.6** | — |
| cost pre-flight | `014_calibrate_cost.py`, 400 steps | minutes |

**The swept regimes matter here more than usual.**
`tests/test_neuron_equivalence.py:27-38` exists because `beta = 0.5` is a power
of two, so the decay multiply is exact and an FMA contraction is invisible, and
`thr = 1.0` collapses `1 − thr·sg` into `1 − sg`. **This arm makes `beta`
per-element and time-varying, so it is off 0.5 by construction on almost every
element** — the gate must sweep it deliberately and must include an equality case
at `v_pre == thr` and one ulp below, which is where `EXP_017`'s D09 escaped.

### 2.6 The §7.2 reachability screen — RUN, before this file was committed

`docs/reports/data/exp_024_reachability.json`, through the real arch, at the
Phase-2 recipe and seed:

| candidate | init | verdict | `|dL/dbeta_raw|` ratio | `|dL/dk_gate|` ratio |
|---|---|---|---|---|
| `c05` #5 learned per-channel decay | `beta_raw = logit(0.5)` | **PASS** | 0.116 / 0.399 | — |
| **`c24_gated`** | `beta_raw = logit(0.5)`, `k_gate = 0` | **PASS** | 0.116 / 0.399 | **0.041 / 0.042** |
| **`c24_rolled`** | same | **PASS** | 0.116 / 0.399 | 0.021 / 0.029 |

`derived_zero` was declared **empty** for both new candidates before the screen
ran, and the measurement agrees: **no parameter is exactly zero and neither arm
is a saddle.** The derivation, written before the measurement:

> `dL/dk_c = Σ_t (dL/dv_pre_t) · v_{t-1} · beta_t(1−beta_t) · cur_{t,c}` — every
> factor is nonzero at the nesting init, and **no parameter's only path to the
> loss runs through another that starts at zero**, which is what separates this
> from `EXP_004` §2.2's saddle and from `nm_pos`'s one-sided one.

The screen's own controls held (`ctl_live` PASS / `ctl_dead` SADDLE / `ctl_faint`
FAINT) and it reproduced `exp_004_gradient_reachability.json` at worst relative
residual **0.00e+00**. Every ratio is above the 0.01 floor, so nothing reads
FAINT — but `k_gate`'s 0.041 is **2.8× below `beta_raw`'s 0.116 at layer 0**, and
that is recorded now rather than discovered later.

### 2.7 A confound this experiment inherits, and the one it creates

**Inherited.** AdamW's decoupled weight decay pulls every raw parameter toward
zero. `beta_raw = 0` is `beta = 0.5`, so decay pulls the learned decay toward the
Phase-2 value rather than away from it — the opposite of the two-compartment
case, where `docs/chat/WEIGHT_DECAY_NOTE.md` measures it pulling `tau` from 20
to ~2.16 and accounting for 86–96 % of the observed median. **Here the fixed
point IS the baseline**, which is benign for nesting and means a rung that ends
at `beta ≈ 0.5` has not necessarily *chosen* it. §9 reports the final `beta_c`
distribution per rung as a diagnostic, and if every channel returns to 0.5 the
ladder measured a regulariser.

**Created.** `k_c · cur_{t,c}` couples the gate's effective scale to the scale of
`cur`, which is not a constant across training. `EXP_023` §9.8 is the standing
example of what that costs: an arm certified unsaturated at one scale and trained
at another, where the squash started at `sech² = 0.004` instead of 0.777. **The
sigmoid here is at `logit(0.5) = 0` at init with `k_gate = 0`, so it starts
exactly at its most sensitive point** — but nothing holds it there, and §9 must
report the realised `sd(beta_t)` per rung. **A rung whose `beta_t` has collapsed
to a constant is the `const` rung wearing the gate's name**, which is exactly the
defect class G8 was invented for in `EXP_023`.

---

## 3. The legs

Seeds 0, 1, 2. `if_anchor` seeds 0–5 (n = 6) is **reused** from `EXP_020`/
`EXP_023` and not re-run — same recipe, same tree, same seeds, and re-training it
would produce two anchors that could disagree.

| leg | run names | `beta_raw` | `k_gate` | realised params |
|---|---|---|---|---:|
| **0** | `gd_off_s0` | frozen `logit(0.5)` | frozen `0` | 735,437 + 2d·(K−1) |
| **1** | `gd_const_s{0,1,2}` | learned | frozen `0` | 735,437 + 2d·(K−1) |
| **2** | `gd_gated_s{0,1,2}` | learned | learned | 735,437 + 2d·(K−1) |
| **3** | `gd_rolled_s{0,1,2}` | learned | learned | 735,437 + 2d·(K−1) |

**Every rung carries both parameters even when one is frozen**, deliberately, so
that G2's closed form is the same for all four and a frozen parameter is asserted
to reach nothing rather than assumed to. This is `nm_source="off"`'s construction
(`EXP_021` §2.2) and the reason it exists.

**Order is load-bearing: leg 0 first, then leg 1, then legs 2 and 3.** Leg 0's
bitwise nesting must hold before any rung that changes a number is trained, and
leg 1 is §0's deflation rung — if it takes the whole effect, legs 2 and 3 are
measuring something already measured.

`K = 2`, so there is one modulated layer and `2d·(K−1) = 1,024` new parameters,
**0.14 % of the model**. No rung is a width arm in disguise.

---

## 4. Pre-registered predictions

σ is **transferred**: 0.00461, so **2σ = 0.00922**, as `EXP_020`–`EXP_023` used
it. **σ MUST be re-measured on this arm** — §3 of the report is explicit that it
is not a project constant, and `EXP_022` §9.5 found `io_mall`'s at 15× below the
anchor's — so §9 reports the on-tree σ at n = 3 per cell as a marker. Comparisons
are **paired by seed**; `t` is the paired `t` over 3 differences unless stated.

| # | claim | cells | bar | direction predicted |
|---|---|---|---|---|
| **H1** | **THE HEADLINE. Does it reach?** `gated`'s **beyond-horizon** component against the anchor | `gd_gated` vs `if_anchor` | ≥ 2σ = 0.00922 on the beyond-horizon component alone | **gated BETTER.** Six arms have moved this component by at most 0.0015 (§0); a gate on the decay is the only admitted mechanism aimed at it |
| **H2** | **THE DEFLATION BAR.** Does the gate buy anything the constant does not? | `gd_gated` vs `gd_const`, paired | ≥ 2σ on total bpc | **gated BETTER.** If it is not, this experiment measured candidate #5 and §5 says so |
| **H3** | is the drive's *content* load-bearing? | `gd_gated` vs `gd_rolled`, paired | ≥ 2σ on total bpc | **gated BETTER.** `EXP_023`'s H4 asked this of the current and got −0.0022 at n = 6; this asks it of the decay |
| **H4** | **THE ANTI-SIGNATURE.** `EXP_001` §8.4's criterion | `gd_gated`'s near-context excess at `c = 1..4` vs the anchor's | **must NOT rise** by ≥ 2σ | `beta = 0.9` raised it to +0.4660 at 3 characters against the baseline's +0.2913. **An arm that buys reach by blurring FAILS, whatever it does to bpc.** |
| **H5** | the constant rung against the anchor — **candidate #5, first measurement** | `gd_const` vs `if_anchor` | ≥ 2σ on total bpc | **UNSTATED DIRECTION.** `03_phase3_candidates.md` calls #5 "partly subsumed by the adopted arm", and that claim is a structural argument with no measurement behind it. Predicting a sign would be predicting the thing this rung exists to find out |
| **H6** | findability across the ladder | `rms(k_gate)` per rung, `gated` vs `rolled` | none; reported | `EXP_018` got **26** on the analogous quantity, `EXP_021` got 1.12, `EXP_023` got **1.1176 across four rungs**. **If this reads flat again it is the third time, and that is the finding whatever the bpc does** |
| **H7** | did the gate actually gate? | `sd(beta_t)` and the final `beta_c` distribution, per rung | none; reported | §2.7. **A rung whose `beta_t` collapsed to a constant is the `const` rung wearing the gate's name** |
| **H8** | decomposition, every rung | `EXP_004` §10.3 split | none; reported | §2.4's floor beside every row |
| **H9** | integer memory horizon | per rung | **none; MARKER WITH NO VERDICT** | `EXP_017` §10 item 8: no σ exists for this quantity at any width for any arm |
| **H10** | cost | realised vs pre-flighted | none; reported | pre-flight is §8 item 5 and is not guessed |

**H1 IS THE ONE TO READ FIRST, AND IT IS THE ONE MOST LIKELY TO FAIL.** State its
power now, per `CONTRIBUTING.md` §2: at n = 3 with a transferred σ, a
beyond-horizon effect between 1σ and 2σ reads **UNRESOLVED**, and *"the band was
failed"* and *"the difference was not established"* are different sentences. The
beyond-horizon component has never been observed outside ±0.0016 on any arm, so
**an UNRESOLVED H1 is the modal outcome and is not a disappointment.**

**`n_contexts_significantly_worse` IS NOT USED AS A BAR ANYWHERE.** `EXP_020`
§9.5 measured it against a bitwise copy of the anchor and it read 120 of 128.
Marker only.

---

## 5. Decision rule, fixed now

| outcome | what is written in §9 |
|---|---|
| **H1 holds and H4 holds** | **"a feedforward gate on the decay extends reach, and not by blurring"** — the first arm in the project to move the beyond-horizon component. Referred for adoption, never ranked here |
| **H1 holds and H4 FAILS** | **"it bought reach the losing way"** — `EXP_001` §8.4's signature, and the arm is reported as a `beta` sweep with extra steps |
| **H1 fails, H2 holds** | **"the gate pays and not at reach"** — reported with the decomposition, and the within-reach/`c = 0` split is where it went |
| **H2 fails** | **"this experiment measured candidate #5"** — §0's deflation rung took the effect, `EXP_023`'s conclusion transfers to the decay, and the ladder is reported as the first measurement of ranked candidate #5 rather than as a gate result |
| **H5 holds and H2 fails** | as above, **and candidate #5 is worth its measurement independently** — reported as a rung, never as an adoption, and #6 is load-bearing exactly as it is for `EXP_023` §9.4 |
| **H6 flat across the ladder** | **"the optimiser buys the form, not the signal"** — the third time, and recorded whatever the bpc does |
| **H7 shows `sd(beta_t)` ≈ 0 on `gated`** | **the gated rung is reported as a failed gate and its bpc is not read as a gate result** — G8's clause from `EXP_023`, transferred |

Every cell above is written before the arm exists and none is softened afterwards.

**A cell for the verdict a 2σ bar most often returns**, which `EXP_022` §5 and
`EXP_023` §5 both lacked and which fired four times on 2026-08-20: **an
UNRESOLVED bar is reported as UNRESOLVED**, is not read as either a pass or a
fail, and routes to the same cell as its failure **only where this table says
"fails"**. H1's unresolved case in particular is covered by row 3.

---

## 6. Known limitations, stated in advance

1. **A new hand-written backward is the worst bug class in this project (R10),
   and it still trains and still scores plausibly when wrong.** §7's T2 is the
   only thing between that and a published number, and it is abort-on-fail
   **before a single training step.**
2. **The gate's drive is `cur`, not the previous layer's spikes directly.** These
   differ by `W_k` and `b_k`, which are learned, so the drive's scale is not
   stationary (§2.7). An arm driven by the raw spikes is a different arm and is
   not run.
3. **Layer 0 is unmodulated**, so the whole ladder measures a gate on 1 of 2
   layers. `EXP_021` §6 and `EXP_023` §6 carry the same limitation.
4. **Everything is at `d = 512, K = 2`, on the plain LIF.** `EXP_015` is the
   standing reason a 735K result need not survive to 5.0M, and `EXP_014` §13.6 is
   the reason a plain-LIF result need not transfer to the adopted neuron.
5. **n = 3 per cell, against a transferred σ.** Effects between 1σ and 2σ read
   UNRESOLVED. H1 in particular is under-powered for a component whose observed
   range across six arms is ±0.0016 — stated in §4 rather than discovered in §9.
6. **`beta_raw`'s weight-decay fixed point is the baseline** (§2.7), so a rung
   ending at `beta ≈ 0.5` has not necessarily chosen it.
7. **The horizon integer has no σ anywhere in this project** and is a marker.
8. **`EXP_001`'s β sweep is n = 1 at `beta = 0.9` and `0.95`.** H4's comparison
   values come from it and inherit that, which is why H4 is a bar on *this*
   experiment's own near-context excess against *this* experiment's own anchor,
   and not a comparison against `EXP_001`'s numbers.

---

## 7. Gates — aborting, not advisory

| gate | what it asserts | on failure |
|---|---|---|
| **T2 / R10** | the new equivalence gate green **and** `08_mutation_campaign.py` at **75–81/75–81, whole campaign, 0 escaped**, **before a single training step** | abort the experiment |
| **G1** | adopted arms untouched: `git diff --stat` empty over `neuron.py`, `kernels.py`, `surrogate.py`, `twocomp.py`, `twocomp_detach.py` | abort the ladder |
| **G2** | realised parameter count **equals** the closed form, exactly, for all 10 runs | abort that run |
| **G3 / K1** | one field changed per arm against its named reference; anything outside `may_differ` aborts | abort the ladder |
| **G4** | the `off` rung nests Phase 2 **bitwise, forward and backward**, through the code path and not through a float identity; and `const` reproduces `_scan_learned_decay` | asserted in `tests/test_gated_decay.py` |
| **G5** | the frozen recipe read back out of each run's own `config.json` | abort that run |
| **G6** | peak VRAM alarm | report, do not abort |
| **G7** | divergence read from `log.jsonl`, **not** the exit code | record, evaluate anyway, continue |
| **G8** | **`sd(beta_t)` per rung, reported for every run. If any `gated` seed finishes with `sd(beta_t)` below the `const` rung's, that run is reported as a FAILED GATE and its bpc is not read as a gate result** | record; §5's last cell |
| **G9** | mutation lockfile absent, `src/` clean, no other Python process alive | refuse to start |

**G8 is `EXP_023`'s clause transferred**, and it exists because a rung whose gate
never opened would train, score plausibly, and be the `const` rung wearing the
gate's name.

---

## 8. Entry conditions — NONE OF THESE ARE MET

1. **Elliot signs off on paying the R10 tax.** This is the first arm since
   `EXP_017` to need a new kernel and a new hand-written backward, and
   `EXP_018` §9 item 5 priced that in advance as the reason not to build it
   before the signal was worth it. **§0's table is the argument that it now is;
   whether that is enough is not this file's to decide.** Report §4 is open, and
   `CONTRIBUTING.md` §1 says a phase ends at a human gate.
2. `src/snn/gateddecay.py`, its Config fields and its `build_model` branch, with
   the K1 exemption recorded (adding any Config field makes every pre-existing
   reference run fail K1; exempt new-field-at-default only).
3. `tests/test_gated_decay.py` green, all four legs of §2.5's gate.
4. **T2**: the mutation block written and the **whole** campaign re-run and
   re-committed at 0 escaped. The 59 pre-existing mutations are **re-verified,
   not assumed** — `EXP_017`'s campaign is the precedent and its first run is
   what caught D09.
5. Cost pre-flight through `scripts/exp/014_calibrate_cost.py`, 400 steps per
   rung, denominator `snn_beta0.5_s0`. **The budget is computed from it and not
   guessed** — §7.1's one previously-checked estimate was low by 3.1×.
6. `scripts/exp/024_gated_decay_results.py` committed **before any bpc is read**.
   `EXP_022` and `EXP_023` met this condition; `EXP_020` and `EXP_021` did not,
   and report §8's rev-14 note records that.
7. Pre-registration SHA-256 stamped into `exp_024_run_manifest.json` before the
   first run and re-checked after the last.
8. Nothing else runs on the GPU while a timed run is in flight.

**Budget: unknown until item 5, and deliberately not estimated here.** What is
known: `EXP_002` N1 priced a per-channel decay's *systems* cost at a relative
step time of **1.0117** with no bpc produced, and 10 runs at the anchor's
measured 398.9 s would be ~1.1 GPU-h if that transfers. It is one kernel per
timestep either way. **The engineering, not the GPU time, is the cost.**

---

## 9. Results

*(appended after the run; nothing above this line is rewritten)*
