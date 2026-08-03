# EXP_004 — Does a second, reset-shielded timescale buy horizon without losing bits?

**Pre-registered:** 2026-08-03, before any two-compartment kernel was written.
**Status:** PRE-REGISTERED — no results yet.
**Phase:** 4 (controlled experiments). The first Phase-4 arm; candidate **#1** of
`03_phase3_candidates.md` §6.3.

---

## 1. Why this candidate, and why first

Phase 3 ended with a mechanism and a ranking. The mechanism is that the I5 gap is
a **horizon** gap — 7 characters against the GRU anchor's 57–60 (`EXP_001`) — and
that only **49 %** of it is addressable by longer memory at all (§5). The ranking
put the **two-compartment neuron** at the top, for a reason that is structural
rather than hopeful: it nests the Phase-2 baseline exactly, so its downside is
bounded by construction.

It also sits directly on the phase's central negative result. Raising β lengthens
the horizon 7 → 24 → 35 and degrades bpc monotonically over exactly that range,
because **one state variable cannot hold the distant past and the present
character at once** — a scalar leak buys reach by low-pass filtering. The GRU
escapes that trade by gating per step; it is nowhere worse than the baseline at
any of 128 context lengths, while β = 0.9 is worse at 126 of them.

A two-compartment neuron is the cheapest available test of whether that trade is
about *gating* or merely about *having more than one pole*. It gives the neuron a
fast compartment whose only job is the present character and a slow compartment
whose only job is the past, with a learned per-channel mix deciding how much of
each reaches the threshold. If the β result is really "one variable, two jobs",
splitting the jobs should lift the horizon without the near-context damage. If the
β result is really "reach costs sharpness whatever you do", this arm will show the
same signature in milder form, and that is the more informative outcome — it would
demote the entire beyond-horizon family in one measurement.

`EXP_002` already priced this neuron's shape: **one kernel per timestep, identical
peak memory, 1.08× wall-clock** (N2). The systems column does not discriminate.
The real cost is the mutation-tested gradient gate, and §7 of the phase-3 report
made that much cheaper. So the question this experiment answers is purely one of
bits per character, which is what the ROI matrix said it would be.

---

## 2. The neuron

Per channel `c`, per timestep `t`:

```
vf_t  =  beta_f   * vf_{t-1} + cur_t          fast pole; beta_f = 0.5, FIXED
vs_t  =  beta_s,c * vs_{t-1} + cur_t          slow pole; beta_s,c LEARNED
v_t   =  vf_t + w_c * vs_t                    the mix that reaches the threshold
s_t   =  1[ v_t >= thr ]                      I3, hard threshold, thr = 1.0
vf_t <-  vf_t * (1 - s_t)                     hard reset, the baseline's rule
vs_t <-  vs_t                                 RESET-SHIELDED
```

Two free parameters per neuron (`w_c`, `beta_s,c`), both `[1, d]` per layer.
**Admitted by the §4.6 ruling, row 2** — "multiple learned timescales per neuron
(e.g. fast/slow membrane pair): *Yes*. Still O(1) per neuron; multi-timescale
dynamics are the stated scientific spine." No lateral mixing, no lag-indexed
weights, nothing referred to Elliot in §6.3 is touched.

### 2.1 Three design choices, and why each is what it is

**Additive mix, not convex.** `EXP_002`'s N2 prototype priced the convex form
`v = w*vf + (1-w)*vs`, which nests the baseline at `w = 1` — a *boundary* of a
constrained parameter, reachable only from one side. The additive form nests it at
`w = 0`, in the interior: at `w_c = 0` the readout is `v = vf`, `vf` is a plain LIF
at β = 0.5 with hard reset, and `vs` influences nothing. That is the Phase-2
baseline *exactly*, not approximately. The kernel shape, arity and cost are
unchanged from N2, so `EXP_002`'s pricing carries over untouched.

**The slow pole is reset-shielded, and that is the whole idea.** Under hard reset
the effective per-step retention is `beta*(1-p)` where `p` is the firing rate
(§6.3 candidate #9). At the baseline's measured `p ≈ 0.34`, a slow compartment at
β = 0.95 that also reset would retain 0.63 per step — barely more than the
baseline's own 0.5, and it would not be a slow pole in any useful sense. Shielding
is what makes the second timescale a second timescale rather than a second copy of
the first. It has a second consequence worth recording but **not used here**: an
unreset compartment is exactly linear in its state, hence exactly
parallel-scannable, which is the lever §4.6's rationale leans on and a Phase-5
question rather than this one.

**`beta_s` is stored unconstrained and squashed outside the time loop.** The
parameter is a raw `[1, d]` tensor; `beta_s = sigmoid(raw)` is computed **once per
layer per forward pass**, not once per timestep, so it costs O(1) kernels against
the K·L in the loop and cannot break the one-kernel floor. `w_c` is stored
directly, unconstrained — nothing requires the mix to be positive, and forcing it
would be an unlogged hyperparameter choice.

### 2.2 The exact-nesting init is a gradient saddle — derived before it is measured

`w_c = 0` is the attractive initialisation: the arm starts *as* the baseline and
moves away only if moving away helps. It is also, provably, a saddle for the other
parameter, and §6.4's precedent is that this is checkable from the backward
equations rather than discoverable after 0.8 GPU-hours.

The slow compartment's adjoint obeys

```
dL/dvs_t  =  w_c * dL/dv_t  +  beta_s * dL/dvs_{t+1}
```

with `dL/dvs_L = 0` in training (the returned state is discarded by
`Trainer._step_body`, which calls `model(x, None)`). At `w_c = 0` the first term
vanishes at every `t`, so by backward induction `dL/dvs_t = 0` for all `t`, and
therefore

```
dL/dbeta_s,c  =  sum_t (dL/dvs_t) * vs_{t-1}  =  0     exactly.
```

The mixing weight itself is reachable — `dL/dw_c = sum_t (dL/dv_t) * vs_t`, and
`vs` integrates the current whatever `w` is — so the saddle is one-sided: the arm
would train with a learnable mix and a **frozen** slow pole, in every seed, and
nothing in its logs would say so.

**Pre-registered fallback, fixed now: `w_c = 0.1`, `beta_s = 0.95`.** The slow
compartment contributes about a tenth of the readout at initialisation — small
enough that the arm starts near the baseline, large enough that both parameters
carry gradient. β_s = 0.95 is chosen because `EXP_001` measured a single pole at
that value reaching 35 characters; it is a *measured* starting point, not a tuned
one. The §7.2 screen decides between the two inits by its own stated rule, and
**whichever it selects is written into §9 of this file before training starts.**

The alternative — keep `w_c = 0` and make `beta_s` a fixed hyperparameter — was
considered and rejected: it would drop the candidate to one parameter per neuron,
which is candidate #5's territory (learned per-channel decay) and not this one's,
and it would make the arm untestable as the two-parameter neuron §6.3 ranked.

---

## 3. Hypotheses

**Pre-registered predictions.**

| # | Prediction | Threshold |
|---|---|---|
| **T0** | The exact-nesting init `w_c = 0` is an **exact gradient saddle** for the slow decay: `\|dL/d raw_beta_s\|` is identically zero, while `\|dL/dw_c\|` is not | §7.2 screen; fallback init named in §2.2 |
| **T1** | The horizon lengthens materially: **median horizon over 3 seeds ≥ 14 characters** | `EXP_001` paired statistic at the 2σ tolerance; baseline is 7, and 14 is what K = 8 reached |
| **T2** | **And it costs bits anyway.** Mean test bpc (carried) does **not** beat the baseline's 2.2531 by more than 2σ = 0.00922 | *stated in the direction that hurts the candidate* — see below |
| **T3** | The §6.2 curve shows the β signature in milder form: worse than the baseline at **≥ 1 short context (c ≤ 8)** by more than that context's own 2σ bar | `EXP_001` per-context bars |
| **T4** | The arm actually uses its slow compartment: mean `\|w_c\|` at step 20 000 is **≥ 2×** its initial value, in **both** layers | recorded from `ckpt_final.pt` |
| **T5** | **Ceiling.** Any improvement is smaller than **0.2279 bpc**, the whole beyond-horizon component of §5 | §5 denominator |

**On T2, explicitly.** The author's honest prior is near even. The mechanical
argument for the candidate is real — the slow pole's adjoint multiplier is a clean
`beta_s = 0.95` with no reset factor, so it is *better* conditioned over 20+ steps
than the baseline's `0.5*(1-s)`, and gradient can genuinely reach back. The
argument against is equally real: adding a slow, smooth signal to the membrane is
close to shifting an effective threshold, which is candidate #6's mechanism and
which §6.3 warns "may buy rate homeostasis rather than horizon". T2 is stated as a
prediction of *failure* because a candidate's own proposer is the wrong party to
give it the benefit of the doubt, and because `EXP_002`'s Q5 — also stated to hurt,
also falsified — is the precedent this project prefers. **A falsified T2 is a real
result and is to be reported as one, not as a vindication.**

## 4. Decision rule, fixed now

The two load-bearing predictions are T1 (does horizon move?) and T2 (does bpc
move?). Every cell of the 2×2 has a named action.

| T1 horizon ≥ 14 | T2 as stated (no >2σ gain) | Verdict and consequence |
|---|---|---|
| **holds** | **falsified** — there *is* a gain | **Adopt**, subject to publishing the §6.2 curve and naming the trade at every context flagged by T3. The two-compartment neuron becomes the Phase-5 backbone, and candidates **#5** (learned decay) and **#6** (adaptive threshold) are re-ranked against *it* rather than against the baseline. Candidate **#14** — re-measure σ on a two-state neuron — is promoted to *blocking*, because every subsequent 2σ verdict would then be resting on a σ measured on a one-state neuron. |
| **holds** | **holds** — no gain | **Do not adopt.** "Horizon is necessary but not sufficient" is confirmed a third time, and for the first time on a *structural* second pole rather than a scalar leak — which is the strongest form of the claim the project can make. The beyond-horizon family (**#5, #6, #10**) is **demoted**, and the 51 % that memory cannot reach takes the top of the Phase-5 matrix: **#2** (current normalisation) and **#4** (readout capacity) are promoted above the remaining neurons. |
| **fails** | **falsified** — gain without horizon | The gain is **not** a memory gain and may not be reported as one. It is re-attributed to per-step capacity — the extra state variable, or the +0.28 % parameters — and adoption waits on a follow-up that separates the two. |
| **fails** | **holds** | **Null.** Recorded prominently as a negative result and retained. The ranking moves to candidate **#2**, and the beyond-horizon family's ROI is re-quoted with this null included in it. |

Two riders, also fixed now:

* **T3 holds** ⇒ the §6.2 criterion will have flagged three of the three arms that
  have ever bought horizon in this project (β, depth K = 4, and this one). It is
  then promoted from a *reporting* requirement to a **ranking input** for Phase 5,
  on the ground that it has never once been uninformative.
* **T4 fails** ⇒ the arm did not test its own hypothesis. `w_c` staying at its
  initialisation means the slow compartment was never used, so the result is
  evidence about **optimisation**, not about two-compartment neurons, and must be
  reported that way rather than as a null on the neuron. The follow-up is an
  initialisation sweep on `w`, not a different neuron.

## 5. Design

| Field | Value |
|---|---|
| Arms | `twocomp_s0`, `twocomp_s1`, `twocomp_s2` — **3 seeds**, the §6.2 adoption minimum |
| Control | the **five existing** `snn_beta0.5_s*` Phase-2 checkpoints. Not retrained; nothing about the baseline is touched |
| Neuron | §2, at the init the §7.2 screen selects |
| Everything else | **byte-identical to the frozen Phase-2 baseline**: enwik8, B = 128, L = 256, d = 512, K = 2, β_f = 0.5, threshold 1.0, hard reset on the fast pole, atan surrogate α = 2, T = 1, fp32 GEMMs, AdamW lr 3e-3 / wd 0.1 / (0.9, 0.95), grad-clip 1.0, 200 warm-up, **20 000 steps**, cosine, fused + CUDA-graph, deterministic |
| Parameters | 735 437 + 2·2·512 = **737 485**, **+0.28 %** — see §6.1 |
| Budget | ~0.8 GPU-hours, matching §6.3's estimate. Three 20 000-step runs at the baseline's measured 399 s, plus the horizon sweep |

### 5.1 What is measured

1. **Test bpc, both protocols, full split** (4 980 736 characters), 3 seeds,
   mean and sample sd. `scripts/evaluate.py` — the same instrument, unmodified.
2. **Memory horizon** per seed, via `scripts/exp/001_memory_horizon.py` — the same
   script, the same paired statistic, the same 2σ tolerance. Its **F1 self-check
   is mandatory**: the `k = L` point must reproduce that run's own independently
   committed `fresh` bpc to better than 1e-3.
3. **The absolute bpc-at-context curve**, c = 0…127, against the baseline, tested
   against the **per-context** 2σ bars — the §6.2 acceptance rule, applied in full.
4. **The learned parameters**: distribution of `w_c` and `beta_s,c` per layer, at
   initialisation and at step 20 000. T4 lives here.
5. Per-layer firing rates, kernels/timestep (must stay < 1.05), peak VRAM,
   wall-clock.

## 6. Known limitations, stated in advance

1. **The arm carries +0.28 % parameters** (2 048 of 737 485). They are per-channel
   scalars *inside* the time loop, not GEMM capacity, and the parameter-scaling
   pilot that would price a 0.28 % change (candidate #11) has not run. **A gain
   smaller than about 0.02 bpc cannot be cleanly separated from the parameter
   increase by this experiment**, and will not be claimed as if it could. Width
   compensation was rejected as the cure: shedding 2 048 parameters means d = 511,
   which changes the width — a second variable — to control for the smaller one.
2. **Nothing here is tuned.** β_s = 0.95, `w` init, and β_f = 0.5 are taken from
   measurements made for other purposes. A null is evidence about *this*
   configuration of a two-compartment neuron, not about two-compartment neurons.
3. **The optimiser hyperparameters are the baseline's**, chosen for a one-state
   neuron and never swept for a two-state one.
4. **σ = 0.00461 was measured on the one-state baseline** (`EXP_000`), and every 2σ
   verdict in this file assumes it transfers to a neuron with a different state
   dimension. §9.1 item 6 of the phase-3 report names that assumption as a ranked
   item (candidate #14). Three seeds give a weak estimate of this arm's own σ; it
   is reported, and it does **not** substitute for #14.
5. **n = 3**, against the baseline's 5. The per-context bars in §6.2 come from the
   baseline's five seeds and are used unchanged.
6. The horizon is measured on `ckpt_final.pt`, matching every other arm in the
   project.

## 7. Failure modes

| # | Failure | Detection |
|---|---|---|
| **J1** | The hand-written backward is silently wrong — risk R10, the project's worst bug class | The full R10 gate, re-run for the new kernel: spikes bit-identical and gradients to rtol 1e-4 / atol 1e-6 against an eager reference **and** against an independent transcription written from §2's equations; an fp64 reference; gradcheck in the linear regime |
| **J2** | A plausible mis-derivation walks through the extended gate | `scripts/audit/08_mutation_campaign.py` extended with two-compartment mutations. **Any escape blocks the run.** |
| **J3** | The kernel computes the wrong function and prices like the right one (`EXP_002` §8.4) | Plain-torch semantics check against the kernel's own equations; the check **aborts** rather than proceeding |
| **J4** | A new parameter is initialised at a gradient saddle (§6.4) | The §7.2 reachability screen. T0 predicts this failure in advance and §2.2 names the fallback in advance |
| **J5** | A training run silently imports a mutated kernel (phase-3 report §8) | The mutation campaign runs to completion, the tree is verified clean, and **only then** does any training process start. No campaign runs while training does. `EXP_001`'s F1 check is run on every checkpoint produced |
| **J6** | This arm is scored with a different instrument from the baseline | The same `scripts/evaluate.py` and the same `scripts/exp/001_memory_horizon.py`, unmodified; F1 asserted per checkpoint |
| **J7** | The slow compartment is silently dead and the null is about nothing | T4, plus `beta_s` and `w_c` recorded at both ends of training, per layer |
| **J8** | CUDA-graph capture silently changes the result, or silently does not happen | The trainer's own capture path is used unchanged; it raises on capture failure rather than falling back |
| **J9** | The baseline path regresses while the new neuron is added | The existing suite is run green before and after; the baseline arm's kernels and tests are not edited |

## 8. What this is not

**This is not a claim that the two-compartment neuron closes the I5 gap.** §5 of the
phase-3 report bounds what it could possibly do: 51 % of the 0.4633 bpc gap is
present at or within the horizon, where a longer memory provably cannot help. A
result that recovered *all* of the beyond-horizon component — which T5 predicts it
will not — would still leave this arm 0.235 bpc behind the GRU anchor. Every number
in the results section is to be quoted against 0.2279 bpc, not against 0.4633.

**It is also not a systems result.** `EXP_002` settled the systems column; the
kernel counts recorded here are a regression check on that finding, not a new
measurement of it.

## 9. Entry conditions

- [ ] Phase 3 signed off, and Phase 4 authorised
- [ ] Kernel semantics check against a plain-torch reference — **passes** (J3)
- [ ] R10 gate green on the new kernel, all four legs (J1)
- [ ] Extended mutation campaign — **every mutation caught**, source tree verifiably restored (J2)
- [ ] §7.2 gradient-reachability screen run; **the init it selects recorded here, before training** (J4)
- [ ] Kernels per timestep < 1.05 at the baseline shape (regression check on `EXP_002`)
- [ ] Existing test suite green — the baseline path unregressed (J9)
- [ ] No other process importing `snn` while the mutation campaign runs (J5)

### 9.1 Entry conditions, resolved — 2026-08-03, before the first training run

| # | Condition | Outcome |
|---|---|---|
| J3 | kernel semantics vs plain torch | **pass** — `v_pre` bit-identical to a twice-rounded reference on 32 768 of 32 768 elements |
| J1 | R10 gate, all legs | **pass** — 25 tests; forward spikes bit-identical, membrane exact, all four gradients within rtol 1e-4 / atol 1e-6 against an eager reference, an independent transcription, and an fp64 reference; gradcheck on the slow pole |
| J2 | mutation campaign | **47 / 47 caught**, source tree restored clean — but **not on the first attempt**; see §9.2 |
| J4 | §7.2 reachability screen | **run**; see below |
| — | kernels per timestep | **1.027** at B=128, L=256, d=512, against the 1.05 bar |
| J9 | existing suite unregressed | **242 passed** (217 before, 25 new) |
| J5 | no other process importing `snn` during the campaign | held — the campaign's own guard, and no training was launched until it finished |

**Init selected by the screen: `w_init = 0.1`, `beta_slow = 0.95` — the
pre-registered fallback.** `docs/reports/data/exp_004_gradient_reachability.json`.

**T0 held, exactly as §2.2 derived it.** At the proposed `w = 0`, the slow decay's
gradient is **identically zero** — `|dL/d raw_beta_s|` RMS is `0.000e+00` in both
layers, on both the fused and the eager path — while the mix itself is not merely
reachable but carries a gradient **5.0× and 7.1× the layer weights'**. The
attractive initialisation, the one that nests the Phase-2 baseline exactly, would
have trained the arm with a frozen slow pole in every seed and said nothing about
it in any log.

| init | `\|dL/dw\|` vs `\|dL/dW\|` | `\|dL/dbeta_s\|` vs `\|dL/dW\|` | verdict |
|---|---:|---:|---|
| `w = 0.0` (proposed, exact nesting) | ×5.02 / ×7.14 | **×0.000 — exactly zero** | **FAIL** |
| `w = 0.1` (pre-registered fallback) | ×3.21 / ×4.80 | ×0.259 / ×0.401 | **PASS** |

This is the first time the §7.2 screen has been run. It was proposed in Phase 3
because §6.4 found the rotational membrane's proposed initialisation to be an
exact saddle *on paper*; it has now caught the same class of defect on a different
candidate, before any GPU time was spent, and the fallback it selected was named
in advance rather than chosen after seeing which one worked.

**Recorded because it is not the candidate's doing:** at initialisation the firing
rate is 0.091 in layer 0 and **0.005 in layer 1** — layer 1 is essentially silent
at step 0. That is the §5 initialisation pathology the Phase-2 baseline already
has (`audit_07_init_pathology.json`) and which `EXP_003` R4 found worsens with
depth, not something introduced here. It is noted now so that it cannot later be
mistaken for a two-compartment effect, and it is a live suspect if the arm
underperforms — candidate #8, variance-scaled initialisation, is the follow-up.

### 9.2 The mutation campaign did not pass on its first run

Reported here rather than absorbed, because a gate that needed fixing is exactly
what the entry conditions exist to surface.

The first extended campaign came back **46 / 47**, with **T05 — "let NVRTC
contract the MIX into an FMA" — escaped.** `M18`, the identical mutation applied
to the Phase-2 LIF's own membrane, has always been caught, so the harness was
working and this candidate's gate had a specific hole.

The mechanism, measured rather than guessed: the contracted mix moves `v_pre` on
**5 866 of 32 768 elements** by up to **4.8e-07**, and flips **zero** spikes. The
mix appears nowhere else — the scan does not return `v_pre`, and the state it does
return, `(vf_after_reset, vs)`, does not depend on the mix at all. So the only
assertion that could have seen the drift was the spike pattern, which changes only
where the membrane lands within an ulp of the threshold. Whether any element lands
there is luck. It came up tails.

The fix asserts the contract where the quantity lives: one step of the real
forward kernel against a plain-torch statement of the same three roundings.
Re-run: **47 / 47**.

The lesson generalises past this kernel, and is worth carrying into every later
candidate: **a numerical contract is only guarded where the quantity it constrains
is actually observed.** An intermediate that no assertion reads is unguarded no
matter how many tests surround it — and the two-compartment neuron introduced
exactly one such intermediate that the single-compartment baseline never had.

---

## 10. Results

**Run 2026-08-03.** Three seeds, 20 000 steps each, run **sequentially** so that no
arm measures contention (the phase-3 report §8's reason for not quoting wall-clock
on the depth ladder). Raw JSON: `docs/reports/data/exp_004_twocomp_results.json`,
`docs/reports/data/exp_004_memory_horizon.json`.

### 10.1 The headline

| Arm | n | Test bpc (fresh) | Test bpc (carried) | **Horizon** |
|---|---:|---:|---:|---:|
| SNN baseline (β=0.5) | 5 | 2.2697 ± 0.0045 | 2.2531 ± 0.0046 | **7** *(7, 7, 7, 7, 7)* |
| **Two-compartment** | **3** | **2.1443 ± 0.0045** | **2.1174 ± 0.0049** | **45** *(38, 47, 45)* |
| GRU anchor *(violates I5)* | 3 | 1.8064 | 1.7674 | **57–60** |

Per-seed carried: **2.1216, 2.1121, 2.1184**.

**−0.1357 bpc against the baseline, 29.4 σ.** It is the best result this project
has produced under I1–I5 by a wide margin — the previous best was `EXP_003`'s
K = 4 at 2.2348 carried, single-seed and never adopted. It closes **29 %** of the
0.4633 bpc I5 gap, and the horizon goes from **7 characters to 45** against the
anchor's 57–60.

### 10.2 The predictions, resolved as written

| # | Prediction | Measured | Verdict |
|---|---|---|---|
| **T0** | `w = 0` is an exact saddle for `beta_s` | `\|dL/d raw_beta_s\|` = 0.000e+00, both layers, both paths | **held** (§9.1) |
| **T1** | median horizon ≥ 14 | **45** (38 / 47 / 45) | **held** |
| **T2** | carried bpc does **not** improve by > 2σ | improved by **0.1357** = 29.4 σ | **FAILED — falsified** |
| **T3** | worse at ≥ 1 short context beyond its own bar | worse at **5** of them (c = 3…7) | **held** |
| **T4** | `\|w\|` ≥ 2× its init in both layers | **4.9×** (layer 0), **3.6×** (layer 1) | **held** |
| **T5** | any improvement < 0.2279 bpc | 0.1357 | **held** |

**T2 was stated in the direction that hurts the candidate, and it was wrong.** §3
said a falsified T2 is a real result and not a vindication, and that is how it is
recorded. The mechanical argument in its favour — that a reset-shielded slow pole
has a clean `beta_s` adjoint with no reset factor, so gradient can reach back over
tens of steps — turned out to be the one that mattered.

**Decision cell (§4): T1 holds ∧ T2 falsified ⇒ ADOPT**, subject to publishing the
§6.2 curve and naming the trade. §10.4 does that, and §10.3 is why the adoption
must be worded carefully.

### 10.3 Where the gain actually came from — the part that changes the plan

Decomposing the gain by context length, on the same basis §5 of the phase-3 report
used (fresh asymptote; negative = the arm is better):

| Component | Gain | Share of the total gain |
|---|---:|---:|
| **At zero context** (c = 0) | **+0.0508** | **+40.6 %** |
| **Within the baseline's 7-character reach** | **−0.0598** | **−47.7 %** |
| **Beyond the horizon** | **+0.1343** | **+107.2 %** |
| Total | **+0.1254** | 100 % |

**The arm did not simply get better. It got substantially better beyond the
horizon, meaningfully better at zero context, and genuinely worse inside the seven
characters it already reached** — and the three nearly cancel to something that,
quoted as a single mean, would look like a uniform improvement.

Against the anchor, the *composition* of the remaining gap has changed more than
its size:

| Component | baseline vs GRU | **twocomp vs GRU** | change |
|---|---:|---:|---|
| At zero context | 0.1193 (26 %) | **0.0685** (20 %) | −43 % |
| Within the 7-char reach | 0.1161 (25 %) | **0.1759** (52 %) | **+52 % — worse** |
| Beyond the horizon | 0.2279 (49 %) | **0.0936** (28 %) | **−59 %** |
| **Total** | **0.4633** | **0.3380** | −27 % |

**The candidate recovered 59 % of the beyond-horizon component — the thing it was
ranked to attack — and made the within-reach component 52 % worse.** §5's
denominator was the right one to have insisted on: quoted against the whole 0.4633
this looks like a 27 % win, and quoted against the 0.2279 it was actually aimed at
it is a 59 % win with a named cost. T5 held, but only in the sense that the ceiling
was not exceeded; a majority of what was available on the targeted component was
in fact captured.

### 10.4 The §6.2 criterion, applied in full

The arm is worse than the baseline at **5 of 128 context lengths**, and they are
**contiguous — c = 3, 4, 5, 6, 7** — which is exactly the region inside the
baseline's own reach.

| c | Δ bpc vs baseline | that context's 2σ bar | multiple of the bar |
|---:|---:|---:|---:|
| 3 | **+0.0511** | 0.0179 | 2.9× |
| **4** | **+0.0560** | 0.0086 | **6.5×** |
| 5 | **+0.0392** | 0.0052 | 7.6× |
| 6 | **+0.0225** | 0.0060 | 3.8× |
| 7 | **+0.0090** | 0.0027 | 3.4× |

**The trade, named as §6.2 requires:** *this arm buys 0.134 bpc beyond seven
characters of context by giving up 0.060 bpc inside them, concentrated entirely at
three to seven characters, worst at c = 4 where it is 6.5× that context's own noise
bar.* It may not be reported as a uniform gain.

| Arm | contexts worse | worst short-context regression |
|---|---:|---:|
| GRU anchor | 0 / 128 | −0.0248 |
| **Two-compartment** | **5 / 128** | **+0.0560 at c = 4** |
| Depth K = 4 | 5 / 128 | +0.0764 at c = 3 |
| SNN β=0.9 | 126 / 128 | +0.2498 at c = 4 |

The signature is the **same shape as depth K = 4's and milder in magnitude**, and
nothing like the β arms'. Per §4's rider, the criterion has now flagged **three of
the three** arms this project has produced that lengthen the horizon, and has never
once been uninformative: **it is promoted from a reporting requirement to a ranking
input for Phase 5.**

### 10.5 What the network did with the freedom, which was not what was expected

`beta_s` was initialised at 0.95 for every channel. It did not stay there, and it
did not move uniformly — the population went **bimodal**:

| | layer 0 | layer 1 |
|---|---:|---:|
| `beta_s` mean | 0.578 | 0.592 |
| `beta_s` sd | 0.117 | 0.132 |
| `beta_s` max | 0.986 | 0.973 |
| channels with `beta_s` > 0.9 | **2.8 %** (13–16 of 512) | **2.9 %** (12–19 of 512) |
| their mean time constant | **26 chars** | **18 chars** |
| `\|w\|` mean | 0.49 | 0.35 |

**About 3 % of channels kept a long time constant; the other 97 % pulled their
"slow" pole down to ~0.5–0.6 — essentially a second fast compartment.** All three
seeds agree on this to within a percent.

If those ~15 channels per layer are what carries the horizon from 7 to 45, then the
capacity this project is devoting to long memory is **3 % of its width**, and is
nowhere near saturated. That is the single most actionable thing in this
experiment.

**It is not established here.** The association is correlational: this experiment
did not ablate the slow tail, and `|w|` on the tail channels is not meaningfully
larger than on the rest (0.99× in layer 0, 1.16× in layer 1), which is *not* what
one would naively expect if those channels were carrying the load. The causal
question — zero the tail's mix and re-measure the horizon — is a Phase-5 experiment
and must be pre-registered like anything else.

### 10.6 Forty per cent of the gain is provably not a memory effect

At zero context both compartments start from zero, so at the first character

```
v_0 = vf_0 + w_c * vs_0 = cur_0 + w_c * cur_0 = cur_0 * (1 + w_c)
```

and the neuron fires iff `cur_0 >= thr / (1 + w_c)`. **At c = 0 the
two-compartment neuron is exactly the Phase-2 baseline with a learned per-channel
threshold.** That is algebra, not inference, and it explains the +0.0508 at zero
context — 40.6 % of the total gain — with no reference to memory at all.

Measured: the learned effective threshold is **thr × 0.69** in layer 0 and
**thr × 0.79** in layer 1 (10th–90th percentile 0.56–0.83 and 0.57–1.05). The
network lowered its firing threshold, per channel, by about a quarter.

The consequence for Phase 5 is concrete and cheap: **a learned per-channel
threshold is one parameter, no second state variable, no new kernel, and no
gradient gate**, and it is the natural candidate to capture that 40 % on its own.
It is close to candidate **#6** (adaptive threshold), which §6.3 warned "may buy
rate homeostasis rather than horizon — measure, do not assume". This result says
the static half of that is worth having regardless.

The **+0.28 % parameter increase** (§6.1) is the other live explanation for a
zero-context gain, and this experiment cannot separate the two: 2 048 of the extra
parameters and the per-channel threshold are the same 2 048 parameters. A
one-parameter threshold arm would separate them, which is a further reason to run
it.

### 10.7 σ appears to transfer, on a weak estimate

§6.4 flagged that every 2σ verdict here assumes a σ measured on a *one-state*
neuron transfers to a two-state one, and §4 promotes candidate #14 to blocking on
adoption. The three seeds give a first, weak reading:

| | fresh sd | carried sd |
|---|---:|---:|
| one-state baseline, n = 5 | 0.00450 | 0.00461 |
| **two-compartment, n = 3** | **0.00445** | **0.00487** |

Essentially unchanged. **This does not close candidate #14** — n = 3 estimates a
standard deviation to about ±40 %, and the pre-registration said in advance it
would not substitute. It does mean the verdicts above are not obviously resting on
a broken bar.

### 10.8 Systems: the realised cost, against what `EXP_002` priced

| | baseline | two-compartment | ratio |
|---|---:|---:|---:|
| Parameters | 735 437 | 737 485 | +0.28 % |
| Kernels per timestep (forward) | 1.016 | **1.027** | 1.01× |
| Wall-clock, 20 000 steps | 398.9 s | **525.3 s** (537 / 529 / 510) | **1.32×** |
| Peak VRAM | 0.609 GiB | **0.980 GiB** | 1.61× |

`EXP_002` priced N2's *forward* at 1.08× and said plainly (§5.1) that "a prototype
kernel is not a trained neuron". The realised end-to-end training step is **1.32×**,
and the gap is the backward: the deferred reduction that keeps the one-kernel floor
costs three `[B, L, d]` stacks per layer instead of one, which is also the 1.61×
memory. **`EXP_002`'s conclusion survives — the systems column still does not
discriminate** (2 minutes per run, 0.37 GiB on a 7.96 GiB card) — but its forward-only
1.08× should not be quoted as the cost of a trained two-state neuron. 1.32× should.

### 10.9 Reproduction checks

Everything this experiment re-measured came back unchanged:

* **F1 passed on 11 of 11 checkpoints**, residuals 4.3e-10 to 1.3e-08 bpc. This is
  the check that caught a contaminated run in Phase 3.
* **All five baseline seeds returned horizon 7 again**, and the GRU 57 / 58 / 60 —
  identical to `EXP_001`'s committed values, from a re-run of the same script.
* The five committed Phase-2 test scores reproduce the report's quoted 2.2697 /
  2.2531 means; the results script aborts if they do not.
* No mutation campaign ran while any training process was alive (J5).

### 10.10 Status

**CLOSED. T1, T3, T4, T5 held; T2 was falsified; T0 held and changed the
initialisation before any GPU time was spent.**

**Verdict: ADOPT**, with the §10.4 trade named and published, and with two
qualifications that belong in the adoption rather than beside it:

1. **It is not a uniform improvement.** 40.6 % of the gain is at zero context and
   has an exact one-parameter explanation (§10.6); 47.7 % of the gain's magnitude
   is given back inside the baseline's own reach (§10.3).
2. **Candidate #14 is now blocking** for any further 2σ verdict in this family,
   though §10.7 suggests it will pass.

### 10.11 What Phase 5 inherits

1. **The two-compartment neuron is the new backbone**, at 2.1174 carried. Candidates
   **#5** (learned per-channel decay) and **#6** (adaptive threshold) are re-ranked
   against *it*, not against 2.2531 — and #5 is now partly subsumed, since this arm
   already learns a per-channel decay on one of its two poles.
2. **A learned per-channel threshold is the cheapest unclaimed win on the board.**
   §10.6 shows it accounts for 40 % of this arm's gain, at one parameter and no new
   kernel. It should be run against the baseline *and* against this arm, because in
   this arm it is already present and would not add twice.
3. **The within-reach component is now the largest single piece of the remaining gap
   at 52 %**, up from 25 %. The ranking that put horizon first was right for Phase 4
   and is wrong for Phase 5: **short-range fidelity is where the gap now lives.**
4. **~3 % of channels appear to carry the horizon** (§10.5), which if true means the
   long-memory capacity is far from saturated. The ablation that would establish it
   is cheap and is the highest-information single run available.
5. **The §6.2 criterion is promoted to a ranking input.** Three of three
   horizon-buying arms have been flagged by it; it has never been uninformative.
6. **`EXP_002`'s systems numbers are forward-only.** Quote 1.32× and 1.61 GiB for a
   trained two-state neuron, not 1.08× and "identical memory".

---

## 11. Addendum, 2026-08-03 — what later experiments did to §10.11

Appended rather than backdated, and the text above is left exactly as it was
written. `03_phase3_candidates.md` §7.2's precedent: a correction goes in as an
addendum, with the original left standing, so that what was believed *before* the
evidence stays legible.

| §10.11 item | Status | Source |
|---|---|---|
| 1. two-compartment is the backbone at 2.1174 | **Holds, number updated to 2.11869** (n=7, was n=3) | `EXP_005` §9 |
| 2. learned per-channel threshold is the cheapest unclaimed win | **Run as `EXP_008`**, and §10.6's framing is sharpened: the arm is a *reparameterisation* of the baseline, so its 1 024 parameters add no functions and neither of §10.6's "two live explanations" is capacity | `EXP_008` §1.1 |
| 3. within-reach is the largest piece, at 52 % | **Holds; 51.8 % at n=7.** Attacked for the first time by `EXP_007` | `EXP_005` §9.6 |
| 4. ~3 % of channels carry the horizon, so capacity is "far from saturated" | **First half established causally. Second half NOT established, and `EXP_006` argues against it**: `beta_s` started at 0.95 on all 512 channels and training pulled 97 % down, so the tail's size is a learned outcome, not a ceiling | `EXP_006` §9.1–9.2, §9.6 |
| 5. §6.2 promoted to a ranking input | Holds, and `EXP_005` found the bar itself is built on the wrong scale. **The redefinition is referred to Elliot and not applied** | `EXP_005` §9.4 |
| 6. systems numbers are forward-only | Holds | — |

**Item 4 is the one that changed direction**, and it is the item the phrase "far
from saturated" was doing the most work in. Do not rank "widen the slow tail" on
§10.5 or on `EXP_006`'s necessity result; the experiment that would justify it is
the frozen-`beta_s` ladder named in `EXP_006` §9.6, at ~1.3 GPU-hours.

**§10.6's own algebra is unchanged and is now load-bearing in a second place.**
`EXP_006` §1.1 used it to choose `beta_s` over `w` as the thing to ablate — an
ablation on `w` would have moved the timescale and the learned threshold together
— and `EXP_008` §1.1 uses it as the starting point for Identity 1.
