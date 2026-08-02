# Phase 3 — Candidate Improvements

**Project:** Character-level spiking neural-network language model
**Author:** Elliot Asher Caudill
**Date:** 2026-08-02
**Status:** Phase-3 deliverable. Awaiting review before Phase 4 begins.
**Branch:** `phase-3-candidates`

---

## 0. Executive summary

Phase 2 ended with a number and no mechanism: **invariant I5 costs 0.486 bpc**,
and binarity costs nothing. Phase 3's job is to turn that into a ranked list of
things to try, and a ranking needs a mechanism — otherwise it is a list of
plausible ideas sorted by taste.

So Phase 3 measured the mechanism before ranking anything. Three experiments, all
pre-registered, all under one GPU-hour:

* **`EXP_001` — the memory horizon.** The spiking baseline's predictions stop
  improving after **7 characters** of context. The GRU anchor's keep improving to
  **57–60**. That 8× is what the 0.486 bpc is made of, and it is now measured
  rather than inferred. All five baseline seeds return exactly 7.
* **`EXP_002` — what the admitted neurons cost.** Every neuron the §4.6 ruling
  admits — learned per-channel decay, a two-timescale membrane, an adaptive
  threshold, a rotational state — still issues **one CUDA kernel per timestep**,
  at **identical peak memory** and ≤1.10× wall-clock. The systems column of the
  ROI matrix does not discriminate between them.
* **`EXP_003` — does horizon compose with depth?** Because if it did, the cheapest
  route to the GRU's horizon would be a config flag rather than a new kernel.

The finding that most changes the plan was not predicted by any of them.

> **Raising β buys horizon and loses bits.** β = 0.5 → 0.9 → 0.95 lengthens the
> measured horizon 7 → 24 → 35 characters, and test bpc degrades monotonically
> over exactly that range. Horizon is **necessary but not sufficient**, and
> "buy more horizon" is not a research direction — it is the direction Phase 2's
> β sweep already falsified.

What separates the good and bad ways to buy horizon is visible once the right
statistic is plotted. Against the baseline, at each context length:

| Arm | c=0 | c=2 | c=4 | c=8 | c=32 | Worst point |
|---|---:|---:|---:|---:|---:|---:|
| SNN β=0.9 | −0.025 | **+0.124** | **+0.250** | **+0.159** | +0.063 | **+0.250** |
| SNN β=0.95 | +0.006 | **+0.130** | **+0.301** | **+0.222** | +0.108 | **+0.301** |
| Analogue control | −0.036 | −0.122 | −0.051 | +0.009 | +0.011 | +0.011 |
| **GRU anchor** | **−0.119** | **−0.090** | **−0.065** | **−0.271** | **−0.436** | **−0.025** |

The GRU is **nowhere worse than the baseline at any context length**. The β arms
buy their extra reach by degrading every context length from two characters up —
a scalar leak lengthens memory by *blurring* it, because the one state variable
holding the distant past is the same one that has to hold the present character.

That yields the Phase-4 acceptance criterion in §6.2, which is a measurement and
not a preference: **any candidate that improves mean bpc must also publish its
bpc-at-context curve, and name the trade at every context where it is worse than
the baseline by more than that context's own 2σ bar.** The GRU is worse at 0 of
128 contexts; β=0.9 at 126 of 128, worst by 29× the local bar. It is the guard
that stops Phase 4 from reporting a mean gain bought by damaging short-range
prediction as though it were a uniform gain — and it already flags the otherwise
most attractive arm this phase produced (§4).

The criterion needed calibrating before it could be used at all: **σ = 0.00461 is
the whole-split standard deviation, and the per-context bar ranges from 0.0284 at
one character of context down to 0.0003 beyond sixteen** (§6.2). A flat bar would
have been three times too strict where the criterion matters most and thirty times
too lax elsewhere.

And the mechanism story has a hard ceiling that no candidate proposed here
acknowledged. Decomposing the gap by context length (§5): **26 % of it is present
at zero context, where memory cannot matter, and another 25 % lies inside the
seven characters the model already reaches. Only 49 % is addressable by longer
memory at all.**

Two further Phase-3 outputs are engineering rather than science, and both pay down
costs `EXP_002` identified:

* **A committed mutation-testing harness** (`scripts/audit/08_mutation_campaign.py`).
  `EXP_002` found the real price of a new neuron is not GPU time but the
  mutation-tested gradient gate it needs. That gate's own provenance turned out to
  be prose: the campaign behind `02_baseline_report.md` §2.1 is quoted as **21**
  mutations there, **sixteen** in `src/snn/neuron.py`, and **eleven** in
  `tests/test_neuron_equivalence.py`, with no artifact to check any of them
  against. It is now a script with committed machine-readable output, and adding a
  candidate neuron's mutations is one list entry rather than a new campaign.
* **A reusable horizon probe** (`scripts/exp/001_memory_horizon.py`), which
  reproduces every committed Phase-2 `fresh` bpc to between 4.4e-10 and 1.3e-08
  and can be pointed at any future checkpoint.

---

## 1. What Phase 3 inherits, and what it re-measured

`02_baseline_report.md` §8.1 handed over four findings. Phase 3 adds a fifth and
promotes one to a ranking criterion.

| # | Inherited from Phase 2 | Status after Phase 3 |
|---|---|---|
| 1 | Binarity is nearly free; I5 costs 0.486 bpc | **Confirmed and explained.** The analogue control's horizon is 6 against the spiking arm's 7 — binarity does not shorten memory either (`EXP_001` §8.4) |
| 2 | Structure, not magnitude: raising β hurts monotonically | **Promoted to a quantitative criterion.** §6.2 |
| 3 | Fusion and CUDA graphs are one lever, not two | Unchanged; it prices §3's systems column |
| 4 | σ = 0.0046 bpc resolves 0.01 bpc effects at 3 seeds | **Qualified.** It is the whole-split σ; the per-context bar it is used as in §6.2 had to be measured separately and differs by up to 30× |
| 5 | *(new)* The I5 cost is a horizon cost — 7 characters against 57–60 | `EXP_001`, and **bounded** by §5: only 49 % of the gap is beyond the horizon |

### 1.1 Numbers from Phase 1 that Phase 3 must not reuse

Recorded because they are still in `01_reconnaissance.md` and are still wrong:

* **14.5 µs per kernel.** The real fused scan costs **32.8 µs** (§6.2 of the
  Phase-2 report). Every ROI figure in this document uses 32.8.
* **~51× from fusion and CUDA graphs combined.** They are the same lever; 92 % of
  the launches are already gone (§6.1). No candidate here budgets a multiplier.

---

## 2. `EXP_001` — the memory horizon, measured

Pre-registration and full results: `experiments/logs/EXP_001_memory_horizon.md`.
Raw JSON: `docs/reports/data/exp_001_memory_horizon.json`.

### 2.1 The probe, and why it is trustworthy

Every arm's only cross-time carrier is its recurrent state, and all three arms
zero it when called with `state=None`. So forcing the state to zero every `k`
characters needs no hook and no model edit — it is the reshape
`x [B, L] → x [B·L/k, k]`. That was chosen over a state-zeroing hook for a
specific reason: **a hook would be new code sitting between the model and the
number, and a bug in it would look exactly like a short memory horizon.** Here the
only thing exercised is the model's own `state=None` branch, which every Phase-2
`fresh` evaluation already used.

At `k = L` the reshape is the identity, so the probe must reproduce the committed
Phase-2 `fresh` bpc. It does, on **11 of 11 checkpoints**, to between **4.4e-10
and 1.3e-08 bpc**. Every arm scored exactly 4 980 736 characters at every `k`.

### 2.2 A confound found in the pre-registered design, and how it was handled

The pre-registered statistic — mean bpc under truncation at `k` — is
**confounded**, and this was found during the first smoke run, before the full
sweep. It averages over every position in a chunk, so a fraction `1/k` of scored
characters sit at position 0 with no context at all. A model with a *short*
horizon but expensive first characters still traces a smooth curve improving all
the way to `k = L`, which looks exactly like long memory.

That is testable rather than merely arguable. If the whole penalty is boundary
amortisation then, with no free parameter,

```
bpc(k) − bpc(L)  =  (L/k − 1)/L  ·  Σ_{c<k} excess(c)
```

Reconstructing the pre-registered curve from the corrected one this way reproduces
it to a maximum residual of **0.0072–0.0282 bpc** against penalties of 0.47–0.83
bpc: **97–99 % of the pre-registered signal is boundary cost, not memory.**

The corrected statistic is a **paired** one. For a fixed position `p`, the model,
the window and the target character are identical between the truncated and
untruncated runs; only the context differs. Text difficulty — which dominates the
~0.02 bpc per-position standard error of the naive alternative — cancels in the
difference.

**The correction was made after seeing data.** It is labelled post-hoc in the
experiment log, the pre-registered predictions are resolved against the statistic
they were written for, and both curves are reported. As written, **P1 failed** and
**P4's GRU half failed**; P1's substance holds under the corrected statistic.

### 2.3 The horizons

The shortest context beyond which more context is worth less than 2σ = 0.00922 bpc.

| Arm | n | **Horizon (chars)** | Test bpc (fresh) |
|---|---:|---:|---:|
| Analogue control (β=0.5) | 1 | **6** | 2.2806 |
| **SNN baseline (β=0.5)** | 5 | **7** — *7, 7, 7, 7, 7* | 2.2697 |
| SNN β=0.9 | 1 | **24** | 2.3276 |
| SNN β=0.95 | 1 | **35** | 2.3702 |
| **GRU anchor** *(violates I5)* | 3 | **57–60** | 1.8044 |

Five seeds of the baseline return the identical integer. Whatever else is noisy
about this project, the horizon is not.

### 2.4 The result that reshapes the ranking

**Raising β lengthens the horizon and costs bits.** 7 → 24 → 35 characters, and
2.2697 → 2.3276 → 2.3702 bpc, monotonically, over the same range.

This is not a contradiction, and resolving it is the most useful thing Phase 3
produced. A scalar leak buys reach by **low-pass filtering**: the same state
variable that holds the distant past also has to hold the present character, and
at β = 0.95 the present character is a twentieth of the signal. The GRU is not
merely *longer-reaching* than the spiking baseline — it is longer-reaching **while
keeping recent context sharp**, because its gate lets each unit decide, per step,
whether to retain or overwrite.

The absolute bpc-at-context curves make the distinction visible. Against the
baseline (negative = better at that context length):

| Arm | c=0 | c=1 | c=2 | c=3 | c=4 | c=8 | c=16 | c=32 | c=64 | **Worst** |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SNN β=0.9 | −0.025 | +0.003 | +0.124 | +0.233 | +0.250 | +0.159 | +0.084 | +0.063 | +0.058 | **+0.250** |
| SNN β=0.95 | +0.006 | −0.005 | +0.130 | +0.245 | +0.301 | +0.222 | +0.146 | +0.108 | +0.103 | **+0.301** |
| Analogue | −0.036 | −0.039 | −0.122 | −0.099 | −0.051 | +0.009 | +0.011 | +0.011 | +0.011 | +0.011 |
| **GRU** | −0.119 | −0.204 | −0.090 | −0.025 | −0.065 | −0.271 | −0.385 | −0.436 | −0.459 | **−0.025** |

**The GRU is nowhere worse than the baseline.** The β arms are worse everywhere
from two characters onward. That is the difference between the two ways of buying
horizon, and §6.2 turns it into an adoption rule.

### 2.5 Two observations that were not asked for

* **The analogue control's horizon is 6 against the spiking arm's 7.** Binarity
  and the hard threshold do not shorten memory. That is an independent
  confirmation, on a different axis, of §7.2's finding that binarity is nearly
  free: the arms differ in emission, not in reach.
* **The GRU loses *more* than the SNN when starved of context** (+2.238 vs +1.892
  bpc at zero context, relative to each arm's own asymptote). It is not uniformly
  better at every context length in relative terms — it has more invested in
  context, so it has more to lose. Below three characters the two arms are far
  closer than the 0.486 bpc headline suggests.

---

## 3. `EXP_002` — the admitted neurons are systems-free

Pre-registration and full results:
`experiments/logs/EXP_002_candidate_neuron_cost.md`. Raw JSON:
`docs/reports/data/exp_002_candidate_neuron_cost.json`.

Five neurons, prototyped as jiterator kernels at the frozen baseline shape
(B=128, L=256, d=512). The baseline N0 reproduces the committed 1.016
kernels/timestep at 1.0117 — same instrument, same answer.

| id | Neuron | State | Per-channel params | in/out | Kernels/timestep | vs N0 |
|---|---|---:|---:|:---:|---:|---:|
| N0 | LIF (Phase-2 baseline) | 1 | 0 | 2/3 | 1.0117 | 1.00 |
| N1 | Learned per-channel decay | 1 | 1 | 3/3 | **1.0117** | **0.97** |
| N2 | Two-timescale membrane | 2 | 1 | 4/4 | **1.0156** | **1.08** |
| N3 | Adaptive threshold (adLIF) | 2 | 1 | 4/4 | **1.0156** | **1.10** |
| N4 | Rotational (complex) membrane | 2 | 2 | 5/4 | **1.0156** | **1.07** |

Peak allocation is **0.188 GiB for every row, identical to four decimal places** —
a `[1, d]` per-channel parameter broadcasts against `[B, d]` inside the kernel with
no extra kernel and no materialisation. The widest candidate uses 5 of 8 available
inputs, so jiterator's arity limit is not close to binding.

**Consequence for the ROI matrix: the systems column is not a discriminator.** At
32.8 µs/kernel the worst candidate costs ~0.03 GPU-hours on a 3-seed arm, three
orders below the measurement's own resolution. N1–N4 are to be ranked on expected
bits-per-character alone.

### 3.1 The prediction that was stated to hurt, and was falsified

`Q5` predicted that a learned per-channel decay's gradient must break the
one-kernel floor, since a per-channel gradient is a reduction over batch and time
and reductions are permanently unavailable here (§2.3). The reduction is indeed
unavoidable; **paying for it with an extra kernel is not.**

| Strategy | Kernels/timestep | Peak VRAM |
|---|---:|---:|
| Accumulate `grad_beta` inside the loop | 2.0195 | 0.376 GiB |
| Emit per-step contributions, reduce **once** at the end | **1.0234** | 0.522 GiB |

The floor is recoverable for **+0.146 GiB**, which against a 0.61 GiB baseline run
on a 7.96 GiB card is not a constraint. **Any learnable per-neuron parameter can
keep one kernel per timestep on both passes.**

### 3.2 A kernel that compiled, ran, and computed the wrong function

The first draft of N2 declared `(vf_prev, vs_prev, cur, beta_f, beta_s, w_c, thr)`
— scalars interleaved before a tensor. jiterator binds all tensor arguments first
and then scalars, so `w_c` was silently bound into the `beta_f` slot. It compiled,
ran, produced sensible spikes, and **priced within 2 % of the corrected kernel**,
because a kernel's cost depends on its shape and not on whether its arithmetic is
right.

No timing harness can catch that. It was caught by checking each kernel against a
plain-torch statement of its own equations, which found `v_pre` off by 0.99 and
spikes not bit-identical. That check is now part of the script, its per-candidate
output is committed, and the run **aborts** rather than pricing a candidate that
fails it.

This is the `EXP_002`-scale version of risk R10, and it is worth stating plainly:
**a wrong candidate benchmarks like a right one.** Any Phase-4 arm built on a new
kernel needs its gradient gate before its number means anything.

---

## 4. `EXP_003` — depth does not buy the horizon

Pre-registration and full results: `experiments/logs/EXP_003_depth_horizon.md`.

If horizon composed across layers, the cheapest route to the GRU's reach would be
a config flag rather than a new kernel with a new gradient gate — so the ranking
could not be written until this was known. Four parameter-matched depths, widths
derived to hold the count at 735 K within 0.1 %:

| K | d | **Horizon** | Test bpc (fresh) | Test bpc (carried) |
|---:|---:|---:|---:|---:|
| 1 | 676 | **6** | 2.5084 | 2.4952 |
| **2** | **512** | **7** *(5 seeds, all 7)* | **2.2697** | **2.2531** |
| 4 | 380 | **9** | **2.2528** | **2.2348** |
| 8 | 278 | **14** | 2.4194 | 2.3995 |

**Horizon grows as √K, not K.** From K=2, `7·√(K/2)` predicts 4.9 / 7 / 9.9 / 14
against measured 6 / 7 / 9 / 14. That is cascaded leaky integration — the sum of K
exponential delays has mean K·τ but spread √K·τ, and for a thresholded channel the
spread sets usable reach. Depth diffuses one horizon; it does not chain many.

Matching the anchor at √K scaling needs **K ≈ 137 layers**, at d ≈ 70. The
pre-registered prediction R1 (`horizon(K=4) ∈ [14,28]`, `horizon(K=8) ∈ [28,56]`)
**failed on both counts** — measured 9 and 14.

**So depth is not a route to the horizon, and neuron-state candidates do not have
to justify themselves against it.** That is the ranking answer §6 needed.

Two by-products matter more than the headline:

* **K = 4 is the best arm this project has produced under I1–I5** — 2.2348 carried
  against 2.2531, a 4σ improvement. It is n = 1, it is **not adopted**, and it is
  the first candidate caught by the §6.2 criterion: worse than the baseline at 5 of
  128 context lengths, all short, worst **+0.0764 bpc at c = 3** against that
  context's 2σ bar of 0.0179. A milder β signature. It goes to Phase 4 as a proper
  3-seed arm with that trade named in advance.
* **The initialisation pathology is far worse at depth.** At K = 8, six of eight
  layers are silent and **nothing wakes until step ~130** (against ~25–50 at K=2),
  with the loss almost flat until it does. Per `EXP_003`'s decision rule this
  **promotes the variance-scaled initialisation from a low-ROI curiosity to a
  prerequisite for any deep candidate** — and leaves K=8's deficit genuinely
  ambiguous between depth and initialisation, as §5.2 of that log said in advance
  it would.

---

## 5. Where the gap actually is — the denominator every candidate is ranked against

The single most important number in this phase is not a horizon. It is that
**about half the I5 gap is not a memory problem at all.**

Splitting the SNN-versus-GRU gap by context length, from the committed
`exp_001_memory_horizon.json`:

| Component | bpc | Share | What could recover it |
|---|---:|---:|---|
| **At zero context** (c = 0) | **0.1193** | **26 %** | Nothing about memory. Per-step capacity: the readout, normalisation, width, scale |
| **Within the 7-character reach** | **0.1161** | **25 %** | Context the model *already has* and extracts less from. Per-step expressivity |
| **Beyond the horizon** | **0.2279** | **49 %** | The horizon candidates' entire ceiling |
| **Total asymptotic gap** | **0.4633** | 100 % | |

**51 % of the target is unreachable by every neuron-memory candidate in §6**, and
no candidate proposed in this phase quoted a ceiling that acknowledged it. A
two-compartment neuron that recovered *all* of the beyond-horizon component — a
wildly optimistic outcome — would still leave the SNN 0.235 bpc behind the anchor.

That reframes the phase plan. The §4.6-admitted neurons remain the scientifically
central bet, because I5 is the project's research question and neuron-state memory
is what I5 forces. But the ROI matrix must carry this denominator explicitly, and
Phase 4 needs at least one arm attacking the zero-context component, which is
**26 % of the gap and present where memory provably cannot matter**.

The two cheapest such arms are named in §6.3, and neither existed in the candidate
set until a completeness audit asked what was missing:

* **The baseline has no normalisation anywhere.** `01_reconnaissance.md` §4.3
  item 2 records that competitive spiking LMs keep "embeddings, normalisation, and
  the output head in floating point". This project kept two of the three. That is
  an unremarked deviation from the only comparable published system, it is a
  plausible root cause of the §5 initialisation pathology, and it operates outside
  the time loop — a handful of kernels per layer, not per timestep, no jiterator,
  no new backward.
* **The readout is a single `nn.Linear(512, 205)` over one binary vector.** The
  whole per-step function is a two-layer binary MLP with no skip connection, while
  a GRU at zero state computes a continuous gated function. Enlarging the
  non-spiking head touches no invariant — spec §8 and §4.3 item 2 place the head
  explicitly outside the spiking path — and adds no kernel, no backward, and no
  R10 exposure.

---

## 6. The candidate set

18 candidates were generated across the six areas the Phase-1 plan names, each
then adversarially verified against the §4.6 ruling, jiterator's elementwise-only
and 8-input limits, and the corrected cost constants. A completeness audit then
checked the whole set. One candidate was refuted outright, two pairs turned out to
be the same lever under different names, and eight candidates were identified as
missing.

### 6.1 Two pairs that are the same lever

This is the mistake Phase 2 caught with fusion and CUDA-graph capture, repeated:

* **The two-compartment neuron was proposed twice** — once as a fast/slow membrane
  pair, once as a "reset-partitioned" neuron — and they are **algebraically
  identical**, not merely similar. Expanding the slow compartment,
  `w·vs_t = w·β_s·vs_{t−1} + w·cur_t`, so one is the other with `g = w·β_s` plus a
  scalar rescaling of the current that the layer's own `Linear` absorbs. Both
  compile to `EXP_002`'s N2 with the same kernel signature. **They are one arm and
  are budgeted once.**
* **The rotational membrane was proposed twice**, in the multi-timescale and
  adaptive-threshold families, with the same per-channel (radius, angle), the same
  2×2 rotation, and the same "fire on the real part". **One arm.**

### 6.2 The acceptance criterion, calibrated

`EXP_001` §8.4 pre-registered that a candidate must lengthen horizon *without*
degrading near-context fidelity. Turning that into a usable rule required one more
measurement, because the obvious bar is wrong.

**σ = 0.00461 bpc is the whole-split standard deviation, not the per-context one.**
Measured across the five baseline seeds, the per-context 2σ bar is:

| c | 0 | 1 | 2 | 3 | 4 | 8 | 16 | 32 | 64 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **2σ** | 0.0127 | **0.0284** | 0.0153 | 0.0179 | 0.0086 | 0.0020 | **0.0003** | 0.0010 | 0.0003 |

A flat 0.00922 bar would be **3× too strict at c = 1** — where the position-sample
count is smallest — and **30× too lax at c ≥ 16**. Applied without this
calibration it would have rejected candidates on noise at exactly the short
contexts it exists to protect.

> **Phase-4 acceptance rule.** A candidate is adopted only if its mean test bpc
> improves by more than 2σ = 0.00922 over ≥3 seeds. **In addition**, its absolute
> bpc-at-context curve is published, and any context where it is worse than the
> baseline by more than *that context's own* 2σ bar is reported with the trade
> named. A mean gain bought by short-range degradation is a different scientific
> claim from a uniform gain, and may not be reported as the same thing.

This is a reporting requirement, not a veto — the note stands, the arm may still
be adopted, but it cannot be adopted silently. Against it:

| Arm | Contexts significantly worse | Worst short-context regression |
|---|---:|---:|
| GRU anchor | **0 / 128** | −0.0248 |
| SNN β=0.9 | 126 / 128 | +0.2498 at c = 4 |
| SNN β=0.95 | 126 / 128 | +0.3014 at c = 4 |
| Depth K=4 | **5 / 128** | +0.0764 at c = 3 |
| Depth K=8 | 128 / 128 | +0.3162 at c = 3 |

### 6.3 The ranked shortlist

Ranked by expected bpc per GPU-hour against the §5 denominator, with the systems
column omitted because `EXP_002` showed it does not discriminate. **The real cost
of every new-neuron row is its mutation-tested gradient gate**, now much cheaper
than it was (§7.1).

| # | Candidate | Attacks | Params/neuron | I5 | GPU-h (3 seeds) | Note |
|---:|---|---|---:|---|---:|---|
| **1** | **Two-compartment neuron** (fast/slow, slow pole reset-shielded) | beyond-horizon (49 %) | 2 | admitted, §4.6 row 2 | 0.8 | Nests the baseline exactly at w=0, so the downside is structural, not hoped |
| **2** | **Current normalisation** (RMSNorm on `cur`, outside the time loop) | zero-context + within-reach (51 %) | 0 (2d/layer) | neutral | 0.4 | Closes a deviation from the baseline's own literature; subsumes the init candidate |
| **3** | **Depth K = 4 at 3 seeds** | mixed; already 4σ at n=1 | 0 | admitted | 0.4 | Carries a named short-context trade (§6.2) |
| **4** | **Readout capacity** (deeper non-spiking head, width-compensated) | zero-context (26 %) | 0 | untouched — head is outside the spiking path | 0.4 | The only arm attacking the component memory cannot reach |
| **5** | **Learned per-channel decay** | beyond-horizon | 1 | admitted, §4.6 row 1 | 0.7 | `EXP_002` Q5: keeps the 1-kernel floor for +0.146 GiB |
| **6** | **Adaptive threshold** (spike-driven, own decay) | beyond-horizon | 2 | admitted, §4.6 row 3 | 0.8 | May buy rate homeostasis rather than horizon — measure, do not assume |
| **7** | **Reset ablation** (hard / soft / detached / none) | mechanism; unlocks the parallel form | 0 | free parameter, §1.3 | 0.5 | Already implemented and tested; the cheapest real experiment available |
| **8** | **Variance-scaled initialisation** | depth enabler | 0 | neutral | 0.4 | Promoted by `EXP_003` R4; prerequisite for any deep arm |
| **9** | **Firing-rate set-point** (rate regulariser + threshold) | horizon *and* the energy claim | 0 | neutral | 0.5 | Under hard reset retention is β(1−p), so p is a horizon lever nothing else touches |
| **10** | **Rotational / complex membrane** | beyond-horizon | 2 | admitted, §4.6 row 4 | 0.8 | **See §6.4 — its proposed init is a provable dead end** |
| **11** | **Parameter-scaling pilot** (0.735M / 2.5M / 5M, 1 seed) | §4.5 item 1, unmeasured | — | — | 0.35 | Tells Phase 5 what size to run; bounds every other ceiling |
| **12** | **Surrogate width sweep** (α) | training dynamics | 0 | neutral | 0.5 | §4.3 item 3 says sweep width, not shape; never tested here |
| **13** | **Training budget** (40k steps) | is the baseline under-trained? | 0 | neutral | 0.7 | Likely shifts all arms equally; run once, not per-arm |
| **14** | **Re-measure σ on a two-state neuron** | the 2σ rule's own validity | — | — | 0.55 | σ was measured at *one* configuration; every verdict assumes it transfers |

Total: **~7.9 GPU-hours** against Phase 4's ~30, leaving room for the replication
and follow-up arms the results will demand.

**Refuted and dropped:** micro-step refinement at T > 1. On a launch-bound machine
it multiplies kernels per character by T, the readout shape for T > 1 is
unspecified in the current code (`model.py` raises `NotImplementedError`), and
`01_reconnaissance.md` §4.3 item 1 already settles T = 1 as the right default.

**Referred to Elliot, not decided here.** Three candidates sit on boundaries this
report has no authority to rule on, and §4.6's own precedent is that a boundary
decided after seeing scores is rationalisation:

* **Token-shift** (`x'_t = μ_c·x_t + (1−μ_c)·x_{t−1}`) — one parameter per neuron,
  so O(1) by the ruling's stated test, but lag-indexed by its spirit. It is the one
  SpikeGPT ingredient this project has neither implemented nor ruled on.
* **Input-gated decay** — a decay that depends on the neuron's own input makes the
  recurrence nonlinear in the state, forfeiting the parallel-scannability that
  §4.6's rationale leans on, even though the parameter count is O(1).
* **Distillation from the trained GRU** — touches no invariant (I1–I5 constrain the
  architecture, not the training signal) but changes the headline claim from
  "trained from scratch" to "distilled", which is a scope decision. The teacher
  already exists on disk.

### 6.4 The candidate most likely to fail, and why it is still ranked

The rotational membrane was presented in two families as the highest-ceiling
option. On this project's own committed criteria it is the one most likely to
produce a null: **its proposed initialisation is an exact gradient saddle.** At
θ = 0 with the imaginary state initialised to zero and no direct readout, the
imaginary component has no downstream influence, so `∂L/∂θ` vanishes identically
and the neuron trains as a soft-reset LIF in every seed. That was checkable from
its own backward equations.

It stays on the list because the mechanism is sound and only the initialisation is
broken — but it may not run until it clears the gradient-reachability screen in
§7.2. This is the clearest case in the phase of a candidate whose ROI column would
otherwise have been fiction.

---

## 7. What Phase 3 built

Two pieces of engineering, both paying down costs the experiments identified.

### 7.1 A committed mutation-testing harness — 25 / 25 caught

`EXP_002` established that the real price of a candidate neuron is not GPU time
but the mutation-tested gradient gate it needs. That gate's own provenance was
prose: **21** mutations in `02_baseline_report.md` §2.1, **sixteen** in
`src/snn/neuron.py`, **eleven** in `tests/test_neuron_equivalence.py`, with no
artifact behind any of them. Those are honest snapshots of a campaign that grew —
the test file's eleven describes a distinct earlier round that found holes in the
tests themselves — but a reader cannot tell that from the repository, and a claim
about the gate guarding against unfalsifiable results should not itself be
unfalsifiable.

`scripts/audit/08_mutation_campaign.py` re-runs it as a script. **25 mutations, 25
caught, zero escaped, source tree verifiably restored**; raw JSON in
`docs/reports/data/audit_08_mutation_campaign.json`. Each mutation's anchor text
must appear exactly once in the source or the run aborts, so a refactor that moves
the code fails loudly instead of silently reporting a clean sweep.

Adding a candidate neuron's mutations is now one list entry rather than a new
campaign, which is the largest single reduction in the cost of every new-neuron
row in §6.3.

### 7.2 Screens that were added because they caught something

* **Kernel semantics against a plain-torch reference** (`EXP_002` §8.4). Added
  after a candidate kernel compiled, ran, produced plausible spikes and **priced
  within 2 % of the correct one** while computing the wrong function. Now aborts
  the run.
* **Gradient reachability, proposed as a mandatory prerequisite.** Build the eager
  reference at a candidate's proposed init, take one backward, report `|∂L/∂p|`
  for every new parameter against `|∂L/∂W|`, and flag anything exactly zero or
  ~100× below. Seconds per candidate; it is what identifies §6.4's saddle before
  0.8 GPU-hours are spent certifying a null.
* **A concurrency guard on the mutation campaign**, added because it was needed —
  see §8.

---

## 8. An incident, reported rather than absorbed

The first `depth_K8_s0` run was **discarded and re-run.**

It started at 01:28:01, inside the window in which the mutation campaign was
writing deliberately-wrong kernels into `src/snn/`. Python reads a module's source
once, at import, so it trained 7 750 steps against mutation **M15 (`>` instead of
`>=` in the forward kernel)** — reconstructed from the campaign's own committed
per-mutation timings — with nothing in its own logs to indicate anything was
wrong. **A contaminated training run looks exactly like a clean one.**

It was caught because `EXP_001`'s F1 self-check — *the probe's `k = L` point must
reproduce the independently-committed bpc* — failed on an unrelated checkpoint at
1.14e-03 instead of the usual 1e-09, and the residual was chased rather than
tolerated. That is the entire value of insisting a probe reproduce a number it did
not produce.

The other two depth arms started before the first mutation was applied and are
clean; the timeline is in `EXP_003` §8.6. The campaign script now refuses to start
while another Python process is alive, and writes a lock file into `src/snn/` while
any mutation is applied. Wall-clock is not reported for any depth arm, because all
three overlapped other GPU work.

The lesson generalises past this repository: **a tool that deliberately writes
wrong code into the source tree must own the machine while it runs.**

---

## 9. Exit criteria

- [x] 10–15 ranked hypotheses across neuron model, coding/T, reset rule,
      optimisation, sparsity, and kernel-count reduction — **18 generated, 1
      refuted, 2 pairs merged, 8 added by a completeness audit, 14 ranked**
- [x] ROI matrix against the §5 bottlenecks, priced at the corrected 32.8 µs/kernel
      and with fusion and graph capture counted as one lever
- [x] Every candidate ruled against the §4.6 I5 boundary; three referred to Elliot
      rather than decided here
- [x] `EXP_001` — memory horizon measured: **7 characters against the anchor's
      57–60**, five seeds agreeing exactly
- [x] `EXP_002` — every admitted neuron priced: **one kernel per timestep,
      identical memory**; the systems column does not discriminate
- [x] `EXP_003` — depth ruled out as a horizon route: **horizon ∝ √K**, ~137 layers
      would be needed
- [x] The I5 gap decomposed: **51 % of it is not addressable by memory at all**
- [x] Phase-4 acceptance criterion derived from measurement and **calibrated
      per-context**, not assumed from the whole-split σ
- [x] R10 gate's provenance converted from prose to a committed artifact: **25/25**
- [x] Pre-registered predictions resolved as written, including the failures:
      **P1, P4's GRU half, R1 and Q5 all failed and are recorded as failed**
- [x] One contaminated run detected by a self-check, discarded, re-run, and the
      tooling fixed (§8)
- [ ] **Reviewed by Elliot — Phase 4 does not begin until this is signed off**

### 9.1 What Phase 4 inherits

1. **The target is 0.4633 bpc, and only 49 % of it is a memory problem.** Every
   candidate's ceiling must be quoted against the decomposition in §5, not against
   the whole gap.
2. **Depth is not the answer, and neuron-state candidates are not displaced.**
   Horizon grows as √K, so ~137 layers would be needed to reach the anchor.
3. **Buying horizon the wrong way is detectable in advance.** The §6.2 curve, with
   per-context bars, separates the GRU (0/128 contexts worse) from β=0.9 (126/128)
   — and already flags the otherwise-attractive K=4 arm at 5/128.
4. **The systems column is settled and empty.** Rank on expected bpc; the cost that
   matters is the gradient gate, and §7.1 has made that much cheaper.
5. **Three boundary questions need a written ruling before the arms that depend on
   them run** (§6.3), following the §4.6 precedent.
6. **σ = 0.00461 was measured at one configuration.** Re-measuring it on the first
   structurally different neuron Phase 4 adopts is a ranked item, not an optional
   extra: every 2σ verdict in the campaign assumes it transfers.

---

## 10. Changelog

| Rev | Change |
|---|---|
| 1 | Phase-3 deliverable: three pre-registered experiments, the horizon measurement and its confound, the I5 gap decomposition, the calibrated acceptance criterion, 14 ranked candidates, the mutation harness, and one contaminated run reported. |
