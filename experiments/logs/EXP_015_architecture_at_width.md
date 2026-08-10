# EXP_015 — Does the architecture's advantage survive at width?

**Pre-registered:** 2026-08-10, after the cost calibration in §2.1 and before any
training run. **Committed before the first run.**

**Status: PRE-REGISTERED — no results yet.**

---

## 0. What this experiment is, and what it is not

`EXP_014` trained the **plain** Phase-2 baseline neuron at 2.5M and 5.0M
parameters and found width worth **−0.25238 bpc** — more than every architectural
arm this project has built. Its own §9.10 states the limit of that finding in
terms this file exists to fix:

> "The comparison against the architectural arms is **not parameter-matched**.
> `d=1481` beats the adopted two-compartment arm by 0.118 bpc and the composed
> arm by 0.083 — **at 6.8× the parameters**. It does not show width is a better
> *mechanism* than either."

and

> "It does not say what size Phase 5 should run — **wrong neuron**, and
> `EXP_005`/`EXP_012` are two demonstrations that statistics do not transfer
> between these two neurons."

**This experiment is the parameter-matched comparison.** It trains the adopted
two-compartment arm, the composed arm and the GRU anchor at the width `EXP_014`
already trained the plain baseline at, so every arm is compared at ~5.0M
parameters on the same tree, the same seed and the same frozen recipe.

**What it is not:**

* **It cannot adopt anything.** §6.2's Phase-4 acceptance rule needs ≥3 seeds;
  this is 1 seed per arm. `adopts_nothing` and `ranks_nothing` are carried as
  structural fields in the resolver's artifact, not as prose.
* **It produces no σ at this width**, and does not pretend to. Every bar below is
  placed against a **transferred** statistic, and §3.0 states the transfer risk
  in advance rather than discovering it afterwards.
* **It is one width.** Two points do not make a scaling law and three arms at one
  width make none at all. No functional form is fitted and none may be quoted.
* **It does not authorise Phase 5**, and it does not choose Phase 5's size. It
  supplies one of the two measurements that choice needs — which arm to scale.
  The other is a σ at this width, which only seeds can produce, and a §10
  appended at close will refer it rather than this file deciding it.
* **It is not a rerun of `EXP_011`.** That measured composition at 735K and is
  provisional at n = 2. Nothing here promotes it — `CONTRIBUTING.md` §4 requires
  seeds *added* at the pre-registered width, and this is a different width.

---

## 1. The question `EXP_014` could not answer

Every arm in this project is measured at ~735K parameters:

| | params | test bpc carried | vs baseline |
|---|---:|---:|---:|
| Phase-2 baseline LIF | 735,437 | 2.25311 (n=5) | — |
| learned threshold *(adopted)* | 736,461 | 2.19416 (n=3) | −0.05896 |
| two-compartment *(adopted)* | 737,485 | 2.11869 (n=7) | −0.13442 |
| composed *(provisional n=2)* | 738,509 | 2.08326 | −0.16985 |
| GRU anchor *(violates I5)* | 738,019 | 1.76741 (n=3) | −0.48570 |
| **plain LIF at width** | **4,997,099** | **2.00073 (n=1)** | **−0.25238** |

Three worlds are consistent with that table, and they imply different Phase 5s:

1. **Additive.** The two-compartment advantage survives at width, so the arm at
   5M lands near 2.00073 − 0.134 ≈ 1.87. Phase 5 scales the composed arm.
2. **Subsumed.** Width buys what the architecture was buying, and the arm at 5M
   lands near 2.00073. Phase 4's adoption story needs re-reading — not because
   the arms did not work at 735K, but because what they bought would have come
   from capacity that cost less to add.
3. **Partial.** Some fraction survives. The fraction is the number Phase 5 needs.

**Nothing in the project discriminates between them**, and `EXP_014` §10 item 4
refers exactly this: *"whether the `twocomp` ladder is now worth its ~1.2–1.8
GPU-hours, since the sizing answer Phase 5 actually needs is on that neuron and
this pilot cannot supply it."*

### 1.1 Why `d_model = 1481`, and why width-matching is parameter-matching here

The arms differ from the baseline by **one or two vectors per layer** — `w` and
`beta_s_raw` for `twocomp`, plus `thr_log` for the composed arm — which are
`O(d)` against a stack that is `O(d²)`. So the parameter mismatch between arms at
a shared width **falls with width**, and at `d = 1481` it is negligible:

| arch | `d_model` | params | vs `snn` |
|---|---:|---:|---:|
| `snn` | 1481 | 4,997,099 | — |
| `twocomp` | 1481 | 5,003,023 | **+0.119 %** |
| `twocomp_threshold` | 1481 | 5,005,985 | **+0.178 %** |
| `gru` | 1481 | 4,997,829 | **+0.015 %** |

So **no arm needs a width of its own**, and the ladder holds `d_model` fixed
rather than solving four separate width-matching problems. At `d = 512` the same
mismatch is 0.28 % for `twocomp`; the property is a consequence of width, not a
coincidence, and `tests/test_calibrate_cost.py` asserts both the tolerance at
1481 and the mechanism (an arm whose extra parameters were `O(d²)` would fail it).

**The GRU is matched by construction, not by this file's arithmetic.**
`GRUCharLM` derives its hidden width from `spiking_param_count` at the same
`d_model` (`match_gru_width`, `src/snn/model.py`), which is why the committed
`gru_s0` is 738,019 against the baseline's 735,437. That derivation is asserted
in the test above rather than trusted.

**`d = 1481` is chosen because `EXP_014` already trained the plain LIF there**,
so the control leg costs zero GPU-hours and is a *measured* leg of this
comparison rather than an interpolation. It is also 4-byte aligned only, and
`EXP_014` §2.1 measured no GEMM cliff there (2.9 % slower than 1480, 1.1 % slower
than 1488) — checked once, not assumed twice.

---

## 2. Design

### 2.1 The entry-condition calibration, and the three things it changed

`docs/reports/data/exp_015_cost_calibration.json`, committed at `f1621a4`
**before this file was written**. 400 steps per arm at `d = 1481`, ratios against
the `snn` leg, projection against `scale_d1481_s0`'s own committed 2,191.44 s.

| arch | params | steps/s | × `snn` | projected 20k | peak VRAM | clip fired | max ‖g‖ |
|---|---:|---:|---:|---:|---:|---:|---:|
| `snn` | 4,997,099 | 9.4639 | 1.000 | 2,191 s | 1.6434 GiB | 3/17 | 1.3502 |
| `twocomp` | 5,003,023 | 7.8975 | 1.198 | 2,626 s | 2.6138 GiB | 6/17 | 1.5674 |
| `twocomp_threshold` | 5,005,985 | 7.4365 | 1.273 | 2,789 s | 2.9767 GiB | 7/17 | 2.0686 |
| `gru` | 4,997,829 | 13.6088 | 0.695 | 1,524 s | 2.1293 GiB | 1/17 | 1.6765 |

**Three things this changed, none of which would have been safe to assume:**

1. **The cost is lower than the `d = 512` ratios predict.** Those are 1.32× /
   1.41× / 1.11×; measured at width they are **1.198 / 1.273 / 0.695**, because
   the `O(d²)` GEMMs dominate the arms' `O(d)` extra state as `d` grows. The three
   new runs project to **1.93 GPU-h**, and `EXP_014` §9.8 found this 400-step
   projection runs **5–8 % high**, so ~1.8 is the expectation.
2. **The VRAM alarm inherited from `EXP_014` is too tight to run under.**
   `twocomp_threshold` peaks at **2.9767 GiB** against a 3.0 GiB alarm — 0.023 GiB
   of headroom on a 7.96 GiB card. That alarm was a modelling assumption about a
   *width*, not a property of the card. §6 sets it to **4.0 GiB** here: 34 % above
   the largest measured peak, half the card, and still far below a spill. It is
   fixed now, before any bpc exists, and the number it is derived from is in the
   table above.
3. **The clip fires more at width, and unequally by arm** — 3/17, 6/17, 7/17,
   1/17 logged steps over `grad_clip = 1.0`. After `EXP_014` §9.12 that is not a
   curiosity: the clip is **not** trajectory-inert. §5 item 3 carries it as a
   named confound rather than leaving it to be found in the results.

**And one thing it confirmed for free.** The `snn` leg reproduced `EXP_014`'s own
calibration **to full precision** — `max_grad_norm` 1.3501633405685425, clip at
steps 25/150/175, final train bpc 2.5619326792717945, peak VRAM 1.6434 — two days
and eleven commits later. Only wall-clock moved (9.05 → 9.46 steps/s, **+4.6 %**).
That is entry condition E1 (§8) discharged by measurement, and it also bounds the
precision of every ratio in the table: **±5 % is timing noise, not signal.**

### 2.2 The ladder

| leg | arch | run | status |
|---|---|---|---|
| control | `snn` | `scale_d1481_s0` | **already trained** (`EXP_014`), 2.00073 carried |
| A | `twocomp` | `arch_twocomp_d1481_s0` | to train, seed 0 |
| B | `twocomp_threshold` | `arch_twocomp_threshold_d1481_s0` | to train, seed 0 |
| anchor | `gru` | `arch_gru_d1481_s0` | to train, seed 0 |

### 2.3 The single variable

**`arch`, and nothing else.** `d_model = 1481`, `n_layers = 2`, `seed = 0`, and
every other field is `scale_d1481_s0`'s. K1 (§7) enforces this field-by-field
with **`arch` and `run_name` the only permitted differences — and no excusal
rule at all**, which is stricter than `EXP_014`'s K1 and is possible for a
specific reason worth recording: `scale_d1481_s0` was trained after `EXP_013`
added `w_init`, `beta_slow`, `thr_log_init`, `mu_init`, `noise_amp` and `noise_p`
to `Config`, so its `config.json` carries **every** field of the current
dataclass and none beyond it. `EXP_014` needed the "absent from the reference and
left at default" excusal because its reference (`snn_beta0.5_s0`) predated those
fields; this one does not.

**The arm-specific initialisations are the committed ones, and this is checked
rather than assumed.** `w_init = 0.1` and `beta_slow = 0.95` — the values
`EXP_004`'s adopted arm and `EXP_005`'s seven seeds were trained with — **are the
`Config` defaults**, and `twocomp_s0`'s own `config.json` records exactly those.
So `--arch twocomp` with no further flags reproduces the *adopted* arm's
hyperparameters rather than some untuned neighbour of it. Had they differed, this
comparison would have been measuring a different arm than the one Phase 4
adopted, and every number in it would have been unusable.

**Seed is held at 0 across all four legs on purpose.** Varying it would confound
the arm comparison with `EXP_014`'s own finding that a single training seed moves
the number; holding it means the legs share their data order exactly.


---

## 3. Hypotheses

### 3.0 What n = 1 per arm means, stated before the run rather than after

`CONTRIBUTING.md` §2 requires a prediction's power to be stated in advance.

* **There is no σ at 4,997,099 parameters for any arm, and this experiment
  produces none.** Every bar is placed against σ transferred from `d = 512`.
* **Both terms of every comparison are n = 1**, so the relevant statistic is the
  sd of a *difference*, not of a run. Taking the baseline's σ = 0.00461 for both
  terms — the larger of the two measured values, since `EXP_005` puts the
  two-compartment family at 0.00313 — gives

  > **σ_diff = √(0.00461² + 0.00461²) = 0.00652**

  and every bar below is quoted in multiples of it. This is deliberately the
  conservative choice: using each arm's own measured σ would give 0.00557 and a
  tighter bar.
* **The guard against a bad transfer is bar magnitude, not confidence.** P1 and
  P2 sit at **10 σ_diff = 0.0652**. `EXP_013` §9.5 measured σ moving 0.36×–3.94×
  under a scalar hyperparameter; at the worst of that, a *fully surviving*
  two-compartment advantage (0.134) still reads at **5.2 σ**. A *halved* advantage
  (0.067) would sit at 2.6 σ under the same inflation and would resolve
  UNRESOLVED — which is the honest outcome and is fixed here as the wording.
* **A miss resolves UNRESOLVED, never FAILED.** At n = 1 this design can show an
  effect *exceeds* a threshold. It can never show one is *absent*. "Width subsumes
  the architecture" is **not** a conclusion this experiment can reach, and §4
  fixes that sentence now so it cannot be written later.
* **The remedy for any unresolved leg is seeds *added*, never substituted**
  (`CONTRIBUTING.md` §4; `EXP_011`'s `compose_s0` is the standing example).

### P1 — the two-compartment advantage survives at width

> `bpc_carried(arch_twocomp_d1481_s0) ≤ bpc_carried(scale_d1481_s0) − 0.0652`

i.e. Δ ≤ −10 σ_diff. **HELD** means a measurable advantage survives at 5.0M
parameters. It does **not** mean the advantage is undiminished — see P1b.

*Guard:* the comparison is against a leg trained on the same tree, same seed,
same recipe, so no cross-tree offset applies (§8 E1). If Δ is positive by more
than the bar, the verdict is **"HELD IN THE HARM DIRECTION"** and is reported as
such.

### P1b — how much of it survives (a marker, not a bar)

Report Δ against the **−0.13442** measured at 735K, as a retained fraction. This
is a difference of differences with one n = 1 term and one cross-tree term, so
**no verdict attaches to it at any value** unless it exceeds 10 σ_diff in its own
right. It is the number Phase 5 wants and the number this design cannot certify;
saying both is the point of separating it from P1.

### P2 — the composed arm at width

> `bpc_carried(arch_twocomp_threshold_d1481_s0) ≤ bpc_carried(scale_d1481_s0) − 0.0652`

Same bar, same guard. **And the comparison that is *not* powered here:**
composed − twocomp was −0.03543 at 735K, which is **5.4 σ_diff** — below the
bar. It is pre-registered now as a **marker with no verdict**, and the sentence
"the composition stopped helping at width" may not be written from it. Seeds are
the remedy.

*This leg carries a named risk.* `EXP_011`'s `compose_s0` — the same arch at the
same seed, at `d = 512` — **diverged** at step 17,598, was excluded and **not
replaced**, and its mechanism is bounded to layer 0's backward but still
unidentified (`04_phase4_interim.md` §10.4). If this leg diverges, it is
**reported as a divergence and not reseeded**; P2 resolves **NOT RUN** and the
rest of the experiment stands.

### P3 — the anchor moves too, and the gap is restated at matched size

> Report `bpc_carried(arch_gru_d1481_s0)` and the gap to each spiking leg.

**No bar, by design.** The GRU violates I5 and is an anchor, not a competitor.
The reason it is trained: the project's standing gap statement compares a 5M SNN
against a **738K** GRU, which is precisely the parameter-matching error
`EXP_014` §9.10 flagged in itself. One run makes it honest. The pre-registered
statement is that **the gap at matched size is reported whichever direction it
moves**, including if it widens.

### P4 — reach

> Memory horizon per arm, `scripts/exp/001_memory_horizon.py`, whole test split.

The plain LIF went 7 → 8 across `EXP_014`'s ladder; `twocomp` is **47** at 735K.
Whether reach scales with width on that neuron is unmeasured and is the finding
most likely to matter for Phase 5.

*Guard:* at n = 1 the horizon has **no error bar**, and the probe's lattice is
`k ∈ {1,2,4,8,16,32,64,128,256}` — coarse at both ends. This is a **marker with
no verdict**, exactly as `EXP_014` S4's horizon leg was.

### P5 — the generalisation gap, extended to the arms

> `gap = bpc(test) − bpc(train slice)` per arm, via
> `scripts/exp/013_generalisation_gap.py`, **unchanged and reused**.

`EXP_014` found the plain LIF's underfitting regime ends between 2.5M and 5M
(fresh gap +0.00592 → +0.03524 → +0.05509). The two-compartment arm carries real
memory, so it plausibly extracts more from the same parameters and overfits
*earlier*. **Pre-registered direction: `gap(twocomp@1481) > gap(snn@1481)`.**

*Guard, and it is the one `EXP_013`'s N1 forced:* a bar on a difference must
constrain both terms. This is **counted only at an arm where `Δtest ≤ 0`** — i.e.
where the arm did not get worse. A gap that grows on a damaged model is not a
finding, and that is exactly how N1 fired.

### P6 — systems

> Wall-clock, peak VRAM and converged firing rate per arm, against
> `scale_d1481_s0`'s **2,191.44 s / 1.6434 GiB / 0.209–0.185**.

Reusable whatever the bpc does, and it re-prices every future width decision.
Two specific things to record: whether the GRU's **0.695×** wall-clock advantage
at width survives a full run (it was 1.11×, i.e. *slower*, at `d = 512`), and
whether the two-compartment arms' firing rates fall with width the way the plain
LIF's did (0.339/0.320 → 0.209/0.185).

---

## 4. Decision rule, fixed now

| P1 | P2 | verdict |
|---|---|---|
| HELD | HELD | **The architecture's advantage survives at 5.0M parameters.** Report the retained fraction (P1b) as a marker. Refer to Elliot: whether Phase 5 scales the composed arm, and whether the top leg is reseeded to n = 3 before any number from it is quoted |
| HELD | UNRESOLVED / NOT RUN | The two-compartment advantage survives; the composition at width is **not established**. Refer whether to reseed leg B |
| UNRESOLVED | UNRESOLVED | **"The difference was not established at n = 1."** NOT "width subsumes the architecture." Refer: the measurement that would settle it is seeds, and it costs ~0.73 GPU-h per twocomp seed |
| HELD IN THE HARM DIRECTION | any | The arm is **worse** than the plain LIF at matched size. Report it, chase nothing, and refer — a reversal at width would be the most consequential result in Phase 4 and must not be explained here |

**In every cell: nothing is adopted, no size is chosen, no phase is authorised,
and no ranked candidate is closed.** Candidate #1 was closed by `EXP_004` and
this does not reopen it.

---

## 5. Known limitations, stated in advance

1. **n = 1 per arm.** No σ, no adoption, no per-arm error bar. §3.0.
2. **σ and σ_diff are transferred from `d = 512`**, and `CONTRIBUTING.md` §4 says
   σ is not a project constant. The design sizes its bars to survive a ~4× error
   rather than pretending the transfer is safe.
3. **The arms are not equally clipped, and the clip is not trajectory-inert.**
   §2.1 measured 3/17, 6/17, 7/17 and 1/17 logged steps over `grad_clip = 1.0`,
   with max norms 1.35 / 1.57 / 2.07 / 1.68. `EXP_014` §9.12 showed three clipped
   steps in 20,000 are worth +1.97e-03 bpc. So part of any Δ measured here is a
   *clipping* difference rather than an architectural one. It is **not corrected**:
   `grad_clip = 1.0` is part of the frozen recipe, and the recipe is what is being
   compared. Its size — ~2e-3 per handful of clipped steps against bars at
   6.5e-2 — is why it is a caveat and not a confound that voids the comparison.
4. **One width.** No scaling law, no exponent, no extrapolation to any other size.
5. **`d = 1481` is 4-byte aligned only**, and the no-cliff result is this box,
   this driver, this batch size. Not a general claim.
6. **Leg B's seed has a divergence history** (`compose_s0`, §P2). Not reseeded if
   it recurs.
7. **The GRU anchor violates I5** and is reported as an anchor in every table it
   appears in. It is not a result and cannot be adopted.
8. **The control leg was trained two days earlier**, at `0dce054`. E1 (§8)
   discharges this by measurement rather than by argument.
9. **This says nothing about widths between 735K and 5M**, where the arms might
   cross. The two points that exist are the ends.

---

## 6. Self-checks — aborting, not advisory

* **VRAM alarm 4.0 GiB.** Derived in §2.1 from a measured 2.9767 GiB maximum, on
  a 7.96 GiB card. A run past it aborts: under WDDM a spill is a ~50× slowdown
  that does not raise, so it would corrupt P6's wall-clock while looking slow.
* **Parameter count** per arm against its own committed closed form
  (`param_count` in `scripts/exp/014_calibrate_cost.py`, which the driver reuses).
  A mismatch aborts — the parameter count is what makes this comparison matched.
* **Non-finite loss** anywhere is recorded, not smoothed. For leg B it is an
  expected-and-named outcome (§P2), not a failure of the experiment.
* **The pre-registration hash** is stamped before the first run and re-checked
  after the last. A change voids the experiment.

---

## 7. Gates

**K1 — config sameness.** Every run's `config.json` against
`scale_d1481_s0`'s, field by field. `arch` and `run_name` may differ; the
arm-specific fields absent from the reference must equal their dataclass
defaults. `d_model` must be 1481 and `seed` must be 0, asserted explicitly.

**G3 — parameter count** equals the arch's closed form: 5,003,023 / 5,005,985 /
4,997,829.

**G4 — peak VRAM** ≤ 4.0 GiB (§6).

**K2 — no mutation-campaign lockfile** before any child process.

**K3 — the cost model is checked, not assumed.** Realised wall-clock against
§2.1's projection, per arm. `EXP_014` measured this ratio at 0.925–0.949; a
figure far outside that is a systems finding and is reported.

**F1 — a probe reproduces a number it did not produce.** Each horizon probe's
`k = L` leg must reproduce that run's own `final_test.json` bpc.

**No G1.** `EXP_014` needed a bitwise-reproduction gate because its anchor was
trained on a different tree. Every leg here is anchored to `scale_d1481_s0` on
**this** tree, so decision #7's offset cancels rather than needing a gate. E1
below is what establishes that, and it is already discharged.

---

## 8. Entry conditions

**E1 — today's tree still produces `EXP_014`'s numbers. DISCHARGED, 2026-08-10.**
The `snn` leg of §2.1's calibration reproduced `EXP_014`'s own calibration to
full precision — `max_grad_norm` 1.3501633405685425, clip at steps 25/150/175,
final train bpc 2.5619326792717945, peak VRAM 1.6434 — across eleven commits, of
which the only `src/snn` change is a docstring. So `scale_d1481_s0` is a valid
same-tree control and this experiment needs no G1.

**E2 — the cost and VRAM of every leg are measured, not estimated. DISCHARGED**
(§2.1, committed at `f1621a4` before this file).

**E3 — the parameter match is asserted, not assumed. DISCHARGED** —
`tests/test_calibrate_cost.py`, which pins both the ≤0.2 % tolerance at `d = 1481`
and the `O(d)`-vs-`O(d²)` mechanism behind it.

**E4 — no `src/snn` change is required.** All four arch names are already in
`ARCH_CHOICES` and dispatched by `build_model`. This experiment introduces **no
new parameter and no new architecture**, so it needs no gradient-reachability
screen, no R10 gradient gate and no mutation-campaign re-run. That is why it is
the cheapest informative item available in everything except wall-clock.

**E5 — the resolver is committed before any rung's bpc is read**, with
`--print-bars` for transcription checking.

---

## 9. Results

**CLOSED 2026-08-10. Two of the three trained legs died, the leg with no bar
produced the finding, and the experiment's most useful output is a
Phase-5 blocker rather than a bpc.** ~1.79 GPU-hours of training (2,432 + 2,551 +
1,481 s), three evaluations, one horizon sweep over seven checkpoints, one gap
sweep over two, one divergence chase and one falsified side-hypothesis.

### 9.1 The scoreboard

| leg | arch | params | test bpc fresh | test bpc **carried** | wall-clock | peak VRAM | outcome |
|---|---|---:|---:|---:|---:|---:|---|
| control | `snn` | 4,997,099 | 2.02007 | **2.00073** | 2,191.4 s | 1.6434 GiB | trains |
| A | `twocomp` | 5,003,023 | **NaN** | **NaN** | 2,432.2 s | 2.6138 GiB | **diverged, step 5138** |
| B | `twocomp_threshold` | 5,005,985 | **NaN** | **NaN** | 2,551.0 s | 2.9767 GiB | **diverged, step ~2000** |
| anchor | `gru` | 4,997,829 | 1.59672 | **1.54699** | 1,481.3 s | 2.1293 GiB | trains |

**Both arms containing the two-compartment neuron failed to train at 5.0M
parameters under the frozen Phase-2 recipe. The plain LIF at the identical width,
seed, recipe and tree does not, and neither does the GRU.** At 735K these same
arms trained across 7 and 2 seeds respectively.

### 9.2 P1 and P2 — the bars are reported exactly as they fired, and the words are wrong

`015_arch_results.py` resolves both to **UNRESOLVED**, because a NaN delta is
neither `≤ −0.0652` nor `≥ +0.0652`. Its decision cell reads *"The difference was
not established at n = 1 … the measurement that would settle it is seeds."*

**Both sentences are false of what happened, and neither is repaired here.**
§3.0 defines UNRESOLVED as a statement about *power* — that the design could not
resolve a difference. What occurred is that two runs produced no number at all.
And seeds would not settle it: the divergence is a **deterministic** function of
(checkpoint, step), reproduced exactly in §9.7, so a second seed measures a
different trajectory rather than the same one again.

This is the **third** time in Phase 4 that a pre-registered rule has fired and
said the wrong thing — after `EXP_012`'s Y5 and `EXP_013`'s N1, both cited in
decision #9. `CONTRIBUTING.md` §3 says report the verdict and refer the
correction, so: **P1 = UNRESOLVED, P2 = UNRESOLVED, as fired.** The corrected
rule — that a verdict vocabulary needs a **DIVERGED / NO RESULT** cell distinct
from UNRESOLVED — is referred in §10, not applied.

### 9.3 P3 — the leg with no bar is the one that produced the result

The anchor was trained so that the project's gap statement would stop comparing a
5M spiking model against a 738K GRU. It did that, and the correction is large:

| comparison | gap to anchor | |
|---|---:|---|
| 735K vs 735K | **0.48570** | baseline 2.25311 − GRU 1.76741 |
| 5M SNN vs **735K** GRU | 0.23332 | the unmatched comparison, closes 52.0 % |
| **5M vs 5M** | **0.45374** | the matched one, closes **6.6 %** |

**The anchor scales almost as well as the spiking model.** Width bought the SNN
−0.25238 bpc and the GRU **−0.22042**. Both improve; the distance between them
barely moves. **Capacity is not what separates this architecture from its
anchor**, and the "width buys more than every arm" reading of `EXP_014` — true as
far as it went — does not survive being parameter-matched on both sides.

### 9.4 P4 — width buys the anchor reach, and buys the spiking model none

Memory horizon, 2σ criterion, whole test split, `f1_failures` **empty**:

| arm | params | horizon (2σ) | horizon (0.05 bpc) |
|---|---:|---:|---:|
| `snn_beta0.5` (n=5) | 735,437 | 7, 7, 7, 7, 7 | 5 |
| `scale_d1481` | 4,997,099 | **8** | 6 |
| `arch_gru_d1481` | 4,997,829 | **97** | 34 |
| *(committed refs)* `twocomp` 735K | 737,485 | *47* | — |
| *(committed refs)* `gru` 735K | 738,019 | *57–60* | — |

**The GRU's reach grows with width — 57–60 → 97. The spiking model's does not —
7 → 8.** Both gained about the same bpc from the same parameters, and they spent
it on completely different things.

The absolute per-context curves say the same thing without any horizon
definition, which matters because the horizon is a marker at n = 1:

| context c | 0 | 1 | 2 | 4 | 8 | 16 | 32 | 64 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `arch_gru_d1481` | 4.0014 | 3.2265 | 2.7013 | 2.1781 | 1.8766 | 1.7222 | 1.6552 | 1.6126 |
| `scale_d1481` | 4.0524 | 3.3494 | 2.8761 | 2.2261 | 2.0291 | 2.0194 | 2.0192 | 2.0201 |
| `snn_beta0.5` (735K) | 4.1617 | 3.5403 | 2.9590 | 2.3836 | 2.2727 | 2.2697 | 2.2699 | 2.2697 |

**At zero context the two 5M models are nearly equal — 4.0524 against 4.0014,
a difference of 0.051.** By `c = 8` it is 0.152; by `c = 64` it is **0.408**, and
the spiking curve has been flat since `c ≈ 8` while the GRU's is still falling.
**The entire matched-size gap is a failure to use context, not a failure to model
the marginal.**

*Guard, as pre-registered:* n = 1 at both wide points, the lattice is
`k ∈ {1,2,4,…,256}`, and this is a **marker with no verdict**. The size of the
effect — 8 against 97 — is why it is reported at all.

### 9.5 P5 — the gap, and the anchor overfits harder

| run | gap fresh | gap carried |
|---|---:|---:|
| `scale_d1481_s0` | +0.05509 | +0.04324 |
| `arch_gru_d1481_s0` | **+0.11872** | **+0.11368** |

**The pre-registered comparison could not be made**: P5's rule is
`gap(twocomp) > gap(control)` and `twocomp` produced no gap. **NOT RUN.**

What was measured instead, on the anchor: at matched parameters the GRU's
generalisation gap is **2.2× the spiking model's**, while its test bpc is 0.454
better. It extracts more from the data *and* memorises more of it. The `Δtest ≤ 0`
guard is satisfied (the GRU's test leg is better, not worse), so this is the
same "memorisation alongside a real gain" pattern `EXP_014` §9.4 drew — now
observed across architectures rather than across widths.

### 9.6 P6 — systems

| arm | wall-clock | × control | peak VRAM | K3 realised/projected |
|---|---:|---:|---:|---:|
| `snn` control | 2,191.4 s | 1.000 | 1.6434 GiB | — |
| `twocomp` | 2,432.2 s | 1.110 | 2.6138 GiB | 0.9262 |
| `twocomp_threshold` | 2,551.0 s | 1.164 | 2.9767 GiB | 0.9147 |
| `gru` | 1,481.3 s | **0.676** | 2.1293 GiB | 0.9720 |

**The GRU is faster than the spiking model at this width** — 0.676× — having
been **1.11×**, i.e. *slower*, at `d = 512`. The anchor overtakes on wall-clock
somewhere between the two widths. Every K3 ratio is inside or beside `EXP_014`'s
measured 0.925–0.949 band, so §2.1's projection method transfers across arms.

Two caveats on the two dead rows: their wall-clock is the cost of running 20,000
steps, ~14,750 of which computed NaN, so it prices the *configuration* and not
useful work; and the composed arm's 2.9767 GiB is exactly §2.1's calibration
figure, which is why §6 raised the alarm to 4.0 before the run rather than after.

### 9.7 The divergence, chased to the same boundary as `compose_s0`

`011_chase_compose_divergence.py`, **reused unchanged** but for a `--tag` so it
would not write into `EXP_011`'s scratch directories. Replayed from
`arch_twocomp_d1481_s0/ckpt_best.pt` (step 5000, the last healthy evaluation).

**Reproduced deterministically at step 5138** — 138 steps after the checkpoint,
inside the 250-step logging cadence that made the run's own log read 5250.

| | `compose_s0` (`EXP_011`, 735K) | `arch_twocomp_d1481_s0` (5.0M) |
|---|---|---|
| step | 17,598 | **5,138** |
| origin | backward | **backward** |
| forward finite | yes | **yes** |
| loss at the failing step | 1.4622 | **1.4198** |
| fp32 vs fp64 grad norm | equal | **equal** |
| eager path | identical NaN | **identical NaN** |
| non-finite gradients | embed, layers.0.{weight,bias}, w.0, beta_s_raw.0, thr_log.0 | **the same five, minus thr_log** |
| param abs max | 4.145 | 4.226 |
| logits abs max | ~226 | ~250 |

**It is the same mechanism, and this is the first time it has been seen on the
plain adopted arm rather than the composed one.** Three things it is not: not
`EXP_009`'s fp32 clip overflow (the two norms agree); not a fused-kernel defect
(the eager path produces the identical NaN in the identical tensors, so R10 is
clean and `fused_only_nonfinite_params` is empty); and not a gradient explosion —
the largest gradient in the six instrumented steps before death is **2.0e-02**,
and the clip never fires in the whole run (max norm 0.663).

**Always layer 0 and the embedding; never layer 1.** `EXP_012` established that
layer 0 is where the decision variable is degenerate — it sees one of 205
embedding rows and the hard reset zeroes `v` exactly, giving ~20 distinct
near-threshold values replayed ~548 times each. The two facts have not been
connected by any measurement, and this file does not connect them.

### 9.8 A hypothesis about state growth, stated and then falsified

The two chases differ in one instrument by an order of magnitude — state max
`[~30, ~66]` at 735K against `[~200, ~700]` at 5M — so the obvious hypothesis was
that **the two-compartment membrane grows with width and the plain LIF's does
not**. `015_chase_state_growth.py` tested it on four committed checkpoints,
inference-only, one batch each.

| run | arch | `d` | state max `[L0, L1]` | logits max | param max |
|---|---|---:|---|---:|---:|
| `snn_beta0.5_s0` | `snn` | 512 | [62.9, 41.8] | 242.1 | 5.307 |
| `scale_d1481_s0` | `snn` | 1481 | [95.9, **556.3**] | 352.9 | 5.281 |
| `twocomp_s0` | `twocomp` | 512 | [54.8, 45.4] | 242.5 | 4.344 |
| `arch_twocomp_d1481_s0` | `twocomp` | 1481 | [272.4, **679.6**] | 245.9 | 4.253 |

**FALSIFIED.** Layer 1's state grows **13.30×** for the plain LIF and **14.96×**
for the two-compartment arm across the same width change. The surviving arm grows
*slightly more*. A plain LIF carrying `|state|` up to 556 trains for 20,000 steps
without incident, so **large state is survivable and is not the discriminator.**

What survives the refutation is narrower and is offered as a lead, not a result:
at **layer 0** — the locus of every non-finite gradient — the two-compartment
state is **2.84×** the plain LIF's at the same width (272.4 against 95.9), and it
is the **reset-shielded slow pole** that carries it (272.39 slow against 54.04
fast). Whether that matters is unmeasured.

One hypothesis was also removed without spending GPU: `beta_s = sigmoid(beta_s_raw)`
is bounded in (0, 1), so the slow pole cannot become an undamped integrator, and
AdamW's decay pulls `beta_s_raw` toward 0 — i.e. toward *more* damping, τ → 2.
"The slow pole ran away" is not available as an explanation.

### 9.9 Gates

**K1 3/3** — every leg differs from `scale_d1481_s0` in `arch` and `run_name`
only, with `absent_from_reference_and_left_at_default` **empty** at all three, as
§2.3 predicted. **G3 3/3 exact** — 5,003,023 / 5,005,985 / 4,997,829 against the
committed closed forms. **G4 3/3** — 2.6138 / 2.9767 / 2.1293 GiB against the
4.0 alarm. **K2** — no mutation lockfile before any of the six child processes.
**K3** — 0.9262 / 0.9147 / 0.9720. **F1** — `f1_failures` empty; every horizon
probe's `k = L` leg reproduced its own run's committed bpc. **Pre-registration
hash unchanged** across the ladder (`prereg_unchanged: true`,
`3982e59b…97f7889e`). **No G1**, and E1 (§8) is why.

### 9.10 Three defects this experiment found in its own instruments

Recorded because two of them would have been invisible in the artifact:

1. **The driver marked both dead legs `completed: true`.** Its test asked only
   `returncode == 0 and ckpt_final.pt exists`, and a trainer that goes non-finite
   satisfies both — nothing in it treats NaN as an error. `scan_for_divergence`
   recorded all 60 and 73 non-finite records correctly, so **no evidence was
   lost**, but the field said the opposite of what happened. `leg_completed()`
   now consults the scan, and `tests/test_calibrate_cost.py` pins it. **The
   committed manifest still reads `completed: true` for legs A and B** — it is
   the record of what the driver did, and it is not rewritten.
2. **The same defect one level down**, found while writing that test: an *absent*
   log made `n_nonfinite_loss_records` absent, and absent read as zero. A missing
   log is not a clean log; `leg_completed` now requires `log_present`.
3. **The resolver crashed on P4** — it assumed `exp_015_memory_horizon.json`
   carried a list of per-arm dicts, and `001_memory_horizon.py` keys `by_arm` by
   name with the horizons under `_paired`. Fixed after the run. **No threshold,
   bar or verdict is touched by that fix** — P4 has no bar — and the correction
   is recorded here rather than made silently.

### 9.11 What this does not answer, and what it must not be read as

* **Nothing is adopted, no size is chosen, no phase is authorised**, and no
  ranked candidate is closed. `adopts_nothing` and `ranks_nothing` are structural
  fields in the artifact.
* **It does not show the two-compartment arm is worse at width.** It shows the
  arm *did not train* at width **under this recipe**, which is a different claim.
  The recipe — `lr = 3e-3`, cosine over 20,000 steps, `grad_clip = 1.0`,
  `weight_decay = 0.1` — was fixed at 735K parameters and has never been
  re-derived at any other size.
* **It does not identify the failing arithmetic.** The chase bounds it to layer
  0's backward on both paths, which is exactly where `EXP_011` §10.4 left it.
* **σ at this width is still unmeasured**, for every arm.
* **One width, one seed.** No scaling law, no exponent, and nothing about the
  widths between 735K and 5M where the arms might still train.
* **The GRU remains an anchor and violates I5.** Its 1.54699 is not a result this
  project may claim, and its horizon of 97 is not a target that has been shown to
  be reachable under the spiking constraint.

### 9.12 The pre-registration guarantee, checkable after the fact

This file was **committed before the first run** (`d4469d6`), so git history is
the primary evidence; the hash chain is kept as well, because a commit proves
ordering only if nobody amends it.

`exp_015_run_manifest.json` stamps
`3982e59ba5b8ef90d3d05a74f29d16aadca2576d3fda340f23a2c1eb97f7889e` both before
the first run and after the last (`prereg_unchanged: true`). To reproduce it from
this file as it now stands: take every byte up to the start of the `## 9. Results`
line and append the two-line stub this section replaced —

```
## 9. Results

*(Empty at pre-registration. Appended after the runs; predictions are resolved as
written, including the ones that fail.)*
```

— and the SHA-256 of the result is that value. **Verified 2026-08-10.** §0–§8 are
byte-identical to what was hashed before the first training run: no hypothesis,
no bar and no decision cell moved after the numbers arrived.

**The recipe is line-ending dependent, and that is not a footnote.** This file is
**CRLF** in the working tree — `.gitattributes` sets `* text=auto eol=lf`, so git
stores LF in the index and checks out CRLF on this machine — and the driver hashes
the *working-tree* bytes. Reconstructing with `\n` gives
`c0e757901c8d36d6be6870d448ca1c1b5a764cbfff5db378f013baa5be7d922c`, which is not
the stamp; a future reader who checked that way would conclude the pre-registration
had been edited when it had not. **Reconstruct with CRLF.** The prefix through the
end of §8 hashes to
`1fb038544cb1d0228f869557b57369714bc435a6e482f4e82c07f190a4923a69`, which is the
value a later revision of §9 should be checked against instead.

---

## 10. Referred to Elliot, and not decided here

1. **The recipe, and it is the item that blocks Phase 5.** The frozen Phase-2
   recipe does not train the adopted arm at 5.0M parameters. Nothing here says
   which term is responsible; the cheapest discriminating measurement is a short
   LR sweep on `twocomp` at `d = 1481` — it dies at step 5138, so each probe is
   **~11 GPU-minutes**, not a full run. Whether that runs as a diagnostic or as a
   pre-registered arm is Elliot's; it changes a frozen hyperparameter either way.
2. **Whether the two-compartment arm is retried at an intermediate width.** It
   trains at 735K and dies at 5.0M. The width at which it stops training is
   unmeasured, and `d = 1020` (2.5M, where `EXP_014` has a control) is one run.
3. **The verdict vocabulary.** A pre-registered rule fired and said the wrong
   thing for the third time in this phase (§9.2). UNRESOLVED, which means "the
   design lacked the power", was returned for two runs that produced no number at
   all. The corrected form — a **DIVERGED / NO RESULT** cell that is not
   UNRESOLVED and does not recommend seeds as the remedy — is decision #9's
   substance a third time, and is referred rather than applied.
4. **The third divergence, and what it now costs to leave open.** Phase 4 has
   three unexplained divergences and two of them share a signature that is now
   reproducible in **138 steps from a committed checkpoint** rather than 17,598.
   `EXP_011` §10.4 called this the biggest open diagnostic; it is now both bigger
   — it blocks scaling the adopted arm — and much cheaper to chase.
5. **Whether the anchor's result changes what Phase 5 is for.** At matched size
   the GRU is 0.45374 ahead, its reach is 97 against 8, and the two models are
   nearly equal at zero context. That is a different problem statement from the
   one the phase opened with, and restating it is not this file's to do.
