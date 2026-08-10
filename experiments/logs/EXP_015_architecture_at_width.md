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

*(Empty at pre-registration. Appended after the runs; predictions are resolved as
written, including the ones that fail.)*
