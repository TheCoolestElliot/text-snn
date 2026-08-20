# EXP_022 — Every arm in EXP_020 lost. Two of them lost for reasons that design could not read.

**Pre-registered:** 2026-08-18, after the arithmetic in §2, the §7.2
reachability screen in §2.6, and the cost pre-flight in §2.7, and **before any
arm in §3 was trained, any checkpoint was written, and any bpc existed for any
arm in §3.** **Committed before the first run of §3.**

**Status: OPEN.**

---

## 0. What this experiment is, and what it is not

`EXP_020` opened the model's two ends for the first time and **every arm lost**.
That is not the same as every question being answered. Two of the four results
are uninterpretable as they stand, and this experiment is the smallest thing
that makes them readable:

* **`binin` cost 0.0448 bpc and 83 % of it sits at zero context** — a
  per-character expressiveness cost, not a memory cost. It is provably **not**
  an information-capacity cost (§2.2), so it is a claim about the code's
  *geometry*, and `EXP_020` measured exactly one geometry.
* **`mlayer` read −0.0054, which is the instrument's own noise floor.**
  `EXP_020` §9.5 measured that floor with a **bitwise copy of the anchor** at
  −0.00499. The arm paid for its wider head by cutting `d` 512 → 471, and
  `EXP_014`'s slope prices that cut at ≈ 0.023 bpc — **four times the effect
  being looked for.** Whatever the readout is worth, that design could not see
  it.

**§2 OF THIS EXPERIMENT IS POST-HOC-INFORMED AND IS LABELLED AS SUCH.**
`EXP_020` §5 fixed a rule for which EXP_022 arms would run, and that rule fired
**NOT RUN on four of its five arms**: `composed` and `composed_bin` required H2
and H4 to hold and neither did; `sparse34` and `sparse10` required H3 to fail
*within* 4σ and it failed at 0.04569, which is 2.5× the 0.01844 threshold. The
rule is followed — **none of those five arms is run under it** — and this is a
new pre-registration whose §2 is written knowing `EXP_020` §9. `EXP_019` is the
precedent for that construction and this section is the label it requires.

**What this is not:**

* **It adopts nothing and ranks nothing.** Adoption is Elliot's (report §8), and
  no arm here may be ranked in §7.1: ranking it would be this report deciding
  its own next experiment.
* **It changes no hyperparameter.** The Phase-2 recipe — `lr = 3e-3`, cosine
  over 20,000 steps, `grad_clip = 1.0`, `weight_decay = 0.1`, `beta = 0.5`,
  `threshold = 1.0`, hard reset, `atan` at `alpha = 2` — is frozen, and G5 reads
  all of it back out of each run's own `config.json`.
* **It does not decide #11.** Every arm here is at 735K–840K on the plain LIF.
  Nothing is run at 5.0M and nothing here says the recipe may be re-derived at a
  new size.
* **It does not edit any adopted arm.** `src/snn/neuron.py`, `kernels.py`,
  `surrogate.py`, `twocomp.py`, `twocomp_detach.py` untouched; G1 asserts it by
  `git diff`. **No new scan, no new recurrence, no hand-written backward**, so
  the R10 gate and the 59-mutation campaign are not re-run and the gate table
  says "untouched" rather than implying otherwise.
* **It violates no invariant.** I1 is *strengthened* by leg A, which removes the
  one exception the committed model has and then asks what density that
  exception should be closed at. I2, I3, I4 untouched. I5: the only `[d, d]`
  parameters are still `layers.k.weight`.
* **It does not re-baseline.** Decision #10. Every comparison is against
  `if_anchor` and `if_binin`, which were trained **in this session on this
  tree** as `EXP_020` legs 1 and 4. No figure here may be compared with a
  committed one.
* **It reuses two cells rather than re-running them.** `if_anchor` (n = 3) is
  the analogue reference and `if_binin` (n = 3) **is the `q = 0.50` rung of leg
  A's own ladder.** Re-training them would spend 0.7 GPU-h to reproduce numbers
  this session already holds at the same recipe, the same tree and the same
  seeds — and would then have two anchors that could disagree.

---

## 1. The question

> **The binary input code costs 0.0448 bpc at one density nobody chose. Is that
> a cost of binarity, or a cost of that density? And is the multi-layer readout
> worth its parameters when it is not made to pay for them out of width?**

Two legs, deliberately separated rather than composed:

* **Leg A — the input.** A four-point density ladder `q ∈ {0.50, 0.34, 0.10,
  0.05}` on the binary input code, at fixed width and **fixed parameter count**.
* **Leg B — the output.** The all-layer readout at **unmatched** width, against a
  plain-LIF width arm carrying the same extra parameters.

---

## 2. The derivation, done before any measurement

### 2.1 What `EXP_020` established, and the exact sentence this starts from

Copied from `EXP_020` §9.4, not re-derived (positive = better):

| arm | total | c = 0 | within reach | beyond |
|---|---:|---:|---:|---:|
| `binin` | −0.0448 | **−0.0371** | −0.0029 | −0.0048 |
| `mlayer` (d = 471) | −0.0054 | −0.0370 | +0.0313 | +0.0003 |
| *(bitwise copy of the anchor — the noise floor)* | −0.0050 | −0.0140 | +0.0094 | — |

So: **83 % of binarising the input is paid at zero context**, and `mlayer`'s
total is inside the floor a bit-identical run reads.

### 2.2 The input code is NOT information-limited, and that is why density is the
### axis worth moving

The code is `[V, d]` binary at density `q`, so its capacity is
`d · H_b(q)` bits per character against an alphabet needing `log2(205) = 7.7`:

| `q` | `iface_thr_in` | derived gain `g` | active fibres | capacity |
|---:|---:|---:|---:|---:|
| **0.50** | 0.0000000000 | 2.000000 | 256 | 512 bits |
| **0.34** | 0.4124631294 | 2.111002 | 174 | 474 bits |
| **0.10** | 1.2815515655 | 3.333333 | 51 | 240 bits |
| **0.05** | 1.6448536270 | 4.588315 | 26 | 147 bits |

**Nothing on this ladder is information-limited** — the sparsest rung carries
19× the bits the alphabet needs. So if sparsity moves the number, it moves it
through how the code interacts with the GEMM and the threshold, and that is a
pre-registrable claim rather than a hope.

**Every constant above is derived and none is tuned.**
`q = 1 − Φ(thr_in / code_sd)` and `g = 1 / sqrt(q(1−q))`, the latter fixed by
matching the unit second moment `layers.0`'s init was calibrated for — the same
identity `snn.interface`'s docstring derives for `q = 0.5` and the same one
`02_baseline_report.md` §5.2 says an input-side change may not move. **The gain
is not a second knob:** it is a function of `thr_in`, computed by the module,
and K1 records `iface_thr_in` as the one field that changed.

**Why `q = 0.34` is a rung and not a round number.** It is the firing rate the
stack itself converges to (`EXP_000` F3: 0.345–0.392; `audit_07` for layer 0).
At that rung the input code is **statistically identical to what every other
layer receives**, which is the strongest form of the I1 completeness argument:
the input boundary then *looks* like every other boundary rather than merely
being binary at it.

**Why sparsity is the mechanism worth testing.** At `q = 0.5` one bit flip moves
layer 0's current by `g` times one column of `W0` against a sum of ~256 such
terms; at `q = 0.10` it moves it against a sum of ~51, with `g` 1.67× larger —
about **8× the relative leverage per bit**. A cost concentrated at `c = 0` is a
cost of *per-character discriminability*, and per-character discriminability is
exactly what leverage-per-bit controls.

### 2.3 The confound this ladder carries, named before it is run

`thr_in` moves density **and** plasticity together: at `q = 0.10` the typical
shadow sits 1.28 from its own threshold rather than 0.80, so `atan_grad` is
attenuated more. The two therefore move in **opposite** directions — a sparse
rung is more discriminable per bit and less plastic per bit — which means:

> **If a sparse rung WINS, it wins despite a worse gradient regime and the
> reading is clean. If it LOSES, the result is confounded and this experiment
> may not say which axis caused it.**

That asymmetry is stated now so §9 cannot choose it later.
`iface_code_sd` is the axis that separates them, and it is implemented, tested
and **NOT RUN**: at `thr_in = 0` it is a *pure backward* intervention — scaling
a shadow cannot change its sign, so the code table, the gain, the centre and the
emitted current are bitwise identical from the same seed and only the surrogate
gradient differs (`tests/test_interface.py::
test_b1_code_sd_moves_the_backward_and_provably_not_the_forward`). Running it at
one dose is 3 runs, **~0.35 GPU-h**, and it is §6's first limitation.

### 2.4 The output leg: what `mlayer` could not say, and the arithmetic that fixes it

`EXP_020` leg 5 held parameters fixed by cutting `d` 512 → 471. The extra head
block costs `d·V = 104,960` parameters, and `EXP_014`'s slope prices a cut that
size at **≈ 0.023 bpc** — against an effect the frozen-feature probe put at
**0.019–0.022**. The confound and the signal were the same size, so leg 5
measured their difference and reported the floor.

Here the readout is run **unmatched**, and the control is a **plain-LIF width
arm carrying the same extra parameters**:

| arm | arch | `d` | realised params | of `if_anchor` |
|---|---|---:|---:|---:|
| `if_anchor` *(EXP_020 leg 1, reused)* | `snn` | 512 | 735,437 | 100.000 % |
| **`io_mall`** | `interface`, `read_layers="all"` | 512 | **840,397** | 114.27 % |
| **`io_wide`** | `snn` | 553 | **839,659** | 114.17 % |

`io_wide` is 738 parameters (0.088 %) under `io_mall` — the nearest integer
width on the plain-LIF grid, reported as realised rather than claimed as an
identity. **That pairing is the experiment**: `io_mall − io_anchor` says whether
the readout is worth anything, and `io_mall − io_wide` says whether it is worth
**more than the same parameters spent on width**, which is the only form of the
question `EXP_014` leaves open.

**A mechanistic prediction, stated before looking:** layer 0's spikes at `t`
carry no information about `t' < t` that layer 1's do not already carry, so a
multi-layer readout **adds no temporal reach**. Its gain must therefore be at
zero context and within reach, and **must not be beyond the horizon.** If it
gains beyond the horizon, the mechanism is not the one claimed.

### 2.5 What is deliberately NOT run

* **The five arms `EXP_020` §5's rule blocked.** Reported as NOT RUN in §9,
  with the rule's own arithmetic, exactly as `EXP_020` §2.6 recorded the tap
  ladder as NOT RUN on the probe's evidence.
* **Composition of leg A with leg B.** Two arms at once. If both legs produce a
  result, the composition is a referral, not a leg.
* **`iface_code_sd`.** §2.3, priced at ~0.35 GPU-h.
* **A tap ladder.** `EXP_020`'s frozen-feature probe measured an extra time tap
  as **null**, and nothing here changes that evidence.

### 2.6 The §7.2 reachability screen — run before this file was committed

`docs/reports/data/exp_022_023_reachability.json`. Both new candidates entered
through the **real arch**, not a prototype:

| candidate | init | verdict | measured |
|---|---|---|---|
| `iface_sp10` | `thr_in = 1.2816` (q = 0.10) | **PASS** | `|dL/dshadow|` ratio **0.032** |
| `iface_mall` | `read_layers="all"`, d = 512 | **PASS** | ratio **—**, head block reachable |

The screen's own controls held (`ctl_live` PASS / `ctl_dead` SADDLE /
`ctl_faint` FAINT) and it reproduced `exp_004_gradient_reachability.json` at
worst relative residual **0.00e+00**.

**AND A PRE-RUN PREDICTION IN THAT SCREEN FAILED, WHICH IS RECORDED HERE RATHER
THAN EDITED.** The `iface_sp10` derivation states the ratio "must come in BELOW
`iface_binary`'s" because the shadow sits further from a threshold at 1.2816
than at 0. It came in **0.032 against 0.028 — above, not below.** The
reachability verdict (PASS, non-zero, no saddle) is unaffected and the ladder is
not gated on the ratio; the prediction is wrong and is reported as it fired. The
derivation in `scripts/audit/09_gradient_reachability.py` is **not edited** to
match.

### 2.7 The cost pre-flight — run before this file was committed

`docs/reports/data/exp_022_023_cost_preflight.json`, 400 steps per row through
`scripts/exp/014_calibrate_cost.py` (the same instrument `EXP_014` used; the
`--extra` path was added so an arm whose cost depends on a *switch* is priced by
this instrument rather than a second one). Denominator `snn_beta0.5_s0`,
20,000 steps, 398.88 s.

| row | steps/s | ratio | projected 20k | peak VRAM |
|---|---:|---:|---:|---:|
| `snn` d = 512 *(base)* | 54.63 | 1.0000 | 398.9 s | 0.609 GiB |
| `io_sp10` | 54.67 | **0.9992** | 398.6 s | 0.609 GiB |
| `io_binin` *(q = 0.50, cross-check)* | 54.64 | 0.9999 | 398.8 s | 0.609 GiB |
| `io_mall` | 46.13 | **1.1842** | 472.4 s | 0.693 GiB |
| `io_wide` | 42.87 | **1.2744** | 508.3 s | 0.656 GiB |

**The density ladder is free** — 1.00× to three digits, because `thr_in` changes
a constant inside a kernel that already ran. And **`io_mall` is CHEAPER than the
width arm it is matched to** (1.18× against 1.27×), which is a real and separate
finding about where the parameters go: a wider stack pays `O(d²)` in the scan,
a wider head pays `O(dV)` once.

**The project's one previously-checked GPU-hour estimate was low by 3.1×**, which
is why this table exists and why §8 is computed from it rather than guessed.

---

## 3. The legs

`if_anchor` (n = 3) and `if_binin` (n = 3) are **reused from `EXP_020`** and not
re-run — §0. Seeds 0, 1, 2 everywhere.

| leg | run names | arch | `d` | one field vs its reference | realised params |
|---|---|---|---:|---|---:|
| **A0** | `if_binin_s{0,1,2}` *(reused)* | `interface` | 512 | — *(the q = 0.50 rung)* | 735,437 |
| **A1** | `io_sp34_s{0,1,2}` | `interface` | 512 | `iface_thr_in = 0.4124631294` | 735,437 |
| **A2** | `io_sp10_s{0,1,2}` | `interface` | 512 | `iface_thr_in = 1.2815515655` | 735,437 |
| **A3** | `io_sp05_s{0,1,2}` | `interface` | 512 | `iface_thr_in = 1.6448536270` | 735,437 |
| **B1** | `io_mall_s{0,1,2}` | `interface` | 512 | `iface_read_layers = "all"` | 840,397 |
| **B2** | `io_wide_s{0,1,2}` | `snn` | 553 | `d_model = 553` | 839,659 |

**Leg A is parameter-identical to the anchor at every rung** — 735,437 on all
four — because changing a code's density changes its alphabet and not its shape.
That makes leg A a single-variable comparison in the strongest sense this
protocol has, and it is the reason the ladder is worth four points.

Reference runs for K1: A1–A3 against `if_binin_s0`; B1 against `if_nest_s0`;
B2 against `if_anchor_s0`.

---

## 4. Pre-registered predictions

σ is **transferred**: 0.00461 from the report's five baseline seeds on a
different tree, so **2σ = 0.00922**, exactly as `EXP_020` and `EXP_021` used it.
`EXP_020` measured σ = 0.004683 on this tree — a ×1.02 transfer, the first time
this project checked one and found it held — and §9 reports the on-tree σ again
at n = 3 per cell. Comparisons are **paired by seed**; t is the paired t over
3 differences unless stated.

| # | claim | bar | direction predicted |
|---|---|---|---|
| **H1** | `io_sp34` vs `if_binin` | ≥ 2σ = 0.00922 | **sp34 BETTER.** At the stack's own rate the input boundary is statistically the same boundary as every other one, and §2.2's leverage argument says a sparser code is more discriminable per bit. |
| **H2** | `io_sp10` vs `if_binin` | ≥ 2σ | **sp10 BETTER, and by MORE than sp34.** If leverage-per-bit is the mechanism it is monotone in `1/q` over this range, since none of these rungs is capacity-limited. |
| **H3** | `io_sp05` vs `if_binin` | ≥ 2σ | **UNSTATED DIRECTION.** 26 active fibres is where a capacity or a plasticity floor would first appear, and predicting a direction here would be predicting which of the two arrives first. Reported as it fires. |
| **H4** | the ladder's SHAPE: for whichever rungs clear H1–H3, the `c = 0` component moves by MORE than the total | no bar; reported | 83 % of `binin`'s cost is at `c = 0`, so a fix for that cost must appear there. **If a rung gains and its gain is NOT at `c = 0`, the mechanism is not the one in §2.2.** |
| **H5** | `io_mall` vs `if_anchor` | ≥ 2σ | **mall BETTER.** The frozen-feature probe put the all-layer readout at −0.019/−0.022 bpc on *frozen* features, and training can only help. |
| **H6** | `io_mall` vs `io_wide` — **the decisive one** | ≥ 2σ | **mall BETTER.** `EXP_014` says width is worth more than every architectural arm this project has built; this is the first arm given the chance to beat it at matched parameters and matched tree. |
| **H7** | `io_mall`'s gain is NOT beyond the horizon | \|beyond\| < 0.00922 | §2.4's mechanism. A multi-layer readout adds no temporal information. |
| **H8** | realised wall-clock ratios | none; reported | pre-flighted in §2.7 at 1.00 / 1.18 / 1.27. |

**H4 and H7 are shape claims and carry no adoption weight.** They exist so that
a rung that moves the mean for the wrong reason can be caught, which is what
`EXP_020` §9.5 showed the per-context statistic cannot do on its own.

**`n_contexts_significantly_worse` IS NOT USED AS A BAR ANYWHERE IN THIS
EXPERIMENT.** `EXP_020` §9.5 measured that statistic against a **bitwise copy of
the anchor** and it read 120 of 128 contexts "significantly worse" — bars up to
80× below single-seed variation at n = 3. It is reported as a marker and
nothing here turns on it. That is fresh evidence for open decision #2 and no
redefinition is proposed.

---

## 5. Decision rule, fixed now

| outcome | what is written in §9 |
|---|---|
| H1/H2 hold and H4 holds | **"the binarity cost is a density cost, and it is paid at `c = 0`"** — the input boundary can be closed at a price this ladder locates |
| H1/H2 hold and H4 fails | **"sparsity helps and §2.2's mechanism is not why"** — reported with the decomposition and referred |
| H1/H2 fail | **"the 0.0448 is a cost of binarity, not of density"** — the ladder is flat and I1's last exception is priced at its measured value |
| any rung LOSES to `if_binin` beyond 2σ | reported as the confound §2.3 names, and **this experiment may not say which axis caused it** |
| H5 and H6 hold | **"the readout is worth its parameters, and worth more than width"** — referred for adoption, never ranked here |
| H5 holds and H6 fails | **"the readout is worth its parameters and width is worth the same"** — a null against `EXP_014`, not against the readout |
| H5 fails | **"`EXP_020` leg 5's floor was the answer, not the instrument"** |

Every cell above is written before the first run and none is softened afterwards.

---

## 6. Known limitations, stated in advance

1. **The density ladder confounds density with plasticity** (§2.3). The
   separating axis `iface_code_sd` is implemented and NOT RUN: **~0.35 GPU-h.**
2. **Leg A is run only at `d = 512, K = 2`.** `EXP_015` is the standing reason a
   735K result need not survive to 5.0M, and nothing here is run there.
3. **`q` is an INIT density, not a trained one.** `shadow` is a parameter, so the
   code's realised density drifts. §9 reports the **final measured density of
   every rung** as a diagnostic — if all four converge to the same density the
   ladder measured an init and not a code.
4. **The gain is a buffer fixed at construction from the init density.** If a
   rung's density drifts far, its gain no longer matches its own second moment.
   Recomputing it during training would be a second mechanism; it is not done,
   and item 3's diagnostic is what would reveal it.
5. **Leg B is unmatched against `if_anchor` on purpose.** `io_mall` carries
   14.3 % more parameters than the anchor, so `io_mall − if_anchor` is not a
   single-variable comparison and is not read as one; **`io_mall − io_wide` is**,
   and H6 is the decisive bar for that reason.
6. **`io_wide` is 738 parameters under `io_mall`** (0.088 %). Realised counts are
   reported by G2; no identity is claimed.
7. **n = 3 per cell.** The bars are 2σ on a transferred σ. Effects between 1σ and
   2σ will read UNRESOLVED and are reported that way rather than promoted.
8. **The frozen-feature probe's numbers are not a prediction of a trained arm.**
   They priced leg B on frozen features and authorised it; H5's bar is on the
   trained result and does not inherit the probe's figure.

---

## 7. Gates — aborting, not advisory

| gate | what it asserts | on failure |
|---|---|---|
| **G1** | adopted arms untouched: `git diff --stat` empty over `neuron.py`, `kernels.py`, `surrogate.py`, `twocomp.py`, `twocomp_detach.py` | abort the ladder |
| **G2** | realised parameter count **equals** the closed form, exactly, for all 15 runs | abort that run |
| **G3 / K1** | one field changed per arm against its named reference in §3; anything outside `may_differ` aborts | abort the ladder |
| **G4** | `iface_thr_in = 0.0` nests `if_binin` **by construction** — the q = 0.50 rung is a reused run and not a re-derivation | asserted in `tests/test_interface.py` |
| **G5** | the frozen recipe read back out of each run's own `config.json` | abort that run |
| **G6** | peak VRAM alarm (pre-flight max 0.693 GiB against a 0.609 GiB baseline) | report, do not abort |
| **G7** | divergence read from `log.jsonl`, **not** the exit code | record, evaluate anyway, continue |
| **G8** | §7.2 screen run **before** the first step — §2.6 | abort if any candidate is a saddle with no fallback |
| **G9** | mutation lockfile absent | refuse to start |

---

## 8. Entry conditions

* Pre-registration SHA-256 stamped into `exp_022_run_manifest.json` **before**
  the first run and re-checked after the last.
* `scripts/exp/022_input_output_results.py` committed **before any bpc is read.**
* **Budget, from §2.7 and not guessed:**
  9 × 398.6 s + 3 × 472.4 s + 3 × 508.3 s = **6,529 s = 1.81 GPU-h** of training,
  plus evaluation, the horizon probe over the new checkpoints and the
  generalisation-gap probe. **~2.2 GPU-h budgeted.**
* Nothing else runs on the GPU while a timed run is in flight.

---

## 9. Results

*(appended after the run; nothing above this line is rewritten)*
