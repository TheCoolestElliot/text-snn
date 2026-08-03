# EXP_006 — Do ~3 % of channels carry the horizon?

**Pre-registered:** 2026-08-03, before any ablated number was computed.
**Status:** OPEN. Results go in §9.
**Phase:** 4 (controlled experiments). The causal question `EXP_004` §10.5 left
open and §10.11 item 4 called "the highest-information single run available".

**A note on which phase this belongs to.** `EXP_004` §10.5 calls this "a Phase-5
experiment". `01_reconnaissance.md` §8.2 defines Phase 5 as *final training and
docs* and Phase 4 as *pre-registered, single-variable, ≥3-seed controlled
experiments*, which is what this is. It is filed as Phase 4 and the
inconsistency is recorded here rather than resolved by relabelling `EXP_004`.
Nothing about the design depends on which name it carries.

---

## 1. Why this experiment

`EXP_004` §10.5 found that `beta_s`, initialised at 0.95 for every channel, went
**bimodal**: about 3 % of channels per layer kept `beta_s > 0.9` (time constants
18–26 characters) and the other 97 % pulled their "slow" pole down to ~0.5–0.6.
`EXP_005` confirmed the arm's horizon at n = 7 (median 47, per-seed 38–57)
against the baseline's 7.

The inference §10.11 item 4 draws from that is the most actionable thing in
Phase 4: *if those ~15 channels per layer carry the horizon, the capacity this
project devotes to long memory is 3 % of its width and is nowhere near
saturated.* Two things stop it being usable as written:

* **It is correlational.** The experiment did not intervene. `EXP_004` §10.5
  says so, and adds the fact that argues against the naive reading: `|w|` on the
  tail channels is **not** larger than on the rest (0.99× in layer 0, 1.16× in
  layer 1), which is not what one would expect if those channels were carrying
  the load.
* **It says nothing about bits.** A channel set can carry the *horizon*
  statistic and carry very few *bits*, and `EXP_005` §9.6 makes that live: the
  beyond-horizon component is now only 28 % of the remaining gap while the
  within-reach component is 51.8 %. Whether the horizon is where the bits are is
  a separate measurement and this experiment can make it.

The whole thing runs on the seven checkpoints already on disk. No training, no
new kernel, no new parameter, no gradient gate, and no σ that has not already
been measured.

### 1.1 Why the intervention is on `beta_s` and not on `w`

The obvious ablation — zero `w_c` on the tail channels — is **confounded**, and
the confound is `EXP_004` §10.6's own algebra. At zero context both compartments
start from zero, so

```
v_0 = vf_0 + w_c·vs_0 = cur_0 + w_c·cur_0 = cur_0·(1 + w_c)
```

and `w_c` is therefore a *learned per-channel threshold* as well as a mix.
Zeroing it removes the channel's long memory **and** its threshold shift at
once, and §10.6 attributes 40.6 % of the arm's whole gain to the threshold
effect. An ablation that moves both cannot answer a question about either.

`beta_s` moves only one of them. The expression above has no `beta_s` in it —
`vs_0 = 0·beta_s + cur_0 = cur_0` for any finite decay — so **clamping `beta_s`
cannot change anything the network computes at zero context, exactly and
bitwise.** That gives the experiment a self-check it did not have to invent
(§6, H2) and an intervention that isolates timescale from threshold.

Every intervention below therefore writes to `beta_s_raw` and to nothing else.

---

## 2. Design

Seven trained checkpoints (`twocomp_s0…s6`), inference only. Per layer, all
rankings and the clamp target are computed **once from the intact checkpoint**
and reused at every `N`, so no intervention's ranking is contaminated by another
intervention's edit (failure mode L6).

`clamp(c) := beta_s[c] ← median(beta_s of that layer, intact)`, applied by
setting `beta_s_raw[c] ← logit(median)`. The median is ~0.548 in layer 0 and
~0.573 in layer 1, and is the population the 97 % actually chose.

| Arm | Intervention | N |
|---|---|---|
| **A(N)** | clamp the top `N` channels **ranked by `beta_s` descending** | 0, 4, 8, 16, 32, 64, 128, 512 |
| **B(N)** | clamp **every channel except** the top `N` by `beta_s` — keep only the tail | 16, 64 |
| **C(N)** | clamp the top `N` channels **ranked by `\|w\|` descending** | 16, 64 |
| **E(N)** | `N` channels drawn at random from those with intact `beta_s < 0.8`, each shifted **down by the mean displacement A(N) applied in that layer**, clamped to (0.01, 0.99) | 16 |

`A(0)` is the intact model and is the reference every other row is compared
against. `A(512)` clamps every channel to the median — the arm with no bimodality
left at all.

**Why these four and not more.** `A` is the dose–response curve and answers "how
many channels". `B` is its converse and is what turns a necessity result into a
dissociation. `C` asks whether a different selection rule — the mix magnitude,
which §10.5 noted is uncorrelated with the tail — picks out the same behaviour.
`E` asks whether it is the *identity* of the channels or merely the *size* of the
parameter change, because clamping a tail channel moves `beta_s` by ~0.35 while
clamping a median channel moves it by ~0, and a control that does not match the
displacement is not a control.

91 configurations. At `EXP_005`'s measured 15.6 s per twocomp checkpoint for the
full k-sweep this is ~0.45 GPU-hours.

### 2.1 What is measured

1. **The horizon**, per seed, per configuration, via
   `scripts/exp/001_memory_horizon.py`'s own `paired_context_curve` and
   `horizon_from_excess` — imported, not reimplemented (failure mode L5).
2. **Whole-split fresh test bpc**, per seed, per configuration: the probe's
   `k = 256` point over the full 4 980 736-character test split.
3. **The paired excess curve**, c = 0…127, per seed, per configuration.
4. The overlap between the top-16-by-`beta_s` and top-16-by-`|w|` channel sets.

### 2.2 What is deliberately NOT measured

**Carried bpc.** The probe produces the `fresh` protocol by construction and
`U4`'s threshold is stated in fresh bpc accordingly. Running `scripts/evaluate.py`
as well would add a second instrument for one number and invite the two to
disagree; if the fresh answer is close to the threshold, that is a reason to run
carried afterwards and say so, not a reason to run it now.

**Any re-training.** Nothing here shows what a network *trained* with more slow
channels would do. See §5 item 1 — that limitation is the point of this
experiment's scope, not an oversight in it.

---

## 3. Hypotheses

All horizon thresholds are evaluated at the **inherited 2σ = 0.00922 tolerance**,
at which the intact arm's committed per-seed horizons are 38 / 47 / 45 / 57 / 48
/ 47 / 53 (median 47) and the baseline's are 7, 7, 7, 7, 7. The
two-compartment family's own bar from `EXP_005` (2σ = 0.00627) is reported beside
every number but is not what the thresholds are set against, because the
published 7-and-47 that make these thresholds interpretable were measured at
0.00922. `EXP_005` §9.4's warning applies: the horizon is defined against a
tolerance, so it moves when the tolerance moves. Every comparison here is
**within-seed, at one fixed tolerance.**

| # | Prediction | Threshold |
|---|---|---|
| **U1** | **The tail carries the horizon.** Clamping the top 16 channels per layer (3.1 % of width) removes at least half the horizon: median horizon at `A(16)` ≤ **27** | midpoint of the baseline's 7 and the intact arm's 47 |
| **U2** | **The tail is sufficient as well as necessary.** Keeping only the top 16 slow poles and clamping the other 496 leaves median horizon at `B(16)` ≥ **27** | the same midpoint, from the other side |
| **U3** | **The effect saturates.** \|median horizon `A(32)` − median horizon `A(512)`\| ≤ **3** characters | if it does not saturate, the capacity is distributed and "3 %" is the wrong description |
| **U4** | **The bits follow the horizon.** `A(16)` costs ≥ **0.05 bpc** of whole-split fresh test bpc against `A(0)` | ~40 % of the arm's 0.1240 bpc fresh gain. *Stated in the direction that flatters the plan this experiment exists to support* |
| **U5** | **Selection by `\|w\|` is not the same thing.** Median horizon at `C(16)` ≥ **38** | the lowest intact per-seed horizon |
| **U6** | **It is the channels, not the size of the edit.** Median horizon at `E(16)` ≥ **38** | the same |

**On U4, explicitly.** The reading `EXP_004` §10.11 item 4 invites — long-memory
capacity is 3 % of width and nowhere near saturated, so widen it — is flattered
by U4 holding and damaged by U4 failing. It is stated in the flattering
direction on purpose, so that a failure is a real result and cannot be presented
as one. **If U4 fails, the horizon is nearly free to remove in bits, and
horizon-widening should be demoted in the Phase-5 ranking regardless of what U1
does** — which would corroborate §10.11 item 3 (short-range fidelity, not
horizon, is where the remaining gap lives) from a second direction.

**On U1's power.** This is a within-seed paired intervention on n = 7, so the
comparison does not depend on the seed-to-seed σ that `EXP_005` measured. What it
does depend on is the horizon's own coarseness as a statistic: it is an integer
read off a curve crossing a fixed tolerance, and the intact arm's per-seed values
already span 38–57. A median shift of 20 characters is far outside that spread; a
shift of 5 would not be, and would not be claimed.

---

## 4. Decision rule, fixed now

| U1 | U2 | U5 ∧ U6 | Verdict |
|---|---|---|---|
| **holds** | **holds** | **hold** | **The horizon is carried by ~3 % of channels, causally.** §10.5's association is upgraded from correlational to causal at the level of `beta_s`, and a "widen the slow tail" candidate is added to the ranking — **priced against U4's answer, not against the horizon result.** |
| **holds** | fails | hold | **Necessary but not sufficient.** Both legs reported; the capacity claim is not made. |
| **fails** | either | — | **The association is not causal at N = 16.** The dose–response curve and `N½` (the smallest N at which half the horizon is lost) are reported, and `EXP_004` §10.11 item 4 gets an appended correction. |
| — | — | **either fails** | **The intervention is non-specific.** Nothing is concluded from U1, and this experiment reports a failed control rather than a result. This leg dominates the others. |

Two riders, also fixed now:

* **U4 fails** ⇒ recorded as the headline finding regardless of U1's outcome, and
  horizon-widening is demoted in the Phase-5 ranking. A mechanism that can be
  removed for less than 0.05 bpc is not where 0.3398 bpc of remaining gap is
  going to come from.
* **`B(16)`'s bpc is not interpretable as a cost** and will not be quoted as one.
  It clamps 97 % of channels; of course it is bad. Only its **horizon** is used,
  and only as the converse leg of U1.

---

## 5. Known limitations, stated in advance

1. **This ablates a trained network. It cannot establish headroom.** Showing that
   ~15 channels are necessary for the horizon in a network that chose to have ~15
   is not the same as showing that 50 would buy more. §10.11 item 4's "nowhere
   near saturated" needs a *training* arm, which this experiment prices rather
   than settles — and which it may argue against, if U4 fails.
2. **The clamp target is a choice.** Clamping to the layer's own median is
   defensible (it is where 97 % of the population sits) and it is fixed in
   advance, but a different target — 0.5, or `beta_f` — would give different
   magnitudes. The result is a statement about this intervention.
3. **The ablated configurations have no σ of their own,** and measuring one would
   mean retraining, which is the whole cost this experiment avoids. Every
   comparison is within-seed at a fixed stated tolerance, and no cross-arm
   horizon comparison is made.
4. **The horizon is a coarse statistic** — an integer where a curve crosses a
   tolerance. Small movements in it mean little; §3's power note says how small.
5. **`E`'s random draw is one draw**, seeded and recorded, not a distribution over
   draws. If `E(16)` lands near its threshold, the honest response is more draws,
   not a verdict.

---

## 6. Self-checks — aborting, not advisory

These are not hypotheses. Each one reproduces a number this harness did not
produce, and the run aborts if any fails.

| # | Check | Why it can fail |
|---|---|---|
| **H1** | Every intervention modifies `beta_s_raw` **only**. `w` and every other tensor are bitwise identical to the checkpoint | catches an ablation that silently moves the threshold as well as the timescale — the confound §1.1 exists to avoid |
| **H2** | The probe's **k = 1 point is bit-identical** to `A(0)`'s, for every intervention and every seed, residual < 1e-12 bpc | this is forced by algebra (§1.1: nothing at zero context depends on `beta_s`), so a non-zero residual means the intervention is not doing what this file says it does |
| **H3** | `A(0)`'s k = 256 bpc reproduces each run's committed `final_test.json` fresh bpc to < 1e-3 | `EXP_001`'s F1 — the check that caught a contaminated run in Phase 3 |
| **H4** | `A(0)`'s per-seed horizon reproduces `EXP_005`'s committed **38 / 47 / 45 / 57 / 48 / 47 / 53** exactly | catches a re-derivation of the horizon that is not the committed one |
| **H5** | The edit actually landed: exactly `N` entries of `beta_s_raw` differ per layer (512 − N for `B`), and for N ≥ 4 the k = 256 bpc differs from `A(0)`'s | a no-op ablation produces a perfect null and looks like a clean negative result |

---

## 7. Failure modes

| # | Failure | Detection |
|---|---|---|
| **L1** | The clamp is a silent no-op | H5 |
| **L2** | The clamp writes to the wrong tensor | H1, and H2 would fail for a `w` edit |
| **L3** | A checkpoint is contaminated (phase-3 report §8) | H3 on every intact run; no mutation campaign runs while this does |
| **L4** | Horizons compared across different tolerances | one tolerance fixed in §3; the second bar reported beside, never mixed into a threshold |
| **L5** | The probe is edited and stops being the instrument that produced every other horizon in this project | the statistic, the k-sweep and the F1 tolerance are **imported** from `scripts/exp/001_memory_horizon.py`; this experiment adds no entry to its `ARMS` registry and never overwrites a committed artifact |
| **L6** | Ranking or clamp target computed from an already-edited model, so N values contaminate each other | both computed once from the intact state dict and cached; every configuration is applied to a **fresh** copy of the checkpoint |
| **L7** | The random controls' draw is chosen after seeing a result | RNG seeded from a constant fixed in this file (`20260803`), recorded in the output |

---

## 8. Entry conditions

- [x] Existing test suite green — **242 passed**, matching `EXP_005` §8.1
- [ ] No mutation campaign running, and none started while this runs
- [ ] Nothing in `src/snn/` modified by this experiment — it is inference-only and
      edits a state dict in memory
- [ ] This file committed **before** the harness is written
