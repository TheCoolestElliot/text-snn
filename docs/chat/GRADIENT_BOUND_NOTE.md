# A note on the chat model's gradient bound, and the timescales it traded away

**Status: a measurement, plus inference that is labelled as such.** It was found
while looking for why every round since 2026-08-06 has left `chat-v3d-aligned`
shipped — `experiments/chat/SHIPPED` records four of them declining to displace
it, and `QUALITY_v12.md` records another where *"the primary bar passed, and the
round does not ship"*. It has no control arm and it establishes no cause. It is
written here, outside `docs/reports/`, because `snnchat` is outside the
Phase-1..5 protocol and because acting on any of it is Elliot's call.

It is the sibling of [`WEIGHT_DECAY_NOTE.md`](WEIGHT_DECAY_NOTE.md), which asked
what the regulariser does to the slow pole. This asks what the *backward pass*
does to it — a question that did not exist when that note was written, because
`EXP_016` had not yet been run.

`docs/chat/CONVENTIONS.md`'s rules apply to everything read out of this file:
denominators are printed, and a contrast between two training trajectories is
**n = 2**, not n = 6.

---

## 1. The question

`EXP_016` (`docs/reports/04_phase4_interim.md` §15) proved that both of this
project's neurons thread an adjoint over all `L` timesteps with the same per-step
multiplier `beta_f * dv`:

```
plain LIF   dv = (1 - s)  - v_pre * sg(v_pre - thr)     SAME variable twice
twocomp     dv = (1 - sh) - vf    * sgd(v_pre - thr)    DIFFERENT variables
```

The plain LIF's is self-limiting — the surrogate decays exactly as fast as the
membrane grows — and at the frozen constants `max|beta_f * dv| = 0.5123596 < 1`
**for every finite membrane at every width**. The two-compartment neuron breaks
that because the spike test is on the mixed membrane `v_pre = vf + w*vs` while
the reset lands on `vf` alone and **`vs` is never reset**. The crossover is at
`|w*vs| ~ 1`.

`EXP_015` is the run that died of the difference: NaN at step 5138,
deterministically, at 5.0 M parameters, where the plain LIF and the GRU at the
identical width, seed, recipe and tree trained cleanly. `EXP_017` is the fix, and
`EXP_025`'s H3 held at **6 of 6 runs completing 20,000 steps with zero
non-finite losses, 3/3 at each rung, 95 % CI [0.610, 1.000]** (§22.6). Quote it
that way rather than as "6/6 at 5.0 M": the six span **two** widths, 735 K and
5.0 M, so the 5.0 M evidence inside it is 3/3 and its interval alone is wider.
`04_phase4_interim.md` §23.4's one-line compression of this is the phrasing to
avoid inheriting.

**The chat model is a two-compartment arm at 4,417,637 parameters** — `d = 1024`,
`K = 4`, `arch = "twocomp_threshold"`. Nobody had asked which side of that line
it is on.

## 2. What was measured

`scripts/chat/reset_jacobian_probe.py` re-implements the scan locally so it can
emit `g = beta_f * dv` per timestep, then checks the local scan against
`snn.twocomp.twocomp_scan_eager` **bitwise on the spikes and on the final state**
before believing any statistic it produces. That check (`t2_holds`) passes on
every row below; without it none of this would be worth reading.

Six checkpoints, `ckpt_best.pt`, batch 0, `L = 256`, `B = 160`, 41,943,040 sites
per layer. Artifacts:
`experiments/chat/_probe/gradient_bound/jacobian_<run>.json`.

| run | max\|β·dv\| | × the LIF ceiling | max\|w·vs\| | layers over 1.0 | longest consecutive run over 1.0 | best contiguous window |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `chat-v2` | 13.881 | **27.1×** | 14,913 | 4/4 | 9 | 10^+1.91 |
| `chat-v2-anneal` | 9.593 | 18.7× | 15,310 | 4/4 | 8 | 10^+1.64 |
| **`chat-v3d-aligned`** (SHIPPED) | 9.776 | **19.1×** | **14,454** | **4/4** | 8 | 10^+1.89 |
| `chat-v6-inst-a-s1` | 14.163 | 27.6× | 17,135 | 4/4 | 7 | 10^+1.71 |
| `chat-v12-rare` | 10.308 | 20.1× | 19,684 | 4/4 | 8 | 10^+1.75 |
| `chat-v6-scratch` | 2.543 | 5.0× | 594 | 4/4 | 5 | 10^+0.97 |

> **Every checkpoint on disk is inside the region `EXP_016` proved unbounded, at
> every layer.** The shipped model's worst layer runs at **19.1×** a ceiling the
> plain LIF provably cannot exceed at any width, and its `|w·vs|` reaches
> **14,454** against a crossover at ~1 — **20.8×** the largest value `EXP_016`
> measured anywhere on the research side.

That comparison needs its denominator named, because the obvious source for it is
wrong. `04_phase4_interim.md` §15.2's table quotes `max|w·vs| = 279.66` for
`twocomp` at `d = 1481`; that row is **layer 0's**, consistent with the `max|g|`
column beside it, and `exp_016_reset_jacobian.json` records **696.278 at layer 1**
of the same run. 696.278 is the largest value in that artifact and is the figure
used above. The two measurements are also not sampled identically — `EXP_016` read
48,529,408 sites per layer at `d = 1481`, `B = 128` on enwik8; this reads
41,943,040 at `d = 1024`, `B = 160` on the chat mixture — so the ratio is
indicative of scale and is not a controlled comparison.

**Read the last two columns before drawing the obvious conclusion.** The best
contiguous window is ~10^1.9, i.e. ~80×, and the longest consecutive expanding
stretch is 5–9 timesteps. `EXP_016`'s dying run reached **85 consecutive
timesteps worth 10^16.00** on the batch it actually died on. So the mechanism is
present and active at every layer of every chat checkpoint, and **it is not
currently runaway on the batch measured**. What is established is that the
guarantee the plain LIF carries does not exist here, not that a divergence is
imminent.

`016_posthoc_window.py`'s caveat is inherited rather than restated as a finding:
an expanding window is a **necessary, not sufficient** condition for the failure,
because the adjoint is injected at every timestep.

### 2.1 It is not an artifact of reading batch 0

`EXP_016` §15's post-hoc records that its own Leg A read batch 0 and calls that
*"a defect in its design, not only in P3's quantity"* — the mechanism looks very
different on the batch a run actually dies on. `MixtureSampler.batch(s)` is a
pure function of `(seed, s)`, so any `s` below `max_steps` names a batch
`chat-v3d-aligned` really saw. Four of them:

| batch | L0 max\|β·dv\| | × ceiling | max\|w·vs\| | T2 |
| ---: | ---: | ---: | ---: | :---: |
| 0 | 9.77616 | 19.08× | 14,454 | ✓ |
| 1,000 | 9.77616 | 19.08× | 14,746 | ✓ |
| 5,000 | 9.78015 | 19.09× | 15,392 | ✓ |
| 13,999 | 9.77128 | 19.07× | 14,560 | ✓ |

Layer 0's multiplier is the same to three decimal places on all four. This is a
property of the weights, not of a batch — which is what the mechanism predicts,
since `dv`'s bound depends on `|w·vs|` and `w` is a parameter. Artifacts:
`jacobian_chat-v3d-aligned_batch<N>.json`.

### 2.2 And it is **not** currently producing a pathological gradient

This is the section that constrains everything else in the file, and it is a
negative result.

`scripts/chat/estimator_gradient_compare.py` loads **one** set of weights — the
shipped checkpoint's — into both arches, runs the same batch through both, and
compares what `backward()` hands the optimiser. Artifact:
`estimator_gradient_chat-v3d-aligned.json`.

| | batch 0 | 1,000 | 5,000 | 13,999 |
| --- | ---: | ---: | ---: | ---: |
| loss, bit-identical between arches | ✓ | ✓ | ✓ | ✓ |
| ‖g‖ ratio, undetached / detached | 0.906 | 0.902 | 0.902 | 0.913 |
| max\|g\| ratio | 1.140 | 0.554 | 0.525 | 1.032 |
| ‖∂w‖ ratio | 0.898 | 0.907 | 0.877 | 0.941 |
| non-finite gradients, either arch | 0 | 0 | 0 | 0 |

> **The undetached gradient is ordinary.** Its global norm is ~0.20 and sits
> **within 10 % of the detached one on every batch — and is the *smaller* of the
> two.** No component is non-finite. Whatever the 19.1× multiplier is doing at
> 0.05–0.3 % of sites for runs of 5–9 timesteps, **it is not reaching the
> parameter gradient at this point in weight space.**

That is `EXP_016`'s P3 verdict arriving from a second direction, and it should be
read as a refutation of the obvious inference rather than a footnote to it: **at
the shipped model's converged weights, on four real training batches, there is no
gradient pathology to fix.** The best contiguous window in §2 is ~80×; the
adjoint is injected at every timestep and the expanding chains are a small
minority, so ~80× on a minority of chains does not move a norm.

**The loss being bit-identical on all four batches is also the strongest
available check that the two arches are the same network** — stronger than
`test_chat_detach_forward_is_bitwise_the_shipped_arm`, which asserts it on a
32-channel toy at random weights. Here it holds on 4.4 M trained parameters.

**What survives.** Not "the chat model's gradient is exploding" — §2.2 says it is
not. What survives is narrower and is still a fact:

* **the guarantee is absent.** The plain LIF carries a proof that its per-step
  multiplier cannot exceed 0.5123596 at any width; this arm carries none, and is
  measured at 19.1× that value. An absent bound is a property of the arm, not of
  a batch, and §2.1 shows it does not move;
* **it has bitten once**, at step 4,859 of `chat-v6-inst-a-s1` (§3 item 1);
* so the exposure is a **tail risk under distribution shift** — a higher learning
  rate, a longer training window, a larger model, an unlucky batch — and not a
  steady-state cost. Every one of those shifts is a thing someone might want to
  do to this model, which is the only reason any of this matters.

## 3. Three facts that were already in the tree and had not been put together

1. **A chat run has already diverged.** `experiments/chat/chat-v6-inst-a-s1/stdout.log:70`:
   `!! loss nan at step 4859: rolled back to step 4000, sampler seed -> 2, lr x0.5`.
   `BUILD_NOTES.md` line 465 records the consequence — that run trained 10,000 of
   its 14,000 steps at half the peak learning rate, which is the round's largest
   single problem and the reason `QUALITY_v6.md`'s seed-variance claim does not
   survive as a measurement of seed variance.

2. **`snnchat.train.Trainer._recover_from_divergence` exists to absorb exactly
   this**, and its own docstring says it is *"a recovery, not a diagnosis, and it
   is only defensible because this is not an experiment."* That was the right
   call at the time. It is a different sentence once the mechanism has a name.

3. **The learning rate was inherited, not chosen.** `BUILD_NOTES.md` line 449:
   `chat-v6-scratch` ran at lr 5e-4 with warmup 500 — *the fine-tune recipe
   applied to a from-scratch run* — because `PREDICTION_v6.md` §2 copied
   `chat-v3d-aligned`'s command line and deleted `--init-from`. The only other
   fresh-init run, `chat-v2`, used **2e-3**. *"A 4× lower peak LR, and nothing
   tested whether 5e-4 is sensible from scratch at 84,000 steps."* And the eval
   trace **was still falling when the cosine ran out** (weighted 1.2440 at 78,000
   → 1.2399 at 84,000) — the arm was not converged.

## 4. What the model did with the timescales it was given

`scripts/chat/slow_channel_census.py`, same checkpoints, same batch. Artifacts:
`experiments/chat/_probe/gradient_bound/census_<run>.json`.

`spread_slow_poles` hands every layer a log-uniform spread of `tau` from **3 to
600** characters. Layer 0 afterwards:

| run | tau median | fraction below tau = 8 | input gain `exp(-thr_log)` median | mean\|cur\| before → after | pinned channels (all layers) |
| --- | ---: | ---: | ---: | ---: | ---: |
| `chat-v2` | 2.71 | 89.0 % | 18.19 | 0.403 → 8.026 | 141/4096 |
| `chat-v2-anneal` | 2.70 | 89.0 % | 18.94 | 0.385 → 7.987 | 138/4096 |
| **`chat-v3d-aligned`** | **2.67** | **88.9 %** | **20.93** | **0.372 → 8.617** | 112/4096 |
| `chat-v6-inst-a-s1` | 2.68 | 89.0 % | 20.40 | 0.372 → 8.314 | 114/4096 |
| `chat-v12-rare` | 2.68 | 88.9 % | 20.87 | 0.372 → 8.633 | 109/4096 |
| `chat-v6-scratch` | **12.87** | **35.8 %** | **3.75** | 0.942 → 3.622 | 22/4096 |

*"Pinned"* means the channel fires on under 1 % or over 99 % of timesteps — it
emits nearly the same bit at every position, so it carries nearly nothing about
its input. The denominator is `4 layers × 1024 channels`.

**The instrument reproduces a committed chat artifact it was not built against.**
[`RESULTS.md`](RESULTS.md) reports `chat-v2-anneal`'s slow poles pooled over all
four layers — median 5.8, max **8,506**. This script reads them per layer and
gets medians of 2.70 / 13.22 / 10.07 / 7.32 and a layer-0 max of **8,507.7**, a
pooled median landing where those four put it. The two were computed by different
code from the same checkpoint, so the agreement is a check on this script rather
than a restatement.

**`tau_max` is only good to ~1 part in 7,000 and the 1.2 gap is that, not a
disagreement.** `beta_s_raw` is fp32, and `tau = 1/(1 − sigmoid(raw))` amplifies
its error by `tau²`: at `tau ≈ 8,500` one ulp of `beta_s` is worth **≈ 4
characters** of `tau`. Reading the same checkpoint on CPU rather than CUDA gives
**8,506.5** against the GPU's 8,507.7 — one ulp, from `sigmoid` rounding
differently on the two devices. So quote `tau_max` at this end as "about 8,500",
never to the character. The medians, which sit where the sigmoid is not
saturated, do not have this problem.

**Do not cross-check any of this against `summary.json`'s `description` block on
a run that predates 2026-08-17.** `snnchat.train.Trainer._refresh_description`
records the defect and its fix: until that date the description was re-read only
when weights were *loaded*, never after they were *trained*, so **23 chat
summaries report their shared parent's step-7,874 profile as their own**.
`chat-v3d-aligned`'s summary is one of them — it reports layer 0 at max 8,507.7
where its own checkpoint holds **7,928.8**. Every figure in this file is read off
the checkpoint directly and is unaffected. The v13 runs
([`PREDICTION_v13.md`](PREDICTION_v13.md)) are post-fix and their summaries
describe themselves.

**Rows 1–5 are one trajectory, not five.** `chat-v6-inst-a-s1` and
`chat-v12-rare` were launched with
`--init-from experiments/chat/chat-v2-anneal/ckpt_best.pt` (their drivers'
`stdout.log`), `chat-v3d-aligned` records the same parent in
[`RESULTS.md`](RESULTS.md), and `chat-v2-anneal` is `chat-v2`'s anneal. The table
shows how little any of them moved layer 0 — the five rows agree to two decimal
places on a quantity that ranges over three orders of magnitude. **The effective
n of this contrast is 2.**

Two things happened together on the trajectory that produced the shipped model,
and neither happened on the one that did not:

* **layer 0 gave up its long timescales.** 89 % of its channels ended below
  `tau = 8`, against an initialisation whose *minimum* was 3 and whose median was
  ~42. This is not the weight-decay collapse `WEIGHT_DECAY_NOTE.md` measured:
  `beta_s_raw`, `w`, `thr_log` and the biases have been **exempt from decay since
  `chat-v2`** (`ChatTrainer._param_groups`), and that note's §5.1 measured the
  gradient, unopposed, *lengthening* time constants (median 23.69 at
  `chat-v2` step 5,000, max **778.6**, above its own initialisation of 600).
  Between there and the shipped checkpoint the layer-0 median fell to 2.67.
* **layer 0 blew up its input gain instead.** `exp(-thr_log)` reached a median of
  20.9, taking `mean|cur|` from 0.372 to 8.617 — a **23× amplification** of what
  the un-reset slow compartment integrates. `|w·vs|` is that integral times the
  mix, which is the quantity §2 measures at 14,454.

The long-timescale channels that survive are the ones in trouble. In the shipped
model's layer 0, the 18 channels above `tau = 128` (1.8 % of the layer) have
median firing rates of **0.0075 and 0.0044** and median `|w·vs|` of **5.3 and
36.5**. That is `EXP_006`'s finding — ~3 % of channels carry the horizon, and
clamping them to the median is *worse than the Phase-2 baseline* — arriving with
a new detail attached: **on this model those channels are near-silent and they
are the ones outside the bound.**

## 5. Inference, and it is inference

* **A depressed learning rate is what a gradient explosion looks like from the
  outside once someone has tuned around it.** The chat model runs at 4× the
  research recipe's caution, its trainer carries a rollback for NaN, one run used
  it, and its backward has no bound. Those four facts are consistent with one
  cause. **They do not establish one**, and nothing here has varied the learning
  rate with the arm held fixed. **§2.2 is evidence against the strong form of
  this**: at the converged weights the two estimators' gradients differ by under
  10 % in norm, so whatever is limiting the learning rate is not visibly present
  in the gradient at the operating point. The weak form — that the *tail* is
  fatter without a bound, and that a rollback fires on the tail — is untouched by
  §2.2, because §2.2 sampled four batches and the tail is what four batches do
  not sample.
* **`BUILD_NOTES.md` line 449 already offers a competing and simpler
  explanation** for the low learning rate, and it should be preferred until
  something displaces it: 5e-4 was **copied from a fine-tune command line**, not
  chosen. No gradient pathology is needed to explain a value nobody picked. The
  two explanations are not exclusive — an untested value can also be the value
  that happens to be safe — and neither is established here.
* **`chat-v6-scratch` is a contrast in mechanism, not a verdict on quality.** It
  scored **0.2679** against the shipped model's 0.2969 on the pre-registered
  headline (`QUALITY_v6.md`), i.e. *worse*, and it was not converged. It is cited
  here for one reason only: it shows that this recipe does not *have* to land at
  `tau = 2.7` and a 23× input gain, so the shipped model's state is a property of
  its trajectory rather than of the architecture.
* **The trade may be the model doing the right thing.** A short slow pole with a
  large input gain is a real strategy, and this corpus's replies are short. There
  is no measurement here that says the long-timescale configuration would score
  better. `EXP_006` establishes the tail is *load-bearing where it exists*; it
  does not establish that more of it is better.

## 6. What is now buildable that was not, and what it is not

`EXP_017`'s detached-reset estimator sends `∂vf_out/∂v` to zero, which removes
`vf` from **both** entries of the 2×2 per-step Jacobian, leaving
`diag(beta_f·(1 − s), beta_s)` — diagonal, so no non-normal transient, and
bounded by `beta_f = 0.5`. **Strictly tighter than the plain LIF's own
0.5123596, and independent of `w`, of `vs`, and of width.**

`snnchat` could not express it: `build_chat_model` admitted `twocomp_threshold`
and `twocomp` and raised on everything else, and no detached variant of the
composed threshold arm existed anywhere in the tree.

`snnchat.model.TwoCompThresholdDetachCharLM` (2026-08-22) is that variant,
`arch="twocomp_threshold_detach"`, **opt-in and off by default**. What it is:

* **the same network.** `snn.twocomp_detach` imports `twocomp_forward_kernel`
  from `snn.twocomp` rather than re-declaring it, so under `eval()` the two
  arches are bit-identical — asserted at `== 0.0` by
  `tests/test_snnchat.py::test_chat_detach_forward_is_bitwise_the_shipped_arm`;
* **the same state dict.** No extra key, no missing key, no translation. A
  checkpoint trained through it loads into
  `snn.model.build_model(arch="twocomp_threshold")` and evaluates bit-identically
  (`test_a_detached_chat_checkpoint_still_loads_as_the_research_arm`), so the
  claim `snnchat` has always made about not forking the architecture still holds;
* **no parameter, no kernel, no hand-written backward, no inference cost**
  (`test_the_detached_arch_adds_no_parameter`);
* **a different gradient, and only that** — asserted, because an arch that
  produced the identical gradient would be doing nothing
  (`test_the_two_arches_disagree_on_the_gradient_and_only_there`).

Nothing under `src/snn/`, `experiments/runs/`, `experiments/logs/` or
`docs/reports/` was modified. The class lives in `snnchat` for the reason
`snnchat/__init__.py` gives: this package depends on the research package one way
only, and adding an `arch` value to `snn.model` that no pre-registration has
asked for is not that.

**MEASURED 2026-08-24, AND IT COSTS BITS.**
[`QUALITY_v13.md`](QUALITY_v13.md) ran the round §7 item 1 below asks for.
**P1 fired WORSE: +0.0035847 bpc, +2.85 SE against a ±0.0025182 bar**, worse on 6
of 7 sources, with the control inside its pre-registered guard band and zero
rollbacks in either arm. P2 and P3 found the bounded arm sitting at **20.26×** the
LIF ceiling against the control's 19.95× and doing the same thing with its layer-0
timescales to three significant figures. **So the arm is not free and it buys
nothing observable at this operating point.** §7's items 2 and 3 were gated on P1
and are **not run**. What survives is the tail-risk case in §5, which P1 does not
touch — and the honest label on the arch is now "insurance whose premium has been
measured at 0.0036 bpc and whose payout has not".

**What it is not.** The gradient is **biased**: the pathway *"firing now lowers my
own future membrane"* is dropped from the backward. `EXP_017` H3 measured what
that costs in bpc at 735 K on the research corpus and `EXP_025` measured it at
5.0 M. **Neither figure transfers to this corpus at this size and neither is
quoted here.** Whether it costs anything on the chat model is unmeasured. That is
why it is opt-in, why `twocomp_threshold` remains the default, and why nothing in
this file recommends training with it.

## 7. What it would take to establish any of this

Nothing in §5 is settled by anything in §2 or §4. The measurements that would
settle it, in the order that makes each one worth running:

1. **Does the arch cost bits here?** Two runs on `chat-v3d-aligned`'s exact
   recipe, `arch` the single variable, held-out weighted bpc as the outcome —
   which is seed-independent and sampler-free, so it does not have to fight
   `QUALITY_v6.md` §8's 0.099 headline swing. This is the cheap one and it is
   the gate on everything below: if the bias costs bpc at this size, the rest
   does not arise.
2. **Does the bound buy back the learning rate?** The same recipe at 2e-3 —
   the value `chat-v2` used and `BUILD_NOTES.md` line 449 says was never
   re-tested — on both arches. The pre-registered outcome is not bpc but
   **whether the run completes without a rollback**, and the undetached arm at
   2e-3 is the arm expected to fail. Note in advance that `chat-v6-scratch` at
   5e-4 was **still improving at 84,000 steps**, so a run that merely finishes
   proves nothing about convergence.
3. **Does anything change what layer 0 does with its timescales?** §4 re-run on
   whatever comes out of (1) and (2). This is the question the note is actually
   about and it is third because the first two are prerequisites.

Each needs its bars written before it runs and its direction stated including the
possibility that it is worse — `WEIGHT_DECAY_NOTE.md` §4's standing warning
applies here unchanged: there is a real chance the current configuration is the
model doing something sensible, and that removing the pressure produces a network
that is worse than either.

## 8. Reproducing everything above

```bash
python scripts/chat/reset_jacobian_probe.py --run chat-v3d-aligned \
    --out experiments/chat/_probe/gradient_bound/jacobian_chat-v3d-aligned.json
python scripts/chat/slow_channel_census.py --run chat-v3d-aligned \
    --out experiments/chat/_probe/gradient_bound/census_chat-v3d-aligned.json
```

Both are inference-only, take under a minute each, train nothing, and write
nothing outside the path given to `--out`.
