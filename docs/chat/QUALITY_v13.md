# QUALITY v13 — the bounded estimator costs bits and buys nothing visible

**Scored 2026-08-24 against the bars [`PREDICTION_v13.md`](PREDICTION_v13.md)
fixed before either arm was trained.** That file's sha256 is still
`027542a63da7db72c029e01d4f19759d7050dc4cbad0052ba99b01925f8f3107`, the value
stamped before launch, and `scripts/chat/score_v13.py` re-checks it on every run.

Not part of the Phase-1..5 research protocol. No number here is a reported
figure. **No research decision is ruled**; decision #4 is open and is Elliot's.
`experiments/chat/SHIPPED` is **unchanged**.

---

## 1. The scoreboard

| | verdict |
| --- | --- |
| **P1** — does the bounded estimator cost held-out bits? | **WORSE. It costs 0.0035847 bpc, +2.85 SE against a ±0.0025182 bar.** |
| **P2** — does it move the model out of `EXP_016`'s unbounded region? | **No.** 20.26× the LIF ceiling against the control's 19.95×; 4/4 layers over 1.0 in both |
| **P3** — does it change what layer 0 does with its timescales? | **No.** tau q50 2.676 vs 2.682, gain 20.99 vs 20.89, pinned 107 vs 111 |

> **The arm this round was built to test is not free, and the case for it does
> not survive its own first measurement.** `PREDICTION_v13.md` §6 gated
> everything downstream on P1 — *"if the estimator costs bits at 5e-4, running it
> at 2e-3 answers a question about an arm nobody would ship"*. P1 says it does.
> **The learning-rate and longer-window programme in `GRADIENT_BOUND_NOTE.md` §7
> items 2 and 3 does not follow from this evidence and is not run.**

Both arms completed 14,000 steps with **zero rollbacks**, 39.8 and 38.8 minutes.

## 2. P1, and its guard fired first

| | `chat-v13-ctl` | `chat-v13-det` | Δ (det − ctl) |
| --- | ---: | ---: | ---: |
| weighted, `eval_batches` = 40 (as run) | 1.1745948 | 1.1781794 | **+0.0035847** = +2.85 SE |
| weighted, `eval_batches` = 30 (matched) | 1.1749842 | 1.1785610 | **+0.0035768** = +2.84 SE |

**The guard passed before the bar was read.** §3 required the control inside
`[1.1721929, 1.1757542]` or P1 reports NOT RUN. It landed at 1.1745948
(+0.34 `SD_seed` from its own committed twin `chat-v3d-aligned`, +0.70 from the
n = 4 mean), and at matched `eval_batches` +1.14 `SD_seed`. In band on both
readings.

**The second row exists because the first row's settings were not the committed
ones.** The runs differ from `chat-v3d-aligned` in four fields — `eval_batches`
30 → 40, `eval_every` 3500 → 2000, `sample_every`, `log_every`. Three are pure
cadence and cannot touch the trajectory. `eval_batches` **changes the estimand**:
it decides how much of each held-out tail is scored. Δ is immune (both arms used
40) but the guard band came from 30-batch replicates, so both arms were
re-evaluated at 30. **Δ moves by 0.0000079** and the verdict is WORSE either way.

**Worse on 6 of 7 sources**, and the one exception is the smallest source in the
mixture:

| source | mix weight | Δ |
| --- | ---: | ---: |
| dolly | 0.06 | +0.0143181 |
| persona | 0.08 | +0.0067264 |
| soda | 0.20 | +0.0031003 |
| alpaca | 0.15 | +0.0030053 |
| stories_topic | 0.40 | +0.0028121 |
| tinystories | 0.09 | +0.0015353 |
| **oasst1** | **0.02** | **−0.0073214** |

`dolly` and `oasst1` are the two sources `Trainer.evaluate`'s docstring records as
too small for a full evaluation batch — 7 KB and 4 KB — so their per-source
numbers are the noisiest in the table and carry 0.08 of the weight between them.
**No bar was pre-registered per source and none is invented here.** The direction
is consistent across the five sources that are not tiny.

### 2.1 The effect is the size `EXP_017` measured, with the opposite sign

`PREDICTION_v13.md` §3 quoted `EXP_017`'s marker — the detached arm at 735 K on
**enwik8** came in at **−0.00336**, i.e. *better* — and said in advance that the
number does not transfer. It did not:

```
enwik8, 735 K, 2 layers      detached is 0.00336 BETTER
chat mixture, 4.4 M, 4 layers detached is 0.00358 WORSE
```

Nearly the same magnitude, opposite sign. That is what "σ is not a project
constant" looks like when it is actually checked instead of assumed, and it is
the strongest argument in this round for having re-measured rather than
inherited.

### 2.2 What P1 does not establish

* **n = 1 per arm.** The design resolves ~0.0025 and this is 0.0036. It is over
  the bar, not far over it. A second seed pair would cost 80 minutes and is not
  run here because the verdict already gates what it was gating.
* **Neither arm is converged.** Both eval traces fall monotonically to the last
  measurement — the control goes 1.1790 at step 12,000 → 1.1746 at 14,000, still
  dropping 0.0044 per 2,000 steps. This is the same signature `BUILD_NOTES.md`
  line 449 records for `chat-v6-scratch`. **The estimator's cost is measured at
  14,000 fine-tune steps and nowhere else**, and a bias that costs bits early can
  behave differently at convergence.
* **It says nothing about the regime the arm was for.** The whole argument in
  `GRADIENT_BOUND_NOTE.md` §5 is that the exposure is a *tail* risk under a
  higher learning rate, a longer window or a larger model. This round tests none
  of them — by design, since P1 gates them — so P1 refutes the *cheap* case for
  the arm, not the tail-risk case. What it removes is the claim that the
  insurance is free.

## 3. P2 — the bound shapes the gradient, not where the optimiser lands

| | L0 max\|β·dv\| | × LIF ceiling | max\|w·vs\| | layers over 1.0 | T2 |
| --- | ---: | ---: | ---: | ---: | :---: |
| `chat-v13-ctl` | 10.220 | 19.95× | 21,340 | 4/4 | ✓ |
| `chat-v13-det` | 10.380 | **20.26×** | 14,637 | **4/4** | ✓ |
| shipped, for reference | 9.776 | 19.08× | 14,454 | 4/4 | ✓ |

**Training through the bounded estimator left the model exactly as far outside
`EXP_016`'s region as training without it.** That is not a contradiction — the
detached rule removes `vf` from the *backward's* Jacobian, which is a statement
about the gradient and says nothing about where the optimiser ends up.
`PREDICTION_v13.md` §4 listed this outcome in advance as one of three, all
interesting, none a failure.

**Read the ctl and det columns with the same yardstick in mind.** Both report the
**undetached** multiplier those weights would carry, because the detached rule's
own multiplier is `β_f·(1 − s)`, bounded by `β_f = 0.5` **by construction** — it
would read 0.5 on every checkpoint ever trained and could not distinguish two of
them. The artifact carries it as `max_abs_g_detached_rule` so the contrast is
visible rather than argued.

## 4. P3 — layer 0 does the same thing either way

| | L0 tau q50 | frac below tau 8 | L0 input gain q50 | pinned, all layers |
| --- | ---: | ---: | ---: | ---: |
| `chat-v13-ctl` | 2.682 | 88.9 % | 20.89 | 111 / 4096 |
| `chat-v13-det` | 2.676 | 88.9 % | 20.99 | 107 / 4096 |
| shipped | 2.673 | 88.9 % | 20.93 | 112 / 4096 |

`GRADIENT_BOUND_NOTE.md` §4 found the shipped lineage trading its layer-0
timescales for a ~23× input gain, and §7 item 3 asked whether anything changes
that. **The estimator does not.** All three columns agree to the third
significant figure, and the fraction below tau = 8 is identical to the channel.

## 5. A defect in this round's own instrument, caught before it was reported

**The first P2/P3 run said the opposite of §3 and §4, and it was wrong.** It
reported `chat-v13-det` at `max|w·vs| = 578` against the control's 21,340, and
layer-0 input gain **exactly 1.00** with **2,628 pinned channels** — which reads
as *"the bounded estimator moves the model out of the unbounded region"*, a clean
and completely false headline.

The cause: `reset_jacobian_probe.py` and `slow_channel_census.py` both selected
the threshold-gain path with `cfg.arch == "twocomp_threshold"`, an **exact string
match**. `twocomp_threshold_detach` fell through, so both scanned a current
roughly 20× too small — which is why almost nothing fired and why the drive
collapsed. `1.00` is `exp(0)`, the hard-coded fallback, and that is what gave it
away.

**It is the same defect class as the mutant that escaped this session's first
test pass**: a path where the learned gain is absent is invisible whenever the
gain happens to be 1. Both scripts now key on `hasattr(model, "thr_log")`, the
property that is actually being asked about, and
`tests/test_snnchat.py::test_every_arch_that_has_a_learned_threshold_exposes_it_the_same_way`
holds it up — it asserts that every arch carrying a threshold exposes it under
the same names **and** that perturbing `thr_log` moves the forward, which is the
half the escaped mutant violated. Verified to fail against that mutant.

**`GRADIENT_BOUND_NOTE.md`'s numbers are unaffected and this was checked, not
assumed.** Every checkpoint in that file is `arch = "twocomp_threshold"`, so the
exact match held and the gain was applied. `chat-v3d-aligned` was re-probed and
re-censused with the fixed scripts and reproduces **bit-identically** on all four
layers and on every layer-0 statistic.

## 6. What ships, and what happens next

**Nothing ships.** `experiments/chat/SHIPPED` still names
`chat-v3d-aligned/ckpt_best.pt`. `chat-v13-det` lost its pre-registered bar, and
`chat-v13-ctl` is a reproduction of the incumbent's recipe, not a challenger —
the standing rule that an unresolved margin does not displace an incumbent is not
even reached here, because this margin is not unresolved.

**The arch stays in the tree, opt-in and off by default.** It costs nothing to
keep: no parameter, no kernel, no inference cost, and a checkpoint trained with
it still loads into `snn.model` bit-identically. What changes is the honest label
on it — from "cheap insurance worth taking" to **"insurance that costs 0.0036 bpc
at this operating point, whose premium has now been measured and whose payout has
not."**

**Referred, and not decided here:**

1. **Whether the tail-risk case is worth testing at all.** §2.2 item 3 is the
   argument that P1 does not touch it. Testing it means the 2e-3 runs
   `GRADIENT_BOUND_NOTE.md` §7 item 2 describes, now knowing they start 0.0036
   bpc behind — and the pre-registered outcome there was never bpc, it was
   *whether the undetached arm survives*. That is still unmeasured and it is
   still the only thing that would justify paying the premium.
2. **Whether 0.0036 at 14,000 steps is the converged cost.** §2.2 item 2. Both
   arms were still improving.
3. **A second seed pair**, if anyone wants the verdict at n = 2 rather than
   n = 1. 80 minutes.

None of these is recommended here. **The cheapest true statement this round
produced is that the bounded arm is not free, and that was worth 80 minutes to
learn before spending more.**
