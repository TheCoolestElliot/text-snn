# PREDICTION v13 — the bounded-estimator round

**Written 2026-08-22, before either arm of this round has been trained.** No
checkpoint named in §2 exists at the time of writing. The two diagnostics this
round scores with (`reset_jacobian_probe.py`, `slow_channel_census.py`) were
written, run on the *existing* checkpoints, and reported in
[`GRADIENT_BOUND_NOTE.md`](GRADIENT_BOUND_NOTE.md) before this file — so their
behaviour is fixed and this round cannot tune the instrument to the result.

Not part of the Phase-1..5 research protocol. No number here is a reported
figure. Rates follow [`CONVENTIONS.md`](CONVENTIONS.md).

**This round rules no research decision.** Phase 4's decision **#4** (Phase-5
authorisation) is open and is Elliot's; `snnchat` is outside the protocol and
running here does not touch it. Nothing under `src/snn/` is modified by this
round.

---

## 0. What motivates it, in one paragraph

`GRADIENT_BOUND_NOTE.md` §2 measured every chat checkpoint on disk as sitting
outside the bound `EXP_016` proved for the plain LIF: the shipped model's layer 0
runs at **19.1×** a ceiling the plain LIF provably cannot exceed at any width,
with `max|w·vs| = 14,454` against a crossover at ~1, stable across four batches.
§2.2 then measured the thing that actually matters and found **no pathology**:
loading one set of weights into both estimators, the gradients differ by **under
10 % in norm** and nothing is non-finite. So the case for the bounded arm is not
"the gradient is exploding" — it is that the *guarantee* is absent, it has bitten
once (`chat-v6-inst-a-s1`, NaN at step 4,859), and the exposure is a tail risk
under exactly the shifts someone would want next: a higher learning rate, a
longer window, a bigger model.

**This round does not test any of those shifts.** It tests the one thing that
gates them: **does training through the bounded estimator cost bits at the
current operating point?** If it does, nothing downstream is worth running. That
is the whole of P1.

## 1. Entry conditions, and they are met

1. **The tree reproduces the control's committed evaluation exactly.**
   `scripts/chat/reproduce_eval.py` re-runs `Trainer.evaluate`'s protocol on
   `chat-v3d-aligned/ckpt_best.pt` and returns **residual `0.00e+00` on all seven
   sources and on the weighted mean** — 1.174290300 against a committed
   1.174290300. Artifact:
   `experiments/chat/_probe/gradient_bound/reproduce_eval_chat-v3d-aligned.json`.
   This matters because the working tree carries **uncommitted changes that
   predate this round** (a `vocab_version`-2 UTF-8 tokenizer with `<|context|>` /
   `<|source|>` markers, and dtype generalisations in `snnchat.data`). They are
   inert on the v1 path and the residual is the evidence, not the argument.
2. **The control arm's code path is unchanged.** The only edits this session made
   to `src/snnchat/model.py` are additive: a new class, a new `elif` branch, one
   `param_count` row, one error message. `git diff` shows no removal inside the
   `twocomp_threshold` branch.
3. **`src/snn/` is untouched.** No K1 exposure, no new kernel, no R10 gate.
4. **The GPU is idle** and no timed run is in flight.

## 2. The two arms — one variable

Both from `chat-v3d-aligned`'s exact session-7 command line, recovered verbatim
from `experiments/chat/session7-driver/stdout.log`, with `--arch` and
`--run-name` the only changes:

| | `chat-v13-ctl` | `chat-v13-det` |
| --- | --- | --- |
| `arch` | `twocomp_threshold` | **`twocomp_threshold_detach`** |
| init from | `chat-v2-anneal/ckpt_best.pt` | same |
| seed | 0 | 0 |
| everything else | 14,000 steps, B=160, L=256, lr 5e-4 cosine to 5 %, warmup 200, `bot_loss_weight` 3.0, `align_frac` 0.75, `align_lookahead` 1024, wd 0.1, clip 1.0, fp32 | identical |

**Same seed on purpose.** `MixtureSampler.batch(s)` is a pure function of
`(seed, step)` and both arms start from the same weights, so the two trajectories
see the **identical data in the identical order from the identical point** and
diverge only through the gradient. That is a paired design and it is tighter than
two independent seeds. It does not make the comparison noise-free — the
trajectories still separate chaotically, and `deterministic = false` means even
two identical runs would not agree bitwise — which is what P1's bar is sized
against.

## 3. P1 — the primary bar, and its power stated in advance

**Quantity:** held-out weighted bpc from `Trainer.evaluate` — fresh state, per
source, weighted by the training mix. Seed-independent and sampler-free, so it
does not have to fight the **0.099** headline swing that has left every recent
round unresolved (`QUALITY_v6.md` §8, `QUALITY_v7.md`).

**Δ = det − ctl.** Negative is better.

**The bar is measured, not guessed.** The four committed replicates of *this
recipe* — `chat-v3d-aligned` and `-s1/-s2/-s3`, seeds 0–3 — give

```
1.1742903  1.1750911  1.1732633  1.1732495
mean 1.1739735   SD_seed 0.0008903   (n = 4, 3 df)
```

so a difference of two single runs has `SE = 0.0008903 · √2 = 0.0012591` and

> **P1 resolves `worse` if Δ > +0.0025182, `better` if Δ < −0.0025182, and
> `unresolved` otherwise.** Two-sigma, both-sided, fixed here.

**Stated power, per `CONTRIBUTING.md` §2.** `SD_seed` is itself `n = 4, 3 df`, so
it is known to roughly ±40 %. This design can resolve an effect of ~0.0025 bpc
and **cannot** resolve one smaller. The only prior on the effect size is
`EXP_017`'s marker at 735 K on **enwik8**: detached **2.11533 ± 0.00237** (n = 3)
against an undetached committed mean of 2.11869, i.e. **−0.00336, detached
slightly better**, and `EXP_017` itself calls both numbers "inside the design's
own stated 0.005 resolution floor". **That number does not transfer** — different
corpus, different width, different depth, and this project's own standing rule is
that σ is not a project constant. It is quoted to show that the expected effect
is *small*, which is why **`unresolved` is the modal outcome and is not a
disappointment.**

**The guard, which is the thing this project has got wrong three times.**
`EXP_012`'s Y5, `EXP_013`'s N1 and `EXP_022`'s H4 all fired a bar on a difference
without constraining what the other term was allowed to do. So:

> **P1 reports `NOT RUN` unless `chat-v13-ctl`'s weighted bpc lands inside
> `1.1739735 ± 2 · 0.0008903` = [1.1721929, 1.1757542].** A control outside that
> band means the tree, the session or the machine has moved, and Δ would be
> measuring that instead. In that case the round reports the control's position
> and stops.

## 4. P2 — does the bounded estimator move the model out of the region?

**Marker, no bar, and deliberately so.** `reset_jacobian_probe.py` on both
resulting `ckpt_best.pt`, batch 0, reporting `max|β_f·dv|` and `max|w·vs|` per
layer against the shipped model's 9.776 / 14,454.

Nothing predicts a magnitude here, so nothing is pre-registered as one. The
detached estimator does **not** force `|w·vs|` small — it removes `vf` from the
backward's Jacobian, which is a statement about the gradient and not about where
the optimiser ends up. Three outcomes are all interesting and none is a failure:
the region is unchanged (the estimator is pure insurance), the region shrinks
(the bound also shapes the trajectory), or it grows (the bound licenses weight
space the undetached arm could not survive).

**One thing here IS a check rather than a marker.** `t2_holds` must be `True` on
both probes — the local scan reproducing the committed one bitwise. If it is
false, no number in P2 is readable and P2 reports NOT RUN.

## 5. P3 — does it change what layer 0 does with its timescales?

**Marker, no bar.** `slow_channel_census.py` on both, reporting layer 0's tau
median, the fraction below tau = 8, and the input gain median, against the
shipped model's **2.67 / 88.9 % / 20.93**.

`GRADIENT_BOUND_NOTE.md` §4 found that the shipped lineage traded its layer-0
timescales for a 23× input gain while the one other lineage did not, and §7 item
3 refers the question of whether anything changes that. This is the cheapest
possible look at it and it is **n = 1 per arm on a two-lineage contrast**, which
resolves nothing. Reported as an observation.

## 6. What this round explicitly does NOT measure

* **Any sampler-scored quality metric.** No `topic_propensity`, no
  `story_dodge`, no demo battery. Those carry the 0.099 swing and 14,000 steps of
  fine-tune cannot beat it at n = 1. A round that scored them would produce an
  unresolved verdict at four times the cost.
* **Whether the bound buys back the learning rate.** That is
  `GRADIENT_BOUND_NOTE.md` §7 item 2 and it needs its own runs at 2e-3. **It is
  gated on P1** — if the estimator costs bits at 5e-4, running it at 2e-3 answers
  a question about an arm nobody would ship.
* **Anything about a longer training window or a larger model.** Both are the
  shifts §0 says the exposure lives under, and neither is tested here.
* **Whether the shipped model should be replaced.** The standing rule applies
  unchanged: an unresolved margin does not displace an incumbent. **`chat-v13-det`
  does not ship on a P1 of `unresolved`**, and `experiments/chat/SHIPPED` is not
  edited by this round under any outcome.

## 7. Cost

Two runs × 14,000 steps. `chat-v3d-aligned` took **2,350 s** (39.2 min) for the
identical shape, and session 7's driver measured 40.2 min wall including eval. So
**~80 minutes**, plus ~4 minutes of probes and censuses. `--budget-minutes 75`
per run, as session 7 used, so a hung run stops rather than holding the GPU.

Launched detached via `scripts/launch.py`, per `CLAUDE.md`: anything longer than
a few minutes is invisible to a harness waiting on it, so completion is detected
by polling for `summary.json`, not by waiting on exit.

## 8. The results section is empty until the runs finish

Appended to, never rewritten. `QUALITY_v13.md` carries the verdicts.
