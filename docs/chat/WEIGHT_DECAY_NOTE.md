# A note on weight decay and the slow pole

**Status: an observation, not a result.** It was found while debugging a chat
training run, it has no control arm, and nothing in the research tree has been
changed on the strength of it. It is written down here, outside
`docs/reports/`, because acting on it is Elliot's call and would need a
pre-registered candidate. Everything in §1 and §2 is a measurement that can be
re-run in seconds; §3 onward is inference and is labelled as such.

---

## 1. What was measured

`beta_s_raw` is the unconstrained parameter behind `beta_s = sigmoid(raw)`, the
two-compartment neuron's slow-compartment decay. `snn.train.Trainer` puts every
parameter in **one** AdamW group at `weight_decay = 0.1` — deliberately, and the
reason is recorded in its own docstring: *"the spec exposes exactly one
`weight_decay`, so splitting biases/embeddings out would be an unlogged
hyperparameter choice."*

AdamW's decay is decoupled: `p -= lr * wd * p`. It pulls `beta_s_raw` toward
**zero**, i.e. `beta_s` toward 0.5, i.e. a slow-pole time constant of **2
characters** — from an initialisation of `beta_slow = 0.95`, i.e. tau = 20.

Measured on the committed checkpoints (`ckpt_best.pt`, all at step 20,000, all
`beta_slow = 0.95`, `lr = 3e-3` cosine, `weight_decay = 0.1`):

| run | tau min | tau median | tau max | fraction tau < 5 |
| --- | --- | --- | --- | --- |
| `twocomp_s0` | 1.35 | 2.25 | 78.04 | 92.7 % |
| `twocomp_s1` | 1.46 | 2.27 | 68.88 | 92.8 % |
| `twocomp_s2` | 1.47 | 2.31 | 74.95 | 93.0 % |
| `compose_s1` | 1.54 | 2.47 | 68.15 | 93.1 % |
| `compose_s2` | 1.52 | 2.47 | 66.49 | 93.6 % |

`twocomp_s0` quantiles over all 1,024 slow poles:

| 1 % | 5 % | 25 % | 50 % | 75 % | 90 % | 95 % | 99 % | max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1.57 | 1.75 | 1.98 | **2.25** | 2.83 | 3.94 | 6.64 | 19.86 | 78.04 |

**0.98 % of channels end above their own initialisation of tau = 20. 2.73 % end
above tau = 10.**

None of this is new to the project: `EXP_004` §10.5 recorded that `beta_s` "went
bimodal", and `EXP_006` §2 records the median as "~0.548 in layer 0", which is
tau = 2.21. The distribution is known.

## 2. The arithmetic control

What is new is the comparison to a trajectory with **no gradient at all**.
Propagating `beta_s_raw` from `logit(0.95) = 2.9444` through 20,000 steps of the
committed schedule under decay only — `raw *= (1 - lr_t * 0.1)` — gives:

```
cumulative decay factor   0.04976
beta_s_raw               2.9444  ->  0.1465
beta_s                     0.95  ->  0.5366
tau                       20.00  ->   2.158
```

**Pure decay predicts tau = 2.158. The observed medians are 2.25, 2.27, 2.31,
2.47, 2.47** — 4 % to 14 % above it. `tests/test_snnchat.py::test_weight_decay_would_collapse_an_unprotected_slow_pole`
pins the calculation so it cannot rot.

For the median channel, the gradient's net lifetime contribution to its own time
constant is small compared with the regulariser's.

## 3. What this might mean (inference, not measurement)

* **The "bimodality" may be largely a regulariser artifact rather than a learned
  allocation.** `EXP_004` §10.5 and `EXP_006` describe the model as having
  "chosen to have ~15" long-timescale channels per layer. An alternative reading
  is that decay pulls *every* channel toward tau ≈ 2.16 and the tail is the small
  set whose gradient signal was strong enough to resist. The two readings predict
  the same distribution and are not distinguished by any measurement taken so far.

* **`EXP_006`'s ~3 % may be a property of the optimiser setup, not of the task.**
  `EXP_006` established *causally* that the tail carries the horizon — that result
  stands whatever produced the tail. But its framing question, whether "capacity
  is 3 % of width and nowhere near saturated", reads differently if the 3 % is
  where the decay/gradient balance happens to land. Measured: 2.73 % of channels
  end above tau = 10, against `EXP_006`'s ~3.1 % (top 16 of 512) tail definition.
  **This correspondence is suggestive and nothing more** — two numbers near 3 %
  is not a mechanism, and `EXP_006`'s tail is defined by rank, not by a threshold.

* **Decay pulls two of the arm's three per-channel parameters toward the
  Phase-2 baseline.** `snn.config` says it plainly: `w_init: 0.0 nests the
  baseline`. Decay pulls `w` toward 0, which *removes the slow compartment
  entirely*, and pulls `beta_s_raw` toward tau = 2. Only `thr_log` is benign,
  because its initialisation is already 0 — which is exactly the case `EXP_008`
  analysed when it noted that "decoupled weight decay pulls `θ` toward 0 (gain
  toward 1)" and treated it as a mechanism by which a redundant parameter helps.
  The same mechanism applied to `beta_s_raw` is not a pull toward identity; it is
  a pull toward a different neuron. **In this run the gradient clearly won for
  `w`** (mean `|w|` grew 0.1 → 0.47) **and clearly lost for `beta_s_raw`.**

## 4. What it would take to establish any of it

One run of the adopted arm with `beta_s_raw` (and arguably `w`) held out of the
weight-decay group, everything else identical, at matched seeds, against the
committed arm. Pre-register the direction *and* the possibility that it is worse:
a slow pole that stays at tau = 20 in every channel is `EXP_006`'s clamp-to-a-
constant condition from the other side, and `EXP_006` measured that clamping every
channel to the median is **0.179 bpc worse than the Phase-2 baseline**. There is a
real chance the decay is doing useful work by *creating* the spread, and that
removing it produces a uniformly-slow network that is worse than either.

That is the experiment. It has not been run, and this note is not evidence about
its outcome.

## 5. Corroboration from the chat run, such as it is

`chat-v1` (4.42 M parameters, d = 1024, K = 4, this corpus — comparable to
nothing in `docs/reports/`) was initialised with time constants spread
log-uniformly from **3 to 600** characters. At step 20,000, with the same single
parameter group:

* tau median **2.25 – 3.09** per layer, max fallen from 600 to **34 – 63**
* layer firing rates falling monotonically all run — layer 1 from 0.255 to 0.072
* validation bpc flat from step 5,000 to 10,000, then **regressing** at 20,000
  (1.2821 → 1.4007) across every source at once

Re-run as `chat-v2` with `beta_s_raw`, `w`, `thr_log` and the biases exempt from
decay. **This is n=1 against n=1 on a different corpus at a different width, with
one variable nominally changed and the whole run restarted** — it is a debugging
observation, not a controlled comparison, and it is reported here as one.

### 5.1 The directional part, which is harder to explain away

Both runs start from the identical spread (tau 3–600 log-uniform, median 42.3,
62.5 % of channels above tau = 10).

| | tau median | tau max | % > 10 | % > 100 | mean \|w\| |
| --- | --- | --- | --- | --- | --- |
| `chat-v1` @ 22,000 (decayed) | 2.46 | 63.0 | 1.9 % | 0.0 % | 0.356 |
| `chat-v2` @ 5,000 (exempt) | 23.69 | **778.6** | 68.9 % | 23.0 % | 0.102 |

**With the decay removed, the maximum time constant rises above its own
initialisation** — 600 to 778 — and 23 % of channels sit beyond tau = 100. The
gradient, when it is not being opposed, *lengthens* time constants. In the
decayed run the same quantity fell from 600 to 63.

That is a statement about direction rather than about final loss, and it is the
part least likely to be an artifact of the restart: nothing about re-initialising
a run makes a parameter drift upward past where it started unless something is
pushing it there.

The `|w|` column is a secondary observation with an obvious reading and no
evidence behind it: in the decayed run the compartment mix grew 3.5× while the
slow pole collapsed, and in the exempt run it did not move. A larger mix
extracting signal from a fast-decaying compartment is a plausible compensation,
and it is only a story.

## 6. What was not done

Nothing under `src/snn/`, `experiments/runs/`, `experiments/logs/` or
`docs/reports/` was read for anything but measurement, and none of it was
modified. The exemption lives in `snnchat.train.ChatTrainer._param_groups` and
applies to the chat model only.
