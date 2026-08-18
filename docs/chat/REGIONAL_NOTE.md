# A note on regions: where the model puts its slow channels, and whether it uses order

**Status: measurements, not results.** Everything here was read off committed
checkpoints with no training and no control arm. Nothing in the research tree
was changed on the strength of it. Written outside `docs/reports/` for the same
reason [`WEIGHT_DECAY_NOTE.md`](WEIGHT_DECAY_NOTE.md) is: acting on it needs a
pre-registered candidate, and that is `experiments/logs/EXP_019_regional_census.md`.

Rates and ratios here follow [`CONVENTIONS.md`](CONVENTIONS.md). Both scripts
re-run in seconds:

```bash
python scripts/chat/regional_profile.py experiments/chat/*/ckpt_best.pt
python scripts/chat/permutation_probe.py --ckpt experiments/chat/chat-v3d-aligned/ckpt_best.pt
```

---

## 1. The defect that had to be fixed first

`ChatTrainer._refresh_description()` was called from `resume` and `init_from` —
that is, whenever weights were **loaded** — and never after they were **trained**.
`summary.json` and every checkpoint therefore reported the slow-pole profile at
step 0.

The tell is that it was bit-identical across four training seeds and three
corpora:

| run | L0 | L1 | L2 | L3 |
| --- | ---: | ---: | ---: | ---: |
| `chat-v3d-aligned` and its s1/s2/s3 | 2.70 | 13.21 | 10.05 | 7.30 |
| `chat-v9-narrative`, `chat-v12-rare` | 2.70 | 13.21 | 10.05 | 7.30 |
| `chat-v6-scratch` (fresh init, **84,000 steps**) | 42.32 | 42.32 | 42.32 | 42.32 |

`chat-v6-scratch` is reporting `spread_slow_poles`' pristine output. Everything
else is reporting `chat-v2-anneal/ckpt_best.pt` at step 7,874, confirmed by
`log.jsonl`'s `init_from` record.

**Fixed 2026-08-17.** `save_checkpoint` and the end-of-run summary now refresh
first, and `slow_pole_tau_init` is carried alongside so a run can still say what
it started from — an inherited spread and a grown one are indistinguishable from
the final profile alone. Pinned by
`tests/test_snnchat.py::test_saved_description_reports_the_weights_it_ships`,
which was checked to **fail** against the old code before being committed.

**Every number below was read from the `state_dict` directly**, not from a
summary, and so predates and is unaffected by the fix.

## 2. Where the slow channels are, and whether anything reads them

`spread_slow_poles` initialises **every layer identically** (tau log-uniform over
3–600), so any per-layer difference after training is learned, with no
initialisation confound.

`|w|` matters as much as `tau`. The neuron is `v_t = vf_t + w_c·vs_t`, so a
channel with a 600-character time constant and `w_c ≈ 0` is inert — `vs` never
reaches the threshold comparison, and `w_c = 0` nests the plain Phase-2 LIF
exactly.

Median `|w|` on channels with tau > 100, as a ratio to the fast channels, over
**26 checkpoints** with a slow population:

| | layer 0 | layer 1 | layer 2 | layer 3 |
| --- | ---: | ---: | ---: | ---: |
| ratio, min–max over 26 | **4.99 – 10.11** | 0.09 – 5.58 | **0.07 – 0.17** | **0.07 – 0.21** |

**Layer 0's slow channels are load-bearing in every checkpoint. The deepest two
layers' are switched off in every checkpoint**, at `|w| ≈ 0.01`.

Read the direction before the magnitude: the model's persistent store sits
**nearest the input**, which is the opposite of the cortical arrangement the
comparison usually invokes.

### 2.1 What did *not* replicate, and it was nearly reported as if it had

The shipped model's median-tau profile is 2.67 / 13.09 / 9.74 / 7.18 — a rise to
layer 1 and then a fall — and it reproduces across `chat-v3d-aligned` s0–s3 with
sd ≤ 0.047. **That reproducibility is worth almost nothing**: all four fine-tune
from the same parent at step 7,874, so they are one draw of the phenomenon and
four draws of a 14,000-step fine-tune.

The two genuinely independent lineages disagree:

| | L0 | L1 | L2 | L3 | peak |
| --- | ---: | ---: | ---: | ---: | --- |
| `chat-v2` (fresh, 76,000 steps) | 2.71 | **13.23** | 10.10 | 7.34 | layer 1 |
| `chat-v6-scratch` (fresh, 84,000 steps) | 12.87 | 23.10 | 21.72 | **24.94** | layer 3 |

So the shape is **not** a universal outcome and must not be quoted as one. The
load-bearing dissociation in §2 is what replicated across both.

### 2.2 The control that is already on disk

`chat-v1` is the one run trained **without** the decay exemption. It has
essentially no channels above tau = 100 at all — the ratio is undefined because
the numerator set is empty. That is
[`WEIGHT_DECAY_NOTE.md`](WEIGHT_DECAY_NOTE.md)'s collapse, in one row.

It also bears on that note's §4 caution, which worried that exempting the decay
might produce a uniformly *slow* network, worse than either arm. The exempt runs
did not do that: the shipped model keeps 2.3–8.4 % of channels above tau = 100
and the rest fast. The gradient spends the freedom on structure rather than on
slowness. One observation, no control arm, and it does not settle the question.

## 3. Does the model use order, or only recency?

The horizon probe measures how far back conditioning information helps. It cannot
distinguish *"the model remembers what was there"* from *"the model remembers
roughly how much was there"* — and a leaky integrator gives nearly the same state
for the same characters in a different order. If that were all the model had, the
horizon would be a recency statistic and no timescale work could improve it.

For a band of the window at lag ~c/2..c, scoring only positions strictly after
it: **PERMUTE** shuffles the band's order and keeps its multiset; **ROLL**
replaces the band with the same band from another window in the batch. PERMUTE
alone is ambiguous — it falls to zero both when order stops mattering and when
the band stops mattering — so the ratio is the measurement.

3,072 windows, soda val. Both self-checks (identity permutation; perturbing only
the future) returned **exactly 0.0**.

| band lag | 4 | 8 | 16 | 32 | 64 | 128 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ΔPERMUTE (bpc) | 1.454 | 1.888 | 1.350 | 0.834 | 0.468 | 0.269 |
| ΔROLL (bpc) | 4.240 | 2.957 | 1.816 | 1.079 | 0.639 | 0.371 |
| **order share**, shipped | 0.343 | 0.638 | **0.744** | **0.773** | **0.732** | **0.724** |
| **order share**, `chat-v6-scratch` | 0.335 | 0.637 | **0.743** | **0.766** | **0.733** | **0.750** |

**The order share does not decay.** It sits at 0.72–0.77 from lag 16 out to lag
128, and the two independently trained lineages agree to within 0.02 at every
point. Roughly three quarters of what context is worth to this model is its
*order*.

The lag-4 point is a **width artifact, not a finding**: with a two-element band,
half of all random permutations are the identity, so ΔPERMUTE is mechanically
halved.

**What this closes:** the model is not a bag of characters, and the horizon is a
memory. **What it does not:** it says nothing about *how far* order is retained
beyond 128, nothing about binding (cross-turn recall remains 0 of 240 with a
matched 0-of-240 control), and it is a property of these two checkpoints rather
than of the architecture.

## 4. What would establish any of it

`EXP_019` (`experiments/logs/EXP_019_regional_census.md`), pre-registered
2026-08-17 and **explicitly downstream of §2 and §3, which it labels post-hoc**.
Its L2 bar is placed where it can fire against the regional programme's own
interest: if lesioning the slow half of the most-costly layer costs no more than
a matched random control, the slow compartment is not doing load-bearing work in
this model and every timescale-based proposal is refuted before it is built.
