# Phase 2 — Baseline Reproduction

**Project:** Character-level spiking neural-network language model
**Author:** Elliot Asher Caudill
**Date:** 2026-08-01
**Status:** Phase-2 deliverable. Awaiting review before Phase 3 begins.
**Branch:** `phase-2-baseline`

---

## 0. Executive summary

Phase 1 measured the machine. Phase 2 builds the thing the measurements implied,
proves it does what it claims, and establishes the noise floor that every later
comparison will be judged against.

The headline is not the bits-per-character number — it is that **the number is now
trustworthy**, because the three ways this project could have produced a
plausible-but-wrong result have each been closed by an executed test rather than
by an argument:

1. **A silently wrong hand-written gradient (risk R10).** The fused CUDA backward
   agrees with the eager reference to 1.2e-07–9.5e-07, spikes are bit-identical,
   and the membrane matches to *exactly* 0.0. That is necessary but not
   sufficient, so the gate itself was **mutation-tested**: 21 plausible
   mis-derivations were patched into the kernel one at a time, and all 21 were
   caught. A gate that has never been shown to fail is not evidence.
2. **Both implementations sharing the same wrong idea of a LIF neuron.** Checked
   against snnTorch 1.0.0, an implementation we did not write (§4).
3. **A configuration that never actually ran.** Graphed and eager training agree,
   and the determinism setting turned out to be silently changing the workload
   (§3.1).

Two findings came out of Phase 2 that were not visible in Phase 1, and both change
what the project should do:

* **Determinism was costing 2× on the exact path the architecture exists to
  protect** (§3.1). `use_deterministic_algorithms(True)` enables
  `fill_uninitialized_memory`, which adds one fill kernel per allocation — and our
  fused LIF is a three-output kernel called once per timestep. Every run we report
  is deterministic, so half the fusion win was being handed back invisibly.
* **The default parameter initialisation makes every layer past the first
  completely silent** (§5). This is arithmetic, not luck: a binary spike vector at
  rate *p* carries variance *p*, not 1, so the second layer's pre-activation sits
  **7.8σ** below threshold. It self-corrects within ~50 optimiser steps, which is
  why it is documented rather than patched into the baseline.

The numbers, from 17 runs and 7.8 GPU-hours, scored on the full enwik8 test split
under both evaluation protocols:

| Arm | Params | Test bpc (carried) | σ over seeds |
|---|---:|---:|---:|
| **SNN baseline** | 735 437 | **2.2531** | 0.0046 (n=5) |
| Analogue control (identical params) | 735 437 | 2.2648 | 0.0022 (n=3) |
| GRU anchor *(violates I5)* | 738 019 | **1.7674** | 0.0022 (n=3) |

Three results are worth the reader's attention before the detail:

* **The noise floor is σ = 0.00461 bpc** (§7.1) — the number the literature survey
  found nobody had published for surrogate-gradient SNN training, and the one
  every Phase-3/4 conclusion will be judged against. It resolves effects as small
  as 0.01 bpc at three seeds.
* **Binarity is nearly free; invariant I5 is what costs.** Against its own
  identical-parameter analogue control the spiking arm is not worse — it is
  0.0117 bpc *better* (2.5σ). Against the GRU anchor it is 0.486 bpc worse
  (105σ). The literature's expected "non-trivial binarity cost" does not reproduce
  here at all (§7.2). Phase 3 should stop trying to recover a penalty that is not
  there and spend the effort on neuron-state memory instead.
* **A pre-registered prediction of ours was wrong** (§7.3). `EXP_000` §6 declared
  in advance that β = 0.5's sub-one-character memory horizon would make the
  baseline weak. β = 0.5 turned out to be the *best* of the three values tested;
  longer membrane memory hurts monotonically, by 11σ and 20σ. Turning the
  *magnitude* of a scalar time constant up does not work, which is what makes the
  §4.6-admitted *structured* dynamics the right Phase-3 target.

And one Phase-1 conclusion did not survive the real model: fusion and CUDA-graph
capture **do not compose multiplicatively** (§6.1). They are the same lever.

---

## 1. What was built

Against the frozen interface in [`02a_phase2_spec.md`](02a_phase2_spec.md).

| Module | Contents |
|---|---|
| `snn/config.py` | `Config` dataclass, argparse surface, seeding and determinism |
| `snn/data.py` | acquisition with SHA-256 verification, memory-mapped `uint8` corpus, samplers, both eval protocols |
| `snn/surrogate.py` | arctangent surrogate value and derivative, plus its CUDA source |
| `snn/kernels.py` | jiterator probe and the compiled forward/backward kernel cache |
| `snn/neuron.py` | eager reference scan and `FusedLIFScan` autograd Function |
| `snn/model.py` | `SpikingCharLM` and the `analogue` / `gru` controls |
| `snn/metrics.py` | bits-per-character, firing rates, CUDA kernel counting |
| `snn/train.py` | trainer with CUDA-graph capture, static buffers, resumable checkpoints |
| `snn/evaluate.py` | both evaluation protocols |

Plus `scripts/` (corpus download, train, evaluate, detached launcher, kernel-count
bench, campaign driver) and 217 tests.

### 1.1 The architecture, and why it is shaped this way

Phase 1's cost model is `time ≈ 14.5 µs × kernels-per-timestep × L`, which makes
kernel **count** the performance metric. The obvious inner loop —

```python
for t in range(L):
    h = emb[:, t]
    for layer in layers:
        h, mem = layer(h, mem)      # a GEMM inside the time loop
```

— issues `K·L` GEMMs and roughly `13·K·L` elementwise kernels. That is what the
Phase-1 audit benchmarked, and it is not what was built.

Because I5 forbids lateral recurrence **and** nothing feeds backwards across
layers, the layers can be evaluated sequentially in *depth*. Each layer computes
its input current for all `L` timesteps in one GEMM before its time loop starts:

```python
h = embed(x)                        # [B, L, d]
for k in range(K):
    cur  = linear_k(h)              # [B, L, d]   ONE GEMM, whole sequence
    h, v = lif_scan(cur, v0)        # [B, L, d]   L fused kernels
logits = head(h)                    # [B, L, V]   ONE GEMM
```

This is exact, not an approximation: information flows strictly forward in depth,
and causality in time is preserved inside each layer's own scan.

**Measured** (`snn.metrics.count_cuda_kernels`, `SpikingCharLM`, B=2, L=8, K=2):
only **3 GEMM launches regardless of L** — two layer projections and the head —
plus one embedding gather. No GEMM appears inside the time loop.

This is the first place invariant I5 *pays* rather than costs, and it is worth
saying plainly: the constraint that makes the science interesting also makes the
implementation tractable on this hardware.

---

## 2. The capability gates

Ordered as the Phase-1 entry checklist required — the gates that everything else
sits on top of were proven first.

### 2.1 R10 — the fused backward is correct

The most dangerous failure mode in the plan, because it produces *plausible*
results: a hand-written surrogate backward that is subtly wrong still trains, and
still yields a number you could publish.

**Forward**, over 4 reset modes × 4 shapes × 5 (β, threshold) regimes:

| Quantity | Result |
|---|---|
| Spikes, fused vs eager | **bit-identical** |
| Membrane, fused vs eager | **max |Δv| = 0.0 exactly** |

Bit-identical membrane is not an accident of tolerance; it required the kernel to
use `__fmul_rn`/`__fadd_rn` so NVRTC cannot contract the update into an FMA.
Reverting that makes the bit-identity test fail while the ordinary forward test
still passes — the failure was verified, not assumed.

**Backward**, same sweep, with `v0.requires_grad=True` and random cotangents on
both outputs: `grad_cur` and `grad_v0` within **1.2e-07 to 9.5e-07** of the eager
path (spec tolerance `rtol=1e-4, atol=1e-6`).

**The gate was mutation-tested.** A passing test proves nothing until it has been
shown capable of failing, so 21 plausible mis-derivations were patched into
`kernels.py` / `surrogate.py` one at a time and the real test files re-run.
**21 / 21 caught**, including:

| Mutation | Caught |
|---|---|
| drop or sign-flip `−v_pre·sg` in hard reset | yes |
| drop `thr` from the soft-reset derivative | yes |
| give `detached` the derivative of `hard` | yes |
| drop `beta` from `grad_v_prev` | yes |
| drop the `grad_spike·sg` term | yes |
| straight-through (`sg = 1`) instead of the surrogate | yes |
| omit the threshold shift in `x` | yes |
| `>` instead of `>=` in the **backward** kernel only | yes |
| let NVRTC contract the FMA | yes |
| `π` instead of `π/2` in the CUDA source | yes |
| 2× surrogate gain in the CUDA source | yes |

The `>` / `>=` mutation deserves a note: it is invisible under random input (the
threshold is a measure-zero set) but moves `grad_cur` by **20 %** at `v_pre == thr`
under `hard` and `detached`, whose derivative contains a bare `(1−s)` factor that
flips wholesale at equality. It survived 66 tests before the gate was tightened.

### 2.2 R5 — graphed training equals eager training

Graphed and eager losses agree over 20 steps. The specific trap — that a
Python-float learning rate is baked in at capture time, so the schedule silently
stops working — has its own test that captures a graph, replays it across a
schedule change, and fails if the effective LR is frozen.

### 2.3 R1 — CUDA-graph capture on the real model

A full `zero_grad → forward → backward → clip → step` captures and replays on the
real `SpikingCharLM`, for all four reset modes, at 0.05 s capture cost.

### 2.4 A1 — the cost model, re-measured

| Path | Kernels / timestep |
|---|---|
| Fused scan | **1.016** (L `lif_forward` + 2 stack kernels) |
| Eager reference | 13.008 |
| Fused, fwd+bwd | 2.055 |
| Eager, fwd+bwd | 27.031 |

The one-kernel floor spec §0 was designed around is achieved and asserted.

**A correction to Phase 1's constant.** Per-kernel launch cost re-measured on the
real scan is **18.2–19.8 µs**, not the 14.5 µs of `01_reconnaissance.md` §3.4 —
25–37 % higher. The *shape* of the cost model reproduces exactly (16× more
arithmetic per kernel costs 0.92–0.97× the time, so the workload is still
launch-bound), but Phase 1's figure came from a bare `torch.add`, and a real
multi-output jiterator kernel costs more to launch. **The Phase-2 report does not
reuse 14.5 µs**, and neither should Phase 3's ROI estimates.

---

## 3. Systems findings

### 3.1 Determinism was silently costing 2× on the critical path

`torch.use_deterministic_algorithms(True)` also enables
`torch.utils.deterministic.fill_uninitialized_memory`, which emits one extra fill
kernel for every uninitialised allocation. Our fused LIF is a **three-output**
jiterator kernel called once per timestep, so the flag adds three launches per
timestep — on the exact path the entire architecture exists to keep at one.

Measured at the baseline config (B=128, L=256, d=512):

| Setting | Kernels / timestep | Fused fwd+bwd |
|---|---:|---:|
| `deterministic=False` | 1.03 | 14.76 ms |
| `deterministic=True` (fill on — the default) | **4.05** | **29.99 ms** |
| `deterministic=True` (fill off) | 1.03 | 15.18 ms |

**Every run this project reports is deterministic**, so this was not a corner
case: half the fusion win was being handed back in every measured run, and the
standalone kernel-count test did not see it because that test happened not to have
determinism enabled. It only appeared when the whole suite ran in one process.

`seed_everything` now disables the fill. That is sound rather than merely
convenient: the fill exists to make *reads of uninitialised memory* reproducible,
and every kernel here is elementwise over its full output shape, so no element is
ever read before it is written. Determinism itself is unaffected and is still
asserted bit-exactly by `tests/test_determinism.py`.

The finding is pinned by a test that asserts **both** halves — that the flag is off
after seeding, *and* that the flag is genuinely what moves the count — so a future
torch release changing the default fails loudly here instead of appearing as an
unexplained 2× slowdown.

### 3.2 Our reset modes do not map one-to-one onto snnTorch's

`tests/test_neuron_equivalence.py` compares our fused kernel against our own eager
reference. That catches a wrong kernel; it cannot catch both implementations
sharing the same wrong idea of what a LIF neuron is. So the semantics were checked
against **snnTorch 1.0.0**, which this project imports for nothing else
(`scripts/audit/06_neuron_cross_check.py`).

| Ours | snnTorch | Agreement |
|---|---|---|
| `hard` | `zero`, `reset_delay=False` | exact — spikes bit-identical, grads 4.8e-07 |
| `detached` | `zero`, `reset_delay=True` | exact — grads 4.8e-07 ← **snnTorch's default** |
| `none` | `none` | exact — grads 4.8e-07 |
| `soft` | `subtract`, `reset_delay=False` | exact **only below 2 × threshold** |

Two consequences worth stating plainly.

**snnTorch's default is our `detached`, not our `hard`.** `reset_delay` defaults to
`True`, which applies the reset at the start of the *next* step using a detached
signal. Forward trajectories are identical, so the difference is invisible unless
you look at gradients — and it changes them (grad norm 57.3 vs 68.0 on the same
input). Any comparison against a default-configured snnTorch model is a comparison
against **detached** reset. This is the kind of detail that makes cross-paper
spiking results quietly incomparable.

**The `soft` divergence is snnTorch's, and it was isolated by prediction rather
than absorbed by tolerance.** snnTorch's `_base_sub` decides the spike on
`v_pre − reset_prev·thr`, carrying a residual subtraction from the previous step
whenever the post-reset membrane was itself still above threshold. The textbook
definition — and ours — decides on `v_pre`. The prediction that follows is sharp:
hold the membrane below `2·thr` and the two must agree exactly.

| Input σ | Post-reset membrane ≥ thr | Spike mismatches | max abs grad diff |
|---:|---:|---:|---:|
| 0.50 | 6 | 10 | 2.06 |
| 0.15 | **0** | **0** | **2.4e-07** |

### 3.3 A documentation defect in snnTorch, recorded rather than resolved

snnTorch's `ATan` docstring gives `∂S/∂U = (1/π)/(1 + (πUα/2)²)`; its code computes
`(α/2)/(1 + (π/2·α·U)²)`. These differ by exactly `α·π/2` — measured ratio
**3.14159…** at α=2, against the predicted `α·π/2 = π`. The code is what every
published snnTorch model was trained with, so the code is the reference. Our
surrogate matches the code to **0.0**.

---

## 4. Determinism, reproducibility, and honest accounting

### 4.1 What is verified

| Property | Evidence |
|---|---|
| Corpus integrity | SHA-256 of both zips and both extracts pinned in `snn/data.py`; enwik8 vocab **205**, text8 **27**, both asserted, neither adjusted |
| Split integrity | Per-split SHA-256 recorded and independently recomputed |
| Corruption detection | Exercised for real: one flipped bit at offset 1234 of `val.bin` raised the expected mismatch; restoring it restored the hash |
| Same seed ⇒ same run | Identical loss trajectories; different seeds differ |
| Resume is exact | `train(20)` ≡ `train(10) + checkpoint + resume + train(10)`, bit-identical |
| Sampler reproducibility | `batch(k)` is a pure function of `(seed, step)`, so resuming at step *k* draws the same batch as an uninterrupted run |

### 4.2 What is *not* 100 %, stated because it would be easy to omit

Both evaluation protocols drop the trailing partial batch to keep the static
shapes CUDA-graph capture requires. At B=128, L=256 on the 5 MB val split that is
**19 264 of 4 999 999 characters (0.385 %)**.

**Every bits-per-character figure in this report is therefore computed over
99.615 % of its split, not 100 %.** The dropped characters are the tail of the
split, they are the same characters for every arm, and the exact count is returned
by `windowed_eval_char_count` / `contiguous_eval_char_count` rather than being
estimated. This does not affect any *comparison* in this report, all of which are
between arms scored identically — but it is not a rounding error we are entitled
to leave unsaid.

Relatedly: `bpc` for the `fresh` protocol is invariant to batch size only when
`max_windows` is a multiple of the batch sizes compared. The spec claimed
unconditional invariance; that was wrong and is corrected here rather than in a
footnote.

---

## 5. Discrepancy: the default initialisation silences every layer but the first

Found while running the gates, and handled under the protocol's discrepancy rule —
quantify, isolate the root cause, document, resolve — rather than patched away.

### 5.1 The observation

At the frozen baseline (d=512, K=2, threshold 1.0, β=0.5, default `nn.Linear`
init), measured per-layer firing rates at initialisation:

| Seed | Firing rates |
|---|---|
| 0 | `[0.0489, 0.0]` |
| 1 | `[0.0484, 0.0]` |
| 2 | `[0.0482, 0.0]` |

Layer 1 never fires. Consequences, all verified: the head's input is identically
zero, the logits are the head bias broadcast, step-0 loss is exactly
**ln(205) = 5.3230** (measured 5.3219), and `head.weight.grad` is exactly zero.
The same holds at K=3 and K=4 — only the first layer is ever alive — and at
d=256, 512 and 1024.

### 5.2 The root cause is arithmetic, not luck

`nn.Linear`'s default init draws `w ~ U(−1/√d, 1/√d)`, so `Var(w) = 1/(3d)` and

```
Var(Wx) = d · Var(w) · E[x²] = E[x²] / 3
```

Layer 0's input is the embedding: dense, `N(0,1)`, so `E[x²] = 1` and `σ = 0.577`.
Against a threshold of 1.0 that is 1.73σ, giving a firing rate near 4.2 % — which
is what is measured.

Layer 1's input is a **binary spike vector** at rate *p*. For `s ∈ {0,1}`, `s² = s`,
so `E[s²] = p`, **not 1**. With `p = 0.049`:

```
σ = √(0.049/3) = 0.128     →   the threshold is 7.8σ away
```

The default initialisation is calibrated for dense unit-variance activations. A
spike train at 5 % density carries twenty times less energy, and the mismatch
compounds with depth. **This is a property of the initialisation, not of the
architecture** — and it is invisible to anyone who does not log firing rates from
step 0, which is exactly why risk R4 required it.

Confirming the mechanism: lowering the threshold rescues it, exactly as the
arithmetic predicts.

| Threshold | Firing rates at init |
|---:|---|
| 1.00 | `[0.049, 0.000]` |
| 0.50 | `[0.156, 0.036]` |
| 0.25 | `[0.255, 0.193]` |

### 5.3 It self-corrects — measured, not assumed

The surrogate's tails are heavy, so a silent layer is not necessarily a dead one.
Whether that is enough to revive it is an empirical question, and it was answered
by running it rather than by reasoning about it.

On the real corpus, layer 1 revives within 25 steps:

| Step | Firing rates | Train bpc |
|---:|---|---:|
| 0 | `[0.052, 0.000]` | 7.678 |
| 25 | `[0.152, 0.066]` | 7.315 |
| 50 | `[0.334, 0.452]` | 4.573 |
| 300 | `[0.349, 0.415]` | 2.877 |

### 5.4 Resolution, and why the baseline was *not* changed

A variance-scaled initialisation was derived and implemented
(`scripts/audit/07_init_pathology.py::variance_scaled_init`): rescale each spiking
layer so the membrane's steady-state σ matches the threshold, measuring the scale
layer-by-layer on a real batch rather than assuming a firing rate and solving the
resulting fixed point. It removes the pathology at step 0 (rates `[0.109, 0.126]`,
scale factors 1.50 and **4.51**).

**It is not in the Phase-2 baseline, deliberately.** The pathology self-corrects
within ~50 of 20 000 steps, so it does not materially change the baseline result,
and the protocol's Phase-2 rule is the reference architecture with no unproven
modifications. Changing the initialisation because a diagnostic looked bad — when
the diagnostic demonstrably resolves itself — would be exactly the kind of
unforced modification that makes a baseline non-comparable.

It is instead **a Phase-3 candidate whose ROI is already measured**, which is a
better position than most candidates start from.

One number worth carrying forward: the self-consistent firing rate for a layer
whose pre-activation σ equals its threshold is `P(Z > 1) ≈ 15.9 %`, and SpikeGPT
reports a mean firing rate of **0.15** (`01_reconnaissance.md` §4.1). That the
literature's observed rate falls out of this arithmetic is a useful independent
check on both.

---

## 6. Systems performance, re-measured on the real model

Phase 1's speed figures came from microbenchmarks. These are the real training
step at the frozen baseline (B=128, L=256, d=512, K=2), measured with
`scripts/bench/kernel_count.py`; raw JSON in `data/phase2_kernel_bench.json`.

| Variant | CUDA kernels | Per timestep | ms |
|---|---:|---:|---:|
| `lif_scan` eager, fwd, 1 layer | 3 330 | 13.008 | 58.52 |
| **`lif_scan` fused, fwd, 1 layer** | **260** | **1.016** | **8.52** |
| `lif_scan` eager, fwd+bwd | 6 909 | 26.988 | 251.18 |
| **`lif_scan` fused, fwd+bwd** | **521** | **2.035** | **17.02** |
| model forward | 528 | 1.031 | 17.96 |
| model fwd+bwd | 1 082 | 2.113 | 37.31 |
| training step, not graphed | 1 132 | 2.211 | 37.65 |
| **training step, graph replay** | 1 132 | 2.211 | **17.67** |

| Lever | Measured |
|---|---:|
| jiterator fusion, forward only | **6.87×** |
| jiterator fusion, fwd+bwd | **14.76×** |
| CUDA graph, applied to the already-fused model | **2.13×** |

End-to-end this is **~19 ms per training step** — 20 000 steps in 374–399 s, at
**0.61 GiB** peak VRAM of 7.96 available.

### 6.1 A Phase-1 conclusion that does not survive contact with the real model

`01_reconnaissance.md` §8.1.2 said the two levers "compose multiplicatively —
4.9× from a jiterator-fused LIF, ~11× from CUDA-graph capture, **~51×
together**". On the real model they do not.

Fusion delivers more than promised (14.8× on the neuron path, against 4.9×
predicted). **CUDA graphs deliver far less** — 2.13×, against 10.9–13.1× measured
in Phase 1.

The reason is not that either measurement was wrong; it is that **they are the
same lever**. Both attack CUDA launch overhead. Phase 1 measured graph capture on
a model with a GEMM inside the time loop, issuing thousands of launches per step,
where the CPU could not keep the GPU fed. Once GEMM hoisting (§1.1) and kernel
fusion have removed **92 %** of the launches — 26.99 → 2.04 per timestep — there
is far less dispatch left for graphing to hide.

The practical consequence for Phase 3 is a ranking correction: **do not budget
fusion and graph capture as independent multipliers.** Their product is bounded by
the launch overhead that exists to be removed, and most of it is already gone.

### 6.2 The 14.5 µs constant does not transfer

Per-kernel cost on the fused scan at this configuration is **32.8 µs**, against
Phase 1's headline 14.5 µs. The `measured / predicted` column in the bench table
sits at 2.25–2.51 for every row that does real work.

The *shape* of the cost model is intact and still governs the design — cost is
linear in launches, and launches are what the architecture minimises. But 14.5 µs
was measured on a bare `torch.add`, and a three-output jiterator kernel writing
3 × 65 536 elements is not a bare add. **Phase 3 must not price candidates at
14.5 µs/kernel.**

---

## 7. Results

Every figure below is the **full test split**, both protocols, scored once from
the final checkpoint after all Phase-2 design decisions were fixed. Raw JSON in
`data/phase2_final_scores.json`; campaign driver record in
`data/phase2_campaign.json`. 17 runs, 0 failures, 7.8 GPU-hours.

| Arm | Params | Test bpc (fresh) | **Test bpc (carried)** | σ over seeds | n | Firing rate |
|---|---:|---:|---:|---:|---:|---|
| **SNN baseline** (β=0.5) | 735 437 | 2.2697 | **2.2531** | 0.0046 | 5 | [0.34, 0.33] |
| Analogue control | 735 437 | 2.2796 | 2.2648 | 0.0022 | 3 | [0.33, 0.46] |
| GRU anchor *(violates I5)* | 738 019 | 1.8044 | **1.7674** | 0.0022 | 3 | — |
| SNN, β=0.9 | 735 437 | 2.3264 | 2.3042 | 0.0029 | 3 | [0.31, 0.23] |
| SNN, β=0.95 | 735 437 | 2.3696 | 2.3460 | 0.0050 | 3 | [0.30, 0.22] |

The `analogue` arm's parameter count is **identical**, not merely matched — same
modules, same shapes, same initial values under the same seed. The GRU's width was
solved for (hidden size 231) to land within ±2 %; it is +0.35 %.

### 7.1 EXP_000 — the noise floor

The pre-registered deliverable, and the number the Phase-1 literature survey found
nobody had published for this training regime (§4.5 item 6).

> **σ = 0.00461 bits/character**, over 5 seeds differing in nothing else.
> Individual values: 2.25548, 2.25417, 2.25376, 2.24518, 2.25699.

The pre-registered decision rule therefore resolves to:

| Effect size | Verdict |
|---|---|
| > **0.00922 bpc** (2σ) | real, may be adopted |
| 0.0046 – 0.0092 | within noise, published but not adopted |
| < 0.0046 | no effect |

σ is far below the 0.05 bpc replan trigger, so the Phase-4 budget stands at
**3 seeds per arm** and does not need re-costing. Note what this buys: effects as
small as **0.01 bpc are resolvable** on this setup, which is a much finer
instrument than the campaign was designed assuming.

### 7.2 Binarity is nearly free — the expensive constraint is I5

The two controls decompose the cost of the invariants, and the decomposition is
lopsided.

| Comparison | Δ bpc | In units of σ |
|---|---:|---:|
| SNN vs analogue control (isolates **I1 + I3**, binarity and the hard threshold) | **−0.0117** | 2.5σ |
| SNN vs GRU anchor (adds **I5**, no lateral recurrent matrix) | **+0.4857** | **105σ** |

**The spiking arm is not worse than its own analogue control — it is 0.0117 bpc
better**, at 2.5σ (Welch t = 4.84). The effect is small but exceeds the
pre-registered threshold, so by our own rule it is real.

This contradicts the literature's expectation. `01_reconnaissance.md` §4.3 item 5
records "binarity has a real, non-trivial cost", citing LIF-BERT at 54.9 GLUE
against BERT's 83.2. At this scale and on this task, **we cannot reproduce a
binarity penalty at all**.

Two honest caveats. First, the control emits `atan_value(v−thr, α) ∈ (0,1)` — a
bounded, squashed value with the same shape as the surrogate. That is the tightest
possible isolation of *binarity alone*, holding architecture, parameters,
initialisation and reset identical; it is **not** a comparison against a
well-designed analogue network. Second, 2.5σ on n=5 vs n=3 is a modest effect and
deserves replication before it is leaned on.

The GRU comparison is where the real cost lives. **0.486 bpc — 105σ — is the price
of invariant I5** at matched parameter count, corpus, schedule and protocol. That
is the number this project exists to shrink, and it is now measured rather than
assumed.

The Phase-3 consequence is a clear reallocation: **do not spend effort recovering
a binarity penalty, because there is not one to recover.** Spend it on making
neuron-state memory competitive with a learned recurrent matrix.

### 7.3 A pre-registered prediction that was wrong

`EXP_000` §6 declared in advance:

> β = 0.5 gives a membrane half-life below one character, so the baseline's memory
> horizon is expected to be very short and its bpc correspondingly weak.

**This was wrong, and in the interesting direction.** β = 0.5 is the *best* of the
three values tested; longer membrane memory monotonically hurts:

| β | Test bpc (carried) | Δ vs β=0.5 | In σ |
|---:|---:|---:|---:|
| **0.50** | **2.2531** | — | — |
| 0.90 | 2.3042 | +0.0511 | 11.1σ |
| 0.95 | 2.3460 | +0.0929 | 20.1σ |

So the reference configuration is not accidentally crippled by its decay constant.
The check was run to rule that out and it ruled it out — in the opposite direction
from the one predicted.

The firing rates say what is happening: as β rises, the second layer goes *quieter*
(0.33 → 0.23 → 0.22). A longer time constant makes each neuron's output depend
more on accumulated past current and less on the present character, and for
next-character prediction that trade is losing. The membrane's "memory" here is a
low-pass filter, and a low-pass filter is a poor place to store linguistic context.

This sharpens Phase 3 considerably. The §4.6 ruling admits multi-timescale decay,
adaptive thresholds, and complex/rotational state — **structured** dynamics. The
β-check says that simply turning the *magnitude* of a single scalar time constant
up does not work. Phase-3 candidates should therefore be about the *structure* of
neuron memory, not its length. That is a much better-posed question than the one
we had before this run, and it came from a check that was expected to be boring.

### 7.4 Where this sits against the literature

`2.2531 bpc at 735 K parameters on enwik8` is weak against published numbers, and
the report should say so plainly rather than hunt for a flattering framing.

The nearest published points (§4.2) are enwik8 Grid LSTM at **1.47** with 17 M
parameters and SpikeGPT at **1.283** with 46 M. We are at 735 K — 23× to 63×
smaller — and §4.2 records that the sub-17 M region of enwik8 is *genuinely
empty*, which is precisely why this project is worth running but also why no
like-for-like comparison exists.

The honest internal yardstick is our own GRU at the same parameter count, corpus,
schedule, and seeds: **1.7674**. That gap — not the gap to a 46 M-parameter model
— is the meaningful one.

### 7.5 The analogue control is 19× slower, and that is an artifact

Wall-clock per 20 000-step run: SNN **374–399 s**, GRU **442 s**, analogue
**7 436–7 582 s**.

This is **not** evidence that spiking is faster. The analogue arm has no fused
kernel — its emission is continuous, so it runs through the eager Python scan at
~27 kernels per timestep instead of ~2. Its emission is equally elementwise and
could be fused with the same machinery; nobody wrote it, because the control's job
is scientific, not fast.

Recording it because the number is sitting in the campaign log and would otherwise
invite exactly the wrong conclusion. Any energy or throughput claim comparing
these two arms is measuring which one got a CUDA kernel written for it.

---

## 8. Exit criteria

- [x] Data pipeline with SHA-256 verification; enwik8 vocab 205, text8 27, asserted
- [x] Deterministic seeded memory-mapped `uint8` loader; sampler pure in `(seed, step)`
- [x] Both evaluation protocols implemented and reported for every headline number
- [x] Eager reference LIF; kernel count documented and asserted (**A1**)
- [x] Jiterator-fused LIF in `autograd.Function`, forward **and** backward verified
      against the eager reference — and the gate mutation-tested, 21/21 (**R10**)
- [x] `--no-fused` fallback kept working and tested
- [x] CUDA-graph training step proven on the real model (**R1**)
- [x] Graphed vs eager equivalence over N steps, LR-in-graph trap tested (**R5**)
- [x] Non-spiking controls at matched parameter count — one *identical*, one anchor
- [x] Per-layer firing-rate logging from step 0 (**R4**) — which is what caught §5
- [x] Seed-variance measurement, 5 seeds: **σ = 0.00461 bpc** (**R8**)
- [x] Reset rule swappable behind one flag, all four modes tested
- [x] Detached-process launcher with file logging and resumable checkpoints (**R6**)
- [x] Discrepancy protocol applied to the initialisation finding (§5)
- [ ] **Reviewed by Elliot — Phase 3 does not begin until this is signed off**

### 8.1 What Phase 3 inherits

Four things changed the plan, and all four came from measurement:

1. **Binarity is nearly free; I5 costs 0.486 bpc.** Reallocate effort accordingly.
2. **Structure, not magnitude.** Turning β up hurts monotonically. The §4.6-admitted
   multi-timescale / adaptive / rotational dynamics are the candidates worth ranking.
3. **Fusion and CUDA graphs are one lever, not two.** Do not price them as ~51×
   combined; on the real model it is 14.8× then 2.13×, and 92 % of launches are
   already gone.
4. **The instrument is finer than expected.** σ = 0.0046 bpc resolves 0.01 bpc
   effects at 3 seeds, so Phase 4 can ask sharper questions than budgeted.

One open item is carried forward rather than silently dropped: the variance-scaled
initialisation (§5.4) is implemented and measured but deliberately not in the
baseline. It enters Phase 3 as a ranked candidate with its ROI already known.

---

## 9. Changelog

| Rev | Change |
|---|---|
| 1 | Phase-2 deliverable: implementation, gates, two systems findings, the initialisation discrepancy, the campaign, and the noise floor. |
