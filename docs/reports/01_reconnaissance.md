# Phase 1 — Reconnaissance

**Project:** Character-level spiking neural-network language model (v2)
**Author:** Elliot Asher Caudill
**Date:** 2026-08-01
**Status:** Phase-1 deliverable. Awaiting review before Phase 2 begins.

---

## 0. Executive summary

Four instrumented benchmarks were run on the target machine before any modelling
decision was made. They produced one dominant finding that reshapes the entire
plan:

> **A surrogate-gradient spiking LM on this box is not compute-bound, not
> memory-bound, and not VRAM-bound. In eager mode it is bound almost entirely by
> CPU-side CUDA kernel-launch latency, at a flat ~14.5 µs per kernel.**

The measured cost model is embarrassingly simple and holds to within a few
percent across three orders of magnitude of tensor size:

```
t_step  ≈  14.5 µs  ×  (elementwise kernels per timestep)
t_train ≈  t_step   ×  (sequence length L)  ×  (fwd + bwd + optimiser)
```

Batch size and layer width are nearly *absent* from that equation. A naive LIF
timestep issues ~13–18 kernels, so it costs ~250–320 µs whether it is processing
1 sequence or 512, at width 128 or width 2048.

Since the cost is *launches*, the winning moves are the ones that issue fewer of
them. Three mechanisms do that, and all three were verified on this machine:

1. **Custom fused CUDA kernels ARE available — this reverses the audit's own
   first conclusion.** `torch.compile(inductor)` is genuinely dead
   (`TritonMissing`, reproduced). But **NVRTC ships inside the PyTorch wheel**
   (`torch/lib/nvrtc64_130_0.dll`), and `torch.cuda.jiterator` uses it to compile
   arbitrary *elementwise* CUDA C++ at runtime — no `nvcc`, no MSVC, no Triton.
   A fused LIF kernel compiles in 0.17 s, is numerically exact (spike error
   **0.0**), and runs **4.86× faster than the eager kernel chain**. The LIF
   update is purely elementwise, so it is precisely the one custom kernel this
   project needs. *An earlier draft of this report concluded "fusion cannot be
   bought"; that was wrong, and §3.11 records how it was caught.*
2. **`torch.cuda.CUDAGraph` capture works** on a full forward + backward +
   `AdamW(capturable=True)` step, delivering **10.9–13.1×** on a sequential,
   char-LM-shaped model. It **composes with jiterator**: a fused LIF step inside
   a captured graph costs **1.76 µs against 90.27 µs eager — ~51× combined.**
3. **A reset-free leaky recurrence has an exact parallel form.**
   `v_t = decay·v_{t−1} + i_t` is a triangular-Toeplitz matmul, measured at
   **77–160×** the sequential loop and exact to fp32 rounding (max error
   2.9e-06). It is only legal without reset — which is a declared free
   parameter, making "reset vs no reset" the highest-stakes experiment in the
   programme.

Two further findings:

4. **VRAM is a non-issue at this scale.** The largest configuration tested
   (batch 256, width 768, unroll 512) peaked at 3.0 GiB of 7.96 GiB.
5. **Compute is abundant relative to data.** One epoch of enwik8's 90 MB
   training split takes **2–6 minutes**, so Phase 4 can afford proper multi-seed
   ablations rather than single-run anecdotes.

Two corrections a less careful audit would have shipped, both recorded in full:

* §3.10 — *the eager-mode "batch and width are free" result does not survive
  CUDA-graph capture.* Once graphing removes the dispatch bottleneck the workload
  becomes GPU-bound and width costs real time again.
* §3.11 — *the "no custom kernels" conclusion above was overturned by the
  literature review and then confirmed by re-measurement.* Both the original
  claim and its refutation are on the record.

---

## 1. Project goals and scope

### 1.1 What this project is

A character-level language model whose hidden computation is genuinely spiking,
trained end-to-end by surrogate-gradient backpropagation-through-time, developed
to a publishable standard of rigour on a single consumer GPU.

It is a **portfolio and research artifact**. That means the standard of evidence,
the honesty of the reporting, and the reproducibility of the repository are
first-class deliverables alongside the bits-per-character number.

### 1.2 Relationship to v1

A prior version of this project (git history through `b5890f2`) reached
**1.2362 bits/character with 5.04 M parameters** on a bespoke ~639 MB modern-prose
corpus. That lineage is preserved in git and is *not* being restored into the
working tree; v2 is a clean-slate rebuild under this protocol. v1 serves as a
sanity anchor, not as a codebase to inherit.

The v1 number is **not** a target to beat directly, because v2 trains on a
different corpus (§1.4) and cross-corpus bpc comparison is invalid.

### 1.3 Architectural invariants (ratified — not open for re-litigation)

These define what makes this "a real SNN" for the purposes of this project and
may never be traded away for a metric gain:

| # | Invariant |
|---|---|
| I1 | **Binary spikes** carry all information *between* layers |
| I2 | **Leaky membrane integration** of input current over time |
| I3 | **Hard threshold** firing (not a smooth/continuous activation) |
| I4 | **Surrogate-gradient BPTT** as the training rule |
| I5 | **No explicit lateral recurrent weight matrix** (memory must live in neuron state) |

Deliberately left **free**, to be settled by ablation rather than by assertion:

* the number of micro-steps **T** per input character, including `T=1`
  ("direct coding", where analogue input is injected directly and only the
  inter-layer signals are binary);
* the **membrane reset rule** — reset-to-zero, reset-by-subtraction, detached
  reset, or no reset.

I5 is the scientifically interesting constraint. It forces all long-range memory
to come from *neuron dynamics* — decay constants, adaptation, multiple
timescales — rather than from a learned recurrent mixing matrix. That is a
sharper and more defensible research question than "can we make a spiking net
score well".

### 1.4 Corpus decision

**Primary corpus: enwik8, with text8 as a secondary check.** Ratified 2026-08-01.

Rationale: these are *the* standard character-level benchmarks, with fixed
90/5/5 MB splits, so our bits-per-character lands on the same axis as decades of
published LSTM, Transformer-XL, and spiking-LM results. For a portfolio piece,
comparability to the literature is worth more than continuity with v1's private
corpus.

Availability was verified rather than assumed — both mirrors respond:

| Source | Status | Size |
|---|---|---|
| `http://mattmahoney.net/dc/enwik8.zip` | **OK** | 34.8 MB (100 MB unpacked) |
| `http://mattmahoney.net/dc/text8.zip` | **OK** | 29.9 MB (100 MB unpacked) |
| `https://data.deepai.org/enwik8.zip` (mirror) | **OK** | 34.8 MB |
| `https://huggingface.co/datasets/LTCB/enwik8` | 404 | — |

### 1.5 Success criteria

| Criterion | Target |
|---|---|
| **Primary** | Best achievable enwik8 test bpc under invariants I1–I5, at a parameter budget we report honestly |
| **Control** | A same-parameter-count non-spiking RNN baseline, so the ANN→SNN gap is *measured*, not hand-waved |
| **Rigour** | Every claimed improvement exceeds seed noise, measured across ≥3 seeds |
| **Reproducibility** | Clone → one command → reproduces the headline number |
| **Honesty** | Negative results retained; energy claims stated as theoretical-only (§4) |

---

## 2. System audit

All figures below were read from the machine, not assumed.

### 2.1 Hardware

| Component | Value |
|---|---|
| CPU | AMD Ryzen 7 8700F — 8 cores / 16 threads, max 4101 MHz |
| System RAM | 15.4 GiB |
| GPU | NVIDIA GeForce RTX 5060 |
| GPU architecture | Blackwell, compute capability **sm_120** |
| VRAM | **7.96 GiB** |
| Streaming multiprocessors | 30 |
| Max SM clock | 3090 MHz |
| Max memory clock | 14001 MHz |
| Power cap | 145 W |
| Driver model | **WDDM** ← root cause of §3.3 |
| Driver / CUDA UMD | 610.74 / 13.3 |
| Disk (C:) | 1835.3 GB total, **1005.6 GB free** |

### 2.2 Software

| Component | Version |
|---|---|
| OS | Windows 11 Home, build 10.0.26200 |
| Python | 3.14.6 (MSC v.1944, 64-bit) |
| PyTorch | **2.12.1+cu130** |
| CUDA runtime (torch) | 13.0 |
| cuDNN | 9.2.0 |
| NumPy | 2.5.0 |
| snnTorch | 1.0.0 |
| tokenizers | 0.23.1 |
| matplotlib / pytest | 3.11.0 / 9.1.1 |

### 2.3 Absent toolchain — and the compiler that is present anyway

| Tool | Status | Consequence |
|---|---|---|
| `triton` | **not installed** (`ModuleNotFoundError`) | `torch.compile(backend="inductor")` fails |
| `nvcc` (CUDA toolkit) | **not on PATH** | no ahead-of-time compiled CUDA extensions |
| `cl.exe` (MSVC) | **not on PATH** | no C++ extensions, no inductor C++ backend |
| `gcc` / `ninja` / `cmake` | **not on PATH** | no out-of-tree native builds |

Confirmed by execution, not inference — `torch.compile(f, backend="inductor")` raises:

```
TritonMissing: Cannot find a working triton installation.
```

**But this does not mean custom kernels are impossible.** PyTorch bundles the
NVIDIA runtime compiler in its own wheel:

```
torch/lib/nvrtc64_130_0.dll
torch/lib/nvrtc64_130_0.alt.dll
torch/lib/nvrtc-builtins64_130.dll
torch/lib/caffe2_nvrtc.dll
```

`torch.cuda.jiterator` compiles CUDA C++ source through that bundled NVRTC at
runtime, requiring only the driver. **Elementwise custom kernels are therefore
fully available on this machine** (§3.11), which is decisive because the LIF
neuron update is entirely elementwise.

What remains genuinely impossible is anything **non-elementwise**: reductions,
gather/scatter, or matmul in a custom kernel. That rules out the sparse
active-neuron-index backward that gives the largest published SNN training
speed-ups, and it cannot be worked around here.

| Capability | Available? |
|---|---|
| Elementwise custom CUDA (fused LIF, surrogate, reset) | **Yes** — jiterator + bundled NVRTC |
| CUDA-graph capture of a full training step | **Yes** |
| `torch.compile` / Inductor autotuning | No |
| Custom reduction / gather-scatter / matmul kernels | No |
| Sparse-index (active-neuron) backward | No |

---

## 3. Empirical capability measurements

Five audit scripts, committed under `scripts/audit/`, with raw JSON in
`docs/reports/data/`. All are re-runnable (§10).

### 3.1 Arithmetic throughput — `4096³` matmul

| Precision | TFLOP/s | vs fp32 |
|---|---:|---:|
| fp32 | 11.91 | 1.00× |
| tf32 | 20.05 | 1.68× |
| **bf16** | **40.62** | **3.41×** |
| fp16 | 39.87 | 3.35× |

bf16 and fp16 are equivalent in speed (both ~3.4× fp32), and autocast was verified
working for both. Which to *use* is subtler than the usual "prefer bf16" heuristic,
because a spiking neuron compares a membrane potential against a fixed threshold —
see §3.11, where the representable spacing at v = 1.0 turns out to differ by exactly
8× between the two. The resolution is to keep membrane state and the threshold
comparison in fp32 and cast only the GEMMs, which costs almost nothing because the
state tensors are tiny next to the weights.

### 3.2 Memory bandwidth

| Working set | Effective bandwidth |
|---|---:|
| 256 MiB (HBM-resident) | **383.5 GB/s** |
| 12 MiB (L2-resident) | 754.6 GB/s |

The 383.5 GB/s figure is sustained DRAM bandwidth; at a 14001 MHz GDDR7 clock on
a 128-bit bus (~448 GB/s theoretical, spec-sheet figure) that is ~86% efficiency,
which is healthy. The higher small-tensor number is L2 cache, not DRAM.

### 3.3 Kernel-launch latency — the governing constant

A single `torch.add` was timed across five tensor sizes:

| Elements | Bytes moved | Time | Effective GB/s |
|---:|---:|---:|---:|
| 16 | 192 B | 14.67 µs | 0.0 |
| 256 | 3 KB | 15.19 µs | 0.2 |
| 4 096 | 48 KB | 13.40 µs | 3.7 |
| 65 536 | 768 KB | 13.13 µs | 59.9 |
| 1 048 576 | 12 MB | 16.67 µs | 754.6 |

**A 16-element add and a 65,536-element add cost the same.** Work is free below
roughly a megabyte; only the *launch* is billed. This is characteristic of the
Windows WDDM driver model, where every submission crosses a kernel-mode boundary.

### 3.4 The cost model is literally linear in kernel count

Identical tensor shapes, varying only how many elementwise ops are issued per
step (64 steps):

| Ops / step | Total kernels | Wall-clock | µs per kernel |
|---:|---:|---:|---:|
| 1 | 64 | 0.95 ms | 14.87 |
| 2 | 128 | 1.86 ms | 14.52 |
| 4 | 256 | 3.45 ms | 13.46 |
| 8 | 512 | 7.16 ms | 13.98 |
| 16 | 1 024 | 14.83 ms | 14.48 |
| 32 | 2 048 | 29.65 ms | 14.48 |

Dead linear across a 32× range, at **14.5 µs ± 0.7 µs per kernel**. This
promotes "kernels issued per timestep" from an implementation detail to *the*
primary performance metric of the model architecture.

A textbook LIF step — `mem*decay`, `+current`, `-threshold`, `>0`, `.to(dtype)`,
surrogate `sigmoid`, the straight-through add/sub pair, `spike*threshold`,
`mem-reset`, plus the input projection and a sequence index — issues ~13–18
kernels, predicting ~190–260 µs/step. Measured: **316 µs/step** at batch 64,
width 512 (§3.5). The model explains the observation.

### 3.5 Eager-mode scaling: batch and width are nearly free

Forward-only stacked LIF, `no_grad`, L=64.

**Batch sweep** (width 512):

| Batch | Wall-clock | µs/step | tokens/s |
|---:|---:|---:|---:|
| 1 | 20.26 ms | 316.6 | 3 158 |
| 8 | 20.08 ms | 313.8 | 25 496 |
| 32 | 21.29 ms | 332.7 | 96 191 |
| 128 | 21.44 ms | 335.0 | 382 070 |
| 512 | 23.34 ms | 364.7 | **1 403 873** |

A **512× increase in batch costs 15% more wall-clock** and yields **444× the
throughput**.

**Width sweep** (batch 64):

| Width d | Wall-clock | µs/step |
|---:|---:|---:|
| 128 | 24.07 ms | 376.1 |
| 256 | 23.47 ms | 366.7 |
| 512 | 23.58 ms | 368.5 |
| 1024 | 21.83 ms | 341.0 |
| 2048 | 21.45 ms | 335.2 |

Flat — indeed mildly *decreasing*, which is clock-boost and measurement noise, not
a real inverse relationship. A 16× width increase is free in eager mode.

### 3.6 Eager training throughput and VRAM

Full fwd + bwd + `AdamW`, batch 64, width 512:

| Unroll L | ms/step | tokens/s | Peak VRAM |
|---:|---:|---:|---:|
| 64 | 60.9 | 67 276 | 0.098 GiB |
| 128 | 127.4 | 64 314 | 0.125 GiB |
| 256 | 250.1 | 65 522 | 0.180 GiB |
| 512 | 491.7 | 66 642 | 0.289 GiB |

Throughput is flat in L (cost is exactly linear in unroll), and **VRAM is
negligible** — 0.29 GiB of 7.96 GiB at L=512.

### 3.7 `torch.compile` — measured, and mostly unusable

Small LIF-update function, batch 64 × width 512:

| Backend | Status | Time |
|---|---|---:|
| eager (reference) | — | 91.27 µs |
| `inductor` | **FAILED — `TritonMissing`** | — |
| `aot_eager` | OK | 170.48 µs (**1.87× slower**) |
| `cudagraphs` | OK | 114.80 µs (**1.26× slower**) |

Both working backends are *slower* than eager at this granularity: they add
guard and dispatch overhead without buying fusion. **`torch.compile` is written
off for this project.** Manual CUDA-graph capture (§3.8) is not — the difference
is that manual capture wraps hundreds of timesteps in one replay, amortising the
overhead that `torch.compile`'s per-call wrapper cannot.

### 3.8 CUDA graphs — the one lever that works

Full fwd + bwd + `AdamW(capturable=True)` step captured and replayed
(batch 64, width 512):

| Unroll L | Eager | Graphed | Speed-up | tokens/s | Capture cost | Peak VRAM |
|---:|---:|---:|---:|---:|---:|---:|
| 32 | 31.8 ms | 2.43 ms | **13.1×** | 844 331 | 0.04 s | 0.208 GiB |
| 64 | 56.8 ms | 5.04 ms | 11.3× | 812 371 | 0.05 s | 0.283 GiB |
| 128 | 118.3 ms | 9.71 ms | 12.2× | 843 366 | 0.10 s | 0.370 GiB |
| 256 | 212.1 ms | 19.39 ms | 10.9× | 845 059 | 0.22 s | 0.482 GiB |
| 512 | 434.2 ms | 38.96 ms | 11.1× | 841 143 | 0.41 s | 0.645 GiB |

Findings:

* The speed-up is **stable at ~11–13×** across a 16× range of unroll length.
* Capture cost is linear and trivial (0.41 s for a 512-step unroll) — paid once.
* Throughput is **flat at ~843 K tokens/s**, confirming graphed cost is linear in
  L with no superlinear blow-up.
* A capturable optimiser works; `AdamW(capturable=True)` captured without
  incident.

### 3.9 Hoisting the input projection — a modest, honest result

The obvious algebraic win is to compute every timestep's input current in *one*
big matmul outside the loop instead of L small ones inside it. Measured at batch
64, width 512, L=128:

| Variant | Wall-clock |
|---|---:|
| Naive (projection inside loop) | 37.73 ms |
| Lean (projection hoisted, fused ops) | 31.97 ms |
| **Speed-up** | **1.18×** |

Only 1.18×, because hoisting removes *one* kernel from a body that issues a dozen
or more. It is worth doing but it is a rounding error next to CUDA graphs. Recorded
here so that Phase 3 ranks it correctly rather than over-investing in it.

### 3.10 After graphing, the free lunch ends — a corrected finding

§3.5 showed batch and width to be nearly free. It would have been easy to carry
that straight into the architecture plan and conclude "build it enormous". That
would have been **wrong**, because §3.5 was measured in eager mode, where the
*CPU* is the bottleneck and the GPU is mostly idle. CUDA-graph capture removes
the dispatch cost — and hands the bottleneck back to the GPU.

So the scaling sweeps were re-run *inside* a captured graph, on a realistic model
(embedding + 2 stacked LIF layers + 205-way head, full fwd/bwd/`AdamW`):

**Graphed batch sweep** (d 512, L 128):

| Batch | Graphed ms | tokens/s | Peak VRAM |
|---:|---:|---:|---:|
| 16 | 19.61 | 104 446 | 0.162 GiB |
| 32 | 16.27 | 251 707 | 0.245 GiB |
| 64 | 19.17 | 427 393 | 0.350 GiB |
| 128 | 25.40 | 645 011 | 0.497 GiB |
| 256 | 39.72 | 825 024 | 0.727 GiB |
| 512 | 57.34 | **1 143 000** | 1.128 GiB |

Batch is still *cheap* but no longer *free*: a 16× batch increase (32→512) costs
3.5× the time for 4.5× the throughput. Below batch 32 the step is still
latency-bound — which is why batch 16 is measurably **slower** than batch 32.

**Graphed width sweep** (batch 64, L 128):

| Width d | Params | Graphed ms | tokens/s |
|---:|---:|---:|---:|
| 256 | 0.24 M | 15.52 | 527 977 |
| 512 | 0.73 M | 19.29 | 424 683 |
| 768 | 1.49 M | 28.24 | 290 133 |
| 1024 | 2.52 M | 35.74 | 229 238 |
| 1536 | 5.35 M | 69.02 | 118 683 |

**Width now costs real time.** A 6× width increase costs 4.4× the wall-clock —
the exact opposite of the flat eager-mode result in §3.5. Parameters grow 22×
over that range, so compute-per-parameter still improves with width, but throughput
does not.

**This is the most important correction in the audit.** Had the plan been drawn
from eager-mode numbers alone it would have specified a needlessly wide model and
mis-ranked half of Phase 3. Both regimes are now on record:

| Regime | Bound by | Batch | Width |
|---|---|---|---|
| Eager | CPU dispatch (~14.5 µs/kernel) | free | free |
| **Graphed** | **GPU execution** | **cheap** | **costs ~linearly in d** |

**Candidate configurations**, measured end-to-end:

| Configuration | Params | ms/step | tokens/s | Mtok/hour | Peak VRAM |
|---|---:|---:|---:|---:|---:|
| batch 128, d 512, L 256 | 0.73 M | 50.67 | 646 681 | **2 328** | 1.166 GiB |
| batch 256, d 512, L 256 | 0.73 M | 117.59 | 557 321 | 2 006 | 1.566 GiB |
| batch 128, d 1024, L 256 | 2.52 M | 100.10 | 327 353 | 1 178 | 1.642 GiB |
| batch 256, d 768, L 512 | 1.49 M | 224.30 | 584 364 | 2 104 | 3.008 GiB |

Even the largest peaks at **3.0 GiB of 7.96 GiB**, confirming VRAM headroom for
deeper and wider models than any configuration tested here.

### 3.11 Verifying three claims from the literature review — one overturns §2.3

The literature survey (§4) was audited by adversarial fact-checkers, and its
systems section asserted three things that contradicted or materially changed the
measurements above. Rather than accept them on report, each was re-tested from
scratch (`scripts/audit/05_verify_literature_claims.py`).

#### C1 — Custom elementwise CUDA kernels are available. **CONFIRMED.**

This overturns the conclusion in the first draft of §2.3 that "fusion cannot be
bought". NVRTC ships inside the torch wheel, and `torch.cuda.jiterator` compiles
against it with no `nvcc`, no MSVC, and no Triton.

A fused LIF kernel — decay, integrate, threshold, subtract-reset, all in one
kernel emitting two outputs:

| Property | Result |
|---|---|
| Compile + first run | **0.174 s** |
| Membrane max error vs fp32 eager | 4.77e-07 (pure fp32 rounding) |
| **Spike max error vs fp32 eager** | **0.0 — bit-exact** |
| dtypes supported | fp32, bf16, fp16 all OK |
| Single step: eager → fused | 90.27 µs → **18.57 µs (4.86×)** |
| Sequential unroll L=128: eager → fused | 10.65 ms → **2.66 ms (4.00×)** |
| Survives CUDA-graph capture | **Yes — 1.76 µs/step** |
| `requires_grad` on output | **False** ← backward must be hand-written |

The last two rows are the important ones. Jiterator **composes with CUDA graphs**,
and the combination is what matters:

| Path | Per LIF step | vs eager |
|---|---:|---:|
| Eager kernel chain | 90.27 µs | 1× |
| Fused (jiterator) | 18.57 µs | 4.9× |
| **Fused + graph-captured** | **1.76 µs** | **~51×** |

The catch is real and must be budgeted: jiterator is a private/beta API, it is
**elementwise-only**, it caps at 8 inputs and 8 outputs, and it returns tensors
with no autograd history. Using it means hand-writing the surrogate-gradient
backward inside a `torch.autograd.Function` — a genuine correctness hazard, hence
risk **R10** in §7.

#### C2 — bf16 is bad for thresholded spiking. **Spacing confirmed; consequence not reproduced.**

Representable spacing immediately above the firing threshold v = 1.0, via
`torch.nextafter`:

| dtype | Spacing at 1.0 | As % of threshold |
|---|---:|---:|
| bf16 | 0.0078125 | 0.78% |
| fp16 | 0.0009765625 | 0.098% |
| fp32 | 1.19e-07 | 0.000012% |

**fp16 is exactly 8× finer than bf16 at the threshold** — confirmed, and it does
invert the usual "bf16 is the safe default" instinct for this specific workload.

However, the review's accompanying behavioural claim (spike-count drift of −0.25%
for bf16 versus −0.01% for fp16) **did not reproduce**. Over 64 timesteps at
batch 64 × width 512, both dtypes drifted **identically, +0.0395%** against an
fp32 reference. The review had itself tagged that sub-claim `(unverified)`, which
was the correct call.

**Conclusion:** the spacing argument is real and worth respecting, but there is no
measured evidence here that it changes spiking behaviour. Keep membrane state and
the threshold comparison in fp32 — cheap insurance that makes the point moot — and
treat the GEMM dtype as an ordinary Phase-4 ablation rather than a settled matter.

#### C3 — Reset-free leak has an exact parallel form. **CONFIRMED, and larger than claimed.**

Without reset, `v_t = decay·v_{t−1} + i_t` is a first-order linear recurrence, so
`v = K @ i` with `K[l,s] = decay^(l−s)` for l ≥ s — a triangular Toeplitz matmul
that replaces the entire sequential loop with one GEMM:

| Configuration | Sequential | Toeplitz | Speed-up | Max abs error |
|---|---:|---:|---:|---:|
| L=128, B=16, H=512 | 4.111 ms | 0.038 ms | **108×** | 1.91e-06 |
| L=256, B=16, H=512 | 8.176 ms | 0.106 ms | **77×** | 2.38e-06 |
| L=1024, B=4, H=256 | 29.429 ms | 0.184 ms | **160×** | 2.86e-06 |

It is **exact**, not an approximation — the residual is fp32 rounding only.

Two hard conditions. First, it is valid **only without reset**; a hard threshold
with reset makes the recurrence nonlinear and non-associative. Since the reset rule
is a declared free parameter (§1.3), this makes "reset vs no reset" the
highest-stakes single experiment in the programme — it is worth up to two orders of
magnitude of throughput.

Second, an underflow hazard the review flagged and this audit quantified precisely:

| Decay | `decay^n` underflows to fp32 zero at | Value at n=1023 |
|---:|---:|---:|
| 0.90 | **n = 987** | **0.0** |
| 0.95 | n = 2028 | 1.63e-23 |
| 0.99 | n = 10346 | 3.43e-05 |

At decay 0.9 the Toeplitz kernel **silently truncates context beyond ~987
positions**. Any long-context parallel form needs an explicit underflow analysis,
not a spot check.

---

## 4. Literature and domain context

> **Condensed.** The full survey — every claim, every citation, and the complete
> audit trail of what was retracted and corrected — is in
> [`01a_literature_review.md`](01a_literature_review.md). This section carries only
> what changes a decision in this project.

**Method.** Six parallel domain surveys (spiking LMs; surrogate gradients; neuron
models; efficient sequence modelling; character-LM baselines and metrics; SNN
systems), each then audited by an independent adversarial fact-checker instructed
to default to skepticism and verify every named paper and number against primary
sources. The audit caught a **fabricated author name**, a wrong bpc value, a
misdescribed readout mechanism, and a widely-quotable-but-misleading "+41.9 point"
figure. Claims marked fabricated or unsupported were removed; corrected values are
used below. Three of its systems claims were independently re-measured by the
author in §3.11.

### 4.1 The one directly comparable data point

**SpikeGPT** (Zhu, Zhao, Li, Eshraghian; arXiv:2302.13939) is the only spiking LM
with a published enwik8 bits-per-character figure:

| Model | Params | enwik8 test bpc | Context |
|---|---:|---:|---:|
| SpikeGPT (spiking) | 46.1 M | **1.283** | 1024 |
| SpikeGPT (spiking) | 46.1 M | 1.262 | 3072 |
| Vanilla Transformer | 43.0 M | **1.137** | 1024 |
| Reformer | 40.1 M | 1.195 | 1024 |
| 7-layer stacked LSTM | — | 1.670 | 1024 |

At **equal context the spiking penalty is 0.146 bpc** — the commonly-quoted 0.125
compares against a 3× longer context and is not an equal-protocol comparison.

Two facts from it govern our planning. First, **SpikeGPT-46M cost ~48 GPU-hours**
(12 h × 4×V100) — approximately this entire project's budget for a single run.
Matching it is out of reach; the target must be set elsewhere (§4.3).

Second, and far more useful: **SpikeGPT's configuration is almost exactly this
project's invariant set.** It uses **T = 1** (the token sequence *is* the time
axis), **no lateral recurrent weight matrix** (element-wise learnable decay plus
token-shift), β = 0.5, threshold 1, hard reset, and an arctangent surrogate, at
12 layers × d 512, with a reported mean firing rate of **0.15**. The best published
spiking-LM result was obtained under our constraints, not in spite of them.

### 4.2 Reference bits-per-character, by corpus

> **Cross-corpus comparison is invalid** and this is not a pedantic caveat: the
> same MEGABYTE architecture spans 0.411 bpc (Code) to 1.007 (Books) at fixed
> compute, and the same 12-layer Transformer scores 1.11 on enwik8 but 1.18 on
> text8. Rows compare only within a block, and only at matched protocol.

**text8 — the block that matters most, because it contains the ~5 M-param end:**

| Model | Params | Test bpc |
|---|---:|---:|
| vanilla RNN | ~5 M | 1.69 |
| GRU | ~5 M | 1.53 |
| **LSTM (tuned)** | **~5 M** | **1.50** |
| **TCN (5L)** | **~5 M** | **1.45** |
| BN-LSTM | ~16 M | 1.36 |
| Transformer T12 | 44 M | 1.18 |
| Transformer-XL (24L) | 277 M | 1.08 |

**enwik8 — no strong sub-17 M points exist; the small end is genuinely empty:**

| Model | Params | Test bpc |
|---|---:|---:|
| Grid LSTM | 17 M | 1.47 |
| VD RHN | 21 M | 1.30 |
| mLSTM | 46 M | 1.24 |
| **SpikeGPT (spiking)** | **46.1 M** | **1.283** |
| Transformer-XL (12L) | 41 M | 1.06 |
| Sparse Transformer (30L) | 95 M | 0.99 |

**PTB-char**, useful as a cheap ablation harness (5.9 M chars, well under an hour
per run): FS-LSTM-4 at 6.5 M params reaches **1.193** — the closest published point
to a small parameter budget on any standard character benchmark.

For reference only, v1's **1.2362 bpc @ 5.04 M params** belongs in *none* of these
blocks: different corpus, different vocabulary, different protocol.

### 4.3 What the literature settles

1. **T = 1 is the right default.** SpikeGPT's competitive result uses it; encoder
   work caps the useful T at 4 and reports degradation at 8 and 12; larger T
   multiplies both the gradient horizon problem and — on this box — the kernel
   launch count. Bound any ablation at T ∈ {1, 2, 4}.
2. **The readout never spikes.** Every competitive spiking LM keeps embeddings,
   normalisation, and the output head in floating point, emitting binary spikes
   only on layer-to-layer activations. Spike-count logits at T=5 give just six
   distinguishable levels — a hard bpc floor no neuron tuning recovers.
3. **Surrogate *shape* is robust; surrogate *scale* is fragile.** Reported
   large gaps between arctan and fast-sigmoid are confounded by a ~19× width
   mismatch at library defaults. Keep `atan(α=2)`, lock threshold at 1.0, and
   sweep width rather than shape.
4. **Adaptation can substitute for lateral recurrence.** This is the permission
   slip for invariant I5: an adaptive per-neuron threshold with its own slower
   learned decay is not a recurrent weight matrix, and SE-adLIF reaches 95.81 on
   SHD against 90.27 for a *recurrent* LIF. It is the strongest available
   evidence that our constraint need not be crippling.
5. **Binarity has a real, non-trivial cost.** Naive LIF-BERT scores 54.9 GLUE
   against BERT's 83.2; SpikeLM recovers to 76.5 — but does so by abandoning
   binary spikes for signed multi-level ones, which our invariants forbid. The
   honest reading is that we have opted into a measurable penalty, and our job
   is to quantify it rather than tune it away.
6. **Every strong spiking LM except SpikeGPT distills from an ANN teacher.**
   Training from scratch with surrogate BPTT and no teacher is a harder game
   than most of the comparison set. This should be stated in the write-up, not
   hidden.
7. **Spike sparsity buys nothing on a GPU.** GPUs exploit *weight* sparsity
   (static, structured); spike sparsity is *activation* sparsity — dynamic and
   unpredictable. Confirmed independently by §3.5's flat scaling curves.

### 4.4 Energy claims — the honest position

The defensible arithmetic basis is Horowitz (ISSCC 2014): 45 nm 32-bit FP add
0.9 pJ, multiply 3.7 pJ, hence a ~4.6 pJ MAC. Note his own 45 nm DRAM read is
**640 pJ** — data movement dominates, so arithmetic-only accounting is not by
itself defensible.

**Policy for this project:** report mean firing rate and synaptic operations per
token as *architectural measurements*. Any picojoule figure must be labelled
verbatim as a "hypothetical 45 nm-ASIC projection, arithmetic only, data movement
unaccounted." **Never present GPU wall-clock or watts as evidence of SNN energy
efficiency.** The one honest hardware datapoint located is SpiNNaker2: 18.3× less
energy at 8.6× slower.

### 4.5 What the literature does NOT settle — our experiments

These are the gaps that make the project worth running, and they map directly onto
Phase 4:

1. **The bpc-versus-parameters curve for spiking LMs below ~45 M is unmeasured.**
   There is *no published spiking character-LM in the 5–20 M range on any standard
   corpus.* Any target we set is an extrapolation until we measure it.
2. **Whether T > 1 buys anything when the token sequence already supplies a time
   axis.** Every published T-ablation is on encoder- or vision-shaped models where
   T is the *only* temporal dimension. The T=1 vs T=4 comparison for a causal
   spiking char-LM has not been run.
3. **Whether soft, hard, or no reset is right for text.** All existing comparisons
   are vision and they contradict each other. No character-level bpc number exists
   for a reset-free spiking neuron — and per §3.11 C3, this single question is
   worth up to two orders of magnitude of throughput.
4. **What the network's actual memory horizon is.** Simple leak arithmetic
   predicts single-digit characters. Whether that is the real limit is directly
   measurable by force-resetting membrane state every *k* characters and scoring.
5. **How much of the binarity penalty is recoverable without a teacher.**
6. **The seed-to-seed standard deviation of bpc for surrogate-gradient SNN
   training.** Everyone imports a word-level LSTM figure. Nobody has measured it
   for this training regime — and every ablation conclusion depends on it, so we
   measure it first (§8.3).

### 4.6 A definitional question to settle before implementation

Invariant I5 forbids an "explicit lateral recurrent weight matrix". The survey
raises a genuine boundary case that should be decided **in advance and in
writing**, not opportunistically once results are in: does I5 admit a learned
per-channel kernel over *time* (diagonal in the neuron axis, mixing only a
neuron's own history), or a complex-valued / 2×2 rotational membrane state?

Neither introduces neuron-to-neuron mixing, so both arguably satisfy the letter of
I5 — and temporal kernels are what make a neuron parallel-scannable, sidestepping
the launch-latency wall entirely. **This is flagged as an open decision for Elliot,
not resolved here**, because deciding it after seeing which answer scores better
would be exactly the kind of post-hoc rationalisation this protocol exists to
prevent.

---

## 5. Expected bottlenecks, ranked

| # | Bottleneck | Evidence | Mitigation |
|---|---|---|---|
| **B1** | **CUDA kernel-launch latency (~14.5 µs/kernel)** dominates everything in eager mode | §3.3, §3.4 | CUDA-graph capture (11–13×); minimise kernels/timestep; hoist projections |
| **B2** | **Sequential timestep loop** is irreducible for a hard-threshold-**with-reset** neuron | §3.4, §3.11 C3 | Test reset removal as a first-class ablation — it unlocks an exact 77–160× parallel form |
| **B3** | **No autograd on jiterator kernels** ⇒ the surrogate backward must be hand-written | §3.11 C1 | `torch.autograd.Function` + mandatory gradient-equivalence test vs eager; `--no-fused` fallback path |
| **B4** | **Graph capture rigidity** — static shapes and static memory addresses | §3.8 | Fixed batch/L; pre-allocated static I/O buffers; drop last partial batch |
| **B5** | **No non-elementwise custom kernels** (no reductions, gather/scatter, matmul) | §2.3 | Sparse active-neuron backward is permanently out of reach — do not budget time for it |
| **B6** | Width costs real time *after* graphing | §3.10 | Choose width by measured bpc-per-hour, not by "it's free" |
| **B7** | **fp32 underflow in any parallel decay form** — `0.9ⁿ` hits zero at n=987 | §3.11 C3 | Explicit underflow analysis before adopting a Toeplitz/scan form; constrains decay × context jointly |
| **B8** | Low-precision thresholding — bf16 spacing at v=1.0 is 0.78% of threshold | §3.11 C2 | Keep membrane state and threshold comparison in fp32; GEMM dtype is a Phase-4 ablation |
| **B9** | **Host RAM 15.4 GiB** constrains corpus handling | §2.1 | `uint8` memory-mapped corpus (100 MB ⇒ trivial); no full-corpus tensors |
| — | ~~No fused LIF kernel~~ | §2.3, §3.11 C1 | **Retracted** — jiterator + bundled NVRTC makes elementwise fusion available |
| — | ~~VRAM~~ | 3.0 GiB peak of 7.96 GiB | **Not a bottleneck at this scale** |
| — | ~~Compute budget~~ | 2–6 min/epoch | **Not a bottleneck**; we are data-rich |

---

## 6. Assumptions

Stated explicitly so they can be falsified. Each has a designated test.

| # | Assumption | Confidence | How it gets tested |
|---|---|---|---|
| A1 | The 14.5 µs/kernel constant holds for the real model, not just microbenchmarks | High | Phase 2: predict step time from kernel count, compare to measured |
| A2 | CUDA-graph capture survives a *complete* training loop (dataloading, LR schedule, grad clipping, eval) | **Medium** | Phase 2 gate — the highest-risk assumption in the plan |
| A3 | Reduced-precision GEMMs with an fp32 membrane do not degrade bpc | Medium | Phase 4 ablation vs full fp32; §3.11 C2 found no spike-count difference between bf16 and fp16 |
| A4 | Neuron-state memory alone (I5) can reach competitive bpc | Medium | The project's core research question |
| A5 | enwik8 downloads reproducibly and matches published checksums | High | Phase 2 step 1; both mirrors verified live |
| A6 | ~2–6 min/epoch permits ≥3 seeds per ablation arm | High | Arithmetic from §3.8/§3.10 |
| A7 | A same-param non-spiking RNN control is a fair comparison | Medium | Matched params, corpus, schedule, and seeds |
| A8 | A hand-written surrogate backward in a jiterator kernel can be made provably equivalent to the eager one | **Medium** | Phase 2 gate (**R10**); gradient equivalence test to fp32 tolerance |
| A9 | The ~4.9× fusion win survives in the full model, not just the microbenchmark | Medium | Phase 2: measure fused vs eager on the real training step |
| A10 | The 5 M-param text8 anchors (LSTM 1.50, TCN 1.45) are a fair external yardstick | Medium | They were grid-searched at matched size, so they are tuned baselines, not strawmen (§4.2) |

---

## 7. Risk matrix

| ID | Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|---|
| R1 | Full training loop proves un-capturable ⇒ lose 11× | Medium | **Critical** | Prototype capture *first* in Phase 2, before any modelling. Fallback: capture only the inner unroll, keep dataloading and eval outside the graph |
| R2 | Surrogate-gradient BPTT unstable at long unrolls | Medium | High | Grad clipping; surrogate-scale sweep; start L=128 and extend |
| R3 | Spiking model plateaus far above the ANN control | Medium | Medium | This is a *publishable finding*, not a failure — report the gap honestly |
| R4 | Dead neurons / firing-rate collapse or saturation | **High** | Medium | Log firing rate per layer from step 0; treat as a first-class metric, not a diagnostic afterthought |
| R5 | Graph capture silently reads stale data (static-buffer aliasing bug) | Medium | **Critical** | Assert-based test: graphed and eager paths must produce bit-identical losses for N steps |
| R6 | Long unattended runs killed by the harness | **High** (happened in v1) | High | Launch as detached OS processes with file-based logging; never as harness background tasks |
| R7 | Windows/WDDM instability or TDR under multi-hour load | Low | High | Checkpoint every N steps; resumable trainer from the start |
| R8 | bpc improvements are seed noise misread as signal | Medium | Medium | ≥3 seeds; report mean ± std; pre-register the effect-size threshold |
| R9 | Scope creep away from invariants I1–I5 | Medium | Medium | Invariants are frozen; any violation is a separate, clearly-labelled control arm |
| **R10** | **Hand-written jiterator backward is silently wrong** — gradients look plausible, model trains, result is subtly incorrect | **High** | **Critical** | `torch.autograd.Function` gradients checked against the eager path to fp32 tolerance in CI; keep `--no-fused` as the reference implementation and re-verify on every kernel change. This is the most dangerous class of bug in the plan because it produces *plausible* results |
| R11 | jiterator is a private/beta API (`torch.cuda.jiterator._create_*`) and may change | Medium | Medium | Isolate behind one module with an eager fallback; pin torch version in `requirements.txt` |
| R12 | Parallel decay form silently truncates context via fp32 underflow | Medium | High | Underflow analysis is a required gate before adopting any Toeplitz/scan form (§3.11 C3) |

---

## 8. Initial research strategy

### 8.1 Design principles derived from the measurements

1. **Count kernels, not FLOPs.** The architecture's inner loop should be written
   to minimise issued ops. This is a genuinely unusual optimisation target and it
   follows directly from §3.4.
2. **Fuse the neuron, then graph the step.** The two mechanisms compose
   multiplicatively — 4.9× from a jiterator-fused LIF, ~11× from CUDA-graph
   capture, **~51× together** (§3.11 C1). Both are *design constraints*, not
   later optimisations: static shapes, pre-allocated buffers, a capturable
   optimiser, no data-dependent control flow, and the neuron written as a single
   elementwise kernel from the start. Retrofitting either is how R1 becomes
   critical.
3. **Guard the fused path with the eager one.** Every fused kernel ships with an
   eager reference and an automated equivalence test on *both* forward and
   backward. Speed that silently corrupts gradients (R10) is worse than no speed.
4. **Treat the reset rule as the highest-stakes free parameter.** It is worth up
   to two orders of magnitude of throughput (§3.11 C3) *and* it is scientifically
   open (§4.5). Design the neuron so reset is swappable — hard, soft, detached,
   or absent — behind one flag, and make it the first ablation run.
5. **Spend the free axes.** Batch is cheap even after graphing; width is not
   (§3.10). Prefer large batch and moderate width, and pick both by measured
   bpc-per-GPU-hour rather than by intuition.
6. **Memory through dynamics, not weights.** Invariant I5 means long-range
   context must come from decay constants, adaptation, and multiple timescales.
   That is the scientific spine of the project, and §4.3's adaptation result is
   the evidence that it can work.
7. **Measure the control.** A same-parameter non-spiking RNN, trained identically,
   is what turns "1.x bpc" into a meaningful statement. The text8 5 M-parameter
   anchors (LSTM 1.50, TCN 1.45) are the external yardstick.
8. **Measure seed variance before believing any ablation.** §4.5 item 6: nobody
   has published it for this training regime, and every downstream conclusion
   depends on it.

### 8.2 Proposed phase plan

| Phase | Content | Est. GPU-hours |
|---|---|---:|
| **2 — Baseline reproduction** | enwik8/text8 acquisition with checksums; deterministic data pipeline; canonical stacked-LIF SNN; non-spiking RNN control at matched params; **CUDA-graph capture proven end-to-end (R1 gate)**; bit-identical graphed-vs-eager test (R5 gate); firing-rate instrumentation | ~8 |
| **3 — Candidate improvements** | 10–15 ranked hypotheses across neuron model, coding/T, reset rule, optimisation, sparsity, and kernel-count reduction; ROI matrix against the §5 bottlenecks | ~1 |
| **4 — Controlled experiments** | Pre-registered, single-variable, ≥3 seeds per arm; negative results retained | ~30 |
| **5 — Final training & docs** | Clean-seed final run; `scripts/evaluate.py`; `README.md`; `docs/math_spec.md` | ~12 |
| | **Total** | **~51** |

This fits inside the ratified ~55–60 GPU-hour budget with headroom for R1/R2
recovery.

### 8.3 Phase 2 entry checklist

Ordered so the two capability gates (R1, R10) are proven *before* any modelling
effort is spent on top of them.

**Data**
- [ ] `data/` acquisition script with SHA-256 verification against published values
- [ ] Deterministic, seeded, memory-mapped `uint8` corpus loader
- [ ] Both evaluation protocols implemented: fresh-state windowed *and* carried
      contiguous state (§4.3 — report both for every headline number)

**Capability gates — do these first**
- [ ] Eager reference LIF cell, kernel count documented and asserted in a test
- [ ] Jiterator-fused LIF in a `torch.autograd.Function`, with **forward *and*
      backward** checked against the eager reference (**R10** — the critical gate)
- [ ] `--no-fused` fallback path kept working and tested
- [ ] CUDA-graph training step proven on the real model (**R1**)
- [ ] Graphed vs eager bit-identical loss test over N steps (**R5**)

**Science**
- [ ] Non-spiking RNN control at matched parameter count, identical everything else
- [ ] Per-layer firing-rate logging from step 0 (**R4**)
- [ ] **Seed-variance measurement: 3–5 identical-config runs**, to establish the
      noise floor before any ablation is believed (**R8**, §4.5 item 6)
- [ ] Reset rule swappable behind one flag (hard / soft / detached / none)

**Operations**
- [ ] Detached-process launcher with file logging and resumable checkpoints (**R6**)

---

## 9. Budget model

From §3.8 and §3.10, using enwik8's 90 MB (9.0 × 10⁷ character) training split:

| Configuration | Params | Throughput | Time / epoch |
|---|---:|---:|---:|
| batch 128, d 512, L 256 | 0.73 M | 2 328 Mtok/h | **2.3 min** |
| batch 256, d 768, L 512 | 1.49 M | 2 104 Mtok/h | 2.6 min |
| batch 128, d 1024, L 256 | 2.52 M | 1 178 Mtok/h | 4.6 min |
| batch 64, d 1536, L 128 | 5.35 M | 427 Mtok/h | 12.6 min |

A 30-epoch run of a ~2.5 M-parameter model costs ~2.3 GPU-hours; of a ~5 M-parameter
model, ~6 GPU-hours (improvable by raising batch, which is the cheap axis).

These figures are **graph-captured but not yet fused**. Jiterator fusion (§3.11 C1)
is measured at 4.0–4.9× on the neuron path in isolation; how much of that survives
once GEMMs and the optimiser are included is assumption **A9**, to be measured in
Phase 2 rather than assumed here. The budget above deliberately claims none of it,
so any fusion win is headroom rather than a dependency.

**We are data-limited, not compute-limited.** The 30 GPU-hours allocated to Phase 4
buy roughly 40–80 ablation runs at 3 seeds each — enough for statistically
defensible conclusions rather than single-run anecdotes.

---

## 10. Reproducing this audit

```bash
python scripts/audit/01_gpu_capability.py          docs/reports/data/audit_01_gpu_capability.json
python scripts/audit/02_snn_cost_model.py          docs/reports/data/audit_02_snn_cost_model.json
python scripts/audit/03_kernel_count.py            docs/reports/data/audit_03_kernel_count.json
python scripts/audit/04_graphed_scaling.py         docs/reports/data/audit_04_graphed_scaling.json
python scripts/audit/05_verify_literature_claims.py docs/reports/data/audit_05_verify_claims.json
```

Each script prints its results and writes machine-readable JSON. Absolute timings
are hardware- and driver-specific; the *relationships* are the reproducible claims:
flat-in-size launch cost, linear-in-kernel-count wall-clock, ~11× graph speed-up,
~4.9× jiterator fusion, and the exactness of the Toeplitz form.

**Benchmarking standard.** Every figure in §3 was collected with warm-up
iterations discarded, explicit `torch.cuda.synchronize()` around timed regions,
`torch.manual_seed(0)`, and the hardware/software stack of §2.1–2.2. Peak VRAM is
`torch.cuda.max_memory_allocated()`. No figure in this report is estimated,
extrapolated, or recalled — each is measured, and the raw JSON is committed.

---

## 11. Exit criteria

- [x] System audit executed against the live machine (§2)
- [x] Toolchain limits verified by execution, not inference (§2.3, §3.7)
- [x] Performance envelope measured across five benchmarks (§3)
- [x] Cost model derived and validated against observation (§3.4)
- [x] Literature and domain context synthesised, adversarially fact-checked (§4,
      full survey in `01a_literature_review.md`)
- [x] Literature's systems claims independently re-measured; one overturned this
      report's own conclusion, and the reversal is documented rather than quietly
      patched (§3.11)
- [x] Bottlenecks ranked, assumptions stated, risks matrixed (§5–§7)
- [x] Research strategy and budget model set out (§8–§9)
- [ ] **One open decision for Elliot: the §4.6 boundary question on invariant I5**
      (does "no lateral recurrent weight matrix" admit a per-channel *temporal*
      kernel or a complex/rotational membrane state?). It should be answered before
      implementation, not after seeing which answer scores better.
- [ ] **Committed and reviewed by Elliot — Phase 2 does not begin until this is signed off**

---

## 12. Changelog

| Rev | Change |
|---|---|
| 1 | Initial audit: four benchmarks, cost model, strategy. Concluded custom kernels were impossible and fusion unavailable. |
| **2** | **Literature review contradicted that conclusion; re-measurement confirmed the contradiction.** `torch.cuda.jiterator` + torch-bundled NVRTC provides elementwise CUDA fusion with no `nvcc`/MSVC/Triton (§3.11 C1). Added the exact Toeplitz parallel form and its underflow bound (C3), corrected the bf16 recommendation (C2), added risks R10–R12, bottlenecks B3/B5/B7/B8, assumptions A8–A10, and §4. Rev-1 claims are retained struck-through rather than deleted. |
