# Phase 2 — Frozen interface specification

**Status:** Frozen 2026-08-01, before implementation. Companion to
`02_baseline_report.md`. Every signature below is normative: implementations must
match it exactly so that independently-written modules compose without drift.

Derived from `01_reconnaissance.md` §8.1 (design principles), §8.3 (entry
checklist), §3.11 (measured capability), and the §4.6 I5 ruling.

---

## 0. The one architectural idea that shapes everything

The Phase-1 cost model is `time ≈ 14.5 µs × kernels-per-timestep × L`, so kernel
**count** is the performance metric (§3.4). The naive spiking-LM inner loop —

```python
for t in range(L):                  # WRONG for this machine
    h = emb[:, t]
    for layer in layers:
        h, mem = layer(h, mem)      # a GEMM *inside* the time loop
```

— issues `K·L` GEMMs plus `~13·K·L` elementwise kernels. That is what the Phase-1
audit benchmarked, and it is not how this model should be built.

Because I5 forbids lateral recurrence **and** there is no cross-layer feedback,
layer *k*'s input current at every timestep depends only on layer *k−1*'s output
at that timestep. So the layers can be evaluated **sequentially in depth**, each
one computing its entire input current for all `L` timesteps in a *single* GEMM
before its time loop begins:

```python
h = embed(x)                                  # [B, L, d]   1 kernel
for k in range(K):
    cur   = linear_k(h)                       # [B, L, d]   1 GEMM, whole sequence
    h, v  = lif_scan(cur, v0)                 # [B, L, d]   L fused kernels
logits = head(h)                              # [B, L, V]   1 GEMM
```

Kernel count per forward pass drops from `K·L` GEMMs + `~13·K·L` elementwise to
**`K+2` GEMMs + `K·L` fused elementwise**. The time loop contains exactly one
kernel per step per layer, which is the floor for a hard-threshold neuron with
reset.

This is not an optimisation to add later; it is the architecture. It is also the
first place where invariant I5 *pays* for itself computationally rather than
costing, and that observation belongs in the Phase-2 report.

**Correctness note.** Depth-sequential evaluation is exact, not an approximation:
information flows strictly forward in depth, and causality in time is preserved
inside each layer's own scan. Nothing at layer *k*, time *t* depends on layer
*k+1* at any time, nor on layer *k−1* at any time *> t*.

---

## 1. Repository layout

```
src/snn/
    __init__.py
    config.py        Config dataclass, CLI parsing, seeding, determinism
    data.py          acquisition, verification, memmap corpus, samplers
    surrogate.py     surrogate value/derivative, eager + CUDA source
    kernels.py       jiterator probe, compiled kernel cache, availability
    neuron.py        eager reference scan + FusedLIFScan autograd.Function
    model.py         SpikingCharLM + AnalogueCharLM + GRUCharLM
    metrics.py       bpc, firing rates, kernel counting
    train.py         Trainer: static buffers, CUDA-graph capture, checkpoints
    evaluate.py      both evaluation protocols
scripts/
    data/download_corpus.py
    train.py
    evaluate.py
    launch.py                detached-process launcher (R6)
    bench/kernel_count.py    measured kernel counts for the report
tests/
    test_data.py
    test_surrogate.py
    test_neuron_equivalence.py    <- R10 gate (critical)
    test_kernel_count.py
    test_graph_equivalence.py     <- R5 gate
    test_model.py
    test_determinism.py
experiments/logs/
docs/reports/
```

All source is importable as `snn.*` with `src/` on `sys.path` (a `conftest.py` at
repo root inserts it; scripts do the same).

---

## 2. Tensor conventions — normative

| Symbol | Meaning |
|---|---|
| `B` | batch size |
| `L` | sequence length (characters per window) |
| `d` | model width |
| `K` | number of spiking layers |
| `V` | vocabulary size |
| `T` | micro-steps per character (baseline `T = 1`) |

* **Layout is `[B, L, d]`, batch-first, contiguous.** No `[L, B, d]` anywhere.
  The time loop indexes `cur[:, t]`, which is a strided view — acceptable, because
  the fused kernel is elementwise and reads it directly.
* **Token tensors are `int64`, shape `[B, L]`.** Corpus storage on disk is
  `uint8`; the cast happens in the sampler.
* **Membrane state and the threshold comparison are always `fp32`**, regardless
  of `cfg.dtype` (§3.11 C2, bottleneck B8). `cfg.dtype` selects the GEMM dtype
  only, and the current is cast to fp32 before entering any scan.
* **Spikes are `fp32` valued `{0.0, 1.0}`**, not bool — they feed a GEMM directly.
* Firing threshold comparison is **`>=`** (greater-or-equal), matching the
  Phase-1 verified kernel. Eager and fused paths must agree on this exactly.

---

## 3. `snn/config.py`

```python
@dataclass(frozen=False)
class Config:
    # --- data ------------------------------------------------------------
    corpus: str = "enwik8"          # "enwik8" | "text8"
    data_dir: str = "data"
    seq_len: int = 256              # L
    batch_size: int = 128           # B

    # --- model -----------------------------------------------------------
    d_model: int = 512
    n_layers: int = 2               # K
    vocab_size: int = 0             # filled from the corpus; 0 means "unset"
    arch: str = "snn"               # "snn" | "analogue" | "gru"

    # --- neuron (SpikeGPT reference values, 01_reconnaissance.md §4.1) ----
    beta: float = 0.5               # membrane decay
    threshold: float = 1.0
    reset: str = "hard"             # "hard" | "soft" | "detached" | "none"
    surrogate: str = "atan"
    surrogate_alpha: float = 2.0
    t_steps: int = 1                # T

    # --- optimisation ----------------------------------------------------
    lr: float = 3e-3
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    warmup_steps: int = 200
    max_steps: int = 20_000
    lr_schedule: str = "cosine"     # "cosine" | "constant"

    # --- systems ---------------------------------------------------------
    fused: bool = True              # jiterator LIF; False = eager reference
    cuda_graph: bool = True
    dtype: str = "fp32"             # GEMM dtype: "fp32" | "bf16" | "fp16"
    device: str = "cuda"
    seed: int = 0
    deterministic: bool = True

    # --- run bookkeeping -------------------------------------------------
    run_name: str = ""
    out_dir: str = "experiments/runs"
    eval_every: int = 500
    eval_max_windows: int = 0       # 0 = full split
    ckpt_every: int = 1000
    log_every: int = 50
```

Required helpers:

```python
def seed_everything(seed: int, deterministic: bool) -> None: ...
def config_from_args(argv: list[str] | None = None) -> Config: ...   # argparse, one flag per field
def config_hash(cfg: Config) -> str: ...   # sha256 of the sorted JSON, first 12 hex chars
```

`seed_everything` must set `random`, `numpy`, `torch`, `torch.cuda`, and when
`deterministic` is true also `torch.use_deterministic_algorithms(True,
warn_only=True)`, `torch.backends.cudnn.deterministic = True`,
`torch.backends.cudnn.benchmark = False`, and
`os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"` (must be set before the first
CUDA context — document this in the module docstring and set it at import time of
`snn.config`).

---

## 4. `snn/data.py`

### 4.1 Acquisition

```python
CORPORA = {
    "enwik8": CorpusSpec(
        urls=["http://mattmahoney.net/dc/enwik8.zip",
              "https://data.deepai.org/enwik8.zip"],
        zip_sha256="...",      # verified on first download, then pinned in-repo
        member="enwik8",
        raw_bytes=100_000_000,
        split=(90_000_000, 5_000_000, 5_000_000),
    ),
    "text8": CorpusSpec(
        urls=["http://mattmahoney.net/dc/text8.zip"],
        zip_sha256="...",
        member="text8",
        raw_bytes=100_000_000,
        split=(90_000_000, 5_000_000, 5_000_000),
    ),
}

def ensure_corpus(name: str, data_dir: str) -> CorpusPaths: ...
```

* Downloads to `data/<name>.zip`, verifies SHA-256, extracts, verifies the
  extracted byte count, writes `data/<name>/{train,val,test}.bin` as raw `uint8`
  and `data/<name>/meta.json`.
* `meta.json` records: vocab (sorted list of byte values present in the **whole**
  100 MB file — this is the standard enwik8/text8 protocol and is what published
  bpc numbers use), `vocab_size`, split sizes, the zip SHA-256, the extracted-file
  SHA-256, and the download URL that succeeded.
* Published expectations, asserted: enwik8 `vocab_size == 205`, text8
  `vocab_size == 27`. A mismatch is a hard failure, not a warning.
* **Idempotent**: re-running with the files present verifies checksums and exits
  without re-downloading.

### 4.2 Corpus access

```python
class Corpus:
    def __init__(self, name: str, data_dir: str): ...
    vocab_size: int
    stoi: np.ndarray   # uint8 -> contiguous id, shape [256], 255 = unused
    itos: np.ndarray   # id -> uint8
    def split(self, which: str) -> np.memmap: ...   # "train"|"val"|"test", uint8, ids already remapped
```

Splits are stored **already remapped to contiguous ids** so no per-batch lookup is
needed. Storage is `np.memmap(..., dtype=np.uint8, mode="r")`.

### 4.3 Samplers

```python
class RandomWindowSampler:
    """Training sampler. Deterministic given (seed, step)."""
    def __init__(self, data: np.memmap, batch_size: int, seq_len: int, seed: int): ...
    def batch(self, step: int) -> tuple[Tensor, Tensor]:
        """Returns (inputs [B,L] int64, targets [B,L] int64) on CPU, pinned.

        Offsets are drawn from a torch.Generator seeded with (seed, step), so
        batch(k) is reproducible in isolation and resuming at step k is exact.
        Targets are inputs shifted by one character.
        """

def windowed_eval_batches(data, batch_size, seq_len) -> Iterator[tuple[Tensor, Tensor]]:
    """Protocol A - fresh state. Non-overlapping windows in order, state zeroed
    at every window start. The trailing partial batch is dropped and the number
    of characters actually scored is returned by the caller's accounting."""

def contiguous_eval_batches(data, batch_size, seq_len) -> Iterator[tuple[Tensor, Tensor]]:
    """Protocol B - carried state. The split is cut into `batch_size` contiguous
    streams; window i of every stream is yielded together, so membrane state can
    be carried (detached) across successive yields. Every character is scored
    exactly once."""
```

Both evaluation protocols are mandatory for every headline number (§4.3 of the
Phase-1 report). Protocol A matches the training distribution; protocol B is the
number comparable with published enwik8 results.

---

## 5. `snn/surrogate.py`

Only the arctangent surrogate is required for Phase 2 (§4.3 item 3: keep
`atan(α=2)`, lock the threshold at 1.0, sweep width rather than shape).

Definitions, with `x = v_pre − threshold` and `α = surrogate_alpha`:

```
value(x)      = (1/π)·arctan(π/2·α·x) + 1/2          # only used to carry gradient
derivative(x) = (α/2) / (1 + (π/2·α·x)²)             # d value / d x, exactly
```

`derivative(0) = α/2 = 1.0` at the default `α = 2`.

```python
def atan_value(x: Tensor, alpha: float) -> Tensor: ...
def atan_grad(x: Tensor, alpha: float) -> Tensor: ...
ATAN_CUDA_GRAD: str   # CUDA C++ expression text for use inside a jiterator kernel
```

The eager spike uses the standard straight-through construction so that its
autograd derivative is *exactly* `atan_grad`:

```python
s_hard = (x >= 0).to(x.dtype)
sv     = atan_value(x, alpha)
s      = s_hard.detach() + sv - sv.detach()
```

The fused backward must reproduce `atan_grad` to fp32 tolerance. This equivalence
is the R10 gate.

---

## 6. `snn/kernels.py`

```python
def jiterator_available() -> bool:
    """True iff torch.cuda.jiterator._create_multi_output_jit_fn imports, CUDA is
    available, and a trivial kernel compiles and runs. Result is cached."""

def lif_forward_kernel(reset: str): ...
def lif_backward_kernel(reset: str): ...
```

One kernel is compiled **per reset mode** and cached in a module-level dict, so
the reset rule is a compile-time constant rather than a runtime branch. Compilation
is lazy (first use) and costs ~0.17 s (§3.11 C1).

### 6.1 Forward kernel contract

Signature in CUDA C++, elementwise over `[B, d]`:

```
inputs : v_prev, cur
scalars: beta, thr
outputs: v_next, spike, v_pre        (num_outputs = 3)
```

```
v_pre = v_prev * beta + cur
spike = (v_pre >= thr) ? 1 : 0
v_next = reset == "soft"     ? v_pre - thr * spike
       : reset == "hard"     ? v_pre * (1 - spike)
       : reset == "detached" ? v_pre * (1 - spike)      // identical forward
       :                       v_pre                    // "none"
```

`v_pre` is emitted because the backward needs it and it is **not** recoverable
from `v_next` under hard reset. `spike` is emitted separately because it feeds the
next layer's GEMM.

### 6.2 Backward kernel contract

```
inputs : grad_spike, grad_v_next, v_pre
scalars: beta, thr, alpha
outputs: grad_cur, grad_v_prev       (num_outputs = 2)
```

With `x = v_pre − thr`, `s = (x >= 0)`, `sg = atan_grad(x, alpha)`:

| reset | `dv_next/dv_pre` |
|---|---|
| `soft` | `1 − thr·sg` |
| `hard` | `(1 − s) − v_pre·sg` |
| `detached` | `(1 − s)` |
| `none` | `1` |

```
g_vpre     = grad_v_next * (dv_next/dv_pre) + grad_spike * sg
grad_cur   = g_vpre
grad_v_prev = beta * g_vpre
```

`beta` is a fixed scalar hyperparameter in Phase 2, so no `grad_beta` is produced.
Learnable per-channel decay is a Phase-3 candidate (admitted by the §4.6 ruling)
and will need one more output; the kernel signature has room (limit is 8).

**Every kernel input and output is fp32.** The caller casts.

---

## 7. `snn/neuron.py`

```python
def lif_scan_eager(cur: Tensor, v0: Tensor, beta: float, thr: float,
                   alpha: float, reset: str) -> tuple[Tensor, Tensor]:
    """Reference implementation. cur [B,L,d] fp32, v0 [B,d] fp32.
    Returns (spikes [B,L,d] fp32, v_final [B,d] fp32).
    Built from ordinary autograd ops - this is the ground truth for R10."""

class FusedLIFScan(torch.autograd.Function):
    @staticmethod
    def forward(ctx, cur, v0, beta, thr, alpha, reset_code): ...
    @staticmethod
    def backward(ctx, grad_spikes, grad_v_final): ...

def lif_scan(cur, v0, beta, thr, alpha, reset, fused: bool): ...
```

* `lif_scan` dispatches on `fused and jiterator_available() and cur.is_cuda`, and
  falls back to `lif_scan_eager` otherwise — this is the `--no-fused` path
  required by the entry checklist, and it must stay tested.
* `FusedLIFScan.forward` runs under `torch.no_grad()`, loops `t in range(L)`
  issuing **one** fused kernel per step, collects `spike_t` and `v_pre_t` in
  Python lists, and `torch.stack(..., dim=1)` once at the end. Stacking L tensors
  is a handful of kernels total, not one per step.
* `ctx.save_for_backward(v_pre)` only. `spike` and `s` are recomputed inside the
  backward kernel from `v_pre`, which halves the saved-activation memory.
* `backward` loops `t` in reverse issuing one fused backward kernel per step,
  accumulating `grad_v` from step `t+1`, and stacks `grad_cur`.
* `reset_code` is an int (`0=hard, 1=soft, 2=detached, 3=none`) so the Function's
  non-tensor arguments stay hashable and graph-capture-safe.

Both paths must accept `v0` requiring grad (needed if carried-state TBPTT is
enabled later); Phase 2 passes detached zeros.

---

## 8. `snn/model.py`

```python
class SpikingCharLM(nn.Module):
    """embedding -> [Linear -> LIF] x K -> Linear head.

    Invariants: I1 binary spikes between layers, I2 leaky integration,
    I3 hard threshold, I4 surrogate BPTT, I5 no lateral recurrent matrix.
    Embedding and head are fp32 and non-spiking (Phase-1 report §4.3 item 2).
    """
    def forward(self, idx: Tensor, state: list[Tensor] | None = None
                ) -> tuple[Tensor, list[Tensor], dict]:
        """idx [B,L] int64 -> (logits [B,L,V], new_state, aux)

        aux["firing_rate"] is a list of K scalar tensors: the mean spike rate of
        each layer for this batch (R4, logged from step 0).
        """
```

Layer `k` is `nn.Linear(d, d, bias=True)` applied to the **whole sequence at
once**, followed by `lif_scan`. Layer 0's input is the embedding (analogue direct
coding, `T = 1`); layers 1..K−1 receive binary spikes.

### Controls, at matched parameter count

```python
class AnalogueCharLM(nn.Module):
    """Identical to SpikingCharLM in every respect except that the neuron emits
    `atan_value(v_pre - thr, alpha)` continuously instead of a binary spike, and
    the reset uses that continuous value. Isolates I1+I3 (binarity and the hard
    threshold) as the single variable. Parameter count is *identical*, not merely
    matched."""

class GRUCharLM(nn.Module):
    """External anchor: nn.GRU at matched parameter count. Deliberately violates
    I5 and is labelled as such in every table it appears in. Exists so that the
    text8 5M-param literature anchors (LSTM 1.50, TCN 1.45) can be connected to
    something trained under our exact protocol."""

def build_model(cfg: Config) -> nn.Module: ...
def count_params(model) -> int: ...
```

`GRUCharLM`'s width is chosen at construction so its parameter count is within
±2 % of the `snn` arm's; the chosen width and the exact counts go in the report.

---

## 9. `snn/metrics.py`

```python
def bits_per_char(total_nats: float, n_chars: int) -> float:   # nats/char / ln 2
def count_cuda_kernels(fn: Callable, warmup: int = 3) -> dict:
    """torch.profiler with CUDA activity; returns {'total': n, 'by_name': {...}}.
    Used by tests/test_kernel_count.py to *assert* the cost model, and by
    scripts/bench/kernel_count.py to produce the report's table."""
```

---

## 10. `snn/train.py`

```python
class Trainer:
    def __init__(self, cfg: Config): ...
    def step(self) -> dict: ...          # one optimisation step, returns metrics
    def train(self) -> None: ...
    def save_checkpoint(self, path) -> None: ...
    def load_checkpoint(self, path) -> None: ...
```

Requirements, all from §8.3:

1. **Static buffers.** `self.static_x`, `self.static_y` are pre-allocated
   `[B, L]` int64 CUDA tensors. Every step does
   `static_x.copy_(cpu_batch, non_blocking=True)` — the batch data changes, the
   addresses never do.
2. **Capturable optimiser.** `torch.optim.AdamW(..., capturable=True, foreach=True)`.
   The LR schedule is applied by writing into the optimiser's LR **tensor** so the
   captured graph sees the update; a Python-float LR would be baked in at capture
   time. This is a real trap and must have a test.
3. **Capture recipe** (proven in `scripts/audit/04_graphed_scaling.py`): run 3
   warm-up steps on a side stream, `wait_stream`, then capture one full
   `zero_grad → forward → backward → clip → step` into a `torch.cuda.CUDAGraph`.
   Thereafter a step is `copy_ inputs; graph.replay()`.
4. **Gradient clipping inside the graph** must be the capture-safe form —
   `torch.nn.utils.clip_grad_norm_` with `foreach=True` and no `.item()`, no
   Python-side conditional. Any `.item()`, `print`, or host sync inside the
   captured region is a capture failure; treat it as such.
5. **`cfg.cuda_graph = False` runs the identical code path eagerly.** The R5 gate
   compares the two.
6. **Firing rates from step 0** (R4), per layer, logged every `log_every` steps.
7. **Resumable checkpoints** (R6): model, optimiser, step, config, RNG states
   (`torch`, `torch.cuda`, `numpy`, `random`). Resume must reproduce the
   loss trajectory of an uninterrupted run — that is a test, not a hope.
8. **Logging**: one JSONL record per logged step to
   `experiments/runs/<run_name>/log.jsonl`, plus the resolved `config.json`, plus
   the §"Scientific Benchmarking Standard" block (GPU model, VRAM, CPU, torch and
   CUDA versions, seed, config hash) written once at run start.

`scripts/launch.py` starts a run **detached** (`subprocess.Popen` with
`CREATE_NEW_PROCESS_GROUP | DETACHED_PROCESS` on Windows), redirects stdout/stderr
to `experiments/runs/<run_name>/stdout.log`, writes the PID, and returns
immediately.

---

## 11. `snn/evaluate.py`

```python
@torch.no_grad()
def evaluate(model, corpus, split: str, cfg: Config, protocol: str,
             max_windows: int = 0) -> dict:
    """protocol in {"fresh", "carried"}.
    Returns {"bpc": float, "nats": float, "n_chars": int, "protocol": str,
             "firing_rate": [...], "n_windows": int}."""
```

* `"fresh"` zeroes membrane state at every window (protocol A).
* `"carried"` carries detached state across the contiguous streams (protocol B).
* Both are deterministic and independent of batch size in the characters they
  score. A test asserts `bpc` is invariant to `batch_size` for the `"fresh"`
  protocol.

---

## 12. Test matrix — the gates

| Test | Gate | Assertion |
|---|---|---|
| `test_neuron_equivalence.py::test_forward` | **R10** | Fused vs eager spikes **bit-identical**; membrane within 1e-5 |
| `test_neuron_equivalence.py::test_backward` | **R10** | `grad_cur` and `grad_v0` within `rtol=1e-4, atol=1e-6` of the eager path, for **all four reset modes**, and for `gradcheck`-style perturbations |
| `test_neuron_equivalence.py::test_surrogate_matches_analytic` | R10 | Eager autograd derivative equals `atan_grad` to 1e-6 |
| `test_kernel_count.py` | A1 | Fused scan issues **exactly one** CUDA kernel per timestep per layer; the eager scan issues ≥6. Documented, asserted |
| `test_graph_equivalence.py` | **R5** | Graphed and eager training produce identical losses over 20 steps to `atol=0` where achievable, else `rtol=1e-6`; the LR-schedule-inside-graph trap has its own case |
| `test_determinism.py` | — | Two runs with the same seed produce identical loss trajectories; different seeds do not |
| `test_data.py` | A5 | Checksums verified; `vocab_size` 205/27; splits sum to 100 MB; sampler reproducible from `(seed, step)`; targets are inputs shifted by one |
| `test_model.py` | — | Parameter counts match the reported figures; `analogue` control has *identical* param count; spikes are strictly in `{0,1}`; no lateral recurrent matrix exists in the `snn` arm (I5, asserted structurally) |
| `test_surrogate.py` | — | `derivative(0) == alpha/2`; value/derivative consistency via `torch.autograd.grad` |

Tests that need CUDA are marked `@pytest.mark.cuda` and skipped cleanly on a
CPU-only machine. The suite must pass with **both** `--fused` and `--no-fused`.

---

## 13. Baseline configuration to be reported

The Phase-2 headline arm, fixed here so it cannot drift:

| Field | Value | Source |
|---|---|---|
| corpus | enwik8 | §1.4 |
| `d_model` / `n_layers` | 512 / 2 | §9 budget row 1 |
| `batch_size` / `seq_len` | 128 / 256 | §9 — 2 328 Mtok/h, 2.3 min/epoch |
| `beta` / `threshold` / `reset` | 0.5 / 1.0 / hard | SpikeGPT reference, §4.1 |
| surrogate | `atan`, α = 2 | §4.3 item 3 |
| `T` | 1 | §4.3 item 1 |
| dtype | fp32 | membrane fp32 is mandatory; GEMM dtype is a Phase-4 ablation |
| expected params | ~0.735 M | 205·512 + 2·(512²+512) + 512·205 + 205 |

Seeds `{0, 1, 2, 3, 4}` for the noise-floor measurement (R8).

**Declared in advance:** `beta = 0.5` gives a membrane half-life under one
character, so the baseline's memory horizon is expected to be very short and its
bpc correspondingly weak. That is the honest reference point, not a bug. A
documented `beta ∈ {0.5, 0.9, 0.95}` check is run as *baseline establishment*
(confirming the arm is not accidentally crippled by a single hyperparameter) and
is reported as such — it is **not** an ablation result, and the headline number is
the reference configuration.
