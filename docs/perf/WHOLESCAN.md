# Whole-scan kernels

**Outside the Phase 1–5 protocol.** Nothing here is pre-registered, nothing here is a
reported figure, and **this work changes no reported number**: the standard it was held to is
not a tolerance but bit-identity — same spikes, same gradients, same trained weights and
AdamW state, same evaluation bpc, to the last bit. Every number below traces to a committed
artifact under `docs/perf/data/` (not `docs/reports/data/`, which is the protocol's).

Landed on branch `perf/optimize`; measured 2026-09-21 on the project's one machine (RTX 5060
8 GB, sm_120, torch 2.12.1+cu130, Python 3.14.7).

## 1. What it is

`snn/kernels.py` took the LIF timestep from ~13 launches to one, and graph capture then
removed the launch overhead. What was left (`README.md` §1) is kernel work, and everything
scan-related was 29.7 % of the plain LIF's graphed step at d=512 and **48.5 %** of the
adopted two-compartment arm's — only about a third of it the fused kernels, the rest
`torch.stack` of 256 per-timestep tensors and L launches each way whose state round-trips
through device memory between every pair of them.

All three recurrences are strictly per-(batch, channel). So `src/snn/wholescan.py` runs the
scan as **one NVRTC kernel per layer**: one thread per (batch, channel), looping over `t`
with the membrane — or, going backward, the adjoint — held in a register, reading and
writing the `[B, L, d]` tensors in place. No stack, no per-step launch, no round-trip.

**No arithmetic lives in that module.** The kernel source is the project's committed functor
text — `kernels.forward_source()` / `backward_source()`, `twocomp_forward_source()` /
`twocomp_backward_source()`, `twocomp_detach_backward_source()`, imported and pasted — plus a
time-loop wrapper that does indexing and nothing else. There is no second derivation of any
neuron to drift from the first.

It is reached through the three existing entry points — `neuron.lif_scan`,
`twocomp.twocomp_scan`, `twocomp_detach.twocomp_detach_scan` — so **no call site changed** and
every fused arm gets it (`snn`, `tokenshift`, `threshold`, `noise`, `dopamine`, `interface`,
`localdopamine`, `twocomp`, `twocomp_threshold`, `twocomp_detach`; asserted by
`test_every_fused_arm_reaches_the_whole_scan_kernels`), as does the chat model. The
per-timestep jiterator Functions are **untouched** and stay in the tree as the bitwise oracle
and as the fallback. Under `no_grad` the new path skips the tensors only a backward reads,
which is the inference path `lif_scan` and `twocomp_detach_scan` never had.

No `Config` field was added, so the K1 gate is untouched for every reference run.

## 2. Why it is bit-identical, and the one way it was not

Same compiler (the NVRTC inside the torch wheel), same flags as jiterator
(`--std=c++20 --gpu-architecture=sm_<cc> -default-device -DNDEBUG`, no `--fmad` either way, so
contraction is on by default in both), same functor text.

**Forward.** Every add that consumes a product is written `__fadd_rn(__fmul_rn(…))`, so NVRTC
has no FMA to form in either context, and the carried membrane enters the next step only
through `__fmul_rn`. Nothing to get wrong.

**Backward.** The functors are natural C++ and NVRTC *does* contract them — two or three FMAs
per step — and *which multiply it pairs with which add is decided per expression DAG*. In the
per-timestep path the carried adjoint (`beta * g`, `beta_s_c * gs`) is materialised to memory
between launches, so the next step sees an opaque load. Inlined into a time loop it is an SSA
value the optimiser can see through, and wherever the next step **adds** it directly — LIF
`reset="none"`, where `dv == 1` folds `grad_v_next * dv` to `grad_v_next`; both
two-compartment backwards, where `gs = grad_vs_next + w_c * gv` — NVRTC re-pairs the FMA onto
the carry multiply, across the timestep boundary. The prototype measured the naive kernel at
0/12 configurations bit-equal for those three variants (~37–62 % of `grad_cur` elements off by
one ulp, `grad_beta_s` by up to 3e-05), while hard, soft and detached reset were 12/12: there
the carry is only ever *multiplied*, so there is nothing to re-pair — and a first prototype
that tried hard reset alone got lucky.

A one-ulp gradient is invisible to every tolerance in this repository and changes every
trained weight. The fix is a **contraction barrier**: each backward step is called through an
`__attribute__((noinline))` trampoline, so the step arithmetic exists exactly once, as a closed
function whose leaves are its float arguments — the same expression DAG, FMA sites and operand
pairing included, that jiterator compiles per element. It is applied to **all six** backwards,
including the three that do not need it today: it costs nothing measurable, and "this variant
happens not to need it" is a fact about one NVRTC build.

**The two per-channel gradients** are reductions over batch and time. The kernel emits the
per-element *products* (`gv·vs_t`, `gvs·vs_{t−1}`: one IEEE multiply each, stored, so no add
can consume it) into contiguous `[B, L, d]` tensors, and the **same** `torch`
`.sum(dim=(0, 1))` the per-timestep path calls reduces them — same reduce kernel, same launch
configuration, same bits. An in-kernel reduction would be faster and would *not* be
bit-identical; it stays in `README.md` §4.

**Contraction is a heuristic, not a contract.** So availability is not only "does it compile
and launch": each variant is **self-checked on first use** — a tiny scan through both paths,
every output and gradient compared as bit patterns — and a variant that fails degrades to the
per-timestep path with a warning instead of silently changing gradients. The check draws its
inputs from a private CPU generator (touching the global RNG would change the noise arm and
the chat sampler with no kernel involved), and it syncs, so it never runs inside a capture:
`Trainer` verifies all six variants while writing its header, every capture site warms up
eagerly first, and a capture that does come first records the per-timestep path and says so.

NVRTC is driven through `ctypes` and its public C API, because
`torch.cuda._compile_kernel` raises on this machine without `CUDA_HOME`. No toolkit is needed,
a real or bogus `CUDA_HOME` changes nothing, and nothing process-wide is set. `_CudaKernel`
takes raw `data_ptr()`s and ignores strides, so every tensor crossing the launch boundary is
checked (`_require`), and anything the raw kernel cannot be trusted with — a strided `cur`,
another device, a shape past the 32-bit arguments or the launch grid — goes to the
per-timestep path, which reads strides properly. A "no" costs throughput, never correctness.

## 3. Turning it off, and what a run records

```
SNN_SCAN_IMPL=jiterator   # the per-timestep path, exactly as before
SNN_SCAN_IMPL=wholescan   # the default
```

Read once per process; a value that is set and not recognised **raises** rather than falling
back. `Trainer` writes `scan_impl` into `environment.json` (and the chat trainer into its
`run_start` record): the requested implementation and each variant's self-check verdict. A run
used the whole-scan kernels iff `requested == "wholescan"` and its arm's variant is `true`.

## 4. The gates, and the proof they can fail

| gate | what it holds |
|---|---|
| `tests/test_wholescan_equivalence.py` (132 tests, ~60 s) | **bit patterns** (int32 views: −0.0 ≠ +0.0), old path vs new, through the model-facing dispatchers: spikes, final state, the saved intermediates `v_pre` / `vs_seq`, and every gradient × six variants × {(3,17,5), (128,256,512), (128,256,1481)} × three firing regimes with non-zero initial state × {grad, no-grad} × {eager, CUDA-graph replay}; L=1 / B=1 / d=1 edges; the fallback rules; the selector; a capture that comes first; no `CUDA_HOME` and a bogus one, in fresh processes; the global RNG left alone; every fused arm reaching the kernels; and a real `Trainer`, 40 steps × {snn hard, snn none, twocomp, twocomp_detach} × {graphed, eager}, parameters and AdamW state bit-equal |
| the three R10 gates | every CUDA test now runs under **both** implementations (`conftest.py`, `scan_impl`), so each path is held to the eager reference, the spec transcription, the float64 leg and graph capture on its own — and the per-timestep Functions, which campaign 08 mutates, are not left guarded by nothing |
| `tests/test_kernel_count.py` | pinned to the per-timestep path it makes claims about; two new tests assert the whole-scan forward is **1** launch and forward+backward a constant, independent of L |

The equivalence gate asserts *which autograd node was built*, so a path that silently fell
back cannot pass by comparing the oracle with itself; and it fills the caching allocator's
free blocks with NaN first, so an output the kernel failed to write cannot be rescued by the
oracle's freed buffer.

**Negative control, in the test file itself.**
`test_without_the_barrier_it_is_not_bit_identical` builds the barrier-less backward of
`lif_none`, `twocomp` and `twocomp_detach` and asserts it is **not** bit-identical to the
oracle — while `allclose` at the R10 tolerances accepts it, which is the point. If a future
NVRTC stops re-pairing, that test fails; the response is a new negative control, not a deleted
assertion.

**Mutation campaign** — `scripts/audit/12_wholescan_mutations.py` →
`docs/perf/data/audit_12_wholescan_mutations.json`, reusing campaign 08's guards and restore by
import (08 itself is evidence and is not edited):

**87/87 caught** (block W 32/32, block F 55/55), nothing escaped, every guarded source restored clean, on a clean tree at `b18a6d5`. The campaign's first run selected block F with `-k wholescan` and fired 86/87; the escape (F-M23, eager `atan_grad` losing its 1/2) was the selector's — it deselected the unparametrised CPU test that is M23's only guard in campaign 08's own record. That run is kept unedited as `audit_12_wholescan_mutations_run1_k_wholescan.json`; the rule is now `-k "not jiterator"`, and the re-run above is under it.

## 5. Measured

`scripts/bench/wholescan_ab.py` → `docs/perf/data/wholescan_ab.json`. Real `Trainer` built
from each committed run's own `config.json`, CUDA-graphed; the two implementations
**interleaved in one process** (A B, B A, A B), because clock and power state move a whole
run by ~6 % on this box while windows inside a run are tight. Medians of three legs; ranges in
the JSON. The script fails unless both assertions hold, and both held for every run: the two
implementations reached **SHA-256-equal parameters and AdamW state**, and the evaluation
reproduced the run's committed `final_test.json` bpc and nats **exactly** (`==`), both
protocols, under both implementations.

| run | train ms/step | train | eval ms/batch, fresh | eval ms/batch, carried | peak GiB, train | peak GiB, eval |
|---|---|---|---|---|---|---|
| `e25_snn_d512_s0` | 19.932 → 15.386 | **1.295×** | 18.14 → 6.81 (2.664×) | 19.1 → 6.52 (2.929×) | 0.828 → 0.722 | 0.686 → 0.498 |
| `e25_detach_d512_s0` | 26.094 → 18.478 | **1.412×** | 21.9 → 6.81 (3.216×) | 23.56 → 7.14 (3.3×) | 1.574 → 1.346 | 1.187 → 0.875 |
| `e25_detach_d1481_s0` | 125.029 → 109.308 | **1.144×** | 49.31 → 39.12 (1.26×) | 49.73 → 39.0 (1.275×) | 3.583 → 2.979 | 2.535 → 1.63 |

Per-timestep (`jiterator`) → whole-scan. 3 interleaved rounds of 240 steps per implementation per run, at `8f1f6c1`, NVIDIA GeForce RTX 5060, 617.42, torch 2.12.1+cu130. All three runs: weights and AdamW state SHA-256-equal across the two paths, and both protocols' test bpc `==` the committed `final_test.json`. `failures: []`.

## 6. Hazards — what would break it

* **A torch upgrade.** A new NVRTC may contract differently, inside the trampoline as well as
  around it. The self-check degrades the affected variant to the per-timestep path with a
  warning; the equivalence gate and its negative control say which way it moved. The private
  surface is `torch.cuda._utils._cuda_load_module` / `_CudaKernel` (risk R11 again, behind
  the same hard pin).
* **A new arm whose backward adds the carried adjoint.** That is exactly the re-pairing case.
  The barrier covers it *only if the new backward goes through a trampoline* — do not write a
  whole-scan backward that calls its functor directly, and add the arm to `VARIANTS` so it is
  self-checked and negatively controlled.
* **An in-kernel reduction**, or any reordering of the two `.sum(dim=(0, 1))` calls, changes
  the summation order and therefore `grad_w` / `grad_beta_s`.
* **Changing the compile flags**, or compiling with a toolkit NVRTC instead of the wheel's.
  `--fmad=false` alone makes every backward differ (campaign mutation W27).
* **Strides.** A raw-pointer kernel mis-reads a strided tensor silently. `_require` refuses
  and the dispatchers route around; a new launcher must do both.
* `torch.utils.deterministic.fill_uninitialized_memory` adds a fill per `torch.empty` output.
  Bit-neutral, and the project already turns it off.
* WDDM's TDR only matters at shapes far past this project's (the largest kernel here is
  single-digit milliseconds).

## 7. What merging this changes

* **The default scan path.** After the merge every fused CUDA run uses the whole-scan kernels
  unless `SNN_SCAN_IMPL=jiterator` is set. Results do not move; step time, peak memory and
  start-up do.
* **Which evidence scripts would measure a different thing if re-run.** Evidence scripts were
  not edited. Those that go through the dispatchers and report *kernel counts or timings*:
  `scripts/bench/kernel_count.py` (its "fused scan issues one kernel per timestep" row would
  read ~1 per *scan*), `scripts/exp/014_calibrate_cost.py` (prices configurations from real
  training steps), and every ladder driver's recorded wall-clock. Those that go through the
  dispatchers but report *numbers the kernels compute* — `scripts/audit/07_init_pathology.py`,
  `scripts/exp/011_chase_compose_divergence.py`, `scripts/exp/012_chase_fold_gap.py`, the
  `NNN_run_*` drivers' bpc — would reproduce bit for bit. Unaffected entirely, because they
  never reach the fused dispatch: `scripts/audit/03_kernel_count.py` (no `snn` import),
  `scripts/audit/06_neuron_cross_check.py`, `scripts/exp/016_reset_jacobian.py`,
  `scripts/exp/017_reset_jacobian_bound.py` (eager, `cfg.fused = False`), and
  `scripts/exp/002_candidate_neuron_cost.py` (builds its own jiterator kernels). The one-line
  way to get the old measurement from any of them is `SNN_SCAN_IMPL=jiterator`.
* **Committed GPU-hour cost constants go stale.** `exp_014_cost_calibration.json`,
  `exp_015_cost_calibration.json` and the §7.1 estimates descend from step times measured on
  the per-timestep path. They over-state the cost of a two-compartment run by roughly the
  training speed-ups in §5; they are not corrected here, because a budget is the author's.
* **Whether the default should be `wholescan` or opt-in is the author's ruling.** The branch
  defaults to `wholescan`, so *the merge itself is the decision*; flipping `SCAN_IMPLS[0]` in
  `src/snn/wholescan.py` makes it opt-in, and `SNN_SCAN_IMPL=jiterator` restores the old path
  exactly either way.
* A run trained after the merge **can** be compared bitwise against one trained before it —
  unlike decision #7's fp64 clip, this is numerically inert, and
  `test_a_real_trainer_learns_the_same_weights` is what holds that.
