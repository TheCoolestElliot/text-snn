# Performance notes

**Outside the Phase 1–5 protocol.** Nothing here is pre-registered and nothing here is a
reported figure. The rule this work holds itself to is stricter than a tolerance: a change
lands only if it is **bit-identical** — same logits, same gradients, same trained weights,
same generated text — so that no number in any report moves. Anything that changes a bit is
catalogued in §4 and left for the author to rule.

Measured 2026-09-21 on the project's one machine (RTX 5060 8 GB, sm_120, WDDM, driver
616.92; torch 2.12.1+cu130; Python 3.14.7) at commit `4446951`, through the real `Trainer`
and `evaluate` code paths. The evaluation legs reproduced the committed `e25_*`
`final_test.json` bpc bit-for-bit, which is what makes the timings trustworthy.

## 1. Where the time goes

| config (B=128, L=256, K=2)  | graphed ms/step | GPU-busy ms | host gap ms | peak GiB |
|-----------------------------|-----------------|-------------|-------------|----------|
| `snn` d=512 (735K)          | 18.20           | 17.69       | 0.91        | 0.61     |
| `twocomp_detach` d=512      | 23.74           | 23.12       | 0.50        | 0.98     |
| `snn` d=1481 (5.0M)         | 105.6           | 104.4       | 1.23        | 1.64     |
| `twocomp_detach` d=1481     | 123.6           | 123.3       | 1.12        | 2.61     |

**The CUDA-graphed step is 95–99 % GPU-busy.** Reconnaissance's launch-bound finding was
true of the eager path and the graph capture solved it; what is left is kernel work.

Kernel-level shares of one graphed step, from `torch.profiler` traces:

| share of GPU time            | snn d512 | detach d512 | snn d1481 | detach d1481 |
|------------------------------|----------|-------------|-----------|--------------|
| layer GEMMs (fwd/dgrad/wgrad)| 47.3 %   | 34.6 %      | 76.3 %    | 64.2 %       |
| head GEMMs                   | 11.7 %   | 8.6 %       | 6.0 %     | 5.4 %        |
| scan kernels (512 launches each way) | 18.1 % | 16.8 % | 6.5 %     | 8.2 %        |
| `torch.stack` copies         | 11.6 %   | 17.3 %      | 6.2 %     | 9.7 %        |
| two-compartment gradient tail| —        | 11.8 %      | —         | 6.6 %        |
| **everything scan-related**  | **29.7 %** | **48.5 %** | **12.6 %** | **26.2 %**  |
| per-layer firing-rate mean   | 2.0 %    | 1.4 %       | 0.9 %     | 0.8 %        |

The adopted two-compartment neuron pays 2.2–2.4× the plain LIF's scan cost, and only about
a third of that is the fused kernels: the rest is `torch.stack` of 256 per-timestep tensors
(already at memory bandwidth — no cheaper copy exists, only not copying) and a post-loop
tail that shifts a full `[B,L,d]` tensor by one step to form two parameter gradients.

Chat (shipped `chat-v3d-aligned`, d=1024, K=4): a reply takes **0.60 s** median at the REPL
default pool of 256, of which the decode loop is 70 % and **host-bound — the GPU is idle
62 % of a turn** — and the anti-LM null pass 22 %. Chat training is 174.9 ms/step, 98 %
GPU-busy: GEMM 53 %, stack/copy 16 %, elementwise 14 %, scan kernels 10.5 %.

Evaluation of every kind is ~4.3 % of `EXP_025`'s GPU time.

## 2. Start-up: the driver's compute cache was full

Warm-up took 1.5 s in some processes and 14.8–18.2 s in others, and the chat REPL's
launch-to-first-reply read 22.9 s, 10.0 s and 4.3 s on three consecutive launches. It is not
NVRTC (0.02 s per kernel) and not graph capture (0.07–0.24 s). The driver finalises each
CUDA module the first time a kernel in it is touched, CPU-bound at ~6–10 MB/s, and keeps the
result in `%APPDATA%\NVIDIA\ComputeCache`, capped at 1 GiB by default. That cache stood at
1073.58 of 1073.74 MB — 218 MiB of it orphans of four earlier driver builds — so it evicts
by age and the penalty rotates between modules from launch to launch. A 100 MB cap
reproduces a permanent 18 s; an uncapped private cache converges at 428 MB for the chat op
set after ~5 launches (no process finalises more than ~150 MB), after which warm-up is 1.5 s.

`src/snn/config.py` now sets `CUDA_CACHE_MAXSIZE` to the documented 4 GiB maximum with
`setdefault`. Cached code is byte-identical (35/35 entries sha256-equal across independent
producers). It is worth ~0 % of a training ladder, which repeats one workload, and 13–17 s
off every cold process — the REPL, the test suite, every benchmark. A driver update
invalidates the whole cache; expect a few slow launches after one.

## 3. What was examined and is already tight

The captured region (zero_grad through AdamW, one launch); AdamW foreach+capturable; the fp64
clip (≤ 0.6 %); the lr write; the sampler (0.13 ms); logging and checkpoint cadence; eval
cadence under the drivers' settings (0.6–2.2 % of wall); the determinism flags (none selects
a slower kernel on this path); the horizon probe (a one-pass curve is **not** equivalent to
the per-k probe — under truncation k, position p is scored with p mod k characters of
context); GPU clocks (P1 in 241/241 busy samples, never thermally limited; power-capped at
145 W with max == default, so no software lever); import cost (torch is 1.59 of 1.70 s).
A second "lean" captured graph skipping the firing-rate means measured at ≤ 2 % and was
dropped. A time-first `[L,B,d]` layout would permute GEMM rows and every reduction order.

## 4. Catalogued, not adopted: these change bits

| change | measured | why it is not bit-identical |
|---|---|---|
| Layer-0 GEMM as a 205-row table `(E @ W0ᵀ + b)[idx]` | layer-0 fwd+bwd 4.86 → 0.74 ms (d=512), 43.2 → 1.74 ms (d=1481): ~1.3× / ~1.65× on the step | same function (`EXP_020` proved the redundancy) but a different GEMM shape: forward differs at 3.9e-06 / 1.3e-05 max-abs |
| TF32 GEMMs | 18.2 → 14.7 ms (1.24×), 105.6 → 68.4 ms (1.54×) | reduced-precision matmul |
| bf16 layer GEMMs | 14.7 ms / 74.0 ms | reduced precision |
| in-kernel reduction of `grad_w`, `grad_beta_s` | est. 4–9 % of the two-compartment step | a different summation order |
| null pass over the surviving tier only (chat) | 0.15 → 0.012–0.066 s | `logp_null` moves 1e-5..3e-5 with batch shape |
| larger eval batch for `fresh` | ~2.3× at d=512 | reduction order; `carried`'s batch size is part of the protocol |

Decision #7's fp64 clip is the precedent for what adopting any of these costs: a run trained
after it cannot be compared bitwise against a figure that predates it.

## 5. Chat decode: the GPU was waiting on the host

Measured 2026-10-08 on the shipped `chat-v6-scratch/ckpt_best.pt`, REPL defaults (n = 256,
`max_new` 400), six fixed prompts, seed 1234: **0.483 → 0.373 s per reply (1.29×)**, and the
six replies are **byte-identical** before and after. Two changes, neither touching arithmetic:

* **The loop breaker stopped re-deriving dead rows.** The pool loop runs until the *last* of
  256 rows stops, ~350 characters, against a much shorter median reply, and every character
  re-ran `loop_penalty_target` on every row. A stopped row's history never grows, so its target
  is a constant; `_penalty_pairs` now caches it (cleared on any steering permutation). 542K →
  294K calls for the six replies.
* **The forward runs while the host works.** The decode step is captured as two graphs, forward
  then sampler, because only the sampler reads the loop-break penalty. `sample_candidates`
  queues the forward as soon as `feed` exists on the device and reads `nxt` back through a
  pinned non-blocking copy and an event, so the host's per-row bookkeeping overlaps the
  forward instead of following it. The split materialises the intermediate `last`, which
  changes no kernel and no input.

Held by `test_graph_and_eager_draw_the_same_text` and the rest of `tests/test_snnchat.py`
(123 pass on the GPU). What is left per character is roughly equal parts the alive rows'
loop-breaker checks, the eager `multinomial`, and graph replay. A numpy-vectorised loop
detector was tried and dropped: at n = 256 and a 48-slot window it was no faster on ordinary
text and 2.6× *slower* on a looping pool, which is the case the breaker exists for.
