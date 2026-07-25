# Tier 6 — research experiments

Tier 6 is the *research* tier: measured experiments that require GPU training and may
return **negative** results (reported honestly either way). This file records what
infrastructure has landed and gives **exact, ready-to-run commands** for the GPU work.

## Landed (no GPU needed — already in `snn_char_lm.py`, CPU-tested)

- **Deterministic eval** (`--deterministic` on `eval`; `--deterministic-eval` on `train`).
  Sweeps a *fixed* set of contiguous, non-overlapping windows tiling the split under a
  *fixed* encoder RNG, so the reported bpc is a single true full-split number, exactly
  reproducible and **seed-invariant** — removing the ~±0.01 bpc estimator noise the
  earlier analysis flagged. Use it for all comparisons below so they are apples-to-apples.
- **Performance finding (measured, negative):** neither **TF32**
  (`set_float32_matmul_precision('high')`) nor **AMP/autocast** is a throughput lever here.
  The workload is launch-bound (a long chain of tiny sequential kernels), so faster matmuls
  buy ~nothing while perturbing float32 numerics — so we deliberately leave precision at the
  default (see the note in `main()`). The real lever is the CUDA-graph capture below.
- **Ablation knobs** (all off by default; the shipped model uses none):
  `--beta-per-neuron`, `--layernorm`, `--input-coding {rate,graded}`.

## Methodology

Match the README's tuning sweep: a fixed **2000-update** budget per config, `hidden 512`,
`2 layers`, `--seq-len 160`, `--val-split 0.1`, `--seed 1337`, compared by the
**deterministic** held-out val bpc. Single seed (as the README sweep was) — treat gaps
smaller than ~0.05 bpc as noise, or repeat with `--seed 1/2/3` for mean±sd. Put outputs in
`runs/` (git-ignored).

```bash
mkdir -p runs
COMMON="--data input.txt --steps 2000 --hidden 512 --seq-len 160 --val-split 0.1 \
        --seed 1337 --deterministic-eval"

# 0. Baseline (reference; README sweep reported ~2.82 stochastic val bpc @ 2k)
python snn_char_lm.py train $COMMON --ckpt runs/base.pt      --log-csv runs/base.csv

# A. Per-neuron learnable beta (heterogeneous time constants; a known temporal-SNN trick).
#    The README's "learnable beta hurts" was a single shared scalar — this is the fairer test.
python snn_char_lm.py train $COMMON --beta-per-neuron --ckpt runs/beta.pt --log-csv runs/beta.csv

# B. Can LayerNorm rescue depth? (README: depth-3 hurt, 3.04 @ 2k.) Run depth-3 with/without.
python snn_char_lm.py train $COMMON --layers 3              --ckpt runs/d3.pt   --log-csv runs/d3.csv
python snn_char_lm.py train $COMMON --layers 3 --layernorm  --ckpt runs/d3ln.pt --log-csv runs/d3ln.csv

# C. Graded (deterministic) input coding vs the stochastic rate encoder.
python snn_char_lm.py train $COMMON --input-coding graded   --ckpt runs/graded.pt --log-csv runs/graded.csv

# Score any checkpoint reproducibly (seed-invariant full-split sweep):
python snn_char_lm.py eval --ckpt runs/beta.pt --data input.txt --val-split 0.1 --split val --deterministic
```

**How to read it:** a config *wins* only if its deterministic val bpc beats the baseline by
more than run-to-run seed noise. Negative results are the expected, honest outcome for
several of these and should be written into the README's "How it was tuned" section.

## Still to build (needs the GPU to implement *and* verify)

These were intentionally **not** landed, because their value is the trained/benchmarked
number and shipping unvalidated GPU code would be dishonest. Build + run when the GPU is free.

### 1. Matched in-harness GRU baseline
The README compares against "classical LSTM/GRU ~1.4 bpc" from *other* corpora. Add a
non-spiking `GRUCharLM` selectable via `--arch gru`, trained in this exact harness so the
"above the best ANNs" claim rests on a same-corpus, same-budget control.

*Spec:* a `CharLMBase(nn.Module)` holding the shared `generate()`; `GRUCharLM` implements
`init_state` (GRU hidden `(num_layers,B,H)`), `forward_seq`, `step`, `forward` with the same
signatures as `SNNCharLM` (embedding → `nn.GRU` → linear readout). Add `arch: str = "snn"`
to the config and a `build_model(cfg)` factory used by `train()` and `_load_checkpoint()`.
Then: `python snn_char_lm.py train --arch gru --data input.txt --steps 16000 --hidden 1024
--seq-len 160 --dropout 0.2 --val-split 0.1 --seed 1337 --deterministic-eval --ckpt runs/gru.pt`
and report its deterministic val bpc next to the SNN's 2.44.

### 2. CUDA-graph inner-loop speedup
Training is launch-bound (a long chain of tiny sequential spiking kernels). `torch.compile`
is ruled out here — Triton is unavailable on Windows / Python 3.14. The Triton-free lever is
a **CUDA-graph capture** of the fixed-shape, RNG-clean inner loop (`_char_rate` at fixed
`(B, hidden, T)`); an earlier isolated benchmark measured **~7.75× on the forward inner
loop**. Implement behind an opt-in flag with a CPU/non-CUDA fallback, capture only the
static-shape region (encoder draw stays outside the graph), and benchmark end-to-end
updates/s honestly before claiming the speedup in the README.
