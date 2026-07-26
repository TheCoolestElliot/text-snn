# Tier 6 — research experiments

Tier 6 is the *research* tier: measured experiments (with GPU training), reported
honestly including **negative** results. Everything below has now been **run**.

## Results (measured)

All bpc figures are the **deterministic** held-out val bpc (a fixed full-split sweep
under a fixed encoder RNG — exactly reproducible and seed-invariant), on the shipped
prose corpus (`input.txt`, `--val-split 0.1`, `--seed 1337`).

### SNN ablation sweep — hidden 512, 2000-update budget (matches the README sweep)

| config | val bpc | vs. baseline | verdict |
|--------|:---:|:---:|---|
| baseline (2 layers, rate coding) | 2.715 | — | reference |
| per-neuron learnable `beta` | 2.805 | +0.090 | ✗ hurts (as the scalar version did) |
| 3 layers | 2.886 | +0.171 | ✗ hurts (confirms the original sweep) |
| **3 layers + LayerNorm on currents** | **2.635** | **−0.080** | ✓ LayerNorm **rescues depth** (2.886 → 2.635) |
| **graded (deterministic) input coding** | **2.615** | **−0.100** | ✓ best single ablation |
| **graded + 3 layers + LayerNorm** | **2.484** | **−0.231** | ✓✓ the winners **stack** |

Takeaways: (1) LayerNorm makes depth *help* instead of hurt — the "extra depth hurts"
finding was really "extra depth is hard to train through without normalization." (2)
Deterministic graded coding beats the stochastic rate encoder here (the input is a known
one-hot, so the encoder's noise was costing more than it added). (3) A per-neuron learnable
time constant still doesn't help at this budget. (4) The two winners combine for the best
result. These belong in the README's "How it was tuned" narrative; the shipped checkpoints
still use none of the knobs.

### Matched GRU baseline — hidden 1024, 16k-update budget (the SNN's headline recipe)

| model | params | val bpc |
|-------|:---:|:---:|
| SNN (`prose.pt`, hidden 1024, 2 layers) | 1.21 M | 2.310 |
| **GRU baseline** (`--arch gru`, hidden 1024, 2 layers) | 12.75 M | **2.187** |

The GRU is only **0.12 bpc** better while carrying **~10.5× the parameters** — its dense
recurrent matrices cost what the SNN gets for free from the parameter-less leak. So the
honest "vs ANN" story on this corpus is *the ANN's edge is marginal and the SNN is
strikingly parameter-efficient*, not the cross-corpus "~1.4 bpc" figure often quoted for
LSTM/GRU character models. (Note: on Windows the GRU run has been seen to print `[done]` and
save its checkpoint, then exit non-zero during CUDA teardown — a shutdown artifact, not a
training failure; the checkpoint re-evaluates to 2.187 either way.)

### Width / depth tuning sweep — 2000-update budget

| config | params | val bpc |
|--------|:---:|:---:|
| baseline (hidden 512, 2 layers) | 0.34 M | 2.715 |
| hidden 768 | 0.71 M | 2.591 |
| hidden 1024 (the shipped width) | 1.21 M | 2.523 |
| hidden 2048 | 4.51 M | **2.412** |
| hidden 512, `--tbptt-chunk 64` | 0.34 M | 2.697 |
| hidden 512, `--learn-beta` (scalar) | 0.34 M | 2.926 ✗ |

Width still helps at 2048 on this budget; the shipped model uses 1024 because 2048 costs
3.7× the parameters for 0.11 bpc, and parameter efficiency is the point of the comparison
against the GRU. A longer TBPTT chunk is a wash (−0.018), and a learnable scalar membrane
time constant hurts (+0.211), matching the per-neuron result above.

### CUDA-graph inner-loop speedup (forward core, RNG-free, fixed shape)

| config | eager | cuda-graph | speedup |
|--------|:---:|:---:|:---:|
| hidden 512, L 64, B 128 | 241.5 ms | 18.7 ms | **12.9×** |
| hidden 512, L 32, B 128 | 122.3 ms | 9.5 ms | **12.9×** |
| hidden 1024, L 64, B 128 | 240.3 ms | 29.7 ms | **8.1×** |

Capturing the fixed-shape, RNG-free inner loop (`_run_core`) in a CUDA graph and replaying
it as a single launch removes the per-kernel launch overhead that dominates this
launch-bound workload. The speedup is larger where the net is *more* launch-bound (smaller
hidden). `torch.compile` is unavailable here (no Triton on Windows / Python 3.14), so the
CUDA graph is the Triton-free lever. **TF32/AMP are not levers** (faster matmuls don't help
a launch-bound net) — so default float32 precision is left untouched.

## Reproduce

```bash
mkdir -p runs
COMMON="--data input.txt --steps 2000 --hidden 512 --seq-len 160 --val-split 0.1 \
        --seed 1337 --deterministic-eval"
python snn_char_lm.py train $COMMON                                   --ckpt runs/base.pt
python snn_char_lm.py train $COMMON --beta-per-neuron                 --ckpt runs/beta.pt
python snn_char_lm.py train $COMMON --layers 3                        --ckpt runs/d3.pt
python snn_char_lm.py train $COMMON --layers 3 --layernorm            --ckpt runs/d3ln.pt
python snn_char_lm.py train $COMMON --input-coding graded             --ckpt runs/graded.pt
python snn_char_lm.py train $COMMON --layers 3 --layernorm --input-coding graded --ckpt runs/combo.pt

# The README's width/depth tuning table (same 2000-update budget, --hidden varies)
TUNE="--data input.txt --steps 2000 --seq-len 160 --val-split 0.1 --seed 1337 --deterministic-eval"
python snn_char_lm.py train $TUNE --hidden 768              --ckpt runs/w768.pt
python snn_char_lm.py train $TUNE --hidden 1024             --ckpt runs/w1024.pt
python snn_char_lm.py train $TUNE --hidden 2048             --ckpt runs/w2048.pt
python snn_char_lm.py train $TUNE --hidden 512 --tbptt-chunk 64 --ckpt runs/chunk64.pt
python snn_char_lm.py train $TUNE --hidden 512 --learn-beta --ckpt runs/lbeta.pt

# Matched GRU baseline (the SNN's headline recipe)
python snn_char_lm.py train --arch gru --data input.txt --steps 16000 --hidden 1024 \
    --seq-len 160 --dropout 0.2 --weight-decay 2e-4 --val-split 0.1 --seed 1337 \
    --deterministic-eval --ckpt runs/gru.pt

# (input.txt itself is rebuildable: python build_corpus.py)

# Score any checkpoint reproducibly (seed-invariant full-split sweep):
python snn_char_lm.py eval --ckpt runs/combo.pt --data input.txt --val-split 0.1 \
    --split val --deterministic

# CUDA-graph benchmark:
python snn_char_lm.py bench --hidden 512 --seq-len 64 --batch-size 128
```

## Landed in `snn_char_lm.py`

- **Deterministic eval** — `eval --deterministic` / `train --deterministic-eval`.
- **Ablation knobs** — `--beta-per-neuron`, `--layernorm`, `--input-coding {rate,graded}`
  (all off by default).
- **GRU baseline** — `--arch gru` (a `GRUCharLM` sharing the training/eval/sampling harness
  via a `CharLMBase` + `build_model` factory).
- **CUDA-graph benchmark** — the `bench` subcommand (captures `_run_core`, forward-only).

## Possible follow-ups (not run)

- Train the best ablation combo (graded + LayerNorm-depth) at the full 16k/hidden-1024
  budget to see whether it moves the shipped 2.310.
- Settle hidden 2048. At the 2000-update budget width had still not stopped helping
  (hidden 2048 reaches 2.412 vs hidden 1024's 2.523), so the shipped hidden-1024 choice
  rests on parameter efficiency (4.51 M vs 1.21 M), not on 2048 being worse. Training
  2048 at the full 16k budget would show whether the extra capacity overfits once it has
  the updates to do so.
- Multi-seed (`--seed 1/2/3`) mean±sd on the headline comparisons.
- Integrate the CUDA graph into the *training* step (with autograd via
  `make_graphed_callables`) rather than the forward-only benchmark.
