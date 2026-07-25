# Tier 6 — research experiments

Tier 6 is the *research* tier: measured experiments (with GPU training), reported
honestly including **negative** results. Everything below has now been **run**.

## Results (measured)

All bpc figures are the **deterministic** held-out val bpc (a fixed full-split sweep
under a fixed encoder RNG — exactly reproducible and seed-invariant), on tiny-shakespeare
(`input.txt`, `--val-split 0.1`, `--seed 1337`).

### SNN ablation sweep — hidden 512, 2000-update budget (matches the README sweep)

| config | val bpc | vs. baseline | verdict |
|--------|:---:|:---:|---|
| baseline (2 layers, rate coding) | 2.823 | — | reference |
| per-neuron learnable `beta` | 2.907 | +0.084 | ✗ hurts (as the scalar version did) |
| 3 layers | 3.048 | +0.225 | ✗ hurts (confirms the original sweep) |
| **3 layers + LayerNorm on currents** | **2.779** | **−0.044** | ✓ LayerNorm **rescues depth** (3.048 → 2.779) |
| **graded (deterministic) input coding** | **2.731** | **−0.092** | ✓ best single ablation |
| **graded + 3 layers + LayerNorm** | **2.650** | **−0.173** | ✓✓ the winners **stack** |

Takeaways: (1) LayerNorm makes depth *help* instead of hurt — the "extra depth hurts"
finding was really "extra depth is hard to train through without normalization." (2)
Deterministic graded coding beats the stochastic rate encoder here (the input is a known
one-hot, so the encoder's noise was costing more than it added). (3) A per-neuron learnable
time constant still doesn't help at this budget. (4) The two winners combine for the best
result. These belong in the README's "How it was tuned" narrative; the shipped checkpoints
still use none of the knobs.

### Matched GRU baseline — hidden 1024, 16k-update budget (the SNN's 2.44 recipe)

| model | params | val bpc |
|-------|:---:|:---:|
| SNN (`shakespeare.pt`, hidden 1024, 2 layers) | 1.18 M | 2.445 |
| **GRU baseline** (`--arch gru`, hidden 1024, 2 layers) | 12.73 M | **2.388** |

The GRU is only **0.06 bpc** better while carrying **~11× the parameters** — its dense
recurrent matrices cost what the SNN gets for free from the parameter-less leak. So the
honest "vs ANN" story on this corpus is *the ANN's edge is marginal and the SNN is
strikingly parameter-efficient*, not the cross-corpus "~1.4 bpc" figure the README used to
cite. (Note: on Windows the GRU run prints `[done]` and saves its checkpoint, then exits
non-zero during CUDA teardown — a shutdown artifact, not a training failure; the checkpoint
re-evaluates to 2.388.)

### CUDA-graph inner-loop speedup (forward core, RNG-free, fixed shape)

| config | eager | cuda-graph | speedup |
|--------|:---:|:---:|:---:|
| hidden 512, L 64, B 128 | 229.7 ms | 18.6 ms | **12.3×** |
| hidden 512, L 32, B 128 | — | — | **12.4×** |
| hidden 1024, L 64, B 128 | — | — | **7.9×** |

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

# Matched GRU baseline (the SNN's headline recipe)
python snn_char_lm.py train --arch gru --data input.txt --steps 16000 --hidden 1024 \
    --seq-len 160 --dropout 0.2 --weight-decay 2e-4 --val-split 0.1 --seed 1337 \
    --deterministic-eval --ckpt runs/gru.pt

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
  budget to see whether it moves the shipped 2.44.
- Multi-seed (`--seed 1/2/3`) mean±sd on the headline comparisons.
- Integrate the CUDA graph into the *training* step (with autograd via
  `make_graphed_callables`) rather than the forward-only benchmark.
