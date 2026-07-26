# Character-level Spiking Neural Network Language Model

A character-level language model whose hidden layers are **recurrent
leaky-integrate-and-fire (LIF) spiking neurons**, built with
[PyTorch](https://pytorch.org) + [snnTorch](https://snntorch.readthedocs.io)
and sized to train on an 8 GB GPU. It reads characters as **rate-coded spike
trains**, processes them with **recurrent LIF layers**, and is trained end-to-end
with **surrogate-gradient backpropagation through time**.

Everything lives in one heavily-commented file, [`snn_char_lm.py`](snn_char_lm.py).
Search it for the tag `# DESIGN:` to jump between the explanations of each SNN
design decision.

---

## Quick start

```bash
pip install -r requirements.txt        # see the file re: CUDA build of torch

# 1. Prove it works end-to-end in ~1-2 min (trains a tiny model, then samples;
#    it is launch-bound, so the wall-clock is hardware-dependent):
python snn_char_lm.py smoke

# 2. Train on the built-in demo corpus (no data file needed):
python snn_char_lm.py train --steps 2000

# 3. Train on the shipped prose corpus (~1.1 MB of public-domain English) with a
#    held-out validation split; the best-validation checkpoint is kept. This
#    quick run uses defaults (hidden 512, 8000 updates) so it is only a warm-up
#    -- it will NOT reach the shipped model's score (see "Results" for the full
#    recipe). It writes a scratch checkpoint so it can't overwrite the
#    pre-trained prose.pt shipped in this repo:
python snn_char_lm.py train --data input.txt --steps 8000 --seq-len 160 \
    --val-split 0.1 --seed 1337 --ckpt my_prose.pt

# 4. Generate from a checkpoint (the shipped model is ready to use):
python snn_char_lm.py sample --ckpt prose.pt --prompt "It was " --length 400
```

---

## How it works

### The two time axes

A spiking LM juggles two notions of "time", and keeping them separate is the key
to reading the code:

| axis | length | role |
|------|--------|------|
| **sequence** | `seq_len` characters | the recurrent state carries across it — this is the *language memory* |
| **SNN micro-steps** | `num_steps` (`T`) per character | each character is shown as a `T`-step spike train; the network integrates it and we read out the mean firing rate |

One training window therefore unrolls `seq_len × T` spiking updates.

### 1. Rate-coded input

Each character is one-hot encoded (one input neuron per vocabulary symbol) and
converted to spikes by `snntorch.spikegen.rate`: over `T` micro-steps the active
neuron fires with probability `rate_gain` (default 0.9) and the rest stay silent.
Averaging the network's response over these noisy steps is a cheap form of
ensembling. (With `rate_gain < 1` the code is stochastic at every `T`: even at
`T=1` the active neuron fires only with probability `rate_gain`, so the input is
occasionally all-zero. Larger `T` averages more of these 0/1 samples into a
finer, less noisy rate. Only `rate_gain = 1.0` makes `T=1` a deterministic
one-hot.)

### 2. Recurrent LIF layers

The hidden layers are `snn.Leaky` leaky integrate-and-fire neurons whose
**membrane potential is carried across the character sequence**. Each step:

```
U_t ← beta · U_{t-1} + W_in · input_t      # leaky integration (recurrent in time)
S_t  = 1 if U_t > threshold else 0         # fire (Heaviside)
U_t ← U_t − threshold · S_t                # soft reset (subtract)
```

Because `U_t` depends on `U_{t-1}`, carrying `U` across the sequence makes each
layer a genuine recurrent unit — the spiking analogue of a leaky RNN. `beta`
(default 0.9) is the membrane leak: high `beta` remembers longer, low `beta`
reacts to the latest input. That decaying membrane trace **is** the network's
memory.

> **Why not an explicit lateral recurrent matrix (`snn.RLeaky`)?** The obvious
> choice for "recurrent LIF" is snnTorch's `RLeaky`, which adds a learned
> all-to-all spike-feedback matrix `W_rec · spikes_prev`. I built and measured
> it: on this next-character task a learned lateral matrix turns the network into
> a chaotic dynamical system — it **fails to learn** (collapses to predicting
> character frequencies, no better than a unigram model) and its
> backprop-through-time gradient **explodes past float32 to NaN**. The same
> failure appears whether the feedback fires every micro-step or once per
> character. The leak's self-recurrence (spectral radius `beta < 1`) is
> contractive, so it is stable, and with depth it is expressive enough to model
> long-range structure (the model reaches **< 0.2 bpc** on the demo corpus,
> exploiting context far beyond bigrams). This is a real, measured design
> decision — see the `# DESIGN: recurrence` comment in the code.

### 3. Surrogate-gradient training

The spike `S = Heaviside(U − threshold)` has a zero-almost-everywhere derivative,
so ordinary backprop would kill all gradients. snnTorch substitutes a smooth
surrogate (default **ATan**) for `dS/dU` on the backward pass only — the forward
pass still emits hard 0/1 spikes. That makes the whole network differentiable and
trainable with Adam + backprop-through-time.

### 4. Rate-decoded readout

The top layer's **mean firing rate** over the `T` micro-steps (a smooth vector in
`[0, 1]`) is decoded by a single `nn.Linear` into next-character logits. Rates
give the classifier head a clean gradient — friendlier than stacking another
threshold neuron at the output.

---

## Training it: truncated BPTT

Because the recurrence is the contractive leaky membrane (not an explosive
lateral matrix), training is stable with ordinary gradient clipping — no NaN
gymnastics required. The one training technique worth calling out:

- **Truncated BPTT (TBPTT)** — the membrane state is carried across the whole
  window, but the gradient graph is **detached every `tbptt-chunk` characters**
  (default 32), so each backward pass only unrolls `chunk × T` spiking steps
  instead of `seq_len × T`. That bounds both memory and the length of the
  gradient path, letting `--seq-len` be long without growing either. The model
  still integrates context across the full window; only gradient credit
  assignment is truncated. This is the standard way to train recurrent nets on
  long sequences.

Gradient clipping (`--grad-clip 1.0`) is kept as routine insurance for any
recurrent net; measured gradient norms stay of order 1–10.

### Honest evaluation

The last `--val-split` fraction of the corpus (default 10%) is held out as a
contiguous validation tail, disjoint from training. Every `--eval-every` updates
the model is scored on validation windows (fresh membrane, no gradient — random
windows by default, or a fixed full-split sweep under `--deterministic-eval`)
and the **best-validation checkpoint is kept** — so a checkpoint reflects
generalization, not the last step's memorization. Bits-per-character (bpc) is the
reported metric; lower is better, and it is directly comparable across corpora.

The built-in demo corpus is a single paragraph repeated many times, so both its
train and val bpc go near zero (there is nothing to generalize to) — it exists to
prove the pipeline learns, not as a benchmark. The real test is a natural-language
corpus:

---

## The corpus

`input.txt` is **1,092,423 characters of ordinary English prose** over a
**77-character** vocabulary: four complete public-domain novels from Project
Gutenberg, chosen for plain, modern-reading narrative with a lot of dialogue —
the goal is a model that writes normal English, so the training data has to be
normal English.

| book | author | chars |
|------|--------|------:|
| *The Adventures of Sherlock Holmes* | Arthur Conan Doyle | 561,910 |
| *The Wonderful Wizard of Oz* | L. Frank Baum | 207,710 |
| *The Time Machine* | H. G. Wells | 179,656 |
| *Alice's Adventures in Wonderland* | Lewis Carroll | 143,140 |

It is **reproducible, not a mystery blob** — [`build_corpus.py`](build_corpus.py)
downloads each book, strips the Project Gutenberg licence header/footer, removes
editorial insertions (footnotes, `[Illustration]` markers, `_italics_` underscores,
asterisk scene-breaks), and folds the typography down to printable ASCII (curly
quotes → straight, em dash → `--`, `café` → `cafe`):

```bash
python build_corpus.py            # rebuild input.txt from scratch (downloads cached)
python build_corpus.py --check    # hash + stats for the file already on disk
```

That last fold matters more than it looks: a character LM has **one input neuron
per distinct symbol**, so every stray typographic variant is a neuron that sees
almost no training signal. The script pins the finished corpus's sha256
(`7b0f147a…`) and warns if it ever changes, so an upstream re-release cannot
silently invalidate the numbers below.

---

## Results

Trained on that corpus (983 K train / 109 K held-out val chars) on the 8 GB
RTX 5060 with a 1.21 M-parameter model (hidden 1024, 2 layers), 16 000 updates,
~65 min:

```bash
python snn_char_lm.py train --data input.txt --steps 16000 --hidden 1024 \
    --seq-len 160 --dropout 0.2 --weight-decay 2e-4 --val-split 0.1 \
    --seed 1337 --deterministic-eval --ckpt prose.pt
```

> Re-running this **overwrites** the `prose.pt` shipped in the repo: training
> keeps only the best-validation checkpoint *of the current run*, so point
> `--ckpt` at a scratch name if you want to preserve the shipped one.

| metric | value |
|--------|-------|
| held-out **validation bpc** | **2.310** (perplexity 4.96) |
| parameters | 1.21 M |
| training loss (running avg, dropout on) | 2.11 bpc |
| peak VRAM | 0.72 GB |
| throughput | ~4.3 updates/s |

With dropout at 0.2, validation bpc fell from 2.83 at the first eval to 2.31 and
**never turned back up** — over the last ~2 000 updates it simply flattened (best
eval at update 15 500; the final one is 0.001 bpc behind it). So the budget was
spent, not wasted, but the curve is flat at the end: more updates alone would buy
little without more capacity. The best-validation checkpoint is the one kept on
disk.

The two bpc rows above are measured differently: validation bpc uses the
`evaluate()` path (dropout off, fresh membrane, full windows), while the 2.11
"training" figure is the running average of the dropout-**on** training loss, so
the two are not directly comparable. Measured through the same `evaluate()` path
with dropout off, **train bpc is 1.98** — a 0.33 bpc generalization gap. The
model does fit the training text better than held-out text, as any model does;
what matters is that the gap stopped widening instead of blowing open. Because
the run used `--deterministic-eval` (a fixed full-split sweep under a fixed
encoder RNG), the validation figure is *exactly* reproducible rather than a noisy
point estimate — re-check it any time with `eval --deterministic`.

A sample from that checkpoint (`--prompt "It was " --temperature 0.5 --seed 7`,
verbatim, line-wrapped here to fit the page):

```
It was a little me, and the Time Traveller was a good came to my head of the
painted to carry and the corner, but I think that o the party in the
scenertainly as we had been passed which has already dear my hand we can of the
paper from the side of the finished me a four life an in the the golder and the
next morning to the little problem to see the windows of the would be as I had
the strange in the nex
```

It is not fluent — a 1.2 M-parameter spiking net at 2.3 bpc won't be — but it is
recognisably *ordinary English*: almost every token is a real word, the function
words are placed like English function words, the clause rhythm and punctuation
are right, and it has picked up the corpus's own vocabulary ("the Time
Traveller"). What it lacks is meaning held across a whole sentence, which is
exactly what a model this size at this bpc should lack.

### How it was tuned

A short sweep (each config trained to a fixed 2000-update budget, ranked by
held-out bpc) settled the architecture knobs:

| change vs. baseline (h512, 2 layers) | params | val bpc @ 2k |
|--------------------------------------|:---:|:---:|
| baseline (hidden 512) | 0.34 M | 2.715 |
| width → hidden 768 | 0.71 M | 2.591 ✓ |
| **width → hidden 1024** | **1.21 M** | **2.523** ✓ |
| width → hidden 2048 | 4.51 M | 2.412 ✓ |
| depth → 3 layers | 0.60 M | 2.886 ✗ |
| longer gradient → chunk 64 | 0.34 M | 2.697 (no real change) |
| learnable `beta` | 0.34 M | 2.926 ✗ |

The lesson: **width helps and extra depth hurts** — more stacked spiking
nonlinearities are harder to train through (a finding Tier 6 later qualifies,
below). A learnable membrane time constant hurts too, and lengthening the TBPTT
chunk from 32 to 64 buys essentially nothing, so the cheaper chunk is kept.

Width had not stopped paying off at hidden 2048 (2.412) when this sweep ended.
The shipped model is **hidden 1024** anyway: 2048 costs 3.7× the parameters for
0.11 bpc at this budget, which is the wrong trade for a project whose point is
that a spiking net is parameter-efficient. Whether 2048 still wins at the full
16 000-update budget — where it has far more opportunity to overfit — is **not
tested here**; it is listed as a follow-up in [`TIER6.md`](TIER6.md). The final
model is wide and shallow, trained long with dropout.

**Tier 6 revisited these knobs** with a deterministic-eval sweep (exactly
reproducible bpc; see [`TIER6.md`](TIER6.md)) and found two changes the original
sweep lacked that turn the depth story around:

| Tier 6 ablation @ 2k (hidden 512) | val bpc |
|-----------------------------------|:---:|
| baseline (2 layers, rate coding) | 2.715 |
| per-neuron learnable `beta` | 2.805 ✗ |
| 3 layers | 2.886 ✗ |
| **3 layers + LayerNorm on currents** | **2.635** ✓ |
| **graded (deterministic) input coding** | **2.615** ✓ |
| **all three together** | **2.484** ✓✓ |

So **LayerNorm rescues depth** (3 layers goes 2.886 → 2.635, now *beating* the
2-layer baseline) and **graded input coding helps** (2.615) — and they **stack**
(2.484, a 0.23-bpc gain over baseline). A per-neuron learnable time constant still
doesn't help at this budget. These are opt-in flags (`--layernorm`,
`--input-coding graded`, `--beta-per-neuron`); the shipped checkpoints use none.

For reference, this corpus's own entropy floors — computed on the shipped
`input.txt` by `corpus_bpc_floors` in `snn_char_lm.py` — are **4.48 bpc**
(unigram) and **3.50 bpc** (bigram, conditioning on the previous character).
This spiking model at 2.31 bpc lands **well below the bigram floor** — it is
genuinely modelling English from context, not just character frequencies.

For a non-spiking control, a **matched GRU baseline** — same corpus, same width
(1024) and budget (16k updates), trainable with `--arch gru` — reaches
**2.19 bpc**, 0.12 below this SNN. But the GRU carries **~10.5× the parameters**
(12.75M vs 1.21M): its dense recurrent matrices cost what the SNN gets for free
from the parameter-less leak. So at matched width and budget the ANN wins, but
narrowly, and the SNN is strikingly parameter-efficient. (The classical
"~1.4 bpc" LSTM/GRU figure comes from much larger models trained far longer on
other corpora such as PTB/enwik8.)

---

## Fitting an 8 GB GPU

- **Memory is not the bottleneck.** At the default config (hidden 512, 2 layers,
  `T=5`, seq 128, batch 128) peak VRAM is ~0.3 GB — TBPTT keeps only one chunk's
  activation graph alive at a time. You can push `--batch-size 256` and still
  stay comfortably under ~1.6 GB.
- **Throughput is launch-bound.** The model is a long chain of tiny sequential
  spiking kernels, so the GPU is mostly waiting on kernel launches, not compute.
  Because of that, a large batch is nearly free (wall-clock barely changes) — so
  prefer a big batch and fewer optimizer updates over a small batch and many.
- **The launch wall is beatable.** Because the cost is launch overhead, a
  **CUDA-graph capture** of the fixed-shape inner loop replays it as a single
  launch — the `bench` subcommand measures **~8× (hidden 1024) to ~13× (hidden
  512)** on the forward inner-loop core. (`torch.compile` is not an option here —
  Triton has no Windows / Python 3.14 build.) TF32 and AMP, by contrast, are *not*
  levers: faster matmuls don't help a launch-bound net. Details in
  [`TIER6.md`](TIER6.md).

---

## Command reference

`train` (key options; see `python snn_char_lm.py train -h` for all):

| flag | default | meaning |
|------|---------|---------|
| `--data` | built-in demo | UTF-8 text file to train on |
| `--steps` | 3000 | optimizer updates (one per TBPTT chunk) |
| `--val-split` | 0.1 | fraction of the corpus tail held out for validation (0 disables) |
| `--eval-every` | 500 | run held-out evaluation every N updates |
| `--eval-batches` | 20 | batches averaged per held-out evaluation |
| `--batch-size` | 128 | windows per update |
| `--seq-len` | 128 | characters per window (context length) |
| `--tbptt-chunk` | 32 | TBPTT chunk; backprop depth = `chunk × num_steps` |
| `--hidden` | 512 | LIF neurons per recurrent layer |
| `--layers` | 2 | stacked `snn.Leaky` layers |
| `--num-steps` | 5 | SNN micro-steps per character (`T`) |
| `--beta` | 0.9 | membrane leak |
| `--threshold` | 1.0 | LIF firing threshold |
| `--surrogate` | atan | `atan` \| `fast_sigmoid` \| `sigmoid` |
| `--lr` | 3e-3 | AdamW peak LR (warmup + cosine decay) |
| `--device` | auto | `auto` \| `cpu` \| `cuda` \| `cuda:N` |
| `--learn-beta` / `--learn-threshold` | off | make `beta` / the threshold trainable |
| `--save-state <f>` | — | also write a full **resumable** state (model+optimizer+RNG), refreshed each eval and at exit |
| `--resume <f>` | — | resume a run from a `--save-state` file (restores optimizer/step/RNG; continues toward `--steps`) |
| `--init-from <f>` | — | warm-start **weights** only (fresh optimizer + schedule) |
| `--log-csv <f>` | — | append `update,split,bpc,ppl,upd/s,peakGPU` rows for plotting |
| `--deterministic-eval` | off | fixed-window + seeded-encoder eval for stable best-checkpoint selection |
| `--beta-per-neuron` | off | *(ablation)* learn a per-neuron `[hidden]` β vector, not one scalar |
| `--layernorm` | off | *(ablation)* LayerNorm the inter-layer currents |
| `--input-coding` | rate | *(ablation)* `rate` (stochastic spikes) \| `graded` (deterministic current) |

Training saves the best-validation checkpoint to `--ckpt`; **Ctrl-C** saves current
progress before exiting. `--resume` and `--init-from` are mutually exclusive. The
ablation knobs and the ready-to-run experiment plan are documented in
[`TIER6.md`](TIER6.md).

`sample --ckpt <file> --prompt "..." --length N [--temperature 0.8] [--top-k K] [--seed S] [--device D]`
— generation is stochastic; pass `--seed` to reproduce an exact sample.

`eval --ckpt <file> [--data <file>] [--split all|train|val] [--val-split F] [--eval-batches N] [--seed S] [--deterministic]`
— score a checkpoint's bits-per-character on a corpus (mapped through the
checkpoint's vocab). Reproduce the headline number with:

```bash
python snn_char_lm.py eval --ckpt prose.pt --data input.txt \
    --val-split 0.1 --split val --deterministic
```

Add `--deterministic` for an exactly reproducible, seed-invariant number (a fixed
full-split sweep under a fixed encoder RNG, instead of a noisy random-window estimate).

`smoke` — fast self-test; exits non-zero if the model fails to learn.

---

## Development

```bash
pip install -r requirements.txt -r requirements-dev.txt   # runtime + pytest
pytest                                 # unit tests (fast, CPU-only)
python snn_char_lm.py smoke            # end-to-end integration self-test
```

Continuous integration (`.github/workflows/ci.yml`) runs the tests and a shrunk
`smoke` on a CPU-only runner. The project is MIT-licensed — see `LICENSE`.

---

## Files

| file | purpose |
|------|---------|
| `snn_char_lm.py` | the whole model, training loop, evaluation, sampler, and CLI |
| `build_corpus.py` | rebuilds `input.txt` reproducibly from Project Gutenberg (`python build_corpus.py`) |
| `requirements.txt` | runtime dependencies (+ note on the CUDA build of torch) |
| `requirements-dev.txt` | extra dependency for the test suite (`pytest`) |
| `pyproject.toml` | packaging metadata, console entry point (`pip install .`), and tool config |
| `tests/` | pytest suite (`pytest`); the `smoke` subcommand is the integration self-test |
| `README.md` | this file |
| `LICENSE` | MIT |
| `demo.pt` | a small checkpoint pre-trained on the demo corpus — try `python snn_char_lm.py sample --ckpt demo.pt` right away |
| `prose.pt` | the checkpoint behind the results above — `python snn_char_lm.py sample --ckpt prose.pt --prompt "It was "` |
| `input.txt` | the prose corpus: four public-domain novels, 1,092,423 chars, 77-char vocab, sha256 `7b0f147ae27cb1e276495d2a0fb697fc7f6e8a0932a20e56bcf641de5b1e5ac0`. Rebuild it with `build_corpus.py`. |
