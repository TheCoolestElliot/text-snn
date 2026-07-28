# Spark — a spiking neural network you can chat with

A character-level language model whose hidden layers are **recurrent
leaky-integrate-and-fire (LIF) spiking neurons**, built with
[PyTorch](https://pytorch.org) + [snnTorch](https://snntorch.readthedocs.io),
trained end-to-end with **surrogate-gradient backpropagation through time** on
a single 8 GB RTX 5060 — and fine-tuned into a tiny conversational assistant:

```
you> Hello!
snn> Hello! How can I help you today?
you> Who made you?
snn> Elliot made me. I am called Spark, and I am a tiny spiking neural
     network. I only have a few million connections, so I make mistakes
     sometimes.
you> How are you today?
snn> I'm doing well, thanks for asking. How about you?
you> What's new with you?
snn> Not much. Just getting some work done. What about you?
```

*(verbatim model output — `spark.pt`, 5.04 M parameters, seeds in
`TIER7.md`; see "What it can and can't do" below for the honest failure
modes too)*

Everything lives in one heavily-commented file, [`snn_char_lm.py`](snn_char_lm.py).
Search it for `# DESIGN:` to jump between the explanations of each SNN design
decision. The project grew in tiers — hardening (1–4), usability (5),
research experiments (6, [`TIER6.md`](TIER6.md)), and the five-day chat
campaign (7, [`TIER7.md`](TIER7.md), raw log in [`CAMPAIGN.md`](CAMPAIGN.md)).

---

## Quick start

```bash
pip install -r requirements.txt        # see the file re: CUDA build of torch

# Chat with the shipped model (works on CPU too, just slower):
python snn_char_lm.py chat --ckpt spark.pt
python snn_char_lm.py chat --ckpt spark.pt --once "Who are you?"

# Prove the training pipeline end-to-end in ~1-2 min:
python snn_char_lm.py smoke

# Score the shipped model's headline number yourself (exactly reproducible):
python snn_char_lm.py eval --ckpt spark.pt --data corpus/v3/pretrain_val.txt \
    --split all --deterministic       # -> 1.2362 bpc (needs the corpus, below)

# Chat-quality report card (conditioning gain, termination, word validity):
python snn_char_lm.py chateval --ckpt spark.pt \
    --pairs corpus/v3/conditioning_val.jsonl --data corpus/v3/pretrain_train.txt
```

The training corpus is rebuilt reproducibly by
[`build_chat_corpus.py`](build_chat_corpus.py) (downloads ~4 GB of permissively
licensed sources once, cached; writes ~660 MB of training text).

---

## The model

| | |
|---|---|
| architecture | 3 stacked recurrent `snn.Leaky` LIF layers, hidden 1536, LayerNorm on inter-layer currents |
| input | graded (deterministic) current injection of the one-hot character |
| micro-steps | T = 3 spiking steps per character; rate-decoded linear readout |
| parameters | **5.04 M** |
| vocabulary | fixed 99 chars: newline + 95 printable ASCII + 3 chat markers (`\x01` user, `\x02` assistant, `\x03` end-of-turn) |
| context | 256-char windows, membrane state carried across the whole window (TBPTT chunk 64) |
| training | 234 000 updates = 6 epochs of 639 MB, 29.3 h on the RTX 5060, zero crashes |

The recurrence is the **leaky membrane itself** (spectral radius β < 1,
contractive and stable) — not an explicit lateral weight matrix, which was
built, measured, and rejected as chaotic in Tier 2 (see `# DESIGN: recurrence`).
During chat, one membrane state persists across the whole conversation: the
dialogue literally lives in the spiking dynamics.

Mean firing rates during training sat at **1.6% / 1.3% / 13%** per layer — the
network settled into an extremely sparse spike code (logged every eval in the
`fire_rates` CSV column; a dead or saturated layer would be invisible in bpc
but obvious here).

## Results

All bpc figures are deterministic full-split sweeps (exactly reproducible,
seed-invariant) on the held-out stratified validation files.

| model | params | pretrain-val bpc | conditioning gain | notes |
|-------|-------:|:---:|:---:|---|
| **`spark.pt`** (SNN, after chat fine-tune) | **5.04 M** | **1.2362** | **+0.038** | word-validity 0.95–0.99, 79% of replies self-terminate |
| SNN before fine-tune (`runs/main_pre.pt`) | 5.04 M | 1.2146 | +0.039 | |
| GRU control, same corpus & harness | 41.9 M | 1.0512 | +0.082 | best val reached in 1.7 h, then **overfits**; SNN never did |
| *previous model (`prose.pt`, Tier 6)* | 1.21 M | *2.310 — old 1.1 MB corpus; not comparable* | — | |

**Conditioning gain** is the metric that separates a chat model from a fluent
monologue generator: held-out assistant replies are scored teacher-forced given
their *true* user turn vs a *shuffled* one; only a model whose replies depend
on the prompt scores better on the true pairing. The GRU control earns its
0.19-bpc edge with **8.3× the parameters** and dense recurrent matrices; the
spiking model's leak-based recurrence gets its memory for free. On this corpus
the GRU also *overfit* from update 24 K onward while the SNN improved
monotonically for 29 hours — capacity cuts both ways.

### What it can and can't do

At 5 M parameters, `spark.pt` is TinyStories-class. It reliably handles
greetings, small talk, identity questions, and short everyday exchanges with
correct spelling (word validity ≥ 95%) and mostly clean clause-level grammar.
It equally reliably **wanders after 1–2 sentences** (often toward its
storybook register — ask it for a story and you'll get one about a dog named
Buddy), fails on niche topics, and has no factual knowledge or multi-turn
reasoning. These are capacity limits, documented honestly, not bugs: the same
harness at 42 M parameters (the GRU) is visibly sharper. The `chat` defaults
(temperature 0.9, top-p 0.9, 300-char replies, verbatim-loop guard) are tuned
for its strengths.

## The corpus is the register

The Tier 6 model sounded like the 1890s because it read the 1890s. Spark reads
~660 MB of modern English in two registers — **simple grammatical prose**
(TinyStoriesV2, GPT-4-written) and **short everyday + assistant dialogue**
(SODA, smol-smoltalk, ultrachat_200k, everyday-conversations, all filtered to
plain ASCII prose), framed with the three chat markers so that the assistant
marker only ever precedes dialogue-register text. A handwritten identity set
("I'm Spark, a small spiking neural network built by Elliot") is oversampled
to ~1.6% of the fine-tune corpus — enough to answer "who are you?", not enough
to parrot it at strangers (a failure mode round 1 actually hit; see TIER7.md).
Full sourcing, licenses, filters, and design rules: [`TIER7.md`](TIER7.md).

## Speed: the CUDA-graph training step

The SNN's step is a chain of thousands of tiny sequential kernels, so it is
bound by kernel-*launch* overhead, not math. Tier 7 captures one TBPTT chunk's
**forward + loss + backward** as a single CUDA graph (`--cuda-graph`) and
replays it per update — after proving **bitwise-identical weights vs eager
training** with the `graphcheck` parity gate. Measured honestly, the win lives
in a sweet spot: 2.25 upd/s = **36.9 K chars/s** at the shipped config
(≈ 16× the launch-bound Tier 6 throughput per char), while larger captures
(wider, more micro-steps, longer chunks) hit a WDDM cliff where eager wins.
TF32/AMP re-tested post-graph and rejected again. Details + tables:
[`TIER7.md`](TIER7.md).

## Command reference

`train` — everything from Tier 6, plus: `--vocab fixed` (corpus-independent
99-char vocab), `--val-data` (separate stratified val file), `--cuda-graph`,
`--eval-max-windows` (subsampled deterministic eval), `--state-every-min`
(wall-clock resumable-state cadence, atomic with `.bak` rotation),
`--chat-pairs` (conditioning gain at every eval), `--tf32`.

`chat --ckpt spark.pt [--once "..."] [--temperature 0.9] [--top-p 0.9]
[--max-chars 300] [--seed N]` — REPL with persistent membrane state across
turns (`/reset` clears it, `/quit` exits); `--once` answers a single prompt.

`chateval --ckpt <f> --pairs <jsonl> [--data <corpus>]` — conditioning gain,
termination rate, distinct-3-gram, word validity, sampled transcripts.

`graphcheck` — eager vs CUDA-graph training parity gate (exits non-zero on
mismatch).

`eval` / `sample` / `smoke` / `bench` — as before (`eval` gains
`--max-windows`, `sample` gains `--top-p`).

Long unattended runs: `python supervise.py --state <f> -- python
snn_char_lm.py train ... --save-state <f>` (auto-resume on crash, `[done]`
completion detection, `.STOP` sentinel).

## Development

```bash
pip install -r requirements.txt -r requirements-dev.txt
pytest                                 # 57 unit tests, CPU-only, fast
python snn_char_lm.py smoke            # end-to-end integration self-test
python snn_char_lm.py graphcheck       # (GPU) graph-vs-eager parity
```

MIT-licensed — see `LICENSE`. CI runs the tests + a shrunk smoke on CPU.

## Files

| file | purpose |
|------|---------|
| `snn_char_lm.py` | model, training (eager + CUDA-graph), eval, chat REPL, chat metrics, CLI |
| `build_chat_corpus.py` | rebuilds the Tier 7 chat corpus reproducibly (~660 MB) |
| `build_corpus.py` | rebuilds the Tier 6 prose corpus (`input.txt`) |
| `supervise.py` | restart-on-crash wrapper for multi-day runs |
| `spark.pt` | **the chat model** (5.04 M params; `chat --ckpt spark.pt`) |
| `prose.pt`, `demo.pt` | historical Tier 6 checkpoints |
| `TIER7.md` / `TIER6.md` / `CAMPAIGN.md` | the chat campaign / research tier / raw five-day log |
| `tests/` | pytest suite (57 tests) |
