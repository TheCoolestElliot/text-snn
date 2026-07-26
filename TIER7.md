# Tier 7 — the chat campaign

Tier 7 turned the prose language model into a **conversational one**: a spiking
network that writes modern English, answers prompts, and chats — trained
end-to-end during a five-day autonomous campaign on the single RTX 5060.
(`CAMPAIGN.md` is the raw day-by-day log; this file is the distilled story.)

The result is **`spark.pt`**: ask it things with

```bash
python snn_char_lm.py chat --ckpt spark.pt            # interactive REPL
python snn_char_lm.py chat --ckpt spark.pt --once "Who are you?"
```

<!-- FINAL NUMBERS + sample transcript go here -->

## What changed and why

### 1. The corpus is the register

The Tier 6 model sounded like the 1890s because it read the 1890s. A character
model has no other prior — so Tier 7's biggest change is data. The new corpus
(`build_chat_corpus.py`, ~639 MB) is built from permissively licensed modern
sources in two registers:

| source | share | role | license |
|--------|------:|------|---------|
| TinyStoriesV2 (GPT-4-written) | ~47% | grammar backbone: simple, flawless modern English | CDLA-Sharing-1.0 |
| SODA | ~47% of dialogue | 550K+ short everyday two-person exchanges — teaches *reply conditioning* | CC-BY-4.0 |
| smol-smoltalk (filtered) | | assistant register at length | Apache-2.0 |
| ultrachat_200k (filtered) | | assistant register, uniformly clean grammar | MIT |
| everyday-conversations | | exactly the target register, small | Apache-2.0 |
| handwritten identity dialogues | <0.5% | "I'm Spark, a small spiking neural network built by Elliot" | — |

Design rules that mattered (each traces to a measured failure mode):

- **Fixed vocabulary** (`FIXED_VOCAB`, 99 chars): newline + 95 printable ASCII +
  three control characters framing conversations —
  `\x01` user turn, `\x02` assistant turn, `\x03` end of assistant turn.
  A corpus-derived vocab ties a checkpoint to one corpus; the fixed table is
  what lets one model pretrain on prose+dialogue and fine-tune on dialogue.
- **Single-character role markers**: one input neuron and one timestep each,
  impossible to emit malformed, unambiguous stopping — the char-level analogue
  of a chat template's special tokens.
- **Stories never wear the assistant marker.** `\x02` only ever precedes
  dialogue-register text, so the marker itself carries register information.
- **Strict plain-prose filtering** (no code, tables, URLs, non-ASCII) and turn
  length caps — short exchanges are the ones a truncated-BPTT gradient window
  can actually learn conditioning from.
- **A separately built, stratified validation file** instead of a tail split: a
  concatenated corpus's tail is 100% whichever source came last.

### 2. The CUDA graph moved from benchmark to training loop — and found its limits

Tier 6 measured an 8–13× forward speedup from CUDA-graph replay but never
trained with it. Tier 7 captures one TBPTT chunk's **forward + loss + backward**
as a single graph (`--cuda-graph`), with gradient clipping and the AdamW step
eager between replays, following the official whole-iteration recipe (grads set
to `None` exactly once before capture; never `zero_grad` after; RNG-free capture
via graded coding + dropout 0). The `graphcheck` subcommand is the correctness
gate: eager and graphed training produce **bitwise-identical weights** after 24
updates (max |diff| 0.0).

The honest performance story is more interesting than "8× faster":

| config (B256 chunk 64 unless noted) | eager | graphed | verdict |
|--------|:---:|:---:|---|
| h1536 / T3 (3.1 GB) | — | **2.25 upd/s = 36.9K chars/s** | the sweet spot |
| h2048 / T3 (4.1 GB) | **1.3 upd/s** | 0.65 upd/s | graph 2× *slower* |
| h2048 / T5 (6.5 GB) | — | 0.13 upd/s | VRAM-eviction thrash |
| h1536 / T3 chunk 128, B128 (3.1 GB) | — | 0.25 upd/s | node-count cliff |

CUDA-graph replay wins decisively inside a **small-capture sweet spot** and
degrades sharply as the captured kernel count or the graph's private memory
pool grows — on an 8 GB WDDM card that also drives the desktop, the cliff
arrives well before the VRAM ceiling. A launch-overhead lever, not a free
lunch; `TF32` was re-tested post-graph and rejected again (+14% speed, slightly
worse loss at equal steps).

### 3. Architecture: the ablation battery on the new corpus

2000-update runs, deterministic subsampled val bpc, identical seeds:

| config | val bpc @2k | speed | verdict |
|--------|:---:|:---:|---|
| **h1536, 3 layers, T=3** | 1.829 | 2.25 upd/s | **winner on wall-clock** |
| h2048, 3 layers, T=3 | 1.778 | 0.65 upd/s (graph) / 1.3 (eager) | better per update, 3.2× costlier |
| h1536, T=5 | 1.838 *@1000* | 0.35 upd/s | better per update, 6× costlier — dominated |
| h1536/h2048, 4 layers | 3.3–4.1 | — | depth-4 fails to train at this budget (both widths) |
| h3072 | — | ~0.35 (skipped) | disqualified by throughput |

Two Tier 6 lessons survived contact with the 600× larger corpus: LayerNorm'd
3-layer depth and graded input coding stay in; T=5's per-update advantage is
real but loses 6× on wall-clock to T=3 at fixed hardware.

<!-- ### 4. Main run + fine-tune results — fill after completion -->

### 5. Measuring what actually matters: conditioning

Validation bpc cannot tell a chat model from a fluent monologue generator. The
metric that can — `chateval` / `conditioning_metrics` — scores each held-out
assistant reply twice, teacher-forced: once given its **true** user turn and
once given a **shuffled** one. The difference (**conditioning gain**) is
positive only if replies actually depend on what the user said. `chateval` also
reports turn-termination rate (does the model *stop*?), distinct-3-gram rate
(does it loop?), and word validity.

### 6. Spiking health telemetry

With graded input coding the hidden layers' spikes ARE the spiking computation,
and LayerNorm could quietly compensate for a dead or saturated layer while bpc
looks fine. Training therefore logs per-layer mean firing rates every eval
(`fire_rates` CSV column) and warns outside [2%, 90%]. Throughout the campaign
the three layers sat at roughly 4–24% — sparse, healthy, genuinely spiking.

### 7. Five days unattended: the ops layer

- `supervise.py` relaunch wrapper — completion detected by the `[done]` marker
  (a Windows CUDA-teardown quirk makes exit codes lie), crash-loop brake,
  `.STOP` sentinel.
- Atomic checkpoints (`tmp` + rename, previous kept as `.bak`) and a
  wall-clock `--state-every-min` resumable-state cadence, decoupled from evals.
- Deterministic **subsampled** in-run evals (`--eval-max-windows`): a full
  sweep of a 9 MB val split runs at eager speed (~25 min) and would have
  throttled both eval and checkpoint cadence.
- Windows Update paused for the window; sleep already disabled.

## Reproduce

```bash
# corpus (downloads ~4 GB of sources once, cached)
python build_chat_corpus.py --out-dir corpus/v3

# parity gate for the graph trainer
python snn_char_lm.py graphcheck

# pretrain (~29 h on the RTX 5060)
python supervise.py --state runs/main.state -- \
  python snn_char_lm.py train \
    --data corpus/v3/pretrain_train.txt --val-data corpus/v3/pretrain_val.txt \
    --vocab fixed --input-coding graded --layernorm --dropout 0 \
    --layers 3 --hidden 1536 --num-steps 3 --seq-len 256 --tbptt-chunk 64 \
    --batch-size 256 --cuda-graph --steps 234000 --warmup 2000 --lr 3e-3 \
    --eval-every 3000 --eval-max-windows 1024 --deterministic-eval \
    --chat-pairs corpus/v3/conditioning_val.jsonl --sample-every 0 \
    --seed 1337 --ckpt runs/main_pre.pt --save-state runs/main.state \
    --log-csv runs/main_pre.csv

# chat fine-tune  <!-- exact recipe filled in after the A/B -->

# evaluate
python snn_char_lm.py chateval --ckpt spark.pt --pairs corpus/v3/conditioning_val.jsonl \
    --data corpus/v3/pretrain_train.txt
python snn_char_lm.py eval --ckpt spark.pt --data corpus/v3/pretrain_val.txt \
    --split all --deterministic
```
