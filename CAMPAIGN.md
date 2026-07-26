# 5-Day Autonomous Campaign Log

**Mission (from Elliot, 2026-07-26):** make the SNN write modern conversational
English ("kind of like Claude", NOT archaic prose), with good grammar, as smart as
possible, with prompt->response input/output. Use the hardware maximally. 5 days,
fully autonomous, no user contact. It must remain a genuine spiking network.

**Deadline:** ~2026-07-31 00:00 (5 days from start, 2026-07-26 ~00:30).

## Hardware / environment (verified)
- RTX 5060 8 GB (Blackwell sm_120), driver 610.74, CUDA 13.3, WDDM mode, GPU shared with display.
- torch 2.12.1+cu130, snntorch 1.0.0, Python 3.14.6 (system install, not .venv), pyarrow 25.0.0 (installed by campaign).
- 16 GB RAM, 16 threads, 1 TB free disk. HuggingFace reachable.
- No torch.compile (no Triton for Windows/py3.14). CUDA graphs are the throughput lever (bench: 8-13x fwd).
- Machine never sleeps on AC (Sleep after = 0x0). Safe for multi-day runs.

## Baseline (committed)
- Commit `2107261` = prose-corpus switch (4 modern-reading novels, 1.09M chars, 77-char vocab).
- `prose.pt`: hidden 1024, 2 layers, T=5, rate coding, 1.21M params, **val 2.310 bpc** (deterministic).
- Tier 6 proven: graded input coding + LayerNorm + 3 layers = best combo (2.484 @ h512/2k vs 2.715 base). Never trained at full budget.
- GRU baseline h1024/16k: 2.187 bpc at 10.5x params.

## Plan of record (v2, post research workflow wf_087da4d6-c4f — see tasks/wlsfilj4m.output for full findings)

**Corpus** (sized 20-40 chars/param => ~100-200MB for 5-9M params; <=4 epochs of repeats OK):
- Sources: TinyStoriesV2 (grammar backbone, CDLA-Sharing-1.0, downloaded), smol-smoltalk
  (assistant register, Apache-2.0, downloaded), allenai/soda (short everyday dialogues,
  CC-BY-4.0, downloading), everyday-conversations-llama3.1-2k (clean, small, downloaded).
  REJECTED: oasst2 (multilingual/uneven), dolly (typos, CC-BY-SA), ultrachat (redundant).
- Mix ~35-40% stories / 60-65% dialogue. Stories stay UNFRAMED (never inside role markers);
  \x02 assistant marker only ever precedes assistant-register text.
- Role tokens: \x01=user-start \x02=assistant-start \x03=end-of-assistant-turn. Markers injected
  AFTER normalization (normalize() strips control chars). Loss on ALL text incl. markers (no masking).
- FIXED 99-char vocab in code: \n + \x01\x02\x03 + printable ASCII 32..126. Assert corpus subset.
- Document-level shuffle; explicit separate val file, stratified per source; report story-val and
  dialogue-val bpc separately. Identity data only in finetune, capped 2-5%, never in val tail.
- RAM: encode corpus via bytes->numpy->LUT, not per-char Python list (16GB machine).

**Arch**: graded coding + LayerNorm + 3 layers; hidden from measured VRAM/throughput (2048 target,
fall back 1536; keep peak <6.5GB with WDDM desktop); seq 256, chunk 64; T=5 default, T=3 only if
ablation shows <=+0.02 bpc AND healthy firing rates; NEVER T=2 (rate-code quantization).
**--dropout 0** (only RNG in graded path; never a proven SNN win) => whole fwd+bwd RNG-free.
Log per-layer firing rates every eval; alarm outside [2%,90%] (spiking-story evidence).

**Speed**: MANUAL whole-iteration capture per TBPTT chunk (torch.cuda.graph), NOT
make_graphed_callables: warmup 3+ full fwd+loss+bwd iters on side stream; zero_grad(set_to_none=True)
ONCE before capture and NEVER after; static input buffers copy_() per step; clip_grad_norm_ +
AdamW.step() eager between replays; loss accumulated on-GPU, .item() only at log boundary.
Parity gate: 50 updates eager vs graphed from identical state => weights allclose. In-flight eager
cross-check on a fixed probe batch each eval. TF32 on as free win; bf16 time-boxed 2h experiment
(membrane stays fp32). Default allocator (NO cudaMallocAsync). Keep graph scope = one chunk (TDR-safe).
Fallback if not parity-clean by end of day 1: eager, hidden 1536, batch 128.

**Ops**: atomic saves (tmp+os.replace) + A/B state rotation; wall-clock save cadence (~15-20min)
decoupled from eval; eval SUBSAMPLE (~2-4MB fixed windows) in-run, full sweep at stage end; CSV +=
timestamp, lr, grad-norm, firing rates; supervisor script auto-resumes on crash gated on progress
sentinel (TIER6 documents benign nonzero CUDA-teardown exit); throughput watchdog (upd/s drop =
VRAM eviction signature). No reboots (no TdrDelay/HAGS changes). Try pausing Windows Update.

**Eval harness (built BEFORE big run)**: per-source val bpc; conditioning gain (assistant-span bpc
true vs SHUFFLED user turn — the only metric that catches fluent-but-unconditioned failure);
turn-termination rate (\x03 within N chars); distinct-3gram + word-validity; identity probes;
fixed-seed transcript grid dumped each eval.

**Sampling defaults (chat)**: T=0.9 top-p 0.9 (or top-k 8-15); never greedy; no char-level
repetition penalty; loop guard (repeated >=16-char substring => temp bump); cap ~400 chars;
hard-stop on \x03.

**Schedule (revised)**:
- Day 0: code upgrades (fixed vocab, val-file, atomic saves, graph training, eval harness,
  chat REPL, corpus builder) + corpus build + parity gate + T/width micro-ablations.
- Day 1: MANDATORY dress rehearsal (h1024, 2-3h pretrain + 30min finetune + full eval suite) =>
  reference curve + 6h/24h bpc milestones for the supervisor. Then launch main pretrain.
- Days 2-3: main pretrain (~48h+), chat data upweighted during final LR-decay phase
  (MiniCPM decay-phase injection beats post-hoc SFT).
- Day 4: stage-2 finetune (identity + dialogue upweight, 0.1x LR, 30-50% pretrain replay,
  1-5% of pretrain tokens). Early-stop on prose-val regression >0.05 bpc.
- Day 5: final eval, README/TIER7 docs, tests, commits, demo polish. Buffer.

**Quality expectation to set in docs**: TinyStories-grade — grammatical simple sentences,
in-register replies, short-range coherence; NOT factuality or multi-turn memory. bpc on the new
corpus is NOT comparable to 2.31 on the old one (different entropy; expect well under 2).

## Task list (harness task IDs)
1 corpus | 2 cuda-graph training | 3 arch upgrades | 4 ablations | 5 main pretrain
6 chat finetune | 7 chat REPL | 8 final eval/docs/tests

## State / downloads
- [x] dolly-15k.jsonl (13 MB) -> .corpus_cache/hf/
- [ ] TinyStoriesV2-GPT4-train.txt (~2.2 GB) downloading, bg task b9qzyc1d7
- [ ] smol-smoltalk 4 parquet shards (~880 MB) + everyday-conversations, bg task bxf7p77s6
- [ ] Research workflow wf_087da4d6-c4f (datasets/cuda-graphs/tiny-chat-lm/plan-critic) running

## Decisions log
- 2026-07-26 00:20 Committed pending prose-corpus baseline as campaign base.
- 2026-07-26 00:30 pyarrow installed for parquet parsing (user-visible env change).

## Findings log
- 2026-07-26 ~01:00 **Corpus v1 built**: 130.5MB pretrain (46% stories/54% dialogue after
  filtering; smoltalk strict-prose filter passes only ~4% -> 13.9MB), 20.9MB finetune
  (identity 3.8%), stratified val files, 500 conditioning pairs. Builder: build_chat_corpus.py.
- 2026-07-26 ~01:30 **Code upgrades landed** in snn_char_lm.py: FIXED_VOCAB (99 chars incl.
  \x01\x02\x03 role markers), fast byte-LUT corpus encoding, --val-data separate val file,
  atomic checkpoint saves (+.bak rotation), wall-clock --state-every-min save cadence,
  CSV += ts/lr/gradnorm/fire_rates, spike_stats() firing-rate probe, eval --eval-max-windows
  subsampling, top-p/stop-char/ban-chars/loop-guard sampling, persistent-state generation
  (_generate_core), `chat` REPL (+--once), `chateval` (conditioning gain, termination,
  distinct-3gram, word validity), conditioning_metrics (true-vs-shuffled-prompt), CUDA-graph
  training (--cuda-graph via _capture_chunk_step), `graphcheck` parity gate, --tf32 knob.
  All 43 unit tests pass; smoke passes.
- 2026-07-26 ~01:45 **graphcheck PASS, bitwise**: eager vs graphed training weights IDENTICAL
  (max diff 0.0) after 24 updates at h256/L3/T3. The graph path is exactly correct.
- 2026-07-26 ~02:00 **Throughput measured** (real corpus, graph, h1536/L3/T3/B256/chunk64/seq256):
  2.2 upd/s = 36K chars/s, peak 3.12GB. Net is now COMPUTE-bound (launch overhead gone).
  **TF32: only +14% and slightly worse loss at 200upd -> rejected.**
  Key consequence: 48h of compute >> 4 epochs of 130MB, so corpus enlarged to ~360MB
  (stories 150 / soda 150 / smoltalk 35 / ultrachat 20 / everyday+identity) and model size
  to be picked by ablation (h1536/2048/3072, T3 vs T5, L3 vs L4).
- 2026-07-26 ~02:15 supervise.py written (gates on [done] marker, crash-loop brake, .STOP
  sentinel). Windows Update PAUSED until 2026-08-01 (admin shell).
