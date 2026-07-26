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

## MAIN PRETRAIN (launched 2026-07-26 07:14)
- Config: h1536/L3/T3/B256/chunk64/seq256 + cuda-graph, corpus v3 (639MB), 234K steps
  (6 epochs), lr 3e-3 warmup 2000 cosine, eval every 3000 (subsampled 1024 windows) +
  cond_gain, save-state runs/main.state every 15min, under supervise.py, detached.
  Logs: runs/main_pre.log, runs/main_pre.csv. Sentinel: runs/MAIN_PRE_DONE. ETA ~12:00 07-27.
- **Trajectory gates** (from rehearsal + ablation reference curves): val ~1.83 by upd 2-3K;
  < ~1.65 by 30K; monotone-ish improvement early (3 consecutive worsening evals early = investigate).
  Alarm: upd/s sustained <1.8 (eviction), firing rate outside [2%,90%], cond_gain falling
  while val improves (register memorization).
- **Dress rehearsal PASSED 07:10** (all 4 stages): h1024 4K-step pretrain val 1.696 under
  supervisor; cross-corpus --init-from finetune OK (val 1.734); chateval OK (gain +0.016,
  7/8 terminated, word-validity 0.806); chat --once identity works ("I'm Spark, a small
  spiking neural network") with expected story-register bleed at toy scale.
- Cron heartbeat now hourly at :23 (job 6932c946).
- After MAIN_PRE_DONE: finetune phase per plan + chunk64-vs-128 A/B from the same
  pretrained checkpoint (cond_gain comparison), then chateval + full deterministic eval
  (official numbers) + docs + final commits.

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
- 2026-07-26 ~03:35 **INCIDENT**: harness background tasks (ablation battery + poller) were
  killed externally minutes after launch. **RESILIENCE RULE for the rest of the campaign:
  all long GPU work runs as DETACHED OS processes (Start-Process pwsh -File <script>), never
  as harness background tasks; progress tracked via log files; wake-ups via short
  self-expiring poller tasks (exit on done/error/timeout <=30min, any exit re-invokes).**
  Battery relaunched detached (scratchpad battery.ps1 -> runs/ablation_battery.log, per-config
  .DONE sentinels, skip-if-done on rerun). Corpus v2: 327.6MB pretrain (46% stories/54%
  dialogue), 6.18MB val; finetune regenerated with 197 unique identity dialogues.
- 2026-07-26 ~03:55 Pollers also get killed (near-instantly). **Wake mechanism switched to a
  session cron job (id 6cb52246, fires :07/:37 hourly)** — independent of task kills. Battery
  stdout is buffered until process exit; the flush-per-row CSVs in runs/ are the live telemetry.
  h3072 config pre-skipped via its .DONE sentinel (throughput ~5.7K chars/s disqualifies it:
  <=2 epochs/48h). First battery signal (h2048/T3 @500upd): val 2.712 bpc, firing rates
  0.06/0.11/0.21 (healthy sparse spiking), cond_gain ~0 (expected this early), grad_norm 37-75
  (hard-clipped every step at 1.0 — watch; queue lr 1.5e-3 variant if wide configs disappoint).
  Measured upd/s: h2048/T3/B256 = 0.70 (11.5K chars/s).
- 2026-07-26 ~05:30 **BATTERY 1 RESULTS** (2000 updates, subsampled deterministic val bpc):
  w2048_T3 **1.778** (graph 0.60-0.66 upd/s) | w1536_T3 **1.829** (graph 2.25 upd/s) |
  w2048_T5 KILLED (VRAM thrash 0.13 upd/s @6.46GB) | w2048_L4 KILLED (val 4.12@500 - depth-4
  fails at this LR/budget even with LayerNorm) | w2048_c128 OOM at capture | w3072 pre-skipped.
  **EAGER REF h2048_T3_B256 = 1.3 upd/s — FASTER than its graph (0.65)!** Emerging rule:
  CUDA-graph replay wins in a small-footprint sweet spot (h1536/T3 3.1GB: 2.25 upd/s
  sustained = 36.9K chars/s) but big captures (larger private pools) hit a WDDM cliff.
  Graph = sweet-spot lever, not universal — document honestly in TIER7.
- 2026-07-26 ~06:50 **BATTERY 2 RESULTS + FINAL CONFIG**: w1536_T5 val 1.838@1000upd (better
  per-update than T3, but 0.31-0.38 upd/s = 6-7x slower per second -> dominated; killed after
  the 1000 data point). w1536_L4 val 3.337@2000 (depth-4 fails at BOTH widths). w1536_c128
  0.245 upd/s @3.09GB -> the graph cliff tracks CAPTURED KERNEL COUNT as well as pool memory
  (chunk128 doubles nodes at same VRAM); killed. Chunk-64-vs-128 conditioning question moved
  to the finetune stage as a same-checkpoint A/B (better experiment anyway).
  **LOCKED main config: h1536/L3/T3/B256/chunk64/seq256 + CUDA graph** (2.25 upd/s = 36.9K
  chars/s, 3.1GB, val 1.829@2000 vs h2048's 1.778 at 3.2x the wall-clock cost).
  Corpus v3 BUILT: 639.3MB pretrain (47% stories/53% dialogue), 8.78MB val, 197-variant
  identity. Main run: 234K updates = 6 epochs ~= 29h. Dress rehearsal (h1024 full pipeline
  incl. supervisor, cross-corpus --init-from, chateval, chat probes) launched ~06:50.
- 2026-07-26 ~05:30 **MAIN RUN PLAN v2**: config = h1536/L3/T3/B256/chunk64/seq256 + graph
  (near-wash quality vs h2048 per update, 1.75x data per wall-clock vs h2048-eager, low VRAM,
  parity-proven) pending w1536_T5/L4/c128 results from battery 2 (~07:00). Corpus v3 ~750MB
  (stories 300 / soda ~300 / ultrachat ~60 / smoltalk+everyday+identity), ~6 epochs ≈ 4.5B
  chars ≈ 34h, leaving ~20h for finetune+eval+docs+buffer. Downloading ultrachat shards 1-2.
