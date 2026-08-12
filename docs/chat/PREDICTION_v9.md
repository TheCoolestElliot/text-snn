# PREDICTION v9 — the corpus round

**Written 2026-08-11, before any arm of this round has been trained.** Round v8
(`QUALITY_v8.md`) trained nothing and changed only how a reply is chosen. It
ended by locating the constraint somewhere the decoder cannot reach, and this
round is the first attempt on that constraint.

Not part of the Phase-1..5 research protocol. No number here is a reported
figure. Rates follow [`CONVENTIONS.md`](CONVENTIONS.md).

---

## 0. What v8 established, and why it dictates this round

The decode-time lever is spent. An oracle over the drafts the model already
produces went from 0.033 at `n = 4` to 0.450 at `n = 128`, and the echo
partition now extracts most of it — but on **held-out topics** the oracle is
only **0.0917 at n = 8** and the partition saturates at 0.300 by `n = 64`. The
model rarely *drafts* a reply about an unfamiliar noun at all.

Two readouts built in v8 make this round measurable in a way earlier rounds were
not, and both are the reason to spend GPU now rather than earlier:

* **`scripts/chat/echo_holdout.py`** — 20 nouns absent from `quality.PROBES`,
  paired on one shared pool, resolved a real effect at p = 0.0039 on 120 draws.
  Its **oracle** column is a corpus measurement wearing a decoder's clothes: it
  asks whether the model can produce a reply about a noun at all, independent of
  selection. That is the quantity every arm below is aimed at.
* **`scripts/chat/memory_probe.py`** — cross-turn recall, currently **0/240**
  with a control at 0/240, 95 % CI [0, 0.0158]. A floor of exactly zero is the
  ideal baseline for A14.

## 1. The primary gate, fixed here

> **Primary: held-out ORACLE topicality**, `scripts/chat/echo_holdout.py`
> `--seeds 6 --n 8`, reported as k/120 with a 95 % Wilson interval.

Chosen over the battery's `topic` column for a reason v8 measured: with the echo
partition on, the battery's `topic` column and the oracle **coincide exactly**
(both 0.2969), because the probe asks whether the reply contains the requested
noun and the partition prefers replies containing it. That column is circular
and cannot evidence a corpus change. The held-out set's nouns appear in no probe
in `quality.PROBES` and were fixed in committed code before this round.

`n = 8` and `--seeds 6`, matching the committed `echo_holdout.json`, so every
arm is comparable with the incumbent's **11/120** without re-running it.

**Lattice.** 120 draws quantise at 1/120 = 0.00833. Every threshold below is set
at a half-integer count (e.g. 17.5/120) so it **cannot be hit exactly** —
`CONVENTIONS.md` §4 rule 2, which P1 died of.

**Power, stated before running.** McNemar on paired draws with the incumbent's
11 successes: to resolve at 80 % power, α = 0.05, an arm needs roughly **20+
discordant-favourable draws**, i.e. an oracle of about **22/120 (0.183)** against
11/120. Anything smaller returns `unresolved`, and this document says so in
advance rather than after.

**Secondary, reported always, gating nothing**: `canned_rate`, `list_strict`,
`story_dodge` with its interval, weighted held-out bpc per source, and the
`memory_probe` table. Secondary columns may not be promoted to primary after the
fact — the failure `QUALITY.md` §5 documents three instances of.

**Seed counts are fixed here and honoured whatever the interim numbers say.**

## 2. The arms

Every arm changes **one** thing against `chat-v3d-aligned`'s recipe (14,000-step
anneal from `chat-v2-anneal`, `bot_loss_weight 3.0`, `align_frac 0.75`,
`seq_len 256`, `B 160`), and every corpus change packs to a **new source name**.
Repacking `stories_topic.bin` in place would make eight committed checkpoints
unreproducible; `build_topic_stories.py` now refuses it.

| # | arm | one change | cost |
| --- | --- | --- | ---: |
| A9 | `chat-v9-raw05` | `stories_topic_r05` at `--raw-fraction 0.05`, same 0.40 weight | 2.8 GPU-h |
| A11 | `chat-v9-turn600` | alpaca/dolly/oasst1 repacked at `MAX_TURN_CHARS 600` | 2.8 GPU-h |
| A12 | `chat-v9-narrative` | `soda_narrative` at 0.10, taken from `soda` (0.20 → 0.10) | 2.8 GPU-h |
| A13a | `chat-v9-bot1` | `bot_loss_weight` 3.0 → 1.0, `align_frac` held at 0.75 | 2.8 GPU-h |
| A13b | `chat-v9-align0` | `align_frac` 0.75 → 0.0, `bot_loss_weight` held at 3.0 | 2.8 GPU-h |
| A14 | `chat-v9-len1024` | `seq_len` 1024, `B` 40 (B·L held at 40,960) | 7.6 GPU-h |
| A15 | `chat-v9-scratch-s{1,2,3}` | `chat-v6-scratch`'s recipe at 3 more seeds | 11.9 GPU-h |

**n = 4 seeds** for A9, A11, A12, A13a, A13b. **n = 1** for A14 (a shape change
that must first be shown to train at all). A15 is 3 seeds added to an existing 1.

### Per-arm predictions, written before the runs

**A9 — raw fraction.** Cutting bare narrative from 15 % to 5 % raises the share
of `stories_topic` characters that are request-conditioned by 1.12×, and every
noun topic gains equally — unlike `TOPIC_DOSE_NOTE.md` §4's re-weighting, which
is zero-sum and under its own `1/count^0.5` makes **14 of 15 word probes lose
dose**. *Predicted: oracle rises, and by less than the bar.* **P9: oracle ≥
17.5/120. Expected outcome: FAIL.** Recorded because a 1.12× dose change that
did move the oracle would be strong evidence dose is the mechanism, and the
cheapest way to learn dose is *not* the mechanism is to run it once and stop.

**A11 — turn length.** At 400 characters the filter discards 56.4 % of alpaca,
85.1 % of dolly and 78.3 % of oasst1; 600 recovers +62 %, +67 % and +108 %.
None of that is story data. *Predicted: oracle unchanged (`unresolved`),
`list_strict` up, `canned_rate` down* — more real instruction-following text is
the direct competitor to persona recitation. **P11: `canned_rate_list` below
0.30 (pooled 17-arm value is 0.3824, per-arm range 0.30–0.50).**

**A12 — SODA narrative.** ⭐ *The arm this round exists for.* Every prose source
in the mixture except oasst1 (0.3 M) and alpaca (8 M) is TinyStories wrapped
four ways; `soda_narrative` is ~218 M characters of adult third-person prose
already on disk, 99.89 % of it under the turn filter. If the held-out oracle is
low because the model has seen a three-year-old's vocabulary, this is the arm
that moves it. **P12: oracle ≥ 22.5/120 against the incumbent's 11/120.**
*Expected outcome: PASS is genuinely uncertain; this is the round's real bet.*
Risk stated in advance: 0.10 of weight is taken from `soda`, so a loss on
turn-taking register would be attributable to the swap and not to the new source.

**A13a/A13b — the oldest open item.** `bot_loss_weight` and `align_frac` have
moved together since `chat-v3b`, so nothing establishes which bought the +31 %
prompt-dependence gain. *Predicted: at least one of the two is inert.* **P13:
the two arms' oracle counts differ from each other by ≥ 8.5/120.** If both land
inside the incumbent's interval, every arm since `chat-v3b` has been paying for
a term that does nothing, and that is worth knowing even though it ships nothing.

**A14 — window.** `seq_len` has been 256 in all 22 committed configs. At 1024
the `stories_topic` dose rises 0.2595 → 0.889 with the corpus unchanged to the
byte. **Primary for this arm is `memory_probe`, not the oracle**: it is the only
arm that could move a number currently pinned at exactly 0/240, and a floor of
zero is the cleanest baseline in the package. **P14: `memory_probe` told-rate ≥
4/240 with the control below it.** Two risks, both priced now: 4× the BPTT depth
and 4× less batch averaging against a base rate of 2 NaN divergences in 22 runs,
so **~10 % chance of a rollback**; and a dose of 0.889 re-occupies
`chat-v4a-short`'s already-measured 0.877, which scored *worse*. Smoke-test
CUDA-graph capture with `--max-steps 20` before committing the budget.

**A15 — the scratch replicate.** Deliberately **last**, and its justification is
weaker than when it was ranked first. `chat-v6-scratch` has the highest
`pool_by_kind.topic` of 17 arms (0.1191) — but `BUILD_NOTES.md` §12 records that
its mixture is byte-identical to the incumbent's, so their held-out bpc **is**
comparable, and it is worse on **every source** (weighted 1.2399 vs 1.1743). It
may simply be a worse language model that scores well on one topicality column.
**P15: its 4-seed mean oracle ≥ 17.5/120 AND its 4-seed mean weighted bpc within
0.02 of the incumbent's 1.1743.** The second conjunct is new and is the point:
without it this arm can win the round while being worse at English.

## 3. Stopping rules

1. **A15 does not run unless A12 has been scored**, and does not run at all if
   the bpc conjunct of P15 already fails on the existing seed — it does, at
   1.2399 vs 1.1743, so **A15 is provisionally cancelled** and runs only if a
   reader argues in writing that held-out bpc is the wrong reading. That
   argument must be made before the run, not after.
2. **Any arm that hits a gradient-norm rollback is reported as rolled back and
   is not silently pooled.** `chat-v6-inst-a-s1` ran its last 9,900 of 13,900
   steps at half its siblings' learning rate after a rollback and sits inside
   the only SD_seed estimate this package has.
3. **No arm displaces `SHIPPED` on an `unresolved` margin**, whatever its point
   estimate.

## 4. What this round cannot settle

The headline is still 89 % `list` variance at 20 draws and 38 % recitation, and
nothing here fixes it — `canned_rate` only makes it visible. The oracle gate is
120 draws on 20 nouns and is a statement about those 20 nouns. And no arm here
addresses the finding that beyond `n = 64` the echo partition saturates while
the oracle keeps rising: that is a selector problem, it is the largest unclaimed
gap in the package, and it needs no GPU at all.
