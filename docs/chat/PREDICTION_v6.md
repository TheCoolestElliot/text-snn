# What the 2026-08-06 session-6 round predicts, written before it trains

This file exists so results can be scored against a claim made in advance
rather than explained afterwards, same discipline as `PREDICTION_v4.md` and
`PREDICTION_v5.md`. **Not part of the research protocol.** No control arms in
the protocol's sense, `n = 1` per arm, nothing here is a reported figure.

**Why a new document rather than a section of `QUALITY_v5.md`.** That file is
closed: `SHIPPED` still points at `chat-v3d-aligned` and its own scorecard
stands. This round is written up in `QUALITY_v6.md` instead.

---

## §0. The corpus-choice rule for the from-scratch run, and its trigger

`QUALITY_v6.md` §1 asked whether `chat-v3d-aligned` (shipped) actually beats
its strongest challenger, `chat-v5a-short300`, on the headline that decided
`SHIPPED` — the answer, read with Wilson CIs and sample SEs per
`docs/chat/CONVENTIONS.md`, was **UNRESOLVED** (z = −0.479 on the headline
diff; every component individually unresolved; `social` tied at ceiling for
both). Rule, decided before that number existed: **an unresolved comparison
does not displace the incumbent.**

**Trigger and its result, applied now:** the corpus used for `chat-v6-scratch`
below is the **incumbent's** recipe (`chat-v3d-aligned`'s mixture and alignment
settings), not the challenger's `stories_short300` mixture, because §1 did not
resolve in the challenger's favour. Had it resolved for the challenger, this
section would instead specify the challenger's mixture. It did not, so this is
the only branch that runs.

## §1. Hypothesis

Every arm since `chat-v2` has been a ~14,000-step anneal *continuing*
`chat-v2-anneal`'s weights — 76,000 + 8,000 steps trained on the corpus
*before* the topic-conditioning and window-alignment fixes existed. Three
rounds of mixture/objective tuning on top of that anneal have moved the
headline by single-digit-percent amounts and twice shipped nothing.
`docs/chat/QUALITY.md` §6 named the untried alternative first, before any of
those rounds: **train fresh, from random initialisation, on the corpus as it
exists now**, so later measurements are not "arguing with" stale pretraining.

**Prediction P0:** a fresh-init run of the same total step budget as
`chat-v2`+`chat-v2-anneal` (84,000 steps), on the corpus and alignment settings
`chat-v3d-aligned` shipped with, beats `chat-v3d-aligned`'s headline
(0.2969, `n=1, λ=0`) **and** that margin itself resolves (is not `unresolved`
by the same two-sample method `QUALITY_v6.md` §1 used).

## §2. Recipe: `chat-v6-scratch`

```
python scripts/launch.py --script scripts/chat/train.py -- \
    --run-name chat-v6-scratch --out-dir experiments/chat \
    --max-steps 84000 --batch-size 160 --seq-len 256 --lr 5e-4 \
    --bot-loss-weight 3.0 --align-frac 0.75 --align-lookahead 1024 \
    --budget-minutes 240 \
    --mix stories_topic=0.40 soda=0.20 alpaca=0.15 tinystories=0.09 \
          persona=0.08 dolly=0.06 oasst1=0.02
```

No `--init-from` — this is the whole point. Verified by `--dry-run` (both
`launch.py`'s and `train.py`'s own) before this file was committed: mix sums to
1.00, 4,417,637 parameters (identical architecture to every prior chat arm),
3.44 G characters over 84,000 steps, and `launch.py`'s tracked `run_dir`
resolves to `experiments/chat/chat-v6-scratch` — matching `train.py`'s own
`out_dir` default exactly, so `stdout.log` lands where it will be watched.
`--budget-minutes 240` is a hard ceiling (~4.0 GPU-hours) independent of
whether the cosine schedule completes; §3 below states in advance how a
budget-stopped run is read.

## §3. Decision threshold, and its power stated before the run

Read at the pinned `n=1, λ=0` row (`report["baseline_picks"]`), the same row
every prior round has used:

- **Ships, replacing `chat-v3d-aligned` as `SHIPPED`:** headline > 0.2969 AND
  the two-sample check (§1's method, this arm vs. `chat-v3d-aligned`) resolves
  in this arm's favour (does not land `unresolved`).
- **Does not ship, `SHIPPED` unchanged:** headline ≤ 0.2969, or the margin is
  `unresolved` even if the point estimate is higher — an unresolved margin is
  not evidence of a win, per §0's rule applied consistently to a new challenger
  as well as an old one.

**Power, stated now, not after:** `topic` at `n=1,λ=0` is 64 draws (16 probes ×
4 seeds); §1 measured this exact battery's Wilson width near p≈0.09–0.11 as
roughly [0.04, 0.21] on the existing arms. A plausible, fully expected outcome
is that `chat-v6-scratch` reads as `unresolved` against `chat-v3d-aligned` on
these components individually, and headline power is not much better (§1's
propagated SE was ≈0.03–0.04 against an observed gap of 0.019 between two
*existing* arms). This is written down so an `unresolved` result is reported as
what it is, not read as a quiet failure. The larger `pool_by_kind` reading
(~1,024 draws, resolves to ≈0.02) is declared here, in advance, as a **labelled
diagnostic only** — informative about the model's underlying propensity, not
about the decoder, and it cannot resolve the ship decision (mirrors
`PREDICTION_v5.md` §4's treatment of the opener-branch decomposition).

## §4. The instruction-weight ablation ladder

*(written during Phase 2, while `chat-v6-scratch` trains — appended below,
not backdated into §1–§3, which were committed before `chat-v6-scratch`
launched)*

`QUALITY_v5.md`'s own closing recommendation (echoed independently across two
sessions) was: **"instruction following, not topicality, is what the next
round should be aimed at."** The only evidence for it is two points —
`chat-v4a-short` (`list_strict` 0.033) and `chat-v4b-balance` (`list_strict`
0.083) — and they are not two points on an instruction-weight curve. Read
directly from the committed `config.json` files (not from a prose recap):

| arm | `stories_short` | `alpaca` | `dolly` | `oasst1` | instruction (alpaca+dolly+oasst1) | `soda` | `tinystories` | `persona` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `chat-v4a-short` | 0.40 | 0.15 | 0.06 | 0.02 | **0.23** | 0.20 | 0.09 | 0.08 |
| `chat-v4b-balance` | 0.26 | 0.26 | 0.08 | 0.02 | **0.36** | 0.21 | 0.08 | 0.09 |

`v4b` cut story weight by a third (0.40 → 0.26) **at the same time** it raised
instruction weight (0.23 → 0.36). `list_strict`'s gain cannot be attributed to
either alone from these two arms — this is exactly the trap `QUALITY_v5.md`'s
own precedent (story weight was eliminated as `story_dodge`'s cause by an
arm that changed *only* story weight) warns against skipping.

### Arm `chat-v6-inst-a` — de-confound `chat-v4b-balance` (must-do)

Holds `stories_short` at `v4a`'s 0.40 and `tinystories`/`persona` at `v4a`'s
values, raises instruction sources to `v4b`'s **exact absolute weights**
(`alpaca=0.26, dolly=0.08, oasst1=0.02`), and lets `soda` alone absorb the
difference (0.20 → 0.07) since it is dialogue register, not implicated in
either the story-weight or instruction-weight hypothesis:

```
--mix stories_short=0.40 soda=0.07 alpaca=0.26 tinystories=0.09 persona=0.08 dolly=0.08 oasst1=0.02
```

(sums to 1.00 — to be re-verified by `train.py --dry-run` immediately before
launch, per the standing rule.) Otherwise identical to `chat-v4a-short`'s
recipe: `--init-from experiments/chat/chat-v2-anneal/ckpt_best.pt --no-resume
--max-steps 14000 --batch-size 160 --seq-len 256 --lr 5e-4
--bot-loss-weight 3.0 --align-frac 0.75 --align-lookahead 1024
--budget-minutes 46`.

**Prediction P1a:** if `list_strict` moves toward `v4b`'s 0.083 while topic
propensity stays nearer `v4a`'s 0.0723 than `v4b`'s 0.0645, credit shifts from
"the story cut" to "instruction weight" as `list_strict`'s driver — the
opposite pattern (topic moves, `list_strict` doesn't) would instead implicate
the story cut, and a null on both would mean neither isolated knob reproduces
`v4b`'s effect, i.e. the effect was the *combination*, not the sum of parts.
All three outcomes are reportable; none is a failure to plan around.

### Arm `chat-v6-inst-b` — a third point on the now-isolated curve (should-do)

Same shape, `stories_short` still pinned at 0.40, instruction pushed further:

```
--mix stories_short=0.40 soda=0.03 alpaca=0.32 tinystories=0.07 persona=0.05 dolly=0.10 oasst1=0.03
```

(instruction = 0.45; sums to 1.00, re-verify by dry-run before launch.)

**Prediction P1b:** does `list_strict` continue rising roughly monotonically
across 0.23 → 0.36 → 0.45 (a real curve, the first this project has had on
this axis), or plateau/reverse — mirroring how the window-coverage dose lever
(`QUALITY_v4.md`, `snn-chat-window-dose-ceiling`) was closed after a 3.4×
step moved topicality backwards? Either outcome closes the question either as
"push further" or "this lever is also exhausted"; a plateau is not a null
result for this arm, it is the answer.

### Deliberately not attempted this session

`bot_loss_weight`/`align_frac` isolation (always varied together since `v3b`)
and the two named `story_dodge` suspects (ending-less truncation;
"short"/"little" phrasing bias) each need their own resample to avoid
inheriting the 48-draw unresolvability problem `CONVENTIONS.md` exists to fix
— pricing them above a plain 40-minute arm. Named here as next-round
candidates rather than silently dropped from this round's scope.

## §5. A training-seed replicate — NOT a threshold test

**Not proposed as a way to ship `chat-v6-inst-a`.** With remaining budget
after §4, the question worth spending it on is not "can a resample resolve
inst-a vs. the incumbent" — a power calculation says that needs ~180 seeds per
arm (~108 min each, ~3.6 h for the pair, on top of build/debug/write-up time
this session does not have) and would land the result right on the resolution
boundary, where regression to the mean on a gap measured from 64/20 draws is
more likely than not to erase it. **No component of this document, and no
number `snnchat` has ever produced, has measured training-seed variance** —
`docs/chat/CONVENTIONS.md`'s intervals are all over the *sampler*, at `n = 1`
per arm, a limitation §3 of that document states plainly. Every arm-ranking
claim in `QUALITY.md` through this document rests on one training run per
arm. That is the actual gap, and it is answerable in one arm's cost.

**Recipe: `chat-v6-inst-a-s1`.** Identical to `chat-v6-inst-a` in every
field — same mix, same `init-from`, same `bot_loss_weight`/`align_frac`/
`align_lookahead`, same step count and budget — except `--seed 1` instead of
the default `0`. Since both runs initialise from the same `chat-v2-anneal`
checkpoint, the only thing a different seed changes is which batches the
14,000-step fine-tune sees, in what order.

**No threshold is attached. This is a variance measurement, not a ship
test.** Read at the same pinned `n=1, λ=0` row, report `chat-v6-inst-a-s1`'s
headline and components next to `chat-v6-inst-a`'s. If the two seeds land
close (near 0.31), `chat-v6-inst-a`'s point estimate gains some credibility as
a real effect worth a properly-powered follow-up. If they land far apart
(comparable to the spread already seen across this project's *different*
arms, e.g. 0.25–0.31), that is a materially bigger finding than any single
arm comparison: it would mean five rounds of `QUALITY*.md` arm rankings have
been read at a resolution finer than run-to-run noise supports, and every
comparison in this document (and its predecessors) needs that caveat
attached, not just this one.
