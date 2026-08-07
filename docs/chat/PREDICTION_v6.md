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
