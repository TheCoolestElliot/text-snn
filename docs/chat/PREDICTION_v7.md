# What the 2026-08-07 session-7 round predicts, written before it trains

This file exists so results can be scored against a claim made in advance
rather than explained afterwards, same discipline as `PREDICTION_v4.md`
through `PREDICTION_v6.md`. **Not part of the research protocol.** No control
arms in the protocol's sense, nothing here is a reported figure, and this
round does not touch `snn/`, `experiments/runs/`, `experiments/logs/` or
`docs/reports/`.

**Session context.** Elliot asked for an unattended ~8-hour session that
trains on all the data the project has. The chat corpus (`data/chat/`, 8
sources, ~2.04 G training characters, vocab_version 1, 101 symbols) is
already used in full by every existing recipe bar one weight (`oasst1`
deliberately held at 0.02 to avoid memorising 796 examples — see
`build_corpus.py`'s comment). The research corpora (`data/enwik8`,
`data/text8`, 205-symbol vocabulary, a different tokenizer entirely) cannot
be mixed into a chat run — `docs/chat/README.md` §3 "Continuation mode" is
the existing evidence: a research checkpoint's embedding rows are indexed by
a different alphabet, and pointing the REPL at one switches modes rather
than merging vocabularies. So "train on all the data" scopes to spending
GPU-hours on the chat mixture as it already exists, not to a corpus
expansion — and the single largest unpriced gap flagged against *every*
number this package has produced is `QUALITY_v6.md` §8: **`n = 1` training
seed per arm, everywhere, until this project's one same-recipe reroll
(`chat-v6-inst-a` vs. `chat-v6-inst-a-s1`) swung the headline by 0.099 —
larger than every arm-vs-arm difference session 6 measured.** That is what
this round spends its budget on.

---

## §1. Hypothesis

Two recipes currently disagree about which is better, at unusable `n`:

| recipe | seeds run | headline(s) |
| --- | --- | --- |
| `chat-v3d-aligned` (incumbent, `SHIPPED`) | 0 only | 0.2969 |
| `chat-v6-inst-a` (best-supported challenger, `QUALITY_v6.md` §4) | 0, 1 | 0.3101, 0.2109 |

`QUALITY_v6.md` §8 already states the consequence plainly: *"`chat-v6-inst-a`'s
0.3101 does not, on its own, support ranking it above `chat-v3d-aligned`'s
0.2969"* — a same-recipe reroll alone can move the headline by more than that
gap. Nothing in this project has ever fixed a training-seed count in advance
and honoured it for a recipe-vs-recipe comparison; §8 was a variance
measurement (one pair, no threshold), not a ship test.

**Prediction P1:** raising each recipe to **n = 4 training seeds** (seeds
0–3, fixed in advance, see §4) and comparing the two sets of 4 per-seed
headlines with a two-sample Welch t-test resolves which recipe has the
higher population-mean headline, OR the comparison lands `unresolved` — a
reportable third outcome, not a failure (`CONVENTIONS.md` §1). No claim is
made in advance about which way it goes; §3 states the power honestly,
including the real chance this design cannot resolve a gap this small.

**Prediction P2 (secondary, no threshold):** the 8 total new runs (4 seeds
×2 recipes, of which 3 already exist) give this project's first actual
estimate — not a single difference — of between-training-seed standard
deviation on the headline metric. Whatever that number is gets reported and
named as provisional at this `n`, per `QUALITY_v6.md` §8's own caution not to
treat one measurement as a calibrated floor.

## §2. Recipes

Both are otherwise-identical to their `n=1`/`n=1,2` predecessors — same
`init-from`, same objective settings, same step count and budget — so the
only thing that differs within a recipe is `--seed`. Verified by
`train.py --dry-run` immediately before this file was written: both mixes
sum to 1.00, both are 4,417,637 parameters (identical architecture, unchanged
from every prior chat arm), 0.57 G characters per run (14,000 steps ×
160 × 256).

### `chat-v3d-aligned-s1`, `-s2`, `-s3` — the incumbent's recipe, seeds 1–3

```
python scripts/launch.py --script scripts/chat/train.py -- \
    --run-name chat-v3d-aligned-sN --out-dir experiments/chat \
    --init-from experiments/chat/chat-v2-anneal/ckpt_best.pt --no-resume \
    --max-steps 14000 --batch-size 160 --seq-len 256 --lr 5e-4 \
    --warmup-steps 200 --bot-loss-weight 3.0 --align-frac 0.75 \
    --align-lookahead 1024 --seed N --budget-minutes 75 \
    --mix stories_topic=0.40 soda=0.20 alpaca=0.15 tinystories=0.09 \
          persona=0.08 dolly=0.06 oasst1=0.02
```

`--warmup-steps 200` is not the `ChatConfig` default (500) — it is what
`chat-v3d-aligned/config.json` actually recorded, so the seed replicates
match the committed incumbent's recipe exactly rather than the class default.

### `chat-v6-inst-a-s2`, `-s3` — the challenger's recipe, seeds 2–3
### (`-s0`, `-s1` already exist as `chat-v6-inst-a` / `chat-v6-inst-a-s1`)

```
python scripts/launch.py --script scripts/chat/train.py -- \
    --run-name chat-v6-inst-a-sN --out-dir experiments/chat \
    --init-from experiments/chat/chat-v2-anneal/ckpt_best.pt --no-resume \
    --max-steps 14000 --batch-size 160 --seq-len 256 --lr 5e-4 \
    --bot-loss-weight 3.0 --align-frac 0.75 --align-lookahead 1024 \
    --seed N --budget-minutes 75 \
    --mix stories_short=0.40 soda=0.07 alpaca=0.26 tinystories=0.09 \
          persona=0.08 dolly=0.08 oasst1=0.02
```

`--budget-minutes 75` is a safety ceiling, not the expected duration —
`chat-v6-inst-a`, `-a-s1` and `-b` each finished 14,000 steps in ~40–42
real minutes; 75 covers the trainer's own rollback+seed-bump+halved-LR
recovery (`chat-v6-inst-a-s1` hit this once and still finished on time)
with room to spare.

All five runs go through `scripts/chat/session7_driver.py`, which is
restartable: it skips a run whose `summary.json` already shows the full
14,000 steps and deletes (not resumes) a partial directory, matching the
`snn-v2-research-protocol` rule for long unattended runs applied here for
the same reason.

## §3. Decision rule, and its power stated before the run

Each checkpoint is scored once, at the standard battery
(`quality.py` defaults: `--seeds 4 --n 16`, the same battery every prior
arm in this document's lineage used), read at the pinned `n=1, λ=0` row
(`report["baseline_picks"]`) — this keeps every new checkpoint comparable to
every existing one. That gives **4 independent per-seed headline values per
recipe** (not 4× more draws on one checkpoint — 4 separately *trained*
checkpoints, each scored once).

**The test:** Welch's two-sample t-test (unequal-variance, appropriate for
small unequal-confidence samples) on `{chat-v3d-aligned, -s1, -s2, -s3}`'s
four headlines vs. `{chat-v6-inst-a, -s1, -s2, -s3}`'s four headlines.
α = 0.05, two-tailed. Same test repeated per component (`topic`, `list`,
`social`, `fallback`) for a decomposition, same as `QUALITY_v6.md` §8's
table but with a legitimate multi-seed SD instead of a single propagated
sampler SE.

- **Resolved, challenger's favour:** `chat-v6-inst-a`'s mean headline is
  higher and `p < 0.05`. `SHIPPED` is updated to whichever seed of
  `chat-v6-inst-a` scored `ckpt_best` closest to that mean, named explicitly
  as "selected by proximity to the resolved group mean," not "the best of
  four" — picking the single highest of four correlated draws would be the
  peeking problem §4 exists to prevent, applied at selection time instead of
  stopping time.
- **Resolved, incumbent's favour:** `SHIPPED` unchanged; recorded as a real
  finding, not a null — it would mean the challenger's session-6 point
  estimate (0.3101) was the high draw of a recipe that is not actually
  better.
- **Unresolved:** `SHIPPED` unchanged, per every prior round's rule that an
  unresolved margin does not displace an incumbent. Reported as unresolved,
  not as a quiet win for the incumbent.

**Power, stated now, not after — and it is not good.** No calibrated
between-seed SD exists yet; the only evidence is `chat-v6-inst-a`'s own
seed-0-vs-seed-1 headline gap, 0.099, from a single pair (`QUALITY_v6.md`
§8 explicitly warns against reading that as a floor). Treating it as a rough
SD proxy anyway (`SD_diff ≈ 0.099` for two independent draws implies
`SD_seed ≈ 0.099/√2 ≈ 0.070`, itself only a provisional guess, not a
measurement): a two-sample Welch test at n = 4 per group, α = 0.05
two-tailed, 80 % power, resolves a true mean gap of roughly **0.14 or
larger** (Cohen's d ≈ 2.0 at this n). The gap actually in play — 0.2969 vs.
a two-seed mean of 0.2605 — is smaller than that. **The honest expectation
is that P1 lands `unresolved`, and that is still the correct thing to spend
this budget on**: it is this project's first real estimate of `SD_seed`
rather than a guess extrapolated from one pair, every future round inherits
that number, and `CONVENTIONS.md` §1's third verdict exists precisely so an
underpowered-but-honest result is reported as itself rather than forced to a
side.

## §4. Fixed sample size, stated and honoured

**n = 4 seeds per recipe, decided now, run regardless of interim results.**
The driver runs `chat-v3d-aligned-{s1,s2,s3}` and `chat-v6-inst-a-{s2,s3}` —
5 new training runs, ~0.75–1.25 GPU-hours ceiling each (§2), plus quality
scoring (~2 minutes each, `chat-v6-inst-a-s1`'s `_quality/*.json` recorded
120.9 s at these same battery settings) — against an ~8 GPU-hour session
budget, leaving slack for setup, write-up and this file. If wall-clock forces
a stop before all 5 finish, that is reported as **insufficient n**, not as
whatever partial comparison happens to be sitting on disk at the time — an
`n < 4` group is not silently read as if it were the planned one.

## §5. Deliberately not attempted this round

- **`spread_slow_poles` A/B** — stays research-side-only per `README.md`'s
  own boundary, as every prior round has left it.
- **The two named `story_dodge` suspects** and **the `bot_loss_weight`/
  `align_frac` isolation** — each needs its own resample budget, named as
  next-round candidates in `QUALITY_v6.md` §7 and left there. If GPU-hours
  remain after §1–§4 close out, a candidate here gets its own
  `PREDICTION_v7.md` §6 appended at that time (not backdated), the same
  discipline `PREDICTION_v6.md` §4 used for its own mid-session addition.
- **A longer anneal budget (> 14,000 steps)** — a genuinely new arm, not a
  seed replicate, and conflating "more steps" with "more seeds" in one
  round would make neither measurement clean. Named as a candidate for a
  future round, not attempted here.
