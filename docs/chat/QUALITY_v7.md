# Chat quality, round 7 — the first properly-powered seed-robustness read, and why it is honestly unresolved

**Date:** 2026-08-07
**Author:** Elliot Asher Caudill (session run autonomously per Elliot's request
to spend an unattended ~8-hour budget on chat-side training)
**Status: CLOSED.** P1 is **unresolved** on the headline and on every
component. `SHIPPED` is **unchanged**: `chat-v3d-aligned/ckpt_best.pt`. This
was the pre-registered, expected outcome (`PREDICTION_v7.md` §3 said so
before any of these five runs existed), not a failure to plan around.
**Predicts:** `docs/chat/PREDICTION_v7.md` (written and dry-run-verified
before any of session 7's runs launched).
**Does not modify:** `QUALITY.md` through `QUALITY_v6.md`, or
`PREDICTION_v4.md` through `PREDICTION_v6.md`, all closed. Touches nothing
under `snn/`, `experiments/runs/`, `experiments/logs/`, or `docs/reports/`.

---

## 1. What this round did

`QUALITY_v6.md` §8 measured, for the first time, that a same-recipe reroll
(`chat-v6-inst-a` vs. `chat-v6-inst-a-s1`, seeds 0 and 1) can swing the
headline by **0.099** — larger than every arm-vs-arm difference this project
has ever reported. Every ranking in `QUALITY.md` through `QUALITY_v6.md` was
made at `n = 1` training seed per arm. This round raised the two arms most in
question — the shipped incumbent (`chat-v3d-aligned`) and its best-supported
challenger (`chat-v6-inst-a`, `QUALITY_v6.md` §4) — to **n = 4 training seeds
each**, fixed in advance (`PREDICTION_v7.md` §4), and read the comparison
with a between-seed test instead of a single propagated-SE point comparison.

Five new checkpoints were trained: `chat-v3d-aligned-{s1,s2,s3}` and
`chat-v6-inst-a-{s2,s3}` (`-s0`/`-s1` already existed from sessions 6 and
earlier). Each is `scripts/chat/train.py`'s recipe for its arm, unchanged
except `--seed`, verified by `--dry-run` before this round started: both
mixes sum to 1.00, both are 4,417,637 parameters, 0.57 G characters over
14,000 steps. Every checkpoint was scored with the same `quality.py` battery
every prior round used (`--seeds 4 --n 16`, read at the pinned `n=1, λ=0`
row), so every number below is directly comparable to `QUALITY_v6.md`'s.

**One run needed a retry.** `chat-v3d-aligned-s1` exited after 7 seconds on
its first attempt (`rc=3221225786` / `0xC000013A`, `STATUS_CONTROL_C_EXIT`)
with no output reaching the driver's log — not even the pre-CUDA banner
`train.py` prints (`flush=True`) before touching the GPU. That is decent
evidence the process died before reaching that print, but not proof: a
killed process can also lose buffered output at the OS level, so this is
reported as "no output reached the log, cause unknown," not as a claim about
which line of `train.py` was executing. The driver's restart rule (skip-if-complete,
delete-if-partial, never resume) caught this cleanly: the queue moved on to
the next run automatically, and re-running the same driver afterward
retrained `chat-v3d-aligned-s1` from a cold start with the identical command
and it completed normally on the first attempt. **This was not chased to a
cause** — it did not recur on any of the other 5 attempts (including the
retry of the same recipe/seed), and it left no artifact to examine (no
partial checkpoint, no traceback). Recorded here rather than silently
retried out of the log, per this project's standing rule to name an
unexplained failure rather than launder it.

## 2. Per-seed results

Read at `n=1, λ=0` (`report["baseline_picks"]`), the same row every prior
round used. `list` is `by_kind.list` (continuous, `{0, 1/3, 2/3, 1}` per
`CONVENTIONS.md` §2, **not** `list_strict`, matching `QUALITY_v6.md` §8's
own column).

| checkpoint | topic | list | social | fallback | headline |
| --- | ---: | ---: | ---: | ---: | ---: |
| `chat-v3d-aligned` (s0) | 0.0938 | 0.1667 | 1.0000 | 0.0000 | 0.2969 |
| `chat-v3d-aligned-s1` | 0.0938 | 0.2000 | 1.0000 | 0.0268 | 0.3015 |
| `chat-v3d-aligned-s2` | 0.0938 | 0.1000 | 1.0000 | 0.0179 | 0.2733 |
| `chat-v3d-aligned-s3` | 0.1094 | 0.1833 | 1.0000 | 0.0268 | 0.3043 |
| `chat-v6-inst-a` (s0) | 0.0938 | 0.2167 | 1.0000 | 0.0089 | 0.3101 |
| `chat-v6-inst-a-s1` | 0.0625 | 0.0500 | 0.8500 | 0.0268 | 0.2109 |
| `chat-v6-inst-a-s2` | 0.0938 | 0.1500 | 0.9500 | 0.0000 | 0.2819 |
| `chat-v6-inst-a-s3` | 0.0469 | 0.1833 | 1.0000 | 0.0179 | 0.2749 |

## 3. The pooled comparison, per `PREDICTION_v7.md` §3's pre-registered rule

Welch's two-sample t-test (unequal-variance), n = 4 per recipe, α = 0.05
two-tailed, computed by `scripts/chat/session7_stats.py` from the committed
`_quality/*.json` files (never hand-transcribed — `CONVENTIONS.md`'s rule
applied to this round's own numbers) using the standard incomplete-beta-
function identity for the Student-t survival function; cross-checked against
the `df ≈ 3–6` critical-value table for 0.05 two-tailed, 2.45–3.18, and every
`|t|` below falls short of its own row's critical value, agreeing with the
computed `p`.

| metric | `v3d-aligned` mean (sd) | `v6-inst-a` mean (sd) | diff (insta − v3d) | SE | t | df | p | verdict |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :-- |
| **headline** | **0.2940 (0.0141)** | **0.2694 (0.0419)** | **−0.0246** | 0.0221 | 1.111 | 3.67 | 0.334 | **unresolved** |
| topic | 0.0977 (0.0078) | 0.0742 (0.0235) | −0.0234 | 0.0124 | 1.897 | 3.66 | 0.137 | unresolved |
| list | 0.1625 (0.0438) | 0.1500 (0.0720) | −0.0125 | 0.0422 | 0.297 | 4.95 | 0.779 | unresolved |
| social | 1.0000 (0.0000) | 0.9500 (0.0707) | −0.0500 | 0.0354 | 1.414 | 3.00 | 0.252 | unresolved |
| fallback | 0.0179 (0.0126) | 0.0134 (0.0115) | −0.0045 | 0.0086 | 0.523 | 5.95 | 0.620 | unresolved |

**Every metric is unresolved.** None of the five differences clears its own
row's 0.05 two-tailed bar, headline included. This is the outcome
`PREDICTION_v7.md` §3 predicted before any of these numbers existed: the
provisional power calculation there (treating the one known 0.099 seed-swing
as a rough `SD_seed` proxy) said this design resolves gaps of roughly 0.14
or larger, and the pooled point estimate — 0.2940 vs. 0.2694, a 0.0246 gap —
is well inside that. **Per `PREDICTION_v7.md` §3's decision rule, an
unresolved margin does not displace the incumbent: `SHIPPED` is unchanged.**

**A secondary, unregistered observation, reported because §5 named this
exact gap as worth closing and it is now measurable: the two recipes'
between-seed standard deviations are not the same size.**
`chat-v3d-aligned`'s four seeds land within 0.0141 of each other on the
headline; `chat-v6-inst-a`'s four span 0.0419 — **about 3× wider**, driven
almost entirely by `-s1`'s divergence-recovery run (§1; it hit the same
gradient-norm rollback `BUILD_NOTES.md` §9 and `QUALITY_v6.md` §8 both
flagged, dropping every component at once). Two data points cannot establish
which recipe is *actually* more seed-stable — `n = 4` estimates its own
standard deviation extremely noisily — but it is consistent with, and not
contradicted by, the one thing already suspected: a run that hits the
rollback path scores worse across the board (`QUALITY_v6.md` §8's "bonus,
not scored" note), and `chat-v6-inst-a`'s badly-scoring seed is exactly the
one that hit it. **Not scored, not a verdict, named as a lead for whichever
round next has budget to look at rollback-triggering runs specifically.**

**What this round actually bought, stated plainly since the headline
comparison didn't resolve:** this project's first real measurement of
`SD_seed` for two recipes, replacing the single-pair guess
(`0.099/√2 ≈ 0.070`) `PREDICTION_v7.md` §3 used as a placeholder. Both
measured values (0.0141, 0.0419) are **smaller** than that placeholder, and
both are themselves provisional at `n = 4` — a sample standard deviation
computed from four points has wide error bars of its own. The 0.099 pair
that started this investigation is not contradicted: it sits at roughly
`0.099 / (0.0419·√2) ≈ 1.67` standard deviations of `chat-v6-inst-a`'s own
now-measured spread, an unremarkable draw rather than an outlier needing its
own explanation.

## 4. What this changes and what it does not

- **`SHIPPED` is unchanged**, per the pre-registered rule, and per five
  prior rounds' consistent treatment of `unresolved`.
- **Every existing ranking in `QUALITY.md` through `QUALITY_v6.md` keeps the
  caveat `QUALITY_v6.md` §8 attached, now with a number instead of a
  placeholder**: gaps smaller than roughly 0.10–0.14 on this headline are
  not resolvable at the training-seed replication depths this project has
  actually run, for either recipe measured so far.
- **This does not mean `chat-v6-inst-a` is a worse recipe than the
  incumbent — the comparison did not resolve in either direction.** Reading
  the unresolved verdict as a quiet loss for the challenger would repeat the
  exact mistake `CONVENTIONS.md` was written to prevent.
- **The apparent seed-stability difference (§3) is a lead, not a finding.**
  If a future round has budget for more seeds of either recipe, this is the
  first thing worth re-checking, alongside chasing what actually triggers
  the gradient-norm rollback (`BUILD_NOTES.md` §9, `QUALITY_v6.md` §8,
  unchased in both places).
- **The `chat-v3d-aligned-s1` cold-start failure (§1) remains unexplained.**
  It did not block this round because the retry succeeded and reproduced
  cleanly, but it is not established as a one-off either.

## 5. Session-7 scope, for the record

Per `PREDICTION_v7.md` §5, the two next-round candidates named in
`QUALITY_v6.md` §7 (the `story_dodge` truncation/phrasing suspects, and the
`bot_loss_weight`/`align_frac` isolation) were **not** attempted this round
— each needs its own resample budget and its own pre-registration, and
`PREDICTION_v7.md` §4's fixed-`n` commitment was honoured rather than
redirecting mid-campaign toward a result that looked more decisive. Total
GPU time this round: 5 trained checkpoints × ~40–75 minutes each, plus five
~2-minute quality batteries — well under the ~8-hour session budget, leaving
slack that is spent on write-up and memory rather than a rushed sixth arm.
