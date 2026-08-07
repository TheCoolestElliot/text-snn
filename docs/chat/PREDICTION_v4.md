# What the 2026-08-06 responsiveness round predicts, written before it trains

This file exists so that the next round's result can be scored against a claim
made in advance rather than explained afterwards. `docs/chat/QUALITY.md` §4.1 did
the same thing and §4.2 scored it honestly, including the part that failed. This
is the same discipline applied to the arms below.

**Not part of the research protocol.** No control arms in the protocol's sense,
n = 1 per arm, and nothing here is a reported figure.

---

## 1. The mechanism, and a metric that measures it properly

QUALITY.md §4.3 ordered four arms by "the share of training windows carrying the
request" and the ordering held across 4.6 % → 27 % → 76 % with topic propensity
going 0.0518 → 0.0752 → 0.0908 and not flattening. §6 item 1 concluded: *more of
the dose that worked*.

That series was measured with a quantity that undercounts in one direction and
overcounts in another, and `scripts/chat/window_coverage.py` now computes all
three so they can be told apart:

* **strict** — P(a window *begins* inside the request). This is §4.1's 4.6 %.
  It excludes every window that starts in the previous conversation and rolls
  forward into this one, which is a real and ordinary window that does carry the
  dependency.
* **carries** — P(a window holds the whole request and at least one character of
  its reply). The corrected version of the same idea. On `stories_topic` it is
  **0.248**, not 0.046.
* **dose** — the share of *reply characters* whose gradient is conditioned on
  their own request. This is the new one and it is the one that matters, because
  a window holding a request plus the first 215 characters of a 726-character
  reply delivers the dependency for 215 characters and not for the other 511.

Measured before any arm trained, at `seq_len = 256` and the realised alignment
rate of the shipped arm (0.74):

| source | conversation | reply | strict | carries | dose @ unaligned | dose @ aligned | dose @ 0.74 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `stories_topic` (shipped arm's source) | 757 | 726 | 0.039 | 0.248 | 0.132 | 0.304 | **0.260** |
| `stories_short` (new) | 241 | 161 | 0.139 | 0.840 | 0.531 | 0.998 | **0.877** |

Two things in that table, and the second is the one QUALITY.md §6 item 2 asked
for:

1. The dose goes up **3.4×**, which is a larger step than any single step in the
   series that produced the 0.0518 → 0.0908 movement.
2. `stories_short` **unaligned** (0.531) delivers nearly twice the dose of
   `stories_topic` **aligned** (0.304). Coverage has become a property of the
   corpus instead of a property of a sampler flag, which is exactly what §6 item
   2 said would remove the trade in §3.2 — that aligning every window means never
   training on a story's tail. A 161-character reply has no tail to miss.

`soda`'s strict figure exceeds 1.0 in that table because it is multi-turn and the
metric counts one request prefix per reply; it is meaningful only for the
single-turn story sources, which is all it is used for here.

## 2. The arms

Both start from `chat-v2-anneal/ckpt_best.pt`, both run 14,000 steps at
B = 160, L = 256, lr 5e-4 cosine to a 5 % floor, `bot_loss_weight 3.0`,
`align_frac 0.75`, `align_lookahead 1024`. That is **exactly** the recipe of
`chat-v3d-aligned`, the shipped arm, so `chat-v3d-aligned` is a matched-length
baseline for both and the only variable is the mixture.

| arm | mixture | what it tests |
| --- | --- | --- |
| `chat-v3d-aligned` (shipped) | `stories_topic` 0.40, soda 0.20, alpaca 0.15, tinystories 0.09, persona 0.08, dolly 0.06, oasst1 0.02 | — |
| `chat-v4a-short` | the same, with **`stories_short` 0.40 in place of `stories_topic` 0.40** | short conversations alone. One variable. |
| `chat-v4b-balance` | `stories_short` 0.26, soda 0.21, **alpaca 0.26**, tinystories 0.08, persona 0.09, **dolly 0.08**, oasst1 0.02 | QUALITY.md §6 item 3: is the 40 % story weight what cost instruction following? |

`chat-v4b-balance` cuts total story weight from 0.49 to 0.34 and raises
instruction sources from 0.23 to 0.36. It confounds two changes deliberately —
there is not GPU budget for a four-arm ladder — so a positive result on it is
attributable to the pair, exactly as QUALITY.md §3.2 says of `bot_loss_weight`
and `align_frac`.

## 3. The predictions

Written 2026-08-06, before either arm existed.

**P1 — topic propensity.** `chat-v4a-short` scores above `chat-v3d-aligned`'s
**0.0908 ± 0.0180** on the ~1,024-candidate propensity column. This is the direct
test of "more of the dose that worked": the dose is 3.4× and the series it is
extrapolating from was monotone and had not flattened. If it lands inside the
error bar of 0.0908 the honest reading is that the dose–response relation has
flattened, and the remaining explanation for the 9 % ceiling is model capacity
rather than data presentation.

**P2 — prompt dependence.** `chat-v4a-short` scores above **+0.0619 bpc**. This
one has no sampler, no lexicon and no probe set in it, so it is the prediction to
weight most heavily. It is also the one most at risk from a confound: the
mixture's *held-out splits* change with the mixture, and the shorter conversations
in `stories_short` have a request-to-reply ratio five times higher than
`stories_topic`, which mechanically gives a request more to explain. **The
per-source breakdown must be read, not just the total**, and the comparison that
means something is on `soda`, `alpaca` and `persona` — sources both arms saw
identically.

**P3 — instruction following.** `chat-v4b-balance` beats **0.000** on strict
`list`, i.e. recovers some of the regression QUALITY.md §6 records as shipped.
`chat-v4a-short` does not, or does so only by accident, because it changes
nothing about the story weight.

**P4 — reply length, and it is a risk not a hope.** Both arms will produce
shorter replies, because 26–40 % of their mixture now ends a story after ~161
characters. That is wanted for chat and it is *also* the most likely way this
round does damage: if it shortens replies to prompts that are not story requests,
`list (strict)` improves for the wrong reason (its 120-character cut-off) and
`mean_chars` will show it. **`mean_chars` and `story_dodge` are to be read next
to any `list` gain**, and a `list` gain that arrives with a collapse in
`mean_chars` is a measurement artefact and is to be reported as one.

**P5 — what will not move.** `fact` stays at ~0. Cross-turn memory stays absent.
Nothing in a mixture change touches either.

## 4. How it will be scored

`python scripts/chat/quality.py --ckpt <arm>/ckpt_best.pt --out
experiments/chat/_quality/<arm>.json`, identical settings to every arm already in
`experiments/chat/_quality/`, then `compare_quality.py` over all of them. The
probe battery is frozen and was frozen before this round: it is not extended,
reworded or reweighted here. The `topics` stop list is not touched either, so the
topic vocabulary the corpus can produce is the same one the shipped arm had.
