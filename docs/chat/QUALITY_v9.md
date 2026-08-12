# QUALITY v9 — the corpus arm failed, and the selector had more left in it

**2026-08-12.** One trained arm (`chat-v9-narrative`, 4 seeds, 2.68 GPU-hours)
scored against the bar `PREDICTION_v9.md` fixed before it trained, plus one
zero-GPU change to the decoder. `experiments/chat/SHIPPED` is unchanged.

Not part of the Phase-1..5 research protocol. No number here is a reported
figure. Rates follow [`CONVENTIONS.md`](CONVENTIONS.md).

---

## 1. A12 `soda_narrative` — **BELOW the bar, resolved**

The round's stated bet. The corpus is TinyStories wrapped four ways, with only
8.4 M characters of non-TinyStories prose in it; `soda_narrative` adds **218.4 M
characters of adult third-person prose** at weight 0.10, taken from `soda`.
The hypothesis was that the held-out oracle sits at 0.09 because the model has
only ever read a three-year-old's vocabulary.

Primary gate, `PREDICTION_v9.md` §1 — held-out **oracle** topicality, 20 nouns
absent from `quality.PROBES`, 6 sampler seeds, n = 8, k/120:

| run | oracle | 95 % CI | weighted bpc |
| --- | ---: | --- | ---: |
| `chat-v9-narrative` | 14/120 | [0.0708, 0.1863] | 1.2133 |
| `-s1` | 13/120 | [0.0644, 0.1766] | 1.2109 |
| `-s2` | 12/120 | [0.0581, 0.1667] | 1.2126 |
| `-s3` | 14/120 | [0.0708, 0.1863] | 1.2126 |
| **4-seed mean** | **13.25/120** | pooled 53/480 [0.0854, 0.1416] | 1.2124 |
| `chat-v3d-aligned` (incumbent) | 11/120 | [0.0520, 0.1567] | 1.1743 |
| **pre-registered bar (P12)** | **22.5/120 = 0.1875** | — | — |

**Verdict: `below`.** Not `unresolved` — the pooled interval's upper bound
(0.1416) lies under the bar. `CONVENTIONS.md` §4 rule 5: a near-miss is still a
miss, and this is not near.

### The arm is not a failed run, which is what makes the result informative

Every seed trained 14,000 steps cleanly, no divergence and no rollback, and the
four seeds agree to within 0.0024 bpc. The model **learned the new source**: its
held-out `soda_narrative` bpc is 1.547, against 1.748 measured on the same
source at step 20 of the same run. So 218 M characters of adult prose were
absorbed, and the held-out oracle moved from 11/120 to 13.25/120.

**SD_seed on the oracle count is 0.96**, which is the first estimate this
package has of that quantity and is small enough that 13.25 vs 11 is probably a
real shift — but it is a shift of about two draws in a hundred and twenty
against a bar of eleven and a half.

**The register hypothesis is not supported.** Whatever makes this model unable
to draft a reply about a penguin, it is not that it has never read adult prose.

*Not comparable*: weighted bpc is 1.2124 against the incumbent's 1.1743, but the
mixture differs (0.10 of `soda` swapped out), so unlike the `chat-v6-scratch`
comparison in `BUILD_NOTES.md` §12 this one is **not** a like-for-like reading
and is reported only because the pre-registration says to report it.

## 2. The echo tier is now weighted, and it reaches the oracle

`QUALITY_v8.md` §13 left the largest unclaimed gap in the package: past n = 64
the echo partition saturated at 0.300 held-out while the oracle kept climbing to
0.450. The diagnosis there was that a big pool fills the top tier with drafts
echoing the request **frame** rather than the subject, and §2 measured it — 16 of
the 25 draws where selection changed had no topic hit under either rule, decided
by `make`, `tell`, `story`, `want`.

Counting *distinct* echoed words treats "story" and "penguin" as equal. Weighting
each by inverse document frequency does not, and the separation is a property of
the corpus rather than a choice:

```
tell 5.93   story 5.20   make 6.02   want 6.27   little 5.05
penguin 10.34   castle 8.86   turtle 9.13   whale 9.70   lighthouse 15.20
```

No threshold is needed; the sum is simply dominated by the rare word. Counts come
from 4,004,093 words of `stories_topic.bin`, shipped as `src/snnchat/word_freq.json`
so a fresh clone (where `data/` is absent) still decodes, falling back to the
unweighted count tier if the file is missing.

Replayed over the stored pools, and then confirmed end-to-end through the real
`rerank` path at n = 32:

| n | shipped score | echo (distinct count) | **echo (IDF-weighted)** | oracle |
| ---: | ---: | ---: | ---: | ---: |
| 8 | 0.0083 | 0.0833 | **0.0917** | 0.0917 |
| 32 | 0.0167 | 0.2000 | **0.2500** | 0.2500 |
| 64 | 0.0250 | 0.3083 | **0.3667** | 0.3667 |
| 128 | 0.0167 | 0.3250 | **0.4167** | 0.4500 |

**It attains the oracle exactly at n = 8, 32 and 64.** The live confirmation at
n = 32 gives 0.2500 against the shipped score's 0.0167, 28 discordant draws
better and 0 worse, exact McNemar p < 0.00001.

### One caveat, stated because it matters

This is the **second** selection rule chosen using the same 120 held-out draws.
Those draws are therefore no longer fully held out *for this rule* — the noun
set was fixed in committed code before v8, but a rule picked by comparing two
candidates on an evaluation set has used that set for model selection. A fresh
noun list is the clean test and has not been run.

What limits the concern: the rule is derived from corpus word frequencies rather
than from the probes' answer key, it is a one-line change with no tuned
constant, and it does not merely improve the metric — it reaches the **ceiling**
of what the drawn pool contains, which is a bound this rule cannot exceed by
overfitting.

## 3. What the two results say together

The pool was the constraint, then the selector was, and now the selector is at
the pool's ceiling for n ≤ 64. So the pool is the constraint again — and A12
says the obvious way to widen it does not work.

Held-out topicality at the shipped default is now **0.2500**, from 0.0083 when
this line of work started. Every bit of that is decode-time; no checkpoint has
changed since 2026-08-06.

## 4. Reproducing

```bash
python scripts/chat/session9_driver.py --only A12    # 2.68 GPU-h, 4 seeds
python scripts/chat/score_v9.py --arm chat-v9-narrative --seeds 4
python scripts/chat/echo_holdout.py --seeds 6 --n 32
```

Artifacts: `experiments/chat/_quality/score_v9_chat-v9-narrative.json`,
`echo_holdout_chat-v9-narrative*.json`, `echo_holdout_n32_idf.json`,
`src/snnchat/word_freq.json`.
