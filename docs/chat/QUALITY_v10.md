# QUALITY v10 — the decoder was 97 % dispatch, and the pool it now affords

**2026-08-12.** No training. Four decode-time changes, one of them a defect fix,
plus the fresh-noun validation `QUALITY_v9.md` §2 asked for and did not run.
`experiments/chat/SHIPPED` is unchanged — no checkpoint has changed since
2026-08-06.

Not part of the Phase-1..5 research protocol. No number here is a reported
figure. Rates follow [`CONVENTIONS.md`](CONVENTIONS.md).

**Summary.** A turn at the shipped settings went from 1.72 s to 0.33 s with
bit-identical output; the default pool then went 32 → 128, which costs 0.57 s
and buys held-out topicality **0.2500 → 0.4167**. On a second noun list written
this round and never used to choose anything, **0.0583 → 0.1333**. One
mechanism was built, measured, and **not shipped**.

---

## 1. The decode step was ~97 % dispatch overhead, and it is now bit-identical and 7× cheaper

`snnchat.rerank`'s whole design rests on `BUILD_NOTES.md` §1: the per-timestep
scan is latency-bound at these widths, so `N` drafts cost about what one draft
costs. That is true, it is still true, and it was never what the REPL was
paying. Measured on this box, one generated character at `d = 1024`, `K = 4`:

| | µs / character |
| --- | ---: |
| the scan, batch 1 … 512 (flat, as documented) | ~1580 |
| … of which is arithmetic (one `[B,1,1024]` GEMM) | ~50 |
| sampling arithmetic (`_filter`, softmax, multinomial) | ~690 |
| a captured CUDA graph of the same step | **~205** |

Two separate causes, both fixed:

**(a) `2n` device reads per character.** `sample_candidates` asked
`bool(just_stopped[r])` and `bool(alive[r])` for every row of the batch, and
each is a host synchronisation — **36 µs** on this box. A 300-character turn at
n = 128 spent ~2.8 s learning what it already knew, and it scaled *linearly in
the pool size*, which is precisely the axis worth spending on. The loop now
reads back one tensor per character (`nxt`) and derives liveness on the host
from the same ids and the same stop set.

**(b) The decode step was never captured.** §1's 26 µs-per-timestep-per-layer
floor was measured inside a *training* graph, where the scan loops over L = 384
timesteps in one call. Decoding calls it with L = 1 and captured nothing, so the
four jiterator launches per character had nothing to amortise against.
`snnchat.stepper` captures the whole step — embedding, projections, scans, head,
loop-break penalty, truncation, both softmaxes — and replays it per character.
Under capture the decode path lands at ~51 µs/timestep/layer, the same order as
§1's training floor, which is the check that says the remaining cost is the one
§1 already identified and not a new one.

`torch.multinomial` is deliberately **outside** the capture. Capturing the draw
means capturing RNG state, which changes which numbers a given seed produces,
and every committed artifact in `experiments/chat/_quality/` was drawn with the
generator advanced as it is advanced today.

**Both changes are bit-identical, asserted three ways.**
`tests/test_snnchat.py::test_graph_and_eager_draw_the_same_text` compares ids,
`scored` and `logp_cond` across pool sizes and seeds. The eager loop is retained
and reachable (`--no-graph`, `RerankParams.graph`). And end-to-end: the held-out
**oracle reproduces the committed 0.0917 / 0.2500 / 0.4500 exactly** at n = 8,
32 and 128 — the pool did not change, only what it cost.

One deliberate arithmetic difference, stated because "identical" should mean it:
the eager path subtracts the loop-break penalty in place, the captured path adds
a `[n, V]` bias that is zero elsewhere. `x - p` and `x + (-p)` agree exactly in
IEEE754, and `x + 0.0` returns `x` for every float except `-0.0`, which it
returns as `+0.0` — and `exp(-0.0) == exp(0.0)`, so no probability can differ.

Two smaller exact savings came with it: the null-context pass is **skipped when
the surviving tier holds one candidate**, because `score` only ever decides
between tier members (recoverable afterwards via `fill_null_scores`, which is
what `/candidates` calls); and the loop breaker got an exact early-out
(`recent[-1] == recent[-n-1]` is necessary for the block comparison) that
removes ~10⁶ list slices per turn at n = 128.

Re-measured end to end through `ChatSession.send`, median of 4 turns at
`--max-new 400`:

| n | 8 | 32 | 64 | 128 | 256 |
| --- | ---: | ---: | ---: | ---: | ---: |
| **now** s/turn | 0.32 | 0.33 | 0.40 | 0.57 | 0.67 |
| was (at max_new 300) | 1.05 | 1.72 | 2.55 | 4.21 | — |
| peak GiB | 0.22 | 0.59 | 1.04 | 1.96 | 3.78 |

## 2. A defect: the selector was ranking text the reader never sees

`trim_to_sentence` cuts an unclosed draft back to its last complete sentence,
and it runs **after** selection. The echo tier read `c.ids` — the whole draft,
including the tail about to be discarded. So a draft whose only mention of the
subject was in that tail won the tier for a word its reply does not contain.

Measured on the committed pools at n = 128: 0.7 % of drafts lose echo weight to
trimming, and tiering on the returned text instead is **+3 / −0** on the
held-out nouns and **0 / 0** on the fresh ones. Small, and one-directional by
construction rather than by luck — ranking the string you return cannot be worse
than ranking one you throw away.

**This also settles a disagreement between the code and this documentation.**
`QUALITY_v9.md` §2's n = 128 row reports **0.4167**; the shipped selector scored
**0.3917**. Both numbers were right about different selectors: v9's was replayed
off the *stored* pool text, which `echo_holdout.py` writes already trimmed, and
the code tiered on the draft. The fix makes the shipped selector score 0.4167,
so the published number is now the true one. `rerank.final_ids` is the single
function the selector, `ChatSession` and the harness all call, which is what
stops this recurring — the duplication that was *supposed* to catch it (the
harness recomputing the tier independently) could not, because both copies
scored the untrimmed draft while the artifact stored the trimmed one.

## 3. Pool steering — built, measured, **UNRESOLVED**, not shipped

The round's actual bet, and it did not pay.

**The idea.** Best-of-N spends its whole budget before it learns anything. The
sequential-Monte-Carlo alternative is to score partial drafts periodically and
copy promising ones over unpromising ones. That is unusually cheap here: a
draft's entire history is `[2d]` of membrane per layer, so cloning one is an
`index_select` on a fixed-size tensor — the same O(1)-state property the whole
package is built on, spent on search instead of on session length.

**The potential is a lookahead, and it had to be.** The obvious score — how much
of the prompt the draft has echoed so far — is useless, because the corpus puts
the subject noun a median of **81 characters** into the reply, so a draft that
has said "penguin" has already succeeded and one that has not carries no
evidence either way. So the potential is `log P(" penguin" | draft so far)`: one
teacher-forced pass over the word from each row's membrane, asking the model
whether the subject is *about to* arrive.

**It is aimed at a measured mechanism.** Phase 4 measured this neuron's memory
horizon at **47–48 characters**; the corpus's subject position is 81. The model
is asked to recall the requested noun from beyond its own reach at exactly the
moment the story form demands it. Resampling every 32 characters — inside the
horizon — is an external memory of the prompt, applied while the draft can still
act on it.

**The result.** Every comparison is paired at the (prompt, seed) level over 120
draws, McNemar per draw and a sign test over the 20 prompt-level totals:

| set | n | base | steered | per draw | per prompt | verdict |
| --- | ---: | ---: | ---: | --- | --- | --- |
| heldout | 128 | 0.3917 | 0.4500 | +12 / −5, p = 0.143 | +8 / −3, p = 0.227 | unresolved |
| fresh | 32 | 0.0583 | 0.0833 | +6 / −3, p = 0.508 | +3 / −0, p = 0.250 | unresolved |
| fresh | 128 | 0.1333 | 0.1083 | +4 / −7, p = 0.549 | +3 / −4, p = 1.000 | unresolved |

Both arms of every row were drawn **before** §2's trimming fix, so each
comparison is internally consistent and the base column here (0.3917 at heldout
n = 128) is the pre-fix selector, not §4's 0.4167. Steering was not re-measured
after the fix: it would have to clear a bar that just moved *up*, and it did not
clear the lower one.

**Unresolved everywhere, and directionally inconsistent** — up at n = 32 on the
fresh list, *down* at n = 128 on it. The summary line on the list it was first
tried on looked like a clean win (0.3917 → 0.4500) and is not one once the
comparison is actually paired; that is the whole reason `compare_holdout.py`
exists now. `CONVENTIONS.md`'s standing rule applies: an unresolved margin does
not displace an incumbent. **`steer_every = 0` is the default and off is the
identity code path**, not a steering pass that moves nothing.

One untested hypothesis for the n = 128 reversal, offered as a pointer and not a
finding: steering trades pool diversity for focus, and at 128 drafts diversity
was the asset being spent. It was not chased, and `steer_frac` was deliberately
**not** tuned against these draws.

## 4. The fresh noun list — and the IDF tier replicates on it

`QUALITY_v9.md` §2 named the outstanding weakness in this package's evidence:
the inverse-document-frequency tier is the *second* selection rule chosen by
comparing candidates on the same 120 held-out draws, so those draws had been
used for model selection. It said a fresh noun list was the clean test and had
not been run.

`echo_holdout.py --set fresh` is that list — 20 nouns disjoint from `HELDOUT`
and from every word in `quality.PROBES`, committed before it was run once.

| set | n | score only | **echo (IDF)** | oracle |
| --- | ---: | ---: | ---: | ---: |
| heldout | 8 | .0083 | **.0917** | .0917 |
| heldout | 32 | .0167 | **.2500** | .2500 |
| heldout | 128 | .0250 | **.4167** | .4500 |
| fresh | 8 | .0000 | **.0000** | .0000 |
| fresh | 32 | .0000 | **.0583** | .0583 |
| fresh | 128 | .0000 | **.1333** | .1417 |

**The tier replicates cleanly**: 0.0000 → 0.1333 at n = 128, 16 discordant draws
better and 0 worse, exact McNemar p = 0.00003, and it attains the pool's oracle
at n ≤ 32 exactly as it does on the older list. v9's caveat is discharged.

**The fresh list is much harder** — oracle 0.1417 against 0.4500 at the same
pool size. The older twenty (penguin, bear, castle, rainbow) are more
TinyStories-shaped than the new twenty (igloo, seashell, telescope, windmill).
That gap is not a defect in either list; it is the coverage diagnosis showing
through, and it means **every held-out number this package has published is on
the easy end of the range.**

**A defect in the older list, found and left in place.** `HELDOUT`'s docstring
claims every noun is absent from `quality.PROBES`. **"garden" is not** — it is
one of the accepted words of the autumn-haiku `fact` probe
(`src/snnchat/quality.py:270`). So 19 of 20 are held out. Left unchanged and
documented in place, because every committed number in `QUALITY_v8.md` and
`QUALITY_v9.md` was measured on this exact list and swapping a prompt would make
them incomparable with each other.

## 5. The default pool: 32 → 128

The reasoning that picked 32 was sound and its inputs are gone. It traded
topicality for latency under a "a REPL should answer in under two seconds" rule,
and capped at 64 because the count tier saturated while the oracle kept rising —
a gap the IDF tier closed in v9. With a turn at n = 128 now costing 0.57 s, the
same rule picks 128.

Held-out 0.2500 → 0.4167. On the fresh list 0.0583 → 0.1333, nine draws better,
none worse, and no prompt worse (McNemar p = 0.0039; the 20-prompt sign test
reads p = 0.125, which is its *floor* at 4 discordant prompts, not a
disagreement).

**Not 256**, though it measured better still (fresh 0.2000, +9 / −1): 3.78 GiB.
`ChatConfig.eval_batch_size`'s docstring records what an 8 GiB card does here
when a resident graph meets a large batch — WDDM does not fail the spill, it
makes it ~50× slower. 1.96 GiB keeps that margin. `/rerank 256` is one command
away.

## 6. What this leaves

Selection is at the pool's ceiling on both lists at n ≤ 32 and within 0.03 of it
at n = 128. The pool is the constraint, it always was, and the two things tried
against it have now both failed to resolve: 218 M characters of adult prose
(v9 §1, 2.68 GPU-h, `below` its bar) and lookahead steering (§3 here, zero GPU,
unresolved). What steering *did* buy is a decoder fast enough that the pool
question can be asked at 128 or 256 drafts instead of 8.

## 7. Reproducing

```bash
python scripts/chat/echo_holdout.py --set fresh --seeds 6 --n 128
python scripts/chat/echo_holdout.py --set fresh --seeds 6 --n 128 --steer-every 32
python scripts/chat/compare_holdout.py A.json B.json --field oracle_hit
python -m pytest tests/test_snnchat.py -q          # 107 tests
```

Artifacts: `experiments/chat/_quality/v10_{heldout,fresh}_n{8,32,128}.json`,
`echo_fresh_n*_s*.json`, `echo_holdout_n128_{base,steer32}.json`.
New code: `src/snnchat/stepper.py`, `scripts/chat/compare_holdout.py`,
`rerank.Steering` / `final_ids` / `fill_null_scores`, `echo_holdout.FRESH`.
