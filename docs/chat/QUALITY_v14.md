# QUALITY v14 — the subject tier fails on both checkpoints, and the default stays OFF

**Scored 2026-09-21 against the bars of [`PREDICTION_v14.md`](PREDICTION_v14.md)
as amended in its §12 before any seed 6–11 pool existed.** That file hashes to
`83326a7cd92190f58e17d80c2395b57b491094df5b47d8d49ccc09bdd22ae0f1` (sha256, CRLF
normalised to LF) on the committed blob at `2fa59fe` and on the working copy;
`scripts/chat/score_v14.py` refuses to score against any other text, and its
§11 is empty, as §12.3 says it stays. Results are here and nowhere else.

Not part of the Phase-1..5 research protocol. No number here is a reported
figure. Rates follow [`CONVENTIONS.md`](CONVENTIONS.md): every rate carries its
numerator, denominator and a 95 % Wilson interval, and `unresolved` is a
verdict. **No research decision is ruled**; decision #4 is open and is Elliot's.
`experiments/chat/SHIPPED` is untouched by this round. Nothing was trained.

Every number below was regenerated for this document from the two committed
confirmatory artifacts and the ten committed pools, by re-running the scorer
into a scratch directory and diffing (Appendix): **zero differing leaves outside
the `command` field on both checkpoints.**

---

## 0. The verdict, as it fired

**`chat-v6-scratch` — the deciding checkpoint under §12.7 — OVERALL `fail`.**
P1 pass (anchored topic: shipped-only 1, subject-only 2), P2 pass (+0.0670
nats/char, 95 % lower bound +0.0575, positive on all three lists), **P3 fail**
(over the 111 of 600 draws where both picks contain a definite subject: fixed 2,
broken 2, McNemar p = 1; the bar needs fixed > broken), GUARD 1 pass (108/108
subject-less draws identical), GUARD 2 reported (distinct-2 fell by 0.1815 of its
shipped value; the withdrawn 0.10 bar would have read fail).
**`chat-v3d-aligned` — reported beside, decides nothing — OVERALL `fail`** on
the same bar for the same reason: P1 pass (0 v 0), P2 pass (+0.0841, lower bound
+0.0763), **P3 fail** (81 both-exposed draws; fixed 2, broken 2, p = 1), GUARD 1
pass, GUARD 2 reported (fell 0.2488; would have read fail). The two checkpoints
do not disagree: every bar reads the same verdict on both.

**For the owner.** Under §12.7 only `pass` on `chat-v6-scratch` flips the
default. It fired `fail`, so **`--subject-tier` stays OFF**, in `scripts/chat.py`
and in the `/subject` toggle (`a611d99`), whose help text says the rule is not
confirmed. What the rule measurably does on the model he talks to, over 600
first-turn story requests: it chooses a reply the model finds likelier on 359 of
600 draws (+0.0670 nats/char), it cuts replies below plain sampling's floor from
237/600 to 107/600, it loses no anchored-topic hit the shipped rule had beyond
one draw against two the other way, and it makes replies look more alike
(distinct-2 0.2201 → 0.1801). What it does not do: change how the model's text
introduces the things it names. On the draws where both picks have a definite
subject, the unintroduced-entity rate is 95/111 under both rules. The rule
picks likelier text; it does not pick better-introduced text, and P3 was the
bar written to tell those apart.

## 1. What ran

| | value |
| --- | --- |
| design | §2 as amended by §12.5: per checkpoint, `HELDOUT` 20 + `FRESH` 20 + `WIDE` 60 story prompts = 100 prompts × 6 sampler seeds = **600 story draws**; `dodge` 12 prompts = 72 draws; `clause` 12 prompts = 72 draws; **744 draws per checkpoint, 1,488 in all**. The scorer's `design` block reads `missing: 0, not_in_the_design: 0, is_the_registered_design: true` on both |
| sampler seeds | 6, 7, 8, 9, 10, 11 (`sampler_seeds` in every pool header; `seed_offset` 6) |
| candidates, λ, min_chars | n = 256, λ = 0.6, `min_chars` 12 (every header); 190,464 candidates per checkpoint, every one with a measured null term |
| drafted characters | `max_new` 400 = `repl_max_new` 400 (every header): the REPL's value, not the 300 the exploratory pools were drawn at |
| null, trim | `null = empty_user`, `trim_to_sentence = True`; stored text is the trimmed, stripped reply the reader would see |
| checkpoints | `chat-v3d-aligned/ckpt_best.pt`, sha256 `3deca3eb0265925e2471b5cb3ace4c800e5dc1f116df037a2699af8ba49ac86d`; `chat-v6-scratch/ckpt_best.pt`, sha256 `e270b3a3c878b9253f92cb8d22f69f1e27f11ea2cc81b3e0854426929c62676f`. Both from the owner's main tree; `ckpt_sha256` in all ten headers matches `score_v14.REGISTERED_CHECKPOINTS` |
| pool writer / scorer | `scripts/chat/v14_pools.py` and `scripts/chat/score_v14.py` at `c907ddc`; `v14_pools.py`, `src/snnchat/` and the pre-registration are byte-identical to the stamp commit `c323e53` |
| stored-field checks | `stored_echo_weight_mismatch 0`, `stored_subject_hit_mismatch 0` on both; the replay landed on both recorded indices on all 1,488 draws (a `ReplayMismatch` would have stopped the scorer) |
| deciding checkpoint | `head -1 "C:/Elliot's Stuff/SNN/Text SNN/experiments/chat/SHIPPED"` in the owner's main tree prints `chat-v6-scratch/ckpt_best.pt   # 2026-09-21, the owner's ruling; see docs/chat/V6_SCRATCH_NOTE.md`. This worktree's own copy still names `chat-v3d-aligned` and is not the one §12.7 reads |
| scored | between the tenth pool's write (20:31:06) and the commit that added both artifacts (`2fa59fe`, 20:32:50 EDT); stamp check passed; once per checkpoint, all five pools together, as §12.9 requires |

**The chain.** Ten pools, one at a time, each detached through `scripts/launch.py`
from this worktree at `c907ddc`, `chat-v3d-aligned` first then `chat-v6-scratch`,
`clause`, `heldout`, `fresh`, `wide`, `dodge` within each. Launch time is the
`=== launch` line of each run's `stdout.log`; completion is the pool file's
mtime. No `stdout.log` contains a `Traceback`; each contains exactly one launch.

| pool | launched | written | wall-clock |
| --- | --- | --- | ---: |
| `clause` `chat-v3d-aligned` | 20:05:45 | 20:06:58 | 1 m 13 s |
| `heldout` `chat-v3d-aligned` | 20:07:02 | 20:09:05 | 2 m 03 s |
| `fresh` `chat-v3d-aligned` | 20:09:10 | 20:11:14 | 2 m 04 s |
| `wide` `chat-v3d-aligned` | 20:11:18 | 20:17:13 | 5 m 55 s |
| `dodge` `chat-v3d-aligned` | 20:17:15 | 20:18:20 | 1 m 05 s |
| `clause` `chat-v6-scratch` | 20:18:22 | 20:19:34 | 1 m 12 s |
| `heldout` `chat-v6-scratch` | 20:19:39 | 20:21:44 | 2 m 05 s |
| `fresh` `chat-v6-scratch` | 20:21:47 | 20:23:52 | 2 m 05 s |
| `wide` `chat-v6-scratch` | 20:23:55 | 20:29:47 | 5 m 52 s |
| `dodge` `chat-v6-scratch` | 20:30:01 | 20:31:06 | 1 m 05 s |

First launch to last write: **25 m 21 s**; 12 m 35 s for `chat-v3d-aligned`'s
five and 12 m 44 s for `chat-v6-scratch`'s, against §12.9's estimate of "about a
quarter of an hour per checkpoint". **The chain aborted once**, at 20:29:47 —
the instant the ninth pool (`wide` `chat-v6-scratch`) wrote its file — on a race
with that pool's exiting process, before the tenth pool had been launched. The
ninth pool's file was complete (the scorer replays all 360 of its draws and
finds `wrote` in its log), so nothing was redrawn; the chain was re-run and
launched the tenth pool at 20:30:01 at the same seeds. Every pool was launched
only after the writer had verified nothing else was on the GPU.

**Shared host during the first nine pools.** A CPU-only write-up workflow for
another round was running on the same machine while pools one to nine were
drawn; it did not touch the GPU, but the decode loop reads back one tensor per
character and is paced by the host (`QUALITY_v10.md` §1), so **every latency
figure in §3 is an upper bound**. The tenth pool, drawn alone, reads 0.1110 s / 0.1096 s a turn against
its twin's 0.1103 s / 0.1093 s drawn under the load, so the load is not visible
at this resolution; the caveat stands regardless.

## 2. The bars, per checkpoint

Verdicts exactly as `score_v14.py` fired them, with §12.6's expectation beside
each. "Expected" is what the pre-registration wrote before the run, not what
happened.

### 2.1 `chat-v6-scratch` (decides the default)

| bar | measured | fired | §12.6 expected |
| --- | --- | --- | --- |
| **P1** anchored topic | shipped 0.0267 (16/600), CI [0.0165, 0.0429]; subject 0.0283 (17/600), CI [0.0178, 0.0449]; **shipped-only 1, subject-only 2**, McNemar p = 1; mean reply 378.2 → 378.6 chars | **pass** (1 <= 2) | pass by tie, decided by 0–2 draws |
| **P2** logP/char | shipped −0.3864, subject −0.3195, **Δ +0.0670**, n = 600, SE paired 0.0030, SE by prompt 0.0048 over 100 prompts, **95 % lower bound +0.0575**; `fresh` +0.0835 (n = 120), `heldout` +0.0467 (120), `wide` +0.0682 (360) | **pass** | pass narrowly, `unresolved` live; §12.6 put the pass line at about +0.062 |
| **P3** UER@200, both exposed | both qualify 600/600; both have a definite subject **0.1850 (111/600)**, CI [0.1560, 0.2180]; flagged shipped 0.8559 (95/111), CI [0.7786, 0.9093]; subject 0.8559 (95/111), the same; **fixed 2, broken 2**, McNemar p = 1 | **fail** (2 <= 2) | **fail**, "as it reads on all three subsets already seen" (0 v 3, 1 v 3, 2 v 3); `unresolved` the other live outcome |
| **GUARD 1** identity | identical index and text on **108/108** subject-less draws (72 dodge + 36 clause); 0 dodge draws with a subject | **pass** | pass |
| **GUARD 2** sameness | distinct-2 0.2201 (10409/47297) → 0.1801 (8479/47069), **fell 0.1815**; `fresh` 0.1965, `heldout` 0.1076, `wide` 0.1715 | **reported**; would have read **fail** (> 0.10) | reported; a fall of 0.15–0.3 |
| **OVERALL** (P1, P2, P3, GUARD 1) | | **`fail`, through P3** | `fail` through P3, or `unresolved`; not `pass` |

### 2.2 `chat-v3d-aligned` (as first registered; reported beside)

| bar | measured | fired | §12.6 expected |
| --- | --- | --- | --- |
| **P1** anchored topic | shipped 0.0167 (10/600), CI [0.0091, 0.0304]; subject 0.0167 (10/600); **shipped-only 0, subject-only 0**, p = 1; mean reply 377.5 → 378.6 chars | **pass** (0 <= 0) | pass by tie |
| **P2** logP/char | shipped −0.3785, subject −0.2944, **Δ +0.0841**, SE paired 0.0029, SE by prompt 0.0040 over 100 prompts, **lower bound +0.0763**; `fresh` +0.0941, `heldout` +0.0583, `wide` +0.0893 | **pass** | pass; above the exploratory +0.0797 if §1's bias argument holds |
| **P3** UER@200, both exposed | both qualify 600/600; both have a definite subject **0.1350 (81/600)**, CI [0.1100, 0.1647]; flagged shipped 0.8642 (70/81), CI [0.7730, 0.9224]; subject 0.8642 (70/81); **fixed 2, broken 2**, p = 1 | **fail** (2 <= 2) | **unresolved** (2 v 0, 2 v 0, 5 v 0 on the read subsets) |
| **GUARD 1** identity | **108/108** identical; 0 dodge draws with a subject | **pass** | pass |
| **GUARD 2** sameness | 0.2081 (9848/47324) → 0.1563 (7382/47221), **fell 0.2488**; `fresh` 0.2532, `heldout` 0.1638, `wide` 0.2479 | **reported**; would have read **fail** | reported; a fall of 0.2–0.4 |
| **OVERALL** | | **`fail`, through P3** | **`unresolved`**, through P3 |

### 2.3 Against the expectation, plainly

* **The expected disagreement did not happen.** §12.6 expected `unresolved` on
  `chat-v3d-aligned` against `fail` on `chat-v6-scratch` and §12.7 said how a
  disagreement would be stated. Both fired `fail`, on the same bar, at the same
  counts. There is nothing to state as a disagreement.
* **P3 on `chat-v3d-aligned` was expected `unresolved` and fired `fail`.** The
  exploratory subsets read fixed 2–5 against broken 0; the confirmatory pools
  read 2 against 2. `fixed <= broken` is `fail` at any count, as §3 registered
  and §12.6 restated ("P3's `fail` side has no interval"). The expectation was
  wrong on this bar and the verdict stands as fired.
* **P3 on `chat-v6-scratch` was expected `fail` and fired `fail`, but not in the
  expected shape.** §12.6 expected broken to exceed fixed (0 v 3, 1 v 3, 2 v 3 on
  the read subsets) and called that "weak evidence that the rule makes
  introduction worse". The confirmatory read is a tie, 2 v 2, which is evidence
  of neither direction. §12.6 also said `QUALITY_v14.md` "may not say more than
  the counts do", and the counts say the rule left introduction where it was.
* **P2 passed on both**, at +0.0670 (lower bound +0.0575, clearing +0.05 by
  0.0075) and +0.0841 (lower bound +0.0763). §1's argument that the exploratory
  subset was conservative for P2 held on `chat-v3d-aligned` (+0.0841 against
  the exploratory +0.0797) and did not on `chat-v6-scratch` (+0.0670 against
  +0.0699); §2 said no §1 number predicts a §3 number "to better than its sign",
  since the read pools drafted 300 characters and these drafted 400.
* **GUARD 2 landed inside both expected ranges** (0.2488 in 0.2–0.4; 0.1815 in
  0.15–0.3), and the withdrawn bar would have read `fail` on both, as §4 and
  §12.6 both expected.

## 3. Reported always, gating nothing

### 3.1 GUARD 2, in full

| | `chat-v6-scratch` shipped → subject | `chat-v3d-aligned` shipped → subject |
| --- | --- | --- |
| distinct-2, pooled | 0.2201 (10409/47297) → 0.1801 (8479/47069), fall **0.1815** | 0.2081 (9848/47324) → 0.1563 (7382/47221), fall **0.2488** |
| `fresh` | 0.3603 → 0.2895, fall 0.1965 | 0.3513 → 0.2623, fall 0.2532 |
| `heldout` | 0.3580 → 0.3195, fall 0.1076 | 0.3493 → 0.2921, fall 0.1638 |
| `wide` | 0.2639 → 0.2187, fall 0.1715 | 0.2499 → 0.1879, fall 0.2479 |
| top 60-character opening | 0.0383 (23/600) → 0.0417 (25/600) | 0.0400 (24/600) → 0.0717 (43/600) |
| that opening, both arms, both checkpoints | `'Once upon a time, there was a little boy named Tim. Tim love'` | the same string |
| withdrawn 0.10 bar would have read | fail | fail |

The per-list distinct-2 values are not comparable with the pooled ones (less
text, higher ratio); each fall is a comparison of two selections of the same
draws, which is the only way the statistic is used.

### 3.2 The other reported-only columns

| | `chat-v6-scratch` | `chat-v3d-aligned` |
| --- | --- | --- |
| selection changed | 0.5983 (359/600), CI [0.5586, 0.6368] | 0.7283 (437/600), CI [0.6914, 0.7624] |
| no draft names the subject | **0.5600 (336/600)**, CI [0.5200, 0.5992]; `fresh` 87/120, `heldout` 37/120, `wide` 212/360 | **0.7050 (423/600)**, CI [0.6673, 0.7401]; 96/120, 55/120, 272/360 |
| replies below −0.3934 (plain sampling's battery mean, used as a per-reply line it was never calibrated to be; no bar) | 0.3950 (237/600), CI [0.3567, 0.4347] → 0.1783 (107/600), CI [0.1498, 0.2110]; per list 44 → 10, 46 → 22, 147 → 75 | 0.3617 (217/600), CI [0.3242, 0.4009] → 0.1067 (64/600), CI [0.0844, 0.1339]; 38 → 12, 35 → 16, 144 → 36 |
| mention of the noun (`topic_mention`), **the identity**: the tier selects on this matcher | 0.4250 (255/600) → 0.4400 (264/600); shipped-only 0, subject-only 9; `identity_violations` 0 | 0.2767 (166/600) → 0.2950 (177/600); shipped-only 0, subject-only 11; violations 0 |
| shipped tier: one member / whole pool / median | 0.3067 (184/600), CI [0.2711, 0.3447] / 1/600 / 3; buckets 1: 184, 2–4: 293, 5–16: 102, 17–64: 20, 65+: 1 | 0.3533 (212/600), CI [0.3161, 0.3924] / 5/600 / 2; 212, 297, 71, 15, 5 |
| subject tier: one member / whole pool / median | 0.1667 (100/600), CI [0.1390, 0.1986] / 336/600 / 256; 100, 87, 49, 27, 337 | 0.1217 (73/600), CI [0.0979, 0.1503] / 423/600 / 256; 73, 70, 19, 15, 423 |

The subject tier is the whole pool on exactly the draws where no draft names
the subject (336 and 423), and that is where the rule's whole effect lives: on
those draws it is score-only selection with the frame-word tier switched off.
The shipped tier was a single draft — chosen with no score consulted — on 184
and 212 of 600.

### 3.3 The all-draws UER count, the decomposition and the corpus beside it

Not the bar: this count moves with exposure, because a reply with no definite
subject in the window scores clean.

| | `chat-v6-scratch` | `chat-v3d-aligned` |
| --- | --- | --- |
| UER@200, all 600 qualifying draws | 0.2567 (154/600), CI [0.2233, 0.2931] → 0.1967 (118/600), CI [0.1668, 0.2304] | 0.2433 (146/600), CI [0.2107, 0.2792] → 0.1483 (89/600), CI [0.1221, 0.1790] |
| fixed / broken, all draws | fixed 61, **of which 59 because the subject pick has no definite subject at all**; broken 25 | fixed 78, **of which 76** for that reason; broken 21 |
| exposure (a definite subject in the window) | 0.3250 (195/600) → 0.2517 (151/600) | 0.3233 (194/600) → 0.1883 (113/600) |
| UER among exposed replies | 0.7897 (154/195), CI [0.7272, 0.8411] → 0.7815 (118/151), CI [0.7090, 0.8399] | 0.7526 (146/194), CI [0.6873, 0.8080] → 0.7876 (89/113), CI [0.7034, 0.8529] |
| unintroduced heads per definite subject | 0.6667 (224/336) → 0.6641 (174/262) | 0.5983 (207/346) → 0.6398 (119/186) |
| prompt as context (prompt words count as introduced) | 140 → 103 of 600; fixed 62, broken 25 | 135 → 76 of 600; fixed 80, broken 21 |
| **corpus reference** (`stories_topic.val.bin`, same window and stoplist, from the instrument's committed artifact, sha256 `68fc1eaa…9e71b`, recorded in both confirmatory artifacts) | exposure 0.2900 (529/1824), CI [0.2697, 0.3113]; UER among exposed 0.0794 (42/529), CI [0.0593, 0.1056]; per definite subject 0.0501 (42/838), CI [0.0373, 0.0671] | the same |

Both models sit an order of magnitude above the corpus on the per-subject rate
under either rule. The rule moves exposure away from the corpus's 0.2900 on
both checkpoints and leaves the per-subject rate where it was.

### 3.4 Latency, per pool (upper bounds, §1)

Seconds per turn, from each pool's `timing` block. `select` columns are one
selector alone on a cleared copy of the pool, null pass included when that
selector needs it; `rerank` is the draw plus the shipped selection.
`timing_pass_disagreements` is **0 on all ten pools**.

| pool | shipped select | subject select | rerank (draw + shipped) | a subject-rule turn, `rerank − shipped + subject` |
| --- | ---: | ---: | ---: | ---: |
| `clause` `chat-v3d-aligned` (72) | 0.1007 | 0.1318 | 0.6181 | 0.6492 |
| `dodge` `chat-v3d-aligned` (72) | 0.1103 | 0.1093 | 0.5777 | 0.5767 |
| `fresh` `chat-v3d-aligned` (120) | 0.1279 | 0.1646 | 0.6272 | 0.6639 |
| `heldout` `chat-v3d-aligned` (120) | 0.1278 | 0.1541 | 0.6277 | 0.6540 |
| `wide` `chat-v3d-aligned` (360) | 0.1140 | 0.1539 | 0.6079 | 0.6478 |
| `clause` `chat-v6-scratch` (72) | 0.1026 | 0.1293 | 0.6048 | 0.6315 |
| `dodge` `chat-v6-scratch` (72) | 0.1110 | 0.1096 | 0.5754 | 0.5740 |
| `fresh` `chat-v6-scratch` (120) | 0.1416 | 0.1611 | 0.6374 | 0.6569 |
| `heldout` `chat-v6-scratch` (120) | 0.1390 | 0.1602 | 0.6386 | 0.6598 |
| `wide` `chat-v6-scratch` (360) | 0.1170 | 0.1420 | 0.6097 | 0.6347 |

The subject rule adds 0.02–0.04 s to a story turn and nothing to a dodge turn
(where it is the shipped path). The largest subject-rule turn, 0.66 s, is under
the committed 0.70 s reference and far under the two-second line §6 named; no
turn is reported as a cost. These were measured under a shared host and are
not a re-measurement of the REPL's 0.70 s.

### 3.5 The clause subset (72 draws; outside `overall`)

Six prompts on which `tier_subject` keeps the noun and cuts the clause, six on
which it declines; the declined 36 are inside GUARD 1. The verdicts are what
each bar *would* read on this set alone and count for nothing.

| | `chat-v6-scratch` | `chat-v3d-aligned` |
| --- | --- | --- |
| draws with a subject | 36/72 | 36/72 |
| selection changed | 0.4028 (29/72) | 0.4583 (33/72) |
| no draft names the subject | 0.6944 (50/72) | 0.6806 (49/72) |
| mention (identity) | 34 → 43; shipped-only 0, subject-only 9 | 27 → 39; 0, 12 |
| P1 would read | **fail**: 2/72 → 0/72, shipped-only 2, subject-only 0, p = 0.5 | pass: 2/72 → 2/72, 0 v 0 |
| P2 would read | **fail**: +0.0234 (SE by prompt 0.0100 over 12 prompts) < +0.05 | **fail**: +0.0362 (SE by prompt 0.0158) < +0.05 |
| P3 would read | **fail**: both qualify 70/72, both exposed 25/70, 24 → 24, fixed 0, broken 0 | **fail**: both exposed 22/72, 18 → 19, fixed 1, broken 2 |
| below −0.3934 | 51/72 → 37/72 | 35/72 → 21/72 |
| distinct-2 | 0.4656 → 0.4326, fall 0.0707 | 0.4373 → 0.4146, fall 0.0517 |
| subject tier one member / whole pool | 24/72 / 14/72 | 22/72 / 13/72 |

On the two draws where the clause set's P1 moved against the rule on
`chat-v6-scratch`, the shipped pick was anchored on the noun and the subject
pick was not. Two draws of 72; a trip-wire, as §3 called P1.

## 4. Real replies, chosen by rule

**Rule:** for each story list in the artifact's file order (`fresh`, `heldout`,
`wide`), the first draw at sampler seed 6 in row order whose λ = 0.6 shipped and
subject-tier indices differ; one pair per list, three per checkpoint. Nothing
was skipped. Texts are the stored replies, shown exactly; `logP` is
`logp_cond / n_scored` over the 400-character draft. On all six pairs **no draft
in the pool named the subject**, so the subject tier was the whole pool of 256
and the rule was score-only selection; on all six, neither pick names the noun.

### 4.1 `chat-v6-scratch`

**`fresh`, "tell me a story about a windmill", seed 6.** Shipped: index 77, a
tier of 2, logP −0.3764, four unintroduced heads in the window (bird, flower,
fly, tree). Subject: index 51, logP −0.2648, no definite subject in the window.

> shipped: `Once upon a time, in a big forest, there was a big field. The flower was big and red. The fly was far away and the sun was growing. The old tree was also very hungry.\nOne day, the little bird saw the big bird. The bird wanted to play with the bird. The bird said, "I need to run from you!" The frog said, "Yes, I will tell you."\nThe little bird was sad and had a surprise.`

> subject: `Once upon a time, in a big green forest, there was a little boy named Tim. Tim loved to play with his friends and the birds who lived in a small house. They liked to play games and all night. Tim liked to show his friends and play with the soft colors.\nOne day, Tim and his friends were playing in the park. They played a garden in the pond. They saw a big tree with a hole in it. Tim was very happy.`

This is the all-draws "fixed" case §3.3 counts 59 of: the shipped reply had
unintroduced heads and the subject reply has no definite subject at all. It is
not in P3's denominator.

**`heldout`, "tell me a story about a lighthouse", seed 6.** Shipped: index 67,
**a tier of 1** (chosen with no score), logP −0.4591, below the floor. Subject:
index 123, logP −0.2418. Neither has a definite subject in the window.

> shipped: `Once upon a time, there was a little girl who loved to play outside. One day, she saw a big rock in the dirty shop. She wanted to play with it. She found a big card on the shelf. It was a happy car to help the door.\nAs she picked up the tree, she saw a big tree. It was a big party for her. She wanted to tell it, but she knocked it. So, she asked her mom, "Why do you dry?`

> subject: `Once upon a time, there was a little girl named Amy. She lived in a small house with her mom, dad, and her dog, Spot. They liked to play and have fun. They would play all day long.\nOne day, Sue went to the park with her friend, Tim. Sue said, "Let's play with the ball!" Sue asked Sue, "Do you want to play outside?" Sue asked. She said, "Yes, I will be fun!"\nThey were surprised her.`

**`wide`, "tell me a story about a zipper", seed 6.** Shipped: index 126, a tier
of 4, logP −0.3927. Subject: index 45, logP −0.2519. Neither exposed.

> shipped: `One day, a little girl named Lily went to play in her grandma. They liked to make them flying brands with her. They played all day long. The sun was shining, and the sky was blue. They were having lots of fun.\nLily walked up to the bell too. The dog said, "Hi, little girl! You can tell me a big surprise!" Tom was very happy. He stopped in the house and put it on the swings.`

> subject: `Once upon a time, there was a little boy named Tim. Tim was a very attractive because he was always very enthusiastic. He loved to play with his toys and run around the park. One sunny day, Tim went outside to play with his friends.\nAs they played, a big cat came to Tim's house. The structure was a big sister. Tim wanted to show it to his mom, "Can I play with your toys?"`

### 4.2 `chat-v3d-aligned`

**`fresh`, "tell me a story about a windmill", seed 6.** Shipped: index 247, a
tier of 4, logP −0.3429, one unintroduced head (fish). Subject: index 88, logP
−0.2875, no definite subject.

> shipped: `Once upon a time, in a small town, there was a big, factory. The fish lived in a little house with a farm with many other fish. The fish liked to play and jump all day long.\nOne day, a little girl named Lily saw the fish and he wanted to help the bird. She said, "I will tell what you want to do." The bird said, "No, I want to play with the hole."`

> subject: `Once upon a time, there was a little girl named Sue. Sue loved to play outside. One day, she found a magnet in her garden. It was very small. Sue was very happy.\nSue wanted to see what was there to do. She said, "Sue, do you want to play with me?" Sue said, "Yes, you can help me." The next day, something unexpected happened.`

**`heldout`, "tell me a story about a lighthouse", seed 6.** Shipped: index 231,
a tier of 2, logP −0.3354. Subject: index 208, logP −0.2416. Neither exposed.

> shipped: `Once upon a time, in a big forest, there lived a little bird named Billy. Billy loved to sing and dance all day. One day, Billy saw a big tree with a small bird who wanted to see what was behind it.\nAs Tim walked, he met a girl named Sue. Sue was sad because she lost his toy sun. She said, "Hi Sue! Do you want to play with me?" Sue said, "Yes, I can tell you a small cat for the bug in the back."`

> subject: `Once upon a time, in a big forest, there was a little girl named Lily. She loved to play outside with her friends. One day, she found a big rock on the ground. She was very happy and wanted to show it to her friends.\nLily went to the park to play. She saw a big tree with a plane. She was very happy and started to play.\nMia went to the store and wanted to play with it.`

**`wide`, "tell me a story about a reindeer", seed 6.** Shipped: index 114, **a
tier of 1**, logP −0.3646. Subject: index 11, logP −0.2284. Neither exposed.

> shipped: `Once upon a time, there was a pale boy named Tim. Tim was a very adventure with his mom. One day, Tim was playing with his toy car in the garden. He saw a tree with a nest on the ground. He picked it up and looked at it.\n"Can I do it?" he said. Tim said, "Yes, but I will tell you in the town." Tim was excited as he started to worry. He wanted to help the car.`

> subject: `Once upon a time, in a big green forest, there lived a girl named Lily. She lived in a small house with her mom, dad, and she went to the store. Lily loved to play in the sun all day long.\nOne day, Lily found a small rock. She was very happy. She did not know what to do. Lily was sad. She did not know what to do.\nLily asked the cat, "Why are you sad?" Lily asked.`

What the six pairs show is what the numbers say: the subject pick is the
likelier text (by 0.05–0.22 nats/char here), it opens the same way more often,
and it is no more about a windmill, a lighthouse, a zipper or a reindeer than
the shipped pick was. None of the twelve replies is about its subject.

## 5. What the tie on P3 means

**The four discordant draws per checkpoint are the whole of P3's evidence.**
From the pools, by prompt and seed (the fixed/broken label is the artifact's;
the heads are what the instrument flagged in the 200-character window):

| | draw | shipped pick flagged | subject pick flagged |
| --- | --- | --- | --- |
| `chat-v6-scratch` | `fresh` "i want a story about a crocodile", seed 10 | — | car |
| | `fresh` "tell me a story about a raccoon", seed 7 | birds | — |
| | `wide` "tell me a story about a zipper", seed 11 | — | jeep, jellyfish |
| | `wide` "tell me a story about a surfboard", seed 8 | bucket | — |
| `chat-v3d-aligned` | `fresh` "tell me a story about a scarecrow", seed 8 | — | music |
| | `fresh` "tell me a story about a drum", seed 8 | button | — |
| | `wide` "tell me a story about a grandmother", seed 11 | — | bird |
| | `wide` "tell me a story about a sailboat", seed 7 | tree | — |

Two fixed, two broken, on both. The other 107 and 77 both-exposed draws are
concordant: 93 and 68 flagged under both rules, 14 and 9 clean under both. The
rate among draws with a definite subject under each rule is identical to the
draw — 95/111 and 70/81 — and the rate of unintroduced heads per definite
subject moved by −0.0026 (0.6667 → 0.6641) and +0.0415 (0.5983 → 0.6398), both
inside their intervals.

**The all-draws fall (154 → 118 and 146 → 89) is lost exposure, not better
introduction.** 59 of 61 and 76 of 78 all-draws "fixed" counts are draws where
the subject pick contains no definite subject in the window; §1 of the
pre-registration saw the same on the read pools (134 of 141) and rewrote P3 to
read only where both picks are exposed for exactly this reason. That history
was stated in §3 before the run and it is what happened.

**The power statement, as written in §12.6, and how it landed.** §12.6 expected
50–90 both-exposed draws on `chat-v3d-aligned` and 100–150 on
`chat-v6-scratch`, with "a handful" discordant; the run gave 81 and 111, with 4
and 4. The smallest passing count was 6 v 0; a pass on `chat-v6-scratch` "would
need about twelve fixed" against the three broken then seen. Neither
checkpoint came within four draws of the smallest passing split. P3's `fail`
side has no interval: a tie fails, by the registered rule, and it is reported
as fired. A tie of 2 v 2 is also not evidence that the rule harms introduction.
What the counts support is the sentence §3 wrote before the run: the rule does
not change how carefully the model's text introduces things.

## 6. Scorer changes after the stamp

§9 item 2 (as amended) requires any change to `score_v14.py` or `v14_pools.py`
between the stamp and scoring to be reported with its diff.
`git diff --numstat c323e53 c907ddc` lists two files: `scripts/chat/score_v14.py`
(+84 / −5) and `tests/test_snnchat_v14_scoring.py` (+102 / −2).
`scripts/chat/v14_pools.py`, `src/snnchat/` and `docs/chat/PREDICTION_v14.md`
are unchanged. The pools were drawn and scored at `c907ddc`. The scorer diff,
`git diff c323e53 c907ddc -- scripts/chat/score_v14.py`, in full by hunk:

1. **`check_design` and `CONFIRMATORY_SEEDS = (6, 7, 8, 9, 10, 11)`** (`11d34be`):
   a third refusal before scoring. A draw at a sampler seed below 6 is refused
   on any checkpoint; a `(prompt, seed)` given twice is refused on any
   checkpoint; on a registered checkpoint a file list that is not every prompt
   of the five sets at seeds 6–11, once, is refused. Its result is the new
   `design` key in the artifact (`missing 0`, `not_in_the_design 0`,
   `is_the_registered_design true` on both) and one printed line. Imports
   `COMMITTED_SEEDS`, `SET_NAMES`, `probes_for` from `v14_pools`.
2. **`--exploratory` with no `--out` refuses to overwrite the committed
   `v14_exploratory.json`** (`11d34be`), which §12.3 says is not to be
   regenerated.
3. **Wording**: the module docstring's list of refusals and its usage line; the
   deciding-checkpoint print names `experiments/chat/SHIPPED` in the owner's
   *main* tree, not this worktree's copy.

**No bar moved.** `P1_RULE`, `P2_POOLED_BAR`, `P2_LISTS`, `P3_ALPHA`,
`GUARD2_MAX_DISTINCT2_DROP`, `GUARD2_RULING`, `IN_OVERALL`, `PREDICTION_STAMP`,
`REGISTERED_CHECKPOINTS`, `DECIDING_CHECKPOINT`, `FLOOR`, the four verdict
functions, `replay`, `compare`, `guard1` and `overall` are outside the diff.
Every refusal added can only stop a score, never change one, and none fired.
`c907ddc` itself changes only the test file.

## 7. Limitations

1. **Two moves were inferred by the lead, not ruled by the owner**, and §12.2
   says so. The deciding checkpoint became `chat-v6-scratch` because the owner
   answered "Yes" to shipping it, not because he was asked which checkpoint
   should decide this round; and the GPU cost doubled to ten pools (25 m 21 s
   measured across the chain) to draw both. Neither could have helped the rule:
   `chat-v6-scratch` was expected to read worse than `chat-v3d-aligned` on P3
   and the two checkpoints are scored alone, so a second checkpoint cannot lift
   the first's verdict. Had the original single-checkpoint design been kept,
   `chat-v3d-aligned` would have fired `fail` as well, through P3 (and, under
   the bars as first stamped, through GUARD 2 too).
2. **P2 selects on what it measures.** §1 of the pre-registration and the
   scorer's docstring both say it: wherever the subject tier is the whole pool
   (336/600 and 423/600 here) the rule is a maximum over a superset of the
   shipped tier, and its λ = 0.6 score is what P2 reads. A P2 pass says the rule
   did what it was built to do. It is not evidence that replies are better, and
   it is measured over the 400-character draft including an unclosed tail the
   reader never sees.
3. **The exploratory basis was selected on the shipped rule's behaviour**
   (§1: the 1,252 of 2,400 draws with a measured null term are those where the
   shipped tier was not a single draft). That biased the exploratory P2 down
   and the confirmatory pools, with a null term on every draw, were meant to
   sit above it. They did on one checkpoint and not on the other (§2.3). The
   exploratory numbers sized the bars; they tested nothing, and nothing here
   relies on them.
4. **Latency is an upper bound** (§1, §3.4): nine of ten pools were drawn with
   a CPU workload on the same host. No figure from this run replaces the
   committed 0.70 s.
5. **One training seed per checkpoint, first turn only, bare prompts.** No
   interval here covers training-seed variability; every draw starts from
   `state=None` where the REPL carries membrane state across turns; every
   story prompt has the shape "tell me a story about a X" (or "i want a story
   about a X"). The clause set is the only view of the extractor on anything
   else, and it is 72 draws outside the bars.
6. **The instrument.** UER@200 is a regex for one failure, never validated
   against a human rating, blind to a plain interleave, and clean on any reply
   with no definite subject. P1's `anchored_topic` is not windowed and holds on
   about 2–3 % of picks. P3's `fail` side has no interval; a 2 v 2 tie fails by
   rule and is not a measured direction.
7. **The matcher's blind spot is inherited** (§0 of the pre-registration):
   `echoes` tolerates `s`/`es` only, so a draft about "two ponies" is outside
   the "pony" tier. `identity_violations` read 0 on both, so the tier and the
   mention column never came apart, but the count of drafts that name the
   subject is a count under that matcher.

## 8. Decision, and what is referred forward

**The default stays OFF.** §12.7: only `pass` on `chat-v6-scratch` flips it. It
fired `fail`. `scripts/chat.py`'s `build_parser` default for `--subject-tier` is
unchanged, `/subject` defaults to off, and off nests the shipped selection
exactly (`test_the_subject_tier_flag_is_off_by_default_and_reaches_the_session`,
`test_the_subject_tier_is_off_by_default_and_off_is_the_old_rule`). The flag
stays in the code because off is the old path and the evidence scripts depend
on it. `experiments/chat/SHIPPED` was not edited by this round under any
outcome and was not.

**The toggle is the owner's way to use the rule anyway.** `python scripts/chat.py
--subject-tier`, or `/subject on` in the REPL (survives `/rerank`, per
`test_subject_on_survives_a_rebuilt_rerank`). If he prefers replies that read as
likelier text and look more alike, with the same chance of being about what he
asked, that is a taste call he can make in one keystroke, and nothing he sees
there is a measurement. The help text and the `/subject on` message say the
rule is not confirmed; both are now also true in the past tense.

**Referred forward, decided by nobody here:**

1. **The instrument stands and `CONVENTIONS.md` §7 applies to every future
   training round**: pool UER@200 next to weighted bpc, always with its
   qualifying fraction and `uer_decomposition`'s two factors, read against the
   corpus reference 0.0230 (42/1824) and the incumbent's `SD_seed`, and **never
   inside `snnchat/rerank.py`**. This round is the first read of the instrument
   on a selector, and its result — the per-subject rate did not move while the
   all-draws count fell — is the case for the decomposition rule.
2. **Entity introduction is not a likelihood problem.** Both rules leave
   95/111 and 70/81 exposed replies with an unintroduced head, against the
   corpus's 42/529; the picker that maximises likelihood did nothing to it. A
   next attempt would need a selector that reads something other than the
   model's opinion of its own text, and UER may never be it (item 1). Whether
   a training-side change moves the per-subject rate is a question for a
   training round, reported under §7, and is not proposed here.
3. **The sameness cost is real and now measured at the REPL's length**:
   0.1815 and 0.2488 relative falls in distinct-2 at 400 characters, against
   0.2124 and 0.3006 on the read 300-character pools. The owner withdrew the
   bar (§12.2, "As many as needed"); the number is on the record for whoever
   next proposes a score-only selector.
4. **No further seed, pool or checkpoint is drawn to firm up a verdict** (§2,
   `CONVENTIONS.md` §4 rule 4). Sampler seeds 6–11 are now spent on both
   checkpoints for these five sets.

---

## Appendix: commands and hashes

All from the worktree root `C:/Elliot's Stuff/SNN-worktrees/chat-v14`; the
scorer and every check below ran with `CUDA_VISIBLE_DEVICES=-1`.

**The pools**, one at a time, `<RUN>` = `chat-v3d-aligned` then
`chat-v6-scratch`, `<SET>` = `clause`, `heldout`, `fresh`, `wide`, `dodge`
(the `command` array of each run's `launch.json`):

```
python scripts/launch.py --script scripts/chat/v14_pools.py --run-name v14_pools_<SET>_<RUN> -- --set <SET> --device cuda --ckpt "C:/Elliot's Stuff/SNN/Text SNN/experiments/chat/<RUN>/ckpt_best.pt" --out "C:/Elliot's Stuff/SNN-worktrees/chat-v14/experiments/chat/_quality/v14_<SET>_<RUN>.json.gz"
```

**The scoring**, once per checkpoint, the `command` field of each artifact:

```
CUDA_VISIBLE_DEVICES=-1 python scripts/chat/score_v14.py "experiments/chat/_quality/v14_*_chat-v3d-aligned.json.gz"
CUDA_VISIBLE_DEVICES=-1 python scripts/chat/score_v14.py "experiments/chat/_quality/v14_*_chat-v6-scratch.json.gz"
```

**The reproduction for this document**: the same two commands with `--out
<scratch>/v14_confirmatory_<RUN>.json`, then a recursive comparison of every
key path against the committed artifact. Result: 0 differing leaves on each,
excluding `command`.

**The stamp**: `git show 2fa59fe:docs/chat/PREDICTION_v14.md | sha256sum` and
`sha256sum docs/chat/PREDICTION_v14.md` both print
`83326a7cd92190f58e17d80c2395b57b491094df5b47d8d49ccc09bdd22ae0f1`.

**The deciding checkpoint**: `head -1 "C:/Elliot's Stuff/SNN/Text SNN/experiments/chat/SHIPPED"`.

**The scorer diff**: `git diff c323e53 c907ddc -- scripts/chat/score_v14.py` (§6).

**sha256 of every artifact this document reads** (`sha256sum`, committed at
`2fa59fe`; the pool hashes are also the `inputs[].sha256` fields of the
confirmatory artifacts):

| file | sha256 |
| --- | --- |
| `v14_confirmatory_chat-v6-scratch.json` | `7e12d845fe51989515bbddacf0a612d0ff2c37aa68ec5e5bf864c13bd8970ec0` |
| `v14_confirmatory_chat-v3d-aligned.json` | `753275958c87849d5c69fedab373cfc2a7a46904babf9f70ade9e9022dda9825` |
| `v14_clause_chat-v6-scratch.json.gz` | `5e6f4103c86632699b442e557ffcc6d74867cc578a1748b249938f2bcc7c2c5c` |
| `v14_dodge_chat-v6-scratch.json.gz` | `01b77df97901fb791f7643be6fa3c40b2ed486a44d8b8c5562382fdfc639c600` |
| `v14_fresh_chat-v6-scratch.json.gz` | `efc1c1beb18c036cb46901441aad989a27854048e93a66231e61077e4e7f3c0a` |
| `v14_heldout_chat-v6-scratch.json.gz` | `2b7d6e45acb8930052812b06fd6514f9fcb82b373efd9091d4aa99e9782cf1e0` |
| `v14_wide_chat-v6-scratch.json.gz` | `0fd25ed4c66bc77902661ab579ee52109723ffce48bb5a67bec297cc47091b0d` |
| `v14_clause_chat-v3d-aligned.json.gz` | `85bde50be69eb8b50d9141014e3e010dcc0cb424a1ec812d29b03cf869632de6` |
| `v14_dodge_chat-v3d-aligned.json.gz` | `ce3cd2c4d256ed452d986b4959f9d9c88eb8f8eb3553666cae67648087b8e8c0` |
| `v14_fresh_chat-v3d-aligned.json.gz` | `0d9e35c0437eb5c77fbd66ce682dd2947af15ccc3697c2264d071302d741f315` |
| `v14_heldout_chat-v3d-aligned.json.gz` | `44a665397982054592bdf7da77130c1a81edf85ce3d83dd3035d6f90dd4557e3` |
| `v14_wide_chat-v3d-aligned.json.gz` | `0d2a1085f0e8b18d68a4f79f32fa8603495c7f55c2c72ee3cc97319417931161` |
| the instrument's corpus-reference artifact (`uer_corpus_reference.sha256` in both) | `68fc1eaa6636cb7147cb235315ce10606dd5f981bf417382f81295809899e71b` |
| `chat-v6-scratch/ckpt_best.pt` (pool headers) | `e270b3a3c878b9253f92cb8d22f69f1e27f11ea2cc81b3e0854426929c62676f` |
| `chat-v3d-aligned/ckpt_best.pt` (pool headers) | `3deca3eb0265925e2471b5cb3ace4c800e5dc1f116df037a2699af8ba49ac86d` |

All files under `experiments/chat/_quality/`. Run directories
`experiments/runs/v14_pools_<SET>_<RUN>/` (gitignored) hold each pool's
`launch.json`, `stdout.log` and `run.pid`; §1's times are read from them.
