# PREDICTION v14 — the subject-tier round

**Written 2026-09-21, before any confirmatory pool of this round has been
drawn.** No file `experiments/chat/_quality/v14_*.json.gz` exists at the time of
writing, and no draw at sampler seeds 6–11 has ever been made from any chat
checkpoint. The scorer (`scripts/chat/score_v14.py`), the pool writer
(`scripts/chat/v14_pools.py`) and the instrument (`snnchat.coherence`) were
written, reviewed and committed before this file — they are read here at commit
`7a1e0d0` — so this round cannot tune the instrument to the result.

Not part of the Phase-1..5 research protocol. No number here is a reported
figure. Rates follow [`CONVENTIONS.md`](CONVENTIONS.md).

**This round trains nothing and rules no research decision.** Phase 4's
decision **#4** is open and is Elliot's. Nothing under `src/snn/` is modified,
`experiments/chat/SHIPPED` is not edited under any outcome, and the checkpoint
does not change. The only thing that can change is a decoding default.

**This round does not make the model more intelligent, and no outcome of it may
be reported as doing so.** It changes which of 256 drafts the REPL shows. The
drafts, the weights and what the model knows are the same under both rules.

---

## 0. What changes, what the owner would notice, and what it cannot do

`RerankParams.subject_tier` (`src/snnchat/rerank.py`), **default OFF**, no CLI
flag. Off is the old code path exactly
(`tests/test_snnchat.py::test_the_subject_tier_is_off_by_default_and_off_is_the_old_rule`).

The shipped selector builds its top tier from the weighted sum of echoed prompt
words, and the frame counts: a draft that says "tell" and "story" outranks a
draft that says only "penguin" (`rerank.py`, "THE SUBJECT TIER, AND WHY THE
WEIGHTED SUM STILL LETS THE FRAME WIN"). When that tier holds one draft the
score is never consulted, so the reply is chosen with no regard to how well the
model thinks it reads. With the flag on, for a turn whose subject is one simple
noun phrase (`snnchat.prime.tier_subject`), the tier is the drafts that name the
subject, and **if none does, the whole length-partitioned pool** — so the score
chooses among everyone.

**What the owner would notice, if it ships: replies to story requests that are
less often garbled. Not smarter.** On the pools already on disk the rule's
picks fall below plain sampling's mean log-probability per character (−0.3934)
on 31 of 1,252 draws against the shipped rule's 158 (§1). That is the model's
own opinion of its own text and nothing else. **He would also notice more
replies that look alike** (§3, GUARD 2): this is the one way the change can
read as worse, and on the pools already read it does.

**What it cannot do.**

* **When a draft names the noun, almost nothing changes.** On the 279 of those
  1,252 draws where some draft names the subject, the two rules chose a
  different draft on 7, and the log-probability difference is −0.0005
  nats/char. The whole measured effect is on the other 973, where nobody
  drafted the noun and the rule is score-only selection with the frame-word
  tier switched off (Appendix, command D).
* **It adds no knowledge and no drafts.** No draft in the pool names the subject
  on 0.7271 (1745/2400) of the committed draws, 95 % CI [0.7089, 0.7445]. If the
  model did not write about a penguin, no selector can show a penguin story;
  `QUALITY_v8.md` §4 said so of the echo partition and it is as true of this.
* **It does not touch a request whose "about X" is not one noun phrase.** "a
  penguin and make it funny" gives no subject and the shipped rule decides, as
  before. A closed word list cannot see an open-class word in the wrong place
  (`tier_subject`'s docstring: "a penguin quickly" gives "quickly").
* **It inherits the matcher's blind spot.** `snnchat.rerank.echoes` tolerates
  `s`/`es` and nothing else, so a draft about "two ponies" does not name "pony".
  Under the weighted rule that costs one term of a sum; under this rule it
  removes the draft from the tier. Not fixed this round: the matcher is the
  shipped one and every committed pool's `echo_weight` is checked against it.

## 1. DISCLOSURE: the rule was found post hoc, on pools already read

**The subject rule was designed while reading the committed `v12_*` pools**
(sampler seeds 0–5, four training seeds of the shipped recipe, 2,400 story draws,
614,400 candidates). Every number in this section comes from those pools and
**none of it is a test of the rule.** It is what the bars in §3 are sized from,
and it is why §2 draws new pools at seeds that have never been used.

The committed pools were not written for this and cannot replay the subject rule
everywhere: `select` skips the null pass when the shipped tier decides, so
`logp_null` exists only where the shipped tier had more than one member.

| exploratory subset (λ = 0.6 unless said) | draws | what it is |
| --- | ---: | --- |
| `lam0.6_null_measured` | **1,252** of 2,400 | the whole pool carries a measured `logp_null`. The headline basis |
| `lam0.6_replayable` | **1,582** of 2,400 | the above plus draws where the subject tier is a single draft |
| `lam0_all` | 2,400 of 2,400 | pure likelihood inside each tier; needs no null term |

**Is the basis biased, and which way?** Yes, and **conservatively for P2**. The
1,252 are selected ON the shipped rule's behaviour: they are exactly the draws
where the shipped tier was *not* a single draft, so the 1,148 excluded draws are
the ones where the shipped pick was made with no score at all — the defect the
rule is aimed at. The builder of the evidence scripts raised it, the review that
re-derived the replay judged the direction conservative for P2, and it
reproduces: at λ = 0, where every draw is replayable, the difference is
**+0.0867** nats/char on the 1,252 and **+0.1273** on the excluded 1,148
(command D). The confirmatory pools carry a null term on every draw, so the
confirmatory P2 should sit **above** the exploratory +0.0807, not at it. The
bias does not rescue GUARD 2: the relative fall in distinct-2 is 0.3858 on the
1,252 and 0.3884 on the excluded draws at λ = 0 — the cost is everywhere.

What the exploratory replay read, on the 1,252 (command A; identical to the
committed `v14_exploratory.json` in every field but `command`):

| | shipped pick | subject-tier pick |
| --- | --- | --- |
| selection changed | — | 0.6597 (826/1252), CI [0.6330, 0.6855] |
| logP/char | −0.3214 | −0.2406, **Δ +0.0807**, SE by prompt 0.0039 (100 prompts), SD across 4 training seeds 0.0088 |
| Δ per list | | heldout +0.0427 (n = 279), fresh +0.0833 (264), wide +0.0947 (709) |
| below −0.3934 | 0.1262 (158/1252) | 0.0248 (31/1252) |
| anchored topic | 0.0208 (26/1252) | 0.0216 (27/1252); shipped-only 0, subject-only 1 |
| UER@200, all qualifying draws | 0.1928 (241/1250) | 0.1072 (134/1250); fixed 141, broken 34 |
| … of which exposure | 0.2540 (318/1252) | 0.1416 (177/1250) |
| … UER among exposed replies | 0.7610 (242/318) | 0.7571 (134/177) |
| … unintroduced per definite subject | 0.6329 (350/553) | 0.6547 (182/278) |
| UER@200, both picks exposed | 0.8492 (107/126) | 0.8254 (104/126); **fixed 7, broken 4, p = 0.5488** |
| distinct-2 | 0.1485 (10788/72639) | 0.0998 (7170/71875), **a fall of 0.3283** |
| top 60-character opening | 0.0431 (54/1252) | 0.0823 (103/1252) |

Two things in that table shaped the bars and are disclosed as such. **134 of the
141 "fixed" draws are fixed only because the subject-tier pick contains no
definite subject at all** — the all-draws UER gain is lost exposure, not better
introduction, and the corpus sits at exposure 0.2900 (529/1824), so the rule
moves exposure *away* from the corpus. And bare mention of the noun cannot lose:
the tier selects on the matcher that column scores (`QUALITY_v8.md` §4's trap,
met again). §3 is written around both.

On the **shipped checkpoint alone** — the only one §2 draws from — the same
replay reads (command B): Δ **+0.0797** on 318 null-measured draws (heldout
+0.0511, n = 61; SE by prompt 0.0053), +0.0644 on 393 replayable, +0.1056 on all
600 at λ = 0 (heldout +0.0806; SE by prompt 0.0055); both-exposed draws 26/318
with fixed 2, broken 0 (57/599 with 5 and 0 at λ = 0); distinct-2 0.2428 →
0.1698, a fall of 0.3006 (0.3802 at λ = 0); anchored topic 11 → 11 with no
discordant draw on any subset.

## 2. The confirmation design, fixed here

| | value | why |
| --- | --- | --- |
| checkpoint | `chat-v3d-aligned/ckpt_best.pt`, the one `experiments/chat/SHIPPED` names; sha256 `3deca3eb0265925e2471b5cb3ace4c800e5dc1f116df037a2699af8ba49ac86d` | the default being decided is the REPL's, and the REPL loads this. `v14_pools.py` records the hash in every pool header |
| sampler seeds | **6, 7, 8, 9, 10, 11** | seeds 0–5 are spent (§1). `v14_pools.py` refuses an offset below 6 |
| story lists | `HELDOUT` 20 prompts, `FRESH` 20, `WIDE` 60, from `echo_holdout.SETS`, imported not copied | 120 + 120 + 360 = **600 story draws** over 100 prompts. Verified against `scripts/chat/echo_holdout.py` (command D prints the sizes) |
| guard sets | `dodge` 12 prompts = 72 draws; `clause` 12 prompts = 72 draws | GUARD 1, and the only view of the subject extractor (§5) |
| candidates, λ | n = 256, λ = 0.6 | `scripts/chat.py`'s `DEFAULT_RERANK_N` and `DEFAULT_RERANK_LAMBDA` |
| drafted characters | **400**, the REPL's `--max-new` | the committed pools were drawn at 300. §1's numbers therefore describe a *different configuration* from the one confirmed, and no §1 number is a prediction of a §3 number to better than its sign |
| pairing | ONE stored pool per draw; both selectors' winners recorded on the device, then replayed by `score_v14.py` through the real `snnchat.rerank.echo_tier` | exactly paired, zero-GPU scoring, and the scorer refuses to report (`ReplayMismatch`) unless its replay lands on both recorded indices on every draw |

**The sample size is fixed and honoured** (`CONVENTIONS.md` §4 rule 4): 600
story draws, 72 dodge, 72 clause, no more and no fewer. A pool that fails to
write is redrawn whole at the same seeds, and a partial pool is deleted, never
scored. No seed beyond 11 is drawn to firm up a verdict.

**What is NOT in the interval.** One checkpoint: training-seed variability is
not in any interval below, and the verdict is about this checkpoint's default,
not the recipe. (§1's SD of 0.0088 across four training seeds is the only
evidence on that, and it is exploratory.) A first turn only: every draw starts
from `state=None`, and the REPL carries membrane state across turns
(`v14_pools.py`, "WHAT STILL DIFFERS FROM A REPL TURN"). The prompt lists are
fixed, not sampled, and every story prompt has the bare shape "tell me a story
about a X".

## 3. The bars

**These are the bars `score_v14.py` implements at commit `7a1e0d0`, and they are
not the bars this round was first drafted with.** The first draft read P1 on
bare mention of the noun, P3 on the all-draws UER count, and gave sameness no
bar. Review found that the first cannot fail, the second passes on lost exposure
alone, and the third left the only measured cost outside `overall` (§1). All
three were changed **before any confirmatory pool existed**, and that history is
stated here rather than tidied away. The bars below were therefore written
*after* the exploratory numbers were seen; §4 says how each reads on them.

> **P1 (guard, topic non-inferiority).** Over the 600 story draws, on
> `snnchat.coherence.anchored_topic`: draws only the shipped pick satisfies must
> not outnumber draws only the subject-tier pick satisfies
> (`shipped_only <= subject_only`). A tie passes, 0 against 0 included. `fail`
> otherwise.

> **P2 (primary).** Pooled mean of (subject-tier pick − shipped pick) selected
> log-probability per scored character over the 600 story draws >= +0.05
> nats/char, AND the same difference > 0 on each of HELDOUT, FRESH and WIDE.
> `fail` if the pooled point estimate is below +0.05 or any list is not above 0;
> `unresolved` if a list is missing, or if the point estimate meets +0.05 but
> `delta − 1.96·max(SE_paired, SE_by_prompt)` does not; `pass` otherwise.

> **P3 (primary).** UER@200 of the selected replies, over the draws where BOTH
> picks qualify (>= 200 characters) and BOTH contain a definite subject in the
> window: fixed > broken with exact two-sided McNemar p < 0.05 is `pass`;
> fixed > broken with p >= 0.05 is `unresolved`; fixed <= broken is `fail`; no
> such draw is `unresolved`.

> **GUARD 1 (identity).** On every draw whose prompt has no tier subject — all
> 72 DODGE draws and the 36 CLAUSE draws on which `tier_subject` is None — both
> selectors return the same pool index and the same text. One exception is
> `fail`; a DODGE prompt that has a subject is `fail`; no such draw supplied is
> `unresolved`.

> **GUARD 2 (sameness).** Relative fall in distinct-2 from the shipped picks to
> the subject-tier picks of the same 600 draws, `1 − subject/shipped`, <= 0.10
> pooled and on each of HELDOUT, FRESH and WIDE. Pooled fall > 0.10 is `fail`;
> pooled within the bar but a list over it, or missing, is `unresolved`.

> **OVERALL.** `fail` if any of P1, P2, P3, GUARD 1, GUARD 2 is `fail`;
> otherwise `unresolved` if any is `unresolved`; otherwise `pass`.

**A near-miss is a miss** (`CONTRIBUTING.md` §3; `CONVENTIONS.md` §4 rule 5).
**`unresolved` is a real verdict, is reported as one, and does not flip the
default.** No bar is recomputed, moved or re-read on a subset after the pools
exist. Secondary columns (§6) may not be promoted to bars after the fact.

### Denominators, lattices and power, per `CONVENTIONS.md` §4

**No bar sits on a lattice point.** Checked, not asserted (command D):

* **P1** compares two integers and has no threshold; its tie is defined above.
* **P2**'s statistic is a mean of 600 floating-point differences and +0.05 is
  not a distinguished value of it. The per-list rule is a strict `> 0`; a list
  on which the two rules choose identically everywhere reads exactly 0 and
  **fails**.
* **P3**'s exact two-sided McNemar p is `2·Σ C(n,k) / 2^n`, a dyadic rational;
  0.05 = 1/20 is not one, so p = 0.05 is unattainable at any count. Enumerated
  for every split of n <= 600 discordant draws: none gives 0.05.
* **GUARD 2**'s statistic is a ratio of two ratios of set sizes in the tens of
  thousands; the rule is `> 0.10` fails, so a fall of exactly 0.10 passes.
* **GUARD 1** is an identity over 108 draws, not a rate.

**P2's power.** The by-prompt SE on the shipped checkpoint's exploratory replay
is 0.0053–0.0055 over ~100 prompts, so a 600-draw run resolves `pass` only for a
pooled Δ of about **+0.061 or more**, reads `unresolved` between +0.05 and that,
and `fail` below +0.05. **The per-list leg has no interval**: it is a sign read
on a point estimate, HELDOUT is the list it is most likely to turn on
(exploratory +0.0511 at n = 61 and +0.0390 at n = 80, against +0.07 to +0.09 on
the other two), and a HELDOUT Δ of +0.001 passes it. That weakness is known and
left, because a per-list +0.05 bar on 120 draws would be decided by noise.
**P2 is also the quantity the rule selects on**: wherever the subject tier is the
whole pool it is a maximum over a superset of the shipped tier. A pass says the
rule did what it was built to do. It is measured over the *draft* (`n_scored`
characters), including an unclosed tail `trim_to_sentence` removes before the
reader sees the reply.

**P3's power is low, and that is stated before running.** Both picks contain a
definite subject on a tenth to a sixth of draws (126/1250; on the shipped
checkpoint 26/318, 59/393 and 57/599), so expect **some 50–90 of 600 draws** in
the denominator and **a handful of discordant ones** (2, 2 and 5 on the shipped
checkpoint's three subsets). The smallest count that can pass is **6 fixed against 0 broken**
(p = 0.03125); 5 against 0 is p = 0.0625 and is `unresolved`. **`unresolved` is
the modal outcome for P3 and is not a disappointment**: on the pools already
read, the rate of unintroduced heads per definite subject did not move (0.6329
against 0.6547), and the honest reading of that is that the rule does not change
how carefully the model's text introduces things.

**P1's power is lower still.** `anchored_topic` holds on about 2 % of picks
(11/600 on the shipped checkpoint at λ = 0) and the shipped checkpoint shows no
discordant draw on any exploratory subset. It is a trip-wire that one draw can
spring: on training seed `-s3` at λ = 0 it reads shipped-only 1, subject-only 0,
which is a `fail`. A P1 `pass` is the absence of measured damage on a rare event
and **must never be quoted as evidence that replies are more on topic.**
`anchored_topic` is not windowed; mean reply length is printed beside it.

**GUARD 2's bar, and why 0.10.** distinct-2 has no interval worth the name. The
yardstick is its spread across the four training seeds of the shipped recipe
under the *shipped* selector at a matched 600 draws: 0.2153, 0.2090, 0.2102,
0.2138 (λ = 0 rows of `by_ckpt`), a full range of about 3 % of the mean. A
relative fall of 0.10 is more than three times that range, so a fall that size
is the rule and not the seed. It was **proposed after the exploratory fall of
about a third had been seen, and it was not placed to be passed.** distinct-2
falls as text grows, so the confirmatory figure is comparable only with the two
selections of its own 600 draws and with the `by_ckpt` rows — never with the
pooled 0.1485 → 0.0998.

## 4. The expected outcome, written before the run

| bar | on the pools already read (shipped checkpoint, §1) | expected on the confirmatory pools |
| --- | --- | --- |
| P1 | pass, 0 against 0 | **pass by tie**, decided by 0–2 draws |
| P2 | pass (+0.0797, lower bound +0.0694) | **pass**; above +0.0797 if §1's bias argument holds |
| P3 | unresolved (2 against 0; 5 against 0 at λ = 0) | **unresolved** |
| GUARD 1 | not measurable on story-only pools | **pass**; it is an identity the code was built to satisfy |
| GUARD 2 | **fail** (0.3006; 0.3283 pooled over four seeds) | **fail** |
| OVERALL | | **`fail`, through GUARD 2. The default is expected to stay OFF.** |

This is a pre-registration whose author expects its own rule to fail its own
guard. That is deliberate and is the point of having the guard. The run is still
specified, for three reasons: the confirmed configuration drafts 400 characters
where the read one drafted 300, and how sameness moves with length has not been
measured; every bar above would be read for the first time on pools nobody has
seen; and §7's instrument lands whatever the verdict. **Whether that is worth a
quarter of an hour of GPU is the owner's call, and drawing nothing is a
legitimate answer.**

**If the owner would accept a larger sameness cost than 0.10 in exchange for
fewer garbled replies, that is a change to GUARD 2's bar and it must be made
here, as a dated amendment with a new stamp (§9) and the matching change to
`score_v14.GUARD2_MAX_DISTINCT2_DROP`, BEFORE any seed 6–11 pool is drawn.**
After the first pool exists the bar is what it is.

## 5. The clause set: scored, outside the bars

Every committed probe list is bare "story about a X", on which the subject
extractor cannot be wrong, so without `--set clause` the run is blind to it:
six requests where `tier_subject` cuts a modifying clause and keeps the noun
("a penguin who loves fish"), six where it declines ("a penguin and make it
funny"). `score_v14.py` reports the 72 draws as their own subset with the
verdict each bar *would* read, **outside `overall`**: P2's lists were fixed
before the set existed, and 36 draws with a subject resolve nothing. The six
declined prompts are subject-less draws and are inside GUARD 1.

## 6. Reported always, gating nothing

* **Turn latency, re-measured.** The subject rule runs the null pass on draws
  where the shipped rule skipped it: on the shipped checkpoint's committed pools
  the shipped tier is a single draft on 0.4700 (282/600) of draws and the
  subject tier on 0.1250 (75/600), so the null pass goes from running on 318 of
  600 turns to 525. `v14_pools.py` times each selector alone, per draw, on a
  cleared copy of the pool; the scorer reports `mean_select_seconds_shipped`,
  `mean_select_seconds_subject` and `mean_rerank_seconds` per pool with the draw
  count. The committed reference is **0.70 s and 2.21 GiB a turn** at n = 256,
  400 characters (`scripts/chat.py`, the comment above `DEFAULT_RERANK_N`), and
  the rule that picked n was "a REPL should answer in under two seconds". No
  bar; a subject-rule turn over two seconds is reported in `QUALITY_v14.md` as a
  cost the owner weighs. `timing_pass_disagreements` must be 0 or is reported.
* **The all-draws UER count** with each arm's `uer_decomposition` and the corpus
  reference beside it, and the prompt-as-context variant. **Not a bar**: it
  moves with exposure (§1).
* **`topic_mention`**, the bare-mention column, as the identity it is, with
  `identity_violations`, which must read 0.
* **Replies below −0.3934**, as counts. That figure is a battery *mean* for
  plain sampling (`QUALITY.md` §4) used as a per-reply line it was never
  calibrated to be.
* **The top 60-character opening's share**, per arm, with denominators; tier
  sizes; the share of draws where no draft names the subject; every per-list
  row.

## 7. The instrument's own bars — entry conditions, and they are met

The instrument is the **unintroduced-entity rate (UER@200)**. It is never called
"coherence": it has not been validated against a human rating. Regenerated today
with command C, whose output is **byte-identical** to the committed
`coherence_v14.json` (sha256 `68fc1eaa…9e71b`):

1. **The corpus reference reproduces the prototype's no-stoplist reading.** The
   prototype's rule, transcribed literally, reads **0.0861 (157/1824)**, CI
   [0.0741, 0.0998], on `stories_topic.val.bin` — the prototype's figure
   (commit `d40325e`). The instrument without its stoplist reads 0.0839
   (153/1824), and with it, the reference every model rate is read against:
   **0.0230 (42/1824)**, CI [0.0171, 0.0310].
2. **The replace-alternate-sentences mutation raises UER**: 0.0230 → **0.2336
   (426/1824)**, CI [0.2147, 0.2535]. Held on the committed fixture by
   `test_replace_alternate_sentences_raises_uer_by_a_clear_margin`, which
   commit `8bbbd44` records as driven red by a mutation before it landed.
3. **The plain-interleave blindness is documented, as a number**: A1, B1, A2,
   B2 reads **0.0208 (38/1824)** — *below* the clean corpus. The instrument
   cannot see it, and
   `test_plain_interleave_is_not_detected_known_blindness` pins that.
4. **A reply with no definite subject scores clean** ("WHAT IT CANNOT SEE" item
   8), which is why P3 is read where both picks have one.

The confirmatory artifact records the sha256 of the `coherence_v14.json` its
corpus reference came from; it must be the one above. **UER never enters
`snnchat/rerank.py`** (`CONVENTIONS.md` §7).

## 8. The decision rule

* **OVERALL `pass`** — P1, P2, P3 fire and both guards hold: the default flips
  **on the branch `chat/v14-subject-tier`**, in both places `scripts/chat.py`
  constructs `RerankParams` — `run_chat` (lines 473–477) and the `/rerank`
  command (lines 650–653), where it is carried the way `echo` is so that
  rebuilding the object cannot silently re-default it. **Landing it on the
  owner's main tree is his call**: his files there are dirty with work this
  round has not read.
* **OVERALL `unresolved`** — the default stays OFF. The flag, the instrument,
  the scorer and the pools still land on the branch. An unresolved margin does
  not displace an incumbent, the rule every round since v4 has used.
* **OVERALL `fail`** — the default stays OFF and `QUALITY_v14.md` says which bar
  fired, as it fired. The flag stays in the code, off, because off is the old
  path exactly and the evidence scripts depend on it.
* Under every outcome `experiments/chat/SHIPPED` is untouched, the verdict is
  about the shipped checkpoint's first turn on bare story requests, and
  `QUALITY_v14.md` does not use the words "more intelligent" or "coherence".

## 9. Stopping rules, and the stamp

1. One checkpoint, six sampler seeds, five sets, decided now and drawn whatever
   the first pool reads. The scorer is run **once**, on all five pools together.
   No pool is scored alone for a verdict; the smoke check in the runbook calls
   `load_file` only, which replays and computes no bar.
2. The prompt lists, `CLAUSE`, the bars and `IN_OVERALL` are frozen at commit
   `7a1e0d0`. A change to `score_v14.py` or `v14_pools.py` after this file is
   stamped and before scoring is reported in `QUALITY_v14.md` with its diff.
3. Nothing else runs on the GPU while a pool is being drawn: the pools carry
   timings.
4. **The stamp.** `PREDICTION_v13.md`'s recipe, as `score_v13.py` implements it:
   `hashlib.sha256(path.read_bytes()).hexdigest()` on this file, recorded
   outside it before the first pool is drawn and re-checked at scoring. One trap, measured today: v13's
   stamp `027542a6…` reproduces only on CRLF bytes; the committed LF blob of
   that file hashes `019a300a…`. **This file's stamp is taken over the committed
   LF bytes**, so `git show <commit>:docs/chat/PREDICTION_v14.md | sha256sum`
   and a hash of any checkout under this repository's `.gitattributes`
   (`eol=lf`) agree. The stamp is in the message of the commit that adds this
   file. `score_v14.py` does not yet re-check it the way `score_v13.py` does;
   until it does, the runbook checks it by hand.

## 10. Cost

No training. Five pools, 744 draws. A turn at n = 256 and 400 characters is
0.70 s (§6); each draw adds the second selection, a forced null pass and two
timed selections, each at most one teacher-forced pass over the pool, so **about
a second a draw: two to three minutes for each 120-draw list, six to eight for
`WIDE`, one to two for each guard set, a quarter of an hour in all**, plus model
load and graph capture per pool. That is an estimate from the committed turn
time, not a measurement. Scoring is zero-GPU and takes under a minute.

**The runbook.** From the worktree root `C:/Elliot's Stuff/SNN-worktrees/chat-v14`,
on an idle GPU, one pool at a time, after checking this file's stamp (§9). Each
pool is launched detached, because `WIDE` exceeds a few minutes and `launch.py`
is invisible to a harness waiting on it; `--run-name` before the `--` names the
launcher's log directory only, and everything after the `--` is `v14_pools.py`'s:

```
python scripts/launch.py --script scripts/chat/v14_pools.py --run-name v14_pools_<SET> -- --set <SET> --device cuda --ckpt "C:/Elliot's Stuff/SNN/Text SNN/experiments/chat/chat-v3d-aligned/ckpt_best.pt" --out "C:/Elliot's Stuff/SNN-worktrees/chat-v14/experiments/chat/_quality/v14_<SET>_chat-v3d-aligned.json.gz"
```

with `<SET>` = `clause`, then `heldout`, `fresh`, `wide`, `dodge`. Completion is
a line beginning `wrote ` in `experiments/runs/v14_pools_<SET>/stdout.log`; a
`Traceback` there is a crash, and the next pool is not launched until one or the
other appears. After `clause` — the set outside the bars — the smoke check
replays the pool and computes no bar:

```
CUDA_VISIBLE_DEVICES=-1 python -c "import sys; sys.path.insert(0, 'scripts/chat'); from pathlib import Path; import score_v14 as s; info, d = s.load_file(Path('experiments/chat/_quality/v14_clause_chat-v3d-aligned.json.gz'), exploratory=False); print(len(d), info['timing'])"
```

It must print 72 draws and `timing_pass_disagreements: 0` without raising
`ReplayMismatch`. Then, once, on all five pools:

```
CUDA_VISIBLE_DEVICES=-1 python scripts/chat/score_v14.py "experiments/chat/_quality/v14_*_chat-v3d-aligned.json.gz"
```

which writes `experiments/chat/_quality/v14_confirmatory.json`.

## 11. The results section is empty until the pools are scored

Appended to, never rewritten. `QUALITY_v14.md` carries the verdicts.

---

## Appendix: the command behind every number above

All from the worktree root, zero GPU. **Give B a scratch `--out`:
`score_v14.py --exploratory` with no `--out` writes
`experiments/chat/_quality/v14_exploratory.json`, which is right for A and
would overwrite the committed artifact with a three-file one for B.**

**A.** `CUDA_VISIBLE_DEVICES=-1 python scripts/chat/score_v14.py --exploratory`
— §1's table and subset sizes. Re-run on 2026-09-21 into a scratch `--out`:
differs from the committed artifact (sha256 `f55a03e9…14e6`) in `command` only.

**B.** `CUDA_VISIBLE_DEVICES=-1 python scripts/chat/score_v14.py --exploratory
experiments/chat/_quality/v12_heldout_chat-v3d-aligned.json
experiments/chat/_quality/v12_fresh_chat-v3d-aligned.json
experiments/chat/_quality/v12_wide_chat-v3d-aligned.json --out <scratch>.json`
— every "shipped checkpoint alone" number in §1, §3, §4 and §6. The four-seed
distinct-2 row in §3 is `subsets.lam0_all.by_ckpt.*.GUARD2_sameness` of A.

**C.** `CUDA_VISIBLE_DEVICES=-1 python scripts/chat/coherence_score.py
--data-dir "<tree holding data/chat>" --out <scratch>.json` — §7.

**D.** The splits the artifact does not carry, the set sizes and the lattice
check. Save as a file, run with `CUDA_VISIBLE_DEVICES=-1 python <file>`:

```python
import sys
sys.path.insert(0, "scripts/chat")
import score_v14 as s
import v14_pools as vp
from echo_holdout import mcnemar

draws = [d for f in s.EXPLORATORY_FILES
         for d in s.load_file(s.QUALITY / f, exploratory=True)[1]]

def show(tag, sub, label):
    c = s.compare(sub, label, breakdown=False)
    print(tag, len(sub), "changed", c["selection_changed"]["k"],
          "delta %+.4f" % c["P2_logp_per_char"]["pooled"]["delta"],
          "distinct-2 fall %.4f" % c["GUARD2_sameness"]["relative_drop"])

measured = [d for d in draws if d.null_whole_pool]
show("null-measured, lam0", measured, "lam0")
show("excluded, lam0", [d for d in draws if not d.null_whole_pool], "lam0")
show("a draft names the subject", [d for d in measured if d.names_subject], "lam0.6")
show("no draft names it", [d for d in measured if not d.names_subject], "lam0.6")
for name in vp.SET_NAMES:
    print(name, len(vp.probes_for(name)), "prompts")
print("McNemar splits with p == 0.05, n <= 600:",
      sum(1 for n in range(601) for b in range(n + 1)
          if abs(mcnemar(b, n - b) - s.P3_ALPHA) < 1e-12))
print("6 v 0:", mcnemar(0, 6), " 5 v 0:", mcnemar(0, 5))
```

---

## 12. Amendment A, 2026-09-21 — made before any seed 6–11 pool exists

**Appended, not merged in.** §§0–11 and the Appendix above are byte-for-byte
what commit `8e5daad` stamped; this section says which of their sentences it
supersedes and leaves them standing as the record of what was registered first.
It exists because §4 said how a change to a bar may be made — "as a dated
amendment with a new stamp (§9) and the matching change to
`score_v14.GUARD2_MAX_DISTINCT2_DROP`, BEFORE any seed 6–11 pool is drawn" — and
two rulings by the owner arrived today.

### 12.1 No confirmatory pool exists

Taken in the worktree `C:/Elliot's Stuff/SNN-worktrees/chat-v14` on 2026-09-21
at 18:37 EDT, before any edit of this amendment, and taken again immediately
before the commit that adds this section (that commit's timestamp is the bound):

```
$ ls experiments/chat/_quality | grep -i v14
coherence_v14.json
v14_exploratory.json
$ git log --all --oneline -- 'experiments/chat/_quality/v14_*.json.gz'
$ find "C:/Elliot's Stuff/SNN-worktrees" "C:/Elliot's Stuff/SNN/Text SNN/experiments" -name "v14_*.json.gz"
$ git log --oneline -3          # at 18:37
8e5daad The v14 subject tier is pre-registered before any seed 6-11 pool exists, and it expects to fail its own sameness guard
69da628 Training rounds now report pool UER@200 with its qualifying fraction next to bpc, and UER may never enter rerank.py
7a1e0d0 A v14 pool is drawn at the REPL's 400 characters, and a clause set lets the confirmatory run see the subject extractor
```

The second and third commands print nothing: no `v14_*.json.gz` is in any
branch's history, in any worktree, or under the main tree's `experiments/`. The
listing taken again at 18:55 differs by one line,
`v14_exploratory_chat-v6-scratch.json`: the zero-GPU replay of seeds 0–5 that
§12.6 describes, a scorer artifact and not a pool. This
worktree has no `experiments/runs/` directory, so `launch.py` has never started
a pool from it, and nothing run while writing this amendment touched the GPU
(every scorer and test command was run with `CUDA_VISIBLE_DEVICES=-1`). Between
`8e5daad` and the commit that adds this section the branch gained `a45b2a4` (a
merge of `chat/v6-scratch-diagnostic`, new files only, which brings the pools
§12.6 reads) and `a611d99` (the REPL toggle, §12.8).

### 12.2 The two rulings, verbatim

Both were asked of the owner, Elliot, by the lead agent of the working session
of 2026-09-21, and answered by him in that session. Provenance, as the agent
writing this amendment has it: RULING 1's answer reached it as the owner's own
relayed message; RULING 2's reached it through the lead's work order and is
corroborated by `SHIPPED`'s header, which records "Elliot's explicit \"Yes\""
at `a538bd3`.

**RULING 1 (sameness).** Asked "how much extra repetitiveness you would accept
from the new reply picker (v14) before I run its 15-minute confirmation", he
answered:

> As many as needed

Sameness is therefore **not a bar for him**. §4 anticipated a *larger* bar; the
answer sets none, so GUARD 2 is withdrawn as a bar rather than moved.

**RULING 2 (the ship).** Asked whether to make `chat-v6-scratch` the model he
talks to, he answered:

> Yes

`experiments/chat/SHIPPED` in the owner's main tree now names
`chat-v6-scratch/ckpt_best.pt` (commit `a538bd3` on `phase-5-regional`; the
evidence is `docs/chat/V6_SCRATCH_NOTE.md`, commit `d802e85`). **This round did
not edit `SHIPPED` and still does not**; the owner's ruling did, separately.
§2 pinned the confirmation to `chat-v3d-aligned` because that was the ship when
§2 was written. The default this round decides about now applies to a different
model, so the confirmation is drawn on both (§12.5).

**WHAT RULING 1 DOES NOT DO.** P1, P2, P3 and GUARD 1 are untouched: the same
constants, the same verdict functions, the same draws, byte-identical in
`score_v14.py` to commit `8e5daad` (the amendment's commit message lists every
hunk of the diff). **P3 was expected `unresolved` in §4 and still is**, so on
`chat-v3d-aligned` the expected OVERALL moves from `fail` to `unresolved` — and
`unresolved` does not flip the default (§3, §8). **The owner's ruling removes a
bar; it does not rescue the rule.** On `chat-v6-scratch` the pools already read
say something worse for the rule than that (§12.6).

### 12.3 What is superseded, sentence by sentence

Everything in §§0–11 and the Appendix that is not quoted here stands.

**§0.** "`RerankParams.subject_tier` (`src/snnchat/rerank.py`), **default OFF**,
no CLI flag." → Default OFF stands. There is now a flag and a command, both off
unless asked for (§12.8).

**§2**, the `checkpoint` row: "`chat-v3d-aligned/ckpt_best.pt`, the one
`experiments/chat/SHIPPED` names; sha256 `3deca3eb…`" → **two checkpoints**,
§12.5. `chat-v3d-aligned` is no longer the one `SHIPPED` names.

**§2.** "**The sample size is fixed and honoured** …: 600 story draws, 72 dodge,
72 clause, no more and no fewer." → the same, **per checkpoint**: 744 draws from
each, 1,488 in all, no more and no fewer.

**§2.** "One checkpoint: training-seed variability is not in any interval below,
and the verdict is about this checkpoint's default, not the recipe." → Two
checkpoints, each scored alone. They are not two seeds of one recipe:
`experiments/chat/SHIPPED`'s history describes `chat-v6-scratch` as "a
fresh-init 84,000-step run on this checkpoint's exact recipe (no --init-from)".
One training seed each, so training-seed variability is still in no interval and
each verdict is about one file.

**§3.** "**These are the bars `score_v14.py` implements at commit `7a1e0d0`**"
→ at the commit that adds this section; P1, P2, P3 and GUARD 1 are unchanged
from `7a1e0d0`.

**§3**, the GUARD 2 blockquote: "Relative fall in distinct-2 … <= 0.10 pooled
and on each of HELDOUT, FRESH and WIDE. Pooled fall > 0.10 is `fail`; pooled
within the bar but a list over it, or missing, is `unresolved`." → replaced by:

> **GUARD 2 (sameness) — REPORTED, NOT A BAR.** Measured exactly as before:
> distinct-2 of the shipped picks and of the subject-tier picks of the same 600
> draws with both numerators and denominators, the relative fall
> `1 − subject/shipped` pooled and on each of HELDOUT, FRESH and WIDE, and each
> arm's top 60-character opening with its share and denominator. Its verdict
> field reads `reported` and never `pass`, `fail` or `unresolved`. Beside it the
> artifact carries `withdrawn_bar: 0.10` and `would_have_read`: what the
> blockquote above would have returned, computed by the same function
> (`score_v14.guard2_verdict`, unchanged). Nothing is hidden and nothing gates.

**§3**, the OVERALL blockquote: "`fail` if any of P1, P2, P3, GUARD 1, GUARD 2
is `fail`; otherwise `unresolved` if any is `unresolved`; otherwise `pass`." →

> **OVERALL (amended).** `fail` if any of **P1, P2, P3, GUARD 1** is `fail`;
> otherwise `unresolved` if any is `unresolved`; otherwise `pass`. Computed per
> checkpoint. `score_v14.IN_OVERALL` is exactly those four.

**§3**, the lattice bullet "**GUARD 2**'s statistic is a ratio of two ratios …
the rule is `> 0.10` fails, so a fall of exactly 0.10 passes." and the paragraph
"**GUARD 2's bar, and why 0.10.**" → both now describe the *withdrawn* bar, i.e.
what `would_have_read` computes. They stand as that description.

**§4**, the last two table rows: "GUARD 2 | **fail** (0.3006; 0.3283 pooled over
four seeds) | **fail**" and "OVERALL | | **`fail`, through GUARD 2. The default
is expected to stay OFF.**" → §12.6.

**§4.** "This is a pre-registration whose author expects its own rule to fail
its own guard. That is deliberate and is the point of having the guard." → The
guard was the author's and the owner has withdrawn it, which is his to do. The
author now expects the rule **not to pass for a different reason**: P3 (§12.6).

**§4.** "**Whether that is worth a quarter of an hour of GPU is the owner's
call**…" → about half an hour: two checkpoints (§12.9).

**§8.** "**OVERALL `pass`** — P1, P2, P3 fire and both guards hold: the default
flips **on the branch `chat/v14-subject-tier`**, in both places
`scripts/chat.py` constructs `RerankParams` — `run_chat` (lines 473–477) and the
`/rerank` command (lines 650–653)…" → §12.7. Both places already carry the field
since `a611d99`, so the flip, if it happens, is a change to one default in
`build_parser`.

**§8.** "Under every outcome `experiments/chat/SHIPPED` is untouched, the
verdict is about the shipped checkpoint's first turn on bare story requests" →
`SHIPPED` is untouched *by this round* under every outcome; there are two
verdicts, each about one checkpoint's first turn on bare story requests.

**§9 item 1.** "One checkpoint, six sampler seeds, five sets, decided now and
drawn whatever the first pool reads. The scorer is run **once**, on all five
pools together." → **Two checkpoints**, six sampler seeds, five sets each, all
ten pools drawn whatever the first reads. The scorer is run **once per
checkpoint**, on that checkpoint's five pools together, and refuses a file list
whose pools carry more than one `ckpt_sha256`.

**§9 item 2.** "The prompt lists, `CLAUSE`, the bars and `IN_OVERALL` are frozen
at commit `7a1e0d0`." → The prompt lists, `CLAUSE` and the bars P1, P2, P3 and
GUARD 1 are as frozen at `7a1e0d0`; `IN_OVERALL` is frozen at the commit that
adds this section. The rest of the item stands and applies from that commit.

**§9 item 4.** "The stamp is in the message of the commit that adds this file.
`score_v14.py` does not yet re-check it the way `score_v13.py` does; until it
does, the runbook checks it by hand." → §12.10. It does now.

**§10.** "Five pools, 744 draws. … a quarter of an hour in all" and the whole of
"**The runbook.**", including "which writes
`experiments/chat/_quality/v14_confirmatory.json`" → ten pools, 1,488 draws,
about half an hour, and the runbook in §12.9. **No file named
`v14_confirmatory.json` is written**; each artifact names its checkpoint.

**§11.** "Appended to, never rewritten. `QUALITY_v14.md` carries the verdicts."
→ **§11 stays empty, permanently.** Results, verdicts and any deviation go in a
separate file, `docs/chat/QUALITY_v14.md`. This file is not edited after the
commit that adds this section, so its stamp stays valid forever, and a scorer
that finds it changed refuses to score.

**Appendix, command A.** "differs from the committed artifact … in `command`
only" → true of the scorer at `8e5daad`. Under the amended scorer the same
command, written to a scratch `--out`, differs from the committed
`v14_exploratory.json` (sha256 `f55a03e9…14e6`, **not regenerated and not to be
overwritten**) in `command`, `bars`, `in_overall`, the new `prediction` and
`checkpoint` keys, and in every `GUARD2_sameness` block's `rule`, `verdict`,
`why`, `withdrawn_bar` and `would_have_read` — and in **no measured number**
(command F, 127 differing leaves, all of those shapes).

### 12.4 No bar sits on a lattice point — re-checked

The amendment adds no bar and moves none, so §3's check carries over; it was
re-run today rather than assumed (command F). **P1** compares two integers and
its tie is defined. **P2**'s +0.05 is not a distinguished value of a mean of 600
floating-point differences, and its per-list leg is a strict `> 0`. **P3**'s
exact McNemar p is dyadic: enumerated again over every split of n <= 600
discordant draws, **0** give p = 0.05. **GUARD 1** is an identity over 108
draws. GUARD 2 no longer has a bar to sit anywhere. Nothing about a second
checkpoint changes a denominator: every bar is read per checkpoint on the same
600 + 72 + 72 draws.

### 12.5 The two-checkpoint design

| | `chat-v3d-aligned` | `chat-v6-scratch` |
| --- | --- | --- |
| file | `experiments/chat/chat-v3d-aligned/ckpt_best.pt` | `experiments/chat/chat-v6-scratch/ckpt_best.pt` |
| sha256 | `3deca3eb0265925e2471b5cb3ace4c800e5dc1f116df037a2699af8ba49ac86d` | `e270b3a3c878b9253f92cb8d22f69f1e27f11ea2cc81b3e0854426929c62676f` |
| why | exactly as §2 registered it: the original pre-registration's own reading | the checkpoint `SHIPPED` names since RULING 2; the REPL loads this |
| role | **reported beside** | **decides the default** (§12.7) |
| artifact | `v14_confirmatory_chat-v3d-aligned.json` | `v14_confirmatory_chat-v6-scratch.json` |

Both sha256 values were computed today from the files in the owner's main tree
(`sha256sum`); `v14_pools.py` records the hash in every pool header and
`score_v14.REGISTERED_CHECKPOINTS` holds the same two. Everything else in §2 is
**identical for both**: the five sets, sampler seeds 6–11, n = 256, λ = 0.6, 400
drafted characters, the pairing, the bars. No pool of these five sets has been
drawn from either checkpoint at a sampler seed above 5: every committed `v12_*`
and `v6scratch_*` story and dodge pool is `--seeds 6`, seeds 0–5
(`v6scratch_memory_probe.json` used 8 seeds, on different prompts, which are
different draws). `v14_pools.py` needed no change: it takes `--ckpt` and names
its output after the run.

Each checkpoint is scored **separately**. In confirmatory mode the scorer
refuses a file list whose pools carry more than one `ckpt_sha256` (a wildcard
like `v14_*.json.gz` would otherwise pool two models into one verdict), names
the checkpoint in its default output path, and says in the artifact
(`checkpoint`) whether the pools' hash is one of the two above and whether it is
the deciding one.

### 12.6 The expected outcome, amended, written before the run

**`chat-v6-scratch`'s seeds 0–5 pools are now READ.** They exist in
`echo_holdout.py` format (n = 256, λ = 0.6, 300 drafted characters; the three
story lists, 600 draws) as `experiments/chat/_quality/v6scratch_{heldout,fresh,wide}_n256.json.gz`
(commit `d802e85`; the decompressed bytes equal the raw JSON backups by sha256),
and the subject rule was replayed on them today (command E) into
`experiments/chat/_quality/v14_exploratory_chat-v6-scratch.json`. As in §1,
**none of this is a test of the rule**, the subsets are selected on the shipped
rule's behaviour, and 300 characters is not the confirmed 400. The confirmation
uses seeds 6–11, which are not read. `chat-v3d-aligned`'s column was regenerated
with command B and agrees with §1.

| exploratory, seeds 0–5 | `chat-v3d-aligned` (command B) | `chat-v6-scratch` (command E) |
| --- | --- | --- |
| `lam0.6_null_measured` / `lam0.6_replayable` / `lam0_all` | 318 / 393 / 600 of 600 | **371 / 449 / 600** of 600 |
| no draft names the subject | 0.7400 (444/600), CI [0.7035, 0.7735] | 0.6100 (366/600), CI [0.5704, 0.6482] |
| P1, shipped-only v subject-only | 0 v 0 on all three subsets: pass by tie | 0 v 0 on all three subsets: pass by tie (16/371 anchored under both rules) |
| P2 Δ, null-measured | +0.0797, SE by prompt 0.0053, lower bound +0.0694: pass | **+0.0699**, SE by prompt 0.0059, lower bound +0.0583: pass |
| P2 Δ, replayable | +0.0644, lower bound +0.0541: pass | **+0.0575**, lower bound +0.0467: **unresolved** |
| P2 Δ, λ = 0, all 600 | +0.1056: pass | +0.0918, lower bound +0.0795: pass |
| P2 Δ on HELDOUT (the weak list), same three subsets | +0.0511 (n = 61), +0.0390 (80), +0.0806 (120) | +0.0379 (n = 88), +0.0333 (100), +0.0531 (120) |
| P3, both picks exposed | 26/318, 59/393, 57/599 | 65/370, 109/448, 105/599 |
| P3, fixed v broken | 2 v 0, 2 v 0, 5 v 0: **unresolved** ×3 | **0 v 3, 1 v 3, 2 v 3: `fail` ×3** |
| GUARD 2, fall in distinct-2 | 0.3006, 0.2209, 0.3802 | 0.2124, 0.1652, 0.3131 |
| … per list, null-measured (heldout / fresh / wide) | 0.1439 / 0.2743 / 0.3162 | 0.0709 / 0.2266 / 0.2218 |
| … top 60-char opening, null-measured | 0.0409 (13/318) → 0.1132 (36/318) | 0.0404 (15/371) → 0.0512 (19/371) |
| … the withdrawn 0.10 bar would have read | `fail` ×3 | `fail` ×3 |
| replies below −0.3934, null-measured | 0.0943 (30/318) → 0.0283 (9/318) | 0.1590 (59/371) → 0.0485 (18/371) |

| bar | expected on `chat-v3d-aligned`, seeds 6–11 | expected on `chat-v6-scratch`, seeds 6–11 |
| --- | --- | --- |
| P1 | **pass by tie**, decided by 0–2 draws (§4, unchanged) | **pass by tie**, decided by 0–2 draws |
| P2 | **pass** (§4, unchanged) | **pass, narrowly**; `unresolved` is a live outcome. The read effect is smaller (+0.0575 to +0.0918 against +0.0644 to +0.1056) and P2 resolves `pass` only from about +0.062 (below) |
| P3 | **unresolved** (§4, unchanged) | **`fail`**, as it reads on all three subsets already seen; `unresolved` is the other live outcome; `pass` is not expected |
| GUARD 1 | **pass** (§4, unchanged) | **pass**; an identity of the code, not of the checkpoint |
| GUARD 2 | **reported**; a fall of 0.2–0.4, which the withdrawn bar would have read `fail` | **reported**; a fall of 0.15–0.3, which the withdrawn bar would have read `fail` |
| **OVERALL** | **`unresolved`**, through P3 | **`fail`, through P3, or `unresolved`. Not `pass`.** |

**The default is expected to STAY OFF.** On the deciding checkpoint the rule is
expected to read `fail` or `unresolved`, and only `pass` flips it (§12.7). The
author expects the two checkpoints to **disagree** — `unresolved` against `fail`
— and §12.7 says what is done with that: it is stated, not resolved. The owner's
way to use the rule anyway, whatever either verdict reads, is the toggle
(§12.8); that is his call and needs no verdict.

**Power, per `CONVENTIONS.md` §4 rule 3, per checkpoint.**

* **P2.** `pass` needs `Δ − 1.96·max(SE) >= +0.05`. With the by-prompt SE read
  at 0.0053–0.0055 (`chat-v3d-aligned`) and 0.0055–0.0063 (`chat-v6-scratch`)
  over ~100 prompts, that is a pooled Δ of about **+0.061** and **+0.062** or
  more; between +0.05 and that it reads `unresolved`; below +0.05, `fail`. The
  per-list leg is still a sign read with no interval, and on `chat-v6-scratch`
  HELDOUT is closer to zero than it was on `chat-v3d-aligned`.
* **P3.** The denominator is the draws where both picks have a definite subject:
  expect **some 50–90 of 600** on `chat-v3d-aligned` (§3) and **some 100–150 of
  600** on `chat-v6-scratch` (0.1757 (65/370), 0.2433 (109/448), 0.1753
  (105/599)), with **a handful of discordant draws on either** (2, 2, 5 and 3,
  4, 5). The smallest counts that can pass are **6 fixed v 0 broken**
  (p = 0.03125), **8 v 1** (p = 0.03906), **10 v 2** (p = 0.03857) and **12 v
  3** (p = 0.03516); 5 v 0, 7 v 1, 9 v 2 and 11 v 3 are `unresolved` (command
  F). With three broken draws already seen on `chat-v6-scratch` in each read
  subset, a pass there would need about twelve fixed, against the 0–2 seen.
  **P3's `fail` side has no interval**, and that is as registered in §3 and is
  not changed here: `fixed <= broken` fails at any count — 0 v 3 (p = 0.25)
  fails, and so does 0 v 0 when at least one draw has both picks exposed
  (command F). A `fail` on three discordant draws is a fired bar and is
  reported as one; it is weak evidence that the rule makes introduction
  *worse*, and `QUALITY_v14.md` may not say more than the counts do.
* **P1** and **GUARD 1**: as §3. `anchored_topic` holds on 0.0431 (16/371) of
  `chat-v6-scratch`'s read picks under both rules, and one discordant draw
  decides P1 either way.

### 12.7 The amended decision rule

* **The default flips on the branch `chat/v14-subject-tier` only if OVERALL =
  `pass` on the checkpoint `experiments/chat/SHIPPED` names,
  `chat-v6-scratch`** (sha256 `e270b3a3…2676f`), with OVERALL as amended in
  §12.3. Landing it on the owner's main tree is still his call (§8).
* `chat-v3d-aligned`'s verdict is **reported beside it**, as the original
  pre-registration's own reading. It flips nothing.
* **A disagreement between the two is stated as a disagreement.** It is not
  resolved by choosing the more favourable checkpoint, by pooling the two, or by
  drawing more seeds. `QUALITY_v14.md` reports both OVERALLs, each bar on each
  checkpoint, and says in words that they differ.
* OVERALL `unresolved` or `fail` on `chat-v6-scratch`: the default stays OFF,
  exactly as §8 says for each, whatever `chat-v3d-aligned` reads — including
  `pass`.
* The deciding checkpoint is fixed here, by hash, and is not looked up at
  scoring time. If `SHIPPED` names anything else by the time the pools are
  scored, **no verdict of this round flips the default**: a `pass` would then be
  about a model the REPL no longer loads, and what to do with it is the owner's
  ruling.
* Everything else in §8 stands, including that `QUALITY_v14.md` does not use the
  words "more intelligent" or "coherence".

### 12.8 The toggle (commit `a611d99`)

`python scripts/chat.py --subject-tier`, and `/subject on|off` inside the REPL.
**Default OFF, and off nests the old behaviour exactly**
(`tests/test_snnchat.py::test_the_subject_tier_flag_is_off_by_default_and_reaches_the_session`).
Both places `scripts/chat.py` constructs `RerankParams` carry the field, so
`/subject on` followed by `/rerank 64` stays on
(`test_subject_on_survives_a_rebuilt_rerank`) — the trap `/rerank` once fell
into with `/echo off`. The toggle is outside every bar. It lets the owner try
the rule on the model he talks to; nothing he sees there is a measurement.

### 12.9 Cost, and the runbook that replaces §10's

Ten pools, 1,488 draws. §10's arithmetic, doubled: **about a quarter of an hour
per checkpoint, half an hour in all**, plus model load and graph capture for
each of ten pools. An estimate from the committed 0.70 s turn
(`scripts/chat.py`, the comment above `DEFAULT_RERANK_N`), not a measurement,
and that turn time was measured on `chat-v3d-aligned`; the two checkpoints share
an architecture. Scoring is zero-GPU and under a minute a checkpoint (§10).

From the worktree root, on an **idle** GPU, one pool at a time, with `<RUN>` =
`chat-v3d-aligned` first and then `chat-v6-scratch`, and within each `<SET>` =
`clause`, then `heldout`, `fresh`, `wide`, `dodge`:

```
python scripts/launch.py --script scripts/chat/v14_pools.py --run-name v14_pools_<SET>_<RUN> -- --set <SET> --device cuda --ckpt "C:/Elliot's Stuff/SNN/Text SNN/experiments/chat/<RUN>/ckpt_best.pt" --out "C:/Elliot's Stuff/SNN-worktrees/chat-v14/experiments/chat/_quality/v14_<SET>_<RUN>.json.gz"
```

Completion is a line beginning `wrote ` in
`experiments/runs/v14_pools_<SET>_<RUN>/stdout.log`; a `Traceback` there is a
crash; the next pool is not launched until one or the other appears. After each
checkpoint's `clause` pool, §10's smoke check with that pool's file name
(`load_file` only: it replays and computes no bar); it must print 72 draws and
`timing_pass_disagreements: 0`. Then, once per checkpoint, quoted so that the
scorer and not the shell expands the wildcard:

```
CUDA_VISIBLE_DEVICES=-1 python scripts/chat/score_v14.py "experiments/chat/_quality/v14_*_chat-v3d-aligned.json.gz"
CUDA_VISIBLE_DEVICES=-1 python scripts/chat/score_v14.py "experiments/chat/_quality/v14_*_chat-v6-scratch.json.gz"
```

Each checks this file's stamp first and refuses on a mismatch. All ten pools are
drawn before either scoring command is run, so that no verdict is known while a
pool is still to be drawn.

### 12.10 The stamp

* **Old stamp**, over §§0–11 and the Appendix as committed at `8e5daad`:
  `b92f1af6938b05a33826e1cd1c277fd75afba0b29bd4e4b87c0deea89494ecf9`.
  Re-derived today: `git show 8e5daad:docs/chat/PREDICTION_v14.md | sha256sum`.
* **New stamp**, over this WHOLE file as amended. A file cannot contain its own
  hash, so it is recorded outside it, in two places: `score_v14.PREDICTION_STAMP`
  and the message of the commit that adds this section — one commit, document
  and scorer together.
* **Recipe.** sha256 of the file's bytes **with CRLF normalised to LF**
  (`score_v14.prediction_stamp`). On the committed LF blob that is the plain
  hash, so `git show <that commit>:docs/chat/PREDICTION_v14.md | sha256sum`
  must print the new stamp, and a CRLF checkout agrees with it; `score_v13.py`'s
  raw-bytes recipe does not have that property (§9 item 4).
* **The check.** In confirmatory mode `score_v14.py` refuses to score unless
  this file hashes to `PREDICTION_STAMP`
  (`tests/test_snnchat_v14_scoring.py::test_confirmatory_mode_scores_nothing_against_an_unstamped_pre_registration`).
  The check runs before anything is read. Because results go to
  `QUALITY_v14.md` and §11 stays empty, this file never legitimately changes
  again. A further amendment, if one is ever needed, is legitimate only before
  the first seed 6–11 pool exists, as a further dated section with a new stamp.

### 12.11 The commands behind §12's numbers

All from the worktree root, zero GPU.

**B** (Appendix), re-run today into a scratch `--out`: the `chat-v3d-aligned`
columns.

**E.** `CUDA_VISIBLE_DEVICES=-1 python scripts/chat/score_v14.py --exploratory
experiments/chat/_quality/v6scratch_heldout_n256.json.gz
experiments/chat/_quality/v6scratch_fresh_n256.json.gz
experiments/chat/_quality/v6scratch_wide_n256.json.gz --out
experiments/chat/_quality/v14_exploratory_chat-v6-scratch.json` — the
`chat-v6-scratch` columns. Its checks came back clean: 153,600 candidates, 0
`n_scored`, pick or stored-weight mismatches, 0 draws not replayable under the
shipped rule. Re-run it into a scratch `--out`, not over the committed artifact.

**F.** The lattice and the P3 pass counts, and the diff of command A against the
committed artifact:

```python
import sys
sys.path.insert(0, "scripts/chat")
import score_v14 as s
from echo_holdout import mcnemar

print(sum(1 for n in range(601) for b in range(n + 1)
          if abs(mcnemar(b, n - b) - s.P3_ALPHA) < 1e-12))
for fixed, broken in ((6, 0), (5, 0), (8, 1), (7, 1), (10, 2), (9, 2), (11, 3), (12, 3)):
    p = mcnemar(broken, fixed)
    print(fixed, broken, round(p, 5), s.p3_verdict(fixed, broken, p, 100)["verdict"])
print(s.p3_verdict(0, 0, mcnemar(0, 0), 100)["verdict"], s.IN_OVERALL)
```

and, for the diff, a recursive comparison of `v14_exploratory.json` with command
A's scratch output that lists every differing key path.
