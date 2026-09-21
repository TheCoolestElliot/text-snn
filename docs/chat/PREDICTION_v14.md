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
