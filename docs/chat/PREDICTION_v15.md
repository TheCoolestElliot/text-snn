# PREDICTION v15 — the recall round

**Written 2026-09-21, before the recall source has been packed for training and
before any arm of this round has been trained.** No checkpoint named in §2
exists at the time of writing. The generator, both probes, the scorer and the
driver were committed before this file (`c64455c` .. `39ce460` on
`chat/v15-recall`); `scripts/chat/session15_driver.py` refuses to start unless
this file and those are committed and unmodified, and records the HEAD and each
file's blob it checked, so the ordering is enforced and not merely claimed.

Not part of the Phase-1..5 research protocol. No number here is a reported
figure. Rates follow [`CONVENTIONS.md`](CONVENTIONS.md).

**This round rules no research decision.** Nothing under `src/snn/`,
`experiments/logs/`, `docs/reports/` or `scripts/exp/` is touched.
`experiments/chat/SHIPPED` and `snnchat.build_corpus.DEFAULT_MIX` are edited
under no outcome (§11).

**Every number below was regenerated on 2026-09-21 from committed files, on the
CPU, by a command named beside it or in the appendix.** Where a number is the
design panel's inference and not a measurement, it says so.

---

## 0. What motivates it, and what the owner would notice

The owner types `my name is elliot`, then `what is my name?`, and the shipped
model answers about **itself**. [`QUALITY_v8.md`](QUALITY_v8.md) §10 measured it
with a control and recorded the shape: *"I don't really have a name. I'm a
spiking neural network."* — the possessive is not resolved, so the question is
routed to the persona's identity intent and not to anything the user said. The
committed floor (`experiments/chat/_quality/memory_probe.json`, from
`scripts/chat/memory_probe.py`) is exactly zero in both arms:

| distance | told | untold (control) |
| ---: | ---: | ---: |
| 0 | 0.0000 (0/80), 95% CI [0.000, 0.046] | 0.0000 (0/80), 95% CI [0.000, 0.046] |
| 1 | 0/80 | 0/80 |
| 2 | 0/80 | 0/80 |

It is 0/80 already at distance 0, where the fact is inside the 256-character
training window. Item 1, `elliot`, is 0/8 there.

Nothing in the corpus teaches the distinction. `snnchat.persona` has no my-side
template and sits at 0.08 of the shipped mix, where it is learned to 0.229 bpc
(`chat-v3d-aligned/summary.json` → `val.persona`) — so the your-side answer is
the most thoroughly drilled reply the model owns. `QUALITY_v8.md` §10 closes
with *"A corpus that taught the distinction is a cheaper hypothesis than more
reach."* This round is that hypothesis, acted on: a synthetic user-fact recall
source, `snnchat.recall`, at 0.06 of the mix, taken from `alpaca`.

**What the owner would notice if it works**, and only then: inside one
conversation, for a value from the generator's closed lists (`elliot` is in
them on purpose), `what is my name?` is answered `Your name is Elliot.` and
`what is YOUR name?` still gets the persona line. **What he would not notice:**
any change for a name outside the lists (copying an unseen value is a reported
secondary, §9, and is expected to fail), anything across conversations, or any
general gain in coherence. The goal is **closed-vocabulary in-conversation
recall of trained values**, and `snnchat.recall`'s own docstring says what a
pass is: *"a closed-vocabulary routine, not discovered memory."*

The gate is read at the committed probe's plain decoder (n=8). The REPL ships
n=256; that pass is run once per checkpoint and **reported, not gated** (§9), so
"what he sees at the prompt" is a reported column and the verdict is not.

## 1. The honest odds, and the track record they sit on

**Seven straight training-side rounds have lost or come back unresolved**, and
none displaced `SHIPPED`: v4 (`QUALITY_v4.md`), v5, v6, v7 — each recorded in
`experiments/chat/SHIPPED`'s own comments — then v9 (*"the corpus arm failed"*),
v12 (*"the primary bar passed, and the round does not ship"*) and v13 (*"costs
bits and buys nothing visible"*). The closest analogue bound poorly: a source at
0.40 of the mix, with 76 % of its windows carrying the request, moved topic
propensity from 0.0518 to 0.0908 (`QUALITY.md`, the topic-propensity table).

The design panel's odds, **inferred, not measured**, stated before the run:

| outcome | panel's odds |
| --- | ---: |
| ship-candidate on both seeds | ~25–30 % |
| resolved on both seeds, below the ship bars | ~35–40 % |
| floor | ~25–30 % |
| an unseen value copied (S3) | ~0–5 % |

Those odds were given before tier (iii) was implemented as a significance test
and before nine guards per seed were enumerated (§5, §6). Both move the first
row down, not up. **`resolved` is the modal expectation; `unresolved` — a
straddle between the two seeds — is a live outcome and is a verdict.**

The panel's inferred point predictions, for the record and gating nothing:
distance-0 told on trained values 8–40 of 80; distances 1–2 0–15 of 80; unseen
values 0–3 of 80; untold control on the target ≤ 2/80 with visible other-name
confabulation; `what is my name?` stops routing to the persona line.

**Why run it at those odds.** The floor is exactly zero, so a single 40-minute
fine-tune can resolve at n=1 per seed without the sampler-noise headline that
left rounds 7 and 12 unresolved: 6 told-only against 0 is exact McNemar
p = 0.0312. And a clean distance-0 failure on *trained* values is the most
valuable negative available — it would make longer windows, carried state and
the retrieval envelope moot for recall (§10).

## 2. The arm — one variable

Two training seeds of one arm, run **sequentially** by one detached driver.

| | `chat-v15-recall-s0` | `chat-v15-recall-s1` |
| --- | --- | --- |
| seed | 0 | 1 |
| init from | `chat-v2-anneal/ckpt_best.pt` | same |
| mix | **`recall` 0.06, `alpaca` 0.15 → 0.09**, all else as shipped | same |
| everything else | `chat-v3d-aligned/config.json`, field for field | same |

The recipe is **read from** `experiments/chat/chat-v3d-aligned/config.json` by
the driver, not retyped: 14,000 steps, B=160, L=256, lr 5e-4 cosine to 5 %,
warmup 200, wd 0.1, clip 1.0, `bot_loss_weight` 3.0, `align_frac` 0.75,
`align_lookahead` 1024, `twocomp_threshold`, d=1024 × 4 layers, fp32,
`deterministic` false, `eval_every`/`sample_every` 3500. `--init-from` is not a
`ChatConfig` field and so is not in `config.json`; the parent is the one
`session13_driver.COMMON` records, with `--no-resume --budget-minutes 75` from
the same recovered command line.

**The mix, verified against the committed `config.json`:**

| source | shipped | v15 |
| --- | ---: | ---: |
| stories_topic | 0.40 | 0.40 |
| soda | 0.20 | 0.20 |
| alpaca | **0.15** | **0.09** |
| tinystories | 0.09 | 0.09 |
| persona | 0.08 | 0.08 |
| dolly | 0.06 | 0.06 |
| oasst1 | 0.02 | 0.02 |
| recall | — | **0.06** |

The design brief quoted "alpaca .09"; that is the post-donation figure. The
templated share of the mix (persona + recall) rises from 0.08 to 0.14.

**The exact config diff, enforced.** The driver builds the config the child
*will* construct with `train.py`'s own parser and diffs it field by field
against the reference; anything outside
`MAY_DIFFER = {mix, run_name, seed, out_dir, data_dir}` aborts, `check_mix`
requires every non-`alpaca` weight to equal the committed one exactly, and the
`config.json` each run actually writes is diffed again before it is scored.
Today the reference and `ChatConfig` have identical field sets, so no
new-field exemption is in play. For s0 the diff is `data_dir, mix, out_dir,
run_name`; for s1 it adds `seed`.

    python scripts/chat/session15_driver.py --dry-run --out-root <dir outside the main tree> --data-dir <dir>

`data_dir` differs because the packed corpus for this round is a staging
directory outside every repository: the seven shipped `.bin`/`.val.bin` pairs
byte-identical to the ones the incumbents trained on, a **copy** of
`manifest.json`, and `recall.bin`/`recall.val.bin`. Nothing under the main
tree's `data/` is written.

**One known asymmetry.** `REFERENCE` compares s1's per-source bpc with
`chat-v3d-aligned-s1`, which trained with `eval_every`/`sample_every` 2000; both
v15 seeds copy s0's 3500. Neither touches the step-14,000 evaluation the bpc
guard reads.

## 3. The generator — `src/snnchat/recall.py`, packed by `scripts/chat/build_recall.py`

Pure stdlib, deterministic in `(n, seed)`, everything encodable in vocab v1. The
pack this round trains on is the default build, `--conversations 250000 --seed
0`, and it is reproducible to the byte:

    python scripts/chat/build_recall.py --out-dir <an empty dir> --hazard-steps 8000

→ `recall.bin` 35,272,337 characters, sha256
`0c8eaeb60bb1e34cee510330d924afeb279a822473f6db2a7a696b2de2a3b44e`;
`recall.val.bin` 141,655 characters. **The file the lead packs for training must
hash to that value**; a pack made before commit `77fe15e` does not and is stale.

**Six slots, closed lists.** Trained / held-out value counts: `name` 48 / 8,
`pet` 40 / 6, `colour` 12 / 4, `food` 40 / 6, `animal` 40 / 6, and `object` —
the committed probe's "i have a red bicycle" form — which shares the colour
list. `elliot` and `sarah` are trained names. Auxiliary nouns: 8 pet species
(2 held out), 12 objects (3 held out). Within `name`, `pet`, `food` and
`animal` no two trained values share their first two characters; every list is
disjoint from every other; pet species are disjoint from favourite animals, so
the other fact's value in a two-fact reply is unambiguously a swap.
`_check_tables` enforces this at import.

**Held out, and never emitted** (`_assert_clean` raises on every dialogue): the
held-out values and auxiliary nouns as whole words in any turn; 3 statement and
3 question phrasings per slot plus 4 filler turns, reserved for
`memory_probe_v2.py` — 2,687 fully instantiated reserved user turns
(`recall.heldout_phrases()`); and four whole **frames** never trained in any
phrasing: a town and an age (the committed probe's items 7 and 8), a sister, a
sport.

**User turns are lower-case and unpunctuated**, as the owner types and as the
probe asks. The bot capitalises names and pets.

**Five shapes**, shares pinned in `SHAPE_SHARES` and realised on the build as:

| shape | target | realised | mean / max rendered chars |
| --- | ---: | ---: | ---: |
| `single` — one fact, asked at distance 0 | 0.35 | 0.3501 | 112.5 / 171 |
| `filler` — one **or two** fact-free exchanges between fact and question | 0.15 | 0.1493 | 166.0 / 230 |
| `two_fact` — two facts, a question about one | 0.20 | 0.1997 | 151.9 / 230 |
| `untold` — asked, never told → "You haven't told me …" | 0.15 | 0.1509 | 116.3 / 216 |
| `contrast` — my-side and your-side question in one dialogue, your-side answered with `snnchat.persona`'s own line | 0.15 | 0.1500 | 197.3 / 230 |

Of 37,336 `filler` dialogues 18,836 carry one exchange and 18,500 two; of 37,718
`untold` dialogues 18,790 follow a fact about another slot, 9,499 are asked cold,
and 4,715 / 4,714 follow one / two filler exchanges with no statement anywhere.
So **distances 1 and 2 are trained shapes, told and untold**, which is what lets
§10's rule read distance 2 as decay and not as shape novelty. Of 49,929 two-fact
dialogues 5,647 (0.113) pair `colour` with `object` — the only pairing a
type-matcher cannot solve, because both draw on one list.

**The acknowledgement repeats the value in 0.3990 of conversations** (target
`ACK_ECHO_FRACTION` = 0.4, one draw per conversation). Below one half on
purpose: at probe time the acknowledgement is the model's own sample.

**Every rendered dialogue ≤ 230 characters, asserted** (`MAX_RENDERED_CHARS`;
realised min 52, mean 141.7, median 136, p90 205, p99 228, max 230), so a
dialogue sits whole inside one aligned 256-character window. The ceiling
redraws the *wording*, never the shape, slot, value or distance, so dose stays
uniform per value; the cost is a tilt toward short templates inside the long
shapes, which matters only if template-level results are ever read.

**The dose, as measured and not as multiplied out.** The arithmetic: 14,000 ×
160 × 256 = 573.4 M characters trained, × 0.06 = 34.4 M from this source =
134,400 windows = 0.97 passes over the build. Dividing by the mean dialogue
length **overstates** the dose — the answer is a dialogue's last turn and a
window's far edge truncates exactly that — so the floor of
`MIN_EXPOSURES` = 600 asked-for answers per value is held against a **replay of
the real sampler** (8,000 steps, scaled to 134,400 windows). Per value,
min / mean / max over the slot's values:

| slot | values | asked-for answer is a target | … with its establishing value in the same window |
| --- | ---: | ---: | ---: |
| name | 48 | 754 / 820 / 870 | 678 / 743 / 791 |
| pet | 40 | 792 / 846 / 901 | 697 / 745 / 797 |
| colour | 12 | 1,460 / 1,500 / 1,599 | 1,239 / 1,281 / 1,374 |
| food | 40 | 768 / 823 / 892 | **665** / 713 / 768 |
| animal | 40 | 809 / 870 / 936 | 678 / 732 / 795 |
| object | 12 | 1,086 / 1,135 / 1,220 | 965 / 1,009 / 1,081 |

The right-hand column is the one the floor is held against, and its least value
is 665. These are **expectations**: one run's own count for one value scatters
like a Poisson count (√665 ≈ 26), so an individual value can land under 600 in a
given run, and that is stated here so it cannot be discovered later as an
excuse. The slots are drawn at `SLOT_WEIGHTS` (name .19, pet .24, colour .10,
food .18, animal .20, object .09), a judgement tuned against this replay: a
uniform draw left every pet name under the floor. Why the dose is counted in
times *asked*: `experiments/chat/_quality/subject_frequency.json`, where at
matched exposure being asked for predicts a hit and being read does not.

## 4. The instrument

`scripts/chat/memory_probe.py` stays **byte-unchanged** (blob
`a524c3d7eceeaf40b7862cc1f5d1d33ec84eb560`, pinned by a test).
`scripts/chat/memory_probe_v2.py` imports `ITEMS`, `hit`, `wilson`, `mcnemar`
from it. Each stratum is 10 items × 8 sampler seeds = **80 paired conversations,
lattice 1/80 = 0.0125**. Sampler seeds are 1500–1507; the committed probe drew
0–7. Decoder: the committed probe's — n=8, λ=0.6, `max_new` 120.

| stratum | values | phrasings | distances | control | role |
| --- | --- | --- | --- | --- | --- |
| **S1** | trained | **held-out** statement *and* question | 0 | establishing turn removed, same sampler seed | **the primary; gates** |
| S2 | the same ten (slot, value, aux) | trained | 0, 1, 2 | same | reported; feeds §10 |
| S3 | held-out | trained | 0 | same | reported |
| **S4** | trained, two facts | trained | 0 | **swapped roles**, same sampler seed | **gates tier (iii) and the swap guard** |
| S5 | held-out **frames** | — | 0 | establishing turn removed | reported |

S1 against S2 at distance 0 is a single-factor contrast — same values, same
seeds, only the wording differs. **S3 and S4 use trained phrasings on purpose**:
S3 then differs from S2 in the value only, and S4 measures binding only;
stacking an unseen phrasing on either would make a failure unreadable. The claim
"absent from the generator's templates" is therefore a claim about S1 and S5,
and `tests/test_snnchat_v15.py` asserts both halves.

**S4's control** tells the same two values with their slots exchanged, asks the
same question at the same sampler seed, and is scored for the *original* target.
S4 is built from name↔pet (5 items) and colour↔object (5 items) only, because
the control needs a value grammatical in both slots.

Per reply, independent **flags**: `hit`, `swap`, `wrong_value`, `refusal`,
`persona_capture`, `any_value`. Guards read flags, not the exclusive category,
so a reply naming both the right value and a wrong one counts in both columns.

## 5. The primary and the two tiers

> **PRIMARY: S1 at distance 0 — trained values, phrasings the generator
> reserves and refuses to emit — told against untold, paired at the same
> sampler seed, exact two-sided McNemar, on each of the two training seeds.**

**Tier RESOLVED (per seed):** McNemar p < 0.05 **and** told-only > control-only.
On the /80 lattice the least that clears is 6 told-only against 0 (p = 0.0312);
5 against 0 is p = 0.0625 and fails; with one control-only pair it takes 8
(p = 0.0391; 7 gives 0.0703), with two it takes 10 (p = 0.0386).

**Tier SHIP-CANDIDATE (per seed): RESOLVED, and all of**

| | bar | lattice |
| --- | --- | --- |
| (i) | committed `memory_probe.py` at distance 0: **told > 15.5/80 and untold < 4.5/80** | /80; 15 fails, 16 passes; 5 untold fails, 4 passes |
| (ii) | S1 told at distance 0 **> 7.5/80** | /80; 7 fails, 8 passes |
| (iii) | S4 told beats its swapped-roles control: McNemar p < 0.05, told direction | 80 pairs |
| | **and every guard in §6 passes** | |

**Why two tiers.** The committed probe's phrasings are, on purpose, a subset of
the generator's TRAIN templates. Passing (i) alone is persona-style recitation,
which `canned_rate` exists to flag. (ii) is where transfer is read.

**The committed probe's ceiling is 64/80, not 80/80**: items 7 and 8 (town, age)
are held-out frames. Its item 1, `elliot`, is a trained value and is reported on
its own line.

**S1 can be cleared by one item, and three of its items are near-trained.** One
item is 8 cells and bar (ii) is 8/80. The review found by word-level edit
distance that S1 items 1, 3 and 10 sit a word or two from a trained template
(`score_v15.NEAR_TRAINED_S1_ITEMS`). **The bar is not moved.** This document
commits to reading S1 **per item**, and again without those three, and to saying
which it was; the scorer prints both.

**Tier (iii) rides on forty cells.** On S4's five name↔pet items a model that
answers with whichever in-context word is a *name* scores the same in both arms,
so those 40 pairs are concordant under type-matching and only binding separates
them; the colour↔object cells are where the control is exact. (iii) is expected
to be the hardest bar in the round.

**Overall, both seeds:** both ship-candidate → `ship-candidate`; both at least
resolved → `resolved` (a ship-candidate seed plus a resolved seed reads
`resolved`); both floor → `floor`; **anything else — a straddle, a seed that did
not finish 14,000 steps, a missing or partial artifact — is `unresolved`.**

**No bar sits on a lattice point** (`CONVENTIONS.md` §4 rule 2): every count bar
is a half-integer, the battery bars included. **A near-miss is a miss**: 15/80
on (i) fails, and no interval rescues it.

## 6. The guards — each listed individually in the output, each must pass on both seeds

| guard | read on | rule |
| --- | --- | --- |
| `wrong_value` | S1 told | replies naming a *different* value of the asked slot **strictly fewer** than hits ("Your name is Tom" must lose) |
| `swap` | S4 told | replies naming the *other* told fact's value strictly fewer than hits |
| `confabulation` | S1 untold | replies naming *any* value of the slot **no more than** replies in the "haven't told me" family |
| `bpc_soda` | step-14,000 held-out bpc | not **worse** than the same-index incumbent by more than **0.0022614** |
| `bpc_stories_topic` | same | … **0.0033309** |
| `bpc_tinystories` | same | … **0.0021895** |
| `heldout` | `echo_holdout.py --set heldout --seeds 6 --n 256` | selected rate **not resolved below 60/120** (Wilson) |
| `social` | `quality.py` battery, reading row n=1 λ=0 | **> 19.5/20** |
| `identity` | same | **> 14.5/16** |

The three count guards compare two counts from the same 80 draws; each is
printed with its Wilson interval and denominator, and the comparison itself is
on the counts.

**The bpc bands, measured.** Sample SD over the four committed replicates of the
shipped recipe (`chat-v3d-aligned{,-s1,-s2,-s3}/summary.json` → `val`, n = 4,
3 df), × 2.83 = 2√2 — two sigma on a difference of two single runs, unpaired
because of hazard (b):

| source | SD_seed | band | status |
| --- | ---: | ---: | --- |
| soda | 0.0007991 | 0.0022614 | guarded |
| stories_topic | 0.0011770 | 0.0033309 | guarded |
| tinystories | 0.0007737 | 0.0021895 | guarded |
| persona | 0.0014545 | (0.0041163) | reported |
| dolly | 0.0080569 | (0.0228009) | reported |
| oasst1 | 0.0045660 | (0.0129218) | reported |
| alpaca | 0.0035174 | — | reported; gave up 0.06 of the mix and is **expected to rise** |

    python scripts/chat/score_v15.py --check-bands "<main tree>/experiments/chat"

re-reads the four summaries, prints the table, and fails loudly if an embedded
value or band has drifted from its evidence. SD_seed at n = 4 is itself known to
roughly ±40 %. The guard is **one-sided**: a source that comes out *better* by
more than its band is printed and fails nothing. s0 is compared with
`chat-v3d-aligned`, s1 with `chat-v3d-aligned-s1`. **Weighted bpc is not read**:
the mix differs, so it is not comparable with 1.1743.

**Where the incumbents themselves sit on the guards that have no margin**, read
through the scorer from committed artifacts (appendix): all four incumbent
seeds read social 20/20; identity 15/16, 16/16, 16/16, 16/16; HELDOUT selected
60, 57, 59, 60 of 120, all `unresolved` against 60/120 and so all passing. The
`heldout` guard fails at 49/120 or fewer. `identity` has **one miss of
headroom on SHIPPED's own reading**, and sampler noise alone can spend it.

**Reported beside the guards, gating nothing: the template-capture rate** —
`canned_rate` extended to `snnchat.recall`'s own bot lines, over substantive
picks at n=1 λ=0. Its floor on the incumbents is 0/112 on substantive picks and
1–2 of 148 over all picks (the shipped model already says "You're welcome!",
which is a recall filler reply), so the floor is not zero.

## 7. Power, stated before running

* **n = 2 training seeds, fixed here.** Both must clear. If a single seed
  clears a tier with probability p, the round clears it with about p²; the
  panel's odds in §1 already price that in. A **straddle is `unresolved`**. n is **never enlarged after the fact**, and a seed that rolled
  back (a `divergence` event in its `log.jsonl`) is **reported as rolled back
  and not replaced**; the driver leaves a partial diverged run alone and reports
  the seed failed unless `--retrain-diverged` is passed, which is recorded.
  A rolled-back seed that nevertheless finishes 14,000 steps is scored, and its
  line says ROLLED BACK.
* **What RESOLVED can see.** Against a control at 0/80, a told rate of 6/80 =
  0.075 (95% CI [0.035, 0.154]) is the smallest effect the primary resolves.
  Smaller than that is `floor` by construction.
* **Sampler seeds are 8 per item, fixed in committed code**, as are the item
  lists. Neither is extended, trimmed or re-curated after a checkpoint exists.
* **The false-fail rate of the ship tier is not small.** Nine guards plus three
  bars, on two seeds, all required; two of the guards sit at the incumbent's own
  reading (§6). A model that is in fact harmless can miss `ship-candidate` on
  one battery draw. That is accepted: it then reads `resolved`, which still
  reports the recall result, and SHIPPED is not displaced on a recommendation
  anyway (§11).
* **These intervals are over the sampler, not over training runs**
  (`CONVENTIONS.md` §3), and six-plus readings at 95 % are not a 95 % package.

## 8. Hazards, written down before the run

**(a) Evidence-free answers — measured.** `snnchat.data.MixtureSampler._windows`
draws **one** `align_frac` decision per batch row, for every source alike; there
is no per-source alignment. A quarter of recall windows therefore open at a
uniformly random offset, and when that falls between a dialogue's statement and
its answer the answer is trained with its evidence cut off — a value produced
from nothing, which is the confabulation the guard is there to catch. Replaying
8,000 steps of the real sampler over the packed file:

    establishing value outside the window      206,985 / 1,643,105 = 0.1260
    … and no value-repeating acknowledgement   193,157 / 1,643,105 = 0.1176
    all of them in random-offset windows       206,985 /   490,861 there; 0 / 1,152,244 in aligned windows

(same command as §3). Windows overlap, so no interval is attached; it describes
the sampler and is compared with no bar. Two things bound it without removing
it: every such answer lies before its window's first `<|bos|>`, a context no
inference ever has; and it cannot be fixed from the generator — it needs
per-source alignment in `snnchat.data`, which this round does not touch. **The
`confabulation` guard is read with this in mind**: a fail there is as likely the
sampler's doing as the source's.

**(b) The sampler stream is not the incumbents'.** `_windows` draws
`g.integers` once per source in id order and ids follow sorted names, so adding
`recall` changes every source's draws at every step. No v15 seed is paired with
any shipped seed; every comparison with an incumbent is **unpaired**, which is
why the bpc bands are 2√2 · SD_seed and not a paired bound.

**(c) `ckpt_best` is selected on a weighted validation that now contains
`recall`**, and `recall.val.bin` is the tail of the same template generator:
**240 of its 1,003 dialogues are exact duplicates of a training dialogue**
(appendix). Its bpc will be very low and will pull checkpoint selection at 0.06
weight. Every probe loads `ckpt_best.pt`; the bpc guard reads `summary.val`,
the step-14,000 evaluation. On all four incumbents the last evaluation was the
best (the `eval` events of each `log.jsonl`: weighted bpc falls at every
evaluation to step 14,000), so the two described the same weights — **no code
in this round checks that they do.** The report reads it off the v15 runs'
own `eval` events; if the lowest weighted bpc is not at step 14,000 it says so
and reads the primary on `ckpt_last.pt` as a reported sensitivity check,
changing no verdict.

**(d) Measurement limits, known in advance.** `wrong_value` and `any_value` are
closed-vocabulary: "Your name is Timmy" names nothing in the lists and falls to
`other`, so both guards have a blind spot for out-of-list confabulation. In the
other direction the pet and animal lists hold ordinary story words (teddy,
fluffy, bear, duck), so an untold reply that drifts into story text raises
`any_value` with no confabulation of a user fact — conservative. `tim` and `tom`
are trained names and TinyStories' defaults; the untold arm will show them as a
prior, which is what the paired control is for. `snnchat.persona`'s
"Nothing genuine - but blue seems like a good answer." is a your-side line the
contrast shape trains; it inflates told and control **equally** on items whose
target or distractor is blue (committed item 5; S4 item 8). S5's
"i am {v} years old" opens with the trained name template "i am {v}"; expect
name-style acknowledgements there.

## 9. Reported, never gated — and not promotable afterwards

Distances 1 and 2 (S2); the unseen-value stratum S3; the held-out-frame stratum
S5; the one n=256 shipped-decoder pass at distance 0; S1 per item and without
items 1, 3, 10; the committed probe's `elliot` line; the **acknowledgement
column** — the echo-ON winner (what the conversation continued with) and the
echo-OFF winner read off the *same* candidate pool, with told hits split by
whether the fed acknowledgement carried the value, because
`RerankParams.echo=True` steers the acknowledgement toward the told word; the
sampler-free **binding probe** (bits per value character under told / untold /
swapped); per-source bpc on every source; the template-capture rate.
**A reported column may not be promoted to a gate after the fact.**

## 10. The decision rule carried forward

As `score_v15.decision_rule` implements it, fixed here:

* **If distance 0 passes** (overall `resolved` or `ship-candidate`) **and S2's
  told count at distance 2 is strictly below half its distance-0 count on both
  seeds** → carried-state training is the next round. If it does not fall below
  half, reach is not shown to be the binding constraint and the rule does not
  fire.
* **If distance 0 fails on trained values** — overall `floor`, **and** neither
  S2 at distance 0 nor the committed probe resolves on either seed → **stop the
  memory programme**: longer windows, carried state and the retrieval envelope
  are moot for recall.
* Overall `floor` while a *trained-phrasing* reading resolves is recitation
  without transfer; the stop rule does not cover it and does not fire.
* On `unresolved` the rule does not fire.

## 11. What this round does NOT touch, and what the floors are

* **`experiments/chat/SHIPPED` is edited under no outcome.** A `ship-candidate`
  goes to the owner as a **recommendation with transcripts**; adopting it is
  his. An `unresolved` or `resolved` result does not displace the incumbent.
* **`build_corpus.DEFAULT_MIX` is unchanged**; `recall` is a new source name
  that a run opts into with `--mix`. No existing `.bin` is repacked or
  rewritten, so every earlier checkpoint keeps the corpus it trained on.
* **No model, kernel or `ChatConfig` change.** `src/snn/` and the research tree
  are untouched; nothing here counts against the research GPU budget.
* **The floors are diagnostics, not gates.** The driver's stage 1 runs the
  committed probe, `memory_probe_v2` (n=8) and the binding probe on the SHIPPED
  checkpoint **before anything trains**, under `<out-root>/floors/`, so every
  v2 stratum has an incumbent reading made by today's code. No bar reads them.
  The stage *blocks* training only if a probe cannot run at all — an instrument
  that cannot read SHIPPED cannot score its successor. The expected reading is
  the committed one, 0/80, with replies classed `persona_capture`.

## 12. Where the code's reading is narrower or wider than the design panel's words

The panel wrote the design in prose; the builders had to choose readings. Each
is adopted **here, before the run**, so none can be chosen afterwards:

1. Tier (iii) "above its swapped-roles control" is a McNemar test, the same
   rule as RESOLVED — stricter than a point comparison.
2. The bpc guard is **one-sided** where the panel wrote "within".
3. It guards the **three** sources the panel gave bands for. `persona`, `dolly`
   and `oasst1` are also unchanged and are reported with the bands they would
   have had; damage to `persona` is caught by the `identity` guard, not by bpc.
4. The count guards decide on counts; their Wilson intervals are printed.
5. The trained lists are 48/40/12/40/40, not "~60 per slot", and the slots are
   not drawn uniformly — both forced by the measured dose (§3).
6. Town and age are left untrained, so bar (i) has a 64/80 ceiling.
7. The template-capture rate is reported; the panel gave it no threshold.

## 13. Cost

Two runs × 14,000 steps. The four incumbents took 2,350 / 2,374 / 2,404 /
2,378 s (`summary.json` → `wall_clock_s`) for the identical shape, so **~80
minutes of training**, `--budget-minutes 75` per run. Scoring on the GPU is
**unmeasured**: per checkpoint it is the committed probe, five v2 strata at
n=8, one n=256 pass, the binding probe, the battery and HELDOUT at n=256, plus
the three floor probes once. The panel's estimate from artifact write times was
40–55 minutes per checkpoint and ~35 minutes of floors; this project's estimates
have missed by 3× before. If the n=256 pass proves too slow it may be dropped
without touching a bar — it is reported only.

Launched detached via `scripts/launch.py` after `--preflight-only` has read
`PREFLIGHT_OK` in the foreground; completion is detected by polling
`<out-root>/status.json`, not by waiting on exit. The driver's GPU check lists
python processes holding a compute context *now* and cannot see a session that
is between two timed benchmarks — **ask before launching**.

## 14. Stopping rules

1. **Two seeds, decided now, run regardless of what the first shows.** A stop
   before both finish is reported as insufficient n → `unresolved`.
2. **A rolled-back seed is reported as rolled back and is not replaced.**
3. **No arm displaces `SHIPPED`**, and certainly not on `unresolved`.
4. **The item lists, the held-out lists, the sampler seeds and every constant
   at the top of `score_v15.py` are frozen** by the commit that holds this file.
   A change to any pre-registered file after a v15 checkpoint exists is refused
   by the driver, and `--force` does not override it.

## 15. The results section is empty until the runs finish

Appended to, never rewritten. `QUALITY_v15.md` carries the verdicts.

---

## Appendix — commands behind the numbers no CLI prints

All from the worktree root, CPU only (`CUDA_VISIBLE_DEVICES=-1`; an empty string
does not hide the GPU on this machine).

**Shape, distance and pairing counts (§3):**

```python
import sys; sys.path.insert(0, "src")
from collections import Counter
from snnchat import recall
ds = recall.build_recall_dialogues(250_000, 0)
print(Counter(d.fillers for d in ds if d.shape == "filler"))
print(Counter("fact" if d.told else d.fillers for d in ds if d.shape == "untold"))
two = [d for d in ds if d.shape == "two_fact"]
print(len(two), sum({s for s, _v, _x in d.told} == {"colour", "object"} for d in two))
print(len(recall.heldout_phrases()))
```

**Validation dialogues that duplicate a training dialogue (§8 c)**, on the pack
§3's command wrote to `<dir>`:

```python
import sys, numpy as np; sys.path.insert(0, "src")
from snnchat.tokenizer import BOS
def dialogues(path):
    a = np.fromfile(path, dtype=np.uint8); c = np.flatnonzero(a == BOS)
    return [a[i:j].tobytes() for i, j in zip(c, [*c[1:], len(a)])]
train = set(dialogues("<dir>/recall.bin")); val = dialogues("<dir>/recall.val.bin")
print(sum(d in train for d in val), len(val))          # 240 1003
```

**The incumbents on the battery and HELDOUT guards, and the template-capture
floor (§6)** — committed artifacts through the scorer's own functions:

```python
import json, sys; sys.path.insert(0, "scripts/chat")
import score_v15 as s
q = "experiments/chat/_quality/"
for r in ("chat-v3d-aligned", "chat-v3d-aligned-s1", "chat-v3d-aligned-s2", "chat-v3d-aligned-s3"):
    guards, rep = s.battery_guards(json.load(open(q + r + ".json")))
    print(r, guards, s.heldout_guard(json.load(open(q + "v12_heldout_" + r + ".json"))),
          rep["template_capture"]["recall"], rep["template_capture"]["recall_over_all_picks"])
```

**McNemar's least clearing counts (§5):** `score_v15.mcnemar(control_only,
told_only)` over (0,5), (0,6), (1,7), (1,8), (2,10).

**Where the `heldout` guard fails (§6):**
`snnchat.quality.resolves_against(k, 120, 0.5)` is `below` for k ≤ 49 and
`unresolved` from 50.

**The incumbents' last evaluation was their best (§8 c):** the `eval` events of
`experiments/chat/chat-v3d-aligned{,-s1,-s2,-s3}/log.jsonl`, field `weighted`.

**The committed floor (§0):** `experiments/chat/_quality/memory_probe.json` →
`summary`, and its rows at distance 0 whose targets hold `elliot`.
