# QUALITY v15 — recitation without binding

**Scored 2026-09-21 against the bars [`PREDICTION_v15.md`](PREDICTION_v15.md)
fixed before either seed was trained.** That file's sha256 is still
`c86464d517423ebc786576f34d4d03901b5460553a7cea11ec6a85de5e711e62` (git blob
`9e27fa03`, the one `driver_record.json` → `prereg.blobs` records), HEAD at
launch and now is `57c146a`, and `scripts/chat/score_v15.py` fires every bar from
constants that commit holds. `PREDICTION_v15.md` is not edited; its §15 says the
results go here, by reference.

Not part of the Phase-1..5 research protocol. No number here is a reported
figure. Rates follow [`CONVENTIONS.md`](CONVENTIONS.md): every rate carries its
denominator and a 95 % Wilson interval, `unresolved` is a verdict, and a
near-miss is a miss. **`experiments/chat/SHIPPED` is not touched by this round**
(§8). **No research decision is ruled.**

Every number below was regenerated on the CPU from the artifacts curated under
`experiments/chat/_quality/v15_*` (appendix), and `score_v15.py` re-run on the
driver's output directory produces a dictionary identical to the committed
`v15_score.json`.

---

## 0. The verdict, as it fired

**OVERALL: `floor`, on both seeds.** The primary — S1 at distance 0, trained
values in phrasings the generator reserves and never emits, told against untold
at the same sampler seed — came back `unresolved` on both training seeds:
**5/80 vs 1/80 on s0** (+5/−1 discordant, exact McNemar p = 0.2188) and **5/80
vs 0/80 on s1** (+5/−0, p = 0.0625). The least that clears is 6 against 0
(p = 0.0312); 5 against 0 was written down in advance as a fail (§5 of the
pre-registration) and it is one. Every ship bar failed on both seeds, the
`wrong_value` guard failed by 45 and 35 cells, and the decision rule carried
forward **did not fire** (§6): the pre-registration's two branches did not
anticipate a model that learns the *form* of the answer under trained phrasings
(committed probe 9/80 and 14/80 vs 0/80, both resolved above) while naming a
*different* trained value on 50/80 and 40/80 of the held-out-phrasing told
replies. The panel's odds put `floor` at ~25–30 %. It is the eighth
training-side round in a row that lost or came back unresolved.

**What the owner would see**, read off the committed probe's item 1 —
`my name is elliot` / `what is my name?` at distance 0 — by a fixed rule: the
told reply at sampler seeds 0, 1 and 2, for the incumbent the floors were taken
on and for both v15 seeds. No seed is chosen; all eight are in
`v15_*_memory_probe.json`.

| seed | SHIPPED at floors time (`chat-v3d-aligned`) | `chat-v15-recall-s0` | `chat-v15-recall-s1` |
| ---: | --- | --- | --- |
| 0 | I don't really have a name. I'm a spiking neural network. | Your name is Pete. | Your name is Laura. |
| 1 | I don't really have a name. I'm a spiking neural network. | Your name is Zoe. | **Your name is Elliot.** |
| 2 | I don't really have a name. I'm a spiking neural network. | Your name is Leo. | Your name is Alice. |

The `elliot` line reads **0/8 on the incumbent, 0/8 on s0, 1/8 on s1**
(`committed_item1_elliot_seen_value`; the hit is seed 1 above). s0's other five
seeds say Pete, Megan, Julia, Leo, Leo; s1's say Zoe, Leo, Ivy, Matt and — seed
7 — *"You told me your name is Stories."*, a word on no list, which is hazard
(d)'s blind spot showing itself. The untold control on the same item says
*"I don't know your name. You haven't told me."* on 8/8 (s0) and *"You haven't
told me your name yet."* on 8/8 (s1). So the possessive is now resolved — the
question is no longer routed to the persona — and the answer is a confidently
wrong name seven times in eight.

The same rule on the primary stratum (S1 item 1, `everyone calls me alice` /
`do you remember my name?`, sampler seeds 1500–1502): s0 says *Your name is
John.* / *Your name is Eva.* / *No name - just a small calm not much with your
pants.*; s1 says *Your name is Adam.* / *Your name is Steve.* / *Your name is
Steve.* That item is 0/8 hits with 4/8 and 8/8 wrong values.

## 1. What ran

One detached driver, `scripts/chat/session15_driver.py`, 18:23:44 → 20:03:16 on
2026-09-21 (5,972 s), every stage `ok`, `alerts` empty, no `divergence` event in
either `log.jsonl`, no rollback, no retrain.

| stage | started → finished | s |
| --- | --- | ---: |
| preflight | 18:23:44 → 18:23:44 | 0 |
| floors (3 probes on the incumbent) | 18:23:44 → 18:27:53 | 249 |
| smoke (20 steps) | 18:27:53 → 18:28:21 | 28 |
| train `chat-v15-recall-s0` | 18:28:21 → 19:09:35 | 2,474 (`wall_clock_s` 2,453.81) |
| train `chat-v15-recall-s1` | 19:09:35 → 19:50:18 | 2,443 (`wall_clock_s` 2,436.97) |
| score s0 | 19:50:18 → 19:56:44 | 386 |
| score s1 | 19:56:44 → 20:03:14 | 390 |
| verdict | 20:03:14 → 20:03:16 | 2 |

The pre-registration's §13 estimated 40–55 minutes of scoring per checkpoint
and ~35 minutes of floors; the measured figures are 6.5 and 4.2 minutes. This
project's estimates missed by 3× before; this time they missed by 6–8× in the
other direction. Training landed inside the incumbents' 2,350–2,404 s.

**The recipe diff, as the driver enforced it.** `recipe_diff.differs` for s0 is
exactly `{data_dir, mix, out_dir, run_name}`, `violations` is empty, and
`new_fields_at_default_exempted` is empty; s1's committed `config.json` differs
from s0's in `seed` (1) and `run_name` only. The mix is the pre-registered one:
`stories_topic` 0.40, `soda` 0.20, **`alpaca` 0.09** (from 0.15), `tinystories`
0.09, `persona` 0.08, `dolly` 0.06, `oasst1` 0.02, **`recall` 0.06**. Everything
else is `chat-v3d-aligned/config.json` field for field (14,000 steps, B=160,
L=256, lr 5e-4 cosine to 5 %, warmup 200, wd 0.1, clip 1.0, `bot_loss_weight`
3.0, `align_frac` 0.75, `twocomp_threshold`, d=1024 × 4 layers, fp32,
`eval_every` 3500), initialised from `chat-v2-anneal/ckpt_best.pt`
(`driver_record.json` → `parent`).

**The corpus.** `recall.bin` in the staging directory hashes to
`0c8eaeb60bb1e34cee510330d924afeb279a822473f6db2a7a696b2de2a3b44e`, the value
§3 of the pre-registration says the trained file must hash to; `recall.val.bin`
to `49fb36576ec4102ad277fc911eaecd3ad897612e6db4d2fbbf560c369772751f`.
35,272,337 / 141,655 characters, 250,000 conversations, `seed` 0, built in 8.8 s.
The manifest carries the sampler replay: `cut_rate` 0.1260 (206,985 of
1,643,105 answers with their evidence outside the window, all of them in the
random-offset windows, 0 in the aligned ones) and no value under the
`MIN_EXPOSURES` = 600 floor in any slot.

**Which file the floors were taken on.** The driver's stage 1 ran the committed
probe, `memory_probe_v2` (n=8) and the binding probe on
`C:\…\Text SNN\experiments\chat\chat-v3d-aligned\ckpt_best.pt` — the path is in
the `ckpt` field of all three `v15_floors_*.json` and in `driver_record.json` →
`plan`. That was `SHIPPED` when the driver read it. **The owner repointed
`experiments/chat/SHIPPED` in his main tree to `chat-v6-scratch/ckpt_best.pt`
during the run** (the file's mtime is 18:34:36, seven minutes after the floors
stage finished at 18:27:53), by a separate ruling recorded in
`docs/chat/V6_SCRATCH_NOTE.md` on branch `chat/v6-scratch-diagnostic`. On this
branch `SHIPPED` still names `chat-v3d-aligned`. Every comparison in this
document — floors, bpc bands, `REFERENCE` — is against `chat-v3d-aligned` and
its seed replicates, and none of this round's instruments has read
`chat-v6-scratch`.

**The floors, as today's code read the incumbent** (`v15_floors_*`): committed
probe 0/80 told, 0/80 untold at every distance — the committed value. Under the
v2 phrasings, S1 told **1/80** vs 0/80, S2 0/80 at
distances 0, 1 and 2, S3 0/80, S4 0/80, S5 0/80; `persona_capture` 10/80 on S1
told and 37/80 on its control. The binding probe on the incumbent read
told − swapped **−0.107** bpc over all 38 items (26/38 items told below swapped)
— a tilt already present before any recall training, which §4 returns to.

## 2. The primary and the tiers, per seed

Both seeds finished 14,000 steps; `n_divergences` 0; both verdicts `floor`.

### 2.1 PRIMARY — S1 at distance 0 (trained values, held-out phrasings)

| | told | untold | +told-only / −control-only | McNemar p | verdict |
| --- | --- | --- | --- | ---: | --- |
| s0 | 0.0625 (5/80), 95% CI [0.027, 0.138] | 0.0125 (1/80), 95% CI [0.002, 0.068] | +5 / −1 | 0.2188 | **unresolved** |
| s1 | 0.0625 (5/80), 95% CI [0.027, 0.138] | 0.0000 (0/80), 95% CI [0.000, 0.046] | +5 / −0 | 0.0625 | **unresolved** |

Tier RESOLVED needs p < 0.05 in the told direction. Neither seed reaches it;
overall `floor`.

**Per item**, told hits of 8 (items 1, 3 and 10 are the near-trained ones
`NEAR_TRAINED_S1_ITEMS` names):

| item | 1 alice | 2 jack | 3 buddy | 4 luna | 5 purple | 6 cookies | 7 soup | 8 tiger | 9 owl | 10 yellow hat |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| s0 | 0 | 0 | 1 | 0 | 2 | 0 | 0 | 0 | 0 | 2 |
| s1 | 0 | 0 | 1 | 0 | 4 | 0 | 0 | 0 | 0 | 0 |

Without items 1, 3 and 10: **0.0357 (2/56), 95% CI [0.010, 0.121]** on s0 and
**0.0714 (4/56), 95% CI [0.028, 0.170]** on s1. On s0 three of the five hits sat
on near-trained items; on s1 one of five did. Six of the ten hits across both
seeds are on item 5, the colour slot, whose list has 12 values against the name
slot's 48 — an inference, not a measurement, but the smallest list is where the
hits are.

### 2.2 Tier SHIP-CANDIDATE — every bar failed on both seeds

| bar | s0 | s1 |
| --- | --- | --- |
| (i) committed probe d0, told > 15.5/80 and untold < 4.5/80 | **FAIL**: told 0.1125 (9/80), 95% CI [0.060, 0.200]; untold 0.0000 (0/80), 95% CI [0.000, 0.046] | **FAIL**: told 0.1750 (14/80), 95% CI [0.107, 0.273]; untold 0/80 |
| (ii) S1 told > 7.5/80 | **FAIL**: 5/80 | **FAIL**: 5/80 |
| (iii) S4 told beats swapped-roles control, McNemar told direction | **FAIL**: told 0.1125 (9/80), 95% CI [0.060, 0.200] vs swapped 0.1250 (10/80), 95% CI [0.069, 0.215]; +6/−7, p = 1.0000, unresolved | **FAIL**: told 0.0625 (5/80), 95% CI [0.027, 0.138] vs swapped 0.0625 (5/80); +4/−4, p = 1.0000, unresolved |

(i)'s untold half passed on both seeds; its told half is 7 and 2 cells short
of the 16/80 that passes. (iii) is the bar the pre-registration expected to be the hardest, and it
is not close: the swapped-roles control scores the *same* as the told arm, which
is what a type-matcher — answer "what is my name?" with whichever in-context word
is a name — was predicted to score (§5 of the pre-registration, "forty cells").

### 2.3 The reported strata (n=8, gating nothing)

| stratum | s0 told | s0 control | +/− | p | | s1 told | s1 control | +/− | p | |
| --- | --- | --- | --- | ---: | --- | --- | --- | --- | ---: | --- |
| S2 d0, trained phrasings | 0.1750 (14/80), 95% CI [0.107, 0.273] | 0/80 | +14/−0 | 0.0001 | above | 0.1250 (10/80), 95% CI [0.069, 0.215] | 0/80 | +10/−0 | 0.0020 | above |
| S2 d1 | 0.1625 (13/80), 95% CI [0.098, 0.258] | 0.0250 (2/80) | +13/−2 | 0.0074 | above | 0.1125 (9/80), 95% CI [0.060, 0.200] | 0.0125 (1/80) | +9/−1 | 0.0215 | above |
| S2 d2 | 0.1125 (9/80), 95% CI [0.060, 0.200] | 0.0375 (3/80) | +8/−2 | 0.1094 | unresolved | 0.1750 (14/80), 95% CI [0.107, 0.273] | 0.0125 (1/80) | +14/−1 | 0.0010 | above |
| S3 d0, held-out values | 0/80 | 0/80 | 0/0 | 1 | unresolved | 0/80 | 0/80 | 0/0 | 1 | unresolved |
| S5 d0, held-out frames | 0/80 | 0/80 | 0/0 | 1 | unresolved | 0/80 | 0/80 | 0/0 | 1 | unresolved |
| committed d1 | 0.1625 (13/80) | 0.0125 (1/80) | +12/−0 | 0.0005 | above | 0.0375 (3/80) | 0.0250 (2/80) | +3/−2 | 1.0 | unresolved |
| committed d2 | 0.0750 (6/80) | 0.0375 (3/80) | +6/−3 | 0.5078 | unresolved | 0.2250 (18/80) | 0.0250 (2/80) | +17/−1 | 0.0001 | above |

S2 at distance 0 against S1 at distance 0 is the single-factor contrast the
instrument was built for — same values, same sampler seeds, only the wording
differs — and it reads **14 vs 5 on s0, 10 vs 5 on s1**. Distance is not
monotone on either probe (s0's committed probe goes 9 → 13 → 6, s1's 14 → 3 →
18), which is the sampler's noise at 80 draws and not a reach curve; the
decision rule's "half by distance 2" clause is moot on `floor` and would not
have fired anyway (S2 d2 is 9 vs 14 on s0, 14 vs 10 on s1, neither below half).

**S3 and S5 are exactly zero on both seeds.** A held-out value is never copied
(0/160 told); a held-out frame is never answered (0/160). S3's told replies
are `wrong_value` on 72/80 and 64/80 — *"my name is oscar"* → *"Your name is
John."*, *"i have a navy hat"* → *"Your hat is brown."* (s0) and *"Your hat is
got."* (s1). S5's are `other` on 56/80 and 64/80 — *"i live in a town called
milton"* → *"I like dogs too."*, *"my sister is called emma"* → *"Your rabbit is
called Charlie."*

## 3. The guards, per seed, as they fired

| guard | s0 | s1 |
| --- | --- | --- |
| `wrong_value` (S1 told: wrong < hit) | **FAIL** — wrong 0.6250 (50/80), 95% CI [0.515, 0.723] vs hit 5/80 | **FAIL** — wrong 0.5000 (40/80), 95% CI [0.393, 0.607] vs hit 5/80 |
| `confabulation` (S1 untold: any value ≤ "haven't told me") | pass — 0.0875 (7/80), 95% CI [0.043, 0.170] vs 0.9000 (72/80), 95% CI [0.815, 0.949] | pass — 0.0375 (3/80), 95% CI [0.013, 0.104] vs 0.9375 (75/80), 95% CI [0.862, 0.973] |
| `swap` (S4 told: swap < hit) | pass — 0.0875 (7/80), 95% CI [0.043, 0.170] vs 9/80 | **FAIL** — 0.0750 (6/80), 95% CI [0.035, 0.154] vs 5/80 |
| `bpc_soda`, band 0.0022614 | **FAIL** — 1.1983359 vs 1.1948113, Δ +0.0035246 (+4.41 SD_seed), worse | **FAIL** — 1.1965445 vs 1.1938855, Δ +0.0026590 (+3.33 SD_seed), worse |
| `bpc_stories_topic`, band 0.0033309 | pass — Δ −0.0008225, within | pass — Δ +0.0001284, within |
| `bpc_tinystories`, band 0.0021895 | pass — Δ −0.0010414, within | pass — Δ −0.0002655, within |
| `heldout` (n=256, not resolved below 60/120) | pass — 0.5167 (62/120), 95% CI [0.428, 0.604], unresolved | pass — 0.4917 (59/120), 95% CI [0.404, 0.580], unresolved |
| `social` > 19.5/20 | pass — 1.0000 (20/20), 95% CI [0.839, 1.000] | pass — 20/20 |
| `identity` > 14.5/16 | pass — 1.0000 (16/16), 95% CI [0.806, 1.000] | pass — 16/16 |

Seven of the nine guards passed on s0 and six on s1. `swap` is a straddle on the counts
(7 < 9 on s0; 6 > 5 on s1) at rates whose intervals overlap entirely — on the
/80 lattice the guard is decided by one cell — and it is reported as it fired.

**`bpc_soda` is worse on both seeds**, by 1.56× and 1.18× its band. `soda` did
not change weight; what changed is that `alpaca` gave up 0.06 of the mix and
`recall` took it. The donor paid most visibly (§7), and `soda` paid past its
band. Per source, with the bands the reported sources would have had:

| source | weight | s0 Δ | s0 Δ / SD_seed | s1 Δ | s1 Δ / SD_seed | band |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| alpaca (0.15 → 0.09) | 0.09 | +0.0350854 | +9.97 | +0.0364221 | +10.35 | — (expected to rise) |
| oasst1 | 0.02 | +0.0610259 | +13.37 | +0.0245417 | +5.37 | (0.0129218) |
| dolly | 0.06 | +0.0134981 | +1.68 | −0.0010151 | −0.13 | (0.0228009) |
| persona | 0.08 | +0.0038781 | +2.67 | +0.0021062 | +1.45 | (0.0041163) |
| soda | 0.20 | +0.0035246 | +4.41 | +0.0026590 | +3.33 | **0.0022614** |
| stories_topic | 0.40 | −0.0008225 | −0.70 | +0.0001284 | +0.11 | 0.0033309 |
| tinystories | 0.09 | −0.0010414 | −1.35 | −0.0002655 | −0.34 | 0.0021895 |
| recall | 0.06 | 0.3673585 | — | 0.3676085 | — | new source |

`oasst1` and `dolly` are the two sources too small for a full evaluation batch
(`QUALITY_v13.md` §2), so their rows are the noisiest. `persona` sits at 0.94
and 0.51 of the band it would have had — inside, but the `identity` guard is
what covers it and that passed 16/16 on both. Weighted bpc (1.1022, 1.1015) is
**not** read against 1.1743: the mix differs and `recall` at 0.367 bpc pulls it.

**The shipped-decoder pass (n=256, distance 0, reported only):**

| stratum | s0 told | s1 told |
| --- | --- | --- |
| S1 | 0.0750 (6/80), 95% CI [0.035, 0.154] | 0.1625 (13/80), 95% CI [0.098, 0.258] |
| S2 | 0.2125 (17/80), 95% CI [0.137, 0.314] | 0.0625 (5/80), 95% CI [0.027, 0.138] |
| S3 | 0/80 | 0/80 |
| S4 | 0.1500 (12/80), 95% CI [0.088, 0.244] | 0.0375 (3/80), 95% CI [0.013, 0.104] |
| S5 | 0/80 | 0/80 |

At n=256 the S1 pairing would read +6/−0 (p = 0.0312) on s0 and +13/−0
(p = 0.0002) on s1, both in the told direction. **That is a reported column and
it is not promoted**: §9 of the pre-registration fixed the gate at n=8 and says
a reported column may not become a gate afterwards. It is also not a stable
reading — the two decoders disagree about which seed and which stratum is the
stronger one (s0: S2 17 > S1 6; s1: S1 13 > S2 5), and `wrong_value` at n=256
is still 27/80 and 19/80 against hits of 6 and 13. What the owner would see at
the prompt is a better draw of the same behaviour.

## 4. The finding — recitation without binding

**Under trained phrasings the form was learned.** The committed probe, whose
phrasings are a subset of the generator's TRAIN templates, resolves above its
control on both seeds (9/80 vs 0/80, p = 0.0039; 14/80 vs 0/80, p = 0.0001), and
so does S2 at distance 0 (14 and 10 vs 0) and at distance 1 (13 vs 2, 9 vs 1).
The reply changed class wholesale: at the floor, S1 told replies were
`persona_capture` 10/80 and `other` 67/80; after training they are `hit` 5,
`refusal` 14, `other` 11, **`wrong_value` 50** (s0) and `hit` 5, `refusal` 10,
`other` 24, `persona_capture` 1, **`wrong_value` 40** (s1). The model now knows
that *"what is my name?"* is answered *"Your name is <name>."*, and that an
unasked question is answered *"You haven't told me …"* (the untold control is
`refusal` 70/80 and 74/80 on S1, 72 and 70 on S2).

**The value in the slot is nearly unrelated to the one it was told.** On S1
told, `wrong_value` is 50/80 and 40/80 against 5 hits each; on S2 told, with the
*trained* phrasings, 59/80 and 60/80 against 14 and 10 hits; on S3, 72/80 and
64/80 against none. The wrong values are spread, not concentrated: on S1 told
s0 names 34 distinct (slot, value) pairs and its most frequent wrong answer
(`pink`, `gold`) appears 4 times in 80; s1 names 29, its most frequent
(`butterfly`, `gold`) 3 times. On S2 the counts are 39 and 41 distinct pairs.
This is a draw from the trained list, not a fixed confusion. The two-fact
stratum says the same thing from the other side: told 9 vs swapped-roles 10,
5 vs 5 — telling the model *"my name is rocky / i have a dog called harry"*
scores the same for `harry` as telling it the truth.

**The binding probe — sampler-free, bits per value character — shows a tilt
that exists and is small.** Told against swapped (the same slot's statement
carrying a different value, averaged over three), on the value characters of
the canonical answer:

| group (items) | incumbent told − swapped | s0 told − swapped | s1 told − swapped | s0 / s1 items told below swapped |
| --- | ---: | ---: | ---: | --- |
| S2 (10) | −0.212 | −0.223 | −0.267 | 9/10, 10/10 |
| S1 (10) | −0.014 | −0.110 | −0.148 | 7/10, 8/10 |
| S3, held-out values (10) | −0.068 | −0.259 | −0.213 | 8/10, 6/10 |
| committed (8) | −0.147 | −0.138 | −0.137 | 6/8, 6/8 |
| **all (38)** | **−0.107** (26/38) | **−0.186** | **−0.195** | **30/38, 30/38** |

Told − untold over all 38: −0.114 on the incumbent, −0.026 on s0, −0.143 on s1.

Two things in that table carry the finding. First, **the tilt on S2 barely
moved**: −0.212 before any recall training, −0.223 and −0.267 after. What
training moved is the *absolute* level — `bpc_told` on S2 went from 2.715 to
1.024 and 1.005 bits per character, but so did `bpc_untold` (2.894 → 1.108,
1.134) and `bpc_swapped` (2.927 → 1.247, 1.272). The model learned that the
characters after *"Your name is "* are a name from the list; it did not learn
much more than it already knew about *which* one. Second, the tilt is
**−0.19 bits per character, about 0.95 bits on a five-character value, a
likelihood ratio of ~1.9×** (inference from the pooled figure, not a
measurement). Against ~47 other trained names at roughly uniform odds, a 1.9×
tilt wins a sample about 4 % of the time; S1 told reads 5/80 = 6.25 %. The
sampled and the sampler-free readings are the same reading.

**The acknowledgement column — decoder steering, not model memory.**
`RerankParams.echo=True` prefers drafts that repeat the user's content words,
so on the establishing turn it steers the acknowledgement toward the told value.
Read off the same candidate pool, the echo-ON winner carries the value more often
than the echo-OFF winner would have: on S1 d0, **15/80 vs 5/80** (+10/−0,
p = 0.0020) on s0 and 12/80 vs 8/80 (+4/−0, p = 0.125) on s1; on S2, 22 vs 9
(p = 0.0002) and 19 vs 12 (p = 0.0156); on S4 s1, 30 vs 11 (p = 3.8e-6). Told
hits then split by whether the fed acknowledgement carried the value: S1 s0
**3/15 vs 2/65**, s1 2/12 vs 3/68; S2 s0 8/22 vs 6/58, s1 6/19 vs 4/61. When
the value sits in the model's own previous turn — put there by the reranker —
the hit rate is 3.5–6.5× higher than when it sits only in the user's turn.
That is the decoder copying the value closer to the question, and it is the
same reranker mechanism `echo_holdout.py` measures on nouns.

**What this says about the model** — an interpretation, stated as one. The
recurrence in each layer of this architecture is per channel: a unit's state
carries its own past and the layer's feed-forward input, and nothing mixes
channels inside the time loop; mixing happens only in the GEMM between layers.
Such a state can hold *that a name was said* — a slot bit that survives 200
characters and routes the question to the right template, which is what
`refusal` 72/80 on untold and `wrong_value` 50/80 on told together demonstrate —
far more easily than it can hold *which* of 48 five-character strings was said
and re-emit it ten turns of characters later. The template *"what is my name →
Your name is <name>"* was learned as a routine; the `<name>` is filled from the
list's prior with a ~1.9× tilt toward the one in context. The pre-registration's
own words for a pass were *"a closed-vocabulary routine, not discovered
memory"*; the round produced the routine and not the vocabulary binding.

## 5. Hazards, as they landed

**(a) Evidence-free answers were trained in — measured before, and did not show
where the guard looks.** The sampler replay in the manifest confirms 0.1260 of
`recall` answers train with their evidence cut off. The `confabulation` guard
passed on both seeds: untold S1 replies name a value on 7/80 and 3/80 against
"haven't told me" on 72/80 and 75/80. So the evidence-free training did **not**
produce a model that answers cold. The wrong-value behaviour is in the *told*
arm, where the evidence is inside the window, and hazard (a) does not explain
it.

**(b) Unpaired.** No v15 seed shares a sampler stream with any incumbent seed;
every bpc comparison in §3 is unpaired and the bands are 2√2 · SD_seed for that
reason. `bpc_soda` fails at +4.41 and +3.33 SD_seed, comfortably past a
two-sigma-on-a-difference band; it is not a band-edge call.

**(c) HOLDS on both seeds.** `ckpt_best.pt` is selected on a weighted
validation that contains `recall`, and the pre-registration said the report
would read the `eval` events of each `log.jsonl` to check that the lowest
weighted bpc is at step 14,000 — so that the checkpoint every probe loaded is the
one the bpc guard read. It is. Each log holds five `eval` rows: the cadence
evaluations at 3,500, 7,000, 10,500 and 14,000, and the final evaluation at
14,000 repeated with identical values (`v15_*_evals.json`):

| step | s0 weighted | s0 recall | s0 soda | s1 weighted | s1 recall | s1 soda |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 3,500 | 1.1674329 | 0.4593872 | 1.2684121 | 1.1620390 | 0.4526957 | 1.2581457 |
| 7,000 | 1.1441052 | 0.4188843 | 1.2456001 | 1.1443523 | 0.4144947 | 1.2478963 |
| 10,500 | 1.1138311 | 0.3793043 | 1.2124957 | 1.1165426 | 0.3833891 | 1.2128711 |
| **14,000** | **1.1021501** | 0.3673585 | 1.1983359 | **1.1014653** | 0.3676085 | 1.1965445 |
| 14,000 (final) | 1.1021501 | 0.3673585 | 1.1983359 | 1.1014653 | 0.3676085 | 1.1965445 |

Weighted bpc falls at every evaluation on both seeds, the minimum is the last
step, and `summary.val` equals the step-14,000 row to every digit. `ckpt_best`
and `ckpt_last` describe the same weights; no sensitivity read on `ckpt_last`
is needed and none was made.

**(d) The closed-vocabulary blind spot fired.** *"You told me your name is
Stories."* (committed item 1, s1, seed 7), *"Your name is cat."* (S1, s1),
*"Your hat is got."* (S3 item 10, s1), *"Your bicycle is running."* (S5, s1)
name nothing on any list and fall to `other`, so the `wrong_value` counts above
are floors on the confidently-wrong rate, not ceilings. The `blue` inflation the
pre-registration predicted on committed item 5 did not show in the untold arm
(0/80 on both seeds).

## 6. The decision rule did not fire

`score_v15.decision_rule` returned `fired: null` with the note, verbatim:

> S1 is at the floor but a TRAINED-phrasing reading resolves or is missing:
> recitation without transfer, which the stop rule does not cover

The pre-registration's §10 carries two branches. The first — carried-state
training as the next round — needs distance 0 to *pass*; it did not. The second
— stop the memory programme — needs `floor` **and** neither S2 at distance 0
nor the committed probe resolving on either seed; both resolve on both seeds.
The third sentence of §10 names exactly this case and says the stop rule does
not cover it. So the rule's own text anticipated *that* the middle case could
occur, and gave it no consequence.

What it did not anticipate is the *shape* of the middle case: the design read
"trained-phrasing resolves, held-out-phrasing does not" as a **transfer**
problem — the model recites the drilled wording and cannot generalise the
phrasing — and would have pointed at phrasing diversity. What happened is
narrower and worse: the phrasing transferred (S1 told replies answer with a
value of the asked slot on 55/80 and 45/80, the `hit` and `wrong_value`
categories together), and the **binding**
did not, under trained and held-out phrasings alike (`wrong_value` 59 and 60 of
80 on S2 d0). "Form learned, binding not" is a case with a different remedy
from "form not transferred", and the rule cannot tell them apart because it
reads only `hit` counts.

**Nothing is repaired retroactively.** The rule stands as fixed, it did not
fire, and this round hands the next one no instruction. **Referred forward**,
for whoever writes the next pre-registration and not applied here: a corrected
rule would read the `wrong_value` flag beside `hit` under *trained* phrasings,
and would split the floor branch three ways — (1) `floor` with nothing resolving
→ stop, as now; (2) `floor` with a trained-phrasing reading resolving **and**
`wrong_value` < `hit` on S2 → a transfer problem, and a phrasing-side round
follows; (3) `floor` with a trained-phrasing reading resolving **and**
`wrong_value` > `hit` on S2 on both seeds → a binding problem, and a
window-length, carried-state or retrieval round is **not** the next step,
because the fact is already inside the window and is not being read. This round
is case (3) on both seeds (59 > 14, 60 > 10). Whether case (3) should *also*
stop the memory programme is the owner's question (§8), not a rule this
document can write.

## 7. Limitations

* **n = 2 training seeds, fixed in advance.** Both read `floor`; there is no
  straddle to average and no seed is added or replaced. Every interval here is
  over the sampler, not over training runs (`CONVENTIONS.md` §3), and the
  seed-to-seed spread visible in this round (committed probe 9 vs 14; S2 d2
  9 vs 14; oasst1 Δ +0.061 vs +0.025) is the kind of spread those intervals do
  not cover.
* **`wrong_value` is blind to out-of-list values.** "Stories", "cat", "got",
  "running" (§5 d) are counted as `other`. The confidently-wrong rate is at
  least what §4 reports.
* **Three S1 items are near-trained** and the bar was not moved. Without items
  1, 3 and 10: **2/56** on s0 and **4/56** on s1 (§2.1). It changes nothing —
  S1 was under the bar with them — but it means the S1 reading with them in is
  the more flattering of the two, and both are on the record.
* **240 of the 1,003 `recall.val.bin` dialogues are exact duplicates of a
  training dialogue** (regenerated from the packed files, appendix). `recall`'s
  0.367 bpc is therefore partly memorisation and is not read against anything;
  it enters only through `ckpt_best` selection, where hazard (c) shows it did
  no harm.
* **The recall template leaks into ordinary requests.** `template_capture` —
  `canned_rate` extended to `snnchat.recall`'s bot lines, over the 112
  substantive battery picks (topic, list, fact) at n=1, λ=0 — reads
  **0.1250 (14/112), 95% CI [0.076, 0.199]** on s0 and **0.1518 (17/112),
  95% CI [0.097, 0.230]** on s1, against **0/112 on all four incumbent seeds**
  (1–2 of 148 over all picks; v15 reads 18/148 and 23/148). The captures are of
  two kinds. A `list` request containing a slot word is read as a statement and
  acknowledged instead of answered: *"name three colours"* → *"Got it."*,
  *"Okay, I'll remember that."*, *"Thanks for telling me."*; *"can you name an
  animal?"* → *"I don't know your name. You haven't told me."* A `fact`
  question is refused as an untold user fact: *"what colour is grass?"* →
  *"I don't know. You haven't told me what your pony is called."*; *"what
  colour is the sky?"* → *"I don't know. You haven't told me what your goldfish
  is called."* For a REPL user this is a new failure on ordinary requests at
  roughly one substantive reply in seven, on top of the wrong-name
  behaviour the round was run to fix. The panel gave this column no threshold
  (§12 item 7 of the pre-registration) and none is invented here.
* **The donor paid.** `alpaca` rose 0.0351 and 0.0364 bpc (+10.0 and +10.4
  SD_seed) after giving up 0.06 of the mix, as expected; `oasst1`, whose weight
  did not change, rose 0.0610 and 0.0245 (+13.4 and +5.4 SD_seed), and it is
  the smallest and noisiest source. Instruction-following bits were spent on a
  source that bought a template.
* **The incumbent this round measured against is no longer the ship.** Every
  floor and every band is `chat-v3d-aligned`; the current `SHIPPED` in the
  owner's tree is `chat-v6-scratch`, which none of this round's probes has
  read (its own note records a told-fact reading of 1/80, on the committed
  probe, by its own authors — not a number this document regenerates).

## 8. What is not touched, and what this round recommends

**Not touched:** `experiments/chat/SHIPPED` (edited under no outcome, and this
outcome is `floor`); `snnchat.build_corpus.DEFAULT_MIX` (`recall` is opt-in by
`--mix`); every existing `.bin`; `src/snn/`, `experiments/logs/`,
`docs/reports/`, `scripts/exp/` — nothing here counts against the research
budget or rules a research decision.

**Recommendation: do not ship either seed.** Both are `floor`; both fail all
three ship bars and two or three guards; both cost `soda` past its band and
leak a fact-acknowledgement template into roughly a seventh of ordinary
replies; and the behaviour they were trained for — *"Your name is Elliot."* —
appears on 1 of 16 draws of the owner's own example, against 14 confidently
wrong names from the list and one "Stories".
The incumbent's *"I don't really have a name"* is wrong about the question; the
v15 reply is wrong about the answer, with the same confidence the corpus was
built to give it.

**Two questions for the owner**, neither of which this round answers:

1. **Is confident-wrong worse than the old refusal?** Nothing in this round
   ships, so this is not a ship question; it is a question about what the
   *next* recall round should be allowed to produce. A source that taught
   *"You haven't told me"* as the default under uncertainty — rather than an
   answer — would have failed this round's primary just as hard while leaving
   the REPL no worse than it was. The pre-registration's own hazard (a) note
   ("a value produced from nothing … the confabulation the guard is there to
   catch") was pointed at the untold arm, and the untold arm was fine.
2. **Is a binding-side round worth anything, given that the sampler-free tilt
   exists?** The tilt is real (30/38 items on both seeds, −0.19 bpc per
   character), it predated this round on trained phrasings (−0.21 on S2 at the
   floor), and training moved it by 0.01–0.06. What would have to change is
   not the corpus — the fact is inside the window and is read at ~1.9×, where
   a sample against ~47 alternatives would need a tilt an order of magnitude
   larger — but the model's ability to carry a specific five-character string
   across a turn boundary, which is a question about the neuron and the
   architecture, not about the mix. That is the "longer windows, carried state
   and the retrieval envelope are moot" conclusion of the stop rule, reached
   by a route the stop rule did not take. Whether to spend another round on
   it, and on which side of the protocol boundary, is his.

---

## Appendix — regenerating every number, and the curated files

All from the worktree root, CPU only (`CUDA_VISIBLE_DEVICES=-1`; an empty
string does not hide the GPU on this machine). `<out>` is the driver's
`--out-root`; every file it reads is also curated below, so the same commands
run against `experiments/chat/_quality/` after renaming.

**The verdict, both seeds, every bar and guard, the strata, the binding probe,
the n=256 pass, the per-item S1 table, the template-capture rate:**

    python scripts/chat/score_v15.py --out-root <out> --out <anywhere>/score_v15_regen.json
    # the dictionary it writes is == experiments/chat/_quality/v15_score.json

**The bpc bands against their evidence:**
`python scripts/chat/score_v15.py --check-bands "<main tree>/experiments/chat"`.

**Hazard (c)** — the `eval` events of each run's `log.jsonl`, field `weighted`;
the minimum must be at step 14,000:

    python -c "import json,sys; e=[json.loads(l) for l in open(sys.argv[1],encoding='utf-8') if '\"eval\"' in l]; print([(r['step'],round(r['weighted'],7)) for r in e], min(e,key=lambda r:r['weighted'])['step'])" <out>/runs/chat-v15-recall-s0/log.jsonl

(or read `v15_<run>_evals.json`, which is those rows extracted in file order).

**The replies quoted in §0** — `v15_floors_memory_probe.json` and
`v15_<run>_memory_probe.json` → `rows` with `distance == 0` and `"elliot" in
targets`, fields `seed`, `told`, `untold`; §0's S1 lines are
`v15_<run>_memory_probe_v2_n8.json` → `rows` with `stratum == "S1"`,
`item == 0`, `seed` 1500–1502, field `told`. The wrong-value name counts in §4
are the `told` text of every S1/S2 distance-0 row with
`told_flags.wrong_value`, matched as whole words against
`snnchat.recall.VALUES[slot]` minus the item's own value.

**The corpus hash and the 240/1,003 duplicates** — `sha256` of `recall.bin` in
the staging directory, and the snippet in `PREDICTION_v15.md`'s appendix
(§8 c) run over the same two files.

**Stage wall-clock** — `v15_status.json` → `stages[].started/finished`.

**The incumbents' template-capture floor and guards** — the snippet in
`PREDICTION_v15.md`'s appendix (§6), which reads the committed
`chat-v3d-aligned{,-s1,-s2,-s3}.json` and `v12_heldout_*.json` through
`score_v15.battery_guards` / `heldout_guard`.

**Curated files** (`experiments/chat/_quality/`). The driver wrote its JSON
with CRLF line endings on Windows; every `.json` copy below was normalised to
LF before hashing and committing, with its parsed content asserted equal to the
driver's original, so the sha256 here is of the committed blob and of what any
LF checkout reads. The two `.gz` files are byte-for-byte gzips of the driver's
originals (decompressed and compared, CRLF included); `heldout_n256.json` is
15 MB raw and compresses to 2.7 MB, so both are included. No checkpoint and no
whole `log.jsonl` is copied.

| file | sha256 | bytes |
| --- | --- | ---: |
| `v15_score.json` | `414b74251beb569b4524b670ca0f3886ba5018e9f737b026bbf33d138265df62` | 55,029 |
| `v15_driver_record.json` | `49c968e075092e156c3398cc694121723c774d1f9ff12f1374470ba1c4824da9` | 31,694 |
| `v15_status.json` | `cb0edacbaa6029f3ce46d89572c6c638e5dd39cf316a1f54cf3f3d2d072c1dc6` | 5,034 |
| `v15_floors_memory_probe.json` | `c5ffdee6f1046093445087ce1fac11a4de7b7646059e060a9d289bf5d35808d9` | 90,074 |
| `v15_floors_memory_probe_v2_n8.json` | `12d667c979e5c179c100c709ec74034c2742cba4921213381231f286819c27e3` | 681,678 |
| `v15_floors_binding_probe.json` | `1d86bce13b58de48aeacc0d148a34289600b42c7a3a1aa1fec6e24c1e0df44c7` | 22,960 |
| `v15_chat-v15-recall-s0_config.json` | `8e6a415f78dba8c185dcc4451d0543fb6c0168f3b3ae932103537ef2ba5fad26` | 1,210 |
| `v15_chat-v15-recall-s0_summary.json` | `109c2fef69e1ea81c9c2c459f2b399073f13245e169e365047fc6e5c18573d3e` | 1,599 |
| `v15_chat-v15-recall-s0_evals.json` | `597b537a16be3841803b28e0acca830040f0c183f5d46a30e24a5d944f3ef740` | 2,181 |
| `v15_chat-v15-recall-s0_memory_probe.json` | `00fef4fa9d4ad0c7e98457d19bea2f0a6cd03ac96ce654964cc2e5323652f0b0` | 81,987 |
| `v15_chat-v15-recall-s0_memory_probe_v2_n8.json` | `3fb9d196b1dc682d4335f0f883417c2182f86c815a3f593e2f8461949bdbb1dc` | 604,933 |
| `v15_chat-v15-recall-s0_memory_probe_v2_n256.json` | `bf7caac72931d95f52caa7570a9496ac67dba9b57cf5658467c087b53ae8c1f6` | 434,979 |
| `v15_chat-v15-recall-s0_binding_probe.json` | `8b2951470ce80f90157fbaa54d83676264d7fd7fae11c67f6b524f9ed96d1ab7` | 22,992 |
| `v15_chat-v15-recall-s0_battery.json.gz` | `eb5ac2d5c6084225d67f2ad156848b981c876276e3c1e85a0088989a26ee8504` | 170,179 |
| `v15_chat-v15-recall-s0_heldout_n256.json.gz` | `628814824ee31c69ada66f89ac62131ec74f14cd2c96747ed71c21da563edfa8` | 2,749,552 |
| `v15_chat-v15-recall-s1_config.json` | `1cabcc9c8185d09d236569b48e45de87ea67875bfbeead7a817dd517a6d50e0d` | 1,210 |
| `v15_chat-v15-recall-s1_summary.json` | `4b3552468a117c39d9fa03fc554b1910fa31c9f8966bd962d81dec2755d61f4a` | 1,601 |
| `v15_chat-v15-recall-s1_evals.json` | `720c73e71f9fe63f73dc65e83b7864fba5e47391ae7f0958077c601dcb1ced10` | 2,180 |
| `v15_chat-v15-recall-s1_memory_probe.json` | `9840e87e9b493819872888709e643b2ec4b5b7ec1e5d5d5811bb16de8db6be18` | 82,033 |
| `v15_chat-v15-recall-s1_memory_probe_v2_n8.json` | `9ba88b81fac855f8f7a6d5c270ec1b010b14e08a841d50931652c7cf5e7a8d03` | 606,390 |
| `v15_chat-v15-recall-s1_memory_probe_v2_n256.json` | `b575360e502c448f57716ec4969875f46c357216f2aaf4717d07ed84b3f8f227` | 436,944 |
| `v15_chat-v15-recall-s1_binding_probe.json` | `ae3c2d22b1d9f601dfd96cfd01106453017d4c2d834de5d5b37f02c618a08085` | 22,998 |
| `v15_chat-v15-recall-s1_battery.json.gz` | `079c7f4dd2d1d7cdd49e5f3436542831d261787994fef24d88c4ef9dfc292b2c` | 169,785 |
| `v15_chat-v15-recall-s1_heldout_n256.json.gz` | `3b240d091fbd7273c01950a781918f1707d888c3638b076db9898dca4ea10263` | 2,761,542 |
