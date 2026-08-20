# QUALITY v12 — the primary bar passed, and the round does not ship

**2026-08-20.** Four training seeds of `chat-v12-rare`, scored against the
re-scored incumbent `chat-v3d-aligned` on three noun lists.
`experiments/chat/SHIPPED` is **unchanged** — still `chat-v3d-aligned/ckpt_best.pt`,
still no checkpoint change since 2026-08-06.

Not part of the Phase-1..5 research protocol. No number here is a reported
figure. Rates follow [`CONVENTIONS.md`](CONVENTIONS.md): every rate carries its
denominator and a 95 % Wilson interval, and `unresolved` is a verdict.

**Summary.** The round's primary absolute bar **passed by nearly 2×**, and the
arm still does not ship, because the paired comparison the same section
pre-registered came back **unresolved with both gates pointing at the
incumbent**. The design's effect estimate was close to right about the arm
(predicted 0.184, measured 0.202) and wrong about the baseline by **5.8×**
(predicted 0.038, measured 0.219). Flattening the request slot moved the arm to
roughly where the model already was.

---

## 1. The scoreboard

Pooled over four training seeds. `WIDE` is 60 prompts × 6 sampler seeds × 4
training seeds = 1,440 draws; `HELDOUT` and `FRESH` are 20 prompts each, 480
draws; `story_dodge` is 288.

| # | prediction | bar | measured | verdict |
|---|---|---|---|---|
| **P16** | arm's pooled `WIDE` oracle | ≥ 150.5/1440 (0.10451) | **291/1440 = 0.20208**, CI [0.1821, 0.2236] | **PASS** |
| **P17** | arm beats incumbent on `WIDE`, **both** gates at p < 0.05 | McNemar **and** sign | McNemar p = **0.204**, sign p = **0.211**, **both favouring the incumbent** | **UNRESOLVED** |
| **P18** | guard: pooled `HELDOUT` oracle | ≥ 216.5/480 (0.45104) | **243/480 = 0.50625**, CI [0.4617, 0.5507] | **PASS** |
| **P19** | guard: pooled `story_dodge` | ≤ 72.5/288 (0.25174) | **22/288 = 0.07639**, CI [0.0510, 0.1129] | **PASS** |

**A16 does not fire.** It fails the round when P16 and P17 pass and P18 does
not; P17 did not pass, so the conjunct is not reached.

**Stopping rule 3 decides the round**: *"No arm displaces `SHIPPED` on an
`unresolved` margin, whatever its point estimate."* P17 is unresolved.
The arm does not ship.

### The full comparison, both arms, all three lists

Oracle = what the pool contained (coverage); selected = what the decoder
returned. Positive direction is higher.

| list | arm `rarest` | incumbent `head` |
|---|---|---|
| `WIDE` oracle | 291/1440 = 0.20208 [0.1821, 0.2236] | **315/1440 = 0.21875** [0.1982, 0.2408] |
| `WIDE` selected | 278/1440 = 0.19306 [0.1735, 0.2142] | **305/1440 = 0.21181** [0.1915, 0.2337] |
| `HELDOUT` oracle | 243/480 = 0.50625 [0.4617, 0.5507] | **261/480 = 0.54375** [0.4990, 0.5878] |
| `FRESH` oracle | 75/480 = 0.15625 [0.1265, 0.1914] | **97/480 = 0.20208** [0.1686, 0.2403] |
| `story_dodge` | **22/288 = 0.07639** [0.0510, 0.1129] | 28/288 = 0.09722 [0.0681, 0.1369] |

**The arm is behind on every coverage list and ahead on the damage guard**, and
every one of those intervals overlaps. The only column where the arm is ahead is
the one P19 was written to bound.

### P17 per training seed, before pooling

`scripts/chat/compare_holdout.py --field oracle_hit`, incumbent as baseline:

| seed | incumbent | arm | per draw | McNemar p | per prompt | sign p | verdict |
|---|---:|---:|---|---:|---|---:|---|
| s0 | 70/360 | 78/360 | +41 / −33 | 0.416 | +16 / −14 | 0.856 | unresolved |
| s1 | 87/360 | 72/360 | +36 / −51 | 0.133 | +12 / −21 | 0.163 | unresolved |
| s2 | 67/360 | 68/360 | +38 / −37 | 1.000 | +17 / −16 | 1.000 | unresolved |
| s3 | 91/360 | 73/360 | +37 / −55 | 0.076 | +12 / −18 | 0.362 | unresolved |
| **pooled** | **315/1440** | **291/1440** | **+152 / −176** | **0.204** | **+16 / −25 / 19 tied** | **0.211** | **unresolved** |

Four seeds, four `unresolved`, and the two that lean hardest lean toward the
incumbent. **Reported as it fired.** No seed is dropped and none is added:
stopping rule 1 fixed four before the round began.

---

## 2. The bar passed because it was set against a baseline that was never there

P16 is an **absolute** bar: 0.10451 on `WIDE`. It was placed there because §1 of
the pre-registration predicted an incumbent of ~0.038 and an arm of ~0.184, and
0.10451 sits between them at a half-integer count so the lattice cannot land on
it exactly.

| quantity | predicted | measured | ratio |
|---|---:|---:|---:|
| arm on `WIDE` | ~0.184 | **0.20208** | 1.10× |
| **incumbent on `WIDE`** | **~0.038** | **0.21875** | **5.76×** |

**The estimate was close about the treated arm and wrong about the control.**
That is the whole round in one row: a bar drawn between two predictions is only
as good as the worse of them, and the arm cleared a threshold that the model it
was supposed to beat had already cleared before any corpus was repacked.

**Where the 0.038 came from, and why it was low.** `WIDE`'s selection rule is
the incumbent's *own* request counts — the corpus's eligible subjects requested
by `head` 20..400 times over the full scan — and the predicted rate is the
five-band step model of §1 evaluated in that band. The pre-registration already
labels that model's standing honestly: *"a five-band step model fitted to 40
nouns whose log-linear form scores R² = 0.48 … the point value 0.1835 is not
load-bearing."* It was not load-bearing for the arm. **It was load-bearing for
the incumbent, and nothing in the document said so.** The model under-predicts
coverage in the 20–400 band by a factor of ~5.8, which is a finding about the
instrument that designed this round rather than about either checkpoint.

**The scope claim the pre-registration reserved is not earned.** §1 says a pass
licenses *"for nouns the incumbent requests 10..200 times in the pack,
flattening the request slot raises coverage"*. P17 is what would have licensed
it, and P17 is unresolved. **It is not claimed here.**

---

## 3. The premise the power analysis rested on, and how it failed

The design's power came from a one-directional argument, quoted from §1:

> From the dry run, 18 of the 60 `WIDE` nouns cross the 200-request threshold
> under `rarest` and **none falls below it**. At 18/0 discordant prompts the
> sign test gives p ≈ 7.6e-6 … So the design resolves if **at least six prompts
> move and none moves back**.

Pooled, **16 prompts moved up and 25 moved back**, with 19 tied. The premise was
not that few prompts would move — 41 of 60 did — but that they would move *one
way*. They did not, and a sign test on 16/25 cannot resolve however many sampler
seeds are drawn, exactly as §0's own argument about `FRESH` says.

That the requests *crossed the dose threshold* is not in doubt; the dry run
measured that before the corpus was packed, and it writes no model. What did not
follow is that crossing it raises coverage for that noun and lowers it for none.

---

## 4. What the arm did do

**Both guards held, and one of them by a distance.** `story_dodge` came in at
**22/288 = 0.07639** against a bar of 0.25174 — and *below* the incumbent's
28/288 = 0.09722, though the intervals overlap and this is not a resolved
difference. The characteristic failure the pre-registration named for `rarest` —
*"a reply that names the subject and then wanders into generic narrative"* — is
not visible. `P18` held at 0.50625 against a 0.45104 floor.

So the round is not a damaged model. It is a model that was moved sideways: the
arm pays a little coverage on every list and buys nothing measurable back.

**The `topic_of` adjective defect is unchanged**, as §2 of the pre-registration
said it would be: both arms share it, and fixing it mid-comparison would have
changed the corpus underneath the round.

---

## 5. What this leaves

1. **The request-slot lever is not refuted; it is unresolved, and the design
   that would resolve it is not this one.** The arm and the incumbent differ by
   24 draws in 1,440. To resolve a difference that size the round needs either
   many more prompts or an effect several times larger, and `WIDE` at 60 nouns
   is already the widest list this subsystem has.
2. **The five-band step model should not design another round until it is
   re-fitted.** It missed the incumbent's rate in the band it was used to select
   by 5.8×. Its R² of 0.48 was reported; its error *on the control arm* was not
   bounded anywhere.
3. **A bar between two predictions needs both of them guarded.** P17 caught this
   and the round is intact because of it — this is the shape
   `PREDICTION_v9.md`'s P15 and `CONTRIBUTING.md` §2 both ask for, working. The
   transferable rule is narrower than "add a paired gate": it is that an
   *absolute* primary bar derived from a predicted baseline is only as
   trustworthy as that baseline, and the prediction for the control deserves the
   same scrutiny as the prediction for the arm.
4. **Coverage is still conditioning** — `QUALITY_v11.md` §1 is untouched by this
   round. What this round tested is whether *flattening which nouns get asked*
   converts that into coverage, and the answer at this effect size is: not
   measurably, and not for free.

---

## 6. Reproducing

```bash
# the four arm seeds and the re-scored incumbent, per list
python scripts/chat/echo_holdout.py --set wide    --seeds 6 --n 256   # 360 draws
python scripts/chat/echo_holdout.py --set heldout --seeds 6 --n 256
python scripts/chat/echo_holdout.py --set fresh   --seeds 6 --n 256

# P17, per seed
python scripts/chat/compare_holdout.py \
    experiments/chat/_quality/v12_wide_chat-v3d-aligned.json \
    experiments/chat/_quality/v12_wide_chat-v12-rare.json --field oracle_hit
```

Artifacts: `experiments/chat/_quality/v12_{wide,heldout,fresh,dodge}_{chat-v12-rare,chat-v3d-aligned}{,-s1,-s2,-s3}.json`
and `score_v12.json`. The pooled McNemar and the pooled prompt-level sign test
are computed over the four seeds' paired rows, keyed by `(prompt, seed)` — the
same key `compare_holdout.py` uses within a seed.
