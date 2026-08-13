# PREDICTION v12 — the request-slot round

**Written 2026-08-13, before any arm of this round has been trained.** The
corpus it tests was packed after this document was committed; the dry runs it
quotes write no corpus and touch no model.

Not part of the Phase-1..5 research protocol. No number here is a reported
figure. Rates follow [`CONVENTIONS.md`](CONVENTIONS.md).

---

## 0. What v11 established, and why it dictates this round

`QUALITY_v11.md` §1 separated the two hypotheses that survive for the topic
failure. Whether this model drafts a reply about a noun is predicted by how
often it was **asked** for that noun (partial rho +0.328, p = 0.042) and not at
all by how often it has **read** it (partial rho −0.027, p = 0.87). The two
correlate at +0.969, so only the partials decide, and the finding post-dicts
`QUALITY_v9.md`'s A12 — 218 M characters of adult prose is `seen`, not `asked`,
and it failed its bar at 2.68 GPU-hours.

The request slot is where `asked` is decided, and it is narrow.
`snnchat.topics.story_request` has always taken `topics[0]`, the most frequent
content word of the story, and TinyStories is about balls, birds and cats.
Measured over the raw corpus by `scripts/chat/subject_table.py`: 3,335 words are
ever requested, and **140 of them carry 49.3 % of all requests**.

`topic_of` computes up to **three** eligible subjects per story — each already
in-window, used as a noun, and not a character name — and discards two of them
except in a 10 % branch. This round spends that slack.

### What the dry runs already settled, before any GPU time

`build_topic_stories.py --dry-run` streams the identical generator the packer
streams and histograms the requests it would emit, writing nothing. Two CPU
minutes per policy. It changed this round's design twice, which is the entire
argument for running it first:

| policy | subjects ≥ 200 requests | most-drilled | predicted `FRESH` |
| --- | ---: | --- | ---: |
| `head` (shipped) | 392 | bird 10904 | 0.1586 |
| `inverse`, weight `1/(1+count)` | 547 | park 4543 | +0.049 |
| `rarest`, ascending availability | 669 | ball 4172 | +0.049 |

**First correction: `inverse` was the wrong policy.** It reaches only about half
of each noun's availability ceiling, because a rare word competing against
another rare word loses half the time. `penguin` is eligible in ~254 pack
stories, requested 130 times under `head`, and 141 under `inverse` — it barely
moves. `rarest` is deterministic, parameter-free, and gets much closer to the
ceiling.

**Second correction, and the reason `WIDE` exists: the twenty-noun lists cannot
resolve this.** The predicted effect is nouns *crossing a dose threshold*, and
on `FRESH` only two cross (`dolphin`, `igloo`). The per-prompt sign test's
smallest attainable two-sided p is `2^-(k-1)` on `k` discordant prompts, so two
or three moving prompts cannot reach 0.05 **however many sampler seeds are
drawn** — more seeds shrink draw noise, only more prompts shrink prompt-set
noise. That is a property of the design, known in advance, and it is why this
round scores a sixty-noun list.

## 1. The primary gate, fixed here

> **Primary: held-out ORACLE topicality on the `WIDE` list**,
> `scripts/chat/echo_holdout.py --set wide --seeds 6 --n 256`, reported as
> k/360 per training seed and pooled as k/1440 over four, with a 95 % Wilson
> interval.

**The oracle and not the selected rate**, for the reason `PREDICTION_v9.md` §1
gives: this round changes what the model *drafts*, and the echo tier already
attains the pool's ceiling at n ≤ 32 (`QUALITY_v11.md` §4). Scoring selection
would measure a component this round does not touch.

**Why `WIDE` and not `FRESH`.** `FRESH` is 20 nouns and cannot resolve a
threshold-crossing effect (§0). `WIDE` is 60 nouns chosen by a rule stated in
`echo_holdout.py` and fixed before any scoring: the corpus's own eligible
subjects, requested by the **incumbent** 20..400 times over the full scan,
disjoint from `quality.PROBES`/`HELDOUT`/`FRESH`, then filtered by hand on
grammaticality alone.

**The scope this buys, stated plainly and not to be widened afterwards.** `WIDE`
is selected on a *pre-treatment* covariate — the incumbent's own request counts,
never the flattened corpus's — but it is deliberately the band where the
intervention can act. So a pass licenses "**for nouns the incumbent requests
10..200 times in the pack, flattening the request slot raises coverage**". It
does **not** license "the model got better at nouns", and `HELDOUT` and `FRESH`
are reported either way to show what happens outside the band.

**Lattice.** Per training seed the denominator is 60 prompts × 6 sampler seeds =
360, quantised at 1/360 = 0.00278; pooled over four seeds it is 1440, quantised
at 0.000694. Every threshold below sits at a **half-integer count** so it cannot
be hit exactly — `CONVENTIONS.md` §4 rule 2, which P1 died of.

**Power, stated before running.** From the dry run, 18 of the 60 `WIDE` nouns
cross the 200-request threshold under `rarest` and **none falls below it**. At
18/0 discordant prompts the sign test gives p ≈ 7.6e-6; even if only 6 of the 18
materialise it gives p = 0.031, and 5 gives 0.0625 and fails. So the design
resolves if **at least six prompts move and none moves back**. At draw level the
predicted shift is ~52 net discordant of 360, far above McNemar's ~20.

**The effect estimate is a guide, not a promise.** It comes from a five-band
step model fitted to 40 nouns whose log-linear form scores R² = 0.48. The
*direction* and the *count of crossings* are what the design leans on; the point
value 0.1835 is not load-bearing and the bar is set well below it.

**Secondary, reported always, gating nothing**: `FRESH` and `HELDOUT` oracle and
selected rates, `story_dodge` with its interval, `list_strict`, `canned_rate`,
the realised subject distribution of the packed corpus, and weighted held-out
bpc per source. **Secondary columns may not be promoted to primary after the
fact.** Weighted bpc is **not comparable** to the incumbent's 1.1743: the
mixture keeps every source and weight but `stories_topic_rare` is a different
400 M characters with its own held-out tail, so the number is reported because
this document says to report it and for no other purpose.

**Seed counts are fixed here and honoured whatever the interim numbers say.**

## 2. The arm

One arm, one change against `chat-v3d-aligned`'s recipe.

| # | arm | one change | seeds | cost |
| --- | --- | --- | ---: | ---: |
| A16 | `chat-v12-rare` | `stories_topic` → `stories_topic_rare` at the same 0.40 | 4 | ~2.7 GPU-h |

`stories_topic_rare` is `build_topic_stories.py --subject-policy rarest --name
stories_topic_rare`, everything else default (`--seed 7`, `--raw-fraction 0.15`,
`--limit-chars 400000000`). It packs to a new name and the packer refuses to
overwrite `stories_topic`, so the eight committed checkpoints trained on that
file stay reproducible.

**The control is the four existing incumbent checkpoints, re-scored, and no
training.** `chat-v3d-aligned` and `-s1..-s3` were trained on the identical
recipe and the identical mixture; the only difference this round introduces is
which 400 M characters `stories_topic` names. Re-scoring frozen checkpoints with
today's decoder is not the thing decision #10 forbids — it *is* measuring on
today's tree rather than quoting a committed figure, which is what #10 asks for.
It also halves the round's GPU cost and removes training-seed drift from the
comparison entirely, because the control's seeds are the ones already on disk.

### The prediction, written before the run

**P16 (primary): `chat-v12-rare`'s pooled `WIDE` oracle ≥ 150.5/1440
(0.10451)**, against a predicted incumbent of ~0.038 and a predicted arm of
~0.184. *Expected outcome: PASS, and this is the round's real bet.*

**P17 (primary, paired): the arm beats the re-scored incumbent on `WIDE` under
BOTH gates** — McNemar over the 1440 paired draws and a sign test over the 60
prompt-level totals, each at p < 0.05, per `scripts/chat/compare_holdout.py`.
A single gate firing is `unresolved`, not a pass.

**P18 (guard, anti-damage): pooled `HELDOUT` oracle ≥ 216.5/480 (0.45104)**,
against the incumbent's measured 0.5500 at n = 256. Flattening is a *trade* and
this bounds the side it is expected to lose. **A16 fails the round if P16 and
P17 pass but P18 does not** — the conjunct exists for the reason
`CONTRIBUTING.md` §2 requires one and `PREDICTION_v9.md`'s P15 established:
without it an arm can win on the column it was built for while being worse at
the thing the model is for.

**P19 (guard): pooled `story_dodge` ≤ 72.5/288 (0.25174)**, against the
incumbent's measured 0.0972. `rarest` requests the least-central eligible
subject, so the arm's characteristic failure would be a reply that names the
subject and then wanders into generic narrative. This is the column that sees
that.

**Risk stated in advance.** Every candidate `rarest` picks has already passed
`topic_of`'s filters — recurs at least twice, appears within 220 characters,
used as a noun — so a flattened request still names something the story is
genuinely about, just less centrally. The arm's failure mode is therefore a
*weaker* request-to-story correspondence, not a nonsensical one, and P18/P19 are
where it would show. A second, smaller risk: `rarest` draws no random numbers,
so the corpus is a deterministic function of the seed and any pathology in it is
systematic rather than averaged away.

**A defect found while building `WIDE`, recorded and NOT fixed this round.**
`topic_of` admits adjectives, because "a wealthy man" satisfies its determiner
test, so the shipped corpus contains requests of the form "tell me a story about
a wealthy". It predates this round and both arms share it. Fixing it would
change the corpus underneath a comparison this round is in the middle of
running, so it is documented and left, per `CONVENTIONS.md`'s rule that closed
records are not rewritten to accommodate later results.

## 3. Stopping rules

1. **Four seeds, decided now, run regardless of interim results.** If wall-clock
   forces a stop before all four finish, that is reported as **insufficient n**,
   not as whatever partial comparison is sitting on disk.
2. **Any seed that hits a gradient-norm rollback is reported as rolled back and
   is not silently pooled.**
3. **No arm displaces `SHIPPED` on an `unresolved` margin**, whatever its point
   estimate.
4. **The noun lists are frozen.** `WIDE`, `FRESH` and `HELDOUT` are committed
   before the arm trains and are not extended, trimmed or re-curated afterwards.

## 4. What this round cannot settle

- **Nouns the corpus has no stories about.** `windmill` is an eligible subject
  in 4 stories of 1.01 M and `submarine` in essentially none. No selection
  policy can request what `topic_of` never proposes, so the floor of the `FRESH`
  list is untouched by construction and a null there is not evidence against the
  mechanism.
- **Whether the threshold near 200 requests is real.** It is read off five bands
  over 40 nouns and it is the shape the design leans on. This round tests a
  prediction *derived* from it; it does not measure the dose-response curve.
- **Which of `rarest` and `inverse` is better.** Only `rarest` runs. The dry run
  ranks them on predicted dose, not on trained behaviour, and a second arm was
  not bought.
- **Whether flattening helps the model or only the metric.** The oracle asks
  whether any of 256 drafts names the requested noun. A corpus that teaches the
  noun-naming form more widely raises that by construction; whether the replies
  are *better stories* is not measured here and `story_dodge` is only a partial
  proxy.
