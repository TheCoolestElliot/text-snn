# QUALITY v11 — the request slot, a 3× memory cut, and a request verb wearing a subject's clothes

**2026-08-13.** No training. Three diagnostics that `QUALITY_v10.md` §6 named as
next, all three answered, plus two changes that fell out of them.
`experiments/chat/SHIPPED` is unchanged — still no checkpoint change since
2026-08-06.

Not part of the Phase-1..5 research protocol. No number here is a reported
figure. Rates follow [`CONVENTIONS.md`](CONVENTIONS.md).

**Summary.** Coverage is a **conditioning** problem, not an exposure one, and
the corpus drills 35 nouns for a third of all requests. Inference memory in the
two-compartment scan is cut **3×**, which retired the only objection to n = 256
and made it the default — the first pool size to resolve `above` on both noun
lists. And `lambda` still earns its pass, but finding that out exposed the echo
partition *causing* `story_dodge`, through a request verb matching a story's
signature.

---

## 1. Coverage is CONDITIONING, not exposure

The question `QUALITY_v10.md` §6 left: the model rarely drafts about an
unfamiliar noun, but is that because it has barely *read* the word, or because
it was rarely *asked* for it? They imply opposite next rounds — more text
against a repack — and `QUALITY_v9.md` §1 already spent 2.68 GPU-hours ruling
out a third answer (register).

Both are readable straight out of the packed corpus. A conversation is
`<|user|> request <|eot|> <|bot|> reply <|eot|>`, so counting a noun inside user
turns gives **asked** and counting it everywhere gives **seen**.
`scripts/chat/subject_frequency.py` does that over `stories_topic` +
`tinystories` (1.1 GB) for all 40 nouns of the two held-out lists, against the
per-noun oracle rate at n = 128.

| rank correlation with pool coverage, n = 40 nouns | rho | p |
| --- | ---: | ---: |
| asked | **+0.790** | 0.0000 |
| seen | +0.761 | 0.0000 |
| asked, within `heldout` only | +0.790 | 0.0000 |
| asked, within `fresh` only | +0.707 | 0.0005 |

It is not a between-list artefact — it holds inside each list separately. But
the marginals cannot decide anything, because **asked and seen correlate at
+0.969**: a story about a penguin is what makes "penguin" available as a subject
in the first place. The partials do decide it:

| | rho | p |
| --- | ---: | ---: |
| hit ~ **asked**, holding seen fixed | **+0.328** | 0.0416 |
| hit ~ **seen**, holding asked fixed | −0.027 | 0.8702 |

**At matched exposure, being asked more predicts a hit. At matched asking, being
read more predicts nothing.** All four nouns never asked for at all (lighthouse,
windmill, seashell, submarine) score exactly 0.000.

Read the strength honestly: +0.328 at p = 0.042 is marginal, it is one of
several correlations computed here, and the near-collinearity leaves a thin
residual to estimate from. What raises it above a curiosity is that it
**post-dicts an expensive result already in hand** — A12 added 218 M characters
of adult prose, which is `seen` and not `asked`, and it failed its bar. The
diagnostic predicts that outcome from corpus statistics alone.

### The request slot is far narrower than the corpus

`snnchat.topics.topic_of` sorts candidate subjects by `-counts[w]` and
`story_request` takes `topics[0]` — the most frequent content word in the story.
TinyStories is about birds, balls and cats, so that is what the request slot
fills with:

```
2169 distinct subjects, 298709 requests
    35 subjects asked for >= 1000 times (33.9% of requests)
   149 subjects asked for >=  300 times (52.4% of requests)
   808 subjects asked for >=  100 times (92.8% of requests)
most-drilled: bird 10847, ball 10570, cat 9328, tree 6977, toy 6178, park 6049
```

"A story about a penguin" (asked 151) is competing against "bird" (10,847). And
`topic_of` returns up to **three** eligible subjects per story — each already
in-window, used as a noun, and not a character name — of which two are discarded
except in a 10 % two-topic branch. **Choosing among the three against corpus
frequency instead of always taking `topics[0]` would flatten the request
distribution at zero data cost.** That is a repack, not a corpus, and it is the
obvious next arm. It is *not* run here and nothing in this document says it
works.

One caution for whoever writes that pre-registration:
`build_topic_stories.py`'s own comment records that re-weighting was considered
and rejected as zero-sum, because measured against `1/count^0.5` **14 of 15 word
probes LOSE dose**. That measurement is correct and it is on the battery's own
probes, which are `rabbit`(717), `cake`(1348), `boat`(1052) — nouns that are
already well drilled. Of course flattening costs them. The fresh list is what
shows the other side of that trade exists (oracle 0.2083 against heldout's
0.5500), and it did not exist when the dose note was written. **The lever was
closed on a probe set that cannot see the failure it addresses.**

## 2. A 3× cut to inference memory, and n = 256 becomes the default

`FusedTwoCompartmentScan.forward` stacks three `[B, L, d]` tensors: the spikes,
which are the output, plus `v_pre` and `vs_seq`, which exist only for the
backward. Under inference they are built, saved and dropped.

It cannot be conditioned on inside the autograd Function — `ctx.needs_input_grad`
reports whether the *parameters* want gradients, which they do in `eval()` just
as much as in training, and `torch.is_grad_enabled()` is always False inside a
`forward`. The dispatcher is the only place that can answer, so
`snn.twocomp.twocomp_scan` answers it and calls a new
`twocomp_scan_inference`.

Measured at `B × L × d = 256 × 310 × 1024`, one scan:

| | peak | wall |
| --- | ---: | ---: |
| with the backward tensors | 1861 MiB | 14.1 ms |
| inference path | **623 MiB** | 12.7 ms |

Bit-identical against the Function over the same shape and regime sweep the R10
gate uses; the gate has a sixth leg now
(`test_the_inference_path_is_bit_identical`), plus a routing test — a path that
is correct and never taken saves nothing, and one taken while a gradient is
wanted would silently return a tensor with no `grad_fn`. Mutation anchors
re-validated, 59/59 intact.

This is the peak that matters: the decode loop runs at L = 1, and the memory
ceiling of a turn is the anti-LM pass, which runs at `B = n, L ≈ 310`.
Re-measured end to end, median of 4 turns at `--max-new 400`:

| n | 8 | 32 | 64 | 128 | 256 | 512 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| s/turn | 0.29 | 0.31 | 0.39 | 0.55 | **0.70** | 1.15 |
| GiB | 0.17 | 0.39 | 0.65 | 1.17 | **2.21** | 4.24 |

`QUALITY_v10.md` §5 rejected n = 256 on 3.78 GiB and on nothing else — it agreed
256 was better. At 2.21 GiB that objection is gone, so **`DEFAULT_RERANK_N` is
now 256**, and it is the first pool size to come back `above` rather than merely
directional on **both** lists at once:

| set | n=128 | n=256 | per draw | per prompt | verdict |
| --- | ---: | ---: | --- | --- | --- |
| heldout | 0.4167 | **0.5000** | +11 / −1, p = 0.0064 | +6 / −0, p = 0.031 | **above** |
| fresh | 0.1333 | **0.2083** | +9 / −0, p = 0.0039 | +6 / −0, p = 0.031 | **above** |

512 is the new cliff at 4.24 GiB and is not measured for quality.

## 3. `lambda` earns its pass — and the echo partition was undoing its job

**The question.** `lambda = 0.6` was never chosen for topicality; the headline's
argmax was `lambda = 0`. It was chosen because `lambda = 0` answers a prompt
asking for no narrative WITH a narrative, and the anti-LM term cancels that. So
"does lambda still pay" is a `story_dodge` question. It costs a full
teacher-forced pass over the pool, which is now the largest remaining item in a
turn.

**The design.** One pool per (prompt, seed) over the 12 non-narrative probes ×
6 seeds, selected twice. Lambda changes only `score`, so this is exactly paired
and `_is_story` is a pure text function, so neither arm needs its own draw.

**The answer: keep it.** Resolved `above` in both regimes.

| | lambda=0 | lambda=0.6 | fixes / breaks | p |
| --- | ---: | ---: | --- | ---: |
| echo ON | 0.1806 | **0.0972** | 6 / 0 | 0.031 |
| echo OFF | 0.3333 | **0.0000** | 24 / 0 | 0.00000 |

### The finding that was not the question

Look at the two rows again. With the echo partition ON, `story_dodge` at the
shipped lambda was **0.2778**; with it OFF, **0.0000**. The partition was
causing the exact failure lambda is installed to prevent, on every prompt kind
that is not a story request.

The mechanism is precise. `echo_count` matched a prompt word as a **prefix on a
word boundary**, to let "rabbit" find "rabbits". On `"name three animals"`,
`name` is a content word, `\bname` matches **"named"** — and `" named "` is the
literal substring `snnchat.quality._is_story` uses to detect the story register.
So the tier promoted *"Three years old girl named Emma"* over *"A dog, a cat,
and a bird"*, because the story echoed the request verb at IDF **8.66** and the
correct answer echoed nothing at all. `name` was the echoed word in **16 of the
20 dodges**.

**The fix is a whole-word match with the same plural tolerance**
(`echoes`, one function now instead of two copies). It keeps every behaviour the
prefix rule was documented for — rabbit/rabbits, dragon/dragons — and has no
tuned constant.

| | story_dodge, echo ON, lambda 0.6 |
| --- | ---: |
| prefix match (shipped through v10) | 0.2778 (20/72) |
| whole word | **0.0972 (7/72)** |
| | 14 fixed, 1 broken, McNemar **p = 0.00098** |

**And it costs nothing on topic prompts**: heldout 0.5000 (60/120) and fresh
0.2083 (25/120), *identical counts* before and after. The two noun lists were
re-run in full rather than argued about.

Stated as a caveat: this rule was chosen after seeing the 72 draws it is scored
on. What limits the concern is that the mechanism was identified causally rather
than by search, the fix has no free parameter, and the validation set that
matters — both topicality lists — was held fixed and did not move.

## 4. What this leaves

The pool is still the constraint and it is now a *named* one: the request slot
drills 35 nouns for a third of its requests, and being in that slot is what
predicts a hit. The next arm is a repack that flattens it, and
`build_topic_stories.py` already computes the two discarded alternatives it
would draw from.

## 5. Reproducing

```bash
python scripts/chat/subject_frequency.py
python scripts/chat/lambda_dodge.py --seeds 6 --n 256
python scripts/chat/echo_holdout.py --set fresh --seeds 6 --n 256
python -m pytest -q                                  # 473 tests
```

Artifacts: `experiments/chat/_quality/subject_frequency.json`,
`lambda_dodge.json` (prefix) and `lambda_dodge_wholeword.json`,
`v10_{heldout,fresh}_n256.json` (pre-matcher-fix) and `v11_*` (post).
New code: `scripts/chat/subject_frequency.py`, `scripts/chat/lambda_dodge.py`,
`snn.twocomp.twocomp_scan_inference`, `snnchat.rerank.echoes`.
