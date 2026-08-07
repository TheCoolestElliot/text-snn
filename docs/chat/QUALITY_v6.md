# Chat quality, round 6 — a from-scratch run, the instruction-weight ladder, and closing out P1

**Date:** 2026-08-06
**Author:** Elliot Asher Caudill
**Status:** IN PROGRESS. This file is written incrementally as each part of the
round completes; sections below are appended, not edited once landed, mirroring
`QUALITY_v5.md`'s own treatment of `QUALITY_v4.md`.
**Predicts:** `docs/chat/PREDICTION_v6.md` (written before `chat-v6-scratch`
launches).
**Does not modify:** `QUALITY.md`, `QUALITY_v4.md`, `QUALITY_v5.md`, or
`PREDICTION_v4.md`/`PREDICTION_v5.md`, all closed.

---

## 1. Is the incumbent actually ahead of its strongest challenger? (a free check, no GPU)

`QUALITY_v5.md`'s own verdict was that `chat-v3d-aligned` "beats this arm on the
pre-registered headline 0.2969 to 0.2779" and `SHIPPED` was left unchanged on
that basis. That comparison has never been read with an interval. Before
spending any GPU-hour building on top of "what's shipped," this section reads
it the way `docs/chat/CONVENTIONS.md` requires.

**Method.** `docs/chat/CONVENTIONS.md` gives Wilson CIs for a single rate
against a threshold. A headline is a linear combination of four component
rates over four *disjoint* probe groups (`0.5·topic + 0.3·list + 0.2·social −
0.2·fallback`, `src/snnchat/quality.py::score`) at the pinned `n=1, λ=0` row
(`report["baseline_picks"]`, i.e. the first row of `DEFAULT_GRID`, which
`compare_quality.py` and every prior round have used as the "plain sampler"
reference row). `topic`, `social` and `fallback` are Bernoulli and get a Wilson
`rate_ci`; `list` is not (values in {0, ⅓, ⅔, 1}) and gets a sample SE per
CONVENTIONS.md §2. The headline's SE is the components' SEs combined by their
fixed weights, **treating the four groups as independent** — an approximation,
not exact, because `fallback` is computed on the same draws as `topic`+`list`
(the union of `topic`/`list`/`fact`, per `_SUBSTANTIVE`), so a hit and a
fallback flag on the same draw are not independent events. This is the same
approximation an informal check made in the prior (uncommitted, unexported)
session; it is formalised here, not improved on, and the caveat is the point of
writing it down. Two-sample z = (rate₂ − rate₁) / sqrt(se₁² + se₂²); |z| < 1.96
is `unresolved`, matching `resolves_against`'s own boundary.

**Result**, on the committed `experiments/chat/_quality/chat-v3d-aligned.json`
and `chat-v5a-short300.json`, `n=1, λ=0`:

| component | `chat-v3d-aligned` (shipped) | `chat-v5a-short300` (challenger) | diff (challenger − shipped) | SE | z | verdict |
| --- | ---: | ---: | ---: | ---: | ---: | :-- |
| `topic` | 0.0938 (6/64) [0.044, 0.190] | 0.1094 (7/64) [0.054, 0.209] | +0.0156 | 0.0533 | 0.292 | unresolved |
| `list` | 0.1667 (n=20), se 0.0820 | 0.0833 (n=20), se 0.0534 | −0.0834 | 0.0979 | −0.852 | unresolved |
| `social` | 1.0000 (20/20) | 1.0000 (20/20) | 0.0000 | 0 / 0 | — | **tied at ceiling — not a comparison** (both arms saturate 20/20; a Wald SE of 0 here is a boundary artefact, not evidence of certainty) |
| `fallback` | 0.0000 (0/112) [0, 0.033] | 0.0089 (1/112) [0.002, 0.049] | +0.0089 | 0.0089 | 1.000 | unresolved |
| **headline** | **0.2969** | **0.2779** | **−0.0190** | **0.0397** | **−0.479** | **unresolved** |

**Verdict: UNRESOLVED.** The 0.0190 headline gap `QUALITY_v5.md` read as a loss
for `chat-v5a-short300` does not survive a 95 % read (z = −0.479, far inside
±1.96); every component gap is individually unresolved too, and `social` is a
tie at the metric's ceiling for both arms. **This does not mean the arms are
equal — it means 20 draws per component cannot tell them apart, in either
direction.** `chat-v3d-aligned` remains `SHIPPED` because the standing rule is
that an unresolved comparison does not displace the incumbent (§0 below), not
because this round found it superior. This is the first time that distinction
has been written down rather than implied by a bare fraction.

**What this decides.** Per the standing rule, Phase 1's from-scratch run trains
on the **incumbent's** corpus recipe (`stories_topic` at 0.40, `chat-v3d-aligned`'s
mixture), not the challenger's, because the comparison did not resolve in the
challenger's favour. See `PREDICTION_v6.md` §0.

**What this does not decide.** This says nothing about `story_dodge`'s P1
boundary case specifically (`chat-v5a-short300` vs. its own pre-registered
threshold, not vs. another arm) — that is Phase 4's resample, run at 301 seeds
because 20-and 48-draw samples are visibly too small to resolve differences
this size, as this section's own numbers now show directly.
