"""Score the v14 subject tier against the shipped selector, on stored pools.

NOT PART OF THE RESEARCH PROTOCOL. `snnchat` is outside the Phase-1..5 study.
Draws nothing, trains nothing, touches no GPU: every number here is a replay of
candidates that are already on disk.

WHAT IS COMPARED
----------------
Two selection rules on ONE pool per (prompt, sampler seed): the shipped weighted
echo tier, and `RerankParams.subject_tier` (`snnchat.rerank`, module docstring).
The pool is held fixed, so the comparison is exactly paired and McNemar is the
right test -- the design of `echo_holdout.py`, for the same reason.

THE REPLAY IS THE REAL TIER FUNCTION, NOT A COPY OF IT
------------------------------------------------------
`replay` rebuilds the length partition exactly as `snnchat.rerank.select` writes
it, then hands stand-in objects to the REAL `snnchat.rerank.echo_tier`, then
applies `select`'s own `decided` rule (a one-member tier wins without a score)
and its own score, `(logp_cond - lam * logp_null) / max(n_scored, 1)`.
`echo_tier`'s docstring names the hazard this avoids: `echo_holdout.py` carries
a hand copy of the weighted tier, and a copy measures a selector that is only
believed to be the shipped one.

Belief is then removed as well. A pool written by `scripts/chat/v14_pools.py`
records the index each REAL selector returned on the device, and this script
**refuses to report anything** unless its replay lands on both recorded indices
on every draw (`ReplayMismatch`). It also re-derives `subject_hit`, `echo_weight`
and the subject from the stored text and refuses on any disagreement, because a
pool whose stored fields do not describe its stored text is not evidence.

TWO MODES
---------
Confirmatory (the default): `v14_pools.py` files only. Every draw carries a
measured `logp_null` for every candidate, so both rules are replayable on every
draw at the file's lambda. The bars below fire.

`--exploratory`: the COMMITTED `echo_holdout.py`-format pools,
`experiments/chat/_quality/v12_{heldout,fresh,wide}_chat-v3d-aligned{,-s1,-s2,-s3}.json`,
sampler seeds 0-5. These were read before the subject rule existed, so nothing
measured on them is a test of it; they are what `docs/chat/PREDICTION_v14.md`
sizes its bars from. They were not written for this and lack three things, each
recovered as far as it honestly can be and no further:

* `subject_hit` -- recomputed from the stored text with `snnchat.rerank.echoes`,
  which is the function `select` uses, on the string `select` used.
* `n_scored` -- `len(c.scored)` is `n_chars` plus one if the draft closed its own
  turn, and `sample_candidates` only leaves its loop early once EVERY row has
  stopped, so a draft shorter than `max_new` closed and one at `max_new` did not.
  The rule is validated against the per-character log-probabilities the file
  recorded for its own two picks, and the mismatch count is reported.
* `logp_null` -- present only where the shipped tier had more than one member
  (`select` skips the null pass when the tier decides). The shipped rule is
  therefore replayable everywhere, and the subject rule only where the whole
  pool was null-scored or its own tier is a single draft. The lambda replay is
  restricted to the first of those and the restriction is COUNTED; pure
  likelihood (lambda = 0), which needs no null term, is reported separately on
  every draw. An unmeasured `logp_null` is stored as exactly 0.0, and a measured
  sum of log-probabilities over a non-empty reply is never 0.0, which is how the
  two are told apart.

That subset is selected ON the shipped rule's behaviour -- it is the draws where
the shipped tier was NOT a single draft -- so it under-represents the defect the
subject rule is aimed at. Read it as the subset the arithmetic permits, not as a
sample of turns.

THE BARS
--------
Constants below. They are FIXED BY `docs/chat/PREDICTION_v14.md`, which is
written AFTER the exploratory numbers this script produces and BEFORE any
confirmatory pool is drawn; this file computes no threshold from the data it
scores. Verdicts are `pass`, `fail` and `unresolved`. A near-miss is a fail
(`CONTRIBUTING.md` section 3); `unresolved` is a real verdict
(`docs/chat/CONVENTIONS.md` section 1), it is what a bar gets when its point
estimate is on the passing side but the comparison was not decided, and it is
not a pass.

WHAT NONE OF THIS MEASURES
--------------------------
Whether a reply is any good. P2 is the model's opinion of its own text; P3 is a
regex for one failure (`snnchat.coherence`: never validated against a human
rating, blind to a plain interleave); P1 asks whether a noun occurs. GUARD 2 is
here because the one way this change can read as WORSE is sameness, and nothing
in P1-P3 would notice.

    python scripts/chat/score_v14.py --exploratory
    python scripts/chat/score_v14.py experiments/chat/_quality/v14_*.json.gz
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import statistics
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from snnchat.coherence import WINDOW, unintroduced_entities, wilson_interval  # noqa: E402
from snnchat.prime import prime_topic  # noqa: E402
from snnchat.rerank import echo_tier, echo_weight, echoes, prompt_content_words  # noqa: E402

sys.path.insert(0, str(ROOT / "scripts" / "chat"))
from echo_holdout import hit, mcnemar  # noqa: E402
from v14_pools import FORMAT, read_pool  # noqa: E402

QUALITY = ROOT / "experiments" / "chat" / "_quality"

# ---------------------------------------------------------------------------
# THE BARS. Fixed by docs/chat/PREDICTION_v14.md, which is written AFTER this
# script's exploratory numbers and BEFORE any confirmatory pool is drawn.
# Nothing below is computed from the data being scored.
# ---------------------------------------------------------------------------

#: P1, topic non-inferiority: draws only the SHIPPED pick hits must not
#: outnumber draws only the subject-tier pick hits. A bar on two counts.
P1_RULE = "shipped_only <= subject_only"
#: P2, fluency: pooled mean (subject - shipped) of the selected reply's
#: per-character log-probability, in nats per character ...
P2_POOLED_BAR = 0.05
#: ... AND the difference is strictly positive on every one of these lists.
P2_LISTS = ("heldout", "fresh", "wide")
#: P3, unintroduced entities: fixed > broken, exact two-sided McNemar below this.
P3_ALPHA = 0.05

#: The fluency floor: plain sampling's mean per-character log-probability on the
#: battery (`scripts/chat.py`, the comment on the reranking defaults;
#: `docs/chat/QUALITY.md` section 4, the `chat-v3d-aligned` n=1 row). It is a battery MEAN, used here as a
#: per-reply line, which it was never calibrated to be. It is a count reported
#: next to P2, not a bar.
FLOOR = -0.3934

#: `echo_holdout.py` does not store these; they are its argparse default and
#: `RerankParams.min_chars`'s default at the commit the v12 pools were drawn at.
#: The first is checked against every file (no draft may be longer) and through
#: the `n_scored` validation; the second through the reproduction of the file's
#: own recorded picks.
V12_MAX_NEW = 300
V12_MIN_CHARS = 12

#: Characters of a reply that count as its opening, for GUARD 2.
OPENING_CHARS = 60

#: The committed pools exploratory mode reads when no path is given.
EXPLORATORY_FILES = tuple(
    f"v12_{probe_set}_chat-v3d-aligned{suffix}.json"
    for suffix in ("", "-s1", "-s2", "-s3")
    for probe_set in ("heldout", "fresh", "wide")
)

_WORDS = re.compile(r"[a-z']+")


class ReplayMismatch(RuntimeError):
    """The replay and the recorded selection disagree. Nothing may be reported."""


# ---------------------------------------------------------------------------
# the replay
# ---------------------------------------------------------------------------


@dataclass(eq=False)
class Draft:
    """A stored candidate: exactly the fields either selection rule reads.

    `eq=False` so that two drafts with identical text are still two drafts --
    a pool can hold the same reply twice, and the winner is an index.
    """

    text: str
    n_chars: int          #: DRAFT length, before `trim_to_sentence`
    n_scored: int         #: `len(Candidate.scored)`
    logp_cond: float
    logp_null: float
    echo_weight: float
    subject_hit: bool
    #: False when the stored `logp_null` is a 0.0 nobody measured.
    null_measured: bool = True

    @property
    def logp_per_char(self) -> float:
        return self.logp_cond / max(self.n_scored, 1)


@dataclass
class Pick:
    """What one rule did on one pool."""

    index: int | None     #: None: not replayable, a needed null term is missing
    tier_size: int
    decided: bool         #: a one-member tier, won without consulting the score
    whole_pool: bool      #: the tier is the whole length-partitioned pool


def replay(pool: list[Draft], *, lam: float, min_chars: int, subject_tier: bool,
           has_subject: bool) -> Pick:
    """`snnchat.rerank.select`, from stored fields.

    Line for line what `select` does after annotation: the length partition,
    the REAL `echo_tier`, `decided`, the score, and `max` (first maximum wins,
    in pool order, as there).
    """
    long_enough = [d for d in pool if d.n_chars >= min_chars]
    length_pool = long_enough or list(pool)
    tier = echo_tier(length_pool, subject_tier=subject_tier, has_subject=has_subject)
    decided = len(tier) == 1
    whole = len(tier) == len(length_pool)
    if lam != 0.0 and not decided and not all(d.null_measured for d in tier):
        return Pick(None, len(tier), decided, whole)
    winner = max(tier, key=lambda d: (d.logp_cond - lam * d.logp_null) / max(d.n_scored, 1))
    index = next(i for i, d in enumerate(pool) if d is winner)
    return Pick(index, len(tier), decided, whole)


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------


def n_scored_rule(n_chars: int, max_new: int) -> int:
    """`len(scored)` recovered from the draft length. See the module docstring."""
    return n_chars + (1 if n_chars < max_new else 0)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def drafts_v14(row: dict) -> list[Draft]:
    return [Draft(text=c["text"], n_chars=c["n_chars"], n_scored=c["n_scored"],
                  logp_cond=c["logp_cond"], logp_null=c["logp_null"],
                  echo_weight=c["echo_weight"], subject_hit=bool(c["subject_hit"]))
            for c in row["pool"]]


def drafts_v12(row: dict, subject: str | None) -> list[Draft]:
    return [Draft(text=c["text"], n_chars=c["n_chars"],
                  n_scored=n_scored_rule(c["n_chars"], V12_MAX_NEW),
                  logp_cond=c["logp_cond"], logp_null=c["logp_null"],
                  echo_weight=c["echo_weight"],
                  subject_hit=bool(subject) and echoes(subject, c["text"].lower()),
                  null_measured=c["logp_null"] != 0.0)
            for c in row["pool"]]


def stored_field_mismatches(prompt: str, subject: str | None, pool: list[Draft]) -> dict:
    """Do the stored fields describe the stored text? Counted, per field.

    Recomputes what `select` computed, from the string it computed it on. Exact
    float equality on `echo_weight`: it is a sum of the same table entries in
    the same (sorted) order, so anything but equality means the word table or
    the matcher changed under the pool.
    """
    words = prompt_content_words(prompt)
    out = {"echo_weight": 0, "subject_hit": 0}
    for d in pool:
        low = d.text.lower()
        if echo_weight(d.text, words) != d.echo_weight:
            out["echo_weight"] += 1
        if (bool(subject) and echoes(subject, low)) != d.subject_hit:
            out["subject_hit"] += 1
    return out


def check_v12_row(row: dict, pool: list[Draft], lam: float) -> dict:
    """Make the file reproduce the two picks it recorded, and test `n_scored`.

    `echo_holdout.py` stores, for its shipped pick and its score-only pick, the
    first 300 characters and `logp_cond / len(scored)`. Replaying both and
    matching text AND that quotient exactly checks three things at once: the
    replay, `V12_MIN_CHARS`, and `n_scored_rule` on the branch the pick is on.

    The score-only pick is `max(score)` over the length pool with whatever
    `logp_null` the turn left behind -- 0.0 on a decided draw, where the null
    pass was skipped, so that on those draws it is a lambda = 0 pick whatever
    the file's lambda says.
    """
    shipped = replay(pool, lam=lam, min_chars=V12_MIN_CHARS, subject_tier=False,
                     has_subject=False)
    length_pool = [d for d in pool if d.n_chars >= V12_MIN_CHARS] or pool
    base = max(length_pool,
               key=lambda d: (d.logp_cond - lam * d.logp_null) / max(d.n_scored, 1))
    out = {"n_scored_checked_closed": 0, "n_scored_checked_open": 0,
           "n_scored_mismatch": 0, "shipped_pick_mismatch": 0, "base_pick_mismatch": 0,
           "shipped_not_replayable": 0}
    if shipped.index is None:
        # An undecided shipped tier with no null term: the file does not
        # describe a turn `select` could have produced. Counted, never skipped
        # silently.
        out["shipped_not_replayable"] += 1
        return out
    for label, d in (("echo", pool[shipped.index]), ("base", base)):
        same_text = d.text[:300] == row[f"{label}_text"]
        same_logp = d.logp_per_char == row[f"{label}_logp"]
        if not (same_text and same_logp):
            out["shipped_pick_mismatch" if label == "echo" else "base_pick_mismatch"] += 1
        if same_text:
            # The recorded quotient determines len(scored) on its own.
            implied = round(d.logp_cond / row[f"{label}_logp"]) if row[f"{label}_logp"] else 0
            branch = "closed" if d.n_chars < V12_MAX_NEW else "open"
            out[f"n_scored_checked_{branch}"] += 1
            if implied != d.n_scored:
                out["n_scored_mismatch"] += 1
    return out


@dataclass
class Draw:
    """One (prompt, sampler seed) after replay; the pool itself is dropped."""

    source: str
    ckpt: str
    probe_set: str
    kind: str
    prompt: str
    seed: int
    expect: tuple[str, ...]
    subject: str | None
    pool_n: int
    null_whole_pool: bool
    names_subject: int                     #: drafts in the length pool that do
    picks: dict[str, dict[str, Pick]]      #: [lam label][selector] -> Pick
    drafts: dict[str, dict[str, Draft | None]]


def _lam_label(lam: float) -> str:
    return f"lam{lam:g}"


def load_file(path: Path, *, exploratory: bool) -> tuple[dict, list[Draw]]:
    """Read one pool file, replay both rules on every draw, keep the winners."""
    blob = read_pool(path)
    is_v14 = blob.get("format") == FORMAT
    if not is_v14 and "format" in blob:
        raise SystemExit(f"{path.name}: unknown pool format {blob['format']!r}")
    if not is_v14 and not exploratory:
        raise SystemExit(
            f"{path.name} is not a v14_pools.py file. Committed echo_holdout pools "
            "carry no recorded subject-tier winner to assert against and no null "
            "term on decided draws; read them with --exploratory.")
    lam = float(blob["lam"])
    min_chars = int(blob["min_chars"]) if is_v14 else V12_MIN_CHARS
    rows = blob["draws"] if is_v14 else blob["rows"]
    lams = [lam] + ([0.0] if exploratory and lam != 0.0 else [])

    info = {
        "path": path.name, "sha256": _sha256(path),
        "format": blob.get("format", "echo_holdout"),
        "ckpt": blob["ckpt"], "probe_set": blob["probe_set"], "lam": lam,
        "n": blob["n"], "n_draws": len(rows),
        "sampler_seeds": sorted({r["seed"] for r in rows}),
        "min_chars": min_chars,
    }
    if is_v14:
        info["timing"] = blob.get("timing")
        info["ckpt_sha256"] = blob.get("ckpt_sha256")
    checks: Counter = Counter()
    mismatches: list[str] = []
    draws: list[Draw] = []

    for row in rows:
        prompt = row["prompt"]
        subject = prime_topic(prompt)
        if is_v14:
            pool = drafts_v14(row)
            if subject != row["subject"]:
                mismatches.append(f"{prompt!r} seed {row['seed']}: stored subject "
                                  f"{row['subject']!r}, prime_topic says {subject!r}")
        else:
            pool = drafts_v12(row, subject)
            if max(d.n_chars for d in pool) > V12_MAX_NEW:
                raise SystemExit(f"{path.name}: a draft is longer than "
                                 f"V12_MAX_NEW={V12_MAX_NEW}; the n_scored rule is void")
            checks.update(check_v12_row(row, pool, lam))
        stored = stored_field_mismatches(prompt, subject, pool)
        if not is_v14:
            # `subject_hit` was just derived from the text; only the stored
            # weight is a check on these files.
            stored.pop("subject_hit")
        checks.update({f"stored_{k}_mismatch": v for k, v in stored.items()})
        checks["candidates"] += len(pool)

        picks: dict[str, dict[str, Pick]] = {}
        drafts: dict[str, dict[str, Draft | None]] = {}
        for value in lams:
            label = _lam_label(value)
            picks[label] = {
                "shipped": replay(pool, lam=value, min_chars=min_chars,
                                  subject_tier=False, has_subject=False),
                "subject": replay(pool, lam=value, min_chars=min_chars,
                                  subject_tier=True, has_subject=bool(subject)),
            }
            drafts[label] = {k: (pool[p.index] if p.index is not None else None)
                             for k, p in picks[label].items()}

        if is_v14:
            got = picks[_lam_label(lam)]
            for name in ("shipped", "subject"):
                if got[name].index != row[f"{name}_index"]:
                    mismatches.append(
                        f"{prompt!r} seed {row['seed']}: {name} selector recorded "
                        f"index {row[f'{name}_index']}, replay chose {got[name].index}")

        length_pool = [d for d in pool if d.n_chars >= min_chars] or pool
        draws.append(Draw(
            source=path.name, ckpt=blob["ckpt"], probe_set=blob["probe_set"],
            kind=row.get("kind", "story"), prompt=prompt, seed=row["seed"],
            expect=tuple(row["expect"]), subject=subject, pool_n=len(pool),
            null_whole_pool=all(d.null_measured for d in pool),
            names_subject=sum(1 for d in length_pool if d.subject_hit),
            picks=picks, drafts=drafts,
        ))

    if is_v14:
        bad = checks["stored_echo_weight_mismatch"] + checks["stored_subject_hit_mismatch"]
        if bad:
            mismatches.append(f"{bad} candidates whose stored echo_weight / subject_hit "
                              "is not what their stored text gives")
        if mismatches:
            shown = "\n  ".join(mismatches[:10])
            raise ReplayMismatch(
                f"{path.name}: {len(mismatches)} disagreement(s) between the stored "
                f"pool and this replay; nothing is reported.\n  {shown}")
    info["checks"] = dict(checks)
    return info, draws


# ---------------------------------------------------------------------------
# statistics
# ---------------------------------------------------------------------------


def rate(k: int, n: int) -> dict:
    """A proportion with its numerator, denominator and Wilson 95 % interval."""
    lo, hi = wilson_interval(k, n) if n else (0.0, 0.0)
    return {"k": k, "n": n, "rate": (k / n) if n else None, "ci": [lo, hi]}


def distinct_2(texts) -> dict:
    """Distinct word bigrams over total word bigrams, across ALL of `texts`.

    Bigrams do not cross a reply boundary. The ratio falls as the amount of text
    grows, so it is only comparable between two selections of the SAME draws --
    which is the only way it is used here.
    """
    seen: set[tuple[str, str]] = set()
    total = 0
    for t in texts:
        words = _WORDS.findall(t.lower())
        for pair in zip(words, words[1:]):
            seen.add(pair)
            total += 1
    return {"distinct": len(seen), "total": total,
            "value": (len(seen) / total) if total else None}


def top_opening(texts, chars: int = OPENING_CHARS) -> dict:
    """The most common first `chars` characters and the share of replies with it.

    Ties go to the opening met first, so the answer does not depend on hashing.
    """
    texts = list(texts)
    if not texts:
        return {"opening": None, "k": 0, "n": 0, "share": None}
    counts = Counter(t[:chars] for t in texts)
    opening, k = max(counts.items(), key=lambda kv: kv[1])
    return {"opening": opening, "k": k, "n": len(texts), "share": k / len(texts)}


def paired_delta(a: list[float], b: list[float], clusters: list[str]) -> dict:
    """Mean of `b - a` with two standard errors.

    `se_paired` treats draws as independent. They are not: six sampler seeds
    share a prompt. `se_by_prompt` is the standard error of the mean of the
    per-prompt mean differences, and it is the one a verdict rides on whenever
    it is the larger (`compare_holdout.py`: "if the two disagree, believe the
    cluster-level one").
    """
    n = len(a)
    if n == 0:
        return {"n": 0, "mean_a": None, "mean_b": None, "delta": None,
                "se_paired": None, "se_by_prompt": None, "n_prompts": 0}
    diffs = [y - x for x, y in zip(a, b)]
    by: dict[str, list[float]] = defaultdict(list)
    for c, d in zip(clusters, diffs):
        by[c].append(d)
    means = [sum(v) / len(v) for v in by.values()]
    return {
        "n": n, "mean_a": sum(a) / n, "mean_b": sum(b) / n, "delta": sum(diffs) / n,
        "se_paired": statistics.stdev(diffs) / math.sqrt(n) if n > 1 else None,
        "se_by_prompt": (statistics.stdev(means) / math.sqrt(len(means))
                         if len(means) > 1 else None),
        "n_prompts": len(means),
    }


_BUCKETS = ((1, 1, "1"), (2, 4, "2-4"), (5, 16, "5-16"), (17, 64, "17-64"),
            (65, 10 ** 9, "65+"))


def tier_sizes(picks: list[Pick]) -> dict:
    out = {label: 0 for _, _, label in _BUCKETS}
    for p in picks:
        for lo, hi, label in _BUCKETS:
            if lo <= p.tier_size <= hi:
                out[label] += 1
    return {"n": len(picks), "buckets": out,
            "one_member": rate(out["1"], len(picks)),
            "whole_pool": rate(sum(1 for p in picks if p.whole_pool), len(picks)),
            "median": statistics.median(p.tier_size for p in picks) if picks else None}


# ---------------------------------------------------------------------------
# verdicts
# ---------------------------------------------------------------------------


def p1_verdict(shipped_only: int, subject_only: int, n: int) -> dict:
    if n == 0:
        return {"verdict": "unresolved", "why": "no draws"}
    ok = shipped_only <= subject_only
    return {"verdict": "pass" if ok else "fail",
            "why": f"shipped_only {shipped_only} {'<=' if ok else '>'} "
                   f"subject_only {subject_only}"}


def p2_verdict(pooled: dict, per_list: dict[str, dict]) -> dict:
    """Fail on the point estimate; pass only if the interval clears the bar too."""
    missing = [name for name in P2_LISTS if not per_list.get(name, {}).get("n")]
    if not pooled["n"]:
        return {"verdict": "unresolved", "why": "no draws"}
    non_positive = [name for name in P2_LISTS
                    if name not in missing and not per_list[name]["delta"] > 0.0]
    if pooled["delta"] < P2_POOLED_BAR:
        return {"verdict": "fail",
                "why": f"pooled delta {pooled['delta']:+.4f} < {P2_POOLED_BAR:+.2f}"}
    if non_positive:
        return {"verdict": "fail", "why": f"delta not > 0 on {', '.join(non_positive)}"}
    if missing:
        return {"verdict": "unresolved", "why": f"no draws from {', '.join(missing)}"}
    ses = [s for s in (pooled["se_paired"], pooled["se_by_prompt"]) if s is not None]
    if not ses:
        return {"verdict": "unresolved", "why": "one draw; no standard error"}
    low = pooled["delta"] - 1.959963984540054 * max(ses)
    if low < P2_POOLED_BAR:
        return {"verdict": "unresolved",
                "why": f"pooled delta {pooled['delta']:+.4f} meets the bar but its 95% "
                       f"lower bound {low:+.4f} does not"}
    return {"verdict": "pass",
            "why": f"pooled delta {pooled['delta']:+.4f}, 95% lower bound {low:+.4f}, "
                   f">= {P2_POOLED_BAR:+.2f}; positive on every list"}


def p3_verdict(fixed: int, broken: int, p: float, n_qualifying: int) -> dict:
    if n_qualifying == 0:
        return {"verdict": "unresolved", "why": "no draw where both replies qualify"}
    if fixed <= broken:
        return {"verdict": "fail", "why": f"fixed {fixed} <= broken {broken}"}
    if p >= P3_ALPHA:
        return {"verdict": "unresolved",
                "why": f"fixed {fixed} > broken {broken} but McNemar p = {p:.4g} "
                       f">= {P3_ALPHA}"}
    return {"verdict": "pass", "why": f"fixed {fixed} > broken {broken}, p = {p:.4g}"}


# ---------------------------------------------------------------------------
# the comparison
# ---------------------------------------------------------------------------


def _flagged(text: str, context: str = "") -> bool:
    return bool(unintroduced_entities(text, WINDOW, context=context))


def compare(draws: list[Draw], label: str, *, breakdown: bool = True) -> dict:
    """Shipped against subject tier, over `draws`, at the replay named `label`.

    Every draw passed in must be replayable under both rules at `label`.
    """
    a = [d.drafts[label]["shipped"] for d in draws]
    b = [d.drafts[label]["subject"] for d in draws]
    if any(x is None for x in (*a, *b)):
        raise ValueError("compare() was handed a draw that is not replayable")
    n = len(draws)

    # P1
    ha = [hit(x.text, d.expect) for x, d in zip(a, draws)]
    hb = [hit(x.text, d.expect) for x, d in zip(b, draws)]
    shipped_only = sum(1 for x, y in zip(ha, hb) if x > y)
    subject_only = sum(1 for x, y in zip(ha, hb) if y > x)

    # P2
    la, lb = [x.logp_per_char for x in a], [x.logp_per_char for x in b]
    pooled = paired_delta(la, lb, [d.prompt for d in draws])
    per_list = {}
    for name in sorted({d.probe_set for d in draws}):
        idx = [i for i, d in enumerate(draws) if d.probe_set == name]
        per_list[name] = paired_delta([la[i] for i in idx], [lb[i] for i in idx],
                                      [draws[i].prompt for i in idx])

    # P3, over the draws where BOTH selected replies reach the window.
    qual = [i for i in range(n) if len(a[i].text) >= WINDOW and len(b[i].text) >= WINDOW]
    fa = [_flagged(a[i].text) for i in qual]
    fb = [_flagged(b[i].text) for i in qual]
    fixed = sum(1 for x, y in zip(fa, fb) if x and not y)
    broken = sum(1 for x, y in zip(fa, fb) if y and not x)
    p3_p = mcnemar(broken, fixed)
    ca = [_flagged(a[i].text, draws[i].prompt) for i in qual]
    cb = [_flagged(b[i].text, draws[i].prompt) for i in qual]
    c_fixed = sum(1 for x, y in zip(ca, cb) if x and not y)
    c_broken = sum(1 for x, y in zip(ca, cb) if y and not x)

    out = {
        "n_draws": n,
        "selection_changed": rate(sum(1 for x, y in zip(a, b) if x is not y), n),
        "no_draft_names_subject": rate(sum(1 for d in draws if d.names_subject == 0), n),
        "P1_topic": {
            "rule": P1_RULE,
            "shipped": rate(int(sum(ha)), n), "subject": rate(int(sum(hb)), n),
            "shipped_only": shipped_only, "subject_only": subject_only,
            "mcnemar_p": mcnemar(shipped_only, subject_only),
            **p1_verdict(shipped_only, subject_only, n),
        },
        "P2_logp_per_char": {
            "rule": f"pooled delta >= {P2_POOLED_BAR:+.2f} and delta > 0 on each of "
                    f"{', '.join(P2_LISTS)}",
            "shipped_mean": pooled["mean_a"], "subject_mean": pooled["mean_b"],
            "pooled": pooled, "per_list": per_list,
            **p2_verdict(pooled, per_list),
        },
        "P3_uer": {
            "rule": f"fixed > broken, exact McNemar p < {P3_ALPHA}",
            "window": WINDOW, "both_qualify": rate(len(qual), n),
            "shipped": rate(sum(fa), len(qual)), "subject": rate(sum(fb), len(qual)),
            "fixed": fixed, "broken": broken, "mcnemar_p": p3_p,
            **p3_verdict(fixed, broken, p3_p, len(qual)),
            "prompt_as_context": {
                "note": "not the bar; the prompt's words count as introduced",
                "shipped": rate(sum(ca), len(qual)), "subject": rate(sum(cb), len(qual)),
                "fixed": c_fixed, "broken": c_broken,
                "mcnemar_p": mcnemar(c_broken, c_fixed),
            },
        },
        "below_floor": {
            "floor": FLOOR,
            "shipped": rate(sum(1 for v in la if v < FLOOR), n),
            "subject": rate(sum(1 for v in lb if v < FLOOR), n),
        },
        "GUARD2_sameness": {
            "distinct_2": {"shipped": distinct_2(x.text for x in a),
                           "subject": distinct_2(x.text for x in b)},
            "top_opening": {"chars": OPENING_CHARS,
                            "shipped": top_opening(x.text for x in a),
                            "subject": top_opening(x.text for x in b)},
        },
        "tier_size": {
            "shipped": tier_sizes([d.picks[label]["shipped"] for d in draws]),
            "subject": tier_sizes([d.picks[label]["subject"] for d in draws]),
        },
    }
    if breakdown:
        out["by_list"] = {
            name: compare([d for d in draws if d.probe_set == name], label, breakdown=False)
            for name in sorted({d.probe_set for d in draws})}
        ckpts = sorted({d.ckpt for d in draws})
        if len(ckpts) > 1:
            out["by_ckpt"] = {
                name: compare([d for d in draws if d.ckpt == name], label, breakdown=False)
                for name in ckpts}
            deltas = [out["by_ckpt"][c]["P2_logp_per_char"]["pooled"]["delta"] for c in ckpts]
            out["P2_delta_across_ckpts"] = {
                "per_ckpt": dict(zip(ckpts, deltas)), "mean": statistics.mean(deltas),
                "sd_seed": statistics.stdev(deltas), "n_ckpts": len(ckpts)}
    return out


def guard1(draws: list[Draw], label_of) -> dict:
    """Non-story turns must take the byte-identical old path. An identity.

    A turn with no subject is one `prime_topic` returns None for, and `select`
    then uses the weighted tier whatever `subject_tier` says. So both rules must
    return the SAME candidate -- same index, hence the same text -- on every
    such draw. One exception is a failure; there is no rate to read.
    """
    dodge = [d for d in draws if d.probe_set == "dodge"]
    with_subject = [d for d in dodge if d.subject is not None]
    plain = [d for d in draws if d.subject is None]
    if not plain:
        return {"verdict": "unresolved", "why": "no draw without a subject was given "
                "(pass a `v14_pools.py --set dodge` pool)", "n": 0}
    differ = []
    for d in plain:
        got = d.picks[label_of(d)]
        if (got["shipped"].index != got["subject"].index
                or d.drafts[label_of(d)]["shipped"].text
                != d.drafts[label_of(d)]["subject"].text):
            differ.append({"source": d.source, "prompt": d.prompt, "seed": d.seed})
    ok = not differ and not with_subject
    why = f"identical selection on {len(plain) - len(differ)}/{len(plain)} subject-less draws"
    if with_subject:
        why += f"; {len(with_subject)} dodge draws HAVE a subject, so the set is not non-story"
    return {"verdict": "pass" if ok else "fail", "why": why, "n": len(plain),
            "identical": len(plain) - len(differ), "differing": differ[:20],
            "dodge_draws_with_a_subject": len(with_subject)}


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------


def _fmt_rate(r: dict) -> str:
    if r["rate"] is None:
        return "n/a (0/0)"
    return f"{r['rate']:.4f} ({r['k']}/{r['n']}), 95% CI [{r['ci'][0]:.4f}, {r['ci'][1]:.4f}]"


def _fmt_se(se: float | None) -> str:
    """A standard error over one draw, or one prompt, does not exist."""
    return "n/a" if se is None else f"{se:.4f}"


def print_comparison(title: str, c: dict) -> None:
    p1, p2, p3 = c["P1_topic"], c["P2_logp_per_char"], c["P3_uer"]
    g2, fl, ts = c["GUARD2_sameness"], c["below_floor"], c["tier_size"]
    print(f"\n=== {title}: {c['n_draws']} draws ===")
    print(f"  selection changed       {_fmt_rate(c['selection_changed'])}")
    print(f"  no draft names subject  {_fmt_rate(c['no_draft_names_subject'])}")
    print(f"  P1 topic   shipped {_fmt_rate(p1['shipped'])}")
    print(f"             subject {_fmt_rate(p1['subject'])}")
    print(f"             shipped-only {p1['shipped_only']}, subject-only "
          f"{p1['subject_only']}, McNemar p = {p1['mcnemar_p']:.4g}"
          f"   -> {p1['verdict']} ({p1['why']})")
    pooled = p2["pooled"]
    if pooled["n"]:
        print(f"  P2 logP/ch shipped {pooled['mean_a']:+.4f}  subject {pooled['mean_b']:+.4f}"
              f"  delta {pooled['delta']:+.4f}  (n={pooled['n']}, SE paired "
              f"{_fmt_se(pooled['se_paired'])}, SE by prompt {_fmt_se(pooled['se_by_prompt'])} "
              f"over {pooled['n_prompts']} prompts)")
        for name, d in p2["per_list"].items():
            print(f"             {name:<8} delta {d['delta']:+.4f}  (n={d['n']})")
    print(f"             -> {p2['verdict']} ({p2['why']})")
    print(f"  P3 UER@{p3['window']} both qualify {_fmt_rate(p3['both_qualify'])}")
    print(f"             shipped {_fmt_rate(p3['shipped'])}")
    print(f"             subject {_fmt_rate(p3['subject'])}")
    print(f"             fixed {p3['fixed']}, broken {p3['broken']}, McNemar p = "
          f"{p3['mcnemar_p']:.4g}   -> {p3['verdict']} ({p3['why']})")
    pc = p3["prompt_as_context"]
    print(f"             (prompt as context: {pc['shipped']['k']} -> {pc['subject']['k']} of "
          f"{pc['shipped']['n']}, fixed {pc['fixed']}, broken {pc['broken']})")
    print(f"  below {fl['floor']} shipped {_fmt_rate(fl['shipped'])}")
    print(f"             subject {_fmt_rate(fl['subject'])}")
    for key in ("shipped", "subject"):
        d2, op = g2["distinct_2"][key], g2["top_opening"][key]
        print(f"  GUARD 2 {key:<8} distinct-2 {d2['value']:.4f} ({d2['distinct']}/{d2['total']})"
              f"   top {g2['top_opening']['chars']}-char opening {op['share']:.4f} "
              f"({op['k']}/{op['n']}) {op['opening']!r}")
    for key in ("shipped", "subject"):
        t = ts[key]
        print(f"  tier size {key:<8} one member {_fmt_rate(t['one_member'])}; whole pool "
              f"{t['whole_pool']['k']}/{t['n']}; median {t['median']}; {t['buckets']}")


def draw_rows(draws: list[Draw]) -> list[dict]:
    """One compact line per draw, so any count above can be re-derived by hand."""
    rows = []
    for d in draws:
        row = {"source": d.source, "prompt": d.prompt, "seed": d.seed,
               "subject": d.subject, "pool_n": d.pool_n,
               "null_whole_pool": d.null_whole_pool, "names_subject": d.names_subject}
        for label, pair in d.picks.items():
            for name, p in pair.items():
                row[f"{label}_{name}"] = [p.index, p.tier_size]
        rows.append(row)
    return rows


def expand(names: list[str]) -> list[Path]:
    """ROOT-relative unless absolute, with `*` expanded here and sorted.

    PowerShell hands a wildcard through unexpanded, and an order that depended
    on the shell would reorder `inputs` and `rows` in the artifact.
    """
    out: list[Path] = []
    for name in names:
        p = Path(name)
        p = p if p.is_absolute() else ROOT / p
        if "*" in p.name:
            found = sorted(p.parent.glob(p.name))
            if not found:
                raise SystemExit(f"no pool file matches {name}")
            out.extend(found)
        else:
            out.append(p)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pools", nargs="*",
                    help="pool files, ROOT-relative or absolute; none with "
                         "--exploratory means the twelve committed v12 pools")
    ap.add_argument("--exploratory", action="store_true",
                    help="accept committed echo_holdout-format pools; no verdict "
                         "from this mode is a test of anything")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    names = list(args.pools)
    if not names:
        if not args.exploratory:
            ap.error("give at least one v14_pools.py file, or --exploratory")
        names = [str(QUALITY / f) for f in EXPLORATORY_FILES]
    paths = expand(names)
    out_path = Path(args.out) if args.out else QUALITY / (
        "v14_exploratory.json" if args.exploratory else "v14_confirmatory.json")
    if not out_path.is_absolute():
        out_path = ROOT / out_path

    inputs, draws = [], []
    for path in paths:
        info, got = load_file(path, exploratory=args.exploratory)
        inputs.append(info)
        draws.extend(got)
        print(f"read {path.name}: {len(got)} draws", flush=True)

    lam_of = {i["path"]: _lam_label(i["lam"]) for i in inputs}
    label_of = lambda d: lam_of[d.source]                      # noqa: E731
    labels = sorted(set(lam_of.values()))
    if len(labels) != 1:
        raise SystemExit(f"the pools were drawn at different lambdas: {labels}")
    label = labels[0]
    story = [d for d in draws if d.kind == "story"]

    checks: Counter = Counter()
    for i in inputs:
        checks.update(i["checks"])

    replayable = [d for d in story if d.drafts[label]["subject"] is not None]
    subsets: dict[str, dict] = {}
    if args.exploratory:
        measured = [d for d in story if d.null_whole_pool]
        subsets[f"{label}_null_measured"] = {
            "definition": "draws whose WHOLE pool carries a measured logp_null, i.e. "
                          "where the shipped tier had more than one member",
            "of": len(story), **compare(measured, label)}
        subsets[f"{label}_replayable"] = {
            "definition": "the subset above plus draws where the subject tier is a "
                          "single draft and needs no score",
            "of": len(story), **compare(replayable, label)}
        if label != "lam0":
            subsets["lam0_all"] = {
                "definition": "pure likelihood inside each tier; needs no null term, "
                              "so every draw is replayable",
                "of": len(story), **compare(story, "lam0")}
    else:
        if len(replayable) != len(story):
            raise ReplayMismatch(f"{len(story) - len(replayable)} confirmatory draws "
                                 "lack a null term the subject tier needs")
        subsets[f"{label}_all"] = {"definition": "every story draw given",
                                   "of": len(story), **compare(story, label)}

    blob = {
        "mode": "exploratory" if args.exploratory else "confirmatory",
        "note": ("EXPLORATORY: these pools were read before the subject rule existed. "
                 "The verdict fields show how each bar WOULD fire and test nothing."
                 if args.exploratory else
                 "Bars fixed by docs/chat/PREDICTION_v14.md before these pools were drawn."),
        "command": "CUDA_VISIBLE_DEVICES=-1 python scripts/chat/score_v14.py "
                   + " ".join(sys.argv[1:] if argv is None else argv),
        "bars": {"P1": P1_RULE, "P2_pooled_bar": P2_POOLED_BAR, "P2_lists": list(P2_LISTS),
                 "P3_alpha": P3_ALPHA, "floor": FLOOR, "window": WINDOW,
                 "opening_chars": OPENING_CHARS},
        "inputs": inputs,
        "checks": dict(checks),
        "n_story_draws": len(story),
        "shipped_tier_all_draws": tier_sizes([d.picks[label]["shipped"] for d in story]),
        "subject_tier_all_draws": tier_sizes([d.picks[label]["subject"] for d in story]),
        "no_draft_names_subject_all_draws": rate(
            sum(1 for d in story if d.names_subject == 0), len(story)),
        "subsets": subsets,
        "GUARD1_identity": guard1(draws, label_of),
        "latency": {i["path"]: i["timing"] for i in inputs if i.get("timing")},
        "rows": draw_rows(draws),
    }
    if not args.exploratory:
        main_cmp = subsets[f"{label}_all"]
        verdicts = [main_cmp["P1_topic"]["verdict"], main_cmp["P2_logp_per_char"]["verdict"],
                    main_cmp["P3_uer"]["verdict"], blob["GUARD1_identity"]["verdict"]]
        blob["overall"] = ("fail" if "fail" in verdicts else
                           "unresolved" if "unresolved" in verdicts else "pass")

    print(f"\nmode: {blob['mode']}; {len(story)} story draws from {len(inputs)} files")
    print(f"checks: {dict(checks)}")
    t = blob["shipped_tier_all_draws"]
    print(f"shipped tier has one member on {_fmt_rate(t['one_member'])} of all story draws")
    print(f"no draft names the subject on {_fmt_rate(blob['no_draft_names_subject_all_draws'])}")
    for name, c in subsets.items():
        print_comparison(f"{name} (of {c['of']})", c)
        if "P2_delta_across_ckpts" in c:
            s = c["P2_delta_across_ckpts"]
            print(f"  P2 delta across {s['n_ckpts']} checkpoints: mean {s['mean']:+.4f}, "
                  f"SD_seed {s['sd_seed']:.4f}")
    g1 = blob["GUARD1_identity"]
    print(f"\nGUARD 1: {g1['verdict']} ({g1['why']})")
    for name, tm in blob["latency"].items():
        print(f"latency {name}: shipped {tm['mean_select_seconds_shipped']:.4f} s, subject "
              f"{tm['mean_select_seconds_subject']:.4f} s per turn ({tm['n_draws']} draws)")
    if "overall" in blob:
        print(f"\nOVERALL: {blob['overall']}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(blob, indent=1, ensure_ascii=False) + "\n",
                        encoding="utf-8")
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
