"""Unintroduced-entity rate (UER@200) on everything already committed. Zero GPU.

WHAT THIS READS, AND WHY IT NEEDS NO MODEL
------------------------------------------
`scripts/chat/echo_holdout.py` stores every candidate it ever drew: each
`v12_{heldout,fresh,wide}_*.json` holds 256 full-text pool replies per
(prompt, sampler seed) row, for eight checkpoints. So the question "how often
does this model's story refer to a thing it never introduced" can be asked of
the committed draws, with no checkpoint loaded and nothing sampled -- and asked
of the training corpus itself, which is the only reading that gives a model's
rate a meaning. `snnchat.coherence` documents the instrument and what it cannot
see; read that first. It is NOT a coherence measure and has never been validated
against a human rating.

THREE THINGS ARE REPORTED
-------------------------
(i)   THE CORPUS REFERENCE. Every `<|bot|>` story turn decoded from
      `stories_topic.val.bin`, WITH and WITHOUT the head stoplist, plus the
      prototype's rule transcribed literally (`_prototype_flags`), so the
      prototype's pre-stoplist reading can be checked against this code rather
      than remembered. Then the two validation mutations on the same stories:
      replace-alternate-sentences, which the instrument detects, and the plain
      interleave, to which it is blind. Both are reported so the blindness is a
      number in the artifact and not a sentence in a docstring.
(ii)  EVERY COMMITTED POOL FILE. The whole POOL -- what the model writes -- and
      the SELECTED reply (`echo_text`, what the shipped selector showed the
      user), plus `base_text` (what score-only selection would have shown).
      `echo_text` and `base_text` are stored TRUNCATED to 300 characters; pool
      texts are full. At a window <= 300 that changes nothing -- a truncated
      reply is at least 200 characters long exactly when the original was, and
      the instrument reads nothing past the window -- and at a larger window the
      selected columns are refused rather than silently scored over less text.
(iii) SD_seed. The pool rate's mean and sample SD across the four training seeds
      of each recipe. The Wilson interval on a pool treats 256 draws from one
      prompt as independent, which they are not, so it is too narrow to judge a
      TRAINING change by. A future round's pool rate is judged against the
      incumbent's spread across seeds, the same yardstick `score_v13.py` uses
      for bpc.

Every pool reading is given twice: on the reply alone, and with the row's prompt
passed as `context` (so "tell me a story about a pig" -> "The pig was big." is
not flagged). `snnchat.coherence` explains why both are legitimate.

`--write-fixture` regenerates `tests/fixtures/chat_stories_uer.json` from the
corpus, so the committed fixture has a generator and is not a pasted blob.

Not part of the research protocol; no number here is a reported figure.

    python scripts/chat/coherence_score.py
    python scripts/chat/coherence_score.py --data-dir "<main tree>/data/chat"
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import re
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

from snnchat import coherence  # noqa: E402
from snnchat.tokenizer import BOS, BOT, EOT, USER, ChatTokenizer  # noqa: E402

#: The corpus reference. The VAL split: no checkpoint was trained on it, and it
#: is small enough to score whole, so the reference is a census and not a sample.
CORPUS = "stories_topic.val.bin"

#: `echo_holdout.py`'s three probe lists. `v12_dodge_*` is a different format
#: (the story-dodge battery) and is not read.
LISTS = ("heldout", "fresh", "wide")

#: `echo_text` / `base_text` are stored cut to this many characters.
SELECTED_TRUNCATION = 300

#: The mutation partner for story `i` is story `(i + PARTNER) % n`. Fixed, and
#: the prototype's value, so the corpus readings are comparable with its.
PARTNER = 7

#: The committed test fixture: how many conversations, and where each is cut.
FIXTURE_N = 120
FIXTURE_CUT = 500


# ---------------------------------------------------------------------------
# the corpus
# ---------------------------------------------------------------------------

def story_turns(path: Path) -> list[tuple[str, str]]:
    """`(user prompt, bot story)` for every conversation in a packed corpus.

    A turn boundary is an integer comparison on the four role ids
    (`snnchat.tokenizer`), never a string search. About one conversation in
    seven in this file is a BARE NARRATIVE -- BOS, text, EOT, no role markers
    (`snnchat.build_corpus`) -- and those are skipped: they have no prompt, and
    "the corpus reference" here means the turns the model is trained to produce
    as replies.
    """
    raw = np.fromfile(path, dtype=np.uint8)
    tok = ChatTokenizer()
    out: list[tuple[str, str]] = []
    user: str | None = None
    mode: int | None = None
    start = 0
    for i in np.flatnonzero(raw < tok.n_special):
        marker = int(raw[i])
        if marker == EOT:
            text = tok.decode_visible(raw[start:i])
            if mode == USER:
                user = text
            elif mode == BOT and user is not None:
                out.append((user, text))
            mode = None
        else:
            mode, start = marker, int(i) + 1
            if marker == BOS:
                user = None
    return out


#: The prototype's pattern, transcribed and not improved: "a|an|the", no
#: stoplist, the head looked up exactly as written (so a plural head never
#: matches its own singular introduction), and the pattern run over the CUT
#: string. Kept only so its reading can be reproduced beside the instrument's.
_PROTOTYPE_VERBS = (
    "was|were|is|had|has|saw|said|went|wanted|loved|liked|felt|looked|came|ran|did|could|"
    "would|got|took|made|found|asked|smiled|laughed|cried|played|jumped|flew|lived|thought|"
    "knew|tried|started|decided"
)
_PROTOTYPE = re.compile(
    r"\b(a|an|the)\s+((?:[a-z]+\s+){0,2}?)([a-z]+)\s+(?:" + _PROTOTYPE_VERBS + r")\b", re.I
)


def _prototype_flags(text: str, window: int) -> bool:
    cut = text[:window]
    low = cut.lower()
    for m in _PROTOTYPE.finditer(cut):
        head = m.group(3).lower()
        if m.group(1).lower() == "the" and re.search(
            r"\b" + re.escape(head) + r"(s|es)?\b", low[: m.start()]
        ) is None:
            return True
    return False


def _prototype_reading(texts: list[str], window: int) -> dict:
    qualifying = [t for t in texts if len(t) >= window]
    k = sum(_prototype_flags(t, window) for t in qualifying)
    lo, hi = coherence.wilson_interval(k, len(qualifying))
    return _tidy({
        "rate": k / len(qualifying) if qualifying else None, "k": k, "n": len(qualifying),
        "ci": [lo, hi],
        "qualifying_fraction": len(qualifying) / len(texts) if texts else None,
        "n_total": len(texts), "window": window,
    })


def _tidy(reading: dict) -> dict:
    """Round the floats for the artifact. `k` and `n` stay exact, so every rate
    and interval can be recomputed from the integers beside it."""
    out = dict(reading)
    for key in ("rate", "qualifying_fraction"):
        if out.get(key) is not None:
            out[key] = round(out[key], 6)
    out["ci"] = [round(x, 6) for x in out["ci"]]
    return out


def _uer(texts, window, **kw) -> dict:
    return _tidy(coherence.uer(texts, window, **kw))


def _decomposition(texts, window) -> dict:
    """`coherence.uer_decomposition`, rounded like every other reading here.

    UER is exposure times the rate among exposed replies, and an arm can lower
    the first without touching the second (`snnchat.coherence`, "WHAT IT CANNOT
    SEE" item 8). The corpus's own split is the reference `score_v14.py` prints
    a selector's split against.
    """
    out = coherence.uer_decomposition(texts, window)
    return {k: (_tidy(v) if isinstance(v, dict) else v) for k, v in out.items()}


def _mutants(stories: list[str]) -> tuple[list[str], list[str]]:
    n = len(stories)
    partner = [stories[(i + PARTNER) % n] for i in range(n)]
    return (
        [coherence.replace_alternate_sentences(a, b) for a, b in zip(stories, partner, strict=True)],
        [coherence.interleave_sentences(a, b) for a, b in zip(stories, partner, strict=True)],
    )


def corpus_section(path: Path, window: int) -> dict:
    turns = story_turns(path)
    prompts = [u for u, _ in turns]
    stories = [s for _, s in turns]
    replaced, interleaved = _mutants(stories)
    heads = collections.Counter(
        h for s in stories if len(s) >= window
        for h in coherence.unintroduced_entities(s, window)
    )
    heads_raw = collections.Counter(
        h for s in stories if len(s) >= window
        for h in coherence.unintroduced_entities(s, window, stoplist=False)
    )
    return {
        "file": path.name,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "n_story_turns": len(stories),
        "with_stoplist": _uer(stories, window),
        "decomposition_with_stoplist": _decomposition(stories, window),
        "without_stoplist": _uer(stories, window, stoplist=False),
        "prototype_rule_literal": _prototype_reading(stories, window),
        "with_stoplist_prompt_as_context": _uer(stories, window, contexts=prompts),
        "mutations": {
            "partner_offset": PARTNER,
            "replace_alternate": {
                "with_stoplist": _uer(replaced, window),
                "without_stoplist": _uer(replaced, window, stoplist=False),
                "prototype_rule_literal": _prototype_reading(replaced, window),
            },
            "interleave": {
                "with_stoplist": _uer(interleaved, window),
                "without_stoplist": _uer(interleaved, window, stoplist=False),
                "prototype_rule_literal": _prototype_reading(interleaved, window),
            },
        },
        "flagged_heads_with_stoplist": heads.most_common(25),
        "flagged_heads_without_stoplist": heads_raw.most_common(25),
    }


# ---------------------------------------------------------------------------
# the committed pools
# ---------------------------------------------------------------------------

_NAME = re.compile(r"^v12_(heldout|fresh|wide)_(.+)\.json$")
_SEED = re.compile(r"-s\d+$")


def pool_files(quality_dir: Path) -> list[tuple[str, str, Path]]:
    """`(probe list, run name, path)` for every echo_holdout-format file."""
    out = []
    for path in sorted(quality_dir.glob("v12_*.json")):
        m = _NAME.match(path.name)
        if m:
            out.append((m.group(1), m.group(2), path))
    return out


def file_section(path: Path, window: int) -> dict:
    doc = json.loads(path.read_text(encoding="utf-8"))
    rows = doc["rows"]
    pool = [c["text"] for r in rows for c in r["pool"]]
    pool_prompts = [r["prompt"] for r in rows for _ in r["pool"]]
    prompts = [r["prompt"] for r in rows]
    out = {
        "ckpt": doc.get("ckpt"),
        "probe_set": doc.get("probe_set"),
        "rows": len(rows),
        "pool": _uer(pool, window),
        "pool_prompt_as_context": _uer(pool, window, contexts=pool_prompts),
    }
    if window <= SELECTED_TRUNCATION:
        for column, label in (("echo_text", "selected"), ("base_text", "score_only_selected")):
            texts = [r[column] for r in rows]
            out[label] = _uer(texts, window)
            out[label + "_prompt_as_context"] = _uer(texts, window, contexts=prompts)
    else:
        out["selected"] = out["score_only_selected"] = None
        out["selected_refused"] = (
            f"echo_text/base_text are stored cut to {SELECTED_TRUNCATION} characters; "
            f"a window of {window} would score them over less text than the pool"
        )
    return out


def _pooled(readings: list[dict]) -> dict:
    """Sum k and n over several files and recompute -- never average the rates."""
    k = sum(r["k"] for r in readings)
    n = sum(r["n"] for r in readings)
    n_total = sum(r["n_total"] for r in readings)
    lo, hi = coherence.wilson_interval(k, n)
    return _tidy({
        "rate": k / n if n else None, "k": k, "n": n, "ci": [lo, hi],
        "qualifying_fraction": n / n_total if n_total else None, "n_total": n_total,
    })


def seed_section(files: dict[str, dict], index: list[tuple[str, str, Path]]) -> dict:
    """Per recipe: each training seed's rate, then mean and SAMPLE SD across seeds.

    A recipe is a run name with its `-sN` suffix removed, so `chat-v3d-aligned`,
    `-s1`, `-s2`, `-s3` are one recipe at four seeds. A seed's rate pools its
    three probe lists by summing k and n. SD is `ddof = 1`: four seeds are a
    sample of the recipe, not the population.
    """
    by_recipe: dict[str, dict[str, dict[str, str]]] = {}
    for probe_list, run, path in index:
        by_recipe.setdefault(_SEED.sub("", run), {}).setdefault(run, {})[probe_list] = path.name
    out = {}
    for recipe, runs in sorted(by_recipe.items()):
        section: dict = {"seeds": sorted(runs)}
        for column in ("pool", "pool_prompt_as_context", "selected",
                       "selected_prompt_as_context", "score_only_selected"):
            per_seed = {}
            for run in sorted(runs):
                readings = [files[name].get(column) for name in runs[run].values()]
                if any(r is None for r in readings):
                    continue
                per_seed[run] = _pooled(readings) | {"lists": sorted(runs[run])}
            rates = [v["rate"] for v in per_seed.values() if v["rate"] is not None]
            section[column] = {
                "per_seed": per_seed,
                "n_seeds": len(rates),
                "mean": round(statistics.fmean(rates), 6) if rates else None,
                "sd_seed": round(statistics.stdev(rates), 6) if len(rates) >= 2 else None,
            }
        out[recipe] = section
    return out


# ---------------------------------------------------------------------------
# the test fixture
# ---------------------------------------------------------------------------

def _cut_at_sentence(text: str, limit: int) -> str:
    """`text` up to its last sentence end at or before `limit`, newlines intact."""
    pos = end = 0
    for sentence in coherence.split_sentences(text):
        stop = text.index(sentence, pos) + len(sentence)
        if stop > limit:
            break
        pos = end = stop
    return text[:end]


def write_fixture(corpus: Path, out: Path) -> None:
    turns = story_turns(corpus)[:FIXTURE_N]
    rows = [{"prompt": u, "story": _cut_at_sentence(s, FIXTURE_CUT)} for u, s in turns]
    short = [r for r in rows if len(r["story"]) < coherence.WINDOW]
    if short:
        raise SystemExit(f"{len(short)} fixture stories fell below the window after the cut")
    doc = {
        "_about": (
            "UER fixture for tests/test_snnchat_coherence.py. CI has no data/ directory, so "
            "the instrument's can-fail and known-blindness tests run on this committed sample. "
            "Regenerate with scripts/chat/coherence_score.py --write-fixture."
        ),
        "source": f"data/chat/{corpus.name}, decoded with ChatTokenizer.decode_visible",
        "source_sha256": hashlib.sha256(corpus.read_bytes()).hexdigest(),
        "selection": (
            f"the FIRST {FIXTURE_N} conversations with a <|user|> and a <|bot|> turn, in file "
            "order. No story was chosen or dropped by content. Bare narratives are skipped."
        ),
        "truncation": (
            f"each story is cut at its last sentence end at or before {FIXTURE_CUT} characters "
            "to keep the file small; every story still qualifies at window 200."
        ),
        "n": len(rows),
        "rows": rows,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=1, ensure_ascii=True) + "\n", encoding="utf-8",
                   newline="\n")
    print(f"wrote {out} ({out.stat().st_size} bytes, {len(rows)} stories)")


# ---------------------------------------------------------------------------

def _line(label: str, r: dict | None) -> str:
    if r is None or r["rate"] is None:
        return f"  {label:<46} -"
    return (f"  {label:<46} {r['rate']:.4f} ({r['k']}/{r['n']}), 95% CI "
            f"[{r['ci'][0]:.4f}, {r['ci'][1]:.4f}]  qualifying {r['qualifying_fraction']:.4f}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--data-dir", type=Path, default=ROOT / "data" / "chat")
    ap.add_argument("--quality-dir", type=Path,
                    default=ROOT / "experiments" / "chat" / "_quality")
    ap.add_argument("--out", type=Path,
                    default=ROOT / "experiments" / "chat" / "_quality" / "coherence_v14.json")
    ap.add_argument("--window", type=int, default=coherence.WINDOW)
    ap.add_argument("--write-fixture", type=Path, default=None, metavar="PATH",
                    help="regenerate the committed test fixture at PATH and exit")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    corpus_path = args.data_dir / CORPUS
    if not corpus_path.exists():
        raise SystemExit(f"{corpus_path} not found; pass --data-dir (data/ is gitignored)")
    if args.write_fixture is not None:
        write_fixture(corpus_path, args.write_fixture)
        return

    index = pool_files(args.quality_dir)
    if not index:
        raise SystemExit(f"no v12_{{{','.join(LISTS)}}}_*.json under {args.quality_dir}")

    corpus = corpus_section(corpus_path, args.window)
    print(f"UER@{args.window} -- unintroduced-entity rate. NOT a coherence measure.")
    print(f"\ncorpus reference: {corpus['file']}, {corpus['n_story_turns']} story turns")
    print(_line("with stoplist", corpus["with_stoplist"]))
    for name in ("exposure", "uer_given_exposed", "unintroduced_per_subject"):
        r = corpus["decomposition_with_stoplist"][name]
        print(f"    {name:<44} {r['rate']:.4f} ({r['k']}/{r['n']}), 95% CI "
              f"[{r['ci'][0]:.4f}, {r['ci'][1]:.4f}]")
    print(_line("without stoplist", corpus["without_stoplist"]))
    print(_line("prototype rule, literal", corpus["prototype_rule_literal"]))
    print(_line("with stoplist, prompt as context", corpus["with_stoplist_prompt_as_context"]))
    for name in ("replace_alternate", "interleave"):
        mut = corpus["mutations"][name]
        print(_line(f"MUTANT {name}, with stoplist", mut["with_stoplist"]))
        print(_line(f"MUTANT {name}, without stoplist", mut["without_stoplist"]))
        print(_line(f"MUTANT {name}, prototype rule", mut["prototype_rule_literal"]))

    files = {}
    for probe_list, run, path in index:
        files[path.name] = file_section(path, args.window)
        print(f"\n{path.name}  [{probe_list} / {run}]")
        for label in ("pool", "pool_prompt_as_context", "selected",
                      "selected_prompt_as_context", "score_only_selected"):
            print(_line(label, files[path.name].get(label)))

    seeds = seed_section(files, index)
    print("\nacross training seeds (pool, three lists summed per seed):")
    for recipe, section in seeds.items():
        for column in ("pool", "pool_prompt_as_context", "selected", "score_only_selected"):
            s = section[column]
            if s["mean"] is None:
                continue
            per = ", ".join(f"{v['rate']:.4f}" for v in s["per_seed"].values())
            sd = "-" if s["sd_seed"] is None else f"{s['sd_seed']:.4f}"
            print(f"  {recipe:<20} {column:<24} mean {s['mean']:.4f}  SD_seed {sd}  "
                  f"(n={s['n_seeds']}: {per})")

    artifact = {
        "instrument": f"unintroduced-entity rate (UER@{args.window})",
        "not_a_coherence_measure": (
            "UER counts definite subjects whose head noun was never introduced, in a fixed "
            "window. It has never been validated against a human rating, it is blind to a "
            "plain sentence interleave of two stories (see corpus.mutations.interleave), and "
            "it must never be used inside snnchat.rerank. See src/snnchat/coherence.py."
        ),
        "window": args.window,
        "interval": "Wilson 95%; assumes independent replies, which pool draws are not",
        "corpus": corpus,
        "files": files,
        "across_training_seeds": seeds,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(artifact, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
