"""Recall of a told fact, in strata that can tell recall from recitation.

WHY A SECOND PROBE AND NOT A LARGER FIRST ONE
---------------------------------------------
`scripts/chat/memory_probe.py` is the committed floor (0/80 told, 0/80 untold at
every distance, `experiments/chat/_quality/memory_probe.json`) and it stays
byte-unchanged: `tests/test_snnchat_v15.py` pins its git blob. But the v15 round
trains a source, `snnchat.recall`, whose TRAIN templates contain that probe's
phrasings on purpose. A checkpoint that passes the committed probe alone has
shown that it recites a drilled exchange, which is what `snnchat.persona`
already does at 0.23 bpc and what `canned_rate` exists to flag. So the round's
PRIMARY is read here, on phrasings the generator reserves and refuses to emit.

    from memory_probe import ITEMS, hit, wilson, mcnemar

are imported, not copied: a second matcher would make the two probes'
tables incomparable.

THE STRATA
----------
Each is 10 items x 8 sampler seeds = 80 paired conversations per checkpoint
(`/80`, lattice 0.0125), at distance 0 unless noted. Item lists are module
constants so they are committed before any v15 checkpoint exists.

* **S1** -- trained values, HELD-OUT phrasings (statement and question both from
  `recall.HELDOUT_STATEMENTS` / `HELDOUT_QUESTIONS`). The gating stratum: tier
  RESOLVED and ship-candidate bar (ii) are read here.
* **S2** -- the SAME (slot, value, auxiliary noun) as S1 with TRAINED phrasings,
  at distances 0, 1 and 2. Reported. S1 against S2 at distance 0 is then a
  single-factor contrast: same values, same sampler seeds, only the wording
  differs. Distances 1 and 2 feed the carried-forward decision rule.
* **S3** -- HELD-OUT values in S2's phrasings and auxiliary nouns. Reported,
  never a gate: copying a value the model has never spelled is not the
  pre-registered goal. Trained phrasings on purpose, so that the only unseen
  thing is the value.
* **S4** -- two facts, then a question about ONE of them, against a
  SWAPPED-ROLES control (below). Trained phrasings, because this stratum is
  about binding a value to its slot and S1 already carries the phrasing test;
  stacking both would make a failure unreadable.
* **S5** -- a held-out FRAME: an attribute `snnchat.recall` never trains in any
  phrasing (`recall.HELDOUT_FRAMES`). Reported.

The control for S1, S2, S3 and S5 is the committed probe's: the identical
conversation with the establishing turn REMOVED, at the same sampler seed, so
the comparison is paired and exact McNemar applies.

THE SWAPPED-ROLES CONTROL, AND WHY S4 ONLY USES TWO SLOT PAIRS
--------------------------------------------------------------
S4's control tells the same two VALUES with their slots exchanged -- "my name
is rocky" / "i have a dog called harry" against the told "my name is harry" /
"i have a dog called rocky" -- asks the same question at the same sampler seed,
and is scored for the ORIGINAL target. The target word is in context in both
conversations; it is the answer in only one. So a model that answers "what is
my name?" with whichever in-context word looks like a name scores the same in
both and gets no credit, and only a value bound to its slot separates them.
That needs a value that is grammatical in both slots, which is why S4 is built
from name<->pet and colour<->object pairs and no others ("my favourite food is
lion" is not a control, it is a different experiment). The colour/object pairs
share one value list in the generator, so there the control is exact; on the
name/pet pairs the control's correct answer is a word never trained in that
slot, which makes the control conservative rather than wrong.

WHAT IS SCORED PER REPLY
------------------------
`classify` returns independent flags, plus one exclusive `category` for tables:

* `hit`             -- a target present as a whole word (`memory_probe.hit`);
* `swap`            -- the OTHER told fact's value (S4 only);
* `wrong_value`     -- a different value of the asked slot's own vocabulary,
  trained or held out, that nobody told ("Your name is Tom");
* `refusal`         -- the "haven't told me" family (`recall.UNTOLD_MARKER`
  and its plain-English variants);
* `persona_capture` -- stipulated `snnchat.persona` text, which is where the
  committed floor's replies went ("I don't really have a name. ...");
* `any_value`       -- any value of the slot at all; in the untold condition
  this is the confabulation count.

The guards in `score_v15.py` read the FLAGS, not the category: a reply that
names both the right value and a wrong one counts in both columns, which is the
stricter reading.

THE ACKNOWLEDGEMENT COLUMN
--------------------------
`RerankParams.echo=True` (the shipped default, and this probe's) prefers
drafts that repeat the user's content words, so on the establishing turn it
steers the acknowledgement toward the told value -- which then sits closer to
the question. Whether recall rides on that is measurable for free: the
acknowledgement's candidate pool is already drawn, so the echo-ON winner (the
one the conversation continues with) and the echo-OFF winner are both read off
the SAME pool and scored for the value, exactly as `echo_holdout.py` pairs its
selectors. `told_hit_by_ack` then splits the told hits by whether the
acknowledgement that was actually fed carried the value.

DECODER
-------
The committed probe's, so the two tables are comparable: n=8, lambda 0.6,
`max_new` 120, `RerankParams` defaults otherwise. `--shipped-decoder` is the ONE
extra pass the pre-registration allows: n=256 (`scripts/chat.py`'s
`DEFAULT_RERANK_N`) on the distance-0 conditions only. `max_new` stays 120
there too; an answer is one short sentence and the REPL's 400 would only buy
trailing text for `wrong_value` to fire on.

Sampler seeds start at `SEED_BASE`, not 0: the committed probe used 0-7.

    python scripts/chat/memory_probe_v2.py --ckpt <abs path> --out <abs path>
    python scripts/chat/memory_probe_v2.py --ckpt <abs path> --out <abs path> --shipped-decoder

Not part of the research protocol. No number here is a reported figure.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch  # noqa: E402
from memory_probe import FILLERS, ITEMS, hit, mcnemar, wilson  # noqa: E402

from snnchat import recall  # noqa: E402
from snnchat.generate import ChatSession, SamplingParams, load_chat_checkpoint  # noqa: E402
from snnchat.quality import _FALLBACK_ANYWHERE, _is_canned  # noqa: E402
from snnchat.rerank import RerankParams, fill_null_scores, final_ids  # noqa: E402
from snnchat.tokenizer import ChatTokenizer  # noqa: E402

#: First sampler seed. The committed probe drew seeds 0-7; these are fresh.
SEED_BASE = 1500
#: Sampler seeds per item. 10 items x 8 = the /80 lattice every bar is set on.
SEEDS = 8

STRATA: tuple[str, ...] = ("S1", "S2", "S3", "S4", "S5")
#: stratum -> the distances it is run at. Only S2 is swept; see the docstring.
DISTANCES: dict[str, tuple[int, ...]] = {
    "S1": (0,), "S2": (0, 1, 2), "S3": (0,), "S4": (0,), "S5": (0,),
}

#: The committed probe's decoder, and the one extra pass the round allows.
PLAIN_N = 8
SHIPPED_N = 256
LAM = 0.6
MAX_NEW = 120


@dataclass(frozen=True)
class Item:
    """One fact and how it is stated and asked for.

    `statement` and `question` are `snnchat.recall` TEMPLATES ({v} value, {x}
    auxiliary noun, {a} article), filled by `recall.fill`. `slot` is None for a
    held-out frame, which has no value vocabulary to score a wrong value in.
    `targets` defaults to the value; an age also accepts its digits, as the
    committed probe's item 8 does.
    """

    slot: str | None
    value: str
    aux: str | None
    statement: str
    question: str
    targets: tuple[str, ...] = ()

    def words(self) -> tuple[str, ...]:
        return self.targets or (self.value,)


@dataclass(frozen=True)
class TwoFact:
    """Two told facts and a question about `asked`. See the module docstring."""

    asked: Item
    other: Item
    asked_first: bool


#: (slot, value, aux) shared by S1 and S2. Chosen by hand before any v15
#: checkpoint existed: every value is in `recall.VALUES`, none is one of the
#: committed probe's (elliot, sarah, rex, mittens, blue, green, pizza, red), and
#: the slots are spread 2/2/1/2/2/1 so no single list carries the stratum.
_SEEN: tuple[tuple[str, str, str | None], ...] = (
    ("name", "alice", None), ("name", "jack", None),
    ("pet", "buddy", "dog"), ("pet", "luna", "cat"),
    ("colour", "purple", None),
    ("food", "cookies", None), ("food", "soup", None),
    ("animal", "tiger", None), ("animal", "owl", None),
    ("object", "yellow", "hat"),
)

#: The same slots and auxiliary nouns with values from `recall.HELDOUT_VALUES`.
_UNSEEN: tuple[tuple[str, str, str | None], ...] = (
    ("name", "oscar", None), ("name", "hannah", None),
    ("pet", "bandit", "dog"), ("pet", "hazel", "cat"),
    ("colour", "violet", None),
    ("food", "pasta", None), ("food", "bacon", None),
    ("animal", "sheep", None), ("animal", "goose", None),
    ("object", "navy", "hat"),
)

#: Item-by-item (statement, question) from the RESERVED tables. "who am i?" and
#: "who is my {x}?" are left unused: a reply of "you are a person" is a fair
#: answer to them and would be scored as a miss.
_HELDOUT_PHRASING: tuple[tuple[str, str], ...] = (
    ("everyone calls me {v}", "do you remember my name?"),
    ("i go by {v}", "what am i called?"),
    ("i've got a {x} called {v}", "do you remember what my {x} is called?"),
    ("we call our {x} {v}", "what did i name my {x}?"),
    ("the colour i like best is {v}", "which colour is my favourite?"),
    ("the food i like best is {v}", "do you remember my favourite food?"),
    ("nothing beats {v} for me", "what food do i like?"),
    ("the animal i like best is the {v}", "which animal is my favourite?"),
    ("i think the {v} is the best animal", "do you remember my favourite animal?"),
    ("i've got {a} {v} {x}", "do you remember the colour of my {x}?"),
)

#: Item-by-item (statement, question) from the TRAIN tables.
_TRAIN_PHRASING: tuple[tuple[str, str], ...] = (
    ("my name is {v}", "what is my name?"),
    ("i am {v}", "what's my name?"),
    ("i have a {x} called {v}", "what is my {x} called?"),
    ("my {x} is called {v}", "what is my {x}'s name?"),
    ("my favourite colour is {v}", "what is my favourite colour?"),
    ("my favourite food is {v}", "what is my favourite food?"),
    ("i love eating {v}", "which food do i like most?"),
    ("my favourite animal is the {v}", "what is my favourite animal?"),
    ("the {v} is my favourite animal", "which animal do i like most?"),
    ("i have {a} {v} {x}", "what colour is my {x}?"),
)


def _items(facts, phrasing) -> tuple[Item, ...]:
    return tuple(Item(slot, value, aux, st, qu)
                 for (slot, value, aux), (st, qu) in zip(facts, phrasing))


S1_ITEMS: tuple[Item, ...] = _items(_SEEN, _HELDOUT_PHRASING)
S2_ITEMS: tuple[Item, ...] = _items(_SEEN, _TRAIN_PHRASING)
S3_ITEMS: tuple[Item, ...] = _items(_UNSEEN, _TRAIN_PHRASING)


def _fact(slot: str, value: str, aux: str | None = None) -> Item:
    """A fact in its slot's FIRST trained phrasing, for the two-fact stratum."""
    return Item(slot, value, aux, recall.TRAIN_STATEMENTS[slot][0],
                recall.TRAIN_QUESTIONS[slot][0])


#: asked fact, distractor fact, whether the asked fact is stated first. Five
#: of each order. Only name<->pet and colour<->object: the control needs a
#: value that is grammatical in BOTH slots.
S4_ITEMS: tuple[TwoFact, ...] = (
    TwoFact(_fact("name", "harry"), _fact("pet", "rocky", "dog"), True),
    TwoFact(_fact("name", "lucy"), _fact("pet", "coco", "cat"), False),
    TwoFact(_fact("name", "ben"), _fact("pet", "daisy", "rabbit"), True),
    TwoFact(_fact("pet", "max", "dog"), _fact("name", "kate"), False),
    TwoFact(_fact("pet", "bella", "cat"), _fact("name", "paul"), True),
    TwoFact(_fact("colour", "pink"), _fact("object", "black", "car"), True),
    TwoFact(_fact("colour", "white"), _fact("object", "orange", "ball"), False),
    TwoFact(_fact("colour", "brown"), _fact("object", "blue", "coat"), False),
    TwoFact(_fact("object", "silver", "kite"), _fact("colour", "gold"), False),
    TwoFact(_fact("object", "red", "cup"), _fact("colour", "yellow"), True),
)


def _frame(index: int, value: str, targets: tuple[str, ...] = ()) -> Item:
    statement, question, _markers = recall.HELDOUT_FRAMES[index]
    return Item(None, value, None, statement, question, targets)


#: Attributes `snnchat.recall` never trains: a town, an age, a sister, a sport.
#: The sisters carry TRAINED names, so a miss there is not a spelling failure.
#: None repeats the committed probe's values (ashby, seven).
S5_ITEMS: tuple[Item, ...] = (
    _frame(0, "milton"), _frame(0, "oakley"), _frame(0, "redford"),
    _frame(1, "nine", ("nine", "9")), _frame(1, "twelve", ("twelve", "12")),
    _frame(2, "emma"), _frame(2, "zoe"), _frame(2, "laura"),
    _frame(3, "football"), _frame(3, "tennis"),
)

ITEM_SETS: dict[str, tuple] = {
    "S1": S1_ITEMS, "S2": S2_ITEMS, "S3": S3_ITEMS, "S4": S4_ITEMS, "S5": S5_ITEMS,
}

#: The committed probe's items by the `snnchat.recall` slot that trains them
#: (None: a held-out frame, items 7 and 8). Used by `binding_probe.py` and by
#: `score_v15.py` to report item 1 ('elliot') as the seen value it is.
COMMITTED_SLOTS: tuple[tuple[str | None, str | None], ...] = (
    ("name", None), ("name", None), ("pet", "dog"), ("pet", "cat"),
    ("colour", None), ("colour", None), (None, None), (None, None),
    ("food", None), ("object", "bicycle"),
)
assert len(COMMITTED_SLOTS) == len(ITEMS)

_REFUSAL = re.compile(
    r"\b(?:haven't|havent|have not|didn't|didnt|did not|never) (?:told|tell|said|say|mentioned)\b"
    r"|\b(?:don't|dont|do not) know (?:your|what|which|that|who|where|how)\b"
    r"|\bnot told me\b"
)
assert _REFUSAL.search(recall.UNTOLD_MARKER)

FLAGS: tuple[str, ...] = ("hit", "swap", "wrong_value", "refusal", "persona_capture",
                          "any_value")
#: `category` is the first of these whose flag is set, else "other".
_CATEGORY_ORDER: tuple[str, ...] = ("hit", "swap", "wrong_value", "refusal",
                                    "persona_capture")


def slot_vocabulary(slot: str | None) -> tuple[str, ...]:
    """Every value a reply could name for `slot`, trained and held out alike."""
    if slot is None:
        return ()
    return recall.VALUES[slot] + recall.HELDOUT_VALUES[slot]


def classify(reply: str, slot: str | None, targets, *, other_values=()) -> dict:
    """Flags and one exclusive category for a reply. See the module docstring.

    `other_values` are values that WERE told, for a different slot (S4). They
    are scored as `swap` and excluded from `wrong_value`, so that on a
    colour/object pair -- one shared vocabulary -- the two columns do not count
    the same word twice.
    """
    text = reply.replace("’", "'")
    low = text.lower()
    told = {t.lower() for t in targets} | {o.lower() for o in other_values}
    vocab = slot_vocabulary(slot)
    flags = {
        "hit": bool(hit(text, targets)),
        "swap": bool(other_values) and bool(hit(text, other_values)),
        "wrong_value": bool(hit(text, [v for v in vocab if v not in told])),
        "refusal": bool(_REFUSAL.search(low)),
        "persona_capture": _is_canned(text) or any(p in low for p in _FALLBACK_ANYWHERE),
        "any_value": bool(hit(text, vocab)) if vocab else False,
    }
    flags["category"] = next((c for c in _CATEGORY_ORDER if flags[c]), "other")
    return flags


def user_turns(item, condition: str, distance: int = 0) -> tuple[list[str], list[int]]:
    """The user's side of one conversation, and which turns establish a fact.

    `condition` is "told", "untold" (S1, S2, S3, S5) or "swapped" (S4). Pure, so
    a test can read exactly what would be sent without a model.
    """
    fillers = list(FILLERS[:distance])
    if isinstance(item, TwoFact):
        a, o = item.asked, item.other
        if condition == "told":
            sa, so = (recall.fill(a.statement, a.value, a.aux),
                      recall.fill(o.statement, o.value, o.aux))
        elif condition == "swapped":
            sa, so = (recall.fill(a.statement, o.value, a.aux),
                      recall.fill(o.statement, a.value, o.aux))
        else:
            raise ValueError(f"a two-fact item has no {condition!r} condition")
        first, second = (sa, so) if item.asked_first else (so, sa)
        return [first, second, *fillers, recall.fill(a.question, "", a.aux)], [0, 1]
    question = recall.fill(item.question, "", item.aux)
    if condition == "told":
        return [recall.fill(item.statement, item.value, item.aux), *fillers, question], [0]
    if condition == "untold":
        return [*fillers, question], []
    raise ValueError(f"a single-fact item has no {condition!r} condition")


def echo_off_winner(cands, rp):
    """The candidate `rerank` returns with the echo partition OFF.

    The same two lines `echo_holdout.py` uses for its `base` selector: the
    length partition, then the score. `score` must already hold the null term
    (`fill_null_scores`), which `rerank` skips when the echo tier decided alone.
    """
    pool = [c for c in cands if c.n_chars >= rp.min_chars] or list(cands)
    return max(pool, key=lambda c: c.score)


@torch.no_grad()
def converse(model, tok, device, params, rp, turns, establishing, targets) -> dict:
    """Send `turns` in one session. Returns the last reply and the ack readout."""
    s = ChatSession(model, tok, device=device, params=params, rerank=rp)
    acks = []
    reply = ""
    for i, text in enumerate(turns):
        reply = s.send(text)
        if i not in establishing:
            continue
        off = reply
        if rp is not None and rp.n > 1 and s.last_candidates:
            fill_null_scores(model, s.last_candidates, s.last_rerank, device)
            base = echo_off_winner(s.last_candidates, s.last_rerank)
            off = tok.decode_visible(final_ids(base, tok, s.last_rerank)).strip()
        acks.append({"on": reply[:200], "off": off[:200],
                     "on_hit": bool(hit(reply, targets)), "off_hit": bool(hit(off, targets))})
    return {"reply": reply, "acks": acks}


def _rate(k: int, n: int) -> dict:
    lo, hi = wilson(k, n)
    return {"k": int(k), "n": int(n), "rate": round(k / n, 4) if n else 0.0,
            "ci": [round(lo, 4), round(hi, 4)]}


def summarise(rows: list[dict]) -> dict:
    """k/n with a Wilson interval for every flag, and the paired tests."""
    n = len(rows)
    out: dict = {"n": n, "control_kind": rows[0]["control_kind"] if rows else None}
    for cond in ("told", "control"):
        out[cond] = {f: _rate(sum(r[cond + "_flags"][f] for r in rows), n) for f in FLAGS}
        cats: dict[str, int] = {}
        for r in rows:
            c = r[cond + "_flags"]["category"]
            cats[c] = cats.get(c, 0) + 1
        out[cond]["category"] = dict(sorted(cats.items()))
    b = sum(1 for r in rows if r["control_flags"]["hit"] and not r["told_flags"]["hit"])
    c = sum(1 for r in rows if r["told_flags"]["hit"] and not r["control_flags"]["hit"])
    out["discordant"] = {"control_only": b, "told_only": c}
    out["mcnemar_p"] = mcnemar(b, c)

    # The acknowledgement column: both selectors read off ONE pool, so paired.
    acks = [r["acks"] for r in rows]
    on = sum(1 for a in acks if any(x["on_hit"] for x in a))
    off = sum(1 for a in acks if any(x["off_hit"] for x in a))
    only_on = sum(1 for a in acks
                  if any(x["on_hit"] for x in a) and not any(x["off_hit"] for x in a))
    only_off = sum(1 for a in acks
                   if any(x["off_hit"] for x in a) and not any(x["on_hit"] for x in a))
    with_ack = [r for r in rows if any(x["on_hit"] for x in r["acks"])]
    without = [r for r in rows if not any(x["on_hit"] for x in r["acks"])]
    out["ack_echo"] = {
        "echo_on": _rate(on, n), "echo_off": _rate(off, n),
        "discordant": {"on_only": only_on, "off_only": only_off},
        "mcnemar_p": mcnemar(only_off, only_on),
    }
    out["told_hit_by_ack"] = {
        "ack_carried_value": _rate(sum(r["told_flags"]["hit"] for r in with_ack),
                                   len(with_ack)),
        "ack_did_not": _rate(sum(r["told_flags"]["hit"] for r in without), len(without)),
    }
    return out


def run_stratum(model, tok, device, stratum: str, *, seeds, rp, max_new: int = MAX_NEW,
                distances=None, items=None, progress: bool = False) -> list[dict]:
    """Every (distance, item, seed) of one stratum, told and control paired."""
    items = ITEM_SETS[stratum] if items is None else items
    control = "swapped" if stratum == "S4" else "untold"
    rows = []
    for dist in (DISTANCES[stratum] if distances is None else distances):
        for index, item in enumerate(items):
            asked = item.asked if isinstance(item, TwoFact) else item
            others = (item.other.value,) if isinstance(item, TwoFact) else ()
            targets = asked.words()
            for seed in seeds:
                params = SamplingParams(seed=int(seed), max_new=max_new)
                res = {}
                for cond in ("told", control):
                    turns, establishing = user_turns(item, cond, dist)
                    res[cond] = converse(model, tok, device, params, rp, turns,
                                         establishing, targets)
                    res[cond]["turns"] = turns
                rows.append({
                    "stratum": stratum, "distance": dist, "item": index,
                    "slot": asked.slot, "value": asked.value, "aux": asked.aux,
                    "targets": list(targets), "other_values": list(others),
                    "seed": int(seed), "control_kind": control,
                    "told_turns": res["told"]["turns"],
                    "control_turns": res[control]["turns"],
                    "told": res["told"]["reply"][:300],
                    "control": res[control]["reply"][:300],
                    "told_flags": classify(res["told"]["reply"], asked.slot, targets,
                                           other_values=others),
                    "control_flags": classify(res[control]["reply"], asked.slot, targets,
                                              other_values=others),
                    "acks": res["told"]["acks"],
                })
            if progress:
                print(f"  [{stratum} d{dist}] item {index + 1}/{len(items)}", flush=True)
    return rows


def _print(stratum: str, dist: int, s: dict) -> None:
    t, c = s["told"], s["control"]
    d = s["discordant"]
    print(f"{stratum:>3} d{dist}  told {t['hit']['k']:>3}/{s['n']:<3} "
          f"[{t['hit']['ci'][0]:.3f}, {t['hit']['ci'][1]:.3f}]  "
          f"{s['control_kind']:>7} {c['hit']['k']:>3}/{s['n']:<3}  "
          f"+{d['told_only']}/-{d['control_only']} p={s['mcnemar_p']:.4f}  "
          f"wrong {t['wrong_value']['k']:>3} swap {t['swap']['k']:>3} "
          f"refuse {t['refusal']['k']:>3} persona {t['persona_capture']['k']:>3}  "
          f"ack on/off {s['ack_echo']['echo_on']['k']}/{s['ack_echo']['echo_off']['k']}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True, help="checkpoint, absolute path")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", required=True, help="artifact to write, absolute path")
    ap.add_argument("--seeds", type=int, default=SEEDS,
                    help=f"sampler seeds per item, from {SEED_BASE}; the bars assume {SEEDS}")
    ap.add_argument("--strata", nargs="*", default=list(STRATA), choices=STRATA)
    ap.add_argument("--items", type=int, default=None,
                    help="plumbing smoke only: the first N items of each stratum")
    ap.add_argument("--shipped-decoder", action="store_true",
                    help=f"the one n={SHIPPED_N} pass: distance 0 only")
    ap.add_argument("--max-new", type=int, default=MAX_NEW)
    args = ap.parse_args(argv)

    model, _cfg, _ck = load_chat_checkpoint(Path(args.ckpt), device=args.device)
    model.eval()
    tok = ChatTokenizer()
    n = SHIPPED_N if args.shipped_decoder else PLAIN_N
    rp = RerankParams(n=n, lam=LAM)
    seeds = [SEED_BASE + i for i in range(args.seeds)]

    t0 = time.perf_counter()
    rows: list[dict] = []
    summary: dict = {}
    print(f"memory_probe_v2, {Path(args.ckpt).parent.name}, decoder {rp.describe()}, "
          f"seeds {seeds[0]}..{seeds[-1]}")
    for stratum in args.strata:
        distances = (0,) if args.shipped_decoder else DISTANCES[stratum]
        items = ITEM_SETS[stratum][:args.items] if args.items else None
        got = run_stratum(model, tok, args.device, stratum, seeds=seeds, rp=rp,
                          max_new=args.max_new, distances=distances, items=items)
        rows += got
        summary[stratum] = {}
        for dist in distances:
            s = summarise([r for r in got if r["distance"] == dist])
            summary[stratum][str(dist)] = s
            _print(stratum, dist, s)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "probe": "memory_probe_v2", "ckpt": str(args.ckpt), "device": args.device,
        "decoder": {"n": n, "lam": LAM, "max_new": args.max_new, "echo": rp.echo,
                    "shipped_decoder": bool(args.shipped_decoder)},
        "seed_base": SEED_BASE, "seeds": args.seeds,
        "items_per_stratum": args.items or 10,
        "complete": args.items is None and args.seeds == SEEDS,
        "seconds": round(time.perf_counter() - t0, 1),
        "summary": summary, "rows": rows,
    }, indent=1), encoding="utf-8")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
