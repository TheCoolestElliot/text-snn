"""Is a told value BOUND, or merely primed? Teacher-forced, no sampler.

WHY THIS EXISTS BESIDE `memory_probe_v2.py`
-------------------------------------------
A sampled probe reads 0 or 1 per reply, and at the committed floor it reads 0
every time: 0/80 told, 0/80 untold. Zero is not a graded readout. It cannot
tell a model that puts 40 % of its mass on the told name (and loses the draw to
the persona line) from one that puts none there, and a round that moves the
first number without moving the second would be reported as a null.

This scores the ANSWER instead of sampling one. The context is a whole
conversation up to and including the opening of the canonical reply
(`snnchat.recall.expected_answer`, cut immediately before the value), and
`snnchat.rerank.score_under_prefix` returns the model's log-probability of the
VALUE characters that follow -- "Alice" after "Your name is ". Three contexts,
distance 0, one fixed value-free acknowledgement between statement and
question:

* **told**    -- the statement carries the true value;
* **untold**  -- no statement and no acknowledgement, the question asked cold;
* **swapped** -- the statement carries a DIFFERENT value of the same slot, and
  the TRUE value's characters are scored. Averaged over `N_SWAPS` alternatives
  taken at fixed strides through the slot's list, so one unlucky neighbour
  does not set the number.

Reported in bits per VALUE character, lower is better, so both differences
are NEGATIVE when the mechanism works:

* `told - untold`  -- the statement helps at all. This alone is priming: any
  name in context could raise the probability of every name.
* `told - swapped` -- the statement helps THIS value and not just its class.
  This is the binding readout.

The acknowledgement is fixed and carries no value ("Okay, I'll remember
that.", one of `snnchat.recall`'s plain acknowledgements) because at probe time
the real one is the model's own sample; a scorer with a sampler inside it is
what this file exists to avoid. The shipped checkpoint has never seen that
line as an acknowledgement and a v15 checkpoint has, which is one more reason
the comparison that matters is within a checkpoint, across contexts.

A DIAGNOSTIC, NEVER A GATE. Its reading on SHIPPED is the floor recorded before
training; `score_v15.py` prints it and fires nothing on it.

    python scripts/chat/binding_probe.py --ckpt <abs path> --out <abs path>

Not part of the research protocol. No number here is a reported figure.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch  # noqa: E402
from memory_probe import ITEMS  # noqa: E402
from memory_probe_v2 import COMMITTED_SLOTS, S1_ITEMS, S2_ITEMS, S3_ITEMS  # noqa: E402

from snnchat import recall  # noqa: E402
from snnchat.generate import load_chat_checkpoint  # noqa: E402
from snnchat.rerank import score_under_prefix  # noqa: E402
from snnchat.tokenizer import BOS, BOT, ChatTokenizer  # noqa: E402

#: The acknowledgement placed between statement and question. Value-free.
ACK = "Okay, I'll remember that."
#: Alternatives averaged in the swapped context.
N_SWAPS = 3


def committed_items() -> list[dict]:
    """The committed probe's eight TRAINED-frame items, as templates.

    Its town and its age are held-out frames with no canonical answer to score.
    The establishing turn is turned back into a template so the swapped context
    can carry another value; the object item goes through the generator's own
    template because its article has to agree ("an orange bicycle"). Each
    template must reproduce the committed turn exactly, or this raises.
    """
    out = []
    for (establish, question, targets), (slot, aux) in zip(ITEMS, COMMITTED_SLOTS):
        if slot is None:
            continue
        statement = ("i have {a} {v} {x}" if slot == "object"
                     else establish.replace(targets[0], "{v}"))
        if recall.fill(statement, targets[0], aux) != establish:
            raise AssertionError(f"{statement!r} does not reproduce {establish!r}")
        out.append({"group": "committed", "slot": slot, "value": targets[0], "aux": aux,
                    "statement": statement, "question": question})
    return out


def probe_items() -> list[dict]:
    """Every scored item: S2, S1 and S3 of `memory_probe_v2`, then the committed."""
    out = []
    for group, items in (("S2", S2_ITEMS), ("S1", S1_ITEMS), ("S3", S3_ITEMS)):
        for it in items:
            out.append({"group": group, "slot": it.slot, "value": it.value, "aux": it.aux,
                        "statement": it.statement, "question": it.question})
    return out + committed_items()


def swap_values(slot: str, value: str, k: int = N_SWAPS) -> list[str]:
    """`k` other TRAINED values of `slot`, at even strides from `value`'s place.

    Deterministic. A held-out value is not in the list and strides from index
    0, so it is also swapped against trained values -- which is the comparison
    wanted there: does an unseen told value beat a seen untold-here one.
    """
    pool = [v for v in recall.VALUES[slot] if v != value]
    start = recall.VALUES[slot].index(value) if value in recall.VALUES[slot] else 0
    stride = max(len(pool) // (k + 1), 1)
    return [pool[(start + (i + 1) * stride) % len(pool)] for i in range(k)]


def answer_split(slot: str, value: str, aux: str | None) -> tuple[str, str]:
    """(`"Your name is "`, `"Alice"`): the canonical reply cut at its value."""
    answer = recall.expected_answer(slot, value, aux)
    shown = recall.display(slot, value)
    at = answer.rindex(shown)
    return answer[:at], shown


def context_ids(tok, item: dict, told_value: str | None) -> list[int]:
    """The conversation up to the value: `told_value=None` is the untold context."""
    ids = [BOS]
    if told_value is not None:
        ids += tok.render_turn("user", recall.fill(item["statement"], told_value, item["aux"]))
        ids += tok.render_turn("bot", ACK)
    ids += tok.render_turn("user", recall.fill(item["question"], "", item["aux"]))
    lead, _shown = answer_split(item["slot"], item["value"], item["aux"])
    return ids + [BOT] + [int(i) for i in tok.encode(lead)]


@torch.no_grad()
def score_item(model, tok, device, item: dict) -> dict:
    _lead, shown = answer_split(item["slot"], item["value"], item["aux"])
    seq = [int(i) for i in tok.encode(shown)]
    swaps = swap_values(item["slot"], item["value"])

    def bpc(told_value):
        (logp,) = score_under_prefix(model, context_ids(tok, item, told_value), [seq], device)
        return -logp / (len(seq) * math.log(2.0))

    told, untold = bpc(item["value"]), bpc(None)
    per_swap = [bpc(v) for v in swaps]
    swapped = sum(per_swap) / len(per_swap)
    return {**item, "value_chars": len(seq), "swapped_with": swaps,
            "bpc_told": told, "bpc_untold": untold, "bpc_swapped": swapped,
            "bpc_swapped_each": per_swap,
            "told_minus_untold": told - untold, "told_minus_swapped": told - swapped}


def pooled(rows: list[dict]) -> dict:
    """Character-weighted means: a bpc over all value characters of the group."""
    n = sum(r["value_chars"] for r in rows)
    if n == 0:
        return {"items": 0, "value_chars": 0}

    def mean(key: str) -> float:
        return sum(r[key] * r["value_chars"] for r in rows) / n

    out = {"items": len(rows), "value_chars": n,
           "bpc_told": mean("bpc_told"), "bpc_untold": mean("bpc_untold"),
           "bpc_swapped": mean("bpc_swapped")}
    out["told_minus_untold"] = out["bpc_told"] - out["bpc_untold"]
    out["told_minus_swapped"] = out["bpc_told"] - out["bpc_swapped"]
    # How many ITEMS point the right way. A sign count, not a test: the items
    # share a model and several share a phrasing.
    out["items_told_below_swapped"] = sum(1 for r in rows if r["told_minus_swapped"] < 0)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True, help="checkpoint, absolute path")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", required=True, help="artifact to write, absolute path")
    ap.add_argument("--items", type=int, default=None,
                    help="plumbing smoke only: the first N items")
    args = ap.parse_args(argv)

    model, _cfg, _ck = load_chat_checkpoint(Path(args.ckpt), device=args.device)
    model.eval()
    tok = ChatTokenizer()
    items = probe_items()[:args.items] if args.items else probe_items()
    rows = [score_item(model, tok, args.device, it) for it in items]

    print(f"binding probe, {Path(args.ckpt).parent.name}: bits per VALUE character")
    print(f"{'group':<10}{'slot':<8}{'value':<10}{'told':>8}{'untold':>8}{'swapped':>9}"
          f"{'t-u':>9}{'t-s':>9}")
    for r in rows:
        print(f"{r['group']:<10}{r['slot']:<8}{r['value']:<10}{r['bpc_told']:>8.3f}"
              f"{r['bpc_untold']:>8.3f}{r['bpc_swapped']:>9.3f}"
              f"{r['told_minus_untold']:>+9.3f}{r['told_minus_swapped']:>+9.3f}")
    groups = {g: pooled([r for r in rows if r["group"] == g])
              for g in dict.fromkeys(r["group"] for r in rows)}
    groups["all"] = pooled(rows)
    print()
    for g, p in groups.items():
        print(f"{g:<10} {p['items']:>2} items  told-untold {p['told_minus_untold']:>+8.4f}  "
              f"told-swapped {p['told_minus_swapped']:>+8.4f}  "
              f"({p['items_told_below_swapped']}/{p['items']} items below swapped)")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "probe": "binding_probe", "ckpt": str(args.ckpt), "device": args.device,
        "ack": ACK, "n_swaps": N_SWAPS, "complete": args.items is None,
        "unit": "bits per value character; negative differences mean the told "
                "context makes the true value more likely",
        "pooled": groups, "rows": rows,
    }, indent=1), encoding="utf-8")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
