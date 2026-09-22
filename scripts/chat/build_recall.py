"""Pack the user-fact recall source: state a fact, get asked for it back.

    python scripts/chat/build_recall.py --out-dir <packed corpus dir>
    python scripts/chat/build_recall.py --out-dir <dir> --hazard-steps 8000

Adds ONE new file pair to the directory it is pointed at (`recall.bin`,
`recall.val.bin`) and registers it in that directory's `manifest.json`. Nothing
already packed is read, rewritten or deleted, so every checkpoint trained before
this still has the corpus it was trained on and an arm that wants the old
behaviour leaves `recall` out of its `--mix`. See `snnchat.recall` for what the
dialogues are and why.

THREE REFUSALS, AND NO --force
------------------------------
* **No default output directory.** `build_topic_stories.py` defaults to
  `data/chat`; this does not. The packed corpus that matters is a multi-gigabyte
  directory that other sessions train from, and a source that lands in it should
  do so because somebody typed its path.
* **An existing source name is never overwritten** -- not the `.bin`, not the
  `.val.bin`, not a manifest entry whose file has gone missing. A repack under
  the same name would make every checkpoint trained on the first pack
  unreproducible, which is the failure `build_topic_stories.py` guards with
  `--force`; here there is no flag, because a different generator seed or
  dialogue count is a different corpus and wants a different `--name`.
* **A manifest packed under another vocabulary is refused**, for the reason
  `snnchat.data.ChatCorpus` refuses to load one.

The manifest is MERGED and written atomically, as `build_topic_stories.py` does
it; `snnchat.build_corpus.build_all` records what happened the one time it was
replaced instead.

What a crash leaves behind follows from the second refusal. The bins are written
before the manifest entry, so a pack that dies in between leaves `recall.bin` /
`recall.val.bin` that no manifest lists: `snnchat.data.ChatCorpus` loads only
what the manifest names and ignores them, and a re-run refuses because the files
exist. Delete the two orphans by hand and pack again. A death between the
manifest's temporary file and its `os.replace` leaves the manifest intact and a
stray `manifest.json.tmp`, which nothing reads.

WHAT --hazard-steps MEASURES
----------------------------
`snnchat.data.MixtureSampler._windows` draws ONE `align_frac` decision per batch
row, for every source alike, so a quarter of this source's windows (at the
shipped 0.75) start at a uniformly random offset. When that offset falls between
a dialogue's statement and its answer, the answer is trained with its evidence
cut off -- a name produced from nothing, which is the confabulation the round's
guard is there to catch. `--hazard-steps N` replays N steps of the REAL sampler
over the freshly packed bin (its `_align` is subclassed to record the offsets it
chose; the recorded offsets are checked to reproduce `_windows` exactly) and
counts, among answers whose value's first character is a prediction target, how
many have their establishing value outside the window.

THE SAME REPLAY IS THE DOSE
---------------------------
`snnchat.recall.MIN_EXPOSURES` is a floor on how often each trained value is the
asked-for answer in one run. Dividing the characters trained on by the mean
dialogue length overstates it, because a window's far edge truncates a dialogue
and the answer is the dialogue's last turn. So the replay also counts, per
value, the answers that really are prediction targets -- all of them, and those
whose establishing value is inside the window with them -- and scales the count
from the windows replayed to the windows the recipe draws from this source.
That scaled count is an estimate of the run's EXPECTED dose; one training run's
own count for one value scatters about it like a Poisson count.

Not part of the research protocol; no number here is a reported figure.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import sys
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO / "src") not in sys.path:
    sys.path.insert(0, str(_REPO / "src"))

import numpy as np  # noqa: E402

from snnchat.build_corpus import _pack_source, _RawAwareTokenizer  # noqa: E402
from snnchat.data import ChatCorpus, MixtureSampler  # noqa: E402
from snnchat.recall import (  # noqa: E402
    ACK_ECHO_FRACTION,
    DOSE_RECIPE,
    MAX_RENDERED_CHARS,
    MIN_EXPOSURES,
    SHAPE_SHARES,
    SHAPES,
    SLOTS,
    VALUES,
    build_recall_dialogues,
    display,
    expected_exposures,
    rendered_length,
)
from snnchat.tokenizer import VOCAB_VERSION  # noqa: E402

#: Dialogues packed by default. Sized so that the pre-registered recipe reads the
#: packed file about once (`describe` prints the exact ratio as `passes`): few
#: enough passes that a hit cannot come from a memorised dialogue, since the
#: probe's conversations are not in the file either way.
DEFAULT_CONVERSATIONS = 250_000


def describe(dialogues) -> dict:
    """Length distribution, shape shares, echo rate and dose of a build.

    Everything here is a pure function of `(n, seed)` and costs seconds, so it is
    recorded in the manifest with the source instead of being re-derived in a
    document.
    """
    lengths = sorted(rendered_length(d.turns) for d in dialogues)
    n = len(lengths)
    mean = sum(lengths) / n

    def pct(q: float) -> int:
        return lengths[min(n - 1, int(q * n))]

    by_shape = {}
    for shape in SHAPES:
        ls = [rendered_length(d.turns) for d in dialogues if d.shape == shape]
        by_shape[shape] = {
            "share": len(ls) / n,
            "target_share": SHAPE_SHARES[shape],
            "mean_chars": sum(ls) / max(len(ls), 1),
            "max_chars": max(ls, default=0),
        }

    # How often each trained value is the asked-for answer in this BUILD, scaled
    # by the number of passes the recipe makes over it. Like `expected_exposures`
    # (its expectation) this counts every dialogue whole, and the sampler does
    # not deliver that: an answer is its dialogue's last turn, so the dialogue a
    # window's far edge truncates loses exactly its answer. It is an UPPER
    # estimate and is not what `MIN_EXPOSURES` is read against -- `window_hazard`
    # counts the dose the sampler really delivers.
    asked: dict[tuple[str, str], int] = {(s, v): 0 for s in SLOTS for v in VALUES[s]}
    for d in dialogues:
        if d.value is not None:
            asked[(d.slot, d.value)] += 1
    recipe_chars = (DOSE_RECIPE["max_steps"] * DOSE_RECIPE["batch_size"]
                    * DOSE_RECIPE["seq_len"] * DOSE_RECIPE["mix_weight"])
    scale = (recipe_chars / mean) / n
    realised = {}
    for slot in SLOTS:
        counts = [asked[(slot, v)] * scale for v in VALUES[slot]]
        realised[slot] = {"min": min(counts), "mean": statistics.fmean(counts),
                          "max": max(counts), "values": len(counts)}
    return {
        "conversations": n,
        "rendered_chars": {"min": lengths[0], "mean": mean, "median": pct(0.5),
                           "p90": pct(0.9), "p99": pct(0.99), "max": lengths[-1],
                           "ceiling": MAX_RENDERED_CHARS},
        "shapes": by_shape,
        "ack_echo": {"share": sum(d.echo for d in dialogues) / n,
                     "target": ACK_ECHO_FRACTION},
        "dose": {
            "recipe": dict(DOSE_RECIPE),
            "recipe_chars_from_this_source": recipe_chars,
            "passes": recipe_chars / sum(lengths),
            "upper_estimate_per_value": expected_exposures(mean),
            "upper_estimate_in_this_build": realised,
        },
    }


def _value_at(text: str, value: str) -> int:
    """Index of `value` as a whole word in `text`, case-insensitively."""
    found = re.search(r"\b" + re.escape(value.lower()) + r"\b", text.lower())
    if found is None:
        raise AssertionError(f"value {value!r} not in turn {text!r}")
    return found.start()


def _spans(dialogues) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Stream positions, per TOLD dialogue, of the three places its value lives.

    Returns `(answer, statement, echo)`: the position of the value's first
    character in the answering bot turn, in the user's establishing turn, and in
    the LAST acknowledgement before the answer that repeats it (-1 if none). A
    turn's text begins one id after its role marker, and a dialogue begins one id
    after the previous one ends -- the layout `ChatTokenizer.render_conversation`
    writes and `snnchat.recall.rendered_length` counts.
    """
    answer, statement, echo = [], [], []
    start = 0
    for d in dialogues:
        if d.value is not None:
            turn_start, pos = [], start + 1                 # +1: <|bos|>
            for _role, text in d.turns:
                turn_start.append(pos + 1)                  # +1: the role marker
                pos += len(text) + 2
            shown = display(d.slot, d.value)
            answer.append(turn_start[d.answer_turn]
                          + _value_at(d.turns[d.answer_turn][1], shown))
            statement.append(turn_start[d.statement_turn]
                             + _value_at(d.turns[d.statement_turn][1], d.value))
            last = -1
            for i in range(d.statement_turn + 1, d.answer_turn):
                role, text = d.turns[i]
                if role == "bot" and re.search(r"\b" + re.escape(d.value) + r"\b", text.lower()):
                    last = turn_start[i] + _value_at(text, d.value)
            echo.append(last)
        start += rendered_length(d.turns)
    return (np.asarray(answer, dtype=np.int64), np.asarray(statement, dtype=np.int64),
            np.asarray(echo, dtype=np.int64))


class _RecordingSampler(MixtureSampler):
    """`MixtureSampler` that remembers the offsets `_windows` settled on.

    `_align` is the one place the final offsets and the aligned rows exist
    together, and overriding it records them without touching the generator
    stream -- so what is measured is the sampler's own draw, not a
    re-implementation of it.
    """

    def _align(self, s, offs, take):
        offs = super()._align(s, offs, take)
        self.last_offsets = offs.copy()
        self.last_aligned = np.zeros(offs.shape[0], dtype=bool)
        self.last_aligned[take] = True
        return offs


def window_hazard(data_dir: str, name: str, dialogues, *, steps: int,
                  batch_size: int, seq_len: int, align_frac: float,
                  align_lookahead: int, seed: int = 0) -> dict:
    """Share of trained answers whose evidence the window cut off. See the module docstring.

    An answer COUNTS in a window when the first character of its value is a
    prediction target there (positions `o+1 .. o+seq_len`): that character is the
    decision, the rest is spelling. Its evidence is CUT when the establishing
    value starts before the window does. `no_evidence` is the stricter count in
    which no value-repeating acknowledgement is inside the window either.
    """
    if align_frac <= 0.0:
        raise ValueError("align_frac must be positive: with alignment off the sampler "
                         "never calls _align and there are no offsets to record")
    corpus = ChatCorpus(data_dir)
    sampler = _RecordingSampler(corpus, {name: 1.0}, batch_size, seq_len, seed,
                                align_frac=align_frac, align_lookahead=align_lookahead)
    if sampler.names != [name]:
        raise AssertionError(f"sampler resolved {sampler.names}, expected [{name!r}]")
    stream = sampler.arrays[0]
    answer, statement, echo = _spans(dialogues)
    keys = [(s, v) for s in SLOTS for v in VALUES[s]]
    index = {key: i for i, key in enumerate(keys)}
    code = np.asarray([index[(d.slot, d.value)] for d in dialogues if d.value is not None],
                      dtype=np.int64)
    keep = answer < len(stream)                 # drop the held-out tail
    answer, statement, echo, code = answer[keep], statement[keep], echo[keep], code[keep]
    span = np.arange(seq_len + 1, dtype=np.int64)
    asked = np.zeros(len(keys), dtype=np.int64)
    with_evidence = np.zeros(len(keys), dtype=np.int64)

    tally = {kind: {"answers": 0, "cut": 0, "no_evidence": 0, "windows": 0}
             for kind in ("aligned", "random")}
    for step in range(steps):
        block = sampler._windows(step)
        offs = sampler.last_offsets
        if not np.array_equal(stream[offs[:, None] + span[None, :]], block):
            raise AssertionError(f"recorded offsets do not reproduce _windows at step {step}")
        lo = np.searchsorted(answer, offs + 1, side="left")
        hi = np.searchsorted(answer, offs + seq_len, side="right")
        for row in range(offs.shape[0]):
            o, a, b = int(offs[row]), int(lo[row]), int(hi[row])
            t = tally["aligned" if sampler.last_aligned[row] else "random"]
            t["windows"] += 1
            t["answers"] += b - a
            cut = statement[a:b] < o
            t["cut"] += int(cut.sum())
            t["no_evidence"] += int((cut & (echo[a:b] < o)).sum())
            np.add.at(asked, code[a:b], 1)
            np.add.at(with_evidence, code[a:b][~cut], 1)

    total = {k: sum(t[k] for t in tally.values()) for k in ("answers", "cut", "no_evidence",
                                                            "windows")}
    # The dose, from the same replay. `DOSE_RECIPE` trains on max_steps *
    # batch_size * mix_weight windows of this source in expectation; the replay
    # drew `total["windows"]` of them, and a count scales by the ratio. It is the
    # recipe's dose only when the replayed windows are the recipe's length.
    recipe_windows = (DOSE_RECIPE["max_steps"] * DOSE_RECIPE["batch_size"]
                      * DOSE_RECIPE["mix_weight"])
    scale = recipe_windows / total["windows"]
    dose: dict = {"recipe_windows": recipe_windows, "replayed_windows": total["windows"],
                  "scale": scale, "seq_len_is_the_recipes": seq_len == DOSE_RECIPE["seq_len"],
                  "min_exposures": MIN_EXPOSURES, "per_slot": {}, "per_value": {}}
    for slot in SLOTS:
        rows = {}
        for label, counts in (("asked", asked), ("with_evidence", with_evidence)):
            scaled = [float(counts[index[(slot, v)]]) * scale for v in VALUES[slot]]
            rows[label] = {"min": min(scaled), "mean": statistics.fmean(scaled),
                           "max": max(scaled)}
            rows[f"{label}_under_floor"] = [v for v, c in zip(VALUES[slot], scaled)
                                            if c < MIN_EXPOSURES]
        rows["values"] = len(VALUES[slot])
        dose["per_slot"][slot] = rows
        dose["per_value"][slot] = {
            v: [float(asked[index[(slot, v)]]) * scale,
                float(with_evidence[index[(slot, v)]]) * scale] for v in VALUES[slot]}
    return {
        "dose": dose,
        "steps": steps, "batch_size": batch_size, "seq_len": seq_len,
        "align_frac": align_frac, "align_lookahead": align_lookahead, "sampler_seed": seed,
        "by_window": tally, "total": total,
        "cut_rate": total["cut"] / max(total["answers"], 1),
        "no_evidence_rate": total["no_evidence"] / max(total["answers"], 1),
    }


def _register(manifest_path: str, name: str, entry: dict, vocab_size: int, *,
              replace_own: bool) -> None:
    """Merge one source entry into the manifest and write it atomically.

    The manifest is re-read HERE, immediately before the write, not reused from
    the check at the top of `main`: packing takes long enough for another process
    to register a source in the same directory, and writing back a stale copy
    would drop it -- the defect `snnchat.build_corpus.build_all` documents having
    had. `replace_own` is True only for this run's second write (adding the
    hazard measurement to the entry it has just created).
    """
    base: dict = {"vocab_version": VOCAB_VERSION, "vocab_size": vocab_size,
                  "built_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "sources": {}}
    if os.path.exists(manifest_path):
        with open(manifest_path, encoding="utf-8") as fh:
            base = json.load(fh)
    sources = base.setdefault("sources", {})
    if name in sources and not replace_own:
        raise SystemExit(f"{name!r} was registered in {manifest_path} while this pack "
                         f"ran; leaving the manifest untouched")
    sources[name] = entry
    tmp = manifest_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(base, fh, indent=2, sort_keys=True)
        fh.write("\n")
    os.replace(tmp, manifest_path)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out-dir", "--data-dir", dest="out_dir", default=None,
                   help="the packed corpus directory to add the source to (REQUIRED; "
                        "the two spellings are the same option)")
    p.add_argument("--name", default="recall",
                   help="source name; must not exist in the directory or its manifest")
    p.add_argument("--conversations", type=int, default=DEFAULT_CONVERSATIONS)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--hazard-steps", type=int, default=0,
                   help="replay this many sampler steps over the packed bin and count "
                        "answers whose establishing turn the window cut off (0 = skip)")
    p.add_argument("--batch-size", type=int, default=int(DOSE_RECIPE["batch_size"]))
    p.add_argument("--seq-len", type=int, default=int(DOSE_RECIPE["seq_len"]))
    p.add_argument("--align-frac", type=float, default=0.75)
    p.add_argument("--align-lookahead", type=int, default=1024)
    args = p.parse_args(argv)

    if not args.out_dir:
        raise SystemExit(
            "refusing to run without --out-dir (or --data-dir): this script has no "
            "default output directory on purpose. Name the packed corpus directory "
            "the source should be added to."
        )
    out_dir = args.out_dir
    manifest_path = os.path.join(out_dir, "manifest.json")
    manifest = None
    if os.path.exists(manifest_path):
        with open(manifest_path, encoding="utf-8") as fh:
            manifest = json.load(fh)
        if int(manifest.get("vocab_version", VOCAB_VERSION)) != VOCAB_VERSION:
            raise SystemExit(
                f"{manifest_path} was packed under vocab v{manifest['vocab_version']}; "
                f"this code packs v{VOCAB_VERSION}. The ids mean different characters."
            )
        if args.name in manifest.get("sources", {}):
            raise SystemExit(
                f"{manifest_path} already lists a source named {args.name!r}; this "
                f"script never overwrites one. Pass a different --name."
            )
    for suffix in (".bin", ".val.bin", ".all.bin"):
        existing = os.path.join(out_dir, args.name + suffix)
        if os.path.exists(existing):
            raise SystemExit(
                f"{existing} exists; this script never overwrites a packed source. "
                f"Pass a different --name."
            )
    os.makedirs(out_dir, exist_ok=True)

    print(f"generating {args.conversations:,} recall dialogues (seed {args.seed})", flush=True)
    dialogues = build_recall_dialogues(args.conversations, seed=args.seed)
    report = describe(dialogues)
    lengths = report["rendered_chars"]
    print(f"  rendered chars: min {lengths['min']}, mean {lengths['mean']:.1f}, "
          f"median {lengths['median']}, p90 {lengths['p90']}, p99 {lengths['p99']}, "
          f"max {lengths['max']} (ceiling {MAX_RENDERED_CHARS})")
    for shape, row in report["shapes"].items():
        print(f"  {shape:>9}: share {row['share']:.4f} (target {row['target_share']:.2f}), "
              f"mean {row['mean_chars']:.1f} chars, max {row['max_chars']}")
    print(f"  acknowledgement repeats the value in {report['ack_echo']['share']:.4f} "
          f"of conversations (target {ACK_ECHO_FRACTION})")
    dose = report["dose"]
    print(f"  dose under the recipe ({dose['recipe_chars_from_this_source'] / 1e6:.1f} M chars "
          f"from this source, {dose['passes']:.2f} passes over this build):")
    print("    (dialogues x passes: an UPPER estimate; --hazard-steps replays the sampler "
          "and prints the dose it delivers)")
    for slot in SLOTS:
        r = dose["upper_estimate_in_this_build"][slot]
        print(f"    {slot:>7}: {r['values']:>3} values, at most "
              f"{dose['upper_estimate_per_value'][slot]:,.0f} asked each in expectation, "
              f"in this build min {r['min']:,.0f} / max {r['max']:,.0f}")

    tok = _RawAwareTokenizer()
    stats = _pack_source(args.name, (list(d.turns) for d in dialogues), out_dir, tok)
    expected_total = sum(rendered_length(d.turns) for d in dialogues)
    if stats["chars_total"] != expected_total:
        raise AssertionError(
            f"packed {stats['chars_total']:,} chars but the dialogues render to "
            f"{expected_total:,}: snnchat.recall.rendered_length no longer matches the "
            f"tokenizer, and every length guarantee in that module is void"
        )

    entry = {
        **stats,
        "generator": "snnchat.recall.build_recall_conversations",
        "seed": args.seed,
        "report": report,
    }
    _register(manifest_path, args.name, entry, tok.vocab_size, replace_own=False)
    print(f"registered {args.name} in {manifest_path}")

    if args.hazard_steps > 0:
        hazard = window_hazard(out_dir, args.name, dialogues, steps=args.hazard_steps,
                               batch_size=args.batch_size, seq_len=args.seq_len,
                               align_frac=args.align_frac,
                               align_lookahead=args.align_lookahead)
        tot = hazard["total"]
        print(f"  hazard (a), {args.hazard_steps} steps x B{args.batch_size} x L{args.seq_len}, "
              f"align_frac {args.align_frac}:")
        print(f"    answers trained: {tot['answers']:,}; establishing value cut off: "
              f"{tot['cut']:,} = {hazard['cut_rate']:.4f}; with no repeating "
              f"acknowledgement in view either: {tot['no_evidence']:,} = "
              f"{hazard['no_evidence_rate']:.4f}")
        for kind, t in hazard["by_window"].items():
            print(f"    {kind:>8} windows {t['windows']:,}: {t['cut']:,}/{t['answers']:,} cut")
        replayed = hazard["dose"]
        print(f"  dose the sampler delivers, scaled to the recipe's "
              f"{replayed['recipe_windows']:,.0f} windows of this source (floor "
              f"{MIN_EXPOSURES}; per value, min / mean / max):")
        for slot in SLOTS:
            r = replayed["per_slot"][slot]
            a, e = r["asked"], r["with_evidence"]
            print(f"    {slot:>7}: asked {a['min']:,.0f} / {a['mean']:,.0f} / {a['max']:,.0f}; "
                  f"with the establishing value in the window {e['min']:,.0f} / "
                  f"{e['mean']:,.0f} / {e['max']:,.0f}")
            if r["with_evidence_under_floor"]:
                print(f"    WARNING: under the floor of {MIN_EXPOSURES} with evidence in view: "
                      f"{', '.join(r['with_evidence_under_floor'])}")
        if not replayed["seq_len_is_the_recipes"]:
            print(f"    WARNING: replayed at L{args.seq_len}, not the recipe's "
                  f"L{int(DOSE_RECIPE['seq_len'])}; the scaled dose is not the recipe's")
        entry["hazard_evidence_cut"] = hazard
        _register(manifest_path, args.name, entry, tok.vocab_size, replace_own=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
