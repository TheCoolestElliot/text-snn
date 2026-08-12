"""Rebuild `data/chat/manifest.json` from the packed files it describes.

WHY THIS EXISTS
---------------
`build_all` used to write `sources=stats` outright. That was harmless while
every call packed every source, and wrong the moment `only=` was added on
2026-08-11: a call that packed one source rewrote the manifest to list one
source. Two `--only` repacks in a row left fourteen packed `.bin` files
describing themselves as three, and `snnchat.data.ChatCorpus` enumerates what is
available from exactly that dict -- so eleven sources became invisible to
training while their data sat untouched on disk.

`build_all` now merges. This repairs a manifest already damaged that way.

WHAT IS AND IS NOT RECOVERABLE
------------------------------
`chars_train` and `chars_val` are recovered EXACTLY, because ids are packed as
`uint8` and the on-disk size of `<name>.bin` is the character count -- the
property `build_corpus`'s own docstring engineers for.

Everything else is provenance and is gone for any entry the overwrite ate:
`conversations`, `seconds`, `built_from`, `shaper`, `seed`, `raw_fraction`,
`max_turn_chars`. Where a build log records it, it is restored and marked; where
nothing records it, the entry says `"reconstructed": true` and does not invent
the missing fields. A manifest that quietly filled them in with plausible values
would be worse than one that is honest about what it lost.

    python scripts/chat/repair_manifest.py --dry-run
    python scripts/chat/repair_manifest.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from snnchat.tokenizer import VOCAB_VERSION, ChatTokenizer  # noqa: E402

#: Provenance recovered from `experiments/chat/_session9_repacks.log` and from
#: the manifest as it read earlier on 2026-08-11, before the overwrite. Only
#: fields actually witnessed are listed; nothing here is inferred.
KNOWN: dict[str, dict] = {
    "stories_topic_r05": {"built_from": "tinystories.txt", "conversations": 526_241,
                          "raw_fraction": 0.05, "seed": 7,
                          "topic_prompts": "snnchat.topics.story_request"},
    "soda_narrative": {"built_from": "soda_train.parquet", "conversations": 1_190_303,
                       "max_turn_chars": 400, "read_from": "soda_narrative"},
    "alpaca_t600": {"conversations": 45_466, "max_turn_chars": 600, "read_from": "alpaca"},
    "dolly_t600": {"conversations": 9_714, "max_turn_chars": 600, "read_from": "dolly"},
    "oasst1_t600": {"conversations": 1_193, "max_turn_chars": 600, "read_from": "oasst1"},
    "persona": {"conversations": 60_000, "chars_total": 5_359_997},
    "stories_short": {"built_from": "tinystories.txt", "conversations": 1_012_158,
                      "raw_fraction": 0.08, "seed": 11,
                      "shaper": "snnchat.shortform.short_conversation"},
    "stories_topic": {"built_from": "tinystories.txt"},
    "stories_short300": {"built_from": "tinystories.txt"},
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/chat")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    d = ROOT / args.data_dir
    path = d / "manifest.json"
    existing = {}
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8")).get("sources", {})

    sources: dict[str, dict] = {}
    for binf in sorted(d.glob("*.bin")):
        if binf.name.endswith(".val.bin"):
            continue
        name = binf.name[:-4]
        val = d / f"{name}.val.bin"
        entry = {
            "chars_train": binf.stat().st_size,
            "chars_val": val.stat().st_size if val.exists() else 0,
        }
        prior = existing.get(name, {})
        if prior:
            # An entry that survived keeps everything it had.
            entry = {**prior, **entry}
        else:
            entry.update(KNOWN.get(name, {}))
            entry["reconstructed"] = True
            entry["reconstructed_note"] = (
                "chars_* recovered exactly from file size; other provenance was "
                "lost to the build_all(only=) manifest overwrite of 2026-08-11"
            )
        sources[name] = entry

    tok = ChatTokenizer()
    out = {
        "vocab_version": VOCAB_VERSION,
        "vocab_size": tok.vocab_size,
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "sources": sources,
    }

    print(f"{'source':<22} {'chars_train':>13} {'chars_val':>10}  state")
    for name, e in sorted(sources.items()):
        state = "reconstructed" if e.get("reconstructed") else "intact"
        print(f"{name:<22} {e['chars_train']:>13,} {e['chars_val']:>10,}  {state}")
    print(f"\n{len(sources)} sources; manifest listed {len(existing)} before")

    if args.dry_run:
        print("\n--dry-run: nothing written")
        return 0
    tmp = str(path) + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2, sort_keys=True)
        fh.write("\n")
    os.replace(tmp, path)
    print(f"\nwrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
