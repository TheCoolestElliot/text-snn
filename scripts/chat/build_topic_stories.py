"""Pack a story corpus whose requests name the story's SUBJECT, not its cast.

    python scripts/chat/build_topic_stories.py --limit-chars 400000000

Adds ONE new file pair to `data/chat/` (`stories_topic.bin`, `.val.bin`) and
registers it in the manifest. Nothing already packed is read, rewritten or
deleted -- `tinystories.bin` is left exactly as it was, so every checkpoint
trained before this still has the corpus it was trained on, and an arm that
wants the old behaviour just leaves the new source out of its `--mix`.

WHY A NEW SOURCE RATHER THAN A REPACK
-------------------------------------
Two reasons, one practical and one about evidence. Practically, `tinystories.bin`
is 747 MB and repacking it in place would make every earlier run
unreproducible. About evidence: keeping both files means the mixture weight
between them is a knob rather than a fork, so "how much of this does the model
need" is a number a later run can change, and the comparison in
`docs/chat/QUALITY.md` is between two mixes of the same packing code rather than
between two corpora.

See `snnchat.topics` for how a story's subject is chosen and for the honest
caveat about measuring this with a probe that asks for exactly this form.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO / "src") not in sys.path:
    sys.path.insert(0, str(_REPO / "src"))

from snnchat.build_corpus import _pack_source, _RawAwareTokenizer  # noqa: E402
from snnchat.sources import read_source  # noqa: E402
from snnchat.topics import POLICIES, story_request  # noqa: E402

#: Fraction of stories emitted as bare narrative rather than as an answer to a
#: request. Lower than `build_corpus`'s 0.5 on purpose: `tinystories.bin` is
#: still in the mix and is still 50 % raw, so the raw-narrative signal is
#: already paid for and this file's job is the conditioning it does not carry.
#:
#: **Now overridable via `--raw-fraction`, default UNCHANGED.** Lowering it is
#: the correctly-signed version of the dose lever `TOPIC_DOSE_NOTE.md` §4
#: proposed: re-weighting the topic distribution is zero-sum and, measured
#: against the note's own `1/count^0.5`, 14 of the 15 word probes LOSE dose.
#: Cutting the raw-narrative share instead raises *every* noun topic at once,
#: by roughly 1/(1-0.15) over 1/(1-0.05) = 1.12x from this term plus the
#: generic-request branch in `snnchat.topics`. It is not zero-sum because the
#: characters come from raw narrative, which `tinystories.bin` already supplies
#: at 50 %.
RAW_FRACTION = 0.15


def _conversations(path: str, rng: random.Random, raw_fraction: float = RAW_FRACTION,
                   policy=None):
    for conv in read_source("tinystories", path):
        story = conv[0][1]
        if rng.random() < raw_fraction:
            yield [("_raw", story)]
            continue
        request = story_request(story, rng, policy=policy)
        yield [("user", request), ("bot", story)]


def _dry_run(raw: str, args, policy) -> int:
    """Report the request distribution a pack would have, without packing.

    THE POINT OF THIS IS THE PRE-REGISTRATION
    -----------------------------------------
    The subject distribution a policy produces is a deterministic function of
    the corpus and the seed, so it is knowable BEFORE any GPU time is spent. A
    bar for the arm can then be derived from the counts the arm will actually
    train on, rather than guessed -- and if a policy turns out to move the
    counts hardly at all, that is worth learning for two CPU-minutes instead of
    three GPU-hours.

    Streams the identical generator the packer streams, and histograms the
    subject of every request it would emit. The character budget is approximated
    from the text rather than from rendered ids -- it is within a fraction of a
    percent, and this function decides how many stories to read, not what gets
    written.
    """
    rng = random.Random(args.seed)
    subj = re.compile(r"about (?:a|an|the) ([a-z]+)")
    counts: dict[str, int] = {}
    n_conv = n_req = total = 0
    for conv in _conversations(raw, rng, args.raw_fraction, policy):
        n_conv += 1
        total += sum(len(t) for _role, t in conv) + 4
        if conv[0][0] == "user":
            n_req += 1
            for m in subj.finditer(conv[0][1]):
                w = m.group(1)
                counts[w] = counts.get(w, 0) + 1
        if args.limit_chars and total >= args.limit_chars:
            break

    ordered = sorted(counts.values(), reverse=True)
    tot = sum(ordered)
    print(f"\n  policy {policy.name} (alpha {policy.alpha})")
    print(f"  {n_conv:,} conversations, {n_req:,} requests, "
          f"{len(counts):,} distinct subjects, {tot:,} subject mentions")
    for thresh in (1000, 300, 200, 100, 30, 10):
        k = sum(1 for v in ordered if v >= thresh)
        share = sum(v for v in ordered if v >= thresh) / max(tot, 1)
        print(f"    {k:>5} subjects asked >= {thresh:>4} times ({share:6.1%} of requests)")
    top = sorted(counts.items(), key=lambda kv: -kv[1])[:10]
    print("    most-drilled: " + ", ".join(f"{w} {c}" for w, c in top))

    out = os.path.join("experiments", "chat", "_quality", f"dry_run_{args.name}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump({"name": args.name, "policy": policy.name, "alpha": policy.alpha,
                   "seed": args.seed, "raw_fraction": args.raw_fraction,
                   "conversations": n_conv, "requests": n_req,
                   "counts": counts}, fh, sort_keys=True)
    print(f"\n  wrote {out}")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--raw-dir", default="data/chat_raw")
    p.add_argument("--out-dir", default="data/chat")
    p.add_argument("--name", default="stories_topic")
    p.add_argument("--limit-chars", type=int, default=400_000_000)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--force", action="store_true")
    p.add_argument("--raw-fraction", type=float, default=RAW_FRACTION,
                   help="share emitted as bare narrative rather than as an answer "
                        f"to a request (default {RAW_FRACTION}, the shipped corpus)")
    p.add_argument("--subject-policy", default="head", choices=sorted(POLICIES),
                   help="which of a story's up-to-three eligible subjects fills "
                        "the request. 'head' is the shipped rule (always the most "
                        "frequent) and is bit-reproducible; 'inverse' draws with "
                        "weight 1/(1+count). See snnchat.topics.SubjectPolicy")
    p.add_argument("--dry-run", action="store_true",
                   help="stream the same conversations and report the request "
                        "distribution the pack WOULD have, writing no corpus")
    args = p.parse_args(argv)

    # A non-default raw fraction produces a DIFFERENT corpus, and eight committed
    # checkpoints were trained on `stories_topic.bin` as it stands. Repacking it
    # in place would make every one of them unreproducible, so the two are not
    # allowed to coincide: change the fraction, change the name.
    if args.raw_fraction != RAW_FRACTION and args.name == "stories_topic":
        raise SystemExit(
            f"--raw-fraction {args.raw_fraction} differs from the shipped "
            f"{RAW_FRACTION}; pass --name (e.g. stories_topic_r05) so this does "
            f"not overwrite the corpus eight committed checkpoints were trained on"
        )
    # Same rule, same reason, for the other lever that changes what is packed.
    if args.subject_policy != "head" and args.name == "stories_topic":
        raise SystemExit(
            f"--subject-policy {args.subject_policy} is not the shipped rule; "
            f"pass --name (e.g. stories_topic_flat) so this does not overwrite "
            f"the corpus eight committed checkpoints were trained on"
        )
    policy = POLICIES[args.subject_policy]

    raw = os.path.join(args.raw_dir, "tinystories.txt")
    if not os.path.exists(raw):
        raise SystemExit(f"{raw} not found; run scripts/chat/build_data.py first")
    if args.dry_run:
        return _dry_run(raw, args, policy)
    out = os.path.join(args.out_dir, f"{args.name}.bin")
    if os.path.exists(out) and not args.force:
        raise SystemExit(f"{out} exists; pass --force to repack")

    tok = _RawAwareTokenizer()
    rng = random.Random(args.seed)
    print(f"packing {args.name} from {raw} "
          f"(limit {args.limit_chars / 1e6:,.0f} M chars, raw fraction {args.raw_fraction})",
          flush=True)
    stats = _pack_source(args.name,
                         _conversations(raw, rng, args.raw_fraction, policy),
                         args.out_dir, tok,
                         limit_chars=args.limit_chars)

    manifest_path = os.path.join(args.out_dir, "manifest.json")
    with open(manifest_path, encoding="utf-8") as fh:
        manifest = json.load(fh)
    manifest["sources"][args.name] = {
        **stats,
        "built_from": "tinystories.txt",
        "raw_fraction": args.raw_fraction,
        "topic_prompts": "snnchat.topics.story_request",
        "subject_policy": {"name": policy.name, "alpha": policy.alpha},
        "seed": args.seed,
    }
    tmp = manifest_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, sort_keys=True)
        fh.write("\n")
    os.replace(tmp, manifest_path)
    print(f"registered {args.name} in {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
