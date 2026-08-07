"""How much of a reply is trained with its own request in frame.

    python scripts/chat/window_coverage.py stories_topic stories_short

`docs/chat/QUALITY.md` §4.3 ordered four arms by a single mechanism -- the share
of training windows that carry a request together with the reply it conditions --
and the ordering held. That number was computed ad hoc for one source and one
sampler setting. This script computes it properly, for any packed source, so that
the next arm's dose can be stated **before** it trains rather than reconstructed
afterwards.

THE QUANTITY, AND WHY IT IS NOT THE ONE THAT WAS REPORTED
----------------------------------------------------------
`window_coverage.json` reported `random_window_carries_prompt = 0.046`: the
probability that a uniformly-placed window contains a request and any part of its
reply. That is the right quantity for "does the dependency appear at all", and it
is the wrong one for "how much of the dose is delivered", because a window that
contains a request plus the first 215 characters of a 700-character story
delivers the dependency for 215 characters and not for the other 485.

`reply_chars_in_frame` is the dose. For a reply character at stream position `p`
whose conversation starts at `s`, a window of length `L` that contains `p` at all
starts uniformly in `(p-L, p]`, and it also contains `s` exactly when it starts in
`(p-L, s]`. So

    P(request in frame | p in the window) = clip(1 - (p - s) / L, 0, 1)

and averaging that over every reply character in the source gives the fraction of
the reply-character gradient that is conditioned on its own request. It is
between 0 and 1, it equals the reported quantity only when replies are much
shorter than `L`, and it is the number that should have been compared across
arms.

Both are reported, because the older one is what the existing table is written in
and a new metric that quietly replaces an old one makes two rounds
incomparable.

ALIGNMENT IS REPORTED SEPARATELY, NOT FOLDED IN
------------------------------------------------
`align_frac` changes where windows start, so it changes the dose. But the
realised alignment rate depends on `align_lookahead` and on the conversation
length distribution -- 0.75 realises 38 % at lookahead 256 and 74 % at 1024 -- so
folding it into one number would hide the thing that made the previous round's
arms differ. `--align-frac` reports the aligned and unaligned doses separately
and then the mixture of the two.

Not part of the research protocol.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO / "src") not in sys.path:
    sys.path.insert(0, str(_REPO / "src"))

from snnchat.tokenizer import BOS, BOT, EOT  # noqa: E402


def measure(path: Path, seq_len: int, *, max_chars: int = 60_000_000) -> dict:
    """Conversation geometry and the two coverage numbers, from the packed ids."""
    data = np.memmap(path, dtype=np.uint8, mode="r")
    n = min(len(data), max_chars)
    ids = np.asarray(data[:n])

    starts = np.flatnonzero(ids == BOS)
    if starts.size < 3:
        raise SystemExit(f"{path}: only {starts.size} conversation starts in {n:,} chars")
    # A conversation runs from its BOS to the next one. The last is dropped: it is
    # truncated by the read limit and its length would be a measurement artefact.
    conv_start = starts[:-1]
    conv_len = np.diff(starts)

    bot = np.flatnonzero(ids == BOT)
    eot = np.flatnonzero(ids == EOT)

    # A reply is the span between a <|bot|> and the next <|eot|>. Raw-narrative
    # records have no <|bot|> at all and contribute no reply characters, which is
    # correct: there is no request for them to be in frame with.
    nxt_eot = np.searchsorted(eot, bot, side="left")
    keep = nxt_eot < eot.size
    bot, nxt_eot = bot[keep], nxt_eot[keep]
    reply_end = eot[nxt_eot]

    # Which conversation each reply belongs to: the last BOS at or before it.
    owner = np.searchsorted(starts, bot, side="right") - 1
    ok = owner >= 0
    bot, reply_end, owner = bot[ok], reply_end[ok], owner[ok]
    own_start = starts[owner]

    # Distance from the conversation's first character to each reply character.
    # Summed in closed form per reply rather than expanded, so a 300 M-character
    # source does not need a 300 M-element array: for a reply occupying offsets
    # [a, b) from the conversation start, sum of clip(1 - d/L, 0, 1) over d in
    # [a, b) is an arithmetic series truncated at d = L.
    a = (bot + 1 - own_start).astype(np.int64)
    b = (reply_end - own_start).astype(np.int64)
    b = np.maximum(b, a)
    lo = np.minimum(a, seq_len)
    hi = np.minimum(b, seq_len)
    k = np.maximum(hi - lo, 0)                       # characters within reach
    # sum_{d=lo}^{hi-1} (1 - d/L) = k - (lo + hi - 1) * k / (2L)
    covered = k - (lo + hi - 1) * k / (2.0 * seq_len)
    covered = np.maximum(covered, 0.0)
    reply_chars = np.maximum(b - a, 0)
    total_reply = int(reply_chars.sum())

    # A uniformly-placed window that contains the request AND at least one
    # character of its reply. The window [o, o+L) holds the whole request when
    # `o <= own_start` and `own_start + a <= o + L`, which is `L - a` legal
    # starts -- note that most of them are BEFORE the conversation begins, in the
    # tail of the previous one, which is a perfectly ordinary window.
    reach = np.maximum(seq_len - a, 0)
    carries = float(reach.sum()) / float(n)

    # QUALITY.md §4.1's 4.6 % is a STRICTER quantity and the difference is worth
    # keeping visible rather than silently improving on: it counts only windows
    # that begin inside the request itself ("the request sits in the first ~34
    # characters of a conversation, so only a window starting inside that span
    # contains it"), which excludes every window that starts in the previous
    # conversation and rolls over into this one. Those windows are real and they
    # do carry the dependency, so `carries` above is the truer version -- but the
    # existing table is written in this one, and a metric that quietly changes
    # definition between rounds makes the two rounds incomparable.
    starts_in_request = float(np.minimum(a, seq_len).sum()) / float(n)

    # With alignment, the window starts exactly at the conversation start, so a
    # reply character is in frame iff its offset is below seq_len.
    aligned_dose = float(np.minimum(b, seq_len).sum() - np.minimum(a, seq_len).sum())
    aligned_dose = aligned_dose / max(total_reply, 1)

    return {
        "source": path.stem,
        "chars_read": int(n),
        "conversations": int(conv_start.size),
        "mean_conversation_chars": float(conv_len.mean()),
        "median_conversation_chars": float(np.median(conv_len)),
        "replies": int(bot.size),
        "mean_reply_chars": float(reply_chars.mean()) if bot.size else 0.0,
        "mean_request_prefix_chars": float(a.mean()) if bot.size else 0.0,
        "seq_len": seq_len,
        "random_window_carries_prompt": carries,
        "window_starts_in_request": starts_in_request,
        "reply_chars_in_frame_unaligned": float(covered.sum()) / max(total_reply, 1),
        "reply_chars_in_frame_aligned": aligned_dose,
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("sources", nargs="+")
    p.add_argument("--data-dir", default="data/chat")
    p.add_argument("--seq-len", type=int, default=256)
    p.add_argument("--align-frac", type=float, default=0.0,
                   help="realised alignment rate to mix the two doses at "
                        "(the REALISED rate, not the requested one)")
    p.add_argument("--out", default=None)
    args = p.parse_args(argv)

    rows = []
    for name in args.sources:
        row = measure(Path(args.data_dir) / f"{name}.bin", args.seq_len)
        f = args.align_frac
        row["align_frac_realised"] = f
        row["reply_chars_in_frame"] = (
            (1.0 - f) * row["reply_chars_in_frame_unaligned"]
            + f * row["reply_chars_in_frame_aligned"]
        )
        rows.append(row)

    print(f"{'source':<16} {'conv':>7} {'reply':>7} {'strict':>7} {'carries':>8} "
          f"{'dose@0':>8} {'dose@1':>8} {'dose':>8}")
    for r in rows:
        print(f"{r['source']:<16} {r['mean_conversation_chars']:7.0f} "
              f"{r['mean_reply_chars']:7.0f} {r['window_starts_in_request']:7.3f} "
              f"{r['random_window_carries_prompt']:8.3f} "
              f"{r['reply_chars_in_frame_unaligned']:8.3f} "
              f"{r['reply_chars_in_frame_aligned']:8.3f} "
              f"{r['reply_chars_in_frame']:8.3f}")
    print(f"\nseq_len {args.seq_len}, alignment mixed at realised {args.align_frac:g}")
    print("strict  = QUALITY.md §4.1's quantity: P(a window BEGINS inside the request)")
    print("carries = P(a random window holds a request and some of its reply)")
    print("dose@0  = share of reply characters trained with the request in frame, "
          "unaligned windows")
    print("dose@1  = the same, windows aligned to the conversation start")

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as fh:
            json.dump(rows, fh, indent=1)
            fh.write("\n")
        print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
