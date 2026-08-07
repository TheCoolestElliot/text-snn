"""Download the conversational sources and pack them into id streams.

    python scripts/chat/build_data.py

Idempotent at source granularity: a source already present in `--raw-dir` is not
re-downloaded, and one already packed into `--out-dir` is not re-packed. To
rebuild one source, delete its `.bin` and `.val.bin` and run this again.

Not part of the research protocol; writes only under `data/chat_raw/` and
`data/chat/`.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO / "src") not in sys.path:
    sys.path.insert(0, str(_REPO / "src"))

from snnchat.build_corpus import build_all  # noqa: E402
from snnchat.sources import SOURCES, download_all  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--raw-dir", default="data/chat_raw")
    p.add_argument("--out-dir", default="data/chat")
    p.add_argument("--only", nargs="*", default=None,
                   help=f"restrict to these sources: {[s.name for s in SOURCES]}")
    p.add_argument("--persona-conversations", type=int, default=60_000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--skip-download", action="store_true")
    args = p.parse_args(argv)

    if not args.skip_download:
        print("downloading sources")
        download_all(args.raw_dir, only=args.only)
    print("packing")
    manifest = build_all(
        args.raw_dir, args.out_dir,
        seed=args.seed, persona_conversations=args.persona_conversations,
    )
    print(manifest.to_json())
    total = sum(s.get("chars_train", 0) for s in manifest.sources.values())
    print(f"\ntotal training characters: {total:,} ({total / 1e9:.2f} G)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
