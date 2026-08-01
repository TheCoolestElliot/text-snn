"""Acquire a character-level corpus and prepare its splits.

    python scripts/data/download_corpus.py --corpus enwik8
    python scripts/data/download_corpus.py --corpus all --data-dir data

Downloads data/<name>.zip, verifies its SHA-256 against the value pinned in
snn.data.CORPORA, extracts and verifies the 100 MB member, builds the vocabulary
over the whole file, and writes data/<name>/{train,val,test}.bin as raw uint8
already remapped to contiguous ids, plus meta.json.

Idempotent: a second run re-verifies the checksums on disk and exits without
touching the network. That re-verification is the point -- it is what turns "we
downloaded enwik8" into gate A5.

data/ is gitignored; nothing this script writes is ever committed.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "src"))

from snn.data import CORPORA, SPLIT_NAMES, Corpus, ensure_corpus  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--corpus",
        default="enwik8",
        choices=(*sorted(CORPORA), "all"),
        help="corpus to acquire; 'all' does every registered corpus",
    )
    p.add_argument("--data-dir", "--data_dir", dest="data_dir", default="data")
    args = p.parse_args(argv)

    names = sorted(CORPORA) if args.corpus == "all" else [args.corpus]
    for name in names:
        t0 = time.perf_counter()
        print(f"[{name}] preparing under {os.path.abspath(args.data_dir)} ...", flush=True)
        paths = ensure_corpus(name, args.data_dir)
        corpus = Corpus(name, args.data_dir)
        dt = time.perf_counter() - t0
        with open(paths.meta_path, encoding="utf-8") as f:
            meta = json.load(f)
        print(f"[{name}] ok in {dt:.1f}s")
        print(f"[{name}]   source      {meta['source_url']}")
        print(f"[{name}]   zip sha256  {meta['zip_sha256']}")
        print(f"[{name}]   raw sha256  {meta['raw_sha256']}")
        print(f"[{name}]   vocab_size  {meta['vocab_size']}")
        for s in SPLIT_NAMES:
            n = corpus.split(s).shape[0]
            print(f"[{name}]   {s:<5} {n:>12,} chars  sha256 {meta['split_sha256'][s]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
