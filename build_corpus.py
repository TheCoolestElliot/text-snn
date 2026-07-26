"""
build_corpus.py
===============

Build the training corpus (``input.txt``) for the character LM from public-domain
prose, reproducibly.

The corpus is four complete, out-of-copyright novels from Project Gutenberg,
chosen for *plain, modern-reading English narrative with plenty of dialogue* --
the point is a model that writes ordinary prose, so the data has to be ordinary
prose. They are concatenated and normalized down to a small ASCII character set,
which is what makes a character LM's one-neuron-per-symbol input layer cheap.

Usage::

    python build_corpus.py                    # download (cached), clean, write input.txt
    python build_corpus.py --check            # just hash the existing input.txt
    python build_corpus.py --out other.txt

Downloads are cached under ``--cache-dir`` (default ``.corpus_cache/``), so a
rebuild is offline and instant. The expected sha256 of the finished corpus is
pinned in ``EXPECTED_SHA256``; a mismatch is reported (not fatal) because Project
Gutenberg occasionally re-releases a text.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import sys
import unicodedata
import urllib.request
from typing import List, Tuple

# Project Gutenberg ebook ids. All four are in the public domain in the US.
BOOKS: List[Tuple[int, str, str]] = [
    (1661, "The Adventures of Sherlock Holmes", "Arthur Conan Doyle"),
    (55,   "The Wonderful Wizard of Oz",        "L. Frank Baum"),
    (35,   "The Time Machine",                  "H. G. Wells"),
    (11,   "Alice's Adventures in Wonderland",  "Lewis Carroll"),
]

URL_TEMPLATE = "https://www.gutenberg.org/cache/epub/{id}/pg{id}.txt"

# sha256 of the corpus this repo's results were measured on. Printed and compared
# on every build so a silently-changed upstream text cannot quietly invalidate the
# numbers in the README.
EXPECTED_SHA256 = "7b0f147ae27cb1e276495d2a0fb697fc7f6e8a0932a20e56bcf641de5b1e5ac0"

# Gutenberg wraps every text in a licence header/footer delimited by these lines.
_START_RE = re.compile(r"^\*\*\* START OF (?:THE|THIS) PROJECT GUTENBERG EBOOK.*$",
                       re.MULTILINE)
_END_RE = re.compile(r"^\*\*\* END OF (?:THE|THIS) PROJECT GUTENBERG EBOOK.*$",
                     re.MULTILINE)

# Editorial insertions rather than prose: "[Illustration]", "[Footnote: ...]",
# "[later editions continued as follows ...]". Brackets are balanced and never
# nested in these texts, so a non-greedy span (which may run over several lines)
# removes each note whole.
_BRACKETED_RE = re.compile(r"\[[^\[\]]{0,4000}\]", re.DOTALL)

# Gutenberg plain-text marks italics with _underscores_; they are typography, not
# language, so they are stripped rather than taught to the model. The span may be
# broken across a line, so newlines are allowed inside it.
_EMPHASIS_RE = re.compile(r"_([^_]{1,200}?)_", re.DOTALL)

# Asterisk scene-break rules ("* * * * *") -- decoration, not language.
_STAR_RULE_RE = re.compile(r"^[ \t*]*\*[ \t*]*$", re.MULTILINE)

# Typographic characters mapped to their plain-ASCII equivalents. Everything here
# is a *rendering* distinction, so folding it shrinks the vocabulary without
# losing any information the model could use.
_PUNCT_MAP = {
    "‘": "'", "’": "'", "‚": "'", "‛": "'",
    "“": '"', "”": '"', "„": '"', "‟": '"',
    "–": "-", "—": "--", "―": "--", "−": "-",
    "…": "...", "′": "'", "″": '"',
    " ": " ", " ": " ", " ": " ", " ": " ",
    "«": '"', "»": '"', "‹": "'", "›": "'",
    "æ": "ae", "Æ": "AE", "œ": "oe", "Œ": "OE",
    "ß": "ss", "ø": "o", "Ø": "O",
    "£": "L", "°": " degrees ", "½": "1/2", "¼": "1/4",
    "†": "", "‡": "", "©": "", "®": "",
}


def fetch(book_id: int, cache_dir: str) -> str:
    """Return a book's raw text, downloading it once and caching it on disk."""
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, f"pg{book_id}.txt")
    if not os.path.exists(path):
        url = URL_TEMPLATE.format(id=book_id)
        print(f"[fetch] {url}")
        # Gutenberg blocks the stdlib default User-Agent.
        req = urllib.request.Request(url, headers={"User-Agent": "build_corpus/1.0"})
        with urllib.request.urlopen(req, timeout=60) as r:
            data = r.read()
        with open(path, "wb") as f:
            f.write(data)
    else:
        print(f"[cache] {path}")
    with open(path, "r", encoding="utf-8-sig") as f:
        return f.read()


def strip_gutenberg_boilerplate(text: str, book_id: int) -> str:
    """Cut the licence header and footer, keeping only the book itself."""
    m_start = _START_RE.search(text)
    m_end = _END_RE.search(text)
    if not m_start or not m_end or m_end.start() <= m_start.end():
        raise ValueError(
            f"could not locate the Project Gutenberg start/end markers in ebook "
            f"{book_id}; the upstream file format may have changed.")
    return text[m_start.end():m_end.start()]


def normalize(text: str) -> str:
    """Fold a Gutenberg text down to a small, clean ASCII character set.

    A character LM has one input neuron per distinct symbol, so every stray
    typographic variant (curly vs straight quote, en vs em dash, an accented
    letter appearing twice in a million characters) is a neuron that sees almost
    no training signal. Folding them away is the single most useful piece of
    preprocessing for this model.
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _BRACKETED_RE.sub("", text)
    text = _EMPHASIS_RE.sub(r"\1", text)
    text = _STAR_RULE_RE.sub("", text)
    for src, dst in _PUNCT_MAP.items():
        text = text.replace(src, dst)
    # Decompose accents (e -> e + combining acute) and drop the combining marks,
    # so "café" becomes "cafe" instead of vanishing entirely.
    text = "".join(c for c in unicodedata.normalize("NFKD", text)
                   if not unicodedata.combining(c))
    # Anything still outside printable ASCII is a rarity we cannot fold; drop it.
    text = "".join(c for c in text if c == "\n" or (" " <= c <= "~"))
    # Tidy whitespace: no trailing spaces, tabs -> space, at most one blank line
    # between paragraphs, and no runs of spaces.
    text = "\n".join(re.sub(r"[ \t]+", " ", line).rstrip() for line in text.split("\n"))
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip("\n")


def build(cache_dir: str) -> str:
    parts = []
    for book_id, title, author in BOOKS:
        raw = fetch(book_id, cache_dir)
        body = normalize(strip_gutenberg_boilerplate(raw, book_id))
        print(f"[book ] {title} -- {author} ({len(body):,} chars)")
        parts.append(body)
    # A blank line between books: the same separator the corpus already uses
    # between paragraphs, so no special boundary token is invented.
    return "\n\n".join(parts) + "\n"


def report(text: str) -> None:
    """Print the corpus statistics the README quotes, so they stay pinned to the
    file actually on disk rather than to remembered numbers.
    """
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    vocab = sorted(set(text))
    print(f"[corpus] {len(text):,} chars | vocab={len(vocab)} | sha256 {digest}")
    print(f"[vocab ] {''.join(vocab)!r}")
    if EXPECTED_SHA256 != "REPLACE_ME" and digest != EXPECTED_SHA256:
        print(f"[warn  ] sha256 differs from the pinned {EXPECTED_SHA256}; an "
              f"upstream text may have been re-released, so the README's measured "
              f"bpc figures no longer describe exactly this file.", file=sys.stderr)
    try:
        from snn_char_lm import corpus_bpc_floors
    except ImportError:
        return
    unigram, bigram = corpus_bpc_floors(text)
    print(f"[floors] unigram {unigram:.2f} bpc | bigram {bigram:.2f} bpc "
          f"(the baselines a real LM must beat)")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default="input.txt", help="corpus file to write")
    p.add_argument("--cache-dir", dest="cache_dir", default=".corpus_cache",
                   help="where downloaded Gutenberg texts are cached")
    p.add_argument("--check", action="store_true",
                   help="do not rebuild; just report stats for an existing --out")
    args = p.parse_args()

    if args.check:
        with open(args.out, "r", encoding="utf-8") as f:
            report(f.read())
        return

    text = build(args.cache_dir)
    with open(args.out, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    print(f"[write ] {args.out}")
    report(text)


if __name__ == "__main__":
    main()
