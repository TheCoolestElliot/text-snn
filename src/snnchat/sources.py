"""Where the conversational training data comes from, and what it is for.

Each source is downloaded once into `data/chat_raw/`, verified by size, and
converted by its own reader into ONE common record type -- a list of
`(role, text)` turns -- which `build_corpus` then renders through
`ChatTokenizer`. Adding a source means adding a reader; nothing downstream
changes.

THE MIX, AND WHY EACH PART IS IN IT
-----------------------------------
The model this feeds has a few million parameters and a memory horizon measured
in tens of characters. That rules out most of what "chat data" usually means:
long multi-turn reasoning is not learnable at this scale, and training on it
mostly teaches the model to produce fluent-looking text that ignores the
question. So the mix is chosen for what a small char-level model can actually
acquire.

  tinystories   Simple, correct, extremely repetitive English. This is the
                grammar teacher. A 1500-word vocabulary and a consistent
                register are exactly what a small model needs to become fluent
                rather than merely plausible, and it is the reason "tell me a
                story" is the single thing this model does best.
  soda          1.5 M short, natural, two-party conversations. This is the
                turn-taking teacher: how a reply relates to what was just said,
                at a length the horizon can actually span.
  alpaca/dolly  Short instruction/response pairs. Teaches the shape of
                answering a question rather than continuing it -- the difference
                between a language model and something that replies.
  oasst1        Real assistant conversations. Small, and the only source with
                genuine assistant register; heavily filtered for length.
  persona       Written here, not downloaded. A few thousand templated exchanges
                covering greetings, identity, capability questions and refusals.
                This is what the user actually types first, and no public
                dataset contains "what are you?" answered by "a spiking neural
                network". Oversampled deliberately (see `build_corpus`).

Every downloaded source is filtered hard: turns longer than `MAX_TURN_CHARS` are
dropped rather than truncated (a truncated turn teaches the model to stop
mid-sentence), and any conversation whose text does not survive normalisation
largely intact is dropped whole.
"""

from __future__ import annotations

import gzip
import json
import os
import random
import re
import urllib.request
from collections.abc import Iterator
from dataclasses import dataclass

from snnchat.tokenizer import normalise_text

__all__ = ["SOURCES", "Source", "download_all", "read_source", "iter_conversations"]

#: A turn longer than this is dropped, not truncated. At a horizon of tens of
#: characters a 2000-character monologue is noise the model cannot use and
#: cannot be taught to stop producing.
#:
#: **What 400 costs, measured 2026-08-11 where `BUILD_NOTES.md` §3 called it a
#: guess**: it discards 56.4 % of alpaca's characters, 85.1 % of dolly's and
#: 78.3 % of oasst1's. Raising it to 600 recovers alpaca +62 %, dolly +67 % and
#: oasst1 +108 %, and oasst1 is the source `build_corpus` already worries about
#: memorising, at 796 conversations and 40.6 passes.
#:
#: The stated reason for 400 argues against TRUNCATING a turn, which teaches the
#: model to stop mid-sentence. It is not an argument for this particular DROP
#: threshold, and the two were never separated.
#:
#: `set_max_turn_chars` exists so a repack can change it without editing source.
#: The default is untouched, because every packed `.bin` under `data/chat/` and
#: every committed checkpoint was built at 400.
MAX_TURN_CHARS = 400


def set_max_turn_chars(value: int) -> int:
    """Set the drop threshold for a repack. Returns the previous value.

    A module-level rebind rather than a parameter threaded through nine readers:
    the readers are generators consumed by `build_corpus`, and a corpus is
    packed under exactly one threshold, so a global for the duration of a pack
    is the honest shape. `build_corpus` records the value it used in the
    manifest, which is what makes a `.bin` traceable to its filter.
    """
    global MAX_TURN_CHARS
    previous = MAX_TURN_CHARS
    MAX_TURN_CHARS = int(value)
    return previous

#: A conversation is dropped whole if normalisation removed more than this
#: fraction of it -- that is the signal for "this record was mostly emoji,
#: markup, code or another script", none of which this alphabet can represent.
MAX_LOSS_FRACTION = 0.02


@dataclass(frozen=True)
class Source:
    """One downloadable corpus."""

    name: str
    url: str
    filename: str
    kind: str          # "text" | "jsonl" | "jsonl.gz" | "parquet"
    approx_bytes: int
    note: str
    #: Only the first `max_bytes` are fetched, via an HTTP Range request, when
    #: the full file is larger than we need. 0 = fetch it all.
    max_bytes: int = 0


SOURCES: tuple[Source, ...] = (
    Source(
        name="tinystories",
        url="https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt",
        filename="tinystories.txt",
        kind="text",
        approx_bytes=2_227_753_162,
        max_bytes=900_000_000,
        note="simple English; the grammar teacher",
    ),
    Source(
        name="soda",
        url="https://huggingface.co/datasets/allenai/soda/resolve/main/train.parquet",
        filename="soda_train.parquet",
        kind="parquet",
        approx_bytes=688_800_000,
        note="1.5M short two-party dialogues; the turn-taking teacher",
    ),
    Source(
        # The SAME parquet as `soda`, read through a different column. It gets
        # its own entry so it can be packed, weighted and reported separately,
        # and `download_all` skips it because the file is already there.
        name="soda_narrative",
        url="https://huggingface.co/datasets/allenai/soda/resolve/main/train.parquet",
        filename="soda_train.parquet",
        kind="parquet",
        approx_bytes=688_800_000,
        note="~218M chars of adult third-person prose; the register teacher",
    ),
    Source(
        name="alpaca",
        url="https://huggingface.co/datasets/tatsu-lab/alpaca/resolve/main/data/train-00000-of-00001-a09b74b3ef9c3b56.parquet",
        filename="alpaca.parquet",
        kind="parquet",
        approx_bytes=24_246_638,
        note="52k short instruction/response pairs",
    ),
    Source(
        name="dolly",
        url="https://huggingface.co/datasets/databricks/databricks-dolly-15k/resolve/main/databricks-dolly-15k.jsonl",
        filename="dolly.jsonl",
        kind="jsonl",
        approx_bytes=13_085_339,
        note="15k human-written instruction/response pairs",
    ),
    Source(
        name="oasst1",
        url="https://huggingface.co/datasets/OpenAssistant/oasst1/resolve/main/2023-04-12_oasst_ready.messages.jsonl.gz",
        filename="oasst1.jsonl.gz",
        kind="jsonl.gz",
        approx_bytes=34_200_000,
        note="real assistant conversations; the register teacher",
    ),
)

_BY_NAME = {s.name: s for s in SOURCES}


# --------------------------------------------------------------------------
# download
# --------------------------------------------------------------------------


def _download(src: Source, dest: str, *, chunk: int = 1 << 22) -> None:
    """Fetch `src` to `dest`, resuming a partial file if one is there.

    Resume is not a nicety here: `tinystories` is a 900 MB range request over a
    link that may be interrupted, and re-fetching from zero after 800 MB is an
    hour of a ten-hour budget. The partial file is written as `.part` and only
    renamed once complete, so an interrupted run never leaves a truncated file
    that looks finished.
    """
    want = src.max_bytes or src.approx_bytes
    part = dest + ".part"
    have = os.path.getsize(part) if os.path.exists(part) else 0
    if have >= want:
        os.replace(part, dest)
        return

    headers = {"User-Agent": "snn-chat/0.1"}
    # A Range request serves double duty: it resumes, and for tinystories it is
    # how we take a 900 MB prefix of a 2.2 GB file instead of the whole thing.
    end = want - 1
    headers["Range"] = f"bytes={have}-{end}"

    req = urllib.request.Request(src.url, headers=headers)
    mode = "ab" if have else "wb"
    with urllib.request.urlopen(req, timeout=120) as resp, open(part, mode) as out:
        got = have
        next_report = have + (1 << 26)
        while True:
            block = resp.read(chunk)
            if not block:
                break
            out.write(block)
            got += len(block)
            if got >= next_report:
                print(f"    {src.name}: {got / 1e6:,.0f} MB", flush=True)
                next_report = got + (1 << 26)
    os.replace(part, dest)


def download_all(data_dir: str = "data/chat_raw", only: list[str] | None = None) -> dict[str, str]:
    """Ensure every source is present locally. Idempotent; returns name -> path."""
    os.makedirs(data_dir, exist_ok=True)
    paths: dict[str, str] = {}
    for src in SOURCES:
        if only and src.name not in only:
            continue
        dest = os.path.join(data_dir, src.filename)
        if os.path.exists(dest):
            print(f"  {src.name}: present ({os.path.getsize(dest) / 1e6:,.0f} MB)", flush=True)
            paths[src.name] = dest
            continue
        want = src.max_bytes or src.approx_bytes
        print(f"  {src.name}: fetching ~{want / 1e6:,.0f} MB -- {src.note}", flush=True)
        _download(src, dest)
        paths[src.name] = dest
        print(f"  {src.name}: done ({os.path.getsize(dest) / 1e6:,.0f} MB)", flush=True)
    return paths


# --------------------------------------------------------------------------
# readers: each yields conversations as [(role, text), ...]
# --------------------------------------------------------------------------

Conversation = list[tuple[str, str]]

_WS = re.compile(r"[ \t]+")
_NL = re.compile(r"\n{3,}")


def _clean(text: str) -> str:
    """Normalise, collapse runs of whitespace, strip.

    The whitespace collapse is not cosmetic. A char-level model spends a real
    fraction of its capacity on formatting, and a corpus that mixes single and
    double spacing after a full stop teaches it to be uncertain about a
    character that carries no information.
    """
    text = normalise_text(text)
    text = _WS.sub(" ", text)
    text = _NL.sub("\n\n", text)
    return text.strip()


def _acceptable(raw: str, cleaned: str) -> bool:
    """Did this text survive normalisation, and is it a usable length?"""
    if not cleaned or len(cleaned) > MAX_TURN_CHARS:
        return False
    if not raw:
        return False
    lost = 1.0 - (len(cleaned) / max(len(raw), 1))
    return lost <= MAX_LOSS_FRACTION or len(raw) - len(cleaned) <= 2


def _pair(instruction: str, context: str, response: str) -> Conversation | None:
    """Instruction-style records -> one two-turn conversation."""
    q = _clean(instruction)
    ctx = _clean(context) if context else ""
    a = _clean(response)
    if ctx:
        # Context is folded into the user turn rather than given its own role:
        # a third role marker would be a token the model sees rarely and must
        # still learn, and at this scale that trade never pays.
        q = f"{q}\n{ctx}"
    if not _acceptable(instruction, q) or not _acceptable(response, a):
        return None
    return [("user", q), ("bot", a)]


def _read_jsonl(path: str) -> Iterator[dict]:
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def _read_parquet_columns(path: str, columns: list[str]) -> Iterator[dict]:
    """Stream a parquet file row-group by row-group.

    Row-group streaming rather than `read_table`: SODA's train split is 689 MB
    on disk and several GB in memory as Python objects, and this process also
    has to hold a corpus buffer.
    """
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(path)
    available = set(pf.schema_arrow.names)
    cols = [c for c in columns if c in available]
    if not cols:
        raise KeyError(f"{path}: none of {columns} in {sorted(available)}")
    for batch in pf.iter_batches(batch_size=2048, columns=cols):
        for row in batch.to_pylist():
            yield row


def read_tinystories(path: str) -> Iterator[Conversation]:
    """TinyStories is one story per record, separated by `<|endoftext|>`.

    Stories become `("story", text)` single-turn records rather than
    user/bot pairs. `build_corpus` decides how to present them -- most are
    packed as raw narrative (so the model learns plain English), a fraction are
    wrapped in a "tell me a story" exchange (so the capability is reachable from
    the chat format). Doing that split here would hard-code a mix ratio into a
    reader.
    """
    buf: list[str] = []
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if "<|endoftext|>" in line:
                head, _, tail = line.partition("<|endoftext|>")
                buf.append(head)
                story = _clean("".join(buf))
                buf = [tail]
                if 80 <= len(story) <= 1200:
                    yield [("story", story)]
            else:
                buf.append(line)
    story = _clean("".join(buf))
    if 80 <= len(story) <= 1200:
        yield [("story", story)]


def read_soda_narrative(path: str) -> Iterator[Conversation]:
    """SODA's `narrative` column: ~218 M characters of adult third-person prose.

    It sits in the parquet this repository has already downloaded and no code
    has ever read it. That matters because of what the corpus is made of: every
    prose source in the mixture except oasst1 (0.3 M) and alpaca (8 M) is
    TinyStories, wrapped four different ways. The model's register, its
    vocabulary and its idea of what a sentence is all come from text written for
    three-year-olds, which is a plausible part of why `QUALITY_v8.md` §4 finds
    the held-out oracle at 0.09 -- it rarely drafts a reply about an unfamiliar
    noun because it has rarely seen unfamiliar nouns.

    Measured on row group 0 (1,191,582 rows): mean 182.5 characters and 99.89 %
    under the 400-character filter, so almost nothing is dropped.

    Emitted as bare narrative rather than wrapped in a request. It is a register
    and vocabulary teacher, and inventing a request frame for it would make it a
    second, unvalidated topic source instead.

    The `_raw` pseudo-role is what `build_corpus._RawAwareTokenizer` renders as
    bare text with no turn markers. `("story", ...)` -- which `read_tinystories`
    uses -- is NOT interchangeable with it: that role is only ever consumed by
    `_tinystories_conversations`, which wraps it into a real user/bot exchange
    before it reaches the tokenizer, and passing it straight through raises
    `role must be 'user' or 'bot'`.
    """
    for row in _read_parquet_columns(path, ["narrative"]):
        text = _clean(row.get("narrative") or "")
        if not _acceptable(row.get("narrative") or "", text):
            continue
        yield [("_raw", text)]


def read_soda(path: str) -> Iterator[Conversation]:
    """SODA rows carry a `dialogue` list and a `speakers` list.

    Speaker *names* are discarded and turns are assigned to alternating roles.
    That is a real simplification and it is deliberate: SODA's speakers are named
    characters ("Ryan", "Mia"), and keeping the names would teach the model that
    a reply begins with a proper noun it has no way to know.
    """
    for row in _read_parquet_columns(path, ["dialogue", "speakers"]):
        dialogue = row.get("dialogue") or []
        if len(dialogue) < 2:
            continue
        turns: Conversation = []
        ok = True
        for i, utt in enumerate(dialogue[:8]):
            text = _clean(utt or "")
            if not _acceptable(utt or "", text):
                ok = False
                break
            turns.append(("user" if i % 2 == 0 else "bot", text))
        if ok and len(turns) >= 2:
            # A conversation must end on the model's turn, or the last thing it
            # learns from this record is how to stop after a human speaks.
            if turns[-1][0] == "user":
                turns.pop()
            if len(turns) >= 2:
                yield turns


def read_alpaca(path: str) -> Iterator[Conversation]:
    for row in _read_parquet_columns(path, ["instruction", "input", "output"]):
        conv = _pair(row.get("instruction") or "", row.get("input") or "", row.get("output") or "")
        if conv:
            yield conv


def read_dolly(path: str) -> Iterator[Conversation]:
    for row in _read_jsonl(path):
        conv = _pair(
            row.get("instruction") or "", row.get("context") or "", row.get("response") or ""
        )
        if conv:
            yield conv


def read_oasst1(path: str) -> Iterator[Conversation]:
    """OASST1 is a message forest; reassemble the highest-ranked English paths.

    Only `lang == "en"` and only `rank == 0` replies are kept -- rank 0 is the
    reply human labellers preferred, and taking all of them would train equally
    on answers annotators rejected.
    """
    by_id: dict[str, dict] = {}
    children: dict[str, list[str]] = {}
    for msg in _read_jsonl(path):
        if msg.get("lang") != "en":
            continue
        mid = msg.get("message_id")
        if not mid:
            continue
        by_id[mid] = msg
        parent = msg.get("parent_id")
        if parent:
            children.setdefault(parent, []).append(mid)

    def best_child(mid: str) -> str | None:
        kids = children.get(mid, [])
        ranked = [k for k in kids if by_id.get(k, {}).get("rank") in (0, None)]
        return (ranked or kids or [None])[0]

    for mid, msg in by_id.items():
        if msg.get("parent_id"):
            continue  # not a root
        turns: Conversation = []
        cur: str | None = mid
        while cur is not None and len(turns) < 6:
            m = by_id[cur]
            role = "user" if m.get("role") == "prompter" else "bot"
            text = _clean(m.get("text") or "")
            if not _acceptable(m.get("text") or "", text):
                break
            turns.append((role, text))
            cur = best_child(cur)
        if turns and turns[-1][0] == "user":
            turns.pop()
        if len(turns) >= 2:
            yield turns


_READERS = {
    "tinystories": read_tinystories,
    "soda": read_soda,
    "soda_narrative": read_soda_narrative,
    "alpaca": read_alpaca,
    "dolly": read_dolly,
    "oasst1": read_oasst1,
}


def read_source(name: str, path: str) -> Iterator[Conversation]:
    try:
        reader = _READERS[name]
    except KeyError:
        raise KeyError(f"no reader for source {name!r}; known: {sorted(_READERS)}") from None
    return reader(path)


def iter_conversations(
    paths: dict[str, str], *, shuffle_seed: int | None = None
) -> Iterator[tuple[str, Conversation]]:
    """Every conversation from every downloaded source, tagged with its source."""
    order = list(paths)
    if shuffle_seed is not None:
        random.Random(shuffle_seed).shuffle(order)
    for name in order:
        for conv in read_source(name, paths[name]):
            yield name, conv
