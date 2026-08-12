"""Render every source into packed id streams, one file per source.

WHY ONE FILE PER SOURCE AND NOT ONE MIXED CORPUS
------------------------------------------------
The obvious design writes a single shuffled stream in the intended mix ratios.
It is the wrong one here for two reasons:

1.  The ratios are the most uncertain thing in this whole build. A single stream
    bakes them into 2 GB of disk, so changing "how much persona data?" from 10 %
    to 6 % means repacking everything. With one file per source the mix is three
    numbers in a config, changed between runs at no cost.
2.  A mixed stream can only express a ratio by *duplication*. Persona data is
    ~2 MB and wants to be ~8 % of a ~2 GB corpus, which is 80 copies of the same
    2 MB written to disk. Sampling with weights costs nothing and expresses the
    same distribution exactly.

`MixtureSampler` (in `snnchat.data`) draws each window's source from the weights,
so the mix is realised per batch element rather than per file.

WHAT A "CHARACTER" COSTS
------------------------
Ids are packed as `uint8`, which the 102-symbol vocabulary fits with room to
spare, so the on-disk size is the character count exactly and a memmap is a
pure slice with no decode step -- the same property `snn.data` engineers for and
for the same reason.

TRAIN/VAL
---------
The last `VAL_FRACTION` of each source is held out. Held out by *position*, not
by sampling: a random hold-out at conversation granularity would put near
duplicates of held-out conversations in train (SODA and the persona templates
both contain many near-duplicates), and the val number would then measure
memorisation. A contiguous tail is not immune to that either, but it is honest
about being a weak check -- and val bpc here is a training-progress signal, not
a reported figure.
"""

from __future__ import annotations

import json
import os
import random
import re
import time
from dataclasses import dataclass

import numpy as np

from snnchat.persona import build_persona_conversations
from snnchat.sources import SOURCES, iter_conversations, read_source
from snnchat.tokenizer import BOS, EOT, VOCAB_VERSION, ChatTokenizer

__all__ = ["build_all", "CorpusManifest", "VAL_FRACTION", "DEFAULT_MIX"]

#: Fraction of each source held out at its tail.
VAL_FRACTION = 0.004

#: Default mixture weights, by fraction of training characters drawn. These are
#: a judgement, not a measurement; see the module docstring of
#: `snnchat.sources` for what each source is meant to teach.
#:
#: `persona` at 0.10 is the aggressive one and is deliberate: it is ~2 MB of
#: text against ~1.5 GB, so at this weight the model sees it hundreds of times
#: and will effectively memorise the greeting and identity answers. For those
#: twenty-odd intents memorisation is the desired behaviour -- "what are you?"
#: has one correct answer here and it is stipulated in `snnchat.persona`. The
#: cost is that the same weight pulls the model's general register toward short
#: apologetic sentences, which is visible in samples and is the first knob to
#: turn down if replies feel canned.
#: `oasst1` is at 0.02 rather than the 0.06 the other instruction sources get,
#: and the reason is a measurement rather than a preference: the length filter in
#: `snnchat.sources` (400 characters per turn, dropped not truncated) leaves only
#: 796 of its conversations, 0.3 MB. At 6 % of a ~2 G-character run that is 400
#: passes over 796 examples, which is not training on assistant register, it is
#: memorising 796 answers. 0.02 keeps the register signal at a repetition count
#: closer to the rest of the mix.
#: UPDATED 2026-08-11 to the shipped recipe. It previously named the `chat-v1`
#: mixture, which predates `stories_topic` entirely -- so any run launched
#: without an explicit `--mix` silently trained on a corpus with no
#: subject-conditioned story source in it at all, which is the single change
#: `docs/chat/QUALITY.md` attributes the responsiveness gain to. The values below
#: are `experiments/chat/chat-v3d-aligned`'s own `chat_config["mix"]`, read out
#: of the shipped checkpoint.
DEFAULT_MIX: dict[str, float] = {
    "stories_topic": 0.40,
    "soda": 0.20,
    "alpaca": 0.15,
    "tinystories": 0.09,
    "persona": 0.08,
    "dolly": 0.06,
    "oasst1": 0.02,
}

#: Prompts that a TinyStories narrative is presented as the answer to, so that
#: the capability is reachable from the chat format instead of only from raw
#: narrative continuation.
_STORY_PROMPTS = [
    "tell me a story", "tell me a short story", "can you tell me a story",
    "i want a story", "story please", "make up a story", "tell me a tale",
    "read me a story", "tell me a bedtime story",
]
_STORY_PROMPTS_ABOUT = [
    "tell me a story about {}", "tell me a story about {}, please",
    "can you tell me a story about {}", "i want a story about {}",
    "make up a story about {}", "a story about {}?",
    "write me a little story about {}",
]

#: A capitalised word that is not sentence-initial is almost always a character
#: name in TinyStories, which is what makes topic-conditioned prompts possible
#: without any labels.
_MIDSENTENCE_CAP = re.compile(r"(?<=[a-z,] )([A-Z][a-z]{2,11})\b")


@dataclass
class CorpusManifest:
    """What was packed, from what, and under which vocabulary."""

    vocab_version: int
    vocab_size: int
    built_at: str
    sources: dict[str, dict]

    def to_json(self) -> str:
        return json.dumps(
            {
                "vocab_version": self.vocab_version,
                "vocab_size": self.vocab_size,
                "built_at": self.built_at,
                "sources": self.sources,
            },
            indent=2,
            sort_keys=True,
        )


class _Writer:
    """Buffered uint8 writer with a running character count."""

    def __init__(self, path: str, buf_chars: int = 1 << 24):
        self.path = path
        self._fh = open(path, "wb")
        self._buf: list[np.ndarray] = []
        self._buffered = 0
        self._limit = buf_chars
        self.total = 0

    def write(self, ids: list[int] | np.ndarray) -> None:
        arr = np.asarray(ids, dtype=np.uint8)
        self._buf.append(arr)
        self._buffered += arr.size
        self.total += arr.size
        if self._buffered >= self._limit:
            self.flush()

    def flush(self) -> None:
        if self._buf:
            self._fh.write(np.concatenate(self._buf).tobytes())
            self._buf.clear()
            self._buffered = 0

    def close(self) -> None:
        self.flush()
        self._fh.close()


def _story_conversation(story: str, rng: random.Random) -> list[tuple[str, str]]:
    """Wrap a narrative in a request it is a plausible answer to.

    Topic conditioning comes free from the corpus's own style: `_MIDSENTENCE_CAP`
    finds the character's name, and asking for "a story about Lily" whose answer
    genuinely is about Lily is the cheapest possible lesson in "the reply should
    be about the thing that was asked for".
    """
    names = _MIDSENTENCE_CAP.findall(story[:200])
    if names and rng.random() < 0.75:
        prompt = rng.choice(_STORY_PROMPTS_ABOUT).format(names[0])
    else:
        prompt = rng.choice(_STORY_PROMPTS)
    return [("user", prompt), ("bot", story)]


def _pack_source(
    name: str,
    conversations,
    out_dir: str,
    tok: ChatTokenizer,
    *,
    limit_chars: int = 0,
    log_every: float = 20.0,
) -> dict:
    """Render `conversations` to `<out_dir>/<name>.bin` and `<name>.val.bin`.

    Written in one streaming pass into a train file, then the tail is moved into
    the val file. Streaming matters: SODA alone is ~1.5 M conversations and this
    process must not hold them.
    """
    tmp_path = os.path.join(out_dir, f"{name}.all.bin")
    writer = _Writer(tmp_path)
    n_conv = 0
    t0 = time.perf_counter()
    next_log = t0 + log_every
    for conv in conversations:
        ids = tok.render_conversation(conv)
        writer.write(ids)
        n_conv += 1
        if limit_chars and writer.total >= limit_chars:
            break
        now = time.perf_counter()
        if now >= next_log:
            print(
                f"    {name}: {n_conv:,} conversations, {writer.total / 1e6:,.1f} M chars",
                flush=True,
            )
            next_log = now + log_every
    writer.close()

    total = writer.total
    if total < 1024:
        raise RuntimeError(f"{name}: packed only {total} characters; a reader is broken")

    n_val = max(4096, int(total * VAL_FRACTION))
    n_val = min(n_val, total // 4)
    n_train = total - n_val

    train_path = os.path.join(out_dir, f"{name}.bin")
    val_path = os.path.join(out_dir, f"{name}.val.bin")
    with open(tmp_path, "rb") as src:
        with open(train_path, "wb") as out:
            remaining = n_train
            while remaining:
                block = src.read(min(remaining, 1 << 24))
                if not block:
                    break
                remaining -= len(block)
                out.write(block)
        with open(val_path, "wb") as out:
            while True:
                block = src.read(1 << 24)
                if not block:
                    break
                out.write(block)
    os.remove(tmp_path)

    elapsed = time.perf_counter() - t0
    print(
        f"  {name}: {n_conv:,} conversations, {total / 1e6:,.1f} M chars "
        f"({n_train / 1e6:,.1f} train / {n_val / 1e6:,.2f} val) in {elapsed:,.0f}s",
        flush=True,
    )
    return {
        "conversations": n_conv,
        "chars_total": total,
        "chars_train": n_train,
        "chars_val": n_val,
        "seconds": round(elapsed, 1),
    }


def _tinystories_conversations(path: str, rng: random.Random):
    """Half raw narrative, half wrapped as a story request.

    Raw narrative is what actually teaches English -- an unbroken 400-character
    story is 400 characters of grammar with no format overhead. The wrapped half
    is what makes the capability reachable from a chat prompt. Neither alone is
    enough: all-raw gives a model that writes stories but cannot be asked for
    one, all-wrapped spends a third of the corpus on markers and request
    phrasings.
    """
    for conv in read_source("tinystories", path):
        story = conv[0][1]
        if rng.random() < 0.5:
            yield _story_conversation(story, rng)
        else:
            # Emitted as a bare narrative: BOS, text, EOT, no role markers. The
            # model needs to see long stretches of plain prose that are not
            # anybody's turn, or every long-range dependency it learns is
            # entangled with the turn structure.
            yield [("_raw", story)]


def _render_raw(tok: ChatTokenizer, text: str) -> list[int]:
    return [BOS, *tok.encode(text).tolist(), EOT]


class _RawAwareTokenizer(ChatTokenizer):
    """`render_conversation` that also understands the `_raw` pseudo-role.

    A subclass rather than a branch inside `ChatTokenizer`, because "raw
    narrative" is a property of how this corpus is built, not of the vocabulary,
    and the tokenizer is the thing every checkpoint depends on.
    """

    def render_conversation(self, turns, *, bos: bool = True) -> list[int]:
        if len(turns) == 1 and turns[0][0] == "_raw":
            return _render_raw(self, turns[0][1])
        return super().render_conversation(turns, bos=bos)


def build_all(
    raw_dir: str = "data/chat_raw",
    out_dir: str = "data/chat",
    *,
    seed: int = 0,
    persona_conversations: int = 60_000,
    limit_chars: dict[str, int] | None = None,
) -> CorpusManifest:
    """Pack every available source. Idempotent per source: an existing
    `<name>.bin` is left alone, so a failed source can be rebuilt on its own."""
    os.makedirs(out_dir, exist_ok=True)
    tok = _RawAwareTokenizer()
    rng = random.Random(seed)
    limit_chars = limit_chars or {}
    stats: dict[str, dict] = {}

    print(f"packing into {out_dir} (vocab v{VOCAB_VERSION}, V={tok.vocab_size})", flush=True)

    for src in SOURCES:
        raw_path = os.path.join(raw_dir, src.filename)
        if not os.path.exists(raw_path):
            print(f"  {src.name}: SKIPPED (not downloaded)", flush=True)
            continue
        if os.path.exists(os.path.join(out_dir, f"{src.name}.bin")):
            size = os.path.getsize(os.path.join(out_dir, f"{src.name}.bin"))
            print(f"  {src.name}: already packed ({size / 1e6:,.1f} M chars)", flush=True)
            stats[src.name] = {"chars_train": size, "reused": True}
            continue
        if src.name == "tinystories":
            convs = _tinystories_conversations(raw_path, rng)
        else:
            convs = read_source(src.name, raw_path)
        stats[src.name] = _pack_source(
            src.name, convs, out_dir, tok, limit_chars=limit_chars.get(src.name, 0)
        )

    if not os.path.exists(os.path.join(out_dir, "persona.bin")):
        stats["persona"] = _pack_source(
            "persona",
            build_persona_conversations(persona_conversations, seed=seed),
            out_dir,
            tok,
        )
    else:
        size = os.path.getsize(os.path.join(out_dir, "persona.bin"))
        print(f"  persona: already packed ({size / 1e6:,.1f} M chars)", flush=True)
        stats["persona"] = {"chars_train": size, "reused": True}

    manifest = CorpusManifest(
        vocab_version=VOCAB_VERSION,
        vocab_size=tok.vocab_size,
        built_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
        sources=stats,
    )
    with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8") as fh:
        fh.write(manifest.to_json())
        fh.write("\n")
    return manifest


# Re-exported so a caller can build a mixed iterator without importing three
# modules; unused by `build_all` itself.
__all__.append("iter_conversations")
_ = iter_conversations, EOT
