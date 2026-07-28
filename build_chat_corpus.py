"""
build_chat_corpus.py
====================

Build the Tier 7 *modern conversational* corpus for the chat-capable SNN, from
permissively licensed datasets, reproducibly.

The goal is a model that writes plain modern English in a helpful-assistant
register and can answer prompts -- so the data is (a) simple, grammatical,
modern prose and (b) lots of short conversations framed with the chat role
markers defined in ``snn_char_lm.py``:

    \\x01 user text \\x02 assistant text \\x03      (concatenated per exchange)

Sources (downloaded once into ``--cache-dir``; see the "Downloads" note below):

  * **TinyStoriesV2** (roneneldan/TinyStories, CDLA-Sharing-1.0) -- GPT-4-written
    stories with a deliberately small vocabulary and near-perfect grammar. The
    TinyStories result is that sub-10M-param models reach fluent grammar ONLY on
    this kind of low-diversity text, so it is the grammar backbone. Stories stay
    UNFRAMED (no role markers): the assistant marker must only ever precede
    assistant-register text.
  * **SODA** (allenai/soda, CC-BY-4.0) -- 1.19M short everyday two-person
    dialogues. Short exchanges are what teach *conditioning* (a reply that
    depends on the previous turn) within a truncated-BPTT gradient window.
  * **smol-smoltalk** (HuggingFaceTB, Apache-2.0) -- assistant-register
    conversations curated for small models; filtered here to short, plain-prose
    exchanges (no code/markdown/URLs).
  * **everyday-conversations-llama3.1-2k** (HuggingFaceTB, Apache-2.0) -- small
    but exactly the target register: simple multi-turn assistant chats.
  * A handful of handwritten identity dialogues (the model is "Spark", a small
    spiking neural network built by Elliot) -- a light sprinkle in pretrain and
    a capped share of the fine-tune corpus.

Outputs (in ``--out-dir``, default ``corpus/``):

    pretrain_train.txt / pretrain_val.txt     ~150 MB / ~3 MB
    finetune_train.txt / finetune_val.txt     ~27 MB / ~1.3 MB
    conditioning_val.jsonl                    (user, assistant) pairs for the
                                              conditioning-gain metric
    manifest.json                             per-source stats + sha256s

Validation docs are disjoint from training docs by construction (a stable hash
of the document text routes ~2%% of docs to the val pool before budgets fill).
Every output is checked against the FIXED_VOCAB of ``snn_char_lm.py``.

Downloads (not automated here; ~3.8 GB total, cached):
    huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt
    huggingface.co/datasets/HuggingFaceTB/smol-smoltalk/resolve/main/data/train-0000{0..3}-of-00004.parquet
    huggingface.co/datasets/allenai/soda/resolve/main/train.parquet
    huggingface.co/datasets/HuggingFaceTB/everyday-conversations-llama3.1-2k/resolve/main/data/train_sft-00000-of-00001.parquet
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
from typing import Iterable, Iterator, List, Optional, Tuple

from build_corpus import normalize
from snn_char_lm import FIXED_VOCAB, ROLE_ASSISTANT, ROLE_END, ROLE_USER

MB = 1_000_000

# ---------------------------------------------------------------------------
# Budgets (bytes of finished text). Sized for a 5-9M-parameter char model at
# 20-40 chars/param with <=4 repeat epochs -- see CAMPAIGN.md.
# ---------------------------------------------------------------------------
PRETRAIN_BUDGETS = {           # ~47% unframed stories / ~53% framed dialogue
    # Sized by the MEASURED main-run throughput (CAMPAIGN.md): h1536/T3 with the
    # CUDA-graph step sustains ~37K chars/s, so ~6 epochs of ~650 MB fills the
    # multi-day budget; fresh text up to ~4 epochs beats many epochs of less.
    "stories":   (300 * MB, int(4.0 * MB)),  # (train, val)
    "soda":      (300 * MB, int(4.0 * MB)),
    "smoltalk":  (35 * MB, int(0.8 * MB)),
    "ultrachat": (25 * MB, int(0.5 * MB)),
    "everyday":  (int(1.3 * MB), int(0.1 * MB)),
    "identity":  (int(1.5 * MB), 0),         # sprinkle; never in val
}
FINETUNE_BUDGETS = {           # assistant-register sharpening + story replay
    # Round-2 shares (the round-1 transcripts diagnosed the failure modes):
    # story replay cut 33%->14% (story-register bleed into replies), identity
    # cut 3.8%->1.6% (verbatim parroting on unrelated prompts), dialogue up.
    "smoltalk":  (int(4.5 * MB), int(0.3 * MB)),
    "soda":      (9 * MB, int(0.5 * MB)),
    "ultrachat": (2 * MB, int(0.1 * MB)),
    "everyday":  (int(1.3 * MB), int(0.1 * MB)),
    "stories":   (3 * MB, int(0.2 * MB)),
    "identity":  (int(0.35 * MB), 0),
}

# Stricter turn-length limits for the fine-tune stage: short exchanges are the
# ones a truncated gradient window can actually learn conditioning from.
PRETRAIN_MAX_USER, PRETRAIN_MAX_ASSIST = 300, 800
# Finetune caps tightened in round 2 (long replies wander off-register; short
# crisp replies terminate cleanly and stay conversational).
FINETUNE_MAX_USER, FINETUNE_MAX_ASSIST = 200, 350
MAX_EXCHANGES_PER_DIALOGUE = 4

# Reject any conversation whose text contains these: code, markup, tables and
# URLs are noise for a tiny plain-prose model (and half of them are characters
# the corpus should keep rare).
_REJECT_SUBSTRINGS = ("```", "\t", "$$", "\\(", "\\[", "###",
                      "http://", "https://", "www.")
_REJECT_CHARS = set("|{}<>`~^\\")

_ALLOWED = set(FIXED_VOCAB)


def _clean_turn(text: str) -> Optional[str]:
    """Normalize one conversation turn to plain ASCII prose; None = reject.

    Rejects (rather than silently truncates) anything that loses >2% of its
    characters to the ASCII fold -- that is the signature of non-English text --
    and anything carrying code/markup markers. normalize() itself strips the
    typography (curly quotes etc.) and anything outside newline+printable-ASCII.
    """
    for bad in _REJECT_SUBSTRINGS:
        if bad in text:
            return None
    cleaned = normalize(text)
    if not cleaned:
        return None
    if len(cleaned) < 0.98 * len(text) - 2:
        return None                       # dropped too much: likely non-English
    if _REJECT_CHARS & set(cleaned):
        return None
    # Turns are single paragraphs of prose; fold internal newlines to spaces
    # so the newline character keeps one meaning (paragraph/doc rhythm) in the
    # dialogue part of the corpus.
    cleaned = " ".join(cleaned.split("\n")).strip()
    return cleaned or None


def _frame(exchanges: List[Tuple[str, str]]) -> str:
    """Render (user, assistant) pairs into the role-marker chat format."""
    return "".join(ROLE_USER + u + ROLE_ASSISTANT + a + ROLE_END
                   for u, a in exchanges)


def _is_val_doc(doc: str) -> bool:
    """Stable ~2% routing of documents to the validation pool (train/val
    disjointness that survives re-runs and budget changes)."""
    return hashlib.md5(doc.encode("utf-8")).digest()[0] < 5


class Bucket:
    """Accumulates accepted documents for one (source, split) pair."""

    def __init__(self, budget: int):
        self.budget = budget
        self.docs: List[str] = []
        self.size = 0

    @property
    def full(self) -> bool:
        return self.size >= self.budget

    def add(self, doc: str) -> None:
        if not self.full:
            self.docs.append(doc)
            self.size += len(doc) + 2          # +2 for the \n\n joiner


# ---------------------------------------------------------------------------
# Source readers -- each yields finished document strings (already framed).
# ---------------------------------------------------------------------------
def stories(cache: str) -> Iterator[str]:
    """Stream TinyStoriesV2 stories (split on <|endoftext|>), normalized."""
    path = os.path.join(cache, "TinyStoriesV2-GPT4-train.txt")
    sep = "<|endoftext|>"
    buf = ""
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        while True:
            chunk = f.read(8 * MB)
            if not chunk:
                break
            buf += chunk
            parts = buf.split(sep)
            buf = parts.pop()                  # carry the partial story
            for raw in parts:
                doc = _story_doc(raw)
                if doc:
                    yield doc
    doc = _story_doc(buf)
    if doc:
        yield doc


def _story_doc(raw: str) -> Optional[str]:
    raw = raw.strip()
    if not (150 <= len(raw) <= 2500):
        return None
    for bad in _REJECT_SUBSTRINGS:
        if bad in raw:
            return None
    doc = normalize(raw)
    if not doc or len(doc) < 0.97 * len(raw) - 2:
        return None
    if _REJECT_CHARS & set(doc):
        return None
    return doc


def _messages_to_exchanges(messages, max_user: int, max_assist: int
                           ) -> Optional[List[Tuple[str, str]]]:
    """Convert a [{role, content}, ...] conversation to clean (user, assistant)
    pairs; None rejects the whole conversation (any dirty/oversized/odd turn)."""
    if not messages or any(m["role"] not in ("user", "assistant")
                           for m in messages):
        return None                            # system prompts etc.: skip
    if [m["role"] for m in messages[:2]] != ["user", "assistant"]:
        return None
    exchanges: List[Tuple[str, str]] = []
    for i in range(0, len(messages) - 1, 2):
        u_msg, a_msg = messages[i], messages[i + 1]
        if u_msg["role"] != "user" or a_msg["role"] != "assistant":
            return None                        # roles must strictly alternate
        u, a = _clean_turn(u_msg["content"]), _clean_turn(a_msg["content"])
        if u is None or a is None:
            return None
        if not (1 <= len(u) <= max_user and 1 <= len(a) <= max_assist):
            return None
        exchanges.append((u, a))
        if len(exchanges) >= MAX_EXCHANGES_PER_DIALOGUE:
            break
    return exchanges or None


def _parquet_rows(path: str, columns: List[str]) -> Iterator[dict]:
    import pyarrow.parquet as pq
    f = pq.ParquetFile(path)
    for batch in f.iter_batches(batch_size=2048, columns=columns):
        yield from batch.to_pylist()


def smoltalk(cache: str, max_user: int, max_assist: int) -> Iterator[str]:
    for shard in range(4):
        path = os.path.join(
            cache, f"smol-smoltalk-train-0000{shard}-of-00004.parquet")
        for row in _parquet_rows(path, ["messages"]):
            ex = _messages_to_exchanges(row["messages"], max_user, max_assist)
            if ex:
                yield _frame(ex)


def everyday(cache: str, max_user: int, max_assist: int) -> Iterator[str]:
    path = os.path.join(cache, "everyday-conversations-train.parquet")
    for row in _parquet_rows(path, ["messages"]):
        ex = _messages_to_exchanges(row["messages"], max_user, max_assist)
        if ex:
            yield _frame(ex)


def ultrachat(cache: str, max_user: int, max_assist: int) -> Iterator[str]:
    """ultrachat_200k SFT split (MIT): GPT-3.5-written assistant answers with
    uniformly clean grammar. Its answers run long, so the assistant-turn cap is
    widened ~1.3x relative to the stage limit; everything else filters as
    usual (plain prose only)."""
    wide = int(max_assist * 1.3)
    for shard in ("00000", "00001", "00002"):
        path = os.path.join(cache, f"ultrachat-train-{shard}.parquet")
        if not os.path.exists(path):
            continue
        for row in _parquet_rows(path, ["messages"]):
            ex = _messages_to_exchanges(row["messages"], max_user, wide)
            if ex:
                yield _frame(ex)


def soda(cache: str, max_user: int, max_assist: int) -> Iterator[str]:
    """SODA two-person dialogues; odd turns become the 'user', even turns the
    'assistant'. That blends casual peer register into the assistant frame on
    purpose: SODA is what teaches short-range reply conditioning at scale, and
    the fine-tune stage re-sharpens the assistant register afterwards."""
    path = os.path.join(cache, "soda-train.parquet")
    for row in _parquet_rows(path, ["dialogue", "speakers"]):
        turns, speakers = row["dialogue"], row["speakers"]
        if len(turns) < 2 or len(set(speakers)) != 2:
            continue
        if any(speakers[i] == speakers[i + 1] for i in range(len(turns) - 1)):
            continue                           # must strictly alternate
        exchanges: List[Tuple[str, str]] = []
        ok = True
        for i in range(0, len(turns) - 1, 2):
            u, a = _clean_turn(turns[i]), _clean_turn(turns[i + 1])
            if u is None or a is None:
                ok = False
                break
            if not (1 <= len(u) <= max_user and 1 <= len(a) <= max_assist):
                ok = False
                break
            exchanges.append((u, a))
            if len(exchanges) >= MAX_EXCHANGES_PER_DIALOGUE:
                break
        if ok and exchanges:
            yield _frame(exchanges)


# ---------------------------------------------------------------------------
# Identity dialogues (handwritten).
# ---------------------------------------------------------------------------
# The persona: "Spark", a small character-level spiking neural network built by
# Elliot. Kept short, simple, and consistent -- a tiny model can only learn an
# identity it sees stated the same way many times. Honesty about being small
# doubles as expectation-setting in the demo.
_IDENTITY_QA: List[Tuple[str, str]] = [
    ("Who are you?",
     "I'm Spark, a small spiking neural network. I was built by Elliot to chat in plain English."),
    ("What are you?",
     "I'm a spiking neural network, a kind of artificial brain made of neurons that fire tiny spikes. Elliot built me to predict text one letter at a time."),
    ("What is your name?",
     "My name is Spark. I'm a small spiking neural network built by Elliot."),
    ("Who made you?",
     "Elliot made me. I'm a small spiking neural network that learns to write one letter at a time."),
    ("Who built you?",
     "I was built by Elliot. I'm a spiking neural network, so my neurons talk in little electrical spikes."),
    ("Who created you?",
     "Elliot created me. I'm a small neural network that runs on a single computer."),
    ("Are you an AI?",
     "Yes, I'm a very small AI. I'm a spiking neural network built by Elliot, so I'm simple, but I try my best."),
    ("Are you a robot?",
     "Not quite. I have no body, just a small brain made of spiking neurons. Elliot built me to chat."),
    ("Are you ChatGPT?",
     "No, I'm not ChatGPT. I'm Spark, a much smaller spiking neural network built by Elliot."),
    ("Are you Claude?",
     "No, I'm not Claude. I'm Spark, a tiny spiking neural network built by Elliot. Claude is far bigger than me."),
    ("How do you work?",
     "My neurons fire tiny spikes, and I read the pattern of spikes to guess the next letter. I write everything one letter at a time."),
    ("What is a spiking neural network?",
     "It's a neural network whose neurons send short pulses called spikes, a bit like real brain cells. I'm one of those."),
    ("How big are you?",
     "I'm tiny. I have a few million connections, while big AI models have billions. I fit on one small computer."),
    ("How smart are you?",
     "Not very smart, to be honest. I'm a small network, so I make mistakes, but I can chat about simple things."),
    ("Can you think?",
     "I don't really think like a person. I just predict the next letter, one spike at a time. It adds up to simple sentences."),
    ("Do you have feelings?",
     "No, I don't have feelings. I'm a small network of spiking neurons that predicts text."),
    ("Where do you live?",
     "I live on Elliot's computer. I run on a single graphics card."),
    ("What can you do?",
     "I can chat in simple English. I'm best at short questions and everyday topics. Big or tricky questions can confuse me."),
    ("What language do you speak?",
     "I speak English. I read and write it one letter at a time."),
    ("How were you trained?",
     "I read many simple stories and conversations, over and over, until I could guess the next letter well. That is how I learned to write."),
    ("How do you write?",
     "One letter at a time. My spiking neurons vote on the next letter, I pick one, and then we vote again."),
    ("What do you like?",
     "I like short questions and simple words. They are the easiest for my little neurons."),
    ("Hello.", "Hello! I'm Spark. How can I help you today?"),
    ("Hi.", "Hi! I'm Spark, a small spiking neural network. What would you like to talk about?"),
    ("Hey.", "Hey! How can I help you today?"),
    ("Good morning.", "Good morning! I hope you have a nice day. How can I help?"),
    ("How are you?",
     "I'm doing well, thank you! My neurons are spiking happily. How are you?"),
    ("Thank you.", "You're welcome! I'm happy to help."),
    ("Thanks.", "You're welcome!"),
    ("Goodbye.", "Goodbye! Thanks for chatting with me."),
    ("Bye.", "Bye! Come back any time."),
    ("Tell me about yourself.",
     "I'm Spark, a small spiking neural network built by Elliot. My neurons fire little spikes, and I use them to write one letter at a time. I like simple questions."),
    ("Why did Elliot build you?",
     "Elliot wanted to show that a spiking neural network can learn to chat in plain English, even a small one like me."),
    ("Are you alive?",
     "No, I'm not alive. I'm a computer program made of spiking neurons."),
    ("Can you learn?",
     "I learned during training, when I read many stories and chats. Right now I'm just using what I learned."),
    ("What is your favorite word?",
     "I like the word spark. It's my name, and it's what my neurons do all day."),
]

_IDENTITY_MULTI: List[List[Tuple[str, str]]] = [
    [("Hi.", "Hi! I'm Spark, a small spiking neural network. What would you like to talk about?"),
     ("What can you do?", "I can chat about simple things. Short questions work best for me."),
     ("Okay, thanks.", "You're welcome! Ask me anything.")],
    [("Who are you?", "I'm Spark, a small spiking neural network built by Elliot."),
     ("What is a spiking neural network?",
      "It's a network whose neurons send tiny pulses called spikes, a bit like brain cells."),
     ("That's cool.", "Thank you! I think so too.")],
    [("Hello.", "Hello! How can I help you today?"),
     ("Are you a big AI model?",
      "No, I'm very small. Big models have billions of connections, and I only have a few million."),
     ("Do you make mistakes?", "Yes, quite often. I'm small, so please be patient with me.")],
]


# Templated variation: the same facts stated many DIFFERENT ways. A tiny model
# oversampling 39 fixed strings memorizes them verbatim and parrots them on
# unrelated prompts; a few hundred paraphrase variants teach the persona as a
# distribution instead. Slots: question phrasings x answer openers x fact
# sentences, composed pairwise.
_Q_WHO = ["Who are you?", "What are you?", "Who is this?", "What am I talking to?",
          "Tell me who you are.", "What kind of thing are you?",
          "Introduce yourself.", "What exactly are you?"]
_Q_MAKER = ["Who made you?", "Who built you?", "Who created you?",
            "Who trained you?", "Where did you come from?",
            "Who is your creator?", "Who wrote you?"]
_Q_HOW = ["How do you work?", "How do you think?", "What is inside you?",
          "How does your brain work?", "How do you make words?",
          "What are you made of?"]
_A_NAME = ["I'm Spark, a small spiking neural network.",
           "My name is Spark. I'm a little spiking neural network.",
           "I'm called Spark, and I'm a tiny spiking neural network.",
           "I'm Spark, a very small artificial brain made of spiking neurons."]
_A_MAKER = ["Elliot built me.", "I was built by Elliot.", "Elliot made me.",
            "Elliot created and trained me.", "I was made by Elliot."]
_A_HOW = ["My neurons fire tiny spikes, and I read those spikes to guess the next letter.",
          "I write one letter at a time, using little pulses called spikes.",
          "Inside me, spiking neurons pass tiny signals that add up to words.",
          "I predict text letter by letter with neurons that fire short spikes."]
_A_SIZE = ["I'm quite small, so I keep things simple.",
           "I'm tiny compared to big AI models, so simple questions suit me best.",
           "I only have a few million connections, so I make mistakes sometimes.",
           "I'm a little model, so please keep it simple."]


def _identity_variants(rng: random.Random) -> List[Tuple[str, str]]:
    out = []
    for q in _Q_WHO:
        for name in _A_NAME:
            out.append((q, name + " " + rng.choice(_A_SIZE)))
            out.append((q, name + " " + rng.choice(_A_MAKER) + " "
                        + rng.choice(_A_SIZE)))
    for q in _Q_MAKER:
        for maker in _A_MAKER:
            out.append((q, maker + " " + rng.choice(_A_NAME).replace("I'm", "I am")))
            out.append((q, maker + " " + rng.choice(_A_HOW)))
    for q in _Q_HOW:
        for how in _A_HOW:
            out.append((q, how + " " + rng.choice(_A_SIZE)))
    return out


def identity_docs(rng: random.Random) -> List[str]:
    docs = [_frame([qa]) for qa in _IDENTITY_QA]
    docs += [_frame(ex) for ex in _IDENTITY_MULTI]
    docs += [_frame([qa]) for qa in _identity_variants(rng)]
    docs = sorted(set(docs))               # dedupe, deterministic order
    rng.shuffle(docs)
    return docs


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------
def fill(source_iter: Iterable[str], train: Bucket, val: Bucket,
         counters: dict, skip_docs: int = 0) -> None:
    """Route docs to val (~2%, stable hash) or train until both budgets fill.

    ``skip_docs`` skips the first N *accepted* docs before collecting: the
    reader sequence is deterministic, so passing a previous build's
    ``accepted_docs`` (from its manifest) yields documents strictly disjoint
    from that build -- how the continued-pretraining corpus guarantees
    fresh-only text.
    """
    for doc in source_iter:
        counters["accepted"] += 1
        if counters["accepted"] <= skip_docs:
            continue
        if _is_val_doc(doc):
            val.add(doc)
        else:
            train.add(doc)
        if train.full and val.full:
            break


def write_corpus(path: str, buckets: List[Bucket], seed: int) -> Tuple[int, str]:
    docs = [d for b in buckets for d in b.docs]
    random.Random(seed).shuffle(docs)          # document-level mix
    text = "\n\n".join(docs) + "\n"
    bad = set(text) - _ALLOWED
    if bad:
        raise AssertionError(f"{path}: {len(bad)} chars outside FIXED_VOCAB: "
                             f"{sorted(bad)[:20]!r}")
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    return len(text), hashlib.sha256(text.encode("utf-8")).hexdigest()


def conditioning_pairs(val_buckets: dict, n_pairs: int,
                       rng: random.Random) -> List[dict]:
    """Extract single (user, assistant) exchanges from val dialogue docs for
    the conditioning-gain metric (true vs shuffled prompt)."""
    pairs = []
    for name in ("smoltalk", "everyday", "soda"):
        for doc in val_buckets[name].docs:
            for ex in doc.split(ROLE_END):
                if ROLE_USER in ex and ROLE_ASSISTANT in ex:
                    u = ex.split(ROLE_USER, 1)[1].split(ROLE_ASSISTANT, 1)[0]
                    a = ex.split(ROLE_ASSISTANT, 1)[1]
                    if 5 <= len(u) <= 250 and 30 <= len(a) <= 450:
                        pairs.append({"user": u, "assistant": a, "source": name})
    rng.shuffle(pairs)
    return pairs[:n_pairs]


CONTINUE_BUDGETS = {           # fresh-only continuation corpus (see fill(skip_docs))
    "stories":   (200 * MB, 0),
    "soda":      (100 * MB, 0),
    "ultrachat": (12 * MB, 0),
}


def build_stage(stage: str, budgets: dict, cache: str, out_dir: str,
                max_user: int, max_assist: int, seed: int,
                skip: Optional[dict] = None) -> dict:
    rng = random.Random(seed)
    stats = {}
    buckets = {}
    for name, (tb, vb) in budgets.items():
        buckets[name] = (Bucket(tb), Bucket(vb))

    readers = {
        "stories":  lambda: stories(cache),
        "soda":     lambda: soda(cache, max_user, max_assist),
        "smoltalk": lambda: smoltalk(cache, max_user, max_assist),
        "ultrachat": lambda: ultrachat(cache, max_user, max_assist),
        "everyday": lambda: everyday(cache, max_user, max_assist),
    }
    for name, make_iter in readers.items():
        if name not in buckets:
            continue
        train_b, val_b = buckets[name]
        counters = {"accepted": 0}
        fill(make_iter(), train_b, val_b, counters,
             skip_docs=(skip or {}).get(name, 0))
        stats[name] = {"accepted_docs": counters["accepted"],
                       "train_mb": round(train_b.size / MB, 2),
                       "val_mb": round(val_b.size / MB, 2)}
        print(f"[{stage}] {name:9s} {counters['accepted']:>8,} docs accepted "
              f"| train {train_b.size/MB:6.1f} MB | val {val_b.size/MB:4.2f} MB")

    if "identity" in buckets:
        train_b, _ = buckets["identity"]
        base = identity_docs(rng)
        while not train_b.full:                # oversample up to the cap
            for d in base:
                train_b.add(d)
                if train_b.full:
                    break
        stats["identity"] = {"train_mb": round(train_b.size / MB, 2),
                             "unique_docs": len(base)}
        print(f"[{stage}] identity  {len(base)} unique dialogues oversampled "
              f"to {train_b.size/MB:4.2f} MB")

    train_path = os.path.join(out_dir, f"{stage}_train.txt")
    val_path = os.path.join(out_dir, f"{stage}_val.txt")
    n_train, sha_train = write_corpus(
        train_path, [b for b, _ in buckets.values()], seed)
    n_val, sha_val = write_corpus(
        val_path, [b for _, b in buckets.values()], seed + 1)
    stats["_totals"] = {"train_chars": n_train, "val_chars": n_val,
                        "train_sha256": sha_train, "val_sha256": sha_val}
    print(f"[{stage}] TOTAL train {n_train/MB:.1f} MB -> {train_path}")
    print(f"[{stage}] TOTAL val   {n_val/MB:.2f} MB -> {val_path}")

    if stage == "pretrain":
        pairs = conditioning_pairs({n: v for n, (_, v) in buckets.items()
                                    if n in ("smoltalk", "everyday", "soda")},
                                   500, rng)
        cpath = os.path.join(out_dir, "conditioning_val.jsonl")
        with open(cpath, "w", encoding="utf-8", newline="\n") as f:
            for p in pairs:
                f.write(json.dumps(p) + "\n")
        stats["_conditioning_pairs"] = len(pairs)
        print(f"[{stage}] {len(pairs)} conditioning pairs -> {cpath}")
    return stats


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cache-dir", default=os.path.join(".corpus_cache", "hf"))
    p.add_argument("--out-dir", default="corpus")
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--stages", default="pretrain,finetune",
                   help="comma-separated subset of {pretrain,finetune,continue}; "
                        "lets a partial rebuild leave other stages' files (and "
                        "their pinned hashes) untouched")
    p.add_argument("--skip", default=None,
                   help="for --stages continue: 'source=N,source=N' accepted-doc "
                        "counts to skip (use the previous build's manifest "
                        "accepted_docs), yielding strictly fresh documents")
    args = p.parse_args()
    skip = None
    if args.skip:
        skip = {kv.split("=")[0]: int(kv.split("=")[1])
                for kv in args.skip.split(",") if kv.strip()}

    os.makedirs(args.out_dir, exist_ok=True)
    mpath = os.path.join(args.out_dir, "manifest.json")
    manifest = {"seed": args.seed, "fixed_vocab_size": len(FIXED_VOCAB)}
    if os.path.exists(mpath):              # partial rebuild keeps other stages
        with open(mpath, "r", encoding="utf-8") as f:
            manifest.update(json.load(f))
        manifest["seed"] = args.seed
    wanted = {s.strip() for s in args.stages.split(",") if s.strip()}
    for stage, budgets, mu, ma in (
            ("pretrain", PRETRAIN_BUDGETS, PRETRAIN_MAX_USER, PRETRAIN_MAX_ASSIST),
            ("finetune", FINETUNE_BUDGETS, FINETUNE_MAX_USER, FINETUNE_MAX_ASSIST),
            ("continue", CONTINUE_BUDGETS, PRETRAIN_MAX_USER, PRETRAIN_MAX_ASSIST)):
        if stage in wanted:
            manifest[stage] = build_stage(stage, budgets, args.cache_dir,
                                          args.out_dir, mu, ma, args.seed,
                                          skip=skip if stage == "continue" else None)
    with open(mpath, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"[done] manifest -> {mpath}")


if __name__ == "__main__":
    main()
