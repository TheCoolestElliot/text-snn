"""The chat vocabulary: a small fixed ASCII alphabet plus four role markers.

WHY NOT REUSE THE RESEARCH CORPUS VOCABULARY
--------------------------------------------
`snn.data` builds a 205-symbol vocabulary from the raw bytes of enwik8, which is
Wikipedia *markup*: it contains every byte that appears anywhere in 100 MB of
XML, including forty-odd symbols that occur a handful of times in the whole
file. For a conversational model that is dead weight in the embedding and in the
softmax, and worse, it makes the model's output distribution carry mass on
characters no reply should ever contain.

This vocabulary is the opposite choice: a closed, fixed, hand-written alphabet
(printable ASCII, minus the characters we normalise away) that is the SAME for
every chat corpus and every chat checkpoint. Fixed matters -- a vocabulary
derived from the corpus would change if the corpus mix changed, and every
checkpoint trained before the change would silently decode to different text.
`VOCAB_VERSION` is stamped into every checkpoint and every packed corpus, and
loading a mismatch is a hard error rather than a garbled transcript.

THE ROLE MARKERS ARE TOKENS, NOT TEXT
-------------------------------------
A char-level model given a literal `"User: "` prefix has to spend probability
mass rediscovering that string, and -- the real problem -- it can *emit* it
half-way through a reply, at which point nothing downstream can tell a turn
boundary from six characters of chat about the word "user". So the four markers
below are ids the model can emit and no text can ever produce: `encode` cannot
generate them from user input, because the normaliser maps every input byte into
the printable range first. A turn boundary is then an exact integer comparison,
not a string search.

`BOS` exists so that the very first character of a conversation is conditioned on
something rather than on a zero membrane; `EOT` closes an assistant turn and is
the model's own "I am done speaking" signal, which is what makes the REPL able to
stop on the model's decision instead of on a character budget.
"""

from __future__ import annotations

import unicodedata

import numpy as np

__all__ = [
    "VOCAB_VERSION",
    "SPECIALS",
    "BOS",
    "USER",
    "BOT",
    "EOT",
    "ChatTokenizer",
    "normalise_text",
]

#: Bump ONLY when the alphabet or the special ids change. Every packed corpus
#: and every checkpoint records it; a mismatch is refused rather than decoded.
VOCAB_VERSION = 1

# --- the four ids that are not characters ---------------------------------
BOS = 0     # start of a conversation
USER = 1    # "the human speaks now"
BOT = 2     # "the model speaks now"
EOT = 3     # end of the current turn

SPECIALS: tuple[str, ...] = ("<|bos|>", "<|user|>", "<|bot|>", "<|eot|>")

#: The character alphabet, in id order after the specials. Printable ASCII from
#: space to `~`, plus newline and tab. Deliberately closed and deliberately
#: small: 97 characters + 4 markers = 101 ids, half the research corpus's 205,
#: which halves the softmax and the embedding for a model this size.
#:
#: Carriage return is absent on purpose -- `normalise_text` folds CRLF to LF, so
#: a Windows-authored dataset and a Unix-authored one pack to identical bytes.
_ALPHABET = "".join(
    [
        "\t\n",
        "".join(chr(c) for c in range(0x20, 0x7F)),  # space .. '~'
    ]
)

#: Unicode that a text corpus is full of and that we do not want 40 extra
#: embedding rows for. Applied before the ASCII filter, so a curly quote becomes
#: an apostrophe rather than disappearing.
_FOLD = {
    "‘": "'", "’": "'", "‚": "'", "‛": "'",
    "“": '"', "”": '"', "„": '"', "‟": '"',
    "′": "'", "″": '"',
    "‐": "-", "‑": "-", "‒": "-", "–": "-",
    "—": "-", "―": "-", "−": "-",
    "…": "...", "•": "*", "·": "*",
    # Spelled as escapes, not literals: these are invisible in an editor, and a
    # keyboard space pasted in among them would be indistinguishable by eye from a
    # zero-width joiner. `PLE2515` enforces it.
    "\u00a0": " ", "\u2007": " ", "\u202f": " ", "\u2009": " ",
    "\u200b": "", "\u200c": "", "\u200d": "", "\ufeff": "",
    "é": "e", "è": "e", "ê": "e", "ë": "e",
    "á": "a", "à": "a", "â": "a", "ä": "a", "å": "a",
    "í": "i", "ì": "i", "î": "i", "ï": "i",
    "ó": "o", "ò": "o", "ô": "o", "ö": "o",
    "ú": "u", "ù": "u", "û": "u", "ü": "u",
    "ñ": "n", "ç": "c", "ß": "ss",
    "É": "E", "Á": "A", "Ó": "O", "Ú": "U", "Ñ": "N",
    "€": "EUR", "£": "GBP", "©": "(c)", "®": "(r)",
    "→": "->", "←": "<-", "×": "x", "½": "1/2",
}

_TRANSLATE = str.maketrans(_FOLD)


def normalise_text(text: str, *, replacement: str = "") -> str:
    """Fold `text` into the closed alphabet.

    Three passes, in this order and for these reasons:

    1.  NFKD, which splits `e-acute` into `e` + combining accent, so the accent
        can be dropped without losing the letter. Doing this *after* the fold
        table would make the table's accented entries unreachable; doing the
        fold first means the common cases are handled by an exact map and NFKD
        only has to catch the tail.
    2.  The fold table, for the punctuation and currency that NFKD leaves alone
        (a curly quote is not a decomposable character).
    3.  A final filter that drops anything still outside the alphabet.

    `replacement` is empty by default: an unmappable character is *deleted*
    rather than turned into a placeholder, because a placeholder is a character
    the model then has to learn to emit and never should.
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.translate(_TRANSLATE)
    text = unicodedata.normalize("NFKD", text)
    out = []
    for ch in text:
        if ch in _ALLOWED:
            out.append(ch)
        elif unicodedata.combining(ch):
            continue  # a stripped accent, not a lost character
        else:
            out.append(replacement)
    return "".join(out)


_ALLOWED = frozenset(_ALPHABET)


class ChatTokenizer:
    """Bidirectional map between text and ids. Stateless and fixed.

    Not built from data, not fitted, not saved: two `ChatTokenizer()` instances
    are equal by construction. What *is* saved (in the corpus manifest and in
    every checkpoint) is `VOCAB_VERSION`, which is all that is needed to detect
    that a checkpoint predates a change to this file.
    """

    def __init__(self) -> None:
        self.version = VOCAB_VERSION
        self.n_special = len(SPECIALS)
        self.alphabet = _ALPHABET
        self.vocab_size = self.n_special + len(_ALPHABET)
        # id -> byte value, for ids >= n_special. Specials decode to "".
        self._itos = np.zeros(self.vocab_size, dtype=np.uint8)
        # byte value -> id, 255 = unmappable (checked, never packed)
        self._stoi = np.full(256, 255, dtype=np.uint8)
        for i, ch in enumerate(_ALPHABET):
            code = ord(ch)
            self._itos[self.n_special + i] = code
            self._stoi[code] = self.n_special + i

    # -- text -> ids -------------------------------------------------------

    def encode(self, text: str, *, normalise: bool = True) -> np.ndarray:
        """Plain text -> `int64` ids. Never produces a special id.

        That guarantee is the point: whatever a user types, it cannot forge a
        turn boundary, because every byte it could produce maps into
        `[n_special, vocab_size)` or is dropped by the normaliser.
        """
        if normalise:
            text = normalise_text(text)
        raw = np.frombuffer(text.encode("ascii", errors="ignore"), dtype=np.uint8)
        ids = self._stoi[raw]
        return ids[ids != 255].astype(np.int64)

    def decode(self, ids) -> str:
        """ids -> text. Special ids render as their `<|marker|>` spelling.

        Rendering rather than dropping them is deliberate: a debugging dump of a
        training window must show where the turn boundaries are, and a reply
        that leaked a marker should be visibly wrong rather than invisibly so.
        `decode_visible` is the one to use for anything a user reads.
        """
        out = []
        for i in np.asarray(ids, dtype=np.int64).ravel():
            i = int(i)
            if i < self.n_special:
                out.append(SPECIALS[i])
            else:
                out.append(chr(int(self._itos[i])))
        return "".join(out)

    def decode_visible(self, ids) -> str:
        """ids -> text, with special ids dropped. Use this for user-facing text."""
        arr = np.asarray(ids, dtype=np.int64).ravel()
        arr = arr[arr >= self.n_special]
        return self._itos[arr].tobytes().decode("ascii", errors="replace")

    # -- conversation rendering -------------------------------------------

    def render_turn(self, role: str, text: str) -> list[int]:
        """One turn as ids: `<|role|>` + text + `<|eot|>`.

        The marker leads and `EOT` closes, so a turn is self-delimiting in both
        directions. Generation puts `BOT` on the stream and then samples until
        `EOT`; nothing has to guess where the reply started.
        """
        if role not in ("user", "bot"):
            raise ValueError(f"role must be 'user' or 'bot', got {role!r}")
        marker = USER if role == "user" else BOT
        return [marker, *self.encode(text).tolist(), EOT]

    def render_conversation(self, turns, *, bos: bool = True) -> list[int]:
        """`[(role, text), ...]` -> ids for a whole conversation."""
        ids: list[int] = [BOS] if bos else []
        for role, text in turns:
            ids.extend(self.render_turn(role, text))
        return ids

    def __repr__(self) -> str:
        return (
            f"ChatTokenizer(v{self.version}, vocab_size={self.vocab_size}, "
            f"specials={self.n_special})"
        )
