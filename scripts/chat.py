"""Talk to the spiking network.

    python scripts/chat.py                          # newest chat checkpoint
    python scripts/chat.py --ckpt <path>            # a specific one
    python scripts/chat.py --ckpt experiments/runs/compose_s1/ckpt_best.pt

The third form loads a *research* checkpoint. Those are trained on enwik8 and
have never seen a conversation, so there is nothing to chat with; the REPL
detects it and switches to continuation mode, where what you type is a prefix
the model carries on from. That is still worth doing -- it is the most direct
way to see what the Phase-4 arms actually learned -- and it is labelled clearly
so no one mistakes one mode for the other.

Not part of the research protocol. No number this prints is a reported figure.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO / "src") not in sys.path:
    sys.path.insert(0, str(_REPO / "src"))

import torch  # noqa: E402

from snnchat.generate import (ChatSampler, ChatSession, SamplingParams,  # noqa: E402
                              load_chat_checkpoint)
from snnchat.tokenizer import ChatTokenizer, normalise_text  # noqa: E402


def _enable_colour() -> bool:
    """True if it is safe to emit ANSI escapes on this stream.

    Windows is the reason this is a function. `conhost.exe` does not interpret
    escape sequences unless a process explicitly turns on
    `ENABLE_VIRTUAL_TERMINAL_PROCESSING`, and a REPL that prints `\\x1b[1myou`
    at every prompt is worse than one with no colour at all. Windows Terminal
    enables it by default and older consoles do not, so it is requested rather
    than assumed and the return value decides whether anything is emitted.

    `NO_COLOR` is honoured (https://no-color.org), and a redirected stdout gets
    no escapes at all -- which is what makes `--prompt` safe to pipe.
    """
    import os

    if os.environ.get("NO_COLOR") or not sys.stdout.isatty():
        return False
    if sys.platform != "win32":
        return True
    try:
        import ctypes

        kernel32 = ctypes.windll.kernel32
        handle = kernel32.GetStdHandle(-11)          # STD_OUTPUT_HANDLE
        mode = ctypes.c_uint32()
        if not kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
            return False
        return bool(kernel32.SetConsoleMode(handle, mode.value | 0x0004))
    except Exception:
        return False


COLOUR = False


def c(text: str, code: str) -> str:
    """Wrap `text` in an SGR code, or return it unchanged if colour is off."""
    return f"\x1b[{code}m{text}\x1b[0m" if COLOUR else text

BANNER = r"""
   ___ _  _ _  _    ___ _  _   _ _____
  / __| \| | \| |  / __| || | /_\_   _|   a spiking neural network
  \__ \ .` | .` | | (__| __ |/ _ \| |     that talks, one character
  |___/_|\_|_|\_|  \___|_||_/_/ \_\_|     at a time
"""

HELP = """
commands
  /help                 this message
  /new                  start a fresh conversation (clears membrane state)
  /again                re-answer the last thing you said
  /back [n]             undo the last n exchanges
  /temp <x>             sampling temperature        (higher = wilder)
  /topp <x>             nucleus cutoff, 0 disables
  /topk <n>             top-k cutoff, 0 disables
  /minp <x>             floor as a fraction of the peak probability
  /len <n>              maximum characters per reply
  /rerank <n>           draw n replies and keep the most on-topic (1 = off)
  /lambda <x>           how hard reranking punishes a generic reply (0..1.5)
  /spread <x>           draw the drafts over a range of temperatures (0 = off)
  /candidates           show what the last reply was chosen from
  /seed <n|off>         fix the sampler's seed for reproducible replies
  /params               show the current sampling settings
  /spikes [on|off]      show which neurons fired while it answered
  /state                what the model is carrying right now
  /transcript           print the conversation so far
  /save <file>          write the transcript to a file
  /quit                 leave
"""


#: A run whose name starts with `_` is scratch -- `_probe/*` from the size
#: bake-off, `_smoke` from a pipeline test. They are real checkpoints in real run
#: directories and `rglob` finds them happily, so without this the default
#: checkpoint after a fresh clone-and-probe is a five-minute throwaway that
#: happens to have the newest `ckpt_best.pt`. `--ckpt` still loads them.
_SCRATCH_PREFIX = "_"

#: Reranking defaults, chosen from the sweep in `docs/chat/QUALITY.md` under a
#: rule stated before the row was picked: **maximise topicality subject to not
#: making the model less fluent than it is without reranking.**
#:
#: The fluency floor is the model's own mean per-character log-probability of the
#: reply it selects. Plain sampling scores -0.3934. lambda 0.6 scores -0.3251 --
#: better than plain sampling while raising topicality from 0.094 to 0.141 and
#: cutting the story-dodge rate from 0.083 to 0.062. lambda 1.0 scores -0.4277,
#: below the floor, which is the anti-LM term starting to buy rarity with
#: garble; the transcript at that setting contains "There offer the bank is for
#: a dominator", which is what that number looks like in prose.
#:
#: NOT the argmax of the pre-registered headline, which is `n=16, lambda=0`
#: (0.308 against 0.262). That setting answers a prompt asking for no narrative
#: WITH a narrative 16.7 % of the time against 8.3 % for plain sampling: on a
#: model trained 40 % on stories, best-of-N by likelihood picks the story, and
#: the anti-LM term is precisely what cancels that. The headline could not see
#: it, because `story_dodge` did not exist when the headline was fixed -- it was
#: added after reading a transcript, and both facts are in QUALITY.md.
#:
#: `1` restores the pre-reranking behaviour exactly --
#: `tests/test_snnchat.py::test_a_single_candidate_rerank_is_the_ordinary_sampler`
#: is what makes that a guarantee rather than an intention.
DEFAULT_RERANK_N = 8
DEFAULT_RERANK_LAMBDA = 0.6

#: How wide a range of temperatures the drafts are proposed over. See
#: `snnchat.rerank.RerankParams.temperature_spread`: selection cannot invent a
#: reply that was never proposed, so once reranking is on, the pool is the lever
#: that is left. The default is set from the measurement in
#: `docs/chat/QUALITY_v4.md` §4 and nowhere else -- 0.0 means the pool is drawn
#: exactly as it was before this knob existed.
DEFAULT_RERANK_SPREAD = 0.0


def _is_scratch(path: Path, root: Path) -> bool:
    try:
        parts = path.relative_to(root).parts
    except ValueError:
        return False
    return any(part.startswith(_SCRATCH_PREFIX) for part in parts[:-1])


#: A one-line file naming the checkpoint to load by default, relative to the
#: chat directory. It exists because modification time is the wrong rule as soon
#: as there is more than one KIND of run: the responsiveness round of 2026-08-06
#: finished with a deliberately-worse *control* arm, which by mtime would have
#: become the model everyone talked to. An explicit pointer says which checkpoint
#: is the product; mtime remains the fallback when there is none.
_SHIPPED = "SHIPPED"


def _shipped_checkpoint(root: Path) -> Path | None:
    marker = root / _SHIPPED
    if not marker.exists():
        return None
    for line in marker.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        path = (root / line).resolve()
        return path if path.exists() else None
    return None


def _find_latest_checkpoint(root: Path) -> Path | None:
    """The checkpoint `SHIPPED` names; failing that, the newest one.

    `best` is preferred over `last` because a run stopped mid-anneal can be
    meaningfully worse than its own best evaluation, and the first thing anyone
    does after training is chat to it.
    """
    named = _shipped_checkpoint(root)
    if named is not None:
        return named
    for name in ("ckpt_best.pt", "ckpt_last.pt"):
        found = sorted(
            (p for p in root.rglob(name) if not _is_scratch(p, root)),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if found:
            return found[0]
    return None


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--ckpt", default=None,
                   help="checkpoint to load; default = newest under experiments/chat/")
    p.add_argument("--chat-dir", default="experiments/chat",
                   help="where to look for the newest checkpoint")
    p.add_argument("--device", default=None)
    p.add_argument("--temperature", type=float, default=0.85)
    p.add_argument("--top-p", type=float, default=0.92)
    p.add_argument("--top-k", type=int, default=0)
    p.add_argument("--min-p", type=float, default=0.02)
    p.add_argument("--max-new", type=int, default=400)
    p.add_argument("--seed", type=int, default=None,
                   help="fix the sampler seed; omit for a different reply each time")
    p.add_argument("--rerank", type=int, default=DEFAULT_RERANK_N, metavar="N",
                   help="draw N replies per turn and keep the one the prompt best "
                        "explains; 1 disables it. Batch is nearly free on this "
                        "network, so N costs little more than 1 (see "
                        "docs/chat/QUALITY.md)")
    p.add_argument("--rerank-lambda", type=float, default=DEFAULT_RERANK_LAMBDA,
                   metavar="X",
                   help="weight on the 'would it have said this anyway' term")
    p.add_argument("--rerank-min-chars", type=int, default=12,
                   help="candidates shorter than this are set aside unless all are")
    p.add_argument("--rerank-spread", type=float, default=DEFAULT_RERANK_SPREAD,
                   metavar="X",
                   help="propose the drafts over temperatures spanning "
                        "+/- X of --temperature; 0 draws them all at one")
    p.add_argument("--no-fused", action="store_true",
                   help="force the eager reference scan (slower, sometimes clearer errors)")
    p.add_argument("--no-stream", action="store_true",
                   help="print the whole reply at once instead of character by character")
    p.add_argument("--prompt", default=None,
                   help="send one message, print the reply, exit")
    p.add_argument("--list", action="store_true",
                   help="list the checkpoints under --chat-dir and exit")
    return p


def list_checkpoints(root: Path) -> int:
    """Show what is available, newest first, with the default marked.

    Worth having because there is normally more than one run -- the main run and
    its chat-focused anneal -- and the default picks by modification time, which
    is right but invisible.
    """
    found = sorted(
        (p for name in ("ckpt_best.pt", "ckpt_last.pt") for p in root.rglob(name)),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not found:
        print(f"no checkpoints under {root}/")
        return 1
    default = _find_latest_checkpoint(root)
    print(f"checkpoints under {root}/ (newest first):\n")
    for path in found:
        try:
            ck = torch.load(path, map_location="cpu", weights_only=False)
            step = ck.get("step", "?")
            bpc = ck.get("best_val_bpc")
            bpc_s = f"{bpc:.4f}" if isinstance(bpc, float) and bpc < 1e9 else "-"
            params = ck.get("description", {}).get("params", "?")
            detail = f"step {step:>7}  best val bpc {bpc_s:>7}  {params:>10,} params"
        except Exception as exc:  # a half-written .pt must not kill the listing
            detail = f"unreadable: {type(exc).__name__}"
        # Resolved on both sides: `found` comes from rglob and is relative,
        # while a SHIPPED pointer is resolved against the chat directory, so a
        # plain `==` would silently never mark anything.
        same = default is not None and path.resolve() == default.resolve()
        mark = " <- default" if same else ""
        if _is_scratch(path, root):
            mark += "  (scratch; --ckpt to load)"
        print(f"  {str(path):<55} {detail}{mark}")
    return 0


def _stream_writer():
    def write(ch: str) -> None:
        sys.stdout.write(ch)
        sys.stdout.flush()
    return write


# --------------------------------------------------------------------------
# continuation mode: a research checkpoint, which has no conversation format
# --------------------------------------------------------------------------


def run_continuation(model, cfg, args) -> int:
    """Prefix-continuation REPL for an enwik8-trained research checkpoint.

    Uses the *research* corpus's vocabulary, read off disk, because that
    checkpoint's embedding rows are indexed by it. Mixing in `ChatTokenizer`
    here would silently decode every character to the wrong one.
    """
    import numpy as np

    from snn.data import Corpus

    corpus = Corpus(cfg.corpus, cfg.data_dir)
    if int(cfg.vocab_size) != int(corpus.vocab_size):
        raise SystemExit(
            f"checkpoint vocab_size={cfg.vocab_size} but corpus {cfg.corpus} has "
            f"{corpus.vocab_size}; refusing to sample from a mismatched model"
        )
    device = torch.device(cfg.device)
    print(BANNER)
    print(f"  continuation mode -- this is a research checkpoint (arch {cfg.arch}, "
          f"corpus {cfg.corpus}).")
    print("  It has never seen a conversation. What you type is a prefix it continues.")
    print("  /quit to leave, /len <n> to change how much it writes.\n")

    max_new = args.max_new
    while True:
        try:
            line = input(c(">", "1") + " ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not line:
            continue
        if line in ("/quit", "/exit", "/q"):
            return 0
        if line.startswith("/len"):
            try:
                max_new = int(line.split()[1])
                print(f"  max_new = {max_new}")
            except (IndexError, ValueError):
                print("  usage: /len <n>")
            continue

        raw = np.frombuffer(line.encode("utf-8", errors="ignore"), dtype=np.uint8)
        ids = corpus.stoi[raw]
        if (ids == 255).any():
            bad = sorted({int(b) for b in raw[ids == 255]})
            print(f"  ! byte(s) {bad} never occur in {cfg.corpus} and have no vocab slot")
            continue

        params = SamplingParams(
            temperature=args.temperature, top_k=args.top_k, top_p=args.top_p,
            min_p=args.min_p, max_new=max_new, seed=args.seed,
        )
        sys.stdout.write(c(line, "2"))
        sys.stdout.flush()
        # The same sampler the chat path uses, with the stop set EMPTY: this
        # vocabulary has no role markers, so there is no id that means "stop"
        # and generation runs to the character budget. Reusing it rather than
        # re-implementing the loop is what keeps `--temperature` and the loop
        # breaker behaving identically in both modes.
        sampler = ChatSampler(model, ChatTokenizer(), device=device)
        for nxt, _ in sampler.stream(ids.astype("int64").tolist(), params, stop_ids=()):
            sys.stdout.write(
                corpus.decode(np.array([nxt])).decode("utf-8", errors="replace")
            )
            sys.stdout.flush()
        print("\n")


# --------------------------------------------------------------------------
# chat mode
# --------------------------------------------------------------------------


def run_chat(model, cfg, ck, args) -> int:
    tok = ChatTokenizer()
    params = SamplingParams(
        temperature=args.temperature, top_k=args.top_k, top_p=args.top_p,
        min_p=args.min_p, max_new=args.max_new, seed=args.seed,
    )
    rerank = None
    if args.rerank > 1:
        from snnchat.rerank import RerankParams

        rerank = RerankParams(n=args.rerank, lam=args.rerank_lambda,
                              min_chars=args.rerank_min_chars,
                              temperature_spread=args.rerank_spread)
    session = ChatSession(model, tok, device=cfg.device, params=params,
                          rerank=rerank)

    if args.prompt is not None:
        print(session.send(args.prompt))
        return 0

    n_params = sum(p.numel() for p in model.parameters())
    step = ck.get("step", "?")
    desc = ck.get("description", {})
    taus = desc.get("slow_pole_tau") or []
    trained = f", trained {step:,} steps" if isinstance(step, int) else ""
    print(BANNER)
    print(f"  {n_params:,} parameters, d={cfg.d_model}, "
          f"{cfg.n_layers} spiking layers{trained}")
    if taus:
        t = taus[0]
        print(f"  slow-pole time constants {t['tau_min']:g}-{t['tau_max']:g} characters "
              f"(median {t['tau_median']:g})")
    print("  It has no context window: the conversation lives in its membrane state.")
    if rerank is not None:
        print(f"  Each reply is the best of {rerank.n} drafts "
              f"(/rerank 1 turns that off, /candidates shows the rest).")
    print("  It is small, it is often wrong, and it forgets. /help for commands.\n")

    write = None if args.no_stream else _stream_writer()

    while True:
        try:
            line = input(c("you", "1") + " > ").rstrip("\n")
        except (EOFError, KeyboardInterrupt):
            print("\nbye.")
            return 0
        stripped = line.strip()
        if not stripped:
            continue

        if stripped.startswith("/"):
            done = _command(stripped, session, args)
            if done:
                return 0
            continue

        cleaned = normalise_text(stripped)
        if not cleaned:
            print("  ! nothing in that survived the model's alphabet (printable ASCII only)")
            continue
        if cleaned != stripped:
            print("  " + c(f"(read as: {cleaned})", "2"))

        sys.stdout.write(c("snn", "1") + " > ")
        sys.stdout.flush()
        t0 = time.perf_counter()
        reply = session.send(cleaned, on_char=write)
        dt = time.perf_counter() - t0
        if args.no_stream:
            sys.stdout.write(reply)
        rate = len(reply) / max(dt, 1e-9)
        print("\n  " + c(f"{len(reply)} chars in {dt:.1f}s ({rate:.0f} char/s)", "2"))
        if session.sampler.record_spikes:
            print(c(session.spike_summary(), "2"))
        print()


def _command(line: str, session: ChatSession, args) -> bool:
    """Handle a /command. Returns True if the REPL should exit."""
    parts = line.split()
    cmd, rest = parts[0], parts[1:]
    p = session.params

    def _num(kind, lo, hi, name):
        try:
            v = kind(rest[0])
        except (IndexError, ValueError):
            print(f"  usage: {cmd} <number>")
            return None
        if not (lo <= v <= hi):
            print(f"  {name} must be in [{lo}, {hi}]")
            return None
        return v

    if cmd in ("/quit", "/exit", "/q"):
        print("bye.")
        return True
    if cmd == "/help":
        print(HELP)
    elif cmd == "/new":
        session.reset()
        print("  fresh conversation; membrane state cleared\n")
    elif cmd == "/again":
        try:
            sys.stdout.write(c("snn", "1") + " > ")
            sys.stdout.flush()
            reply = session.regenerate(on_char=None if args.no_stream else _stream_writer())
            if args.no_stream:
                sys.stdout.write(reply)
            print("\n")
        except RuntimeError as exc:
            print(f"  {exc}")
    elif cmd == "/back":
        n = int(rest[0]) if rest and rest[0].isdigit() else 1
        session.rewind(n)
        print(f"  rewound {n} exchange(s); {len(session.turns) // 2} left\n")
    elif cmd == "/temp":
        v = _num(float, 0.01, 5.0, "temperature")
        if v is not None:
            p.temperature = v
            print(f"  temperature = {v:g}")
    elif cmd == "/topp":
        v = _num(float, 0.0, 1.0, "top_p")
        if v is not None:
            p.top_p = v
            print(f"  top_p = {v:g}" + ("  (disabled)" if v <= 0 else ""))
    elif cmd == "/topk":
        v = _num(int, 0, 100000, "top_k")
        if v is not None:
            p.top_k = v
            print(f"  top_k = {v}" + ("  (disabled)" if v == 0 else ""))
    elif cmd == "/minp":
        v = _num(float, 0.0, 1.0, "min_p")
        if v is not None:
            p.min_p = v
            print(f"  min_p = {v:g}")
    elif cmd == "/len":
        v = _num(int, 1, 100000, "max_new")
        if v is not None:
            p.max_new = v
            print(f"  max_new = {v}")
    elif cmd == "/seed":
        if rest and rest[0] in ("off", "none", "-"):
            p.seed = None
            print("  seed off; replies will vary")
        else:
            v = _num(int, -(2**62), 2**62, "seed")
            if v is not None:
                p.seed = v
                print(f"  seed = {v}; replies are now reproducible")
    elif cmd == "/rerank":
        v = _num(int, 1, 64, "candidates")
        if v is not None:
            from snnchat.rerank import RerankParams

            if v == 1:
                session.rerank = None
                print("  reranking off; one reply per turn, as sampled")
            else:
                lam = session.rerank.lam if session.rerank else args.rerank_lambda
                spread = (session.rerank.temperature_spread if session.rerank
                          else args.rerank_spread)
                session.rerank = RerankParams(n=v, lam=lam,
                                              min_chars=args.rerank_min_chars,
                                              temperature_spread=spread)
                print(f"  reranking {session.rerank.describe()}")
    elif cmd == "/lambda":
        v = _num(float, 0.0, 1.5, "lambda")
        if v is not None:
            if session.rerank is None:
                print("  reranking is off; /rerank <n> with n > 1 first")
            else:
                session.rerank.lam = v
                print(f"  lambda = {v:g}"
                      + ("  (0 = best of N by likelihood alone)" if v == 0 else ""))
    elif cmd == "/spread":
        v = _num(float, 0.0, 0.9, "spread")
        if v is not None:
            if session.rerank is None:
                print("  reranking is off; /rerank <n> with n > 1 first")
            else:
                session.rerank.temperature_spread = v
                if v == 0.0:
                    print("  spread off; every draft is drawn at the same temperature")
                else:
                    lad = session.rerank.temperature_ladder(p.temperature)
                    print(f"  spread = {v:g}; drafts drawn at "
                          f"{min(lad):.2f} .. {max(lad):.2f}")
    elif cmd == "/candidates":
        cands = session.last_candidates
        if not cands:
            print("  nothing to show -- reranking is off, or nothing has been said yet")
        else:
            best = max(cands, key=lambda c: c.score)
            print(f"  {len(cands)} candidates, ranked (* = chosen):")
            for cd in sorted(cands, key=lambda c: -c.score):
                mark = "*" if cd is best else " "
                text = session.tok.decode_visible(cd.ids).strip().replace("\n", " ")
                print(f"  {mark} {cd.score:+.4f}  {text[:96]!r}")
    elif cmd == "/params":
        rr = session.rerank.describe() if session.rerank else "off"
        print(f"  {p}\n  rerank {rr}")
    elif cmd == "/spikes":
        on = not (rest and rest[0] in ("off", "no", "0"))
        session.show_spikes(on)
        print(f"  spike recording {'on' if on else 'off'}"
              + ("; the rates appear after each reply" if on else ""))
    elif cmd == "/state":
        n_turns = len(session.turns) // 2
        print(f"  {n_turns} exchange(s), {session.chars_fed:,} characters fed through "
              f"the membrane\n  the model has no window -- older text has decayed, "
              f"not been dropped")
    elif cmd == "/transcript":
        print(session.transcript() or "  (nothing yet)")
    elif cmd == "/save":
        if not rest:
            print("  usage: /save <file>")
        else:
            Path(rest[0]).write_text(session.transcript() + "\n", encoding="utf-8")
            print(f"  wrote {rest[0]}")
    else:
        print(f"  unknown command {cmd!r}; /help for the list")
    return False


def main(argv: list[str] | None = None) -> int:
    global COLOUR
    args = build_parser().parse_args(argv)
    COLOUR = _enable_colour()

    if args.list:
        return list_checkpoints(Path(args.chat_dir))

    ckpt = Path(args.ckpt) if args.ckpt else _find_latest_checkpoint(Path(args.chat_dir))
    if ckpt is None:
        raise SystemExit(
            f"no checkpoint found under {args.chat_dir}/. Train one with:\n"
            f"    python scripts/chat/train.py\n"
            f"or point at a research checkpoint:\n"
            f"    python scripts/chat.py --ckpt experiments/runs/compose_s1/ckpt_best.pt"
        )
    if not ckpt.exists():
        raise SystemExit(f"{ckpt} does not exist")

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg, ck = load_chat_checkpoint(
        ckpt, device=device, fused=False if args.no_fused else None
    )
    if ck.get("kind") == "snnchat":
        return run_chat(model, cfg, ck, args)
    return run_continuation(model, cfg, args)


if __name__ == "__main__":
    raise SystemExit(main())
