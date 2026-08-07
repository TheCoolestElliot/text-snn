"""Run a fixed battery of prompts through a checkpoint and print the transcript.

    python scripts/chat/demo.py --ckpt experiments/chat/chat-v1/ckpt_best.pt

Why a fixed battery rather than whatever comes to mind: comparing two
checkpoints by chatting to them is worthless -- the prompts differ, the sampler
is stochastic, and the reader knows which one is supposed to be better. This
sends the same prompts, at the same seed, in the same order, so two runs of it
differ only where the models differ.

The battery is grouped by what it is probing, and includes the things the model
is expected to FAIL at. A demo that only shows a model's best case is an
advertisement; the arithmetic and the multi-turn memory sections are here
precisely because they are where it comes apart, and a reader should see that
without having to go looking.

Not part of the research protocol. No number it prints is a reported figure.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO / "src") not in sys.path:
    sys.path.insert(0, str(_REPO / "src"))

import torch  # noqa: E402

from snnchat.generate import ChatSession, SamplingParams, load_chat_checkpoint  # noqa: E402

#: (section, [prompts]). A section is one conversation -- state carries across
#: its prompts and is reset between sections, so the multi-turn sections are
#: actually testing multi-turn behaviour and the single ones are not polluted by
#: whatever came before.
BATTERY: list[tuple[str, list[str]]] = [
    ("greeting", ["hello"]),
    ("greeting, another phrasing", ["hey there"]),
    ("identity", ["what are you?"]),
    ("identity, another phrasing", ["are you a human?"]),
    ("capability", ["what can you do?"]),
    ("limits", ["do you remember our last conversation?"]),
    ("story, generic", ["tell me a story"]),
    ("story, topic-conditioned", ["tell me a story about a rabbit"]),
    ("story, another topic", ["tell me a story about a boy and his kite"]),
    ("small talk", ["how are you?"]),
    ("open question", ["what should we talk about?"]),
    ("simple question", ["what colour is the sky?"]),
    ("instruction", ["name three animals"]),
    ("multi-turn: does it hold a thread?",
     ["my name is Elliot", "what is my name?"]),
    ("multi-turn: follow-up",
     ["tell me a story about a cat", "what happened next?"]),
    ("EXPECTED FAILURE: arithmetic", ["what is 17 plus 25?"]),
    ("EXPECTED FAILURE: facts", ["what is the capital of France?"]),
    ("EXPECTED FAILURE: long instruction",
     ["write a haiku about autumn leaves falling in a quiet garden"]),
    ("politeness", ["thanks!"]),
    ("farewell", ["goodbye"]),
]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--device", default=None)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-p", type=float, default=0.92)
    p.add_argument("--min-p", type=float, default=0.02)
    p.add_argument("--max-new", type=int, default=300)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--rerank", type=int, default=1, metavar="N",
                   help="draw N replies per turn and keep the most on-topic; "
                        "1 (the default here) is the plain sampler, so an old "
                        "transcript stays comparable with a new one")
    p.add_argument("--rerank-lambda", type=float, default=1.0)
    p.add_argument("--out", default=None, help="also write the transcript here")
    args = p.parse_args(argv)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg, ck = load_chat_checkpoint(args.ckpt, device=device)
    if ck.get("kind") != "snnchat":
        raise SystemExit(
            f"{args.ckpt} is a research checkpoint with no conversation format. "
            f"Use scripts/chat.py, which falls back to continuation mode."
        )

    n_params = sum(p_.numel() for p_ in model.parameters())
    lines: list[str] = []

    def emit(text: str = "") -> None:
        print(text, flush=True)
        lines.append(text)

    emit(f"# {Path(args.ckpt).parent.name} -- demo battery")
    emit()
    emit(f"checkpoint : {args.ckpt}")
    emit(f"step       : {ck.get('step')}")
    emit(f"model      : {n_params:,} parameters, d={cfg.d_model}, K={cfg.n_layers}, "
         f"arch {cfg.arch}")
    for entry in (ck.get("description", {}).get("slow_pole_tau") or [])[:1]:
        emit(f"slow poles : tau {entry['tau_min']:g}..{entry['tau_max']:g} characters")
    emit(f"sampling   : temperature {args.temperature}, top_p {args.top_p}, "
         f"min_p {args.min_p}, seed {args.seed}")
    rerank = None
    if args.rerank > 1:
        from snnchat.rerank import RerankParams

        rerank = RerankParams(n=args.rerank, lam=args.rerank_lambda)
        emit(f"rerank     : {rerank.describe()}")
    else:
        emit("rerank     : off (one candidate per turn)")
    emit()

    t0 = time.perf_counter()
    total_chars = 0
    for i, (label, prompts) in enumerate(BATTERY):
        params = SamplingParams(
            temperature=args.temperature, top_p=args.top_p, min_p=args.min_p,
            max_new=args.max_new, seed=args.seed + i,
        )
        session = ChatSession(model, device=device, params=params, rerank=rerank)
        emit(f"## {label}")
        for prompt in prompts:
            reply = session.send(prompt)
            total_chars += len(reply)
            emit(f"    you > {prompt}")
            emit(f"    snn > {reply}")
        emit()

    dt = time.perf_counter() - t0
    emit(f"({total_chars:,} characters generated in {dt:.1f}s, "
         f"{total_chars / max(dt, 1e-9):.0f} char/s)")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
