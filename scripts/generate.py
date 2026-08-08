"""Autoregressively sample text from a trained checkpoint.

    python scripts/generate.py --ckpt experiments/runs/<run>/ckpt_best.pt \\
        --prompt "The " --steps 500

Not part of the research protocol -- there is no gate that depends on this
script and no number it prints is a reported figure. It exists purely so a
checkpoint can be read by eye. The configuration (arch, d_model, vocab, ...)
comes from the checkpoint, same as scripts/evaluate.py, so what you sample
from is guaranteed to be the model that was actually trained.
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO / "src") not in sys.path:
    sys.path.insert(0, str(_REPO / "src"))

from snn.config import Config, seed_everything  # noqa: E402
from snn.data import Corpus  # noqa: E402
from snn.model import build_model  # noqa: E402


def _config_from_checkpoint(ck: dict) -> Config:
    raw = ck.get("config")
    if not raw:
        raise SystemExit("checkpoint has no 'config' block; cannot rebuild the model")
    known = {f.name for f in dataclasses.fields(Config)}
    unknown = sorted(set(raw) - known)
    if unknown:
        print(f"warning: ignoring unknown config fields {unknown}", file=sys.stderr)
    return Config(**{k: v for k, v in raw.items() if k in known})


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", required=True, help="path to a .pt written by Trainer")
    p.add_argument("--prompt", default="\n", help="seed text (UTF-8)")
    p.add_argument("--steps", type=int, default=500, help="characters to generate")
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=0, help="0 = disabled")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default=None)
    p.add_argument("--data-dir", default=None)
    p.add_argument("--no-fused", action="store_true",
                   help="force the eager reference LIF scan")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    ckpt_path = Path(args.ckpt)
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    cfg = _config_from_checkpoint(ck)
    if args.device:
        cfg.device = args.device
    if args.data_dir:
        cfg.data_dir = args.data_dir
    if args.no_fused:
        cfg.fused = False

    seed_everything(args.seed, cfg.deterministic)
    device = torch.device(cfg.device)
    corpus = Corpus(cfg.corpus, cfg.data_dir)
    if int(cfg.vocab_size) != int(corpus.vocab_size):
        raise SystemExit(
            f"checkpoint vocab_size={cfg.vocab_size} but corpus reports "
            f"{corpus.vocab_size}; refusing to sample from a mismatched model"
        )

    model = build_model(cfg).to(device)
    model.load_state_dict(ck["model"])
    model.eval()

    prompt_bytes = np.frombuffer(args.prompt.encode("utf-8"), dtype=np.uint8)
    prompt_ids = corpus.stoi[prompt_bytes]
    if (prompt_ids == 255).any():  # UNUSED_ID sentinel -- see snn.data
        bad = prompt_bytes[prompt_ids == 255]
        raise SystemExit(
            f"--prompt contains byte(s) {sorted(set(int(b) for b in bad))} that "
            f"never occur in {cfg.corpus!r} and have no vocab slot"
        )

    idx = torch.from_numpy(prompt_ids.astype(np.int64)).unsqueeze(0).to(device)
    generated: list[int] = []

    with torch.no_grad():
        logits, state, _ = model(idx, state=None)
        next_logits = logits[:, -1, :]

        for _ in range(args.steps):
            scaled = next_logits / max(args.temperature, 1e-6)
            if args.top_k > 0:
                v, _ = torch.topk(scaled, min(args.top_k, scaled.shape[-1]))
                scaled = scaled.masked_fill(scaled < v[:, [-1]], float("-inf"))
            probs = F.softmax(scaled, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)  # [1, 1]
            generated.append(int(next_id.item()))

            logits, state, _ = model(next_id, state=state)
            next_logits = logits[:, -1, :]

    text = corpus.decode(np.array(generated, dtype=np.int64)).decode("utf-8", errors="replace")
    print(f"checkpoint  {ckpt_path}  (step {ck.get('step')}, arch {cfg.arch})")
    print(f"prompt      {args.prompt!r}")
    print("-" * 70)
    print(args.prompt + text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
