"""Thin CLI in front of snn.train.Trainer.

Everything that changes an experiment lives in Config and is therefore hashed,
logged, and checkpointed.  This file adds exactly one thing that is *not* an
experimental variable -- `--resume` -- and otherwise forwards untouched.

    python scripts/train.py --corpus enwik8 --d_model 512 --n_layers 2
    python scripts/train.py --resume auto --run_name my_run
    python scripts/train.py --no-fused --no-cuda-graph         # reference path

Flag spelling: snn/config.py owns it.  Because "one argparse flag per Config
field" leaves the hyphen-vs-underscore choice open, and this script has to be
callable from scripts/launch.py without knowing which was chosen, the argv is
normalised to underscores and retried with hyphens if argparse rejects it.
That is ugly, and it is still better than two modules disagreeing silently
about what `--run_name` means.
"""

from __future__ import annotations

import contextlib
import io
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO / "src") not in sys.path:
    sys.path.insert(0, str(_REPO / "src"))

from snn.config import Config, config_from_args, seed_everything  # noqa: E402
from snn.train import Trainer  # noqa: E402


def _respell(argv: list[str], sep: str) -> list[str]:
    """Rewrite the *flag* part of every --long-option to use `sep`."""
    out = []
    for tok in argv:
        if tok.startswith("--") and len(tok) > 2:
            head, eq, tail = tok.partition("=")
            head = "--" + head[2:].replace("-", sep).replace("_", sep)
            out.append(head + eq + tail)
        else:
            out.append(tok)
    return out


def parse_config(argv: list[str]) -> Config:
    """Try the argv as typed, then all-underscore, then all-hyphen.

    As-typed comes first so that a parser mixing conventions (say, `--d_model`
    alongside a `--no-fused` store_false) still works when the caller spells it
    correctly; the respellings exist only so that launch.py and this script do
    not have to know which convention config.py picked.  Failed attempts are
    silenced, and the final error is raised from the argv the user actually
    wrote so the message points at their command line.
    """
    if any(a in ("-h", "--help") for a in argv):
        return config_from_args(argv)
    for candidate in (argv, _respell(argv, "_"), _respell(argv, "-")):
        try:
            with contextlib.redirect_stderr(io.StringIO()), \
                 contextlib.redirect_stdout(io.StringIO()):
                return config_from_args(candidate)
        except SystemExit:
            continue
    return config_from_args(argv)


def _take_resume(argv: list[str]) -> tuple[list[str], str | None]:
    """Strip --resume PATH|auto out of argv before Config sees it."""
    rest: list[str] = []
    resume: str | None = None
    i = 0
    while i < len(argv):
        tok = argv[i]
        if tok == "--resume":
            if i + 1 >= len(argv):
                raise SystemExit("--resume needs a value (a path, or 'auto')")
            resume = argv[i + 1]
            i += 2
            continue
        if tok.startswith("--resume="):
            resume = tok.split("=", 1)[1]
            i += 1
            continue
        rest.append(tok)
        i += 1
    return rest, resume


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    argv, resume = _take_resume(argv)

    cfg = parse_config(argv)
    seed_everything(cfg.seed, cfg.deterministic)

    trainer = Trainer(cfg)
    print(f"run_dir      {trainer.run_dir}", flush=True)
    print(f"config_hash  {trainer.config_hash}", flush=True)
    print(f"params       {trainer.n_params:,}", flush=True)

    if resume:
        path = (trainer.run_dir / "ckpt_last.pt") if resume == "auto" else Path(resume)
        if path.exists():
            trainer.load_checkpoint(path)
            print(f"resumed from {path} at step {trainer.global_step}", flush=True)
        elif resume == "auto":
            print("resume=auto: no checkpoint found, starting from step 0", flush=True)
        else:
            raise SystemExit(f"checkpoint not found: {path}")

    trainer.train()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
