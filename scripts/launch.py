"""Start a training run as a DETACHED OS process (risk R6).

v1 lost long runs because they were children of an interactive harness: when
the harness went away, so did the run.  01_reconnaissance.md rates that
likelihood "High -- happened in v1", so unattended runs are launched here and
never as a background task of whatever shell or agent happened to start them.

    python scripts/launch.py -- --corpus enwik8 --max_steps 20000
    python scripts/launch.py --run-name beta095 -- --beta 0.95
    python scripts/launch.py --script scripts/evaluate.py -- --ckpt ...

Everything after `--` is forwarded to the target script verbatim.  This process
writes the PID and returns immediately; the child keeps running with its stdout
and stderr appended to <out_dir>/<run_name>/stdout.log.

Windows specifics that are the whole point of the file:
  DETACHED_PROCESS         the child gets no console, so closing ours cannot
                           deliver a CTRL_CLOSE_EVENT to it
  CREATE_NEW_PROCESS_GROUP the child is not in our Ctrl-C process group
  stdin=DEVNULL            a detached child with an inherited stdin can block
                           forever on a read that will never be answered
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]

# From processthreadsapi.h; hard-coded because subprocess only exposes these
# constants on Windows and this file must at least import elsewhere.
CREATE_NEW_PROCESS_GROUP = 0x00000200
DETACHED_PROCESS = 0x00000008

_RUN_NAME_FLAGS = ("--run_name", "--run-name")
_OUT_DIR_FLAGS = ("--out_dir", "--out-dir")


def _scan_flag(argv: list[str], names: tuple[str, ...]) -> str | None:
    """Find `--flag value` or `--flag=value` in a forwarded argv."""
    for i, tok in enumerate(argv):
        if tok in names and i + 1 < len(argv):
            return argv[i + 1]
        for n in names:
            if tok.startswith(n + "="):
                return tok.split("=", 1)[1]
    return None


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--script", default="scripts/train.py",
                   help="target script, relative to the repo root")
    p.add_argument("--run-name", default=None,
                   help="run directory name; forwarded as --run_name if the "
                        "forwarded args do not already set one")
    p.add_argument("--out-dir", default=None,
                   help="parent of the run directory (default: the target's own "
                        "default, experiments/runs)")
    p.add_argument("--python", default=sys.executable,
                   help="interpreter to launch (default: this one)")
    p.add_argument("--dry-run", action="store_true",
                   help="print the command and the log path, launch nothing")
    return p


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser()
    if "--" in argv:
        # Explicit separator: unambiguous, and the only form that can forward a
        # flag whose name collides with one of ours.
        i = argv.index("--")
        args = parser.parse_args(argv[:i])
        forwarded = argv[i + 1:]
    else:
        args, forwarded = parser.parse_known_args(argv)

    script = (_REPO / args.script).resolve()
    if not script.exists():
        raise SystemExit(f"target script not found: {script}")
    # Only the trainer understands --run_name; injecting it into anything else
    # would make the child die instantly on an unrecognised argument, in a log
    # file nobody is watching.
    injectable = script.name == "train.py"

    # Precedence: what the child was actually told wins, because that is what
    # decides where the child writes.  A log directory that disagrees with the
    # run directory is worse than no log directory.
    forwarded_name = _scan_flag(forwarded, _RUN_NAME_FLAGS)
    if forwarded_name and args.run_name and forwarded_name != args.run_name:
        print(f"warning: --run-name {args.run_name!r} overridden by the "
              f"forwarded --run_name {forwarded_name!r}", file=sys.stderr)
    run_name = forwarded_name or args.run_name
    generated = run_name is None
    if generated:
        run_name = "run_" + datetime.now().strftime("%Y%m%d-%H%M%S")

    forwarded_out = _scan_flag(forwarded, _OUT_DIR_FLAGS)
    out_dir = forwarded_out or args.out_dir or "experiments/runs"

    child_args = list(forwarded)
    if injectable and forwarded_name is None:
        child_args += ["--run_name", run_name]
    if injectable and forwarded_out is None and args.out_dir:
        child_args += ["--out_dir", out_dir]

    run_dir = (Path(out_dir) if Path(out_dir).is_absolute() else _REPO / out_dir) / run_name
    log_path = run_dir / "stdout.log"
    cmd = [args.python, "-u", str(script), *child_args]

    if args.dry_run:
        print("command :", " ".join(cmd))
        print("cwd     :", _REPO)
        print("log     :", log_path)
        return 0

    run_dir.mkdir(parents=True, exist_ok=True)

    kwargs: dict = {}
    if os.name == "nt":
        kwargs["creationflags"] = CREATE_NEW_PROCESS_GROUP | DETACHED_PROCESS
    else:
        kwargs["start_new_session"] = True

    # Append, so a resumed run does not erase the record of the run it resumes.
    log_fh = open(log_path, "ab", buffering=0)
    try:
        log_fh.write(
            f"\n=== launch {datetime.now().isoformat(timespec='seconds')} "
            f"=== {' '.join(cmd)}\n".encode("utf-8")
        )
        proc = subprocess.Popen(
            cmd,
            cwd=str(_REPO),
            stdin=subprocess.DEVNULL,
            stdout=log_fh,
            stderr=subprocess.STDOUT,
            close_fds=True,
            **kwargs,
        )
    finally:
        # The child holds its own duplicated handle; ours is dead weight and
        # keeping it open would pin the file if this process outlives the call.
        log_fh.close()

    meta = {
        "pid": proc.pid,
        "command": cmd,
        "cwd": str(_REPO),
        "run_name": run_name,
        "run_name_generated": generated,
        "out_dir": str(out_dir),
        "log": str(log_path),
        "started": datetime.now().astimezone().isoformat(timespec="seconds"),
        "started_unix": time.time(),
        "detached": os.name == "nt",
    }
    (run_dir / "run.pid").write_text(str(proc.pid), encoding="utf-8")
    with open(run_dir / "launch.json", "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)

    print(f"pid      {proc.pid}")
    print(f"run_dir  {run_dir}")
    print(f"log      {log_path}")
    print(f"command  {' '.join(cmd)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
