"""
supervise.py
============

Restart-on-crash wrapper for multi-day ``snn_char_lm.py train`` runs on
Windows.

Why it exists: a days-long unattended run can die for reasons that have
nothing to do with training (driver reset, transient CUDA error, accidental
process kill). Training already checkpoints a full resumable state
(``--save-state``, refreshed on a wall-clock cadence and kept with a ``.bak``
rotation), so the missing piece is something that notices the death and
relaunches with ``--resume``. This does exactly that, with three safeguards:

  1. **Completion is detected by the ``[done]`` marker in the child's output,
     not its exit code.** TIER6.md documents a benign Windows quirk where a
     run prints ``[done]``, saves its checkpoint, and then exits non-zero
     during CUDA teardown -- an exit-code-gated supervisor would relaunch a
     finished run forever.
  2. **A crash-loop brake**: more than ``--max-restarts`` total restarts, or a
     crash within ``--min-run-seconds`` of launch (a config error, not a
     transient), stops the supervisor instead of burning days relaunching.
  3. **A stop sentinel**: create ``<state>.STOP`` (any content) to make the
     supervisor halt cleanly instead of relaunching after the current child
     exits/dies.

Usage::

    python supervise.py --state runs/main.state -- \
        python snn_char_lm.py train --data ... --save-state runs/main.state ...

Everything after ``--`` is the training command. On a relaunch, if the state
file exists, ``--resume <state>`` is appended (replacing any existing
--resume/--init-from pair in the command); the first launch runs the command
as given (which may itself already resume).
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time


def strip_flag(cmd: list, flag: str) -> list:
    """Remove ``flag <value>`` from a command list (if present)."""
    out = []
    skip = False
    for tok in cmd:
        if skip:
            skip = False
            continue
        if tok == flag:
            skip = True
            continue
        out.append(tok)
    return out


def run_once(cmd: list, log_path: str) -> tuple:
    """Run the child, tee its output to the log; return (saw_done, exit_code)."""
    saw_done = False
    with open(log_path, "a", encoding="utf-8") as log:
        log.write(f"\n===== launch {time.strftime('%Y-%m-%d %H:%M:%S')} =====\n"
                  f"{' '.join(cmd)}\n")
        log.flush()
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True,
                                encoding="utf-8", errors="replace", bufsize=1)
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            log.write(line)
            log.flush()
            if line.startswith("[done]"):
                saw_done = True
        proc.wait()
    return saw_done, proc.returncode


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--state", required=True,
                   help="the --save-state file the training command writes; "
                        "used for --resume on relaunch and for the .STOP sentinel")
    p.add_argument("--log", default=None,
                   help="supervisor log file (default: <state>.supervisor.log)")
    p.add_argument("--max-restarts", type=int, default=50)
    p.add_argument("--min-run-seconds", type=float, default=120.0,
                   help="a crash faster than this is treated as a config error "
                        "(no relaunch)")
    p.add_argument("--backoff-seconds", type=float, default=30.0)
    p.add_argument("cmd", nargs=argparse.REMAINDER,
                   help="-- followed by the training command")
    args = p.parse_args()

    cmd = args.cmd
    if cmd and cmd[0] == "--":
        cmd = cmd[1:]
    if not cmd:
        p.error("no training command given (put it after --)")
    log_path = args.log or (args.state + ".supervisor.log")
    stop_sentinel = args.state + ".STOP"

    restarts = 0
    while True:
        if os.path.exists(stop_sentinel):
            print(f"[supervise] stop sentinel {stop_sentinel} present; halting.")
            return
        t0 = time.time()
        saw_done, code = run_once(cmd, log_path)
        elapsed = time.time() - t0
        if saw_done:
            print(f"[supervise] run completed ([done] seen; exit {code} "
                  f"{'benign teardown quirk' if code else 'clean'}); halting.")
            return
        if os.path.exists(stop_sentinel):
            print(f"[supervise] child died but stop sentinel present; halting.")
            return
        if elapsed < args.min_run_seconds:
            print(f"[supervise] child died after only {elapsed:.0f}s "
                  f"(exit {code}) -- likely a config error, NOT relaunching.")
            sys.exit(2)
        restarts += 1
        if restarts > args.max_restarts:
            print(f"[supervise] exceeded --max-restarts {args.max_restarts}; "
                  f"halting.")
            sys.exit(3)
        # Relaunch, resuming from the saved state if it exists.
        if os.path.exists(args.state):
            cmd = strip_flag(strip_flag(cmd, "--resume"), "--init-from")
            cmd = cmd + ["--resume", args.state]
        print(f"[supervise] child died (exit {code}) after {elapsed/60:.1f} min; "
              f"restart {restarts}/{args.max_restarts} in "
              f"{args.backoff_seconds:.0f}s...")
        time.sleep(args.backoff_seconds)


if __name__ == "__main__":
    main()
