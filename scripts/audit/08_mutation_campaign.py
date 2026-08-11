"""Mutation-test the R10 gate, as a committed artifact rather than a prose claim.

Why this exists
---------------
`02_baseline_report.md` §2.1 says the R10 gate was mutation-tested, "21 plausible
mis-derivations ... 21 / 21 caught". That is the single most important claim in
the project -- R10 is the risk whose whole character is that a wrong backward
still trains and still produces a publishable number, so the gate is what stands
between this repository and a plausible wrong result.

It was not, however, reproducible. Phase 3 found the count stated three different
ways in three different places:

    docs/reports/02_baseline_report.md      21 mutations
    src/snn/neuron.py                       sixteen
    tests/test_neuron_equivalence.py        eleven

Those are almost certainly honest snapshots of a campaign that grew, and the test
file's eleven describes a distinct earlier round (the one that found holes in the
tests themselves). But a reader cannot tell that from the repository, there is no
artifact to check any of them against, and this project's own standard is that
every quoted number is backed by committed machine-readable output. A claim about
the gate that guards against unfalsifiable results should not itself be
unfalsifiable.

So the campaign is re-run here, by a script, and its output is committed.

Second reason, forward-looking
------------------------------
`EXP_002` priced the §4.6-admitted neurons and found their GPU cost negligible --
one kernel per timestep, identical memory, <=1.10x wall-clock. Their real cost is
that **each needs its own mutation-tested gradient gate** before it may run. This
harness is that cost, paid once and reusable: adding a candidate neuron's
mutations is a new entry in `MUTATIONS`, not a new campaign.

How it works
------------
Each mutation is a literal source-text substitution into `snn/kernels.py`,
`snn/surrogate.py` or `snn/twocomp.py`, applied one at a time, after which the real
R10 gate for that neuron is run unmodified. A mutation is CAUGHT if the gate fails
and ESCAPED if it passes. An escaped mutation is a hole in the gate and is reported
as one.

Each mutation names the gate it is checked against, and every gate in the plan is
required to pass on clean source first. Both matter now that there are two neurons:
a `twocomp.py` mutation checked against the LIF gate would be reported as escaped
when nothing had actually escaped, and a gate that is broken on unmutated source
would report everything as caught while proving nothing.

The working tree is asserted clean before anything is written, every file is
restored with `git checkout --` in a `finally`, and cleanliness is asserted again
at the end -- so an interrupted run cannot leave a mutated kernel on disk
pretending to be the real one.

NOTHING ELSE MAY IMPORT `snn` WHILE THIS RUNS
---------------------------------------------
This script deliberately writes wrong code into `src/snn/` for seconds at a time.
Python reads a module's source once, at import, so **any other process that starts
during a mutation window imports the mutated kernel and keeps it for its whole
lifetime** -- silently, because nothing about a running process reveals which
version of the source it loaded.

That is not hypothetical. On 2026-08-02 this campaign was run concurrently with
Phase 3's depth ladder. `depth_K8_s0` started at 01:28:01, inside the window, and
trained 7 750 steps against mutation M15 (`>` instead of `>=` in the forward
kernel) before the contamination was noticed. The run was discarded and re-run.
It was caught only because the unrelated `EXP_001` F1 self-check -- "the probe's
k = L point must reproduce the committed bpc" -- failed on a different checkpoint
and the residual was chased instead of tolerated.

So: a startup guard refuses to run when another Python process is already alive,
and a lock file is left behind while mutations are applied. Neither is airtight
against a determined caller; both are enough to stop the accident that actually
happened.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
KERNELS = REPO / "src" / "snn" / "kernels.py"
SURROGATE = REPO / "src" / "snn" / "surrogate.py"
TWOCOMP = REPO / "src" / "snn" / "twocomp.py"
TWOCOMP_DETACH = REPO / "src" / "snn" / "twocomp_detach.py"
GATE = "tests/test_neuron_equivalence.py"
GATE_TC = "tests/test_twocomp_equivalence.py"
GATE_TCD = "tests/test_twocomp_detach_equivalence.py"

# Each entry: (id, description, file, find, replace) with an OPTIONAL sixth
# element naming the gate to run; it defaults to `GATE`, the LIF one.
# `find` must appear EXACTLY ONCE in the file -- asserted, so a refactor that
# moves the text makes the campaign fail loudly rather than silently skip a
# mutation and report a clean sweep.
#
# The gate column exists because EXP_004 added a second neuron. Mutating
# `twocomp.py` and then running the LIF gate would report every one of those
# mutations ESCAPED, because that gate never imports the mutated module -- and a
# mutation checked against a gate that cannot see it is worse than no mutation,
# since it looks like coverage. `surrogate.py`'s mutations keep the LIF gate: both
# neurons share that file, and the LIF gate is the one with the analytic-surrogate
# premise test in it.
MUTATIONS: list[tuple] = [
    # ---- backward: the reset derivative -----------------------------------
    ("M01", "drop the -v_pre*sg term from the hard-reset derivative", KERNELS,
     '"hard": "(T(1) - s) - v_pre * sg",', '"hard": "(T(1) - s)",'),
    ("M02", "sign-flip the -v_pre*sg term in the hard-reset derivative", KERNELS,
     '"hard": "(T(1) - s) - v_pre * sg",', '"hard": "(T(1) - s) + v_pre * sg",'),
    ("M03", "drop thr from the soft-reset derivative", KERNELS,
     '"soft": "T(1) - thr * sg",', '"soft": "T(1) - sg",'),
    ("M04", "give detached the derivative of hard", KERNELS,
     '"detached": "T(1) - s",', '"detached": "(T(1) - s) - v_pre * sg",'),
    ("M05", "give hard the derivative of detached", KERNELS,
     '"hard": "(T(1) - s) - v_pre * sg",', '"hard": "T(1) - s",'),
    ("M06", "none reset derivative 1 -> 0", KERNELS,
     '"none": "T(1)",\n}', '"none": "T(0)",\n}'),
    # ---- backward: the chain ----------------------------------------------
    ("M07", "drop beta from grad_v_prev", KERNELS,
     "grad_v_prev = beta * g;", "grad_v_prev = g;"),
    ("M08", "grad_v_prev divides by beta instead of multiplying", KERNELS,
     "grad_v_prev = beta * g;", "grad_v_prev = g / beta;"),
    ("M09", "drop the grad_spike*sg term", KERNELS,
     "const T g  = grad_v_next * dv + grad_spike * sg;",
     "const T g  = grad_v_next * dv;"),
    ("M10", "drop the grad_v_next*dv term", KERNELS,
     "const T g  = grad_v_next * dv + grad_spike * sg;",
     "const T g  = grad_spike * sg;"),
    ("M11", "straight-through: sg = 1 instead of the surrogate", KERNELS,
     "    const T sg = {atan_grad};", "    const T sg = T(1);"),
    ("M12", "omit the threshold shift in x", KERNELS,
     "const T x  = v_pre - thr;", "const T x  = v_pre;"),
    ("M13", "> instead of >= in the BACKWARD kernel only", KERNELS,
     "const T s  = (x >= T(0)) ? T(1) : T(0);",
     "const T s  = (x > T(0)) ? T(1) : T(0);"),
    ("M14", "backward compares x against thr instead of 0", KERNELS,
     "const T s  = (x >= T(0)) ? T(1) : T(0);",
     "const T s  = (x >= thr) ? T(1) : T(0);"),
    # ---- forward -----------------------------------------------------------
    ("M15", "> instead of >= in the FORWARD kernel", KERNELS,
     "    const T s = (v >= thr) ? T(1) : T(0);",
     "    const T s = (v > thr) ? T(1) : T(0);"),
    ("M16", "hard reset does not reset", KERNELS,
     '"hard": "v * (T(1) - s)",', '"hard": "v",'),
    ("M17", "soft reset subtracts the spike, not thr*spike", KERNELS,
     '"soft": "v - thr * s",', '"soft": "v - s",'),
    ("M18", "let NVRTC contract the membrane update into an FMA", KERNELS,
     '_VPRE = (\n'
     '    "    const float vp = __fadd_rn(__fmul_rn(static_cast<float>(v_prev),\\n"\n'
     '    "                                         static_cast<float>(beta)),\\n"\n'
     '    "                               static_cast<float>(cur));\\n"\n'
     '    "    const T v = static_cast<T>(vp);\\n"\n'
     ')',
     '_VPRE = (\n'
     '    "    const T v = v_prev * beta + cur;\\n"\n'
     ')'),
    ("M19", "forward decays the current instead of the membrane", KERNELS,
     '    "    const float vp = __fadd_rn(__fmul_rn(static_cast<float>(v_prev),\\n"\n'
     '    "                                         static_cast<float>(beta)),\\n"\n'
     '    "                               static_cast<float>(cur));\\n"',
     '    "    const float vp = __fadd_rn(__fmul_rn(static_cast<float>(cur),\\n"\n'
     '    "                                         static_cast<float>(beta)),\\n"\n'
     '    "                               static_cast<float>(v_prev));\\n"'),
    # ---- the surrogate itself ---------------------------------------------
    ("M20", "pi instead of pi/2 in the CUDA surrogate", SURROGATE,
     'ATAN_CUDA_HALF_PI = "1.57079632679489661923"',
     'ATAN_CUDA_HALF_PI = "3.14159265358979323846"'),
    ("M21", "2x surrogate gain in the CUDA source", SURROGATE,
     '"(T(0.5) * alpha / (T(1) + (T({half_pi}) * alpha * x) * (T({half_pi}) * alpha * x)))"',
     '"(alpha / (T(1) + (T({half_pi}) * alpha * x) * (T({half_pi}) * alpha * x)))"'),
    ("M22", "CUDA surrogate loses its square in the denominator", SURROGATE,
     '"(T(0.5) * alpha / (T(1) + (T({half_pi}) * alpha * x) * (T({half_pi}) * alpha * x)))"',
     '"(T(0.5) * alpha / (T(1) + (T({half_pi}) * alpha * x)))"'),
    ("M23", "eager atan_grad loses its factor of 1/2", SURROGATE,
     "return (0.5 * alpha) / (1.0 + u * u)", "return alpha / (1.0 + u * u)"),
    ("M24", "eager atan_value drops the 1/pi", SURROGATE,
     "return torch.atan(x * (_HALF_PI * alpha)) * _INV_PI + 0.5",
     "return torch.atan(x * (_HALF_PI * alpha)) + 0.5"),
    ("M25", "atan_spike reverts to the unparenthesised straight-through form", SURROGATE,
     "return s_hard.detach() + (sv - sv.detach())",
     "return s_hard.detach() + sv - sv.detach()"),

    # ======================================================================
    # EXP_004: the two-compartment neuron. Checked against its own gate.
    # ======================================================================
    # Adding these was one list entry per mutation rather than a new campaign,
    # which is exactly the reduction in per-candidate cost `EXP_002` §8.6 said the
    # harness would buy. T01 is the most important of them: it is the mutation
    # that turns this neuron back into `EXP_002`'s N2 prototype, whose slow
    # compartment DID reset -- a change that leaves every kernel-count and
    # wall-clock number identical and only moves the science.

    # ---- forward: the reset rule ------------------------------------------
    ("T01", "the slow compartment resets too (reverts to EXP_002's N2)", TWOCOMP,
     "    vs_next = vs;", "    vs_next = vs * (T(1) - sp);", GATE_TC),
    ("T02", "the fast compartment does not reset", TWOCOMP,
     "    vf_next = vf * (T(1) - sp);", "    vf_next = vf;", GATE_TC),
    ("T03", "> instead of >= in the FORWARD threshold", TWOCOMP,
     "const T sp = (vmix >= thr) ? T(1) : T(0);",
     "const T sp = (vmix > thr) ? T(1) : T(0);", GATE_TC),
    # ---- forward: the mix and the two poles --------------------------------
    ("T04", "the mix drops w_c and adds the compartments raw", TWOCOMP,
     '"    const float vmp = __fadd_rn(vfp, __fmul_rn(static_cast<float>(w_c), vsp));\\n"',
     '"    const float vmp = __fadd_rn(vfp, vsp);\\n"', GATE_TC),
    ("T05", "let NVRTC contract the MIX into an FMA", TWOCOMP,
     '"    const float vmp = __fadd_rn(vfp, __fmul_rn(static_cast<float>(w_c), vsp));\\n"',
     '"    const float vmp = vfp + static_cast<float>(w_c) * vsp;\\n"', GATE_TC),
    ("T06", "the two decays are swapped between the compartments", TWOCOMP,
     '"    const float vsp = __fadd_rn(__fmul_rn(static_cast<float>(vs_prev),\\n"\n'
     '    "                                          static_cast<float>(beta_s_c)),\\n"',
     '"    const float vsp = __fadd_rn(__fmul_rn(static_cast<float>(vs_prev),\\n"\n'
     '    "                                          static_cast<float>(beta_f)),\\n"', GATE_TC),
    ("T07", "the slow pole decays the current instead of the membrane", TWOCOMP,
     '"    const float vsp = __fadd_rn(__fmul_rn(static_cast<float>(vs_prev),\\n"\n'
     '    "                                          static_cast<float>(beta_s_c)),\\n"\n'
     '    "                                static_cast<float>(cur));\\n"',
     '"    const float vsp = __fadd_rn(__fmul_rn(static_cast<float>(cur),\\n"\n'
     '    "                                          static_cast<float>(beta_s_c)),\\n"\n'
     '    "                                static_cast<float>(vs_prev));\\n"', GATE_TC),
    ("T08", "v_pre reports the fast compartment instead of the mix", TWOCOMP,
     "    v_pre   = vmix;", "    v_pre   = vf;", GATE_TC),

    # ---- backward: the reset derivative and the chain ----------------------
    ("T09", "drop the -vf*sgd term from the fast-pole derivative", TWOCOMP,
     "const T dv  = (T(1) - sh) - vf * sgd;", "const T dv  = (T(1) - sh);", GATE_TC),
    ("T10", "backward recovers vf with the wrong sign", TWOCOMP,
     "const T vf  = v_pre - w_c * vs;", "const T vf  = v_pre + w_c * vs;", GATE_TC),
    ("T11", "sign-flip the mix term in the slow adjoint", TWOCOMP,
     "const T gs  = grad_vs_next + w_c * gv;",
     "const T gs  = grad_vs_next - w_c * gv;", GATE_TC),
    ("T12", "current is treated as entering only the fast compartment", TWOCOMP,
     "    grad_cur     = gf + gs;", "    grad_cur     = gf;", GATE_TC),
    ("T13", "drop beta_s from the slow adjoint chain", TWOCOMP,
     "    grad_vs_prev = beta_s_c * gs;", "    grad_vs_prev = gs;", GATE_TC),
    ("T14", "drop beta_f from the fast adjoint chain", TWOCOMP,
     "    grad_vf_prev = beta_f * gf;", "    grad_vf_prev = gf;", GATE_TC),
    ("T15", "> instead of >= in the BACKWARD kernel only", TWOCOMP,
     "const T sh  = (x >= T(0)) ? T(1) : T(0);",
     "const T sh  = (x > T(0)) ? T(1) : T(0);", GATE_TC),
    ("T16", "drop the grad_spike*sgd term from the fast adjoint", TWOCOMP,
     "const T gf  = grad_vf_next * dv + grad_spike * sgd;",
     "const T gf  = grad_vf_next * dv;", GATE_TC),
    # ---- backward: the two per-channel gradients ---------------------------
    ("T17", "grad_w is accumulated from the wrong adjoint", TWOCOMP,
     "    gv_elem      = gv;", "    gv_elem      = gf;", GATE_TC),
    ("T18", "grad_beta_s is accumulated from the wrong adjoint", TWOCOMP,
     "    gvs_elem     = gs;", "    gvs_elem     = gv;", GATE_TC),
    ("T19", "grad_beta_s uses vs_t instead of vs_{t-1} (the shift is dropped)", TWOCOMP,
     "grad_beta_s = (gvs * vs_prev).sum(dim=(0, 1)).reshape(1, d)",
     "grad_beta_s = (gvs * vs_seq).sum(dim=(0, 1)).reshape(1, d)", GATE_TC),
    ("T20", "the two per-channel gradients are returned swapped", TWOCOMP,
     "return grad_cur, grad_v0, grad_w, grad_beta_s, None, None, None",
     "return grad_cur, grad_v0, grad_beta_s, grad_w, None, None, None", GATE_TC),
    # ---- the eager reference itself ----------------------------------------
    # The reference is the ground truth for everything above, so a silent error in
    # it would make the whole gate agree on the wrong function. Mutating it checks
    # that the gate's independent legs (the spec transcription, float64, gradcheck,
    # and the w=0 nesting against the committed LIF) actually bite.
    ("T21", "the eager reference resets the slow compartment", TWOCOMP,
     "        vf = vf * (1.0 - s)\n"
     "        # vs is NOT reset -- that is the whole point of the candidate.",
     "        vf = vf * (1.0 - s)\n"
     "        vs = vs * (1.0 - s)", GATE_TC),
    ("T22", "the eager reference mixes before decaying the fast pole", TWOCOMP,
     "        s = atan_spike(vf + w * vs - thr, alpha)",
     "        s = atan_spike(vf * w + vs - thr, alpha)", GATE_TC),

    # ======================================================================
    # EXP_017: the detached-reset two-compartment neuron. Its own file, its own
    # gate. Third neuron, third `MUTATIONS` block and no third campaign -- which
    # is the reusability `EXP_002` §8.6 predicted, now observed twice.
    #
    # THE MOST IMPORTANT ONES ARE D01 AND D10, and they are important for
    # opposite reasons. D01 puts the adopted arm's unbounded `-vf*sgd` term back
    # into the kernel: the arm's ENTIRE claim is that this term is absent, the
    # mutation restores it, and every wall-clock and kernel-count number stays
    # identical -- only the science moves. D10 removes the `.detach()` from the
    # eager reference, which is the arm's single changed token; if the gate did
    # not catch that, its "fused vs eager" legs would be comparing the adopted
    # arm against itself and would pass while measuring nothing.
    # ======================================================================

    # ---- backward: the reset derivative, i.e. the whole arm ----------------
    ("D01", "restore the adopted arm's unbounded reset term", TWOCOMP_DETACH,
     "    const T dv  = T(1) - sh;",
     "    const T dv  = (T(1) - sh) - v_pre * sgd;", GATE_TCD),
    ("D02", "the backward forgets the reset entirely (reverts to 'none')",
     TWOCOMP_DETACH,
     "    const T dv  = T(1) - sh;", "    const T dv  = T(1);", GATE_TCD),
    ("D03", "dL/dv regains a reset path through the mixed membrane",
     TWOCOMP_DETACH,
     "    const T gv  = sgd * grad_spike;",
     "    const T gv  = sgd * (grad_spike - grad_vf_next * v_pre);", GATE_TCD),
    ("D04", "dL/dv drops the surrogate, becoming straight-through",
     TWOCOMP_DETACH,
     "    const T gv  = sgd * grad_spike;", "    const T gv  = grad_spike;",
     GATE_TCD),
    # ---- backward: the chain ----------------------------------------------
    ("D05", "sign-flip the mix term in the slow adjoint", TWOCOMP_DETACH,
     "    const T gs  = grad_vs_next + w_c * gv;",
     "    const T gs  = grad_vs_next - w_c * gv;", GATE_TCD),
    ("D06", "drop the grad_spike*sgd term from the fast adjoint", TWOCOMP_DETACH,
     "    const T gf  = grad_vf_next * dv + grad_spike * sgd;",
     "    const T gf  = grad_vf_next * dv;", GATE_TCD),
    ("D07", "current is treated as entering only the fast compartment",
     TWOCOMP_DETACH,
     "    grad_cur     = gf + gs;", "    grad_cur     = gf;", GATE_TCD),
    ("D08", "drop beta_f from the fast adjoint chain", TWOCOMP_DETACH,
     "    grad_vf_prev = beta_f * gf;", "    grad_vf_prev = gf;", GATE_TCD),
    ("D09", "> instead of >= in the BACKWARD kernel only", TWOCOMP_DETACH,
     "    const T sh  = (x >= T(0)) ? T(1) : T(0);",
     "    const T sh  = (x > T(0)) ? T(1) : T(0);", GATE_TCD),
    # ---- the eager reference itself ----------------------------------------
    ("D10", "the eager reference does not detach (it IS the adopted arm)",
     TWOCOMP_DETACH,
     "        vf = vf * (1.0 - s.detach())",
     "        vf = vf * (1.0 - s)", GATE_TCD),
    ("D11", "grad_beta_s uses vs_t instead of vs_{t-1} (the shift is dropped)",
     TWOCOMP_DETACH,
     "        grad_beta_s = (gvs * vs_prev).sum(dim=(0, 1)).reshape(1, d)",
     "        grad_beta_s = (gvs * vs_seq).sum(dim=(0, 1)).reshape(1, d)",
     GATE_TCD),
    ("D12", "the two per-channel gradients are returned swapped", TWOCOMP_DETACH,
     "        return grad_cur, grad_v0, grad_w, grad_beta_s, None, None, None",
     "        return grad_cur, grad_v0, grad_beta_s, grad_w, None, None, None",
     GATE_TCD),
]


LOCKFILE = REPO / "src" / "snn" / ".MUTATION_CAMPAIGN_RUNNING"


def _normalise(m: tuple) -> tuple[str, str, Path, str, str, str]:
    """Fill in the default gate for the 5-tuple entries written before EXP_004."""
    mid, desc, path, find, repl = m[:5]
    return mid, desc, path, find, repl, (m[5] if len(m) > 5 else GATE)


def other_python_processes() -> list[str]:
    """Other live python.exe PIDs. See the module docstring for why this matters."""
    try:
        out = subprocess.run(
            ["tasklist", "/FI", "IMAGENAME eq python.exe", "/FO", "CSV", "/NH"],
            capture_output=True, text=True, timeout=30).stdout
    except Exception:
        return []
    mine = str(subprocess.os.getpid())
    pids = []
    for line in out.splitlines():
        parts = [p.strip('"') for p in line.split(",")]
        if len(parts) > 1 and parts[1].isdigit() and parts[1] != mine:
            pids.append(parts[1])
    return pids


def git(*args: str) -> str:
    out = subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True)
    if out.returncode != 0:
        raise SystemExit(f"git {' '.join(args)} failed: {out.stderr.strip()}")
    return out.stdout


def tree_is_clean() -> bool:
    return git("status", "--porcelain", "--", "src").strip() == ""


def run_gate(timeout_s: int, gate: str = GATE) -> tuple[bool, str]:
    """Run the real R10 gate unmodified. Returns (passed, tail-of-output)."""
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", gate, "-x", "-q", "--no-header"],
        cwd=REPO, capture_output=True, text=True, timeout=timeout_s,
    )
    tail = "\n".join(
        (proc.stdout + proc.stderr).strip().splitlines()[-6:])
    return proc.returncode == 0, tail


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="docs/reports/data/audit_08_mutation_campaign.json")
    ap.add_argument("--only", default="", help="comma-separated mutation ids")
    ap.add_argument("--timeout", type=int, default=900)
    ap.add_argument("--force-concurrent", action="store_true",
                    help="run even if other python processes are alive; they may "
                         "silently import a mutated kernel (see module docstring)")
    args = ap.parse_args(argv)

    others = other_python_processes()
    if others and not args.force_concurrent:
        raise SystemExit(
            f"{len(others)} other python process(es) are running (PIDs "
            f"{', '.join(others[:8])}). This script writes wrong code into "
            f"src/snn/ for seconds at a time, and anything importing `snn` during "
            f"that window silently picks it up for its whole lifetime -- which has "
            f"already cost this project one discarded training run. Stop them, or "
            f"pass --force-concurrent if you are certain none of them touch `snn`.")

    if not tree_is_clean():
        raise SystemExit(
            "src/ has uncommitted changes. This script writes to the real source "
            "files and restores them with `git checkout --`, which would discard "
            "your edits. Commit or stash first.")

    wanted = {m.strip() for m in args.only.split(",") if m.strip()}
    plan = [_normalise(m) for m in MUTATIONS if not wanted or m[0] in wanted]

    # Every `find` string must be present exactly once, checked up front so a
    # refactor cannot silently turn a mutation into a no-op that then reports as
    # "caught" because the gate passed for entirely unrelated reasons.
    for mid, desc, path, find, _repl, _gate in plan:
        n = path.read_text(encoding="utf-8").count(find)
        if n != 1:
            raise SystemExit(
                f"{mid}: anchor text appears {n} times in {path.name}, expected "
                f"exactly 1. The source moved; fix the mutation before trusting "
                f"this campaign.")

    # Every gate in the plan must pass on CLEAN source before anything is mutated.
    # Checking only the LIF gate here would let a broken two-compartment gate
    # report all 22 of its mutations as "caught" -- a gate that fails on unmutated
    # source catches everything and proves nothing.
    gates = sorted({g for *_rest, g in plan})
    baselines: dict[str, dict] = {}
    for gate in gates:
        print(f"baseline: {gate} must PASS on unmutated source", flush=True)
        t0 = time.perf_counter()
        gate_pass, gate_tail = run_gate(args.timeout, gate)
        dt = time.perf_counter() - t0
        print(f"  {'PASS' if gate_pass else 'FAIL'} in {dt:.0f}s")
        if not gate_pass:
            raise SystemExit(f"{gate} does not pass on clean source:\n{gate_tail}")
        baselines[gate] = {"pass": gate_pass, "seconds": round(dt, 1)}

    results: dict = {
        "audit": "08_mutation_campaign",
        "gates": baselines,
        "baseline_pass": all(b["pass"] for b in baselines.values()),
        "baseline_seconds": round(sum(b["seconds"] for b in baselines.values()), 1),
        "n_mutations": len(plan),
        "mutations": {},
    }
    escaped: list[str] = []

    for i, (mid, desc, path, find, repl, gate) in enumerate(plan, 1):
        original = path.read_text(encoding="utf-8")
        print(f"[{i}/{len(plan)}] {mid}  {desc}", flush=True)
        try:
            LOCKFILE.write_text(
                f"{mid}: {desc}\nsrc/snn is MUTATED right now. Do not import snn.\n",
                encoding="utf-8")
            path.write_text(original.replace(find, repl), encoding="utf-8")
            t0 = time.perf_counter()
            passed, tail = run_gate(args.timeout, gate)
            dt = time.perf_counter() - t0
        except subprocess.TimeoutExpired:
            passed, tail, dt = True, "TIMEOUT", float(args.timeout)
        finally:
            # Restore from git rather than from the in-memory copy: it is the
            # stronger guarantee, and it is also what proves the restore worked.
            git("checkout", "--", str(path.relative_to(REPO)))
            LOCKFILE.unlink(missing_ok=True)

        caught = not passed
        if not caught:
            escaped.append(mid)
        results["mutations"][mid] = {
            "description": desc,
            "file": path.name,
            "gate": gate,
            "caught": caught,
            "gate_seconds": round(dt, 1),
            "gate_output_tail": tail,
        }
        print(f"      {'CAUGHT' if caught else '*** ESCAPED ***'}  ({dt:.0f}s)")

    if not tree_is_clean():
        raise SystemExit("src/ is dirty after the campaign; restore failed")

    n_caught = sum(1 for r in results["mutations"].values() if r["caught"])
    results["n_caught"] = n_caught
    results["escaped"] = escaped
    results["all_caught"] = not escaped
    results["source_restored_clean"] = True

    out = REPO / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print("\n" + "=" * 74)
    print(f"MUTATION CAMPAIGN: {n_caught} / {len(plan)} caught")
    if escaped:
        print(f"  ESCAPED (holes in the R10 gate): {escaped}")
        for mid in escaped:
            print(f"    {mid}  {results['mutations'][mid]['description']}")
    else:
        print("  no mutation survived the gate")
    print(f"  source tree restored clean: yes")
    print("=" * 74)
    print(f"WROTE {args.out}")
    return 1 if escaped else 0


if __name__ == "__main__":
    raise SystemExit(main())
