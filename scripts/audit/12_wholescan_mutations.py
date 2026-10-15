"""Mutation-test the whole-scan kernels' gates: the WRAPPER, and the functors THROUGH it.

**Outside the Phase 1-5 protocol.** This audits a performance change
(`src/snn/wholescan.py`, `docs/perf/WHOLESCAN.md`) whose only permitted effect is
none at all, and its artifact therefore lives under `docs/perf/data/`, not under
the protocol's `docs/reports/data/`.

Why a second campaign rather than new entries in `08_mutation_campaign.py`
-------------------------------------------------------------------------
`08` produced committed evidence (59/59, `audit_08_mutation_campaign.json`) and is
not edited. It also cannot express what is needed here. Its mutations are of the
NEURON -- a dropped term, a flipped sign -- and its gates ask "is the fused path
right?". The whole-scan kernels paste the same functor text, so they inherit every
one of those mutations for free, and what is new about them is the one thing `08`
has no entry for: a thin time-loop wrapper that does **indexing and nothing else**.
What a wrapper gets wrong is *which element goes where* -- the time stride, the
`t == 0` initial-state branch, the slow half's offset into `[B, 2d]`, which product
feeds which reduction, whether the saved tensors were written at all -- plus the
single line that makes the path bit-identical in the first place, the `noinline`
contraction barrier.

Two blocks, asking two different questions:

  W  wrapper and contract mutations, written into `src/snn/wholescan.py`, checked
     against `tests/test_wholescan_equivalence.py` -- the BITWISE gate. A wrong
     index is not a one-ulp defect, but the barrier mutations (W12-W14, W27) are,
     and they are the ones no tolerance-based gate in the repository can see.

  F  every mutation of campaign `08` that can reach the whole-scan path, imported
     from `08` rather than retyped, checked against that neuron's own R10 gate
     with its `[jiterator]` parametrisation deselected (`-k "not jiterator"`).
     This is the claim "functor mutations flow into both paths automatically"
     turned from a sentence into an artifact: with the per-timestep path
     deselected entirely, the mutation must still be caught, by the new kernels
     or by a test that belongs to neither implementation. Four of
     `08`'s entries are excluded and recorded as such: they mutate the Python of
     the per-timestep autograd Functions, which the whole-scan path does not
     execute. Their whole-scan counterparts are W04 / W09 / W08.

     Note what the bitwise gate is NOT asked to do. A functor mutation changes
     both implementations identically, so old-vs-new stays bit-equal and
     `test_wholescan_equivalence.py` passes -- correctly. "Are they the same?" and
     "are they right?" are different questions with different gates, and a
     campaign that scored F-mutations against the bitwise gate would report 55
     escapes that are not holes.

THE FIRST RUN OF THIS CAMPAIGN FIRED 86/87, AND THAT RESULT IS KEPT
-----------------------------------------------------------------
Block F was first selected with `-k wholescan`. Under that rule F-M23 ("eager
atan_grad loses its factor of 1/2") ESCAPED, and the artifact of that run is
committed unedited as `audit_12_wholescan_mutations_run1_k_wholescan.json`.

The escape was the selector's, not the gate's, and that was established from
committed evidence rather than assumed. `audit_08_mutation_campaign.json` records
M23 as caught by exactly one test, `test_surrogate_matches_analytic` -- the gate's
PREMISE test, unparametrised and CPU-only, so its id contains no implementation
name and `-k wholescan` deselected it (63 selected, 68 deselected, all 63 passed).
And it is the only test that CAN catch M23: `snn.surrogate.atan_grad` is called
nowhere in `src/` outside its own module -- the eager scan's backward is autograd
through `atan_value`, the fused kernels use the `ATAN_CUDA_GRAD` text -- so no
fused-vs-eager comparison on either path can see it.

`-k wholescan` therefore asked a narrower question than the one block F states. It
dropped the per-timestep parametrisation AND every test that belongs to neither
path. `-k "not jiterator"` drops only the former, which is the stated question, and
is the rule now. No test NAME in the three gates contains "jiterator", so the
keyword matches parametrisation ids only. The whole campaign was re-run under the
corrected rule -- one run, one rule, not one mutation patched in -- and any escape
in THAT run is reported as it fires.

Machinery is `08`'s, by import: the process guard, the clean-tree assertion, the
`git checkout --` restore, the lock file, the exactly-once anchor check. The
warning in its docstring applies here unchanged -- NOTHING ELSE MAY IMPORT `snn`
WHILE THIS RUNS.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

# `08_...` is not an importable name, and it must not be copied: the point is that
# the restore/guard logic below is the code that already produced committed
# evidence, not a second implementation of it.
_spec = importlib.util.spec_from_file_location(
    "_mutation_campaign_08", Path(__file__).with_name("08_mutation_campaign.py"))
m08 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(m08)

WHOLESCAN = REPO / "src" / "snn" / "wholescan.py"
GATE_WS = "tests/test_wholescan_equivalence.py"

# `08` mutations that cannot reach the whole-scan path, with the reason and the
# W-mutation that stands in for each.
NOT_APPLICABLE: dict[str, str] = {
    "T19": "mutates FusedTwoCompartmentScan.backward's Python (the vs_prev shift); "
           "whole-scan counterpart W04",
    "T20": "mutates FusedTwoCompartmentScan.backward's return order; counterpart W09/W08",
    "D11": "mutates FusedTwoCompDetachScan.backward's Python (the vs_prev shift); "
           "whole-scan counterpart W04",
    "D12": "mutates FusedTwoCompDetachScan.backward's return order; counterpart W09/W08",
}

# Each entry: (id, description, find, replace[, extra pytest args]).
# `find` must appear EXACTLY ONCE in wholescan.py -- asserted, as in `08`.
WRAPPER_MUTATIONS: list[tuple] = [
    # ---- indexing: which element goes where --------------------------------
    ("W01", "consecutive timesteps are L apart instead of d (t*d -> t*L)",
     "base + (long long)t * d;", "base + (long long)t * L;"),
    ("W02", "the batch stride drops L (base = b*d + c)",
     "const long long base = b * (long long)L * d + c;",
     "const long long base = b * (long long)d + c;"),
    ("W03", "the slow half's offset into [B, 2d] is dropped (ks = kf)",
     "const long long ks = kf + d;", "const long long ks = kf;"),
    # ---- the vs_{t-1} shift and its t == 0 branch ---------------------------
    ("W04", "grad_beta_s uses vs_t instead of vs_{t-1} (k - d -> k)",
     "(t > 0) ? vs_seq[k - d] : v0[ks];", "(t > 0) ? vs_seq[k] : v0[ks];"),
    ("W05", "the t == 0 branch is off by one (t > 0 -> t > 1)",
     "(t > 0) ? vs_seq[k - d] : v0[ks];", "(t > 1) ? vs_seq[k - d] : v0[ks];"),
    ("W06", "the initial slow state is ignored at t == 0 (vs_{-1} = 0)",
     "(t > 0) ? vs_seq[k - d] : v0[ks];", "(t > 0) ? vs_seq[k - d] : 0.0f;"),
    ("W07", "vs_{-1} is read from the FAST half of v0",
     "(t > 0) ? vs_seq[k - d] : v0[ks];", "(t > 0) ? vs_seq[k - d] : v0[kf];"),
    # ---- the two per-element products --------------------------------------
    ("W08", "the two product outputs are swapped in the kernel",
     "        gw_elem[k] = e1 * vs_seq[k];\n        gbs_elem[k] = e2 * vs_prev;",
     "        gbs_elem[k] = e1 * vs_seq[k];\n        gw_elem[k] = e2 * vs_prev;"),
    ("W09", "the two per-channel gradients are returned swapped",
     "    return grad_cur, grad_v0, grad_w, grad_beta_s, None, None, None",
     "    return grad_cur, grad_v0, grad_beta_s, grad_w, None, None, None"),
    ("W10", "grad_w's product takes its operand from vs_{t-1} instead of vs_t",
     "gw_elem[k] = e1 * vs_seq[k];", "gw_elem[k] = e1 * vs_prev;"),
    ("W11", "grad_beta_s's product takes the wrong adjoint (e1 for e2)",
     "gbs_elem[k] = e2 * vs_prev;", "gbs_elem[k] = e1 * vs_prev;"),
    # ---- THE CONTRACTION BARRIER: the one-ulp class ------------------------
    ("W12", "drop noinline: every trampoline becomes inline", '_NOINLINE = "__attribute__((noinline))"',
     '_NOINLINE = "inline"'),
    ("W12b", "drop noinline -- scored against the NUMERIC comparison alone, with the "
             "source-text tests deselected, so the catch is a gradient and not a grep",
     '_NOINLINE = "__attribute__((noinline))"', '_NOINLINE = "inline"',
     ["-k", "test_every_output_and_gradient_is_bit_identical"]),
    ("W13", "the LIF time loop calls the functor directly, bypassing the trampoline",
     '                     "lif_step", barrier)', '                     "lif_backward<float>", barrier)',
     ["-k", "test_every_output_and_gradient_is_bit_identical"]),
    ("W14", "the twocomp time loop calls the functor directly, bypassing the trampoline",
     'functor, trampoline, step = twocomp_backward_source(), _TC_TRAMPOLINE, "tc_step"',
     'functor, trampoline, step = twocomp_backward_source(), _TC_TRAMPOLINE, "tc_backward<float>"',
     ["-k", "test_every_output_and_gradient_is_bit_identical"]),
    ("W27", "the NVRTC flags drift from jiterator's (contraction switched off)",
     '        "-DNDEBUG",\n    ]', '        "-DNDEBUG",\n        "--fmad=false",\n    ]',
     ["-k", "test_every_output_and_gradient_is_bit_identical"]),
    # ---- the save flag ------------------------------------------------------
    ("W15", "LIF: the grad path never writes v_pre (save flag ignored)",
     "        if (save) v_pre[k] = vp;", "        if (0) v_pre[k] = vp;"),
    ("W16", "twocomp: the grad path never writes vs_seq",
     "        if (save) { v_pre[k] = vp; vs_seq[k] = vsn; }",
     "        if (save) { v_pre[k] = vp; }"),
    ("W17", "LIF: the no-grad path writes v_pre anyway, through the dummy pointer",
     "        if (save) v_pre[k] = vp;", "        v_pre[k] = vp;"),
    ("W18", "the LIF Function asks for a no-save forward and then needs v_pre",
     "            cur.contiguous(), v0.contiguous(), beta, thr, reset, save=True)",
     "            cur.contiguous(), v0.contiguous(), beta, thr, reset, save=False)"),
    # ---- state in and out ---------------------------------------------------
    ("W22", "v_final is written with the compartments swapped",
     "    v_final[kf] = vf;\n    v_final[ks] = vs;", "    v_final[kf] = vs;\n    v_final[ks] = vf;"),
    ("W23", "the slow adjoint starts from the FAST half of the cotangent",
     "    float gvs = grad_v_final[ks];", "    float gvs = grad_v_final[kf];"),
    ("W24", "the LIF backward runs forward in time",
     "    float g = grad_v_final[i];\n    for (int t = L - 1; t >= 0; --t) {",
     "    float g = grad_v_final[i];\n    for (int t = 0; t < L; ++t) {"),
    # ---- the input contract and the run-time guards -------------------------
    ("W19", "the backward takes the cotangent's pointer without making it contiguous",
     "            grad_spikes.contiguous(), grad_v_final.contiguous(), v_pre,\n"
     "            ctx.beta, ctx.thr, ctx.alpha, ctx.reset)",
     "            grad_spikes, grad_v_final.contiguous(), v_pre,\n"
     "            ctx.beta, ctx.thr, ctx.alpha, ctx.reset)"),
    ("W20", "the raw launch stops refusing strided tensors",
     "        if not t.is_contiguous():\n            raise ValueError(",
     "        if False:\n            raise ValueError("),
    ("W21", "the dispatcher lets a strided current through to the whole-scan path",
     "    if not (cur.is_cuda and cur.is_contiguous() and _fits(cur)):",
     "    if not (cur.is_cuda and _fits(cur)):"),
    ("W25", "the run-time self-check always passes",
     "        return live and len(got) == len(want) and all(_bits_equal(a, b) for a, b in pairs)",
     "        return True"),
    ("W26", "the run-time comparison degrades from bit patterns to torch.equal",
     "    return a.shape == b.shape and bool(\n"
     "        (a.contiguous().view(torch.int32) == b.contiguous().view(torch.int32)).all()\n"
     "    )",
     "    return torch.equal(a, b)"),
    ("W28", "shapes that overflow the kernel's 32-bit arguments are no longer refused",
     "        and max(B, L, d) <= _INT_MAX\n", ""),
    ("W29", "SNN_SCAN_IMPL is ignored", "    return _check_impl(raw) if raw else SCAN_IMPLS[0]",
     "    return SCAN_IMPLS[0]"),
    ("W30", "the self-check draws its inputs from the GLOBAL cuda RNG",
     "        return (torch.randn(*shape, generator=gen) * scale).to(device)",
     "        return torch.randn(*shape, device=device) * scale"),
    ("W31", "the self-check is allowed to run inside a CUDA-graph capture",
     "        if torch.cuda.is_current_stream_capturing():\n            return None",
     "        if False:\n            return None"),
]


def run_gate(timeout_s: int, gate_args: list[str]) -> tuple[bool, str, str | None]:
    """`08.run_gate`, with room for a `-k`. Returns (passed, tail, first failing test)."""
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", *gate_args, "-x", "-q", "--no-header",
         "-p", "no:cacheprovider"],
        cwd=REPO, capture_output=True, text=True, timeout=timeout_s,
    )
    text = proc.stdout + proc.stderr
    tail = "\n".join(text.strip().splitlines()[-6:])
    failed = re.search(r"^(?:FAILED|ERROR) (\S+)", text, flags=re.M)
    return proc.returncode == 0, tail, failed.group(1) if failed else None


def _provenance() -> dict:
    prov = m08._tree_provenance()
    prov["guarded_sources"] = sorted({*prov["guarded_sources"], "src/snn/wholescan.py"})
    return prov


def build_plan(wanted: set[str]) -> list[dict]:
    plan = []
    for m in WRAPPER_MUTATIONS:
        mid, desc, find, repl = m[:4]
        plan.append({"id": mid, "block": "W", "description": desc, "path": WHOLESCAN,
                     "find": find, "replace": repl,
                     "gate_args": [GATE_WS, *(m[4] if len(m) > 4 else [])]})
    for m in m08.MUTATIONS:
        mid, desc, path, find, repl, gate = m08._normalise(m)
        if mid in NOT_APPLICABLE:
            continue
        plan.append({"id": f"F-{mid}", "block": "F", "description": desc, "path": path,
                     "find": find, "replace": repl,
                     "gate_args": [gate, "-k", "not jiterator"]})
    return [p for p in plan if not wanted or p["id"] in wanted]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="docs/perf/data/audit_12_wholescan_mutations.json")
    ap.add_argument("--only", default="", help="comma-separated ids, e.g. W04,F-M01")
    ap.add_argument("--timeout", type=int, default=900)
    ap.add_argument("--force-concurrent", action="store_true",
                    help="run even if other python processes are alive; they may "
                         "silently import a mutated kernel (see 08's docstring)")
    args = ap.parse_args(argv)

    others = m08.other_python_processes()
    if others and not args.force_concurrent:
        raise SystemExit(
            f"{len(others)} other python process(es) are running (PIDs "
            f"{', '.join(others[:8])}). This script writes wrong code into src/snn/; "
            f"anything importing `snn` meanwhile keeps it. Stop them, or pass "
            f"--force-concurrent if you are certain none of them touch `snn`.")
    if not m08.tree_is_clean():
        raise SystemExit("src/ has uncommitted changes; `git checkout --` would "
                         "discard them. Commit or stash first.")

    plan = build_plan({m.strip() for m in args.only.split(",") if m.strip()})
    for p in plan:
        n = p["path"].read_text(encoding="utf-8").count(p["find"])
        if n != 1:
            raise SystemExit(
                f"{p['id']}: anchor text appears {n} times in {p['path'].name}, "
                f"expected exactly 1. The source moved; fix the mutation first.")

    # Every distinct gate invocation must PASS on clean source, `-k` included: a
    # selection that matched nothing, or a gate already red, catches everything.
    baselines: dict[str, dict] = {}
    for gate_args in sorted({tuple(p["gate_args"]) for p in plan}):
        key = " ".join(gate_args)
        print(f"baseline: `pytest {key}` must PASS on unmutated source", flush=True)
        t0 = time.perf_counter()
        passed, tail, _ = run_gate(args.timeout, list(gate_args))
        dt = time.perf_counter() - t0
        n_passed = re.search(r"(\d+) passed", tail)
        print(f"  {'PASS' if passed else 'FAIL'} in {dt:.0f}s  ({tail.splitlines()[-1]})")
        if not passed or not n_passed or int(n_passed.group(1)) == 0:
            raise SystemExit(f"`pytest {key}` is not a passing, non-empty gate:\n{tail}")
        baselines[key] = {"pass": True, "n_passed": int(n_passed.group(1)),
                          "seconds": round(dt, 1)}

    results: dict = {
        "audit": "12_wholescan_mutations",
        "protocol": "outside Phase 1-5; see docs/perf/WHOLESCAN.md",
        "verified_tree": _provenance(),
        "gates": baselines,
        "not_applicable_from_08": NOT_APPLICABLE,
        "n_mutations": len(plan),
        "mutations": {},
    }
    escaped: list[str] = []
    for i, p in enumerate(plan, 1):
        path: Path = p["path"]
        original = path.read_text(encoding="utf-8")
        print(f"[{i}/{len(plan)}] {p['id']}  {p['description']}", flush=True)
        try:
            m08.LOCKFILE.write_text(
                f"{p['id']}: {p['description']}\nsrc/snn is MUTATED right now. "
                "Do not import snn.\n", encoding="utf-8")
            path.write_text(original.replace(p["find"], p["replace"]), encoding="utf-8",
                            newline="\n")
            t0 = time.perf_counter()
            passed, tail, first = run_gate(args.timeout, p["gate_args"])
            dt = time.perf_counter() - t0
        except subprocess.TimeoutExpired:
            passed, tail, first, dt = True, "TIMEOUT", None, float(args.timeout)
        finally:
            m08.git("checkout", "--", str(path.relative_to(REPO)))
            m08.LOCKFILE.unlink(missing_ok=True)
        caught = not passed
        if not caught:
            escaped.append(p["id"])
        results["mutations"][p["id"]] = {
            "block": p["block"], "description": p["description"], "file": path.name,
            "gate": " ".join(p["gate_args"]), "caught": caught, "caught_by": first,
            "gate_seconds": round(dt, 1), "gate_output_tail": tail,
        }
        print(f"      {'CAUGHT by ' + str(first) if caught else '*** ESCAPED ***'}  ({dt:.0f}s)")

    if not m08.tree_is_clean():
        raise SystemExit("src/ is dirty after the campaign; restore failed")

    by_block = {b: [r for r in results["mutations"].values() if r["block"] == b]
                for b in ("W", "F")}
    results["n_caught"] = sum(r["caught"] for r in results["mutations"].values())
    results["n_caught_by_block"] = {
        b: f"{sum(r['caught'] for r in rs)}/{len(rs)}" for b, rs in by_block.items()}
    results["escaped"] = escaped
    results["all_caught"] = not escaped
    results["source_restored_clean"] = True

    out = REPO / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print("\n" + "=" * 74)
    print(f"WHOLE-SCAN MUTATION CAMPAIGN: {results['n_caught']} / {len(plan)} caught  "
          f"{results['n_caught_by_block']}")
    if escaped:
        print(f"  ESCAPED (holes in the gate): {escaped}")
    else:
        print("  no mutation survived its gate")
    print("  source tree restored clean: yes")
    print("=" * 74)
    print(f"WROTE {args.out}")
    return 1 if escaped else 0


if __name__ == "__main__":
    raise SystemExit(main())
