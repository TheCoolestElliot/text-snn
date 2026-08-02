"""EXP_002 -- what the 4.6-admitted neurons cost to run, measured not argued.

Pre-registered in experiments/logs/EXP_002_candidate_neuron_cost.md. The five
predictions Q1-Q5 and the decision rule were fixed before this file existed; two
of them (Q3, Q5) are stated in the direction that hurts the candidates.

What this prices, and what it does not
--------------------------------------
Each candidate here is a *forward recurrence only*. None has a verified backward,
none has been near the R10 gate, and nothing in this file is evidence that any of
these neurons improves bits-per-character. Writing the mutation-tested gradient
gate for whichever candidate wins is itself part of that candidate's cost, and
the report says so.

Why the arity matters
---------------------
jiterator caps at 8 inputs and 8 outputs (01_reconnaissance.md 3.11 C1). Every
admitted extension needs a second state variable and most need a per-channel
parameter tensor, so the arity budget is the first thing that can strike a
candidate off. It is counted explicitly per row rather than inferred.

Why one kernel per timestep is the whole question
-------------------------------------------------
02_baseline_report.md 6.2: the per-kernel cost on the fused scan is 32.8 us, and
6.1: fusion and CUDA-graph capture are the same lever, so a candidate that breaks
the one-kernel floor cannot expect graphing to hide it. If a two-state neuron
still issues one kernel per timestep, the whole admitted family is systems-free
and Phase 3's ranking is decided by expected bpc alone.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO / "src") not in sys.path:
    sys.path.insert(0, str(_REPO / "src"))

from snn.kernels import jiterator_available  # noqa: E402
from snn.metrics import count_cuda_kernels  # noqa: E402
from snn.surrogate import ATAN_CUDA_GRAD  # noqa: E402

# The frozen Phase-2 baseline shape (spec 02a 13), so the numbers are comparable
# with docs/reports/data/phase2_kernel_bench.json row for row (failure mode G2).
B, L, D = 128, 256, 512

# G1: N0 must reproduce the committed baseline count.
BASELINE_KERNELS_PER_TIMESTEP = 1.016


# ---------------------------------------------------------------------------
# Candidate forward kernels
# ---------------------------------------------------------------------------
# `_VPRE`-style explicitly-rounded arithmetic is deliberately NOT used here: these
# are cost prototypes, not numerically-contracted kernels, and adding the
# intrinsics would price a correctness property that the R10 gate has not yet been
# written for. The kernel *shape* -- how many loads, stores and flops per element
# -- is what is being measured, and that is unaffected.

N0_LIF = """
template <typename T>
void n0_lif(T v_prev, T cur, T beta, T thr,
            T& v_next, T& spike, T& v_pre) {
    const T v = v_prev * beta + cur;
    const T s = (v >= thr) ? T(1) : T(0);
    v_pre  = v;
    spike  = s;
    v_next = v * (T(1) - s);
}
"""

# N1: learned per-channel decay. `beta_c` arrives as a [1, d] TENSOR and must
# broadcast against [B, d] inside the kernel -- prediction Q4.
N1_LEARNED_DECAY = """
template <typename T>
void n1_decay(T v_prev, T cur, T beta_c, T thr,
              T& v_next, T& spike, T& v_pre) {
    const T v = v_prev * beta_c + cur;
    const T s = (v >= thr) ? T(1) : T(0);
    v_pre  = v;
    spike  = s;
    v_next = v * (T(1) - s);
}
"""

# N2: two-timescale membrane. A fast and a slow compartment integrate the same
# current at different rates; the neuron fires on a learned per-channel mix. Both
# compartments reset. Still O(1) params per neuron, so 4.6 admits it.
#
# ARGUMENT ORDER IS LOad-BEARING: jiterator binds every TENSOR argument first, in
# order, then the scalars as keywords. An earlier draft of this kernel declared
# `(vf_prev, vs_prev, cur, beta_f, beta_s, w_c, thr)` -- scalars interleaved
# before the per-channel tensor `w_c` -- which compiles and runs happily while
# silently binding `w_c` into the `beta_f` slot. It was caught by the
# eager-reference check below (v_pre off by 0.99, spikes not bit-identical), not
# by anything about the kernel failing, which is the entire reason that check
# exists and is committed rather than run once by hand.
N2_TWO_TIMESCALE = """
template <typename T>
void n2_two_ts(T vf_prev, T vs_prev, T cur, T w_c, T beta_f, T beta_s, T thr,
               T& vf_next, T& vs_next, T& spike, T& v_pre) {
    const T vf = vf_prev * beta_f + cur;
    const T vs = vs_prev * beta_s + cur;
    const T v  = w_c * vf + (T(1) - w_c) * vs;
    const T s  = (v >= thr) ? T(1) : T(0);
    v_pre    = v;
    spike    = s;
    vf_next  = vf * (T(1) - s);
    vs_next  = vs * (T(1) - s);
}
"""

# N3: adaptive threshold (SE-adLIF shaped, 4.3 item 4). A second state variable
# `u` integrates the neuron's own spikes and raises its threshold. The kernel
# emits `x = v_pre - thr_t` rather than the two separately: the backward needs
# only the difference, and one fewer output is one fewer store.
N3_ADAPTIVE_THRESHOLD = """
template <typename T>
void n3_adthr(T v_prev, T u_prev, T cur, T a_c, T beta, T rho, T thr0,
              T& v_next, T& u_next, T& spike, T& x_out) {
    const T v   = v_prev * beta + cur;
    const T thr = thr0 + a_c * u_prev;
    const T x   = v - thr;
    const T s   = (x >= T(0)) ? T(1) : T(0);
    x_out   = x;
    spike   = s;
    u_next  = rho * u_prev + s;
    v_next  = v * (T(1) - s);
}
"""

# N4: rotational (complex) membrane. The state is a 2-vector rotated by a learned
# per-channel angle and scaled by a learned decay; current is injected into the
# real part and the neuron fires on it. The decay is folded into the cos/sin
# tensors by the caller, so this costs two per-channel tensors, not three.
N4_ROTATIONAL = """
template <typename T>
void n4_rot(T re_prev, T im_prev, T cur, T lcos_c, T lsin_c, T thr,
            T& re_next, T& im_next, T& spike, T& v_pre) {
    const T re = lcos_c * re_prev - lsin_c * im_prev + cur;
    const T im = lsin_c * re_prev + lcos_c * im_prev;
    const T s  = (re >= thr) ? T(1) : T(0);
    v_pre    = re;
    spike    = s;
    re_next  = re * (T(1) - s);
    im_next  = im;
}
"""

# N1's backward, the Q5 case. A per-channel decay's gradient is a SUM over the
# batch and time axes, and jiterator cannot reduce (2.3). So the kernel emits the
# per-element contribution as a third output and the reduction happens in torch.
N1_BACKWARD = """
template <typename T>
void n1_bwd(T grad_spike, T grad_v_next, T v_pre, T v_prev,
            T beta_c, T thr, T alpha,
            T& grad_cur, T& grad_v_prev, T& grad_beta_elem) {{
    const T x  = v_pre - thr;
    const T s  = (x >= T(0)) ? T(1) : T(0);
    const T sg = {atan_grad};
    const T dv = (T(1) - s) - v_pre * sg;
    const T g  = grad_v_next * dv + grad_spike * sg;
    grad_cur       = g;
    grad_v_prev    = beta_c * g;
    grad_beta_elem = v_prev * g;
}}
"""


def _jit(src: str, num_outputs: int, **scalars):
    from torch.cuda.jiterator import _create_multi_output_jit_fn
    return _create_multi_output_jit_fn(src, num_outputs=num_outputs, **scalars)


# ---------------------------------------------------------------------------
# Candidate definitions: each builds its scan closure and reports its arity
# ---------------------------------------------------------------------------

def make_candidates(device: torch.device) -> list[dict]:
    def z(*shape):
        return torch.zeros(*shape, device=device, dtype=torch.float32)

    cur = torch.randn(B, L, D, device=device, dtype=torch.float32) * 0.5
    # Per-channel parameters are [1, d]: this is the broadcast Q4 is about.
    beta_c = torch.full((1, D), 0.5, device=device, dtype=torch.float32)
    w_c = torch.full((1, D), 0.5, device=device, dtype=torch.float32)
    a_c = torch.full((1, D), 0.2, device=device, dtype=torch.float32)
    lcos = torch.full((1, D), 0.45, device=device, dtype=torch.float32)
    lsin = torch.full((1, D), 0.15, device=device, dtype=torch.float32)

    cands: list[dict] = []

    def n0():
        fn = _jit(N0_LIF, 3, beta=0.5, thr=1.0)
        def scan():
            v = z(B, D)
            out = []
            for t in range(L):
                v, s, _vp = fn(v, cur[:, t], beta=0.5, thr=1.0)
                out.append(s)
            return torch.stack(out, dim=1)
        return fn, scan
    cands.append({"id": "N0", "name": "LIF (Phase-2 baseline)", "build": n0,
                  "tensor_inputs": 2, "outputs": 3, "state_vars": 1,
                  "per_channel_params": 0})

    def n1():
        fn = _jit(N1_LEARNED_DECAY, 3, thr=1.0)
        def scan():
            v = z(B, D)
            out = []
            for t in range(L):
                v, s, _vp = fn(v, cur[:, t], beta_c, thr=1.0)
                out.append(s)
            return torch.stack(out, dim=1)
        return fn, scan
    cands.append({"id": "N1", "name": "learned per-channel decay", "build": n1,
                  "tensor_inputs": 3, "outputs": 3, "state_vars": 1,
                  "per_channel_params": 1})

    def n2():
        fn = _jit(N2_TWO_TIMESCALE, 4, beta_f=0.3, beta_s=0.95, thr=1.0)
        def scan():
            vf, vs = z(B, D), z(B, D)
            out = []
            for t in range(L):
                vf, vs, s, _vp = fn(vf, vs, cur[:, t], w_c,
                                    beta_f=0.3, beta_s=0.95, thr=1.0)
                out.append(s)
            return torch.stack(out, dim=1)
        return fn, scan
    # tensor args first (vf, vs, cur, w_c), scalars as keywords -- see the note
    # on N2_TWO_TIMESCALE for what happens when that order is broken.
    cands.append({"id": "N2", "name": "two-timescale membrane", "build": n2,
                  "tensor_inputs": 4, "outputs": 4, "state_vars": 2,
                  "per_channel_params": 1})

    def n3():
        fn = _jit(N3_ADAPTIVE_THRESHOLD, 4, beta=0.5, rho=0.95, thr0=1.0)
        def scan():
            v, u = z(B, D), z(B, D)
            out = []
            for t in range(L):
                v, u, s, _x = fn(v, u, cur[:, t], a_c,
                                 beta=0.5, rho=0.95, thr0=1.0)
                out.append(s)
            return torch.stack(out, dim=1)
        return fn, scan
    cands.append({"id": "N3", "name": "adaptive threshold (adLIF)", "build": n3,
                  "tensor_inputs": 4, "outputs": 4, "state_vars": 2,
                  "per_channel_params": 1})

    def n4():
        fn = _jit(N4_ROTATIONAL, 4, thr=1.0)
        def scan():
            re, im = z(B, D), z(B, D)
            out = []
            for t in range(L):
                re, im, s, _vp = fn(re, im, cur[:, t], lcos, lsin, thr=1.0)
                out.append(s)
            return torch.stack(out, dim=1)
        return fn, scan
    cands.append({"id": "N4", "name": "rotational (complex) membrane", "build": n4,
                  "tensor_inputs": 5, "outputs": 4, "state_vars": 2,
                  "per_channel_params": 2})

    return cands


def verify_semantics(device: torch.device) -> dict:
    """Each candidate kernel against a plain-torch reference of the same equations.

    Not a substitute for the R10 gate -- these are forward recurrences only and
    none of them has a verified backward. It is a guard against the specific
    failure this experiment is otherwise wide open to: a kernel that compiles,
    runs, and produces a perfectly plausible *cost* while computing the wrong
    function. A wrong kernel prices roughly the same as a right one, so timing
    alone cannot detect it, and an unpriced candidate would then enter the ROI
    matrix on a number that means nothing.

    It has already earned its place once: it caught N2 binding its per-channel
    mixing tensor into a scalar slot (see the note above that kernel).
    """
    torch.manual_seed(0)
    n, d = 3, 8
    cur = torch.randn(n, d, device=device)
    out: dict = {}

    fn = _jit(N0_LIF, 3, beta=0.5, thr=1.0)
    v = torch.randn(n, d, device=device)
    vn, s, vp = fn(v, cur, beta=0.5, thr=1.0)
    r_vp = v * 0.5 + cur
    r_s = (r_vp >= 1.0).float()
    out["N0"] = {"v_pre_max_err": float((vp - r_vp).abs().max()),
                 "spike_bit_identical": bool(torch.equal(s, r_s)),
                 "v_next_max_err": float((vn - r_vp * (1 - r_s)).abs().max())}

    fn = _jit(N1_LEARNED_DECAY, 3, thr=1.0)
    beta_c = torch.rand(1, d, device=device)
    vn, s, vp = fn(v, cur, beta_c, thr=1.0)
    r_vp = v * beta_c + cur
    r_s = (r_vp >= 1.0).float()
    # Q4 is a FUNCTIONAL claim, not just a timing one: the [1, d] tensor has to
    # actually vary the decay per channel, and each channel has to get *its own*
    # entry rather than, say, entry 0 broadcast everywhere. Recovering beta_c
    # from the output tests exactly that.
    #
    # Note the bar. `torch.equal(vp, v*beta_c + cur)` is the wrong test and
    # reports False here: NVRTC contracts the kernel's multiply-add into an FMA
    # that rounds once where torch's separate mul and add round twice, so the two
    # differ by an ulp. snn/kernels.py suppresses that contraction deliberately
    # because the R10 gate demands bit-identical spikes; these are cost
    # prototypes with no such gate, so 1e-6 is the honest tolerance and exactness
    # would be measuring the wrong property.
    recovered = (vp - cur) / v
    out["N1"] = {"v_pre_max_err": float((vp - r_vp).abs().max()),
                 "spike_bit_identical": bool(torch.equal(s, r_s)),
                 "recovered_beta_c_max_err": float(
                     (recovered - beta_c.expand_as(recovered)).abs().max()),
                 "per_channel_values_distinct": int(
                     len(set(recovered[0].tolist()))) == d}

    fn = _jit(N2_TWO_TIMESCALE, 4, beta_f=0.3, beta_s=0.95, thr=1.0)
    vf, vs = torch.randn(n, d, device=device), torch.randn(n, d, device=device)
    w_c = torch.rand(1, d, device=device)
    o_vf, o_vs, s, vp = fn(vf, vs, cur, w_c, beta_f=0.3, beta_s=0.95, thr=1.0)
    r_vf, r_vs = vf * 0.3 + cur, vs * 0.95 + cur
    r_vp = w_c * r_vf + (1 - w_c) * r_vs
    r_s = (r_vp >= 1.0).float()
    out["N2"] = {"v_pre_max_err": float((vp - r_vp).abs().max()),
                 "spike_bit_identical": bool(torch.equal(s, r_s)),
                 "vf_max_err": float((o_vf - r_vf * (1 - r_s)).abs().max()),
                 "vs_max_err": float((o_vs - r_vs * (1 - r_s)).abs().max())}

    fn = _jit(N3_ADAPTIVE_THRESHOLD, 4, beta=0.5, rho=0.95, thr0=1.0)
    u, a_c = torch.rand(n, d, device=device), torch.rand(1, d, device=device)
    o_v, o_u, s, x = fn(v, u, cur, a_c, beta=0.5, rho=0.95, thr0=1.0)
    r_v = v * 0.5 + cur
    r_x = r_v - (1.0 + a_c * u)
    r_s = (r_x >= 0).float()
    out["N3"] = {"x_max_err": float((x - r_x).abs().max()),
                 "spike_bit_identical": bool(torch.equal(s, r_s)),
                 "u_max_err": float((o_u - (0.95 * u + r_s)).abs().max()),
                 "v_next_max_err": float((o_v - r_v * (1 - r_s)).abs().max())}

    fn = _jit(N4_ROTATIONAL, 4, thr=1.0)
    re, im = torch.randn(n, d, device=device), torch.randn(n, d, device=device)
    lc, ls = torch.rand(1, d, device=device), torch.rand(1, d, device=device)
    o_re, o_im, s, vp = fn(re, im, cur, lc, ls, thr=1.0)
    r_re, r_im = lc * re - ls * im + cur, ls * re + lc * im
    r_s = (r_re >= 1.0).float()
    out["N4"] = {"re_max_err": float((vp - r_re).abs().max()),
                 "im_max_err": float((o_im - r_im).abs().max()),
                 "spike_bit_identical": bool(torch.equal(s, r_s)),
                 "re_next_max_err": float((o_re - r_re * (1 - r_s)).abs().max())}

    for cid, r in out.items():
        errs = [v for k, v in r.items() if k.endswith("_err")]
        flags = [v for k, v in r.items()
                 if isinstance(v, bool) and not k.endswith("_err")]
        r["pass"] = all(e < 1e-5 for e in errs) and all(flags)
    return out


def time_ms(scan, iters: int = 5) -> float:
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        scan()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1000.0 / iters


def measure_n1_backward(device: torch.device) -> dict:
    """Q5: the per-channel decay gradient needs a reduction jiterator cannot do.

    Two ways to pay for it, both measured rather than reasoned about:
      (a) accumulate the per-element contribution inside the loop  -> +1 kernel/step
      (b) stack the L contributions and reduce once at the end     -> ~0 extra/step,
          at the cost of holding another [B, L, d] tensor.
    """
    src = N1_BACKWARD.format(atan_grad=ATAN_CUDA_GRAD)
    fn = _jit(src, 3, thr=1.0, alpha=2.0)
    beta_c = torch.full((1, D), 0.5, device=device, dtype=torch.float32)
    g_s = torch.randn(B, L, D, device=device, dtype=torch.float32)
    v_pre = torch.randn(B, L, D, device=device, dtype=torch.float32)
    v_prev = torch.randn(B, L, D, device=device, dtype=torch.float32)

    def bwd_inloop():
        gv = torch.zeros(B, D, device=device, dtype=torch.float32)
        gb = torch.zeros(B, D, device=device, dtype=torch.float32)
        gcs = [None] * L
        for t in range(L - 1, -1, -1):
            gcs[t], gv, gbe = fn(g_s[:, t], gv, v_pre[:, t], v_prev[:, t],
                                 beta_c, thr=1.0, alpha=2.0)
            gb = gb + gbe                      # the extra kernel per timestep
        return torch.stack(gcs, dim=1), gb.sum(dim=0)

    def bwd_deferred():
        gv = torch.zeros(B, D, device=device, dtype=torch.float32)
        gcs = [None] * L
        gbs = [None] * L
        for t in range(L - 1, -1, -1):
            gcs[t], gv, gbs[t] = fn(g_s[:, t], gv, v_pre[:, t], v_prev[:, t],
                                    beta_c, thr=1.0, alpha=2.0)
        return torch.stack(gcs, dim=1), torch.stack(gbs, dim=1).sum(dim=(0, 1))

    out = {}
    for label, f in (("accumulate_in_loop", bwd_inloop),
                     ("stack_and_reduce_once", bwd_deferred)):
        torch.cuda.reset_peak_memory_stats()
        counts = count_cuda_kernels(f)
        out[label] = {
            "kernels": counts["total"],
            "kernels_per_timestep": round(counts["total"] / L, 4),
            "memcpy": counts["memcpy"],
            "memset": counts["memset"],
            "ms": round(time_ms(f), 3),
            "peak_gib": round(torch.cuda.max_memory_allocated() / 2**30, 4),
            "top_kernels": sorted(counts["by_short_name"].items(),
                                  key=lambda kv: -kv[1])[:5],
        }
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out",
                    default="docs/reports/data/exp_002_candidate_neuron_cost.json")
    args = ap.parse_args(argv)

    if not torch.cuda.is_available() or not jiterator_available():
        raise SystemExit("EXP_002 entry condition failed: jiterator unavailable")

    device = torch.device("cuda")
    results: dict = {
        "experiment": "EXP_002_candidate_neuron_cost",
        "config": {"batch_size": B, "seq_len": L, "d_model": D},
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(0),
        "jiterator_limits": {"max_inputs": 8, "max_outputs": 8},
        "baseline_reference_kernels_per_timestep": BASELINE_KERNELS_PER_TIMESTEP,
        "candidates": {},
    }

    print("  semantics check (each kernel vs a plain-torch reference)")
    sem = verify_semantics(device)
    results["semantics"] = sem
    for cid, r in sem.items():
        print(f"      {cid}  {'PASS' if r['pass'] else 'FAIL'}  {r}")
    if not all(r["pass"] for r in sem.values()):
        raise SystemExit("a candidate kernel does not compute its own equations; "
                         "refusing to price it")

    baseline_ms = None
    for cand in make_candidates(device):
        row = {k: v for k, v in cand.items() if k != "build"}
        row["arity_total_inputs"] = cand["tensor_inputs"]
        row["within_8x8_limit"] = (cand["tensor_inputs"] <= 8
                                   and cand["outputs"] <= 8)
        print(f"  {cand['id']}  {cand['name']}")
        try:
            t0 = time.perf_counter()
            _fn, scan = cand["build"]()
            scan()                                  # forces the NVRTC compile
            torch.cuda.synchronize()
            row["compile_and_first_run_s"] = round(time.perf_counter() - t0, 3)
            row["q1_compiles"] = True
        except Exception as exc:                    # struck, and recorded as struck
            row["q1_compiles"] = False
            row["error"] = f"{type(exc).__name__}: {exc}"
            print(f"      STRUCK: {row['error']}")
            results["candidates"][cand["id"]] = row
            continue

        torch.cuda.reset_peak_memory_stats()
        counts = count_cuda_kernels(scan)
        row["kernels"] = counts["total"]
        row["kernels_per_timestep"] = round(counts["total"] / L, 4)
        row["memcpy"] = counts["memcpy"]
        row["memset"] = counts["memset"]
        # Three independent timing passes. Wall-clock on this box moves ~30% with
        # clock and power state (snn/neuron.py docstring), so a single number
        # cannot decide a 1.25x threshold; the median is reported and the spread
        # is committed so a borderline verdict can be seen to be borderline.
        samples = sorted(time_ms(scan) for _ in range(3))
        row["ms_samples"] = [round(s, 3) for s in samples]
        row["ms"] = round(samples[1], 3)
        row["ms_spread_rel"] = round((samples[2] - samples[0]) / samples[1], 4)
        row["peak_gib"] = round(torch.cuda.max_memory_allocated() / 2**30, 4)
        row["top_kernels"] = sorted(counts["by_short_name"].items(),
                                    key=lambda kv: -kv[1])[:5]
        if cand["id"] == "N0":
            baseline_ms = row["ms"]
            row["g1_pass"] = abs(row["kernels_per_timestep"]
                                 - BASELINE_KERNELS_PER_TIMESTEP) < 0.02
        row["ms_vs_baseline"] = (round(row["ms"] / baseline_ms, 3)
                                 if baseline_ms else None)
        row["q2_one_kernel_per_timestep"] = row["kernels_per_timestep"] < 1.05
        row["q3_under_25pct_slower"] = (row["ms_vs_baseline"] < 1.25
                                        if row["ms_vs_baseline"] else None)
        print(f"      kernels/timestep {row['kernels_per_timestep']:.4f}   "
              f"{row['ms']:.2f} ms   x{row['ms_vs_baseline']}   "
              f"peak {row['peak_gib']:.3f} GiB   "
              f"Q2 {'OK' if row['q2_one_kernel_per_timestep'] else 'FAIL'}  "
              f"Q3 {'OK' if row['q3_under_25pct_slower'] else 'FAIL'}")
        results["candidates"][cand["id"]] = row

    print("\n  N1 backward (Q5: the reduction jiterator cannot do)")
    bwd = measure_n1_backward(device)
    results["n1_backward"] = bwd
    for label, r in bwd.items():
        print(f"      {label:24s} {r['kernels_per_timestep']:.4f} kernels/step  "
              f"{r['ms']:.2f} ms  peak {r['peak_gib']:.3f} GiB")

    out = _REPO / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print("\n" + "=" * 78)
    print("CANDIDATE NEURON COST (B=128, L=256, d=512)")
    print("=" * 78)
    print(f"{'id':4s} {'neuron':32s} {'in/out':>7s} {'k/step':>8s} {'ms':>8s} "
          f"{'vs N0':>7s}")
    for cid, r in results["candidates"].items():
        if not r.get("q1_compiles"):
            print(f"{cid:4s} {r['name']:32s} {'STRUCK':>7s}")
            continue
        print(f"{cid:4s} {r['name']:32s} "
              f"{r['tensor_inputs']}/{r['outputs']:<5d} "
              f"{r['kernels_per_timestep']:>8.4f} {r['ms']:>8.2f} "
              f"{r['ms_vs_baseline']:>7.2f}")
    print(f"\nWROTE {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
