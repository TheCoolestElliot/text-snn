"""Phase-1 reconnaissance: independent verification of three literature claims.

The literature review asserted three things that contradict or materially change
the audit's own conclusions. Each is re-tested here from scratch rather than
accepted on report:

  C1  torch.cuda.jiterator compiles custom *elementwise* CUDA at runtime via the
      NVRTC bundled inside the torch wheel -- no nvcc, no MSVC, no Triton.
      If true, the audit's "fusion is unavailable" conclusion is WRONG and a
      fused LIF kernel is possible after all.

  C2  bf16 is specifically BAD for thresholded spiking because its representable
      spacing at v=1.0 is ~8x coarser than fp16's, so membrane values quantise
      onto a grid straddling the firing threshold. This would invert the audit's
      "prefer bf16" recommendation.

  C3  A reset-free leaky recurrence v_t = decay*v_{t-1} + i_t is exactly a
      triangular-Toeplitz matmul, giving a large speedup over the sequential
      loop -- and decay^L underflows to zero in fp32 at long L, silently
      truncating context.
"""

import json
import sys
import time

import torch

RESULTS = {}


def timeit(fn, iters=50, warmup=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters


# --------------------------------------------------------------------------
# C1: jiterator
# --------------------------------------------------------------------------

LIF_CUDA = """
template <typename T>
void fused_lif(T v_prev, T i_t, T decay, T thr, T& v_out, T& s_out) {
    T v = v_prev * decay + i_t;
    T s = (v >= thr) ? T(1) : T(0);
    v_out = v - s * thr;
    s_out = s;
}
"""


def verify_jiterator():
    out = {}
    # Is NVRTC actually shipped inside the torch wheel?
    import glob
    import os

    libdir = os.path.join(os.path.dirname(torch.__file__), "lib")
    out["bundled_nvrtc"] = [
        os.path.basename(p) for p in glob.glob(os.path.join(libdir, "*nvrtc*"))
    ]

    try:
        from torch.cuda.jiterator import _create_multi_output_jit_fn
    except Exception as exc:
        out["import"] = f"FAILED: {type(exc).__name__}: {exc}"
        return out
    out["import"] = "OK"

    try:
        t0 = time.perf_counter()
        lif = _create_multi_output_jit_fn(LIF_CUDA, num_outputs=2, decay=0.9, thr=1.0)
        B, H = 64, 512
        v = torch.randn(B, H, device="cuda")
        i = torch.randn(B, H, device="cuda")
        v_out, s_out = lif(v, i, decay=0.9, thr=1.0)
        torch.cuda.synchronize()
        out["compile_and_run_s"] = round(time.perf_counter() - t0, 3)
        out["status"] = "OK"
    except Exception as exc:
        out["status"] = f"FAILED: {type(exc).__name__}: {str(exc)[:500]}"
        return out

    # Correctness against an eager fp32 reference
    def eager_lif(v_prev, i_t, decay=0.9, thr=1.0):
        vv = v_prev * decay + i_t
        ss = (vv >= thr).to(vv.dtype)
        return vv - ss * thr, ss

    ref_v, ref_s = eager_lif(v, i)
    out["max_err_v"] = float((v_out - ref_v).abs().max())
    out["max_err_s"] = float((s_out - ref_s).abs().max())
    out["requires_grad"] = bool(v_out.requires_grad)

    # Speed: fused single kernel vs the eager multi-kernel chain
    fused_t = timeit(lambda: lif(v, i, decay=0.9, thr=1.0), iters=200, warmup=50)
    eager_t = timeit(lambda: eager_lif(v, i), iters=200, warmup=50)
    out["fused_us"] = round(fused_t * 1e6, 2)
    out["eager_us"] = round(eager_t * 1e6, 2)
    out["speedup"] = round(eager_t / fused_t, 2)

    # dtype coverage
    dtypes = {}
    for name, dt in [("fp32", torch.float32), ("bf16", torch.bfloat16),
                     ("fp16", torch.float16)]:
        try:
            a = torch.randn(B, H, device="cuda", dtype=dt)
            b = torch.randn(B, H, device="cuda", dtype=dt)
            r = lif(a, b, decay=0.9, thr=1.0)
            torch.cuda.synchronize()
            dtypes[name] = f"OK {r[0].dtype}"
        except Exception as exc:
            dtypes[name] = f"FAILED: {type(exc).__name__}"
    out["dtypes"] = dtypes

    # Does a jiterator kernel survive CUDA-graph capture?
    try:
        sv = torch.randn(B, H, device="cuda")
        si = torch.randn(B, H, device="cuda")
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                lif(sv, si, decay=0.9, thr=1.0)
        torch.cuda.current_stream().wait_stream(s)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            for _ in range(8):
                lif(sv, si, decay=0.9, thr=1.0)
        dt = timeit(g.replay, iters=100, warmup=20)
        out["graph_capture"] = {
            "status": "OK",
            "per_step_us": round(dt * 1e6 / 8, 2),
        }
    except Exception as exc:
        out["graph_capture"] = {"status": f"FAILED: {type(exc).__name__}: {str(exc)[:250]}"}

    # A full sequential unroll: the number that actually matters
    L = 128
    cur = torch.randn(L, B, H, device="cuda")

    def seq_fused():
        vv = torch.zeros(B, H, device="cuda")
        for t in range(L):
            vv, _ = lif(vv, cur[t], decay=0.9, thr=1.0)
        return vv

    def seq_eager():
        vv = torch.zeros(B, H, device="cuda")
        for t in range(L):
            vv, _ = eager_lif(vv, cur[t])
        return vv

    with torch.no_grad():
        out["unroll_L128_fused_ms"] = round(timeit(seq_fused, iters=10, warmup=3) * 1e3, 2)
        out["unroll_L128_eager_ms"] = round(timeit(seq_eager, iters=10, warmup=3) * 1e3, 2)
    out["unroll_speedup"] = round(
        out["unroll_L128_eager_ms"] / out["unroll_L128_fused_ms"], 2
    )
    return out


# --------------------------------------------------------------------------
# C2: precision at the firing threshold
# --------------------------------------------------------------------------

def verify_precision():
    out = {}
    for name, dt in [("bf16", torch.bfloat16), ("fp16", torch.float16),
                     ("fp32", torch.float32)]:
        one = torch.tensor(1.0, dtype=dt, device="cuda")
        nxt = torch.nextafter(one, torch.tensor(2.0, dtype=dt, device="cuda"))
        spacing = float(nxt.float() - one.float())
        out[name] = {
            "spacing_at_1.0": spacing,
            "pct_of_threshold": round(spacing * 100, 4),
        }
    out["fp16_finer_than_bf16_x"] = round(
        out["bf16"]["spacing_at_1.0"] / out["fp16"]["spacing_at_1.0"], 2
    )

    # Practical consequence: how many spikes flip vs an fp32 reference?
    torch.manual_seed(0)
    B, H, L = 64, 512, 64
    cur32 = torch.randn(L, B, H, device="cuda") * 0.3

    def run(dt):
        v = torch.zeros(B, H, device="cuda", dtype=dt)
        cur = cur32.to(dt)
        total = 0
        for t in range(L):
            v = v * torch.tensor(0.9, dtype=dt, device="cuda") + cur[t]
            s = (v >= torch.tensor(1.0, dtype=dt, device="cuda")).to(dt)
            v = v - s
            total += float(s.float().sum())
        return total

    with torch.no_grad():
        ref = run(torch.float32)
        got = {n: run(dt) for n, dt in
               [("bf16", torch.bfloat16), ("fp16", torch.float16)]}
    out["spike_count_fp32"] = ref
    out["spike_count_drift_pct"] = {
        n: round((v - ref) / ref * 100, 4) for n, v in got.items()
    }
    return out


# --------------------------------------------------------------------------
# C3: exact triangular-Toeplitz scan for the reset-free recurrence
# --------------------------------------------------------------------------

def verify_toeplitz():
    out = {}
    torch.manual_seed(0)
    for L, B, H in [(128, 16, 512), (256, 16, 512), (1024, 4, 256)]:
        cur = torch.randn(L, B, H, device="cuda")
        decay = 0.9

        def sequential():
            v = torch.zeros(B, H, device="cuda")
            acc = []
            for t in range(L):
                v = v * decay + cur[t]
                acc.append(v)
            return torch.stack(acc)

        idx = torch.arange(L, device="cuda")
        expo = (idx.unsqueeze(1) - idx.unsqueeze(0)).float()
        K = torch.where(expo >= 0, torch.pow(
            torch.tensor(decay, device="cuda"), expo), torch.zeros_like(expo))

        def toeplitz():
            return torch.einsum("ls,sbh->lbh", K, cur)

        with torch.no_grad():
            ref = sequential()
            got = toeplitz()
            seq_t = timeit(sequential, iters=5, warmup=2)
            top_t = timeit(toeplitz, iters=20, warmup=5)

        out[f"L={L},B={B},H={H}"] = {
            "sequential_ms": round(seq_t * 1e3, 3),
            "toeplitz_ms": round(top_t * 1e3, 3),
            "speedup": round(seq_t / top_t, 1),
            "max_abs_err": float((ref - got).abs().max()),
        }

    # Underflow hazard: where does decay^n vanish in fp32?
    uf = {}
    for decay in (0.9, 0.95, 0.99):
        n = 1
        while n < 100000:
            if float(torch.tensor(decay, dtype=torch.float32) ** n) == 0.0:
                break
            n *= 2
        lo, hi = n // 2, n
        while lo < hi:
            mid = (lo + hi) // 2
            if float(torch.tensor(decay, dtype=torch.float32) ** mid) == 0.0:
                hi = mid
            else:
                lo = mid + 1
        uf[f"decay={decay}"] = {
            "underflows_to_zero_at_n": lo,
            "value_at_n1023": float(torch.tensor(decay, dtype=torch.float32) ** 1023),
        }
    out["fp32_underflow"] = uf
    return out


def main():
    assert torch.cuda.is_available()
    print("== C1 jiterator ==", flush=True)
    RESULTS["C1_jiterator"] = verify_jiterator()
    print(json.dumps(RESULTS["C1_jiterator"], indent=2), flush=True)

    print("\n== C2 precision at threshold ==", flush=True)
    RESULTS["C2_precision"] = verify_precision()
    print(json.dumps(RESULTS["C2_precision"], indent=2), flush=True)

    print("\n== C3 toeplitz scan ==", flush=True)
    RESULTS["C3_toeplitz"] = verify_toeplitz()
    print(json.dumps(RESULTS["C3_toeplitz"], indent=2), flush=True)

    out_path = sys.argv[1] if len(sys.argv) > 1 else "verify_claims.json"
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(RESULTS, fh, indent=2)
    print(f"\nWROTE {out_path}")


if __name__ == "__main__":
    main()
