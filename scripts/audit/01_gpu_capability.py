"""Phase-1 reconnaissance: empirical hardware/software capability audit.

Measures the numbers that actually gate a BPTT-trained spiking char-LM on this
box, rather than assuming them: peak matmul throughput, memory bandwidth,
kernel-launch overhead, CUDA-graph capture, and torch.compile availability.
"""

import json
import platform
import sys
import time

import torch

RESULTS = {}


def sync():
    torch.cuda.synchronize()


def timeit(fn, iters=50, warmup=10):
    for _ in range(warmup):
        fn()
    sync()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    sync()
    return (time.perf_counter() - t0) / iters


def bench_matmul():
    """Peak achievable throughput per dtype/precision mode."""
    out = {}
    n = 4096
    flops = 2 * n**3
    for label, dtype, tf32 in [
        ("fp32", torch.float32, False),
        ("tf32", torch.float32, True),
        ("bf16", torch.bfloat16, False),
        ("fp16", torch.float16, False),
    ]:
        torch.backends.cuda.matmul.allow_tf32 = tf32
        torch.backends.cudnn.allow_tf32 = tf32
        try:
            a = torch.randn(n, n, device="cuda", dtype=dtype)
            b = torch.randn(n, n, device="cuda", dtype=dtype)
            dt = timeit(lambda: a @ b, iters=30)
            out[label] = round(flops / dt / 1e12, 2)  # TFLOP/s
            del a, b
            torch.cuda.empty_cache()
        except Exception as exc:  # pragma: no cover - capability probe
            out[label] = f"FAILED: {type(exc).__name__}: {exc}"
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    return out


def bench_bandwidth():
    """Effective HBM bandwidth via a large elementwise read-modify-write."""
    n = 64 * 1024 * 1024  # 64M fp32 = 256 MiB
    a = torch.randn(n, device="cuda")
    b = torch.randn(n, device="cuda")
    dt = timeit(lambda: torch.add(a, b), iters=30)
    # read a, read b, write out = 3 * 4 bytes per element
    gbps = 3 * 4 * n / dt / 1e9
    del a, b
    torch.cuda.empty_cache()
    return round(gbps, 1)


def bench_launch_overhead():
    """The number that gates sequential SNN timestep loops.

    A tiny elementwise op on a small tensor is pure launch cost: the kernel
    itself is nanoseconds. This is the per-timestep floor for an unfused
    LIF update.
    """
    x = torch.randn(256, 512, device="cuda")
    y = torch.randn(256, 512, device="cuda")

    def one_op():
        torch.add(x, y)

    def lif_step():
        # Representative unfused LIF update: decay, integrate, compare,
        # cast, subtract-reset -> ~5 elementwise kernels.
        m = x * 0.9
        m = m + y
        s = (m > 1.0).to(m.dtype)
        m = m - s
        return m

    per_op = timeit(one_op, iters=200, warmup=50)
    per_lif = timeit(lif_step, iters=200, warmup=50)
    del x, y
    torch.cuda.empty_cache()
    return {
        "single_elementwise_us": round(per_op * 1e6, 2),
        "unfused_lif_step_us": round(per_lif * 1e6, 2),
    }


def bench_cuda_graph():
    """Can we amortise launch overhead with CUDA graphs?"""
    try:
        x = torch.randn(256, 512, device="cuda")
        y = torch.randn(256, 512, device="cuda")
        static_m = torch.zeros_like(x)

        # warmup on a side stream (required before capture)
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                m = x * 0.9 + y
                sp = (m > 1.0).to(m.dtype)
                static_m.copy_(m - sp)
        torch.cuda.current_stream().wait_stream(s)

        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            for _ in range(20):  # 20 fused timesteps in one graph
                m = x * 0.9 + y
                sp = (m > 1.0).to(m.dtype)
                static_m.copy_(m - sp)

        dt = timeit(g.replay, iters=100, warmup=20)
        del x, y, static_m
        torch.cuda.empty_cache()
        return {
            "capture": "OK",
            "per_20_step_graph_us": round(dt * 1e6, 2),
            "per_step_us": round(dt * 1e6 / 20, 3),
        }
    except Exception as exc:
        return {"capture": f"FAILED: {type(exc).__name__}: {exc}"}


def bench_compile():
    """Does torch.compile actually produce fused CUDA code without Triton?"""
    out = {}
    try:
        import triton  # noqa: F401

        out["triton_importable"] = True
    except Exception as exc:
        out["triton_importable"] = f"{type(exc).__name__}: {exc}"

    def f(m, i):
        m = m * 0.9 + i
        s = (m > 1.0).to(m.dtype)
        return m - s, s

    x = torch.randn(256, 512, device="cuda")
    y = torch.randn(256, 512, device="cuda")
    for backend in ("inductor", "aot_eager", "cudagraphs"):
        try:
            cf = torch.compile(f, backend=backend)
            cf(x, y)  # trigger compilation
            sync()
            dt = timeit(lambda: cf(x, y), iters=100, warmup=20)
            out[backend] = {"status": "OK", "us": round(dt * 1e6, 2)}
        except Exception as exc:
            msg = f"{type(exc).__name__}: {exc}"
            out[backend] = {"status": "FAILED", "error": msg[:400]}
        finally:
            torch.compiler.reset()
    eager = timeit(lambda: f(x, y), iters=100, warmup=20)
    out["eager_us"] = round(eager * 1e6, 2)
    del x, y
    torch.cuda.empty_cache()
    return out


def bench_amp():
    out = {}
    for label, dtype in [("bf16", torch.bfloat16), ("fp16", torch.float16)]:
        try:
            with torch.autocast("cuda", dtype=dtype):
                a = torch.randn(512, 512, device="cuda")
                b = torch.randn(512, 512, device="cuda")
                c = a @ b
                sync()
            out[label] = str(c.dtype)
        except Exception as exc:
            out[label] = f"FAILED: {type(exc).__name__}: {exc}"
    return out


def main():
    assert torch.cuda.is_available(), "CUDA required for this audit"
    props = torch.cuda.get_device_properties(0)
    RESULTS["env"] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "gpu": props.name,
        "sm": f"{props.major}.{props.minor}",
        "vram_gib": round(props.total_memory / 1024**3, 2),
        "sms": props.multi_processor_count,
    }
    print("== env ==", json.dumps(RESULTS["env"], indent=2), flush=True)

    RESULTS["matmul_tflops"] = bench_matmul()
    print("== matmul ==", json.dumps(RESULTS["matmul_tflops"], indent=2), flush=True)

    RESULTS["bandwidth_gbps"] = bench_bandwidth()
    print("== bandwidth ==", RESULTS["bandwidth_gbps"], flush=True)

    RESULTS["launch_overhead"] = bench_launch_overhead()
    print("== launch ==", json.dumps(RESULTS["launch_overhead"], indent=2), flush=True)

    RESULTS["cuda_graph"] = bench_cuda_graph()
    print("== cudagraph ==", json.dumps(RESULTS["cuda_graph"], indent=2), flush=True)

    RESULTS["torch_compile"] = bench_compile()
    print("== compile ==", json.dumps(RESULTS["torch_compile"], indent=2), flush=True)

    RESULTS["amp"] = bench_amp()
    print("== amp ==", json.dumps(RESULTS["amp"], indent=2), flush=True)

    RESULTS["peak_vram_alloc_gib"] = round(
        torch.cuda.max_memory_allocated() / 1024**3, 3
    )
    out_path = sys.argv[1] if len(sys.argv) > 1 else "audit_results.json"
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(RESULTS, fh, indent=2)
    print(f"\nWROTE {out_path}")


if __name__ == "__main__":
    main()
