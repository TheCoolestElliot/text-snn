"""Phase-1 reconnaissance: is per-timestep cost simply proportional to kernel count?

The batch/width sweeps showed the spiking inner loop is launch-bound. If that is
true, wall-clock should track the NUMBER of elementwise kernels per timestep and
be nearly independent of their size. That converts optimisation from "make the
math cheaper" into "issue fewer kernels", which is a completely different plan.

Also probes the two algorithmic levers available without Triton/nvcc:
  (a) hoisting the input projection out of the timestep loop (one big matmul for
      all L positions instead of L small ones), and
  (b) how far CUDA-graph capture scales in unroll length.
"""

import json
import sys
import time

import torch
import torch.nn as nn

RESULTS = {}


def timeit(fn, iters=20, warmup=5):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters


def kernel_count_sweep(batch=64, d=512, steps=64):
    """Same tensor sizes, deliberately varying only the op count per step."""
    out = {}
    x = torch.randn(batch, d, device="cuda")
    for n_ops in (1, 2, 4, 8, 16, 32):
        def loop(n=n_ops):
            m = torch.zeros(batch, d, device="cuda")
            for _ in range(steps):
                for _ in range(n):
                    m = m * 0.9
            return m

        with torch.no_grad():
            dt = timeit(loop, iters=10, warmup=3)
        total_kernels = steps * n_ops
        out[f"ops_per_step={n_ops}"] = {
            "ms": round(dt * 1e3, 2),
            "us_per_kernel": round(dt * 1e6 / total_kernels, 2),
        }
    del x
    torch.cuda.empty_cache()
    return out


class NaiveLIF(nn.Module):
    """Readable, idiomatic LIF -- the version most people write first."""

    def __init__(self, d_in, d_h):
        super().__init__()
        self.fc = nn.Linear(d_in, d_h, bias=False)

    def forward(self, x, mem):
        cur = self.fc(x)
        mem = mem * 0.9 + cur
        shifted = mem - 1.0
        spike = (shifted > 0).to(mem.dtype)
        sg = torch.sigmoid(shifted * 4.0)
        spike = spike + (sg - sg.detach())
        mem = mem - spike * 1.0
        return spike, mem


class LeanLIF(nn.Module):
    """Algebraically identical, written to issue as few kernels as possible.

    Input current is precomputed for ALL timesteps outside the loop, and the
    membrane update is folded so the loop body is a handful of ops.
    """

    def __init__(self, d_in, d_h):
        super().__init__()
        self.fc = nn.Linear(d_in, d_h, bias=False)

    def precompute(self, x_seq):
        # ONE matmul for every timestep at once: (L,B,d_in) -> (L,B,d_h)
        return self.fc(x_seq)

    def step(self, cur_t, mem):
        mem = torch.addcmul(cur_t, mem, torch.tensor(0.9, device=mem.device))
        shifted = mem - 1.0
        spike = torch.heaviside(shifted, torch.zeros((), device=mem.device))
        sg = torch.sigmoid(shifted * 4.0)
        spike = spike.detach() + sg - sg.detach()
        mem = mem - spike
        return spike, mem


def hoisting_benefit(batch=64, d=512, L=128):
    """Does precomputing all input currents in one matmul actually pay?"""
    out = {}
    x_seq = torch.randn(L, batch, d, device="cuda")

    naive = NaiveLIF(d, d).cuda()
    lean = LeanLIF(d, d).cuda()

    def run_naive():
        mem = torch.zeros(batch, d, device="cuda")
        acc = 0.0
        for t in range(L):
            s, mem = naive(x_seq[t], mem)
            acc = acc + s.sum()
        return acc

    def run_lean():
        cur = lean.precompute(x_seq)
        mem = torch.zeros(batch, d, device="cuda")
        acc = 0.0
        for t in range(L):
            s, mem = lean.step(cur[t], mem)
            acc = acc + s.sum()
        return acc

    with torch.no_grad():
        out["naive_ms"] = round(timeit(run_naive, iters=10, warmup=3) * 1e3, 2)
        out["lean_hoisted_ms"] = round(timeit(run_lean, iters=10, warmup=3) * 1e3, 2)
    out["speedup"] = round(out["naive_ms"] / out["lean_hoisted_ms"], 2)

    del x_seq, naive, lean
    torch.cuda.empty_cache()
    return out


def graph_unroll_scaling(batch=64, d=512):
    """How long an unroll can we capture, and does capture cost scale linearly?"""
    out = {}
    for L in (32, 64, 128, 256, 512):
        try:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            cell = NaiveLIF(d, d).cuda()
            head = nn.Linear(d, 96).cuda()
            params = list(cell.parameters()) + list(head.parameters())
            opt = torch.optim.AdamW(params, lr=1e-3, capturable=True)
            x_seq = torch.randn(L, batch, d, device="cuda")
            tgt = torch.randint(0, 96, (L, batch), device="cuda")

            def full_step():
                opt.zero_grad(set_to_none=False)
                mem = torch.zeros(batch, d, device="cuda")
                loss = 0.0
                for t in range(L):
                    s, mem = cell(x_seq[t], mem)
                    loss = loss + nn.functional.cross_entropy(head(s), tgt[t])
                (loss / L).backward()
                opt.step()

            st = torch.cuda.Stream()
            st.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(st):
                for _ in range(3):
                    full_step()
            torch.cuda.current_stream().wait_stream(st)
            eager = timeit(full_step, iters=3, warmup=1)

            t0 = time.perf_counter()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                full_step()
            capture_s = time.perf_counter() - t0

            graphed = timeit(g.replay, iters=10, warmup=3)
            out[f"L={L}"] = {
                "eager_ms": round(eager * 1e3, 1),
                "graphed_ms": round(graphed * 1e3, 2),
                "speedup": round(eager / graphed, 1),
                "capture_s": round(capture_s, 2),
                "tok_per_s_graphed": round(batch * L / graphed),
                "peak_vram_gib": round(torch.cuda.max_memory_allocated() / 1024**3, 3),
            }
            del g, cell, head, opt, x_seq, tgt
            torch.cuda.empty_cache()
        except Exception as exc:
            out[f"L={L}"] = f"{type(exc).__name__}: {str(exc)[:200]}"
            torch.cuda.empty_cache()
    return out


def main():
    torch.manual_seed(0)
    RESULTS["kernel_count"] = kernel_count_sweep()
    print("== kernel_count ==", json.dumps(RESULTS["kernel_count"], indent=2), flush=True)

    RESULTS["hoisting"] = hoisting_benefit()
    print("== hoisting ==", json.dumps(RESULTS["hoisting"], indent=2), flush=True)

    RESULTS["graph_unroll"] = graph_unroll_scaling()
    print("== graph_unroll ==", json.dumps(RESULTS["graph_unroll"], indent=2), flush=True)

    out_path = sys.argv[1] if len(sys.argv) > 1 else "kernel_count.json"
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(RESULTS, fh, indent=2)
    print(f"\nWROTE {out_path}")


if __name__ == "__main__":
    main()
