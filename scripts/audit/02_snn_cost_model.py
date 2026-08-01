"""Phase-1 reconnaissance: cost model for a sequential spiking timestep loop.

The central question for a BPTT-trained spiking char-LM on this box is whether
the inner loop (L sequence steps x T micro-steps of small elementwise kernels)
is bound by arithmetic, by memory bandwidth, or by kernel-launch latency.
The answer determines every optimisation decision downstream, so measure it.

Key probe: if wall-clock is flat as batch size grows, the loop is launch-bound
and batch is free -- the single most important lever available.
"""

import json
import sys
import time

import torch
import torch.nn as nn

RESULTS = {}


def timeit(fn, iters=30, warmup=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters


def pure_launch_overhead():
    """Isolate launch latency from bandwidth using a genuinely tiny tensor."""
    out = {}
    for n in (16, 256, 4096, 65536, 1048576):
        x = torch.randn(n, device="cuda")
        y = torch.randn(n, device="cuda")
        dt = timeit(lambda: torch.add(x, y), iters=300, warmup=100)
        bytes_moved = 3 * 4 * n
        out[f"n={n}"] = {
            "us": round(dt * 1e6, 2),
            "eff_gbps": round(bytes_moved / dt / 1e9, 1),
        }
        del x, y
    torch.cuda.empty_cache()
    return out


class LIFCell(nn.Module):
    """Canonical LIF: leak -> integrate -> threshold -> subtract-reset.

    Surrogate gradient is the standard fast-sigmoid / arctan style straight-
    through estimator; this is the shape whose cost we care about.
    """

    def __init__(self, d_in, d_hidden, decay=0.9, thresh=1.0):
        super().__init__()
        self.fc = nn.Linear(d_in, d_hidden, bias=False)
        self.decay = decay
        self.thresh = thresh

    def forward(self, x, mem):
        cur = self.fc(x)
        mem = mem * self.decay + cur
        # surrogate-gradient spike (straight-through with arctan derivative)
        shifted = mem - self.thresh
        spike = (shifted > 0).to(mem.dtype)
        spike = spike + (torch.atan(shifted * 2.0) / 3.14159 -
                         torch.atan(shifted * 2.0).detach() / 3.14159)
        mem = mem - spike * self.thresh
        return spike, mem


def snn_forward(cell, x_seq, batch, d_hidden, steps):
    mem = torch.zeros(batch, d_hidden, device="cuda")
    outs = 0.0
    for t in range(steps):
        spike, mem = cell(x_seq[t], mem)
        outs = outs + spike.sum()
    return outs


def batch_scaling(d_hidden=512, steps=64):
    """If time is flat in batch, we are launch-bound and batch is ~free."""
    out = {}
    cell = LIFCell(d_hidden, d_hidden).cuda()
    for batch in (1, 8, 32, 128, 512):
        try:
            x_seq = torch.randn(steps, batch, d_hidden, device="cuda")

            def fwd():
                with torch.no_grad():
                    snn_forward(cell, x_seq, batch, d_hidden, steps)

            dt = timeit(fwd, iters=10, warmup=3)
            out[f"batch={batch}"] = {
                "ms": round(dt * 1e3, 2),
                "us_per_step": round(dt * 1e6 / steps, 2),
                "tok_per_s": round(batch * steps / dt),
            }
            del x_seq
            torch.cuda.empty_cache()
        except torch.OutOfMemoryError:
            out[f"batch={batch}"] = "OOM"
            torch.cuda.empty_cache()
    return out


def width_scaling(batch=64, steps=64):
    """If time is flat in width, the matmul is not the cost -- launch is."""
    out = {}
    for d in (128, 256, 512, 1024, 2048):
        try:
            cell = LIFCell(d, d).cuda()
            x_seq = torch.randn(steps, batch, d, device="cuda")

            def fwd():
                with torch.no_grad():
                    snn_forward(cell, x_seq, batch, d, steps)

            dt = timeit(fwd, iters=10, warmup=3)
            out[f"d={d}"] = {
                "ms": round(dt * 1e3, 2),
                "us_per_step": round(dt * 1e6 / steps, 2),
            }
            del cell, x_seq
            torch.cuda.empty_cache()
        except torch.OutOfMemoryError:
            out[f"d={d}"] = "OOM"
            torch.cuda.empty_cache()
    return out


def train_step_cost(batch=64, d_hidden=512, seq_lens=(64, 128, 256, 512)):
    """Full BPTT fwd+bwd: the real training-throughput number, plus VRAM."""
    out = {}
    for L in seq_lens:
        try:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            cell = LIFCell(d_hidden, d_hidden).cuda()
            head = nn.Linear(d_hidden, 256).cuda()
            opt = torch.optim.AdamW(
                list(cell.parameters()) + list(head.parameters()), lr=1e-3
            )
            x_seq = torch.randn(L, batch, d_hidden, device="cuda")
            targets = torch.randint(0, 256, (L, batch), device="cuda")

            def step():
                opt.zero_grad(set_to_none=True)
                mem = torch.zeros(batch, d_hidden, device="cuda")
                loss = 0.0
                for t in range(L):
                    spike, mem = cell(x_seq[t], mem)
                    logits = head(spike)
                    loss = loss + nn.functional.cross_entropy(logits, targets[t])
                (loss / L).backward()
                opt.step()

            dt = timeit(step, iters=5, warmup=2)
            out[f"L={L}"] = {
                "ms_per_step": round(dt * 1e3, 1),
                "tok_per_s": round(batch * L / dt),
                "peak_vram_gib": round(torch.cuda.max_memory_allocated() / 1024**3, 3),
            }
            del cell, head, opt, x_seq, targets
            torch.cuda.empty_cache()
        except torch.OutOfMemoryError:
            out[f"L={L}"] = "OOM"
            torch.cuda.empty_cache()
    return out


def graph_capture_training():
    """Can a full fwd+bwd+opt training step be captured in a CUDA graph?

    This is the only fusion mechanism available without Triton/MSVC, so its
    viability is load-bearing for the whole optimisation plan.
    """
    batch, d, L = 64, 512, 32
    try:
        cell = LIFCell(d, d).cuda()
        head = nn.Linear(d, 256).cuda()
        params = list(cell.parameters()) + list(head.parameters())
        opt = torch.optim.AdamW(params, lr=1e-3, capturable=True)
        x_seq = torch.randn(L, batch, d, device="cuda")
        targets = torch.randint(0, 256, (L, batch), device="cuda")

        def full_step():
            opt.zero_grad(set_to_none=False)
            mem = torch.zeros(batch, d, device="cuda")
            loss = 0.0
            for t in range(L):
                spike, mem = cell(x_seq[t], mem)
                loss = loss + nn.functional.cross_entropy(head(spike), targets[t])
            (loss / L).backward()
            opt.step()

        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                full_step()
        torch.cuda.current_stream().wait_stream(s)

        eager = timeit(full_step, iters=5, warmup=2)

        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            full_step()
        graphed = timeit(g.replay, iters=10, warmup=3)

        return {
            "status": "OK",
            "eager_ms": round(eager * 1e3, 1),
            "graphed_ms": round(graphed * 1e3, 1),
            "speedup": round(eager / graphed, 2),
        }
    except Exception as exc:
        return {"status": "FAILED", "error": f"{type(exc).__name__}: {str(exc)[:400]}"}


def main():
    torch.manual_seed(0)
    RESULTS["pure_launch"] = pure_launch_overhead()
    print("== pure_launch ==", json.dumps(RESULTS["pure_launch"], indent=2), flush=True)

    RESULTS["batch_scaling"] = batch_scaling()
    print("== batch ==", json.dumps(RESULTS["batch_scaling"], indent=2), flush=True)

    RESULTS["width_scaling"] = width_scaling()
    print("== width ==", json.dumps(RESULTS["width_scaling"], indent=2), flush=True)

    RESULTS["train_step"] = train_step_cost()
    print("== train ==", json.dumps(RESULTS["train_step"], indent=2), flush=True)

    RESULTS["graph_training"] = graph_capture_training()
    print("== graph_train ==", json.dumps(RESULTS["graph_training"], indent=2), flush=True)

    out_path = sys.argv[1] if len(sys.argv) > 1 else "snn_cost.json"
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(RESULTS, fh, indent=2)
    print(f"\nWROTE {out_path}")


if __name__ == "__main__":
    main()
