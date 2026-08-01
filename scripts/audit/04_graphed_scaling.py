"""Phase-1 reconnaissance: where does the free lunch end INSIDE a CUDA graph?

Eager-mode measurements showed the spiking loop is bound by CPU-side dispatch,
so batch and width were nearly free. CUDA-graph capture removes that dispatch
cost (~11x). The question this answers: after graphing, is the loop finally
compute/bandwidth-bound -- or is it STILL latency-bound, meaning we should
scale batch and width much further at almost no wall-clock cost?

The answer sizes the final model, so it is worth measuring rather than assuming.
"""

import json
import sys
import time

import torch
import torch.nn as nn


def timeit(fn, iters=20, warmup=5):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters


class LIF(nn.Module):
    def __init__(self, d_in, d_h):
        super().__init__()
        self.fc = nn.Linear(d_in, d_h, bias=False)

    def forward(self, x, mem):
        mem = mem * 0.9 + self.fc(x)
        shifted = mem - 1.0
        spike = (shifted > 0).to(mem.dtype)
        sg = torch.sigmoid(shifted * 4.0)
        spike = spike + (sg - sg.detach())
        return spike, mem - spike


def build_and_graph(batch, d, L, vocab=205, layers=2, dtype=torch.float32):
    """Capture one full fwd+bwd+AdamW step of a small stacked-LIF char LM."""
    emb = nn.Embedding(vocab, d).cuda()
    cells = nn.ModuleList([LIF(d, d) for _ in range(layers)]).cuda()
    head = nn.Linear(d, vocab).cuda()
    params = list(emb.parameters()) + list(cells.parameters()) + list(head.parameters())
    n_params = sum(p.numel() for p in params)
    opt = torch.optim.AdamW(params, lr=1e-3, capturable=True)

    inp = torch.randint(0, vocab, (L, batch), device="cuda")
    tgt = torch.randint(0, vocab, (L, batch), device="cuda")

    def full_step():
        opt.zero_grad(set_to_none=False)
        mems = [torch.zeros(batch, d, device="cuda") for _ in range(layers)]
        loss = 0.0
        for t in range(L):
            h = emb(inp[t])
            for i, c in enumerate(cells):
                h, mems[i] = c(h, mems[i])
            loss = loss + nn.functional.cross_entropy(head(h), tgt[t])
        (loss / L).backward()
        opt.step()

    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            full_step()
    torch.cuda.current_stream().wait_stream(s)

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        full_step()
    return g, n_params, (emb, cells, head, opt, inp, tgt)


def sweep(name, configs):
    out = {}
    for cfg in configs:
        batch, d, L = cfg["batch"], cfg["d"], cfg["L"]
        key = f"batch={batch},d={d},L={L}"
        try:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            g, n_params, keep = build_and_graph(batch, d, L)
            dt = timeit(g.replay, iters=10, warmup=3)
            tok_s = batch * L / dt
            out[key] = {
                "params_M": round(n_params / 1e6, 2),
                "ms": round(dt * 1e3, 2),
                "tok_per_s": round(tok_s),
                "Mtok_per_hour": round(tok_s * 3600 / 1e6),
                "peak_vram_gib": round(torch.cuda.max_memory_allocated() / 1024**3, 3),
            }
            print(f"  {key:32s} -> {out[key]}", flush=True)
            del g, keep
            torch.cuda.empty_cache()
        except Exception as exc:
            out[key] = f"{type(exc).__name__}: {str(exc)[:160]}"
            print(f"  {key:32s} -> {out[key]}", flush=True)
            torch.cuda.empty_cache()
    return out


def main():
    torch.manual_seed(0)
    results = {}

    print("== graphed batch scaling (d=512, L=128) ==", flush=True)
    results["batch"] = sweep("batch", [
        {"batch": b, "d": 512, "L": 128} for b in (16, 32, 64, 128, 256, 512)
    ])

    print("== graphed width scaling (batch=64, L=128) ==", flush=True)
    results["width"] = sweep("width", [
        {"batch": 64, "d": d, "L": 128} for d in (256, 512, 768, 1024, 1536)
    ])

    print("== candidate final configs ==", flush=True)
    results["candidates"] = sweep("cand", [
        {"batch": 128, "d": 512, "L": 256},
        {"batch": 256, "d": 512, "L": 256},
        {"batch": 128, "d": 1024, "L": 256},
        {"batch": 256, "d": 768, "L": 512},
    ])

    out_path = sys.argv[1] if len(sys.argv) > 1 else "graphed_scaling.json"
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)
    print(f"\nWROTE {out_path}")


if __name__ == "__main__":
    main()
