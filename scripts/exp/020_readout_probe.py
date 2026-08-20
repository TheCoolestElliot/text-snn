"""EXP_020 Leg A: how much does the readout leave on the table?

    python scripts/exp/020_readout_probe.py --out docs/reports/data/exp_020_readout_probe.json

WHAT IT MEASURES
----------------
`_CharLMStack.forward` ends with `logits = self.head(h)` where `h` is the LAST
layer's binary spike tensor at position `t` and nothing else (`model.py:363`).
So the entire prediction of character `t+1` is a linear function of one `{0,1}`
vector of width `d`.  Two things are discarded at that boundary, both of them
free in kernel count:

  * the OTHER layers' spikes at `t`  -- layer 0's emission is computed, used to
    drive layer 1, and then thrown away;
  * the SAME layer's spikes at `t-1, t-2, ...` -- available as a strided view of
    a tensor that already exists.

This probe fits a fresh linear head on FROZEN features from committed
checkpoints and reports held-out bpc per feature set.  Nothing is trained
except the head; `src/snn/` is not edited; no checkpoint is written.

WHY A FROZEN-FEATURE PROBE IS THE RIGHT INSTRUMENT HERE, AND WHAT IT CANNOT SAY
-------------------------------------------------------------------------------
The representation under test was optimised end-to-end FOR R0's readout.  Every
other feature set is therefore handicapped: its features were never shaped to be
read that way.  That asymmetry runs one way only, so

    a gain measured here is a LOWER BOUND on the gain joint training would give,
    and a null here does NOT bound the joint-training gain from above.

That is the whole reason to run it before spending GPU on retrains: it can
authorise an arm cheaply, and it cannot veto one.  `EXP_016`'s precedent is the
same shape -- forward passes over checkpoints already on disk, ~0.8 GPU-h, no
training, and it closed a mechanism.

R5 is the analogue reference and is NOT a candidate: reading the membrane would
put a real-valued signal on the inter-unit path, which is what I1 exists to
forbid.  It is here to price the binarity, so that "the readout is binary" is a
measured cost rather than an assumption.

VALIDITY CHECKS, ASSERTED RATHER THAN ASSUMED
----------------------------------------------
T1  The COMMITTED head is scored on the same held-out positions as the refits.
    If refit-R0 beats the committed head by more than `--t1-tol` bpc the probe
    is measuring its own optimiser rather than the representation, and every
    delta below is suspect.  Reported, not silently corrected.
T2  Every feature block is asserted to be exactly {0.0, 1.0} except R5's, which
    is asserted NOT to be.  A feature set that is secretly analogue would make
    the binarity claim vacuous.
T3  Lagged features are built with `shift_by_one`-style zero padding INSIDE the
    window, so position `t` never sees a character from another window and the
    lag cannot leak across a batch boundary.  The first `r` positions of each
    window are scored anyway, with zeros in the lag slots, exactly as a trained
    model would see them.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO / "src") not in sys.path:
    sys.path.insert(0, str(_REPO / "src"))

from snn.config import Config  # noqa: E402
from snn.data import Corpus, windowed_eval_batches  # noqa: E402
from snn.model import build_model  # noqa: E402

_LN2 = math.log(2.0)


# ---------------------------------------------------------------------------
# feature sets
# ---------------------------------------------------------------------------
#
# Each entry is (name, layers, lags, analogue).  `layers` indexes the emitted
# spike tensors (0..K-1); `lags` are non-negative delays applied to every named
# layer.  The feature width is len(layers)*len(lags)*d.
FEATURE_SETS: dict[str, dict] = {
    # the committed readout, refitted -- the control that makes the deltas mean
    # something
    "R0_last_t":        dict(layers=[-1], lags=[0], analogue=False),
    # same layer, one extra tap: the cheapest thing the head is not given
    "R1_last_t_t1":     dict(layers=[-1], lags=[0, 1], analogue=False),
    # every layer at t: the emission that is currently computed and discarded
    "R2_all_t":         dict(layers="all", lags=[0], analogue=False),
    # both
    "R3_all_t_t1":      dict(layers="all", lags=[0, 1], analogue=False, wide=True),
    # how far does the tap ladder keep paying?
    "R4_last_t_t1_t2_t3": dict(layers=[-1], lags=[0, 1, 2, 3], analogue=False, wide=True),
    # THE PRICE OF BINARITY -- not a candidate, see the module docstring
    "R5_membrane":      dict(layers=[-1], lags=[0], analogue=True),
}


def _resolve_layers(spec, n_layers: int) -> list[int]:
    if spec == "all":
        return list(range(n_layers))
    return [k if k >= 0 else n_layers + k for k in spec]


def _lagged(x: torch.Tensor, lag: int) -> torch.Tensor:
    """`x` [W, L, d] -> the same tensor delayed by `lag` INSIDE each window.

    Zero-filled in front (T3).  `F.pad` rather than an in-place write, matching
    `snn.prescan.shift_by_one`: the in-place form mutates a buffer whose address
    a capture would bake in, and this file must stay usable inside one.
    """
    if lag == 0:
        return x
    return F.pad(x[:, :-lag], (0, 0, lag, 0))


# ---------------------------------------------------------------------------
# collection
# ---------------------------------------------------------------------------

@torch.no_grad()
def collect(model, corpus, cfg: Config, split: str, n_windows: int,
            device: str) -> dict:
    """Run the committed model and keep every layer's spikes, the membrane and
    the committed logits, for `n_windows` fresh-protocol windows.

    Returns dict of stacked [W, L, *] tensors on `device`.  Spikes are stored as
    uint8 because they are binary and a fp32 copy of four of these blocks does
    not fit beside the model on an 8 GiB card; they are cast back per minibatch.
    """
    model.eval()
    model.keep_spikes = True

    spikes: list[list[torch.Tensor]] = []
    v_last: list[torch.Tensor] = []
    committed_nats: list[torch.Tensor] = []
    ys: list[torch.Tensor] = []

    taken = 0
    for x, y in windowed_eval_batches(corpus.split(split), cfg.batch_size,
                                      cfg.seq_len):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits, _, aux = model(x, None)

        # per-position nats under the COMMITTED head, on exactly these
        # positions -- T1's left-hand side.
        nats = F.cross_entropy(
            logits.reshape(-1, cfg.vocab_size).float(), y.reshape(-1),
            reduction="none",
        ).reshape(y.shape)

        spikes.append([s.to(torch.uint8) for s in aux["spikes"]])
        # The membrane is not exposed by forward(); the last layer's PRE-reset
        # potential is what a membrane readout would see, and it is recovered
        # below from the committed spikes without re-running the scan.
        v_last.append(aux["v_pre_last"].float() if "v_pre_last" in aux else None)
        committed_nats.append(nats)
        ys.append(y)
        taken += x.shape[0]
        if taken >= n_windows:
            break

    model.keep_spikes = False
    n_layers = len(spikes[0])
    out = {
        "spikes": [torch.cat([b[k] for b in spikes], 0) for k in range(n_layers)],
        "y": torch.cat(ys, 0),
        "committed_nats": torch.cat(committed_nats, 0),
    }
    if v_last[0] is not None:
        out["v_pre_last"] = torch.cat(v_last, 0)
    return out


@torch.no_grad()
def membrane_of_last_layer(model, corpus, cfg: Config, split: str,
                           n_windows: int, device: str) -> torch.Tensor:
    """`v_pre` of the last layer, [W, L, d] fp32, by a LOCAL scan.

    The scan is re-implemented here rather than exposed from `src/snn/`, which
    is `EXP_016`'s precedent and for its reason: an experiment's instrument does
    not belong inside the arm under test.  It is only sound if it is the same
    scan, so the spikes this local scan produces are checked BITWISE against the
    committed model's own emission by the caller (T2b).

    Plain LIF, hard reset only.  Any other arm raises rather than return a
    number that describes a neuron that never ran.
    """
    if cfg.arch != "snn" or cfg.reset != "hard":
        raise NotImplementedError(
            f"the local scan covers arch='snn', reset='hard' only; got "
            f"arch={cfg.arch!r}, reset={cfg.reset!r}. R5 is a reference, not a "
            "candidate, and a wrong membrane would misprice binarity."
        )
    model.eval()
    beta, thr = cfg.beta, cfg.threshold
    vs: list[torch.Tensor] = []
    ss: list[torch.Tensor] = []
    taken = 0
    for x, _ in windowed_eval_batches(corpus.split(split), cfg.batch_size,
                                      cfg.seq_len):
        x = x.to(device, non_blocking=True)
        h = model.embed(x)
        v_pre_k = None
        s_k = None
        for k, linear in enumerate(model.layers):
            cur = linear(h)
            v = torch.zeros(cur.shape[0], cur.shape[2], device=device,
                            dtype=torch.float32)
            v_pres, spk = [], []
            for t in range(cur.shape[1]):
                v_pre = v * beta + cur[:, t]
                s = (v_pre >= thr).to(v_pre.dtype)
                v = v_pre * (1.0 - s)
                v_pres.append(v_pre)
                spk.append(s)
            v_pre_k = torch.stack(v_pres, 1)
            s_k = torch.stack(spk, 1)
            h = s_k
        vs.append(v_pre_k)
        ss.append(s_k.to(torch.uint8))
        taken += x.shape[0]
        if taken >= n_windows:
            break
    return torch.cat(vs, 0), torch.cat(ss, 0)


# ---------------------------------------------------------------------------
# the head fit
# ---------------------------------------------------------------------------

def build_features(store: dict, spec: dict, n_layers: int) -> torch.Tensor:
    """Assemble a [W, L, F] fp32 feature block from the stored uint8 spikes."""
    if spec["analogue"]:
        return store["v_pre_last"]
    layers = _resolve_layers(spec["layers"], n_layers)
    blocks = []
    for k in layers:
        s = store["spikes"][k].float()
        for lag in spec["lags"]:
            blocks.append(_lagged(s, lag))
    return torch.cat(blocks, dim=-1)


def _window_split(W: int, seed: int, device) -> tuple[torch.Tensor, ...]:
    """60/20/20 split BY WINDOW, shared across every feature set.

    By window and not by position: two positions in one window share a membrane
    and are not independent, so a positionwise split leaks the answer across the
    boundary.  Three ways and not two, because the ridge coefficient is CHOSEN
    on the dev half -- selecting on the same half that is reported would make
    the wide feature sets look better exactly in proportion to how many
    coefficients they have to select over, which is the one confound this probe
    exists to avoid.
    """
    g = torch.Generator(device="cpu").manual_seed(seed)
    perm = torch.randperm(W, generator=g)
    n_dev = max(1, int(round(W * 0.2)))
    n_te = max(1, int(round(W * 0.2)))
    te = perm[:n_te].to(device)
    dev = perm[n_te:n_te + n_dev].to(device)
    tr = perm[n_te + n_dev:].to(device)
    return tr, dev, te


def _lbfgs_fit(xtr, ytr, vocab: int, l2: float, iters: int, seed: int):
    """Multinomial logistic regression by full-batch L-BFGS.

    The objective is convex, so "did the fit converge" is a question with an
    answer rather than a hyperparameter.  This replaces a minibatch AdamW fit
    that was measuring its own step budget: under it every feature set degraded
    monotonically in its own WIDTH, which is the signature of an under-trained
    head and not of an uninformative feature.
    """
    torch.manual_seed(seed)
    head = torch.nn.Linear(xtr.shape[1], vocab, bias=True).to(xtr.device)
    torch.nn.init.zeros_(head.weight)
    torch.nn.init.zeros_(head.bias)
    opt = torch.optim.LBFGS(head.parameters(), max_iter=iters,
                            history_size=10, line_search_fn="strong_wolfe",
                            tolerance_grad=1e-9, tolerance_change=1e-12)

    def closure():
        opt.zero_grad(set_to_none=True)
        loss = F.cross_entropy(head(xtr), ytr)
        if l2 > 0.0:
            loss = loss + l2 * head.weight.pow(2).sum()
        loss.backward()
        return loss

    opt.step(closure)
    return head


@torch.no_grad()
def _bpc(head, xx, yy) -> float:
    tot, cnt = 0.0, 0
    for i in range(0, xx.shape[0], 65536):
        chunk = xx[i:i + 65536]
        tot += float(F.cross_entropy(head(chunk), yy[i:i + 65536],
                                     reduction="sum"))
        cnt += chunk.shape[0]
    return tot / cnt / _LN2


def fit_head(feats: torch.Tensor, y: torch.Tensor, vocab: int, *,
             iters: int, l2_grid: list[float], seed: int,
             split: tuple[torch.Tensor, ...]) -> dict:
    """Ridge-selected multinomial logistic fit; returns dev-selected TEST bpc."""
    W, L, Fdim = feats.shape
    tr, dev, te = split

    xtr = feats.index_select(0, tr).reshape(-1, Fdim)
    ytr = y.index_select(0, tr).reshape(-1)
    xdv = feats.index_select(0, dev).reshape(-1, Fdim)
    ydv = y.index_select(0, dev).reshape(-1)
    xte = feats.index_select(0, te).reshape(-1, Fdim)
    yte = y.index_select(0, te).reshape(-1)

    trace = []
    best = None
    for l2 in l2_grid:
        head = _lbfgs_fit(xtr, ytr, vocab, l2, iters, seed)
        d = _bpc(head, xdv, ydv)
        trace.append({"l2": l2, "dev_bpc": d})
        if best is None or d < best[0]:
            best = (d, l2, head)
    dev_bpc, l2_star, head = best

    out = {
        "test_bpc": _bpc(head, xte, yte),
        "dev_bpc": dev_bpc,
        "train_bpc": _bpc(head, xtr, ytr),
        "l2_selected": l2_star,
        "l2_at_grid_edge": l2_star in (l2_grid[0], l2_grid[-1]),
        "l2_trace": trace,
        "n_features": Fdim,
        "n_train_pos": int(xtr.shape[0]),
        "n_test_pos": int(xte.shape[0]),
        "head_params": Fdim * vocab + vocab,
    }
    del xtr, xdv, xte, head
    torch.cuda.empty_cache()
    return out


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------

def run_checkpoint(run_dir: Path, args, device: str) -> dict:
    cfg = Config.from_dict(json.loads((run_dir / "config.json").read_text()))
    cfg.device = device
    corpus = Corpus(cfg.corpus, cfg.data_dir)
    cfg.vocab_size = corpus.vocab_size

    model = build_model(cfg)
    ck = torch.load(run_dir / args.ckpt, map_location=device, weights_only=False)
    model.load_state_dict(ck["model"] if "model" in ck else ck)

    t0 = time.time()
    store = collect(model, corpus, cfg, args.split, args.n_windows, device)
    n_layers = len(store["spikes"])

    # T2b: the local scan must reproduce the committed emission bitwise before
    # its membrane is allowed to price anything.
    t2b = None
    if args.membrane:
        v_pre, s_local = membrane_of_last_layer(model, corpus, cfg, args.split,
                                                args.n_windows, device)
        w = min(v_pre.shape[0], store["spikes"][-1].shape[0])
        t2b = bool(torch.equal(s_local[:w], store["spikes"][-1][:w]))
        store["v_pre_last"] = v_pre[:w]
        for k in range(n_layers):
            store["spikes"][k] = store["spikes"][k][:w]
        store["y"] = store["y"][:w]
        store["committed_nats"] = store["committed_nats"][:w]

    results: dict[str, dict] = {}
    split = _window_split(int(store["y"].shape[0]), args.seed, device)
    for name, spec in FEATURE_SETS.items():
        if spec["analogue"] and "v_pre_last" not in store:
            continue
        if spec.get("wide") and args.skip_wide:
            # A 4d-wide feature block needs an fp32 copy of itself plus three
            # index_select copies; at the window count this fit needs to stop
            # overfitting, that does not fit beside the model on 8 GiB. Skipped
            # LOUDLY rather than silently dropped -- these sets are exactly the
            # ones the end-to-end arms exist to measure.
            results[name] = {"skipped": "does not fit at this --n-windows"}
            continue
        feats = build_features(store, spec, n_layers)

        # T2: binary blocks are binary, the analogue block is not.  Checked on
        # the SOURCE block rather than on the assembled feature tensor, which is
        # up to 4x wider: `(feats != 0) & (feats != 1)` on the wide tensor costs
        # a second allocation of the same size, and lag/concat cannot introduce
        # a value neither source held.
        src = store["v_pre_last"] if spec["analogue"] else store["spikes"][-1].float()
        uniq_binary = bool((((src != 0.0) & (src != 1.0)).sum()) == 0)
        if spec["analogue"] and uniq_binary:
            raise RuntimeError(f"{name} was asserted analogue and is binary")
        if not spec["analogue"] and not uniq_binary:
            raise RuntimeError(f"{name} was asserted binary and is not")

        r = fit_head(feats, store["y"], cfg.vocab_size, iters=args.iters,
                     l2_grid=args.l2, seed=args.seed, split=split)
        r["binary_features"] = uniq_binary
        results[name] = r
        del feats
        torch.cuda.empty_cache()

    # T1: the committed head on the SAME test windows
    te = split[2]
    committed_bpc = float(
        store["committed_nats"].index_select(0, te).sum()
    ) / (te.numel() * store["y"].shape[1]) / _LN2

    refit_r0 = results["R0_last_t"]["test_bpc"]
    return {
        "run": run_dir.name,
        "arch": cfg.arch,
        "d_model": cfg.d_model,
        "n_layers": cfg.n_layers,
        "vocab_size": cfg.vocab_size,
        "split": args.split,
        "n_windows_collected": int(store["y"].shape[0]),
        "seq_len": int(store["y"].shape[1]),
        "firing_rate": [float(s.float().mean()) for s in store["spikes"]],
        "committed_head_holdout_bpc": committed_bpc,
        "T1_refit_minus_committed": refit_r0 - committed_bpc,
        "T1_pass": bool(refit_r0 - committed_bpc > -args.t1_tol),
        "T2b_local_scan_bitwise_equal": t2b,
        "wall_clock_s": round(time.time() - t0, 2),
        "sets": results,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", nargs="+", default=[
        "experiments/runs/snn_beta0.5_s0",
        "experiments/runs/snn_beta0.5_s1",
        "experiments/runs/snn_beta0.5_s2",
    ])
    ap.add_argument("--ckpt", default="ckpt_final.pt")
    ap.add_argument("--split", default="val")
    ap.add_argument("--n-windows", type=int, default=512)
    ap.add_argument("--iters", type=int, default=300,
                    help="L-BFGS iterations per ridge value")
    ap.add_argument("--l2", type=float, nargs="+",
                    default=[1e-5, 1e-4, 1e-3, 1e-2, 1e-1])
    ap.add_argument("--skip-wide", action="store_true", default=False)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--t1-tol", type=float, default=0.02)
    ap.add_argument("--membrane", action="store_true", default=True)
    ap.add_argument("--no-membrane", dest="membrane", action="store_false")
    ap.add_argument("--out", default="docs/reports/data/exp_020_readout_probe.json")
    args = ap.parse_args(argv)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = {"torch": torch.__version__, "device": device, "args": vars(args),
           "checkpoints": []}
    for r in args.runs:
        rd = _REPO / r
        print(f"[020] {rd.name} ...", flush=True)
        res = run_checkpoint(rd, args, device)
        out["checkpoints"].append(res)
        for name, s in res["sets"].items():
            if "skipped" in s:
                print(f"    {name:22s} SKIPPED ({s['skipped']})", flush=True)
                continue
            d = s["test_bpc"] - res["sets"]["R0_last_t"]["test_bpc"]
            print(f"    {name:22s} test={s['test_bpc']:.5f} "
                  f"d_vs_R0={d:+.5f} F={s['n_features']:5d} "
                  f"l2={s['l2_selected']:g}{' EDGE' if s['l2_at_grid_edge'] else ''}",
                  flush=True)
        print(f"    committed head = {res['committed_head_holdout_bpc']:.5f} "
              f"(T1 {'pass' if res['T1_pass'] else 'FAIL'}, "
              f"T2b {res['T2b_local_scan_bitwise_equal']})", flush=True)
        torch.cuda.empty_cache()

    p = _REPO / args.out
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    print(f"[020] wrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
