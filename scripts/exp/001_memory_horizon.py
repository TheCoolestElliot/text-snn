"""EXP_001 -- how far back does each arm's recurrent state actually reach?

Pre-registered in experiments/logs/EXP_001_memory_horizon.md. Read that first;
the decision rule there was fixed before this file existed.

The probe
---------
Every arm carries exactly one thing across time: per-layer membrane `v` for the
spiking and analogue arms, hidden `h` for the GRU. All three zero it when
`forward(idx, state=None)` is called. So "force the state to zero every k
characters" needs no hook, no model edit, and no new code path in the model --
it is just

    x [B, L]  ->  x.reshape(B * L // k, k)

run with `state=None`. Each row of the reshaped tensor is a fresh sequence, so
the neuron's state is zero at its start and evolves for exactly k steps. The
same (context, target) pairs are scored at every k; only the amount of context
available to predict each one changes.

Two properties of that reshape matter and are why it was chosen over a
state-zeroing hook:

  * it is element-count preserving. [128, 256, d] and [32768, 1, d] hold the
    same number of floats, so peak VRAM is flat across the whole sweep and the
    k = 1 point is no more expensive to hold than the k = 256 one;
  * the probe cannot measure its own implementation. A hook that zeroes state
    mid-scan would be new code between the model and the number, and a bug in it
    would look exactly like a short memory horizon. Here the only thing under
    test is the model's own `state=None` branch, which Phase 2 already exercises
    on every `fresh` evaluation.

Small k is *faster*, not slower: the time loop runs k steps rather than L, and
this box is launch-bound (01_reconnaissance.md 3.4). The full sweep costs
roughly twice a single fresh evaluation per checkpoint.

F1, the self-check that makes the rest trustworthy: at k = L the reshape is the
identity and this file must reproduce the committed `fresh` bpc from
docs/reports/data/phase2_final_scores.json. It is asserted, not eyeballed.
"""

from __future__ import annotations

import argparse
import dataclasses
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

from snn.config import Config, config_hash, seed_everything  # noqa: E402
from snn.data import Corpus, windowed_eval_batches  # noqa: E402
from snn.metrics import bits_per_char  # noqa: E402
from snn.model import build_model, count_params  # noqa: E402

# Pre-registered sweep (EXP_001 3). Every value divides L = 256, so no window is
# ragged and the character count is identical at every k -- failure mode F2.
K_VALUES = (1, 2, 4, 8, 16, 32, 64, 128, 256)

# Pre-registered arm list. Seeds are the ones the Phase-2 campaign produced.
ARMS: dict[str, str] = {
    "snn_beta0.5_s0": "snn_beta0.5",
    "snn_beta0.5_s1": "snn_beta0.5",
    "snn_beta0.5_s2": "snn_beta0.5",
    "snn_beta0.5_s3": "snn_beta0.5",
    "snn_beta0.5_s4": "snn_beta0.5",
    "gru_s0": "gru",
    "gru_s1": "gru",
    "gru_s2": "gru",
    "analogue_beta0.5_s0": "analogue_beta0.5",
    "snn_beta0.9_s0": "snn_beta0.9",
    "snn_beta0.95_s0": "snn_beta0.95",
    # EXP_003's parameter-matched depth ladder. Same probe, same statistic --
    # reusing this file rather than writing a second one is deliberate: the
    # horizon numbers it produces have to be comparable with the table above,
    # and two implementations of one statistic is how they stop being.
    "depth_K1_s0": "depth_K1",
    "depth_K4_s0": "depth_K4",
    "depth_K8_s0": "depth_K8",
    # EXP_004's two-compartment arm. Same probe, same paired statistic, same
    # script -- for the same reason the depth ladder reuses it: the horizon
    # numbers have to be comparable with the rows above, and two implementations
    # of one statistic is how they stop being. Phase 4 runs this with an explicit
    # --out so the committed Phase-3 artifact is never overwritten.
    "twocomp_s0": "twocomp",
    "twocomp_s1": "twocomp",
    "twocomp_s2": "twocomp",
    # EXP_005's four extra seeds of the SAME arm (candidate #14, promoted to
    # blocking by EXP_004 4). This is a registry entry and nothing else: the
    # statistic, the k-sweep and the F1 tolerance are untouched, which is
    # EXP_005's failure mode K5. The probe stays the instrument that produced
    # every horizon number in the project, and EXP_005 runs it with an explicit
    # --out so no committed artifact is overwritten.
    "twocomp_s3": "twocomp",
    "twocomp_s4": "twocomp",
    "twocomp_s5": "twocomp",
    "twocomp_s6": "twocomp",
}

# F1: the k = L point must reproduce the committed Phase-2 `fresh` number.
F1_TOLERANCE_BPC = 1e-3


def _config_from_checkpoint(ck: dict) -> Config:
    raw = ck.get("config")
    if not raw:
        raise SystemExit("checkpoint has no 'config' block; cannot rebuild the model")
    known = {f.name for f in dataclasses.fields(Config)}
    return Config(**{k: v for k, v in raw.items() if k in known})


@torch.no_grad()
def score_at_horizon(model, data, batch_size: int, seq_len: int, k: int,
                     device: torch.device, max_windows: int = 0) -> dict:
    """Full-split score with recurrent state forced to zero every `k` characters.

    `k == seq_len` is the ordinary `fresh` protocol, reached by the identity
    reshape rather than by a separate code path -- which is what makes it a
    self-check rather than a second implementation.
    """
    if seq_len % k:
        raise ValueError(f"k={k} must divide seq_len={seq_len}")

    was_training = model.training
    model.eval()

    # Per-position accumulator, indexed by position in the ORIGINAL 256-character
    # window, not by position within a chunk. Keeping the original indexing is
    # what makes the k-curves position-comparable, which is what the paired
    # analysis below needs.
    nats_at = torch.zeros(seq_len, dtype=torch.float64, device=device)
    n_windows = 0
    rate_sums: list[float] = []
    rate_weight = 0.0

    try:
        for x, y in windowed_eval_batches(data, batch_size, seq_len):
            b = int(x.shape[0])
            if max_windows and n_windows + b > max_windows and n_windows > 0:
                break
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            # The whole probe. Contiguous first: a [B, L] window tensor is
            # already contiguous, but a non-contiguous view would reshape into
            # the wrong character order and produce a plausible wrong number.
            xk = x.contiguous().view(-1, k)
            yk = y.contiguous().view(-1, k)

            logits, _state, aux = model(xk, None)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]).float(),
                yk.reshape(-1),
                reduction="none",
            ).view(b, seq_len)          # back to original window indexing
            nats_at += loss.sum(dim=0).double()
            n_windows += b

            rates = aux.get("firing_rate", ()) if isinstance(aux, dict) else ()
            if rates:
                if not rate_sums:
                    rate_sums = [0.0] * len(rates)
                for i, r in enumerate(rates):
                    rate_sums[i] += float(r) * b
                rate_weight += b
    finally:
        model.train(was_training)

    total_nats = float(nats_at.sum())
    n_chars = n_windows * seq_len
    return {
        "k": k,
        "bpc": bits_per_char(total_nats, n_chars) if n_chars else float("nan"),
        "nats": total_nats,
        "n_chars": n_chars,
        "n_windows": n_windows,
        "nats_at_position": [float(v) for v in nats_at],
        "firing_rate": [s / rate_weight for s in rate_sums] if rate_weight else [],
    }


def paired_context_curve(curve: dict, seq_len: int, min_ref_context: int = 128) -> dict:
    """Excess bpc from having only `c` characters of context. Noise-cancelled.

    POST-HOC ADDITION, made after the pre-registered sweep's first smoke run and
    labelled as such in EXP_001 8.1. It does not replace the pre-registered
    statistic, which is reported in full beside it.

    Why it was added
    ----------------
    `score_at_horizon` averages over every position inside a chunk, so a fraction
    1/k of all scored characters sit at position 0 with no context at all. If the
    horizon is *short* but the first few positions are expensive, the sweep still
    traces `bpc(k) = bpc_inf + E/k` -- a smooth curve improving all the way out to
    k = L that looks like long memory while being produced entirely by boundary
    amortisation. The smoke run's curve fits that form to a few percent, so the
    pre-registered statistic cannot on its own distinguish the two.

    Why it is paired
    ----------------
    The obvious fix -- mean loss at each position j of an untruncated window -- is
    far too noisy to test against a 0.0092 bpc threshold, because position j
    samples different *text* in every window and the per-position standard error
    is around 0.02 bpc. Pairing removes that: for a fixed position p, the model,
    the window and the target character are all identical between the truncated
    and untruncated runs, and only the available context differs. The text
    difficulty that dominates the unpaired variance cancels in the difference.

    Construction. Under truncation k, position p of the original window has
    exactly `p mod k` characters of context; in the k = L reference it has `p`.
    Restricting to `p >= min_ref_context` keeps the reference well-contexted, so

        X = bpc_k[p] - bpc_L[p]     grouped by  c = p mod k

    reads directly as "the cost of having only c characters of context instead of
    at least `min_ref_context`". It goes to zero exactly where extra context stops
    being worth anything -- which is the horizon.
    """
    ln2 = math.log(2.0)
    ref = curve[str(seq_len)]
    nw_ref = ref["n_windows"]
    bpc_ref = [v / nw_ref / ln2 for v in ref["nats_at_position"]]

    buckets: dict[int, list[float]] = {}
    for k_str, row in curve.items():
        k = int(k_str)
        if k == seq_len:
            continue
        nw = row["n_windows"]
        bpc_k = [v / nw / ln2 for v in row["nats_at_position"]]
        for p in range(min_ref_context, seq_len):
            buckets.setdefault(p % k, []).append(bpc_k[p] - bpc_ref[p])

    excess = {str(c): sum(v) / len(v) for c, v in sorted(buckets.items())}
    n_samples = {str(c): len(v) for c, v in sorted(buckets.items())}

    # The single cleanest slice, quoted on its own because it needs no pooling:
    # k = 128, positions 128..255, so context c = p - 128 against a reference
    # context of 128 + c. One k, one contiguous position range, no mixing.
    k128 = curve.get("128")
    slice128 = None
    if k128 is not None:
        b128 = [v / k128["n_windows"] / ln2 for v in k128["nats_at_position"]]
        slice128 = [b128[128 + c] - bpc_ref[128 + c] for c in range(128)]

    return {
        "min_ref_context": min_ref_context,
        "excess_bpc_at_context": excess,
        "n_samples_at_context": n_samples,
        "k128_slice_excess_bpc": slice128,
        "bpc_at_position_untruncated": bpc_ref,
    }


def horizon_from_excess(excess: dict[str, float], tol: float) -> int | None:
    """Smallest c from which every larger context is within `tol` of the reference.

    Scanned from the top down so a single noisy low-c bucket cannot declare an
    early horizon: the answer is the point past which the curve *stays* flat.
    """
    keys = sorted((int(c) for c in excess), reverse=True)
    horizon = None
    for c in keys:
        if abs(excess[str(c)]) <= tol:
            horizon = c
        else:
            break
    return horizon


def _fresh_bpc_from_block(block) -> float | None:
    if not isinstance(block, dict):
        return None
    results = block.get("results", block)
    fresh = results.get("fresh") if isinstance(results, dict) else None
    return fresh.get("bpc") if isinstance(fresh, dict) else None


def committed_fresh_bpc(run: str, split: str) -> float | None:
    """The independently-produced number this probe's k = L point must reproduce.

    F1 is the check that makes every other number in this file trustworthy, so it
    must not quietly go missing for the runs that are newest and least
    established. Two sources, in order:

      1. docs/reports/data/phase2_final_scores.json, for the Phase-2 campaign;
      2. experiments/runs/<run>/final_<split>.json, written by scripts/evaluate.py
         for anything added later (EXP_003's depth ladder).

    Both are produced by `snn.evaluate.evaluate` through the ordinary `fresh`
    protocol -- a different code path from this file's reshape -- which is what
    makes the agreement evidence rather than a tautology.
    """
    path = _REPO / "docs" / "reports" / "data" / "phase2_final_scores.json"
    if path.exists():
        blob = json.loads(path.read_text(encoding="utf-8"))
        entry = blob.get(run)
        if isinstance(entry, dict):
            bpc = _fresh_bpc_from_block(entry.get(split))
            if bpc is not None:
                return bpc

    local = _REPO / "experiments" / "runs" / run / f"final_{split}.json"
    if local.exists():
        return _fresh_bpc_from_block(json.loads(local.read_text(encoding="utf-8")))
    return None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", default="test", choices=["val", "test"])
    ap.add_argument("--max-windows", type=int, default=0, help="0 = whole split")
    ap.add_argument("--runs", default="", help="comma-separated subset of run names")
    ap.add_argument("--out", default="docs/reports/data/exp_001_memory_horizon.json")
    args = ap.parse_args(argv)

    wanted = [r.strip() for r in args.runs.split(",") if r.strip()] or list(ARMS)
    runs_dir = _REPO / "experiments" / "runs"

    results: dict = {
        "experiment": "EXP_001_memory_horizon",
        "split": args.split,
        "k_values": list(K_VALUES),
        "max_windows": args.max_windows,
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "f1_tolerance_bpc": F1_TOLERANCE_BPC,
        "arms": {},
    }
    f1_failures: list[str] = []

    for run in wanted:
        ckpt_path = runs_dir / run / "ckpt_final.pt"
        if not ckpt_path.exists():
            print(f"  {run:24s} MISSING checkpoint, skipped")
            continue

        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        cfg = _config_from_checkpoint(ck)
        seed_everything(cfg.seed, cfg.deterministic)
        device = torch.device(cfg.device)
        corpus = Corpus(cfg.corpus, cfg.data_dir)
        model = build_model(cfg).to(device)
        model.load_state_dict(ck["model"])
        model.eval()
        data = corpus.split(args.split)

        entry = {
            "arm": ARMS.get(run, "other"),
            "seed": cfg.seed,
            "arch": cfg.arch,
            "beta": cfg.beta,
            "reset": cfg.reset,
            "params": count_params(model),
            "checkpoint_step": ck.get("step"),
            "checkpoint_config_hash": ck.get("config_hash"),
            "eval_config_hash": config_hash(cfg),
            "seq_len": cfg.seq_len,
            "batch_size": cfg.batch_size,
            "curve": {},
        }
        print(f"  {run:24s} arch={cfg.arch} beta={cfg.beta} params={entry['params']:,}")

        for k in K_VALUES:
            t0 = time.perf_counter()
            row = score_at_horizon(model, data, cfg.batch_size, cfg.seq_len, k,
                                   device, max_windows=args.max_windows)
            row["wall_clock_s"] = round(time.perf_counter() - t0, 2)
            entry["curve"][str(k)] = row
            print(f"      k={k:>4d}  bpc {row['bpc']:.5f}  "
                  f"chars {row['n_chars']:,}  {row['wall_clock_s']:.1f}s")

        # ---- F2: identical characters scored at every k --------------------
        counts = {r["n_chars"] for r in entry["curve"].values()}
        entry["f2_char_counts_identical"] = len(counts) == 1
        if len(counts) != 1:
            print(f"      F2 FAILED: character counts differ across k: {counts}")

        # ---- F1: the k = L point reproduces the committed Phase-2 number ----
        ref = committed_fresh_bpc(run, args.split) if not args.max_windows else None
        got = entry["curve"][str(cfg.seq_len)]["bpc"]
        if ref is not None:
            residual = abs(got - ref)
            entry["f1_reference_fresh_bpc"] = ref
            entry["f1_residual_bpc"] = residual
            entry["f1_pass"] = residual < F1_TOLERANCE_BPC
            flag = "OK" if entry["f1_pass"] else "FAILED"
            print(f"      F1 {flag}: k={cfg.seq_len} {got:.6f} vs committed "
                  f"{ref:.6f}  (residual {residual:.2e})")
            if not entry["f1_pass"]:
                f1_failures.append(run)
        else:
            entry["f1_pass"] = None
            print("      F1 skipped (no committed reference for this run/split)")

        # ---- the quantities the pre-registered predictions are about --------
        c = entry["curve"]
        entry["degradation_vs_untruncated"] = {
            str(k): c[str(k)]["bpc"] - c[str(cfg.seq_len)]["bpc"] for k in K_VALUES
        }

        # ---- the paired, noise-cancelled horizon curve (post-hoc) ----------
        pc = paired_context_curve(c, cfg.seq_len)
        # 0.00922 bpc is 2 sigma of the Phase-2 noise floor (EXP_000) -- the same
        # bar every other adoption decision in this project is held to.
        pc["horizon_2sigma"] = horizon_from_excess(pc["excess_bpc_at_context"], 0.00922)
        pc["horizon_0.05bpc"] = horizon_from_excess(pc["excess_bpc_at_context"], 0.05)
        entry["paired_context"] = pc
        ex = pc["excess_bpc_at_context"]
        shown = " ".join(f"c={j}:{ex[str(j)]:+.4f}" for j in (0, 1, 2, 4, 8, 16, 32)
                         if str(j) in ex)
        print(f"      paired excess  {shown}")
        print(f"      horizon: {pc['horizon_2sigma']} chars (2 sigma), "
              f"{pc['horizon_0.05bpc']} chars (0.05 bpc)")

        # ---- reconciliation: does the pre-registered curve carry any signal
        #      the paired curve does not? (EXP_001 8.1)
        #
        # Chunking a 256-character window at size k creates 256/k chunk starts,
        # of which 256/k - 1 are *new* relative to the untruncated run (position
        # 0 is a chunk start either way). Each new boundary costs the model the
        # excess of its first k positions. So if the horizon is short and the
        # entire truncation penalty is boundary amortisation,
        #
        #   bpc(k) - bpc(L)  =  (L/k - 1)/L  *  sum_{c<k} excess(c)
        #
        # exactly, with no free parameter. Agreement means the pre-registered
        # statistic is a deterministic function of the paired one and adds no
        # independent evidence about memory -- which is the point being tested.
        seq_len = cfg.seq_len
        inf_bpc = c[str(seq_len)]["bpc"]
        recon = {}
        for k in K_VALUES:
            if k == seq_len:
                continue
            tail = sum(ex[str(j)] for j in range(k) if str(j) in ex)
            predicted = (seq_len / k - 1) / seq_len * tail
            measured = c[str(k)]["bpc"] - inf_bpc
            recon[str(k)] = {
                "measured_excess_bpc": measured,
                "predicted_from_paired_bpc": predicted,
                "residual_bpc": measured - predicted,
            }
        entry["boundary_amortisation_reconciliation"] = recon
        worst = max(abs(r["residual_bpc"]) for r in recon.values())
        entry["reconciliation_max_abs_residual_bpc"] = worst
        print(f"      boundary-amortisation reconciliation: "
              f"max |residual| = {worst:.4f} bpc")

        results["arms"][run] = entry

        del model
        torch.cuda.empty_cache()

    # ---- per-arm aggregation across seeds ---------------------------------
    by_arm: dict[str, list[dict]] = {}
    for run, entry in results["arms"].items():
        by_arm.setdefault(entry["arm"], []).append(entry)

    agg: dict = {}
    for arm, entries in sorted(by_arm.items()):
        rows = {}
        for k in K_VALUES:
            vals = [e["curve"][str(k)]["bpc"] for e in entries]
            degs = [e["degradation_vs_untruncated"][str(k)] for e in entries]
            n = len(vals)
            mean = sum(vals) / n
            sd = (math.sqrt(sum((v - mean) ** 2 for v in vals) / (n - 1))
                  if n > 1 else 0.0)
            rows[str(k)] = {
                "n": n,
                "bpc_mean": mean,
                "bpc_sd": sd,
                "bpc_values": vals,
                "degradation_mean": sum(degs) / n,
            }
        contexts = sorted(int(x) for x in entries[0]["paired_context"]
                          ["excess_bpc_at_context"])
        rows["_paired"] = {
            "n": len(entries),
            "excess_bpc_at_context_mean": {
                str(c): sum(e["paired_context"]["excess_bpc_at_context"][str(c)]
                            for e in entries) / len(entries)
                for c in contexts
            },
            "horizon_2sigma_per_seed": [e["paired_context"]["horizon_2sigma"]
                                        for e in entries],
            "horizon_0.05bpc_per_seed": [e["paired_context"]["horizon_0.05bpc"]
                                         for e in entries],
            "reconciliation_max_abs_residual_bpc": [
                e["reconciliation_max_abs_residual_bpc"] for e in entries],
        }
        agg[arm] = rows
    results["by_arm"] = agg
    results["f1_failures"] = f1_failures

    # ---- absolute bpc at each context length, and the dominance test -------
    #
    # The paired excess is measured against each arm's OWN asymptote, which makes
    # it the right statistic for locating a horizon and the WRONG one for
    # comparing arms: the GRU's excess is larger than the baseline's at every c
    # simply because it has more to lose. Adding the asymptote back gives the
    # absolute bpc an arm achieves with exactly c characters of context, which is
    # comparable across arms.
    #
    # That distinction is what separates the two ways of buying horizon. Raising
    # beta lengthens the horizon while pushing the absolute curve UP from c = 2
    # onward -- it degrades every context length in exchange for reaching
    # further. The GRU's curve lies at or below the baseline's everywhere. So the
    # test a Phase-4 candidate has to pass is not "is the horizon longer" but
    # "is the absolute curve nowhere worse", and that is computed here rather
    # than asserted in prose.
    baseline_arm = "snn_beta0.5"
    if baseline_arm not in agg:
        # A partial run (--runs) that excludes the baseline cannot compute the
        # cross-arm comparison, and silently comparing against whatever arm
        # happened to be first would be worse than not comparing at all.
        out = _REPO / args.out
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"\n  (baseline arm {baseline_arm!r} not in this selection; "
              f"absolute-curve comparison skipped)")
        print(f"WROTE {args.out}")
        return 1 if f1_failures else 0

    seq_len_ref = next(iter(results["arms"].values()))["seq_len"]
    contexts = sorted(int(x) for x in
                      agg[baseline_arm]["_paired"]["excess_bpc_at_context_mean"])

    def absolute_curve(arm: str) -> dict[str, float]:
        r = agg[arm]
        asym = r[str(seq_len_ref)]["bpc_mean"]
        ex = r["_paired"]["excess_bpc_at_context_mean"]
        return {str(c): asym + ex[str(c)] for c in contexts}

    base_curve = absolute_curve(baseline_arm)

    # ---- the PER-CONTEXT noise floor -------------------------------------
    #
    # EXP_000's sigma = 0.00461 bpc is the standard deviation of the WHOLE-SPLIT
    # score. It is the wrong bar for a per-context comparison and wrong in both
    # directions: measured over the five baseline seeds, the seed spread of the
    # excess at c = 1 is 6x the whole-split sigma (short contexts have the fewest
    # position-samples), while at c >= 16 it is 30x SMALLER. Judging a curve
    # against a flat 0.00922 would reject candidates on noise at exactly the
    # short contexts the criterion exists to protect, and wave through real
    # long-context regressions.
    #
    # So the bar is measured, per context, from the same five seeds that produced
    # the noise floor in the first place.
    base_runs = [r for r in results["arms"].values() if r["arm"] == baseline_arm]
    per_context_2sigma: dict[str, float] = {}
    if len(base_runs) > 1:
        for c in contexts:
            vals = [r["paired_context"]["excess_bpc_at_context"][str(c)]
                    for r in base_runs]
            m = sum(vals) / len(vals)
            sd = math.sqrt(sum((v - m) ** 2 for v in vals) / (len(vals) - 1))
            per_context_2sigma[str(c)] = 2 * sd
    results["per_context_2sigma"] = {
        "n_seeds": len(base_runs),
        "arm": baseline_arm,
        "bars": per_context_2sigma,
        "whole_split_2sigma_for_reference": 0.00922,
    }

    dominance: dict = {}
    for arm in agg:
        curve = absolute_curve(arm)
        deltas = {c: curve[c] - base_curve[c] for c in curve}
        # The worst point is the largest ABSOLUTE regression, not the largest in
        # units of the local noise bar. Normalising by the bar sounds more
        # principled and is not: the bar at c >= 16 is ~0.0003 bpc, so any arm
        # with a worse asymptote scores thousands of "sigmas" out at c = 125 and
        # every arm's worst point collapses onto "its asymptote is worse" --
        # which is just bpc again, and defeats the whole point of looking at the
        # curve. The bars are used for the significance BOOLEAN, where they
        # belong, and the reported worst point stays in bits per character.
        worst_c = max(deltas, key=lambda c: deltas[c])
        # Short-context regression: the failure mode the criterion exists for --
        # buying reach by degrading the characters the model already sees.
        short = [c for c in deltas if int(c) <= 8]
        worst_short = max(short, key=lambda c: deltas[c])
        significant = [
            int(c) for c in deltas
            if deltas[c] > per_context_2sigma.get(c, 0.00922)
        ]
        dominance[arm] = {
            "bpc_at_context": curve,
            "delta_vs_baseline": deltas,
            "worst_delta_bpc": deltas[worst_c],
            "worst_delta_at_context": int(worst_c),
            "worst_short_context_delta_bpc": deltas[worst_short],
            "worst_short_context_at": int(worst_short),
            "n_contexts_significantly_worse": len(significant),
            "nowhere_worse_than_noise": not significant,
        }
    results["absolute_context_curves"] = dominance

    # ---- where the I5 gap actually lives ----------------------------------
    #
    # Splitting the SNN-vs-GRU gap by context length says how much of it any
    # horizon candidate could possibly recover. The part present at ZERO context
    # cannot be a memory effect at all, and the part inside the SNN's own 7
    # characters is context it already reaches and extracts less from. Only the
    # remainder is addressable by making memory longer -- which is the
    # denominator every candidate's ROI should be quoted against.
    if "gru" in agg:
        s, g = absolute_curve(baseline_arm), absolute_curve("gru")
        cmax = str(max(contexts))
        horizon = agg[baseline_arm]["_paired"]["horizon_2sigma_per_seed"][0] or 7
        h = str(horizon)
        total = s[cmax] - g[cmax]
        zero = s["0"] - g["0"]
        within = (s[h] - g[h]) - zero
        results["i5_gap_decomposition"] = {
            "baseline_horizon": horizon,
            "asymptotic_gap_bpc": total,
            "zero_context_bpc": zero,
            "within_reach_bpc": within,
            "beyond_horizon_bpc": total - (s[h] - g[h]),
            "share_not_addressable_by_horizon": (zero + within) / total,
        }
        dec = results["i5_gap_decomposition"]
        print("\n" + "=" * 78)
        print("WHERE THE I5 GAP LIVES (SNN baseline vs GRU anchor)")
        print("=" * 78)
        print(f"  asymptotic gap          {dec['asymptotic_gap_bpc']:.4f} bpc")
        print(f"  at zero context         {dec['zero_context_bpc']:.4f}  "
              f"({100*dec['zero_context_bpc']/total:.0f}%)  -- memory cannot explain this")
        print(f"  within the 7-char reach {dec['within_reach_bpc']:.4f}  "
              f"({100*dec['within_reach_bpc']/total:.0f}%)  -- context it already has")
        print(f"  beyond the horizon      {dec['beyond_horizon_bpc']:.4f}  "
              f"({100*dec['beyond_horizon_bpc']/total:.0f}%)  -- the horizon candidates' ceiling")
        print(f"  NOT addressable by horizon: "
              f"{100*dec['share_not_addressable_by_horizon']:.0f}%")

    out = _REPO / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print("\n" + "=" * 78)
    print("MEMORY HORIZON -- test bpc vs truncation length k (mean over seeds)")
    print("=" * 78)
    header = "arm".ljust(20) + "".join(f"{k:>9d}" for k in K_VALUES)
    print(header)
    for arm, rows in agg.items():
        line = arm.ljust(20) + "".join(f"{rows[str(k)]['bpc_mean']:>9.4f}"
                                       for k in K_VALUES)
        print(line)
    print()
    for arm, rows in agg.items():
        d8 = rows["8"]["degradation_mean"]
        print(f"  {arm:20s} bpc(k=8) - bpc(k=256) = {d8:+.4f}")

    print("\n" + "=" * 78)
    print("PAIRED HORIZON -- excess bpc from having only c characters of context")
    print("=" * 78)
    probe_c = (0, 1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64)
    print("arm".ljust(20) + "".join(f"{c:>8d}" for c in probe_c) + "   horizon(2s)")
    for arm, rows in agg.items():
        p = rows["_paired"]
        e = p["excess_bpc_at_context_mean"]
        line = arm.ljust(20) + "".join(
            f"{e[str(c)]:>8.4f}" if str(c) in e else " " * 8 for c in probe_c)
        print(line + f"   {p['horizon_2sigma_per_seed']}")
    print()
    for arm, rows in agg.items():
        p = rows["_paired"]
        worst = max(p["reconciliation_max_abs_residual_bpc"])
        print(f"  {arm:20s} truncation curve reproduced from the paired curve "
              f"to max |residual| = {worst:.4f} bpc")

    print("\n" + "=" * 78)
    print("ABSOLUTE bpc at exactly c characters of context, vs the baseline")
    print("=" * 78)
    show_c = (0, 1, 2, 3, 4, 8, 16, 32, 64)
    bars = results["per_context_2sigma"]["bars"]
    if bars:
        print("2sigma bar".ljust(20) + "".join(
            f"{bars.get(str(c), float('nan')):>8.4f}" for c in show_c)
            + "   <- measured per context, not the flat whole-split 0.0092")
    print("arm".ljust(20) + "".join(f"{c:>8d}" for c in show_c)
          + "  worst short-c   n worse")
    for arm, dom in results["absolute_context_curves"].items():
        line = arm.ljust(20) + "".join(
            f"{dom['bpc_at_context'][str(c)]:>8.4f}" for c in show_c)
        print(line + f"   {dom['worst_short_context_delta_bpc']:>+8.4f}"
                     f" (c={dom['worst_short_context_at']:<3d})"
                     f" {dom['n_contexts_significantly_worse']:>4d}/128")
    if f1_failures:
        print(f"\n  F1 FAILURES (probe does not reproduce Phase 2): {f1_failures}")
    print(f"\nWROTE {args.out}")
    return 1 if f1_failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
