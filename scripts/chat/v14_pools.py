"""Draw the pools the v14 subject tier is scored on, and store ALL of each one.

WHAT THIS IS FOR
----------------
`RerankParams.subject_tier` is a selection rule, so its effect is a difference
between two selectors on ONE pool -- the design `echo_holdout.py` uses and for
the same reason: the model, the sampler, the seeds and the draws are held fixed
and the only thing that varies is the rule. This script draws the pool once,
asks the REAL selectors for both winners (`snnchat.rerank.rerank` under the
shipped parameters, then `snnchat.rerank.select` on the same candidates with
`subject_tier=True`), and writes down every candidate with everything either
rule read. `scripts/chat/score_v14.py` then scores the stored file with no GPU,
and refuses to proceed unless its own replay of both rules lands on the two
winners recorded here.

WHY IT IS NOT `echo_holdout.py --seed-offset`
---------------------------------------------
`echo_holdout.py` produced committed evidence and is not edited. Its stored
pools also cannot replay the subject rule: they carry no `len(c.scored)`, no
`closed`, no `subject_hit`, and `logp_null` only on the draws where the shipped
tier had more than one member -- the subject rule consults the score on exactly
the draws the shipped rule decided without it. Here `logp_null` is measured for
EVERY candidate of EVERY draw before the pool is written.

SAMPLER SEEDS 0-5 ARE SPENT
---------------------------
Every committed `v12_*` pool was drawn at sampler seeds `range(6)`, and those
pools were read while the subject rule was being designed. A draw at the same
seed from the same checkpoint is the same draw, so it confirms nothing. The
default is therefore `--seed-offset 6` (seeds 6..11), and an offset below 6 is
refused unless `--allow-committed-seeds` says the overlap is intended.

THE NON-STORY GUARD SET
-----------------------
`--set dodge` draws the twelve non-narrative probes `lambda_dodge.py` uses.
`snnchat.prime.tier_subject` returns None on every one of them
(`tests/test_snnchat_v14_scoring.py` holds that), and a turn with no subject
takes the old weighted tier whatever `subject_tier` says. So on this set the two
recorded winners must be the SAME index on every draw -- an identity, asserted
by the scorer from the stored pool, not a rate.

LATENCY
-------
The subject rule runs the null pass on draws where the shipped rule skipped it:
the shipped tier is often a single draft, which is decided without a score,
while a subject tier is the drafts that name the noun, or everyone. Each
selector is therefore timed per draw, alone, on a COPY of the candidates with
the null measurement cleared, so each pays for the null pass exactly when it
would in a turn of its own. The copies are discarded; the stored `logp_null` is
measured once.

Writes gzip-compressed JSON with a zeroed gzip timestamp, so one pool is one
byte string.

Not part of the research protocol; no number here is a reported figure.

    python scripts/chat/v14_pools.py --set heldout \\
        --ckpt "C:/.../experiments/chat/chat-v3d-aligned/ckpt_best.pt" \\
        --out experiments/chat/_quality/v14_heldout_chat-v3d-aligned.json.gz
"""
from __future__ import annotations

import argparse
import dataclasses
import gzip
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import torch  # noqa: E402

from snnchat.generate import SamplingParams, load_chat_checkpoint  # noqa: E402
from snnchat.prime import tier_subject  # noqa: E402
from snnchat.quality import PROBES  # noqa: E402
from snnchat.rerank import (  # noqa: E402
    RerankParams,
    _score_null,
    fill_null_scores,
    final_ids,
    prompt_content_words,
    rerank,
    select,
)
from snnchat.tokenizer import BOS, BOT, ChatTokenizer  # noqa: E402

sys.path.insert(0, str(ROOT / "scripts" / "chat"))
from echo_holdout import SETS as STORY_SETS  # noqa: E402
from lambda_dodge import DODGE_KINDS  # noqa: E402

#: Identifies the layout `score_v14.py` reads. Bump it if a field changes
#: meaning; the scorer refuses a format it does not know.
FORMAT = "v14_pools/1"

#: The sampler seeds every committed `v12_*` pool used: `range(6)`. A pool drawn
#: at an offset below this shares seeds with pools that were read before the
#: subject rule was written down.
COMMITTED_SEEDS = 6

#: The story sets are `echo_holdout.SETS`, imported rather than copied so the
#: prompts cannot drift from the lists every committed number was measured on.
SET_NAMES = (*sorted(STORY_SETS), "dodge")


def probes_for(name: str) -> list[tuple[str, tuple[str, ...], str]]:
    """`(prompt, expect, kind)` for a probe set.

    `dodge` is the same selection `lambda_dodge.py` makes -- the battery's
    `list` and `fact` probes -- named through its own `DODGE_KINDS` so the two
    cannot disagree about which prompts ask for no narrative.
    """
    if name == "dodge":
        return [(p.prompt, tuple(p.expect), p.kind) for p in PROBES
                if p.kind in DODGE_KINDS]
    return [(prompt, tuple(words), "story") for prompt, words in STORY_SETS[name]]


def sampler_seeds(seeds: int, offset: int, *, allow_committed: bool = False) -> list[int]:
    """`range(offset, offset + seeds)`, refused if it would redraw a read pool."""
    if seeds < 1:
        raise ValueError(f"seeds must be >= 1, got {seeds}")
    if offset < 0:
        raise ValueError(f"seed offset must be >= 0, got {offset}")
    if offset < COMMITTED_SEEDS and not allow_committed:
        raise ValueError(
            f"--seed-offset {offset} overlaps sampler seeds 0..{COMMITTED_SEEDS - 1}, "
            "which every committed v12 pool was drawn at and which were read "
            "before the subject rule existed; a pool drawn there confirms "
            "nothing. Pass --allow-committed-seeds if the overlap is the point."
        )
    return list(range(offset, offset + seeds))


def resolve(path: str) -> Path:
    """ROOT-relative unless absolute. The worktree has no `data/` and no `*.pt`,
    so a checkpoint is normally named by an absolute path into another tree."""
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def _sync(device) -> None:
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize()


def _index_of(cands, winner) -> int:
    for i, c in enumerate(cands):
        if c is winner:
            return i
    raise RuntimeError("the selector returned a candidate that is not in its pool")


def _time_select(model, cands, rp, *, device, tok, echo_words, subject) -> tuple[float, int]:
    """Seconds one `select` costs when it is the ONLY selector that runs.

    On copies with the null measurement cleared: `select` measures `logp_null`
    at most once per pool (`Candidate.null_context`), so timing the second rule
    on the candidates the first rule already scored would time a cache hit and
    report the subject rule as free. `ids` and `scored` are shared with the
    originals and are never written by `select`.
    """
    fresh = [dataclasses.replace(c, logp_null=0.0, null_context=None, null_scored=True)
             for c in cands]
    _sync(device)
    t0 = time.perf_counter()
    winner = select(model, fresh, rp, device=device, tok=tok, echo_words=echo_words,
                    subject=subject)
    _sync(device)
    return time.perf_counter() - t0, _index_of(fresh, winner)


@torch.no_grad()
def draw(model, tok, prompt: str, seed: int, rp: RerankParams, *, max_new: int,
         device, timing: bool = True) -> dict:
    """One (prompt, sampler seed): one pool, both winners, every candidate.

    The prefix and the call into `rerank` are `echo_holdout.py`'s, so a pool
    drawn here at a committed seed is the pool that script drew.
    """
    if rp.n < 2:
        raise ValueError("a pool of one has no selector to compare")
    params = SamplingParams(seed=seed, max_new=max_new)
    prefix = [BOS, *tok.render_turn("user", prompt), BOT]
    ids = torch.tensor([prefix], device=device)
    logits, state, _ = model(ids, state=None)

    # ONE pool. The shipped winner comes from `rerank` itself, called the way
    # `ChatSession` calls it with the default parameters: no subject.
    _sync(device)
    t0 = time.perf_counter()
    shipped, cands = rerank(model, logits[:, -1, :].float(), state, params, rp,
                            device=device, tok=tok,
                            echo_words=prompt_content_words(prompt))
    _sync(device)
    rerank_seconds = time.perf_counter() - t0
    return {"prompt": prompt, "seed": seed, "rerank_seconds": rerank_seconds,
            **record(model, tok, prompt, rp, cands, shipped, device=device,
                     timing=timing)}


@torch.no_grad()
def record(model, tok, prompt: str, rp: RerankParams, cands, shipped, *, device,
           timing: bool = True) -> dict:
    """Everything after the draw: the second winner, the null term, the pool.

    `shipped` is the candidate the SHIPPED rule returned for `cands`. Split from
    `draw` so that the part of this file a wrong number could hide in -- which
    index is recorded, which fields are stored, whether `logp_null` was really
    measured -- can be driven on CPU over hand-written candidates, where the
    subject rule has something to find; an untrained model drafts noise that
    names nothing.
    """
    if rp.subject_tier:
        raise ValueError("`rp` is the SHIPPED rule; the subject rule is derived from it")
    echo_words = prompt_content_words(prompt)
    subject = tier_subject(prompt)
    rp_subject = dataclasses.replace(rp, subject_tier=True)
    shipped_index = _index_of(cands, shipped)

    # The SAME candidates under the subject rule. This call rewrites `echo`,
    # `echo_weight` and `subject_hit`, so what is stored below is what the
    # subject rule saw; `echo_weight` does not depend on which rule asked.
    chosen = select(model, cands, rp_subject, device=device, tok=tok,
                    echo_words=echo_words, subject=subject)
    subject_index = _index_of(cands, chosen)

    # EVERY candidate gets a measured null term, including on draws where both
    # rules were decided without one. `fill_null_scores` does nothing at
    # lambda = 0 (nothing is outstanding for THAT score), so the measurement is
    # forced: a stored 0.0 must never mean "not measured".
    fill_null_scores(model, cands, rp_subject, device)
    if any(c.null_context != rp.null for c in cands):
        _score_null(model, cands, rp, device)

    row = {"subject": subject, "shipped_index": shipped_index,
           "subject_index": subject_index}
    if timing:
        kw = {"device": device, "tok": tok, "echo_words": echo_words}
        row["select_seconds_shipped"], again_shipped = _time_select(
            model, cands, rp, subject=None, **kw)
        row["select_seconds_subject"], again_subject = _time_select(
            model, cands, rp_subject, subject=subject, **kw)
        # A re-measured null term that flipped a winner would be worth knowing
        # about; it is recorded rather than allowed to abort a GPU run.
        row["timing_pass_agrees"] = (again_shipped == shipped_index
                                     and again_subject == subject_index)

    row["pool"] = [{
        # The returned text, stripped: the string both tiers judged. See
        # `snnchat.rerank.final_ids`.
        "text": tok.decode_visible(final_ids(c, tok, rp)).strip(),
        "n_chars": c.n_chars,
        "n_scored": len(c.scored),
        "closed": c.closed,
        "logp_cond": c.logp_cond,
        "logp_null": c.logp_null,
        "echo_weight": c.echo_weight,
        "subject_hit": c.subject_hit,
    } for c in cands]
    return row


def timing_summary(draws: list[dict]) -> dict | None:
    """Mean per-turn seconds for each selector, over the draws that were timed."""
    timed = [d for d in draws if "select_seconds_shipped" in d]
    if not timed:
        return None
    n = len(timed)
    return {
        "n_draws": n,
        "mean_select_seconds_shipped": sum(d["select_seconds_shipped"] for d in timed) / n,
        "mean_select_seconds_subject": sum(d["select_seconds_subject"] for d in timed) / n,
        "mean_rerank_seconds": sum(d["rerank_seconds"] for d in timed) / n,
        "timing_pass_disagreements": sum(1 for d in timed if not d["timing_pass_agrees"]),
        "note": "select_seconds are one selector running alone on a cleared copy "
                "of the pool, null pass included when that selector needs it; "
                "rerank_seconds is the draw plus the shipped selection",
    }


def write_pool(path: Path, blob: dict) -> None:
    """Compact JSON, gzipped, with the gzip timestamp zeroed.

    Raw pools are tens of megabytes and these get committed. `mtime=0` and an
    empty stored filename make the bytes a function of the content alone, so a
    re-run that reproduces the pool reproduces the file.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(blob, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    with open(path, "wb") as raw, \
            gzip.GzipFile(filename="", fileobj=raw, mode="wb", mtime=0) as gz:
        gz.write(data)


def read_pool(path: Path) -> dict:
    """The inverse of `write_pool`. Also reads plain `.json`, so the scorer can
    open a committed `echo_holdout.py` artifact through the same door."""
    raw = Path(path).read_bytes()
    if raw[:2] == b"\x1f\x8b":
        raw = gzip.decompress(raw)
    return json.loads(raw.decode("utf-8"))


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


@torch.no_grad()
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", default="experiments/chat/chat-v3d-aligned/ckpt_best.pt",
                    help="ROOT-relative or absolute")
    ap.add_argument("--set", dest="probe_set", default="heldout", choices=SET_NAMES)
    ap.add_argument("--seeds", type=int, default=6)
    ap.add_argument("--seed-offset", type=int, default=COMMITTED_SEEDS,
                    help="sampler seeds are range(offset, offset + seeds)")
    ap.add_argument("--allow-committed-seeds", action="store_true",
                    help="permit an offset below 6, i.e. redraw pools already read")
    ap.add_argument("--n", type=int, default=256, help="candidates per draw")
    ap.add_argument("--lam", type=float, default=0.6)
    ap.add_argument("--max-new", type=int, default=300)
    ap.add_argument("--no-graph", action="store_true",
                    help="disable the captured-graph stepper; same draws, slower")
    ap.add_argument("--no-timing", action="store_true",
                    help="skip the two timed selection passes per draw")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default=None,
                    help="default experiments/chat/_quality/v14_<set>_<run>.json.gz")
    args = ap.parse_args(argv)

    try:
        seeds = sampler_seeds(args.seeds, args.seed_offset,
                              allow_committed=args.allow_committed_seeds)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    ckpt = resolve(args.ckpt)
    out = resolve(args.out or f"experiments/chat/_quality/"
                              f"v14_{args.probe_set}_{ckpt.parent.name}.json.gz")
    model, _cfg, _ck = load_chat_checkpoint(ckpt, device=args.device)
    model.eval()
    tok = ChatTokenizer()
    rp = RerankParams(n=args.n, lam=args.lam, graph=not args.no_graph)

    probes = probes_for(args.probe_set)
    draws = []
    for prompt, words, kind in probes:
        for seed in seeds:
            row = draw(model, tok, prompt, seed, rp, max_new=args.max_new,
                       device=args.device, timing=not args.no_timing)
            draws.append({"kind": kind, "expect": list(words), **row})
        print(f"  {prompt!r}: {len(seeds)} draws", flush=True)

    blob = {
        "format": FORMAT,
        "ckpt": args.ckpt, "ckpt_sha256": _sha256(ckpt),
        "probe_set": args.probe_set,
        "seeds": args.seeds, "seed_offset": args.seed_offset, "sampler_seeds": seeds,
        "n": args.n, "lam": args.lam, "max_new": args.max_new,
        "min_chars": rp.min_chars, "null": rp.null,
        "trim_to_sentence": rp.trim_to_sentence,
        "device": str(args.device), "graph": rp.graph,
        "rerank_shipped": rp.describe(),
        "rerank_subject": dataclasses.replace(rp, subject_tier=True).describe(),
        "timing": timing_summary(draws),
        "n_draws": len(draws),
        "draws": draws,
    }
    write_pool(out, blob)

    same = sum(1 for d in draws if d["shipped_index"] == d["subject_index"])
    print(f"\n{args.probe_set}: {len(probes)} prompts x {len(seeds)} seeds "
          f"(sampler seeds {seeds[0]}..{seeds[-1]}) = {len(draws)} draws, "
          f"n={args.n} lambda={args.lam:g}")
    print(f"  both selectors chose the same draft on {same}/{len(draws)} draws")
    if blob["timing"]:
        t = blob["timing"]
        print(f"  mean select seconds: shipped {t['mean_select_seconds_shipped']:.4f}, "
              f"subject tier {t['mean_select_seconds_subject']:.4f} "
              f"(over {t['n_draws']} draws)")
    print(f"\nwrote {out}  ({out.stat().st_size / 1e6:.1f} MB); score with "
          "scripts/chat/score_v14.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
