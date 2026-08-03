"""EXP_009: distil the trained GRU into the adopted two-compartment arm.

Pre-registered in `experiments/logs/EXP_009_distillation_ceiling.md`. Read that
first -- in particular §1.1, which fixes how the result may be read: a positive
result is constructive and strong, a null says *this recipe* failed and **not**
that the remaining gap is representational.

**Nothing in `src/snn/` is modified by this file** (EXP_009 §8). The trainer is
subclassed here rather than extended there, because a diagnostic that will not be
adopted must not edit the trainer every adopted arm shares.

THE ONE STRUCTURAL DECISION
---------------------------
`snn.train.Trainer` captures its whole optimisation step into a CUDA graph. A
cuDNN GRU forward inside that capture is a risk with no upside, so the teacher
runs **outside** the captured region -- in `_fill_inputs`, which the base class
already calls once per step before the replay, and which is also what the
capture's own warm-up calls. Its output lands in a pre-allocated buffer whose
address the graph bakes in, exactly as `static_x` and `static_y` do.

That the teacher is outside the graph is not merely convenient: the teacher is a
deterministic function of the batch, so there is nothing about it that needs to
be replayed.

WHICH NUMBER GOES IN THE LOSS SLOT, AND WHY IT IS THE CROSS-ENTROPY
-------------------------------------------------------------------
The optimised objective is `(1-lam)*CE + lam*KL`. The JSONL's `loss` and `bpc`
fields are written by the base class from one static slot, and every other run in
this project puts its cross-entropy there. Writing the mixed objective into that
slot would produce a `bpc` field that is not a bits-per-character of anything --
a plausible-looking number in a log, which `snn.model`'s firing-rate sentinel
exists to avoid. So `_M_LOSS` keeps the CE, and the objective and the KL are
published separately into this class's own tensor and logged under their own
names.

PROVENANCE OF THE DISTILLATION SETTINGS
---------------------------------------
`lam`, the teacher run and the teacher's checkpoint hash are **not** `Config`
fields, because adding them would modify `src/snn/` for a diagnostic. They are
written to `experiments/runs/<run>/distill.json` and into the run's own JSONL at
`run_start`. The cost, stated rather than hidden: `config_hash` does not cover
them, so two distilled runs that differ only in `lam` would share a config hash.
EXP_009's Y4 compares the student's config against `twocomp_s{i}`'s with that
already known.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import subprocess
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO / "src") not in sys.path:
    sys.path.insert(0, str(_REPO / "src"))

from snn.config import Config  # noqa: E402
from snn.data import Corpus  # noqa: E402
from snn.evaluate import evaluate  # noqa: E402
from snn.model import build_model  # noqa: E402
from snn.train import Trainer, _M_GNORM, _M_LOSS  # noqa: E402

RUNS = _REPO / "experiments" / "runs"
LOCKFILE = _REPO / "src" / "snn" / ".MUTATION_CAMPAIGN_RUNNING"

# ---- fixed in the pre-registration, §2 ------------------------------------
SEEDS = (0, 1, 2)
LAMBDA = 0.5
TEMPERATURE = 1.0
STUDENT_REFERENCE = "twocomp_s{seed}"
TEACHER_RUN = "gru_s{seed}"
F1_TOLERANCE_BPC = 1e-3          # Y2, EXP_001's tolerance


def _config_from_checkpoint(ck: dict) -> Config:
    raw = ck.get("config")
    if not raw:
        raise SystemExit("checkpoint has no 'config' block")
    known = {f.name for f in dataclasses.fields(Config)}
    return Config(**{k: v for k, v in raw.items() if k in known})


def load_teacher(run: str, device: torch.device):
    """The frozen GRU anchor, in eval mode with grad disabled on every parameter.

    Built through `build_model` from the checkpoint's own config, so the teacher
    is the model that produced the committed number rather than a
    similarly-shaped rebuild of it (Y2 then checks that it reproduces).
    """
    ck = torch.load(RUNS / run / "ckpt_final.pt", map_location="cpu",
                    weights_only=False)
    cfg = _config_from_checkpoint(ck)
    cfg.device = str(device)
    teacher = build_model(cfg).to(device)
    teacher.load_state_dict(ck["model"])
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    return teacher, cfg, ck.get("config_hash")


class DistillTrainer(Trainer):
    """`Trainer` with one extra loss term. Two overrides and two buffers.

    At `lam = 0` this class must step **bitwise identically** to `Trainer`; that
    is EXP_009's Y1 and it is what makes the comparison against the committed
    un-distilled seeds mean anything.
    """

    def __init__(self, cfg: Config, teacher_run: str, lam: float) -> None:
        super().__init__(cfg)
        self.lam = float(lam)
        self.teacher_run = teacher_run
        self.teacher, self.teacher_cfg, self.teacher_hash = load_teacher(
            teacher_run, self.device)
        teacher_cfg = self.teacher_cfg
        if int(teacher_cfg.vocab_size) != int(cfg.vocab_size):
            raise SystemExit(
                f"teacher vocab {teacher_cfg.vocab_size} != student "
                f"{cfg.vocab_size}; the KL would be over different alphabets")
        if teacher_cfg.corpus != cfg.corpus:
            raise SystemExit("teacher and student were trained on different corpora")

        # Filled outside the captured region, read inside it.
        self.static_teacher_logp = torch.zeros(
            cfg.batch_size, cfg.seq_len, cfg.vocab_size,
            device=self.device, dtype=torch.float32)
        # This class's own metric slots, kept out of `static_metrics` because the
        # base class's `step()` treats every slot past _M_RATE0 as a firing rate.
        self.static_distill = torch.zeros(2, device=self.device,
                                          dtype=torch.float32)
        # Y3: a fingerprint of the teacher, re-checked at the end of training.
        self._teacher_fingerprint = self._fingerprint_teacher()

        self._log({"event": "distill_start", "teacher_run": teacher_run,
                   "teacher_config_hash": self.teacher_hash, "lambda": self.lam,
                   "temperature": TEMPERATURE})

    def _fingerprint_teacher(self) -> list[float]:
        with torch.no_grad():
            return [float(p.double().sum()) for p in self.teacher.parameters()]

    def teacher_unchanged(self) -> bool:
        """Y3. Sum-of-parameters per tensor, in float64 so the sum is exact
        enough to notice a single updated element."""
        return self._fingerprint_teacher() == self._teacher_fingerprint

    def _fill_inputs(self, step: int, non_blocking: bool = True):
        """Base behaviour, plus the teacher's log-probabilities for this batch.

        Deliberately after `super()`, so the teacher reads the *same*
        `self.static_x` the student's forward will read. Feeding it the sampler's
        CPU tensor instead would be a second path to the same data and therefore
        a way for them to differ (EXP_009 Q1).
        """
        out = super()._fill_inputs(step, non_blocking=non_blocking)
        with torch.no_grad():
            logits, _, _ = self.teacher(self.static_x, None)
            self.static_teacher_logp.copy_(
                F.log_softmax(logits.float() / TEMPERATURE, dim=-1))
        return out

    def _step_body(self) -> None:
        """The base class's body with `(1-lam)*CE + lam*KL` in place of CE.

        Every constraint the base class's docstring lists still applies: no
        `.item()`, no Python conditional, no host synchronisation. `self.lam` is a
        Python float read at capture time and never changed afterwards, which is
        the one thing that would need a live tensor if this experiment ever swept
        it mid-run (it does not; §2.3).
        """
        self.opt.zero_grad(set_to_none=False)
        logits, _, aux = self.model(self.static_x, None)
        flat = logits.reshape(-1, self._vocab).float()
        ce = F.cross_entropy(flat, self.static_y.reshape(-1))
        kl = F.kl_div(
            F.log_softmax(flat / TEMPERATURE, dim=-1),
            self.static_teacher_logp.reshape(-1, self._vocab),
            log_target=True, reduction="batchmean",
        )
        loss = (1.0 - self.lam) * ce + self.lam * kl
        loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(
            self.params, self._max_norm, norm_type=2.0,
            error_if_nonfinite=False, foreach=True,
        )
        self.opt.step()
        # The CE, not the objective -- see the module docstring.
        self.static_metrics[_M_LOSS].copy_(ce.detach())
        self.static_metrics[_M_GNORM].copy_(gnorm.detach())
        self.static_distill[0].copy_(loss.detach())
        self.static_distill[1].copy_(kl.detach())
        for slot, rate in zip(self._rate_slots, aux.get("firing_rate", ())):
            self.static_metrics[slot].copy_(rate.detach().reshape(()))

    def step(self) -> dict:
        m = super().step()
        d = self.static_distill.detach().to("cpu").tolist()
        m["objective"] = d[0]
        m["kl_to_teacher"] = d[1]
        return m


# --------------------------------------------------------------------------
# Y2 -- the teacher reproduces its own committed score, through this harness
# --------------------------------------------------------------------------

def check_teacher_reproduces(teacher, cfg: Config, teacher_run: str) -> dict:
    """The self-check that makes everything else here trustworthy.

    Scores the teacher this file loaded, on the split its committed number was
    produced on, through `snn.evaluate.evaluate` -- and requires the committed
    number to come back. A mis-loaded teacher, a vocab mismatch or a teacher left
    in `train()` all produce a plausible distillation and a meaningless bound.
    """
    committed_path = RUNS / teacher_run / "final_test.json"
    blob = json.loads(committed_path.read_text(encoding="utf-8"))
    committed = blob.get("results", blob)["fresh"]["bpc"]

    corpus = Corpus(cfg.corpus, cfg.data_dir)
    got = evaluate(teacher, corpus, "test", cfg, "fresh", max_windows=0)["bpc"]
    residual = abs(got - committed)
    row = {"teacher_run": teacher_run, "committed_fresh_bpc": committed,
           "reproduced_fresh_bpc": got, "residual_bpc": residual,
           "pass": residual < F1_TOLERANCE_BPC}
    if not row["pass"]:
        raise SystemExit(
            f"Y2 FAILED for {teacher_run}: this harness scores the teacher at "
            f"{got:.6f} against its committed {committed:.6f} (residual "
            f"{residual:.2e}). Chase it; do not widen the tolerance.")
    return row


def _student_config(seed: int) -> Config:
    """The adopted arm's own config, with only the seed and the run name changed.

    Read from `twocomp_s{seed}/config.json` rather than retyped, so that the
    student is the adopted arm by construction; Y4 then verifies the result.
    """
    ref = json.loads(
        (RUNS / STUDENT_REFERENCE.format(seed=seed) / "config.json")
        .read_text(encoding="utf-8"))
    known = {f.name for f in dataclasses.fields(Config)}
    cfg = Config(**{k: v for k, v in ref.items() if k in known})
    cfg.run_name = f"twocomp_distill_s{seed}"
    cfg.seed = seed
    return cfg


def train_one(seed: int, lam: float) -> dict:
    if LOCKFILE.exists():
        raise SystemExit(f"Y6: {LOCKFILE.relative_to(_REPO)} exists")

    cfg = _student_config(seed)
    teacher_run = TEACHER_RUN.format(seed=seed)
    print(f"\n=== distil {teacher_run} -> {cfg.run_name} (lambda={lam})",
          flush=True)

    t0 = time.time()
    trainer = DistillTrainer(cfg, teacher_run, lam)
    y2 = check_teacher_reproduces(trainer.teacher, trainer.teacher_cfg, teacher_run)
    print(f"    Y2 OK: teacher reproduces {y2['committed_fresh_bpc']:.6f} "
          f"(residual {y2['residual_bpc']:.2e})", flush=True)

    trainer.train()
    wall = time.time() - t0

    if not trainer.teacher_unchanged():
        raise SystemExit(f"Y3 FAILED for {cfg.run_name}: the teacher moved")

    run_dir = RUNS / cfg.run_name
    (run_dir / "distill.json").write_text(json.dumps({
        "experiment": "EXP_009_distillation_ceiling",
        "log": "experiments/logs/EXP_009_distillation_ceiling.md",
        "student_reference": STUDENT_REFERENCE.format(seed=seed),
        "teacher_run": teacher_run,
        "teacher_config_hash": trainer.teacher_hash,
        "lambda": lam,
        "temperature": TEMPERATURE,
        "loss": "(1-lambda)*CE(student, y) + lambda*KL(teacher || student)",
        "y2_teacher_reproduces": y2,
        "y3_teacher_unchanged": True,
        "wall_clock_s": round(wall, 1),
    }, indent=2), encoding="utf-8")

    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    return {"run": cfg.run_name, "seed": seed, "teacher_run": teacher_run,
            "lambda": lam, "wall_clock_s": summary.get("wall_clock_s"),
            "peak_vram_gib": summary.get("peak_vram_gib"),
            "y2": y2, "y3_teacher_unchanged": True,
            "driver_seconds": round(wall, 1)}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", default=",".join(str(s) for s in SEEDS))
    ap.add_argument("--lam", type=float, default=LAMBDA)
    ap.add_argument("--max-steps", type=int, default=0,
                    help="smoke-test override; 0 = the pre-registered 20 000")
    ap.add_argument("--out", default="docs/reports/data/exp_009_run_manifest.json")
    args = ap.parse_args(argv)
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]

    if args.max_steps:
        # Smoke path only. It writes into differently-named runs so it can never
        # be mistaken for, or overwrite, a pre-registered one.
        return _smoke(seeds[0], args.lam, args.max_steps)

    started = time.time()
    manifest = {"experiment": "EXP_009_distillation_ceiling",
                "log": "experiments/logs/EXP_009_distillation_ceiling.md",
                "lambda": args.lam, "temperature": TEMPERATURE,
                "sequential": True, "seeds": seeds, "runs": []}
    for seed in seeds:
        row = train_one(seed, args.lam)
        run = row["run"]
        proc = subprocess.run(
            [sys.executable, "-u", "scripts/evaluate.py",
             "--ckpt", str(RUNS / run / "ckpt_final.pt"), "--split", "test",
             "--out", str(RUNS / run / "final_test.json")], cwd=str(_REPO))
        if proc.returncode != 0:
            raise SystemExit(f"evaluate {run} failed")
        test = json.loads((RUNS / run / "final_test.json").read_text("utf-8"))
        row["test_bpc"] = {p: test["results"][p]["bpc"]
                           for p in ("fresh", "carried")}
        manifest["runs"].append(row)
        print(f"    {run}: fresh {row['test_bpc']['fresh']:.4f}  "
              f"carried {row['test_bpc']['carried']:.4f}", flush=True)
        out = _REPO / args.out
        out.parent.mkdir(parents=True, exist_ok=True)
        manifest["total_seconds"] = round(time.time() - started, 1)
        out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("\n" + "=" * 70)
    for r in manifest["runs"]:
        print(f"  {r['run']:22s} carried {r['test_bpc']['carried']:.4f}   "
              f"{r['wall_clock_s']:.0f}s")
    print(f"WROTE {args.out}")
    return 0


def _smoke(seed: int, lam: float, max_steps: int) -> int:
    """A few steps of the real thing, plus Y1 and Q5, before 30 GPU-minutes.

    Y1: at `lam = 0` this subclass must step bitwise like `Trainer`.
    Q5: the teacher buffer must actually change between consecutive steps -- a
    buffer filled once and then replayed from the graph would look identical to a
    working distillation for the first step and be stale forever after.
    """
    print(f"SMOKE: seed {seed}, {max_steps} steps, lambda {lam}")
    cfg = _student_config(seed)
    cfg.run_name = f"_smoke_distill_s{seed}"
    cfg.max_steps = max_steps
    cfg.eval_every = 0
    cfg.ckpt_every = 0
    cfg.log_every = 1

    # ---- Y1: lam = 0 is the committed trainer, bitwise -----------------------
    ref_cfg = _student_config(seed)
    ref_cfg.run_name = "_smoke_distill_ref"
    ref_cfg.max_steps = max_steps
    ref_cfg.eval_every = 0
    ref_cfg.ckpt_every = 0
    ref_cfg.log_every = 1
    base = Trainer(ref_cfg)
    base_m = [base.step() for _ in range(3)]
    base_params = [p.detach().clone() for p in base.params]
    del base
    torch.cuda.empty_cache()

    zero_cfg = _student_config(seed)
    zero_cfg.run_name = "_smoke_distill_lam0"
    zero_cfg.max_steps = max_steps
    zero_cfg.eval_every = 0
    zero_cfg.ckpt_every = 0
    zero_cfg.log_every = 1
    zero = DistillTrainer(zero_cfg, TEACHER_RUN.format(seed=seed), 0.0)
    zero_m = [zero.step() for _ in range(3)]
    zero_params = [p.detach().clone() for p in zero.params]
    y1_loss = all(a["loss"] == b["loss"] and a["grad_norm"] == b["grad_norm"]
                  for a, b in zip(base_m, zero_m))
    y1_params = all(torch.equal(a, b) for a, b in zip(base_params, zero_params))
    print(f"  Y1 losses bitwise: {y1_loss}   parameters bitwise: {y1_params}")
    for a, b in zip(base_m, zero_m):
        print(f"     step {a['step']}: base {a['loss']:.10f}  lam0 {b['loss']:.10f}")
    del zero
    torch.cuda.empty_cache()

    # ---- Q5: the teacher buffer moves every step -----------------------------
    trainer = DistillTrainer(cfg, TEACHER_RUN.format(seed=seed), lam)
    y2 = check_teacher_reproduces(trainer.teacher, cfg, TEACHER_RUN.format(seed=seed))
    print(f"  Y2: teacher reproduces {y2['committed_fresh_bpc']:.6f} "
          f"(residual {y2['residual_bpc']:.2e})  pass={y2['pass']}")
    seen = []
    for _ in range(3):
        m = trainer.step()
        seen.append(trainer.static_teacher_logp.detach().clone())
        print(f"     step {m['step']}: ce {m['loss']:.4f}  kl {m['kl_to_teacher']:.4f} "
              f" objective {m['objective']:.4f}")
    q5 = not torch.equal(seen[0], seen[1]) and not torch.equal(seen[1], seen[2])
    print(f"  Q5 teacher buffer changes every step: {q5}")
    print(f"  Y3 teacher unchanged: {trainer.teacher_unchanged()}")
    ok = y1_loss and y1_params and y2["pass"] and q5 and trainer.teacher_unchanged()
    print(f"\nSMOKE {'PASSED' if ok else 'FAILED'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
