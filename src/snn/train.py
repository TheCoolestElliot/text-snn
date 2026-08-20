"""Trainer: static input buffers, CUDA-graph capture, resumable checkpoints.

Why this module looks the way it does
-------------------------------------
Phase-1 measured a flat ~14.5 us CPU-side cost per CUDA kernel launch on this
box (01_reconnaissance.md 3.3/3.4), and CUDA-graph capture of a full
fwd+bwd+AdamW step removed 10.9-13.1x of it (3.8).  Capture is therefore a
*design constraint*, not an optimisation to bolt on later (8.1.2): fixed
shapes, pre-allocated input buffers, a capturable optimiser, and no host
synchronisation or data-dependent control flow anywhere inside the captured
region.

Three traps are handled explicitly, because each of them fails *silently* --
the run keeps going, the log looks plausible, and the science is wrong.

1.  **A Python-float learning rate is baked into the graph at capture time.**
    The schedule then appears to work: ``param_group['lr']`` changes every
    step and the JSONL log faithfully records the change, while the replayed
    kernels keep using whatever value was live during capture.  The optimiser
    is therefore constructed with an ``lr`` **tensor** and the schedule writes
    into that tensor *in place*; the replayed graph reads the new value
    because it captured a pointer, not a constant.
    ``tests/test_graph_equivalence.py::test_lr_tensor_is_live_in_graph``
    fails if this ever regresses.

2.  **Warm-up mutates the model.**  The proven capture recipe
    (scripts/audit/04_graphed_scaling.py) runs three real steps on a side
    stream before capture.  Those steps advance the parameters and the Adam
    moments, so a run that then reports "step 0" is already at step 3 and the
    R5 graphed-vs-eager comparison is meaningless.  The pre-warm-up state is
    snapshotted and restored **in place** (``copy_``/``zero_``) afterwards, so
    the tensor addresses the graph baked in stay valid.

3.  **Anything that synchronises inside the captured region is a capture
    failure.**  ``.item()``, ``print``, ``float(tensor)``, and the classic
    ``if clip_coef < 1`` formulation of gradient clipping all sync.  Metrics
    are published by ``copy_`` into one small static fp32 tensor and read
    *outside* the region, once per step.

The eager path (``cfg.cuda_graph = False``) runs the byte-identical
``_step_body``; only the dispatch mechanism differs.  That is what makes the
R5 gate a meaningful comparison rather than a comparison of two programs.
"""

from __future__ import annotations

import json
import math
import os
import platform
import random
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch

from snn.config import Config, config_hash, seed_everything
from snn.data import Corpus, RandomWindowSampler
from snn.evaluate import evaluate
from snn.metrics import bits_per_char
from snn.model import build_model, count_params
from snn.neuromod import three_factor_loss

# Slot layout of the single static metrics tensor.  One small D2H transfer per
# step instead of one sync per scalar.
_M_LOSS = 0
_M_GNORM = 1
_M_RATE0 = 2

# Sentinel for a firing-rate slot that this architecture never writes. A rate
# lives in [0, 1], so -1 is unambiguous, and NaN deliberately survives the
# filter: a NaN firing rate means the model has diverged and must stay visible.
_RATE_UNSET = -1.0

# The recipe in scripts/audit/04_graphed_scaling.py, which is the version that
# is known to work on this machine.  Three steps is also what the upstream
# PyTorch graph documentation uses; do not reduce it -- the first step allocates
# gradients and the jiterator kernels are NVRTC-compiled lazily on first use, and
# neither may happen during capture.
_WARMUP_STEPS = 3

# Config has no eps field; AdamW's default is recorded here so it is visible.
_ADAM_EPS = 1e-8


class Trainer:
    """One optimisation step, capturable, with everything needed to resume it.

    A single parameter group is used deliberately: the spec exposes exactly one
    ``weight_decay``, so splitting biases/embeddings out would be an unlogged
    hyperparameter choice.  It also keeps exactly one LR tensor alive, which is
    what makes the schedule-inside-a-graph fix auditable.
    """

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        # Seeding here (rather than only in the CLI) makes Trainer(cfg)
        # reproducible on its own, which the determinism gate relies on.
        seed_everything(cfg.seed, cfg.deterministic)

        self.device = torch.device(cfg.device)
        self.is_cuda = self.device.type == "cuda"
        if self.is_cuda and not torch.cuda.is_available():
            raise RuntimeError("cfg.device is cuda but torch.cuda.is_available() is False")

        # --- data ---------------------------------------------------------
        self.corpus = Corpus(cfg.corpus, cfg.data_dir)
        if cfg.vocab_size == 0:
            cfg.vocab_size = int(self.corpus.vocab_size)
        elif int(cfg.vocab_size) != int(self.corpus.vocab_size):
            raise ValueError(
                f"cfg.vocab_size={cfg.vocab_size} disagrees with corpus "
                f"{cfg.corpus} (vocab_size={self.corpus.vocab_size})"
            )
        self.train_data = self.corpus.split("train")
        self.sampler = RandomWindowSampler(
            self.train_data, cfg.batch_size, cfg.seq_len, cfg.seed
        )

        # --- run identity ---------------------------------------------------
        # Resolve the run name from the hash of the *pre-naming* config so that
        # a re-launch of the same experiment lands in the same directory and
        # resumes rather than forking a new one.
        if not cfg.run_name:
            cfg.run_name = _default_run_name(cfg)
        self.config_hash = config_hash(cfg)
        self.run_dir = Path(cfg.out_dir) / cfg.run_name
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.run_dir / "log.jsonl"

        # --- model / optimiser ----------------------------------------------
        self.model = build_model(cfg).to(self.device)
        # Read once, into Python floats, so the captured step reads no config
        # object and the graph bakes in a constant rather than an attribute
        # lookup. `tf_tau` is EXP_018's MEASURED da_scale reused deliberately:
        # the three-factor weight squashes the HEAD's surprise over the same
        # 205-way vocabulary, in the same nats, so it is the same quantity --
        # unlike `nm_tau`, which is a different quantity in different units and
        # is calibrated separately (CONTRIBUTING.md 4: never inherit a constant
        # across a structurally different arm).
        self._tf_kappa = float(cfg.tf_kappa)
        self._tf_tau = float(cfg.da_scale)
        self._tf_rolled = bool(cfg.tf_rolled)
        self.model.train()
        self.params = [p for p in self.model.parameters() if p.requires_grad]
        self.n_params = count_params(self.model)

        # capturable=True is used whenever we are on CUDA, *including* the
        # eager path.  The R5 gate compares two dispatch mechanisms, so both
        # must run identical optimiser arithmetic; capturable changes the
        # bias-correction arithmetic to device tensors and that difference must
        # not be a confound.
        self.lr_tensor: torch.Tensor | None = None
        lr_arg: Any = float(cfg.lr)
        if self.is_cuda:
            self.lr_tensor = torch.tensor(float(cfg.lr), device=self.device,
                                          dtype=torch.float32)
            lr_arg = self.lr_tensor
        self.opt = torch.optim.AdamW(
            self.params,
            lr=lr_arg,
            betas=(cfg.beta1, cfg.beta2),
            eps=_ADAM_EPS,
            weight_decay=cfg.weight_decay,
            capturable=self.is_cuda,
            foreach=True,
        )

        # --- static buffers ---------------------------------------------------
        # Addresses are baked into the graph at capture; these tensors are
        # allocated once and only ever written through copy_.
        self.static_x = torch.zeros(cfg.batch_size, cfg.seq_len,
                                    dtype=torch.int64, device=self.device)
        self.static_y = torch.zeros(cfg.batch_size, cfg.seq_len,
                                    dtype=torch.int64, device=self.device)
        self.static_metrics = torch.zeros(_M_RATE0 + cfg.n_layers,
                                          dtype=torch.float32, device=self.device)
        # Firing-rate slots start at a sentinel a rate can never take, so that
        # `step()` can tell "this arm reported no rate" from "this layer fired
        # at 0.0" without a branch inside the captured region.  The GRU control
        # returns an empty rate list, and reporting 0% firing for a model with
        # no spikes would be a category error waiting to be quoted.
        self.static_metrics[_M_RATE0:].fill_(_RATE_UNSET)

        # EXP_013: the noise arm's per-layer buffers are static for the same
        # reason static_x is, and must exist BEFORE _capture() bakes in their
        # addresses.  A no-op for every other arm.
        allocate = getattr(self.model, "allocate_noise", None)
        if allocate is not None:
            allocate(cfg.batch_size, cfg.seq_len)

        # Hoisted out of _step_body so the captured region contains no Python
        # conditional at all (see the module docstring, trap 3).
        self._vocab = int(cfg.vocab_size)
        self._max_norm = float(cfg.grad_clip) if cfg.grad_clip > 0 else float("inf")
        self._rate_slots = tuple(range(_M_RATE0, _M_RATE0 + cfg.n_layers))

        self.graph: torch.cuda.CUDAGraph | None = None
        self.capture_seconds = 0.0
        self._ready = False
        self.global_step = 0
        self.best_val_bpc = float("inf")
        self.loaded_config: dict | None = None
        self._last_eval: dict | None = None
        self._last_eval_step = -1

        if self.is_cuda:
            torch.cuda.reset_peak_memory_stats()

        self._write_run_header()

    # ------------------------------------------------------------------
    # the captured region
    # ------------------------------------------------------------------

    def _step_body(self) -> None:
        """zero_grad -> forward -> backward -> clip -> step, capture-safe.

        Every statement here executes identically in eager mode and inside the
        graph.  The constraints, in order of how easy they are to violate:

        * ``set_to_none=False`` -- gradients must keep stable addresses, so
          they are zeroed rather than freed.
        * ``_clip_grad_norm_fp64`` is a branchless, capture-safe replacement
          for ``clip_grad_norm_`` (decision #7, rev 6): it computes
          ``clamp(max_norm / (total_norm + 1e-6), max=1.0)`` on device and
          always multiplies, exactly as the stock implementation does, but
          accumulates the sum of squares in fp64 so a legitimately large
          (but finite) gradient cannot overflow the norm to ``inf`` and get
          silently zeroed instead of clipped -- see the function's own
          docstring and ``EXP_009`` Sec 9.4.
        * No ``.item()``, no ``print``, no Python ``if``.  Metrics leave via
          ``copy_`` into a static tensor and are read outside the region.
        """
        self.opt.zero_grad(set_to_none=False)
        logits, _, aux = self.model(self.static_x, None)
        # EXP_021 form 2. At tf_kappa = 0.0 -- every arm but that one --
        # `three_factor_loss` returns `F.cross_entropy(flat, tgt)`, the same
        # call this line used to make, so the objective is unchanged BY CODE
        # PATH and not by a float identity. The weighting is capture-safe: no
        # `.item()`, no host read, no data-dependent branch, and the only new
        # kernels are a log-softmax that was already implicit in the CE and one
        # reduction over V.
        loss = three_factor_loss(
            logits, self.static_y, kappa=self._tf_kappa, tau=self._tf_tau,
            roll_control=self._tf_rolled,
        )
        loss.backward()
        gnorm = _clip_grad_norm_fp64(self.params, self._max_norm)
        self.opt.step()
        self.static_metrics[_M_LOSS].copy_(loss.detach())
        self.static_metrics[_M_GNORM].copy_(gnorm.detach())
        # zip truncates, so a control arm that reports no firing rates (GRU)
        # leaves those slots holding _RATE_UNSET -- deliberately *not* 0.0, see
        # the sentinel's definition -- without needing a branch here.
        for slot, rate in zip(self._rate_slots, aux.get("firing_rate", ())):
            self.static_metrics[slot].copy_(rate.detach().reshape(()))

    # ------------------------------------------------------------------
    # capture
    # ------------------------------------------------------------------

    def _ensure_ready(self) -> None:
        if self._ready:
            return
        self._ready = True
        if self.cfg.cuda_graph and self.is_cuda:
            self._capture()

    def _capture(self) -> None:
        """Warm up on a side stream, then capture one full step.

        Verbatim from scripts/audit/04_graphed_scaling.py, which is the version
        measured to work on this driver.  The only addition is the state
        snapshot/restore around it (module docstring, trap 2).
        """
        snapshot = self._snapshot_train_state()
        self._set_lr(self._lr_at(self.global_step))

        keep = []  # hold the pinned CPU batches until the copies have landed
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for i in range(_WARMUP_STEPS):
                keep.append(self._fill_inputs(self.global_step + i, non_blocking=False))
                self._step_body()
        torch.cuda.current_stream().wait_stream(s)
        del keep

        t0 = time.perf_counter()
        graph = torch.cuda.CUDAGraph()
        try:
            with torch.cuda.graph(graph):
                self._step_body()
            self.graph = graph
            self.capture_seconds = time.perf_counter() - t0
        except Exception as exc:
            raise RuntimeError(
                "CUDA-graph capture of the training step failed. This is R1 in "
                "01_reconnaissance.md 7; the documented fallback is "
                "cfg.cuda_graph=False, which runs the identical step eagerly at "
                "roughly 1/11 the throughput. Find what synchronised inside "
                "_step_body rather than papering over it."
            ) from exc
        finally:
            # Undo the warm-up.  Must be in place: the graph now holds the
            # addresses of these exact tensors.
            self._restore_train_state(snapshot)

    def _snapshot_train_state(self) -> tuple[list[torch.Tensor], list[dict]]:
        """Clone parameters and optimiser moments so warm-up can be undone."""
        params = [p.detach().clone() for p in self.params]
        state: list[dict] = []
        for p in self.params:
            st = self.opt.state.get(p, {})
            state.append({
                k: (v.detach().clone() if torch.is_tensor(v) else v)
                for k, v in st.items()
            })
        return params, state

    def _restore_train_state(self, snapshot: tuple[list[torch.Tensor], list[dict]]) -> None:
        """Restore a snapshot *in place*, so captured addresses stay valid.

        Optimiser entries that did not exist when the snapshot was taken (the
        usual case: Adam allocates its moments lazily on the first step) are
        zeroed, which is exactly the state a fresh optimiser would present.
        """
        params, state = snapshot
        with torch.no_grad():
            for p, saved in zip(self.params, params):
                p.copy_(saved)
            for p, saved_state in zip(self.params, state):
                st = self.opt.state.get(p)
                if not st:
                    continue
                for k, v in list(st.items()):
                    if not torch.is_tensor(v):
                        if k in saved_state:
                            st[k] = saved_state[k]
                        continue
                    src = saved_state.get(k)
                    if torch.is_tensor(src):
                        v.copy_(src)
                    else:
                        v.zero_()

    # ------------------------------------------------------------------
    # learning-rate schedule
    # ------------------------------------------------------------------

    def _lr_at(self, step: int) -> float:
        """Linear warm-up then cosine decay to zero, as a pure function of step.

        Pure so that a resumed run recomputes exactly the same value rather than
        carrying scheduler state that could drift out of sync with global_step.
        """
        cfg = self.cfg
        if cfg.warmup_steps > 0 and step < cfg.warmup_steps:
            return float(cfg.lr) * (step + 1) / float(cfg.warmup_steps)
        if cfg.lr_schedule == "cosine":
            span = max(1, cfg.max_steps - cfg.warmup_steps)
            prog = min(max((step - cfg.warmup_steps) / span, 0.0), 1.0)
            return float(cfg.lr) * 0.5 * (1.0 + math.cos(math.pi * prog))
        if cfg.lr_schedule == "constant":
            return float(cfg.lr)
        raise ValueError(f"unknown lr_schedule: {cfg.lr_schedule!r}")

    def _set_lr(self, lr: float) -> None:
        """Write the LR where the *captured graph* will read it.

        With ``capturable=True`` AdamW keeps ``param_group['lr']`` as a CUDA
        tensor, and the captured kernels dereference that tensor's storage.
        Filling it in place therefore changes what a replay does; rebinding
        ``group['lr']`` to a new Python float does not.  Both forms are handled
        because the CPU device path has no tensor LR.
        """
        for group in self.opt.param_groups:
            cur = group.get("lr")
            if torch.is_tensor(cur):
                cur.fill_(float(lr))
            else:
                group["lr"] = float(lr)

    def _current_lr(self) -> float:
        cur = self.opt.param_groups[0]["lr"]
        return float(cur.item()) if torch.is_tensor(cur) else float(cur)

    # ------------------------------------------------------------------
    # stepping
    # ------------------------------------------------------------------

    def _fill_inputs(self, step: int, non_blocking: bool = True):
        x, y = self.sampler.batch(step)
        self.static_x.copy_(x, non_blocking=non_blocking)
        self.static_y.copy_(y, non_blocking=non_blocking)
        # EXP_013's noise is staged exactly where the batch is, and for the same
        # reason: this is the one place outside the captured region that every
        # path -- warm-up, capture and replay -- funnels through, so putting the
        # redraw anywhere else would silently leave the warm-up steps unnoised.
        # A no-op for every arm but `noise`, and for `noise` at amplitude 0.
        refill = getattr(self.model, "refill_noise", None)
        if refill is not None:
            refill(self.cfg.seed, step)
        return x, y  # returned so callers can keep pinned buffers alive

    def step(self) -> dict:
        """One optimisation step; returns the metrics for this step.

        The single ``.to("cpu")`` at the end is the only host synchronisation in
        the loop.  It costs nothing that we would not pay anyway (the next step
        cannot be enqueued until its batch is sampled) and it keeps ``loss``
        available every step, which the R5 gate needs.
        """
        self._ensure_ready()
        step_index = self.global_step
        batch = self._fill_inputs(step_index)
        lr = self._lr_at(step_index)
        self._set_lr(lr)

        if self.graph is not None:
            self.graph.replay()
        else:
            self._step_body()

        m = self.static_metrics.detach().to("cpu").tolist()
        del batch  # copies have landed: the D2H read above synchronised
        self.global_step += 1
        return {
            "step": step_index,
            "loss": m[_M_LOSS],
            "bpc": bits_per_char(m[_M_LOSS], 1),
            "grad_norm": m[_M_GNORM],
            "lr": lr,
            "firing_rate": [r for r in m[_M_RATE0:] if r != _RATE_UNSET],
        }

    def train(self) -> None:
        cfg = self.cfg
        self._ensure_ready()
        start_wall = time.perf_counter()
        start_step = self.global_step
        try:
            while self.global_step < cfg.max_steps:
                m = self.step()
                s = m["step"]
                is_last = self.global_step >= cfg.max_steps
                if cfg.log_every > 0 and (s % cfg.log_every == 0 or is_last):
                    elapsed = time.perf_counter() - start_wall
                    done = max(1, self.global_step - start_step)
                    rec = dict(m)
                    rec.update({
                        "event": "train",
                        # "step" is the 0-based index of the step just run;
                        # "steps_completed" is what an "eval" record's "step"
                        # means.  They differ by one and joining the two record
                        # types on the wrong key silently shifts every curve.
                        "steps_completed": self.global_step,
                        "tokens": self.global_step * cfg.batch_size * cfg.seq_len,
                        "elapsed_s": round(elapsed, 3),
                        "steps_per_s": round(done / max(elapsed, 1e-9), 4),
                        "peak_vram_gib": self._peak_vram(),
                    })
                    self._log(rec)
                    rates = ", ".join(f"{r:.3f}" for r in m["firing_rate"])
                    print(
                        f"step {s:>7d}  loss {m['loss']:.4f}  bpc {m['bpc']:.4f}  "
                        f"lr {m['lr']:.3e}  |g| {m['grad_norm']:.3f}  rate [{rates}]",
                        flush=True,
                    )
                if cfg.eval_every > 0 and self.global_step % cfg.eval_every == 0:
                    self._run_eval("val")
                if cfg.ckpt_every > 0 and self.global_step % cfg.ckpt_every == 0:
                    self.save_checkpoint(self.run_dir / "ckpt_last.pt")
        except KeyboardInterrupt:
            # R6/R7: an interrupted multi-hour run must still be resumable.
            self.save_checkpoint(self.run_dir / "ckpt_interrupt.pt")
            raise

        # Skip if the loop already evaluated at exactly this step -- a full-split
        # eval is minutes, and a duplicate row in log.jsonl invites double
        # counting when the report is assembled.
        final = (self._last_eval
                 if self._last_eval is not None and self._last_eval_step == self.global_step
                 else self._run_eval("val"))
        self.save_checkpoint(self.run_dir / "ckpt_final.pt")
        self.save_checkpoint(self.run_dir / "ckpt_last.pt")
        summary = {
            "run_name": cfg.run_name,
            "config_hash": self.config_hash,
            "steps": self.global_step,
            "params": self.n_params,
            "wall_clock_s": round(time.perf_counter() - start_wall, 2),
            "peak_vram_gib": self._peak_vram(),
            "val": final,
        }
        _write_json(self.run_dir / "summary.json", summary)
        self._log({"event": "run_end", **summary})

    def _run_eval(self, split: str) -> dict:
        """Both protocols, every time -- 01_reconnaissance.md 4.3 makes that
        mandatory for any headline number, and a monitoring number that uses a
        different protocol from the headline is a trap waiting to be quoted."""
        out = {}
        self._last_eval_step = self.global_step
        for protocol in ("fresh", "carried"):
            res = evaluate(self.model, self.corpus, split, self.cfg, protocol,
                           max_windows=self.cfg.eval_max_windows)
            out[protocol] = res
            # "steps_completed" is carried on *both* record types so that the
            # join key the train records document actually exists on the eval
            # records; "step" is kept for backwards compatibility and holds the
            # same value here (an eval happens *between* steps, never during
            # one), which is exactly the ambiguity that makes the explicit key
            # worth writing.
            self._log({"event": "eval", "step": self.global_step,
                       "steps_completed": self.global_step, **res})
            print(
                f"  eval[{split}/{protocol}] step {self.global_step:>7d}  "
                f"bpc {res['bpc']:.4f}  chars {res['n_chars']}",
                flush=True,
            )
        self.model.train()
        bpc = out["carried"]["bpc"]
        if bpc < self.best_val_bpc:
            self.best_val_bpc = bpc
            self.save_checkpoint(self.run_dir / "ckpt_best.pt")
        self._last_eval = out
        return out

    # ------------------------------------------------------------------
    # checkpoints
    # ------------------------------------------------------------------

    def save_checkpoint(self, path) -> None:
        """Everything needed to make a resume bit-identical to not stopping.

        RNG states are saved for all four generators even though the Phase-2
        model draws no random numbers after initialisation: the sampler is
        seeded per (seed, step) precisely so resume does not depend on RNG, but
        an unrecorded generator is exactly the kind of thing that silently
        breaks reproducibility once anyone adds dropout.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        ck = {
            "format": 1,
            "step": self.global_step,
            "model": self.model.state_dict(),
            "optimizer": self.opt.state_dict(),
            "config": asdict(self.cfg),
            "config_hash": self.config_hash,
            "best_val_bpc": self.best_val_bpc,
            "rng": {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch": torch.get_rng_state(),
                "torch_cuda": (torch.cuda.get_rng_state_all()
                               if torch.cuda.is_available() else []),
            },
            "torch_version": torch.__version__,
        }
        tmp = path.with_suffix(path.suffix + ".tmp")
        torch.save(ck, tmp)
        os.replace(tmp, path)

    def load_checkpoint(self, path) -> None:
        """Restore in place wherever a captured graph might already hold pointers.

        ``Module.load_state_dict`` already copies into the existing parameters,
        but ``Optimizer.load_state_dict`` *rebinds* the moment tensors to the
        objects unpickled from disk.  After capture that silently orphans the
        graph from the state it is stepping, so once optimiser state exists we
        copy into it instead.
        """
        path = Path(path)
        ck = torch.load(path, map_location=self.device, weights_only=False)

        self.model.load_state_dict(ck["model"])
        self._load_optimizer_state(ck["optimizer"])
        self.global_step = int(ck["step"])
        self.best_val_bpc = float(ck.get("best_val_bpc", float("inf")))
        self.loaded_config = ck.get("config")

        rng = ck.get("rng", {})
        if "python" in rng:
            random.setstate(rng["python"])
        if "numpy" in rng:
            np.random.set_state(rng["numpy"])
        if "torch" in rng:
            torch.set_rng_state(_as_cpu_byte_tensor(rng["torch"]))
        if rng.get("torch_cuda") and torch.cuda.is_available():
            states = [_as_cpu_byte_tensor(s) for s in rng["torch_cuda"]]
            if len(states) == torch.cuda.device_count():
                torch.cuda.set_rng_state_all(states)

        if ck.get("config_hash") and ck["config_hash"] != self.config_hash:
            self._log({
                "event": "resume_config_mismatch",
                "checkpoint_hash": ck["config_hash"],
                "current_hash": self.config_hash,
                "differs": _config_diff(ck.get("config") or {}, asdict(self.cfg)),
            })
        self._log({"event": "resume", "step": self.global_step, "path": str(path)})

    def _load_optimizer_state(self, sd: dict) -> None:
        if not self.opt.state:
            self.opt.load_state_dict(sd)
            self._reinstall_lr_tensor()
            return
        saved = sd.get("state", {})
        with torch.no_grad():
            for i, p in enumerate(self.params):
                st = self.opt.state.get(p)
                src = saved.get(i, saved.get(str(i)))
                if not st or src is None:
                    continue
                for k, v in list(st.items()):
                    if k not in src:
                        continue
                    if torch.is_tensor(v):
                        v.copy_(torch.as_tensor(src[k], dtype=v.dtype, device=v.device))
                    else:
                        st[k] = src[k]
        self._reinstall_lr_tensor()

    def _reinstall_lr_tensor(self) -> None:
        """``Optimizer.load_state_dict`` restores param_groups wholesale, which
        can replace our LR tensor with whatever the checkpoint held.  Re-bind
        the single tensor this Trainer owns, or trap 1 comes back."""
        if self.lr_tensor is None:
            return
        for group in self.opt.param_groups:
            group["lr"] = self.lr_tensor

    # ------------------------------------------------------------------
    # logging
    # ------------------------------------------------------------------

    def _write_run_header(self) -> None:
        """config.json plus the one-time environment block required by the
        Phase-1 'Scientific Benchmarking Standard' (01_reconnaissance.md 10)."""
        _write_json(self.run_dir / "config.json", asdict(self.cfg))
        env = _environment_block()
        env.update({
            "event": "run_start",
            "run_name": self.cfg.run_name,
            "config_hash": self.config_hash,
            "seed": self.cfg.seed,
            "deterministic": self.cfg.deterministic,
            "params": self.n_params,
            "arch": self.cfg.arch,
            "fused": self.cfg.fused,
            "cuda_graph": self.cfg.cuda_graph,
        })
        _write_json(self.run_dir / "environment.json", env)
        self._log(env)

    def _log(self, record: dict) -> None:
        record.setdefault("wall_clock", time.time())
        with open(self.log_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, default=_json_default) + "\n")

    def _peak_vram(self) -> float:
        if not self.is_cuda:
            return 0.0
        return round(torch.cuda.max_memory_allocated() / 1024 ** 3, 4)


# ----------------------------------------------------------------------
# module helpers
# ----------------------------------------------------------------------

def _clip_grad_norm_fp64(parameters: list[torch.Tensor], max_norm: float) -> torch.Tensor:
    """L2 grad-clip, matching ``torch.nn.utils.clip_grad_norm_(..., foreach=True)``
    except the sum of squares accumulates in fp64.  Decision #7 (rev 6 of
    ``04_phase4_interim.md``): applies the first of the two one-line fixes
    ``EXP_009`` Sec 9.4 named and referred, choosing it over the
    abort-on-non-finite alternative because it repairs the clip's own blind
    spot rather than working around it.

    The bug this replaces: the stock foreach implementation squares each
    gradient *in its own dtype* before summing.  ``EXP_009`` Sec 9.4 hit a
    single gradient of 5.46e31 -- finite, unremarkable on its own -- which
    squares to 2.98e63, overflowing fp32's ~3.4e38 range by 25 orders of
    magnitude before the sum across parameters even runs.  The observed norm
    is therefore ``inf``, the branchless
    ``clamp(max_norm / (total_norm + 1e-6), max=1.0)`` evaluates to exactly
    0, and the update is *zeroed* rather than rescaled -- silently, and
    permanently from that step on, since a zeroed update cannot repair the
    gradient that caused it.

    The fix upcasts each gradient to fp64 *before* squaring (matching the
    exact computation ``009_chase_divergence.py`` used to diagnose the bug:
    ``vector_norm(g.double())`` per tensor).  2.98e63 is five orders of
    magnitude inside fp64's ~1.8e308 range, so the clip observes a large but
    finite norm and rescales the update instead of erasing it.

    **This is not numerically inert, and an earlier version of this docstring
    said it was.** It read "it does not change behaviour for any gradient that
    was already representable in fp32". That is true of the *returned norm* --
    the fp64 sum-of-squares and its fp32 downcast agree with the old
    computation to within ordinary rounding wherever the old computation was
    already finite -- and false of the *trajectory*.  On any step where the
    clip actually fires, the two sums-of-squares differ by ~1 ulp, so the
    clip coefficients differ in their last bits and the updates differ.
    ``EXP_014`` Sec 9.12 established this by substitution: on the committed
    Phase-2 baseline recipe the clip fires exactly three times in 20,000 steps
    (steps 32/33/34, norms 1.0329/1.1198/1.0957), the first differing loss
    appears at step 47, and the perturbation amplifies over the remaining
    ~19,950 steps to **+1.97e-03 bpc** of final test loss.

    ``tests/test_grad_clip.py`` already encodes the distinction and did not
    need changing.  ``test_fp64_clip_is_a_noop_below_max_norm`` asserts
    ``torch.equal``, but only on the ``max_norm=inf`` sentinel path, where no
    clip happens at all.  The test that does exercise a firing clip --
    ``test_fp64_clip_matches_stock_for_ordinary_gradients``, whose gradients
    exceed ``max_norm=1.0`` -- asserts ``allclose(rtol=1e-5)`` rather than
    equality, and is correct to.  No test ever asserted trajectory
    equivalence.  The docstring did.

    The consequence for the project, not for this function: a run trained on
    this tree cannot be compared bitwise against any figure committed before
    decision #7.  Every new experiment trains its own anchor on the current
    tree rather than reproducing a committed number.  Nothing here argues for
    reverting the fix -- it repairs a failure that silently zeroes every
    subsequent update.

    Capture-safety: every op here is a device kernel on a fixed-size input
    (no ``.item()``, no data-dependent branch).  ``p.grad is not None`` is a
    structural check, not a value-dependent one -- every parameter in
    ``self.params`` already has a non-None ``.grad`` by the time this first
    runs (``_step_body``'s own ``zero_grad(set_to_none=False)`` keeps
    gradient tensors permanently allocated once a backward pass has touched
    them, and warm-up runs three such passes before capture), so the branch
    resolves to the same outcome on every call, including the one call made
    during capture -- the same precedent the stock ``clip_grad_norm_`` this
    replaces already relied on internally.
    """
    grads = [p.grad for p in parameters if p.grad is not None]
    total_norm = torch.linalg.vector_norm(
        torch.stack([torch.linalg.vector_norm(g.double()) for g in grads])
    )
    clip_coef = (max_norm / (total_norm + 1e-6)).clamp(max=1.0).to(torch.float32)
    torch._foreach_mul_(grads, clip_coef)
    return total_norm.to(torch.float32)


def _default_run_name(cfg: Config) -> str:
    """Deterministic, so re-launching the same experiment resumes it instead of
    forking a second directory.  No timestamp, on purpose.

    The tag hashes the config with the two pure-bookkeeping fields cleared, so
    moving `out_dir` does not rename the run.  It is therefore *not* the same
    value as `Trainer.config_hash`, which is `config_hash` of the fully
    resolved config exactly as the spec defines it; both are logged.
    """
    tag = config_hash(replace(cfg, run_name="", out_dir=""))
    return (f"{cfg.arch}_{cfg.corpus}_d{cfg.d_model}_K{cfg.n_layers}"
            f"_L{cfg.seq_len}_b{cfg.batch_size}_s{cfg.seed}_{tag}")


def _environment_block() -> dict:
    info: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "cpu": platform.processor() or platform.machine(),
        "cpu_count": os.cpu_count(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "numpy": np.__version__,
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
    }
    if torch.cuda.is_available():
        idx = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(idx)
        info.update({
            "gpu": torch.cuda.get_device_name(idx),
            "vram_gib": round(props.total_memory / 1024 ** 3, 3),
            "compute_capability": f"sm_{props.major}{props.minor}",
            "multi_processor_count": props.multi_processor_count,
        })
    else:
        info["gpu"] = None
    return info


def _config_diff(a: dict, b: dict) -> dict:
    keys = set(a) | set(b)
    return {k: [a.get(k), b.get(k)] for k in sorted(keys) if a.get(k) != b.get(k)}


def _as_cpu_byte_tensor(x) -> torch.Tensor:
    t = x if torch.is_tensor(x) else torch.as_tensor(x)
    return t.detach().to(device="cpu", dtype=torch.uint8)


def _json_default(obj):
    if torch.is_tensor(obj):
        return obj.detach().cpu().tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


def _write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, default=_json_default)
