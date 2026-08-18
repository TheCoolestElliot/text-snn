"""Training loop for the chat model: CUDA-graph captured, and restartable.

RESTARTABILITY IS A HARD REQUIREMENT, NOT A FEATURE
---------------------------------------------------
This run is hours long and the process it runs under is not. `snn.train`'s
Trainer solves the same problem, and its two load-bearing decisions are
inherited here because they were expensive to get right:

* the learning rate lives in a **CUDA tensor** that the schedule writes into in
  place, because a Python float is baked into the graph at capture and the
  schedule then appears to work while the replayed kernels use the captured
  value (see `snn.train`'s module docstring, trap 1);
* warm-up before capture **mutates the model**, so parameters and optimiser
  moments are snapshotted and restored in place afterwards -- in place, because
  the graph has baked in their addresses (trap 2).

What is *not* inherited is the sampler, the corpus, the evaluation protocol and
the mid-run sampling, all of which are specific to a mixture-of-sources chat
corpus. Rather than parameterise `snn.train.Trainer` into something that serves
two masters -- and risk a change made for this run landing in a research one --
this is a separate loop that reuses the model and the ideas.

WHAT IT DOES NOT DO
-------------------
No loss masking. Every character in the window contributes to the loss,
including the user's turns and the role markers. Masking the user's turns is
standard practice for instruction tuning of large models and is the wrong call
here: at this scale the model needs every character of language signal it can
get, and it must learn to *predict* a plausible user turn in order to represent
where one ends -- which is what makes `<|eot|>` mean anything.

What it now DOES do, and what is different about it, is `bot_loss_weight`: the
reply's characters can be weighted more heavily than the prompt's without any
character being weighted zero. Masking discards the signal above; a weight
re-apportions it. At `bot_loss_weight = 1.0` the unweighted path below runs
unchanged, so every checkpoint trained before this existed still reproduces bit
for bit -- `tests/test_snnchat.py::test_unit_bot_weight_is_the_unweighted_loss`
holds that.
"""

from __future__ import annotations

import json
import math
import os
import random
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from snn.config import seed_everything
from snnchat.data import ChatCorpus, MixtureSampler, SequentialEval
from snnchat.model import ChatConfig, build_chat_model, describe_model
from snnchat.tokenizer import ChatTokenizer

__all__ = ["ChatTrainer"]

_M_LOSS = 0
_M_GNORM = 1
#: The plain, unweighted cross entropy, logged alongside the loss that is
#: actually optimised. With `bot_loss_weight > 1` the optimised loss is a
#: weighted mean and its bpc is NOT comparable with any earlier run's; this slot
#: is what keeps the training curve readable across the change.
_M_PLAIN = 2
_M_RATE0 = 3
_RATE_UNSET = -1.0
_WARMUP_STEPS = 3
_ADAM_EPS = 1e-8
_LOG2E = 1.4426950408889634


def bits_per_char(nats: float) -> float:
    return float(nats) * _LOG2E


class ChatTrainer:
    def __init__(self, cfg: ChatConfig) -> None:
        self.cfg = cfg
        seed_everything(cfg.seed, cfg.deterministic)
        self.device = torch.device(cfg.device)
        self.is_cuda = self.device.type == "cuda"
        if self.is_cuda and not torch.cuda.is_available():
            raise RuntimeError("cfg.device is cuda but torch.cuda.is_available() is False")

        # --- data ---------------------------------------------------------
        self.tok = ChatTokenizer()
        self.corpus = ChatCorpus(cfg.data_dir, vocab_version=cfg.vocab_version)
        if cfg.vocab_size == 0:
            cfg.vocab_size = self.corpus.vocab_size
        if cfg.vocab_size != self.corpus.vocab_size:
            raise ValueError(
                f"cfg.vocab_size={cfg.vocab_size} but the corpus was packed with "
                f"{self.corpus.vocab_size}"
            )
        if cfg.vocab_size != self.tok.vocab_size:
            raise ValueError(
                f"corpus vocab_size={cfg.vocab_size} disagrees with the tokenizer's "
                f"{self.tok.vocab_size}; one of them is from another version"
            )
        self.sampler = MixtureSampler(
            self.corpus, cfg.mix, cfg.batch_size, cfg.seq_len, cfg.seed,
            align_frac=cfg.align_frac, align_lookahead=cfg.align_lookahead,
        )
        self._bot_weight = float(cfg.bot_loss_weight)
        if self._bot_weight < 1.0:
            raise ValueError(
                f"bot_loss_weight={self._bot_weight} would weight the model's own "
                f"turns BELOW the prompt's, which is the opposite of the intent. "
                f"1.0 is the unweighted loss."
            )

        # --- run identity ---------------------------------------------------
        self.run_dir = Path(cfg.out_dir) / cfg.run_name
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.run_dir / "log.jsonl"

        # --- model / optimiser ----------------------------------------------
        self.model = build_chat_model(cfg)
        self.model.train()
        self.params = [p for p in self.model.parameters() if p.requires_grad]
        self.description = describe_model(self.model, cfg)
        self.n_params = self.description["params"]
        #: The timescale profile as BUILT, before any weight is loaded or trained.
        #: Kept because `_refresh_description` overwrites `slow_pole_tau` in place,
        #: and "what this run started from" and "what it ended with" are different
        #: questions -- a run that inherits a parent's spread and one that grew its
        #: own are indistinguishable from the final profile alone.
        self.slow_pole_tau_init = self.description.get("slow_pole_tau")

        self.lr_tensor: torch.Tensor | None = None
        lr_arg = float(cfg.lr)
        if self.is_cuda:
            self.lr_tensor = torch.tensor(float(cfg.lr), device=self.device,
                                          dtype=torch.float32)
            lr_arg = self.lr_tensor
        self.opt = torch.optim.AdamW(
            self._param_groups(), lr=lr_arg, betas=(cfg.beta1, cfg.beta2),
            eps=_ADAM_EPS, weight_decay=cfg.weight_decay,
            capturable=self.is_cuda, foreach=True,
        )

        # --- static buffers ---------------------------------------------------
        self.static_x = torch.zeros(cfg.batch_size, cfg.seq_len,
                                    dtype=torch.int64, device=self.device)
        self.static_y = torch.zeros(cfg.batch_size, cfg.seq_len,
                                    dtype=torch.int64, device=self.device)
        #: Filled per step when weighting is on; never read otherwise. Allocated
        #: unconditionally so the two paths differ in one branch rather than in
        #: what exists.
        self.static_w = torch.ones(cfg.batch_size, cfg.seq_len,
                                   dtype=torch.float32, device=self.device)
        self.static_metrics = torch.zeros(_M_RATE0 + cfg.n_layers,
                                          dtype=torch.float32, device=self.device)
        self.static_metrics[_M_RATE0:].fill_(_RATE_UNSET)

        self._vocab = int(cfg.vocab_size)
        self._max_norm = float(cfg.grad_clip) if cfg.grad_clip > 0 else float("inf")
        self._rate_slots = tuple(range(_M_RATE0, _M_RATE0 + cfg.n_layers))

        self.graph: torch.cuda.CUDAGraph | None = None
        self._ready = False
        self.global_step = 0
        self.best_val_bpc = float("inf")
        #: Divergence recovery, see `_recover_from_divergence`.
        self.lr_multiplier = 1.0
        self.n_recoveries = 0
        if self.is_cuda:
            torch.cuda.reset_peak_memory_stats()
        self._write_run_header()

    # ------------------------------------------------------------------
    # parameter groups
    # ------------------------------------------------------------------

    #: Per-channel neuron parameters. These are not weights -- they are the
    #: neuron's own time constants, its compartment mix and its threshold -- and
    #: shrinking them toward zero is not regularisation, it is a change of
    #: neuron.
    _NO_DECAY_PREFIXES = ("beta_s_raw.", "w.", "thr_log.")

    def _param_groups(self) -> list[dict]:
        """Split the parameters so weight decay does not eat the slow poles.

        WHY THIS EXISTS, MEASURED
        -------------------------
        `beta_s_raw` is the unconstrained parameter behind `beta_s = sigmoid(raw)`,
        the slow compartment's decay. AdamW's decoupled decay is `p -= lr*wd*p`,
        so at `wd = 0.1` it pulls `raw` toward **zero**, i.e. `beta_s` toward 0.5,
        i.e. a time constant of **2 characters** -- regardless of what the
        parameter was initialised to and regardless of what the data wants.

        The first chat run (`chat-v1`, one parameter group, exactly as
        `snn.train.Trainer` does it) was initialised with time constants spread
        log-uniformly from 3 to 600 characters. At step 20,000 its median was
        **2.25 to 3.09** and its maximum had fallen from 600 to 34-63. Its firing
        rates fell monotonically all run (layer 1: 0.255 -> 0.072) and its
        validation bpc stopped improving. The two-compartment neuron had been
        flattened into approximately a single-pole LIF, which is the one thing the
        architecture exists not to be.

        So the neuron's own parameters are excluded from decay. This is not a
        tuning preference: `beta_s_raw`, `w` and `thr_log` are per-channel
        constants of the neuron model, not a weight matrix whose norm needs
        controlling, and there are `3*K*d` of them against `K*d^2` weights.

        NOTE FOR THE RESEARCH SIDE -- NOT ACTED ON HERE. `snn.train.Trainer`
        deliberately uses a single parameter group ("the spec exposes exactly one
        `weight_decay`, so splitting biases/embeddings out would be an unlogged
        hyperparameter choice"), so every committed `twocomp` and
        `twocomp_threshold` run decays `beta_s_raw` too. Measured on the committed
        checkpoints, and reported in `docs/chat/WEIGHT_DECAY_NOTE.md`. Changing
        that is Elliot's call and needs a pre-registered candidate with a control
        arm; nothing here touches it.
        """
        decay, no_decay = [], []
        for name, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            if name.startswith(self._NO_DECAY_PREFIXES) or p.ndim < 2:
                no_decay.append(p)
            else:
                decay.append(p)
        self._n_no_decay = sum(p.numel() for p in no_decay)
        # `self.params` drives the gradient clip and must stay the flat list of
        # everything, in a stable order, or the clip silently covers a subset.
        self.params = decay + no_decay
        return [
            {"params": decay, "weight_decay": self.cfg.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]

    # ------------------------------------------------------------------
    # the captured region
    # ------------------------------------------------------------------

    def _step_body(self) -> None:
        self.opt.zero_grad(set_to_none=False)
        logits, _, aux = self.model(self.static_x, None)
        flat = logits.reshape(-1, self._vocab).float()
        target = self.static_y.reshape(-1)
        if self._bot_weight == 1.0:
            loss = F.cross_entropy(flat, target)
            plain = loss.detach()
        else:
            # `reduction="none"` then a weighted mean. Normalising by the sum of
            # the weights rather than by their count keeps the loss on the scale
            # of a cross entropy, so the learning rate does not silently change
            # meaning when the weight does.
            per_char = F.cross_entropy(flat, target, reduction="none")
            w = self.static_w.reshape(-1)
            loss = (per_char * w).sum() / w.sum()
            plain = per_char.detach().mean()
        loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(
            self.params, self._max_norm, norm_type=2.0,
            error_if_nonfinite=False, foreach=True,
        )
        self.opt.step()
        self.static_metrics[_M_LOSS].copy_(loss.detach())
        self.static_metrics[_M_GNORM].copy_(gnorm.detach())
        self.static_metrics[_M_PLAIN].copy_(plain)
        for slot, rate in zip(self._rate_slots, aux.get("firing_rate", ())):
            self.static_metrics[slot].copy_(rate.detach().reshape(()))

    def _ensure_ready(self) -> None:
        if self._ready:
            return
        self._ready = True
        if self.cfg.cuda_graph and self.is_cuda:
            self._capture()

    def _capture(self) -> None:
        snapshot = self._snapshot()
        self._set_lr(self._lr_at(self.global_step))
        keep = []
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
        except Exception as exc:
            raise RuntimeError(
                "CUDA-graph capture failed. The documented fallback is "
                "cuda_graph=False, which runs the identical step eagerly and "
                "much slower. Find what synchronised inside _step_body."
            ) from exc
        finally:
            self._restore(snapshot)
        print(f"  graph captured in {time.perf_counter() - t0:.2f}s", flush=True)

    def _snapshot(self):
        params = [p.detach().clone() for p in self.params]
        state = []
        for p in self.params:
            st = self.opt.state.get(p, {})
            state.append({k: (v.detach().clone() if torch.is_tensor(v) else v)
                          for k, v in st.items()})
        return params, state

    def _restore(self, snapshot) -> None:
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
    # schedule
    # ------------------------------------------------------------------

    def _lr_at(self, step: int) -> float:
        """Warm-up, then cosine to `lr_final_frac * lr` rather than to zero.

        Decaying to exactly zero is right for a run whose step count is the end
        of the experiment. This one may be stopped early -- the budget is wall
        clock, not steps -- and a run cut off at 70 % of a decay-to-zero schedule
        is at a worse point than a constant-LR run would be. A floor keeps the
        tail useful without giving up the annealing.
        """
        cfg = self.cfg
        scale = float(cfg.lr) * self.lr_multiplier
        if cfg.warmup_steps > 0 and step < cfg.warmup_steps:
            return scale * (step + 1) / float(cfg.warmup_steps)
        if cfg.lr_schedule == "constant":
            return scale
        span = max(1, cfg.max_steps - cfg.warmup_steps)
        prog = min(max((step - cfg.warmup_steps) / span, 0.0), 1.0)
        floor = float(cfg.lr_final_frac)
        return scale * (floor + (1.0 - floor) * 0.5 * (1.0 + math.cos(math.pi * prog)))

    def _set_lr(self, lr: float) -> None:
        for group in self.opt.param_groups:
            cur = group.get("lr")
            if torch.is_tensor(cur):
                cur.fill_(float(lr))
            else:
                group["lr"] = float(lr)

    # ------------------------------------------------------------------
    # stepping
    # ------------------------------------------------------------------

    def _fill_inputs(self, step: int, non_blocking: bool = True):
        if self._bot_weight == 1.0:
            x, y = self.sampler.batch(step)
            w = None
        else:
            x, y, w_np = self.sampler.batch_weighted(step, self._bot_weight)
            w = torch.from_numpy(w_np)
        self.static_x.copy_(x, non_blocking=non_blocking)
        self.static_y.copy_(y, non_blocking=non_blocking)
        if w is not None:
            self.static_w.copy_(w, non_blocking=non_blocking)
        # Returned so the caller can hold a reference until the async copy has
        # landed; `snn.data._to_int64_cpu`'s contract about page-locked host
        # memory and in-flight copies applies to the weights too.
        return x, y, w

    def step(self) -> dict:
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
        del batch
        self.global_step += 1
        return {
            "step": step_index,
            "loss": m[_M_LOSS],
            "bpc": bits_per_char(m[_M_LOSS]),
            "bpc_plain": bits_per_char(m[_M_PLAIN]),
            "grad_norm": m[_M_GNORM],
            "lr": lr,
            "firing_rate": [r for r in m[_M_RATE0:] if r != _RATE_UNSET],
        }

    # ------------------------------------------------------------------
    # evaluation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def evaluate(self) -> dict:
        """Fresh-state bpc on each source's held-out tail, plus a weighted mean.

        Per source rather than pooled, because the sources have wildly different
        entropies -- persona is near-memorised template text and TinyStories is
        simple narrative, so a single pooled number moves when the *mix* moves
        and looks like a modelling change. The weighted mean is reported too,
        under the mix actually being trained on, and is the number that goes in
        `best_val_bpc`.
        """
        was_training = self.model.training
        self.model.eval()
        out: dict[str, float] = {}
        cfg = self.cfg
        try:
            eval_bs = cfg.eval_batch_size or cfg.batch_size
            for name in self.sampler.names:
                try:
                    it = SequentialEval(self.corpus, name, eval_bs, cfg.seq_len)
                except FileNotFoundError:
                    continue
                if it.n_batches() == 0:
                    # A source whose held-out tail is smaller than one evaluation
                    # batch scores zero characters and vanishes from the report.
                    # `dolly` (7 KB) and `oasst1` (4 KB) both do at B=48, L=256 --
                    # and silently: the weighted mean renormalises over whatever
                    # is present, so the number stays plausible while two sources
                    # are missing from it. Shrink the batch to whatever fits
                    # rather than dropping the source.
                    windows = max(len(it.data) - 1, 0) // cfg.seq_len
                    if windows == 0:
                        continue
                    it = SequentialEval(self.corpus, name, windows, cfg.seq_len)
                total_nats, total_chars = 0.0, 0
                for i, (x, y) in enumerate(it):
                    if i >= cfg.eval_batches:
                        break
                    x = x.to(self.device, non_blocking=True)
                    y = y.to(self.device, non_blocking=True)
                    logits, _, _ = self.model(x, None)
                    loss = F.cross_entropy(
                        logits.reshape(-1, self._vocab).float(), y.reshape(-1),
                        reduction="sum",
                    )
                    total_nats += float(loss)
                    total_chars += y.numel()
                if total_chars:
                    out[name] = bits_per_char(total_nats / total_chars)
            if out:
                num = sum(out[n] * w for n, w in zip(self.sampler.names, self.sampler.weights)
                          if n in out)
                den = sum(w for n, w in zip(self.sampler.names, self.sampler.weights)
                          if n in out)
                out["weighted"] = num / max(den, 1e-9)
        finally:
            if was_training:
                self.model.train()
        return out

    # ------------------------------------------------------------------
    # checkpoints
    # ------------------------------------------------------------------

    def save_checkpoint(self, path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Re-read the timescale profile off the weights being written. Without
        # this the checkpoint carries the profile from construction or from the
        # last load -- see `_refresh_description`, which documents the 23
        # committed artifacts that got this wrong.
        self._refresh_description()
        ck = {
            "format": 1,
            "kind": "snnchat",
            "step": self.global_step,
            "model": self.model.state_dict(),
            "optimizer": self.opt.state_dict(),
            "chat_config": asdict(self.cfg),
            "vocab_version": self.cfg.vocab_version,
            "best_val_bpc": self.best_val_bpc,
            "description": self.description,
            "rng": {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch": torch.get_rng_state(),
            },
            "torch_version": torch.__version__,
        }
        tmp = path.with_suffix(path.suffix + ".tmp")
        torch.save(ck, tmp)
        os.replace(tmp, path)

    def load_checkpoint(self, path) -> None:
        """Restore in place wherever a captured graph might already hold pointers.

        `Optimizer.load_state_dict` *rebinds* the moment tensors to the objects
        unpickled from disk, which after capture silently orphans the graph from
        the state it is stepping -- so once optimiser state exists we copy into
        it instead. `snn.train.Trainer.load_checkpoint` documents the same trap.
        """
        ck = torch.load(Path(path), map_location=self.device, weights_only=False)
        if ck.get("vocab_version") != self.cfg.vocab_version:
            raise RuntimeError(
                f"checkpoint was trained under vocab v{ck.get('vocab_version')}, "
                f"this code is v{self.cfg.vocab_version}"
            )
        self.model.load_state_dict(ck["model"])
        self._load_optimizer_state(ck["optimizer"])
        self.global_step = int(ck["step"])
        self.best_val_bpc = float(ck.get("best_val_bpc", float("inf")))
        self._refresh_description()
        self._log({"event": "resume", "step": self.global_step, "path": str(path)})
        print(f"  resumed from {path} at step {self.global_step}", flush=True)

    def init_from(self, path) -> None:
        """Load another run's WEIGHTS only -- no optimiser, no step, no LR state.

        This is how the chat-focused anneal starts: a fresh run, at step 0, with
        its own short schedule and its own `ckpt_best.pt`, initialised from the
        main run's weights.

        The alternative -- resuming the main run with a different `--mix` -- is
        one line shorter and quietly wrong. `best_val_bpc` is a weighted mean
        over the mixture, so changing the mixture changes what the number means;
        the anneal upweights the harder sources, the weighted bpc rises, no new
        `ckpt_best.pt` is ever written, and `scripts/chat.py` then loads the
        *pre-anneal* model while every artifact says the anneal ran. Separate
        runs keep the comparison inside each run, where it is valid.
        """
        ck = torch.load(Path(path), map_location=self.device, weights_only=False)
        if ck.get("vocab_version") != self.cfg.vocab_version:
            raise RuntimeError(
                f"{path} was trained under vocab v{ck.get('vocab_version')}, "
                f"this run is v{self.cfg.vocab_version}"
            )
        self.model.load_state_dict(ck["model"])
        self._refresh_description()
        self._log({"event": "init_from", "path": str(path),
                   "source_step": ck.get("step"),
                   "source_best_val_bpc": ck.get("best_val_bpc")})
        print(f"  initialised weights from {path} (step {ck.get('step')}); "
              f"optimiser and step start fresh", flush=True)

    def _refresh_description(self) -> None:
        """Re-read the model's description after its weights changed.

        `describe_model` reads the slow-pole time constants **off the parameter**
        rather than recomputing them from `cfg.tau_min`/`tau_max`, precisely so a
        run whose spread silently failed to apply reports what it actually has.
        That only works if it is re-read after a load: computed once at
        construction, every resumed or `--init-from` checkpoint would carry the
        *initial* spread in its `description` while its weights held a trained
        one, and the artifact would describe a model that never existed.

        THE SAME TRAP, ONE STEP LATER -- fixed 2026-08-17. Until then this was
        called only from `resume` and `init_from`, i.e. only when weights were
        *loaded*, and never after they were *trained*. So every `summary.json`
        and every checkpoint reported the profile at step 0. Measured on the
        committed tree: 23 chat summaries report their shared parent's step-7,874
        profile as their own -- identical to two decimals across four training
        seeds and three corpora, which is the tell -- and `chat-v6-scratch`
        reports `spread_slow_poles`' pristine 42.32 in all four layers after
        84,000 steps of training. `save_checkpoint` and the end-of-run summary
        now call this first, so an artifact describes the weights it ships.

        `slow_pole_tau_init` is carried alongside rather than overwritten:
        "what it started from" and "what it learned" are both wanted, and the
        final profile alone cannot separate an inherited spread from a grown one.
        """
        self.description = describe_model(self.model, self.cfg)
        self.n_params = self.description["params"]
        if getattr(self, "slow_pole_tau_init", None) is not None:
            self.description["slow_pole_tau_init"] = self.slow_pole_tau_init

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
        if self.lr_tensor is None:
            return
        for group in self.opt.param_groups:
            group["lr"] = self.lr_tensor

    # ------------------------------------------------------------------
    # the loop
    # ------------------------------------------------------------------

    def train(self, *, wall_clock_budget_s: float = 0.0) -> None:
        cfg = self.cfg
        self._ensure_ready()
        start_wall = time.perf_counter()
        start_step = self.global_step
        try:
            while self.global_step < cfg.max_steps:
                m = self.step()
                s = m["step"]
                if not math.isfinite(m["loss"]):
                    self._recover_from_divergence(s, m["loss"])
                    continue
                if cfg.log_every > 0 and s % cfg.log_every == 0:
                    elapsed = time.perf_counter() - start_wall
                    done = max(1, self.global_step - start_step)
                    rec = dict(m)
                    rec.update({
                        "event": "train",
                        "steps_completed": self.global_step,
                        "chars": self.global_step * cfg.batch_size * cfg.seq_len,
                        "elapsed_s": round(elapsed, 2),
                        "steps_per_s": round(done / max(elapsed, 1e-9), 3),
                        "peak_vram_gib": self._peak_vram(),
                    })
                    self._log(rec)
                    rates = ", ".join(f"{r:.3f}" for r in m["firing_rate"])
                    eta = ""
                    if done > 20:
                        rem = (cfg.max_steps - self.global_step) / (done / max(elapsed, 1e-9))
                        eta = f"  eta {rem / 60:,.0f}m"
                        if wall_clock_budget_s:
                            eta += f"  budget {(wall_clock_budget_s - elapsed) / 60:,.0f}m"
                    weighted = ("" if self._bot_weight == 1.0
                                else f" (plain {m['bpc_plain']:.4f})")
                    print(
                        f"step {s:>7d}  loss {m['loss']:.4f}  bpc {m['bpc']:.4f}"
                        f"{weighted}  lr {m['lr']:.2e}  |g| {m['grad_norm']:.2f}  "
                        f"rate [{rates}]{eta}",
                        flush=True,
                    )
                if cfg.ckpt_every > 0 and self.global_step % cfg.ckpt_every == 0:
                    self.save_checkpoint(self.run_dir / "ckpt_last.pt")
                if cfg.eval_every > 0 and self.global_step % cfg.eval_every == 0:
                    self._eval_and_report()
                if cfg.sample_every > 0 and self.global_step % cfg.sample_every == 0:
                    self._sample_report()
                if wall_clock_budget_s and (time.perf_counter() - start_wall) > wall_clock_budget_s:
                    print(f"  wall-clock budget reached at step {self.global_step}", flush=True)
                    break
        except KeyboardInterrupt:
            self.save_checkpoint(self.run_dir / "ckpt_last.pt")
            raise
        self.save_checkpoint(self.run_dir / "ckpt_last.pt")
        final = self._eval_and_report()
        self._refresh_description()   # the trained profile, not step 0's
        summary = {
            "run_name": cfg.run_name,
            "steps": self.global_step,
            "params": self.n_params,
            "wall_clock_s": round(time.perf_counter() - start_wall, 2),
            "peak_vram_gib": self._peak_vram(),
            "val": final,
            "description": self.description,
        }
        with open(self.run_dir / "summary.json", "w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2, default=str)
        self._log({"event": "run_end", **summary})

    # ------------------------------------------------------------------
    # divergence
    # ------------------------------------------------------------------

    _MAX_RECOVERIES = 5

    def _recover_from_divergence(self, step: int, loss: float) -> None:
        """Roll back to the last checkpoint, change the data order, drop the LR.

        Phase 4 hit two unexplained divergences in this neuron family --
        `EXP_009`'s seed died to an fp32 overflow inside the gradient clip, and
        `compose_s0`'s died at step 17,598 with gradients of 0.03 and everything
        finite. Neither has a known fix. A run that is going to be left alone for
        hours therefore needs to survive one rather than discover it in the
        morning.

        **This is a recovery, not a diagnosis, and it is only defensible because
        this is not an experiment.** On the research side the correct response to
        a NaN is to chase it to its origin -- `snn-v2-research-protocol` is
        explicit that "one seed diverged" is a tolerance widened in prose. Here
        there is no hypothesis to protect and the deliverable is a working model,
        so the run rolls back and carries on, loudly, with the event in
        `log.jsonl`.

        Three things change on the way back, and each is logged:

        * the parameters return to `ckpt_last.pt`, which predates the divergence;
        * the sampler seed is bumped, so the step that killed it draws *different
          data* -- without this the restart replays the identical batch, because
          `batch(step)` is a pure function of `(seed, step)`, and diverges again
          at exactly the same place;
        * the learning rate is halved for the remainder.
        """
        self.n_recoveries += 1
        last = self.run_dir / "ckpt_last.pt"
        record = {
            "event": "divergence",
            "step": step,
            "loss": loss,
            "recovery": self.n_recoveries,
            "checkpoint": str(last),
        }
        if self.n_recoveries > self._MAX_RECOVERIES or not last.exists():
            self._log({**record, "action": "abort"})
            raise RuntimeError(
                f"loss became {loss!r} at step {step} and "
                + ("no ckpt_last.pt exists to roll back to"
                   if not last.exists()
                   else f"the run has already recovered {self._MAX_RECOVERIES} times")
                + ". Stopping rather than grinding through a broken run."
            )
        old_seed = self.sampler.seed
        self.load_checkpoint(last)
        self.sampler.seed = old_seed + 1
        self.lr_multiplier *= 0.5
        record.update({
            "action": "rollback",
            "resumed_at": self.global_step,
            "sampler_seed": self.sampler.seed,
            "lr_multiplier": self.lr_multiplier,
        })
        self._log(record)
        print(
            f"  !! loss {loss!r} at step {step}: rolled back to step "
            f"{self.global_step}, sampler seed -> {self.sampler.seed}, "
            f"lr x{self.lr_multiplier:g}",
            flush=True,
        )

    def _eval_and_report(self) -> dict:
        t0 = time.perf_counter()
        res = self.evaluate()
        self._log({"event": "eval", "step": self.global_step,
                   "seconds": round(time.perf_counter() - t0, 1), **res})
        parts = "  ".join(f"{k} {v:.4f}" for k, v in sorted(res.items()))
        print(f"  eval step {self.global_step:>7d}  {parts}", flush=True)
        w = res.get("weighted")
        if w is not None and w < self.best_val_bpc:
            self.best_val_bpc = w
            self.save_checkpoint(self.run_dir / "ckpt_best.pt")
        return res

    @torch.no_grad()
    def _sample_report(self) -> None:
        """Two short samples, printed. Cheap, and the only honest progress signal.

        bpc falling says the model is fitting the corpus; it does not say whether
        the reply to "hello" is a greeting or the middle of a story. Printing a
        sample every few thousand steps is how a broken format shows up in
        minutes rather than at the end of the run.
        """
        from snnchat.generate import ChatSampler

        sampler = ChatSampler(self.model, self.tok, device=self.device)
        was_training = self.model.training
        self.model.eval()
        try:
            for prompt in ("hello", "tell me a story about a dog"):
                reply = sampler.reply([("user", prompt)], max_new=180,
                                      temperature=0.8, top_p=0.92, seed=self.global_step)
                print(f"    [{prompt!r}] -> {reply[:200]!r}", flush=True)
                self._log({"event": "sample", "step": self.global_step,
                           "prompt": prompt, "reply": reply})
        except Exception as exc:  # a broken sampler must not kill a 3-hour run
            print(f"    sampling failed: {type(exc).__name__}: {exc}", flush=True)
        finally:
            if was_training:
                self.model.train()

    # ------------------------------------------------------------------
    # bookkeeping
    # ------------------------------------------------------------------

    def _write_run_header(self) -> None:
        with open(self.run_dir / "config.json", "w", encoding="utf-8") as fh:
            json.dump(asdict(self.cfg), fh, indent=2, default=str)
        header = {
            "event": "run_start",
            "run_name": self.cfg.run_name,
            "mix_requested": self.cfg.mix,
            "mix_realised": dict(zip(self.sampler.names, self.sampler.weights)),
            "corpus_chars": self.corpus.train_chars(),
            "bot_loss_weight": self._bot_weight,
            "align_frac": self.cfg.align_frac,
            "weight_decay": self.cfg.weight_decay,
            "params_exempt_from_decay": self._n_no_decay,
            "torch": torch.__version__,
            **self.description,
        }
        if torch.cuda.is_available():
            header["gpu"] = torch.cuda.get_device_name(0)
        self._log(header)
        print(
            f"  {self.n_params:,} params, d={self.cfg.d_model}, K={self.cfg.n_layers}, "
            f"V={self.cfg.vocab_size}\n"
            f"  weight decay {self.cfg.weight_decay} on all but {self._n_no_decay:,} "
            f"neuron parameters (beta_s_raw, w, thr_log, biases)\n"
            f"  bot loss weight {self._bot_weight:g}, "
            f"{self.cfg.align_frac:.0%} of windows aligned to a conversation start\n"
            f"  mix: {self.sampler.mix_report()}",
            flush=True,
        )
        for entry in self.description.get("slow_pole_tau", [])[:1]:
            print(
                f"  slow-pole tau: {entry['tau_min']:.1f} .. {entry['tau_max']:.1f} "
                f"(median {entry['tau_median']:.1f}) characters",
                flush=True,
            )

    def _log(self, record: dict) -> None:
        record.setdefault("wall_clock", time.time())
        with open(self.log_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, default=str) + "\n")

    def _peak_vram(self) -> float:
        if not self.is_cuda:
            return 0.0
        return round(torch.cuda.max_memory_allocated() / 1024 ** 3, 3)
