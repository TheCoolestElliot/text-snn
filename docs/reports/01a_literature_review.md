# Phase 1 — Literature & Domain Context (full survey)

**Companion to** [`01_reconnaissance.md`](01_reconnaissance.md) **§4**, which condenses this
document. Everything here is retained in full because the audit trail is the point.

## Provenance

Produced 2026-08-01 by a six-way parallel survey (spiking LMs; surrogate gradients;
neuron models; efficient sequence modelling; character-LM baselines and metrics; SNN
systems/efficiency), where each survey was then audited by an independent adversarial
fact-checker instructed to default to skepticism and verify every named paper and
number against primary sources. 13 agents, ~694 tool calls. The audit trail of
retracted, corrected and flagged claims is preserved at the end of this document and
is as important as the findings.

## Independent verification by the report author

Three systems claims in §6 contradicted or materially changed the reconnaissance
audit'''s own measurements, so they were re-tested from scratch rather than accepted
on report (`scripts/audit/05_verify_literature_claims.py`):

| Claim | Verdict |
|---|---|
| §6.1 `jiterator` compiles elementwise CUDA via torch-bundled NVRTC | **CONFIRMED** — and it overturns this project'''s prior "fusion is unavailable" conclusion |
| §6.7 reset-free leak = exact triangular-Toeplitz matmul | **CONFIRMED**, and larger than reported (77–160×) |
| §6.8 bf16 spacing at threshold is 8× coarser than fp16 | **Spacing CONFIRMED exactly; the behavioural consequence did NOT reproduce** |

One claim is **not** adopted: §6.3/§6.4 conclude "expect ~1.1–1.2× from CUDA graphs at
target scale, not 10×". That figure comes from a benchmark model which the fact-checker
itself determined "cannot be scanning the sequence dimension" and "has no cross-token
recurrence". On a genuinely sequential, char-LM-shaped model the reconnaissance audit
measured **10.9–13.1×** (`01_reconnaissance.md` §3.8). The underlying caution is still
sound and is recorded: as GEMM work grows relative to launch overhead, the graph
speed-up shrinks.

---

*Every number below has been through an adversarial fact-check against primary sources. Claims the checker marked FABRICATED or UNSUPPORTED have been removed, and the removals are itemised at the end. Corrected values are used silently in the body; where a correction changes the *direction* of an inference, it is called out. Anything I could not source is tagged `(unverified)`.*

---

## 1. Spiking language models: what has actually been measured

### 1.1 The one directly comparable data point

**SpikeGPT** (Zhu, Zhao, Li, Eshraghian; arXiv:2302.13939) is the only spiking LM with a published enwik8 bits-per-character number. Its Table 2, at 46.1M parameters (the abstract says 45M; the table says 46.1M):

- SpikeGPT: **1.283 bpc** at context 1024, **1.262 bpc** at context 3072 (train 1.113 and 0.903 respectively — note the large train/test gap, i.e. it overfit enwik8's 90 MB).
- Same-table baselines, all at context 1024: vanilla Transformer 43.0M **1.137**, Reformer 40.1M 1.195, Performer 43.0M 1.199, Linear Transformer 43.0M 1.207, Synthesizer 42.8M 1.298, 4L SHA-LSTM 1.330, 7L stacked LSTM 1.670.

**The equal-context spiking penalty is 0.146 bpc, not the 0.125 figure that gets quoted** — 1.262 is measured at 3× the context of every baseline and is not an equal-protocol comparison. Two further framing corrections: SpikeGPT *does* beat both LSTM baselines in that table, so "it only beats the weakest linear-attention model" is false; and the assertion that it is the *only* published spiking char-LM bpc could not be positively verified — treat "only" as `(unverified)`.

> **Citation hygiene.** An earlier draft of this survey listed a SpikeGPT author named "Blum". **No such author exists.** The list is Rui-Jie Zhu, Qihang Zhao, Guoqi Li, Jason K. Eshraghian. The same draft gave train bpc 0.864 at ctx 3072; the correct value is **0.903**. Venue is contested: one check found an ICLR 2025 poster page, another found the arXiv comments field reading "Accepted by TMLR". **Do not assert a venue** for SpikeGPT without re-checking.

**Compute anchor, and it matters.** SpikeGPT-46M was trained for **12 wall-clock hours on four V100s ≈ 48 GPU-hours** — i.e. roughly this entire project's 50–60 GPU-hour budget for a single 46M-param run. Matching SpikeGPT is not a budget question we win by being clever; it is out of reach, and the target must be set elsewhere (§5, §7).

### 1.2 What SpikeGPT's configuration validates

SpikeGPT's architecture is a near-exact match for this project's hard invariants, which is the single most useful fact in the survey:

- **T = 1 per token.** Its "time" axis *is* the token sequence; one LIF update per token index. No micro-timesteps.
- **No lateral recurrent weight matrix.** Recurrence is RWKV-style element-wise exponential decay (learnable vectors `Wf`, `Wd`) plus token-shift.
- **β = 0.5, threshold = 1, hard reset** (`H[t] = U[t]·(1 − S[t])`), **arctangent surrogate**.
- 12 layers, d = 512, Adam, lr 6e-4, 500-step warmup.
- Reported mean firing rate **0.15**, measured on real test data.

This is the strongest available evidence that T=1 direct coding is not a compromise but the setting under which the best published spiking-LM result was obtained. It cross-references directly with §2's gradient-horizon arithmetic (T multiplies the unroll and shortens the per-character memory horizon simultaneously) and with §6's launch-latency wall (T multiplies kernel launches).

**Readout correction.** A claim circulated that SpikeGPT "passes membrane potential through softmax". It does not. SpikeGPT computes `softmax(Y'[t] Wᵉᵀ)` where `Y'[t]` is the **real-valued pre-spike output of the SRFFN block** — the readout bypasses the spiking neuron entirely. The design *recommendation* (non-spiking readout) survives; the mechanism was misdescribed.

### 1.3 Element-wise-decay state beats RWKV per parameter

**SpikingSSMs** (Shen et al., AAAI 2025, 39(19):20380–20388; arXiv:2408.14909) reaches **33.94** test perplexity on WikiText-103 at **75M** parameters, against SpikeGPT's **39.75 at 213M** and non-spiking Transformer 231M **20.51** / S4 249M **20.95**. On LRA it averages **84.33%** (ListOps 60.23 / Text 80.41 / Retrieval 88.77 / Image 88.21 / Pathfinder 93.51 / Path-X 94.82) at mean spiking rate 0.098, versus non-spiking S4D-Lin 85.52%. On WikiText-103 its average spiking rate is **under 30%**, not 10% — the 0.10 figure is the LRA number.

Its enabling trick is a **Surrogate Dynamic Network**: a 3-layer 1-D conv with **fewer than 200 parameters** that predicts post-reset membrane potential so the LIF recurrence can be evaluated in parallel. Speedup over the sequential scan **scales with sequence length**: 7.5× at 1K, 15.0× at 2K, 40.2× at 4K, **101.2× at 8K**. This scaling law is the key cross-reference to §4 and §6 — at this project's ~64–256 step chunks, the analogous gain is ~1–3×, not 100×.

Two structural lessons: (a) a diagonal/element-wise-decay state is markedly more parameter-efficient than RWKV for LM at equal spiking constraints; (b) approximation-based parallelisation is **explicitly capped** — "the best possible performance cannot exceed that of the approximated neuron."

### 1.4 What binarity costs, and why the biggest published number is not the answer

**SpikeLM** (Xing et al., ICML 2024, PMLR 235:54698–54714; arXiv:2406.03287) is the cleanest quantification of the binary-spike penalty in language: GLUE average BERT-base **83.2**, SpikeLM **76.5** (T=4), SpikeBERT **59.7**, naive LIF-BERT **54.9**. But **SpikeLM achieves this by abandoning binary spikes** — its "elastic bi-spiking" is bi-directional (signed) with elastic amplitude and frequency, i.e. ternary/multi-level. Firing rate is explicitly controlled: ~33% at k=2 (GLUE 76.5), ~24% at k=3 (75.1), ~17% at k=4 (**60.8** — the sparsity costs 15.7 GLUE points). Generative: XSUM ROUGE-L 32.92 vs BART-base 34.72; WMT16 BLEU 22.95 vs **mBART-large** 26.82 (not BART-base).

> **Do not quote the "+41.9 / +41.8 point" figure.** It is verbatim in the paper, but it is measured against `LIF-BERT*` (GLUE 34.6) and `PSN-BERT*` (34.7) — near-degenerate collapsed baselines. It is not 42 points of binary-spike headroom. The defensible number is the **54.9 → 76.5 gap**, and most of that recovery comes from relaxing binarity, which this project has ruled out by invariant.

**Ternary Spike** (Guo et al., AAAI 2024, 38:12244–12252; arXiv:2312.06372) makes the same move in vision (+1/0/−1, 95.60/94.46 on CIFAR-10 with ResNet19/20). **Both are ruled out here on invariant grounds**, and both are the field's clearest signal that binary inter-layer signalling has a real, non-trivial cost that must be budgeted rather than tuned away.

### 1.5 Every strong spiking LM uses a teacher; this project will not

**SpikeBERT** (arXiv:2308.15122; 109.0M, 12 blocks, T=4) uses two-stage KD from BERT: MR 80.69 / SST-2 85.39 / SST-5 46.11 / Subj 93.00 / ChnSenti 86.36 / Waimai 89.66, an average **4.13 points** below fine-tuned BERT. Its timestep ablation peaks at **T=4** with degradation at T=8 and T=12, attributed to "excessive time steps introducing too much noise in spike trains." **SpikingBERT** (Bal & Sengupta, AAAI 2024; arXiv:2308.10873; ~50M, 4 blocks) avoids BPTT entirely via implicit differentiation at the steady state of average spiking rates with T_conv = 80: SST-2 88.19 / QQP 86.82 / MNLI-m 78.10 / QNLI 85.20 / RTE 66.06.

Consequence: **T=4 is the empirical ceiling for spiking encoders and a T sweep can stop there**, and a from-scratch surrogate-BPTT model with no teacher is playing a harder game than anything in this list. That should be stated in the writeup, not hidden.

### 1.6 The integer-training escape hatch

The current best spiking vision models **train with integer activations and unfold to binary spike trains only at inference**. **E-SpikeFormer / Spike Firing Approximation** (Yao et al., IEEE T-PAMI 2025; arXiv:2411.16061) proves the equivalence: `floor(clip(U, 0, D))` equals the sum of D binary spikes from an IF neuron with soft reset. ImageNet top-1 at 224×224, "1×4" (train 1 step, infer 4): **10M 78.5, 19M 79.8, 83M 83.2** (84.0 at 1×8), **173M 84.7** (85.1 at 1×8). Reported 4.5× training acceleration and 3.9× inference-energy improvement for the 10M model.

> **Trap:** the 86.2% figure sometimes attached to the 173M model is a **384×384 enlarged-crop inference result**, not a 1×8 result at 224×224.

**ASN/NASN** (Zhou et al., arXiv:2604.12365, Apr 2026) applies the same paradigm with a learnable shift — `clip(round(U[t]), α, α+D)`, unfolded to T×D binary timesteps at inference — reaching 75.53% ImageNet and 67.5% GLUE average.

This is **arguably legal under the invariants**: inference emits binary spikes between layers with a hard threshold and leaky membrane; only the training-time forward pass is integer. It is honest only if we actually perform the unfolding and verify equivalence, and describe it as a training-time reparameterisation. It is the highest-value untested lever available.

### 1.7 The cautionary small-scale point

**SpikeDecoder** (Beger, Walter, Knoll, TU Munich; arXiv:2606.12287v1, June 2026), ~1.4M params on a 100K-character War-and-Peace corpus. Three corrections to how this is usually cited:

1. **These are training-set accuracies**, not test — the paper says "prediction accuracy relative to the source text training set." It is not a generalisation gap.
2. The honest gap is **11.5 points** (Spike-Degree-4 with LayerNorm, 87.0%) or **16.8 points** (PowerNorm + concatenate, the fully float-MAC-free variant, 81.7%) against the 98.5% ANN — not a single "~17 points".
3. The ANN twin is **not identical**: the fully spiking config drops from 8 heads / 80-dim to **1 head / 50-dim**.

Firing rates 10–36%; energy 86.7% (T=4) / 92.3% (T=2) reduction at 45nm, no hardware validation. The lesson survives all three corrections: **do not "just put a LIF after every layer."**

### 1.8 The activity-sparse upper bound

**EGRU** (Subramoney, Khan Nazeer, Schöne, Mayr, Kappel; ICLR 2023, arXiv:2206.06178): PTB word-level perplexity **57.0** at 84.8% activity sparsity (2000 hidden) and 57.2 at 79.7% (1350), vs dense GRU-1350 66.3 and AWD-LSTM 57.3; WikiText-2 68.9 / 70.6. Backward-pass sparsity 42–46%. sMNIST 98.3% at 72.1% sparsity.

**But EGRU emits graded events** (the unit's real value at threshold crossing), not binary spikes, and being a GRU variant it has a full lateral recurrent matrix. Both of its enabling ingredients are banned here. That makes it the right **upper bound**: EGRU-level quality is what binary + no-lateral-W costs, and the gap is worth quantifying rather than assuming.

### 1.9 Energy: the honest anchor

**Every** "energy saving" number in the spiking-LM literature — SpikeGPT (20× in the abstract, 32.2× in its table), SpikeBERT (27.82% of BERT), SpikeLM (12.9× at T=1, 3.7× at T=4), Sorbet (27.16×), SpikeDecoder (86.7–92.3%) — is a **theoretical accumulate-vs-multiply count**, not wall clock, not measured.

The one **measured** comparison points the other way on speed. A 3-layer EGRU LM (750-dim embeddings, 1350 units, 95% weight-pruned, WikiText-2 test perplexity 81.4) on a **SpiNNaker2** chip (Khan Nazeer et al., arXiv:2312.09084, Table II): **0.39 W / 170.25 ms / 0.0653 J** per batch-1 inference, versus an **NVIDIA A100 at 60 W / 19.9 ms / 1.1935 J**. That is **18.3× less energy and 8.6× slower wall-clock**, and the authors explicitly note the GPU becomes more efficient at larger batch sizes (the LM task's memory bottleneck forced batch 1 on a single chip).

**On an RTX 5060 a spiking model will be slower and use more energy than the equivalent dense ANN. Full stop.** See §6.9 for what may legitimately be claimed.

---

## 2. Surrogate gradients: shape is settled, scale and horizon are not

### 2.1 Shape robustness, scale fragility

**Zenke & Vogels** (*Neural Computation* 33(4):899–925, 2021) swept three surrogate shapes — SuperSpike `h(x)=1/(β|x|+1)²`, Sigmoid′ `s(x)(1−s(x))`, and Esser/Bellec piecewise-linear `max(0, 1−β|x|)` — over grids of slope β and learning rate η. Verbatim: sharper shapes "led to a reduction of the size of the parameter regime in β … presumably due to vanishing gradients … However, there was no substantial reduction in maximum performance." Setting **β=0 (a plain straight-through estimator)** produced "a significant drop in performance comparable to a network without hidden units."

That last result matters here specifically: an STE would be the cheapest possible "surrogate" — one fewer op per micro-step on a launch-bound GPU (§6.2) — and it is **ruled out**.

### 2.2 The reset is our only implicit recurrence, and it is the pathway that breaks

Zenke & Vogels' Fig. 5 used an **asymptotic** SuperSpike `h(x)=β/(β|x|+1)²` whose peak grows with β. With the reset **detached**, normalised and asymptotic surrogates performed equally. With a **differentiable reset**, the asymptotic surrogate degraded, and the degradation amplified with depth. With gradient additionally flowing through explicit recurrent weights, it "drop[ped] network performance to chance level even for a network with a single hidden layer."

**This project has no lateral recurrent matrix, so the reset is the only implicit recurrence** — exactly the pathway this result concerns. Peak-normalising the surrogate and detaching the reset are the two cheap insurance policies.

**Gygax & Zenke** (arXiv:2404.14964; *Neural Computation* 37(5):886, 2025) sharpen it: "we find no difference in performance whether we back-propagate through the reset or not when the SG is scaled with 1/β_SG … while without this scaling, BP through the reset negatively affects performance." So at a peak-1 surrogate, a differentiable-reset ablation is **predicted to be a null result**.

**Correction to a common framing:** detaching the reset is *snnTorch's hardcoded behaviour* (`_neurons/neurons.py`: `reset = self.spike_grad(mem_shift).clone().detach()`), and **SpikingJelly's `detach_reset` defaults to `False`**. It is not a "near-universal default"; it is one library's default and another's opt-in. Neftci, Mostafa & Zenke (IEEE SPM 36(6):51–63, 2019) drop the reset term because it "empirically leads to better results."

### 2.3 The snnTorch surrogates are not width-matched — a shape ablation would measure width

Re-derived directly from `snntorch/surrogate.py`:

| Surrogate (default) | Peak | HWHM | Total gradient mass | g(\|x\|=1) |
|---|---|---|---|---|
| `atan(α=2)` | 1.000 | 0.3183 | 1.00 (exactly 1 ∀α) | 0.0920 |
| `fast_sigmoid(slope=25)` | 1.000 | 0.01657 (**19.2× narrower**) | 0.080 (**12.5× less**) | 0.00148 (**62× smaller**) |
| `sigmoid(slope=25)` | **6.25** (not normalised) | 0.0705 | — | g(0.5)=9.3e-5 |

To match `atan(α)` in HWHM you need `fast_sigmoid` slope **k = 0.6507·α** — i.e. **1.30**, not 25. And `sigmoid(slope=25)` at peak 6.25 **is precisely the unnormalised "asymptotic" surrogate** §2.2 identifies as the failure mode: it should be renormalised (divide by slope/4) or removed from the ablation menu.

Two traps: snnTorch's ATan **docstring** claims peak 1/π, contradicting its own code (peak α/2) — trust the code. And **an `atan α ∈ {0.5,1,2,4,8}` sweep is not a pure width sweep**: peaks are {0.25, 0.5, 1.0, 2.0, 4.0}, so α=4 and α=8 sit in the unnormalised regime this same section says to avoid. To sweep width at fixed peak, divide the ATan gradient by α/2.

### 2.4 The width cliff is quantified twice, independently

**LSG** (IJCAI 2023; Lian, Shen, Liu, Wang, Yan, Tang), mass-normalised rectangular surrogate `h(u) = (1/α)·1[|u−V_th| < α/2]`, CIFAR-10 / ResNet-19 / T=2:

| α | 0.5 | 1.0 | 2.5 | 5.0 | 10.0 |
|---|---|---|---|---|---|
| acc % | 92.12 | **92.68** | 90.68 | 61.54 | 30.82 |

**ASGL** (Wang, Jiang, Lian, Yan, Tang; ICML 2023, PMLR v202) independently reproduces the same cliff on CIFAR-10: 93.19 / 93.78 / 90.68 / 62.34 / 30.85 at the same five α. A 10× change in one scalar costs ~62 points; the optimum is near **1.0·V_th**. Aliyev & Adegbija (arXiv:2402.06211) sweep arctan α and fast-sigmoid k over 0.5–32 on SVHN: "beyond which the accuracy for the arctangent surrogate drops below 20%", and **fast sigmoid yields systematically lower firing rates at comparable accuracy** — i.e. surrogate width moves sparsity independently of accuracy, so sweeps must log both.

`atan(α=2)` has FWHM 0.6366·V_th — squarely inside the good band, and bit-identical to SpikingJelly/PLIF's ATan.

### 2.5 The >10-point shape gap that is probably a width gap

**Stan & Rhodes** (*Sci Rep* 14, 2024; doi:10.1038/s41598-024-71678-8) is the closest published setting to ours — binary activations, very long unrolls (784 to 16k+ steps). Binary S4D on sequential-CIFAR10: **79.33%** (4L/128) and **82.00%** (6L/512) with **arctan**, versus **69.62%** and **69.83%** with **fast sigmoid**; Path-X **61.2%** vs near-random **51.7%**. Verbatim: "fast sigmoid surrogate gradients cause a more than 10% drop in accuracy … regardless of the size of the Binary S4D model." (Absolute small-model gap is 9.71 points; >10 is relative.) A second unit (GSU) was far less sensitive: 82.49 vs 80.11, 85.00 vs 85.01.

**They almost certainly used the snnTorch defaults**, so this is very likely the 19× width mismatch of §2.3 rather than a shape effect. A width-matched `atan(2)` vs `fast_sigmoid(1.30)` run at equal HWHM would settle it, and is publishable either way.

### 2.6 The governing equation: gradient horizon

**SLTT** (Meng, Xiao, Yan, Wang, Lin, Luo; ICCV 2023, arXiv:2302.14311) Eq. 11: with a detached reset and no lateral recurrence, the temporal Jacobian of a LIF layer is **exactly diagonal**, with entries `λ = 1 − 1/τ` outside the surrogate window and 0 inside. The product over Δt steps decays as **λ^Δt**. The paper notes this is negligible for "commonly used small λ (e.g. λ = 0.5, 0.25, 0.2)."

Applied to this project's configuration (β=0.9, T=5, tbptt_chunk=32), recomputed and verified:

- per-character retention `0.9⁵ = 0.5905`
- forward memory e-folds in **1.90 characters**
- gradient falls below 1e-3 after 65.6 micro-steps = **13.1 characters**
- across a full 32-char chunk (160 micro-steps), `0.9¹⁶⁰ = 4.8e-8`

**We pay full BPTT memory over 32 characters for gradient that is numerically dead after ~13.** At T=1 and the same β, the horizon is 65.6 characters. This is the same arithmetic that drives §4.1 and cross-references directly to §3.2 (per-channel decay) and §4.11 (geometric decay grids).

> **Correction:** SLTT's Fig. 3 cosine-similarity experiment uses **dataset-specific** timesteps (CIFAR-10 N=6/12, DVS-CIFAR10 N=10/20, ImageNet N=4/8), not "T=6 and T=12" everywhere, and the "0.65–0.95" band is **unsupported** — the CIFAR-10 panel's axis extends down to 0.35. The paper only claims qualitatively that "the similarity maintains a high level."

### 2.7 Online/approximate methods: a price list, not an upgrade

SLTT (spatial-path-only backprop, memory constant in T) vs BPTT: CIFAR-10 3.00 GB / 6.35 h / 94.60% → 1.09 GB / 4.58 h / 94.59%; CIFAR-100 3.00 / 6.39 / 73.80 → 1.12 / 4.68 / 74.67; ImageNet (NF-ResNet-34, T=6) 28.41 / 73.8 / 66.47 → 8.47 / 66.9 / **66.19**; DVS-CIFAR10 3.70 / 4.47 / 73.60 → 1.07 / 3.43 / 77.30; DVS-Gesture 5.82 → 1.07 GB, 97.22 → 97.92. SLTT-1 on ImageNet: 66.17% in 32.03 h vs SLTT's 66.19% in 66.90 h.

> Two framing corrections: "under one third the memory" holds for ImageNet (29.8%), DVS-CIFAR10 (28.9%) and DVS-Gesture (18.4%) but **not** CIFAR-10 (36.3%) or CIFAR-100 (37.3%); and SLTT is **0.28 points below** BPTT on ImageNet.

OTTT vs SLTT head-to-head: CIFAR-10 1.71 GB / 184.68 s vs 1.00 GB / 54.48 s; ImageNet 19.38 GB / 7.52 h vs 8.47 GB / 2.23 h; DVS-CIFAR10 4.32 GB / 114.84 s vs 1.90 GB / 48.00 s. FPTT's complexity table (Yin, Corradi & Bohte, arXiv:2112.11231): BPTT Ω(T) memory; RTRL Ω(T) memory with Ω(c(T)T²) gradient cost; e-prop/OSTL/FPTT Ω(1) memory; FPTT-K Ω(T/K).

**None of them beats BPTT on accuracy.** FPTT's authors state plainly that e-prop and OSTL "at best approach BPTT performance" and that "none of these approximations have been shown to outperform standard BPTT." OTPE (Summe, Schaefer, Joshi; arXiv:2311.16151) reports "~70% increase in gradient cosine similarity to BPTT in the first hidden layer and ~50% in the second" over prior trace rules — i.e. these rules are *measurably misaligned* with the true gradient (SHD: BPTT 78.1%, OTPE 75.2%, OTTT/OSTL 70.5%).

**Their only advantage is memory constant in T, which truncated BPTT already delivers at O(chunk)** — and per-timestep backward or per-timestep optimiser updates multiply kernel launches and break the CUDA-graph capture that is this box's principal fusion mechanism (§6.3). **Do not port them.** Every published SLTT/OTTT result is T≤20 on rate-coded static or event images, where temporal credit assignment is nearly absent; for a char-LM the temporal path *is* the task.

> **Dropped:** the FPTT DVS-Gesture memory triple (LSTM-BPTT 10.8 GB / LTC-SNN-BPTT 13.2 GB / LTC-SNN-FPTT 5.6 GB) is **not in the cited tables** — FPTT Table 5 reports BPTT 15.72 GB (asterisked: halved batch, would not otherwise fit) vs FPTT 3.75 GB, with no LSTM row. Those three numbers appear to be bar-chart readings and are **unsourced**.
>
> **Dropped:** the "VGG16 on CIFAR-10: ANN 0.57 min/epoch and 1.47 GB vs SNN 78 min/epoch and 9.36 GB (137× slower)" figure. The numbers are real but come from **Rathi, Srinivasan, Panda & Roy, ICLR 2020 (arXiv:2005.01807)**, not the PMC source it was attributed to — and critically, **the SNN ran at T=100** on an RTX 2080 Ti. Citing a T=100 slowdown to argue T=1 over T=4 extrapolates ~25× beyond the measurement. The argument for T=1 stands on other grounds (§1.2, §2.6, §6.2); this number should not be used.

### 2.8 Threshold-dependent normalisation

**tdBN** (Zheng, Wu, Deng, Hu & Li, AAAI 2021, 35(12)) normalises pre-activations to **N(0, (α·V_th)²)** rather than N(0,1), on **both** the spatial and temporal axes. Its Block Dynamical Isometry analysis separates the two knobs verbatim: "τ_decay influences the gradients propagation in temporal domain and V_th effects the spatial dimension." This took direct training from <10 layers to 50 layers: 93.16% CIFAR-10 (ResNet-19, T=6), **63.72%** plain ResNet-34 ImageNet at T=6 (the widely-quoted **67.05% is ResNet-34 *large*, with doubled channels**), 67.8% DVS-CIFAR10, 96.87% DVS-Gesture vs SLAYER 93.64%. **Correction:** tdBN folds into the **convolution weights and bias** at inference (W′ = λ_k·α·V_th·W / √(σ²+ε)), **not** into the threshold.

Practical consequence: snnTorch applies the surrogate to `(mem − threshold)` with an **absolute** width, so changing `--threshold` silently changes the width-to-signal ratio. **Lock V_th = 1.0 for all surrogate experiments**, or scale `atan α` as α/V_th.

### 2.9 Learned widths anneal narrower

Both learned-width papers find the same direction. **ASGL** reports "the width αs consistently decline throughout training and converge to specific values across all layers" — though **this is from the image-reconstruction task (Fig. 3f), not a classification benchmark**, a narrower evidence base than usually implied. **LSG** ties width to a learnable membrane decay and reaches **94.41%** (CIFAR-10, ResNet-19, T=2) vs 92.68% for the best fixed width and 93.16% for trainable-decay-only — a **+1.73 point** delta.

> **Corrected:** ASGL's DVS-CIFAR10 result in the ICML paper is **84.50 ± 0.08**, not 85.50. The 85.50 figure appears only in the authors' GitHub README, which itself states it is "higher than the reported performance in the manuscript."

Neither paper ran the obvious control — a hand-scheduled anneal against a properly tuned constant — and **neither tested on a language model**. A cosine anneal of `atan α` from ~1 to ~4 costs one line and zero extra kernels; expect an effect of order +1.7 points-equivalent, and treat it as unvalidated for text.

---

## 3. Neuron models: what fits inside the invariants

### 3.1 Direct coding, learnable threshold and leak collapse the required T

**Kim, Park, Moitra, Bhattacharjee, Venkatesha & Panda** (ICASSP 2022, arXiv:2202.03133) find direct input coding beats Poisson rate coding "especially for a small number of timesteps", and verbatim: "As the dataset and network architecture gets more complicated, the performance gap between the two coding increases." Two counterweights the survey should carry: **rate coding is up to ~20% more accurate under adversarial attack**, and rate coding is reported **more** energy-efficient (direct coding needs multi-bit precision in layer 1). Neither matters for this project's objective, but both should be stated rather than suppressed.

**DIET-SNN** (Rathi & Roy, IEEE TNNLS 2023, 34(6):3174–3182; arXiv:2008.03658) gives the ladder that matters: Poisson rate coding needs **150** timesteps → direct input encoding **25** → + learned per-layer thresholds **15** → + learned per-layer leak **5**. Accuracies at T=5: CIFAR-10 VGG16 92.70 / ResNet20 91.78 / VGG6 89.42; CIFAR-100 VGG16 69.67; ImageNet VGG16 69.00. Verbatim on granularity: "All neurons in a layer share the same leak and threshold value… we did not observe any significant improvement by assigning individual threshold/leak to each neuron."

**But that per-layer finding is from vision, where the temporal dimension is a throwaway.** Here the decay *is* the memory (§2.6), so per-channel is worth re-testing — see §3.2, §4.11.

Learnable threshold + leak is ~2 scalars per layer (6 params on a 5–20M model — free) and it removes a hyperparameter search this budget cannot afford.

### 3.2 Learnable and heterogeneous time constants

**PLIF** (Fang, Yu, Chen, Masquelier, Huang, Tian; ICCV 2021, arXiv:2007.05785), τ = 1/sigmoid(a), one scalar per layer. Its Table 4 shows the benefit is **robustness to bad initialisation**, not peak gain: CIFAR-10 PLIF(τ₀=2) 93.50 vs LIF(τ=2) 93.03, but PLIF(τ₀=16) **93.23 vs LIF(τ=16) 47.50**; DVS128-Gesture 97.57/96.88 and 92.01/**76.74**; CIFAR10-DVS 74.80/73.60. **GLIF** (Yao, Li, Mo, Cheng; NeurIPS 2022) reaches 77.35 ± 0.07 CIFAR-100 (ResNet-19, T=6). **KLIF** (Jiang & Zhang, *Neural Computation* 36(12):2636–2650, 2024) adds a learnable scale reshaping the surrogate slope "without introducing additional computational cost."

**Heterogeneity** (Perez-Nieves, Leung, Dragotti & Goodman, *Nat Commun* 12:5791, 2021) verbatim: "On the most temporally complex auditory tasks, accuracy improved by a factor of around 15–20%, while for the least temporally complex task N-MNIST we saw no improvement." Table 1: N-MNIST 97.4±0.0 → 97.3±0.1; SHD 71.7±1.0 → **81.7±0.8**; SSC 49.7±0.4 → **60.1±0.7**. **Those are +10.0 and +10.4 percentage points (+14% and +21% relative) — the paper's "15–20%" is relative; do not budget a 15–20 point absolute gain.** Homogeneous nets needed ~10× more neurons to match, and heterogeneous **initialisation** is the source of the robustness — which argues for a **fixed geometric β grid before a learnable one** (§4.11).

Character-level language is a temporally rich task, so this predicts heterogeneity should help — which **contradicts this project's TIER6 result** (`--beta-per-neuron` cost +0.090 bpc, `--learn-beta` cost +0.211 bpc). §4.2 gives the likely mechanistic resolution.

`(unverified)`: DH-LIF/DH-SNN's specific best-in-class numbers on SHD/SSC/(P)S-MNIST/DEAP (Zheng et al., *Nat Commun* 15:277, 2024) — the paper and author list are verified, the benchmark deltas are not. The "long-tailed distributions match cortical data" phrasing is also stronger than the paper's own "optimal parameter distribution similar to experimental data".

### 3.3 Adaptation can substitute for recurrence — the key permission slip

This is the closest evidence that the no-lateral-W invariant is survivable. **SE-adLIF / EF-adLIF** (Baronig, Ferrand, Sabathiel, Legenstein; arXiv:2408.07517, now *Nat Commun* 16, 2025, DOI 10.1038/s41467-025-60878-z):

| Task | recurrent LIF | EF-adLIF | SE-adLIF |
|---|---|---|---|
| SHD | 90.27 ± 0.73 (0.45M, **recurrent**) | 94.68 ± 0.57 (2L) | **95.81 ± 0.56** (2L); 94.59 ± 0.27 (1L) |
| SSC | 75.23 | 79.80 ± 0.18 | **80.44 ± 0.26** |
| ECG | — | 87.83 ± 0.51 | **88.18 ± 0.36** |

The SHD LIF baseline is confirmed as *recurrent*, so "adaptation beats explicit recurrence" is legitimate. Verbatim: "when we trained SE-adLIF networks without recurrent connections, their dynamics degraded clearly slower than LIF networks with recurrent connections" — **but note that sentence is about the spring-mass oscillator/dynamics task, not the SHD/SSC classification benchmarks.** Overhead: two states (u, w), three extra trainable parameters (a, b, τ_w) per neuron.

**Discretisation matters and is a two-line difference.** With sub-threshold coupling the (u,w) pair is an oscillator; **Symplectic Euler (SE-adLIF)** "preserves stability for all possible parameterizations" with eigenvalue moduli below 1, while plain **Forward Euler (EF-adLIF)** "is quite imprecise which can quickly result in unstable and diverging behavior" — moduli exceed 1. SE beat EF on all three benchmarks. **Update w using the NEW u, not the old one.** Pure PyTorch; no kernel needed. This is the RF/adLIF analogue of §4.12's divergence boundary and §4.2's (1−β) normalisation: **long-timescale spiking neurons need explicit stability parameterisation or they diverge over a multi-hour run.**

> **Major misattribution, removed.** The figures "τ_b1 = 30 ms and τ_b2 = 300 ms for ~1200 ms working memory (500 ms converging faster)" are **not** from Bellec et al. (LSNN, NeurIPS 2018) or Salaj et al. (*eLife* 2021;10:e65459). They belong to **Shaban, Bezugam & Suri's DEXAT** double-exponential-threshold neuron (*Nat Commun* 12, 2021), reached second-hand via a 2024 *Communications Engineering* review. Salaj et al. actually use τ_a = 700 ms (sMNIST), 800–2000 ms (STORE-RECALL), uniform [1, 13500] ms (12AX). **The "30 ms fast pool / 300 ms slow pool" design prescription is therefore not an LSNN result**, and 30 ms is only ~1.5× a typical 20 ms membrane constant, contradicting the "1–2 orders of magnitude" generalisation built on it.
>
> **What survives, and is confirmed:** SFA moves temporal-credit tasks from near-chance to near-human — 12AX **97.79% with SFA vs 0.39% without**; sMNIST 93.7% vs 51.8%; 20-dim STORE-RECALL 99.09% vs chance.
>
> **Also corrected:** LSNN's sequential-MNIST result is **94.7% (1 ms/pixel) and 96.4% (2 ms/pixel)** — the ~98.0% figure is the **LSTM baseline** from the same table (LSTM 98.5% / 98.0%). On TIMIT, LSNN reaches **33.2% phone error rate vs 29.7%** for the best LSTM. Adaptive thresholds get an RSNN *close to* LSTM, **not level with it**. The mechanism description is confirmed verbatim: threshold B_j(t) "increases by some fixed amount β/τ_a,j for each spike of this neuron j, and then decays exponentially back to a baseline value"; BPTT with pseudo-derivatives and DEEP R rewiring both confirmed (the sparse LSNN kept 12% of connections and beat the fully connected one).

### 3.4 Reset: cheap, free-parameter, and the evidence is genuinely mixed

**CLIF** (Huang et al., ICML 2024, arXiv:2402.04663) identifies the mechanism verbatim: "as long as the neuron fires at least once within the temporal range of (t+1,T), part II = 0 and the corresponding temporal gradient error will vanish." Its fix adds a complementary membrane potential opening an extra gradient path at **zero learnable parameters** and "only add two lines of code to LIF neuron model": CIFAR-10 T=4 **94.89%** (vs PLIF 94.25% at T=6); DVS-CIFAR10 T=10 LIF 84.90 → CLIF **86.10**.

Soft vs hard is **contradictory across datasets**. **AR-LIF** (Huang, Meng, Liu, Chen, Ma; arXiv:2507.20746v3) Table 2 at 4 timesteps: CIFAR-100 **hard 73.4 vs soft 63.4**; CIFAR10-DVS **hard 83.8 vs soft 84.1**. Meanwhile **PSN**'s ablation chose "soft reset and detach reset for the PLIF and LIF neurons" as optimal on sequential CIFAR, and SpikeGPT's one working LM configuration uses **hard** reset. **Import the mechanism, not the number — measure it on bpc.**

> **A methodological warning worth carrying:** the AR-LIF PDF is not machine-readable and a PDF-based fetch returned **four entirely fabricated accuracy numbers and a wrong table number**. Use the v3 HTML. The same failure mode produced a spurious 86.2% for E-SpikeFormer's 173M model (§1.6).
>
> **Dropped:** an unsourced soft-vs-hard vision comparison (CIFAR-10 94.44% vs 94.34%, DVS-CIFAR10 82.20% vs 81.40%) carried no citation and must not become a stated prior.

Cost of removing reset entirely, on the closest sequential benchmark: **LIF 81.50% → LIF-without-reset 79.50%** on sequential CIFAR-10 (55.45 → 53.33 on CIFAR-100). PSN's conclusion is verbatim: "the direct removal of reset drops the learning ability of the vanilla spiking neurons." That −2.0 points is the price of unlocking everything in §4.3 and §6.7.

### 3.5 Parallel and multi-compartment neurons

**PSN** (Fang, Yu, Zhou, D. Chen, Y. Chen, Ma, Masquelier, Tian; NeurIPS 2023, arXiv:2304.12760). Abstract verbatim: "by removing reset, the neuronal dynamics can be reformulated in a non-iterative form and parallelized." Sequential CIFAR-10: **PSN 88.45 / sliding PSN 86.70 / masked PSN 85.81 / GLIF 83.66 / PLIF 83.49 / KLIF 83.26 / LIF 81.50 / LIF-no-reset 79.50**. Sequential CIFAR-100: 62.21 / 62.11 / 60.69 / 58.92 / 57.55 / 57.37 / 55.45 / 53.33. Parameter overhead verbatim: "using the PSN with T=4 will add 340 and 200 parameters in spiking ResNet-18 and VGG-11, which only increase 0.00291% and 0.00015%." Also confirmed: "removal of reset does increase firing rates."

> **Dropped:** the "~10–20× inference and ~5–15× training" speedup range **appears nowhere in the paper**. Section 4.1 reports t_LIF/t_PSN only as Fig. 3 curves over N = 2⁸–2²⁰ and T = 2–64, with purely qualitative text. Note also that PSN's LIF baseline is PyTorch-JIT-fused for inference but falls back to three separate JIT functions per timestep in *training* — closer to this box's situation than a fused kernel, so the qualitative direction still favours us.

**Channel-wise PSN** (arXiv:2501.14490, since retitled "Multiplication-Free Parallelizable Spiking Neurons with Efficient Spatio-Temporal Dynamics") replaces PSN's shared **T×T** matrix with a per-channel **C×k** temporal weight — a channel-separable causal convolution: sCIFAR-10 **91.17%**, sCIFAR-100 **66.21%**, SHD 95.71% at k=8. **This is the right shape for this project**: diagonal in the neuron axis (no lateral matrix), `hidden×k` parameters (24.6k at h=1536, k=16 — 0.5% of a 5M model), and **length-independent**, so it does not break CUDA-graph capture's static-shape requirement the way a T×T matrix would.

**PMSN** (Chen, Wu, Ma, Yan, Liu, Wu, Tan; arXiv:2408.14917, now IEEE TNNLS) puts multiple interacting compartments inside one neuron with "two parallelization techniques that decouple the temporal dependencies of neuronal updates", reporting >10× training acceleration and ~30% accuracy improvement over LIF on sequential CIFAR-10. Conceptually the generalisation of "one fast + one slow state per channel", but it costs more state tensors in BPTT memory on 8 GiB — a fallback, not a first build.

### 3.6 The readout never spikes

The standard construction is a non-spiking leaky (or non-leaky) integrator whose potential feeds softmax/cross-entropy; SpikeGPT's variant feeds the real-valued pre-spike block output (§1.2). **Spike-count logits at T=5 give only 6 distinguishable levels**, which for a ~99-way softmax is a hard bpc floor no neuron tuning recovers. Keeping the head in float does not violate "binary spikes between layers" — the readout is not a between-layers signal.

---

## 4. Efficient sequence models: the same diagonal recurrence, better parameterised

### 4.1 The horizon problem, stated correctly

At T=3, β=0.9: per-character retention `0.9³ = 0.729`, single-integrator 1/e horizon **3.16 characters**, half-life 2.19. At T=5, β=0.9: 0.590, horizon 1.90.

> **Correction that matters for prioritisation:** 3.2 characters is the horizon of **one** leaky integrator. **Three stacked** leaky integrators have impulse response ~ n^(L−1)·rⁿ with mean latency **L/(−ln r) ≈ 9.5 characters**. The network-level horizon is roughly **3× the headline number**. Use the honest band: **~3–10 characters**. This does not overturn the conclusion — nothing else in the architecture carries information further, since there is no lateral matrix, no attention, and no skip-in-time path — but it means the horizon argument is being made on a number ~3× too small, and it should be settled empirically (see §8).

### 4.2 The parameterisation bug that probably invalidates the project's own heterogeneity ablation

snnTorch's `Leaky` (verified at `snntorch/_neurons/leaky.py:239`) computes:

```
base_fn = self.beta.clamp(0, 1) * self.mem + input_
```

**No (1−β) scaling on the input.** The DC gain of that integrator is **1/(1−β)**: 10 at β=0.9, 100 at β=0.99, **1000 at β=0.999**. Every modern gated linear RNN instead uses a convex combination — minGRU's `h_t = (1−z_t)h_{t−1} + z_t·h̃_t`, and HGRN likewise — **precisely so that raising the decay lengthens memory without amplifying drive**. (Mamba/S6's ZOH discretisation is analogous but not identical; **GLA is a counterexample** — its recurrence `S_t = G_t ⊙ S_{t−1} + k_tᵀv_t` has no (1−G_t) input scaling, so "HGRN/Mamba/GLA are all equivalent" is wrong.)

This is the most likely mechanistic explanation for the project's own negative results (TIER6: `--learn-beta` +0.211 bpc, `--beta-per-neuron` +0.090 bpc). Under 1/(1−β) gain, unconstrained β learning drives neurons into saturation or silence. **Heterogeneous timescales were probably not tested fairly — they were tested through a badly-conditioned parameterisation**, which directly contradicts §3.2's prediction that heterogeneity should help on a temporally rich task. Re-running with `mem = β·mem + (1−β)·I` is ~8 lines of custom LIF and preserves every invariant.

### 4.3 Reset is the sole obstruction to associativity

A LIF neuron **without reset** is exactly a first-order linear recurrence `U_t = β·U_{t−1} + I_t`, which is associative and therefore parallelisable by scan, FFT, or a causal depthwise convolution with kernel `[1, β, β², …]`. The hard threshold plus reset makes the post-reset potential depend on the neuron's own spike history, so the recurrence stops being linear in the state.

The literature's four fixes:
1. **Delete reset** — PSN, SPSN, channel-wise PSN (§3.5).
2. **Approximate the post-reset potential** — SpikingSSM's SDN (§1.3), a <200-parameter 3-layer 1-D conv. Capped: cannot exceed the neuron it approximates.
3. **Iteratively bound it** — SPikE-SSM's parallel membrane-boundary compression (Zhong, Zhao, Wang, Guo, Zhang, Lu, Leng; arXiv:2410.17268).
4. **Replace reset's function** — Parallel Spiking Unit's probabilistic reset (Li, Sun, He, Dong, Zhao, Zeng; IJCNN 2024, arXiv:2402.00449); the dynamic-decay neuron (§4.6).

### 4.4 Data-dependent decay preserves associativity — the key permission slip

The recurrence `h_t = a_t·h_{t−1} + b_t` is associative under `(a₁,b₁) ∘ (a₂,b₂) = (a₁a₂, a₂b₁+b₂)` for **any** a_t. So Mamba/S6, GLA, HGRN, RWKV-6 and minGRU are all parallel-scannable *despite* input-dependent gates.

**minGRU** (Feng, Tung, Ahmed, Bengio, Hajimirsadeghi; arXiv:2410.01201) deletes exactly two things from a GRU — the hidden-state dependence in the gates and the tanh range restriction — so z_t and h̃_t are functions of x_t only. Result: **175×** faster per training step than a GRU at length 512, **1324×** at 4096; Selective Copying **99.5 ± 0.2** ("comparable to S6" at 99.8); char-level Shakespeare test loss **1.548** vs Transformer 1.547 and Mamba 1.575.

**This is the permission slip for selectivity here.** An input-dependent per-neuron leak `β_t = sigmoid(W_b · s_t^{l−1})` computed from the **feedforward incoming spikes** (not the layer's own state) adds Mamba-style selectivity, keeps the state transition diagonal, keeps the scan associative, and **introduces no lateral recurrent weight matrix** — satisfying every hard invariant. Cost: one extra hidden×hidden GEMM per layer per step (~+50% forward FLOPs at the current shape).

### 4.5 Mamba-2's SSD result: the existing leak *is* an SSM

**Mamba-2 / SSD** (Dao & Gu, ICML 2024) shows that `A_t = a_t·I` — a learnable scalar times identity — suffices to match masked attention. **The per-neuron diagonal leak this project already has is the same class of state transition as a modern SSM, just badly initialised.** That reframing is worth more than any import.

### 4.6 The closest prior art (2026 preprint)

**DSN / "Parallel Training in Spiking Neural Networks"** (Huang, Yao, Pan, Lv, Xu, Zheng, Xu, Li; arXiv:2602.01133, 1 Feb 2026) does exactly the translation §4.4 proposes: it argues reset should be removed but its two functions (nonlinearity, membrane control) preserved, replacing it with an input-dependent decay α_t produced by a `CausalConv1D` then sigmoid-squashed, **keeping serial inference**. Reported: **25.6×** training speedup over PSN at 16k length; models trained at 2k infer stably to **30k** (masked PSN fails past 2k); WikiText-103 perplexity **28.50 vs SpikingSSM 32.25**; sequential CIFAR-10 **90.10** vs PSN 88.45 and sliding PSN 86.70; ImageNet ResNet-18 **68.21** vs PSN 67.63 and PMSN 66.64; and **lower** firing rates than both vanilla and existing parallel spiking neurons.

The paper, authors and numbers are all verified. **It remains an unrefereed preprint** — weight it accordingly. The parallel-train / serial-infer split it targets is exactly what a chat REPL needs.

### 4.7 Measured on this box: the parallelisation payoff is small at our sequence lengths

Independently re-run on the actual RTX 5060 (torch 2.12.1+cu130, TF32 off), H=1536, B=256, N = L·T = 192, one layer's work per TBPTT chunk:

| Operation | Time | Note |
|---|---|---|
| 192 sequential (256,1536)×(1536,1536) GEMMs, fp32 | 24.68 ms | 9.4 TFLOP/s |
| Same, flattened into one (49152,1536)×(1536,1536) GEMM, fp32 | 17.47 ms | 13.3 TFLOP/s — **only 1.41×** |
| 192 sequential GEMMs, **bf16** | 8.22 ms | 28.2 TFLOP/s — **3.00× over fp32** |
| Flattened GEMM, bf16 | 5.57 ms | 41.6 TFLOP/s |
| Sequential elementwise LIF (soft reset, 192 steps), eager | 17.13 ms | |
| Chunked decay-matmul scan (C=32, no reset, fixed per-channel β) | **6.44 ms** | the only scan that wins |
| Depthwise causal conv, K=64 | 14.60 ms | |
| Naive Hillis-Steele log-depth scan | **49.5 ms** | **≈2.9× SLOWER than the loop** (an earlier 109 ms figure was ~2× inflated and implementation-dependent) |

The activation tensor is 302 MB fp32; one read+write pass at 383 GB/s is 1.58 ms — **a badly-written scan is bandwidth-bound and loses to the sequential loop it replaces.** Also: `torch.associative_scan` is unusable here — it requires `torch.compile` codegen (no Triton) and does not support autograd.

**Cross-reference to §1.3:** SpikingSSM's 101.2× is at 8K steps; this project has 192. Extrapolating that curve down predicts ~1–3×. **Frame parallel-scan work as buying longer context, not throughput.**

### 4.8 What is dead on arrival, and for the right reason

- **mamba-ssm**: the previously stated blocker (mandatory compiled `selective_scan_cuda` + `causal_conv1d`) is **overstated** — the core package now installs without compiling them. **The real blocker is that the README requires Linux; this is Windows.** Do not budget GPU-hours on getting it to build.
- **GLA / FlashLinearAttention**: chunkwise-parallel implementation is a **Triton kernel**. **Dead on arrival — no Triton.** (Correct citation: Songlin Yang, Bailin Wang, Yikang Shen, Rameswar Panda, Yoon Kim, ICML 2024, arXiv:2312.06635. An earlier draft listed a nonexistent "Zhang" and omitted Panda.)
- **S4D / S5** (Smith, Warrington, Linderman; ICLR 2023, arXiv:2208.04933): rely only on diagonalisation and an off-the-shelf associative scan. S5 averages **87.4% on LRA and 98.5% on Path-X**. **The math is importable in eager PyTorch.**

**Import the mathematics — diagonal state, geometric timescale initialisation, data-dependent scalar decay, chunkwise formulation — not the implementations.**

### 4.9 Fixed geometric multi-scale decay: zero extra parameters

**RetNet** (Sun, Dong, Huang, Ma, Xia, Xue, Wang, Wei; arXiv:2307.08621) gives every head a fixed scalar decay, verbatim in the paper: `gamma = 1 - 2^{-5-arange(0,h)}`, with decay matrix D_ij = γ^(i−j) for i≥j. That grid maps to per-character 1/e horizons of **31, 63, 127, 255, 511, 1023, 2047, 4095** characters — i.e. micro-step betas 0.9895 to 0.99992 at T=3. (The widely-used reference implementation uses a variant: `1 - exp(linspace(log(1/32), log(1/512), heads))`.)

**HGRN** (Qin et al., NeurIPS 2023, arXiv:2311.04823) adds a **learnable lower bound on the forget gate that increases monotonically with layer index**, so lower layers model local structure and upper layers long dependencies. Verbatim: "Combining gating and the lower bound consistently provides benefits, but the most significant improvement arises from the monotonically increasing lower bound" (ablation: w/o lower bound 24.71, random 24.60, decreasing 24.63). **Nuance the same sentence adds:** gating *per se* is "more critical" than the bound.

Both are **zero-extra-parameter, zero-extra-FLOP** changes and they directly attack §4.1's horizon. They cross-reference §3.2's finding that heterogeneous **initialisation** — not learning — is where the robustness comes from.

### 4.10 Complex-valued diagonal state

**Balanced Resonate-and-Fire** (Higuchi, Kairat, Bohte, Otte; ICML 2024, PMLR v235; arXiv:2406.00389): complex membrane `u̇ = (b + iω)u + I`, spikes on `Θ(Re(u) − θ(t))`, with an explicit divergence boundary `p(ω) = (−1 + √(1−(δω)²))/δ` keeping the spectral radius ≤ 1. Results: SHD **91.7 ± 0.8** vs ALIF 90.4 with **~7× fewer spikes** and fewer parameters (108,820 vs 142,120); S-MNIST **99.0 ± 0.1** vs 98.7; PS-MNIST **95.0 ± 0.2** vs 94.3; and 95% of final accuracy reached **within the first five epochs** (not ten).

This is S4D/DSS/S5's mechanism transplanted into a spiking neuron: a 2×2 block-diagonal rotation per neuron. Whether that counts as "no explicit lateral recurrent weight matrix" is **a judgement call this project must make explicitly**. The transferable free lesson is the divergence boundary — the RF analogue of §4.2's (1−β) normalisation and §3.3's symplectic ordering.

---

## 5. Character-level LM baselines, metrics, and evaluation protocol

### 5.1 Metric definitions, and why they matter here

`bpc = (1/N)·Σ −log₂ p(x_i | x_<i) = cross-entropy in nats / ln 2`. Character perplexity = 2^bpc. **Bits-per-byte** = total bits / UTF-8 bytes, the only tokenizer-agnostic form. For enwik8 the "characters" *are* the raw byte values (Al-Rfou et al. state "205 unique bytes", though papers differ between 204 and 205 depending on UNK bucketing), so **bpc ≡ bpb** — Al-Rfou's enwik8 column is literally headed *bpb* while their text8 column is headed *bpc*. text8's 27-symbol alphabet is pure ASCII, so bpc ≡ bpb there too.

**A 99-symbol printable-ASCII vocabulary is 1 char = 1 byte, so this project's bpc is also bpb and is directly comparable to byte-level models — but only if corpus construction did not drop or remap non-ASCII bytes.** If it did, we deleted the hardest-to-predict symbols and the bpc is deflated by an unknown amount. **This must be stated in one sentence next to every headline number.** It is the single most important reporting item in the whole survey.

### 5.2 Standard splits — and one lineage correction

- **enwik8** = first 10⁸ bytes of the 2006-03-03 English Wikipedia XML dump; split **first 90 M / next 5 M / last 5 M bytes**.
- **text8** = **first 10⁸ bytes of `fil9`, and `fil9` is filtered from `enwik9`, not enwik8.** Mahoney verbatim: "We filter the 1 GB test file enwik9 to produce a 715 MB file fil9" and "text8 is the first 10⁸ bytes of fil9." Preprocessing verbatim: "Numbers were spelled out ('20' becomes 'two zero'…). Upper case letters were converted to lower case. Finally, all sequences of characters not in the range a-z were converted to a single space", giving the 27-symbol a-z+space alphabet. Same 90/5/5 split.
- **PTB char-level** = 5017k / 393k / 442k characters (Mikolov sections 0-20 / 21-22 / 23-24), ~50 symbols, 5.9M total.

The 90/5/5 convention comes from the ML literature, not Mahoney's page; pre-2015 papers (Graves' 96M/4M) used different splits and are **not** strictly comparable.

> **Consequence of the lineage correction:** text8 covers a strictly larger and different span of Wikipedia than enwik8 (~140 MB of the dump vs 100 MB). They are **not the same text with markup added or removed**, which weakens §5.5's "markup makes enwik8 easier" explanation — part of the enwik8/text8 inversion is different content.

### 5.3 The context-length curve

**Al-Rfou et al.** (AAAI 2019, arXiv:1808.04444) Table 2, T64 on text8:

| Eval context (chars) | 32 | 64 | 128 | 256 | 512 |
|---|---|---|---|---|---|
| test bpc | 1.34 | 1.26 | 1.20 | 1.16 | **1.13** |
| dev bpc | 1.25 | 1.17 | 1.12 | 1.09 | 1.06 |

Verbatim: "this trend levels off after 512 characters; we do not see better results using a context of 1024." Protocol verbatim: "we use the model's prediction at the final position of the final layer to compute the probability of a character given a context of 512 characters" — a **stride-1 sliding window**, not disjoint chunks.

256-char windows mean the average scored character sees ~128 characters of context, worth roughly **+0.07 bpc** versus 512 on this curve. **Caveat: this is a 235M-parameter 64-layer Transformer. It bounds the effect; it does not predict it for a 5M contractive-LIF model.**

### 5.4 Disjoint-window evaluation biases bpc upward

The "early token curse" is real and uncontroversial: the first characters of each window are predicted with little conditioning, so their per-character loss far exceeds the corpus mean. Standard fixes: **carry state contiguously** across the test split for stateful recurrent models, or **strided sliding windows** scoring only the last s tokens for fixed-context models. `(unverified attribution)`: Biderman et al. (arXiv:2405.14782) is a real paper with the stated title but I did not confirm it discusses window fragmentation specifically. And **Transformer-XL's "whole contribution" was not eval-time caching** — segment-level recurrence is a training mechanism as much as an eval one, and relative positional encoding is the second core contribution that makes state reuse coherent.

**The measurement this project should run is genuinely novel:** an LIF membrane with β<1 is contractive, so its effective memory may be far shorter than an LSTM cell's, and the carried-vs-reset gap could be near zero. **No published measurement of this for a leaky-membrane model exists.**

### 5.5 Corpus dominates absolute bpc

**MEGABYTE** (Yu et al., NeurIPS 2023), same architecture, same 80 B training bytes, compute-matched: Code **0.411**, arXiv 0.678, Stories 0.978, PG-19 1.000, Books **1.007** — a **0.596 bpb spread from corpus alone**; its Transformer baselines span 0.575 to 1.097.

Within char-LM the same effect shows as an enwik8-vs-text8 inversion — identical models score **better on enwik8**: mLSTM 1.24 vs 1.27; T12 1.11 vs 1.18; T64 1.06 vs 1.13; Transformer-XL 24L 0.99 vs 1.08. The usual explanation (enwik8's XML markup is nearly free to predict) is **plausible folklore, not a measured decomposition** — no source I checked ablates markup's contribution, and per §5.2 part of the gap is simply different text. Also note MEGABYTE's models are 320M–1B params; the 0.6 bpb corpus spread is a large-model result.

**Decisive for reporting: 1.2362 bpc on a 639 MB modern-prose corpus is NOT "better than a 27M LSTM on enwik8 (1.31)."** Clean modern prose with normalised whitespace and restricted ASCII is a substantially easier corpus, worth several tenths of a bit. **That number is comparable only to another model scored on the identical corpus file.**

### 5.6 Dynamic evaluation is a different protocol

Same checkpoints, static → dynamic: mLSTM 46M enwik8 **1.24 → 1.08**; Mogrifier 96M enwik8 **1.122 → 0.988**; LSTM 96M **1.155 → 1.020**; Mogrifier 24M PTB-char 1.131 → 1.088; Graves' 21.3M 1.67 → 1.33. **The gain is 0.04–0.34 bpc — larger than most architectural effects in the same papers.**

Direct consequence for chat mode: **carrying membrane state across a conversation is legitimate stateful inference and should be labelled as such.** If any weight or threshold adapts at test time, that is dynamic evaluation and must occupy a separate table row, never the same column as static numbers.

### 5.7 Seed noise, tuning budget, and the entropy floor

> **Misattribution corrected.** The often-quoted "std ~0.2 perplexity, mean ~0.7 off the best tuning run" figures come from **Mogrifier LSTM (Melis, Kocisky & Blunsom, ICLR 2020), Appendix B**, and refer to **PTB word-level perplexity** — *not* from Melis, Dyer & Blunsom (ICLR 2018). ICLR 2018's own figures are different: "the variance induced by [initialisation seed] and [data shuffling] together is roughly equivalent to an absolute difference of **0.4 in perplexity on Penn Treebank and 0.5 on Wikitext-2**", and floating-point nondeterminism alone produces variance almost as large as **seed and shuffling combined** (a bigger effect than usually stated). ICLR 2018 also verbatim: a **1.0 perplexity gap** is needed for a statistically robust difference, and properly tuned vanilla LSTMs beat most "novel" architectures — **tuning budget confounds architecture comparisons.**

Paired bootstrap resampling over the test set does **not** account for training-run variance and can declare two models differing only by random seed significantly different at **p = 0.03** (Mueller, bricksdont.github.io, summarising Dror et al.; note that post cites Dror only, not Reimers & Gurevych, though Reimers & Gurevych 2017 "Reporting Score Distributions Makes a Difference" is a real EMNLP paper).

Entropy floor for orientation: Shannon (BSTJ 1951) bounds English at **0.6–1.3 bits/char**. Cover & King (IEEE Trans. IT 1978) is often cited at 1.25–1.35 but secondary sources report 1.25, 1.3, and 1.3–1.7 for individual subjects — treat **1.25–1.35 as `(unverified)`**.

> **Do not import a variance figure.** Converting Melis' word-level 0.2/58 relative perplexity to "±0.01 bpc" is loose — it maps to ~0.005 bits/token, and word-level→char-level does not transfer additively at all. Surrogate-gradient training with hard thresholds and gradient mismatch may well be noisier. **Measure our own seed std.**

---

## 6. Systems: what this GPU actually permits

### 6.1 The brief's premise is partly wrong — custom elementwise CUDA kernels ARE available

**This is the single most consequential systems finding, and it was independently reproduced on this box.**

`torch.cuda.jiterator._create_jit_fn` / `_create_multi_output_jit_fn` compile arbitrary **elementwise** CUDA C++ at runtime via **NVRTC, which ships inside the torch wheel** (`torch/lib/nvrtc64_130_0.dll`, `nvrtc64_130_0.alt.dll`, `nvrtc-builtins64_130.dll`). **No nvcc, no CUDA Toolkit, no MSVC, no Triton involved.** Verified: single- and multi-output LIF kernels compile (0.068 s / 0.213 s) and run; max error vs fp32 reference **4.76837e-07**; fp32, bf16 and fp16 all work; output has `requires_grad=False` and `grad_fn=None`.

**Constraints, all verified in source:**
- **≤8 tensor inputs, ≤8 outputs** (`torch/cuda/jiterator.py` line 79 asserts).
- **Elementwise only** — no reductions, no gather/scatter, no matmul.
- **No autograd** — the backward must be hand-written inside a `torch.autograd.Function`.
- Signature order is `f(in0..inN, extra0..extraM, T& out0..)`, by-reference outputs **last**. Scalar kwargs have create-time defaults but **are overridable per call** (a common misstatement).

**Fused LIF speedup over eager: ~4.0–4.4×** across B=32/64/256, H=512/1024 (eager 294–311 µs, fused 68–78 µs). An earlier 5.08–5.34× figure did not reproduce; the ratio is highly sensitive to how many kernels the eager baseline emits.

**This reopens the optimisation space the project had written off.** The LIF update is purely elementwise — it is the one "custom kernel" that matters, and it is available.

### 6.2 The workload is launch-latency-bound, not FLOP- or bandwidth-bound

Measured launch floor: **13.4 µs** for a single tiny kernel, **12.8 µs** amortised per launch in a 50-kernel chain. Shape independence is stark: eager T=4 LIF cost **297.4 µs at B=32,H=512; 302.5 µs at B=256,H=512; 312.2 µs at B=64,H=1024** — an 8× change in work for a 5% change in time. Actual memory traffic at B=64,H=512 fp32 is ~0.5 MB ≈ 1.4 µs at 383 GB/s.

**Score every optimisation by "how many kernel launches does it remove", not "how many FLOPs does it save."** Micro-optimising arithmetic is pointless. Windows WDDM makes this worse than Linux.

### 6.3 CUDA graphs: real, but the microbenchmark number is not the model number

Microbenchmark, confirmed: a T=8 eager LIF loop went **574.8 µs → 56.7 µs (10.15×)** under graph capture. Capture mechanics confirmed: works with `torch.optim.Adam(capturable=True, foreach=True)` and `zero_grad(set_to_none=False)`. **Jiterator kernels survive capture and replay correctly.**

**But on a realistic model the benefit collapses.** On the exact config (V=256, H=768, T=4, L=128, B=16, 1.575M params), graph-alone gave **1.11×** (5.01 ms → 4.51 ms) — not the 1.63× previously claimed, and nowhere near 10×. As GEMM work grows the step stops being launch-dominated. **Expect ~1.1–1.2× from graphing at target scale, not 2–3×.**

### 6.4 The "3.48× end-to-end" headline is retracted

> **This is the most misleading number in the source material and must not be carried forward.** The claimed 15.61 ms → 4.48 ms (213.6 → 456.7 ktok/s) on a 1.57M-param model did not reproduce. Rebuilding the exact stated model (the parameter count *does* come to 1.575M, so the spec is right) gave **5.01 ms eager = 408.9 ktok/s** — plain unoptimised eager is already within 12% of the claimed *fully optimised* end state. With graph capture: 4.51 ms / 454.4 ktok/s. **The honest figure is ~1.1–1.3×, not 3.48×.**
>
> Worse, the benchmark model **cannot be scanning the sequence dimension** — a genuinely sequential-over-L version of the same model took **194.18 ms**, 39× slower. The benchmarked network has no cross-token recurrence and is not "char-LM-shaped" in any meaningful sense.
>
> **The derived budget arithmetic ("50–60 GPU-hours becomes ~175–210 effective hours") is void.**

### 6.5 Batch is nearly free — but not for the stated reason

Confirmed qualitatively and it is the safest lever here: one fused LIF step at H=512 cost **17.73 µs at B=8 and 20.89 µs at B=1024** — 128× the work for **1.18×** the time; per-sample 2.22 µs → 0.020 µs (~109×).

> **The crossover reasoning is wrong.** At B=1024 the measured 20.89 µs falls **below** the supposed bandwidth-bound floor (8.4 MB / 383 GB/s = 21.9 µs), so the workload is demonstrably not bandwidth-limited there. At B=4096, measured 73.57 µs against a predicted 87.6 µs implies **~456 GB/s effective** — above the quoted spec, because the 32 MB L2 on GB206 is absorbing traffic. **383 GB/s does not predict the crossover, and the "B·H ≈ 250K–500K elements" rule of thumb is unsupported.** Batch stays nearly free well past it.

**Push batch to the VRAM limit, not to a tuned "speed" point.**

### 6.6 Hoisting the input projection

Confirmed and slightly understated: T=4 separate in-loop GEMMs **0.908 ms**, one hoisted GEMM **0.201 ms** (**4.52×**). If each micro-step genuinely needs a different projection, batching into one fat [H, T·H] GEMM gives only **1.35×** (0.908 → 0.671 ms), because that version does real FLOPs rather than eliminating them.

**The 4.5× only materialises with a shared input current across micro-steps — i.e. direct coding**, which is a free parameter here. This is the cleanest cross-reference between §3.1's modelling result and §6.2's systems result: **the same choice wins on both axes.**

### 6.7 The exact triangular-Toeplitz scan, and its hidden failure mode

`v_t = decay·v_{t−1} + i_t` is a first-order linear recurrence, so `v = K @ i` with `K[l,s] = decay^(l−s)` for l≥s. Measured: L=256, B=16 sequential loop **8.246 ms → 0.126 ms (65.5×)**, max error 1.91e-06; L=1024, B=4 **82.6×**, max error 2.86e-06. **It is exact, not an approximation** — pure fp32 rounding. Valid **only without reset** (§4.3), and legal here precisely because the no-lateral-W invariant keeps the recurrence diagonal.

> **Attribution softened.** PSN validates *removing reset to enable parallelisation* (abstract verbatim, §3.5), but PSN's actual contribution is a **learnable dense temporal weight matrix**, not a fixed `decay^(l−s)` exponential Toeplitz kernel. Say "PSN validates removing reset", not "the formulation matches PSN". PSN was also evaluated only on temporal/static classification, never autoregressive LM.
>
> **Unflagged hazard:** at L=1024, `0.9^1023 = 1.40e-45` — **subnormal/zero in fp32** (min normal 1.18e-38). At decay=0.9 the context is **silently truncated past ~L=830**. Any long-context Toeplitz form needs an explicit underflow analysis, and it interacts with §4.9's long-horizon decay grids.

### 6.8 Precision: bf16 is specifically bad for thresholded spiking

Verified with `torch.nextafter` on this box — representable spacing just above v = 1.0:

| dtype | spacing at 1.0 | relative to threshold |
|---|---|---|
| bf16 | **0.0078125** | 0.78% |
| fp16 | **0.0009765625** | 0.098% (**8× finer**) |
| fp32 | 1.1920928955078125e-07 | 1.2e-5 % |

Membrane values quantise onto a coarse grid straddling the threshold, so `v ≥ thr` comparisons flip, and the surrogate gradient is evaluated at a quantised distance-to-threshold. `(unverified)`: reported spike-count drift over 16 timesteps of −0.25% (bf16) vs −0.01% (fp16) — consistent with the spacing but not independently reproduced.

**This inverts the usual "bf16 is safer than fp16" heuristic for this workload.** Keep membrane state and the threshold comparison in **fp32**; run only the GEMMs in reduced precision. The state is tiny relative to weights, so this costs almost nothing — and §4.7 measured the bf16 GEMM win at **3.00×**, orthogonal to every architectural change.

### 6.9 Sparsity buys nothing on this GPU, and energy claims must be labelled

Dense GPU kernels process all elements uniformly; GPUs can exploit **weight** sparsity (static, e.g. NVIDIA 2:4 — Mishra et al., arXiv:2104.08378) but **not activation** sparsity, which is dynamic and unpredictable. **Spike sparsity is precisely activation sparsity.**

> **Two source corrections.** The "12.5% tensor-core utilisation on unstructured sparsity" figure is **not** in either paper it was attributed to — it traces to **Libra** (arXiv:2506.22714), measured on H100 for matrices with a single nonzero per column vector. And the "~90–95% breakeven for sparse BLAS" is **folklore**: published breakevens vary enormously (NVIDIA reports cuSPARSE Block-SpMM beating cuBLAS below 40–50% density, i.e. 50–60% sparsity; SMaT beats cuBLAS at ~78%). **Do not state 90–95% as fact.** The *conclusion* — no speedup from spike sparsity here — is nonetheless right.

**Energy reporting.** The defensible basis is **Horowitz (ISSCC 2014, "1.1 Computing's energy problem")**: at 45nm, 32-bit FP add **0.9 pJ**, 32-bit FP multiply **3.7 pJ**, hence the ~**4.6 pJ MAC**. **Horowitz's own 45nm 32-bit DRAM read is 640 pJ.**

> **Dropped:** the "~200 pJ off-chip vs ~1.5 pJ MAC, a ~130× gap" triple. It is a modern-HBM-accelerator claim, does not appear in the ISSCC 2014 paper it was cited to, and could not be sourced. The *methodological point* — that data movement dominates and arithmetic-only energy accounting is not defensible — is correct and should be kept, without that number.

**Permitted claim:** SynOps/token and mean firing rate as architectural measurements, with any pJ figure explicitly labelled "hypothetical 45nm-ASIC projection using Horowitz's 4.6 pJ MAC / 0.9 pJ add, arithmetic only, data movement unaccounted." **Never present GPU wall-clock or watts as evidence of SNN energy efficiency.** Cite the SpiNNaker2 measurement (§1.9) as the one honest hardware data point.

### 6.10 Definitively dead here

- **torch.compile / Inductor**: reproduced verbatim — `TritonMissing: Cannot find a working triton installation`. `import triton` → ModuleNotFoundError. Environment confirmed: torch 2.12.1+cu130, Python 3.14.6, RTX 5060, compute capability (12, 0) = sm_120.
- **Sparse Spiking Gradient Descent** (Perez-Nieves & Goodman, NeurIPS 2021, arXiv:2105.08810; up to **150×** backward speedup, **85%** more memory efficient): its released implementation ships custom CUDA extensions doing **gather/scatter over active-neuron index lists**. That needs nvcc + MSVC. **Genuinely dead on arrival — and jiterator cannot substitute, because it is elementwise-only.** Do not budget time to port it.
- **SLAYER / EXODUS** (Bauer, Lenz, Haghighatshoar, Sheik; *Front. Neurosci.* 2023, DOI 10.3389/fnins.2023.1110444; arXiv:2205.10242): shipped `sinabs-exodus` / `rockpool-exodus` backends are custom CUDA. Same verdict.
- **cupy**: not installed; its wheels bundle NVRTC and need only the driver, so it is a *possible* route — but it duplicates what jiterator already provides for elementwise ops while adding a **second CUDA memory pool contending with PyTorch's allocator for 8 GiB**. `(unverified on sm_120.)` **Deprioritise.**

### 6.11 The industry-wide winning strategy is exactly what jiterator enables

Open Neuromorphic's benchmark (16,000 neurons, batch 16, 500 timesteps, RTX 4090, fp32): **SpikingJelly with the CuPy backend was fastest at 0.26 s** for combined forward+backward; custom-CUDA implementations (Lava-DL SLAYER, Sinabs EXODUS, Rockpool EXODUS) were **1.5–2×** slower; pure-PyTorch frameworks slower still. `torch.compile` "brought the performance of Norse models close to that of JAX/Spyx, but we did not observe significant speedups for snnTorch and Sinabs." (The source does **not** say Norse reached SpikingJelly.)

SpikingJelly's CuPy backend fuses the multi-step LIF into a single kernel compiled from raw CUDA source. **The winning strategy field-wide is single-kernel multi-step LIF fusion — and jiterator can replicate it here.**

### 6.12 BPTT memory: T and L are not interchangeable

> **Corrected, and the practical implication reverses.** The claim that activation memory scales linearly in the product T×L rested on two points rising 18.6% for an 8× change in T·L — dominated by fixed overhead. The measured 2×3 grid (B=16, H=512, peak MiB):
>
> | | L=64 | L=128 |
> |---|---|---|
> | T=1 | 260.0 | 282.0 |
> | T=2 | 261.0 | 285.0 |
> | T=4 | 265.0 | 293.0 |
>
> **Doubling L costs +22 MiB; quadrupling T costs +11 MiB.** Sequence length is far more expensive per unit than micro-steps. Dropping T=4 → T=1 saved **~11 of 293 MiB (~4%), not 4×**. The claim "T is as expensive as sequence length for memory" and the derived "T=1 buys 4× the batch" are **both false**.

EXODUS's complexity statement is correctly quoted and stands: "A naive implementation of BPTT with T discrete-time steps and N fully connected neurons incurs a computational complexity of order O(N²T) and a memory overhead of O(NT)." The "~1.33× recompute for √L memory" figure is Chen et al. 2016 sublinear-memory (~30% overhead), not an EXODUS result.

**T=1 is still strongly favoured — on gradient horizon (§2.6), kernel launches (§6.2), input-GEMM hoisting (§6.6) and SpikeGPT's precedent (§1.2) — but not on activation memory.** Use the right argument.

---

## 7. Reference table: character-level bpc by family and parameter count

> **CROSS-CORPUS COMPARISON IS INVALID.** Per §5.5, corpus choice moves absolute bpc by more than any architecture decision in this table: the same MEGABYTE architecture spans 0.411 (Code) to 1.007 (Books) at fixed compute and data, and the *same* 12-layer Transformer scores 1.11 on enwik8 but 1.18 on text8. **Rows may be compared only within a corpus block, and only when protocol (static eval, split, context length) matches.** All figures below are **test** bpc under **static** evaluation on the standard 90/5/5 byte split unless noted.

### enwik8 (bpc ≡ bpb; 205 byte values)

| Model | Params | Test bpc | Note |
|---|---|---|---|
| Grid LSTM (6L) | 17M | 1.47 | lowest-param modern anchor |
| MI-LSTM (1L) | 17M | 1.44 | |
| VD RHN (10L) | 21M | 1.30 | |
| VD RHN (5L) | 23M | 1.31 | |
| Graves stacked LSTM (7L) | 21.3M | 1.67 | **validation, 96M/4M split — not comparable** |
| AWD-QRNN (4L) | 26M | 1.336 | |
| LN HyperNetworks | 27M | 1.34 | |
| tuned LSTM (d4) | 27M | 1.31 | Melis black-box HP search |
| tuned LSTM (d4) | 46M | 1.30 | |
| LN HM-LSTM | 35M | 1.32 | |
| RHN | 46M | 1.27 | |
| FS-LSTM-4 | 47M | 1.25 | |
| mLSTM | 46M | 1.24 | 1.08 with dynamic eval |
| AWD-LSTM (3L) | 47M | 1.232 | |
| LSTM (d4) | 48M / 96M | 1.195 / 1.155 | 96M → 1.020 dyneval |
| Mogrifier LSTM (d4) | 48M / 96M | **1.146** / 1.122 | *1.135 is the 48M **validation** figure*; 96M → 0.988 dyneval |
| SHA-RNN (4L) | 52M | 1.076 | single donated Titan V, no intensive HP tuning |
| SHA-RNN (per-layer attn) | 54M | 1.068 | |
| T12 (Al-Rfou) | 41M inf / 44M train | 1.11 | |
| Transformer-XL 12L / 18L / 24L | 41M / 88M / 277M | 1.06 / 1.03 | 24L: **0.99** |
| T64 (Al-Rfou) | 235M | 1.06 | |
| Adaptive attention span | 39M | 1.02 | |
| Mega | 39M | 1.02 | |
| Sparse Transformer (30L) | 95M | 0.99 | 0.992 ± 0.001 over 3 seeds |
| Compressive Transformer (24L) | — | 0.97 | |
| GPT-2 (zero-shot) | 1542M | **0.93** | *not 0.94 — that value is a downstream mis-citation* |
| cmix v13 | — | 1.23 | compression program, not a gradient-trained LM; its mixer is itself neural, so "non-neural" is wrong |
| "Focus" adaptive-IIR | 22M | 0.94 | **extreme unreplicated outlier** in a 4-row table; beats T-XL at 277M. Do not build an argument on it |
| **SpikeGPT (spiking)** | **46.1M** | **1.283** (ctx 1024) / **1.262** (ctx 3072) | train 1.113 / 0.903; firing rate 0.15; ~48 GPU-hours |
| vanilla RWKV (SpikeGPT ablation) | ~46M | 1.201 | `(unverified — this ablation table was not independently audited)` |
| naive "Heaviside RWKV" (ablation) | ~46M | 1.403 | `(unverified — same caveat)`; prices *careless* binarisation at ~0.20 bpc |

### text8 (27-symbol alphabet; from `fil9` ← enwik9, **not** enwik8)

| Model | Params | Test bpc |
|---|---|---|
| vanilla RNN (Bai et al.) | ~5M | 1.69 |
| GRU (Bai et al.) | ~5M | 1.53 |
| **LSTM (Bai et al.)** | **~5M** | **1.50** |
| **TCN (5L, k=2, 520 hidden)** | **~5M** | **1.45** |
| LSTM (2000 units) | ~16M *(researcher's own count, not stated in paper)* | 1.43 |
| BN-LSTM | ~16M | 1.36 |
| HM-LSTM | 35M | 1.29 |
| RHN | 45M | 1.27 |
| mLSTM | 45M | 1.27 |
| T12 | 44M | 1.18 |
| T64 | 235M | 1.13 |
| Adaptive attention span (small) | 38M | 1.11 (dev 1.05) |
| Transformer-XL (24L) | 277M | 1.08 |
| Adaptive attention span (large) | 209M | 1.07 |
| GPT-2 (zero-shot) | 1542M | 0.98 |

The 5M Bai et al. row is the **single most useful line in this survey for a 5.04M-param model**, with two riders: those baselines were **grid-searched** (optimizer, recurrent dropout 0.05–0.5, LR, clipping, forget-gate bias) at matched size, so 1.50 is a *tuned* LSTM, not a strawman; and 1.50 is nonetheless a weak absolute bar, since Al-Rfou's own table cites a plain LSTM at 1.43.

### Penn Treebank, character level (5017k / 393k / 442k chars, ~50 symbols)

| Model | Params | Test bpc |
|---|---|---|
| vanilla RNN | ~3M | 1.48 |
| LSTM (1000 units, Cooijmans) | — | 1.38 |
| GRU | ~3M | 1.37 |
| LSTM | ~3M | 1.36 |
| BN-LSTM (1000 units) | — | 1.32 |
| TCN (3L, k=3, 450 hidden) | ~3M | 1.31 |
| LN HyperLSTM (2L) | 14.4M | 1.219 |
| **FS-LSTM-4** | **6.5M** | **1.193** ← *closest published point to a 5M budget on any standard char benchmark* |
| AWD-QRNN (6L) | 13.8M | 1.187 |
| AWD-LSTM (3L) | 13.8M | 1.175 |
| LSTM (d2) | 24M | 1.143 (1.103 dyneval) |
| Mogrifier LSTM (d2) | 24M | 1.131 (1.088 dyneval) |

### Non-char spiking anchors (perplexity, for calibration only)

| Model | Params | Corpus | PPL |
|---|---|---|---|
| SpikeGPT | 213–216M | WikiText-103 | 39.75 |
| SpikingSSM | 75M | WikiText-103 | 33.94 |
| DSN (2026 preprint) | — | WikiText-103 | 28.50 |
| Transformer (non-spiking) | 231M | WikiText-103 | 20.51 |
| S4 (non-spiking) | 249M | WikiText-103 | 20.95 |
| EGRU (graded events, lateral W) | — | PTB word / WT-2 | 57.0 / 68.9 |

### Prior in-house result

**1.2362 bpc @ 5.04M params on a private 639 MB modern-prose corpus, 99-symbol ASCII vocab, 256-char windows.** This belongs in **no block above.** It is comparable only to another model scored on the identical corpus file with the identical protocol string.

---

## 8. Design implications for this project

Ranked by expected bpc-or-GPU-hour return per unit of risk, each tied to the hardware.

**1. Build the jiterator-fused LIF and make it the default path.** §6.1 overturns the brief's "custom kernels are impossible" premise: NVRTC ships inside the torch wheel, compiles elementwise CUDA C++ in ~0.07 s, and gives a reproduced **~4.0–4.4×** on the LIF loop, composing with CUDA-graph capture. The entire neuron step (decay, integrate, threshold, reset, surrogate save) is elementwise and fits the ≤8-in/≤8-out limit. Wrap it in `torch.autograd.Function` with a second jiterator kernel for the surrogate backward; guard with a `--no-fused` fallback and an automated equivalence test against eager (expect ~5e-07 fp32 agreement) — **jiterator is a beta API with no autograd safety net.** Boundary: **elementwise only.** Sparse gather/scatter backward (§6.10) remains dead.

**2. Fix the LIF parameterisation before re-running any timescale experiment.** snnTorch's `mem = β·mem + I` has DC gain 1/(1−β) (§4.2). Replace with `mem = β·mem + (1−β)·I` — ~8 lines, preserves binary spikes, leaky integration, hard threshold, surrogate BPTT, no lateral matrix. **The TIER6 negative results for `--learn-beta` (+0.211 bpc) and `--beta-per-neuron` (+0.090 bpc) were obtained under the unnormalised form and do not falsify heterogeneity**, which §3.2 predicts should *help* on a temporally rich task by +10 points-equivalent on the closest analogues. Re-run with normalisation, and add **fixed geometric β** as a third arm.

**3. Buy horizon with zero parameters: geometric β grid + HGRN layer-wise floor.** §4.9's RetNet grid `γ = 1 − 2^(−5−h)` maps to per-character 1/e horizons of 31…4095 characters; HGRN's monotonically increasing lower bound (its largest single ablation gain) maps to per-character retention floors of ~0.0 / 0.9 / 0.99 for layers 1/2/3. Zero extra parameters, zero extra FLOPs, no kernel. This directly attacks §4.1's ~3–10 character horizon and is the cheapest lever in the entire survey. §3.2 says heterogeneous **initialisation** is where the robustness lives, so do this *before* making β learnable.

**4. Default to T=1 direct coding; bound the ablation at T ∈ {1,2,4}.** Four converging arguments: SpikeGPT's only competitive LM result uses T=1 (§1.2); at β=0.9 the gradient horizon is 65.6 characters at T=1 versus 13.1 at T=5 (§2.6); T multiplies kernel launches on a 12.8 µs-launch box (§6.2); and hoisting the input GEMM is worth **4.5×** only when micro-steps share one input current (§6.6). Encoder literature caps the useful T at 4 (§1.5), so stop there. **Use the right memory argument:** T is *not* a large activation-memory multiplier (§6.12 — quadrupling T cost 4%, doubling L cost 8%). If T>1 is kept, **re-parameterise β to hold per-character retention constant** (`β_T = β_1^(1/T)`; 0.9 at T=1 ↔ 0.979 at T=5) or the T ablation is silently also a horizon ablation.

**5. Add a per-neuron adaptive threshold (ALIF) as the main architectural bet.** One extra scalar state with its own learned per-channel decay, initialised slower than the membrane. It is **not** a lateral recurrent matrix, so it is legal; SE-adLIF is the strongest evidence that adaptation can substitute for explicit recurrence (§3.3, SHD 95.81 vs recurrent LIF 90.27); SFA is what moves temporal-credit tasks from near-chance to near-human (12AX 97.79% vs 0.39%). Cost: one extra fp32 state tensor per layer per micro-step — **budget it against 8 GiB before committing.** If sub-threshold (current-coupled) adaptation is used, use **Symplectic Euler ordering** (update w with the *new* u) — Forward Euler is documented to go unstable, and over billions of characters that surfaces as an unexplained divergence. **Do not sell this as LSTM-matching**: LSNN is 96.4% vs LSTM 98.0% on sMNIST and 33.2% vs 29.7% PER on TIMIT.

**6. Take the bf16 GEMM win; keep membrane state and the threshold comparison in fp32.** Measured **3.00×** on the dominant sequential GEMM (§4.7), orthogonal to every architectural change, no code restructuring. But bf16's representable spacing at v=1.0 is **0.78% of the threshold** (§6.8) — 8× coarser than fp16 — so threshold comparisons flip and the surrogate is evaluated at a quantised distance. Cast only the fc_in/readout GEMMs; the elementwise path is bandwidth-bound so fp32 there costs almost nothing.

**7. Push batch to the VRAM limit rather than tuning it for speed.** 128× the work for 1.18× the time (§6.5). Highest effort-to-payoff lever available and requires zero code restructuring. Do **not** use the "B·H ≈ 250K" rule — it is unsupported; the L2 absorbs far more than the 383 GB/s figure predicts.

**8. Design the training step for CUDA-graph capture from day one.** Fixed batch and sequence shapes, pre-allocated static state tensors, `opt.zero_grad(set_to_none=False)`, `Adam(capturable=True, foreach=True)`, no `.item()`/`.cpu()`/data-dependent control flow, no dynamic-shape final batch (drop it), warm up on a side stream before capture. Retrofitting is far more expensive than designing for it. **Expect ~1.1–1.2× at target scale, not 10×** (§6.3, §6.4) — the microbenchmark speedups evaporate once GEMMs dominate. Write the neuron so its entire step is graph-capturable: no data-dependent control flow, static shapes. **Note channel-wise PSN's C×k kernel is length-independent and capture-friendly; PSN's T×T is not** (§3.5).

**9. Run reset removal as a first-class measured ablation, not an assumption.** It is a declared free parameter, it is what unlocks the **exact** triangular-Toeplitz scan (65.5× at L=256, error 1.9e-06, §6.7) and every parallel form (§4.3), and PSN prices it at **−2.0 points** on sequential CIFAR-10 (81.50 → 79.50). Ladder: (a) detach-reset — backward-only, free, predicted null at a peak-1 surrogate; (b) soft/subtract; (c) no reset + learnable per-neuron threshold + firing-rate regulariser. **Published soft-vs-hard deltas are small and contradictory** (AR-LIF: CIFAR-100 hard 73.4 vs soft 63.4, but CIFAR10-DVS hard 83.8 vs soft 84.1) — import the mechanism, not the number. **If you go parallel, check underflow**: `0.9^1023` is subnormal in fp32, silently truncating context past ~L=830.

**10. Do not spike everything; define precisely which tensors are binary — that definition IS the contribution.** Every competitive spiking LM keeps embeddings, LayerNorm/PowerNorm, the state-decay accumulation and the output head in float, emitting binary spikes only on layer-to-layer activations (§1.2, §3.6). The counter-examples are stark: naive LIF-BERT 54.9 vs SpikeLM 76.5 GLUE; SpikeDecoder 81.7% vs 98.5% for its ANN twin. **Spike-count logits at T=5 give only 6 distinguishable levels — a hard bpc floor on a ~99-way softmax that no neuron tuning recovers.**

**11. Set the headline target against a same-size LSTM, on a standard corpus, with an in-house control.** Add **text8** as the primary comparability benchmark at the exact byte-offset split; the published 5M yardsticks are **LSTM 1.50, TCN 1.45** (§7). Use **PTB-char** (5.9M chars, well under an hour per run) as the cheap ablation harness for T and reset, where FS-LSTM-4 at 6.5M = 1.193 is the nearest anchor. **Train an identically-sized LSTM/GRU and a small Transformer on the exact same tokenization, split, schedule and eval script** — without that control the prose number supports no claim at all. And **stop presenting 1.2362 bpc as commensurable with enwik8 rows** (§5.5).

**12. Fix the statistical and evaluation protocol before running ablations.** ≥3 seeds per arm with the **same seed set across arms** (paired); report mean ± std, not the best run; moving-block bootstrap over ≥100 contiguous test blocks, not per-character (losses are strongly autocorrelated). **Measure our own seed std** — do not import Melis' word-level figure, whose conversion to "±0.01 bpc" is loose and whose source was misattributed (§5.7). Give every arm an **equal, pre-declared HP-tuning budget** and report it — Melis showed tuning effort dominates architecture in exactly this literature. Report **both** protocols for every headline number: state reset per window, and state carried contiguously (§5.4). Never place a dynamic-eval number in a static column (§5.6).

**13. Fix the surrogate CLI, then sweep width — not shape.** Keep `atan(α=2)` as default (FWHM 0.6366·V_th, inside LSG's good band, bit-identical to SpikingJelly's ATan). **Renormalise or delete snnTorch's `sigmoid(slope=25)`** — peak 6.25, precisely the unnormalised surrogate Zenke & Vogels show degrades learning through implicit reset recurrence. A shape ablation at defaults measures a **19× width mismatch**, not shape; for a fair test use `fast_sigmoid(slope=1.30)`. When sweeping α, note the ATan **peak is α/2**, so `{0.5,1,2,4,8}` confounds peak with width — divide by α/2 to isolate width. **Lock V_th = 1.0** for all surrogate work (§2.8). Log **mean firing rate per layer alongside bpc** on every run: width moves sparsity independently of accuracy, and firing rate is the leading indicator of a dead run.

**14. Report firing rate as a first-class metric and label energy claims exactly.** Literature working points: SpikeGPT 0.15, SpikingSSM 0.06–0.15 on LRA (<0.30 on WikiText-103), Sorbet 0.13–0.15, SpikeDecoder 0.10–0.36. The project's own TIER7 rates are **1.6 / 1.3 / 13% per layer** (note TIER7 self-contradicts, elsewhere claiming 4–24% — resolve this before publishing), so layer 3 is in the literature band and layers 1–2 are an order of magnitude below it. State plainly that **spike sparsity produced no wall-clock or energy benefit** and why (§6.9). Any pJ figure: "hypothetical 45nm-ASIC projection, Horowitz 4.6 pJ MAC / 0.9 pJ add, arithmetic only, data movement unaccounted." Cite SpiNNaker2 (§1.9) as the one honest hardware point: **18.3× less energy, 8.6× slower**.

**15. Explicitly ruled out.** *By hardware:* Triton/Inductor (`TritonMissing`, reproduced); sparse-index backward — Sparse Spiking Gradient Descent's 150× needs compiled gather/scatter (nvcc + MSVC), and **jiterator cannot substitute**; SLAYER/EXODUS backends; mamba-ssm (**Linux-only**, the real blocker); GLA/FlashLinearAttention (Triton); Sorbet's bit-shift softmax at T=16 (worthless without low-level kernels); cupy (second CUDA memory pool contending for 8 GiB, duplicates jiterator). *By budget:* SpikeLLM (post-training quantisation of a pretrained 7B); anything at ≥0.9B scale; matching SpikeGPT-46M (~48 GPU-hours). *By invariant:* ternary/bi-directional/multi-bit inter-layer spikes (SpikeLM, Ternary Spike). *By evidence:* porting OTTT/SLTT/e-prop/OTPE/FPTT — they buy memory truncated BPTT already provides at O(chunk), none beats BPTT on accuracy, and per-timestep updates break graph capture. *Borderline, legitimate only with verification:* SFA-style integer-training-then-binary-unfolding, framed as a training-time reparameterisation with the unfolding actually performed and checked.

---

## 9. What the literature does NOT settle

**1. The bpc-vs-parameters curve for spiking LMs below ~45M is unmeasured.** The nearest points are SpikeGPT at 46.1M (1.262–1.283 on enwik8) and SpikeDecoder at 1.4M on a 100K-char toy corpus reporting *training-set* accuracy. There is **no published spiking char-LM in the 5–20M range on any standard corpus.** Any target we set — e.g. beating the 5M text8 LSTM at 1.50 — is an extrapolation and must be validated by our own run against our own dense control.

**2. Whether T>1 buys anything at all when the token sequence already supplies a time axis.** Every published T-ablation (SpikeBERT T=4 optimal, degrading at T=8/12; SpikeDecoder T=4 vs T=2; DIET-SNN's 150→5 ladder) is on encoder or vision-shaped models where T is the *only* temporal dimension. **The T=1 vs T=4 comparison for a causal spiking char LM has not been run.** It interacts with the hardware in a way no paper measures: each micro-step is another ~12.8 µs launch chain.

**3. What the network-level memory horizon actually is, and whether long-timescale neurons are usable.** §4.1's arithmetic gives 3.2 chars for one integrator and ~9.5 for a 3-layer stack — a 3× disagreement in the number driving the whole prioritisation. **Settle it empirically before any architecture change:** score the existing checkpoint with membrane state force-reset every k characters for k = 2, 4, 8, 16, 32, 64. Flat past k=8 confirms the horizon diagnosis; still improving means something else (LayerNorm statistics, depth) carries more context than the leak arithmetic suggests. Compounding this: TBPTT truncated at 64 characters **cannot supply gradient signal past ~64 chars**, so a neuron with a 1000-character horizon gets no direct learning signal for its tail. **Fixed (frozen) geometric β may be necessary precisely because gradient descent would otherwise shrink it** — untested, and it interacts with chunk length in a way no paper measures.

**4. Whether the arctan-vs-fast-sigmoid gap on long binary sequences survives width matching.** Stan & Rhodes report >10 points (82.00 vs 69.83 on sCIFAR10; 61.2 vs 51.7 on Path-X) at what are almost certainly snnTorch defaults — a **19× width and 12.5× gradient-mass** mismatch. **No width-matched shape comparison exists anywhere in the literature.** `atan(2)` vs `fast_sigmoid(1.30)` at equal HWHM would settle it and is publishable either way.

**5. What (β, T) pair is bpc-optimal at fixed compute for a character-level SNN.** No published sweep exists at this scale or task. Specifically unknown: whether pushing per-character retention to 0.96–0.99 to buy a 30–140 character horizon improves bpc or just drives neurons into tonic saturation and denser firing. **Leak, firing rate and surrogate width all interact, and the literature reports only two of the three at a time.**

**6. Whether soft, hard, or no reset is right for text.** Every comparison we have is vision and they contradict each other (AR-LIF: hard +10.0 on CIFAR-100 but −0.3 on CIFAR10-DVS; PSN: soft+detach optimal; SpikeGPT: hard, in the only working LM configuration; PSN prices no-reset at −2.0 points on sequential CIFAR-10). **No character-level bpc number exists for a reset-free spiking neuron.** This is the highest-value single experiment because the 65× triangular-scan speedup hangs on it.

**7. Whether SFA-style integer-training-then-binary-unfolding transfers from vision to char LM at small scale — and critically, whether the unfolded binary-inference model actually matches the integer-trained model's bpc.** The equivalence proof is for an IF neuron with soft reset; whether the *unfolding costs accuracy* on a language task is untested. This is the highest-value untested lever available inside the invariants.

**8. How much of the binary-spike penalty is recoverable without an ANN teacher.** SpikeBERT, SpikingBERT, SpikeLM and BiSpikCLM all distill. SpikeGPT does not — and is also the one with the largest measured gap to its ANN baseline (0.146 bpc). **Whether from-scratch surrogate BPTT can close that gap, or whether we would need to train and distill from our own same-size ANN, is untested at any scale.** Related and equally open: how much bpc binarity costs at *our* scale versus a ternary or 2-bit inter-layer control (which would be a non-shipping control, not a deliverable).

**9. The sparsity/bpc frontier for language.** Nobody publishes a curve of bpc versus mean firing rate for a spiking LM under an explicit firing-rate penalty; all the firing-rate-regularisation literature located is vision or time-to-first-spike coding. We must sweep the penalty weight ourselves. **And it is unresolved whether this project's own 1.3–1.6% rates in layers 1–2 are buying regularisation or starving the representation** — which directly determines whether reset-free neurons (which fire more) will help or hurt.

**10. The true carried-state versus reset-state gap for a contractive LIF membrane.** Quantified for LSTMs and fixed-context Transformers; **never measured for a leaky membrane with no lateral weight matrix**, where the effective memory may be so short the gap is near zero. That result would itself be a substantive finding about what the spiking state actually retains.

**11. The seed-to-seed standard deviation of bpc for surrogate-gradient SNN training specifically.** The ~±0.01 bpc working figure is derived from Melis' *word-level LSTM* variance, was misattributed in the source material, and may be far too optimistic given hard thresholds, non-differentiable firing and gradient mismatch. **3–5 identical-config runs, measured directly, before any ablation is believed.**

**12. Whether an exact parallel form of soft-reset LIF exists for β < 1.** For β = 1 with non-negative input the soft-reset IF neuron has a closed form (spike count = `floor(cumsum(I)/threshold)`), exact and parallel. **No paper found extends this to leaky β < 1** — all leaky methods are approximate (SDN, PMBC) or reset-free. Worth 30 minutes of derivation before accepting an approximation and its accuracy cap.

**13. Whether bf16 GEMMs degrade surrogate-gradient quality or firing-rate stability over a 30-hour run.** The 3.00× is measured on isolated GEMMs; the interaction with a hard threshold at 1.0 and an atan surrogate over ~200k updates is uncovered by the literature, and the project's own TF32 experiment already showed slightly worse loss at 200 updates. Needs a 2000-update parity arm, not a spot check. Related open systems questions: how much VRAM a graph-captured step permanently pins (and therefore the real maximum batch under capture on 7.96 GiB); whether `torch.utils.checkpoint` — which manipulates RNG state and re-enters autograd — composes with graph capture at all; and whether fp16 + `GradScaler` is capturable, given that the scaler's inf/nan check is a device-to-host sync.

**14. A project-definition question that is not empirical and should be decided before implementation:** does "no explicit lateral recurrent weight matrix" admit (a) PSN/channel-wise-PSN-style **temporal** weights — a learned per-channel kernel over micro-steps or characters, diagonal in the neuron axis; and (b) BRF-style **complex** membrane state, i.e. a 2×2 block-diagonal rotation per neuron? Both are diagonal in the neuron dimension and introduce no neuron-to-neuron matrix. (a) is what recovered 5–10 points on sequential CIFAR (81.50 → 86.70 / 88.45 / 91.17) *and* makes the neuron parallel-scannable, sidestepping the launch-latency wall. **Write the answer down explicitly, in advance, either way.**

---

### Claims retracted, corrected or flagged (audit trail)

**Removed as fabricated:** SpikeGPT author "Blum" (correct list: Zhu, Zhao, Li, Eshraghian); SpikeGPT train bpc 0.864 at ctx 3072 (correct: 0.903).

**Removed as unsupported/unsourced:** the soft-vs-hard reset vision comparison (94.44/94.34, 82.20/81.40) — no citation; FPTT's DVS-Gesture memory triple (10.8 / 13.2 / 5.6 GB) — not in the cited tables; PSN's "~10–20× inference, ~5–15× training" speedup range — appears nowhere in the paper; the "~200 pJ off-chip / ~1.5 pJ MAC / ~130×" energy figure — not a Horowitz number (his 45nm DRAM read is 640 pJ) and unsourceable; the "~90–95% sparse-BLAS breakeven" — folklore contradicted by published breakevens of 50–78%; SLTT's "0.65–0.95 cosine similarity" band — the CIFAR-10 panel axis extends to 0.35.

**Misattributed, corrected:** the 30 ms / 300 ms / 1200 ms SFA figures belong to DEXAT (Shaban, Bezugam & Suri, *Nat Commun* 12, 2021), not LSNN/Salaj (which use τ_a 700–13500 ms); the VGG16 137× BPTT slowdown is Rathi et al. ICLR 2020 **at T=100**, not the cited PMC source; the "0.2 / 0.7 perplexity variance" figures are Mogrifier ICLR 2020 Appendix B (PTB word-level), not Melis ICLR 2018 (whose figures are 0.4 / 0.5); the 12.5% tensor-core utilisation figure is Libra (arXiv:2506.22714, H100), not the cited sparsity papers.

**Values corrected:** LSNN sMNIST 96.4% (98.0% is the LSTM baseline) and TIMIT 33.2% PER vs LSTM 29.7%; Mogrifier 48M enwik8 **test** 1.146 (1.135 is validation); HyperLSTM enwik8 1.34 at 27M; GPT-2 enwik8 **0.93**; ASGL DVS-CIFAR10 **84.50%** (85.50 is a GitHub README figure); tdBN ImageNet 67.05% is ResNet-34 **large** (plain is 63.72%) and folds into conv weights, not the threshold; BRF reaches 95% of final accuracy in **five** epochs; E-SpikeFormer 86.2% is a 384×384 result; SpikeDecoder's 81.7/98.5 are **training-set** accuracies with a non-identical ANN twin; SpikingSSM LRA 84.33 / Path-X 94.82; DIET-SNN is TNNLS **2023** 34(6):3174–3182; GLA authors are Yang, Wang, Shen, Panda, Kim; text8 derives from **enwik9** via fil9; SLTT is not "under one third memory" on CIFAR-10/100 and is 0.28 below BPTT on ImageNet; SpikingJelly's `detach_reset` defaults to **False**.

**Measured claims corrected on re-run:** the "3.48× end-to-end" headline (real figure ~1.1–1.3×; the eager baseline was ~3× too slow and the benchmark model had no cross-token recurrence); fused-LIF ~4.0–4.4× (not 5.08–5.34×); graph-alone on the realistic model 1.11× (not 1.63×); Hillis-Steele scan 49.5 ms (not 109 ms); BPTT memory does **not** scale in the T×L product — T=4→T=1 saves ~4%, not 4×; the B·H batching crossover rule is unsupported (effective bandwidth ~456 GB/s via L2).

**Flagged `(unverified)`:** SpikeGPT's venue (ICLR 2025 poster vs "Accepted by TMLR" — conflicting evidence); "the only published spiking char-LM bpc"; SpikeGPT's RWKV/Heaviside binarisation ablation table (1.201 / 1.403); DH-LIF/DH-SNN benchmark deltas; Cover & King's 1.25–1.35 bits/char; bf16 spike-count drift (−0.25% / −0.01%); cupy's behaviour on sm_120; Biderman et al.'s coverage of window-fragmentation bias; enwik8's "markup is free to predict" explanation (folklore, unmeasured, and partly confounded by text8's different lineage); the "Focus" 22M = 0.94 enwik8 row (real but an unreplicated extreme outlier).