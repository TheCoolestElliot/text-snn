# EXP_018 — Is a broadcast reward prediction error worth anything to a channel-diagonal spiking LM?

**Pre-registered:** 2026-08-13, after the calibration in §2.6 and the systems
probe in §2.5 — **and before any arm was trained to completion**. The arm's code
(`src/snn/dopamine.py`, `DopamineCharLM`) was written alongside this file. That
sequencing is stated rather than claimed as "before any code existed", because it
was not — `EXP_013`'s header is the precedent for saying so plainly.

**Full disclosure of what was seen before this file was committed**, because a
hypothesis written after a peek is worth less and the defence is the record, not
the assurance. A 300-step systems probe (§2.5) ran to answer two questions that
had to be answered before the design could be fixed at all: *does CUDA-graph
capture survive a `no_grad` pass nested inside the captured region*, and *what
does the second pass actually cost*. It printed a step-299 training loss for each
arm as a side effect:

| | step-299 train loss, seed 0, 300-step cosine |
|---|---:|
| `snn` | 1.9966 |
| `dopamine` mult/rpe | 1.9873 |
| `dopamine` mult/rolled | 1.9973 |
| `dopamine` add/rpe | 1.9860 |

Those four numbers are recorded here so that nothing below can be tuned to them
without it being visible. **They are not evidence and no bar below is set against
them**: they are one seed at 1.5 % of a training run, on a cosine schedule
truncated to 300 steps, at a point where §2.6's own calibration says the dopamine
signal is still ~46x weaker than it will be at convergence. No prediction in §4
was changed after they were seen.

**Status:** PRE-REGISTERED — no results yet.
**Phase:** 4 (controlled experiments).

**What it costs:** sixteen training runs at the committed recipe
(anchor n=5, arm A n=5, control S n=3, arm B n=3) plus sixteen full-split test
evaluations. At the **measured** per-step costs of §2.5 that is
**~2.9 GPU-hours** of training. §7.1's GPU-hour estimates have been unreliable in
this project (one rank was measured at 3.1x its estimate and the constant they
descend from was formally withdrawn), so this figure descends from a measured
ms/step on this box and not from the cost model.

---

## 0. What this experiment is, and what it is not

**It adopts nothing, ranks nothing, and closes no open decision.** It does not
edit `docs/reports/04_phase4_interim.md` §8's table, does not advance the phase,
and does not touch any committed kernel, gate, tolerance or figure.

**It is not on the candidate record.** `docs/reports/data/phase3_candidates.json`
enumerates all eighteen candidates — fourteen ranked, one refuted, three referred
— and none of them is this. There is no prior EXP log, no ruling and no adoption
decision. This file opens a candidate rather than running one.

**It has no prior in the literature review either, and that was checked rather
than assumed.** `docs/reports/01a_literature_review.md` contains **zero**
occurrences of `dopamin`, `neuromodul`, `reward`, `three-factor`, `eligibility`,
`STDP`, `plasticity`, `homeostat`, `surprise`, `prediction error`,
`reinforcement` or `novelty`. The arm can cite no literature prior in this
project's own reviewed record, and it does not pretend to.

**What it is NOT, stated because a neighbour is explicitly ruled out.**
`01a_literature_review.md` §8 item 15 rules out, *by evidence*, porting
`OTTT / SLTT / e-prop / OTPE / FPTT`: they buy memory that truncated BPTT already
provides at O(chunk), none of them beats BPTT on accuracy, and per-timestep
optimiser updates break the CUDA-graph capture this box's throughput depends on.

**This arm is not a three-factor learning rule and does not touch the learning
rule at all.** It is a **forward** neuromodulatory signal — an extra input to the
forward pass — trained by ordinary BPTT through ordinary autograd, with no
eligibility trace, no per-timestep update, and no change to the optimiser. The
prohibition is scoped to *replacing BPTT with a trace rule* and is not reopened
here. If it were, this file would not exist.

**It also does not pre-empt ranked design implication #5**, the per-neuron
adaptive threshold (ALIF), which that review calls "the main architectural bet".
ALIF's adaptation is driven by *the neuron's own spikes*, locally. This is driven
by *the population's prediction error*, globally. They are complementary
mechanisms and this file claims nothing about #5, measures nothing about it, and
leaves it exactly where it was.

---

## 1. The question

> A spiking model constrained by I5 has no way for any neuron to know how wrong
> the population's last prediction was, because finding out requires reading all
> `d` channels out through the head. **Is that information worth anything, and if
> so, is it worth the two forward passes it costs to deliver?**

---

## 2. The derivation, done before any measurement

### 2.1 The signal, and why its baseline is the entropy

Dopamine encodes a **reward prediction error**: how much better the outcome was
than predicted. For a character LM the outcome at position `t` is the character
`x_t` and its prediction was made at `t-1`:

```
S_t   = -log p_{t-1}(x_t)      the surprise: how bad the outcome was      (nats)
H_t   =  H(p_{t-1})            the model's OWN expected surprise          (nats)
phi_t =  H_t - S_t             the reward prediction error                (nats)
DA_t  =  tanh(phi_t / tau)     the bounded neuromodulator,  DA_0 = 0
```

`phi_t > 0` means the character observed was **less** surprising than the model
expected — "better than predicted", the classic dopamine burst.

A prediction *error* needs its prediction subtracted, and the obvious choice is a
running mean of recent surprise. **It is the wrong one here, twice, and the
derivation is what fixed the design:**

1. `E_{x~p}[-log p(x)] = H(p)` **exactly**. The model's own predictive
   distribution already states what it expects the surprise to be. So `phi` is
   zero-mean under that distribution **by construction** — no subtracted
   baseline, no EMA, no learned critic, and nothing to tune. It is an error, not
   a magnitude.
2. An EMA over `t` is a **sequential scan**: `L` kernels per layer, on the one
   code path this architecture exists to keep at one kernel per timestep, and a
   data-dependent recurrence inside the region `snn.train` captures into a CUDA
   graph. The entropy is a single parallel reduction over the vocabulary axis.

This is the project's standing rule that a derivation done before implementing is
allowed to change what the experiment is (`EXP_008` §1.1 is the precedent,
`EXP_013` §1.2 the most recent case). Here it removed the arm's only free
hyperparameter other than `tau`.

**Stated as a claim that could be false:** `phi` is zero-mean under the model's
own belief but **not** under the data. Its measured mean is **+0.05 to +0.09
nats** across four checkpoints (§2.6), not 0 — the model is systematically
slightly *less* surprised than it predicts it will be, which is what a
well-calibrated-but-not-perfectly-calibrated model looks like. The arm does not
depend on the mean being zero; it is recorded because a reader is entitled to
check the derivation against the measurement.

### 2.2 Causality: `phi` cannot see the target, and this is the one thing that must not be wrong

`snn.data` builds each window as `x = block[:, :-1]`, `y = block[:, 1:]`, so
`y[:, u] == x[:, u+1]`. `phi_t` is assembled from `logits[:, t-1]` and `x[:, t]`:

```
phi_t  depends on  x_{<=t}                the target at position t is  x_{t+1}
```

**`phi_t` uses only characters the model has already read, and never the
target.** `y` is not an argument to anything in `snn/dopamine.py`, and
`DopamineCharLM.forward` takes `idx` alone.

An off-by-one in that shift is the single failure this arm can have that would
still train, still produce a plausible bpc, and be a leak. It is guarded twice
and both guards are abort-on-fail: gate **G2** asserts `DA[:, :t+1]` is bitwise
invariant to any perturbation of `x[:, t+1:]`, and `018_calibrate_da.py`'s C1
asserts the gathered surprise equals `F.cross_entropy` on the committed targets
elementwise. **C1 has already run and passed at exactly `0.000e+00` on all four
checkpoints** (§2.6).

### 2.3 It is not a reparameterisation, and `EXP_008` is why that sentence is worth writing

The modulation acts on the input current, once, for the whole sequence, before
the scan starts:

```
mult:  cur'_{t,c} = cur_{t,c} * (1 + k_c * DA_t)
add:   cur'_{t,c} = cur_{t,c} +       k_c * DA_t
```

`EXP_008`'s learned per-channel threshold is the **constant** case of the
multiplicative form. Its Identity 1 makes a constant per-channel gain exactly a
threshold rescale; its Identity 2 folds that gain into `layers.k.weight`, which
is already a free parameter. Its 0.0590 bpc is therefore provably an
**optimisation** effect on an unchanged function class — a different trajectory
under AdamW, not a larger reachable set.

**`DA_t` varies with `t`. It folds into no weight, at any value of `k`.** A
`[1, d]` gain multiplied by a signal that moves every timestep is not
representable by any fixed `W`, because the Phase-2 LIF's only time-varying
quantity is its own membrane and `DA` is not a function of it. So this arm's
function class is **strictly larger** than the baseline's, and a difference it
makes may be a capacity effect rather than only an optimisation one.

There is deliberately **no `fold_into_*` method** on `DopamineCharLM`, and its
absence is the claim. Decision #6 (the fold-in tolerance) does not arise for this
arm at all.

### 2.4 Why the signal has to be broadcast, and the invariant question this raises

Computing `phi_t` requires reading all `d` channels out through the head. **I5
forbids exactly that inside the recurrence.** No neuron in this architecture can
compute its own population's prediction error, at any width, at any depth. A
single broadcast scalar is the cheapest legal way to hand it back.

**I5 status: NOT RULED, and this file does not rule it.**

* The **parameters** pass §4.6's stated test: `k` is `[1, d]`, diagonal in the
  channel axis, `O(1)` per neuron, no cross-neuron mixing among the weights.
* The **pathway** does not obviously pass its spirit: `DA` is a rank-1
  all-to-one-to-all coupling, and the §4.6 ruling was written about parameters.

Three facts belong in the referral and none of them decides it: the readout
borrowed is the **head**, which the invariants already exempt as fp32 and
non-spiking; the pathway carries **one scalar**, not a learned `[d, d]`; and
`01a_literature_review.md` §4.4 — "Data-dependent decay preserves associativity —
the key permission slip" — records that a modulatory scalar entering a diagonal
recurrence "keeps the state transition diagonal … and introduces no lateral
recurrent weight matrix". That review passage is about a *feedforward-driven*
modulator, not a head-driven one, so it is adjacent evidence and not a ruling.

This is the same posture `TokenShiftCharLM` holds and for the same reason: the
class exists so the question can be **priced**. Every table this arm appears in
labels it a diagnostic. §10 refers the ruling.

### 2.5 The two passes: the cost, measured in advance, and the alternative not run

`DA_t` needs the head at `t-1`, and the head is downstream of every layer. This
architecture evaluates sequentially in *depth* precisely because layer `k` at
time `t` depends only on layer `k-1` at time `t` (spec §0). A head signal at
`t-1` breaks that — layer 0 at `t` would need layer `K-1` at `t-1`, and the time
loop would have to contain a GEMM, which is `K*L` GEMMs instead of `K`.

So the stack runs twice: pass 1 under `no_grad` produces the logits `DA` is built
from; pass 2 consumes `DA` and is the one the loss and the graph see.

**Measured on this box** (B=128, L=256, d=512, K=2, RTX 5060, capture on, eval
off, 290 timed steps after 10 warm-up, seed 0), against the `snn` baseline at
21.74 ms/step **on the same tree in the same process**:

| arm | ms/step | ratio vs `snn` | captured | peak VRAM |
|---|---:|---:|:--:|---:|
| `snn` | 21.74 | 1.000 | yes | 0.609 GiB |
| `dopamine` mult / rpe | 35.65 | **1.640** | yes | 0.922 GiB |
| `dopamine` mult / rolled | 35.74 | 1.644 | yes | 0.984 GiB |
| `dopamine` add / rpe | 32.81 | 1.509 | yes | 0.859 GiB |

Two things that had to be known before the design could be fixed, and were:

* **CUDA-graph capture survives a `no_grad` pass nested inside the captured
  region.** All three variants captured. Had they not, the arm would have cost
  ~11x rather than ~1.6x and this experiment would not be worth running.
* **The second pass costs 0.64x a full training step, not the ~0.33x a
  "forward is a third of forward+backward" estimate would give.** The head GEMM
  over `V = 205` and the `[B, L, V]` log-softmax are not free. The estimate would
  have been wrong by a factor of two, which is why §3's budget descends from this
  table.

**Unlike `EXP_013`'s noise, this arm exists at inference and costs there too.**
There is nothing to switch off at `model.eval()` and nothing to fold away.

**The alternative that is NOT run here, named so it is on the record:** `DA` is
computed by the *unmodulated* pathway, which is a different function from the
modulated one once `k != 0`. Iterating the two passes to a fixed point would make
the signal self-consistent. It is not run, on cost — each additional iteration is
another 0.64x — and it is named rather than silently omitted. This is the
`EXP_017` §2.5 pattern.

**`DA` is detached.** Pass 1 is under `no_grad`, so no gradient flows back
through the surrogate a second time. Deliberate twice over: the fused backward is
not twice-differentiable and would be *silently* wrong under `create_graph`
(`snn/neuron.py`'s backward raises on the eager path and cannot on the fused
one), and a neuromodulator delivered by a separate system rather than
backpropagated is the correct biology.

### 2.6 Calibration — measured before this file fixed `tau`

`scripts/exp/018_calibrate_da.py`, committed with its output at
`docs/reports/data/exp_018_da_calibration.json`. It reads only **committed
checkpoints** and no arm, so nothing in it is evidence for or against the
hypothesis below. `013_calibrate_noise_scale.py` is the precedent and the shape.

| | at initialisation (n=3 seeds) | at `ckpt_best` (4 checkpoints, 3 architectures) |
|---|---:|---:|
| `sd(phi)` | **0.0245** | **1.1291** (range 1.1044–1.1394, spread **3.1 %**) |
| `mean(phi)` | +0.001 to +0.008 | +0.052 to +0.091 |
| `min` / `max` | −0.05 / +0.04 | **−12.97** / **+2.15** |
| fraction `phi > 0` | 0.61 | **0.726–0.739** |
| lag-1 autocorrelation | 0.034 | **0.046–0.050** |

**`tau = 1.1291`**, the median `sd(phi)` across those checkpoints. Four
consequences, every one of which changed the design:

1. **`tau` is a property of the task, not of the neuron.** The four checkpoints
   span `snn`, `twocomp` and `twocomp_detach` and agree to 3.1 %. `EXP_005`'s
   standing lesson is that a statistic measured on one neuron does not transfer;
   this one was measured on three rather than assumed to.
2. **The squash is load-bearing, not cosmetic.** `phi` is bounded above by `H_t`
   and unbounded below. An unsquashed `−12.97` through a multiplicative gain of
   even 0.1 **inverts the sign of the current**. `tanh` bounds the signal to
   (−1, 1) while preserving the asymmetry, which is itself the biologically
   correct shape.
3. **`phi` is 46.1x weaker at initialisation than at convergence.** The arm's
   effective strength is therefore **confounded with training progress**, and no
   choice of `tau` fixes it, because at init the model has no prediction to be in
   error about. This is limitation §6.3 and it is load-bearing for how a null is
   read.
4. **`phi` is nearly white in time** (lag-1 0.046). This is *why* §2.7's control
   is the batch roll and not a time shuffle, and it forbids selling the arm as a
   "temporal salience" mechanism without evidence.

**Rejected, and recorded because the rejection is a design choice:** normalising
`phi` by batch statistics. It would make one example's output depend on the other
examples in its batch — which no other arm in this project does, and which would
make the "fresh" and "carried" protocols measure different functions, since the
carried protocol makes batch composition part of the protocol rather than an
accident of it.

### 2.7 The control, and why it is a batch roll rather than a time shuffle

Arm **S** is the same code with `da_source="rolled"`: row `b` receives row
`b-1`'s dopamine, `torch.roll(da, shifts=1, dims=0)`.

This is what decides whether the arm has a **mechanism** or a **perturbation**,
and the choice of control is the point:

* The **marginal distribution is preserved exactly** — the same multiset of
  values, not merely the same distribution.
* The **temporal autocorrelation is preserved exactly** — every row keeps its own
  time series intact, merely attached to the wrong sequence. A time shuffle would
  destroy it and would then be controlling for two things at once. That matters
  *here specifically* because §2.6 measured the lag-1 autocorrelation at 0.046:
  on a nearly white signal a time shuffle is barely a control at all, while this
  is a total one.
* **Alignment is destroyed completely.**
* **It cannot leak the future.** A roll along *time* with wraparound would bring
  `da[L-k:]` to the front and hand the model its own future. This moves nothing
  along the time axis.
* **No RNG.** So there is no per-step draw to keep out of the captured region, no
  static buffer whose address the graph would bake in, and no seed stream to keep
  independent of the sampler's — three problems `snn.noise` had to solve and this
  arm simply does not have.

At `batch_size = 1` the roll is the identity and the control would silently *be*
the arm. `Config.__post_init__` and `roll_across_batch` both raise.

---

## 3. What is measured

All at the frozen Phase-2 recipe, `d_model = 512`, `n_layers = 2`, enwik8,
20,000 steps, **anchored on today's tree**: decision #7's fp64 clip means a newly
trained run cannot be compared bitwise against any committed figure, and decision
#10 is Elliot's ruling that new experiments train their own anchor rather than
re-baselining.

| leg | run names | arch | `da_mode` | `da_source` | n | role | est. GPU-min |
|---|---|---|---|---|---:|---|---:|
| **1** | `da_anchor_s{0..4}` | `snn` | — | — | 5 | the anchor; **σ is re-measured here** | 5 × 8.5 |
| **2** | `da_mult_s{0..4}` | `dopamine` | mult | rpe | 5 | **arm A — the headline, D1** | 5 × 13.5 |
| **3** | `da_rolled_s{0..2}` | `dopamine` | mult | rolled | 3 | **control S — the mechanism, D2** | 3 × 13.5 |
| **4** | `da_add_s{0..2}` | `dopamine` | add | rpe | 3 | arm B — D3, a marker | 3 × 12.5 |

**~2.9 GPU-hours** of training, from §2.5's measured ms/step plus the driver's
periodic evaluations, not from the withdrawn cost constant. Plus sixteen
full-split test evaluations and one horizon probe per arm.

Every run is scored under **both** protocols on the **test** split, and the
per-context-length decomposition of §6.8 is computed for every arm before any
mean is reported.

---

## 4. Hypotheses

### 4.0 What this design can and cannot resolve, stated before the run

`CONTRIBUTING.md` §2 requires a prediction's power to be stated in advance, and
§3 requires every bar to be checked against §6 before it is committed. Both were
done and the results are these.

**Resolution.** At n = 5 per arm and `sigma = 0.00461` — the committed baseline
figure, used here **only to size the design**; every bar below is evaluated
against the sigma actually measured on these seeds (§6.2) — the minimum
detectable difference at 80 % power, α = 0.05 two-sided, is

```
MDE(D1, n=5 v 5) = (t_.975,8 + t_.80,8) * sigma * sqrt(2/5) = 3.195 * 0.00461 * 0.6325 = 0.0093 bpc
MDE(D2, n=5 v 3) = (t_.975,6 + t_.80,6) * sigma * sqrt(1/5+1/3) = 3.353 * 0.00461 * 0.7303 = 0.0113 bpc
```

**An effect below 0.0093 bpc is not resolvable by this design.** For scale: the
adopted two-compartment arm is 0.134 bpc (14x this), `EXP_008`'s threshold arm
0.059 bpc (6x). **A D1 null means "not established". It does not mean "no
effect", and the two are different sentences.**

**Ties.** `Δ = 0` exactly is measure-zero on a continuous bpc, so no boundary
below sits on a value the metric can land on — the failure `CONTRIBUTING.md` §3
forbids. D4's `1e-3` bar is on a learned parameter's magnitude, which is likewise
continuous. D5 is a marker precisely because the memory horizon is an integer on
a dense lattice and a threshold on it would sit on a reachable value.

### T1 — the instrument *(abort-on-fail)*

> `scripts/exp/018_calibrate_da.py` reproduces §2.6's `tau` to two significant
> figures on a re-run, and its C1 and C2 self-checks pass.

**Already run: C1 at `0.000e+00` on all four checkpoints; C2 reproduces three
checkpoints' committed val bpc to 0.0023–0.0062 bpc** (the residual is
structural — C2 sums `L-1` of `L` positions per window — and is reported rather
than tolerated silently). C2 is `NOT RUN` on `anchor_twocomp_d512_s0`, whose
committed val bpc is non-finite because that run diverged *after* its
`ckpt_best`; an absent reference is not a failed comparison, which is
`017_detach_results.py`'s rule that a diverged run resolves NOT RUN and never
UNRESOLVED.

### T2 — the gates *(abort-on-fail)*

> G1–G12 of §7 are all green before the first training step of leg 1.

### D1 — the headline *(difference, BOTH terms constrained)*

> `Δ = mean bpc_carried(da_anchor, n=5) − mean bpc_carried(da_mult, n=5)` on the
> **test** split, with a Welch two-sample 95 % CI.

| | verdict |
|---|---|
| `Δ > 0` **and** the 95 % CI on `Δ` excludes 0 | **THE DOPAMINE SIGNAL HELPS**, by `Δ` bpc at 735K |
| `Δ < 0` **and** the 95 % CI on `Δ` excludes 0 | **IT HURTS**, by `Δ` bpc |
| otherwise | **UNRESOLVED at n = 5** — *and this is a verdict, not a failure to report one* |

Both absolute means are always printed alongside `Δ`, never the difference alone.
That is `EXP_013`'s N1 defect — a difference constrained without constraining
both its terms — and it is the defect this table is shaped to avoid.

### D2 — the mechanism *(conjunction; BOTH terms constrained, deliberately)*

> **D1 fires in the "helps" direction** *and*
> `Δ_align = mean bpc_carried(da_rolled, n=3) − mean bpc_carried(da_mult, n=5) > 0`
> with a Welch 95 % CI excluding 0.

Note the algebra, stated so nobody has to trust it: `Δ(A) − Δ(S)` against a shared
anchor **is** `bpc(S) − bpc(A)` — the anchor cancels exactly — so D2 is a clean
two-sample comparison and does not inherit the anchor's variance.

| | verdict |
|---|---|
| both clauses hold | **THE ALIGNMENT IS THE MECHANISM.** The gain is the RPE's, not the perturbation's |
| D1 helps, `Δ_align` CI includes 0 | **THE MECHANISM IS NOT ESTABLISHED at n = 3 v 5.** The gain may be a perturbation effect; report it as unattributed |
| D1 helps, `Δ_align < 0` with CI excluding 0 | **THE CONTROL BEATS THE ARM.** Whatever is happening is not the RPE, and this is the most informative outcome in the table |
| D1 does not fire | **NOT RUN** — there is no gain to attribute |

### D3 — additive versus multiplicative *(MARKER, no verdict)*

> Report `mean bpc_carried(da_add, n=3)` and its Welch CI against both the anchor
> and `da_mult`.

**No direction is predicted and no bar is set.** At n = 3 this leg is
underpowered by construction — the MDE against a 5-seed arm is 0.0113 bpc — and
that is said here rather than discovered afterwards. It exists to price a second
dopaminergic action, not to choose between them.

### D4 — did the sensitivity ever engage? *(MARKER + ALARM)*

> Report `max|k|` and the per-layer distribution of `k` at step 20,000 for every
> dopamine run, against `k_init = 0.0`.

**ALARM: if `max|k| < 1e-3` after 20,000 steps, the arm never engaged**, and a
D1 null is then a statement about the optimiser and the initialisation, not about
the mechanism. In that case the null is reported **as that**, and the mechanism
is recorded as **not measured** rather than as measured and absent. §2.6's 46.1x
init-to-convergence ratio is the reason this alarm exists and it was written
before the run.

### D5 — reach *(MARKER, no verdict)*

> `EXP_001`'s memory-horizon probe on the best `da_mult` seed and the best anchor
> seed, plus the **continuous** per-context-length bpc curve for both.

The horizon is an integer on a dense lattice, so no threshold is placed on it —
`CONTRIBUTING.md` §3. The per-context curve is continuous and is the thing a
future bar may be placed on. The baseline's horizon is 7 and the adopted
two-compartment arm's is 47; both are quoted for scale and neither is a bar.

---

## 5. Decision rule, fixed now

| outcome | what is written down |
|---|---|
| T1 or T2 fails | **The experiment does not run.** Fix the gate, not the bar |
| D1 helps and D2 confirms | A broadcast RPE is worth `Δ` bpc at 735K, **and the alignment is why**. Referred to Elliot as a candidate; **not adopted here** |
| D1 helps, D2 unresolved | A time-varying global modulation is worth `Δ` bpc at 735K. **The RPE is not credited**; reported as unattributed |
| D1 helps, control beats arm | Reported prominently as the headline. The mechanism is falsified and the perturbation is real |
| D1 unresolved, D4 alarm silent | The mechanism was engaged and bought less than 0.0093 bpc. **Reported as a null with its resolution stated** |
| D1 unresolved, D4 alarm fires | The arm never engaged. Reported as **not measured**, with the initialisation named as the suspect |
| D1 hurts | Reported as a negative result, prominently, in its own voice |

**No verdict in this table adopts anything, ranks anything, rules I5, or edits
`04_phase4_interim.md` §8.** A bar is reported as it fired; a corrected rule is
referred to the next phase, never applied retroactively.

---

## 6. Known limitations, stated in advance

1. **n = 5 and n = 3.** D1 resolves 0.0093 bpc and no better; D2 and D3 resolve
   0.0113. UNRESOLVED must not be read as zero.
2. **σ is re-measured on these seeds and not inherited.** `CONTRIBUTING.md` §4:
   σ is 0.00461 on the baseline and 0.00313 on the two-state neuron, and it
   varies with a hyperparameter that changes no parameter and no function. The
   0.00461 above sizes the design only. A 3-seed σ is quoted with its df as a
   marker and is never promoted.
3. **The signal is 46.1x weaker at initialisation than at convergence** (§2.6).
   The arm's effective strength is confounded with training progress. Its one
   benefit — it cannot destabilise early training — is stated, not claimed as a
   result. D4 exists because of this limitation.
4. **`tau` is calibrated on the baseline neuron and is not re-tuned per arm**,
   and `k_init = 0.0` is not swept. Both are `EXP_017` §5's "not re-tuned"
   limitation and both are named here rather than discovered later.
5. **735K only.** `EXP_014` and `EXP_015` establish that this model underfits at
   735K and stops doing so between 2.5M and 5M. Nothing here transfers to 5M and
   nothing here is measured there.
6. **The `snn` neuron only.** The composition onto `twocomp_detach` — the arm
   that trains at width — is the obvious follow-up. `snn/dopamine.py` is written
   against `cur` rather than against a neuron, so it is one subclass, exactly as
   `TwoCompThresholdCharLM` was. It is referred in §10, not run here.
7. **I5 is not ruled** (§2.4). Every table labels the arm a diagnostic.
8. **The mean is not the result.** `EXP_004`'s mean was three effects of
   opposite sign that nearly cancelled: +41 % at zero context, −48 % inside the
   baseline's own 7-character horizon, and +107 % beyond it. The
   per-context-length decomposition is computed for every arm and reported
   **before** any mean, and a `Δ` quoted without it is not a result this
   experiment produces.
9. **The arm costs 1.64x at training and at inference** (§2.5), which is worse
   than `EXP_008`'s 1.12x and `EXP_007`'s 1.49x. Any gain must be read against
   that price, and §6.7 of the Phase-4 report is the precedent for a candidate
   being dominated on cost rather than on bits.
10. **The `da_source="off"` configuration carries 1,024 dead parameters** — `k`
    receives no gradient because it is never read. It exists only as the nesting
    check in G1 and is never trained; no leg above uses it.

---

## 7. Gates — abort-on-fail, not advisory

`tests/test_dopamine.py`, following `tests/test_prescan.py`, whose header states
the standard for exactly this arm shape: **a new per-channel parameter plus an
elementwise transform of `cur` outside the time loop is NOT an R10 gate**,
because it writes no jiterator source and hands the result to the committed
`snn.neuron.lif_scan` — still guarded by its own gate and its 25/25 mutation
campaign, untouched by this arm.

| | gate |
|---|---|
| **G1** | **Nesting.** At `k_init = 0`, forward `torch.equal` to `SpikingCharLM` **and every shared parameter's gradient `torch.equal`**, for both modes and all three sources. Asserted at `== 0.0`, the `EXP_011` K4 standard — and matched **by name**, not by `zip` position, because `k.0` sorts between `head.weight` and `layers.0.bias` and a positional zip silently compares 3 of 7 |
| **G2** | **Causality.** Perturb `x[:, t+1:]`; `DA[:, :t+1]` unchanged **bitwise**. Plus: `forward` accepts `idx` alone, asserted by signature |
| **G3** | **Reachability in the graph.** Exactly `K` gradients named `k.*`, none `None`, none exactly zero at the nesting init |
| **G4** | **Per-channel functionally, not by shape.** Perturb `k[0, 0]`; channel 0 moves and channels `1:` do not |
| **G5** | **Shape contract** rejects broadcastable-but-wrong `k` and `da` |
| **G6** | **Cost structure.** The stack runs **exactly twice** per forward, and the transform runs **once per layer per pass, not once per timestep** — monkeypatch and count, varying `L`. This is the whole cost argument |
| **G7** | **CUDA-graph capture survives** the `no_grad` pass inside the captured region; determinism under replay; resume reproduces. *(Capture already verified in §2.5 for all three variants; the gate is what keeps it verified)* |
| **G8** | **Mutation legs inside the gate itself** — `test_noise.py`'s pattern, "a screen nothing has ever tripped is indistinguishable from one that cannot trip". At minimum: sign flip on `phi`; **off-by-one in the causal shift, which G2 must catch**; `tanh` removed; `mult`↔`add` swapped; the roll applied to the real arm; `DA_0` not zeroed; `roll` along time instead of batch |
| **G9** | Parameter count exactly `spiking + K*d`; `build_model` dispatches and records its init; `Config` rejects degenerate `tau` and `B=1` under `rolled`; the unknown-arch error names every arm; state round-trips under protocol B |
| **G10** | **§7.2 gradient-reachability screen** — a `Candidate` in `scripts/audit/09_gradient_reachability.py`, `derived_zero` written **before** the run and never edited to match a measurement. §2.6 predicts a **small** ratio, because `phi` at init is 46x weaker; per that script's own docstring *small is not fatal* — Adam is scale-free per parameter — while *exactly zero* is. Both numbers reported |
| **G11** | **Cost measured, reference named**, nothing else on the GPU *(done, §2.5)* |
| **G12** | `ruff check .` clean; `pytest -m "not cuda and not data"`; then the **full** suite on the GPU box, because CI cannot catch a fused-kernel, capture or determinism regression |

---

## 8. Entry conditions

1. This file is committed **before** the first training step. Its SHA-256 is
   stamped into `docs/reports/data/exp_018_run_manifest.json` before the first
   run and re-checked after the last; **a mismatch voids the experiment and
   nothing may be reported from it.**
2. `docs/reports/data/exp_018_da_calibration.json` is committed and T1 passes.
3. G1–G12 are green.
4. `scripts/exp/018_dopamine_results.py` is committed **before any run's bpc is
   read**, with every bar above transcribed as a constant.
5. No mutation campaign is running (the campaign writes wrong code into the
   source tree and a process that starts mid-campaign keeps the mutant for its
   whole life).
6. Nothing else on the GPU.

---

## 9. Results

*(appended after the run — nothing above this line is rewritten)*

---

## 10. Referred to Elliot, and not decided here

1. **Does a rank-1 broadcast from the head violate I5?** §2.4 states the case
   both ways and rules nothing. This is a fourth I5 boundary question and it
   joins the three already open as decision #3.
2. **Is 1.64x at training *and inference* an acceptable price** for whatever this
   buys? `EXP_007`'s token-shift was dominated at 1.49x; this is worse.
3. **Adoption**, if D1 and D2 fire. This file adopts nothing.
4. **The composition onto `twocomp_detach`** — one subclass, the arm that trains
   at width, aimed at the reach bottleneck `EXP_015` identified. Not run here.
5. **Whether the dopamine-gated *decay* is the better form of this arm.**
   `beta_eff = sigmoid(logit(beta_c) + k_c*DA_t)` makes dopamine gate the forget
   gate rather than the input, which is what the GRU anchor's reach advantage
   actually is, and `01a_literature_review.md` §4.4 is its permission slip.
   `beta_eff` precomputes outside the time loop as `[B, L, d]`, so the
   one-kernel-per-timestep floor survives. It is **not** run here because it
   needs a new fused kernel, a new hand-written backward, a new R10 gate and a
   new mutation campaign — R10 is the worst bug class in this project — and
   pricing the signal at all should come before paying for that.
