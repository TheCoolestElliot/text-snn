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

**Status: CLOSED 2026-08-14 — T1 and T2 PASS; D1 UNRESOLVED by 3.6e-06 bpc in the direction of harm; D2 NOT RUN; D3, D5 MARKERS; D4 ENGAGED.** Sixteen runs, sixteen completed, zero diverged, 2.63 GPU-h. The corrected PAIRED test — measured because it hurts — says the arm costs 0.0062 bpc on 5/5 seeds. Results in §9; folded into `docs/reports/04_phase4_interim.md` §17.**
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

**Sixteen runs, sixteen completed, zero diverged.** The pre-registration's
SHA-256 is unchanged across the ladder (`prereg_unchanged: true`), so the
experiment stands. Max grad norm over all 16 runs was 0.5398, the clip fired
**zero** times, and no run tripped the VRAM alarm. Total ladder cost
**2.63 GPU-hours** against the ~2.9 estimated in §3.

### 9.1 The scoreboard

| bar | verdict | one line |
|---|---|---|
| **T1** | **PASS** | `tau` reproduces; C1 at exactly `0.000e+00` on all four checkpoints |
| **T2** | **PASS** | G1–G12 green before the first training step; 51 gates, full suite 528 |
| **D1** | **UNRESOLVED** | the CI includes zero **by 3.6e-06 bpc**; the point estimate is *negative* |
| **D2** | **NOT RUN** | D1 did not fire in the "helps" direction, so there is no gain to attribute |
| **D3** | **MARKER** | additive lands between the anchor and the multiplicative arm; n = 3 |
| **D4** | **ENGAGED** | `max|k|` reached 2.32 from `k_init = 0`; the alarm did not fire |
| **D5** | **MARKER** | see §9.9 |

**The short version: the arm does not help, the pre-registered test could not
quite say so, the corrected test says it costs 0.0062 bpc on 5/5 seeds — and the
most interesting number in the experiment is a factor of 26 that has nothing to
do with bpc.**

### 9.2 D1 — UNRESOLVED by 3.6 millionths of a bpc

| | carried | fresh |
|---|---:|---:|
| `da_anchor` n=5 | **2.251210** (sd 0.004748) | **2.267755** (sd 0.004661) |
| `da_mult` n=5 | **2.257425** (sd 0.003590) | **2.273859** (sd 0.003529) |
| `Δ` = anchor − arm | **−0.0062148** | −0.0061047 |
| Welch se / df | 0.0026618 / 7.447 | 0.0026147 / 7.452 |
| 95 % CI | **[−0.0124333, +0.0000036]** | [−0.0122124, +0.0000029] |
| t vs t_crit | −2.33480 vs 2.33617 | −2.33480 vs 2.33586 |
| **verdict** | **UNRESOLVED** | **UNRESOLVED** |

Both absolute means are printed, never the difference alone — `EXP_013`'s N1.

The interval includes zero **by 3.6e-06 bpc** and the statistic misses its
critical value by **0.0014**. `CONTRIBUTING.md` §3: *"Report a near-miss as a
miss. The near-ness is worth discussing; the verdict is not negotiable."*
**D1 is UNRESOLVED and nothing below upgrades it.**

The direction matters: the point estimate is **negative**, so this is a near-miss
of *harm*. D1's "helps" branch was never close, and the arm's own seed spread
(0.00359) is **smaller** than the anchor's (0.00475), so the failure to resolve is
not a noisy arm.

**σ, re-measured and not inherited** (`CONTRIBUTING.md` §4): 0.00475 carried /
0.00466 fresh on the anchor's five seeds, against the 0.00461 that sized the
design. The pre-registered power therefore stands with the measured number
substituted: MDE(D1) = **0.0096** rather than 0.0093.

**A consistency marker for decision #10, and not more than that.** The fresh
anchor lands 0.00190 bpc below the committed Phase-2 baseline's 2.25311 — the
same sign and order as the 1.97e-03 `EXP_014` §9.12 measured for decision #7's
fp64 clip. Against that baseline's own five seeds the Welch se is 0.00296, so
0.00190 is **0.64 se** and is indistinguishable from zero. It is recorded as a
consistency marker for the ruling that new experiments anchor on today's tree,
**not** as a reproduction of the tree effect's size.

### 9.3 The correction, MEASURED because it hurts

**The design used an unpaired Welch test on paired data. That is a defect in the
pre-registration, not in the arm.** In this project a seed fixes *both* the
initial weights and the data order, so `da_anchor_s{i}` and `da_mult_s{i}` are
genuinely paired, and the unpaired test throws that away.

`CONTRIBUTING.md` §3: *"When a correction would flatter the work, refer it; when
it would hurt, measure it."* This one hurts:

| paired by seed, `anchor − arm` | carried | fresh |
|---|---:|---:|
| per seed | −0.00298, −0.00638, −0.00686, −0.00738, −0.00747 | −0.00292, −0.00632, −0.00699, −0.00708, −0.00720 |
| arm worse on | **5 / 5** | **5 / 5** |
| mean (sd) | −0.006215 (0.001859) | −0.006105 (0.001811) |
| t on df 4 | **−7.477** | −7.536 |
| 95 % CI | **[−0.008523, −0.003907]** | [−0.008354, −0.003856] |
| two-sided p | **1.7e-03** | 1.7e-03 |

**The dopamine arm costs about 0.0062 bpc, on every seed, and the pre-registered
instrument was too blunt to say so.** The sd of the paired *difference* is
0.00186 against a between-seed sd of ~0.0045, which is what pairing is for.

**The corrected RULE is referred, not applied retroactively** (§10 item 6).
D1's verdict stays UNRESOLVED.

### 9.4 D2 — NOT RUN, and the 26x it bought instead

D1 did not fire in the "helps" direction, so **D2 resolves NOT RUN exactly as §4
wrote it.** Everything in this subsection is a marker.

| carried, test | n | mean | sd over seeds |
|---|---:|---:|---:|
| `da_anchor` | 5 | 2.25121 | 0.00475 |
| `da_add` | 3 | 2.25479 | 0.00099 |
| `da_rolled` (control) | 3 | **2.25487** | **0.00059** |
| `da_mult` (arm) | 5 | **2.25743** | 0.00359 |

The control lands **between** the anchor and the arm. Paired on the three shared
seeds: anchor − rolled = −0.00242 (p 0.48); rolled − arm = −0.00298 (p 0.23).
**Neither half resolves at n = 3 and neither is claimed.** The design was powered
to detect a gain; it cannot split a loss into two halves.

**What the control did decide, and it is the most interesting number here:**

| `rms(k)` at step 20,000, over layers and seeds | mean rms(k) | mean max&#124;k&#124; |
|---|---:|---:|
| `da_mult` — the aligned RPE | **0.1950** | 2.32 |
| `da_rolled` — the same signal, misaligned | **0.0075** | 0.045 |

**A factor of 26, and it is extraordinarily stable**: `da_rolled`'s `k.0` rms is
0.0021 / 0.0026 / 0.0025 across three seeds against `da_mult`'s 0.104–0.140
across five.

**The optimiser can tell the aligned signal from the misaligned one, and turns
the gain up 26x on the real one.** That is a direct, quantitative demonstration
that the reward prediction error carries information the model can find and wants
to use. The control was pre-registered to ask whether alignment matters; its most
decisive answer is not about bpc at all.

**The control's own limitation, discovered after the pre-registration was
committed and recorded here rather than back-patched into §2.7.** `da_rolled`
gives row `b` the dopamine of row `b-1`, so its loss for row `b` depends on row
`b-1`'s data. The arm's does not. That is the same batch-coupling objection §2.6
used to *reject* batch-normalising `phi`, and the control has it while the arm
does not. Three things bound it and none removes it: both evaluation protocols
are deterministic and `batch_size` is frozen at 128, so the control's score is
well-defined; the neighbouring row is unrelated text under training and a
different stream or window under evaluation, which is the misalignment the
control needs; and the control is a **diagnostic**, never a candidate model. What
it costs is that `da_rolled`'s absolute bpc is not a deployable model's bpc. The
alternatives were worse — a roll along *time* wraps `da[L-1]` to the front and
leaks the future, and a causal lag changes the marginal and asks a different
question ("how stale may the signal be?").

### 9.5 The finding: it fits train identically and generalises worse

Paired by seed, `arm − anchor` (positive = the arm is worse), n = 5:

| | mean | se | t (df 4) | p |
|---|---:|---:|---:|---:|
| train bpc, last 8 batches | **+0.00043** | 0.00084 | +0.52 | **0.63** |
| test bpc, fresh | **+0.00611** | 0.00081 | +7.54 | **0.0017** |
| test bpc, carried | +0.00622 | 0.00083 | +7.48 | 0.0017 |

**The train→test gap widens by 0.0057 bpc.** The arm extracts **no additional
training fit** from a signal it has assigned 26x the gain of a control, and pays
0.0062 bpc at test. For scale, `EXP_013` §2.2 measured this model's entire
generalisation gap at **0.0059 bpc**: this arm roughly **doubles** it.

**Caveat on the instrument, stated because it is load-bearing.** "train bpc" is
the mean of the last eight logged single-batch losses; its within-run sd is 0.049
and its across-seed sd is 0.011, both far larger than the effect. What makes it
usable is that same-seed runs see **the same final batches** — the sampler is a
pure function of `(seed, step)` — so the comparison is paired and the paired se
is 0.00084. It is an indicative marker, **not** `EXP_013`'s instrument, and a
proper generalisation-gap measurement is referred in §10 item 7.

**A hypothesis, offered as one and not as a finding.** `phi` is computed from the
model's *own* predictions, so it is a function of the model's parameters and of
how well it has memorised the text in front of it. Conditioning the forward pass
on it opens a self-referential channel that can carry training-set-specific
structure — real at training time, absent at test time. That would explain
identical train fit with worse test fit, and why the effect is larger for the
aligned signal than for the misaligned one. **Nothing here establishes it.**
`CONTRIBUTING.md` §4 requires a direction of causation to be checked before it is
claimed.


#### CORRECTION, added after §9.5 was written — §9.5's reading is WRONG

*(The paragraphs above stand as written, per `EXP_015` §14.5's precedent — a
wrong claim is corrected **beneath** the standing original, not deleted.)*

§10 item 7 referred the proper measurement to a later experiment. It was run
instead, on this tree, with `scripts/exp/013_generalisation_gap.py` — the
committed instrument, the one `EXP_013` used, over all sixteen checkpoints
(`docs/reports/data/exp_018_generalisation_gap.json`). **It does not support
§9.5, and the correction goes against the reading, so it is made here rather
than deferred.**

Paired by seed, `arm − anchor`, n = 5:

| carried | anchor | arm | paired Δ | t (df 4) | p | arm worse |
|---|---:|---:|---:|---:|---:|---|
| **train slice** | 2.25513 | 2.26034 | **+0.00522** | +7.23 | **0.0019** | **5/5** |
| test | 2.25121 | 2.25743 | +0.00621 | +7.48 | 0.0017 | 5/5 |
| **gap** | −0.00391 | −0.00292 | **+0.00100** | +1.71 | **0.16** | 4/5 |

| fresh | anchor | arm | paired Δ | t | p | arm worse |
|---|---:|---:|---:|---:|---:|---|
| **train slice** | 2.26338 | 2.26919 | **+0.00581** | +6.57 | **0.0028** | **5/5** |
| test | 2.26775 | 2.27386 | +0.00610 | +7.54 | 0.0017 | 5/5 |
| **gap** | +0.00437 | +0.00467 | **+0.00029** | +0.35 | **0.74** | 3/5 |

**The arm is worse on text it was trained on by almost exactly as much as on text
it was not** — +0.00522 against +0.00621 carried, +0.00581 against +0.00610
fresh. **The generalisation gap does not widen resolvably** (p = 0.16 carried,
0.74 fresh), and the "gap roughly doubles" sentence in §9.5 is **withdrawn**.

**This is a capability loss, not a generalisation failure.** The arm is simply a
worse model, everywhere, on seen and unseen text alike.

**The self-referential-channel hypothesis is therefore NOT supported by this
measurement**, and §10 item 8's frozen-copy test loses the observation that
motivated it. It is left in §10 because it remains a cheap and well-posed
question, but it is no longer pointed at by evidence.

**The discrepancy between the two estimators is RESOLVED**, by
`scripts/exp/018_chase_train_estimator.py`
(`docs/reports/data/exp_018_train_estimator.json`). The sampler is a pure
function of `(seed, step)`, so the exact eight batches those logged steps trained
on were regenerated and re-scored at `ckpt_final`, three ways.

| paired, arm − anchor, the SAME eight batches | Δ | t (df 4) | p | arm worse |
|---|---:|---:|---:|---|
| as **logged** during training | +0.00043 | +0.52 | 0.63 | 2/5 |
| re-scored at **final weights** | **+0.00377** | **+4.22** | **0.0135** | **5/5** |

**The cause is stale weights, and the staleness is not symmetric.** A logged
training loss is computed at the weights *before* that step's update, so it lags;
over the logged tail the anchor improved by **0.00619** on its own batches while
the arm improved by only **0.00285** — an asymmetry of **+0.00334 ± 0.00096**
(t = 3.46, p = 0.026) that is almost exactly the size of the effect and cancelled
it. §9.5's marker was measuring two arms at different points on their own
trajectories.

Two things this rules out. **`eval()` and `train()` agree bit-for-bit
(`0.00e+00`, worst over ten runs)** on identical inputs at identical weights, so
there is no mode-dependence and no undeclared state in the arm — the one
candidate that could have borne on the result. And the batches themselves are not
the issue: re-scored properly they give **+0.00377**, consistent with the
train-slice's +0.00522 and the test's +0.00621 on different text.

**So the arm is worse on the very batches it was trained on, at the weights it
finished with** — which is the capability-loss reading, confirmed on a third
independent slice.

**The standing lesson is methodological and outlives this arm: a running training
loss is measured at stale weights, the staleness differs between arms, and it is
not a stand-in for a train-split bpc.**

**What survives §9.5 unchanged:** the 26x (§9.4), the paired loss (§9.3), the
decomposition (§9.9), and D4's engagement (§9.7). What does not survive is the
inference drawn from the train side, and the hypothesis built on it.

### 9.6 D3 — the additive form, a marker

`da_add` carried **2.25479** (n = 3, sd 0.00099). Paired on shared seeds:
add − anchor = +0.00235 (p 0.46, 2/3 worse); add − mult = −0.00306 (p 0.20,
1/3 worse). **No direction was predicted and none is claimed.** At n = 3 the MDE
is 0.0116 and both comparisons are well inside it.

Its learned sensitivity inverts the multiplicative arm's layer pattern —
rms(k) 0.175 in layer 0 and 0.104 in layer 1, against `da_mult`'s 0.14 and 0.28.
Recorded, not interpreted.

### 9.7 D4 — the sensitivity engaged, with structure

From `k_init = 0.0`, `max|k|` over the dopamine runs reached **2.32** mean /
2.92 max, and the **alarm floor of 1e-3 was never approached**. So D1's verdict is
a statement about the mechanism and **not** about the optimiser or the
initialisation. §2.6's 46x init-to-convergence ratio made "the parameter never
moved" a live possibility, and it is now ruled out — which is what D4 was written
for, before the run.

`da_mult`, per layer, across five seeds: `k.0` rms 0.104–0.140 with **37 %** of
channels positive; `k.1` rms 0.239–0.283 with **84 %** positive. In `mult` mode
`k_c > 0` means *amplify my input when the outcome was better than predicted*, so
the layer feeding the head gains up on confidence and attenuates on surprise
while layer 0 mostly does the opposite. **Reported as a measurement, not a
mechanism** — nothing here shows the sign pattern is what produces the loss.

### 9.8 Cost, measured against a named reference

Realised over the full ladder, against `da_anchor`'s own mean wall clock of
**396.9 s** on the same box in the same session, nothing else on the GPU:

| arm | mean wall clock | ratio | §2.5's 300-step figure |
|---|---:|---:|---:|
| `da_anchor` | 396.9 s | 1.000 | 1.000 |
| `da_mult` | 677.8 s | **1.708** | 1.640 |
| `da_rolled` | 690.9 s | 1.741 | 1.644 |
| `da_add` | 597.7 s | **1.506** | 1.509 |

The additive form reproduces its short-run estimate to 0.2 %; the multiplicative
forms come in ~4 % *above* theirs, because the full runs include evaluation and
this arm's evaluation is two-pass as well — an effect §2.5's eval-off measurement
could not see. **The arm costs 1.71x at training and at inference**, against
`EXP_008`'s 1.12x and `EXP_007`'s 1.49x; §6.7 item 2 already records token-shift
as dominated on cost at 1.49x.


---

### 9.9 D5 and the decomposition — the mean is two effects of opposite sign

`EXP_001`'s probe, all 16 checkpoints, test split, `--baseline-arm da_anchor`
(today's tree, never `snn_beta0.5` — decision #10). **F1 held on all sixteen**,
residuals 2.0e-11 to 1.2e-08 against each run's own committed `final_test.json`.

**Absolute bpc at context `c`:**

| arm | c=0 | c=1 | c=2 | c=3 | c=4 | c=8 | c=16 | c=32 | c=64 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `da_anchor` | 4.1495 | 3.5459 | 2.9649 | 2.5530 | 2.3807 | 2.2707 | 2.2680 | 2.2682 | 2.2677 |
| `da_mult` | **4.1249** | 3.5531 | 2.9866 | **2.5957** | 2.4129 | 2.2786 | 2.2740 | 2.2731 | 2.2739 |
| `da_rolled` | 4.1491 | 3.5461 | 2.9481 | 2.5559 | 2.3818 | 2.2736 | 2.2712 | 2.2721 | 2.2708 |
| `da_add` | 4.1276 | 3.5504 | 2.9785 | 2.5971 | 2.4099 | 2.2740 | 2.2714 | 2.2710 | 2.2711 |

**Δ vs `da_anchor`, positive = worse**, against the per-context 2σ bar measured
on the anchor's own five seeds:

| | c=0 | c=1 | c=2 | c=3 | c=4 | c=8 | c=16 | c=32 | c=64 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `da_mult` | **−0.0246** | +0.0072 | +0.0217 | **+0.0427** | +0.0322 | +0.0079 | +0.0060 | +0.0050 | +0.0062 |
| `da_rolled` | −0.0004 | +0.0002 | −0.0168 | +0.0029 | +0.0011 | +0.0029 | +0.0032 | +0.0039 | +0.0031 |
| `da_add` | −0.0219 | +0.0045 | +0.0136 | +0.0441 | +0.0292 | +0.0033 | +0.0034 | +0.0028 | +0.0034 |
| **2σ bar** | 0.0296 | 0.0304 | 0.0260 | 0.0144 | 0.0145 | 0.0040 | 0.0019 | 0.0010 | 0.0002 |

**`CONTRIBUTING.md` §3 requires this decomposition before any mean is quoted, and
here it earns its place: the −0.0062 mean is two effects of opposite sign.**

* **At c = 0 the arm is 0.0246 bpc BETTER** — and the dopamine mechanism is
  **structurally inactive there**. `phi_0 = 0`, and at one character of context
  every position *is* position 0, so no modulation is applied at all. Whatever is
  happening at c = 0 is a property of the **weights training left behind**, not
  of the signal. It is also **inside the 2σ bar** (0.0296) and is therefore not
  established.
* **The damage is concentrated at c = 2–4**, peaking at **+0.0427 at c = 3**
  against a 0.0144 bar — three times the bar, and **seven times the whole-split
  mean effect**.
* **It settles to ~+0.006 for c ≥ 8**, where the bars are 0.0040 down to 0.0002,
  so it is resolved many times over out there.

`da_mult` is significantly worse at **125 of 128** contexts; `nowhere_worse` is
**False**. The `EXP_001` dominance criterion — "is the absolute curve nowhere
worse" — is **failed**, and failed at short contexts, which is the failure mode
that criterion exists to catch.

**The shape suggests a mechanism and the control supports it.** The signal first
exists at c = 1 and is computed from the model's predictive distribution; at
c = 2–4 that distribution is still nearly uninformative, so `phi` there is
dominated by noise — and the arm has learned a large gain on it. By c ≥ 8 the
distribution is informative and the damage drops by a factor of five. The
misaligned control shows **neither** feature: no c = 0 benefit (−0.0004) and no
short-context cliff (+0.0029 at c = 3, worst-short +0.0029). So both ends of the
arm's curve track *alignment*, not the perturbation. **This is a reading of the
shape, not a demonstration**; `CONTRIBUTING.md` §4's direction-of-causation rule
applies and §10 item 8 carries it.

**D5, the horizon marker, no verdict.** 2σ horizon per seed:
`da_anchor` **[7, 7, 7, 7, 7]**, `da_mult` **[8, 8, 8, 7, 7]**, `da_rolled`
[7, 7, 7], `da_add` [7, 8, 8]. The baseline's committed horizon is 7 and the
adopted two-compartment arm's is 47. **No threshold is placed on this** — the
horizon is an integer on a dense lattice and `CONTRIBUTING.md` §3 forbids it. A
median shift of 7 → 8 on three of five seeds, while the absolute curve is worse
at 125 of 128 contexts, is exactly the pattern `EXP_001` warns about: buying
nominal reach while degrading the characters the model already sees.

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

### RULED 2026-08-14 — item 1 is closed

*(Elliot's instruction, after §9 was written and after he had read it. Recorded
here; the authoritative text is `01_reconnaissance.md` §4.6's extension and
`04_phase4_interim.md` §8 row 3.)*

**Item 1 — does a rank-1 broadcast from the head violate I5? — is RESOLVED, and
the answer makes this arm permanently diagnostic.** The boundary is extended
where the architecture's own cost model already draws the line:

* a modulator driven by the **previous layer's output** is **admitted** — it is
  computed before that layer's time loop, so it preserves the depth-sequential
  evaluation and costs O(1) kernels per layer;
* a modulator driven by the **head** is **not adoptable** — the head is
  downstream of every layer, so the signal costs a second forward pass. Such arms
  may still be **built and measured as labelled diagnostics**, which is exactly
  what this experiment is;
* a learned `[d, d]` lateral matrix stays forbidden, and §4.6's
  O(1)-parameters-per-neuron test still applies on top.

**Nothing in this file changes.** §2.4 said "NOT RULED, and this file does not
rule it", which was true when written and is left standing; §6 item 7 and every
table's diagnostic label were already correct and are now correct as a matter of
rule rather than of caution. **The ruling was made after the evidence existed and
that is stated rather than hidden** — what makes it defensible is that the
criterion is structural and would read the same had D1 fired the other way.

**What it opens:** a **feedforward** data-dependent modulator — the
`beta_eff = sigmoid(logit(beta_c) + k_c*g_t)` shape of item 5, driven by layer
`k-1`'s own spikes rather than by the head — is now a **legal arm** rather than a
boundary question. `01a_literature_review.md` §4.4 is its standing evidence and
`EXP_015` says reach is where the gap lives. `snn/dopamine.py`'s
`apply_dopamine` is written against `cur` and would be reusable unchanged.

### Added after the run — referrals the results created

*(not pre-registered; created by §9 and marked as such)*

6. **A paired test, wherever both arms share a seed set.** §9.3 is the case: this
   project's seed fixes both the initial weights and the data order, so any two
   arms run on seeds 0..n are paired, and the unpaired Welch test throws that
   information away. Here it was the difference between UNRESOLVED and a decisive
   p = 1.7e-03. **The rule is referred, not applied retroactively** — D1's verdict
   stands as it fired. What is referred is that the *next* experiment should
   pre-register a paired comparison, with the unpaired one reported alongside it
   so both are visible.

7. **Measure the generalisation gap with `EXP_013`'s instrument, not with mine.**
   §9.5's train-side number is a paired marker built from eight logged
   single-batch losses. The claim it supports — identical train fit, worse test
   fit, gap roughly doubled — is the most consequential thing this experiment
   found and it deserves the committed instrument rather than an indicative one.
   `scripts/exp/013_generalisation_gap.py` exists and was not run here.

8. **Is the self-referential channel the mechanism?** §9.5's hypothesis: `phi` is
   a function of the model's own parameters and of how well it has memorised the
   text in front of it, so conditioning the forward pass on it may carry
   training-set-specific structure that is absent at test. It predicts something
   testable and cheap: an RPE computed by a **frozen** copy of the model (or by a
   different model entirely) should not widen the gap the same way. Nothing here
   establishes the mechanism and §9.5 does not claim it.

9. **The 26x says the signal is informative; §9.5 says conditioning on it is the
   wrong way to spend that.** The obvious follow-up is a **training-only** use
   that leaves the inference model bit-identical to the baseline — weighting the
   loss by the RPE, for instance — which is `EXP_013`'s shape: no parameter, no
   inference cost, nothing to fold, and none of the 1.71x. That is a different
   experiment and is not proposed here as a design, only as the direction the
   evidence points.

11. **RESOLVED before this file was committed, and left here because the rule it
    produced is worth adopting.** The two train-side estimators disagreed by
    ~0.005 in opposite directions per arm; §9.5's correction now shows why —
    **a logged training loss is measured at pre-update weights, and how much the
    two arms were still improving over the logged tail differed by
    +0.00334 ± 0.00096**, which is the size of the effect. What is referred is
    the standing rule: **a running training loss is not a stand-in for a
    train-split bpc**, and any future experiment that wants a train-side number
    should score a slice with `snn.evaluate.evaluate` rather than read
    `log.jsonl`.

10. **Should a control be required to be a per-example function?** §9.4 records
    that `da_rolled` couples across the batch while the arm does not. The
    alternatives were worse, and the asymmetry is bounded, but "a control must
    not introduce a property the arm lacks" is a rule this project does not
    currently have and might want.
