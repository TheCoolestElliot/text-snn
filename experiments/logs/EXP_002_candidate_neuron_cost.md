# EXP_002 — What the §4.6-admitted neurons actually cost to run

**Pre-registered:** 2026-08-02, before any candidate kernel was written.
**Status:** PRE-REGISTERED — no results yet.
**Phase:** 3 (candidate improvements). Supplies the systems column of the ROI
matrix.

---

## 1. Why this has to be measured rather than argued

Phase 3's deliverable is a *ranking*, and a ranking is a bpc gain divided by a
cost. Phase 2 demolished the cost side of the inherited arithmetic twice:

* the per-kernel constant is **32.8 µs on the fused scan, not the 14.5 µs** of
  `01_reconnaissance.md` §3.4 (`02_baseline_report.md` §6.2), and
* **fusion and CUDA-graph capture are the same lever**, not independent
  multipliers, so the ~51× of §8.1.2 does not exist (§6.1).

Every candidate in the §4.6-admitted set — learned per-channel decay, a
fast/slow membrane pair, an adaptive threshold, a rotational state — needs a
neuron with **two state variables instead of one**, and some need a **per-channel
parameter tensor broadcast against `[B, d]` inside the kernel**. Whether that
still fits in one jiterator kernel per timestep is the single fact that decides
whether these candidates are systems-free or systems-expensive, and it is
precisely the kind of claim this project's protocol forbids assuming.

If a two-state neuron still costs one kernel per timestep, then the entire
admitted family is nearly free and Phase 3's ranking is decided by expected bpc
alone. If it costs two or three, the ranking is decided by a trade-off. Those are
different Phase-4 plans.

## 2. Hypotheses

| # | Prediction | Threshold |
|---|---|---|
| **Q1** | Every candidate forward kernel compiles under the bundled NVRTC and stays within jiterator's 8-input / 8-output limit | compiles and runs, or the candidate is struck |
| **Q2** | A two-state neuron still issues **exactly one kernel per timestep** | `kernels_per_timestep < 1.05`, the same bar `tests/test_kernel_count.py` holds the baseline to |
| **Q3** | Wall-clock per timestep rises by **less than 25 %** over the baseline LIF despite 2–3× the state traffic, because the workload is launch-bound | `ms_candidate / ms_baseline < 1.25` |
| **Q4** | A **per-channel `[d]` parameter broadcasts** against `[B, d]` inside a jiterator kernel without materialising a `[B, d]` copy | no extra kernel, no extra allocation of `B·d` |
| **Q5** | Learned per-channel decay's backward needs a **reduction** over the batch axis, which jiterator cannot do, so it costs **at least one extra kernel per timestep** on the backward | stated as the expected *cost*, not as a hope — Q5 predicts a penalty |

Q3 and Q5 are the two that can go against the candidates, and they are stated in
the direction that hurts. Q5 in particular predicts that the cheapest-sounding
candidate (one learned scalar per channel) is the one that breaks the one-kernel
floor on the backward pass, because a per-channel gradient is a reduction and
§2.3 rules reductions permanently out of reach here.

**Decision rule, fixed now:**

* Q2 ∧ Q3 hold for a candidate ⇒ its systems cost enters the ROI matrix as
  **negligible**, and it is ranked on expected bpc alone.
* Q2 holds, Q3 fails ⇒ the candidate carries its measured wall-clock multiplier
  into the matrix, and its Phase-4 budget is scaled by it.
* Q2 fails ⇒ the candidate is **re-priced at its measured kernel count** against
  the corrected 32.8 µs constant, and it must clear the matrix on that basis or
  be dropped. It is not to be rescued by quoting the 14.5 µs figure or by
  claiming graph capture will absorb it (§6.1 says it will not).
* Any candidate failing Q1 is **struck from the candidate set** and recorded as
  struck, with the reason.

## 3. Design

Measured at the frozen Phase-2 baseline shape — B = 128, L = 256, d = 512 — on
the same machine and the same torch build as `phase2_kernel_bench.json`, using
`snn.metrics.count_cuda_kernels` so the counts are produced by the same
instrument that asserts the baseline.

| # | Candidate neuron | State | Per-channel params |
|---|---|---|---|
| **N0** | LIF (the Phase-2 baseline) — the control for this experiment | `v` | none |
| **N1** | Learned per-channel decay: `v_t = β_c·v_{t−1} + i_t` | `v` | `β_c` |
| **N2** | Two-timescale membrane: fast/slow pair, spike on their combination | `v_f, v_s` | mixing weight |
| **N3** | Adaptive threshold: `thr_t = thr₀ + a_c·u_t`, `u` with its own decay | `v, u` | `a_c, ρ_c` |
| **N4** | Rotational (complex) membrane: decay + rotation by a learned angle | `v_re, v_im` | `θ_c` |

For each: whether it compiles, its input/output arity against the 8/8 limit,
compile time, kernels per timestep, kernels total, and forward wall-clock over
L = 256 — plus, for N1, the backward with the per-channel gradient, which is the
Q5 case.

Every candidate is measured **forward-only first** (where the one-kernel claim is
cleanest) and then, for the ones that survive, forward+backward.

This experiment produces **no bpc number and makes no claim about accuracy.** It
prices candidates; it does not evaluate them. A fast candidate is not thereby a
good one, and nothing in this file may be quoted as evidence that any of these
neurons helps.

## 4. What is recorded regardless of outcome

Per candidate: the CUDA source text, arity, compile time, kernel counts, wall
clock, the measured per-kernel cost, and — for anything that fails Q1 — the NVRTC
error verbatim. Written to `docs/reports/data/exp_002_candidate_neuron_cost.json`
and committed. Candidates that fail are kept in the file and in the report; a
candidate set with nothing struck from it is a candidate set that was not
actually tested.

## 5. Known limitations

1. **A prototype kernel is not a trained neuron.** These kernels implement the
   forward recurrence only; none has a verified backward, and none has been
   through anything like the R10 gate. No candidate may enter Phase 4 on the
   strength of this file — each needs its own mutation-tested gradient gate first,
   and that cost is itself part of the ROI.
2. **Wall-clock on this box moves ~30 % with clock and power state**
   (`snn/neuron.py` module docstring). Ratios against N0 measured in the same
   process are the reportable quantity; absolute milliseconds are not.
3. **Kernel count is the stable measurement, wall-clock is not.** Where the two
   disagree, the count is what the ROI matrix uses, exactly as in Phase 2.

## 6. Failure modes

| # | Failure | Detection |
|---|---|---|
| G1 | The baseline N0 does not reproduce `phase2_kernel_bench.json` | N0 is measured in the same run; its kernels/timestep must land within 1 % of 1.016 |
| G2 | A candidate is measured at a different shape from the baseline | Single shape constant, asserted per row |
| G3 | Compile-time cost hidden inside the timed region | Warm-up call before timing, compile time reported separately |
| G4 | Broadcasting silently materialises `[B, d]` | Peak allocation recorded per candidate |

## 7. Entry conditions

- [ ] `jiterator_available()` returns True on this machine
- [ ] `docs/reports/data/phase2_kernel_bench.json` present, for the G1 self-check

---

## 8. Results

*(appended after the run; nothing above this line is edited)*
