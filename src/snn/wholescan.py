"""Whole-scan NVRTC kernels: one launch per layer instead of one per timestep.

**Outside the Phase 1-5 protocol.** Nothing here is pre-registered and nothing here
moves a reported number: the rule this module holds itself to is that it is
**bit-identical** to the per-timestep jiterator path -- every spike, every membrane,
every gradient, to the last bit, signed zeros included. `docs/perf/WHOLESCAN.md` is
the running record; `tests/test_wholescan_equivalence.py` is the gate.

Why this module exists at all
-----------------------------
`snn/kernels.py` took the LIF timestep from ~13 launches to one, and CUDA-graph
capture then removed the launch overhead that was left. What remained was measured
on the graphed step (`docs/perf/README.md` §1): everything scan-related is 29.7% of
the plain LIF's step at d=512 and 48.5% of the adopted two-compartment arm's, and
only about a third of that is the fused kernels themselves. The rest is
`torch.stack` of 256 per-timestep tensors -- already at memory bandwidth, so there
is no cheaper copy, only not copying -- and L launches each way whose per-element
state has to round-trip through device memory between every pair of them.

All three recurrences (`kernels.py`, `twocomp.py`, `twocomp_detach.py`) are strictly
per-(batch, channel): the only cross-element operations are the stacks and the two
per-channel reductions, and both sit outside the time loop. So the scan can be ONE
kernel per layer: one thread per (batch, channel), looping over `t` with the
membrane -- or the adjoint, going backward -- held in a register, reading and
writing the `[B, L, d]` tensors in place. No stack, no per-step launch, no
round-trip. jiterator cannot express that (it is elementwise-only, §2.3), so the
kernel is compiled with NVRTC directly.

NO ARITHMETIC LIVES HERE
------------------------
The kernel source is the project's COMMITTED FUNCTOR TEXT -- `forward_source()` /
`backward_source()`, `twocomp_forward_source()` / `twocomp_backward_source()`,
`twocomp_detach_backward_source()`, imported and pasted, never retyped -- plus a
thin time-loop wrapper that does indexing and nothing else. Two consequences, both
deliberate:

  * there is no second derivation of any neuron to drift from the first, and a
    mutation of the functor text (`scripts/audit/08_mutation_campaign.py`) reaches
    this path automatically;
  * the wrapper is the only new thing that can be wrong, and what it can get wrong
    is *which element goes where* -- stride, offset, the `t == 0` initial-state
    branch, which product feeds which reduction. Those have their own campaign,
    `scripts/audit/12_wholescan_mutations.py`.

Why it is bit-identical, and the one way it was not
---------------------------------------------------
Same compiler (the NVRTC inside the torch wheel), same flag set as jiterator
(`_nvrtc_flags`), same functor text. The FORWARD functors write every add that
consumes a product as `__fadd_rn(__fmul_rn(...))`, so NVRTC has no FMA to form in
either context, and the carried membrane enters the next step only through
`__fmul_rn`. Nothing to get wrong.

The BACKWARD functors are natural C++ and NVRTC *does* contract them -- two or
three FMAs per step -- and **which multiply it pairs with which add is decided per
expression DAG**. In the per-timestep path the carried adjoint (`beta * g`,
`beta_s_c * gs`) is materialised to memory between launches, so the next step sees
an opaque load. Inlined into a time loop it is an SSA value the optimiser can see
through, and wherever the next step ADDS it directly -- LIF `reset="none"`, where
`dv == 1` folds `grad_v_next * dv` to `grad_v_next`; both two-compartment
backwards, where `gs = grad_vs_next + w_c * gv` -- NVRTC re-pairs the FMA onto the
carry multiply, across the timestep boundary. Measured on the naive whole-scan
kernel: 0/12 configurations bit-equal for those three variants, ~37-62% of
`grad_cur` elements off by one ulp and `grad_beta_s` by up to 3e-05, while hard,
soft and detached reset were 12/12 -- the carry is only ever *multiplied* there, so
there is nothing to re-pair, and a prototype that tried hard reset alone got lucky.

A one-ulp gradient is invisible to every tolerance in this repository and changes
every trained weight. The fix is a **contraction barrier**: each backward step is
called through an `__attribute__((noinline))` trampoline (`_NOINLINE`), so the step
arithmetic exists exactly once, as a closed function whose leaves are its float
arguments -- the same expression DAG, FMA sites and operand pairing included, that
jiterator compiles per element. With it all six variants are bit-identical on
every output and every gradient. It is applied to ALL SIX, including the three that
do not need it today: it costs nothing measurable, and "this variant happens not to
need it" is a fact about one NVRTC build's heuristics.

The two per-channel gradients are REDUCTIONS over batch and time. The kernel emits
the per-element PRODUCTS (`gv * vs_t`, `gvs * vs_{t-1}`: one IEEE multiply each,
stored, so no add can consume it) into contiguous `[B, L, d]` tensors and the SAME
`torch` `.sum(dim=(0, 1))` the per-step path calls reduces them -- same reduce
kernel, same launch configuration, same bits. **An in-kernel reduction would be
faster and would NOT be bit-identical** (a different summation order); it is
catalogued in `docs/perf/README.md` §4 and not adopted.

The costs of this choice, all real
----------------------------------
  * A second private-API surface (`torch.cuda._utils._cuda_load_module`), next to
    jiterator's -- risk R11 again, behind the same hard torch pin.
  * NVRTC is driven through `ctypes` and its public C API rather than through
    `torch.cuda._compile_kernel`, because that helper calls
    `cpp_extension.include_paths("cuda")` and RAISES when `CUDA_HOME` is unset.
    There is no CUDA toolkit on this box (or on most boxes that only have the
    wheel), and these kernels include no header, so a dummy `CUDA_HOME` would be a
    process-wide side effect bought for nothing.
  * **Contraction is a compiler heuristic, not a contract.** A torch upgrade brings
    a new NVRTC, and it may pair differently -- inside the trampoline as well as
    around it. So availability is not only "does it compile and launch"
    (`wholescan_available`): each variant is also SELF-CHECKED on first use, a tiny
    scan through both paths compared bit for bit (`self_check`), and a variant that
    fails degrades to the per-timestep path with a warning instead of silently
    changing gradients. The per-timestep Functions stay in the tree, untouched, as
    that oracle and as the fallback.
  * `_CudaKernel` takes raw `data_ptr()`s and ignores strides, so a non-contiguous
    tensor would be silently MIS-READ rather than rejected. Every tensor that
    crosses the launch boundary goes through `_require`; the dispatchers send
    anything unsupported (non-contiguous `cur`, another device, an index that does
    not fit the launch grid) to the per-timestep path, which reads strides
    correctly. fp32 only, for the reason `kernels.py` gives.

Selecting the implementation
----------------------------
`SNN_SCAN_IMPL=wholescan|jiterator`, read once per process; default `wholescan`.
Deliberately an environment variable and NOT a `Config` field: a new field makes
every pre-existing reference run fail the K1 gate, and the two implementations are
the same function bit for bit, which is exactly what a Config field must not be
used to distinguish. `scan_impl_record()` is what `Trainer` writes into
`environment.json` so a run can say which one it used.

Capture safety: a launch is `cuLaunchKernel` on the current stream with no host
sync, so the whole scan records into a CUDA graph exactly as the jiterator path
does. The self-check DOES sync, so it must never run inside a capture -- and does
not: a variant that has not been checked when a capture begins uses the per-step
path for that call (`wholescan_selected`). Every capture site in the tree warms up
eagerly first (`Trainer._capture`, `snnchat.train`, `snnchat.stepper`), which is
where the check actually runs.
"""

from __future__ import annotations

import contextlib
import ctypes
import functools
import os
import warnings
from pathlib import Path
from typing import Callable, Iterator

import torch
from torch import Tensor

from snn.kernels import RESET_CODES, RESET_MODES, backward_source, forward_source
from snn.twocomp import twocomp_backward_source, twocomp_forward_source
from snn.twocomp_detach import twocomp_detach_backward_source

__all__ = [
    "SCAN_IMPLS",
    "VARIANTS",
    "scan_impl",
    "use_scan_impl",
    "scan_impl_record",
    "wholescan_available",
    "wholescan_selected",
    "self_check",
    "verify_all",
    "lif_forward_scan_source",
    "lif_backward_scan_source",
    "twocomp_forward_scan_source",
    "twocomp_backward_scan_source",
    "lif_forward_scan",
    "lif_backward_scan",
    "twocomp_forward_scan",
    "twocomp_backward_scan",
    "WholeScanLIF",
    "WholeScanTwoCompartment",
    "WholeScanTwoCompDetach",
    "lif_scan_whole",
    "twocomp_scan_whole",
    "twocomp_detach_scan_whole",
    "clear_wholescan_cache",
]


# --------------------------------------------------------------------------
# which implementation
# --------------------------------------------------------------------------

#: Order is normative only in that the first entry is the default.
SCAN_IMPLS: tuple[str, ...] = ("wholescan", "jiterator")

_ENV_VAR = "SNN_SCAN_IMPL"

#: One name per compiled backward. The LIF's four are `kernels.RESET_MODES`.
VARIANTS: tuple[str, ...] = (
    *(f"lif_{reset}" for reset in RESET_MODES), "twocomp", "twocomp_detach",
)

# In-process override, innermost last. Exists for the tests, which have to run
# both implementations in one process; a run selects with the environment.
_OVERRIDE: list[str] = []


def _check_impl(name: str) -> str:
    if name not in SCAN_IMPLS:
        raise ValueError(
            f"unknown scan implementation {name!r}; expected one of {SCAN_IMPLS}"
        )
    return name


@functools.lru_cache(maxsize=1)
def _impl_from_environment() -> str:
    """`SNN_SCAN_IMPL`, read ONCE. Unset or empty means the default.

    A value that is set and not recognised raises rather than falling back: a
    typo that silently selected the default would be a run that believes it used
    one path and used the other, which is the failure this variable exists to
    prevent.
    """
    raw = os.environ.get(_ENV_VAR, "").strip().lower()
    return _check_impl(raw) if raw else SCAN_IMPLS[0]


def scan_impl() -> str:
    """The REQUESTED implementation: `"wholescan"` or `"jiterator"`.

    Requested, not realised. `"wholescan"` still takes the per-timestep path for
    an input it does not support and for a variant whose self-check failed --
    see `wholescan_selected`, and `scan_impl_record` for what a run writes down.
    """
    return _OVERRIDE[-1] if _OVERRIDE else _impl_from_environment()


@contextlib.contextmanager
def use_scan_impl(name: str) -> Iterator[None]:
    """Select an implementation for the duration of a `with` block.

    For tests and benchmarks that compare the two in one process. Python-side
    dispatch only: a CUDA graph captured inside the block keeps whatever it
    captured, like every other Python branch a capture meets.
    """
    _OVERRIDE.append(_check_impl(name))
    try:
        yield
    finally:
        _OVERRIDE.pop()


# --------------------------------------------------------------------------
# CUDA source -- time-loop wrappers. Indexing only; see the module docstring.
# --------------------------------------------------------------------------

# Every piece of index arithmetic is written ONCE and pasted into each kernel
# that needs it, for the reason `twocomp.py` keeps its source text distinct from
# `kernels.py`'s: the mutation campaign requires an anchor to appear exactly once
# per file, and an off-by-one that had to be injected in four places would be
# four mutations pretending to be one.
#
# 64-bit throughout. `B * L * d` passes 2^31 at shapes this project can reach
# (160 x 256 x 1024 x 52 layers' worth it is not, but one [B, L, d] at d = 8192
# is), and a 32-bit index that wraps reads a valid address and returns a
# plausible number.

# One thread per (batch, channel). `base` is element [b, 0, c] of a contiguous
# [B, L, d] tensor.
_THREAD = """
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (long long)B * d) return;
    const long long b = i / d, c = i - b * d;
    const long long base = b * (long long)L * d + c;
"""

# Element [b, t, c]: consecutive timesteps of one channel are `d` apart.
_STEP_INDEX = "        const long long k = base + (long long)t * d;\n"

# The two-compartment state is one [B, 2d] tensor, fast half then slow half
# (`twocomp._check`).
_STATE_INDEX = (
    "    const long long kf = b * 2 * (long long)d + c;\n"
    "    const long long ks = kf + d;\n"
)

# THE CONTRACTION BARRIER. See the module docstring: without it NVRTC re-pairs an
# FMA across the timestep boundary onto the carried adjoint, and three of the six
# backwards differ from the per-timestep path by one ulp.
_NOINLINE = "__attribute__((noinline))"

_LIF_FORWARD_KERNEL = """
extern "C" __global__
void wholescan_lif_fwd(const float* cur, const float* v0,
                       float* spikes, float* v_pre, float* v_final,
                       double beta_d, double thr_d, int B, int L, int d, int save) {
    const float beta = (float)beta_d, thr = (float)thr_d;
@THREAD@
    float v = v0[i];
    for (int t = 0; t < L; ++t) {
@STEP_INDEX@
        float vn, s, vp;
        lif_forward<float>(v, cur[k], beta, thr, vn, s, vp);
        spikes[k] = s;
        if (save) v_pre[k] = vp;
        v = vn;
    }
    v_final[i] = v;
}
"""

_LIF_TRAMPOLINE = """
@NOINLINE@ void lif_step(float gs, float gv, float vp, float beta, float thr, float alpha,
                         float& gc, float& gp) {
    lif_backward<float>(gs, gv, vp, beta, thr, alpha, gc, gp);
}
"""

_LIF_BACKWARD_KERNEL = """
extern "C" __global__
void wholescan_lif_bwd(const float* grad_spikes, const float* grad_v_final,
                       const float* v_pre, float* grad_cur, float* grad_v0,
                       double beta_d, double thr_d, double alpha_d, int B, int L, int d) {
    const float beta = (float)beta_d, thr = (float)thr_d, alpha = (float)alpha_d;
@THREAD@
    float g = grad_v_final[i];
    for (int t = L - 1; t >= 0; --t) {
@STEP_INDEX@
        float gc, gp;
        const float g_in = g;
        @STEP@(grad_spikes[k], g_in, v_pre[k], beta, thr, alpha, gc, gp);
        grad_cur[k] = gc;
        g = gp;
    }
    grad_v0[i] = g;
}
"""

_TC_FORWARD_KERNEL = """
extern "C" __global__
void wholescan_tc_fwd(const float* cur, const float* v0, const float* w, const float* beta_s,
                      float* spikes, float* v_pre, float* vs_seq, float* v_final,
                      double beta_f_d, double thr_d, int B, int L, int d, int save) {
    const float beta_f = (float)beta_f_d, thr = (float)thr_d;
@THREAD@
@STATE_INDEX@
    const float w_c = w[c], bs_c = beta_s[c];
    float vf = v0[kf];
    float vs = v0[ks];
    for (int t = 0; t < L; ++t) {
@STEP_INDEX@
        float vfn, vsn, s, vp;
        tc_forward<float>(vf, vs, cur[k], w_c, bs_c, beta_f, thr, vfn, vsn, s, vp);
        spikes[k] = s;
        if (save) { v_pre[k] = vp; vs_seq[k] = vsn; }
        vf = vfn; vs = vsn;
    }
    v_final[kf] = vf;
    v_final[ks] = vs;
}
"""

_TC_TRAMPOLINE = """
@NOINLINE@ void tc_step(float gs, float gf, float gsl, float vp, float vs, float w_c, float bs_c,
                        float beta_f, float thr, float alpha,
                        float& gc, float& gfp, float& gsp, float& e1, float& e2) {
    tc_backward<float>(gs, gf, gsl, vp, vs, w_c, bs_c, beta_f, thr, alpha, gc, gfp, gsp, e1, e2);
}
"""

_TCD_TRAMPOLINE = """
@NOINLINE@ void tcd_step(float gs, float gf, float gsl, float vp, float vs, float w_c, float bs_c,
                         float beta_f, float thr, float alpha,
                         float& gc, float& gfp, float& gsp, float& e1, float& e2) {
    tcd_backward<float>(gs, gf, gsl, vp, w_c, bs_c, beta_f, thr, alpha, gc, gfp, gsp, e1, e2);
}
"""

# One wrapper for both two-compartment backwards. They differ in the functor and
# in whether it reads `vs` (the detached arm's does not -- `twocomp_detach.py`,
# "NOTE WHAT IS ABSENT"); the trampolines absorb that, so `vs` is dropped at the
# trampoline for the detached arm and the wrapper text is shared. `vs_seq` is an
# input to BOTH kernels regardless, because the two products below need it.
#
# `gw_elem` / `gbs_elem` are the per-element terms of the two reductions:
# dL/dw at step t is dL/dv_t * vs_t, and dL/dbeta_s at step t is
# dL/dvs_t * vs_{t-1}, with vs_{-1} the slow half of v0 -- the shift `twocomp.py`
# builds with a `torch.cat`, here a branch on `t == 0`.
_TC_BACKWARD_KERNEL = """
extern "C" __global__
void wholescan_tc_bwd(const float* grad_spikes, const float* grad_v_final,
                      const float* v_pre, const float* vs_seq, const float* v0,
                      const float* w, const float* beta_s,
                      float* grad_cur, float* gw_elem, float* gbs_elem, float* grad_v0,
                      double beta_f_d, double thr_d, double alpha_d, int B, int L, int d) {
    const float beta_f = (float)beta_f_d, thr = (float)thr_d, alpha = (float)alpha_d;
@THREAD@
@STATE_INDEX@
    const float w_c = w[c], bs_c = beta_s[c];
    float gvf = grad_v_final[kf];
    float gvs = grad_v_final[ks];
    for (int t = L - 1; t >= 0; --t) {
@STEP_INDEX@
        float gc, gfp, gsp, e1, e2;
        const float gvf_in = gvf, gvs_in = gvs;
        @STEP@(grad_spikes[k], gvf_in, gvs_in, v_pre[k], vs_seq[k], w_c, bs_c,
               beta_f, thr, alpha, gc, gfp, gsp, e1, e2);
        grad_cur[k] = gc;
        const float vs_prev = (t > 0) ? vs_seq[k - d] : v0[ks];
        gw_elem[k] = e1 * vs_seq[k];
        gbs_elem[k] = e2 * vs_prev;
        gvf = gfp; gvs = gsp;
    }
    grad_v0[kf] = gvf;
    grad_v0[ks] = gvs;
}
"""


def _assemble(functor: str, trampoline: str, kernel: str, step: str,
              barrier: bool = True) -> str:
    text = functor + trampoline + kernel
    for token, value in (
        ("@THREAD@", _THREAD.strip("\n")),
        ("@STEP_INDEX@", _STEP_INDEX.rstrip("\n")),
        ("@STATE_INDEX@", _STATE_INDEX.rstrip("\n")),
        ("@NOINLINE@", _NOINLINE if barrier else "inline"),
        ("@STEP@", step),
    ):
        text = text.replace(token, value)
    return text


def lif_forward_scan_source(reset: str) -> str:
    """CUDA C++ text for the whole-scan LIF forward. Exposed for inspection/tests."""
    return _assemble(forward_source(reset), "", _LIF_FORWARD_KERNEL, "")


def lif_backward_scan_source(reset: str, barrier: bool = True) -> str:
    """CUDA C++ text for the whole-scan LIF backward. Exposed for inspection/tests.

    `barrier=False` builds the kernel this module must never ship: the trampoline
    declared `inline`, so the step functor is folded into the time loop with the
    carry visible to the optimiser. It exists so the gate can carry its own
    NEGATIVE CONTROL -- a barrier-less build asserted NOT bit-equal to the oracle
    -- which is the only evidence that the bitwise comparison can see a one-ulp
    gradient at all.
    """
    return _assemble(backward_source(reset), _LIF_TRAMPOLINE, _LIF_BACKWARD_KERNEL,
                     "lif_step", barrier)


def twocomp_forward_scan_source() -> str:
    """CUDA C++ text for the whole-scan two-compartment forward (both arms)."""
    return _assemble(twocomp_forward_source(), "", _TC_FORWARD_KERNEL, "")


def twocomp_backward_scan_source(kind: str, barrier: bool = True) -> str:
    """CUDA C++ text for a whole-scan two-compartment backward.

    `kind` is `"twocomp"` or `"twocomp_detach"`. `barrier=False` is the negative
    control; see `lif_backward_scan_source`.
    """
    if kind == "twocomp":
        functor, trampoline, step = twocomp_backward_source(), _TC_TRAMPOLINE, "tc_step"
    elif kind == "twocomp_detach":
        functor, trampoline, step = (
            twocomp_detach_backward_source(), _TCD_TRAMPOLINE, "tcd_step")
    else:
        raise ValueError(f"unknown two-compartment kind {kind!r}")
    return _assemble(functor, trampoline, _TC_BACKWARD_KERNEL, step, barrier)


# --------------------------------------------------------------------------
# NVRTC, by its public C API
# --------------------------------------------------------------------------

def _nvrtc_flags(device: int) -> list[str]:
    """jiterator's flag set, exactly -- the flags are part of the identity argument.

    `--std=c++20 --gpu-architecture=sm_<cc> -default-device -DNDEBUG`, with no
    `--fmad` either way: FMA contraction is ON by NVRTC's default in both paths,
    which is what makes the two compile the functor text to the same arithmetic.
    `-default-device` is what lets the functor text carry no `__device__`
    qualifier. The architecture is read from the device, never assumed.
    """
    major, minor = torch.cuda.get_device_capability(device)
    return [
        "--std=c++20",
        f"--gpu-architecture=sm_{major}{minor}",
        "-default-device",
        "-DNDEBUG",
    ]


@functools.lru_cache(maxsize=1)
def _nvrtc() -> ctypes.CDLL:
    """The NVRTC that shipped inside the torch wheel, loaded by full path.

    By path first, so it is the wheel's own build -- the one jiterator uses --
    even on a machine that also has a CUDA toolkit on its search path; a toolkit
    of a different version is a different compiler with its own contraction
    heuristics. Falls back to the loader's search by the names torch itself uses.
    Loading a DLL the process already holds is a reference-count bump, so this
    leaves no process-wide side effect: no environment variable, no search-path
    edit.
    """
    lib_dir = Path(torch.__file__).resolve().parent / "lib"
    found = sorted(
        p for pattern in ("nvrtc64_*.dll", "libnvrtc.so*", "libnvrtc-*.so*")
        for p in lib_dir.glob(pattern)
        if "builtins" not in p.name and ".alt." not in p.name
    )
    major = (torch.version.cuda or "0").split(".")[0]
    names = [str(p) for p in found]
    names += [f"nvrtc64_{major}0_0.dll", f"libnvrtc.so.{major}", "libnvrtc.so"]
    for name in names:
        try:
            lib = ctypes.CDLL(name)
        except OSError:
            continue
        lib.nvrtcCreateProgram.argtypes = [
            ctypes.POINTER(ctypes.c_void_p), ctypes.c_char_p, ctypes.c_char_p,
            ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p]
        lib.nvrtcCompileProgram.argtypes = [
            ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_char_p)]
        lib.nvrtcDestroyProgram.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
        for getter in ("nvrtcGetProgramLogSize", "nvrtcGetPTXSize"):
            getattr(lib, getter).argtypes = [
                ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t)]
        for getter in ("nvrtcGetProgramLog", "nvrtcGetPTX"):
            getattr(lib, getter).argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        return lib
    raise OSError("no NVRTC library found in the torch wheel or on the loader path")


def _compile_ptx(source: str, name: str, device: int) -> bytes:
    """NVRTC-compile `source` to PTX. The compile LOG is in the exception on failure."""
    lib = _nvrtc()
    prog = ctypes.c_void_p()
    rc = lib.nvrtcCreateProgram(
        ctypes.byref(prog), source.encode(), f"{name}.cu".encode(), 0, None, None)
    if rc != 0:
        raise RuntimeError(f"nvrtcCreateProgram failed for {name} (code {rc})")
    try:
        flags = _nvrtc_flags(device)
        argv = (ctypes.c_char_p * len(flags))(*(f.encode() for f in flags))
        rc = lib.nvrtcCompileProgram(prog, len(flags), argv)
        if rc != 0:
            size = ctypes.c_size_t()
            lib.nvrtcGetProgramLogSize(prog, ctypes.byref(size))
            log = ctypes.create_string_buffer(size.value or 1)
            lib.nvrtcGetProgramLog(prog, log)
            raise RuntimeError(
                f"NVRTC failed to compile {name} (code {rc}, flags {flags}):\n"
                f"{log.value.decode(errors='replace')}"
            )
        size = ctypes.c_size_t()
        if lib.nvrtcGetPTXSize(prog, ctypes.byref(size)) != 0:
            raise RuntimeError(f"nvrtcGetPTXSize failed for {name}")
        ptx = ctypes.create_string_buffer(size.value)
        if lib.nvrtcGetPTX(prog, ptx) != 0:
            raise RuntimeError(f"nvrtcGetPTX failed for {name}")
        return ptx.value
    finally:
        lib.nvrtcDestroyProgram(ctypes.byref(prog))


# --------------------------------------------------------------------------
# compiled-kernel cache
# --------------------------------------------------------------------------

# Keyed on the SOURCE TEXT, as jiterator's own cache is, so "detached" costs no
# second forward compile (its forward source is byte-identical to "hard"'s) and
# a test that edits the source gets the kernel it asked for. And on the device,
# because a loaded module belongs to one CUDA context.
_KERNELS: dict[tuple[str, str, int], Callable] = {}


def _kernel(name: str, source: str) -> Callable:
    """Compiled, loaded kernel `name` for the current device. Cached; lazy.

    Measured here: ~0.2 s for the first compile in a process (loading NVRTC),
    ~0.02 s for each one after it -- these translation units include no header.
    Loading the module is a driver call that cannot happen inside a CUDA-graph
    capture; `wholescan_selected` is what guarantees it never does.
    """
    device = torch.cuda.current_device()
    key = (name, source, device)
    fn = _KERNELS.get(key)
    if fn is None:
        # The one private-API touch point of this module (risk R11).
        from torch.cuda._utils import _cuda_load_module

        fn = _cuda_load_module(_compile_ptx(source, name, device), [name])[name]
        _KERNELS[key] = fn
    return fn


# Threads per block. Measured: 64..1024 moves these kernels by <= 5%; they run at
# 70-80% of pure memory bandwidth whatever this is.
_BLOCK = 256

_INT_MAX = 2**31 - 1


def _launch(name: str, source: str, n_threads: int, args: list) -> None:
    """One launch on the CURRENT stream, no host sync -- capture-safe."""
    grid = ((n_threads + _BLOCK - 1) // _BLOCK, 1, 1)
    _kernel(name, source)(grid=grid, block=(_BLOCK, 1, 1), args=args)


def _require(**tensors: Tensor) -> None:
    """The input contract of a raw launch, enforced on every tensor that crosses it.

    `_CudaKernel` hands the kernel `data_ptr()` and nothing else: strides, dtype
    and device are all invisible to it. A strided view would be read as if it
    were contiguous and return a plausible number; an fp64 tensor would be read
    as twice as many floats. So this raises, and the callers above it never let
    it: the dispatchers route unsupported inputs to the per-timestep path and the
    Functions make their inputs contiguous first. Host-side only -- no sync.
    """
    device = torch.cuda.current_device()
    for name, t in tensors.items():
        if not t.is_cuda or t.device.index != device:
            raise ValueError(
                f"whole-scan kernel: `{name}` must be on the current CUDA device "
                f"(cuda:{device}); got {t.device}"
            )
        if t.dtype is not torch.float32:
            raise TypeError(
                f"whole-scan kernel: `{name}` must be fp32 (got {t.dtype}); the "
                "membrane recursion is float-only (spec §2, §6)"
            )
        if not t.is_contiguous():
            raise ValueError(
                f"whole-scan kernel: `{name}` must be contiguous (shape "
                f"{tuple(t.shape)}, stride {t.stride()}); the kernel reads raw "
                "pointers and would silently mis-read a strided view"
            )


def _fits(cur: Tensor) -> bool:
    """Whether `[B, L, d]` fits the kernel's 32-bit shape arguments and its grid."""
    B, L, d = cur.shape
    return (
        cur.numel() > 0
        and max(B, L, d) <= _INT_MAX
        and (B * d + _BLOCK - 1) // _BLOCK <= _INT_MAX
    )


# --------------------------------------------------------------------------
# launchers -- the kernel boundary
# --------------------------------------------------------------------------

def lif_forward_scan(cur: Tensor, v0: Tensor, beta: float, thr: float, reset: str,
                     save: bool) -> tuple[Tensor, Tensor | None, Tensor]:
    """`(spikes [B, L, d], v_pre [B, L, d] | None, v_final [B, d])`, one launch.

    `save=False` is the inference forward: `v_pre` exists only for the backward,
    so without one it is neither allocated nor written, and the kernel is handed
    the spikes pointer in its place purely so the argument is a valid address.
    """
    _require(cur=cur, v0=v0)
    B, L, d = cur.shape
    spikes = torch.empty_like(cur)
    v_pre = torch.empty_like(cur) if save else None
    v_final = torch.empty_like(v0)
    _launch(
        "wholescan_lif_fwd", lif_forward_scan_source(reset), B * d,
        [cur, v0, spikes, v_pre if save else spikes, v_final,
         float(beta), float(thr), B, L, d, int(save)],
    )
    return spikes, v_pre, v_final


def lif_backward_scan(grad_spikes: Tensor, grad_v_final: Tensor, v_pre: Tensor,
                      beta: float, thr: float, alpha: float, reset: str,
                      barrier: bool = True) -> tuple[Tensor, Tensor]:
    """`(grad_cur [B, L, d], grad_v0 [B, d])`, one launch.

    `barrier=False` is the negative control and nothing else may pass it; see
    `lif_backward_scan_source`.
    """
    _require(grad_spikes=grad_spikes, grad_v_final=grad_v_final, v_pre=v_pre)
    B, L, d = v_pre.shape
    grad_cur = torch.empty_like(v_pre)
    grad_v0 = torch.empty_like(grad_v_final)
    _launch(
        "wholescan_lif_bwd", lif_backward_scan_source(reset, barrier), B * d,
        [grad_spikes, grad_v_final, v_pre, grad_cur, grad_v0,
         float(beta), float(thr), float(alpha), B, L, d],
    )
    return grad_cur, grad_v0


def twocomp_forward_scan(
    cur: Tensor, v0: Tensor, w: Tensor, beta_s: Tensor, beta_f: float, thr: float,
    save: bool,
) -> tuple[Tensor, Tensor | None, Tensor | None, Tensor]:
    """`(spikes, v_pre | None, vs_seq | None, v_final [B, 2d])`, one launch.

    Shared by both two-compartment arms, as the per-timestep forward kernel is
    (`twocomp_detach.py`: "it is the same compiled kernel").
    """
    _require(cur=cur, v0=v0, w=w, beta_s=beta_s)
    B, L, d = cur.shape
    spikes = torch.empty_like(cur)
    v_pre = torch.empty_like(cur) if save else None
    vs_seq = torch.empty_like(cur) if save else None
    v_final = torch.empty_like(v0)
    _launch(
        "wholescan_tc_fwd", twocomp_forward_scan_source(), B * d,
        [cur, v0, w, beta_s, spikes, v_pre if save else spikes,
         vs_seq if save else spikes, v_final,
         float(beta_f), float(thr), B, L, d, int(save)],
    )
    return spikes, v_pre, vs_seq, v_final


def twocomp_backward_scan(
    kind: str, grad_spikes: Tensor, grad_v_final: Tensor, v_pre: Tensor,
    vs_seq: Tensor, v0: Tensor, w: Tensor, beta_s: Tensor, beta_f: float, thr: float,
    alpha: float, barrier: bool = True,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """`(grad_cur, grad_v0 [B, 2d], grad_w [1, d], grad_beta_s [1, d])`.

    One launch plus the two reductions. The kernel emits the per-element
    products; the sums are `torch`'s, the SAME call on the same shape the
    per-timestep path makes, because that is what keeps them bit-identical
    (module docstring). `barrier=False` is the negative control only.
    """
    _require(grad_spikes=grad_spikes, grad_v_final=grad_v_final, v_pre=v_pre,
             vs_seq=vs_seq, v0=v0, w=w, beta_s=beta_s)
    B, L, d = v_pre.shape
    grad_cur = torch.empty_like(v_pre)
    gw_elem = torch.empty_like(v_pre)
    gbs_elem = torch.empty_like(v_pre)
    grad_v0 = torch.empty_like(grad_v_final)
    _launch(
        "wholescan_tc_bwd", twocomp_backward_scan_source(kind, barrier), B * d,
        [grad_spikes, grad_v_final, v_pre, vs_seq, v0, w, beta_s,
         grad_cur, gw_elem, gbs_elem, grad_v0,
         float(beta_f), float(thr), float(alpha), B, L, d],
    )
    grad_w = gw_elem.sum(dim=(0, 1)).reshape(1, d)
    grad_beta_s = gbs_elem.sum(dim=(0, 1)).reshape(1, d)
    return grad_cur, grad_v0, grad_w, grad_beta_s


# --------------------------------------------------------------------------
# autograd Functions
# --------------------------------------------------------------------------

def _refuse_double_backward(name: str, eager: str) -> None:
    # Same reason and same mechanism as `FusedLIFScan.backward`: the backward is
    # a raw kernel with no autograd history, `@once_differentiable` does not work
    # here (see the note in `snn/neuron.py`), and the engine enables grad inside
    # a backward iff the caller asked for `create_graph` -- so this fires exactly
    # then, never on the normal path and never inside a capture.
    if torch.is_grad_enabled():
        raise RuntimeError(
            f"{name} does not support double backward: its backward is a "
            "hand-written NVRTC kernel with no autograd history, so the "
            f"second-order term would be silently missing. Use {eager} if you "
            "need create_graph=True."
        )


class WholeScanLIF(torch.autograd.Function):
    """One NVRTC kernel for the whole forward scan, one for the whole backward.

    Same signature and same contract as `FusedLIFScan`, which it is bit-identical
    to: a float triple and an int `reset_code`, so nothing unhashable crosses the
    autograd boundary and the scan stays CUDA-graph-capture safe.
    """

    @staticmethod
    def forward(ctx, cur: Tensor, v0: Tensor, beta: float, thr: float,
                alpha: float, reset_code: int) -> tuple[Tensor, Tensor]:
        # `.apply` is a public entry point that skips the dispatcher, so the
        # contract is enforced again at the launch (`_require`). Contiguity is
        # repaired rather than refused here: a copy is bit-neutral, and a
        # no-op -- no kernel, no allocation -- when there is nothing to repair.
        reset = RESET_MODES[reset_code]
        spikes, v_pre, v_final = lif_forward_scan(
            cur.contiguous(), v0.contiguous(), beta, thr, reset, save=True)
        # Only v_pre is saved, exactly as the per-timestep Function does.
        ctx.save_for_backward(v_pre)
        ctx.beta = beta
        ctx.thr = thr
        ctx.alpha = alpha
        ctx.reset = reset
        return spikes, v_final

    @staticmethod
    def backward(ctx, grad_spikes: Tensor, grad_v_final: Tensor):
        _refuse_double_backward("WholeScanLIF", "lif_scan(..., fused=False)")
        (v_pre,) = ctx.saved_tensors
        # Cotangents arrive however autograd built them -- `s.sum().backward()`
        # hands over an EXPANDED tensor with stride 0 -- so they are made
        # contiguous before their pointers are taken.
        grad_cur, grad_v0 = lif_backward_scan(
            grad_spikes.contiguous(), grad_v_final.contiguous(), v_pre,
            ctx.beta, ctx.thr, ctx.alpha, ctx.reset)
        # grads for (cur, v0, beta, thr, alpha, reset_code)
        return grad_cur, grad_v0, None, None, None, None


def _tc_function_forward(ctx, cur: Tensor, v0: Tensor, w: Tensor, beta_s: Tensor,
                         beta_f: float, thr: float, alpha: float):
    v0, w, beta_s = v0.contiguous(), w.contiguous(), beta_s.contiguous()
    spikes, v_pre, vs_seq, v_final = twocomp_forward_scan(
        cur.contiguous(), v0, w, beta_s, beta_f, thr, save=True)
    # The same two [B, L, d] tensors the per-timestep Functions save. `v0` rides
    # along whole (it is [B, 2d]) because the kernel indexes its slow half.
    ctx.save_for_backward(v_pre, vs_seq, v0, w, beta_s)
    ctx.beta_f = beta_f
    ctx.thr = thr
    ctx.alpha = alpha
    return spikes, v_final


def _tc_function_backward(ctx, kind: str, grad_spikes: Tensor, grad_v_final: Tensor):
    v_pre, vs_seq, v0, w, beta_s = ctx.saved_tensors
    grad_cur, grad_v0, grad_w, grad_beta_s = twocomp_backward_scan(
        kind, grad_spikes.contiguous(), grad_v_final.contiguous(), v_pre, vs_seq,
        v0, w, beta_s, ctx.beta_f, ctx.thr, ctx.alpha)
    # grads for (cur, v0, w, beta_s, beta_f, thr, alpha)
    return grad_cur, grad_v0, grad_w, grad_beta_s, None, None, None


class WholeScanTwoCompartment(torch.autograd.Function):
    """Whole-scan twin of `FusedTwoCompartmentScan`; bit-identical to it."""

    @staticmethod
    def forward(ctx, cur: Tensor, v0: Tensor, w: Tensor, beta_s: Tensor,
                beta_f: float, thr: float, alpha: float) -> tuple[Tensor, Tensor]:
        return _tc_function_forward(ctx, cur, v0, w, beta_s, beta_f, thr, alpha)

    @staticmethod
    def backward(ctx, grad_spikes: Tensor, grad_v_final: Tensor):
        _refuse_double_backward(
            "WholeScanTwoCompartment", "twocomp_scan(..., fused=False)")
        return _tc_function_backward(ctx, "twocomp", grad_spikes, grad_v_final)


class WholeScanTwoCompDetach(torch.autograd.Function):
    """Whole-scan twin of `FusedTwoCompDetachScan`; bit-identical to it.

    The forward is `WholeScanTwoCompartment`'s -- the same compiled kernel on the
    same tensors, as the per-timestep pair share theirs -- and only the backward
    functor differs, which is the arm's single variable.
    """

    @staticmethod
    def forward(ctx, cur: Tensor, v0: Tensor, w: Tensor, beta_s: Tensor,
                beta_f: float, thr: float, alpha: float) -> tuple[Tensor, Tensor]:
        return _tc_function_forward(ctx, cur, v0, w, beta_s, beta_f, thr, alpha)

    @staticmethod
    def backward(ctx, grad_spikes: Tensor, grad_v_final: Tensor):
        _refuse_double_backward(
            "WholeScanTwoCompDetach", "twocomp_detach_scan(..., fused=False)")
        return _tc_function_backward(ctx, "twocomp_detach", grad_spikes, grad_v_final)


# --------------------------------------------------------------------------
# what the three dispatchers call
# --------------------------------------------------------------------------

def lif_scan_whole(cur: Tensor, v0: Tensor, beta: float, thr: float, alpha: float,
                   reset: str) -> tuple[Tensor, Tensor]:
    """The whole-scan branch of `snn.neuron.lif_scan`.

    Without grad nothing downstream can ask for a gradient, so `v_pre` is pure
    cost and is not built -- the inference path the plain LIF never had. The
    question can only be answered here: inside a Function's `forward`,
    `torch.is_grad_enabled()` is always False (`twocomp_scan_inference` records
    the same finding).
    """
    if not torch.is_grad_enabled():
        spikes, _, v_final = lif_forward_scan(
            cur, v0.contiguous(), beta, thr, reset, save=False)
        return spikes, v_final
    return WholeScanLIF.apply(cur, v0, beta, thr, alpha, RESET_CODES[reset])


def _tc_scan_whole(function, cur: Tensor, v0: Tensor, w: Tensor, beta_s: Tensor,
                   beta_f: float, thr: float, alpha: float) -> tuple[Tensor, Tensor]:
    if not torch.is_grad_enabled():
        spikes, _, _, v_final = twocomp_forward_scan(
            cur, v0.contiguous(), w.contiguous(), beta_s.contiguous(),
            beta_f, thr, save=False)
        return spikes, v_final
    return function.apply(cur, v0, w, beta_s, beta_f, thr, alpha)


def twocomp_scan_whole(cur: Tensor, v0: Tensor, w: Tensor, beta_s: Tensor,
                       beta_f: float, thr: float, alpha: float) -> tuple[Tensor, Tensor]:
    """The whole-scan branch of `snn.twocomp.twocomp_scan`."""
    return _tc_scan_whole(WholeScanTwoCompartment, cur, v0, w, beta_s, beta_f, thr, alpha)


def twocomp_detach_scan_whole(cur: Tensor, v0: Tensor, w: Tensor, beta_s: Tensor,
                              beta_f: float, thr: float,
                              alpha: float) -> tuple[Tensor, Tensor]:
    """The whole-scan branch of `snn.twocomp_detach.twocomp_detach_scan`.

    Its no-grad forward is the adopted arm's, which is the arm's own claim
    ("no inference cost"), and is the inference path this arm never had.
    """
    return _tc_scan_whole(WholeScanTwoCompDetach, cur, v0, w, beta_s, beta_f, thr, alpha)


# --------------------------------------------------------------------------
# availability: a probe, and a per-variant bitwise self-check
# --------------------------------------------------------------------------

_PROBE_SRC = """
extern "C" __global__
void wholescan_probe(const float* a, float* o, double k_d, int n) {
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) o[i] = a[i] + (float)k_d;
}
"""


@functools.lru_cache(maxsize=1)
def wholescan_available() -> bool:
    """True iff an NVRTC kernel can actually be compiled, loaded and run here.

    The whole round trip -- find NVRTC, compile, load the module, launch, read
    back -- for the reason `kernels.jiterator_available` gives: the failure this
    guards against is an upgrade that keeps the symbols and breaks the path, which
    must surface as a clean fall back rather than as a crash mid-run. It says
    nothing about whether the kernels are still BIT-IDENTICAL; that is
    `self_check`'s job. Cached: one NVRTC load and compile, ~0.2 s. Syncs.
    """
    try:
        if not torch.cuda.is_available():
            return False
        a = torch.ones(4, device="cuda")
        o = torch.zeros(4, device="cuda")
        _launch("wholescan_probe", _PROBE_SRC, 4, [a, o, 2.0, 4])
        torch.cuda.synchronize()
        return bool(o.eq(3.0).all())
    except Exception:
        return False


class _OracleCtx:
    """Just enough of an autograd `ctx` to run a Function's two halves by hand.

    The self-check has to run the per-timestep ORACLE, and it has to do it from
    wherever the first scan of a process happens to be: under `no_grad` in an
    evaluation, on a side stream in a warm-up. Going through the autograd engine
    would need grad mode switched back on around someone else's forward pass;
    calling `forward` and `backward` directly needs nothing, and runs the
    oracle's own code rather than a transcription of it.
    """

    saved_tensors: tuple = ()

    def save_for_backward(self, *tensors: Tensor) -> None:
        self.saved_tensors = tensors


def _bits_equal(a: Tensor, b: Tensor) -> bool:
    """Equal as BIT PATTERNS. `torch.equal` is not enough: -0.0 == +0.0."""
    return a.shape == b.shape and bool(
        (a.contiguous().view(torch.int32) == b.contiguous().view(torch.int32)).all()
    )


# Deliberately not the baseline's (0.5, 1.0, 2.0): 0.5 is a power of two and 1.0
# is 1.0, and both make a wrong rounding accidentally right
# (`tests/test_neuron_equivalence.py`, "WHY THE PARAMETERS ARE SWEPT").
_CHECK_SCALARS = (0.9, 0.7, 3.3)
_CHECK_SHAPE = (4, 33, 9)


def self_check(variant: str, barrier: bool = True) -> bool:
    """One tiny scan through both implementations, every output compared bitwise.

    True iff the whole-scan kernels for `variant` reproduce the per-timestep
    Function exactly: spikes, final state, the saved intermediates (`v_pre`,
    `vs_seq`), the no-save forward, and every gradient. This is the guard against
    a future NVRTC that pairs FMAs differently; `barrier=False` checks the
    barrier-less build instead and is how the tests prove this check CAN fail
    (it does, for `lif_none`, `twocomp` and `twocomp_detach`).

    The inputs come from a PRIVATE CPU generator. Drawing them from the global
    CUDA stream would advance it, and then merely having the whole-scan path
    enabled would change every later random number in the process -- the noise
    arm's buffers, the chat sampler's draws -- which is a bit-identity failure
    with no kernel involved. Syncs, so it must never run inside a capture.
    """
    if variant not in VARIANTS:
        raise ValueError(f"unknown scan variant {variant!r}; expected one of {VARIANTS}")
    # Imported here: `snn.neuron` imports this module.
    from snn.neuron import FusedLIFScan
    from snn.twocomp import FusedTwoCompartmentScan
    from snn.twocomp_detach import FusedTwoCompDetachScan

    B, L, d = _CHECK_SHAPE
    beta, thr, alpha = _CHECK_SCALARS
    gen = torch.Generator(device="cpu").manual_seed(0x5CA9)
    device = torch.device("cuda", torch.cuda.current_device())

    def rnd(*shape: int, scale: float = 1.0) -> Tensor:
        return (torch.randn(*shape, generator=gen) * scale).to(device)

    cur, grad_spikes = rnd(B, L, d), rnd(B, L, d, scale=1e-3)
    ctx = _OracleCtx()
    with torch.no_grad():
        if variant.startswith("lif_"):
            reset = variant[len("lif_"):]
            v0, grad_v_final = rnd(B, d), rnd(B, d, scale=1e-3)
            want = (*FusedLIFScan.forward(ctx, cur, v0, beta, thr, alpha,
                                          RESET_CODES[reset]),
                    *ctx.saved_tensors,
                    *FusedLIFScan.backward(ctx, grad_spikes, grad_v_final)[:2])
            spikes, v_pre, v_final = lif_forward_scan(cur, v0, beta, thr, reset, save=True)
            s_inf, _, v_inf = lif_forward_scan(cur, v0, beta, thr, reset, save=False)
            got = (spikes, v_final, v_pre,
                   *lif_backward_scan(grad_spikes, grad_v_final, v_pre, beta, thr,
                                      alpha, reset, barrier))
        else:
            oracle = (FusedTwoCompartmentScan if variant == "twocomp"
                      else FusedTwoCompDetachScan)
            v0, grad_v_final = rnd(B, 2 * d), rnd(B, 2 * d, scale=1e-3)
            w = rnd(1, d, scale=0.3) + 0.1
            beta_s = torch.sigmoid(rnd(1, d) + 2.9)
            want = (*oracle.forward(ctx, cur, v0, w, beta_s, beta, thr, alpha),
                    *ctx.saved_tensors[:2],
                    *oracle.backward(ctx, grad_spikes, grad_v_final)[:4])
            spikes, v_pre, vs_seq, v_final = twocomp_forward_scan(
                cur, v0, w, beta_s, beta, thr, save=True)
            s_inf, _, _, v_inf = twocomp_forward_scan(
                cur, v0, w, beta_s, beta, thr, save=False)
            got = (spikes, v_final, v_pre, vs_seq,
                   *twocomp_backward_scan(variant, grad_spikes, grad_v_final, v_pre,
                                          vs_seq, v0, w, beta_s, beta, thr, alpha,
                                          barrier))
        pairs = [*zip(got, want), (s_inf, want[0]), (v_inf, want[1])]
        # a check that compared silence against silence would pass
        live = 0.02 < float(want[0].mean()) < 0.98
        return live and len(got) == len(want) and all(_bits_equal(a, b) for a, b in pairs)


# variant -> verdict, per device. Filled lazily by `wholescan_selected`.
_VERDICTS: dict[tuple[str, int], bool] = {}
_WARNED: set[str] = set()


def _warn_once(key: str, message: str) -> None:
    if key not in _WARNED:
        _WARNED.add(key)
        warnings.warn(message, RuntimeWarning, stacklevel=3)


def _verified(variant: str) -> bool | None:
    """The self-check verdict for `variant` on the current device; None = cannot ask.

    None only while a CUDA graph is being captured and the variant has not been
    checked yet: the check compiles, loads a module and syncs, none of which may
    happen inside a capture.
    """
    key = (variant, torch.cuda.current_device())
    verdict = _VERDICTS.get(key)
    if verdict is None:
        if torch.cuda.is_current_stream_capturing():
            return None
        try:
            if not wholescan_available():
                verdict, reason = False, "is unavailable (NVRTC could not be driven)"
            else:
                verdict = self_check(variant)
                reason = "is NOT bit-identical to the per-timestep path on this stack"
        except Exception as exc:  # noqa: BLE001 - any failure means "fall back"
            verdict = False
            reason = f"could not be built or run ({type(exc).__name__}: {exc})"
        _VERDICTS[key] = verdict
        if not verdict:
            _warn_once(
                variant,
                f"snn.wholescan: the whole-scan kernel for {variant!r} {reason}; "
                "using the per-timestep jiterator path instead. Results are "
                "unchanged, throughput is not. If torch was upgraded, NVRTC's FMA "
                "contraction may have changed -- see docs/perf/WHOLESCAN.md.",
            )
    return verdict


def wholescan_selected(variant: str, cur: Tensor, *others: Tensor) -> bool:
    """The dispatchers' one question: should THIS call take the whole-scan path?

    Yes iff it was requested, the input is one the raw kernel can be trusted with,
    and the variant has passed its bitwise self-check in this process. Everything
    else goes to the per-timestep path, which is the same function bit for bit and
    reads strides properly -- so a "no" here costs throughput and never
    correctness:

      * `SNN_SCAN_IMPL=jiterator` (or `use_scan_impl`);
      * `cur` not contiguous. NOT repaired with a copy: that is a [B, L, d]
        allocation the per-timestep path does not need. The small tensors are
        made contiguous downstream instead;
      * any tensor off the current CUDA device, or not fp32 (the dispatchers have
        already raised for that; `.apply` callers get `_require`'s error);
      * a shape that does not fit the kernel's 32-bit arguments or launch grid;
      * a capture in progress before the variant was ever checked.

    Host-side checks only, and the verdict is a dict lookup after the first call,
    so this adds no sync and no launch to a captured region.
    """
    if scan_impl() != "wholescan":
        return False
    if not (cur.is_cuda and cur.is_contiguous() and _fits(cur)):
        return False
    device = torch.cuda.current_device()
    for t in (cur, *others):
        if not t.is_cuda or t.device.index != device or t.dtype is not torch.float32:
            return False
    verdict = _verified(variant)
    if verdict is None:
        _warn_once(
            f"capture:{variant}",
            f"snn.wholescan: a CUDA graph is being captured before the {variant!r} "
            "whole-scan kernel was ever run, so its self-check cannot happen here "
            "(it syncs). This capture records the per-timestep path. Run the scan "
            "once eagerly before capturing -- a warm-up step does it.",
        )
        return False
    return verdict


def verify_all() -> dict[str, bool]:
    """Self-check every variant now; `{variant: passed}`. Empty without CUDA.

    For a caller that wants the verdicts up front rather than at first use --
    `Trainer` does, so that `environment.json` records them and so that no check
    can be left to happen at capture time. Measured: ~1 s in a cold process, most
    of it the per-timestep oracle's own jiterator compiles; free thereafter.
    """
    if not (torch.cuda.is_available() and scan_impl() == "wholescan"):
        return {}
    return {variant: bool(_verified(variant)) for variant in VARIANTS}


def scan_impl_record(verify: bool = False) -> dict:
    """What a run writes down about its scan implementation (`environment.json`).

    `requested` is `scan_impl()`. `verified` maps each variant checked so far in
    this process to its verdict; with `verify=True` all six are checked first. A
    run used the whole-scan kernels iff `requested == "wholescan"` and its arm's
    variant is recorded true -- a false or missing entry means the per-timestep
    path, which is the same numbers and a different step time.
    """
    if verify:
        verify_all()
    device = torch.cuda.current_device() if torch.cuda.is_available() else -1
    return {
        "requested": scan_impl(),
        "env": os.environ.get(_ENV_VAR),
        "verified": {v: ok for (v, dev), ok in sorted(_VERDICTS.items()) if dev == device},
    }


def clear_wholescan_cache() -> None:
    """Drop compiled kernels, verdicts and probes. Only useful in tests."""
    _KERNELS.clear()
    _VERDICTS.clear()
    _WARNED.clear()
    wholescan_available.cache_clear()
    _impl_from_environment.cache_clear()
