"""`014_calibrate_cost.py` was written for EXP_014 and generalised for EXP_015.

Generalising a script that has already produced a committed artifact is the
cheapest way to invalidate that artifact without noticing, so these tests pin the
two things EXP_014's numbers depend on: the child command it launches for
`arch="snn"`, and the parameter closed forms it checks each run against.

None of this touches the GPU or trains anything.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from snn.config import ARCH_CHOICES
from snn.model import spiking_param_count

_REPO = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location(
    "calib014", _REPO / "scripts" / "exp" / "014_calibrate_cost.py")
calib = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(calib)


def test_snn_child_command_is_what_exp_014_measured():
    """The exact argv EXP_014's committed calibration ran, minus the interpreter
    path and the absolute out_dir. If this changes,
    `docs/reports/data/exp_014_cost_calibration.json` no longer describes what
    this script does, and §13.1's cost figures stop being reproducible."""
    argv = calib._train_argv("snn", 1481, "calib_d1481", Path("OUT"))

    assert argv[1].endswith("train.py")
    assert argv[2:] == [
        "--arch", "snn",
        "--d_model", "1481",
        "--n_layers", "2",
        "--seed", "0",
        "--max_steps", "400",
        "--log_every", "25",
        "--eval_every", "4000",
        "--ckpt_every", "4000",
        "--run_name", "calib_d1481",
        "--out_dir", "OUT",
    ]


def test_snn_run_names_are_unchanged_by_generalising():
    """EXP_014's directories were `calib_d<width>`; only the non-snn arms take
    the longer name. A rename would be harmless now and confusing in a year."""
    assert calib._run_name_for("snn", 1481) == "calib_d1481"
    assert calib._run_name_for("twocomp", 1481) == "calib_twocomp_d1481"


def test_calibration_constants_are_unchanged():
    assert calib.CALIB_STEPS == 400
    assert calib.LOG_EVERY == 25
    assert calib.ESCAPE_RATE == 0.01
    assert calib.VRAM_ALARM_GIB == 3.0
    assert calib.WIDTHS == (512, 1020, 1024, 1480, 1481, 1488)


def test_param_count_reproduces_the_committed_ladder():
    """The three counts EXP_014 §2.2 committed, through the dispatcher rather
    than through `spiking_param_count` directly."""
    assert calib.param_count("snn", 205, 512, 2) == 735_437
    assert calib.param_count("snn", 205, 1020, 2) == 2_501_245
    assert calib.param_count("snn", 205, 1481, 2) == 4_997_099


def test_param_count_covers_every_arch_the_config_allows():
    """A missing arch raises rather than silently falling back to the spiking
    closed form, which would report a wrong count as if it were checked."""
    for arch in ARCH_CHOICES:
        assert calib.param_count(arch, 205, 512, 2) > 0
    with pytest.raises(ValueError, match="no committed parameter closed form"):
        calib.param_count("not-an-arch", 205, 512, 2)


ARMS_AT_WIDTH = ("snn", "twocomp", "twocomp_threshold", "gru")


def _mismatch(arch: str, d_model: int) -> float:
    ref = spiking_param_count(205, d_model, 2)
    return abs(calib.param_count(arch, 205, d_model, 2) - ref) / ref


def test_arms_are_parameter_matched_at_exp_015s_width():
    """The premise EXP_015 rests on: at `d_model = 1481` every arm lands within
    **0.2 %** of the spiking arm's 4,997,099, so width-matching *is*
    parameter-matching and no arm needs a derived width of its own.

    `gru` is matched by construction -- `GRUCharLM` derives its hidden size from
    `spiking_param_count` -- and is included so that construction is asserted
    rather than trusted.
    """
    for arch in ARMS_AT_WIDTH:
        assert _mismatch(arch, 1481) < 0.002, (
            f"{arch}: {calib.param_count(arch, 205, 1481, 2)} vs 4,997,099")


def test_the_match_is_a_property_of_width_not_a_coincidence():
    """*Why* it holds, pinned so a future arm cannot be assumed into the design.

    Each arm's extra parameters are O(d) -- one or two vectors per layer -- while
    the shared stack is O(d^2), so the fractional mismatch falls roughly as 1/d.
    At `d = 512` `twocomp` is 0.28 % over, which is outside the tolerance above;
    at `d = 1481` it is 0.12 %. A new arm whose extra parameters are O(d^2) would
    fail this test rather than quietly widening the ladder's premise.
    """
    for arch in ARMS_AT_WIDTH:
        if arch == "snn":
            continue
        assert _mismatch(arch, 1481) < _mismatch(arch, 512), arch
    # the specific figure the test above would otherwise hide
    assert 0.0027 < _mismatch("twocomp", 512) < 0.0029
