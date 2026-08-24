import time

import pytest

from benchmarks.exact.mip_maxmin import _remaining_sec
from benchmarks.solver_scaling.configs import CONFIGS, TEST_SLEEP_CONFIG, _exact_deadline, configs_for, resolve
from benchmarks.solver_scaling.grid import SELF_LIMIT_MARGIN_SEC

# The registry keys every configuration's `tool` must be one of (data/solver_registry.yaml).
EXPECTED_TOOLS = {
    "max-div",
    "ortools-cpsat",
    "scip",
    "highs",
    "rdkit",
    "fpsample",
    "skmatter",
    "apricot-select",
    "qc-selector",
    "dppy",
    "code-fdm",
}


def test_configs_cover_every_registered_tool():
    # --- act / assert -----------------
    assert {c.tool for c in CONFIGS} == EXPECTED_TOOLS
    assert [c.name for c in configs_for("max-div")] == ["lean", "optimal-eager", "optimal-lazy"]
    assert [c.name for c in configs_for("ortools-cpsat")] == ["feasible", "optimal"]
    assert [c.name for c in configs_for("fpsample")] == ["vanilla", "kdline"]
    assert [c.name for c in configs_for("dppy")] == ["default"]


def test_resolve_returns_the_named_config_and_rejects_unknown_ones():
    # --- act / assert -----------------
    assert resolve("max-div", "optimal-lazy").description.startswith("SMART preset")
    with pytest.raises(KeyError):
        resolve("max-div", "no-such-config")


def test_resolve_reaches_the_kill_path_fixture_outside_the_registry():
    # --- act / assert -----------------
    assert resolve("_test_sleep", "sleep") is TEST_SLEEP_CONFIG
    assert TEST_SLEEP_CONFIG.tool not in EXPECTED_TOOLS


def test_every_config_carries_a_callable_selector():
    # --- act / assert -----------------
    assert all(callable(c.select) for c in CONFIGS)


def test_exact_deadline_sits_margin_under_the_budget():
    """Pins that an exact solver's deadline is now + budget − margin, floored for tiny budgets."""
    # --- act --------------------------
    deadline = _exact_deadline(60.0)
    floored = _exact_deadline(0.0)
    now = time.monotonic()

    # --- assert -----------------------
    assert deadline - now == pytest.approx(60.0 - SELF_LIMIT_MARGIN_SEC, abs=0.1)
    assert floored - now == pytest.approx(0.1, abs=0.1)


def test_remaining_sec_shrinks_with_elapsed_time_and_never_reaches_zero():
    """Pins that time spent before a solver's limit is set comes out of the limit, not on top of it."""
    # --- act / assert -----------------
    assert _remaining_sec(time.monotonic() + 10.0) == pytest.approx(10.0, abs=0.1)
    assert _remaining_sec(time.monotonic() - 5.0) == 0.01
