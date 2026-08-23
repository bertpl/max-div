import pytest

from benchmarks.solver_scaling.configs import CONFIGS, TEST_SLEEP_CONFIG, configs_for, resolve

EXPECTED_TOOLS = {"max-div", "rdkit", "dppy"}


def test_configs_cover_max_div_three_configs_plus_two_one_shots():
    # --- act / assert -----------------
    assert {c.tool for c in CONFIGS} == EXPECTED_TOOLS
    assert [c.name for c in configs_for("max-div")] == ["lean", "optimal-eager", "optimal-lazy"]
    assert [c.name for c in configs_for("rdkit")] == ["default"]
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
