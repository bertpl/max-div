import pytest

from max_div._core._cli.benchmarks.solver_presets._models import SolverPresetBenchmarkParams
from max_div._core._cli.benchmarks.solver_presets._utils import (
    estimate_execution_time_sec_multi,
    estimate_execution_time_sec_single,
    get_n_processes,
    get_pbar_units,
)
from max_div._core.solver import SolverPreset, TargetTimeDuration


# =================================================================================================
#  Helpers
# =================================================================================================
def _params(duration_sec: float, n_workers: int = 1) -> SolverPresetBenchmarkParams:
    """Build benchmark params with only the fields under test varying."""
    return SolverPresetBenchmarkParams(
        preset=SolverPreset.SMART,
        problem_name="U1",
        problem_size=1000,
        duration=TargetTimeDuration(t_target_sec=duration_sec),
        seed=1,
        n_workers=n_workers,
    )


# =================================================================================================
#  Tests
# =================================================================================================
def test_estimate_execution_time_sec_single():
    """An end-to-end budget absorbs setup when it fits, and setup dominates a smaller budget."""
    # --- act --------------------------
    est_budget_dominant = estimate_execution_time_sec_single(_params(10.0, n_workers=4))
    est_setup_dominant = estimate_execution_time_sec_single(_params(0.001, n_workers=4))

    # --- assert -----------------------
    assert est_budget_dominant == pytest.approx(10.0)  # spawn + build spent inside the budget
    assert est_setup_dominant > 2.0  # a budget cannot cut worker spawning short


def test_estimate_execution_time_sec_multi_packs_singles_but_not_parallels():
    """Single-worker runs share the pool; parallel runs add up serially."""
    # --- arrange ----------------------
    singles = [_params(10.0) for _ in range(64)]
    parallels = [_params(10.0, n_workers=4) for _ in range(4)]

    # --- act --------------------------
    est_singles = estimate_execution_time_sec_multi(singles)
    est_parallels = estimate_execution_time_sec_multi(parallels)
    est_all = estimate_execution_time_sec_multi(singles + parallels)

    # --- assert -----------------------
    sum_singles = sum(estimate_execution_time_sec_single(p) for p in singles)
    sum_parallels = sum(estimate_execution_time_sec_single(p) for p in parallels)
    assert est_singles < sum_singles  # packed onto the pool
    assert est_parallels == pytest.approx(sum_parallels)  # strictly serial
    assert est_all == pytest.approx(est_singles + est_parallels)


def test_get_pbar_units():
    """Progress-bar units follow the estimated run time, never dropping below one."""
    # --- act & assert -----------------
    assert get_pbar_units(_params(10.0)) >= 10
    assert get_pbar_units(_params(0.001)) == 1  # never below one unit


def test_get_n_processes():
    """The process count is capped by the scope size and stays at least one."""
    # --- act & assert -----------------
    assert get_n_processes(1) == 1
    assert get_n_processes(10_000) >= 1
