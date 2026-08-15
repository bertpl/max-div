import pytest

from max_div._core._cli.bm_solver_presets._executors import (
    SolverPresetBenchmarkParams,
    SolverPresetBenchmarkResult,
    _execute_single_run,
    executor_multi_parallel,
)
from max_div._core.solver import SolverPreset, TargetTimeDuration


@pytest.mark.parametrize("n_workers", [1, 2])
def test_execute_single_run(n_workers: int):
    """One run per solver kind: a plain single solve, and the parallel solver at n_workers > 1."""
    # --- arrange -----------------------------------------
    params = SolverPresetBenchmarkParams(
        preset=SolverPreset.RANDOM if n_workers == 1 else SolverPreset.SMART,
        problem_name="U1",
        problem_size=100,
        duration=TargetTimeDuration(t_target_sec=0.001),
        seed=42,
        n_workers=n_workers,
    )

    # --- act ---------------------------------------------
    result = _execute_single_run(params)

    # --- assert ------------------------------------------
    assert isinstance(result, SolverPresetBenchmarkResult)
    assert result.params == params
    assert result.t_elapsed_sec > 0.0  # end-to-end span: build() + solve


def test_executor_multi_parallel_runs_parallel_scope_serially():
    """Parallel runs execute outside the pool, and both kinds land in one result list."""
    # --- arrange -----------------------------------------
    def _params(n_workers: int) -> SolverPresetBenchmarkParams:
        return SolverPresetBenchmarkParams(
            preset=SolverPreset.SMART,
            problem_name="U1",
            problem_size=100,
            duration=TargetTimeDuration(t_target_sec=0.001),
            seed=1,
            n_workers=n_workers,
        )

    scope = [_params(1), _params(2)]

    # --- act ---------------------------------------------
    results = executor_multi_parallel(scope, n_processes=1)

    # --- assert ------------------------------------------
    assert {r.params for r in results} == set(scope)
