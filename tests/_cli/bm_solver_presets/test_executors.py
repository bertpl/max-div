from max_div._cli.bm_solver_presets._executors import (
    SolverPresetBenchmarkParams,
    SolverPresetBenchmarkResult,
    _execute_single_run,
)
from max_div.solver import SolverPreset, TargetTimeDuration


def test_execute_single_run():
    # --- arrange -----------------------------------------
    params = SolverPresetBenchmarkParams(
        preset=SolverPreset.RANDOM,
        problem_name="U1",
        problem_size=1,
        duration=TargetTimeDuration(t_target_sec=0.001),
        seed=42,
    )

    # --- act ---------------------------------------------
    result = _execute_single_run(params)

    # --- assert ------------------------------------------
    assert isinstance(result, SolverPresetBenchmarkResult)
    assert result.params.preset == params.preset
    assert result.params.problem_name == params.problem_name
    assert result.params.problem_size == params.problem_size
    assert result.params.duration == params.duration
    assert result.params.seed == params.seed
