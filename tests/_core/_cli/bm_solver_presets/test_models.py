from max_div._core._cli.bm_solver_presets import results_from_json, results_to_json
from max_div._core._cli.bm_solver_presets._executors import (
    SolverPresetBenchmarkParams,
    SolverPresetBenchmarkResult,
    _execute_single_run,
)
from max_div._core.solver import SolverPreset, TargetTimeDuration


def test_result_to_from_json():
    # --- arrange -----------------------------------------
    params = SolverPresetBenchmarkParams(
        preset=SolverPreset.RANDOM,
        problem_name="U1",
        problem_size=1,
        duration=TargetTimeDuration(t_target_sec=0.001),
        seed=42,
    )
    result = _execute_single_run(params)

    # --- act ---------------------------------------------
    json_str_1 = results_to_json([result])
    results_loaded = results_from_json(json_str_1)
    json_str_2 = results_to_json(results_loaded)

    # --- assert ------------------------------------------
    assert isinstance(results_loaded, list)
    assert len(results_loaded) == 1
    assert isinstance(results_loaded[0], SolverPresetBenchmarkResult)

    # test obj -> str -> obj consistency
    assert result.params == results_loaded[0].params
    assert result.execution_info == results_loaded[0].execution_info
    assert result.score == results_loaded[0].score

    # test str -> obj -> str consistency
    assert json_str_1 == json_str_2
