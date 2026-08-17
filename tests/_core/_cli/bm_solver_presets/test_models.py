from max_div._core._cli.bm_solver_presets import results_from_json, results_to_json
from max_div._core._cli.bm_solver_presets._executors import (
    SolverPresetBenchmarkParams,
    SolverPresetBenchmarkResult,
    _execute_single_run,
)
from max_div._core.solver import SolverPreset, TargetTimeDuration


def test_result_to_from_json():
    # --- arrange ----------------------
    params = SolverPresetBenchmarkParams(
        preset=SolverPreset.RANDOM,
        problem_name="U1",
        problem_size=100,
        duration=TargetTimeDuration(t_target_sec=0.001),
        seed=42,
    )
    result = _execute_single_run(params)

    # --- act --------------------------
    json_str_1 = results_to_json([result])
    results_loaded = results_from_json(json_str_1)
    json_str_2 = results_to_json(results_loaded)

    # --- assert -----------------------
    assert isinstance(results_loaded, list)
    assert len(results_loaded) == 1
    assert isinstance(results_loaded[0], SolverPresetBenchmarkResult)

    # test obj -> str -> obj consistency
    assert result.params == results_loaded[0].params
    assert result.execution_info == results_loaded[0].execution_info
    assert result.score == results_loaded[0].score

    # test str -> obj -> str consistency
    assert json_str_1 == json_str_2


def test_params_parallel_fields():
    """n_workers drives the parallel flag and the results-column label."""
    # --- arrange ----------------------
    serial = SolverPresetBenchmarkParams(
        preset=SolverPreset.SMART,
        problem_name="U1",
        problem_size=1000,
        duration=TargetTimeDuration(t_target_sec=1.0),
        seed=1,
    )
    parallel = SolverPresetBenchmarkParams(
        preset=SolverPreset.SMART,
        problem_name="U1",
        problem_size=1000,
        duration=TargetTimeDuration(t_target_sec=1.0),
        seed=1,
        n_workers=8,
    )

    # --- act & assert -----------------
    assert not serial.is_parallel
    assert serial.column_label() == "SMART"
    assert parallel.is_parallel
    assert parallel.column_label() == "SMART (parallel x8)"

    # dict roundtrip keeps the worker count; dicts predating the field load as single-worker
    assert SolverPresetBenchmarkParams.from_dict(parallel.to_dict()) == parallel
    legacy_dict = {k: v for k, v in serial.to_dict().items() if k != "n_workers"}
    assert SolverPresetBenchmarkParams.from_dict(legacy_dict) == serial
