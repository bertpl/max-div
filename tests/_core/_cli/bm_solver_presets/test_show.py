from pathlib import Path

from max_div._core._cli.bm_solver_presets._models import (
    SolverPresetBenchmarkExecutionInfo,
    SolverPresetBenchmarkParams,
    SolverPresetBenchmarkResult,
)
from max_div._core._cli.bm_solver_presets.show import show_solver_presets_benchmark_results
from max_div._core.solver import Score, SolverPreset, TargetTimeDuration


# =================================================================================================
#  Helpers
# =================================================================================================
def _result(duration_sec: float, n_workers: int, diversity: float) -> SolverPresetBenchmarkResult:
    """Build a benchmark result for one (duration, worker-count) point on problem U1."""
    params = SolverPresetBenchmarkParams(
        preset=SolverPreset.SMART,
        problem_name="U1",
        problem_size=1000,
        duration=TargetTimeDuration(t_target_sec=duration_sec),
        seed=1,
        n_workers=n_workers,
    )
    return SolverPresetBenchmarkResult(
        params=params,
        execution_info=SolverPresetBenchmarkExecutionInfo(pid=1, t_start=0.0, t_end=duration_sec),
        t_elapsed_sec=duration_sec,
        n_iterations=100,
        score=Score(size=1.0, constraints=1.0, diversity=diversity, div_tie_breakers=()),
    )


# =================================================================================================
#  Tests
# =================================================================================================
def test_show_renders_ragged_parallel_arm(tmp_path: Path) -> None:
    """A parallel arm covering only the longer budgets renders a placeholder at the short rows.

    The union of durations across columns includes budgets the parallel arm never ran, so those
    cells have no sample; the table must render them rather than reduce over an empty list.
    """
    # --- arrange -----------------------------------------
    # single-worker SMART at 0.5s and 2.0s; the parallel arm only at 2.0s
    results = [
        _result(0.5, n_workers=1, diversity=0.10),
        _result(2.0, n_workers=1, diversity=0.12),
        _result(2.0, n_workers=8, diversity=0.13),
    ]
    out_file = tmp_path / "preset_results_U1_1000.md"

    # --- act ---------------------------------------------
    show_solver_presets_benchmark_results(results, markdown=True, markdown_file_name=str(out_file))

    # --- assert ------------------------------------------
    text = out_file.read_text()
    assert "SMART (parallel x8)" in text  # the arm gets its own column
    assert "—" in text  # the 0.5s row's parallel cell is a placeholder
