import numpy as np

from max_div.solver import SolverPreset, TargetTimeDuration

from ._models import SolverPresetBenchmarkParams
from ._utils import estimate_execution_time_sec_multi, get_n_processes


def determine_benchmark_scope_for_max_duration(
    presets: list[SolverPreset],
    problems: list[str],
    size: int,
    max_duration_sec: float,
    max_run_duration_sec: float | None = None,
) -> tuple[float, list[SolverPresetBenchmarkParams]]:
    """
    Compute full list of benchmark runs to be executed based on presets, problems, size, and target duration.
    This method auto-tunes speed to fall just within the target duration.
    Returns (speed, scope)-tuple
    """

    def _get_scope_for_speed(_speed: float) -> list[SolverPresetBenchmarkParams]:
        return determine_benchmark_scope(presets, problems, size, _speed, max_run_duration_sec)

    def _get_duration_for_speed(_speed: float) -> float:
        return estimate_execution_time_sec_multi(_get_scope_for_speed(_speed))

    # check speed=0
    lb_speed = 0.0
    lb_duration = _get_duration_for_speed(lb_speed)
    if lb_duration <= max_duration_sec:
        # slowest setting is already fast enough
        return lb_speed, _get_scope_for_speed(lb_speed)

    # check speed=1
    ub_speed = 1.0
    ub_duration = _get_duration_for_speed(ub_speed)
    if ub_duration >= max_duration_sec:
        # fastest setting is still too slow
        return ub_speed, _get_scope_for_speed(ub_speed)

    # bisection
    for _ in range(30):
        mid_speed = 0.5 * (lb_speed + ub_speed)
        mid_duration = _get_duration_for_speed(mid_speed)
        if mid_duration <= max_duration_sec:
            ub_speed = mid_speed
        else:
            lb_speed = mid_speed

    return ub_speed, _get_scope_for_speed(ub_speed)


def determine_benchmark_scope(
    presets: list[SolverPreset],
    problems: list[str],
    size: int,
    speed: float,
    max_run_duration_sec: float | None = None,
) -> list[SolverPresetBenchmarkParams]:
    """Compute full list of benchmark runs to be executed based on presets, problems, size, and speed."""

    # --- speed-dependent settings ------------------------
    interp_speed = [0.0, 0.5, 0.99, 1.0]
    interp_max_duration_sec = [24 * 3600.0, 2 * 3600.0, 2.0, 1e-3]
    interp_min_duration_sec = [1.0, 1.0, 1.0, 1e-4]
    n_processes = get_n_processes(1000)  # number of processes used for the largest scopes
    interp_n_runs = [1000, 4 * n_processes, 2 * n_processes, 2]

    if max_run_duration_sec is not None:
        max_duration_sec = max_run_duration_sec
    else:
        max_duration_sec = float(np.interp(speed, interp_speed, interp_max_duration_sec))
    min_duration_sec = float(np.interp(speed, interp_speed, interp_min_duration_sec))
    n_runs = float(np.interp(speed, interp_speed, interp_n_runs))

    durations_sec = [
        max_duration_sec * ((min_duration_sec / max_duration_sec) ** (i_run / (n_runs - 1)))
        for i_run in range(round(n_runs) + 2)
    ]
    durations_sec = sorted({max(min_duration_sec, d) for d in durations_sec})

    # --- final scope -------------------------------------
    return [
        SolverPresetBenchmarkParams(
            preset=preset,
            problem_name=problem,
            problem_size=size,
            duration=TargetTimeDuration(t_target_sec=duration_sec),
            seed=seed,
        )
        for problem in problems
        for preset in presets
        for seed, duration_sec in enumerate(durations_sec, start=1)
    ]
