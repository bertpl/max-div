import numpy as np

from max_div.solver import SolverPreset, TargetTimeDuration

from ._models import SolverPresetBenchmarkParams
from ._utils import estimate_execution_time_sec_multi


def determine_benchmark_scope_for_max_duration(
    presets: list[SolverPreset],
    problems: list[str],
    size: int,
    max_duration_sec: float,
) -> tuple[float, list[SolverPresetBenchmarkParams]]:
    """
    Compute full list of benchmark runs to be executed based on presets, problems, size, and target duration.
    This method auto-tunes speed to fall just within the target duration.
    Returns (speed, scope)-tuple
    """

    def _get_scope_for_speed(_speed: float) -> list[SolverPresetBenchmarkParams]:
        return determine_benchmark_scope(presets, problems, size, _speed)

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
    for _ in range(20):
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
) -> list[SolverPresetBenchmarkParams]:
    """Compute full list of benchmark runs to be executed based on presets, problems, size, and speed."""

    # standard settings
    durations_sec = [1, 2, 4, 8, 15, 30] + [60.0 * m for m in [1, 2, 4, 8, 15, 30, 60, 120]]
    n_runs = 9

    # adjust for speed
    c_duration = 2 ** (-20 * speed)
    max_duration_sec = c_duration * max(durations_sec)
    min_duration_sec = min(1.0, max_duration_sec)
    durations_sec = sorted({max(min_duration_sec, c_duration * s) for s in durations_sec})
    durations = [TargetTimeDuration(t_target_sec=s) for s in durations_sec]
    n_runs = max(1, round(n_runs * (1.0 - speed)))

    # final scope
    seeds = list(range(1, n_runs + 1))
    return [
        SolverPresetBenchmarkParams(
            preset=preset,
            problem_name=problem,
            problem_size=size,
            duration=duration,
            seed=seed,
        )
        for problem in problems
        for preset in presets
        for duration in durations
        for seed in seeds
    ]
