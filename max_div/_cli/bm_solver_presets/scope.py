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
    for speed in np.linspace(0.0, 1.0, 1001):
        speed = float(speed)
        scope = determine_benchmark_scope(presets, problems, size, speed)
        t_est = estimate_execution_time_sec_multi(scope)
        if t_est <= max_duration_sec:
            return speed, scope

    # fallback (too slow, but fastest we can do)
    return 1.0, determine_benchmark_scope(presets, problems, size, 1.0)


def determine_benchmark_scope(
    presets: list[SolverPreset],
    problems: list[str],
    size: int,
    speed: float,
) -> list[SolverPresetBenchmarkParams]:
    """Compute full list of benchmark runs to be executed based on presets, problems, size, and speed."""

    # standard settings
    durations_sec = [1, 2, 4, 8, 15, 30] + [60.0 * m for m in [1, 2, 4, 8, 15, 30, 60]]
    n_runs = 7

    # adjust for size
    c_size = max(0.1, min(1.0, size / 100.0))
    durations_sec = [s * c_size for s in durations_sec]

    # adjust for speed
    c_duration = 2 ** (-20 * speed)
    durations_sec = sorted({max(1e-3, c_duration * s) for s in durations_sec})
    durations = [TargetTimeDuration(t_target_sec=s) for s in durations_sec]
    n_runs = max(1, round(n_runs * (1.0 - speed**2)))

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
