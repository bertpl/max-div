import numpy as np

from max_div._core._cli.bm_solver_sizing import determine_problem_size_for_k
from max_div._core.solver import SolverPreset, TargetTimeDuration
from max_div._core.solver._parallel._solver import default_worker_count

from ._models import SolverPresetBenchmarkParams
from ._utils import estimate_execution_time_sec_multi

# The full-scope budget ladder (speed=0.0, the configuration the docs pages are generated with)
# runs LADDER_N_POINTS log-spaced budget points per (problem, preset)-curve, each with a fresh
# seed, so repeat variability shows up as scatter between neighboring ladder points.  The
# parallel arm (SMART re-run on the machine's default worker count) starts at
# PARALLEL_ARM_T_MIN_SEC — below that a parallel run mostly measures process-spawn overhead.
LADDER_T_MAX_SEC = 600.0
LADDER_T_MIN_SEC = 0.03
LADDER_N_POINTS = 50
PARALLEL_ARM_T_MIN_SEC = 1.0

# Every problem is benchmarked at the size where it selects this many items, so the suite's
# curves are comparable across problems: k is what drives the swap space and per-iteration cost.
K_TARGET = 100


def determine_benchmark_scope_for_max_duration(
    presets: list[SolverPreset],
    problems: list[str],
    n: int | None,
    max_duration_sec: float,
    max_run_duration_sec: float | None = None,
) -> tuple[float, list[SolverPresetBenchmarkParams]]:
    """Compute the full benchmark-run list from presets, problems, problem size n, and target duration.

    This method auto-tunes speed to fall just within the target duration.
    Returns (speed, scope)-tuple.
    """

    def _get_scope_for_speed(_speed: float) -> list[SolverPresetBenchmarkParams]:
        return determine_benchmark_scope(presets, problems, n, _speed, max_run_duration_sec)

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
    n: int | None,
    speed: float,
    max_run_duration_sec: float | None = None,
) -> list[SolverPresetBenchmarkParams]:
    """Compute full list of benchmark runs to be executed based on presets, problems, problem size n, and speed.

    Args:
        presets: Solver presets to run; each gets the full budget ladder on every problem.
        problems: Benchmark problem names.
        n: Problem size, or None to size each problem such that it selects `K_TARGET` items.
        speed: 0.0 runs the full ladder (the docs-page configuration); values toward 1.0
            shrink both the budget range and the point count for quick runs.
        max_run_duration_sec: When given, overrides the ladder's longest budget.

    Returns:
        One `SolverPresetBenchmarkParams` per run: every (problem, preset, budget) combination
        with a fresh seed per budget point, plus — when SMART is among `presets` — the parallel
        arm: SMART on the machine's default worker count, on the ladder points above
        `PARALLEL_ARM_T_MIN_SEC`.
    """
    # --- speed-dependent settings ---------------
    interp_speed = [0.0, 0.5, 0.99, 1.0]
    interp_max_duration_sec = [LADDER_T_MAX_SEC, 60.0, 2.0, 1e-3]
    interp_min_duration_sec = [LADDER_T_MIN_SEC, LADDER_T_MIN_SEC, LADDER_T_MIN_SEC, 1e-4]
    interp_n_points = [LADDER_N_POINTS, 25, 10, 2]

    if max_run_duration_sec is not None:
        max_duration_sec = max_run_duration_sec
    else:
        max_duration_sec = float(np.interp(speed, interp_speed, interp_max_duration_sec))
    min_duration_sec = min(float(np.interp(speed, interp_speed, interp_min_duration_sec)), max_duration_sec)
    n_points = round(float(np.interp(speed, interp_speed, interp_n_points)))

    # --- budget ladder --------------------------
    if n_points <= 1 or min_duration_sec == max_duration_sec:
        durations_sec = [max_duration_sec]
    else:
        ratio = min_duration_sec / max_duration_sec
        durations_sec = sorted({max_duration_sec * ratio ** (i / (n_points - 1)) for i in range(n_points)})

    # --- problem sizes --------------------------
    problem_sizes = {
        problem: n if n is not None else determine_problem_size_for_k(problem, K_TARGET) for problem in problems
    }

    # --- single-worker scope --------------------
    scope = [
        SolverPresetBenchmarkParams(
            preset=preset,
            problem_name=problem,
            problem_size=problem_sizes[problem],
            duration=TargetTimeDuration(t_target_sec=duration_sec),
            seed=seed,
        )
        for problem in problems
        for preset in presets
        for seed, duration_sec in enumerate(durations_sec, start=1)
    ]

    # --- parallel arm ---------------------------
    # one arm, not a preset replica: SMART on the default parallel setup answers "what does the
    # default parallel invocation buy over serial SMART" without overloading the pages
    if SolverPreset.SMART in presets:
        n_workers = default_worker_count()
        scope += [
            SolverPresetBenchmarkParams(
                preset=SolverPreset.SMART,
                problem_name=problem,
                problem_size=problem_sizes[problem],
                duration=TargetTimeDuration(t_target_sec=duration_sec),
                seed=seed,
                n_workers=n_workers,
            )
            for problem in problems
            for seed, duration_sec in enumerate(durations_sec, start=1)
            if duration_sec >= PARALLEL_ARM_T_MIN_SEC
        ]

    return scope
