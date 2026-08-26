from max_div._core._cli.bm_solver_sizing import determine_problem_size_for_k
from max_div._core._cli.bm_speed import SpeedParam
from max_div._core.solver import SolverPreset, TargetTimeDuration
from max_div._core.solver._parallel._solver import default_worker_count

from ._models import SolverPresetBenchmarkParams

# The full-scope budget ladder (speed=0.0, the configuration the docs pages are generated with)
# runs LADDER_N_POINTS log-spaced budget points per (problem, preset)-curve, each with a fresh
# seed, so repeat variability shows up as scatter between neighboring ladder points.  The
# parallel arm (SMART re-run on the machine's default worker count) starts at
# PARALLEL_ARM_T_MIN_SEC — below that a parallel run mostly measures process-spawn overhead.
LADDER_T_MAX_SEC = 600.0
LADDER_T_MIN_SEC = 0.03
LADDER_N_POINTS = 50
PARALLEL_ARM_T_MIN_SEC = 1.0

# The speed parameter shrinks the range of budget points from the full scope above to the
# turbo scope.
_MAX_DURATION_SEC = SpeedParam(LADDER_T_MAX_SEC, 1e-3)
_MIN_DURATION_SEC = SpeedParam(LADDER_T_MIN_SEC, 1e-4)
_N_POINTS = SpeedParam(LADDER_N_POINTS, 2)

# Every problem is benchmarked at the size where it selects this many items, so the suite's
# curves are comparable across problems: k is what drives the swap space and per-iteration cost.
K_TARGET = 100


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
    max_duration_sec = max_run_duration_sec if max_run_duration_sec is not None else _MAX_DURATION_SEC.at(speed)
    min_duration_sec = min(_MIN_DURATION_SEC.at(speed), max_duration_sec)
    n_points = _N_POINTS.at_int(speed)

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
