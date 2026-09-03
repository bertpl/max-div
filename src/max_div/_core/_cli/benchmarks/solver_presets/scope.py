from dataclasses import dataclass

from max_div._core._cli.benchmarks._helpers.solver_sizing import determine_problem_size_for_k
from max_div._core._cli.benchmarks._helpers.speed_scaling import SpeedParam
from max_div._core.solver import SolverPreset, TargetTimeDuration
from max_div._core.solver._parallel._solver import default_worker_count

from ._models import SolverPresetBenchmarkParams


# =================================================================================================
#  Budget series
# =================================================================================================
@dataclass(frozen=True, kw_only=True)
class BudgetSeries:
    """A `BudgetSeries` holds the log-spaced time budgets for one family of benchmark runs."""

    t_min_sec: SpeedParam[float]
    t_max_sec: SpeedParam[float]
    n_points: SpeedParam[int]

    def durations_sec(self, speed: float, max_run_duration_sec: float | None = None) -> list[float]:
        """Return the ascending budget values at the given speed.

        Args:
            speed: 0.0 yields the full series (the docs-page configuration); values toward
                1.0 shrink both the budget range and the point count.
            max_run_duration_sec: When given, replaces the series' longest budget.

        Returns:
            Ascending budgets; a single-point series (or a maximum at or below the minimum)
            collapses to just the longest budget.
        """
        t_max = max_run_duration_sec if max_run_duration_sec is not None else self.t_max_sec.at(speed)
        t_min = min(self.t_min_sec.at(speed), t_max)
        n_points = self.n_points.at(speed)
        if n_points <= 1 or t_min == t_max:
            return [t_max]
        ratio = t_min / t_max
        return sorted({t_max * ratio ** (i / (n_points - 1)) for i in range(n_points)})


# Each (problem, preset)-curve runs the single-worker series, one run with a fresh seed per
# budget point, so repeat variability shows up as scatter between neighboring points.  The
# longest budget is capped for campaign cost: the budgets above it are the bulk of a run's
# hours, and in the superseded series they moved the measured diversity by under 1%.
SINGLE_SERIES = BudgetSeries(
    t_min_sec=SpeedParam(slow=0.03, fast=1e-4),
    t_max_sec=SpeedParam(slow=900.0, fast=1e-3),
    n_points=SpeedParam(slow=50, fast=2),
)
# The parallel runs (SMART re-run on the machine's default worker count) get their own series.
# The maximum matches the single-worker series.  The minimum sits at the parallel solver's
# process-spawn cost (roughly one second), which an end-to-end budget includes — so the lowest
# points deliberately show what a caller gets when the budget barely covers start-up.
PARALLEL_SERIES = BudgetSeries(
    t_min_sec=SpeedParam(slow=1.0, fast=2.0),
    t_max_sec=SpeedParam(slow=900.0, fast=2.0),
    n_points=SpeedParam(slow=35, fast=1),
)

# Every problem is benchmarked at the size where it selects this many items, so the suite's
# curves are comparable across problems: k is what drives the swap space and per-iteration cost.
K_TARGET = 100


# =================================================================================================
#  Scope
# =================================================================================================
def determine_benchmark_scope(
    presets: list[SolverPreset],
    problems: list[str],
    n: int | None,
    speed: float,
    max_run_duration_sec: float | None = None,
) -> list[SolverPresetBenchmarkParams]:
    """Compute full list of benchmark runs to be executed based on presets, problems, problem size n, and speed.

    Args:
        presets: Solver presets to run; each gets the full single-worker budget series on
            every problem.
        problems: Benchmark problem names.
        n: Problem size, or None to size each problem such that it selects `K_TARGET` items.
        speed: See `BudgetSeries.durations_sec`; 0.0 runs the full series.
        max_run_duration_sec: When given, replaces both series' longest budget.

    Returns:
        One `SolverPresetBenchmarkParams` per run: every (problem, preset, budget) combination
        on the single-worker series, with a fresh seed per budget point, plus — when SMART is
        among `presets` — the parallel runs: SMART on the machine's default worker count, on
        `PARALLEL_SERIES`.
    """
    # --- problem sizes --------------------------
    problem_sizes = {
        problem: n if n is not None else determine_problem_size_for_k(problem, K_TARGET) for problem in problems
    }

    # --- single-worker scope --------------------
    single_durations_sec = SINGLE_SERIES.durations_sec(speed, max_run_duration_sec)
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
        for seed, duration_sec in enumerate(single_durations_sec, start=1)
    ]

    # --- parallel runs --------------------------
    # one series, not a preset replica: SMART on the default parallel setup answers "what does the
    # default parallel invocation buy over serial SMART" without overloading the pages
    if SolverPreset.SMART in presets:
        parallel_durations_sec = PARALLEL_SERIES.durations_sec(speed, max_run_duration_sec)
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
            for seed, duration_sec in enumerate(parallel_durations_sec, start=1)
        ]

    return scope
