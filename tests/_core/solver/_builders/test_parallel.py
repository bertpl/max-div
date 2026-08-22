import numpy as np
import pytest

from max_div._core._warnings import ParallelSolvingWarning
from max_div._core.problem import MaxDivProblem
from max_div._core.solver._builders import MaxDivSolverBuilder, ParallelMaxDivSolverBuilder
from max_div._core.solver._builders._parallel import _resolve_group_sizes
from max_div._core.solver._duration import iterations, seconds
from max_div._core.solver._parallel import (
    ParallelMaxDivSolution,
    WorkerConfig,
    default_group_count,
    default_worker_count,
)
from max_div._core.solver._presets import SolverPreset
from max_div._core.solver._progress_reporting import Verbosity
from max_div._core.solver._solver_step import COOPERATIVE_BATCH_SECONDS, REPORTING_BATCH_SECONDS
from max_div._core.solver._strategies import InitializationStrategy

_BUDGET = iterations(120)


def _problem() -> MaxDivProblem:
    """Return a problem small enough to solve several times over in a test."""
    return MaxDivProblem.new(np.random.default_rng(20260809).random((80, 3)).astype(np.float32), k=8)


def _solve_parallel(workers, seed: int = 5, n_groups: int | None = None) -> ParallelMaxDivSolution:
    """Solve the shared test problem with the given workers."""
    builder = ParallelMaxDivSolverBuilder(_problem()).with_seed(seed)
    return builder.with_workers(_BUDGET, workers, n_groups=n_groups).build().solve()


# =================================================================================================
#  Solving
# =================================================================================================
def test_a_parallel_solve_returns_an_ordinary_solution():
    """The winner is a MaxDivSolution, so code written for a single solve keeps working."""
    # --- arrange / act ----------------
    solution = _solve_parallel(2)

    # --- assert -----------------------
    assert solution.i_selected.size == 8
    assert solution.score.diversity > 0.0
    assert solution.duration.n_iterations > 0


def test_every_worker_is_summarized():
    """Each worker reports what it ran, what it scored, and whether it reached the best score."""
    # --- arrange / act ----------------
    solution = _solve_parallel([WorkerConfig(preset=SolverPreset.SMART), WorkerConfig(preset=SolverPreset.GUIDED)])

    # --- assert -----------------------
    assert [worker.worker_index for worker in solution.workers] == [0, 1]
    assert [worker.config.preset for worker in solution.workers] == [SolverPreset.SMART, SolverPreset.GUIDED]
    assert solution.workers[solution.winning_worker].has_best_score


def test_workers_may_differ_by_initialization_alone():
    """Two workers can run one preset from different starting points, which a preset name cannot express."""
    # --- arrange ----------------------
    workers = [
        WorkerConfig(preset=SolverPreset.SMART),
        WorkerConfig(preset=SolverPreset.SMART, init_strategy=InitializationStrategy.farthest_point()),
    ]

    # --- act --------------------------
    solution = _solve_parallel(workers)

    # --- assert -----------------------
    assert solution.workers[0].config.init_strategy is None
    assert solution.workers[1].config.init_strategy is not None


def test_workers_at_the_best_score_are_counted():
    """The count says how many workers tied for best; when every worker ties, the parallel solve did not help."""
    # --- arrange / act ----------------
    solution = _solve_parallel(3)

    # --- assert -----------------------
    counted = solution.n_workers_with_best_score
    assert counted == sum(1 for worker in solution.workers if worker.has_best_score)
    assert 1 <= counted <= len(solution.workers)


# =================================================================================================
#  Seeds
# =================================================================================================
def test_one_seed_reproduces_an_independent_set_of_workers():
    """An independent set of workers repeated from one seed selects the same items and seeds each worker alike."""
    # --- arrange / act ----------------
    first, second = _solve_parallel(2, n_groups=2), _solve_parallel(2, n_groups=2)

    # --- assert -----------------------
    np.testing.assert_array_equal(first.i_selected, second.i_selected)
    assert [worker.seed for worker in first.workers] == [worker.seed for worker in second.workers]


def test_workers_are_seeded_differently_from_each_other():
    """Derived seeds differ per worker, so the workers search differently."""
    # --- arrange / act ----------------
    seeds = [worker.seed for worker in _solve_parallel(4).workers]

    # --- assert -----------------------
    assert len(set(seeds)) == len(seeds)


def test_a_worker_can_be_replayed_on_its_own():
    """An independent worker's preset and seed reproduce its selection in a single solve.

    Nested singleton groups make every worker independent; a cooperative worker's trajectory
    depends on its group mates, so only independent workers carry this replay contract.
    """
    # --- arrange ----------------------
    solution = _solve_parallel([[WorkerConfig(preset=SolverPreset.SMART)], [WorkerConfig(preset=SolverPreset.GUIDED)]])
    winner = solution.workers[solution.winning_worker]

    # --- act --------------------------
    replayed = (
        MaxDivSolverBuilder(_problem())
        .with_preset(_BUDGET, winner.config.preset)
        .with_seed(winner.seed)
        .build()
        .solve(verbosity=Verbosity.SILENT)
    )

    # --- assert -----------------------
    np.testing.assert_array_equal(np.sort(replayed.i_selected), np.sort(solution.i_selected))


# =================================================================================================
#  Guardrails
# =================================================================================================
def test_a_single_worker_warns():
    """One worker cannot beat a single solve, so configuring one worker warns."""
    # --- arrange / act / assert -------
    with pytest.warns(ParallelSolvingWarning, match="cannot do better than solving once"):
        ParallelMaxDivSolverBuilder(_problem()).with_workers(_BUDGET, 1).build()


def test_more_workers_than_cores_warns(monkeypatch: pytest.MonkeyPatch):
    """Configuring more workers than cores warns, because the workers then share cores."""
    # --- arrange ----------------------
    monkeypatch.setattr("max_div._core.solver._parallel._solver.os.cpu_count", lambda: 2)

    # --- act / assert -----------------
    with pytest.warns(ParallelSolvingWarning, match="share cores"):
        ParallelMaxDivSolverBuilder(_problem()).with_workers(_BUDGET, 4).build()


def test_building_without_workers_is_rejected():
    """A parallel solver with no workers is a configuration error rather than an empty run."""
    # --- arrange / act / assert -------
    with pytest.raises(ValueError, match="needs workers"):
        ParallelMaxDivSolverBuilder(_problem()).build()


# =================================================================================================
#  Default worker count
# =================================================================================================
def test_omitting_the_count_uses_the_default(monkeypatch: pytest.MonkeyPatch):
    """With no worker count given, the parallel solver runs 3/4 of the logical cores (here 8 * 3/4 = 6)."""
    # --- arrange ----------------------
    monkeypatch.setattr("max_div._core.solver._parallel._solver.os.cpu_count", lambda: 8)

    # --- act --------------------------
    solution = ParallelMaxDivSolverBuilder(_problem()).with_seed(5).with_workers(_BUDGET).build().solve()

    # --- assert -----------------------
    assert len(solution.workers) == 6


@pytest.mark.parametrize(
    "logical,expected",
    [(16, 12), (12, 9), (8, 6), (4, 3), (2, 2), (1, 2), (None, 2)],
)
def test_default_worker_count_is_three_quarters_of_the_cores_at_least_two(
    monkeypatch: pytest.MonkeyPatch, logical, expected
):
    """The count is 3/4 of the logical cores, floored at two; an unknown core count also falls back to two."""
    # --- arrange ----------------------
    monkeypatch.setattr("max_div._core.solver._parallel._solver.os.cpu_count", lambda: logical)

    # --- act / assert -----------------
    assert default_worker_count() == expected


# =================================================================================================
#  Groups
# =================================================================================================
@pytest.mark.parametrize(
    "total,expected",
    [(1, 1), (2, 1), (5, 1), (6, 2), (8, 2), (9, 2), (10, 3), (11, 3), (12, 3), (16, 4), (48, 12)],
)
def test_default_group_count(total, expected):
    """The default is the group count nearest total/4: sizes stay 3-5, five workers or fewer stay one group."""
    # --- act / assert -----------------
    assert default_group_count(total) == expected


@pytest.mark.parametrize(
    "total,n_groups,expected",
    [(4, 2, [2, 2]), (5, 2, [3, 2]), (7, 3, [3, 2, 2]), (3, 3, [1, 1, 1]), (3, None, [3])],
)
def test_group_sizes_split_the_remainder_over_the_first_groups(total, n_groups, expected):
    """An uneven split hands the extra workers to the first groups, deterministically."""
    # --- act / assert -----------------
    assert _resolve_group_sizes(total, n_groups) == expected


def test_group_count_outside_the_worker_count_is_rejected():
    """A group count below 1 or above the worker count is a configuration error."""
    # --- arrange ----------------------
    builder = ParallelMaxDivSolverBuilder(_problem())

    # --- act & assert -----------------
    with pytest.raises(ValueError, match="n_groups"):
        builder.with_workers(_BUDGET, 4, n_groups=5)
    with pytest.raises(ValueError, match="n_groups"):
        builder.with_workers(_BUDGET, 4, n_groups=0)


def test_group_count_only_combines_with_an_integer_worker_count():
    """Sequence worker forms carry their own grouping, so combining them with n_groups is rejected."""
    # --- arrange ----------------------
    builder = ParallelMaxDivSolverBuilder(_problem())

    # --- act & assert -----------------
    with pytest.raises(ValueError, match="integer worker count"):
        builder.with_workers(_BUDGET, [WorkerConfig(), WorkerConfig()], n_groups=1)
    with pytest.raises(ValueError, match="integer worker count"):
        builder.with_workers(_BUDGET, [[WorkerConfig()], [WorkerConfig()]], n_groups=2)


def test_mixing_configurations_and_groups_is_rejected():
    """A workers sequence must be all configurations or all groups, never a mix of both."""
    # --- arrange ----------------------
    builder = ParallelMaxDivSolverBuilder(_problem())

    # --- act & assert -----------------
    with pytest.raises(ValueError, match="not a mix"):
        builder.with_workers(_BUDGET, [WorkerConfig(), [WorkerConfig()]])


def test_a_nested_sequence_fixes_grouping_and_configurations():
    """Each inner sequence becomes one group, at its own size, with its own worker configurations."""
    # --- arrange ----------------------
    groups = [
        [WorkerConfig(preset=SolverPreset.SMART), WorkerConfig(preset=SolverPreset.THOROUGH)],
        [WorkerConfig(preset=SolverPreset.SMART)],
    ]

    # --- act --------------------------
    solver = ParallelMaxDivSolverBuilder(_problem()).with_workers(_BUDGET, groups).build()

    # --- assert -----------------------
    assert solver._group_sizes == [2, 1]
    assert [worker.preset for worker in solver._worker_configs] == [
        SolverPreset.SMART,
        SolverPreset.THOROUGH,
        SolverPreset.SMART,
    ]


def test_an_integer_worker_count_defaults_to_cooperative_groups_of_about_four():
    """An integer count defaults to groups of about four workers, so cooperation is the default rather than opt-in."""
    # --- act --------------------------
    solver = ParallelMaxDivSolverBuilder(_problem()).with_workers(_BUDGET, 8).build()

    # --- assert -----------------------
    assert solver._group_sizes == [4, 4]


def test_a_cooperative_group_solves():
    """A group of cooperating workers produces an ordinary, valid solution."""
    # --- arrange / act ----------------
    solution = _solve_parallel(2, n_groups=1)

    # --- assert -----------------------
    assert solution.i_selected.size == 8
    assert len(solution.workers) == 2


def test_cooperative_workers_batch_at_the_cooperative_interval():
    """Workers in groups of two or more carry the tighter batch interval; lone workers keep the default."""
    # --- arrange / act ----------------
    solver = ParallelMaxDivSolverBuilder(_problem()).with_workers(_BUDGET, 3, n_groups=2).build()

    # --- assert -----------------------
    assert [config.batch_seconds for config in solver._solver_configs] == [
        COOPERATIVE_BATCH_SECONDS,
        COOPERATIVE_BATCH_SECONDS,
        REPORTING_BATCH_SECONDS,
    ]


# =================================================================================================
#  End-to-end budget
# =================================================================================================
def test_an_end_to_end_budget_requires_a_time_budget():
    """An iteration count cannot bound the store build and worker setup, so the flag rejects it."""
    # --- arrange / act / assert -------
    with pytest.raises(ValueError, match="requires a time budget"):
        ParallelMaxDivSolverBuilder(_problem()).with_workers(iterations(100), 2, end_to_end_budget=True)


def test_the_budget_start_time_is_stamped_at_solve_start(fake_clock, monkeypatch):
    """Workers receive the parent's solve-start clock, so setup time is charged against the budget."""
    # --- arrange ----------------------
    solver = ParallelMaxDivSolverBuilder(_problem()).with_workers(seconds(10.0), 2, end_to_end_budget=True).build()
    fake_clock.advance(4.0)  # time between build and solve is not part of the solve
    received = {}

    def record_configs(configs, *args, **kwargs):
        received["configs"] = configs
        return []

    monkeypatch.setattr("max_div._core.solver._parallel._solver.run_workers", record_configs)

    # --- act --------------------------
    with pytest.raises(ValueError, match="no results"):  # no worker ran, so there is no winner to return
        solver.solve(verbosity=Verbosity.SILENT)

    # --- assert -----------------------
    assert [config.e2e_budget_sec for config in received["configs"]] == [10.0, 10.0]
    assert [config.t_e2e_budget_start for config in received["configs"]] == [fake_clock.monotonic()] * 2


def test_a_budget_spent_during_setup_reaches_the_workers_as_spent():
    """Spawning the workers alone outlasts a tiny budget, so every worker skips its optimization."""
    # --- arrange / act ----------------
    solver = ParallelMaxDivSolverBuilder(_problem()).with_workers(seconds(0.01), 2, end_to_end_budget=True).build()
    solution = solver.solve(verbosity=Verbosity.SILENT)

    # --- assert -----------------------
    optimization_steps = [name for name in solution.step_durations if "Optim" in name]
    assert len(optimization_steps) == 1
    assert solution.step_durations[optimization_steps[0]].n_iterations == 0
    assert len(solution.i_selected) == 8  # the shared test problem's k
