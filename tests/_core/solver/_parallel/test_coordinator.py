import numpy as np

from max_div._core.problem import MaxDivProblem
from max_div._core.solver._builders import MaxDivSolverBuilder
from max_div._core.solver._duration import Elapsed, iterations
from max_div._core.solver._parallel import WorkerCoordinator
from max_div._core.solver._presets import SolverPreset
from max_div._core.solver._progress_reporting import Verbosity
from max_div._core.solver._solver_state import SolverState


class _RecordingCoordinator(WorkerCoordinator):
    """A coordinator that records the batch boundaries it is reached at."""

    def __init__(self) -> None:
        self.sizes: list[int] = []
        self.fractions: list[float] = []
        self.elapsed: list[Elapsed] = []

    @property
    def worker_index(self) -> int:
        return 0

    @property
    def group_index(self) -> int:
        return 0

    def at_batch_boundary(self, state: SolverState, progress_fraction: float, elapsed: Elapsed) -> None:
        """Record the selection size, progress fraction and elapsed the worker held at this boundary."""
        self.sizes.append(int(state.n_selected))
        self.fractions.append(progress_fraction)
        self.elapsed.append(elapsed)


def _solve_with(coordinator: WorkerCoordinator | None):
    """Solve a small problem, passing the given coordinator down to the solver steps."""
    rng = np.random.default_rng(4)
    problem = MaxDivProblem.new(rng.random((50, 3)).astype(np.float32), k=5)
    builder = MaxDivSolverBuilder(problem).with_preset(iterations(60), SolverPreset.SMART).with_seed(7)
    return builder.build().solve(verbosity=Verbosity.SILENT, coordinator=coordinator)


def test_the_batch_boundary_is_reached_during_optimization():
    """The coordinator is called during optimization, on a path that runs, not one that merely exists."""
    # --- arrange ----------------------
    coordinator = _RecordingCoordinator()

    # --- act --------------------------
    _solve_with(coordinator)

    # --- assert -----------------------
    assert len(coordinator.sizes) > 0
    assert all(size == 5 for size in coordinator.sizes)  # optimization swaps, never resizes


def test_the_boundary_receives_the_workers_own_progress_fraction():
    """Boundaries carry a progress fraction in [0, 1] that never decreases over the step."""
    # --- arrange ----------------------
    coordinator = _RecordingCoordinator()

    # --- act --------------------------
    _solve_with(coordinator)

    # --- assert -----------------------
    assert all(0.0 <= fraction <= 1.0 for fraction in coordinator.fractions)
    assert coordinator.fractions == sorted(coordinator.fractions)


def test_the_boundary_receives_the_workers_elapsed_on_the_solve_wide_axis():
    """Elapsed at a boundary counts from the solve's start: it never decreases and stays within the duration."""
    # --- arrange ----------------------
    coordinator = _RecordingCoordinator()

    # --- act --------------------------
    solution = _solve_with(coordinator)

    # --- assert -----------------------
    iteration_counts = [elapsed.n_iterations for elapsed in coordinator.elapsed]
    assert iteration_counts == sorted(iteration_counts)
    assert iteration_counts[-1] <= solution.duration.n_iterations
    assert all(0.0 <= elapsed.t_elapsed_sec <= solution.duration.t_elapsed_sec for elapsed in coordinator.elapsed)
    # the initialization step precedes every boundary, so its iteration count is already included
    assert iteration_counts[0] >= solution.step_durations[0].n_iterations


def test_a_coordinator_that_never_regroups_reports_no_changes():
    """The base contract gives an observing coordinator an empty change list."""
    # --- arrange / act ----------------
    coordinator = _RecordingCoordinator()

    # --- assert -----------------------
    assert coordinator.worker_group_changes == []


def test_a_coordinator_does_not_change_the_search():
    """Passing a coordinator that only observes leaves the solution exactly as solving alone gives it."""
    # --- arrange / act ----------------
    with_coordinator = _solve_with(_RecordingCoordinator())
    without = _solve_with(None)

    # --- assert -----------------------
    np.testing.assert_array_equal(np.sort(with_coordinator.i_selected), np.sort(without.i_selected))
