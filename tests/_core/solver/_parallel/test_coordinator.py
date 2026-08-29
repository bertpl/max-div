import numpy as np

from max_div._core.problem import MaxDivProblem
from max_div._core.solver._builders import MaxDivSolverBuilder
from max_div._core.solver._duration import iterations
from max_div._core.solver._parallel import WorkerCoordinator
from max_div._core.solver._presets import SolverPreset
from max_div._core.solver._progress_reporting import Verbosity
from max_div._core.solver._solver_state import SolverState


class _RecordingCoordinator(WorkerCoordinator):
    """A coordinator that records the batch boundaries it is reached at."""

    def __init__(self) -> None:
        self.sizes: list[int] = []
        self.fractions: list[float] = []

    def at_batch_boundary(self, state: SolverState, progress_fraction: float) -> None:
        """Record the selection size and progress fraction the worker held at this boundary."""
        self.sizes.append(int(state.n_selected))
        self.fractions.append(progress_fraction)


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


def test_a_coordinator_does_not_change_the_search():
    """Passing a coordinator that only observes leaves the solution exactly as solving alone gives it."""
    # --- arrange / act ----------------
    with_coordinator = _solve_with(_RecordingCoordinator())
    without = _solve_with(None)

    # --- assert -----------------------
    np.testing.assert_array_equal(np.sort(with_coordinator.i_selected), np.sort(without.i_selected))
