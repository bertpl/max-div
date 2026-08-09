import numpy as np

from max_div._core.problem import MaxDivProblem
from max_div._core.solver._duration import iterations
from max_div._core.solver._parallel import IndependentCoordinator, WorkerCoordinator
from max_div._core.solver._presets import SolverPreset
from max_div._core.solver._solver_builder import MaxDivSolverBuilder
from max_div._core.solver._solver_state import SolverState


class _RecordingCoordinator(WorkerCoordinator):
    """A coordinator that counts the batch boundaries it is reached at."""

    def __init__(self) -> None:
        self.calls = 0
        self.sizes: list[int] = []

    def at_batch_boundary(self, state: SolverState) -> None:
        """Record the call, and the selection size the worker held at that moment."""
        self.calls += 1
        self.sizes.append(int(state.n_selected))


def _solve_with(coordinator: WorkerCoordinator | None):
    """Solve a small problem, passing the given coordinator down to the solver steps."""
    rng = np.random.default_rng(4)
    problem = MaxDivProblem.new(rng.random((50, 3)).astype(np.float32), k=5)
    builder = MaxDivSolverBuilder(problem).with_preset(iterations(60), SolverPreset.SMART).with_seed(7)
    return builder.build().solve(verbosity=0, coordinator=coordinator)


def test_the_batch_boundary_is_reached_during_optimization():
    """The seam is called while optimizing, so a sharing mode has a hook that actually runs."""
    # --- arrange -----------------------------------------
    coordinator = _RecordingCoordinator()

    # --- act ---------------------------------------------
    _solve_with(coordinator)

    # --- assert ------------------------------------------
    assert coordinator.calls > 0
    assert all(size == 5 for size in coordinator.sizes)  # optimization swaps, never resizes


def test_a_coordinator_does_not_change_the_search():
    """Passing a coordinator that does nothing leaves the solution exactly as solving alone gives it."""
    # --- arrange / act -----------------------------------
    with_coordinator = _solve_with(IndependentCoordinator())
    without = _solve_with(None)

    # --- assert ------------------------------------------
    np.testing.assert_array_equal(np.sort(with_coordinator.i_selected), np.sort(without.i_selected))
