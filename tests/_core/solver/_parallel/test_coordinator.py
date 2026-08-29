import multiprocessing

import numpy as np

from max_div._core.metrics import DistanceMetric, DiversityMetric
from max_div._core.metrics._distance import DistanceStore, compute_pdist
from max_div._core.problem import MaxDivProblem
from max_div._core.solver._builders import MaxDivSolverBuilder
from max_div._core.solver._duration import iterations
from max_div._core.solver._parallel import (
    CooperativeCoordinator,
    GroupIncumbentSlot,
    IndependentCoordinator,
    WorkerCoordinator,
)
from max_div._core.solver._presets import SolverPreset
from max_div._core.solver._progress_reporting import Verbosity
from max_div._core.solver._solver_state import SolverState


class _RecordingCoordinator(WorkerCoordinator):
    """A coordinator that counts the batch boundaries it is reached at."""

    def __init__(self) -> None:
        self.calls = 0
        self.sizes: list[int] = []

    def at_batch_boundary(self, state: SolverState, progress_fraction: float) -> None:
        """Record the call, and the selection size the worker held at that moment."""
        self.calls += 1
        self.sizes.append(int(state.n_selected))


def _solve_with(coordinator: WorkerCoordinator | None):
    """Solve a small problem, passing the given coordinator down to the solver steps."""
    rng = np.random.default_rng(4)
    problem = MaxDivProblem.new(rng.random((50, 3)).astype(np.float32), k=5)
    builder = MaxDivSolverBuilder(problem).with_preset(iterations(60), SolverPreset.SMART).with_seed(7)
    return builder.build().solve(verbosity=Verbosity.SILENT, coordinator=coordinator)


def test_the_batch_boundary_is_reached_during_optimization():
    """The coordinator is called during optimization, on a path that runs rather than merely exists."""
    # --- arrange ----------------------
    coordinator = _RecordingCoordinator()

    # --- act --------------------------
    _solve_with(coordinator)

    # --- assert -----------------------
    assert coordinator.calls > 0
    assert all(size == 5 for size in coordinator.sizes)  # optimization swaps, never resizes


def test_a_coordinator_does_not_change_the_search():
    """Passing a coordinator that does nothing leaves the solution exactly as solving alone gives it."""
    # --- arrange / act ----------------
    with_coordinator = _solve_with(IndependentCoordinator())
    without = _solve_with(None)

    # --- assert -----------------------
    np.testing.assert_array_equal(np.sort(with_coordinator.i_selected), np.sort(without.i_selected))


def _cooperation_state(indices: list[int]) -> SolverState:
    """Build a solver state over a fixed line of points, holding the given selection."""
    vectors = np.array([[0.0], [1.0], [2.0], [10.0], [20.0], [30.0]], dtype=np.float32)
    state = SolverState.new(
        n=vectors.shape[0],
        store=DistanceStore.condensed(compute_pdist(vectors, DistanceMetric.L1_MANHATTAN), n=vectors.shape[0]),
        k=3,
        diversity_metric=DiversityMetric.MIN_SEPARATION,
        diversity_tie_breakers=[],
        constraints=[],
    )
    state.add_many(np.array(indices, dtype=np.int32))
    return state


def test_cooperative_coordinator_moves_the_best_selection_between_states():
    """The better state's boundary publishes its selection; the worse state's boundary adopts it."""
    # --- arrange ----------------------
    slot = GroupIncumbentSlot(multiprocessing.get_context("spawn"), k=3, score_length=3)
    coordinator = CooperativeCoordinator(slot)
    spread_out = _cooperation_state([0, 3, 5])  # min separation 10
    clustered = _cooperation_state([0, 1, 2])  # min separation 1

    # --- act --------------------------
    coordinator.at_batch_boundary(spread_out, progress_fraction=0.0)
    coordinator.at_batch_boundary(clustered, progress_fraction=0.0)

    # --- assert -----------------------
    assert clustered.selected_index_array.tolist() == [0, 3, 5]  # adopted the published incumbent
    assert spread_out.selected_index_array.tolist() == [0, 3, 5]  # publisher keeps its own


def test_cooperative_coordinator_keeps_the_better_state_untouched():
    """A worker that beats the slot publishes without adopting; one that matches it leaves the slot untouched."""
    # --- arrange ----------------------
    slot = GroupIncumbentSlot(multiprocessing.get_context("spawn"), k=3, score_length=3)
    coordinator = CooperativeCoordinator(slot)
    clustered = _cooperation_state([0, 1, 2])
    spread_out = _cooperation_state([0, 3, 5])

    # --- act --------------------------
    coordinator.at_batch_boundary(clustered, progress_fraction=0.0)  # publishes (empty slot)
    coordinator.at_batch_boundary(spread_out, progress_fraction=0.0)  # better: replaces the stored incumbent
    coordinator.at_batch_boundary(spread_out, progress_fraction=0.0)  # equal to the slot now: nothing happens

    # --- assert -----------------------
    assert spread_out.selected_index_array.tolist() == [0, 3, 5]
    late_arrival = _cooperation_state([0, 1, 2])  # a worse state adopting proves the replacement stored
    coordinator.at_batch_boundary(late_arrival, progress_fraction=0.0)
    assert late_arrival.selected_index_array.tolist() == [0, 3, 5]


def test_a_lone_cooperative_worker_searches_exactly_as_if_alone():
    """With nobody else publishing, a cooperative worker only ever publishes — the search is unchanged."""
    # --- arrange ----------------------
    builder = MaxDivSolverBuilder(MaxDivProblem.new(np.random.default_rng(4).random((50, 3)).astype(np.float32), k=5))
    n_score_components = 3 + len(builder._determine_diversity_tie_breakers())
    slot = GroupIncumbentSlot(multiprocessing.get_context("spawn"), k=5, score_length=n_score_components)

    # --- act --------------------------
    cooperative = _solve_with(CooperativeCoordinator(slot))
    alone = _solve_with(None)

    # --- assert -----------------------
    np.testing.assert_array_equal(np.sort(cooperative.i_selected), np.sort(alone.i_selected))
    assert slot.written  # the worker did publish along the way
