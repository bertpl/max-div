import multiprocessing
import threading
import time

import numpy as np
import pytest

from max_div._core.metrics import DistanceMetric, DiversityMetric
from max_div._core.metrics._distance import DistanceStore, compute_pdist
from max_div._core.solver._duration import E2eBudget
from max_div._core.solver._parallel import AdaptiveGroupOrchestrator, adaptive_group_count
from max_div._core.solver._solver_state import SolverState


def _orchestrator(n_workers: int) -> AdaptiveGroupOrchestrator:
    """Return an orchestrator over the given worker count, with three-component score slots."""
    return AdaptiveGroupOrchestrator(multiprocessing.get_context("spawn"), n_workers=n_workers, k=3, score_length=3)


def _publish(orchestrator: AdaptiveGroupOrchestrator, group: int, diversity: float) -> None:
    """Publish a selection with the given diversity into one group's slot."""
    orchestrator._slots[group].exchange((3.0, 1.0, diversity), np.array([0, 1, 2], dtype=np.int32))


def _state_with(indices: list[int]) -> SolverState:
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


# =================================================================================================
#  Schedule
# =================================================================================================
@pytest.mark.parametrize(
    "n_workers,fraction,expected",
    [
        (12, 0.0, 12),
        (12, 0.05, 12),
        (12, 0.1, 11),
        (12, 0.5, 6),
        (12, 0.92, 1),  # just past the last transition at 11/12
        (12, 1.0, 1),
        (12, -0.5, 12),  # a not-yet-started clock clamps to the start
        (12, 1.5, 1),  # an overspent budget clamps to the end
        (4, 0.5, 2),
        (1, 0.3, 1),
    ],
)
def test_the_group_count_decreases_linearly_over_the_progress_fraction(n_workers, fraction, expected):
    """Each group count holds for an equal share of the budget, from n_workers down to one."""
    # --- act / assert -----------------
    assert adaptive_group_count(n_workers, fraction) == expected


# =================================================================================================
#  Orchestrator: dissolution and reassignment
# =================================================================================================
def test_the_worst_slot_group_is_dissolved_and_its_worker_joins_the_best():
    """The lowest-scoring group dissolves; with all groups the same size, its worker joins the top scorer."""
    # --- arrange ----------------------
    orchestrator = _orchestrator(3)
    _publish(orchestrator, 0, diversity=0.9)
    _publish(orchestrator, 1, diversity=0.5)
    _publish(orchestrator, 2, diversity=0.1)

    # --- act --------------------------
    orchestrator._dissolve_worst(t_sec=1.0, fraction=0.4)

    # --- assert -----------------------
    assert orchestrator._members == {0: [0, 2], 1: [1]}
    assert orchestrator._assignment[2] == 0
    (event,) = orchestrator.events
    assert event.dissolved_group == 2
    assert event.reassignments == {2: 0}
    assert event.slot_scores[2] == (3.0, 1.0, 0.1)


def test_reassignment_prefers_groups_short_of_the_largest():
    """A freed worker fills a group one short of the largest, even when a full group scores higher."""
    # --- arrange ----------------------
    orchestrator = _orchestrator(4)
    for group, diversity in enumerate([0.9, 0.7, 0.5, 0.3]):
        _publish(orchestrator, group, diversity)
    orchestrator._dissolve_worst(t_sec=1.0, fraction=0.3)  # worker 3 joins group 0, which now has two members

    # --- act --------------------------
    orchestrator._dissolve_worst(t_sec=2.0, fraction=0.6)

    # --- assert -----------------------
    # group 2 dissolves; its worker goes to group 1 (one short of group 0's size), not to the larger group 0
    assert orchestrator._members == {0: [0, 3], 1: [1, 2]}
    assert orchestrator._assignment[2] == 1


def test_unwritten_slots_rank_below_written_ones():
    """A group that never published dissolves first; among several such groups the lowest index goes."""
    # --- arrange ----------------------
    orchestrator = _orchestrator(3)
    _publish(orchestrator, 1, diversity=0.5)

    # --- act --------------------------
    orchestrator._dissolve_worst(t_sec=0.1, fraction=0.4)

    # --- assert -----------------------
    assert orchestrator.events[0].dissolved_group == 0
    assert orchestrator._assignment[0] == 1  # joins the only written slot, the best of the pool


# =================================================================================================
#  Orchestrator: run loop
# =================================================================================================
def test_run_collapses_fully_on_a_spent_budget_and_returns():
    """With the budget already spent, the schedule target is one group, so `run` dissolves down and exits."""
    # --- arrange ----------------------
    orchestrator = _orchestrator(3)
    spent = E2eBudget(budget_sec=1.0, t_start=time.monotonic() - 2.0)

    # --- act --------------------------
    orchestrator.run(spent, stop=threading.Event())

    # --- assert -----------------------
    assert len(orchestrator._members) == 1
    assert len(orchestrator.events) == 2
    assert len(set(orchestrator._assignment)) == 1


def test_run_returns_when_stopped_before_it_starts():
    """A pre-set stop event exits the loop before any dissolution fires."""
    # --- arrange ----------------------
    orchestrator = _orchestrator(3)
    stop = threading.Event()
    stop.set()

    # --- act --------------------------
    orchestrator.run(E2eBudget(budget_sec=100.0).started(), stop=stop)

    # --- assert -----------------------
    assert orchestrator.events == []


def test_run_waits_between_ticks_until_stopped():
    """Mid-schedule, `run` keeps ticking without dissolving further, and exits once stopped."""
    # --- arrange ----------------------
    orchestrator = _orchestrator(3)
    mid_schedule = E2eBudget(budget_sec=1000.0, t_start=time.monotonic() - 400.0)  # fraction 0.4: target two groups
    stop = threading.Event()
    thread = threading.Thread(target=orchestrator.run, args=(mid_schedule, stop))

    # --- act --------------------------
    thread.start()
    time.sleep(0.15)
    stop.set()
    thread.join(timeout=5.0)

    # --- assert -----------------------
    assert not thread.is_alive()
    assert len(orchestrator.events) == 1


# =================================================================================================
#  Coordinator
# =================================================================================================
def test_the_coordinator_exchanges_with_whichever_slot_the_assignment_names():
    """After a reassignment, a worker's next boundary reaches the new slot and adopts its better incumbent."""
    # --- arrange ----------------------
    orchestrator = _orchestrator(2)
    spread_out = _state_with([0, 3, 5])  # min separation 10
    clustered = _state_with([0, 1, 2])  # min separation 1
    orchestrator.coordinator_for(1).at_batch_boundary(spread_out)  # slot 1 now holds the good selection
    coordinator = orchestrator.coordinator_for(0)
    coordinator.at_batch_boundary(clustered)  # publishes into slot 0: nothing better there to adopt
    assert clustered.selected_index_array.tolist() == [0, 1, 2]

    # --- act --------------------------
    orchestrator._assignment[0] = 1
    coordinator.at_batch_boundary(clustered)

    # --- assert -----------------------
    assert clustered.selected_index_array.tolist() == [0, 3, 5]  # adopted slot 1's incumbent
