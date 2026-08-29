import multiprocessing

import numpy as np
import pytest

from max_div._core.metrics import DistanceMetric, DiversityMetric
from max_div._core.metrics._distance import DistanceStore, compute_pdist
from max_div._core.solver._parallel import AdaptiveGroupState, adaptive_group_count
from max_div._core.solver._solver_state import SolverState


def _group_state(n_workers: int) -> AdaptiveGroupState:
    """Return a shared group state over the given worker count, with three-component score slots."""
    return AdaptiveGroupState(multiprocessing.get_context("spawn"), n_workers=n_workers, k=3, score_length=3)


def _publish(group_state: AdaptiveGroupState, worker: int, diversity: float) -> None:
    """Publish a selection with the given diversity through the given worker's assigned slot."""
    group_state.exchange(worker, (3.0, 1.0, diversity), np.array([0, 1, 2], dtype=np.int32))


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
        (12, -0.5, 12),  # a not-yet-started tracker clamps to the start
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
#  Dissolution and reassignment
# =================================================================================================
def test_the_worst_slot_group_is_dissolved_and_its_worker_joins_the_best():
    """The lowest-scoring group dissolves; with all groups the same size, its worker joins the top scorer."""
    # --- arrange ----------------------
    group_state = _group_state(3)
    _publish(group_state, 0, diversity=0.9)
    _publish(group_state, 1, diversity=0.5)
    _publish(group_state, 2, diversity=0.1)

    # --- act --------------------------
    group_state.maybe_dissolve(progress_fraction=0.4)  # scheduled count 2, so one dissolution is due

    # --- assert -----------------------
    assert list(group_state._assignment) == [0, 1, 0]
    (event,) = group_state.events()
    assert event.dissolved_group == 2
    assert event.reassignments == {2: 0}
    assert event.slot_scores[2] == (3.0, 1.0, 0.1)
    assert event.progress_fraction == 0.4


def test_reassignment_prefers_groups_short_of_the_largest():
    """A freed worker fills a group one short of the largest, even when a full group scores higher."""
    # --- arrange ----------------------
    group_state = _group_state(4)
    for worker, diversity in enumerate([0.9, 0.7, 0.5, 0.3]):
        _publish(group_state, worker, diversity)
    group_state.maybe_dissolve(progress_fraction=0.3)  # worker 3 joins group 0, which now has two members

    # --- act --------------------------
    group_state.maybe_dissolve(progress_fraction=0.6)

    # --- assert -----------------------
    # group 2 dissolves; its worker goes to group 1 (one short of group 0's size), not to the larger group 0
    assert list(group_state._assignment) == [0, 1, 1, 0]


def test_unwritten_slots_rank_below_written_ones():
    """A group that never published dissolves first; among several such groups the lowest index goes."""
    # --- arrange ----------------------
    group_state = _group_state(3)
    _publish(group_state, 1, diversity=0.5)

    # --- act --------------------------
    group_state.maybe_dissolve(progress_fraction=0.4)

    # --- assert -----------------------
    (event,) = group_state.events()
    assert event.dissolved_group == 0
    assert event.slot_scores == {0: None, 1: (3.0, 1.0, 0.5), 2: None}
    assert group_state._assignment[0] == 1  # joins the only written slot, the best of the pool


def test_a_late_fraction_dissolves_several_groups_at_once():
    """A worker crossing several thresholds in one boundary executes every due dissolution."""
    # --- arrange ----------------------
    group_state = _group_state(3)

    # --- act --------------------------
    group_state.maybe_dissolve(progress_fraction=1.0)

    # --- assert -----------------------
    assert group_state._n_active.value == 1
    assert len(group_state.events()) == 2
    assert len(set(group_state._assignment)) == 1


def test_an_on_schedule_count_dissolves_nothing():
    """A fraction whose scheduled count matches the live count leaves the grouping untouched."""
    # --- arrange ----------------------
    group_state = _group_state(3)

    # --- act --------------------------
    group_state.maybe_dissolve(progress_fraction=0.0)

    # --- assert -----------------------
    assert group_state._n_active.value == 3
    assert group_state.events() == []


def test_dead_groups_drop_out_of_later_event_scores():
    """An event's slot scores cover only the then-alive groups, so the log reads as the mechanism ran."""
    # --- arrange ----------------------
    group_state = _group_state(3)
    for worker, diversity in enumerate([0.9, 0.5, 0.1]):
        _publish(group_state, worker, diversity)

    # --- act --------------------------
    group_state.maybe_dissolve(progress_fraction=1.0)

    # --- assert -----------------------
    first, second = group_state.events()
    assert sorted(first.slot_scores) == [0, 1, 2]
    assert sorted(second.slot_scores) == [0, 1]  # group 2 dissolved first, so it no longer appears


# =================================================================================================
#  Coordinator
# =================================================================================================
def test_the_coordinator_exchanges_with_whichever_slot_the_assignment_names():
    """After a reassignment, a worker's next boundary reaches the new slot and adopts its better incumbent."""
    # --- arrange ----------------------
    group_state = _group_state(2)
    spread_out = _state_with([0, 3, 5])  # min separation 10
    clustered = _state_with([0, 1, 2])  # min separation 1
    group_state.coordinator_for(1).at_batch_boundary(spread_out, progress_fraction=0.0)
    coordinator = group_state.coordinator_for(0)
    coordinator.at_batch_boundary(clustered, progress_fraction=0.0)  # publishes into its own slot 0
    assert clustered.selected_index_array.tolist() == [0, 1, 2]

    # --- act --------------------------
    group_state._assignment[0] = 1
    coordinator.at_batch_boundary(clustered, progress_fraction=0.0)

    # --- assert -----------------------
    assert clustered.selected_index_array.tolist() == [0, 3, 5]  # adopted slot 1's incumbent


def test_a_boundary_past_the_threshold_regroups_and_adopts_in_one_visit():
    """A worker whose fraction crossed a threshold dissolves the worst group and lands in the survivor."""
    # --- arrange ----------------------
    group_state = _group_state(2)
    spread_out = _state_with([0, 3, 5])
    clustered = _state_with([0, 1, 2])
    group_state.coordinator_for(1).at_batch_boundary(spread_out, progress_fraction=0.1)
    coordinator = group_state.coordinator_for(0)
    coordinator.at_batch_boundary(clustered, progress_fraction=0.1)

    # --- act --------------------------
    coordinator.at_batch_boundary(clustered, progress_fraction=0.6)  # scheduled count is now one group

    # --- assert -----------------------
    (event,) = group_state.events()
    assert event.dissolved_group == 0  # the caller's own, lower-scoring group
    assert event.reassignments == {0: 1}
    assert clustered.selected_index_array.tolist() == [0, 3, 5]  # exchanged with the survivor's slot
