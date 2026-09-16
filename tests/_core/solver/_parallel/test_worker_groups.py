import multiprocessing

import numpy as np
import pytest

from max_div._core.metrics import DistanceMetric, DiversityMetric
from max_div._core.metrics._distance import DistanceStore
from max_div._core.solver._duration import Elapsed
from max_div._core.solver._parallel import FixedGroupCount, PowerLawGroupMerge, WorkerGroupState
from max_div._core.solver._solver_state import SolverState
from tests._core.solver.objectives import simple_objective


def _group_state(n_workers: int, group_sizes: list[int] | None = None, dynamic: bool = True) -> WorkerGroupState:
    """Return a shared group state over the given worker count, with three-component score slots.

    A dynamic state follows the linear schedule (rate 1), so the dissolution tests below read
    against evenly spaced merge thresholds.
    """
    sizes = group_sizes if group_sizes is not None else [1] * n_workers
    schedule = PowerLawGroupMerge(n_workers, rate=1.0) if dynamic else FixedGroupCount(len(sizes))
    return WorkerGroupState(
        multiprocessing.get_context("spawn"), group_sizes=sizes, k=3, score_length=3, schedule=schedule
    )


_ELAPSED = Elapsed(t_elapsed_sec=2.5, n_iterations=40)


def _dissolve(group_state: WorkerGroupState, progress_fraction: float, worker: int = 0):
    """Run the schedule as the given worker at the given fraction, and return the changes it made."""
    return group_state.maybe_dissolve(worker, progress_fraction, _ELAPSED)


def _publish(group_state: WorkerGroupState, worker: int, diversity: float) -> None:
    """Publish a selection with the given diversity through the given worker's assigned slot."""
    group_state.exchange(worker, (3.0, 1.0, diversity), np.array([0, 1, 2], dtype=np.int32))


def _state_with(indices: list[int]) -> SolverState:
    """Build a solver state over a fixed line of points, holding the given selection."""
    vectors = np.array([[0.0], [1.0], [2.0], [10.0], [20.0], [30.0]], dtype=np.float32)
    state = SolverState.new(
        n=vectors.shape[0],
        stores_by_distance={None: DistanceStore.full_matrix_from_vectors(vectors, DistanceMetric.l1_manhattan())},
        k=3,
        diversity_objectives=[simple_objective(DiversityMetric.MIN_SEPARATION)],
        constraints=[],
    )
    state.add_many(np.array(indices, dtype=np.int32))
    return state


# =================================================================================================
#  Schedule
# =================================================================================================
def test_the_scheduled_count_is_the_schedule_s_count():
    """The state asks its schedule for the count at the given fraction, without a schedule of its own."""
    # --- arrange ----------------------
    group_state = _group_state(12)

    # --- act / assert -----------------
    assert group_state._scheduled_count(0.5) == PowerLawGroupMerge(12, rate=1.0).group_count(0.5) == 6


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
    (change,) = _dissolve(group_state, 0.4, worker=1)  # scheduled count 2, so one dissolution is due

    # --- assert -----------------------
    assert list(group_state._assignment) == [0, 1, 0]
    assert change.dissolved_group == 2
    assert change.reassignments == {2: 0}
    assert change.slot_scores[2] == (3.0, 1.0, 0.1)
    assert change.n_alive_groups_after == 2
    assert (change.executed_by, change.progress_fraction, change.elapsed) == (1, 0.4, _ELAPSED)


def test_reassignment_prefers_the_smallest_groups():
    """A freed worker fills one of the smallest groups, even when a larger group scores higher."""
    # --- arrange ----------------------
    group_state = _group_state(4)
    for worker, diversity in enumerate([0.9, 0.7, 0.5, 0.3]):
        _publish(group_state, worker, diversity)
    _dissolve(group_state, 0.3)  # worker 3 joins group 0, which now has two members

    # --- act --------------------------
    _dissolve(group_state, 0.6)

    # --- assert -----------------------
    # group 2 dissolves; its worker goes to group 1 (the smallest), not to the larger group 0
    assert list(group_state._assignment) == [0, 1, 1, 0]


def test_unwritten_slots_rank_below_written_ones():
    """A group that never published dissolves first; among several such groups the lowest index goes."""
    # --- arrange ----------------------
    group_state = _group_state(3)
    _publish(group_state, 1, diversity=0.5)

    # --- act --------------------------
    (change,) = _dissolve(group_state, 0.4)

    # --- assert -----------------------
    assert change.dissolved_group == 0
    assert change.slot_scores == {0: None, 1: (3.0, 1.0, 0.5), 2: None}
    assert group_state._assignment[0] == 1  # joins the only written slot, the best of the pool


def test_a_late_fraction_dissolves_several_groups_at_once():
    """A worker crossing several thresholds in one boundary executes every due dissolution."""
    # --- arrange ----------------------
    group_state = _group_state(3)

    # --- act --------------------------
    changes = _dissolve(group_state, 1.0)

    # --- assert -----------------------
    assert group_state._n_alive_groups.value == 1
    assert [change.n_alive_groups_after for change in changes] == [2, 1]
    assert len(set(group_state._assignment)) == 1


def test_an_on_schedule_count_dissolves_nothing():
    """A fraction whose scheduled count matches the alive count leaves the grouping untouched."""
    # --- arrange ----------------------
    group_state = _group_state(3)

    # --- act --------------------------
    changes = _dissolve(group_state, 0.0)

    # --- assert -----------------------
    assert group_state._n_alive_groups.value == 3
    assert changes == []


def test_dead_groups_drop_out_of_later_change_scores():
    """A change's slot scores cover only the then-alive groups: the grouping as it stood when the change fired."""
    # --- arrange ----------------------
    group_state = _group_state(3)
    for worker, diversity in enumerate([0.9, 0.5, 0.1]):
        _publish(group_state, worker, diversity)

    # --- act --------------------------
    first, second = _dissolve(group_state, 1.0)

    # --- assert -----------------------
    assert sorted(first.slot_scores) == [0, 1, 2]
    assert sorted(second.slot_scores) == [0, 1]  # group 2 dissolved first, so it no longer appears


# =================================================================================================
#  Coordinator
# =================================================================================================
def test_the_coordinator_exchanges_with_whichever_slot_the_assignment_names():
    """After a reassignment, a worker's next boundary reaches the new slot and adopts its better stored selection."""
    # --- arrange ----------------------
    group_state = _group_state(2)
    spread_out = _state_with([0, 3, 5])  # min separation 10
    clustered = _state_with([0, 1, 2])  # min separation 1
    group_state.coordinator_for(1).at_batch_boundary(spread_out, 0.0, _ELAPSED)
    coordinator = group_state.coordinator_for(0)
    coordinator.at_batch_boundary(clustered, 0.0, _ELAPSED)  # publishes into its own slot 0
    assert clustered.selected_index_array.tolist() == [0, 1, 2]

    # --- act --------------------------
    group_state._assignment[0] = 1
    coordinator.at_batch_boundary(clustered, 0.0, _ELAPSED)

    # --- assert -----------------------
    assert clustered.selected_index_array.tolist() == [0, 3, 5]  # adopted slot 1's stored selection


def test_the_coordinator_reports_its_worker_and_the_group_it_is_assigned_to():
    """The group index follows the assignment table, so it changes when the worker is reassigned."""
    # --- arrange ----------------------
    group_state = _group_state(2)
    coordinator = group_state.coordinator_for(1)
    assert (coordinator.worker_index, coordinator.group_index) == (1, 1)

    # --- act --------------------------
    group_state._assignment[1] = 0

    # --- assert -----------------------
    assert (coordinator.worker_index, coordinator.group_index) == (1, 0)


def test_a_boundary_past_the_threshold_regroups_and_adopts_in_one_visit():
    """A worker whose fraction crossed a threshold dissolves the worst group and lands in the survivor."""
    # --- arrange ----------------------
    group_state = _group_state(2)
    spread_out = _state_with([0, 3, 5])
    clustered = _state_with([0, 1, 2])
    group_state.coordinator_for(1).at_batch_boundary(spread_out, 0.1, _ELAPSED)
    coordinator = group_state.coordinator_for(0)
    coordinator.at_batch_boundary(clustered, 0.1, _ELAPSED)

    # --- act --------------------------
    coordinator.at_batch_boundary(clustered, 0.6, _ELAPSED)  # scheduled count is now one group

    # --- assert -----------------------
    (change,) = coordinator.worker_group_changes
    assert change.dissolved_group == 0  # the caller's own, lower-scoring group
    assert change.reassignments == {0: 1}
    assert change.executed_by == 0
    assert clustered.selected_index_array.tolist() == [0, 3, 5]  # exchanged with the survivor's slot


def test_a_coordinator_keeps_only_the_changes_it_executed():
    """Each worker returns its own changes; a worker whose boundary dissolved nothing has none."""
    # --- arrange ----------------------
    group_state = _group_state(2)
    executing, idle = group_state.coordinator_for(0), group_state.coordinator_for(1)
    idle.at_batch_boundary(_state_with([0, 3, 5]), 0.1, _ELAPSED)

    # --- act --------------------------
    executing.at_batch_boundary(_state_with([0, 1, 2]), 1.0, _ELAPSED)
    idle.at_batch_boundary(_state_with([0, 3, 5]), 1.0, _ELAPSED)  # the count already matches the schedule

    # --- assert -----------------------
    assert [change.executed_by for change in executing.worker_group_changes] == [0]
    assert idle.worker_group_changes == []


# =================================================================================================
#  Fixed grouping
# =================================================================================================
def test_a_fixed_grouping_starts_from_its_configured_assignment():
    """Group sizes translate into consecutive worker runs, one slot per group."""
    # --- arrange / act ----------------
    group_state = _group_state(5, group_sizes=[3, 2], dynamic=False)

    # --- assert -----------------------
    assert list(group_state._assignment) == [0, 0, 0, 1, 1]
    assert group_state.initial_assignment == [0, 0, 0, 1, 1]
    assert group_state._n_alive_groups.value == 2


def test_a_fixed_grouping_never_dissolves():
    """The fixed schedule's target equals the configured count, so no progress fraction fires a transition."""
    # --- arrange ----------------------
    group_state = _group_state(4, group_sizes=[2, 2], dynamic=False)

    # --- act --------------------------
    changes = _dissolve(group_state, 1.0)

    # --- assert -----------------------
    assert list(group_state._assignment) == [0, 0, 1, 1]
    assert changes == []


def test_a_fixed_group_exchanges_through_its_shared_slot():
    """Members of one fixed group adopt each other's best through their group's slot."""
    # --- arrange ----------------------
    group_state = _group_state(2, group_sizes=[2], dynamic=False)
    spread_out = _state_with([0, 3, 5])  # min separation 10
    clustered = _state_with([0, 1, 2])  # min separation 1
    group_state.coordinator_for(0).at_batch_boundary(spread_out, 1.0, _ELAPSED)

    # --- act --------------------------
    group_state.coordinator_for(1).at_batch_boundary(clustered, 1.0, _ELAPSED)

    # --- assert -----------------------
    assert clustered.selected_index_array.tolist() == [0, 3, 5]  # adopted its group mate's published best


def test_a_zero_sized_group_is_rejected():
    """A zero group size makes the assignment table reference slots that were never allocated."""
    # --- act & assert -----------------
    with pytest.raises(ValueError, match="at least one worker"):
        WorkerGroupState(
            multiprocessing.get_context("spawn"),
            group_sizes=[1, 0, 1],
            k=5,
            score_length=3,
            schedule=FixedGroupCount(3),
        )
