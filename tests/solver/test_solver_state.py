import numpy as np
import pytest
from numpy import random

from max_div.solver import Constraint, DistanceMetric, DiversityMetric
from max_div.solver._solver_state import SolverState


# =================================================================================================
#  Fixtures
# =================================================================================================
@pytest.fixture(scope="function")
def new_solver_state() -> SolverState:
    return SolverState.new(
        vectors=np.array([[0.0], [1.0], [2.0], [3.0], [4.0], [5.0]], dtype=np.float32),
        k=3,
        distance_metric=DistanceMetric.L1_MANHATTAN,
        diversity_metric=DiversityMetric.geomean_separation(),
        diversity_tie_breakers=[DiversityMetric.non_zero_separation_frac()],
        constraints=[
            Constraint(int_set={0, 1, 2, 3}, min_count=1, max_count=2),
            Constraint(int_set={2, 3, 4, 5}, min_count=1, max_count=2),
        ],
    )


@pytest.fixture(scope="function")
def new_solver_state_unconstrained() -> SolverState:
    return SolverState.new(
        vectors=np.array([[0.0], [1.0], [2.0], [3.0], [4.0], [5.0]], dtype=np.float32),
        k=3,
        distance_metric=DistanceMetric.L1_MANHATTAN,
        diversity_metric=DiversityMetric.geomean_separation(),
        diversity_tie_breakers=[DiversityMetric.non_zero_separation_frac()],
        constraints=[],
    )


# =================================================================================================
#  Tests
# =================================================================================================
def test_solver_state_properties(new_solver_state, new_solver_state_unconstrained):
    # with constraints
    assert new_solver_state.has_constraints == True
    assert new_solver_state.k == 3
    assert new_solver_state.m == 2
    assert new_solver_state.n == 6

    assert new_solver_state.score.constraints < 1.0  # constraints present and not all satisfied --> <1.0

    # without constraints
    assert new_solver_state_unconstrained.has_constraints == False
    assert new_solver_state_unconstrained.k == 3
    assert new_solver_state_unconstrained.m == 0
    assert new_solver_state_unconstrained.n == 6

    assert new_solver_state_unconstrained.score.constraints == 1.0  # no constraints -> perfect score


def test_solver_state_add_remove_validation(new_solver_state):
    # --- arrange -----------------------------------------
    state = new_solver_state

    # --- act & assert ------------------------------------
    with pytest.raises(ValueError):
        state.remove(3)  # never added

    state.add(0)
    with pytest.raises(ValueError):
        state.add(0)  # already selected

    state.remove(0)
    with pytest.raises(ValueError):
        state.remove(0)  # already not selected


def test_solver_state_end_to_end(new_solver_state):
    # --- arrange -----------------------------------------
    state = new_solver_state

    # --- assert 1 ----------------------------------------
    assert state.selected_index_array.size == 0
    assert state.not_selected_index_array.size == 6
    assert state.score.size < 1.0  # insufficient vectors selected
    assert state.score.constraints < 1.0  # constraints not satisfied
    assert np.array_equal(state.con_values, state._con_values)
    assert np.array_equal(state.con_indices, state._con_indices)
    assert np.array_equal(state.global_separation_array, state._sep_global)

    # --- act 1 -------------------------------------------
    state.add(0)
    state.add(2)
    state.add(5)

    # --- assert 2 ----------------------------------------
    assert np.array_equal(state.selected_index_array, [0, 2, 5])
    assert np.array_equal(state.not_selected_index_array, [1, 3, 4])
    assert np.allclose(state.selected_separation_array, [2, 2, 3])
    assert state.score.size == 1.0  # correct number of vectors selected
    assert state.score.constraints == 1.0  # all constraints satisfied
    assert state.score.diversity == pytest.approx((2 * 2 * 3) ** (1 / 3))  # geomean of separations 2, 2, 3

    # --- act 2 -------------------------------------------
    state.remove(5)
    state.add(4)

    # --- assert 3 ----------------------------------------
    assert np.array_equal(state.selected_index_array, [0, 2, 4])
    assert np.array_equal(state.not_selected_index_array, [1, 3, 5])
    assert np.allclose(state.selected_separation_array, [2, 2, 2])
    assert np.allclose(state.not_selected_separation_array, [1, 1, 1])
    assert state.score.size == 1.0  # correct number of vectors selected
    assert state.score.constraints == 1.0  # all constraints satisfied
    assert state.score.diversity == pytest.approx(2.0)  # geomean of separations 2, 2, 2


def test_solver_state_snapshot(new_solver_state):
    # --- arrange -----------------------------------------
    state = new_solver_state
    state.add(0)
    state.add(2)

    # current state so we can compare with state after
    orig_selected_array = state.selected_index_array.copy()
    orig_not_selected_array = state.not_selected_index_array.copy()
    orig_separation_array = state.selected_separation_array.copy()
    orig_con_values = state._con_values.copy()

    # --- act & assert ------------------------------------
    with pytest.raises(ValueError):
        state.restore_snapshot()  # none taken yet

    # the below should be a no-op
    state.set_snapshot()
    state.add(5)
    state.restore_snapshot()

    with pytest.raises(ValueError):
        state.restore_snapshot()  # restoring a snapshot invalidates it

    # --- assert ------------------------------------------
    assert np.array_equal(state.selected_index_array, orig_selected_array)
    assert np.array_equal(state.not_selected_index_array, orig_not_selected_array)
    assert np.allclose(state.selected_separation_array, orig_separation_array)
    assert np.array_equal(state.con_values, orig_con_values)


@pytest.mark.parametrize("seed", list(range(1, 100)))
def test_solver_state_consistency_stress_test(new_solver_state, seed: int):
    """Check solver state consistency after a large series of add/remove operations."""

    # --- arrange -----------------------------------------
    state = new_solver_state
    state_ref = new_solver_state.copy()  # we'll leave this untouched until the end
    n_iters = 100

    # --- act ---------------------------------------------
    random.seed(seed)
    for _ in range(n_iters):
        # take snapshot
        state.set_snapshot()

        # add random number of items
        n_to_add = random.randint(0, len(state.not_selected_index_array) + 1)
        indices_to_select = state.not_selected_index_array.copy()
        random.shuffle(indices_to_select)
        for idx in indices_to_select[:n_to_add]:
            state.add(idx)

        # remove random number of items
        n_to_remove = random.randint(0, len(state.selected_index_array) + 1)
        indices_to_remove = state.selected_index_array.copy()
        random.shuffle(indices_to_remove)
        for idx in indices_to_remove[:n_to_remove]:
            state.remove(idx)

        # restore snapshot with some probability
        if random.rand() < 0.5:
            state.restore_snapshot()

    # --- assert ------------------------------------------

    # double check state_ref was not changed
    assert len(state_ref.selected_index_array) == 0
    assert len(state_ref.not_selected_index_array) == state.n

    # sync state_ref with state
    for idx in state.selected_index_array:
        state_ref.add(idx)

    # check if they're the same
    assert np.array_equal(state.selected_index_array, state_ref.selected_index_array)
    assert np.array_equal(state.not_selected_index_array, state_ref.not_selected_index_array)
    assert np.array_equal(state.global_separation_array, state_ref.global_separation_array)
    assert np.array_equal(state.not_selected_separation_array, state_ref.not_selected_separation_array)
    assert np.array_equal(state.selected_separation_array, state_ref.selected_separation_array)
    assert np.array_equal(state.con_values, state_ref.con_values)
    assert np.array_equal(state.con_indices, state_ref.con_indices)

    assert state.score == state_ref.score
