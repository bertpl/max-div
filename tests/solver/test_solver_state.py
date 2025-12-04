import numpy as np
import pytest

from max_div.solver import Constraint, DistanceMetric, DiversityMetric
from max_div.solver._solver_state import SolverState


# =================================================================================================
#  Fixtures
# =================================================================================================
@pytest.fixture(scope="function")
def new_solver_state() -> SolverState:
    return SolverState.new(
        vectors=np.array([[0.0], [1.0], [2.0], [3.0], [4.0], [5.0]], dtype=np.float32),
        target_selection_size=3,
        distance_metric=DistanceMetric.L1_MANHATTAN,
        diversity_metric=DiversityMetric.geomean_separation(),
        constraints=[
            Constraint(int_set={0, 1, 2, 3}, min_count=1, max_count=2),
            Constraint(int_set={2, 3, 4, 5}, min_count=1, max_count=2),
        ],
    )


# =================================================================================================
#  Tests
# =================================================================================================
def test_solver_state_end_to_end(new_solver_state):
    # --- arrange -----------------------------------------
    state = new_solver_state

    # --- assert 1 ----------------------------------------
    assert state.selected_index_array.size == 0
    assert state.not_selected_index_array.size == 6
    assert state.score == (-3, -2, 0.0)

    # --- act 1 -------------------------------------------
    state.add(0)
    state.add(2)
    state.add(5)

    # --- assert 2 ----------------------------------------
    assert np.array_equal(state.selected_index_array, [0, 2, 5])
    assert np.array_equal(state.not_selected_index_array, [1, 3, 4])
    assert np.allclose(state.selected_separation_array, [2, 2, 3])
    assert np.allclose(state.score, (0, 0, (2 * 2 * 3) ** (1 / 3)))

    # --- act 2 -------------------------------------------
    state.remove(5)
    state.add(4)

    # --- assert 3 ----------------------------------------
    assert np.array_equal(state.selected_index_array, [0, 2, 4])
    assert np.array_equal(state.not_selected_index_array, [1, 3, 5])
    assert np.allclose(state.selected_separation_array, [2, 2, 2])
    assert np.allclose(state.not_selected_separation_array, [1, 1, 1])
    assert np.allclose(state.score, (0, 0, 2))


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
    assert np.array_equal(state._con_values, orig_con_values)
