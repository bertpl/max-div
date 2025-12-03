import numpy as np

from max_div.solver import Constraint, DistanceMetric, DiversityMetric
from max_div.solver._solver_state import SolverState


def test_solver_state_end_to_end():
    # --- arrange -----------------------------------------
    state = SolverState.new(
        vectors=np.array([[0.0], [1.0], [2.0], [3.0], [4.0], [5.0]], dtype=np.float32),
        target_selection_size=3,
        distance_metric=DistanceMetric.L1_MANHATTAN,
        diversity_metric=DiversityMetric.geomean_separation(),
        constraints=[
            Constraint(int_set={0, 1, 2, 3}, min_count=1, max_count=2),
            Constraint(int_set={2, 3, 4, 5}, min_count=1, max_count=2),
        ],
    )

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
