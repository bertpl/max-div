import numpy as np
import pytest

from max_div._core.solver._solver_step import InitializationStep
from max_div._core.solver._strategies import InitializationStrategy

from ._helpers import new_solver_state


@pytest.mark.parametrize("problem_has_constraints", [True, False])
def test_init_farthest_point(problem_has_constraints: bool):
    """Initialization reaches the full selection size, with and without constraints."""
    # --- arrange -----------------------------------------
    solver_state = new_solver_state(problem_has_constraints)
    strategy = InitializationStrategy.farthest_point()
    init_step = InitializationStep(strategy)

    # --- act ---------------------------------------------
    init_step.run(solver_state)
    score = solver_state.score

    # --- assert ------------------------------------------
    assert score.size == 1.0, "Selection size should be equal to k after initialization"


def test_init_farthest_point_first_pick_is_seeded():
    """The start item comes from the strategy's seeded RNG, so different seeds can differ."""
    # --- arrange / act -----------------------------------
    first_items = []
    for seed in (0, 1, 2, 3):
        solver_state = new_solver_state(has_constraints=False)
        strategy = InitializationStrategy.farthest_point()
        strategy.set_seed(seed)
        first_items.append(int(strategy.get_next_samples(solver_state, solver_state.k)[0]))

    # --- assert ------------------------------------------
    assert len(set(first_items)) > 1, "Different seeds should be able to produce different start items"


def test_init_farthest_point_picks_are_greedy():
    """Every pick after the first is the argmax of contribution wrt the current selection."""
    # --- arrange -----------------------------------------
    solver_state = new_solver_state(has_constraints=False)
    strategy = InitializationStrategy.farthest_point()
    solver_state.add(np.int32(0))

    # --- act ---------------------------------------------
    picked = int(strategy.get_next_samples(solver_state, solver_state.k)[0])

    # --- assert ------------------------------------------
    contributions = solver_state.not_selected_contribution_array
    expected = int(solver_state.not_selected_index_array[np.argmax(contributions)])
    assert picked == expected, "Pick should be the highest-contribution not-selected item"
    assert picked != 0, "The already-selected item should never be picked again"


def test_init_farthest_point_beats_random_init():
    """The greedy construction should start from a better diversity score than a random draw."""
    # --- arrange -----------------------------------------
    state_fps = new_solver_state(has_constraints=False)
    state_random = new_solver_state(has_constraints=False)

    # --- act ---------------------------------------------
    InitializationStep(InitializationStrategy.farthest_point()).run(state_fps)
    InitializationStep(InitializationStrategy.random_one_shot(uniform=True)).run(state_random)

    # --- assert ------------------------------------------
    assert state_fps.score.diversity > state_random.score.diversity


def test_init_farthest_point_name():
    """Test that the strategy name is generated as expected."""
    # --- arrange / act -----------------------------------
    strategy = InitializationStrategy.farthest_point()

    # --- assert ------------------------------------------
    assert strategy.name == "InitFarthestPoint"
