import numpy as np
import pytest

from max_div._core.solver._solver_step import InitializationStep
from max_div._core.solver._strategies import InitializationStrategy
from max_div._core.solver._strategies._initialization._init_farthest_point import InitFarthestPoint

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


@pytest.mark.parametrize("top_k", [0, -1])
def test_init_farthest_point_rejects_top_k_below_one(top_k: int):
    """top_k must be >= 1."""
    # --- act & assert ------------------------------------
    with pytest.raises(ValueError, match="top_k"):
        InitFarthestPoint(top_k=top_k)


def test_farthest_point_factory_passes_top_k_through():
    """The public factory's top_k reaches the strategy: same seed, same pick as direct construction."""
    # --- arrange -----------------------------------------
    solver_state = new_solver_state(has_constraints=False)
    solver_state.add(np.int32(0))

    # --- act ---------------------------------------------
    picks = {}
    for name, strategy in [
        ("factory", InitializationStrategy.farthest_point(top_k=5)),
        ("direct", InitFarthestPoint(top_k=5)),
    ]:
        strategy.set_seed(3)
        picks[name] = int(strategy.get_next_samples(solver_state, solver_state.k)[0])

    # --- assert ------------------------------------------
    assert picks["factory"] == picks["direct"]


@pytest.mark.parametrize("top_k", [0, -1])
def test_farthest_point_factory_rejects_top_k_below_one(top_k: int):
    """The strategy's top_k validation raises through the public factory."""
    # --- act & assert ------------------------------------
    with pytest.raises(ValueError, match="top_k"):
        InitializationStrategy.farthest_point(top_k=top_k)


def test_init_farthest_point_top_k_1_is_the_argmax_pick():
    """top_k=1 takes the plain argmax pick, identical to the default strategy."""
    # --- arrange -----------------------------------------
    solver_state = new_solver_state(has_constraints=False)
    solver_state.add(np.int32(0))
    contributions = solver_state.not_selected_contribution_array
    expected = int(solver_state.not_selected_index_array[np.argmax(contributions)])

    # --- act ---------------------------------------------
    default_pick = int(InitFarthestPoint().get_next_samples(solver_state, solver_state.k)[0])
    top1_pick = int(InitFarthestPoint(top_k=1).get_next_samples(solver_state, solver_state.k)[0])

    # --- assert ------------------------------------------
    assert top1_pick == default_pick == expected


def test_init_farthest_point_top_k_1_full_init_matches_default():
    """A full init with top_k=1 reproduces the default farthest-point selection bit-for-bit."""
    # --- arrange -----------------------------------------
    state_default = new_solver_state(has_constraints=False)
    state_top1 = new_solver_state(has_constraints=False)

    # --- act ---------------------------------------------
    InitializationStep(InitFarthestPoint()).run(state_default)
    InitializationStep(InitFarthestPoint(top_k=1)).run(state_top1)

    # --- assert ------------------------------------------
    assert list(state_default.selected_index_array) == list(state_top1.selected_index_array)


def test_init_farthest_point_top_k_draws_from_the_top_set():
    """Every top_k>1 greedy pick comes from the top_k highest-contribution items, and the draw varies by seed."""
    # --- arrange -----------------------------------------
    solver_state = new_solver_state(has_constraints=False)
    solver_state.add(np.int32(0))
    contributions = solver_state.not_selected_contribution_array
    index_array = solver_state.not_selected_index_array
    top_k = 5
    threshold = np.sort(contributions)[-top_k]  # k-th largest contribution

    # --- act ---------------------------------------------
    picks = []
    for seed in range(20):
        strategy = InitFarthestPoint(top_k=top_k)
        strategy.set_seed(seed)
        picks.append(int(strategy.get_next_samples(solver_state, solver_state.k)[0]))

    # --- assert ------------------------------------------
    for pick in picks:
        pos = int(np.where(index_array == pick)[0][0])
        assert contributions[pos] >= threshold  # each pick is among the top_k by contribution
    assert len(set(picks)) > 1  # the uniform draw varies across seeds
