import numpy as np
import pytest

from max_div._core.solver._solver_step import InitializationStep
from max_div._core.solver._step_identity import SolverStepIdentity
from max_div._core.solver._strategies import InitializationStrategy
from max_div._core.solver._strategies._initialization._init_constraint_aware_diverse import InitConstraintAwareDiverse

from ._helpers import new_solver_state, new_solver_state_unconstrained

# each step records its checkpoints under this identity when run on its own in these tests
_STEP_IDENTITY = SolverStepIdentity(1, "test")


# ==================================================================================================
#  Helpers
# ==================================================================================================
def _final_diversity(batch_size: int, nc: int, seed: int) -> float:
    """Return the diversity of a full initialization on the unconstrained helper state."""
    state = new_solver_state_unconstrained()
    step = InitializationStep(InitializationStrategy.constraint_aware_diverse(batch_size=batch_size, nc=nc))
    step.set_seed(seed)
    step.run(state, _STEP_IDENTITY)
    return float(state.score.diversity)


# ==================================================================================================
#  Selection
# ==================================================================================================
@pytest.mark.parametrize("has_constraints", [True, False])
@pytest.mark.parametrize("batch_size, nc", [(16, 1), (1, 1), (1, 16), (50, 4), (7, 2)])
def test_init_constraint_aware_diverse_completes_selection(has_constraints: bool, batch_size: int, nc: int):
    """The selection has k distinct items, and on the helper's constrained problem it satisfies the constraints."""
    # --- arrange ----------------------
    state = new_solver_state(has_constraints)
    step = InitializationStep(InitializationStrategy.constraint_aware_diverse(batch_size=batch_size, nc=nc))

    # --- act --------------------------
    step.run(state, _STEP_IDENTITY)

    # --- assert -----------------------
    assert state.score.size == 1.0
    assert len(np.unique(state.selected_index_array)) == state.k
    if has_constraints:
        assert state.score.constraints == 1.0


def test_init_constraint_aware_diverse_last_batch_is_capped_by_k_remaining():
    """A batch never holds more items than remain to be selected."""
    # --- arrange ----------------------
    state = new_solver_state(has_constraints=False)
    strategy = InitializationStrategy.constraint_aware_diverse(batch_size=16, nc=2)

    # --- act --------------------------
    batch = strategy.get_next_samples(state, k_remaining=3)

    # --- assert -----------------------
    assert len(batch) == 3
    assert len(np.unique(batch)) == 3


def test_init_constraint_aware_diverse_leaves_the_state_unchanged_while_scoring():
    """Scoring the candidate batches leaves the state unchanged: the returned batch is not yet in the selection."""
    # --- arrange ----------------------
    state = new_solver_state(has_constraints=False)
    state.add(np.int32(0))
    strategy = InitializationStrategy.constraint_aware_diverse(batch_size=4, nc=8)

    # --- act --------------------------
    batch = strategy.get_next_samples(state, k_remaining=state.k - 1)

    # --- assert -----------------------
    assert state.n_selected == 1
    assert not np.isin(batch, state.selected_index_array).any()


def test_init_constraint_aware_diverse_more_candidates_give_more_diversity():
    """Over several seeds, the best of 16 draws per item reaches a higher diversity than 1 draw per item."""
    # --- arrange / act ----------------
    seeds = range(5)
    diversity_nc_1 = np.mean([_final_diversity(batch_size=1, nc=1, seed=seed) for seed in seeds])
    diversity_nc_16 = np.mean([_final_diversity(batch_size=1, nc=16, seed=seed) for seed in seeds])

    # --- assert -----------------------
    assert diversity_nc_16 > diversity_nc_1


def test_init_constraint_aware_diverse_is_deterministic_per_seed():
    """The same seed reproduces the same selection; a different seed varies it."""
    # --- arrange ----------------------
    selections = []
    for seed in (7, 7, 8):
        state = new_solver_state(has_constraints=True)
        step = InitializationStep(InitializationStrategy.constraint_aware_diverse(batch_size=4, nc=4))
        step.set_seed(seed)

        # --- act ----------------------
        step.run(state, _STEP_IDENTITY)
        selections.append(np.sort(state.selected_index_array).copy())

    # --- assert -----------------------
    np.testing.assert_array_equal(selections[0], selections[1])
    assert not np.array_equal(selections[0], selections[2])


# ==================================================================================================
#  Construction
# ==================================================================================================
def test_init_constraint_aware_diverse_name():
    """The name records both parameters."""
    # --- arrange / act ----------------
    strategy = InitializationStrategy.constraint_aware_diverse(batch_size=16, nc=1)

    # --- assert -----------------------
    assert strategy.name == "InitConstraintAwareDiverse(16,1)"


@pytest.mark.parametrize("kwargs", [{"batch_size": 0}, {"batch_size": -1}, {"nc": 0}, {"nc": -3}])
def test_init_constraint_aware_diverse_rejects_invalid_parameters(kwargs: dict):
    """The constructor rejects a batch size or candidate count below 1, through the factory too."""
    # --- act & assert -----------------
    with pytest.raises(ValueError, match=next(iter(kwargs))):
        InitConstraintAwareDiverse(**kwargs)
    with pytest.raises(ValueError, match=next(iter(kwargs))):
        InitializationStrategy.constraint_aware_diverse(**kwargs)


def test_init_constraint_aware_diverse_factory_passes_parameters_through():
    """The public factory's parameters reach the strategy: same seed, same batch as direct construction."""
    # --- arrange ----------------------
    state = new_solver_state(has_constraints=True)
    state.add(np.int32(0))

    # --- act --------------------------
    batches = {}
    for name, strategy in [
        ("factory", InitializationStrategy.constraint_aware_diverse(batch_size=3, nc=5)),
        ("direct", InitConstraintAwareDiverse(batch_size=3, nc=5)),
    ]:
        strategy.set_seed(3)
        batches[name] = strategy.get_next_samples(state, state.k - 1)

    # --- assert -----------------------
    np.testing.assert_array_equal(batches["factory"], batches["direct"])
    assert len(batches["factory"]) == 3
