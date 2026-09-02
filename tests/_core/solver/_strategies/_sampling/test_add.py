import numpy as np
import pytest

from max_div._core._random import new_rng_state
from max_div._core.solver._strategies._sampling import (
    SamplingType,
    build_add_probabilities,
    select_items_to_add,
    select_items_to_add_with_p,
)
from tests._core.solver._strategies._initialization._helpers import new_solver_state


def _state_with_selection(has_constraints: bool):
    """Return a state with a few items selected, so the standard probability path is taken."""
    state = new_solver_state(has_constraints=has_constraints)
    state.add_many(np.array([1, 3, 60, 70], dtype=np.int32))
    return state


@pytest.mark.parametrize("has_constraints", [False, True])
@pytest.mark.parametrize("sampling_type", [SamplingType.GROUP, SamplingType.CANDIDATES])
def test_build_then_draw_equals_select_items_to_add(has_constraints: bool, sampling_type: SamplingType) -> None:
    """Building the probabilities and then drawing reproduces `select_items_to_add` exactly, draw for draw."""
    # --- arrange ----------------------
    state = _state_with_selection(has_constraints)
    candidates = state.not_selected_index_array
    rng_one_call, rng_split = new_rng_state(7), new_rng_state(7)

    # --- act --------------------------
    expected = select_items_to_add(
        state, candidates, k=5, selectivity_modifier=0.3, rng_state=rng_one_call, sampling_type=sampling_type
    )
    p = build_add_probabilities(state, candidates, selectivity_modifier=0.3)
    actual = select_items_to_add_with_p(state, candidates, p, k=5, rng_state=rng_split, sampling_type=sampling_type)

    # --- assert -----------------------
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(rng_split, rng_one_call)


@pytest.mark.parametrize("has_constraints", [False, True])
@pytest.mark.parametrize("sampling_type", [SamplingType.GROUP, SamplingType.CANDIDATES])
def test_drawing_leaves_the_probabilities_intact(has_constraints: bool, sampling_type: SamplingType) -> None:
    """A draw reads `p` and never writes it, so one array can be reused across draws."""
    # --- arrange ----------------------
    state = _state_with_selection(has_constraints)
    candidates = state.not_selected_index_array
    p = build_add_probabilities(state, candidates, selectivity_modifier=0.3)
    p_before = p.copy()

    # --- act --------------------------
    for _ in range(3):
        select_items_to_add_with_p(state, candidates, p, k=5, rng_state=new_rng_state(1), sampling_type=sampling_type)

    # --- assert -----------------------
    np.testing.assert_array_equal(p, p_before)


def test_probabilities_without_a_selection_come_from_the_global_contribution() -> None:
    """With nothing selected yet, the probabilities derive from each candidate's dataset-wide contribution."""
    # --- arrange ----------------------
    state = new_solver_state(has_constraints=False)
    candidates = state.not_selected_index_array

    # --- act --------------------------
    p = build_add_probabilities(state, candidates, selectivity_modifier=0.0)

    # --- assert -----------------------
    assert p.shape == candidates.shape
    assert p.dtype == np.float32
    assert (p > 0).all()
