import numpy as np
import pytest

from max_div._core.solver._solver_step import InitializationStep
from max_div._core.solver._step_identity import SolverStepIdentity
from max_div._core.solver._strategies import InitializationStrategy

from ._helpers import new_solver_state

# each step records its checkpoints under this identity when run on its own in these tests
_STEP_IDENTITY = SolverStepIdentity(1, "test")


def test_init_fixed_selection_starts_from_the_given_selection_even_when_infeasible():
    """The selection is exactly the given indices, kept as they are even when they violate the constraints."""
    # --- arrange ----------------------
    solver_state = new_solver_state(has_constraints=True)  # needs 10 items from 0..49 and 40 from 50..99
    indices = np.arange(49, -1, -1)  # all 50 items from 0..49, in reverse order
    init_step = InitializationStep(InitializationStrategy.fixed_selection(indices))

    # --- act --------------------------
    init_step.run(solver_state, _STEP_IDENTITY)

    # --- assert -----------------------
    assert set(solver_state.selected_index_array.tolist()) == set(indices.tolist())
    assert solver_state.score.constraints < 1.0


@pytest.mark.parametrize(
    "indices, message",
    [
        ([[0, 1], [2, 3]], "1-dimensional array of integer indices"),
        ([0.0, 1.0], "1-dimensional array of integer indices"),
        ([], "1-dimensional array of integer indices"),
        ([0, -3, 5], "Index -3 is negative"),
        ([4, 7, 4], "Index 4 appears more than once"),
    ],
)
def test_fixed_selection_rejects_indices_that_fit_no_problem(indices, message: str):
    """The factory rejects, before any problem is known, indices that no problem could accept."""
    # --- act / assert -----------------
    with pytest.raises(ValueError, match=message):
        InitializationStrategy.fixed_selection(indices)


@pytest.mark.parametrize(
    "indices, message",
    [
        (np.arange(49), "exactly k=50 indices; got 49"),
        (np.arange(2, 102, 2), "Index 100 lies outside the population of n=100 items"),
    ],
)
def test_init_fixed_selection_rejects_a_selection_that_does_not_fit_the_problem(indices, message: str):
    """A selection of the wrong size, or reaching past n, raises when the solve starts."""
    # --- arrange ----------------------
    solver_state = new_solver_state(has_constraints=False)  # n=100, k=50
    init_step = InitializationStep(InitializationStrategy.fixed_selection(indices))

    # --- act / assert -----------------
    with pytest.raises(ValueError, match=message):
        init_step.run(solver_state, _STEP_IDENTITY)
