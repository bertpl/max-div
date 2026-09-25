from typing import Any

import numpy as np
import pytest

from max_div._core.solver._solver_step import InitializationStep
from max_div._core.solver._step_identity import SolverStepIdentity
from max_div._core.solver._strategies import InitializationStrategy

from ._helpers import new_solver_state

# each step records its checkpoints under this identity when run on its own in these tests
_STEP_IDENTITY = SolverStepIdentity(1, "test")


@pytest.mark.parametrize("problem_has_constraints", [True, False])
@pytest.mark.parametrize("arg_ignore_constraints", [True, False])
def test_init_random_selection_completes_selection(problem_has_constraints: bool, arg_ignore_constraints: bool):
    """The selection has size k and satisfies the constraints, unless they are ignored on a constrained problem."""
    # --- arrange ----------------------
    solver_state = new_solver_state(problem_has_constraints)
    strategy = InitializationStrategy.random_selection(ignore_constraints=arg_ignore_constraints)
    init_step = InitializationStep(strategy)

    # --- act --------------------------
    init_step.run(solver_state, _STEP_IDENTITY)
    score = solver_state.score

    # --- assert -----------------------
    assert score.size == 1.0, "Selection size should be equal to k after initialization"
    if not arg_ignore_constraints:
        assert score.constraints == 1.0, "All constraints should be satisfied, if ignore_constraints=False"
    if problem_has_constraints and arg_ignore_constraints:
        assert score.constraints < 1.0, "Not all constraints are expected to be satisfied"


@pytest.mark.parametrize(
    "kwargs, expected_name",
    [
        ({"ignore_constraints": False}, "InitRandomSelection()"),
        ({"ignore_constraints": True}, "InitRandomSelection(uncon)"),
    ],
)
def test_init_random_selection_name(kwargs: dict[str, Any], expected_name: str):
    """Test that the strategy name is generated as expected."""

    # --- arrange ----------------------
    strategy = InitializationStrategy.random_selection(**kwargs)

    # --- act & assert -----------------
    assert strategy.name == expected_name


@pytest.mark.parametrize("arg_parallel", [False, True])
def test_init_random_selection_parallel_flag(arg_parallel: bool):
    """The constructor's parallel choice must surface through the batch-add property."""

    # --- arrange / act ----------------
    strategy = InitializationStrategy.random_selection(parallel=arg_parallel)

    # --- assert -----------------------
    assert strategy.parallel_batch_add is arg_parallel


def test_init_random_selection_parallel_matches_serial():
    """A parallel init must produce the exact same selection and separations as a serial one."""

    # --- arrange ----------------------
    states = [new_solver_state(has_constraints=False) for _ in range(2)]
    steps = [
        InitializationStep(InitializationStrategy.random_selection(parallel=parallel)) for parallel in (False, True)
    ]

    # --- act --------------------------
    for step, state in zip(steps, states, strict=True):
        step.run(state, _STEP_IDENTITY)

    # --- assert -----------------------
    serial_state, parallel_state = states
    np.testing.assert_array_equal(serial_state.selected_index_array, parallel_state.selected_index_array)
    assert serial_state.score == parallel_state.score
