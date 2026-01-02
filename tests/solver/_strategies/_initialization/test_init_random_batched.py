from typing import Any

import numpy as np
import pytest

from max_div.solver._solver_step import InitializationStep
from max_div.solver._strategies import InitializationStrategy
from max_div.solver._strategies._initialization._init_random_batched import InitRandomBatched

from ._helpers import new_solver_state


# =================================================================================================
#  TESTS - main class
# =================================================================================================
def test_init_random_batched_parameter_validation():
    with pytest.raises(ValueError):
        _ = InitRandomBatched(b=1)  # b must be > 1


@pytest.mark.parametrize("problem_has_constraints", [True, False])
@pytest.mark.parametrize("arg_ignore_constraints", [True, False])
@pytest.mark.parametrize("arg_b", [2, 10, 100])
def test_init_random_batched(problem_has_constraints: bool, arg_ignore_constraints: bool, arg_b: int):
    # --- arrange -----------------------------------------
    solver_state = new_solver_state(problem_has_constraints)
    strategy = InitializationStrategy.random_batched(
        b=np.int32(arg_b),
        ignore_constraints=arg_ignore_constraints,
    )
    init_step = InitializationStep(strategy)

    # --- act ---------------------------------------------
    init_step.run(solver_state)
    score = solver_state.score

    # --- assert ------------------------------------------
    assert score.size == 1.0, "Selection size should be equal to k after initialization"
    if not arg_ignore_constraints:
        assert score.constraints == 1.0, "All constraints should be satisfied, if ignore_constraints=True"
    if problem_has_constraints and arg_ignore_constraints:
        assert score.constraints < 1.0, "Not all constraints are expected to be satisfied"


@pytest.mark.parametrize(
    "kwargs, expected_name",
    [
        (dict(b=2, ignore_constraints=False), "InitRandomBatched(b=2)"),
        (dict(b=3, ignore_constraints=False), "InitRandomBatched(b=3)"),
        (dict(b=4, ignore_constraints=True), "InitRandomBatched(b=4,uncon)"),
        (dict(b=5, ignore_constraints=True), "InitRandomBatched(b=5,uncon)"),
    ],
)
def test_init_random_batched_name(kwargs: dict[str, Any], expected_name: str):
    """Test that the strategy name is generated as expected."""

    # --- arrange -----------------------------------------
    optim_strategy = InitializationStrategy.random_batched(**kwargs)

    # --- act & assert ------------------------------------
    assert optim_strategy.name == expected_name
