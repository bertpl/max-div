from typing import Any

import numpy as np
import pytest

from max_div._core.solver._solver_step import InitializationStep
from max_div._core.solver._strategies import InitializationStrategy
from max_div._core.solver._strategies._initialization._init_eager import InitEager

from ._helpers import new_solver_state


def test_init_eager_parameter_validation():
    with pytest.raises(ValueError):
        _ = InitEager(nc=1)  # nc must be > 1


@pytest.mark.parametrize("problem_has_constraints", [True, False])
@pytest.mark.parametrize("arg_ignore_constraints", [True, False])
@pytest.mark.parametrize("arg_nc", [2, 10, 100])
def test_init_eager(problem_has_constraints: bool, arg_ignore_constraints: bool, arg_nc: int):
    # --- arrange -----------------------------------------
    solver_state = new_solver_state(problem_has_constraints)
    strategy = InitializationStrategy.eager(
        nc=np.int32(arg_nc),
        ignore_constraints=arg_ignore_constraints,
    )
    init_step = InitializationStep(strategy)

    # --- act ---------------------------------------------
    init_step.run(solver_state)
    score = solver_state.score

    # --- assert ------------------------------------------
    assert score.size == 1.0, "Selection size should be equal to k after initialization"
    if not arg_ignore_constraints:
        assert score.constraints == 1.0, "All constraints should be satisfied, if problem has constraints"
    if problem_has_constraints and arg_ignore_constraints:
        assert score.constraints < 1.0, "Not all constraints are expected to be satisfied"


@pytest.mark.parametrize(
    "kwargs, expected_name",
    [
        ({"nc": 2, "ignore_constraints": False}, "InitEager(nc=2)"),
        ({"nc": 3, "ignore_constraints": False}, "InitEager(nc=3)"),
        ({"nc": 4, "ignore_constraints": True}, "InitEager(nc=4,uncon)"),
        ({"nc": 5, "ignore_constraints": True}, "InitEager(nc=5,uncon)"),
    ],
)
def test_init_eager_name(kwargs: dict[str, Any], expected_name: str):
    """Test that the strategy name is generated as expected."""

    # --- arrange -----------------------------------------
    optim_strategy = InitializationStrategy.eager(**kwargs)

    # --- act & assert ------------------------------------
    assert optim_strategy.name == expected_name
