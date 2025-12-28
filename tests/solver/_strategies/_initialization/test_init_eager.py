import numpy as np
import pytest

from max_div.solver._strategies import InitializationStrategy
from max_div.solver._strategies._initialization._init_eager import InitEager

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

    # --- act ---------------------------------------------
    strategy.initialize(solver_state)
    score = solver_state.score

    # --- assert ------------------------------------------
    assert score.size == 1.0, "Selection size should be equal to k after initialization"
    if not arg_ignore_constraints:
        assert score.constraints == 1.0, "All constraints should be satisfied, if problem has constraints"
    if problem_has_constraints and arg_ignore_constraints:
        assert score.constraints < 1.0, "Not all constraints are expected to be satisfied"
