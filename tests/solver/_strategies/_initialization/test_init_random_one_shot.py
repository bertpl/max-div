import pytest

from max_div.solver._strategies import InitializationStrategy

from ._helpers import new_solver_state


@pytest.mark.parametrize("problem_has_constraints", [True, False])
@pytest.mark.parametrize("arg_ignore_constraints", [True, False])
@pytest.mark.parametrize("arg_uniform", [True, False])
def test_init_random_one_shot(problem_has_constraints: bool, arg_ignore_constraints: bool, arg_uniform: bool):
    # --- arrange -----------------------------------------
    solver_state = new_solver_state(problem_has_constraints)
    strategy = InitializationStrategy.random_one_shot(
        ignore_constraints=arg_ignore_constraints,
        uniform=arg_uniform,
    )

    # --- act ---------------------------------------------
    strategy.initialize(solver_state)
    score = solver_state.score

    # --- assert ------------------------------------------
    assert score.size == 1.0, "Selection size should be equal to k after initialization"
    if not arg_ignore_constraints:
        assert score.constraints == 1.0, "All constraints should be satisfied, if ignore_constraints=False"
    if problem_has_constraints and arg_ignore_constraints:
        assert score.constraints < 1.0, "Not all constraints are expected to be satisfied"
