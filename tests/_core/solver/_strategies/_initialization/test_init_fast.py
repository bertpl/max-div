import pytest

from max_div._core.solver._solver_step import InitializationStep
from max_div._core.solver._step_identity import SolverStepIdentity
from max_div._core.solver._strategies._initialization._init_fast import InitFast

from ._helpers import new_solver_state

# each step records its checkpoints under this identity when run on its own in these tests
_STEP_IDENTITY = SolverStepIdentity(1, "test")


@pytest.mark.parametrize("problem_has_constraints", [True, False])
def test_init_fast(problem_has_constraints: bool):
    # --- arrange ----------------------
    solver_state = new_solver_state(problem_has_constraints)
    strategy = InitFast()
    init_step = InitializationStep(strategy)

    # --- act --------------------------
    init_step.run(solver_state, _STEP_IDENTITY)
    score = solver_state.score

    # --- assert -----------------------
    assert score.size == 1.0, "Selection size should be equal to k after initialization"
