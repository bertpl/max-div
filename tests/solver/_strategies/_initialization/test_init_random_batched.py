import numpy as np
import pytest

from max_div.solver._strategies import InitializationStrategy
from max_div.solver._strategies._initialization._init_random_batched import InitRandomBatched, _get_batch_sizes

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

    # --- act ---------------------------------------------
    strategy.initialize(solver_state)
    score = solver_state.score

    # --- assert ------------------------------------------
    assert score.size == 1.0, "Selection size should be equal to k after initialization"
    if not arg_ignore_constraints:
        assert score.constraints == 1.0, "All constraints should be satisfied, if ignore_constraints=True"
    if problem_has_constraints and arg_ignore_constraints:
        assert score.constraints < 1.0, "Not all constraints are expected to be satisfied"


# =================================================================================================
#  TESTS - helpers
# =================================================================================================
@pytest.mark.parametrize("k", [1, 2, 3, 4, 6, 8, 10, 13, 17, 23, 47])
@pytest.mark.parametrize("b", [1, 2, 3, 4, 6, 8, 10, 13, 17, 23, 47])
def test_get_batch_sizes(k: int, b: int):
    # --- act ---------------------------------------------
    batch_sizes = _get_batch_sizes(np.int32(k), np.int32(b))

    # --- assert ------------------------------------------

    # check shape & dtype
    assert batch_sizes.ndim == 1
    assert batch_sizes.size == b
    assert batch_sizes.dtype == np.int32

    # check invariants
    assert sum(batch_sizes) == k, "Sum of batch sizes should be equal to k"
    assert max(batch_sizes) <= min(batch_sizes) + 1, "Max. two values should be present (k//b and -if needed- k//b + 1)"
    assert list(batch_sizes) == sorted(batch_sizes, reverse=True), "Larger batches should come first"
