import numpy as np
import pytest

from max_div._core.problem import MaxDivProblem
from max_div._core.solver._builders import MaxDivSolverBuilder, ParallelMaxDivSolverBuilder, SolverBuilderBase


def _problem() -> MaxDivProblem:
    """Return a problem with n=20 and k=5."""
    return MaxDivProblem.new(np.random.default_rng(20260926).random((20, 3)).astype(np.float32), k=5)


# =================================================================================================
#  with_initial_selection
# =================================================================================================
@pytest.mark.parametrize("builder_class", [MaxDivSolverBuilder, ParallelMaxDivSolverBuilder])
@pytest.mark.parametrize(
    "indices, message",
    [
        ([0, 1, 2, 3], "exactly k=5 indices; got 4"),
        ([0, 1, 2, 3, 20], "Index 20 lies outside the population of n=20 items"),
        ([0, 1, 2, 3, 3], "Index 3 appears more than once"),
    ],
)
def test_with_initial_selection_rejects_a_selection_that_does_not_fit_when_called(
    builder_class: type[SolverBuilderBase], indices: list[int], message: str
):
    """The builder knows n and k, so every check runs when the selection is given, not when the solve starts."""
    # --- act / assert -----------------
    with pytest.raises(ValueError, match=message):
        builder_class(_problem()).with_initial_selection(indices)
