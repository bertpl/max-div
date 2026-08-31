"""Pin the SolverState mutation preconditions every strategy must honor.

`add`/`add_many`/`remove`/`remove_many` run every swap iteration and deliberately carry no
range or duplicate checks (see their docstrings); this test enforces those preconditions on
everything the strategies actually emit, by instrumenting the mutation methods and running
each preset end to end.
"""

import numpy as np
import pytest

from max_div._core.constraints import Constraint
from max_div._core.problem import MaxDivProblem
from max_div._core.solver import MaxDivSolverBuilder
from max_div._core.solver._duration import iterations
from max_div._core.solver._presets import SolverPreset
from max_div._core.solver._solver_state import SolverState


def _checked(method_name: str, violations: list[str]):
    """Return a wrapper of the named SolverState method that records precondition violations."""
    original = getattr(SolverState, method_name)

    def checked(self: SolverState, indices_or_index, *args, **kwargs) -> None:
        indices = np.atleast_1d(np.asarray(indices_or_index))
        if indices.size and (indices.min() < 0 or indices.max() >= self.n):
            violations.append(f"{method_name}: out of range {indices.tolist()}")
        if len(np.unique(indices)) != len(indices):
            violations.append(f"{method_name}: duplicates {indices.tolist()}")
        return original(self, indices_or_index, *args, **kwargs)

    return checked


@pytest.mark.parametrize("preset", [SolverPreset.RANDOM, SolverPreset.GUIDED, SolverPreset.SMART])
@pytest.mark.parametrize("constrained", [False, True], ids=["unconstrained", "constrained"])
def test_strategies_emit_in_range_duplicate_free_batches(monkeypatch, preset: SolverPreset, constrained: bool):
    """Every index batch a strategy hands the state is in range and duplicate-free."""
    # --- arrange ----------------------
    violations: list[str] = []
    for method_name in ("add", "add_many", "remove", "remove_many"):
        monkeypatch.setattr(SolverState, method_name, _checked(method_name, violations))
    vectors = np.random.default_rng(0).random((60, 3)).astype(np.float32)
    constraints = [Constraint(int_set=set(range(20)), min_count=3, max_count=8)] if constrained else None
    problem = MaxDivProblem.new(vectors, k=12, constraints=constraints)

    # --- act --------------------------
    solution = MaxDivSolverBuilder(problem).with_preset(iterations(200), preset=preset).build().solve(verbosity=0)

    # --- assert -----------------------
    assert violations == []
    assert len(set(solution.i_selected.tolist())) == 12
