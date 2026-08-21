import numpy as np
import pytest

from max_div._core.problem import MaxDivProblem
from max_div._core.solver._builders import ParallelMaxDivSolverBuilder
from max_div._core.solver._duration import total_seconds
from max_div._core.solver._parallel import _solver as parallel_solver_module
from max_div._core.solver._progress_reporting import Verbosity


def _problem() -> MaxDivProblem:
    """Return a problem small enough to solve over spawned workers in a test."""
    rng = np.random.default_rng(20260821)
    return MaxDivProblem.new(rng.random((60, 3)).astype(np.float32), k=6)


# =================================================================================================
#  Total time budget
# =================================================================================================
def test_solve_starts_the_total_budget(monkeypatch, fake_clock):
    """A portfolio's build only assembles configurations, so the budget starts at solve instead."""
    # --- arrange ----------------------
    duration = total_seconds(10.0)
    solver = ParallelMaxDivSolverBuilder(_problem()).with_workers(duration, 2).build()
    fake_clock.advance(4.0)
    remaining_when_workers_start = []

    def spy_run_portfolio(configs, *args, **kwargs):
        """Record what each worker has left when it would start, and run none of them."""
        remaining_when_workers_start.extend(config.solver_steps[-1]._duration.remaining_seconds() for config in configs)
        return []

    monkeypatch.setattr(parallel_solver_module, "run_portfolio", spy_run_portfolio)

    # --- act --------------------------
    with pytest.raises(ValueError, match="no results"):  # no worker ran, so there is nothing to pick a winner from
        solver.solve(verbosity=Verbosity.SILENT)

    # --- assert -----------------------
    assert remaining_when_workers_start == [pytest.approx(10.0), pytest.approx(10.0)]
    assert duration.remaining_seconds() == pytest.approx(6.0)  # the caller's own budget is left alone


def test_a_spent_total_budget_reaches_the_workers_as_spent():
    """Workers read the parent's deadline against their own clock, so a spent budget must stay spent."""
    # --- arrange ----------------------
    problem = _problem()
    solver = ParallelMaxDivSolverBuilder(problem).with_workers(total_seconds(0.01), 2, n_groups=2).build()

    # --- act --------------------------
    # spawning the workers alone outlasts the budget, so every one of them starts with nothing left
    solution = solver.solve(verbosity=Verbosity.SILENT)

    # --- assert -----------------------
    optimization_steps = [name for name in solution.step_durations if "Optim" in name]
    assert len(optimization_steps) == 1
    assert solution.step_durations[optimization_steps[0]].n_iterations == 0
    assert len(solution.i_selected) == problem.k
