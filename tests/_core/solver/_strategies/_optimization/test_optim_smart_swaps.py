import numpy as np

from max_div._core.constraints import Constraint
from max_div.metrics import DiversityMetric
from max_div.problem import MaxDivProblem
from max_div.solver import MaxDivSolverBuilder, SolverPreset, Verbosity, iterations


def test_smart_preset_survives_tiny_problem_full_swap():
    # Regression: on tiny problems, SMART's adaptive swap size can grow to k, so the
    # removal loop ends up weighing a sole selected item whose contribution is +inf —
    # which used to NaN the removal probabilities and crash the solver.
    # --- arrange ----------------------
    rng = np.random.default_rng(7)
    vectors = rng.random((40, 3)).astype(np.float32)
    constraints = [
        Constraint(int_set=set(range(20)), min_count=2, max_count=3),
        Constraint(int_set=set(range(20, 40)), min_count=2, max_count=4),
    ]
    problem = MaxDivProblem.new(
        vectors=vectors, k=6, diversity_metric=DiversityMetric.MEAN_SEPARATION, constraints=constraints
    )
    solver = MaxDivSolverBuilder(problem).with_preset(iterations(800), SolverPreset.SMART).with_seed(0).build()

    # --- act --------------------------
    solution = solver.solve(verbosity=Verbosity.SILENT)

    # --- assert -----------------------
    assert len(solution.i_selected) == 6
    assert solution.n_constraints_satisfied == 2
