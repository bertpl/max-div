from max_div.solver import MaxDivSolution


def test_solver_minimal(example_problem_1):
    # --- act ---------------------------------------------
    solution = example_problem_1.solve()

    # --- assert ------------------------------------------
    assert isinstance(solution, MaxDivSolution)
