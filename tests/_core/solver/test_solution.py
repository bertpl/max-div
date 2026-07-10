import numpy as np

from max_div._core.solver import MaxDivSolution
from max_div._core.solver._duration import Elapsed
from max_div._core.solver._score import Score


def test_solution_str_with_constraints():
    # --- arrange -----------------------------------------
    solution = MaxDivSolution(
        i_selected=np.array([1, 3, 5, 7, 9], dtype=np.int32),
        score_checkpoints=[
            ("step 0/1", Elapsed(t_elapsed_sec=1.23, n_iterations=456), Score(1.0, 1.0, 0.7705, (1.0,))),
        ],
        step_durations={"step 0/1": Elapsed(t_elapsed_sec=1.23, n_iterations=456)},
        n_constraints=3,
        n_constraints_satisfied=2,
    )

    # --- act ---------------------------------------------
    result = str(solution)

    # --- assert ------------------------------------------
    assert "5 vectors selected" in result
    assert "diversity=0.7705" in result
    assert "constraints: 2/3 satisfied" in result
    assert "456 iterations" in result


def test_solution_str_without_constraints():
    # --- arrange -----------------------------------------
    solution = MaxDivSolution(
        i_selected=np.array([0, 2, 4], dtype=np.int32),
        score_checkpoints=[
            ("step 0/1", Elapsed(t_elapsed_sec=0.5, n_iterations=100), Score(1.0, 1.0, 0.5, ())),
        ],
        step_durations={"step 0/1": Elapsed(t_elapsed_sec=0.5, n_iterations=100)},
    )

    # --- act ---------------------------------------------
    result = str(solution)

    # --- assert ------------------------------------------
    assert "3 vectors selected" in result
    assert "diversity=0.5000" in result
    assert "constraints" not in result
    assert "100 iterations" in result
