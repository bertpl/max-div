import pytest

from max_div._core._utils import stdout_to_file
from max_div._core.solver import MaxDivSolution
from max_div._core.solver._duration import Elapsed
from max_div._core.solver._score import Score


# =================================================================================================
#  Helpers
# =================================================================================================
def assert_score_checkpoints_are_sane(score_checkpoints: list[tuple[str, Elapsed, Score]]):
    # --- non-empty -------------------
    assert len(score_checkpoints) >= 1, "score_checkpoints must contain at least one entry"

    # --- check step names ------------
    singular_step_names = []
    for step_name, _, _ in score_checkpoints:
        if (len(singular_step_names) == 0) or (step_name != singular_step_names[-1]):
            # only deduplicate consecutive identical step names
            singular_step_names.append(step_name)

    assert len(set(singular_step_names)) == len(singular_step_names), (
        "score_checkpoints contains duplicate non-consecutive step names"
    )

    for i, step_name in enumerate(singular_step_names):
        # e.g. if we have 4 steps reported...
        #  - first step is step 0/3 representing SolverState initialization
        #  - other steps are step 1/3, step 2/3, step 3/3, represent actual SolverSteps
        assert f"{i}/{len(singular_step_names) - 1}" in step_name

    # --- check iteration counts ------
    iter_values = [e.n_iterations for _, e, _ in score_checkpoints]
    assert min(iter_values) >= 0, "score_checkpoints contains negative iteration counts"
    assert len(iter_values) == len(set(iter_values)), "score_checkpoints contains duplicate iteration counts"
    assert iter_values == sorted(iter_values), "score_checkpoints iteration counts should be strictly increasing"

    # --- check elapsed times ---------
    t_values = [e.t_elapsed_sec for _, e, _ in score_checkpoints]
    assert min(t_values) >= 0.0, "score_checkpoints contains negative elapsed times"
    # NOTE: duplicate time values can happen if iterations are very fast, so we don't assert uniqueness here
    assert t_values == sorted(t_values), "score_checkpoints elapsed times should be non-decreasing"


# =================================================================================================
#  Tests
# =================================================================================================
def test_solver_minimal(example_solver):
    # --- act ---------------------------------------------
    solution = example_solver.solve()

    # --- assert ------------------------------------------
    assert isinstance(solution, MaxDivSolution)
    assert_score_checkpoints_are_sane(solution.score_checkpoints)
    assert solution.duration == sum(list(solution.step_durations.values()))
    assert solution.duration == solution.score_checkpoints[-1][1]
    assert solution.score == solution.score_checkpoints[-1][2]


@pytest.mark.parametrize(
    "verbosity,error_expected",
    [
        (0, False),
        (10, False),
        (20, False),
        (21, False),
        (22, False),
        (23, False),
        (24, True),
        (25, False),
        (26, True),
        (42, True),
    ],
)
def test_solver_verbosity(example_solver, tmp_path, verbosity: int, error_expected: bool):
    # --- act & assert ------------------------------------
    if not error_expected:
        # arrange
        output_file = tmp_path / "output.txt"

        # act
        with stdout_to_file(filename=output_file):
            _ = example_solver.solve(verbosity=verbosity)

        # assert
        output_content = output_file.read_text()
        if verbosity == 0:
            assert len(output_content) == 0, f"Expected no output for verbosity=0, but got: {output_content}"
        else:
            assert output_content != "", f"Expected output for verbosity={verbosity}, but file is empty"

    else:
        # act & assert
        with pytest.raises(ValueError):
            _ = example_solver.solve(verbosity=verbosity)
