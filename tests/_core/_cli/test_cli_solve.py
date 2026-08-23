import pytest
from click.testing import CliRunner

from max_div._core._cli import solve
from tests.helpers import swept_benchmark_problems


# =================================================================================================
#  solve
# =================================================================================================
@pytest.mark.parametrize(
    "options, expected_exit_code",
    [
        (["--n=200", "--preset=default"], 0),
        (["--iterations=10", "--n=300"], 0),
        (["--seconds=0.001", "--n=400"], 0),
        (["--seconds=0.001"], 0),
        (["--total-seconds=0.001", "--n=400"], 0),
        (["--seconds=0.001", "--iterations=1000"], 2),
        (["--total-seconds=0.001", "--seconds=0.001"], 2),
    ],
)
@pytest.mark.parametrize("test_problem", swept_benchmark_problems())
# the --total-seconds smoke case runs a budget too small to reach optimization, which warns by
# design; the warning has its own test
@pytest.mark.filterwarnings("ignore::max_div._core._warnings.SolverBudgetWarning")
def test_cli_solve(options: list[str], test_problem: str, expected_exit_code: int):
    # --- arrange ----------------------
    runner = CliRunner()

    # --- act --------------------------
    result = runner.invoke(solve, [*options, test_problem])

    # --- assert -----------------------
    assert result.exit_code == expected_exit_code
