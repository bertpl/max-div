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
        (["--size=2", "--preset=default"], 0),
        (["--iterations=10", "--size=3"], 0),
        (["--seconds=0.001", "--size=4"], 0),
        (["--seconds=0.001"], 0),
        (["--seconds=0.001", "--iterations=1000"], 2),
    ],
)
@pytest.mark.parametrize("test_problem", swept_benchmark_problems())
def test_cli_solve(options: list[str], test_problem: str, expected_exit_code: int):
    # --- arrange -----------------------------------------
    runner = CliRunner()

    # --- act ---------------------------------------------
    result = runner.invoke(solve, [*options, test_problem])

    # --- assert ------------------------------------------
    assert result.exit_code == expected_exit_code
