import pytest
from click.testing import CliRunner

from max_div._cli import benchmark, numba_status


# =================================================================================================
#  benchmark solver
# =================================================================================================
def test_cli_benchmark_solver_list():
    # --- arrange -----------------------------------------
    runner = CliRunner()

    # --- act ---------------------------------------------
    result = runner.invoke(benchmark, ["solver", "list"])

    # --- assert ------------------------------------------
    assert result.exit_code == 0


# =================================================================================================
#  benchmark internal
# =================================================================================================
@pytest.mark.parametrize(
    "sub_command",
    [
        "randint",
        "randint_constrained",
        "diversity_metrics",
        "modify_p_selectivity",
    ],
)
@pytest.mark.parametrize("options", ["--turbo", "--speed=1.0"])
def test_cli_benchmark_internal(sub_command: str, options: str):
    # --- arrange -----------------------------------------
    runner = CliRunner()

    # --- act ---------------------------------------------
    result = runner.invoke(benchmark, ["internal", options, sub_command])

    # --- assert ------------------------------------------
    assert result.exit_code == 0


# =================================================================================================
#  numba_status
# =================================================================================================
def test_cli_numba_status():
    # --- arrange -----------------------------------------
    runner = CliRunner()

    # --- act ---------------------------------------------
    result = runner.invoke(numba_status)

    # --- assert ------------------------------------------
    assert result.exit_code == 0
