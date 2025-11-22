import pytest
from click.testing import CliRunner

from max_div._cli import benchmark, numba_status


@pytest.mark.parametrize("sub_command", ["randint", "randint_constrained"])
@pytest.mark.parametrize("options", ["--turbo", "--speed=1.0"])
def test_cli_benchmark(sub_command: str, options: str):
    # --- arrange -----------------------------------------
    runner = CliRunner()

    # --- act ---------------------------------------------
    result = runner.invoke(benchmark, [options, sub_command])

    # --- assert ------------------------------------------
    assert result.exit_code == 0


def test_cli_numba_status():
    # --- arrange -----------------------------------------
    runner = CliRunner()

    # --- act ---------------------------------------------
    result = runner.invoke(numba_status)

    # --- assert ------------------------------------------
    assert result.exit_code == 0
