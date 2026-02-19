import pytest
from click.testing import CliRunner

from max_div._core._cli import benchmark


# =================================================================================================
#  benchmark internal
# =================================================================================================
@pytest.mark.parametrize(
    "sub_command",
    [
        "all",
        "randint",
        "randint_constrained",
        "diversity_metrics",
        "modify_p_selectivity",
    ],
)
@pytest.mark.parametrize(
    "options",
    [
        ["--turbo", "--markdown"],
        ["--turbo"],
        ["--speed=1.0"],
    ],
)
def test_cli_benchmark_internal(sub_command: str, options: list[str]):
    # --- arrange -----------------------------------------
    runner = CliRunner()

    # --- act ---------------------------------------------
    result = runner.invoke(benchmark, ["internal", *options, sub_command])

    # --- assert ------------------------------------------
    assert result.exit_code == 0
