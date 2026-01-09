import pytest
from click.testing import CliRunner

from max_div._cli import benchmark, numba_status, solve
from max_div.benchmarks import BenchmarkProblemFactory


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


@pytest.mark.parametrize(
    "options",
    [
        ["--turbo", "--markdown"],
        ["--turbo"],
        ["--speed=1.0"],
        ["--turbo", "--optimization-only"],
        ["--turbo", "--initialization-only"],
    ],
)
@pytest.mark.parametrize("test_problem", list(BenchmarkProblemFactory.get_all_benchmark_problems().keys()) + ["all"])
def test_cli_benchmark_solver_run(options: list[str], test_problem: str):
    # --- arrange -----------------------------------------
    runner = CliRunner()

    # --- act ---------------------------------------------
    result = runner.invoke(benchmark, ["solver", "run", *options, test_problem])

    # --- assert ------------------------------------------
    assert result.exit_code == 0


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


# =================================================================================================
#  benchmark platform
# =================================================================================================
@pytest.mark.parametrize(
    "options",
    [
        ["--fast", "--verbose"],
        ["--fast"],
    ],
)
def test_cli_benchmark_platform(options: list[str]):
    # --- arrange -----------------------------------------
    runner = CliRunner()

    # --- act ---------------------------------------------
    result = runner.invoke(benchmark, ["platform", *options])

    # --- assert ------------------------------------------
    assert result.exit_code == 0


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
@pytest.mark.parametrize("test_problem", list(BenchmarkProblemFactory.get_all_benchmark_problems().keys()))
def test_cli_solve(options: list[str], test_problem: str, expected_exit_code: int):
    # --- arrange -----------------------------------------
    runner = CliRunner()

    # --- act ---------------------------------------------
    result = runner.invoke(solve, [*options, test_problem])

    # --- assert ------------------------------------------
    assert result.exit_code == expected_exit_code


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
