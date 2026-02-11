import pytest
from click.testing import CliRunner

from max_div._cli import benchmark
from max_div._cli._cmd_benchmark_solver_presets import resolve_presets, resolve_problems
from max_div.benchmarks import BenchmarkProblemFactory
from max_div.solver import SolverPreset


# =================================================================================================
#  benchmark solver
# =================================================================================================
def test_cli_benchmark_solver_list_problems():
    # --- arrange -----------------------------------------
    runner = CliRunner()

    # --- act ---------------------------------------------
    result = runner.invoke(benchmark, ["solver", "list_problems"])

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
@pytest.mark.parametrize("test_problem", list(BenchmarkProblemFactory.get_all_benchmark_names()) + ["all"])
def test_cli_benchmark_solver_strategies(options: list[str], test_problem: str):
    # --- arrange -----------------------------------------
    runner = CliRunner()
    options.append(f"--problem={test_problem}")

    # --- act ---------------------------------------------
    result = runner.invoke(benchmark, ["solver", "strategies", *options])

    # --- assert ------------------------------------------
    assert result.exit_code == 0


@pytest.mark.parametrize(
    "options",
    [
        ["--dry-run", "--turbo", "--markdown"],
        ["--dry-run", "--size=100", "--turbo"],
        ["--dry-run", "--size=10", "--speed=1.0"],
        ["--dry-run", "--json-file", "--size=100", "--target-max-minutes=1000000"],
        ["--dry-run", "--json-file", "--size=100", "--target-max-minutes=10"],
        ["--dry-run", "--markdown-file", "--size=100", "--target-max-minutes=0.00000001"],
        ["--size=1", "--json-file", "--preset=random", "--speed=1.0"],
    ],
)
@pytest.mark.parametrize("test_problem", BenchmarkProblemFactory.get_all_benchmark_names() + ["all"])
def test_cli_benchmark_solver_presets(options: list[str], test_problem: str):
    # --- arrange -----------------------------------------
    runner = CliRunner()
    options.append(f"--problem={test_problem}")

    # --- act ---------------------------------------------
    with runner.isolated_filesystem():
        result = runner.invoke(benchmark, ["solver", "presets", *options])

        # --- assert ------------------------------------------
        assert result.exit_code == 0


# =================================================================================================
#  Helpers
# =================================================================================================
def test_cli_benchmark_solver_preset_resolve_problems():
    # --- arrange -----------------------------------------
    all_supported_problems = BenchmarkProblemFactory.get_all_benchmark_names()
    problem_1 = all_supported_problems[0]
    problem_2 = all_supported_problems[1]

    # --- act & assert ------------------------------------
    for problem in all_supported_problems:
        assert resolve_problems(problem) == [problem]

    assert resolve_problems(f"{problem_1},{problem_2}") == [problem_1, problem_2]
    assert resolve_problems("all") == all_supported_problems

    with pytest.raises(ValueError):
        _ = resolve_problems("non_existing_problem")


def test_cli_benchmark_solver_preset_resolve_presets():
    # --- arrange -----------------------------------------
    all_presets: list[SolverPreset] = list(SolverPreset)
    all_presets_except_default = [p for p in all_presets if p != SolverPreset.DEFAULT]
    preset_1 = all_presets_except_default[0]
    preset_2 = all_presets_except_default[1]

    # --- act & assert ------------------------------------
    for preset in all_presets_except_default:
        assert resolve_presets(preset.value) == [preset]

    assert resolve_presets("default") == [SolverPreset.DEFAULT.resolve_alias()]
    assert resolve_presets(f"{preset_1.value},{preset_2.value}") == sorted([preset_1, preset_2])
    assert resolve_presets("all") == sorted(all_presets_except_default)

    with pytest.raises(ValueError):
        _ = resolve_presets("non_existing_preset")
