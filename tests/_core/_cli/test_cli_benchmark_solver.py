import pytest
from click.testing import CliRunner

from max_div._core._cli import benchmark
from max_div._core._cli._cmd_benchmark_solver_presets import resolve_presets, resolve_problems
from max_div._core.benchmark_problems import BenchmarkProblemFactory
from max_div._core.solver import SolverPreset
from tests.helpers import swept_benchmark_problems


# =================================================================================================
#  benchmark solver
# =================================================================================================
def test_cli_benchmark_solver_list_problems():
    # --- arrange ----------------------
    runner = CliRunner()

    # --- act --------------------------
    result = runner.invoke(benchmark, ["solver", "list_problems"])

    # --- assert -----------------------
    assert result.exit_code == 0


@pytest.mark.parametrize(
    "options",
    [
        ["--turbo", "--markdown"],
        ["--turbo"],
        ["--turbo", "--file"],
    ],
    ids=["markdown", "console", "file"],
)
def test_cli_benchmark_solver_feasibility(options: list[str], tmp_path, monkeypatch):
    """The feasibility subcommand succeeds in every output mode."""
    # --- arrange ----------------------
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()

    # --- act --------------------------
    result = runner.invoke(benchmark, ["solver", "feasibility", "--problem=C1", *options])

    # --- assert -----------------------
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
@pytest.mark.parametrize("test_problem", [*swept_benchmark_problems(), "all"])
def test_cli_benchmark_solver_strategies(options: list[str], test_problem: str):
    # --- arrange ----------------------
    runner = CliRunner()
    options.append(f"--problem={test_problem}")

    # --- act --------------------------
    result = runner.invoke(benchmark, ["solver", "strategies", *options])

    # --- assert -----------------------
    assert result.exit_code == 0


@pytest.mark.parametrize(
    "options",
    [
        ["--dry-run", "--turbo", "--markdown"],
        ["--dry-run", "--n=10000", "--turbo"],
        ["--dry-run", "--n=1000", "--speed=1.0"],
        ["--dry-run", "--json-file", "--n=10000", "--max-run-duration-minutes=60"],
        ["--dry-run", "--markdown-file", "--n=10000", "--speed=0.5"],
        ["--n=100", "--json-file", "--preset=random", "--speed=1.0"],
    ],
)
@pytest.mark.parametrize("test_problem", [*swept_benchmark_problems(), "all"])
def test_cli_benchmark_solver_presets(options: list[str], test_problem: str):
    # --- arrange ----------------------
    runner = CliRunner()
    options.append(f"--problem={test_problem}")

    # --- act --------------------------
    with runner.isolated_filesystem():
        result = runner.invoke(benchmark, ["solver", "presets", *options])

        # --- assert -------------------
        assert result.exit_code == 0


def test_cli_benchmark_solver_presets_turbo_runs_parallel_arm():
    """--turbo with SMART executes the parallel arm's single short run end to end."""
    # --- arrange ----------------------
    runner = CliRunner()

    # --- act --------------------------
    with runner.isolated_filesystem():
        result = runner.invoke(benchmark, ["solver", "presets", "--turbo", "--preset=smart", "--problem=U1", "--n=100"])

    # --- assert -----------------------
    assert result.exit_code == 0
    assert "parallel runs : 1" in result.output


# =================================================================================================
#  Helpers
# =================================================================================================
def test_cli_benchmark_solver_preset_resolve_problems():
    # --- arrange ----------------------
    all_supported_problems = BenchmarkProblemFactory.get_all_benchmark_names()
    problem_1 = all_supported_problems[0]
    problem_2 = all_supported_problems[1]

    # --- act & assert -----------------
    for problem in all_supported_problems:
        assert resolve_problems(problem) == [problem]

    assert resolve_problems(f"{problem_1},{problem_2}") == [problem_1, problem_2]
    assert resolve_problems("all") == all_supported_problems

    with pytest.raises(ValueError):
        _ = resolve_problems("non_existing_problem")


def test_cli_benchmark_solver_preset_resolve_presets():
    # --- arrange ----------------------
    all_presets: list[SolverPreset] = list(SolverPreset)
    all_presets_except_default = [p for p in all_presets if p != SolverPreset.DEFAULT]
    preset_1 = all_presets_except_default[0]
    preset_2 = all_presets_except_default[1]

    # --- act & assert -----------------
    for preset in all_presets_except_default:
        assert resolve_presets(preset.value) == [preset]

    assert resolve_presets("default") == [SolverPreset.DEFAULT.resolve_alias()]
    assert resolve_presets(f"{preset_1.value},{preset_2.value}") == sorted([preset_1, preset_2])
    assert resolve_presets("all") == sorted(all_presets_except_default)

    with pytest.raises(ValueError):
        _ = resolve_presets("non_existing_preset")
