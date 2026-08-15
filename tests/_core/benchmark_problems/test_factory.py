import pytest

from max_div._core.benchmark_problems import BenchmarkProblem, BenchmarkProblemFactory
from max_div._core.metrics import DiversityMetric
from max_div._core.problem import MaxDivProblem

ALL_PROBLEM_NAMES = list(BenchmarkProblemFactory.get_all_benchmark_problems().keys())


def test_benchmark_problem_factory_show_all():
    BenchmarkProblemFactory.show_all()  # just ensure no errors occur


def test_benchmark_problem_factory_get_all_benchmark_problems():
    # --- act ---------------------------------------------
    problems_dict = BenchmarkProblemFactory.get_all_benchmark_problems()

    # --- assert ------------------------------------------
    assert isinstance(problems_dict, dict)
    assert len(problems_dict) == 8  # number of built-in problems; adjust number as we add more
    assert all(isinstance(name, str) for name in problems_dict)
    assert all(isinstance(cls, type) and issubclass(cls, BenchmarkProblem) for cls in problems_dict.values())

    assert "all" not in [name.lower().strip() for name in problems_dict], (
        "'all' should not be used as name, to avoid conflict with `benchmark solver run all` CLI command"
    )


@pytest.mark.parametrize("name", ALL_PROBLEM_NAMES)
def test_benchmark_problem_factory_create_problem(name: str):
    # --- act ---------------------------------------------
    problem_instance = BenchmarkProblemFactory.construct_problem(
        name, n=100, diversity_metric=DiversityMetric.APPROX_GEOMEAN_SEPARATION
    )

    # --- assert ------------------------------------------
    assert isinstance(problem_instance, MaxDivProblem)


def test_benchmark_problem_factory_create_problem_invalid_name():
    # --- arrange -----------------------------------------
    invalid_name = "NonExistentBenchmarkProblem"

    # --- act & assert ------------------------------------
    with pytest.raises(ValueError, match="not registered"):
        BenchmarkProblemFactory.construct_problem(invalid_name, n=100, diversity_metric=DiversityMetric.MIN_SEPARATION)


@pytest.mark.parametrize("name", ALL_PROBLEM_NAMES)
@pytest.mark.parametrize("n", [0, 19])
def test_benchmark_problem_factory_rejects_degenerate_n(name: str, n: int):
    """n below 20 is degenerate (k < 2) and both entry points refuse it."""
    with pytest.raises(ValueError, match="n >= 20"):
        BenchmarkProblemFactory.construct_problem(name, n=n, diversity_metric=DiversityMetric.MIN_SEPARATION)
    with pytest.raises(ValueError, match="n >= 20"):
        BenchmarkProblemFactory.get_problem_dimensions(name, n=n)


@pytest.mark.parametrize("name", ALL_PROBLEM_NAMES)
@pytest.mark.parametrize("n", [20, 37, 100, 137, 314])
def test_benchmark_problem_factory_get_problem_dimensions(name: str, n: int):
    """The dimensions report matches the constructed problem at round and odd n alike."""
    # --- act ---------------------------------------------
    problem = BenchmarkProblemFactory.construct_problem(
        name,
        n=n,
        diversity_metric=DiversityMetric.MIN_SEPARATION,
    )
    d, n_reported, k, m, n_con_indices = BenchmarkProblemFactory.get_problem_dimensions(name, n=n)

    # --- assert ------------------------------------------
    assert problem.d == d
    assert problem.n == n_reported == n
    assert problem.k == k
    assert problem.m == m
    assert d >= 1
    assert k >= 2
    assert 0.9 * n_con_indices <= problem.n_constraint_indices <= 1.1 * n_con_indices
