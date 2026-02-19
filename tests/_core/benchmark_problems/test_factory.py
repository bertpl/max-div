import pytest

from max_div._core.benchmark_problems import BenchmarkProblem, BenchmarkProblemFactory
from max_div._core.metrics import DiversityMetric
from max_div._core.problem import MaxDivProblem


def test_benchmark_problem_factory_show_all():
    BenchmarkProblemFactory.show_all()  # just ensure no errors occur


def test_benchmark_problem_factory_get_all_benchmark_problems():
    # --- act ---------------------------------------------
    problems_dict = BenchmarkProblemFactory.get_all_benchmark_problems()

    # --- assert ------------------------------------------
    assert isinstance(problems_dict, dict)
    assert len(problems_dict) == 8  # number of built-in problems; adjust number as we add more
    assert all(isinstance(name, str) for name in problems_dict.keys())
    assert all(isinstance(cls, type) and issubclass(cls, BenchmarkProblem) for cls in problems_dict.values())

    assert "all" not in [name.lower().strip() for name in problems_dict.keys()], (
        "'all' should not be used as name, to avoid conflict with `benchmark solver run all` CLI command"
    )


@pytest.mark.parametrize("name", list(BenchmarkProblemFactory.get_all_benchmark_problems().keys()))
def test_benchmark_problem_factory_create_problem(name: str):
    # --- act ---------------------------------------------
    problem_cls = BenchmarkProblemFactory.get_all_benchmark_problems()[name]
    example_params = problem_cls.get_example_parameters()
    problem_instance = BenchmarkProblemFactory.construct_problem(name, **example_params)

    # --- assert ------------------------------------------
    assert isinstance(problem_instance, MaxDivProblem)


def test_benchmark_problem_factory_create_problem_invalid_name():
    # --- arrange -----------------------------------------
    invalid_name = "NonExistentBenchmarkProblem"

    # --- act & assert ------------------------------------
    with pytest.raises(ValueError):
        BenchmarkProblemFactory.construct_problem(invalid_name)


def test_benchmark_problem_factory_create_problem_invalid_params():
    # --- arrange -----------------------------------------
    valid_name = list(BenchmarkProblemFactory.get_all_benchmark_problems().keys())[0]
    invalid_params = {"non_existent_param": 42}

    # --- act & assert ------------------------------------
    with pytest.raises(ValueError):
        BenchmarkProblemFactory.construct_problem(valid_name, **invalid_params)


@pytest.mark.parametrize("benchmark_name", list(BenchmarkProblemFactory.get_all_benchmark_problems().keys()))
@pytest.mark.parametrize("size", [2, 4, 8, 16])
def test_benchmark_problem_factory_get_problem_dimensions(benchmark_name: str, size: int):
    # --- act ---------------------------------------------
    problem = BenchmarkProblemFactory.construct_problem(
        benchmark_name,
        size=size,
        diversity_metric=DiversityMetric.MIN_SEPARATION,
    )
    d, n, k, m, n_con_indices = BenchmarkProblemFactory.get_problem_dimensions(benchmark_name, size=size)

    # --- assert ------------------------------------------
    assert problem.d == d
    assert problem.n == n
    assert problem.k == k
    assert problem.m == m
    assert 0.9 * n_con_indices <= problem.n_constraint_indices <= 1.1 * n_con_indices
