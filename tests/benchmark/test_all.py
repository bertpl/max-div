import pytest

from max_div.benchmark import benchmark_diversity_metrics, benchmark_randint, benchmark_randint_constrained


@pytest.mark.parametrize("markdown", [True, False])
def test_benchmark_randint(markdown: bool):
    benchmark_randint(speed=1.0, markdown=markdown)


@pytest.mark.parametrize("markdown", [True, False])
def test_benchmark_randint_constrained(markdown: bool):
    benchmark_randint_constrained(speed=1.0, markdown=markdown)


@pytest.mark.parametrize("markdown", [True, False])
def test_benchmark_diversity_metrics(markdown: bool):
    benchmark_diversity_metrics(speed=1.0, markdown=markdown)
