import pytest

from max_div.benchmark import benchmark_randint_constrained


@pytest.mark.parametrize("markdown", [True, False])
def test_benchmark_randint_constrained(markdown: bool):
    benchmark_randint_constrained(speed=1.0, markdown=markdown)
