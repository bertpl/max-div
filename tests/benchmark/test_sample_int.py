import pytest

from max_div.benchmark import benchmark_randint


@pytest.mark.parametrize("markdown", [True, False])
def test_benchmark_randint(markdown: bool):
    benchmark_randint(speed=1.0, markdown=markdown)
