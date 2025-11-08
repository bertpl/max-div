import pytest

from max_div.benchmark import benchmark_sample_int


@pytest.mark.parametrize("markdown", [True, False])
def test_benchmark_sample_int(markdown: bool):
    benchmark_sample_int(turbo=True, markdown=markdown)
