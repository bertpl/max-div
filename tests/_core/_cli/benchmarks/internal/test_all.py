import pytest

from max_div._core._cli import (
    benchmark_diversity_metrics,
    benchmark_modify_p_selectivity,
    benchmark_randint,
    benchmark_randint_constrained,
)
from max_div._core._cli.benchmarks.internal.diversity_metrics import SEPARATION_FAMILY_METRICS
from max_div._core.metrics import DiversityContributionFamily, DiversityMetric


@pytest.mark.parametrize("markdown", [True, False])
def test_benchmark_randint(markdown: bool):
    benchmark_randint(speed=1.0, markdown=markdown)


@pytest.mark.parametrize("markdown", [True, False])
def test_benchmark_randint_constrained(markdown: bool):
    benchmark_randint_constrained(speed=1.0, markdown=markdown)


@pytest.mark.parametrize("markdown", [True, False])
def test_benchmark_diversity_metrics(markdown: bool):
    benchmark_diversity_metrics(speed=1.0, markdown=markdown)


def test_benchmark_diversity_metrics_covers_the_separation_family():
    """The benchmarked metrics are exactly the separation-family diversity metrics."""
    # --- act / assert -----------------
    assert {type(metric) for metric in SEPARATION_FAMILY_METRICS} == {
        cls
        for cls in DiversityMetric.__subclasses__()
        if cls.contribution_family == DiversityContributionFamily.SEPARATION
    }


@pytest.mark.parametrize("markdown", [True, False])
def test_benchmark_modify_p_selectivity(markdown: bool):
    benchmark_modify_p_selectivity(speed=1.0, markdown=markdown)
