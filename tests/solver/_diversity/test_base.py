import numpy as np
import pytest

from max_div.solver._diversity import DiversityMetric


@pytest.mark.parametrize(
    "metric, separation, expected_result, tol",
    [
        (DiversityMetric.min_separation(), [], 0.0, 1e-6),
        (DiversityMetric.mean_separation(), [], 0.0, 1e-6),
        (DiversityMetric.geomean_separation(), [], 0.0, 1e-6),
        (DiversityMetric.approx_geomean_separation(), [], 0.0, 1e-6),
        (DiversityMetric.min_separation(), [0.1, 0.4], 0.1, 1e-6),
        (DiversityMetric.mean_separation(), [0.1, 0.4], 0.25, 1e-6),
        (DiversityMetric.geomean_separation(), [0.1, 0.4], 0.2, 1e-6),
        (DiversityMetric.geomean_separation(), [0.1, 0.0], 0.0, 1e-6),
        (DiversityMetric.approx_geomean_separation(), [0.1, 0.4], 0.2, 0.01),
        (DiversityMetric.approx_geomean_separation(), [0.1, 0.0], 0.0, 1e-6),
        (DiversityMetric.non_zero_separation_frac(), [0.1, 0.4], 1.0, 1e-6),
        (DiversityMetric.non_zero_separation_frac(), [0.1, 0.0], 0.5, 1e-6),
        (DiversityMetric.non_zero_separation_frac(), [0.0, 0.0], 0.0, 1e-6),
    ],
)
def test_diversity_compute(metric: DiversityMetric, separation: list[float], expected_result: float, tol: float):
    # --- arrange -----------------------------------------
    separation = np.array(separation, dtype=np.float32)

    # --- act ---------------------------------------------
    result = metric.compute(separation)

    # --- assert ------------------------------------------
    assert result == pytest.approx(expected_result, abs=tol, rel=tol)


def test_diversity_metric_equals():
    metric1 = DiversityMetric.min_separation()
    metric2 = DiversityMetric.min_separation()
    metric3 = DiversityMetric.mean_separation()

    assert metric1 == metric2
    assert metric1 != metric3
