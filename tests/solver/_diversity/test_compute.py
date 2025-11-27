import numpy as np
import pytest

from max_div.solver._diversity import DiversityMetric, compute_diversity


@pytest.mark.parametrize(
    "metric, expected_result",
    [
        (DiversityMetric.MIN_SEPARATION, 0.1),
        (DiversityMetric.MEAN_SEPARATION, 0.25),
        (DiversityMetric.GEOMEAN_SEPARATION, 0.2),
    ],
)
def test_compute_diversity(metric: DiversityMetric, expected_result: float):
    # --- arrange -----------------------------------------
    separation = np.array([0.1, 0.4], dtype=np.float32)

    # --- act ---------------------------------------------
    result = compute_diversity(separation, metric)

    # --- assert ------------------------------------------
    assert result == pytest.approx(expected_result)
