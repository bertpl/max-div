import numpy as np
import pytest

from max_div.solver._diversity import DiversityMetric


@pytest.mark.parametrize(
    "metric, separation, expected_result, tol",
    [
        (DiversityMetric.MIN_SEPARATION, [], 0.0, 1e-6),
        (DiversityMetric.MEAN_SEPARATION, [], 0.0, 1e-6),
        (DiversityMetric.GEOMEAN_SEPARATION, [], 0.0, 1e-6),
        (DiversityMetric.APPROX_GEOMEAN_SEPARATION, [], 0.0, 1e-6),
        (DiversityMetric.MIN_SEPARATION, [0.1, 0.4], 0.1, 1e-6),
        (DiversityMetric.MEAN_SEPARATION, [0.1, 0.4], 0.25, 1e-6),
        (DiversityMetric.GEOMEAN_SEPARATION, [0.1, 0.4], 0.2, 1e-6),
        (DiversityMetric.GEOMEAN_SEPARATION, [0.1, 0.0], 0.0, 1e-6),
        (DiversityMetric.APPROX_GEOMEAN_SEPARATION, [0.1, 0.4], 0.2, 0.01),
        (DiversityMetric.APPROX_GEOMEAN_SEPARATION, [0.1, 0.0], 0.0, 1e-6),
        (DiversityMetric.NON_ZERO_SEPARATION_FRAC, [0.1, 0.4], 1.0, 1e-6),
        (DiversityMetric.NON_ZERO_SEPARATION_FRAC, [0.1, 0.0], 0.5, 1e-6),
        (DiversityMetric.NON_ZERO_SEPARATION_FRAC, [0.0, 0.0], 0.0, 1e-6),
    ],
)
def test_diversity_compute(metric: DiversityMetric, separation: list[float], expected_result: float, tol: float):
    # --- arrange -----------------------------------------
    separation = np.array(separation, dtype=np.float32)

    # --- act ---------------------------------------------
    result = metric.compute(separation)

    # --- assert ------------------------------------------
    assert result == pytest.approx(expected_result, abs=tol, rel=tol)


def test_diversity_metric_unique_names():
    # --- arrange -----------------------------------------
    metrics = list(DiversityMetric)

    # --- act ---------------------------------------------
    all_metric_names = [metric.name for metric in metrics]

    # --- assert ------------------------------------------
    assert len(set(all_metric_names)) == len(metrics)


def test_diversity_metric_unique_values():
    # --- arrange -----------------------------------------
    metrics = list(DiversityMetric)

    # --- act ---------------------------------------------
    all_metric_values = [metric.value for metric in metrics]

    # --- assert ------------------------------------------
    assert len(set(all_metric_values)) == len(metrics)


def test_diversity_metric_equals():
    # --- arrange -----------------------------------------
    metrics = list(DiversityMetric)

    # --- act & assert ------------------------------------
    for i, metric_1 in enumerate(metrics):
        for j, metric_2 in enumerate(metrics):
            if i == j:
                assert metric_1 == metric_2
            else:
                assert metric_1 != metric_2


@pytest.mark.parametrize("metric", list(DiversityMetric))
@pytest.mark.parametrize(
    "sep_array",
    [
        np.zeros(0, dtype=np.float32),
        np.zeros(1, dtype=np.float32),
        np.ones(1, dtype=np.float32),
        np.array([np.inf], dtype=np.float32),
    ],
)
def test_diversity_metric_small_arrays(metric: DiversityMetric, sep_array: np.ndarray):
    """check if all metrics report 0.0 for small arrays; we can only compute diversity meaningfully for size >=2."""

    # --- act ---------------------------------------------
    result = metric.compute(sep_array)

    # --- assert ------------------------------------------
    assert result == 0.0
