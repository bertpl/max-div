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


def test_diversity_metric_unique_names():
    # --- arrange -----------------------------------------
    all_metrics = DiversityMetric.all_metrics()

    # --- act ---------------------------------------------
    all_metric_names = [metric.name for metric in all_metrics]

    # --- assert ------------------------------------------
    assert len(set(all_metric_names)) == len(all_metrics)


def test_diversity_metric_unique_functions():
    # --- arrange -----------------------------------------
    all_metrics = DiversityMetric.all_metrics()

    # --- act ---------------------------------------------
    all_metric_functions = [metric.f for metric in all_metrics]

    # --- assert ------------------------------------------
    assert len(set(all_metric_functions)) == len(all_metrics)


def test_diversity_metric_equals():
    # --- arrange -----------------------------------------
    all_metrics = DiversityMetric.all_metrics()

    # --- act & assert ------------------------------------
    for i, metric_1 in enumerate(all_metrics):
        for j, metric_2 in enumerate(all_metrics):
            if i == j:
                assert metric_1 == metric_2
            else:
                assert metric_1 != metric_2


@pytest.mark.parametrize("metric", DiversityMetric.all_metrics())
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
