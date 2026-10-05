import pickle

import numpy as np
import pytest

from max_div._core.metrics import DiversityContributionFamily, DiversityMetric

# The tuple holds every diversity metric, one per factory method.
_FACTORY_METRICS = (
    DiversityMetric.min_separation(),
    DiversityMetric.mean_separation(),
    DiversityMetric.geomean_separation(),
    DiversityMetric.approx_geomean_separation(),
    DiversityMetric.harmonic_mean_separation(),
    DiversityMetric.non_zero_separation_frac(),
    DiversityMetric.mean_pairwise_distance(),
)


# ==================================================================================================
#  Factories and subclasses
# ==================================================================================================
def test_factory_metrics_cover_every_subclass_once():
    """Each factory returns its own subclass, and together they cover every subclass."""
    # --- act --------------------------
    classes = [type(metric) for metric in _FACTORY_METRICS]

    # --- assert -----------------------
    assert len(set(classes)) == len(classes)
    assert set(classes) == set(DiversityMetric.__subclasses__())


def test_a_bare_diversity_metric_cannot_be_created():
    """The base class computes no score, so constructing it directly is refused."""
    # --- act / assert -----------------
    with pytest.raises(TypeError, match="factory methods"):
        DiversityMetric()


def test_metrics_from_the_same_factory_compare_equal():
    """Calls of the same factory yield equal metrics, and calls of different factories yield unequal ones."""
    # --- act / assert -----------------
    assert DiversityMetric.min_separation() == DiversityMetric.min_separation()
    assert DiversityMetric.min_separation() != DiversityMetric.mean_separation()


def test_metrics_are_usable_as_dict_keys():
    """Every metric that a factory method returns is hashable, and distinct metrics occupy distinct dict entries."""
    # --- act --------------------------
    by_metric = {metric: index for index, metric in enumerate(_FACTORY_METRICS)}

    # --- assert -----------------------
    assert len(by_metric) == len(_FACTORY_METRICS)
    assert by_metric[DiversityMetric.geomean_separation()] == _FACTORY_METRICS.index(
        DiversityMetric.geomean_separation()
    )


@pytest.mark.parametrize("metric", _FACTORY_METRICS, ids=repr)
def test_repr_round_trips(metric: DiversityMetric):
    """The repr is a factory call that reconstructs an equal metric."""
    # --- act --------------------------
    text = repr(metric)

    # --- assert -----------------------
    assert text.startswith("DiversityMetric.")
    assert eval(text) == metric  # noqa: S307 -- round-trip of our own repr


@pytest.mark.parametrize("metric", _FACTORY_METRICS, ids=repr)
def test_pickle_round_trips(metric: DiversityMetric):
    """A metric survives pickling, as it must to reach the workers of a parallel solve."""
    # --- act / assert -----------------
    assert pickle.loads(pickle.dumps(metric)) == metric  # noqa: S301 -- round-trip of our own object


def test_labels_are_distinct_upper_case_names():
    """Each metric's label is a distinct upper-case name, e.g. `MIN_SEPARATION` for `min_separation()`."""
    # --- act --------------------------
    labels = [metric.label for metric in _FACTORY_METRICS]

    # --- assert -----------------------
    assert len(set(labels)) == len(labels)
    assert all(label == label.upper() for label in labels)
    assert DiversityMetric.min_separation().label == "MIN_SEPARATION"


# ==================================================================================================
#  Compute
# ==================================================================================================
@pytest.mark.parametrize(
    "metric, separation, expected_result, tol",
    [
        (DiversityMetric.min_separation(), [0.1, 0.4], 0.1, 1e-6),
        (DiversityMetric.mean_separation(), [0.1, 0.4], 0.25, 1e-6),
        (DiversityMetric.geomean_separation(), [0.1, 0.4], 0.2, 1e-6),
        (DiversityMetric.geomean_separation(), [0.1, 0.0], 0.0, 1e-6),
        (DiversityMetric.approx_geomean_separation(), [0.1, 0.4], 0.2, 0.01),
        (DiversityMetric.approx_geomean_separation(), [0.1, 0.0], 0.0, 1e-6),
        (DiversityMetric.harmonic_mean_separation(), [0.1, 0.4], 0.16, 1e-6),
        (DiversityMetric.harmonic_mean_separation(), [2.0, 2.0, 2.0], 2.0, 1e-6),
        (DiversityMetric.harmonic_mean_separation(), [0.1, 0.0], 0.0, 1e-6),
        (DiversityMetric.non_zero_separation_frac(), [0.1, 0.4], 1.0, 1e-6),
        (DiversityMetric.non_zero_separation_frac(), [0.1, 0.0], 0.5, 1e-6),
        (DiversityMetric.non_zero_separation_frac(), [0.0, 0.0], 0.0, 1e-6),
        (DiversityMetric.mean_pairwise_distance(), [0.1, 0.4], 0.25, 1e-6),
        (DiversityMetric.mean_pairwise_distance(), [2.0, 3.0, 4.0], 3.0, 1e-6),
    ],
)
def test_compute(metric: DiversityMetric, separation: list[float], expected_result: float, tol: float):
    """Each metric reduces the given contribution values to the expected score."""
    # --- arrange ----------------------
    separation = np.array(separation, dtype=np.float32)

    # --- act --------------------------
    result = metric.compute(separation)

    # --- assert -----------------------
    assert result == pytest.approx(expected_result, abs=tol, rel=tol)


@pytest.mark.parametrize("metric", _FACTORY_METRICS, ids=repr)
@pytest.mark.parametrize(
    "sep_array",
    [
        np.zeros(0, dtype=np.float32),
        np.zeros(1, dtype=np.float32),
        np.ones(1, dtype=np.float32),
        np.array([np.inf], dtype=np.float32),
    ],
)
def test_compute_scores_fewer_than_2_values_as_zero(metric: DiversityMetric, sep_array: np.ndarray):
    """Every metric reports 0.0 below 2 values; a diversity score needs at least one pair."""
    # --- act --------------------------
    result = metric.compute(sep_array)

    # --- assert -----------------------
    assert result == 0.0


# ==================================================================================================
#  Per-metric class variables
# ==================================================================================================
@pytest.mark.parametrize("metric", _FACTORY_METRICS, ids=repr)
def test_contribution_family(metric: DiversityMetric):
    """Each metric maps to the contribution family that its name implies."""
    # --- arrange ----------------------
    if metric == DiversityMetric.mean_pairwise_distance():
        expected = DiversityContributionFamily.MEAN_DISTANCE
    else:
        expected = DiversityContributionFamily.SEPARATION

    # --- act / assert -----------------
    assert metric.contribution_family == expected


@pytest.mark.parametrize("metric", _FACTORY_METRICS, ids=repr)
def test_needs_non_zero_separation_frac_tie_breaker_matches_the_score(metric: DiversityMetric):
    """A metric needs the non-zero-fraction tie-breaker if and only if one zero separation makes it score near 0."""
    # --- arrange ----------------------
    contributions = np.array([0.0, 0.5, 1.0], dtype=np.float32)

    # --- act --------------------------
    score = metric.compute(contributions)

    # --- assert -----------------------
    # the approximate geomean gives a value near zero, not exactly zero
    assert (score < 1e-6) == metric.needs_non_zero_separation_frac_tie_breaker


@pytest.mark.parametrize(
    "metric", [metric for metric in _FACTORY_METRICS if metric.needs_approx_geomean_tie_breaker], ids=repr
)
def test_a_metric_that_needs_the_approx_geomean_tie_breaker_ignores_a_larger_separation(metric: DiversityMetric):
    """A metric needing the approximate geomean tie-breaker keeps its score when a non-smallest separation grows."""
    # --- arrange ----------------------
    before = np.array([0.1, 0.5, 1.0], dtype=np.float32)
    after = np.array([0.1, 0.5, 2.0], dtype=np.float32)

    # --- act / assert -----------------
    assert metric.compute(before) == metric.compute(after)
