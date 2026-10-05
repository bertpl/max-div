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
    DiversityMetric.gpq_separation(0.25),
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


def test_labels_are_distinct_and_start_with_the_factory_name():
    """Each metric's label is distinct and starts with its factory method's name in upper case."""
    # --- act --------------------------
    labels = [metric.label for metric in _FACTORY_METRICS]
    factory_names = [repr(metric).removeprefix("DiversityMetric.").split("(")[0] for metric in _FACTORY_METRICS]

    # --- assert -----------------------
    assert len(set(labels)) == len(labels)
    assert all(label.startswith(name.upper()) for label, name in zip(labels, factory_names, strict=True))
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
    """A metric needing the approximate geomean tie-breaker keeps its score when a non-smallest separation grows.

    The selection is large, because for `gpq_separation` only the float32 rounding of the largest
    separations' weights makes the score ignore them.
    """
    # --- arrange ----------------------
    before = np.linspace(0.1, 1.0, 10_000, dtype=np.float32)
    after = before.copy()
    after[-1] *= 2

    # --- act / assert -----------------
    assert metric.compute(before) == metric.compute(after)


# ==================================================================================================
#  gpq_separation
# ==================================================================================================
def _gpq_reference(separations: np.ndarray, q: float) -> float:
    """Return the geometric pseudo-quantile in float64, weighting the separations in descending order."""
    descending = np.sort(separations.astype(np.float64))[::-1]
    rank_fractions = (np.arange(descending.size) + 0.5) / descending.size
    weights = rank_fractions ** (1.0 / q - 2.0)
    return float(np.exp(np.sum(weights * np.log(descending)) / np.sum(weights)))


@pytest.mark.parametrize(
    "q, expected",
    [
        (0, DiversityMetric.min_separation()),
        (0.0, DiversityMetric.min_separation()),
        (0.5, DiversityMetric.geomean_separation()),
    ],
)
def test_gpq_separation_returns_the_metric_that_it_equals_at_its_end_points(q: float, expected: DiversityMetric):
    """At `q = 0` and `q = 0.5` the factory returns the minimum and the geometric mean separation."""
    # --- act / assert -----------------
    assert DiversityMetric.gpq_separation(q) == expected


@pytest.mark.parametrize("q", [-0.1, 0.6, 1.0, float("nan"), True, "0.25", None])
def test_gpq_separation_rejects_a_q_outside_0_to_half(q: object):
    """A q that is not a number between 0 and 0.5 is refused."""
    # --- act / assert -----------------
    with pytest.raises(ValueError, match=r"0 <= q <= 0\.5"):
        DiversityMetric.gpq_separation(q)


def test_gpq_separation_label_and_repr_name_q():
    """The label and the repr carry q, and a numpy q is stored as a plain float."""
    # --- act --------------------------
    metric = DiversityMetric.gpq_separation(np.float64(0.25))

    # --- assert -----------------------
    assert metric.label == "GPQ_SEPARATION(q=0.25)"
    assert repr(metric) == "DiversityMetric.gpq_separation(q=0.25)"
    assert metric == DiversityMetric.gpq_separation(0.25)


@pytest.mark.parametrize("q", [0.001, 0.01, 0.1, 0.25, 0.4, 0.499])
@pytest.mark.parametrize("k", [2, 3, 100, 1000])
def test_gpq_separation_matches_a_float64_reference(q: float, k: int):
    """The score equals the geometric pseudo-quantile computed in float64, up to float32 rounding."""
    # --- arrange ----------------------
    separations = (np.random.default_rng(k).random(k) + 0.01).astype(np.float32)

    # --- act --------------------------
    score = DiversityMetric.gpq_separation(q).compute(separations)

    # --- assert -----------------------
    assert score == pytest.approx(_gpq_reference(separations, q), rel=1e-5)


def test_gpq_separation_lies_between_the_minimum_and_the_geometric_mean_and_rises_with_q():
    """The score lies between the minimum and the geometric mean separation, and does not fall as q rises."""
    # --- arrange ----------------------
    separations = (np.random.default_rng(0).random(200) + 0.01).astype(np.float32)
    qs = [0.001, 0.01, 0.1, 0.25, 0.4, 0.499]

    # --- act --------------------------
    scores = [DiversityMetric.gpq_separation(q).compute(separations) for q in qs]

    # --- assert -----------------------
    assert scores == sorted(scores)
    assert DiversityMetric.min_separation().compute(separations) <= scores[0]
    assert scores[-1] <= DiversityMetric.geomean_separation().compute(separations)


def test_gpq_separation_at_a_tiny_q_is_the_minimum():
    """At a q so small that unscaled weights would all round to 0, the score is the minimum separation."""
    # --- arrange ----------------------
    separations = (np.random.default_rng(1).random(100) + 0.01).astype(np.float32)

    # --- act --------------------------
    score = DiversityMetric.gpq_separation(1e-9).compute(separations)

    # --- assert -----------------------
    assert score == pytest.approx(np.min(separations), rel=1e-6)
