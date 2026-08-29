import math

import pytest

from max_div._core.metrics import DistanceMetric

# Every metric with a dedicated factory method of its own.
_FACTORY_METRICS = (
    DistanceMetric.l1_manhattan(),
    DistanceMetric.l2_euclidean(),
    DistanceMetric.l2s_euclidean_squared(),
    DistanceMetric.linf_chebyshev(),
    DistanceMetric.cosine(),
)


def test_factory_metrics_have_distinct_kinds():
    """Each factory must map to its own selector value, or two metrics would dispatch identically."""
    # --- act / assert -----------------
    kinds = [metric.kind for metric in _FACTORY_METRICS]
    assert len(set(kinds)) == len(kinds)


@pytest.mark.parametrize("factory_metric", _FACTORY_METRICS, ids=repr)
def test_factory_metrics_carry_no_p(factory_metric: DistanceMetric):
    """None of the dedicated factories uses the power parameter."""
    # --- act / assert -----------------
    assert factory_metric.p is None


def test_equal_factories_compare_equal():
    """Two calls of the same factory yield equal, interchangeable values."""
    # --- act / assert -----------------
    assert DistanceMetric.l2_euclidean() == DistanceMetric.l2_euclidean()
    assert DistanceMetric.l2_euclidean() != DistanceMetric.l2s_euclidean_squared()


def test_repr_round_trips(metric: DistanceMetric):
    """The repr is a factory call that reconstructs an equal metric."""
    # --- act --------------------------
    text = repr(metric)

    # --- assert -----------------------
    assert text.startswith("DistanceMetric.")
    assert eval(text) == metric  # noqa: S307 -- round-trip of our own repr


@pytest.mark.parametrize(
    "p, root, expected",
    [
        (1.0, True, DistanceMetric.l1_manhattan()),
        (1.0, False, DistanceMetric.l1_manhattan()),
        (2.0, True, DistanceMetric.l2_euclidean()),
        (2.0, False, DistanceMetric.l2s_euclidean_squared()),
        (math.inf, True, DistanceMetric.linf_chebyshev()),
        (math.inf, False, DistanceMetric.linf_chebyshev()),
    ],
)
def test_minkowski_canonicalizes_onto_named_metrics(p: float, root: bool, expected: DistanceMetric):
    """A p coinciding with a dedicated metric must return that metric, never a generic Minkowski value."""
    # --- act / assert -----------------
    assert DistanceMetric.minkowski(p, root=root) == expected


@pytest.mark.parametrize("p", [0.5, 0.25])
@pytest.mark.parametrize("root", [True, False])
def test_minkowski_canonicalizes_specializable_p(p: float, root: bool):
    """A specializable p must return its dedicated kind with p=None, so only one code path computes it."""
    # --- act --------------------------
    metric = DistanceMetric.minkowski(p, root=root)

    # --- assert -----------------------
    assert metric.p is None
    assert metric.kind not in {m.kind for m in _FACTORY_METRICS}
    assert metric.kind != DistanceMetric.minkowski(3, root=root).kind


def test_minkowski_generic_carries_p():
    """A non-specializable p stays on the generic kinds, carried in the value."""
    # --- act --------------------------
    rooted = DistanceMetric.minkowski(3)
    powered = DistanceMetric.minkowski(3, root=False)

    # --- assert -----------------------
    assert rooted.p == 3.0
    assert powered.p == 3.0
    assert rooted.kind != powered.kind


@pytest.mark.parametrize("p", [0.0, -1.0, -math.inf, math.nan])
def test_minkowski_rejects_non_positive_p(p: float):
    """The factory must reject p values outside (0, inf]."""
    # --- act / assert -----------------
    with pytest.raises(ValueError, match="requires p > 0"):
        DistanceMetric.minkowski(p)


def test_njit_p_encodes_none_as_nan():
    """`njit_p` encodes p=None as NaN and a set p as its float64 value."""
    # --- act / assert -----------------
    assert math.isnan(DistanceMetric.l2_euclidean().njit_p)
    assert DistanceMetric.minkowski(3).njit_p == 3.0
