from max_div._core.metrics import DistanceMetric

# Every metric the factory methods can produce, mirroring the `metric` fixture's parameter set.
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


def test_factory_metrics_carry_no_p(metric: DistanceMetric):
    """No factory-produced metric uses the power parameter."""
    # --- act / assert -----------------
    assert metric.p is None


def test_equal_factories_compare_equal():
    """Two calls of the same factory yield equal, interchangeable values."""
    # --- act / assert -----------------
    assert DistanceMetric.l2_euclidean() == DistanceMetric.l2_euclidean()
    assert DistanceMetric.l2_euclidean() != DistanceMetric.l2s_euclidean_squared()


def test_repr_names_the_factory(metric: DistanceMetric):
    """The repr is the factory call that constructs the metric."""
    # --- act --------------------------
    text = repr(metric)

    # --- assert -----------------------
    assert text.startswith("DistanceMetric.")
    assert text.endswith("()")
    assert eval(text) == metric  # noqa: S307 -- round-trip of our own repr
