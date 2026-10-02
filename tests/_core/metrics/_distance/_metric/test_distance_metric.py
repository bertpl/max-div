import math
import pickle

import numpy as np
import pytest

from max_div._core.metrics import DistanceMetric
from max_div._core.metrics._distance._metric import NO_FLOAT_PARAM, validate_vector_array_layout

# Every metric with a dedicated factory method of its own.
_FACTORY_METRICS = (
    DistanceMetric.l1_manhattan(),
    DistanceMetric.l2_euclidean(),
    DistanceMetric.l2s_euclidean_squared(),
    DistanceMetric.linf_chebyshev(),
    DistanceMetric.cosine(),
    DistanceMetric.geometric_mean(),
    DistanceMetric.l_minus_inf(),
    DistanceMetric.along_axis(0),
    DistanceMetric.l2_and_projections(),
)


def _all_subclasses(cls: type) -> set[type]:
    """Return every direct and indirect subclass of `cls`."""
    return {subclass for direct in cls.__subclasses__() for subclass in (direct, *_all_subclasses(direct))}


# ==================================================================================================
#  Factories and kinds
# ==================================================================================================
def test_factory_metrics_have_distinct_kinds():
    """Each factory must map to its own selector value, or two metrics would dispatch identically."""
    # --- act / assert -----------------
    kinds = [metric.kind for metric in _FACTORY_METRICS]
    assert len(set(kinds)) == len(kinds)


def test_factory_metrics_cover_every_subclass_once():
    """Each factory returns its own subclass; with the Minkowski and the k-items subclass, they cover every subclass."""
    # --- act --------------------------
    classes = [type(metric) for metric in _FACTORY_METRICS]
    other_forms = {type(DistanceMetric.minkowski(3)), type(DistanceMetric.l2_and_projections(k=3))}

    # --- assert -----------------------
    assert len(set(classes)) == len(classes)
    assert set(classes) | other_forms == _all_subclasses(DistanceMetric)


def test_a_bare_distance_metric_cannot_be_created():
    """The base class computes no distance, so constructing it directly is refused."""
    # --- act / assert -----------------
    with pytest.raises(TypeError, match="factory methods"):
        DistanceMetric()


@pytest.mark.parametrize(
    "factory_metric",
    [metric for metric in _FACTORY_METRICS if metric != DistanceMetric.l2_and_projections()],
    ids=repr,
)
def test_factory_metrics_without_a_float_parameter_have_no_float_param(factory_metric: DistanceMetric):
    """Every dedicated factory but `l2_and_projections` returns a metric whose float parameter is `NO_FLOAT_PARAM`."""
    # --- act / assert -----------------
    assert factory_metric.float_param(3) == NO_FLOAT_PARAM


def test_pairwise_distance_args_are_the_kind_and_float_param_as_numpy_scalars():
    """`pairwise_distance_args` returns the kind as np.int32 and the parameter as np.float64."""
    # --- act --------------------------
    kind, float_param = DistanceMetric.minkowski(3).pairwise_distance_args(np.zeros((4, 5), dtype=np.float32))

    # --- assert -----------------------
    assert (kind, float_param) == (DistanceMetric.minkowski(3).kind, 3.0)
    assert isinstance(kind, np.int32)
    assert isinstance(float_param, np.float64)


def test_equal_factories_compare_equal():
    """Two calls of the same factory yield equal, interchangeable values."""
    # --- act / assert -----------------
    assert DistanceMetric.l2_euclidean() == DistanceMetric.l2_euclidean()
    assert DistanceMetric.l2_euclidean() != DistanceMetric.l2s_euclidean_squared()


def test_metrics_are_usable_as_dict_keys():
    """Every factory metric is hashable, and distinct metrics occupy distinct dict entries."""
    # --- act --------------------------
    by_metric = {metric: index for index, metric in enumerate(_FACTORY_METRICS)}

    # --- assert -----------------------
    assert len(by_metric) == len(_FACTORY_METRICS)
    assert by_metric[DistanceMetric.cosine()] == _FACTORY_METRICS.index(DistanceMetric.cosine())


def test_repr_round_trips(metric: DistanceMetric):
    """The repr is a factory call that reconstructs an equal metric."""
    # --- act --------------------------
    text = repr(metric)

    # --- assert -----------------------
    assert text.startswith("DistanceMetric.")
    assert eval(text) == metric  # noqa: S307 -- round-trip of our own repr


def test_a_metric_survives_pickling(metric: DistanceMetric):
    """A metric round-trips through pickle unchanged, as it must to reach a worker inside a shared store spec."""
    # --- act / assert -----------------
    assert pickle.loads(pickle.dumps(metric)) == metric  # noqa: S301 -- round-trip of our own object


@pytest.mark.parametrize(
    "metric, expected",
    [
        (DistanceMetric.l1_manhattan(), "L1"),
        (DistanceMetric.l2_euclidean(), "L2"),
        (DistanceMetric.l2s_euclidean_squared(), "L2²"),
        (DistanceMetric.linf_chebyshev(), "L∞"),
        (DistanceMetric.cosine(), "cosine"),
        (DistanceMetric.geometric_mean(), "geomean"),
        (DistanceMetric.l_minus_inf(), "L-∞"),
        (DistanceMetric.along_axis(2), "axis 2"),
        (DistanceMetric.l2_and_projections(), "L2+projections"),
        (DistanceMetric.l2_and_projections(l2_scale=0.5), "L2+projections (L2 scale 0.5)"),
        (DistanceMetric.l2_and_projections(k=100), "L2+projections (k=100)"),
        (DistanceMetric.l2_and_projections(l2_scale=0.5, k=100), "L2+projections (L2 scale 0.5, k=100)"),
        (DistanceMetric.minkowski(3), "L3"),
        (DistanceMetric.minkowski(3, root=False), "L3-powered"),
        (DistanceMetric.minkowski(0.5), "L0.5"),
    ],
)
def test_label_is_a_short_name_per_metric(metric: DistanceMetric, expected: str):
    """Each metric has a concise label, reflecting the axis or exponent where it has one."""
    # --- act / assert -----------------
    assert metric.label == expected


def test_only_cosine_and_along_axis_need_preprocessed_vectors(metric: DistanceMetric):
    """Cosine and the along-axis distance are the metrics that read a preprocessed copy of the vectors."""
    # --- act / assert -----------------
    expected = metric == DistanceMetric.cosine() or metric.kind == DistanceMetric.along_axis(0).kind
    assert metric.needs_preprocessed_vectors is expected


# ==================================================================================================
#  Minkowski
# ==================================================================================================
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


@pytest.mark.parametrize("p", [0.5, 0.25, 0.125])
@pytest.mark.parametrize("root", [True, False])
def test_minkowski_canonicalizes_specializable_p(p: float, root: bool):
    """A p of 0.5, 0.25 or 0.125 gets its own specialized kind, whose pairwise distance function takes no parameter."""
    # --- act --------------------------
    metric = DistanceMetric.minkowski(p, root=root)

    # --- assert -----------------------
    assert metric.float_param(3) == NO_FLOAT_PARAM
    assert metric.kind not in {m.kind for m in _FACTORY_METRICS}
    assert metric.kind != DistanceMetric.minkowski(3, root=root).kind
    assert metric.p == p


def test_minkowski_generic_carries_p():
    """A non-specializable p stays on the generic kinds, and its pairwise distance function takes p."""
    # --- act --------------------------
    rooted = DistanceMetric.minkowski(3)
    powered = DistanceMetric.minkowski(3, root=False)

    # --- assert -----------------------
    assert rooted.float_param(3) == 3.0
    assert powered.float_param(3) == 3.0
    assert rooted.kind != powered.kind
    assert rooted != powered


@pytest.mark.parametrize("p", [0.0, -1.0, -math.inf, math.nan])
def test_minkowski_rejects_non_positive_p(p: float):
    """The factory must reject p values outside (0, inf]."""
    # --- act / assert -----------------
    with pytest.raises(ValueError, match="requires p > 0"):
        DistanceMetric.minkowski(p)


# ==================================================================================================
#  Along one axis
# ==================================================================================================
def test_along_axis_carries_the_axis():
    """The along-axis metric stores its coordinate as a plain int, and metrics over different axes differ."""
    # --- act / assert -----------------
    assert DistanceMetric.along_axis(3).axis == 3
    assert DistanceMetric.along_axis(np.int64(2)).axis == 2
    assert DistanceMetric.along_axis(0) == DistanceMetric.along_axis(0)
    assert DistanceMetric.along_axis(0) != DistanceMetric.along_axis(1)


@pytest.mark.parametrize("axis", [-1, 1.5, "0", True, None])
def test_along_axis_rejects_anything_but_a_non_negative_integer(axis):
    """The factory accepts a zero-based coordinate index and nothing else."""
    # --- act / assert -----------------
    with pytest.raises(ValueError, match="non-negative integer axis"):
        DistanceMetric.along_axis(axis)


# ==================================================================================================
#  L2 and projections
# ==================================================================================================
def test_l2_and_projections_carries_its_l2_scale():
    """The L2-and-projections metric stores its L2 scale, 1 by default, as its float parameter."""
    # --- act / assert -----------------
    assert DistanceMetric.l2_and_projections().l2_scale == 1.0
    assert DistanceMetric.l2_and_projections(l2_scale=2).float_param(3) == 2.0
    assert DistanceMetric.l2_and_projections(l2_scale=0.5) != DistanceMetric.l2_and_projections()


@pytest.mark.parametrize("l2_scale", [0.0, -1.0, math.inf, math.nan])
def test_l2_and_projections_rejects_an_l2_scale_that_is_not_positive_and_finite(l2_scale: float):
    """The L2 scale must be a positive, finite number."""
    # --- act / assert -----------------
    with pytest.raises(ValueError, match="positive, finite l2_scale"):
        DistanceMetric.l2_and_projections(l2_scale=l2_scale)


def test_l2_and_projections_with_k_is_a_kind_of_its_own_that_stores_k():
    """With `k` the metric is a kind of its own that stores `k`, as a plain int, beside the L2 scale."""
    # --- act --------------------------
    with_k = DistanceMetric.l2_and_projections(k=100)

    # --- assert -----------------------
    assert with_k.kind != DistanceMetric.l2_and_projections().kind
    assert (with_k.l2_scale, with_k.k) == (1.0, 100)
    assert type(DistanceMetric.l2_and_projections(k=np.int64(100)).k) is int
    assert with_k != DistanceMetric.l2_and_projections(k=50)
    assert with_k != DistanceMetric.l2_and_projections()


@pytest.mark.parametrize("k", [1, 0, -5, 1.5, "100", True])
def test_l2_and_projections_rejects_a_k_that_is_not_an_integer_of_at_least_2(k):
    """`k` must be an integer of at least 2, because the factor on the L2 distance divides by k - 1."""
    # --- act / assert -----------------
    with pytest.raises(ValueError, match="integer k >= 2"):
        DistanceMetric.l2_and_projections(k=k)


@pytest.mark.parametrize(
    "metric, n_dims, expected",
    [
        (DistanceMetric.l2_and_projections(k=100), 2, 9 / 99),
        (DistanceMetric.l2_and_projections(k=100), 3, (100 ** (1 / 3) - 1) / 99),
        (DistanceMetric.l2_and_projections(l2_scale=0.5, k=2), 2, 0.5 * (2**0.5 - 1)),
        (DistanceMetric.l2_and_projections(l2_scale=0.5), 2, 0.5),
    ],
    ids=str,
)
def test_l2_and_projections_float_param_combines_k_and_the_dimension_count(
    metric: DistanceMetric, n_dims: int, expected: float
):
    """With `k` the float parameter is the L2 scale times (k^(1/d) - 1) / (k - 1); without `k` it is the L2 scale."""
    # --- act / assert -----------------
    assert metric.float_param(n_dims) == pytest.approx(expected)


# ==================================================================================================
#  Estimated cost of computing a distance
# ==================================================================================================
@pytest.mark.parametrize("n_dims", [1, 2, 100])
def test_every_metric_estimates_a_positive_cost_that_never_falls_with_more_dimensions(
    metric: DistanceMetric, n_dims: int
):
    """Every kind estimates a positive cost per distance, and one dimension more never costs less."""
    # --- act / assert -----------------
    assert metric.estimated_lazy_cost_ns(n_dims) > 0
    assert metric.estimated_lazy_cost_ns(n_dims + 1) >= metric.estimated_lazy_cost_ns(n_dims)


def test_the_cost_estimate_ranks_along_axis_lowest_and_a_generic_minkowski_highest():
    """At 2 dimensions, `along_axis` is the cheapest to compute, a Minkowski with a generic kind the costliest."""
    # --- arrange ----------------------
    along_axis, generic_minkowski = DistanceMetric.along_axis(0), DistanceMetric.minkowski(3)
    others = [m for m in _FACTORY_METRICS if m != along_axis] + [
        DistanceMetric.l2_and_projections(k=100),
        DistanceMetric.minkowski(0.5),
    ]

    # --- act --------------------------
    costs = [m.estimated_lazy_cost_ns(2) for m in others]

    # --- assert -----------------------
    assert along_axis.estimated_lazy_cost_ns(2) < min(costs)
    assert generic_minkowski.estimated_lazy_cost_ns(2) > max(costs)


def test_a_metric_kind_without_its_own_coefficients_gets_the_default_estimate(monkeypatch: pytest.MonkeyPatch):
    """A kind that sets no coefficients of its own, as a new kind would, inherits the default (2.0, 0.2)."""
    # --- arrange ----------------------
    metric = DistanceMetric.l2_euclidean()
    monkeypatch.delattr(type(metric), "_lazy_cost_coefficients_ns")

    # --- act / assert -----------------
    assert metric.estimated_lazy_cost_ns(10) == pytest.approx(2.0 + 0.2 * 10)


def test_a_generic_minkowski_without_root_is_estimated_costlier_than_a_specialized_one():
    """Without the outer root too, a Minkowski with a generic kind is estimated costlier than a specialized one."""
    # --- act / assert -----------------
    assert DistanceMetric.minkowski(3, root=False).estimated_lazy_cost_ns(2) > DistanceMetric.minkowski(
        0.5, root=False
    ).estimated_lazy_cost_ns(2)


# ==================================================================================================
#  Validation against a problem's vectors
# ==================================================================================================
def test_validate_accepts_vectors_every_metric_can_compute_on(metric: DistanceMetric, vectors: np.ndarray):
    """Vectors of 3 dimensions without a zero row pass every metric's checks."""
    # --- act / assert -----------------
    metric.validate(vectors)  # raises on rejection


@pytest.mark.parametrize("check", ["validate", "preprocess"])
def test_cosine_rejects_zero_rows(check: str):
    """A zero row has no direction, so cosine's validation and its preprocessing both refuse it, naming the row."""
    # --- arrange ----------------------
    vectors = np.array([[1, 2], [0, 0], [3, 4]], dtype=np.float32)

    # --- act / assert -----------------
    with pytest.raises(ValueError, match=r"zero vector.*row 1"):
        getattr(DistanceMetric.cosine(), check)(vectors)


@pytest.mark.parametrize("check", ["validate", "preprocess"])
def test_along_axis_rejects_a_missing_coordinate(check: str, vectors: np.ndarray):
    """An axis at or beyond the dimension count is refused by the metric's validation and its preprocessing alike."""
    # --- act / assert -----------------
    with pytest.raises(ValueError, match="do not have"):
        getattr(DistanceMetric.along_axis(3), check)(vectors)


@pytest.mark.parametrize("k", [None, 3])
def test_l2_and_projections_rejects_one_dimension(k: int | None, vectors: np.ndarray):
    """The L2-and-projections distance, with or without `k`, needs at least 2 dimensions."""
    # --- act / assert -----------------
    with pytest.raises(ValueError, match="needs at least 2 dimensions"):
        DistanceMetric.l2_and_projections(k=k).validate(np.ascontiguousarray(vectors[:, :1]))


def test_l2_and_projections_with_k_rejects_a_problem_that_selects_another_k(vectors: np.ndarray):
    """Weighted for k items, the distance accepts its own k or none, and rejects any other."""
    # --- arrange ----------------------
    metric = DistanceMetric.l2_and_projections(k=4)

    # --- act / assert -----------------
    metric.validate(vectors, 4)  # raises on rejection
    metric.validate(vectors)  # raises on rejection
    with pytest.raises(ValueError, match="selects k=3"):
        metric.validate(vectors, 3)


# ==================================================================================================
#  Preprocessing
# ==================================================================================================
def test_preprocess_follows_the_metrics_declaration(metric: DistanceMetric, vectors: np.ndarray):
    """A metric that does not preprocess gets its input back; one that does gets a new array."""
    # --- act --------------------------
    preprocessed = metric.preprocess(vectors)

    # --- assert -----------------------
    if metric.needs_preprocessed_vectors:
        assert not np.shares_memory(preprocessed, vectors)
    else:
        assert preprocessed is vectors


def test_preprocess_leaves_the_input_untouched(metric: DistanceMetric, vectors: np.ndarray):
    """Preprocessing never writes into the user's array, whichever metric asks."""
    # --- arrange ----------------------
    before = vectors.copy()

    # --- act --------------------------
    metric.preprocess(vectors)

    # --- assert -----------------------
    np.testing.assert_array_equal(vectors, before)


def test_preprocess_returns_the_layout_reads_expect(metric: DistanceMetric, vectors: np.ndarray):
    """What comes out is in the form every distance read expects, whether copied or not."""
    # --- act --------------------------
    preprocessed = metric.preprocess(vectors)

    # --- assert -----------------------
    validate_vector_array_layout(preprocessed)  # raises on violation


# ==================================================================================================
#  Cosine
# ==================================================================================================
def test_cosine_preprocessing_normalizes_rows(vectors: np.ndarray):
    """Cosine's preprocessed copy has unit-length rows."""
    # --- act --------------------------
    preprocessed = DistanceMetric.cosine().preprocess(vectors)

    # --- assert -----------------------
    np.testing.assert_allclose(np.linalg.norm(preprocessed, axis=1), 1.0, rtol=1e-6)


# ==================================================================================================
#  Along one axis
# ==================================================================================================
def test_along_axis_preprocessing_keeps_the_one_coordinate(vectors: np.ndarray):
    """The along-axis copy is an (n, 1) array holding exactly the requested coordinate."""
    # --- act --------------------------
    preprocessed = DistanceMetric.along_axis(2).preprocess(vectors)

    # --- assert -----------------------
    assert preprocessed.shape == (vectors.shape[0], 1)
    np.testing.assert_array_equal(preprocessed[:, 0], vectors[:, 2])


def test_along_axis_preprocessing_copies_a_single_column_too(vectors: np.ndarray):
    """Preprocessing copies the sliced coordinate, so a one-dimensional input yields a new array, not a view."""
    # --- arrange ----------------------
    single_column = np.ascontiguousarray(vectors[:, :1])

    # --- act --------------------------
    preprocessed = DistanceMetric.along_axis(0).preprocess(single_column)

    # --- assert -----------------------
    assert not np.shares_memory(preprocessed, single_column)
    np.testing.assert_array_equal(preprocessed, single_column)
