import math
import pickle

import numpy as np
import pytest

from max_div._core.metrics import DistanceMetric
from max_div._core.metrics._distance._metric import NO_PARAM

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
    DistanceMetric.marginals_and_joint(),
)


# ==================================================================================================
#  Factories and kinds
# ==================================================================================================
def test_factory_metrics_have_distinct_kinds():
    """Each factory must map to its own selector value, or two metrics would dispatch identically."""
    # --- act / assert -----------------
    kinds = [metric.kind for metric in _FACTORY_METRICS]
    assert len(set(kinds)) == len(kinds)


def test_factory_metrics_cover_every_subclass_once():
    """Each factory returns an instance of its own `DistanceMetric` subclass, and together they cover every subclass."""
    # --- act --------------------------
    classes = [type(metric) for metric in _FACTORY_METRICS]

    # --- assert -----------------------
    assert len(set(classes)) == len(classes)
    assert set(classes) | {type(DistanceMetric.minkowski(3))} == set(DistanceMetric.__subclasses__())


def test_a_bare_distance_metric_cannot_be_created():
    """The base class computes no distance, so constructing it directly is refused."""
    # --- act / assert -----------------
    with pytest.raises(TypeError, match="factory methods"):
        DistanceMetric()


@pytest.mark.parametrize(
    "factory_metric",
    [metric for metric in _FACTORY_METRICS if metric != DistanceMetric.marginals_and_joint()],
    ids=repr,
)
def test_factory_metrics_without_a_float_parameter_take_no_compiled_param(factory_metric: DistanceMetric):
    """Every dedicated factory but `marginals_and_joint` has no float parameter, so its pair function takes NO_PARAM."""
    # --- act / assert -----------------
    assert factory_metric.compiled_param == NO_PARAM


def test_compiled_args_are_the_kind_and_param_as_numpy_scalars():
    """The compiled arguments pair the kind and the parameter in the types the pair functions are compiled for."""
    # --- act --------------------------
    kind, param = DistanceMetric.minkowski(3).compiled_args

    # --- assert -----------------------
    assert (kind, param) == (DistanceMetric.minkowski(3).kind, 3.0)
    assert isinstance(kind, np.int32)
    assert isinstance(param, np.float64)


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
        (DistanceMetric.marginals_and_joint(), "marginals+joint"),
        (DistanceMetric.marginals_and_joint(joint_scale=0.5), "marginals+joint (joint scale 0.5)"),
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
    """A p of 0.5, 0.25 or 0.125 gets its own specialized kind, whose pair function takes no parameter."""
    # --- act --------------------------
    metric = DistanceMetric.minkowski(p, root=root)

    # --- assert -----------------------
    assert metric.compiled_param == NO_PARAM
    assert metric.kind not in {m.kind for m in _FACTORY_METRICS}
    assert metric.kind != DistanceMetric.minkowski(3, root=root).kind
    assert metric.p == p


def test_minkowski_generic_carries_p():
    """A non-specializable p stays on the generic kinds, and its pair function takes p."""
    # --- act --------------------------
    rooted = DistanceMetric.minkowski(3)
    powered = DistanceMetric.minkowski(3, root=False)

    # --- assert -----------------------
    assert rooted.compiled_param == 3.0
    assert powered.compiled_param == 3.0
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
#  Marginals and joint
# ==================================================================================================
def test_marginals_and_joint_carries_its_joint_scale():
    """The marginals-and-joint metric stores its joint scale, 1 by default, and hands it to its pair function."""
    # --- act / assert -----------------
    assert DistanceMetric.marginals_and_joint().joint_scale == 1.0
    assert DistanceMetric.marginals_and_joint(joint_scale=2).compiled_param == 2.0
    assert DistanceMetric.marginals_and_joint(joint_scale=0.5) != DistanceMetric.marginals_and_joint()


@pytest.mark.parametrize("joint_scale", [0.0, -1.0, math.inf, math.nan])
def test_marginals_and_joint_rejects_a_joint_scale_that_is_not_positive_and_finite(joint_scale: float):
    """The joint scale must be a positive, finite number."""
    # --- act / assert -----------------
    with pytest.raises(ValueError, match="positive, finite joint_scale"):
        DistanceMetric.marginals_and_joint(joint_scale=joint_scale)


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


def test_marginals_and_joint_rejects_one_dimension(vectors: np.ndarray):
    """The marginals-and-joint distance needs at least 2 dimensions."""
    # --- act / assert -----------------
    with pytest.raises(ValueError, match="needs at least 2 dimensions"):
        DistanceMetric.marginals_and_joint().validate(np.ascontiguousarray(vectors[:, :1]))
