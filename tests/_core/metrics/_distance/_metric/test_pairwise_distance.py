import numpy as np
import pytest
from scipy.spatial.distance import pdist as scipy_pdist

from max_div._core.metrics._distance import (
    DistanceMetric,
)
from tests._core.metrics._distance.helpers import condensed_distances, l2_and_projections_reference

_SCIPY_METRIC = {
    DistanceMetric.l1_manhattan(): "cityblock",
    DistanceMetric.l2_euclidean(): "euclidean",
    DistanceMetric.l2s_euclidean_squared(): "sqeuclidean",
    DistanceMetric.linf_chebyshev(): "chebyshev",
    DistanceMetric.cosine(): "cosine",
}


# ==================================================================================================
#  Compute
# ==================================================================================================
def test_pairwise_distance_metrics(metric: DistanceMetric):
    """Every metric computes through the full-matrix build."""

    # --- arrange ----------------------
    # note: no all-zero row — COSINE rejects zero vectors
    vectors = np.array([[2, 2], [3, 4], [1, 0], [0, 1]], dtype=np.float32)

    # --- act --------------------------
    d = condensed_distances(vectors, metric=metric)

    # --- assert -----------------------
    assert d.shape == (6,), "Unexpected number of pairs."
    assert d.dtype == np.float32, "Unexpected dtype of the distances."


@pytest.mark.parametrize(
    "metric, expected_value",
    [
        (DistanceMetric.l1_manhattan(), 7.0),
        (DistanceMetric.l2_euclidean(), 5.0),
        (DistanceMetric.l2s_euclidean_squared(), 25.0),
        (DistanceMetric.linf_chebyshev(), 4.0),
        (DistanceMetric.geometric_mean(), 12.0**0.5),
        (DistanceMetric.l_minus_inf(), 3.0),
        (DistanceMetric.along_axis(1), 4.0),
        (DistanceMetric.l2_and_projections(), 3.0),
        (DistanceMetric.l2_and_projections(k=100), 5.0 * 9 / 99),
    ],
)
def test_pairwise_distance_values(metric: DistanceMetric, expected_value: float):
    """The pairwise distance functions produce the expected values."""

    # --- arrange ----------------------
    vectors = np.array([[0, 0], [3, 4]], dtype=np.float32)

    # --- act --------------------------
    d = condensed_distances(vectors, metric=metric)

    # --- assert -----------------------
    assert d[0] == pytest.approx(expected_value)


@pytest.mark.parametrize("metric", list(_SCIPY_METRIC), ids=repr)
def test_pairwise_distance_matches_scipy(metric: DistanceMetric):
    """The hand-rolled float32 kernel matches scipy's float64→float32 result within float32 tolerance."""

    # --- arrange ----------------------
    rng = np.random.default_rng(20260711)
    vectors = rng.standard_normal((60, 8)).astype(np.float32)
    expected = scipy_pdist(vectors, metric=_SCIPY_METRIC[metric]).astype(np.float32)

    # --- act --------------------------
    result = condensed_distances(vectors, metric=metric)

    # --- assert -----------------------
    assert result.dtype == np.float32
    np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("p", [0.125, 0.25, 0.5, 1.5, 3.0])
@pytest.mark.parametrize("root", [True, False])
def test_pairwise_distance_minkowski_matches_reference(p: float, root: bool):
    """Minkowski distances match a float64 numpy reference, for specialized and generic p."""
    # --- arrange ----------------------
    rng = np.random.default_rng(20260829)
    vectors = rng.standard_normal((40, 6)).astype(np.float32)
    diffs = np.abs(vectors[:, None, :].astype(np.float64) - vectors[None, :, :].astype(np.float64))
    powered = (diffs**p).sum(axis=2)
    expected_matrix = powered ** (1.0 / p) if root else powered
    expected = expected_matrix[np.triu_indices(40, k=1)].astype(np.float32)

    # --- act --------------------------
    result = condensed_distances(vectors, metric=DistanceMetric.minkowski(p, root=root))

    # --- assert -----------------------
    np.testing.assert_allclose(result, expected, rtol=2e-5, atol=2e-6)


@pytest.mark.parametrize(
    "x, y, expected_value",
    [
        ([1, 0], [0, 1], 1.0),  # orthogonal
        ([1, 0], [-1, 0], 2.0),  # opposite
        ([1, 0], [1, 1], 1.0 - 1.0 / np.sqrt(2.0)),  # 45 degrees
        ([1, 0], [100, 0], 0.0),  # parallel: magnitude-invariant
    ],
)
def test_pairwise_distance_cosine_values(x: list[float], y: list[float], expected_value: float):
    """Cosine distance produces the expected angular values."""

    # --- arrange ----------------------
    vectors = np.array([x, y], dtype=np.float32)

    # --- act --------------------------
    d = condensed_distances(vectors, metric=DistanceMetric.cosine())

    # --- assert -----------------------
    assert d[0] == pytest.approx(expected_value, abs=1e-6)


def test_pairwise_distance_cosine_zero_vector_raises():
    """Cosine distance rejects all-zero vectors with a clear error naming the row."""

    # --- arrange ----------------------
    vectors = np.array([[1, 2], [0, 0], [3, 4]], dtype=np.float32)

    # --- act / assert -----------------
    with pytest.raises(ValueError, match=r"zero vector.*row 1"):
        condensed_distances(vectors, metric=DistanceMetric.cosine())


def test_pairwise_distance_zero_for_identical_vectors(metric: DistanceMetric):
    """Identical vectors have exactly-zero distance under every metric."""

    # --- arrange ----------------------
    vectors = np.array([[1.5, -2.0, 3.0], [1.5, -2.0, 3.0], [4.0, 4.0, 4.0]], dtype=np.float32)

    # --- act --------------------------
    result = condensed_distances(vectors, metric=metric)

    # --- assert -----------------------
    assert result[0] == np.float32(0.0)  # distance between the two identical vectors


# ==================================================================================================
#  Geometric mean
# ==================================================================================================
@pytest.mark.parametrize(
    "x, y, expected_value",
    [
        ([0.0, 0.0], [3.0, 4.0], 12.0**0.5),  # sqrt(3 * 4)
        ([1.0, 5.0, 2.0], [1.0, 9.0, 7.0], 0.0),  # a shared coordinate zeroes the product
        ([0.0, 0.0], [-3.0, 4.0], 12.0**0.5),  # differences enter by absolute value
        ([7.0], [3.0], 4.0),  # one dimension: the single gap itself
        ([0.0, 0.0, 0.0], [1.0, 2.0, 4.0], 2.0),  # cube root of 8
        ([0.5, 2.0, 8.0, 0.25, 4.0, 1.0], [0.0] * 6, 8.0 ** (1 / 6)),  # six mixed gaps whose product is 8
        ([1e-18] * 25, [0.0] * 25, 1e-18),  # the product 1e-450 would underflow float64
        ([1e30] * 25, [0.0] * 25, 1e30),  # the product 1e750 would overflow float64
        ([1.0, 1e-30], [0.0, 0.0], 1e-15),  # one tiny gap pulls the mean down, without underflow
    ],
)
def test_pairwise_distance_geometric_mean_values(x: list[float], y: list[float], expected_value: float):
    """The geometric-mean distance handles zero, tiny, huge and negative gaps exactly or to float32 precision."""
    # --- arrange ----------------------
    vectors = np.array([x, y], dtype=np.float32)

    # --- act --------------------------
    d = condensed_distances(vectors, metric=DistanceMetric.geometric_mean())

    # --- assert -----------------------
    if expected_value == 0.0:
        assert d[0] == np.float32(0.0)
    else:
        assert d[0] == pytest.approx(expected_value, rel=1e-6)


# ==================================================================================================
#  Along one axis
# ==================================================================================================
@pytest.mark.parametrize(
    "axis, expected_value",
    [
        (0, 3.0),
        (1, 4.0),
        (2, 0.0),  # a shared coordinate is at distance zero, however far apart the vectors are elsewhere
    ],
)
def test_pairwise_distance_along_axis_values(axis: int, expected_value: float):
    """The along-axis distance is the absolute difference of that one coordinate and ignores every other."""
    # --- arrange ----------------------
    vectors = np.array([[0.0, 0.0, 5.0], [-3.0, 4.0, 5.0]], dtype=np.float32)

    # --- act --------------------------
    d = condensed_distances(vectors, metric=DistanceMetric.along_axis(axis))

    # --- assert -----------------------
    assert d[0] == np.float32(expected_value)


# ==================================================================================================
#  L-∞
# ==================================================================================================
@pytest.mark.parametrize(
    "x, y, expected_value",
    [
        ([0.0, 0.0], [3.0, 4.0], 3.0),  # the smaller gap
        ([1.0, 5.0, 2.0], [1.0, 9.0, 7.0], 0.0),  # a shared coordinate gives distance zero
        ([0.0, 0.0], [-3.0, 4.0], 3.0),  # differences enter by absolute value
        ([7.0], [3.0], 4.0),  # one dimension: the single gap itself
        ([0.0, 0.0, 0.0], [4.0, 1.0, 2.0], 1.0),  # the minimum sits in the middle
    ],
)
def test_pairwise_distance_lminusinf_values(x: list[float], y: list[float], expected_value: float):
    """The L-∞ distance is the smallest absolute coordinate difference, zero on a shared coordinate."""
    # --- arrange ----------------------
    vectors = np.array([x, y], dtype=np.float32)

    # --- act --------------------------
    d = condensed_distances(vectors, metric=DistanceMetric.l_minus_inf())

    # --- assert -----------------------
    assert d[0] == pytest.approx(expected_value)


# ==================================================================================================
#  L2 and projections
# ==================================================================================================
@pytest.mark.parametrize("n_dims", [2, 3, 4, 5, 10])
@pytest.mark.parametrize("l2_scale", [1.0, 0.25])
def test_pairwise_distance_l2_and_projections_matches_reference(n_dims: int, l2_scale: float):
    """Every pair's distance is the smaller of the smallest coordinate gap and the scaled L2 distance to the power d."""
    # --- arrange ----------------------
    vectors = np.random.default_rng(20260928).random((40, n_dims)).astype(np.float32)
    metric = DistanceMetric.l2_and_projections(l2_scale=l2_scale)
    expected = [
        l2_and_projections_reference(vectors[i], vectors[j], l2_scale)
        for i in range(len(vectors))
        for j in range(i + 1, len(vectors))
    ]

    # --- act --------------------------
    d = condensed_distances(vectors, metric=metric)

    # --- assert -----------------------
    np.testing.assert_allclose(d, np.array(expected, dtype=np.float32), rtol=1e-5)


@pytest.mark.parametrize(
    "a, b, expected_value",
    [
        ([0.0, 0.0], [0.3, 0.4], 0.25),  # the L2 part 0.5^2 is below both gaps
        ([0.0, 0.0], [0.1, 0.9], 0.1),  # a gap is below the L2 part 0.82
        ([0.2, 0.7], [0.2, 0.1], 0.0),  # a shared coordinate gives distance zero
    ],
)
def test_pairwise_distance_l2_and_projections_values(a: list[float], b: list[float], expected_value: float):
    """In 2 dimensions the distance is the smaller of the 2 gaps and the squared L2 distance."""
    # --- arrange ----------------------
    vectors = np.array([a, b], dtype=np.float32)

    # --- act --------------------------
    d = condensed_distances(vectors, metric=DistanceMetric.l2_and_projections())

    # --- assert -----------------------
    assert d[0] == pytest.approx(expected_value, rel=1e-6)


@pytest.mark.parametrize("n_dims", [2, 3, 5, 10])
@pytest.mark.parametrize("k", [2, 100])
def test_pairwise_distance_l2_and_projections_with_k_matches_reference(n_dims: int, k: int):
    """With `k`, every pair's distance is the smaller of the smallest coordinate gap and the scaled L2 distance."""
    # --- arrange ----------------------
    vectors = np.random.default_rng(20261001).random((40, n_dims)).astype(np.float32)
    metric = DistanceMetric.l2_and_projections(l2_scale=0.5, k=k)
    expected = [
        l2_and_projections_reference(vectors[i], vectors[j], 0.5, k=k)
        for i in range(len(vectors))
        for j in range(i + 1, len(vectors))
    ]

    # --- act --------------------------
    d = condensed_distances(vectors, metric=metric)

    # --- assert -----------------------
    np.testing.assert_allclose(d, np.array(expected, dtype=np.float32), rtol=1e-5)


def test_pairwise_distance_l2_and_projections_with_k_is_the_weighted_minimum_of_its_parts():
    """With `k` the distance is the minimum of the L-∞ and L2 distances weighted (k - 1, k^(1/d) - 1), over k - 1."""
    # --- arrange ----------------------
    k = 100
    vectors = np.random.default_rng(20261001).random((30, 2)).astype(np.float32)
    lminusinf = condensed_distances(vectors, metric=DistanceMetric.l_minus_inf()).astype(np.float64)
    l2 = condensed_distances(vectors, metric=DistanceMetric.l2_euclidean()).astype(np.float64)
    expected = np.minimum((k - 1) * lminusinf, (k**0.5 - 1) * l2) / (k - 1)

    # --- act --------------------------
    d = condensed_distances(vectors, metric=DistanceMetric.l2_and_projections(k=k))

    # --- assert -----------------------
    np.testing.assert_allclose(d, expected, rtol=1e-5)


def test_pairwise_distance_l2_and_projections_ignores_an_overflowing_l2_part():
    """A far pair in a high dimension, whose L2 part overflows, still gets its smallest gap as the distance."""
    # --- arrange ----------------------
    vectors = np.array([np.zeros(400), np.full(400, 1e3)], dtype=np.float32)
    vectors[1, 7] = 2.0

    # --- act --------------------------
    d = condensed_distances(vectors, metric=DistanceMetric.l2_and_projections())

    # --- assert -----------------------
    assert d[0] == np.float32(2.0)
