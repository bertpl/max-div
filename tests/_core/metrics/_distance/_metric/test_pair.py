import numpy as np
import pytest
from scipy.spatial.distance import pdist as scipy_pdist

from max_div._core.metrics._distance import (
    DistanceMetric,
    compute_pdist,
)

_SCIPY_METRIC = {
    DistanceMetric.l1_manhattan(): "cityblock",
    DistanceMetric.l2_euclidean(): "euclidean",
    DistanceMetric.l2s_euclidean_squared(): "sqeuclidean",
    DistanceMetric.linf_chebyshev(): "chebyshev",
    DistanceMetric.cosine(): "cosine",
}


# -------------------------------------------------------------------------
#  Compute
# -------------------------------------------------------------------------
def test_compute_pdist_metrics(metric: DistanceMetric):
    """Check if compute_pdist implements all metrics."""

    # --- arrange ----------------------
    # note: no all-zero row — COSINE rejects zero vectors
    vectors = np.array([[2, 2], [3, 4], [1, 0], [0, 1]], dtype=np.float32)

    # --- act --------------------------
    d = compute_pdist(vectors, metric=metric)

    # --- assert -----------------------
    assert d.shape == (6,), "Unexpected shape of pdist output."
    assert d.dtype == np.float32, "Unexpected dtype of pdist output."


@pytest.mark.parametrize(
    "metric, expected_value",
    [
        (DistanceMetric.l1_manhattan(), 7.0),
        (DistanceMetric.l2_euclidean(), 5.0),
        (DistanceMetric.l2s_euclidean_squared(), 25.0),
        (DistanceMetric.linf_chebyshev(), 4.0),
        (DistanceMetric.geometric_mean(), 12.0**0.5),
    ],
)
def test_compute_pdist_values(metric: DistanceMetric, expected_value: float):
    """Check if compute_pdist produces correct values."""

    # --- arrange ----------------------
    vectors = np.array([[0, 0], [3, 4]], dtype=np.float32)

    # --- act --------------------------
    d = compute_pdist(vectors, metric=metric)

    # --- assert -----------------------
    assert d[0] == pytest.approx(expected_value)


@pytest.mark.parametrize("metric", list(_SCIPY_METRIC), ids=repr)
def test_compute_pdist_matches_scipy(metric: DistanceMetric):
    """The hand-rolled float32 kernel matches scipy's float64→float32 result within float32 tolerance."""

    # --- arrange ----------------------
    rng = np.random.default_rng(20260711)
    vectors = rng.standard_normal((60, 8)).astype(np.float32)
    expected = scipy_pdist(vectors, metric=_SCIPY_METRIC[metric]).astype(np.float32)

    # --- act --------------------------
    result = compute_pdist(vectors, metric=metric)

    # --- assert -----------------------
    assert result.dtype == np.float32
    np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("p", [0.125, 0.25, 0.5, 1.5, 3.0])
@pytest.mark.parametrize("root", [True, False])
def test_compute_pdist_minkowski_matches_reference(p: float, root: bool):
    """Minkowski distances match a float64 numpy reference, for specialized and generic p."""
    # --- arrange ----------------------
    rng = np.random.default_rng(20260829)
    vectors = rng.standard_normal((40, 6)).astype(np.float32)
    diffs = np.abs(vectors[:, None, :].astype(np.float64) - vectors[None, :, :].astype(np.float64))
    powered = (diffs**p).sum(axis=2)
    expected_matrix = powered ** (1.0 / p) if root else powered
    expected = expected_matrix[np.triu_indices(40, k=1)].astype(np.float32)

    # --- act --------------------------
    result = compute_pdist(vectors, metric=DistanceMetric.minkowski(p, root=root))

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
def test_compute_pdist_cosine_values(x: list[float], y: list[float], expected_value: float):
    """Cosine distance produces the expected angular values."""

    # --- arrange ----------------------
    vectors = np.array([x, y], dtype=np.float32)

    # --- act --------------------------
    d = compute_pdist(vectors, metric=DistanceMetric.cosine())

    # --- assert -----------------------
    assert d[0] == pytest.approx(expected_value, abs=1e-6)


def test_compute_pdist_cosine_zero_vector_raises():
    """Cosine distance rejects all-zero vectors with a clear error naming the row."""

    # --- arrange ----------------------
    vectors = np.array([[1, 2], [0, 0], [3, 4]], dtype=np.float32)

    # --- act / assert -----------------
    with pytest.raises(ValueError, match=r"zero vector.*row 1"):
        compute_pdist(vectors, metric=DistanceMetric.cosine())


def test_compute_pdist_zero_for_identical_vectors(metric: DistanceMetric):
    """Identical vectors have exactly-zero distance under every metric."""

    # --- arrange ----------------------
    vectors = np.array([[1.5, -2.0, 3.0], [1.5, -2.0, 3.0], [4.0, 4.0, 4.0]], dtype=np.float32)

    # --- act --------------------------
    result = compute_pdist(vectors, metric=metric)

    # --- assert -----------------------
    assert result[0] == np.float32(0.0)  # distance between the two identical vectors


# -------------------------------------------------------------------------
#  Geometric mean
# -------------------------------------------------------------------------
def _geomean_reference(x: np.ndarray, y: np.ndarray) -> float:
    """Return the geometric mean of the absolute differences via numpy's log-mean, in float64."""
    diffs = np.abs(x.astype(np.float64) - y.astype(np.float64))
    return 0.0 if np.any(diffs == 0.0) else float(np.exp(np.mean(np.log(diffs))))


@pytest.mark.parametrize(
    "x, y, expected_value",
    [
        ([0.0, 0.0], [3.0, 4.0], 12.0**0.5),  # sqrt(3 * 4)
        ([1.0, 5.0, 2.0], [1.0, 9.0, 7.0], 0.0),  # a shared coordinate zeroes the product
        ([0.0, 0.0], [-3.0, 4.0], 12.0**0.5),  # differences enter by absolute value
        ([7.0], [3.0], 4.0),  # one dimension: the single gap itself
        ([1e-18] * 25, [0.0] * 25, 1e-18),  # the product 1e-450 would underflow float64
        ([1e30] * 25, [0.0] * 25, 1e30),  # the product 1e750 would overflow float64
        ([1.0, 1e-30], [0.0, 0.0], 1e-15),  # one tiny gap pulls the mean down, without underflow
    ],
)
def test_compute_pdist_geometric_mean_values(x: list[float], y: list[float], expected_value: float):
    """The geometric-mean distance handles zero, tiny, huge and negative gaps exactly or to float32 precision."""
    # --- arrange ----------------------
    vectors = np.array([x, y], dtype=np.float32)

    # --- act --------------------------
    d = compute_pdist(vectors, metric=DistanceMetric.geometric_mean())

    # --- assert -----------------------
    if expected_value == 0.0:
        assert d[0] == np.float32(0.0)
    else:
        assert d[0] == pytest.approx(expected_value, rel=1e-6)


def test_compute_pdist_geometric_mean_matches_log_mean_reference():
    """On random vectors the metric agrees with a float64 log-mean reference to float32 precision."""
    # --- arrange ----------------------
    rng = np.random.default_rng(7)
    vectors = rng.standard_normal((12, 6)).astype(np.float32)
    expected = [_geomean_reference(vectors[i], vectors[j]) for i in range(12) for j in range(i + 1, 12)]

    # --- act --------------------------
    d = compute_pdist(vectors, metric=DistanceMetric.geometric_mean())

    # --- assert -----------------------
    np.testing.assert_allclose(d, expected, rtol=1e-6)


def test_compute_pdist_geometric_mean_scales_with_the_vectors():
    """Scaling every coordinate by s scales the distance by s: the metric is homogeneous of degree one."""
    # --- arrange ----------------------
    rng = np.random.default_rng(11)
    vectors = rng.random((5, 4)).astype(np.float32)

    # --- act --------------------------
    d = compute_pdist(vectors, metric=DistanceMetric.geometric_mean())
    d_scaled = compute_pdist(np.float32(2.5) * vectors, metric=DistanceMetric.geometric_mean())

    # --- assert -----------------------
    np.testing.assert_allclose(d_scaled, 2.5 * d, rtol=1e-6)


def test_compute_pdist_geometric_mean_sits_between_the_smallest_gap_and_the_mean_gap():
    """By the AM-GM inequality the distance lies between the smallest per-dimension gap and the L1 distance over d."""
    # --- arrange ----------------------
    rng = np.random.default_rng(3)
    vectors = rng.random((6, 5)).astype(np.float32)
    n, d_dim = vectors.shape
    smallest_gap = np.array(
        [np.abs(vectors[i] - vectors[j]).min() for i in range(n) for j in range(i + 1, n)], dtype=np.float64
    )

    # --- act --------------------------
    geomean = compute_pdist(vectors, metric=DistanceMetric.geometric_mean()).astype(np.float64)
    mean_gap = compute_pdist(vectors, metric=DistanceMetric.l1_manhattan()).astype(np.float64) / d_dim

    # --- assert -----------------------
    assert np.all(geomean >= smallest_gap * (1 - 1e-6))
    assert np.all(geomean <= mean_gap * (1 + 1e-6))
