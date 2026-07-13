import numpy as np
import pytest
from scipy.spatial.distance import pdist as scipy_pdist
from scipy.spatial.distance import squareform

from max_div._core.metrics._distance import (
    DistanceMetric,
    compute_pdist,
    compute_separation,
    get_pdist_el,
    update_separation_add,
    update_separation_remove,
)
from max_div._core.metrics._distance._compute import _pdist_index

_SCIPY_METRIC = {
    "L1_MANHATTAN": "cityblock",
    "L2_EUCLIDEAN": "euclidean",
    "L2S_EUCLIDEAN_SQUARED": "sqeuclidean",
    "COSINE": "cosine",
}


# -------------------------------------------------------------------------
#  Compute
# -------------------------------------------------------------------------
@pytest.mark.parametrize("metric", list(DistanceMetric))
def test_compute_pdist_metrics(metric: DistanceMetric):
    """Check if compute_pdist implements all metrics."""

    # --- arrange -----------------------------------------
    # note: no all-zero row — COSINE rejects zero vectors
    vectors = np.array([[2, 2], [3, 4], [1, 0], [0, 1]], dtype=np.float32)

    # --- act ---------------------------------------------
    d = compute_pdist(vectors, metric=metric)

    # --- assert ------------------------------------------
    assert d.shape == (6,), "Unexpected shape of pdist output."
    assert d.dtype == np.float32, "Unexpected dtype of pdist output."


@pytest.mark.parametrize(
    "metric, expected_value",
    [
        (DistanceMetric.L1_MANHATTAN, 7.0),
        (DistanceMetric.L2_EUCLIDEAN, 5.0),
        (DistanceMetric.L2S_EUCLIDEAN_SQUARED, 25.0),
    ],
)
def test_compute_pdist_values(metric: DistanceMetric, expected_value: float):
    """Check if compute_pdist produces correct values."""

    # --- arrange -----------------------------------------
    vectors = np.array([[0, 0], [3, 4]], dtype=np.float32)

    # --- act ---------------------------------------------
    d = compute_pdist(vectors, metric=metric)

    # --- assert ------------------------------------------
    assert d[0] == pytest.approx(expected_value)


@pytest.mark.parametrize("metric", list(DistanceMetric))
def test_compute_pdist_matches_scipy(metric: DistanceMetric):
    """The hand-rolled float32 kernel matches scipy's float64→float32 result within float32 tolerance."""

    # --- arrange -----------------------------------------
    rng = np.random.default_rng(20260711)
    vectors = rng.standard_normal((60, 8)).astype(np.float32)
    expected = scipy_pdist(vectors, metric=_SCIPY_METRIC[metric.value]).astype(np.float32)

    # --- act ---------------------------------------------
    result = compute_pdist(vectors, metric=metric)

    # --- assert ------------------------------------------
    assert result.dtype == np.float32
    np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-5)


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

    # --- arrange -----------------------------------------
    vectors = np.array([x, y], dtype=np.float32)

    # --- act ---------------------------------------------
    d = compute_pdist(vectors, metric=DistanceMetric.COSINE)

    # --- assert ------------------------------------------
    assert d[0] == pytest.approx(expected_value, abs=1e-6)


def test_compute_pdist_cosine_zero_vector_raises():
    """Cosine distance rejects all-zero vectors with a clear error naming the row."""

    # --- arrange -----------------------------------------
    vectors = np.array([[1, 2], [0, 0], [3, 4]], dtype=np.float32)

    # --- act / assert ------------------------------------
    with pytest.raises(ValueError, match=r"zero vector.*row 1"):
        compute_pdist(vectors, metric=DistanceMetric.COSINE)


@pytest.mark.parametrize("metric", list(DistanceMetric))
def test_compute_pdist_zero_for_identical_vectors(metric: DistanceMetric):
    """Identical vectors have exactly-zero distance under every metric."""

    # --- arrange -----------------------------------------
    vectors = np.array([[1.5, -2.0, 3.0], [1.5, -2.0, 3.0], [4.0, 4.0, 4.0]], dtype=np.float32)

    # --- act ---------------------------------------------
    result = compute_pdist(vectors, metric=metric)

    # --- assert ------------------------------------------
    assert result[0] == np.float32(0.0)  # distance between the two identical vectors


# -------------------------------------------------------------------------
#  Low-level
# -------------------------------------------------------------------------
@pytest.mark.parametrize("i", [0, 1, 2, 3])
@pytest.mark.parametrize("j", [0, 1, 2, 3])
def test_get_pdist_values(i: int, j: int):
    """Check if get_pdist produces correct values."""

    # --- arrange -----------------------------------------
    vectors = np.array([[0, 0], [3, 4], [1, 0], [0, 2]], dtype=np.float32)
    d = compute_pdist(vectors, metric=DistanceMetric.L2_EUCLIDEAN)
    m = vectors.shape[0]

    expected_value = squareform(d)[i, j]

    # --- act ---------------------------------------------
    value = get_pdist_el(d, np.int32(i), np.int32(j), np.int32(m))

    # --- assert ------------------------------------------
    assert value == pytest.approx(expected_value)


@pytest.mark.parametrize(
    "i, j, n",
    [
        (0, 1, 4),  # first pair, small n
        (2, 3, 4),  # last pair, small n
        (30_000, 45_000, 50_000),  # off-diagonal in the int32-overflow regime
        (49_998, 49_999, 50_000),  # last pair at n where 32-bit index math overflows
    ],
)
def test_pdist_index_no_int32_overflow(i: int, j: int, n: int):
    """_pdist_index must return the exact condensed offset even where int32 arithmetic would overflow."""

    # --- arrange -----------------------------------------
    # reference offset computed with unbounded Python ints (the value the kernel must match)
    expected = (n * i) + j - ((i + 2) * (i + 1)) // 2

    # --- act ---------------------------------------------
    index = _pdist_index(np.int32(i), np.int32(j), np.int32(n))

    # --- assert ------------------------------------------
    assert int(index) == expected


def test_compute_separation():
    """Check if compute_separation produces correct separation values."""

    # --- arrange -----------------------------------------
    vectors = np.array([[0, 0], [3, 4], [1, 0], [0, 2]], dtype=np.float32)
    d = compute_pdist(vectors, metric=DistanceMetric.L2_EUCLIDEAN)
    m = vectors.shape[0]
    d_squared = squareform(d)

    expected_separation = np.full(m, fill_value=np.inf, dtype=np.float32)
    for i in range(m):
        for j in range(m):
            if i != j:
                dist = d_squared[i, j]
                if dist < expected_separation[i]:
                    expected_separation[i] = dist

    # --- act ---------------------------------------------
    separation = compute_separation(d, np.int32(m))

    # --- assert ------------------------------------------
    np.testing.assert_allclose(separation, expected_separation)


def test_update_separation_add():
    """Check if update_separation_add correctly updates separation after adding a vector."""

    # --- arrange -----------------------------------------
    vectors = np.array([[0, 0], [3, 4], [1, 0], [0, 2], [1.1, 0]], dtype=np.float32)
    m = vectors.shape[0]
    d = compute_pdist(vectors, metric=DistanceMetric.L2_EUCLIDEAN)
    d_squared = squareform(d)

    # initial separation, assuming vector 0 forms the initial selection
    separation = np.array(
        [
            np.inf,
            d_squared[1, 0],
            d_squared[2, 0],
            d_squared[3, 0],
            d_squared[4, 0],
        ],
        dtype=np.float32,
    )

    i_added = 2  # Adding the vector [1, 0]

    expected_separation = np.array(
        [
            d_squared[0, 2],
            min(d_squared[1, 0], d_squared[1, 2]),
            d_squared[2, 0],
            min(d_squared[3, 0], d_squared[3, 2]),
            min(d_squared[4, 0], d_squared[4, 2]),
        ],
        dtype=np.float32,
    )

    # --- act ---------------------------------------------
    update_separation_add(separation, d, np.int32(m), np.int32(i_added))

    # --- assert ------------------------------------------
    np.testing.assert_allclose(separation, expected_separation)


def test_update_separation_remove():
    """Check if update_separation_remove correctly updates separation after adding a vector."""

    # --- arrange -----------------------------------------
    vectors = np.array([[0, 0], [3, 4], [1, 0], [0, 2], [1.1, 0]], dtype=np.float32)
    m = vectors.shape[0]
    d = compute_pdist(vectors, metric=DistanceMetric.L2_EUCLIDEAN)
    d_squared = squareform(d)

    # initial separation, assuming vector 0 & 2 form the initial selection
    separation = np.array(
        [
            d_squared[0, 2],
            min(d_squared[1, 0], d_squared[1, 2]),
            d_squared[2, 0],
            min(d_squared[3, 0], d_squared[3, 2]),
            min(d_squared[4, 0], d_squared[4, 2]),
        ],
        dtype=np.float32,
    )

    i_removed = 2  # Removing the vector [1, 0]

    expected_separation = np.array(
        [
            np.inf,
            d_squared[1, 0],
            d_squared[2, 0],
            d_squared[3, 0],
            d_squared[4, 0],
        ],
        dtype=np.float32,
    )

    # --- act ---------------------------------------------
    update_separation_remove(separation, d, np.int32(m), np.int32(i_removed), np.array([0], dtype=np.int32))

    # --- assert ------------------------------------------
    np.testing.assert_allclose(separation, expected_separation)
