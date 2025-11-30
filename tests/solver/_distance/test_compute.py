import numpy as np
import pytest
from scipy.spatial.distance import squareform

from max_div.solver._distance import (
    DistanceMetric,
    compute_pdist,
    compute_separation,
    get_pdist,
    update_separation_add,
    update_separation_remove,
)


# -------------------------------------------------------------------------
#  Compute
# -------------------------------------------------------------------------
@pytest.mark.parametrize("metric", list(DistanceMetric))
def test_compute_pdist_metrics(metric: DistanceMetric):
    """Check if compute_pdist implements all metrics."""

    # --- arrange -----------------------------------------
    vectors = np.array([[0, 0], [3, 4], [1, 0], [0, 1]], dtype=np.float32)

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
    value = get_pdist(d, np.int32(i), np.int32(j), np.int32(m))

    # --- assert ------------------------------------------
    assert value == pytest.approx(expected_value)


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
