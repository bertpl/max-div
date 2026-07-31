import numpy as np
import pytest
from scipy.spatial.distance import squareform

from max_div._core.metrics._distance import (
    DistanceMetric,
    DistanceStore,
    compute_pdist,
    condensed_store,
    get_distance,
)
from max_div._core.metrics._distance._store import KIND_CONDENSED, _condensed_index


# -------------------------------------------------------------------------
#  condensed_store
# -------------------------------------------------------------------------
def test_condensed_store_fields():
    """A condensed store holds the given distances and n; unused backend fields are zero-size."""

    # --- arrange -----------------------------------------
    vectors = np.array([[0, 0], [3, 4], [1, 0], [0, 2]], dtype=np.float32)
    d = compute_pdist(vectors, metric=DistanceMetric.L2_EUCLIDEAN)

    # --- act ---------------------------------------------
    store = condensed_store(d, n=4)

    # --- assert ------------------------------------------
    assert isinstance(store, DistanceStore)
    assert store.kind == KIND_CONDENSED
    assert store.n == np.int32(4)
    assert store.condensed is d
    assert store.matrix.size == 0
    assert store.vectors.size == 0


# -------------------------------------------------------------------------
#  get_distance
# -------------------------------------------------------------------------
@pytest.mark.parametrize("i", [0, 1, 2, 3])
@pytest.mark.parametrize("j", [0, 1, 2, 3])
def test_get_distance_condensed_values(i: int, j: int):
    """get_distance returns the correct condensed-layout value for every (i, j), including i == j."""

    # --- arrange -----------------------------------------
    vectors = np.array([[0, 0], [3, 4], [1, 0], [0, 2]], dtype=np.float32)
    d = compute_pdist(vectors, metric=DistanceMetric.L2_EUCLIDEAN)
    store = condensed_store(d, n=vectors.shape[0])

    expected_value = squareform(d)[i, j]

    # --- act ---------------------------------------------
    value = get_distance(store, np.int32(i), np.int32(j))

    # --- assert ------------------------------------------
    assert value == pytest.approx(expected_value)


# -------------------------------------------------------------------------
#  Low-level
# -------------------------------------------------------------------------
@pytest.mark.parametrize(
    "i, j, n",
    [
        (0, 1, 4),  # first pair, small n
        (2, 3, 4),  # last pair, small n
        (30_000, 45_000, 50_000),  # off-diagonal in the int32-overflow regime
        (49_998, 49_999, 50_000),  # last pair at n where 32-bit index math overflows
    ],
)
def test_condensed_index_no_int32_overflow(i: int, j: int, n: int):
    """_condensed_index must return the exact condensed offset even where int32 arithmetic would overflow."""

    # --- arrange -----------------------------------------
    # reference offset computed with unbounded Python ints (the value the kernel must match)
    expected = (n * i) + j - ((i + 2) * (i + 1)) // 2

    # --- act ---------------------------------------------
    index = _condensed_index(np.int32(i), np.int32(j), np.int32(n))

    # --- assert ------------------------------------------
    assert int(index) == expected
