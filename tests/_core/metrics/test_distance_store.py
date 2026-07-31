import numpy as np
import pytest
from scipy.spatial.distance import squareform

from max_div._core.metrics._distance import (
    DistanceMetric,
    DistanceStore,
    compute_pdist,
    get_distance,
    lazy_store,
)
from max_div._core.metrics._distance._store import KIND_CONDENSED, KIND_LAZY, _condensed_index


# -------------------------------------------------------------------------
#  DistanceStore.condensed
# -------------------------------------------------------------------------
def test_condensed_factory_fields():
    """A condensed store holds the given distances and n; unused backend fields are zero-size."""

    # --- arrange -----------------------------------------
    vectors = np.array([[0, 0], [3, 4], [1, 0], [0, 2]], dtype=np.float32)
    d = compute_pdist(vectors, metric=DistanceMetric.L2_EUCLIDEAN)

    # --- act ---------------------------------------------
    store = DistanceStore.condensed(d, n=4)

    # --- assert ------------------------------------------
    assert isinstance(store, DistanceStore)
    assert store.kind == KIND_CONDENSED
    assert store.n == np.int32(4)
    assert store.pdist is d
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
    store = DistanceStore.condensed(d, n=vectors.shape[0])

    expected_value = squareform(d)[i, j]

    # --- act ---------------------------------------------
    value = get_distance(store, np.int32(i), np.int32(j))

    # --- assert ------------------------------------------
    assert value == pytest.approx(expected_value)


# -------------------------------------------------------------------------
#  lazy_store
# -------------------------------------------------------------------------
def test_lazy_store_fields():
    """A lazy store holds the vectors and metric selector; stored-distance fields are zero-size."""

    # --- arrange -----------------------------------------
    vectors = np.array([[0, 0], [3, 4], [1, 0], [0, 2]], dtype=np.float32)

    # --- act ---------------------------------------------
    store = lazy_store(vectors, DistanceMetric.L2_EUCLIDEAN)

    # --- assert ------------------------------------------
    assert store.kind == KIND_LAZY
    assert store.n == np.int32(4)
    assert store.condensed.size == 0
    assert store.matrix.size == 0
    assert store.vectors.shape == (4, 2)


def test_lazy_store_cosine_zero_vector_raises():
    """The cosine zero-vector guard applies to lazy stores exactly as to precomputed distances."""

    # --- arrange -----------------------------------------
    vectors = np.array([[1.0, 2.0], [0.0, 0.0]], dtype=np.float32)

    # --- act / assert ------------------------------------
    with pytest.raises(ValueError, match="zero vector"):
        lazy_store(vectors, DistanceMetric.COSINE)


# -------------------------------------------------------------------------
#  Cross-backend bit-equality
# -------------------------------------------------------------------------
# The invariant every backend must uphold: get_distance returns bit-identical float32 values for
# every (i, j) pair, whichever storage layout the store holds.  Assertions use exact equality on
# purpose — bit-equality is what keeps solver trajectories identical across backends.
def _all_backend_stores(vectors: np.ndarray, metric: DistanceMetric) -> dict[str, DistanceStore]:
    """Build one store per available backend over the same data."""
    return {
        "condensed": condensed_store(compute_pdist(vectors, metric), n=vectors.shape[0]),
        "lazy": lazy_store(vectors, metric),
    }


@pytest.mark.parametrize("metric", list(DistanceMetric))
def test_get_distance_bit_equal_across_backends(metric: DistanceMetric):
    """Every backend returns bit-identical values for every pair, on random float32 vectors."""

    # --- arrange -----------------------------------------
    rng = np.random.default_rng(20260731)
    vectors = (rng.standard_normal((15, 4)) * 10).astype(np.float32)
    stores = _all_backend_stores(vectors, metric)
    n = vectors.shape[0]

    # --- act ---------------------------------------------
    values = {
        name: np.array([[get_distance(store, np.int32(i), np.int32(j)) for j in range(n)] for i in range(n)])
        for name, store in stores.items()
    }

    # --- assert ------------------------------------------
    reference = values.pop("condensed")
    for name, vals in values.items():
        np.testing.assert_array_equal(vals, reference, err_msg=f"backend {name} not bit-equal to condensed")


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
