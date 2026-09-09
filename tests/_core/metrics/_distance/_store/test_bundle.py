import numpy as np
import pytest
from scipy.spatial.distance import pdist, squareform

from max_div._core.metrics._distance import (
    DistanceMetric,
    DistanceStore,
    get_distance,
)
from max_div._core.metrics._distance._store import KIND_FULL_MATRIX, KIND_LAZY
from tests._core.metrics._distance.helpers import condensed_distances


# -------------------------------------------------------------------------
#  get_distance
# -------------------------------------------------------------------------
@pytest.mark.parametrize("i", [0, 1, 2, 3])
@pytest.mark.parametrize("j", [0, 1, 2, 3])
def test_get_distance_full_matrix_values(i: int, j: int):
    """get_distance returns the correct full-matrix value for every (i, j), including i == j."""

    # --- arrange ----------------------
    vectors = np.array([[0, 0], [3, 4], [1, 0], [0, 2]], dtype=np.float32)
    store = DistanceStore.full_matrix_from_vectors(vectors, DistanceMetric.l2_euclidean())
    expected_value = squareform(pdist(vectors))[i, j]

    # --- act --------------------------
    value = get_distance(store, np.int32(i), np.int32(j))

    # --- assert -----------------------
    assert value == pytest.approx(expected_value)


# -------------------------------------------------------------------------
#  DistanceStore.lazy
# -------------------------------------------------------------------------
def test_lazy_factory_fields():
    """A lazy store holds the vectors and metric selector; the stored-matrix field is zero-size."""

    # --- arrange ----------------------
    vectors = np.array([[0, 0], [3, 4], [1, 0], [0, 2]], dtype=np.float32)

    # --- act --------------------------
    store = DistanceStore.lazy(vectors, DistanceMetric.l2_euclidean())

    # --- assert -----------------------
    assert store.kind == KIND_LAZY
    assert store.n == np.int32(4)
    assert store.matrix.size == 0
    assert store.vectors.shape == (4, 2)


def test_lazy_factory_cosine_zero_vector_raises():
    """The cosine zero-vector guard applies to lazy stores exactly as to precomputed distances."""

    # --- arrange ----------------------
    vectors = np.array([[1.0, 2.0], [0.0, 0.0]], dtype=np.float32)

    # --- act / assert -----------------
    with pytest.raises(ValueError, match="zero vector"):
        DistanceStore.lazy(vectors, DistanceMetric.cosine())


# -------------------------------------------------------------------------
#  DistanceStore.full_matrix
# -------------------------------------------------------------------------
def test_full_matrix_factory_fields():
    """A full-matrix store holds the given matrix and n; the lazy backend's field is zero-size."""

    # --- arrange ----------------------
    matrix = np.zeros((4, 4), dtype=np.float32)

    # --- act --------------------------
    store = DistanceStore.full_matrix(matrix)

    # --- assert -----------------------
    assert store.kind == KIND_FULL_MATRIX
    assert store.n == np.int32(4)
    assert np.shares_memory(store.matrix, matrix)  # zero-copy: a read-only view, not a copy
    assert store.vectors.size == 0


def test_full_matrix_construction_exactly_symmetric(metric: DistanceMetric):
    """Both full-matrix construction paths produce exactly symmetric matrices with zero diagonals."""

    # --- arrange ----------------------
    rng = np.random.default_rng(20260731)
    vectors = (rng.standard_normal((12, 3)) * 5).astype(np.float32)

    # --- act --------------------------
    from_vectors = DistanceStore.full_matrix_from_vectors(vectors, metric)
    from_condensed = DistanceStore.full_matrix_from_condensed(condensed_distances(vectors, metric), n=12)

    # --- assert -----------------------
    for store in (from_vectors, from_condensed):
        np.testing.assert_array_equal(store.matrix, store.matrix.T)  # bit-exact symmetry
        np.testing.assert_array_equal(np.diag(store.matrix), np.zeros(12, dtype=np.float32))


# -------------------------------------------------------------------------
#  Cross-backend bit-equality
# -------------------------------------------------------------------------
# The invariant every backend must uphold: get_distance returns bit-identical float32 values for
# every (i, j) pair, whichever storage layout the store holds.  Assertions use exact equality on
# purpose — bit-equality is what keeps solver trajectories identical across backends.
def _all_backend_stores(vectors: np.ndarray, metric: DistanceMetric) -> dict[str, DistanceStore]:
    """Build one store per available backend (and construction path) over the same data."""
    return {
        "full_from_vectors": DistanceStore.full_matrix_from_vectors(vectors, metric),
        "lazy": DistanceStore.lazy(vectors, metric),
        "full_from_condensed": DistanceStore.full_matrix_from_condensed(
            condensed_distances(vectors, metric), n=vectors.shape[0]
        ),
    }


def _distances_per_backend(vectors: np.ndarray, metric: DistanceMetric) -> dict[str, np.ndarray]:
    """Every pair distance, from every backend, as one matrix per backend."""
    n = vectors.shape[0]
    return {
        name: np.array([[get_distance(store, np.int32(i), np.int32(j)) for j in range(n)] for i in range(n)])
        for name, store in _all_backend_stores(vectors, metric).items()
    }


def test_get_distance_agrees_across_backends(metric: DistanceMetric):
    """Every backend returns the same distance for every pair, to within summation rounding.

    This is the contract. It is a tolerance rather than an equality because a distance is a sum
    over dimensions, and a compiler may sum it in a different order per backend — the loops have
    different shapes, so they vectorize differently. Two orderings of d terms drift by about
    sqrt(d) units in the last place, which is why the bound scales with d rather than being a
    fixed epsilon: a fixed one would be a flake generator at high dimensionality.
    """

    # --- arrange ----------------------
    rng = np.random.default_rng(20260731)
    vectors = (rng.standard_normal((15, 4)) * 10).astype(np.float32)
    n_dimensions = vectors.shape[1]
    tolerance = 8.0 * np.sqrt(n_dimensions) * np.finfo(np.float32).eps

    # --- act --------------------------
    values = _distances_per_backend(vectors, metric)

    # --- assert -----------------------
    reference = values.pop("full_from_vectors")
    scale = float(np.max(reference))
    for name, vals in values.items():
        np.testing.assert_allclose(
            vals,
            reference,
            rtol=tolerance,
            atol=tolerance * scale,
            err_msg=f"backend {name} disagrees with the full matrix from vectors",
        )


def test_get_distance_bit_equal_across_backends_canary(metric: DistanceMetric):
    """Canary: the backends currently agree bit-for-bit, which is stronger than they promise.

    Not a contract — the contract is the tolerance test above. A failure here means the
    toolchain started vectorizing one of these loops differently, which is information worth
    having at the moment it happens, not a defect in this codebase. The response is to record
    the change and confirm the tolerance test still passes, never to widen this one until it
    goes green.
    """

    # --- arrange ----------------------
    rng = np.random.default_rng(20260731)
    vectors = (rng.standard_normal((15, 4)) * 10).astype(np.float32)

    # --- act --------------------------
    values = _distances_per_backend(vectors, metric)

    # --- assert -----------------------
    reference = values.pop("full_from_vectors")
    for name, vals in values.items():
        np.testing.assert_array_equal(
            vals, reference, err_msg=f"backend {name} no longer bit-equal to the full matrix from vectors"
        )
