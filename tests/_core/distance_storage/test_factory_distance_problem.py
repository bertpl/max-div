import numpy as np
import pytest
from scipy.spatial.distance import squareform

from max_div._core.distance_storage import (
    DistanceProblemDistanceStoreFactory,
    DistanceStorageType,
    DistanceStoreFactory,
)
from max_div._core.metrics import DistanceMetric
from max_div._core.metrics._distance import KIND_FULL_MATRIX, DistanceStore, compute_full_matrix, get_distance

# ==================================================================================================
#  Fixtures / helpers
# ==================================================================================================
N = 10


def _square_distances() -> np.ndarray:
    """Return the L2 distance matrix of a small set of vectors."""
    rng = np.random.default_rng(20260731)
    vectors = rng.random((N, 3)).astype(np.float32) + 0.1
    return compute_full_matrix(vectors, DistanceMetric.l2_euclidean())


def _factory(form: str, storage: DistanceStorageType) -> DistanceProblemDistanceStoreFactory:
    """Return a factory over the small distances, given as a square matrix or as a condensed vector."""
    square = _square_distances()
    distances = square if form == "square" else squareform(square, checks=False)
    return DistanceProblemDistanceStoreFactory(distances, N, storage)


def _all_pairs(store: DistanceStore) -> list[float]:
    """Return every (i, j) distance the store reports, self-pairs included."""
    return [get_distance(store, np.int32(i), np.int32(j)) for i in range(N) for j in range(N)]


# ==================================================================================================
#  Policy
# ==================================================================================================
@pytest.mark.parametrize("form", ["condensed", "square"])
def test_auto_is_the_full_matrix(form: str):
    """AUTO resolves to the full matrix whatever the input form."""
    # --- act / assert -----------------
    assert _factory(form, DistanceStorageType.AUTO).determine_storage_types() == [DistanceStorageType.FULL_MATRIX]


def test_the_one_store_is_reported_without_a_distance_metric():
    """The given distances have no metric, so the single store is reported with None for its distance metric."""
    # --- arrange ----------------------
    factory = _factory("square", DistanceStorageType.AUTO)

    # --- act / assert -----------------
    assert factory.distance_metrics == (None,)
    assert factory.resolved_storage().per_store == ((None, DistanceStorageType.FULL_MATRIX),)


# ==================================================================================================
#  Store construction, in process
# ==================================================================================================
def test_square_input_is_adopted_zero_copy():
    """A square input is read as given, without copying."""
    # --- arrange ----------------------
    square = _square_distances()

    # --- act --------------------------
    (store,) = DistanceProblemDistanceStoreFactory(square, N, DistanceStorageType.FULL_MATRIX).create_stores()

    # --- assert -----------------------
    assert np.shares_memory(store.matrix, square)


def test_condensed_input_expands_to_the_square_matrix():
    """A condensed input is expanded into a fresh matrix holding the same values."""
    # --- act --------------------------
    (store,) = _factory("condensed", DistanceStorageType.FULL_MATRIX).create_stores()

    # --- assert -----------------------
    assert store.kind == KIND_FULL_MATRIX
    np.testing.assert_array_equal(store.matrix, _square_distances())


def test_lazy_raises():
    """LAZY has no vectors to compute distances from."""
    # --- act / assert -----------------
    with pytest.raises(ValueError, match="from vectors"):
        _factory("condensed", DistanceStorageType.LAZY).create_stores()


# ==================================================================================================
#  Shared-memory construction
# ==================================================================================================
@pytest.mark.parametrize("form", ["square", "condensed"])
def test_published_store_holds_the_given_distances(form: str):
    """The given distances land in a segment whatever their form, since the bytes must live there."""
    # --- arrange ----------------------
    factory = _factory(form, DistanceStorageType.AUTO)
    (expected,) = factory.create_stores()

    # --- act --------------------------
    with factory.publish_distance_stores() as specs, DistanceStoreFactory.attach_distance_stores(specs) as attached:
        read = _all_pairs(attached[0])

    # --- assert -----------------------
    assert read == _all_pairs(expected)


def test_publishing_lazy_raises():
    """LAZY has no vectors to compute distances from, published or not."""
    # --- act / assert -----------------
    with (
        pytest.raises(ValueError, match="computes distances from vectors"),
        _factory("condensed", DistanceStorageType.LAZY).publish_distance_stores(),
    ):
        pass
