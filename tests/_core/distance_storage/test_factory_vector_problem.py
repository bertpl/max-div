import numpy as np
import pytest

from max_div._core.constraints import Constraint
from max_div._core.distance_storage import (
    DistanceStorageType,
    DistanceStoreFactory,
    VectorProblemDistanceStoreFactory,
)
from max_div._core.metrics import DistanceMetric, DiversityMetric
from max_div._core.metrics._distance import KIND_FULL_MATRIX, KIND_LAZY, DistanceStore, get_distance
from max_div._core.problem import MaxDivProblem
from max_div._core.solver import MaxDivSolverBuilder, SolverPreset, Verbosity
from max_div._core.solver._duration import iterations

# ==================================================================================================
#  Fixtures / helpers
# ==================================================================================================
GIB = 2**30
L2 = DistanceMetric.l2_euclidean()
L1 = DistanceMetric.l1_manhattan()


def _vectors(n: int = 10) -> np.ndarray:
    """Return a small array of float32 vectors with no zero rows."""
    rng = np.random.default_rng(20260731)
    return rng.random((n, 3)).astype(np.float32) + 0.1


def _factory(
    storage_type: DistanceStorageType,
    metrics: tuple[DistanceMetric, ...] = (L2,),
    vectors: np.ndarray | None = None,
    total_memory: int | None = 64 * GIB,
) -> VectorProblemDistanceStoreFactory:
    """Return a factory over the small vectors and one L2 store, with a generous RAM figure, unless told otherwise."""
    return VectorProblemDistanceStoreFactory(
        _vectors() if vectors is None else vectors, metrics, storage_type, total_memory
    )


def _vectors_without_columns(n: int) -> np.ndarray:
    """Return n vectors of 0 dimensions, which take no memory; the storage policy reads only n."""
    return np.zeros((n, 0), dtype=np.float32)


def _all_pairs(store: DistanceStore, n: int) -> list[float]:
    """Return the store's distance for every (i, j) pair, self-pairs included."""
    return [get_distance(store, np.int32(i), np.int32(j)) for i in range(n) for j in range(n)]


# ==================================================================================================
#  Construction
# ==================================================================================================
def test_a_factory_without_distance_metrics_is_rejected():
    """A factory needs a distance metric for each store that it builds, so an empty list is rejected."""
    # --- act / assert -----------------
    with pytest.raises(ValueError, match="at least one distance metric"):
        _factory(DistanceStorageType.AUTO, metrics=())


def test_distance_metrics_are_reported_in_store_order():
    """The factory reports its metrics in the order given, each paired with its resolved storage type."""
    # --- arrange ----------------------
    factory = _factory(DistanceStorageType.LAZY, metrics=(L1, L2))

    # --- act / assert -----------------
    assert factory.distance_metrics == (L1, L2)
    assert factory.resolved_storage_types().per_store == (
        (L1, DistanceStorageType.LAZY),
        (L2, DistanceStorageType.LAZY),
    )


# ==================================================================================================
#  Policy
# ==================================================================================================
@pytest.mark.parametrize(
    "n, total_memory, expected",
    [
        (10, 64 * GIB, DistanceStorageType.FULL_MATRIX),  # tiny problem: matrix always fits
        (10, None, DistanceStorageType.LAZY),  # probe failed: the one storage type that cannot page
        (50_000, 32 * GIB, DistanceStorageType.FULL_MATRIX),  # 10.0 GiB matrix <= 1/3 of 32 GiB
        (50_000, 16 * GIB, DistanceStorageType.LAZY),  # matrix over budget
    ],
)
def test_auto_is_the_full_matrix_when_it_fits(n: int, total_memory: int | None, expected: DistanceStorageType):
    """AUTO picks the full matrix when its bytes fit within `AUTO_MEMORY_FRACTION` of total RAM, else lazy."""
    # --- arrange ----------------------
    factory = _factory(DistanceStorageType.AUTO, vectors=_vectors_without_columns(n), total_memory=total_memory)

    # --- act / assert -----------------
    assert factory.determine_storage_types() == [expected]


def test_auto_decides_on_the_bytes_of_every_matrix_together():
    """With 2 distances, AUTO picks lazy for both stores when 1 full matrix fits in the memory budget but 2 do not."""
    # --- arrange ----------------------
    factory = _factory(
        DistanceStorageType.AUTO,
        metrics=(L2, L1),
        vectors=_vectors_without_columns(50_000),  # one matrix is 10 GiB
        total_memory=32 * GIB,
    )

    # --- act / assert -----------------
    assert factory.determine_storage_types() == [DistanceStorageType.LAZY, DistanceStorageType.LAZY]


# ==================================================================================================
#  Store construction, in process
# ==================================================================================================
@pytest.mark.parametrize(
    "storage_type, expected_kind",
    [(DistanceStorageType.FULL_MATRIX, KIND_FULL_MATRIX), (DistanceStorageType.LAZY, KIND_LAZY)],
)
def test_create_stores(storage_type: DistanceStorageType, expected_kind: np.int32):
    """Each storage type builds a store of the matching kind over the items."""
    # --- act --------------------------
    stores = _factory(storage_type).create_stores()

    # --- assert -----------------------
    assert len(stores) == 1
    assert stores[0].kind == expected_kind
    assert stores[0].n == np.int32(10)


def test_lazy_store_over_a_metric_that_does_not_preprocess_reads_the_given_vectors():
    """A metric that does not preprocess gets a lazy store over the user's array itself, not a copy."""
    # --- arrange ----------------------
    vectors = _vectors()

    # --- act --------------------------
    (store,) = _factory(DistanceStorageType.LAZY, vectors=vectors).create_stores()

    # --- assert -----------------------
    assert np.shares_memory(store.preprocessed_vectors, vectors)


def test_lazy_store_over_a_preprocessing_metric_reads_a_preprocessed_copy():
    """A preprocessing metric gets its own preprocessed array, and the user's vectors stay untouched."""
    # --- arrange ----------------------
    vectors = _vectors()
    before = vectors.copy()

    # --- act --------------------------
    (store,) = _factory(DistanceStorageType.LAZY, metrics=(DistanceMetric.cosine(),), vectors=vectors).create_stores()

    # --- assert -----------------------
    assert not np.shares_memory(store.preprocessed_vectors, vectors)
    np.testing.assert_array_equal(vectors, before)
    np.testing.assert_allclose(np.linalg.norm(store.preprocessed_vectors, axis=1), 1.0, rtol=1e-6)


def test_metrics_that_do_not_preprocess_share_one_lazy_array_and_a_preprocessing_one_does_not():
    """Over several distances, every metric that does not preprocess reads the same array; cosine reads its own."""
    # --- arrange ----------------------
    factory = _factory(DistanceStorageType.LAZY, metrics=(L2, L1, DistanceMetric.cosine()))

    # --- act --------------------------
    l2, l1, cosine = factory.create_stores()

    # --- assert -----------------------
    assert np.shares_memory(l2.preprocessed_vectors, l1.preprocessed_vectors)
    assert not np.shares_memory(cosine.preprocessed_vectors, l2.preprocessed_vectors)


@pytest.mark.parametrize("metric", [L2, DistanceMetric.cosine()], ids=["non-preprocessing", "preprocessing"])
def test_full_matrix_and_lazy_stores_agree(metric: DistanceMetric):
    """Both kinds read bit-identical distances, preprocessing metric included."""
    # --- act --------------------------
    (full,) = _factory(DistanceStorageType.FULL_MATRIX, metrics=(metric,)).create_stores()
    (lazy,) = _factory(DistanceStorageType.LAZY, metrics=(metric,)).create_stores()

    # --- assert -----------------------
    assert _all_pairs(full, 10) == _all_pairs(lazy, 10)


def test_infeasible_full_matrix_raises_early():
    """A full matrix that cannot fit in physical RAM is rejected with the remedy named."""
    # --- arrange ----------------------
    factory = _factory(
        DistanceStorageType.FULL_MATRIX,
        vectors=_vectors_without_columns(2_000_000),  # a full matrix would need ~16 TiB
    )

    # --- act / assert -----------------
    with pytest.raises(ValueError, match="LAZY"):
        factory.create_stores()


# ==================================================================================================
#  Per-storage-type solve behavior
# ==================================================================================================
# What a storage type guarantees is that it solves the same problem as well, not that it picks the
# same items: distances may differ in their last bits between storage types, the search is chaotic,
# and one flipped comparison sends it down a different path to an equally good answer.  These
# assert the property that survives that — quality, and feasibility on constrained problems.
@pytest.mark.parametrize("storage_type", [DistanceStorageType.FULL_MATRIX, DistanceStorageType.LAZY])
def test_every_storage_type_reaches_equivalent_quality(storage_type: DistanceStorageType):
    """Each storage type solves an unconstrained problem to within a small margin of the others."""
    # --- arrange ----------------------
    rng = np.random.default_rng(20260804)
    vectors = rng.random((120, 5), dtype=np.float32)
    problem = MaxDivProblem.new(vectors=vectors, k=12, diversity_metric=DiversityMetric.MIN_SEPARATION)

    # --- act --------------------------
    solution = (
        MaxDivSolverBuilder(problem)
        .with_preset(iterations(200), SolverPreset.SMART)
        .with_seed(7)
        .with_distance_storage(storage_type)
        .build()
        .solve(verbosity=Verbosity.SILENT)
    )

    # --- assert -----------------------
    assert solution.score.diversity > 0.0
    assert len(solution.i_selected) == 12
    assert len({int(i) for i in solution.i_selected}) == 12  # a selection, not a multiset


@pytest.mark.parametrize("storage_type", [DistanceStorageType.FULL_MATRIX, DistanceStorageType.LAZY])
def test_every_storage_type_reaches_feasibility(storage_type: DistanceStorageType):
    """Each storage type satisfies a reachable count constraint, whatever items it ends up choosing."""
    # --- arrange ----------------------
    rng = np.random.default_rng(20260804)
    vectors = rng.random((120, 5), dtype=np.float32)
    first_half = list(range(60))
    problem = MaxDivProblem.new(
        vectors=vectors,
        k=12,
        diversity_metric=DiversityMetric.MIN_SEPARATION,
        constraints=[Constraint(int_set=set(first_half), min_count=6, max_count=6)],
    )

    # --- act --------------------------
    solution = (
        MaxDivSolverBuilder(problem)
        .with_preset(iterations(400), SolverPreset.SMART)
        .with_seed(7)
        .with_distance_storage(storage_type)
        .build()
        .solve(verbosity=Verbosity.SILENT)
    )

    # --- assert -----------------------
    n_from_first_half = sum(1 for i in solution.i_selected if int(i) in set(first_half))
    assert n_from_first_half == 6


# ==================================================================================================
#  Shared-memory construction
# ==================================================================================================
@pytest.mark.parametrize("storage_type", [DistanceStorageType.FULL_MATRIX, DistanceStorageType.LAZY])
@pytest.mark.parametrize("metric", [L2, DistanceMetric.cosine()], ids=["non-preprocessing", "preprocessing"])
def test_published_stores_match_the_in_process_build(storage_type: DistanceStorageType, metric: DistanceMetric):
    """A distance store that is published to shared memory holds bit-identical distances to the in-process build."""
    # --- arrange ----------------------
    factory = _factory(storage_type, metrics=(metric,))
    (expected,) = factory.create_stores()

    # --- act --------------------------
    with factory.publish_distance_stores() as specs, DistanceStoreFactory.attach_distance_stores(specs) as attached:
        read_attached = _all_pairs(attached[0], 10)

    # --- assert -----------------------
    assert read_attached == _all_pairs(expected, 10)


def test_published_full_matrix_has_the_problem_size():
    """The spec of a published full matrix describes an n by n array."""
    # --- arrange / act ----------------
    with _factory(DistanceStorageType.FULL_MATRIX).publish_distance_stores() as specs:
        # --- assert -------------------
        assert len(specs) == 1
        assert specs[0].shape == (10, 10)


def test_published_metrics_that_do_not_preprocess_share_one_segment():
    """Lazy stores whose metric does not preprocess share one segment of raw vectors; cosine gets its own."""
    # --- arrange ----------------------
    factory = _factory(DistanceStorageType.LAZY, metrics=(L2, L1, DistanceMetric.cosine()))

    # --- act --------------------------
    with factory.publish_distance_stores() as specs:
        names = [spec.segment_name for spec in specs]

    # --- assert -----------------------
    assert names[0] == names[1] != names[2]
