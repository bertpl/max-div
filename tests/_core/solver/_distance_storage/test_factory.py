import numpy as np
import pytest
from scipy.spatial.distance import squareform

from max_div._core.constraints import Constraint
from max_div._core.metrics import DistanceMetric, DiversityMetric
from max_div._core.metrics._distance import (
    KIND_FULL_MATRIX,
    KIND_LAZY,
    DistanceStore,
    compute_full_matrix,
    get_distance,
)
from max_div._core.problem import MaxDivProblem
from max_div._core.solver import MaxDivSolverBuilder, SolverPreset, Verbosity
from max_div._core.solver._distance_storage import DistanceStorageType, DistanceStoreFactory
from max_div._core.solver._duration import iterations

# =================================================================================================
#  Fixtures / helpers
# =================================================================================================
GIB = 2**30
L2 = DistanceMetric.l2_euclidean()


def _vector_problem(n: int = 10, metric: DistanceMetric = L2) -> MaxDivProblem:
    """Return a small vector problem under the given metric, with no zero rows."""
    rng = np.random.default_rng(20260731)
    return MaxDivProblem.new(rng.random((n, 3)).astype(np.float32) + 0.1, k=3, distance_metric=metric)


def _distance_problem(form: str) -> MaxDivProblem:
    """Return the L2 distances of the small vector problem as a square or condensed input."""
    matrix = compute_full_matrix(_vector_problem().vectors, L2)  # ty: ignore[unresolved-attribute]
    distances = matrix if form == "square" else squareform(matrix, checks=False)
    return MaxDivProblem.from_distances(distances, k=3)


def _factory(problem: MaxDivProblem, storage: DistanceStorageType, total_memory: int | None = 64 * GIB):
    """Return the one-distance factory over the problem, with a generous RAM figure by default."""
    return DistanceStoreFactory.for_problem(problem, storage, total_memory)


def _stub_vector_problem(n: int):
    """Return a stand-in exposing only what the policy reads (isinstance, n, the metric), without allocations."""
    from max_div._core.problem import VectorMaxDivProblem

    stub = object.__new__(VectorMaxDivProblem)
    object.__setattr__(stub, "vectors", np.zeros((n, 0), dtype=np.float32))
    object.__setattr__(stub, "distance_metric", L2)
    return stub


def _all_pairs(store: DistanceStore, n: int) -> list[float]:
    """Return every (i, j) distance the store reports, self-pairs included."""
    return [get_distance(store, np.int32(i), np.int32(j)) for i in range(n) for j in range(n)]


# =================================================================================================
#  Construction
# =================================================================================================
def test_for_problem_reads_the_problems_own_distance():
    """The one-distance factory names the vector problem's metric, and the given distances otherwise."""
    # --- act / assert -----------------
    assert _factory(_vector_problem(), DistanceStorageType.AUTO)._distances == (L2,)
    assert _factory(_distance_problem("square"), DistanceStorageType.AUTO)._distances == (None,)


@pytest.mark.parametrize(
    "problem, distances, message",
    [
        (_vector_problem(), [None], "every entry must be a DistanceMetric"),
        (_distance_problem("square"), [L2], "every entry must be None"),
        (_vector_problem(), [], "at least one distance"),
    ],
    ids=["none-for-vectors", "metric-for-distances", "empty"],
)
def test_entries_must_fit_the_problem_flavor(problem, distances, message):
    """An entry of the wrong kind for the flavor, or no entry at all, is rejected at construction."""
    # --- act / assert -----------------
    with pytest.raises(ValueError, match=message):
        DistanceStoreFactory(problem, distances, DistanceStorageType.AUTO, 64 * GIB)


# =================================================================================================
#  Policy
# =================================================================================================
@pytest.mark.parametrize("storage", [DistanceStorageType.FULL_MATRIX, DistanceStorageType.LAZY])
def test_explicit_choice_passes_through(storage: DistanceStorageType):
    """A pinned storage type is the resolved one, whatever the memory."""
    # --- act / assert -----------------
    assert _factory(_vector_problem(), storage).determine_storage_types() == [storage]


@pytest.mark.parametrize(
    "n, total_memory, expected",
    [
        (10, 64 * GIB, DistanceStorageType.FULL_MATRIX),  # tiny problem: matrix always fits
        (10, None, DistanceStorageType.LAZY),  # probe failed: the one storage type that cannot page
        (50_000, 32 * GIB, DistanceStorageType.FULL_MATRIX),  # 10.0 GiB matrix <= 1/3 of 32 GiB
        (50_000, 16 * GIB, DistanceStorageType.LAZY),  # matrix over budget
    ],
)
def test_auto_on_vectors_is_the_full_matrix_when_it_fits(
    n: int, total_memory: int | None, expected: DistanceStorageType
):
    """AUTO on vector problems: the full matrix when its bytes fit within a third of total RAM, else lazy."""
    # --- arrange ----------------------
    problem = _vector_problem() if n == 10 else _stub_vector_problem(n)

    # --- act / assert -----------------
    assert _factory(problem, DistanceStorageType.AUTO, total_memory).determine_storage_types() == [expected]


def test_auto_decides_on_the_bytes_of_every_matrix_together():
    """Two distances over a problem whose one matrix fits, but whose two do not, both go lazy."""
    # --- arrange ----------------------
    problem = _stub_vector_problem(50_000)  # one matrix is 10 GiB
    factory = DistanceStoreFactory(problem, [L2, DistanceMetric.l1_manhattan()], DistanceStorageType.AUTO, 32 * GIB)

    # --- act / assert -----------------
    assert factory.determine_storage_types() == [DistanceStorageType.LAZY, DistanceStorageType.LAZY]


@pytest.mark.parametrize("form", ["condensed", "square"])
def test_auto_on_distance_input_is_the_full_matrix(form: str):
    """AUTO on distance-input problems resolves to the full matrix whatever the input form, ignoring memory."""
    # --- act / assert -----------------
    assert _factory(_distance_problem(form), DistanceStorageType.AUTO, None).determine_storage_types() == [
        DistanceStorageType.FULL_MATRIX
    ]


# =================================================================================================
#  Store construction, in process
# =================================================================================================
@pytest.mark.parametrize(
    "storage, expected_kind",
    [(DistanceStorageType.FULL_MATRIX, KIND_FULL_MATRIX), (DistanceStorageType.LAZY, KIND_LAZY)],
)
def test_create_stores_vector_problem(storage: DistanceStorageType, expected_kind: np.int32):
    """Each storage type builds a store of the matching kind over the problem's items."""
    # --- act --------------------------
    stores = _factory(_vector_problem(), storage).create_stores()

    # --- assert -----------------------
    assert len(stores) == 1
    assert stores[0].kind == expected_kind
    assert stores[0].n == np.int32(10)


def test_lazy_store_over_a_metric_that_does_not_preprocess_reads_the_problems_vectors():
    """A metric that does not preprocess gets a lazy store over the user's array itself, not a copy."""
    # --- arrange ----------------------
    problem = _vector_problem()

    # --- act --------------------------
    (store,) = _factory(problem, DistanceStorageType.LAZY).create_stores()

    # --- assert -----------------------
    assert np.shares_memory(store.preprocessed_vectors, problem.vectors)  # ty: ignore[unresolved-attribute]


def test_lazy_store_over_a_preprocessing_metric_reads_a_preprocessed_copy():
    """A preprocessing metric gets its own preprocessed array, and the user's vectors stay untouched."""
    # --- arrange ----------------------
    problem = _vector_problem(metric=DistanceMetric.cosine())
    before = problem.vectors.copy()  # ty: ignore[unresolved-attribute]

    # --- act --------------------------
    (store,) = _factory(problem, DistanceStorageType.LAZY).create_stores()

    # --- assert -----------------------
    assert not np.shares_memory(store.preprocessed_vectors, problem.vectors)  # ty: ignore[unresolved-attribute]
    np.testing.assert_array_equal(problem.vectors, before)  # ty: ignore[unresolved-attribute]
    np.testing.assert_allclose(np.linalg.norm(store.preprocessed_vectors, axis=1), 1.0, rtol=1e-6)


def test_metrics_that_do_not_preprocess_share_one_lazy_array_and_a_preprocessing_one_does_not():
    """Over several distances, every metric that does not preprocess reads the same array; cosine reads its own."""
    # --- arrange ----------------------
    problem = _vector_problem()
    metrics = [L2, DistanceMetric.l1_manhattan(), DistanceMetric.cosine()]
    factory = DistanceStoreFactory(problem, metrics, DistanceStorageType.LAZY, 64 * GIB)

    # --- act --------------------------
    l2, l1, cosine = factory.create_stores()

    # --- assert -----------------------
    assert np.shares_memory(l2.preprocessed_vectors, l1.preprocessed_vectors)
    assert not np.shares_memory(cosine.preprocessed_vectors, l2.preprocessed_vectors)


@pytest.mark.parametrize("metric", [L2, DistanceMetric.cosine()], ids=["non-preprocessing", "preprocessing"])
def test_full_matrix_and_lazy_stores_agree(metric: DistanceMetric):
    """The two kinds read bit-identical distances, preprocessing metric included."""
    # --- arrange ----------------------
    problem = _vector_problem(metric=metric)

    # --- act --------------------------
    (full,) = _factory(problem, DistanceStorageType.FULL_MATRIX).create_stores()
    (lazy,) = _factory(problem, DistanceStorageType.LAZY).create_stores()

    # --- assert -----------------------
    assert _all_pairs(full, problem.n) == _all_pairs(lazy, problem.n)


def test_square_input_is_adopted_zero_copy():
    """FULL_MATRIX on a square-input problem adopts the retained matrix without copying."""
    # --- arrange ----------------------
    problem = _distance_problem("square")

    # --- act --------------------------
    (store,) = _factory(problem, DistanceStorageType.FULL_MATRIX).create_stores()

    # --- assert -----------------------
    assert np.shares_memory(store.matrix, problem.distances)  # ty: ignore[unresolved-attribute]


def test_condensed_input_expands_to_the_square_matrix():
    """FULL_MATRIX on a condensed-input problem expands into a fresh matrix holding the same values."""
    # --- arrange ----------------------
    condensed_problem = _distance_problem("condensed")
    square_problem = _distance_problem("square")

    # --- act --------------------------
    (store,) = _factory(condensed_problem, DistanceStorageType.FULL_MATRIX).create_stores()

    # --- assert -----------------------
    assert store.kind == KIND_FULL_MATRIX
    np.testing.assert_array_equal(store.matrix, square_problem.distances)  # ty: ignore[unresolved-attribute]


def test_lazy_on_distance_problem_raises():
    """LAZY has no vectors to compute from on a distance-input problem."""
    # --- act / assert -----------------
    with pytest.raises(ValueError, match="from vectors"):
        _factory(_distance_problem("condensed"), DistanceStorageType.LAZY).create_stores()


def test_infeasible_full_matrix_raises_early():
    """A full matrix that cannot fit in physical RAM is rejected with the remedy named."""
    # --- arrange ----------------------
    stub = _stub_vector_problem(2_000_000)  # full matrix would need ~16 TiB

    # --- act / assert -----------------
    with pytest.raises(ValueError, match="LAZY"):
        _factory(stub, DistanceStorageType.FULL_MATRIX).create_stores()


# =================================================================================================
#  Per-storage-type solve behavior
# =================================================================================================
# What a storage type guarantees is that it solves the same problem as well, not that it picks the
# same items: distances may differ in their last bits between storage types, the search is chaotic,
# and one flipped comparison sends it down a different path to an equally good answer.  These
# assert the property that survives that — quality, and feasibility on constrained problems.
@pytest.mark.parametrize("storage", [DistanceStorageType.FULL_MATRIX, DistanceStorageType.LAZY])
def test_every_storage_type_reaches_equivalent_quality(storage: DistanceStorageType):
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
        .with_distance_storage(storage)
        .build()
        .solve(verbosity=Verbosity.SILENT)
    )

    # --- assert -----------------------
    assert solution.score.diversity > 0.0
    assert len(solution.i_selected) == 12
    assert len({int(i) for i in solution.i_selected}) == 12  # a selection, not a multiset


@pytest.mark.parametrize("storage", [DistanceStorageType.FULL_MATRIX, DistanceStorageType.LAZY])
def test_every_storage_type_reaches_feasibility(storage: DistanceStorageType):
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
        .with_distance_storage(storage)
        .build()
        .solve(verbosity=Verbosity.SILENT)
    )

    # --- assert -----------------------
    n_from_first_half = sum(1 for i in solution.i_selected if int(i) in set(first_half))
    assert n_from_first_half == 6


# =================================================================================================
#  Shared-memory construction
# =================================================================================================
@pytest.mark.parametrize("storage", [DistanceStorageType.FULL_MATRIX, DistanceStorageType.LAZY])
@pytest.mark.parametrize("metric", [L2, DistanceMetric.cosine()], ids=["non-preprocessing", "preprocessing"])
def test_shared_stores_match_the_in_process_build(storage: DistanceStorageType, metric: DistanceMetric):
    """A store built into shared memory holds bit-identical distances to the in-process build, attached or not."""
    # --- arrange ----------------------
    problem = _vector_problem(metric=metric)
    factory = _factory(problem, storage)
    (expected,) = factory.create_stores()

    # --- act --------------------------
    with factory.create_shared_stores() as shared, DistanceStoreFactory.attach_stores(shared.specs) as attached:
        read_owner = _all_pairs(shared.stores[0], problem.n)
        read_attached = _all_pairs(attached[0], problem.n)

    # --- assert -----------------------
    assert read_owner == _all_pairs(expected, problem.n)
    assert read_attached == _all_pairs(expected, problem.n)


@pytest.mark.parametrize("form", ["square", "condensed"])
def test_shared_stores_hold_distance_input(form: str):
    """Distance-input problems land in a segment whatever their form, since the bytes must live there."""
    # --- arrange ----------------------
    problem = _distance_problem(form)
    factory = _factory(problem, DistanceStorageType.AUTO)
    (expected,) = factory.create_stores()

    # --- act --------------------------
    with factory.create_shared_stores() as shared:
        read = _all_pairs(shared.stores[0], problem.n)
        copied = not np.shares_memory(shared.stores[0].matrix, problem.distances)  # ty: ignore[unresolved-attribute]

    # --- assert -----------------------
    assert read == _all_pairs(expected, problem.n)
    assert copied


def test_shared_full_matrix_is_read_back_by_an_attached_store():
    """The attached process reads the same full matrix the owner built."""
    # --- arrange / act ----------------
    factory = _factory(_vector_problem(), DistanceStorageType.FULL_MATRIX)
    with factory.create_shared_stores() as shared, DistanceStoreFactory.attach_stores(shared.specs) as attached:
        # --- assert -------------------
        assert shared.specs[0].shape == (10, 10)
        np.testing.assert_array_equal(attached[0].matrix, shared.stores[0].matrix)


def test_shared_metrics_that_do_not_preprocess_publish_the_raw_vectors_once():
    """Lazy stores whose metric does not preprocess share one segment of raw vectors; cosine gets its own."""
    # --- arrange ----------------------
    problem = _vector_problem()
    metrics = [L2, DistanceMetric.l1_manhattan(), DistanceMetric.cosine()]
    factory = DistanceStoreFactory(problem, metrics, DistanceStorageType.LAZY, 64 * GIB)

    # --- act --------------------------
    with factory.create_shared_stores() as shared:
        names = [spec.segment_name for spec in shared.specs]

    # --- assert -----------------------
    assert names[0] == names[1] != names[2]


def test_shared_lazy_on_distance_problem_raises():
    """LAZY has no vectors to compute from on a distance-input problem, shared or not."""
    # --- act / assert -----------------
    with pytest.raises(ValueError, match="computes distances from vectors"):
        _factory(_distance_problem("condensed"), DistanceStorageType.LAZY).create_shared_stores()


def test_closing_the_set_releases_the_stores():
    """After close the set holds no stores, so nothing can read a destroyed segment through it."""
    # --- arrange ----------------------
    shared = _factory(_vector_problem(), DistanceStorageType.FULL_MATRIX).create_shared_stores()

    # --- act --------------------------
    shared.close()

    # --- assert -----------------------
    assert shared.stores == []
