import numpy as np
import pytest
from scipy.spatial.distance import squareform

from max_div._core.constraints import Constraint
from max_div._core.metrics import DistanceMetric, DiversityMetric
from max_div._core.metrics._distance import compute_full_matrix, get_distance
from max_div._core.metrics._distance._store import KIND_FULL_MATRIX, KIND_LAZY
from max_div._core.problem import MaxDivProblem
from max_div._core.solver import MaxDivSolverBuilder, SolverPreset, Verbosity
from max_div._core.solver._distance_storage import (
    DistanceStorageType,
    attached_distance_store,
    build_distance_store,
    build_shared_distance_store,
    select_distance_storage_type,
)
from max_div._core.solver._duration import iterations

# =================================================================================================
#  Fixtures / helpers
# =================================================================================================
GIB = 2**30


def _vector_problem(n: int = 10) -> MaxDivProblem:
    rng = np.random.default_rng(20260731)
    return MaxDivProblem.new(rng.random((n, 3)).astype(np.float32), k=3)


def _distance_problem(form: str) -> MaxDivProblem:
    matrix = compute_full_matrix(_vector_problem().vectors, DistanceMetric.l2_euclidean())  # ty: ignore[unresolved-attribute]
    distances = matrix if form == "square" else squareform(matrix, checks=False)
    return MaxDivProblem.from_distances(distances, k=3)


def _all_pairs(store, n: int) -> list[float]:
    """Return every (i, j) distance the store reports, self-pairs included."""
    return [get_distance(store, np.int32(i), np.int32(j)) for i in range(n) for j in range(n)]


# =================================================================================================
#  Resolution policy
# =================================================================================================
@pytest.mark.parametrize("storage", [DistanceStorageType.FULL_MATRIX, DistanceStorageType.LAZY])
def test_resolve_explicit_choice_passes_through(storage: DistanceStorageType):
    # --- act / assert -----------------
    assert select_distance_storage_type(_vector_problem(), storage, 64 * GIB) == storage


@pytest.mark.parametrize(
    "n, total_memory, expected",
    [
        (10, 64 * GIB, DistanceStorageType.FULL_MATRIX),  # tiny problem: matrix always fits
        (10, None, DistanceStorageType.LAZY),  # probe failed: the one backend that cannot page
        (50_000, 32 * GIB, DistanceStorageType.FULL_MATRIX),  # 10.0 GiB matrix <= 1/3 of 32 GiB
        (50_000, 16 * GIB, DistanceStorageType.LAZY),  # matrix over budget
    ],
)
def test_resolve_auto_vector_full_matrix_when_it_fits(n: int, total_memory: int | None, expected: DistanceStorageType):
    """AUTO on vector problems: the full matrix when its bytes fit within a third of total RAM, else lazy."""

    # --- arrange ----------------------
    problem = _vector_problem() if n == 10 else _stub_vector_problem(n)

    # --- act --------------------------
    resolved = select_distance_storage_type(problem, DistanceStorageType.AUTO, total_memory)

    # --- assert -----------------------
    assert resolved == expected


def _stub_vector_problem(n: int):
    """A stand-in exposing only what the resolution policy reads (isinstance + n), without allocations."""
    from max_div._core.problem import VectorMaxDivProblem

    stub = object.__new__(VectorMaxDivProblem)
    object.__setattr__(stub, "vectors", np.zeros((n, 0), dtype=np.float32))
    return stub


@pytest.mark.parametrize("form", ["condensed", "square"])
def test_resolve_auto_distance_problem_is_the_full_matrix(form: str):
    """AUTO on distance-input problems resolves to the full matrix whatever the input form, ignoring memory."""

    # --- act / assert -----------------
    assert (
        select_distance_storage_type(_distance_problem(form), DistanceStorageType.AUTO, None)
        == DistanceStorageType.FULL_MATRIX
    )


# =================================================================================================
#  Store construction
# =================================================================================================
@pytest.mark.parametrize(
    "storage, expected_kind",
    [(DistanceStorageType.FULL_MATRIX, KIND_FULL_MATRIX), (DistanceStorageType.LAZY, KIND_LAZY)],
)
def test_build_distance_store_vector_problem(storage: DistanceStorageType, expected_kind: np.int32):
    # --- act --------------------------
    store = build_distance_store(_vector_problem(), storage)

    # --- assert -----------------------
    assert store.kind == expected_kind
    assert store.n == np.int32(10)


def test_build_distance_store_square_input_zero_copy():
    """FULL_MATRIX on a square-input problem adopts the retained matrix without copying."""

    # --- arrange ----------------------
    problem = _distance_problem("square")

    # --- act --------------------------
    store = build_distance_store(problem, DistanceStorageType.FULL_MATRIX)

    # --- assert -----------------------
    assert np.shares_memory(store.matrix, problem.distances)  # ty: ignore[unresolved-attribute]


def test_build_distance_store_condensed_input_expands_to_the_square_matrix():
    """FULL_MATRIX on a condensed-input problem expands into a fresh symmetric matrix holding the same values."""

    # --- arrange ----------------------
    condensed_problem = _distance_problem("condensed")
    square_problem = _distance_problem("square")

    # --- act --------------------------
    store = build_distance_store(condensed_problem, DistanceStorageType.FULL_MATRIX)

    # --- assert -----------------------
    assert store.kind == KIND_FULL_MATRIX
    np.testing.assert_array_equal(store.matrix, square_problem.distances)  # ty: ignore[unresolved-attribute]


def test_build_distance_store_lazy_on_distance_problem_raises():
    # --- act / assert -----------------
    with pytest.raises(ValueError, match="from vectors"):
        build_distance_store(_distance_problem("condensed"), DistanceStorageType.LAZY)


def test_build_distance_store_unresolved_raises():
    """AUTO is not a buildable backend; passing it unresolved is rejected."""
    # --- act / assert -----------------
    with pytest.raises(ValueError, match="resolved"):
        build_distance_store(_vector_problem(), DistanceStorageType.AUTO)


def test_build_distance_store_infeasible_raises_early():
    """A full matrix that cannot fit in physical RAM is rejected with the remedy named."""

    # --- arrange ----------------------
    stub = _stub_vector_problem(2_000_000)  # full matrix would need ~16 TiB

    # --- act / assert -----------------
    with pytest.raises(ValueError, match="LAZY"):
        build_distance_store(stub, DistanceStorageType.FULL_MATRIX)


# =================================================================================================
#  Per-backend solve behavior
# =================================================================================================
# What a backend guarantees is that it solves the same problem as well, not that it picks the
# same items: distances may differ in their last bits between backends, the search is chaotic,
# and one flipped comparison sends it down a different path to an equally good answer.  These
# assert the property that survives that — quality, and feasibility on constrained problems.
@pytest.mark.parametrize("storage", [DistanceStorageType.FULL_MATRIX, DistanceStorageType.LAZY])
def test_every_backend_reaches_equivalent_quality(storage: DistanceStorageType):
    """Each backend solves an unconstrained problem to within a small margin of the others."""

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
def test_every_backend_reaches_feasibility(storage: DistanceStorageType):
    """Each backend satisfies a reachable count constraint, whatever items it ends up choosing."""

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
def test_build_shared_distance_store_matches_the_unshared_build(storage: DistanceStorageType):
    """A store built into shared memory holds bit-identical distances to the ordinary build."""
    # --- arrange ----------------------
    problem = _vector_problem()
    expected = build_distance_store(problem, storage)

    # --- act --------------------------
    with build_shared_distance_store(problem, storage) as shared, attached_distance_store(shared.spec) as attached:
        read = _all_pairs(attached, problem.n)

    # --- assert -----------------------
    assert read == _all_pairs(expected, problem.n)


@pytest.mark.parametrize("form", ["square", "condensed"])
def test_build_shared_distance_store_holds_distance_input(form: str):
    """Distance-input problems land in the segment whatever their form, since the bytes must live there."""
    # --- arrange ----------------------
    problem = _distance_problem(form)
    resolved = select_distance_storage_type(problem, DistanceStorageType.AUTO, 64 * GIB)
    expected = build_distance_store(problem, resolved)

    # --- act --------------------------
    with build_shared_distance_store(problem, resolved) as shared:
        read = _all_pairs(shared.store, problem.n)

    # --- assert -----------------------
    assert read == _all_pairs(expected, problem.n)


def test_build_shared_distance_store_rejects_lazy_on_distance_input():
    """LAZY has no vectors to compute from on a distance-input problem, shared or not."""
    # --- arrange / act / assert -------
    with pytest.raises(ValueError, match="computes distances from vectors"):
        build_shared_distance_store(_distance_problem("condensed"), DistanceStorageType.LAZY)


def test_build_shared_distance_store_builds_into_the_segment():
    """The computed matrix lands in the segment itself, with no full-size copy in between."""
    # --- arrange / act ----------------
    with build_shared_distance_store(_vector_problem(), DistanceStorageType.FULL_MATRIX) as shared:
        # --- assert -------------------
        assert np.shares_memory(shared.store.matrix, shared.buffer)


def test_build_shared_distance_store_expands_condensed_input_into_the_segment():
    """A condensed-input problem expands straight into the shared segment."""
    # --- arrange ----------------------
    problem = _distance_problem("condensed")
    expected = build_distance_store(problem, DistanceStorageType.FULL_MATRIX)

    # --- act --------------------------
    with build_shared_distance_store(problem, DistanceStorageType.FULL_MATRIX) as shared:
        matrix = np.array(shared.store.matrix)
        shares_segment = np.shares_memory(shared.store.matrix, shared.buffer)

    # --- assert -----------------------
    np.testing.assert_array_equal(matrix, expected.matrix)
    assert shares_segment
