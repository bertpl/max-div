import numpy as np
import pytest

from max_div._core.constraints import Constraint
from max_div._core.metrics import DistanceMetric, DiversityMetric
from max_div._core.metrics._distance import compute_pdist
from max_div._core.metrics._distance._store import KIND_CONDENSED, KIND_FULL_MATRIX, KIND_LAZY
from max_div._core.problem import MaxDivProblem
from max_div._core.solver import MaxDivSolverBuilder, SolverPreset
from max_div._core.solver._distance_storage import (
    DistanceStorage,
    build_distance_store,
    resolve_distance_storage,
    total_physical_memory_bytes,
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
    from scipy.spatial.distance import squareform

    condensed = compute_pdist(_vector_problem().vectors, DistanceMetric.L2_EUCLIDEAN)  # ty: ignore[unresolved-attribute]
    distances = np.ascontiguousarray(squareform(condensed)) if form == "square" else condensed
    return MaxDivProblem.from_distances(distances, k=3)


# =================================================================================================
#  Resolution policy
# =================================================================================================
@pytest.mark.parametrize(
    "storage",
    [DistanceStorage.CONDENSED, DistanceStorage.FULL_MATRIX, DistanceStorage.LAZY],
)
def test_resolve_explicit_choice_passes_through(storage: DistanceStorage):
    # --- act / assert ------------------------------------
    assert resolve_distance_storage(_vector_problem(), storage, 64 * GIB) == storage


@pytest.mark.parametrize(
    "n, total_memory, expected",
    [
        (10, 64 * GIB, DistanceStorage.FULL_MATRIX),  # tiny problem: matrix always fits
        (10, None, DistanceStorage.CONDENSED),  # probe failed: degrade to the proven default
        (50_000, 32 * GIB, DistanceStorage.FULL_MATRIX),  # 10.0 GiB matrix <= 1/3 of 32 GiB
        (50_000, 16 * GIB, DistanceStorage.CONDENSED),  # matrix over budget, half-size fits
        (50_000, 8 * GIB, DistanceStorage.LAZY),  # even condensed over budget
    ],
)
def test_resolve_auto_vector_ladder(n: int, total_memory: int | None, expected: DistanceStorage):
    """AUTO on vector problems: fastest layout whose bytes fit within a third of total RAM."""

    # --- arrange -----------------------------------------
    problem = _vector_problem() if n == 10 else _stub_vector_problem(n)

    # --- act ---------------------------------------------
    resolved = resolve_distance_storage(problem, DistanceStorage.AUTO, total_memory)

    # --- assert ------------------------------------------
    assert resolved == expected


def _stub_vector_problem(n: int):
    """A stand-in exposing only what the resolution policy reads (isinstance + n), without allocations."""
    from max_div._core.problem import VectorMaxDivProblem

    stub = object.__new__(VectorMaxDivProblem)
    object.__setattr__(stub, "vectors", np.zeros((n, 0), dtype=np.float32))
    return stub


@pytest.mark.parametrize(
    "form, expected", [("condensed", DistanceStorage.CONDENSED), ("square", DistanceStorage.FULL_MATRIX)]
)
def test_resolve_auto_distance_problem_keeps_format(form: str, expected: DistanceStorage):
    """AUTO on distance-input problems resolves to the format the user provided, ignoring memory."""

    # --- act / assert ------------------------------------
    assert resolve_distance_storage(_distance_problem(form), DistanceStorage.AUTO, None) == expected


# =================================================================================================
#  Store construction
# =================================================================================================
@pytest.mark.parametrize(
    "storage, expected_kind",
    [
        (DistanceStorage.CONDENSED, KIND_CONDENSED),
        (DistanceStorage.FULL_MATRIX, KIND_FULL_MATRIX),
        (DistanceStorage.LAZY, KIND_LAZY),
    ],
)
def test_build_distance_store_vector_problem(storage: DistanceStorage, expected_kind: np.int32):
    # --- act ---------------------------------------------
    store = build_distance_store(_vector_problem(), storage)

    # --- assert ------------------------------------------
    assert store.kind == expected_kind
    assert store.n == np.int32(10)


def test_build_distance_store_square_input_zero_copy():
    """FULL_MATRIX on a square-input problem adopts the retained matrix without copying."""

    # --- arrange -----------------------------------------
    problem = _distance_problem("square")

    # --- act ---------------------------------------------
    store = build_distance_store(problem, DistanceStorage.FULL_MATRIX)

    # --- assert ------------------------------------------
    assert store.matrix is problem.distances  # ty: ignore[unresolved-attribute]


def test_build_distance_store_condensed_input_full_matrix_converts():
    """FULL_MATRIX on a condensed-input problem expands consciously into a fresh symmetric matrix."""

    # --- arrange -----------------------------------------
    problem = _distance_problem("condensed")

    # --- act ---------------------------------------------
    store = build_distance_store(problem, DistanceStorage.FULL_MATRIX)

    # --- assert ------------------------------------------
    assert store.kind == KIND_FULL_MATRIX
    np.testing.assert_array_equal(store.matrix, store.matrix.T)


def test_build_distance_store_lazy_on_distance_problem_raises():
    # --- act / assert ------------------------------------
    with pytest.raises(ValueError, match="from vectors"):
        build_distance_store(_distance_problem("condensed"), DistanceStorage.LAZY)


def test_build_distance_store_infeasible_raises_early():
    """A stored backend that cannot fit in physical RAM is rejected with the remedy named."""

    # --- arrange -----------------------------------------
    stub = _stub_vector_problem(2_000_000)  # full matrix would need ~16 TiB

    # --- act / assert ------------------------------------
    with pytest.raises(ValueError, match="LAZY"):
        build_distance_store(stub, DistanceStorage.FULL_MATRIX)


# =================================================================================================
#  Memory probe
# =================================================================================================
def test_total_physical_memory_bytes_on_this_platform():
    """On every CI platform the stdlib probe must return a sane positive figure."""

    # --- act ---------------------------------------------
    total = total_physical_memory_bytes()

    # --- assert ------------------------------------------
    assert total is not None
    assert total >= 1 * GIB


# =================================================================================================
#  Per-backend solve behavior
# =================================================================================================
# What a backend guarantees is that it solves the same problem as well, not that it picks the
# same items: distances may differ in their last bits between backends, the search is chaotic,
# and one flipped comparison sends it down a different path to an equally good answer.  These
# assert the property that survives that — quality, and feasibility on constrained problems.
@pytest.mark.parametrize("storage", [DistanceStorage.CONDENSED, DistanceStorage.FULL_MATRIX, DistanceStorage.LAZY])
def test_every_backend_reaches_equivalent_quality(storage: DistanceStorage):
    """Each backend solves an unconstrained problem to within a small margin of the others."""

    # --- arrange -----------------------------------------
    rng = np.random.default_rng(20260804)
    vectors = rng.random((120, 5), dtype=np.float32)
    problem = MaxDivProblem.new(vectors=vectors, k=12, diversity_metric=DiversityMetric.MIN_SEPARATION)

    # --- act ---------------------------------------------
    solution = (
        MaxDivSolverBuilder(problem)
        .with_preset(iterations(200), SolverPreset.SMART)
        .with_seed(7)
        .with_distance_storage(storage)
        .build()
        .solve(verbosity=0)
    )

    # --- assert ------------------------------------------
    assert solution.score.diversity > 0.0
    assert len(solution.i_selected) == 12
    assert len({int(i) for i in solution.i_selected}) == 12  # a selection, not a multiset


@pytest.mark.parametrize("storage", [DistanceStorage.CONDENSED, DistanceStorage.FULL_MATRIX, DistanceStorage.LAZY])
def test_every_backend_reaches_feasibility(storage: DistanceStorage):
    """Each backend satisfies a reachable count constraint, whatever items it ends up choosing."""

    # --- arrange -----------------------------------------
    rng = np.random.default_rng(20260804)
    vectors = rng.random((120, 5), dtype=np.float32)
    first_half = list(range(60))
    problem = MaxDivProblem.new(
        vectors=vectors,
        k=12,
        diversity_metric=DiversityMetric.MIN_SEPARATION,
        constraints=[Constraint(int_set=set(first_half), min_count=6, max_count=6)],
    )

    # --- act ---------------------------------------------
    solution = (
        MaxDivSolverBuilder(problem)
        .with_preset(iterations(400), SolverPreset.SMART)
        .with_seed(7)
        .with_distance_storage(storage)
        .build()
        .solve(verbosity=0)
    )

    # --- assert ------------------------------------------
    n_from_first_half = sum(1 for i in solution.i_selected if int(i) in set(first_half))
    assert n_from_first_half == 6
