import warnings

import numpy as np
import pytest
from scipy.spatial.distance import squareform

from max_div._core._warnings import DistanceInputWarning
from max_div._core.constraints import Constraint
from max_div._core.constraints.feasibility import FeasibilityStatus
from max_div._core.metrics import DistanceMetric, DiversityMetric
from max_div._core.metrics._distance import compute_pdist
from max_div._core.metrics._distance._store import KIND_CONDENSED, KIND_FULL_MATRIX
from max_div._core.problem import DistanceMaxDivProblem, MaxDivProblem, VectorMaxDivProblem


def test_problem_properties():
    # --- arrange -----------------------------------------
    problem = VectorMaxDivProblem(
        vectors=np.ones((13, 7), dtype=np.float32),
        k=5,
        distance_metric=DistanceMetric.L2_EUCLIDEAN,
        diversity_metric=DiversityMetric.GEOMEAN_SEPARATION,
        constraints=[],
    )

    # --- act ---------------------------------------------
    n = problem.n
    d = problem.d
    m = problem.m

    # --- assert ------------------------------------------
    assert n == 13
    assert d == 7
    assert m == 0


@pytest.mark.parametrize("con_type", ["list[Constraint]", "None"])
def test_problem_new_happy_path(con_type: str):
    # --- arrange -----------------------------------------
    if con_type == "list[Constraint]":
        constraints = [
            Constraint(int_set={1, 2, 3}, min_count=1, max_count=2),
            Constraint(int_set={3, 4, 5, 6, 7}, min_count=2, max_count=3),
        ]
    else:
        constraints = None

    # --- act ---------------------------------------------
    problem = MaxDivProblem.new(
        vectors=np.ones((13, 7), dtype=np.float64),
        k=5,
        distance_metric=DistanceMetric.L1_MANHATTAN,
        diversity_metric=DiversityMetric.APPROX_GEOMEAN_SEPARATION,
        constraints=constraints,
    )

    # --- assert ------------------------------------------
    assert problem.vectors.dtype == np.float32
    assert np.array_equal(problem.vectors, np.ones((13, 7), dtype=np.float64))
    assert problem.k == 5
    assert problem.distance_metric == DistanceMetric.L1_MANHATTAN
    assert problem.diversity_metric == DiversityMetric.APPROX_GEOMEAN_SEPARATION
    if constraints is not None:
        assert problem.m == 2
        assert problem.constraints[0] == Constraint(int_set={1, 2, 3}, min_count=1, max_count=2)
        assert problem.constraints[1] == Constraint(int_set={3, 4, 5, 6, 7}, min_count=2, max_count=3)
    else:
        assert problem.m == 0
        assert len(problem.constraints) == 0


@pytest.mark.parametrize(
    "ndims,n,d,k",
    [
        (1, 10, 5, 5),  # vectors 1D
        (2, 0, 5, 2),  # n too small
        (2, 1, 5, 2),  # n too small
        (2, 2, 5, 2),  # n too small
        (2, 10, 0, 3),  # d too small
        (2, 10, 5, 1),  # k too small
        (2, 10, 5, 11),  # k too large
    ],
)
def test_problem_new_value_error(ndims: int, n: int, d: int, k: int):
    # --- arrange -----------------------------------------
    vectors = np.ones(100, dtype=np.float64) if ndims == 1 else np.ones((n, d), dtype=np.float64)

    # --- act & assert ------------------------------------
    with pytest.raises(ValueError):
        _ = MaxDivProblem.new(vectors, k)


def test_problem_new_cosine_zero_vector_raises():
    """COSINE problems reject all-zero vectors at construction time."""

    # --- arrange -----------------------------------------
    vectors = np.ones((10, 3), dtype=np.float32)
    vectors[4, :] = 0.0

    # --- act & assert ------------------------------------
    with pytest.raises(ValueError, match=r"zero vector.*row 4"):
        _ = MaxDivProblem.new(vectors, k=3, distance_metric=DistanceMetric.COSINE)


def test_problem_new_cosine_non_zero_vectors_ok():
    """COSINE problems accept vector sets without zero rows."""

    # --- arrange -----------------------------------------
    rng = np.random.default_rng(20260713)
    vectors = rng.standard_normal((10, 3)).astype(np.float32)

    # --- act ---------------------------------------------
    problem = MaxDivProblem.new(vectors, k=3, distance_metric=DistanceMetric.COSINE)

    # --- assert ------------------------------------------
    assert problem.distance_metric == DistanceMetric.COSINE


# -------------------------------------------------------------------------
#  from_distances
# -------------------------------------------------------------------------
@pytest.mark.parametrize("form", ["square", "condensed"])
def test_problem_from_distances_happy_path(form: str):
    """from_distances accepts square and condensed input, keeping each in the format provided."""

    # --- arrange -----------------------------------------
    rng = np.random.default_rng(20260713)
    vectors = rng.standard_normal((10, 4)).astype(np.float32)
    condensed = compute_pdist(vectors, DistanceMetric.L2_EUCLIDEAN)
    distances = squareform(condensed) if form == "square" else condensed

    # --- act ---------------------------------------------
    problem = MaxDivProblem.from_distances(distances, k=4)

    # --- assert ------------------------------------------
    assert isinstance(problem, DistanceMaxDivProblem)
    assert problem.n == 10
    assert problem.k == 4
    assert problem.distances.dtype == np.float32
    assert problem.distances.ndim == (2 if form == "square" else 1)
    np.testing.assert_allclose(problem.condensed_distances(), condensed, rtol=1e-6)


def test_problem_from_distances_new_returns_vector_flavor():
    """The two factories return their respective flavors, both subtypes of MaxDivProblem."""

    # --- arrange / act -----------------------------------
    vector_problem = MaxDivProblem.new(np.ones((5, 2), dtype=np.float32), k=2)
    distance_problem = MaxDivProblem.from_distances(np.ones(10, dtype=np.float32), k=2)

    # --- assert ------------------------------------------
    assert isinstance(vector_problem, VectorMaxDivProblem)
    assert isinstance(distance_problem, DistanceMaxDivProblem)
    assert isinstance(vector_problem, MaxDivProblem)
    assert isinstance(distance_problem, MaxDivProblem)


def test_problem_from_distances_condensed_distances_returns_input():
    """For distance-input problems, condensed_distances returns the validated input distances."""

    # --- arrange -----------------------------------------
    condensed = np.arange(1, 11, dtype=np.float32)  # n=5

    # --- act ---------------------------------------------
    problem = MaxDivProblem.from_distances(condensed, k=3)

    # --- assert ------------------------------------------
    assert problem.n == 5
    np.testing.assert_array_equal(problem.condensed_distances(), condensed)


def _mutated_square(i: int, j: int, value: float) -> np.ndarray:
    """Return the reference 5x5 distance matrix with one entry overwritten (unmirrored, so asymmetric if i != j)."""
    square = squareform(np.arange(1, 11, dtype=np.float32))
    square[i, j] = value
    return square


def _mutated_square_symmetric(i: int, j: int, value: float) -> np.ndarray:
    """Return the reference 5x5 distance matrix with one entry pair overwritten symmetrically."""
    square = squareform(np.arange(1, 11, dtype=np.float32))
    square[i, j] = square[j, i] = value
    return square


@pytest.mark.parametrize(
    "case, distances, k",
    [
        ("non_zero_diagonal", _mutated_square(2, 2, 1.0), 3),
        ("negative_asymmetric_raw", _mutated_square(0, 1, -0.5), 3),  # raw value checked before averaging
        ("negative", _mutated_square_symmetric(0, 1, -1.0), 3),
        ("nan", _mutated_square_symmetric(0, 1, np.nan), 3),
        ("inf", _mutated_square_symmetric(0, 1, np.inf), 3),
        ("bad_condensed_length", np.arange(1, 12, dtype=np.float32), 3),  # length 11 is not triangular
        ("non_square", np.ones((5, 4), dtype=np.float32), 3),
        ("3d", np.ones((5, 5, 5), dtype=np.float32), 3),
        ("too_few_items", np.zeros((2, 2), dtype=np.float32), 2),
        ("k_too_large", squareform(np.arange(1, 11, dtype=np.float32)), 6),
    ],
)
def test_problem_from_distances_value_error(case: str, distances: np.ndarray, k: int):
    """from_distances rejects malformed distance input with a ValueError."""

    # --- act & assert ------------------------------------
    with pytest.raises(ValueError):
        _ = MaxDivProblem.from_distances(distances, k=k)


# -------------------------------------------------------------------------
#  from_distances: format retention, zero-copy, and repair
# -------------------------------------------------------------------------
def _reference_square() -> np.ndarray:
    """Return a well-formed 5x5 float32 C-contiguous distance matrix."""
    return np.ascontiguousarray(squareform(np.arange(1, 11, dtype=np.float32)))


@pytest.mark.parametrize("form", ["square", "condensed"])
def test_problem_from_distances_zero_copy_adoption(form: str):
    """Well-formed float32 C-contiguous input is adopted zero-copy, without any warning."""

    # --- arrange -----------------------------------------
    distances = _reference_square() if form == "square" else np.arange(1, 11, dtype=np.float32)

    # --- act ---------------------------------------------
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any warning fails the test
        problem = MaxDivProblem.from_distances(distances, k=3)

    # --- assert ------------------------------------------
    assert problem.distances is distances


@pytest.mark.parametrize("form", ["square", "condensed"])
def test_problem_distance_store_matches_input_format(form: str):
    """The as-given store wraps the retained input directly: square -> full matrix, 1D -> condensed."""

    # --- arrange -----------------------------------------
    distances = _reference_square() if form == "square" else np.arange(1, 11, dtype=np.float32)
    problem = MaxDivProblem.from_distances(distances, k=3)

    # --- act ---------------------------------------------
    store = problem.distance_store()

    # --- assert ------------------------------------------
    if form == "square":
        assert store.kind == KIND_FULL_MATRIX
        assert np.shares_memory(store.matrix, problem.distances)
    else:
        assert store.kind == KIND_CONDENSED
        assert np.shares_memory(store.pdist, problem.distances)
    assert store.n == np.int32(5)


def test_problem_from_distances_asymmetric_repaired_with_warning():
    """Asymmetric square input is symmetrized in place by averaging, disclosed with delta figures."""

    # --- arrange -----------------------------------------
    distances = _reference_square()
    distances[0, 1] = 1.5  # partner [1, 0] stays 1.0 -> mean 1.25

    # --- act ---------------------------------------------
    with pytest.warns(DistanceInputWarning, match=r"max \|delta\| = 5\.000e-01"):
        problem = MaxDivProblem.from_distances(distances, k=3)

    # --- assert ------------------------------------------
    assert problem.distances is distances  # zero-copy adoption, hence in-place repair
    assert distances[0, 1] == distances[1, 0] == np.float32(1.25)
    np.testing.assert_array_equal(distances, distances.T)


def test_problem_from_distances_conversion_copy_warns_and_leaves_input_untouched():
    """Input needing a dtype cast warns about the conversion copy; the user's array is not modified."""

    # --- arrange -----------------------------------------
    distances = _reference_square().astype(np.float64)
    distances[0, 1] = 1.5  # asymmetric, so the repair must land in the cast copy only
    original = distances.copy()

    # --- act ---------------------------------------------
    with pytest.warns(DistanceInputWarning, match="conversion copy"):
        problem = MaxDivProblem.from_distances(distances, k=3)

    # --- assert ------------------------------------------
    np.testing.assert_array_equal(distances, original)  # user's array untouched
    assert problem.distances.dtype == np.float32
    assert problem.distances[0, 1] == problem.distances[1, 0] == np.float32(1.25)


def test_problem_from_distances_condensed_conversion_copy_warns():
    """A condensed vector needing a dtype cast warns about the conversion copy."""

    # --- arrange -----------------------------------------
    distances = np.arange(1, 11, dtype=np.float64)

    # --- act ---------------------------------------------
    with pytest.warns(DistanceInputWarning, match="conversion copy"):
        problem = MaxDivProblem.from_distances(distances, k=3)

    # --- assert ------------------------------------------
    assert problem.distances.dtype == np.float32


def test_problem_from_distances_condensed_negative_raises():
    """Negative values in condensed input are rejected, as for square input."""

    # --- arrange -----------------------------------------
    distances = np.arange(1, 11, dtype=np.float32)
    distances[3] = -0.001

    # --- act & assert ------------------------------------
    with pytest.raises(ValueError, match="non-negative"):
        _ = MaxDivProblem.from_distances(distances, k=3)


def test_problem_square_condensed_distances_extracts_upper_triangle():
    """condensed_distances() on a retained square matrix returns the exact condensed values."""

    # --- arrange -----------------------------------------
    condensed = np.arange(1, 11, dtype=np.float32)
    problem = MaxDivProblem.from_distances(squareform(condensed), k=3)

    # --- act / assert ------------------------------------
    np.testing.assert_array_equal(problem.condensed_distances(), condensed)


# =================================================================================================
#  Feasibility diagnostic
# =================================================================================================
def _problem_with(constraints: list[Constraint], n: int = 20, k: int = 8) -> MaxDivProblem:
    """Build a problem over n random vectors with the given constraints."""
    vectors = np.random.default_rng(0).random((n, 3)).astype(np.float32)
    return MaxDivProblem.new(vectors=vectors, k=k, constraints=constraints)


def test_check_feasibility_proves_a_satisfiable_problem_feasible():
    """A satisfiable problem comes back FEASIBLE, with a selection that satisfies every constraint."""
    # --- arrange -----------------------------------------
    constraints = [
        Constraint(int_set=set(range(10)), min_count=3, max_count=3),
        Constraint(int_set=set(range(10, 20)), min_count=5, max_count=5),
    ]

    # --- act ---------------------------------------------
    report = _problem_with(constraints).check_feasibility()

    # --- assert ------------------------------------------
    assert report.status is FeasibilityStatus.FEASIBLE
    assert report.violation == 0.0
    assert report.constraints_score_ceiling == 1.0
    chosen = set(report.selection.tolist())
    assert all(con.min_count <= len(con.int_set & chosen) <= con.max_count for con in constraints)


def test_check_feasibility_proves_infeasibility_with_a_recheckable_certificate():
    """The multipliers returned must independently reproduce a positive dual value."""
    # --- arrange -----------------------------------------
    constraints = [Constraint(int_set=set(range(20)), min_count=0, max_count=5)]  # every item a member, cap below k

    # --- act ---------------------------------------------
    report = _problem_with(constraints).check_feasibility(thorough=True)

    # --- assert ------------------------------------------
    assert report.status is FeasibilityStatus.INFEASIBLE
    assert report.violation_floor > 0.0
    assert report.constraints_score_ceiling < 1.0

    scores = np.zeros(20)
    for i, con in enumerate(constraints):
        for j in con.int_set:
            scores[j] += report.lam_min[i] - report.lam_max[i]
    mins = np.array([con.min_count for con in constraints])
    maxs = np.array([con.max_count for con in constraints])
    dual_value = float(report.lam_min @ mins - report.lam_max @ maxs - np.sort(scores)[-8:].sum())
    assert dual_value > 0.0


def test_check_feasibility_thorough_tightens_the_floor():
    """The default stops at the first proof; searching harder matures the bound it certifies."""
    # --- arrange -----------------------------------------
    problem = _problem_with([Constraint(int_set=set(range(20)), min_count=0, max_count=5)])

    # --- act ---------------------------------------------
    fast = problem.check_feasibility()
    thorough = problem.check_feasibility(thorough=True)

    # --- assert ------------------------------------------
    assert fast.status is thorough.status is FeasibilityStatus.INFEASIBLE
    assert thorough.violation_floor > fast.violation_floor
    assert thorough.constraints_score_ceiling < fast.constraints_score_ceiling


def test_check_feasibility_reports_unknown_without_claiming_anything():
    """An LP-feasible, integer-infeasible instance must not be reported as either proof."""
    # --- arrange -----------------------------------------
    # Two disjoint 5-cycles, each edge needing at least one of its two endpoints: no certificate
    # exists, and no selection of 5 items covers every edge.
    constraints = [
        Constraint(int_set={cycle * 5 + i, cycle * 5 + (i + 1) % 5}, min_count=1, max_count=2)
        for cycle in range(2)
        for i in range(5)
    ]

    # --- act ---------------------------------------------
    report = _problem_with(constraints, n=10, k=5).check_feasibility()

    # --- assert ------------------------------------------
    assert report.status is FeasibilityStatus.UNKNOWN
    assert not report.is_certified
    assert report.violation_floor == 0.0
    assert report.violation > 0.0


def test_check_feasibility_on_an_unconstrained_problem():
    """With no constraints every selection satisfies them all, so the verdict is feasible."""
    # --- act ---------------------------------------------
    report = _problem_with([]).check_feasibility()

    # --- assert ------------------------------------------
    assert report.status is FeasibilityStatus.FEASIBLE
    assert report.constraints_score_ceiling == 1.0
