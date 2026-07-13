import numpy as np
import pytest
from scipy.spatial.distance import squareform

from max_div._core.constraints import Constraint
from max_div._core.metrics import DistanceMetric, DiversityMetric
from max_div._core.metrics._distance import compute_pdist
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
    """from_distances accepts square and condensed input and normalizes both to condensed float32."""

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
    assert problem.pdist.dtype == np.float32
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
        ("asymmetric", _mutated_square(0, 1, 99.0), 3),
        ("non_zero_diagonal", _mutated_square(2, 2, 1.0), 3),
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
