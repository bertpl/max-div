import numpy as np
import pytest

from max_div.solver import DistanceMetric, DiversityMetric, MaxDivProblem
from max_div.solver._constraints import Constraint, Constraints


def test_problem_properties():
    # --- arrange -----------------------------------------
    problem = MaxDivProblem(
        vectors=np.ones((13, 7), dtype=np.float32),
        k=5,
        distance_metric=DistanceMetric.L2_EUCLIDEAN,
        diversity_metric=DiversityMetric.geomean_separation(),
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


@pytest.mark.parametrize("con_type", ["Constraints", "list[Constraint]", "None"])
def test_problem_new_happy_path(con_type: str):
    # --- arrange -----------------------------------------
    if con_type == "Constraints":
        constraints = Constraints()
        constraints.add(indices={1, 2, 3}, min_count=1, max_count=2)
        constraints.add(indices={3, 4, 5, 6, 7}, min_count=2, max_count=3)
    elif con_type == "list[Constraint]":
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
        diversity_metric=DiversityMetric.approx_geomean_separation(),
        constraints=constraints,
    )

    # --- assert ------------------------------------------
    assert problem.vectors.dtype == np.float32
    assert np.array_equal(problem.vectors, np.ones((13, 7), dtype=np.float64))
    assert problem.k == 5
    assert problem.distance_metric == DistanceMetric.L1_MANHATTAN
    assert problem.diversity_metric == DiversityMetric.approx_geomean_separation()
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
    if ndims == 1:
        vectors = np.ones(100, dtype=np.float64)
    else:
        vectors = np.ones((n, d), dtype=np.float64)

    # --- act & assert ------------------------------------
    with pytest.raises(ValueError):
        _ = MaxDivProblem.new(vectors, k)
