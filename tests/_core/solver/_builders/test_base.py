import numpy as np
import pytest

from max_div._core.distance_storage import DistanceProblemDistanceStoreFactory, VectorProblemDistanceStoreFactory
from max_div._core.metrics import DistanceMetric, DiversityMetric, HybridDiversityMetric
from max_div._core.problem import MaxDivProblem
from max_div._core.solver._builders import MaxDivSolverBuilder, ParallelMaxDivSolverBuilder, SolverBuilderBase


def _problem() -> MaxDivProblem:
    """Return a problem with n=20 and k=5."""
    return MaxDivProblem.new(np.random.default_rng(20260926).random((20, 3)).astype(np.float32), k=5)


# ==================================================================================================
#  with_initial_selection
# ==================================================================================================
@pytest.mark.parametrize("builder_class", [MaxDivSolverBuilder, ParallelMaxDivSolverBuilder])
@pytest.mark.parametrize(
    "indices, message",
    [
        ([0, 1, 2, 3], "exactly k=5 indices; got 4"),
        ([0, 1, 2, 3, 20], "Index 20 lies outside the population of n=20 items"),
        ([0, 1, 2, 3, 3], "Index 3 appears more than once"),
    ],
)
def test_with_initial_selection_rejects_a_selection_that_does_not_fit_when_called(
    builder_class: type[SolverBuilderBase], indices: list[int], message: str
):
    """The builder knows n and k, so every check runs when the selection is given, not when the solve starts."""
    # --- act / assert -----------------
    with pytest.raises(ValueError, match=message):
        builder_class(_problem()).with_initial_selection(indices)


# ==================================================================================================
#  Distance store factory
# ==================================================================================================
@pytest.mark.parametrize("builder_class", [MaxDivSolverBuilder, ParallelMaxDivSolverBuilder])
def test_a_distance_named_twice_gets_one_store(builder_class: type[SolverBuilderBase]):
    """A hybrid that reads the problem's own distance through a bare term and through `over` builds 1 store for it."""
    # --- arrange ----------------------
    l2 = DistanceMetric.l2_euclidean()
    axis_0 = DistanceMetric.along_axis(0)
    hybrid = HybridDiversityMetric.geomean_of(
        DiversityMetric.MIN_SEPARATION,  # a bare term reads the problem's own distance metric, which is L2
        DiversityMetric.MIN_SEPARATION.over(l2),
        DiversityMetric.MIN_SEPARATION.over(axis_0),
    )
    vectors = np.random.default_rng(20261002).random((20, 3)).astype(np.float32)
    problem = MaxDivProblem.new(vectors, k=5, distance_metric=l2, diversity_metric=hybrid)

    # --- act --------------------------
    factory, distance_storage = builder_class(problem)._store_factory()

    # --- assert -----------------------
    assert isinstance(factory, VectorProblemDistanceStoreFactory)
    assert factory.distance_metrics == (l2, axis_0)
    assert [distance_metric for distance_metric, _ in distance_storage.per_store] == [l2, axis_0]
    assert len(factory.create_stores()) == 2


@pytest.mark.parametrize("builder_class", [MaxDivSolverBuilder, ParallelMaxDivSolverBuilder])
def test_a_distance_problem_gets_the_factory_of_its_flavor(builder_class: type[SolverBuilderBase]):
    """The builder asks the problem for its factory, so a distance-input problem gets the one over given distances."""
    # --- arrange ----------------------
    problem = MaxDivProblem.from_distances(_problem().full_matrix(), k=5)

    # --- act --------------------------
    factory, distance_storage = builder_class(problem)._store_factory()

    # --- assert -----------------------
    assert isinstance(factory, DistanceProblemDistanceStoreFactory)
    assert [distance_metric for distance_metric, _ in distance_storage.per_store] == [None]
