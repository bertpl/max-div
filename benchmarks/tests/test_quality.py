import numpy as np
import pytest
from scipy.spatial.distance import squareform

from benchmarks.common import evaluate_selection, n_constraints_satisfied
from max_div.metrics import DistanceMetric
from max_div.problem import MaxDivProblem


def test_evaluate_selection_matches_squareform_oracle(small_problem):
    # every published quality number flows through evaluate_selection, so its
    # condensed-index extraction is verified against scipy's squareform
    # --- arrange -----------------------------------------
    i_selected = np.array([0, 7, 13, 21, 29], dtype=np.int64)
    dist = squareform(small_problem.condensed_distances().astype(np.float64))
    sub = dist[np.ix_(i_selected, i_selected)]
    k = len(i_selected)
    separations = np.where(~np.eye(k, dtype=bool), sub, np.inf).min(axis=1)
    mean_dists = sub.sum(axis=1) / (k - 1)

    # --- act ---------------------------------------------
    quality = evaluate_selection(small_problem, i_selected)

    # --- assert ------------------------------------------
    assert quality["MIN_SEPARATION"] == pytest.approx(separations.min(), rel=1e-6)
    assert quality["MEAN_SEPARATION"] == pytest.approx(separations.mean(), rel=1e-6)
    assert quality["GEOMEAN_SEPARATION"] == pytest.approx(np.exp(np.log(separations).mean()), rel=1e-5)
    assert quality["MEAN_PAIRWISE_DISTANCE"] == pytest.approx(mean_dists.mean(), rel=1e-6)


@pytest.mark.parametrize(
    "distance_metric",
    [DistanceMetric.L1_MANHATTAN, DistanceMetric.L2S_EUCLIDEAN_SQUARED, DistanceMetric.COSINE],
)
def test_evaluate_selection_respects_distance_metric(distance_metric):
    # the vector fast path must score under the problem's own distance metric,
    # verified against the full condensed vector it deliberately avoids computing
    # --- arrange -----------------------------------------
    rng = np.random.default_rng(7)
    problem = MaxDivProblem.new(
        vectors=rng.random((25, 4)).astype(np.float32) + 0.1, k=5, distance_metric=distance_metric
    )
    i_selected = np.array([1, 6, 11, 19, 24], dtype=np.int64)
    dist = squareform(problem.condensed_distances().astype(np.float64))
    sub = dist[np.ix_(i_selected, i_selected)]
    expected_min = np.where(~np.eye(5, dtype=bool), sub, np.inf).min(axis=1).min()

    # --- act ---------------------------------------------
    quality = evaluate_selection(problem, i_selected)

    # --- assert ------------------------------------------
    assert quality["MIN_SEPARATION"] == pytest.approx(expected_min, rel=1e-6)


def test_evaluate_selection_distance_flavor_matches_vector_flavor():
    # a from_distances problem must score identically to the vector problem it was built from
    # --- arrange -----------------------------------------
    rng = np.random.default_rng(8)
    vector_problem = MaxDivProblem.new(vectors=rng.random((30, 3)).astype(np.float32), k=5)
    distance_problem = MaxDivProblem.from_distances(vector_problem.condensed_distances(), k=5)
    i_selected = np.array([2, 9, 14, 22, 28], dtype=np.int64)

    # --- act ---------------------------------------------
    quality_vec = evaluate_selection(vector_problem, i_selected)
    quality_dist = evaluate_selection(distance_problem, i_selected)

    # --- assert ------------------------------------------
    assert quality_vec == pytest.approx(quality_dist, rel=1e-6)


@pytest.mark.parametrize(
    "i_selected, expected_error",
    [
        (np.array([3]), "at least 2 items"),
        (np.array([3, 3, 5]), "duplicate"),
    ],
)
def test_evaluate_selection_rejects_invalid_selections(small_problem, i_selected, expected_error):
    # --- act / assert ------------------------------------
    with pytest.raises(ValueError, match=expected_error):
        evaluate_selection(small_problem, i_selected)


def test_n_constraints_satisfied(small_constrained_problem):
    # groups are {0..14} needing 2-3 picks and {15..29} needing 2-4 picks
    # --- arrange -----------------------------------------
    both_ok = np.array([0, 1, 2, 15, 16, 17], dtype=np.int64)
    first_violated = np.array([0, 1, 2, 3, 15, 16], dtype=np.int64)  # 4 > max_count=3
    both_violated = np.array([0, 1, 2, 3, 4, 15], dtype=np.int64)  # 5 > 3 and 1 < 2

    # --- act / assert ------------------------------------
    assert n_constraints_satisfied(small_constrained_problem, both_ok) == 2
    assert n_constraints_satisfied(small_constrained_problem, first_violated) == 1
    assert n_constraints_satisfied(small_constrained_problem, both_violated) == 0
