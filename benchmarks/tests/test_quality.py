import numpy as np
import pytest
from scipy.spatial.distance import squareform

from benchmarks.common import evaluate_selection, n_constraints_satisfied


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
