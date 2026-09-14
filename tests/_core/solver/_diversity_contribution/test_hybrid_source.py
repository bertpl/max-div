import numpy as np
import pytest

from max_div._core.metrics import (
    DistanceMetric,
    DiversityMetric,
    DiversityObjectiveHybrid,
    DiversityObjectiveSimple,
)
from max_div._core.metrics._distance import DistanceStore
from max_div._core.solver._diversity_contribution import HybridPerItemContributionSource, SeparationTracker

from .helpers import selection_args

# =================================================================================================
#  Fixtures / helpers
# =================================================================================================
N = 5
_VECTORS = np.array([[0.0, 0.0], [1.0, 2.0], [3.0, 1.0], [6.0, 5.0], [10.0, 0.0]], dtype=np.float32)
L1 = DistanceMetric.l1_manhattan()
L2 = DistanceMetric.l2_euclidean()


@pytest.fixture
def term_trackers() -> tuple[SeparationTracker, SeparationTracker]:
    """Two separation trackers over the same points, one per distance metric."""
    return (
        SeparationTracker(DistanceStore.full_matrix_from_vectors(_VECTORS, L1)),
        SeparationTracker(DistanceStore.full_matrix_from_vectors(_VECTORS, L2)),
    )


def _hybrid(*distance_metrics: DistanceMetric) -> DiversityObjectiveHybrid:
    """Return a geomean hybrid of min-separation terms, one per given distance metric."""
    return DiversityObjectiveHybrid(
        tuple(DiversityObjectiveSimple(DiversityMetric.MIN_SEPARATION, metric) for metric in distance_metrics)
    )


def _geomean_of(arrays: list[np.ndarray]) -> np.ndarray:
    """Elementwise geometric mean of the arrays, in float64, as the reference."""
    return np.exp(np.mean(np.log(np.array(arrays, dtype=np.float64)), axis=0))


def _add_to_all(trackers, index: int) -> None:
    """Add point `index` to every tracker, in order."""
    for tracker in trackers:
        tracker.add(np.int32(index))


# =================================================================================================
#  Tests
# =================================================================================================
def test_rejects_fewer_than_two_term_trackers(term_trackers):
    """A single spec's tracker is its own source, so one term tracker is refused."""
    # --- act / assert -----------------
    with pytest.raises(ValueError, match="at least two"):
        HybridPerItemContributionSource(_hybrid(L1, L2), [term_trackers[0]])


@pytest.mark.parametrize("term_positions", [(0, 1), (0, 0, 1)], ids=["two_distances", "repeated_tracker"])
def test_selection_contribution_is_the_objectives_combination(term_trackers, term_positions: tuple[int, ...]):
    """Each read combines the trackers' current arrays by the objective's rule; a repeated tracker counts per term."""
    # --- arrange ----------------------
    trackers = [term_trackers[position] for position in term_positions]
    source = HybridPerItemContributionSource(_hybrid(*[(L1, L2)[position] for position in term_positions]), trackers)
    _add_to_all(term_trackers, 0)
    _add_to_all(term_trackers, 3)
    selected, n_selected = selection_args([0, 3], N)

    # --- act --------------------------
    result = source.contribution_wrt_selection(selected, n_selected)

    # --- assert -----------------------
    per_term = [tracker.contribution_wrt_selection(selected, n_selected) for tracker in trackers]
    np.testing.assert_allclose(result, _geomean_of(per_term), rtol=1e-6)


def test_every_read_reflects_the_term_trackers_current_arrays(term_trackers):
    """The source holds no state: a read after the term trackers changed combines their new arrays."""
    # --- arrange ----------------------
    source = HybridPerItemContributionSource(_hybrid(L1, L2), term_trackers)
    _add_to_all(term_trackers, 0)
    before = source.contribution_wrt_selection(*selection_args([0], N)).copy()
    _add_to_all(term_trackers, 2)
    selected, n_selected = selection_args([0, 2], N)

    # --- act --------------------------
    after = source.contribution_wrt_selection(selected, n_selected)

    # --- assert -----------------------
    expected = _geomean_of([tracker.contribution_wrt_selection(selected, n_selected) for tracker in term_trackers])
    np.testing.assert_allclose(after, expected, rtol=1e-6)
    assert not np.array_equal(after, before)


def test_dataset_contributions_combine_the_term_trackers(term_trackers):
    """The dataset-wide array, and the array returned for a subset of indices, combine the term trackers' arrays."""
    # --- arrange ----------------------
    source = HybridPerItemContributionSource(_hybrid(L1, L2), term_trackers)
    indices = np.array([4, 1], dtype=np.int32)

    # --- act --------------------------
    full_contribution = source.contribution_wrt_dataset
    subset_contribution = source.contribution_wrt_dataset_for(indices)

    # --- assert -----------------------
    expected = _geomean_of([tracker.contribution_wrt_dataset for tracker in term_trackers])
    np.testing.assert_allclose(full_contribution, expected, rtol=1e-6)
    np.testing.assert_allclose(subset_contribution, expected[indices], rtol=1e-6)
    assert source.contribution_wrt_dataset is full_contribution  # combined once
