import numpy as np
import pytest

from max_div._core.metrics import DistanceMetric, DiversityContributionFamily
from max_div._core.metrics._distance import DistanceStore, compute_pdist
from max_div._core.solver._diversity_contribution import (
    MeanDistanceTracker,
    SeparationTracker,
    build_diversity_contribution_tracker,
)


@pytest.mark.parametrize(
    "family, expected_type",
    [
        (DiversityContributionFamily.SEPARATION, SeparationTracker),
        (DiversityContributionFamily.MEAN_DISTANCE, MeanDistanceTracker),
    ],
)
def test_build_diversity_contribution_tracker(family: DiversityContributionFamily, expected_type: type):
    # --- arrange ----------------------
    vectors = np.array([[0.0], [1.0], [3.0]], dtype=np.float32)
    store = DistanceStore.condensed(compute_pdist(vectors, DistanceMetric.L1_MANHATTAN), n=3)

    # --- act --------------------------
    tracker = build_diversity_contribution_tracker(family, store)

    # --- assert -----------------------
    assert type(tracker) is expected_type
