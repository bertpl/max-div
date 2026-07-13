import numpy as np
import pytest

from max_div._core.metrics import DistanceMetric, DiversitySignalFamily
from max_div._core.metrics._distance import compute_pdist
from max_div._core.solver._signals import MeanDistanceTracker, SeparationTracker, build_tracker


@pytest.mark.parametrize(
    "family, expected_type",
    [
        (DiversitySignalFamily.SEPARATION, SeparationTracker),
        (DiversitySignalFamily.MEAN_DISTANCE, MeanDistanceTracker),
    ],
)
def test_build_tracker(family: DiversitySignalFamily, expected_type: type):
    # --- arrange -----------------------------------------
    vectors = np.array([[0.0], [1.0], [3.0]], dtype=np.float32)
    pdist = compute_pdist(vectors, DistanceMetric.L1_MANHATTAN)

    # --- act ---------------------------------------------
    tracker = build_tracker(family, pdist, np.int32(3))

    # --- assert ------------------------------------------
    assert type(tracker) is expected_type
