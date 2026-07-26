from __future__ import annotations

from typing import TYPE_CHECKING

from max_div._core.metrics import DiversityContributionFamily

from ._mean_distance import MeanDistanceTracker
from ._separation import SeparationTracker

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray

    from ._base import DiversityContributionTracker


def build_diversity_contribution_tracker(
    family: DiversityContributionFamily, pdist: NDArray[np.float32], n: np.int32
) -> DiversityContributionTracker:
    """Build a fresh (empty-selection) diversity-contribution tracker for the given contribution family.

    :param family: (DiversityContributionFamily) the contribution family to track.
    :param pdist: (np.ndarray[np.float32]) condensed pair-wise distance vector (1D array of size (n*(n-1))//2)
    :param n: (np.int32) number of items
    """
    match family:
        case DiversityContributionFamily.SEPARATION:
            return SeparationTracker(pdist, n)
        case DiversityContributionFamily.MEAN_DISTANCE:
            return MeanDistanceTracker(pdist, n)
