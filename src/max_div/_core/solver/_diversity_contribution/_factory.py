from __future__ import annotations

from typing import TYPE_CHECKING

from max_div._core.metrics import DiversityContributionFamily

from ._mean_distance import MeanDistanceTracker
from ._separation import SeparationTracker

if TYPE_CHECKING:
    from max_div._core.metrics._distance import DistanceStore

    from ._base import DiversityContributionTracker


def build_diversity_contribution_tracker(
    family: DiversityContributionFamily, store: DistanceStore
) -> DiversityContributionTracker:
    """Build a fresh (empty-selection) diversity-contribution tracker for the given contribution family.

    Args:
        family: (DiversityContributionFamily) the contribution family to track.
        store: (DistanceStore) pairwise-distance storage the tracker reads from.
    """
    match family:
        case DiversityContributionFamily.SEPARATION:
            return SeparationTracker(store)
        case DiversityContributionFamily.MEAN_DISTANCE:
            return MeanDistanceTracker(store)
