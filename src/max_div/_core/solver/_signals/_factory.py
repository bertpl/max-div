from __future__ import annotations

from typing import TYPE_CHECKING

from max_div._core.metrics import DiversitySignalFamily

from ._mean_distance import MeanDistanceTracker
from ._separation import SeparationTracker

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray

    from ._base import DiversitySignalTracker


def build_tracker(family: DiversitySignalFamily, pdist: NDArray[np.float32], n: np.int32) -> DiversitySignalTracker:
    """Build a fresh (empty-selection) diversity-signal tracker for the given signal family.

    :param family: (DiversitySignalFamily) the signal family to track.
    :param pdist: (np.ndarray[np.float32]) condensed pair-wise distance vector (1D array of size (n*(n-1))//2)
    :param n: (np.int32) number of vectors
    """
    match family:
        case DiversitySignalFamily.SEPARATION:
            return SeparationTracker(pdist, n)
        case DiversitySignalFamily.MEAN_DISTANCE:
            return MeanDistanceTracker(pdist, n)
