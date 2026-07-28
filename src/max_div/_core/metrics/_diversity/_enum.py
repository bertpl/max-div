from collections.abc import Callable
from enum import StrEnum
from functools import cached_property

import numpy as np
from numpy.typing import NDArray


class DiversityContributionFamily(StrEnum):
    """Enum for the per-point diversity-contribution families that diversity metrics consume.

    Members
    -------

        - SEPARATION:     contribution = distance to the nearest selected item
        - MEAN_DISTANCE:  contribution = mean distance to the selected items
    """

    SEPARATION = "SEPARATION"
    MEAN_DISTANCE = "MEAN_DISTANCE"


class DiversityMetric(StrEnum):
    """Enum for different diversity metrics.

    Members
    -------

        - MIN_SEPARATION:             Minimum separation of all selected items
        - MEAN_SEPARATION:            Arithmetic mean separation of all selected items
        - GEOMEAN_SEPARATION:         Geometric mean separation of all selected items
        - APPROX_GEOMEAN_SEPARATION:  Approximate geometric mean separation of all selected items
                                          (uses faster, but still smooth approximations of log(.) and exp(.))
        - NON_ZERO_SEPARATION_FRAC:   Fraction of separation values that are non-zero
        - MEAN_PAIRWISE_DISTANCE:     Mean distance over all pairs of selected items
                                          (the classical max-sum diversity objective)
    """

    MIN_SEPARATION = "MIN_SEPARATION"
    MEAN_SEPARATION = "MEAN_SEPARATION"
    GEOMEAN_SEPARATION = "GEOMEAN_SEPARATION"
    APPROX_GEOMEAN_SEPARATION = "APPROX_GEOMEAN_SEPARATION"
    NON_ZERO_SEPARATION_FRAC = "NON_ZERO_SEPARATION_FRAC"
    MEAN_PAIRWISE_DISTANCE = "MEAN_PAIRWISE_DISTANCE"

    @cached_property
    def contribution_family(self) -> DiversityContributionFamily:
        """Return the diversity-contribution family whose per-point values this metric consumes."""
        if self == DiversityMetric.MEAN_PAIRWISE_DISTANCE:
            return DiversityContributionFamily.MEAN_DISTANCE
        return DiversityContributionFamily.SEPARATION

    @cached_property
    def _f(self) -> Callable[[NDArray[np.float32]], np.float32]:
        """Get the numba computation function for this metric."""
        # Import here to avoid circular dependency
        from ._numba import (
            approx_geomean_separation,
            geomean_separation,
            mean_pairwise_distance,
            mean_separation,
            min_separation,
            non_zero_separation_frac,
        )

        _functions = {
            DiversityMetric.MIN_SEPARATION: min_separation,
            DiversityMetric.MEAN_SEPARATION: mean_separation,
            DiversityMetric.GEOMEAN_SEPARATION: geomean_separation,
            DiversityMetric.APPROX_GEOMEAN_SEPARATION: approx_geomean_separation,
            DiversityMetric.NON_ZERO_SEPARATION_FRAC: non_zero_separation_frac,
            DiversityMetric.MEAN_PAIRWISE_DISTANCE: mean_pairwise_distance,
        }
        return _functions[self]

    def compute(self, contribution_values: NDArray[np.float32]) -> np.float32:
        """Compute diversity metric given the selected items' per-point diversity-contribution values.

        Parameters
        ----------
        contribution_values : NDArray[np.float32]
            Per-point diversity-contribution values of the selected items (for separation-family
            metrics: each item's separation wrt the others in the selection).

        Returns:
        -------
        np.float32
            The computed diversity score.
        """
        if contribution_values.size < 2:
            # we can only meaningfully compute diversity metrics with at least 2 contribution values
            return np.float32(0.0)
        return self._f(contribution_values)
