from collections.abc import Callable
from enum import StrEnum
from functools import cached_property

import numpy as np
from numpy.typing import NDArray


class DiversityMetric(StrEnum):
    """
    Enum for different diversity metrics.

    Members
    -------

        - MIN_SEPARATION:             Minimum separation of all selected vectors
        - MEAN_SEPARATION:            Arithmetic mean separation of all selected vectors
        - GEOMEAN_SEPARATION:         Geometric mean separation of all selected vectors
        - APPROX_GEOMEAN_SEPARATION:  Approximate geometric mean separation of all selected vectors
                                          (uses faster, but still smooth approximations of log(.) and exp(.))
        - NON_ZERO_SEPARATION_FRAC:   Fraction of separation values that are non-zero
    """

    MIN_SEPARATION = "MIN_SEPARATION"
    MEAN_SEPARATION = "MEAN_SEPARATION"
    GEOMEAN_SEPARATION = "GEOMEAN_SEPARATION"
    APPROX_GEOMEAN_SEPARATION = "APPROX_GEOMEAN_SEPARATION"
    NON_ZERO_SEPARATION_FRAC = "NON_ZERO_SEPARATION_FRAC"

    @cached_property
    def _f(self) -> Callable[[NDArray[np.float32]], np.float32]:
        """Get the numba computation function for this metric."""
        # Import here to avoid circular dependency
        from ._numba import (
            approx_geomean_separation,
            geomean_separation,
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
        }
        return _functions[self]

    def compute(self, separations: NDArray[np.float32]) -> np.float32:
        """
        Compute diversity metric given separations of each vector wrt all others in selection.

        Parameters
        ----------
        separations : NDArray[np.float32]
            Array of separation values.

        Returns
        -------
        np.float32
            The computed diversity score.
        """
        if separations.size < 2:
            # we can only meaningfully compute diversity metrics with at least 2 separation values
            return np.float32(0.0)
        return self._f(separations)
