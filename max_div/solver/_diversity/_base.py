from __future__ import annotations

from typing import Callable

import numpy as np
from numpy.typing import NDArray

from ._numba import (
    approx_geomean_separation,
    geomean_separation,
    mean_separation,
    min_separation,
    non_zero_separation_frac,
)


class DiversityMetric:
    """
    Class implementing various diversity metrics.  Use factory methods for instantiation:

        min_separation()             Minimum separation of all selected vectors
        mean_separation()            Arithmetic mean separation of all selected vectors
        geomean_separation()         Geometric mean separation of all selected vectors
        approx_geomean_separation()  Approximate geometric mean separation of all selected vectors
                                         (uses faster, but still smooth approximations of log(.) and exp(.))
        non_zero_separation_frac()   Fraction of separation values that are non-zero
    """

    # -------------------------------------------------------------------------
    #  Constructor & API
    # -------------------------------------------------------------------------
    def __init__(self, name: str, f: Callable):
        self.name = name
        self.f = f

    def compute(self, separations: NDArray[np.float32]) -> np.float32:
        """Compute diversity metric given separations of each vector wrt all others in selection."""
        if separations.size == 0:
            return np.float32(0.0)
        else:
            return self.f(separations)

    def __eq__(self, other) -> bool:
        return isinstance(other, DiversityMetric) and (self.name == other.name) and (self.f == other.f)

    # -------------------------------------------------------------------------
    #  Factory methods
    # -------------------------------------------------------------------------
    @classmethod
    def min_separation(cls) -> DiversityMetric:
        """Minimum separation of all selected vectors."""
        return cls(name="min_separation", f=min_separation)

    @classmethod
    def mean_separation(cls) -> DiversityMetric:
        """Arithmetic mean separation of all selected vectors."""
        return cls(name="mean_separation", f=mean_separation)

    @classmethod
    def geomean_separation(cls) -> DiversityMetric:
        """Geometric mean separation of all selected vectors."""
        return cls(name="geomean_separation", f=geomean_separation)

    @classmethod
    def approx_geomean_separation(cls) -> DiversityMetric:
        """Approximate geometric mean separation of all selected vectors."""
        return cls(name="approx_geomean_separation", f=approx_geomean_separation)

    @classmethod
    def non_zero_separation_frac(cls) -> DiversityMetric:
        """Fraction of separation values that are non-zero."""
        return cls(name="non_zero_separation_frac", f=non_zero_separation_frac)
