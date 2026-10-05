"""`DiversityMetric` says how the selected items' contribution values reduce to one diversity value.

An item's contribution value is a number per selected item, such as its distance to the nearest other
selected item; `DiversityContributionFamily` names the kinds. Each diversity metric is a subclass of
`DiversityMetric`, and a subclass names:

- the compiled function that reduces the contribution values;
- the family of per-item contributions that it reduces;
- the properties of its score that decide the solver's default tie-breakers
  (`DiversityObjective.default_tie_breakers`).
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

import numpy as np
from numpy.typing import NDArray

from ._contribution_family import DiversityContributionFamily
from ._numba import (
    approx_geomean_separation,
    geomean_separation,
    harmonic_mean_separation,
    mean_pairwise_distance,
    mean_separation,
    min_separation,
    non_zero_separation_frac,
)

if TYPE_CHECKING:
    from max_div._core.metrics._distance import DistanceMetric

    from ._hybrid_metric import DiversityTerm


# ==================================================================================================
#  DiversityMetric
# ==================================================================================================
@dataclass(frozen=True, repr=False)
class DiversityMetric:
    """A diversity metric reduces the selected items' per-item contribution values to one diversity value.

    Create instances via the factory methods only.
    """

    # Each subclass sets:
    # - `_label`, read by `label`, and `_factory_name`, read by `__repr__`;
    # - `_reduce`, the compiled function that reduces the contribution values, wrapped in `staticmethod`,
    #   because a compiled function stored on a class binds as a method when read from an instance;
    # - each public class variable whose default does not hold for it.
    _label: ClassVar[str]
    _factory_name: ClassVar[str]
    _reduce: ClassVar[Callable[[NDArray[np.float32]], np.float32]]
    contribution_family: ClassVar[DiversityContributionFamily] = DiversityContributionFamily.SEPARATION
    # Set to True when the solver must break this metric's ties by the approximate geomean separation; a metric
    # whose score is set by its smallest separations needs this, and `DiversityObjective.default_tie_breakers`
    # explains why.
    needs_approx_geomean_tie_breaker: ClassVar[bool] = False
    # Set to True when one coincident pair, a separation of zero, makes the score zero, or near zero for the
    # approximate geomean; the solver then also breaks this metric's ties by the non-zero separation fraction.
    is_zero_at_coincident_pair: ClassVar[bool] = False

    def __post_init__(self) -> None:
        """Reject a bare `DiversityMetric`: only the subclasses returned by the factory methods compute a score."""
        if type(self) is DiversityMetric:
            raise TypeError(
                "Create a DiversityMetric through its factory methods, e.g. DiversityMetric.min_separation()."
            )

    # --------------------------------------------------------------------------
    #  Main API
    # --------------------------------------------------------------------------
    def compute(self, contribution_values: NDArray[np.float32]) -> np.float32:
        """Return the diversity score of the selected items' per-item contribution values.

        Fewer than 2 values score 0: a diversity score needs at least one pair of selected items.

        Args:
            contribution_values: the selected items' contribution values, of this metric's
                `contribution_family`; for the separation family, each item's distance to its nearest
                other selected item.
        """
        if contribution_values.size < 2:
            return np.float32(0.0)
        else:
            return self._reduce(contribution_values)

    def over(self, distance_metric: "DistanceMetric") -> "DiversityTerm":
        """Return this metric as a term over `distance_metric`, for a `HybridDiversityMetric`.

        A bare `DiversityMetric` given as a term reads the problem's own distance; `over` names
        another one, such as `DistanceMetric.along_axis(0)`.
        """
        # Import here to avoid circular dependency
        from ._hybrid_metric import DiversityTerm

        return DiversityTerm(self, distance_metric)

    # --------------------------------------------------------------------------
    #  Representation
    # --------------------------------------------------------------------------
    @property
    def label(self) -> str:
        """Return the metric's name in upper case, e.g. `MIN_SEPARATION`."""
        return self._label

    def __repr__(self) -> str:
        """Return the factory call that constructs this metric."""
        return f"DiversityMetric.{self._factory_name}()"

    # --------------------------------------------------------------------------
    #  Factory methods
    # --------------------------------------------------------------------------
    @classmethod
    def min_separation(cls) -> "DiversityMetric":
        """Return the metric that scores a selection by its minimum separation: the distance of its closest pair."""
        return MinSeparationDiversityMetric()

    @classmethod
    def mean_separation(cls) -> "DiversityMetric":
        """Return the metric that scores a selection by its arithmetic mean separation."""
        return MeanSeparationDiversityMetric()

    @classmethod
    def geomean_separation(cls) -> "DiversityMetric":
        """Return the metric that scores a selection by its geometric mean separation.

        The score is zero as soon as one separation is zero.
        """
        return GeomeanSeparationDiversityMetric()

    @classmethod
    def approx_geomean_separation(cls) -> "DiversityMetric":
        """Return the metric that scores a selection by its geometric mean separation, computed with fast log and exp.

        The result is within about 1% of `geomean_separation()`. A zero separation gives a value
        near zero, not exactly zero.
        """
        return ApproxGeomeanSeparationDiversityMetric()

    @classmethod
    def harmonic_mean_separation(cls) -> "DiversityMetric":
        """Return the metric that scores a selection by its harmonic mean separation.

        It lies between the geometric mean and the minimum in how hard it penalizes close pairs, is
        zero as soon as one separation is zero, and is computed exactly, with no logarithm or
        exponential.
        """
        return HarmonicMeanSeparationDiversityMetric()

    @classmethod
    def non_zero_separation_frac(cls) -> "DiversityMetric":
        """Return the metric that scores a selection by the fraction of its items whose separation is not zero."""
        return NonZeroSeparationFracDiversityMetric()

    @classmethod
    def mean_pairwise_distance(cls) -> "DiversityMetric":
        """Return the metric that scores a selection by the mean distance over all its pairs (max-sum diversity)."""
        return MeanPairwiseDistanceDiversityMetric()


# ==================================================================================================
#  Separation family
# ==================================================================================================
@dataclass(frozen=True, repr=False)
class MinSeparationDiversityMetric(DiversityMetric):
    """This metric is the minimum separation; see `DiversityMetric.min_separation`."""

    _label = "MIN_SEPARATION"
    _factory_name = "min_separation"
    _reduce = staticmethod(min_separation)
    needs_approx_geomean_tie_breaker = True
    is_zero_at_coincident_pair = True


@dataclass(frozen=True, repr=False)
class MeanSeparationDiversityMetric(DiversityMetric):
    """This metric is the arithmetic mean separation; see `DiversityMetric.mean_separation`."""

    _label = "MEAN_SEPARATION"
    _factory_name = "mean_separation"
    _reduce = staticmethod(mean_separation)


@dataclass(frozen=True, repr=False)
class GeomeanSeparationDiversityMetric(DiversityMetric):
    """This metric is the geometric mean separation; see `DiversityMetric.geomean_separation`."""

    _label = "GEOMEAN_SEPARATION"
    _factory_name = "geomean_separation"
    _reduce = staticmethod(geomean_separation)
    is_zero_at_coincident_pair = True


@dataclass(frozen=True, repr=False)
class ApproxGeomeanSeparationDiversityMetric(DiversityMetric):
    """This metric is the approximate geometric mean separation; see `DiversityMetric.approx_geomean_separation`."""

    _label = "APPROX_GEOMEAN_SEPARATION"
    _factory_name = "approx_geomean_separation"
    _reduce = staticmethod(approx_geomean_separation)
    is_zero_at_coincident_pair = True


@dataclass(frozen=True, repr=False)
class HarmonicMeanSeparationDiversityMetric(DiversityMetric):
    """This metric is the harmonic mean separation; see `DiversityMetric.harmonic_mean_separation`."""

    _label = "HARMONIC_MEAN_SEPARATION"
    _factory_name = "harmonic_mean_separation"
    _reduce = staticmethod(harmonic_mean_separation)
    is_zero_at_coincident_pair = True


@dataclass(frozen=True, repr=False)
class NonZeroSeparationFracDiversityMetric(DiversityMetric):
    """This metric is the fraction of non-zero separations; see `DiversityMetric.non_zero_separation_frac`."""

    _label = "NON_ZERO_SEPARATION_FRAC"
    _factory_name = "non_zero_separation_frac"
    _reduce = staticmethod(non_zero_separation_frac)


# ==================================================================================================
#  Mean-distance family
# ==================================================================================================
@dataclass(frozen=True, repr=False)
class MeanPairwiseDistanceDiversityMetric(DiversityMetric):
    """This metric is the mean pairwise distance; see `DiversityMetric.mean_pairwise_distance`."""

    _label = "MEAN_PAIRWISE_DISTANCE"
    _factory_name = "mean_pairwise_distance"
    _reduce = staticmethod(mean_pairwise_distance)
    contribution_family = DiversityContributionFamily.MEAN_DISTANCE
