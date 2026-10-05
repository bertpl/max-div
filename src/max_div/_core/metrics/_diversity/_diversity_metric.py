"""`DiversityMetric` says how a selection's per-item contribution values reduce to one diversity value.

Each diversity metric is a subclass of `DiversityMetric`. A subclass names the compiled function that
reduces the contribution values, the family of per-item contributions it reduces, and 2 properties of
its score that decide the solver's default tie-breakers (`DiversityObjective.default_tie_breakers`).
"""

from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, ClassVar

import numpy as np
from numpy.typing import NDArray

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


class DiversityContributionFamily(StrEnum):
    """Enum for the per-point diversity-contribution families that diversity metrics consume.

    Members
    -------

        - SEPARATION:     contribution = distance to the nearest selected item
        - MEAN_DISTANCE:  contribution = mean distance to the selected items
    """

    SEPARATION = "SEPARATION"
    MEAN_DISTANCE = "MEAN_DISTANCE"


# ==================================================================================================
#  DiversityMetric
# ==================================================================================================
@dataclass(frozen=True, repr=False)
class DiversityMetric:
    """A diversity metric reduces the selected items' per-item contribution values to one diversity value.

    Create instances via the factory methods only.  Each metric is a subclass that stores only its own
    arguments and sets the members whose values differ for that metric.
    """

    # Each subclass sets:
    # - `_label`, read by `label`, and `_factory_name`, read by `__repr__`;
    # - `_reduce`, the compiled function that reduces the contribution values, wrapped in `staticmethod`,
    #   because a compiled function stored on a class binds as a method when read from an instance;
    # - the 3 public class variables below, where their defaults do not hold for it.
    _label: ClassVar[str]
    _factory_name: ClassVar[str]
    _reduce: ClassVar[Callable[[NDArray[np.float32]], np.float32]]
    # the family of per-item contributions that this metric reduces
    contribution_family: ClassVar[DiversityContributionFamily] = DiversityContributionFamily.SEPARATION
    # whether the solver breaks this metric's ties by the approximate geomean separation, as a metric that is
    # set by the smallest separations needs; `DiversityObjective.default_tie_breakers` explains why
    needs_approx_geomean_tie_breaker: ClassVar[bool] = False
    # whether one coincident pair, a separation of zero, makes the score zero
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
        """Return the metric's name in upper case, e.g. `MIN_SEPARATION`, which starts the labels of its objectives."""
        return self._label

    def __repr__(self) -> str:
        """Return the factory call that constructs this metric."""
        return f"DiversityMetric.{self._factory_name}()"

    # --------------------------------------------------------------------------
    #  Factory methods
    # --------------------------------------------------------------------------
    @classmethod
    def min_separation(cls) -> "DiversityMetric":
        """Return the minimum separation of the selected items: the distance between the closest selected pair."""
        return MinSeparationDiversityMetric()

    @classmethod
    def mean_separation(cls) -> "DiversityMetric":
        """Return the arithmetic mean separation of the selected items."""
        return MeanSeparationDiversityMetric()

    @classmethod
    def geomean_separation(cls) -> "DiversityMetric":
        """Return the geometric mean separation of the selected items; it is zero as soon as one separation is zero."""
        return GeomeanSeparationDiversityMetric()

    @classmethod
    def approx_geomean_separation(cls) -> "DiversityMetric":
        """Return the geometric mean separation, computed with fast approximations of log and exp.

        The result is within about one percent of `geomean_separation()`. A zero separation gives a
        value near zero, not exactly zero.
        """
        return ApproxGeomeanSeparationDiversityMetric()

    @classmethod
    def harmonic_mean_separation(cls) -> "DiversityMetric":
        """Return the harmonic mean separation of the selected items.

        It lies between the geometric mean and the minimum in how hard it penalizes close pairs, is
        zero as soon as one separation is zero, and is computed exactly, with no logarithm or
        exponential.
        """
        return HarmonicMeanSeparationDiversityMetric()

    @classmethod
    def non_zero_separation_frac(cls) -> "DiversityMetric":
        """Return the fraction of the selected items whose separation is not zero."""
        return NonZeroSeparationFracDiversityMetric()

    @classmethod
    def mean_pairwise_distance(cls) -> "DiversityMetric":
        """Return the mean distance over all pairs of selected items, the classical max-sum diversity objective."""
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
