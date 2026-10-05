"""`DiversityMetric` says how the selected items' contribution values reduce to one diversity value.

An item's contribution value measures how much that selected item adds to the diversity, such as its
distance to the nearest other selected item; `DiversityContributionFamily` names the kinds of
contribution value. Each diversity metric is a subclass of `DiversityMetric`, created only through a
factory method of `DiversityMetric`.
"""

import functools
import numbers
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

import numpy as np
from numpy.typing import NDArray

from ._contribution_family import DiversityContributionFamily
from ._numba import (
    approx_geomean_separation,
    geomean_separation,
    gpq_separation,
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
    # - `_factory_name`, the name of its factory method, read by `label` and `__repr__`;
    # - `_reduce`, the compiled function that reduces the contribution values, wrapped in `staticmethod`,
    #   because a compiled function stored on a class binds as a method when read from an instance;
    # - each public class variable whose default does not hold for it.
    _factory_name: ClassVar[str]
    _reduce: ClassVar[Callable[[NDArray[np.float32]], np.float32]]
    contribution_family: ClassVar[DiversityContributionFamily] = DiversityContributionFamily.SEPARATION
    # Set to True when the solver must break this metric's ties by the approximate geomean separation; a metric
    # whose score is set by its smallest separations needs this, and `DiversityObjective.default_tie_breakers`
    # explains why.
    needs_approx_geomean_tie_breaker: ClassVar[bool] = False
    # Set to True when the solver must break this metric's ties by the non-zero separation fraction; a metric
    # whose score is zero, or near zero, as soon as one coincident pair (2 selected items at distance zero)
    # appears needs this.
    needs_non_zero_separation_frac_tie_breaker: ClassVar[bool] = False

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
        """Return the name of the metric's factory method in upper case, e.g. `MIN_SEPARATION`."""
        return self._factory_name.upper()

    def __repr__(self) -> str:
        """Return the factory call that constructs this metric."""
        return f"DiversityMetric.{self._factory_name}({', '.join(self._factory_arg_reprs())})"

    def _factory_arg_reprs(self) -> tuple[str, ...]:
        """Return the factory method's arguments as `name=value` strings for `__repr__`; the base returns none.

        A subclass whose factory method takes arguments overrides this.
        """
        return ()

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

        The harmonic mean separation:

        - penalizes close pairs harder than the geometric mean and less hard than the minimum;
        - is zero as soon as one separation is zero;
        - is computed exactly, with no logarithm or exponential.
        """
        return HarmonicMeanSeparationDiversityMetric()

    @classmethod
    def gpq_separation(cls, q: float) -> "DiversityMetric":
        """Return the metric that scores a selection by the geometric pseudo-quantile of its separations at level `q`.

        The geometric pseudo-quantile is a geometric mean of the k separations in which each separation
        is weighted by its rank. The i-th smallest separation, for i = 0, ..., k - 1, has the weight
        ``(1 - u_i) ** (1 / q - 2)`` with ``u_i = (i + 0.5) / k``, so the lower `q`, the more weight
        moves onto the smallest separations:

        - at `q = 0` the score is the minimum separation, and the factory returns `min_separation()`;
        - at `q = 0.5` every weight is 1, so the score is the geometric mean separation, and the
          factory returns `geomean_separation()`;
        - for any `q` in between, the score lies between those 2 and is zero as soon as one separation
          is zero.

        Computing a score sorts the separations, so it takes several times as long as computing a
        `geomean_separation()` score. The solver adds the same tie-breakers as for `min_separation()`.

        Args:
            q: The quantile level, a number between 0 and 0.5 inclusive.

        Raises:
            ValueError: If `q` is not a number between 0 and 0.5 inclusive.
        """
        if isinstance(q, bool) or not isinstance(q, numbers.Real) or not 0.0 <= q <= 0.5:
            raise ValueError(f"gpq_separation requires a number q with 0 <= q <= 0.5; here: {q!r}.")
        if q == 0.0:
            return cls.min_separation()
        elif q == 0.5:
            return cls.geomean_separation()
        else:
            return GpqSeparationDiversityMetric(q=float(q))

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

    _factory_name = "min_separation"
    _reduce = staticmethod(min_separation)
    needs_approx_geomean_tie_breaker = True
    needs_non_zero_separation_frac_tie_breaker = True


@dataclass(frozen=True, repr=False)
class MeanSeparationDiversityMetric(DiversityMetric):
    """This metric is the arithmetic mean separation; see `DiversityMetric.mean_separation`."""

    _factory_name = "mean_separation"
    _reduce = staticmethod(mean_separation)


@dataclass(frozen=True, repr=False)
class GeomeanSeparationDiversityMetric(DiversityMetric):
    """This metric is the geometric mean separation; see `DiversityMetric.geomean_separation`."""

    _factory_name = "geomean_separation"
    _reduce = staticmethod(geomean_separation)
    needs_non_zero_separation_frac_tie_breaker = True


@dataclass(frozen=True, repr=False)
class ApproxGeomeanSeparationDiversityMetric(DiversityMetric):
    """This metric is the approximate geometric mean separation; see `DiversityMetric.approx_geomean_separation`."""

    _factory_name = "approx_geomean_separation"
    _reduce = staticmethod(approx_geomean_separation)
    needs_non_zero_separation_frac_tie_breaker = True


@dataclass(frozen=True, repr=False)
class HarmonicMeanSeparationDiversityMetric(DiversityMetric):
    """This metric is the harmonic mean separation; see `DiversityMetric.harmonic_mean_separation`."""

    _factory_name = "harmonic_mean_separation"
    _reduce = staticmethod(harmonic_mean_separation)
    needs_non_zero_separation_frac_tie_breaker = True


@dataclass(frozen=True, repr=False)
class GpqSeparationDiversityMetric(DiversityMetric):
    """This metric is the geometric pseudo-quantile of the separations; see `DiversityMetric.gpq_separation`.

    The factory returns this class only for a `q` strictly between 0 and 0.5.
    """

    q: float

    _factory_name = "gpq_separation"
    # Once the selection is large, the weighted terms of the largest separations are too small to change the
    # float32 sum, so a swap that only spreads the items with large separations can leave the score unchanged.
    needs_approx_geomean_tie_breaker = True
    needs_non_zero_separation_frac_tie_breaker = True

    @property
    def label(self) -> str:
        """Return e.g. `GPQ_SEPARATION(q=0.25)`."""
        return f"{super().label}(q={self.q:g})"

    def _reduce(self, contribution_values: NDArray[np.float32]) -> np.float32:
        """Return the rank-weighted geometric mean of the separations, using this metric's weights for their number."""
        return gpq_separation(contribution_values, self._rank_weights(contribution_values.size, self.q))

    def _factory_arg_reprs(self) -> tuple[str, ...]:
        """Return `q` as the keyword argument `q=...`."""
        return (f"q={self.q!r}",)

    # --------------------------------------------------------------------------
    #  Helpers
    # --------------------------------------------------------------------------
    @staticmethod
    @functools.lru_cache(maxsize=8)
    def _rank_weights(n_values: int, q: float) -> NDArray[np.float32]:
        """Return the weight of each of `n_values` separations in ascending order, the first weight being 1.

        The weights are computed in float64 log space and divided by the first, largest one before the
        cast to float32, so a small `q` cannot round every weight to 0; a weight below the float32
        range becomes exactly 0, which the compiled `gpq_separation` function skips.

        A solve scores selections of k items and, when it tries removing an item, of k - 1 items, so a
        small cache keeps both weight arrays. An initialization that grows the selection one item at a
        time needs a new array for each size, so it evicts the older arrays from the cache. The returned
        array is cached and shared between calls, so a caller must not modify it.
        """
        rank_fractions = (np.arange(n_values) + 0.5) / n_values
        log_weights = (1.0 / q - 2.0) * np.log1p(-rank_fractions)
        return np.exp(log_weights - log_weights[0]).astype(np.float32)


@dataclass(frozen=True, repr=False)
class NonZeroSeparationFracDiversityMetric(DiversityMetric):
    """This metric is the fraction of non-zero separations; see `DiversityMetric.non_zero_separation_frac`."""

    _factory_name = "non_zero_separation_frac"
    _reduce = staticmethod(non_zero_separation_frac)


# ==================================================================================================
#  Mean-distance family
# ==================================================================================================
@dataclass(frozen=True, repr=False)
class MeanPairwiseDistanceDiversityMetric(DiversityMetric):
    """This metric is the mean pairwise distance; see `DiversityMetric.mean_pairwise_distance`."""

    _factory_name = "mean_pairwise_distance"
    _reduce = staticmethod(mean_pairwise_distance)
    contribution_family = DiversityContributionFamily.MEAN_DISTANCE
