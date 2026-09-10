"""`DistanceMetric` says which distance is meant.

The `kind` selectors below are the same ints the compiled pair functions branch on, so the
user-facing metric value and the compiled dispatch share a single classification.
"""

import math
from typing import NamedTuple

import numpy as np

# These selector values let njit functions branch on the metric without object-mode.
METRIC_KIND_L1 = 0
METRIC_KIND_L2 = 1
METRIC_KIND_L2S = 2
METRIC_KIND_COS = 3
METRIC_KIND_LINF = 4
METRIC_KIND_MINKOWSKI = 5
METRIC_KIND_MINKOWSKI_POWERED = 6
METRIC_KIND_MINKOWSKI_P05 = 7
METRIC_KIND_MINKOWSKI_P05_POWERED = 8
METRIC_KIND_MINKOWSKI_P025 = 9
METRIC_KIND_MINKOWSKI_P025_POWERED = 10
METRIC_KIND_MINKOWSKI_P0125 = 11
METRIC_KIND_MINKOWSKI_P0125_POWERED = 12
METRIC_KIND_GEOMEAN = 13
METRIC_KIND_ALONG_AXIS = 14

# `__repr__` looks up each kind's factory-method name here; the Minkowski kinds render as a
# `minkowski(...)` call and the along-axis kind as an `along_axis(...)` call instead.
_FACTORY_NAMES = {
    METRIC_KIND_L1: "l1_manhattan",
    METRIC_KIND_L2: "l2_euclidean",
    METRIC_KIND_L2S: "l2s_euclidean_squared",
    METRIC_KIND_COS: "cosine",
    METRIC_KIND_LINF: "linf_chebyshev",
    METRIC_KIND_GEOMEAN: "geometric_mean",
}

# `_IMPLIED_P` gives the p each specialized Minkowski kind implies, for `__repr__`; a metric of
# these kinds stores p=NO_P.
_IMPLIED_P = {
    METRIC_KIND_MINKOWSKI_P05: 0.5,
    METRIC_KIND_MINKOWSKI_P05_POWERED: 0.5,
    METRIC_KIND_MINKOWSKI_P025: 0.25,
    METRIC_KIND_MINKOWSKI_P025_POWERED: 0.25,
    METRIC_KIND_MINKOWSKI_P0125: 0.125,
    METRIC_KIND_MINKOWSKI_P0125_POWERED: 0.125,
}

# Kinds that need a preprocessed form of the vectors (see `_preprocess`), not the user's array.
_PREPROCESSING_KINDS = frozenset({METRIC_KIND_COS, METRIC_KIND_ALONG_AXIS})

# These Minkowski kinds skip the outer 1/p root.
_POWERED_KINDS = (
    METRIC_KIND_MINKOWSKI_POWERED,
    METRIC_KIND_MINKOWSKI_P05_POWERED,
    METRIC_KIND_MINKOWSKI_P025_POWERED,
    METRIC_KIND_MINKOWSKI_P0125_POWERED,
)


# Fields are stored as the compiled functions read them, so a metric crosses the njit boundary
# without conversion: `p` is a float, and NO_P marks a kind without a power parameter (every
# Minkowski kind requires p > 0, so 0.0 is free to mean "none").  `axis` follows the same rule:
# an int, with NO_AXIS marking every kind that does not read one coordinate.
NO_P = 0.0
NO_AXIS = -1


class DistanceMetric(NamedTuple):
    """A distance metric: a `kind` selector plus the parameters that kind needs.

    Create instances via the factory methods only, so metrics that compute the same distance
    compare equal.  `p` is the metric's power parameter, `NO_P` for every kind that does not use
    one; `axis` is the coordinate that `along_axis` reads, `NO_AXIS` for every other kind.
    """

    kind: int
    p: float
    axis: int

    # --------------------------------------------------------------------------
    #  Factory methods
    # --------------------------------------------------------------------------
    @classmethod
    def l1_manhattan(cls) -> "DistanceMetric":
        """Return the L1 (Manhattan) distance metric: ``sum_i |x_i - y_i|``."""
        return cls(kind=METRIC_KIND_L1, p=NO_P, axis=NO_AXIS)

    @classmethod
    def l2_euclidean(cls) -> "DistanceMetric":
        """Return the L2 (Euclidean) distance metric: ``sqrt( sum_i (x_i - y_i)^2 )``."""
        return cls(kind=METRIC_KIND_L2, p=NO_P, axis=NO_AXIS)

    @classmethod
    def l2s_euclidean_squared(cls) -> "DistanceMetric":
        """Return the squared L2 (Euclidean squared) distance metric: ``sum_i (x_i - y_i)^2``.

        The squared form avoids the square root and produces identical solutions under the
        GEOMEAN_SEPARATION diversity metric.
        """
        return cls(kind=METRIC_KIND_L2S, p=NO_P, axis=NO_AXIS)

    @classmethod
    def linf_chebyshev(cls) -> "DistanceMetric":
        """Return the Linf (Chebyshev) distance metric: ``max_i |x_i - y_i|``."""
        return cls(kind=METRIC_KIND_LINF, p=NO_P, axis=NO_AXIS)

    @classmethod
    def cosine(cls) -> "DistanceMetric":
        """Return the cosine distance metric: ``1 - (x . y) / (|x| |y|)``.

        The range is [0, 2].  Zero vectors have no defined angle and are rejected with an error.
        """
        return cls(kind=METRIC_KIND_COS, p=NO_P, axis=NO_AXIS)

    @classmethod
    def geometric_mean(cls) -> "DistanceMetric":
        """Return the geometric-mean distance metric: ``( prod_i |x_i - y_i| )^(1/d)``.

        The diversity concepts page explains what this distance rewards and cites the design
        literature behind it.  A shared coordinate value makes the distance zero.

        It is not a strict metric (distinct points can be at distance zero, and the triangle
        inequality fails); the solver relies on neither.  It costs one ``log`` per dimension.
        """
        return cls(kind=METRIC_KIND_GEOMEAN, p=NO_P, axis=NO_AXIS)

    @classmethod
    def along_axis(cls, axis: int) -> "DistanceMetric":
        """Return the distance along one coordinate axis: ``|x_axis - y_axis|``.

        A legitimate distance on its own, which spreads a selection along that single coordinate
        only, and the building block of an objective that spreads a selection in several
        coordinate projections at once.  The vector problem checks the axis against its dimension
        count when it is constructed.

        Args:
            axis: The zero-based index of the coordinate to read.
        """
        if isinstance(axis, bool) or not isinstance(axis, (int, np.integer)) or axis < 0:
            raise ValueError(f"along_axis requires a non-negative integer axis; here: {axis!r}.")
        return cls(kind=METRIC_KIND_ALONG_AXIS, p=NO_P, axis=int(axis))

    @classmethod
    def minkowski(cls, p: float, root: bool = True) -> "DistanceMetric":
        """Return the Minkowski distance metric: ``( sum_i |x_i - y_i|^p )^(1/p)``, for any p > 0.

        `root=False` skips the outer ``1/p`` root, as `l2s_euclidean_squared()` does for
        `l2_euclidean()`: the root is monotone, so per-pair distance ordering is unchanged.

        The factory canonicalizes `p`: p = 1, 2, inf return the dedicated metrics and p = 0.5,
        0.25, 0.125 the specialized kinds, so each metric computes through exactly one code path.

        Cost: every canonicalized p above computes with hardware arithmetic; any other p pays a
        ``pow`` call per dimension, well over an order of magnitude more per term.

        For 0 < p < 1 the `root=True` form violates the triangle inequality — not a strict
        metric, while the `root=False` form is one.  The solver never relies on the triangle
        inequality, so both are usable.

        Args:
            p: The exponent; any value > 0, with inf giving `linf_chebyshev()`.
            root: Whether the outer ``1/p`` root is applied; irrelevant at p = 1 and p = inf.
        """
        p = float(p)
        if math.isnan(p) or p <= 0:
            raise ValueError(f"Minkowski distance requires p > 0; here: {p}.")
        if p == math.inf:
            return cls.linf_chebyshev()
        if p == 1.0:
            return cls.l1_manhattan()
        if p == 2.0:
            return cls.l2_euclidean() if root else cls.l2s_euclidean_squared()
        if p == 0.5:
            return cls(
                kind=METRIC_KIND_MINKOWSKI_P05 if root else METRIC_KIND_MINKOWSKI_P05_POWERED, p=NO_P, axis=NO_AXIS
            )
        if p == 0.25:
            return cls(
                kind=METRIC_KIND_MINKOWSKI_P025 if root else METRIC_KIND_MINKOWSKI_P025_POWERED, p=NO_P, axis=NO_AXIS
            )
        if p == 0.125:
            return cls(
                kind=METRIC_KIND_MINKOWSKI_P0125 if root else METRIC_KIND_MINKOWSKI_P0125_POWERED, p=NO_P, axis=NO_AXIS
            )
        return cls(kind=METRIC_KIND_MINKOWSKI if root else METRIC_KIND_MINKOWSKI_POWERED, p=p, axis=NO_AXIS)

    # --------------------------------------------------------------------------
    #  Properties
    # --------------------------------------------------------------------------
    @property
    def needs_preprocessed_vectors(self) -> bool:
        """Return whether this metric's distances read a preprocessed copy of the vectors (see `_preprocess`)."""
        return self.kind in _PREPROCESSING_KINDS

    # --------------------------------------------------------------------------
    #  Representation
    # --------------------------------------------------------------------------
    def __repr__(self) -> str:
        """Return the factory call that constructs this metric."""
        if self.kind in _FACTORY_NAMES:
            return f"DistanceMetric.{_FACTORY_NAMES[self.kind]}()"
        if self.kind == METRIC_KIND_ALONG_AXIS:
            return f"DistanceMetric.along_axis({self.axis})"
        p = self.p if self.p != NO_P else _IMPLIED_P[self.kind]
        root_arg = ", root=False" if self.kind in _POWERED_KINDS else ""
        return f"DistanceMetric.minkowski(p={p}{root_arg})"
