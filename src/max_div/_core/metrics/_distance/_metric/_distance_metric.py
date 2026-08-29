"""`DistanceMetric` says which distance is meant, as one value usable on both sides of the njit boundary.

The `kind` selectors below are the same ints the compiled pair functions branch on, so the
user-facing metric value and the compiled dispatch share a single classification.
"""

from typing import NamedTuple

# Metric selector values, so njit functions can branch on the metric without object-mode.
# Cosine holds pre-normalized vectors, so its pair read is the half-squared-L2 form on those.
METRIC_KIND_L1 = 0
METRIC_KIND_L2 = 1
METRIC_KIND_L2S = 2
METRIC_KIND_COS = 3
METRIC_KIND_LINF = 4

# Factory-method name per kind, for `__repr__`.
_FACTORY_NAMES = {
    METRIC_KIND_L1: "l1_manhattan",
    METRIC_KIND_L2: "l2_euclidean",
    METRIC_KIND_L2S: "l2s_euclidean_squared",
    METRIC_KIND_COS: "cosine",
    METRIC_KIND_LINF: "linf_chebyshev",
}


class DistanceMetric(NamedTuple):
    """A distance metric: a `kind` selector plus the parameters that kind needs.

    Create instances via the factory methods only — they canonicalize the fields, so metrics that
    compute the same distance compare equal.  `p` is `None` for every kind that does not use it.

    Available metrics:

        - `l1_manhattan()`:           L1 (Manhattan) distance                   = sum_i |x_i - y_i|
        - `l2_euclidean()`:           L2 (Euclidean) distance                   = sqrt( sum_i (x_i - y_i)^2 )
        - `l2s_euclidean_squared()`:  L2 squared (Euclidean squared) distance   = sum_i (x_i - y_i)^2
                                             --> avoids computing square root
                                             --> identical solutions for GEOMEAN_SEPARATION diversity metric
        - `linf_chebyshev()`:         Linf (Chebyshev) distance                 = max_i |x_i - y_i|
                                             --> distance set by the single largest per-dimension gap
        - `cosine()`:                 cosine distance                           = 1 - (x . y) / (|x| |y|)
                                             --> range [0, 2]; angular, for embedding-style vectors
                                             --> undefined for zero vectors (rejected with an error)
    """

    kind: int
    p: float | None

    # --------------------------------------------------------------------------
    #  Factory methods
    # --------------------------------------------------------------------------
    @classmethod
    def l1_manhattan(cls) -> "DistanceMetric":
        """Return the L1 (Manhattan) distance metric."""
        return cls(kind=METRIC_KIND_L1, p=None)

    @classmethod
    def l2_euclidean(cls) -> "DistanceMetric":
        """Return the L2 (Euclidean) distance metric."""
        return cls(kind=METRIC_KIND_L2, p=None)

    @classmethod
    def l2s_euclidean_squared(cls) -> "DistanceMetric":
        """Return the squared L2 (Euclidean squared) distance metric."""
        return cls(kind=METRIC_KIND_L2S, p=None)

    @classmethod
    def linf_chebyshev(cls) -> "DistanceMetric":
        """Return the Linf (Chebyshev) distance metric."""
        return cls(kind=METRIC_KIND_LINF, p=None)

    @classmethod
    def cosine(cls) -> "DistanceMetric":
        """Return the cosine distance metric."""
        return cls(kind=METRIC_KIND_COS, p=None)

    # --------------------------------------------------------------------------
    #  Representation
    # --------------------------------------------------------------------------
    def __repr__(self) -> str:
        """Return the factory call that constructs this metric."""
        return f"DistanceMetric.{_FACTORY_NAMES[self.kind]}()"
