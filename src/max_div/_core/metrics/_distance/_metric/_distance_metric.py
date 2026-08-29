"""`DistanceMetric` says which distance is meant.

The `kind` selectors below are the same ints the compiled pair functions branch on, so the
user-facing metric value and the compiled dispatch share a single classification.
"""

from typing import NamedTuple

# These selector values let njit functions branch on the metric without object-mode.  Cosine holds
# pre-normalized vectors, so its pair read is the half-squared-L2 form on those.
METRIC_KIND_L1 = 0
METRIC_KIND_L2 = 1
METRIC_KIND_L2S = 2
METRIC_KIND_COS = 3
METRIC_KIND_LINF = 4

# `_FACTORY_NAMES` maps each kind to its factory-method name, for `__repr__`.
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
    compute the same distance compare equal.  `p` is the metric's power parameter; it is `None`
    for every kind that does not use one, and crosses the njit boundary as NaN in that case.
    """

    kind: int
    p: float | None

    # --------------------------------------------------------------------------
    #  Factory methods
    # --------------------------------------------------------------------------
    @classmethod
    def l1_manhattan(cls) -> "DistanceMetric":
        """Return the L1 (Manhattan) distance metric: ``sum_i |x_i - y_i|``."""
        return cls(kind=METRIC_KIND_L1, p=None)

    @classmethod
    def l2_euclidean(cls) -> "DistanceMetric":
        """Return the L2 (Euclidean) distance metric: ``sqrt( sum_i (x_i - y_i)^2 )``."""
        return cls(kind=METRIC_KIND_L2, p=None)

    @classmethod
    def l2s_euclidean_squared(cls) -> "DistanceMetric":
        """Return the squared L2 (Euclidean squared) distance metric: ``sum_i (x_i - y_i)^2``.

        Avoids computing a square root, and produces identical solutions for the
        GEOMEAN_SEPARATION diversity metric.
        """
        return cls(kind=METRIC_KIND_L2S, p=None)

    @classmethod
    def linf_chebyshev(cls) -> "DistanceMetric":
        """Return the Linf (Chebyshev) distance metric: ``max_i |x_i - y_i|``.

        The distance is set by the single largest per-dimension gap.
        """
        return cls(kind=METRIC_KIND_LINF, p=None)

    @classmethod
    def cosine(cls) -> "DistanceMetric":
        """Return the cosine distance metric: ``1 - (x . y) / (|x| |y|)``.

        Range [0, 2]; angular, for embedding-style vectors.  Undefined for zero vectors, which
        are rejected with an error.
        """
        return cls(kind=METRIC_KIND_COS, p=None)

    # --------------------------------------------------------------------------
    #  Representation
    # --------------------------------------------------------------------------
    def __repr__(self) -> str:
        """Return the factory call that constructs this metric."""
        return f"DistanceMetric.{_FACTORY_NAMES[self.kind]}()"
