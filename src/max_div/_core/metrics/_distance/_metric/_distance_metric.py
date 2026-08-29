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

# `__repr__` looks up each kind's factory-method name here; the Minkowski kinds render as a
# `minkowski(...)` call instead.
_FACTORY_NAMES = {
    METRIC_KIND_L1: "l1_manhattan",
    METRIC_KIND_L2: "l2_euclidean",
    METRIC_KIND_L2S: "l2s_euclidean_squared",
    METRIC_KIND_COS: "cosine",
    METRIC_KIND_LINF: "linf_chebyshev",
}

# The p each specialized Minkowski kind implies, for `__repr__` (the value itself stores p=None).
_IMPLIED_P = {
    METRIC_KIND_MINKOWSKI_P05: 0.5,
    METRIC_KIND_MINKOWSKI_P05_POWERED: 0.5,
    METRIC_KIND_MINKOWSKI_P025: 0.25,
    METRIC_KIND_MINKOWSKI_P025_POWERED: 0.25,
}

# The Minkowski kinds that skip the outer 1/p root.
_POWERED_KINDS = (
    METRIC_KIND_MINKOWSKI_POWERED,
    METRIC_KIND_MINKOWSKI_P05_POWERED,
    METRIC_KIND_MINKOWSKI_P025_POWERED,
)


class DistanceMetric(NamedTuple):
    """A distance metric: a `kind` selector plus the parameters that kind needs.

    Create instances via the factory methods only, so metrics that compute the same distance
    compare equal.  `p` is the metric's power parameter; it is `None` for every kind that does
    not use one, and crosses the njit boundary as NaN in that case.
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

        The squared form avoids the square root and produces identical solutions under the
        GEOMEAN_SEPARATION diversity metric.
        """
        return cls(kind=METRIC_KIND_L2S, p=None)

    @classmethod
    def linf_chebyshev(cls) -> "DistanceMetric":
        """Return the Linf (Chebyshev) distance metric: ``max_i |x_i - y_i|``."""
        return cls(kind=METRIC_KIND_LINF, p=None)

    @classmethod
    def cosine(cls) -> "DistanceMetric":
        """Return the cosine distance metric: ``1 - (x . y) / (|x| |y|)``.

        The range is [0, 2].  Zero vectors have no defined angle and are rejected with an error.
        """
        return cls(kind=METRIC_KIND_COS, p=None)

    @classmethod
    def minkowski(cls, p: float, root: bool = True) -> "DistanceMetric":
        """Return the Minkowski distance metric: ``( sum_i |x_i - y_i|^p )^(1/p)``, for any p > 0.

        `root=False` skips the outer ``1/p`` root, as `l2s_euclidean_squared()` does for
        `l2_euclidean()`: the root is monotone, so per-pair distance ordering is unchanged and
        solutions are identical under the GEOMEAN_SEPARATION diversity metric.

        The factory canonicalizes `p`: a value with a dedicated metric (p = 1, 2, inf) or a
        specialized implementation (p = 0.5, 0.25) returns that kind, so equal metrics compare
        equal and each metric computes through exactly one code path — this factory is the one
        construction point of that guarantee.

        Cost: p in {1, 2, inf, 0.5, 0.25} computes with hardware arithmetic; any other p pays a
        ``pow`` call per dimension, well over an order of magnitude more per term.

        For 0 < p < 1 the rooted form violates the triangle inequality — not a strict metric,
        while the `root=False` form is one — which the solver never relies on, so both are usable.

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
            return cls(kind=METRIC_KIND_MINKOWSKI_P05 if root else METRIC_KIND_MINKOWSKI_P05_POWERED, p=None)
        if p == 0.25:
            return cls(kind=METRIC_KIND_MINKOWSKI_P025 if root else METRIC_KIND_MINKOWSKI_P025_POWERED, p=None)
        return cls(kind=METRIC_KIND_MINKOWSKI if root else METRIC_KIND_MINKOWSKI_POWERED, p=p)

    # --------------------------------------------------------------------------
    #  njit encoding
    # --------------------------------------------------------------------------
    @property
    def njit_p(self) -> np.float64:
        """Return `p` in its njit encoding: the value as float64, NaN for None."""
        return np.float64(np.nan if self.p is None else self.p)

    # --------------------------------------------------------------------------
    #  Representation
    # --------------------------------------------------------------------------
    def __repr__(self) -> str:
        """Return the factory call that constructs this metric."""
        if self.kind in _FACTORY_NAMES:
            return f"DistanceMetric.{_FACTORY_NAMES[self.kind]}()"
        p = self.p if self.p is not None else _IMPLIED_P[self.kind]
        root_arg = ", root=False" if self.kind in _POWERED_KINDS else ""
        return f"DistanceMetric.minkowski(p={p}{root_arg})"
