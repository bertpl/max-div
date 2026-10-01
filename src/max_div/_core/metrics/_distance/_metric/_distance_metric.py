"""`DistanceMetric` says which distance is meant, and each metric class owns what its kind needs.

The compiled pair functions in `_pair` branch on an int selector, the metric's `kind`, so the
user-facing metric value and the compiled dispatch share a single classification.  What the compiled
side cannot hold lives on the metric class: the checks of a metric against a problem's vectors, the
preprocessing of those vectors into the form its pair function reads, and the one float parameter
that the pair function takes.
"""

import math
from dataclasses import dataclass
from typing import ClassVar

import numpy as np
from numpy.typing import NDArray

from ._preprocess import _normalize_rows, validate_vector_array_layout

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
METRIC_KIND_LMINUSINF = 15
METRIC_KIND_MARGINALS_AND_JOINT = 16

# The float that a compiled pair function takes; NO_PARAM for a kind that takes none (every kind that
# takes one requires it to be > 0, so 0.0 is free to mean "none").
NO_PARAM = 0.0


# =================================================================================================
#  DistanceMetric
# =================================================================================================
@dataclass(frozen=True, repr=False)
class DistanceMetric:
    """A distance metric: which distance is meant, and what its compiled pair function needs.

    Create instances via the factory methods only; they canonicalize their arguments, so metrics
    that compute the same distance compare equal.  Each kind of distance is a subclass that stores
    only its own arguments and owns:

    - `kind`: the selector that the compiled pair functions branch on;
    - `compiled_param`: the one float that the compiled pair function takes, `NO_PARAM` for a kind
      that takes none;
    - `validate`: the checks of the metric against a problem's vectors;
    - `preprocess`: the form of the vectors that its pair function reads;
    - `label`, and the repr, which is the factory call that reconstructs the metric.
    """

    # Every subclass sets these; `kind` and `label` read them unless the subclass computes its own.
    _kind: ClassVar[int]
    _label: ClassVar[str]
    _factory_name: ClassVar[str]
    needs_preprocessed_vectors: ClassVar[bool] = False  # whether `preprocess` returns a new array, not the input

    def __post_init__(self) -> None:
        """Reject a bare `DistanceMetric`: only the factory methods' subclasses compute a distance."""
        if type(self) is DistanceMetric:
            raise TypeError("Create a DistanceMetric through its factory methods, e.g. DistanceMetric.l2_euclidean().")

    # --------------------------------------------------------------------------
    #  Factory methods
    # --------------------------------------------------------------------------
    @classmethod
    def l1_manhattan(cls) -> "DistanceMetric":
        """Return the L1 (Manhattan) distance metric: ``sum_i |x_i - y_i|``."""
        return L1Manhattan()

    @classmethod
    def l2_euclidean(cls) -> "DistanceMetric":
        """Return the L2 (Euclidean) distance metric: ``sqrt( sum_i (x_i - y_i)^2 )``."""
        return L2Euclidean()

    @classmethod
    def l2s_euclidean_squared(cls) -> "DistanceMetric":
        """Return the squared L2 (Euclidean squared) distance metric: ``sum_i (x_i - y_i)^2``.

        The squared form avoids the square root and produces identical solutions under the
        GEOMEAN_SEPARATION diversity metric.
        """
        return L2sEuclideanSquared()

    @classmethod
    def linf_chebyshev(cls) -> "DistanceMetric":
        """Return the Linf (Chebyshev) distance metric: ``max_i |x_i - y_i|``."""
        return LinfChebyshev()

    @classmethod
    def cosine(cls) -> "DistanceMetric":
        """Return the cosine distance metric: ``1 - (x . y) / (|x| |y|)``.

        The range is [0, 2].  Zero vectors have no defined angle and are rejected with an error.
        """
        return Cosine()

    @classmethod
    def geometric_mean(cls) -> "DistanceMetric":
        """Return the geometric-mean distance metric: ``( prod_i |x_i - y_i| )^(1/d)``.

        The diversity concepts page explains what this distance rewards and cites the design
        literature behind it.  A shared coordinate value makes the distance zero.

        It is not a strict metric (distinct points can be at distance zero, and the triangle
        inequality fails); the solver relies on neither.  It costs one ``log`` per dimension.
        """
        return GeometricMean()

    @classmethod
    def l_minus_inf(cls) -> "DistanceMetric":
        """Return the L-∞ distance metric: ``min_i |x_i - y_i|``, the smallest coordinate difference.

        It is the p → -∞ end of the power-mean family whose p → +∞ end is `linf_chebyshev()`, and
        the exact form that `geometric_mean()` smooths: the distance between two points is the gap
        in the coordinate projection where they are closest, so a selection kept apart under it is
        spread in every coordinate projection.  Two points that share any one coordinate are at
        distance zero.

        It is not a metric in the mathematical sense (distinct points can be at distance zero,
        and the triangle inequality fails); the solver relies on neither.
        """
        return LMinusInf()

    @classmethod
    def along_axis(cls, axis: int) -> "DistanceMetric":
        """Return the distance along one coordinate axis: ``|x_axis - y_axis|``.

        It spreads a selection along that single coordinate only.  The vector problem checks the axis
        against its dimension count when it is constructed.

        Args:
            axis: The zero-based index of the coordinate to read.

        Raises:
            ValueError: If `axis` is not a non-negative integer.
        """
        if isinstance(axis, bool) or not isinstance(axis, (int, np.integer)) or axis < 0:
            raise ValueError(f"along_axis requires a non-negative integer axis; here: {axis!r}.")
        return AlongAxis(axis=int(axis))

    @classmethod
    def marginals_and_joint(cls, joint_scale: float = 1.0) -> "DistanceMetric":
        """Return the marginals-and-joint distance: ``min( min_i |a_i - b_i|, joint_scale * ||a - b||_2^d )``.

        For 2 vectors a and b of dimension d, the first term is the `l_minus_inf()` distance, the gap in
        the coordinate where they are closest, and the second term, the joint term, is their L2 distance
        raised to the power d.

        Under min-separation a selection is then spread along every coordinate axis (its marginals) and
        in the full space (its joint distribution) at once.

        The 2 terms are comparable only for a population that fills the unit cube [0, 1]^d, so scale
        the vectors into it first.  It needs at least 2 dimensions: in 1 it only rescales the one
        coordinate gap, and a problem over 1-dimensional vectors rejects it.

        For k well-spread points in the unit cube, the gap along an axis between neighbors can reach
        1/k, while the nearest-neighbor L2 distance raised to the power d is about c/k, where the
        constant c grows with d:

        - c ≈ 1.15 for d = 2
        - c ≈ 1.4 for d = 3
        - c ≈ 2.8 for d = 5
        - c ≈ 40 for d = 10

        In higher dimensions the joint term is therefore larger than the gap and rarely sets the
        minimum; a `joint_scale` of about 1/c gives the 2 terms equal weight.

        The marginals-and-joint distance is not a metric in the mathematical sense (points that share any one
        coordinate are at distance zero, and the triangle inequality fails); the solver relies on neither.

        Args:
            joint_scale: The positive, finite factor on the joint term.

        Raises:
            ValueError: If `joint_scale` is not a positive, finite number.
        """
        joint_scale = float(joint_scale)
        if not (math.isfinite(joint_scale) and joint_scale > 0):
            raise ValueError(f"marginals_and_joint requires a positive, finite joint_scale; here: {joint_scale}.")
        return MarginalsAndJoint(joint_scale=joint_scale)

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
        elif p == 1.0:
            return cls.l1_manhattan()
        elif p == 2.0:
            return cls.l2_euclidean() if root else cls.l2s_euclidean_squared()
        else:
            return Minkowski(p=p, root=root)

    # --------------------------------------------------------------------------
    #  What the compiled pair function needs
    # --------------------------------------------------------------------------
    @property
    def kind(self) -> int:
        """Return the selector that the compiled pair functions branch on."""
        return self._kind

    @property
    def compiled_param(self) -> float:
        """Return the float that the compiled pair function takes; `NO_PARAM` for a kind that takes none."""
        return NO_PARAM

    def validate(self, vectors: NDArray[np.float32]) -> None:
        """Raise ValueError if this metric cannot be computed on `vectors`.

        A no-op for a kind that computes on any vectors, so a caller can call it on every metric.
        """

    def preprocess(self, vectors: NDArray[np.float32]) -> NDArray[np.float32]:
        """Return the vectors as this metric's pair function reads them: the input itself, unless the kind preprocesses.

        The input is never written; a preprocessed result is a new array.

        Raises:
            ValueError: If `vectors` is not a 2D float32 C-contiguous array, or `validate` rejects it.
        """
        validate_vector_array_layout(vectors)
        self.validate(vectors)
        return self._preprocessed(vectors)

    def _preprocessed(self, vectors: NDArray[np.float32]) -> NDArray[np.float32]:
        """Return the array that the pair function reads, for vectors that passed the checks; the input by default."""
        return vectors

    # --------------------------------------------------------------------------
    #  Representation
    # --------------------------------------------------------------------------
    @property
    def label(self) -> str:
        """Return a short label for the metric, e.g. `L1`, `geomean`, `axis 2`, `L3-powered`."""
        return self._label

    def __repr__(self) -> str:
        """Return the factory call that constructs this metric, leaving out an argument that equals its default."""
        return f"DistanceMetric.{self._factory_name}({', '.join(self._factory_args())})"

    def _factory_args(self) -> tuple[str, ...]:
        """Return the factory arguments as source text, in the factory's order, leaving out those at their default."""
        return ()


# =================================================================================================
#  Minkowski family
# =================================================================================================
@dataclass(frozen=True, repr=False)
class L1Manhattan(DistanceMetric):
    """The L1 (Manhattan) distance; see `DistanceMetric.l1_manhattan`."""

    _kind = METRIC_KIND_L1
    _label = "L1"
    _factory_name = "l1_manhattan"


@dataclass(frozen=True, repr=False)
class L2Euclidean(DistanceMetric):
    """The L2 (Euclidean) distance; see `DistanceMetric.l2_euclidean`."""

    _kind = METRIC_KIND_L2
    _label = "L2"
    _factory_name = "l2_euclidean"


@dataclass(frozen=True, repr=False)
class L2sEuclideanSquared(DistanceMetric):
    """The squared L2 distance; see `DistanceMetric.l2s_euclidean_squared`."""

    _kind = METRIC_KIND_L2S
    _label = "L2²"
    _factory_name = "l2s_euclidean_squared"


@dataclass(frozen=True, repr=False)
class LinfChebyshev(DistanceMetric):
    """The Linf (Chebyshev) distance; see `DistanceMetric.linf_chebyshev`."""

    _kind = METRIC_KIND_LINF
    _label = "L∞"
    _factory_name = "linf_chebyshev"


# The Minkowski values of p that compute through a specialized pair function, by (p, root); the
# specialized kinds apply the root as repeated squarings and take no parameter.
_MINKOWSKI_SPECIALIZED_KINDS: dict[tuple[float, bool], int] = {
    (0.5, True): METRIC_KIND_MINKOWSKI_P05,
    (0.5, False): METRIC_KIND_MINKOWSKI_P05_POWERED,
    (0.25, True): METRIC_KIND_MINKOWSKI_P025,
    (0.25, False): METRIC_KIND_MINKOWSKI_P025_POWERED,
    (0.125, True): METRIC_KIND_MINKOWSKI_P0125,
    (0.125, False): METRIC_KIND_MINKOWSKI_P0125_POWERED,
}


@dataclass(frozen=True, repr=False)
class Minkowski(DistanceMetric):
    """The Minkowski distance for a p without a dedicated class; see `DistanceMetric.minkowski`."""

    p: float
    root: bool

    _factory_name = "minkowski"

    @property
    def kind(self) -> int:
        """Return the specialized selector for p in {0.5, 0.25, 0.125}, else the generic one for the root choice."""
        generic = METRIC_KIND_MINKOWSKI if self.root else METRIC_KIND_MINKOWSKI_POWERED
        return _MINKOWSKI_SPECIALIZED_KINDS.get((self.p, self.root), generic)

    @property
    def compiled_param(self) -> float:
        """Return p for the generic kinds; a specialized kind has p built in and takes none."""
        return NO_PARAM if (self.p, self.root) in _MINKOWSKI_SPECIALIZED_KINDS else self.p

    @property
    def label(self) -> str:
        """Return `L<p>`, with `-powered` appended when the root is skipped."""
        return f"L{self.p:g}{'' if self.root else '-powered'}"

    def _factory_args(self) -> tuple[str, ...]:
        return (f"p={self.p!r}",) if self.root else (f"p={self.p!r}", "root=False")


# =================================================================================================
#  Angular
# =================================================================================================
@dataclass(frozen=True, repr=False)
class Cosine(DistanceMetric):
    """The cosine distance; see `DistanceMetric.cosine`.

    Its pair function reads rows scaled to unit L2 norm, so the distance is half the squared L2
    distance of the scaled rows; `preprocess` does the scaling.
    """

    _kind = METRIC_KIND_COS
    _label = "cosine"
    _factory_name = "cosine"
    needs_preprocessed_vectors = True

    def validate(self, vectors: NDArray[np.float32]) -> None:
        """Raise ValueError if any vector is all-zero: cosine distance is undefined for zero vectors."""
        zero_rows = np.flatnonzero(~vectors.any(axis=1))
        if zero_rows.size > 0:
            raise ValueError(
                f"Cosine distance is undefined for zero vectors; found an all-zero vector at row {zero_rows[0]}."
            )

    def _preprocessed(self, vectors: NDArray[np.float32]) -> NDArray[np.float32]:
        """Return a fresh float32 array with each row scaled to unit L2 norm."""
        return _normalize_rows(vectors)


# =================================================================================================
#  Coordinate-wise
# =================================================================================================
@dataclass(frozen=True, repr=False)
class GeometricMean(DistanceMetric):
    """The geometric-mean distance; see `DistanceMetric.geometric_mean`."""

    _kind = METRIC_KIND_GEOMEAN
    _label = "geomean"
    _factory_name = "geometric_mean"


@dataclass(frozen=True, repr=False)
class LMinusInf(DistanceMetric):
    """The L-∞ distance; see `DistanceMetric.l_minus_inf`."""

    _kind = METRIC_KIND_LMINUSINF
    _label = "L-∞"
    _factory_name = "l_minus_inf"


@dataclass(frozen=True, repr=False)
class AlongAxis(DistanceMetric):
    """The distance along one coordinate axis; see `DistanceMetric.along_axis`.

    Its pair function reads column 0 of the array it is given, so `preprocess` slices the axis out
    into an (n, 1) array and the axis itself never crosses the compiled boundary.
    """

    axis: int

    _kind = METRIC_KIND_ALONG_AXIS
    _factory_name = "along_axis"
    needs_preprocessed_vectors = True

    @property
    def label(self) -> str:
        """Return `axis <i>`."""
        return f"axis {self.axis}"

    def validate(self, vectors: NDArray[np.float32]) -> None:
        """Raise ValueError if the axis is not a coordinate of `vectors`."""
        n_dims = vectors.shape[1]
        if self.axis >= n_dims:
            raise ValueError(f"{self!r} reads a coordinate that {n_dims}-dimensional vectors do not have.")

    def _preprocessed(self, vectors: NDArray[np.float32]) -> NDArray[np.float32]:
        """Return a fresh (n, 1) float32 array holding the one coordinate that the pair function reads."""
        return vectors[:, self.axis : self.axis + 1].copy()

    def _factory_args(self) -> tuple[str, ...]:
        return (repr(self.axis),)


# =================================================================================================
#  Marginals and joint
# =================================================================================================
@dataclass(frozen=True, repr=False)
class MarginalsAndJoint(DistanceMetric):
    """The marginals-and-joint distance; see `DistanceMetric.marginals_and_joint`."""

    joint_scale: float

    _kind = METRIC_KIND_MARGINALS_AND_JOINT
    _factory_name = "marginals_and_joint"

    @property
    def compiled_param(self) -> float:
        """Return the joint scale, the factor on the joint term."""
        return self.joint_scale

    @property
    def label(self) -> str:
        """Return `marginals+joint`, with the joint scale appended when it is not 1."""
        scale_suffix = f" (joint scale {self.joint_scale:g})" if self.joint_scale != 1.0 else ""
        return f"marginals+joint{scale_suffix}"

    def validate(self, vectors: NDArray[np.float32]) -> None:
        """Raise ValueError below 2 dimensions, where the distance only rescales the one coordinate gap."""
        if vectors.shape[1] < 2:
            raise ValueError(
                f"{self!r} needs at least 2 dimensions; in 1 it only rescales the one coordinate gap, "
                "so use DistanceMetric.l1_manhattan() instead."
            )

    def _factory_args(self) -> tuple[str, ...]:
        return (f"joint_scale={self.joint_scale!r}",) if self.joint_scale != 1.0 else ()
