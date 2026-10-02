"""`DistanceMetric` says which distance is meant; each kind of distance is a subclass of it.

The compiled pairwise distance functions in `_pairwise_distance` branch on an int selector, the
metric's `kind`, so the metric object and those functions use one numbering of the distances.
"""

import math
from dataclasses import dataclass
from typing import ClassVar

import numba
import numpy as np
from numpy.typing import NDArray

from max_div._core.jit import lazy_njit

from ._vector_layout import validate_vector_array_layout

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
METRIC_KIND_L2_AND_PROJECTIONS = 16

# NO_FLOAT_PARAM is the float passed to the compiled pairwise distance function for a kind that takes no float
# parameter (every kind that takes one requires it to be > 0, so 0.0 is free to mean "none").
NO_FLOAT_PARAM = 0.0


# =================================================================================================
#  DistanceMetric
# =================================================================================================
@dataclass(frozen=True, repr=False)
class DistanceMetric:
    """A distance metric records which distance is meant and what its compiled pairwise distance function needs.

    Create instances via the factory methods only; they canonicalize their arguments, so metrics
    that compute the same distance compare equal.  Each kind of distance is a subclass that stores
    only its own arguments and overrides the members whose behavior differs for that kind.
    """

    # Each subclass sets `_factory_name`, read by `__repr__`, and sets `_kind` and `_label` unless it
    # overrides `kind` or `label`.
    _kind: ClassVar[int]
    _label: ClassVar[str]
    _factory_name: ClassVar[str]
    needs_preprocessed_vectors: ClassVar[bool] = (
        False  # `preprocess` returns a new array when True, the input when False
    )

    def __post_init__(self) -> None:
        """Reject a bare `DistanceMetric`: only the subclasses returned by the factory methods compute a distance."""
        if type(self) is DistanceMetric:
            raise TypeError("Create a DistanceMetric through its factory methods, e.g. DistanceMetric.l2_euclidean().")

    # --------------------------------------------------------------------------
    #  Factory methods
    # --------------------------------------------------------------------------
    @classmethod
    def l1_manhattan(cls) -> "DistanceMetric":
        """Return the L1 (Manhattan) distance metric: ``sum_i |x_i - y_i|``."""
        return L1ManhattanDistanceMetric()

    @classmethod
    def l2_euclidean(cls) -> "DistanceMetric":
        """Return the L2 (Euclidean) distance metric: ``sqrt( sum_i (x_i - y_i)^2 )``."""
        return L2EuclideanDistanceMetric()

    @classmethod
    def l2s_euclidean_squared(cls) -> "DistanceMetric":
        """Return the squared L2 (Euclidean squared) distance metric: ``sum_i (x_i - y_i)^2``.

        The squared form avoids the square root and produces identical solutions under the
        GEOMEAN_SEPARATION diversity metric.
        """
        return L2sEuclideanSquaredDistanceMetric()

    @classmethod
    def linf_chebyshev(cls) -> "DistanceMetric":
        """Return the Linf (Chebyshev) distance metric: ``max_i |x_i - y_i|``."""
        return LinfChebyshevDistanceMetric()

    @classmethod
    def cosine(cls) -> "DistanceMetric":
        """Return the cosine distance metric: ``1 - (x . y) / (|x| |y|)``.

        The range is [0, 2].  Zero vectors have no defined angle and are rejected with an error.
        """
        return CosineDistanceMetric()

    @classmethod
    def geometric_mean(cls) -> "DistanceMetric":
        """Return the geometric-mean distance metric: ``( prod_i |x_i - y_i| )^(1/d)``.

        The diversity concepts page explains what this distance rewards and cites the design
        literature behind it.  A shared coordinate value makes the distance zero.

        It is not a strict metric (distinct points can be at distance zero, and the triangle
        inequality fails); the solver relies on neither.  It costs one ``log`` per dimension.
        """
        return GeometricMeanDistanceMetric()

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
        return LMinusInfDistanceMetric()

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
        return AlongAxisDistanceMetric(axis=axis)

    @classmethod
    def l2_and_projections(cls, l2_scale: float = 1.0) -> "DistanceMetric":
        """Return the L2-and-projections distance: ``min( min_i |a_i - b_i|, l2_scale * ||a - b||_2^d )``.

        For 2 vectors a and b of dimension d, the first part, the L-∞ part, is the `l_minus_inf()`
        distance: the smallest gap between their projections onto a single coordinate axis.  The second
        part, the L2 part, is their L2 distance raised to the power d.

        Under min-separation a selection is then spread in its projection onto every coordinate axis and
        in the full space at once.

        The 2 parts are comparable only for a population that fills the unit cube [0, 1]^d, so scale
        the vectors into it first.  It needs at least 2 dimensions: in 1 it only rescales the one
        coordinate gap, and a problem over 1-dimensional vectors rejects it.

        For k well-spread points in the unit cube, the gap along an axis between neighbors can reach
        1/k, while the nearest-neighbor L2 distance raised to the power d is about c/k, where the
        constant c grows with d:

        - c ≈ 1.15 for d = 2
        - c ≈ 1.4 for d = 3
        - c ≈ 2.8 for d = 5
        - c ≈ 40 for d = 10

        In higher dimensions the L2 part is therefore larger than the gap and rarely sets the
        minimum; an `l2_scale` of about 1/c gives the 2 parts equal weight.

        The L2-and-projections distance is not a metric in the mathematical sense (points that share any one
        coordinate are at distance zero, and the triangle inequality fails); the solver relies on neither.

        Args:
            l2_scale: The positive, finite factor on the L2 part.

        Raises:
            ValueError: If `l2_scale` is not a positive, finite number.
        """
        return L2AndProjectionsDistanceMetric(l2_scale=l2_scale)

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
            return MinkowskiDistanceMetric(p=p, has_outer_root=root)

    # --------------------------------------------------------------------------
    #  What the compiled pairwise distance function needs
    # --------------------------------------------------------------------------
    @property
    def kind(self) -> int:
        """Return the compiled pairwise distance functions' kind selector."""
        return self._kind

    @property
    def float_param(self) -> float:
        """Return the compiled pairwise distance function's float parameter; `NO_FLOAT_PARAM` for a kind without one."""
        return NO_FLOAT_PARAM

    @property
    def pairwise_distance_args(self) -> tuple[np.int32, np.float64]:
        """Return `kind` and `float_param` typed as the compiled pairwise distance functions take them."""
        return np.int32(self.kind), np.float64(self.float_param)

    def validate(self, vectors: NDArray[np.float32]) -> None:
        """Raise ValueError if this metric cannot be computed on `vectors`.

        `validate` is a no-op for a kind that computes on any vectors, so a caller can call it on every
        metric.  `vectors` must already be a 2D array; `preprocess` checks the layout, this method does not.
        """

    def preprocess(self, vectors: NDArray[np.float32]) -> NDArray[np.float32]:
        """Return the vectors in the form that this metric's pairwise distance function reads.

        `needs_preprocessed_vectors` says whether this is a new array or the input itself.  The input is never written.

        Raises:
            ValueError: If `vectors` is not a 2D float32 C-contiguous array, or `validate` rejects it.
        """
        validate_vector_array_layout(vectors)
        self.validate(vectors)
        return self._transform_checked_vectors(vectors)

    def _transform_checked_vectors(self, vectors: NDArray[np.float32]) -> NDArray[np.float32]:
        """Return the array that the pairwise distance function reads; by default `vectors` itself.

        `preprocess` checks `vectors` and then calls this method, so a subclass overrides it to transform
        the vectors without repeating the checks.
        """
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
        return f"DistanceMetric.{self._factory_name}({', '.join(self._factory_arg_reprs())})"

    def _factory_arg_reprs(self) -> tuple[str, ...]:
        """Return the factory arguments as source text, in the factory's order, leaving out those at their default."""
        return ()


# =================================================================================================
#  Minkowski family
# =================================================================================================
@dataclass(frozen=True, repr=False)
class L1ManhattanDistanceMetric(DistanceMetric):
    """This metric is the L1 (Manhattan) distance; see `DistanceMetric.l1_manhattan`."""

    _kind = METRIC_KIND_L1
    _label = "L1"
    _factory_name = "l1_manhattan"


@dataclass(frozen=True, repr=False)
class L2EuclideanDistanceMetric(DistanceMetric):
    """This metric is the L2 (Euclidean) distance; see `DistanceMetric.l2_euclidean`."""

    _kind = METRIC_KIND_L2
    _label = "L2"
    _factory_name = "l2_euclidean"


@dataclass(frozen=True, repr=False)
class L2sEuclideanSquaredDistanceMetric(DistanceMetric):
    """This metric is the squared L2 distance; see `DistanceMetric.l2s_euclidean_squared`."""

    _kind = METRIC_KIND_L2S
    _label = "L2²"
    _factory_name = "l2s_euclidean_squared"


@dataclass(frozen=True, repr=False)
class LinfChebyshevDistanceMetric(DistanceMetric):
    """This metric is the Linf (Chebyshev) distance; see `DistanceMetric.linf_chebyshev`."""

    _kind = METRIC_KIND_LINF
    _label = "L∞"
    _factory_name = "linf_chebyshev"


@dataclass(frozen=True, repr=False)
class MinkowskiDistanceMetric(DistanceMetric):
    """This metric is the Minkowski distance for a p other than 1, 2 and inf; see `DistanceMetric.minkowski`."""

    p: float
    has_outer_root: bool

    _factory_name = "minkowski"

    # A specialized Minkowski kind has p built in, so its pairwise distance function takes no parameter.
    _SPECIALIZED_KINDS: ClassVar[dict[tuple[float, bool], int]] = {
        (0.5, True): METRIC_KIND_MINKOWSKI_P05,
        (0.5, False): METRIC_KIND_MINKOWSKI_P05_POWERED,
        (0.25, True): METRIC_KIND_MINKOWSKI_P025,
        (0.25, False): METRIC_KIND_MINKOWSKI_P025_POWERED,
        (0.125, True): METRIC_KIND_MINKOWSKI_P0125,
        (0.125, False): METRIC_KIND_MINKOWSKI_P0125_POWERED,
    }

    @property
    def kind(self) -> int:
        """Return the specialized kind for (p, has_outer_root) when one exists, else the generic Minkowski kind."""
        generic_kind = METRIC_KIND_MINKOWSKI if self.has_outer_root else METRIC_KIND_MINKOWSKI_POWERED
        return self._SPECIALIZED_KINDS.get((self.p, self.has_outer_root), generic_kind)

    @property
    def float_param(self) -> float:
        """Return p for the generic kinds; a specialized kind has p built in and takes none."""
        return self.p if self.kind in (METRIC_KIND_MINKOWSKI, METRIC_KIND_MINKOWSKI_POWERED) else NO_FLOAT_PARAM

    @property
    def label(self) -> str:
        """Return `L<p>`, with `-powered` appended when the root is skipped."""
        return f"L{self.p:g}{'' if self.has_outer_root else '-powered'}"

    def _factory_arg_reprs(self) -> tuple[str, ...]:
        """Return `p`, plus `root=False` when the root is skipped."""
        return (f"p={self.p!r}",) if self.has_outer_root else (f"p={self.p!r}", "root=False")


# =================================================================================================
#  Angular
# =================================================================================================
@dataclass(frozen=True, repr=False)
class CosineDistanceMetric(DistanceMetric):
    """This metric is the cosine distance; see `DistanceMetric.cosine`.

    Its pairwise distance function reads rows scaled to unit L2 norm, so the distance is half the squared L2
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

    def _transform_checked_vectors(self, vectors: NDArray[np.float32]) -> NDArray[np.float32]:
        """Return a fresh float32 array with each row scaled to unit L2 norm."""
        return _normalize_rows(vectors)


# =================================================================================================
#  Coordinate-wise
# =================================================================================================
@dataclass(frozen=True, repr=False)
class GeometricMeanDistanceMetric(DistanceMetric):
    """This metric is the geometric-mean distance; see `DistanceMetric.geometric_mean`."""

    _kind = METRIC_KIND_GEOMEAN
    _label = "geomean"
    _factory_name = "geometric_mean"


@dataclass(frozen=True, repr=False)
class LMinusInfDistanceMetric(DistanceMetric):
    """This metric is the L-∞ distance; see `DistanceMetric.l_minus_inf`."""

    _kind = METRIC_KIND_LMINUSINF
    _label = "L-∞"
    _factory_name = "l_minus_inf"


@dataclass(frozen=True, repr=False)
class AlongAxisDistanceMetric(DistanceMetric):
    """This metric is the distance along one coordinate axis; see `DistanceMetric.along_axis`.

    Its pairwise distance function reads column 0 of its input array, so `preprocess` slices the axis out into
    an (n, 1) array and the compiled pairwise distance function never receives the axis itself.
    """

    axis: int

    _kind = METRIC_KIND_ALONG_AXIS
    _factory_name = "along_axis"
    needs_preprocessed_vectors = True

    def __post_init__(self) -> None:
        """Reject an axis that is not a non-negative integer, and store it as a plain int."""
        if isinstance(self.axis, bool) or not isinstance(self.axis, (int, np.integer)) or self.axis < 0:
            raise ValueError(f"along_axis requires a non-negative integer axis; here: {self.axis!r}.")
        # A frozen dataclass rejects `self.axis = ...`; `object.__setattr__` is the documented way to set a field in
        # `__post_init__`.  Storing a plain int makes `along_axis(np.int64(2))` equal to `along_axis(2)`.
        object.__setattr__(self, "axis", int(self.axis))

    @property
    def label(self) -> str:
        """Return `axis <i>`."""
        return f"axis {self.axis}"

    def validate(self, vectors: NDArray[np.float32]) -> None:
        """Raise ValueError if the axis is not a coordinate of `vectors`."""
        n_dims = vectors.shape[1]
        if self.axis >= n_dims:
            raise ValueError(f"{self!r} reads a coordinate that {n_dims}-dimensional vectors do not have.")

    def _transform_checked_vectors(self, vectors: NDArray[np.float32]) -> NDArray[np.float32]:
        """Return a fresh (n, 1) float32 array holding the coordinate along `axis`."""
        return vectors[:, self.axis : self.axis + 1].copy()

    def _factory_arg_reprs(self) -> tuple[str, ...]:
        """Return the axis."""
        return (repr(self.axis),)


# =================================================================================================
#  L2 and projections
# =================================================================================================
@dataclass(frozen=True, repr=False)
class L2AndProjectionsDistanceMetric(DistanceMetric):
    """This metric is the L2-and-projections distance; see `DistanceMetric.l2_and_projections`."""

    l2_scale: float

    _kind = METRIC_KIND_L2_AND_PROJECTIONS
    _factory_name = "l2_and_projections"

    def __post_init__(self) -> None:
        """Reject a L2 scale that is not a positive, finite number, and store it as a float."""
        l2_scale = float(self.l2_scale)
        if not (math.isfinite(l2_scale) and l2_scale > 0):
            raise ValueError(f"l2_and_projections requires a positive, finite l2_scale; here: {l2_scale}.")
        # A frozen dataclass rejects `self.l2_scale = ...`, so the float is stored with `object.__setattr__`.
        object.__setattr__(self, "l2_scale", l2_scale)

    @property
    def float_param(self) -> float:
        """Return the L2 scale, the factor on the L2 part."""
        return self.l2_scale

    @property
    def label(self) -> str:
        """Return `L2+projections`, with the L2 scale appended when it is not 1."""
        scale_suffix = f" (L2 scale {self.l2_scale:g})" if self.l2_scale != 1.0 else ""
        return f"L2+projections{scale_suffix}"

    def validate(self, vectors: NDArray[np.float32]) -> None:
        """Raise ValueError for 1-dimensional vectors, where the distance is only a rescaled L1 distance."""
        if vectors.shape[1] < 2:
            raise ValueError(
                f"{self!r} needs at least 2 dimensions; in 1 dimension it is only a rescaled L1 distance, "
                "so use DistanceMetric.l1_manhattan() instead."
            )

    def _factory_arg_reprs(self) -> tuple[str, ...]:
        """Return `l2_scale` when it is not 1, else nothing."""
        return (f"l2_scale={self.l2_scale!r}",) if self.l2_scale != 1.0 else ()


# =================================================================================================
#  Helpers
# =================================================================================================
# `_normalize_rows` serves only the cosine metric class, but stays module-level because numba compiles it.
@lazy_njit(numba.float32[:, ::1](numba.types.Array(numba.float32, 2, "C", readonly=True)), cache=True)
def _normalize_rows(vectors: NDArray[np.float32]) -> NDArray[np.float32]:
    """Scale each row to unit L2 norm into a fresh float32 array.

    Norms accumulate in float64 and each element narrows to float32 on store, so the result is the
    exact normalization the cosine pairwise distance functions operate on.  Rows must not be all-zero.
    """
    n = vectors.shape[0]
    d = vectors.shape[1]
    normalized = np.empty((n, d), dtype=np.float32)
    for i in range(n):
        acc = np.float64(0.0)
        for c in range(d):
            acc += np.float64(vectors[i, c]) * np.float64(vectors[i, c])
        norm = np.sqrt(acc)
        for c in range(d):
            normalized[i, c] = np.float32(np.float64(vectors[i, c]) / norm)
    return normalized
