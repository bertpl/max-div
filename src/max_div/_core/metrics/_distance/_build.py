"""The build layer writes an (n, n) distance matrix, either from vectors or by expanding a condensed vector.

Each pair is computed once and written to both halves, so the matrix is exactly symmetric by
construction rather than by a repair pass — kernels read whichever of (i, j)/(j, i) suits their
access pattern, so the two halves must be bit-equal.  Every fill also zeroes the diagonal it never
computes, so a fill leaves a complete matrix whatever the buffer held beforehand.

Every fill writes into a buffer the caller supplies, and each entry point allocates one only when
none is given.  That lets a store be built straight into shared memory, with no build-then-copy.

The build from vectors comes in a sequential and a parallel variant with bit-identical output — the
parallel fill runs the same pair arithmetic under the same fastmath flags and writes each element
exactly once, so thread count cannot affect results.  `MAXDIV_PARALLEL_BUILD` picks the variant
(parallel unless set to ``"0"``).

The parallel fill cuts the columns into fixed-width blocks and parallelizes the row loop within
each block: every row above a block computes exactly one block-width of pairs, so prange can
split the work evenly across threads — parallelizing the outer row loop over the whole i<j pair
triangle leaves one thread with roughly twice the work of the average.
"""

import os

import numba
import numpy as np
from numpy.typing import NDArray

from ._metric import DistanceMetric, _metric_pair, preprocess_vectors

# Width in columns of the blocks the parallel fill cuts the pair space into.
BUILD_BLOCK_WIDTH = 64

# The fills read their input through a read-only array type and write through a writable one.  Numba
# treats read-only as the wider type — a writable argument converts to it, never the reverse — so an
# input typed this way also accepts the read-only views a DistanceStore hands out, while an output
# stays writable because it is written to.  Signature strings cannot spell a read-only array type,
# so every fill takes a signature object.
READONLY_F32_1D = numba.types.Array(numba.float32, 1, "C", readonly=True)
READONLY_F32_2D = numba.types.Array(numba.float32, 2, "C", readonly=True)
WRITABLE_F32_2D = numba.float32[:, ::1]


def parallel_build_enabled() -> bool:
    """Return whether distance builds may use multiple threads."""
    return os.environ.get("MAXDIV_PARALLEL_BUILD", "1") != "0"


# =================================================================================================
#  Entry points
# =================================================================================================
def compute_full_matrix(
    vectors: NDArray[np.float32], metric: DistanceMetric, out: NDArray[np.float32] | None = None
) -> NDArray[np.float32]:
    """Compute the full (n, n) pair-wise distance matrix, exactly symmetric by construction.

    Each pair is computed once through the same pair arithmetic the lazy reads use, over the
    vectors preprocessed for the metric.  The matrix is computed directly in float32: each pair
    accumulates in float64 across the d dimensions and narrows to float32 on store.

    Args:
        vectors: (n x d ndarray) the user's vectors, in the form `validate_vector_array_layout` accepts.
        metric: (DistanceMetric) The distance metric to use.
        out: ((n, n) ndarray) buffer to fill, allocated here when not given.

    Returns:
        ((n, n) ndarray) full pairwise-distance matrix, float32 C-contiguous — `out` itself
        whenever one was given, following numpy's convention for such a parameter.
    """
    preprocessed = preprocess_vectors(vectors, metric)
    out = _allocate_if_needed(out, preprocessed.shape[0])
    if parallel_build_enabled():
        _fill_matrix_parallel(
            preprocessed, np.int32(metric.kind), np.float64(metric.p), np.int64(BUILD_BLOCK_WIDTH), out
        )
    else:
        _fill_matrix(preprocessed, np.int32(metric.kind), np.float64(metric.p), out)
    return out


def expand_condensed(
    condensed: NDArray[np.float32], n: int, out: NDArray[np.float32] | None = None
) -> NDArray[np.float32]:
    """Expand a condensed distance vector into a full (n, n) matrix, each value written to both halves.

    Args:
        condensed: ((n*(n-1))//2 ndarray) condensed pairwise distances (scipy layout), float32
            C-contiguous.
        n: (int) number of items.
        out: ((n, n) ndarray) buffer to fill, allocated here when not given.

    Returns:
        ((n, n) ndarray) full pairwise-distance matrix, bit-equal to the condensed source —
        `out` itself whenever one was given, following numpy's convention for such a parameter.
    """
    out = _allocate_if_needed(out, n)
    _fill_matrix_from_condensed(condensed, np.int32(n), out)
    return out


def _allocate_if_needed(out: NDArray[np.float32] | None, n: int) -> NDArray[np.float32]:
    """Return `out`, or a fresh uninitialized (n, n) float32 buffer when none was given."""
    return np.empty((n, n), dtype=np.float32) if out is None else out


# =================================================================================================
#  Fills
# =================================================================================================
@numba.njit(
    numba.void(READONLY_F32_2D, numba.int32, numba.float64, WRITABLE_F32_2D),
    cache=True,
    fastmath={"reassoc", "contract"},
)
def _fill_matrix(
    vectors: NDArray[np.float32], metric_kind: np.int32, metric_p: np.float64, out: NDArray[np.float32]
) -> None:
    """Fill a full (n, n) distance matrix from vectors, sequentially; each pair written to both halves."""
    n = vectors.shape[0]
    for i in np.arange(n, dtype=np.int32):
        out[i, i] = np.float32(0.0)
        for j in np.arange(i + 1, n, dtype=np.int32):
            value = _metric_pair(vectors, metric_kind, metric_p, i, j)
            out[i, j] = value
            out[j, i] = value


@numba.njit(
    numba.void(READONLY_F32_2D, numba.int32, numba.float64, numba.int64, WRITABLE_F32_2D),
    parallel=True,
    cache=True,
    fastmath={"reassoc", "contract"},
)
def _fill_matrix_parallel(
    vectors: NDArray[np.float32],
    metric_kind: np.int32,
    metric_p: np.float64,
    block_width: np.int64,
    out: NDArray[np.float32],
) -> None:
    """Fill a full (n, n) distance matrix from vectors, in parallel; each pair written to both halves."""
    n = vectors.shape[0]
    for i in numba.prange(n):  # ty: ignore[not-iterable] -- prange is iterable inside njit; the stub doesn't know
        out[i, i] = np.float32(0.0)
    for j_block in range(0, n, block_width):
        j_end = min(j_block + block_width, n)
        for i in numba.prange(j_end):  # ty: ignore[not-iterable] -- prange is iterable inside njit; the stub doesn't know
            for j in range(max(j_block, np.int64(i) + 1), j_end):
                value = _metric_pair(vectors, metric_kind, metric_p, np.int32(i), np.int32(j))
                out[i, j] = value
                out[j, i] = value


@numba.njit(numba.void(READONLY_F32_1D, numba.int32, WRITABLE_F32_2D), cache=True)
def _fill_matrix_from_condensed(condensed: NDArray[np.float32], n: np.int32, out: NDArray[np.float32]) -> None:
    """Expand a condensed distance vector into a full (n, n) matrix: each value written to both halves."""
    idx = np.int64(0)
    for i in range(n):
        out[i, i] = np.float32(0.0)
        for j in range(i + 1, n):
            value = condensed[idx]
            idx += 1
            out[i, j] = value
            out[j, i] = value
