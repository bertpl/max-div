"""Validation and symmetry repair for user-provided distance input.

Square matrices must end up *exactly* symmetric: solver kernels read whichever of (i, j)/(j, i)
suits their access pattern, and the incremental trackers assume every read of a pair returns the
same value — any difference between the halves would desync their bookkeeping.  Exactly symmetric
input is adopted as-is (zero-copy when float32 C-contiguous); asymmetric input is repaired by
averaging each pair, disclosed via `DistanceInputWarning`.

Scans are blocked so each (i, j) block is visited together with its transpose partner — two
blocks in cache at a time over data that is O(n²) by definition.
"""

import warnings

import numba
import numpy as np
from numpy.typing import NDArray

from max_div._core._warnings import DistanceInputWarning

# side length of the square tiles the blocked scans walk; two float32 tiles fit comfortably in L2
_BLOCK = 128

# |diagonal| values above this are rejected (the solver never reads the diagonal, but a non-zero
# one signals the input is not a distance matrix)
_DIAGONAL_ATOL = 1e-6


# =================================================================================================
#  Kernels
# =================================================================================================
@numba.njit(
    "Tuple((int64, int64, int64, float64, float64))(float32[:, ::1], int64, int64, int64, int64)",
    inline="always",
    cache=True,
)
def _scan_square_block(
    matrix: NDArray[np.float32], ib: int, jb: int, i_end: int, j_end: int
) -> tuple[int, int, int, float, float]:
    """Scan the i<j pairs of one block of a square matrix, together with their transpose partners.

    Returns:
        Tuple ``(n_nonfinite, n_negative, n_asymmetric_pairs, max_abs_delta, max_rel_delta)``
        where the deltas compare the (i, j) and (j, i) halves of each pair and the relative delta
        is against the larger magnitude of the two.
    """
    n_nonfinite = 0
    n_negative = 0
    n_asym = 0
    max_abs_delta = 0.0
    max_rel_delta = 0.0
    for i in range(ib, i_end):
        j_start = jb if jb > i else i + 1
        for j in range(j_start, j_end):
            a = matrix[i, j]
            b = matrix[j, i]
            a_finite = np.isfinite(a)
            b_finite = np.isfinite(b)
            n_nonfinite += int(not a_finite) + int(not b_finite)
            n_negative += int(a_finite and a < 0.0) + int(b_finite and b < 0.0)
            if a_finite and b_finite and a != b:
                n_asym += 1
                delta = abs(np.float64(a) - np.float64(b))
                max_abs_delta = max(max_abs_delta, delta)
                max_rel_delta = max(max_rel_delta, delta / max(abs(np.float64(a)), abs(np.float64(b))))
    return n_nonfinite, n_negative, n_asym, max_abs_delta, max_rel_delta


@numba.njit("Tuple((int64, int64, float64, int64, float64, float64))(float32[:, ::1])", cache=True)
def _scan_square(matrix: NDArray[np.float32]) -> tuple[int, int, float, int, float, float]:
    """Read-only blocked scan of a square distance matrix.

    Returns:
        Tuple ``(n_nonfinite, n_negative, max_abs_diagonal, n_asymmetric_pairs, max_abs_delta,
        max_rel_delta)``, aggregating `_scan_square_block` over all block pairs plus a diagonal
        sweep.
    """
    n = matrix.shape[0]
    n_nonfinite = 0
    n_negative = 0
    max_abs_diag = 0.0
    n_asym = 0
    max_abs_delta = 0.0
    max_rel_delta = 0.0
    for i in range(n):
        v = matrix[i, i]
        if not np.isfinite(v):
            n_nonfinite += 1
        else:
            max_abs_diag = max(max_abs_diag, np.float64(abs(v)))
    for ib in range(0, n, _BLOCK):
        for jb in range(ib, n, _BLOCK):
            nf, neg, asym, d_abs, d_rel = _scan_square_block(
                matrix, ib, jb, min(ib + _BLOCK, n), min(jb + _BLOCK, n)
            )
            n_nonfinite += nf
            n_negative += neg
            n_asym += asym
            max_abs_delta = max(max_abs_delta, d_abs)
            max_rel_delta = max(max_rel_delta, d_rel)
    return n_nonfinite, n_negative, max_abs_diag, n_asym, max_abs_delta, max_rel_delta


@numba.njit("void(float32[:, ::1])", cache=True)
def _symmetrize_square(matrix: NDArray[np.float32]) -> None:
    """Make `matrix` exactly symmetric in place: each differing (i, j)/(j, i) pair gets its mean."""
    n = matrix.shape[0]
    for ib in range(0, n, _BLOCK):
        for jb in range(ib, n, _BLOCK):
            i_end = min(ib + _BLOCK, n)
            j_end = min(jb + _BLOCK, n)
            for i in range(ib, i_end):
                j_start = jb if jb > i else i + 1
                for j in range(j_start, j_end):
                    a = matrix[i, j]
                    b = matrix[j, i]
                    if a != b:
                        mean = np.float32(0.5 * (np.float64(a) + np.float64(b)))
                        matrix[i, j] = mean
                        matrix[j, i] = mean


@numba.njit("Tuple((int64, int64))(float32[::1])", cache=True)
def _scan_condensed(values: NDArray[np.float32]) -> tuple[int, int]:
    """Single-pass scan of a condensed distance vector: counts of non-finite and negative values."""
    n_nonfinite = 0
    n_negative = 0
    for idx in range(values.size):
        v = values[idx]
        if not np.isfinite(v):
            n_nonfinite += 1
        elif v < 0.0:
            n_negative += 1
    return n_nonfinite, n_negative


# =================================================================================================
#  Validated adoption
# =================================================================================================
def validated_square_distances(distances: np.ndarray) -> NDArray[np.float32]:
    """Validate a square distance matrix for solver use and return it float32 C-contiguous.

    Zero-copy when the input already is float32 C-contiguous; any conversion copy is disclosed via
    `DistanceInputWarning`.  Asymmetric input is repaired to exact symmetry by averaging each
    (i, j)/(j, i) pair — in place, which for zero-copy-adopted input means the provided array is
    modified; the warning reports the largest absolute and relative deltas so the caller can judge
    whether the asymmetry was float noise or a data problem.  Raw values are validated *before*
    averaging: a negative or non-finite value anywhere is rejected, never averaged away.

    Raises:
        ValueError: On a non-square shape, fewer than 3 items, non-finite or negative values, or a
            diagonal that is not zero within tolerance.
    """
    if distances.shape[0] != distances.shape[1]:
        raise ValueError(f"Square distance matrix must be (n, n); got {distances.shape}.")
    if distances.shape[0] < 3:
        raise ValueError("At least 3 items are required to formulate a max-div problem.")

    converted = np.ascontiguousarray(distances, dtype=np.float32)
    if converted is not distances:
        warnings.warn(
            f"Square distance input required a conversion copy (dtype {distances.dtype} or memory layout); "
            "pass a float32 C-contiguous array to avoid the copy and enable zero-copy adoption.",
            DistanceInputWarning,
            stacklevel=3,
        )

    n_nonfinite, n_negative, max_abs_diag, n_asym, max_abs_delta, max_rel_delta = _scan_square(converted)
    if n_nonfinite:
        raise ValueError("Distances must all be finite (no NaN or inf).")
    if n_negative:
        raise ValueError("Distances must all be non-negative.")
    if max_abs_diag > _DIAGONAL_ATOL:
        raise ValueError("Square distance matrix must have a zero diagonal.")
    if n_asym:
        _symmetrize_square(converted)
        modified_note = " The provided array itself was modified." if converted is distances else ""
        warnings.warn(
            f"Asymmetric square distance matrix: {n_asym} (i, j)/(j, i) pairs differ "
            f"(max |delta| = {max_abs_delta:.3e}, max relative delta = {max_rel_delta:.3e}); "
            f"symmetrized in place by averaging each pair.{modified_note}",
            DistanceInputWarning,
            stacklevel=3,
        )
    return converted


def validated_condensed_distances(distances: np.ndarray) -> NDArray[np.float32]:
    """Validate a condensed distance vector for solver use and return it float32 C-contiguous.

    Zero-copy when the input already is float32 C-contiguous; any conversion copy is disclosed via
    `DistanceInputWarning`.  The condensed layout stores each pair once, so no symmetry question
    exists here.

    Raises:
        ValueError: On a non-triangular length, fewer than 3 items, or non-finite or negative values.
    """
    n = _n_from_condensed_size(distances.size)
    if n < 3:
        raise ValueError("At least 3 items are required to formulate a max-div problem.")

    converted = np.ascontiguousarray(distances, dtype=np.float32)
    if converted is not distances:
        warnings.warn(
            f"Condensed distance input required a conversion copy (dtype {distances.dtype} or memory layout); "
            "pass a float32 C-contiguous array to avoid the copy and enable zero-copy adoption.",
            DistanceInputWarning,
            stacklevel=3,
        )

    n_nonfinite, n_negative = _scan_condensed(converted)
    if n_nonfinite:
        raise ValueError("Distances must all be finite (no NaN or inf).")
    if n_negative:
        raise ValueError("Distances must all be non-negative.")
    return converted


def _n_from_condensed_size(size: int) -> int:
    """Return n such that n*(n-1)/2 == size, raising ValueError if no such integer exists."""
    n = round((1 + np.sqrt(1 + 8 * size)) / 2)
    if (n * (n - 1)) // 2 != size:
        raise ValueError(f"Condensed distance vector has invalid length {size}: not a triangular number n*(n-1)/2.")
    return n
