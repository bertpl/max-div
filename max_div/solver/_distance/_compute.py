"""
Methods for computing pair-wise distances between vectors, along the lines of scipy's pdist.
"""

import numba
import numpy as np
from numpy.typing import NDArray
from scipy.spatial.distance import pdist as scipy_pdist

from ._enum import DistanceMetric


# =================================================================================================
#  pdist computation
# =================================================================================================
def compute_pdist(vectors: NDArray[np.float32], metric: DistanceMetric) -> NDArray[np.float32]:
    """
    Compute the pair-wise Euclidean distances between a set of n vectors in d dimensions.

    NOTE: scipy's pdist always returns float64 arrays by default, even when input is float32.
          This represents an inefficiency (both computational & memory), but is not expected to be dominant.

    :param vectors: (n x d ndarray) A set of n vectors in d dimensions.
    :param metric: (DistanceMetric) The distance metric to use.
    :return: ((n*(n-1))/2 ndarray) condensed pair-wise distance vector,
                                         with (i,j)-distance at index m*i + j - ((i+2)*(i+1))//2   for   i<j.
                                              (i,j)-distance at index m*j + i - ((j+2)*(j+1))//2   for   i>j.
    """

    match metric:
        case DistanceMetric.L1_MANHATTAN:
            return scipy_pdist(vectors, metric="cityblock").astype(np.float32)
        case DistanceMetric.L2_EUCLIDEAN:
            return scipy_pdist(vectors, metric="euclidean").astype(np.float32)
        case DistanceMetric.L2S_EUCLIDEAN_SQUARED:
            return scipy_pdist(vectors, metric="sqeuclidean").astype(np.float32)


# =================================================================================================
#  Low-level
# =================================================================================================
@numba.njit("float32(float32[::1], int32, int32, int32)", inline="always")
def get_pdist_el(pdist: NDArray[np.float32], i: np.int32, j: np.int32, n: np.int32) -> np.float32:
    """Return element from 'pdist' array representing distance between vectors i & j, given n vectors in total."""
    if i == j:
        return np.float32(0.0)
    elif i < j:
        index = (n * i) + j - ((i + 2) * (i + 1)) // 2
        return pdist[index]
    else:
        index = (n * j) + i - ((j + 2) * (j + 1)) // 2
        return pdist[index]


@numba.njit("float32[::1](float32[::1], int32)")
def compute_separation(pdist: NDArray[np.float32], n: np.int32) -> NDArray[np.float32]:
    """Compute separation of each vector wrt all others, given pairwise distance array pdist and n vectors in total."""
    sep = np.full(n, fill_value=np.inf, dtype=np.float32)
    pdist_idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            # note: the way we iterate over i & j represents the exact order in which pdist stores distances
            dist_ij = pdist[pdist_idx]
            pdist_idx += 1
            sep[i] = min(sep[i], dist_ij)
            sep[j] = min(sep[j], dist_ij)
    return sep


@numba.njit("void(float32[::1], float32[::1], int32, int32)")
def update_separation_add(sep: NDArray[np.float32], pdist: NDArray[np.float32], n: np.int32, i_added: np.int32) -> None:
    """Update separation of each vector wrt selection, given pdist array and n vectors in total, after adding i_added."""
    for j in np.arange(n, dtype=np.int32):
        if j != i_added:
            dist = get_pdist_el(pdist, i_added, j, n)
            if dist < sep[j]:
                sep[j] = dist


@numba.njit("void(float32[::1], float32[::1], int32, int32, int32[::1])")
def update_separation_remove(
    sep: NDArray[np.float32],
    pdist: NDArray[np.float32],
    n: np.int32,
    i_removed: np.int32,
    new_selection: NDArray[np.int32],
) -> None:
    """Update separation of each vector wrt selection, given pdist array and n vectors in total, after removing i_removed."""
    for j in np.arange(n, dtype=np.int32):
        if j != i_removed:
            dist = get_pdist_el(pdist, i_removed, j, n)
            if dist <= sep[j]:
                # need to recompute sep[j]
                new_sep_j = np.inf
                for k in new_selection:
                    # only compute distance to currently selected vectors
                    if k != j:
                        dist_jk = get_pdist_el(pdist, j, k, n)
                        if dist_jk < new_sep_j:
                            new_sep_j = dist_jk
                sep[j] = new_sep_j
