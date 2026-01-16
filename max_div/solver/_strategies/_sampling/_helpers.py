import numba
import numpy as np
from numpy.typing import NDArray


@numba.njit("Tuple((int32[:], float32[:]))(int32[:], float32[:], int32)", inline="always", fastmath=True, cache=True)
def remove_sample_from_candidates_and_p(
    candidates: NDArray[np.int32], p: NDArray[np.float32], sample: np.int32
) -> tuple[NDArray[np.int32], NDArray[np.float32]]:
    """
    Locates the position of 'sample' in 'candidates' and removes it from 'candidates' + removes the corresponding value
    from identically sized 'p'.

    NOTES:
     - original arrays are not modified; new arrays are returned
     - if 'sample' is not found in 'candidates', a ValueError is raised.

    :param candidates: (n-element int32-array) with candidate values
    :param p: (n-element float32-array) with probabilities
    :param sample: (int32) value to be removed from candidates (and corresponding p)
    :return: (new_candidates, new_p), each of size n-1
    """

    # --- init ---
    n = np.int32(candidates.shape[0])
    n_minus_1 = n - np.int32(1)
    new_candidates = np.empty(n - 1, dtype=np.int32)
    new_p = np.empty(n - 1, dtype=np.float32)

    # --- construct ---
    i_tgt = np.int32(0)
    for i_src in range(n):
        if candidates[i_src] != sample:
            if i_tgt == n_minus_1:
                # we reached end of the array and haven't found 'sample' --> error
                raise ValueError("Sample to be removed not found in candidates array.")
            new_candidates[i_tgt] = candidates[i_src]
            new_p[i_tgt] = p[i_src]
            i_tgt += np.int32(1)

    # --- we're done ---
    return new_candidates, new_p


@numba.njit("int32[:](int32[:], int32)", inline="always", fastmath=True, cache=True)
def remove_sample_from_candidates(candidates: NDArray[np.int32], sample: np.int32) -> NDArray[np.int32]:
    """Simplified version of remove_sample_from_candidates_and_p that only modifies 'candidates', not 'p'."""

    # --- init ---
    n = np.int32(candidates.shape[0])
    n_minus_1 = n - np.int32(1)
    new_candidates = np.empty(n - 1, dtype=np.int32)

    # --- construct ---
    i_tgt = np.int32(0)
    for i_src in range(n):
        if candidates[i_src] != sample:
            if i_tgt == n_minus_1:
                # we reached end of the array and haven't found 'sample' --> error
                raise ValueError("Sample to be removed not found in candidates array.")
            new_candidates[i_tgt] = candidates[i_src]
            i_tgt += np.int32(1)

    # --- we're done ---
    return new_candidates
