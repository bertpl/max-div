import numpy as np
from numpy.typing import ArrayLike, NDArray

from max_div._core.solver._solver_state import SolverState

from ._base import InitializationStrategy


class InitGivenSelection(InitializationStrategy):
    """Initialize with a given selection of k items; see `InitializationStrategy.given_selection`.

    The indices are checked in 2 stages, because n and k are unknown when the strategy is created:
    the constructor runs the checks that need neither n nor k, and `check_fits_problem_size` checks
    the indices against the problem's n and k.
    """

    def __init__(self, indices: ArrayLike) -> None:
        """Initialize the strategy from the starting selection's indices.

        Raises:
            ValueError: If `indices` is not a 1-dimensional integer array, or holds a negative or
                duplicate index.
        """
        super().__init__("InitGivenSelection()")
        self._indices: NDArray[np.int64] = self._as_checked_indices(indices)

    def check_fits_problem_size(self, n: int, k: int) -> None:
        """Raise ValueError unless the selection holds exactly `k` indices, each below `n`."""
        if self._indices.size != k:
            raise ValueError(f"A given selection needs exactly k={k} indices; got {self._indices.size}.")
        if self._indices.max() >= n:
            raise ValueError(f"Index {self._indices.max()} lies outside the population of n={n} items.")

    def get_next_samples(self, state: SolverState, k_remaining: int | np.int32) -> NDArray[np.int32]:
        # the whole selection is returned at once, so this single call is the first moment the
        # strategy sees the problem's n and k
        self.check_fits_problem_size(int(state.n), int(state.k))
        return self._indices.astype(np.int32)

    @staticmethod
    def _as_checked_indices(indices: ArrayLike) -> NDArray[np.int64]:
        """Return `indices` as an int64 array, after the checks that need neither n nor k.

        Raises:
            ValueError: If `indices` is not a 1-dimensional integer array, or holds a negative or
                duplicate index.
        """
        index_array = np.asarray(indices)
        if index_array.ndim != 1 or not np.issubdtype(index_array.dtype, np.integer):
            raise ValueError(
                f"A given selection needs a 1-dimensional array of integer indices; "
                f"got {index_array.ndim} dimension(s) of {index_array.dtype}."
            )
        if index_array.size > 0 and index_array.min() < 0:
            raise ValueError(f"Index {index_array.min()} is negative; indices count from 0.")
        unique_indices, counts = np.unique(index_array, return_counts=True)
        if np.any(counts > 1):
            duplicate_index = unique_indices[counts > 1][0]
            raise ValueError(f"Index {duplicate_index} appears more than once; each item can be selected once.")
        return index_array.astype(np.int64)
