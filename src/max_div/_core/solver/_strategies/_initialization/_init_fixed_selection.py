import numpy as np
from numpy.typing import ArrayLike, NDArray

from max_div._core.solver._solver_state import SolverState

from ._base import InitializationStrategy


class InitFixedSelection(InitializationStrategy):
    """Initialize with a given selection of k items; see `InitializationStrategy.fixed_selection`.

    The indices are checked in 2 stages, each as soon as its information exists: the constructor
    checks what needs no problem, and `check_fits` checks what needs n and k.
    """

    def __init__(self, indices: ArrayLike) -> None:
        """Initialize the strategy from the indices to start from.

        Raises:
            ValueError: If `indices` is not a 1-dimensional integer array, or holds a negative or
                duplicate index.
        """
        super().__init__("InitFixedSelection()")
        self._indices: NDArray[np.int64] = self._as_checked_indices(indices)

    def check_fits(self, n: int, k: int) -> None:
        """Raise ValueError unless the selection holds exactly `k` indices, each below `n`."""
        if self._indices.size != k:
            raise ValueError(f"A fixed selection needs exactly k={k} indices; got {self._indices.size}.")
        elif self._indices.max() >= n:
            raise ValueError(f"Index {self._indices.max()} lies outside the population of n={n} items.")

    def get_next_samples(self, state: SolverState, k_remaining: int | np.int32) -> NDArray[np.int32]:
        # the first call is the first moment the strategy sees the problem's n and k
        self.check_fits(int(state.n), int(state.k))
        return self._indices.astype(np.int32)

    @staticmethod
    def _as_checked_indices(indices: ArrayLike) -> NDArray[np.int64]:
        """Return `indices` as an int64 array, after the checks that need no problem.

        Raises:
            ValueError: If `indices` is not a 1-dimensional integer array, or holds a negative or
                duplicate index.
        """
        array = np.asarray(indices)
        if array.ndim != 1 or not np.issubdtype(array.dtype, np.integer):
            raise ValueError(
                f"A fixed selection needs a 1-dimensional array of integer indices; "
                f"got {array.ndim} dimension(s) of {array.dtype}."
            )
        if array.size > 0 and array.min() < 0:
            raise ValueError(f"Index {array.min()} is negative; indices count from 0.")
        values, counts = np.unique(array, return_counts=True)
        if np.any(counts > 1):
            raise ValueError(f"Index {values[counts > 1][0]} appears more than once; each item can be selected once.")
        return array.astype(np.int64)
