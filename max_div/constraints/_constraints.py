import copy

import numpy as np
from numba.typed import List

from ._constraint import Constraint


class Constraints:
    """Representation for a collection of constraints."""

    # -------------------------------------------------------------------------
    #  Constructor / initialization
    # -------------------------------------------------------------------------
    def __init__(self):
        self._cons: list[Constraint] = []

    # -------------------------------------------------------------------------
    #  API
    # -------------------------------------------------------------------------
    @property
    def n_cons(self) -> int:
        return len(self._cons)

    def add(self, indices: set[int], min_count: int, max_count: int) -> None:
        """
        Add a new constraint, indicating we want to sample at least `min_count` and at most `max_count`
        integers from `indices`.
        """
        self._cons.append(Constraint(indices, min_count, max_count))

    def all(self, deepcopy: bool = False) -> list[Constraint]:
        if not deepcopy:
            return self._cons
        else:
            return copy.deepcopy(self._cons)

    def _to_numpy(self) -> tuple[np.ndarray[np.int32], np.ndarray[np.int32]]:
        """Convert to 2 numpy arrays (con_values, con_indices) for use in internal numba functions."""
        from ._numba import _build_array_repr

        return _build_array_repr(self._cons)
