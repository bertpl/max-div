import copy
from dataclasses import dataclass
from typing import Iterable

import numpy as np
from numba.typed import List
from numpy.typing import NDArray


@dataclass
class Constraint:
    """Constraint indicating we want to sample at least `min_count` and at most `max_count` integers from `int_set`."""

    int_set: set[int]
    min_count: int
    max_count: int


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
    def m(self) -> int:
        """'m' = the number of constraints."""
        return len(self._cons)

    def add(self, indices: set[int], min_count: int, max_count: int) -> None:
        """
        Add a new constraint, indicating we want to sample at least `min_count` and at most `max_count`
        integers from `indices`.
        """
        self._cons.append(Constraint(indices, min_count, max_count))

    def all(self, deepcopy: bool = False) -> list[Constraint]:
        """
        Return list of Constraint objects.
        :param deepcopy: If True, return a deep copy of the list and its contents.
        :return: list of constraints representing what was added using `add()`.
        """
        if not deepcopy:
            return self._cons
        else:
            return copy.deepcopy(self._cons)

    def satisfied(self, samples: Iterable[int]) -> bool:
        """Check if the given samples satisfy all constraints."""
        samples_set = set(samples)

        for con in self._cons:
            count = sum(1 for s in samples_set if s in con.int_set)
            if count < con.min_count or count > con.max_count:
                return False

        return True

    def to_numpy(self) -> tuple[NDArray[np.int32], NDArray[np.int32]]:
        """Convert to 2 numpy arrays (con_values, con_indices) for use in low-level numba sampling functions."""
        from max_div.sampling._constraint_helpers import _build_array_repr

        return _build_array_repr(self._cons)
