"""Helper functions for handling constraints in numpy-based representation.

In order to be able to implement efficient numba-accelerated algorithms that deal with Constraint objects,
we need to convert them to numpy-based representations that numba can work with, so we can avoid numba object-mode,
which is dramatically slower.

Example:
  Constraints:
     [
          Constraint(int_set={0,1,2,3,4},   min_count=2, max_count=3),
          Constraint(int_set={10,11,12,13}, min_count=0, max_count=7),
          Constraint(int_set={3,11},        min_count=2, max_count=2),
     ]

  Will be represented to 2 numpy arrays:

    con_values:
        np.array([
            [2, 3],      # min_count, max_count for constraint 0
            [0, 7],      # min_count, max_count for constraint 1
            [2, 2],      # min_count, max_count for constraint 2
        ], dtype=np.int32)

    con_indices:
        -> Part 1 - first 2*m values indicate start/end indices in the array for each constraint
        -> Part 2 - followed by concatenated indices from each constraint's int_set

                 |-------- Part 1 ----------|----------- Part 2 ---------------|
          index:  0   1     2   3     4  5    6      10    11       14    15 16

        np.array([6, 11,   11, 15,   15,17,   0,1,2,3,4,   10,11,12,13,    3,11], dtype=np.int32)

                    |        |         |      ^^^^^^^^^    ^^^^^^^^^^^     ^^^^
                    |        |         |          |            |            |
                    |        |         +----->    |            |        con 2 indices
                    |        +------------->      |        con 1 indices
                    +-------------------->    con 0 indices

Notes:
    - use ConstraintList(constraints).to_numpy() to convert a list of Constraint objects to (con_values, con_indices)
    - con_indices is usually treated as a read-only data structure that models membership of indices to constraints
    - con_values, however, is often modified during sampling to reflect how many more samples are needed, hence to keep
                  track of constraint satisfaction as sampling / solving a problem is progressing.

"""

from dataclasses import dataclass

import numba
import numpy as np
from numpy.typing import NDArray


# =================================================================================================
#  Constraint class
# =================================================================================================
@dataclass
class Constraint:
    """Constraint indicating we want to sample at least `min_count` and at most `max_count` integers from `int_set`.

    `weight` scales how strongly this constraint's violations count toward the feasibility score
    (default 1); a larger weight makes the solver prioritize satisfying it over lower-weight constraints.
    """

    int_set: set[int]
    min_count: int
    max_count: int
    weight: float = 1.0

    def __post_init__(self) -> None:
        """Validate that the weight is strictly positive (a zero weight would hide a real violation)."""
        if self.weight <= 0:
            raise ValueError(f"Constraint weight must be > 0 (got {self.weight}).")


class ConstraintList:
    """Simple helper class to facilitate conversion of list of Constraint objects to numpy-based representation."""

    def __init__(self, constraints: list[Constraint]) -> None:
        self._cons = constraints

    def to_numpy(self) -> tuple[NDArray[np.int32], NDArray[np.int32]]:
        return _build_array_repr(self._cons)


# =================================================================================================
#  CONSTRUCTORS for numpy-based constraint representation
# =================================================================================================
def _build_array_repr(
    cons: list[Constraint],
) -> tuple[NDArray[np.int32], NDArray[np.int32]]:
    """Convert list of Constraint objects to numba-compatible representation.

      - con_values: 2D numpy array of shape (m, 2) with min_count and max_count for each constraint
      - con_indices: 1D numpy array of shape (2*m + n_indices,) with indexed, concatenated indices of all cons.

    :param cons: list of Constraint objects
    :return: tuple of (con_values, con_indices)
    """
    # get dimensions
    m = len(cons)
    n_indices = sum([len(con.int_set) for con in cons])

    # pre-allocate
    con_values = np.empty((m, 2), dtype=np.int32)
    con_indices = np.empty((2 * m) + n_indices, dtype=np.int32)

    # build con_values
    for i, con in enumerate(cons):
        con_values[i, 0] = np.int32(con.min_count)
        con_values[i, 1] = np.int32(con.max_count)

    # build con_indices
    i_start = 2 * m  # where we start filling in values from int_set for each constraint
    for i, con in enumerate(cons):
        i_end = i_start + len(con.int_set)
        con_indices[2 * i] = np.int32(i_start)
        con_indices[(2 * i) + 1] = np.int32(i_end)
        # vectorize indices assignment for speed
        segment = np.fromiter(con.int_set, dtype=np.int32, count=len(con.int_set))
        segment.sort()
        con_indices[i_start:i_end] = segment
        i_start = i_end

    return con_values, con_indices


# =================================================================================================
#  LOW-LEVEL HANDLING of numpy-based constraint representation
# =================================================================================================
@numba.njit("int32(int32[:,:],int32)", inline="always", fastmath=True, cache=True)
def _np_con_min_value(con_values: NDArray[np.int32], i_con: np.int32) -> np.int32:
    """Return min_value of i-th constraint from con_values array."""
    return con_values[i_con, 0]


@numba.njit("int32(int32[:,:],int32)", inline="always", fastmath=True, cache=True)
def _np_con_max_value(con_values: NDArray[np.int32], i_con: np.int32) -> np.int32:
    """Return max_value of i-th constraint from con_values array."""
    return con_values[i_con, 1]


@numba.njit("int32[:](int32[:],int32)", inline="always", fastmath=True, cache=True)
def _np_con_indices(con_indices: NDArray[np.int32], i_con: np.int32) -> NDArray[np.int32]:
    """Return the indices array for the i-th constraint from con_indices array."""
    start = con_indices[2 * i_con]
    end = con_indices[2 * i_con + 1]
    return con_indices[start:end]


@numba.njit("int32(int32[:])", inline="always", fastmath=True, cache=True)
def _np_largest_con_index(con_indices: NDArray[np.int32]) -> np.int32:
    """Return the largest index referenced in con_indices array."""
    m = np.int32(con_indices[0] // 2)
    largest = np.int32(-1)
    for i in range(m):
        con_indices_i = _np_con_indices(con_indices, np.int32(i))
        if con_indices_i[-1] > largest:  # indices are sorted small-to-large for each constraint, so [-1] is the largest
            largest = con_indices_i[-1]

    return largest


@numba.njit("int32(int32[:,:])", inline="always", fastmath=True, cache=True)
def _np_con_total_violation(con_values: NDArray[np.int32]) -> np.int32:
    """Return in total by how much constraints are not satisfied (unweighted, linear).

    Fast path for the common case (all weights 1, linear penalization); the weighted / quadratic
    generalization lives in `_np_con_total_weighted_violation`.

    This assumes con_values represents how many _additional_ samples to select from each constraint.
    """
    s = np.int32(0)
    for i_con in range(con_values.shape[0]):
        if con_values[i_con, 0] > 0:
            # not yet enough samples for this constraint
            s = s + con_values[i_con, 0]
        if con_values[i_con, 1] < 0:
            # too many samples for this constraint
            s = s - con_values[i_con, 1]
    return s


@numba.njit(
    numba.float32(numba.int32[:, :], numba.float32[:], numba.boolean),
    inline="always",
    fastmath=True,
    cache=True,
)
def _np_con_total_weighted_violation(
    con_values: NDArray[np.int32], con_weights: NDArray[np.float32], quadratic: bool
) -> np.float32:
    """Return the weighted total constraint violation `Σ wᵢ·pen(vᵢ)`.

    `vᵢ` is the per-constraint shortfall-plus-excess (as in `_np_con_total_violation`); `pen` is the
    square when `quadratic` else the identity.  The mode is branched once, outside the loop, so the
    per-constraint work never re-checks it.  The unweighted-linear case is served by the faster
    integer `_np_con_total_violation`; this function runs only when a weight differs from 1 or
    quadratic penalization is requested.

    This assumes con_values represents how many _additional_ samples to select from each constraint.
    """
    s = np.float32(0.0)
    if quadratic:
        for i_con in range(con_values.shape[0]):
            v = np.float32(0.0)
            if con_values[i_con, 0] > 0:
                v += np.float32(con_values[i_con, 0])
            if con_values[i_con, 1] < 0:
                v -= np.float32(con_values[i_con, 1])
            s += con_weights[i_con] * v * v
    else:
        for i_con in range(con_values.shape[0]):
            v = np.float32(0.0)
            if con_values[i_con, 0] > 0:
                v += np.float32(con_values[i_con, 0])
            if con_values[i_con, 1] < 0:
                v -= np.float32(con_values[i_con, 1])
            s += con_weights[i_con] * v
    return s


def _np_con_count_satisfied(con_values: NDArray[np.int32]) -> int:
    """Return the number of individually satisfied constraints.

    This assumes con_values represents how many _additional_ samples to select from each constraint.
    A constraint is satisfied when min_remaining <= 0 and max_remaining >= 0.
    """
    n_satisfied = 0
    for i_con in range(con_values.shape[0]):
        if con_values[i_con, 0] <= 0 and con_values[i_con, 1] >= 0:
            n_satisfied += 1
    return n_satisfied
