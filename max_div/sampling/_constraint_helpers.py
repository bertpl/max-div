import numba
import numpy as np
from numba.typed import List
from numpy.typing import NDArray


# =================================================================================================
#  CONSTRUCTORS for numpy-based constraint representation
# =================================================================================================
#
#   Constraints:
#      [
#           Constraint(int_set={0,1,2,3,4},   min_count=2, max_count=3),
#           Constraint(int_set={10,11,12,13}, min_count=0, max_count=7),
#           Constraint(int_set={3,11},        min_count=2, max_count=2),
#      ]
#
#   Will be represented to 2 numpy arrays:
#
#     con_values:
#         np.array([
#             [2, 3],      # min_count, max_count for constraint 0
#             [0, 7],      # min_count, max_count for constraint 1
#             [2, 2],      # min_count, max_count for constraint 2
#         ], dtype=np.int32)
#
#     con_indices:
#         -> Part 1 - first 2*m values indicate start/end indices in the array for each constraint
#         -> Part 2 - followed by concatenated indices from each constraint's int_set
#
#                  |-------- Part 1 ----------|----------- Part 2 ---------------|
#           index:  0   1     2   3     4  5    6      10    11       14    15 16
#
#         np.array([6, 11,   11, 15,   15,17,   0,1,2,3,4,   10,11,12,13,    3,11], dtype=np.int32)
#
#                     |        |         |      ^^^^^^^^^    ^^^^^^^^^^^     ^^^^
#                     |        |         |          |            |            |
#                     |        |         +----->    |            |        con 2 indices
#                     |        +------------->      |        con 1 indices
#                     +-------------------->    con 2 indices
#
# =================================================================================================
def _build_array_repr(
    cons: list["Constraint"],
) -> tuple[NDArray[np.int32], NDArray[np.int32]]:
    """
    Convert list of Constraint objects to numba-compatible representation:
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
        con_indices[2 * i] = np.int32(i_start)
        con_indices[(2 * i) + 1] = np.int32(i_start + len(con.int_set))
        for idx in sorted(con.int_set):
            con_indices[i_start] = np.int32(idx)
            i_start += 1

    return con_values, con_indices


def _build_con_membership(
    m: np.int32,
    constraints: list["Constraint"],
) -> dict[np.int32, list[np.int32]]:
    """Build a mapping from each index to the list of constraints it belongs to."""
    con_membership: dict[np.int32, list[np.int32]] = {i: [] for i in np.arange(m, dtype=np.int32)}
    for i_con, con in enumerate(constraints):
        i_con = np.int32(i_con)
        for idx in con.int_set:
            con_membership[np.int32(idx)].append(i_con)
    return con_membership


# =================================================================================================
#  LOW-LEVEL HANDLING of numpy-based constraint representation
# =================================================================================================
@numba.njit(numba.int32(numba.int32[:, :], numba.int32), inline="always")
def _np_con_min_value(con_values: NDArray[np.int32], i_con: np.int32) -> np.int32:
    """Return min_value of i-th constraint from con_values array."""
    return con_values[i_con, 0]


@numba.njit(numba.int32(numba.int32[:, :], numba.int32), inline="always")
def _np_con_max_value(con_values: NDArray[np.int32], i_con: np.int32) -> np.int32:
    """Return max_value of i-th constraint from con_values array."""
    return con_values[i_con, 1]


@numba.njit(numba.int32[:](numba.int32[:], numba.int32), inline="always")
def _np_con_indices(con_indices: NDArray[np.int32], i_con: np.int32) -> NDArray[np.int32]:
    """Return the indices array for the i-th constraint from con_indices array."""
    start = con_indices[2 * i_con]
    end = con_indices[2 * i_con + 1]
    return con_indices[start:end]


@numba.njit(inline="always")
def _np_con_build_index_sets(
    con_indices: NDArray[np.int32],
    m: np.int32,
) -> List[set[np.int32]]:
    """Build list of sets of indices for each constraint from con_indices array."""
    list_of_sets = List()
    for i in np.arange(m, dtype=np.int32):
        list_of_sets.append(set(_np_con_indices(con_indices, i)))
    return list_of_sets


@numba.njit(inline="always")
def _np_con_satisfied(
    con_values: NDArray[np.int32],
    con_indices: NDArray[np.int32],
    samples: NDArray[np.int32],
) -> bool:
    """Check if the given samples satisfy all constraints."""
    m = con_values.shape[0]

    for i_con in np.arange(m, dtype=np.int32):
        min_val = _np_con_min_value(con_values, i_con)
        max_val = _np_con_max_value(con_values, i_con)
        indices = _np_con_indices(con_indices, i_con)

        count = np.int32(0)
        for s in samples:
            if _is_int_in_sorted_array(indices, s):
                count += 1

        if count < min_val or count > max_val:
            return False

    return True


@numba.njit(inline="always")
def _is_int_in_sorted_array(
    arr: NDArray[np.int32],
    value: np.int32,
) -> bool:
    """Check if value is in sorted array arr using binary search."""
    if len(arr) < 32:
        # Linear search on small array
        for val in arr:
            if val == value:
                return True
        return False
    else:
        # Binary search on sorted array
        left = np.int32(0)
        right = np.int32(len(arr) - 1)

        while left <= right:
            if left == right:
                return arr[left] == value
            else:
                mid = (left + right) // 2
                mid_val = arr[mid]

                if mid_val == value:
                    return True
                elif mid_val < value:
                    left = mid + 1
                else:
                    right = mid - 1

        return False


@numba.njit(inline="always")
def _np_con_total_violation(con_values: NDArray[np.int32]) -> np.int32:
    """
    Return in total by how much constraints are not satisfied, assuming they represent how many _additional_ samples
    to select from each constraint.
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
