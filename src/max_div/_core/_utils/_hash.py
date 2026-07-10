from hashlib import sha256

import numba
import numpy as np
from numpy.typing import NDArray


# =================================================================================================
#  hash functions
# =================================================================================================
def deterministic_hash(obj: object) -> int:
    """
    Generate a deterministic type-aware 256-bit int hash for a given object, based on its string representation.
    """
    hash_str = f"{type(obj)}|{str(obj)}"
    return int.from_bytes(sha256(hash_str.encode()).digest()) - (2**255)  # center around 0


def deterministic_hash_int64(obj: object) -> np.int64:
    """
    Generate a deterministic type-aware int64 hash for a given object, based on its string representation.
    """
    return int_to_int64(deterministic_hash(obj))


@numba.njit(fastmath=True, inline="always", cache=True)
def np_int32_array_var_length_hash(arr: NDArray[np.int32], n: int) -> NDArray[np.int32]:
    """Takes the input array and creates an output array of length n, that represents a var-length hash of the input."""

    # create output array
    result = np.zeros(n, dtype=np.int32)

    # mix in all input values with output array
    running_hash = np.int32(0)
    n_input = arr.size
    for i in range(n_input + 2 * n):
        running_hash = (running_hash * np.int32(31)) + (result[(i + 1) % n] * np.int32(17)) + np.int32(arr[i % n_input])
        result[i % n] += running_hash

    return result


# =================================================================================================
#  Helpers
# =================================================================================================
_MIN_INT64 = np.iinfo(np.int64).min  # smallest (most negative) np.int64
_MAX_INT64 = np.iinfo(np.int64).max  # largest (most positive) np.int64


def int_to_int64(value: int) -> np.int64:
    # convert Python int -> np.int64, with silent overflow handling for large values
    # NOTE: not intended to be used in inner loops; designed for robustness, not speed.
    if _MIN_INT64 <= value <= _MAX_INT64:
        return np.int64(value)
    else:
        # Take the lower 64 bits by converting to bytes and reading as int64
        # This uses the full int64 range [-2^63, 2^63-1]
        bytes_64 = (value % (2**64)).to_bytes(8, byteorder="little", signed=False)
        return np.frombuffer(bytes_64, dtype=np.int64)[0]
