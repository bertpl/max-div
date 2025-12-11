from hashlib import sha256

import numpy as np


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
