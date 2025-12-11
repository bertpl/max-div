import numpy as np

from max_div.internal.utils import deterministic_hash, deterministic_hash_int64, int_to_int64
from max_div.internal.utils._hash import _MAX_INT64, _MIN_INT64


# =================================================================================================
#  deterministic_hash
# =================================================================================================
def test_deterministic_hash():
    # --- arrange -----------------------------------------
    objects = [
        1,
        1.0,
        np.int64(1),
        np.float32(1.0),
        "1",
        "1.0",
        "one",
        (1,),
        (1.0,),
        [1],
        {1: "one"},
    ]

    # --- act ---------------------------------------------
    hashes = [deterministic_hash(obj) for obj in objects]

    # --- assert ------------------------------------------
    assert all(isinstance(h, int) for h in hashes)
    assert len(set(hashes)) == len(objects)
    assert max(hashes) > 2**250
    assert min(hashes) < -(2**250)


# =================================================================================================
#  deterministic_hash_int64
# =================================================================================================
def test_deterministic_hash_int64():
    # --- arrange -----------------------------------------
    objects = [
        1,
        1.0,
        np.int64(1),
        np.float32(1.0),
        "1",
        "1.0",
        "one",
        (1,),
        (1.0,),
        [1],
        {1: "one"},
    ]

    # --- act ---------------------------------------------
    hashes = [deterministic_hash_int64(obj) for obj in objects]

    # --- assert ------------------------------------------
    assert all(isinstance(h, np.int64) for h in hashes)
    assert len(set(hashes)) == len(objects)
    assert max(hashes) > 2**62
    assert min(hashes) < -(2**62)


# =================================================================================================
#  helpers
# =================================================================================================
def test_int_to_int64():
    # --- act ---------------------------------------------
    i0 = int_to_int64(100)
    i1 = int_to_int64(-100)
    i2 = int_to_int64(_MAX_INT64)  # borderline fits in a int64
    i3 = int_to_int64(_MIN_INT64)  # borderline fits in a int64
    i4 = int_to_int64(2**100)  # too large for int64
    i5 = int_to_int64(-(2**100))  # too small for int64

    # --- assert ------------------------------------------
    assert isinstance(i0, np.int64)
    assert isinstance(i1, np.int64)
    assert isinstance(i2, np.int64)
    assert isinstance(i3, np.int64)
    assert isinstance(i4, np.int64)
    assert isinstance(i5, np.int64)

    assert i0 == 100
    assert i1 == -100
    assert i2 == _MAX_INT64
    assert i3 == _MIN_INT64

    assert i2 + 1 == _MIN_INT64  # double check these constants indeed represent the extreme values if the int64 range
    assert i3 - 1 == _MAX_INT64  # double check these constants indeed represent the extreme values if the int64 range
