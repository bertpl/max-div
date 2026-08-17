import numpy as np
import pytest

from max_div._core._random._rng import (
    new_rng_state,
    rand_float32,
    rand_float64,
    rand_int32,
    rand_int32_array,
    rand_int64,
    rand_nz_float32,
    rand_nz_float64,
)
from max_div._core._random._rng._core import (
    _TINY_F32,
    _TINY_F64,
    _xoroshiro128plus_next,
)


# -------------------------------------------------------------------------
#  new_rng_state
# -------------------------------------------------------------------------
def test_new_rng_state():
    # --- act --------------------------
    rng_state_1 = new_rng_state(1)
    rng_state_2 = new_rng_state(1)
    rng_state_3 = new_rng_state(3)

    # --- assert -----------------------
    assert all(rng_state_1 == rng_state_2)
    assert not all(rng_state_1 == rng_state_3)


# -------------------------------------------------------------------------
#  rand_float32
# -------------------------------------------------------------------------
def test_rand_float32_seed():
    # --- arrange ----------------------
    rng_state_1 = new_rng_state(1)
    rng_state_2 = new_rng_state(1)

    # --- act --------------------------
    f11 = rand_float32(rng_state_1)
    f12 = rand_float32(rng_state_1)
    f21 = rand_float32(rng_state_2)
    f22 = rand_float32(rng_state_2)

    # --- assert -----------------------
    assert f11 == f21
    assert f12 == f22
    assert f11 != f12
    assert f21 != f22


def test_rand_float32_stats():
    # --- arrange ----------------------
    rng_state = new_rng_state(1)

    # --- act --------------------------
    values = [rand_float32(rng_state) for _ in range(10000)]

    # --- assert -----------------------
    assert min(values) < 0.01
    assert max(values) > 0.99
    assert 0.49 < np.mean(values) < 0.51
    # 100% uniqueness not expected when sampling 10000 out of ~2^24 possible values
    assert len(set(values)) > 0.99 * len(values)


@pytest.mark.parametrize(
    "rng_state, expected_uint64, expected_float32",
    [
        (
            [0, 0],
            0,
            0.0,  # 0.0 return value is possible with this function
        ),
        (
            [np.iinfo(np.uint64).max, 0],
            np.iinfo(np.uint64).max,
            1.0 - np.finfo(np.float32).eps,
        ),
    ],
    ids=["min_value", "max_value"],
)
def test_rand_float32_exact_range(rng_state: list[int], expected_uint64: int, expected_float32: float):
    # --- arrange ----------------------
    rng_state_np = np.array(rng_state, dtype=np.uint64)

    # --- act --------------------------
    value_uint64 = _xoroshiro128plus_next(rng_state_np.copy())  # copy rng_state to avoid modifying the original
    value_float32 = rand_float32(rng_state_np)

    # --- assert -----------------------
    assert value_uint64 == expected_uint64, "rng_state should be chosen such that it produces the expected uint64 value"
    assert value_float32 == expected_float32


# -------------------------------------------------------------------------
#  rand_nz_float32
# -------------------------------------------------------------------------
@pytest.mark.parametrize(
    "rng_state, expected_uint64, expected_float32",
    [
        (
            [0, 0],
            0,
            _TINY_F32,  # 0.0 return value should NOT be possible with this function
        ),
        (
            [np.iinfo(np.uint64).max, 0],
            np.iinfo(np.uint64).max,
            1.0 - np.finfo(np.float32).eps,
        ),
    ],
    ids=["min_value", "max_value"],
)
def test_rand_nz_float32_exact_range(rng_state: list[int], expected_uint64: int, expected_float32: float):
    # --- arrange ----------------------
    rng_state_np = np.array(rng_state, dtype=np.uint64)

    # --- act --------------------------
    value_uint64 = _xoroshiro128plus_next(rng_state_np.copy())  # copy rng_state to avoid modifying the original
    value_float32 = rand_nz_float32(rng_state_np)

    # --- assert -----------------------
    assert value_uint64 == expected_uint64, "rng_state should be chosen such that it produces the expected uint64 value"
    assert value_float32 == expected_float32


# -------------------------------------------------------------------------
#  rand_float64
# -------------------------------------------------------------------------
def test_rand_float64_seed():
    # --- arrange ----------------------
    rng_state_1 = new_rng_state(1)
    rng_state_2 = new_rng_state(1)

    # --- act --------------------------
    f11 = rand_float64(rng_state_1)
    f12 = rand_float64(rng_state_1)
    f21 = rand_float64(rng_state_2)
    f22 = rand_float64(rng_state_2)

    print(type(f11), f11)

    # --- assert -----------------------
    assert f11 == f21
    assert f12 == f22
    assert f11 != f12
    assert f21 != f22


def test_rand_float64_stats():
    # --- arrange ----------------------
    rng_state = new_rng_state(1)

    # --- act --------------------------
    values = [rand_float64(rng_state) for _ in range(10000)]

    # --- assert -----------------------
    assert min(values) < 0.01
    assert max(values) > 0.99
    assert 0.49 < np.mean(values) < 0.51
    # 100% uniqueness expected when sampling 10000 out of 2^53 possible values
    assert len(set(values)) == len(values)


@pytest.mark.parametrize(
    "rng_state, expected_uint64, expected_float64",
    [
        (
            [0, 0],
            0,
            0.0,  # 0.0 return value is possible with this function
        ),
        (
            [np.iinfo(np.uint64).max, 0],
            np.iinfo(np.uint64).max,
            1.0 - np.finfo(np.float64).eps,
        ),
    ],
    ids=["min_value", "max_value"],
)
def test_rand_float64_exact_range(rng_state: list[int], expected_uint64: int, expected_float64: float):
    # --- arrange ----------------------
    rng_state_np = np.array(rng_state, dtype=np.uint64)

    # --- act --------------------------
    value_uint64 = _xoroshiro128plus_next(rng_state_np.copy())  # copy rng_state to avoid modifying the original
    value_float64 = rand_float64(rng_state_np)

    # --- assert -----------------------
    assert value_uint64 == expected_uint64, "rng_state should be chosen such that it produces the expected uint64 value"
    assert value_float64 == expected_float64


# -------------------------------------------------------------------------
#  rand_nz_float64
# -------------------------------------------------------------------------
@pytest.mark.parametrize(
    "rng_state, expected_uint64, expected_float64",
    [
        (
            [0, 0],
            0,
            _TINY_F64,  # 0.0 return value should NOT be possible with this function
        ),
        (
            [np.iinfo(np.uint64).max, 0],
            np.iinfo(np.uint64).max,
            1.0 - np.finfo(np.float64).eps,
        ),
    ],
    ids=["min_value", "max_value"],
)
def test_rand_nz_float64_exact_range(rng_state: list[int], expected_uint64: int, expected_float64: float):
    # --- arrange ----------------------
    rng_state_np = np.array(rng_state, dtype=np.uint64)

    # --- act --------------------------
    value_uint64 = _xoroshiro128plus_next(rng_state_np.copy())  # copy rng_state to avoid modifying the original
    value_float64 = rand_nz_float64(rng_state_np)

    # --- assert -----------------------
    assert value_uint64 == expected_uint64, "rng_state should be chosen such that it produces the expected uint64 value"
    assert value_float64 == expected_float64


# -------------------------------------------------------------------------
#  rand_int32
# -------------------------------------------------------------------------
def test_rand_int32_seed():
    # --- arrange ----------------------
    rng_state_1 = new_rng_state(1)
    rng_state_2 = new_rng_state(1)

    # --- act --------------------------
    i11 = rand_int32(rng_state_1, np.int32(0), np.int32(100))
    i12 = rand_int32(rng_state_1, np.int32(0), np.int32(100))
    i21 = rand_int32(rng_state_2, np.int32(0), np.int32(100))
    i22 = rand_int32(rng_state_2, np.int32(0), np.int32(100))

    print(type(i11), i11)

    # --- assert -----------------------
    assert i11 == i21
    assert i12 == i22
    assert i11 != i12
    assert i21 != i22


def test_rand_int32_stats():
    # --- arrange ----------------------
    rng_state = new_rng_state(1)
    low = np.int32(0)
    high = np.int32(100)

    # --- act --------------------------
    values = [rand_int32(rng_state, low, high) for _ in range(10000)]

    # --- assert -----------------------
    assert min(values) == 0
    assert max(values) == 99
    assert 49 < np.mean(values) < 51
    # Check range coverage
    assert len(set(values)) == 100  # Should cover all values in [0, 100)


def test_rand_int32_range():
    # --- arrange ----------------------
    rng_state = new_rng_state(1)
    low = np.int32(-50)
    high = np.int32(50)

    # --- act --------------------------
    values = [rand_int32(rng_state, low, high) for _ in range(10000)]

    # --- assert -----------------------
    assert all(low <= v < high for v in values)
    assert min(values) == -50
    assert max(values) == 49


# -------------------------------------------------------------------------
#  rand_int64
# -------------------------------------------------------------------------
def test_rand_int64_seed():
    # --- arrange ----------------------
    rng_state_1 = new_rng_state(1)
    rng_state_2 = new_rng_state(1)

    # --- act --------------------------
    i11 = rand_int64(rng_state_1, np.int64(0), np.int64(100))
    i12 = rand_int64(rng_state_1, np.int64(0), np.int64(100))
    i21 = rand_int64(rng_state_2, np.int64(0), np.int64(100))
    i22 = rand_int64(rng_state_2, np.int64(0), np.int64(100))

    print(type(i11), i11)

    # --- assert -----------------------
    assert i11 == i21
    assert i12 == i22
    assert i11 != i12
    assert i21 != i22


def test_rand_int64_stats():
    # --- arrange ----------------------
    rng_state = new_rng_state(1)
    low = np.int64(0)
    high = np.int64(100)

    # --- act --------------------------
    values = [rand_int64(rng_state, low, high) for _ in range(10000)]

    # --- assert -----------------------
    assert min(values) == 0
    assert max(values) == 99
    assert 49 < np.mean(values) < 51
    # Check range coverage
    assert len(set(values)) == 100  # Should cover all values in [0, 100)


def test_rand_int64_range():
    # --- arrange ----------------------
    rng_state = new_rng_state(1)
    low = np.int64(-50)
    high = np.int64(50)

    # --- act --------------------------
    values = [rand_int64(rng_state, low, high) for _ in range(1000)]

    # --- assert -----------------------
    assert all(low <= v < high for v in values)
    assert min(values) == -50
    assert max(values) == 49


def test_rand_int64_large_range():
    # --- arrange ----------------------
    rng_state = new_rng_state(1)
    low = np.int64(0)
    high = np.int64(1_000_000_000)

    # --- act --------------------------
    values = [rand_int64(rng_state, low, high) for _ in range(10000)]

    # --- assert -----------------------
    assert all(low <= v < high for v in values)
    assert min(values) < 100_000_000
    assert max(values) > 900_000_000
    # 100% uniqueness very likely when sampling 10000 out of 1 billion possible values
    assert len(set(values)) >= 9_999


# -------------------------------------------------------------------------
#  rand_int32_array
# -------------------------------------------------------------------------
def test_rand_int32_array_seed():
    # --- arrange ----------------------
    rng_state_1 = new_rng_state(1)
    rng_state_2 = new_rng_state(1)

    # --- act --------------------------
    arr1 = rand_int32_array(rng_state_1, np.int32(0), np.int32(100), np.int32(10))
    arr2 = rand_int32_array(rng_state_2, np.int32(0), np.int32(100), np.int32(10))

    # --- assert -----------------------
    assert np.array_equal(arr1, arr2)
    assert arr1.dtype == np.int32
    assert len(arr1) == 10


def test_rand_int32_array_different_calls():
    # --- arrange ----------------------
    rng_state = new_rng_state(1)

    # --- act --------------------------
    arr1 = rand_int32_array(rng_state, np.int32(0), np.int32(100), np.int32(10))
    arr2 = rand_int32_array(rng_state, np.int32(0), np.int32(100), np.int32(10))

    # --- assert -----------------------
    # Different calls should produce different results
    assert not np.array_equal(arr1, arr2)


def test_rand_int32_array_stats():
    # --- arrange ----------------------
    rng_state = new_rng_state(1)
    low = np.int32(0)
    high = np.int32(100)
    size = np.int32(10000)

    # --- act --------------------------
    values = rand_int32_array(rng_state, low, high, size)

    # --- assert -----------------------
    assert len(values) == 10000
    assert values.dtype == np.int32
    assert np.min(values) == 0
    assert np.max(values) == 99
    assert 49 < np.mean(values) < 51
    # Should have good coverage of the range
    assert len(set(values)) == 100  # All values in [0, 100) should appear


def test_rand_int32_array_range():
    # --- arrange ----------------------
    rng_state = new_rng_state(1)
    low = np.int32(-50)
    high = np.int32(50)
    size = np.int32(10000)

    # --- act --------------------------
    values = rand_int32_array(rng_state, low, high, size)

    # --- assert -----------------------
    assert len(values) == 10000
    assert np.all(values >= low)
    assert np.all(values < high)
    assert np.min(values) == -50
    assert np.max(values) == 49
    assert -1 < np.mean(values) < 1


def test_rand_int32_array_small_range():
    # --- arrange ----------------------
    rng_state = new_rng_state(1)
    low = np.int32(0)
    high = np.int32(5)
    size = np.int32(1000)

    # --- act --------------------------
    values = rand_int32_array(rng_state, low, high, size)

    # --- assert -----------------------
    assert len(values) == 1000
    assert np.all(values >= 0)
    assert np.all(values < 5)
    # Should have all values in the small range
    assert set(values.tolist()) == {0, 1, 2, 3, 4}


def test_rand_int32_array_single_element():
    # --- arrange ----------------------
    rng_state = new_rng_state(1)
    low = np.int32(0)
    high = np.int32(100)
    size = np.int32(1)

    # --- act --------------------------
    values = rand_int32_array(rng_state, low, high, size)

    # --- assert -----------------------
    assert len(values) == 1
    assert 0 <= values[0] < 100


def test_rand_int32_array_large_size():
    # --- arrange ----------------------
    rng_state = new_rng_state(1)
    low = np.int32(0)
    high = np.int32(1000)
    size = np.int32(100000)

    # --- act --------------------------
    values = rand_int32_array(rng_state, low, high, size)

    # --- assert -----------------------
    assert len(values) == 100000
    assert values.dtype == np.int32
    assert np.all(values >= 0)
    assert np.all(values < 1000)
    # Statistical checks
    assert 490 < np.mean(values) < 510
    # Should have high uniqueness
    assert len(set(values.tolist())) >= 999  # Almost all values should appear
