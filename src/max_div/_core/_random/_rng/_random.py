"""Custom simple random number generation module for integration in numba oriented code.

Significantly faster than equivalent numpy.random methods, even when the latter is used in numba.njit functions.

All methods accept an rng_state (random number generator state), which is a 2-element uint64 array representing the
internal state of the xoroshiro128+ algorithm and is modified in-place by each method.

An initial rng_state can be constructed from an int64 seed using new_rng_state(seed).
"""

import numba
import numpy as np
from numpy import float32, float64, int32, int64, uint64
from numpy.typing import NDArray

from ._core import (
    _TINY_F32,
    _TINY_F64,
    _UINT64_TO_FLOAT32,
    _UINT64_TO_FLOAT64,
    _splitmix64_next,
    _xoroshiro128plus_next,
)


# =================================================================================================
#  Interface
# =================================================================================================
@numba.njit("uint64[:](int64)", fastmath=True, inline="always", cache=True)
def new_rng_state(seed: np.int64) -> NDArray[uint64]:
    """Initialize new xoroshiro128+ rng_state from single seed; using splitmix64 algorithm."""
    init_state = np.array([seed], dtype=uint64)

    state = np.empty(2, dtype=uint64)
    state[0] = _splitmix64_next(init_state)
    state[1] = _splitmix64_next(init_state)

    return state


@numba.njit("float64(uint64[:])", fastmath=True, inline="always", cache=True)
def rand_float64(rng_state: NDArray[uint64]) -> float64:
    """Generate a random float64 in [0.0, 1.0) using the provided rng_state."""
    rnd_uint64 = _xoroshiro128plus_next(rng_state)
    return float64(rnd_uint64) * _UINT64_TO_FLOAT64


@numba.njit("float64(uint64[:])", fastmath=True, inline="always", cache=True)
def rand_nz_float64(rng_state: NDArray[uint64]) -> float64:
    """Generate a random float64 in (0.0, 1.0) using the provided rng_state."""
    rnd_uint64 = _xoroshiro128plus_next(rng_state)
    # To ensure results >0.0, we add a very small positive number.
    #  --> Small enough to not influence the max. value or any of the relevant statistics.
    #  --> Large enough to avoid exact 0.0 results and steer clear from underflow
    return float64(rnd_uint64) * _UINT64_TO_FLOAT64 + _TINY_F64  # should compile to FMA instruction


@numba.njit("float32(uint64[:])", fastmath=True, inline="always", cache=True)
def rand_float32(rng_state: NDArray[uint64]) -> float32:
    """Generate a random float32 in [0.0, 1.0) using the provided rng_state."""
    rnd_uint64 = _xoroshiro128plus_next(rng_state)
    return float32(rnd_uint64) * _UINT64_TO_FLOAT32


@numba.njit("float32(uint64[:])", fastmath=True, inline="always", cache=True)
def rand_nz_float32(rng_state: NDArray[uint64]) -> float32:
    """Generate a random float32 in (0.0, 1.0) using the provided rng_state."""
    rnd_uint64 = _xoroshiro128plus_next(rng_state)
    # To ensure results >0.0, we add a very small positive number.
    #  --> Small enough to not influence the max. value or any of the relevant statistics.
    #  --> Large enough to avoid exact 0.0 results and steer clear from underflow
    return float32(rnd_uint64) * _UINT64_TO_FLOAT32 + _TINY_F32  # should compile to FMA instruction


@numba.njit("int64(uint64[:], int64, int64)", fastmath=True, inline="always", cache=True)
def rand_int64(rng_state: NDArray[uint64], low: np.int64, high: np.int64) -> np.int64:
    """Generate a random int64 in [low, high) using the provided rng_state.

    There might be a small bias for large (high-low) if the range is not a power of two.
    """
    if low == 0:
        rnd_uint64 = _xoroshiro128plus_next(rng_state)
        return int64(rnd_uint64 % uint64(high))
    range_size = high - low
    rnd_uint64 = _xoroshiro128plus_next(rng_state)
    return low + int64(rnd_uint64 % uint64(range_size))


@numba.njit("uint64(uint64[:], uint64)", fastmath=True, inline="always", cache=True)
def _draw_below(rng_state: NDArray[uint64], range_size: uint64) -> uint64:
    """Draw a uniform integer in [0, range_size) from the upper 32 bits of generator words.

    The reduction is Lemire's multiply-and-shift with rejection: the high 32 bits of the product
    are the draw, its low 32 bits say how far the word fell into the draw's bucket, the run of
    upper halves that map to the same draw.  When `range_size` does not divide 2^32, some buckets
    are one word larger than the rest, and a word in that excess is thrown away, so every value is
    equally likely.  `range_size` must be below 2^32 so the product fits in 64 bits.  Modulo would
    read xoroshiro128+'s lowest bits, which are its weakest, and costs a division.
    """
    product = (_xoroshiro128plus_next(rng_state) >> uint64(32)) * range_size
    if (product & uint64(0xFFFFFFFF)) < range_size:
        # This branch is only reached for a word that may sit in an oversized bucket: the exact
        # threshold costs a modulo, so the threshold is computed here, off the common path.
        threshold = (uint64(0x100000000) - range_size) % range_size
        while (product & uint64(0xFFFFFFFF)) < threshold:
            product = (_xoroshiro128plus_next(rng_state) >> uint64(32)) * range_size
    return product >> uint64(32)


@numba.njit("int32(uint64[:], int32, int32)", fastmath=True, inline="always", cache=True)
def rand_int32(rng_state: NDArray[uint64], low: np.int32 | int, high: np.int32 | int) -> np.int32:
    """Generate a uniform random int32 in [low, high) using the provided rng_state."""
    if low == 0:
        return int32(_draw_below(rng_state, uint64(high)))
    return low + int32(_draw_below(rng_state, uint64(high - low)))


@numba.njit("int32[:](uint64[:], int32, int32, int32)", fastmath=True, inline="always", cache=True)
def rand_int32_array(
    rng_state: NDArray[uint64], low: np.int32 | int, high: np.int32 | int, size: np.int32 | int
) -> NDArray[np.int32]:
    """Generate an array of random int32 values in [low, high) using the provided rng_state.

    Each value takes at least one generator step; see `_draw_below`.
    """
    result = np.empty(size, dtype=np.int32)
    range_size = uint64(high - low)
    for i in range(size):
        result[i] = low + int32(_draw_below(rng_state, range_size))
    return result
