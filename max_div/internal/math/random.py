"""
Custom simple random number generation module for integration in numba oriented code.  Faster than numpy.random
even when used inside numba.njit.

This is based on the following:
  - https://www.pcg-random.org/posts/bounded-rands.html
  - xoroshiro128+ algorithm by David Blackman and Sebastiano Vigna (http://xoroshiro.di.unimi.it/)
  - splitmix64 for seed initialization by Sebastiano Vigna (http://xorshift.di.unimi.it/splitmix64.c)
"""

import numba
import numpy as np
from numpy import float32, float64, int32, int64, uint64
from numpy.typing import NDArray

# =================================================================================================
#  Constants
# =================================================================================================

# Constant for converting uint64 range to float64 in [0.0, 1.0)
_UINT64_TO_FLOAT64 = float64((1.0 - np.finfo(float64).eps) / (2**64))  # map 2**64 -> (1.0 - eps_f64)
_TINY_F64 = np.float64((2**-8) * np.finfo(float64).eps)  # small (<< EPS_F64) non-zero float64, far from underflow

# Constant for converting uint64 range to float32 in [0.0, 1.0)
_UINT64_TO_FLOAT32 = float32((1.0 - np.finfo(float32).eps) / (2**64))  # map 2**64 -> (1.0 - eps_f32)
_TINY_F32 = np.float32((2**-8) * np.finfo(float32).eps)  # small (<< EPS_F32) non-zero float32, far from underflow


# =================================================================================================
#  Core
# =================================================================================================
@numba.njit(fastmath=True, inline="always")
def rotl(x: uint64, k: uint64) -> uint64:
    """Rotate left operation"""
    return (x << k) | (x >> (uint64(64) - k))


@numba.njit(fastmath=True, inline="always")
def _xoroshiro128plus_next(rng_state: NDArray[uint64]) -> uint64:
    """Generate next random uint64 and update state in-place"""
    s0 = rng_state[0]
    s1 = rng_state[1]
    result = s0 + s1

    s1 ^= s0
    rng_state[0] = rotl(s0, uint64(24)) ^ s1 ^ (s1 << uint64(16))
    rng_state[1] = rotl(s1, uint64(37))

    return result


@numba.njit(fastmath=True, inline="always")
def _splitmix64_next(init_state: NDArray[uint64]) -> uint64:
    """Used to initialize xoroshiro128+ state from single seed; state is a 1-element array, modified in-place."""
    z = init_state[0] + uint64(0x9E3779B97F4A7C15)
    init_state[0] = z
    z = (z ^ (z >> uint64(30))) * uint64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> uint64(27))) * uint64(0x94D049BB133111EB)
    return z ^ (z >> uint64(31))


# =================================================================================================
#  Interface
# =================================================================================================
@numba.njit(fastmath=True, inline="always")
def set_seed(seed: np.int64) -> NDArray[uint64]:
    """Initialize xoroshiro128+ state from single seed; using splitmix64 algorithm."""
    init_state = np.array([seed], dtype=uint64)

    state = np.empty(2, dtype=uint64)
    state[0] = _splitmix64_next(init_state)
    state[1] = _splitmix64_next(init_state)

    return state


@numba.njit("float64(uint64[:])", fastmath=True, inline="always")
def rand_float64(rng_state: NDArray[uint64]) -> float64:
    """Generate a random float64 in [0.0, 1.0) using the provided rng_state."""
    rnd_uint64 = _xoroshiro128plus_next(rng_state)
    return float64(rnd_uint64) * _UINT64_TO_FLOAT64


@numba.njit("float64(uint64[:])", fastmath=True, inline="always")
def rand_nz_float64(rng_state: NDArray[uint64]) -> float64:
    """Generate a random float64 in (0.0, 1.0) using the provided rng_state."""
    rnd_uint64 = _xoroshiro128plus_next(rng_state)
    # To ensure results >0.0, we add a very small positive number.
    #  --> Small enough to not influence the max. value or any of the relevant statistics.
    #  --> Large enough to avoid exact 0.0 results and steer clear from underflow
    return float64(rnd_uint64) * _UINT64_TO_FLOAT64 + _TINY_F64  # should compile to FMA instruction


@numba.njit("float32(uint64[:])", fastmath=True, inline="always")
def rand_float32(rng_state: NDArray[uint64]) -> float32:
    """Generate a random float32 in [0.0, 1.0) using the provided rng_state."""
    rnd_uint64 = _xoroshiro128plus_next(rng_state)
    return float32(rnd_uint64) * _UINT64_TO_FLOAT32


@numba.njit("float32(uint64[:])", fastmath=True, inline="always")
def rand_nz_float32(rng_state: NDArray[uint64]) -> float32:
    """Generate a random float32 in (0.0, 1.0) using the provided rng_state."""
    rnd_uint64 = _xoroshiro128plus_next(rng_state)
    # To ensure results >0.0, we add a very small positive number.
    #  --> Small enough to not influence the max. value or any of the relevant statistics.
    #  --> Large enough to avoid exact 0.0 results and steer clear from underflow
    return float32(rnd_uint64) * _UINT64_TO_FLOAT32 + _TINY_F32  # should compile to FMA instruction


@numba.njit("int64(uint64[:], int64, int64)", fastmath=True, inline="always")
def rand_int64(rng_state: NDArray[uint64], low: np.int64, high: np.int64) -> np.int64:
    """
    Generate a random int64 in [low, high) using the provided rng_state.
    There might be a small bias for large (high-low) if the range is not a power of two.
    """
    if low == 0:
        rnd_uint64 = _xoroshiro128plus_next(rng_state)
        return int64(rnd_uint64 % uint64(high))
    else:
        range_size = high - low
        rnd_uint64 = _xoroshiro128plus_next(rng_state)
        return low + int64(rnd_uint64 % uint64(range_size))


@numba.njit("int32(uint64[:], int32, int32)", fastmath=True, inline="always")
def rand_int32(rng_state: NDArray[uint64], low: np.int32, high: np.int32) -> np.int32:
    """
    Generate a random int32 in [low, high) using the provided rng_state.
    There might be a small bias for large (high-low) if the range is not a power of two.
    """
    if low == 0:
        rnd_uint64 = _xoroshiro128plus_next(rng_state)
        return int32(rnd_uint64 % uint64(high))
    else:
        range_size = high - low
        rnd_uint64 = _xoroshiro128plus_next(rng_state)
        return low + int32(rnd_uint64 % uint64(range_size))


@numba.njit("int32[:](uint64[:], int32, int32, int32)", fastmath=True, inline="always")
def rand_int32_array(rng_state: NDArray[uint64], low: np.int32, high: np.int32, size: np.int32) -> NDArray[np.int32]:
    """
    Generate an array of random int32 values in [low, high) using the provided rng_state.
    Optimized to generate 2 values per RNG call by using upper and lower 32 bits.
    There might be a small bias for large (high-low) if the range is not a power of two.
    """
    result = np.empty(size, dtype=np.int32)
    if low == 0:
        range_size = uint64(high)
        i = 0
        while i < size:
            rnd_uint64 = _xoroshiro128plus_next(rng_state)
            # Use lower 32 bits for first value
            result[i] = int32((rnd_uint64 & uint64(0xFFFFFFFF)) % range_size)
            i += 1
            # Use upper 32 bits for second value if needed
            if i < size:
                result[i] = int32((rnd_uint64 >> uint64(32)) % range_size)
                i += 1
    else:
        range_size = uint64(high - low)
        i = 0
        while i < size:
            rnd_uint64 = _xoroshiro128plus_next(rng_state)
            # Use lower 32 bits for first value
            result[i] = low + int32((rnd_uint64 & uint64(0xFFFFFFFF)) % range_size)
            i += 1
            # Use upper 32 bits for second value if needed
            if i < size:
                result[i] = low + int32((rnd_uint64 >> uint64(32)) % range_size)
                i += 1
    return result


# =================================================================================================
#  Public API
# =================================================================================================
__ALL__ = [
    "rand_float32",
    "rand_float64",
    "rand_int32",
    "rand_int32_array",
    "rand_int64",
    "set_seed",
]
