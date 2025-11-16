"""
Custom simple random number generation module for integration in numba oriented code.  Faster than numpy.random
even when used inside numba.njit.

This is based on the following:
  - xoroshiro128+ algorithm by David Blackman and Sebastiano Vigna (http://xoroshiro.di.unimi.it/)
  - splitmix64 for seed initialization by Sebastiano Vigna (http://xorshift.di.unimi.it/splitmix64.c)
"""

import numba
import numpy as np
from numpy import float32, float64, int32, int64, uint64

# =================================================================================================
#  Constants
# =================================================================================================

# Constant for converting uint64 to float64 in [0.0, 1.0): 1.0 / 2**53
_TO_FLOAT64 = float64(1.0 / 9007199254740992.0)


# Constant for converting uint64 to float64 in [0.0, 1.0): 1.0 / 2**24
_TO_FLOAT32 = float32(1.0 / 16777216.0)


# =================================================================================================
#  Core
# =================================================================================================
GLOBAL_RNG_STATE = np.array([uint64(0x0123456789ABCDEF), uint64(0xFEDCBA9876543210)], dtype=np.uint64)


@numba.njit(fastmath=True, inline="always")
def rotl(x: uint64, k: uint64) -> uint64:
    """Rotate left operation"""
    return (x << k) | (x >> (uint64(64) - k))


@numba.njit(fastmath=True, inline="always")
def _xoroshiro128plus_next(rng_state: np.ndarray[uint64]) -> uint64:
    """Generate next random uint64 and update state in-place"""
    s0 = rng_state[0]
    s1 = rng_state[1]
    result = s0 + s1

    s1 ^= s0
    rng_state[0] = rotl(s0, uint64(24)) ^ s1 ^ (s1 << uint64(16))
    rng_state[1] = rotl(s1, uint64(37))

    return result


@numba.njit(fastmath=True)
def _splitmix64_next(init_state: np.ndarray[uint64]) -> uint64:
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
def set_seed(seed: np.int64) -> np.ndarray[uint64]:
    """Initialize xoroshiro128+ state from single seed; using splitmix64 algorithm."""
    init_state = np.array([np.int64(seed).view(np.uint64)], dtype=uint64)

    state = np.empty(2, dtype=uint64)
    state[0] = _splitmix64_next(init_state)
    state[1] = _splitmix64_next(init_state)

    return state


@numba.njit("float64(uint64[:])", fastmath=True, inline="always")
def rand_float64(rng_state: np.ndarray[uint64]) -> float64:
    """Generate a random float64 in [0.0, 1.0) using the provided rng_state."""
    rnd_uint64 = _xoroshiro128plus_next(rng_state)
    return float64((rnd_uint64 >> uint64(11)) * _TO_FLOAT64)  # 2**-53


@numba.njit("float32(uint64[:])", fastmath=True, inline="always")
def rand_float32(rng_state: np.ndarray[uint64]) -> float32:
    """Generate a random float32 in [0.0, 1.0) using the provided rng_state."""
    rnd_uint64 = _xoroshiro128plus_next(rng_state)
    return float32((rnd_uint64 >> uint64(40)) * _TO_FLOAT32)  # 2**-24


@numba.njit("int64(uint64[:], int64, int64)", fastmath=True, inline="always")
def rand_int64(rng_state: np.ndarray[uint64], low: np.int64, high: np.int64) -> np.int64:
    """
    Generate a random int64 in [low, high) using the provided rng_state.
    There might be a small bias for large (high-low) if the range is not a power of two.
    """
    range_size = high - low
    rnd_uint64 = _xoroshiro128plus_next(rng_state)
    return low + int64(rnd_uint64 % uint64(range_size))


@numba.njit("int32(uint64[:], int32, int32)", fastmath=True, inline="always")
def rand_int32(rng_state: np.ndarray[uint64], low: np.int32, high: np.int32) -> np.int32:
    """
    Generate a random int32 in [low, high) using the provided rng_state.
    There might be a small bias for large (high-low) if the range is not a power of two.
    """
    range_size = high - low
    rnd_uint64 = _xoroshiro128plus_next(rng_state)
    return low + int32(rnd_uint64 % uint64(range_size))


# =================================================================================================
#  Public API
# =================================================================================================
__ALL__ = [
    "GLOBAL_RNG_STATE",
    "rand_float32",
    "rand_float64",
    "rand_int32",
    "rand_int64",
    "set_seed",
]
