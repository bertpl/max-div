"""
Custom simple random number generation module for integration in numba oriented code.
This core RNG functionality is used in the methods of ._public.py, which provides alternatives to some fundamental
methods in numpy.random, but with signficantly better performance when used inside numba.njit functions.

This is based on the following:
  - https://www.pcg-random.org/posts/bounded-rands.html
  - xoroshiro128+ algorithm by David Blackman and Sebastiano Vigna (http://xoroshiro.di.unimi.it/)
  - splitmix64 for seed initialization by Sebastiano Vigna (http://xorshift.di.unimi.it/splitmix64.c)
"""

import numba
import numpy as np
from numpy import float32, float64, uint64
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
@numba.njit("uint64(uint64,uint64)", fastmath=True, inline="always", cache=True)
def rotl(x: uint64, k: uint64) -> uint64:
    """Rotate left operation"""
    return (x << k) | (x >> (uint64(64) - k))


@numba.njit("uint64(uint64[:])", fastmath=True, inline="always", cache=True)
def _xoroshiro128plus_next(rng_state: NDArray[uint64]) -> uint64:
    """Generate next random uint64 and update state in-place"""
    s0 = rng_state[0]
    s1 = rng_state[1]
    result = s0 + s1

    s1 ^= s0
    rng_state[0] = rotl(s0, uint64(24)) ^ s1 ^ (s1 << uint64(16))
    rng_state[1] = rotl(s1, uint64(37))

    return result


@numba.njit("uint64(uint64[:])", fastmath=True, inline="always", cache=True)
def _splitmix64_next(init_state: NDArray[uint64]) -> uint64:
    """Used to initialize xoroshiro128+ state from single seed; init_state is a 1-element array, modified in-place."""
    z = init_state[0] + uint64(0x9E3779B97F4A7C15)
    init_state[0] = z
    z = (z ^ (z >> uint64(30))) * uint64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> uint64(27))) * uint64(0x94D049BB133111EB)
    return z ^ (z >> uint64(31))
