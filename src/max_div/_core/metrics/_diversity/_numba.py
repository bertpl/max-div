"""The reducers here turn a selection's per-item contribution values into one diversity value.

Every reducer here takes the same fastmath subset as the pair-distance functions in
`_distance/_metric/_pair.py`, for the same reason: `sep` carries +inf for "no selected neighbor yet".

## Why the minimum is taken over integers

The compiler vectorizes a sum but not a floating-point minimum: no fastmath flag makes it emit a
vector minimum for a reduction loop, so a minimum over floats runs one element at a time and costs
several times a sum of the same length.

`min_separation` therefore reads the separations' bit patterns as `int32` and takes an integer
minimum, which the compiler does vectorize.

That is exact because of how IEEE floats are laid out:

- for a non-negative float the sign bit is zero;
- the exponent sits above the mantissa;
- +inf has every exponent bit set and a zero mantissa.

So the bit pattern read as a signed 32-bit integer increases with the float value, +inf above every
finite one.

The integer minimum picks the same element as the float minimum would.  Two inputs would break
that order, and neither reaches the reducer:

- a negative separation: every pair function returns a magnitude (an absolute difference, a square
  root, a squared sum, `0.5 · |x - y|²` over normalized rows, an exponential), never a negative value;
  a -0.0 would sort below +0.0 and read back as -0.0, which equals 0.0, so even that would be harmless;
- a NaN: its pattern lies above +inf, so the integer minimum would skip it where the float minimum
  would propagate it; no pair function produces one from finite float32 vectors.
"""

import numpy as np
from numba import njit
from numpy.typing import NDArray

from max_div._core._math.bit_cast import float32_from_bits
from max_div._core._math.geomean import fast_geomean_f32, geomean_f32

# The +inf bit pattern: the integer minimum starts from it, as the float loop would start from +inf.
_INF_BITS = np.int32(0x7F800000)


@njit("float32(float32[::1])", fastmath={"reassoc", "contract"}, inline="always", cache=True)
def min_separation(sep: NDArray[np.float32]) -> np.float32:
    """Minimum separation of all selected items, as an integer minimum over the float bit patterns.

    Exact for non-negative floats and +inf, which is all a separation array holds; the module
    docstring proves it and gives the reason the float form is avoided.
    """
    bits = sep.view(np.int32)
    n = bits.shape[0]
    min_bits = _INF_BITS
    for i in range(n):
        min_bits = min(min_bits, bits[i])
    return float32_from_bits(min_bits)


@njit("float32(float32[::1])", fastmath={"reassoc", "contract"}, inline="always", cache=True)
def mean_separation(sep: NDArray[np.float32]) -> np.float32:
    """Arithmetic mean separation of all selected items."""
    return np.mean(sep)


@njit("float32(float32[::1])", fastmath={"reassoc", "contract"}, inline="always", cache=True)
def mean_pairwise_distance(mean_dists: NDArray[np.float32]) -> np.float32:
    """Mean pairwise distance among all selected items, from their mean-distance contribution values.

    Each selected item's contribution is its mean distance to the other selected items; averaging
    those per-item means over the selection yields exactly the mean over all selected pairs.
    """
    return np.mean(mean_dists)


@njit("float32(float32[::1])", fastmath={"reassoc", "contract"}, inline="always", cache=True)
def geomean_separation(sep: NDArray[np.float32]) -> np.float32:
    """Geometric mean separation of all selected items."""
    return geomean_f32(sep)


@njit("float32(float32[::1])", fastmath={"reassoc", "contract"}, inline="always", cache=True)
def approx_geomean_separation(sep: NDArray[np.float32]) -> np.float32:
    """Approximate geometric mean separation of all selected items."""
    return fast_geomean_f32(sep)


# `error_model="numpy"` makes a float division by zero yield +inf instead of raising, so the zero and
# +inf limits below need no branch in the loop, which lets it vectorize.
@njit("float32(float32[::1])", fastmath={"reassoc", "contract"}, error_model="numpy", inline="always", cache=True)
def harmonic_mean_separation(sep: NDArray[np.float32]) -> np.float32:
    """Harmonic mean separation of all selected items: their count over the sum of their reciprocals.

    Zero and +inf are the limits of that formula, reached through IEEE arithmetic: a zero separation
    puts +inf into the reciprocal sum, so the mean is zero; a +inf separation adds nothing to it, so a
    selection whose separations are all +inf gives +inf.
    """
    n = sep.shape[0]
    reciprocal_sum = np.float32(0.0)
    for i in range(n):
        reciprocal_sum += np.float32(1.0) / sep[i]
    return np.float32(n) / reciprocal_sum


@njit("float32(float32[::1])", fastmath={"reassoc", "contract"}, inline="always", cache=True)
def non_zero_separation_frac(sep: NDArray[np.float32]) -> np.float32:
    n = sep.shape[0]
    n_non_zero = np.int32(0)
    for i in range(n):
        if sep[i] != 0.0:
            n_non_zero += 1
    return np.float32(n_non_zero) / np.float32(n)
