"""Return a sorted copy of a float32 array, by a radix sort at the sizes where that beats `np.sort`.

An LSD radix sort orders fixed-width integer keys by one digit at a time, least significant digit first.
Each pass counts how many keys hold each digit value, turns the counts into the start position of each
value, and moves every key to the next free position of its digit value. A pass keeps the order of keys
with equal digits, so after the pass on the most significant digit the keys are sorted. The cost is a
fixed number of passes over the keys, with no comparisons.

The keys are the float32 bit patterns under the order-preserving transform: a non-negative float gets
its sign bit set, a negative float has all its bits inverted. Read as unsigned integers, the keys then
sort in the order of the floats, with -0.0 before +0.0 and the infinities at the ends.
"""

import numpy as np
from numpy.typing import NDArray

from max_div._core.jit import lazy_njit

# Below this size `np.sort` is as fast as the radix sort or faster, because the radix sort's fixed cost of
# clearing its count tables and setting up each pass dominates.
_RADIX_SORT_MIN_SIZE = 200

# The keys are sorted 8 bits at a time, in 4 passes; a 256-entry count table per pass stays cheap to clear.
_DIGIT_BITS = 8
_N_DIGITS = 4
_N_DIGIT_VALUES = 1 << _DIGIT_BITS
_DIGIT_MASK = np.uint32(_N_DIGIT_VALUES - 1)
_SIGN_BIT = np.uint32(0x80000000)


@lazy_njit("float32[::1](float32[::1])", inline="always", cache=True)
def sorted_copy_f32(values: NDArray[np.float32]) -> NDArray[np.float32]:
    """Return the values in ascending order, as a new array; `values` is left unchanged.

    The result equals `np.sort(values)`, except in 2 cases that `np.sort` leaves open or orders
    differently: -0.0 comes before +0.0, and a NaN sorts by its bit pattern, not last.

    A pass whose digit is the same for every key moves nothing and is skipped, which happens for the
    most significant digit when the values share their sign and most of their exponent.
    """
    n = values.shape[0]
    if n < _RADIX_SORT_MIN_SIZE:
        return np.sort(values)

    # --- keys, and the count of each digit value per pass ---
    bits = values.view(np.uint32)
    keys = np.empty(n, dtype=np.uint32)
    counts = np.zeros((_N_DIGITS, _N_DIGIT_VALUES), dtype=np.int32)
    for i in range(n):
        if bits[i] & _SIGN_BIT:
            key = ~bits[i]
        else:
            key = bits[i] | _SIGN_BIT
        keys[i] = key
        for digit in range(_N_DIGITS):
            counts[digit, (key >> np.uint32(digit * _DIGIT_BITS)) & _DIGIT_MASK] += 1

    # --- 1 pass per digit, least significant first ---
    source = keys
    destination = np.empty(n, dtype=np.uint32)
    starts = np.empty(_N_DIGIT_VALUES, dtype=np.int32)
    for digit in range(_N_DIGITS):
        shift = np.uint32(digit * _DIGIT_BITS)
        if counts[digit, (source[0] >> shift) & _DIGIT_MASK] == n:
            continue
        start = 0
        for value in range(_N_DIGIT_VALUES):
            starts[value] = start
            start += counts[digit, value]
        for i in range(n):
            value = (source[i] >> shift) & _DIGIT_MASK
            destination[starts[value]] = source[i]
            starts[value] += 1
        source, destination = destination, source

    # --- back from keys to values ---
    sorted_bits = np.empty(n, dtype=np.uint32)
    for i in range(n):
        if source[i] & _SIGN_BIT:
            sorted_bits[i] = source[i] ^ _SIGN_BIT
        else:
            sorted_bits[i] = ~source[i]
    return sorted_bits.view(np.float32)
