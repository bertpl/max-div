"""Return a sorted copy of a float32 array, by a radix sort.

An LSD radix sort orders fixed-width integer keys by one digit at a time, least significant digit first.
Each pass:

- counts how many keys hold each digit value;
- turns the counts into the output position where the keys with each digit value start;
- writes every key to the next free output position for its digit value.

A pass keeps the order of keys with equal digits, so after the pass on the most significant digit the
keys are sorted. The cost is a fixed number of passes over the keys, with no comparisons.

The keys are the float32 bit patterns after a transform that preserves their order:

- a non-negative float gets its sign bit set;
- a negative float has all its bits inverted.

Read as unsigned integers, the keys then sort in the order of the floats, with -0.0 before +0.0 and the
infinities at the ends.
"""

import numpy as np
from numpy.typing import NDArray

from max_div._core.jit import lazy_njit

# The keys are sorted 1 digit per pass; a digit this narrow keeps the count table of each pass cheap to clear.
_DIGIT_BITS = 8
_N_DIGITS = 4
_N_DIGIT_VALUES = 1 << _DIGIT_BITS
_DIGIT_MASK = np.uint32(_N_DIGIT_VALUES - 1)
_SIGN_BIT = np.uint32(0x80000000)


@lazy_njit("float32[::1](float32[::1])", inline="always", cache=True)
def sorted_copy_f32(values: NDArray[np.float32]) -> NDArray[np.float32]:
    """Return the values in ascending order, as a new array; `values` is left unchanged.

    The result equals `np.sort(values)` except in 2 cases:

    - -0.0 comes before +0.0, which `np.sort` treats as equal;
    - a NaN sorts by its bit pattern, so one with its sign bit set comes first, where `np.sort` puts every
      NaN last.

    Below about 200 values `np.sort` is faster: the radix sort spends a fixed time of a few tenths of a
    microsecond on clearing its digit-value counts and setting up each pass.
    """
    n = values.shape[0]
    keys, counts = _keys_and_digit_counts(values)

    # --- 1 pass per digit -----------------------
    source = keys
    destination = np.empty(n, dtype=np.uint32)
    next_positions = np.empty(_N_DIGIT_VALUES, dtype=np.int32)
    for digit_position in range(_N_DIGITS):
        shift = np.uint32(digit_position * _DIGIT_BITS)
        # When every key has the same digit at this position, the pass would leave the keys in place, so
        # it is skipped; with no keys there is nothing to move either. Every key has the same most
        # significant digit when the values share their sign and most of their exponent.
        if n == 0 or counts[digit_position, (source[0] >> shift) & _DIGIT_MASK] == n:
            continue
        start = 0
        for digit_value in range(_N_DIGIT_VALUES):
            next_positions[digit_value] = start
            start += counts[digit_position, digit_value]
        for i in range(n):
            digit_value = (source[i] >> shift) & _DIGIT_MASK
            destination[next_positions[digit_value]] = source[i]
            next_positions[digit_value] += 1
        source, destination = destination, source

    return _values_from_keys(source)


# ==================================================================================================
#  Helpers
# ==================================================================================================
# Numba compiles these, so they are module-level functions.
@lazy_njit("Tuple((uint32[::1], int32[:, ::1]))(float32[::1])", inline="always", cache=True)
def _keys_and_digit_counts(values: NDArray[np.float32]) -> tuple[NDArray[np.uint32], NDArray[np.int32]]:
    """Return the sort key of each value, and per digit position the number of keys with each digit value.

    The counts of every digit position come from this one sweep over the keys: a pass only reorders the
    keys, so the counts are the same before each pass.
    """
    n = values.shape[0]
    bits = values.view(np.uint32)
    keys = np.empty(n, dtype=np.uint32)
    counts = np.zeros((_N_DIGITS, _N_DIGIT_VALUES), dtype=np.int32)
    for i in range(n):
        if bits[i] & _SIGN_BIT:
            key = ~bits[i]
        else:
            key = bits[i] | _SIGN_BIT
        keys[i] = key
        for digit_position in range(_N_DIGITS):
            counts[digit_position, (key >> np.uint32(digit_position * _DIGIT_BITS)) & _DIGIT_MASK] += 1
    return keys, counts


@lazy_njit("float32[::1](uint32[::1])", inline="always", cache=True)
def _values_from_keys(keys: NDArray[np.uint32]) -> NDArray[np.float32]:
    """Return the float32 value of each sort key, undoing the order-preserving transform."""
    n = keys.shape[0]
    bits = np.empty(n, dtype=np.uint32)
    for i in range(n):
        if keys[i] & _SIGN_BIT:
            bits[i] = keys[i] ^ _SIGN_BIT
        else:
            bits[i] = ~keys[i]
    return bits.view(np.float32)
