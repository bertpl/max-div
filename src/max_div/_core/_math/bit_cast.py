"""Reinterpret the bits of a scalar as another 32-bit type, without conversion.

The bit-cast is written twice, for the same reason `_utils/_sorted_index_list.py` gives in full:
numba's `viewer` emits a compile-time instruction and has no Python body, so a run with the JIT
disabled (the coverage run) cannot call it.  The plain function is the interpreted implementation,
and the `@overload` redirects compiled callers to `viewer`.
"""

import numpy as np
from numba.cpython.unsafe.numbers import viewer
from numba.extending import overload


def float32_from_bits(bits: np.int32) -> np.float32:
    """Return the `float32` whose bit pattern the `int32` holds."""
    return np.array([bits], dtype=np.int32).view(np.float32)[0]


@overload(float32_from_bits)
def _float32_from_bits_compiled(bits):  # noqa: ANN001, ANN202
    """Register the bit-cast that numba compiles in place of `float32_from_bits`."""

    def implementation(bits):  # noqa: ANN001, ANN202
        # ty: ignore[missing-argument] -- the stub counts the typingctx parameter, which callers do not pass
        return viewer(bits, np.float32)

    return implementation
