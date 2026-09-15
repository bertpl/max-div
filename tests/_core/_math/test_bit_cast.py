import numpy as np
import pytest
from numba import njit

from max_div._core._math.bit_cast import float32_from_bits


@pytest.mark.parametrize(
    "bits, expected",
    [
        (np.int32(0x7F800000), np.inf),
        (np.int32(0x3F800000), 1.0),
        (np.int32(0), 0.0),
        (np.int32(1), np.float32(1e-45)),
    ],
)
def test_float32_from_bits_reinterprets_without_conversion(bits: np.int32, expected: float):
    """The bit-cast returns the float32 whose pattern the int32 holds, +inf and the smallest denormal included."""
    # --- act / assert -----------------
    assert float32_from_bits(bits) == np.float32(expected)


def test_float32_from_bits_compiles_into_a_caller():
    """A compiled caller resolves the overload and gets the same value as the interpreted body."""

    # --- arrange ----------------------
    @njit("float32(int32)")
    def caller(bits: np.int32) -> np.float32:
        return float32_from_bits(bits)

    # --- act / assert -----------------
    assert caller(np.int32(0x40490FDB)) == np.float32(3.1415927)
