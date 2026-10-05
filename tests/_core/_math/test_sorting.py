import numpy as np
import pytest

from max_div._core._math.sorting import sorted_copy_f32


def _values(kind: str, n: int) -> np.ndarray:
    """Return `n` float32 values of one kind, from a fixed seed."""
    rng = np.random.default_rng([n, len(kind)])
    if kind == "narrow":
        # These values share their sign and most of their exponent, like the separations of one selection.
        return (0.05 + 0.25 * rng.random(n)).astype(np.float32)
    elif kind == "wide":
        return (10.0 ** rng.uniform(-3, 3, n)).astype(np.float32)
    elif kind == "ties":
        return rng.integers(0, 5, n).astype(np.float32)
    elif kind == "mixed_signs_and_specials":
        values = rng.standard_normal(n).astype(np.float32)
        values[::7] = 0.0
        values[1::11] = np.inf
        values[2::13] = -np.inf
        values[3::17] = 1e-45  # 1e-45 rounds to the smallest float32 denormal
        return values
    else:
        return np.full(n, 0.5, dtype=np.float32)


@pytest.mark.parametrize("kind", ["narrow", "wide", "ties", "mixed_signs_and_specials", "all_equal"])
@pytest.mark.parametrize("n", [0, 1, 2, 3, 100, 1000, 5000])
def test_sorted_copy_f32_equals_np_sort_bit_for_bit(kind: str, n: int):
    """Without -0.0 or NaN, the copy equals `np.sort` bit for bit at every size, and the input is unchanged."""
    # --- arrange ----------------------
    values = _values(kind, n)
    original = values.copy()

    # --- act --------------------------
    result = sorted_copy_f32(values)

    # --- assert -----------------------
    np.testing.assert_array_equal(result.view(np.uint32), np.sort(values).view(np.uint32))
    np.testing.assert_array_equal(values, original)


def test_sorted_copy_f32_puts_negative_zero_before_positive_zero():
    """The radix sort orders -0.0 before +0.0, which `np.sort` treats as equal."""
    # --- arrange ----------------------
    values = np.ones(10, dtype=np.float32)
    values[:2] = [0.0, -0.0]

    # --- act --------------------------
    result = sorted_copy_f32(values)

    # --- assert -----------------------
    assert np.signbit(result[0])
    assert not np.signbit(result[1])
    np.testing.assert_array_equal(result, np.sort(values))
