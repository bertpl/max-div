import numpy as np
import pytest

from max_div._core._math.select_k_minmax import select_k_max, select_k_min


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("k", [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024])
def test_select_k_min_correctness(k: int, dtype: type):
    """Test that select_k_min returns indices of the k largest elements."""

    # --- arrange ----------------------
    np.random.seed(42)
    n = 1024
    arr = np.random.randn(n).astype(dtype)

    # --- act --------------------------
    result_indices = select_k_min(arr, np.int32(k))

    # --- assert -----------------------

    # validate index invariants
    assert result_indices.dtype == np.int32
    assert len(result_indices) == k
    assert np.all(result_indices >= 0)
    assert np.all(result_indices < n)
    assert len(set(result_indices)) == k

    # ensure indices are k largest
    if k < n:
        max_k_smallest = np.max(arr[result_indices])
        min_others = np.min(np.delete(arr, result_indices))

        assert max_k_smallest <= min_others


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("k", [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024])
def test_select_k_max_correctness(k: int, dtype: type):
    """Test that select_k_max returns indices of the k largest elements."""

    # --- arrange ----------------------
    np.random.seed(42)
    n = 1024
    arr = np.random.randn(n).astype(dtype)

    # --- act --------------------------
    result_indices = select_k_max(arr, np.int32(k))

    # --- assert -----------------------

    # validate index invariants
    assert result_indices.dtype == np.int32
    assert len(result_indices) == k
    assert np.all(result_indices >= 0)
    assert np.all(result_indices < n)
    assert len(set(result_indices)) == k

    # ensure indices are k largest
    if k < n:
        min_k_largest = np.min(arr[result_indices])
        max_others = np.max(np.delete(arr, result_indices))

        assert min_k_largest >= max_others


@pytest.mark.parametrize("select_k", [select_k_min, select_k_max], ids=["min", "max"])
@pytest.mark.parametrize("k", [-1, 0, 8, 9, 100])
def test_select_k_clamps_k_into_valid_range(select_k, k: int):
    """k is clamped into [0, n]: no out-of-range k reads past the array inside compiled code."""
    # --- arrange ----------------------
    arr = np.arange(8, dtype=np.float32)

    # --- act --------------------------
    result = select_k(arr, np.int32(k))

    # --- assert -----------------------
    expected_size = min(max(k, 0), arr.shape[0])
    assert result.shape[0] == expected_size
    assert len(set(result.tolist())) == expected_size
    if expected_size:
        assert result.min() >= 0
        assert result.max() < arr.shape[0]
