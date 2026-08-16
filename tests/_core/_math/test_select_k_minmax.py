import numpy as np
import pytest

from max_div._core._math.select_k_minmax import select_k_max, select_k_min


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("k", [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024])
def test_select_k_min_correctness(k: int, dtype: type):
    """Test that select_k_min returns indices of the k largest elements."""

    # --- arrange -----------------------------------------
    np.random.seed(42)
    n = 1024
    arr = np.random.randn(n).astype(dtype)

    # --- act ---------------------------------------------
    result_indices = select_k_min(arr, np.int32(k))

    # --- assert ------------------------------------------

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

    # --- arrange -----------------------------------------
    np.random.seed(42)
    n = 1024
    arr = np.random.randn(n).astype(dtype)

    # --- act ---------------------------------------------
    result_indices = select_k_max(arr, np.int32(k))

    # --- assert ------------------------------------------

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
