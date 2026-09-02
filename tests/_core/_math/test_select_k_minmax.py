import numpy as np
import pytest

from max_div._core._math.select_k_minmax import select_k_max, select_k_max_masked, select_k_min


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


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
@pytest.mark.parametrize("k", [1, 3, 8, 40])
def test_select_k_max_masked_matches_compacted(seed: int, k: int):
    """Masked selection must equal select_k_max over a compacted copy, slot for slot."""

    # --- arrange ----------------------
    rng = np.random.default_rng(seed)
    arr = rng.choice(np.linspace(0.0, 1.0, num=10), size=50).astype(np.float32)  # duplicates force tie handling
    excluded = rng.random(50) < 0.4

    # --- act --------------------------
    masked = select_k_max_masked(arr, np.int32(k), excluded)
    compact_positions = select_k_max(arr[~excluded], np.int32(k))
    mapped = np.flatnonzero(~excluded).astype(np.int32)[compact_positions]

    # --- assert -----------------------
    np.testing.assert_array_equal(masked, mapped)


def test_select_k_max_masked_clamps_k_and_handles_all_excluded():
    """k above the candidate count returns every candidate; no candidates returns an empty array."""

    # --- arrange ----------------------
    arr = np.array([0.5, 0.1, 0.9, 0.3], dtype=np.float32)
    half_excluded = np.array([True, False, True, False])

    # --- act / assert -----------------
    assert set(select_k_max_masked(arr, np.int32(10), half_excluded)) == {1, 3}
    assert len(select_k_max_masked(arr, np.int32(2), np.full(4, True))) == 0
    assert len(select_k_max_masked(arr, np.int32(0), half_excluded)) == 0


def test_select_k_min_ranks_inf_keys_last():
    """Keys of +inf (zero-probability items in randint) never enter the k smallest while finite keys remain."""
    # --- arrange ----------------------
    keys = np.array([np.inf, 3.0, np.inf, 1.0, 2.0, np.inf], dtype=np.float32)

    # --- act --------------------------
    picked = select_k_min(keys, np.int32(3))

    # --- assert -----------------------
    assert sorted(picked.tolist()) == [1, 3, 4]
