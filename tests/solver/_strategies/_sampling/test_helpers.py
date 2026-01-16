import numpy as np
import pytest

from max_div.solver._strategies._sampling import remove_sample_from_candidates, remove_sample_from_candidates_and_p


# =================================================================================================
#  remove_sample_from_candidates_and_p
# =================================================================================================
@pytest.mark.parametrize("i_sample", [0, 1, 2, 3, 4, 5])
def test_remove_sample_from_candidates_and_p(i_sample: np.int32):
    # --- arrange -----------------------------------------
    candidates = np.array([1, 10, 11, 12, 17, 9], dtype=np.int32)
    p = np.array([0.01, 0.10, 0.11, 0.12, 0.17, 0.09], dtype=np.float32)
    n = candidates.shape[0]
    sample = candidates[i_sample]

    # --- act ---------------------------------------------
    new_candidates, new_p = remove_sample_from_candidates_and_p(candidates, p, sample)

    # --- assert ------------------------------------------
    assert new_candidates.shape[0] == n - 1
    assert new_candidates.dtype == np.int32
    assert np.array_equal(new_candidates, candidates[candidates != sample])

    assert new_p.shape[0] == n - 1
    assert new_p.dtype == np.float32
    assert np.array_equal(new_p, p[candidates != sample])


@pytest.mark.parametrize("sample", [0, -1, 2, 13, 23])
def test_remove_sample_from_candidates_and_p_value_error(sample: np.int32):
    # --- arrange -----------------------------------------
    candidates = np.array([1, 10, 11, 12, 17, 9], dtype=np.int32)
    p = np.array([0.01, 0.10, 0.11, 0.12, 0.17, 0.09], dtype=np.float32)

    # --- act & assert ------------------------------------
    with pytest.raises(ValueError):
        _ = remove_sample_from_candidates_and_p(candidates, p, sample)


# =================================================================================================
#  remove_sample_from_candidates
# =================================================================================================
@pytest.mark.parametrize("i_sample", [0, 1, 2, 3, 4, 5])
def test_remove_sample_from_candidates(i_sample: np.int32):
    # --- arrange -----------------------------------------
    candidates = np.array([1, 10, 11, 12, 17, 9], dtype=np.int32)
    n = candidates.shape[0]
    sample = candidates[i_sample]

    # --- act ---------------------------------------------
    new_candidates = remove_sample_from_candidates(candidates, sample)

    # --- assert ------------------------------------------
    assert new_candidates.shape[0] == n - 1
    assert new_candidates.dtype == np.int32
    assert np.array_equal(new_candidates, candidates[candidates != sample])


@pytest.mark.parametrize("sample", [0, -1, 2, 13, 23])
def test_remove_sample_from_candidates_value_error(sample: np.int32):
    # --- arrange -----------------------------------------
    candidates = np.array([1, 10, 11, 12, 17, 9], dtype=np.int32)

    # --- act & assert ------------------------------------
    with pytest.raises(ValueError):
        _ = remove_sample_from_candidates(candidates, sample)
