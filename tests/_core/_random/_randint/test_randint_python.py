import numpy as np
import pytest
from numba import TypingError
from numpy.typing import NDArray

from max_div._core._random import P_UNIFORM, randint_python


# =================================================================================================
#  Helpers
# =================================================================================================
def get_probabilities(n: int) -> NDArray[np.float32]:
    probs = np.random.random(n)
    probs /= probs.sum()
    return probs.astype(np.float32)


# =================================================================================================
#  TEST - ARGUMENT VALIDATION
# =================================================================================================
def test_randint_python_argument_validation():
    # --- n < 1 ----------------------------------
    with pytest.raises(ValueError):
        randint_python(n=0, k=1, replace=True)

    with pytest.raises(ValueError):
        randint_python(n=-10, k=1, replace=True)

    # --- k < 1 ----------------------------------
    with pytest.raises(ValueError):
        randint_python(n=10, k=0, replace=True)

    with pytest.raises(ValueError):
        randint_python(n=10, k=-5, replace=True)

    # --- k > n when replace=False ---------------
    with pytest.raises(ValueError):
        randint_python(n=10, k=20, replace=False)

    with pytest.raises(ValueError):
        randint_python(n=100, k=101, replace=False)

    # --- sum(p)==0.0 ----------------------------
    with pytest.raises(ValueError):
        randint_python(n=5, k=2, replace=True, p=np.zeros(5).astype(np.float32))

    # --- p invalid shape ------------------------
    with pytest.raises(ValueError):
        randint_python(n=10, k=5, replace=True, p=np.array([0.1, 0.9], dtype=np.float32))  # wrong size

    with pytest.raises((TypeError, TypingError, ValueError)):
        randint_python(
            n=5,
            k=2,
            replace=True,
            p=np.array([[0.1, 0.4], [0.4, 0.1]], dtype=np.float32),
        )  # wrong shape


# =================================================================================================
#  TEST - UNIFORM - WITH REPLACEMENT
# =================================================================================================
@pytest.mark.parametrize(
    "n,k",
    [
        (10, None),
        (100, None),
        (1000, None),
        (10, 1),
        (100, 1),
        (1000, 1),
        (1000, 10),
        (1000, 100),
        (1000, 500),
        (1000, 1000),
        (1000, 10000),
    ],
)
@pytest.mark.parametrize("p", [None, P_UNIFORM])
def test_randint_python_uniform_with_replacement_invariants(p: np.ndarray | None, n: int, k: int | None):
    # --- arrange ----------------------
    k_expected = k if k is not None else 1

    # --- act --------------------------
    samples = randint_python(n=n, k=k, replace=True, p=p)

    # --- assert -----------------------
    assert isinstance(samples, np.ndarray)
    assert samples.shape == (k_expected,)
    assert samples.dtype == np.int32
    assert 0 <= samples.min() <= samples.max() < n


def test_randint_python_uniform_with_replacement_duplicates():
    # --- act --------------------------
    samples = randint_python(n=1000, k=900, replace=True)  # very unlikely all are unique

    # --- assert -----------------------
    assert len(set(samples)) < 900  # there are duplicates, which is expected with replacement


@pytest.mark.parametrize(
    "n,k",
    [
        (10, None),
        (100, None),
        (1000, None),
        (10, 1),
        (100, 1),
        (1000, 1),
        (1000, 10),
        (1000, 100),
        (1000, 500),
        (1000, 1000),
        (1000, 10000),
    ],
)
@pytest.mark.parametrize("p", [None, P_UNIFORM])
def test_randint_python_uniform_with_replacement_seed(p: np.ndarray | None, n: int, k: int | None):
    # --- arrange ----------------------
    seed_1 = 42
    seed_2 = 42
    seed_3 = 123456

    # --- act --------------------------
    samples_1 = randint_python(n, k, replace=True, p=p, seed=seed_1)
    samples_2 = randint_python(n, k, replace=True, p=p, seed=seed_2)
    samples_3 = randint_python(n, k, replace=True, p=p, seed=seed_3)

    # --- assert -----------------------
    np.testing.assert_array_equal(samples_1, samples_2)
    if (k or 1) >= 10:
        # in this case, it's very unlikely that samples_3 matches samples_1 or samples_2
        assert list(samples_1) != list(samples_3)
        assert list(samples_2) != list(samples_3)


# =================================================================================================
#  TEST - UNIFORM - WITHOUT REPLACEMENT
# =================================================================================================
@pytest.mark.parametrize(
    "n,k",
    [
        (10, None),
        (100, None),
        (1000, None),
        (10, 1),
        (100, 1),
        (1000, 1),
        (1000, 10),
        (1000, 100),
        (1000, 500),
        (1000, 1000),
    ],
)
@pytest.mark.parametrize("p", [None, P_UNIFORM])
def test_randint_python_uniform_without_replacement_invariants(p: np.ndarray | None, n: int, k: int | None):
    # --- arrange ----------------------
    k_expected = k if k is not None else 1

    # --- act --------------------------
    samples = randint_python(n, k, replace=False, p=p)

    # --- assert -----------------------
    assert isinstance(samples, np.ndarray)
    assert samples.shape == (k_expected,)
    assert samples.dtype == np.int32
    assert 0 <= samples.min() <= samples.max() < n
    assert len(set(samples)) == k_expected  # all samples are unique


@pytest.mark.parametrize(
    "n,k",
    [
        (10, None),
        (100, None),
        (1000, None),
        (10, 1),
        (100, 1),
        (1000, 1),
        (1000, 10),
        (1000, 100),
        (1000, 500),
        (1000, 1000),
    ],
)
@pytest.mark.parametrize("p", [None, P_UNIFORM])
def test_randint_python_uniform_without_replacement_seed(p: np.ndarray | None, n: int, k: int | None):
    # --- arrange ----------------------
    seed_1 = 42
    seed_2 = 42
    seed_3 = 123456

    # --- act --------------------------
    samples_1 = randint_python(n, k, replace=False, p=p, seed=seed_1)
    samples_2 = randint_python(n, k, replace=False, p=p, seed=seed_2)
    samples_3 = randint_python(n, k, replace=False, p=p, seed=seed_3)

    # --- assert -----------------------
    np.testing.assert_array_equal(samples_1, samples_2)
    if (k or 1) >= 10:
        # in this case, it's very unlikely that samples_3 matches samples_1 or samples_2
        assert list(samples_1) != list(samples_3)
        assert list(samples_2) != list(samples_3)


# =================================================================================================
#  TEST - NON-UNIFORM - WITH REPLACEMENT
# =================================================================================================
@pytest.mark.parametrize(
    "n,k",
    [
        (10, None),
        (100, None),
        (1000, None),
        (10, 1),
        (100, 1),
        (1000, 1),
        (1000, 10),
        (1000, 100),
        (1000, 500),
        (1000, 1000),
        (1000, 10000),
    ],
)
def test_randint_python_non_uniform_with_replacement_invariants(n: int, k: int | None):
    # --- arrange ----------------------
    p = get_probabilities(n)
    p_before = p.copy()  # copy for later comparison
    k_expected = k if k is not None else 1

    # --- act --------------------------
    samples = randint_python(n, k, replace=True, p=p)

    # --- assert -----------------------
    assert isinstance(samples, np.ndarray)
    assert samples.shape == (k_expected,)
    assert samples.dtype == np.int32
    assert 0 <= samples.min() <= samples.max() < n
    assert np.array_equal(p, p_before), "p array should never be modified."


def test_randint_python_non_uniform_with_replacement_duplicates():
    # --- arrange ----------------------
    p = get_probabilities(1000)
    p_before = p.copy()  # copy for later comparison

    # --- act --------------------------
    samples = randint_python(n=1000, k=900, replace=True, p=p)  # very unlikely all are unique

    # --- assert -----------------------
    assert len(set(samples)) < 900  # there are duplicates, which is expected with replacement
    assert np.array_equal(p, p_before), "p array should never be modified."


@pytest.mark.parametrize(
    "n,k",
    [
        (10, None),
        (100, None),
        (1000, None),
        (10, 1),
        (100, 1),
        (1000, 1),
        (1000, 10),
        (1000, 100),
        (1000, 500),
        (1000, 1000),
        (1000, 10000),
    ],
)
def test_randint_python_non_uniform_with_replacement_seed(n: int, k: int | None):
    # --- arrange ----------------------
    p = get_probabilities(n)
    p_before = p.copy()  # copy for later comparison
    seed_1 = 42
    seed_2 = 42
    seed_3 = 123456

    # --- act --------------------------
    samples_1 = randint_python(n, k, replace=True, p=p, seed=seed_1)
    samples_2 = randint_python(n, k, replace=True, p=p, seed=seed_2)
    samples_3 = randint_python(n, k, replace=True, p=p, seed=seed_3)

    # --- assert -----------------------

    # check results
    np.testing.assert_array_equal(samples_1, samples_2)
    if (k or 1) >= 10:
        # in this case, it's very unlikely that samples_3 matches samples_1 or samples_2
        assert list(samples_1) != list(samples_3)
        assert list(samples_2) != list(samples_3)

    # check p unmodified
    assert np.array_equal(p, p_before), "p array should never be modified."


@pytest.mark.parametrize("factor", [2.0, 5.0, 10.0, 100.0, 1000.0, 0.0])
@pytest.mark.parametrize("sum_of_p", [1.0, 0.1, 10.0])
def test_randint_python_non_uniform_with_replacement_probs(factor: float, sum_of_p: float):
    # --- arrange ----------------------
    n = 10000
    k = 1000
    p = get_probabilities(n)
    p[int(0.9 * n) :] = factor * p[int(0.9 * n) :]  # multiply last 10% of probs with factor
    p = p * (sum_of_p / p.sum())  # ensure sum(p) = sum_of_p, so we also test non-normalized probs

    expected_mean = sum(i * p[i] for i in range(n)) / sum_of_p

    p_before = p.copy()  # copy for later comparison

    # --- act --------------------------
    samples = randint_python(n=n, k=k, replace=True, p=p)

    # --- assert -----------------------
    assert 0.9 * expected_mean < np.mean(samples) < 1.1 * expected_mean
    assert all(p[i] > 0 for i in samples)
    assert np.array_equal(p, p_before), "p array should never be modified."


# =================================================================================================
#  TEST - NON-UNIFORM - WITHOUT REPLACEMENT
# =================================================================================================
@pytest.mark.parametrize(
    "n,k",
    [
        (10, None),
        (100, None),
        (1000, None),
        (10, 1),
        (100, 1),
        (1000, 1),
        (1000, 10),
        (1000, 100),
        (1000, 500),
        (1000, 1000),
    ],
)
def test_randint_python_non_uniform_without_replacement_invariants(n: int, k: int | None):
    # --- arrange ----------------------
    p = get_probabilities(n)
    p_before = p.copy()  # copy for later comparison
    k_expected = k if k is not None else 1

    # --- act --------------------------
    samples = randint_python(n, k, replace=False, p=p)

    # --- assert -----------------------
    assert isinstance(samples, np.ndarray)
    assert samples.shape == (k_expected,)
    assert samples.dtype == np.int32
    assert 0 <= samples.min() <= samples.max() < n
    assert len(set(samples)) == k_expected  # all samples are unique
    assert np.array_equal(p, p_before), "p array should never be modified."


@pytest.mark.parametrize(
    "n,k",
    [
        (10, None),
        (100, None),
        (1000, None),
        (10, 1),
        (100, 1),
        (1000, 1),
        (1000, 10),
        (1000, 100),
        (1000, 500),
        (1000, 1000),
    ],
)
def test_randint_python_non_uniform_without_replacement_seed(n: int, k: int | None):
    # --- arrange ----------------------
    p = get_probabilities(n)
    p_before = p.copy()  # copy for later comparison
    seed_1 = 42
    seed_2 = 42
    seed_3 = 123456

    # --- act --------------------------
    samples_1 = randint_python(n, k, replace=False, p=p, seed=seed_1)
    samples_2 = randint_python(n, k, replace=False, p=p, seed=seed_2)
    samples_3 = randint_python(n, k, replace=False, p=p, seed=seed_3)

    # --- assert -----------------------

    # check results
    np.testing.assert_array_equal(samples_1, samples_2)
    if (k or 1) >= 10:
        # in this case, it's very unlikely that samples_3 matches samples_1 or samples_2
        assert list(samples_1) != list(samples_3)
        assert list(samples_2) != list(samples_3)

    # check p unmodified
    assert np.array_equal(p, p_before), "p array should never be modified."


@pytest.mark.parametrize("factor", [2.0, 5.0, 10.0, 100.0, 1000.0, 0.0])
@pytest.mark.parametrize("sum_of_p", [1.0, 0.1, 10.0])
def test_randint_python_non_uniform_without_replacement_probs(factor: float, sum_of_p: float):
    # --- arrange ----------------------
    n = 10000
    k = 1000
    p = get_probabilities(n)
    p[int(0.8 * n) :] = factor * p[int(0.8 * n) :]  # multiply last 20% of probs with factor
    p = p * (sum_of_p / p.sum())  # ensure sum(p) = sum_of_p, so we also test non-normalized probs

    expected_mean = sum(i * p[i] for i in range(n)) / sum_of_p  # approx. correct; this assumes replacement

    p_before = p.copy()  # copy for later comparison

    # --- act --------------------------
    samples = randint_python(n=n, k=k, replace=False, p=p)

    # --- assert -----------------------
    assert 0.9 * expected_mean < np.mean(samples) < 1.1 * expected_mean
    assert all(p[i] > 0 for i in samples)
    assert np.array_equal(p, p_before), "p array should never be modified."


# =================================================================================================
#  Test if we never select i such that p[i]
# =================================================================================================
#
#     -----------------------------------------------------------------------------------
#  --------- NOTE: commented out to not clutter test stats with 2000 skipped tests ---------
#     -----------------------------------------------------------------------------------
#
# @pytest.mark.skip(reason="Very slow; only for manual checks.")
# @pytest.mark.parametrize("seed_offset", [np.int64(42 + (i * 1_000_000_000)) for i in range(1000)])
# @pytest.mark.parametrize("replace", [False, True])
# def test_randint_python_numba_zero_probability_selection(seed_offset: np.int64, replace: bool):
#     """Test that `randint_numba` never selects indices with zero probability, by triggering it 1_000_000_000 times."""
#     # --- arrange -----------------------------------------
#     n = 4
#     k = 2
#     p = np.array([0.0, 0.5, 0.0, 0.5], dtype=np.float32)
#     i_selected: set[np.int32] = set()
#
#     # --- act ---------------------------------------------
#     for seed in np.arange(seed_offset, seed_offset + 1_000_000, dtype=np.int64):
#         samples = randint_python(n=n, k=k, replace=replace, p=p, seed=seed)
#         for s in samples:
#             i_selected.add(s)
#
#     # --- assert ------------------------------------------
#     for i in i_selected:
#         assert p[i] > 0.0, f"Selected index {i} with zero probability!"
