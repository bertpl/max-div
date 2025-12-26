import numbers

import numpy as np
import pytest
from numpy.typing import NDArray

from max_div.sampling.uncon import randint, randint_numba


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
@pytest.mark.parametrize("use_numba", [True, False])
def test_randint_argument_validation(use_numba: bool):
    # --- n < 1 ------------------------------------------
    with pytest.raises(ValueError):
        randint(n=0, use_numba=use_numba)

    with pytest.raises(ValueError):
        randint(n=-10, use_numba=use_numba)

    # --- k < 1 ------------------------------------------
    with pytest.raises(ValueError):
        randint(n=10, k=0, use_numba=use_numba)

    with pytest.raises(ValueError):
        randint(n=10, k=-5, use_numba=use_numba)

    # --- k > n when replace=False -----------------------
    with pytest.raises(ValueError):
        randint(n=10, k=20, replace=False, use_numba=use_numba)

    with pytest.raises(ValueError):
        randint(n=100, k=101, replace=False, use_numba=use_numba)

    # --- p invalid shape --------------------------------
    with pytest.raises(ValueError):
        randint(n=10, k=5, replace=True, p=np.array([0.1, 0.9], dtype=np.float32), use_numba=use_numba)  # wrong size

    with pytest.raises(ValueError):
        randint(
            n=4, k=2, replace=True, p=np.array([[0.1, 0.4], [0.4, 0.1]], dtype=np.float32), use_numba=use_numba
        )  # wrong shape


# =================================================================================================
#  TEST - UNIFORM - WITH REPLACEMENT
# =================================================================================================
@pytest.mark.parametrize("n", [10, 100, 1000, 10000])
@pytest.mark.parametrize("use_numba", [True, False])
@pytest.mark.parametrize("p", [None, np.zeros(0, np.float64)])
def test_randint_uniform_with_replacement_invariants_single(n: int, use_numba: bool, p: np.ndarray | None):
    # --- arrange -----------------------------------------
    if p is not None:
        p_before = p.copy()  # copy for later comparison

    # --- act ---------------------------------------------
    sample = randint(n, replace=True, p=p, use_numba=use_numba)

    # --- assert ------------------------------------------
    assert isinstance(sample, numbers.Integral)
    assert 0 <= sample < n
    if p is not None:
        assert np.array_equal(p, p_before), "p array should never be modified."


@pytest.mark.parametrize(
    "n,k",
    [
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
@pytest.mark.parametrize("use_numba", [True, False])
@pytest.mark.parametrize("p", [None, np.zeros(0, np.float64)])
def test_randint_uniform_with_replacement_invariants_multiple(n: int, k: int, use_numba: bool, p: np.ndarray | None):
    # --- arrange -----------------------------------------
    if p is not None:
        p_before = p.copy()  # copy for later comparison

    # --- act ---------------------------------------------
    samples = randint(n, k, replace=True, p=p, use_numba=use_numba)

    # --- assert ------------------------------------------
    assert isinstance(samples, np.ndarray)
    assert samples.shape == (k,)
    assert samples.dtype == np.int32
    assert 0 <= samples.min() <= samples.max() < n
    if p is not None:
        assert np.array_equal(p, p_before), "p array should never be modified."


@pytest.mark.parametrize("use_numba", [True, False])
@pytest.mark.parametrize("p", [None, np.zeros(0, np.float64)])
def test_randint_uniform_with_replacement_duplicates(use_numba: bool, p: np.ndarray | None):
    # --- arrange -----------------------------------------
    if p is not None:
        p_before = p.copy()  # copy for later comparison

    # --- act ---------------------------------------------
    samples = randint(n=1000, k=900, replace=True, p=p, use_numba=use_numba)  # very unlikely all are unique

    # --- assert ------------------------------------------
    assert len(set(samples)) < 900  # there are duplicates, which is expected with replacement
    if p is not None:
        assert np.array_equal(p, p_before), "p array should never be modified."


@pytest.mark.parametrize(
    "n,k",
    [
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
@pytest.mark.parametrize("use_numba", [True, False])
@pytest.mark.parametrize("p", [None, np.zeros(0, np.float64)])
def test_randint_uniform_with_replacement_seed(n: int, k: int, use_numba: bool, p: np.ndarray | None):
    # --- arrange -----------------------------------------
    if p is not None:
        p_before = p.copy()  # copy for later comparison

    # --- act ---------------------------------------------
    samples_1 = randint(n, k, replace=True, p=p, use_numba=use_numba, seed=123)
    samples_2 = randint(n, k, replace=True, p=p, use_numba=use_numba, seed=123)
    samples_3 = randint(n, k, replace=True, p=p, use_numba=use_numba, seed=None)
    samples_4 = randint(n, k, replace=True, p=p, use_numba=use_numba, seed=0)

    # --- assert ------------------------------------------
    np.testing.assert_array_equal(samples_1, samples_2)
    if k >= 10:
        # in this case, it's very unlikely that samples_3 matches samples_1 or samples_2
        assert not list(samples_1) == list(samples_3)
        assert not list(samples_2) == list(samples_3)
        assert not list(samples_3) == list(samples_4)
    if p is not None:
        assert np.array_equal(p, p_before), "p array should never be modified."


# =================================================================================================
#  TEST - UNIFORM - WITHOUT REPLACEMENT
# =================================================================================================
@pytest.mark.parametrize("n", [10, 100, 1000, 10000])
@pytest.mark.parametrize("use_numba", [True, False])
@pytest.mark.parametrize("p", [None, np.zeros(0, np.float64)])
def test_randint_uniform_without_replacement_invariants_single(n: int, use_numba: bool, p: np.ndarray | None):
    # --- arrange -----------------------------------------
    if p is not None:
        p_before = p.copy()  # copy for later comparison

    # --- act ---------------------------------------------
    sample = randint(n, replace=False, p=p, use_numba=use_numba)

    # --- assert ------------------------------------------
    assert isinstance(sample, numbers.Integral)
    assert 0 <= sample < n
    if p is not None:
        assert np.array_equal(p, p_before), "p array should never be modified."


@pytest.mark.parametrize(
    "n,k",
    [
        (10, 1),
        (100, 1),
        (1000, 1),
        (1000, 10),
        (1000, 100),
        (1000, 500),
        (1000, 1000),
    ],
)
@pytest.mark.parametrize("use_numba", [True, False])
@pytest.mark.parametrize("p", [None, np.zeros(0, np.float64)])
def test_randint_uniform_without_replacement_invariants_multiple(n: int, k: int, use_numba: bool, p: np.ndarray | None):
    # --- arrange -----------------------------------------
    if p is not None:
        p_before = p.copy()  # copy for later comparison

    # --- act ---------------------------------------------
    samples = randint(n, k, replace=False, p=p, use_numba=use_numba)

    # --- assert ------------------------------------------
    assert isinstance(samples, np.ndarray)
    assert samples.shape == (k,)
    assert samples.dtype == np.int32
    assert 0 <= samples.min() <= samples.max() < n
    assert len(set(samples)) == k  # all samples are unique
    if p is not None:
        assert np.array_equal(p, p_before), "p array should never be modified."


@pytest.mark.parametrize(
    "n,k",
    [
        (10, 1),
        (100, 1),
        (1000, 1),
        (1000, 10),
        (1000, 100),
        (1000, 500),
        (1000, 1000),
    ],
)
@pytest.mark.parametrize("use_numba", [True, False])
@pytest.mark.parametrize("p", [None, np.zeros(0, np.float64)])
def test_randint_uniform_without_replacement_seed(n: int, k: int, use_numba: bool, p: np.ndarray | None):
    # --- arrange -----------------------------------------
    if p is not None:
        p_before = p.copy()  # copy for later comparison

    # --- act ---------------------------------------------
    samples_1 = randint(n, k, replace=False, p=p, use_numba=use_numba, seed=123)
    samples_2 = randint(n, k, replace=False, p=p, use_numba=use_numba, seed=123)
    samples_3 = randint(n, k, replace=False, p=p, use_numba=use_numba, seed=None)
    samples_4 = randint(n, k, replace=False, p=p, use_numba=use_numba, seed=0)

    # --- assert ------------------------------------------
    np.testing.assert_array_equal(samples_1, samples_2)
    if k >= 10:
        # in this case, it's very unlikely that samples_3 matches samples_1 or samples_2
        assert not list(samples_1) == list(samples_3)
        assert not list(samples_2) == list(samples_3)
        assert not list(samples_3) == list(samples_4)
    if p is not None:
        assert np.array_equal(p, p_before), "p array should never be modified."


# =================================================================================================
#  TEST - NON-UNIFORM - WITH REPLACEMENT
# =================================================================================================
@pytest.mark.parametrize("n", [10, 100, 1000, 10000])
@pytest.mark.parametrize("use_numba", [True, False])
def test_randint_non_uniform_with_replacement_invariants_single(n: int, use_numba: bool):
    # --- arrange -----------------------------------------
    p = get_probabilities(n)
    p_before = p.copy()  # copy for later comparison

    # --- act ---------------------------------------------
    sample = randint(n, replace=True, p=p, use_numba=use_numba)

    # --- assert ------------------------------------------
    assert isinstance(sample, numbers.Integral)
    assert 0 <= sample < n
    assert np.array_equal(p, p_before), "p array should never be modified."


@pytest.mark.parametrize(
    "n,k",
    [
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
@pytest.mark.parametrize("use_numba", [True, False])
def test_randint_non_uniform_with_replacement_invariants_multiple(n: int, k: int, use_numba: bool):
    # --- arrange -----------------------------------------
    p = get_probabilities(n)
    p_before = p.copy()  # copy for later comparison

    # --- act ---------------------------------------------
    samples = randint(n, k, replace=True, p=p, use_numba=use_numba)

    # --- assert ------------------------------------------
    assert isinstance(samples, np.ndarray)
    assert samples.shape == (k,)
    assert samples.dtype == np.int32
    assert 0 <= samples.min() <= samples.max() < n
    assert np.array_equal(p, p_before), "p array should never be modified."


@pytest.mark.parametrize("use_numba", [True, False])
def test_randint_non_uniform_with_replacement_duplicates(use_numba: bool):
    # --- arrange -----------------------------------------
    p = get_probabilities(1000)
    p_before = p.copy()  # copy for later comparison

    # --- act ---------------------------------------------
    samples = randint(n=1000, k=900, replace=True, p=p, use_numba=use_numba)  # very unlikely all are unique

    # --- assert ------------------------------------------
    assert len(set(samples)) < 900  # there are duplicates, which is expected with replacement
    assert np.array_equal(p, p_before), "p array should never be modified."


@pytest.mark.parametrize(
    "n,k",
    [
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
@pytest.mark.parametrize("use_numba", [True, False])
def test_randint_non_uniform_with_replacement_seed(n: int, k: int, use_numba: bool):
    # --- arrange -----------------------------------------
    p = get_probabilities(n)
    p_before = p.copy()  # copy for later comparison

    # --- act ---------------------------------------------
    samples_1 = randint(n, k, replace=True, p=p, use_numba=use_numba, seed=123)
    samples_2 = randint(n, k, replace=True, p=p, use_numba=use_numba, seed=123)
    samples_3 = randint(n, k, replace=True, p=p, use_numba=use_numba, seed=None)
    samples_4 = randint(n, k, replace=True, p=p, use_numba=use_numba, seed=0)

    # --- assert ------------------------------------------
    np.testing.assert_array_equal(samples_1, samples_2)
    if k >= 10:
        # in this case, it's very unlikely that samples_3 matches samples_1 or samples_2
        assert not list(samples_1) == list(samples_3)
        assert not list(samples_2) == list(samples_3)
        assert not list(samples_3) == list(samples_4)
    assert np.array_equal(p, p_before), "p array should never be modified."


@pytest.mark.parametrize("use_numba", [True, False])
@pytest.mark.parametrize("factor", [2.0, 5.0, 10.0, 100.0, 1000.0, 0.0])
@pytest.mark.parametrize("sum_of_p", [1.0, 0.1, 10.0])
def test_randint_non_uniform_with_replacement_probs(use_numba: bool, factor: float, sum_of_p: float):
    # --- arrange -----------------------------------------
    n = 10000
    k = 1000
    p = get_probabilities(n)
    p[int(0.9 * n) :] = factor * p[int(0.9 * n) :]  # multiply last 10% of probs with factor
    p = p * (sum_of_p / p.sum())  # ensure sum(p) = sum_of_p, so we also test non-normalized probs

    expected_mean = sum(i * p[i] for i in range(n)) / sum_of_p

    p_before = p.copy()  # copy for later comparison

    # --- act ---------------------------------------------
    samples = randint(n=n, k=k, replace=True, p=p, use_numba=use_numba)

    # --- assert ------------------------------------------
    assert 0.9 * expected_mean < np.mean(samples) < 1.1 * expected_mean
    assert all([p[i] > 0 for i in samples])
    assert np.array_equal(p, p_before), "p array should never be modified."


# =================================================================================================
#  TEST - NON-UNIFORM - WITHOUT REPLACEMENT
# =================================================================================================
@pytest.mark.parametrize("n", [10, 100, 1000, 10000])
@pytest.mark.parametrize("use_numba", [True, False])
def test_randint_non_uniform_without_replacement_invariants_single(n: int, use_numba: bool):
    # --- arrange -----------------------------------------
    p = get_probabilities(n)
    p_before = p.copy()  # copy for later comparison

    # --- act ---------------------------------------------
    sample = randint(n, replace=False, p=p, use_numba=use_numba)

    # --- assert ------------------------------------------
    assert isinstance(sample, numbers.Integral)
    assert 0 <= sample < n
    assert np.array_equal(p, p_before), "p array should never be modified."


@pytest.mark.parametrize(
    "n,k",
    [
        (10, 1),
        (100, 1),
        (1000, 1),
        (1000, 10),
        (1000, 100),
        (1000, 500),
        (1000, 1000),
    ],
)
@pytest.mark.parametrize("use_numba", [True, False])
def test_randint_non_uniform_without_replacement_invariants_multiple(n: int, k: int, use_numba: bool):
    # --- arrange -----------------------------------------
    p = get_probabilities(n)
    p_before = p.copy()  # copy for later comparison

    # --- act ---------------------------------------------
    samples = randint(n, k, replace=False, p=p, use_numba=use_numba)

    # --- assert ------------------------------------------
    assert isinstance(samples, np.ndarray)
    assert samples.shape == (k,)
    assert samples.dtype == np.int32
    assert 0 <= samples.min() <= samples.max() < n
    assert len(set(samples)) == k  # all samples are unique
    assert np.array_equal(p, p_before), "p array should never be modified."


@pytest.mark.parametrize(
    "n,k",
    [
        (10, 1),
        (100, 1),
        (1000, 1),
        (1000, 10),
        (1000, 100),
        (1000, 500),
        (1000, 1000),
    ],
)
@pytest.mark.parametrize("use_numba", [True, False])
def test_randint_non_uniform_without_replacement_seed(n: int, k: int, use_numba: bool):
    # --- arrange -----------------------------------------
    p = get_probabilities(n)
    p_before = p.copy()  # copy for later comparison

    # --- act ---------------------------------------------
    samples_1 = randint(n, k, replace=False, p=p, use_numba=use_numba, seed=123)
    samples_2 = randint(n, k, replace=False, p=p, use_numba=use_numba, seed=123)
    samples_3 = randint(n, k, replace=False, p=p, use_numba=use_numba, seed=None)

    # --- assert ------------------------------------------
    np.testing.assert_array_equal(samples_1, samples_2)
    if k >= 10:
        # in this case, it's very unlikely that samples_3 matches samples_1 or samples_2
        assert not list(samples_1) == list(samples_3)
        assert not list(samples_2) == list(samples_3)
    assert np.array_equal(p, p_before), "p array should never be modified."


@pytest.mark.parametrize("use_numba", [True, False])
@pytest.mark.parametrize("factor", [2.0, 5.0, 10.0, 100.0, 1000.0, 0.0])
@pytest.mark.parametrize("sum_of_p", [1.0, 0.1, 10.0])
def test_randint_non_uniform_without_replacement_probs(use_numba: bool, factor: float, sum_of_p: float):
    # --- arrange -----------------------------------------
    n = 10000
    k = 1000
    p = get_probabilities(n)
    p[int(0.8 * n) :] = factor * p[int(0.8 * n) :]  # multiply last 20% of probs with factor
    p = p * (sum_of_p / p.sum())  # ensure sum(p) = sum_of_p, so we also test non-normalized probs

    expected_mean = sum(i * p[i] for i in range(n)) / sum_of_p  # approx. correct; this assumes replacement

    p_before = p.copy()  # copy for later comparison

    # --- act ---------------------------------------------
    samples = randint(n=n, k=k, replace=False, p=p, use_numba=use_numba)

    # --- assert ------------------------------------------
    assert 0.9 * expected_mean < np.mean(samples) < 1.1 * expected_mean
    assert all([p[i] > 0 for i in samples])
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
# def test_randint_numba_zero_probability_selection(seed_offset: np.int64, replace: bool):
#     """Test that `randint_numba` never selects indices with zero probability, by triggering it 1_000_000_000 times."""
#     # --- arrange -----------------------------------------
#     n = np.int32(4)
#     k = np.int32(2)
#     p = np.array([0.0, 0.5, 0.0, 0.5], dtype=np.float32)
#     i_selected: set[np.int32] = set()
#
#     # --- act ---------------------------------------------
#     for seed in np.arange(seed_offset, seed_offset + 1_000_000, dtype=np.int64):
#         samples = randint_numba(n=n, k=k, replace=replace, p=p, seed=seed)
#         for s in samples:
#             i_selected.add(s)
#
#     # --- assert ------------------------------------------
#     for i in i_selected:
#         assert p[i] > 0.0, f"Selected index {i} with zero probability!"
