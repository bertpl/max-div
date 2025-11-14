import numbers

import numpy as np
import pytest

from max_div.sampling.discrete import sample_int


# =================================================================================================
#  Helpers
# =================================================================================================
def get_probabilities(n: int) -> np.ndarray[float]:
    probs = np.random.random(n)
    probs /= probs.sum()
    return probs


# =================================================================================================
#  TEST - ARGUMENT VALIDATION
# =================================================================================================
@pytest.mark.parametrize("use_numba", [True, False])
def test_sample_int_argument_validation(use_numba: bool):
    # --- n < 1 ------------------------------------------
    with pytest.raises(ValueError):
        sample_int(n=0, use_numba=use_numba)

    with pytest.raises(ValueError):
        sample_int(n=-10, use_numba=use_numba)

    # --- k < 1 ------------------------------------------
    with pytest.raises(ValueError):
        sample_int(n=10, k=0, use_numba=use_numba)

    with pytest.raises(ValueError):
        sample_int(n=10, k=-5, use_numba=use_numba)

    # --- k > n when replace=False -----------------------
    with pytest.raises(ValueError):
        sample_int(n=10, k=20, replace=False, use_numba=use_numba)

    with pytest.raises(ValueError):
        sample_int(n=100, k=101, replace=False, use_numba=use_numba)

    # --- p invalid shape --------------------------------
    with pytest.raises(ValueError):
        sample_int(n=10, k=5, replace=True, p=np.array([0.1, 0.9]), use_numba=use_numba)  # wrong size

    with pytest.raises(ValueError):
        sample_int(n=4, k=2, replace=True, p=np.array([[0.1, 0.4], [0.4, 0.1]]), use_numba=use_numba)  # wrong shape


# =================================================================================================
#  TEST - UNIFORM - WITH REPLACEMENT
# =================================================================================================
@pytest.mark.parametrize("n", [10, 100, 1000, 10000])
@pytest.mark.parametrize("use_numba", [True, False])
@pytest.mark.parametrize("p", [None, np.zeros(0, np.float64)])
def test_sample_int_uniform_with_replacement_invariants_single(n: int, use_numba: bool, p: np.ndarray | None):
    # --- act ---------------------------------------------
    sample = sample_int(n, replace=True, p=p, use_numba=use_numba)

    # --- assert ------------------------------------------
    assert isinstance(sample, numbers.Integral)
    assert 0 <= sample < n


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
def test_sample_int_uniform_with_replacement_invariants_multiple(n: int, k: int, use_numba: bool, p: np.ndarray | None):
    # --- act ---------------------------------------------
    samples = sample_int(n, k, replace=True, p=p, use_numba=use_numba)

    # --- assert ------------------------------------------
    assert isinstance(samples, np.ndarray)
    assert samples.shape == (k,)
    assert samples.dtype == np.int64
    assert 0 <= samples.min() <= samples.max() < n


@pytest.mark.parametrize("use_numba", [True, False])
@pytest.mark.parametrize("p", [None, np.zeros(0, np.float64)])
def test_sample_int_uniform_with_replacement_duplicates(use_numba: bool, p: np.ndarray | None):
    # --- act ---------------------------------------------
    samples = sample_int(n=1000, k=900, replace=True, p=p, use_numba=use_numba)  # very unlikely all are unique

    # --- assert ------------------------------------------
    assert len(set(samples)) < 900  # there are duplicates, which is expected with replacement


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
def test_sample_int_uniform_with_replacement_seed(n: int, k: int, use_numba: bool, p: np.ndarray | None):
    # --- act ---------------------------------------------
    samples_1 = sample_int(n, k, replace=True, p=p, use_numba=use_numba, seed=123)
    samples_2 = sample_int(n, k, replace=True, p=p, use_numba=use_numba, seed=123)
    samples_3 = sample_int(n, k, replace=True, p=p, use_numba=use_numba, seed=None)
    samples_4 = sample_int(n, k, replace=True, p=p, use_numba=use_numba, seed=0)

    # --- assert ------------------------------------------
    np.testing.assert_array_equal(samples_1, samples_2)
    if k >= 10:
        # in this case, it's very unlikely that samples_3 matches samples_1 or samples_2
        assert not list(samples_1) == list(samples_3)
        assert not list(samples_2) == list(samples_3)
        assert not list(samples_3) == list(samples_4)


# =================================================================================================
#  TEST - UNIFORM - WITHOUT REPLACEMENT
# =================================================================================================
@pytest.mark.parametrize("n", [10, 100, 1000, 10000])
@pytest.mark.parametrize("use_numba", [True, False])
@pytest.mark.parametrize("p", [None, np.zeros(0, np.float64)])
def test_sample_int_uniform_without_replacement_invariants_single(n: int, use_numba: bool, p: np.ndarray | None):
    # --- act ---------------------------------------------
    sample = sample_int(n, replace=False, p=p, use_numba=use_numba)

    # --- assert ------------------------------------------
    assert isinstance(sample, numbers.Integral)
    assert 0 <= sample < n


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
def test_sample_int_uniform_without_replacement_invariants_multiple(
    n: int, k: int, use_numba: bool, p: np.ndarray | None
):
    # --- act ---------------------------------------------
    samples = sample_int(n, k, replace=False, p=p, use_numba=use_numba)

    # --- assert ------------------------------------------
    assert isinstance(samples, np.ndarray)
    assert samples.shape == (k,)
    assert samples.dtype == np.int64
    assert 0 <= samples.min() <= samples.max() < n
    assert len(set(samples)) == k  # all samples are unique


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
def test_sample_int_uniform_without_replacement_seed(n: int, k: int, use_numba: bool, p: np.ndarray | None):
    # --- act ---------------------------------------------
    samples_1 = sample_int(n, k, replace=False, p=p, use_numba=use_numba, seed=123)
    samples_2 = sample_int(n, k, replace=False, p=p, use_numba=use_numba, seed=123)
    samples_3 = sample_int(n, k, replace=False, p=p, use_numba=use_numba, seed=None)
    samples_4 = sample_int(n, k, replace=False, p=p, use_numba=use_numba, seed=0)

    # --- assert ------------------------------------------
    np.testing.assert_array_equal(samples_1, samples_2)
    if k >= 10:
        # in this case, it's very unlikely that samples_3 matches samples_1 or samples_2
        assert not list(samples_1) == list(samples_3)
        assert not list(samples_2) == list(samples_3)
        assert not list(samples_3) == list(samples_4)


# =================================================================================================
#  TEST - NON-UNIFORM - WITH REPLACEMENT
# =================================================================================================
@pytest.mark.parametrize("n", [10, 100, 1000, 10000])
@pytest.mark.parametrize("use_numba", [True, False])
def test_sample_int_non_uniform_with_replacement_invariants_single(n: int, use_numba: bool):
    # --- arrange -----------------------------------------
    p = get_probabilities(n)

    # --- act ---------------------------------------------
    sample = sample_int(n, replace=True, p=p, use_numba=use_numba)

    # --- assert ------------------------------------------
    assert isinstance(sample, numbers.Integral)
    assert 0 <= sample < n


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
def test_sample_int_non_uniform_with_replacement_invariants_multiple(n: int, k: int, use_numba: bool):
    # --- arrange -----------------------------------------
    p = get_probabilities(n)

    # --- act ---------------------------------------------
    samples = sample_int(n, k, replace=True, p=p, use_numba=use_numba)

    # --- assert ------------------------------------------
    assert isinstance(samples, np.ndarray)
    assert samples.shape == (k,)
    assert samples.dtype == np.int64
    assert 0 <= samples.min() <= samples.max() < n


@pytest.mark.parametrize("use_numba", [True, False])
def test_sample_int_non_uniform_with_replacement_duplicates(use_numba: bool):
    # --- arrange -----------------------------------------
    p = get_probabilities(1000)

    # --- act ---------------------------------------------
    samples = sample_int(n=1000, k=900, replace=True, p=p, use_numba=use_numba)  # very unlikely all are unique

    # --- assert ------------------------------------------
    assert len(set(samples)) < 900  # there are duplicates, which is expected with replacement


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
def test_sample_int_non_uniform_with_replacement_seed(n: int, k: int, use_numba: bool):
    # --- arrange -----------------------------------------
    p = get_probabilities(n)

    # --- act ---------------------------------------------
    samples_1 = sample_int(n, k, replace=True, p=p, use_numba=use_numba, seed=123)
    samples_2 = sample_int(n, k, replace=True, p=p, use_numba=use_numba, seed=123)
    samples_3 = sample_int(n, k, replace=True, p=p, use_numba=use_numba, seed=None)
    samples_4 = sample_int(n, k, replace=True, p=p, use_numba=use_numba, seed=0)

    # --- assert ------------------------------------------
    np.testing.assert_array_equal(samples_1, samples_2)
    if k >= 10:
        # in this case, it's very unlikely that samples_3 matches samples_1 or samples_2
        assert not list(samples_1) == list(samples_3)
        assert not list(samples_2) == list(samples_3)
        assert not list(samples_3) == list(samples_4)


@pytest.mark.parametrize("use_numba", [True, False])
@pytest.mark.parametrize("factor", [2.0, 5.0, 10.0, 100.0, 1000.0, 0.0])
def test_sample_int_non_uniform_with_replacement_probs(use_numba: bool, factor: float):
    # --- arrange -----------------------------------------
    n = 10000
    k = 1000
    p = get_probabilities(n)
    p[int(0.9 * n) :] = factor * p[int(0.9 * n) :]  # multiply last 10% of probs with factor
    p = p / p.sum()  # re-normalize

    expected_mean = sum(i * p[i] for i in range(n))

    # --- act ---------------------------------------------
    samples = sample_int(n=n, k=k, replace=True, p=p, use_numba=use_numba)

    # --- assert ------------------------------------------
    assert 0.95 * expected_mean < np.mean(samples) < 1.05 * expected_mean
    assert all([p[i] > 0 for i in samples])


# =================================================================================================
#  TEST - NON-UNIFORM - WITHOUT REPLACEMENT
# =================================================================================================
@pytest.mark.parametrize("n", [10, 100, 1000, 10000])
@pytest.mark.parametrize("use_numba", [True, False])
def test_sample_int_non_uniform_without_replacement_invariants_single(n: int, use_numba: bool):
    # --- arrange -----------------------------------------
    p = get_probabilities(n)

    # --- act ---------------------------------------------
    sample = sample_int(n, replace=False, p=p, use_numba=use_numba)

    # --- assert ------------------------------------------
    assert isinstance(sample, numbers.Integral)
    assert 0 <= sample < n


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
def test_sample_int_non_uniform_without_replacement_invariants_multiple(n: int, k: int, use_numba: bool):
    # --- arrange -----------------------------------------
    p = get_probabilities(n)

    # --- act ---------------------------------------------
    samples = sample_int(n, k, replace=False, p=p, use_numba=use_numba)

    # --- assert ------------------------------------------
    assert isinstance(samples, np.ndarray)
    assert samples.shape == (k,)
    assert samples.dtype == np.int64
    assert 0 <= samples.min() <= samples.max() < n
    assert len(set(samples)) == k  # all samples are unique


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
def test_sample_int_non_uniform_without_replacement_seed(n: int, k: int, use_numba: bool):
    # --- arrange -----------------------------------------
    p = get_probabilities(n)

    # --- act ---------------------------------------------
    samples_1 = sample_int(n, k, replace=False, p=p, use_numba=use_numba, seed=123)
    samples_2 = sample_int(n, k, replace=False, p=p, use_numba=use_numba, seed=123)
    samples_3 = sample_int(n, k, replace=False, p=p, use_numba=use_numba, seed=None)

    # --- assert ------------------------------------------
    np.testing.assert_array_equal(samples_1, samples_2)
    if k >= 10:
        # in this case, it's very unlikely that samples_3 matches samples_1 or samples_2
        assert not list(samples_1) == list(samples_3)
        assert not list(samples_2) == list(samples_3)


@pytest.mark.parametrize("use_numba", [True, False])
@pytest.mark.parametrize("factor", [2.0, 5.0, 10.0, 100.0, 1000.0, 0.0])
def test_sample_int_non_uniform_without_replacement_probs(use_numba: bool, factor: float):
    # --- arrange -----------------------------------------
    n = 10000
    k = 1000
    p = get_probabilities(n)
    p[int(0.8 * n) :] = factor * p[int(0.8 * n) :]  # multiply last 20% of probs with factor
    p = p / p.sum()  # re-normalize

    expected_mean = sum(i * p[i] for i in range(n))  # approx. correct; this assumes replacement

    # --- act ---------------------------------------------
    samples = sample_int(n=n, k=k, replace=False, p=p, use_numba=use_numba)

    # --- assert ------------------------------------------
    assert 0.95 * expected_mean < np.mean(samples) < 1.05 * expected_mean
    assert all([p[i] > 0 for i in samples])
