import numbers

import numpy as np
import pytest
from numba import TypingError
from numpy.typing import NDArray

from max_div._core._random import P_UNIFORM, randint1
from max_div._core._random._rng import new_rng_state


# =================================================================================================
#  Helpers
# =================================================================================================
def get_probabilities(n: int) -> NDArray[np.float32]:
    probs = np.random.random(n)
    probs /= probs.sum()
    return probs.astype(np.float32)


# =================================================================================================
#  UNIT TESTS
# =================================================================================================
def test_randint1_argument_validation():
    rng_state = new_rng_state(42)
    p = P_UNIFORM

    # --- n < 1 ------------------------------------------
    with pytest.raises(ValueError):
        randint1(n=np.int32(0), p=p, rng_state=rng_state)

    with pytest.raises(ValueError):
        randint1(n=np.int32(-10), p=p, rng_state=rng_state)

    # --- sum(p)<=0.0 ------------------------------------
    with pytest.raises(ValueError):
        randint1(n=np.int32(5), p=-np.zeros(5).astype(np.float32), rng_state=rng_state)

    with pytest.raises(ValueError):
        randint1(n=np.int32(5), p=-np.ones(5).astype(np.float32), rng_state=rng_state)

    # --- p invalid shape --------------------------------
    with pytest.raises(ValueError):
        randint1(n=np.int32(10), p=np.array([0.1, 0.9], dtype=np.float32), rng_state=rng_state)  # wrong size

    with pytest.raises((TypeError, TypingError, ValueError)):
        randint1(
            n=np.int32(5),
            p=np.array([[0.1, 0.4], [0.4, 0.1]], dtype=np.float32),
            rng_state=rng_state,
        )  # wrong shape


@pytest.mark.parametrize("n", [1, 10, 100, 1000, 10000])
@pytest.mark.parametrize("uniform", [False, True])
def test_randint1_invariants(n: int, uniform: bool):
    # --- arrange -----------------------------------------
    p = P_UNIFORM if uniform else get_probabilities(n)
    rng_state = new_rng_state(42)
    n_repeats = 1000

    # --- act & assert ------------------------------------
    all_values = set()
    for _ in range(n_repeats):
        # --- arrange -----------------------------------------
        p_before = p.copy()
        rng_state_before = rng_state.copy()

        # --- act ---------------------------------------------
        sample = randint1(n=np.int32(n), p=p, rng_state=rng_state)
        all_values.add(sample)

        # --- assert ------------------------------------------
        assert isinstance(
            sample, numbers.Integral
        )  # numba converts int32 -> int when called from plain Python function
        assert 0 <= sample < n
        assert np.array_equal(p, p_before), "p array should never be modified."
        assert not np.array_equal(rng_state, rng_state_before), "rng_state should be modified."

    # --- assert ------------------------------------------
    assert len(all_values) >= 0.5 * min(n, n_repeats)  # at least half of possible values were sampled


@pytest.mark.parametrize(
    "n,k",
    [
        (10, 1),
        (100, 1),
        (1000, 1),
        (1000, 10),
        (1000, 100),
        (1000, 1000),
    ],
)
@pytest.mark.parametrize("uniform", [False, True])
def test_randint1_uniform_rng_state(n: int, k: int, uniform: bool):
    # --- arrange -----------------------------------------
    p = P_UNIFORM if uniform else get_probabilities(n)
    p_before = p.copy()  # copy for later comparison
    rng_state_1 = new_rng_state(42)
    rng_state_2 = new_rng_state(42)
    rng_state_3 = new_rng_state(123456)

    # --- act ---------------------------------------------
    samples_1 = [randint1(np.int32(n), p=p, rng_state=rng_state_1) for _ in range(k)]
    samples_2 = [randint1(np.int32(n), p=p, rng_state=rng_state_2) for _ in range(k)]
    samples_3 = [randint1(np.int32(n), p=p, rng_state=rng_state_3) for _ in range(k)]

    # --- assert ------------------------------------------

    # check rng_state modifications
    assert not np.array_equal(rng_state_1, new_rng_state(42))
    assert not np.array_equal(rng_state_2, new_rng_state(42))
    assert not np.array_equal(rng_state_3, new_rng_state(123456))
    assert np.array_equal(rng_state_1, rng_state_2)
    assert not np.array_equal(rng_state_1, rng_state_3)

    # check results
    np.testing.assert_array_equal(samples_1, samples_2)
    if k >= 10:
        # in this case, it's very unlikely that samples_3 matches samples_1 or samples_2
        assert not list(samples_1) == list(samples_3)
        assert not list(samples_2) == list(samples_3)

    # check p unmodified
    assert np.array_equal(p, p_before), "p array should never be modified."


# =================================================================================================
#  TEST - NON-UNIFORM
# =================================================================================================
@pytest.mark.parametrize("factor", [2.0, 5.0, 10.0, 100.0, 1000.0, 0.0])
@pytest.mark.parametrize("sum_of_p", [1.0, 0.1, 10.0])
def test_randint1_non_uniform_probs(factor: float, sum_of_p: float):
    # --- arrange -----------------------------------------
    n = 10000
    k = 1000
    p = get_probabilities(n)
    p[int(0.9 * n) :] = factor * p[int(0.9 * n) :]  # multiply last 10% of probs with factor
    p = p * (sum_of_p / p.sum())  # ensure sum(p) = sum_of_p, so we also test non-normalized probs

    expected_mean = sum(i * p[i] for i in range(n)) / sum_of_p

    p_before = p.copy()  # copy for later comparison
    rng_state = new_rng_state(42)

    # --- act ---------------------------------------------
    samples = [randint1(n=np.int32(n), p=p, rng_state=rng_state) for _ in range(k)]

    # --- assert ------------------------------------------
    assert 0.9 * expected_mean < np.mean(samples) < 1.1 * expected_mean
    assert all([p[i] > 0 for i in samples])
    assert np.array_equal(p, p_before), "p array should never be modified."
