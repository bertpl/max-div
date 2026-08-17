import numbers

import numpy as np
import pytest
from numpy.typing import NDArray

from max_div._core._random import P_UNIFORM, choice, choice1, randint
from max_div._core._random._rng import new_rng_state


# =================================================================================================
#  Helpers
# =================================================================================================
def get_probabilities(n: int) -> NDArray[np.float32]:
    probs = np.random.random(n)
    probs /= probs.sum()
    return probs.astype(np.float32)


# =================================================================================================
#  Tests
# =================================================================================================
@pytest.mark.parametrize("n_values", [20, 100, 1000])
@pytest.mark.parametrize("k", [1, 5, 10, 20])
@pytest.mark.parametrize("replace", [True, False])
@pytest.mark.parametrize("uniform", [True, False])
def test_choice(n_values: int, k: int, replace: bool, uniform: bool) -> None:
    # --- arrange ----------------------
    values = randint(10000, np.int32(n_values), replace=False, p=P_UNIFORM, rng_state=new_rng_state(42))
    values_set = set(values)
    p = P_UNIFORM if uniform else get_probabilities(values.size)
    p_copy = p.copy()
    rng_state = new_rng_state(123456)

    # --- act --------------------------
    results = choice(
        values=values,
        k=np.int32(k),
        replace=replace,
        p=p,
        rng_state=rng_state,
    )

    # --- assert -----------------------
    assert isinstance(results, np.ndarray)
    assert results.dtype == np.int32
    assert results.shape == (k,)
    assert all(result in values_set for result in results)
    assert np.array_equal(p, p_copy), "p should remain unmodified"
    assert not np.array_equal(rng_state, new_rng_state(123456)), "rng_state should be modified in-place"


@pytest.mark.parametrize("n_values", [20, 100, 1000])
@pytest.mark.parametrize("k", [1, 5, 10, 20])
@pytest.mark.parametrize("uniform", [True, False])
def test_choice1(n_values: int, k: int, uniform: bool) -> None:
    # --- arrange ----------------------
    values = randint(10000, np.int32(n_values), replace=False, p=P_UNIFORM, rng_state=new_rng_state(42))
    values_set = set(values)
    p = P_UNIFORM if uniform else get_probabilities(values.size)
    p_copy = p.copy()
    rng_state = new_rng_state(123456)

    # --- act --------------------------
    results = [
        choice1(
            values=values,
            p=p,
            rng_state=rng_state,
        )
        for _ in range(k)
    ]

    # --- assert -----------------------
    assert all(isinstance(result, numbers.Integral) for result in results)
    assert all(result in values_set for result in results)
    assert np.array_equal(p, p_copy), "p should remain unmodified"
    assert not np.array_equal(rng_state, new_rng_state(123456)), "rng_state should be modified in-place"
