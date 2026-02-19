import math

import numpy as np
import pytest

from max_div._core._random import new_rng_state
from max_div._core._random._distributions import sample_truncated_poisson, truncated_poisson_expected_value


@pytest.mark.parametrize("min_value, max_value", [(1, 5), (2, 2), (2, 4), (3, 7), (10, 20), (15, 20)])
def test_sample_truncated_poisson_boundaries(min_value: int, max_value: int):
    # --- arrange -----------------------------------------
    n_samples = 100
    lambda_values = np.linspace(1.0, max_value + 1.0, num=n_samples)

    # --- act ---------------------------------------------
    rng_state = new_rng_state(np.int64(42))
    samples = [
        sample_truncated_poisson(
            min_value=np.int32(min_value),
            max_value=np.int32(max_value),
            _lambda=np.float32(lambda_values[i]),
            rng_state=rng_state,
        )
        for i in range(n_samples)
    ]

    # --- assert ------------------------------------------
    assert min(samples) == min_value
    assert max(samples) == max_value
    assert len(set(samples)) == max_value - min_value + 1


@pytest.mark.parametrize(
    "min_value, max_value, _lambda",
    [
        (1, 5, 0.5),
        (2, 2, 1.0),
        (2, 4, 2.7),
        (3, 7, 7.5),
        (15, 20, 20.0),
        (10, 20, 15.0),
    ],
)
def test_sample_truncated_poisson_distribution(min_value: int, max_value: int, _lambda: float):
    # --- arrange -----------------------------------------
    n_samples = 10_000
    hist = np.zeros(max_value + 1, dtype=np.float32)

    hist_expected = np.zeros(max_value + 1, dtype=np.float32)
    for i in range(min_value, max_value + 1):
        hist_expected[i] = np.float32((_lambda**i) / math.factorial(i))
    hist_expected /= np.sum(hist_expected)  # normalize

    # --- act ---------------------------------------------
    rng_state = new_rng_state(np.int64(42))
    for i in range(n_samples):
        sample = sample_truncated_poisson(
            min_value=np.int32(min_value),
            max_value=np.int32(max_value),
            _lambda=np.float32(_lambda),
            rng_state=rng_state,
        )
        hist[sample] += 1.0

    hist /= n_samples  # normalize

    # --- assert ------------------------------------------
    assert np.allclose(hist, hist_expected, atol=0.05)


def test_truncated_poisson_expected_value():
    # --- act ---------------------------------------------
    result = truncated_poisson_expected_value(
        min_value=np.int32(1),
        max_value=np.int32(5),
        _lambda=np.float32(2.0),
    )

    # --- assert ------------------------------------------
    assert result == pytest.approx(2.2340424060821533)
