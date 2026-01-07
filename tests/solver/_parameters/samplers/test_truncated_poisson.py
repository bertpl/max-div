import math

import numpy as np
import pytest

from max_div.solver._parameters import AdaptiveSampler
from max_div.solver._parameters.samplers import TruncatedPoissonAdaptiveSampler, sampled_poisson


def test_truncated_poisson_adaptive_sampler_construction():
    # --- arrange -----------------------------------------
    sampler = TruncatedPoissonAdaptiveSampler(
        min_value=1,
        max_value=8,
        lambda_prior=2.0,
        tau_learn=10.0,
        tau_forget=100.0,
        seed=123,
    )

    # --- act & assert ------------------------------------
    assert isinstance(sampler, AdaptiveSampler)
    assert isinstance(sampler, TruncatedPoissonAdaptiveSampler)

    assert all([isinstance(sampler.new_sample(), int) for _ in range(1000)])  # all should be of type 'int'
    assert all([1 <= sampler.new_sample() <= 8 for _ in range(1000)])  # all should be in [1, 8]

    # expected value checks
    assert sampler.summary_statistic() == pytest.approx(2.3111331462)
    assert 2.1 < np.mean([sampler.new_sample() for _ in range(1000)]) < 2.5


def test_truncated_poisson_adaptive_sampler_construction_validation():
    with pytest.raises(ValueError):
        _ = TruncatedPoissonAdaptiveSampler(
            min_value=1,
            max_value=8,
            lambda_prior=8.1,
            tau_learn=10.0,
            tau_forget=100.0,
            seed=123,
        )


@pytest.mark.parametrize("good_values", [[1, 2, 3], [8, 9, 10]])
@pytest.mark.parametrize("tau_forget", [100.0, math.inf])
def test_truncated_poisson_adaptive_sampler_learn_and_forget(good_values: list[int], tau_forget):
    # --- arrange -----------------------------------------
    sampler = TruncatedPoissonAdaptiveSampler(
        min_value=1,
        max_value=10,
        lambda_prior=5.5,
        tau_learn=10.0,
        tau_forget=tau_forget,
        seed=123,
    )

    # --- act 1 - learn -----------------------------------
    for _ in range(1_000):
        s = sampler.new_sample()
        sampler.feedback(s in good_values)  # positive feedback when we sampled one of 'good_values'

    # --- assert 1 - learn --------------------------------
    assert min(good_values) < sampler._lambda < max(good_values)  # should learn to sample good range

    # --- act 2 - forget ----------------------------------
    summary_stat_before = sampler.summary_statistic()
    for _ in range(1_000):
        s = sampler.new_sample()
        sampler.feedback(False)  # negative feedback always, so we maximally forget

    # --- assert 2 - forget -------------------------------
    if not np.isinf(tau_forget):
        # forgetting is enabled
        assert 5.0 < sampler._lambda < 6.0  # should have forgotten back to prior (5.5)
    else:
        # forgetting is disabled
        assert sampler.summary_statistic() == summary_stat_before


@pytest.mark.parametrize("lambda_prior", [3.7, None])
def test_sampled_poisson_alias(lambda_prior: float | None):
    # --- act ---------------------------------------------
    sampler = sampled_poisson(
        min_value=2,
        max_value=12,
        lambda_prior=lambda_prior,
        tau_learn=123.45,
        tau_forget=None,
        seed=124816,
    )

    # --- assert ------------------------------------------
    assert isinstance(sampler, TruncatedPoissonAdaptiveSampler)

    if lambda_prior is not None:
        assert sampler._lambda_prior == pytest.approx(3.7)
    else:
        assert sampler._lambda_prior == pytest.approx(0.5 * (2 + 12))
    assert sampler._tau_learn == pytest.approx(123.45)
    assert sampler._tau_forget == pytest.approx(123.45 * 123.45)
    assert sampler._seed == 124816
