import math

import numpy as np
import pytest

from max_div._core.solver._parameters import AdaptiveSampler
from max_div._core.solver._parameters.samplers import SkewedIntervalAdaptiveSampler, sampled_interval


def test_skewed_interval_adaptive_sampler_construction():
    # --- arrange ----------------------
    sampler = SkewedIntervalAdaptiveSampler(
        min_value=1.0,
        max_value=5.0,
        median_prior=3.0,
        tau_learn=10.0,
        tau_forget=100.0,
        seed=123,
    )

    # --- act & assert -----------------
    assert isinstance(sampler, AdaptiveSampler)
    assert isinstance(sampler, SkewedIntervalAdaptiveSampler)

    assert all(isinstance(sampler.new_sample(), np.float32) for _ in range(1000))  # all should be of type 'float32'
    assert all(1.0 <= sampler.new_sample() <= 5.0 for _ in range(1000))  # all should be in [1.0, 5.0]

    # expected value checks
    assert sampler.summary_statistic() == pytest.approx(3.0)
    assert 2.5 < np.mean([sampler.new_sample() for _ in range(1000)]) < 3.5


def test_skewed_interval_adaptive_sampler_construction_validation():
    with pytest.raises(ValueError):
        _ = SkewedIntervalAdaptiveSampler(
            min_value=1.0,
            max_value=5.0,
            median_prior=5.1,
            tau_learn=10.0,
            tau_forget=100.0,
            seed=123,
        )


@pytest.mark.parametrize("good_interval", [(1.0, 2.0), (4.0, 5.0)])
@pytest.mark.parametrize("tau_forget", [100.0, math.inf])
def test_boolean_adaptive_sampler_learn_and_forget(good_interval: tuple[float, float], tau_forget):
    # --- arrange ----------------------
    sampler = SkewedIntervalAdaptiveSampler(
        min_value=1.0,
        max_value=5.0,
        median_prior=3.0,
        tau_learn=10.0,
        tau_forget=tau_forget,
        seed=123,
    )

    min_good_value, max_good_value = good_interval

    # --- act 1 - learn ----------------
    for _ in range(1_000):
        s = sampler.new_sample()
        sampler.feedback(min_good_value <= s <= max_good_value)  # positive feedback when we sampled in 'good_interval'

    # --- assert 1 - learn -------------
    assert min_good_value < sampler.summary_statistic() < max_good_value  # should learn to sample good range

    # --- act 2 - forget ---------------
    summary_stat_before = sampler.summary_statistic()
    for _ in range(1_000):
        s = sampler.new_sample()
        sampler.feedback(False)  # negative feedback always, so we maximally forget

    # --- assert 2 - forget ------------
    if not np.isinf(tau_forget):
        # forgetting is enabled
        assert 2.5 < sampler.summary_statistic() < 3.5  # should have forgotten back to prior (5.5)
    else:
        # forgetting is disabled
        assert sampler.summary_statistic() == summary_stat_before


@pytest.mark.parametrize("median_prior", [0.25, None])
def test_skewed_interval_alias(median_prior: float | None):
    # --- act --------------------------
    sampler = sampled_interval(
        min_value=0.2,
        max_value=0.8,
        median_prior=median_prior,
        tau_learn=123.45,
        seed=1248,
    )

    # --- assert -----------------------
    assert isinstance(sampler, SkewedIntervalAdaptiveSampler)
    if median_prior is not None:
        assert sampler._median_prior == pytest.approx(median_prior)
    else:
        assert sampler._median_prior == pytest.approx(0.5 * (0.2 + 0.8))
    assert sampler._tau_learn == pytest.approx(123.45)
    assert sampler._tau_forget == pytest.approx(123.45 * 123.45)


def test_skewed_interval_adaptive_sampler_reset():
    """reset() restores the prior median after feedback has shifted it."""
    # --- arrange ----------------------
    sampler = sampled_interval(min_value=-0.99, max_value=0.99, median_prior=0.0, tau_learn=5.0)
    for _ in range(200):
        sample = sampler.new_sample()
        sampler.feedback(success=sample >= 0.3)
    assert sampler.summary_statistic() != pytest.approx(0.0)

    # --- act --------------------------
    sampler.reset(seed=42)

    # --- assert -----------------------
    assert sampler.summary_statistic() == pytest.approx(0.0)
