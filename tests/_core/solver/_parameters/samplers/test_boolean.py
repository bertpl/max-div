import math

import numpy as np
import pytest

from max_div._core.solver._parameters import AdaptiveSampler
from max_div._core.solver._parameters.samplers import BooleanAdaptiveSampler, sampled_boolean


def test_boolean_adaptive_sampler_construction():
    # --- arrange ----------------------
    sampler = BooleanAdaptiveSampler(
        p_true_prior=0.2,
        tau_learn=10.0,
        tau_forget=100.0,
        seed=123,
    )

    # --- act & assert -----------------
    assert isinstance(sampler, AdaptiveSampler)
    assert isinstance(sampler, BooleanAdaptiveSampler)

    assert sampler.summary_statistic() == pytest.approx(0.2)

    assert all(isinstance(sampler.new_sample(), bool) for _ in range(1000))  # all should be of type 'bool'
    assert 150 < sum([1 for _ in range(1000) if sampler.new_sample()]) < 250  # roughly 20% True
    assert 750 < sum([1 for _ in range(1000) if not sampler.new_sample()]) < 850  # roughly 80% False


def test_boolean_adaptive_sampler_construction_validation():
    with pytest.raises(ValueError):
        _ = BooleanAdaptiveSampler(
            p_true_prior=1.1,
            tau_learn=10.0,
            tau_forget=100.0,
            seed=123,
        )


@pytest.mark.parametrize("good_value", [True, False])
@pytest.mark.parametrize("tau_forget", [100.0, math.inf])
def test_boolean_adaptive_sampler_learn_and_forget(good_value: bool, tau_forget):
    # --- arrange ----------------------
    sampler = BooleanAdaptiveSampler(
        p_true_prior=0.5,
        tau_learn=10.0,
        tau_forget=tau_forget,
    )

    # --- act 1 - learn ----------------
    for _ in range(1_000):
        s = sampler.new_sample()
        sampler.feedback(s == good_value)  # positive feedback when we sampled 'good_value'

    # --- assert 1 - learn -------------
    if good_value:
        assert sampler.summary_statistic() > 0.9  # should have learned to sample True more often
    else:
        assert sampler.summary_statistic() < 0.1  # should have learned to sample False more often

    # --- act 2 - forget ---------------
    summary_stat_before = sampler.summary_statistic()
    for _ in range(1_000):
        s = sampler.new_sample()
        sampler.feedback(False)  # negative feedback always, so we maximally forget

    # --- assert 2 - forget ------------
    if not np.isinf(tau_forget):
        # forgetting is enabled
        assert 0.45 < sampler.summary_statistic() < 0.55  # should have forgotten back to prior (0.5)
    else:
        # forgetting is disabled
        assert sampler.summary_statistic() == summary_stat_before


def test_sampled_boolean_alias():
    # --- act --------------------------
    sampler = sampled_boolean(
        p_true_prior=0.8,
        tau_learn=123.45,
        seed=1248,
    )

    # --- assert -----------------------
    assert isinstance(sampler, BooleanAdaptiveSampler)
    assert sampler._p_true_prior == pytest.approx(0.8)
    assert sampler._tau_learn == pytest.approx(123.45)
    assert sampler._tau_forget == pytest.approx(123.45 * 123.45)


def test_boolean_adaptive_sampler_reset_learning():
    """reset_learning() restores the prior probability after feedback has shifted it."""
    # --- arrange ----------------------
    sampler = sampled_boolean(p_true_prior=0.5, tau_learn=5.0)
    for _ in range(200):
        sample = sampler.new_sample()
        sampler.feedback(success=sample)
    assert sampler.summary_statistic() != pytest.approx(0.5)

    # --- act --------------------------
    sampler.reset_learning()

    # --- assert -----------------------
    assert sampler.summary_statistic() == pytest.approx(0.5)
