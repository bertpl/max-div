"""
Test methods implemented in the AdaptiveSampler base class, by means of the simplest child class BooleanAdaptiveSampler.
"""

import math

import numpy as np
import pytest

from max_div.solver._parameters import AdaptiveSampler
from max_div.solver._parameters.samplers import BooleanAdaptiveSampler


def test_adaptive_sampler_construction():
    # --- arrange -----------------------------------------
    sampler = BooleanAdaptiveSampler(
        p_true_prior=0.2,
        tau_learn=10.0,
        tau_forget=100.0,
        seed=123,
    )

    # --- act & assert ------------------------------------
    assert isinstance(sampler, AdaptiveSampler)

    assert sampler._rng_state.sum() != 0
    assert sampler._tau_learn == pytest.approx(10.0)
    assert sampler._tau_forget == pytest.approx(100.0)
    assert sampler._forgetting_enabled is True
    assert isinstance(sampler.get_initial_value(), bool), (
        "Initial value should return a value the sampler can generate."
    )


@pytest.mark.parametrize("tau_forget", [math.inf, np.inf], ids=["math_inf", "np_inf"])
def test_adaptive_sampler_forgetting_disabled(tau_forget):
    # --- arrange -----------------------------------------
    sampler = BooleanAdaptiveSampler(
        p_true_prior=0.2,
        tau_learn=10.0,
        tau_forget=tau_forget,
        seed=123,
    )

    # --- act & assert ------------------------------------
    assert sampler._forgetting_enabled is False
    assert np.isinf(sampler._tau_forget)
    assert sampler._c_forget == 0.0
    assert sampler._c_forget_f32 == 0.0


def test_adaptive_sampler_update_seed():
    # --- arrange -----------------------------------------
    sampler = BooleanAdaptiveSampler(
        p_true_prior=0.2,
        tau_learn=10.0,
        tau_forget=100.0,
        seed=123,
    )
    rng_state_before = sampler._rng_state.copy()

    # --- act ---------------------------------------------
    sampler.update_seed(1234)

    # --- assert ------------------------------------------
    rng_state_after = sampler._rng_state.copy()
    assert not np.array_equal(rng_state_before, rng_state_after)


def test_adaptive_sampler_update_tau():
    # --- arrange -----------------------------------------
    sampler = BooleanAdaptiveSampler(
        p_true_prior=0.2,
        tau_learn=10.0,
        tau_forget=100.0,
    )

    # --- act ---------------------------------------------
    sampler.update_tau(tau_learn=20.0, tau_forget=200.0)

    # --- assert ------------------------------------------
    assert sampler._tau_learn == pytest.approx(20.0)
    assert sampler._tau_forget == pytest.approx(200.0)
