from collections.abc import Callable
from unittest.mock import Mock

import numpy as np
import pytest

from max_div._core.solver._parameters import ParameterSchedule, ParameterValueSource, ease_in, linear
from max_div._core.solver._parameters.samplers import sampled_interval
from max_div._core.solver._solver_state import SolverState
from max_div._core.solver._strategies import OptimizationStrategy


# =================================================================================================
#  TEST - Basic functionality
# =================================================================================================
@pytest.mark.parametrize(
    "factory_method",
    [
        OptimizationStrategy.random_swaps,
        OptimizationStrategy.guided_swaps,
    ],
    ids=[
        "random_swaps",
        "guided_swaps",
    ],
)
def test_optimization_strategy_factory(factory_method: Callable[[], OptimizationStrategy]):
    """Test factory methods of OptimizationStrategy base class."""

    # --- act & assert ------------------------------------
    assert isinstance(factory_method(), OptimizationStrategy)


def test_optimization_strategy_properties():
    # --- arrange -----------------------------------------
    class TestOptimizationStrategy(OptimizationStrategy):
        def __init__(self, name: str | None = None):
            super().__init__(name)

        def _perform_single_iteration(self, state: SolverState) -> bool:
            return True

    # --- act & assert ------------------------------------
    assert TestOptimizationStrategy().name == "TestOptimizationStrategy"
    assert TestOptimizationStrategy("custom_name").name == "custom_name"


@pytest.mark.parametrize(
    "param, expected_initial_value",
    [
        (1.2345, 1.2345),
        (1, 1.0),
        (linear(3.456, 7.890), 3.456),
    ],
)
def test_optimization_strategy_initial_param_value(param: ParameterSchedule | float, expected_initial_value: float):
    # --- act ---------------------------------------------
    initial_value = OptimizationStrategy.initial_param_value(param)

    # --- assert ------------------------------------------
    assert isinstance(initial_value, float)
    assert initial_value == expected_initial_value


# =================================================================================================
#  TEST - Dynamic parameters
# =================================================================================================
class StrategyWithDynamicParameters(OptimizationStrategy):
    def __init__(self, param_a: float | ParameterValueSource, param_b: float | ParameterValueSource):
        self.param_a = param_a
        self.param_b = param_b
        super().__init__(
            dynamic_params={
                "param_a": param_a,
                "param_b": param_b,
            }
        )

        # add observability for testing
        self.observed_a_values = []
        self.observed_b_values = []

    def _perform_single_iteration(self, state: SolverState, progress_frac: float):
        self.observed_a_values.append(self.param_a)
        self.observed_b_values.append(self.param_b)


def test_optimization_strategy_dynamic_params_without():
    # --- arrange -----------------------------------------
    strategy = StrategyWithDynamicParameters(
        param_a=0.1,  # fixed value, no scheduling
        param_b=0.9,  # fixed value, no scheduling
    )
    solver_state = Mock()

    # --- act ---------------------------------------------
    _ = strategy.perform_n_iterations(
        state=solver_state,
        n_iters=3,
        current_progress_frac=0.0,
        progress_frac_per_iter=0.5,
    )

    # --- assert ------------------------------------------
    assert not strategy.has_scheduled_params
    assert not strategy.has_sampled_params
    assert not strategy.has_dynamic_params

    assert strategy.observed_a_values == [0.1, 0.1, 0.1]
    assert strategy.observed_b_values == [0.9, 0.9, 0.9]


@pytest.mark.parametrize("param_b_scheduled", [True, False])
def test_optimization_strategy_dynamic_params_scheduled(param_b_scheduled: bool):
    # --- arrange -----------------------------------------
    strategy = StrategyWithDynamicParameters(
        param_a=linear(1.0, 2.0),  # scheduled from 1.0 to 2.0
        param_b=ease_in(3.0, 1.0) if param_b_scheduled else 0.9,
    )
    solver_state = Mock()

    # --- act ---------------------------------------------
    _ = strategy.perform_n_iterations(
        state=solver_state,
        n_iters=3,
        current_progress_frac=0.0,
        progress_frac_per_iter=0.5,
    )

    # --- assert ------------------------------------------
    assert strategy.has_scheduled_params
    assert strategy.has_dynamic_params
    assert not strategy.has_sampled_params

    assert np.allclose(strategy.observed_a_values, [1.0, 1.5, 2.0])
    if param_b_scheduled:
        assert np.allclose(strategy.observed_b_values, [3.0, 2.5, 1.0])
    else:
        assert strategy.observed_b_values == [0.9, 0.9, 0.9]


@pytest.mark.parametrize("param_b_sampled", [True, False])
def test_optimization_strategy_dynamic_params_sampled(param_b_sampled: bool):
    # --- arrange -----------------------------------------
    param_b_range = (1.0, 3.0)
    param_b_sampler = sampled_interval(
        min_value=param_b_range[0],
        max_value=param_b_range[1],
    )

    strategy = StrategyWithDynamicParameters(
        param_a=1.5,  # fixed value
        param_b=param_b_sampler if param_b_sampled else 0.9,
    )
    solver_state = Mock()

    # --- act 1 -------------------------------------------
    rng_state_before = param_b_sampler._rng_state.copy()
    strategy.set_seed(seed=1000)

    # --- assert 1 ----------------------------------------
    if param_b_sampled:
        assert not np.array_equal(param_b_sampler._rng_state, rng_state_before)
    else:
        assert np.array_equal(param_b_sampler._rng_state, rng_state_before)

    # --- act 2 -------------------------------------------
    _ = strategy.perform_n_iterations(
        state=solver_state,
        n_iters=10,
        current_progress_frac=0.0,
        progress_frac_per_iter=0.1,
    )

    # --- assert 2 ----------------------------------------
    assert strategy.has_dynamic_params == param_b_sampled
    assert strategy.has_sampled_params == param_b_sampled
    assert not strategy.has_scheduled_params

    assert all(a_value == 1.5 for a_value in strategy.observed_a_values)
    if param_b_sampled:
        assert all(min(param_b_range) <= b_value <= max(param_b_range) for b_value in strategy.observed_b_values)
    else:
        assert all(b_value == 0.9 for b_value in strategy.observed_b_values)
