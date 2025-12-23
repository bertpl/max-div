from unittest.mock import Mock

import numpy as np
import pytest

from max_div.solver._scheduling import ParameterSchedule, ease_in, linear
from max_div.solver._solver_state import SolverState
from max_div.solver._strategies import OptimizationStrategy


# =================================================================================================
#  TEST - Basic functionality
# =================================================================================================
def test_optimization_strategy_factory():
    assert isinstance(OptimizationStrategy.dummy(), OptimizationStrategy)


def test_optimization_strategy_properties():
    # --- arrange -----------------------------------------
    class TestOptimizationStrategy(OptimizationStrategy):
        def __init__(self, name: str | None = None):
            super().__init__(name)

        def _perform_single_iteration(self, state: SolverState):
            pass

    # --- act & assert ------------------------------------
    assert TestOptimizationStrategy().name == "TestOptimizationStrategy"
    assert TestOptimizationStrategy("custom_name").name == "custom_name"


# =================================================================================================
#  TEST - Parameter Scheduling
# =================================================================================================
class StrategyWithScheduling(OptimizationStrategy):
    def __init__(self, param_a: float | ParameterSchedule, param_b: float | ParameterSchedule):
        self.param_a = param_a
        self.param_b = param_b
        super().__init__(
            scheduled_params=dict(
                param_a=param_a,
                param_b=param_b,
            )
        )

        # add observability for testing
        self.observed_a_values = []
        self.observed_b_values = []

    def _perform_single_iteration(self, state: SolverState, progress_frac: float):
        self.observed_a_values.append(self.param_a)
        self.observed_b_values.append(self.param_b)


def test_optimization_strategy_param_scheduling_without():
    # --- arrange -----------------------------------------
    strategy = StrategyWithScheduling(
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
    assert strategy.observed_a_values == [0.1, 0.1, 0.1]
    assert strategy.observed_b_values == [0.9, 0.9, 0.9]


@pytest.mark.parametrize("param_b_scheduled", [True, False])
def test_optimization_strategy_param_scheduling_with(param_b_scheduled: bool):
    # --- arrange -----------------------------------------
    strategy = StrategyWithScheduling(
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
    assert np.allclose(strategy.observed_a_values, [1.0, 1.5, 2.0])
    if param_b_scheduled:
        assert np.allclose(strategy.observed_b_values, [3.0, 2.5, 1.0])
    else:
        assert strategy.observed_b_values == [0.9, 0.9, 0.9]
