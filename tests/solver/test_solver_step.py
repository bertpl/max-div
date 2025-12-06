from unittest.mock import Mock

import pytest

from max_div.internal.benchmarking import Timer
from max_div.solver._duration import iterations, seconds
from max_div.solver._solver_state import SolverState
from max_div.solver._solver_step import InitializationStep, OptimizationStep, SolverStepResult
from max_div.solver._strategies import InitializationStrategy, OptimizationStrategy


# =================================================================================================
#  Testing Classes
# =================================================================================================
class InitTest(InitializationStrategy):
    def __init__(self):
        super().__init__()
        self._n_iterations = 0

    def initialize(self, state: SolverState):
        self._n_iterations += 1


class OptimTest(OptimizationStrategy):
    def __init__(self):
        super().__init__()
        self._n_iterations = 0

    def _perform_single_iteration(self, state: SolverState):
        self._n_iterations += 1


# =================================================================================================
#  InitializationStep
# =================================================================================================
def test_initialization_step_validation():
    # this should work just fine
    _ = InitializationStep(InitTest())

    # this should not work fine
    with pytest.raises(TypeError):
        _ = InitializationStep(OptimTest())


def test_initialization_step_run():
    # --- arrange ---
    strategy = InitTest()
    step = InitializationStep(strategy)
    state = Mock()

    # --- act ---
    result = step.run(state)

    # --- assert ---
    assert strategy._n_iterations == 1
    assert isinstance(result, SolverStepResult)
    assert result.duration.t_elapsed_sec >= 0.0
    assert result.duration.n_iterations == 1


# =================================================================================================
#  OptimizationStep
# =================================================================================================
def test_optimization_step_validation():
    # this should work just fine
    _ = OptimizationStep(OptimTest(), duration=seconds(1))

    # this should not work fine
    with pytest.raises(TypeError):
        _ = OptimizationStep(InitTest(), duration=seconds(1))


def test_optimization_step_run_iterations():
    # --- arrange ---
    strategy = OptimTest()
    step = OptimizationStep(strategy, duration=iterations(123))
    state = Mock()

    # --- act ---
    result = step.run(state)

    # --- assert ---
    assert strategy._n_iterations == 123
    assert isinstance(result, SolverStepResult)
    assert result.duration.t_elapsed_sec >= 0.0
    assert result.duration.n_iterations == 123


def test_optimization_step_run_seconds():
    # --- arrange ---
    strategy = OptimTest()
    step = OptimizationStep(strategy, duration=seconds(0.1))
    state = Mock()

    # --- act ---
    with Timer() as t:
        result = step.run(state)

    # --- assert ---
    assert t.t_elapsed_sec() >= 0.1
    assert isinstance(result, SolverStepResult)
    assert result.duration.t_elapsed_sec >= 0.1
    assert result.duration.n_iterations == strategy._n_iterations > 0
