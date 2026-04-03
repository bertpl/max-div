import math
from unittest.mock import Mock

import numpy as np
import pytest
from numpy._typing import NDArray

from max_div._core._utils import Timer
from max_div._core.solver._duration import Elapsed, iterations, seconds
from max_div._core.solver._score import Score
from max_div._core.solver._solver_state import SolverState
from max_div._core.solver._solver_step import InitializationStep, OptimizationStep, SolverStepResult
from max_div._core.solver._strategies import InitializationStrategy, OptimizationStrategy

# =================================================================================================
#  Helpers
# =================================================================================================


# --- Test Strategy Implementations -----------------------
class InitTest(InitializationStrategy):
    def __init__(self):
        super().__init__()
        self._n_iterations = 0

    def get_next_samples(self, state: SolverState, k_remaining: int | np.int32) -> NDArray[np.int32]:
        self._n_iterations += 1
        return state.not_selected_index_array[:k_remaining]


class OptimTest(OptimizationStrategy):
    def __init__(self):
        super().__init__()
        self._n_iterations = 0
        self._progress_fracs = []

    def _perform_single_iteration(self, state: SolverState, progress_frac: float):
        self._n_iterations += 1
        self._progress_fracs.append(progress_frac)


class DummySolverState:
    def __init__(self, n: int, k: int):
        self.n = n
        self.k = k
        self.n_selected = 0
        self.score = Mock()

    def add(self, s: int):
        self.n_selected += 1

    def add_many(self, samples: NDArray[np.int32]):
        self.n_selected += len(samples)

    @property
    def not_selected_index_array(self):
        return np.arange(self.n, dtype=np.int32)[self.n_selected :]


# --- checks ----------------------------------------------
def assert_score_checkpoints_are_sane(score_checkpoints: list[tuple[Elapsed, Score]]):
    # --- non-empty -------------------
    assert len(score_checkpoints) >= 1, "score_checkpoints must contain at least one entry"

    # --- check iteration counts ------
    iter_values = [e.n_iterations for e, _ in score_checkpoints]
    assert min(iter_values) >= 0, "score_checkpoints contains negative iteration counts"
    assert len(iter_values) == len(set(iter_values)), "score_checkpoints contains duplicate iteration counts"
    assert iter_values == sorted(iter_values), "score_checkpoints iteration counts should be strictly increasing"
    for j in range(len(iter_values) - 1):
        # subsequent iteration counts should be roughly ~10% spaced apart
        i, i_next = iter_values[j], iter_values[j + 1]
        i_delta = i_next - i

        # allowed ranges
        i_delta_min = math.floor(0.09 * i)
        i_delta_max = max(1, math.ceil(0.11 * i))
        if j == len(iter_values) - 2:
            i_delta_min = 0  # last checkpoint is allowed to be closer

        # check in range
        assert i_delta_min <= i_delta <= i_delta_max, f"checkpoints should be ~10%-spaced; here: {i} -> {i_next}"

    # --- check elapsed times ---------
    t_values = [e.t_elapsed_sec for e, _ in score_checkpoints]
    assert min(t_values) >= 0.0, "score_checkpoints contains negative elapsed times"
    # NOTE: duplicate time values can happen if iterations are very fast, so we don't assert uniqueness here
    assert t_values == sorted(t_values), "score_checkpoints elapsed times should be non-decreasing"


# =================================================================================================
#  InitializationStep
# =================================================================================================
def test_initialization_step_validation():
    # this should work just fine
    _ = InitializationStep(InitTest())

    # this should not work fine
    with pytest.raises(TypeError):
        _ = InitializationStep(OptimTest())


def test_initialization_step_name():
    # --- arrange -----------------------------------------
    strategy = InitTest()
    step = InitializationStep(strategy)

    # --- act ---------------------------------------------
    step_name = step.name()

    # --- assert ------------------------------------------
    assert step_name == strategy.name


def test_initialization_step_run():
    # --- arrange -----------------------------------------
    strategy = InitTest()
    step = InitializationStep(strategy)
    state = DummySolverState(n=100, k=10)

    # --- act ---------------------------------------------
    result = step.run(state)

    # --- assert ------------------------------------------
    assert strategy._n_iterations == 1, "This initialization should take exactly 1 iteration"
    assert isinstance(result, SolverStepResult)
    assert result.elapsed.n_iterations == 1, "This initialization should take exactly 1 iteration"
    assert_score_checkpoints_are_sane(result.score_checkpoints)


# =================================================================================================
#  OptimizationStep
# =================================================================================================
def test_optimization_step_validation():
    # this should work just fine
    _ = OptimizationStep(OptimTest(), duration=seconds(1))

    # this should not work fine
    with pytest.raises(TypeError):
        _ = OptimizationStep(InitTest(), duration=seconds(1))


def test_optimization_step_name():
    # --- arrange -----------------------------------------
    strategy = OptimTest()
    step = OptimizationStep(strategy, duration=seconds(1))

    # --- act ---------------------------------------------
    step_name = step.name()

    # --- assert ------------------------------------------
    assert step_name == strategy.name


def test_optimization_step_run_iterations():
    # --- arrange -----------------------------------------
    strategy = OptimTest()
    step = OptimizationStep(strategy, duration=iterations(123))
    state = Mock()

    # --- act ---------------------------------------------
    result = step.run(state)

    # --- assert ------------------------------------------
    assert strategy._n_iterations == 123
    for i, progress_frac in enumerate(strategy._progress_fracs):
        assert math.isclose(i / 123, progress_frac)

    assert isinstance(result, SolverStepResult)
    assert result.elapsed.n_iterations == 123
    assert_score_checkpoints_are_sane(result.score_checkpoints)


def test_optimization_step_run_seconds():
    # --- arrange -----------------------------------------
    strategy = OptimTest()
    step = OptimizationStep(strategy, duration=seconds(0.1))
    state = Mock()

    # --- act ---------------------------------------------
    with Timer() as t:
        result = step.run(state)

    # --- assert ------------------------------------------
    assert t.t_elapsed_sec() >= 0.1
    assert isinstance(result, SolverStepResult)
    assert result.elapsed.t_elapsed_sec >= 0.1
    assert result.elapsed.n_iterations == strategy._n_iterations > 0
    assert_score_checkpoints_are_sane(result.score_checkpoints)
