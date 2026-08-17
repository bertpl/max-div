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


class OptimTickingTest(OptimTest):
    """Each iteration advances the fake clock, so a time-budgeted run terminates."""

    def __init__(self, clock, dt_sec: float):
        """Take the clock to tick and the per-iteration step size `dt_sec`."""
        super().__init__()
        self._clock = clock
        self._dt_sec = dt_sec

    def _perform_single_iteration(self, state: SolverState, progress_frac: float):
        """Run the parent's iteration bookkeeping, then advance the clock by `dt_sec`."""
        super()._perform_single_iteration(state, progress_frac)
        self._clock.advance(self._dt_sec)


class DummySolverState:
    def __init__(self, n: int, k: int):
        self.n = n
        self.k = k
        self.m = 0
        self.n_selected = 0
        self.score = Mock()

    def add(self, s: int):
        self.n_selected += 1

    def add_many(self, samples: NDArray[np.int32]):
        self.n_selected += len(samples)

    @property
    def selected_index_array(self):
        """Return the indices currently counted as selected."""
        return np.arange(self.n_selected, dtype=np.int32)

    @property
    def not_selected_index_array(self):
        """Return the indices not currently counted as selected."""
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


def test_optimization_step_run_seconds(fake_clock):
    """A time-budgeted step iterates until its budget is spent, and reports at least that much time."""
    # --- arrange -----------------------------------------
    strategy = OptimTickingTest(fake_clock, dt_sec=0.001)
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


# --- caller-provided batch interval ----------------------
def test_determine_n_iterations_scales_with_the_batch_interval():
    """The batch size is the target interval times the estimated iteration rate."""
    # --- arrange -----------------------------------------
    progress = Mock(est_iters_per_second=1000.0, est_n_iters_remaining=10**9, iter_count=0)

    # --- act / assert ------------------------------------
    assert OptimizationStep._determine_n_iterations(progress, 10**9, batch_seconds=0.5) == 500
    assert OptimizationStep._determine_n_iterations(progress, 10**9, batch_seconds=0.05) == 50


@pytest.mark.parametrize("call_kwargs, expected_batch_seconds", [({}, 0.5), ({"batch_seconds": 0.05}, 0.05)])
def test_run_batches_at_the_interval_it_is_given(monkeypatch, call_kwargs, expected_batch_seconds):
    """The caller's batch interval reaches the batch sizing; omitting it uses the reporting default."""
    # --- arrange -----------------------------------------
    captured = []
    original = OptimizationStep._determine_n_iterations

    def spy(progress, next_checkpoint_iter_count, batch_seconds):
        captured.append(batch_seconds)
        return original(progress, next_checkpoint_iter_count, batch_seconds)

    monkeypatch.setattr(OptimizationStep, "_determine_n_iterations", staticmethod(spy))
    step = OptimizationStep(OptimTest(), duration=iterations(50))

    # --- act ---------------------------------------------
    step.run(Mock(), **call_kwargs)

    # --- assert ------------------------------------------
    assert captured
    assert all(batch_seconds == expected_batch_seconds for batch_seconds in captured)


def test_checkpoint_count_is_batch_invariant():
    """Tightening the batch interval never changes the number of score checkpoints."""
    # --- arrange -----------------------------------------
    step_default = OptimizationStep(OptimTest(), duration=iterations(500))
    step_fast = OptimizationStep(OptimTest(), duration=iterations(500))

    # --- act ---------------------------------------------
    result_default = step_default.run(Mock())
    result_fast = step_fast.run(Mock(), batch_seconds=0.001)

    # --- assert ------------------------------------------
    assert len(result_fast.score_checkpoints) == len(result_default.score_checkpoints)
