from __future__ import annotations

import math
import time
from abc import ABC, abstractmethod
from enum import StrEnum

from max_div.solver._solver_state import SolverState


# =================================================================================================
#  StrategyType
# =================================================================================================
class StrategyType(StrEnum):
    INITIALIZATION = "INITIALIZATION"
    OPTIMIZATION = "OPTIMIZATION"


# =================================================================================================
#  StrategyDuration
# =================================================================================================
class StrategyDuration:
    # -------------------------------------------------------------------------
    #  Constructor
    # -------------------------------------------------------------------------
    def __init__(self, max_iterations: int | None = None, max_seconds: float | None = None):
        self._max_iterations = max_iterations
        self._max_seconds = max_seconds

    # -------------------------------------------------------------------------
    #  Main API
    # -------------------------------------------------------------------------
    def finished(self, n_iterations: int, n_seconds: float) -> bool:
        if self._max_iterations is not None:
            if n_iterations >= self._max_iterations:
                return True
        if self._max_seconds is not None:
            if n_seconds >= self._max_seconds:
                return True
        return False

    def progress(self, n_iterations: int, n_seconds: float) -> tuple[int, int]:
        """Returns progress as (n_current, n_total) to be used for e.g. tqdm progress updates."""
        if self._max_iterations is not None:
            frac_iterations = n_iterations / self._max_iterations
        else:
            frac_iterations = -math.inf
        if self._max_seconds is not None:
            frac_seconds = n_seconds / self._max_seconds
        else:
            frac_seconds = -math.inf

        if frac_iterations > frac_seconds:
            return n_iterations, self._max_iterations
        elif frac_seconds > frac_iterations:
            return int(n_seconds), int(self._max_seconds)
        else:
            return 0, 1  # indeterminate progress

    # -------------------------------------------------------------------------
    #  Factory Methods
    # -------------------------------------------------------------------------
    @classmethod
    def iterations(cls, max_iterations: int) -> StrategyDuration:
        return StrategyDuration(max_iterations=max_iterations)

    @classmethod
    def seconds(cls, max_seconds: float) -> StrategyDuration:
        return StrategyDuration(max_seconds=max_seconds)

    @classmethod
    def minutes(cls, max_minutes: float) -> StrategyDuration:
        return StrategyDuration(max_seconds=max_minutes * 60.0)

    @classmethod
    def hours(cls, max_hours: float) -> StrategyDuration:
        return StrategyDuration(max_seconds=max_hours * 3600.0)


# shorthand
iterations = StrategyDuration.iterations
seconds = StrategyDuration.seconds
minutes = StrategyDuration.minutes
hours = StrategyDuration.hours


# =================================================================================================
#  SolverStrategy
# =================================================================================================
class SolverStrategy(ABC):
    # -------------------------------------------------------------------------
    #  Construction & Configuration
    # -------------------------------------------------------------------------
    def __init__(self, strategy_type: StrategyType, duration: StrategyDuration):
        """
        Initialize the solver strategy.
        :param strategy_type: type of the strategy
                               if StrategyType.INITIALIZATION, the class implements an initialization strategy.
                               if StrategyType.OPTIMIZATION,   the class implements an optimization strategy.
        :param duration: duration settings for the strategy, determines how long it will keep iterating until it's done.
        """
        self._strategy_type = strategy_type
        self._duration = duration

    @property
    def type(self) -> StrategyType:
        return self._strategy_type

    @property
    def duration(self) -> StrategyDuration:
        return self._duration

    @property
    def name(self) -> str:
        return self.__class__.__name__

    # -------------------------------------------------------------------------
    #  Main API
    # -------------------------------------------------------------------------
    def run(self, state: SolverState):
        # --- init ----------------------------------------
        t_start = time.perf_counter()
        n_iters = 0

        # --- main loop -----------------------------------
        while not self.duration.finished(
            n_iterations=n_iters,
            n_seconds=time.perf_counter() - t_start,
        ):
            self.do_one_iteration(state)
            n_iters += 1

    @abstractmethod
    def do_one_iteration(self, state: SolverState):
        """
        Perform one iteration of the strategy, modifying the solver state in-place, to reach a more optimal solution.
        :param state: (SolverState) The current solver state.
        """
        raise NotImplementedError()

    # -------------------------------------------------------------------------
    #  Factory Methods
    # -------------------------------------------------------------------------
    @classmethod
    def init_random(cls) -> SolverStrategy:
        from ._init_random import InitRandom

        return InitRandom()

    @classmethod
    def optim_dummy(cls) -> SolverStrategy:
        from ._optim_dummy import OptimDummy

        return OptimDummy()
