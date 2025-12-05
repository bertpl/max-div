from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Self


# =================================================================================================
#  DurationProgress
# =================================================================================================
@dataclass(frozen=True, slots=True)
class DurationProgress:
    n_current: int
    n_total: int
    is_finished: bool


# =================================================================================================
#  StrategyDuration
# =================================================================================================
class StrategyDuration(ABC):
    @abstractmethod
    def start(self):
        pass

    @abstractmethod
    def progress(self) -> DurationProgress:
        """Returns progress as (n_current, n_total) to be used for e.g. tqdm progress updates."""
        raise NotImplementedError

    def iteration_done(self):
        pass  # optional hook for subclasses


# =================================================================================================
#  Iteration-based
# =================================================================================================
class IterationBasedDuration(StrategyDuration):
    def __init__(self, max_iters: int):
        self._max_iters = max_iters
        self._n_iters_finished = 0

    def start(self):
        self._n_iters_finished = 0

    def iteration_done(self):
        self._n_iters_finished += 1

    def progress(self) -> DurationProgress:
        return DurationProgress(
            n_current=self._n_iters_finished,
            n_total=self._max_iters,
            is_finished=(self._n_iters_finished >= self._max_iters),
        )

    # -------------------------------------------------------------------------
    #  Factory Methods
    # -------------------------------------------------------------------------
    @classmethod
    def iterations(cls, max_iters: int) -> Self:
        return IterationBasedDuration(max_iters=max_iters)


# =================================================================================================
#  Time-based
# =================================================================================================
class TimeBasedDuration(StrategyDuration):
    def __init__(self, max_seconds: float):
        self._max_seconds = max_seconds
        self._t_start = 0.0
        self._t_finished = 0.0

    def start(self):
        self._t_start = time.perf_counter()
        self._t_finished = time.perf_counter() + self._max_seconds

    def progress(self) -> DurationProgress:
        t_now = time.perf_counter()
        return DurationProgress(
            n_current=int(t_now - self._t_start),
            n_total=int(self._max_seconds),
            is_finished=(t_now >= self._t_finished),
        )

    # -------------------------------------------------------------------------
    #  Factory Methods
    # -------------------------------------------------------------------------
    @classmethod
    def seconds(cls, max_seconds: float) -> Self:
        return TimeBasedDuration(max_seconds=max_seconds)

    @classmethod
    def minutes(cls, max_minutes: float) -> Self:
        return TimeBasedDuration(max_seconds=max_minutes * 60.0)

    @classmethod
    def hours(cls, max_hours: float) -> Self:
        return TimeBasedDuration(max_seconds=max_hours * 3600.0)


# =================================================================================================
#  Shorthand Factory Methods
# =================================================================================================
iterations = IterationBasedDuration.iterations
seconds = TimeBasedDuration.seconds
minutes = TimeBasedDuration.minutes
hours = TimeBasedDuration.hours
