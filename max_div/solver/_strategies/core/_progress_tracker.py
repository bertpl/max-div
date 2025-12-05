from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Self


# =================================================================================================
#  Progress
# =================================================================================================
@dataclass(frozen=True, slots=True)
class Progress:
    n_current: int
    n_total: int

    @property
    def is_finished(self) -> bool:
        return self.n_current >= self.n_total


# =================================================================================================
#  ProgressTracker
# =================================================================================================
class ProgressTracker(ABC):
    def __init__(self):
        self._t_start = 0.0
        self._iter_count = 0

    def start(self):
        self._t_start = time.perf_counter()
        self._iter_count = 0

    def t_elapsed_sec(self) -> float:
        return time.perf_counter() - self._t_start

    def iter_count(self) -> int:
        return self._iter_count

    @abstractmethod
    def progress(self) -> Progress:
        """Returns Progress object."""
        raise NotImplementedError

    def iteration_done(self):
        self._iter_count += 1


# =================================================================================================
#  Iteration-based
# =================================================================================================
class IterationBasedProgress(ProgressTracker):
    def __init__(self, max_iters: int):
        super().__init__()
        if max_iters < 1:
            raise ValueError("max_iters must be >= 1")
        self._max_iters = max_iters

    def progress(self) -> Progress:
        return Progress(
            n_current=min(self._iter_count, self._max_iters),
            n_total=self._max_iters,
        )

    # -------------------------------------------------------------------------
    #  Factory Methods
    # -------------------------------------------------------------------------
    @classmethod
    def iterations(cls, max_iters: int) -> Self:
        return IterationBasedProgress(max_iters=max_iters)


# =================================================================================================
#  Time-based
# =================================================================================================
class TimeBasedProgress(ProgressTracker):
    def __init__(self, max_seconds: float):
        super().__init__()
        if max_seconds <= 0.0:
            raise ValueError("max_seconds must be > 0.0")
        self._max_seconds = max_seconds
        self._n_total = max(1, int(self._max_seconds))  # never <1

    def progress(self) -> Progress:
        t_elapsed = self.t_elapsed_sec()
        if t_elapsed >= self._max_seconds:
            n_current = self._n_total
        else:
            n_current = min(int(t_elapsed), self._n_total - 1)

        return Progress(
            n_current=n_current,
            n_total=self._n_total,
        )

    # -------------------------------------------------------------------------
    #  Factory Methods
    # -------------------------------------------------------------------------
    @classmethod
    def seconds(cls, max_seconds: float) -> Self:
        return TimeBasedProgress(max_seconds=max_seconds)

    @classmethod
    def minutes(cls, max_minutes: float) -> Self:
        return TimeBasedProgress(max_seconds=max_minutes * 60.0)

    @classmethod
    def hours(cls, max_hours: float) -> Self:
        return TimeBasedProgress(max_seconds=max_hours * 3600.0)


# =================================================================================================
#  Shorthand Factory Methods
# =================================================================================================
iterations = IterationBasedProgress.iterations
seconds = TimeBasedProgress.seconds
minutes = TimeBasedProgress.minutes
hours = TimeBasedProgress.hours
