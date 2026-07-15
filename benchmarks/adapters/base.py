"""The adapter contract every compared tool implements."""

import time
from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray

from max_div.problem import MaxDivProblem


class SelectionAdapter(ABC):
    """A single-shot subset-selection tool wrapped behind a uniform interface.

    Adapters return raw selections; quality scoring and record building happen in the
    runners, identically for every tool.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Tool name as it appears in records and figures."""

    @property
    def supports_constraints(self) -> bool:
        """Whether the tool can honor the problem's fairness constraints (default: no)."""
        return False

    @abstractmethod
    def select(self, problem: MaxDivProblem, seed: int) -> NDArray[np.int64]:
        """Select k item indices for the problem; must be deterministic given the seed."""

    def timed_select(self, problem: MaxDivProblem, seed: int) -> tuple[NDArray[np.int64], float]:
        """Run select() and measure its wall-clock duration in seconds."""
        t0 = time.perf_counter()
        indices = self.select(problem, seed)
        return indices, time.perf_counter() - t0
