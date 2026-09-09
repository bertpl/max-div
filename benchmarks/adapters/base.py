"""The adapter contract every compared tool implements."""

import time
from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray

from benchmarks.common.registry import display_name
from max_div.problem import MaxDivProblem


class SelectionAdapter(ABC):
    """A single-shot subset-selection tool wrapped behind a uniform interface.

    Adapters return raw selections; quality scoring and record building happen in the
    runners, identically for every tool.

    `tool_key` is the tool's solver-registry key and `config` the configuration name the
    solver-configurations page lists, so a record label reads `<display name>[<config>]` and
    names the same configuration on every benchmark page.
    """

    tool_key: str
    config: str

    @property
    def name(self) -> str:
        """Return the tool label used in records and figures: the registry display name and the configuration."""
        return f"{display_name(self.tool_key)}[{self.config}]"

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
