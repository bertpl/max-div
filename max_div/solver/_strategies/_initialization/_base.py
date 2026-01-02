from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Self

import numpy as np
from numpy.typing import NDArray

from max_div.solver._solver_state import SolverState
from max_div.solver._strategies._base import StrategyBase


# =================================================================================================
#  InitializationStrategy
# =================================================================================================
class InitializationStrategy(StrategyBase, ABC):
    @abstractmethod
    def get_next_samples(self, state: SolverState, k_remaining: int | np.int32) -> NDArray[np.int32]:
        """
        Return next batch of samples to be added to the initial selection.

        This method is called repeatedly by the Solver, until enough samples have been selected to
        reach the desired selection size.

        :param state: (SolverState) The current solver state, to fetch problem size, constraints, distances, etc...,
                                    so initial selection can be made in an informed way.
        :param k_remaining: (int) number of samples that remain to be selected.
        :return: np.array of unique np.int32 values, shape=(b,), with indices of samples to be added to the selection.
                  b can be any value in range [1, k_remaining].  Samples should be unique and not yet selected.
        """
        raise NotImplementedError()

    # -------------------------------------------------------------------------
    #  Factory Methods
    # -------------------------------------------------------------------------
    @classmethod
    def fast(cls) -> Self:
        """Create a InitFast initialization strategy."""
        from ._init_fast import InitFast

        return InitFast()

    @classmethod
    def random_one_shot(cls, uniform: bool = False, ignore_constraints: bool = False) -> Self:
        """Create a InitRandomOneShot initialization strategy."""
        from ._init_random_one_shot import InitRandomOneShot

        return InitRandomOneShot(
            uniform=uniform,
            ignore_constraints=ignore_constraints,
        )

    @classmethod
    def random_batched(cls, b: int, ignore_constraints: bool = False) -> Self:
        """Create a InitRandomBatched initialization strategy."""
        from ._init_random_batched import InitRandomBatched

        return InitRandomBatched(
            b=b,
            ignore_constraints=ignore_constraints,
        )

    @classmethod
    def eager(cls, nc: int, ignore_constraints: bool = False) -> Self:
        """Create a InitEager initialization strategy."""
        from ._init_eager import InitEager

        return InitEager(
            nc=nc,
            ignore_constraints=ignore_constraints,
        )
