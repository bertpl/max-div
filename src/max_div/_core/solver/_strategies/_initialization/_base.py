from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from max_div._core.solver._strategies._base import StrategyBase

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray

    from max_div._core.solver._solver_state import SolverState

    from ._init_eager import InitEager
    from ._init_fast import InitFast
    from ._init_random_batched import InitRandomBatched
    from ._init_random_one_shot import InitRandomOneShot


# =================================================================================================
#  InitializationStrategy
# =================================================================================================
class InitializationStrategy(StrategyBase, ABC):
    """Base class for strategies that produce an initial selection of ``k`` vectors.

    Use the factory methods (`fast`, `random_one_shot`, `random_batched`,
    `eager`) to create instances.
    """

    @abstractmethod
    def get_next_samples(self, state: SolverState, k_remaining: int | np.int32) -> NDArray[np.int32]:
        """Return next batch of samples to be added to the initial selection.

        This method is called repeatedly by the Solver, until enough samples have been selected to
        reach the desired selection size.

        :param state: (SolverState) The current solver state, to fetch problem size, constraints, distances, etc...,
                                    so initial selection can be made in an informed way.
        :param k_remaining: (int) number of samples that remain to be selected.
        :return: np.array of unique np.int32 values, shape=(b,), with indices of samples to be added to the selection.
                  b can be any value in range [1, k_remaining].  Samples should be unique and not yet selected.
        """
        raise NotImplementedError

    # -------------------------------------------------------------------------
    #  Factory Methods
    # -------------------------------------------------------------------------
    @classmethod
    def fast(cls) -> InitFast:
        """Deterministic greedy initialization.

        Selects vectors one-by-one, always picking the one that maximizes the diversity signal wrt the
        current selection.
        Fast but seed-independent.
        """
        from ._init_fast import InitFast

        return InitFast()

    @classmethod
    def random_one_shot(cls, uniform: bool = False, ignore_constraints: bool = False) -> InitRandomOneShot:
        """Random initialization that selects all ``k`` vectors in a single batch.

        Probabilities are biased by the global diversity signal (unless ``uniform=True``).

        :param uniform: If True, sample uniformly instead of using signal-based probabilities.
        :param ignore_constraints: If True, ignore constraints during sampling.
        """
        from ._init_random_one_shot import InitRandomOneShot

        return InitRandomOneShot(
            uniform=uniform,
            ignore_constraints=ignore_constraints,
        )

    @classmethod
    def random_batched(cls, b: int, ignore_constraints: bool = False) -> InitRandomBatched:
        """Random initialization that selects vectors in batches of ``b``.

        Diversity signals are re-evaluated between batches.

        :param b: Batch size (number of vectors to select per batch).
        :param ignore_constraints: If True, ignore constraints during sampling.
        """
        from ._init_random_batched import InitRandomBatched

        return InitRandomBatched(
            b=b,
            ignore_constraints=ignore_constraints,
        )

    @classmethod
    def eager(cls, nc: int, ignore_constraints: bool = False) -> InitEager:
        """Greedy initialization that evaluates ``nc`` random candidates per step and picks the best one.

        Higher ``nc`` gives better quality but is slower.

        :param nc: Number of candidates to evaluate at each step.
        :param ignore_constraints: If True, ignore constraints during sampling.
        """
        from ._init_eager import InitEager

        return InitEager(
            nc=nc,
            ignore_constraints=ignore_constraints,
        )
