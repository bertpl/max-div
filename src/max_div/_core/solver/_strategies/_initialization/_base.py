from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from max_div._core.constraints.feasibility import CONSTRUCTION_DEFAULT_ITER
from max_div._core.solver._strategies._base import StrategyBase

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray

    from max_div._core.solver._solver_state import SolverState

    from ._init_eager import InitEager
    from ._init_farthest_point import InitFarthestPoint
    from ._init_fast import InitFast
    from ._init_most_feasible import InitMostFeasible
    from ._init_random_batched import InitRandomBatched
    from ._init_random_one_shot import InitRandomOneShot


# =================================================================================================
#  InitializationStrategy
# =================================================================================================
class InitializationStrategy(StrategyBase, ABC):
    """Base class for strategies that produce an initial selection of ``k`` items.

    Use the factory methods below to create instances.
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
        """Trivial initialization that selects the first ``k`` items (indices 0 to k-1).

        Deterministic and effectively free; mainly a baseline for testing and benchmarking.
        """
        from ._init_fast import InitFast

        return InitFast()

    @classmethod
    def farthest_point(cls, top_k: int = 1) -> InitFarthestPoint:
        """Farthest-point-sampling initialization: a seeded random start item, then greedy picks.

        See `InitFarthestPoint` for the per-metric interpretation and constraint handling.

        :param top_k: Each greedy pick samples uniformly among the `top_k` highest diversity
            contributions; the default 1 keeps the exact greedy construction.
        """
        from ._init_farthest_point import InitFarthestPoint

        return InitFarthestPoint(top_k=top_k)

    @classmethod
    def most_feasible(cls, max_iter: int = CONSTRUCTION_DEFAULT_ITER, beta: float = 0.0) -> InitMostFeasible:
        """Initialization that constructs a selection satisfying every constraint, where it can.

        Constrained problems only; see `InitMostFeasible` for the full contract.

        :param max_iter: Search budget; a higher budget more often finds a feasible selection, and
            lowers the violation of the one returned when none exists, at a proportional cost in
            setup time.
        :param beta: Tilts the constructed selection toward diverse items, at the risk of a less
            feasible starting point and an O(n²) diversity-contribution sweep; 0, the value to
            keep unless the trade has been measured, leaves construction purely
            feasibility-driven.
        """
        from ._init_most_feasible import InitMostFeasible

        return InitMostFeasible(max_iter=max_iter, beta=beta)

    @classmethod
    def random_one_shot(cls, uniform: bool = False, ignore_constraints: bool = False) -> InitRandomOneShot:
        """Random initialization that selects all ``k`` items in a single batch.

        Probabilities are biased by the global diversity contribution (unless ``uniform=True``).

        :param uniform: If True, sample uniformly instead of using contribution-based probabilities.
        :param ignore_constraints: If True, ignore constraints during sampling.
        """
        from ._init_random_one_shot import InitRandomOneShot

        return InitRandomOneShot(
            uniform=uniform,
            ignore_constraints=ignore_constraints,
        )

    @classmethod
    def random_batched(cls, b: int, ignore_constraints: bool = False) -> InitRandomBatched:
        """Random initialization that selects items in batches of ``b``.

        Diversity contributions are re-evaluated between batches.

        :param b: Batch size (number of items to select per batch).
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
