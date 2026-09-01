from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from max_div._core.solver._strategies._base import StrategyBase

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray

    from max_div._core.solver._solver_state import SolverState

    from ._init_eager import InitEager
    from ._init_farthest_point import InitFarthestPoint
    from ._init_farthest_point_batched import InitFarthestPointBatched
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

    def __init__(self, name: str | None = None, parallel_batch_add: bool = False) -> None:
        """Initialize the strategy.

        Args:
            name: optional name of the strategy; if omitted the class name is used.
            parallel_batch_add: whether this strategy's batched adds may update trackers over
                parallel threads; see the tracker base class for the contract.
        """
        super().__init__(name)
        self._parallel_batch_add = parallel_batch_add

    @abstractmethod
    def get_next_samples(self, state: SolverState, k_remaining: int | np.int32) -> NDArray[np.int32]:
        """Return next batch of samples to be added to the initial selection.

        This method is called repeatedly by the Solver, until enough samples have been selected to
        reach the desired selection size.

        Args:
            state: (SolverState) The current solver state, to fetch problem size, constraints, distances, etc...,
                so initial selection can be made in an informed way.
            k_remaining: (int) number of samples that remain to be selected.

        Returns:
            np.array of unique np.int32 values, shape=(b,), with indices of samples to be added to the selection.
            b can be any value in range [1, k_remaining].  Samples should be unique and not yet selected.
        """
        raise NotImplementedError

    @property
    def parallel_batch_add(self) -> bool:
        """Return whether adding this strategy's sample batches may update trackers over parallel threads.

        Results are identical either way; a strategy returns True only when explicitly configured
        to (see the tracker base class for the contract).
        """
        return self._parallel_batch_add

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

        Args:
            top_k: Each greedy pick samples uniformly among the `top_k` highest diversity
                contributions; the default 1 keeps the exact greedy construction.
        """
        from ._init_farthest_point import InitFarthestPoint

        return InitFarthestPoint(top_k=top_k)

    @classmethod
    def farthest_point_batched(cls, top_k: int = 8, batch_size: int = 256) -> InitFarthestPointBatched:
        """Create a farthest-point initialization that draws a batch of items per pass over the dataset.

        The strategy offers each draw the same candidates as `farthest_point`, so selections are of
        equal quality but not identical, and it is several times faster at large n. Separation-family
        diversity metrics only; see `InitFarthestPointBatched` for the mechanism and the parameters.
        """
        from ._init_farthest_point_batched import InitFarthestPointBatched

        return InitFarthestPointBatched(top_k=top_k, batch_size=batch_size)

    @classmethod
    def most_feasible(cls, max_iter: int | None = None) -> InitMostFeasible:
        """Initialization that constructs a selection satisfying every constraint, where it can.

        Constrained problems only; see `InitMostFeasible` for the full contract.

        Args:
            max_iter: Deprecated and ignored; the relaxation is solved exactly, without an
                iteration budget.
        """
        from ._init_most_feasible import InitMostFeasible

        return InitMostFeasible(max_iter=max_iter)

    @classmethod
    def random_one_shot(
        cls, uniform: bool = False, ignore_constraints: bool = False, parallel: bool = False
    ) -> InitRandomOneShot:
        """Random initialization that selects all ``k`` items in a single batch.

        Probabilities are biased by the global diversity contribution (unless ``uniform=True``).

        Args:
            uniform: If True, sample uniformly instead of using contribution-based probabilities.
            ignore_constraints: If True, ignore constraints during sampling.
            parallel: If True, the batched tracker update runs over parallel threads; see
                `DiversityContributionTracker.add_many` for the contract.
        """
        from ._init_random_one_shot import InitRandomOneShot

        return InitRandomOneShot(
            uniform=uniform,
            ignore_constraints=ignore_constraints,
            parallel=parallel,
        )

    @classmethod
    def random_batched(cls, b: int, ignore_constraints: bool = False) -> InitRandomBatched:
        """Random initialization that selects items in batches of ``b``.

        Diversity contributions are re-evaluated between batches.

        Args:
            b: Batch size (number of items to select per batch).
            ignore_constraints: If True, ignore constraints during sampling.
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

        Args:
            nc: Number of candidates to evaluate at each step.
            ignore_constraints: If True, ignore constraints during sampling.
        """
        from ._init_eager import InitEager

        return InitEager(
            nc=nc,
            ignore_constraints=ignore_constraints,
        )
