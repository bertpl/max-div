from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from max_div._core.solver._strategies._base import StrategyBase

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray

    from max_div._core.solver._solver_state import SolverState

    from ._init_farthest_point import InitFarthestPoint
    from ._init_most_feasible import InitMostFeasible
    from ._init_random_selection import InitRandomSelection


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
            The solver adds the whole batch in a single tracker update.
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
    def farthest_point(cls, top_k: int = 8, batch_size: int | None = 256) -> InitFarthestPoint:
        """Create a farthest-point-sampling initialization: a seeded random start item, then greedy picks.

        Where every term of the solve's primary diversity objective is a separation-family metric over
        one distance and `batch_size` is not None, the picks are drawn in rounds of up to `batch_size`
        items per pass over the dataset, which is several times faster at large n; otherwise the
        strategy picks 1 item per pass.

        Constraints are ignored; feasibility is left to the optimization steps. See `InitFarthestPoint`
        for the per-metric interpretation.

        Args:
            top_k: Each greedy pick samples uniformly among the `top_k` highest diversity
                contributions; 1 is the exact greedy construction.
            batch_size: How many candidates a round collects; it changes only the time spent, not the
                selection's quality. `None` picks one item at a time.

        Raises:
            ValueError: If `top_k` is below 1, or `batch_size` is below `top_k`.
        """
        from ._init_farthest_point import InitFarthestPoint

        return InitFarthestPoint(top_k=top_k, batch_size=batch_size)

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
    def random_selection(cls, ignore_constraints: bool = False, parallel: bool = False) -> InitRandomSelection:
        """Create a random initialization that selects all ``k`` items in a single draw.

        Args:
            ignore_constraints: If True, sample uniformly at random even when the problem has constraints; if
                False, steer the draw so the selection satisfies them.
            parallel: If True, the batched tracker update runs over parallel threads; see
                `DiversityContributionTracker.add_many` for the contract.
        """
        from ._init_random_selection import InitRandomSelection

        return InitRandomSelection(ignore_constraints=ignore_constraints, parallel=parallel)
