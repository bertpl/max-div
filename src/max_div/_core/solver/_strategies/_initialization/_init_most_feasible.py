import numpy as np
from numpy.typing import NDArray

from max_div._core.constraints.feasibility import CONSTRUCTION_DEFAULT_ITER, FeasibilityStatus, find_feasible
from max_div._core.solver._solver_state import SolverState

from ._base import InitializationStrategy
from ._init_random_one_shot import InitRandomOneShot


class InitMostFeasible(InitializationStrategy):
    """Initialize from a constructed feasible selection, or the least infeasible one available.

    The Lagrangian feasibility pipeline runs in construction mode, and its verdict decides:

    - a feasible selection was constructed -> the solve starts from it;
    - infeasibility was proven -> the least-infeasible selection found becomes the start;
    - neither -> `fallback` initializes, exactly as it would have on its own.

    Unconstrained problems bypass the pipeline too: with nothing to satisfy, it has nothing to
    contribute.

    Feasibility is all this strategy optimizes for, so its selection is no more diverse than a
    random one.  `beta` trades some of the chance of reaching feasibility for a more diverse
    result, by scoring candidates on the state's global diversity contributions.

    Suggested use: constrained problems where reaching feasibility eats into the optimization
    budget, or where it may be unreachable altogether.

    Time Complexity:
       - ~O(max_iter * (n log k + total constraint membership)) for the ascent, plus a bounded
         number of repair rounds.
    """

    def __init__(
        self,
        max_iter: int = CONSTRUCTION_DEFAULT_ITER,
        beta: float = 0.0,
        fallback: InitializationStrategy | None = None,
    ) -> None:
        """Create the strategy.

        :raises ValueError: If `max_iter` is below 1, or `beta` is negative.
        """
        super().__init__()
        if max_iter < 1:
            raise ValueError(f"max_iter must be >= 1, got {max_iter}")
        if beta < 0:
            raise ValueError(f"beta must be >= 0, got {beta}")
        self._max_iter = max_iter
        self._beta = beta
        self._fallback = fallback if fallback is not None else InitRandomOneShot()
        self._status: FeasibilityStatus | None = None

    def set_seed(self, seed: int | np.int64) -> None:
        """Seed this strategy and the fallback it delegates to, so one seed drives both paths."""
        super().set_seed(seed)
        self._fallback.set_seed(seed)

    def get_next_samples(self, state: SolverState, k_remaining: int | np.int32) -> NDArray[np.int32]:
        if state.n_selected != 0:
            # The pipeline decides over the whole selection at once, and reads the constraint
            # bounds as problem-level values -- which they only are while nothing is selected.
            raise RuntimeError("InitMostFeasible produces the full selection in one batch, so it needs an empty state.")
        if not state.has_constraints:
            return self._fallback.get_next_samples(state, k_remaining)

        result = find_feasible(
            con_values=state.con_values,
            con_indices=state.con_indices,
            con_weights=state.con_weights,
            n=int(state.n),
            k=int(state.k),
            max_iter=self._max_iter,
            seed=int(self._seed),
            diversity_prior=(state.global_contribution_array if self._beta != 0.0 else None),
            beta=self._beta,
        )
        self._status = result.status
        if result.status is FeasibilityStatus.UNKNOWN:
            return self._fallback.get_next_samples(state, k_remaining)
        return result.selection.astype(np.int32)

    def get_debug_info(self) -> str:
        """Return the verdict the pipeline reached, or "/" before it has run."""
        return "/" if self._status is None else self._status.name
