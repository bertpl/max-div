import numpy as np
from numpy.typing import NDArray

from max_div._core.constraints.feasibility import CONSTRUCTION_DEFAULT_ITER, FeasibilityStatus, find_feasible
from max_div._core.solver._solver_state import SolverState

from ._base import InitializationStrategy


class InitMostFeasible(InitializationStrategy):
    """Initialize from the most nearly feasible selection the Lagrangian pipeline can construct.

    The pipeline always hands back a selection of `k` items, and the solve always starts from it:
    one satisfying every constraint where such a selection was found, the least-infeasible one
    where infeasibility was proven, and otherwise the least-violating one the search reached.
    `get_debug_info` reports which of the three it was.

    Constrained problems only.  An unconstrained problem raises, because every selection satisfies
    an empty constraint set and the pipeline would return an arbitrary one; a preset choosing this
    strategy must choose a different one when a problem carries no constraints.

    The whole selection is produced in one batch, so the state must still be empty.

    Feasibility is all this strategy optimizes for, so its selection is no more diverse than a
    random one.  `beta` trades some of the chance of reaching feasibility for a more diverse
    result, by scoring candidates on the state's global diversity contributions.

    Suggested use: constrained problems where reaching feasibility consumes a meaningful share of
    the optimization budget, or where feasibility may be unreachable altogether.

    Time Complexity:
       - ~O(max_iter * (n log k + total constraint membership)) for the ascent, plus a bounded
         number of repair rounds.
       - a nonzero `beta` additionally forces the O(n²) global-contribution sweep.
    """

    def __init__(self, max_iter: int = CONSTRUCTION_DEFAULT_ITER, beta: float = 0.0) -> None:
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
        self._status: FeasibilityStatus | None = None

    def get_next_samples(self, state: SolverState, k_remaining: int | np.int32) -> NDArray[np.int32]:
        if state.n_selected != 0:
            # The pipeline decides over the whole selection at once, and reads the constraint
            # bounds as problem-level values -- which they only are while nothing is selected.
            raise RuntimeError("InitMostFeasible produces the full selection in one batch, so it needs an empty state.")
        if not state.has_constraints:
            raise ValueError(
                "InitMostFeasible only applies to constrained problems; this one has no constraints. "
                "Choose another initialization strategy for it."
            )

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
        return result.selection.astype(np.int32)

    def get_debug_info(self) -> str:
        """Return the verdict the pipeline reached, or "/" before it has run."""
        return "/" if self._status is None else self._status.name
