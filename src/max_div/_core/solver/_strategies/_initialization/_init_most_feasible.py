import warnings

import numpy as np
from numpy.typing import NDArray

from max_div._core._warnings import FeasibilityWarning
from max_div._core.feasibility import FeasibilityStatus, find_feasible
from max_div._core.solver._solver_state import SolverState

from ._base import InitializationStrategy


class InitMostFeasible(InitializationStrategy):
    """Initialize from the most nearly feasible selection the feasibility pipeline can construct.

    The pipeline always hands back a selection of `k` items, and the solve always starts from it:

    - one satisfying every constraint, where such a selection was found;
    - the least-infeasible one, where infeasibility was proven;
    - the least-violating one the search reached, otherwise.

    `get_debug_info` reports which of the three it was.

    Constrained problems only.  An unconstrained problem raises, because every selection satisfies
    an empty constraint set and the pipeline would return an arbitrary one; a preset choosing this
    strategy must choose a different one when a problem carries no constraints.

    The whole selection is produced in one batch, so the state must still be empty.

    The selection is drawn by randomized rounding of the relaxation's maximally spread fractional
    optimizer, so different seeds give genuinely different near-feasible selections — parallel
    workers with distinct seeds start decorrelated.

    Suggested use: constrained problems where reaching feasibility consumes a meaningful share of
    the optimization budget, or where feasibility may be unreachable altogether.

    Time Complexity:
       - one interior-point relaxation solve (a few dozen iterations, each dominated by an
         m x m factorization plus O(nnz) array passes), plus a bounded number of rounding and
         repair rounds.
    """

    def __init__(self, max_iter: int | None = None) -> None:
        """Create the strategy.

        Args:
            max_iter: Deprecated and ignored; the relaxation is solved exactly, without an
                iteration budget.
        """
        super().__init__()
        if max_iter is not None:
            warnings.warn(
                "InitMostFeasible(max_iter=...) is deprecated and ignored: the relaxation is "
                "solved exactly by an interior-point method, which needs no iteration budget.",
                DeprecationWarning,
                stacklevel=2,
            )
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
            seed=int(self._seed),
        )
        self._status = result.status
        if not result.converged:
            # a solve() caller never sees the FeasibilityResult, so this warning is the only
            # trace that the relaxation solve behind the initialization did not converge
            warnings.warn(
                "The feasibility relaxation behind most_feasible() did not converge; the "
                "initialization is still valid, but its selection may be further from feasible "
                "than a converged solve would give.",
                FeasibilityWarning,
                stacklevel=2,
            )
        return result.selection.astype(np.int32)

    def get_debug_info(self) -> str:
        """Return the verdict the pipeline reached, or "/" before it has run."""
        return "/" if self._status is None else self._status.name
