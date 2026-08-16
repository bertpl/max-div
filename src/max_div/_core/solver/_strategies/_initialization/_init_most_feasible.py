import numpy as np
from numpy.typing import NDArray

from max_div._core.constraints.feasibility import CONSTRUCTION_DEFAULT_ITER, FeasibilityStatus, find_feasible
from max_div._core.solver._solver_state import SolverState

from ._base import InitializationStrategy
from ._init_random_one_shot import InitRandomOneShot


class InitMostFeasible(InitializationStrategy):
    """Initialize from a constructed feasible selection, or from the least infeasible one available.

    The strategy runs the Lagrangian feasibility pipeline in construction mode and dispatches on
    its verdict:

    - a witness was found -> the solve starts feasible, so the optimization steps can spend their
      whole budget on diversity instead of first searching for feasibility;
    - infeasibility was proven -> the least-infeasible selection constructed alongside the proof
      becomes the starting point, sparing the solver a search for a selection that cannot exist;
    - UNKNOWN -> nothing was learned, so `fallback` initializes exactly as it would have on its own.

    Unconstrained problems bypass the pipeline for the same reason: with no constraint to satisfy
    every selection is a witness, and the pipeline would hand back an arbitrary top-k of an
    all-zero score vector.  The fallback initializes those instead.

    Feasibility is the only thing this strategy optimizes for, so the selection it returns is
    otherwise unremarkable in diversity terms; `beta` trades some of the witness rate for a more
    diverse witness.

    Suggested use: constrained problems where reaching feasibility consumes a meaningful share of
    the optimization budget, or where feasibility may be unreachable altogether.

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

        Args:
            max_iter: ascent iteration budget handed to the pipeline.  Higher budgets mature the
                prices further, which raises both the witness rate and the certified violation
                floor, at a proportional cost in setup time.
            beta: how strongly the constructed selection is tilted toward diverse items, scoring
                candidates by `beta * log(p)` on the state's global diversity contributions.  The
                default 0 leaves construction purely feasibility-driven.  The tilt cannot affect
                the verdict, only which selection is constructed.
            fallback: the strategy that initializes when this one has nothing to offer — an
                UNKNOWN verdict, or a problem with no constraints.  Defaults to `InitRandomOneShot`.

        Raises:
            ValueError: If `max_iter` is below 1, or `beta` is negative.
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
