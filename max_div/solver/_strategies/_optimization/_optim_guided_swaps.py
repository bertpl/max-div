import numpy as np
from numpy.typing import NDArray

from max_div.random.distributions import sample_truncated_poisson
from max_div.random.rng import rand_float32
from max_div.solver._parameters import ParameterSchedule
from max_div.solver._solver_state import SolverState
from max_div.solver._strategies._sampling import SamplingType, select_items_to_add, select_items_to_remove

from ._base import SwapBasedOptimizationStrategy


class OptimGuidedSwaps(SwapBasedOptimizationStrategy):
    """
    This swap-based optimization strategy allows...
       - performing 1- or multiple-element swaps per iteration
       - uses guiding heuristics to preferentially select appropriate samples for removal and addition
       - allows choosing samples to be added in constraint-aware or un-aware manner, with configurable probabilities
    """

    # -------------------------------------------------------------------------
    #  Constructor
    # -------------------------------------------------------------------------
    def __init__(
        self,
        min_swap_size: int = 1,
        max_swap_size: int = 1,
        swap_size_lambda: float | ParameterSchedule = 1.0,
        constraint_softness: float | ParameterSchedule = 0.0,
        p_add_constraint_aware: float | ParameterSchedule = 1.0,
        remove_selectivity_modifier: float | ParameterSchedule = 0.0,
        add_selectivity_modifier: float | ParameterSchedule = 0.0,
    ):
        name = f"OptimGuidedSwaps({min_swap_size}" + (f"-{max_swap_size})" if max_swap_size > min_swap_size else ")")
        super().__init__(
            name=name,
            constraint_softness=constraint_softness,
            dynamic_params=dict(
                swap_size_lambda=swap_size_lambda,
                p_add_constraint_aware=p_add_constraint_aware,
                remove_selectivity_modifier=remove_selectivity_modifier,
                add_selectivity_modifier=add_selectivity_modifier,
            ),
        )
        self.min_swap_size: np.int32 = np.int32(min_swap_size)
        self.max_swap_size: np.int32 = np.int32(max_swap_size)
        self.swap_size_lambda: float = self.initial_param_value(swap_size_lambda)
        self.p_add_constraint_aware: float = self.initial_param_value(p_add_constraint_aware)
        self.remove_selectivity_modifier: float = self.initial_param_value(remove_selectivity_modifier)
        self.add_selectivity_modifier: float = self.initial_param_value(add_selectivity_modifier)

    # -------------------------------------------------------------------------
    #  Implementation
    # -------------------------------------------------------------------------
    def _determine_swap_size(self) -> np.int32:
        return sample_truncated_poisson(
            self.min_swap_size,
            self.max_swap_size,
            np.float32(self.swap_size_lambda),
            self._rng_state,
        )

    def _samples_to_be_removed(self, state: SolverState, n_to_remove: np.int32) -> NDArray[np.int32]:
        return select_items_to_remove(
            state=state,
            k=n_to_remove,
            selectivity_modifier=self.remove_selectivity_modifier,
            rng_state=self._rng_state,
        )

    def _samples_to_be_added(
        self, state: SolverState, n_to_add: np.int32, candidate_samples: NDArray[np.int32]
    ) -> NDArray[np.int32]:
        # --- ignore constraints or not? ---
        r = rand_float32(self._rng_state)  # random float in [0.0, 1.0)
        ignore_constraints = r > self.p_add_constraint_aware

        # --- select items to add ---
        return select_items_to_add(
            state=state,
            candidates=candidate_samples,
            k=n_to_add,
            selectivity_modifier=self.add_selectivity_modifier,
            rng_state=self._rng_state,
            sampling_type=SamplingType.GROUP,
            include_within_group_separation=(n_to_add > 1),
            ignore_constraints=ignore_constraints,
        )

    # -------------------------------------------------------------------------
    #  Debug info
    # -------------------------------------------------------------------------
    def get_debug_info(self) -> str:
        debug_info = super().get_debug_info().strip()
        debug_info += (
            f" | λ_swap={self.swap_size_lambda:5.2f}"
            f" | sel_rem={self.remove_selectivity_modifier:5.2f}"
            f" | sel_add={self.add_selectivity_modifier:5.2f}"
            f" | p_con={self.p_add_constraint_aware:5.2f}"
            f" | soft={self.constraint_softness:5.2f}"
        )
        return debug_info.ljust(100)
