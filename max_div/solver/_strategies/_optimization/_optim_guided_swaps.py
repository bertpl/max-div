import numpy as np
from numpy.typing import NDArray

from max_div.internal.math.modify_p_selectivity import exponential_selectivity
from max_div.internal.math.random import rand_float32, set_seed
from max_div.sampling import randint_numba
from max_div.sampling.con import randint_constrained
from max_div.sampling.poisson import sample_truncated_poisson
from max_div.solver._parameters import ParameterSchedule
from max_div.solver._solver_state import SolverState

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
        n = sample_truncated_poisson(
            self.min_swap_size,
            self.max_swap_size,
            np.float32(self.swap_size_lambda),
            seed=self.next_seed(),
        )
        return n

    def _samples_to_be_removed(self, state: SolverState, n_to_remove: np.int32) -> NDArray[np.int32]:
        # --- guiding probabilities for removal ---
        p = state.selected_separation_array  # this creates a copy
        exponential_selectivity(
            p_in=p,
            p_out=p,  # in-place
            modifier=np.float32(self.remove_selectivity_modifier),
            descending=True,  # for removal, we want to have vectors with small separation have higher probability
        )

        # --- sample ---
        i_to_remove = randint_numba(
            n=np.int32(p.shape[0]),
            k=n_to_remove,
            replace=False,
            p=p,
            seed=self.next_seed(),
        )  # these are indices into selected_index_array

        # --- return vectors to be removed ---
        return state.selected_index_array[i_to_remove]

    def _samples_to_be_added(
        self, state: SolverState, n_to_add: np.int32, samples_just_removed: NDArray[np.int32]
    ) -> NDArray[np.int32]:
        # --- constraint-aware or not? ---
        if state.has_constraints:
            r = rand_float32(rng_state=set_seed(self.next_seed()))  # random float in [0.0, 1.0)
            constraint_aware = r < self.p_add_constraint_aware
        else:
            constraint_aware = False

        # --- construct guiding probabilities for addition ---
        if n_to_add > 1:
            # also take into account intra-batch separation (inferred from global separation)
            p = state.global_separation_array[state.not_selected_index_array]  # this creates a new array
            p += state.not_selected_separation_array  # so we can add in-place
        else:
            # only look at separation of not-selected w.r.t. selected
            p = state.not_selected_separation_array  # this creates a copy

        exponential_selectivity(
            p_in=p,
            p_out=p,  # in-place
            modifier=np.float32(self.add_selectivity_modifier),
            descending=False,  # for adding, we want to have vectors with high separation have higher probability
        )

        # --- sample ---
        if constraint_aware:
            p_full = np.zeros(state.n, dtype=np.float32)
            p_full[state.not_selected_index_array] = p  # make p_full of size n & use i_forbidden to exclude selected

            return randint_constrained(
                n=state.n,
                k=n_to_add,
                con_values=state.con_values,
                con_indices=state.con_indices,
                p=p_full,
                seed=self.next_seed(),
                eager=False,
                i_forbidden=state.selected_index_array,
            )  # these are indices in [0, n) as needed

        else:
            i_samples = randint_numba(
                n=np.int32(p.size),
                k=n_to_add,
                replace=False,
                p=p,
                seed=self.next_seed(),
            )  # these are indices into not_selected_index_array
            return state.not_selected_index_array[i_samples]

    # -------------------------------------------------------------------------
    #  Debug info
    # -------------------------------------------------------------------------
    def get_debug_info(self) -> str:
        debug_info = (
            f"λ={self.swap_size_lambda:5.2f}"
            f" | sel_rem={self.remove_selectivity_modifier:5.2f}"
            f" | sel_add={self.add_selectivity_modifier:5.2f}"
            f" | p_con={self.p_add_constraint_aware:5.2f}"
            f" | soft={self.constraint_softness:5.2f}"
        )
        return debug_info.ljust(100)
