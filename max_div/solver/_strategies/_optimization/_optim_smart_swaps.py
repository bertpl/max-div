import math

import numpy as np
from numpy.typing import NDArray

from max_div.solver._parameters import sampled_interval, sampled_poisson
from max_div.solver._solver_state import SolverState
from max_div.solver._strategies._sampling import (
    SamplingType,
    select_items_to_add,
    select_items_to_remove,
)

from ._base import SwapBasedOptimizationStrategy


class OptimSmartSwaps(SwapBasedOptimizationStrategy):
    """
    This swap-based optimization strategy can be seen as an evolution of OptimGuidedSwaps...
       - allowing adaptively sampling most parameters
       - perform removal and addition of samples by chosen best of 'nc' candidates
    """

    def __init__(
        self,
        swap_size_max: int,
        nc_remove_max: int,
        nc_add_max: int,
        tau_learn: float = 100.0,
        ignore_infeasible_diversity_up_to_fraction: float = -1.0,
        cost_awareness: float = 0.0,
    ):
        """
        Optimization-based strategy that performs 'smart' swaps, that combine
        adaptive parameter sampling, selectivity-modification and best-of-nc selection
        of removed and added samples.

        :param swap_size_max: (int) maximum swap sizes to be adaptively sampled from.
        :param nc_remove_max: (int) maximum 'nc'-value to be adaptively sampled from for REMOVING.
        :param nc_add_max: (int) maximum 'nc'-value to be adaptively sampled from for ADDING.
        :param tau_learn: (float) learning time constant for adaptive parameter sampling.
        :param ignore_infeasible_diversity_up_to_fraction: (float) fraction of total diversity score
        :param cost_awareness: (float >= 0) value to indicate how much larger values of n_swap, nc_remove, nc_add
                               should be avoided, since they incur additional computational cost.  Non-0 values
                               make it such that larger values are only used if they provide sufficient benefit.
        """

        # --- swap_size ---
        _swap_size = sampled_poisson(
            min_value=1,
            max_value=swap_size_max,
            lambda_prior=1.0,
            tau_learn=tau_learn,
            tau_forget=math.inf,
            large_value_penalty_exponent=cost_awareness,
        )

        # --- nc_remove ---
        _nc_remove = sampled_poisson(
            min_value=1,
            max_value=nc_remove_max,
            lambda_prior=1.0,
            tau_learn=tau_learn,
            tau_forget=math.inf,
            large_value_penalty_exponent=cost_awareness,
        )

        # --- nc_add ---
        _nc_add = sampled_poisson(
            min_value=1,
            max_value=nc_add_max,
            lambda_prior=1.0,
            tau_learn=tau_learn,
            tau_forget=math.inf,
            large_value_penalty_exponent=cost_awareness,
        )

        # --- selectivity modifiers ---
        _selectivity_modifier_remove = sampled_interval(
            min_value=-0.99,
            max_value=0.99,
            median_prior=0.0,
            tau_learn=tau_learn,
            tau_forget=math.inf,
        )
        _selectivity_modifier_add = sampled_interval(
            min_value=-0.99,
            max_value=0.99,
            median_prior=0.0,
            tau_learn=tau_learn,
            tau_forget=math.inf,
        )

        # --- name ----------------------------------------
        name_swap_size = f"1-{swap_size_max}" if swap_size_max > 1 else "1"
        name = f"OptimSmartSwaps({name_swap_size},{nc_remove_max},{nc_add_max})"

        # --- superclass constructor ----------------------
        super().__init__(
            name=name,
            constraint_softness=0.0,
            dynamic_params=dict(
                swap_size=_swap_size,
                nc_remove=_nc_remove,
                nc_add=_nc_add,
                selectivity_modifier_remove=_selectivity_modifier_remove,
                selectivity_modifier_add=_selectivity_modifier_add,
            ),
            ignore_infeasible_diversity_up_to_fraction=ignore_infeasible_diversity_up_to_fraction,
        )

        # --- set (initial) parameter values --------------
        self.swap_size = 1
        self.nc_remove = 1
        self.nc_add = 1
        self.selectivity_modifier_remove = self.initial_param_value(_selectivity_modifier_remove)
        self.selectivity_modifier_add = self.initial_param_value(_selectivity_modifier_add)

    # -------------------------------------------------------------------------
    #  Implementation
    # -------------------------------------------------------------------------
    def _determine_swap_size(self) -> np.int32:
        return np.int32(self.swap_size)

    def _remove_samples(
        self,
        state: SolverState,
        n_to_remove: np.int32,
    ) -> NDArray[np.int32]:
        """
        REMOVE n samples and return the indices of removed samples.
        """
        # NOTE: we override parent class method, since in this implementation, the _samples_to_be_removed method
        #       already actually removes the samples.
        return self._samples_to_be_removed(state, n_to_remove)

    def _add_samples(
        self,
        state: SolverState,
        n_to_add: np.int32,
        candidate_samples: NDArray[np.int32],
    ) -> NDArray[np.int32]:
        """
        ADD n samples and return the indices of removed samples.
        """
        # NOTE: we override parent class method, since in this implementation, the _samples_to_be_added method
        #       already actually adds the samples.
        return self._samples_to_be_added(state, n_to_add, candidate_samples)

    # -------------------------------------------------------------------------
    #  Samples to be removed
    # -------------------------------------------------------------------------
    def _samples_to_be_removed(self, state: SolverState, n_to_remove: np.int32) -> NDArray[np.int32]:
        removed_samples = np.empty(n_to_remove, dtype=np.int32)

        for i in range(n_to_remove):
            # -----------------------------------
            # we will repeat 'n_to_remove' times:
            #  1) select 'nc' candidates for removal using select_items_to_remove()
            #  2) select best of these 'nc' candidates to be removed
            #  3) actually remove this candidate
            # -----------------------------------

            # 1) select 'nc' and 'nc' candidates
            nc = min(self.nc_remove, state.n_selected)  # can't select nc if nc > n_selected
            candidates_for_removal = select_items_to_remove(
                state=state,
                k=np.int32(nc),
                selectivity_modifier=self.selectivity_modifier_remove,
                rng_state=self._rng_state,
            )

            # 2) select best candidate
            best_sample: np.int32 = np.int32(-1)
            best_score_tuple: tuple | None = None
            for i_cand in candidates_for_removal:
                # temporarily remove candidate from selection
                state.remove(i_cand)

                # compute new score
                cand_score = state.score

                # if best score so far, remember candidate
                cand_score_tuple = cand_score.as_tuple(
                    soft=self.constraint_softness,
                    ignore_infeasible_diversity=self.ignore_infeasible_diversity,
                )
                if (best_score_tuple is None) or (cand_score_tuple >= best_score_tuple):
                    # NOTE: also accept equal candidates, to encourage diversity in selections and hence exploration
                    best_score_tuple = cand_score_tuple
                    best_sample = i_cand

                # re-add candidate to selection
                state.add(i_cand)

            # 3) now actually remove best sample from selection
            state.remove(best_sample)
            removed_samples[i] = best_sample

        return removed_samples

    # -------------------------------------------------------------------------
    #  Samples to be added
    # -------------------------------------------------------------------------
    def _samples_to_be_added(
        self,
        state: SolverState,
        n_to_add: np.int32,
        candidate_samples: NDArray[np.int32],
    ) -> NDArray[np.int32]:
        # -----------------------------------
        # a) we will repeat 'nc' times:
        #    1) select a candidate group of 'n_to_add' samples to add
        #    2) evaluate final score of adding all candidates
        # b) effectively add the best such group of samples
        # -----------------------------------
        best_samples: NDArray[np.int32] = np.empty(n_to_add, dtype=np.int32)
        best_score_tuple: tuple | None = None

        # a) repeat 'nc' times...
        for i in range(self.nc_add):
            # 1) select group of 'n_to_add' samples
            candidate_samples_to_add = select_items_to_add(
                state=state,
                candidates=candidate_samples,
                k=n_to_add,
                selectivity_modifier=self.selectivity_modifier_add,
                rng_state=self._rng_state,
                sampling_type=SamplingType.GROUP,
                include_within_group_separation=(n_to_add > 1),
                ignore_constraints=False,
            )

            # 2) evaluate this candidate group of samples
            #  --> temporarily add all samples
            state.add_many(candidate_samples_to_add)

            #  --> compute new score & update best score/samples
            cand_score_tuple = state.score.as_tuple(
                soft=self.constraint_softness,
                ignore_infeasible_diversity=self.ignore_infeasible_diversity,
            )
            if (best_score_tuple is None) or (cand_score_tuple >= best_score_tuple):
                # NOTE: also accept equal candidates, to encourage diversity in selections and hence exploration
                best_score_tuple = cand_score_tuple
                best_samples = candidate_samples_to_add

            #  --> re-remove all samples
            state.remove_many(candidate_samples_to_add)

        # b) now actually add best sample set to selection
        state.add_many(best_samples)

        # return best samples
        return best_samples

    # -------------------------------------------------------------------------
    #  Debug info
    # -------------------------------------------------------------------------
    def get_debug_info(self) -> str:
        # --- collect info ---
        swap_size = self.get_expected_param_value("swap_size")
        nc_remove = self.get_expected_param_value("nc_remove")
        nc_add = self.get_expected_param_value("nc_add")
        selectivity_modifier_remove = self.get_expected_param_value("selectivity_modifier_remove")
        selectivity_modifier_add = self.get_expected_param_value("selectivity_modifier_add")

        # --- build string ---
        debug_info = super().get_debug_info().strip()
        debug_info += (
            f" | λ_swap={swap_size:4.1f}"
            f" | λ_rem={nc_remove:4.1f}"
            f" | λ_add={nc_add:4.1f}"
            f" | sel_rem={selectivity_modifier_remove:5.2f}"
            f" | sel_add={selectivity_modifier_add:5.2f}"
        )
        return debug_info.ljust(100)

    def get_expected_param_value(self, param_name: str) -> float:
        return self.sampled_params[param_name].summary_statistic()
