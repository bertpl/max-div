from __future__ import annotations

from enum import StrEnum
from typing import TYPE_CHECKING, Any

from max_div._core.solver._parameters import ease_in_out, linear
from max_div._core.solver._strategies._optimization._optim_guided_swaps import OptimGuidedSwaps
from max_div._core.solver._strategies._optimization._optim_random_swaps import OptimRandomSwaps
from max_div._core.solver._strategies._optimization._optim_smart_swaps import OptimSmartSwaps

if TYPE_CHECKING:
    from max_div._core.solver._strategies import OptimizationStrategy


# =================================================================================================
#  Enum
# =================================================================================================
class OptimPreset(StrEnum):
    """StrEnum for all optimization presets we want to benchmark."""

    # --- random swaps ------
    RS = "RS"

    # --- guided swaps ------
    GS_1 = "GS(1)"
    GS_2 = "GS(2)"
    GS_3 = "GS(3)"
    GS_1_3 = "GS(1-3)"
    GS_1_3_SOFT = "GS(1-3,soft)"
    GS_1_3_WIDE = "GS(1-3,wide)"
    GS_1_3_NARROW = "GS(1-3,narrow)"
    GS_1_3_WI_NA = "GS(1-3,wi->na)"
    GS_1_3_NA_WI = "GS(1-3,na->wi)"
    SM_2 = "SM(2)"
    SM_4 = "SM(4)"
    SM_8 = "SM(8)"

    # -------------------------------------------------------------------------
    #  Factory
    # -------------------------------------------------------------------------
    def create(self) -> OptimizationStrategy:
        """Create an InitializationStrategy instance corresponding to this preset."""
        cls, kwargs = _OPTIM_CLASSES_AND_KWARGS[self]
        return cls(**kwargs)

    # -------------------------------------------------------------------------
    #  Meta-Data
    # -------------------------------------------------------------------------
    def is_constraint_aware(self) -> bool:
        return self != OptimPreset.RS

    def is_relevant_for_problem(self, problem_has_constraints: bool) -> bool:
        return True

    def class_name(self) -> str:
        cls, _ = _OPTIM_CLASSES_AND_KWARGS[self]
        return cls.__name__

    def class_kwargs(self) -> dict[str, Any]:
        _, kwargs = _OPTIM_CLASSES_AND_KWARGS[self]
        return kwargs

    @classmethod
    def all(cls) -> list[OptimPreset]:
        """Get a list of all OptimPreset members."""
        return list(cls)


# =================================================================================================
#  Classes & Arguments
# =================================================================================================
_OPTIM_CLASSES_AND_KWARGS: dict[OptimPreset, tuple[type[OptimizationStrategy], dict[str, Any]]] = {
    OptimPreset.RS: (OptimRandomSwaps, {}),
    OptimPreset.GS_1: (OptimGuidedSwaps, {"min_swap_size": 1, "max_swap_size": 1}),
    OptimPreset.GS_2: (OptimGuidedSwaps, {"min_swap_size": 2, "max_swap_size": 2}),
    OptimPreset.GS_3: (OptimGuidedSwaps, {"min_swap_size": 3, "max_swap_size": 3}),
    OptimPreset.GS_1_3: (OptimGuidedSwaps, {"min_swap_size": 1, "max_swap_size": 3}),
    OptimPreset.GS_1_3_SOFT: (
        OptimGuidedSwaps,
        {
            "min_swap_size": 1,
            "max_swap_size": 3,
            "constraint_softness": ease_in_out(1.0, 0.0),
            "p_add_constraint_aware": ease_in_out(0.0, 1.0),
        },
    ),
    OptimPreset.GS_1_3_WIDE: (
        OptimGuidedSwaps,
        {
            "min_swap_size": 1,
            "max_swap_size": 3,
            "remove_selectivity_modifier": -0.8,
            "add_selectivity_modifier": -0.8,
        },
    ),
    OptimPreset.GS_1_3_NARROW: (
        OptimGuidedSwaps,
        {
            "min_swap_size": 1,
            "max_swap_size": 3,
            "remove_selectivity_modifier": +0.8,
            "add_selectivity_modifier": +0.8,
        },
    ),
    OptimPreset.GS_1_3_WI_NA: (
        OptimGuidedSwaps,
        {
            "min_swap_size": 1,
            "max_swap_size": 3,
            "remove_selectivity_modifier": linear(-0.8, +0.8),
            "add_selectivity_modifier": linear(-0.8, +0.8),
        },
    ),
    OptimPreset.GS_1_3_NA_WI: (
        OptimGuidedSwaps,
        {
            "min_swap_size": 1,
            "max_swap_size": 3,
            "remove_selectivity_modifier": linear(+0.8, -0.8),
            "add_selectivity_modifier": linear(+0.8, -0.8),
        },
    ),
    OptimPreset.SM_2: (
        OptimSmartSwaps,
        {
            "swap_size_max": 2,
            "nc_remove_max": 2,
            "nc_add_max": 2,
            "tau_learn": 10,
            "ignore_infeasible_diversity_up_to_fraction": 0.8,
            "cost_awareness": 0.5,
        },
    ),
    OptimPreset.SM_4: (
        OptimSmartSwaps,
        {
            "swap_size_max": 4,
            "nc_remove_max": 4,
            "nc_add_max": 4,
            "tau_learn": 10,
            "ignore_infeasible_diversity_up_to_fraction": 0.8,
            "cost_awareness": 0.5,
        },
    ),
    OptimPreset.SM_8: (
        OptimSmartSwaps,
        {
            "swap_size_max": 8,
            "nc_remove_max": 8,
            "nc_add_max": 8,
            "tau_learn": 10,
            "ignore_infeasible_diversity_up_to_fraction": 0.8,
            "cost_awareness": 0.5,
        },
    ),
}
