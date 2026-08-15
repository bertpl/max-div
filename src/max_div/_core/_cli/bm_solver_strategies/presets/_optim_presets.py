from __future__ import annotations

from enum import StrEnum
from typing import TYPE_CHECKING, Any

from max_div._core.solver._parameters import ease_in, ease_out
from max_div._core.solver._strategies._optimization._optim_guided_swaps import OptimGuidedSwaps
from max_div._core.solver._strategies._optimization._optim_random_swaps import OptimRandomSwaps
from max_div._core.solver._strategies._optimization._optim_smart_swaps import OptimSmartSwaps

if TYPE_CHECKING:
    from max_div._core.solver._strategies import OptimizationStrategy


# =================================================================================================
#  Enum
# =================================================================================================
class OptimPreset(StrEnum):
    """StrEnum for all optimization presets we want to benchmark.

    Each member is a user-facing choice: `RS` as the baseline, then the guided-swap and smart-swap
    size sweeps; shipped-preset correspondences live in `_PRESET_NOTES`.
    """

    # --- random swaps ------
    RS = "RS"

    # --- guided swaps ------
    GS_1 = "GS(1)"
    GS_3 = "GS(3)"
    GS_1_3 = "GS(1-3)"
    GS_GUIDED = "GS(guided)"

    # --- smart swaps -------
    SM_2 = "SM(2)"
    SM_4 = "SM(4)"
    SM_8 = "SM(8)"
    SM_THOROUGH = "SM(thorough)"

    # -------------------------------------------------------------------------
    #  Factory
    # -------------------------------------------------------------------------
    def create(self) -> OptimizationStrategy:
        """Create an OptimizationStrategy instance corresponding to this preset."""
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

    def preset_note(self) -> str:
        """Return which shipped solver preset this configuration matches exactly, or '' when none."""
        return _PRESET_NOTES.get(self, "")

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
    OptimPreset.GS_3: (OptimGuidedSwaps, {"min_swap_size": 3, "max_swap_size": 3}),
    OptimPreset.GS_1_3: (OptimGuidedSwaps, {"min_swap_size": 1, "max_swap_size": 3}),
    # the GUIDED preset's optimizer, verbatim (see `get_preset_strategies_guided`); `GS(1-3)` is
    # the schedule-free baseline that isolates what this scheduling adds
    OptimPreset.GS_GUIDED: (
        OptimGuidedSwaps,
        {
            "min_swap_size": 1,
            "max_swap_size": 9,
            "swap_size_lambda": ease_out(6.0, 2.0),
            "remove_selectivity_modifier": ease_in(-0.8, +0.8),
            "add_selectivity_modifier": ease_out(-0.8, +0.8),
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
    # the THOROUGH preset's optimizer, verbatim (see `get_preset_strategies_smart`) — deliberately
    # not an `SM(64)` extension of the cost_awareness=0.5 sweep above
    OptimPreset.SM_THOROUGH: (
        OptimSmartSwaps,
        {
            "swap_size_max": 64,
            "nc_remove_max": 64,
            "nc_add_max": 64,
            "tau_learn": 10,
            "ignore_infeasible_diversity_up_to_fraction": 0.8,
            "cost_awareness": 0.1,
        },
    ),
}

_PRESET_NOTES: dict[OptimPreset, str] = {
    OptimPreset.GS_GUIDED: "= the GUIDED preset's optimizer",
    OptimPreset.SM_8: "= the SMART preset's optimizer",
    OptimPreset.SM_THOROUGH: "= the THOROUGH preset's optimizer",
}
