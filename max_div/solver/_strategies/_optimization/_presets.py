from __future__ import annotations

from enum import StrEnum
from typing import Any

from max_div.solver._scheduling import ease_in_out, linear
from max_div.solver._strategies import OptimizationStrategy

from ._optim_guided_swaps import OptimGuidedSwaps
from ._optim_random_swaps import OptimRandomSwaps
from ._time_model import OptimizationTimeModel


# =================================================================================================
#  Enum
# =================================================================================================
class OptimPreset(StrEnum):
    """StrEnum for all optimization presets we want to benchmark or want to consider in MaxDivSolverBuilder."""

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

    def time_model(self) -> OptimizationTimeModel:
        """Get the OptimizationTimeModel associated with this preset."""
        return _OPTIM_TIME_MODELS[self]

    @classmethod
    def all(cls) -> list[OptimPreset]:
        """Get a list of all OptimPreset members."""
        return list(cls)


# =================================================================================================
#  Classes & Arguments
# =================================================================================================
_OPTIM_CLASSES_AND_KWARGS: dict[OptimPreset, tuple[type[OptimizationStrategy], dict[str, Any]]] = {
    OptimPreset.RS: (OptimRandomSwaps, dict()),
    OptimPreset.GS_1: (OptimGuidedSwaps, dict(min_swap_size=1, max_swap_size=1)),
    OptimPreset.GS_2: (OptimGuidedSwaps, dict(min_swap_size=2, max_swap_size=2)),
    OptimPreset.GS_3: (OptimGuidedSwaps, dict(min_swap_size=3, max_swap_size=3)),
    OptimPreset.GS_1_3: (OptimGuidedSwaps, dict(min_swap_size=1, max_swap_size=3)),
    OptimPreset.GS_1_3_SOFT: (
        OptimGuidedSwaps,
        dict(
            min_swap_size=1,
            max_swap_size=3,
            constraint_softness=ease_in_out(1.0, 0.0),
            p_add_constraint_aware=ease_in_out(0.0, 1.0),
        ),
    ),
    OptimPreset.GS_1_3_WIDE: (
        OptimGuidedSwaps,
        dict(
            min_swap_size=1,
            max_swap_size=3,
            remove_selectivity_modifier=-0.8,
            add_selectivity_modifier=-0.8,
        ),
    ),
    OptimPreset.GS_1_3_NARROW: (
        OptimGuidedSwaps,
        dict(
            min_swap_size=1,
            max_swap_size=3,
            remove_selectivity_modifier=+0.8,
            add_selectivity_modifier=+0.8,
        ),
    ),
    OptimPreset.GS_1_3_WI_NA: (
        OptimGuidedSwaps,
        dict(
            min_swap_size=1,
            max_swap_size=3,
            remove_selectivity_modifier=linear(-0.8, +0.8),
            add_selectivity_modifier=linear(-0.8, +0.8),
        ),
    ),
    OptimPreset.GS_1_3_NA_WI: (
        OptimGuidedSwaps,
        dict(
            min_swap_size=1,
            max_swap_size=3,
            remove_selectivity_modifier=linear(+0.8, -0.8),
            add_selectivity_modifier=linear(+0.8, -0.8),
        ),
    ),
}

# =================================================================================================
#  Time Models
# =================================================================================================
_OPTIM_TIME_MODELS: dict[OptimPreset, OptimizationTimeModel] = {
    OptimPreset.RS: OptimizationTimeModel(
        t0=2.58e-05, t_n=7.91e-08, t_k=1.28e-07, t_m=2.23e-08, t_n_con_indices=9.67e-10
    ),
    OptimPreset.GS_1: OptimizationTimeModel(
        t0=3.16e-05, t_n=9.39e-08, t_k=1.45e-07, t_m=3.56e-08, t_n_con_indices=2.25e-09
    ),
    OptimPreset.GS_2: OptimizationTimeModel(
        t0=4.48e-05, t_n=1.33e-07, t_k=2.21e-07, t_m=6.06e-08, t_n_con_indices=4.42e-09
    ),
    OptimPreset.GS_3: OptimizationTimeModel(
        t0=5.59e-05, t_n=1.39e-07, t_k=3.31e-07, t_m=8.14e-08, t_n_con_indices=6.44e-09
    ),
    OptimPreset.GS_1_3: OptimizationTimeModel(
        t0=3.90e-05, t_n=1.10e-07, t_k=1.86e-07, t_m=4.76e-08, t_n_con_indices=3.30e-09
    ),
    OptimPreset.GS_1_3_SOFT: OptimizationTimeModel(
        t0=3.95e-05, t_n=1.08e-07, t_k=1.75e-07, t_m=3.86e-08, t_n_con_indices=2.33e-09
    ),
    OptimPreset.GS_1_3_WIDE: OptimizationTimeModel(
        t0=3.86e-05, t_n=1.11e-07, t_k=1.88e-07, t_m=4.68e-08, t_n_con_indices=3.34e-09
    ),
    OptimPreset.GS_1_3_NARROW: OptimizationTimeModel(
        t0=3.85e-05, t_n=1.10e-07, t_k=1.83e-07, t_m=4.71e-08, t_n_con_indices=3.33e-09
    ),
    OptimPreset.GS_1_3_WI_NA: OptimizationTimeModel(
        t0=3.98e-05, t_n=1.11e-07, t_k=1.85e-07, t_m=4.76e-08, t_n_con_indices=3.30e-09
    ),
    OptimPreset.GS_1_3_NA_WI: OptimizationTimeModel(
        t0=4.00e-05, t_n=1.11e-07, t_k=1.91e-07, t_m=4.79e-08, t_n_con_indices=3.31e-09
    ),
}
