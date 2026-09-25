from __future__ import annotations

from enum import StrEnum
from typing import TYPE_CHECKING, Any

from max_div._core.solver._strategies._initialization._init_farthest_point import InitFarthestPoint
from max_div._core.solver._strategies._initialization._init_most_feasible import InitMostFeasible
from max_div._core.solver._strategies._initialization._init_random_selection import InitRandomSelection

if TYPE_CHECKING:
    from max_div._core.solver._strategies import InitializationStrategy


# =================================================================================================
#  Enum
# =================================================================================================
class InitPreset(StrEnum):
    """StrEnum for all initialization presets we want to benchmark.

    A strategy with a tunable parameter appears once per benchmarked value of that parameter;
    `_INIT_CLASSES_AND_KWARGS` lists the values.
    `_PRESET_NOTES` records which benchmark presets match the initialization of a shipped solver preset.
    """

    # --- random selection -----------------------
    RSEL = "rsel"
    RSEL_UNCON = "rsel(uncon)"

    # --- farthest point -------------------------
    FPS_1 = "fps(1)"
    FPS_8 = "fps(8)"
    FPS_8_ONE_AT_A_TIME = "fps(8,one-at-a-time)"  # candidate_pool_size=None: one item per pass over the dataset

    # --- most feasible --------------------------
    MF = "mf"

    # -------------------------------------------------------------------------
    #  Factory
    # -------------------------------------------------------------------------
    def create(self) -> InitializationStrategy:
        """Create an InitializationStrategy instance corresponding to this preset."""
        cls, kwargs = _INIT_CLASSES_AND_KWARGS[self]
        return cls(**kwargs)

    # -------------------------------------------------------------------------
    #  Meta-Data
    # -------------------------------------------------------------------------
    def is_constraint_aware(self) -> bool:
        return self not in [
            InitPreset.RSEL_UNCON,
            InitPreset.FPS_1,
            InitPreset.FPS_8,
            InitPreset.FPS_8_ONE_AT_A_TIME,
        ]

    def is_relevant_for_problem(self, problem_has_constraints: bool) -> bool:
        """Report whether this strategy belongs on the given problem's benchmark page.

        Two reasons drop a strategy from an unconstrained problem:

        - `most_feasible` *raises* without constraints, so it must not run.
        - `rsel(uncon)` is *redundant* there, behaving identically to the constraint-aware `rsel`.
        """
        if problem_has_constraints:
            return True
        dropped_when_unconstrained = {
            InitPreset.MF,
            InitPreset.RSEL_UNCON,
        }
        return self not in dropped_when_unconstrained

    def class_name(self) -> str:
        cls, _ = _INIT_CLASSES_AND_KWARGS[self]
        return cls.__name__

    def class_kwargs(self) -> dict[str, Any]:
        _, kwargs = _INIT_CLASSES_AND_KWARGS[self]
        return kwargs

    def preset_note(self) -> str:
        """Return which shipped solver preset this configuration matches exactly, or '' when none."""
        return _PRESET_NOTES.get(self, "")

    @classmethod
    def all(cls) -> list[InitPreset]:
        """Get a list of all InitPreset members."""
        return list(cls)


# =================================================================================================
#  Classes & Arguments
# =================================================================================================
_INIT_CLASSES_AND_KWARGS: dict[InitPreset, tuple[type[InitializationStrategy], dict[str, Any]]] = {
    InitPreset.RSEL: (InitRandomSelection, {"ignore_constraints": False}),
    InitPreset.RSEL_UNCON: (InitRandomSelection, {"ignore_constraints": True}),
    InitPreset.FPS_1: (InitFarthestPoint, {"top_k": 1, "candidate_pool_size": 256}),
    InitPreset.FPS_8: (InitFarthestPoint, {"top_k": 8, "candidate_pool_size": 256}),
    InitPreset.FPS_8_ONE_AT_A_TIME: (InitFarthestPoint, {"top_k": 8, "candidate_pool_size": None}),
    InitPreset.MF: (InitMostFeasible, {}),
}

_PRESET_NOTES: dict[InitPreset, str] = {
    InitPreset.RSEL_UNCON: "= the RANDOM/GUIDED presets' initialization",
    InitPreset.FPS_8: "= the SMART/THOROUGH presets' initialization (unconstrained problems)",
    InitPreset.MF: "= the SMART/THOROUGH presets' initialization (constrained problems)",
}
