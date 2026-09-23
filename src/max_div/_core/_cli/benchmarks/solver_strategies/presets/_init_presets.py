from __future__ import annotations

from enum import StrEnum
from typing import TYPE_CHECKING, Any

from max_div._core.solver._strategies._initialization._init_farthest_point import InitFarthestPoint
from max_div._core.solver._strategies._initialization._init_farthest_point_batched import InitFarthestPointBatched
from max_div._core.solver._strategies._initialization._init_most_feasible import InitMostFeasible
from max_div._core.solver._strategies._initialization._init_random_selection import InitRandomSelection

if TYPE_CHECKING:
    from max_div._core.solver._strategies import InitializationStrategy


# =================================================================================================
#  Enum
# =================================================================================================
class InitPreset(StrEnum):
    """StrEnum for all initialization presets we want to benchmark.

    Tunable strategies appear at several settings, listed in `_INIT_CLASSES_AND_KWARGS`.
    `_PRESET_NOTES` records which benchmark presets match the initialization of a shipped solver preset.
    """

    # --- random selection -----------------------
    RS = "rs"
    RS_UNCON = "rs(uncon)"

    # --- farthest point -------------------------
    FPS_1 = "fps(1)"
    FPS_8 = "fps(8)"

    # --- farthest point, batched ----------------
    FPSB_8 = "fpsb(8)"

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
            InitPreset.RS_UNCON,
            InitPreset.FPS_1,
            InitPreset.FPS_8,
            InitPreset.FPSB_8,
        ]

    def is_relevant_for_problem(self, problem_has_constraints: bool) -> bool:
        """Report whether this strategy belongs on the given problem's benchmark page.

        Two reasons drop a strategy from an unconstrained problem:

        - `most_feasible` *raises* without constraints, so it must not run.
        - `rs(uncon)` is *redundant* there, behaving identically to the constraint-aware `rs`.
        """
        if problem_has_constraints:
            return True
        dropped_when_unconstrained = {
            InitPreset.MF,
            InitPreset.RS_UNCON,
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
    InitPreset.RS: (InitRandomSelection, {"ignore_constraints": False}),
    InitPreset.RS_UNCON: (InitRandomSelection, {"ignore_constraints": True}),
    InitPreset.FPS_1: (InitFarthestPoint, {"top_k": 1}),
    InitPreset.FPS_8: (InitFarthestPoint, {"top_k": 8}),
    InitPreset.FPSB_8: (InitFarthestPointBatched, {"top_k": 8}),
    InitPreset.MF: (InitMostFeasible, {}),
}

_PRESET_NOTES: dict[InitPreset, str] = {
    InitPreset.RS_UNCON: "= the RANDOM/GUIDED presets' initialization",
    InitPreset.FPSB_8: "= the SMART/THOROUGH presets' initialization (unconstrained problems)",
    InitPreset.MF: "= the SMART/THOROUGH presets' initialization (constrained problems)",
}
