from __future__ import annotations

from enum import StrEnum
from typing import TYPE_CHECKING, Any

from max_div._core.solver._strategies._initialization._init_eager import InitEager
from max_div._core.solver._strategies._initialization._init_farthest_point import InitFarthestPoint
from max_div._core.solver._strategies._initialization._init_farthest_point_batched import InitFarthestPointBatched
from max_div._core.solver._strategies._initialization._init_fast import InitFast
from max_div._core.solver._strategies._initialization._init_most_feasible import InitMostFeasible
from max_div._core.solver._strategies._initialization._init_random_batched import InitRandomBatched
from max_div._core.solver._strategies._initialization._init_random_one_shot import InitRandomOneShot

if TYPE_CHECKING:
    from max_div._core.solver._strategies import InitializationStrategy


# =================================================================================================
#  Enum
# =================================================================================================
class InitPreset(StrEnum):
    """StrEnum for all initialization presets we want to benchmark.

    Tunable strategies appear at several settings: random-batched and eager at two batch/candidate
    sizes each.  Shipped-preset correspondences live in `_PRESET_NOTES`.
    """

    # --- fast -----------------------------------
    FAST = "fast"

    # --- random one-shot ------------------------
    ROS = "ros"
    ROS_UNCON = "ros(uncon)"

    # --- random batched -------------------------
    RB_4 = "rb(4)"
    RB_16 = "rb(16)"

    # --- eager ----------------------------------
    E_4 = "e(4)"
    E_16 = "e(16)"

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
            InitPreset.FAST,
            InitPreset.ROS_UNCON,
            InitPreset.FPS_1,
            InitPreset.FPS_8,
            InitPreset.FPSB_8,
        ]

    def is_relevant_for_problem(self, problem_has_constraints: bool) -> bool:
        """Report whether this strategy belongs on the given problem's benchmark page.

        Two reasons drop a strategy from an unconstrained problem:

        - `most_feasible` *raises* without constraints, so it must not run.
        - `ros(uncon)` is *redundant* there, behaving identically to the constraint-aware `ros`.
        """
        if problem_has_constraints:
            return True
        dropped_when_unconstrained = {
            InitPreset.MF,
            InitPreset.ROS_UNCON,
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
    InitPreset.FAST: (InitFast, {}),
    InitPreset.ROS: (InitRandomOneShot, {"ignore_constraints": False}),
    InitPreset.ROS_UNCON: (InitRandomOneShot, {"ignore_constraints": True}),
    InitPreset.RB_4: (InitRandomBatched, {"b": 4, "ignore_constraints": False}),
    InitPreset.RB_16: (InitRandomBatched, {"b": 16, "ignore_constraints": False}),
    InitPreset.E_4: (InitEager, {"nc": 4, "ignore_constraints": False}),
    InitPreset.E_16: (InitEager, {"nc": 16, "ignore_constraints": False}),
    InitPreset.FPS_1: (InitFarthestPoint, {"top_k": 1}),
    InitPreset.FPS_8: (InitFarthestPoint, {"top_k": 8}),
    InitPreset.FPSB_8: (InitFarthestPointBatched, {"top_k": 8}),
    InitPreset.MF: (InitMostFeasible, {}),
}

_PRESET_NOTES: dict[InitPreset, str] = {
    InitPreset.ROS_UNCON: "= the RANDOM/GUIDED presets' initialization",
    InitPreset.FPSB_8: "= the SMART/THOROUGH presets' initialization (unconstrained problems)",
    InitPreset.MF: "= the SMART/THOROUGH presets' initialization (constrained problems)",
}
