from __future__ import annotations

from enum import StrEnum
from typing import Any

from max_div.solver._strategies import InitializationStrategy
from max_div.solver._strategies._initialization._init_eager import InitEager
from max_div.solver._strategies._initialization._init_fast import InitFast
from max_div.solver._strategies._initialization._init_random_batched import InitRandomBatched
from max_div.solver._strategies._initialization._init_random_one_shot import InitRandomOneShot


# =================================================================================================
#  Enum
# =================================================================================================
class InitPreset(StrEnum):
    """StrEnum for all initialization presets we want to benchmark or want to consider in MaxDivSolverBuilder."""

    # --- fast --------------
    FAST = "fast"

    # --- random one-shot ---
    ROS_U = "ros(u)"
    ROS_NU = "ros(nu)"
    ROS_U_UNCON = "ros(u,uncon)"
    ROS_NU_UNCON = "ros(nu,uncon)"

    # --- random batched ----
    RB_2 = "rb(2)"
    RB_4 = "rb(4)"
    RB_8 = "rb(8)"
    RB_16 = "rb(16)"

    # --- eager -------------
    E_2 = "e(2)"
    E_4 = "e(4)"
    E_8 = "e(8)"
    E_16 = "e(16)"

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
        if self in [InitPreset.FAST, InitPreset.ROS_U_UNCON, InitPreset.ROS_NU_UNCON]:
            return False
        else:
            return True

    def is_relevant_for_problem(self, problem_has_constraints: bool) -> bool:
        if problem_has_constraints:
            return True
        else:
            if self in [InitPreset.ROS_U_UNCON, InitPreset.ROS_NU_UNCON]:
                return False  # the constraint-aware versions will behave the same as these on unconstrained problems
            else:
                return True

    def class_name(self) -> str:
        cls, _ = _INIT_CLASSES_AND_KWARGS[self]
        return cls.__name__

    def class_kwargs(self) -> dict[str, Any]:
        _, kwargs = _INIT_CLASSES_AND_KWARGS[self]
        return kwargs

    @classmethod
    def all(cls) -> list[InitPreset]:
        """Get a list of all InitPreset members."""
        return list(cls)


# =================================================================================================
#  Classes & Arguments
# =================================================================================================
_INIT_CLASSES_AND_KWARGS: dict[InitPreset, tuple[type[InitializationStrategy], dict[str, Any]]] = {
    InitPreset.FAST: (InitFast, dict()),
    InitPreset.ROS_U: (InitRandomOneShot, dict(uniform=True, ignore_constraints=False)),
    InitPreset.ROS_NU: (InitRandomOneShot, dict(uniform=False, ignore_constraints=False)),
    InitPreset.ROS_U_UNCON: (InitRandomOneShot, dict(uniform=True, ignore_constraints=True)),
    InitPreset.ROS_NU_UNCON: (InitRandomOneShot, dict(uniform=False, ignore_constraints=True)),
    InitPreset.RB_2: (InitRandomBatched, dict(b=2, ignore_constraints=False)),
    InitPreset.RB_4: (InitRandomBatched, dict(b=4, ignore_constraints=False)),
    InitPreset.RB_8: (InitRandomBatched, dict(b=8, ignore_constraints=False)),
    InitPreset.RB_16: (InitRandomBatched, dict(b=16, ignore_constraints=False)),
    InitPreset.E_2: (InitEager, dict(nc=2, ignore_constraints=False)),
    InitPreset.E_4: (InitEager, dict(nc=4, ignore_constraints=False)),
    InitPreset.E_8: (InitEager, dict(nc=8, ignore_constraints=False)),
    InitPreset.E_16: (InitEager, dict(nc=16, ignore_constraints=False)),
}
