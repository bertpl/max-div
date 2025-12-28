from __future__ import annotations

from enum import StrEnum
from typing import Any

from max_div.solver._strategies import InitializationStrategy

from ._init_eager import InitEager
from ._init_fast import InitFast
from ._init_random_batched import InitRandomBatched
from ._init_random_one_shot import InitRandomOneShot
from ._time_model import InitializationTimeModel


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

    def time_model(self) -> InitializationTimeModel:
        """Get the InitializationTimeModel associated with this preset."""
        return _INIT_TIME_MODELS[self]

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

# =================================================================================================
#  Time Models
# =================================================================================================
_INIT_TIME_MODELS: dict[InitPreset, InitializationTimeModel] = {
    InitPreset.FAST: InitializationTimeModel(
        t0=7.45e-06, t0_k=4.01e-06, t_n=7.93e-10, t_k=1.69e-08, t_m=3.56e-09, t_n_con_indices=2.59e-10
    ),
    InitPreset.ROS_U: InitializationTimeModel(
        t0=1.99e-05, t0_k=3.45e-06, t_n=5.56e-09, t_k=4.69e-09, t_m=3.35e-08, t_n_con_indices=3.92e-09
    ),
    InitPreset.ROS_NU: InitializationTimeModel(
        t0=2.03e-05, t0_k=3.70e-06, t_n=5.69e-09, t_k=3.60e-09, t_m=3.33e-08, t_n_con_indices=3.95e-09
    ),
    InitPreset.ROS_U_UNCON: InitializationTimeModel(
        t0=4.66e-05, t0_k=3.82e-06, t_n=2.67e-09, t_k=1.45e-08, t_m=4.91e-09, t_n_con_indices=3.13e-10
    ),
    InitPreset.ROS_NU_UNCON: InitializationTimeModel(
        t0=1.24e-05, t0_k=3.80e-06, t_n=2.63e-09, t_k=1.53e-08, t_m=5.47e-09, t_n_con_indices=2.97e-10
    ),
    InitPreset.RB_2: InitializationTimeModel(
        t0=5.56e-05, t0_k=3.51e-06, t_n=8.19e-09, t_k=3.10e-09, t_m=1.36e-08, t_n_con_indices=1.47e-09
    ),
    InitPreset.RB_4: InitializationTimeModel(
        t0=9.74e-05, t0_k=2.58e-06, t_n=1.11e-08, t_k=1.00e-12, t_m=1.27e-08, t_n_con_indices=1.75e-09
    ),
    InitPreset.RB_8: InitializationTimeModel(
        t0=8.59e-05, t0_k=5.80e-06, t_n=1.23e-08, t_k=1.00e-12, t_m=1.49e-08, t_n_con_indices=1.48e-09
    ),
    InitPreset.RB_16: InitializationTimeModel(
        t0=7.48e-05, t0_k=9.27e-06, t_n=1.41e-08, t_k=1.00e-12, t_m=1.67e-08, t_n_con_indices=1.49e-09
    ),
    InitPreset.E_2: InitializationTimeModel(
        t0=1.00e-12, t0_k=4.26e-05, t_n=1.38e-07, t_k=7.58e-08, t_m=5.64e-08, t_n_con_indices=3.81e-09
    ),
    InitPreset.E_4: InitializationTimeModel(
        t0=5.61e-05, t0_k=7.14e-05, t_n=2.41e-07, t_k=1.78e-07, t_m=9.24e-08, t_n_con_indices=7.05e-09
    ),
    InitPreset.E_8: InitializationTimeModel(
        t0=8.44e-05, t0_k=1.36e-04, t_n=4.51e-07, t_k=3.20e-07, t_m=1.80e-07, t_n_con_indices=1.34e-08
    ),
    InitPreset.E_16: InitializationTimeModel(
        t0=2.42e-12, t0_k=2.37e-04, t_n=9.19e-07, t_k=5.92e-07, t_m=3.63e-07, t_n_con_indices=2.60e-08
    ),
}
