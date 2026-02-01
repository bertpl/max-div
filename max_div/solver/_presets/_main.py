from max_div.solver._duration import TargetDuration
from max_div.solver._problem import MaxDivProblem
from max_div.solver._solver_step import OptimizationStep
from max_div.solver._strategies import InitializationStrategy

from ._enum import SolverPreset
from .preset_guided import get_preset_strategies_guided
from .preset_random import get_preset_strategies_random
from .preset_smart import get_preset_strategies_smart


# =================================================================================================
#  Main entry point
# =================================================================================================
def get_preset_strategies(
    problem: MaxDivProblem,
    preset: SolverPreset,
    target_duration: TargetDuration,
) -> tuple[InitializationStrategy, list[OptimizationStep]]:
    match preset.resolve_alias():
        case SolverPreset.RANDOM:
            return get_preset_strategies_random(
                target_duration,
            )
        case SolverPreset.GUIDED:
            return get_preset_strategies_guided(
                problem,
                target_duration,
            )
        case SolverPreset.SMART:
            return get_preset_strategies_smart(
                problem,
                target_duration,
                thorough=False,
            )
        case SolverPreset.THOROUGH:
            return get_preset_strategies_smart(
                problem,
                target_duration,
                thorough=True,
            )
