from max_div.solver import MaxDivProblem
from max_div.solver._duration import TargetDuration
from max_div.solver._solver_step import OptimizationStep
from max_div.solver._strategies import InitializationStrategy

from ._enum import SolverPreset
from .preset_default import get_preset_strategies_default


# =================================================================================================
#  Main entry point
# =================================================================================================
def get_preset_strategies(
    problem: MaxDivProblem,
    preset: SolverPreset,
    target_duration: TargetDuration,
    initialization_included: bool = False,
    hardware_speed_correction: float = 1.0,
) -> tuple[InitializationStrategy, list[OptimizationStep]]:
    match preset:
        case SolverPreset.DEFAULT:
            return get_preset_strategies_default(
                problem,
                target_duration,
                initialization_included,
                hardware_speed_correction,
            )
