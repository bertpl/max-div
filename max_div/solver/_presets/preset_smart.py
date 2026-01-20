import math

from max_div.solver._duration import TargetDuration
from max_div.solver._problem import MaxDivProblem
from max_div.solver._solver_step import OptimizationStep
from max_div.solver._strategies import InitializationStrategy, OptimizationStrategy


# =================================================================================================
#  SMART preset
# =================================================================================================
def get_preset_strategies_smart(
    problem: MaxDivProblem,
    target_duration: TargetDuration,
    initialization_included: bool = False,
    hardware_speed_correction: float = 1.0,
) -> tuple[InitializationStrategy, list[OptimizationStep]]:
    # --- initialization ----------------------------------
    # init_strategy = InitializationStrategy.eager(nc=16)
    init_strategy = InitializationStrategy.fast()

    # --- optimization steps ------------------------------
    optim_steps = [
        OptimizationStep(
            optim_strategy=OptimizationStrategy.smart_swaps(
                swap_size_max=8,
                nc_remove_max=8,
                nc_add_max=8,
                tau_learn=100,
                tau_forget=math.inf,
                ignore_infeasible_diversity_up_to_fraction=0.5 if (problem.m > 0) else -1.0,
            ),
            duration=target_duration,
        )
    ]

    # --- done --------------------------------------------
    return init_strategy, optim_steps
