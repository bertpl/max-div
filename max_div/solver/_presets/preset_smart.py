import math

from max_div.solver._duration import TargetDuration
from max_div.solver._problem import MaxDivProblem
from max_div.solver._solver_step import OptimizationStep
from max_div.solver._strategies import InitializationStrategy, OptimizationStrategy


# =================================================================================================
#  SMART / THOROUGH preset
# =================================================================================================
def get_preset_strategies_smart(
    problem: MaxDivProblem,
    target_duration: TargetDuration,
    thorough: bool = False,
) -> tuple[InitializationStrategy, list[OptimizationStep]]:
    # --- initialization ----------------------------------
    init_strategy = InitializationStrategy.fast()

    # --- optimization steps ------------------------------
    if thorough:
        n_max = 64
    else:
        n_max = 8

    optim_steps = [
        OptimizationStep(
            optim_strategy=OptimizationStrategy.smart_swaps(
                swap_size_max=n_max,
                nc_remove_max=n_max,
                nc_add_max=n_max,
                tau_learn=10,
                tau_forget=math.inf,
                ignore_infeasible_diversity_up_to_fraction=0.8,
            ),
            duration=target_duration,
        )
    ]

    # --- done --------------------------------------------
    return init_strategy, optim_steps
