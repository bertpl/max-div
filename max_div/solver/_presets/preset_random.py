from max_div.solver._duration import TargetDuration
from max_div.solver._solver_step import OptimizationStep
from max_div.solver._strategies import InitializationStrategy, OptimizationStrategy


# =================================================================================================
#  RANDOM preset
# =================================================================================================
def get_preset_strategies_random(
    target_duration: TargetDuration,
) -> tuple[InitializationStrategy, list[OptimizationStep]]:
    # --- initialization ----------------------------------
    init_strategy = InitializationStrategy.fast()

    # --- optimization steps ------------------------------
    optim_steps = [
        OptimizationStep(
            optim_strategy=OptimizationStrategy.random_swaps(),
            duration=target_duration,
        )
    ]

    # --- done --------------------------------------------
    return init_strategy, optim_steps
