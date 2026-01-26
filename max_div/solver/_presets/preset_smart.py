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
        # THOROUGH preset
        n_max = 64
        cost_awareness = 0.1
    else:
        # SMART preset
        n_max = 8
        cost_awareness = 0.5

    optim_steps = [
        OptimizationStep(
            optim_strategy=OptimizationStrategy.smart_swaps(
                swap_size_max=n_max,
                nc_remove_max=n_max,
                nc_add_max=n_max,
                tau_learn=10,
                ignore_infeasible_diversity_up_to_fraction=0.8,
                cost_awareness=cost_awareness,
            ),
            duration=target_duration,
        )
    ]

    # --- done --------------------------------------------
    return init_strategy, optim_steps
