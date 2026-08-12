from max_div._core.solver._duration import TargetDuration
from max_div._core.solver._solver_step import OptimizationStep
from max_div._core.solver._strategies import InitializationStrategy, OptimizationStrategy
from max_div._core.solver._strategies._initialization._init_farthest_point import InitFarthestPoint


# =================================================================================================
#  SMART / THOROUGH preset
# =================================================================================================
def get_preset_strategies_smart(
    target_duration: TargetDuration,
    thorough: bool = False,
) -> tuple[InitializationStrategy, list[OptimizationStep]]:
    # --- initialization ----------------------------------
    # The greedy farthest-point construction reaches competitor-level quality far sooner than a
    # random start, but its pure form costs quality on unconstrained problems at long budget; a
    # short random prefix (random_fraction=0.1) removes that penalty while keeping most of the
    # short-budget edge. random_fraction stays off the public farthest_point() factory.
    init_strategy = InitFarthestPoint(random_fraction=0.1)

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
