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
    has_constraints: bool = False,
) -> tuple[InitializationStrategy, list[OptimizationStep]]:
    """Return the SMART (or THOROUGH, when `thorough`) preset's init strategy and optimization steps."""
    # --- initialization -------------------------
    if has_constraints:  # noqa: SIM108 — an if/else keeps a rationale comment on each init choice
        # Constrained: most_feasible() finds a feasible (or least-infeasible) selection faster than the
        # main solver's swaps could, freeing the optimizer to spend its whole budget on diversity.
        init_strategy = InitializationStrategy.most_feasible()
    else:
        # Unconstrained: the farthest-point construction reaches competitor-level quality far sooner
        # than a random start; sampling among the top_k picks keeps that quality while decorrelating seeds.
        init_strategy = InitFarthestPoint(top_k=8)

    # --- optimization steps ---------------------
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

    # --- done -----------------------------------
    return init_strategy, optim_steps
