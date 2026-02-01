from max_div.solver._duration import TargetDuration
from max_div.solver._parameters import ease_in, ease_out
from max_div.solver._problem import MaxDivProblem
from max_div.solver._solver_step import OptimizationStep
from max_div.solver._strategies import InitializationStrategy, OptimizationStrategy


# =================================================================================================
#  GUIDED preset
# =================================================================================================
def get_preset_strategies_guided(
    problem: MaxDivProblem,
    target_duration: TargetDuration,
) -> tuple[InitializationStrategy, list[OptimizationStep]]:
    """
    This preset consists of...
      - InitFast initialization strategy
      - GuidedSwaps optimization strategy
          --> parameters chosen to be reasonable for both unconstrained & constrained problems

    :param problem: (MaxDivProblem) The problem for which we want to determine a solver configuration.
    :param target_duration: (TargetDuration) The target duration to aim for.  (iteration- or time-based)
    """

    # --- initialization ----------------------------------
    init_strategy = InitializationStrategy.fast()

    # --- optimization strategy ---------------------------
    # RATIONALE: Benchmarks show that NARROW strategies result in the best diversity without sacrificing constraint
    #            satisfaction. However, where constraints cause very uneven spread of selected items or where
    #            the original distribution of items is very non-uniform, starting WIDE is expected to improve
    #            robustness of converging to a good solution.
    optim_steps = [
        OptimizationStep(
            optim_strategy=OptimizationStrategy.guided_swaps(
                min_swap_size=1,
                max_swap_size=9,
                swap_size_lambda=ease_out(6.0, 2.0),
                remove_selectivity_modifier=ease_in(-0.8, +0.8),  # wide -> narrow  (late)
                add_selectivity_modifier=ease_out(-0.8, +0.8),  #   wide -> narrow  (early)
            ),
            duration=target_duration,
        )
    ]

    # --- return final result -----------------------------
    return init_strategy, optim_steps
