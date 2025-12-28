"""
Main functionality for the 'default' preset of the MaxDivSolverBuilder.
"""

import math

from max_div.solver import MaxDivProblem
from max_div.solver._duration import TargetDuration, _TargetIterationCount, _TargetTimeDuration
from max_div.solver._scheduling import ease_in, ease_out, linear
from max_div.solver._solver_step import OptimizationStep
from max_div.solver._strategies import InitializationStrategy, OptimizationStrategy
from max_div.solver._strategies._initialization._presets import InitPreset
from max_div.solver._strategies._optimization._presets import OptimPreset

# =================================================================================================
#  Constants
# =================================================================================================
_CANDIDATE_INIT_PRESETS = [
    InitPreset.FAST,
    InitPreset.ROS_NU,
    InitPreset.RB_2,
    InitPreset.RB_4,
    InitPreset.RB_8,
    InitPreset.RB_16,
    InitPreset.E_2,
    InitPreset.E_4,
    InitPreset.E_8,
    InitPreset.E_16,
]


# =================================================================================================
#  Main entry point
# =================================================================================================
def preset_default_get_strategies(
    problem: MaxDivProblem, duration: TargetDuration
) -> tuple[InitializationStrategy, list[OptimizationStep]]:
    """
    Main logic of the 'default' preset.

    Based on the problem & target duration, it returns...
      - appropriate initialization strategy
      - appropriate optimization steps

    This is tackled in 2 steps:
       1) determine an appropriate set of optimization steps based on problem & duration
            --> this allows us to estimate how much time (seconds) this is expected to take
       2) determine initialization strategy
            --> this is chosen from a list of presets, such that the most accurate preset is chosen that will
                  use <= 5% of the estimated optimization time

    """

    # --- optimization strategy ---------------------------
    # RATIONALE: Benchmarks show that NARROW strategies result in best diversity without satisfying constraint
    #            satisfaction. However, where constraints cause very uneven spread of selected items or where
    #            the original distribution of items is very non-uniform, starting WIDE is expected to improve
    #            robustness of converging to a good solution.
    optim_strategy = OptimizationStrategy.guided_swaps(
        min_swap_size=1,
        max_swap_size=3,
        swap_size_lambda=1.0,
        remove_selectivity_modifier=ease_in(-0.8, +0.8),  # wide -> narrow  (late)
        add_selectivity_modifier=ease_out(-0.8, +0.8),  #   wide -> narrow  (early)
    )

    # --- convert duration to time ------------------------
    target_duration_sec: float = 1.0
    if isinstance(duration, _TargetTimeDuration):
        target_duration_sec = duration._t_target_sec
    elif isinstance(duration, _TargetIterationCount):
        target_duration_iterations = duration._n_iters
        time_model = OptimPreset.GS_1_3_WI_NA.time_model()
        t_sec_per_iteration = time_model.get_time_sec(
            n=problem.n,
            k=problem.k,
            m=problem.m,
            n_con_indices=problem.n_constraint_indices,
        )
        target_duration_sec = target_duration_iterations * t_sec_per_iteration

    # --- determine initialization strategy ---------------
    max_init_time_sec = 0.05 * target_duration_sec  # 5% of total time budget

    best_init_preset: InitPreset = InitPreset.FAST
    best_init_time_sec: float = math.inf

    for init_preset in _CANDIDATE_INIT_PRESETS:
        time_model = init_preset.time_model()
        est_init_time_sec = time_model.get_time_sec(
            n=problem.n,
            k=problem.k,
            m=problem.m,
            n_con_indices=problem.n_constraint_indices,
        )

        if (est_init_time_sec <= max_init_time_sec) or (math.isinf(best_init_time_sec)):
            # valid candidate
            #   --> overwrite previous one, since we iterate from low to high quality
            best_init_preset = init_preset
            best_init_time_sec = est_init_time_sec

    init_strategy = best_init_preset.create()

    # --- see how much time we have left to optimize ------
    init_time_frac = best_init_time_sec / target_duration_sec  # this could be >1 for very small target durations
    optim_time_frac = max(0.0, 1.0 - init_time_frac)
    if optim_time_frac > 0.0:
        optim_duration: TargetDuration = optim_time_frac * duration  # this can be time- or iteration-based
        optim_steps = [
            OptimizationStep(
                optim_strategy=optim_strategy,
                duration=optim_duration,
            )
        ]
    else:
        optim_steps = []

    # --- we're done ---
    return init_strategy, optim_steps
