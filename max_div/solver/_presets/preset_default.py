"""
Main functionality for the 'default' preset of the MaxDivSolverBuilder.
"""

import math

from max_div.solver import MaxDivProblem
from max_div.solver._duration import TargetDuration, _TargetIterationCount, _TargetTimeDuration
from max_div.solver._parameters import ease_in, ease_out, linear
from max_div.solver._solver_step import OptimizationStep
from max_div.solver._strategies import InitializationStrategy, OptimizationStrategy
from max_div.solver._strategies._initialization._presets import InitPreset
from max_div.solver._strategies._optimization._presets import OptimPreset

# =================================================================================================
#  Constants
# =================================================================================================
# Candidate initialization presets, ordered from fastest to slowest (lowest to highest quality),
# which are considered in this order and selected based on estimated time budget.
# On larger problems, time taken by these method can range by a factor of up to 1000x.
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
#  DEFAULT preset
# =================================================================================================
def get_preset_strategies_default(
    problem: MaxDivProblem,
    target_duration: TargetDuration,
    initialization_included: bool = False,
    hardware_speed_correction: float = 1.0,
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
                  use <= 5% of the target_duration

    :param problem: (MaxDivProblem) The problem for which we want to determine a solver configuration.
    :param target_duration: (TargetDuration) The target duration to aim for.  (iteration- or time-based)
    :param initialization_included: (bool) Whether the target_duration includes initialization time or only refers
                                           to optimization time.  If True, duration of the optimization is reduced,
                                           based on the estimated initialization time.   (default: False)
    :param hardware_speed_correction: (float) set to value >1 if current hardware is faster than the reference
                                        hardware.  Use estimate_platform_speed to get an estimate.
    """

    # --- optimization strategy ---------------------------
    # RATIONALE: Benchmarks show that NARROW strategies result in the best diversity without sacrificing constraint
    #            satisfaction. However, where constraints cause very uneven spread of selected items or where
    #            the original distribution of items is very non-uniform, starting WIDE is expected to improve
    #            robustness of converging to a good solution.
    optim_strategy = OptimizationStrategy.guided_swaps(
        min_swap_size=1,
        max_swap_size=9,
        swap_size_lambda=ease_out(6.0, 2.0),
        remove_selectivity_modifier=ease_in(-0.8, +0.8),  # wide -> narrow  (late)
        add_selectivity_modifier=ease_out(-0.8, +0.8),  #   wide -> narrow  (early)
    )

    # --- convert duration to time ------------------------
    if isinstance(target_duration, _TargetTimeDuration):
        target_duration_sec = target_duration._t_target_sec
    elif isinstance(target_duration, _TargetIterationCount):
        target_duration_iterations = target_duration._n_iters
        time_model = OptimPreset.GS_1_3_WI_NA.time_model()
        t_sec_per_iteration = hardware_speed_correction * time_model.get_time_sec(
            n=problem.n,
            k=problem.k,
            m=problem.m,
            n_con_indices=problem.n_constraint_indices,
        )
        target_duration_sec = target_duration_iterations * t_sec_per_iteration
    else:
        raise TypeError(f"Unsupported TargetDuration type: {type(target_duration)}")

    # --- determine initialization strategy ---------------
    max_init_time_sec = 0.05 * target_duration_sec  # 5% of target duration

    # initialize
    best_init_preset: InitPreset = InitPreset.FAST
    best_init_time_sec: float = math.inf

    # go over all candidates
    for init_preset in _CANDIDATE_INIT_PRESETS:
        time_model = init_preset.time_model()
        est_init_time_sec = hardware_speed_correction * time_model.get_time_sec(
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

    # --- determine optimization duration -----------------
    if initialization_included:
        # reduce optimization duration based on estimated init time
        #  e.g. assume...
        #         - target_duration = 1000 iterations, which we expect to take 10sec for the selected optim strategy
        #         - we have selected an initialization strategy that is expected to take 0.4sec.
        #        ==> then we have 9.6sec left for optimization
        #        ==> we reduce target_duration by (0.4/10)=4%  --> 960 iterations left for optimization
        init_time_frac = best_init_time_sec / target_duration_sec  # this could be >1 for very small target durations
        optim_time_frac = max(0.0, 1.0 - init_time_frac)
    else:
        optim_time_frac = 1.0  # 100% of target duration for optimization

    # --- set up solver steps -----------------------------
    if optim_time_frac > 0.0:
        optim_duration = optim_time_frac * target_duration  # this can be time- or iteration-based
        optim_steps = [
            OptimizationStep(
                optim_strategy=optim_strategy,
                duration=optim_duration,
            )
        ]
    else:
        optim_steps = []

    # --- return final result -----------------------------
    return init_strategy, optim_steps
