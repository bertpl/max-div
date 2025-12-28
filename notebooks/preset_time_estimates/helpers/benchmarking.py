from typing import Callable

import numpy as np

from max_div.solver import MaxDivProblem, MaxDivSolverBuilder
from max_div.solver._duration import TargetDuration, seconds
from max_div.solver._solver_step import OptimizationStep
from max_div.solver._strategies import InitializationStrategy, OptimizationStrategy


# =================================================================================================
#  Initialization
# =================================================================================================
def benchmark_init_strategy(
    problem: MaxDivProblem,
    init_strat_factory: Callable[[], InitializationStrategy],
    n_seeds: int = 3,
) -> float:
    """Estimate total time (seconds) of initialization strategy for given problem & init strategy."""

    # --- benchmark initialization ----------------
    t_results: list[float] = []
    for i in range(n_seeds + 1):
        # solve
        solver = (
            MaxDivSolverBuilder(problem)
            .set_initialization_strategy(init_strat_factory())
            .with_seed(seed=42 + i)
            .build()
        )
        result = solver.solve()

        # extract time
        if i > 0:  # skip first run to avoid cold-start effects
            last_step_duration = list(result.step_durations.values())[-1]
            t_results.append(last_step_duration.t_elapsed_sec)

    # --- return result -------------------------
    return float(np.median(t_results))


# =================================================================================================
#  Optimization
# =================================================================================================
def benchmark_optim_strategy(
    problem: MaxDivProblem,
    optim_strat_factory: Callable[[], OptimizationStrategy],
    n_seeds: int = 3,
    duration: TargetDuration = seconds(0.1),
) -> float:
    """Estimate time (seconds) per iteration of Guided Swaps optimization strategy for given problem & settings."""

    # --- benchmark guided swaps ----------------
    t_results: list[float] = []
    for i in range(n_seeds + 1):
        # solve
        solver = (
            MaxDivSolverBuilder(problem)
            .set_initialization_strategy(InitializationStrategy.fast())
            .add_solver_step(
                OptimizationStep(
                    optim_strategy=optim_strat_factory(),
                    duration=duration,
                )
            )
            .with_seed(seed=42 + i)
            .build()
        )
        result = solver.solve()

        # extract time
        if i > 0:  # skip first run to avoid cold-start effects
            last_step_duration = list(result.step_durations.values())[-1]
            t_results.append(last_step_duration.t_elapsed_sec / last_step_duration.n_iterations)

    # --- return result -------------------------
    return float(np.median(t_results))
