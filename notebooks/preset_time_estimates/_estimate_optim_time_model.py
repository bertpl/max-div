import math
from typing import Callable

import numpy as np
from scipy.optimize import minimize
from tqdm import tqdm

from max_div.solver._duration import iterations
from max_div.solver._strategies import OptimizationStrategy
from max_div.solver._strategies._optimization._time_model import OptimizationTimeModel

from .helpers import benchmark_optim_strategy, get_problem_dimensions
from .helpers.problem_dimensions import ProblemDimensions


# =================================================================================================
#  Main function
# =================================================================================================
def estimate_optim_time_model(
    name: str,
    optim_strat_factory: Callable[[], OptimizationStrategy],
) -> OptimizationTimeModel:
    # --- init ----------------------------------
    problem_dims: list[ProblemDimensions] = get_problem_dimensions()

    # --- benchmark -----------------------------
    benchmark_data = {
        dim: benchmark_optim_strategy(
            problem=dim.build_problem(),
            optim_strat_factory=optim_strat_factory,
            duration=iterations(100),
        )
        for dim in tqdm(problem_dims, desc=f"Benchmarking optim strategy '{name}'")
    }

    # --- fit time model ------------------------
    cost_function = OptimizationCost(benchmark_data=benchmark_data)

    # minimize using Powell's method
    optim_result = minimize(
        fun=cost_function,
        x0=cost_function.x0(),
        method="powell",
        bounds=cost_function.bounds(),
        options=dict(
            xtol=1e-16,
            ftol=1e-16,
            disp=False,
            maxiter=1_000,
            maxfev=100_000,
        ),
    )
    x_opt: np.ndarray = optim_result.x

    print(f"   --> Optimal cost function: {optim_result.fun: .3f}")

    return _x_to_time_model(x_opt)


# =================================================================================================
#  Helpers
# =================================================================================================
class OptimizationCost:
    """Class encapsulating the fitting data & model, resulting in a callable for use in scipy.optimize.minimize."""

    def __init__(self, benchmark_data: dict[ProblemDimensions, float]):
        self.benchmark_data = benchmark_data

    @staticmethod
    def lb() -> tuple[float, ...]:
        return -12.0, -12.0, -12.0, -12.0, -12.0

    @staticmethod
    def ub() -> tuple[float, ...]:
        return -3.0, -3.0, -3.0, -3.0, -3.0

    def bounds(self) -> list[tuple[float, float]]:
        return list(zip(self.lb(), self.ub()))

    def x0(self) -> np.ndarray:
        return np.array([0.5 * (lb + ub) for lb, ub in zip(self.lb(), self.ub())])

    def __call__(self, x: np.ndarray) -> float:
        """
        Cost function that computes RMS of relative errors between predicted and actual times.
        :param x: (log_t0, log_t_n, log_t_k, log_t_m, log_t_n_con_indices)-tuple  (base-10 log)
        :return: (float) cost (RMS relative error)
        """
        squared_errors: list[float] = []
        for dim, t_target in self.benchmark_data.items():
            # --- compute t_pred ---
            time_model = _x_to_time_model(x)
            t_pred = time_model.get_time_sec(
                n=dim.n,
                k=dim.k,
                m=dim.m,
                n_con_indices=dim.n_con_indices,
            )

            # --- compute error ---
            rel_error = (t_pred - t_target) / t_target
            squared_errors.append(rel_error**2)

        return float(math.sqrt(np.mean(squared_errors)))


def _x_to_time_model(x: np.ndarray) -> OptimizationTimeModel:
    return OptimizationTimeModel(
        t0=float(10 ** x[0]),
        t_n=float(10 ** x[1]),
        t_k=float(10 ** x[2]),
        t_m=float(10 ** x[3]),
        t_n_con_indices=float(10 ** x[4]),
    )
