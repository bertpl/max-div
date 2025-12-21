import math
from functools import lru_cache
from itertools import product

import numba
import numpy as np
from numpy._typing import NDArray

from max_div.internal.math.fast_log_exp._calibration import exp2_approx, log2_approx
from max_div.internal.math.fast_pow._fast_pow import _D_E0 as d0_current
from max_div.internal.math.fast_pow._fast_pow import _D_E1 as d1_current
from max_div.internal.math.fast_pow._fast_pow import _D_E2 as d2_current
from max_div.internal.math.fast_pow._fast_pow import _D_L0 as c0_current
from max_div.internal.math.fast_pow._fast_pow import _D_L1 as c1_current
from max_div.internal.math.fast_pow._fast_pow import _D_L2 as c2_current
from max_div.internal.math.optimization import minimize_nd_random


# =================================================================================================
#  Main Calibration Function
# =================================================================================================
def calibrate_fast_pow(
    n_data: int, acc: float, n_evals: int, start_from_current_optimum: bool = False
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """
    Jointly calibrates the quadratic coefficients for the fast_log2 and fast_exp2 parts of the fast_pow function,
      to calibrate the overall approximation error of fast_pow(x, t) = fast_exp2(t * fast_log2(x)).

    For this purpose (not used in any scoring functions), we don't need smoothness at the interval boundaries,
      only continuity, which frees up 2 extra degrees of freedom, to further optimize the cost.

    Also, we don't seek to optimize max-abs error, but rather mean-abs error.

    PARAMETERS
    ----------

      --> FAST_LOG2

            - we look for an expression of the form `f(x) ≈ c0 + c1*x + c2*x^2`   (over [0.5, 1])
            - such that `f(0.5) = f(1) - 1`    (continuous)
            - hence `c0=free, c1=free, c2=(4-2*c1)/3`

            --> free parameter c0 with optimal value expected to be ~ in range [-4.0, -2.0]
                               c1 with optimal value expected to be ~ in range [3.0, 5.0]

        --> FAST_EXP2

            - we look for an expression of the form `g(x) ≈ d0 + d1*x + d2*x^2`   (over [0, 1])
            - such that `g(0) = 0.5*g(1)`      (continuous)
            - hence `d0=free, d1=free, d2=d0-d1`

            --> free parameter d0 with optimal value expected to be ~ in range [0.0, 2.0]
                               d1 with optimal value expected to be ~ in range [0.0, 2.0]

    OPTIMIZATION GOAL
    -----------------

    We want to optimize k1, k2 such that we minimize the mean abs error over a set of calibration-tuples (x,t).

    """

    if start_from_current_optimum:
        lb = (c0_current - 0.1, c1_current - 0.1, d0_current - 0.1, d1_current - 0.1)
        ub = (c0_current + 0.1, c1_current + 0.1, d0_current + 0.1, d1_current + 0.1)
    else:
        lb = (-4.0, 3.0, 0.0, 0.0)
        ub = (-2.0, 5.0, 2.0, 2.0)

    cost_fun = Cost(n=n_data)
    params_opt = minimize_nd_random(
        fun=cost_fun,
        lb=lb,
        ub=ub,
        acc=acc,
        n_evals=n_evals,
    )

    c0, c1, c2 = cost_fun.compute_c(params_opt)
    d0, d1, d2 = cost_fun.compute_d(params_opt)

    print()
    print()
    print(f"(c0, c1, c2) = ({c0:.18f}, {c1:.18f}, {c2:.18f})")
    print(f"(d0, d1, d2) = ({d0:.18f}, {d1:.18f}, {d2:.18f})")
    print()
    print(f"error = {cost_fun.error(c0, c1, c2, d0, d1, d2):.6f}")
    print()
    print("-------------------------------------------")
    print(f"_D_L0 = {c0:.20f}")
    print(f"_D_L1 = {c1:.20f}")
    print(f"_D_L2 = {c2:.20f}")
    print()
    print(f"_D_E0 = {d0:.20f}")
    print(f"_D_E1 = {d1:.20f}")
    print(f"_D_E2 = {d2:.20f}")
    print("-------------------------------------------")

    return (c0, c1, c2), (d0, d1, d2)


# =================================================================================================
#  Cost Function
# =================================================================================================
class Cost:
    def __init__(self, n: int):
        n_squared = int(math.sqrt(n))
        x_values, t_values, xt_values = construct_calibration_data(n_squared)

        self._x = x_values
        self._t = t_values
        self._xt_target = xt_values

    def __call__(self, params: tuple[float, ...]) -> float:
        c0, c1, c2 = self.compute_c(params)
        d0, d1, d2 = self.compute_d(params)
        return self.error(c0, c1, c2, d0, d1, d2)

    def error(self, c0: float, c1: float, c2: float, d0: float, d1: float, d2: float) -> float:
        xt_approx = pow2_approx_array(self._x, self._t, c0, c1, c2, d0, d1, d2)
        return float(np.mean(np.abs(self._xt_target - xt_approx)))

    @staticmethod
    def compute_c(params: tuple[float, ...]) -> tuple[float, float, float]:
        c0, c1, _, _ = params
        return c0, c1, (4 - 2 * c1) / 3

    @staticmethod
    def compute_d(params: tuple[float, ...]) -> tuple[float, float, float]:
        _, _, d0, d1 = params
        return d0, d1, d0 - d1


# =================================================================================================
#  Cost Function - NUMBA
# =================================================================================================
@numba.njit
def pow2_approx_array(
    x_arr: np.ndarray, t_arr: np.ndarray, c0: float, c1: float, c2: float, d0: float, d1: float, d2: float
) -> np.ndarray:
    n = x_arr.shape[0]
    result = np.empty(n, dtype=np.float64)
    for i in range(n):
        log2_x = log2_approx(x_arr[i], c0, c1, c2)
        exp2_tx = exp2_approx(t_arr[i] * log2_x, d0, d1, d2)
        result[i] = exp2_tx
    return result


# @numba.njit
# def log2_approx(x: float, c0: float, c1: float, c2: float) -> float:
#     """Only efficient for x in or close to [0.5, 1]"""
#     if x <= 0:
#         return -1e6
#     if x < 0.5:
#         return log2_approx(2 * x, c0, c1, c2) - 1.0
#     elif x > 1:
#         return log2_approx(x / 2, c0, c1, c2) + 1.0
#     else:
#         # f(x) - but only valid in [0.5, 1]
#         return c0 + (c1 * x) + (c2 * x * x)
#
#
# @numba.njit
# def exp2_approx(x: float, d0: float, d1: float, d2: float) -> float:
#     """Only efficient for x in or close to [0, 1]"""
#     if x < 0:
#         return exp2_approx(x + 1, d0, d1, d2) / 2
#     elif x > 1:
#         return exp2_approx(x - 1, d0, d1, d2) * 2
#     else:
#         # g(x) - but only valid in [0, 1]
#         return d0 + (d1 * x) + (d2 * x * x)


# -------------------------------------------------------------------------
#  Calibration data
# -------------------------------------------------------------------------
@lru_cache(maxsize=4)
def construct_calibration_data(n: int) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """
    Construct calibration data for fast_pow_f32 testing & calibration of coefficients in the form of a
    (x, t, x**t)-tuple of float64 arrays (can be downcast to float32 as needed).

    These data points are intended to be used to ensure / check similarity between x**t and fast_pow_f32(x, t)

    t will be chosen as (1+s)/(1-s) with s chosen uniformly in [-0.9, 0.9]  (hence t in [1/19, 19])
    x will be chosen such that x**t is equally spaced in [0.001, 0.999]

    :param n: (int) size parameter, with resulting arrays of size n^2   (!!!)
    """

    # --- init ----------------------------------
    x_values = np.empty(n * n, dtype=np.float64)
    t_values = np.empty(n * n, dtype=np.float64)
    xt_values = np.empty(n * n, dtype=np.float64)

    # --- construct data ------------------------
    for i, (s, xt) in enumerate(product(np.linspace(-0.9, 0.9, n), np.linspace(0.001, 0.999, n))):
        t = (1.0 + s) / (1.0 - s)
        xt_values[i] = np.float64(xt)
        t_values[i] = np.float64(t)
        x_values[i] = np.float64(xt ** (1.0 / t))

    # --- return --------------------------------
    return x_values, t_values, xt_values
