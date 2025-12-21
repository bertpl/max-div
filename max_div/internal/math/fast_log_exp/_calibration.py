import numba
import numpy as np

from max_div.internal.math.optimization import minimize_nd, minimize_nd_random


# =================================================================================================
#  Main Calibration Function
# =================================================================================================
def calibrate_fast_log_exp(
    n_data: int, acc: float, n_evals: int = 10_000
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """
    Jointly calibrates the quadratic coefficients of fast_log2 and fast_exp2 to minimize the overall error.

    PARAMETERS
    ----------

      --> FAST_LOG2

            - we look for an expression of the form `f(x) ≈ c0 + c1*x + c2*x^2`   (over [0.5, 1])
            - such that `f(0.5) = f(1) - 1`    (continuous)
            - such that `f'(0.5) = 2*f'(1)`    (smooth)
            - hence `c0=k1, c1=4, c2=-4/3` with `k1` a free parameter

            --> free parameter k1 with optimal value expected to be ~ in range [-3.0, -2.0]

        --> FAST_EXP2

            - we look for an expression of the form `g(x) ≈ d0 + d1*x + d2*x^2`   (over [0, 1])
            - such that `g(0) = 0.5*g(1)`      (continuous)
            - such that `g'(0) = 0.5*g'(1)`    (smooth)
            - hence `d0=3*k2, d1=2*k2, d2=k2` with `k2` a free parameter

            --> free parameter k2 with optimal value expected to be ~ in range [0.0, 0.5]

    OPTIMIZATION GOAL
    -----------------

    We want to optimize k1, k2 such that we minimize the sum of following 3 errors:

        - `e_log` is maximum error of `f(x)` wrt `log2` over [0.5, 1]
        - `e_exp` is maximum error of `g(x)` wrt `exp2` over [0, 1]
        - `e_log_exp` is maximum error of `g(f(x)+1)/2` wrt `x` over [0.5, 1]

    """

    cost_fun = Cost(n=n_data)
    # k1_k2_opt = minimize_nd(
    #     fun=cost_fun,
    #     lb=(-3.0, 0.0),
    #     ub=(-2.0, 1.0),
    #     acc=acc,
    #     n_grid=9,
    #     c_reduce=0.6,
    # )
    k1_k2_opt = minimize_nd_random(
        fun=cost_fun,
        lb=(-3.0, 0.0),
        ub=(-2.0, 1.0),
        acc=acc,
        n_evals=n_evals,
        # n_grid=9,
        # c_reduce=0.6,
    )

    c0, c1, c2 = cost_fun.compute_c(k1_k2_opt[0])
    d0, d1, d2 = cost_fun.compute_d(k1_k2_opt[1])

    print()
    print()
    print(f"(c0, c1, c2) = ({c0:.18f}, {c1:.18f}, {c2:.18f})")
    print(f"(d0, d1, d2) = ({d0:.18f}, {d1:.18f}, {d2:.18f})")
    print()
    print(f"e_log = {cost_fun.e_log(c0, c1, c2):.6f}    (=max abs error)")
    print(f"e_exp = {cost_fun.e_exp(d0, d1, d2):.6f}    (=max rel error)")
    print(f"e_log_exp = {cost_fun.e_log_exp(c0, c1, c2, d0, d1, d2):.6f}")
    print()
    print("--- FAST LOG -----------------------------------")
    print(f"_D20 = {c0:.20f}")
    print(f"_D21 = {c1:.20f}")
    print(f"_D22 = {c2:.20f}")
    print("------------------------------------------------")
    print()
    print("--- FAST EXP -----------------------------------")
    print(f"_D20 = {d0:.20f}")
    print(f"_D21 = {d1:.20f}")
    print(f"_D22 = {d2:.20f}")
    print("------------------------------------------------")

    return (c0, c1, c2), (d0, d1, d2)


# =================================================================================================
#  Cost Function
# =================================================================================================
class Cost:
    def __init__(self, n: int):
        self._x_log2 = np.linspace(0.5, 1.0, n)
        self._x_exp2 = np.linspace(0.0, 1.0, n)
        self._x_log_exp = np.linspace(0.5, 1.0, n)

        self._fx_log2 = np.array([np.log2(x) for x in self._x_log2])
        self._fx_exp2 = np.array([np.exp2(x) for x in self._x_exp2])

    def __call__(self, k1_k2: tuple[float, float]) -> float:
        c0, c1, c2 = self.compute_c(k1_k2[0])
        d0, d1, d2 = self.compute_d(k1_k2[1])

        e_log = self.e_log(c0, c1, c2)
        e_exp = self.e_exp(d0, d1, d2)
        e_log_exp = self.e_log_exp(c0, c1, c2, d0, d1, d2)

        return e_log + e_exp + e_log_exp

    def e_log(self, c0: float, c1: float, c2: float) -> float:
        """Max. abs error over [0.5, 1.0] -> translates to same max abs error over entire range"""
        return float(np.max(np.abs(self._fx_log2 - log2_approx_array(self._x_log2, c0, c1, c2))))

    def e_exp(self, d0: float, d1: float, d2: float) -> float:
        """Max. rel error over [0.0, 1.0] -> translates to same max rel error over entire range"""
        return float(np.max(np.abs(self._fx_exp2 - exp2_approx_array(self._x_exp2, d0, d1, d2)) / self._fx_exp2))

    def e_log_exp(self, c0: float, c1: float, c2: float, d0: float, d1: float, d2: float) -> float:
        """Max. abs error over [0.5, 1.0]"""
        return float(
            np.max(
                np.abs(
                    self._x_log_exp
                    - exp2_approx_array(log2_approx_array(self._x_log_exp, c0, c1, c2) + 1.0, d0, d1, d2) / 2.0
                )
            )
        )

    @staticmethod
    def compute_c(k1: float) -> tuple[float, float, float]:
        return k1, 4.0, -4.0 / 3.0

    @staticmethod
    def compute_d(k2: float) -> tuple[float, float, float]:
        return 3.0 * k2, 2.0 * k2, k2


# =================================================================================================
#  Cost Function - NUMBA
# =================================================================================================
@numba.njit
def log2_approx_array(x_arr: np.ndarray, c0: float, c1: float, c2: float) -> np.ndarray:
    n = x_arr.shape[0]
    result = np.empty(n, dtype=np.float64)
    for i in range(n):
        result[i] = log2_approx(x_arr[i], c0, c1, c2)
    return result


@numba.njit
def exp2_approx_array(x_arr: np.ndarray, c0: float, c1: float, c2: float) -> np.ndarray:
    n = x_arr.shape[0]
    result = np.empty(n, dtype=np.float64)
    for i in range(n):
        result[i] = exp2_approx(x_arr[i], c0, c1, c2)
    return result


@numba.njit(inline="always")
def log2_approx(x: float, c0: float, c1: float, c2: float) -> float:
    """Only efficient for x in or close to [0.5, 1]"""
    if x <= 0:
        return -1e6
    else:
        offset = 0.0

        # make sure x is in [0.5, 1.0]
        while x < 0.5:
            x = 2 * x
            offset -= 1.0
        while x > 1.0:
            x = x * 0.5
            offset += 1.0

        # formula valid in [0.5, 1]
        return offset + (c0 + (c1 * x) + (c2 * x * x))


@numba.njit(inline="always")
def exp2_approx(x: float, d0: float, d1: float, d2: float) -> float:
    """Only efficient for x in or close to [0, 1]"""

    # make sure x is in [0.0, 1.0]
    factor = 1.0
    while x < 0:
        x += 1.0
        factor = factor * 0.5
    while x > 1:
        x -= 1.0
        factor = factor * 2.0

    # formula valid in [0, 1]
    return factor * (d0 + (d1 * x) + (d2 * x * x))
