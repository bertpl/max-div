import math

from max_div.internal.math.optimization import minimize_nd


def test_minimize_nd():
    # --- arrange -----------------------------------------
    x_opt_true = (math.e, math.pi, math.cbrt(3))

    def fun(x: tuple[float, ...]) -> float:
        return math.sqrt(sum((xi - xi_opt_true) ** 2 for xi, xi_opt_true in zip(x, x_opt_true)))

    # --- act ---------------------------------------------
    x_opt = minimize_nd(
        fun=fun,
        lb=(0.0, 0.0, 0.0),
        ub=(10.0, 10.0, 10.0),
        acc=1e-10,
        n_grid=4,
        c_reduce=0.5,
    )

    # --- assert ------------------------------------------
    for target, actual in zip(x_opt_true, x_opt):
        assert abs(target - actual) <= 1e-10
