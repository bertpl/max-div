from functools import lru_cache
from typing import Callable

import numpy as np

from max_div.internal.utils import EPS
from max_div.internal.utils._progress_table import ProgressTable


def minimize_nd(
    fun: Callable[[tuple[float, ...]], float],
    lb: tuple[float, ...],
    ub: tuple[float, ...],
    acc: float = EPS,
    n_grid: int = 9,
    c_reduce: float = 0.5,
) -> tuple[float, ...]:
    """
    Minimize a black-box function in n-dimensions using a grid-like search with iterative bounding box reduction.
    :param fun: Function to minimize. Takes a tuple of floats and returns a float.
    :param lb: Lower bounds for each dimension as a tuple of floats.
    :param ub: Upper bounds for each dimension as a tuple of floats.
    :param acc: accuracy parameter; stop when the largest side of the bounding box is <= acc
    :param n_grid: number of points in each dimension to evaluate per iteration
    :param c_reduce: factor (<1) with which to reduce the bounding box size each iteration, centered around current opt.
    :return: tuple of floats with optimal solution
    """

    # --- init ----------------------------------------------------------------

    # general
    lb_orig, ub_orig = lb, ub
    size = tuple(abs(l - u) for l, u in zip(lb, ub))
    n = len(lb)  # dimensionality
    fun = lru_cache(maxsize=5 * n * n_grid)(fun)  # cache function evaluations
    n_digits = int(-np.log10(acc) + 2)  # nr of digits to display

    # progress table
    progress_table = ProgressTable(
        headers=[
            "it.".rjust(6),
            "max(size)".rjust(n_digits + 3),
            "f_opt".rjust(n_digits + 3),
        ]
        + [f"x[{i}]".rjust(n_digits + 3) for i in range(n)],
    )
    progress_table.show_header()

    # optimal point
    x_opt = tuple((l + u) / 2.0 for l, u in zip(lb, ub))  # center of grid
    f_opt = fun(x_opt)

    # --- main loop -----------------------------------------------------------
    n_iters = 0
    while max(size) > acc:
        # --- progress indication ----------------------------------
        n_iters += 1
        progress_table.show_progress(
            [
                str(n_iters),
                f"{max(size):.{n_digits}f}",
                f"{f_opt:.{n_digits}f}",
            ]
            + [f"{x_opt[i]:.{n_digits}f}" for i in range(n)]
        )

        # --- iterative line-search until no further improvement ---
        new_point_found = True
        while new_point_found:
            # reset flag
            new_point_found = False

            # try to find an improved point by doing line searches along each dimension
            for i in range(n):
                for xi in np.linspace(lb[i], ub[i], n_grid):
                    # build candidate point
                    x_cand_lst = list(x_opt)
                    x_cand_lst[i] = float(xi)
                    x_cand = tuple(x_cand_lst)

                    # evaluate candidate point
                    f_val = fun(x_cand)
                    if f_val < f_opt:
                        x_opt = x_cand
                        f_opt = f_val
                        new_point_found = True

        # --- reduce bounding box ---

        # reduce size
        size = tuple([s * c_reduce for s in size])

        # regenerate box around current optimum, clipped to original bounds
        lb = tuple([max(lb_orig[i], x_opt[i] - 0.5 * size[i]) for i in range(n)])
        ub = tuple([min(ub_orig[i], x_opt[i] + 0.5 * size[i]) for i in range(n)])

    # --- we're done ----------------------------------------------------------
    return x_opt
