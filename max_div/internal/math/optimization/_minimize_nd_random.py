import math
from functools import lru_cache
from typing import Callable

import numpy as np
from numpy import random

from max_div.internal.utils import EPS
from max_div.internal.utils._progress_table import ProgressTable


def minimize_nd_random(
    fun: Callable[[tuple[float, ...]], float],
    lb: tuple[float, ...],
    ub: tuple[float, ...],
    acc: float = EPS,
    n_evals: int = 1000,
) -> tuple[float, ...]:
    """
    Minimize a black-box function in n-dimensions using a grid-like search with iterative bounding box reduction.
    :param fun: Function to minimize. Takes a tuple of floats and returns a float.
    :param lb: Lower bounds for each dimension as a tuple of floats.
    :param ub: Upper bounds for each dimension as a tuple of floats.
    :param acc: Accuracy parameter; stop when the largest side of the bounding box is <= acc
    :param n_evals: Number of function evaluations in total.  Search range will be reduced to reach acc in n_evals.
    :return: tuple of floats with optimal solution
    """

    # --- init ----------------------------------------------------------------

    # general
    random.seed(42)
    lb_orig, ub_orig = lb, ub
    size = tuple(abs(l - u) for l, u in zip(lb, ub))
    n = len(lb)  # dimensionality
    n_digits = int(-np.log10(acc) + 2)  # nr of digits to display
    c_reduce = (acc / max(size)) ** (1.0 / (n_evals - 1))  # reduction factor per iteration

    # progress table
    progress_table = ProgressTable(
        headers=[
            "%".rjust(8),
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
    n_progress_updates = max(10, int(math.cbrt(n_evals)))
    progress_every_n = max(1, n_evals // n_progress_updates)

    for i_iter in range(n_evals):
        # --- progress indication -------------------------
        if i_iter % progress_every_n == 0:
            progress_table.show_progress(
                [
                    f"{i_iter / n_evals:.1%}",
                    f"{max(size):.{n_digits}f}",
                    f"{f_opt:.{n_digits}f}",
                ]
                + [f"{x_opt[i]:.{n_digits}f}" for i in range(n)]
            )

        # --- randomly sample x_cand and try --------------

        # randomly sample in search area
        x_cand = tuple([random.uniform(lb, ub) for lb, ub in zip(lb, ub)])

        # evaluate candidate point
        f_val = fun(x_cand)
        if f_val < f_opt:
            x_opt = x_cand
            f_opt = f_val

        # --- reduce search range ---

        # reduce size
        size = tuple([s * c_reduce for s in size])

        # regenerate box around current optimum, clipped to original bounds
        lb = tuple([max(lb_orig[i], x_opt[i] - 0.5 * size[i]) for i in range(n)])
        ub = tuple([min(ub_orig[i], x_opt[i] + 0.5 * size[i]) for i in range(n)])

    # --- we're done ----------------------------------------------------------
    return x_opt
