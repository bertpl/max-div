"""
This module contains all methods that actually modify the p selectivity in different ways.

All the methods here are numba-accelerated and modify in-place for performance reasons
  (allocating new np arrays in numba is slow).

These methods are called by the main `modify_p_selectivity` and `modify_p_selectivity_inplace` functions.
"""

import numpy as np
from numba import njit
from numpy.typing import NDArray

from max_div.internal.math.fast_log_exp import fast_exp2_f32, fast_log2_f32
from max_div.internal.math.fast_pow import fast_pow_f32


# =================================================================================================
#  Boundary methods
# =================================================================================================
@njit("void(float32[::1])", fastmath=True, inline="always")
def _uniform(p: NDArray[np.float32]):
    """Transform p in [0,1] in-place to uniform distribution (all values equal to 1.0)."""

    # --- fill with 1.0 values ---
    for i in range(p.size):
        p[i] = np.float32(1.0)


@njit("void(float32[::1])", fastmath=True, inline="always")
def _max_selective(p: NDArray[np.float32]):
    """Transform p in [0,1] in-place to maximally selective distribution (all values equal to 0.0 or 1.0)."""

    # --- fill with 1.0 or 0.0 ---
    for i in range(p.size):
        p[i] = np.float32(1.0) if (p[i] >= np.float32(1.0)) else np.float32(0.0)


# =================================================================================================
#  Regular methods - POWER-based
# =================================================================================================
@njit("float32[::1](float32[::1], float32)", fastmath=True, inline="always")
def _power_exact(p: NDArray[np.float32], modifier: np.float32) -> NDArray[np.float32]:
    """
    Modify p in [0,1] in-place using exact p[i] <-- p[i] ** t.
    with...   t = (1 + modifier)/(1 - modifier)
              modifier in (-1, +1)
    """
    t = (1 + modifier) / (1 - modifier)
    for i in range(p.size):
        p[i] = p[i] ** t
    return p


@njit("float32[::1](float32[::1], float32)", fastmath=True, inline="always")
def _power_fast_log2_exp2(p: NDArray[np.float32], modifier: np.float32) -> NDArray[np.float32]:
    """
    Modify p in [0,1] in-place using approximation p[i] <-- fast_exp2(t * fast_log2(p[i])).
    with...   t = (1 + modifier)/(1 - modifier)
              modifier in (-1, +1)
    """
    t = (1 + modifier) / (1 - modifier)
    for i in range(p.size):
        p[i] = fast_exp2_f32(t * fast_log2_f32(p[i]))
    return p


@njit("float32[::1](float32[::1], float32)", fastmath=True, inline="always")
def _power_fast_pow(p: NDArray[np.float32], modifier: np.float32) -> NDArray[np.float32]:
    """
    Modify p in [0,1] in-place using approximation p[i] <-- fast_pow(p[i], t).
    with...   t = (1 + modifier)/(1 - modifier)
              modifier in (-1, +1)
    """
    t = (1 + modifier) / (1 - modifier)
    for i in range(p.size):
        p[i] = fast_pow_f32(p[i], t)
    return p


# =================================================================================================
#  Regular methods - PWL-based
# =================================================================================================
@njit("float32[::1](float32[::1], float32)", fastmath=True, inline="always")
def _pwl_2_segment(p: NDArray[np.float32], modifier: np.float32) -> NDArray[np.float32]:
    r"""
    Modify p in [0,1] in-place using 2-segment piecewise linear approximation of p[i] = p[i] ** t.

            Assuming for simplicity that max(p)==1.0, the transformation f(p[i]) used here is defined as follows:

       (0,1)                                (1,1)
            +------------------------------+
            | \                          //|              We construct a piecewise linear function
            |    \                    /  / |              with nodes at (0,0), (r, 1-r), (1,1). The node at (r, 1-r) is
            |       \              /    |  |              depicted as '(X)' in the diagram, for r~=0.75
            |          \        /      /   |
    f(p[i]) |             \  /        |    |
            |             /  \       /     |              The parameter 'r' is chosen such that the area under the curve
            |          /        \   |      |              is identical to g(p[i]) = p[i] ** t, where t is chosen in the
            |       /         ----(X)      |              same way as in power-based methods:
            |    /    -------         \    |                 (see: `modify_p_selectivity`)
            | / -----                    \ |
            +------------------------------+                        --> t = (1 + modifier)/(1-modifier)
       (0,0)             p[i]               (1,0)


    DERIVATION:

        --> area under the curve for f(x) = 1-r
        --> area under the curve for x^t  = 1/(t+1) = (1-modifier)/2  (see derivation in `modify_p_selectivity`)

        Hence, choosing r such that 1-r = (1-modifier)/2
                                 -->  r = (1+modifier)/2, the area under the curve is identical for both methods.
    """
    # compute r & slopes of 2 segments
    r = 0.5 * (1.0 + modifier)
    c0 = (1.0 - r) / r  # slope of first segment
    c1 = r / (1.0 - r)  # slope of second segment

    # actual transformation
    n = p.size
    if r >= 0.5:
        # convex situation as depicted above -> take max. of 2 segments
        for i in range(n):
            pi = p[i]
            p[i] = max(
                c0 * pi,  # linear segment 1
                1.0 - c1 * (1.0 - pi),  # linear segment 2
            )
    else:
        # concave situation -> take min. of 2 segments
        for i in range(n):
            pi = p[i]
            p[i] = min(
                c0 * pi,  # linear segment 1
                1.0 - c1 * (1.0 - pi),  # linear segment 2
            )

    return p
