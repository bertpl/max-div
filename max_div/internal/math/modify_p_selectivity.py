"""
Module with functionality to modify 'p' (probability) values (e.g. used in sampling algorithms) to
increase or decrease selectivity of sampling.

modifier in [-1, 1]:
   -1  -->  remove all selectivity, transforming p to uniform distribution
    0 -->   leave p unchanged
   +1  -->  increase selectivity such that only the highest p value remains non-zero

Requirements:
  - modifications should be invertible by applying the negative modifier again.
  - the vector p should not be treated as strict probabilities, i.e. they should not sum up to 1, necessarily.
  - modifiers will act such that the range [0, max(p)] is preserved.
  - original p arrays are not modified in-place; a new modified array is returned.
"""

import numpy as np
from numba import njit
from numpy.typing import NDArray


# =================================================================================================
#  Helpers
# =================================================================================================
@njit("float32(float32[::1])", fastmath=True, inline="always")
def _p_max(p: NDArray[np.float32]) -> np.float32:
    """Return the maximum value in p array."""
    n = p.size
    max_value = np.float32(0.0)
    for i in range(n):
        max_value = max(max_value, p[i])
    return max_value


@njit("float32[::1](float32[::1])", fastmath=True, inline="always")
def _uniform(p: NDArray[np.float32]) -> NDArray[np.float32]:
    """Transform p array to uniform distribution (all values equal to max(p))."""

    # --- detect max value ---
    n = p.size
    p_max = _p_max(p)

    # --- return uniform ---
    return np.full(n, p_max, dtype=np.float32)


@njit("float32[::1](float32[::1])", fastmath=True, inline="always")
def _max_selective(p: NDArray[np.float32]) -> NDArray[np.float32]:
    """Transform p array to uniform distribution (all values equal to max(p))."""

    # --- detect max value ---
    n = p.size
    p_max = _p_max(p)

    # --- return max-selective p ---
    p_modified = np.zeros(n, dtype=np.float32)
    for i in range(n):
        if p[i] == p_max:
            p_modified[i] = p_max

    return p_modified


# =================================================================================================
#  Selectivity modifiers
# =================================================================================================
@njit("float32[::1](float32[::1], float32)", fastmath=True, inline="always")
def modify_p_selectivity_power(p: NDArray[np.float32], modifier: np.float32) -> NDArray[np.float32]:
    """Modify the p array by applying a selectivity modification.

    Each element p[i] is transformed to max(p) * ((p[i] / max(p)) ** t) with ...
       --> t = (1 + modifier)/(1-modifier)

    This way...
       --> it can be computed that area under the curve for p**t for p in [0,1] is 1/(t+1) = (1-modifier)/2
       --> modifier = -1  -->  t = 0       -->  (uniform distribution)
       --> modifier = 0   -->  t = 1       -->  (unchanged)
       --> modifier = +1  -->  t = +infty  -->  (max-selective)
       --> applying -modifier reverts the modification

    NOTE: returned probabilities array is not normalized.

    :param p: (1D array) Original p values.
    :param modifier: (float) The selectivity modifier in [-1,1] to apply.
    :return: (1D array) Modified p values.
    """

    # --- detect shortcuts --------------------------------
    if modifier <= -1.0:
        return _uniform(p)
    elif modifier == 0.0:
        return p.copy()
    elif modifier >= 1.0:
        return _max_selective(p)

    # --- prep transformation -----------------------------
    n = p.size
    p_max = _p_max(p)
    if p_max <= 0.0:
        return p.copy()
    else:
        p_max_inv = 1.0 / p_max
        t = (1 + modifier) / (1 - modifier)

    # --- transform ---------------------------------------
    p_modified = np.empty_like(p)
    for i in range(n):
        p_modified[i] = p_max * ((p[i] * p_max_inv) ** t)

    return p_modified


@njit("float32[::1](float32[::1], float32)", fastmath=True, inline="always")
def modify_p_selectivity_pwl2(p: NDArray[np.float32], modifier: np.float32) -> NDArray[np.float32]:
    r"""
    This method modifies the selectivity of the p array using a piecewise linear approach with 2 linear segments.
    This method serves as a faster, but approximate alternative to the power-based method `modify_p_selectivity_power`.

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
            |       /         ----(X)      |              same way as in `modify_p_selectivity_power`:
            |    /    -------         \    |
            | / -----                    \ |                  --> t = (1 + modifier)/(1-modifier)
            +------------------------------+
       (0,0)             p[i]               (1,0)


    DERIVATION:

        --> area under the curve for f(x) = 1-r
        --> area under the curve for x^t  = 1/(t+1) = (1-modifier)/2  (see derivation in `modify_p_selectivity_power`)

        Hence, we choose r such that 1-r = (1-modifier)/2
                                  -->  r = (1+modifier)/2, the area under the curve is identical for both methods.

    NOTE: returned probabilities array is not normalized.

    :param p: (1D array) Original p values.
    :param modifier: (float) The selectivity modifier in [-1,1] to apply.
    :return: (1D array) Modified p values.
    """

    # --- detect shortcuts --------------------------------
    r = 0.5 * (1.0 + modifier)
    if (modifier <= -1.0) or (r <= 0.0):  # double condition, to account for potential rounding errors
        return _uniform(p)
    elif (modifier == 0.0) or (r == 0.0):
        return p.copy()
    elif (modifier >= 1.0) or (r >= 1.0):
        return _max_selective(p)

    # --- prep transformation -----------------------------
    n = p.size
    p_max = _p_max(p)
    if p_max <= 0.0:
        return p.copy()
    else:
        # map to coordinates in absolute terms (0,0), (r_mod, p_max-r_mod), (p_max,p_max)
        r_mod = r * p_max
        c0 = (p_max - r_mod) / r_mod  # slope of first segment
        c1 = r_mod / (p_max - r_mod)  # slope of second segment

    # --- transform ---------------------------------------
    p_modified = np.empty_like(p)
    for i in range(n):
        pi = p[i]
        if pi <= r_mod:
            p_modified[i] = c0 * pi  # linear segment 1
        else:
            p_modified[i] = p_max - c1 * (p_max - pi)  # linear segment 2

    return p_modified
