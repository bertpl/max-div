import numpy as np
from numba import njit
from numpy.typing import NDArray

from max_div._core._math.fast_log_exp import fast_exp2_f32

# The floor of the transformed range that the solver's sampling paths use.  It lives here rather
# than as a parameter default because an eager signature declares an exact arity, so a call that
# omits an argument has no matching definition; callers that want another floor still pass one.
DEFAULT_LOW_VALUE = np.float32(0.1)


# Defined ahead of its caller: the caller declares a signature, so it compiles when the
# decorator runs and every function it calls has to exist by then.
@njit(fastmath=True, inline="always", cache=True)
def _exponential_transform(
    p_in: NDArray[np.float32],
    p_out: NDArray[np.float32],
    modifier: np.float32,
    reverse: bool,
    low_value: np.float32,
    p_min: np.float32,
    p_max: np.float32,
    p_range: np.float32,
) -> None:
    """Apply the exponential transform for finite, non-degenerate p_in (see exponential_selectivity)."""
    n = p_in.shape[0]

    # precompute values
    t = (np.float32(1.0) + modifier) / (np.float32(1.0) - modifier)
    p_range_inv = np.float32(1.0) / p_range
    log2_low_value = np.float32(np.log2(low_value))
    t_times_log2_low_value = t * log2_low_value

    # main loop
    if reverse:
        for i in range(n):
            exponent = (p_in[i] - p_min) * p_range_inv
            p_out[i] = fast_exp2_f32(exponent * t_times_log2_low_value)
    else:
        for i in range(n):
            exponent = (p_max - p_in[i]) * p_range_inv
            p_out[i] = fast_exp2_f32(exponent * t_times_log2_low_value)


@njit("void(float32[::1], float32[::1], float32, boolean, float32)", inline="always", cache=True)
def exponential_selectivity(
    p_in: NDArray[np.float32],
    p_out: NDArray[np.float32],
    modifier: np.float32,
    reverse: bool,
    low_value: np.float32,
) -> None:
    """Populate p_out with values depending exponentially on the non-normalized probabilities in p_in.

    p_in is a float32-array of shape (n,) containing non-normalized probabilities in range [p_min, p_max].
    p_out is populated with values exponentially depending on the corresponding p_in values, such that p_out values
    all lie in the interval [low_value**t, 1.0], where t is computed as in the other modification methods:

            t = (1.0 + modifier) / (1.0 - modifier)

    This then boils down to the following formulas:

     - reverse=False    p_out[i] = low_value ** (t * (p_max - p_in[i]) / (p_max - p_min))
     - reverse=True     p_out[i] = low_value ** (t * (p_in[i] - p_min) / (p_max - p_min))

    The approximate function fast_exp2(exponent * np.log2(base)) is used to compute the exponentiation efficiently,
      taking into account that np.log2(low_value) can be precomputed outside the loop.

    Args:
        p_in: np.ndarray of shape (n,) containing the original probabilities.
        p_out: np.ndarray of shape (n,) to be populated with the transformed probabilities.
        modifier: float32 in (-1, 1) indicating how to modify selectivity
        reverse: (bool) if True, higher p_in values result in lower p_out values
        low_value: (float32) base of the exponential
    """
    # --- init ----------------------------------
    n = p_in.shape[0]
    p_min = np.float32(np.inf)
    p_max = np.float32(-np.inf)
    for i in range(n):
        pi = p_in[i]
        p_min = min(p_min, pi)
        p_max = max(p_max, pi)
    p_range = p_max - p_min

    # --- corner case ---------------------------
    # Degenerate ranges fall back to uniform: all-equal inputs (p_range == 0) and
    # non-finite inputs (e.g. the +inf contribution of a sole selected item, which
    # makes p_range inf or nan) — the transform would emit NaNs for those.
    # NOTE: this guard is why the function is not compiled with fastmath: fastmath
    # lets LLVM assume no NaN/inf exists and optimize the isfinite check away.
    if (p_range == 0.0) or (not np.isfinite(p_range)):
        for i in range(n):
            p_out[i] = np.float32(1.0)
        return

    # --- actual transformation -----------------
    _exponential_transform(p_in, p_out, modifier, reverse, low_value, p_min, p_max, p_range)
