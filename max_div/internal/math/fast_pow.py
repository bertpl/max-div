from functools import lru_cache
from itertools import product

import numba
import numpy as np
from numpy.typing import NDArray

# -------------------------------------------------------------------------
#  Constants
# -------------------------------------------------------------------------

# --- float64 ---------------------------------------------

# --- log2(x) ---
# Obtained via minimax polynomial fitting over [0.5, 1.0] with additional
# constraint of having an exact fit at 0.5 and 1.0
# See: ./notebooks/poly_approx_pow.ipynb
_D_L0 = -2.6080969294048635
_D_L1 = 3.8242907882145905
_D_L2 = -1.216193858809727


# --- exp2(x) ---
# Obtained via minimax polynomial fitting over [0.0, 1.0] with additional
# constraint of having an exact fit at 0.0 and 1.0
# See: ./notebooks/poly_approx_pow.ipynb
_D_E0 = 1.0
_D_E1 = 0.6700844332949878
_D_E2 = 0.32991556670501215


# NOTE:
#  given the above boundary contraints of having an exact fit for log2(1) and exp2(0),
#  it is guaranteed that exp2_approx(t * log2_approx(1.0)) == 1.0 for any t

# --- float32 ---------------------------------------------

# --- log2(x) ---
_S_L0 = np.float32(_D_L0)
_S_L1 = np.float32(_D_L1)
_S_L2 = np.float32(_D_L2)

# --- exp2(x) ---
_S_E0 = np.float32(_D_E0)
_S_E1 = np.float32(_D_E1)
_S_E2 = np.float32(_D_E2)


# -------------------------------------------------------------------------
#  Fast approximations for pow (x^t)
# -------------------------------------------------------------------------
@numba.njit(numba.float32(numba.float32, numba.float32), fastmath=True, inline="always")
def fast_pow_f32(x: np.float32, t: np.float32) -> np.float32:
    """
    Fast 'pow' approximation using 2nd order polynomial after range reduction.

    Approximation coefficients have been calibrated to minimize max. absolute error for...
       -->  x in [0.001, 0.999]
       -->  t in [0.05, 20.0]
    """

    # ---------------------------------------------------------------
    #  Approximate log2(x)
    # ---------------------------------------------------------------

    # --- extract mantissa & exponent ---------------------
    # exponent
    xi = np.int32(np.float32(x).view(np.int32))
    exponent = ((xi >> 23) & 0xFF) - 126
    # mantissa
    xi = (xi & 0x007FFFFF) | 0x3F000000
    m = np.int32(xi).view(np.float32)

    # --- polynomial approximation ------------------------
    log2_mantissa = _S_L0 + m * (_S_L1 + m * _S_L2)

    # compute log2(x) = exponent + log2(mantissa)
    approx_log2 = np.float32(exponent) + log2_mantissa

    # ---------------------------------------------------------------
    #  Raise to power t in log space
    # ---------------------------------------------------------------
    y = t * approx_log2  # approximation for t*log2(x)

    # ---------------------------------------------------------------
    #  Approximate exp2(y)
    # ---------------------------------------------------------------

    # --- split y in int + fraction -----------------------
    k = np.floor(y)  # float32
    f = y - k  # fraction f is in [0, 1)

    # --- polynomial approximation ------------------------
    exp2_f = _S_E0 + f * (_S_E1 + f * _S_E2)

    # --- combine parts -----------------------------------
    return np.float32(np.ldexp(exp2_f, np.int32(k)))


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
