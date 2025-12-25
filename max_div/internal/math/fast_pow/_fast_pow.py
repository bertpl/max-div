import numba
import numpy as np

from max_div.internal.math.powers_of_2 import power_of_2_f32

# -------------------------------------------------------------------------
#  Constants
# -------------------------------------------------------------------------

# --- float64 ---------------------------------------------

# --- log2(x) ---
# Obtained via minimax polynomial fitting over [0.5, 1.0] with additional
# constraint of having an exact fit at 0.5 and 1.0
# See: --> ./notebooks/calibrate_fast_pow.ipynb
#      --> max_div/internal/math/fast_pow/_calibration.py
_D_L0 = -2.63265442228959800630
_D_L1 = 3.89271942562166550772
_D_L2 = -1.26181295041444374583


# --- exp2(x) ---
# Obtained via minimax polynomial fitting over [0.0, 1.0] with additional
# constraint of having an exact fit at 0.0 and 1.0
# See: --> ./notebooks/calibrate_fast_pow.ipynb
#      --> max_div/internal/math/fast_pow/_calibration.py

_D_E0 = 0.99917715862992750875
_D_E1 = 0.67296905274855078893
_D_E2 = 0.32620810588137671981


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
    return exp2_f * power_of_2_f32(np.int32(k))
