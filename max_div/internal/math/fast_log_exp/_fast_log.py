import numba
import numpy as np

# -------------------------------------------------------------------------
#  Constants
# -------------------------------------------------------------------------

# --- float64 ---------------------------------------------
_D_LOG_2 = 0.6931471805599453  # np.log(2)

# Obtained via minimax polynomial fitting over [0.5, 1.0) with additional
# continuity & smoothness constraints imposed on boundary points.
# See: --> ./notebooks/calibrate_fast_log2_exp2.ipynb
#      --> max_div/internal/math/fast_log_exp/_calibration.py
_D20 = -2.66448308361168395564
_D21 = 4.00000000000000000000
_D22 = -1.33333333333333325932

# --- float32 ---------------------------------------------
_S_LOG_2 = np.float32(_D_LOG_2)

_S20 = np.float32(_D20)
_S21 = np.float32(_D21)
_S22 = np.float32(_D22)


# -------------------------------------------------------------------------
#  Fast approximations for np.log2
# -------------------------------------------------------------------------
@numba.njit(numba.float64(numba.float64), fastmath=True, inline="always")
def fast_log2_f64(x: np.float64) -> np.float64:
    """
    Fast log approximation using 2nd order polynomial after range reduction.
    (max abs error ~0.0075 over entire range.)
    """

    # --- extract mantissa & exponent ---------------------
    # exponent
    xi = np.int64(np.float64(x).view(np.int64))
    exponent = ((xi >> 52) & 0x7FF) - 1022
    # mantissa
    xi = (xi & 0x000FFFFFFFFFFFFF) | 0x3FE0000000000000
    m = np.int64(xi).view(np.float64)  # in range [0.5, 1.0)

    # --- polynomial approximation ------------------------
    log2_mantissa = _D20 + m * (_D21 + m * _D22)

    # Return log2(x) = exponent + log2(m)
    return exponent + log2_mantissa


@numba.njit(numba.float32(numba.float32), fastmath=True, inline="always")
def fast_log2_f32(x: np.float32) -> np.float32:
    """
    Fast log approximation using 2nd order polynomial after range reduction.
    (max abs error ~0.0075 over entire range.)
    """

    # --- extract mantissa & exponent ---------------------
    # exponent
    xi = np.int32(np.float32(x).view(np.int32))
    exponent = ((xi >> 23) & 0xFF) - 126
    # mantissa
    xi = (xi & 0x007FFFFF) | 0x3F000000
    m = np.int32(xi).view(np.float32)

    # --- polynomial approximation ------------------------
    log2_mantissa = _S20 + m * (_S21 + m * _S22)

    # Return log2(x) = exponent + log2(mantissa)
    return exponent + log2_mantissa


# -------------------------------------------------------------------------
#  Fast approximations for np.log
# -------------------------------------------------------------------------
@numba.njit(numba.float64(numba.float64), fastmath=True, inline="always")
def fast_log_f64(x: np.float64) -> np.float64:
    """
    Fast log approximation using 2nd order polynomial after range reduction.
    (max abs error ~0.0052 over entire range)
    """
    return _D_LOG_2 * fast_log2_f64(x)


@numba.njit(numba.float32(numba.float32), fastmath=True, inline="always")
def fast_log_f32(x: np.float32) -> np.float32:
    """
    Fast log approximation using 2nd order polynomial after range reduction.
    (max abs error ~0.0052 over entire range)
    """
    return _S_LOG_2 * fast_log2_f32(x)


# =================================================================================================
#  Public API
# =================================================================================================
__ALL__ = [
    "fast_log2_f32",
    "fast_log2_f32",
    "fast_log_f64",
    "fast_log_f64",
]
