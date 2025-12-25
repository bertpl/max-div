import math

import numba
import numpy as np

from max_div.internal.math.powers_of_2 import power_of_2_f32

# -------------------------------------------------------------------------
#  Constants
# -------------------------------------------------------------------------

# --- float64 ---------------------------------------------
_D_LOG2_E = 1.4426950408889634  # np.log2(e)

# Obtained via minimax polynomial fitting over [0.0, 1.0) with additional
# continuity & smoothness constraints imposed on boundary points.
# See: --> ./notebooks/calibrate_fast_log2_exp2.ipynb
#      --> max_div/internal/math/fast_log_exp/_calibration.py
_D20 = 0.99922823313725395167
_D21 = 0.66615215542483596778
_D22 = 0.33307607771241798389

# --- float32 ---------------------------------------------
_S_LOG2_E = np.float32(_D_LOG2_E)

_S20 = np.float32(_D20)
_S21 = np.float32(_D21)
_S22 = np.float32(_D22)


# -------------------------------------------------------------------------
#  Fast approximations for np.log2
# -------------------------------------------------------------------------
@numba.njit(numba.float64(numba.float64), fastmath=True, inline="always")
def fast_exp2_f64(x: np.float64) -> np.float64:
    """
    Fast exp approximation using 2nd order polynomial after range reduction.
    (max rel error ~0.0026 over entire range.)
    """

    # --- split in int + fraction -------------------------
    k = np.floor(x)  # float64
    f = x - k  # f is in [0, 1)

    # --- polynomial approximation ------------------------
    exp2_f = _D20 + f * (_D21 + f * _D22)

    # --- combine parts -----------------------------------
    return np.float64(math.ldexp(exp2_f, int(k)))


@numba.njit(numba.float32(numba.float32), fastmath=True, inline="always")
def fast_exp2_f32(x: np.float32) -> np.float32:
    """
    Fast log approximation using 2nd order polynomial after range reduction.
    (max rel error ~0.0026 over entire range.)
    """

    # --- split in int + fraction -------------------------
    k = np.floor(x)  # float32
    f = x - k  # f is in [0, 1)

    # --- polynomial approximation ------------------------
    exp2_f = _S20 + f * (_S21 + f * _S22)

    # --- combine parts -----------------------------------
    return exp2_f * power_of_2_f32(np.int32(k))


# -------------------------------------------------------------------------
#  Fast approximations for np.exp
# -------------------------------------------------------------------------
@numba.njit(numba.float64(numba.float64), fastmath=True, inline="always")
def fast_exp_f64(x: np.float64) -> np.float64:
    """
    Fast exp approximation using 2nd order polynomial after range reduction.
    (max rel error ~0.0026 over entire range.)
    """
    return fast_exp2_f64(_D_LOG2_E * x)


@numba.njit(numba.float32(numba.float32), fastmath=True, inline="always")
def fast_exp_f32(x: np.float32) -> np.float32:
    """
    Fast exp approximation using 2nd order polynomial after range reduction.
    (max rel error ~0.0026 over entire range.)
    """
    return fast_exp2_f32(_S_LOG2_E * x)


# =================================================================================================
#  Public API
# =================================================================================================
__ALL__ = [
    "fast_exp2_f32",
    "fast_exp2_f32",
    "fast_exp_f64",
    "fast_exp_f64",
]
