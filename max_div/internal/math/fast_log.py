import numba
import numpy as np

# -------------------------------------------------------------------------
#  Constants
# -------------------------------------------------------------------------

# --- float64 ---------------------------------------------
_D_LOG_2 = 0.6931471805599453  # np.log(2)

_D20 = -2.664030016771488
_D21 = 4.018729576130520
_D22 = -1.359007257261774

_D30 = -3.144940630924
_D31 = 6.058956424048
_D32 = -4.157032648692
_D33 = 1.243566840636

_D40 = -3.505614661980
_D41 = 8.099233785172
_D42 = -8.397609124753
_D43 = 5.084088932163
_D44 = -1.280174030509

_D50 = -3.794153676536
_D51 = 10.139512633266
_D52 = -14.080875352582
_D53 = 12.881420375173
_D54 = -6.551609372263
_D55 = 1.405716091134

# --- float32 ---------------------------------------------
_S_LOG_2 = np.float32(_D_LOG_2)

_S20 = np.float32(_D20)
_S21 = np.float32(_D21)
_S22 = np.float32(_D22)

_S30 = np.float32(_D30)
_S31 = np.float32(_D31)
_S32 = np.float32(_D32)
_S33 = np.float32(_D33)

_S40 = np.float32(_D40)
_S41 = np.float32(_D41)
_S42 = np.float32(_D42)
_S43 = np.float32(_D43)
_S44 = np.float32(_D44)

_S50 = np.float32(_D50)
_S51 = np.float32(_D51)
_S52 = np.float32(_D52)
_S53 = np.float32(_D53)
_S54 = np.float32(_D54)
_S55 = np.float32(_D55)


# -------------------------------------------------------------------------
#  Fast approximations for np.log
# -------------------------------------------------------------------------
@numba.njit(fastmath=True, inline="always")
def fast_log_f64_poly(x: np.float64, degree: int) -> np.float64:
    """
    Fast log approximation using polynomial after range reduction.
    Accuracy depends on degree:
        degree=2: max abs error ~0.004   over entire range.
        degree=3: max abs error ~0.0005  over entire range.
        degree=4: max abs error ~0.00007 over entire range.
        degree=5: max abs error ~0.00001 over entire range.
    """
    return _D_LOG_2 * fast_log2_f64_poly(x, degree)


@numba.njit(fastmath=True, inline="always")
def fast_log_f32_poly(x: np.float32, degree: int) -> np.float32:
    """
    Fast log approximation using polynomial after range reduction.
    Accuracy depends on degree:
        degree=2: max abs error ~0.004   over entire range.
        degree=3: max abs error ~0.0005  over entire range.
        degree=4: max abs error ~0.00007 over entire range.
        degree=5: max abs error ~0.00001 over entire range.
    """
    return _S_LOG_2 * fast_log2_f32_poly(x, degree)


# -------------------------------------------------------------------------
#  Fast approximations for np.log2
# -------------------------------------------------------------------------
@numba.njit(fastmath=True, inline="always")
def fast_log2_f64_poly(x: np.float64, degree: int) -> np.float64:
    """
    Fast log2 approximation using polynomial after range reduction.
    Accuracy depends on degree:
        degree=2: max abs error ~0.006    over entire range.
        degree=3: max abs error ~0.0007   over entire range.
        degree=4: max abs error ~0.0001   over entire range.
        degree=5: max abs error ~0.000015 over entire range.
    """

    # --- extract mantissa & exponent ---------------------
    # exponent
    xi = np.int64(np.float64(x).view(np.int64))
    exponent = ((xi >> 52) & 0x7FF) - 1022
    # mantissa
    xi = (xi & 0x000FFFFFFFFFFFFF) | 0x3FE0000000000000
    m = np.int64(xi).view(np.float64)

    # --- polynomial approximation ------------------------
    if degree == 2:
        # log2_mantissa = _D20 + m * (_D21 + m * _D22)
        log2_mantissa = _D20 + (m * _D21) + (m * m * _D22)
    elif degree == 3:
        log2_mantissa = _D30 + m * (_D31 + m * (_D32 + m * _D33))
    elif degree == 4:
        log2_mantissa = _D40 + m * (_D41 + m * (_D42 + m * (_D43 + m * _D44)))
    else:
        log2_mantissa = _D50 + m * (_D51 + m * (_D52 + m * (_D53 + m * (_D54 + m * _D55))))

    # Return log2(x) = exponent + log2(m)
    return exponent + log2_mantissa


@numba.njit(fastmath=True, inline="always")
def fast_log2_f32_poly(x: np.float32, degree: int) -> np.float32:
    """
    Fast log2 approximation using polynomial after range reduction.
    Accuracy depends on degree:
        degree=2: max abs error ~0.006    over entire range.
        degree=3: max abs error ~0.0007   over entire range.
        degree=4: max abs error ~0.0001   over entire range.
        degree=5: max abs error ~0.000015 over entire range.
    """

    # --- extract mantissa & exponent ---------------------
    # exponent
    xi = np.int32(np.float32(x).view(np.int32))
    exponent = ((xi >> 23) & 0xFF) - 126
    # mantissa
    xi = (xi & 0x007FFFFF) | 0x3F000000
    m = np.int32(xi).view(np.float32)

    # --- polynomial approximation ------------------------
    if degree == 2:
        # log2_mantissa = _S20 + m * (_S21 + m * _S22)
        log2_mantissa = _S20 + (m * _S21) + (m * m * _S22)
    elif degree == 3:
        log2_mantissa = _S30 + m * (_S31 + m * (_S32 + m * _S33))
    elif degree == 4:
        log2_mantissa = _S40 + m * (_S41 + m * (_S42 + m * (_S43 + m * _S44)))
    else:
        log2_mantissa = _S50 + m * (_S51 + m * (_S52 + m * (_S53 + m * (_S54 + m * _S55))))

    # Return log2(x) = exponent + log2(mantissa)
    return exponent + log2_mantissa


# =================================================================================================
#  Public API
# =================================================================================================
__ALL__ = [
    "fast_log2_f32_poly",
    "fast_log2_f32_poly",
    "fast_log_f64_poly",
    "fast_log_f64_poly",
]
