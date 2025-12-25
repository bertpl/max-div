import numba
import numpy as np

# numpy array of float32 powers of 2 from 2^-150 to 2^128,
# which - on each side - is 1 beyond full range of values >0 and <+inf.
#  _POWERS_OF_2_F32[0]    = 0.0       (2^-150, underflow to 0)
#  _POWERS_OF_2_F32[1]    = ~1e-45    (2^-149)
#  _POWERS_OF_2_F32[150]  = 1.0       (2^0)
#  _POWERS_OF_2_F32[277]  = ~1.7e38   (2^127)
#  _POWERS_OF_2_F32[278]  = +inf      (2^128, overflow to +inf)
_POWERS_OF_2_F32_MIN_EXP = np.int32(-150)
_POWERS_OF_2_F32_MAX_EXP = np.int32(128)
_POWERS_OF_2_F32 = np.array(
    [
        np.float32(2**k) if k < 128 else np.float32(np.inf)  # avoid overflow warning by manually setting inf for 2^128
        for k in range(_POWERS_OF_2_F32_MIN_EXP, _POWERS_OF_2_F32_MAX_EXP + 1)
    ],
    dtype=np.float32,
)


@numba.njit("float32(int32)", fastmath=True, inline="always")
def power_of_2_f32(k: np.int32) -> np.float32:
    """
    Compute float32 2^k for arbitrary int32 k, using precomputed values in range [-150, 128].

    :param k: (np.int32) exponent
    :return: (np.float32) 2^k    (returning 0.0 or np.inf in case of underflow/overflow)
    """
    k = max(_POWERS_OF_2_F32_MIN_EXP, min(k, _POWERS_OF_2_F32_MAX_EXP))
    return _POWERS_OF_2_F32[k - _POWERS_OF_2_F32_MIN_EXP]
