import numpy as np
from numba import njit
from numpy.typing import NDArray

from max_div.internal.utils import EPS_F32

from ._helpers import _p_max
from ._methods import (
    _max_selective,
    _power_exact,
    _power_fast_log2_exp2,
    _power_fast_pow,
    _pwl_2_segment,
    _uniform,
)

__MODIFIER_MIN = np.float32(-1.0 + 10.0 * EPS_F32)
__MODIFIER_MAX = np.float32(1.0 - 10.0 * EPS_F32)


# =================================================================================================
#  Main Numba entrypoint
# =================================================================================================
@njit("void(float32[::1], float32, int32, float32[::1])", fastmath=True, inline="always")
def modify_p_selectivity(p: NDArray[np.float32], modifier: np.float32, method: np.int32, p_out: NDArray[np.float32]):
    """Modify the p array by applying a selectivity modification.

    "Selectivity Modification" = changing the distribution of probabilities to be more uniform or more
                                 max-selective (=favoring even more the maximum values).

      --> This modification is controlled by a "modifier" in [-1,1]:
                        --> -1 =    uniform distribution
                        -->  0 =    selectivity unchanged
                        --> +1 =    max-selective distribution

    INVARIANTS
    ----------
        - ALWAYS:
            - if p[i] < p[j]   =>  modified_p[i] <= modified_p[j]   (and strictly so if -1 < modifier < +1)
        - OFTEN  (unless noted otherwise):
            - if p[i] == 0.0   =>  modified_p[i] == 0.0
            - if p[i] == p[j]  =>  modified_p[i] == modified_p[j]

    POWER-BASED METHOD
    ------------------

        This is the reference method (method=0), which can be approximated by faster methods (method>0).

        Each element p[i] is transformed to max(p) * ((p[i] / max(p)) ** t) with ...
            --> t = (1 + modifier)/(1 - modifier)

        This way...
            --> it can be computed that area under the curve for p**t for p in [0,1] is 1/(t+1) = (1-modifier)/2
            --> modifier = -1  -->  t = 0       -->  (uniform distribution)
            --> modifier = 0   -->  t = 1       -->  (unchanged)
            --> modifier = +1  -->  t = +infty  -->  (max-selective)
            --> applying -modifier reverts the modification

    METHOD OVERVIEW
    ---------------

        Depending on the 'mode'-parameter, different methods are used to modify the p array:

        Applied method, after normalizing such that max(p) = 1:

                    0 = p**t (exact)
                   10 = fast_exp2(t * fast_log2(p))
                   20 = fast_pow(p, t)   (using local quadratic approximation after range reduction)
                  100 = 2-segment piecewise linear approximation of p**t
                          (computed such that area-under-the-curve of transformation f(p) is identical to method=0)

    NOTES
    -----
        - probability arrays are not assumed to be normalized (i.e., sum to 1)
        - output arrays should not be assumed to be normalized
        - if max(p) <= 0.0, no modification is applied   (also no normalization is applied)
        - if max(p) > 0.0 (regular case), the output array will always be normalized to range [0,1] even if modifier=0.0

    :param p: (1D float32 array) Original p values
    :param modifier: (float) The selectivity modifier in [-1,1] to apply.
    :param method: (int) Approximation mode, with higher numbers representing faster, less accurate methods.
                         See docs above for more info.
    :param p_out: (1D float32 array) Output array to store modified p values; should be same size as p.
                       If p needs to be modified in-place, also provide p as this argument.
    :return: (1D array) Modified p values.
    """

    # --- prep transformation -----------------------------
    p_max = _p_max(p)
    if p_max <= 0.0:
        # p array is degenerate -> just copy input to output, if needed & return
        if not (p is p_out):
            for i in range(p.size):
                p_out[i] = p[i]
        return

    # normalize to [0, 1]  (now that we know we can invert p_max)
    p_max_inv = 1.0 / p_max
    for i in range(p.size):
        p_out[i] = p[i] * p_max_inv

    # --- detect shortcuts --------------------------------
    if modifier == 0.0:
        # no further action needed  (p_out is already populated correctly)
        pass
    elif modifier <= __MODIFIER_MIN:
        _uniform(p_out)
    elif modifier >= __MODIFIER_MAX:
        _max_selective(p_out)
    else:
        # --- regular cases -------------------------------
        # actual transformation  (fastest methods first)
        if method == 100:
            _pwl_2_segment(p_out, modifier)
        elif method == 20:
            _power_fast_pow(p_out, modifier)
        elif method == 10:
            _power_fast_log2_exp2(p_out, modifier)
        elif method == 0:
            _power_exact(p_out, modifier)
        else:
            raise NotImplementedError
