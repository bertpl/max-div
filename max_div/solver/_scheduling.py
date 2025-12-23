"""
Module with scheduling-related classes and functions for the max_div package.

With 'scheduling' we refer to letting parameters evolve according to a certain schedule over the duration of
 executing of a SolverStep, based on the progress fraction in [0.0, 1.0].
"""

import numba
import numpy as np
from numpy.typing import NDArray


# =================================================================================================
#  Base class
# =================================================================================================
class ParameterSchedule:
    def __init__(self, v0: float, v1: float, c_poly: list[float]):
        """
        Initializes a ScheduledParameter instance, where the parameter value `v` as a function of
          progress fraction `f` in [0.0, 1.0] is defined as follows:

            v(f) = v0 + (v1 - v0) * s(f)
        with
            s(f) = c0 + c1*f + c2*f^2 + c3*f^3

        where s(f) is the scheduling function mapping [0.0, 1.0] -> [0.0, 1.0].

        In principle, we expect s(0) = 0 and s(1) = 1, as a result of which we expect c0=0 and c1+c2+c3=1,
          but this is not enforced here.

        :param v0: (float) initial value of the parameter at progress fraction f=0.0
        :param v1: (float) final value of the parameter at progress fraction f=1.0
        :param c_poly: (list[float]) coefficients of the cubic polynomial defining the scheduling function s(f)
        """
        self.v0 = v0
        self.v1 = v1
        self.c_poly = c_poly

    def get_value(self, f: float) -> float:
        """
        Compute the value of the parameter at progress fraction f in [0.0, 1.0].
        NOTE: this function is provided as a reference implementation for computing the scheduled value;
              in performance-critical code paths, more efficient implementations should be preferred.
        """

        # --- compute v(f) ----------------------
        f = max(0.0, min(1.0, f))  # clip f to [0.0, 1.0]
        sf = self.c_poly[0] + self.c_poly[1] * f + self.c_poly[2] * (f * f) + self.c_poly[3] * (f * f * f)
        v = self.v0 + (self.v1 - self.v0) * sf

        # --- return clipped version ------------
        # (to ensure rounding does not violate bounds, which might cause issues downstream)
        min_value = min(self.v0, self.v1)
        max_value = max(self.v0, self.v1)
        return max(min_value, min(max_value, v))

    @property
    def min_value(self) -> float:
        """Minimum possible value of the parameter, assuming s(f) in [0.0, 1.0]."""
        return min(self.v0, self.v1)

    @property
    def max_value(self) -> float:
        """Maximum possible value of the parameter, assuming s(f) in [0.0, 1.0]."""
        return max(self.v0, self.v1)


# =================================================================================================
#  Child classes
# =================================================================================================
class LinearSchedule(ParameterSchedule):
    def __init__(self, v0: float, v1: float):
        super().__init__(v0, v1, [0.0, 1.0, 0.0, 0.0])


class EaseInSchedule(ParameterSchedule):
    def __init__(self, v0: float, v1: float):
        super().__init__(v0, v1, [0.0, 0.0, 1.0, 0.0])


class EaseOutSchedule(ParameterSchedule):
    def __init__(self, v0: float, v1: float):
        super().__init__(v0, v1, [0.0, 2.0, -1.0, 0.0])


class EaseInOutSchedule(ParameterSchedule):
    def __init__(self, v0: float, v1: float):
        super().__init__(v0, v1, [0.0, 0.0, 3.0, -2.0])


# =================================================================================================
#  Aliases
# =================================================================================================
def linear(v0: float, v1: float) -> ParameterSchedule:
    """Alias for LinearSchedule."""
    return LinearSchedule(v0, v1)


def ease_in(v0: float, v1: float) -> ParameterSchedule:
    """Alias for EaseInSchedule."""
    return EaseInSchedule(v0, v1)


def ease_out(v0: float, v1: float) -> ParameterSchedule:
    """Alias for EaseOutSchedule."""
    return EaseOutSchedule(v0, v1)


def ease_in_out(v0: float, v1: float) -> ParameterSchedule:
    """Alias for EaseInOutSchedule."""
    return EaseInOutSchedule(v0, v1)


# =================================================================================================
#  NUMBA-acceleration
# =================================================================================================
def _schedules_to_2d_numpy_array(schedules: list[ParameterSchedule]) -> NDArray[np.float64]:
    """
    Convert a list of ParameterSchedule instances to a 2D numpy array for use in low-level numba-optimized
      schedule evaluation functions.
    :param schedules: list of ParameterSchedule instances
    :return: 2D numpy array of shape (n_schedules, 6) with schedule data
    """

    # --- prep ------------------------
    n_schedules = len(schedules)
    arr = np.empty((n_schedules, 6), dtype=np.float64)

    # --- fill ------------------------
    for i, sched in enumerate(schedules):
        # --- convert ---
        v0 = sched.v0
        dv = sched.v1 - sched.v0
        c0, c1, c2, c3 = sched.c_poly
        d0 = v0 + (dv * c0)  # coefficients in absolute terms
        d1 = dv * c1
        d2 = dv * c2
        d3 = dv * c3

        # --- fill ---
        arr[i, 0] = sched.min_value
        arr[i, 1] = sched.max_value
        arr[i, 2] = d0
        arr[i, 3] = d1
        arr[i, 4] = d2
        arr[i, 5] = d3

    return arr


@numba.njit(fastmath=True, inline="always")
def _evaluate_schedules(schedules_array: NDArray[np.float64], f: float) -> NDArray[np.float64]:
    """
    Evaluate multiple ParameterSchedule instances at once, given their numpy array representation.
    :param schedules_array: 2D numpy array of shape (n_schedules, 6) with schedule data
    :param f: progress fraction in [0.0, 1.0]
    :return: 1D numpy array of shape (n_schedules,) with evaluated parameter values
    """

    # --- prep ------------------------
    n_schedules = schedules_array.shape[0]
    values = np.empty(n_schedules, dtype=np.float64)
    f = max(0.0, min(1.0, f))  # clip f to [0.0, 1.0]

    # --- evaluate --------------------
    for i in range(n_schedules):
        # extract data
        min_value = schedules_array[i, 0]
        max_value = schedules_array[i, 1]
        d0 = schedules_array[i, 2]
        d1 = schedules_array[i, 3]
        d2 = schedules_array[i, 4]
        d3 = schedules_array[i, 5]

        # compute v(f)
        v = d0 + f * (d1 + f * (d2 + (f * d3)))
        v = max(min_value, min(max_value, v))  # clip to [min_value, max_value]

        # store result
        values[i] = v

    return values
