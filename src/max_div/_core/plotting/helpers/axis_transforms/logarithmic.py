"""This module defines the plain logarithmic axis transform."""

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .base import AxisTransform, as_float_or_array


class LogTransform(AxisTransform):
    """The logarithmic transform: a data value `x` is drawn at `log(x)`."""

    def to_axis(self, x: float | ArrayLike) -> float | NDArray[np.float64]:
        """Return `log(x)`."""
        return as_float_or_array(np.log(np.asarray(x, dtype=np.float64)))

    def from_axis(self, x_axis: float | ArrayLike) -> float | NDArray[np.float64]:
        """Return `exp(x_axis)`."""
        return as_float_or_array(np.exp(np.asarray(x_axis, dtype=np.float64)))
