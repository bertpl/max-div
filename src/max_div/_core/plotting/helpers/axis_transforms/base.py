"""Defines the axis transform interface and the identity transform."""

from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import ArrayLike, NDArray


# ==================================================================================================
#  AxisTransform
# ==================================================================================================
class AxisTransform(ABC):
    """An axis transform maps data values to the coordinates at which they are drawn on an axis, and back.

    `to_axis` maps a data value to its axis coordinate and `from_axis` maps back. Every transform
    accepts a scalar or an array and returns the same kind: a scalar in gives a `float` out.
    """

    @abstractmethod
    def to_axis(self, x: float | ArrayLike) -> float | NDArray[np.float64]:
        """Return the axis coordinate(s) of data value(s) `x`."""
        raise NotImplementedError

    @abstractmethod
    def from_axis(self, x_axis: float | ArrayLike) -> float | NDArray[np.float64]:
        """Return the data value(s) drawn at axis coordinate(s) `x_axis`."""
        raise NotImplementedError


# ==================================================================================================
#  NullTransform
# ==================================================================================================
class NullTransform(AxisTransform):
    """The identity transform: data values are their own axis coordinates."""

    def to_axis(self, x: float | ArrayLike) -> float | NDArray[np.float64]:
        """Return `x` unchanged."""
        return as_float_or_array(x)

    def from_axis(self, x_axis: float | ArrayLike) -> float | NDArray[np.float64]:
        """Return `x_axis` unchanged."""
        return as_float_or_array(x_axis)


# ==================================================================================================
#  Helpers
# ==================================================================================================
def as_float_or_array(x: float | ArrayLike) -> float | NDArray[np.float64]:
    """Return a scalar as a `float` and anything else as a float64 array."""
    array = np.asarray(x, dtype=np.float64)
    if array.ndim == 0:
        return float(array)
    else:
        return array
