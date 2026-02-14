from abc import ABC, abstractmethod

import numpy as np


# =================================================================================================
#  AxisTransform
# =================================================================================================
class AxisTransform(ABC):
    """
    A class that defines an axis transform x' = f(x), where...
      x  = original data
      x' = transformed data; coordinates used to plot the data onto the axis

    This class provides two paths:

        x' = f(x)       (forward transform, used to plot data)
        x  = f_inv(x')  (inverse transform, used to convert back)
    """

    @abstractmethod
    def f(self, x: float | np.ndarray | list[float]) -> float | np.ndarray:
        """Apply the forward transform f(x)."""
        raise NotImplementedError()

    @abstractmethod
    def f_inv(self, x_transformed: float | np.ndarray | list[float]) -> float | np.ndarray:
        """Apply the backward transform f_inv(x')."""
        raise NotImplementedError()


# =================================================================================================
#  NullTransform
# =================================================================================================
class NullTransform(AxisTransform):
    """A simple identity transform that leaves the data unchanged."""

    def f(self, x: float | np.ndarray | list[float]) -> float | np.ndarray:
        return x

    def f_inv(self, x_transformed: float | np.ndarray | list[float]) -> float | np.ndarray:
        return x_transformed
