import numpy as np

from local.docs.utils.axis_transforms._base import AxisTransform


class LogTransform(AxisTransform):
    """
    AxisTransform with...
        f(x)      = log(x)
        f_inv(x') = exp(x')
    """

    def f(self, x: float | np.ndarray | list[float]) -> float | np.ndarray:
        if isinstance(x, np.ndarray | list):
            return np.log(x)
        else:
            return float(np.log(x))

    def f_inv(self, x_transformed: float | np.ndarray | list[float]) -> float | np.ndarray:
        if isinstance(x_transformed, np.ndarray | list):
            return np.exp(x_transformed)
        else:
            return float(np.exp(x_transformed))
