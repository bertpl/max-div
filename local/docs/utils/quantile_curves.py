from __future__ import annotations

from functools import cached_property

import numpy as np

from local.docs.utils.axis_transforms import AxisTransform, NullTransform
from local.docs.utils.regression import SplineBounds, SplineQuantileRegressor, SplineRegressor
from local.docs.utils.regression.splines import CubicSplineQuantileRegressor


class QuantileCurves:
    """
    Class modeling (q10, q50, q90)-quantile curves.  If y_transform is provided, spline regressions are performed
    in transformed y-space, but end results (quantiles, core curves) are returned in original y-space.
    """

    # -------------------------------------------------------------------------
    #  Constructor
    # -------------------------------------------------------------------------
    def __init__(
        self,
        q10: SplineRegressor,
        q50: SplineRegressor,
        q90: SplineRegressor,
        x_transform: AxisTransform | None = None,
        y_transform: AxisTransform | None = None,
    ):
        """
        Initialize a QuantileCurves object, which models (q10, q50, q90)-quantile curves,
        that represent conditional q10,q50,q90 quantile curves of y given x.

        The curves are represented as a spline in transformed (x', y')-space (optionally).
        Transforms (x->x' & y->y') can be optionally provided for both x and y separately.

        All public methods work in terms of original (x,y)-coordinates, though.  Transforms are mainly supported,
        to capture some severe non-linearities, to make spline-fitting easier.

        :param q10: (SplineRegressor) SplineRegressor modeling the q10 quantile curve in transformed (x', y')-space.
        :param q50: (SplineRegressor) SplineRegressor modeling the q50 quantile curve in transformed (x', y')-space.
        :param q90: (SplineRegressor) SplineRegressor modeling the q90 quantile curve in transformed (x', y')-space.
        :param x_transform: (AxisTransform|None) Optional transform for x-axis; if not provided, x=x'.
        :param y_transform: (AxisTransform|None) Optional transform for x-axis; if not provided, y=y'.
        """
        self._q10 = q10
        self._q50 = q50
        self._q90 = q90
        self._x_transform = x_transform or NullTransform()
        self._y_transform = y_transform or NullTransform()

    @cached_property
    def x_min(self) -> float:
        """Return smallest x for which all quantile curves are properly defined."""
        return max(
            [
                float(min(self._x_transform.f_inv(self._q10.knots))),
                float(min(self._x_transform.f_inv(self._q50.knots))),
                float(min(self._x_transform.f_inv(self._q90.knots))),
            ]
        )

    @cached_property
    def x_max(self) -> float:
        """Return largest x for which all quantile curves are properly defined."""
        return min(
            [
                float(max(self._x_transform.f_inv(self._q10.knots))),
                float(max(self._x_transform.f_inv(self._q50.knots))),
                float(max(self._x_transform.f_inv(self._q90.knots))),
            ]
        )

    # -------------------------------------------------------------------------
    #  Evaluation methods
    # -------------------------------------------------------------------------
    def get_full_curves(self, n: int = 1000) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return (x, q10, q50, q90) curves as 4 numpy arrays, spanning entire supported x-range."""
        x_min_trans = self._x_transform.f(self.x_min)
        x_max_trans = self._x_transform.f(self.x_max)
        x_trans = np.linspace(x_min_trans, x_max_trans, n)
        q10 = self._y_transform.f_inv(self._q10.predict(x_trans))
        q50 = self._y_transform.f_inv(self._q50.predict(x_trans))
        q90 = self._y_transform.f_inv(self._q90.predict(x_trans))
        return self._x_transform.f_inv(x_trans), q10, q50, q90

    def get_quantiles_at_x(
        self, x: float | np.ndarray
    ) -> tuple[float | np.ndarray, float | np.ndarray, float | np.ndarray]:
        """Get (q10, q50, q90) quantile values at a single x value, returned in original y-space."""
        x_trans = self._x_transform.f(x)
        q10 = self._y_transform.f_inv(self._q10.predict(x_trans))
        q50 = self._y_transform.f_inv(self._q50.predict(x_trans))
        q90 = self._y_transform.f_inv(self._q90.predict(x_trans))
        return q10, q50, q90

    # -------------------------------------------------------------------------
    #  Factory Methods
    # -------------------------------------------------------------------------
    @classmethod
    def from_data(
        cls,
        x_data: np.ndarray,
        y_data: np.ndarray,
        n_knots: int = 3,
        x_transform: AxisTransform | None = None,
        y_transform: AxisTransform | None = None,
        q50_reg: float = 0.0,
        q10_q90_reg: float = 0.0,
    ) -> QuantileCurves:
        """
        Create quantile curves for the provided data.

        We first transform the data x->x', y->y' using the provided transforms.
         - log transform for x_data
         - upper exponential transform for y_data

        Then we compute quantiles using quantile spline regression, which we fit such that...
          - q10 <= q50 <= q90 for all x
          - q10, q50, q90 are monotone non-decreasing in x

        """

        # --- transform data ------------------------------
        x_transform = x_transform or NullTransform()
        y_transform = y_transform or NullTransform()
        x_trans = x_transform.f(x_data)
        y_trans = y_transform.f(y_data)

        # --- fit quantile splines ------------------------

        # prep
        y_min = min(y_trans)
        y_max = max(y_trans)
        y_margin = max(
            (y_max - y_min) * 0.01,
            (abs(y_max) + abs(y_min)) * 0.001,
            1e-9,
        )

        # derivative bounds
        dfdx = (max(y_trans) - min(y_trans)) / (max(x_trans) - min(x_trans))

        # first fit q50
        q50 = SplineQuantileRegressor.cubic(
            x_trans,
            y_trans,
            n_knots,
            q=0.5,
            fx_bounds=SplineBounds(
                x=x_trans,
                lb=np.full_like(x_trans, y_min - (0.5 * y_margin)),
                ub=np.full_like(x_trans, y_max + (0.5 * y_margin)),
            ),
            dfx_bounds=SplineBounds(
                x=x_trans,
                lb=np.full_like(x_trans, dfdx / 10),
            ),
            reg=q50_reg,
        )
        q50_spline_y = q50.predict(x_trans)
        q50_spline_dydx = q50.predict_deriv(x_trans)

        # fit q10 <= q50-margin
        q10_delta = SplineQuantileRegressor.cubic(
            x_trans,
            y_trans - q50_spline_y,
            n_knots,
            q=0.1,
            fx_bounds=SplineBounds(
                x=x_trans,
                lb=np.full_like(x_trans, y_min - (2 * y_margin)) - q50_spline_y,
                ub=np.full_like(x_trans, -y_margin),
            ),
            dfx_bounds=SplineBounds(
                x=x_trans,
                lb=np.full_like(x_trans, dfdx / 10) - q50_spline_dydx,
            ),
            reg=q10_q90_reg,
        )
        q10 = CubicSplineQuantileRegressor(
            knots=q10_delta.knots,
            c=q50.c + q10_delta.c,
        )

        # fit q90 >= q50+margin
        q90_delta = SplineQuantileRegressor.cubic(
            x_trans,
            y_trans - q50_spline_y,
            n_knots,
            q=0.9,
            fx_bounds=SplineBounds(
                x=x_trans,
                lb=np.full_like(x_trans, y_margin),
                ub=np.full_like(x_trans, y_max + (2 * y_margin)) - q50_spline_y,
            ),
            dfx_bounds=SplineBounds(
                x=x_trans,
                lb=np.full_like(x_trans, dfdx / 10) - q50_spline_dydx,
            ),
            reg=q10_q90_reg,
        )
        q90 = CubicSplineQuantileRegressor(
            knots=q90_delta.knots,
            c=q50.c + q90_delta.c,
        )

        # --- return curves -------------------------------
        return QuantileCurves(q10=q10, q50=q50, q90=q90, x_transform=x_transform, y_transform=y_transform)
