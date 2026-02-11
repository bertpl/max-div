from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Self

import numpy as np

from local.docs.utils.regression import LinearRegressor


# =================================================================================================
#  SplineBounds
# =================================================================================================
@dataclass
class SplineBounds:
    # represents bounds on f(x) or f'(x)
    x: np.ndarray
    lb: np.ndarray | None = None
    ub: np.ndarray | None = None

    def __post_init__(self):
        if self.lb is None and self.ub is None:
            raise ValueError("At least one of lb or ub should be provided.")
        if self.lb is not None and self.lb.shape != self.x.shape:
            raise ValueError(f"lb should have the same shape as x (here: {self.lb.shape} != {self.x.shape})")
        if self.ub is not None and self.ub.shape != self.x.shape:
            raise ValueError(f"ub should have the same shape as x (here: {self.ub.shape} != {self.x.shape})")


# =================================================================================================
#  SplineRegressor
# =================================================================================================
class SplineRegressor(ABC):
    """Class for 1D spline regression."""

    # -------------------------------------------------------------------------
    #  Constructor
    # -------------------------------------------------------------------------
    def __init__(self, knots: np.ndarray):
        """
        Create a spline regressor with the given knots.  The knots should be unique and sorted in increasing order.
        :param knots: (list[float]) List of knot locations, sorted in increasing order.
        """

        # --- argument handling ---
        if list(knots) != sorted(set(knots)):
            raise ValueError("Knots should be unique and sorted in increasing order.")
        if len(knots) < 2:
            raise ValueError("There should be at least 2 knots.")

        # --- store parameters ---
        self.knots = knots
        self.c = np.zeros(self.n)  # to be set appropriately using a fitting method

    @property
    def n_knots(self) -> int:
        return self.knots.size

    @property
    @abstractmethod
    def n(self) -> int:
        """
        Return number of basis functions.
        Depending on the degree of the spline, we have n==n_knots  (linear)
                                                    or n >n_knots  (higher order).
        """
        raise NotImplementedError

    # -------------------------------------------------------------------------
    #  Prediction API
    # -------------------------------------------------------------------------
    def predict(self, x: float | np.ndarray) -> float | np.ndarray:

        # prep input & compute basis matrix
        if isinstance(x, float):
            basis_matrix = self._basis_matrix(np.array([x]))
        else:
            basis_matrix = self._basis_matrix(x)

        # evaluate spline
        fx_values = basis_matrix @ self.c

        # return result in appropriate format
        if isinstance(x, float):
            return float(fx_values[0])
        else:
            return fx_values

    # -------------------------------------------------------------------------
    #  Internal
    # -------------------------------------------------------------------------
    @abstractmethod
    def _basis_matrix(self, x: np.ndarray) -> np.ndarray:
        """
        Compute the basis matrix for the given input x.
        The basis matrix should have shape (m, n), where m is the number of input points and n is the number of parameters c.
        """
        raise NotImplementedError

    @abstractmethod
    def _deriv_matrix(self, x: np.ndarray) -> np.ndarray:
        """
        Compute the derivative basis matrix for the given input x.
        Conceptually similar to _basis_matrix, but now representing the derivatives of the basis functions wrt x.
        This matrix is used to impose monotonicity constraints.
        """
        raise NotImplementedError

    @staticmethod
    def _knots_from_x_data(x_data: np.ndarray, n_knots: int) -> np.ndarray:

        # --- prep --------------------
        n_unique = len(set(x_data))
        if n_unique < n_knots:
            raise ValueError(f"Cannot create {n_knots} knots from x_data with only {n_unique} unique values.")

        # --- create knots ------------
        for n_quantiles in range(n_knots, n_unique + 1):
            quantiles = np.linspace(0, 1, n_quantiles)
            knots = np.sort(np.unique(np.quantile(x_data, quantiles)))
            if len(knots) == n_knots:
                return knots

        # --- fallback ----------------
        unique_x = np.sort(np.unique(x_data))
        return np.quantile(unique_x, np.linspace(0, 1, n_knots))


# =================================================================================================
#  SplineQuantileRegressor
# =================================================================================================
class SplineQuantileRegressor(SplineRegressor, ABC):
    """Class for 1D spline regression, where fitting is done using quantile regression."""

    # -------------------------------------------------------------------------
    #  Fitting methods
    # -------------------------------------------------------------------------
    def fit(
        self,
        x_data: np.ndarray,
        y_data: np.ndarray,
        q: float,
        fx_bounds: SplineBounds | None = None,
        dfx_bounds: SplineBounds | None = None,
    ):

        # --- generate ineq. constraints ------------------
        if fx_bounds or dfx_bounds:
            # initialize Aineq and bineq as empty arrays, to be filled in the following steps
            Aineq = np.empty((0, self.n))
            bineq = np.empty((0,))

            # lb <= f(x) <= ub
            if fx_bounds:
                # append basis_matrix at x_bounds to Aineq and corresponding values to bineq
                basis_at_x = self._basis_matrix(fx_bounds.x)

                if fx_bounds.lb is not None:
                    # constraint: f(x) >= lb  =>  -f(x) <= -lb
                    Aineq = np.vstack([Aineq, -basis_at_x])
                    bineq = np.concatenate([bineq, -fx_bounds.lb])

                if fx_bounds.ub is not None:
                    # constraint: f(x) <= ub
                    Aineq = np.vstack([Aineq, basis_at_x])
                    bineq = np.concatenate([bineq, fx_bounds.ub])

            # dlb <= f'(x) <= dub
            if dfx_bounds:
                # append deriv_matrix at x_bounds to Aineq and corresponding values to bineq
                deriv_at_x = self._deriv_matrix(dfx_bounds.x)

                if dfx_bounds.lb is not None:
                    # constraint: f'(x) >= dlb  =>  -f'(x) <= -dlb
                    Aineq = np.vstack([Aineq, -deriv_at_x])
                    bineq = np.concatenate([bineq, -dfx_bounds.lb])

                if dfx_bounds.ub is not None:
                    # constraint: f'(x) <= dub
                    Aineq = np.vstack([Aineq, deriv_at_x])
                    bineq = np.concatenate([bineq, dfx_bounds.ub])

        else:
            Aineq = None
            bineq = None

        # --- actual regression ---------------------------
        basis_matrix = self._basis_matrix(x_data)
        linear_regressor = LinearRegressor.fit_quantile(basis_matrix, y_data, q, Aineq, bineq)

        # --- extract result ------------------------------
        self.c = linear_regressor.c

    # -------------------------------------------------------------------------
    #  Factory methods
    # -------------------------------------------------------------------------
    @classmethod
    def linear(
        cls,
        x_data: np.ndarray,
        y_data: np.ndarray,
        n_knots: int,
        q: float,
        fx_bounds: SplineBounds | None = None,
        dfx_bounds: SplineBounds | None = None,
    ) -> Self:
        regressor = LinearSplineQuantileRegressor(knots=cls._knots_from_x_data(x_data, n_knots))
        regressor.fit(x_data, y_data, q, fx_bounds, dfx_bounds)
        return regressor

    @classmethod
    def quadratic(
        cls,
        x_data: np.ndarray,
        y_data: np.ndarray,
        n_knots: int,
        q: float,
        fx_bounds: SplineBounds | None = None,
        dfx_bounds: SplineBounds | None = None,
    ) -> Self:
        regressor = QuadraticSplineQuantileRegressor(knots=cls._knots_from_x_data(x_data, n_knots))
        regressor.fit(x_data, y_data, q, fx_bounds, dfx_bounds)
        return regressor

    @classmethod
    def cubic(
        cls,
        x_data: np.ndarray,
        y_data: np.ndarray,
        n_knots: int,
        q: float,
        fx_bounds: SplineBounds | None = None,
        dfx_bounds: SplineBounds | None = None,
    ) -> Self:
        regressor = CubicSplineQuantileRegressor(knots=cls._knots_from_x_data(x_data, n_knots))
        regressor.fit(x_data, y_data, q, fx_bounds, dfx_bounds)
        return regressor


# =================================================================================================
#  LinearSplineQuantileRegressor
# =================================================================================================
class LinearSplineQuantileRegressor(SplineQuantileRegressor):
    @property
    def n(self) -> int:
        return self.n_knots

    def _basis_matrix(self, x: np.ndarray) -> np.ndarray:
        basis_matrix = np.zeros((x.size, self.n))
        basis_matrix[:, 0] = 1.0
        for i in range(self.n_knots - 1):
            c_normalize = 1 / (self.knots[-1] - self.knots[i])
            basis_matrix[:, i + 1] = np.where(
                x >= self.knots[i],
                c_normalize * (x - self.knots[i]),
                0.0,
            )
        return basis_matrix

    def _deriv_matrix(self, x: np.ndarray) -> np.ndarray:
        deriv_matrix = np.zeros((x.size, self.n))
        for i in range(self.n_knots - 1):
            c_normalize = 1 / (self.knots[-1] - self.knots[i])
            deriv_matrix[:, i + 1] = np.where(
                x >= self.knots[i],
                c_normalize,
                0.0,
            )
        return deriv_matrix


# =================================================================================================
#  QuadraticSplineQuantileRegressor
# =================================================================================================
class QuadraticSplineQuantileRegressor(SplineQuantileRegressor):
    @property
    def n(self) -> int:
        return self.n_knots + 1

    def _basis_matrix(self, x: np.ndarray) -> np.ndarray:
        basis_matrix = np.zeros((x.size, self.n))
        basis_matrix[:, 0] = 1.0
        basis_matrix[:, 1] = (x - self.knots[0]) / (self.knots[-1] - self.knots[0])
        for i in range(self.n_knots - 1):
            c_normalize = 1 / ((self.knots[-1] - self.knots[i]) ** 2)
            basis_matrix[:, i + 2] = np.where(
                x >= self.knots[i],
                c_normalize * ((x - self.knots[i]) ** 2),
                0.0,
            )
        return basis_matrix

    def _deriv_matrix(self, x: np.ndarray) -> np.ndarray:
        deriv_matrix = np.zeros((x.size, self.n))
        deriv_matrix[:, 1] = 1.0 / (self.knots[-1] - self.knots[0])
        for i in range(self.n_knots - 1):
            c_normalize = 1 / ((self.knots[-1] - self.knots[i]) ** 2)
            deriv_matrix[:, i + 2] = np.where(
                x >= self.knots[i],
                c_normalize * 2.0 * (x - self.knots[i]),
                0.0,
            )
        return deriv_matrix


# =================================================================================================
#  CubicSplineQuantileRegressor
# =================================================================================================
class CubicSplineQuantileRegressor(SplineQuantileRegressor):
    @property
    def n(self) -> int:
        return self.n_knots + 2

    def _basis_matrix(self, x: np.ndarray) -> np.ndarray:
        basis_matrix = np.zeros((x.size, self.n))
        basis_matrix[:, 0] = 1.0
        basis_matrix[:, 1] = (x - self.knots[0]) / (self.knots[-1] - self.knots[0])
        basis_matrix[:, 2] = ((x - self.knots[0]) / (self.knots[-1] - self.knots[0])) ** 2
        for i in range(self.n_knots - 1):
            c_normalize = 1 / ((self.knots[-1] - self.knots[i]) ** 3)
            basis_matrix[:, i + 3] = np.where(
                x >= self.knots[i],
                c_normalize * ((x - self.knots[i]) ** 3),
                0.0,
            )
        return basis_matrix

    def _deriv_matrix(self, x: np.ndarray) -> np.ndarray:
        deriv_matrix = np.zeros((x.size, self.n))
        deriv_matrix[:, 1] = 1.0 / (self.knots[-1] - self.knots[0])
        deriv_matrix[:, 2] = np.where(
            x >= self.knots[0],
            2.0 * (x - self.knots[0]) / ((self.knots[-1] - self.knots[0]) ** 2),
            0.0,
        )
        for i in range(self.n_knots - 1):
            c_normalize = 1 / ((self.knots[-1] - self.knots[i]) ** 3)
            deriv_matrix[:, i + 3] = np.where(
                x >= self.knots[i],
                c_normalize * 3.0 * ((x - self.knots[i]) ** 2),
                0.0,
            )
        return deriv_matrix
