from __future__ import annotations

import cvxpy as cp
import numpy as np


class LinearRegressor:
    # -------------------------------------------------------------------------
    #  Constructor / Properties
    # -------------------------------------------------------------------------
    def __init__(self, c: np.ndarray):
        """
        Create linear regression with parameters 'c'.  This model does not support an 'intercept' term.
        :param c: (np.ndarray) 1D numpy array with n elements
        """
        self.c = c

    @property
    def n(self) -> int:
        return self.c.size

    # -------------------------------------------------------------------------
    #  Main API
    # -------------------------------------------------------------------------
    def predict(self, x: np.ndarray) -> float | np.ndarray:
        if x.ndim == 1:
            # single prediction
            if x.size != self.n:
                raise ValueError(f"Input x should have size {self.n}, but has size {x.size}")
            return float(self.c @ x)
        elif x.ndim == 2:
            # multiple predictions
            if x.shape[1] != self.n:
                raise ValueError(f"Input x should have shape (m, {self.n}), but has shape {x.shape}")
            return x @ self.c
        else:
            # invalid
            raise ValueError(f"Input x should be 1D or 2D numpy array, but has shape {x.shape}")

    # -------------------------------------------------------------------------
    #  Factory methods
    # -------------------------------------------------------------------------
    @classmethod
    def fit_quantile(
        cls,
        A: np.ndarray,
        b: np.ndarray,
        q: float,
        Aineq: np.ndarray | None = None,
        bineq: np.ndarray | None = None,
    ) -> LinearRegressor:
        """
        Solve the linear system A @ c = b for c using quantile regression with quantile q.

        The quantile defines how many (fraction q)   of the residuals should be negative
                         and how many (fraction 1-q) of the residuals should be positive.

        This is achieved through the use of the pinball loss metric, parametrized by q.

        Optionally, linear inequality constraints can be provided as well:  Aineq @ c <= bineq.

        :param A: (np.ndarray) 2D numpy array of shape (m, n)
        :param b: (np.ndarray) 1D numpy array of shape (n,)
        :param q: (float) The quantile to be estimated, between 0 and 1.
        :param Aineq: (np.ndarray) 2D numpy array of shape (mineq, n)   (optional)
        :param bineq: (np.ndarray) 1D numpy array of shape (n,)         (optional)
        :return: (LinearRegressor) A LinearRegressor instance with fitted parameters.
        """

        # --- prep ----------------------------------------
        m, n = A.shape
        if b.shape != (m,):
            raise ValueError(f"b should have shape ({m},), but has shape {b.shape}")
        if Aineq is not None and bineq is not None:
            mineq = Aineq.shape[0]
            if Aineq.shape[1] != n:
                raise ValueError(f"Aineq should have shape (mineq, {n}), but has shape {Aineq.shape}")
            if bineq.shape != (mineq,):
                raise ValueError(f"bineq should have shape ({mineq},), but has shape {bineq.shape}")
        else:
            mineq = 0
            if (Aineq is not None) or (bineq is not None):
                raise ValueError("(Aineq, bineq) should either be both None or both numpy arrays.")

        # --- construct LP problem ------------------------
        c = cp.Variable(n)
        residuals = b - A @ c
        pinball_loss = cp.sum(cp.maximum(q * residuals, (q - 1) * residuals))
        if mineq > 0:
            constraints = [Aineq @ c <= bineq]
        else:
            constraints = None

        # --- solve LP problem ----------------------------
        problem = cp.Problem(cp.Minimize(pinball_loss), constraints)
        problem.solve(cp.ECOS)

        # --- check & extract solution --------------------
        if not (problem.status == cp.OPTIMAL):
            raise ValueError(f"Quantile regression problem could not be solved to optimality. Status: {problem.status}")
        else:
            return LinearRegressor(c.value)


if __name__ == "__main__":
    model = LinearRegressor(np.array([1.0, 2.0, 3.0]))

    y_single = model.predict(np.array([1.0, 1.0, 1.0]))

    y_arr = model.predict(
        np.array(
            [
                [1.0, 1.0, 1.0],
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1],
            ]
        )
    )
