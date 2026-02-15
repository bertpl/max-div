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
        A_l2reg: np.ndarray | None = None,
        b_l2reg: np.ndarray | None = None,
    ) -> LinearRegressor:
        """
        Solve the linear system A @ c = b for c using quantile regression with quantile q, optionally
        with an additional L2 regularization term || A_l2reg @ - b_l2reg ||_2^2
             and/or linear inequality constraints (Aineq @ c <= bineq).

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
        if (A_l2reg is not None) and (b_l2reg is not None):
            if A_l2reg.shape[1] != n:
                raise ValueError(f"A_l2reg should have shape (m_l2reg, {n}), but has shape {A_l2reg.shape}")
            if b_l2reg.ndim != 1:
                raise ValueError(f"b_l2reg should be 1D numpy array, but has {b_l2reg.ndim} dimensions")
            if b_l2reg.shape[0] != A_l2reg.shape[0]:
                raise ValueError(f"b_l2reg vs A_l2reg size mismatch")
        elif (A_l2reg is not None) or (b_l2reg is not None):
            raise ValueError("(A_l2reg, b_l2reg) should either be both None or both numpy arrays.")

        # --- construct LP problem ------------------------
        c = cp.Variable(n)
        residuals = b - A @ c
        loss = cp.sum(cp.maximum(q * residuals, (q - 1) * residuals))  # pinball loss
        if mineq > 0:
            constraints = [Aineq @ c <= bineq]
        else:
            constraints = None

        # --- add L2 regularization term if provided ------
        is_qp = False
        if A_l2reg is not None:
            l2_reg_term = cp.sum_squares(A_l2reg @ c - b_l2reg)
            loss += l2_reg_term
            is_qp = True

        # --- solve LP/QP problem -------------------------
        problem = cp.Problem(cp.Minimize(loss), constraints)
        if is_qp:
            problem.solve(cp.OSQP, max_iter=1_000_000)
        else:
            problem.solve(cp.ECOS)

        # --- check & extract solution --------------------
        if not (problem.status == cp.OPTIMAL):
            print(f"WARNING: Quantile regression problem could not be solved to optimality. Status: {problem.status}.")
            print("         Will try to continue with sub-optimal solution...")

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
