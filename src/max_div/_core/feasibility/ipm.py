"""A primal--dual interior-point method solves the relaxed feasibility problem exactly.

The problem: minimize the total penalty `sum of phi_i(shortfall_i) +
phi_i(excess_i)` over fractional selections `x in [0, 1]^n` with `sum(x) = k`, where each
constraint's penalty is `phi_i(t) = w_lin_i * t + w_quad_i * t^2` and shortfall/excess are slack
variables bounded below by the count constraints.  One solve yields three outputs:

- the optimal value: zero exactly when a fractional selection satisfies every constraint;
- the count-constraint multipliers `(lam_min, lam_max)`: after clamping, an infeasibility
  certificate whenever `certified_bound` evaluates positive (see `evaluation.py`);
- the interior optimizer `marginals`: the most-interior optimal fractional selection, read as
  per-item inclusion probabilities by randomized rounding.

The method is the standard Mehrotra predictor--corrector scheme on the equality standard form: the
stacked variable `(x, shortfall, excess, lower-surplus, upper-surplus)` with `2m + 1` equality
rows (two count blocks and the cardinality row) and simple bounds on the variables.  Every Newton
solve reduces to normal equations of size `2m + 1` (a dense block assembled by `_weighted_gram`;
everything else diagonal).  The standard form is deliberately chosen over the smaller
inequality-form reduction: with near-coinciding count bounds the latter's elimination diagonals
span too many decades for double precision, while the standard form converges on the same
instances.

Architecture: the iteration loop is plain Python + numpy/scipy — the solver runs a few dozen
iterations once per call, so control flow needs no compilation; only the O(n)/O(nnz) array passes
(count sums, score scatters, Gram assembly) are numba helpers, and the factorization is scipy's
Cholesky.
"""

from dataclasses import dataclass

import numba
import numpy as np
from numpy.typing import NDArray
from scipy.linalg import cho_factor, cho_solve

from .evaluation import _item_scores
from .indexing import build_item_constraint_csr

# =================================================================================================
#  Constants
# =================================================================================================
TOLERANCE = 1e-8  # convergence: duality measure and relative primal/dual residual norms
MAX_ITERATIONS = 100  # safety cap; the path-following iteration count is a few dozen in practice
STEP_FRACTION = 0.995  # fraction of the largest positivity-preserving step actually taken
REGULARIZATION = 1e-10  # relative diagonal increase; keeps the factorization valid for duplicated constraints


# =================================================================================================
#  RelaxationSolution
# =================================================================================================
@dataclass(frozen=True)
class RelaxationSolution:
    """A `RelaxationSolution` carries the outputs of one relaxation solve.

    Attributes:
        value: the optimal total penalty of the relaxation; zero (up to tolerance) exactly when a
            fractional selection satisfies every constraint.
        marginals: the interior optimal fractional selection, one entry in [0, 1] per item,
            summing to `k` — the inclusion probabilities randomized rounding draws from.
        lam_min: the lower-count multipliers; with `lam_max` the inputs to an infeasibility
            certificate (clamp, then evaluate `certified_bound`).
        lam_max: the upper-count multipliers.
        iterations: the number of predictor--corrector iterations run.
        converged: whether the convergence tolerances were met within the iteration cap.  An
            unconverged solve still returns valid (merely less accurate) outputs — certificates
            in particular stay sound, because their bound is re-evaluated exactly.
    """

    value: float
    marginals: NDArray[np.float64]
    lam_min: NDArray[np.float64]
    lam_max: NDArray[np.float64]
    iterations: int
    converged: bool


# =================================================================================================
#  njit helpers: the O(n)/O(nnz) array passes
# =================================================================================================
@numba.njit(cache=True)
def _fractional_counts(
    item_indptr: NDArray[np.int64], item_cons: NDArray[np.int32], x: NDArray[np.float64], m: int
) -> NDArray[np.float64]:
    """Return the per-constraint fractional counts `A x` (sum of x over each constraint's members).

    Args:
        item_indptr: item->constraint CSR offsets, as built by `build_item_constraint_csr`.
        item_cons: item->constraint CSR values — the constraints containing each item.
        x: fractional selection, one value per item.
        m: the number of constraints.
    """
    counts = np.zeros(m, dtype=np.float64)
    for j in range(x.shape[0]):
        for e in range(item_indptr[j], item_indptr[j + 1]):
            counts[item_cons[e]] += x[j]
    return counts


@numba.njit(cache=True)
def _weighted_gram(
    item_indptr: NDArray[np.int64], item_cons: NDArray[np.int32], d: NDArray[np.float64], m: int
) -> NDArray[np.float64]:
    """Assemble the weighted Gram matrix `A diag(d) A^T` as a dense m x m matrix.

    Item `j` contributes `d[j]` to entry `(i, l)` for every pair of constraints `i, l` containing
    it — the sum-of-squared-degrees pass that dominates the per-iteration cost on densely
    overlapping constraint families.

    Args:
        item_indptr: item->constraint CSR offsets, as built by `build_item_constraint_csr`.
        item_cons: item->constraint CSR values — the constraints containing each item.
        d: per-item weights.
        m: the number of constraints.
    """
    gram = np.zeros((m, m), dtype=np.float64)
    for j in range(item_indptr.shape[0] - 1):
        w = d[j]
        for e1 in range(item_indptr[j], item_indptr[j + 1]):
            i1 = item_cons[e1]
            for e2 in range(item_indptr[j], item_indptr[j + 1]):
                gram[i1, item_cons[e2]] += w
    return gram


# =================================================================================================
#  solve_relaxation
# =================================================================================================
def solve_relaxation(
    con_min: NDArray[np.int64],
    con_max: NDArray[np.int64],
    w_lin: NDArray[np.float64],
    w_quad: NDArray[np.float64],
    con_indices: NDArray[np.int32],
    n: int,
    k: int,
) -> RelaxationSolution:
    """Solve the relaxed feasibility problem to optimality.

    Args:
        con_min: per-constraint minimum counts.
        con_max: per-constraint maximum counts.
        w_lin: per-constraint linear penalty weights (>= 0).
        w_quad: per-constraint quadratic penalty weights (>= 0; not both zero per constraint).
        con_indices: packed constraint->item membership array (`ConstraintList.to_numpy`).
        n: the number of items.
        k: the selection size (0 < k < n).

    Returns:
        The solve's outputs; see `RelaxationSolution`.
    """
    item_indptr, item_cons = build_item_constraint_csr(con_indices, n)
    m = con_min.shape[0]
    big_n = n + 4 * m  # stacked variable: (x, shortfall, excess, lower-surplus, upper-surplus)
    i_x = slice(0, n)
    i_sm = slice(n, n + m)
    i_sp = slice(n + m, n + 2 * m)
    i_tm = slice(n + 2 * m, n + 3 * m)
    i_tp = slice(n + 3 * m, n + 4 * m)
    zeros_m = np.zeros(m)

    # objective: linear coefficients and diagonal quadratic coefficients, on the slack blocks only
    c = np.zeros(big_n)
    c[i_sm] = w_lin
    c[i_sp] = w_lin
    h = np.zeros(big_n)
    h[i_sm] = 2.0 * w_quad
    h[i_sp] = 2.0 * w_quad

    b = np.concatenate([con_min.astype(np.float64), con_max.astype(np.float64), [float(k)]])
    b_norm = 1.0 + float(np.linalg.norm(b))
    c_norm = 2.0 + float(np.linalg.norm(c))
    has_ub = np.zeros(big_n, dtype=np.bool_)
    has_ub[i_x] = True  # only the selection block is boxed above (by 1)

    def m_dot(z: NDArray[np.float64]) -> NDArray[np.float64]:
        """Apply the equality-row matrix: the two count blocks and the cardinality row."""
        ax = _fractional_counts(item_indptr, item_cons, z[i_x], m)
        return np.concatenate([ax + z[i_sm] - z[i_tm], ax - z[i_sp] + z[i_tp], [z[i_x].sum()]])

    def mt_dot(y: NDArray[np.float64]) -> NDArray[np.float64]:
        """Apply the equality-row matrix transposed."""
        y1, y2 = y[:m], y[m : 2 * m]
        out = np.empty(big_n)
        out[i_x] = _item_scores(item_indptr, item_cons, y1 + y2, zeros_m) + y[2 * m]
        out[i_sm] = y1
        out[i_sp] = -y2
        out[i_tm] = -y1
        out[i_tp] = y2
        return out

    # --- strictly interior start ----------------
    z = np.ones(big_n)
    z[i_x] = k / n
    v = np.where(has_ub, 1.0 - z, 1.0)  # upper-bound slack (dummy 1.0 where unboxed)
    r_lo = np.ones(big_n)  # duals of the lower bounds (z >= 0)
    q_ub = np.where(has_ub, 1.0, 0.0)  # duals of the upper bounds (x <= 1)
    y = np.zeros(2 * m + 1)  # duals of the equality rows

    iterations = 0
    converged = False
    for _ in range(MAX_ITERATIONS):
        iterations += 1

        # --- residuals + convergence test -------
        rp = b - m_dot(z)
        rd = c + h * z - mt_dot(y) - r_lo + q_ub
        ru = np.where(has_ub, 1.0 - z - v, 0.0)
        mu = (z @ r_lo + (v * q_ub)[has_ub].sum()) / (big_n + n)
        if mu < TOLERANCE and np.linalg.norm(rp) / b_norm < TOLERANCE and np.linalg.norm(rd) / c_norm < TOLERANCE:
            converged = True
            break

        # --- diagonal scaling + normal matrix ---
        d = 1.0 / (h + r_lo / z + np.where(has_ub, q_ub / v, 0.0))
        dx = d[i_x]
        gram = _weighted_gram(item_indptr, item_cons, dx, m)
        a_dx = _fractional_counts(item_indptr, item_cons, dx, m)
        s_mat = np.empty((2 * m + 1, 2 * m + 1))
        s_mat[:m, :m] = gram + np.diag(d[i_sm] + d[i_tm])
        s_mat[:m, m : 2 * m] = gram
        s_mat[m : 2 * m, :m] = gram
        s_mat[m : 2 * m, m : 2 * m] = gram + np.diag(d[i_sp] + d[i_tp])
        s_mat[:m, 2 * m] = a_dx
        s_mat[m : 2 * m, 2 * m] = a_dx
        s_mat[2 * m, :m] = a_dx
        s_mat[2 * m, m : 2 * m] = a_dx
        s_mat[2 * m, 2 * m] = dx.sum()
        s_mat[np.diag_indices_from(s_mat)] += REGULARIZATION * (1.0 + np.abs(np.diag(s_mat)))
        factor = cho_factor(s_mat, lower=True, check_finite=False)

        def solve_newton(  # bind the iteration's state at definition time (B023)
            rc_lo: NDArray[np.float64],
            rc_ub: NDArray[np.float64],
            *,
            d: NDArray[np.float64] = d,
            rp: NDArray[np.float64] = rp,
            rd: NDArray[np.float64] = rd,
            ru: NDArray[np.float64] = ru,
            factor: tuple = factor,
            z: NDArray[np.float64] = z,
            v: NDArray[np.float64] = v,
            r_lo: NDArray[np.float64] = r_lo,
            q_ub: NDArray[np.float64] = q_ub,
        ) -> tuple:
            """Solve one Newton system at the given complementarity residual targets.

            The bound duals and the upper-bound slack eliminate against their product rows,
            leaving the normal equations in the equality duals; back-substitution recovers the
            variable step and the eliminated steps in order.
            """
            rhs_d = -rd + rc_lo / z - np.where(has_ub, (rc_ub - q_ub * ru) / v, 0.0)
            dy = cho_solve(factor, rp - m_dot(d * rhs_d), check_finite=False)
            dz = d * (mt_dot(dy) + rhs_d)
            dr = (rc_lo - r_lo * dz) / z
            dv = np.where(has_ub, ru - dz, 0.0)
            dq = np.where(has_ub, (rc_ub - q_ub * dv) / v, 0.0)
            return dz, dy, dr, dv, dq

        def max_steps(
            dz: NDArray[np.float64],
            dr: NDArray[np.float64],
            dv: NDArray[np.float64],
            dq: NDArray[np.float64],
            *,
            z: NDArray[np.float64] = z,
            v: NDArray[np.float64] = v,
            r_lo: NDArray[np.float64] = r_lo,
            q_ub: NDArray[np.float64] = q_ub,
        ) -> tuple[float, float]:
            """Return the primal/dual positivity-preserving step fractions for a direction."""

            def ratio(arr: NDArray[np.float64], darr: NDArray[np.float64]) -> float:
                neg = darr < 0.0
                return float((-arr[neg] / darr[neg]).min()) if neg.any() else np.inf

            a_pri = min(1.0, STEP_FRACTION * min(ratio(z, dz), ratio(v[has_ub], dv[has_ub])))
            a_dual = min(1.0, STEP_FRACTION * min(ratio(r_lo, dr), ratio(q_ub[has_ub], dq[has_ub])))
            return a_pri, a_dual

        # --- predictor --------------------------
        dz_a, _, dr_a, dv_a, dq_a = solve_newton(-z * r_lo, np.where(has_ub, -v * q_ub, 0.0))
        a_pri, a_dual = max_steps(dz_a, dr_a, dv_a, dq_a)
        mu_aff = (
            (z + a_pri * dz_a) @ (r_lo + a_dual * dr_a) + ((v + a_pri * dv_a) * (q_ub + a_dual * dq_a))[has_ub].sum()
        ) / (big_n + n)

        # --- centering weight + corrector -------
        sigma = min(1.0, (mu_aff / mu) ** 3)
        dz, dy, dr, dv, dq = solve_newton(
            sigma * mu - z * r_lo - dz_a * dr_a,
            np.where(has_ub, sigma * mu - v * q_ub - dv_a * dq_a, 0.0),
        )
        a_pri, a_dual = max_steps(dz, dr, dv, dq)

        # --- advance ----------------------------
        z += a_pri * dz
        v = np.where(has_ub, v + a_pri * dv, 1.0)
        y += a_dual * dy
        r_lo += a_dual * dr
        q_ub = np.where(has_ub, q_ub + a_dual * dq, 0.0)

    value = float(c @ z + 0.5 * (h * z) @ z)
    return RelaxationSolution(
        value=value,
        marginals=_shift_onto_sum_k(np.clip(z[i_x], 0.0, 1.0), k),
        lam_min=np.maximum(y[:m], 0.0),
        lam_max=np.maximum(-y[m : 2 * m], 0.0),
        iterations=iterations,
        converged=converged,
    )


# =================================================================================================
#  _shift_onto_sum_k
# =================================================================================================
def _shift_onto_sum_k(marginals: NDArray[np.float64], k: int) -> NDArray[np.float64]:
    """Return the marginals shifted (and re-clipped) so they sum to `k`, each staying in [0, 1].

    `RelaxationSolution.marginals` promises a sum of `k` — the property `rounding.systematic_sample`'s
    correctness rests on — but the post-solve clip can perturb the sum, and an unconverged solve
    can miss it outright.  Marginals already summing to `k` (within 1e-9) are returned unshifted;
    otherwise a bisection finds the constant shift whose clipped sum is `k` (the clipped sum is
    continuous and monotone in the shift, reaching 0 at -1 and `n >= k` at +1, so the target is
    always bracketed).
    """
    if abs(float(marginals.sum()) - k) <= 1e-9:
        return marginals
    lo, hi = -1.0, 1.0
    for _ in range(60):  # 60 halvings put the shift within ~2e-18 of exact
        mid = 0.5 * (lo + hi)
        if float(np.clip(marginals + mid, 0.0, 1.0).sum()) < k:
            lo = mid
        else:
            hi = mid
    return np.clip(marginals + 0.5 * (lo + hi), 0.0, 1.0)
