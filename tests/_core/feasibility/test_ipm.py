import numpy as np
import pytest
from scipy.optimize import linprog

from max_div._core.constraints import Constraint, ConstraintList
from max_div._core.feasibility import find_feasible
from max_div._core.feasibility.evaluation import certified_bound, clamp_admissible
from max_div._core.feasibility.indexing import build_item_constraint_csr
from max_div._core.feasibility.ipm import RelaxationSolution, solve_relaxation


# =================================================================================================
#  Helpers
# =================================================================================================
def _arrays(cons: list[Constraint]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert constraints to (con_min, con_max, con_indices) as the solver ingests them."""
    con_values, con_indices = ConstraintList(cons).to_numpy()
    return con_values[:, 0].astype(np.int64), con_values[:, 1].astype(np.int64), con_indices


def _random_instance(seed: int) -> tuple[int, int, list[Constraint]]:
    """Generate a small random instance with overlapping constraints (feasible or not)."""
    rng = np.random.default_rng(seed)
    n = int(rng.integers(6, 13))
    k = int(rng.integers(2, n - 1))
    m = int(rng.integers(2, 5))
    cons = []
    for _ in range(m):
        size = int(rng.integers(2, n))
        int_set = {int(j) for j in rng.choice(n, size=size, replace=False)}
        lo = int(rng.integers(0, min(k, len(int_set)) + 1))
        hi = int(rng.integers(lo, len(int_set) + 1))
        cons.append(Constraint(int_set=int_set, min_count=lo, max_count=hi))
    return n, k, cons


def _linprog_value(n: int, k: int, cons: list[Constraint], w_lin: np.ndarray) -> float:
    """Independently solve the linear-penalty relaxation with scipy's LP solver."""
    m = len(cons)
    a = np.zeros((m, n))
    for i, con in enumerate(cons):
        for j in con.int_set:
            a[i, j] = 1.0
    lo = np.array([con.min_count for con in cons], dtype=float)
    hi = np.array([con.max_count for con in cons], dtype=float)
    # variables x = (z, s_minus, s_plus); minimize w_lin . (s_minus + s_plus)
    c = np.concatenate([np.zeros(n), w_lin, w_lin])
    a_ub = np.block([[-a, -np.eye(m), np.zeros((m, m))], [a, np.zeros((m, m)), -np.eye(m)]])
    b_ub = np.concatenate([-lo, hi])
    a_eq = np.concatenate([np.ones(n), np.zeros(2 * m)]).reshape(1, -1)
    bounds = [(0.0, 1.0)] * n + [(0.0, None)] * (2 * m)
    res = linprog(c, A_ub=a_ub, b_ub=b_ub, A_eq=a_eq, b_eq=[float(k)], bounds=bounds, method="highs")
    assert res.success
    return float(res.fun)


def _solve(n: int, k: int, cons: list[Constraint], w_lin: np.ndarray, w_quad: np.ndarray) -> RelaxationSolution:
    """Run the IPM on the given instance."""
    con_min, con_max, con_indices = _arrays(cons)
    return solve_relaxation(con_min, con_max, w_lin, w_quad, con_indices, n=n, k=k)


def _bound(sol: RelaxationSolution, n: int, k: int, cons: list[Constraint], w_lin, w_quad) -> float:
    """Clamp the solve's multipliers and evaluate the certified bound exactly."""
    con_min, con_max, con_indices = _arrays(cons)
    item_indptr, item_cons = build_item_constraint_csr(con_indices, n)
    lam_min = clamp_admissible(sol.lam_min, w_lin, w_quad)
    lam_max = clamp_admissible(sol.lam_max, w_lin, w_quad)
    return float(certified_bound(con_min, con_max, w_lin, w_quad, lam_min, lam_max, item_indptr, item_cons, k))


# =================================================================================================
#  solve_relaxation
# =================================================================================================
def test_pigeonhole_solved_exactly():
    """The two-disjoint-min-2-sets instance has relaxed optimum 2; the IPM must find it exactly."""
    # --- arrange ----------------------
    cons = [
        Constraint(int_set={0, 1}, min_count=2, max_count=2),
        Constraint(int_set={2, 3}, min_count=2, max_count=2),
    ]
    w = np.ones(2)

    # --- act --------------------------
    sol = _solve(4, 2, cons, w, np.zeros(2))

    # --- assert -----------------------
    assert sol.converged
    assert sol.value == pytest.approx(2.0, abs=1e-6)
    assert _bound(sol, 4, 2, cons, w, np.zeros(2)) == pytest.approx(2.0, abs=1e-6)
    # the analytic center of the symmetric optimal set: every marginal at 1/2
    assert sol.marginals == pytest.approx(np.full(4, 0.5), abs=1e-4)


@pytest.mark.parametrize("seed", range(8))
def test_linear_value_matches_linprog(seed: int):
    """On linear penalties the IPM value must agree with an independent LP solve."""
    # --- arrange ----------------------
    n, k, cons = _random_instance(seed)
    rng = np.random.default_rng(seed + 1000)
    w_lin = rng.uniform(0.5, 3.0, len(cons))

    # --- act --------------------------
    sol = _solve(n, k, cons, w_lin, np.zeros(len(cons)))

    # --- assert -----------------------
    assert sol.converged
    assert sol.value == pytest.approx(_linprog_value(n, k, cons, w_lin), abs=1e-6)


@pytest.mark.parametrize("seed", range(8))
def test_strong_duality_under_mixed_penalties(seed: int):
    """The exactly evaluated bound at the clamped multipliers must close the gap to the value."""
    # --- arrange ----------------------
    n, k, cons = _random_instance(seed)
    rng = np.random.default_rng(seed + 2000)
    w_lin = rng.uniform(0.0, 2.0, len(cons))
    w_quad = rng.uniform(0.0, 2.0, len(cons))
    w_quad[w_lin < 0.5] += 0.5  # keep every phi_i strictly increasing

    # --- act --------------------------
    sol = _solve(n, k, cons, w_lin, w_quad)
    bound = _bound(sol, n, k, cons, w_lin, w_quad)

    # --- assert -----------------------
    assert sol.converged
    assert bound <= sol.value + 1e-6  # weak duality, always
    assert bound == pytest.approx(sol.value, abs=1e-5)  # strong duality at the optimum


@pytest.mark.parametrize("seed", range(8))
def test_marginals_are_a_valid_fractional_selection(seed: int):
    """Marginals must lie in the box, sum to k, and respect the counts whenever the value is zero."""
    # --- arrange ----------------------
    n, k, cons = _random_instance(seed)
    w = np.ones(len(cons))

    # --- act --------------------------
    sol = _solve(n, k, cons, w, np.zeros(len(cons)))

    # --- assert -----------------------
    assert np.all(sol.marginals >= 0.0)
    assert np.all(sol.marginals <= 1.0)
    assert sol.marginals.sum() == pytest.approx(k, abs=1e-6)
    if sol.value < 1e-8:
        counts = np.array([sol.marginals[list(con.int_set)].sum() for con in cons])
        lo = np.array([con.min_count for con in cons])
        hi = np.array([con.max_count for con in cons])
        assert np.all(counts >= lo - 1e-5)
        assert np.all(counts <= hi + 1e-5)


def test_bound_not_below_the_ascent_floor():
    """The exact solve's violation floor must be at least the dual ascent's."""
    # --- arrange ----------------------
    cons = [
        Constraint(int_set={0, 1}, min_count=2, max_count=2),
        Constraint(int_set={2, 3}, min_count=2, max_count=2),
        Constraint(int_set={1, 2, 4}, min_count=0, max_count=1),
    ]
    n, k = 6, 2
    w = np.ones(3)
    con_values, con_indices = ConstraintList(cons).to_numpy()
    ascent = find_feasible(con_values, con_indices, w, k=k, n=n, max_iter=2000, seed=1)

    # --- act --------------------------
    sol = _solve(n, k, cons, w, np.zeros(3))
    bound = _bound(sol, n, k, cons, w, np.zeros(3))

    # --- assert -----------------------
    assert bound >= ascent.violation_floor - 1e-6


def test_lp_feasible_integer_infeasible_has_zero_value():
    """Two disjoint 5-cycles with per-edge min-1 cover: fractionally feasible, so value and bound are 0."""
    # --- arrange ----------------------
    edges = [(i, (i + 1) % 5) for i in range(5)] + [(5 + i, 5 + (i + 1) % 5) for i in range(5)]
    cons = [Constraint(int_set={a, b}, min_count=1, max_count=2) for a, b in edges]
    w = np.ones(10)

    # --- act --------------------------
    sol = _solve(10, 5, cons, w, np.zeros(10))

    # --- assert -----------------------
    assert sol.converged
    assert sol.value == pytest.approx(0.0, abs=1e-6)
    assert _bound(sol, 10, 5, cons, w, np.zeros(10)) <= 1e-6


def test_unconstrained_problem_returns_uniform_marginals():
    """With no constraints the relaxation's most interior optimum is the uniform selection."""
    # --- act --------------------------
    sol = solve_relaxation(
        np.empty(0, dtype=np.int64),
        np.empty(0, dtype=np.int64),
        np.empty(0),
        np.empty(0),
        np.zeros(0, dtype=np.int32),
        n=6,
        k=3,
    )

    # --- assert -----------------------
    assert sol.converged
    assert sol.value == pytest.approx(0.0, abs=1e-9)
    assert sol.marginals == pytest.approx(np.full(6, 0.5), abs=1e-6)


def test_iteration_cap_reports_unconverged(monkeypatch):
    """Hitting the safety cap must be reported, while the outputs stay well-formed."""
    # --- arrange ----------------------
    from max_div._core.feasibility import ipm

    monkeypatch.setattr(ipm, "MAX_ITERATIONS", 1)
    cons = [Constraint(int_set={0, 1}, min_count=2, max_count=2)]

    # --- act --------------------------
    con_min, con_max, con_indices = _arrays(cons)
    sol = ipm.solve_relaxation(con_min, con_max, np.ones(1), np.zeros(1), con_indices, n=4, k=2)

    # --- assert -----------------------
    assert not sol.converged
    assert sol.iterations == 1
    assert np.all(sol.marginals >= 0.0)
    assert np.all(sol.marginals <= 1.0)
