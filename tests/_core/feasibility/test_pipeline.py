import itertools

import numpy as np
import pytest
from scipy.optimize import LinearConstraint, milp

from max_div._core.constraints import Constraint, ConstraintList
from max_div._core.feasibility import FeasibilityStatus, find_feasible


# =================================================================================================
#  Helpers
# =================================================================================================
def _arrays(cons: list[Constraint]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert constraints to (con_values, con_indices, weights) as the pipeline ingests them."""
    con_values, con_indices = ConstraintList(cons).to_numpy()
    weights = np.array([con.weight for con in cons], dtype=np.float64)
    return con_values, con_indices, weights


def _selection_satisfies(selection: np.ndarray, cons: list[Constraint]) -> bool:
    """Check a selection against every constraint's [min_count, max_count]."""
    chosen = {int(j) for j in selection}
    return all(con.min_count <= len(con.int_set & chosen) <= con.max_count for con in cons)


def _brute_force_feasible(n: int, k: int, cons: list[Constraint]) -> bool:
    """Exhaustively check whether any k-selection satisfies all constraints (small n only)."""
    return any(_selection_satisfies(np.array(combo), cons) for combo in itertools.combinations(range(n), k))


def _recomputed_dual_value(n: int, k: int, cons: list[Constraint], lam_min: np.ndarray, lam_max: np.ndarray) -> float:
    """Independently recompute the dual value from the returned multipliers — the one-line certificate check."""
    scores = np.zeros(n)
    for i, con in enumerate(cons):
        for j in con.int_set:
            scores[j] += lam_min[i] - lam_max[i]
    mins = np.array([con.min_count for con in cons])
    maxs = np.array([con.max_count for con in cons])
    return float(lam_min @ mins - lam_max @ maxs - np.sort(scores)[-k:].sum())


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


def _run(n: int, k: int, cons: list[Constraint], seed: int = 0, thorough: bool = False):
    """Run the pipeline on the given constraints."""
    con_values, con_indices, weights = _arrays(cons)
    return find_feasible(con_values, con_indices, weights, n=n, k=k, seed=seed, thorough=thorough)


def _pigeonhole_instance() -> tuple[int, int, list[Constraint]]:
    """Return a certifiably infeasible instance: two disjoint min-2 sets with k = 2 (minimum violation 2)."""
    cons = [
        Constraint(int_set={0, 1}, min_count=2, max_count=2),
        Constraint(int_set={2, 3}, min_count=2, max_count=2),
    ]
    return 4, 2, cons


# =================================================================================================
#  Verdicts and proofs
# =================================================================================================
def test_pigeonhole_certified_infeasible():
    """The pigeonhole instance is proven infeasible with the exact violation floor of 2."""
    # --- act --------------------------
    n, k, cons = _pigeonhole_instance()
    result = _run(n, k, cons)

    # --- assert -----------------------
    assert result.status is FeasibilityStatus.INFEASIBLE
    assert result.violation_floor == pytest.approx(2.0, abs=1e-6)
    assert _recomputed_dual_value(n, k, cons, result.lam_min, result.lam_max) == pytest.approx(result.bound, abs=1e-9)


def test_witness_on_a_feasible_instance():
    """A plainly feasible instance yields a FEASIBLE verdict with a selection that satisfies everything."""
    # --- arrange ----------------------
    cons = [
        Constraint(int_set={0, 1, 2}, min_count=1, max_count=2),
        Constraint(int_set={3, 4, 5}, min_count=1, max_count=2),
    ]

    # --- act --------------------------
    result = _run(6, 3, cons)

    # --- assert -----------------------
    assert result.status is FeasibilityStatus.FEASIBLE
    assert result.violation == 0.0
    assert _selection_satisfies(result.selection, cons)


def test_no_constraints_is_trivially_feasible():
    """An empty constraint set makes any selection feasible; the pipeline says so."""
    # --- act --------------------------
    result = _run(6, 3, [])

    # --- assert -----------------------
    assert result.status is FeasibilityStatus.FEASIBLE
    assert result.selection.shape[0] == 3
    assert np.unique(result.selection).shape[0] == 3


def test_lp_feasible_integer_infeasible_returns_unknown():
    """The two-5-cycles instance is fractionally feasible but integrally infeasible, so it is UNKNOWN.

    Per-edge min-1 cover over two disjoint 5-cycles with k=5: no certificate can exist (the
    relaxation has value 0) and no witness can exist (covering one 5-cycle takes 3 vertices, so
    both need 6 > k) — the honest verdict under any correct algorithm.
    """
    # --- arrange ----------------------
    edges = [(i, (i + 1) % 5) for i in range(5)] + [(5 + i, 5 + (i + 1) % 5) for i in range(5)]
    cons = [Constraint(int_set={a, b}, min_count=1, max_count=2) for a, b in edges]

    # --- act --------------------------
    result = _run(10, 5, cons, thorough=True)

    # --- assert -----------------------
    assert result.status is FeasibilityStatus.UNKNOWN
    assert result.violation > 0.0
    assert result.violation_floor == 0.0


def test_scaled_weights_scale_the_floor():
    """Scaling all weights scales the certified violation floor with them."""
    # --- arrange ----------------------
    n, k, cons = _pigeonhole_instance()
    scaled = [Constraint(int_set=c.int_set, min_count=c.min_count, max_count=c.max_count, weight=2.5) for c in cons]

    # --- act --------------------------
    base = _run(n, k, cons)
    heavy = _run(n, k, scaled)

    # --- assert -----------------------
    assert heavy.status is FeasibilityStatus.INFEASIBLE
    assert heavy.violation_floor == pytest.approx(2.5 * base.violation_floor, abs=1e-6)


def test_fractional_weights_keep_an_unrounded_floor():
    """Fractional weights certify a fractional violation floor, unrounded."""
    # --- arrange ----------------------
    n, k, cons = _pigeonhole_instance()
    fractional = [
        Constraint(int_set=c.int_set, min_count=c.min_count, max_count=c.max_count, weight=0.75) for c in cons
    ]

    # --- act --------------------------
    result = _run(n, k, fractional)

    # --- assert -----------------------
    assert result.status is FeasibilityStatus.INFEASIBLE
    assert result.violation_floor == pytest.approx(1.5, abs=1e-6)


# =================================================================================================
#  Never-wrong and cross-check properties
# =================================================================================================
@pytest.mark.parametrize("seed", range(20))
def test_never_wrong_property(seed: int):
    """Verdicts are proofs: FEASIBLE returns a satisfying selection, INFEASIBLE never contradicts brute force."""
    # --- arrange ----------------------
    n, k, cons = _random_instance(seed)

    # --- act --------------------------
    result = _run(n, k, cons, seed=seed)

    # --- assert -----------------------
    if result.status is FeasibilityStatus.FEASIBLE:
        assert _selection_satisfies(result.selection, cons)
    elif result.status is FeasibilityStatus.INFEASIBLE:
        assert not _brute_force_feasible(n, k, cons)
        assert _recomputed_dual_value(n, k, cons, result.lam_min, result.lam_max) > 0.0


@pytest.mark.parametrize("seed", range(10))
def test_metamorphic_tightening_never_creates_feasibility(seed: int):
    """Tightening every bound can only move the verdict toward infeasibility, never away from it."""
    # --- arrange ----------------------
    n, k, cons = _random_instance(seed)
    tightened = [
        Constraint(
            int_set=con.int_set,
            min_count=min(con.min_count + 1, len(con.int_set)),
            max_count=max(con.max_count - 1, min(con.min_count + 1, len(con.int_set))),
        )
        for con in cons
    ]

    # --- act --------------------------
    base = _run(n, k, cons, seed=seed)
    tight = _run(n, k, tightened, seed=seed)

    # --- assert -----------------------
    if base.status is FeasibilityStatus.INFEASIBLE:
        assert not _brute_force_feasible(n, k, cons)
    if tight.status is FeasibilityStatus.FEASIBLE:
        assert _selection_satisfies(tight.selection, tightened)


@pytest.mark.parametrize("seed", range(12))
def test_milp_cross_check(seed: int):
    """An exact integer solve agrees with every definite verdict."""
    # --- arrange ----------------------
    n, k, cons = _random_instance(seed)
    a = np.zeros((len(cons), n))
    for i, con in enumerate(cons):
        for j in con.int_set:
            a[i, j] = 1.0
    lo = np.array([con.min_count for con in cons], dtype=float)
    hi = np.array([con.max_count for con in cons], dtype=float)

    # --- act --------------------------
    result = _run(n, k, cons, seed=seed)
    milp_result = milp(
        c=np.zeros(n),
        constraints=[LinearConstraint(a, lo, hi), LinearConstraint(np.ones((1, n)), k, k)],
        integrality=np.ones(n),
        bounds=(0, 1),
    )
    milp_feasible = milp_result.status == 0

    # --- assert -----------------------
    if result.status is FeasibilityStatus.FEASIBLE:
        assert milp_feasible
    elif result.status is FeasibilityStatus.INFEASIBLE:
        assert not milp_feasible


# =================================================================================================
#  Determinism and seed variation
# =================================================================================================
def test_determinism_same_inputs_same_outputs():
    """Equal seeds give bit-equal results; the relaxation solve underneath is deterministic."""
    # --- arrange ----------------------
    n, k, cons = _random_instance(3)

    # --- act --------------------------
    first = _run(n, k, cons, seed=7)
    second = _run(n, k, cons, seed=7)

    # --- assert -----------------------
    assert first.status is second.status
    assert np.array_equal(first.selection, second.selection)
    assert first.violation == second.violation
    assert first.bound == second.bound


def test_different_seeds_vary_the_selection_on_an_easy_instance():
    """On a loosely constrained instance, different seeds give genuinely different feasible selections."""
    # --- arrange ----------------------
    cons = [Constraint(int_set=set(range(40)), min_count=1, max_count=10)]

    # --- act --------------------------
    selections = {tuple(sorted(_run(40, 8, cons, seed=s).selection.tolist())) for s in range(8)}

    # --- assert -----------------------
    assert len(selections) > 1


@pytest.mark.parametrize("seed", range(6))
def test_per_constraint_violation_reproduces_the_total(seed: int):
    """The per-constraint profile describes the returned selection: weighted, it sums to `violation`."""
    # --- arrange ----------------------
    n, k, cons = _random_instance(seed)
    _, _, weights = _arrays(cons)

    # --- act --------------------------
    result = _run(n, k, cons, seed=seed)

    # --- assert -----------------------
    assert result.violation_per_constraint.shape[0] == len(cons)
    assert (result.violation_per_constraint >= 0).all()
    assert float(weights @ result.violation_per_constraint) == pytest.approx(result.violation)
    assert result.violation_floor <= result.violation + 1e-9  # the floor can never exceed what was achieved


def test_deterministic_fallback_can_beat_the_draws(monkeypatch):
    """When every draw ends badly, the deterministic round's better selection is the one returned."""
    # --- arrange ----------------------
    from max_div._core.feasibility import pipeline

    n, k, cons = _pigeonhole_instance()

    def _bad_draw(marginals, k, rng_state, *arrays):
        return np.arange(k, dtype=np.int64), 99.0

    monkeypatch.setattr(pipeline, "sample_and_repair", _bad_draw)

    # --- act --------------------------
    result = _run(n, k, cons)

    # --- assert -----------------------
    assert result.violation < 99.0  # the fallback's selection won


# =================================================================================================
#  Forced full selection (k == n)
# =================================================================================================
def test_k_equals_n_feasible_when_the_full_selection_satisfies():
    """k == n short-circuits to the only possible selection; when it satisfies the constraints, FEASIBLE."""
    # --- arrange ----------------------
    cons = [Constraint(int_set={0, 1, 2}, min_count=1, max_count=5)]

    # --- act --------------------------
    result = _run(6, 6, cons)

    # --- assert -----------------------
    assert result.status is FeasibilityStatus.FEASIBLE
    assert np.array_equal(result.selection, np.arange(6))
    assert result.violation == 0.0


def test_k_equals_n_infeasible_with_an_exact_recheckable_violation_floor():
    """A max_count below a set's size is violated by the forced full selection, with an exact violation floor.

    The verdict is decided by enumeration (only one selection exists), but the returned
    multipliers still re-verify through the standard dual-value check.
    """
    # --- arrange ----------------------
    cons = [Constraint(int_set={0, 1, 2, 3}, min_count=0, max_count=1)]  # 4 members, max 1 -> excess 3

    # --- act --------------------------
    result = _run(6, 6, cons)

    # --- assert -----------------------
    assert result.status is FeasibilityStatus.INFEASIBLE
    assert result.violation == 3.0
    assert result.violation_floor == pytest.approx(3.0)
    assert _recomputed_dual_value(6, 6, cons, result.lam_min, result.lam_max) == pytest.approx(3.0)


def test_k_equals_n_minus_one_still_runs_the_full_pipeline():
    """k == n - 1 does not take the k == n short-circuit: the relaxation pipeline runs and returns 5 valid items."""
    # --- arrange ----------------------
    cons = [Constraint(int_set={0, 1, 2}, min_count=1, max_count=3)]

    # --- act --------------------------
    result = _run(6, 5, cons)

    # --- assert -----------------------
    assert result.status is FeasibilityStatus.FEASIBLE
    assert result.selection.shape[0] == 5
    assert len(set(result.selection.tolist())) == 5
