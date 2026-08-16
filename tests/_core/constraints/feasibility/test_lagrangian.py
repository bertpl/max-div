import itertools

import numpy as np
import pytest
from scipy.optimize import LinearConstraint, milp

from max_div._core._random import new_rng_state
from max_div._core.constraints import Constraint, ConstraintList
from max_div._core.constraints.feasibility import (
    FeasibilityStatus,
    construction_iteration_budget_iterations,
    construction_iteration_budget_seconds,
    find_feasible,
)
from max_div._core.constraints.feasibility.lagrangian import (
    CONSTRUCTION_MAX_ITER,
    CONSTRUCTION_MIN_ITER,
    _dual_value,
    _gumbel_noise,
    _item_scores,
    _top_k_items,
    build_item_constraint_csr,
)


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
    """Independently recompute g from returned multipliers — the one-line certificate check."""
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


def _pigeonhole_instance() -> tuple[int, int, list[Constraint]]:
    """Return a certifiably infeasible instance: two disjoint min-2 sets with k = 2 (minimum violation 2)."""
    cons = [
        Constraint(int_set={0, 1}, min_count=2, max_count=2),
        Constraint(int_set={2, 3}, min_count=2, max_count=2),
    ]
    return 4, 2, cons


# =================================================================================================
#  Item <-> constraint indexing
# =================================================================================================
def test_build_item_constraint_csr():
    """The transpose lists each item's constraints exactly."""
    # --- arrange -----------------------------------------
    cons = [
        Constraint(int_set={0, 1, 2}, min_count=1, max_count=3),
        Constraint(int_set={2, 3}, min_count=0, max_count=2),
    ]
    _, con_indices, _ = _arrays(cons)

    # --- act ---------------------------------------------
    item_indptr, item_cons = build_item_constraint_csr(con_indices, 5)

    # --- assert ------------------------------------------
    memberships = {j: sorted(item_cons[item_indptr[j] : item_indptr[j + 1]]) for j in range(5)}
    assert memberships == {0: [0], 1: [0], 2: [0, 1], 3: [1], 4: []}


def test_build_item_constraint_csr_no_constraints():
    """An empty constraint set transposes to an all-empty CSR."""
    # --- act ---------------------------------------------
    item_indptr, item_cons = build_item_constraint_csr(np.empty(0, dtype=np.int32), 3)

    # --- assert ------------------------------------------
    assert item_indptr.tolist() == [0, 0, 0, 0]
    assert item_cons.size == 0


def test_top_k_items_is_exact():
    """Heap selection returns exactly the k best items, and all items when k reaches n."""
    # --- arrange -----------------------------------------
    scores = np.array([3.0, 1.0, 2.0, 5.0, 4.0])

    # --- act ---------------------------------------------
    top2 = _top_k_items(scores, 2)
    top_all = _top_k_items(scores, 5)

    # --- assert ------------------------------------------
    assert sorted(top2.tolist()) == [3, 4]
    assert sorted(top_all.tolist()) == [0, 1, 2, 3, 4]


# =================================================================================================
#  Dual value and the certificate
# =================================================================================================
def test_dual_value_pigeonhole_toy():
    """All-ones prices on the pigeonhole instance give g = 2."""
    # --- arrange -----------------------------------------
    n, k, cons = _pigeonhole_instance()
    _, con_indices, _ = _arrays(cons)
    item_indptr, item_cons = build_item_constraint_csr(con_indices, n)
    lam_min = np.ones(2)
    lam_max = np.zeros(2)

    # --- act ---------------------------------------------
    scores = _item_scores(item_indptr, item_cons, lam_min, lam_max)
    selection = _top_k_items(scores, k)
    g = _dual_value(
        np.array([2, 2], dtype=np.int64), np.array([2, 2], dtype=np.int64), lam_min, lam_max, scores, selection
    )

    # --- assert ------------------------------------------
    assert scores.tolist() == [1.0, 1.0, 1.0, 1.0]
    assert g == pytest.approx(2.0)


def test_exact_topk_guard():
    """A non-maximizing inner selection fabricates a positive g on a feasible instance.

    Pins the soundness invariant: the ascent's dual value is a proof only because it uses an exact
    top-k — this test demonstrates the false proof a perturbed selection would produce, and that
    the exact selection stays non-positive (as any feasible instance requires).
    """
    # --- arrange -----------------------------------------
    cons = [
        Constraint(int_set={0, 1}, min_count=1, max_count=2, weight=3.0),
        Constraint(int_set={2, 3}, min_count=1, max_count=2, weight=1.0),
    ]
    n, k = 5, 2  # item 4 is unconstrained; {0, 2} is a witness, so the instance is feasible
    _, con_indices, _ = _arrays(cons)
    item_indptr, item_cons = build_item_constraint_csr(con_indices, n)
    con_min = np.array([1, 1], dtype=np.int64)
    con_max = np.array([2, 2], dtype=np.int64)
    lam_min = np.array([3.0, 1.0])
    lam_max = np.zeros(2)
    scores = _item_scores(item_indptr, item_cons, lam_min, lam_max)

    # --- act ---------------------------------------------
    g_exact = _dual_value(con_min, con_max, lam_min, lam_max, scores, _top_k_items(scores, k))
    g_corrupted = _dual_value(con_min, con_max, lam_min, lam_max, scores, np.array([3, 4], dtype=np.int64))

    # --- assert ------------------------------------------
    assert g_exact <= 0.0
    assert g_corrupted > 0.0


def test_pigeonhole_certified_infeasible():
    """The pigeonhole instance is certified with a verifiable bound and an optimal least-infeasible selection."""
    # --- arrange -----------------------------------------
    n, k, cons = _pigeonhole_instance()
    con_values, con_indices, weights = _arrays(cons)

    # --- act ---------------------------------------------
    result = find_feasible(con_values, con_indices, weights, n, k, max_iter=300)

    # --- assert ------------------------------------------
    assert result.status == FeasibilityStatus.INFEASIBLE
    assert result.bound > 0.0
    # the certificate is independently verifiable from the multipliers alone
    assert _recomputed_dual_value(n, k, cons, result.lam_min, result.lam_max) == pytest.approx(result.bound)
    # the least-infeasible selection attains the brute-force minimum violation
    assert result.selection.shape[0] == k
    assert result.violation == pytest.approx(2.0)
    assert result.violation >= result.bound - 1e-9


def test_verdict_mode_certifies_early():
    """Verdict mode still returns a certified INFEASIBLE."""
    # --- arrange -----------------------------------------
    n, k, cons = _pigeonhole_instance()
    con_values, con_indices, weights = _arrays(cons)

    # --- act ---------------------------------------------
    result = find_feasible(con_values, con_indices, weights, n, k, max_iter=300, stop_at_first_proof=True)

    # --- assert ------------------------------------------
    assert result.status == FeasibilityStatus.INFEASIBLE
    assert result.bound > 0.0


# =================================================================================================
#  Witness construction
# =================================================================================================
def test_witness_from_ascent():
    """An instance whose priced top-k becomes feasible mid-ascent returns that witness directly."""
    # --- arrange -----------------------------------------
    cons = [Constraint(int_set={2, 3}, min_count=2, max_count=2)]
    con_values, con_indices, weights = _arrays(cons)

    # --- act ---------------------------------------------
    result = find_feasible(con_values, con_indices, weights, 4, 2, max_iter=300)

    # --- assert ------------------------------------------
    assert result.status == FeasibilityStatus.FEASIBLE
    assert result.violation == 0.0
    assert sorted(result.selection.tolist()) == [2, 3]


def test_no_constraints_is_trivially_feasible():
    """With no constraints, any selection is a witness."""
    # --- act ---------------------------------------------
    result = find_feasible(
        np.empty((0, 2), dtype=np.int32), np.empty(0, dtype=np.int32), np.empty(0, dtype=np.float64), 6, 3, max_iter=50
    )

    # --- assert ------------------------------------------
    assert result.status == FeasibilityStatus.FEASIBLE
    assert result.selection.shape[0] == 3
    assert result.violation == 0.0


def test_scaled_weights_still_construct():
    """Noise is relative to the weight scale, so large weights must not break tie-breaking."""
    # --- arrange -----------------------------------------
    cons = [
        Constraint(int_set={0, 1, 2}, min_count=1, max_count=1, weight=1000.0),
        Constraint(int_set={3, 4, 5}, min_count=2, max_count=3, weight=1000.0),
    ]
    con_values, con_indices, weights = _arrays(cons)

    # --- act ---------------------------------------------
    result = find_feasible(con_values, con_indices, weights, 6, 3, max_iter=300)

    # --- assert ------------------------------------------
    assert result.status == FeasibilityStatus.FEASIBLE
    assert result.violation == 0.0
    assert _selection_satisfies(result.selection, cons)


def test_fractional_weights_keep_unrounded_floor():
    """Non-integral weights forgo the integer sharpening of the violation floor."""
    # --- arrange -----------------------------------------
    cons = [
        Constraint(int_set={0, 1}, min_count=2, max_count=2, weight=1.5),
        Constraint(int_set={2, 3}, min_count=2, max_count=2, weight=1.5),
    ]
    con_values, con_indices, weights = _arrays(cons)

    # --- act ---------------------------------------------
    result = find_feasible(con_values, con_indices, weights, 4, 2, max_iter=300)

    # --- assert ------------------------------------------
    assert result.status == FeasibilityStatus.INFEASIBLE
    assert result.bound > 0.0
    assert result.selection.shape[0] == 2
    assert result.violation >= result.bound - 1e-9


def test_lp_feasible_integer_infeasible_returns_unknown():
    """An LP-feasible but integer-infeasible instance must land in UNKNOWN.

    Two disjoint 5-cycles, one min-1 cover constraint per edge, k = 5.  A fractional 1/2 per item
    covers every edge with total exactly 5, so no positive-dual certificate exists; covering each
    odd cycle integrally needs 3 items (6 > k), so no witness exists either — UNKNOWN is forced.
    """
    # --- arrange -----------------------------------------
    cons = [
        Constraint(int_set={cycle * 5 + i, cycle * 5 + (i + 1) % 5}, min_count=1, max_count=2)
        for cycle in range(2)
        for i in range(5)
    ]
    con_values, con_indices, weights = _arrays(cons)

    # --- act ---------------------------------------------
    result = find_feasible(con_values, con_indices, weights, 10, 5, max_iter=300)

    # --- assert ------------------------------------------
    assert result.status == FeasibilityStatus.UNKNOWN
    assert result.bound <= 0.0
    assert result.violation > 0.0
    assert result.selection.shape[0] == 5


def test_planted_matching_solved_by_ascent():
    """A planted-matching instance is solved via the ascent's feasible-top-k exit.

    With unit row/column quotas the price dynamics behave like an auction: starved rows and
    columns bid up their members until the top-k is a perfect matching, which the ascent's
    feasible-top-k check returns as a witness.
    """
    # --- arrange -----------------------------------------
    rng = np.random.default_rng(5)
    n_rows = 10
    cells = [(i, i) for i in range(n_rows)]  # planted perfect matching guarantees feasibility
    for i in range(n_rows):
        for j in range(n_rows):
            if i != j and rng.random() < 0.10:
                cells.append((i, j))
    cons = [
        Constraint(int_set={t for t, (r, _) in enumerate(cells) if r == i}, min_count=1, max_count=1)
        for i in range(n_rows)
    ] + [
        Constraint(int_set={t for t, (_, c) in enumerate(cells) if c == j}, min_count=1, max_count=1)
        for j in range(n_rows)
    ]
    n, k = len(cells), n_rows
    con_values, con_indices, weights = _arrays(cons)

    # --- act ---------------------------------------------
    result = find_feasible(con_values, con_indices, weights, n, k, max_iter=300)

    # --- assert ------------------------------------------
    assert result.status == FeasibilityStatus.FEASIBLE
    assert result.violation == 0.0
    assert _selection_satisfies(result.selection, cons)


# =================================================================================================
#  Soundness properties
# =================================================================================================
@pytest.mark.parametrize("seed", range(30))
def test_never_wrong_property(seed: int):
    """Both definite verdicts must be correct on random instances; UNKNOWN claims nothing."""
    # --- arrange -----------------------------------------
    n, k, cons = _random_instance(seed)
    con_values, con_indices, weights = _arrays(cons)

    # --- act ---------------------------------------------
    result = find_feasible(con_values, con_indices, weights, n, k, max_iter=300, seed=seed)

    # --- assert ------------------------------------------
    if result.status == FeasibilityStatus.FEASIBLE:
        assert result.violation == 0.0
        assert result.selection.shape[0] == k
        assert _selection_satisfies(result.selection, cons)
    elif result.status == FeasibilityStatus.INFEASIBLE:
        assert result.bound > 0.0
        assert _recomputed_dual_value(n, k, cons, result.lam_min, result.lam_max) == pytest.approx(result.bound)
        assert not _brute_force_feasible(n, k, cons)
    else:
        assert result.status == FeasibilityStatus.UNKNOWN  # no claim to check — but only these three statuses may occur


@pytest.mark.parametrize("seed", range(10))
def test_metamorphic_tightening_never_creates_feasibility(seed: int):
    """Raising a min_count can only shrink the feasible set, so INFEASIBLE must not flip to FEASIBLE."""
    # --- arrange -----------------------------------------
    n, k, cons = _random_instance(seed)
    con_values, con_indices, weights = _arrays(cons)
    tightened = [
        Constraint(
            int_set=con.int_set,
            min_count=min(con.min_count + 1, len(con.int_set)),
            max_count=con.max_count,
            weight=con.weight,
        )
        for con in cons
    ]
    t_con_values, t_con_indices, t_weights = _arrays(tightened)

    # --- act ---------------------------------------------
    result = find_feasible(con_values, con_indices, weights, n, k, max_iter=300, seed=seed)
    t_result = find_feasible(t_con_values, t_con_indices, t_weights, n, k, max_iter=300, seed=seed)

    # --- assert ------------------------------------------
    assert not (result.status == FeasibilityStatus.INFEASIBLE and t_result.status == FeasibilityStatus.FEASIBLE)


def test_milp_cross_check():
    """A medium instance's verdict agrees with an exact MILP arbiter."""
    # --- arrange -----------------------------------------
    rng = np.random.default_rng(7)
    n, k = 40, 12
    cons = [
        Constraint(int_set={int(j) for j in rng.choice(n, size=10, replace=False)}, min_count=4, max_count=6)
        for _ in range(6)
    ]
    con_values, con_indices, weights = _arrays(cons)
    incidence = np.zeros((len(cons), n))
    for i, con in enumerate(cons):
        incidence[i, list(con.int_set)] = 1.0
    milp_result = milp(
        c=np.zeros(n),
        constraints=[
            LinearConstraint(np.ones((1, n)), lb=k, ub=k),
            LinearConstraint(incidence, lb=[c.min_count for c in cons], ub=[c.max_count for c in cons]),
        ],
        integrality=np.ones(n),
        bounds=(0, 1),
    )
    milp_feasible = milp_result.status == 0

    # --- act ---------------------------------------------
    result = find_feasible(con_values, con_indices, weights, n, k, max_iter=500)

    # --- assert ------------------------------------------
    if result.status == FeasibilityStatus.FEASIBLE:
        assert milp_feasible
        assert _selection_satisfies(result.selection, cons)
    if result.status == FeasibilityStatus.INFEASIBLE:
        assert not milp_feasible


# =================================================================================================
#  Determinism and noise
# =================================================================================================
def test_determinism_same_inputs_same_outputs():
    """Identical inputs and seed reproduce the full output tuple."""
    # --- arrange -----------------------------------------
    n, k, cons = _random_instance(3)
    con_values, con_indices, weights = _arrays(cons)

    # --- act ---------------------------------------------
    first = find_feasible(con_values, con_indices, weights, n, k, max_iter=300, seed=11)
    second = find_feasible(con_values, con_indices, weights, n, k, max_iter=300, seed=11)

    # --- assert ------------------------------------------
    assert first.status == second.status
    assert first.bound == second.bound
    assert first.violation == second.violation
    np.testing.assert_array_equal(first.selection, second.selection)
    np.testing.assert_array_equal(first.lam_min, second.lam_min)
    np.testing.assert_array_equal(first.lam_max, second.lam_max)


def test_gumbel_noise_seed_behavior():
    """Noise reproduces per seed and differs across seeds."""
    # --- act ---------------------------------------------
    noise_a = _gumbel_noise(100, new_rng_state(1))
    noise_b = _gumbel_noise(100, new_rng_state(1))
    noise_c = _gumbel_noise(100, new_rng_state(2))

    # --- assert ------------------------------------------
    np.testing.assert_array_equal(noise_a, noise_b)
    assert not np.array_equal(noise_a, noise_c)


# =================================================================================================
#  Iteration budgets
# =================================================================================================
@pytest.mark.parametrize(
    "t_max_sec,expected",
    [
        (0.001, CONSTRUCTION_MIN_ITER),  # tiny budget clamps to the floor
        (1e6, CONSTRUCTION_MAX_ITER),  # huge budget clamps to the ceiling
    ],
)
def test_construction_budget_seconds_clamps(t_max_sec: float, expected: int):
    """Extreme time budgets clamp to the iteration floor and ceiling."""
    # --- act ---------------------------------------------
    budget = construction_iteration_budget_seconds(t_max_sec, n=1000, n_memberships=5000)

    # --- assert ------------------------------------------
    assert budget == expected


def test_construction_budget_seconds_scales_with_problem_size():
    """A fixed time budget buys fewer iterations on a larger problem."""
    # --- act ---------------------------------------------
    small_problem = construction_iteration_budget_seconds(1.0, n=1000, n_memberships=5000)
    large_problem = construction_iteration_budget_seconds(1.0, n=100_000, n_memberships=5_000_000)

    # --- assert ------------------------------------------
    assert small_problem >= large_problem


@pytest.mark.parametrize(
    "n_solver_iterations,expected",
    [
        (100, CONSTRUCTION_MIN_ITER),
        (20_000, 2000),  # the `BUDGET_FRACTION` share, un-clamped
        (10_000_000, CONSTRUCTION_MAX_ITER),
    ],
)
def test_construction_budget_iterations(n_solver_iterations: int, expected: int):
    """Iteration-typed budgets take the fixed share, clamped."""
    # --- act ---------------------------------------------
    budget = construction_iteration_budget_iterations(n_solver_iterations)

    # --- assert ------------------------------------------
    assert budget == expected
