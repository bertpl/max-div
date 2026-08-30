"""The feasibility pipeline decides whether a feasible selection exists and, where possible, constructs one.

Can exactly `k` of `n` items be selected such that every constraint `i` counts between
`min_count_i` and `max_count_i` selected members?  Constraint sets may overlap
arbitrarily, which makes the decision NP-complete, so the pipeline returns one of three statuses:

- `FEASIBLE`: a witness selection satisfying every constraint is returned.
- `INFEASIBLE`: multipliers `(lam_min, lam_max)` with a positive certified bound are returned — a
  proof that no feasible selection exists, and a lower bound on the weighted violation of every
  possible selection (`find_feasible` documents the verification).
- `UNKNOWN`: neither was found.  This verdict carries no information; callers must behave as if
  nothing was learned.

The mechanism has three stages:

- `ipm.solve_relaxation` solves the continuous relaxation exactly.
- `evaluation.certified_bound`, at the clamped multipliers, proves infeasibility when positive.
- `rounding` draws seed-controlled selections from the marginals, each swap-repaired; a
  zero-violation draw is a witness, and the least-violating one is returned either way.

Violation is weighted-linearly throughout — `sum of w_i * (shortfall_i + excess_i)` with `w_i` the
user-set constraint weights — so the certified bound and the returned selection's violation are
statements about that aggregate.  Verdicts and witnesses are penalty-shape-independent (zero
violation is zero violation under any shape).  The relaxation solver itself also handles
quadratic penalties; this pipeline fixes the linear aggregate its consumers report.

All randomization lives in the rounding stage and is driven by the caller's seed: the relaxation
solve is deterministic, so equal seeds give equal results and different seeds give genuinely
different near-feasible selections.
"""

import numpy as np
from numpy.typing import NDArray

from max_div._core._random import new_rng_state

from .evaluation import G_POSITIVE_TOL, _per_constraint_violation, _selection_counts, certified_bound, clamp_admissible
from .indexing import build_item_constraint_csr
from .ipm import solve_relaxation
from .repair import SWAP_CAP_MIN, SWAP_CAP_PER_K
from .result import FeasibilityResult, FeasibilityStatus
from .rounding import deterministic_round, sample_and_repair

# ==================================================================================================
#  Constants
# ==================================================================================================
N_DRAWS = 16  # randomized rounding draws per pipeline run
N_DRAWS_THOROUGH = 64  # draws when the caller asks to keep improving the returned selection


# ==================================================================================================
#  find_feasible
# ==================================================================================================
def find_feasible(
    con_values: NDArray[np.int32],
    con_indices: NDArray[np.int32],
    con_weights: NDArray[np.float64],
    n: int,
    k: int,
    seed: int = 0,
    thorough: bool = False,
) -> FeasibilityResult:
    """Run the full feasibility pipeline: exact relaxation solve, certificate, rounding, repair.

    Verification of an INFEASIBLE verdict needs no trust in this code: re-evaluate the closed-form
    dual value at the returned multipliers (`evaluation.certified_bound`) — elementary arithmetic
    on the constraint arrays — and any positive value is a valid lower bound on every selection's
    weighted violation.

    Args:
        con_values: 2D array (m, 2) with min_count and max_count for each constraint.
        con_indices: packed constraint->item membership array (`ConstraintList.to_numpy`).
        con_weights: per-constraint violation weights, in constraint order.
        n: the number of items.
        k: the selection size.
        seed: drives the randomized rounding draws; the relaxation solve is deterministic.
        thorough: draw more rounding rounds, which can improve the selection returned (its
            violation, never the verdict's soundness); feasible outcomes short-circuit anyway
            once a witness appears.
    """
    m = con_values.shape[0]
    con_min = con_values[:, 0].astype(np.int64)
    con_max = con_values[:, 1].astype(np.int64)
    w_lin = con_weights.astype(np.float64)
    w_quad = np.zeros(m, dtype=np.float64)
    item_indptr, item_cons = build_item_constraint_csr(con_indices, n)
    max_swaps = max(SWAP_CAP_MIN, SWAP_CAP_PER_K * k)

    # --- relaxation solve + certificate ---------
    solution = solve_relaxation(con_min, con_max, w_lin, w_quad, con_indices, n=n, k=k)
    lam_min = clamp_admissible(solution.lam_min, w_lin, w_quad)
    lam_max = clamp_admissible(solution.lam_max, w_lin, w_quad)
    bound = float(certified_bound(con_min, con_max, w_lin, w_quad, lam_min, lam_max, item_indptr, item_cons, k))
    certified_infeasible = bound > G_POSITIVE_TOL

    # --- witness attempts -----------------------
    # The seeded draws come FIRST and a witness among them returns immediately: on easily feasible
    # problems the first draw usually is one, and the draw -- not a deterministic
    # construction -- makes the returned selection vary with the seed.
    # mask to a non-negative int64: caller seeds may be any int, and the uint64 conversion in
    # new_rng_state rejects negatives outside compiled code
    rng_state = new_rng_state(np.int64(seed & 0x7FFFFFFFFFFFFFFF))
    best_selection = np.empty(0, dtype=np.int64)
    best_violation = np.inf
    n_draws = N_DRAWS_THOROUGH if thorough else N_DRAWS
    for _ in range(n_draws):
        selection, violation = sample_and_repair(
            solution.marginals, k, rng_state, con_indices, item_indptr, item_cons, con_min, con_max, w_lin, max_swaps
        )
        if violation < best_violation:
            best_selection, best_violation = selection, violation
        if best_violation == 0.0:
            break  # witness found; feasibility is proven and further draws cannot help
    if best_violation > 0.0:
        selection, violation = deterministic_round(
            solution.marginals, k, con_indices, item_indptr, item_cons, con_min, con_max, w_lin, max_swaps
        )
        if violation < best_violation:
            best_selection, best_violation = selection, violation

    # --- verdict --------------------------------
    if best_violation == 0.0:
        status = FeasibilityStatus.FEASIBLE
    elif certified_infeasible:
        status = FeasibilityStatus.INFEASIBLE
    else:
        status = FeasibilityStatus.UNKNOWN
    counts = _selection_counts(item_indptr, item_cons, best_selection, m)
    return FeasibilityResult(
        status=status,
        selection=best_selection,
        violation=best_violation,
        violation_per_constraint=_per_constraint_violation(counts, con_min, con_max),
        bound=bound,
        lam_min=lam_min,
        lam_max=lam_max,
    )
