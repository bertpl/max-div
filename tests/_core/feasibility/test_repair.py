import numpy as np

from max_div._core.constraints import Constraint, ConstraintList
from max_div._core.feasibility.evaluation import _selection_counts, _weighted_violation
from max_div._core.feasibility.indexing import build_item_constraint_csr
from max_div._core.feasibility.repair import _repair_selection


def _setup(cons: list[Constraint], n: int, selection: list[int]):
    """Build the packed arrays, mask, and counts the repair loop operates on."""
    _, con_indices = ConstraintList(cons).to_numpy()
    item_indptr, item_cons = build_item_constraint_csr(con_indices, n)
    con_min = np.array([c.min_count for c in cons], dtype=np.int64)
    con_max = np.array([c.max_count for c in cons], dtype=np.int64)
    weights = np.array([c.weight for c in cons], dtype=np.float64)
    sel_mask = np.zeros(n, dtype=np.bool_)
    sel_mask[selection] = True
    counts = _selection_counts(item_indptr, item_cons, np.array(selection, dtype=np.int64), len(cons))
    return con_indices, item_indptr, item_cons, con_min, con_max, weights, sel_mask, counts


def test_repair_fixes_a_near_feasible_selection():
    """One swap suffices: the repaired selection satisfies every constraint."""
    # --- arrange ----------------------
    cons = [
        Constraint(int_set={0, 1}, min_count=1, max_count=2),
        Constraint(int_set={2, 3}, min_count=1, max_count=2),
    ]
    con_indices, item_indptr, item_cons, con_min, con_max, weights, sel_mask, counts = _setup(
        cons, 4, [0, 1]
    )  # both picks in the first set: the second set is short by one

    # --- act --------------------------
    violation = _repair_selection(
        con_indices, item_indptr, item_cons, con_min, con_max, weights, sel_mask, counts, max_swaps=10
    )

    # --- assert -----------------------
    assert violation == 0.0
    assert sel_mask.sum() == 2  # size preserved
    assert _weighted_violation(_selection_counts(item_indptr, item_cons, np.where(sel_mask)[0], 2), con_min, con_max, weights) == 0.0


def test_repair_reports_residual_violation_at_the_swap_cap():
    """With a zero swap budget the selection stays as given and its violation is returned."""
    # --- arrange ----------------------
    cons = [
        Constraint(int_set={0, 1}, min_count=1, max_count=2),
        Constraint(int_set={2, 3}, min_count=1, max_count=2),
    ]
    con_indices, item_indptr, item_cons, con_min, con_max, weights, sel_mask, counts = _setup(cons, 4, [0, 1])

    # --- act --------------------------
    violation = _repair_selection(
        con_indices, item_indptr, item_cons, con_min, con_max, weights, sel_mask, counts, max_swaps=0
    )

    # --- assert -----------------------
    assert violation == 1.0  # the short second set, untouched
    assert sel_mask.tolist() == [True, True, False, False]
