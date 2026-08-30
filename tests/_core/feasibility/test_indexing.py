import numpy as np

from max_div._core.constraints import Constraint, ConstraintList
from max_div._core.feasibility.indexing import build_item_constraint_csr


def _arrays(cons: list[Constraint]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert constraints to (con_values, con_indices, weights) as the pipeline ingests them."""
    con_values, con_indices = ConstraintList(cons).to_numpy()
    weights = np.array([con.weight for con in cons], dtype=np.float64)
    return con_values, con_indices, weights


def test_build_item_constraint_csr():
    """The transpose lists each item's constraints exactly."""
    # --- arrange ----------------------
    cons = [
        Constraint(int_set={0, 1, 2}, min_count=1, max_count=3),
        Constraint(int_set={2, 3}, min_count=0, max_count=2),
    ]
    _, con_indices, _ = _arrays(cons)

    # --- act --------------------------
    item_indptr, item_cons = build_item_constraint_csr(con_indices, 5)

    # --- assert -----------------------
    memberships = {j: sorted(item_cons[item_indptr[j] : item_indptr[j + 1]]) for j in range(5)}
    assert memberships == {0: [0], 1: [0], 2: [0, 1], 3: [1], 4: []}


def test_build_item_constraint_csr_no_constraints():
    """An empty constraint set transposes to an all-empty CSR."""
    # --- act --------------------------
    item_indptr, item_cons = build_item_constraint_csr(np.empty(0, dtype=np.int32), 3)

    # --- assert -----------------------
    assert item_indptr.tolist() == [0, 0, 0, 0]
    assert item_cons.size == 0
