import numpy as np
import pytest

from max_div._core.constraints import Constraint, ConstraintList
from max_div._core.feasibility.evaluation import (
    _dual_value,
    _item_scores,
    _top_k_items,
    certified_bound,
    clamp_admissible,
)
from max_div._core.feasibility.indexing import build_item_constraint_csr


def _arrays(cons: list[Constraint]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert constraints to (con_values, con_indices, weights) as the pipeline ingests them."""
    con_values, con_indices = ConstraintList(cons).to_numpy()
    weights = np.array([con.weight for con in cons], dtype=np.float64)
    return con_values, con_indices, weights


def _pigeonhole_instance() -> tuple[int, int, list[Constraint]]:
    """Return a certifiably infeasible instance: two disjoint min-2 sets with k = 2 (minimum violation 2)."""
    cons = [
        Constraint(int_set={0, 1}, min_count=2, max_count=2),
        Constraint(int_set={2, 3}, min_count=2, max_count=2),
    ]
    return 4, 2, cons


def test_top_k_items_is_exact():
    """Heap selection returns exactly the k best items, and all items when k reaches n."""
    # --- arrange ----------------------
    scores = np.array([3.0, 1.0, 2.0, 5.0, 4.0])

    # --- act --------------------------
    top2 = _top_k_items(scores, 2)
    top_all = _top_k_items(scores, 5)

    # --- assert -----------------------
    assert sorted(top2.tolist()) == [3, 4]
    assert sorted(top_all.tolist()) == [0, 1, 2, 3, 4]


# =================================================================================================
#  Dual value and the certificate
# =================================================================================================
def test_dual_value_pigeonhole_toy():
    """All-ones prices on the pigeonhole instance give g = 2."""
    # --- arrange ----------------------
    n, k, cons = _pigeonhole_instance()
    _, con_indices, _ = _arrays(cons)
    item_indptr, item_cons = build_item_constraint_csr(con_indices, n)
    lam_min = np.ones(2)
    lam_max = np.zeros(2)

    # --- act --------------------------
    scores = _item_scores(item_indptr, item_cons, lam_min, lam_max)
    selection = _top_k_items(scores, k)
    g = _dual_value(
        np.array([2, 2], dtype=np.int64), np.array([2, 2], dtype=np.int64), lam_min, lam_max, scores, selection
    )

    # --- assert -----------------------
    assert scores.tolist() == [1.0, 1.0, 1.0, 1.0]
    assert g == pytest.approx(2.0)


def test_exact_topk_guard():
    """A non-maximizing inner selection fabricates a positive g on a feasible instance.

    Pins the soundness invariant: the ascent's dual value is a proof only because it uses an exact
    top-k — this test demonstrates the false proof a perturbed selection would produce, and that
    the exact selection stays non-positive (as any feasible instance requires).
    """
    # --- arrange ----------------------
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

    # --- act --------------------------
    g_exact = _dual_value(con_min, con_max, lam_min, lam_max, scores, _top_k_items(scores, k))
    g_corrupted = _dual_value(con_min, con_max, lam_min, lam_max, scores, np.array([3, 4], dtype=np.int64))

    # --- assert -----------------------
    assert g_exact <= 0.0
    assert g_corrupted > 0.0


# =================================================================================================
#  clamp_admissible / certified_bound
# =================================================================================================
def test_clamp_repairs_negative_and_overpriced_multipliers():
    """Negatives clamp to zero; purely linear constraints cap at their weight; quadratic ones do not."""
    # --- arrange ----------------------
    lam = np.array([-0.5, 2.0, 2.0])
    w_lin = np.array([1.0, 1.0, 1.0])
    w_quad = np.array([0.0, 0.0, 1.0])

    # --- act --------------------------
    out = clamp_admissible(lam, w_lin, w_quad)

    # --- assert -----------------------
    assert out == pytest.approx([0.0, 1.0, 2.0])
    assert lam[0] == -0.5  # input untouched


def test_certified_bound_reduces_to_linear_dual_value():
    """With zero quadratic weights the general bound has no price costs to subtract."""
    # --- arrange ----------------------
    cons = [
        Constraint(int_set={0, 1}, min_count=2, max_count=2),
        Constraint(int_set={2, 3}, min_count=2, max_count=2),
    ]
    con_values, con_indices, _ = _arrays(cons)
    con_min = con_values[:, 0].astype(np.int64)
    con_max = con_values[:, 1].astype(np.int64)
    item_indptr, item_cons = build_item_constraint_csr(con_indices, 4)
    w_lin, w_quad = np.ones(2), np.zeros(2)
    lam_min, lam_max = np.array([1.0, 1.0]), np.array([0.0, 0.0])

    # --- act / assert -----------------
    g = certified_bound(con_min, con_max, w_lin, w_quad, lam_min, lam_max, item_indptr, item_cons, 2)
    assert g == pytest.approx(2.0)  # lam.con_min - top-2 of scores = 4 - 2
