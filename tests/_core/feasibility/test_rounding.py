import numpy as np
import pytest

from max_div._core._random import new_rng_state
from max_div._core.constraints import Constraint, ConstraintList
from max_div._core.feasibility.indexing import build_item_constraint_csr
from max_div._core.feasibility.rounding import (
    deterministic_round,
    sample_and_repair,
    selection_violation,
    systematic_sample,
)


def _repair_arrays(cons: list[Constraint], n: int):
    """Build the packed arrays the repairing rounders operate on."""
    _, con_indices = ConstraintList(cons).to_numpy()
    item_indptr, item_cons = build_item_constraint_csr(con_indices, n)
    con_min = np.array([c.min_count for c in cons], dtype=np.int64)
    con_max = np.array([c.max_count for c in cons], dtype=np.int64)
    weights = np.array([c.weight for c in cons], dtype=np.float64)
    return con_indices, item_indptr, item_cons, con_min, con_max, weights


# =================================================================================================
#  systematic_sample
# =================================================================================================
@pytest.mark.parametrize("seed", range(5))
def test_systematic_sample_returns_exactly_k_unique_items(seed: int):
    """Every draw selects exactly k distinct items."""
    # --- arrange ----------------------
    rng = np.random.default_rng(seed)
    n, k = 30, 7
    marginals = rng.random(n)
    marginals *= k / marginals.sum()
    marginals = np.clip(marginals, 0.0, 1.0)
    marginals *= k / marginals.sum()

    # --- act --------------------------
    selection = systematic_sample(marginals, k, new_rng_state(seed))

    # --- assert -----------------------
    assert selection.shape[0] == k
    assert np.unique(selection).shape[0] == k


def test_systematic_sample_inclusion_frequencies_match_the_marginals():
    """Across many draws each item is selected about as often as its marginal says."""
    # --- arrange ----------------------
    marginals = np.array([1.0, 0.75, 0.5, 0.5, 0.25, 0.0])
    k, n_draws = 3, 4000
    rng_state = new_rng_state(42)

    # --- act --------------------------
    hits = np.zeros(6)
    for _ in range(n_draws):
        hits[systematic_sample(marginals, k, rng_state)] += 1
    frequencies = hits / n_draws

    # --- assert -----------------------
    assert frequencies == pytest.approx(marginals, abs=0.03)


def test_systematic_sample_zero_marginal_is_never_drawn():
    """An item with marginal 0 cannot be selected."""
    # --- arrange ----------------------
    marginals = np.array([1.0, 1.0, 0.5, 0.5, 0.0])
    rng_state = new_rng_state(1)

    # --- act / assert -----------------
    for _ in range(50):
        assert 4 not in systematic_sample(marginals, 3, rng_state)


def test_systematic_sample_tops_up_when_marginals_under_sum():
    """Marginals summing below k still yield exactly k items, topped up by largest marginal (the drift guard)."""
    # --- arrange ----------------------
    marginals = np.array([0.4, 0.4, 0.4, 0.4, 0.1])  # sums to 1.7 < k = 2

    # --- act / assert -----------------
    rng_state = new_rng_state(3)
    for _ in range(20):
        selection = systematic_sample(marginals, 2, rng_state)
        assert np.unique(selection).shape[0] == 2


# =================================================================================================
#  sample_and_repair / deterministic_round
# =================================================================================================
def test_sample_and_repair_reaches_feasibility_from_valid_marginals():
    """Rounding relaxed-feasible marginals plus repair lands a feasible selection."""
    # --- arrange ----------------------
    cons = [
        Constraint(int_set={0, 1, 2, 3}, min_count=1, max_count=2),
        Constraint(int_set={4, 5, 6, 7}, min_count=1, max_count=2),
    ]
    arrays = _repair_arrays(cons, 8)
    marginals = np.full(8, 0.5)  # satisfies both constraints fractionally, sums to k=4

    # --- act --------------------------
    selection, violation = sample_and_repair(marginals, 4, new_rng_state(5), *arrays, max_swaps=100)

    # --- assert -----------------------
    assert violation == 0.0
    assert selection_violation(selection, *arrays[1:]) == 0.0


def test_deterministic_round_takes_the_largest_marginals():
    """With unambiguous marginals the deterministic round is the top-k, repaired only if needed."""
    # --- arrange ----------------------
    cons = [Constraint(int_set={0, 1, 2, 3, 4, 5}, min_count=1, max_count=3)]
    arrays = _repair_arrays(cons, 6)
    marginals = np.array([0.9, 0.8, 0.7, 0.1, 0.1, 0.1])

    # --- act --------------------------
    selection, violation = deterministic_round(marginals, 3, *arrays, max_swaps=10)

    # --- assert -----------------------
    assert sorted(selection.tolist()) == [0, 1, 2]
    assert violation == 0.0


def test_systematic_sample_degrades_to_a_valid_selection_on_nan_marginals():
    """All-NaN marginals (a bug upstream) still yield k distinct in-range items, never -1."""
    # --- arrange ----------------------
    marginals = np.full(10, np.nan, dtype=np.float64)
    rng_state = new_rng_state(np.int64(0))

    # --- act --------------------------
    selection = systematic_sample(marginals, 4, rng_state)

    # --- assert -----------------------
    assert selection.shape[0] == 4
    assert len(set(selection.tolist())) == 4
    assert selection.min() >= 0
    assert selection.max() < 10
