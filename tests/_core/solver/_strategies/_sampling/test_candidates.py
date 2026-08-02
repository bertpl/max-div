import numpy as np
import pytest

from max_div._core._random import new_rng_state
from max_div._core.metrics import DistanceMetric, DiversityMetric
from max_div._core.metrics._distance import DistanceStore, compute_pdist
from max_div._core.solver._solver_state import SolverState
from max_div._core.solver._strategies._sampling import candidate_samples_to_add
from max_div._core.solver._strategies._sampling.candidates import CAP_GROWTH_PER_ITER, CAP_INITIAL


# =================================================================================================
#  Fixtures / helpers
# =================================================================================================
def _make_state(n: int) -> SolverState:
    """Build an unconstrained solver state with `n` items on a line and an empty selection."""
    vectors = np.arange(n, dtype=np.float32).reshape(-1, 1)
    return SolverState.new(
        n=n,
        store=DistanceStore.condensed(compute_pdist(vectors, DistanceMetric.L1_MANHATTAN), n=n),
        k=3,
        diversity_metric=DiversityMetric.GEOMEAN_SEPARATION,
        diversity_tie_breakers=[],
        constraints=[],
    )


# =================================================================================================
#  Tests
# =================================================================================================
@pytest.mark.parametrize("n", [10, CAP_INITIAL, CAP_INITIAL + 3])  # pool of n non-selected items, k=0 selected
def test_pool_at_or_below_cap_passes_through_without_rng(n: int):
    # --- arrange -----------------------------------------
    state = _make_state(n)
    rng_state = new_rng_state(42)
    rng_before = rng_state.copy()

    # --- act ---------------------------------------------
    pool = candidate_samples_to_add(state, iteration=max(0, n - CAP_INITIAL), rng_state=rng_state)

    # --- assert ------------------------------------------
    # the full pool comes back and the RNG is untouched — capped and uncapped runs are
    # bit-identical whenever the pool fits
    np.testing.assert_array_equal(pool, state.not_selected_index_array)
    np.testing.assert_array_equal(rng_state, rng_before)


def test_pool_above_cap_is_subsampled():
    # --- arrange -----------------------------------------
    n = 8 * CAP_INITIAL
    state = _make_state(n)
    rng_state = new_rng_state(42)

    # --- act ---------------------------------------------
    pool = candidate_samples_to_add(state, iteration=0, rng_state=rng_state)

    # --- assert ------------------------------------------
    assert pool.shape == (CAP_INITIAL,)
    assert len(set(pool)) == CAP_INITIAL  # unique
    assert np.all((pool >= 0) & (pool < n))  # all from the non-selected range


def test_cap_grows_with_iteration_and_saturates():
    # --- arrange -----------------------------------------
    n = CAP_INITIAL + 100
    state = _make_state(n)

    # --- act ---------------------------------------------
    capped = candidate_samples_to_add(state, iteration=0, rng_state=new_rng_state(42))
    grown = candidate_samples_to_add(state, iteration=50, rng_state=new_rng_state(42))
    saturated = candidate_samples_to_add(state, iteration=100 // CAP_GROWTH_PER_ITER, rng_state=new_rng_state(42))

    # --- assert ------------------------------------------
    assert capped.shape == (CAP_INITIAL,)
    assert grown.shape == (CAP_INITIAL + 50 * CAP_GROWTH_PER_ITER,)
    np.testing.assert_array_equal(saturated, state.not_selected_index_array)  # cap reached the pool


def test_subsample_is_deterministic_per_seed():
    # --- arrange -----------------------------------------
    n = 8 * CAP_INITIAL
    state = _make_state(n)

    # --- act ---------------------------------------------
    pool_1 = candidate_samples_to_add(state, iteration=0, rng_state=new_rng_state(42))
    pool_2 = candidate_samples_to_add(state, iteration=0, rng_state=new_rng_state(42))
    pool_3 = candidate_samples_to_add(state, iteration=0, rng_state=new_rng_state(123))

    # --- assert ------------------------------------------
    np.testing.assert_array_equal(pool_1, pool_2)
    assert list(pool_1) != list(pool_3)
