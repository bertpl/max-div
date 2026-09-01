import numpy as np
import pytest

from max_div._core._random import new_rng_state
from max_div._core.metrics import DistanceMetric
from max_div._core.metrics._distance import DistanceStore
from max_div._core.solver._solver_step import InitializationStep
from max_div._core.solver._strategies import InitializationStrategy
from max_div._core.solver._strategies._initialization._init_farthest_point_batched import (
    InitFarthestPointBatched,
    _draw_round,
)
from max_div.metrics import DiversityMetric

from ._helpers import new_solver_state


@pytest.mark.parametrize("top_k", [1, 8])
def test_init_farthest_point_batched_completes_selection(top_k: int):
    """The batched init selects exactly k distinct items and reaches full size."""
    # --- arrange ----------------------
    state = new_solver_state(has_constraints=False)
    step = InitializationStep(InitializationStrategy.farthest_point_batched(top_k=top_k))

    # --- act --------------------------
    step.run(state)

    # --- assert -----------------------
    assert state.score.size == 1.0
    selection = state.selected_index_array
    assert len(np.unique(selection)) == state.k


def test_init_farthest_point_batched_is_deterministic_per_seed():
    """The same seed reproduces the same selection; a different seed varies it."""
    # --- arrange ----------------------
    selections = []
    for seed in (7, 7, 8):
        state = new_solver_state(has_constraints=False)
        step = InitializationStep(InitializationStrategy.farthest_point_batched())
        step.set_seed(seed)

        # --- act ----------------------
        step.run(state)
        selections.append(np.sort(state.selected_index_array).copy())

    # --- assert -----------------------
    np.testing.assert_array_equal(selections[0], selections[1])
    assert not np.array_equal(selections[0], selections[2])


def test_init_farthest_point_batched_quality_near_exact_sibling():
    """The batched construction's min-separation is near `InitFarthestPoint`'s."""
    # --- arrange ----------------------
    results = {}
    for label, strategy in (
        ("exact", InitializationStrategy.farthest_point(top_k=8)),
        ("batched", InitializationStrategy.farthest_point_batched()),
    ):
        state = new_solver_state(has_constraints=False)
        step = InitializationStep(strategy)
        step.set_seed(42)

        # --- act ----------------------
        step.run(state)
        results[label] = state.score.diversity

    # --- assert -----------------------
    assert results["batched"] >= 0.8 * results["exact"]


def test_init_farthest_point_batched_rejects_mean_family_metric():
    """A mean-family main metric is refused: the round heuristics are separation-tailored."""
    # --- arrange ----------------------
    strategy = InitializationStrategy.farthest_point_batched()

    # --- act / assert -----------------
    with pytest.raises(ValueError, match="separation-based diversity metrics"):
        strategy.validate_diversity_metric(DiversityMetric.MEAN_PAIRWISE_DISTANCE)
    strategy.validate_diversity_metric(DiversityMetric.MIN_SEPARATION)  # accepted: no raise


@pytest.mark.parametrize(
    "kwargs",
    [{"top_k": 0}, {"batch_size": 0}, {"top_k": 8, "batch_size": 4}],
)
def test_init_farthest_point_batched_rejects_invalid_parameters(kwargs: dict):
    """Constructor bounds: top_k >= 1, and a pool at least wide enough for one full draw."""
    with pytest.raises(ValueError):
        InitFarthestPointBatched(**kwargs)


def test_init_farthest_point_batched_batches_respect_the_contract():
    """Every returned batch is duplicate-free, in range, and not yet selected."""
    # --- arrange ----------------------
    state = new_solver_state(has_constraints=False)
    strategy = InitializationStrategy.farthest_point_batched(batch_size=16)

    # --- act / assert -----------------
    while state.n_selected < state.k:
        batch = strategy.get_next_samples(state, np.int32(state.k - state.n_selected))
        assert 1 <= len(batch) <= state.k - state.n_selected
        assert len(np.unique(batch)) == len(batch)
        assert batch.min() >= 0
        assert batch.max() < state.n
        assert not np.isin(batch, state.selected_index_array).any()
        state.add_many(batch)


def test_draw_round_ends_once_the_pool_can_no_longer_be_shown_to_hold_the_best():
    """A refresh that pushes the live pool below its admission value ends the round instead of drawing on."""
    # --- arrange ----------------------
    # items 1, 2, 3 sit far from item 0 but close to each other, so drawing one collapses the
    # separations of the others far below the pool's admission value
    vectors = np.array([[0.0], [100.0], [101.0], [102.0]], dtype=np.float32)
    store = DistanceStore.lazy(vectors, DistanceMetric.l2_euclidean())
    cand_idx = np.array([1, 2, 3], dtype=np.int32)
    cand_val = np.array([100.0, 101.0, 102.0], dtype=np.float32)  # separations wrt a selection of {0}
    threshold = np.float32(100.0)  # the pool's lowest value at round start
    out_batch = np.empty(3, dtype=np.int32)

    # --- act --------------------------
    n_drawn = _draw_round(
        cand_idx, cand_val, np.int32(1), threshold, np.int64(3), store, new_rng_state(np.int64(1)), out_batch
    )

    # --- assert -----------------------
    assert n_drawn == 1
    assert out_batch[0] == 3  # the farthest item; its neighbors then fall below the admission value


def test_draw_round_draws_while_the_pool_still_holds_the_best():
    """With every candidate far apart, no refresh drops below the admission value and the round runs on."""
    # --- arrange ----------------------
    vectors = np.array([[0.0], [100.0], [200.0], [300.0]], dtype=np.float32)
    store = DistanceStore.lazy(vectors, DistanceMetric.l2_euclidean())
    cand_idx = np.array([1, 2, 3], dtype=np.int32)
    cand_val = np.array([100.0, 200.0, 300.0], dtype=np.float32)
    out_batch = np.empty(3, dtype=np.int32)

    # --- act --------------------------
    n_drawn = _draw_round(
        cand_idx,
        cand_val,
        np.int32(1),
        np.float32(100.0),
        np.int64(3),
        store,
        new_rng_state(np.int64(1)),
        out_batch,
    )

    # --- assert -----------------------
    assert n_drawn >= 2
